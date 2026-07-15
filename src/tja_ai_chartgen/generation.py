import json
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from tja_ai_chartgen.audio.analyze import (
    AudioAnalysisRaw,
    analyze_audio,
    apply_analysis_overrides,
    enrich_tempo_candidates_with_instruments,
)
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.audio.instrument_models import resolve_instrument_model_dir
from tja_ai_chartgen.audio.instruments import AST_WINDOW_SECONDS, analyze_instruments
from tja_ai_chartgen.cancellation import GenerationCancelledError, raise_if_cancelled
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.meter import get_meter_spec, validate_time_signature
from tja_ai_chartgen.features.resolution import build_resolution_plan, output_resolution_for_bar
from tja_ai_chartgen.features.structure import STRUCTURE_FEATURE_VERSION, analyze_song_structure
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, ResolutionPlan, SongAnalysis
from tja_ai_chartgen.tja.quality import QualityReport, build_quality_report
from tja_ai_chartgen.utils.paths import write_json

DEFAULT_AI_REQUEST_TIMEOUT = 300.0
DEFAULT_AI_TRANSPORT_RETRIES = 3
MAX_AI_TRANSPORT_RETRIES = 5
GenerationStageCallback = Callable[[str], None]


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    input_audio: Path
    title: str
    artist: str | None = None
    output_dir: Path = Path("output")
    course: str = "Oni"
    level: int = 10
    all_courses: bool = False
    style: str = "technical"
    density: str = "auto"
    max_bars: int | None = Field(default=None, ge=1)
    bpm_override: float | None = Field(default=None, gt=0)
    offset_override: float | None = None
    time_signature: str | None = None
    use_beatnet: bool = False
    use_instrument_analysis: bool = False
    instrument_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    instrument_model_dir: Path | None = None
    special_notes: bool = False
    use_ai: bool = False
    model: str | None = None
    ai_repair_retries: int = Field(default=2, ge=0)
    ai_request_timeout: float = Field(default=DEFAULT_AI_REQUEST_TIMEOUT, ge=1, le=600)
    ai_transport_retries: int = Field(
        default=DEFAULT_AI_TRANSPORT_RETRIES,
        ge=0,
        le=MAX_AI_TRANSPORT_RETRIES,
    )

    @field_serializer("input_audio", "output_dir", "instrument_model_dir")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None

    @model_validator(mode="after")
    def validate_meter(self) -> "GenerationConfig":
        if self.time_signature is not None:
            validate_time_signature(self.time_signature)
        return self


class GenerationNotice(BaseModel):
    code: str
    level: Literal["info", "warning", "error"]
    stage: Literal["analysis", "generation", "render"]
    message: str
    detail: str | None = None
    scope: str | None = None


class ChartGenerationResult(BaseModel):
    chart_bars: list[ChartBar]
    quality_report: QualityReport
    ai_failure: str | None = None
    used_fallback: bool
    notices: list[GenerationNotice] = Field(default_factory=list)


def load_generation_config(path: Path) -> GenerationConfig:
    if not path.exists():
        raise FileNotFoundError(f"Generation config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid generation config JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("Generation config must be a JSON object.")
    schema_version = payload.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"Unsupported generation config schema_version: {schema_version}")
    for field in ("input_audio", "title"):
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if "schema_version" not in payload:
        payload.pop("ai_base_url", None)
        payload.pop("ai_api_key", None)
    payload.setdefault("schema_version", 1)
    return GenerationConfig.model_validate(payload)


def build_song_analysis(
    *,
    input_audio: Path,
    ogg_path: Path,
    title: str,
    artist: str | None = None,
    max_bars: int | None = None,
    bpm_override: float | None = None,
    offset_override: float | None = None,
    time_signature_override: str | None = None,
    use_beatnet: bool = False,
    use_instrument_analysis: bool = False,
    instrument_device: str = "auto",
    instrument_model_dir: Path | None = None,
    stage_callback: GenerationStageCallback | None = None,
) -> SongAnalysis:
    _report_stage(stage_callback, "convert")
    convert_to_ogg(input_audio, ogg_path)

    _report_stage(stage_callback, "analyze")
    raw = analyze_audio(ogg_path, use_beatnet=use_beatnet)
    raw = apply_analysis_overrides(raw, bpm=bpm_override, offset=offset_override)
    if time_signature_override is not None:
        validate_time_signature(time_signature_override)
        raw = raw.model_copy(update={"time_signature": time_signature_override})

    if use_instrument_analysis:
        _report_stage(stage_callback, "instruments")
        model_dir = resolve_instrument_model_dir(instrument_model_dir)
        instrument_result = analyze_instruments(
            ogg_path,
            model_dir=model_dir,
            device=instrument_device,
            analysis_sample_rate=raw.sample_rate or 22_050,
            analysis_hop_length=raw.hop_length,
            max_duration=_instrument_analysis_duration(raw, max_bars),
        )
        raw = raw.model_copy(
            update={
                "instruments": instrument_result,
                "tempo_candidates": enrich_tempo_candidates_with_instruments(
                    raw.tempo_candidates,
                    instrument_result,
                ),
            }
        )

    _report_stage(stage_callback, "features")
    structure = analyze_song_structure(build_bar_features(raw, max_bars=max_bars))
    bars = structure.bars
    resolution_plan = build_resolution_plan(raw, bars)
    phrase_plan = [
        phrase.model_copy(
            update={
                "resolution": resolution_plan.bar_resolutions[phrase.start_bar]
                if phrase.start_bar < len(resolution_plan.bar_resolutions)
                else resolution_plan.base_resolution
            }
        )
        for phrase in structure.phrases
    ]
    return SongAnalysis(
        analysis_schema_version=9,
        beatnet_analysis_status=raw.beatnet_analysis_status,
        beatnet_analysis_reason=raw.beatnet_analysis_reason,
        spectral_feature_version=raw.spectral.feature_version,
        spectral_analysis_status=raw.spectral.status,
        spectral_analysis_reason=raw.spectral.reason,
        instrument_feature_version=raw.instruments.feature_version,
        instrument_analysis_status=raw.instruments.status,
        instrument_analysis_reason=raw.instruments.reason,
        instrument_demucs_model=raw.instruments.demucs_model,
        instrument_classifier_model=raw.instruments.classifier_model,
        instrument_analysis_device=raw.instruments.device,
        structure_feature_version=STRUCTURE_FEATURE_VERSION,
        structure_confidence=structure.confidence,
        bar_structures=structure.bar_structures,
        phrase_plan=phrase_plan,
        resolution_policy_version=resolution_plan.policy_version,
        resolution_plan=resolution_plan,
        title=title,
        artist=artist,
        audio_file=str(input_audio),
        ogg_file=str(ogg_path),
        bpm=raw.bpm,
        offset=raw.offset,
        time_signature=raw.time_signature,
        analyzer=raw.analyzer,
        tempo_candidates=raw.tempo_candidates,
        tempo_analysis=raw.tempo_analysis,
        bars=bars,
    )


def _instrument_analysis_duration(raw: AudioAnalysisRaw, max_bars: int | None) -> float | None:
    if max_bars is None:
        return None
    meter = get_meter_spec(raw.time_signature)
    bar_length = meter.beats_per_bar * 60.0 / raw.bpm
    requested_end = max(0.0, raw.offset + max_bars * bar_length + AST_WINDOW_SECONDS)
    return min(raw.duration, requested_end)


def build_analysis_notices(
    analysis: SongAnalysis,
    *,
    requested_beatnet: bool = False,
    requested_instrument_analysis: bool = False,
) -> list[GenerationNotice]:
    notices: list[GenerationNotice] = []
    analyzer = analysis.analyzer.lower()
    tempo = analysis.tempo_analysis
    if requested_beatnet and "beatnet" not in analyzer:
        beatnet_rejections = (
            {
                source: reason
                for source, reason in tempo.candidate_rejections.items()
                if source.startswith("beatnet")
            }
            if tempo is not None
            else {}
        )
        if analysis.beatnet_analysis_status == "complete" and beatnet_rejections:
            notices.append(
                GenerationNotice(
                    code="beatnet-candidate-rejected",
                    level="warning",
                    stage="analysis",
                    message="BeatNet 已完成分析，但候选未通过节拍仲裁，已保留基础结果。",
                    detail="; ".join(
                        f"{source}={reason}"
                        for source, reason in sorted(beatnet_rejections.items())
                    ),
                )
            )
        else:
            notices.append(
                GenerationNotice(
                    code="beatnet-fallback",
                    level="warning",
                    stage="analysis",
                    message="BeatNet 增强未生效，已保留 librosa 分析结果。",
                    detail=f"reason={analysis.beatnet_analysis_reason or 'unknown'}",
                )
            )

    if analysis.spectral_analysis_status == "fallback":
        notices.append(
            GenerationNotice(
                code="spectral-analysis-fallback",
                level="warning",
                stage="analysis",
                message="频谱语义增强未生效，已继续使用 onset、RMS 与节拍特征。",
                detail=f"reason={analysis.spectral_analysis_reason or 'unknown'}",
            )
        )

    if requested_instrument_analysis:
        status = analysis.instrument_analysis_status
        reason = analysis.instrument_analysis_reason or "unknown"
        if status == "complete":
            notices.append(
                GenerationNotice(
                    code="instrument-analysis-succeeded",
                    level="info",
                    stage="analysis",
                    message="人声与乐器语义分析已完成。",
                )
            )
        elif status == "partial":
            notices.append(
                GenerationNotice(
                    code="instrument-analysis-partial",
                    level="warning",
                    stage="analysis",
                    message="人声与乐器分析仅部分生效，已使用可用证据继续生成。",
                    detail=f"reason={reason}",
                )
            )
        else:
            code, message = _instrument_fallback_notice(reason)
            notices.append(
                GenerationNotice(
                    code=code,
                    level="warning",
                    stage="analysis",
                    message=message,
                    detail=f"reason={reason}",
                )
            )

    if tempo is not None and tempo.ambiguous and "manual-override" not in analyzer:
        notices.append(
            GenerationNotice(
                code="tempo-arbitration-ambiguous",
                level="warning",
                stage="analysis",
                message="节拍候选得分接近且结论冲突，已保守保留基础结果。",
                detail=(
                    f"selected={tempo.selected_source}; score={tempo.selected_score:.3f}; "
                    f"runner_up={tempo.runner_up_source or 'none'}; "
                    f"runner_up_score={tempo.runner_up_score or 0.0:.3f}"
                ),
            )
        )
    elif tempo is not None and not tempo.accepted and "manual-override" not in analyzer:
        source = "BeatNet" if tempo.fallback_source.startswith("beatnet") else "librosa"
        onset_refinement_reason = tempo.reason in {
            "insufficient_onsets",
            "insufficient_time_span",
            "insufficient_time_coverage",
            "low_normalized_support",
        }
        message = (
            f"onset-grid 节拍校正置信度不足，已保留 {source} 基线。"
            if onset_refinement_reason
            else f"节拍增强候选未通过仲裁，已保留 {source} 基线。"
        )
        notices.append(
            GenerationNotice(
                code="tempo-refinement-fallback",
                level="warning",
                stage="analysis",
                message=message,
                detail=(
                    f"reason={tempo.reason}; support={tempo.normalized_support:.3f}; "
                    f"score={tempo.selected_score:.3f}; onsets={tempo.onset_count}; "
                    f"coverage={tempo.time_coverage:.3f}"
                ),
            )
        )

    if (
        analysis.structure_confidence is not None
        and len(analysis.bars) >= 4
        and analysis.structure_confidence < 0.35
    ):
        notices.append(
            GenerationNotice(
                code="structure-low-confidence",
                level="warning",
                stage="analysis",
                message="乐句与段落边界置信度较低，建议在游玩预览中重点检查结构变化。",
                detail=f"structure_confidence={analysis.structure_confidence:.3f}",
            )
        )

    decision = analysis.resolution_plan.decision if analysis.resolution_plan is not None else None
    if decision is not None and decision.evidence_count == 0:
        notices.append(
            GenerationNotice(
                code="resolution-evidence-fallback",
                level="warning",
                stage="analysis",
                message="缺少可靠细分节奏证据，已使用稳定基础分辨率。",
                detail=decision.reason,
            )
        )
    return notices


def _instrument_fallback_notice(reason: str) -> tuple[str, str]:
    if reason.startswith("missing-dependency:"):
        return (
            "instrument-dependencies-unavailable",
            "人声与乐器分析依赖不可用，已继续使用基础音频特征。",
        )
    if reason.startswith(("missing-model:", "invalid-model-manifest")):
        return (
            "instrument-models-missing",
            "人声与乐器模型未准备完整，已继续使用基础音频特征。",
        )
    if reason.startswith("device-unavailable:"):
        return (
            "instrument-device-unavailable",
            "指定的人声与乐器分析设备不可用，已继续使用基础音频特征。",
        )
    return (
        "instrument-analysis-fallback",
        "人声与乐器分析未生效，已继续使用基础音频特征。",
    )


def generate_chart_bars(
    *,
    analysis: SongAnalysis,
    selected_bars: list[BarFeature],
    course: str,
    level: int,
    style: str,
    density: str,
    special_notes: bool,
    use_ai: bool,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    ai_repair_retries: int = 2,
    ai_request_timeout: float = DEFAULT_AI_REQUEST_TIMEOUT,
    ai_transport_retries: int = DEFAULT_AI_TRANSPORT_RETRIES,
    ai_input_path: Path | None = None,
    ai_output_path: Path | None = None,
    ai_attempts_path: Path | None = None,
    cancel_event: Event | None = None,
) -> ChartGenerationResult:
    chart_bars: list[ChartBar] | None = None
    ai_failure: str | None = None
    raise_if_cancelled(cancel_event)

    if use_ai:
        try:
            chart_bars = _generate_ai_bars(
                analysis=analysis,
                selected_bars=selected_bars,
                course=course,
                level=level,
                style=style,
                density=density,
                special_notes=special_notes,
                model=model,
                api_base=api_base,
                api_key=api_key,
                ai_repair_retries=ai_repair_retries,
                ai_request_timeout=ai_request_timeout,
                ai_transport_retries=ai_transport_retries,
                ai_input_path=ai_input_path,
                ai_output_path=ai_output_path,
                ai_attempts_path=ai_attempts_path,
                cancel_event=cancel_event,
            )
        except GenerationCancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - AI failure must produce a usable fallback.
            ai_failure = _redact_value(str(error), api_key)
            _write_ai_failure(ai_attempts_path, error, api_key)
            _write_ai_failure(ai_output_path, error, api_key)

    raise_if_cancelled(cancel_event)
    used_fallback = chart_bars is None
    if chart_bars is None:
        chart_bars = generate_fallback_chart_bars(
            selected_bars,
            style=style,
            density=density,
            special_notes=special_notes,
            course=course,
            level=level,
            resolution_plan=analysis.resolution_plan,
        )

    _validate_generated_resolutions(
        chart_bars,
        selected_bars,
        analysis.resolution_plan,
    )
    notices: list[GenerationNotice] = []
    if use_ai:
        if ai_failure:
            notices.append(
                GenerationNotice(
                    code="ai-fallback",
                    level="warning",
                    stage="generation",
                    scope=course,
                    message="AI 增强失败，已自动回退到规则生成。",
                    detail=ai_failure,
                )
            )
        else:
            notices.append(
                GenerationNotice(
                    code="ai-generation-succeeded",
                    level="info",
                    stage="generation",
                    scope=course,
                    message="AI 增强已完成，本次谱面来自 AI 输出校验后的结果。",
                )
            )

    return ChartGenerationResult(
        chart_bars=chart_bars,
        quality_report=build_quality_report(
            chart_bars,
            selected_bars,
            analysis.resolution_plan,
        ),
        ai_failure=ai_failure,
        used_fallback=used_fallback,
        notices=notices,
    )


def _validate_generated_resolutions(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    resolution_plan: ResolutionPlan | None,
) -> None:
    if len(chart_bars) != len(feature_bars):
        raise ValueError(
            f"Generated chart bar count {len(chart_bars)} does not match feature bar count "
            f"{len(feature_bars)}"
        )
    issues: list[str] = []
    for position, (chart_bar, feature_bar) in enumerate(
        zip(chart_bars, feature_bars, strict=True)
    ):
        expected = output_resolution_for_bar(
            feature_bar,
            plan=resolution_plan,
            position=position,
        )
        if len(chart_bar.notes) != expected:
            issues.append(
                f"bar {feature_bar.index + 1} expected resolution {expected}, "
                f"got {len(chart_bar.notes)}"
            )
    if issues:
        raise ValueError("Generated chart does not match ResolutionPlan: " + "; ".join(issues))


def _generate_ai_bars(
    *,
    analysis: SongAnalysis,
    selected_bars: list[BarFeature],
    course: str,
    level: int,
    style: str,
    density: str,
    special_notes: bool,
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    ai_repair_retries: int,
    ai_request_timeout: float,
    ai_transport_retries: int,
    ai_input_path: Path | None,
    ai_output_path: Path | None,
    ai_attempts_path: Path | None,
    cancel_event: Event | None,
) -> list[ChartBar]:
    from tja_ai_chartgen.ai.client import generate_chart_bars_with_ai, sanitize_ai_bars
    from tja_ai_chartgen.ai.prompts import build_chart_generation_payload

    selected_indexes = {bar.index for bar in selected_bars}
    selected_analysis = analysis.model_copy(
        update={
            "bars": selected_bars,
            "bar_structures": [
                structure
                for structure in analysis.bar_structures
                if structure.index in selected_indexes
            ],
            "phrase_plan": [
                phrase
                for phrase in analysis.phrase_plan
                if any(
                    phrase.start_bar <= index <= phrase.end_bar
                    for index in selected_indexes
                )
            ],
        }
    )
    if ai_input_path is not None:
        write_json(
            ai_input_path,
            build_chart_generation_payload(
                selected_analysis,
                course,
                level,
                style,
                density,
                special_notes=special_notes,
            ),
        )
    ai_call_kwargs: dict[str, Any] = {
        "api_base": api_base,
        "api_key": api_key,
        "max_repair_attempts": ai_repair_retries,
        "special_notes": special_notes,
        "attempt_log_path": ai_attempts_path,
        "request_timeout": ai_request_timeout,
        "max_transport_retries": ai_transport_retries,
    }
    if cancel_event is not None:
        ai_call_kwargs["cancel_event"] = cancel_event
    ai_bars, ai_output = generate_chart_bars_with_ai(
        selected_analysis,
        course,
        level,
        style,
        density,
        model,
        **ai_call_kwargs,
    )
    _redact_json_file(ai_attempts_path, api_key)
    if ai_output_path is not None:
        write_json(ai_output_path, _redact_value(ai_output, api_key))
    sanitized = sanitize_ai_bars(
        ai_bars,
        expected_count=len(selected_bars),
        expected_bars=selected_bars,
        resolution_plan=analysis.resolution_plan,
    )
    return _reindex_chart_bars(sanitized, selected_bars)


def _reindex_chart_bars(
    chart_bars: list[ChartBar], selected_bars: list[BarFeature]
) -> list[ChartBar]:
    return [
        ChartBar(
            index=feature.index,
            notes=chart_bar.notes,
            time_signature=feature.time_signature,
            balloon_counts=chart_bar.balloon_counts,
        )
        for chart_bar, feature in zip(chart_bars, selected_bars, strict=True)
    ]


def _write_ai_failure(path: Path | None, error: Exception, api_key: str | None) -> None:
    if path is None:
        return
    output = getattr(error, "output", None)
    payload: dict[str, Any] = {"error": _redact_value(str(error), api_key)}
    if isinstance(output, dict):
        payload.update(_redact_value(output, api_key))
    write_json(path, payload)


def _redact_value(value: Any, api_key: str | None) -> Any:
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED]") if api_key else value
    if isinstance(value, dict):
        return {_redact_value(key, api_key): _redact_value(item, api_key) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, api_key) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, api_key) for item in value)
    return value


def _redact_json_file(path: Path | None, api_key: str | None) -> None:
    if path is None or not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    write_json(path, _redact_value(payload, api_key))


def _report_stage(callback: GenerationStageCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)
