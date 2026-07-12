import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from tja_ai_chartgen.audio.analyze import analyze_audio, apply_analysis_overrides
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.meter import validate_time_signature
from tja_ai_chartgen.features.sections import assign_sections
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, SongAnalysis
from tja_ai_chartgen.tja.quality import QualityReport, build_quality_report
from tja_ai_chartgen.utils.paths import write_json

DEFAULT_AI_REQUEST_TIMEOUT = 300.0
DEFAULT_AI_TRANSPORT_RETRIES = 1
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
    special_notes: bool = False
    use_ai: bool = False
    model: str | None = None
    ai_repair_retries: int = Field(default=2, ge=0)
    ai_request_timeout: float = Field(default=DEFAULT_AI_REQUEST_TIMEOUT, ge=1, le=600)
    ai_transport_retries: int = Field(default=DEFAULT_AI_TRANSPORT_RETRIES, ge=0, le=1)

    @field_serializer("input_audio", "output_dir")
    def serialize_path(self, value: Path) -> str:
        return str(value)

    @model_validator(mode="after")
    def validate_meter(self) -> "GenerationConfig":
        if self.time_signature is not None:
            validate_time_signature(self.time_signature)
        return self


class ChartGenerationResult(BaseModel):
    chart_bars: list[ChartBar]
    quality_report: QualityReport
    ai_failure: str | None = None
    used_fallback: bool


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

    _report_stage(stage_callback, "features")
    bars = assign_sections(build_bar_features(raw, max_bars=max_bars))
    return SongAnalysis(
        title=title,
        artist=artist,
        audio_file=str(input_audio),
        ogg_file=str(ogg_path),
        bpm=raw.bpm,
        offset=raw.offset,
        time_signature=raw.time_signature,
        analyzer=raw.analyzer,
        tempo_analysis=raw.tempo_analysis,
        bars=bars,
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
) -> ChartGenerationResult:
    chart_bars: list[ChartBar] | None = None
    ai_failure: str | None = None

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
            )
        except Exception as error:  # noqa: BLE001 - AI failure must produce a usable fallback.
            ai_failure = _redact_value(str(error), api_key)
            _write_ai_failure(ai_attempts_path, error, api_key)
            _write_ai_failure(ai_output_path, error, api_key)

    used_fallback = chart_bars is None
    if chart_bars is None:
        chart_bars = generate_fallback_chart_bars(
            selected_bars,
            style=style,
            density=density,
            special_notes=special_notes,
            course=course,
            level=level,
        )

    return ChartGenerationResult(
        chart_bars=chart_bars,
        quality_report=build_quality_report(chart_bars, selected_bars),
        ai_failure=ai_failure,
        used_fallback=used_fallback,
    )


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
) -> list[ChartBar]:
    from tja_ai_chartgen.ai.client import generate_chart_bars_with_ai, sanitize_ai_bars
    from tja_ai_chartgen.ai.prompts import build_chart_generation_payload

    selected_analysis = analysis.model_copy(update={"bars": selected_bars})
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
    ai_bars, ai_output = generate_chart_bars_with_ai(
        selected_analysis,
        course,
        level,
        style,
        density,
        model,
        api_base=api_base,
        api_key=api_key,
        max_repair_attempts=ai_repair_retries,
        special_notes=special_notes,
        attempt_log_path=ai_attempts_path,
        request_timeout=ai_request_timeout,
        max_transport_retries=ai_transport_retries,
    )
    _redact_json_file(ai_attempts_path, api_key)
    if ai_output_path is not None:
        write_json(ai_output_path, _redact_value(ai_output, api_key))
    sanitized = sanitize_ai_bars(
        ai_bars,
        expected_count=len(selected_bars),
        expected_bars=selected_bars,
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
