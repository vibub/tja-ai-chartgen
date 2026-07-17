from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable

from tja_ai_chartgen.ai.prompts import build_chart_generation_payload
from tja_ai_chartgen.ai.sidecars import load_ai_sidecar
from tja_ai_chartgen.audio.instrument_models import (
    INSTRUMENT_FEATURE_VERSION,
    STEM_ROLE_FEATURE_VERSION,
)
from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw
from tja_ai_chartgen.cli import generate as cli_generate
from tja_ai_chartgen.cli import prepare_instrument_models_command
from tja_ai_chartgen.features.salience import build_bar_hit_salience
from tja_ai_chartgen.features.structure import analyze_song_structure
from tja_ai_chartgen.generation import (
    build_analysis_notices,
    build_song_analysis,
    generate_chart_bars,
    load_generation_config,
)
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import (
    BarFeature,
    GridFeature,
    InstrumentBarFeature,
    InstrumentGridFeature,
    ResolutionPlan,
    SongAnalysis,
)
from tja_ai_chartgen.tja.quality import (
    PRIMARY_RHYTHM_ALIGNMENT_METRICS,
    QUALITY_REPORT_METRIC_POLICY_VERSION,
    build_quality_metric_policy_metadata,
    build_quality_report,
)
from tja_ai_chartgen.web import (
    _analysis_form,
    _instrument_analysis_summary,
    _instrument_progress_message,
    create_app,
)

PHASE_SEVEN_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_SEVEN_ACCEPTANCE_VERSION = "phase-seven-acceptance-v1"
PHASE_SEVEN_BEHAVIOR_MATRIX_VERSION = "phase-seven-consumer-convergence-matrix-v1"


def build_phase_seven_behavior_matrix() -> dict[str, Any]:
    base_bars = _base_bars()
    base_structure = analyze_song_structure(base_bars)
    base_analysis = _analysis_from_structure(base_structure, feature_version=None)
    base_chart = generate_fallback_chart_bars(
        base_structure.bars,
        course="Oni",
        level=10,
        style="technical",
        density="medium",
        resolution_plan=base_analysis.resolution_plan,
    )
    base_payload = build_chart_generation_payload(
        base_analysis,
        "Oni",
        10,
        "technical",
        density="medium",
    )
    base_quality = build_quality_report(
        base_chart,
        base_structure.bars,
        base_analysis.resolution_plan,
    )

    baseline_bar = base_bars[0]
    stem_bar = baseline_bar.model_copy(
        update={
            "instrument": InstrumentBarFeature(drum_activity=0.85),
            "instrument_grid_features": [
                InstrumentGridFeature(grid=12, drum_onset=0.95)
            ],
        }
    )
    baseline_salience = build_bar_hit_salience(baseline_bar)
    stem_salience = build_bar_hit_salience(stem_bar)
    baseline_point = _point(baseline_salience, 12)
    stem_point = _point(stem_salience, 12)
    stem_analysis = SongAnalysis(
        title="Phase 7 stem role",
        audio_file="fixture.wav",
        ogg_file="fixture.ogg",
        bpm=120.0,
        offset=0.0,
        instrument_feature_version=STEM_ROLE_FEATURE_VERSION,
        instrument_analysis_status="complete",
        bars=[stem_bar],
    )
    stem_payload = build_chart_generation_payload(
        stem_analysis,
        "Oni",
        10,
        "technical",
    )

    role_bars, taxonomy_bars = _role_and_taxonomy_bars()
    role_structure = analyze_song_structure(role_bars)
    taxonomy_structure = analyze_song_structure(taxonomy_bars)
    role_analysis = _analysis_from_structure(
        role_structure,
        feature_version=INSTRUMENT_FEATURE_VERSION,
    )
    taxonomy_analysis = _analysis_from_structure(
        taxonomy_structure,
        feature_version=INSTRUMENT_FEATURE_VERSION,
    )
    role_chart = generate_fallback_chart_bars(
        role_structure.bars,
        course="Oni",
        level=10,
        style="performance",
        density="high",
        resolution_plan=role_analysis.resolution_plan,
    )
    taxonomy_chart = generate_fallback_chart_bars(
        taxonomy_structure.bars,
        course="Oni",
        level=10,
        style="performance",
        density="high",
        resolution_plan=taxonomy_analysis.resolution_plan,
    )
    role_payload = build_chart_generation_payload(
        role_analysis,
        "Oni",
        10,
        "performance",
        density="high",
    )
    taxonomy_payload = build_chart_generation_payload(
        taxonomy_analysis,
        "Oni",
        10,
        "performance",
        density="high",
    )
    role_quality = build_quality_report(
        role_chart,
        role_structure.bars,
        role_analysis.resolution_plan,
    )
    taxonomy_quality = build_quality_report(
        taxonomy_chart,
        taxonomy_structure.bars,
        taxonomy_analysis.resolution_plan,
    )

    scenarios = {
        "no_heavy_model_rhythm_path": {
            "passed": (
                len(base_structure.bars) == len(base_bars)
                and len(base_chart) == len(base_bars)
                and base_payload["schema"] == "tja-ai-chartgen-compact-v7"
                and len(base_payload["bar_salience"]) == len(base_bars)
                and base_quality.bar_count == len(base_bars)
                and base_analysis.instrument_analysis_status == "unavailable"
            ),
            "bar_count": len(base_bars),
            "chart_bar_count": len(base_chart),
            "payload_schema": base_payload["schema"],
            "quality_policy_version": QUALITY_REPORT_METRIC_POLICY_VERSION,
        },
        "stem_role_adds_bounded_activity_onset_evidence": {
            "passed": (
                stem_point.hit > baseline_point.hit
                and stem_point.confidence > baseline_point.confidence
                and "stem:drum-agreement" in stem_point.reasons
                and {point.grid for point in stem_salience.points}
                == {point.grid for point in baseline_salience.points}
                and stem_payload["legend"]["instrument_bar_columns"]
                == [
                    "vocal_activity",
                    "vocal_presence_ratio",
                    "drum_activity",
                    "bass_activity",
                    "other_activity",
                    "dominant_source",
                    "confidence",
                ]
            ),
            "baseline_hit": baseline_point.hit,
            "stem_hit": stem_point.hit,
            "baseline_confidence": baseline_point.confidence,
            "stem_confidence": stem_point.confidence,
            "point_grids": [point.grid for point in stem_salience.points],
            "instrument_bar_columns": stem_payload["legend"][
                "instrument_bar_columns"
            ],
        },
        "concrete_taxonomy_does_not_change_core_consumers": {
            "passed": (
                _structure_decisions(role_structure) == _structure_decisions(taxonomy_structure)
                and [bar.model_dump() for bar in role_chart]
                == [bar.model_dump() for bar in taxonomy_chart]
                and role_payload == taxonomy_payload
                and _core_quality_payload(role_quality)
                == _core_quality_payload(taxonomy_quality)
            ),
            "structure_equal": _structure_decisions(role_structure)
            == _structure_decisions(taxonomy_structure),
            "fallback_equal": [bar.model_dump() for bar in role_chart]
            == [bar.model_dump() for bar in taxonomy_chart],
            "payload_equal": role_payload == taxonomy_payload,
            "quality_core_equal": _core_quality_payload(role_quality)
            == _core_quality_payload(taxonomy_quality),
            "quality_primary_metrics": list(PRIMARY_RHYTHM_ALIGNMENT_METRICS),
        },
    }
    return {
        "schema_version": 1,
        "matrix_version": PHASE_SEVEN_BEHAVIOR_MATRIX_VERSION,
        "passed": all(bool(scenario["passed"]) for scenario in scenarios.values()),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def build_phase_seven_no_model_pipeline_evidence(
    input_audio: Path,
    work_dir: Path,
) -> dict[str, Any]:
    heavy_modules = {"torch", "demucs", "transformers"}
    loaded_before = heavy_modules.intersection(sys.modules)
    analysis = build_song_analysis(
        input_audio=input_audio,
        ogg_path=work_dir / "phase-seven.ogg",
        title="Phase 7 no-model fixture",
        max_bars=4,
        use_instrument_analysis=False,
    )
    generation = generate_chart_bars(
        analysis=analysis,
        selected_bars=analysis.bars,
        course="Oni",
        level=10,
        style="technical",
        density="medium",
        special_notes=False,
        use_ai=False,
    )
    loaded_after = heavy_modules.intersection(sys.modules)
    newly_loaded = sorted(loaded_after - loaded_before)
    return {
        "schema_version": 1,
        "passed": (
            bool(analysis.bars)
            and len(generation.chart_bars) == len(analysis.bars)
            and generation.quality_report.bar_count == len(analysis.bars)
            and analysis.instrument_feature_version is None
            and analysis.instrument_analysis_status == "unavailable"
            and (work_dir / "phase-seven.ogg").is_file()
            and generation.ai_failure is None
            and not newly_loaded
        ),
        "audio_fixture": input_audio.name,
        "bar_count": len(analysis.bars),
        "chart_bar_count": len(generation.chart_bars),
        "quality_bar_count": generation.quality_report.bar_count,
        "instrument_feature_version": analysis.instrument_feature_version,
        "instrument_analysis_status": analysis.instrument_analysis_status,
        "used_rule_generator": generation.used_fallback,
        "ai_failure": generation.ai_failure,
        "new_heavy_modules": newly_loaded,
        "ogg_created": (work_dir / "phase-seven.ogg").is_file(),
    }


def build_phase_seven_compatibility_evidence(
    compatibility_dir: Path,
) -> dict[str, Any]:
    raw_instrument = InstrumentAnalysisRaw.model_validate_json(
        (compatibility_dir / "instrument_v1_partial.json").read_text(encoding="utf-8")
    )
    analysis = SongAnalysis.model_validate_json(
        (compatibility_dir / "analysis_v5_instrument_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config = load_generation_config(compatibility_dir / "generation_config_v0.json")
    legacy_input = load_ai_sidecar(compatibility_dir / "ai_input_compact_v4.json")
    legacy_output = load_ai_sidecar(compatibility_dir / "ai_output_legacy.json")
    legacy_attempts = load_ai_sidecar(compatibility_dir / "ai_attempts_legacy.json")
    current_payload = build_chart_generation_payload(
        analysis,
        config.course,
        config.level,
        config.style,
        density=config.density,
    )
    legacy_columns = legacy_input["legend"]["instrument_bar_columns"]
    current_columns = current_payload["legend"]["instrument_bar_columns"]
    legacy_row_bytes = len(
        json.dumps(legacy_input["bar_instruments"][0], ensure_ascii=False, separators=(",", ":"))
    )
    current_row_bytes = len(
        json.dumps(current_payload["bar_instruments"][0], ensure_ascii=False, separators=(",", ":"))
    )
    compatibility_passed = (
        raw_instrument.feature_version == INSTRUMENT_FEATURE_VERSION
        and raw_instrument.status == "partial"
        and raw_instrument.reason == "classifier-load-error:OSError"
        and analysis.analysis_schema_version == 5
        and analysis.instrument_analysis_status == "partial"
        and analysis.bars[0].instrument.dominant_instrument == "guitar"
        and config.schema_version == 1
        and config.instrument_profile == "full"
        and not hasattr(config, "ai_api_key")
        and legacy_input["schema"] == "tja-ai-chartgen-compact-v4"
        and legacy_output["final"]["bars"][0]["notes"] == "1000100010001000"
        and legacy_attempts["fallback_reason"] == "content_repair_exhausted"
    )
    compact_passed = (
        current_payload["schema"] == "tja-ai-chartgen-compact-v7"
        and len(current_columns) < len(legacy_columns)
        and current_row_bytes < legacy_row_bytes
        and "dominant_instrument" not in current_columns
        and "active_instruments" not in current_columns
        and "bar_salience" in current_payload
    )
    return {
        "schema_version": 1,
        "passed": compatibility_passed and compact_passed,
        "legacy_read_compatibility": {
            "passed": compatibility_passed,
            "instrument_feature_version": raw_instrument.feature_version,
            "instrument_status": raw_instrument.status,
            "analysis_schema_version": analysis.analysis_schema_version,
            "config_schema_version": config.schema_version,
            "config_default_profile": config.instrument_profile,
            "input_schema": legacy_input["schema"],
            "output_notes_preserved": legacy_output["final"]["bars"][0]["notes"],
            "attempt_fallback_preserved": legacy_attempts["fallback_reason"],
        },
        "compact_payload": {
            "passed": compact_passed,
            "schema": current_payload["schema"],
            "legacy_column_count": len(legacy_columns),
            "current_column_count": len(current_columns),
            "legacy_row_bytes": legacy_row_bytes,
            "current_row_bytes": current_row_bytes,
            "current_columns": current_columns,
            "has_compact_salience": "bar_salience" in current_payload,
        },
    }


def build_phase_seven_interface_evidence() -> dict[str, Any]:
    prepare_default = _option_default(prepare_instrument_models_command, "profile")
    generate_profile_default = _option_default(cli_generate, "instrument_profile")
    generate_enabled_default = _option_default(cli_generate, "use_instrument_analysis")
    form = _analysis_form()
    stem_progress = _instrument_progress_message("stem-role")
    full_progress = _instrument_progress_message("full")

    stem_analysis = _interface_analysis(STEM_ROLE_FEATURE_VERSION)
    full_analysis = _interface_analysis(INSTRUMENT_FEATURE_VERSION)
    classifier_fallback = stem_analysis.model_copy(
        update={"instrument_analysis_reason": "classifier-load-error:OSError"}
    )
    stem_notices = build_analysis_notices(
        stem_analysis,
        requested_instrument_analysis=True,
    )
    full_notices = build_analysis_notices(
        full_analysis,
        requested_instrument_analysis=True,
    )
    classifier_notices = build_analysis_notices(
        classifier_fallback,
        requested_instrument_analysis=True,
    )
    stem_summary = _instrument_analysis_summary(stem_analysis)
    full_summary = _instrument_analysis_summary(full_analysis)

    local = create_app()
    remote_default = create_app(remote_mode=True)
    remote_explicit = create_app(remote_mode=True, allow_instrument_analysis=True)
    passed = (
        prepare_default == "stem-role"
        and generate_profile_default == "stem-role"
        and generate_enabled_default is False
        and 'value="stem-role" selected' in form
        and "推荐：声部节奏" in form
        and "旧兼容：具体乐器分类" in form
        and "Demucs" in stem_progress
        and "旧 full 兼容模式" in full_progress
        and "推荐的声部节奏增强已完成" in stem_notices[0].message
        and "旧 full 兼容分析已完成" in full_notices[0].message
        and "核心生成仍继续使用" in classifier_notices[0].message
        and "当前模式不运行具体乐器分类" in stem_summary
        and "具体乐器兼容诊断" in full_summary
        and bool(local.state.allow_instrument_analysis)
        and remote_default.state.allow_instrument_analysis is False
        and bool(remote_explicit.state.allow_instrument_analysis)
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "cli": {
            "prepare_default_profile": prepare_default,
            "generate_default_profile": generate_profile_default,
            "generate_analysis_enabled_by_default": generate_enabled_default,
        },
        "web": {
            "recommended_option_selected": 'value="stem-role" selected' in form,
            "legacy_option_labeled": "旧兼容：具体乐器分类" in form,
            "stem_progress": stem_progress,
            "full_progress": full_progress,
            "stem_summary_excludes_taxonomy": "当前模式不运行具体乐器分类"
            in stem_summary,
            "full_summary_marks_diagnostic": "具体乐器兼容诊断" in full_summary,
            "local_allowed": bool(local.state.allow_instrument_analysis),
            "remote_default_allowed": bool(
                remote_default.state.allow_instrument_analysis
            ),
            "remote_explicit_allowed": bool(
                remote_explicit.state.allow_instrument_analysis
            ),
        },
        "notices": {
            "stem_success": stem_notices[0].model_dump(mode="json"),
            "full_success": full_notices[0].model_dump(mode="json"),
            "classifier_fallback": classifier_notices[0].model_dump(mode="json"),
        },
    }


def build_phase_seven_acceptance_report(
    behavior_runs: list[dict[str, Any]],
    pipeline_runs: list[dict[str, Any]],
    compatibility_runs: list[dict[str, Any]],
    interface_runs: list[dict[str, Any]],
    *,
    network_blocked: bool,
) -> dict[str, Any]:
    behavior = behavior_runs[0] if behavior_runs else {}
    pipeline = pipeline_runs[0] if pipeline_runs else {}
    compatibility = compatibility_runs[0] if compatibility_runs else {}
    interface = interface_runs[0] if interface_runs else {}
    scenarios = behavior.get("scenarios", {})
    policy = build_quality_metric_policy_metadata()
    no_heavy_consumers = scenarios.get(
        "no_heavy_model_rhythm_path", {"passed": False}
    )
    checks = {
        "no_heavy_model_rhythm_path": {
            "passed": bool(no_heavy_consumers.get("passed"))
            and bool(pipeline.get("passed")),
            "consumer_evidence": no_heavy_consumers,
            "audio_pipeline_evidence": pipeline,
        },
        "stem_role_only_adds_bounded_evidence": scenarios.get(
            "stem_role_adds_bounded_activity_onset_evidence", {"passed": False}
        ),
        "taxonomy_independent_core_generation": scenarios.get(
            "concrete_taxonomy_does_not_change_core_consumers", {"passed": False}
        ),
        "compact_ai_payload": compatibility.get(
            "compact_payload", {"passed": False}
        ),
        "legacy_persistence_compatibility": compatibility.get(
            "legacy_read_compatibility", {"passed": False}
        ),
        "quality_report_rhythm_mainline": {
            "passed": (
                policy.get("version") == QUALITY_REPORT_METRIC_POLICY_VERSION
                and tuple(policy.get("primary_metrics", ()))
                == PRIMARY_RHYTHM_ALIGNMENT_METRICS
                and policy.get("instrument_metrics_policy") == "diagnostic-only"
                and policy.get("has_unified_quality_score") is False
            ),
            **policy,
        },
        "cli_web_notice_capability_alignment": {
            "passed": bool(interface.get("passed")),
            **interface,
        },
        "deterministic_offline_acceptance": {
            "passed": (
                network_blocked
                and len(behavior_runs) == 2
                and len(pipeline_runs) == 2
                and len(compatibility_runs) == 2
                and len(interface_runs) == 2
                and _runs_stable(behavior_runs)
                and _runs_stable(pipeline_runs)
                and _runs_stable(compatibility_runs)
                and _runs_stable(interface_runs)
            ),
            "network_blocked": network_blocked,
            "behavior_runs_stable": _runs_stable(behavior_runs),
            "pipeline_runs_stable": _runs_stable(pipeline_runs),
            "compatibility_runs_stable": _runs_stable(compatibility_runs),
            "interface_runs_stable": _runs_stable(interface_runs),
            "run_count": min(
                len(behavior_runs),
                len(pipeline_runs),
                len(compatibility_runs),
                len(interface_runs),
            ),
        },
    }
    return {
        "schema_version": PHASE_SEVEN_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_SEVEN_ACCEPTANCE_VERSION,
        "phase": "Phase 7",
        "passed": bool(behavior)
        and bool(pipeline)
        and bool(compatibility)
        and bool(interface)
        and all(bool(check.get("passed")) for check in checks.values()),
        "scenario_count": int(behavior.get("scenario_count", 0)),
        "checks": checks,
        "behavior_matrix": behavior,
        "no_model_pipeline_evidence": pipeline,
        "compatibility_evidence": compatibility,
        "interface_evidence": interface,
    }


def render_phase_seven_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    compact = checks["compact_ai_payload"]
    deterministic = checks["deterministic_offline_acceptance"]
    lines = [
        "# Phase 7 acceptance report",
        "",
        f"- Acceptance: `{report['acceptance_version']}`",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Deterministic scenarios: {report['scenario_count']}",
        "",
        "## Exit conditions",
        "",
        "| Exit condition | Result | Evidence |",
        "| --- | :---: | --- |",
        _check_row(
            "The complete rhythm-first path runs without heavyweight models",
            checks["no_heavy_model_rhythm_path"]["passed"],
            f"audio fixture `{checks['no_heavy_model_rhythm_path'].get('audio_pipeline_evidence', {}).get('audio_fixture')}`; "
            f"{checks['no_heavy_model_rhythm_path'].get('audio_pipeline_evidence', {}).get('bar_count')} bars; "
            f"new heavy modules={checks['no_heavy_model_rhythm_path'].get('audio_pipeline_evidence', {}).get('new_heavy_modules')}",
        ),
        _check_row(
            "htdemucs adds only bounded stem activity/onset evidence",
            checks["stem_role_only_adds_bounded_evidence"]["passed"],
            "drum agreement strengthens an existing canonical transient",
        ),
        _check_row(
            "Missing concrete taxonomy does not reduce core generation",
            checks["taxonomy_independent_core_generation"]["passed"],
            "structure, fallback, compact payload, and core QualityReport are unchanged",
        ),
        _check_row(
            "The AI instrument projection is smaller than the legacy instrument-v1 route",
            compact["passed"],
            f"{compact.get('legacy_column_count')}→{compact.get('current_column_count')} columns; "
            f"{compact.get('legacy_row_bytes')}→{compact.get('current_row_bytes')} bytes",
        ),
        _check_row(
            "Legacy analysis, config, and AI sidecars remain readable",
            checks["legacy_persistence_compatibility"]["passed"],
            "analysis v5, config v0, compact-v4 input, legacy output/attempts",
        ),
        _check_row(
            "QualityReport prioritizes rhythm alignment and keeps instrument metrics diagnostic-only",
            checks["quality_report_rhythm_mainline"]["passed"],
            f"`{checks['quality_report_rhythm_mainline'].get('version')}`",
        ),
        _check_row(
            "CLI/Web defaults, progress, summaries, notices, and remote permissions match capabilities",
            checks["cli_web_notice_capability_alignment"]["passed"],
            "stem-role recommended; full explicitly labeled legacy diagnostics",
        ),
        _check_row(
            "Phase 7 acceptance is deterministic and offline",
            deterministic["passed"],
            f"{deterministic.get('run_count')} blocked-network runs",
        ),
        "",
        "## Consumer convergence matrix",
        "",
        "| Scenario | Result |",
        "| --- | :---: |",
    ]
    for name, scenario in report["behavior_matrix"].get("scenarios", {}).items():
        lines.append(f"| `{name}` | {'PASS' if scenario['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "Phase 7 makes stem-role salience the recommended optional enhancement while preserving the legacy instrument-v1 read path. Concrete instrument taxonomy remains compatibility diagnostics only and is excluded from structure decisions, fallback output, compact AI input, and the QualityReport mainline.",
            "",
        ]
    )
    return "\n".join(lines)


def _base_bars() -> list[BarFeature]:
    bars: list[BarFeature] = []
    for index, energy in enumerate((0.35, 0.5, 0.75, 0.45)):
        grid_features = [
            GridFeature(
                grid=grid,
                onset=True,
                strength=min(1.0, energy + (0.15 if grid == 0 else 0.05)),
                activity=energy,
                beat=grid // 12,
                downbeat=grid == 0,
            )
            for grid in (0, 12, 24, 36)
        ]
        bars.append(
            BarFeature(
                index=index,
                start_time=index * 2.0,
                end_time=(index + 1) * 2.0,
                energy=energy,
                grids_per_bar=48,
                onset_grids=[0, 12, 24, 36],
                accent_grids=[0, 24],
                activity_grids=[energy] * 48,
                grid_features=grid_features,
                beat_grids=[0, 12, 24, 36],
                downbeat_grid=0,
            )
        )
    return bars


def _role_and_taxonomy_bars() -> tuple[list[BarFeature], list[BarFeature]]:
    role_bars: list[BarFeature] = []
    taxonomy_bars: list[BarFeature] = []
    for bar in _base_bars():
        role = InstrumentBarFeature(
            vocal_activity=0.4,
            vocal_presence_ratio=0.5,
            drum_activity=0.85,
            bass_activity=0.55,
            other_activity=0.45,
            dominant_source="drums",
            confidence=0.2,
        )
        taxonomy = role.model_copy(
            update={
                "guitar": 0.9,
                "piano_keyboard": 0.7,
                "strings": 0.6,
                "synth": 0.8,
                "dominant_instrument": "guitar",
                "confidence": 0.99,
            }
        )
        instrument_grids = [
            InstrumentGridFeature(grid=0, bass_onset=0.7),
            InstrumentGridFeature(grid=12, drum_onset=0.9),
            InstrumentGridFeature(grid=24, vocal_onset=0.6),
            InstrumentGridFeature(grid=36, accompaniment_onset=0.7),
        ]
        role_bars.append(
            bar.model_copy(
                update={
                    "instrument": role,
                    "instrument_grid_features": instrument_grids,
                }
            )
        )
        taxonomy_bars.append(
            bar.model_copy(
                update={
                    "instrument": taxonomy,
                    "instrument_grid_features": instrument_grids,
                }
            )
        )
    return role_bars, taxonomy_bars


def _analysis_from_structure(
    structure: Any,
    *,
    feature_version: str | None,
) -> SongAnalysis:
    bar_count = len(structure.bars)
    return SongAnalysis(
        title="Phase 7",
        audio_file="fixture.wav",
        ogg_file="fixture.ogg",
        bpm=120.0,
        offset=0.0,
        instrument_feature_version=feature_version,
        instrument_analysis_status="complete" if feature_version else "unavailable",
        structure_feature_version="structure-v1",
        structure_confidence=structure.confidence,
        bar_structures=structure.bar_structures,
        phrase_plan=structure.phrases,
        resolution_policy_version="song-global-v1",
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16] * bar_count,
        ),
        bars=structure.bars,
    )


def _interface_analysis(feature_version: str) -> SongAnalysis:
    instrument = InstrumentBarFeature(
        drum_activity=0.8,
        guitar=0.9 if feature_version == INSTRUMENT_FEATURE_VERSION else 0.0,
        dominant_source="drums",
        dominant_instrument=(
            "guitar" if feature_version == INSTRUMENT_FEATURE_VERSION else None
        ),
        confidence=0.9,
    )
    return SongAnalysis(
        title="Phase 7 interface",
        audio_file="fixture.wav",
        ogg_file="fixture.ogg",
        bpm=120.0,
        offset=0.0,
        instrument_feature_version=feature_version,
        instrument_analysis_status="complete",
        instrument_demucs_model="htdemucs",
        instrument_classifier_model=(
            "ast" if feature_version == INSTRUMENT_FEATURE_VERSION else None
        ),
        instrument_analysis_device="cpu",
        bars=[
            BarFeature(
                index=0,
                start_time=0.0,
                end_time=2.0,
                energy=0.7,
                instrument=instrument,
            )
        ],
    )


def _structure_decisions(structure: Any) -> list[dict[str, Any]]:
    return [
        {
            "onset_density": item.onset_density,
            "onset_strength_mean": item.onset_strength_mean,
            "onset_strength_max": item.onset_strength_max,
            "accent_density": item.accent_density,
            "rhythm_profile": item.rhythm_profile,
            "boundary_confidence": item.boundary_confidence,
            "phrase_id": item.phrase_id,
            "phrase_progress": item.phrase_progress,
            "phrase_position": item.phrase_position,
            "transition_role": item.transition_role,
            "section_id": item.section_id,
            "section": item.section,
            "fill_candidate_score": item.fill_candidate_score,
        }
        for item in structure.bar_structures
    ]


def _core_quality_payload(report: Any) -> dict[str, Any]:
    return {
        "note_onset_alignment": report.note_onset_alignment,
        "strong_onset_response": report.strong_onset_response,
        "unsupported_note_rate": report.unsupported_note_rate,
        "downbeat_response": report.downbeat_response,
        "fill_burst_alignment": report.fill_burst_alignment,
        "rhythmic_quantization_error": report.rhythmic_quantization_error,
        "silent_range_violation_count": report.silent_range_violation_count,
        "silent_range_violation_rate": report.silent_range_violation_rate,
    }


def _option_default(function: Callable[..., Any], name: str) -> Any:
    default = inspect.signature(function).parameters[name].default
    return getattr(default, "default", default)


def _point(salience: Any, grid: int) -> Any:
    return next(point for point in salience.points if point.grid == grid)


def _runs_stable(runs: list[dict[str, Any]]) -> bool:
    return bool(runs) and all(run == runs[0] for run in runs[1:])


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {'PASS' if passed else 'FAIL'} | {evidence} |"
