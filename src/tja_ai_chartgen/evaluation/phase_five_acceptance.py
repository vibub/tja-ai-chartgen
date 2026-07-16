from __future__ import annotations

from typing import Any

from tja_ai_chartgen.ai.prompts import build_chart_generation_payload
from tja_ai_chartgen.features.salience import (
    build_bar_accent_salience,
    build_bar_burst_salience,
    build_bar_don_ka_salience,
    build_bar_hit_salience,
    is_reliable_burst,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    GridFeature,
    InstrumentBarFeature,
    InstrumentGridFeature,
    SongAnalysis,
)

PHASE_FIVE_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_FIVE_ACCEPTANCE_VERSION = "phase-five-acceptance-v1"
PHASE_FIVE_BEHAVIOR_MATRIX_VERSION = "phase-five-stem-role-matrix-v1"


def build_phase_five_behavior_matrix() -> dict[str, Any]:
    baseline = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.5,
        grid_features=[GridFeature(grid=9, onset=True, strength=0.6)],
    )
    fallback = baseline.model_copy(
        update={
            "instrument": InstrumentBarFeature(),
            "instrument_grid_features": [],
        }
    )
    baseline_hit = build_bar_hit_salience(baseline)
    fallback_hit = build_bar_hit_salience(fallback)

    drum_bar = baseline.model_copy(
        update={
            "instrument": InstrumentBarFeature(drum_activity=0.8),
            "instrument_grid_features": [
                InstrumentGridFeature(grid=9, drum_onset=0.9)
            ],
        }
    )
    drum_hit = build_bar_hit_salience(drum_bar)
    baseline_drum_point = baseline_hit.points[0]
    drum_point = drum_hit.points[0]

    bass_baseline = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
        activity_grids=[0.0] * 12 + [0.5] + [0.0] * 35,
    )
    bass_enhanced = bass_baseline.model_copy(
        update={
            "instrument": InstrumentBarFeature(bass_activity=0.8),
            "instrument_grid_features": [
                InstrumentGridFeature(grid=7, bass_onset=0.9),
                InstrumentGridFeature(grid=12, bass_onset=0.9),
            ],
        }
    )
    bass_baseline_hit = _point(build_bar_hit_salience(bass_baseline), 12)
    bass_enhanced_hit = _point(build_bar_hit_salience(bass_enhanced), 12)
    bass_baseline_color = _point(build_bar_don_ka_salience(bass_baseline), 12)
    bass_enhanced_color = _point(build_bar_don_ka_salience(bass_enhanced), 12)

    context_baseline = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.6,
        phrase_position="phrase_start",
        transition_role="peak",
        grid_features=[GridFeature(grid=6, onset=True, strength=0.5)],
    )
    vocal_bar = context_baseline.model_copy(
        update={
            "instrument": InstrumentBarFeature(vocal_activity=0.8),
            "instrument_grid_features": [
                InstrumentGridFeature(grid=6, vocal_onset=0.9)
            ],
        }
    )
    accompaniment_bar = context_baseline.model_copy(
        update={
            "instrument": InstrumentBarFeature(other_activity=0.8),
            "instrument_grid_features": [
                InstrumentGridFeature(grid=6, accompaniment_onset=0.9)
            ],
        }
    )
    baseline_accent = _point(build_bar_accent_salience(context_baseline), 6)
    vocal_accent = _point(build_bar_accent_salience(vocal_bar), 6)
    accompaniment_accent = _point(build_bar_accent_salience(accompaniment_bar), 6)

    artifact_bar = BarFeature(
        index=3,
        start_time=6.0,
        end_time=8.0,
        energy=0.5,
        instrument=InstrumentBarFeature(
            vocal_activity=0.8,
            drum_activity=0.02,
            bass_activity=0.8,
            other_activity=0.8,
        ),
        instrument_grid_features=[
            InstrumentGridFeature(grid=4, drum_onset=1.0),
            InstrumentGridFeature(grid=8, vocal_onset=1.0),
            InstrumentGridFeature(grid=12, bass_onset=1.0),
            InstrumentGridFeature(grid=16, accompaniment_onset=1.0),
        ],
    )
    artifact_hit = build_bar_hit_salience(artifact_bar)

    burst_bar = BarFeature(
        index=4,
        start_time=8.0,
        end_time=10.0,
        energy=0.6,
        grids_per_bar=48,
        instrument=InstrumentBarFeature(drum_activity=0.8),
        instrument_grid_features=[
            InstrumentGridFeature(grid=27, drum_onset=0.9),
            InstrumentGridFeature(grid=33, drum_onset=0.9),
            InstrumentGridFeature(grid=39, drum_onset=0.9),
            InstrumentGridFeature(grid=45, drum_onset=0.9),
        ],
    )
    burst = build_bar_burst_salience(burst_bar)

    scenarios = {
        "fallback_preserves_default_salience": {
            "passed": baseline_hit.model_dump() == fallback_hit.model_dump(),
            "baseline": baseline_hit.model_dump(mode="json"),
            "fallback": fallback_hit.model_dump(mode="json"),
        },
        "drum_agreement_strengthens_transient": {
            "passed": (
                drum_point.hit > baseline_drum_point.hit
                and drum_point.confidence > baseline_drum_point.confidence
                and "stem:drum-agreement" in drum_point.reasons
            ),
            "baseline_hit": baseline_drum_point.hit,
            "enhanced_hit": drum_point.hit,
            "baseline_confidence": baseline_drum_point.confidence,
            "enhanced_confidence": drum_point.confidence,
            "reasons": drum_point.reasons,
        },
        "bass_reinforces_beat_and_don_only": {
            "passed": (
                bass_enhanced_hit.hit > bass_baseline_hit.hit
                and bass_enhanced_color.don_preference
                > bass_baseline_color.don_preference
                and not any(point.grid == 7 for point in build_bar_hit_salience(bass_enhanced).points)
                and "stem:bass-beat" in bass_enhanced_hit.reasons
                and "color:bass-onset" in bass_enhanced_color.reasons
            ),
            "baseline_hit": bass_baseline_hit.hit,
            "enhanced_hit": bass_enhanced_hit.hit,
            "baseline_don": bass_baseline_color.don_preference,
            "enhanced_don": bass_enhanced_color.don_preference,
            "offbeat_created": any(
                point.grid == 7 for point in build_bar_hit_salience(bass_enhanced).points
            ),
        },
        "vocal_supports_phrase_context": {
            "passed": (
                vocal_accent.accent >= baseline_accent.accent
                and "accent:vocal-context" in vocal_accent.reasons
            ),
            "baseline_accent": baseline_accent.accent,
            "enhanced_accent": vocal_accent.accent,
            "reasons": vocal_accent.reasons,
        },
        "accompaniment_supports_highlight": {
            "passed": (
                accompaniment_accent.accent >= baseline_accent.accent
                and "accent:accompaniment-highlight" in accompaniment_accent.reasons
            ),
            "baseline_accent": baseline_accent.accent,
            "enhanced_accent": accompaniment_accent.accent,
            "reasons": accompaniment_accent.reasons,
        },
        "stem_artifacts_do_not_create_hits": {
            "passed": not artifact_hit.points,
            "point_count": len(artifact_hit.points),
            "onset_evidence_count": artifact_hit.onset_evidence_count,
        },
        "drum_onset_rise_supports_burst": {
            "passed": (
                is_reliable_burst(burst)
                and burst.burst_start_grid == 27
                and burst.burst_end_grid == 45
                and "burst:drum-onset-rise" in burst.burst_reasons
            ),
            "burst_score": burst.burst_score,
            "burst_confidence": burst.burst_confidence,
            "burst_start_grid": burst.burst_start_grid,
            "burst_end_grid": burst.burst_end_grid,
            "reasons": burst.burst_reasons,
        },
    }
    return {
        "schema_version": 1,
        "matrix_version": PHASE_FIVE_BEHAVIOR_MATRIX_VERSION,
        "passed": all(bool(item["passed"]) for item in scenarios.values()),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def build_phase_five_payload_evidence() -> dict[str, Any]:
    taxonomy_instrument = InstrumentBarFeature(
        vocal_activity=0.7,
        vocal_presence_ratio=0.8,
        drum_activity=0.9,
        bass_activity=0.6,
        other_activity=0.5,
        guitar=0.75,
        synth=0.4,
        dominant_source="drums",
        dominant_instrument="guitar",
        confidence=0.85,
    )
    role_only_instrument = taxonomy_instrument.model_copy(
        update={
            "guitar": 0.0,
            "synth": 0.0,
            "dominant_instrument": None,
        }
    )
    bar = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.5,
        instrument=taxonomy_instrument,
    )
    analysis = SongAnalysis(
        title="Phase 5",
        audio_file="fixture.wav",
        ogg_file="fixture.ogg",
        bpm=120.0,
        offset=0.0,
        instrument_feature_version="instrument-v1",
        instrument_analysis_status="complete",
        bars=[bar],
    )
    taxonomy_payload = build_chart_generation_payload(
        analysis,
        "Oni",
        10,
        "technical",
    )
    role_payload = build_chart_generation_payload(
        analysis.model_copy(
            update={
                "instrument_feature_version": "stem-role-v1",
                "bars": [bar.model_copy(update={"instrument": role_only_instrument})],
            }
        ),
        "Oni",
        10,
        "technical",
    )
    columns = taxonomy_payload["legend"]["instrument_bar_columns"]
    taxonomy_row = taxonomy_payload["bar_instruments"][0]
    role_row = role_payload["bar_instruments"][0]
    return {
        "schema_version": 1,
        "payload_schema": taxonomy_payload["schema"],
        "passed": (
            taxonomy_payload["schema"] == "tja-ai-chartgen-compact-v7"
            and taxonomy_row == role_row
            and "dominant_instrument" not in columns
            and "active_instruments" not in columns
            and "guitar" not in str(taxonomy_row)
            and "synth" not in str(taxonomy_row)
        ),
        "instrument_bar_columns": columns,
        "taxonomy_row": taxonomy_row,
        "role_only_row": role_row,
        "removed_taxonomy_columns": ["dominant_instrument", "active_instruments"],
    }


def build_phase_five_acceptance_report(
    behavior_runs: list[dict[str, Any]],
    model_runs: list[dict[str, Any]],
    payload_runs: list[dict[str, Any]],
    web_runs: list[dict[str, Any]],
    *,
    network_blocked: bool,
) -> dict[str, Any]:
    behavior = behavior_runs[0] if behavior_runs else {}
    model = model_runs[0] if model_runs else {}
    payload = payload_runs[0] if payload_runs else {}
    web = web_runs[0] if web_runs else {}
    scenarios = behavior.get("scenarios", {})
    checks = {
        "stem_role_model_profile": {
            "passed": bool(model.get("passed"))
            and bool(model.get("stem_role", {}).get("hashes_verified"))
            and model.get("stem_role", {}).get("ast_bytes") == 0
            and bool(model.get("full", {}).get("hashes_verified"))
            and model.get("full", {}).get("ast_bytes", 0) > 0
            and bool(model.get("profile_mismatch_rejected")),
            **model,
        },
        "offline_loading_and_cost_report": {
            "passed": (
                model.get("benchmark_version") == "instrument-model-benchmark-v1"
                and bool(model.get("network_attempt_blocked"))
                and bool(model.get("offline_environment_applied"))
                and model.get("stem_role", {}).get("status") == "complete"
                and model.get("stem_role", {}).get("stem_frame_count", 0) > 0
                and model.get("stem_role", {}).get("classification_window_count") == 0
                and model.get("stem_role", {}).get("elapsed_seconds", 0) >= 0
                and model.get("stem_role", {}).get("realtime_factor", 0) >= 0
                and model.get("performance_threshold_enforced") is False
            ),
            "benchmark_version": model.get("benchmark_version"),
            "network_attempt_blocked": model.get("network_attempt_blocked"),
            "offline_environment_applied": model.get("offline_environment_applied"),
            "performance_threshold_enforced": model.get("performance_threshold_enforced"),
            "runtime": model.get("stem_role", {}),
        },
        "fallback_preserves_default_salience": scenarios.get(
            "fallback_preserves_default_salience", {"passed": False}
        ),
        "role_evidence_improves_salience": {
            "passed": all(
                bool(scenarios.get(name, {}).get("passed"))
                for name in (
                    "drum_agreement_strengthens_transient",
                    "bass_reinforces_beat_and_don_only",
                    "vocal_supports_phrase_context",
                    "accompaniment_supports_highlight",
                    "drum_onset_rise_supports_burst",
                )
            ),
            "covered_roles": ["drums", "bass", "vocals", "accompaniment"],
            "scenario_names": sorted(scenarios),
        },
        "stem_artifact_guard": scenarios.get(
            "stem_artifacts_do_not_create_hits", {"passed": False}
        ),
        "ai_payload_taxonomy_independence": {
            "passed": bool(payload.get("passed")),
            **payload,
        },
        "remote_web_admin_gate": {
            "passed": (
                bool(web.get("local_allowed"))
                and web.get("remote_default_allowed") is False
                and bool(web.get("remote_explicit_allowed"))
            ),
            **web,
        },
        "deterministic_offline_acceptance": {
            "passed": (
                network_blocked
                and len(behavior_runs) == 2
                and len(model_runs) == 2
                and len(payload_runs) == 2
                and len(web_runs) == 2
                and _runs_stable(behavior_runs)
                and _runs_stable(model_runs)
                and _runs_stable(payload_runs)
                and _runs_stable(web_runs)
            ),
            "network_blocked": network_blocked,
            "behavior_runs_stable": _runs_stable(behavior_runs),
            "model_runs_stable": _runs_stable(model_runs),
            "payload_runs_stable": _runs_stable(payload_runs),
            "web_runs_stable": _runs_stable(web_runs),
            "run_count": min(
                len(behavior_runs),
                len(model_runs),
                len(payload_runs),
                len(web_runs),
            ),
        },
    }
    return {
        "schema_version": PHASE_FIVE_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_FIVE_ACCEPTANCE_VERSION,
        "phase": "Phase 5",
        "passed": bool(behavior)
        and bool(model)
        and bool(payload)
        and bool(web)
        and all(bool(check.get("passed")) for check in checks.values()),
        "scenario_count": int(behavior.get("scenario_count", 0)),
        "checks": checks,
        "behavior_matrix": behavior,
        "model_profile_evidence": model,
        "payload_evidence": payload,
        "web_permission_evidence": web,
    }


def render_phase_five_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    model = checks["stem_role_model_profile"]
    offline = checks["offline_loading_and_cost_report"]
    lines = [
        "# Phase 5 acceptance report",
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
            "Users can prepare htdemucs-only stem-role models without AST",
            model["passed"],
            f"stem AST {model.get('stem_role', {}).get('ast_bytes')} bytes; "
            f"full AST {model.get('full', {}).get('ast_bytes')} bytes",
        ),
        _check_row(
            "Stem-role loading is offline and performance costs remain report-only",
            offline["passed"],
            f"network blocked={offline.get('network_attempt_blocked')}; "
            f"benchmark `{offline.get('benchmark_version')}`",
        ),
        _check_row(
            "Model fallback preserves default rhythmic salience",
            checks["fallback_preserves_default_salience"]["passed"],
            "baseline and fallback salience are identical",
        ),
        _check_row(
            "Drum, bass, vocal, and accompaniment roles improve their bounded salience targets",
            checks["role_evidence_improves_salience"]["passed"],
            ", ".join(checks["role_evidence_improves_salience"]["covered_roles"]),
        ),
        _check_row(
            "Stem artifacts and non-drum onsets cannot create unconditional hits",
            checks["stem_artifact_guard"]["passed"],
            f"{checks['stem_artifact_guard'].get('point_count')} generated points",
        ),
        _check_row(
            "AI compact payload no longer depends on concrete instrument taxonomy",
            checks["ai_payload_taxonomy_independence"]["passed"],
            f"schema `{checks['ai_payload_taxonomy_independence'].get('payload_schema')}`",
        ),
        _check_row(
            "Remote Web instrument analysis still requires explicit administrator permission",
            checks["remote_web_admin_gate"]["passed"],
            "local allowed; remote default denied; remote explicit allowed",
        ),
        _check_row(
            "Phase 5 acceptance is deterministic and offline",
            checks["deterministic_offline_acceptance"]["passed"],
            f"{checks['deterministic_offline_acceptance']['run_count']} blocked-network runs",
        ),
        "",
        "## Stem-role behavior matrix",
        "",
        "| Scenario | Result |",
        "| --- | :---: |",
    ]
    for name, scenario in report["behavior_matrix"].get("scenarios", {}).items():
        lines.append(f"| `{name}` | {'PASS' if scenario['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "Phase 5 keeps htdemucs opt-in. Performance measurements remain environment-specific report-only diagnostics; no cross-machine latency or memory threshold is enforced.",
            "",
        ]
    )
    return "\n".join(lines)


def _point(salience: Any, grid: int) -> Any:
    return next(point for point in salience.points if point.grid == grid)


def _runs_stable(runs: list[dict[str, Any]]) -> bool:
    return bool(runs) and all(run == runs[0] for run in runs[1:])


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {'PASS' if passed else 'FAIL'} | {evidence} |"
