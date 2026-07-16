from __future__ import annotations

from typing import Any

from tja_ai_chartgen.ai.rhythm_repair import (
    AI_REPAIR_STRONG_ONSET_MIN_BARS,
    AI_REPAIR_STRONG_ONSET_MIN_EVALUATED,
    AI_REPAIR_UNSUPPORTED_NOTE_MIN_COUNT,
    AI_REPAIR_UNSUPPORTED_NOTE_MIN_EVALUATED,
    AI_REPAIR_UNSUPPORTED_NOTE_RATE,
    build_rhythm_repair_gate_metadata,
    selected_rhythm_quality_issues,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    ChartBar,
    GridFeature,
    ResolutionPlan,
    SongAnalysis,
)
from tja_ai_chartgen.tja.quality import QualityReport, build_quality_report

PHASE_SIX_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_SIX_ACCEPTANCE_VERSION = "phase-six-acceptance-v1"
PHASE_SIX_BEHAVIOR_MATRIX_VERSION = "phase-six-quality-gate-matrix-v1"
CORE_REPORT_ONLY_METRICS = (
    "note_onset_alignment",
    "strong_onset_response",
    "downbeat_response",
    "unsupported_note_rate",
    "silent_range_violation_rate",
    "fill_burst_alignment",
    "rhythmic_quantization_error",
    "salience_coverage_by_density",
)
DENSITY_HINT_KINDS = {"silent", "rest", "sparse", "normal", "dense", "fill"}
UNIFIED_SCORE_KEYS = {
    "score",
    "overall_score",
    "quality_score",
    "total_score",
    "weighted_score",
}


def build_phase_six_behavior_matrix() -> dict[str, Any]:
    shared_contract = _shared_rule_ai_quality_contract()
    resolution_scenarios = {
        "resolution_equivalence_4_4": _resolution_equivalence_scenario(
            time_signature="4/4",
            canonical_grids=48,
            resolutions=(16, 24, 48),
            onset_grids=(0, 12, 24, 36),
            beat_grids=(0, 12, 24, 36),
        ),
        "resolution_equivalence_3_4": _resolution_equivalence_scenario(
            time_signature="3/4",
            canonical_grids=36,
            resolutions=(12, 18, 36),
            onset_grids=(0, 12, 24),
            beat_grids=(0, 12, 24),
        ),
        "resolution_equivalence_6_8": _resolution_equivalence_scenario(
            time_signature="6/8",
            canonical_grids=36,
            resolutions=(12, 18, 36),
            onset_grids=(0, 6, 12, 18, 24, 30),
            beat_grids=(0, 18),
        ),
    }
    gate_trigger = _selected_gate_trigger_scenario()
    small_sample = _small_sample_guard_scenario()
    scenarios = {
        "rule_and_ai_share_quality_report": shared_contract,
        **resolution_scenarios,
        "selected_repair_gates_trigger": gate_trigger,
        "small_samples_remain_report_only": small_sample,
    }
    return {
        "schema_version": 1,
        "matrix_version": PHASE_SIX_BEHAVIOR_MATRIX_VERSION,
        "passed": all(bool(scenario["passed"]) for scenario in scenarios.values()),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def build_phase_six_benchmark_evidence(benchmark: dict[str, Any]) -> dict[str, Any]:
    charts = benchmark.get("charts", [])
    calibration = benchmark.get("report_only_calibration", {})
    calibration_summary = calibration.get("summary", {})
    course_summaries = calibration.get("course_summaries", {})
    missing_metrics: list[dict[str, Any]] = []
    gate_violations: list[dict[str, Any]] = []
    unified_score_keys: set[str] = set()

    for item in charts:
        report = item.get("report_only", {})
        missing = [key for key in CORE_REPORT_ONLY_METRICS if key not in report]
        if missing:
            missing_metrics.append(
                {
                    "audio": item.get("audio"),
                    "course": item.get("course"),
                    "missing": missing,
                }
            )
        gate_violations.extend(_baseline_gate_violations(item))
        unified_score_keys.update(key for key in report if key in UNIFIED_SCORE_KEYS)

    return {
        "schema_version": 1,
        "benchmark_version": benchmark.get("benchmark_version"),
        "fixture_count": benchmark.get("fixture_count"),
        "chart_count": benchmark.get("chart_count"),
        "deterministic_rate": benchmark.get("summary", {}).get("deterministic_rate"),
        "report_only_chart_count": sum("report_only" in item for item in charts),
        "missing_metric_count": len(missing_metrics),
        "missing_metric_examples": missing_metrics[:8],
        "course_names": sorted(course_summaries),
        "density_hint_names": sorted(
            calibration_summary.get("salience_coverage_by_density", {})
        ),
        "ground_truth_comparison": calibration.get("ground_truth_comparison", {}),
        "selected_gate_violation_count": len(gate_violations),
        "selected_gate_violation_examples": gate_violations[:8],
        "unified_score_keys": sorted(unified_score_keys),
        "calibration_summary": calibration_summary,
    }


def build_phase_six_acceptance_report(
    behavior_runs: list[dict[str, Any]],
    benchmark_runs: list[dict[str, Any]],
    *,
    network_blocked: bool,
) -> dict[str, Any]:
    behavior = behavior_runs[0] if behavior_runs else {}
    benchmark = benchmark_runs[0] if benchmark_runs else {}
    scenarios = behavior.get("scenarios", {})
    comparison = benchmark.get("ground_truth_comparison", {})
    gate_metadata = build_rhythm_repair_gate_metadata()
    selected_metrics = set(gate_metadata.get("selected_metrics", {}))
    report_only_metrics = set(gate_metadata.get("report_only_metrics", []))
    checks = {
        "shared_rule_ai_quality_contract": scenarios.get(
            "rule_and_ai_share_quality_report", {"passed": False}
        ),
        "fixture_ground_truth_metric_coverage": {
            "passed": (
                benchmark.get("benchmark_version") == "chart-alignment-v1"
                and benchmark.get("fixture_count") == 17
                and benchmark.get("chart_count") == 68
                and benchmark.get("report_only_chart_count") == 68
                and benchmark.get("missing_metric_count") == 0
                and set(benchmark.get("course_names", []))
                == {"Easy", "Normal", "Hard", "Oni"}
                and set(benchmark.get("density_hint_names", [])) == DENSITY_HINT_KINDS
            ),
            "benchmark_version": benchmark.get("benchmark_version"),
            "fixture_count": benchmark.get("fixture_count"),
            "chart_count": benchmark.get("chart_count"),
            "report_only_chart_count": benchmark.get("report_only_chart_count"),
            "missing_metric_count": benchmark.get("missing_metric_count"),
            "course_names": benchmark.get("course_names", []),
            "density_hint_names": benchmark.get("density_hint_names", []),
        },
        "equivalent_resolution_stability": {
            "passed": all(
                bool(scenarios.get(name, {}).get("passed"))
                for name in (
                    "resolution_equivalence_4_4",
                    "resolution_equivalence_3_4",
                    "resolution_equivalence_6_8",
                )
            ),
            "families": {
                name: scenarios.get(name, {})
                for name in (
                    "resolution_equivalence_4_4",
                    "resolution_equivalence_3_4",
                    "resolution_equivalence_6_8",
                )
            },
        },
        "calibrated_ab_report": {
            "passed": (
                benchmark.get("deterministic_rate") == 1.0
                and {
                    "note_onset_alignment_delta",
                    "strong_onset_response_delta",
                    "downbeat_response_delta",
                    "unsupported_note_rate_delta",
                    "silent_violation_count_delta",
                }.issubset(comparison)
            ),
            "deterministic_rate": benchmark.get("deterministic_rate"),
            "ground_truth_comparison": comparison,
        },
        "selected_ai_repair_gate_scope": {
            "passed": (
                gate_metadata.get("version") == "ai-rhythm-repair-gate-v1"
                and selected_metrics
                == {
                    "silent_range_violation",
                    "unsupported_note_rate",
                    "strong_onset_response",
                }
                and report_only_metrics
                == {
                    "note_onset_alignment",
                    "downbeat_response",
                    "fill_burst_alignment",
                    "rhythmic_quantization_error",
                    "salience_coverage_by_density",
                }
                and benchmark.get("selected_gate_violation_count") == 0
                and bool(scenarios.get("selected_repair_gates_trigger", {}).get("passed"))
                and bool(scenarios.get("small_samples_remain_report_only", {}).get("passed"))
            ),
            "gate": gate_metadata,
            "baseline_violation_count": benchmark.get("selected_gate_violation_count"),
            "baseline_violation_examples": benchmark.get(
                "selected_gate_violation_examples", []
            ),
        },
        "no_unified_quality_score": {
            "passed": (
                not benchmark.get("unified_score_keys")
                and not UNIFIED_SCORE_KEYS.intersection(QualityReport.model_fields)
            ),
            "benchmark_score_keys": benchmark.get("unified_score_keys", []),
            "quality_report_score_keys": sorted(
                UNIFIED_SCORE_KEYS.intersection(QualityReport.model_fields)
            ),
        },
        "deterministic_offline_acceptance": {
            "passed": (
                network_blocked
                and len(behavior_runs) == 2
                and len(benchmark_runs) == 2
                and _runs_stable(behavior_runs)
                and _runs_stable(benchmark_runs)
            ),
            "network_blocked": network_blocked,
            "behavior_runs_stable": _runs_stable(behavior_runs),
            "benchmark_runs_stable": _runs_stable(benchmark_runs),
            "run_count": min(len(behavior_runs), len(benchmark_runs)),
        },
    }
    return {
        "schema_version": PHASE_SIX_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_SIX_ACCEPTANCE_VERSION,
        "phase": "Phase 6",
        "passed": bool(behavior)
        and bool(benchmark)
        and all(bool(check.get("passed")) for check in checks.values()),
        "fixture_count": int(benchmark.get("fixture_count", 0)),
        "chart_count": int(benchmark.get("chart_count", 0)),
        "scenario_count": int(behavior.get("scenario_count", 0)),
        "checks": checks,
        "behavior_matrix": behavior,
        "benchmark_evidence": benchmark,
    }


def render_phase_six_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    coverage = checks["fixture_ground_truth_metric_coverage"]
    gate = checks["selected_ai_repair_gate_scope"]
    lines = [
        "# Phase 6 acceptance report",
        "",
        f"- Acceptance: `{report['acceptance_version']}`",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Fixtures: {report['fixture_count']}",
        f"- Charts: {report['chart_count']}",
        f"- Deterministic scenarios: {report['scenario_count']}",
        "",
        "## Exit conditions",
        "",
        "| Exit condition | Result | Evidence |",
        "| --- | :---: | --- |",
        _check_row(
            "Rule and AI products use the same QualityReport contract",
            checks["shared_rule_ai_quality_contract"]["passed"],
            "identical structured bars produce identical reports",
        ),
        _check_row(
            "All fixture charts expose every core report-only metric",
            coverage["passed"],
            f"{coverage.get('report_only_chart_count')}/{coverage.get('chart_count')} charts; "
            f"{coverage.get('missing_metric_count')} missing",
        ),
        _check_row(
            "Equivalent 16/24/48 and 12/18/36 encodings remain stable",
            checks["equivalent_resolution_stability"]["passed"],
            "4/4, 3/4, and 6/8 resolution families",
        ),
        _check_row(
            "The fixture benchmark includes a deterministic A/B calibration report",
            checks["calibrated_ab_report"]["passed"],
            "QualityReport canonical salience versus independent fixture ground truth",
        ),
        _check_row(
            "Only calibrated extreme metrics enter AI repair",
            gate["passed"],
            f"`{gate.get('gate', {}).get('version')}`; "
            f"{gate.get('baseline_violation_count')} baseline violations",
        ),
        _check_row(
            "QualityReport has no unified quality score",
            checks["no_unified_quality_score"]["passed"],
            "metric families remain independently interpretable",
        ),
        _check_row(
            "Phase 6 acceptance is deterministic and offline",
            checks["deterministic_offline_acceptance"]["passed"],
            f"{checks['deterministic_offline_acceptance']['run_count']} blocked-network runs",
        ),
        "",
        "## Deterministic behavior matrix",
        "",
        "| Scenario | Result |",
        "| --- | :---: |",
    ]
    for name, scenario in report["behavior_matrix"].get("scenarios", {}).items():
        lines.append(f"| `{name}` | {'PASS' if scenario['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "Phase 6 keeps note/onset, downbeat, fill burst, rhythmic quantization, density coverage, structure, instrument, and resolution diagnostics report-only. Only reliable silence violations, extreme unsupported-note output, and complete multi-bar strong-onset misses enter AI content repair; no unified score is introduced.",
            "",
        ]
    )
    return "\n".join(lines)


def _shared_rule_ai_quality_contract() -> dict[str, Any]:
    feature = _feature_bar(
        time_signature="4/4",
        canonical_grids=48,
        onset_grids=(0, 12, 24, 36),
        beat_grids=(0, 12, 24, 36),
    )
    plan = ResolutionPlan(
        canonical_grids_per_bar=48,
        base_resolution=16,
        bar_resolutions=[16],
    )
    rule_bars = [_chart_bar(16, 48, (0, 12, 24, 36))]
    ai_bars = [rule_bars[0].model_copy(deep=True)]
    rule_report = build_quality_report(rule_bars, [feature], plan)
    ai_report = build_quality_report(ai_bars, [feature], plan)
    return {
        "passed": rule_report.model_dump() == ai_report.model_dump(),
        "quality_report_fields": sorted(QualityReport.model_fields),
        "rule_report": _core_quality_payload(rule_report),
        "ai_report": _core_quality_payload(ai_report),
    }


def _resolution_equivalence_scenario(
    *,
    time_signature: str,
    canonical_grids: int,
    resolutions: tuple[int, ...],
    onset_grids: tuple[int, ...],
    beat_grids: tuple[int, ...],
) -> dict[str, Any]:
    feature = _feature_bar(
        time_signature=time_signature,
        canonical_grids=canonical_grids,
        onset_grids=onset_grids,
        beat_grids=beat_grids,
    )
    reports: dict[str, dict[str, Any]] = {}
    for resolution in resolutions:
        plan = ResolutionPlan(
            canonical_grids_per_bar=canonical_grids,
            base_resolution=resolution,
            bar_resolutions=[resolution],
        )
        report = build_quality_report(
            [_chart_bar(resolution, canonical_grids, onset_grids)],
            [feature],
            plan,
        )
        reports[str(resolution)] = _core_quality_payload(report)
    values = list(reports.values())
    return {
        "passed": bool(values) and all(value == values[0] for value in values[1:]),
        "time_signature": time_signature,
        "resolutions": list(resolutions),
        "reports": reports,
    }


def _selected_gate_trigger_scenario() -> dict[str, Any]:
    silent_analysis = _analysis_from_bars(
        [
            BarFeature(
                index=0,
                start_time=0.0,
                end_time=2.0,
                energy=0.0,
                grids_per_bar=48,
                phrase_position="phrase_middle",
                section="break",
            )
        ]
    )
    silent_issues = selected_rhythm_quality_issues(
        [ChartBar(index=0, notes="1000000000000000")],
        analysis=silent_analysis,
    )

    unsupported_analysis = _analysis_without_transients(bar_count=4)
    unsupported_issues = selected_rhythm_quality_issues(
        [ChartBar(index=index, notes="0001000000010000") for index in range(4)],
        analysis=unsupported_analysis,
    )

    strong_analysis = _analysis_with_offbeat_strong_onsets(bar_count=4)
    strong_issues = selected_rhythm_quality_issues(
        [ChartBar(index=index, notes="1000100010001000") for index in range(4)],
        analysis=strong_analysis,
    )
    return {
        "passed": (
            any("reliable silent ranges" in issue for issue in silent_issues)
            and any("unsupported-note rate" in issue for issue in unsupported_issues)
            and any("ignores every reliable strong onset" in issue for issue in strong_issues)
        ),
        "silent_issues": silent_issues,
        "unsupported_issues": unsupported_issues,
        "strong_onset_issues": strong_issues,
    }


def _small_sample_guard_scenario() -> dict[str, Any]:
    unsupported_analysis = _analysis_without_transients(bar_count=3)
    unsupported_issues = selected_rhythm_quality_issues(
        [ChartBar(index=index, notes="0001000000010000") for index in range(3)],
        analysis=unsupported_analysis,
    )
    strong_analysis = _analysis_with_offbeat_strong_onsets(bar_count=3)
    strong_issues = selected_rhythm_quality_issues(
        [ChartBar(index=index, notes="1000100010001000") for index in range(3)],
        analysis=strong_analysis,
    )
    return {
        "passed": (
            not any("unsupported-note" in issue for issue in unsupported_issues)
            and not any("strong onset" in issue for issue in strong_issues)
        ),
        "unsupported_issues": unsupported_issues,
        "strong_onset_issues": strong_issues,
    }


def _baseline_gate_violations(item: dict[str, Any]) -> list[dict[str, Any]]:
    report = item.get("report_only", {})
    violations: list[dict[str, Any]] = []
    silent = report.get("silent_range_violation_rate", {})
    unsupported = report.get("unsupported_note_rate", {})
    strong = report.get("strong_onset_response", {})
    if silent.get("violation_count", 0) > 0:
        violations.append(_violation(item, "silent_range_violation"))
    if (
        unsupported.get("evaluated_count", 0) >= AI_REPAIR_UNSUPPORTED_NOTE_MIN_EVALUATED
        and unsupported.get("unsupported_count", 0) >= AI_REPAIR_UNSUPPORTED_NOTE_MIN_COUNT
        and unsupported.get("value", 0.0) >= AI_REPAIR_UNSUPPORTED_NOTE_RATE
    ):
        violations.append(_violation(item, "unsupported_note_rate"))
    if (
        item.get("bar_count", 0) >= AI_REPAIR_STRONG_ONSET_MIN_BARS
        and strong.get("evaluated_count", 0) >= AI_REPAIR_STRONG_ONSET_MIN_EVALUATED
        and strong.get("responded_count", 0) == 0
    ):
        violations.append(_violation(item, "strong_onset_response"))
    return violations


def _violation(item: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "audio": item.get("audio"),
        "course": item.get("course"),
        "metric": metric,
    }


def _feature_bar(
    *,
    time_signature: str,
    canonical_grids: int,
    onset_grids: tuple[int, ...],
    beat_grids: tuple[int, ...],
) -> BarFeature:
    return BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.6,
        time_signature=time_signature,
        grids_per_bar=canonical_grids,
        onset_grids=list(onset_grids),
        beat_grids=list(beat_grids),
        downbeat_grid=0,
        grid_features=[
            GridFeature(
                grid=grid,
                onset=True,
                strength=1.0,
                activity=0.8,
                downbeat=grid == 0,
            )
            for grid in onset_grids
        ],
    )


def _chart_bar(
    resolution: int,
    canonical_grids: int,
    grids: tuple[int, ...],
) -> ChartBar:
    notes = ["0"] * resolution
    for index, grid in enumerate(grids):
        position = grid * resolution // canonical_grids
        notes[position] = "1" if index % 2 == 0 else "2"
    return ChartBar(index=0, notes="".join(notes))


def _analysis_without_transients(*, bar_count: int) -> SongAnalysis:
    bars = [
        BarFeature(
            index=index,
            start_time=index * 2.0,
            end_time=(index + 1) * 2.0,
            energy=0.2,
            grids_per_bar=48,
            onset_grids=[],
            beat_grids=[0, 12, 24, 36],
            downbeat_grid=0,
            phrase_position="phrase_start" if index == 0 else "phrase_middle",
            section="verse",
        )
        for index in range(bar_count)
    ]
    return _analysis_from_bars(bars)


def _analysis_with_offbeat_strong_onsets(*, bar_count: int) -> SongAnalysis:
    onset_grids = [6, 18, 30, 42]
    bars = [
        BarFeature(
            index=index,
            start_time=index * 2.0,
            end_time=(index + 1) * 2.0,
            energy=0.2,
            grids_per_bar=48,
            onset_grids=onset_grids,
            beat_grids=[0, 12, 24, 36],
            downbeat_grid=0,
            grid_features=[
                GridFeature(grid=grid, onset=True, strength=1.0, activity=0.8)
                for grid in onset_grids
            ],
            phrase_position="phrase_start" if index == 0 else "phrase_middle",
            section="verse",
        )
        for index in range(bar_count)
    ]
    return _analysis_from_bars(bars)


def _analysis_from_bars(bars: list[BarFeature]) -> SongAnalysis:
    return SongAnalysis(
        title="Phase 6",
        audio_file="fixture.wav",
        ogg_file="fixture.ogg",
        bpm=120.0,
        offset=0.0,
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16] * len(bars),
        ),
        bars=bars,
    )


def _core_quality_payload(report: QualityReport) -> dict[str, Any]:
    payload = report.model_dump(mode="json")
    return {key: payload[key] for key in CORE_REPORT_ONLY_METRICS}


def _runs_stable(runs: list[dict[str, Any]]) -> bool:
    return bool(runs) and all(run == runs[0] for run in runs[1:])


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {'PASS' if passed else 'FAIL'} | {evidence} |"
