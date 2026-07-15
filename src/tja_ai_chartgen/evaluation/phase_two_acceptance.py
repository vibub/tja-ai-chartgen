from __future__ import annotations

from collections import Counter
from math import floor
from typing import Any

from tja_ai_chartgen.ai.client import (
    AI_SALIENCE_VALIDATION_VERSION,
    build_ai_salience_validation_report,
)
from tja_ai_chartgen.evaluation.chart_alignment import (
    CHART_ALIGNMENT_BENCHMARK_VERSION,
)
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import (
    BarFeature,
    GridFeature,
    ResolutionPlan,
    SongAnalysis,
)
from tja_ai_chartgen.tja.quality import playable_hit_count

PHASE_TWO_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_TWO_ACCEPTANCE_VERSION = "phase-two-acceptance-v1"
COURSE_PROFILES = (
    ("Easy", 3, 3.0, 0.38),
    ("Normal", 5, 5.0, 0.50),
    ("Hard", 7, 7.5, 0.70),
    ("Oni", 10, 10.0, 0.82),
)
DENSITIES = ("low", "medium", "auto", "high", "max")
METER_CASES = (
    ("4/4", 48, 16, (16, 24, 48), 2.0, (0, 12, 24, 36)),
    ("3/4", 36, 12, (12, 18, 36), 1.5, (0, 12, 24)),
    ("6/8", 36, 12, (12, 18, 36), 1.5, (0, 18)),
)
REQUIRED_SALIENCE_REPORT_FIELDS = {
    "normal_note_count",
    "representable_note_count",
    "unrepresentable_note_count",
    "reliable_candidate_coverage",
    "strong_transient_coverage",
    "unsupported_note_ratio",
    "silent_bar_note_count",
}
MAX_VIOLATION_EXAMPLES = 32


def build_phase_two_constraint_matrix() -> dict[str, Any]:
    violations: Counter[str] = Counter()
    examples: list[str] = []
    load_configuration_count = 0
    silence_configuration_count = 0
    bar_result_count = 0

    for (
        time_signature,
        canonical_grids,
        legacy_resolution,
        resolutions,
        duration,
        beats,
    ) in METER_CASES:
        dense_bar = _bar(
            0,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.95,
            onset_grids=list(range(canonical_grids)),
            activity=0.9,
            beats=beats,
        )
        counts: dict[tuple[int, str, str], int] = {}
        for resolution in resolutions:
            plan = _resolution_plan(canonical_grids, resolution, bar_count=1)
            for course, level, speed_cap, occupancy_cap in COURSE_PROFILES:
                for density in DENSITIES:
                    chart_bar = generate_fallback_chart_bars(
                        [dense_bar],
                        density=density,
                        course=course,
                        level=level,
                        resolution_plan=plan,
                    )[0]
                    count = playable_hit_count(chart_bar.notes)
                    counts[(resolution, course, density)] = count
                    load_configuration_count += 1
                    bar_result_count += 1
                    prefix = f"{time_signature}:{resolution}:{course}:{density}"
                    _record(
                        len(chart_bar.notes) == resolution,
                        "output-resolution",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        count / duration <= speed_cap,
                        "nps-cap",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        count <= max(1, floor(duration * speed_cap)),
                        "absolute-speed-cap",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        count <= floor(legacy_resolution * occupancy_cap),
                        "legacy-occupancy-cap",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        count / resolution <= occupancy_cap,
                        "output-occupancy-cap",
                        prefix,
                        violations,
                        examples,
                    )

        for resolution in resolutions:
            for course, *_rest in COURSE_PROFILES:
                values = [counts[(resolution, course, density)] for density in DENSITIES]
                _record(
                    values == sorted(values),
                    "density-monotonicity",
                    f"{time_signature}:{resolution}:{course}:{values}",
                    violations,
                    examples,
                )
            for density in DENSITIES:
                values = [
                    counts[(resolution, course, density)]
                    for course, *_rest in COURSE_PROFILES
                ]
                _record(
                    values == sorted(values),
                    "course-monotonicity",
                    f"{time_signature}:{resolution}:{density}:{values}",
                    violations,
                    examples,
                )

        for course, *_rest in COURSE_PROFILES:
            for density in DENSITIES:
                values = {counts[(resolution, course, density)] for resolution in resolutions}
                _record(
                    len(values) == 1,
                    "resolution-equivalence",
                    f"{time_signature}:{course}:{density}:{sorted(values)}",
                    violations,
                    examples,
                )

        boundary_bars = _boundary_bars(
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            beats=beats,
        )
        for resolution in resolutions:
            plan = _resolution_plan(
                canonical_grids,
                resolution,
                bar_count=len(boundary_bars),
            )
            for course, level, *_rest in COURSE_PROFILES:
                for density in DENSITIES:
                    chart_bars = generate_fallback_chart_bars(
                        boundary_bars,
                        density=density,
                        course=course,
                        level=level,
                        resolution_plan=plan,
                    )
                    silence_configuration_count += 1
                    bar_result_count += len(chart_bars)
                    prefix = f"{time_signature}:{resolution}:{course}:{density}"
                    _record(
                        chart_bars[0].notes == "0" * resolution,
                        "leading-silence",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        playable_hit_count(chart_bars[1].notes) > 0,
                        "active-bar-nonempty",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        chart_bars[2].notes == "0" * resolution,
                        "middle-rest",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        playable_hit_count(chart_bars[3].notes) <= 4,
                        "sparse-cap",
                        prefix,
                        violations,
                        examples,
                    )
                    _record(
                        chart_bars[4].notes == "0" * resolution,
                        "trailing-silence",
                        prefix,
                        violations,
                        examples,
                    )

    violation_counts = dict(sorted(violations.items()))
    return {
        "schema_version": 1,
        "matrix_version": "phase-two-constraint-matrix-v1",
        "passed": not violation_counts,
        "load_configuration_count": load_configuration_count,
        "silence_configuration_count": silence_configuration_count,
        "configuration_count": load_configuration_count + silence_configuration_count,
        "bar_result_count": bar_result_count,
        "meter_count": len(METER_CASES),
        "course_count": len(COURSE_PROFILES),
        "density_count": len(DENSITIES),
        "violation_count": sum(violation_counts.values()),
        "violation_counts": violation_counts,
        "violation_examples": examples,
    }


def build_shared_salience_metric_evidence() -> dict[str, Any]:
    bar = _bar(
        0,
        time_signature="4/4",
        canonical_grids=48,
        duration=2.0,
        energy=0.8,
        onset_grids=[0, 6, 12, 18, 24, 30, 36, 42],
        activity=0.8,
        beats=(0, 12, 24, 36),
    )
    plan = _resolution_plan(48, 16, bar_count=1)
    chart_bars = generate_fallback_chart_bars(
        [bar],
        course="Oni",
        level=10,
        resolution_plan=plan,
    )
    analysis = SongAnalysis(
        title="Phase 2 acceptance",
        audio_file="synthetic.wav",
        ogg_file="synthetic.ogg",
        bpm=120.0,
        offset=0.0,
        resolution_plan=plan,
        bars=[bar],
    )
    report = build_ai_salience_validation_report(chart_bars, analysis)
    missing_fields = sorted(REQUIRED_SALIENCE_REPORT_FIELDS - report.keys())
    return {
        "metric_version": report.get("schema"),
        "report_only": report.get("report_only"),
        "rule_chart_evaluated_with_ai_metric": True,
        "required_fields": sorted(REQUIRED_SALIENCE_REPORT_FIELDS),
        "missing_fields": missing_fields,
        "sample_report": report,
    }


def build_phase_two_acceptance_report(
    chart_benchmark_runs: list[dict[str, Any]],
    *,
    committed_chart_baseline: dict[str, Any],
    constraint_matrix: dict[str, Any],
    shared_metric_evidence: dict[str, Any],
    network_blocked: bool,
) -> dict[str, Any]:
    current = chart_benchmark_runs[0] if chart_benchmark_runs else {}
    current_summary = current.get("summary", {})
    baseline_summary = committed_chart_baseline.get("summary", {})
    stable = len(chart_benchmark_runs) >= 2 and all(
        run == current for run in chart_benchmark_runs[1:]
    )

    current_precision = _nested_float(current_summary, "note_onset", "precision")
    baseline_precision = _nested_float(baseline_summary, "note_onset", "precision")
    current_unsupported = _float(current_summary.get("unsupported_note_ratio"))
    baseline_unsupported = _float(baseline_summary.get("unsupported_note_ratio"))
    current_strong = _nested_float(current_summary, "strong_onset_response", "recall")
    baseline_strong = _nested_float(baseline_summary, "strong_onset_response", "recall")
    current_downbeat = _nested_float(current_summary, "downbeat_response", "recall")
    baseline_downbeat = _nested_float(baseline_summary, "downbeat_response", "recall")
    current_silent = _int(current_summary.get("silent_violation_count"), default=-1)
    baseline_silent = _int(baseline_summary.get("silent_violation_count"), default=-1)

    course_summaries = current.get("course_summaries", {})
    course_notes = {
        course: _int(course_summaries.get(course, {}).get("note_count"), default=-1)
        for course, *_rest in COURSE_PROFILES
    }
    course_values = list(course_notes.values())

    checks = {
        "deterministic_offline_benchmark": {
            "passed": (
                stable
                and current.get("benchmark_version") == CHART_ALIGNMENT_BENCHMARK_VERSION
                and _float(current_summary.get("deterministic_rate")) == 1.0
                and network_blocked
            ),
            "run_count": len(chart_benchmark_runs),
            "runs_stable": stable,
            "deterministic_rate": current_summary.get("deterministic_rate"),
            "network_blocked": network_blocked,
        },
        "note_onset_alignment_improved": {
            "passed": current_precision > baseline_precision,
            "baseline_precision": baseline_precision,
            "current_precision": current_precision,
            "delta": round(current_precision - baseline_precision, 6),
        },
        "unsupported_note_ratio_reduced": {
            "passed": current_unsupported < baseline_unsupported,
            "baseline_ratio": baseline_unsupported,
            "current_ratio": current_unsupported,
            "delta": round(current_unsupported - baseline_unsupported, 6),
        },
        "strong_and_downbeat_response_preserved": {
            "passed": (
                current_strong >= baseline_strong
                and current_downbeat >= baseline_downbeat
            ),
            "baseline_strong_recall": baseline_strong,
            "current_strong_recall": current_strong,
            "baseline_downbeat_recall": baseline_downbeat,
            "current_downbeat_recall": current_downbeat,
        },
        "course_load_monotonic": {
            "passed": all(value >= 0 for value in course_values)
            and course_values == sorted(course_values),
            "course_note_counts": course_notes,
        },
        "silent_rest_sparse_not_regressed": {
            "passed": (
                bool(constraint_matrix.get("passed"))
                and current_silent >= 0
                and baseline_silent >= 0
                and current_silent <= baseline_silent
            ),
            "baseline_silent_violation_count": baseline_silent,
            "current_silent_violation_count": current_silent,
            "constraint_matrix_passed": constraint_matrix.get("passed"),
            "constraint_violation_count": constraint_matrix.get("violation_count"),
        },
        "resolution_and_load_constraints": {
            "passed": (
                bool(constraint_matrix.get("passed"))
                and int(constraint_matrix.get("configuration_count", 0)) == 360
                and int(constraint_matrix.get("bar_result_count", 0)) == 1080
            ),
            "configuration_count": constraint_matrix.get("configuration_count"),
            "bar_result_count": constraint_matrix.get("bar_result_count"),
            "violation_counts": constraint_matrix.get("violation_counts", {}),
        },
        "shared_rule_ai_alignment_metric": {
            "passed": (
                shared_metric_evidence.get("metric_version")
                == AI_SALIENCE_VALIDATION_VERSION
                and shared_metric_evidence.get("report_only") is True
                and shared_metric_evidence.get("rule_chart_evaluated_with_ai_metric") is True
                and not shared_metric_evidence.get("missing_fields")
            ),
            "metric_version": shared_metric_evidence.get("metric_version"),
            "report_only": shared_metric_evidence.get("report_only"),
            "missing_fields": shared_metric_evidence.get("missing_fields", []),
            "rule_chart_evaluated_with_ai_metric": shared_metric_evidence.get(
                "rule_chart_evaluated_with_ai_metric"
            ),
        },
    }
    return {
        "schema_version": PHASE_TWO_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_TWO_ACCEPTANCE_VERSION,
        "phase": "Phase 2",
        "passed": bool(current) and all(bool(check["passed"]) for check in checks.values()),
        "fixture_count": int(current.get("fixture_count", 0)),
        "chart_count": int(current.get("chart_count", 0)),
        "checks": checks,
        "baseline_summary": baseline_summary,
        "current_summary": current_summary,
        "course_summaries": course_summaries,
        "constraint_matrix": constraint_matrix,
        "shared_metric_evidence": shared_metric_evidence,
    }


def render_phase_two_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    alignment = checks["note_onset_alignment_improved"]
    unsupported = checks["unsupported_note_ratio_reduced"]
    response = checks["strong_and_downbeat_response_preserved"]
    constraints = checks["resolution_and_load_constraints"]
    lines = [
        "# Phase 2 acceptance report",
        "",
        f"- Acceptance: `{report['acceptance_version']}`",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Fixture count: {report['fixture_count']}",
        f"- Chart count: {report['chart_count']}",
        "",
        "## Exit conditions",
        "",
        "| Exit condition | Result | Evidence |",
        "| --- | :---: | --- |",
        _check_row(
            "Synthetic note/onset alignment improves over the Phase 2 baseline",
            alignment["passed"],
            f"{alignment['baseline_precision']:.6f} → {alignment['current_precision']:.6f} "
            f"({alignment['delta']:+.6f})",
        ),
        _check_row(
            "Unsupported-note ratio decreases",
            unsupported["passed"],
            f"{unsupported['baseline_ratio']:.6f} → {unsupported['current_ratio']:.6f} "
            f"({unsupported['delta']:+.6f})",
        ),
        _check_row(
            "Strong-onset and downbeat response do not regress",
            response["passed"],
            f"strong {response['baseline_strong_recall']:.6f} → "
            f"{response['current_strong_recall']:.6f}; downbeat "
            f"{response['baseline_downbeat_recall']:.6f} → "
            f"{response['current_downbeat_recall']:.6f}",
        ),
        _check_row(
            "Easy/Normal/Hard/Oni load remains monotonic",
            checks["course_load_monotonic"]["passed"],
            str(checks["course_load_monotonic"]["course_note_counts"]),
        ),
        _check_row(
            "Silent/rest/sparse behavior does not regress",
            checks["silent_rest_sparse_not_regressed"]["passed"],
            f"silent violations {checks['silent_rest_sparse_not_regressed']['baseline_silent_violation_count']} → "
            f"{checks['silent_rest_sparse_not_regressed']['current_silent_violation_count']}; "
            f"matrix violations {checks['silent_rest_sparse_not_regressed']['constraint_violation_count']}",
        ),
        _check_row(
            "16/24/48 and 12/18/36 preserve equivalent load and hard constraints",
            constraints["passed"],
            f"{constraints['configuration_count']} configurations, "
            f"{constraints['bar_result_count']} bar results",
        ),
        _check_row(
            "Rule and AI results use the same salience-alignment metric",
            checks["shared_rule_ai_alignment_metric"]["passed"],
            f"{checks['shared_rule_ai_alignment_metric']['metric_version']}, report-only",
        ),
        _check_row(
            "Acceptance is deterministic and offline",
            checks["deterministic_offline_benchmark"]["passed"],
            f"{checks['deterministic_offline_benchmark']['run_count']} blocked-network runs, "
            f"deterministic rate {checks['deterministic_offline_benchmark']['deterministic_rate']:.6f}",
        ),
        "",
        "## Aggregate comparison",
        "",
        "| Metric | Baseline | Current |",
        "| --- | ---: | ---: |",
        f"| Note/onset precision | {alignment['baseline_precision']:.6f} | {alignment['current_precision']:.6f} |",
        f"| Unsupported-note ratio | {unsupported['baseline_ratio']:.6f} | {unsupported['current_ratio']:.6f} |",
        f"| Strong-onset recall | {response['baseline_strong_recall']:.6f} | {response['current_strong_recall']:.6f} |",
        f"| Downbeat recall | {response['baseline_downbeat_recall']:.6f} | {response['current_downbeat_recall']:.6f} |",
        "",
        "## Constraint matrix",
        "",
        f"- Load configurations: {report['constraint_matrix']['load_configuration_count']}",
        f"- Silence/rest/sparse configurations: {report['constraint_matrix']['silence_configuration_count']}",
        f"- Total bar results: {report['constraint_matrix']['bar_result_count']}",
        f"- Violations: {report['constraint_matrix']['violation_count']}",
        "",
        "Phase 2 acceptance keeps AI salience diagnostics report-only. Repair thresholds remain deferred until the metrics are compared with listening results.",
        "",
    ]
    return "\n".join(lines)


def _boundary_bars(
    *,
    time_signature: str,
    canonical_grids: int,
    duration: float,
    beats: tuple[int, ...],
) -> list[BarFeature]:
    return [
        _bar(
            0,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.068,
            onset_grids=[0, canonical_grids - 1],
            rms_dbfs=-59.7,
            peak_rms_dbfs=-46.7,
            relative_rms_db=-53.5,
            sustained_activity_ratio=0.098,
            section="intro",
        ),
        _bar(
            1,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.9,
            onset_grids=list(range(0, canonical_grids, 3)),
            activity=0.8,
            beats=beats,
        ),
        _bar(
            2,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.01,
            section="break",
        ),
        _bar(
            3,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.02,
            onset_grids=[0, canonical_grids // 2],
            beats=beats,
        ),
        _bar(
            4,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.0,
            section="outro",
            phrase_position="song_end",
        ),
    ]


def _bar(
    index: int,
    *,
    time_signature: str,
    canonical_grids: int,
    duration: float,
    energy: float,
    onset_grids: list[int] | None = None,
    activity: float = 0.0,
    beats: tuple[int, ...] = (),
    section: str = "unknown",
    phrase_position: str = "unknown",
    rms_dbfs: float | None = None,
    peak_rms_dbfs: float | None = None,
    relative_rms_db: float | None = None,
    sustained_activity_ratio: float | None = None,
) -> BarFeature:
    onsets = onset_grids or []
    beat_set = set(beats)
    start_time = index * duration
    return BarFeature(
        index=index,
        start_time=start_time,
        end_time=start_time + duration,
        energy=energy,
        rms_dbfs=rms_dbfs,
        peak_rms_dbfs=peak_rms_dbfs,
        relative_rms_db=relative_rms_db,
        sustained_activity_ratio=sustained_activity_ratio,
        time_signature=time_signature,
        grids_per_bar=canonical_grids,
        onset_grids=onsets,
        activity_grids=[activity] * canonical_grids,
        grid_features=[
            GridFeature(
                grid=grid,
                onset=grid in onsets,
                beat=beats.index(grid) + 1 if grid in beat_set else None,
                downbeat=grid == 0 and grid in beat_set,
                strength=1.0 if grid in onsets else 0.0,
                activity=activity,
            )
            for grid in range(canonical_grids)
        ],
        beat_grids=list(beats),
        downbeat_grid=0 if beats else None,
        section=section,
        phrase_position=phrase_position,
    )


def _resolution_plan(
    canonical_grids: int,
    resolution: int,
    *,
    bar_count: int,
) -> ResolutionPlan:
    return ResolutionPlan(
        canonical_grids_per_bar=canonical_grids,
        base_resolution=resolution,
        bar_resolutions=[resolution] * bar_count,
    )


def _record(
    condition: bool,
    category: str,
    detail: str,
    violations: Counter[str],
    examples: list[str],
) -> None:
    if condition:
        return
    violations[category] += 1
    if len(examples) < MAX_VIOLATION_EXAMPLES:
        examples.append(f"{category}:{detail}")


def _nested_float(value: dict[str, Any], key: str, nested_key: str) -> float:
    nested = value.get(key, {})
    return _float(nested.get(nested_key) if isinstance(nested, dict) else None)


def _float(value: Any, *, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {'PASS' if passed else 'FAIL'} | {evidence} |"
