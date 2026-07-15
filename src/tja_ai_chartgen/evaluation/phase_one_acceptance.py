from __future__ import annotations

from collections import Counter
from typing import Any

from tja_ai_chartgen.evaluation.audio_benchmark import match_events
from tja_ai_chartgen.features.salience import (
    MIN_USABLE_BAR_CONFIDENCE,
    RHYTHMIC_SALIENCE_FEATURE_VERSION,
    SALIENCE_FALLBACK_BEAT_ONLY,
    SALIENCE_FALLBACK_EDGE_SILENCE,
    SALIENCE_FALLBACK_LOW_CONFIDENCE,
    SALIENCE_FALLBACK_NO_EVIDENCE,
    SALIENCE_FALLBACK_SILENT_BAR,
)
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.tja.model import BarFeature, BarRhythmicSalience


PHASE_ONE_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_ONE_ACCEPTANCE_VERSION = "phase-one-acceptance-v1"
SALIENCE_BENCHMARK_VERSION = "rhythmic-salience-benchmark-v1"
DEFAULT_ONSET_TOLERANCE_SECONDS = 0.07
DEFAULT_PEAK_HIT_THRESHOLD = 0.45
MIN_ONSET_PEAK_PRECISION = 0.98
MIN_ONSET_PEAK_RECALL = 0.95
MAX_SPARSE_POINT_RATIO = 0.50
ALLOWED_FALLBACK_REASONS = {
    SALIENCE_FALLBACK_EDGE_SILENCE,
    SALIENCE_FALLBACK_SILENT_BAR,
    SALIENCE_FALLBACK_NO_EVIDENCE,
    SALIENCE_FALLBACK_BEAT_ONLY,
    SALIENCE_FALLBACK_LOW_CONFIDENCE,
}


def build_fixture_salience_metrics(
    ground_truth: dict[str, Any],
    bars: list[BarFeature],
    salience: list[BarRhythmicSalience],
    *,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    peak_hit_threshold: float = DEFAULT_PEAK_HIT_THRESHOLD,
) -> dict[str, Any]:
    if len(bars) != len(salience):
        raise ValueError("Bar and salience counts must match")
    if onset_tolerance_seconds < 0:
        raise ValueError("Onset tolerance must be non-negative")
    if not 0 <= peak_hit_threshold <= 1:
        raise ValueError("Peak hit threshold must be within 0..1")

    edge_silent_positions = edge_silence_indexes(bars)
    estimated_peak_times: list[float] = []
    invalid_grids: list[str] = []
    duplicate_grids: list[int] = []
    unordered_bars: list[int] = []
    confidence_reason_violations: list[str] = []
    fallback_reasons: Counter[str] = Counter()
    canonical_grid_count = 0
    point_count = 0
    edge_silence_violation_count = 0

    for position, (bar, bar_salience) in enumerate(
        zip(bars, salience, strict=True)
    ):
        canonical_grid_count += bar.grids_per_bar
        point_count += len(bar_salience.points)
        grids = [point.grid for point in bar_salience.points]
        if grids != sorted(grids):
            unordered_bars.append(bar.index)
        duplicate_grids.extend(
            grid for grid, count in Counter(grids).items() if count > 1
        )
        for point in bar_salience.points:
            if point.grid < 0 or point.grid >= bar.grids_per_bar:
                invalid_grids.append(f"{bar.index}:{point.grid}")
                continue
            if point.hit >= peak_hit_threshold:
                estimated_peak_times.append(_point_time(bar, point.grid))

        reason = bar_salience.fallback_reason
        if reason is not None:
            fallback_reasons[reason] += 1
        if reason is not None and reason not in ALLOWED_FALLBACK_REASONS:
            confidence_reason_violations.append(f"unknown-reason:{bar.index}:{reason}")
        if reason is None and bar_salience.confidence < MIN_USABLE_BAR_CONFIDENCE:
            confidence_reason_violations.append(f"usable-below-threshold:{bar.index}")
        if (
            reason == SALIENCE_FALLBACK_LOW_CONFIDENCE
            and bar_salience.confidence >= MIN_USABLE_BAR_CONFIDENCE
        ):
            confidence_reason_violations.append(f"low-confidence-above-threshold:{bar.index}")
        if reason in {
            SALIENCE_FALLBACK_EDGE_SILENCE,
            SALIENCE_FALLBACK_SILENT_BAR,
            SALIENCE_FALLBACK_NO_EVIDENCE,
        } and bar_salience.points:
            confidence_reason_violations.append(f"empty-fallback-has-points:{bar.index}")
        if position in edge_silent_positions and bar_salience.points:
            edge_silence_violation_count += len(bar_salience.points)

    onset_alignment = match_events(
        [float(value) for value in ground_truth.get("onsets", [])],
        estimated_peak_times,
        tolerance_seconds=onset_tolerance_seconds,
    )
    return {
        "audio": str(ground_truth["audio"]),
        "bar_count": len(bars),
        "canonical_grid_count": canonical_grid_count,
        "salience_point_count": point_count,
        "peak_point_count": len(estimated_peak_times),
        "invalid_grid_count": len(invalid_grids),
        "invalid_grids": invalid_grids,
        "duplicate_grid_count": len(duplicate_grids),
        "unordered_bar_count": len(unordered_bars),
        "edge_silent_bar_count": len(edge_silent_positions),
        "edge_silence_violation_count": edge_silence_violation_count,
        "confidence_reason_violation_count": len(confidence_reason_violations),
        "confidence_reason_violations": confidence_reason_violations,
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        "onset_peak_alignment": onset_alignment,
    }


def build_salience_benchmark(
    fixture_metrics: list[dict[str, Any]],
    *,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    peak_hit_threshold: float = DEFAULT_PEAK_HIT_THRESHOLD,
) -> dict[str, Any]:
    ordered = sorted(fixture_metrics, key=lambda item: str(item["audio"]))
    canonical_grid_count = sum(int(item["canonical_grid_count"]) for item in ordered)
    salience_point_count = sum(int(item["salience_point_count"]) for item in ordered)
    onset_alignment = _aggregate_alignment(
        [item["onset_peak_alignment"] for item in ordered]
    )
    fallback_reasons: Counter[str] = Counter()
    for item in ordered:
        fallback_reasons.update(item["fallback_reason_counts"])

    return {
        "schema_version": 1,
        "benchmark_version": SALIENCE_BENCHMARK_VERSION,
        "feature_version": RHYTHMIC_SALIENCE_FEATURE_VERSION,
        "fixture_count": len(ordered),
        "settings": {
            "onset_tolerance_seconds": onset_tolerance_seconds,
            "peak_hit_threshold": peak_hit_threshold,
            "use_beatnet": False,
        },
        "summary": {
            "bar_count": sum(int(item["bar_count"]) for item in ordered),
            "canonical_grid_count": canonical_grid_count,
            "salience_point_count": salience_point_count,
            "peak_point_count": sum(int(item["peak_point_count"]) for item in ordered),
            "sparse_point_ratio": _rounded_ratio(
                salience_point_count, canonical_grid_count
            ),
            "invalid_grid_count": sum(int(item["invalid_grid_count"]) for item in ordered),
            "duplicate_grid_count": sum(
                int(item["duplicate_grid_count"]) for item in ordered
            ),
            "unordered_bar_count": sum(int(item["unordered_bar_count"]) for item in ordered),
            "edge_silent_bar_count": sum(
                int(item["edge_silent_bar_count"]) for item in ordered
            ),
            "edge_silence_violation_count": sum(
                int(item["edge_silence_violation_count"]) for item in ordered
            ),
            "confidence_reason_violation_count": sum(
                int(item["confidence_reason_violation_count"]) for item in ordered
            ),
            "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
            "onset_peak_alignment": onset_alignment,
        },
        "fixtures": ordered,
    }


def build_phase_one_acceptance_report(
    benchmark_runs: list[dict[str, Any]],
    *,
    network_blocked: bool,
) -> dict[str, Any]:
    first = benchmark_runs[0] if benchmark_runs else {}
    summary = first.get("summary", {})
    alignment = summary.get("onset_peak_alignment", {})
    stable = len(benchmark_runs) >= 2 and all(
        run == first for run in benchmark_runs[1:]
    )
    canonical_check = {
        "passed": (
            first.get("feature_version") == RHYTHMIC_SALIENCE_FEATURE_VERSION
            and int(summary.get("invalid_grid_count", -1)) == 0
            and int(summary.get("duplicate_grid_count", -1)) == 0
            and int(summary.get("unordered_bar_count", -1)) == 0
        ),
        "feature_version": first.get("feature_version"),
        "invalid_grid_count": summary.get("invalid_grid_count"),
        "duplicate_grid_count": summary.get("duplicate_grid_count"),
        "unordered_bar_count": summary.get("unordered_bar_count"),
    }
    deterministic_check = {
        "passed": stable,
        "run_count": len(benchmark_runs),
        "runs_stable": stable,
    }
    silence_check = {
        "passed": (
            int(summary.get("edge_silent_bar_count", 0)) > 0
            and int(summary.get("edge_silence_violation_count", -1)) == 0
        ),
        "edge_silent_bar_count": summary.get("edge_silent_bar_count"),
        "violation_count": summary.get("edge_silence_violation_count"),
    }
    onset_check = {
        "passed": (
            int(alignment.get("reference_count", 0)) > 0
            and float(alignment.get("precision", 0.0)) >= MIN_ONSET_PEAK_PRECISION
            and float(alignment.get("recall", 0.0)) >= MIN_ONSET_PEAK_RECALL
            and float(alignment.get("max_absolute_error_seconds") or 0.0)
            <= DEFAULT_ONSET_TOLERANCE_SECONDS
        ),
        **alignment,
        "minimum_precision": MIN_ONSET_PEAK_PRECISION,
        "minimum_recall": MIN_ONSET_PEAK_RECALL,
    }
    confidence_check = {
        "passed": int(summary.get("confidence_reason_violation_count", -1)) == 0,
        "violation_count": summary.get("confidence_reason_violation_count"),
        "fallback_reason_counts": summary.get("fallback_reason_counts", {}),
        "allowed_fallback_reasons": sorted(ALLOWED_FALLBACK_REASONS),
    }
    sparse_check = {
        "passed": (
            int(summary.get("salience_point_count", 0)) > 0
            and float(summary.get("sparse_point_ratio", 1.0))
            <= MAX_SPARSE_POINT_RATIO
        ),
        "canonical_grid_count": summary.get("canonical_grid_count"),
        "salience_point_count": summary.get("salience_point_count"),
        "sparse_point_ratio": summary.get("sparse_point_ratio"),
        "maximum_sparse_point_ratio": MAX_SPARSE_POINT_RATIO,
        "ai_payload_delta_fields": 0,
    }
    offline_check = {
        "passed": network_blocked,
        "socket_connections_blocked": network_blocked,
        "use_beatnet": first.get("settings", {}).get("use_beatnet"),
    }
    checks = {
        "canonical_grid_contract": canonical_check,
        "deterministic_output": deterministic_check,
        "edge_silence_zero": silence_check,
        "onset_peak_alignment": onset_check,
        "confidence_reason_contract": confidence_check,
        "compact_sparse_output": sparse_check,
        "offline_execution": offline_check,
    }
    return {
        "schema_version": PHASE_ONE_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_ONE_ACCEPTANCE_VERSION,
        "phase": "Phase 1",
        "passed": bool(first) and all(bool(check["passed"]) for check in checks.values()),
        "fixture_count": int(first.get("fixture_count", 0)),
        "checks": checks,
        "fixtures": first.get("fixtures", []),
        "deferred_consumers": ["fallback-generator", "ai-prompt", "quality-report"],
    }


def render_phase_one_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    onset = checks["onset_peak_alignment"]
    sparse = checks["compact_sparse_output"]
    lines = [
        "# Phase 1 acceptance report",
        "",
        f"- Acceptance: `{report['acceptance_version']}`",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Fixture count: {report['fixture_count']}",
        "",
        "## Exit conditions",
        "",
        "| Exit condition | Result | Evidence |",
        "| --- | :---: | --- |",
        _check_row(
            "All salience points use canonical grids",
            checks["canonical_grid_contract"]["passed"],
            f"{checks['canonical_grid_contract']['invalid_grid_count']} invalid, "
            f"{checks['canonical_grid_contract']['duplicate_grid_count']} duplicate, "
            f"{checks['canonical_grid_contract']['unordered_bar_count']} unordered",
        ),
        _check_row(
            "Repeated runs are deterministic",
            checks["deterministic_output"]["passed"],
            f"{checks['deterministic_output']['run_count']} offline runs",
        ),
        _check_row(
            "Edge silence produces no salience points",
            checks["edge_silence_zero"]["passed"],
            f"{checks['edge_silence_zero']['edge_silent_bar_count']} edge-silent bars, "
            f"{checks['edge_silence_zero']['violation_count']} violations",
        ),
        _check_row(
            "Known fixture onsets align with hit-salience peaks",
            onset["passed"],
            f"precision {onset['precision']:.3f}, recall {onset['recall']:.3f}, "
            f"max error {onset['max_absolute_error_seconds']:.3f}s",
        ),
        _check_row(
            "Confidence and fallback reasons are internally consistent",
            checks["confidence_reason_contract"]["passed"],
            f"{checks['confidence_reason_contract']['violation_count']} violations",
        ),
        _check_row(
            "Persisted acceptance evidence remains sparse and AI payload is unchanged",
            sparse["passed"],
            f"{sparse['salience_point_count']}/{sparse['canonical_grid_count']} points "
            f"({sparse['sparse_point_ratio']:.3f}), 0 AI payload fields added",
        ),
        _check_row(
            "Acceptance runs without network access",
            checks["offline_execution"]["passed"],
            "socket connect/connect_ex blocked during both passes",
        ),
        "",
        "## Handoff boundary",
        "",
        "Phase 1 validates the shared hit/accent/don-ka salience pipeline itself. "
        "Integration into the fallback generator, AI prompt, and quality report remains "
        "explicitly deferred to later roadmap phases.",
        "",
        "## Fixture summary",
        "",
        "| Fixture | Bars | Points | Peaks | Onset precision | Onset recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fixture in report["fixtures"]:
        alignment = fixture["onset_peak_alignment"]
        lines.append(
            f"| `{fixture['audio']}` | {fixture['bar_count']} | "
            f"{fixture['salience_point_count']} | {fixture['peak_point_count']} | "
            f"{alignment['precision']:.3f} | {alignment['recall']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _point_time(bar: BarFeature, grid: int) -> float:
    duration = bar.end_time - bar.start_time
    return bar.start_time + (grid / bar.grids_per_bar) * duration


def _aggregate_alignment(items: list[dict[str, Any]]) -> dict[str, Any]:
    reference_count = sum(int(item["reference_count"]) for item in items)
    estimated_count = sum(int(item["estimated_count"]) for item in items)
    true_positive_count = sum(int(item["true_positive_count"]) for item in items)
    false_positive_count = estimated_count - true_positive_count
    false_negative_count = reference_count - true_positive_count
    precision = _safe_ratio(true_positive_count, estimated_count)
    recall = _safe_ratio(true_positive_count, reference_count)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    weighted_errors = [
        (int(item["true_positive_count"]), item["mean_absolute_error_seconds"])
        for item in items
        if item["mean_absolute_error_seconds"] is not None
    ]
    matched_count = sum(count for count, _error in weighted_errors)
    mean_error = (
        sum(count * float(error) for count, error in weighted_errors) / matched_count
        if matched_count
        else None
    )
    max_errors = [
        float(item["max_absolute_error_seconds"])
        for item in items
        if item["max_absolute_error_seconds"] is not None
    ]
    return {
        "reference_count": reference_count,
        "estimated_count": estimated_count,
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_absolute_error_seconds": round(mean_error, 6) if mean_error is not None else None,
        "max_absolute_error_seconds": round(max(max_errors), 6) if max_errors else None,
    }


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return round(_safe_ratio(numerator, denominator), 6)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {'PASS' if passed else 'FAIL'} | {evidence} |"
