from __future__ import annotations

from collections import Counter
from typing import Any

from tja_ai_chartgen.evaluation.chart_alignment import CHART_ALIGNMENT_BENCHMARK_VERSION
from tja_ai_chartgen.features.salience import (
    build_burst_salience,
    is_reliable_burst,
    project_reliable_burst_span,
)
from tja_ai_chartgen.features.salience_candidates import (
    build_salience_candidate_bars,
    rank_accent_candidates,
)
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import (
    BarFeature,
    ChartBar,
    GridFeature,
    ResolutionPlan,
    SpectralGridFeature,
)
from tja_ai_chartgen.tja.quality import note_color_metrics

PHASE_THREE_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_THREE_ACCEPTANCE_VERSION = "phase-three-acceptance-v1"
COURSE_LEVELS = (("Easy", 3), ("Normal", 5), ("Hard", 7), ("Oni", 10))
METER_CASES = (
    ("4/4", 48, (16, 24, 48), 2.0, (0, 12, 24, 36)),
    ("3/4", 36, (12, 18, 36), 1.5, (0, 12, 24)),
    ("6/8", 36, (12, 18, 36), 1.5, (0, 18)),
)
MAX_EXAMPLES = 32


def build_phase_three_behavior_matrix() -> dict[str, Any]:
    accent = _build_accent_metrics()
    color = _build_color_metrics()
    special = _build_special_note_matrix()
    return {
        "schema_version": 1,
        "matrix_version": "phase-three-behavior-matrix-v1",
        "passed": (
            accent["accent_response_rate"] >= 0.9
            and accent["big_note_count"] > 0
            and accent["adjacent_big_note_violation_count"] == 0
            and accent["cadence_response_count"] > 0
            and color["low_attack_don_response_rate"] >= 0.75
            and color["high_attack_ka_response_rate"] >= 0.75
            and 0.1 <= color["ka_ratio"] <= 0.9
            and color["longest_monochrome_run"] <= 4
            and special["passed"]
        ),
        "accent": accent,
        "color": color,
        "special_notes": special,
    }


def build_fill_burst_fixture_evidence(
    ground_truth: dict[str, Any],
    bars: list[BarFeature],
    burst_salience: list[Any],
    chart_bars: list[ChartBar],
) -> dict[str, Any]:
    expected_indexes = sorted(
        {
            position
            for start, end in ground_truth.get("fill_ranges", [])
            for position, bar in enumerate(bars)
            if bar.start_time <= (float(start) + float(end)) / 2 < bar.end_time
        }
    )
    reliable_indexes = [
        position
        for position, salience in enumerate(burst_salience)
        if is_reliable_burst(salience)
    ]
    special_indexes = [
        position
        for position, chart_bar in enumerate(chart_bars)
        if any(note in "57" for note in chart_bar.notes)
    ]
    late_range_count = sum(
        salience.burst_start_grid is not None
        and salience.burst_start_grid >= bars[position].grids_per_bar // 2
        and salience.burst_end_grid is not None
        and salience.burst_end_grid > salience.burst_start_grid
        for position, salience in enumerate(burst_salience)
        if position in reliable_indexes
    )
    durations = [
        _special_note_duration(chart_bars[position], bars[position])
        for position in special_indexes
    ]
    return {
        "fixture": ground_truth.get("audio"),
        "expected_burst_indexes": expected_indexes,
        "reliable_burst_indexes": reliable_indexes,
        "special_note_indexes": special_indexes,
        "late_range_count": late_range_count,
        "special_note_durations_seconds": [round(value, 6) for value in durations],
        "passed": (
            bool(expected_indexes)
            and reliable_indexes == expected_indexes
            and special_indexes == expected_indexes
            and late_range_count == len(expected_indexes)
            and all(value >= 0.25 for value in durations)
        ),
    }


def build_phase_three_acceptance_report(
    chart_benchmark_runs: list[dict[str, Any]],
    behavior_runs: list[dict[str, Any]],
    fill_fixture_runs: list[dict[str, Any]],
    *,
    phase_two_acceptance: dict[str, Any],
    network_blocked: bool,
) -> dict[str, Any]:
    current_chart = chart_benchmark_runs[0] if chart_benchmark_runs else {}
    current_behavior = behavior_runs[0] if behavior_runs else {}
    current_fill = fill_fixture_runs[0] if fill_fixture_runs else {}
    current_summary = current_chart.get("summary", {})
    phase_two_summary = phase_two_acceptance.get("current_summary", {})
    charts_stable = _runs_stable(chart_benchmark_runs)
    behavior_stable = _runs_stable(behavior_runs)
    fill_stable = _runs_stable(fill_fixture_runs)

    current_strong = _nested_float(current_summary, "strong_onset_response", "recall")
    baseline_strong = _nested_float(
        phase_two_summary, "strong_onset_response", "recall"
    )
    current_downbeat = _nested_float(current_summary, "downbeat_response", "recall")
    baseline_downbeat = _nested_float(
        phase_two_summary, "downbeat_response", "recall"
    )
    accent = current_behavior.get("accent", {})
    color = current_behavior.get("color", {})
    special = current_behavior.get("special_notes", {})

    checks = {
        "deterministic_offline_acceptance": {
            "passed": (
                network_blocked
                and charts_stable
                and behavior_stable
                and fill_stable
                and current_chart.get("benchmark_version")
                == CHART_ALIGNMENT_BENCHMARK_VERSION
                and _float(current_summary.get("deterministic_rate")) == 1.0
            ),
            "network_blocked": network_blocked,
            "chart_runs_stable": charts_stable,
            "behavior_runs_stable": behavior_stable,
            "fill_runs_stable": fill_stable,
            "run_count": min(
                len(chart_benchmark_runs), len(behavior_runs), len(fill_fixture_runs)
            ),
        },
        "strong_downbeat_and_cadence_response": {
            "passed": (
                current_strong >= baseline_strong
                and current_downbeat >= baseline_downbeat
                and _float(accent.get("accent_response_rate")) >= 0.9
                and _int(accent.get("cadence_response_count")) > 0
            ),
            "baseline_strong_recall": baseline_strong,
            "current_strong_recall": current_strong,
            "baseline_downbeat_recall": baseline_downbeat,
            "current_downbeat_recall": current_downbeat,
            "accent_response_rate": accent.get("accent_response_rate"),
            "cadence_response_count": accent.get("cadence_response_count"),
        },
        "accent_playability": {
            "passed": (
                _int(accent.get("big_note_count")) > 0
                and _int(accent.get("adjacent_big_note_violation_count")) == 0
                and not accent.get("course_limit_violations")
            ),
            "big_note_count": accent.get("big_note_count"),
            "adjacent_big_note_violation_count": accent.get(
                "adjacent_big_note_violation_count"
            ),
            "course_limit_violations": accent.get("course_limit_violations", []),
        },
        "don_ka_balance": {
            "passed": (
                _float(color.get("low_attack_don_response_rate")) >= 0.75
                and _float(color.get("high_attack_ka_response_rate")) >= 0.75
                and 0.1 <= _float(color.get("ka_ratio")) <= 0.9
                and _int(color.get("longest_monochrome_run"), default=999) <= 4
            ),
            **color,
        },
        "fill_burst_fixture_alignment": {
            "passed": bool(current_fill.get("passed")),
            **current_fill,
        },
        "no_mechanical_phrase_end_fill": {
            "passed": _int(special.get("no_burst_special_note_count"), default=-1)
            == 0,
            "no_burst_special_note_count": special.get(
                "no_burst_special_note_count"
            ),
        },
        "special_note_explainability": {
            "passed": (
                bool(special.get("passed"))
                and _int(special.get("violation_count"), default=-1) == 0
                and _int(special.get("drumroll_count")) > 0
                and _int(special.get("balloon_count")) > 0
                and all(
                    _float(value) >= 0.25
                    for value in current_fill.get(
                        "special_note_durations_seconds", []
                    )
                )
            ),
            "configuration_count": special.get("configuration_count"),
            "drumroll_count": special.get("drumroll_count"),
            "balloon_count": special.get("balloon_count"),
            "minimum_duration_seconds": special.get("minimum_duration_seconds"),
            "violation_count": special.get("violation_count"),
            "violation_examples": special.get("violation_examples", []),
        },
    }
    return {
        "schema_version": PHASE_THREE_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_THREE_ACCEPTANCE_VERSION,
        "phase": "Phase 3",
        "passed": bool(current_chart)
        and bool(current_behavior)
        and bool(current_fill)
        and all(bool(check["passed"]) for check in checks.values()),
        "fixture_count": int(current_chart.get("fixture_count", 0)),
        "chart_count": int(current_chart.get("chart_count", 0)),
        "checks": checks,
        "phase_two_summary": phase_two_summary,
        "current_chart_summary": current_summary,
        "behavior_matrix": current_behavior,
        "fill_fixture_evidence": current_fill,
    }


def render_phase_three_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    response = checks["strong_downbeat_and_cadence_response"]
    color = checks["don_ka_balance"]
    special = checks["special_note_explainability"]
    lines = [
        "# Phase 3 acceptance report",
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
            "Strong-onset, downbeat, and cadence response is preserved or improved",
            response["passed"],
            f"strong {response['baseline_strong_recall']:.6f} → {response['current_strong_recall']:.6f}; "
            f"downbeat {response['baseline_downbeat_recall']:.6f} → {response['current_downbeat_recall']:.6f}; "
            f"accent response {response['accent_response_rate']:.6f}",
        ),
        _check_row(
            "Big-note accents remain sparse and playable",
            checks["accent_playability"]["passed"],
            f"{checks['accent_playability']['big_note_count']} big notes, "
            f"{checks['accent_playability']['adjacent_big_note_violation_count']} adjacency violations",
        ),
        _check_row(
            "Don/ka response and monochrome-run limits do not regress",
            color["passed"],
            f"low→don {color['low_attack_don_response_rate']:.6f}, "
            f"high→ka {color['high_attack_ka_response_rate']:.6f}, "
            f"ka ratio {color['ka_ratio']:.6f}, longest run {color['longest_monochrome_run']}",
        ),
        _check_row(
            "Known fill-burst fixture hits only the expected late-bar bursts",
            checks["fill_burst_fixture_alignment"]["passed"],
            f"expected {checks['fill_burst_fixture_alignment']['expected_burst_indexes']}, "
            f"detected {checks['fill_burst_fixture_alignment']['reliable_burst_indexes']}, "
            f"special {checks['fill_burst_fixture_alignment']['special_note_indexes']}",
        ),
        _check_row(
            "Phrase ends without burst evidence do not create mechanical fills",
            checks["no_mechanical_phrase_end_fill"]["passed"],
            f"{checks['no_mechanical_phrase_end_fill']['no_burst_special_note_count']} special notes",
        ),
        _check_row(
            "Special-note count, duration, range, and balloon load remain explainable",
            special["passed"],
            f"{special['configuration_count']} configurations, {special['drumroll_count']} rolls, "
            f"{special['balloon_count']} balloons, minimum {special['minimum_duration_seconds']:.6f}s, "
            f"{special['violation_count']} violations",
        ),
        _check_row(
            "Acceptance is deterministic and offline",
            checks["deterministic_offline_acceptance"]["passed"],
            f"{checks['deterministic_offline_acceptance']['run_count']} blocked-network runs",
        ),
        "",
        "## Behavior matrix",
        "",
        f"- Accent candidates: {report['behavior_matrix']['accent']['accent_candidate_count']}",
        f"- Color evidence points: {report['behavior_matrix']['color']['evidence_note_count']}",
        f"- Special-note configurations: {report['behavior_matrix']['special_notes']['configuration_count']}",
        f"- Special-note violations: {report['behavior_matrix']['special_notes']['violation_count']}",
        "",
        "Phase 3 acceptance keeps semantic accent, color, burst, and special-note checks deterministic and offline. TJA preflight remains the final structural legality boundary.",
        "",
    ]
    return "\n".join(lines)


def _build_accent_metrics() -> dict[str, Any]:
    bars = [
        _bar(0, onset_grids=[0, 12, 24, 36], strong_grids={0}, beats=(0, 12, 24, 36)),
        _bar(1, onset_grids=[0, 6, 24, 36], strong_grids={6}, beats=(0, 12, 24, 36)),
        _bar(
            2,
            onset_grids=[0, 12, 30, 42],
            strong_grids={30},
            beats=(0, 12, 24, 36),
            phrase_position="phrase_end",
            transition_role="cadence",
            transition_confidence=1.0,
        ),
    ]
    candidates = build_salience_candidate_bars(bars)
    accent_grids = [
        {candidate.grid for candidate in rank_accent_candidates(bar_candidates)}
        for bar_candidates in candidates
    ]
    accent_candidate_count = sum(len(grids) for grids in accent_grids)
    accent_hit_count = 0
    big_note_count = 0
    cadence_response_count = 0
    adjacent_violations = 0
    course_limit_violations: list[str] = []
    for course, level in COURSE_LEVELS:
        chart_bars = generate_fallback_chart_bars(
            bars,
            style="performance",
            density="low",
            course=course,
            level=level,
        )
        course_big_count = 0
        previous_ended_with_big = False
        for position, chart_bar in enumerate(chart_bars):
            if previous_ended_with_big and chart_bar.notes.startswith(("3", "4")):
                adjacent_violations += 1
            step = bars[position].grids_per_bar // len(chart_bar.notes)
            for grid in accent_grids[position]:
                note = chart_bar.notes[grid // step]
                accent_hit_count += note in "1234"
                if position == 2 and note in "1234":
                    cadence_response_count += 1
            big_positions = [
                index for index, note in enumerate(chart_bar.notes) if note in "34"
            ]
            big_note_count += len(big_positions)
            course_big_count += len(big_positions)
            adjacent_violations += sum(
                current - previous == 1
                for previous, current in zip(big_positions, big_positions[1:])
            )
            previous_ended_with_big = chart_bar.notes.endswith(("3", "4"))
        limit = len(bars) * (1 if course in {"Easy", "Normal"} else 2)
        if course_big_count > limit:
            course_limit_violations.append(f"{course}:{course_big_count}>{limit}")
    response_denominator = accent_candidate_count * len(COURSE_LEVELS)
    return {
        "accent_candidate_count": accent_candidate_count,
        "accent_hit_count": accent_hit_count,
        "accent_response_rate": _rounded_ratio(accent_hit_count, response_denominator),
        "big_note_count": big_note_count,
        "cadence_response_count": cadence_response_count,
        "adjacent_big_note_violation_count": adjacent_violations,
        "course_limit_violations": course_limit_violations,
    }


def _build_color_metrics() -> dict[str, Any]:
    bars: list[BarFeature] = []
    low_grids: list[set[int]] = []
    high_grids: list[set[int]] = []
    onset_grids = list(range(0, 48, 6))
    for index in range(8):
        low = {grid for offset, grid in enumerate(onset_grids) if offset % 2 == index % 2}
        high = set(onset_grids) - low
        spectral = [
            SpectralGridFeature(
                grid=grid,
                low_onset_strength=0.9 if grid in low else 0.0,
                high_onset_strength=0.9 if grid in high else 0.0,
                spectral_flux=0.9,
            )
            for grid in onset_grids
        ]
        bars.append(
            _bar(
                index,
                onset_grids=onset_grids,
                strong_grids=set(onset_grids),
                beats=(0, 12, 24, 36),
                spectral=spectral,
                percussive_ratio=0.8,
                brightness=0.8,
            )
        )
        low_grids.append(low)
        high_grids.append(high)

    chart_bars = generate_fallback_chart_bars(
        bars,
        style="hybrid",
        density="high",
        course="Oni",
        level=10,
    )
    low_notes = low_don = high_notes = high_ka = 0
    for position, chart_bar in enumerate(chart_bars):
        step = bars[position].grids_per_bar // len(chart_bar.notes)
        for grid in low_grids[position]:
            note = chart_bar.notes[grid // step]
            low_notes += note in "12"
            low_don += note == "1"
        for grid in high_grids[position]:
            note = chart_bar.notes[grid // step]
            high_notes += note in "12"
            high_ka += note == "2"
    don_count, ka_count, longest_run = note_color_metrics(chart_bars)
    return {
        "evidence_note_count": low_notes + high_notes,
        "low_attack_note_count": low_notes,
        "low_attack_don_count": low_don,
        "low_attack_don_response_rate": _rounded_ratio(low_don, low_notes),
        "high_attack_note_count": high_notes,
        "high_attack_ka_count": high_ka,
        "high_attack_ka_response_rate": _rounded_ratio(high_ka, high_notes),
        "don_count": don_count,
        "ka_count": ka_count,
        "ka_ratio": _rounded_ratio(ka_count, don_count + ka_count),
        "longest_monochrome_run": longest_run,
    }


def _build_special_note_matrix() -> dict[str, Any]:
    violations: Counter[str] = Counter()
    examples: list[str] = []
    configuration_count = 0
    drumroll_count = 0
    balloon_count = 0
    no_burst_special_note_count = 0
    durations: list[float] = []

    for time_signature, grids, resolutions, duration, beats in METER_CASES:
        midpoint = grids // 2
        burst_onsets = sorted({0, grids // 4, *range(midpoint, grids, max(1, grids // 12))})
        roll_bar = _bar(
            0,
            time_signature=time_signature,
            grids=grids,
            duration=duration,
            energy=0.8,
            onset_grids=burst_onsets,
            strong_grids=set(burst_onsets),
            beats=beats,
            phrase_position="phrase_end",
            fill_candidate=True,
        )
        balloon_bar = roll_bar.model_copy(
            update={"energy": 0.95, "phrase_position": "song_end"}
        )
        no_burst_bar = _bar(
            0,
            time_signature=time_signature,
            grids=grids,
            duration=duration,
            energy=0.95,
            onset_grids=list(beats),
            strong_grids=set(beats),
            beats=beats,
            phrase_position="phrase_end",
            transition_role="cadence",
            transition_confidence=1.0,
            fill_candidate=True,
        )
        for resolution in resolutions:
            plan = ResolutionPlan(
                canonical_grids_per_bar=grids,
                base_resolution=resolution,
                bar_resolutions=[resolution],
            )
            for course, level in COURSE_LEVELS:
                prefix = f"{time_signature}:{resolution}:{course}"
                for kind, bar in (("roll", roll_bar), ("balloon", balloon_bar)):
                    chart_bar = generate_fallback_chart_bars(
                        [bar],
                        density="high",
                        special_notes=True,
                        course=course,
                        level=level,
                        resolution_plan=plan,
                    )[0]
                    configuration_count += 1
                    starts = [
                        index for index, note in enumerate(chart_bar.notes) if note in "57"
                    ]
                    ends = [
                        index for index, note in enumerate(chart_bar.notes) if note == "8"
                    ]
                    _record(
                        len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0],
                        "single-closed-special-note",
                        f"{prefix}:{kind}:{chart_bar.notes}",
                        violations,
                        examples,
                    )
                    if not starts or not ends:
                        continue
                    special_kind = chart_bar.notes[starts[0]]
                    drumroll_count += special_kind == "5"
                    balloon_count += special_kind == "7"
                    if kind == "roll":
                        _record(
                            special_kind == "5",
                            "roll-kind",
                            f"{prefix}:{chart_bar.notes}",
                            violations,
                            examples,
                        )
                    else:
                        _record(
                            special_kind == "7" and len(chart_bar.balloon_counts) == 1,
                            "balloon-kind-and-count",
                            f"{prefix}:{chart_bar.notes}:{chart_bar.balloon_counts}",
                            violations,
                            examples,
                        )
                    special_duration = _special_note_duration(chart_bar, bar)
                    durations.append(special_duration)
                    _record(
                        special_duration >= 0.25,
                        "minimum-duration",
                        f"{prefix}:{special_duration}",
                        violations,
                        examples,
                    )
                    salience = build_burst_salience([bar])[0]
                    span = project_reliable_burst_span(
                        bar,
                        salience,
                        output_resolution=resolution,
                    )
                    output_step = grids // resolution
                    _record(
                        span is not None
                        and starts[0] * output_step == span[0]
                        and ends[0] * output_step == span[1],
                        "burst-range-alignment",
                        f"{prefix}:{span}:{starts}:{ends}",
                        violations,
                        examples,
                    )

                no_burst_chart = generate_fallback_chart_bars(
                    [no_burst_bar],
                    density="high",
                    special_notes=True,
                    course=course,
                    level=level,
                    resolution_plan=plan,
                )[0]
                no_burst_special_note_count += sum(
                    note in "57" for note in no_burst_chart.notes
                )

    violation_counts = dict(sorted(violations.items()))
    return {
        "passed": not violation_counts and no_burst_special_note_count == 0,
        "configuration_count": configuration_count,
        "meter_count": len(METER_CASES),
        "resolution_count": sum(len(item[2]) for item in METER_CASES),
        "course_count": len(COURSE_LEVELS),
        "drumroll_count": drumroll_count,
        "balloon_count": balloon_count,
        "no_burst_special_note_count": no_burst_special_note_count,
        "minimum_duration_seconds": round(min(durations), 6) if durations else 0.0,
        "maximum_duration_seconds": round(max(durations), 6) if durations else 0.0,
        "violation_count": sum(violation_counts.values()),
        "violation_counts": violation_counts,
        "violation_examples": examples,
    }


def _bar(
    index: int,
    *,
    time_signature: str = "4/4",
    grids: int = 48,
    duration: float = 2.0,
    energy: float = 0.8,
    onset_grids: list[int],
    strong_grids: set[int],
    beats: tuple[int, ...],
    spectral: list[SpectralGridFeature] | None = None,
    percussive_ratio: float = 0.0,
    brightness: float = 0.0,
    phrase_position: str = "phrase_middle",
    transition_role: str = "stable",
    transition_confidence: float = 0.0,
    fill_candidate: bool = False,
) -> BarFeature:
    beat_numbers = {grid: position for position, grid in enumerate(beats, start=1)}
    return BarFeature(
        index=index,
        start_time=index * duration,
        end_time=(index + 1) * duration,
        energy=energy,
        time_signature=time_signature,
        grids_per_bar=grids,
        onset_grids=onset_grids,
        accent_grids=sorted(strong_grids),
        grid_features=[
            GridFeature(
                grid=grid,
                onset=grid in onset_grids,
                accent=grid in strong_grids,
                strength=1.0 if grid in strong_grids else 0.7 if grid in onset_grids else 0.0,
                beat=beat_numbers.get(grid),
                downbeat=grid == beats[0] if beats else False,
                activity=0.8,
            )
            for grid in range(grids)
        ],
        spectral_grid_features=spectral or [],
        percussive_ratio=percussive_ratio,
        brightness=brightness,
        beat_grids=list(beats),
        downbeat_grid=beats[0] if beats else None,
        phrase_position=phrase_position,
        transition_role=transition_role,
        transition_confidence=transition_confidence,
        fill_candidate=fill_candidate,
    )


def _special_note_duration(chart_bar: ChartBar, feature_bar: BarFeature) -> float:
    starts = [index for index, note in enumerate(chart_bar.notes) if note in "57"]
    ends = [index for index, note in enumerate(chart_bar.notes) if note == "8"]
    if not starts or not ends or ends[0] <= starts[0] or not chart_bar.notes:
        return 0.0
    return (
        (feature_bar.end_time - feature_bar.start_time)
        * (ends[0] - starts[0])
        / len(chart_bar.notes)
    )


def _record(
    passed: bool,
    code: str,
    detail: str,
    violations: Counter[str],
    examples: list[str],
) -> None:
    if passed:
        return
    violations[code] += 1
    if len(examples) < MAX_EXAMPLES:
        examples.append(f"{code}:{detail}")


def _runs_stable(runs: list[dict[str, Any]]) -> bool:
    return len(runs) >= 2 and all(run == runs[0] for run in runs[1:])


def _nested_float(value: dict[str, Any], *keys: str) -> float:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return _float(current)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator if denominator else 0.0, 6)


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {'PASS' if passed else 'FAIL'} | {evidence} |"
