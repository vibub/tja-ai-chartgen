from collections import Counter
from dataclasses import dataclass

from tja_ai_chartgen.features.density import build_density_hints
from tja_ai_chartgen.features.salience import build_don_ka_salience
from tja_ai_chartgen.features.salience_candidates import (
    SalienceCandidate,
    build_salience_candidate_bars,
)
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    ChartBar,
    SongAnalysis,
)

RHYTHM_SKELETON_VERSION = "rhythm-skeleton-v4"
LATTICE_MIXED_MIN_GAIN = 0.25
LATTICE_MIXED_MIN_UNIQUE = 0.18
LATTICE_EXCEPTION_MIN_CONFIDENCE = 0.95
LATTICE_EXCEPTION_MIN_REPETITIONS = 3
LATTICE_EXCEPTION_MAX_RATE = 0.10
AI_SKELETON_MIN_BARS = 16
AI_SKELETON_MIN_HITS = 32
AI_SKELETON_MIN_COVERAGE = 0.85
AI_SKELETON_MAX_EXTRA_RATE = 0.15


@dataclass(frozen=True)
class RhythmLattice:
    kind: str
    steps: tuple[tuple[int, int], ...]
    coverage: float

    def contains(self, tick: int) -> bool:
        return any(tick % step == phase for step, phase in self.steps)


@dataclass(frozen=True)
class RhythmSkeletonComparison:
    expected_count: int
    actual_count: int
    matched_count: int
    missing: tuple[tuple[int, int], ...]
    extras: tuple[tuple[int, int], ...]

    @property
    def coverage(self) -> float:
        if self.expected_count <= 0:
            return 1.0
        return self.matched_count / self.expected_count

    @property
    def extra_rate(self) -> float:
        if self.actual_count <= 0:
            return 0.0
        return len(self.extras) / self.actual_count


def build_rhythm_skeleton(
    analysis: SongAnalysis,
    *,
    course: str,
    level: int,
    style: str,
    density: str,
) -> list[list[int]]:
    """用规则生成器确定 AI 不应随意改写的普通击打时间骨架。"""
    salience_bars = build_don_ka_salience(analysis.bars)
    skeleton_bars = _build_skeleton_chart_bars(
        analysis,
        course=course,
        level=level,
        style=style,
        density=density,
        precomputed_salience=salience_bars,
    )
    raw_skeleton = [
        sorted(_regular_hit_ticks(chart_bar, feature_bar))
        for chart_bar, feature_bar in zip(
            skeleton_bars,
            analysis.bars,
            strict=True,
        )
    ]
    candidate_bars = build_salience_candidate_bars(
        analysis.bars,
        resolution_plan=analysis.resolution_plan,
        salience_bars=salience_bars,
    )
    return _infer_musical_skeleton(
        raw_skeleton,
        analysis.bars,
        candidate_bars,
    )


def conform_chart_to_rhythm_skeleton(
    chart_bars: list[ChartBar],
    analysis: SongAnalysis,
    *,
    course: str,
    level: int,
    style: str,
    density: str,
    skeleton: list[list[int]] | None = None,
) -> list[ChartBar]:
    """固定 AI 普通击打时间，并保留其配色和已校验的长音。"""
    resolved_skeleton = skeleton or build_rhythm_skeleton(
        analysis,
        course=course,
        level=level,
        style=style,
        density=density,
    )
    plan = analysis.resolution_plan
    selected_resolutions = (
        plan.bar_resolutions[: len(chart_bars)] or [plan.base_resolution]
        if plan is not None
        else []
    )
    uses_highest_resolution = bool(
        plan is not None
        and any(
            resolution == plan.canonical_grids_per_bar
            for resolution in selected_resolutions
        )
    )
    if not uses_highest_resolution and (
        len(chart_bars) < AI_SKELETON_MIN_BARS
        or sum(map(len, resolved_skeleton)) < AI_SKELETON_MIN_HITS
    ):
        return chart_bars
    conformed: list[ChartBar] = []
    for chart_bar, feature_bar, required_ticks in zip(
        chart_bars,
        analysis.bars,
        resolved_skeleton,
        strict=True,
    ):
        notes = ["0"] * len(chart_bar.notes)
        long_positions = _copy_long_notes(chart_bar, notes)
        blocked_ticks = {
            _canonical_tick(position, len(notes), feature_bar.grids_per_bar)
            for position in long_positions
        }
        ai_hits = _regular_hits_with_notes(chart_bar, feature_bar)
        for sequence_index, tick in enumerate(required_ticks):
            if tick in blocked_ticks:
                continue
            position = round(tick / feature_bar.grids_per_bar * len(notes))
            position = min(len(notes) - 1, max(0, position))
            source_tick, note = _nearest_hit(ai_hits, tick) or (
                tick,
                "1" if (feature_bar.index + sequence_index) % 2 == 0 else "2",
            )
            if note in "34" and source_tick != tick:
                note = "1" if note == "3" else "2"
            notes[position] = note
        conformed.append(
            chart_bar.model_copy(
                update={
                    "notes": "".join(notes),
                    "balloon_counts": chart_bar.balloon_counts,
                }
            )
        )
    return conformed


def compare_chart_to_rhythm_skeleton(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    skeleton: list[list[int]],
) -> RhythmSkeletonComparison:
    expected_count = 0
    actual_count = 0
    matched_count = 0
    missing: list[tuple[int, int]] = []
    extras: list[tuple[int, int]] = []

    for position, (chart_bar, feature_bar, expected_ticks) in enumerate(
        zip(chart_bars, feature_bars, skeleton, strict=True)
    ):
        actual_ticks = _regular_hit_ticks(chart_bar, feature_bar)
        long_spans = _long_note_spans(chart_bar, feature_bar)
        expected = set(expected_ticks)
        expected_count += len(expected)
        actual_count += len(actual_ticks)
        matched = expected & actual_ticks
        matched_count += len(matched)

        for tick in sorted(expected - actual_ticks):
            if any(start <= tick <= end for start, end in long_spans):
                matched_count += 1
            else:
                missing.append((position, tick))
        extras.extend((position, tick) for tick in sorted(actual_ticks - expected))

    return RhythmSkeletonComparison(
        expected_count=expected_count,
        actual_count=actual_count,
        matched_count=matched_count,
        missing=tuple(missing),
        extras=tuple(extras),
    )


def rhythm_skeleton_issues(
    chart_bars: list[ChartBar],
    analysis: SongAnalysis,
    *,
    course: str,
    level: int,
    style: str,
    density: str,
    skeleton: list[list[int]] | None = None,
) -> list[str]:
    if len(chart_bars) < AI_SKELETON_MIN_BARS:
        return []
    resolved_skeleton = skeleton or build_rhythm_skeleton(
        analysis,
        course=course,
        level=level,
        style=style,
        density=density,
    )
    comparison = compare_chart_to_rhythm_skeleton(
        chart_bars,
        analysis.bars[: len(chart_bars)],
        resolved_skeleton[: len(chart_bars)],
    )
    if comparison.expected_count < AI_SKELETON_MIN_HITS:
        return []

    issues: list[str] = []
    if comparison.coverage < AI_SKELETON_MIN_COVERAGE:
        examples = ", ".join(
            f"bar {position + 1} tick {tick}"
            for position, tick in comparison.missing[:8]
        )
        issues.append(
            "chart misses too much of the deterministic rhythm skeleton: "
            f"{comparison.matched_count}/{comparison.expected_count} "
            f"({comparison.coverage:.1%}), minimum {AI_SKELETON_MIN_COVERAGE:.0%}; "
            f"restore required timing positions ({examples})"
        )
    if comparison.extra_rate > AI_SKELETON_MAX_EXTRA_RATE:
        examples = ", ".join(
            f"bar {position + 1} tick {tick}"
            for position, tick in comparison.extras[:8]
        )
        issues.append(
            "chart adds too many hits outside the deterministic rhythm skeleton: "
            f"{len(comparison.extras)}/{comparison.actual_count} "
            f"({comparison.extra_rate:.1%}), maximum {AI_SKELETON_MAX_EXTRA_RATE:.0%}; "
            f"remove unsupported timing changes ({examples})"
        )
    return issues


def _infer_musical_skeleton(
    raw_skeleton: list[list[int]],
    bars: list[BarFeature],
    candidate_bars: list[list[SalienceCandidate]],
) -> list[list[int]]:
    resolved: list[list[int]] = [[] for _ in raw_skeleton]
    candidate_maps = [
        {candidate.grid: candidate for candidate in candidates}
        for candidates in candidate_bars
    ]
    for start, end in _phrase_ranges(bars):
        lattice = _select_phrase_lattice(
            raw_skeleton,
            candidate_maps,
            start=start,
            end=end,
        )
        tick_counts = Counter(
            tick
            for position in range(start, end)
            for tick in raw_skeleton[position]
        )
        exception_groups: dict[int, list[tuple[int, SalienceCandidate]]] = {}
        base_count = 0
        for position in range(start, end):
            for tick in raw_skeleton[position]:
                candidate = candidate_maps[position].get(tick)
                if lattice.contains(tick):
                    resolved[position].append(tick)
                    base_count += 1
                elif _keep_lattice_exception(
                    candidate,
                    exact_recurrence=tick_counts[tick],
                ):
                    exception_groups.setdefault(tick, []).append((position, candidate))

        exception_budget = max(1, round(base_count * LATTICE_EXCEPTION_MAX_RATE))
        used_exceptions = 0
        ranked_groups = sorted(
            exception_groups.items(),
            key=lambda item: (
                -sum(_lattice_weight(candidate) for _position, candidate in item[1]),
                -len(item[1]),
                item[0],
            ),
        )
        for tick, group in ranked_groups:
            if used_exceptions + len(group) > exception_budget:
                continue
            for position, _candidate in group:
                resolved[position].append(tick)
            used_exceptions += len(group)
        for position in range(start, end):
            resolved[position].sort()

    density_hints = build_density_hints(bars)
    for position, hint in enumerate(density_hints):
        required = min(hint.min_hits, len(raw_skeleton[position]))
        if len(resolved[position]) >= required:
            continue
        selected = set(resolved[position])
        missing = [
            tick
            for tick in raw_skeleton[position]
            if tick not in selected
        ]
        missing.sort(
            key=lambda tick: (
                candidate_maps[position].get(tick).priority
                if candidate_maps[position].get(tick) is not None
                else 99,
                -_lattice_weight(candidate_maps[position].get(tick)),
                tick,
            )
        )
        resolved[position].extend(missing[: required - len(resolved[position])])
        resolved[position].sort()
    return resolved


def _select_phrase_lattice(
    raw_skeleton: list[list[int]],
    candidate_maps: list[dict[int, SalienceCandidate]],
    *,
    start: int,
    end: int,
) -> RhythmLattice:
    weighted_ticks = [
        (tick, _lattice_weight(candidate_maps[position].get(tick)))
        for position in range(start, end)
        for tick in raw_skeleton[position]
    ]
    total_weight = sum(weight for _tick, weight in weighted_ticks)
    if total_weight <= 0 or len(weighted_ticks) < 4:
        return RhythmLattice(kind="straight", steps=((3, 0),), coverage=1.0)

    straight = max(
        (
            _lattice_for_phase(weighted_ticks, step=3, phase=phase, kind="straight")
            for phase in range(3)
        ),
        key=lambda item: (item.coverage, -item.steps[0][1]),
    )
    triplet = max(
        (
            _lattice_for_phase(weighted_ticks, step=2, phase=phase, kind="triplet")
            for phase in range(2)
        ),
        key=lambda item: (item.coverage, -item.steps[0][1]),
    )
    single = max(
        (straight, triplet),
        key=lambda item: (
            item.coverage,
            item.kind == "straight",
            -item.steps[0][1],
        ),
    )
    union_weight = sum(
        weight
        for tick, weight in weighted_ticks
        if straight.contains(tick) or triplet.contains(tick)
    )
    straight_unique = sum(
        weight
        for tick, weight in weighted_ticks
        if straight.contains(tick) and not triplet.contains(tick)
    )
    triplet_unique = sum(
        weight
        for tick, weight in weighted_ticks
        if triplet.contains(tick) and not straight.contains(tick)
    )
    union_coverage = union_weight / total_weight
    if (
        union_coverage - single.coverage >= LATTICE_MIXED_MIN_GAIN
        and straight_unique / total_weight >= LATTICE_MIXED_MIN_UNIQUE
        and triplet_unique / total_weight >= LATTICE_MIXED_MIN_UNIQUE
    ):
        return RhythmLattice(
            kind="mixed",
            steps=straight.steps + triplet.steps,
            coverage=union_coverage,
        )
    return single


def _lattice_for_phase(
    weighted_ticks: list[tuple[int, float]],
    *,
    step: int,
    phase: int,
    kind: str,
) -> RhythmLattice:
    total_weight = sum(weight for _tick, weight in weighted_ticks)
    matched_weight = sum(
        weight
        for tick, weight in weighted_ticks
        if tick % step == phase
    )
    return RhythmLattice(
        kind=kind,
        steps=((step, phase),),
        coverage=matched_weight / total_weight if total_weight > 0 else 0.0,
    )


def _lattice_weight(candidate: SalienceCandidate | None) -> float:
    if candidate is None:
        return 0.25
    kind_bonus = {
        "strong-transient": 0.75,
        "transient": 0.40,
        "rhythmic-skeleton": 0.20,
        "structure-highlight": 0.10,
        "weak-evidence": 0.0,
    }[candidate.kind]
    return 0.25 + candidate.score + candidate.point.confidence * 0.5 + kind_bonus


def _keep_lattice_exception(
    candidate: SalienceCandidate | None,
    *,
    exact_recurrence: int,
) -> bool:
    if candidate is None or candidate.kind != "strong-transient":
        return False
    reasons = set(candidate.point.reasons)
    explicit_onset = "onset" in reasons or "stem:drum-agreement" in reasons
    return bool(
        candidate.point.confidence >= LATTICE_EXCEPTION_MIN_CONFIDENCE
        and exact_recurrence >= LATTICE_EXCEPTION_MIN_REPETITIONS
        and explicit_onset
    )


def _phrase_ranges(bars: list[BarFeature]) -> list[tuple[int, int]]:
    if not bars:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for position in range(1, len(bars)):
        current = bars[position]
        previous = bars[position - 1]
        phrase_changed = (
            current.phrase_id is not None
            and previous.phrase_id is not None
            and current.phrase_id != previous.phrase_id
        )
        explicit_start = current.phrase_position == "phrase_start"
        if phrase_changed or explicit_start:
            ranges.append((start, position))
            start = position
    ranges.append((start, len(bars)))
    if len(ranges) == 1 and len(bars) > 4:
        return [
            (position, min(len(bars), position + 4))
            for position in range(0, len(bars), 4)
        ]
    return ranges


def _build_skeleton_chart_bars(
    analysis: SongAnalysis,
    *,
    course: str,
    level: int,
    style: str,
    density: str,
    precomputed_salience: list[BarRhythmicSalience] | None = None,
) -> list[ChartBar]:
    return generate_fallback_chart_bars(
        analysis.bars,
        style=style,
        density=density,
        special_notes=False,
        course=course,
        level=level,
        resolution_plan=analysis.resolution_plan,
        precomputed_salience=precomputed_salience,
    )


def _copy_long_notes(chart_bar: ChartBar, target: list[str]) -> set[int]:
    blocked: set[int] = set()
    start: int | None = None
    for position, note in enumerate(chart_bar.notes):
        if note in "57":
            start = position
            target[position] = note
        elif note == "8" and start is not None:
            target[position] = note
            blocked.update(range(start, position + 1))
            start = None
    return blocked


def _regular_hits_with_notes(
    chart_bar: ChartBar,
    feature_bar: BarFeature,
) -> list[tuple[int, str]]:
    if not chart_bar.notes:
        return []
    return [
        (
            _canonical_tick(position, len(chart_bar.notes), feature_bar.grids_per_bar),
            note,
        )
        for position, note in enumerate(chart_bar.notes)
        if note in "1234"
    ]


def _nearest_hit(
    hits: list[tuple[int, str]],
    tick: int,
) -> tuple[int, str] | None:
    if not hits:
        return None
    return min(hits, key=lambda item: (abs(item[0] - tick), item[0]))


def _regular_hit_ticks(chart_bar: ChartBar, feature_bar: BarFeature) -> set[int]:
    if not chart_bar.notes:
        return set()
    return {
        _canonical_tick(position, len(chart_bar.notes), feature_bar.grids_per_bar)
        for position, note in enumerate(chart_bar.notes)
        if note in "1234"
    }


def _long_note_spans(
    chart_bar: ChartBar,
    feature_bar: BarFeature,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for position, note in enumerate(chart_bar.notes):
        tick = _canonical_tick(position, len(chart_bar.notes), feature_bar.grids_per_bar)
        if note in "57":
            start = tick
        elif note == "8" and start is not None:
            spans.append((start, tick))
            start = None
    return spans


def _canonical_tick(position: int, resolution: int, canonical_grids: int) -> int:
    return min(
        canonical_grids - 1,
        round(position / resolution * canonical_grids),
    )
