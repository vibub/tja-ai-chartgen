from dataclasses import dataclass

from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, SongAnalysis

RHYTHM_SKELETON_VERSION = "rhythm-skeleton-v3"
AI_SKELETON_MIN_BARS = 16
AI_SKELETON_MIN_HITS = 32
AI_SKELETON_MIN_COVERAGE = 0.85
AI_SKELETON_MAX_EXTRA_RATE = 0.15


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
    skeleton_bars = _build_skeleton_chart_bars(
        analysis,
        course=course,
        level=level,
        style=style,
        density=density,
    )
    return [
        sorted(_regular_hit_ticks(chart_bar, feature_bar))
        for chart_bar, feature_bar in zip(
            skeleton_bars,
            analysis.bars,
            strict=True,
        )
    ]


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
    fallback_bars = _build_skeleton_chart_bars(
        analysis,
        course=course,
        level=level,
        style=style,
        density=density,
    )
    conformed: list[ChartBar] = []
    for chart_bar, fallback_bar, feature_bar, required_ticks in zip(
        chart_bars,
        fallback_bars,
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
        fallback_hits = _regular_hits_with_notes(fallback_bar, feature_bar)
        for tick in required_ticks:
            if tick in blocked_ticks:
                continue
            position = round(tick / feature_bar.grids_per_bar * len(notes))
            position = min(len(notes) - 1, max(0, position))
            source_tick, note = _nearest_hit(ai_hits, tick) or _nearest_hit(
                fallback_hits,
                tick,
            ) or (tick, "1")
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


def _build_skeleton_chart_bars(
    analysis: SongAnalysis,
    *,
    course: str,
    level: int,
    style: str,
    density: str,
) -> list[ChartBar]:
    return generate_fallback_chart_bars(
        analysis.bars,
        style=style,
        density=density,
        special_notes=False,
        course=course,
        level=level,
        resolution_plan=analysis.resolution_plan,
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
