from dataclasses import dataclass
from math import isfinite

from tja_ai_chartgen.tja.model import BarFeature, ChartBar

BIG_NOTE_ISOLATION_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class BigNoteIsolationViolation:
    bar_position: int
    grid: int
    canonical_tick: int
    note: str
    nearest_hit_distance_seconds: float


@dataclass(frozen=True, slots=True)
class _TimedHit:
    time: float
    bar_position: int
    grid: int
    canonical_tick: int
    note: str


def find_big_note_isolation_violations(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    *,
    minimum_seconds: float = BIG_NOTE_ISOLATION_SECONDS,
) -> list[BigNoteIsolationViolation]:
    """查找与任意普通击打距离过近、无法独立双手击打的大音符。"""
    if minimum_seconds < 0:
        raise ValueError("minimum_seconds must be non-negative")

    hits: list[_TimedHit] = []
    for bar_position, (chart_bar, feature_bar) in enumerate(
        zip(chart_bars, feature_bars, strict=False)
    ):
        duration = feature_bar.end_time - feature_bar.start_time
        if (
            not chart_bar.notes
            or not isfinite(duration)
            or duration <= 0
            or feature_bar.grids_per_bar <= 0
        ):
            continue
        for grid, note in enumerate(chart_bar.notes):
            if note not in "1234":
                continue
            canonical_tick = min(
                feature_bar.grids_per_bar - 1,
                round(grid / len(chart_bar.notes) * feature_bar.grids_per_bar),
            )
            hits.append(
                _TimedHit(
                    time=feature_bar.start_time + duration * grid / len(chart_bar.notes),
                    bar_position=bar_position,
                    grid=grid,
                    canonical_tick=canonical_tick,
                    note=note,
                )
            )

    hits.sort(key=lambda hit: (hit.time, hit.bar_position, hit.grid))
    violations: list[BigNoteIsolationViolation] = []
    for position, hit in enumerate(hits):
        if hit.note not in "34":
            continue
        distances = []
        if position > 0:
            distances.append(hit.time - hits[position - 1].time)
        if position + 1 < len(hits):
            distances.append(hits[position + 1].time - hit.time)
        if not distances:
            continue
        nearest = min(distances)
        if nearest > minimum_seconds:
            continue
        violations.append(
            BigNoteIsolationViolation(
                bar_position=hit.bar_position,
                grid=hit.grid,
                canonical_tick=hit.canonical_tick,
                note=hit.note,
                nearest_hit_distance_seconds=max(0.0, nearest),
            )
        )
    return violations
