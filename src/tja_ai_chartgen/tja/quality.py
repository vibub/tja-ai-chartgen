from collections import Counter
from math import isfinite

from pydantic import BaseModel, Field

from tja_ai_chartgen.features.density import BarDensityHint, build_density_hints
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.tja.model import BarFeature, ChartBar


class QualityReport(BaseModel):
    bar_count: int = Field(ge=0)
    density_compliant_bars: int = Field(ge=0)
    density_evaluated_bars: int = Field(ge=0)
    density_compliance_rate: float = Field(ge=0.0, le=1.0)
    silent_bar_note_count: int = Field(ge=0)
    longest_empty_bar_run: int = Field(ge=0)
    repeated_bar_count: int = Field(ge=0)
    repeated_bar_rate: float = Field(ge=0.0, le=1.0)
    don_count: int = Field(ge=0)
    ka_count: int = Field(ge=0)
    ka_ratio: float = Field(ge=0.0, le=1.0)
    longest_monochrome_run: int = Field(ge=0)
    playable_note_count: int = Field(ge=0)
    playable_duration_seconds: float = Field(ge=0.0)
    average_notes_per_second: float = Field(ge=0.0)
    peak_bar_notes_per_second: float = Field(ge=0.0)


def build_quality_report(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
) -> QualityReport:
    paired_count = min(len(chart_bars), len(feature_bars))
    paired_chart_bars = chart_bars[:paired_count]
    paired_feature_bars = feature_bars[:paired_count]
    density_hints = build_density_hints(paired_feature_bars)
    hit_counts = [playable_hit_count(bar.notes) for bar in chart_bars]
    density_compliant = sum(
        density_hint_is_satisfied(hit_count, hint)
        for hit_count, hint in zip(hit_counts, density_hints, strict=False)
    )
    density_evaluated = len(density_hints)

    silent_indexes = edge_silence_indexes(paired_feature_bars)
    silent_bar_note_count = sum(
        playable_hit_count(paired_chart_bars[index].notes)
        for index in silent_indexes
        if index < len(paired_chart_bars)
    )

    nonempty_patterns = [bar.notes for bar in chart_bars if playable_hit_count(bar.notes) > 0]
    repeated_bar_count = sum(count - 1 for count in Counter(nonempty_patterns).values())
    don_count, ka_count, longest_monochrome_run = note_color_metrics(chart_bars)
    normal_note_count = don_count + ka_count
    timed_hit_counts: list[int] = []
    timed_durations: list[float] = []
    for chart_bar, feature_bar in zip(
        paired_chart_bars, paired_feature_bars, strict=True
    ):
        duration = feature_bar.end_time - feature_bar.start_time
        if not isfinite(duration) or duration <= 0:
            continue
        timed_hit_counts.append(playable_hit_count(chart_bar.notes))
        timed_durations.append(duration)
    playable_note_count = sum(timed_hit_counts)
    playable_duration_seconds = sum(timed_durations)
    bar_notes_per_second = [
        hit_count / duration
        for hit_count, duration in zip(timed_hit_counts, timed_durations, strict=True)
    ]

    return QualityReport(
        bar_count=len(chart_bars),
        density_compliant_bars=density_compliant,
        density_evaluated_bars=density_evaluated,
        density_compliance_rate=(
            density_compliant / density_evaluated if density_evaluated else 1.0
        ),
        silent_bar_note_count=silent_bar_note_count,
        longest_empty_bar_run=longest_empty_bar_run(hit_counts),
        repeated_bar_count=repeated_bar_count,
        repeated_bar_rate=(
            repeated_bar_count / len(nonempty_patterns) if nonempty_patterns else 0.0
        ),
        don_count=don_count,
        ka_count=ka_count,
        ka_ratio=ka_count / normal_note_count if normal_note_count else 0.0,
        longest_monochrome_run=longest_monochrome_run,
        playable_note_count=playable_note_count,
        playable_duration_seconds=playable_duration_seconds,
        average_notes_per_second=(
            playable_note_count / playable_duration_seconds
            if playable_duration_seconds
            else 0.0
        ),
        peak_bar_notes_per_second=max(bar_notes_per_second, default=0.0),
    )


def playable_hit_count(notes: str) -> int:
    return sum(character != "0" for character in notes)


def normalized_hit_count(notes: str, expected_length: int) -> float:
    if expected_length <= 0:
        return 0.0
    return playable_hit_count(notes) * 16 / expected_length


def density_hint_is_satisfied(hit_count: int, hint: BarDensityHint) -> bool:
    if hint.max_hits is not None and hit_count > hint.max_hits:
        return False
    return hint.allow_empty or hit_count >= hint.min_hits


def empty_runs(
    hit_counts: list[int],
    indexes: list[int] | None = None,
) -> list[tuple[int, int]]:
    selected_indexes = indexes if indexes is not None else list(range(len(hit_counts)))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous_index: int | None = None

    for index in selected_indexes:
        if index < 0 or index >= len(hit_counts):
            continue
        continues_run = previous_index is not None and index == previous_index + 1
        if not continues_run and start is not None and previous_index is not None:
            runs.append((start, previous_index))
            start = None

        if hit_counts[index] == 0:
            if start is None:
                start = index
        elif start is not None and previous_index is not None:
            runs.append((start, previous_index))
            start = None

        previous_index = index

    if start is not None and previous_index is not None:
        runs.append((start, previous_index))
    return runs


def longest_empty_bar_run(hit_counts: list[int]) -> int:
    return max((end - start + 1 for start, end in empty_runs(hit_counts)), default=0)


def pattern_counts(chart_bars: list[ChartBar]) -> Counter[str]:
    return Counter(bar.notes for bar in chart_bars if playable_hit_count(bar.notes) > 0)


def note_color_metrics(chart_bars: list[ChartBar]) -> tuple[int, int, int]:
    don_count = 0
    ka_count = 0
    longest_run = 0
    current_color: str | None = None
    current_run = 0

    for bar in chart_bars:
        for note in bar.notes:
            if note == "0":
                continue
            if note not in {"1", "2"}:
                current_color = None
                current_run = 0
                continue

            if note == "1":
                don_count += 1
            else:
                ka_count += 1

            if note == current_color:
                current_run += 1
            else:
                current_color = note
                current_run = 1
            longest_run = max(longest_run, current_run)

    return don_count, ka_count, longest_run
