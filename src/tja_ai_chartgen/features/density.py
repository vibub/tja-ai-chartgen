from dataclasses import dataclass

from tja_ai_chartgen.features.silence import edge_silence_indexes, is_silent_bar
from tja_ai_chartgen.tja.model import BarFeature


@dataclass(frozen=True)
class BarDensityHint:
    kind: str
    min_hits: int
    max_hits: int | None
    allow_empty: bool
    count_in_quality_average: bool
    reason: str


def build_density_hints(bars: list[BarFeature]) -> list[BarDensityHint]:
    silent_indexes = edge_silence_indexes(bars)
    return [_density_hint_for_bar(index, bar, silent_indexes) for index, bar in enumerate(bars)]


def density_hint_payload(bars: list[BarFeature]) -> list[dict[str, object]]:
    hints = build_density_hints(bars)
    return [
        {
            "bar": index + 1,
            "kind": hint.kind,
            "min_hits": hint.min_hits,
            "max_hits": hint.max_hits,
            "allow_empty": hint.allow_empty,
            "count_in_quality_average": hint.count_in_quality_average,
            "reason": hint.reason,
        }
        for index, hint in enumerate(hints)
    ]


def _density_hint_for_bar(
    index: int,
    bar: BarFeature,
    silent_indexes: set[int],
) -> BarDensityHint:
    onset_count = len(bar.onset_16)
    activity_level = _activity_level(bar)

    if index in silent_indexes:
        return BarDensityHint(
            kind="silent",
            min_hits=0,
            max_hits=0,
            allow_empty=True,
            count_in_quality_average=False,
            reason="song-start/song-end silence",
        )

    if _is_musical_rest(bar, onset_count, activity_level):
        max_hits = 0 if is_silent_bar(bar) else 2
        return BarDensityHint(
            kind="rest",
            min_hits=0,
            max_hits=max_hits,
            allow_empty=True,
            count_in_quality_average=False,
            reason="low-energy musical pause or break",
        )

    if _is_sustained_musical_bar(bar, onset_count, activity_level):
        return BarDensityHint(
            kind="normal",
            min_hits=3,
            max_hits=12,
            allow_empty=False,
            count_in_quality_average=True,
            reason="sustained musical activity without strong onsets",
        )

    if _is_sparse_musical_bar(bar, onset_count, activity_level):
        return BarDensityHint(
            kind="sparse",
            min_hits=1 if onset_count else 0,
            max_hits=4,
            allow_empty=onset_count == 0,
            count_in_quality_average=False,
            reason="low-energy sparse passage",
        )

    if bar.energy >= 0.45 or onset_count >= max(6, bar.grids_per_bar // 2):
        return BarDensityHint(
            kind="dense",
            min_hits=5,
            max_hits=16,
            allow_empty=False,
            count_in_quality_average=True,
            reason="high-energy or onset-rich bar",
        )

    if bar.fill_candidate or bar.phrase_position in {"phrase_end", "song_end"}:
        return BarDensityHint(
            kind="fill",
            min_hits=3,
            max_hits=14,
            allow_empty=False,
            count_in_quality_average=True,
            reason="phrase ending or fill candidate",
        )

    return BarDensityHint(
        kind="normal",
        min_hits=2,
        max_hits=12,
        allow_empty=False,
        count_in_quality_average=True,
        reason="regular playable phrase bar",
    )


def _activity_level(bar: BarFeature) -> float:
    if not bar.activity_16:
        return 0.0
    return max(bar.activity_16)


def _is_musical_rest(bar: BarFeature, onset_count: int, activity_level: float) -> bool:
    if activity_level >= 0.18:
        return False
    if is_silent_bar(bar):
        return bar.section in {"break", "verse", "unknown"}
    if bar.section != "break":
        return False
    if bar.energy <= 0.025 and onset_count <= 1:
        return True
    return bar.energy <= 0.04 and onset_count == 0


def _is_sustained_musical_bar(bar: BarFeature, onset_count: int, activity_level: float) -> bool:
    active_grid_count = sum(value >= 0.18 for value in bar.activity_16)
    return activity_level >= 0.25 and active_grid_count >= 3 and onset_count <= 3


def _is_sparse_musical_bar(bar: BarFeature, onset_count: int, activity_level: float) -> bool:
    if bar.energy <= 0.08 and onset_count <= 2:
        return True
    if activity_level >= 0.18:
        return False
    return bar.section == "break" and bar.energy <= 0.08 and onset_count <= 3
