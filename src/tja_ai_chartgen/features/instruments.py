from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw
from tja_ai_chartgen.tja.model import InstrumentBarFeature, InstrumentGridFeature


INSTRUMENT_GRID_MINIMUM = 0.05
SOURCE_ACTIVITY_MINIMUM = 0.12
SOURCE_MARGIN_MINIMUM = 0.08
INSTRUMENT_SCORE_MINIMUM = 0.25
INSTRUMENT_MARGIN_MINIMUM = 0.08
INSTRUMENT_LABELS = (
    "guitar",
    "piano_keyboard",
    "strings",
    "brass",
    "woodwind",
    "synth",
    "organ",
    "other_instrument",
)
SOURCE_FIELDS = (
    ("vocals", "vocal_activity"),
    ("drums", "drum_activity"),
    ("bass", "bass_activity"),
    ("other", "other_activity"),
)


@dataclass(frozen=True)
class InstrumentFeatureMapping:
    bars: list[InstrumentBarFeature]
    grids: list[list[InstrumentGridFeature]]


def map_instrument_features(
    raw: InstrumentAnalysisRaw,
    *,
    analysis_start: float,
    bar_length: float,
    grid_length: float,
    grids_per_bar: int,
    bar_count: int,
) -> InstrumentFeatureMapping:
    if bar_count <= 0:
        return InstrumentFeatureMapping([], [])
    activity_frames: list[list[object]] = [[] for _ in range(bar_count)]
    grid_values: list[dict[int, dict[str, float]]] = [{} for _ in range(bar_count)]

    for frame in raw.stem_frames:
        bar_index, grid_index = _time_to_bar_grid(
            frame.time,
            analysis_start=analysis_start,
            grid_length=grid_length,
            grids_per_bar=grids_per_bar,
            bar_count=bar_count,
        )
        if bar_index is None or grid_index is None:
            continue
        activity_frames[bar_index].append(frame)
        values = grid_values[bar_index].setdefault(grid_index, {})
        for name in (
            "vocal_onset",
            "drum_onset",
            "bass_onset",
            "accompaniment_onset",
        ):
            values[name] = max(values.get(name, 0.0), _bounded(getattr(frame, name)))

    bars = [
        _build_bar_feature(
            activity_frames[index],
            raw,
            bar_start=analysis_start + index * bar_length,
            bar_end=analysis_start + (index + 1) * bar_length,
        )
        for index in range(bar_count)
    ]
    grids = [_build_grid_features(values) for values in grid_values]
    return InstrumentFeatureMapping(bars, grids)


def _build_bar_feature(
    frames: list[object],
    raw: InstrumentAnalysisRaw,
    *,
    bar_start: float,
    bar_end: float,
) -> InstrumentBarFeature:
    source_values = {
        output_name: _mean([_bounded(getattr(frame, input_name)) for frame in frames])
        for input_name, output_name in SOURCE_FIELDS
    }
    vocal_presence = _mean(
        [1.0 if _bounded(getattr(frame, "vocals")) >= 0.08 else 0.0 for frame in frames]
    )
    instrument_scores = _window_scores(raw, bar_start=bar_start, bar_end=bar_end)
    dominant_source, source_confidence = _dominant(
        {
            source_name: source_values[output_name]
            for source_name, output_name in SOURCE_FIELDS
        },
        minimum=SOURCE_ACTIVITY_MINIMUM,
        margin=SOURCE_MARGIN_MINIMUM,
    )
    dominant_instrument, instrument_confidence = _dominant(
        instrument_scores,
        minimum=INSTRUMENT_SCORE_MINIMUM,
        margin=INSTRUMENT_MARGIN_MINIMUM,
    )
    if source_values["other_activity"] < 0.05:
        dominant_instrument = None
        instrument_confidence = 0.0
    confidence = max(source_confidence, instrument_confidence)
    return InstrumentBarFeature(
        vocal_activity=_rounded(source_values["vocal_activity"]),
        vocal_presence_ratio=_rounded(vocal_presence),
        drum_activity=_rounded(source_values["drum_activity"]),
        bass_activity=_rounded(source_values["bass_activity"]),
        other_activity=_rounded(source_values["other_activity"]),
        dominant_source=dominant_source,
        dominant_instrument=dominant_instrument,
        confidence=_rounded(confidence),
        **{name: _rounded(instrument_scores.get(name, 0.0)) for name in INSTRUMENT_LABELS},
    )


def _window_scores(
    raw: InstrumentAnalysisRaw,
    *,
    bar_start: float,
    bar_end: float,
) -> dict[str, float]:
    weighted = {name: 0.0 for name in INSTRUMENT_LABELS}
    total_weight = 0.0
    for window in raw.classification_windows:
        overlap = min(bar_end, window.end_time) - max(bar_start, window.start_time)
        if overlap <= 0:
            continue
        total_weight += overlap
        for name in INSTRUMENT_LABELS:
            mix = _bounded(window.mix_scores.get(name, 0.0))
            other = _bounded(window.other_scores.get(name, 0.0))
            weighted[name] += (other * 0.65 + mix * 0.35) * overlap
    if total_weight <= 0:
        return weighted
    return {name: value / total_weight for name, value in weighted.items()}


def _build_grid_features(
    grids: dict[int, dict[str, float]],
) -> list[InstrumentGridFeature]:
    result: list[InstrumentGridFeature] = []
    for grid, values in sorted(grids.items()):
        if max(values.values(), default=0.0) < INSTRUMENT_GRID_MINIMUM:
            continue
        result.append(
            InstrumentGridFeature(
                grid=grid,
                vocal_onset=_rounded(values.get("vocal_onset", 0.0)),
                drum_onset=_rounded(values.get("drum_onset", 0.0)),
                bass_onset=_rounded(values.get("bass_onset", 0.0)),
                accompaniment_onset=_rounded(values.get("accompaniment_onset", 0.0)),
            )
        )
    return result


def _dominant(
    values: dict[str, float],
    *,
    minimum: float,
    margin: float,
) -> tuple[str | None, float]:
    ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] < minimum:
        return None, 0.0
    strongest_name, strongest = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    difference = strongest - runner_up
    if difference < margin:
        return None, 0.0
    confidence = min(1.0, strongest * 0.7 + difference * 0.6)
    return strongest_name, confidence


def _time_to_bar_grid(
    time: float,
    *,
    analysis_start: float,
    grid_length: float,
    grids_per_bar: int,
    bar_count: int,
) -> tuple[int | None, int | None]:
    if not isfinite(time) or grid_length <= 0 or grids_per_bar <= 0:
        return None, None
    relative = time - analysis_start
    if relative < 0:
        return None, None
    absolute_grid = round(relative / grid_length)
    bar_index, grid_index = divmod(absolute_grid, grids_per_bar)
    if bar_index < 0 or bar_index >= bar_count:
        return None, None
    return bar_index, grid_index


def _bounded(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rounded(value: float) -> float:
    return round(_bounded(value), 3)
