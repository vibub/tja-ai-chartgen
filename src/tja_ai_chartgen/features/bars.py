from math import ceil

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.features.meter import MeterSpec, get_meter_spec
from tja_ai_chartgen.tja.model import BarFeature, GridFeature


BEATS_PER_BAR = 4
GRIDS_PER_BAR = 16
PHRASE_LENGTH_BARS = 4


def build_bar_features(raw: AudioAnalysisRaw, max_bars: int | None = None) -> list[BarFeature]:
    if raw.bpm <= 0:
        raise ValueError(f"BPM must be positive, got {raw.bpm}")

    meter = get_meter_spec(raw.time_signature)
    beat_length = 60.0 / raw.bpm
    bar_length = meter.beats_per_bar * beat_length
    grid_length = bar_length / meter.grids_per_bar
    analysis_start = raw.offset
    effective_duration = max(0.0, raw.duration - analysis_start)
    bar_count = max(1, ceil(effective_duration / bar_length))

    if max_bars is not None:
        bar_count = min(bar_count, max_bars)

    onset_grids_by_bar: list[set[int]] = [set() for _ in range(bar_count)]
    strengths_by_bar: list[dict[int, float]] = [{} for _ in range(bar_count)]
    max_onset_strength = max(raw.onset_strengths, default=0.0)

    for onset_time in raw.onset_times:
        bar_index, grid_index = _time_to_bar_grid(
            onset_time,
            analysis_start=analysis_start,
            bar_length=bar_length,
            grid_length=grid_length,
            grids_per_bar=meter.grids_per_bar,
            bar_count=bar_count,
        )
        if bar_index is None or grid_index is None:
            continue

        onset_grids_by_bar[bar_index].add(grid_index)
        strength = _normalized_onset_strength(raw, onset_time, max_onset_strength)
        strengths_by_bar[bar_index][grid_index] = max(
            strengths_by_bar[bar_index].get(grid_index, 0.0),
            strength,
        )

    beat_numbers_by_bar, downbeat_grids_by_bar = _map_beats_to_grids(
        raw,
        meter=meter,
        analysis_start=analysis_start,
        bar_length=bar_length,
        grid_length=grid_length,
        bar_count=bar_count,
    )

    bars: list[BarFeature] = []
    for index, onset_grids in enumerate(onset_grids_by_bar):
        start_time = analysis_start + (index * bar_length)
        end_time = start_time + bar_length
        onset_16 = sorted(onset_grids)
        accent_16 = [grid for grid in onset_16 if grid in meter.accent_grids]
        if not accent_16 and onset_16:
            accent_16 = _strongest_onset_grids(strengths_by_bar[index])

        beat_numbers = beat_numbers_by_bar[index]
        downbeat_grids = downbeat_grids_by_bar[index]
        phrase_position = _phrase_position(index, bar_count)
        energy = min(1.0, sum(strengths_by_bar[index].values()) / meter.grids_per_bar)
        if not raw.onset_strengths:
            energy = min(1.0, len(onset_16) / meter.grids_per_bar)

        bars.append(
            BarFeature(
                index=index,
                start_time=round(start_time, 6),
                end_time=round(end_time, 6),
                energy=round(energy, 3),
                time_signature=meter.time_signature,
                grids_per_bar=meter.grids_per_bar,
                onset_16=onset_16,
                accent_16=accent_16,
                grid_features=_build_grid_features(
                    grids_per_bar=meter.grids_per_bar,
                    onset_grids=onset_grids,
                    accent_grids=set(accent_16),
                    strengths=strengths_by_bar[index],
                    beat_numbers=beat_numbers,
                    downbeat_grids=downbeat_grids,
                ),
                beat_grids=sorted(beat_numbers),
                downbeat_grid=min(downbeat_grids) if downbeat_grids else None,
                phrase_position=phrase_position,
                fill_candidate=phrase_position in {"phrase_end", "song_end"} and bool(onset_16),
            )
        )

    return bars


def _time_to_bar_grid(
    time: float,
    *,
    analysis_start: float,
    bar_length: float,
    grid_length: float,
    grids_per_bar: int,
    bar_count: int,
) -> tuple[int | None, int | None]:
    relative_time = time - analysis_start
    if relative_time < 0:
        return None, None

    bar_index = int(relative_time // bar_length)
    if bar_index < 0 or bar_index >= bar_count:
        return None, None

    time_in_bar = relative_time - (bar_index * bar_length)
    grid_index = round(time_in_bar / grid_length)
    grid_index = max(0, min(grids_per_bar - 1, grid_index))
    return bar_index, grid_index


def _normalized_onset_strength(raw: AudioAnalysisRaw, onset_time: float, max_onset_strength: float) -> float:
    if max_onset_strength <= 0 or not raw.onset_strengths:
        return 1.0

    if raw.sample_rate:
        frame_index = round(onset_time * raw.sample_rate / raw.hop_length)
    elif raw.duration > 0:
        frame_index = round((onset_time / raw.duration) * (len(raw.onset_strengths) - 1))
    else:
        frame_index = 0

    frame_index = max(0, min(len(raw.onset_strengths) - 1, frame_index))
    return round(max(0.0, raw.onset_strengths[frame_index]) / max_onset_strength, 3)


def _map_beats_to_grids(
    raw: AudioAnalysisRaw,
    *,
    meter: MeterSpec,
    analysis_start: float,
    bar_length: float,
    grid_length: float,
    bar_count: int,
) -> tuple[list[dict[int, int]], list[set[int]]]:
    beat_numbers_by_bar: list[dict[int, int]] = [{} for _ in range(bar_count)]
    downbeat_grids_by_bar: list[set[int]] = [set() for _ in range(bar_count)]
    downbeat_times = set(round(time, 6) for time in raw.downbeat_times)

    for beat_index, beat_time in enumerate(raw.beat_times):
        bar_index, grid_index = _time_to_bar_grid(
            beat_time,
            analysis_start=analysis_start,
            bar_length=bar_length,
            grid_length=grid_length,
            grids_per_bar=meter.grids_per_bar,
            bar_count=bar_count,
        )
        if bar_index is None or grid_index is None:
            continue

        beat_number = _beat_number_for_grid(raw, beat_index, grid_index, meter)
        beat_numbers_by_bar[bar_index][grid_index] = beat_number
        if beat_number == 1 or round(beat_time, 6) in downbeat_times:
            downbeat_grids_by_bar[bar_index].add(grid_index)

    for bar_index, beat_numbers in enumerate(beat_numbers_by_bar):
        if 0 in beat_numbers and not downbeat_grids_by_bar[bar_index]:
            downbeat_grids_by_bar[bar_index].add(0)

    return beat_numbers_by_bar, downbeat_grids_by_bar


def _beat_number_for_grid(raw: AudioAnalysisRaw, beat_index: int, grid_index: int, meter: MeterSpec) -> int:
    if beat_index < len(raw.beat_numbers):
        return raw.beat_numbers[beat_index]

    grids_per_beat = meter.grids_per_bar / meter.beats_per_bar
    return int(grid_index // grids_per_beat) + 1


def _strongest_onset_grids(strengths: dict[int, float]) -> list[int]:
    if not strengths:
        return []
    max_strength = max(strengths.values())
    return sorted(grid for grid, strength in strengths.items() if strength == max_strength)


def _build_grid_features(
    *,
    grids_per_bar: int,
    onset_grids: set[int],
    accent_grids: set[int],
    strengths: dict[int, float],
    beat_numbers: dict[int, int],
    downbeat_grids: set[int],
) -> list[GridFeature]:
    return [
        GridFeature(
            grid=grid,
            onset=grid in onset_grids,
            accent=grid in accent_grids,
            beat=beat_numbers.get(grid),
            downbeat=grid in downbeat_grids,
            strength=strengths.get(grid, 0.0),
        )
        for grid in range(grids_per_bar)
    ]


def _phrase_position(index: int, bar_count: int) -> str:
    if index == bar_count - 1:
        return "song_end"

    position = index % PHRASE_LENGTH_BARS
    if position == 0:
        return "phrase_start"
    if position == PHRASE_LENGTH_BARS - 1:
        return "phrase_end"
    return "phrase_middle"
