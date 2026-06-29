from math import ceil

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.tja.model import BarFeature


BEATS_PER_BAR = 4
GRIDS_PER_BAR = 16


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

    for onset_time in raw.onset_times:
        relative_time = onset_time - analysis_start
        if relative_time < 0:
            continue

        bar_index = int(relative_time // bar_length)
        if bar_index < 0 or bar_index >= bar_count:
            continue

        time_in_bar = relative_time - (bar_index * bar_length)
        grid_index = round(time_in_bar / grid_length)
        grid_index = max(0, min(meter.grids_per_bar - 1, grid_index))
        onset_grids_by_bar[bar_index].add(grid_index)

    bars: list[BarFeature] = []
    for index, onset_grids in enumerate(onset_grids_by_bar):
        start_time = analysis_start + (index * bar_length)
        end_time = start_time + bar_length
        onset_16 = sorted(onset_grids)
        accent_16 = [grid for grid in onset_16 if grid in meter.accent_grids]
        if not accent_16 and onset_16:
            accent_16 = [grid for grid in onset_16 if grid in {0, meter.grids_per_bar // 2}]

        bars.append(
            BarFeature(
                index=index,
                start_time=round(start_time, 6),
                end_time=round(end_time, 6),
                energy=min(1.0, len(onset_16) / meter.grids_per_bar),
                time_signature=meter.time_signature,
                grids_per_bar=meter.grids_per_bar,
                onset_16=onset_16,
                accent_16=accent_16,
            )
        )

    return bars
