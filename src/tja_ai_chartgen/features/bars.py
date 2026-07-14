from math import ceil, isfinite, log10, sqrt

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.features.instruments import map_instrument_features
from tja_ai_chartgen.features.meter import MeterSpec, get_meter_spec
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.tja.model import (
    BarFeature,
    GridFeature,
    InstrumentBarFeature,
    SpectralGridFeature,
)


BEATS_PER_BAR = 4
GRIDS_PER_BAR = 16
LOUDNESS_FLOOR_DBFS = -120.0
LOUDNESS_EPSILON = 10 ** (LOUDNESS_FLOOR_DBFS / 20.0)
SUSTAINED_ACTIVITY_DB_RANGE = 20.0
SPECTRAL_GRID_MINIMUM = 0.05


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
    activity_by_bar: list[dict[int, float]] = [{} for _ in range(bar_count)]
    rms_frames_by_bar: list[list[float]] = [[] for _ in range(bar_count)]
    spectral_frames_by_bar: list[dict[str, list[float]]] = [
        _empty_spectral_frames() for _ in range(bar_count)
    ]
    spectral_grids_by_bar: list[dict[int, dict[str, float]]] = [
        {} for _ in range(bar_count)
    ]
    max_onset_strength = max(raw.onset_strengths, default=0.0)

    for onset_time in raw.onset_times:
        bar_index, grid_index = _time_to_bar_grid(
            onset_time,
            analysis_start=analysis_start,
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

    for frame_index, activity in enumerate(raw.activity_envelope):
        if activity <= 0:
            continue
        frame_time = frame_index * raw.hop_length / raw.sample_rate if raw.sample_rate else 0.0
        bar_index, grid_index = _time_to_bar_grid(
            frame_time,
            analysis_start=analysis_start,
            grid_length=grid_length,
            grids_per_bar=meter.grids_per_bar,
            bar_count=bar_count,
        )
        if bar_index is None or grid_index is None:
            continue
        activity_by_bar[bar_index][grid_index] = max(
            activity_by_bar[bar_index].get(grid_index, 0.0),
            float(activity),
        )

    if raw.sample_rate is not None and raw.sample_rate > 0 and raw.hop_length > 0:
        _map_spectral_frames(
            raw,
            analysis_start=analysis_start,
            grid_length=grid_length,
            grids_per_bar=meter.grids_per_bar,
            bar_count=bar_count,
            frames_by_bar=spectral_frames_by_bar,
            grids_by_bar=spectral_grids_by_bar,
        )

    if raw.sample_rate is not None and raw.sample_rate > 0 and raw.hop_length > 0:
        for frame_index, rms in enumerate(raw.rms_envelope):
            if not isfinite(rms) or rms < 0:
                continue
            frame_time = frame_index * raw.hop_length / raw.sample_rate
            bar_index = _time_to_bar_index(
                frame_time,
                analysis_start=analysis_start,
                bar_length=bar_length,
                bar_count=bar_count,
            )
            if bar_index is not None:
                rms_frames_by_bar[bar_index].append(float(rms))

    global_peak_rms = _global_peak_rms(raw.rms_envelope)
    instrument_mapping = map_instrument_features(
        raw.instruments,
        analysis_start=analysis_start,
        bar_length=bar_length,
        grid_length=grid_length,
        grids_per_bar=meter.grids_per_bar,
        bar_count=bar_count,
    )
    beat_numbers_by_bar, downbeat_grids_by_bar = _map_meter_beats_to_grids(
        meter=meter,
        bar_count=bar_count,
    )

    bars: list[BarFeature] = []
    for index, onset_grids in enumerate(onset_grids_by_bar):
        start_time = analysis_start + (index * bar_length)
        end_time = start_time + bar_length
        onset_grid_values = sorted(onset_grids)
        accent_grid_values = [
            grid for grid in onset_grid_values if grid in meter.accent_grids
        ]
        if not accent_grid_values and onset_grid_values:
            accent_grid_values = _strongest_onset_grids(strengths_by_bar[index])

        activity_grid_values = _activity_values(activity_by_bar[index], meter.grids_per_bar)
        beat_numbers = beat_numbers_by_bar[index]
        downbeat_grids = downbeat_grids_by_bar[index]
        phrase_position = _phrase_position(index, bar_count)
        rms_dbfs, peak_rms_dbfs, relative_rms_db, sustained_activity_ratio = (
            _bar_loudness_metrics(rms_frames_by_bar[index], global_peak_rms)
        )
        spectral_summary = _spectral_bar_summary(spectral_frames_by_bar[index])
        spectral_grid_features = _spectral_grid_features(spectral_grids_by_bar[index])
        energy = max(
            min(
                1.0,
                sum(strengths_by_bar[index].values()) / meter.legacy_grids_per_bar,
            ),
            min(1.0, sum(activity_grid_values) / meter.grids_per_bar),
        )
        if not raw.onset_strengths and not raw.activity_envelope:
            energy = min(1.0, len(onset_grid_values) / meter.legacy_grids_per_bar)

        bars.append(
            BarFeature(
                index=index,
                start_time=round(start_time, 6),
                end_time=round(end_time, 6),
                energy=round(energy, 3),
                rms_dbfs=rms_dbfs,
                peak_rms_dbfs=peak_rms_dbfs,
                relative_rms_db=relative_rms_db,
                sustained_activity_ratio=sustained_activity_ratio,
                time_signature=meter.time_signature,
                grids_per_bar=meter.grids_per_bar,
                onset_grids=onset_grid_values,
                accent_grids=accent_grid_values,
                activity_grids=activity_grid_values,
                grid_features=_build_grid_features(
                    grids_per_bar=meter.grids_per_bar,
                    onset_grids=onset_grids,
                    accent_grids=set(accent_grid_values),
                    strengths=strengths_by_bar[index],
                    activities=activity_by_bar[index],
                    beat_numbers=beat_numbers,
                    downbeat_grids=downbeat_grids,
                ),
                spectral_grid_features=spectral_grid_features,
                instrument_grid_features=instrument_mapping.grids[index],
                instrument=instrument_mapping.bars[index],
                low_onset_strength=spectral_summary["low_onset_strength"],
                mid_onset_strength=spectral_summary["mid_onset_strength"],
                high_onset_strength=spectral_summary["high_onset_strength"],
                spectral_flux=spectral_summary["spectral_flux"],
                brightness=spectral_summary["brightness"],
                harmonic_novelty=spectral_summary["harmonic_novelty"],
                texture_novelty=spectral_summary["texture_novelty"],
                percussive_ratio=spectral_summary["percussive_ratio"],
                beat_grids=sorted(beat_numbers),
                downbeat_grid=min(downbeat_grids) if downbeat_grids else None,
                phrase_position=phrase_position,
                fill_candidate=False,
            )
        )

    for index in edge_silence_indexes(bars):
        bars[index] = bars[index].model_copy(
            update={
                "instrument": InstrumentBarFeature(),
                "instrument_grid_features": [],
            }
        )
    return bars


def _empty_spectral_frames() -> dict[str, list[float]]:
    return {
        "low_onset_strength": [],
        "mid_onset_strength": [],
        "high_onset_strength": [],
        "spectral_flux": [],
        "brightness": [],
        "harmonic_novelty": [],
        "texture_novelty": [],
        "percussive_ratio": [],
    }


def _map_spectral_frames(
    raw: AudioAnalysisRaw,
    *,
    analysis_start: float,
    grid_length: float,
    grids_per_bar: int,
    bar_count: int,
    frames_by_bar: list[dict[str, list[float]]],
    grids_by_bar: list[dict[int, dict[str, float]]],
) -> None:
    envelopes = {
        "low_onset_strength": raw.spectral.low_onset_envelope,
        "mid_onset_strength": raw.spectral.mid_onset_envelope,
        "high_onset_strength": raw.spectral.high_onset_envelope,
        "spectral_flux": raw.spectral.spectral_flux_envelope,
        "brightness": raw.spectral.brightness_envelope,
        "harmonic_novelty": raw.spectral.harmonic_novelty_envelope,
        "texture_novelty": raw.spectral.texture_novelty_envelope,
        "percussive_ratio": raw.spectral.percussive_ratio_envelope,
    }
    frame_count = max((len(values) for values in envelopes.values()), default=0)
    for frame_index in range(frame_count):
        frame_time = frame_index * raw.hop_length / raw.sample_rate
        bar_index, grid_index = _time_to_bar_grid(
            frame_time,
            analysis_start=analysis_start,
            grid_length=grid_length,
            grids_per_bar=grids_per_bar,
            bar_count=bar_count,
        )
        if bar_index is None or grid_index is None:
            continue

        frame_values = {
            name: _spectral_value(values, frame_index)
            for name, values in envelopes.items()
        }
        for name, value in frame_values.items():
            frames_by_bar[bar_index][name].append(value)

        grid_values = grids_by_bar[bar_index].setdefault(grid_index, {})
        for name in (
            "low_onset_strength",
            "mid_onset_strength",
            "high_onset_strength",
            "spectral_flux",
        ):
            grid_values[name] = max(grid_values.get(name, 0.0), frame_values[name])


def _spectral_value(values: list[float], index: int) -> float:
    if index >= len(values):
        return 0.0
    value = float(values[index])
    if not isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _spectral_bar_summary(frames: dict[str, list[float]]) -> dict[str, float]:
    return {
        "low_onset_strength": _rounded_max(frames["low_onset_strength"]),
        "mid_onset_strength": _rounded_max(frames["mid_onset_strength"]),
        "high_onset_strength": _rounded_max(frames["high_onset_strength"]),
        "spectral_flux": _rounded_max(frames["spectral_flux"]),
        "brightness": _rounded_mean(frames["brightness"]),
        "harmonic_novelty": _rounded_max(frames["harmonic_novelty"]),
        "texture_novelty": _rounded_max(frames["texture_novelty"]),
        "percussive_ratio": _rounded_mean(frames["percussive_ratio"]),
    }


def _spectral_grid_features(
    grids: dict[int, dict[str, float]],
) -> list[SpectralGridFeature]:
    features: list[SpectralGridFeature] = []
    for grid, values in sorted(grids.items()):
        strongest = max(values.values(), default=0.0)
        if strongest < SPECTRAL_GRID_MINIMUM:
            continue
        features.append(
            SpectralGridFeature(
                grid=grid,
                low_onset_strength=round(values.get("low_onset_strength", 0.0), 3),
                mid_onset_strength=round(values.get("mid_onset_strength", 0.0), 3),
                high_onset_strength=round(values.get("high_onset_strength", 0.0), 3),
                spectral_flux=round(values.get("spectral_flux", 0.0), 3),
            )
        )
    return features


def _rounded_max(values: list[float]) -> float:
    return round(max(values, default=0.0), 3)


def _rounded_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _time_to_bar_index(
    time: float,
    *,
    analysis_start: float,
    bar_length: float,
    bar_count: int,
) -> int | None:
    relative_time = time - analysis_start
    if relative_time < 0 or bar_length <= 0:
        return None

    bar_index = int(relative_time / bar_length)
    if bar_index < 0 or bar_index >= bar_count:
        return None
    return bar_index


def _global_peak_rms(rms_envelope: list[float]) -> float | None:
    values = [float(value) for value in rms_envelope if isfinite(value) and value >= 0]
    if not values:
        return None
    return max(values)


def _bar_loudness_metrics(
    rms_frames: list[float],
    global_peak_rms: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    values = [float(value) for value in rms_frames if isfinite(value) and value >= 0]
    if not values or global_peak_rms is None or not isfinite(global_peak_rms):
        return None, None, None, None

    bar_rms = sqrt(sum(value * value for value in values) / len(values))
    peak_rms = max(values)
    rms_dbfs = _to_dbfs(bar_rms)
    peak_rms_dbfs = _to_dbfs(peak_rms)
    relative_rms_db = rms_dbfs - _to_dbfs(global_peak_rms)
    if peak_rms <= LOUDNESS_EPSILON:
        sustained_activity_ratio = 0.0
    else:
        sustained_threshold = peak_rms * 10 ** (-SUSTAINED_ACTIVITY_DB_RANGE / 20.0)
        sustained_activity_ratio = sum(value >= sustained_threshold for value in values) / len(values)

    return (
        round(rms_dbfs, 3),
        round(peak_rms_dbfs, 3),
        round(relative_rms_db, 3),
        round(sustained_activity_ratio, 3),
    )


def _to_dbfs(value: float) -> float:
    return 20.0 * log10(max(value, LOUDNESS_EPSILON))


def _time_to_bar_grid(
    time: float,
    *,
    analysis_start: float,
    grid_length: float,
    grids_per_bar: int,
    bar_count: int,
) -> tuple[int | None, int | None]:
    relative_time = time - analysis_start
    if relative_time < 0:
        return None, None

    absolute_grid = round(relative_time / grid_length)
    bar_index, grid_index = divmod(absolute_grid, grids_per_bar)
    if bar_index < 0 or bar_index >= bar_count:
        return None, None

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


def _map_meter_beats_to_grids(
    *,
    meter: MeterSpec,
    bar_count: int,
) -> tuple[list[dict[int, int]], list[set[int]]]:
    beat_numbers = _meter_beat_numbers(meter)
    return [beat_numbers.copy() for _ in range(bar_count)], [{0} for _ in range(bar_count)]


def _meter_beat_numbers(meter: MeterSpec) -> dict[int, int]:
    return {grid: beat_number for beat_number, grid in enumerate(meter.beat_grids, start=1)}


def _strongest_onset_grids(strengths: dict[int, float]) -> list[int]:
    if not strengths:
        return []
    max_strength = max(strengths.values())
    return sorted(grid for grid, strength in strengths.items() if strength == max_strength)


def _activity_values(activities: dict[int, float], grids_per_bar: int) -> list[float]:
    return [round(activities.get(grid, 0.0), 3) for grid in range(grids_per_bar)]


def _build_grid_features(
    *,
    grids_per_bar: int,
    onset_grids: set[int],
    accent_grids: set[int],
    strengths: dict[int, float],
    activities: dict[int, float],
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
            strength=max(strengths.get(grid, 0.0), activities.get(grid, 0.0)),
            activity=activities.get(grid, 0.0),
        )
        for grid in range(grids_per_bar)
    ]


def _phrase_position(index: int, bar_count: int) -> str:
    return "song_end" if index == bar_count - 1 else "unknown"
