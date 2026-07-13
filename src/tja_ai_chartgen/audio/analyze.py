from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from pydantic import BaseModel, Field

from tja_ai_chartgen.audio.spectral import SpectralAnalysisRaw, extract_spectral_features
from tja_ai_chartgen.tja.model import TempoAnalysisDecision


MIN_DETECTED_BPM = 89.0
MAX_DETECTED_BPM = 220.0
BPM_SCAN_STEP = 0.5
PHASE_BIN_SECONDS = 0.005
PHASE_WINDOW_SECONDS = 0.045
WAVEFORM_REFINE_WINDOW_SECONDS = 0.035
WAVEFORM_REFINE_STEP_SECONDS = 0.002
MIN_ONSET_GRID_ONSETS = 8
MIN_ONSET_GRID_SPAN_SECONDS = 4.0
MIN_ONSET_GRID_TIME_COVERAGE = 0.5
MIN_ONSET_GRID_SUPPORT = 0.75
MAX_ONSET_GRID_RUNNER_UP_RATIO = 0.9
MIN_DISTINCT_BPM_DISTANCE = 3.0
MIN_DISTINCT_BPM_RATIO = 0.03


@dataclass(frozen=True)
class TempoOffsetEstimate:
    bpm: float
    offset: float
    normalized_support: float
    onset_count: int
    time_coverage: float
    runner_up_bpm: float | None
    runner_up_support: float | None
    accepted: bool
    reason: str

    def to_decision(self, fallback_source: str) -> TempoAnalysisDecision:
        return TempoAnalysisDecision(
            fallback_source=fallback_source,
            selected_source="onset-grid" if self.accepted else fallback_source,
            estimated_bpm=self.bpm,
            estimated_offset=self.offset,
            normalized_support=self.normalized_support,
            onset_count=self.onset_count,
            time_coverage=self.time_coverage,
            runner_up_bpm=self.runner_up_bpm,
            runner_up_support=self.runner_up_support,
            accepted=self.accepted,
            reason=self.reason,
        )


class AudioAnalysisRaw(BaseModel):
    bpm: float
    beat_times: list[float]
    onset_times: list[float]
    onset_strengths: list[float]
    duration: float
    offset: float
    activity_envelope: list[float] = Field(default_factory=list)
    rms_envelope: list[float] = Field(default_factory=list)
    sample_rate: int | None = None
    hop_length: int = 512
    downbeat_times: list[float] = Field(default_factory=list)
    beat_numbers: list[int] = Field(default_factory=list)
    time_signature: str = "4/4"
    analyzer: str = "librosa"
    tempo_analysis: TempoAnalysisDecision | None = None
    spectral: SpectralAnalysisRaw = Field(default_factory=SpectralAnalysisRaw)


def normalize_bpm(bpm: float) -> float:
    if bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm}")

    while bpm < MIN_DETECTED_BPM:
        bpm *= 2
    while bpm > MAX_DETECTED_BPM:
        bpm /= 2
    return round(bpm, 3)


def _beatnet_quarter_note_bpm(pulse_bpm: float, time_signature: str) -> float:
    if time_signature == "6/8":
        pulse_bpm /= 2.0
    return normalize_bpm(pulse_bpm)


def _keep_compound_meter_quarter_note_tempo(
    estimate: TempoOffsetEstimate,
    fallback_bpm: float,
    time_signature: str,
) -> TempoOffsetEstimate:
    if (
        time_signature != "6/8"
        or not estimate.accepted
        or abs(estimate.bpm - (fallback_bpm * 2.0)) > BPM_SCAN_STEP
    ):
        return estimate

    return TempoOffsetEstimate(
        bpm=round(fallback_bpm, 3),
        offset=estimate.offset,
        normalized_support=estimate.normalized_support,
        onset_count=estimate.onset_count,
        time_coverage=estimate.time_coverage,
        runner_up_bpm=estimate.runner_up_bpm,
        runner_up_support=estimate.runner_up_support,
        accepted=True,
        reason="accepted_compound_meter_alias",
    )


def _fallback_resolves_double_tempo_alias(
    best_bpm: float,
    runner_up_bpm: float,
    fallback_bpm: float,
) -> bool:
    slower_bpm, faster_bpm = sorted((best_bpm, runner_up_bpm))
    if abs(faster_bpm - (slower_bpm * 2.0)) > BPM_SCAN_STEP:
        return False
    return abs(best_bpm - fallback_bpm) + BPM_SCAN_STEP < abs(runner_up_bpm - fallback_bpm)


def _rms_envelopes(samples: np.ndarray, *, hop_length: int) -> tuple[list[float], list[float]]:
    rms = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
    if rms.size == 0:
        return [], []

    rms_envelope = [
        max(0.0, float(value)) if np.isfinite(value) else 0.0
        for value in rms
    ]
    max_rms = max(rms_envelope, default=0.0)
    if max_rms <= 0:
        return [0.0 for _ in rms_envelope], rms_envelope
    return [round(value / max_rms, 3) for value in rms_envelope], rms_envelope


def _onset_weights(
    onset_times: list[float],
    onset_envelope: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> list[float]:
    if onset_envelope.size == 0:
        return [1.0 for _ in onset_times]

    weights: list[float] = []
    for onset_time in onset_times:
        frame_index = round(onset_time * sample_rate / hop_length)
        frame_index = max(0, min(len(onset_envelope) - 1, frame_index))
        weights.append(float(max(0.0, onset_envelope[frame_index])))

    max_weight = max(weights, default=0.0)
    if max_weight <= 0:
        return [1.0 for _ in onset_times]
    return [0.1 + (weight / max_weight) for weight in weights]


def _estimate_tempo_and_offset_from_onsets(
    onset_times: list[float],
    weights: list[float],
    samples: np.ndarray,
    *,
    sample_rate: int,
    fallback_bpm: float,
    duration: float,
) -> TempoOffsetEstimate:
    fallback_normalized = (
        normalize_bpm(fallback_bpm)
        if np.isfinite(fallback_bpm) and fallback_bpm > 0
        else 0.0
    )

    def rejected(
        reason: str,
        *,
        onset_count: int = 0,
        time_coverage: float = 0.0,
        bpm: float = fallback_normalized,
        offset: float = 0.0,
        normalized_support: float = 0.0,
        runner_up_bpm: float | None = None,
        runner_up_support: float | None = None,
    ) -> TempoOffsetEstimate:
        return TempoOffsetEstimate(
            bpm=round(float(bpm), 3),
            offset=round(float(offset), 6),
            normalized_support=round(float(normalized_support), 6),
            onset_count=onset_count,
            time_coverage=round(float(time_coverage), 6),
            runner_up_bpm=runner_up_bpm,
            runner_up_support=(
                round(float(runner_up_support), 6) if runner_up_support is not None else None
            ),
            accepted=False,
            reason=reason,
        )

    if len(onset_times) != len(weights) or not np.isfinite(duration) or duration <= 0:
        return rejected("invalid_input")

    try:
        onset_array = np.asarray(onset_times, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
    except (TypeError, ValueError):
        return rejected("invalid_input")
    if onset_array.ndim != 1 or weight_array.ndim != 1 or onset_array.shape != weight_array.shape:
        return rejected("invalid_input")

    valid = np.isfinite(onset_array) & np.isfinite(weight_array) & (onset_array >= 0) & (weight_array > 0)
    onset_array = onset_array[valid]
    weight_array = weight_array[valid]
    onset_count = int(onset_array.size)
    time_span = float(np.ptp(onset_array)) if onset_count >= 2 else 0.0
    time_coverage = min(1.0, max(0.0, time_span / duration))

    if onset_count < MIN_ONSET_GRID_ONSETS:
        return rejected(
            "insufficient_onsets",
            onset_count=onset_count,
            time_coverage=time_coverage,
        )
    if time_span < MIN_ONSET_GRID_SPAN_SECONDS:
        return rejected(
            "insufficient_time_span",
            onset_count=onset_count,
            time_coverage=time_coverage,
        )
    if time_coverage < MIN_ONSET_GRID_TIME_COVERAGE:
        return rejected(
            "insufficient_time_coverage",
            onset_count=onset_count,
            time_coverage=time_coverage,
        )

    total_weight = float(np.sum(weight_array))
    if not np.isfinite(total_weight) or total_weight <= 0:
        return rejected(
            "invalid_input",
            onset_count=onset_count,
            time_coverage=time_coverage,
        )

    candidate_results: list[tuple[float, float, float]] = []
    candidates = np.arange(MIN_DETECTED_BPM, MAX_DETECTED_BPM + BPM_SCAN_STEP, BPM_SCAN_STEP)
    best_bpm = fallback_normalized
    best_offset = 0.0
    best_support = -1.0

    for bpm in candidates:
        interval = 60.0 / float(bpm)
        score, offset = _phase_confidence(onset_array, weight_array, interval)
        normalized_support = min(1.0, max(0.0, float(score / total_weight)))
        candidate_results.append((normalized_support, float(bpm), float(offset)))
        if normalized_support > best_support or (
            np.isclose(normalized_support, best_support, rtol=0.01)
            and abs(bpm - fallback_normalized) < abs(best_bpm - fallback_normalized)
        ):
            best_support = normalized_support
            best_bpm = float(bpm)
            best_offset = float(offset)

    if best_support <= 0:
        return rejected(
            "low_normalized_support",
            onset_count=onset_count,
            time_coverage=time_coverage,
        )

    distinct_distance = max(MIN_DISTINCT_BPM_DISTANCE, best_bpm * MIN_DISTINCT_BPM_RATIO)
    runner_up = max(
        (
            result
            for result in candidate_results
            if abs(result[1] - best_bpm) >= distinct_distance
        ),
        default=None,
        key=lambda result: result[0],
    )
    runner_up_support = runner_up[0] if runner_up is not None else None
    runner_up_bpm = runner_up[1] if runner_up is not None else None

    if (
        runner_up_support is not None
        and runner_up_bpm is not None
        and runner_up_support / best_support >= MAX_ONSET_GRID_RUNNER_UP_RATIO
        and not _fallback_resolves_double_tempo_alias(
            best_bpm,
            runner_up_bpm,
            fallback_normalized,
        )
    ):
        return rejected(
            "ambiguous_candidates",
            onset_count=onset_count,
            time_coverage=time_coverage,
            bpm=best_bpm,
            offset=best_offset,
            normalized_support=best_support,
            runner_up_bpm=runner_up_bpm,
            runner_up_support=runner_up_support,
        )
    if best_support < MIN_ONSET_GRID_SUPPORT:
        return rejected(
            "low_normalized_support",
            onset_count=onset_count,
            time_coverage=time_coverage,
            bpm=best_bpm,
            offset=best_offset,
            normalized_support=best_support,
            runner_up_bpm=runner_up_bpm,
            runner_up_support=runner_up_support,
        )

    adjusted_offset = _adjust_offset_for_offbeats(samples, sample_rate, best_offset, best_bpm)
    return TempoOffsetEstimate(
        bpm=round(float(best_bpm), 3),
        offset=round(float(adjusted_offset), 6),
        normalized_support=round(float(best_support), 6),
        onset_count=onset_count,
        time_coverage=round(float(time_coverage), 6),
        runner_up_bpm=runner_up_bpm,
        runner_up_support=(
            round(float(runner_up_support), 6) if runner_up_support is not None else None
        ),
        accepted=True,
        reason="accepted",
    )


def _phase_confidence(onset_times: np.ndarray, weights: np.ndarray, interval: float) -> tuple[float, float]:
    bin_count = max(24, min(2048, int(round(interval / PHASE_BIN_SECONDS))))
    phases = np.mod(onset_times, interval)
    bins = np.floor(phases / interval * bin_count).astype(int)
    bins = np.clip(bins, 0, bin_count - 1)
    histogram = np.bincount(bins, weights=weights, minlength=bin_count)

    radius = max(1, int(round(PHASE_WINDOW_SECONDS / interval * bin_count)))
    kernel = np.hanning(radius * 2 + 3)[1:-1]
    kernel = kernel / np.max(kernel)
    padded = np.concatenate([histogram[-radius:], histogram, histogram[:radius]])
    smoothed = np.convolve(padded, kernel, mode="valid")

    best_bin = int(np.argmax(smoothed))
    offbeat_bin = (best_bin + (bin_count // 2)) % bin_count
    score = float(smoothed[best_bin] + (smoothed[offbeat_bin] * 0.5))
    coarse_offset = ((best_bin + 0.5) / bin_count) * interval
    offset = _weighted_circular_mean_phase(onset_times, weights, coarse_offset, interval)
    return score, float(offset)


def _weighted_circular_mean_phase(
    phases: np.ndarray,
    weights: np.ndarray,
    center: float,
    interval: float,
) -> float:
    if interval <= 0:
        return center

    distances = np.mod(phases - center + (interval * 0.5), interval) - (interval * 0.5)
    selected = np.abs(distances) <= PHASE_WINDOW_SECONDS
    if not np.any(selected):
        return center % interval

    selected_phases = phases[selected]
    selected_weights = weights[selected]
    full_turn = 2.0 * np.pi
    angles = selected_phases / interval * full_turn
    vector = np.sum(selected_weights * np.exp(1j * angles))
    if np.isclose(abs(vector), 0.0):
        return center % interval
    return float((np.angle(vector) % full_turn) / full_turn * interval)


def _precompute_waveform_support(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_samples = np.abs(samples)
    prefix = np.concatenate([[0.0], np.cumsum(abs_samples, dtype=float)])
    return abs_samples, prefix


def _adjust_offset_for_offbeats(
    samples: np.ndarray,
    sample_rate: int,
    offset: float,
    bpm: float,
) -> float:
    if samples.size == 0 or sample_rate <= 0 or bpm <= 0:
        return _canonical_phase(offset, 60.0 / bpm) if bpm > 0 else offset

    seconds_per_beat = 60.0 / bpm
    half_window = max(1, sample_rate // 20)
    if samples.size < half_window * 2:
        return _canonical_phase(offset, seconds_per_beat)

    abs_samples, prefix = _precompute_waveform_support(samples)
    offbeat = _canonical_phase(offset + (seconds_per_beat * 0.5), seconds_per_beat)

    refined_offset = _refine_offset_with_waveform(
        abs_samples,
        prefix,
        sample_rate,
        offset,
        seconds_per_beat,
    )
    refined_offbeat = _refine_offset_with_waveform(
        abs_samples,
        prefix,
        sample_rate,
        offbeat,
        seconds_per_beat,
    )
    support_a = _slope_support(abs_samples, prefix, sample_rate, refined_offset, seconds_per_beat)
    support_b = _slope_support(abs_samples, prefix, sample_rate, refined_offbeat, seconds_per_beat)
    return _canonical_phase(refined_offset if support_a >= support_b else refined_offbeat, seconds_per_beat)


def _canonical_phase(offset: float, interval: float) -> float:
    if interval <= 0:
        return offset
    phase = offset % interval
    if np.isclose(phase, interval) or np.isclose(phase, 0.0):
        return 0.0
    return float(phase)


def _refine_offset_with_waveform(
    abs_samples: np.ndarray,
    prefix: np.ndarray,
    sample_rate: int,
    offset: float,
    interval: float,
) -> float:
    if abs_samples.size == 0 or sample_rate <= 0 or interval <= 0:
        return offset

    best_offset = offset
    best_support = _slope_support(abs_samples, prefix, sample_rate, offset, interval)
    for delta in np.arange(
        -WAVEFORM_REFINE_WINDOW_SECONDS,
        WAVEFORM_REFINE_WINDOW_SECONDS + WAVEFORM_REFINE_STEP_SECONDS,
        WAVEFORM_REFINE_STEP_SECONDS,
    ):
        candidate = (offset + float(delta)) % interval
        support = _slope_support(abs_samples, prefix, sample_rate, candidate, interval)
        if support > best_support:
            best_offset = candidate
            best_support = support
    return best_offset


def _slope_support(
    abs_samples: np.ndarray,
    prefix: np.ndarray,
    sample_rate: int,
    offset: float,
    interval: float,
) -> float:
    half_window = max(1, sample_rate // 20)
    if abs_samples.size < half_window * 2:
        return 0.0

    total = 0.0
    position = offset * sample_rate
    step = interval * sample_rate
    while position + half_window < abs_samples.size:
        center = int(round(position))
        if center >= half_window:
            left = prefix[center] - prefix[center - half_window]
            right = prefix[center + half_window] - prefix[center]
            total += max(0.0, (right - left) / half_window)
        position += step
    return float(total)


def _regular_beat_times(offset: float, bpm: float, duration: float) -> list[float]:
    if bpm <= 0 or duration <= 0:
        return []

    interval = 60.0 / bpm
    beat_times: list[float] = []
    time = offset
    while time <= duration + (interval * 0.5):
        beat_times.append(round(time, 6))
        time += interval
    return beat_times


def _regular_beat_numbers(beat_count: int, time_signature: str) -> list[int]:
    beats_per_bar = 3 if time_signature in {"3/4", "6/8"} else 4
    return [(index % beats_per_bar) + 1 for index in range(beat_count)]


def _nearest_regular_time(reference: float, phase: float, interval: float) -> float:
    if interval <= 0:
        return reference
    step = round((reference - phase) / interval)
    return round(phase + (step * interval), 6)


def _weights_from_raw_onsets(raw: AudioAnalysisRaw) -> list[float]:
    if not raw.onset_strengths:
        return [1.0 for _ in raw.onset_times]

    max_strength = max(raw.onset_strengths, default=0.0)
    if max_strength <= 0:
        return [1.0 for _ in raw.onset_times]

    weights: list[float] = []
    for onset_time in raw.onset_times:
        if raw.sample_rate:
            frame_index = round(onset_time * raw.sample_rate / raw.hop_length)
        elif raw.duration > 0:
            frame_index = round((onset_time / raw.duration) * (len(raw.onset_strengths) - 1))
        else:
            frame_index = 0
        frame_index = max(0, min(len(raw.onset_strengths) - 1, frame_index))
        weights.append(0.1 + (max(0.0, raw.onset_strengths[frame_index]) / max_strength))
    return weights


def apply_analysis_overrides(
    raw: AudioAnalysisRaw,
    bpm: float | None = None,
    offset: float | None = None,
) -> AudioAnalysisRaw:
    updates: dict[str, float] = {}

    if bpm is not None:
        if bpm <= 0:
            raise ValueError(f"BPM must be positive, got {bpm}")
        updates["bpm"] = round(bpm, 3)

    if offset is not None:
        updates["offset"] = offset

    if not updates:
        return raw

    if not raw.analyzer.endswith("+manual-override"):
        updates["analyzer"] = f"{raw.analyzer}+manual-override"
    return raw.model_copy(update=updates)


def analyze_audio(input_path: Path, use_beatnet: bool = False) -> AudioAnalysisRaw:
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    y, sr = librosa.load(str(input_path), sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    activity_env, rms_env = _rms_envelopes(y, hop_length=hop_length)
    spectral = extract_spectral_features(
        y,
        sample_rate=int(sr),
        hop_length=hop_length,
    )
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        hop_length=hop_length,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length).tolist()
    onset_times_list = [float(value) for value in onset_times]

    bpm = normalize_bpm(tempo_value)
    offset = float(beat_times[0]) if beat_times else 0.0
    analyzer = "librosa"
    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times_list,
        _onset_weights(onset_times_list, onset_env, sample_rate=int(sr), hop_length=hop_length),
        y,
        sample_rate=int(sr),
        fallback_bpm=tempo_value,
        duration=duration,
    )
    if estimate.accepted:
        bpm, offset = estimate.bpm, estimate.offset
        beat_times = _regular_beat_times(offset, bpm, duration)
        analyzer = "librosa+onset-grid"

    raw = AudioAnalysisRaw(
        bpm=bpm,
        beat_times=[float(value) for value in beat_times],
        onset_times=onset_times_list,
        onset_strengths=[float(value) for value in onset_env.tolist()],
        activity_envelope=activity_env,
        rms_envelope=rms_env,
        duration=duration,
        offset=offset,
        sample_rate=int(sr),
        hop_length=hop_length,
        analyzer=analyzer,
        tempo_analysis=estimate.to_decision("librosa"),
        spectral=spectral,
    )

    if not use_beatnet:
        return raw

    return enhance_with_beatnet(input_path, raw)


def enhance_with_beatnet(input_path: Path, raw: AudioAnalysisRaw) -> AudioAnalysisRaw:
    try:
        from BeatNet.BeatNet import BeatNet

        estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
        output = estimator.process(str(input_path))
        return merge_beatnet_output(raw, output)
    except Exception:  # noqa: BLE001 - BeatNet is an optional enhancement.
        return raw


def _normalize_beatnet_rows(output: Any) -> np.ndarray | None:
    try:
        rows = np.asarray(output, dtype=float)
    except (TypeError, ValueError):
        return None

    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
        return None
    return rows[:, :2]


def merge_beatnet_output(raw: AudioAnalysisRaw, output: Any) -> AudioAnalysisRaw:
    rows = _normalize_beatnet_rows(output)
    if rows is None:
        return raw

    beat_times: list[float] = []
    beat_numbers: list[int] = []
    for time_value, beat_number_value in rows:
        if not np.isfinite(time_value) or not np.isfinite(beat_number_value):
            continue
        time = float(time_value)
        beat_number = int(round(float(beat_number_value)))
        if beat_number < 1:
            continue
        beat_times.append(time)
        beat_numbers.append(beat_number)

    if not beat_times:
        return raw

    beatnet_beat_times = beat_times
    beatnet_beat_numbers = beat_numbers
    time_signature = estimate_time_signature(beatnet_beat_numbers)
    bpm = raw.bpm
    offset = beatnet_beat_times[0]
    positive_intervals = [
        later - earlier
        for earlier, later in zip(beat_times, beat_times[1:], strict=False)
        if later > earlier
    ]
    if positive_intervals:
        pulse_bpm = 60.0 / float(np.mean(positive_intervals))
        bpm = _beatnet_quarter_note_bpm(pulse_bpm, time_signature)

    estimate = _estimate_tempo_and_offset_from_onsets(
        raw.onset_times,
        _weights_from_raw_onsets(raw),
        np.asarray([], dtype=float),
        sample_rate=raw.sample_rate or 0,
        fallback_bpm=bpm,
        duration=raw.duration,
    )
    estimate = _keep_compound_meter_quarter_note_tempo(estimate, bpm, time_signature)
    if estimate.accepted:
        bpm, offset = estimate.bpm, estimate.offset

    beat_times = _regular_beat_times(offset, bpm, raw.duration)
    if not beat_times:
        return raw

    beat_numbers = _regular_beat_numbers(len(beat_times), time_signature)
    beat_interval = 60.0 / bpm
    beatnet_downbeats = [
        _nearest_regular_time(time, offset, beat_interval)
        for time, number in zip(beatnet_beat_times, beatnet_beat_numbers, strict=True)
        if number == 1
    ]
    beat_time_set = set(beat_times)
    downbeat_times = sorted({time for time in beatnet_downbeats if time in beat_time_set})
    if not downbeat_times:
        downbeat_times = [time for time, number in zip(beat_times, beat_numbers, strict=True) if number == 1]

    return raw.model_copy(
        update={
            "beat_times": beat_times,
            "beat_numbers": beat_numbers,
            "downbeat_times": downbeat_times,
            "offset": downbeat_times[0] if downbeat_times else offset,
            "time_signature": time_signature,
            "bpm": bpm,
            "analyzer": "beatnet+librosa+onset-grid" if estimate.accepted else "beatnet",
            "tempo_analysis": estimate.to_decision("beatnet"),
        }
    )


def estimate_time_signature(beat_numbers: list[int]) -> str:
    if not beat_numbers:
        return "4/4"

    max_beat_number = max(beat_numbers)
    if max_beat_number == 3:
        return "3/4"
    if max_beat_number == 6:
        return "6/8"
    return "4/4"
