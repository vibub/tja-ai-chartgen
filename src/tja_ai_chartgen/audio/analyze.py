from pathlib import Path
from typing import Any

import librosa
import numpy as np
from pydantic import BaseModel, Field


MIN_DETECTED_BPM = 89.0
MAX_DETECTED_BPM = 220.0
BPM_SCAN_STEP = 0.5
PHASE_BIN_SECONDS = 0.005
PHASE_WINDOW_SECONDS = 0.045


class AudioAnalysisRaw(BaseModel):
    bpm: float
    beat_times: list[float]
    onset_times: list[float]
    onset_strengths: list[float]
    duration: float
    offset: float
    sample_rate: int | None = None
    hop_length: int = 512
    downbeat_times: list[float] = Field(default_factory=list)
    beat_numbers: list[int] = Field(default_factory=list)
    time_signature: str = "4/4"
    analyzer: str = "librosa"


def normalize_bpm(bpm: float) -> float:
    if bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm}")

    while bpm < MIN_DETECTED_BPM:
        bpm *= 2
    while bpm > MAX_DETECTED_BPM:
        bpm /= 2
    return round(bpm, 3)


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
) -> tuple[float, float] | None:
    if len(onset_times) < 4 or len(onset_times) != len(weights):
        return None

    onset_array = np.asarray(onset_times, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(onset_array) & np.isfinite(weight_array) & (onset_array >= 0) & (weight_array > 0)
    onset_array = onset_array[valid]
    weight_array = weight_array[valid]
    if onset_array.size < 4:
        return None

    candidates = np.arange(MIN_DETECTED_BPM, MAX_DETECTED_BPM + BPM_SCAN_STEP, BPM_SCAN_STEP)
    best_bpm = normalize_bpm(fallback_bpm)
    best_offset = 0.0
    best_score = -1.0
    fallback_normalized = normalize_bpm(fallback_bpm)

    for bpm in candidates:
        interval = 60.0 / float(bpm)
        score, offset = _phase_confidence(onset_array, weight_array, interval)
        if score > best_score or (
            np.isclose(score, best_score, rtol=0.01)
            and abs(bpm - fallback_normalized) < abs(best_bpm - fallback_normalized)
        ):
            best_score = score
            best_bpm = float(bpm)
            best_offset = offset

    if best_score <= 0:
        return None

    adjusted_offset = _adjust_offset_for_offbeats(samples, sample_rate, best_offset, best_bpm)
    return round(float(best_bpm), 3), round(float(adjusted_offset), 6)


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
    offset = (best_bin / bin_count) * interval
    return score, float(offset)


def _adjust_offset_for_offbeats(
    samples: np.ndarray,
    sample_rate: int,
    offset: float,
    bpm: float,
) -> float:
    if samples.size == 0 or sample_rate <= 0 or bpm <= 0:
        return offset

    seconds_per_beat = 60.0 / bpm
    offbeat = offset + (seconds_per_beat * 0.5)
    if offbeat > seconds_per_beat:
        offbeat -= seconds_per_beat

    support_a = _slope_support(samples, sample_rate, offset, seconds_per_beat)
    support_b = _slope_support(samples, sample_rate, offbeat, seconds_per_beat)
    return offset if support_a >= support_b else offbeat


def _slope_support(samples: np.ndarray, sample_rate: int, offset: float, interval: float) -> float:
    abs_samples = np.abs(samples)
    half_window = max(1, sample_rate // 20)
    if abs_samples.size < half_window * 2:
        return 0.0

    prefix = np.concatenate([[0.0], np.cumsum(abs_samples, dtype=float)])
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
        if time >= 0:
            beat_times.append(round(time, 6))
        time += interval
    return beat_times


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

    return raw.model_copy(update=updates)


def analyze_audio(input_path: Path, use_beatnet: bool = False) -> AudioAnalysisRaw:
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    y, sr = librosa.load(str(input_path), sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
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
    refined = _estimate_tempo_and_offset_from_onsets(
        onset_times_list,
        _onset_weights(onset_times_list, onset_env, sample_rate=int(sr), hop_length=hop_length),
        y,
        sample_rate=int(sr),
        fallback_bpm=tempo_value,
    )
    if refined is not None:
        bpm, offset = refined
        beat_times = _regular_beat_times(offset, bpm, duration)
        analyzer = "librosa+onset-grid"

    raw = AudioAnalysisRaw(
        bpm=bpm,
        beat_times=[float(value) for value in beat_times],
        onset_times=onset_times_list,
        onset_strengths=[float(value) for value in onset_env.tolist()],
        duration=duration,
        offset=offset,
        sample_rate=int(sr),
        hop_length=hop_length,
        analyzer=analyzer,
    )

    if not use_beatnet:
        return raw

    return enhance_with_beatnet(input_path, raw)


def enhance_with_beatnet(input_path: Path, raw: AudioAnalysisRaw) -> AudioAnalysisRaw:
    try:
        from BeatNet.BeatNet import BeatNet
    except ImportError:
        return raw

    try:
        estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
        output = estimator.process(str(input_path))
    except Exception:  # noqa: BLE001 - BeatNet is an optional enhancement.
        return raw

    return merge_beatnet_output(raw, output)


def merge_beatnet_output(raw: AudioAnalysisRaw, output: Any) -> AudioAnalysisRaw:
    rows = np.asarray(output)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
        return raw

    beat_times: list[float] = []
    beat_numbers: list[int] = []
    for row in rows:
        time = float(row[0])
        beat_number = int(round(float(row[1])))
        if not np.isfinite(time) or beat_number < 1:
            continue
        beat_times.append(time)
        beat_numbers.append(beat_number)

    if not beat_times:
        return raw

    downbeat_times = [time for time, number in zip(beat_times, beat_numbers, strict=True) if number == 1]
    updates: dict[str, Any] = {
        "beat_times": beat_times,
        "beat_numbers": beat_numbers,
        "downbeat_times": downbeat_times,
        "offset": downbeat_times[0] if downbeat_times else beat_times[0],
        "time_signature": estimate_time_signature(beat_numbers),
        "analyzer": "beatnet+librosa",
    }

    positive_intervals = [
        later - earlier
        for earlier, later in zip(beat_times, beat_times[1:], strict=False)
        if later > earlier
    ]
    if positive_intervals:
        updates["bpm"] = normalize_bpm(60.0 / float(np.mean(positive_intervals)))

    return raw.model_copy(update=updates)


def estimate_time_signature(beat_numbers: list[int]) -> str:
    if not beat_numbers:
        return "4/4"

    max_beat_number = max(beat_numbers)
    if max_beat_number == 3:
        return "3/4"
    if max_beat_number == 6:
        return "6/8"
    return "4/4"
