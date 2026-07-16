from dataclasses import dataclass
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType
from typing import Any

import librosa
import numpy as np
from pydantic import BaseModel, Field

from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw
from tja_ai_chartgen.audio.spectral import SpectralAnalysisRaw, extract_spectral_features
from tja_ai_chartgen.tja.model import (
    TempoAnalysisDecision,
    TempoMeterCandidate,
    TempoMeterEvidence,
)


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
MIN_HIGH_EVIDENCE_ONSET_GRID_SUPPORT = 0.68
FULL_ONSET_GRID_EVIDENCE_COUNT = 64
FULL_ONSET_GRID_EVIDENCE_COVERAGE = 0.9
MAX_ONSET_GRID_RUNNER_UP_RATIO = 0.9
MIN_DISTINCT_BPM_DISTANCE = 3.0
MIN_DISTINCT_BPM_RATIO = 0.03
TEMPO_ARBITRATION_VERSION = "tempo-arbitration-v2"
MIN_TEMPO_CANDIDATE_SCORE = 0.45
MIN_BEATNET_INTERVAL_STABILITY = 0.8
MIN_BEATNET_NUMBER_COMPLETENESS = 0.7
MIN_BEATNET_METER_STABILITY = 0.5
MIN_BEATNET_METER_LENGTH_SCORE = 0.65
MIN_PARTIAL_BEATNET_DOWNBEAT_SUPPORT = 0.45
MIN_PARTIAL_BEATNET_TIME_COVERAGE = 0.5
MAX_BEATNET_ONSET_SUPPORT_DEFICIT = 0.08
TEMPO_AMBIGUITY_MARGIN = 0.04
TEMPO_ALIAS_AMBIGUITY_MARGIN = 0.08
TEMPO_AGREEMENT_RATIO = 0.03
OFFSET_AGREEMENT_SECONDS = 0.08
_BEATNET_IMPORT_LOCK = Lock()


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
    tempo_candidates: list[TempoMeterCandidate] = Field(default_factory=list)
    tempo_analysis: TempoAnalysisDecision | None = None
    beatnet_analysis_status: str = "unavailable"
    beatnet_analysis_reason: str | None = None
    spectral: SpectralAnalysisRaw = Field(default_factory=SpectralAnalysisRaw)
    instruments: InstrumentAnalysisRaw = Field(default_factory=InstrumentAnalysisRaw)


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


def _required_onset_grid_support(onset_count: int, time_coverage: float) -> float:
    onset_evidence = np.clip(
        (onset_count - MIN_ONSET_GRID_ONSETS)
        / (FULL_ONSET_GRID_EVIDENCE_COUNT - MIN_ONSET_GRID_ONSETS),
        0.0,
        1.0,
    )
    coverage_evidence = np.clip(
        (time_coverage - MIN_ONSET_GRID_TIME_COVERAGE)
        / (FULL_ONSET_GRID_EVIDENCE_COVERAGE - MIN_ONSET_GRID_TIME_COVERAGE),
        0.0,
        1.0,
    )
    evidence = float(min(onset_evidence, coverage_evidence))
    relaxation = MIN_ONSET_GRID_SUPPORT - MIN_HIGH_EVIDENCE_ONSET_GRID_SUPPORT
    return MIN_ONSET_GRID_SUPPORT - (relaxation * evidence)


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
    required_support = _required_onset_grid_support(onset_count, time_coverage)
    if best_support < required_support:
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


def _candidate_time_coverage(beat_times: list[float], duration: float) -> float:
    if len(beat_times) < 2 or duration <= 0:
        return 0.0
    span = max(0.0, max(beat_times) - min(beat_times))
    return round(min(1.0, span / duration), 6)


def _candidate_interval_stability(beat_times: list[float]) -> float:
    intervals = np.asarray(
        [
            later - earlier
            for earlier, later in zip(beat_times, beat_times[1:], strict=False)
            if later > earlier
        ],
        dtype=float,
    )
    if intervals.size < 2:
        return 0.0
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return 0.0
    relative_deviation = float(np.mean(np.abs(intervals - median_interval))) / median_interval
    return round(max(0.0, min(1.0, 1.0 - relative_deviation)), 6)


def _candidate_onset_support(
    onset_times: list[float],
    weights: list[float],
    *,
    bpm: float,
    offset: float,
) -> float:
    if bpm <= 0 or len(onset_times) != len(weights):
        return 0.0
    try:
        onset_array = np.asarray(onset_times, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
    except (TypeError, ValueError):
        return 0.0
    if onset_array.ndim != 1 or weight_array.ndim != 1 or onset_array.shape != weight_array.shape:
        return 0.0
    valid = np.isfinite(onset_array) & np.isfinite(weight_array) & (weight_array > 0)
    onset_array = onset_array[valid]
    weight_array = weight_array[valid]
    total_weight = float(np.sum(weight_array))
    if onset_array.size == 0 or total_weight <= 0:
        return 0.0

    interval = 60.0 / bpm
    phases = np.mod(onset_array - offset, interval)
    beat_distances = np.minimum(phases, interval - phases)
    offbeat_distances = np.abs(phases - (interval * 0.5))
    beat_support = np.clip(1.0 - (beat_distances / PHASE_WINDOW_SECONDS), 0.0, 1.0)
    offbeat_support = 0.5 * np.clip(
        1.0 - (offbeat_distances / PHASE_WINDOW_SECONDS),
        0.0,
        1.0,
    )
    support = np.maximum(beat_support, offbeat_support)
    return round(float(np.sum(weight_array * support) / total_weight), 6)


def _candidate_interval_statistics(
    beat_times: list[float],
) -> tuple[int, float | None, float | None]:
    intervals = np.asarray(
        [
            later - earlier
            for earlier, later in zip(beat_times, beat_times[1:], strict=False)
            if later > earlier
        ],
        dtype=float,
    )
    if intervals.size == 0:
        return 0, None, None
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0:
        return int(intervals.size), None, None
    coefficient = float(np.std(intervals) / mean_interval)
    return int(intervals.size), round(mean_interval, 6), round(coefficient, 6)


def _beat_number_completeness(beat_numbers: list[int], time_signature: str) -> float:
    expected_size = {"3/4": 3, "4/4": 4, "6/8": 6}[time_signature]
    if len(beat_numbers) < 2:
        return 0.0
    valid_ratio = sum(1 <= number <= expected_size for number in beat_numbers) / len(beat_numbers)
    matching_transitions = sum(
        later == (earlier % expected_size) + 1
        for earlier, later in zip(beat_numbers, beat_numbers[1:], strict=False)
        if 1 <= earlier <= expected_size
    )
    transition_ratio = matching_transitions / (len(beat_numbers) - 1)
    return round(valid_ratio * transition_ratio, 6)


def _meter_stability(beat_numbers: list[int], time_signature: str) -> float:
    expected_size = {"3/4": 3, "4/4": 4, "6/8": 6}[time_signature]
    downbeat_indexes = [index for index, number in enumerate(beat_numbers) if number == 1]
    if not downbeat_indexes:
        return 0.0
    cycles = [
        beat_numbers[start:end]
        for start, end in zip(downbeat_indexes, downbeat_indexes[1:], strict=False)
    ]
    trailing_cycle = beat_numbers[downbeat_indexes[-1] :]
    if len(trailing_cycle) == expected_size:
        cycles.append(trailing_cycle)
    if not cycles:
        return 0.0
    expected_cycle = list(range(1, expected_size + 1))
    return round(sum(cycle == expected_cycle for cycle in cycles) / len(cycles), 6)


def _meter_length_score(
    downbeat_times: list[float],
    *,
    bpm: float,
    time_signature: str,
) -> float:
    if bpm <= 0 or len(downbeat_times) < 2:
        return 0.0
    quarter_notes = 3 if time_signature in {"3/4", "6/8"} else 4
    expected_length = quarter_notes * 60.0 / bpm
    intervals = [
        later - earlier
        for earlier, later in zip(downbeat_times, downbeat_times[1:], strict=False)
        if later > earlier
    ]
    if not intervals or expected_length <= 0:
        return 0.0
    scores = [max(0.0, 1.0 - abs(value - expected_length) / expected_length) for value in intervals]
    return round(float(np.mean(scores)), 6)


def _envelope_support_at_times(
    times: list[float],
    envelope: list[float],
    *,
    sample_rate: int | None,
    hop_length: int,
    normalize: bool = False,
) -> float:
    if not times or not envelope or not sample_rate or sample_rate <= 0 or hop_length <= 0:
        return 0.0
    values = np.asarray(envelope, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return 0.0
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if normalize:
        maximum = float(np.max(values, initial=0.0))
        if maximum <= 0:
            return 0.0
        values = values / maximum
    supports: list[float] = []
    for time in times:
        center = round(time * sample_rate / hop_length)
        start = max(0, center - 1)
        end = min(values.size, center + 2)
        if start < end:
            supports.append(float(np.max(values[start:end], initial=0.0)))
    return round(float(np.mean(supports)), 6) if supports else 0.0


def _tempo_alias(
    onset_support: float,
    half_tempo_support: float,
    double_tempo_support: float,
) -> str:
    if onset_support < 0.5:
        return "none"
    threshold = onset_support * MAX_ONSET_GRID_RUNNER_UP_RATIO
    half_supported = half_tempo_support >= threshold
    double_supported = double_tempo_support >= threshold
    if half_supported and double_supported:
        return "both"
    if half_supported:
        return "half"
    if double_supported:
        return "double"
    return "none"


def _tempo_meter_evidence(
    *,
    bpm: float,
    offset: float,
    time_signature: str,
    beat_times: list[float],
    downbeat_times: list[float],
    onset_times: list[float],
    onset_weights: list[float],
    onset_strengths: list[float],
    spectral: SpectralAnalysisRaw,
    sample_rate: int | None,
    hop_length: int,
    beat_numbers: list[int] | None = None,
    onset_support: float | None = None,
    runner_up_bpm: float | None = None,
    runner_up_support: float | None = None,
) -> tuple[float, TempoMeterEvidence]:
    measured_onset_support = (
        _candidate_onset_support(onset_times, onset_weights, bpm=bpm, offset=offset)
        if onset_support is None
        else round(max(0.0, min(1.0, onset_support)), 6)
    )
    half_tempo_support = _candidate_onset_support(
        onset_times,
        onset_weights,
        bpm=bpm / 2.0,
        offset=offset,
    )
    double_tempo_support = _candidate_onset_support(
        onset_times,
        onset_weights,
        bpm=bpm * 2.0,
        offset=offset,
    )
    interval_count, mean_interval, interval_coefficient = _candidate_interval_statistics(
        beat_times
    )
    downbeat_onset_support = _envelope_support_at_times(
        downbeat_times,
        onset_strengths,
        sample_rate=sample_rate,
        hop_length=hop_length,
        normalize=True,
    )
    downbeat_low_support = _envelope_support_at_times(
        downbeat_times,
        spectral.low_onset_envelope,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    downbeat_percussive_support = _envelope_support_at_times(
        downbeat_times,
        spectral.percussive_ratio_envelope,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    downbeat_support = round(
        (downbeat_onset_support * 0.45)
        + (downbeat_low_support * 0.35)
        + (downbeat_percussive_support * 0.2),
        6,
    )
    comparison_support = (
        runner_up_support
        if runner_up_support is not None
        else max(half_tempo_support, double_tempo_support)
    )
    evidence = TempoMeterEvidence(
        onset_count=sum(
            np.isfinite(time) and np.isfinite(weight) and time >= 0 and weight > 0
            for time, weight in zip(onset_times, onset_weights, strict=False)
        ),
        runner_up_bpm=runner_up_bpm,
        runner_up_support=runner_up_support,
        onset_support_margin=round(measured_onset_support - comparison_support, 6),
        interval_count=interval_count,
        mean_interval_seconds=mean_interval,
        interval_coefficient_of_variation=interval_coefficient,
        beat_number_completeness=_beat_number_completeness(
            beat_numbers or [],
            time_signature,
        ),
        meter_stability=_meter_stability(beat_numbers or [], time_signature),
        downbeat_onset_support=downbeat_onset_support,
        downbeat_low_frequency_support=downbeat_low_support,
        downbeat_percussive_support=downbeat_percussive_support,
        downbeat_support=downbeat_support,
        meter_length_score=_meter_length_score(
            downbeat_times,
            bpm=bpm,
            time_signature=time_signature,
        ),
        half_tempo_support=half_tempo_support,
        double_tempo_support=double_tempo_support,
        tempo_alias=_tempo_alias(
            measured_onset_support,
            half_tempo_support,
            double_tempo_support,
        ),
    )
    return measured_onset_support, evidence


def _tempo_candidate(
    *,
    source: str,
    bpm: float,
    offset: float,
    time_signature: str,
    beat_times: list[float],
    downbeat_times: list[float] | None = None,
    onset_support: float = 0.0,
    time_coverage: float,
    evidence: TempoMeterEvidence | None = None,
    confidence: float = 0.0,
    accepted: bool,
    reason: str | None,
) -> TempoMeterCandidate:
    normalized_beats = [round(float(value), 6) for value in beat_times]
    normalized_downbeats = [round(float(value), 6) for value in downbeat_times or []]
    return TempoMeterCandidate(
        source=source,
        bpm=round(float(bpm), 3),
        offset=round(float(offset), 6),
        time_signature=time_signature,
        beat_times=normalized_beats,
        downbeat_times=normalized_downbeats,
        onset_support=round(max(0.0, min(1.0, float(onset_support))), 6),
        time_coverage=round(max(0.0, min(1.0, float(time_coverage))), 6),
        interval_stability=_candidate_interval_stability(normalized_beats),
        evidence=evidence or TempoMeterEvidence(),
        confidence=round(max(0.0, min(1.0, float(confidence))), 6),
        accepted=accepted,
        reason=reason,
    )


def _replace_tempo_candidates(
    existing: list[TempoMeterCandidate],
    additions: list[TempoMeterCandidate],
) -> list[TempoMeterCandidate]:
    replacement_sources = {candidate.source for candidate in additions}
    return [
        candidate for candidate in existing if candidate.source not in replacement_sources
    ] + additions


def _tempo_score_components(candidate: TempoMeterCandidate) -> dict[str, float]:
    evidence = candidate.evidence
    margin_quality = max(0.0, min(1.0, 0.5 + evidence.onset_support_margin))
    evidence_volume = min(1.0, evidence.onset_count / 32.0)
    sufficient_onsets = evidence.onset_count >= MIN_ONSET_GRID_ONSETS
    if candidate.source.startswith("beatnet"):
        if sufficient_onsets:
            weights = {
                "onset": 0.4,
                "coverage": 0.1,
                "interval": 0.15,
                "margin": 0.05,
                "beat_numbers": 0.1,
                "meter": 0.08,
                "meter_length": 0.05,
                "downbeat": 0.07,
            }
        else:
            weights = {
                "onset": 0.0,
                "coverage": 0.1,
                "interval": 0.35,
                "margin": 0.0,
                "beat_numbers": 0.25,
                "meter": 0.15,
                "meter_length": 0.1,
                "downbeat": 0.05,
            }
    elif sufficient_onsets:
        weights = {
            "onset": 0.6,
            "coverage": 0.15,
            "interval": 0.15,
            "margin": 0.05,
            "volume": 0.05,
        }
    else:
        weights = {
            "onset": 0.0,
            "coverage": 0.4,
            "interval": 0.6,
        }
    values = {
        "onset": candidate.onset_support,
        "coverage": candidate.time_coverage,
        "interval": candidate.interval_stability,
        "margin": margin_quality,
        "volume": evidence_volume,
        "beat_numbers": evidence.beat_number_completeness,
        "meter": evidence.meter_stability,
        "meter_length": evidence.meter_length_score,
        "downbeat": evidence.downbeat_support,
    }
    components = {
        key: round(values[key] * weight, 6)
        for key, weight in weights.items()
    }
    components["total"] = round(min(1.0, sum(components.values())), 6)
    return components


def _tempo_candidate_rejection(
    candidate: TempoMeterCandidate,
    *,
    baseline: TempoMeterCandidate,
) -> str | None:
    source = candidate.source
    if source == baseline.source:
        return None
    if source.endswith("+onset-grid") and not candidate.accepted:
        return f"local-rejection:{candidate.reason or 'unknown'}"
    if not source.startswith("beatnet"):
        return None

    evidence = candidate.evidence
    expected_size = {"3/4": 3, "4/4": 4, "6/8": 6}[candidate.time_signature]
    if evidence.interval_count < max(2, expected_size - 1):
        return "insufficient-tracker-intervals"
    if candidate.interval_stability < MIN_BEATNET_INTERVAL_STABILITY:
        return "unstable-beat-intervals"
    if evidence.beat_number_completeness < MIN_BEATNET_NUMBER_COMPLETENESS:
        return "incomplete-beat-numbers"
    if evidence.meter_stability < MIN_BEATNET_METER_STABILITY:
        return "unstable-meter"
    if (
        len(candidate.downbeat_times) >= 2
        and evidence.meter_length_score < MIN_BEATNET_METER_LENGTH_SCORE
    ):
        return "implausible-meter-length"
    if (
        evidence.onset_count >= MIN_ONSET_GRID_ONSETS
        and baseline.evidence.onset_count >= MIN_ONSET_GRID_ONSETS
        and candidate.onset_support + MAX_BEATNET_ONSET_SUPPORT_DEFICIT
        < baseline.onset_support
    ):
        return "onset-support-below-baseline"
    if candidate.confidence < MIN_TEMPO_CANDIDATE_SCORE:
        return "low-candidate-score"
    return None


def _tempo_candidates_disagree(
    first: TempoMeterCandidate,
    second: TempoMeterCandidate,
) -> bool:
    bpm_ratio = abs(first.bpm - second.bpm) / max(first.bpm, second.bpm)
    if bpm_ratio > TEMPO_AGREEMENT_RATIO or first.time_signature != second.time_signature:
        return True
    interval = 60.0 / max(first.bpm, second.bpm)
    phase_delta = abs((first.offset - second.offset) % interval)
    phase_delta = min(phase_delta, interval - phase_delta)
    return phase_delta > OFFSET_AGREEMENT_SECONDS


def _tempo_alias_pair(first: TempoMeterCandidate, second: TempoMeterCandidate) -> bool:
    slower, faster = sorted((first.bpm, second.bpm))
    return abs(faster - (slower * 2.0)) <= BPM_SCAN_STEP


def _tempo_source_priority(source: str, fallback_source: str) -> int:
    if source == fallback_source:
        return 3
    if source.endswith("+onset-grid"):
        return 2
    if source == "beatnet":
        return 1
    return 0


def arbitrate_tempo_candidates(
    candidates: list[TempoMeterCandidate],
    *,
    fallback_source: str,
) -> tuple[list[TempoMeterCandidate], TempoAnalysisDecision]:
    if not candidates:
        raise ValueError("tempo arbitration requires at least one candidate")
    baseline = next(
        (candidate for candidate in candidates if candidate.source == fallback_source),
        candidates[0],
    )
    scored: list[TempoMeterCandidate] = []
    rejections: dict[str, str] = {}
    for candidate in candidates:
        components = _tempo_score_components(candidate)
        scored_candidate = candidate.model_copy(
            update={
                "score_components": components,
                "confidence": components["total"],
            }
        )
        rejection = _tempo_candidate_rejection(scored_candidate, baseline=baseline)
        if rejection is not None:
            rejections[candidate.source] = rejection
            scored_candidate = scored_candidate.model_copy(
                update={"accepted": False, "selected": False, "reason": rejection}
            )
        else:
            scored_candidate = scored_candidate.model_copy(
                update={"accepted": True, "selected": False, "reason": "eligible"}
            )
        scored.append(scored_candidate)

    eligible = [candidate for candidate in scored if candidate.accepted]
    if not eligible:
        eligible = [
            baseline.model_copy(
                update={
                    "score_components": _tempo_score_components(baseline),
                    "confidence": _tempo_score_components(baseline)["total"],
                    "accepted": True,
                    "reason": "fallback-only",
                }
            )
        ]
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            candidate.confidence,
            _tempo_source_priority(candidate.source, fallback_source),
        ),
        reverse=True,
    )
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    conservative = next(
        (candidate for candidate in ranked if candidate.source == fallback_source),
        baseline,
    )
    local_refinement_source = (
        f"{fallback_source}+onset-grid"
        if not fallback_source.endswith("+onset-grid")
        else None
    )
    ambiguous = bool(
        local_refinement_source
        and "ambiguous_candidates"
        in rejections.get(local_refinement_source, "")
    )
    ambiguity_threshold = TEMPO_AMBIGUITY_MARGIN
    if runner_up is not None and _tempo_alias_pair(top, runner_up):
        ambiguity_threshold = TEMPO_ALIAS_AMBIGUITY_MARGIN
    if ambiguous:
        selected = conservative
    elif (
        runner_up is not None
        and _tempo_candidates_disagree(top, runner_up)
        and top.confidence - runner_up.confidence < ambiguity_threshold
    ):
        ambiguous = True
        selected = conservative
    else:
        selected = top

    updated: list[TempoMeterCandidate] = []
    for candidate in scored:
        if candidate.source == selected.source:
            updated.append(
                candidate.model_copy(
                    update={
                        "accepted": True,
                        "selected": True,
                        "reason": "ambiguous-fallback" if ambiguous else "selected",
                    }
                )
            )
        elif candidate.accepted:
            updated.append(
                candidate.model_copy(
                    update={
                        "selected": False,
                        "reason": "ambiguous" if ambiguous else "lower-score",
                    }
                )
            )
        else:
            updated.append(candidate)
    selected = next(candidate for candidate in updated if candidate.selected)
    ranked_updated = sorted(
        (candidate for candidate in updated if not candidate.selected),
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )
    decision_runner = ranked_updated[0] if ranked_updated else None
    score_margin = (
        round(max(0.0, selected.confidence - decision_runner.confidence), 6)
        if decision_runner is not None
        else None
    )
    decision_reason = "ambiguous-candidates" if ambiguous else "selected-by-score"
    if selected.source == fallback_source and not ambiguous:
        preferred_rejection = (
            rejections.get(decision_runner.source)
            if decision_runner is not None
            else None
        )
        if preferred_rejection is not None:
            decision_reason = preferred_rejection.removeprefix("local-rejection:")
    decision = TempoAnalysisDecision(
        decision_version=TEMPO_ARBITRATION_VERSION,
        fallback_source=fallback_source,
        selected_source=selected.source,
        tempo_source=selected.source,
        meter_source=selected.source,
        estimated_bpm=selected.bpm,
        estimated_offset=selected.offset,
        normalized_support=selected.onset_support,
        onset_count=selected.evidence.onset_count,
        time_coverage=selected.time_coverage,
        runner_up_bpm=selected.evidence.runner_up_bpm,
        runner_up_support=selected.evidence.runner_up_support,
        selected_score=selected.confidence,
        runner_up_source=decision_runner.source if decision_runner is not None else None,
        runner_up_score=decision_runner.confidence if decision_runner is not None else None,
        score_margin=score_margin,
        ambiguous=ambiguous,
        candidate_rejections=rejections,
        accepted=selected.source != fallback_source and not ambiguous,
        reason=decision_reason,
    )
    return updated, decision


def _raw_baseline_candidate(raw: AudioAnalysisRaw) -> TempoMeterCandidate:
    source = "librosa+onset-grid" if "onset-grid" in raw.analyzer else "librosa"
    onset_weights = _weights_from_raw_onsets(raw)
    onset_support, evidence = _tempo_meter_evidence(
        bpm=raw.bpm,
        offset=raw.offset,
        time_signature=raw.time_signature,
        beat_times=raw.beat_times,
        downbeat_times=raw.downbeat_times,
        onset_times=raw.onset_times,
        onset_weights=onset_weights,
        onset_strengths=raw.onset_strengths,
        spectral=raw.spectral,
        sample_rate=raw.sample_rate,
        hop_length=raw.hop_length,
        beat_numbers=raw.beat_numbers,
    )
    return _tempo_candidate(
        source=source,
        bpm=raw.bpm,
        offset=raw.offset,
        time_signature=raw.time_signature,
        beat_times=raw.beat_times,
        downbeat_times=raw.downbeat_times,
        onset_support=onset_support,
        time_coverage=_candidate_time_coverage(raw.beat_times, raw.duration),
        evidence=evidence,
        accepted=True,
        reason="baseline",
    )


def _beatnet_meter_is_reliable(
    candidate: TempoMeterCandidate,
    *,
    tempo_candidate: TempoMeterCandidate,
) -> bool:
    if candidate.source != "beatnet" or not candidate.downbeat_times:
        return False
    bpm_ratio = abs(candidate.bpm - tempo_candidate.bpm) / max(
        candidate.bpm,
        tempo_candidate.bpm,
    )
    if bpm_ratio > TEMPO_AGREEMENT_RATIO:
        return False

    evidence = candidate.evidence
    expected_size = {"3/4": 3, "4/4": 4, "6/8": 6}[candidate.time_signature]
    if evidence.interval_count < max(2, expected_size - 1):
        return False
    if candidate.interval_stability < MIN_BEATNET_INTERVAL_STABILITY:
        return False
    if evidence.beat_number_completeness < MIN_BEATNET_NUMBER_COMPLETENESS:
        return False
    if evidence.meter_stability < MIN_BEATNET_METER_STABILITY:
        return False
    if (
        len(candidate.downbeat_times) >= 2
        and evidence.meter_length_score < MIN_BEATNET_METER_LENGTH_SCORE
    ):
        return False
    if candidate.time_coverage < MIN_PARTIAL_BEATNET_TIME_COVERAGE:
        return False
    if evidence.downbeat_support < MIN_PARTIAL_BEATNET_DOWNBEAT_SUPPORT:
        return False

    interval = 60.0 / tempo_candidate.bpm
    phase_delta = abs((candidate.downbeat_times[0] - tempo_candidate.offset) % interval)
    phase_delta = min(phase_delta, interval - phase_delta)
    return (
        candidate.time_signature != tempo_candidate.time_signature
        or phase_delta > OFFSET_AGREEMENT_SECONDS
    )


def _partial_beatnet_meter_candidate(
    candidates: list[TempoMeterCandidate],
    decision: TempoAnalysisDecision,
    *,
    fallback_source: str,
) -> TempoMeterCandidate | None:
    if decision.ambiguous or decision.selected_source != fallback_source:
        return None
    tempo_candidate = next(
        candidate for candidate in candidates if candidate.source == decision.selected_source
    )
    reliable = [
        candidate
        for candidate in candidates
        if _beatnet_meter_is_reliable(candidate, tempo_candidate=tempo_candidate)
    ]
    if not reliable:
        return None
    return max(
        reliable,
        key=lambda candidate: (
            candidate.evidence.downbeat_support,
            candidate.evidence.meter_stability,
            candidate.evidence.beat_number_completeness,
            candidate.confidence,
        ),
    )


def _apply_tempo_arbitration(
    raw: AudioAnalysisRaw,
    candidates: list[TempoMeterCandidate],
    *,
    fallback_source: str,
) -> AudioAnalysisRaw:
    updated_candidates, decision = arbitrate_tempo_candidates(
        candidates,
        fallback_source=fallback_source,
    )
    selected = next(candidate for candidate in updated_candidates if candidate.selected)
    partial_meter = _partial_beatnet_meter_candidate(
        updated_candidates,
        decision,
        fallback_source=fallback_source,
    )
    meter_candidate = partial_meter or selected
    uses_meter = meter_candidate.source.startswith("beatnet")
    offset = (
        meter_candidate.downbeat_times[0]
        if uses_meter and meter_candidate.downbeat_times
        else selected.offset
    )
    beat_times = (
        _regular_beat_times(offset, selected.bpm, raw.duration)
        if uses_meter
        else selected.beat_times
    )
    if not beat_times:
        return raw.model_copy(
            update={"tempo_candidates": updated_candidates, "tempo_analysis": decision}
        )
    time_signature = meter_candidate.time_signature if uses_meter else selected.time_signature
    beat_numbers = (
        _regular_beat_numbers(len(beat_times), time_signature)
        if uses_meter
        else []
    )
    downbeat_times = (
        _project_downbeats_to_regular_grid(
            meter_candidate.downbeat_times,
            beat_times,
            offset,
            selected.bpm,
        )
        if uses_meter
        else []
    )
    if uses_meter and not downbeat_times:
        downbeat_times = [
            time
            for time, number in zip(beat_times, beat_numbers, strict=True)
            if number == 1
        ]
    analyzer = selected.source
    if partial_meter is not None:
        analyzer = f"{selected.source}+beatnet-meter"
        decision = decision.model_copy(
            update={
                "tempo_source": selected.source,
                "meter_source": partial_meter.source,
                "partial_adoption": True,
                "estimated_offset": downbeat_times[0] if downbeat_times else offset,
                "accepted": True,
                "reason": "adopted-beatnet-meter-downbeat",
            }
        )
    return raw.model_copy(
        update={
            "bpm": selected.bpm,
            "offset": downbeat_times[0] if downbeat_times else offset,
            "beat_times": beat_times,
            "beat_numbers": beat_numbers,
            "downbeat_times": downbeat_times,
            "time_signature": time_signature,
            "analyzer": analyzer,
            "tempo_candidates": updated_candidates,
            "tempo_analysis": decision,
        }
    )


def _regular_beat_numbers(beat_count: int, time_signature: str) -> list[int]:
    beats_per_bar = 3 if time_signature in {"3/4", "6/8"} else 4
    return [(index % beats_per_bar) + 1 for index in range(beat_count)]


def _nearest_regular_time(reference: float, phase: float, interval: float) -> float:
    if interval <= 0:
        return reference
    step = round((reference - phase) / interval)
    return round(phase + (step * interval), 6)


def _project_downbeats_to_regular_grid(
    downbeat_references: list[float],
    beat_times: list[float],
    offset: float,
    bpm: float,
) -> list[float]:
    if bpm <= 0 or not beat_times:
        return []
    beat_interval = 60.0 / bpm
    beat_time_set = set(beat_times)
    return sorted(
        {
            projected
            for reference in downbeat_references
            if (projected := _nearest_regular_time(reference, offset, beat_interval))
            in beat_time_set
        }
    )


def _instrument_support_at_times(
    times: list[float],
    instruments: InstrumentAnalysisRaw,
    field: str,
) -> float | None:
    if not times or not instruments.stem_frames:
        return None
    frame_times = np.asarray([frame.time for frame in instruments.stem_frames], dtype=float)
    frame_values = np.asarray(
        [max(0.0, min(1.0, float(getattr(frame, field)))) for frame in instruments.stem_frames],
        dtype=float,
    )
    frame_intervals = np.diff(frame_times)
    positive_intervals = frame_intervals[frame_intervals > 0]
    tolerance = (
        max(0.05, float(np.median(positive_intervals)) * 1.5)
        if positive_intervals.size
        else 0.1
    )
    supports: list[float] = []
    for time in times:
        insertion = int(np.searchsorted(frame_times, time, side="left"))
        nearest_candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < frame_times.size
        ]
        if not nearest_candidates:
            continue
        nearest = min(nearest_candidates, key=lambda index: abs(frame_times[index] - time))
        if abs(frame_times[nearest] - time) > tolerance:
            continue
        start = max(0, nearest - 1)
        end = min(frame_values.size, nearest + 2)
        supports.append(float(np.max(frame_values[start:end], initial=0.0)))
    return round(float(np.mean(supports)), 6) if supports else None


def enrich_tempo_candidates_with_instruments(
    candidates: list[TempoMeterCandidate],
    instruments: InstrumentAnalysisRaw,
) -> list[TempoMeterCandidate]:
    if instruments.status not in {"complete", "partial"} or not instruments.stem_frames:
        return candidates
    enriched: list[TempoMeterCandidate] = []
    for candidate in candidates:
        evidence = candidate.evidence.model_copy(
            update={
                "auxiliary_drum_onset_support": _instrument_support_at_times(
                    candidate.downbeat_times,
                    instruments,
                    "drum_onset",
                ),
                "auxiliary_bass_onset_support": _instrument_support_at_times(
                    candidate.downbeat_times,
                    instruments,
                    "bass_onset",
                ),
            }
        )
        enriched.append(candidate.model_copy(update={"evidence": evidence}))
    return enriched


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
    onset_weights = _onset_weights(
        onset_times_list,
        onset_env,
        sample_rate=int(sr),
        hop_length=hop_length,
    )
    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times_list,
        onset_weights,
        y,
        sample_rate=int(sr),
        fallback_bpm=tempo_value,
        duration=duration,
    )
    librosa_onset_support, librosa_evidence = _tempo_meter_evidence(
        bpm=bpm,
        offset=offset,
        time_signature="4/4",
        beat_times=[float(value) for value in beat_times],
        downbeat_times=[],
        onset_times=onset_times_list,
        onset_weights=onset_weights,
        onset_strengths=[float(value) for value in onset_env.tolist()],
        spectral=spectral,
        sample_rate=int(sr),
        hop_length=hop_length,
    )
    librosa_candidate = _tempo_candidate(
        source="librosa",
        bpm=bpm,
        offset=offset,
        time_signature="4/4",
        beat_times=[float(value) for value in beat_times],
        onset_support=librosa_onset_support,
        time_coverage=_candidate_time_coverage(
            [float(value) for value in beat_times],
            duration,
        ),
        evidence=librosa_evidence,
        accepted=True,
        reason="baseline",
    )
    onset_grid_beat_times = _regular_beat_times(estimate.offset, estimate.bpm, duration)
    onset_grid_support, onset_grid_evidence = _tempo_meter_evidence(
        bpm=estimate.bpm,
        offset=estimate.offset,
        time_signature="4/4",
        beat_times=onset_grid_beat_times,
        downbeat_times=[],
        onset_times=onset_times_list,
        onset_weights=onset_weights,
        onset_strengths=[float(value) for value in onset_env.tolist()],
        spectral=spectral,
        sample_rate=int(sr),
        hop_length=hop_length,
        onset_support=estimate.normalized_support,
        runner_up_bpm=estimate.runner_up_bpm,
        runner_up_support=estimate.runner_up_support,
    )
    onset_grid_candidate = _tempo_candidate(
        source="librosa+onset-grid",
        bpm=estimate.bpm,
        offset=estimate.offset,
        time_signature="4/4",
        beat_times=onset_grid_beat_times,
        onset_support=onset_grid_support,
        time_coverage=estimate.time_coverage,
        evidence=onset_grid_evidence,
        confidence=estimate.normalized_support,
        accepted=estimate.accepted,
        reason=estimate.reason,
    )
    if estimate.accepted:
        bpm, offset = estimate.bpm, estimate.offset
        beat_times = onset_grid_beat_times
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
        tempo_candidates=[librosa_candidate, onset_grid_candidate],
        tempo_analysis=estimate.to_decision("librosa"),
        spectral=spectral,
    )
    raw = _apply_tempo_arbitration(
        raw,
        raw.tempo_candidates,
        fallback_source="librosa",
    )

    if not use_beatnet:
        return raw

    return enhance_with_beatnet(input_path, raw)


def _ensure_beatnet_numpy_compatibility() -> None:
    # BeatNet 依赖的 madmom 0.16.1 仍会访问 NumPy 1.24 移除的旧别名。
    if "float" not in np.__dict__:
        setattr(np, "float", np.float64)
    if "int" not in np.__dict__:
        setattr(np, "int", np.int_)


def _install_beatnet_import_stubs() -> list[str]:
    installed: list[str] = []
    if importlib.util.find_spec("madmom") is None:
        madmom = ModuleType("madmom")
        madmom.__path__ = []
        features = ModuleType("madmom.features")

        class UnavailableDbnProcessor:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("madmom is unavailable; use BeatNet particle filtering")

        features.DBNDownBeatTrackingProcessor = UnavailableDbnProcessor
        madmom.features = features
        sys.modules["madmom"] = madmom
        sys.modules["madmom.features"] = features
        installed.extend(["madmom.features", "madmom"])

    if importlib.util.find_spec("pyaudio") is None:
        pyaudio = ModuleType("pyaudio")

        class UnavailablePyAudio:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("pyaudio is unavailable; BeatNet stream mode cannot be used")

        pyaudio.PyAudio = UnavailablePyAudio
        pyaudio.paFloat32 = 0
        sys.modules["pyaudio"] = pyaudio
        installed.append("pyaudio")
    return installed


def _import_beatnet_class() -> Any:
    with _BEATNET_IMPORT_LOCK:
        installed_stubs = _install_beatnet_import_stubs()
        original_distribution = importlib.metadata.distribution

        def compatible_distribution(name: str) -> importlib.metadata.Distribution:
            try:
                return original_distribution(name)
            except importlib.metadata.PackageNotFoundError:
                if name != "madmom":
                    raise
                return original_distribution("madmom-prebuilt")

        importlib.metadata.distribution = compatible_distribution
        try:
            from BeatNet.BeatNet import BeatNet

            return BeatNet
        finally:
            importlib.metadata.distribution = original_distribution
            for module_name in installed_stubs:
                sys.modules.pop(module_name, None)


def enhance_with_beatnet(input_path: Path, raw: AudioAnalysisRaw) -> AudioAnalysisRaw:
    try:
        _ensure_beatnet_numpy_compatibility()
        beatnet_class = _import_beatnet_class()

        estimator = beatnet_class(
            model=1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
        )
        output = estimator.process(str(input_path))
        merged = merge_beatnet_output(raw, output)
        if merged is raw:
            return raw.model_copy(
                update={
                    "beatnet_analysis_status": "fallback",
                    "beatnet_analysis_reason": "invalid-or-empty-output",
                }
            )
        return merged.model_copy(
            update={
                "beatnet_analysis_status": "complete",
                "beatnet_analysis_reason": None,
            }
        )
    except ImportError as error:
        return raw.model_copy(
            update={
                "beatnet_analysis_status": "fallback",
                "beatnet_analysis_reason": f"missing-dependency:{error.name or 'BeatNet'}",
            }
        )
    except Exception as error:  # noqa: BLE001 - BeatNet is an optional enhancement.
        return raw.model_copy(
            update={
                "beatnet_analysis_status": "fallback",
                "beatnet_analysis_reason": f"inference-error:{type(error).__name__}",
            }
        )


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

    expected_meter_size = {"3/4": 3, "4/4": 4, "6/8": 6}[time_signature]
    beatnet_downbeat_references = (
        [
            time
            for time, number in zip(beatnet_beat_times, beatnet_beat_numbers, strict=True)
            if number == 1
        ]
        if max(beatnet_beat_numbers) == expected_meter_size
        else []
    )
    onset_weights = _weights_from_raw_onsets(raw)
    beatnet_onset_support, beatnet_evidence = _tempo_meter_evidence(
        bpm=bpm,
        offset=offset,
        time_signature=time_signature,
        beat_times=beatnet_beat_times,
        downbeat_times=beatnet_downbeat_references,
        onset_times=raw.onset_times,
        onset_weights=onset_weights,
        onset_strengths=raw.onset_strengths,
        spectral=raw.spectral,
        sample_rate=raw.sample_rate,
        hop_length=raw.hop_length,
        beat_numbers=beatnet_beat_numbers,
    )
    beatnet_candidate = _tempo_candidate(
        source="beatnet",
        bpm=bpm,
        offset=offset,
        time_signature=time_signature,
        beat_times=beatnet_beat_times,
        downbeat_times=beatnet_downbeat_references,
        onset_support=beatnet_onset_support,
        time_coverage=_candidate_time_coverage(beatnet_beat_times, raw.duration),
        evidence=beatnet_evidence,
        accepted=True,
        reason="valid-output",
    )

    estimate = _estimate_tempo_and_offset_from_onsets(
        raw.onset_times,
        onset_weights,
        np.asarray([], dtype=float),
        sample_rate=raw.sample_rate or 0,
        fallback_bpm=bpm,
        duration=raw.duration,
    )
    estimate = _keep_compound_meter_quarter_note_tempo(estimate, bpm, time_signature)
    refined_beat_times = _regular_beat_times(estimate.offset, estimate.bpm, raw.duration)
    refined_downbeat_times = _project_downbeats_to_regular_grid(
        beatnet_downbeat_references,
        refined_beat_times,
        estimate.offset,
        estimate.bpm,
    )
    refined_onset_support, refined_evidence = _tempo_meter_evidence(
        bpm=estimate.bpm,
        offset=estimate.offset,
        time_signature=time_signature,
        beat_times=refined_beat_times,
        downbeat_times=refined_downbeat_times,
        onset_times=raw.onset_times,
        onset_weights=onset_weights,
        onset_strengths=raw.onset_strengths,
        spectral=raw.spectral,
        sample_rate=raw.sample_rate,
        hop_length=raw.hop_length,
        beat_numbers=beatnet_beat_numbers,
        onset_support=estimate.normalized_support,
        runner_up_bpm=estimate.runner_up_bpm,
        runner_up_support=estimate.runner_up_support,
    )
    refined_candidate = _tempo_candidate(
        source="beatnet+onset-grid",
        bpm=estimate.bpm,
        offset=estimate.offset,
        time_signature=time_signature,
        beat_times=refined_beat_times,
        downbeat_times=refined_downbeat_times,
        onset_support=refined_onset_support,
        time_coverage=estimate.time_coverage,
        evidence=refined_evidence,
        confidence=estimate.normalized_support,
        accepted=estimate.accepted,
        reason=estimate.reason,
    )
    baseline_candidates = raw.tempo_candidates or [_raw_baseline_candidate(raw)]
    fallback_candidate = next(
        (candidate for candidate in baseline_candidates if candidate.selected),
        next(
            (
                candidate
                for candidate in baseline_candidates
                if candidate.source == raw.analyzer
            ),
            baseline_candidates[0],
        ),
    )
    candidates = _replace_tempo_candidates(
        baseline_candidates,
        [beatnet_candidate, refined_candidate],
    )
    return _apply_tempo_arbitration(
        raw,
        candidates,
        fallback_source=fallback_candidate.source,
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
