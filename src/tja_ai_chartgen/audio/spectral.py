from __future__ import annotations

from typing import Literal

import librosa
import numpy as np
from pydantic import BaseModel, Field


SPECTRAL_FEATURE_VERSION = "spectral-v1"
SPECTRAL_N_FFT = 2048
SPECTRAL_MEL_BANDS = 96
MIN_SPECTRAL_SAMPLES = SPECTRAL_N_FFT
SILENCE_EPSILON = 1e-8


class SpectralAnalysisRaw(BaseModel):
    feature_version: str | None = None
    status: Literal["unavailable", "complete", "fallback"] = "unavailable"
    reason: str | None = None
    frame_count: int = Field(default=0, ge=0)
    low_onset_envelope: list[float] = Field(default_factory=list)
    mid_onset_envelope: list[float] = Field(default_factory=list)
    high_onset_envelope: list[float] = Field(default_factory=list)
    spectral_flux_envelope: list[float] = Field(default_factory=list)
    brightness_envelope: list[float] = Field(default_factory=list)
    harmonic_novelty_envelope: list[float] = Field(default_factory=list)
    texture_novelty_envelope: list[float] = Field(default_factory=list)
    percussive_ratio_envelope: list[float] = Field(default_factory=list)


def extract_spectral_features(
    samples: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> SpectralAnalysisRaw:
    if sample_rate <= 0:
        return _fallback("invalid-sample-rate")
    if hop_length <= 0:
        return _fallback("invalid-hop-length")

    waveform = np.asarray(samples, dtype=float)
    if waveform.ndim != 1 or not np.all(np.isfinite(waveform)):
        return _fallback("invalid-waveform")
    if waveform.size < MIN_SPECTRAL_SAMPLES:
        return _fallback("audio-too-short")

    frame_count = _centered_frame_count(waveform.size, hop_length)
    if np.max(np.abs(waveform), initial=0.0) <= SILENCE_EPSILON:
        return _complete_with_zeros(frame_count)

    try:
        harmonic, percussive = librosa.effects.hpss(
            waveform,
            n_fft=SPECTRAL_N_FFT,
            hop_length=hop_length,
        )
        band_onsets = _multi_band_onsets(
            percussive,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        band_onsets = _align_matrix(band_onsets, frame_count)
        normalized_bands = _normalize_joint(band_onsets)

        spectral_flux = _normalize_envelope(np.mean(band_onsets, axis=0))
        brightness = _brightness(
            waveform,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        harmonic_novelty = _harmonic_novelty(
            harmonic,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        percussive_ratio = _percussive_ratio(
            harmonic,
            percussive,
            hop_length=hop_length,
        )
        texture_novelty = _texture_novelty(
            waveform,
            brightness=brightness,
            percussive_ratio=percussive_ratio,
            hop_length=hop_length,
        )

        return SpectralAnalysisRaw(
            feature_version=SPECTRAL_FEATURE_VERSION,
            status="complete",
            frame_count=frame_count,
            low_onset_envelope=_to_values(normalized_bands[0], frame_count),
            mid_onset_envelope=_to_values(normalized_bands[1], frame_count),
            high_onset_envelope=_to_values(normalized_bands[2], frame_count),
            spectral_flux_envelope=_to_values(spectral_flux, frame_count),
            brightness_envelope=_to_values(brightness, frame_count),
            harmonic_novelty_envelope=_to_values(harmonic_novelty, frame_count),
            texture_novelty_envelope=_to_values(texture_novelty, frame_count),
            percussive_ratio_envelope=_to_values(percussive_ratio, frame_count),
        )
    except Exception as error:  # noqa: BLE001 - spectral analysis is an optional enhancement.
        return _fallback(f"extractor-error:{type(error).__name__}")


def _multi_band_onsets(
    samples: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    mel_frequencies = librosa.mel_frequencies(
        n_mels=SPECTRAL_MEL_BANDS,
        fmin=0.0,
        fmax=nyquist,
    )
    low_boundary = _band_boundary(mel_frequencies, min(250.0, nyquist * 0.2), minimum=1)
    mid_boundary = _band_boundary(
        mel_frequencies,
        min(2_000.0, nyquist * 0.7),
        minimum=low_boundary + 1,
    )
    mid_boundary = min(SPECTRAL_MEL_BANDS - 1, mid_boundary)
    low_boundary = min(low_boundary, mid_boundary - 1)

    values = librosa.onset.onset_strength_multi(
        y=samples,
        sr=sample_rate,
        n_fft=SPECTRAL_N_FFT,
        hop_length=hop_length,
        channels=[0, low_boundary, mid_boundary, SPECTRAL_MEL_BANDS],
        n_mels=SPECTRAL_MEL_BANDS,
        fmin=0.0,
        fmax=nyquist,
        center=True,
    )
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 3:
        raise ValueError("unexpected multi-band onset shape")
    return np.maximum(0.0, np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0))


def _band_boundary(frequencies: np.ndarray, cutoff: float, *, minimum: int) -> int:
    boundary = int(np.searchsorted(frequencies, cutoff, side="left"))
    return max(minimum, min(SPECTRAL_MEL_BANDS - 1, boundary))


def _brightness(
    samples: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    centroid = librosa.feature.spectral_centroid(
        y=samples,
        sr=sample_rate,
        n_fft=SPECTRAL_N_FFT,
        hop_length=hop_length,
    )[0]
    nyquist = max(1.0, sample_rate / 2.0)
    return np.clip(np.nan_to_num(centroid / nyquist, nan=0.0), 0.0, 1.0)


def _harmonic_novelty(
    harmonic: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=harmonic,
        sr=sample_rate,
        n_fft=SPECTRAL_N_FFT,
        hop_length=hop_length,
        tuning=0.0,
    )
    chroma = np.maximum(0.0, np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0))
    column_sums = np.sum(chroma, axis=0, keepdims=True)
    normalized = np.divide(
        chroma,
        column_sums,
        out=np.zeros_like(chroma),
        where=column_sums > SILENCE_EPSILON,
    )
    novelty = np.zeros(normalized.shape[1], dtype=float)
    if normalized.shape[1] > 1:
        novelty[1:] = np.sum(np.abs(np.diff(normalized, axis=1)), axis=0) / 2.0
    return _normalize_envelope(novelty)


def _percussive_ratio(
    harmonic: np.ndarray,
    percussive: np.ndarray,
    *,
    hop_length: int,
) -> np.ndarray:
    harmonic_rms = librosa.feature.rms(y=harmonic, hop_length=hop_length)[0]
    percussive_rms = librosa.feature.rms(y=percussive, hop_length=hop_length)[0]
    frame_count = min(harmonic_rms.size, percussive_rms.size)
    harmonic_rms = harmonic_rms[:frame_count]
    percussive_rms = percussive_rms[:frame_count]
    total = harmonic_rms + percussive_rms
    return np.divide(
        percussive_rms,
        total,
        out=np.zeros_like(percussive_rms, dtype=float),
        where=total > SILENCE_EPSILON,
    )


def _texture_novelty(
    samples: np.ndarray,
    *,
    brightness: np.ndarray,
    percussive_ratio: np.ndarray,
    hop_length: int,
) -> np.ndarray:
    flatness = librosa.feature.spectral_flatness(
        y=samples,
        n_fft=SPECTRAL_N_FFT,
        hop_length=hop_length,
    )[0]
    frame_count = min(brightness.size, flatness.size, percussive_ratio.size)
    texture = np.vstack(
        [
            brightness[:frame_count],
            np.clip(np.nan_to_num(flatness[:frame_count], nan=0.0), 0.0, 1.0),
            percussive_ratio[:frame_count],
        ]
    )
    novelty = np.zeros(frame_count, dtype=float)
    if frame_count > 1:
        novelty[1:] = np.mean(np.abs(np.diff(texture, axis=1)), axis=0)
    return _normalize_envelope(novelty)


def _normalize_joint(values: np.ndarray) -> np.ndarray:
    finite = np.maximum(0.0, np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros_like(finite)
    scale = float(np.percentile(positive, 95))
    if scale <= SILENCE_EPSILON:
        scale = float(np.max(positive))
    if scale <= SILENCE_EPSILON:
        return np.zeros_like(finite)
    return np.clip(finite / scale, 0.0, 1.0)


def _normalize_envelope(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return _normalize_joint(array.reshape(1, -1))[0]


def _align_matrix(values: np.ndarray, frame_count: int) -> np.ndarray:
    if values.shape[1] == frame_count:
        return values
    result = np.zeros((values.shape[0], frame_count), dtype=float)
    copied = min(frame_count, values.shape[1])
    result[:, :copied] = values[:, :copied]
    return result


def _to_values(values: np.ndarray, frame_count: int) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    result = np.zeros(frame_count, dtype=float)
    copied = min(frame_count, array.size)
    result[:copied] = np.clip(
        np.nan_to_num(array[:copied], nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    return [round(float(value), 4) for value in result]


def _centered_frame_count(sample_count: int, hop_length: int) -> int:
    return 1 + sample_count // hop_length


def _complete_with_zeros(frame_count: int) -> SpectralAnalysisRaw:
    zeros = [0.0] * frame_count
    return SpectralAnalysisRaw(
        feature_version=SPECTRAL_FEATURE_VERSION,
        status="complete",
        frame_count=frame_count,
        low_onset_envelope=zeros.copy(),
        mid_onset_envelope=zeros.copy(),
        high_onset_envelope=zeros.copy(),
        spectral_flux_envelope=zeros.copy(),
        brightness_envelope=zeros.copy(),
        harmonic_novelty_envelope=zeros.copy(),
        texture_novelty_envelope=zeros.copy(),
        percussive_ratio_envelope=zeros.copy(),
    )


def _fallback(reason: str) -> SpectralAnalysisRaw:
    return SpectralAnalysisRaw(
        feature_version=SPECTRAL_FEATURE_VERSION,
        status="fallback",
        reason=reason,
    )
