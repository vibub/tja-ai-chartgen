import numpy as np

from tja_ai_chartgen.audio import spectral as spectral_module
from tja_ai_chartgen.audio.spectral import extract_spectral_features


def test_extract_spectral_features_returns_aligned_normalized_envelopes():
    sample_rate = 8_000
    duration = 2.0
    times = np.arange(round(sample_rate * duration), dtype=float) / sample_rate
    samples = 0.2 * np.sin(2 * np.pi * 220.0 * times)
    for start in range(0, samples.size, sample_rate // 4):
        end = min(samples.size, start + 32)
        samples[start:end] += np.hanning((end - start) * 2)[end - start :]

    result = extract_spectral_features(
        samples,
        sample_rate=sample_rate,
        hop_length=256,
    )

    assert result.status == "complete"
    assert result.feature_version == "spectral-v1"
    assert result.reason is None
    assert result.frame_count > 0
    envelopes = [
        result.low_onset_envelope,
        result.mid_onset_envelope,
        result.high_onset_envelope,
        result.spectral_flux_envelope,
        result.brightness_envelope,
        result.harmonic_novelty_envelope,
        result.texture_novelty_envelope,
        result.percussive_ratio_envelope,
    ]
    assert all(len(envelope) == result.frame_count for envelope in envelopes)
    assert all(0.0 <= value <= 1.0 for envelope in envelopes for value in envelope)
    assert any(value > 0.0 for value in result.spectral_flux_envelope)
    assert any(value > 0.0 for value in result.brightness_envelope)
    assert any(value > 0.0 for value in result.percussive_ratio_envelope)


def test_extract_spectral_features_handles_digital_silence_without_fallback():
    result = extract_spectral_features(
        np.zeros(4_096, dtype=float),
        sample_rate=8_000,
        hop_length=256,
    )

    assert result.status == "complete"
    assert result.frame_count == 17
    assert result.spectral_flux_envelope == [0.0] * 17
    assert result.harmonic_novelty_envelope == [0.0] * 17


def test_extract_spectral_features_falls_back_for_short_audio():
    result = extract_spectral_features(
        np.zeros(1_024, dtype=float),
        sample_rate=8_000,
        hop_length=256,
    )

    assert result.status == "fallback"
    assert result.reason == "audio-too-short"
    assert result.frame_count == 0


def test_extract_spectral_features_contains_library_failures(monkeypatch):
    monkeypatch.setattr(
        spectral_module.librosa.effects,
        "hpss",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    result = extract_spectral_features(
        np.ones(4_096, dtype=float),
        sample_rate=8_000,
        hop_length=256,
    )

    assert result.status == "fallback"
    assert result.reason == "extractor-error:RuntimeError"
