import pytest

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw, apply_analysis_overrides, normalize_bpm


def test_normalize_bpm_keeps_taiko_friendly_range():
    assert normalize_bpm(60) == 120
    assert normalize_bpm(260) == 130


def test_normalize_bpm_rejects_non_positive_values():
    with pytest.raises(ValueError, match="BPM must be positive"):
        normalize_bpm(0)


def test_apply_analysis_overrides_updates_bpm_and_offset():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0, 0.5, 1.0],
        onset_times=[0.0, 0.5],
        onset_strengths=[],
        duration=2.0,
        offset=0.0,
    )

    updated = apply_analysis_overrides(raw, bpm=180.1234, offset=-0.025)

    assert updated.bpm == 180.123
    assert updated.offset == -0.025
    assert raw.bpm == 120
    assert raw.offset == 0.0


def test_apply_analysis_overrides_rejects_non_positive_bpm():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[],
        onset_strengths=[],
        duration=2.0,
        offset=0.0,
    )

    with pytest.raises(ValueError, match="BPM must be positive"):
        apply_analysis_overrides(raw, bpm=0)
