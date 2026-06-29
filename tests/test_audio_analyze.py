import pytest

from tja_ai_chartgen.audio.analyze import (
    AudioAnalysisRaw,
    apply_analysis_overrides,
    estimate_time_signature,
    merge_beatnet_output,
    normalize_bpm,
)


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


def test_merge_beatnet_output_updates_downbeats_meter_and_offset():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[0.0, 0.5],
        onset_strengths=[],
        duration=4.0,
        offset=0.1,
    )

    updated = merge_beatnet_output(
        raw,
        [
            [0.25, 1],
            [0.75, 2],
            [1.25, 3],
            [1.75, 1],
        ],
    )

    assert updated.analyzer == "beatnet+librosa"
    assert updated.beat_times == [0.25, 0.75, 1.25, 1.75]
    assert updated.beat_numbers == [1, 2, 3, 1]
    assert updated.downbeat_times == [0.25, 1.75]
    assert updated.offset == 0.25
    assert updated.bpm == 120
    assert updated.time_signature == "3/4"


def test_merge_beatnet_output_keeps_raw_when_output_is_empty():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[],
        onset_strengths=[],
        duration=2.0,
        offset=0.1,
    )

    assert merge_beatnet_output(raw, []) is raw


def test_estimate_time_signature_defaults_to_four_four_for_unknown_meter():
    assert estimate_time_signature([1, 2, 3, 4]) == "4/4"
    assert estimate_time_signature([1, 2, 3, 4, 5, 6]) == "6/8"
    assert estimate_time_signature([]) == "4/4"
