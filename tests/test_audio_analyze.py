import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tja_ai_chartgen.audio import analyze as audio_analyze
from tja_ai_chartgen.audio.analyze import (
    AudioAnalysisRaw,
    _estimate_tempo_and_offset_from_onsets,
    _regular_beat_times,
    analyze_audio,
    apply_analysis_overrides,
    enhance_with_beatnet,
    estimate_time_signature,
    merge_beatnet_output,
    normalize_bpm,
)
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.tja.model import ChartBar, ChartMetadata, TempoAnalysisDecision, TjaChart
from tja_ai_chartgen.tja.writer import render_tja


def test_normalize_bpm_keeps_taiko_friendly_range():
    assert normalize_bpm(60) == 120
    assert normalize_bpm(260) == 130


def test_normalize_bpm_rejects_non_positive_values():
    with pytest.raises(ValueError, match="BPM must be positive"):
        normalize_bpm(0)


def test_estimate_tempo_and_offset_from_onsets_selects_periodic_grid():
    onset_times = [0.2 + i * 0.5 for i in range(24)]
    weights = [1.0 for _ in onset_times]
    samples = np.zeros(12_000, dtype=float)

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        weights,
        samples,
        sample_rate=1_000,
        fallback_bpm=90,
        duration=12.0,
    )

    assert estimate.accepted is True
    assert estimate.reason == "accepted"
    assert estimate.bpm == 120
    assert estimate.offset == pytest.approx(0.2, abs=0.006)
    assert estimate.normalized_support >= 0.95
    assert estimate.onset_count == 24
    assert estimate.time_coverage == pytest.approx(11.5 / 12.0)


def test_estimate_tempo_and_offset_from_onsets_uses_fallback_to_resolve_double_tempo_alias():
    onset_times = [0.2 + i * 0.6 for i in range(20)]

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        [1.0 for _ in onset_times],
        np.zeros(12_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=100,
        duration=12.0,
    )

    assert estimate.accepted is True
    assert estimate.bpm == 100
    assert estimate.offset == pytest.approx(0.2, abs=0.006)
    assert estimate.runner_up_bpm == 200


def test_estimate_tempo_and_offset_from_onsets_preserves_sub_bin_phase():
    onset_times = [0.203 + i * 0.5 for i in range(24)]
    weights = [1.0 for _ in onset_times]
    samples = np.zeros(12_000, dtype=float)

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        weights,
        samples,
        sample_rate=1_000,
        fallback_bpm=120,
        duration=12.0,
    )

    assert estimate.accepted is True
    assert estimate.bpm == 120
    assert estimate.offset == pytest.approx(0.203, abs=0.001)


def test_estimate_tempo_and_offset_from_onsets_uses_waveform_to_reject_offbeat():
    onset_times = [0.25 + i * 0.5 for i in range(16)] + [0.5 + i * 0.5 for i in range(16)]
    weights = [1.0 for _ in onset_times]
    samples = np.zeros(10_000, dtype=float)
    for index in range(0, len(samples), 500):
        samples[index : index + 20] = np.linspace(0.0, 1.0, 20)

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        weights,
        samples,
        sample_rate=1_000,
        fallback_bpm=120,
        duration=10.0,
    )

    assert estimate.accepted is True
    assert estimate.bpm == 120
    assert estimate.offset == pytest.approx(0.0, abs=0.006)


def test_estimate_tempo_and_offset_from_onsets_preserves_preoptimization_baseline():
    onset_times = [0.25 + i * 0.5 for i in range(16)] + [0.5 + i * 0.5 for i in range(16)]
    weights = [1.0 for _ in onset_times]
    samples = np.zeros(10_000, dtype=float)
    for index in range(0, len(samples), 500):
        samples[index : index + 20] = np.linspace(0.0, 1.0, 20)

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        weights,
        samples,
        sample_rate=1_000,
        fallback_bpm=120,
        duration=10.0,
    )

    assert estimate.accepted is True
    assert (estimate.bpm, estimate.offset) == (120.0, 0.0)


def test_estimate_tempo_and_offset_from_onsets_rejects_random_input():
    onset_times = np.random.default_rng(42).uniform(0.0, 12.0, 24).tolist()

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        [1.0 for _ in onset_times],
        np.zeros(12_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=120,
        duration=12.0,
    )

    assert estimate.accepted is False
    assert estimate.reason in {"ambiguous_candidates", "low_normalized_support"}


def test_estimate_tempo_and_offset_from_onsets_rejects_four_short_onsets():
    onset_times = [0.2, 0.7, 1.2, 1.7]

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        [1.0 for _ in onset_times],
        np.zeros(8_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=120,
        duration=8.0,
    )

    assert estimate.accepted is False
    assert estimate.reason == "insufficient_onsets"
    assert estimate.onset_count == 4
    assert estimate.time_coverage == pytest.approx(1.5 / 8.0)


def test_estimate_tempo_and_offset_from_onsets_rejects_ambiguous_candidates():
    onset_times = sorted(
        [0.2 + i * 0.5 for i in range(16)]
        + [0.2 + i * (60.0 / 128.0) for i in range(16)]
    )

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        [1.0 for _ in onset_times],
        np.zeros(8_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=120,
        duration=8.0,
    )

    assert estimate.accepted is False
    assert estimate.reason == "ambiguous_candidates"
    assert estimate.runner_up_bpm is not None
    assert estimate.runner_up_support / estimate.normalized_support >= 0.9


def test_estimate_tempo_and_offset_from_onsets_rejects_invalid_input():
    estimate = _estimate_tempo_and_offset_from_onsets(
        [0.2, 0.7],
        [1.0],
        np.zeros(2_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=120,
        duration=0.0,
    )

    assert estimate.accepted is False
    assert estimate.reason == "invalid_input"


def test_estimate_tempo_and_offset_from_onsets_rejects_non_vector_input():
    estimate = _estimate_tempo_and_offset_from_onsets(
        [[0.2], [0.7]],
        [[1.0], [1.0]],
        np.zeros(2_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=120,
        duration=2.0,
    )

    assert estimate.accepted is False
    assert estimate.reason == "invalid_input"


def test_estimate_tempo_and_offset_accepts_dense_high_coverage_grid_with_musical_noise():
    periodic_onsets = [0.2 + i * 0.5 for i in range(120)]
    off_grid_onsets = np.random.default_rng(7).uniform(0.0, 60.0, 50).tolist()
    onset_times = sorted(periodic_onsets + off_grid_onsets)

    estimate = _estimate_tempo_and_offset_from_onsets(
        onset_times,
        [1.0 for _ in onset_times],
        np.asarray([], dtype=float),
        sample_rate=0,
        fallback_bpm=120,
        duration=60.0,
    )

    assert estimate.accepted is True
    assert estimate.bpm == 120
    assert 0.68 <= estimate.normalized_support < 0.75
    assert estimate.onset_count == 170
    assert estimate.time_coverage > 0.9


def test_estimate_tempo_and_offset_normalizes_support_by_total_weight():
    short_onsets = [0.2 + i * 0.5 for i in range(12)]
    long_onsets = [0.2 + i * 0.5 for i in range(24)]

    short_estimate = _estimate_tempo_and_offset_from_onsets(
        short_onsets,
        [1.0 for _ in short_onsets],
        np.zeros(6_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=120,
        duration=6.0,
    )
    long_estimate = _estimate_tempo_and_offset_from_onsets(
        long_onsets,
        [1.0 for _ in long_onsets],
        np.zeros(12_000, dtype=float),
        sample_rate=1_000,
        fallback_bpm=120,
        duration=12.0,
    )

    assert short_estimate.accepted is True
    assert long_estimate.accepted is True
    assert short_estimate.normalized_support == pytest.approx(
        long_estimate.normalized_support,
        abs=0.02,
    )


def test_adjust_offset_for_offbeats_skips_precompute_for_empty_waveform(monkeypatch):
    monkeypatch.setattr(
        audio_analyze,
        "_precompute_waveform_support",
        lambda samples: pytest.fail("empty waveform should not be precomputed"),
        raising=False,
    )

    result = audio_analyze._adjust_offset_for_offbeats(np.asarray([], dtype=float), 1_000, 0.63, 120)

    assert result == pytest.approx(0.13)


def test_adjust_offset_for_offbeats_skips_precompute_for_short_waveform(monkeypatch):
    monkeypatch.setattr(
        audio_analyze,
        "_precompute_waveform_support",
        lambda samples: pytest.fail("short waveform should not be precomputed"),
        raising=False,
    )

    result = audio_analyze._adjust_offset_for_offbeats(np.zeros(99, dtype=float), 1_000, 0.63, 120)

    assert result == pytest.approx(0.13)


@pytest.mark.parametrize(
    ("sample_rate", "bpm", "expected"),
    [(0, 120, 0.13), (-1, 120, 0.13), (1_000, 0, 0.63)],
)
def test_adjust_offset_for_offbeats_preserves_invalid_parameter_semantics(
    monkeypatch,
    sample_rate,
    bpm,
    expected,
):
    monkeypatch.setattr(
        audio_analyze,
        "_precompute_waveform_support",
        lambda samples: pytest.fail("invalid parameters should not precompute waveform"),
        raising=False,
    )

    result = audio_analyze._adjust_offset_for_offbeats(np.zeros(1_000, dtype=float), sample_rate, 0.63, bpm)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(("phase", "expected"), [(0.0, 0.0), (0.25, 0.25)])
def test_adjust_offset_for_offbeats_selects_waveform_supported_phase(phase, expected):
    samples = np.zeros(5_000, dtype=float)
    start = round(phase * 1_000)
    for index in range(start, len(samples), 500):
        samples[index : index + 20] = np.linspace(0.0, 1.0, 20)

    result = audio_analyze._adjust_offset_for_offbeats(samples, 1_000, 0.0, 120)

    assert result == pytest.approx(expected, abs=0.006)


def test_adjust_offset_for_offbeats_prefers_primary_phase_when_support_ties():
    result = audio_analyze._adjust_offset_for_offbeats(np.zeros(5_000, dtype=float), 1_000, 0.1, 120)

    assert result == pytest.approx(0.1)


def test_precompute_waveform_support_builds_absolute_prefix_sum():
    samples = np.asarray([-2.0, 1.0, -3.0])
    original = samples.copy()

    abs_samples, prefix = audio_analyze._precompute_waveform_support(samples)

    np.testing.assert_array_equal(abs_samples, [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(prefix, [0.0, 2.0, 3.0, 6.0])
    np.testing.assert_array_equal(samples, original)
    assert prefix.dtype == np.dtype(float)


def test_adjust_offset_for_offbeats_precomputes_waveform_once(monkeypatch):
    call_count = 0

    def fake_precompute(samples):
        nonlocal call_count
        call_count += 1
        abs_samples = np.abs(samples)
        return abs_samples, np.concatenate([[0.0], np.cumsum(abs_samples, dtype=float)])

    monkeypatch.setattr(audio_analyze, "_precompute_waveform_support", fake_precompute, raising=False)
    samples = np.zeros(5_000, dtype=float)
    for index in range(0, len(samples), 500):
        samples[index : index + 20] = np.linspace(0.0, 1.0, 20)

    audio_analyze._adjust_offset_for_offbeats(samples, 1_000, 0.0, 120)

    assert call_count == 1


def test_regular_beat_times_starts_at_refined_offset():
    assert _regular_beat_times(0.2, 120, 1.3) == [0.2, 0.7, 1.2]


def test_regular_beat_times_keeps_negative_offset_grid_start():
    assert _regular_beat_times(-0.05, 120, 1.0) == [-0.05, 0.45, 0.95]


def test_apply_analysis_overrides_updates_bpm_and_offset():
    tempo_analysis = TempoAnalysisDecision(
        fallback_source="librosa",
        selected_source="onset-grid",
        estimated_bpm=120,
        estimated_offset=0.2,
        normalized_support=0.95,
        onset_count=16,
        time_coverage=0.8,
        accepted=True,
        reason="accepted",
    )
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0, 0.5, 1.0],
        onset_times=[0.0, 0.5],
        onset_strengths=[],
        duration=2.0,
        offset=0.0,
        analyzer="librosa+onset-grid",
        tempo_analysis=tempo_analysis,
    )

    updated = apply_analysis_overrides(raw, bpm=180.1234, offset=-0.025)

    assert updated.bpm == 180.123
    assert updated.offset == -0.025
    assert updated.analyzer == "librosa+onset-grid+manual-override"
    assert updated.tempo_analysis == tempo_analysis
    assert raw.bpm == 120
    assert raw.offset == 0.0
    assert raw.analyzer == "librosa+onset-grid"


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


def test_analyze_audio_keeps_librosa_baseline_when_onset_grid_is_rejected(tmp_path, monkeypatch):
    input_path = tmp_path / "sparse.wav"
    input_path.write_bytes(b"audio")
    samples = np.zeros(8_000, dtype=float)

    monkeypatch.setattr(audio_analyze.librosa, "load", lambda *args, **kwargs: (samples, 1_000))
    monkeypatch.setattr(audio_analyze.librosa, "get_duration", lambda *args, **kwargs: 8.0)
    monkeypatch.setattr(
        audio_analyze.librosa.onset,
        "onset_strength",
        lambda *args, **kwargs: np.ones(8, dtype=float),
    )
    monkeypatch.setattr(audio_analyze, "_rms_envelopes", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        audio_analyze.librosa.beat,
        "beat_track",
        lambda *args, **kwargs: (np.asarray([100.0]), np.asarray([0.1, 0.7])),
    )
    monkeypatch.setattr(
        audio_analyze.librosa.onset,
        "onset_detect",
        lambda *args, **kwargs: np.asarray([0.2, 0.9]),
    )
    monkeypatch.setattr(
        audio_analyze.librosa,
        "frames_to_time",
        lambda frames, **kwargs: np.asarray(frames, dtype=float),
    )

    raw = analyze_audio(input_path)

    assert raw.bpm == 100
    assert raw.offset == 0.1
    assert raw.beat_times == [0.1, 0.7]
    assert raw.analyzer == "librosa"
    assert raw.spectral.status == "complete"
    assert raw.spectral.feature_version == "spectral-v1"
    assert raw.tempo_analysis is not None
    assert raw.tempo_analysis.accepted is False
    assert raw.tempo_analysis.reason == "insufficient_onsets"
    assert raw.tempo_analysis.selected_source == "librosa"


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

    assert updated.analyzer == "beatnet"
    assert updated.beat_times[:4] == [0.25, 0.75, 1.25, 1.75]
    assert updated.beat_numbers[:4] == [1, 2, 3, 1]
    assert updated.downbeat_times[:2] == [0.25, 1.75]
    assert updated.offset == 0.25
    assert updated.bpm == 120
    assert updated.time_signature == "3/4"
    assert updated.tempo_analysis is not None
    assert updated.tempo_analysis.accepted is False
    assert updated.tempo_analysis.reason == "insufficient_onsets"
    assert updated.tempo_analysis.selected_source == "beatnet"


def test_merge_beatnet_output_uses_four_four_grid_when_two_beat_meter_is_unsupported():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.25, 0.75],
        onset_times=[],
        onset_strengths=[],
        duration=4.0,
        offset=0.25,
    )

    updated = merge_beatnet_output(
        raw,
        [[0.25 + index * 0.5, (index % 2) + 1] for index in range(8)],
    )

    assert updated.time_signature == "4/4"
    assert updated.beat_numbers[:8] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert updated.downbeat_times == [0.25, 2.25, 4.25]


def test_merge_beatnet_output_converts_six_eight_pulses_to_quarter_note_bpm():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.25, 0.75],
        onset_times=[],
        onset_strengths=[],
        duration=3.25,
        offset=0.25,
    )

    updated = merge_beatnet_output(
        raw,
        [
            [0.25, 1],
            [0.5, 2],
            [0.75, 3],
            [1.0, 4],
            [1.25, 5],
            [1.5, 6],
            [1.75, 1],
        ],
    )

    assert updated.time_signature == "6/8"
    assert updated.bpm == 120
    assert updated.beat_times[:4] == [0.25, 0.75, 1.25, 1.75]
    assert updated.beat_numbers[:4] == [1, 2, 3, 1]
    assert updated.downbeat_times[:2] == [0.25, 1.75]


def test_six_eight_beatnet_analysis_builds_one_full_length_tja_bar():
    raw = AudioAnalysisRaw(
        bpm=180,
        beat_times=[],
        onset_times=[0.25, 1.0],
        onset_strengths=[],
        duration=1.75,
        offset=0.25,
    )
    updated = merge_beatnet_output(
        raw,
        [
            [0.25, 1],
            [0.5, 2],
            [0.75, 3],
            [1.0, 4],
            [1.25, 5],
            [1.5, 6],
        ],
    )

    bars = build_bar_features(updated)
    chart = TjaChart(
        metadata=ChartMetadata(
            title="Six Eight Song",
            wave="song.ogg",
            bpm=updated.bpm,
            offset=updated.offset,
        ),
        bars=[
            ChartBar(
                index=bar.index,
                notes="100000100000",
                time_signature=bar.time_signature,
            )
            for bar in bars
        ],
    )
    tja = render_tja(chart)

    assert updated.bpm == 120
    assert len(bars) == 1
    assert bars[0].start_time == 0.25
    assert bars[0].end_time == 1.75
    assert bars[0].beat_grids == [0, 18]
    assert "BPM:120" in tja
    assert "#MEASURE 3/4\n100000100000," in tja


def test_merge_beatnet_output_keeps_six_eight_onset_grid_at_quarter_note_tempo():
    eighth_interval = 1.0 / 3.0
    raw = AudioAnalysisRaw(
        bpm=180,
        beat_times=[],
        onset_times=[0.2 + (index * eighth_interval) for index in range(24)],
        onset_strengths=[],
        duration=8.0,
        offset=0.2,
        sample_rate=1_000,
    )
    beatnet_output = [
        [0.2 + (index * eighth_interval), (index % 6) + 1]
        for index in range(19)
    ]

    updated = merge_beatnet_output(raw, beatnet_output)

    assert updated.time_signature == "6/8"
    assert updated.bpm == 90
    assert updated.beat_times[:3] == pytest.approx([0.2, 0.866667, 1.533333])
    assert updated.downbeat_times[:2] == pytest.approx([0.2, 2.2])


def test_merge_beatnet_output_keeps_three_four_pulses_as_quarter_note_bpm():
    raw = AudioAnalysisRaw(
        bpm=100,
        beat_times=[0.2, 0.8],
        onset_times=[],
        onset_strengths=[],
        duration=4.0,
        offset=0.2,
    )

    updated = merge_beatnet_output(
        raw,
        [
            [0.2, 1],
            [0.8, 2],
            [1.4, 3],
            [2.0, 1],
        ],
    )

    assert updated.time_signature == "3/4"
    assert updated.bpm == 100
    assert updated.beat_times[:4] == [0.2, 0.8, 1.4, 2.0]
    assert updated.downbeat_times[:2] == [0.2, 2.0]


def test_merge_beatnet_output_refines_timing_with_onset_grid():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[0.2 + i * 0.5 for i in range(16)],
        onset_strengths=[],
        duration=8.0,
        offset=0.1,
        sample_rate=1_000,
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

    assert updated.analyzer == "beatnet+librosa+onset-grid"
    assert updated.bpm == 120
    assert updated.beat_times[:4] == [0.2, 0.7, 1.2, 1.7]
    assert updated.offset == 0.2
    assert updated.time_signature == "3/4"
    assert updated.tempo_analysis is not None
    assert updated.tempo_analysis.accepted is True
    assert updated.tempo_analysis.selected_source == "onset-grid"


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


def test_merge_beatnet_output_ignores_nan_beat_number_rows():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[],
        onset_strengths=[],
        duration=4.0,
        offset=0.1,
    )

    updated = merge_beatnet_output(
        raw,
        [
            [0.25, 1],
            [0.5, np.nan],
            [0.75, 2],
            [1.25, 3],
            [1.75, 1],
        ],
    )

    assert updated.analyzer == "beatnet"
    assert updated.bpm == 120
    assert updated.downbeat_times[:2] == [0.25, 1.75]


def test_merge_beatnet_output_ignores_infinite_time_rows():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[],
        onset_strengths=[],
        duration=4.0,
        offset=0.1,
    )

    updated = merge_beatnet_output(
        raw,
        [
            [0.25, 1],
            [np.inf, 2],
            [0.75, 2],
            [1.25, 3],
            [1.75, 1],
        ],
    )

    assert updated.analyzer == "beatnet"
    assert updated.bpm == 120
    assert updated.downbeat_times[:2] == [0.25, 1.75]


@pytest.mark.parametrize(
    "output",
    [
        [[0.25, 1], [0.75]],
        np.array([[0.25, 1], ["invalid", 2]], dtype=object),
    ],
)
def test_merge_beatnet_output_keeps_raw_for_invalid_array_structure(output):
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[],
        onset_strengths=[],
        duration=2.0,
        offset=0.1,
    )

    assert merge_beatnet_output(raw, output) is raw


def test_enhance_with_beatnet_applies_numpy_compatibility_and_records_success(monkeypatch):
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[],
        onset_strengths=[],
        duration=2.0,
        offset=0.1,
    )
    constructor_kwargs = {}

    class FakeBeatNet:
        def __init__(self, **kwargs):
            constructor_kwargs.update(kwargs)

        def process(self, input_path):
            assert input_path == "song.ogg"
            return [[0.25, 1], [0.75, 2], [1.25, 3], [1.75, 1]]

    beatnet_package = types.ModuleType("BeatNet")
    beatnet_module = types.ModuleType("BeatNet.BeatNet")
    beatnet_module.BeatNet = FakeBeatNet
    monkeypatch.setitem(sys.modules, "BeatNet", beatnet_package)
    monkeypatch.setitem(sys.modules, "BeatNet.BeatNet", beatnet_module)
    monkeypatch.delitem(np.__dict__, "float", raising=False)
    monkeypatch.delitem(np.__dict__, "int", raising=False)

    updated = enhance_with_beatnet(Path("song.ogg"), raw)

    assert constructor_kwargs == {
        "model": 1,
        "mode": "offline",
        "inference_model": "DBN",
        "plot": [],
        "thread": False,
    }
    assert np.float is np.float64
    assert np.int is np.int_
    assert updated.analyzer == "beatnet"
    assert updated.beatnet_analysis_status == "complete"
    assert updated.beatnet_analysis_reason is None


def test_enhance_with_beatnet_keeps_raw_when_merge_fails(monkeypatch):
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.1, 0.6],
        onset_times=[],
        onset_strengths=[],
        duration=2.0,
        offset=0.1,
    )

    class FakeBeatNet:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, input_path):
            return [[0.25, 1], [0.75, 2]]

    beatnet_package = types.ModuleType("BeatNet")
    beatnet_module = types.ModuleType("BeatNet.BeatNet")
    beatnet_module.BeatNet = FakeBeatNet
    monkeypatch.setitem(sys.modules, "BeatNet", beatnet_package)
    monkeypatch.setitem(sys.modules, "BeatNet.BeatNet", beatnet_module)

    def fail_merge(raw_analysis, output):
        raise ValueError("invalid BeatNet output")

    monkeypatch.setattr("tja_ai_chartgen.audio.analyze.merge_beatnet_output", fail_merge)

    updated = enhance_with_beatnet(Path("song.ogg"), raw)

    assert updated.bpm == raw.bpm
    assert updated.beat_times == raw.beat_times
    assert updated.beatnet_analysis_status == "fallback"
    assert updated.beatnet_analysis_reason == "inference-error:ValueError"


def test_estimate_time_signature_defaults_to_four_four_for_unknown_meter():
    assert estimate_time_signature([1, 2, 3, 4]) == "4/4"
    assert estimate_time_signature([1, 2, 3, 4, 5, 6]) == "6/8"
    assert estimate_time_signature([]) == "4/4"
