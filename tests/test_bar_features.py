import pytest

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.audio.spectral import SpectralAnalysisRaw
from tja_ai_chartgen.features.bars import build_bar_features


@pytest.mark.parametrize(
    ("time_signature", "grids_per_bar"),
    [("4/4", 48), ("3/4", 36), ("6/8", 36)],
)
@pytest.mark.parametrize("offset", [-0.25, 0.0, 0.25])
def test_build_bar_features_quantizes_onsets_across_bar_boundaries(
    time_signature,
    grids_per_bar,
    offset,
):
    grid_length = 0.5 / 12
    bar_length = grids_per_bar * grid_length
    boundary = offset + bar_length
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[boundary - (0.51 * grid_length), boundary - (0.49 * grid_length)],
        onset_strengths=[],
        duration=offset + (2 * bar_length),
        offset=offset,
        time_signature=time_signature,
    )

    bars = build_bar_features(raw)

    assert bars[0].onset_grids == [grids_per_bar - 1]
    assert bars[1].onset_grids == [0]


@pytest.mark.parametrize(
    ("time_signature", "grids_per_bar"),
    [("4/4", 48), ("3/4", 36), ("6/8", 36)],
)
@pytest.mark.parametrize("offset", [-0.25, 0.0, 0.25])
def test_build_bar_features_quantizes_activity_across_bar_boundaries(
    time_signature,
    grids_per_bar,
    offset,
):
    sample_rate = 800
    hop_length = 1
    grid_length = 0.5 / 12
    bar_length = grids_per_bar * grid_length
    boundary = offset + bar_length
    previous_grid_frame = round((boundary - (0.51 * grid_length)) * sample_rate / hop_length)
    next_bar_frame = round((boundary - (0.49 * grid_length)) * sample_rate / hop_length)
    activity_envelope = [0.0] * (next_bar_frame + 1)
    activity_envelope[previous_grid_frame] = 0.5
    activity_envelope[next_bar_frame] = 1.0
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[],
        onset_strengths=[],
        activity_envelope=activity_envelope,
        duration=offset + (2 * bar_length),
        offset=offset,
        sample_rate=sample_rate,
        hop_length=hop_length,
        time_signature=time_signature,
    )

    bars = build_bar_features(raw)

    assert bars[0].activity_grids[grids_per_bar - 1] == 0.5
    assert bars[1].activity_grids[0] == 1.0


def test_build_bar_features_drops_event_that_rounds_past_last_bar():
    grid_length = 0.5 / 12
    bar_length = 48 * grid_length
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[bar_length - (0.49 * grid_length)],
        onset_strengths=[],
        duration=bar_length,
        offset=0.0,
    )

    bars = build_bar_features(raw)

    assert len(bars) == 1
    assert bars[0].onset_grids == []


def test_build_bar_features_drops_event_before_analysis_start():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[0.24],
        onset_strengths=[],
        duration=2.25,
        offset=0.25,
    )

    bars = build_bar_features(raw)

    assert bars[0].onset_grids == []


def test_build_bar_features_maps_onsets_to_canonical_grid():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0, 0.5, 1.0, 1.5, 2.0],
        onset_times=[0.0, 0.5, 1.0, 1.5],
        onset_strengths=[],
        duration=2.0,
        offset=0.0,
    )

    bars = build_bar_features(raw)

    assert bars[0].onset_grids == [0, 12, 24, 36]
    assert bars[0].accent_grids == [0, 12, 24, 36]
    assert bars[0].beat_grids == [0, 12, 24, 36]
    assert bars[0].downbeat_grid == 0
    assert bars[0].grid_features[0].beat == 1
    assert bars[0].grid_features[0].downbeat is True
    assert bars[0].grid_features[12].onset is True


def test_bar_feature_reads_legacy_grid_fields_and_serializes_generic_names():
    from tja_ai_chartgen.tja.model import BarFeature

    bar = BarFeature.model_validate(
        {
            "index": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "energy": 0.5,
            "onset_16": [0, 4],
            "accent_16": [0],
            "activity_16": [0.5, 0.25],
        }
    )

    assert bar.onset_grids == [0, 4]
    assert bar.accent_grids == [0]
    assert bar.activity_grids == [0.5, 0.25]
    payload = bar.model_dump()
    assert payload["onset_grids"] == [0, 4]
    assert payload["accent_grids"] == [0]
    assert payload["activity_grids"] == [0.5, 0.25]
    assert "onset_16" not in payload
    assert "accent_16" not in payload
    assert "activity_16" not in payload


def test_build_bar_features_respects_max_bars():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[0.0, 0.5, 2.0, 2.5],
        onset_strengths=[],
        duration=4.0,
        offset=0.0,
    )

    bars = build_bar_features(raw, max_bars=1)

    assert len(bars) == 1


def test_build_bar_features_distinguishes_three_four_and_six_eight_beats():
    three_four = build_bar_features(
        AudioAnalysisRaw(
            bpm=120,
            beat_times=[],
            onset_times=[0.0, 0.5, 1.0],
            onset_strengths=[],
            duration=1.5,
            offset=0.0,
            time_signature="3/4",
        )
    )[0]
    six_eight = build_bar_features(
        AudioAnalysisRaw(
            bpm=120,
            beat_times=[],
            onset_times=[0.0, 0.75],
            onset_strengths=[],
            duration=1.5,
            offset=0.0,
            time_signature="6/8",
        )
    )[0]

    assert three_four.beat_grids == [0, 12, 24]
    assert [feature.beat for feature in three_four.grid_features if feature.beat] == [1, 2, 3]
    assert six_eight.beat_grids == [0, 18]
    assert six_eight.accent_grids == [0, 18]
    assert [feature.beat for feature in six_eight.grid_features if feature.beat] == [1, 2]
    assert six_eight.downbeat_grid == 0


def test_build_bar_features_supports_three_four_meter():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0, 0.5, 1.0, 1.5],
        onset_times=[0.0, 0.5, 1.0],
        onset_strengths=[],
        duration=1.5,
        offset=0.0,
        time_signature="3/4",
    )

    bars = build_bar_features(raw)

    assert len(bars) == 1
    assert bars[0].time_signature == "3/4"
    assert bars[0].grids_per_bar == 36
    assert bars[0].onset_grids == [0, 12, 24]
    assert bars[0].accent_grids == [0, 12, 24]


def test_build_bar_features_keeps_meter_downbeats_on_bar_start_when_detected_beats_drift():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0.125, 0.625, 1.125, 1.625, 2.125, 2.625, 3.125, 3.625],
        onset_times=[0.125, 0.625, 1.125, 1.625, 2.125, 2.625, 3.125, 3.625],
        onset_strengths=[],
        duration=4.0,
        offset=0.0,
    )

    bars = build_bar_features(raw)

    assert [bar.downbeat_grid for bar in bars] == [0, 0]
    assert [bar.beat_grids for bar in bars] == [[0, 12, 24, 36], [0, 12, 24, 36]]
    assert bars[0].onset_grids == [3, 15, 27, 39]
    assert bars[0].grid_features[0].beat == 1
    assert bars[0].grid_features[0].downbeat is True
    assert bars[0].grid_features[0].onset is False
    assert bars[0].grid_features[3].onset is True


def test_build_bar_features_adds_grid_strength_without_guessing_fill_context():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0, 0.5, 1.0, 1.5, 2.0],
        onset_times=[0.0, 0.5],
        onset_strengths=[0.0, 1.0, 0.0, 2.0],
        duration=2.0,
        offset=0.0,
        sample_rate=2,
        hop_length=1,
    )

    bars = build_bar_features(raw)

    assert bars[0].grid_features[0].strength == 0.0
    assert bars[0].grid_features[12].strength == 0.5
    assert bars[0].phrase_position == "song_end"
    assert bars[0].fill_candidate is False


def test_build_bar_features_maps_activity_envelope_to_grid_features():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0, 0.5, 1.0, 1.5, 2.0],
        onset_times=[],
        onset_strengths=[],
        activity_envelope=[0.0, 0.5, 1.0, 0.25],
        duration=2.0,
        offset=0.0,
        sample_rate=2,
        hop_length=1,
    )

    bars = build_bar_features(raw)

    assert bars[0].onset_grids == []
    assert bars[0].activity_grids[12] == 0.5
    assert bars[0].activity_grids[24] == 1.0
    assert bars[0].grid_features[24].activity == 1.0
    assert bars[0].grid_features[24].strength == 1.0
    assert bars[0].energy > 0


def test_build_bar_features_maps_spectral_envelopes_to_bar_and_sparse_grids():
    zeros = [0.0] * 16
    low_onsets = zeros.copy()
    high_onsets = zeros.copy()
    flux = zeros.copy()
    harmonic_novelty = zeros.copy()
    texture_novelty = zeros.copy()
    low_onsets[2] = 0.8
    high_onsets[10] = 0.9
    flux[2] = 0.7
    flux[10] = 1.0
    harmonic_novelty[8] = 0.75
    texture_novelty[10] = 0.6
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[],
        onset_strengths=[],
        duration=4.0,
        offset=0.0,
        sample_rate=4,
        hop_length=1,
        spectral=SpectralAnalysisRaw(
            feature_version="spectral-v1",
            status="complete",
            frame_count=16,
            low_onset_envelope=low_onsets,
            mid_onset_envelope=zeros,
            high_onset_envelope=high_onsets,
            spectral_flux_envelope=flux,
            brightness_envelope=[0.2] * 8 + [0.8] * 8,
            harmonic_novelty_envelope=harmonic_novelty,
            texture_novelty_envelope=texture_novelty,
            percussive_ratio_envelope=[0.25] * 8 + [0.75] * 8,
        ),
    )

    bars = build_bar_features(raw)

    assert bars[0].low_onset_strength == 0.8
    assert bars[0].high_onset_strength == 0.0
    assert bars[0].spectral_flux == 0.7
    assert bars[0].brightness == 0.2
    assert bars[0].percussive_ratio == 0.25
    assert bars[1].high_onset_strength == 0.9
    assert bars[1].spectral_flux == 1.0
    assert bars[1].brightness == 0.8
    assert bars[1].harmonic_novelty == 0.75
    assert bars[1].texture_novelty == 0.6
    assert bars[0].spectral_grid_features[0].grid == 12
    assert bars[0].spectral_grid_features[0].low_onset_strength == 0.8
    assert bars[1].spectral_grid_features[0].grid == 12
    assert bars[1].spectral_grid_features[0].high_onset_strength == 0.9


def test_build_bar_features_calculates_loudness_and_sustained_activity():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[],
        onset_strengths=[],
        rms_envelope=[0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.5] * 8,
        duration=4.0,
        offset=0.0,
        sample_rate=4,
        hop_length=1,
    )

    bars = build_bar_features(raw)

    assert bars[0].rms_dbfs == pytest.approx(-55.052, abs=0.001)
    assert bars[0].peak_rms_dbfs == pytest.approx(-46.021, abs=0.001)
    assert bars[0].relative_rms_db == pytest.approx(-49.031, abs=0.001)
    assert bars[0].sustained_activity_ratio == pytest.approx(0.125)
    assert bars[1].rms_dbfs == pytest.approx(-6.021, abs=0.001)
    assert bars[1].peak_rms_dbfs == pytest.approx(-6.021, abs=0.001)
    assert bars[1].relative_rms_db == pytest.approx(0.0)
    assert bars[1].sustained_activity_ratio == pytest.approx(1.0)


def test_build_bar_features_uses_finite_loudness_floor_for_digital_silence():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[],
        onset_strengths=[],
        rms_envelope=[0.0] * 8,
        duration=2.0,
        offset=0.0,
        sample_rate=4,
        hop_length=1,
    )

    bar = build_bar_features(raw)[0]

    assert bar.rms_dbfs == -120.0
    assert bar.peak_rms_dbfs == -120.0
    assert bar.relative_rms_db == 0.0
    assert bar.sustained_activity_ratio == 0.0


@pytest.mark.parametrize("time_signature", ["3/4", "6/8"])
def test_build_bar_features_uses_meter_bar_length_for_loudness(time_signature):
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[],
        onset_times=[],
        onset_strengths=[],
        rms_envelope=[0.25] * 6 + [0.5] * 6,
        duration=3.0,
        offset=0.0,
        sample_rate=4,
        hop_length=1,
        time_signature=time_signature,
    )

    bars = build_bar_features(raw)

    assert len(bars) == 2
    assert bars[0].rms_dbfs == pytest.approx(-12.041, abs=0.001)
    assert bars[0].relative_rms_db == pytest.approx(-6.021, abs=0.001)
    assert bars[1].rms_dbfs == pytest.approx(-6.021, abs=0.001)
    assert bars[1].relative_rms_db == pytest.approx(0.0)
