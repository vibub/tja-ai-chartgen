from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.features.bars import build_bar_features


def test_build_bar_features_maps_onsets_to_16th_grid():
    raw = AudioAnalysisRaw(
        bpm=120,
        beat_times=[0, 0.5, 1.0, 1.5, 2.0],
        onset_times=[0.0, 0.5, 1.0, 1.5],
        onset_strengths=[],
        duration=2.0,
        offset=0.0,
    )

    bars = build_bar_features(raw)

    assert bars[0].onset_16 == [0, 4, 8, 12]
    assert bars[0].accent_16 == [0, 4, 8, 12]
    assert bars[0].beat_grids == [0, 4, 8, 12]
    assert bars[0].downbeat_grid == 0
    assert bars[0].grid_features[0].beat == 1
    assert bars[0].grid_features[0].downbeat is True
    assert bars[0].grid_features[4].onset is True


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
    assert bars[0].grids_per_bar == 12
    assert bars[0].onset_16 == [0, 4, 8]
    assert bars[0].accent_16 == [0, 4, 8]


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
    assert [bar.beat_grids for bar in bars] == [[0, 4, 8, 12], [0, 4, 8, 12]]
    assert bars[0].onset_16 == [1, 5, 9, 13]
    assert bars[0].grid_features[0].beat == 1
    assert bars[0].grid_features[0].downbeat is True
    assert bars[0].grid_features[0].onset is False
    assert bars[0].grid_features[1].onset is True


def test_build_bar_features_adds_grid_strength_and_phrase_context():
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
    assert bars[0].grid_features[4].strength == 0.5
    assert bars[0].phrase_position == "song_end"
    assert bars[0].fill_candidate is True
