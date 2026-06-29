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
