from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.tja.model import BarFeature


def test_edge_silence_ignores_low_loudness_transient_onsets_at_song_start():
    bars = [
        _bar(
            0,
            energy=0.068,
            onset_16=[0, 11, 12],
            rms_dbfs=-59.7,
            peak_rms_dbfs=-46.7,
            relative_rms_db=-53.5,
            sustained_activity_ratio=0.098,
        ),
        _bar(
            1,
            energy=0.131,
            onset_16=[0, 1, 2, 4, 7, 8, 10],
            rms_dbfs=-34.9,
            peak_rms_dbfs=-29.0,
            relative_rms_db=-28.7,
            sustained_activity_ratio=0.773,
        ),
    ]

    assert edge_silence_indexes(bars) == {0}


def test_edge_silence_ignores_low_loudness_transient_onsets_at_song_end():
    bars = [
        _bar(0, energy=0.5, onset_16=[0, 4, 8, 12]),
        _bar(
            1,
            energy=0.06,
            onset_16=[3, 9],
            phrase_position="song_end",
            rms_dbfs=-58.0,
            peak_rms_dbfs=-45.0,
            relative_rms_db=-50.0,
            sustained_activity_ratio=0.1,
        ),
    ]

    assert edge_silence_indexes(bars) == {1}


def test_edge_silence_preserves_sustained_quiet_music():
    bars = [
        _bar(
            0,
            energy=0.02,
            onset_16=[12],
            rms_dbfs=-55.0,
            peak_rms_dbfs=-45.0,
            relative_rms_db=-48.0,
            sustained_activity_ratio=0.7,
        ),
        _bar(1, energy=0.5, onset_16=[0, 4, 8, 12]),
    ]

    assert edge_silence_indexes(bars) == set()


def test_edge_silence_preserves_audible_weak_pickup():
    bars = [
        _bar(
            0,
            energy=0.02,
            onset_16=[14],
            rms_dbfs=-55.0,
            peak_rms_dbfs=-35.0,
            relative_rms_db=-48.0,
            sustained_activity_ratio=0.1,
        ),
        _bar(1, energy=0.5, onset_16=[0, 4, 8, 12]),
    ]

    assert edge_silence_indexes(bars) == set()


def test_edge_silence_preserves_onset_rich_quiet_rhythm():
    bars = [
        _bar(
            0,
            energy=0.08,
            onset_16=[0, 3, 6, 9, 12],
            rms_dbfs=-55.0,
            peak_rms_dbfs=-45.0,
            relative_rms_db=-48.0,
            sustained_activity_ratio=0.1,
        ),
        _bar(1, energy=0.5, onset_16=[0, 4, 8, 12]),
    ]

    assert edge_silence_indexes(bars) == set()


def test_edge_silence_does_not_force_matching_middle_bar_to_silence():
    bars = [
        _bar(0, energy=0.5, onset_16=[0, 4, 8, 12]),
        _bar(
            1,
            energy=0.06,
            onset_16=[3, 9],
            rms_dbfs=-58.0,
            peak_rms_dbfs=-45.0,
            relative_rms_db=-50.0,
            sustained_activity_ratio=0.1,
        ),
        _bar(2, energy=0.5, onset_16=[0, 4, 8, 12], phrase_position="song_end"),
    ]

    assert edge_silence_indexes(bars) == set()


def test_edge_silence_stops_after_first_musical_bar():
    bars = [
        _bar(
            0,
            energy=0.06,
            onset_16=[3, 9],
            rms_dbfs=-58.0,
            peak_rms_dbfs=-45.0,
            relative_rms_db=-50.0,
            sustained_activity_ratio=0.1,
        ),
        _bar(1, energy=0.5, onset_16=[0, 4, 8, 12]),
        _bar(2, energy=0.0),
    ]

    assert edge_silence_indexes(bars) == {0}


def test_edge_silence_keeps_legacy_behavior_without_loudness_fields():
    bars = [
        _bar(0, energy=0.06, onset_16=[3]),
        _bar(1, energy=0.0, phrase_position="song_end"),
    ]

    assert edge_silence_indexes(bars) == {1}


def _bar(
    index: int,
    *,
    energy: float,
    onset_16: list[int] | None = None,
    phrase_position: str = "phrase_middle",
    rms_dbfs: float | None = None,
    peak_rms_dbfs: float | None = None,
    relative_rms_db: float | None = None,
    sustained_activity_ratio: float | None = None,
) -> BarFeature:
    return BarFeature(
        index=index,
        start_time=index * 2.0,
        end_time=(index + 1) * 2.0,
        energy=energy,
        onset_16=onset_16 or [],
        phrase_position=phrase_position,
        rms_dbfs=rms_dbfs,
        peak_rms_dbfs=peak_rms_dbfs,
        relative_rms_db=relative_rms_db,
        sustained_activity_ratio=sustained_activity_ratio,
    )
