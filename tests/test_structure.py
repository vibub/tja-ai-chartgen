from math import ceil

import pytest

from tja_ai_chartgen.features.structure import analyze_song_structure
from tja_ai_chartgen.tja.model import BarFeature, InstrumentBarFeature


@pytest.mark.parametrize("phrase_length", [3, 4, 6, 8, 12])
def test_structure_detects_non_fixed_phrase_lengths(phrase_length: int):
    bars = [
        _bar(index, energy=0.2, pattern="quarter")
        for index in range(phrase_length)
    ] + [
        _bar(index + phrase_length, energy=0.82, pattern="dense", activity=0.7)
        for index in range(phrase_length)
    ]

    result = analyze_song_structure(bars)

    assert len(result.phrases) == 2
    assert result.phrases[0].start_bar == 0
    assert result.phrases[0].end_bar == phrase_length - 1
    assert result.phrases[1].start_bar == phrase_length
    assert result.bar_structures[phrase_length - 1].boundary_confidence >= 0.5
    assert {item.phrase_id for item in result.bar_structures[:phrase_length]} == {0}
    assert {item.phrase_id for item in result.bar_structures[phrase_length:]} == {1}


def test_structure_does_not_create_boundaries_every_four_bars_without_evidence():
    bars = [
        _bar(
            index,
            energy=0.48 + (0.015 if index % 2 else 0.0),
            pattern="quarter",
            activity=0.35,
        )
        for index in range(12)
    ]

    result = analyze_song_structure(bars)

    assert len(result.phrases) == 1
    assert {item.phrase_id for item in result.bar_structures} == {0}
    assert result.bars[3].phrase_position == "phrase_middle"
    assert result.bars[7].phrase_position == "phrase_middle"
    assert result.confidence < 0.5


def test_structure_marks_build_up_peak_drop_and_breakdown_roles():
    bars = [
        _bar(0, energy=0.18, pattern="sparse", activity=0.2),
        _bar(1, energy=0.3, pattern="quarter", activity=0.3),
        _bar(2, energy=0.48, pattern="eighth", activity=0.45),
        _bar(3, energy=0.7, pattern="dense", activity=0.65),
        _bar(4, energy=0.98, pattern="dense", activity=0.9),
        _bar(5, energy=0.28, pattern="sparse", activity=0.35),
        _bar(6, energy=0.2, pattern="none", activity=0.42),
        _bar(7, energy=0.18, pattern="none", activity=0.4),
    ]

    result = analyze_song_structure(bars)
    roles = [item.transition_role for item in result.bar_structures]

    assert "build_up" in roles[1:4]
    assert roles[4] == "peak"
    assert roles[5] == "drop"
    assert "breakdown" in roles[6:]


def test_structure_uses_spectral_growth_for_build_up_role():
    bars = [_bar(index, energy=0.5, pattern="quarter", activity=0.45) for index in range(4)]
    for index, intensity in enumerate([0.1, 0.35, 0.65, 0.95]):
        bars[index] = bars[index].model_copy(
            update={
                "spectral_flux": intensity,
                "brightness": intensity,
                "percussive_ratio": intensity,
                "high_onset_strength": intensity,
            }
        )

    result = analyze_song_structure(bars)

    assert "build_up" in [item.transition_role for item in result.bar_structures[:3]]


def test_structure_uses_low_percussive_balance_for_breakdown_role():
    bars = [_bar(index, energy=0.25, pattern="quarter", activity=0.5) for index in range(4)]
    bars = [
        bar.model_copy(
            update={
                "spectral_flux": 0.15,
                "brightness": 0.35,
                "percussive_ratio": 0.1,
                "mid_onset_strength": 0.2,
            }
        )
        for bar in bars
    ]

    result = analyze_song_structure(bars)

    assert "breakdown" in [item.transition_role for item in result.bar_structures]


def test_structure_uses_timbre_and_harmonic_novelty_for_phrase_boundaries():
    bars = [_bar(index, energy=0.5, pattern="quarter", activity=0.45) for index in range(8)]
    for index in range(4):
        bars[index] = bars[index].model_copy(
            update={
                "low_onset_strength": 0.85,
                "mid_onset_strength": 0.1,
                "high_onset_strength": 0.05,
                "spectral_flux": 0.25,
                "brightness": 0.15,
                "percussive_ratio": 0.2,
            }
        )
    for index in range(4, 8):
        bars[index] = bars[index].model_copy(
            update={
                "low_onset_strength": 0.05,
                "mid_onset_strength": 0.2,
                "high_onset_strength": 0.9,
                "spectral_flux": 0.9,
                "brightness": 0.85,
                "percussive_ratio": 0.85,
                "harmonic_novelty": 1.0 if index == 4 else 0.1,
                "texture_novelty": 0.8 if index == 4 else 0.1,
            }
        )

    result = analyze_song_structure(bars)

    assert len(result.phrases) == 2
    assert result.phrases[0].end_bar == 3
    assert result.phrases[1].start_bar == 4
    assert result.bar_structures[3].boundary_confidence >= 0.4


def test_structure_uses_vocal_entry_and_source_switch_for_phrase_boundary():
    bars = [_bar(index, energy=0.5, pattern="quarter", activity=0.45) for index in range(8)]
    for index in range(4):
        bars[index] = bars[index].model_copy(
            update={
                "instrument": InstrumentBarFeature(
                    vocal_activity=0.8,
                    vocal_presence_ratio=0.9,
                    drum_activity=0.15,
                    other_activity=0.25,
                    dominant_source="vocals",
                    confidence=0.8,
                )
            }
        )
    for index in range(4, 8):
        bars[index] = bars[index].model_copy(
            update={
                "instrument": InstrumentBarFeature(
                    vocal_activity=0.1,
                    drum_activity=0.85,
                    bass_activity=0.55,
                    other_activity=0.65,
                    guitar=0.75,
                    dominant_source="drums",
                    dominant_instrument="guitar",
                    confidence=0.85,
                )
            }
        )

    result = analyze_song_structure(bars)

    assert len(result.phrases) == 2
    assert result.phrases[0].end_bar == 3
    assert result.phrases[1].start_bar == 4
    assert result.bar_structures[4].instrument.dominant_source == "drums"


def test_structure_uses_instrument_growth_for_build_up_role():
    bars = [_bar(index, energy=0.5, pattern="quarter", activity=0.45) for index in range(4)]
    for index, value in enumerate([0.1, 0.35, 0.65, 0.95]):
        bars[index] = bars[index].model_copy(
            update={
                "instrument": InstrumentBarFeature(
                    drum_activity=value,
                    bass_activity=value * 0.7,
                    other_activity=value * 0.8,
                    synth=value,
                    dominant_source="drums",
                    confidence=value,
                )
            }
        )

    result = analyze_song_structure(bars)

    assert "build_up" in [item.transition_role for item in result.bar_structures[:3]]


def test_structure_uses_vocal_only_texture_for_breakdown_role():
    bars = [_bar(index, energy=0.5, pattern="quarter", activity=0.5) for index in range(4)]
    bars = [
        bar.model_copy(
            update={
                "instrument": InstrumentBarFeature(
                    vocal_activity=0.7,
                    vocal_presence_ratio=0.9,
                    drum_activity=0.05,
                    bass_activity=0.05,
                    other_activity=0.2,
                    dominant_source="vocals",
                    confidence=0.8,
                )
            }
        )
        for bar in bars
    ]

    result = analyze_song_structure(bars)

    assert "breakdown" in [item.transition_role for item in result.bar_structures]


def test_structure_reuses_section_id_for_returning_phrase():
    bars = [
        *[_bar(index, energy=0.25, pattern="quarter", activity=0.25) for index in range(4)],
        *[
            _bar(index + 4, energy=0.82, pattern="offbeat", activity=0.75)
            for index in range(4)
        ],
        *[
            _bar(index + 8, energy=0.25, pattern="quarter", activity=0.25)
            for index in range(4)
        ],
    ]

    result = analyze_song_structure(bars)

    assert len(result.phrases) == 3
    assert result.phrases[0].section_id == result.phrases[2].section_id
    assert result.phrases[0].section_id != result.phrases[1].section_id
    assert result.phrases[0].section == result.phrases[2].section


def test_fill_candidates_follow_detected_boundaries_and_are_rate_limited():
    bars = [
        *[_bar(index, energy=0.25, pattern="quarter") for index in range(5)],
        *[_bar(index + 5, energy=0.9, pattern="dense", activity=0.8) for index in range(5)],
        *[_bar(index + 10, energy=0.2, pattern="sparse", activity=0.25) for index in range(5)],
    ]

    result = analyze_song_structure(bars)
    candidate_indexes = [bar.index for bar in result.bars if bar.fill_candidate]
    phrase_end_indexes = {phrase.end_bar for phrase in result.phrases}

    assert candidate_indexes
    assert set(candidate_indexes) <= phrase_end_indexes
    assert len(candidate_indexes) <= max(1, ceil(len(bars) / 6))
    assert 3 not in candidate_indexes
    assert 7 not in candidate_indexes


def _bar(
    index: int,
    *,
    energy: float,
    pattern: str,
    activity: float = 0.0,
) -> BarFeature:
    patterns = {
        "none": [],
        "sparse": [0],
        "quarter": [0, 12, 24, 36],
        "offbeat": [6, 18, 30, 42],
        "eighth": list(range(0, 48, 6)),
        "dense": list(range(0, 48, 3)),
    }
    onset_grids = patterns[pattern]
    activity_grids = [activity] * 48 if activity else [0.0] * 48
    return BarFeature(
        index=index,
        start_time=index * 2.0,
        end_time=(index + 1) * 2.0,
        energy=energy,
        grids_per_bar=48,
        onset_grids=onset_grids,
        accent_grids=[grid for grid in onset_grids if grid in {0, 12, 24, 36}],
        activity_grids=activity_grids,
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
    )
