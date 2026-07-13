from tja_ai_chartgen.features.sections import assign_sections
from tja_ai_chartgen.tja.model import BarFeature


def test_assign_sections_uses_relative_energy_for_quiet_songs():
    bars = [
        BarFeature(index=index, start_time=index * 2, end_time=(index + 1) * 2, energy=energy, onset_16=onsets)
        for index, (energy, onsets) in enumerate(
            [(0, []), (0.02, [0]), (0.04, [0, 4, 8, 12]), (0.09, [0, 2, 4, 6, 8])]
        )
    ]

    assigned = assign_sections(bars)

    assert [bar.section for bar in assigned] == ["intro", "intro", "intro", "intro"]


def test_assign_sections_keeps_mixed_bars_in_one_detected_phrase_section():
    bars = [
        BarFeature(index=index, start_time=index * 2, end_time=(index + 1) * 2, energy=0, onset_16=[])
        for index in range(8)
    ]
    bars.extend(
        [
            BarFeature(index=8, start_time=16, end_time=18, energy=0, onset_16=[]),
            BarFeature(index=9, start_time=18, end_time=20, energy=0.04, onset_16=[0, 4, 8, 12]),
            BarFeature(index=10, start_time=20, end_time=22, energy=0.09, onset_16=[0, 2, 4, 6, 8]),
            BarFeature(index=11, start_time=22, end_time=24, energy=0.02, onset_16=[0]),
        ]
    )
    bars.extend(
        [
            BarFeature(index=index, start_time=index * 2, end_time=(index + 1) * 2, energy=0, onset_16=[])
            for index in range(12, 20)
        ]
    )

    assigned = assign_sections(bars)

    assert assigned[8].section != "break"
    assert assigned[9].section in {"verse", "chorus"}
    assert assigned[10].section in {"verse", "chorus"}
    for phrase_id in {bar.phrase_id for bar in assigned[8:12]}:
        phrase_bars = [bar for bar in assigned[8:12] if bar.phrase_id == phrase_id]
        assert len({bar.section_id for bar in phrase_bars}) == 1
        assert len({bar.section for bar in phrase_bars}) == 1


def test_assign_sections_uses_activity_for_sustained_music():
    bars = [
        BarFeature(index=index, start_time=index * 2, end_time=(index + 1) * 2, energy=0, onset_16=[])
        for index in range(8)
    ]
    bars.extend(
        [
            BarFeature(index=8, start_time=16, end_time=18, energy=0, onset_16=[], activity_16=[0.4] * 16),
            BarFeature(index=9, start_time=18, end_time=20, energy=0, onset_16=[]),
        ]
    )
    bars.extend(
        [
            BarFeature(index=index, start_time=index * 2, end_time=(index + 1) * 2, energy=0, onset_16=[])
            for index in range(10, 18)
        ]
    )

    assigned = assign_sections(bars)

    assert assigned[8].section == "verse"
    assert assigned[9].section == "verse"
    assert assigned[8].section_id == assigned[9].section_id
