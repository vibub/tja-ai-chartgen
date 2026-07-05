from tja_ai_chartgen.features.density import build_density_hints, density_hint_payload
from tja_ai_chartgen.tja.model import BarFeature


def test_build_density_hints_marks_middle_rest_and_edge_silence():
    bars = [
        BarFeature(index=5, start_time=10, end_time=12, energy=0.3, onset_16=[0, 4]),
        BarFeature(index=6, start_time=12, end_time=14, energy=0.01, section="break"),
        BarFeature(index=7, start_time=14, end_time=16, energy=0.7, onset_16=[0, 4, 8, 12]),
        BarFeature(index=8, start_time=16, end_time=18, energy=0, phrase_position="song_end", section="outro"),
    ]

    hints = build_density_hints(bars)

    assert [hint.kind for hint in hints] == ["normal", "rest", "dense", "silent"]
    assert hints[1].allow_empty is True
    assert hints[1].count_in_quality_average is False
    assert hints[2].count_in_quality_average is True
    assert hints[3].max_hits == 0


def test_density_hint_payload_is_json_ready():
    bars = [BarFeature(index=5, start_time=10, end_time=12, energy=0.01, section="break")]

    payload = density_hint_payload(bars)

    assert payload == [
        {
            "bar": 1,
            "kind": "rest",
            "min_hits": 0,
            "max_hits": 2,
            "allow_empty": True,
            "count_in_quality_average": False,
            "reason": "low-energy musical pause or break",
        }
    ]
