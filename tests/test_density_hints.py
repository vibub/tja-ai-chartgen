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
    assert hints[0].min_hits == 2
    assert hints[0].max_hits == 12
    assert hints[1].allow_empty is True
    assert hints[1].count_in_quality_average is False
    assert hints[2].min_hits == 5
    assert hints[2].max_hits == 16
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
            "max_hits": 0,
            "allow_empty": True,
            "count_in_quality_average": False,
            "target_scale": 1.0,
            "reason": "low-energy musical pause or break",
        }
    ]


def test_build_density_hints_caps_silent_rest_and_phrase_fill():
    bars = [
        BarFeature(index=9, start_time=18, end_time=20, energy=0, section="break"),
        BarFeature(index=10, start_time=20, end_time=22, energy=0.03, onset_16=[0, 4], section="break"),
        BarFeature(
            index=11,
            start_time=22,
            end_time=24,
            energy=0.04,
            onset_16=[0, 4, 8, 12],
            phrase_position="phrase_end",
            fill_candidate=True,
            fill_candidate_score=0.8,
            transition_role="cadence",
            section="verse",
        ),
    ]

    hints = build_density_hints(bars)

    assert [(hint.kind, hint.min_hits, hint.max_hits) for hint in hints] == [
        ("rest", 0, 0),
        ("sparse", 1, 4),
        ("fill", 3, 14),
    ]


def test_build_density_hints_uses_structure_role_target_scales():
    bars = [
        BarFeature(
            index=0,
            start_time=0,
            end_time=2,
            energy=0.5,
            onset_grids=[0, 4, 8, 12],
            transition_role="build_up",
            phrase_progress=0.0,
        ),
        BarFeature(
            index=1,
            start_time=2,
            end_time=4,
            energy=0.5,
            onset_grids=[0, 4, 8, 12],
            transition_role="build_up",
            phrase_progress=1.0,
        ),
        BarFeature(
            index=2,
            start_time=4,
            end_time=6,
            energy=0.5,
            onset_grids=[0, 4, 8, 12],
            transition_role="peak",
        ),
        BarFeature(
            index=3,
            start_time=6,
            end_time=8,
            energy=0.5,
            onset_grids=[0, 4, 8, 12],
            transition_role="breakdown",
        ),
    ]

    hints = build_density_hints(bars)

    assert [hint.target_scale for hint in hints] == [0.9, 1.1, 1.12, 0.72]


def test_phrase_end_without_fill_score_does_not_force_fill_hint():
    bar = BarFeature(
        index=3,
        start_time=6,
        end_time=8,
        energy=0.2,
        onset_grids=[0, 4, 8, 12],
        phrase_position="phrase_end",
        fill_candidate=False,
    )

    hint = build_density_hints([bar])[0]

    assert hint.kind == "normal"
    assert hint.reason == "phrase ending without enough fill evidence"


def test_build_density_hints_treats_sustained_activity_as_normal():
    bars = [
        BarFeature(
            index=20,
            start_time=40,
            end_time=42,
            energy=0.03,
            onset_16=[],
            activity_16=[0.0, 0.4, 0.5, 0.45] + [0.0] * 12,
            section="break",
        )
    ]

    hints = build_density_hints(bars)

    assert [(hint.kind, hint.min_hits, hint.max_hits, hint.reason) for hint in hints] == [
        ("normal", 3, 12, "sustained musical activity without strong onsets")
    ]


def test_build_density_hints_extends_quiet_edge_silence_with_activity_floor():
    bars = [
        BarFeature(index=0, start_time=0, end_time=2, energy=0, activity_16=[0.0] * 16),
        BarFeature(index=1, start_time=2, end_time=4, energy=0.026, activity_16=[0.0] * 15 + [0.41]),
        BarFeature(index=2, start_time=4, end_time=6, energy=0.6, activity_16=[0.7] * 16),
    ]

    hints = build_density_hints(bars)

    assert [hint.kind for hint in hints] == ["silent", "silent", "normal"]


def test_build_density_hints_treats_edge_transient_noise_as_silent():
    bars = [
        BarFeature(
            index=0,
            start_time=0,
            end_time=2,
            energy=0.068,
            onset_16=[0, 11, 12],
            rms_dbfs=-59.7,
            peak_rms_dbfs=-46.7,
            relative_rms_db=-53.5,
            sustained_activity_ratio=0.098,
            section="intro",
        ),
        BarFeature(
            index=1,
            start_time=2,
            end_time=4,
            energy=0.131,
            onset_16=[0, 1, 2, 4, 7, 8, 10],
            rms_dbfs=-34.9,
            peak_rms_dbfs=-29.0,
            relative_rms_db=-28.7,
            sustained_activity_ratio=0.773,
            section="intro",
        ),
    ]

    hints = build_density_hints(bars)

    assert [hint.kind for hint in hints] == ["silent", "normal"]
    assert hints[0].max_hits == 0
    assert hints[1].min_hits > 0
