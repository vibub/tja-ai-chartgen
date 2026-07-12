import pytest

from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import BarFeature, GridFeature
from tja_ai_chartgen.tja.quality import playable_hit_count


def test_generate_fallback_chart_bars_outputs_valid_16_char_notes():
    bars = [
        _feature(0, energy=0.1),
        _feature(1, energy=0.5),
        _feature(2, energy=0.7),
        _feature(3, energy=0.9),
    ]

    chart_bars = generate_fallback_chart_bars(bars)

    assert len(chart_bars) == len(bars)
    for expected_index, chart_bar in enumerate(chart_bars):
        assert chart_bar.index == expected_index
        assert len(chart_bar.notes) == 16
        assert set(chart_bar.notes) <= set("01234")


def test_generate_fallback_chart_bars_density_controls_hit_count():
    bars = [_feature(0, energy=0.95), _feature(1, energy=0.95)]

    low_bars = generate_fallback_chart_bars(bars, density="low")
    max_bars = generate_fallback_chart_bars(bars, density="max")

    assert [playable_hit_count(bar.notes) for bar in low_bars] == [5, 5]
    assert all(
        playable_hit_count(max_bar.notes) > playable_hit_count(low_bar.notes)
        for low_bar, max_bar in zip(low_bars, max_bars, strict=True)
    )


def test_different_onsets_change_placement_for_same_index_and_energy():
    first = _feature(0, energy=0.5, onsets=[0, 4, 8, 12])
    second = _feature(0, energy=0.5, onsets=[2, 6, 10, 14])

    first_notes = generate_fallback_chart_bars([first], density="medium")[0].notes
    second_notes = generate_fallback_chart_bars([second], density="medium")[0].notes

    assert first_notes != second_notes
    assert _hit_grids(first_notes).issuperset({0, 4, 8, 12})
    assert _hit_grids(second_notes).issuperset({2, 6, 10, 14})


def test_accent_and_strength_prioritize_competing_onsets():
    bar = _feature(
        0,
        energy=0.3,
        onsets=[1, 3, 5, 7],
        accents=[7],
        strengths={1: 0.1, 3: 0.2, 5: 0.9, 7: 0.3},
    )

    notes = generate_fallback_chart_bars([bar], density="low")[0].notes

    assert _hit_grids(notes) == {5, 7}


def test_missing_onsets_falls_back_to_downbeat_and_beats():
    bar = _feature(0, energy=0.5, beats=[0, 4, 8, 12], downbeat=0)

    notes = generate_fallback_chart_bars([bar], density="medium")[0].notes

    assert 0 in _hit_grids(notes)
    assert _hit_grids(notes).issuperset({0, 4, 8, 12})


def test_feature_driven_generation_is_deterministic():
    bar = _feature(
        0,
        energy=0.7,
        onsets=[1, 4, 7, 10, 13],
        accents=[4, 13],
        strengths={1: 0.2, 4: 0.9, 7: 0.4, 10: 0.6, 13: 0.8},
    )

    first = generate_fallback_chart_bars([bar], style="hybrid", density="high")[0]
    second = generate_fallback_chart_bars([bar], style="hybrid", density="high")[0]

    assert first == second


def test_generate_fallback_chart_bars_applies_style_coloring():
    bar = _feature(0, energy=0.7, onsets=[0, 2, 4, 6, 8, 10, 12, 14], accents=[0, 8])

    technical = generate_fallback_chart_bars([bar], style="technical", density="high")[0].notes
    stamina = generate_fallback_chart_bars([bar], style="stamina", density="high")[0].notes
    hybrid = generate_fallback_chart_bars([bar], style="hybrid", density="high")[0].notes
    performance = generate_fallback_chart_bars([bar], style="performance", density="high")[0].notes

    assert len({technical, stamina, hybrid, performance}) == 4
    assert set(performance) & {"3", "4"}


def test_generate_fallback_chart_bars_keeps_edge_silence_empty():
    bars = [
        _feature(0, energy=0, section="intro"),
        _feature(1, energy=0.9, onsets=[0, 4, 8, 12]),
        _feature(2, energy=0, phrase_position="song_end", section="outro"),
    ]

    chart_bars = generate_fallback_chart_bars(bars, density="max", special_notes=True)

    assert chart_bars[0].notes == "0" * 16
    assert playable_hit_count(chart_bars[1].notes) > 0
    assert chart_bars[2].notes == "0" * 16


def test_generate_fallback_chart_bars_keeps_middle_musical_rest_empty():
    bars = [
        _feature(0, energy=0.9, onsets=[0, 4, 8, 12]),
        _feature(1, energy=0.01, section="break"),
        _feature(2, energy=0.9, onsets=[0, 4, 8, 12]),
    ]

    chart_bars = generate_fallback_chart_bars(bars, density="max")

    assert playable_hit_count(chart_bars[0].notes) > 0
    assert chart_bars[1].notes == "0" * 16
    assert playable_hit_count(chart_bars[2].notes) > 0


def test_generate_fallback_chart_bars_rejects_invalid_density():
    bars = [_feature(0, energy=0.5)]

    with pytest.raises(ValueError, match="Invalid density"):
        generate_fallback_chart_bars(bars, density="extreme")


@pytest.mark.parametrize(
    ("grids", "time_signature"),
    [(12, "3/4"), (16, "4/4")],
)
def test_generate_fallback_chart_bars_supports_meter_grid_lengths(grids, time_signature):
    beats = [0, grids // 4, grids // 2, (grids * 3) // 4]
    bar = _feature(
        0,
        energy=0.5,
        grids=grids,
        time_signature=time_signature,
        onsets=beats,
        beats=beats,
    )

    chart_bar = generate_fallback_chart_bars([bar], density="medium")[0]

    assert len(chart_bar.notes) == grids
    assert set(chart_bar.notes) <= set("01234")
    assert chart_bar.time_signature == time_signature


def test_high_density_short_bar_uses_speed_cap():
    bar = _feature(
        0,
        energy=0.95,
        end_time=0.5,
        onsets=list(range(16)),
        accents=[0, 4, 8, 12],
    )

    chart_bar = generate_fallback_chart_bars([bar], density="max")[0]

    assert playable_hit_count(chart_bar.notes) <= 6
    assert playable_hit_count(chart_bar.notes) < 16


def test_legacy_feature_without_grid_features_still_uses_onset_lists():
    bar = BarFeature(
        index=0,
        start_time=0,
        end_time=2,
        energy=0.5,
        onset_16=[1, 5, 9, 13],
        accent_16=[5],
        beat_grids=[0, 4, 8, 12],
        downbeat_grid=0,
    )

    chart_bar = generate_fallback_chart_bars([bar], density="medium")[0]

    assert _hit_grids(chart_bar.notes).issuperset({1, 5, 9, 13})


def test_generate_fallback_chart_bars_can_emit_rolls_and_balloons():
    bars = [_feature(index, energy=0.9) for index in range(8)]

    chart_bars = generate_fallback_chart_bars(bars, density="high", special_notes=True)

    assert chart_bars[3].notes == "5000000080000000"
    assert chart_bars[7].notes == "7000000080000000"
    assert chart_bars[7].balloon_counts == [8]


def _feature(
    index: int,
    *,
    energy: float,
    end_time: float | None = None,
    grids: int = 16,
    time_signature: str = "4/4",
    onsets: list[int] | None = None,
    accents: list[int] | None = None,
    strengths: dict[int, float] | None = None,
    beats: list[int] | None = None,
    downbeat: int | None = 0,
    phrase_position: str = "unknown",
    section: str = "unknown",
) -> BarFeature:
    onsets = onsets or []
    accents = accents or []
    strengths = strengths or {}
    beats = beats or []
    grid_features = [
        GridFeature(
            grid=grid,
            onset=grid in onsets,
            accent=grid in accents,
            beat=beats.index(grid) + 1 if grid in beats else None,
            downbeat=grid == downbeat,
            strength=strengths.get(grid, 1.0 if grid in onsets else 0.0),
        )
        for grid in range(grids)
    ]
    return BarFeature(
        index=index,
        start_time=float(index * 2),
        end_time=float(end_time if end_time is not None else (index + 1) * 2),
        energy=energy,
        time_signature=time_signature,
        grids_per_bar=grids,
        onset_16=onsets,
        accent_16=accents,
        grid_features=grid_features,
        beat_grids=beats,
        downbeat_grid=downbeat,
        phrase_position=phrase_position,
        section=section,
    )


def _hit_grids(notes: str) -> set[int]:
    return {index for index, note in enumerate(notes) if note != "0"}
