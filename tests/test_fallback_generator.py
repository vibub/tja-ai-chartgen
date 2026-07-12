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

    assert all(playable_hit_count(bar.notes) >= 5 for bar in low_bars)
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

    notes = generate_fallback_chart_bars(
        [bar], density="low", course="Easy", level=1
    )[0].notes

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
        _feature(
            0,
            energy=0.068,
            onsets=[0, 11, 12],
            rms_dbfs=-59.7,
            peak_rms_dbfs=-46.7,
            relative_rms_db=-53.5,
            sustained_activity_ratio=0.098,
            section="intro",
        ),
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


@pytest.mark.parametrize("bpm", [120, 180, 240])
def test_course_load_is_monotonic_at_common_bpms(bpm):
    duration = 240 / bpm
    bar = _feature(
        0,
        energy=0.95,
        end_time=duration,
        onsets=list(range(16)),
        accents=[0, 4, 8, 12],
        beats=[0, 4, 8, 12],
    )

    loads = []
    for course, level in [("Easy", 3), ("Normal", 5), ("Hard", 7), ("Oni", 10)]:
        chart_bar = generate_fallback_chart_bars(
            [bar], density="auto", course=course, level=level
        )[0]
        loads.append(playable_hit_count(chart_bar.notes) / duration)

    assert loads == sorted(loads)
    assert loads[0] < loads[-1]


@pytest.mark.parametrize("bpm", [120, 180, 240])
def test_dense_music_keeps_observable_default_course_steps(bpm):
    duration = 240 / bpm
    bars = [
        _feature(
            0,
            energy=0.95,
            end_time=duration,
            onsets=list(range(16)),
            accents=[0, 4, 8, 12],
            beats=[0, 4, 8, 12],
        )
    ]

    hit_counts = []
    for course, level in [("Easy", 3), ("Normal", 5), ("Hard", 7), ("Oni", 10)]:
        chart_bars = generate_fallback_chart_bars(
            bars, density="auto", course=course, level=level
        )
        hit_counts.append(sum(playable_hit_count(bar.notes) for bar in chart_bars))

    assert all(lower < higher for lower, higher in zip(hit_counts, hit_counts[1:]))


def test_course_load_can_expand_active_normal_hint_without_filling_sparse_music():
    active_bars = [
        _feature(
            index,
            energy=0.3,
            onsets=[0, 4, 8, 12],
            accents=[0, 8],
            beats=[0, 4, 8, 12],
        )
        for index in range(4)
    ]
    hard = generate_fallback_chart_bars(
        active_bars, density="auto", course="Hard", level=7
    )
    oni = generate_fallback_chart_bars(
        active_bars, density="auto", course="Oni", level=10
    )

    assert sum(playable_hit_count(bar.notes) for bar in oni) > sum(
        playable_hit_count(bar.notes) for bar in hard
    )

    sparse = _feature(0, energy=0.02, onsets=[0, 8], beats=[0, 4, 8, 12])
    sparse_oni = generate_fallback_chart_bars(
        [sparse], density="max", course="Oni", level=10
    )[0]
    assert playable_hit_count(sparse_oni.notes) <= 4


def test_density_offsets_stay_inside_course_gradient():
    bar = _feature(
        0,
        energy=0.95,
        onsets=list(range(16)),
        accents=[0, 4, 8, 12],
    )

    easy_max = playable_hit_count(
        generate_fallback_chart_bars(
            [bar], density="max", course="Easy", level=3
        )[0].notes
    )
    oni_low = playable_hit_count(
        generate_fallback_chart_bars(
            [bar], density="low", course="Oni", level=10
        )[0].notes
    )

    assert easy_max < oni_low


def test_high_bpm_reduces_grid_occupancy_for_oni():
    hit_counts = []
    for bpm in (120, 180, 240):
        bar = _feature(
            0,
            energy=0.95,
            end_time=240 / bpm,
            onsets=list(range(16)),
            accents=[0, 4, 8, 12],
        )
        chart_bar = generate_fallback_chart_bars(
            [bar], density="max", course="Oni", level=10
        )[0]
        hit_counts.append(playable_hit_count(chart_bar.notes))

    assert hit_counts == sorted(hit_counts, reverse=True)
    assert hit_counts[-1] < 16


def test_density_adjusts_load_within_course():
    bar = _feature(0, energy=0.95, onsets=list(range(16)))

    hit_counts = [
        playable_hit_count(
            generate_fallback_chart_bars(
                [bar], density=density, course="Normal", level=5
            )[0].notes
        )
        for density in ("low", "auto", "high", "max")
    ]

    assert hit_counts == sorted(hit_counts)
    assert hit_counts[0] < hit_counts[-1]


def test_generate_fallback_chart_bars_rejects_unknown_course():
    with pytest.raises(ValueError, match="Invalid course"):
        generate_fallback_chart_bars(
            [_feature(0, energy=0.5)], course="Ura", level=10
        )


def test_course_name_is_case_insensitive():
    bar = _feature(0, energy=0.95, onsets=list(range(16)))

    expected = generate_fallback_chart_bars([bar], course="Easy", level=3)
    actual = generate_fallback_chart_bars([bar], course="easy", level=3)

    assert actual == expected


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


def test_special_notes_require_phrase_or_fill_candidate():
    bars = [_feature(index, energy=0.9, onsets=[8, 12, 14]) for index in range(8)]

    chart_bars = generate_fallback_chart_bars(
        bars, density="high", special_notes=True
    )

    assert all(set(bar.notes) <= set("01234") for bar in chart_bars)
    assert all(not bar.balloon_counts for bar in chart_bars)


def test_special_notes_can_emit_roll_and_balloon_at_active_phrase_candidates():
    bars = [
        _feature(
            0,
            energy=0.8,
            onsets=[8, 10, 12, 14],
            phrase_position="phrase_end",
        ),
        _feature(1, energy=0.8, onsets=[0, 4, 8, 12]),
        _feature(
            2,
            energy=0.95,
            onsets=[8, 10, 12, 14],
            phrase_position="song_end",
            fill_candidate=True,
        ),
    ]

    chart_bars = generate_fallback_chart_bars(
        bars, density="high", special_notes=True, course="Oni", level=10
    )

    assert "5" in chart_bars[0].notes
    assert "8" in chart_bars[0].notes
    assert "7" in chart_bars[2].notes
    assert "8" in chart_bars[2].notes
    assert len(chart_bars[2].balloon_counts) == 1
    assert chart_bars[2].balloon_counts[0] > 0


@pytest.mark.parametrize(
    ("grids", "time_signature"),
    [(12, "3/4"), (16, "4/4")],
)
def test_special_note_span_fits_supported_meter(grids, time_signature):
    bar = _feature(
        0,
        energy=0.9,
        grids=grids,
        time_signature=time_signature,
        onsets=list(range(grids // 2, grids, 2)),
        phrase_position="phrase_end",
        fill_candidate=True,
    )

    chart_bar = generate_fallback_chart_bars(
        [bar], density="high", special_notes=True
    )[0]

    start = next(index for index, note in enumerate(chart_bar.notes) if note in "57")
    end = chart_bar.notes.index("8")
    assert 0 <= start < end < grids
    assert len(chart_bar.notes) == grids


def test_balloon_count_increases_with_course_and_level():
    bar = _feature(
        0,
        energy=0.95,
        end_time=2.0,
        onsets=[8, 10, 12, 14],
        phrase_position="song_end",
        fill_candidate=True,
    )

    counts = []
    for course, level in [("Easy", 3), ("Normal", 5), ("Hard", 7), ("Oni", 10)]:
        chart_bar = generate_fallback_chart_bars(
            [bar],
            density="high",
            special_notes=True,
            course=course,
            level=level,
        )[0]
        counts.append(chart_bar.balloon_counts[0])

    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_balloon_count_increases_with_bar_duration():
    def generate_count(duration: float) -> int:
        bar = _feature(
            0,
            energy=0.95,
            end_time=duration,
            onsets=[8, 10, 12, 14],
            phrase_position="song_end",
            fill_candidate=True,
        )
        return generate_fallback_chart_bars(
            [bar],
            density="high",
            special_notes=True,
            course="Hard",
            level=7,
        )[0].balloon_counts[0]

    assert generate_count(1.0) < generate_count(2.0)


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
    fill_candidate: bool = False,
    section: str = "unknown",
    rms_dbfs: float | None = None,
    peak_rms_dbfs: float | None = None,
    relative_rms_db: float | None = None,
    sustained_activity_ratio: float | None = None,
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
        rms_dbfs=rms_dbfs,
        peak_rms_dbfs=peak_rms_dbfs,
        relative_rms_db=relative_rms_db,
        sustained_activity_ratio=sustained_activity_ratio,
        time_signature=time_signature,
        grids_per_bar=grids,
        onset_16=onsets,
        accent_16=accents,
        grid_features=grid_features,
        beat_grids=beats,
        downbeat_grid=downbeat,
        phrase_position=phrase_position,
        fill_candidate=fill_candidate,
        section=section,
    )


def _hit_grids(notes: str) -> set[int]:
    return {index for index, note in enumerate(notes) if note != "0"}
