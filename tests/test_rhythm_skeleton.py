from tja_ai_chartgen.features.rhythm_grid import is_stable_rhythmic_grid
from tja_ai_chartgen.rules.rhythm_skeleton import (
    AI_SKELETON_MAX_EXTRA_RATE,
    AI_SKELETON_MIN_COVERAGE,
    RHYTHM_SKELETON_VERSION,
    build_rhythm_skeleton,
    compare_chart_to_rhythm_skeleton,
    conform_chart_to_rhythm_skeleton,
    rhythm_skeleton_issues,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    ChartBar,
    ResolutionPlan,
    SongAnalysis,
)


def _analysis(bar_count: int = 16) -> SongAnalysis:
    return SongAnalysis(
        title="Skeleton",
        audio_file="song.wav",
        ogg_file="song.ogg",
        bpm=120.0,
        offset=0.0,
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16] * bar_count,
        ),
        bars=[
            BarFeature(
                index=index,
                start_time=index * 2.0,
                end_time=(index + 1) * 2.0,
                energy=0.7,
                grids_per_bar=48,
                onset_grids=[0, 12, 24, 36],
                beat_grids=[0, 12, 24, 36],
                downbeat_grid=0,
                phrase_position=(
                    "phrase_start" if index % 4 == 0 else "phrase_middle"
                ),
                section="verse",
            )
            for index in range(bar_count)
        ],
    )


def _chart_from_skeleton(skeleton: list[list[int]]) -> list[ChartBar]:
    bars: list[ChartBar] = []
    for index, ticks in enumerate(skeleton):
        notes = ["0"] * 16
        for sequence, tick in enumerate(ticks):
            notes[tick // 3] = "1" if sequence % 2 == 0 else "2"
        bars.append(ChartBar(index=index, notes="".join(notes)))
    return bars


def test_build_rhythm_skeleton_is_deterministic_and_uses_canonical_ticks():
    analysis = _analysis()

    first = build_rhythm_skeleton(
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )
    second = build_rhythm_skeleton(
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )

    assert RHYTHM_SKELETON_VERSION == "rhythm-skeleton-v2"
    assert first == second
    assert len(first) == len(analysis.bars)
    assert all(tick % 3 == 0 for bar in first for tick in bar)


def test_build_rhythm_skeleton_stabilizes_high_resolution_microtiming():
    analysis = _analysis().model_copy(
        update={
            "resolution_plan": ResolutionPlan(
                canonical_grids_per_bar=48,
                base_resolution=48,
                bar_resolutions=[48] * 16,
            ),
            "bars": [
                bar.model_copy(
                    update={
                        "onset_grids": [1, 13, 25, 37],
                        "beat_grids": [0, 12, 24, 36],
                    }
                )
                for bar in _analysis().bars
            ],
        }
    )

    skeleton = build_rhythm_skeleton(
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )

    assert all(
        is_stable_rhythmic_grid(tick, analysis.bars[position].grids_per_bar)
        for position, ticks in enumerate(skeleton)
        for tick in ticks
    )


def test_conform_chart_to_rhythm_skeleton_handles_short_high_resolution_regeneration():
    analysis = _analysis(bar_count=1).model_copy(
        update={
            "resolution_plan": ResolutionPlan(
                canonical_grids_per_bar=48,
                base_resolution=48,
                bar_resolutions=[48],
            ),
        }
    )
    notes = ["0"] * 48
    for tick, note in ((1, "1"), (13, "2"), (25, "1"), (37, "2")):
        notes[tick] = note

    conformed = conform_chart_to_rhythm_skeleton(
        [ChartBar(index=0, notes="".join(notes))],
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )

    assert all(
        is_stable_rhythmic_grid(position, 48)
        for position, note in enumerate(conformed[0].notes)
        if note in "1234"
    )


def test_rhythm_skeleton_comparison_accepts_color_changes_without_moving_hits():
    analysis = _analysis()
    skeleton = build_rhythm_skeleton(
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )
    chart = _chart_from_skeleton(skeleton)

    comparison = compare_chart_to_rhythm_skeleton(
        chart,
        analysis.bars,
        skeleton,
    )

    assert comparison.coverage == 1.0
    assert comparison.extra_rate == 0.0
    assert rhythm_skeleton_issues(
        chart,
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    ) == []


def test_rhythm_skeleton_issues_reject_widespread_timing_rewrite():
    analysis = _analysis()
    skeleton = build_rhythm_skeleton(
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )
    chart = _chart_from_skeleton(skeleton)
    rewritten: list[ChartBar] = []
    for bar in chart:
        notes = ["0"] * 16
        for position, note in enumerate(bar.notes):
            if note in "12":
                notes[(position + 1) % 16] = note
        rewritten.append(bar.model_copy(update={"notes": "".join(notes)}))

    issues = rhythm_skeleton_issues(
        rewritten,
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )

    assert len(issues) == 2
    assert f"minimum {AI_SKELETON_MIN_COVERAGE:.0%}" in issues[0]
    assert f"maximum {AI_SKELETON_MAX_EXTRA_RATE:.0%}" in issues[1]


def test_conform_chart_to_rhythm_skeleton_repairs_timing_without_rejecting_colors():
    analysis = _analysis()
    skeleton = build_rhythm_skeleton(
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )
    chart = _chart_from_skeleton(skeleton)
    shifted = [
        bar.model_copy(
            update={
                "notes": bar.notes[-1] + bar.notes[:-1],
            }
        )
        for bar in chart
    ]

    conformed = conform_chart_to_rhythm_skeleton(
        shifted,
        analysis,
        course="Oni",
        level=9,
        style="technical",
        density="auto",
    )
    comparison = compare_chart_to_rhythm_skeleton(
        conformed,
        analysis.bars,
        skeleton,
    )

    assert comparison.coverage == 1.0
    assert comparison.extra_rate == 0.0
    assert any("2" in bar.notes for bar in conformed)
