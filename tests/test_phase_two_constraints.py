from math import floor

import pytest

from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import BarFeature, GridFeature, ResolutionPlan
from tja_ai_chartgen.tja.quality import playable_hit_count

DENSITIES = ("low", "medium", "auto", "high", "max")
COURSES = (
    ("Easy", 3, 3.0, 0.38),
    ("Normal", 5, 5.0, 0.5),
    ("Hard", 7, 7.5, 0.7),
    ("Oni", 10, 10.0, 0.82),
)
METER_CASES = (
    ("4/4", 48, 16, (16, 24, 48), 2.0, (0, 12, 24, 36)),
    ("3/4", 36, 12, (12, 18, 36), 1.5, (0, 12, 24)),
    ("6/8", 36, 12, (12, 18, 36), 1.5, (0, 18)),
)


@pytest.mark.parametrize(
    ("time_signature", "canonical_grids", "legacy_resolution", "resolutions", "duration", "beats"),
    METER_CASES,
)
def test_phase_two_load_constraints_hold_across_course_density_and_resolution(
    time_signature: str,
    canonical_grids: int,
    legacy_resolution: int,
    resolutions: tuple[int, ...],
    duration: float,
    beats: tuple[int, ...],
):
    bar = _dense_bar(
        time_signature=time_signature,
        canonical_grids=canonical_grids,
        duration=duration,
        beats=beats,
    )
    counts: dict[tuple[int, str, str], int] = {}

    for resolution in resolutions:
        plan = _resolution_plan(canonical_grids, resolution, bar_count=1)
        for course, level, speed_cap, occupancy_cap in COURSES:
            for density in DENSITIES:
                chart_bar = generate_fallback_chart_bars(
                    [bar],
                    density=density,
                    course=course,
                    level=level,
                    resolution_plan=plan,
                )[0]
                count = playable_hit_count(chart_bar.notes)
                counts[(resolution, course, density)] = count

                assert len(chart_bar.notes) == resolution
                assert count / duration <= speed_cap
                assert count <= max(1, floor(duration * speed_cap))
                assert count <= floor(legacy_resolution * occupancy_cap)
                assert count / resolution <= occupancy_cap

    for resolution in resolutions:
        for course, *_rest in COURSES:
            density_counts = [
                counts[(resolution, course, density)] for density in DENSITIES
            ]
            assert density_counts == sorted(density_counts)
        for density in DENSITIES:
            course_counts = [
                counts[(resolution, course, density)]
                for course, *_rest in COURSES
            ]
            assert course_counts == sorted(course_counts)

    for course, *_rest in COURSES:
        for density in DENSITIES:
            resolution_counts = {
                counts[(resolution, course, density)] for resolution in resolutions
            }
            assert len(resolution_counts) == 1


@pytest.mark.parametrize(
    ("time_signature", "canonical_grids", "_legacy_resolution", "resolutions", "duration", "beats"),
    METER_CASES,
)
def test_phase_two_silence_rest_and_sparse_caps_hold_for_all_profiles(
    time_signature: str,
    canonical_grids: int,
    _legacy_resolution: int,
    resolutions: tuple[int, ...],
    duration: float,
    beats: tuple[int, ...],
):
    bars = _silence_rest_sparse_bars(
        time_signature=time_signature,
        canonical_grids=canonical_grids,
        duration=duration,
        beats=beats,
    )

    for resolution in resolutions:
        plan = _resolution_plan(canonical_grids, resolution, bar_count=len(bars))
        for course, level, *_rest in COURSES:
            for density in DENSITIES:
                chart_bars = generate_fallback_chart_bars(
                    bars,
                    density=density,
                    course=course,
                    level=level,
                    resolution_plan=plan,
                )

                assert chart_bars[0].notes == "0" * resolution
                assert playable_hit_count(chart_bars[1].notes) > 0
                assert chart_bars[2].notes == "0" * resolution
                assert playable_hit_count(chart_bars[3].notes) <= 4
                assert chart_bars[4].notes == "0" * resolution


def _dense_bar(
    *,
    time_signature: str,
    canonical_grids: int,
    duration: float,
    beats: tuple[int, ...],
) -> BarFeature:
    return _bar(
        0,
        time_signature=time_signature,
        canonical_grids=canonical_grids,
        duration=duration,
        energy=0.95,
        onset_grids=list(range(canonical_grids)),
        activity=0.9,
        beats=beats,
    )


def _silence_rest_sparse_bars(
    *,
    time_signature: str,
    canonical_grids: int,
    duration: float,
    beats: tuple[int, ...],
) -> list[BarFeature]:
    return [
        _bar(
            0,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.068,
            onset_grids=[0, canonical_grids - 1],
            rms_dbfs=-59.7,
            peak_rms_dbfs=-46.7,
            relative_rms_db=-53.5,
            sustained_activity_ratio=0.098,
            section="intro",
        ),
        _bar(
            1,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.9,
            onset_grids=list(range(0, canonical_grids, 3)),
            activity=0.8,
            beats=beats,
        ),
        _bar(
            2,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.01,
            section="break",
        ),
        _bar(
            3,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.02,
            onset_grids=[0, canonical_grids // 2],
            beats=beats,
        ),
        _bar(
            4,
            time_signature=time_signature,
            canonical_grids=canonical_grids,
            duration=duration,
            energy=0.0,
            section="outro",
            phrase_position="song_end",
        ),
    ]


def _bar(
    index: int,
    *,
    time_signature: str,
    canonical_grids: int,
    duration: float,
    energy: float,
    onset_grids: list[int] | None = None,
    activity: float = 0.0,
    beats: tuple[int, ...] = (),
    section: str = "unknown",
    phrase_position: str = "unknown",
    rms_dbfs: float | None = None,
    peak_rms_dbfs: float | None = None,
    relative_rms_db: float | None = None,
    sustained_activity_ratio: float | None = None,
) -> BarFeature:
    onsets = onset_grids or []
    beat_set = set(beats)
    start_time = index * duration
    return BarFeature(
        index=index,
        start_time=start_time,
        end_time=start_time + duration,
        energy=energy,
        rms_dbfs=rms_dbfs,
        peak_rms_dbfs=peak_rms_dbfs,
        relative_rms_db=relative_rms_db,
        sustained_activity_ratio=sustained_activity_ratio,
        time_signature=time_signature,
        grids_per_bar=canonical_grids,
        onset_grids=onsets,
        activity_grids=[activity] * canonical_grids,
        grid_features=[
            GridFeature(
                grid=grid,
                onset=grid in onsets,
                beat=beats.index(grid) + 1 if grid in beat_set else None,
                downbeat=grid == 0 and grid in beat_set,
                strength=1.0 if grid in onsets else 0.0,
                activity=activity,
            )
            for grid in range(canonical_grids)
        ],
        beat_grids=list(beats),
        downbeat_grid=0 if beats else None,
        section=section,
        phrase_position=phrase_position,
    )


def _resolution_plan(
    canonical_grids: int,
    resolution: int,
    *,
    bar_count: int,
) -> ResolutionPlan:
    return ResolutionPlan(
        canonical_grids_per_bar=canonical_grids,
        base_resolution=resolution,
        bar_resolutions=[resolution] * bar_count,
    )
