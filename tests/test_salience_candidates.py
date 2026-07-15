import pytest

from tja_ai_chartgen.features.salience_candidates import (
    build_salience_candidate_bars,
    is_salience_grid_representable,
    rank_bar_salience_candidates,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    GridFeature,
    ResolutionPlan,
    RhythmicSaliencePoint,
)


def _bar(*, grids_per_bar: int = 48, time_signature: str = "4/4") -> BarFeature:
    return BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.7,
        time_signature=time_signature,
        grids_per_bar=grids_per_bar,
    )


def test_salience_candidates_follow_evidence_priority_and_score_order():
    bar = _bar()
    salience = BarRhythmicSalience(
        confidence=0.8,
        points=[
            RhythmicSaliencePoint(
                grid=15,
                hit=0.5,
                confidence=0.3,
                reasons=["onset"],
            ),
            RhythmicSaliencePoint(
                grid=12,
                hit=0.4,
                accent=0.7,
                confidence=0.8,
                reasons=["accent:phrase-start"],
            ),
            RhythmicSaliencePoint(
                grid=9,
                hit=0.5,
                confidence=0.7,
                reasons=["spectral"],
            ),
            RhythmicSaliencePoint(
                grid=0,
                hit=0.35,
                sustained_activity=0.6,
                confidence=0.8,
                reasons=["downbeat"],
            ),
            RhythmicSaliencePoint(
                grid=6,
                hit=0.9,
                confidence=0.85,
                reasons=["onset"],
            ),
            RhythmicSaliencePoint(
                grid=3,
                hit=0.75,
                confidence=0.7,
                reasons=["onset"],
            ),
        ],
    )

    candidates = rank_bar_salience_candidates(
        bar,
        salience,
        output_resolution=16,
    )

    assert [candidate.grid for candidate in candidates] == [6, 3, 9, 0, 12, 15]
    assert [candidate.kind for candidate in candidates] == [
        "strong-transient",
        "strong-transient",
        "transient",
        "rhythmic-skeleton",
        "structure-highlight",
        "weak-evidence",
    ]
    assert [candidate.reliable for candidate in candidates] == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert candidates[0].score > candidates[1].score


def test_salience_candidates_filter_unrepresentable_and_out_of_range_points():
    bar = _bar()
    salience = BarRhythmicSalience(
        confidence=0.9,
        points=[
            RhythmicSaliencePoint(grid=6, hit=0.8, confidence=0.8, reasons=["onset"]),
            RhythmicSaliencePoint(grid=7, hit=1.0, confidence=1.0, reasons=["onset"]),
            RhythmicSaliencePoint(grid=47, hit=0.9, confidence=0.9, reasons=["onset"]),
            RhythmicSaliencePoint(grid=48, hit=1.0, confidence=1.0, reasons=["onset"]),
        ],
    )

    candidates = rank_bar_salience_candidates(
        bar,
        salience,
        output_resolution=16,
    )

    assert [candidate.grid for candidate in candidates] == [6]


def test_salience_candidates_deduplicate_same_grid_deterministically():
    bar = _bar()
    salience = BarRhythmicSalience(
        confidence=0.9,
        points=[
            RhythmicSaliencePoint(grid=6, hit=0.5, confidence=0.8, reasons=["onset"]),
            RhythmicSaliencePoint(grid=6, hit=0.9, confidence=0.9, reasons=["onset"]),
        ],
    )

    candidates = rank_bar_salience_candidates(
        bar,
        salience,
        output_resolution=16,
    )

    assert len(candidates) == 1
    assert candidates[0].point.hit == 0.9


@pytest.mark.parametrize(
    ("grids_per_bar", "time_signature", "resolution", "representable", "rejected"),
    [
        (48, "4/4", 16, 6, 7),
        (48, "4/4", 24, 2, 3),
        (36, "3/4", 12, 3, 4),
        (36, "6/8", 18, 2, 3),
        (36, "6/8", 36, 35, 36),
    ],
)
def test_salience_grid_representability_matches_meter_resolution_families(
    grids_per_bar: int,
    time_signature: str,
    resolution: int,
    representable: int,
    rejected: int,
):
    bar = _bar(grids_per_bar=grids_per_bar, time_signature=time_signature)

    assert is_salience_grid_representable(
        bar,
        representable,
        output_resolution=resolution,
    )
    assert not is_salience_grid_representable(
        bar,
        rejected,
        output_resolution=resolution,
    )


def test_salience_candidate_filter_rejects_incompatible_resolution():
    bar = _bar()

    with pytest.raises(ValueError, match="incompatible with resolution 20"):
        rank_bar_salience_candidates(
            bar,
            BarRhythmicSalience(),
            output_resolution=20,
        )


def test_build_salience_candidate_bars_uses_each_planned_resolution():
    bars = [
        _bar(),
        _bar().model_copy(
            update={
                "index": 1,
                "start_time": 2.0,
                "end_time": 4.0,
                "grid_features": [
                    GridFeature(grid=2, onset=True, strength=0.9, activity=0.8),
                    GridFeature(grid=3, onset=True, strength=0.9, activity=0.8),
                ],
            }
        ),
    ]
    bars[0] = bars[0].model_copy(
        update={
            "grid_features": [
                GridFeature(grid=2, onset=True, strength=0.9, activity=0.8),
                GridFeature(grid=3, onset=True, strength=0.9, activity=0.8),
            ]
        }
    )
    plan = ResolutionPlan(
        canonical_grids_per_bar=48,
        base_resolution=16,
        bar_resolutions=[16, 24],
    )

    candidate_bars = build_salience_candidate_bars(bars, resolution_plan=plan)

    assert all(candidate.grid != 2 for candidate in candidate_bars[0])
    assert any(candidate.grid == 3 for candidate in candidate_bars[0])
    assert any(candidate.grid == 2 for candidate in candidate_bars[1])
