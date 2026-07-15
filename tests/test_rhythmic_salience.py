import pytest
from pydantic import ValidationError

from tja_ai_chartgen.features.salience import (
    RHYTHMIC_SALIENCE_FEATURE_VERSION,
    build_canonical_rhythmic_evidence,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    GridFeature,
    RhythmicSaliencePoint,
    SpectralGridFeature,
)


def test_rhythmic_salience_version_is_stable():
    assert RHYTHMIC_SALIENCE_FEATURE_VERSION == "rhythmic-salience-v1"


def test_rhythmic_salience_models_serialize_compact_sparse_points():
    salience = BarRhythmicSalience(
        points=[
            RhythmicSaliencePoint(
                grid=12,
                hit=0.8,
                accent=0.6,
                don_preference=0.7,
                sustained_activity=0.5,
                confidence=0.9,
                reasons=["mix-onset", "downbeat"],
            )
        ],
        active_ratio=0.25,
        onset_evidence_count=2,
        confidence=0.85,
    )

    assert salience.model_dump(exclude_defaults=True) == {
        "points": [
            {
                "grid": 12,
                "hit": 0.8,
                "accent": 0.6,
                "don_preference": 0.7,
                "sustained_activity": 0.5,
                "confidence": 0.9,
                "reasons": ["mix-onset", "downbeat"],
            }
        ],
        "active_ratio": 0.25,
        "onset_evidence_count": 2,
        "confidence": 0.85,
    }


def test_rhythmic_salience_defaults_are_independent():
    first = BarRhythmicSalience()
    second = BarRhythmicSalience()

    first.points.append(RhythmicSaliencePoint(grid=0))
    first.points[0].reasons.append("beat")

    assert second.points == []


def test_rhythmic_salience_rejects_invalid_ranges():
    with pytest.raises(ValidationError):
        RhythmicSaliencePoint(grid=-1)
    with pytest.raises(ValidationError):
        RhythmicSaliencePoint(grid=0, hit=1.01)
    with pytest.raises(ValidationError):
        BarRhythmicSalience(active_ratio=-0.01)
    with pytest.raises(ValidationError):
        BarRhythmicSalience(onset_evidence_count=-1)


def test_canonical_rhythmic_evidence_aligns_all_base_channels():
    bar = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.5,
        grids_per_bar=48,
        onset_grids=[12, 47],
        activity_grids=[0.0] * 12 + [0.4] + [0.0] * 35,
        beat_grids=[36, 0, 24, 12],
        downbeat_grid=0,
        grid_features=[
            GridFeature(grid=12, onset=True, strength=0.65, activity=0.7, beat=2),
            GridFeature(grid=24, beat=3),
        ],
        spectral_grid_features=[
            SpectralGridFeature(
                grid=12,
                low_onset_strength=0.6,
                high_onset_strength=0.2,
                spectral_flux=0.5,
            ),
            SpectralGridFeature(
                grid=12,
                low_onset_strength=0.8,
                mid_onset_strength=0.3,
                spectral_flux=0.4,
            ),
        ],
    )

    evidence = build_canonical_rhythmic_evidence(bar)

    assert len(evidence) == 48
    assert [point.grid for point in evidence] == list(range(48))
    assert evidence[0].beat == 1
    assert evidence[0].downbeat is True
    assert evidence[12].onset is True
    assert evidence[12].onset_strength == 0.65
    assert evidence[12].activity == 0.7
    assert evidence[12].beat == 2
    assert evidence[12].low_onset_strength == 0.8
    assert evidence[12].mid_onset_strength == 0.3
    assert evidence[12].high_onset_strength == 0.2
    assert evidence[12].spectral_flux == 0.5
    assert evidence[47].onset is True
    assert evidence[47].onset_strength == 1.0


def test_canonical_rhythmic_evidence_supports_compound_meter_and_ignores_invalid_grids():
    bar = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.3,
        time_signature="6/8",
        grids_per_bar=36,
        onset_grids=[-1, 18, 36],
        activity_grids=[1.5] + [0.0] * 35,
        beat_grids=[18, 0, 18, 50],
        downbeat_grid=50,
        grid_features=[
            GridFeature(grid=18, onset=True, strength=0.75, beat=2, downbeat=True),
            GridFeature(grid=40, onset=True, strength=1.0),
        ],
        spectral_grid_features=[
            SpectralGridFeature(grid=18, high_onset_strength=0.9),
            SpectralGridFeature(grid=40, low_onset_strength=1.0),
        ],
    )

    evidence = build_canonical_rhythmic_evidence(bar)

    assert len(evidence) == 36
    assert evidence[0].activity == 1.0
    assert evidence[0].beat == 1
    assert evidence[0].downbeat is False
    assert evidence[18].onset is True
    assert evidence[18].onset_strength == 0.75
    assert evidence[18].beat == 2
    assert evidence[18].downbeat is True
    assert evidence[18].high_onset_strength == 0.9
    assert sum(point.onset for point in evidence) == 1


def test_canonical_rhythmic_evidence_requires_positive_grid_size():
    bar = BarFeature(
        index=3,
        start_time=0.0,
        end_time=1.0,
        energy=0.0,
        grids_per_bar=0,
    )

    with pytest.raises(ValueError, match="positive canonical grid size"):
        build_canonical_rhythmic_evidence(bar)
