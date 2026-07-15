import pytest
from pydantic import ValidationError

from tja_ai_chartgen.features.salience import RHYTHMIC_SALIENCE_FEATURE_VERSION
from tja_ai_chartgen.tja.model import BarRhythmicSalience, RhythmicSaliencePoint


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
