import pytest
from pydantic import ValidationError

from tja_ai_chartgen.features.salience import (
    RHYTHMIC_SALIENCE_FEATURE_VERSION,
    build_bar_hit_salience,
    build_canonical_rhythmic_evidence,
    build_hit_salience,
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


def test_hit_salience_prioritizes_offbeat_onset_over_beat_skeleton():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.6,
        grids_per_bar=48,
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
        grid_features=[
            GridFeature(grid=0, activity=0.8, beat=1, downbeat=True),
            GridFeature(grid=7, onset=True, strength=0.7, activity=0.6),
            GridFeature(grid=12, activity=0.8, beat=2),
        ],
    )

    salience = build_bar_hit_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[7].hit > points[0].hit > points[12].hit
    assert points[7].reasons == ["onset"]
    assert points[0].reasons == ["downbeat"]
    assert points[12].reasons == ["beat"]
    assert salience.onset_evidence_count == 1
    assert salience.active_ratio == pytest.approx(3 / 48, abs=1e-6)


def test_hit_salience_does_not_create_hits_from_activity_alone():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.7,
        grids_per_bar=48,
        activity_grids=[0.9] * 48,
    )

    salience = build_bar_hit_salience(bar)

    assert salience.points == []
    assert salience.active_ratio == 1.0
    assert salience.onset_evidence_count == 0


def test_hit_salience_keeps_strong_spectral_attack_when_mix_onset_is_missing():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        spectral_grid_features=[
            SpectralGridFeature(grid=10, high_onset_strength=0.8, spectral_flux=0.7)
        ],
    )

    salience = build_bar_hit_salience(bar)

    assert salience.points[0].grid == 10
    assert salience.points[0].hit == 0.6
    assert salience.points[0].reasons == ["spectral"]


def test_hit_salience_merges_adjacent_spectral_frames_but_preserves_onsets():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        grids_per_bar=48,
        grid_features=[
            GridFeature(grid=20, onset=True, strength=0.55),
            GridFeature(grid=21, onset=True, strength=0.5),
        ],
        spectral_grid_features=[
            SpectralGridFeature(grid=5, high_onset_strength=0.6),
            SpectralGridFeature(grid=6, high_onset_strength=0.9),
            SpectralGridFeature(grid=7, high_onset_strength=0.7),
            SpectralGridFeature(grid=19, spectral_flux=0.9),
            SpectralGridFeature(grid=20, low_onset_strength=0.8),
            SpectralGridFeature(grid=21, mid_onset_strength=0.7),
        ],
    )

    salience = build_bar_hit_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert 5 not in points
    assert 6 in points
    assert 7 not in points
    assert 19 not in points
    assert points[6].reasons == ["spectral"]
    assert points[20].reasons == ["onset", "spectral"]
    assert points[20].hit == 0.6
    assert points[21].reasons == ["onset", "spectral"]
    assert points[21].hit == 0.525


def test_hit_salience_uses_structure_as_modifier_not_independent_evidence():
    base = dict(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        grids_per_bar=48,
        grid_features=[GridFeature(grid=9, onset=True, strength=0.6)],
    )

    peak = build_bar_hit_salience(BarFeature(**base, transition_role="peak"))
    breakdown = build_bar_hit_salience(BarFeature(**base, transition_role="breakdown"))
    empty_peak = build_bar_hit_salience(
        BarFeature(
            index=1,
            start_time=2.0,
            end_time=4.0,
            energy=0.5,
            transition_role="peak",
        )
    )

    assert peak.points[0].hit > breakdown.points[0].hit
    assert peak.points[0].reasons == ["onset", "role:peak"]
    assert breakdown.points[0].reasons == ["onset", "role:breakdown"]
    assert empty_peak.points == []


def test_hit_salience_forces_digital_and_edge_silence_to_zero():
    edge_noise = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.02,
        rms_dbfs=-60.0,
        peak_rms_dbfs=-45.0,
        relative_rms_db=-45.0,
        sustained_activity_ratio=0.05,
        onset_grids=[0],
        grid_features=[GridFeature(grid=0, onset=True, strength=0.8)],
    )
    digital_silence = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
    )
    active = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.5,
        phrase_position="song_end",
        grid_features=[GridFeature(grid=6, onset=True, strength=0.7)],
    )

    results = build_hit_salience([edge_noise, digital_silence, active])

    assert results[0].points == []
    assert results[0].active_ratio == 0.0
    assert results[1].points == []
    assert results[2].points[0].grid == 6


def test_hit_salience_is_deterministic():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        beat_grids=[0, 12, 24, 36],
        grid_features=[GridFeature(grid=5, onset=True, strength=0.72, activity=0.4)],
        spectral_grid_features=[
            SpectralGridFeature(grid=5, low_onset_strength=0.8, spectral_flux=0.6)
        ],
    )

    first = build_bar_hit_salience(bar)
    second = build_bar_hit_salience(bar)

    assert first.model_dump() == second.model_dump()
