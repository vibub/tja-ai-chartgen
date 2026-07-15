import pytest
from pydantic import ValidationError

from tja_ai_chartgen.features.salience import (
    RHYTHMIC_SALIENCE_FEATURE_VERSION,
    build_accent_salience,
    build_bar_accent_salience,
    build_bar_hit_salience,
    build_canonical_rhythmic_evidence,
    build_don_ka_salience,
    build_bar_don_ka_salience,
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
        accent_grids=[12],
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
    assert evidence[12].accent_hint is True
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


def test_canonical_rhythmic_evidence_does_not_treat_activity_strength_as_onset():
    bar = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.4,
        grid_features=[GridFeature(grid=8, strength=0.7, activity=0.7)],
    )

    evidence = build_canonical_rhythmic_evidence(bar)

    assert evidence[8].onset is False
    assert evidence[8].onset_strength == 0.0
    assert evidence[8].activity == 0.7


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


def test_accent_salience_uses_downbeat_onset_peaks_and_legacy_hints():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.6,
        grid_features=[
            GridFeature(grid=0, onset=True, strength=0.6, beat=1, downbeat=True),
            GridFeature(grid=6, onset=True, strength=0.9),
            GridFeature(grid=7, onset=True, strength=0.7, accent=True),
        ],
    )

    salience = build_bar_accent_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[0].accent == 0.82
    assert "accent:downbeat" in points[0].reasons
    assert "accent:onset-peak" in points[0].reasons
    assert "accent:downbeat-onset" in points[0].reasons
    assert points[6].accent == 0.855
    assert points[6].reasons[-1] == "accent:onset-peak"
    assert points[7].accent == 0.5
    assert "accent:onset-peak" not in points[7].reasons
    assert points[7].reasons[-1] == "accent:hint"


def test_accent_salience_rejects_weak_local_onset_peaks():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        grid_features=[
            GridFeature(grid=0, beat=1, downbeat=True, activity=0.5),
            GridFeature(grid=6, onset=True, strength=0.4, activity=0.5),
        ],
    )

    salience = build_bar_accent_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[0].accent == 0.58
    assert "accent:downbeat" in points[0].reasons
    assert points[6].accent == 0.0
    assert "accent:onset-peak" not in points[6].reasons


def test_accent_salience_uses_low_attack_without_treating_all_spectral_hits_as_accents():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        grids_per_bar=48,
        spectral_grid_features=[
            SpectralGridFeature(grid=10, low_onset_strength=0.8),
            SpectralGridFeature(grid=24, high_onset_strength=0.8),
        ],
    )

    salience = build_bar_accent_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[10].accent == 0.67
    assert points[10].reasons == ["spectral", "accent:low-attack"]
    assert points[24].accent == 0.0
    assert points[24].reasons == ["spectral"]


def test_accent_salience_applies_section_start_energy_rise_and_cadence_to_first_hit():
    previous = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        phrase_id=0,
        section_id="section-a",
        grid_features=[GridFeature(grid=3, onset=True, strength=0.7)],
    )
    current = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.5,
        energy_delta=0.6,
        boundary_confidence=0.8,
        phrase_id=1,
        section_id="section-b",
        transition_role="cadence",
        spectral_grid_features=[
            SpectralGridFeature(grid=5, high_onset_strength=0.8),
            SpectralGridFeature(grid=15, high_onset_strength=0.8),
        ],
    )

    salience = build_accent_salience([previous, current])[1]
    points = {point.grid: point for point in salience.points}

    assert points[5].accent == 0.784
    assert "accent:section-start" in points[5].reasons
    assert "accent:energy-rise" in points[5].reasons
    assert "accent:role:cadence" in points[5].reasons
    assert points[15].accent == 0.0


def test_accent_salience_does_not_create_hits_from_structure_or_activity():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.8,
        energy_delta=0.8,
        boundary_confidence=1.0,
        phrase_position="phrase_start",
        transition_role="peak",
        activity_grids=[0.9] * 48,
    )

    salience = build_bar_accent_salience(bar)

    assert salience.points == []
    assert salience.active_ratio == 1.0


def test_don_ka_salience_uses_dominant_frequency_and_keeps_mixed_evidence_neutral():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        grids_per_bar=48,
        spectral_grid_features=[
            SpectralGridFeature(grid=4, low_onset_strength=0.9),
            SpectralGridFeature(grid=16, high_onset_strength=0.8),
            SpectralGridFeature(
                grid=28,
                low_onset_strength=0.8,
                mid_onset_strength=0.9,
                high_onset_strength=0.1,
            ),
        ],
    )

    salience = build_bar_don_ka_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[4].don_preference == 0.585
    assert points[4].ka_preference == 0.0
    assert points[4].reasons[-1] == "color:low"
    assert points[16].don_preference == 0.0
    assert points[16].ka_preference == 0.52
    assert points[16].reasons[-1] == "color:high"
    assert points[28].don_preference == 0.0
    assert points[28].ka_preference == 0.0
    assert not any(reason.startswith("color:") for reason in points[28].reasons)


def test_don_ka_salience_uses_downbeat_and_offbeat_as_weak_position_cues():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        grids_per_bar=48,
        beat_grids=[0, 12, 24, 36],
        grid_features=[
            GridFeature(grid=0, onset=True, strength=0.6, beat=1, downbeat=True),
            GridFeature(grid=6, onset=True, strength=0.6),
        ],
    )

    salience = build_bar_don_ka_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[0].don_preference == 0.28
    assert points[0].ka_preference == 0.0
    assert points[0].reasons[-1] == "color:downbeat"
    assert points[6].don_preference == 0.0
    assert points[6].ka_preference == 0.24
    assert points[6].reasons[-1] == "color:offbeat"


def test_don_ka_salience_supports_compound_meter_offbeats():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=3.5,
        energy=0.5,
        time_signature="6/8",
        grids_per_bar=36,
        beat_grids=[0, 18],
        grid_features=[GridFeature(grid=9, onset=True, strength=0.7)],
    )

    salience = build_bar_don_ka_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[9].ka_preference == 0.24
    assert points[9].reasons[-1] == "color:offbeat"


def test_don_ka_salience_uses_brightness_only_with_dominant_high_attack():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        grids_per_bar=48,
        brightness=0.9,
        percussive_ratio=0.8,
        spectral_grid_features=[
            SpectralGridFeature(grid=4, high_onset_strength=0.4),
            SpectralGridFeature(
                grid=20,
                low_onset_strength=0.4,
                mid_onset_strength=0.5,
                high_onset_strength=0.4,
            ),
        ],
    )

    salience = build_bar_don_ka_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[4].ka_preference == 0.33
    assert points[4].reasons[-2:] == ["color:high", "color:bright-percussive"]
    assert points[20].ka_preference == 0.0
    assert "color:bright-percussive" not in points[20].reasons


def test_don_ka_salience_breaks_long_strong_monochrome_runs():
    bars = [
        BarFeature(
            index=1,
            start_time=2.0,
            end_time=4.0,
            energy=0.0,
            grids_per_bar=48,
            spectral_grid_features=[
                SpectralGridFeature(grid=grid, low_onset_strength=0.9)
                for grid in (2, 10, 20, 30, 40)
            ],
        )
    ]

    salience = build_don_ka_salience(bars)[0]

    assert [point.don_preference for point in salience.points] == [
        0.585,
        0.585,
        0.585,
        0.32,
        0.585,
    ]
    assert "color:balance" in salience.points[3].reasons
    assert all(point.don_preference <= 0.75 for point in salience.points)


def test_salience_absolute_gates_reject_weak_noise():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.07,
        grids_per_bar=48,
        beat_grids=[0, 12, 24, 36],
        grid_features=[
            GridFeature(grid=0, activity=0.07, beat=1, downbeat=True),
            GridFeature(grid=7, onset=True, strength=0.07),
        ],
        spectral_grid_features=[
            SpectralGridFeature(grid=19, high_onset_strength=0.11)
        ],
    )

    salience = build_bar_hit_salience(bar)

    assert salience.points == []
    assert salience.confidence == 0.0
    assert salience.fallback_reason == "no-rhythmic-evidence"


def test_salience_confidence_rewards_consistent_transient_evidence():
    onset_only = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        grids_per_bar=48,
        grid_features=[GridFeature(grid=8, onset=True, strength=0.7)],
    )
    corroborated = onset_only.model_copy(
        update={
            "grid_features": [
                GridFeature(grid=8, onset=True, strength=0.7, activity=0.6)
            ],
            "spectral_grid_features": [
                SpectralGridFeature(
                    grid=8,
                    low_onset_strength=0.8,
                    spectral_flux=0.7,
                )
            ],
        }
    )

    onset_salience = build_bar_hit_salience(onset_only)
    corroborated_salience = build_bar_hit_salience(corroborated)

    assert corroborated_salience.points[0].confidence > onset_salience.points[0].confidence
    assert corroborated_salience.confidence > onset_salience.confidence
    assert corroborated_salience.fallback_reason is None


def test_salience_song_percentile_rewards_stronger_global_peaks():
    weak = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.4,
        grids_per_bar=48,
        grid_features=[GridFeature(grid=8, onset=True, strength=0.4)],
    )
    strong = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.4,
        grids_per_bar=48,
        grid_features=[GridFeature(grid=8, onset=True, strength=0.9)],
    )

    independent_weak = build_bar_hit_salience(weak).points[0].confidence
    independent_strong = build_bar_hit_salience(strong).points[0].confidence
    combined = build_hit_salience([weak, strong])
    weak_boost = combined[0].points[0].confidence - independent_weak
    strong_boost = combined[1].points[0].confidence - independent_strong

    assert weak_boost > 0.0
    assert strong_boost > weak_boost


def test_salience_marks_barely_gated_transient_as_low_confidence():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        grids_per_bar=48,
        spectral_grid_features=[
            SpectralGridFeature(grid=10, high_onset_strength=0.12)
        ],
    )

    salience = build_bar_hit_salience(bar)

    assert salience.points[0].grid == 10
    assert salience.points[0].confidence > 0.0
    assert salience.confidence < 0.45
    assert salience.fallback_reason == "low-confidence"


def test_salience_marks_beat_only_output_as_stable_fallback():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.7,
        grids_per_bar=48,
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
    )

    salience = build_bar_hit_salience(bar)

    assert [point.grid for point in salience.points] == [0, 12, 24, 36]
    assert all(point.confidence > 0.0 for point in salience.points)
    assert salience.fallback_reason == "beat-skeleton-only"


def test_salience_preserves_stable_silence_reasons_through_full_pipeline():
    edge = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.0,
    )
    active = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        grid_features=[GridFeature(grid=8, onset=True, strength=0.8)],
    )
    digital_silence = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.0,
        phrase_position="song_end",
    )

    salience = build_don_ka_salience([edge, active, digital_silence])

    assert build_bar_hit_salience(edge).fallback_reason == "silent-bar"
    assert salience[0].fallback_reason == "edge-silence"
    assert salience[0].confidence == 0.0
    assert salience[1].fallback_reason is None
    assert salience[2].fallback_reason == "edge-silence"
