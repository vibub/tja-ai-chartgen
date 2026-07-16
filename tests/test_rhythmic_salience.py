import pytest
from pydantic import ValidationError

from tja_ai_chartgen.features.salience import (
    RHYTHMIC_SALIENCE_FEATURE_VERSION,
    build_accent_salience,
    build_bar_accent_salience,
    build_bar_burst_salience,
    build_bar_don_ka_salience,
    build_bar_hit_salience,
    build_burst_salience,
    build_canonical_rhythmic_evidence,
    build_don_ka_salience,
    build_hit_salience,
    is_reliable_burst,
    project_reliable_burst_span,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    GridFeature,
    InstrumentBarFeature,
    InstrumentGridFeature,
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


def test_rhythmic_salience_serializes_bar_burst_contract():
    salience = BarRhythmicSalience(
        burst_score=0.72,
        burst_confidence=0.81,
        burst_start_grid=24,
        burst_end_grid=42,
        burst_reasons=["burst:onset-density", "burst:spectral-flux"],
    )

    assert salience.model_dump(exclude_defaults=True) == {
        "burst_score": 0.72,
        "burst_confidence": 0.81,
        "burst_start_grid": 24,
        "burst_end_grid": 42,
        "burst_reasons": ["burst:onset-density", "burst:spectral-flux"],
    }


def test_rhythmic_salience_defaults_are_independent():
    first = BarRhythmicSalience()
    second = BarRhythmicSalience()

    first.points.append(RhythmicSaliencePoint(grid=0))
    first.points[0].reasons.append("beat")
    first.burst_reasons.append("burst:onset-density")

    assert second.points == []
    assert second.burst_reasons == []


def test_rhythmic_salience_rejects_invalid_ranges():
    with pytest.raises(ValidationError):
        RhythmicSaliencePoint(grid=-1)
    with pytest.raises(ValidationError):
        RhythmicSaliencePoint(grid=0, hit=1.01)
    with pytest.raises(ValidationError):
        BarRhythmicSalience(active_ratio=-0.01)
    with pytest.raises(ValidationError):
        BarRhythmicSalience(onset_evidence_count=-1)
    with pytest.raises(ValidationError):
        BarRhythmicSalience(burst_score=1.01)
    with pytest.raises(ValidationError):
        BarRhythmicSalience(burst_start_grid=-1)


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


def test_canonical_rhythmic_evidence_aligns_stem_onsets_and_bar_activity():
    bar = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.5,
        grids_per_bar=48,
        instrument=InstrumentBarFeature(
            vocal_activity=0.4,
            drum_activity=0.8,
            bass_activity=0.6,
            other_activity=0.5,
        ),
        instrument_grid_features=[
            InstrumentGridFeature(
                grid=12,
                vocal_onset=0.5,
                drum_onset=0.9,
                bass_onset=0.7,
                accompaniment_onset=0.6,
            ),
            InstrumentGridFeature(grid=12, drum_onset=0.7),
            InstrumentGridFeature(grid=60, drum_onset=1.0),
        ],
    )

    evidence = build_canonical_rhythmic_evidence(bar)

    assert evidence[12].vocal_onset == 0.5
    assert evidence[12].drum_onset == 0.9
    assert evidence[12].bass_onset == 0.7
    assert evidence[12].accompaniment_onset == 0.6
    assert evidence[12].vocal_activity == 0.4
    assert evidence[12].drum_activity == 0.8
    assert evidence[12].bass_activity == 0.6
    assert evidence[12].accompaniment_activity == 0.5
    assert sum(point.drum_onset > 0 for point in evidence) == 1


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


def test_hit_salience_uses_activity_gated_drum_onset_and_rewards_agreement():
    stem_only_bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        instrument=InstrumentBarFeature(drum_activity=0.8),
        instrument_grid_features=[InstrumentGridFeature(grid=9, drum_onset=0.9)],
    )
    agreed_bar = stem_only_bar.model_copy(
        update={
            "grid_features": [GridFeature(grid=9, onset=True, strength=0.6)],
        }
    )

    stem_only = build_bar_hit_salience(stem_only_bar)
    agreed = build_bar_hit_salience(agreed_bar)

    assert stem_only.points[0].grid == 9
    assert stem_only.points[0].hit == 0.45
    assert stem_only.points[0].reasons == ["stem:drum-onset"]
    assert agreed.points[0].hit > stem_only.points[0].hit
    assert agreed.points[0].reasons[:3] == [
        "onset",
        "stem:drum-onset",
        "stem:drum-agreement",
    ]
    assert agreed.points[0].confidence > stem_only.points[0].confidence
    assert agreed.onset_evidence_count == 1


def test_hit_salience_rejects_ungated_stem_artifacts_and_non_drum_onsets():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        instrument=InstrumentBarFeature(
            vocal_activity=0.8,
            drum_activity=0.02,
            bass_activity=0.8,
            other_activity=0.8,
        ),
        instrument_grid_features=[
            InstrumentGridFeature(grid=4, drum_onset=1.0),
            InstrumentGridFeature(grid=8, vocal_onset=1.0),
            InstrumentGridFeature(grid=12, bass_onset=1.0),
            InstrumentGridFeature(grid=16, accompaniment_onset=1.0),
        ],
    )

    salience = build_bar_hit_salience(bar)

    assert salience.points == []
    assert salience.onset_evidence_count == 0


def test_hit_salience_uses_bass_onset_only_to_reinforce_beat_skeleton():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        beat_grids=[0, 12, 24, 36],
        activity_grids=[0.0] * 12 + [0.5] + [0.0] * 35,
        instrument=InstrumentBarFeature(bass_activity=0.8),
        instrument_grid_features=[
            InstrumentGridFeature(grid=7, bass_onset=0.9),
            InstrumentGridFeature(grid=12, bass_onset=0.9),
        ],
    )

    salience = build_bar_hit_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert 7 not in points
    assert points[12].hit > 0.10
    assert points[12].reasons == ["beat", "stem:bass-beat"]


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


def test_accent_salience_uses_drum_vocal_and_accompaniment_context():
    bar = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.6,
        phrase_position="phrase_start",
        transition_role="peak",
        instrument=InstrumentBarFeature(
            vocal_activity=0.7,
            drum_activity=0.8,
            other_activity=0.7,
        ),
        grid_features=[GridFeature(grid=6, onset=True, strength=0.5)],
        instrument_grid_features=[
            InstrumentGridFeature(
                grid=6,
                vocal_onset=0.8,
                drum_onset=0.9,
                accompaniment_onset=0.8,
            )
        ],
    )

    salience = build_bar_accent_salience(bar)
    point = salience.points[0]

    assert point.accent > 0.7
    assert "accent:drum-onset" in point.reasons
    assert "accent:vocal-context" in point.reasons
    assert "accent:accompaniment-highlight" in point.reasons
    assert "stem:accompaniment-support" in point.reasons


def test_vocal_and_accompaniment_onsets_do_not_create_hits_without_base_evidence():
    bar = BarFeature(
        index=2,
        start_time=4.0,
        end_time=6.0,
        energy=0.6,
        phrase_position="phrase_start",
        transition_role="peak",
        instrument=InstrumentBarFeature(vocal_activity=0.8, other_activity=0.8),
        instrument_grid_features=[
            InstrumentGridFeature(
                grid=6,
                vocal_onset=1.0,
                accompaniment_onset=1.0,
            )
        ],
    )

    assert build_bar_accent_salience(bar).points == []


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
    assert points[16].ka_preference == 0.44
    assert points[16].reasons[-1] == "color:high"
    assert points[28].don_preference == 0.0
    assert points[28].ka_preference == 0.0
    assert not any(reason.startswith("color:") for reason in points[28].reasons)


def test_don_ka_salience_uses_bass_onset_as_weak_don_evidence():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.5,
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
        instrument=InstrumentBarFeature(drum_activity=0.7, bass_activity=0.8),
        instrument_grid_features=[
            InstrumentGridFeature(grid=0, drum_onset=0.7, bass_onset=0.9),
            InstrumentGridFeature(grid=6, drum_onset=0.7, bass_onset=0.9),
        ],
    )

    salience = build_bar_don_ka_salience(bar)
    points = {point.grid: point for point in salience.points}

    assert points[0].don_preference > points[6].don_preference
    assert points[6].don_preference < 0.40
    assert "color:bass-onset" in points[0].reasons
    assert "color:bass-onset" in points[6].reasons


def test_don_ka_salience_requires_confident_band_attack_for_color_bias():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        grids_per_bar=48,
        spectral_grid_features=[
            SpectralGridFeature(grid=4, low_onset_strength=0.3),
            SpectralGridFeature(grid=16, high_onset_strength=0.3),
        ],
    )

    salience = build_bar_don_ka_salience(bar)

    assert all(point.don_preference == 0.0 for point in salience.points)
    assert all(point.ka_preference == 0.0 for point in salience.points)
    assert not any(
        reason.startswith("color:")
        for point in salience.points
        for reason in point.reasons
    )


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


def test_don_ka_salience_calibrates_high_attack_with_percussive_brightness():
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

    assert points[4].ka_preference == 0.352
    assert points[4].reasons[-3:] == [
        "color:high",
        "color:percussive-high",
        "color:bright-percussive",
    ]
    assert points[20].ka_preference == 0.0
    assert "color:bright-percussive" not in points[20].reasons


def test_don_ka_salience_does_not_use_global_brightness_without_percussive_support():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.0,
        grids_per_bar=48,
        brightness=1.0,
        percussive_ratio=0.0,
        spectral_grid_features=[
            SpectralGridFeature(grid=4, high_onset_strength=0.4),
        ],
    )

    salience = build_bar_don_ka_salience(bar)
    point = salience.points[0]

    assert point.ka_preference == 0.22
    assert point.reasons[-1] == "color:high"
    assert "color:percussive-high" not in point.reasons
    assert "color:bright-percussive" not in point.reasons


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


def test_burst_salience_uses_late_drum_onset_rise_without_mix_onsets():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.6,
        grids_per_bar=48,
        instrument=InstrumentBarFeature(drum_activity=0.8),
        instrument_grid_features=[
            InstrumentGridFeature(grid=27, drum_onset=0.9),
            InstrumentGridFeature(grid=33, drum_onset=0.9),
            InstrumentGridFeature(grid=39, drum_onset=0.9),
            InstrumentGridFeature(grid=45, drum_onset=0.9),
        ],
    )

    salience = build_bar_burst_salience(bar)

    assert is_reliable_burst(salience)
    assert salience.burst_start_grid == 27
    assert salience.burst_end_grid == 45
    assert "burst:onset-density" in salience.burst_reasons
    assert "burst:drum-onset-rise" in salience.burst_reasons


def test_burst_salience_detects_late_onset_flux_and_percussive_rise():
    bars = [
        BarFeature(
            index=0,
            start_time=0.0,
            end_time=2.0,
            energy=0.5,
            grids_per_bar=48,
            onset_grids=[0, 12, 24, 36],
            percussive_ratio=0.2,
            section_id="section-a",
        ),
        BarFeature(
            index=1,
            start_time=2.0,
            end_time=4.0,
            energy=0.8,
            grids_per_bar=48,
            onset_grids=[0, 12, 24, 30, 36, 42],
            percussive_ratio=0.8,
            phrase_position="phrase_end",
            boundary_confidence=0.9,
            transition_role="cadence",
            transition_confidence=0.9,
            section_id="section-a",
            spectral_grid_features=[
                SpectralGridFeature(grid=30, spectral_flux=0.9),
                SpectralGridFeature(grid=42, spectral_flux=0.8),
            ],
        ),
        BarFeature(
            index=2,
            start_time=4.0,
            end_time=6.0,
            energy=0.9,
            grids_per_bar=48,
            onset_grids=[0, 12, 24, 36],
            transition_role="peak",
            section_id="section-b",
        ),
    ]

    burst = build_burst_salience(bars)[1]

    assert is_reliable_burst(burst)
    assert burst.burst_score > 0.8
    assert burst.burst_confidence > 0.8
    assert burst.burst_start_grid == 24
    assert burst.burst_end_grid == 42
    assert burst.burst_reasons == [
        "burst:onset-density",
        "burst:spectral-flux",
        "burst:percussive-rise",
        "burst:stable-contrast",
        "burst:phrase-end",
        "burst:cadence",
        "burst:next-section",
        "burst:next-highlight",
    ]


def test_burst_salience_accepts_flux_burst_when_onset_detector_is_sparse():
    previous = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.4,
        grids_per_bar=48,
        onset_grids=[0, 12, 24, 36],
        percussive_ratio=0.2,
    )
    burst_bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.6,
        grids_per_bar=48,
        onset_grids=[0],
        percussive_ratio=0.8,
        spectral_grid_features=[
            SpectralGridFeature(grid=30, spectral_flux=0.9),
            SpectralGridFeature(grid=42, spectral_flux=0.8),
        ],
    )

    burst = build_bar_burst_salience(burst_bar, previous_bar=previous)

    assert is_reliable_burst(burst)
    assert burst.burst_start_grid == 30
    assert burst.burst_end_grid == 42
    assert "burst:spectral-flux" in burst.burst_reasons
    assert "burst:percussive-rise" in burst.burst_reasons
    assert "burst:onset-density" not in burst.burst_reasons


def test_burst_salience_detects_strong_onset_density_without_structure_context():
    previous = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.5,
        grids_per_bar=48,
        onset_grids=[0, 12, 24, 36],
    )
    burst_bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.7,
        grids_per_bar=48,
        onset_grids=[0, 12, 24, 28, 32, 36, 40, 44],
    )

    burst = build_bar_burst_salience(burst_bar, previous_bar=previous)

    assert is_reliable_burst(burst)
    assert burst.burst_start_grid == 24
    assert burst.burst_end_grid == 44
    assert "burst:onset-density" in burst.burst_reasons
    assert "burst:stable-contrast" in burst.burst_reasons
    assert not any("phrase" in reason for reason in burst.burst_reasons)


@pytest.mark.parametrize("time_signature", ["3/4", "6/8"])
def test_burst_salience_supports_compact_meter_canonical_grids(
    time_signature: str,
):
    previous = BarFeature(
        index=0,
        start_time=0.0,
        end_time=1.5,
        energy=0.5,
        time_signature=time_signature,
        grids_per_bar=36,
        onset_grids=[0, 9, 18, 27],
    )
    burst_bar = BarFeature(
        index=1,
        start_time=1.5,
        end_time=3.0,
        energy=0.8,
        time_signature=time_signature,
        grids_per_bar=36,
        onset_grids=[0, 9, 18, 21, 24, 27, 30, 33],
    )

    burst = build_bar_burst_salience(burst_bar, previous_bar=previous)

    assert is_reliable_burst(burst)
    assert burst.burst_start_grid == 18
    assert burst.burst_end_grid == 33


def test_reliable_burst_span_projects_to_playable_output_ticks():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.8,
        grids_per_bar=48,
    )
    salience = BarRhythmicSalience(
        burst_score=0.8,
        burst_confidence=0.9,
        burst_start_grid=25,
        burst_end_grid=46,
    )

    assert project_reliable_burst_span(bar, salience, output_resolution=16) == (
        27,
        45,
    )
    assert project_reliable_burst_span(bar, salience, output_resolution=24) == (
        26,
        46,
    )


def test_unreliable_burst_has_no_projected_span():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.8,
        grids_per_bar=48,
    )
    salience = BarRhythmicSalience(
        burst_score=0.39,
        burst_confidence=0.9,
        burst_start_grid=25,
        burst_end_grid=46,
    )

    assert project_reliable_burst_span(bar, salience, output_resolution=16) is None


def test_burst_salience_does_not_promote_phrase_end_without_rhythmic_burst():
    bar = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.6,
        grids_per_bar=48,
        onset_grids=[0, 12, 24, 36],
        phrase_position="phrase_end",
        boundary_confidence=1.0,
        transition_role="cadence",
        transition_confidence=1.0,
        percussive_ratio=0.9,
    )

    burst = build_bar_burst_salience(bar)

    assert not is_reliable_burst(burst)
    assert burst.burst_score == 0.0
    assert burst.burst_confidence == 0.0
    assert burst.burst_start_grid is None
    assert burst.burst_end_grid is None
    assert burst.burst_reasons == []


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
