from __future__ import annotations

from dataclasses import dataclass

from tja_ai_chartgen.features.silence import edge_silence_indexes, is_silent_bar
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    RhythmicSaliencePoint,
)

RHYTHMIC_SALIENCE_FEATURE_VERSION = "rhythmic-salience-v1"
ACTIVE_GRID_THRESHOLD = 0.08
HIT_OUTPUT_THRESHOLD = 0.02
SPECTRAL_EVIDENCE_THRESHOLD = 0.05
STRUCTURE_HIT_MULTIPLIERS = {
    "build_up": 1.04,
    "peak": 1.08,
    "fill": 1.06,
    "cadence": 1.06,
    "drop": 0.92,
    "breakdown": 0.85,
}


@dataclass(frozen=True)
class CanonicalRhythmicEvidencePoint:
    grid: int
    onset: bool = False
    onset_strength: float = 0.0
    activity: float = 0.0
    beat: int | None = None
    downbeat: bool = False
    low_onset_strength: float = 0.0
    mid_onset_strength: float = 0.0
    high_onset_strength: float = 0.0
    spectral_flux: float = 0.0


def build_canonical_rhythmic_evidence(
    bar: BarFeature,
) -> list[CanonicalRhythmicEvidencePoint]:
    """将小节内分散的基础节奏证据对齐到完整 canonical grid。"""
    if bar.grids_per_bar <= 0:
        raise ValueError(f"Bar {bar.index} must have a positive canonical grid size")

    grid_count = bar.grids_per_bar
    onset_grids = {grid for grid in bar.onset_grids if 0 <= grid < grid_count}
    beat_numbers = {
        grid: number
        for number, grid in enumerate(
            sorted({grid for grid in bar.beat_grids if 0 <= grid < grid_count}),
            start=1,
        )
    }
    downbeat_grids = (
        {bar.downbeat_grid}
        if bar.downbeat_grid is not None and 0 <= bar.downbeat_grid < grid_count
        else set()
    )

    activities = [0.0] * grid_count
    for grid, value in enumerate(bar.activity_grids[:grid_count]):
        activities[grid] = _unit_value(value)

    onsets = [grid in onset_grids for grid in range(grid_count)]
    onset_strengths = [1.0 if onset else 0.0 for onset in onsets]
    explicit_onset_strengths: dict[int, float] = {}
    downbeats = [grid in downbeat_grids for grid in range(grid_count)]

    for feature in bar.grid_features:
        if feature.grid < 0 or feature.grid >= grid_count:
            continue
        grid = feature.grid
        onsets[grid] = onsets[grid] or feature.onset or feature.strength > 0
        activities[grid] = max(activities[grid], _unit_value(feature.activity))
        if feature.strength > 0:
            explicit_onset_strengths[grid] = max(
                explicit_onset_strengths.get(grid, 0.0),
                _unit_value(feature.strength),
            )
        if feature.beat is not None and feature.beat > 0:
            beat_numbers[grid] = feature.beat
        downbeats[grid] = downbeats[grid] or feature.downbeat

    for grid, strength in explicit_onset_strengths.items():
        onset_strengths[grid] = strength

    spectral_values: list[dict[str, float]] = [{} for _ in range(grid_count)]
    for feature in bar.spectral_grid_features:
        if feature.grid < 0 or feature.grid >= grid_count:
            continue
        values = spectral_values[feature.grid]
        for name in (
            "low_onset_strength",
            "mid_onset_strength",
            "high_onset_strength",
            "spectral_flux",
        ):
            values[name] = max(values.get(name, 0.0), _unit_value(getattr(feature, name)))

    return [
        CanonicalRhythmicEvidencePoint(
            grid=grid,
            onset=onsets[grid],
            onset_strength=onset_strengths[grid],
            activity=activities[grid],
            beat=beat_numbers.get(grid),
            downbeat=downbeats[grid],
            low_onset_strength=spectral_values[grid].get("low_onset_strength", 0.0),
            mid_onset_strength=spectral_values[grid].get("mid_onset_strength", 0.0),
            high_onset_strength=spectral_values[grid].get("high_onset_strength", 0.0),
            spectral_flux=spectral_values[grid].get("spectral_flux", 0.0),
        )
        for grid in range(grid_count)
    ]


def build_hit_salience(bars: list[BarFeature]) -> list[BarRhythmicSalience]:
    """为一组小节生成 hit salience，并将首尾静音小节强制归零。"""
    silent_indexes = edge_silence_indexes(bars)
    return [
        build_bar_hit_salience(bar, force_silent=position in silent_indexes)
        for position, bar in enumerate(bars)
    ]


def build_bar_hit_salience(
    bar: BarFeature,
    *,
    force_silent: bool = False,
) -> BarRhythmicSalience:
    """融合基础瞬态与节拍骨架，不让持续 activity 单独制造 hit。"""
    evidence = build_canonical_rhythmic_evidence(bar)
    has_transient_evidence = any(
        item.onset or _spectral_strength(item) >= SPECTRAL_EVIDENCE_THRESHOLD
        for item in evidence
    )
    if force_silent or (is_silent_bar(bar) and not has_transient_evidence):
        return BarRhythmicSalience()

    spectral_peaks = _spectral_peak_grids(evidence)
    role_multiplier = STRUCTURE_HIT_MULTIPLIERS.get(bar.transition_role, 1.0)
    points: list[RhythmicSaliencePoint] = []
    for item in evidence:
        spectral_strength = (
            _spectral_strength(item) if item.grid in spectral_peaks else 0.0
        )
        transient_strength = _transient_hit_strength(item, spectral_strength)
        beat_strength = _beat_skeleton_strength(item, bar.energy)
        hit = _unit_value(max(transient_strength, beat_strength) * role_multiplier)
        if hit < HIT_OUTPUT_THRESHOLD:
            continue

        reasons: list[str] = []
        if item.onset:
            reasons.append("onset")
        if spectral_strength >= SPECTRAL_EVIDENCE_THRESHOLD:
            reasons.append("spectral")
        if item.downbeat:
            reasons.append("downbeat")
        elif item.beat is not None:
            reasons.append("beat")
        if role_multiplier != 1.0:
            reasons.append(f"role:{bar.transition_role}")

        points.append(
            RhythmicSaliencePoint(
                grid=item.grid,
                hit=round(hit, 6),
                sustained_activity=round(item.activity, 6),
                reasons=reasons,
            )
        )

    active_grid_count = sum(item.activity >= ACTIVE_GRID_THRESHOLD for item in evidence)
    onset_evidence_count = sum(item.onset for item in evidence)
    return BarRhythmicSalience(
        points=points,
        active_ratio=round(active_grid_count / len(evidence), 6),
        onset_evidence_count=onset_evidence_count,
    )


def _spectral_peak_grids(
    evidence: list[CanonicalRhythmicEvidencePoint],
) -> set[int]:
    strengths = [_spectral_strength(item) for item in evidence]
    peaks: set[int] = set()
    for index, item in enumerate(evidence):
        strength = strengths[index]
        if strength < SPECTRAL_EVIDENCE_THRESHOLD:
            continue
        if item.onset:
            peaks.add(item.grid)
            continue
        left = strengths[index - 1] if index > 0 else 0.0
        right = strengths[index + 1] if index + 1 < len(strengths) else 0.0
        adjacent_onset = (
            (index > 0 and evidence[index - 1].onset)
            or (index + 1 < len(evidence) and evidence[index + 1].onset)
        )
        if not adjacent_onset and strength > left and strength >= right:
            peaks.add(item.grid)
    return peaks


def _spectral_strength(item: CanonicalRhythmicEvidencePoint) -> float:
    band_attack = max(
        item.low_onset_strength,
        item.mid_onset_strength,
        item.high_onset_strength,
    )
    return max(band_attack, item.spectral_flux * 0.8)


def _transient_hit_strength(
    item: CanonicalRhythmicEvidencePoint,
    spectral_strength: float,
) -> float:
    onset_strength = item.onset_strength
    if item.onset and onset_strength <= 0:
        onset_strength = 0.65
    return max(onset_strength, spectral_strength * 0.75)


def _beat_skeleton_strength(
    item: CanonicalRhythmicEvidencePoint,
    bar_energy: float,
) -> float:
    if item.beat is None and not item.downbeat:
        return 0.0
    drive = max(item.activity, _unit_value(bar_energy))
    if drive <= 0:
        return 0.0
    base = 0.26 if item.downbeat else 0.20
    return base * drive


def _unit_value(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
