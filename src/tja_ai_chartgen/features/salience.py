from __future__ import annotations

from dataclasses import dataclass

from tja_ai_chartgen.tja.model import BarFeature

RHYTHMIC_SALIENCE_FEATURE_VERSION = "rhythmic-salience-v1"


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


def _unit_value(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
