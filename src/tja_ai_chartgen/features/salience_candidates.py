from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tja_ai_chartgen.features.resolution import output_resolution_for_bar
from tja_ai_chartgen.features.salience import (
    ACTIVE_GRID_THRESHOLD,
    MIN_USABLE_BAR_CONFIDENCE,
    build_don_ka_salience,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    ResolutionPlan,
    RhythmicSaliencePoint,
)

SalienceCandidateKind = Literal[
    "strong-transient",
    "transient",
    "rhythmic-skeleton",
    "structure-highlight",
    "weak-evidence",
]

STRONG_TRANSIENT_HIT_THRESHOLD = 0.65
ACCENT_CANDIDATE_THRESHOLD = 0.50
CANDIDATE_PRIORITY: dict[SalienceCandidateKind, int] = {
    "strong-transient": 0,
    "transient": 1,
    "rhythmic-skeleton": 2,
    "structure-highlight": 3,
    "weak-evidence": 4,
}


@dataclass(frozen=True)
class SalienceCandidate:
    grid: int
    kind: SalienceCandidateKind
    priority: int
    score: float
    point: RhythmicSaliencePoint

    @property
    def reliable(self) -> bool:
        return self.kind != "weak-evidence"


def build_salience_candidate_bars(
    bars: list[BarFeature],
    *,
    resolution_plan: ResolutionPlan | None = None,
) -> list[list[SalienceCandidate]]:
    """构建全曲候选；每个小节只保留当前输出 resolution 可精确表达的点。"""
    salience_bars = build_don_ka_salience(bars)
    return [
        rank_bar_salience_candidates(
            bar,
            salience,
            output_resolution=output_resolution_for_bar(
                bar,
                plan=resolution_plan,
                position=position,
            ),
        )
        for position, (bar, salience) in enumerate(
            zip(bars, salience_bars, strict=True)
        )
    ]


def rank_bar_salience_candidates(
    bar: BarFeature,
    salience: BarRhythmicSalience,
    *,
    output_resolution: int,
) -> list[SalienceCandidate]:
    """按证据可靠性排序，并过滤越界、重复和不可表达的 canonical 点。"""
    step = _canonical_step(bar, output_resolution)
    candidates = [
        _candidate_from_point(point, bar_confidence=salience.confidence)
        for point in salience.points
        if 0 <= point.grid < bar.grids_per_bar and point.grid % step == 0
    ]
    candidates.sort(key=_candidate_sort_key)

    unique: list[SalienceCandidate] = []
    seen_grids: set[int] = set()
    for candidate in candidates:
        if candidate.grid in seen_grids:
            continue
        seen_grids.add(candidate.grid)
        unique.append(candidate)
    return unique


def rank_accent_candidates(
    candidates: list[SalienceCandidate],
) -> list[SalienceCandidate]:
    """从统一 salience 中选出可靠重音候选，并按强调价值排序。"""
    accent_candidates = [
        candidate
        for candidate in candidates
        if candidate.point.accent >= ACCENT_CANDIDATE_THRESHOLD
        and (
            candidate.reliable
            or "accent:downbeat" in candidate.point.reasons
            or "accent:hint" in candidate.point.reasons
        )
    ]
    return sorted(
        accent_candidates,
        key=lambda candidate: (
            -candidate.point.accent,
            candidate.priority,
            -candidate.score,
            -candidate.point.confidence,
            candidate.grid,
        ),
    )


def is_salience_grid_representable(
    bar: BarFeature,
    grid: int,
    *,
    output_resolution: int,
) -> bool:
    """判断 canonical grid 是否能由目标输出 resolution 无量化误差地表达。"""
    step = _canonical_step(bar, output_resolution)
    return 0 <= grid < bar.grids_per_bar and grid % step == 0


def _canonical_step(bar: BarFeature, output_resolution: int) -> int:
    if output_resolution <= 0 or bar.grids_per_bar % output_resolution:
        raise ValueError(
            f"Bar {bar.index} canonical grid {bar.grids_per_bar} is incompatible with "
            f"resolution {output_resolution}"
        )
    return bar.grids_per_bar // output_resolution


def _candidate_from_point(
    point: RhythmicSaliencePoint,
    *,
    bar_confidence: float,
) -> SalienceCandidate:
    kind = _candidate_kind(point, bar_confidence=bar_confidence)
    confidence = min(point.confidence, bar_confidence) if bar_confidence > 0 else point.confidence
    score = (
        point.hit * 0.55
        + confidence * 0.25
        + point.accent * 0.10
        + point.sustained_activity * 0.05
        + max(point.don_preference, point.ka_preference) * 0.05
    )
    return SalienceCandidate(
        grid=point.grid,
        kind=kind,
        priority=CANDIDATE_PRIORITY[kind],
        score=round(score, 6),
        point=point,
    )


def _candidate_kind(
    point: RhythmicSaliencePoint,
    *,
    bar_confidence: float,
) -> SalienceCandidateKind:
    reasons = set(point.reasons)
    reliable = (
        point.confidence >= MIN_USABLE_BAR_CONFIDENCE
        and bar_confidence >= MIN_USABLE_BAR_CONFIDENCE
    )
    transient = "onset" in reasons or "spectral" in reasons
    if reliable and transient and point.hit >= STRONG_TRANSIENT_HIT_THRESHOLD:
        return "strong-transient"
    if reliable and transient:
        return "transient"
    if reliable and point.sustained_activity >= ACTIVE_GRID_THRESHOLD and (
        "downbeat" in reasons or "beat" in reasons
    ):
        return "rhythmic-skeleton"
    if reliable and (
        point.accent > 0
        or any(reason.startswith("role:") for reason in reasons)
        or any(reason.startswith("accent:") for reason in reasons)
    ):
        return "structure-highlight"
    return "weak-evidence"


def _candidate_sort_key(candidate: SalienceCandidate) -> tuple[float, ...]:
    point = candidate.point
    return (
        float(candidate.priority),
        -candidate.score,
        -point.confidence,
        -point.hit,
        -point.accent,
        float(candidate.grid),
    )
