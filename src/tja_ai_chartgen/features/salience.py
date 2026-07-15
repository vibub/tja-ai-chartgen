from __future__ import annotations

from dataclasses import dataclass

from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.features.silence import edge_silence_indexes, is_silent_bar
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    RhythmicSaliencePoint,
)

RHYTHMIC_SALIENCE_FEATURE_VERSION = "rhythmic-salience-v1"
ACTIVE_GRID_THRESHOLD = 0.08
ABSOLUTE_ONSET_GATE = 0.08
ABSOLUTE_SPECTRAL_GATE = 0.12
ABSOLUTE_BEAT_DRIVE_GATE = 0.08
HIT_OUTPUT_THRESHOLD = 0.02
SPECTRAL_EVIDENCE_THRESHOLD = 0.05
ACCENT_OUTPUT_THRESHOLD = 0.15
MIN_USABLE_BAR_CONFIDENCE = 0.45
COLOR_DOMINANCE_MARGIN = 0.12
COLOR_PREFERENCE_CAP = 0.75
MAX_STRONG_COLOR_RUN = 3
SALIENCE_FALLBACK_EDGE_SILENCE = "edge-silence"
SALIENCE_FALLBACK_SILENT_BAR = "silent-bar"
SALIENCE_FALLBACK_NO_EVIDENCE = "no-rhythmic-evidence"
SALIENCE_FALLBACK_BEAT_ONLY = "beat-skeleton-only"
SALIENCE_FALLBACK_LOW_CONFIDENCE = "low-confidence"
STRUCTURE_HIT_MULTIPLIERS = {
    "build_up": 1.04,
    "peak": 1.08,
    "fill": 1.06,
    "cadence": 1.06,
    "drop": 0.92,
    "breakdown": 0.85,
}
STRUCTURE_ACCENT_MULTIPLIERS = {
    "build_up": 1.03,
    "peak": 1.08,
    "fill": 1.12,
    "cadence": 1.12,
    "drop": 0.95,
    "breakdown": 0.85,
}


@dataclass(frozen=True)
class CanonicalRhythmicEvidencePoint:
    grid: int
    onset: bool = False
    onset_strength: float = 0.0
    accent_hint: bool = False
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
    accent_grids = {grid for grid in bar.accent_grids if 0 <= grid < grid_count}
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
    accent_hints = [grid in accent_grids for grid in range(grid_count)]
    explicit_onset_strengths: dict[int, float] = {}
    downbeats = [grid in downbeat_grids for grid in range(grid_count)]

    for feature in bar.grid_features:
        if feature.grid < 0 or feature.grid >= grid_count:
            continue
        grid = feature.grid
        onsets[grid] = onsets[grid] or feature.onset
        accent_hints[grid] = accent_hints[grid] or feature.accent
        activities[grid] = max(activities[grid], _unit_value(feature.activity))
        if feature.onset and feature.strength > 0:
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
            accent_hint=accent_hints[grid],
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
    global_transient_reference = _global_transient_reference(
        [bar for position, bar in enumerate(bars) if position not in silent_indexes]
    )
    return [
        build_bar_hit_salience(
            bar,
            force_silent=position in silent_indexes,
            global_transient_reference=global_transient_reference,
        )
        for position, bar in enumerate(bars)
    ]


def build_bar_hit_salience(
    bar: BarFeature,
    *,
    force_silent: bool = False,
    global_transient_reference: float | None = None,
) -> BarRhythmicSalience:
    """融合基础瞬态与节拍骨架，不让持续 activity 单独制造 hit。"""
    evidence = build_canonical_rhythmic_evidence(bar)
    has_transient_evidence = any(
        item.onset or _spectral_strength(item) >= SPECTRAL_EVIDENCE_THRESHOLD
        for item in evidence
    )
    if force_silent:
        return BarRhythmicSalience(fallback_reason=SALIENCE_FALLBACK_EDGE_SILENCE)
    if is_silent_bar(bar) and not has_transient_evidence:
        return BarRhythmicSalience(fallback_reason=SALIENCE_FALLBACK_SILENT_BAR)

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
                confidence=round(
                    _point_confidence(
                        item,
                        spectral_strength,
                        evidence,
                        bar_energy=bar.energy,
                        global_transient_reference=global_transient_reference,
                    ),
                    6,
                ),
                reasons=reasons,
            )
        )

    active_grid_count = sum(item.activity >= ACTIVE_GRID_THRESHOLD for item in evidence)
    active_ratio = active_grid_count / len(evidence)
    onset_evidence_count = sum(item.onset for item in evidence)
    point_grids = {point.grid for point in points}
    transient_point_count = sum(
        item.grid in point_grids
        and (
            _effective_onset_strength(item) >= ABSOLUTE_ONSET_GATE
            or (
                item.grid in spectral_peaks
                and _spectral_strength(item) >= ABSOLUTE_SPECTRAL_GATE
            )
        )
        for item in evidence
    )
    confidence = _bar_confidence(
        points,
        active_ratio=active_ratio,
        transient_point_count=transient_point_count,
    )
    return BarRhythmicSalience(
        points=points,
        active_ratio=round(active_ratio, 6),
        onset_evidence_count=onset_evidence_count,
        confidence=round(confidence, 6),
        fallback_reason=_bar_fallback_reason(
            points,
            transient_point_count=transient_point_count,
            confidence=confidence,
        ),
    )


def build_accent_salience(bars: list[BarFeature]) -> list[BarRhythmicSalience]:
    """在 hit salience 上补充歌曲上下文中的 accent salience。"""
    silent_indexes = edge_silence_indexes(bars)
    hit_results = build_hit_salience(bars)
    results: list[BarRhythmicSalience] = []
    for position, (bar, hit_salience) in enumerate(
        zip(bars, hit_results, strict=True)
    ):
        force_silent = position in silent_indexes
        results.append(
            build_bar_accent_salience(
                bar,
                hit_salience=hit_salience,
                force_silent=force_silent,
                start_reason=_bar_start_reason(bars, position),
            )
        )
    return results


def build_bar_accent_salience(
    bar: BarFeature,
    *,
    hit_salience: BarRhythmicSalience | None = None,
    force_silent: bool = False,
    start_reason: str | None = None,
) -> BarRhythmicSalience:
    """只为已有 hit 候选计算 accent，不直接决定大音符。"""
    if force_silent:
        return hit_salience or BarRhythmicSalience(
            fallback_reason=SALIENCE_FALLBACK_EDGE_SILENCE
        )
    base = hit_salience or build_bar_hit_salience(bar)
    if not base.points:
        return base

    evidence = build_canonical_rhythmic_evidence(bar)
    evidence_by_grid = {item.grid: item for item in evidence}
    onset_peaks = _onset_peak_grids(evidence)
    spectral_peaks = _spectral_peak_grids(evidence)
    first_hit_grid = base.points[0].grid
    resolved_start_reason = start_reason or _single_bar_start_reason(bar)
    role_multiplier = STRUCTURE_ACCENT_MULTIPLIERS.get(bar.transition_role, 1.0)

    points: list[RhythmicSaliencePoint] = []
    for point in base.points:
        item = evidence_by_grid[point.grid]
        accent = 0.0
        accent_reasons: list[str] = []

        if item.downbeat:
            accent = max(accent, 0.58)
            accent_reasons.append("accent:downbeat")
        if item.accent_hint:
            accent = max(accent, 0.50)
            accent_reasons.append("accent:hint")
        if item.grid in onset_peaks:
            onset_strength = item.onset_strength if item.onset_strength > 0 else 0.65
            accent = max(accent, 0.45 + onset_strength * 0.45)
            accent_reasons.append("accent:onset-peak")
        if (
            item.grid in spectral_peaks
            and item.low_onset_strength >= SPECTRAL_EVIDENCE_THRESHOLD
        ):
            accent = max(accent, 0.35 + item.low_onset_strength * 0.40)
            accent_reasons.append("accent:low-attack")
        if point.grid == first_hit_grid and resolved_start_reason is not None:
            accent = max(accent, 0.50 + bar.boundary_confidence * 0.25)
            accent_reasons.append(f"accent:{resolved_start_reason}")
        if point.grid == first_hit_grid and bar.energy_delta > 0.10:
            accent = max(accent, 0.35 + bar.energy_delta * 0.35)
            accent_reasons.append("accent:energy-rise")

        if accent >= ACCENT_OUTPUT_THRESHOLD and role_multiplier != 1.0:
            accent = _unit_value(accent * role_multiplier)
            accent_reasons.append(f"accent:role:{bar.transition_role}")
        if accent < ACCENT_OUTPUT_THRESHOLD:
            accent = 0.0
            accent_reasons = []

        points.append(
            point.model_copy(
                update={
                    "accent": round(accent, 6),
                    "reasons": [*point.reasons, *accent_reasons],
                }
            )
        )

    return base.model_copy(update={"points": points})


def build_don_ka_salience(bars: list[BarFeature]) -> list[BarRhythmicSalience]:
    """为已有 hit/accent 点增加咚咔软倾向，并限制连续强单色提示。"""
    accent_results = build_accent_salience(bars)
    results = [
        _apply_bar_don_ka_salience(bar, accent_salience)
        for bar, accent_salience in zip(bars, accent_results, strict=True)
    ]
    return _balance_don_ka_runs(results)


def build_bar_don_ka_salience(
    bar: BarFeature,
    *,
    accent_salience: BarRhythmicSalience | None = None,
) -> BarRhythmicSalience:
    """计算单小节咚咔倾向；最终配色仍由 style 和生成器决定。"""
    base = accent_salience or build_bar_accent_salience(bar)
    return _balance_don_ka_runs([_apply_bar_don_ka_salience(bar, base)])[0]


def _apply_bar_don_ka_salience(
    bar: BarFeature,
    base: BarRhythmicSalience,
) -> BarRhythmicSalience:
    if not base.points:
        return base

    canonical_evidence = build_canonical_rhythmic_evidence(bar)
    evidence = {item.grid: item for item in canonical_evidence}
    spectral_peaks = _spectral_peak_grids(canonical_evidence)
    offbeat_grids = _offbeat_grids(bar)
    points: list[RhythmicSaliencePoint] = []
    for point in base.points:
        item = evidence[point.grid]
        don_preference = 0.0
        ka_preference = 0.0
        color_reasons: list[str] = []

        low_drive = item.low_onset_strength
        mid_drive = item.mid_onset_strength
        high_drive = item.high_onset_strength
        low_dominant = (
            item.grid in spectral_peaks
            and low_drive >= SPECTRAL_EVIDENCE_THRESHOLD
            and low_drive - max(mid_drive, high_drive) >= COLOR_DOMINANCE_MARGIN
        )
        high_dominant = (
            item.grid in spectral_peaks
            and high_drive >= SPECTRAL_EVIDENCE_THRESHOLD
            and high_drive - max(low_drive, mid_drive) >= COLOR_DOMINANCE_MARGIN
        )
        if low_dominant:
            don_preference = max(don_preference, low_drive * 0.65)
            color_reasons.append("color:low")
        elif high_dominant:
            ka_preference = max(ka_preference, high_drive * 0.65)
            color_reasons.append("color:high")

        if item.downbeat:
            don_preference = max(don_preference, 0.28)
            color_reasons.append("color:downbeat")
        if item.grid in offbeat_grids:
            ka_preference = max(ka_preference, 0.24)
            color_reasons.append("color:offbeat")

        if (
            high_dominant
            and bar.brightness >= 0.55
            and bar.percussive_ratio >= 0.45
        ):
            brightness_cue = 0.15 + bar.brightness * bar.percussive_ratio * 0.25
            ka_preference = max(ka_preference, brightness_cue)
            color_reasons.append("color:bright-percussive")

        points.append(
            point.model_copy(
                update={
                    "don_preference": round(
                        min(COLOR_PREFERENCE_CAP, don_preference),
                        6,
                    ),
                    "ka_preference": round(
                        min(COLOR_PREFERENCE_CAP, ka_preference),
                        6,
                    ),
                    "reasons": [*point.reasons, *color_reasons],
                }
            )
        )
    return base.model_copy(update={"points": points})


def _offbeat_grids(bar: BarFeature) -> set[int]:
    grid_count = bar.grids_per_bar
    if grid_count <= 0:
        return set()
    beats = sorted({grid for grid in bar.beat_grids if 0 <= grid < grid_count})
    if not beats:
        meter = get_meter_spec(bar.time_signature)
        beats = list(meter.beat_grids_for_resolution(grid_count))
    if not beats:
        return set()

    offbeats: set[int] = set()
    for index, grid in enumerate(beats):
        next_grid = beats[index + 1] if index + 1 < len(beats) else beats[0] + grid_count
        midpoint = round((grid + next_grid) / 2) % grid_count
        if midpoint not in beats:
            offbeats.add(midpoint)
    return offbeats


def _balance_don_ka_runs(
    bars: list[BarRhythmicSalience],
) -> list[BarRhythmicSalience]:
    dominant_color: str | None = None
    run_length = 0
    balanced_bars: list[BarRhythmicSalience] = []
    for bar in bars:
        points: list[RhythmicSaliencePoint] = []
        for point in bar.points:
            current = _dominant_color_preference(point)
            if current is None:
                dominant_color = None
                run_length = 0
                points.append(point)
                continue
            if current == dominant_color:
                run_length += 1
            else:
                dominant_color = current
                run_length = 1
            if run_length <= MAX_STRONG_COLOR_RUN:
                points.append(point)
                continue

            field = "don_preference" if current == "don" else "ka_preference"
            opponent = (
                point.ka_preference if current == "don" else point.don_preference
            )
            softened = min(getattr(point, field), max(0.15, opponent + 0.08))
            points.append(
                point.model_copy(
                    update={
                        field: round(softened, 6),
                        "reasons": [*point.reasons, "color:balance"],
                    }
                )
            )
            dominant_color = None
            run_length = 0
        balanced_bars.append(bar.model_copy(update={"points": points}))
    return balanced_bars


def _dominant_color_preference(point: RhythmicSaliencePoint) -> str | None:
    difference = point.don_preference - point.ka_preference
    if point.don_preference >= 0.35 and difference >= COLOR_DOMINANCE_MARGIN:
        return "don"
    if point.ka_preference >= 0.35 and -difference >= COLOR_DOMINANCE_MARGIN:
        return "ka"
    return None


def _onset_peak_grids(
    evidence: list[CanonicalRhythmicEvidencePoint],
) -> set[int]:
    strengths = [
        item.onset_strength if item.onset_strength > 0 else 0.65 if item.onset else 0.0
        for item in evidence
    ]
    peaks: set[int] = set()
    for index, item in enumerate(evidence):
        if not item.onset:
            continue
        strength = strengths[index]
        left = strengths[index - 1] if index > 0 else 0.0
        right = strengths[index + 1] if index + 1 < len(strengths) else 0.0
        if strength > left and strength >= right:
            peaks.add(item.grid)
    return peaks


def _bar_start_reason(bars: list[BarFeature], position: int) -> str | None:
    bar = bars[position]
    if position == 0:
        return "song-start"
    if bar.phrase_position == "phrase_start":
        return "phrase-start"

    previous = bars[position - 1]
    if (
        bar.section_id is not None
        and previous.section_id is not None
        and bar.section_id != previous.section_id
    ):
        return "section-start"
    if (
        bar.phrase_id is not None
        and previous.phrase_id is not None
        and bar.phrase_id != previous.phrase_id
    ):
        return "phrase-start"
    return None


def _single_bar_start_reason(bar: BarFeature) -> str | None:
    if bar.index == 0:
        return "song-start"
    if bar.phrase_position == "phrase_start":
        return "phrase-start"
    return None


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


def _point_confidence(
    item: CanonicalRhythmicEvidencePoint,
    spectral_strength: float,
    evidence: list[CanonicalRhythmicEvidencePoint],
    *,
    bar_energy: float,
    global_transient_reference: float | None,
) -> float:
    onset_strength = _effective_onset_strength(item)
    onset_confidence = (
        0.45 + onset_strength * 0.45
        if onset_strength >= ABSOLUTE_ONSET_GATE
        else 0.0
    )
    spectral_confidence = (
        0.30 + spectral_strength * 0.45
        if spectral_strength >= ABSOLUTE_SPECTRAL_GATE
        else 0.0
    )
    drive = max(item.activity, _unit_value(bar_energy))
    beat_confidence = (
        0.20 + drive * 0.25
        if (item.beat is not None or item.downbeat)
        and drive >= ABSOLUTE_BEAT_DRIVE_GATE
        else 0.0
    )
    confidence = max(onset_confidence, spectral_confidence, beat_confidence)
    if onset_confidence > 0 and spectral_confidence > 0:
        confidence += 0.10
    if item.activity >= ACTIVE_GRID_THRESHOLD:
        confidence += 0.05
    confidence += _local_transient_margin(item.grid, evidence) * 0.10
    transient_strength = max(onset_strength, spectral_strength)
    if global_transient_reference is not None and global_transient_reference > 0:
        confidence += min(1.0, transient_strength / global_transient_reference) * 0.05
    return _unit_value(confidence)


def _global_transient_reference(bars: list[BarFeature]) -> float | None:
    strengths: list[float] = []
    for bar in bars:
        evidence = build_canonical_rhythmic_evidence(bar)
        spectral_peaks = _spectral_peak_grids(evidence)
        for item in evidence:
            onset_strength = _effective_onset_strength(item)
            if onset_strength >= ABSOLUTE_ONSET_GATE:
                strengths.append(onset_strength)
            spectral_strength = _spectral_strength(item)
            if (
                item.grid in spectral_peaks
                and spectral_strength >= ABSOLUTE_SPECTRAL_GATE
            ):
                strengths.append(spectral_strength)
    if not strengths:
        return None
    return _percentile(strengths, 0.75)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = _unit_value(quantile) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _local_transient_margin(
    grid: int,
    evidence: list[CanonicalRhythmicEvidencePoint],
) -> float:
    current = evidence[grid]
    current_strength = max(
        _effective_onset_strength(current),
        _spectral_strength(current),
    )
    neighbor_strengths = [0.0]
    if grid > 0:
        neighbor_strengths.append(
            max(
                _effective_onset_strength(evidence[grid - 1]),
                _spectral_strength(evidence[grid - 1]),
            )
        )
    if grid + 1 < len(evidence):
        neighbor_strengths.append(
            max(
                _effective_onset_strength(evidence[grid + 1]),
                _spectral_strength(evidence[grid + 1]),
            )
        )
    return _unit_value(current_strength - max(neighbor_strengths))


def _bar_confidence(
    points: list[RhythmicSaliencePoint],
    *,
    active_ratio: float,
    transient_point_count: int,
) -> float:
    if not points:
        return 0.0
    mean_point_confidence = sum(point.confidence for point in points) / len(points)
    transient_ratio = transient_point_count / len(points)
    event_support = min(1.0, transient_point_count / 4)
    active_support = min(1.0, active_ratio / 0.25)
    return _unit_value(
        mean_point_confidence * 0.65
        + transient_ratio * 0.15
        + event_support * 0.10
        + active_support * 0.10
    )


def _bar_fallback_reason(
    points: list[RhythmicSaliencePoint],
    *,
    transient_point_count: int,
    confidence: float,
) -> str | None:
    if not points:
        return SALIENCE_FALLBACK_NO_EVIDENCE
    if transient_point_count == 0:
        return SALIENCE_FALLBACK_BEAT_ONLY
    if confidence < MIN_USABLE_BAR_CONFIDENCE:
        return SALIENCE_FALLBACK_LOW_CONFIDENCE
    return None


def _effective_onset_strength(item: CanonicalRhythmicEvidencePoint) -> float:
    if not item.onset:
        return 0.0
    return item.onset_strength if item.onset_strength > 0 else 0.65


def _transient_hit_strength(
    item: CanonicalRhythmicEvidencePoint,
    spectral_strength: float,
) -> float:
    onset_strength = _effective_onset_strength(item)
    if onset_strength < ABSOLUTE_ONSET_GATE:
        onset_strength = 0.0
    if spectral_strength < ABSOLUTE_SPECTRAL_GATE:
        spectral_strength = 0.0
    return max(onset_strength, spectral_strength * 0.75)


def _beat_skeleton_strength(
    item: CanonicalRhythmicEvidencePoint,
    bar_energy: float,
) -> float:
    if item.beat is None and not item.downbeat:
        return 0.0
    drive = max(item.activity, _unit_value(bar_energy))
    if drive < ABSOLUTE_BEAT_DRIVE_GATE:
        return 0.0
    base = 0.26 if item.downbeat else 0.20
    return base * drive


def _unit_value(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
