from __future__ import annotations

from math import isfinite

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.tja.model import (
    BarFeature,
    ResolutionDecision,
    ResolutionPlan,
    SongAnalysis,
)


MIN_ONSET_WEIGHT = 0.15
RESOLUTION_ERROR_TOLERANCE_TICKS = 0.2
LOW_CONFIDENCE_ERROR_MARGIN_TICKS = 0.05


def output_resolution_for_bar(
    bar: BarFeature,
    *,
    plan: ResolutionPlan | None = None,
    position: int | None = None,
) -> int:
    if plan is not None:
        if 0 <= bar.index < len(plan.bar_resolutions):
            return plan.bar_resolutions[bar.index]
        if position is not None and 0 <= position < len(plan.bar_resolutions):
            return plan.bar_resolutions[position]
        return plan.base_resolution
    meter = get_meter_spec(bar.time_signature)
    if bar.grids_per_bar == meter.grids_per_bar:
        return meter.legacy_grids_per_bar
    return bar.grids_per_bar


def output_resolution_for_analysis_bar(analysis: SongAnalysis, position: int) -> int:
    if position < 0 or position >= len(analysis.bars):
        return 16
    return output_resolution_for_bar(
        analysis.bars[position],
        plan=analysis.resolution_plan,
        position=position,
    )


def build_resolution_plan(
    raw: AudioAnalysisRaw,
    bars: list[BarFeature],
) -> ResolutionPlan:
    meter = get_meter_spec(raw.time_signature)
    candidates = _candidate_resolutions(meter.beats_per_bar)
    canonical_grids = meter.grids_per_bar
    samples = _weighted_onset_positions(raw, bar_count=len(bars), canonical_grids=canonical_grids)

    if not samples:
        selected = meter.legacy_grids_per_bar
        decision = ResolutionDecision(
            selected_resolution=selected,
            candidate_errors={},
            evidence_count=0,
            confidence=0.0,
            reason="insufficient reliable onset evidence; using the legacy stable resolution",
        )
        return _stable_plan(canonical_grids, selected, len(bars), decision)

    errors = {
        resolution: _weighted_quantization_error(
            samples,
            step=canonical_grids / resolution,
        )
        for resolution in candidates
    }
    qualified = [
        resolution
        for resolution in candidates
        if errors[resolution] <= RESOLUTION_ERROR_TOLERANCE_TICKS
    ]
    if qualified:
        selected = qualified[0]
        selection_reason = "met the quantization error tolerance"
    else:
        best_error = min(errors.values())
        selected = next(
            resolution
            for resolution in candidates
            if errors[resolution]
            <= best_error + LOW_CONFIDENCE_ERROR_MARGIN_TICKS
        )
        selection_reason = (
            "no candidate met the quantization error tolerance; selected the lowest "
            f"stable resolution within {LOW_CONFIDENCE_ERROR_MARGIN_TICKS:.2f} tick(s) "
            "of the best error"
        )

    selected_position = candidates.index(selected)
    selected_error = errors[selected]
    if selected_position == 0:
        confidence = 1.0 - min(1.0, selected_error / RESOLUTION_ERROR_TOLERANCE_TICKS)
    else:
        previous_error = errors[candidates[selected_position - 1]]
        confidence = min(
            1.0,
            max(0.0, previous_error - selected_error)
            / max(RESOLUTION_ERROR_TOLERANCE_TICKS, previous_error),
        )
    reason = (
        f"selected {selected} grids because its weighted quantization error "
        f"is {selected_error:.3f} canonical tick(s); {selection_reason}"
    )
    decision = ResolutionDecision(
        selected_resolution=selected,
        candidate_errors={str(key): round(value, 6) for key, value in errors.items()},
        evidence_count=len(samples),
        confidence=round(confidence, 6),
        reason=reason,
    )
    return _phrase_aware_plan(
        bars=bars,
        samples=samples,
        candidates=candidates,
        canonical_grids=canonical_grids,
        global_selected=selected,
        global_errors=errors,
        global_decision=decision,
    )


def _phrase_aware_plan(
    *,
    bars: list[BarFeature],
    samples: list[tuple[float, float]],
    candidates: list[int],
    canonical_grids: int,
    global_selected: int,
    global_errors: dict[int, float],
    global_decision: ResolutionDecision,
) -> ResolutionPlan:
    phrases = _structure_phrases(bars)
    if len(phrases) < 2:
        return _stable_plan(canonical_grids, global_selected, len(bars), global_decision)

    phrase_resolutions: list[int] = []
    phrase_errors: list[dict[int, float]] = []
    for start, end in phrases:
        phrase_samples = [
            (position, weight)
            for position, weight in samples
            if start * canonical_grids <= position < end * canonical_grids
        ]
        errors = {
            resolution: _weighted_quantization_error(
                phrase_samples,
                step=canonical_grids / resolution,
            )
            for resolution in candidates
        }
        phrase_errors.append(errors)
        phrase_resolutions.append(
            _minimum_resolution(errors, candidates)
            if len(phrase_samples) >= 4
            else candidates[0]
        )

    weights = {resolution: 0 for resolution in candidates}
    for (start, end), resolution in zip(phrases, phrase_resolutions, strict=True):
        weights[resolution] += end - start
    base_resolution = max(candidates, key=lambda resolution: (weights[resolution], -resolution))

    section_resolutions: dict[str, int] = {}
    for (start, _end), resolution in zip(phrases, phrase_resolutions, strict=True):
        section_id = bars[start].section_id
        if section_id:
            section_resolutions[section_id] = max(
                section_resolutions.get(section_id, base_resolution),
                resolution,
            )

    bar_resolutions = [base_resolution] * len(bars)
    upgraded = False
    for phrase_index, ((start, end), resolution, errors) in enumerate(
        zip(phrases, phrase_resolutions, phrase_errors, strict=True)
    ):
        section_id = bars[start].section_id
        if section_id:
            resolution = max(resolution, section_resolutions[section_id])
        if resolution <= base_resolution:
            continue
        if errors.get(base_resolution, 0.0) < 0.35:
            continue
        if phrase_index > 0 and bars[start - 1].boundary_confidence < 0.4:
            continue
        for position in range(start, end):
            bar_resolutions[position] = resolution
        upgraded = True

    change_points = [
        position
        for position in range(1, len(bar_resolutions))
        if bar_resolutions[position] != bar_resolutions[position - 1]
    ]
    maximum_changes = max(2, len(bars) // 8)
    if len(change_points) > maximum_changes:
        stable_resolution = max(bar_resolutions)
        decision = global_decision.model_copy(
            update={
                "selected_resolution": stable_resolution,
                "reason": (
                    "phrase-level evidence would require frequent resolution changes; "
                    f"using stable {stable_resolution}-grid output"
                ),
            }
        )
        return _stable_plan(canonical_grids, stable_resolution, len(bars), decision)
    if not upgraded:
        return _stable_plan(canonical_grids, global_selected, len(bars), global_decision)

    selected_error = global_errors.get(base_resolution, 0.0)
    decision = global_decision.model_copy(
        update={
            "selected_resolution": base_resolution,
            "confidence": round(
                max(0.0, 1.0 - selected_error / max(0.2, selected_error + 0.2)),
                6,
            ),
            "reason": (
                f"selected modal {base_resolution}-grid base and upgraded complete "
                "high-confidence phrases with off-grid onset evidence"
            ),
        }
    )
    return ResolutionPlan(
        canonical_grids_per_bar=canonical_grids,
        base_resolution=base_resolution,
        bar_resolutions=bar_resolutions,
        change_points=change_points,
        policy_version="phrase-stable-v2",
        decision=decision,
    )


def _structure_phrases(bars: list[BarFeature]) -> list[tuple[int, int]]:
    if not bars or any(bar.phrase_id is None for bar in bars):
        return []
    phrases: list[tuple[int, int]] = []
    start = 0
    for position in range(1, len(bars)):
        if bars[position].phrase_id != bars[position - 1].phrase_id:
            phrases.append((start, position))
            start = position
    phrases.append((start, len(bars)))
    if any(end - start < 2 for start, end in phrases):
        return []
    return phrases


def _minimum_resolution(errors: dict[int, float], candidates: list[int]) -> int:
    for resolution in candidates:
        if errors.get(resolution, 0.0) <= RESOLUTION_ERROR_TOLERANCE_TICKS:
            return resolution
    return candidates[-1]


def _stable_plan(
    canonical_grids: int,
    selected: int,
    bar_count: int,
    decision: ResolutionDecision,
) -> ResolutionPlan:
    return ResolutionPlan(
        canonical_grids_per_bar=canonical_grids,
        base_resolution=selected,
        bar_resolutions=[selected] * bar_count,
        change_points=[],
        policy_version="song-global-v1",
        decision=decision,
    )


def _candidate_resolutions(beats_per_bar: float) -> list[int]:
    return sorted(
        {
            round(beats_per_bar * 4),
            round(beats_per_bar * 6),
            round(beats_per_bar * 12),
        }
    )


def _weighted_onset_positions(
    raw: AudioAnalysisRaw,
    *,
    bar_count: int,
    canonical_grids: int,
) -> list[tuple[float, float]]:
    if raw.bpm <= 0 or bar_count <= 0:
        return []
    meter = get_meter_spec(raw.time_signature)
    bar_length = meter.beats_per_bar * 60.0 / raw.bpm
    if not isfinite(bar_length) or bar_length <= 0:
        return []
    analysis_end = raw.offset + (bar_count * bar_length)
    max_strength = max(raw.onset_strengths, default=0.0)
    samples: list[tuple[float, float]] = []
    for onset_time in raw.onset_times:
        if onset_time < raw.offset or onset_time >= analysis_end:
            continue
        weight = _onset_weight(raw, onset_time, max_strength)
        if weight < MIN_ONSET_WEIGHT:
            continue
        absolute_position = (onset_time - raw.offset) / bar_length * canonical_grids
        samples.append((absolute_position, weight))
    return samples


def _onset_weight(raw: AudioAnalysisRaw, onset_time: float, max_strength: float) -> float:
    if max_strength <= 0 or not raw.onset_strengths:
        return 1.0
    if raw.sample_rate is not None and raw.sample_rate > 0 and raw.hop_length > 0:
        frame_index = round(onset_time * raw.sample_rate / raw.hop_length)
    elif raw.duration > 0:
        frame_index = round(onset_time / raw.duration * (len(raw.onset_strengths) - 1))
    else:
        frame_index = 0
    frame_index = max(0, min(len(raw.onset_strengths) - 1, frame_index))
    value = raw.onset_strengths[frame_index]
    if not isfinite(value) or value <= 0:
        return 0.0
    return min(1.0, float(value) / max_strength)


def _weighted_quantization_error(
    samples: list[tuple[float, float]],
    *,
    step: float,
) -> float:
    total_weight = sum(weight for _position, weight in samples)
    if total_weight <= 0 or step <= 0:
        return 0.0
    return sum(
        abs(position - round(position / step) * step) * weight
        for position, weight in samples
    ) / total_weight
