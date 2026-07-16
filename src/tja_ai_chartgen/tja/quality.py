from bisect import bisect_left
from collections import Counter
from fractions import Fraction
from math import isfinite
from statistics import mean

from pydantic import BaseModel, Field

from tja_ai_chartgen.features.density import BarDensityHint, build_density_hints
from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.features.resolution import output_resolution_for_bar
from tja_ai_chartgen.features.salience import (
    ACTIVE_GRID_THRESHOLD,
    build_burst_salience,
    is_reliable_burst,
)
from tja_ai_chartgen.features.salience_candidates import (
    SalienceCandidate,
    build_salience_candidate_bars,
    is_salience_grid_representable,
    rank_accent_candidates,
    rank_bar_salience_candidates,
)
from tja_ai_chartgen.features.silence import edge_silence_indexes, is_silent_bar
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    ChartBar,
    ResolutionPlan,
)


NOTE_STREAM_MAX_GAP_SECONDS = 0.3
NOTE_ONSET_ALIGNMENT_TOLERANCE_SECONDS = 0.05
DOWNBEAT_RESPONSE_TOLERANCE_SECONDS = 0.07
RELIABLE_TRANSIENT_KINDS = {"strong-transient", "transient"}
STRONG_TRANSIENT_KINDS = {"strong-transient"}


class QualityReport(BaseModel):
    bar_count: int = Field(ge=0)
    density_compliant_bars: int = Field(ge=0)
    density_evaluated_bars: int = Field(ge=0)
    density_compliance_rate: float = Field(ge=0.0, le=1.0)
    silent_bar_note_count: int = Field(ge=0)
    longest_empty_bar_run: int = Field(ge=0)
    repeated_bar_count: int = Field(ge=0)
    repeated_bar_rate: float = Field(ge=0.0, le=1.0)
    don_count: int = Field(ge=0)
    ka_count: int = Field(ge=0)
    ka_ratio: float = Field(ge=0.0, le=1.0)
    longest_monochrome_run: int = Field(ge=0)
    playable_note_count: int = Field(ge=0)
    playable_duration_seconds: float = Field(ge=0.0)
    average_notes_per_second: float = Field(ge=0.0)
    peak_bar_notes_per_second: float = Field(ge=0.0)
    active_duration_seconds: float = Field(ge=0.0)
    active_average_notes_per_second: float = Field(ge=0.0)
    longest_note_stream_count: int = Field(ge=0)
    longest_note_stream_seconds: float = Field(ge=0.0)
    accent_candidate_count: int = Field(ge=0)
    accent_hit_count: int = Field(ge=0)
    accent_coverage_rate: float = Field(ge=0.0, le=1.0)
    note_onset_aligned_count: int = Field(default=0, ge=0)
    note_onset_evaluated_count: int = Field(default=0, ge=0)
    note_onset_alignment: float = Field(default=1.0, ge=0.0, le=1.0)
    strong_onset_responded_count: int = Field(default=0, ge=0)
    strong_onset_evaluated_count: int = Field(default=0, ge=0)
    strong_onset_response: float = Field(default=1.0, ge=0.0, le=1.0)
    downbeat_responded_count: int = Field(default=0, ge=0)
    downbeat_evaluated_count: int = Field(default=0, ge=0)
    downbeat_response: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_note_count: int = Field(default=0, ge=0)
    unsupported_note_evaluated_count: int = Field(default=0, ge=0)
    unsupported_note_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    silent_range_evaluated_bar_count: int = Field(default=0, ge=0)
    silent_range_violation_count: int = Field(default=0, ge=0)
    silent_range_violation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    fill_burst_aligned_count: int = Field(default=0, ge=0)
    fill_burst_evaluated_count: int = Field(default=0, ge=0)
    fill_burst_alignment: float = Field(default=1.0, ge=0.0, le=1.0)
    rhythmic_quantization_evaluated_count: int = Field(default=0, ge=0)
    rhythmic_quantization_error: float | None = Field(default=None, ge=0.0)
    drumroll_count: int = Field(ge=0)
    balloon_count: int = Field(ge=0)
    special_note_count: int = Field(ge=0)
    drumroll_duration_seconds: float = Field(ge=0.0)
    balloon_duration_seconds: float = Field(ge=0.0)
    special_note_duration_seconds: float = Field(ge=0.0)
    balloon_required_hits: int = Field(ge=0)
    balloon_hits_per_second: float = Field(ge=0.0)
    structure_density_correlation: float = Field(default=0.0, ge=-1.0, le=1.0)
    peak_contrast: float = 0.0
    build_up_slope_agreement: float = Field(default=0.0, ge=-1.0, le=1.0)
    cadence_variation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    fill_candidate_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    section_motif_consistency: float = Field(default=1.0, ge=0.0, le=1.0)
    section_return_variation: float = Field(default=0.0, ge=0.0, le=1.0)
    highlight_note_contrast: float = 0.0
    drum_onset_hit_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    bass_downbeat_alignment: float = Field(default=1.0, ge=0.0, le=1.0)
    vocal_phrase_response: float = Field(default=1.0, ge=0.0, le=1.0)
    instrument_transition_response: float = Field(default=1.0, ge=0.0, le=1.0)
    instrument_confident_bar_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    instrument_fill_support: float = Field(default=1.0, ge=0.0, le=1.0)
    base_resolution: int = Field(default=0, ge=0)
    resolution_change_count: int = Field(default=0, ge=0)
    resolution_changes_per_100_bars: float = Field(default=0.0, ge=0.0)
    high_resolution_bar_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    resolution_quantization_error: float | None = Field(default=None, ge=0.0)
    avoided_resolution_changes: int = Field(default=0, ge=0)
    cross_course_resolution_consistency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


def build_quality_report(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    resolution_plan: ResolutionPlan | None = None,
) -> QualityReport:
    paired_count = min(len(chart_bars), len(feature_bars))
    paired_chart_bars = chart_bars[:paired_count]
    paired_feature_bars = feature_bars[:paired_count]
    density_hints = build_density_hints(paired_feature_bars)
    hit_counts = [density_hit_count(bar.notes) for bar in chart_bars]
    activity_counts = [chart_activity_count(bar.notes) for bar in chart_bars]
    density_compliant = sum(
        density_hint_is_satisfied(hit_count, hint)
        for hit_count, hint in zip(hit_counts, density_hints, strict=False)
    )
    density_evaluated = len(density_hints)

    silent_indexes = edge_silence_indexes(paired_feature_bars)
    silent_bar_note_count = sum(
        chart_activity_count(paired_chart_bars[index].notes)
        for index in silent_indexes
        if index < len(paired_chart_bars)
    )
    silent_range_indexes = silent_indexes | {
        position
        for position, feature_bar in enumerate(paired_feature_bars)
        if is_silent_bar(feature_bar)
    }
    silent_range_violation_count = sum(
        chart_activity_count(paired_chart_bars[index].notes)
        for index in silent_range_indexes
        if index < len(paired_chart_bars)
    )
    paired_activity_count = sum(
        chart_activity_count(chart_bar.notes) for chart_bar in paired_chart_bars
    )

    nonempty_pattern_counts = pattern_counts(chart_bars)
    nonempty_pattern_count = sum(nonempty_pattern_counts.values())
    repeated_bar_count = sum(count - 1 for count in nonempty_pattern_counts.values())
    don_count, ka_count, longest_monochrome_run = note_color_metrics(chart_bars)
    normal_note_count = don_count + ka_count
    timed_hit_counts: list[int] = []
    timed_durations: list[float] = []
    active_durations: list[float] = []
    note_times: list[float] = []
    paired_notes_per_second = [0.0] * paired_count
    accent_candidate_count = 0
    accent_hit_count = 0
    salience_bars = build_burst_salience(paired_feature_bars)
    canonical_candidate_bars = [
        rank_bar_salience_candidates(
            feature_bar,
            salience,
            output_resolution=feature_bar.grids_per_bar,
        )
        for feature_bar, salience in zip(
            paired_feature_bars,
            salience_bars,
            strict=True,
        )
    ]
    salience_candidate_bars = [
        [
            candidate
            for candidate in candidates
            if is_salience_grid_representable(
                feature_bar,
                candidate.grid,
                output_resolution=output_resolution_for_bar(
                    feature_bar,
                    plan=resolution_plan,
                    position=position,
                ),
            )
        ]
        for position, (feature_bar, candidates) in enumerate(
            zip(paired_feature_bars, canonical_candidate_bars, strict=True)
        )
    ]
    for position, (chart_bar, feature_bar) in enumerate(
        zip(paired_chart_bars, paired_feature_bars, strict=True)
    ):
        duration = feature_bar.end_time - feature_bar.start_time
        if not isfinite(duration) or duration <= 0 or not chart_bar.notes:
            continue
        hit_count = playable_hit_count(chart_bar.notes)
        paired_notes_per_second[position] = hit_count / duration
        timed_hit_counts.append(hit_count)
        timed_durations.append(duration)
        if hit_count:
            active_durations.append(duration)
        note_times.extend(normal_note_times(chart_bar, feature_bar))
        candidates, hits = accent_coverage_counts(
            chart_bar,
            feature_bar,
            salience_candidates=salience_candidate_bars[position],
        )
        accent_candidate_count += candidates
        accent_hit_count += hits
    playable_note_count = sum(timed_hit_counts)
    playable_duration_seconds = sum(timed_durations)
    active_duration_seconds = sum(active_durations)
    reliable_onset_times = _candidate_times(
        paired_feature_bars,
        canonical_candidate_bars,
        kinds=RELIABLE_TRANSIENT_KINDS,
    )
    note_onset_aligned_count = _nearby_time_count(
        note_times,
        reliable_onset_times,
        tolerance_seconds=NOTE_ONSET_ALIGNMENT_TOLERANCE_SECONDS,
    )
    strong_onset_times = _candidate_times(
        paired_feature_bars,
        canonical_candidate_bars,
        kinds=STRONG_TRANSIENT_KINDS,
    )
    special_note_ranges = _special_note_time_ranges(
        paired_chart_bars,
        paired_feature_bars,
    )
    special_supported_strong_onsets, remaining_strong_onsets = (
        _partition_times_by_ranges(strong_onset_times, special_note_ranges)
    )
    strong_onset_responded_count = len(special_supported_strong_onsets)
    strong_onset_responded_count += _matched_reference_count(
        remaining_strong_onsets,
        note_times,
        tolerance_seconds=NOTE_ONSET_ALIGNMENT_TOLERANCE_SECONDS,
    )
    downbeat_times = _candidate_times(
        paired_feature_bars,
        canonical_candidate_bars,
        required_reason="downbeat",
        require_reliable=True,
    )
    downbeat_responded_count = _matched_reference_count(
        downbeat_times,
        note_times,
        tolerance_seconds=DOWNBEAT_RESPONSE_TOLERANCE_SECONDS,
    )
    unsupported_note_count, unsupported_note_evaluated_count = (
        _unsupported_note_counts(
            paired_chart_bars,
            paired_feature_bars,
            canonical_candidate_bars,
            reliable_onset_times=reliable_onset_times,
        )
    )
    fill_burst_aligned_count, fill_burst_evaluated_count = (
        _fill_burst_alignment_counts(
            paired_chart_bars,
            paired_feature_bars,
            salience_bars,
            special_note_ranges=special_note_ranges,
        )
    )
    (
        rhythmic_quantization_evaluated_count,
        rhythmic_quantization_error,
    ) = _rhythmic_quantization_metrics(
        paired_chart_bars,
        paired_feature_bars,
        canonical_candidate_bars,
    )
    bar_notes_per_second = [
        hit_count / duration
        for hit_count, duration in zip(timed_hit_counts, timed_durations, strict=True)
    ]
    longest_stream_count, longest_stream_seconds = longest_note_stream(note_times)
    (
        drumroll_count,
        balloon_count,
        drumroll_duration,
        balloon_duration,
        balloon_required_hits,
    ) = special_note_metrics(chart_bars, feature_bars)
    special_note_count = drumroll_count + balloon_count
    special_note_duration = drumroll_duration + balloon_duration
    structure_metrics = _structure_quality_metrics(
        paired_chart_bars,
        paired_feature_bars,
        paired_notes_per_second,
    )
    instrument_metrics = _instrument_quality_metrics(
        paired_chart_bars,
        paired_feature_bars,
    )
    resolution_metrics = _resolution_quality_metrics(
        chart_bars,
        feature_bars,
        resolution_plan,
    )

    return QualityReport(
        bar_count=len(chart_bars),
        density_compliant_bars=density_compliant,
        density_evaluated_bars=density_evaluated,
        density_compliance_rate=(
            density_compliant / density_evaluated if density_evaluated else 1.0
        ),
        silent_bar_note_count=silent_bar_note_count,
        longest_empty_bar_run=longest_empty_bar_run(activity_counts),
        repeated_bar_count=repeated_bar_count,
        repeated_bar_rate=(
            repeated_bar_count / nonempty_pattern_count if nonempty_pattern_count else 0.0
        ),
        don_count=don_count,
        ka_count=ka_count,
        ka_ratio=ka_count / normal_note_count if normal_note_count else 0.0,
        longest_monochrome_run=longest_monochrome_run,
        playable_note_count=playable_note_count,
        playable_duration_seconds=playable_duration_seconds,
        average_notes_per_second=(
            playable_note_count / playable_duration_seconds
            if playable_duration_seconds
            else 0.0
        ),
        peak_bar_notes_per_second=max(bar_notes_per_second, default=0.0),
        active_duration_seconds=active_duration_seconds,
        active_average_notes_per_second=(
            playable_note_count / active_duration_seconds
            if active_duration_seconds
            else 0.0
        ),
        longest_note_stream_count=longest_stream_count,
        longest_note_stream_seconds=longest_stream_seconds,
        accent_candidate_count=accent_candidate_count,
        accent_hit_count=accent_hit_count,
        accent_coverage_rate=(
            accent_hit_count / accent_candidate_count
            if accent_candidate_count
            else 1.0
        ),
        note_onset_aligned_count=note_onset_aligned_count,
        note_onset_evaluated_count=len(note_times),
        note_onset_alignment=_rounded_metric(
            note_onset_aligned_count / len(note_times) if note_times else 1.0
        ),
        strong_onset_responded_count=strong_onset_responded_count,
        strong_onset_evaluated_count=len(strong_onset_times),
        strong_onset_response=_rounded_metric(
            strong_onset_responded_count / len(strong_onset_times)
            if strong_onset_times
            else 1.0
        ),
        downbeat_responded_count=downbeat_responded_count,
        downbeat_evaluated_count=len(downbeat_times),
        downbeat_response=_rounded_metric(
            downbeat_responded_count / len(downbeat_times) if downbeat_times else 1.0
        ),
        unsupported_note_count=unsupported_note_count,
        unsupported_note_evaluated_count=unsupported_note_evaluated_count,
        unsupported_note_rate=_rounded_metric(
            unsupported_note_count / unsupported_note_evaluated_count
            if unsupported_note_evaluated_count
            else 0.0
        ),
        silent_range_evaluated_bar_count=len(silent_range_indexes),
        silent_range_violation_count=silent_range_violation_count,
        silent_range_violation_rate=_rounded_metric(
            silent_range_violation_count / paired_activity_count
            if paired_activity_count
            else 0.0
        ),
        fill_burst_aligned_count=fill_burst_aligned_count,
        fill_burst_evaluated_count=fill_burst_evaluated_count,
        fill_burst_alignment=_rounded_metric(
            fill_burst_aligned_count / fill_burst_evaluated_count
            if fill_burst_evaluated_count
            else 1.0
        ),
        rhythmic_quantization_evaluated_count=rhythmic_quantization_evaluated_count,
        rhythmic_quantization_error=rhythmic_quantization_error,
        drumroll_count=drumroll_count,
        balloon_count=balloon_count,
        special_note_count=special_note_count,
        drumroll_duration_seconds=drumroll_duration,
        balloon_duration_seconds=balloon_duration,
        special_note_duration_seconds=special_note_duration,
        balloon_required_hits=balloon_required_hits,
        balloon_hits_per_second=(
            balloon_required_hits / balloon_duration if balloon_duration else 0.0
        ),
        **structure_metrics,
        **instrument_metrics,
        **resolution_metrics,
    )


def _structure_quality_metrics(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    notes_per_second: list[float],
) -> dict[str, float]:
    if not chart_bars or not feature_bars:
        return {
            "structure_density_correlation": 0.0,
            "peak_contrast": 0.0,
            "build_up_slope_agreement": 0.0,
            "cadence_variation_rate": 0.0,
            "fill_candidate_precision": 1.0,
            "section_motif_consistency": 1.0,
            "section_return_variation": 0.0,
            "highlight_note_contrast": 0.0,
        }

    has_structure = any(
        bar.phrase_id is not None
        or bar.transition_role != "stable"
        or bar.energy_percentile > 0
        for bar in feature_bars
    )
    energy_values = [
        bar.energy_percentile if has_structure else bar.energy for bar in feature_bars
    ]
    structure_density_correlation = _correlation(energy_values, notes_per_second)

    peak_values = [
        value
        for value, bar in zip(notes_per_second, feature_bars, strict=True)
        if bar.transition_role == "peak"
    ]
    baseline_values = [
        value
        for value, bar in zip(notes_per_second, feature_bars, strict=True)
        if bar.transition_role in {"stable", "breakdown"}
    ]
    peak_contrast = (
        _average(peak_values) - _average(baseline_values) if peak_values else 0.0
    )

    phrase_positions: dict[int, list[int]] = {}
    for position, bar in enumerate(feature_bars):
        if bar.phrase_id is not None:
            phrase_positions.setdefault(bar.phrase_id, []).append(position)
    build_up_agreements: list[float] = []
    for positions in phrase_positions.values():
        if not any(feature_bars[position].transition_role == "build_up" for position in positions):
            continue
        progress = [feature_bars[position].phrase_progress for position in positions]
        values = [notes_per_second[position] for position in positions]
        if len(positions) >= 2:
            build_up_agreements.append(_correlation(progress, values))

    cadence_positions = [
        position
        for position, bar in enumerate(feature_bars)
        if bar.transition_role == "cadence"
    ]
    cadence_variations = [
        _normalized_pattern_signature(chart_bars[position].notes)
        != _normalized_pattern_signature(chart_bars[position - 1].notes)
        for position in cadence_positions
        if position > 0
    ]

    actual_fill_positions = _actual_fill_positions(chart_bars)
    candidate_positions = {
        position for position, bar in enumerate(feature_bars) if bar.fill_candidate
    }
    fill_candidate_precision = (
        len(actual_fill_positions & candidate_positions) / len(actual_fill_positions)
        if actual_fill_positions
        else 1.0
    )

    section_motif_consistency = _section_motif_consistency(chart_bars, feature_bars)
    section_return_variation = _section_return_variation(chart_bars, feature_bars)
    highlight_values = [
        value
        for value, bar in zip(notes_per_second, feature_bars, strict=True)
        if bar.transition_role in {"peak", "drop", "cadence"} or bar.fill_candidate
    ]
    non_highlight_values = [
        value
        for value, bar in zip(notes_per_second, feature_bars, strict=True)
        if bar.transition_role not in {"peak", "drop", "cadence"}
        and not bar.fill_candidate
    ]
    highlight_note_contrast = (
        _average(highlight_values) - _average(non_highlight_values)
        if highlight_values
        else 0.0
    )

    return {
        "structure_density_correlation": _rounded_metric(
            structure_density_correlation
        ),
        "peak_contrast": _rounded_metric(peak_contrast),
        "build_up_slope_agreement": _rounded_metric(
            _average(build_up_agreements) if build_up_agreements else 0.0
        ),
        "cadence_variation_rate": _rounded_metric(
            sum(cadence_variations) / len(cadence_variations)
            if cadence_variations
            else 0.0
        ),
        "fill_candidate_precision": _rounded_metric(fill_candidate_precision),
        "section_motif_consistency": _rounded_metric(section_motif_consistency),
        "section_return_variation": _rounded_metric(section_return_variation),
        "highlight_note_contrast": _rounded_metric(highlight_note_contrast),
    }


def _actual_fill_positions(chart_bars: list[ChartBar]) -> set[int]:
    hit_counts = [density_hit_count(bar.notes) for bar in chart_bars]
    positions: set[int] = set()
    for position, bar in enumerate(chart_bars):
        if any(note in "57" for note in bar.notes):
            positions.add(position)
            continue
        neighbors = [
            hit_counts[index]
            for index in (position - 1, position + 1)
            if 0 <= index < len(hit_counts)
        ]
        if neighbors and hit_counts[position] >= _average(neighbors) + 2:
            positions.add(position)
    return positions


def _section_motif_consistency(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
) -> float:
    sections: dict[str, list[str]] = {}
    for chart_bar, feature_bar in zip(chart_bars, feature_bars, strict=True):
        if not feature_bar.section_id:
            continue
        sections.setdefault(feature_bar.section_id, []).append(
            _rhythm_signature(chart_bar.notes)
        )
    repeated_sections = [patterns for patterns in sections.values() if len(patterns) >= 2]
    if not repeated_sections:
        return 1.0
    consistent = sum(max(Counter(patterns).values()) for patterns in repeated_sections)
    total = sum(len(patterns) for patterns in repeated_sections)
    return consistent / total if total else 1.0


def _section_return_variation(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
) -> float:
    sections: dict[str, dict[int, list[str]]] = {}
    for chart_bar, feature_bar in zip(chart_bars, feature_bars, strict=True):
        if not feature_bar.section_id or feature_bar.phrase_id is None:
            continue
        sections.setdefault(feature_bar.section_id, {}).setdefault(
            feature_bar.phrase_id,
            [],
        ).append(_rhythm_signature(chart_bar.notes))
    comparisons = 0
    variations = 0
    for phrases in sections.values():
        ordered = [tuple(patterns) for _phrase_id, patterns in sorted(phrases.items())]
        if len(ordered) < 2:
            continue
        reference = ordered[0]
        for returned in ordered[1:]:
            comparisons += 1
            variations += returned != reference
    return variations / comparisons if comparisons else 0.0


def _rhythm_signature(notes: str) -> str:
    if not notes:
        return ""
    return "|".join(
        str(Fraction(index, len(notes)))
        for index, note in enumerate(notes)
        if note in "1234"
    )


def _instrument_quality_metrics(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
) -> dict[str, float]:
    drum_candidates = 0
    drum_hits = 0
    bass_candidates = 0
    bass_hits = 0
    vocal_candidates = 0
    vocal_hits = 0
    evidence_bars = 0
    confident_bars = 0
    transition_candidates = 0
    transition_responses = 0
    fill_candidates = 0
    fill_supported = 0

    for position, (chart_bar, feature_bar) in enumerate(
        zip(chart_bars, feature_bars, strict=True)
    ):
        source_values = (
            feature_bar.instrument.vocal_activity,
            feature_bar.instrument.drum_activity,
            feature_bar.instrument.bass_activity,
            feature_bar.instrument.other_activity,
        )
        if max(source_values) > 0.0:
            evidence_bars += 1
            if feature_bar.instrument.confidence >= 0.4:
                confident_bars += 1

        for event in feature_bar.instrument_grid_features:
            if event.drum_onset >= 0.35:
                drum_candidates += 1
                drum_hits += _normal_hit_at(chart_bar, feature_bar, event.grid)
            if event.bass_onset >= 0.35 and event.grid in {
                *feature_bar.beat_grids,
                feature_bar.downbeat_grid,
            }:
                bass_candidates += 1
                bass_hits += _normal_hit_at(chart_bar, feature_bar, event.grid)
            if event.vocal_onset >= 0.35 and (
                feature_bar.phrase_position
                in {"phrase_start", "phrase_end", "song_end"}
                or feature_bar.transition_role == "cadence"
            ):
                vocal_candidates += 1
                vocal_hits += _normal_hit_at(chart_bar, feature_bar, event.grid)

        if position > 0:
            previous_feature = feature_bars[position - 1]
            previous_chart = chart_bars[position - 1]
            current_key = (
                feature_bar.instrument.dominant_source,
                feature_bar.instrument.dominant_instrument,
            )
            previous_key = (
                previous_feature.instrument.dominant_source,
                previous_feature.instrument.dominant_instrument,
            )
            if (
                feature_bar.instrument.confidence >= 0.4
                and previous_feature.instrument.confidence >= 0.4
                and current_key != previous_key
            ):
                transition_candidates += 1
                transition_responses += (
                    _rhythm_signature(chart_bar.notes)
                    != _rhythm_signature(previous_chart.notes)
                )

        fill_evidence = max(
            (
                max(event.drum_onset, event.accompaniment_onset)
                for event in feature_bar.instrument_grid_features
            ),
            default=0.0,
        )
        if feature_bar.fill_candidate and fill_evidence >= 0.35:
            fill_candidates += 1
            midpoint = len(chart_bar.notes) // 2
            fill_supported += any(note in "123457" for note in chart_bar.notes[midpoint:])

    return {
        "drum_onset_hit_coverage": _rounded_metric(
            drum_hits / drum_candidates if drum_candidates else 1.0
        ),
        "bass_downbeat_alignment": _rounded_metric(
            bass_hits / bass_candidates if bass_candidates else 1.0
        ),
        "vocal_phrase_response": _rounded_metric(
            vocal_hits / vocal_candidates if vocal_candidates else 1.0
        ),
        "instrument_transition_response": _rounded_metric(
            transition_responses / transition_candidates if transition_candidates else 1.0
        ),
        "instrument_confident_bar_ratio": _rounded_metric(
            confident_bars / evidence_bars if evidence_bars else 0.0
        ),
        "instrument_fill_support": _rounded_metric(
            fill_supported / fill_candidates if fill_candidates else 1.0
        ),
    }


def _normal_hit_at(
    chart_bar: ChartBar,
    feature_bar: BarFeature,
    canonical_grid: int,
) -> int:
    if not chart_bar.notes or feature_bar.grids_per_bar <= 0:
        return 0
    projected = min(
        len(chart_bar.notes) - 1,
        round(canonical_grid / feature_bar.grids_per_bar * len(chart_bar.notes)),
    )
    return int(chart_bar.notes[projected] in "1234")


def _resolution_quality_metrics(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    resolution_plan: ResolutionPlan | None,
) -> dict[str, int | float | None]:
    if resolution_plan is not None and resolution_plan.bar_resolutions:
        resolutions = []
        for position, chart_bar in enumerate(chart_bars):
            feature_index = (
                feature_bars[position].index
                if position < len(feature_bars)
                else chart_bar.index
            )
            if 0 <= feature_index < len(resolution_plan.bar_resolutions):
                resolutions.append(resolution_plan.bar_resolutions[feature_index])
            elif position < len(resolution_plan.bar_resolutions):
                resolutions.append(resolution_plan.bar_resolutions[position])
            else:
                resolutions.append(len(chart_bar.notes))
        base_resolution = resolution_plan.base_resolution
    else:
        resolutions = [len(bar.notes) for bar in chart_bars]
        base_resolution = (
            Counter(resolutions).most_common(1)[0][0] if resolutions else 0
        )
    resolution_change_count = sum(
        previous != current
        for previous, current in zip(resolutions[:-1], resolutions[1:], strict=True)
    )
    high_resolution_count = 0
    for position, resolution in enumerate(resolutions):
        if position < len(feature_bars):
            base = get_meter_spec(feature_bars[position].time_signature).legacy_grids_per_bar
        else:
            base = 16
        high_resolution_count += resolution > base
    quantization_error: float | None = None
    if resolution_plan is not None and resolution_plan.decision is not None:
        quantization_error = resolution_plan.decision.candidate_errors.get(
            str(resolution_plan.base_resolution)
        )
    avoided_changes = sum(
        feature_bars[position - 1].phrase_id != feature_bars[position].phrase_id
        and resolutions[position - 1] == resolutions[position]
        for position in range(1, min(len(feature_bars), len(resolutions)))
    )
    return {
        "base_resolution": base_resolution,
        "resolution_change_count": resolution_change_count,
        "resolution_changes_per_100_bars": _rounded_metric(
            resolution_change_count / len(resolutions) * 100 if resolutions else 0.0
        ),
        "high_resolution_bar_ratio": _rounded_metric(
            high_resolution_count / len(resolutions) if resolutions else 0.0
        ),
        "resolution_quantization_error": quantization_error,
        "avoided_resolution_changes": avoided_changes,
        "cross_course_resolution_consistency": None,
    }


def _correlation(first: list[float], second: list[float]) -> float:
    count = min(len(first), len(second))
    if count < 2:
        return 0.0
    left = first[:count]
    right = second[:count]
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left) ** 0.5
    right_scale = sum((value - right_mean) ** 2 for value in right) ** 0.5
    if left_scale <= 0 or right_scale <= 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_scale * right_scale)))


def _average(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rounded_metric(value: float) -> float:
    return round(float(value), 6)


def _candidate_times(
    feature_bars: list[BarFeature],
    candidate_bars: list[list[SalienceCandidate]],
    *,
    kinds: set[str] | None = None,
    required_reason: str | None = None,
    require_reliable: bool = False,
) -> list[float]:
    """按 salience 语义筛选 canonical 候选并转换为全曲时间。"""
    times: list[float] = []
    for feature_bar, candidates in zip(feature_bars, candidate_bars, strict=True):
        duration = feature_bar.end_time - feature_bar.start_time
        if (
            not isfinite(duration)
            or duration <= 0
            or feature_bar.grids_per_bar <= 0
        ):
            continue
        times.extend(
            feature_bar.start_time
            + duration * candidate.grid / feature_bar.grids_per_bar
            for candidate in candidates
            if (kinds is None or candidate.kind in kinds)
            and (required_reason is None or required_reason in candidate.point.reasons)
            and (not require_reliable or candidate.reliable)
        )
    return sorted(times)


def _fill_burst_alignment_counts(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    salience_bars: list[BarRhythmicSalience],
    *,
    special_note_ranges: list[tuple[float, float]],
) -> tuple[int, int]:
    """统计特殊音符区间和普通 fill 小节对可靠 burst/fill 的响应。"""
    reliable_burst_ranges = _reliable_burst_time_ranges(feature_bars, salience_bars)
    aligned_special = sum(
        any(_ranges_overlap(special_range, burst_range) for burst_range in reliable_burst_ranges)
        for special_range in special_note_ranges
    )

    special_note_positions = {
        position
        for position, (chart_bar, feature_bar) in enumerate(
            zip(chart_bars, feature_bars, strict=True)
        )
        if any(note in {"5", "7", "8"} for note in chart_bar.notes)
        or any(
            _ranges_overlap(
                (feature_bar.start_time, feature_bar.end_time),
                special_range,
            )
            for special_range in special_note_ranges
        )
    }
    ordinary_fill_positions = _actual_fill_positions(chart_bars) - special_note_positions
    aligned_ordinary = 0
    for position in ordinary_fill_positions:
        if position >= len(feature_bars) or position >= len(salience_bars):
            continue
        feature_bar = feature_bars[position]
        salience = salience_bars[position]
        if feature_bar.fill_candidate:
            aligned_ordinary += 1
            continue
        if not is_reliable_burst(salience):
            continue
        start = salience.burst_start_grid
        end = salience.burst_end_grid
        if start is None or end is None or not chart_bars[position].notes:
            continue
        if any(
            note in "1234"
            and start
            <= round(grid / len(chart_bars[position].notes) * feature_bar.grids_per_bar)
            <= end
            for grid, note in enumerate(chart_bars[position].notes)
        ):
            aligned_ordinary += 1

    evaluated_count = len(special_note_ranges) + len(ordinary_fill_positions)
    return aligned_special + aligned_ordinary, evaluated_count


def _reliable_burst_time_ranges(
    feature_bars: list[BarFeature],
    salience_bars: list[BarRhythmicSalience],
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for feature_bar, salience in zip(feature_bars, salience_bars, strict=True):
        duration = feature_bar.end_time - feature_bar.start_time
        if (
            not isfinite(duration)
            or duration <= 0
            or feature_bar.grids_per_bar <= 0
            or not is_reliable_burst(salience)
        ):
            continue
        start = salience.burst_start_grid
        end = salience.burst_end_grid
        if start is None or end is None or end <= start:
            continue
        ranges.append(
            (
                feature_bar.start_time + duration * start / feature_bar.grids_per_bar,
                feature_bar.start_time + duration * end / feature_bar.grids_per_bar,
            )
        )
    return ranges


def _ranges_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def _rhythmic_quantization_metrics(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    candidate_bars: list[list[SalienceCandidate]],
) -> tuple[int, float | None]:
    """计算普通 note 对最近可靠 salience 的平均 canonical tick 误差。"""
    candidate_events: list[tuple[float, float]] = []
    for feature_bar, candidates in zip(feature_bars, candidate_bars, strict=True):
        duration = feature_bar.end_time - feature_bar.start_time
        if (
            not isfinite(duration)
            or duration <= 0
            or feature_bar.grids_per_bar <= 0
        ):
            continue
        tick_seconds = duration / feature_bar.grids_per_bar
        candidate_events.extend(
            (
                feature_bar.start_time + tick_seconds * candidate.grid,
                tick_seconds,
            )
            for candidate in candidates
            if candidate.reliable
        )
    candidate_events.sort(key=lambda item: item[0])
    candidate_times = [time for time, _tick_seconds in candidate_events]
    if not candidate_events:
        return 0, None

    errors: list[float] = []
    for chart_bar, feature_bar in zip(chart_bars, feature_bars, strict=True):
        duration = feature_bar.end_time - feature_bar.start_time
        if not isfinite(duration) or duration <= 0 or not chart_bar.notes:
            continue
        for grid, note in enumerate(chart_bar.notes):
            if note not in "1234":
                continue
            note_time = feature_bar.start_time + duration * grid / len(chart_bar.notes)
            position = bisect_left(candidate_times, note_time)
            nearby = [
                candidate_events[index]
                for index in (position - 1, position)
                if 0 <= index < len(candidate_events)
            ]
            if not nearby:
                continue
            candidate_time, tick_seconds = min(
                nearby,
                key=lambda item: abs(item[0] - note_time),
            )
            error_seconds = abs(candidate_time - note_time)
            if error_seconds > NOTE_ONSET_ALIGNMENT_TOLERANCE_SECONDS:
                continue
            errors.append(error_seconds / tick_seconds)
    if not errors:
        return 0, None
    return len(errors), _rounded_metric(sum(errors) / len(errors))


def _unsupported_note_counts(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
    candidate_bars: list[list[SalienceCandidate]],
    *,
    reliable_onset_times: list[float],
) -> tuple[int, int]:
    """统计缺少直接节奏证据且不能作为短连接点的普通 note。"""
    note_events: list[tuple[float, int, int]] = []
    for position, (chart_bar, feature_bar) in enumerate(
        zip(chart_bars, feature_bars, strict=True)
    ):
        duration = feature_bar.end_time - feature_bar.start_time
        if not isfinite(duration) or duration <= 0 or not chart_bar.notes:
            continue
        note_events.extend(
            (
                feature_bar.start_time + duration * grid / len(chart_bar.notes),
                position,
                grid,
            )
            for grid, note in enumerate(chart_bar.notes)
            if note in "1234"
        )
    if not note_events:
        return 0, 0

    beat_times = _beat_times(feature_bars)
    structure_times = _candidate_times(
        feature_bars,
        candidate_bars,
        kinds={"structure-highlight"},
    )
    direct_support: list[bool] = []
    for time, position, grid in note_events:
        direct_support.append(
            _has_nearby_time(
                time,
                reliable_onset_times,
                NOTE_ONSET_ALIGNMENT_TOLERANCE_SECONDS,
            )
            or _has_nearby_time(
                time,
                beat_times,
                DOWNBEAT_RESPONSE_TOLERANCE_SECONDS,
            )
            or _has_nearby_time(
                time,
                structure_times,
                DOWNBEAT_RESPONSE_TOLERANCE_SECONDS,
            )
            or _note_has_sustained_activity(
                chart_bars[position],
                feature_bars[position],
                candidate_bars[position],
                grid,
            )
        )

    directly_supported_times = sorted(
        time
        for (time, _position, _grid), supported in zip(
            note_events,
            direct_support,
            strict=True,
        )
        if supported
    )
    supported_count = sum(direct_support)
    supported_count += sum(
        _has_nearby_time(time, directly_supported_times, NOTE_STREAM_MAX_GAP_SECONDS)
        for (time, _position, _grid), supported in zip(
            note_events,
            direct_support,
            strict=True,
        )
        if not supported
    )
    return len(note_events) - supported_count, len(note_events)


def _beat_times(feature_bars: list[BarFeature]) -> list[float]:
    times: list[float] = []
    for feature_bar in feature_bars:
        duration = feature_bar.end_time - feature_bar.start_time
        if (
            not isfinite(duration)
            or duration <= 0
            or feature_bar.grids_per_bar <= 0
        ):
            continue
        grids = set(feature_bar.beat_grids)
        if feature_bar.downbeat_grid is not None:
            grids.add(feature_bar.downbeat_grid)
        times.extend(
            feature_bar.start_time + duration * grid / feature_bar.grids_per_bar
            for grid in grids
            if 0 <= grid < feature_bar.grids_per_bar
        )
    return sorted(times)


def _note_has_sustained_activity(
    chart_bar: ChartBar,
    feature_bar: BarFeature,
    candidates: list[SalienceCandidate],
    note_grid: int,
) -> bool:
    if not chart_bar.notes:
        return False
    position = note_grid / len(chart_bar.notes)
    if feature_bar.activity_grids:
        activity_grid = min(
            len(feature_bar.activity_grids) - 1,
            round(position * len(feature_bar.activity_grids)),
        )
        if feature_bar.activity_grids[activity_grid] >= ACTIVE_GRID_THRESHOLD:
            return True

    canonical_grid = min(
        feature_bar.grids_per_bar - 1,
        round(position * feature_bar.grids_per_bar),
    )
    return any(
        candidate.grid == canonical_grid
        and candidate.point.sustained_activity >= ACTIVE_GRID_THRESHOLD
        for candidate in candidates
    )


def _has_nearby_time(time: float, candidates: list[float], tolerance_seconds: float) -> bool:
    if not candidates or tolerance_seconds < 0:
        return False
    position = bisect_left(candidates, time)
    return (
        position < len(candidates)
        and abs(candidates[position] - time) <= tolerance_seconds
    ) or (
        position > 0
        and abs(candidates[position - 1] - time) <= tolerance_seconds
    )


def _special_note_time_ranges(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
) -> list[tuple[float, float]]:
    """提取由 `5`/`7` 开始并由后续 `8` 闭合的有效持续区间。"""
    ranges: list[tuple[float, float]] = []
    active_start: float | None = None
    for chart_bar, feature_bar in zip(chart_bars, feature_bars, strict=True):
        duration = feature_bar.end_time - feature_bar.start_time
        if not isfinite(duration) or duration <= 0 or not chart_bar.notes:
            active_start = None
            continue
        for grid, note in enumerate(chart_bar.notes):
            time = feature_bar.start_time + duration * grid / len(chart_bar.notes)
            if note in {"5", "7"}:
                active_start = time if active_start is None else None
            elif note == "8" and active_start is not None:
                if time > active_start:
                    ranges.append((active_start, time))
                active_start = None
    return ranges


def _partition_times_by_ranges(
    times: list[float],
    ranges: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """线性划分持续区间内外的有序事件时间。"""
    inside: list[float] = []
    outside: list[float] = []
    range_index = 0
    for time in sorted(times):
        while range_index < len(ranges) and ranges[range_index][1] < time:
            range_index += 1
        if (
            range_index < len(ranges)
            and ranges[range_index][0] <= time <= ranges[range_index][1]
        ):
            inside.append(time)
        else:
            outside.append(time)
    return inside, outside


def _matched_reference_count(
    references: list[float],
    estimates: list[float],
    *,
    tolerance_seconds: float,
) -> int:
    """在时间容差内进行确定性一对一匹配，返回被响应的参考事件数。"""
    if not references or not estimates or tolerance_seconds < 0:
        return 0
    ordered_references = sorted(references)
    ordered_estimates = sorted(estimates)
    reference_index = 0
    estimate_index = 0
    matched = 0
    while (
        reference_index < len(ordered_references)
        and estimate_index < len(ordered_estimates)
    ):
        reference = ordered_references[reference_index]
        estimate = ordered_estimates[estimate_index]
        if estimate < reference - tolerance_seconds:
            estimate_index += 1
        elif estimate > reference + tolerance_seconds:
            reference_index += 1
        else:
            matched += 1
            reference_index += 1
            estimate_index += 1
    return matched


def _nearby_time_count(
    values: list[float],
    candidates: list[float],
    *,
    tolerance_seconds: float,
) -> int:
    """统计有邻近候选的时间点；同一瞬态可支持多个普通 note。"""
    if not values or not candidates or tolerance_seconds < 0:
        return 0
    ordered_values = sorted(values)
    candidate_index = 0
    matched = 0
    for value in ordered_values:
        while (
            candidate_index < len(candidates)
            and candidates[candidate_index] < value - tolerance_seconds
        ):
            candidate_index += 1
        if (
            candidate_index < len(candidates)
            and candidates[candidate_index] <= value + tolerance_seconds
        ):
            matched += 1
    return matched


def playable_hit_count(notes: str) -> int:
    return sum(character in "1234" for character in notes)


def density_hit_count(notes: str) -> int:
    return sum(character != "0" for character in notes)


def chart_activity_count(notes: str) -> int:
    return sum(character in "123457" for character in notes)


def normal_note_times(chart_bar: ChartBar, feature_bar: BarFeature) -> list[float]:
    duration = feature_bar.end_time - feature_bar.start_time
    if not isfinite(duration) or duration <= 0 or not chart_bar.notes:
        return []
    grid_duration = duration / len(chart_bar.notes)
    return [
        feature_bar.start_time + grid * grid_duration
        for grid, note in enumerate(chart_bar.notes)
        if note in "1234"
    ]


def longest_note_stream(note_times: list[float]) -> tuple[int, float]:
    if not note_times:
        return 0, 0.0

    ordered = sorted(note_times)
    longest_count = 1
    longest_seconds = 0.0
    stream_start = ordered[0]
    stream_count = 1
    for previous, current in zip(ordered[:-1], ordered[1:], strict=True):
        if current - previous <= NOTE_STREAM_MAX_GAP_SECONDS:
            stream_count += 1
        else:
            stream_start = current
            stream_count = 1
        stream_seconds = current - stream_start if stream_count > 1 else 0.0
        if stream_count > longest_count or (
            stream_count == longest_count and stream_seconds > longest_seconds
        ):
            longest_count = stream_count
            longest_seconds = stream_seconds
    return longest_count, longest_seconds


def accent_coverage_counts(
    chart_bar: ChartBar,
    feature_bar: BarFeature,
    *,
    salience_candidates: list[SalienceCandidate] | None = None,
) -> tuple[int, int]:
    if not chart_bar.notes or feature_bar.grids_per_bar <= 0:
        return 0, 0
    resolved_candidates = salience_candidates
    if resolved_candidates is None:
        resolved_candidates = build_salience_candidate_bars([feature_bar])[0]
    meter = get_meter_spec(feature_bar.time_signature)
    candidate_limit = max(1, len(meter.beat_positions_quarters))
    feature_candidates = {
        candidate.grid
        for candidate in rank_accent_candidates(resolved_candidates)[:candidate_limit]
    }
    candidates = {
        min(
            len(chart_bar.notes) - 1,
            round(grid / feature_bar.grids_per_bar * len(chart_bar.notes)),
        )
        for grid in feature_candidates
    }
    hits = sum(chart_bar.notes[grid] in "1234" for grid in candidates)
    return len(candidates), hits


def special_note_metrics(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
) -> tuple[int, int, float, float, int]:
    drumroll_count = 0
    balloon_count = 0
    drumroll_duration = 0.0
    balloon_duration = 0.0
    balloon_required_hits = 0

    for bar_index, chart_bar in enumerate(chart_bars):
        balloon_index = 0
        feature_bar = feature_bars[bar_index] if bar_index < len(feature_bars) else None
        bar_duration = (
            feature_bar.end_time - feature_bar.start_time
            if feature_bar is not None
            else 0.0
        )
        valid_duration = isfinite(bar_duration) and bar_duration > 0 and len(chart_bar.notes) > 0

        for start_grid, note in enumerate(chart_bar.notes):
            if note not in {"5", "7"}:
                continue
            end_grid = chart_bar.notes.find("8", start_grid + 1)
            duration = (
                bar_duration * (end_grid - start_grid) / len(chart_bar.notes)
                if valid_duration and end_grid > start_grid
                else 0.0
            )
            if note == "5":
                drumroll_count += 1
                drumroll_duration += duration
                continue

            balloon_count += 1
            balloon_duration += duration
            if balloon_index < len(chart_bar.balloon_counts):
                count = chart_bar.balloon_counts[balloon_index]
                if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    balloon_required_hits += count
            balloon_index += 1

    return (
        drumroll_count,
        balloon_count,
        drumroll_duration,
        balloon_duration,
        balloon_required_hits,
    )


def normalized_hit_count(notes: str, expected_length: int) -> float:
    if expected_length <= 0:
        return 0.0
    return float(density_hit_count(notes))


def density_hint_is_satisfied(hit_count: int, hint: BarDensityHint) -> bool:
    if hint.max_hits is not None and hit_count > hint.max_hits:
        return False
    return hint.allow_empty or hit_count >= hint.min_hits


def empty_runs(
    hit_counts: list[int],
    indexes: list[int] | None = None,
) -> list[tuple[int, int]]:
    selected_indexes = indexes if indexes is not None else list(range(len(hit_counts)))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous_index: int | None = None

    for index in selected_indexes:
        if index < 0 or index >= len(hit_counts):
            continue
        continues_run = previous_index is not None and index == previous_index + 1
        if not continues_run and start is not None and previous_index is not None:
            runs.append((start, previous_index))
            start = None

        if hit_counts[index] == 0:
            if start is None:
                start = index
        elif start is not None and previous_index is not None:
            runs.append((start, previous_index))
            start = None

        previous_index = index

    if start is not None and previous_index is not None:
        runs.append((start, previous_index))
    return runs


def longest_empty_bar_run(hit_counts: list[int]) -> int:
    return max((end - start + 1 for start, end in empty_runs(hit_counts)), default=0)


def pattern_counts(chart_bars: list[ChartBar]) -> Counter[str]:
    return Counter(
        _normalized_pattern_signature(bar.notes)
        for bar in chart_bars
        if chart_activity_count(bar.notes) > 0
    )


def _normalized_pattern_signature(notes: str) -> str:
    if not notes:
        return ""
    return "|".join(
        f"{Fraction(index, len(notes))}:{note}"
        for index, note in enumerate(notes)
        if note != "0"
    )


def note_color_metrics(chart_bars: list[ChartBar]) -> tuple[int, int, int]:
    don_count = 0
    ka_count = 0
    longest_run = 0
    current_color: str | None = None
    current_run = 0

    for bar in chart_bars:
        for note in bar.notes:
            if note == "0":
                continue
            if note not in {"1", "2"}:
                current_color = None
                current_run = 0
                continue

            if note == "1":
                don_count += 1
            else:
                ka_count += 1

            if note == current_color:
                current_run += 1
            else:
                current_color = note
                current_run = 1
            longest_run = max(longest_run, current_run)

    return don_count, ka_count, longest_run
