from dataclasses import dataclass
from math import ceil, floor, isfinite

from tja_ai_chartgen.features.density import BarDensityHint, build_density_hints
from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.features.salience_candidates import (
    SalienceCandidate,
    build_salience_candidate_bars,
    rank_accent_candidates,
)
from tja_ai_chartgen.rules.styles import (
    StyleTemplate,
    get_style_template,
    style_color_sequence,
    style_grid_bias,
)
from tja_ai_chartgen.tja.event_encoder import encode_chart_bar_events
from tja_ai_chartgen.tja.model import (
    BarFeature,
    ChartBar,
    ChartBarEvents,
    ChartHitEvent,
    ChartLongNoteEvent,
    GridFeature,
    InstrumentGridFeature,
    ResolutionPlan,
    SpectralGridFeature,
)

MAX_HITS_PER_SECOND = 12.0
BALLOON_HITS_PER_SECOND = {
    "easy": (2.0, 4.0),
    "normal": (3.0, 5.0),
    "hard": (4.0, 7.0),
    "oni": (5.0, 9.0),
}

DENSITY_LEVELS = ("auto", "low", "medium", "high", "max")
DENSITY_LOAD_MULTIPLIERS = {
    "low": 0.75,
    "medium": 0.9,
    "auto": 1.0,
    "high": 1.1,
    "max": 1.2,
}
WEAK_FILL_RATIOS = {
    "easy": 0.45,
    "normal": 0.5,
    "hard": 0.55,
    "oni": 0.6,
}
UNSUPPORTED_FILL_RATIOS = {
    "easy": 0.1,
    "normal": 0.12,
    "hard": 0.15,
    "oni": 0.2,
}
MINIMUM_SKELETON_HITS = {
    "easy": 1,
    "normal": 2,
    "hard": 2,
    "oni": 3,
}
MAX_CONSECUTIVE_WEAK_HITS = 2
GRID_ACTIVITY_EVIDENCE_THRESHOLD = 0.18
SPECTRAL_GRID_EVIDENCE_THRESHOLD = 0.25
INSTRUMENT_GRID_EVIDENCE_THRESHOLD = 0.35
BIG_NOTE_LIMITS = {
    "easy": 1,
    "normal": 1,
    "hard": 2,
    "oni": 2,
}
STYLE_COLOR_BIAS = 0.30
COLOR_OVERRIDE_MARGIN = 0.05
MAX_GENERATED_MONOCHROME_RUN = 4


@dataclass(frozen=True)
class CourseLoadProfile:
    name: str
    min_level: int
    max_level: int
    min_notes_per_second: float
    max_notes_per_second: float
    speed_cap: float
    max_occupancy: float


COURSE_LOAD_PROFILES = {
    "easy": CourseLoadProfile("Easy", 1, 5, 1.1, 2.1, 3.0, 0.38),
    "normal": CourseLoadProfile("Normal", 1, 7, 1.8, 3.4, 5.0, 0.5),
    "hard": CourseLoadProfile("Hard", 1, 8, 3.2, 5.8, 7.5, 0.7),
    "oni": CourseLoadProfile("Oni", 1, 10, 4.0, 7.0, 10.0, 0.82),
}


def choose_pattern(
    bar: BarFeature,
    style: str = "technical",
    density: str = "auto",
    course: str = "Oni",
    level: int = 10,
) -> str:
    validate_density(density)
    profile = _course_load_profile(course)
    template = get_style_template(style)
    hint = build_density_hints([bar])[0]
    output_resolution = _default_output_resolution(bar)
    return _feature_driven_pattern(
        bar,
        hint,
        density,
        template,
        profile,
        level,
        output_resolution=output_resolution,
    )


def generate_fallback_chart_bars(
    bars: list[BarFeature],
    style: str = "technical",
    density: str = "auto",
    special_notes: bool = False,
    course: str = "Oni",
    level: int = 10,
    resolution_plan: ResolutionPlan | None = None,
) -> list[ChartBar]:
    validate_density(density)
    profile = _course_load_profile(course)
    template = get_style_template(style)
    density_hints = build_density_hints(bars)
    salience_candidate_bars = build_salience_candidate_bars(
        bars,
        resolution_plan=resolution_plan,
    )
    chart_bars: list[ChartBar] = []
    previous_was_special = False
    previous_ended_with_big_note = False
    previous_normal_color: str | None = None
    previous_normal_color_run = 0
    for position, bar in enumerate(bars):
        hint = density_hints[position]
        salience_candidates = salience_candidate_bars[position]
        output_resolution = _resolution_for_bar(resolution_plan, bar, position)
        events: ChartBarEvents | None = None
        if hint.kind in {"silent", "rest"}:
            events = ChartBarEvents(index=bar.index)
            previous_was_special = False
        elif special_notes and not previous_was_special:
            events = _special_bar_events(
                bar,
                density=density,
                template=template,
                profile=profile,
                level=level,
                output_resolution=output_resolution,
            )
            previous_was_special = events is not None
        if events is None:
            events = _feature_driven_events(
                bar,
                hint,
                density,
                template,
                profile,
                level,
                output_resolution=output_resolution,
                salience_candidates=salience_candidates,
                forbid_initial_big_note=previous_ended_with_big_note,
                previous_normal_color=previous_normal_color,
                previous_normal_color_run=previous_normal_color_run,
            )
            previous_was_special = False
        chart_bar = encode_chart_bar_events(
            events,
            canonical_grids_per_bar=bar.grids_per_bar,
            output_resolution=output_resolution,
            time_signature=bar.time_signature,
        )
        chart_bars.append(chart_bar)
        previous_ended_with_big_note = bool(
            chart_bar.notes and chart_bar.notes[-1] in {"3", "4"}
        )
        previous_normal_color, previous_normal_color_run = _ending_color_run(
            chart_bar.notes,
            previous_color=previous_normal_color,
            previous_run=previous_normal_color_run,
        )
    return chart_bars


def validate_density(density: str) -> None:
    if density not in DENSITY_LEVELS:
        allowed = ", ".join(DENSITY_LEVELS)
        raise ValueError(f"Invalid density: {density}. Expected one of: {allowed}")


def _feature_driven_pattern(
    bar: BarFeature,
    hint: BarDensityHint,
    density: str,
    template: StyleTemplate,
    profile: CourseLoadProfile,
    level: int,
    *,
    output_resolution: int,
) -> str:
    events = _feature_driven_events(
        bar,
        hint,
        density,
        template,
        profile,
        level,
        output_resolution=output_resolution,
        salience_candidates=build_salience_candidate_bars([bar])[0],
    )
    return encode_chart_bar_events(
        events,
        canonical_grids_per_bar=bar.grids_per_bar,
        output_resolution=output_resolution,
        time_signature=bar.time_signature,
    ).notes


def _feature_driven_events(
    bar: BarFeature,
    hint: BarDensityHint,
    density: str,
    template: StyleTemplate,
    profile: CourseLoadProfile,
    level: int,
    *,
    output_resolution: int,
    salience_candidates: list[SalienceCandidate],
    forbid_initial_big_note: bool = False,
    previous_normal_color: str | None = None,
    previous_normal_color_run: int = 0,
) -> ChartBarEvents:
    grid_features = _project_grid_features(
        bar,
        _grid_features_for_bar(bar),
        output_resolution=output_resolution,
    )
    spectral_features = _project_spectral_grid_features(
        bar,
        output_resolution=output_resolution,
    )
    instrument_features = _project_instrument_grid_features(
        bar,
        output_resolution=output_resolution,
    )
    target_hits = _target_hit_count(
        bar,
        hint,
        density,
        profile,
        level,
        output_resolution=output_resolution,
        grid_features=grid_features,
    )
    if target_hits <= 0:
        return ChartBarEvents(index=bar.index)

    selected = _select_hit_grids(
        bar,
        grid_features,
        target_hits,
        template,
        hint=hint,
        profile=profile,
        output_resolution=output_resolution,
        salience_candidates=salience_candidates,
        spectral_features=spectral_features,
        instrument_features=instrument_features,
    )
    colors = style_color_sequence(template, _effective_density(density, hint), bar.index)
    big_note_grids = (
        _select_big_note_grids(
            bar,
            selected,
            salience_candidates,
            profile=profile,
            output_resolution=output_resolution,
            forbid_initial_big_note=forbid_initial_big_note,
        )
        if template.name == "performance"
        else set()
    )
    candidates_by_grid = {candidate.grid: candidate for candidate in salience_candidates}
    current_color = previous_normal_color
    current_run = previous_normal_color_run
    hits: list[ChartHitEvent] = []
    for sequence_index, grid in enumerate(sorted(selected)):
        color = colors[sequence_index % len(colors)]
        color = _salience_color(
            color,
            candidates_by_grid.get(grid),
            instrument_features.get(grid),
        )
        if template.name == "performance":
            color = "1" if color in {"1", "3"} else "2"
        if grid in big_note_grids:
            color = "3" if color in {"1", "3"} else "4"
            current_color = None
            current_run = 0
        else:
            color, current_color, current_run = _balance_generated_color(
                color,
                previous_color=current_color,
                previous_run=current_run,
            )
        hits.append(ChartHitEvent(tick=grid, note=color))
    return ChartBarEvents(index=bar.index, hits=hits)


def _target_hit_count(
    bar: BarFeature,
    hint: BarDensityHint,
    density: str,
    profile: CourseLoadProfile,
    level: int,
    *,
    output_resolution: int,
    grid_features: list[GridFeature],
) -> int:
    meter = get_meter_spec(bar.time_signature)
    hint_maximum = hint.max_hits if hint.max_hits is not None else output_resolution
    occupancy_cap = max(
        1,
        floor(meter.legacy_grids_per_bar * profile.max_occupancy),
    )
    maximum = min(hint_maximum, output_resolution, occupancy_cap)
    if hint.kind == "normal":
        evidence_count = max(
            sum(feature.onset for feature in grid_features),
            sum(feature.beat is not None for feature in grid_features),
            sum(feature.activity >= 0.18 for feature in grid_features),
        )
        evidence_allowance = floor(
            meter.legacy_grids_per_bar * profile.max_occupancy / 2
        )
        maximum = min(maximum, max(hint.min_hits, evidence_count + evidence_allowance))
    minimum = min(hint.min_hits, maximum)
    duration = bar.end_time - bar.start_time

    onset_count = sum(feature.onset for feature in grid_features)
    richness = min(1.0, onset_count / max(1, meter.legacy_grids_per_bar / 2))
    musical_factor = 0.75 + min(1.0, max(0.0, (bar.energy + richness) / 2)) * 0.35
    instrument_drive = min(
        1.0,
        bar.instrument.drum_activity * 0.4
        + bar.instrument.bass_activity * 0.25
        + bar.instrument.other_activity * 0.2
        + bar.instrument.vocal_activity * 0.15,
    )
    musical_factor += instrument_drive * 0.08
    if hint.kind in {"dense", "fill"}:
        musical_factor = max(musical_factor, 1.0)
    elif hint.kind == "sparse":
        musical_factor = min(musical_factor, 0.75)

    if isfinite(duration) and duration > 0:
        target_nps = _target_notes_per_second(profile, level, density)
        target = round(target_nps * musical_factor * hint.target_scale * duration)
        speed_cap = max(1, floor(duration * min(profile.speed_cap, MAX_HITS_PER_SECOND)))
        maximum = min(maximum, speed_cap)
        minimum = min(minimum, maximum)
    else:
        position = min(1.0, max(0.0, musical_factor - 0.5))
        target = round((minimum + (maximum - minimum) * position) * hint.target_scale)

    return max(minimum, min(target, maximum))


def _course_load_profile(course: str) -> CourseLoadProfile:
    profile = COURSE_LOAD_PROFILES.get(course.casefold())
    if profile is None:
        allowed = ", ".join(profile.name for profile in COURSE_LOAD_PROFILES.values())
        raise ValueError(f"Invalid course: {course}. Expected one of: {allowed}")
    return profile


def _target_notes_per_second(
    profile: CourseLoadProfile,
    level: int,
    density: str,
) -> float:
    clamped_level = min(profile.max_level, max(profile.min_level, level))
    level_span = profile.max_level - profile.min_level
    level_position = (
        (clamped_level - profile.min_level) / level_span if level_span else 0.0
    )
    base = profile.min_notes_per_second + (
        profile.max_notes_per_second - profile.min_notes_per_second
    ) * level_position
    adjusted = base * DENSITY_LOAD_MULTIPLIERS[density]
    return min(profile.speed_cap, max(0.5, adjusted))


def _grid_features_for_bar(bar: BarFeature) -> list[GridFeature]:
    existing = {
        feature.grid: feature
        for feature in bar.grid_features
        if 0 <= feature.grid < bar.grids_per_bar
    }
    onset_grids = {
        grid for grid in bar.onset_grids if 0 <= grid < bar.grids_per_bar
    }
    accent_grids = {
        grid for grid in bar.accent_grids if 0 <= grid < bar.grids_per_bar
    }
    beat_numbers = {
        grid: number
        for number, grid in enumerate(
            (grid for grid in bar.beat_grids if 0 <= grid < bar.grids_per_bar),
            start=1,
        )
    }
    activity = bar.activity_grids

    features: list[GridFeature] = []
    for grid in range(bar.grids_per_bar):
        feature = existing.get(grid)
        if feature is not None:
            features.append(feature)
            continue
        features.append(
            GridFeature(
                grid=grid,
                onset=grid in onset_grids,
                accent=grid in accent_grids,
                beat=beat_numbers.get(grid),
                downbeat=grid == bar.downbeat_grid,
                strength=1.0 if grid in onset_grids else 0.0,
                activity=activity[grid] if grid < len(activity) else 0.0,
            )
        )
    return features


def _project_grid_features(
    bar: BarFeature,
    features: list[GridFeature],
    *,
    output_resolution: int,
) -> list[GridFeature]:
    if output_resolution <= 0 or bar.grids_per_bar % output_resolution:
        raise ValueError(
            f"Bar {bar.index} canonical grid {bar.grids_per_bar} is incompatible with "
            f"resolution {output_resolution}"
        )
    step = bar.grids_per_bar // output_resolution
    projected = {
        grid: GridFeature(grid=grid)
        for grid in range(0, bar.grids_per_bar, step)
    }
    for feature in features:
        target = round(feature.grid / step) * step
        target = min(bar.grids_per_bar - step, max(0, target))
        current = projected[target]
        projected[target] = GridFeature(
            grid=target,
            onset=current.onset or feature.onset,
            accent=current.accent or feature.accent,
            beat=current.beat if current.beat is not None else feature.beat,
            downbeat=current.downbeat or feature.downbeat,
            strength=max(current.strength, feature.strength),
            activity=max(current.activity, feature.activity),
        )
    return [projected[grid] for grid in sorted(projected)]


def _project_spectral_grid_features(
    bar: BarFeature,
    *,
    output_resolution: int,
) -> dict[int, SpectralGridFeature]:
    if output_resolution <= 0 or bar.grids_per_bar % output_resolution:
        raise ValueError(
            f"Bar {bar.index} canonical grid {bar.grids_per_bar} is incompatible with "
            f"resolution {output_resolution}"
        )
    step = bar.grids_per_bar // output_resolution
    projected: dict[int, SpectralGridFeature] = {}
    for feature in bar.spectral_grid_features:
        if feature.grid < 0 or feature.grid >= bar.grids_per_bar:
            continue
        target = round(feature.grid / step) * step
        target = min(bar.grids_per_bar - step, max(0, target))
        current = projected.get(target, SpectralGridFeature(grid=target))
        projected[target] = SpectralGridFeature(
            grid=target,
            low_onset_strength=max(
                current.low_onset_strength,
                feature.low_onset_strength,
            ),
            mid_onset_strength=max(
                current.mid_onset_strength,
                feature.mid_onset_strength,
            ),
            high_onset_strength=max(
                current.high_onset_strength,
                feature.high_onset_strength,
            ),
            spectral_flux=max(current.spectral_flux, feature.spectral_flux),
        )
    return projected


def _project_instrument_grid_features(
    bar: BarFeature,
    *,
    output_resolution: int,
) -> dict[int, InstrumentGridFeature]:
    if output_resolution <= 0 or bar.grids_per_bar % output_resolution:
        raise ValueError(
            f"Bar {bar.index} canonical grid {bar.grids_per_bar} is incompatible with "
            f"resolution {output_resolution}"
        )
    step = bar.grids_per_bar // output_resolution
    projected: dict[int, InstrumentGridFeature] = {}
    for feature in bar.instrument_grid_features:
        if feature.grid < 0 or feature.grid >= bar.grids_per_bar:
            continue
        target = round(feature.grid / step) * step
        target = min(bar.grids_per_bar - step, max(0, target))
        current = projected.get(target, InstrumentGridFeature(grid=target))
        projected[target] = InstrumentGridFeature(
            grid=target,
            vocal_onset=max(current.vocal_onset, feature.vocal_onset),
            drum_onset=max(current.drum_onset, feature.drum_onset),
            bass_onset=max(current.bass_onset, feature.bass_onset),
            accompaniment_onset=max(
                current.accompaniment_onset,
                feature.accompaniment_onset,
            ),
        )
    return projected


def _select_hit_grids(
    bar: BarFeature,
    grid_features: list[GridFeature],
    target_hits: int,
    template: StyleTemplate,
    *,
    hint: BarDensityHint,
    profile: CourseLoadProfile,
    output_resolution: int,
    salience_candidates: list[SalienceCandidate],
    spectral_features: dict[int, SpectralGridFeature],
    instrument_features: dict[int, InstrumentGridFeature],
) -> set[int]:
    selected: set[int] = set()
    remaining = {feature.grid: feature for feature in grid_features}
    reliable_grids: set[int] = set()
    for candidate in salience_candidates:
        if not candidate.reliable:
            continue
        if candidate.grid not in remaining:
            continue
        selected.add(candidate.grid)
        reliable_grids.add(candidate.grid)
        del remaining[candidate.grid]
        if len(selected) >= target_hits:
            return selected

    weak_limit = _weak_fill_limit(bar, hint, profile, target_hits)
    weak_salience_grids = {
        candidate.grid for candidate in salience_candidates if not candidate.reliable
    }
    supported_grids = {
        grid
        for grid, feature in remaining.items()
        if grid in weak_salience_grids
        or _has_grid_evidence(
            feature,
            spectral_feature=spectral_features.get(grid),
            instrument_feature=instrument_features.get(grid),
        )
    }
    has_supported_evidence = bool(reliable_grids or supported_grids)
    weak_added = _select_supplemental_grids(
        bar,
        remaining,
        selected,
        reliable_grids,
        supported_grids,
        limit=weak_limit,
        target_hits=target_hits,
        template=template,
        output_resolution=output_resolution,
        spectral_features=spectral_features,
        instrument_features=instrument_features,
    )

    unsupported_limit = min(
        weak_limit - weak_added,
        _unsupported_fill_limit(
            bar,
            hint,
            profile,
            target_hits,
            has_supported_evidence=has_supported_evidence,
        ),
    )
    _select_supplemental_grids(
        bar,
        remaining,
        selected,
        reliable_grids,
        set(remaining),
        limit=unsupported_limit,
        target_hits=target_hits,
        template=template,
        output_resolution=output_resolution,
        spectral_features=spectral_features,
        instrument_features=instrument_features,
    )
    return selected


def _select_supplemental_grids(
    bar: BarFeature,
    remaining: dict[int, GridFeature],
    selected: set[int],
    reliable_grids: set[int],
    candidate_grids: set[int],
    *,
    limit: int,
    target_hits: int,
    template: StyleTemplate,
    output_resolution: int,
    spectral_features: dict[int, SpectralGridFeature],
    instrument_features: dict[int, InstrumentGridFeature],
) -> int:
    added = 0
    step = bar.grids_per_bar // output_resolution
    while candidate_grids and added < limit and len(selected) < target_hits:
        eligible = {
            grid
            for grid in candidate_grids
            if grid in remaining
            and not _would_exceed_weak_run(
                grid,
                selected,
                reliable_grids,
                step=step,
            )
        }
        if not eligible:
            break
        best_grid = max(
            eligible,
            key=lambda grid: (
                _grid_score(
                    bar,
                    remaining[grid],
                    selected,
                    template,
                    spectral_feature=spectral_features.get(grid),
                    instrument_feature=instrument_features.get(grid),
                )
                + _connection_score(grid, reliable_grids, step=step),
                -grid,
            ),
        )
        selected.add(best_grid)
        del remaining[best_grid]
        candidate_grids.remove(best_grid)
        added += 1
    return added


def _select_big_note_grids(
    bar: BarFeature,
    selected: set[int],
    salience_candidates: list[SalienceCandidate],
    *,
    profile: CourseLoadProfile,
    output_resolution: int,
    forbid_initial_big_note: bool,
) -> set[int]:
    """按统一 accent salience 选择少量、互不相邻的大音符。"""
    limit = BIG_NOTE_LIMITS[profile.name.casefold()]
    if limit <= 0 or not selected:
        return set()

    step = bar.grids_per_bar // output_resolution
    chosen: set[int] = set()
    for candidate in rank_accent_candidates(salience_candidates):
        if candidate.grid not in selected:
            continue
        if forbid_initial_big_note and candidate.grid == 0:
            continue
        if any(abs(candidate.grid - grid) <= step for grid in chosen):
            continue
        chosen.add(candidate.grid)
        if len(chosen) >= limit:
            break
    return chosen


def _weak_fill_limit(
    bar: BarFeature,
    hint: BarDensityHint,
    profile: CourseLoadProfile,
    target_hits: int,
) -> int:
    if target_hits <= 0:
        return 0
    if hint.kind == "sparse":
        return min(target_hits, 2)
    if bar.transition_role == "breakdown":
        return min(target_hits, 2)
    ratio = WEAK_FILL_RATIOS[profile.name.casefold()]
    return min(target_hits, max(2, ceil(target_hits * ratio)))


def _unsupported_fill_limit(
    bar: BarFeature,
    hint: BarDensityHint,
    profile: CourseLoadProfile,
    target_hits: int,
    *,
    has_supported_evidence: bool,
) -> int:
    if target_hits <= 0 or hint.kind == "sparse" or bar.transition_role == "breakdown":
        return 0
    course = profile.name.casefold()
    limit = floor(target_hits * UNSUPPORTED_FILL_RATIOS[course])
    if not has_supported_evidence and bar.energy >= 0.18:
        limit = max(limit, MINIMUM_SKELETON_HITS[course])
    return min(target_hits, limit)


def _has_grid_evidence(
    feature: GridFeature,
    *,
    spectral_feature: SpectralGridFeature | None,
    instrument_feature: InstrumentGridFeature | None,
) -> bool:
    if (
        feature.onset
        or feature.accent
        or feature.downbeat
        or feature.beat is not None
        or feature.activity >= GRID_ACTIVITY_EVIDENCE_THRESHOLD
    ):
        return True
    if spectral_feature is not None and max(
        spectral_feature.low_onset_strength,
        spectral_feature.mid_onset_strength,
        spectral_feature.high_onset_strength,
        spectral_feature.spectral_flux,
    ) >= SPECTRAL_GRID_EVIDENCE_THRESHOLD:
        return True
    return instrument_feature is not None and max(
        instrument_feature.drum_onset,
        instrument_feature.bass_onset,
        instrument_feature.vocal_onset,
        instrument_feature.accompaniment_onset,
    ) >= INSTRUMENT_GRID_EVIDENCE_THRESHOLD


def _would_exceed_weak_run(
    grid: int,
    selected: set[int],
    reliable_grids: set[int],
    *,
    step: int,
) -> bool:
    weak_grids = selected - reliable_grids
    run_length = 1
    neighbor = grid - step
    while neighbor in weak_grids:
        run_length += 1
        neighbor -= step
    neighbor = grid + step
    while neighbor in weak_grids:
        run_length += 1
        neighbor += step
    return run_length > MAX_CONSECUTIVE_WEAK_HITS


def _connection_score(grid: int, reliable_grids: set[int], *, step: int) -> float:
    if not reliable_grids:
        return 0.0
    distance = min(abs(grid - reliable_grid) for reliable_grid in reliable_grids)
    if distance <= step:
        return 1.5
    if distance <= step * 2:
        return 0.75
    return 0.0


def _grid_score(
    bar: BarFeature,
    feature: GridFeature,
    selected: set[int],
    template: StyleTemplate,
    *,
    spectral_feature: SpectralGridFeature | None,
    instrument_feature: InstrumentGridFeature | None,
) -> float:
    score = 0.0
    if feature.onset:
        score += 10.0
    if feature.accent:
        score += 5.0
    if feature.downbeat:
        score += 4.0
    if feature.beat is not None:
        score += 3.0
    score += feature.strength * 4.0
    score += feature.activity
    if spectral_feature is not None:
        score += spectral_feature.spectral_flux * 3.0
        score += max(
            spectral_feature.low_onset_strength,
            spectral_feature.mid_onset_strength,
            spectral_feature.high_onset_strength,
        ) * 2.0
    if instrument_feature is not None:
        score += instrument_feature.drum_onset * 5.0
        score += instrument_feature.bass_onset * 2.0
        score += instrument_feature.accompaniment_onset * 2.0
        score += instrument_feature.vocal_onset * 0.8
    score += style_grid_bias(template, feature.grid, bar.grids_per_bar, bar.index)

    if selected:
        distance = min(abs(feature.grid - grid) for grid in selected)
        if template.name in {"technical", "performance"}:
            score += min(distance, 4) * 0.25
        elif template.name == "stamina" and distance == 1:
            score += 0.8
        elif template.name == "hybrid":
            score += 0.6 if distance == 1 else min(distance, 3) * 0.15
    return score


def _salience_color(
    color: str,
    candidate: SalienceCandidate | None,
    instrument_feature: InstrumentGridFeature | None,
) -> str:
    """将 style 作为基础票，再用统一软倾向校准咚咔。"""
    base_color = "don" if color in {"1", "3"} else "ka"
    don_score = STYLE_COLOR_BIAS if base_color == "don" else 0.0
    ka_score = STYLE_COLOR_BIAS if base_color == "ka" else 0.0
    if candidate is not None:
        don_score += candidate.point.don_preference
        ka_score += candidate.point.ka_preference
    if instrument_feature is not None:
        neutral_drive = max(
            instrument_feature.drum_onset,
            instrument_feature.vocal_onset,
            instrument_feature.accompaniment_onset,
        )
        if (
            instrument_feature.bass_onset >= 0.35
            and instrument_feature.bass_onset >= neutral_drive - 0.1
        ):
            don_score += instrument_feature.bass_onset * 0.18

    resolved = base_color
    if don_score - ka_score >= COLOR_OVERRIDE_MARGIN:
        resolved = "don"
    elif ka_score - don_score >= COLOR_OVERRIDE_MARGIN:
        resolved = "ka"
    is_big = color in {"3", "4"}
    if resolved == "don":
        return "3" if is_big else "1"
    return "4" if is_big else "2"


def _balance_generated_color(
    color: str,
    *,
    previous_color: str | None,
    previous_run: int,
) -> tuple[str, str | None, int]:
    if color not in {"1", "2"}:
        return color, None, 0
    if color == previous_color and previous_run >= MAX_GENERATED_MONOCHROME_RUN:
        color = "2" if color == "1" else "1"
        return color, color, 1
    if color == previous_color:
        return color, color, previous_run + 1
    return color, color, 1


def _ending_color_run(
    notes: str,
    *,
    previous_color: str | None,
    previous_run: int,
) -> tuple[str | None, int]:
    color = previous_color
    run = previous_run
    for note in notes:
        if note == "0":
            continue
        if note not in {"1", "2"}:
            color = None
            run = 0
            continue
        if note == color:
            run += 1
        else:
            color = note
            run = 1
    return color, run


def _effective_density(density: str, hint: BarDensityHint) -> str:
    if density != "auto":
        return density
    if hint.kind in {"dense", "fill"}:
        return "high"
    if hint.kind == "sparse":
        return "low"
    return "medium"


def _special_bar_events(
    bar: BarFeature,
    *,
    density: str,
    template: StyleTemplate,
    profile: CourseLoadProfile,
    level: int,
    output_resolution: int,
) -> ChartBarEvents | None:
    if density not in {"auto", "high", "max"}:
        return None
    if not bar.fill_candidate:
        return None

    duration = bar.end_time - bar.start_time
    if not isfinite(duration) or duration <= 0 or output_resolution < 3:
        return None
    has_activity = (
        bool(bar.onset_grids)
        or any(value > 0 for value in bar.activity_grids)
        or max(
            bar.instrument.vocal_activity,
            bar.instrument.drum_activity,
            bar.instrument.bass_activity,
            bar.instrument.other_activity,
        )
        > 0.0
    )
    meter = get_meter_spec(bar.time_signature)
    onset_richness = min(
        1.0,
        len(bar.onset_grids) / max(1, meter.legacy_grids_per_bar / 4),
    )
    activity_score = max(
        bar.energy,
        onset_richness,
        max(bar.activity_grids, default=0.0),
        bar.instrument.drum_activity,
        bar.instrument.bass_activity,
        bar.instrument.other_activity,
        bar.instrument.vocal_activity * 0.8,
    )
    if not has_activity or activity_score < template.special_min_energy:
        return None
    if density == "auto" and activity_score < 0.75:
        return None

    start_grid, end_grid = _special_note_span(bar, output_resolution=output_resolution)
    special_duration = duration * (end_grid - start_grid) / bar.grids_per_bar
    is_balloon = special_duration >= duration / 4 and (
        (
            bar.phrase_position == "song_end"
            and activity_score >= template.balloon_min_energy
        )
        or (bar.fill_candidate and bar.energy >= template.balloon_min_energy)
    )
    balloon_count = (
        _balloon_hit_count(
            special_duration,
            profile=profile,
            level=level,
            multiplier=template.balloon_hits_multiplier,
        )
        if is_balloon
        else None
    )
    return ChartBarEvents(
        index=bar.index,
        long_notes=[
            ChartLongNoteEvent(
                start_tick=start_grid,
                end_tick=end_grid,
                kind="balloon" if is_balloon else "drumroll",
                balloon_count=balloon_count,
            )
        ],
    )


def _special_note_span(bar: BarFeature, *, output_resolution: int) -> tuple[int, int]:
    features = _project_grid_features(
        bar,
        _grid_features_for_bar(bar),
        output_resolution=output_resolution,
    )
    step = bar.grids_per_bar // output_resolution
    earliest_start = max(0, (bar.grids_per_bar // 2 // step) * step)
    latest_start = bar.grids_per_bar - (2 * step)
    feature_grids = {
        feature.grid
        for feature in features
        if earliest_start <= feature.grid <= latest_start
        and (feature.onset or feature.accent or feature.beat is not None)
    }
    instrument_features = _project_instrument_grid_features(
        bar,
        output_resolution=output_resolution,
    )
    feature_grids.update(
        grid
        for grid, feature in instrument_features.items()
        if earliest_start <= grid <= latest_start
        and max(feature.drum_onset, feature.accompaniment_onset) >= 0.35
    )
    start_grid = min(feature_grids) if feature_grids else earliest_start

    projected_beats = sorted(
        {
            min(bar.grids_per_bar - step, round(grid / step) * step)
            for grid in bar.beat_grids
        }
    )
    later_beats = [
        grid for grid in projected_beats if start_grid + step < grid < bar.grids_per_bar
    ]
    end_grid = later_beats[-1] if later_beats else bar.grids_per_bar - step
    if end_grid <= start_grid:
        end_grid = min(bar.grids_per_bar - step, start_grid + step)
    return start_grid, end_grid


def _balloon_hit_count(
    duration: float,
    *,
    profile: CourseLoadProfile,
    level: int,
    multiplier: float,
) -> int:
    minimum, maximum = BALLOON_HITS_PER_SECOND[profile.name.casefold()]
    clamped_level = min(profile.max_level, max(profile.min_level, level))
    level_span = profile.max_level - profile.min_level
    level_position = (
        (clamped_level - profile.min_level) / level_span if level_span else 0.0
    )
    hits_per_second = minimum + (maximum - minimum) * level_position
    return max(1, round(duration * hits_per_second * multiplier))


def _resolution_for_bar(
    plan: ResolutionPlan | None,
    bar: BarFeature,
    position: int,
) -> int:
    if plan is not None:
        if 0 <= bar.index < len(plan.bar_resolutions):
            return plan.bar_resolutions[bar.index]
        if 0 <= position < len(plan.bar_resolutions):
            return plan.bar_resolutions[position]
        return plan.base_resolution
    return _default_output_resolution(bar)


def _default_output_resolution(bar: BarFeature) -> int:
    meter = get_meter_spec(bar.time_signature)
    if bar.grids_per_bar == meter.grids_per_bar:
        return meter.legacy_grids_per_bar
    return bar.grids_per_bar
