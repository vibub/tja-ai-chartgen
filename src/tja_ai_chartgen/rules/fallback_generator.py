from dataclasses import dataclass
from math import floor, isfinite

from tja_ai_chartgen.features.density import BarDensityHint, build_density_hints
from tja_ai_chartgen.rules.styles import (
    StyleTemplate,
    get_style_template,
    style_color_sequence,
    style_grid_bias,
)
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, GridFeature

ROLL_PATTERN = "5000000080000000"
BALLOON_PATTERN = "7000000080000000"
BALLOON_COUNT = 8
MAX_HITS_PER_SECOND = 12.0

DENSITY_LEVELS = ("auto", "low", "medium", "high", "max")
DENSITY_LOAD_MULTIPLIERS = {
    "low": 0.3,
    "medium": 0.85,
    "auto": 1.0,
    "high": 1.15,
    "max": 1.3,
}


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
    "easy": CourseLoadProfile("Easy", 1, 5, 1.25, 2.5, 3.0, 0.38),
    "normal": CourseLoadProfile("Normal", 1, 7, 2.25, 4.25, 5.0, 0.5),
    "hard": CourseLoadProfile("Hard", 1, 8, 3.75, 6.5, 7.5, 0.7),
    "oni": CourseLoadProfile("Oni", 1, 10, 5.5, 8.5, 10.0, 0.82),
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
    return _feature_driven_pattern(bar, hint, density, template, profile, level)


def generate_fallback_chart_bars(
    bars: list[BarFeature],
    style: str = "technical",
    density: str = "auto",
    special_notes: bool = False,
    course: str = "Oni",
    level: int = 10,
) -> list[ChartBar]:
    validate_density(density)
    profile = _course_load_profile(course)
    template = get_style_template(style)
    density_hints = build_density_hints(bars)
    chart_bars: list[ChartBar] = []
    for index, bar in enumerate(bars):
        hint = density_hints[index]
        if hint.kind in {"silent", "rest"}:
            chart_bars.append(
                ChartBar(
                    index=bar.index,
                    notes="0" * bar.grids_per_bar,
                    time_signature=bar.time_signature,
                )
            )
            continue

        special_pattern = _special_pattern_for_bar(bar, density, template) if special_notes else None
        if special_pattern == "balloon":
            chart_bars.append(
                ChartBar(
                    index=bar.index,
                    notes=_fit_pattern_to_grid(BALLOON_PATTERN, bar.grids_per_bar),
                    time_signature=bar.time_signature,
                    balloon_counts=[BALLOON_COUNT],
                )
            )
        elif special_pattern == "roll":
            chart_bars.append(
                ChartBar(
                    index=bar.index,
                    notes=_fit_pattern_to_grid(ROLL_PATTERN, bar.grids_per_bar),
                    time_signature=bar.time_signature,
                )
            )
        else:
            chart_bars.append(
                ChartBar(
                    index=bar.index,
                    notes=_feature_driven_pattern(
                        bar, hint, density, template, profile, level
                    ),
                    time_signature=bar.time_signature,
                )
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
) -> str:
    target_hits = _target_hit_count(bar, hint, density, profile, level)
    if target_hits <= 0:
        return "0" * bar.grids_per_bar

    grid_features = _grid_features_for_bar(bar)
    selected = _select_hit_grids(bar, grid_features, target_hits, template)
    colors = style_color_sequence(template, _effective_density(density, hint), bar.index)
    accent_grids = {feature.grid for feature in grid_features if feature.accent or feature.downbeat}
    notes = ["0"] * bar.grids_per_bar
    for sequence_index, grid in enumerate(sorted(selected)):
        color = colors[sequence_index % len(colors)]
        if template.name == "performance" and grid in accent_grids:
            color = "3" if color in {"1", "3"} else "4"
        notes[grid] = color
    return "".join(notes)


def _target_hit_count(
    bar: BarFeature,
    hint: BarDensityHint,
    density: str,
    profile: CourseLoadProfile,
    level: int,
) -> int:
    hint_maximum = hint.max_hits if hint.max_hits is not None else bar.grids_per_bar
    occupancy_cap = max(1, floor(bar.grids_per_bar * profile.max_occupancy))
    maximum = min(hint_maximum, bar.grids_per_bar, occupancy_cap)
    minimum = min(hint.min_hits, maximum)
    duration = bar.end_time - bar.start_time

    richness = min(1.0, len(bar.onset_16) / max(1, bar.grids_per_bar / 2))
    musical_factor = 0.75 + min(1.0, max(0.0, (bar.energy + richness) / 2)) * 0.35
    if hint.kind in {"dense", "fill"}:
        musical_factor = max(musical_factor, 1.0)
    elif hint.kind == "sparse":
        musical_factor = min(musical_factor, 0.75)

    if isfinite(duration) and duration > 0:
        target_nps = _target_notes_per_second(profile, level, density)
        target = round(target_nps * musical_factor * duration)
        speed_cap = max(1, floor(duration * min(profile.speed_cap, MAX_HITS_PER_SECOND)))
        maximum = min(maximum, speed_cap)
        minimum = min(minimum, maximum)
    else:
        position = min(1.0, max(0.0, musical_factor - 0.5))
        target = round(minimum + (maximum - minimum) * position)

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
    existing = {feature.grid: feature for feature in bar.grid_features if 0 <= feature.grid < bar.grids_per_bar}
    onset_grids = {grid for grid in bar.onset_16 if 0 <= grid < bar.grids_per_bar}
    accent_grids = {grid for grid in bar.accent_16 if 0 <= grid < bar.grids_per_bar}
    beat_numbers = {
        grid: number
        for number, grid in enumerate(
            (grid for grid in bar.beat_grids if 0 <= grid < bar.grids_per_bar),
            start=1,
        )
    }
    activity = bar.activity_16

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


def _select_hit_grids(
    bar: BarFeature,
    grid_features: list[GridFeature],
    target_hits: int,
    template: StyleTemplate,
) -> set[int]:
    selected: set[int] = set()
    remaining = {feature.grid: feature for feature in grid_features}
    while remaining and len(selected) < target_hits:
        best_grid = max(
            remaining,
            key=lambda grid: (
                _grid_score(bar, remaining[grid], selected, template),
                -grid,
            ),
        )
        selected.add(best_grid)
        del remaining[best_grid]
    return selected


def _grid_score(
    bar: BarFeature,
    feature: GridFeature,
    selected: set[int],
    template: StyleTemplate,
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
    score += style_grid_bias(template, feature.grid, bar.grids_per_bar, bar.index)

    if selected:
        distance = min(abs(feature.grid - grid) for grid in selected)
        if template.name in {"technical", "performance"}:
            score += min(distance, 4) * 0.25
        elif template.name == "stamina" and distance == 1:
            score += 0.8
        elif template.name == "hybrid":
            score += (0.6 if distance == 1 else min(distance, 3) * 0.15)
    return score


def _effective_density(density: str, hint: BarDensityHint) -> str:
    if density != "auto":
        return density
    if hint.kind in {"dense", "fill"}:
        return "high"
    if hint.kind == "sparse":
        return "low"
    return "medium"


def _special_pattern_for_bar(
    bar: BarFeature,
    density: str,
    template: StyleTemplate,
) -> str | None:
    if density not in {"auto", "high", "max"}:
        return None
    if density == "auto" and bar.energy < 0.75:
        return None
    if bar.index % template.balloon_every == template.balloon_every - 1:
        return "balloon"
    if bar.index % template.special_every == template.special_every - 1:
        return "roll"
    return None


def _fit_pattern_to_grid(pattern: str, grids_per_bar: int) -> str:
    if grids_per_bar <= len(pattern):
        return pattern[:grids_per_bar]
    return pattern.ljust(grids_per_bar, "0")
