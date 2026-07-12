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


def choose_pattern(
    bar: BarFeature,
    style: str = "technical",
    density: str = "auto",
) -> str:
    validate_density(density)
    template = get_style_template(style)
    hint = build_density_hints([bar])[0]
    return _feature_driven_pattern(bar, hint, density, template)


def generate_fallback_chart_bars(
    bars: list[BarFeature],
    style: str = "technical",
    density: str = "auto",
    special_notes: bool = False,
) -> list[ChartBar]:
    validate_density(density)
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
                    notes=_feature_driven_pattern(bar, hint, density, template),
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
) -> str:
    target_hits = _target_hit_count(bar, hint, density)
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


def _target_hit_count(bar: BarFeature, hint: BarDensityHint, density: str) -> int:
    maximum = hint.max_hits if hint.max_hits is not None else bar.grids_per_bar
    maximum = min(maximum, bar.grids_per_bar)
    minimum = min(hint.min_hits, maximum)

    if density == "low":
        target = minimum
    elif density == "medium":
        target = round(minimum + (maximum - minimum) * 0.5)
    elif density == "high":
        target = round(minimum + (maximum - minimum) * 0.75)
    elif density == "max":
        target = maximum
    else:
        richness = min(1.0, len(bar.onset_16) / max(1, bar.grids_per_bar / 2))
        position = min(1.0, max(0.0, (bar.energy + richness) / 2))
        if hint.kind in {"dense", "fill"}:
            position = max(position, 0.65)
        elif hint.kind == "sparse":
            position = min(position, 0.35)
        target = round(minimum + (maximum - minimum) * position)

    duration = bar.end_time - bar.start_time
    if isfinite(duration) and duration > 0:
        speed_cap = max(1, floor(duration * MAX_HITS_PER_SECOND))
        target = min(target, speed_cap)
    return max(0, min(target, maximum))


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
