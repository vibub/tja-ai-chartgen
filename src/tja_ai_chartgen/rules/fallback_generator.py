from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.rules.styles import StyleTemplate, get_style_template
from tja_ai_chartgen.tja.model import BarFeature, ChartBar

ROLL_PATTERN = "5000000080000000"
BALLOON_PATTERN = "7000000080000000"
BALLOON_COUNT = 8

DENSITY_LEVELS = ("auto", "low", "medium", "high", "max")


def choose_pattern(
    bar: BarFeature,
    style: str = "technical",
    density: str = "auto",
) -> str:
    template = get_style_template(style)
    patterns = _select_patterns(bar, density, template)
    return _fit_pattern_to_grid(
        patterns[(bar.index + template.pattern_shift) % len(patterns)],
        bar.grids_per_bar,
    )


def generate_fallback_chart_bars(
    bars: list[BarFeature],
    style: str = "technical",
    density: str = "auto",
    special_notes: bool = False,
) -> list[ChartBar]:
    validate_density(density)
    template = get_style_template(style)
    silent_indexes = edge_silence_indexes(bars)
    chart_bars: list[ChartBar] = []
    for index, bar in enumerate(bars):
        if index in silent_indexes:
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
                    notes=choose_pattern(bar, style, density),
                    time_signature=bar.time_signature,
                )
            )
    return chart_bars


def validate_density(density: str) -> None:
    if density not in DENSITY_LEVELS:
        allowed = ", ".join(DENSITY_LEVELS)
        raise ValueError(f"Invalid density: {density}. Expected one of: {allowed}")


def _select_patterns(bar: BarFeature, density: str, template: StyleTemplate) -> tuple[str, ...]:
    validate_density(density)

    if density != "auto":
        return _patterns_for_density(template, density)

    if bar.energy < 0.30:
        return template.low_patterns
    if bar.energy < 0.55:
        return template.medium_patterns
    if bar.energy < 0.80:
        return template.high_patterns
    return template.max_patterns


def _patterns_for_density(template: StyleTemplate, density: str) -> tuple[str, ...]:
    if density == "low":
        return template.low_patterns
    if density == "medium":
        return template.medium_patterns
    if density == "high":
        return template.high_patterns
    return template.max_patterns


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
