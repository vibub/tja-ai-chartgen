from tja_ai_chartgen.tja.model import BarFeature, ChartBar

LOW_PATTERNS = [
    "1000100010001000",
    "1000200010002000",
]

MID_PATTERNS = [
    "1122001111220011",
    "1012101210121022",
]

HIGH_PATTERNS = [
    "1122112211221122",
    "1212122212121222",
]

PEAK_PATTERNS = [
    "1122112233441122",
    "1211221211223344",
]

ROLL_PATTERN = "5000000080000000"
BALLOON_PATTERN = "7000000080000000"
BALLOON_COUNT = 8

DENSITY_LEVELS = ("auto", "low", "medium", "high", "max")
DENSITY_PATTERNS = {
    "low": LOW_PATTERNS,
    "medium": MID_PATTERNS,
    "high": HIGH_PATTERNS,
    "max": PEAK_PATTERNS,
}


def choose_pattern(
    bar: BarFeature,
    style: str = "technical",
    density: str = "auto",
) -> str:
    patterns = _select_patterns(bar, density)
    style_offset = 1 if style == "stamina" else 0
    return _fit_pattern_to_grid(patterns[(bar.index + style_offset) % len(patterns)], bar.grids_per_bar)


def generate_fallback_chart_bars(
    bars: list[BarFeature],
    style: str = "technical",
    density: str = "auto",
    special_notes: bool = False,
) -> list[ChartBar]:
    validate_density(density)
    chart_bars: list[ChartBar] = []
    for bar in bars:
        special_pattern = _special_pattern_for_bar(bar, density) if special_notes else None
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


def _select_patterns(bar: BarFeature, density: str) -> list[str]:
    validate_density(density)

    if density != "auto":
        return DENSITY_PATTERNS[density]

    if bar.energy < 0.30:
        return LOW_PATTERNS
    if bar.energy < 0.55:
        return MID_PATTERNS
    if bar.energy < 0.80:
        return HIGH_PATTERNS
    return PEAK_PATTERNS


def _special_pattern_for_bar(bar: BarFeature, density: str) -> str | None:
    if density not in {"auto", "high", "max"}:
        return None
    if density == "auto" and bar.energy < 0.75:
        return None
    if bar.index % 8 == 7:
        return "balloon"
    if bar.index % 4 == 3:
        return "roll"
    return None


def _fit_pattern_to_grid(pattern: str, grids_per_bar: int) -> str:
    if grids_per_bar <= len(pattern):
        return pattern[:grids_per_bar]
    return pattern.ljust(grids_per_bar, "0")
