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
    return patterns[(bar.index + style_offset) % len(patterns)]


def generate_fallback_chart_bars(
    bars: list[BarFeature],
    style: str = "technical",
    density: str = "auto",
) -> list[ChartBar]:
    validate_density(density)
    return [ChartBar(index=bar.index, notes=choose_pattern(bar, style, density)) for bar in bars]


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
