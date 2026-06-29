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


def choose_pattern(bar: BarFeature, style: str = "technical") -> str:
    if bar.energy < 0.30:
        patterns = LOW_PATTERNS
    elif bar.energy < 0.55:
        patterns = MID_PATTERNS
    elif bar.energy < 0.80:
        patterns = HIGH_PATTERNS
    else:
        patterns = PEAK_PATTERNS

    style_offset = 1 if style == "stamina" else 0
    return patterns[(bar.index + style_offset) % len(patterns)]


def generate_fallback_chart_bars(
    bars: list[BarFeature],
    style: str = "technical",
) -> list[ChartBar]:
    return [ChartBar(index=bar.index, notes=choose_pattern(bar, style)) for bar in bars]
