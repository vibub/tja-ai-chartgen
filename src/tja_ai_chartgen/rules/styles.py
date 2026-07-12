from dataclasses import dataclass


@dataclass(frozen=True)
class StyleTemplate:
    name: str
    description: str
    pattern_shift: int
    low_patterns: tuple[str, ...]
    medium_patterns: tuple[str, ...]
    high_patterns: tuple[str, ...]
    max_patterns: tuple[str, ...]
    special_min_energy: float
    balloon_min_energy: float
    balloon_hits_multiplier: float


STYLE_TEMPLATES = {
    "technical": StyleTemplate(
        name="technical",
        description="Balanced technical patterns with mixed don/ka placement.",
        pattern_shift=0,
        low_patterns=("1000100010001000", "1000200010002000"),
        medium_patterns=("1122001111220011", "1012101210121022"),
        high_patterns=("1122112211221122", "1212122212121222"),
        max_patterns=("1122112233441122", "1211221211223344"),
        special_min_energy=0.55,
        balloon_min_energy=0.9,
        balloon_hits_multiplier=1.0,
    ),
    "stamina": StyleTemplate(
        name="stamina",
        description="Straightforward repeated streams that emphasize endurance.",
        pattern_shift=1,
        low_patterns=("1000101010001010", "1010100010101000"),
        medium_patterns=("1110111011101110", "2220222022202220"),
        high_patterns=("1112111211121112", "2221222122212221"),
        max_patterns=("1122112211221122", "2211221122112211"),
        special_min_energy=0.55,
        balloon_min_energy=0.9,
        balloon_hits_multiplier=1.0,
    ),
    "hybrid": StyleTemplate(
        name="hybrid",
        description="Mixed technical and stamina phrases for varied drafts.",
        pattern_shift=0,
        low_patterns=("1000100010002000", "1000200010001000"),
        medium_patterns=("1122001110121022", "1012101211220011"),
        high_patterns=("1122112212121222", "1212122211221122"),
        max_patterns=("1122112233441122", "1211221233441212"),
        special_min_energy=0.55,
        balloon_min_energy=0.9,
        balloon_hits_multiplier=1.0,
    ),
    "performance": StyleTemplate(
        name="performance",
        description="Stage-like isolated accents and sparse visual highlights without overusing big notes.",
        pattern_shift=0,
        low_patterns=("3000400030004000", "1000300010004000"),
        medium_patterns=("1030204010302040", "3010401030104010"),
        high_patterns=("3311441133114411", "1133441111334411"),
        max_patterns=("3311441133441122", "1133442233114422"),
        special_min_energy=0.45,
        balloon_min_energy=0.8,
        balloon_hits_multiplier=0.95,
    ),
}

STYLE_LEVELS = tuple(STYLE_TEMPLATES)


def get_style_template(style: str) -> StyleTemplate:
    try:
        return STYLE_TEMPLATES[style]
    except KeyError as error:
        allowed = ", ".join(STYLE_LEVELS)
        raise ValueError(f"Invalid style: {style}. Expected one of: {allowed}") from error


def style_grid_bias(template: StyleTemplate, grid: int, grids_per_bar: int, bar_index: int) -> float:
    if template.name == "technical":
        return 0.7 if grid % 4 in {1, 3} else 0.0
    if template.name == "stamina":
        return 0.6 if grid % 2 == 0 else 0.3
    if template.name == "hybrid":
        if bar_index % 2 == 0:
            return 0.5 if grid % 4 in {1, 3} else 0.1
        return 0.5 if grid % 2 == 0 else 0.2
    if template.name == "performance":
        quarter = max(1, grids_per_bar // 4)
        return 0.8 if grid % quarter == 0 else -0.2
    return 0.0


def style_color_sequence(template: StyleTemplate, density: str, bar_index: int) -> tuple[str, ...]:
    patterns = {
        "low": template.low_patterns,
        "medium": template.medium_patterns,
        "high": template.high_patterns,
        "max": template.max_patterns,
    }
    selected_density = density if density != "auto" else "medium"
    candidates = patterns[selected_density]
    pattern = candidates[(bar_index + template.pattern_shift) % len(candidates)]
    colors = tuple(note for note in pattern if note != "0")
    return colors or ("1", "2")


def validate_style(style: str) -> None:
    get_style_template(style)
