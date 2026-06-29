from tja_ai_chartgen.tja.model import BarFeature


def assign_sections(bars: list[BarFeature]) -> list[BarFeature]:
    total = len(bars)
    assigned: list[BarFeature] = []

    for position, bar in enumerate(bars):
        if position < 8:
            section = "intro"
        elif position >= max(0, total - 8):
            section = "outro"
        elif bar.energy >= 0.75:
            section = "chorus"
        elif bar.energy >= 0.45:
            section = "verse"
        else:
            section = "break"

        assigned.append(bar.model_copy(update={"section": section}))

    return assigned
