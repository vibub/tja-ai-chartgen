from tja_ai_chartgen.tja.model import BarFeature


INTRO_OUTRO_BARS = 8
SILENCE_ENERGY_THRESHOLD = 0.015


def assign_sections(bars: list[BarFeature]) -> list[BarFeature]:
    total = len(bars)
    assigned: list[BarFeature] = []
    max_energy = max((bar.energy for bar in bars), default=0.0)
    verse_threshold = max(0.045, max_energy * 0.35)
    chorus_threshold = max(0.08, max_energy * 0.75)

    for position, bar in enumerate(bars):
        if position < INTRO_OUTRO_BARS:
            section = "intro"
        elif position >= max(0, total - INTRO_OUTRO_BARS):
            section = "outro"
        elif bar.energy <= SILENCE_ENERGY_THRESHOLD and not bar.onset_16:
            section = "break"
        elif bar.energy >= chorus_threshold:
            section = "chorus"
        elif bar.energy >= verse_threshold or len(bar.onset_16) >= 4:
            section = "verse"
        else:
            section = "break"

        assigned.append(bar.model_copy(update={"section": section}))

    return assigned
