from tja_ai_chartgen.features.structure import analyze_song_structure
from tja_ai_chartgen.tja.model import BarFeature


def assign_sections(bars: list[BarFeature]) -> list[BarFeature]:
    return analyze_song_structure(bars).bars
