from tja_ai_chartgen.tja.model import BarFeature

SILENT_EDGE_ENERGY_THRESHOLD = 0.015


def is_silent_bar(bar: BarFeature) -> bool:
    return bar.energy <= SILENT_EDGE_ENERGY_THRESHOLD and not bar.onset_16


def edge_silence_indexes(bars: list[BarFeature]) -> set[int]:
    indexes: set[int] = set()
    if not bars:
        return indexes

    if bars[0].index == 0:
        for index, bar in enumerate(bars):
            if not is_silent_bar(bar):
                break
            indexes.add(index)

    if bars[-1].phrase_position == "song_end":
        for index in range(len(bars) - 1, -1, -1):
            if not is_silent_bar(bars[index]):
                break
            indexes.add(index)

    return indexes
