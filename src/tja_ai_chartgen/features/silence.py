from tja_ai_chartgen.tja.model import BarFeature

SILENT_EDGE_ENERGY_THRESHOLD = 0.015
QUIET_EDGE_ENERGY_THRESHOLD = 0.03
QUIET_EDGE_ACTIVITY_THRESHOLD = 0.05


def _average_activity(bar: BarFeature) -> float:
    if not bar.activity_16:
        return 0.0
    return sum(bar.activity_16) / len(bar.activity_16)


def is_silent_bar(bar: BarFeature) -> bool:
    if bar.onset_16:
        return False
    if bar.energy <= SILENT_EDGE_ENERGY_THRESHOLD:
        return True
    return bar.energy <= QUIET_EDGE_ENERGY_THRESHOLD and _average_activity(bar) <= QUIET_EDGE_ACTIVITY_THRESHOLD


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
