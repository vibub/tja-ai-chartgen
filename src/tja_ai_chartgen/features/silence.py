from math import isfinite

from tja_ai_chartgen.tja.model import BarFeature

SILENT_EDGE_ENERGY_THRESHOLD = 0.015
QUIET_EDGE_ENERGY_THRESHOLD = 0.03
QUIET_EDGE_ACTIVITY_THRESHOLD = 0.05
TRANSIENT_EDGE_RMS_DBFS_THRESHOLD = -50.0
TRANSIENT_EDGE_RELATIVE_RMS_DB_THRESHOLD = -40.0
TRANSIENT_EDGE_PEAK_RMS_DBFS_THRESHOLD = -40.0
TRANSIENT_EDGE_SUSTAINED_RATIO_THRESHOLD = 0.15


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


def _is_transient_edge_noise(bar: BarFeature) -> bool:
    metrics = (
        bar.rms_dbfs,
        bar.peak_rms_dbfs,
        bar.relative_rms_db,
        bar.sustained_activity_ratio,
    )
    if any(value is None or not isfinite(value) for value in metrics):
        return False

    onset_limit = max(1, bar.grids_per_bar // 4)
    return (
        bar.rms_dbfs <= TRANSIENT_EDGE_RMS_DBFS_THRESHOLD
        and bar.relative_rms_db <= TRANSIENT_EDGE_RELATIVE_RMS_DB_THRESHOLD
        and bar.peak_rms_dbfs <= TRANSIENT_EDGE_PEAK_RMS_DBFS_THRESHOLD
        and bar.sustained_activity_ratio <= TRANSIENT_EDGE_SUSTAINED_RATIO_THRESHOLD
        and len(bar.onset_16) <= onset_limit
    )


def _is_edge_silent_bar(bar: BarFeature) -> bool:
    return is_silent_bar(bar) or _is_transient_edge_noise(bar)


def edge_silence_indexes(bars: list[BarFeature]) -> set[int]:
    indexes: set[int] = set()
    if not bars:
        return indexes

    if bars[0].index == 0:
        for index, bar in enumerate(bars):
            if not _is_edge_silent_bar(bar):
                break
            indexes.add(index)

    if bars[-1].phrase_position == "song_end":
        for index in range(len(bars) - 1, -1, -1):
            if not _is_edge_silent_bar(bars[index]):
                break
            indexes.add(index)

    return indexes
