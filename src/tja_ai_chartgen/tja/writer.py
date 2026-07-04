from pathlib import Path

from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.tja.model import ChartBar, TjaChart

DEFAULT_BALLOON_COUNT = 8
TJA_FILE_ENCODING = "cp932"


def render_tja(chart: TjaChart) -> str:
    metadata = chart.metadata
    lines = [
        f"TITLE:{metadata.title}",
    ]

    if metadata.artist:
        lines.append(f"SUBTITLE:-- {metadata.artist}")

    lines.extend(
        [
            f"BPM:{metadata.bpm}",
            f"WAVE:{metadata.wave}",
            f"OFFSET:{_format_tja_offset(metadata.offset)}",
            f"COURSE:{metadata.course}",
            f"LEVEL:{metadata.level}",
            f"MAKER:{metadata.maker}",
        ]
    )

    balloon_counts = _collect_balloon_counts(chart.bars)
    if balloon_counts:
        lines.append(f"BALLOON:{','.join(str(count) for count in balloon_counts)}")

    lines.extend(["", "#START"])

    active_measure_ratio = "1/1"
    for bar in chart.bars:
        meter = get_meter_spec(bar.time_signature)
        if meter.measure_ratio != active_measure_ratio:
            lines.append(f"#MEASURE {meter.measure_ratio}")
            active_measure_ratio = meter.measure_ratio
        lines.append(f"{bar.notes},")

    if active_measure_ratio != "1/1":
        lines.append("#MEASURE 1/1")

    lines.append("#END")

    return "\n".join(lines) + "\n"


def write_tja_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=TJA_FILE_ENCODING, newline="\n")
    return path


def read_tja_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding=TJA_FILE_ENCODING)


def _format_tja_offset(internal_offset: float) -> float:
    tja_offset = -internal_offset
    if tja_offset == 0:
        return 0.0
    return tja_offset


def _collect_balloon_counts(bars: list[ChartBar]) -> list[int]:
    counts: list[int] = []
    for bar in bars:
        for balloon_index, _ in enumerate(position for position, note in enumerate(bar.notes) if note == "7"):
            if balloon_index < len(bar.balloon_counts):
                counts.append(bar.balloon_counts[balloon_index])
            else:
                counts.append(DEFAULT_BALLOON_COUNT)
    return counts
