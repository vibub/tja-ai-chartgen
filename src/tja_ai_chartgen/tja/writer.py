from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.tja.model import TjaChart


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
            f"OFFSET:{metadata.offset}",
            f"COURSE:{metadata.course}",
            f"LEVEL:{metadata.level}",
            f"MAKER:{metadata.maker}",
            "",
            "#START",
        ]
    )

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
