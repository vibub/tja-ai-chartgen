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

    lines.extend(f"{bar.notes}," for bar in chart.bars)
    lines.append("#END")

    return "\n".join(lines) + "\n"
