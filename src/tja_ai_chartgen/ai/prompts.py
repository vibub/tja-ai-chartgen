import json
from typing import Any

from tja_ai_chartgen.tja.model import SongAnalysis


def build_chart_generation_payload(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    special_notes: bool = False,
) -> dict[str, Any]:
    return {
        "title": analysis.title,
        "artist": analysis.artist,
        "bpm": analysis.bpm,
        "offset": analysis.offset,
        "time_signature": analysis.time_signature,
        "course": course,
        "level": level,
        "style": style,
        "density": density,
        "special_notes": special_notes,
        "bars": [bar.model_dump() for bar in analysis.bars],
    }


def build_chart_generation_prompt(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    special_notes: bool = False,
) -> str:
    payload = build_chart_generation_payload(
        analysis,
        course,
        level,
        style,
        density,
        special_notes=special_notes,
    )

    return f"""
You are a Taiko no Tatsujin TJA chart draft generator.

Generate a playable draft chart.

Rules:
1. Output JSON only.
2. Do not include markdown.
3. Output exactly one notes string per input bar.
4. Each notes string length must equal that input bar's grids_per_bar value.
5. Allowed note characters: 0, 1, 2, 3, 4; when special_notes is true, 5, 7, and 8 are also allowed for simple drumrolls and balloons.
6. Do not use branches, BPM changes, delays, or scroll changes.
7. Do not make every bar full density.
8. Low energy bars should have more rests.
9. High energy bars can use denser patterns.
10. Respect the requested density: auto follows bar energy; low is sparse; medium is balanced; high is dense; max is the densest playable MVP draft.
11. For Oni 10, use technical but playable patterns.
12. Avoid repeating the exact same pattern for too many consecutive bars.
13. If special_notes is true and you use a balloon note 7, include balloon_counts with one positive integer per balloon note in that bar.

Input:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
{{
  "bars": [
    {{"bar": 1, "notes": "1000100010001000", "balloon_counts": []}}
  ]
}}
""".strip()
