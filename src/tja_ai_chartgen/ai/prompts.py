import json
from typing import Any

from tja_ai_chartgen.rules.styles import get_style_template
from tja_ai_chartgen.tja.model import SongAnalysis


def build_chart_generation_payload(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    special_notes: bool = False,
    reference_examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    template = get_style_template(style)
    payload = {
        "title": analysis.title,
        "artist": analysis.artist,
        "bpm": analysis.bpm,
        "offset": analysis.offset,
        "time_signature": analysis.time_signature,
        "course": course,
        "level": level,
        "style": style,
        "style_template": {
            "name": template.name,
            "description": template.description,
        },
        "density": density,
        "special_notes": special_notes,
        "bars": [bar.model_dump() for bar in analysis.bars],
    }
    if reference_examples:
        payload["reference_examples"] = reference_examples
    return payload


def build_chart_generation_prompt(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    special_notes: bool = False,
    reference_examples: list[dict[str, Any]] | None = None,
) -> str:
    payload = build_chart_generation_payload(
        analysis,
        course,
        level,
        style,
        density,
        special_notes=special_notes,
        reference_examples=reference_examples,
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
13. Use each bar's grid_features to align notes: onset=true marks likely playable hits, accent=true/downbeat=true marks stronger positions, strength is normalized 0.0-1.0.
14. Prefer stronger accents and downbeats for 1/3 notes, use 2/4 for lighter offbeat responses, and leave weak empty grids as 0 unless density asks for more.
15. Use beat_grids, downbeat_grid, phrase_position, and fill_candidate to shape musical phrasing; phrase_end/song_end bars may vary or fill, phrase_start bars should be stable.
16. If special_notes is true and you use a balloon note 7, include balloon_counts with one positive integer per balloon note in that bar.
17. If reference_examples are present in the input, use them as style and audio-alignment examples only: study how their audio_features map to reference_notes, but do not copy their note-string length. Your output notes must still match the requested input bars' grids_per_bar values.

Input:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
{{
  "bars": [
    {{"bar": 1, "notes": "1000100010001000", "balloon_counts": []}}
  ]
}}
""".strip()
