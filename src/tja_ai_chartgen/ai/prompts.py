import json
from typing import Any

from tja_ai_chartgen.ai.examples import get_reference_examples_prompt
from tja_ai_chartgen.rules.styles import get_style_template
from tja_ai_chartgen.tja.model import SongAnalysis


def build_chart_generation_payload(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    special_notes: bool = False,
    reference_examples_prompt: str | None = None,
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
    if reference_examples_prompt:
        payload["reference_examples_prompt"] = reference_examples_prompt
    return payload


def build_chart_generation_prompt(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    special_notes: bool = False,
    reference_examples_prompt: str | None = None,
) -> str:
    payload = build_chart_generation_payload(
        analysis,
        course,
        level,
        style,
        density,
        special_notes=special_notes,
        reference_examples_prompt=reference_examples_prompt or get_reference_examples_prompt(),
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
15. Big notes 3/4 require both hands hitting together. Use them sparingly as isolated accents on very strong downbeats or accents, preferably after a rest or sparse lead-in.
16. Do not place big notes 3/4 inside dense alternating streams. If a passage has 3 or more consecutive playable hits, use normal 1/2 notes in the stream instead of 3/4.
17. Avoid multiple big notes in one bar unless the bar is intentionally sparse; high/max density should increase 1/2 stream density, not big-note frequency.
18. Use beat_grids, downbeat_grid, phrase_position, and fill_candidate to shape musical phrasing; phrase_end/song_end bars may vary or fill, phrase_start bars should be stable.
19. If special_notes is true and you use a balloon note 7, include balloon_counts with one positive integer per balloon note in that bar.
20. If reference_examples_prompt is present in the input, use it as style and audio-alignment guidance only: study how its precomputed energy, onset, accent, beat, phrase, and section fields map to reference_notes, but do not copy reference note-string length. Your output notes must still match the requested input bars' grids_per_bar values.

Input:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
{{
  "bars": [
    {{"bar": 1, "notes": "1000100010001000", "balloon_counts": []}}
  ]
}}
""".strip()
