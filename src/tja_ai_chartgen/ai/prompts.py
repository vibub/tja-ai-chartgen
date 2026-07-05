import json
from typing import Any

from tja_ai_chartgen.ai.examples import get_reference_examples_prompt
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.rules.styles import get_style_template
from tja_ai_chartgen.tja.model import SongAnalysis


DENSITY_TARGETS = {
    "low": {
        "average_hits_per_16_grid_bar": "2-4",
        "normal_phrase_minimum_hits": 1,
        "description": "sparse beginner-friendly rhythm",
    },
    "medium": {
        "average_hits_per_16_grid_bar": "4-7",
        "normal_phrase_minimum_hits": 2,
        "description": "balanced rhythm with clear rests",
    },
    "high": {
        "average_hits_per_16_grid_bar": "8-11",
        "normal_phrase_minimum_hits": 4,
        "description": "dense advanced rhythm with short rests",
    },
    "max": {
        "average_hits_per_16_grid_bar": "10-13",
        "normal_phrase_minimum_hits": 5,
        "description": "densest playable MVP draft for expert charts",
    },
}


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
    density_target = DENSITY_TARGETS.get(density)
    if density == "auto":
        density_target = _auto_density_target(level)
    forced_silent_bars = [index + 1 for index in sorted(edge_silence_indexes(analysis.bars))]

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
        "density_target": density_target,
        "forced_silent_bars": forced_silent_bars,
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
    density_guidance = _density_guidance(density, level)

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
7. Bars listed in forced_silent_bars are song-start/song-end silence and must be all 0 for their full notes length, regardless of density, course, level, or grid 0 preference.
8. Do not make every bar full density.
9. Low energy bars should have more rests.
10. High energy bars can use denser patterns.
11. Respect the requested density: auto follows bar energy; low is sparse; medium is balanced; high is dense; max is the densest playable MVP draft.
12. Density and difficulty targets: {density_guidance}
13. For Oni 9-10, keep expert-level note volume through normal phrase and break sections; breaks may be simpler, but should still keep a playable beat skeleton unless they are clear song-start or song-end silence.
14. Do not output empty or 1-2 hit bars in the middle of high/max/Oni charts just because energy is low. Use beat_grids and downbeat_grid to add a stable skeleton on low-energy break bars.
15. Avoid repeating the exact same pattern for too many consecutive bars, and avoid reusing one notes string across many phrases. Keep a motif, but vary don/ka answers, offbeats, and phrase-end fills every 4-8 bars.
16. Use each bar's grid_features to align notes: onset=true marks likely playable hits, accent=true/downbeat=true marks stronger positions, strength is normalized 0.0-1.0.
17. Grid 0 is the barline and primary downbeat candidate. In normal phrase bars, prefer starting the bar with a 1/2 note on grid 0 even when onset=false, unless the bar is a pickup, song-start silence, song-end silence, or intentionally syncopated rest.
18. Prefer stronger accents and downbeats for 1/3 notes, use 2/4 for lighter offbeat responses, and leave weak empty grids as 0 unless density asks for more.
19. Big notes 3/4 require both hands hitting together. Use them sparingly as isolated accents on very strong downbeats or accents, preferably after a rest or sparse lead-in.
20. Do not place big notes 3/4 inside dense alternating streams. If a passage has 3 or more consecutive playable hits, use normal 1/2 notes in the stream instead of 3/4.
21. Avoid multiple big notes in one bar unless the bar is intentionally sparse; high/max density should increase 1/2 stream density, not big-note frequency.
22. Use beat_grids, downbeat_grid, phrase_position, and fill_candidate to shape musical phrasing; phrase_end/song_end bars may vary or fill, phrase_start bars should be stable.
23. If special_notes is true, use 5/8 drumrolls or 7 balloons only for occasional phrase-end/fill highlights, and include balloon_counts with one positive integer per 7.
24. If reference_examples_prompt is present in the input, use it as style and audio-alignment guidance only: study how its precomputed energy, onset, accent, beat, phrase, and section fields map to reference_notes, but do not copy reference note-string length. Your output notes must still match the requested input bars' grids_per_bar values.

Input:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
{{
  "bars": [
    {{"bar": 1, "notes": "1000100010001000", "balloon_counts": []}}
  ]
}}
""".strip()


def _auto_density_target(level: int) -> dict[str, Any]:
    if level >= 9:
        return DENSITY_TARGETS["high"]
    if level >= 7:
        return DENSITY_TARGETS["medium"]
    return DENSITY_TARGETS["low"]


def _density_guidance(density: str, level: int) -> str:
    target = DENSITY_TARGETS.get(density)
    if density == "auto":
        target = _auto_density_target(level)
    if not target:
        return "follow the requested course and level while keeping musical rests."

    return (
        f"target average playable hits per 16-grid bar is "
        f"{target['average_hits_per_16_grid_bar']}; "
        f"normal phrase bars should usually have at least "
        f"{target['normal_phrase_minimum_hits']} hits; "
        f"{target['description']}."
    )
