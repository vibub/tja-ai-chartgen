import json
from typing import Any

from tja_ai_chartgen.ai.examples import get_reference_examples_prompt
from tja_ai_chartgen.features.density import density_hint_payload
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.rules.styles import get_style_template
from tja_ai_chartgen.tja.model import BarFeature, GridFeature, SongAnalysis


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

GRID_FEATURE_COLUMNS = ["grid", "onset", "accent", "beat", "downbeat", "strength", "activity"]
BAR_DENSITY_HINT_COLUMNS = [
    "bar",
    "kind",
    "min_hits",
    "max_hits",
    "allow_empty",
    "count_in_quality_average",
    "target_hits",
    "reason",
]
REFERENCE_EXAMPLE_BAR_COLUMNS = [
    "bar",
    "energy",
    "grids_per_bar",
    "onset_16",
    "accent_16",
    "beat_grids",
    "downbeat_grid",
    "phrase_position",
    "fill_candidate",
    "section",
    "reference_notes",
    "reference_note_resolution",
    "balloon_counts",
]


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
        "schema": "tja-ai-chartgen-compact-v1",
        "legend": {
            "bool": "0=false, 1=true",
            "grid_feature_columns": GRID_FEATURE_COLUMNS,
            "bar_density_hint_columns": BAR_DENSITY_HINT_COLUMNS,
            "reference_example_bar_columns": REFERENCE_EXAMPLE_BAR_COLUMNS,
        },
        "title": analysis.title,
        "artist": analysis.artist,
        "bpm": _compact_number(analysis.bpm),
        "offset": _compact_number(analysis.offset),
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
        "density_policy": _density_policy(density, course, level),
        "forced_silent_bars": forced_silent_bars,
        "bar_density_hints": _compact_density_hints(
            analysis.bars,
            quality_density=_quality_density(density, course, level),
        ),
        "special_notes": special_notes,
        "bars": [_compact_bar(bar) for bar in analysis.bars],
    }
    if reference_examples_prompt:
        payload.update(_reference_examples_payload(reference_examples_prompt))
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
13. Follow bar_density_hints using legend.bar_density_hint_columns. Keep each bar within min_hits/max_hits, but aim near target_hits for normal/dense/fill bars; target_hits is more important than merely satisfying min_hits when density_policy.quality_density is high or max. Silent/rest bars with max_hits=0 must be empty, rest bars with max_hits=2 may stay empty or tiny, and sparse bars should stay light.
14. For Oni 9-10, keep expert-level note volume through normal/dense/fill bars and keep the average over bars where count_in_quality_average=1 at or above density_policy.quality_average_min_per_16_grid_bar. Do not force notes into rest bars that represent actual musical pauses.
15. Avoid repeating the exact same pattern for too many consecutive non-rest bars, and avoid reusing one notes string across many phrases. Keep a motif, but vary don/ka answers, offbeats, and phrase-end fills every 4-8 bars.
16. Note colors are part of the chart design: 1/3 are don notes, 2/4 are ka notes. Do not use 1 as the default for every playable hit.
17. Let the chart's style and music decide the don/ka mix, but avoid outputs where nearly all normal 1/2 notes are 1. Use some 2 notes for offbeat responses, back-half answers, syncopated hits, or phrase-end fills.
18. Common useful cells include 1020, 1200, 1012, 1210, 1122, 1221, 1022, and 2012, but do not force a fixed ratio.
19. Avoid long all-don streams such as 1010101010101010 unless the input clearly describes a very plain stamina passage; even then, vary later bars with occasional 2 notes.
20. Use each bar's grid_features to align notes. Decode each row with legend.grid_feature_columns: onset=1 marks likely playable hits, activity/strength mark sustained musical sound such as vocals, guitar, strings, piano, or pads even when onset=0. If activity is high but onset is sparse, this is not a rest; place a simple beat/downbeat skeleton rather than leaving the bar empty.
21. Grid 0 is the barline and primary downbeat candidate. In normal phrase bars, prefer starting the bar with a 1/2 note on grid 0 even when onset=0, unless the bar is a pickup, song-start silence, song-end silence, or intentionally syncopated rest.
22. Prefer stronger accents and downbeats for 1/3 notes, use 2/4 for lighter offbeat responses, and leave weak empty grids as 0 unless density or sustained activity asks for more.
23. Big notes 3/4 require both hands hitting together. Use them sparingly as isolated accents on very strong downbeats or accents, preferably after a rest or sparse lead-in.
24. Do not place big notes 3/4 inside dense streams. If a passage has 3 or more consecutive playable hits, use normal 1/2 notes in the stream instead of 3/4.
25. Avoid multiple big notes in one bar unless the bar is intentionally sparse; high/max density should increase 1/2 stream density, not big-note frequency.
26. Use beat_grids, downbeat_grid, phrase_position, and fill_candidate to shape musical phrasing; phrase_end/song_end bars may vary or fill, phrase_start bars should be stable.
27. If special_notes is true, use 5/8 drumrolls or 7 balloons only for occasional phrase-end/fill highlights, and include balloon_counts with one positive integer per 7.
28. If reference_examples or reference_examples_prompt is present in the input, use it as style and audio-alignment guidance only: study how its precomputed energy, onset, accent, beat, phrase, and section fields map to reference_notes, but do not copy reference note-string length. Your output notes must still match the requested input bars' grids_per_bar values.

Input:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

Output schema:
{{
  "bars": [
    {{"bar": 1, "notes": "1000100010001000", "balloon_counts": []}}
  ]
}}
""".strip()


def _compact_bar(bar: BarFeature) -> dict[str, Any]:
    return {
        "index": bar.index,
        "start_time": _compact_number(bar.start_time),
        "end_time": _compact_number(bar.end_time),
        "energy": _compact_number(bar.energy),
        "time_signature": bar.time_signature,
        "grids_per_bar": bar.grids_per_bar,
        "onset_16": bar.onset_16,
        "accent_16": bar.accent_16,
        "activity_16": [_compact_number(value) for value in bar.activity_16],
        "grid_features": [_compact_grid_feature(feature) for feature in bar.grid_features],
        "beat_grids": bar.beat_grids,
        "downbeat_grid": bar.downbeat_grid,
        "phrase_position": bar.phrase_position,
        "fill_candidate": bar.fill_candidate,
        "section": bar.section,
    }


def _compact_grid_feature(feature: GridFeature) -> list[Any]:
    return [
        feature.grid,
        int(feature.onset),
        int(feature.accent),
        feature.beat,
        int(feature.downbeat),
        _compact_number(feature.strength),
        _compact_number(feature.activity),
    ]


def _compact_density_hints(bars: list[BarFeature], *, quality_density: str) -> list[list[Any]]:
    return [
        [
            hint["bar"],
            hint["kind"],
            hint["min_hits"],
            hint["max_hits"],
            int(bool(hint["allow_empty"])),
            int(bool(hint["count_in_quality_average"])),
            _density_hint_target_hits(hint, quality_density=quality_density),
            hint["reason"],
        ]
        for hint in density_hint_payload(bars)
    ]


def _density_policy(density: str, course: str, level: int) -> dict[str, Any]:
    quality_density = _quality_density(density, course, level)
    return {
        "quality_density": quality_density,
        "quality_average_min_per_16_grid_bar": _quality_average_min(quality_density),
        "target_hits_rule": (
            "Use bar_density_hints target_hits as the normal/dense/fill aim; min_hits is only the floor. "
            "For high/max, stay near target_hits except real rest/sparse/silent bars."
        ),
    }


def _quality_density(density: str, course: str, level: int) -> str:
    if density in {"high", "max"}:
        return density
    if density == "auto" and course.lower() == "oni":
        if level >= 10:
            return "max"
        if level >= 8:
            return "high"
    return density


def _quality_average_min(quality_density: str) -> float:
    if quality_density == "max":
        return 8.5
    if quality_density == "high":
        return 6.5
    if quality_density == "medium":
        return 3.5
    return 0.0


def _density_hint_target_hits(hint: dict[str, object], *, quality_density: str) -> int:
    kind = str(hint["kind"])
    min_hits = int(hint["min_hits"])
    max_hits = hint["max_hits"]
    max_value = int(max_hits) if isinstance(max_hits, int) else None

    if kind == "silent" or max_value == 0:
        return 0
    if kind == "rest":
        return min(max_value or 1, 1)
    if kind == "sparse":
        target = max(min_hits, 2 if quality_density in {"high", "max"} else 1)
        return _clamp_hits(target, max_value)

    if quality_density == "max":
        defaults = {"normal": 8, "dense": 11, "fill": 9}
    elif quality_density == "high":
        defaults = {"normal": 6, "dense": 10, "fill": 8}
    elif quality_density == "medium":
        defaults = {"normal": 4, "dense": 7, "fill": 6}
    else:
        defaults = {"normal": 3, "dense": 5, "fill": 4}

    return _clamp_hits(max(min_hits, defaults.get(kind, min_hits)), max_value)


def _clamp_hits(value: int, max_hits: int | None) -> int:
    if max_hits is None:
        return value
    return min(value, max_hits)


def _reference_examples_payload(reference_examples_prompt: str) -> dict[str, Any]:
    examples = _parse_reference_examples(reference_examples_prompt)
    if examples is None:
        return {"reference_examples_prompt": reference_examples_prompt}

    return {
        "reference_examples_note": (
            "Reference chart examples, precomputed from study charts. These examples are evenly sampled "
            "across each full chart, not only intros. Use them only as style and audio-alignment examples."
        ),
        "reference_examples": [_compact_reference_example(example) for example in examples],
    }


def _parse_reference_examples(reference_examples_prompt: str) -> list[dict[str, Any]] | None:
    start = reference_examples_prompt.find("[")
    end = reference_examples_prompt.rfind("]")
    if start < 0 or end <= start:
        return None

    try:
        data = json.loads(reference_examples_prompt[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or not all(isinstance(example, dict) for example in data):
        return None
    return data


def _compact_reference_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": example.get("title"),
        "artist": example.get("artist"),
        "course": example.get("course"),
        "level": example.get("level"),
        "sample_strategy": example.get("sample_strategy"),
        "total_chart_bars": example.get("total_chart_bars"),
        "total_audio_bars": example.get("total_audio_bars"),
        "bars": [_compact_reference_bar(bar) for bar in example.get("bars", []) if isinstance(bar, dict)],
    }


def _compact_reference_bar(bar: dict[str, Any]) -> list[Any]:
    return [
        bar.get("bar"),
        _compact_number(bar.get("energy")),
        bar.get("grids_per_bar"),
        bar.get("onset_16", []),
        bar.get("accent_16", []),
        bar.get("beat_grids", []),
        bar.get("downbeat_grid"),
        bar.get("phrase_position"),
        int(bool(bar.get("fill_candidate"))),
        bar.get("section"),
        bar.get("reference_notes"),
        bar.get("reference_note_resolution"),
        bar.get("balloon_counts", []),
    ]


def _compact_number(value: Any) -> Any:
    if not isinstance(value, int | float):
        return value
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return int(rounded)
    return rounded


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
