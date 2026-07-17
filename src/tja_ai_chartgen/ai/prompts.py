import json
from typing import Any

from tja_ai_chartgen.ai.examples import get_reference_examples_prompt
from tja_ai_chartgen.ai.reference_windows import get_reference_windows_payload
from tja_ai_chartgen.features.density import density_hint_payload
from tja_ai_chartgen.features.resolution import output_resolution_for_analysis_bar
from tja_ai_chartgen.features.rhythm_grid import is_stable_rhythmic_grid
from tja_ai_chartgen.features.salience import (
    RHYTHMIC_SALIENCE_FEATURE_VERSION,
    build_burst_salience,
    is_reliable_burst,
    project_reliable_burst_span,
)
from tja_ai_chartgen.features.salience_candidates import (
    SalienceCandidate,
    rank_bar_salience_candidates,
)
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.rules.rhythm_skeleton import (
    RHYTHM_SKELETON_VERSION,
    build_rhythm_skeleton,
)
from tja_ai_chartgen.rules.styles import get_style_template
from tja_ai_chartgen.tja.model import BarFeature, InstrumentBarFeature, SongAnalysis


STEM_ROLE_ACTIVITY_MINIMUM = 0.12
STEM_ROLE_MARGIN_MINIMUM = 0.08
AI_SALIENCE_MAX_POINTS_PER_BAR = 16
AI_SALIENCE_MAX_ARBITRARY_POINTS_PER_BAR = 1
AI_ARBITRARY_GRID_MIN_RESOLUTION_CONFIDENCE = 0.40

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

BAR_COLUMNS = [
    "index",
    "start_time",
    "end_time",
    "energy",
    "time_signature",
    "canonical_grids_per_bar",
    "output_resolution",
    "allowed_tick_step",
    "audio_channels",
    "accent_grids",
    "beat_events",
    "downbeat_grid",
    "phrase_position",
    "fill_candidate",
    "section",
]
AUDIO_CHANNEL_COLUMNS = [
    "onset_strength",
    "activity",
    "low_onset",
    "mid_onset",
    "high_onset",
    "spectral_flux",
    "vocal_onset",
    "drum_onset",
    "bass_onset",
    "accompaniment_onset",
]
BEAT_EVENT_COLUMNS = ["grid", "beat"]
SALIENCE_BAR_COLUMNS = [
    "bar_confidence",
    "fallback_reason",
    "burst_score",
    "burst_confidence",
    "burst_start_grid",
    "burst_end_grid",
    "burst_reliable",
    "points",
]
SALIENCE_POINT_COLUMNS = [
    "grid",
    "hit",
    "accent",
    "don_preference",
    "ka_preference",
    "sustained_activity",
    "confidence",
    "kind",
]
INSTRUMENT_BAR_COLUMNS = [
    "vocal_activity",
    "vocal_presence_ratio",
    "drum_activity",
    "bass_activity",
    "other_activity",
    "dominant_source",
    "confidence",
]
BAR_DENSITY_HINT_COLUMNS = [
    "bar",
    "kind",
    "min_hits",
    "max_hits",
    "allow_empty",
    "count_in_quality_average",
    "target_scale",
    "target_hits",
    "reason",
]
BAR_STRUCTURE_COLUMNS = [
    "energy_percentile",
    "energy_delta",
    "boundary_confidence",
    "phrase_id",
    "phrase_progress",
    "transition_role",
    "section_id",
    "section_confidence",
    "fill_candidate_score",
    "low_onset_strength",
    "mid_onset_strength",
    "high_onset_strength",
    "spectral_flux",
    "brightness",
    "harmonic_novelty",
    "texture_novelty",
    "percussive_ratio",
]
PHRASE_COLUMNS = [
    "phrase_id",
    "start_bar",
    "end_bar",
    "section_id",
    "section",
    "mean_energy",
    "energy_trend",
    "primary_role",
    "ending_boundary_confidence",
    "resolution",
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
    compact_salience = _compact_salience_bars(analysis)
    rhythm_skeleton = build_rhythm_skeleton(
        analysis,
        course=course,
        level=level,
        style=style,
        density=density,
    )

    payload = {
        "schema": "tja-ai-chartgen-compact-v7",
        "legend": {
            "bool": "0=false, 1=true",
            "bar_columns": BAR_COLUMNS,
            "audio_channel_columns": AUDIO_CHANNEL_COLUMNS,
            "audio_channel_scale": "-1=present with unknown strength; 0=absent; 1..1000=normalized strength",
            "audio_channel_positions": "channel index i maps to canonical tick i*allowed_tick_step",
            "beat_event_columns": BEAT_EVENT_COLUMNS,
            "salience_bar_columns": SALIENCE_BAR_COLUMNS,
            "salience_point_columns": SALIENCE_POINT_COLUMNS,
            "salience_scale": "0..1000=normalized strength/confidence; kind is the ranked evidence class",
            "rhythm_skeleton_semantics": (
                "one canonical-tick list per bar; deterministic audio-driven ordinary-hit "
                "timing authority; AI may change note colors and limited accents but should "
                "preserve these positions"
            ),
            "instrument_bar_columns": INSTRUMENT_BAR_COLUMNS,
            "instrument_bar_semantics": (
                "coarse stem roles only; dominant_source and confidence are derived from "
                "vocal/drum/bass/other activity; concrete instrument taxonomy is excluded"
            ),
            "bar_density_hint_columns": BAR_DENSITY_HINT_COLUMNS,
            "bar_structure_columns": BAR_STRUCTURE_COLUMNS,
            "phrase_columns": PHRASE_COLUMNS,
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
        "rhythmic_salience_feature_version": RHYTHMIC_SALIENCE_FEATURE_VERSION,
        "bar_salience": compact_salience,
        "rhythm_skeleton_version": RHYTHM_SKELETON_VERSION,
        "rhythm_skeleton": rhythm_skeleton,
        "spectral_feature_version": analysis.spectral_feature_version,
        "spectral_analysis_status": analysis.spectral_analysis_status,
        "instrument_feature_version": analysis.instrument_feature_version,
        "instrument_analysis_status": analysis.instrument_analysis_status,
        "bar_instruments": [_compact_bar_instrument(bar) for bar in analysis.bars],
        "structure_feature_version": analysis.structure_feature_version,
        "structure_confidence": _compact_number(analysis.structure_confidence or 0.0),
        "bar_structure": _compact_bar_structures(analysis),
        "phrase_plan": _compact_phrase_plan(analysis),
        "resolution_policy_version": analysis.resolution_policy_version,
        "bars": [
            _compact_bar(
                bar,
                output_resolution=output_resolution_for_analysis_bar(analysis, position),
            )
            for position, bar in enumerate(analysis.bars)
        ],
    }
    reference_windows = get_reference_windows_payload(course, level)
    if reference_examples_prompt:
        payload.update(_reference_examples_payload(reference_examples_prompt))
    elif reference_windows is not None:
        payload["legend"]["reference_window_bar_columns"] = reference_windows[
            "bar_columns"
        ]
        payload["reference_windows_note"] = (
            "Reference chart examples: anonymous continuous multi-course windows. Source event positions use "
            "[source_index,note] with each source bar's own resolution; project them by musical "
            "phase and never copy source resolution or exact note sequences."
        )
        payload["reference_windows"] = reference_windows["windows"]
    else:
        payload.update(_reference_examples_payload(get_reference_examples_prompt()))
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
        reference_examples_prompt=reference_examples_prompt,
    )
    density_guidance = _density_guidance(density, level)

    return f"""
You are a Taiko no Tatsujin TJA chart draft generator.

Generate a playable draft chart.

Rules:
1. Output JSON only.
2. Do not include markdown.
3. Output exactly one event object per input bar.
4. Use only hits and long_notes in each bar object. Never output the legacy notes or balloon_counts fields.
5. Normal hits use [canonical_tick, note] pairs in hits. note must be 1, 2, 3, or 4.
6. Decode each bars row with legend.bar_columns. Do not output a resolution. Use canonical_grids_per_bar, output_resolution, and allowed_tick_step. Every tick must be in 0..canonical_grids_per_bar-1 and divisible by allowed_tick_step.
7. When special_notes is true, long_notes may contain {{"start_tick":N,"end_tick":N,"kind":"drumroll"}} or {{"start_tick":N,"end_tick":N,"kind":"balloon","balloon_count":N}}. Otherwise long_notes must be empty.
8. Do not use branches, BPM changes, delays, or scroll changes.
9. Bars listed in forced_silent_bars are song-start/song-end silence and must have empty hits and long_notes, regardless of density, course, level, or grid 0 preference.
10. Do not make every bar full density.
11. Low energy bars should have more rests.
12. High energy bars can use denser patterns.
13. Respect the requested density: auto follows bar energy; low is sparse; medium is balanced; high is dense; max is the densest playable MVP draft.
14. Density and difficulty targets: {density_guidance}
15. Follow bar_density_hints using legend.bar_density_hint_columns. Keep each bar within min_hits/max_hits, but aim near target_hits for normal/dense/fill bars; target_hits is more important than merely satisfying min_hits when density_policy.quality_density is high or max. Silent/rest bars with max_hits=0 must be empty, rest bars with max_hits=2 may stay empty or tiny, and sparse bars should stay light.
16. For Oni 9-10, keep expert-level note volume through normal/dense/fill bars and keep the average over bars where count_in_quality_average=1 at or above density_policy.quality_average_min_per_16_grid_bar. Do not force notes into rest bars that represent actual musical pauses.
17. Avoid repeating the exact same event pattern for too many consecutive non-rest bars or across many phrases. Keep a motif, but vary don/ka answers, offbeats, and phrase-end fills every 4-8 bars.
18. Note colors are part of the chart design: 1/3 are don notes, 2/4 are ka notes. Do not use 1 as the default for every playable hit.
19. Let the chart's style and music decide the don/ka mix, but avoid outputs where nearly all normal 1/2 notes are 1. Use some 2 notes for offbeat responses, back-half answers, syncopated hits, or phrase-end fills.
20. When translated to four equal positions, useful rhythmic cells include 1020, 1200, 1012, 1210, 1122, 1221, 1022, and 2012, but do not force a fixed ratio.
21. Avoid long all-don streams such as 1010101010101010 unless the input clearly describes a very plain stamina passage; even then, vary later bars with occasional 2 notes.
22. Decode bar_salience with legend.salience_bar_columns and legend.salience_point_columns. Its sparse points and burst range are already filtered to ticks exactly representable at the bar's output_resolution. Prefer reliable strong-transient, transient, rhythmic-skeleton, and structure-highlight points before weak-evidence points or unsupported style connectors. rhythm_skeleton is the deterministic audio-driven timing authority, and ordinary AI hits are deterministically reconciled to it before acceptance. Follow its positions so your don/ka sequence, isolated big-note choices, and motif changes remain attached to the intended musical events instead of being discarded during reconciliation. A reliable long_note may replace skeleton hits inside its own span.
23. hit is the primary placement score. confidence and bar_confidence determine how strongly to trust it. Use accent only as soft emphasis evidence, and don_preference/ka_preference only as soft color evidence; style and playability still decide the final 1/2/3/4 note.
24. Meet density targets by adding supported connections around salience points, not by filling arbitrary empty ticks. Keep unsupported hits rare, avoid long unsupported streams, and leave weak empty grids as 0. Preserve a stable phrase-level rhythmic lattice: on 48/36 canonical grids, ordinary straight notes should normally use ticks divisible by 3, while triplet or 24th-note passages use ticks divisible by 2. Do not alternate nearby gaps such as 5/7 merely to follow detector microtiming. Ticks outside both stable subgrids must remain rare, repeated musical exceptions rather than per-hit timing corrections; positions explicitly present in rhythm_skeleton are evidence-backed exceptions and must not be shifted solely to satisfy divisibility. Stem onset microtiming only reinforces rhythm context and must not pull an otherwise regular motif by ±1 tick. High sustained_activity without transient points may justify only a simple beat/downbeat skeleton. Grid 0 is the barline and primary downbeat candidate; in normal phrase bars, prefer starting the bar with a 1/2 note on grid 0 only when salience or a justified skeleton supports it.
25. Big notes 3/4 require both hands hitting together. Use them sparingly as isolated accents on very strong downbeats or accents. A 3/4 note must be more than 0.25 seconds from every other playable 1/2/3/4 hit before and after it, including across bar boundaries; otherwise use the same-color normal note 1/2.
26. Do not place big notes 3/4 inside dense or rapid alternating-hand passages. If a passage has 3 or more consecutive playable hits, use normal 1/2 notes. Also treat fast two-hit cells equivalent to 102 or 1002 as alternating-hand patterns: when the real-time gap between those hits is 0.25 seconds or less, both hits must remain normal 1/2 notes, never 3/4. These examples describe timing after output resolution, not literal canonical tick distances.
27. Avoid multiple big notes in one bar unless each big note independently satisfies the 0.25-second isolation rule and the bar is intentionally sparse; high/max density should increase 1/2 stream density, not big-note frequency.
28. Decode beat_events with legend.beat_event_columns, and use it with downbeat_grid, phrase_position, and fill_candidate to shape musical phrasing; phrase_end/song_end bars may vary or fill, phrase_start bars should be stable.
29. If special_notes is true, use at most one long_note in a bar, and only when bar_salience burst_reliable is 1. Keep its start_tick and end_tick inside burst_start_grid..burst_end_grid. Structural fill_candidate may strengthen the choice but cannot replace the burst gate. Prefer drumroll for a shorter/moderate burst and balloon only for a strong, sustained highlight; balloon events require one positive balloon_count.
30. If reference_windows, reference_examples, or reference_examples_prompt is present, use it as style and audio-alignment guidance only. Source events may use a different resolution; convert their musical phase into canonical ticks and never copy source resolution or exact sequences.
31. Decode bar_structure with legend.bar_structure_columns and phrase_plan with legend.phrase_columns. These fields and resolution are read-only analysis; do not return modified structure data.
32. At phrase starts, establish a recognizable motif; in phrase middles, preserve it with small rhythmic and don/ka variations.
33. For build_up roles, increase activity gradually across the whole phrase according to phrase_progress instead of making only the final bar suddenly full.
34. For peak roles, emphasize ordinary hit density and strong accents while still obeying course speed caps; do not express every peak by stacking big notes.
35. Cadence may use a short fill, controlled variation, or deliberate space. Drop may use an impact followed by space or a denser entrance depending on the local features.
36. Breakdown lowers load but is not silence when activity remains present; keep a simple beat/downbeat skeleton.
37. Repeated section_id values should retain a recognizable base motif, with controlled later-song variation rather than exact copying.
38. Do not create a fill merely because a bar number is divisible by 4 or 8. Ordinary hits may follow fill_candidate_score and musical context, but long_notes additionally require the reliable rhythmic burst gate in bar_salience.
39. Decode audio_channels with legend.audio_channel_columns, audio_channel_scale, and audio_channel_positions only as supporting context around bar_salience. Treat low-frequency attacks as soft don evidence, high-frequency attacks as soft ka evidence, and spectral_flux as extra placement evidence. Use brightness, harmonic_novelty, texture_novelty, and percussive_ratio in bar_structure to recognize section changes without forcing a note on every spectral change.
40. Decode bar_instruments with legend.instrument_bar_columns for phrase, motif, and section context, not to choose exact note ticks. These fields contain only coarse vocal/drum/bass/accompaniment roles; do not infer concrete instrument taxonomy or map stems, vocals, or syllables directly to hits. Treat adjacent stem peaks such as 12/13 or 24/25 as one timing neighborhood and keep the stable motif grid instead of reproducing separation latency or onset smearing. Ignore low-confidence roles, and never let stem semantics override salience, silence, density, speed, occupancy, resolution, or playability constraints.

Input:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

Output schema:
{{
  "bars": [
    {{"bar":1,"hits":[[0,"1"],[12,"2"]],"long_notes":[]}}
  ]
}}
""".strip()


def _compact_salience_bars(analysis: SongAnalysis) -> list[list[Any]]:
    salience_bars = build_burst_salience(analysis.bars)
    compact: list[list[Any]] = []
    for position, (bar, salience) in enumerate(
        zip(analysis.bars, salience_bars, strict=True)
    ):
        output_resolution = output_resolution_for_analysis_bar(analysis, position)
        candidates = _select_ai_salience_candidates(
            bar,
            rank_bar_salience_candidates(
                bar,
                salience,
                output_resolution=output_resolution,
            ),
            allow_arbitrary_grids=_allow_ai_arbitrary_grids(
                analysis,
                output_resolution=output_resolution,
            ),
        )
        burst_span = project_reliable_burst_span(
            bar,
            salience,
            output_resolution=output_resolution,
        )
        compact.append(
            [
                _compact_unit(salience.confidence),
                salience.fallback_reason,
                _compact_unit(salience.burst_score),
                _compact_unit(salience.burst_confidence),
                burst_span[0] if burst_span is not None else None,
                burst_span[1] if burst_span is not None else None,
                int(is_reliable_burst(salience) and burst_span is not None),
                [
                    [
                        candidate.grid,
                        _compact_unit(candidate.point.hit),
                        _compact_unit(candidate.point.accent),
                        _compact_unit(candidate.point.don_preference),
                        _compact_unit(candidate.point.ka_preference),
                        _compact_unit(candidate.point.sustained_activity),
                        _compact_unit(candidate.point.confidence),
                        candidate.kind,
                    ]
                    for candidate in candidates
                ],
            ]
        )
    return compact


def _allow_ai_arbitrary_grids(
    analysis: SongAnalysis,
    *,
    output_resolution: int,
) -> bool:
    decision = analysis.resolution_plan.decision if analysis.resolution_plan else None
    return bool(
        decision is not None
        and output_resolution == analysis.resolution_plan.canonical_grids_per_bar
        and decision.confidence >= AI_ARBITRARY_GRID_MIN_RESOLUTION_CONFIDENCE
    )


def _select_ai_salience_candidates(
    bar: BarFeature,
    candidates: list[SalienceCandidate],
    *,
    allow_arbitrary_grids: bool,
) -> list[SalienceCandidate]:
    stable = [
        candidate
        for candidate in candidates
        if is_stable_rhythmic_grid(candidate.grid, bar.grids_per_bar)
    ]
    selected = stable[:AI_SALIENCE_MAX_POINTS_PER_BAR]
    if not allow_arbitrary_grids:
        return selected

    arbitrary = [
        candidate
        for candidate in candidates
        if not is_stable_rhythmic_grid(candidate.grid, bar.grids_per_bar)
        and candidate.point.confidence >= 0.85
        and "onset" in candidate.point.reasons
    ][:AI_SALIENCE_MAX_ARBITRARY_POINTS_PER_BAR]
    if not arbitrary:
        return selected

    selected = stable[: AI_SALIENCE_MAX_POINTS_PER_BAR - len(arbitrary)] + arbitrary
    order = {candidate.grid: position for position, candidate in enumerate(candidates)}
    return sorted(selected, key=lambda candidate: order[candidate.grid])


def _compact_bar_structures(analysis: SongAnalysis) -> list[list[Any]]:
    if analysis.bar_structures:
        return [
            [
                _compact_number(structure.energy_percentile),
                _compact_number(structure.energy_delta),
                _compact_number(structure.boundary_confidence),
                structure.phrase_id,
                _compact_number(structure.phrase_progress),
                structure.transition_role,
                structure.section_id,
                _compact_number(structure.section_confidence),
                _compact_number(structure.fill_candidate_score),
                _compact_number(structure.low_onset_strength),
                _compact_number(structure.mid_onset_strength),
                _compact_number(structure.high_onset_strength),
                _compact_number(structure.spectral_flux),
                _compact_number(structure.brightness),
                _compact_number(structure.harmonic_novelty),
                _compact_number(structure.texture_novelty),
                _compact_number(structure.percussive_ratio),
            ]
            for structure in analysis.bar_structures
        ]
    return [
        [
            _compact_number(bar.energy_percentile),
            _compact_number(bar.energy_delta),
            _compact_number(bar.boundary_confidence),
            bar.phrase_id,
            _compact_number(bar.phrase_progress),
            bar.transition_role,
            bar.section_id,
            _compact_number(bar.section_confidence),
            _compact_number(bar.fill_candidate_score),
            _compact_number(bar.low_onset_strength),
            _compact_number(bar.mid_onset_strength),
            _compact_number(bar.high_onset_strength),
            _compact_number(bar.spectral_flux),
            _compact_number(bar.brightness),
            _compact_number(bar.harmonic_novelty),
            _compact_number(bar.texture_novelty),
            _compact_number(bar.percussive_ratio),
        ]
        for bar in analysis.bars
    ]


def _compact_bar_instrument(bar: BarFeature) -> list[Any]:
    instrument = bar.instrument
    dominant_source, confidence = _stem_role_summary(instrument)
    return [
        _compact_number(instrument.vocal_activity),
        _compact_number(instrument.vocal_presence_ratio),
        _compact_number(instrument.drum_activity),
        _compact_number(instrument.bass_activity),
        _compact_number(instrument.other_activity),
        dominant_source,
        _compact_number(confidence),
    ]


def _stem_role_summary(instrument: InstrumentBarFeature) -> tuple[str | None, float]:
    activities = {
        "vocals": instrument.vocal_activity,
        "drums": instrument.drum_activity,
        "bass": instrument.bass_activity,
        "other": instrument.other_activity,
    }
    ranked = sorted(activities.items(), key=lambda item: (-item[1], item[0]))
    strongest_source, strongest = ranked[0]
    runner_up = ranked[1][1]
    margin = strongest - runner_up
    if strongest < STEM_ROLE_ACTIVITY_MINIMUM or margin < STEM_ROLE_MARGIN_MINIMUM:
        return None, 0.0
    confidence = min(1.0, strongest * 0.7 + margin * 0.6)
    return strongest_source, confidence


def _compact_phrase_plan(analysis: SongAnalysis) -> list[list[Any]]:
    return [
        [
            phrase.phrase_id,
            phrase.start_bar,
            phrase.end_bar,
            phrase.section_id,
            phrase.section,
            _compact_number(phrase.mean_energy),
            _compact_number(phrase.energy_trend),
            phrase.primary_role,
            _compact_number(phrase.ending_boundary_confidence),
            phrase.resolution,
        ]
        for phrase in analysis.phrase_plan
    ]


def _compact_bar(bar: BarFeature, *, output_resolution: int) -> list[Any]:
    return [
        bar.index,
        _compact_number(bar.start_time),
        _compact_number(bar.end_time),
        _compact_number(bar.energy),
        bar.time_signature,
        bar.grids_per_bar,
        output_resolution,
        bar.grids_per_bar // output_resolution,
        _compact_audio_channels(bar, output_resolution=output_resolution),
        bar.accent_grids,
        _compact_beat_events(bar),
        bar.downbeat_grid,
        bar.phrase_position,
        int(bar.fill_candidate),
        bar.section,
    ]


def _compact_audio_channels(bar: BarFeature, *, output_resolution: int) -> list[list[int]]:
    channels = [[0] * output_resolution for _ in AUDIO_CHANNEL_COLUMNS]
    step = bar.grids_per_bar // output_resolution
    onset_strengths = {
        feature.grid: feature.strength for feature in bar.grid_features if feature.onset
    }
    for grid in sorted(set(bar.onset_grids) | set(onset_strengths)):
        slot = _projected_slot(grid, step=step, output_resolution=output_resolution)
        strength = onset_strengths.get(grid)
        if strength is None:
            if channels[0][slot] == 0:
                channels[0][slot] = -1
        else:
            channels[0][slot] = max(channels[0][slot], max(1, _compact_unit(strength)))

    activity_values = bar.activity_grids or [feature.activity for feature in bar.grid_features]
    for grid, value in enumerate(activity_values):
        _set_projected_channel(
            channels[1],
            grid=grid,
            value=value,
            step=step,
            output_resolution=output_resolution,
        )

    for feature in bar.spectral_grid_features:
        for channel, value in zip(
            channels[2:6],
            (
                feature.low_onset_strength,
                feature.mid_onset_strength,
                feature.high_onset_strength,
                feature.spectral_flux,
            ),
            strict=True,
        ):
            _set_projected_channel(
                channel,
                grid=feature.grid,
                value=value,
                step=step,
                output_resolution=output_resolution,
            )

    for feature in bar.instrument_grid_features:
        for channel, value in zip(
            channels[6:10],
            (
                feature.vocal_onset,
                feature.drum_onset,
                feature.bass_onset,
                feature.accompaniment_onset,
            ),
            strict=True,
        ):
            _set_projected_channel(
                channel,
                grid=feature.grid,
                value=value,
                step=step,
                output_resolution=output_resolution,
            )
    return channels


def _set_projected_channel(
    channel: list[int],
    *,
    grid: int,
    value: float,
    step: int,
    output_resolution: int,
) -> None:
    slot = _projected_slot(grid, step=step, output_resolution=output_resolution)
    channel[slot] = max(channel[slot], _compact_unit(value))


def _projected_slot(grid: int, *, step: int, output_resolution: int) -> int:
    return min(output_resolution - 1, max(0, (grid + step // 2) // step))


def _compact_unit(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return round(min(1.0, max(0.0, number)) * 1000)


def _compact_beat_events(bar: BarFeature) -> list[list[int]]:
    beats = {
        feature.grid: feature.beat
        for feature in bar.grid_features
        if feature.beat is not None
    }
    if beats:
        return [[grid, beat] for grid, beat in sorted(beats.items())]
    return [[grid, number] for number, grid in enumerate(bar.beat_grids, start=1)]


def _compact_density_hints(bars: list[BarFeature], *, quality_density: str) -> list[list[Any]]:
    return [
        [
            hint["bar"],
            hint["kind"],
            hint["min_hits"],
            hint["max_hits"],
            int(bool(hint["allow_empty"])),
            int(bool(hint["count_in_quality_average"])),
            _compact_number(float(hint["target_scale"])),
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
    target_scale = float(hint.get("target_scale", 1.0))

    if kind == "silent" or max_value == 0:
        return 0
    if kind == "rest":
        return min(max_value or 1, 1)
    if kind == "sparse":
        base_target = max(min_hits, 2 if quality_density in {"high", "max"} else 1)
    elif quality_density == "max":
        base_target = {"normal": 8, "dense": 11, "fill": 9}.get(kind, min_hits)
    elif quality_density == "high":
        base_target = {"normal": 6, "dense": 10, "fill": 8}.get(kind, min_hits)
    elif quality_density == "medium":
        base_target = {"normal": 4, "dense": 7, "fill": 6}.get(kind, min_hits)
    else:
        base_target = {"normal": 3, "dense": 5, "fill": 4}.get(kind, min_hits)

    target = max(min_hits, round(base_target * target_scale))
    return _clamp_hits(target, max_value)


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
