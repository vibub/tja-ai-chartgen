from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean

from tja_ai_chartgen.features.silence import is_silent_bar
from tja_ai_chartgen.tja.model import BarFeature, BarStructureFeature, PhraseFeature


STRUCTURE_FEATURE_VERSION = "structure-v1"
PROFILE_SIZE = 12
MIN_PHRASE_BARS = 2
MAX_PHRASE_BARS = 16
BOUNDARY_THRESHOLD = 0.40
FILL_CANDIDATE_THRESHOLD = 0.60
SECTION_SIMILARITY_THRESHOLD = 0.82


@dataclass(frozen=True)
class StructureAnalysisResult:
    bars: list[BarFeature]
    bar_structures: list[BarStructureFeature]
    phrases: list[PhraseFeature]
    confidence: float


@dataclass
class _PhraseDraft:
    phrase_id: int
    start_bar: int
    end_bar: int
    mean_energy: float
    peak_energy: float
    energy_trend: float
    primary_role: str
    ending_boundary_confidence: float
    signature: list[float]
    section_index: int = 0
    section: str = "unknown"
    section_confidence: float = 0.0


def analyze_song_structure(bars: list[BarFeature]) -> StructureAnalysisResult:
    if not bars:
        return StructureAnalysisResult([], [], [], 0.0)

    structures = _build_structure_vectors(bars)
    boundary_scores = _boundary_scores(structures)
    boundaries = _select_boundaries(boundary_scores, len(bars))
    structures, phrase_drafts = _assign_phrases(
        structures,
        boundaries,
        boundary_scores,
    )
    structures, phrase_drafts = _assign_transition_roles(structures, phrase_drafts)
    structures, phrase_drafts = _assign_sections(structures, phrase_drafts)
    structures = _assign_fill_candidates(structures, phrase_drafts)
    updated_bars = _apply_structure_to_bars(bars, structures)
    phrases = _build_phrase_features(phrase_drafts)
    confidence = _structure_confidence(boundary_scores, boundaries)
    return StructureAnalysisResult(
        bars=updated_bars,
        bar_structures=structures,
        phrases=phrases,
        confidence=confidence,
    )


def _build_structure_vectors(bars: list[BarFeature]) -> list[BarStructureFeature]:
    energies = [bar.energy for bar in bars]
    percentiles = [_percentile_rank(value, energies) for value in energies]
    edge_silent_indexes = _edge_silent_indexes(bars)
    structures: list[BarStructureFeature] = []

    for position, bar in enumerate(bars):
        grid_count = max(1, bar.grids_per_bar)
        onset_strengths = [
            feature.strength
            for feature in bar.grid_features
            if feature.onset and 0 <= feature.grid < grid_count
        ]
        if not onset_strengths and bar.onset_grids:
            onset_strengths = [1.0] * len(bar.onset_grids)
        activity_values = [
            max(0.0, min(1.0, value)) for value in bar.activity_grids[:grid_count]
        ]
        if len(activity_values) < grid_count:
            activity_values.extend([0.0] * (grid_count - len(activity_values)))
        rhythm_values = [0.0] * grid_count
        for grid in bar.onset_grids:
            if 0 <= grid < grid_count:
                rhythm_values[grid] = 1.0
        for feature in bar.grid_features:
            if feature.onset and 0 <= feature.grid < grid_count:
                rhythm_values[feature.grid] = max(rhythm_values[feature.grid], feature.strength)

        energy_delta = percentiles[position] - percentiles[position - 1] if position else 0.0
        structures.append(
            BarStructureFeature(
                index=bar.index,
                energy_percentile=_rounded(percentiles[position]),
                energy_delta=_rounded(energy_delta),
                onset_density=_rounded(len(set(bar.onset_grids)) / grid_count),
                onset_strength_mean=_rounded(mean(onset_strengths) if onset_strengths else 0.0),
                onset_strength_max=_rounded(max(onset_strengths, default=0.0)),
                activity_mean=_rounded(mean(activity_values) if activity_values else 0.0),
                activity_peak=_rounded(max(activity_values, default=0.0)),
                active_grid_ratio=_rounded(
                    sum(value >= 0.18 for value in activity_values) / grid_count
                ),
                accent_density=_rounded(len(set(bar.accent_grids)) / grid_count),
                low_onset_strength=_rounded(bar.low_onset_strength),
                mid_onset_strength=_rounded(bar.mid_onset_strength),
                high_onset_strength=_rounded(bar.high_onset_strength),
                spectral_flux=_rounded(bar.spectral_flux),
                brightness=_rounded(bar.brightness),
                harmonic_novelty=_rounded(bar.harmonic_novelty),
                texture_novelty=_rounded(bar.texture_novelty),
                percussive_ratio=_rounded(bar.percussive_ratio),
                rhythm_profile=_resample_profile(rhythm_values, use_max=True),
                activity_profile=_resample_profile(activity_values, use_max=False),
                edge_silent=position in edge_silent_indexes,
            )
        )
    return structures


def _edge_silent_indexes(bars: list[BarFeature]) -> set[int]:
    indexes: set[int] = set()
    for position, bar in enumerate(bars):
        if not is_silent_bar(bar):
            break
        indexes.add(position)
    for position in range(len(bars) - 1, -1, -1):
        if not is_silent_bar(bars[position]):
            break
        indexes.add(position)
    return indexes


def _percentile_rank(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    lower = sum(candidate < value for candidate in values)
    equal = sum(candidate == value for candidate in values)
    return (lower + equal / 2) / len(values)


def _resample_profile(values: list[float], *, use_max: bool) -> list[float]:
    if not values:
        return [0.0] * PROFILE_SIZE
    result: list[float] = []
    for target in range(PROFILE_SIZE):
        start = round(target * len(values) / PROFILE_SIZE)
        end = round((target + 1) * len(values) / PROFILE_SIZE)
        segment = values[start : max(start + 1, end)]
        value = max(segment, default=0.0) if use_max else mean(segment)
        result.append(_rounded(value))
    return result


def _boundary_scores(structures: list[BarStructureFeature]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for position in range(1, len(structures)):
        left = structures[max(0, position - 3) : position]
        right = structures[position : min(len(structures), position + 3)]
        energy_difference = abs(
            mean(item.energy_percentile for item in left)
            - mean(item.energy_percentile for item in right)
        )
        onset_difference = abs(
            mean(_onset_drive(item) for item in left)
            - mean(_onset_drive(item) for item in right)
        )
        activity_difference = abs(
            mean(item.activity_mean for item in left)
            - mean(item.activity_mean for item in right)
        )
        rhythm_difference = _profile_distance(
            _mean_profile([item.rhythm_profile for item in left]),
            _mean_profile([item.rhythm_profile for item in right]),
        )
        direct_difference = abs(
            _combined_drive(structures[position])
            - _combined_drive(structures[position - 1])
        )
        score = (
            energy_difference * 0.30
            + onset_difference * 0.20
            + activity_difference * 0.15
            + rhythm_difference * 0.25
            + direct_difference * 0.10
        )
        if any(_has_spectral_evidence(item) for item in [*left, *right]):
            timbre_difference = _profile_distance(
                _mean_profile([_timbre_profile(item) for item in left]),
                _mean_profile([_timbre_profile(item) for item in right]),
            )
            novelty = max(
                structures[position].harmonic_novelty,
                structures[position].texture_novelty,
            )
            spectral_boundary = min(1.0, timbre_difference * 0.65 + novelty * 0.35)
            score = max(score, score * 0.75 + spectral_boundary * 0.55)
        if structures[position - 1].edge_silent != structures[position].edge_silent:
            score = max(score, 0.9)
        scores[position] = _rounded(min(1.0, score))
    return scores


def _onset_drive(structure: BarStructureFeature) -> float:
    return min(1.0, structure.onset_density * 4.0)


def _combined_drive(structure: BarStructureFeature) -> float:
    base = (
        structure.energy_percentile * 0.45
        + _onset_drive(structure) * 0.35
        + structure.activity_mean * 0.20
    )
    if not _has_spectral_evidence(structure):
        return base
    spectral_drive = (
        structure.spectral_flux * 0.45
        + structure.percussive_ratio * 0.30
        + max(
            structure.low_onset_strength,
            structure.mid_onset_strength,
            structure.high_onset_strength,
        )
        * 0.25
    )
    return min(1.0, base + spectral_drive * 0.12)


def _has_spectral_evidence(structure: BarStructureFeature) -> bool:
    return any(
        value > 0.0
        for value in (
            structure.low_onset_strength,
            structure.mid_onset_strength,
            structure.high_onset_strength,
            structure.spectral_flux,
            structure.brightness,
            structure.harmonic_novelty,
            structure.texture_novelty,
            structure.percussive_ratio,
        )
    )


def _timbre_profile(structure: BarStructureFeature) -> list[float]:
    return [
        structure.brightness,
        structure.percussive_ratio,
        structure.low_onset_strength,
        structure.mid_onset_strength,
        structure.high_onset_strength,
    ]


def _spectral_intensity(structure: BarStructureFeature) -> float:
    return min(
        1.0,
        structure.spectral_flux * 0.35
        + structure.percussive_ratio * 0.25
        + structure.brightness * 0.15
        + max(
            structure.low_onset_strength,
            structure.mid_onset_strength,
            structure.high_onset_strength,
        )
        * 0.25,
    )


def _mean_profile(profiles: list[list[float]]) -> list[float]:
    if not profiles:
        return [0.0] * PROFILE_SIZE
    width = len(profiles[0])
    return [mean(profile[index] for profile in profiles) for index in range(width)]


def _profile_distance(first: list[float], second: list[float]) -> float:
    if not first or not second:
        return 0.0
    return mean(abs(left - right) for left, right in zip(first, second, strict=True))


def _select_boundaries(scores: dict[int, float], bar_count: int) -> list[int]:
    candidates = [
        position
        for position, score in scores.items()
        if score >= BOUNDARY_THRESHOLD
        and score >= scores.get(position - 1, -1.0)
        and score >= scores.get(position + 1, -1.0)
    ]
    selected: list[int] = []
    for position in candidates:
        if position < MIN_PHRASE_BARS or bar_count - position < MIN_PHRASE_BARS:
            continue
        if selected and position - selected[-1] < MIN_PHRASE_BARS:
            if scores[position] > scores[selected[-1]]:
                selected[-1] = position
            continue
        selected.append(position)

    boundaries = [0, *selected, bar_count]
    changed = True
    while changed:
        changed = False
        expanded = [boundaries[0]]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if end - start <= MAX_PHRASE_BARS:
                expanded.append(end)
                continue
            possible = range(start + MIN_PHRASE_BARS, min(end - MIN_PHRASE_BARS, start + MAX_PHRASE_BARS) + 1)
            split = max(
                possible,
                key=lambda position: (
                    scores.get(position, 0.0) + _length_prior(position - start),
                    -abs((position - start) - 8),
                ),
            )
            expanded.extend([split, end])
            changed = True
        boundaries = sorted(set(expanded))
    return boundaries


def _length_prior(length: int) -> float:
    if length == 8:
        return 0.04
    if length == 4:
        return 0.02
    return 0.0


def _assign_phrases(
    structures: list[BarStructureFeature],
    boundaries: list[int],
    boundary_scores: dict[int, float],
) -> tuple[list[BarStructureFeature], list[_PhraseDraft]]:
    assigned = list(structures)
    phrases: list[_PhraseDraft] = []
    for phrase_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        phrase_structures = structures[start:end]
        energies = [item.energy_percentile for item in phrase_structures]
        energy_trend = _linear_trend(energies)
        length = end - start
        ending_confidence = 1.0 if end == len(structures) else 0.0
        if end < len(structures):
            ending_confidence = boundary_scores.get(end, 0.0)
        signature = _phrase_signature(phrase_structures)
        phrases.append(
            _PhraseDraft(
                phrase_id=phrase_id,
                start_bar=start,
                end_bar=end - 1,
                mean_energy=mean(energies),
                peak_energy=max(energies, default=0.0),
                energy_trend=energy_trend,
                primary_role=_phrase_role(phrase_structures, energy_trend),
                ending_boundary_confidence=ending_confidence,
                signature=signature,
            )
        )
        for offset, position in enumerate(range(start, end)):
            if length == 1:
                phrase_position = "single"
                progress = 0.0
            elif offset == 0:
                phrase_position = "start"
                progress = 0.0
            elif offset == length - 1:
                phrase_position = "end"
                progress = 1.0
            else:
                phrase_position = "middle"
                progress = offset / (length - 1)
            boundary_confidence = ending_confidence if position == end - 1 else 0.0
            assigned[position] = assigned[position].model_copy(
                update={
                    "boundary_confidence": _rounded(boundary_confidence),
                    "phrase_id": phrase_id,
                    "phrase_progress": _rounded(progress),
                    "phrase_position": phrase_position,
                }
            )
    return assigned, phrases


def _linear_trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = (len(values) - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    if denominator <= 0:
        return 0.0
    slope = sum((index - center) * value for index, value in enumerate(values)) / denominator
    return max(-1.0, min(1.0, slope))


def _phrase_role(structures: list[BarStructureFeature], energy_trend: float) -> str:
    if not structures:
        return "stable"
    mean_energy = mean(item.energy_percentile for item in structures)
    mean_activity = mean(item.activity_mean for item in structures)
    mean_onset = mean(item.onset_density for item in structures)
    has_spectral_evidence = any(_has_spectral_evidence(item) for item in structures)
    mean_percussive = mean(item.percussive_ratio for item in structures)
    mean_flux = mean(item.spectral_flux for item in structures)
    spectral_trend = _linear_trend([_spectral_intensity(item) for item in structures])
    spectral_breakdown = (
        has_spectral_evidence
        and mean_percussive <= 0.25
        and mean_flux <= 0.35
    )
    if mean_activity >= 0.18 and (
        (mean_energy <= 0.4 and mean_onset <= 0.08)
        or (spectral_breakdown and mean_onset <= 0.12)
    ):
        return "breakdown"
    if energy_trend >= 0.055 or (
        has_spectral_evidence
        and energy_trend >= -0.02
        and spectral_trend >= 0.055
    ):
        return "build_up"
    if (
        mean_energy >= 0.72
        or max(item.energy_percentile for item in structures) >= 0.88
        or (
            has_spectral_evidence
            and mean_energy >= 0.6
            and mean(_spectral_intensity(item) for item in structures) >= 0.7
        )
    ):
        return "peak"
    return "stable"


def _phrase_signature(structures: list[BarStructureFeature]) -> list[float]:
    return [
        mean(item.energy_percentile for item in structures),
        mean(_onset_drive(item) for item in structures),
        mean(item.activity_mean for item in structures),
        *_mean_profile([item.rhythm_profile for item in structures]),
    ]


def _assign_transition_roles(
    structures: list[BarStructureFeature],
    phrases: list[_PhraseDraft],
) -> tuple[list[BarStructureFeature], list[_PhraseDraft]]:
    assigned = list(structures)
    phrase_by_id = {phrase.phrase_id: phrase for phrase in phrases}
    for position, structure in enumerate(structures):
        phrase = phrase_by_id[structure.phrase_id]
        previous = structures[position - 1] if position else None
        previous_drive = _combined_drive(previous) if previous is not None else 0.0
        drive = _combined_drive(structure)
        drive_delta = drive - previous_drive if previous is not None else 0.0
        previous_phrase = (
            phrase_by_id.get(structures[position - 1].phrase_id)
            if position and structures[position - 1].phrase_id != structure.phrase_id
            else None
        )

        if structure.edge_silent:
            role = "stable"
            confidence = 0.2
        elif (
            previous is not None
            and structure.phrase_position == "start"
            and abs(drive_delta) >= 0.28
            and (
                previous_phrase is not None
                and previous_phrase.primary_role in {"build_up", "peak"}
                or previous.boundary_confidence >= 0.5
            )
        ):
            role = "drop"
            confidence = min(1.0, abs(drive_delta) + 0.35)
        elif (
            (
                structure.energy_percentile >= 0.82
                or (
                    structure.energy_percentile >= 0.65
                    and structure.spectral_flux >= 0.75
                    and structure.percussive_ratio >= 0.55
                )
            )
            and structure.energy_percentile
            >= max(
                structures[position - 1].energy_percentile if position else 0.0,
                structures[position + 1].energy_percentile
                if position + 1 < len(structures)
                else 0.0,
            )
        ):
            role = "peak"
            confidence = structure.energy_percentile
        elif (
            phrase.primary_role == "breakdown"
            or (
                structure.energy_percentile <= 0.4
                and structure.activity_mean >= 0.18
                and (
                    structure.onset_density <= 0.08
                    or (
                        _has_spectral_evidence(structure)
                        and structure.percussive_ratio <= 0.25
                        and structure.spectral_flux <= 0.35
                    )
                )
            )
        ):
            role = "breakdown"
            confidence = min(1.0, 0.55 + structure.activity_mean)
        elif phrase.primary_role == "build_up" and structure.phrase_progress < 1.0:
            role = "build_up"
            confidence = min(1.0, 0.55 + max(0.0, phrase.energy_trend) * 4)
        elif structure.phrase_position in {"end", "single"} and structure.boundary_confidence >= 0.46:
            role = "cadence"
            confidence = structure.boundary_confidence
        else:
            role = "stable"
            confidence = max(0.35, 1.0 - abs(drive_delta))
        assigned[position] = structure.model_copy(
            update={
                "transition_role": role,
                "transition_confidence": _rounded(confidence),
            }
        )

    for phrase in phrases:
        roles = [assigned[position].transition_role for position in range(phrase.start_bar, phrase.end_bar + 1)]
        phrase.primary_role = _dominant_role(roles, phrase.primary_role)
    return assigned, phrases


def _dominant_role(roles: list[str], fallback: str) -> str:
    for role in ("build_up", "peak", "drop", "breakdown", "cadence"):
        if role in roles:
            return role
    return fallback


def _assign_sections(
    structures: list[BarStructureFeature],
    phrases: list[_PhraseDraft],
) -> tuple[list[BarStructureFeature], list[_PhraseDraft]]:
    centroids: list[list[float]] = []
    members: list[list[int]] = []
    for phrase in phrases:
        similarities = [1.0 - _profile_distance(phrase.signature, centroid) for centroid in centroids]
        if similarities and max(similarities) >= SECTION_SIMILARITY_THRESHOLD:
            section_index = max(range(len(similarities)), key=similarities.__getitem__)
            members[section_index].append(phrase.phrase_id)
            centroids[section_index] = _mean_profile(
                [phrases[index].signature for index in members[section_index]]
            )
            phrase.section_index = section_index
            phrase.section_confidence = similarities[section_index]
        else:
            phrase.section_index = len(centroids)
            phrase.section_confidence = 0.6
            centroids.append(list(phrase.signature))
            members.append([phrase.phrase_id])

    group_sections: dict[int, str] = {}
    for section_index, phrase_ids in enumerate(members):
        group_phrases = [phrases[phrase_id] for phrase_id in phrase_ids]
        mean_energy = mean(phrase.mean_energy for phrase in group_phrases)
        mean_activity = mean(
            mean(structures[position].activity_mean for position in range(phrase.start_bar, phrase.end_bar + 1))
            for phrase in group_phrases
        )
        repeated = len(group_phrases) > 1
        if not repeated and group_phrases[0].phrase_id == 0:
            section = "intro"
        elif not repeated and group_phrases[0].phrase_id == len(phrases) - 1:
            section = "outro"
        elif mean_energy <= 0.35 and mean_activity < 0.35:
            section = "break"
        elif mean_energy >= 0.65:
            section = "chorus"
        else:
            section = "verse"
        group_sections[section_index] = section

    assigned = list(structures)
    for phrase in phrases:
        phrase.section = group_sections[phrase.section_index]
        section_id = f"section-{phrase.section_index + 1}"
        for position in range(phrase.start_bar, phrase.end_bar + 1):
            assigned[position] = assigned[position].model_copy(
                update={
                    "section_id": section_id,
                    "section": phrase.section,
                    "section_confidence": _rounded(phrase.section_confidence),
                }
            )
    return assigned, phrases


def _assign_fill_candidates(
    structures: list[BarStructureFeature],
    phrases: list[_PhraseDraft],
) -> list[BarStructureFeature]:
    scored: list[tuple[float, int]] = []
    assigned = list(structures)
    for phrase_index, phrase in enumerate(phrases):
        position = phrase.end_bar
        structure = structures[position]
        next_phrase = phrases[phrase_index + 1] if phrase_index + 1 < len(phrases) else None
        energy_difference = (
            abs(next_phrase.mean_energy - phrase.mean_energy) if next_phrase is not None else 0.0
        )
        next_highlight = (
            1.0 if next_phrase is not None and next_phrase.primary_role in {"peak", "drop"} else 0.0
        )
        song_end_bonus = (
            0.15
            if phrase_index == len(phrases) - 1 and len(structures) >= 4
            else 0.05
            if phrase_index == len(phrases) - 1
            else 0.0
        )
        onset_richness = min(1.0, _onset_drive(structure))
        spectral_fill_cue = 0.0
        if _has_spectral_evidence(structure):
            spectral_fill_cue = min(
                0.15,
                structure.spectral_flux * 0.06
                + structure.texture_novelty * 0.05
                + structure.high_onset_strength * 0.04,
            )
        score = (
            phrase.ending_boundary_confidence * 0.30
            + onset_richness * 0.15
            + energy_difference * 0.20
            + next_highlight * 0.15
            + song_end_bonus
            + spectral_fill_cue
            + (0.10 if structure.transition_role in {"cadence", "peak"} else 0.0)
        )
        if structure.edge_silent:
            score = 0.0
        elif structure.transition_role == "breakdown":
            score *= 0.55
        score = _rounded(min(1.0, max(0.0, score)))
        assigned[position] = structure.model_copy(
            update={"fill_candidate_score": score}
        )
        if score >= FILL_CANDIDATE_THRESHOLD:
            scored.append((score, position))

    candidate_limit = max(1, ceil(len(structures) / 6))
    selected = {
        position
        for _score, position in sorted(scored, key=lambda item: (-item[0], item[1]))[
            :candidate_limit
        ]
    }
    return [
        structure.model_copy(
            update={
                "fill_candidate_score": structure.fill_candidate_score
                if position in selected
                else min(structure.fill_candidate_score, FILL_CANDIDATE_THRESHOLD - 0.001),
            }
        )
        for position, structure in enumerate(assigned)
    ]


def _apply_structure_to_bars(
    bars: list[BarFeature],
    structures: list[BarStructureFeature],
) -> list[BarFeature]:
    updated: list[BarFeature] = []
    for position, (bar, structure) in enumerate(zip(bars, structures, strict=True)):
        if position == len(bars) - 1:
            legacy_phrase_position = "song_end"
        elif structure.phrase_position == "start":
            legacy_phrase_position = "phrase_start"
        elif structure.phrase_position in {"end", "single"}:
            legacy_phrase_position = "phrase_end"
        else:
            legacy_phrase_position = "phrase_middle"
        updated.append(
            bar.model_copy(
                update={
                    "phrase_position": legacy_phrase_position,
                    "fill_candidate": structure.fill_candidate_score
                    >= FILL_CANDIDATE_THRESHOLD,
                    "section": structure.section,
                    "energy_percentile": structure.energy_percentile,
                    "energy_delta": structure.energy_delta,
                    "boundary_confidence": structure.boundary_confidence,
                    "phrase_id": structure.phrase_id,
                    "phrase_progress": structure.phrase_progress,
                    "transition_role": structure.transition_role,
                    "transition_confidence": structure.transition_confidence,
                    "section_id": structure.section_id,
                    "section_confidence": structure.section_confidence,
                    "fill_candidate_score": structure.fill_candidate_score,
                }
            )
        )
    return updated


def _build_phrase_features(phrases: list[_PhraseDraft]) -> list[PhraseFeature]:
    result: list[PhraseFeature] = []
    for position, phrase in enumerate(phrases):
        previous_similarity = (
            1.0 - _profile_distance(phrase.signature, phrases[position - 1].signature)
            if position
            else None
        )
        next_similarity = (
            1.0 - _profile_distance(phrase.signature, phrases[position + 1].signature)
            if position + 1 < len(phrases)
            else None
        )
        result.append(
            PhraseFeature(
                phrase_id=phrase.phrase_id,
                start_bar=phrase.start_bar,
                end_bar=phrase.end_bar,
                section_id=f"section-{phrase.section_index + 1}",
                section=phrase.section,
                mean_energy=_rounded(phrase.mean_energy),
                peak_energy=_rounded(phrase.peak_energy),
                energy_trend=_rounded(phrase.energy_trend),
                primary_role=phrase.primary_role,
                ending_boundary_confidence=_rounded(phrase.ending_boundary_confidence),
                previous_similarity=_rounded(previous_similarity)
                if previous_similarity is not None
                else None,
                next_similarity=_rounded(next_similarity) if next_similarity is not None else None,
            )
        )
    return result


def _structure_confidence(scores: dict[int, float], boundaries: list[int]) -> float:
    internal = boundaries[1:-1]
    if not internal:
        return _rounded(min(0.45, max(scores.values(), default=0.0) * 0.6))
    selected_scores = [scores.get(position, 0.0) for position in internal]
    return _rounded(min(1.0, mean(selected_scores)))


def _rounded(value: float | None) -> float:
    return round(float(value or 0.0), 6)
