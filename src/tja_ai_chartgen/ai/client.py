import json
import os
from typing import Any

from litellm import completion

from tja_ai_chartgen.ai.prompts import build_chart_generation_prompt
from tja_ai_chartgen.features.density import BarDensityHint, build_density_hints
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, SongAnalysis

ALLOWED_AI_NOTES = set("01234578")
FALLBACK_AI_PATTERN = "1000100010001000"
DEFAULT_AI_REPAIR_RETRIES = 2


class AiOutputRepairError(RuntimeError):
    def __init__(self, message: str, output: dict[str, Any]) -> None:
        super().__init__(message)
        self.output = output


class _AiOutputValidationError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues))
        self.issues = issues


def generate_chart_bars_with_ai(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_repair_attempts: int = DEFAULT_AI_REPAIR_RETRIES,
    special_notes: bool = False,
) -> tuple[list[ChartBar], dict[str, Any]]:
    model_name = model or os.getenv("MODEL", "openai/gpt-4o-mini")
    resolved_api_base = api_base or os.getenv("OPENAI_BASE_URL")
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    repair_attempts = max(0, max_repair_attempts)
    prompt = build_chart_generation_prompt(
        analysis,
        course,
        level,
        style,
        density,
        special_notes=special_notes,
    )
    messages = [{"role": "user", "content": prompt}]
    attempts: list[dict[str, Any]] = []

    for attempt_index in range(repair_attempts + 1):
        response = completion(
            **_completion_kwargs(
                model=model_name,
                messages=messages,
                api_base=resolved_api_base,
                api_key=resolved_api_key,
            )
        )
        content = _extract_response_content(response)

        try:
            data = _parse_ai_json(content)
            bars = _validate_ai_data(
                data,
                analysis=analysis,
                course=course,
                level=level,
                density=density,
                special_notes=special_notes,
            )
        except _AiOutputValidationError as error:
            issues = error.issues
        except json.JSONDecodeError as error:
            issues = [f"output must be valid JSON: {error.msg}"]
        else:
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "ok",
                    "content": content,
                    "data": data,
                }
            )
            return bars, _build_ai_output(
                model=model_name,
                api_base=resolved_api_base,
                api_key_provided=bool(resolved_api_key),
                max_repair_attempts=repair_attempts,
                attempts=attempts,
                final=data,
            )

        attempts.append(
            {
                "attempt": attempt_index + 1,
                "status": "invalid",
                "content": content,
                "issues": issues,
            }
        )

        if attempt_index < repair_attempts:
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": _build_repair_prompt(
                        issues,
                        analysis=analysis,
                        course=course,
                        level=level,
                        density=density,
                        special_notes=special_notes,
                    ),
                }
            )

    output = _build_ai_output(
        model=model_name,
        api_base=resolved_api_base,
        api_key_provided=bool(resolved_api_key),
        max_repair_attempts=repair_attempts,
        attempts=attempts,
        final=None,
    )
    raise AiOutputRepairError(
        f"AI output remained invalid after {len(attempts)} attempt(s): "
        f"{'; '.join(attempts[-1].get('issues', []))}",
        output,
    )


def sanitize_ai_bars(
    bars: list[ChartBar],
    expected_count: int,
    expected_bars: list[BarFeature] | None = None,
) -> list[ChartBar]:
    sanitized: list[ChartBar] = []
    silent_indexes = edge_silence_indexes(expected_bars or [])

    for index in range(expected_count):
        expected_length = expected_bars[index].grids_per_bar if expected_bars and index < len(expected_bars) else 16
        expected_time_signature = (
            expected_bars[index].time_signature if expected_bars and index < len(expected_bars) else "4/4"
        )
        if index < len(bars):
            notes = bars[index].notes
            time_signature = bars[index].time_signature
            balloon_counts = bars[index].balloon_counts
        else:
            notes = FALLBACK_AI_PATTERN
            time_signature = expected_time_signature
            balloon_counts = []

        cleaned_notes = "0" * expected_length if index in silent_indexes else _sanitize_notes(notes, expected_length)
        sanitized.append(
            ChartBar(
                index=index,
                notes=cleaned_notes,
                time_signature=time_signature,
                balloon_counts=balloon_counts[: cleaned_notes.count("7")],
            )
        )

    return sanitized


def _completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_base: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }

    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    return kwargs


def _parse_ai_json(content: str) -> dict[str, Any]:
    data = json.loads(content)
    if not isinstance(data, dict):
        raise _AiOutputValidationError(["output JSON must be an object"])
    return data


def _validate_ai_data(
    data: dict[str, Any],
    analysis: SongAnalysis,
    course: str,
    level: int,
    density: str,
    special_notes: bool,
) -> list[ChartBar]:
    raw_bars = data.get("bars")
    issues: list[str] = []

    if not isinstance(raw_bars, list):
        raise _AiOutputValidationError(["bars must be a list"])

    expected_count = len(analysis.bars)
    if len(raw_bars) != expected_count:
        issues.append(f"bars must contain exactly {expected_count} item(s), got {len(raw_bars)}")

    bars: list[ChartBar] = []
    for index, item in enumerate(raw_bars):
        if not isinstance(item, dict):
            issues.append(f"bars[{index}] must be an object")
            continue

        notes = item.get("notes")
        if not isinstance(notes, str):
            issues.append(f"bars[{index}].notes must be a string")
            continue

        expected_length = _expected_note_length(analysis, index)
        if len(notes) != expected_length:
            issues.append(
                f"bars[{index}].notes must be exactly {expected_length} characters, got {len(notes)}"
            )

        allowed_notes = _allowed_ai_notes(special_notes)
        illegal_characters = sorted(set(notes) - allowed_notes)
        if illegal_characters:
            joined = "".join(illegal_characters)
            issues.append(f"bars[{index}].notes contains illegal character(s): {joined}")

        balloon_count = notes.count("7")
        raw_balloon_counts = item.get("balloon_counts", [])
        if raw_balloon_counts is None:
            raw_balloon_counts = []
        if not isinstance(raw_balloon_counts, list):
            issues.append(f"bars[{index}].balloon_counts must be a list")
            raw_balloon_counts = []
        parsed_balloon_counts: list[int] = []
        for count_index, raw_count in enumerate(raw_balloon_counts):
            if not isinstance(raw_count, int) or raw_count < 1:
                issues.append(f"bars[{index}].balloon_counts[{count_index}] must be a positive integer")
                continue
            parsed_balloon_counts.append(raw_count)
        if balloon_count and len(parsed_balloon_counts) != balloon_count:
            issues.append(
                f"bars[{index}].balloon_counts must contain exactly {balloon_count} item(s)"
            )

        time_signature = _bar_time_signature(analysis, index)
        bars.append(
            ChartBar(
                index=index,
                notes=notes,
                time_signature=time_signature,
                balloon_counts=parsed_balloon_counts,
            )
        )

    if not issues:
        issues.extend(_validate_edge_silence(bars, analysis.bars))
    if not issues:
        issues.extend(
            _validate_chart_quality(
                bars,
                analysis=analysis,
                course=course,
                level=level,
                density=density,
            )
        )

    if issues:
        raise _AiOutputValidationError(issues)

    return bars


def _validate_edge_silence(bars: list[ChartBar], expected_bars: list[BarFeature]) -> list[str]:
    issues: list[str] = []
    for index in sorted(edge_silence_indexes(expected_bars)):
        if index >= len(bars):
            continue
        if _playable_hit_count(bars[index].notes) == 0:
            continue
        issues.append(
            f"bars[{index}].notes must be all 0 because the matching input bar is "
            "song-start/song-end silence"
        )
    return issues


def _validate_chart_quality(
    bars: list[ChartBar],
    *,
    analysis: SongAnalysis,
    course: str,
    level: int,
    density: str,
) -> list[str]:
    quality_density = _quality_density(density, course, level)
    if len(bars) < 8:
        return []

    issues: list[str] = []
    expected_bars = analysis.bars[: len(bars)]
    density_hints = build_density_hints(expected_bars)
    hit_counts = [_playable_hit_count(bar.notes) for bar in bars]
    normalized_hit_counts = [
        _normalized_hit_count(bar.notes, expected_length=_expected_note_length(analysis, index))
        for index, bar in enumerate(bars)
    ]
    quality_indexes = [
        index
        for index, hint in enumerate(density_hints)
        if hint.count_in_quality_average
    ]
    playable_indexes = [
        index
        for index, hint in enumerate(density_hints)
        if hint.kind not in {"silent", "rest"}
    ]

    issues.extend(_density_hint_issues(hit_counts, density_hints))

    if quality_indexes and quality_density in {"high", "max"}:
        thresholds = _quality_thresholds(quality_density)
        average_hits = sum(normalized_hit_counts[index] for index in quality_indexes) / len(quality_indexes)
        if average_hits < thresholds["average"]:
            issues.append(
                "chart quality is too sparse: average playable hits per 16-grid bar "
                f"is {average_hits:.1f}, expected at least {thresholds['average']:.1f} "
                f"for density {density} / {course} level {level}"
            )

        sparse_indexes = [
            index
            for index in quality_indexes
            if normalized_hit_counts[index] < thresholds["normal_min"]
        ]
        sparse_limit = max(1, len(quality_indexes) // 12)
        if len(sparse_indexes) > sparse_limit:
            examples = ", ".join(str(index + 1) for index in sparse_indexes[:8])
            issues.append(
                "chart quality has too many sparse middle bars: "
                f"{len(sparse_indexes)} bar(s) below {thresholds['normal_min']:.1f} "
                f"normalized hits; examples: {examples}"
            )

        empty_runs = _empty_runs_in_quality_bars(hit_counts, quality_indexes)
        if empty_runs:
            examples = ", ".join(f"bars {start + 1}-{end + 1}" for start, end in empty_runs[:4])
            issues.append(f"chart quality has empty middle bar runs that need beat skeletons: {examples}")

    color_issue = _note_color_balance_issue(bars, quality_indexes=playable_indexes)
    if color_issue:
        issues.append(color_issue)

    repeated = _overused_patterns(bars)
    if repeated:
        examples = ", ".join(f"{pattern} x{count}" for pattern, count in repeated[:4])
        issues.append(f"chart quality repeats exact note strings too often: {examples}")

    return issues


def _playable_hit_count(notes: str) -> int:
    return sum(character != "0" for character in notes)


def _normalized_hit_count(notes: str, expected_length: int) -> float:
    if expected_length <= 0:
        return 0.0
    return _playable_hit_count(notes) * 16 / expected_length


def _quality_density(density: str, course: str, level: int) -> str:
    if density in {"high", "max"}:
        return density
    if density == "auto" and course.lower() == "oni":
        if level >= 10:
            return "max"
        if level >= 8:
            return "high"
    return density


def _quality_thresholds(density: str) -> dict[str, float]:
    if density == "max":
        return {"average": 8.5, "normal_min": 4.0}
    if density == "high":
        return {"average": 6.5, "normal_min": 3.0}
    if density == "medium":
        return {"average": 3.5, "normal_min": 1.0}
    return {"average": 0.0, "normal_min": 0.0}


def _density_hint_issues(
    hit_counts: list[int],
    density_hints: list[BarDensityHint],
) -> list[str]:
    issues: list[str] = []
    for index, (hit_count, hint) in enumerate(zip(hit_counts, density_hints, strict=False)):
        if hint.max_hits is not None and hit_count > hint.max_hits:
            issues.append(
                f"bars[{index}].notes is too dense for {hint.kind} density hint: "
                f"{hit_count} hit(s), expected at most {hint.max_hits}; reason: {hint.reason}"
            )
        if not hint.allow_empty and hit_count < hint.min_hits:
            issues.append(
                f"bars[{index}].notes is too sparse for {hint.kind} density hint: "
                f"{hit_count} hit(s), expected at least {hint.min_hits}; reason: {hint.reason}"
            )
    return issues


def _note_color_balance_issue(
    bars: list[ChartBar],
    *,
    quality_indexes: list[int],
) -> str | None:
    normal_don = 0
    normal_ka = 0
    longest_don_run = 0
    current_don_run = 0

    for index in quality_indexes:
        if index >= len(bars):
            continue
        for note in bars[index].notes:
            if note == "1":
                normal_don += 1
                current_don_run += 1
                longest_don_run = max(longest_don_run, current_don_run)
            elif note == "2":
                normal_ka += 1
                current_don_run = 0
            elif note != "0":
                current_don_run = 0

    normal_notes = normal_don + normal_ka
    if normal_notes < 32:
        return None

    ka_ratio = normal_ka / normal_notes
    if normal_ka < 4 and ka_ratio < 0.03:
        return (
            "chart quality is nearly all don notes: normal note 2 ratio "
            f"is {ka_ratio:.2f}; add a few 2 notes to offbeat/answer/fill hits"
        )
    if longest_don_run > 32 and ka_ratio < 0.08:
        return (
            "chart quality has an overlong all-don run: "
            f"{longest_don_run} consecutive normal 1 notes; break long 1 streams with occasional 2 notes"
        )
    return None


def _empty_runs_in_quality_bars(
    hit_counts: list[int],
    quality_indexes: list[int],
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous_index: int | None = None

    for index in quality_indexes:
        continues_run = previous_index is not None and index == previous_index + 1
        if not continues_run and start is not None and previous_index is not None:
            runs.append((start, previous_index))
            start = None

        if hit_counts[index] == 0:
            if start is None:
                start = index
        elif start is not None and previous_index is not None:
            runs.append((start, previous_index))
            start = None

        previous_index = index

    if start is not None and previous_index is not None:
        runs.append((start, previous_index))
    return runs


def _overused_patterns(bars: list[ChartBar]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for bar in bars:
        if _playable_hit_count(bar.notes) == 0:
            continue
        counts[bar.notes] = counts.get(bar.notes, 0) + 1

    limit = max(4, len(bars) // 12)
    return sorted(
        ((pattern, count) for pattern, count in counts.items() if count > limit),
        key=lambda item: item[1],
        reverse=True,
    )


def _build_repair_prompt(
    issues: list[str],
    analysis: SongAnalysis,
    course: str,
    level: int,
    density: str,
    special_notes: bool,
) -> str:
    lengths = [_expected_note_length(analysis, index) for index in range(len(analysis.bars))]
    allowed = ", ".join(sorted(_allowed_ai_notes(special_notes)))
    forced_silent_bars = [index + 1 for index in sorted(edge_silence_indexes(analysis.bars))]
    return f"""
Your previous output was invalid and cannot be used as a TJA chart draft.

Fix the output and return JSON only.

Validation errors:
{json.dumps(issues, ensure_ascii=False, indent=2)}

Required schema:
{{
  "bars": [
    {{"bar": 1, "notes": "1000100010001000"}}
  ]
}}

Rules:
- bars must contain exactly {len(analysis.bars)} item(s).
- Notes length per bar must match the input bar grids: {lengths}.
- Allowed notes characters: {allowed}.
- If using balloon note 7, include one positive integer in balloon_counts for each 7 in that bar.
- Course/difficulty request: {course} level {level}, density {density}.
- Forced silent bars: {forced_silent_bars}. These song-start/song-end silence bars must be all 0 for their full notes length.
- Respect bar_density_hints from the original input: rest bars may stay empty, sparse bars may stay light, and normal/dense/fill bars should carry the requested course density.
- If validation errors mention low chart quality, increase 1/2 note density on normal/dense/fill bars, keep real rest bars empty or sparse, add more 2/4 ka notes for offbeat/answer/fill hits, and vary repeated or all-don patterns without changing bar count or note lengths.
- Do not include markdown, comments, explanations, or extra text.
""".strip()


def _build_ai_output(
    *,
    model: str,
    api_base: str | None,
    api_key_provided: bool,
    max_repair_attempts: int,
    attempts: list[dict[str, Any]],
    final: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "api_base": api_base,
        "api_key_provided": api_key_provided,
        "max_repair_attempts": max_repair_attempts,
        "attempts": attempts,
        "final": final,
    }


def _sanitize_notes(notes: str, expected_length: int) -> str:
    cleaned = "".join(character if character in ALLOWED_AI_NOTES else "0" for character in notes)
    return cleaned[:expected_length].ljust(expected_length, "0")


def _allowed_ai_notes(special_notes: bool) -> set[str]:
    if special_notes:
        return ALLOWED_AI_NOTES
    return set("01234")


def _expected_note_length(analysis: SongAnalysis, index: int) -> int:
    if index < len(analysis.bars):
        return analysis.bars[index].grids_per_bar
    return 16


def _bar_time_signature(analysis: SongAnalysis, index: int) -> str:
    if index < len(analysis.bars):
        return analysis.bars[index].time_signature
    return analysis.time_signature


def _extract_response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])

    choices = response.choices
    message = choices[0].message
    return str(message.content)
