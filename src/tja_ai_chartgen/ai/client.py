import json
import os
from typing import Any

from litellm import completion

from tja_ai_chartgen.ai.prompts import build_chart_generation_prompt
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
    reference_examples: list[dict[str, Any]] | None = None,
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
        reference_examples=reference_examples,
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
            bars = _validate_ai_data(data, analysis=analysis, special_notes=special_notes)
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
                reference_example_count=len(reference_examples or []),
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
        reference_example_count=len(reference_examples or []),
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

        cleaned_notes = _sanitize_notes(notes, expected_length)
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

    if issues:
        raise _AiOutputValidationError(issues)

    return bars


def _build_repair_prompt(
    issues: list[str],
    analysis: SongAnalysis,
    special_notes: bool,
) -> str:
    lengths = [_expected_note_length(analysis, index) for index in range(len(analysis.bars))]
    allowed = ", ".join(sorted(_allowed_ai_notes(special_notes)))
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
    reference_example_count: int = 0,
) -> dict[str, Any]:
    return {
        "model": model,
        "api_base": api_base,
        "api_key_provided": api_key_provided,
        "max_repair_attempts": max_repair_attempts,
        "reference_example_count": reference_example_count,
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
