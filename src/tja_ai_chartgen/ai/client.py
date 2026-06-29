import json
import os
from typing import Any

from litellm import completion

from tja_ai_chartgen.ai.prompts import build_chart_generation_prompt
from tja_ai_chartgen.tja.model import ChartBar, SongAnalysis

ALLOWED_AI_NOTES = set("01234")
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
) -> tuple[list[ChartBar], dict[str, Any]]:
    model_name = model or os.getenv("MODEL", "openai/gpt-4o-mini")
    resolved_api_base = api_base or os.getenv("OPENAI_BASE_URL")
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    repair_attempts = max(0, max_repair_attempts)
    prompt = build_chart_generation_prompt(analysis, course, level, style, density)
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
            bars = _validate_ai_data(data, expected_count=len(analysis.bars))
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
                    "content": _build_repair_prompt(issues, expected_count=len(analysis.bars)),
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


def sanitize_ai_bars(bars: list[ChartBar], expected_count: int) -> list[ChartBar]:
    sanitized: list[ChartBar] = []

    for index in range(expected_count):
        if index < len(bars):
            notes = bars[index].notes
        else:
            notes = FALLBACK_AI_PATTERN

        sanitized.append(ChartBar(index=index, notes=_sanitize_notes(notes)))

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


def _validate_ai_data(data: dict[str, Any], expected_count: int) -> list[ChartBar]:
    raw_bars = data.get("bars")
    issues: list[str] = []

    if not isinstance(raw_bars, list):
        raise _AiOutputValidationError(["bars must be a list"])

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

        if len(notes) != 16:
            issues.append(f"bars[{index}].notes must be exactly 16 characters, got {len(notes)}")

        illegal_characters = sorted(set(notes) - ALLOWED_AI_NOTES)
        if illegal_characters:
            joined = "".join(illegal_characters)
            issues.append(f"bars[{index}].notes contains illegal character(s): {joined}")

        bars.append(ChartBar(index=index, notes=notes))

    if issues:
        raise _AiOutputValidationError(issues)

    return bars


def _build_repair_prompt(issues: list[str], expected_count: int) -> str:
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
- bars must contain exactly {expected_count} item(s).
- Each notes value must be exactly 16 characters long.
- Allowed notes characters: 0, 1, 2, 3, 4.
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


def _sanitize_notes(notes: str) -> str:
    cleaned = "".join(character if character in ALLOWED_AI_NOTES else "0" for character in notes)
    return cleaned[:16].ljust(16, "0")


def _extract_response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])

    choices = response.choices
    message = choices[0].message
    return str(message.content)
