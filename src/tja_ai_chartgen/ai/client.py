import json
import os
from typing import Any

from litellm import completion

from tja_ai_chartgen.ai.prompts import build_chart_generation_prompt
from tja_ai_chartgen.tja.model import ChartBar, SongAnalysis

ALLOWED_AI_NOTES = set("01234")
FALLBACK_AI_PATTERN = "1000100010001000"


def generate_chart_bars_with_ai(
    analysis: SongAnalysis,
    course: str,
    level: int,
    style: str,
    density: str = "auto",
    model: str | None = None,
) -> tuple[list[ChartBar], dict[str, Any]]:
    model_name = model or os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    prompt = build_chart_generation_prompt(analysis, course, level, style, density)

    response = completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = _extract_response_content(response)
    data = json.loads(content)
    raw_bars = data.get("bars", [])

    bars = [
        ChartBar(index=index, notes=str(item.get("notes", "")))
        for index, item in enumerate(raw_bars)
        if isinstance(item, dict)
    ]

    return bars, data


def sanitize_ai_bars(bars: list[ChartBar], expected_count: int) -> list[ChartBar]:
    sanitized: list[ChartBar] = []

    for index in range(expected_count):
        if index < len(bars):
            notes = bars[index].notes
        else:
            notes = FALLBACK_AI_PATTERN

        sanitized.append(ChartBar(index=index, notes=_sanitize_notes(notes)))

    return sanitized


def _sanitize_notes(notes: str) -> str:
    cleaned = "".join(character if character in ALLOWED_AI_NOTES else "0" for character in notes)
    return cleaned[:16].ljust(16, "0")


def _extract_response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])

    choices = response.choices
    message = choices[0].message
    return str(message.content)
