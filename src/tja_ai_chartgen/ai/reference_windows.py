from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


REFERENCE_WINDOWS_PATH = Path(__file__).with_name("reference_windows.json")


@lru_cache(maxsize=1)
def _load_reference_windows() -> dict[str, Any] | None:
    if not REFERENCE_WINDOWS_PATH.is_file():
        return None
    try:
        payload = json.loads(REFERENCE_WINDOWS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    if not isinstance(payload.get("windows"), list):
        return None
    return payload


def get_reference_windows_payload(
    course: str,
    level: int,
    *,
    max_windows: int = 3,
) -> dict[str, Any] | None:
    payload = _load_reference_windows()
    if payload is None or max_windows <= 0:
        return None

    windows = [
        window
        for window in payload["windows"]
        if isinstance(window, dict)
        and str(window.get("course", "")).casefold() == course.casefold()
    ]
    if not windows:
        windows = [window for window in payload["windows"] if isinstance(window, dict)]
    selected = _select_windows(windows, level=level, max_windows=max_windows)
    if not selected:
        return None
    return {
        "schema_version": payload["schema_version"],
        "bar_columns": payload.get("bar_columns", []),
        "windows": selected,
    }


def _select_windows(
    windows: list[dict[str, Any]],
    *,
    level: int,
    max_windows: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    remaining = list(windows)
    for role in ("intro", "peak", "cadence"):
        role_windows = [window for window in remaining if window.get("role") == role]
        if not role_windows:
            continue
        chosen = min(
            role_windows,
            key=lambda window: (
                abs(int(window.get("level", 0)) - level),
                str(window.get("source_id", "")) in used_sources,
                -float(window.get("average_notes_per_second", 0.0))
                if role == "peak"
                else int(window.get("start_bar", 0)),
                str(window.get("source_id", "")),
            ),
        )
        selected.append(chosen)
        used_sources.add(str(chosen.get("source_id", "")))
        remaining.remove(chosen)
        if len(selected) >= max_windows:
            return selected

    for chosen in sorted(
        remaining,
        key=lambda window: (
            abs(int(window.get("level", 0)) - level),
            str(window.get("source_id", "")) in used_sources,
            str(window.get("source_id", "")),
            int(window.get("start_bar", 0)),
        ),
    ):
        selected.append(chosen)
        used_sources.add(str(chosen.get("source_id", "")))
        if len(selected) >= max_windows:
            break
    return selected
