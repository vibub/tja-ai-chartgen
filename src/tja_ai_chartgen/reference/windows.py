from __future__ import annotations

from statistics import mean
from typing import Any

from tja_ai_chartgen.reference.model import ReferenceBar, ReferenceCourse, ReferenceTja


REFERENCE_WINDOW_SCHEMA_VERSION = 1
DEFAULT_WINDOW_SIZE = 8
BAR_COLUMNS = [
    "relative_bar",
    "resolution",
    "measure_ratio",
    "bpm",
    "tempo_changes",
    "gogo",
    "normal_hit_count",
    "events",
    "balloon_counts",
]


def build_reference_windows(
    parsed_files: list[ReferenceTja],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict[str, Any]:
    if window_size < 2:
        raise ValueError("Reference window size must be at least 2 bars")

    windows = [
        window
        for parsed in parsed_files
        for course in parsed.courses
        for window in _course_windows(parsed.source_id, course, window_size)
    ]
    return {
        "schema_version": REFERENCE_WINDOW_SCHEMA_VERSION,
        "window_size": window_size,
        "source_count": len(parsed_files),
        "course_count": sum(len(parsed.courses) for parsed in parsed_files),
        "bar_columns": BAR_COLUMNS,
        "windows": windows,
    }


def _course_windows(
    source_id: str,
    course: ReferenceCourse,
    window_size: int,
) -> list[dict[str, Any]]:
    bars = course.bars
    if not bars:
        return []
    size = min(window_size, len(bars))
    last_start = len(bars) - size
    candidates = [
        (0, "intro"),
        (last_start, "cadence"),
        (_peak_window_start(bars, size), "peak"),
    ]
    selected: list[tuple[int, str]] = []
    seen_starts: set[int] = set()
    for start, role in candidates:
        start = min(last_start, max(0, start))
        if start in seen_starts:
            continue
        seen_starts.add(start)
        selected.append((start, role))

    return [
        _window_payload(source_id, course, start, size, role)
        for start, role in selected
    ]


def _peak_window_start(bars: list[ReferenceBar], size: int) -> int:
    if len(bars) <= size:
        return 0
    scores = [
        mean(_bar_notes_per_second(bar) for bar in bars[start : start + size])
        for start in range(len(bars) - size + 1)
    ]
    return max(range(len(scores)), key=lambda start: (scores[start], -start))


def _window_payload(
    source_id: str,
    course: ReferenceCourse,
    start: int,
    size: int,
    role: str,
) -> dict[str, Any]:
    bars = course.bars[start : start + size]
    duration = sum(max(0.0, bar.end_time - bar.start_time) for bar in bars)
    normal_hit_count = sum(_normal_hit_count(bar.notes) for bar in bars)
    return {
        "source_id": source_id,
        "course": course.course,
        "level": course.level,
        "role": role,
        "start_bar": start,
        "end_bar": start + size - 1,
        "average_notes_per_second": round(
            normal_hit_count / duration if duration > 0 else 0.0,
            6,
        ),
        "gogo_ratio": round(sum(bar.gogo for bar in bars) / len(bars), 6),
        "bars": [
            [
                relative_index,
                bar.resolution,
                bar.measure_ratio,
                round(bar.bpm, 6),
                [[change.position, round(change.bpm, 6)] for change in bar.tempo_changes],
                int(bar.gogo),
                _normal_hit_count(bar.notes),
                [[index, note] for index, note in enumerate(bar.notes) if note != "0"],
                bar.balloon_counts,
            ]
            for relative_index, bar in enumerate(bars)
        ],
    }


def _bar_notes_per_second(bar: ReferenceBar) -> float:
    duration = bar.end_time - bar.start_time
    return _normal_hit_count(bar.notes) / duration if duration > 0 else 0.0


def _normal_hit_count(notes: str) -> int:
    return sum(note in "1234" for note in notes)
