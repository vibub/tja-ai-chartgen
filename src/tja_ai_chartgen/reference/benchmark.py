from __future__ import annotations

from collections import Counter
from typing import Any

from tja_ai_chartgen.reference.model import ReferenceCourse, ReferenceTja


def build_reference_benchmark(parsed_files: list[ReferenceTja]) -> dict[str, Any]:
    course_metrics = [
        _course_metrics(parsed.source_id, course)
        for parsed in parsed_files
        for course in parsed.courses
    ]
    resolution_counts: Counter[int] = Counter()
    course_level_counts: Counter[str] = Counter()
    for metrics in course_metrics:
        resolution_counts.update(
            {
                int(resolution): int(count)
                for resolution, count in metrics["resolution_counts"].items()
            }
        )
        course_level_counts[f"{metrics['course']}:{metrics['level']}"] += 1

    return {
        "schema_version": 1,
        "file_count": len(parsed_files),
        "course_count": len(course_metrics),
        "bar_count": sum(int(item["bar_count"]) for item in course_metrics),
        "gogo_bar_count": sum(int(item["gogo_bar_count"]) for item in course_metrics),
        "resolution_change_count": sum(
            int(item["resolution_change_count"]) for item in course_metrics
        ),
        "resolution_counts": _string_key_counts(resolution_counts),
        "course_level_counts": dict(sorted(course_level_counts.items())),
        "courses": course_metrics,
    }


def _course_metrics(source_id: str, course: ReferenceCourse) -> dict[str, Any]:
    resolution_counts = Counter(bar.resolution for bar in course.bars)
    normal_counts = [_normal_note_count(bar.notes) for bar in course.bars]
    durations = [max(0.0, bar.end_time - bar.start_time) for bar in course.bars]
    total_duration = sum(durations)
    gogo_pairs = [
        (count, duration)
        for bar, count, duration in zip(course.bars, normal_counts, durations, strict=True)
        if bar.gogo
    ]
    gogo_note_count = sum(count for count, _duration in gogo_pairs)
    gogo_duration = sum(duration for _count, duration in gogo_pairs)
    per_bar_nps = [
        count / duration
        for count, duration in zip(normal_counts, durations, strict=True)
        if duration > 0
    ]

    return {
        "source_id": source_id,
        "course": course.course,
        "level": course.level,
        "bar_count": len(course.bars),
        "resolution_counts": _string_key_counts(resolution_counts),
        "resolution_change_count": sum(
            previous.resolution != current.resolution
            for previous, current in zip(course.bars[:-1], course.bars[1:], strict=True)
        ),
        "normal_note_count": sum(normal_counts),
        "special_note_count": sum(
            character in "5679" for bar in course.bars for character in bar.notes
        ),
        "playable_duration_seconds": round(total_duration, 6),
        "average_notes_per_second": round(
            sum(normal_counts) / total_duration if total_duration else 0.0,
            6,
        ),
        "peak_bar_notes_per_second": round(max(per_bar_nps, default=0.0), 6),
        "gogo_bar_count": len(gogo_pairs),
        "gogo_average_notes_per_second": round(
            gogo_note_count / gogo_duration if gogo_duration else 0.0,
            6,
        ),
    }


def _normal_note_count(notes: str) -> int:
    return sum(character in "1234" for character in notes)


def _string_key_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(key): counts[key] for key in sorted(counts)}
