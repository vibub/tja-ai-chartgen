from __future__ import annotations

from hashlib import sha256
from typing import Any

from tja_ai_chartgen.evaluation.audio_benchmark import match_events
from tja_ai_chartgen.tja.model import BarFeature, ChartBar


CHART_ALIGNMENT_SCHEMA_VERSION = 1
CHART_ALIGNMENT_BENCHMARK_VERSION = "chart-alignment-v1"
DEFAULT_NOTE_ONSET_TOLERANCE_SECONDS = 0.05
DEFAULT_BEAT_EVIDENCE_TOLERANCE_SECONDS = 0.07
NOTE_START_CHARACTERS = "1234579"


def build_chart_alignment_metrics(
    ground_truth: dict[str, Any],
    feature_bars: list[BarFeature],
    chart_bars: list[ChartBar],
    *,
    course: str,
    level: int,
    deterministic: bool,
    note_onset_tolerance_seconds: float = DEFAULT_NOTE_ONSET_TOLERANCE_SECONDS,
    beat_evidence_tolerance_seconds: float = DEFAULT_BEAT_EVIDENCE_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    if len(feature_bars) != len(chart_bars):
        raise ValueError(
            f"Feature bar count {len(feature_bars)} does not match chart bar count "
            f"{len(chart_bars)}"
        )

    note_times = _chart_note_times(feature_bars, chart_bars)
    onset_times = _float_times(ground_truth.get("onsets", []))
    strong_onsets = _float_times(ground_truth.get("strong_onsets", []))
    beat_times = _float_times(ground_truth.get("beats", []))
    downbeats = _float_times(ground_truth.get("downbeats", []))
    silent_ranges = _time_ranges(ground_truth.get("silent_ranges", []))
    fill_ranges = _time_ranges(ground_truth.get("fill_ranges", []))
    fill_onsets = [time for time in onset_times if _in_any_range(time, fill_ranges)]
    fill_notes = [time for time in note_times if _in_any_range(time, fill_ranges)]

    evidenced_note_count = sum(
        _has_nearby_time(time, onset_times, note_onset_tolerance_seconds)
        or _has_nearby_time(time, beat_times, beat_evidence_tolerance_seconds)
        for time in note_times
    )
    silent_violation_count = sum(_in_any_range(time, silent_ranges) for time in note_times)
    note_count = len(note_times)

    return {
        "audio": str(ground_truth["audio"]),
        "course": course,
        "level": level,
        "bar_count": len(chart_bars),
        "note_count": note_count,
        "note_times_hash": _note_times_hash(note_times),
        "deterministic": deterministic,
        "note_onset": match_events(
            onset_times,
            note_times,
            tolerance_seconds=note_onset_tolerance_seconds,
        ),
        "strong_onset_response": match_events(
            strong_onsets,
            note_times,
            tolerance_seconds=note_onset_tolerance_seconds,
        ),
        "downbeat_response": match_events(
            downbeats,
            note_times,
            tolerance_seconds=beat_evidence_tolerance_seconds,
        ),
        "fill_onset_response": match_events(
            fill_onsets,
            fill_notes,
            tolerance_seconds=note_onset_tolerance_seconds,
        ),
        "evidenced_note_count": evidenced_note_count,
        "unsupported_note_count": note_count - evidenced_note_count,
        "unsupported_note_ratio": _ratio(note_count - evidenced_note_count, note_count),
        "silent_violation_count": silent_violation_count,
        "silent_violation_ratio": _ratio(silent_violation_count, note_count),
    }


def build_chart_alignment_benchmark(
    chart_metrics: list[dict[str, Any]],
    *,
    fixture_count: int,
    use_beatnet: bool,
    style: str,
    density: str,
    special_notes: bool,
    note_onset_tolerance_seconds: float = DEFAULT_NOTE_ONSET_TOLERANCE_SECONDS,
    beat_evidence_tolerance_seconds: float = DEFAULT_BEAT_EVIDENCE_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    return {
        "schema_version": CHART_ALIGNMENT_SCHEMA_VERSION,
        "benchmark_version": CHART_ALIGNMENT_BENCHMARK_VERSION,
        "fixture_ground_truth_schema_version": 1,
        "fixture_count": fixture_count,
        "chart_count": len(chart_metrics),
        "settings": {
            "use_beatnet": use_beatnet,
            "style": style,
            "density": density,
            "special_notes": special_notes,
            "note_onset_tolerance_seconds": note_onset_tolerance_seconds,
            "beat_evidence_tolerance_seconds": beat_evidence_tolerance_seconds,
        },
        "summary": _aggregate_chart_metrics(chart_metrics),
        "course_summaries": {
            course: _aggregate_chart_metrics(
                [item for item in chart_metrics if item["course"] == course]
            )
            for course in sorted({str(item["course"]) for item in chart_metrics})
        },
        "charts": chart_metrics,
    }


def render_chart_alignment_markdown(benchmark: dict[str, Any]) -> str:
    settings = benchmark["settings"]
    summary = benchmark["summary"]
    lines = [
        "# Synthetic chart alignment benchmark",
        "",
        f"- Benchmark: `{benchmark['benchmark_version']}`",
        f"- Fixture count: {benchmark['fixture_count']}",
        f"- Chart count: {benchmark['chart_count']}",
        f"- BeatNet: `{str(settings['use_beatnet']).lower()}`",
        f"- Rule profile: `{settings['style']}` / `{settings['density']}` / "
        f"special notes `{str(settings['special_notes']).lower()}`",
        f"- Tolerances: onset {settings['note_onset_tolerance_seconds']:.3f}s, "
        f"beat evidence {settings['beat_evidence_tolerance_seconds']:.3f}s",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Note/onset", "note_onset"),
        ("Strong onset response", "strong_onset_response"),
        ("Downbeat response", "downbeat_response"),
        ("Fill onset response", "fill_onset_response"),
    ):
        metric = summary[key]
        lines.append(
            f"| {label} | {metric['precision']:.6f} | {metric['recall']:.6f} | "
            f"{metric['f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Diagnostic | Value |",
            "| --- | ---: |",
            f"| Notes | {summary['note_count']} |",
            f"| Unsupported-note ratio | {summary['unsupported_note_ratio']:.6f} |",
            f"| Silent-range violations | {summary['silent_violation_count']} |",
            f"| Deterministic charts | {summary['deterministic_chart_count']} / "
            f"{summary['chart_count']} |",
            "",
            "## Per-course metrics",
            "",
            "| Course | Charts | Notes | Note/onset precision | Strong response | "
            "Downbeat response | Unsupported | Silent violations | Deterministic |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for course, metrics in benchmark["course_summaries"].items():
        lines.append(
            f"| {course} | {metrics['chart_count']} | {metrics['note_count']} | "
            f"{metrics['note_onset']['precision']:.6f} | "
            f"{metrics['strong_onset_response']['recall']:.6f} | "
            f"{metrics['downbeat_response']['recall']:.6f} | "
            f"{metrics['unsupported_note_ratio']:.6f} | "
            f"{metrics['silent_violation_count']} | "
            f"{metrics['deterministic_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-chart metrics",
            "",
            "| Audio | Course | Notes | Note/onset P | Onset R | Strong R | Downbeat R | "
            "Fill R | Unsupported | Silent | Deterministic |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for item in benchmark["charts"]:
        fill_recall = _format_optional_metric(item["fill_onset_response"], "recall")
        lines.append(
            f"| `{item['audio']}` | {item['course']} {item['level']} | {item['note_count']} | "
            f"{item['note_onset']['precision']:.6f} | {item['note_onset']['recall']:.6f} | "
            f"{item['strong_onset_response']['recall']:.6f} | "
            f"{item['downbeat_response']['recall']:.6f} | {fill_recall} | "
            f"{item['unsupported_note_ratio']:.6f} | {item['silent_violation_count']} | "
            f"{'✓' if item['deterministic'] else '✗'} |"
        )
    lines.extend(
        [
            "",
            "普通音符及长音起点会映射到分析后的小节时间轴。Note/onset precision 表示谱面落点由真实瞬态支持的比例；无证据音符允许由理论 beat 支持。",
            "",
            "该报告用于同一环境下的修改前后 A/B。CI 校验 schema、覆盖范围和指标边界，不要求不同 librosa、BeatNet 或平台版本逐值一致。",
            "",
        ]
    )
    return "\n".join(lines)


def _chart_note_times(
    feature_bars: list[BarFeature],
    chart_bars: list[ChartBar],
) -> list[float]:
    result: list[float] = []
    for feature_bar, chart_bar in zip(feature_bars, chart_bars, strict=True):
        resolution = len(chart_bar.notes)
        duration = max(0.0, feature_bar.end_time - feature_bar.start_time)
        if resolution <= 0 or duration <= 0:
            continue
        for position, character in enumerate(chart_bar.notes):
            if character in NOTE_START_CHARACTERS:
                result.append(feature_bar.start_time + duration * position / resolution)
    return sorted(result)


def _aggregate_chart_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    note_count = sum(int(item["note_count"]) for item in items)
    evidenced_note_count = sum(int(item["evidenced_note_count"]) for item in items)
    unsupported_note_count = sum(int(item["unsupported_note_count"]) for item in items)
    silent_violation_count = sum(int(item["silent_violation_count"]) for item in items)
    deterministic_count = sum(bool(item["deterministic"]) for item in items)
    return {
        "chart_count": len(items),
        "note_count": note_count,
        "note_onset": _aggregate_match_metrics(items, "note_onset"),
        "strong_onset_response": _aggregate_match_metrics(items, "strong_onset_response"),
        "downbeat_response": _aggregate_match_metrics(items, "downbeat_response"),
        "fill_onset_response": _aggregate_match_metrics(items, "fill_onset_response"),
        "evidenced_note_count": evidenced_note_count,
        "unsupported_note_count": unsupported_note_count,
        "unsupported_note_ratio": _ratio(unsupported_note_count, note_count),
        "silent_violation_count": silent_violation_count,
        "silent_violation_ratio": _ratio(silent_violation_count, note_count),
        "deterministic_chart_count": deterministic_count,
        "deterministic_rate": _ratio(deterministic_count, len(items)),
    }


def _aggregate_match_metrics(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [item[key] for item in items]
    reference_count = sum(int(metric["reference_count"]) for metric in metrics)
    estimated_count = sum(int(metric["estimated_count"]) for metric in metrics)
    true_positive_count = sum(int(metric["true_positive_count"]) for metric in metrics)
    false_positive_count = sum(int(metric["false_positive_count"]) for metric in metrics)
    false_negative_count = sum(int(metric["false_negative_count"]) for metric in metrics)
    error_sum = sum(
        float(metric["mean_absolute_error_seconds"]) * int(metric["true_positive_count"])
        for metric in metrics
        if metric["mean_absolute_error_seconds"] is not None
    )
    maximum_errors = [
        float(metric["max_absolute_error_seconds"])
        for metric in metrics
        if metric["max_absolute_error_seconds"] is not None
    ]
    return {
        "reference_count": reference_count,
        "estimated_count": estimated_count,
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "precision": _ratio(true_positive_count, true_positive_count + false_positive_count),
        "recall": _ratio(true_positive_count, true_positive_count + false_negative_count),
        "f1": _f1(true_positive_count, false_positive_count, false_negative_count),
        "mean_absolute_error_seconds": round(error_sum / true_positive_count, 6)
        if true_positive_count
        else None,
        "max_absolute_error_seconds": round(max(maximum_errors), 6)
        if maximum_errors
        else None,
    }


def _float_times(values: Any) -> list[float]:
    return sorted(float(value) for value in values)


def _time_ranges(values: Any) -> list[tuple[float, float]]:
    return [(float(start), float(end)) for start, end in values]


def _in_any_range(time: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start <= time < end for start, end in ranges)


def _has_nearby_time(time: float, candidates: list[float], tolerance: float) -> bool:
    return any(abs(time - candidate) <= tolerance for candidate in candidates)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return round(2 * true_positive / denominator, 6) if denominator else 0.0


def _note_times_hash(note_times: list[float]) -> str:
    payload = ",".join(f"{time:.9f}" for time in note_times).encode()
    return sha256(payload).hexdigest()[:16]


def _format_optional_metric(metric: dict[str, Any], key: str) -> str:
    if not metric["reference_count"]:
        return "—"
    return f"{metric[key]:.6f}"
