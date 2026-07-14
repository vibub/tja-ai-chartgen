from __future__ import annotations

from math import isfinite
from typing import Any

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw


AUDIO_BENCHMARK_SCHEMA_VERSION = 1
AUDIO_BENCHMARK_VERSION = "audio-alignment-v1"
DEFAULT_ONSET_TOLERANCE_SECONDS = 0.05
DEFAULT_BEAT_TOLERANCE_SECONDS = 0.07
DEFAULT_DOWNBEAT_TOLERANCE_SECONDS = 0.07
_TEMPO_RELATION_TOLERANCE = 0.05


def match_events(
    reference_times: list[float],
    estimated_times: list[float],
    *,
    tolerance_seconds: float,
) -> dict[str, int | float | None]:
    if tolerance_seconds < 0:
        raise ValueError("Event matching tolerance must be non-negative")

    references = _normalized_times(reference_times)
    estimates = _normalized_times(estimated_times)
    reference_index = 0
    estimate_index = 0
    errors: list[float] = []

    while reference_index < len(references) and estimate_index < len(estimates):
        reference = references[reference_index]
        estimate = estimates[estimate_index]
        difference = estimate - reference
        if abs(difference) <= tolerance_seconds:
            errors.append(abs(difference))
            reference_index += 1
            estimate_index += 1
        elif difference < 0:
            estimate_index += 1
        else:
            reference_index += 1

    true_positive_count = len(errors)
    false_positive_count = len(estimates) - true_positive_count
    false_negative_count = len(references) - true_positive_count
    precision = _safe_ratio(true_positive_count, len(estimates))
    recall = _safe_ratio(true_positive_count, len(references))
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "reference_count": len(references),
        "estimated_count": len(estimates),
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_absolute_error_seconds": (
            round(sum(errors) / len(errors), 6) if errors else None
        ),
        "max_absolute_error_seconds": round(max(errors), 6) if errors else None,
    }


def build_fixture_audio_metrics(
    ground_truth: dict[str, Any],
    analysis: AudioAnalysisRaw,
    *,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    beat_tolerance_seconds: float = DEFAULT_BEAT_TOLERANCE_SECONDS,
    downbeat_tolerance_seconds: float = DEFAULT_DOWNBEAT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    if ground_truth.get("schema_version") != 1:
        raise ValueError("Unsupported fixture ground truth schema version")

    expected_bpm = float(ground_truth["bpm"])
    bpm_absolute_error = abs(analysis.bpm - expected_bpm)
    expected_downbeats = [float(value) for value in ground_truth["downbeats"]]
    if analysis.downbeat_times:
        estimated_downbeats = analysis.downbeat_times
        downbeat_source = "analyzer"
    else:
        estimated_downbeats = _derived_downbeats(analysis)
        downbeat_source = "derived-offset-meter"

    first_downbeat_error = (
        abs(estimated_downbeats[0] - float(ground_truth["first_downbeat"]))
        if estimated_downbeats
        else None
    )
    silent_ranges = [
        (float(start), float(end)) for start, end in ground_truth["silent_ranges"]
    ]
    return {
        "audio": ground_truth["audio"],
        "analyzer": analysis.analyzer,
        "estimated_bpm": round(analysis.bpm, 6),
        "expected_bpm": expected_bpm,
        "bpm_absolute_error": round(bpm_absolute_error, 6),
        "bpm_relative_error": round(bpm_absolute_error / expected_bpm, 6),
        "half_time_error": _tempo_matches(analysis.bpm, expected_bpm / 2.0),
        "double_time_error": _tempo_matches(analysis.bpm, expected_bpm * 2.0),
        "expected_time_signature": ground_truth["time_signature"],
        "estimated_time_signature": analysis.time_signature,
        "meter_correct": analysis.time_signature == ground_truth["time_signature"],
        "first_downbeat_absolute_error_seconds": (
            round(first_downbeat_error, 6) if first_downbeat_error is not None else None
        ),
        "downbeat_source": downbeat_source,
        "silent_onset_false_positive_count": sum(
            any(start <= onset < end for start, end in silent_ranges)
            for onset in analysis.onset_times
        ),
        "onset": match_events(
            ground_truth["onsets"],
            analysis.onset_times,
            tolerance_seconds=onset_tolerance_seconds,
        ),
        "strong_onset": match_events(
            ground_truth["strong_onsets"],
            analysis.onset_times,
            tolerance_seconds=onset_tolerance_seconds,
        ),
        "beat": match_events(
            ground_truth["beats"],
            analysis.beat_times,
            tolerance_seconds=beat_tolerance_seconds,
        ),
        "downbeat": match_events(
            expected_downbeats,
            estimated_downbeats,
            tolerance_seconds=downbeat_tolerance_seconds,
        ),
    }


def build_audio_benchmark(
    fixture_metrics: list[dict[str, Any]],
    *,
    use_beatnet: bool,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    beat_tolerance_seconds: float = DEFAULT_BEAT_TOLERANCE_SECONDS,
    downbeat_tolerance_seconds: float = DEFAULT_DOWNBEAT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIO_BENCHMARK_SCHEMA_VERSION,
        "benchmark_version": AUDIO_BENCHMARK_VERSION,
        "fixture_ground_truth_schema_version": 1,
        "settings": {
            "use_beatnet": use_beatnet,
            "onset_tolerance_seconds": onset_tolerance_seconds,
            "beat_tolerance_seconds": beat_tolerance_seconds,
            "downbeat_tolerance_seconds": downbeat_tolerance_seconds,
        },
        "fixture_count": len(fixture_metrics),
        "summary": {
            "onset": _aggregate_event_metrics(fixture_metrics, "onset"),
            "strong_onset": _aggregate_event_metrics(fixture_metrics, "strong_onset"),
            "beat": _aggregate_event_metrics(fixture_metrics, "beat"),
            "downbeat": _aggregate_event_metrics(fixture_metrics, "downbeat"),
            "mean_bpm_absolute_error": _mean_metric(
                fixture_metrics, "bpm_absolute_error"
            ),
            "max_bpm_absolute_error": _max_metric(
                fixture_metrics, "bpm_absolute_error"
            ),
            "mean_bpm_relative_error": _mean_metric(
                fixture_metrics, "bpm_relative_error"
            ),
            "half_time_error_count": sum(
                bool(item["half_time_error"]) for item in fixture_metrics
            ),
            "double_time_error_count": sum(
                bool(item["double_time_error"]) for item in fixture_metrics
            ),
            "meter_accuracy": round(
                _safe_ratio(
                    sum(bool(item["meter_correct"]) for item in fixture_metrics),
                    len(fixture_metrics),
                ),
                6,
            ),
            "mean_first_downbeat_absolute_error_seconds": _mean_metric(
                fixture_metrics,
                "first_downbeat_absolute_error_seconds",
            ),
            "silent_onset_false_positive_count": sum(
                int(item["silent_onset_false_positive_count"])
                for item in fixture_metrics
            ),
        },
        "fixtures": fixture_metrics,
    }


def _normalized_times(values: list[float]) -> list[float]:
    return sorted(float(value) for value in values if isfinite(value) and value >= 0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _tempo_matches(estimated_bpm: float, target_bpm: float) -> bool:
    if target_bpm <= 0:
        return False
    return abs(estimated_bpm - target_bpm) / target_bpm <= _TEMPO_RELATION_TOLERANCE


def _derived_downbeats(analysis: AudioAnalysisRaw) -> list[float]:
    quarter_notes_per_bar = {"4/4": 4.0, "3/4": 3.0, "6/8": 3.0}.get(
        analysis.time_signature,
        4.0,
    )
    bar_duration = quarter_notes_per_bar * 60.0 / analysis.bpm
    downbeat = analysis.offset
    while downbeat < 0:
        downbeat += bar_duration

    downbeats: list[float] = []
    while downbeat < analysis.duration:
        downbeats.append(round(downbeat, 9))
        downbeat += bar_duration
    return downbeats


def _aggregate_event_metrics(
    fixture_metrics: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, int | float | None]:
    metrics = [item[metric_name] for item in fixture_metrics]
    reference_count = sum(int(item["reference_count"]) for item in metrics)
    estimated_count = sum(int(item["estimated_count"]) for item in metrics)
    true_positive_count = sum(int(item["true_positive_count"]) for item in metrics)
    false_positive_count = sum(int(item["false_positive_count"]) for item in metrics)
    false_negative_count = sum(int(item["false_negative_count"]) for item in metrics)
    precision = _safe_ratio(true_positive_count, estimated_count)
    recall = _safe_ratio(true_positive_count, reference_count)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    matched_error_pairs = [
        (int(item["true_positive_count"]), item["mean_absolute_error_seconds"])
        for item in metrics
        if item["mean_absolute_error_seconds"] is not None
    ]
    total_matches = sum(count for count, _error in matched_error_pairs)
    mean_error = (
        sum(count * float(error) for count, error in matched_error_pairs) / total_matches
        if total_matches
        else None
    )
    max_errors = [
        float(item["max_absolute_error_seconds"])
        for item in metrics
        if item["max_absolute_error_seconds"] is not None
    ]
    return {
        "reference_count": reference_count,
        "estimated_count": estimated_count,
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_absolute_error_seconds": (
            round(mean_error, 6) if mean_error is not None else None
        ),
        "max_absolute_error_seconds": round(max(max_errors), 6) if max_errors else None,
    }


def _mean_metric(fixture_metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in fixture_metrics if item[key] is not None]
    return round(sum(values) / len(values), 6) if values else None


def _max_metric(fixture_metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in fixture_metrics if item[key] is not None]
    return round(max(values), 6) if values else None
