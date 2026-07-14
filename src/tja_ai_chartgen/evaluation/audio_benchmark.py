from __future__ import annotations

from collections import Counter
from math import floor, isfinite
from typing import Any

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.resolution import build_resolution_plan


AUDIO_BENCHMARK_SCHEMA_VERSION = 2
AUDIO_BENCHMARK_VERSION = "audio-alignment-v2"
DEFAULT_ONSET_TOLERANCE_SECONDS = 0.05
DEFAULT_BEAT_TOLERANCE_SECONDS = 0.07
DEFAULT_DOWNBEAT_TOLERANCE_SECONDS = 0.07
DEFAULT_BAND_ONSET_THRESHOLD = 0.25
_TEMPO_RELATION_TOLERANCE = 0.05
_RESOLUTION_EPSILON_SECONDS = 1e-6


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
    band_onset_threshold: float = DEFAULT_BAND_ONSET_THRESHOLD,
) -> dict[str, Any]:
    if ground_truth.get("schema_version") != 1:
        raise ValueError("Unsupported fixture ground truth schema version")
    if not 0.0 <= band_onset_threshold <= 1.0:
        raise ValueError("Band onset threshold must be between 0 and 1")

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
    low_band_estimates = _envelope_peak_times(
        analysis.spectral.low_onset_envelope,
        sample_rate=analysis.sample_rate,
        hop_length=analysis.hop_length,
        threshold=band_onset_threshold,
    )
    high_band_estimates = _envelope_peak_times(
        analysis.spectral.high_onset_envelope,
        sample_rate=analysis.sample_rate,
        hop_length=analysis.hop_length,
        threshold=band_onset_threshold,
    )
    fill_ranges = [(float(start), float(end)) for start, end in ground_truth["fill_ranges"]]
    fill_references = _times_in_ranges(ground_truth["onsets"], fill_ranges)
    fill_estimates = _times_in_ranges(
        analysis.onset_times,
        fill_ranges,
        margin=onset_tolerance_seconds,
    )
    first_downbeat = float(ground_truth["first_downbeat"])
    pickup_references = [
        float(value) for value in ground_truth["onsets"] if float(value) < first_downbeat
    ]
    pickup_estimates = [
        value
        for value in analysis.onset_times
        if value < first_downbeat + onset_tolerance_seconds
    ]
    return {
        "audio": ground_truth["audio"],
        "analyzer": analysis.analyzer,
        "spectral_status": analysis.spectral.status,
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
        "low_band_onset": _optional_event_metrics(
            ground_truth["low_band_onsets"],
            low_band_estimates,
            tolerance_seconds=onset_tolerance_seconds,
        ),
        "high_band_onset": _optional_event_metrics(
            ground_truth["high_band_onsets"],
            high_band_estimates,
            tolerance_seconds=onset_tolerance_seconds,
        ),
        "pickup_onset": _optional_event_metrics(
            pickup_references,
            pickup_estimates,
            tolerance_seconds=onset_tolerance_seconds,
        ),
        "fill_onset": _optional_event_metrics(
            fill_references,
            fill_estimates,
            tolerance_seconds=onset_tolerance_seconds,
        ),
        "resolution": _build_resolution_metrics(ground_truth, analysis),
    }


def build_audio_benchmark(
    fixture_metrics: list[dict[str, Any]],
    *,
    use_beatnet: bool,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    beat_tolerance_seconds: float = DEFAULT_BEAT_TOLERANCE_SECONDS,
    downbeat_tolerance_seconds: float = DEFAULT_DOWNBEAT_TOLERANCE_SECONDS,
    band_onset_threshold: float = DEFAULT_BAND_ONSET_THRESHOLD,
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
            "band_onset_threshold": band_onset_threshold,
        },
        "fixture_count": len(fixture_metrics),
        "summary": {
            "onset": _aggregate_event_metrics(fixture_metrics, "onset"),
            "strong_onset": _aggregate_event_metrics(fixture_metrics, "strong_onset"),
            "beat": _aggregate_event_metrics(fixture_metrics, "beat"),
            "downbeat": _aggregate_event_metrics(fixture_metrics, "downbeat"),
            "low_band_onset": _aggregate_optional_event_metrics(
                fixture_metrics, "low_band_onset"
            ),
            "high_band_onset": _aggregate_optional_event_metrics(
                fixture_metrics, "high_band_onset"
            ),
            "pickup_onset": _aggregate_optional_event_metrics(
                fixture_metrics, "pickup_onset"
            ),
            "fill_onset": _aggregate_optional_event_metrics(
                fixture_metrics, "fill_onset"
            ),
            "resolution": _aggregate_resolution_metrics(fixture_metrics),
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


def render_audio_benchmark_markdown(benchmark: dict[str, Any]) -> str:
    summary = benchmark["summary"]
    settings = benchmark["settings"]
    resolution = summary["resolution"]
    lines = [
        "# Synthetic audio fixture benchmark",
        "",
        f"- Benchmark: `{benchmark['benchmark_version']}`",
        f"- Fixture count: {benchmark['fixture_count']}",
        f"- BeatNet: `{str(settings['use_beatnet']).lower()}`",
        (
            "- Tolerances: "
            f"onset {settings['onset_tolerance_seconds']:.3f}s, "
            f"beat {settings['beat_tolerance_seconds']:.3f}s, "
            f"downbeat {settings['downbeat_tolerance_seconds']:.3f}s"
        ),
        f"- Band onset threshold: {settings['band_onset_threshold']:.3f}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Precision | Recall | F1 | Mean error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("onset", "Onset"),
        ("strong_onset", "Strong onset"),
        ("beat", "Beat"),
        ("downbeat", "Downbeat"),
        ("low_band_onset", "Low-band onset"),
        ("high_band_onset", "High-band onset"),
        ("pickup_onset", "Pickup onset"),
        ("fill_onset", "Fill onset"),
    ):
        metric = summary[key]
        lines.append(
            f"| {label} | {_format_metric(metric, 'precision')} | "
            f"{_format_metric(metric, 'recall')} | {_format_metric(metric, 'f1')} | "
            f"{_format_seconds(metric.get('mean_absolute_error_seconds'))} |"
        )
    lines.extend(
        [
            "",
            "## Tempo, meter, and resolution",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Mean BPM absolute error | {summary['mean_bpm_absolute_error']:.6f} |",
            f"| Mean BPM relative error | {summary['mean_bpm_relative_error']:.6f} |",
            f"| Meter accuracy | {summary['meter_accuracy']:.6f} |",
            (
                "| Mean first-downbeat error | "
                f"{_format_seconds(summary['mean_first_downbeat_absolute_error_seconds'])} |"
            ),
            (
                "| Resolution expressible-event ratio | "
                f"{resolution['expressible_event_ratio']:.6f} |"
            ),
            (
                "| Mean resolution quantization error | "
                f"{_format_seconds(resolution['mean_quantization_error_seconds'])} |"
            ),
            f"| Under-resolved bars | {resolution['under_resolved_bar_count']} |",
            (
                "| Unnecessary high-resolution bars | "
                f"{resolution['unnecessary_high_resolution_bar_count']} |"
            ),
            f"| Resolution changes | {resolution['resolution_change_count']} |",
            f"| Silent-range onset false positives | {summary['silent_onset_false_positive_count']} |",
            "",
            "## Per-fixture metrics",
            "",
            (
                "| Audio | BPM error | Meter | Onset F1 | Beat F1 | Downbeat F1 | "
                "Band F1 | Pickup recall | Fill recall | Base res. | Expressible | Quant. error |"
            ),
            (
                "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: |"
            ),
        ]
    )
    for item in benchmark["fixtures"]:
        band_values = [
            metric["f1"]
            for metric in (item["low_band_onset"], item["high_band_onset"])
            if metric is not None
        ]
        band_f1 = sum(band_values) / len(band_values) if band_values else None
        lines.append(
            f"| `{item['audio']}` | {item['bpm_absolute_error']:.6f} | "
            f"{'✓' if item['meter_correct'] else '✗'} | {item['onset']['f1']:.6f} | "
            f"{item['beat']['f1']:.6f} | {item['downbeat']['f1']:.6f} | "
            f"{_format_optional_number(band_f1)} | "
            f"{_format_optional_event(item['pickup_onset'], 'recall')} | "
            f"{_format_optional_event(item['fill_onset'], 'recall')} | "
            f"{item['resolution']['base_resolution']} | "
            f"{item['resolution']['expressible_event_ratio']:.6f} | "
            f"{_format_seconds(item['resolution']['mean_quantization_error_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "This report is a deterministic regression baseline. Metric values may vary across "
            "librosa, BeatNet, and platform versions; CI validates structure and ranges rather "
            "than requiring cross-environment byte equality.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _envelope_peak_times(
    values: list[float],
    *,
    sample_rate: int | None,
    hop_length: int,
    threshold: float,
) -> list[float]:
    if not values or not sample_rate or sample_rate <= 0 or hop_length <= 0:
        return []
    candidates: list[tuple[float, int]] = []
    for index, value in enumerate(values):
        previous = values[index - 1] if index > 0 else 0.0
        following = values[index + 1] if index + 1 < len(values) else 0.0
        if value >= threshold and value >= previous and value > following:
            candidates.append((float(value), index))

    selected: list[tuple[float, int]] = []
    minimum_separation_frames = max(1, round(0.08 * sample_rate / hop_length))
    for value, index in sorted(candidates, reverse=True):
        if all(abs(index - selected_index) >= minimum_separation_frames for _, selected_index in selected):
            selected.append((value, index))
    return sorted(round(index * hop_length / sample_rate, 9) for _, index in selected)


def _times_in_ranges(
    values: list[float],
    ranges: list[tuple[float, float]],
    *,
    margin: float = 0.0,
) -> list[float]:
    return [
        float(value)
        for value in values
        if any(start - margin <= float(value) < end + margin for start, end in ranges)
    ]


def _optional_event_metrics(
    reference_times: list[float],
    estimated_times: list[float],
    *,
    tolerance_seconds: float,
) -> dict[str, int | float | None] | None:
    if not reference_times:
        return None
    return match_events(
        reference_times,
        estimated_times,
        tolerance_seconds=tolerance_seconds,
    )


def _build_resolution_metrics(
    ground_truth: dict[str, Any],
    analysis: AudioAnalysisRaw,
) -> dict[str, Any]:
    plan = build_resolution_plan(analysis, build_bar_features(analysis))
    time_signature = str(ground_truth["time_signature"])
    quarter_notes_per_bar = 4.0 if time_signature == "4/4" else 3.0
    bar_duration = quarter_notes_per_bar * 60.0 / float(ground_truth["bpm"])
    first_downbeat = float(ground_truth["first_downbeat"])
    candidates = [16, 24, 48] if time_signature == "4/4" else [12, 18, 36]
    events_by_bar: dict[int, list[float]] = {}
    pre_downbeat_event_count = 0
    for value in ground_truth["onsets"]:
        onset = float(value)
        relative = onset - first_downbeat
        if relative < 0:
            pre_downbeat_event_count += 1
            continue
        bar_index = max(0, floor((relative + _RESOLUTION_EPSILON_SECONDS) / bar_duration))
        position = relative - bar_index * bar_duration
        if position < 0:
            position = 0.0
        events_by_bar.setdefault(bar_index, []).append(position)

    quantization_errors: list[float] = []
    expressible_event_count = 0
    under_resolved_bar_count = 0
    unnecessary_high_resolution_bar_count = 0
    expected_minimum_counts: Counter[int] = Counter()
    selected_counts: Counter[int] = Counter()
    for bar_index in sorted(events_by_bar):
        positions = events_by_bar[bar_index]
        selected_resolution = (
            plan.bar_resolutions[bar_index]
            if bar_index < len(plan.bar_resolutions)
            else plan.base_resolution
        )
        selected_counts[selected_resolution] += 1
        selected_errors = [
            _quantization_error(position, bar_duration, selected_resolution)
            for position in positions
        ]
        quantization_errors.extend(selected_errors)
        expressible_event_count += sum(
            error <= _RESOLUTION_EPSILON_SECONDS for error in selected_errors
        )
        if any(error > _RESOLUTION_EPSILON_SECONDS for error in selected_errors):
            under_resolved_bar_count += 1

        minimum_resolution = candidates[-1]
        for candidate in candidates:
            if all(
                _quantization_error(position, bar_duration, candidate)
                <= _RESOLUTION_EPSILON_SECONDS
                for position in positions
            ):
                minimum_resolution = candidate
                break
        expected_minimum_counts[minimum_resolution] += 1
        if selected_resolution > minimum_resolution:
            unnecessary_high_resolution_bar_count += 1

    evaluated_event_count = len(quantization_errors)
    evaluated_bar_count = len(events_by_bar)
    return {
        "policy_version": plan.policy_version,
        "base_resolution": plan.base_resolution,
        "bar_resolutions": plan.bar_resolutions,
        "resolution_change_count": len(plan.change_points),
        "evaluated_bar_count": evaluated_bar_count,
        "evaluated_event_count": evaluated_event_count,
        "pre_downbeat_event_count": pre_downbeat_event_count,
        "expressible_event_count": expressible_event_count,
        "expressible_event_ratio": round(
            _safe_ratio(expressible_event_count, evaluated_event_count),
            6,
        ),
        "mean_quantization_error_seconds": (
            round(sum(quantization_errors) / evaluated_event_count, 6)
            if quantization_errors
            else None
        ),
        "max_quantization_error_seconds": (
            round(max(quantization_errors), 6) if quantization_errors else None
        ),
        "under_resolved_bar_count": under_resolved_bar_count,
        "unnecessary_high_resolution_bar_count": unnecessary_high_resolution_bar_count,
        "unnecessary_high_resolution_bar_ratio": round(
            _safe_ratio(unnecessary_high_resolution_bar_count, evaluated_bar_count),
            6,
        ),
        "high_resolution_bar_count": selected_counts[candidates[-1]],
        "high_resolution_bar_ratio": round(
            _safe_ratio(selected_counts[candidates[-1]], evaluated_bar_count),
            6,
        ),
        "selected_resolution_counts": _string_key_counts(selected_counts),
        "expected_minimum_resolution_counts": _string_key_counts(expected_minimum_counts),
    }


def _quantization_error(position: float, bar_duration: float, resolution: int) -> float:
    if resolution <= 0 or bar_duration <= 0:
        return bar_duration
    grid_duration = bar_duration / resolution
    return abs(position - round(position / grid_duration) * grid_duration)


def _aggregate_event_metrics(
    fixture_metrics: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, int | float | None]:
    return _aggregate_event_metric_values([item[metric_name] for item in fixture_metrics])


def _aggregate_optional_event_metrics(
    fixture_metrics: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, int | float | None]:
    metrics = [item[metric_name] for item in fixture_metrics if item[metric_name] is not None]
    result = _aggregate_event_metric_values(metrics)
    result["applicable_fixture_count"] = len(metrics)
    return result


def _aggregate_event_metric_values(
    metrics: list[dict[str, int | float | None]],
) -> dict[str, int | float | None]:
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


def _aggregate_resolution_metrics(fixture_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["resolution"] for item in fixture_metrics]
    evaluated_event_count = sum(int(item["evaluated_event_count"]) for item in metrics)
    expressible_event_count = sum(int(item["expressible_event_count"]) for item in metrics)
    error_pairs = [
        (int(item["evaluated_event_count"]), item["mean_quantization_error_seconds"])
        for item in metrics
        if item["mean_quantization_error_seconds"] is not None
    ]
    mean_error = (
        sum(count * float(error) for count, error in error_pairs) / evaluated_event_count
        if evaluated_event_count
        else None
    )
    selected_counts: Counter[int] = Counter()
    expected_counts: Counter[int] = Counter()
    for item in metrics:
        selected_counts.update(
            {int(key): int(value) for key, value in item["selected_resolution_counts"].items()}
        )
        expected_counts.update(
            {
                int(key): int(value)
                for key, value in item["expected_minimum_resolution_counts"].items()
            }
        )
    max_errors = [
        float(item["max_quantization_error_seconds"])
        for item in metrics
        if item["max_quantization_error_seconds"] is not None
    ]
    return {
        "evaluated_bar_count": sum(int(item["evaluated_bar_count"]) for item in metrics),
        "evaluated_event_count": evaluated_event_count,
        "pre_downbeat_event_count": sum(
            int(item["pre_downbeat_event_count"]) for item in metrics
        ),
        "expressible_event_count": expressible_event_count,
        "expressible_event_ratio": round(
            _safe_ratio(expressible_event_count, evaluated_event_count),
            6,
        ),
        "mean_quantization_error_seconds": (
            round(mean_error, 6) if mean_error is not None else None
        ),
        "max_quantization_error_seconds": round(max(max_errors), 6) if max_errors else None,
        "under_resolved_bar_count": sum(
            int(item["under_resolved_bar_count"]) for item in metrics
        ),
        "unnecessary_high_resolution_bar_count": sum(
            int(item["unnecessary_high_resolution_bar_count"]) for item in metrics
        ),
        "resolution_change_count": sum(
            int(item["resolution_change_count"]) for item in metrics
        ),
        "selected_resolution_counts": _string_key_counts(selected_counts),
        "expected_minimum_resolution_counts": _string_key_counts(expected_counts),
    }


def _string_key_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(key): counts[key] for key in sorted(counts)}


def _mean_metric(fixture_metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in fixture_metrics if item[key] is not None]
    return round(sum(values) / len(values), 6) if values else None


def _max_metric(fixture_metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in fixture_metrics if item[key] is not None]
    return round(max(values), 6) if values else None


def _format_metric(metric: dict[str, Any], key: str) -> str:
    return f"{float(metric[key]):.6f}"


def _format_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}s"


def _format_optional_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _format_optional_event(metric: dict[str, Any] | None, key: str) -> str:
    return "—" if metric is None else f"{float(metric[key]):.6f}"
