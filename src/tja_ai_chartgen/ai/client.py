import asyncio
import json
import os
import re
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

from litellm import acompletion, completion

from tja_ai_chartgen.ai.prompts import build_chart_generation_prompt
from tja_ai_chartgen.ai.rhythm_repair import (
    build_rhythm_repair_gate_metadata as _rhythm_repair_gate_metadata,
    selected_rhythm_quality_issues as _selected_rhythm_quality_issues,
)
from tja_ai_chartgen.cancellation import GenerationCancelledError, raise_if_cancelled
from tja_ai_chartgen.features.density import BarDensityHint, build_density_hints
from tja_ai_chartgen.features.resolution import (
    output_resolution_for_analysis_bar,
    output_resolution_for_bar,
)
from tja_ai_chartgen.features.rhythm_grid import (
    AI_ARBITRARY_GRID_MAX_RATE,
    AI_ARBITRARY_GRID_MIN_COUNT,
    AI_ARBITRARY_GRID_MIN_HITS,
    chart_arbitrary_grid_positions,
)
from tja_ai_chartgen.features.salience import (
    build_burst_salience,
    project_reliable_burst_span,
)
from tja_ai_chartgen.features.salience_candidates import build_salience_candidate_bars
from tja_ai_chartgen.features.silence import edge_silence_indexes
from tja_ai_chartgen.rules.rhythm_skeleton import rhythm_skeleton_issues
from tja_ai_chartgen.tja.event_encoder import EventEncodingError, encode_chart_bar_events
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    ChartBar,
    ChartBarEvents,
    ChartHitEvent,
    ChartLongNoteEvent,
    ResolutionPlan,
    SongAnalysis,
)
from tja_ai_chartgen.tja.playability import (
    BIG_NOTE_ISOLATION_SECONDS,
    find_big_note_isolation_violations,
)
from tja_ai_chartgen.tja.quality import (
    chart_activity_count,
    density_hit_count,
    empty_runs,
    normalized_hit_count,
    note_color_metrics,
    pattern_counts,
)

ALLOWED_AI_NOTES = set("01234578")
DEFAULT_AI_REPAIR_RETRIES = 2
DEFAULT_AI_REQUEST_TIMEOUT = 300.0
DEFAULT_AI_TRANSPORT_RETRIES = 3
MAX_AI_TRANSPORT_RETRIES = 5
MIN_AI_REQUEST_TIMEOUT = 1.0
MAX_AI_REQUEST_TIMEOUT = 600.0
AI_SALIENCE_VALIDATION_VERSION = "ai-salience-validation-v1"
SALIENCE_VALIDATION_EXAMPLE_LIMIT = 16
MIN_AI_SPECIAL_NOTE_DURATION_SECONDS = 0.25


class AiOutputRepairError(RuntimeError):
    def __init__(self, message: str, output: dict[str, Any]) -> None:
        super().__init__(message)
        self.output = output


class AiProviderError(RuntimeError):
    def __init__(self, message: str, output: dict[str, Any]) -> None:
        super().__init__(message)
        self.output = output


class _AiProviderCallError(RuntimeError):
    def __init__(self, error: Exception, fallback_reason: str) -> None:
        super().__init__(str(error))
        self.error = error
        self.fallback_reason = fallback_reason


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
    attempt_log_path: Path | None = None,
    request_timeout: float = DEFAULT_AI_REQUEST_TIMEOUT,
    max_transport_retries: int = DEFAULT_AI_TRANSPORT_RETRIES,
    cancel_event: Event | None = None,
) -> tuple[list[ChartBar], dict[str, Any]]:
    if not MIN_AI_REQUEST_TIMEOUT <= request_timeout <= MAX_AI_REQUEST_TIMEOUT:
        raise ValueError("AI request timeout must be between 1 and 600 seconds")
    if not 0 <= max_transport_retries <= MAX_AI_TRANSPORT_RETRIES:
        raise ValueError(
            f"AI transport retries must be between 0 and {MAX_AI_TRANSPORT_RETRIES}"
        )

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
    transport_attempts: list[dict[str, Any]] = []

    for attempt_index in range(repair_attempts + 1):
        raise_if_cancelled(cancel_event)
        try:
            response = _completion_with_transport_retries(
                model=model_name,
                messages=messages,
                api_base=resolved_api_base,
                api_key=resolved_api_key,
                content_attempt=attempt_index + 1,
                request_timeout=request_timeout,
                max_transport_retries=max_transport_retries,
                transport_attempts=transport_attempts,
                cancel_event=cancel_event,
            )
        except _AiProviderCallError as error:
            output = _build_ai_output(
                model=model_name,
                api_base=resolved_api_base,
                api_key_provided=bool(resolved_api_key),
                max_repair_attempts=repair_attempts,
                request_timeout=request_timeout,
                max_transport_retries=max_transport_retries,
                attempts=attempts,
                transport_attempts=transport_attempts,
                fallback_reason=error.fallback_reason,
                final=None,
                api_key=resolved_api_key,
            )
            _write_attempt_log(
                attempt_log_path,
                model=model_name,
                api_base=resolved_api_base,
                api_key_provided=bool(resolved_api_key),
                max_repair_attempts=repair_attempts,
                request_timeout=request_timeout,
                max_transport_retries=max_transport_retries,
                attempts=attempts,
                transport_attempts=transport_attempts,
                fallback_reason=error.fallback_reason,
                final=None,
                api_key=resolved_api_key,
            )
            safe_message = _redact_sensitive_value(str(error.error), resolved_api_key)
            raise AiProviderError(
                f"AI provider request failed ({type(error.error).__name__}): {safe_message}",
                output,
            ) from None

        content = _extract_response_content(response)

        try:
            data = _parse_ai_json(content)
            bars = _validate_ai_data(
                data,
                analysis=analysis,
                course=course,
                level=level,
                style=style,
                density=density,
                special_notes=special_notes,
            )
        except _AiOutputValidationError as error:
            issues = error.issues
        except json.JSONDecodeError as error:
            issues = [f"output must be valid JSON: {error.msg}"]
        else:
            salience_validation = build_ai_salience_validation_report(bars, analysis)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "ok",
                    "content": content,
                    "data": data,
                    "salience_validation": salience_validation,
                }
            )
            output = _build_ai_output(
                model=model_name,
                api_base=resolved_api_base,
                api_key_provided=bool(resolved_api_key),
                max_repair_attempts=repair_attempts,
                request_timeout=request_timeout,
                max_transport_retries=max_transport_retries,
                attempts=attempts,
                transport_attempts=transport_attempts,
                fallback_reason=None,
                final=data,
                api_key=resolved_api_key,
                salience_validation=salience_validation,
            )
            should_write_attempts = any(
                attempt.get("status") == "invalid" for attempt in attempts
            ) or any(attempt.get("status") == "error" for attempt in transport_attempts)
            if attempt_log_path and should_write_attempts:
                _write_attempt_log(
                    attempt_log_path,
                    model=model_name,
                    api_base=resolved_api_base,
                    api_key_provided=bool(resolved_api_key),
                    max_repair_attempts=repair_attempts,
                    request_timeout=request_timeout,
                    max_transport_retries=max_transport_retries,
                    attempts=attempts,
                    transport_attempts=transport_attempts,
                    fallback_reason=None,
                    final=data,
                    api_key=resolved_api_key,
                    salience_validation=salience_validation,
                )
            return bars, output

        attempts.append(
            {
                "attempt": attempt_index + 1,
                "status": "invalid",
                "content": content,
                "issues": issues,
            }
        )
        _write_attempt_log(
            attempt_log_path,
            model=model_name,
            api_base=resolved_api_base,
            api_key_provided=bool(resolved_api_key),
            max_repair_attempts=repair_attempts,
            request_timeout=request_timeout,
            max_transport_retries=max_transport_retries,
            attempts=attempts,
            transport_attempts=transport_attempts,
            fallback_reason=None,
            final=None,
            api_key=resolved_api_key,
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
        request_timeout=request_timeout,
        max_transport_retries=max_transport_retries,
        attempts=attempts,
        transport_attempts=transport_attempts,
        fallback_reason="invalid_content_retries_exhausted",
        final=None,
        api_key=resolved_api_key,
    )
    _write_attempt_log(
        attempt_log_path,
        model=model_name,
        api_base=resolved_api_base,
        api_key_provided=bool(resolved_api_key),
        max_repair_attempts=repair_attempts,
        request_timeout=request_timeout,
        max_transport_retries=max_transport_retries,
        attempts=attempts,
        transport_attempts=transport_attempts,
        fallback_reason="invalid_content_retries_exhausted",
        final=None,
        api_key=resolved_api_key,
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
    resolution_plan: ResolutionPlan | None = None,
) -> list[ChartBar]:
    if len(bars) != expected_count:
        raise ValueError(
            f"AI output must contain exactly {expected_count} bars, got {len(bars)}"
        )

    sanitized: list[ChartBar] = []
    issues: list[str] = []
    silent_indexes = edge_silence_indexes(expected_bars or [])

    for index, bar in enumerate(bars):
        expected_length = (
            output_resolution_for_bar(
                expected_bars[index],
                plan=resolution_plan,
                position=index,
            )
            if expected_bars and index < len(expected_bars)
            else 16
        )
        expected_time_signature = (
            expected_bars[index].time_signature
            if expected_bars and index < len(expected_bars)
            else bar.time_signature
        )
        if len(bar.notes) != expected_length:
            issues.append(
                f"bar {index + 1} expected resolution {expected_length}, got {len(bar.notes)}"
            )
        illegal_characters = sorted(set(bar.notes) - ALLOWED_AI_NOTES)
        if illegal_characters:
            issues.append(
                f"bar {index + 1} contains unsupported character(s): "
                f"{''.join(illegal_characters)}"
            )
        if len(bar.balloon_counts) != bar.notes.count("7"):
            issues.append(
                f"bar {index + 1} must provide exactly {bar.notes.count('7')} balloon count(s)"
            )
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in bar.balloon_counts
        ):
            issues.append(f"bar {index + 1} balloon counts must be positive integers")

        notes = "0" * expected_length if index in silent_indexes else bar.notes
        balloon_counts = [] if index in silent_indexes else bar.balloon_counts
        sanitized.append(
            ChartBar(
                index=index,
                notes=notes,
                time_signature=expected_time_signature,
                balloon_counts=balloon_counts,
            )
        )

    if issues:
        raise ValueError("AI chart failed defensive validation: " + "; ".join(issues))
    return sanitized


def build_ai_salience_validation_report(
    bars: list[ChartBar],
    analysis: SongAnalysis,
) -> dict[str, Any]:
    """生成首轮仅报告的 AI/salience 对齐诊断，不参与 repair gate。"""
    candidate_bars = build_salience_candidate_bars(
        analysis.bars,
        resolution_plan=analysis.resolution_plan,
    )
    silent_indexes = edge_silence_indexes(analysis.bars)
    normal_note_count = 0
    representable_note_count = 0
    unsupported_notes: list[list[int]] = []
    missed_reliable: list[list[int]] = []
    missed_strong: list[list[int]] = []
    reliable_candidate_count = 0
    reliable_candidate_hit_count = 0
    strong_transient_count = 0
    strong_transient_hit_count = 0
    silent_bar_note_count = 0
    longest_strong_transient_miss_run = 0
    current_strong_transient_miss_run = 0

    for position, (feature_bar, candidates) in enumerate(
        zip(analysis.bars, candidate_bars, strict=True)
    ):
        chart_bar = bars[position] if position < len(bars) else None
        selected_grids: set[int] = set()
        if chart_bar is not None:
            if position in silent_indexes:
                silent_bar_note_count += chart_activity_count(chart_bar.notes)
            normal_note_count += sum(note in "1234" for note in chart_bar.notes)
            expected_resolution = output_resolution_for_analysis_bar(analysis, position)
            if len(chart_bar.notes) == expected_resolution:
                step = feature_bar.grids_per_bar // expected_resolution
                selected_grids = {
                    grid * step
                    for grid, note in enumerate(chart_bar.notes)
                    if note in "1234"
                }
                representable_note_count += len(selected_grids)

        candidate_grids = {candidate.grid for candidate in candidates}
        unsupported_notes.extend(
            [position + 1, grid]
            for grid in sorted(selected_grids - candidate_grids)
        )

        for candidate in candidates:
            if not candidate.reliable:
                continue
            reliable_candidate_count += 1
            matched = candidate.grid in selected_grids
            if matched:
                reliable_candidate_hit_count += 1
            elif len(missed_reliable) < SALIENCE_VALIDATION_EXAMPLE_LIMIT:
                missed_reliable.append([position + 1, candidate.grid])

            if candidate.kind != "strong-transient":
                continue
            strong_transient_count += 1
            if matched:
                strong_transient_hit_count += 1
                current_strong_transient_miss_run = 0
            else:
                current_strong_transient_miss_run += 1
                longest_strong_transient_miss_run = max(
                    longest_strong_transient_miss_run,
                    current_strong_transient_miss_run,
                )
                if len(missed_strong) < SALIENCE_VALIDATION_EXAMPLE_LIMIT:
                    missed_strong.append([position + 1, candidate.grid])

    unsupported_note_count = len(unsupported_notes)
    unrepresentable_note_count = normal_note_count - representable_note_count
    return {
        "schema": AI_SALIENCE_VALIDATION_VERSION,
        "report_only": True,
        "normal_note_count": normal_note_count,
        "representable_note_count": representable_note_count,
        "unrepresentable_note_count": unrepresentable_note_count,
        "reliable_candidate_count": reliable_candidate_count,
        "reliable_candidate_hit_count": reliable_candidate_hit_count,
        "reliable_candidate_coverage": _ratio(
            reliable_candidate_hit_count,
            reliable_candidate_count,
            empty=1.0,
        ),
        "strong_transient_count": strong_transient_count,
        "strong_transient_hit_count": strong_transient_hit_count,
        "strong_transient_coverage": _ratio(
            strong_transient_hit_count,
            strong_transient_count,
            empty=1.0,
        ),
        "longest_strong_transient_miss_run": longest_strong_transient_miss_run,
        "unsupported_note_count": unsupported_note_count,
        "unsupported_note_ratio": _ratio(
            unsupported_note_count,
            normal_note_count,
            empty=0.0,
        ),
        "silent_bar_note_count": silent_bar_note_count,
        "unsupported_note_examples": unsupported_notes[:SALIENCE_VALIDATION_EXAMPLE_LIMIT],
        "missed_reliable_examples": missed_reliable,
        "missed_strong_transient_examples": missed_strong,
    }


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return round(numerator / denominator if denominator else empty, 6)


def _completion_with_transport_retries(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_base: str | None,
    api_key: str | None,
    content_attempt: int,
    request_timeout: float,
    max_transport_retries: int,
    transport_attempts: list[dict[str, Any]],
    cancel_event: Event | None,
) -> Any:
    for transport_attempt in range(1, max_transport_retries + 2):
        raise_if_cancelled(cancel_event)
        started_at = perf_counter()
        try:
            completion_kwargs = _completion_kwargs(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                request_timeout=request_timeout,
            )
            response = (
                asyncio.run(_cancelable_acompletion(completion_kwargs, cancel_event))
                if cancel_event is not None
                else completion(**completion_kwargs)
            )
        except GenerationCancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - all provider call failures share one retry budget.
            transport_attempts.append(
                {
                    "content_attempt": content_attempt,
                    "transport_attempt": transport_attempt,
                    "status": "error",
                    "elapsed_seconds": max(0.0, perf_counter() - started_at),
                    "error_type": type(error).__name__,
                    "error": _redact_sensitive_value(str(error), api_key),
                }
            )
            if transport_attempt <= max_transport_retries:
                continue
            raise _AiProviderCallError(error, "transport_retries_exhausted") from None
        else:
            transport_attempts.append(
                {
                    "content_attempt": content_attempt,
                    "transport_attempt": transport_attempt,
                    "status": "ok",
                    "elapsed_seconds": max(0.0, perf_counter() - started_at),
                }
            )
            return response

    raise AssertionError("transport retry loop exited without a response")


async def _cancelable_acompletion(
    completion_kwargs: dict[str, Any],
    cancel_event: Event,
) -> Any:
    raise_if_cancelled(cancel_event)
    request_task = asyncio.create_task(acompletion(**completion_kwargs))
    while not request_task.done():
        if cancel_event.is_set():
            request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass
            raise GenerationCancelledError("Generation job was cancelled by the user")
        await asyncio.wait({request_task}, timeout=0.05)
    return await request_task


def _completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_base: str | None,
    api_key: str | None,
    request_timeout: float,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "timeout": request_timeout,
        "max_retries": 0,
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
    style: str,
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

    burst_salience_bars = build_burst_salience(analysis.bars)
    bars: list[ChartBar] = []
    for index, item in enumerate(raw_bars):
        if index >= expected_count:
            issues.append(f"bars[{index}] has no matching input bar")
            continue
        if not isinstance(item, dict):
            issues.append(f"bars[{index}] must be an object")
            continue
        if item.get("bar", index + 1) != index + 1:
            issues.append(f"bars[{index}].bar must be {index + 1}")

        if "notes" in item or "balloon_counts" in item:
            issues.append(
                f"bars[{index}] uses the legacy notes schema; output hits and long_notes instead"
            )
            chart_bar = None
        else:
            chart_bar = _parse_event_bar(
                item,
                index=index,
                analysis=analysis,
                burst_salience=burst_salience_bars[index],
                special_notes=special_notes,
                issues=issues,
            )
        if chart_bar is not None:
            bars.append(chart_bar)

    if not issues:
        issues.extend(_validate_big_note_isolation(bars, analysis.bars))
    if not issues:
        issues.extend(_validate_rhythmic_grid_stability(bars, analysis.bars))
    if not issues:
        issues.extend(
            rhythm_skeleton_issues(
                bars,
                analysis,
                course=course,
                level=level,
                style=style,
                density=density,
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


def _parse_event_bar(
    item: dict[str, Any],
    *,
    index: int,
    analysis: SongAnalysis,
    burst_salience: BarRhythmicSalience,
    special_notes: bool,
    issues: list[str],
) -> ChartBar | None:
    issue_count = len(issues)
    raw_hits = item.get("hits", [])
    raw_long_notes = item.get("long_notes", [])
    if not isinstance(raw_hits, list):
        issues.append(f"bars[{index}].hits must be a list")
        return None
    if not isinstance(raw_long_notes, list):
        issues.append(f"bars[{index}].long_notes must be a list")
        return None

    hits: list[ChartHitEvent] = []
    for hit_index, raw_hit in enumerate(raw_hits):
        if not isinstance(raw_hit, list) or len(raw_hit) != 2:
            issues.append(f"bars[{index}].hits[{hit_index}] must be [tick, note]")
            continue
        tick, note = raw_hit
        if not isinstance(tick, int) or isinstance(tick, bool):
            issues.append(f"bars[{index}].hits[{hit_index}][0] must be an integer")
            continue
        if isinstance(note, int) and not isinstance(note, bool) and note in {1, 2, 3, 4}:
            note = str(note)
        if not isinstance(note, str):
            issues.append(f"bars[{index}].hits[{hit_index}][1] must be a string or integer 1-4")
            continue
        hits.append(ChartHitEvent(tick=tick, note=note))

    long_notes: list[ChartLongNoteEvent] = []
    for note_index, raw_note in enumerate(raw_long_notes):
        if not isinstance(raw_note, dict):
            issues.append(f"bars[{index}].long_notes[{note_index}] must be an object")
            continue
        start_tick = raw_note.get("start_tick")
        end_tick = raw_note.get("end_tick")
        kind = raw_note.get("kind")
        balloon_count = raw_note.get("balloon_count")
        if not isinstance(start_tick, int) or isinstance(start_tick, bool):
            issues.append(
                f"bars[{index}].long_notes[{note_index}].start_tick must be an integer"
            )
            continue
        if not isinstance(end_tick, int) or isinstance(end_tick, bool):
            issues.append(
                f"bars[{index}].long_notes[{note_index}].end_tick must be an integer"
            )
            continue
        if not isinstance(kind, str):
            issues.append(f"bars[{index}].long_notes[{note_index}].kind must be a string")
            continue
        if balloon_count is not None and (
            not isinstance(balloon_count, int) or isinstance(balloon_count, bool)
        ):
            issues.append(
                f"bars[{index}].long_notes[{note_index}].balloon_count must be an integer"
            )
            continue
        long_notes.append(
            ChartLongNoteEvent(
                start_tick=start_tick,
                end_tick=end_tick,
                kind=kind,
                balloon_count=balloon_count,
            )
        )

    if long_notes and not special_notes:
        issues.append(f"bars[{index}].long_notes must be empty when special_notes is false")
        return None
    if len(issues) > issue_count:
        return None

    feature_bar = analysis.bars[index]
    output_resolution = output_resolution_for_analysis_bar(analysis, index)
    if long_notes:
        issues.extend(
            _validate_ai_long_notes(
                long_notes,
                index=index,
                feature_bar=feature_bar,
                burst_salience=burst_salience,
                output_resolution=output_resolution,
            )
        )
    if len(issues) > issue_count:
        return None

    try:
        return encode_chart_bar_events(
            ChartBarEvents(index=index, hits=hits, long_notes=long_notes),
            canonical_grids_per_bar=feature_bar.grids_per_bar,
            output_resolution=output_resolution,
            time_signature=feature_bar.time_signature,
        )
    except EventEncodingError as error:
        issues.extend(
            f"bars[{index}].{issue.code}: {issue.message}" for issue in error.issues
        )
        return None


def _validate_ai_long_notes(
    long_notes: list[ChartLongNoteEvent],
    *,
    index: int,
    feature_bar: BarFeature,
    burst_salience: BarRhythmicSalience,
    output_resolution: int,
) -> list[str]:
    issues: list[str] = []
    if len(long_notes) > 1:
        issues.append(f"bars[{index}].long_notes must contain at most one event")

    burst_span = project_reliable_burst_span(
        feature_bar,
        burst_salience,
        output_resolution=output_resolution,
    )
    if burst_span is None:
        issues.append(f"bars[{index}].long_notes require reliable burst salience")
        return issues

    burst_start, burst_end = burst_span
    bar_duration = feature_bar.end_time - feature_bar.start_time
    for note_index, long_note in enumerate(long_notes):
        if long_note.start_tick < burst_start or long_note.end_tick > burst_end:
            issues.append(
                f"bars[{index}].long_notes[{note_index}] must stay inside reliable burst "
                f"range {burst_start}..{burst_end}"
            )
        duration = (
            bar_duration
            * (long_note.end_tick - long_note.start_tick)
            / feature_bar.grids_per_bar
        )
        if duration < MIN_AI_SPECIAL_NOTE_DURATION_SECONDS:
            issues.append(
                f"bars[{index}].long_notes[{note_index}] duration must be at least "
                f"{MIN_AI_SPECIAL_NOTE_DURATION_SECONDS:.2f} seconds"
            )
    return issues


def _validate_big_note_isolation(
    bars: list[ChartBar],
    expected_bars: list[BarFeature],
) -> list[str]:
    return [
        (
            f"bars[{violation.bar_position}].hits has big note {violation.note} at "
            f"canonical tick {violation.canonical_tick} only "
            f"{violation.nearest_hit_distance_seconds:.3f}s from another playable hit; "
            f"big notes require more than {BIG_NOTE_ISOLATION_SECONDS:.2f}s isolation "
            "on both sides, so use normal note 1/2 instead"
        )
        for violation in find_big_note_isolation_violations(bars, expected_bars)
    ]


def _validate_rhythmic_grid_stability(
    bars: list[ChartBar],
    expected_bars: list[BarFeature],
) -> list[str]:
    evaluated, arbitrary = chart_arbitrary_grid_positions(bars, expected_bars)
    if evaluated < AI_ARBITRARY_GRID_MIN_HITS:
        return []
    rate = len(arbitrary) / evaluated
    if len(arbitrary) < AI_ARBITRARY_GRID_MIN_COUNT or rate <= AI_ARBITRARY_GRID_MAX_RATE:
        return []
    examples = ", ".join(
        f"bar {bar_position + 1} tick {canonical_tick}"
        for bar_position, _grid, canonical_tick in arbitrary[:8]
    )
    return [
        (
            "chart rhythm uses too many arbitrary finest-grid positions: "
            f"{len(arbitrary)}/{evaluated} ({rate:.1%}), maximum "
            f"{AI_ARBITRARY_GRID_MAX_RATE:.0%}; prefer stable straight ticks divisible "
            "by 3 or triplet/24th ticks divisible by 2, and reserve other ticks for "
            f"rare explicit microtiming ({examples})"
        )
    ]


def _validate_edge_silence(bars: list[ChartBar], expected_bars: list[BarFeature]) -> list[str]:
    issues: list[str] = []
    for index in sorted(edge_silence_indexes(expected_bars)):
        if index >= len(bars):
            continue
        if chart_activity_count(bars[index].notes) == 0:
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
    issues: list[str] = []
    expected_bars = analysis.bars[: len(bars)]
    density_hints = build_density_hints(expected_bars)
    hit_counts = [density_hit_count(bar.notes) for bar in bars]
    issues.extend(_density_hint_issues(hit_counts, density_hints))
    issues.extend(
        _selected_rhythm_quality_issues(
            bars,
            analysis=analysis,
        )
    )

    if len(bars) < 8:
        return issues

    quality_density = _quality_density(density, course, level)
    normalized_hit_counts = [
        normalized_hit_count(bar.notes, expected_length=_expected_note_length(analysis, index))
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
    quality_bars = [bars[index] for index in quality_indexes if index < len(bars)]
    normal_don, normal_ka, longest_don_run = note_color_metrics(quality_bars)
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
    return empty_runs(hit_counts, quality_indexes)


def _overused_patterns(bars: list[ChartBar]) -> list[tuple[str, int]]:
    counts = pattern_counts(bars)
    limit = max(4, len(bars) // 12)
    return sorted(
        ((pattern, count) for pattern, count in counts.items() if count > limit),
        key=lambda item: item[1],
        reverse=True,
    )


def _compact_repair_issues(issues: list[str], *, max_groups: int = 40) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for issue in issues:
        signature = re.sub(r"\[\d+\]", "[*]", issue)
        grouped.setdefault(signature, []).append(issue)

    compact: list[str] = []
    for signature, examples in list(grouped.items())[:max_groups]:
        if len(examples) == 1:
            compact.append(examples[0])
            continue
        compact.append(
            f"{signature} ({len(examples)} occurrences; examples: "
            f"{'; '.join(examples[:3])})"
        )
    omitted = len(grouped) - max_groups
    if omitted > 0:
        compact.append(f"{omitted} additional validation error group(s) omitted")
    return compact


def _build_repair_prompt(
    issues: list[str],
    analysis: SongAnalysis,
    course: str,
    level: int,
    density: str,
    special_notes: bool,
) -> str:
    forced_silent_bars = [index + 1 for index in sorted(edge_silence_indexes(analysis.bars))]
    compact_issues = _compact_repair_issues(issues)
    return f"""
Your previous output was invalid and cannot be used as a TJA chart draft.

Fix the output and return JSON only.

Validation errors (repeated errors are grouped):
{json.dumps(compact_issues, ensure_ascii=False, separators=(",", ":"))}

Required schema:
{{
  "bars": [
    {{"bar":1,"hits":[[0,"1"],[12,"2"]],"long_notes":[]}}
  ]
}}

Rules:
- bars must contain exactly {len(analysis.bars)} item(s).
- Use the canonical ticks, per-bar resolution, timing, density, structure, and bar_salience from the original input, plus its rhythm_skeleton. Do not output a resolution.
- rhythm_skeleton is the deterministic timing authority. Restore its missing ordinary-hit positions and remove hits outside it when validation reports skeleton coverage or extra-hit errors. A reliable long note may replace only skeleton hits inside its own span; color and accent changes must not move timing positions.
- Prioritize reliable strong-transient, transient, rhythmic-skeleton, and structure-highlight salience points. Keep unsupported hits rare and use only short supported connectors when density requires them.
- Keep a stable rhythmic lattice. On 48/36 canonical grids, ordinary straight notes normally use ticks divisible by 3 and triplet/24th passages use ticks divisible by 2. Remove detector microtiming jitter such as alternating 5/7 gaps, and keep ticks outside both subgrids below 10% of playable hits.
- Stem onset timing is supporting evidence, not permission to copy separation latency or adjacent onset smearing. Merge nearby stem peaks such as 12/13 into the stable phrase grid.
- Use only hits and long_notes; never return legacy notes or balloon_counts fields.
- Normal hit notes are 1, 2, 3, or 4.
- Big notes 3/4 require more than 0.25 seconds of real-time isolation from every other playable hit before and after them, including across bar boundaries. Otherwise keep the same color as normal note 1/2.
- Rapid alternating-hand cells equivalent to 102 or 1002 must use only normal 1/2 notes when the real-time gap between their hits is 0.25 seconds or less; never upgrade either hit to 3/4.
- long_notes must be empty unless special_notes is true. Use at most one per bar, require reliable burst salience, and keep its ticks inside the projected burst range. Structural fill_candidate may strengthen the choice but cannot replace the burst gate. Balloons require a positive balloon_count.
- Course/difficulty request: {course} level {level}, density {density}.
- Forced silent bars: {forced_silent_bars}. These song-start/song-end silence bars must have empty hits and long_notes.
- Respect bar_density_hints from the original input: keep each bar inside its min_hits/max_hits range and aim near target_hits when present; rest bars may stay empty, sparse bars may stay light, normal bars should be moderate, and dense/fill bars should be busier without sudden full-density spikes unless the hint allows it.
- If a bar has high activity/strength but sparse onset markers, treat it as sustained music rather than silence; add a simple beat/downbeat skeleton instead of leaving it empty.
- Preserve the original phrase_id, transition_role, section_id, fill score, and resolution. Build-ups should rise across phrase_progress; breakdowns keep a light skeleton; low fill scores must not create mechanical 4/8-bar fills.
- If validation errors mention low chart quality, increase 1/2 note density on normal/dense/fill bars toward target_hits, keep real rest bars empty or sparse, add more 2/4 ka notes for offbeat/answer/fill hits, and vary repeated or all-don patterns without changing bar count or analyzed timing/structure.
- If validation errors mention chart rhythm, remove every note from reliable silent ranges, move extreme unsupported hits onto nearby salience/beat/activity evidence, and respond to a few reliable strong onsets. Do not map every onset or fill intentional rest space just to improve a metric.
- Do not include markdown, comments, explanations, or extra text.
""".strip()


def _build_ai_output(
    *,
    model: str,
    api_base: str | None,
    api_key_provided: bool,
    max_repair_attempts: int,
    request_timeout: float,
    max_transport_retries: int,
    attempts: list[dict[str, Any]],
    transport_attempts: list[dict[str, Any]],
    fallback_reason: str | None,
    final: dict[str, Any] | None,
    api_key: str | None,
    salience_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = {
        "model": model,
        "api_base": api_base,
        "api_key_provided": api_key_provided,
        "max_repair_attempts": max_repair_attempts,
        "request_timeout": request_timeout,
        "max_transport_retries": max_transport_retries,
        "rhythm_repair_gate": _rhythm_repair_gate_metadata(),
        "attempts": attempts,
        "transport_attempts": transport_attempts,
        "fallback_reason": fallback_reason,
        "final": final,
    }
    if salience_validation is not None:
        output["salience_validation"] = salience_validation
    return _redact_sensitive_value(output, api_key)


def _write_attempt_log(
    path: Path | None,
    *,
    model: str,
    api_base: str | None,
    api_key_provided: bool,
    max_repair_attempts: int,
    request_timeout: float,
    max_transport_retries: int,
    attempts: list[dict[str, Any]],
    transport_attempts: list[dict[str, Any]],
    fallback_reason: str | None,
    final: dict[str, Any] | None,
    api_key: str | None,
    salience_validation: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _build_ai_output(
                model=model,
                api_base=api_base,
                api_key_provided=api_key_provided,
                max_repair_attempts=max_repair_attempts,
                request_timeout=request_timeout,
                max_transport_retries=max_transport_retries,
                attempts=attempts,
                transport_attempts=transport_attempts,
                fallback_reason=fallback_reason,
                final=final,
                api_key=api_key,
                salience_validation=salience_validation,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _redact_sensitive_value(value: Any, api_key: str | None) -> Any:
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED]") if api_key else value
    if isinstance(value, dict):
        return {
            _redact_sensitive_value(key, api_key): _redact_sensitive_value(item, api_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_value(item, api_key) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_value(item, api_key) for item in value)
    return value



def _expected_note_length(analysis: SongAnalysis, index: int) -> int:
    return output_resolution_for_analysis_bar(analysis, index)


def _extract_response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])

    choices = response.choices
    message = choices[0].message
    return str(message.content)
