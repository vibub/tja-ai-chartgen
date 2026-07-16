from __future__ import annotations

from typing import Any

from tja_ai_chartgen.tja.model import ChartBar, SongAnalysis
from tja_ai_chartgen.tja.quality import build_quality_report

AI_RHYTHM_REPAIR_GATE_VERSION = "ai-rhythm-repair-gate-v1"
AI_REPAIR_UNSUPPORTED_NOTE_MIN_EVALUATED = 8
AI_REPAIR_UNSUPPORTED_NOTE_MIN_COUNT = 4
AI_REPAIR_UNSUPPORTED_NOTE_RATE = 0.5
AI_REPAIR_STRONG_ONSET_MIN_BARS = 4
AI_REPAIR_STRONG_ONSET_MIN_EVALUATED = 8


def selected_rhythm_quality_issues(
    bars: list[ChartBar],
    *,
    analysis: SongAnalysis,
) -> list[str]:
    """只把经 fixture 校准的极端节奏错误接入 AI repair。"""
    feature_bars = analysis.bars[: len(bars)]
    report = build_quality_report(
        bars,
        feature_bars,
        analysis.resolution_plan,
    )
    issues: list[str] = []
    if report.silent_range_violation_count:
        issues.append(
            "chart rhythm has notes inside reliable silent ranges: "
            f"{report.silent_range_violation_count} activity start(s) across "
            f"{report.silent_range_evaluated_bar_count} silent bar(s); remove all hits "
            "and long-note starts from those bars"
        )

    if (
        report.unsupported_note_evaluated_count >= AI_REPAIR_UNSUPPORTED_NOTE_MIN_EVALUATED
        and report.unsupported_note_count >= AI_REPAIR_UNSUPPORTED_NOTE_MIN_COUNT
        and report.unsupported_note_rate >= AI_REPAIR_UNSUPPORTED_NOTE_RATE
    ):
        issues.append(
            "chart rhythm has an extreme unsupported-note rate: "
            f"{report.unsupported_note_count}/{report.unsupported_note_evaluated_count} "
            f"({report.unsupported_note_rate:.3f}); move most normal hits onto reliable "
            "salience, beat/downbeat, activity, or structure evidence and keep unsupported "
            "connectors within 0.3 seconds of a directly supported hit"
        )

    if (
        len(feature_bars) >= AI_REPAIR_STRONG_ONSET_MIN_BARS
        and report.strong_onset_evaluated_count >= AI_REPAIR_STRONG_ONSET_MIN_EVALUATED
        and report.strong_onset_responded_count == 0
    ):
        issues.append(
            "chart rhythm ignores every reliable strong onset across a multi-bar range: "
            f"0/{report.strong_onset_evaluated_count} responded; add a small number of "
            "nearby normal hits or valid burst-backed long notes without mapping every onset"
        )
    return issues


def build_rhythm_repair_gate_metadata() -> dict[str, Any]:
    return {
        "version": AI_RHYTHM_REPAIR_GATE_VERSION,
        "selected_metrics": {
            "silent_range_violation": {"maximum_violation_count": 0},
            "unsupported_note_rate": {
                "minimum_evaluated_count": AI_REPAIR_UNSUPPORTED_NOTE_MIN_EVALUATED,
                "minimum_unsupported_count": AI_REPAIR_UNSUPPORTED_NOTE_MIN_COUNT,
                "repair_at_or_above_rate": AI_REPAIR_UNSUPPORTED_NOTE_RATE,
            },
            "strong_onset_response": {
                "minimum_bar_count": AI_REPAIR_STRONG_ONSET_MIN_BARS,
                "minimum_evaluated_count": AI_REPAIR_STRONG_ONSET_MIN_EVALUATED,
                "minimum_responded_count": 1,
            },
        },
        "report_only_metrics": [
            "note_onset_alignment",
            "downbeat_response",
            "fill_burst_alignment",
            "rhythmic_quantization_error",
            "salience_coverage_by_density",
        ],
    }
