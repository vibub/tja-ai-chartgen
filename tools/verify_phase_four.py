from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import sys
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.audio.analyze import analyze_audio
from tja_ai_chartgen.evaluation.audio_benchmark import build_fixture_audio_metrics
from tja_ai_chartgen.evaluation.phase_four_acceptance import (
    FOCUSED_FIXTURES,
    PHASE_FOUR_FIXTURE_SET_VERSION,
    build_phase_four_acceptance_report,
    build_phase_four_behavior_matrix,
    render_phase_four_acceptance_markdown,
)
from tja_ai_chartgen.utils.paths import write_json

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_four_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_four_acceptance.md"
DEFAULT_FIXTURE_FILENAME = "click_4_4.wav"


def run_phase_four_acceptance(fixture_dir: Path) -> dict[str, Any]:
    fixture_runs: list[dict[str, Any]] = []
    behavior_runs: list[dict[str, Any]] = []
    default_runs: list[dict[str, Any]] = []
    with _blocked_network_connections():
        for _run_index in range(2):
            fixture_runs.append(_benchmark_focused_fixtures(fixture_dir))
            behavior_runs.append(build_phase_four_behavior_matrix())
            default_runs.append(_benchmark_default_path(fixture_dir))
    return build_phase_four_acceptance_report(
        fixture_runs,
        behavior_runs,
        default_runs,
        network_blocked=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline Phase 4 tempo-arbitration acceptance checks."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        result = run_phase_four_acceptance(args.fixture_dir)
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_four_acceptance_markdown(result),
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 4 acceptance failed: {error}", file=sys.stderr)
        return 1

    summary = result["fixture_evidence"]["summary"]
    print(
        f"Phase 4 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['fixture_count']} focused fixtures, "
        f"rejected={summary['rejected_fixture_count']}, "
        f"beat regressions={summary['beat_f1_regression_count']}, "
        f"downbeat regressions={summary['downbeat_f1_regression_count']})"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


def _benchmark_focused_fixtures(fixture_dir: Path) -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    for audio_name, category in FOCUSED_FIXTURES:
        audio_path = fixture_dir / audio_name
        event_path = fixture_dir / f"{Path(audio_name).stem}.events.json"
        ground_truth = _read_json(event_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing Phase 4 fixture: {audio_path}")
        baseline = analyze_audio(audio_path, use_beatnet=False)
        enhanced = analyze_audio(audio_path, use_beatnet=True)
        baseline_metrics = build_fixture_audio_metrics(ground_truth, baseline)
        enhanced_metrics = build_fixture_audio_metrics(ground_truth, enhanced)
        decision = enhanced.tempo_analysis
        if decision is None:
            raise ValueError(f"Missing tempo decision for {audio_name}")
        variation = decision.tempo_variation
        fixtures.append(
            {
                "audio": audio_name,
                "category": category,
                "expected_bpm": float(ground_truth["bpm"]),
                "expected_time_signature": ground_truth["time_signature"],
                "baseline": _metric_summary(baseline_metrics),
                "enhanced": _metric_summary(enhanced_metrics),
                "beatnet_analysis_status": enhanced.beatnet_analysis_status,
                "decision": {
                    "decision_version": decision.decision_version,
                    "selected_source": decision.selected_source,
                    "tempo_source": decision.tempo_source,
                    "meter_source": decision.meter_source,
                    "partial_adoption": decision.partial_adoption,
                    "ambiguous": decision.ambiguous,
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "candidate_rejections": decision.candidate_rejections,
                    "candidate_sources": [
                        candidate.source for candidate in enhanced.tempo_candidates
                    ],
                    "selected_candidate_count": sum(
                        candidate.selected for candidate in enhanced.tempo_candidates
                    ),
                    "tempo_variation_version": (
                        variation.diagnostic_version if variation is not None else None
                    ),
                    "tempo_variation_source": (
                        variation.source if variation is not None else None
                    ),
                    "tempo_variation_classification": (
                        variation.classification if variation is not None else None
                    ),
                    "fixed_bpm_constrained": (
                        variation.fixed_bpm_constrained
                        if variation is not None
                        else None
                    ),
                },
            }
        )

    summary = {
        "beatnet_complete_count": sum(
            item["beatnet_analysis_status"] == "complete" for item in fixtures
        ),
        "rejected_fixture_count": sum(
            bool(item["decision"]["candidate_rejections"])
            and "beatnet" not in item["enhanced"]["analyzer"]
            for item in fixtures
        ),
        "full_selection_count": sum(
            str(item["decision"]["selected_source"]).startswith("beatnet")
            for item in fixtures
        ),
        "partial_adoption_count": sum(
            bool(item["decision"]["partial_adoption"]) for item in fixtures
        ),
        "bpm_regression_count": sum(
            item["enhanced"]["bpm_absolute_error"]
            > item["baseline"]["bpm_absolute_error"] + 0.5
            for item in fixtures
        ),
        "beat_f1_regression_count": sum(
            item["enhanced"]["beat_f1"] + 0.01 < item["baseline"]["beat_f1"]
            for item in fixtures
        ),
        "downbeat_f1_regression_count": sum(
            item["enhanced"]["downbeat_f1"] + 0.01
            < item["baseline"]["downbeat_f1"]
            for item in fixtures
        ),
        "pickup_regression_count": sum(
            item["enhanced"]["pickup_recall"] is not None
            and item["baseline"]["pickup_recall"] is not None
            and item["enhanced"]["pickup_recall"] + 0.01
            < item["baseline"]["pickup_recall"]
            for item in fixtures
        ),
        "half_time_error_count": sum(
            bool(item["enhanced"]["half_time_error"]) for item in fixtures
        ),
        "double_time_error_count": sum(
            bool(item["enhanced"]["double_time_error"]) for item in fixtures
        ),
        "stable_four_four_preserved": any(
            item["category"] == "stable-4/4"
            and item["enhanced"]["meter_correct"]
            and item["enhanced"]["bpm_absolute_error"] <= 0.5
            and item["enhanced"]["beat_f1"] + 0.01
            >= item["baseline"]["beat_f1"]
            and item["enhanced"]["downbeat_f1"] + 0.01
            >= item["baseline"]["downbeat_f1"]
            for item in fixtures
        ),
    }
    return {
        "schema_version": 1,
        "fixture_set_version": PHASE_FOUR_FIXTURE_SET_VERSION,
        "fixture_count": len(fixtures),
        "categories": [category for _, category in FOCUSED_FIXTURES],
        "summary": summary,
        "fixtures": fixtures,
    }


def _benchmark_default_path(fixture_dir: Path) -> dict[str, Any]:
    audio_path = fixture_dir / DEFAULT_FIXTURE_FILENAME
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing default-path fixture: {audio_path}")
    analysis = analyze_audio(audio_path, use_beatnet=False)
    decision = analysis.tempo_analysis
    candidate_sources = [candidate.source for candidate in analysis.tempo_candidates]
    return {
        "audio": DEFAULT_FIXTURE_FILENAME,
        "passed": (
            analysis.beatnet_analysis_status == "unavailable"
            and not any(source.startswith("beatnet") for source in candidate_sources)
            and decision is not None
            and decision.decision_version == "tempo-arbitration-v3"
            and analysis.bpm > 0
            and bool(analysis.beat_times)
        ),
        "analyzer": analysis.analyzer,
        "bpm": analysis.bpm,
        "beatnet_analysis_status": analysis.beatnet_analysis_status,
        "candidate_sources": candidate_sources,
        "selected_source": decision.selected_source if decision is not None else None,
        "decision_version": decision.decision_version if decision is not None else None,
    }


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    pickup = metrics.get("pickup_onset")
    return {
        "analyzer": metrics["analyzer"],
        "estimated_bpm": metrics["estimated_bpm"],
        "bpm_absolute_error": metrics["bpm_absolute_error"],
        "estimated_time_signature": metrics["estimated_time_signature"],
        "meter_correct": metrics["meter_correct"],
        "first_downbeat_absolute_error_seconds": metrics[
            "first_downbeat_absolute_error_seconds"
        ],
        "beat_f1": metrics["beat"]["f1"],
        "downbeat_f1": metrics["downbeat"]["f1"],
        "pickup_recall": pickup["recall"] if pickup is not None else None,
        "half_time_error": metrics["half_time_error"],
        "double_time_error": metrics["double_time_error"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


@contextmanager
def _blocked_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during Phase 4 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 4 acceptance")

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
