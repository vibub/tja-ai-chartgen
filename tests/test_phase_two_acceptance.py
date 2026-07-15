from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_two_acceptance import (
    build_phase_two_acceptance_report,
    build_phase_two_constraint_matrix,
    build_shared_salience_metric_evidence,
    render_phase_two_acceptance_markdown,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_two_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_two_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_two.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_two", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_two = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_two)


def _benchmark(
    *,
    precision: float = 0.64,
    unsupported: float = 0.04,
    strong_recall: float = 0.97,
    downbeat_recall: float = 0.98,
    silent_violations: int = 4,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_version": "chart-alignment-v1",
        "fixture_count": 1,
        "chart_count": 4,
        "settings": {"use_beatnet": False},
        "summary": {
            "chart_count": 4,
            "note_count": 100,
            "note_onset": {"precision": precision},
            "strong_onset_response": {"recall": strong_recall},
            "downbeat_response": {"recall": downbeat_recall},
            "unsupported_note_ratio": unsupported,
            "silent_violation_count": silent_violations,
            "deterministic_rate": 1.0,
        },
        "course_summaries": {
            "Easy": {"note_count": 10},
            "Normal": {"note_count": 20},
            "Hard": {"note_count": 30},
            "Oni": {"note_count": 40},
        },
        "charts": [],
    }


def _baseline() -> dict[str, object]:
    return _benchmark(
        precision=0.62,
        unsupported=0.06,
        strong_recall=0.92,
        downbeat_recall=0.93,
        silent_violations=5,
    )


def _matrix() -> dict[str, object]:
    return {
        "passed": True,
        "load_configuration_count": 180,
        "silence_configuration_count": 180,
        "configuration_count": 360,
        "bar_result_count": 1080,
        "violation_count": 0,
        "violation_counts": {},
    }


def _shared_metric() -> dict[str, object]:
    return {
        "metric_version": "ai-salience-validation-v1",
        "report_only": True,
        "rule_chart_evaluated_with_ai_metric": True,
        "missing_fields": [],
    }


def _report() -> dict[str, object]:
    benchmark = _benchmark()
    return build_phase_two_acceptance_report(
        [benchmark, benchmark],
        committed_chart_baseline=_baseline(),
        constraint_matrix=_matrix(),
        shared_metric_evidence=_shared_metric(),
        network_blocked=True,
    )


def test_build_phase_two_constraint_matrix_covers_all_profiles():
    matrix = build_phase_two_constraint_matrix()

    assert matrix["passed"] is True
    assert matrix["configuration_count"] == 360
    assert matrix["bar_result_count"] == 1080
    assert matrix["violation_count"] == 0
    assert matrix["violation_counts"] == {}


def test_build_shared_salience_metric_evidence_uses_ai_report_contract():
    evidence = build_shared_salience_metric_evidence()

    assert evidence["metric_version"] == "ai-salience-validation-v1"
    assert evidence["report_only"] is True
    assert evidence["rule_chart_evaluated_with_ai_metric"] is True
    assert evidence["missing_fields"] == []
    assert evidence["sample_report"]["unrepresentable_note_count"] == 0


def test_build_phase_two_acceptance_report_passes_all_exit_conditions():
    report = _report()

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-two-acceptance-v1"
    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"].values())
    assert report["checks"]["note_onset_alignment_improved"]["delta"] == 0.02
    assert report["checks"]["unsupported_note_ratio_reduced"]["delta"] == -0.02


def test_phase_two_acceptance_fails_on_alignment_or_constraint_regression():
    current = _benchmark(precision=0.61, unsupported=0.07)
    matrix = _matrix() | {"passed": False, "violation_count": 1}

    report = build_phase_two_acceptance_report(
        [current, current],
        committed_chart_baseline=_baseline(),
        constraint_matrix=matrix,
        shared_metric_evidence=_shared_metric(),
        network_blocked=True,
    )

    assert report["passed"] is False
    assert report["checks"]["note_onset_alignment_improved"]["passed"] is False
    assert report["checks"]["unsupported_note_ratio_reduced"]["passed"] is False
    assert report["checks"]["resolution_and_load_constraints"]["passed"] is False


def test_render_phase_two_acceptance_markdown_lists_exit_conditions():
    markdown = render_phase_two_acceptance_markdown(_report())

    assert "# Phase 2 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "Synthetic note/onset alignment improves" in markdown
    assert "Unsupported-note ratio decreases" in markdown
    assert "Easy/Normal/Hard/Oni load remains monotonic" in markdown
    assert "Rule and AI results use the same salience-alignment metric" in markdown


def test_run_phase_two_acceptance_runs_twice_with_network_blocked(
    tmp_path, monkeypatch
):
    baseline = _baseline()
    (tmp_path / "chart_alignment_baseline.json").write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    benchmark = _benchmark()
    original_connect = socket.socket.connect
    network_states: list[bool] = []

    def fake_benchmark(_fixture_dir: Path):
        network_states.append(socket.socket.connect is not original_connect)
        return benchmark

    monkeypatch.setattr(verify_phase_two, "benchmark_charts", fake_benchmark)
    monkeypatch.setattr(verify_phase_two, "build_phase_two_constraint_matrix", _matrix)
    monkeypatch.setattr(
        verify_phase_two,
        "build_shared_salience_metric_evidence",
        _shared_metric,
    )

    report = verify_phase_two.run_phase_two_acceptance(tmp_path)

    assert report["passed"] is True
    assert network_states == [True, True]
    assert socket.socket.connect is original_connect


def test_phase_two_main_writes_json_and_markdown(tmp_path, monkeypatch):
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(
        verify_phase_two,
        "run_phase_two_acceptance",
        lambda _fixture_dir: _report(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_two.py",
            "--fixture-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_two.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_two_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-two-acceptance-v1"
    assert acceptance["phase"] == "Phase 2"
    assert acceptance["passed"] is True
    assert acceptance["fixture_count"] == 17
    assert acceptance["chart_count"] == 68
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["checks"]["note_onset_alignment_improved"]["delta"] > 0
    assert acceptance["checks"]["unsupported_note_ratio_reduced"]["delta"] < 0
    assert acceptance["constraint_matrix"]["configuration_count"] == 360
    assert acceptance["constraint_matrix"]["bar_result_count"] == 1080

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "ai-salience-validation-v1" in markdown
