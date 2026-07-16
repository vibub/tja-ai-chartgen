from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_six_acceptance import (
    build_phase_six_acceptance_report,
    build_phase_six_behavior_matrix,
    build_phase_six_benchmark_evidence,
    render_phase_six_acceptance_markdown,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
BASELINE_PATH = FIXTURE_DIR / "chart_alignment_baseline.json"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_six_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_six_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_six.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_six", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_six = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_six)


def _baseline() -> dict[str, object]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _evidence() -> dict[str, object]:
    return build_phase_six_benchmark_evidence(_baseline())


def _report() -> dict[str, object]:
    behavior = build_phase_six_behavior_matrix()
    evidence = _evidence()
    return build_phase_six_acceptance_report(
        [behavior, behavior],
        [evidence, evidence],
        network_blocked=True,
    )


def test_phase_six_behavior_matrix_covers_contract_resolution_and_repair_gates():
    matrix = build_phase_six_behavior_matrix()

    assert matrix["matrix_version"] == "phase-six-quality-gate-matrix-v1"
    assert matrix["passed"] is True
    assert matrix["scenario_count"] == 6
    assert all(scenario["passed"] for scenario in matrix["scenarios"].values())
    assert matrix["scenarios"]["resolution_equivalence_4_4"]["resolutions"] == [
        16,
        24,
        48,
    ]
    assert matrix["scenarios"]["resolution_equivalence_3_4"]["resolutions"] == [
        12,
        18,
        36,
    ]


def test_phase_six_benchmark_evidence_covers_all_fixtures_and_metrics():
    evidence = _evidence()

    assert evidence["benchmark_version"] == "chart-alignment-v1"
    assert evidence["fixture_count"] == 17
    assert evidence["chart_count"] == 68
    assert evidence["report_only_chart_count"] == 68
    assert evidence["missing_metric_count"] == 0
    assert evidence["selected_gate_violation_count"] == 0
    assert evidence["unified_score_keys"] == []
    assert set(evidence["course_names"]) == {"Easy", "Normal", "Hard", "Oni"}
    assert set(evidence["density_hint_names"]) == {
        "silent",
        "rest",
        "sparse",
        "normal",
        "dense",
        "fill",
    }


def test_build_phase_six_acceptance_report_passes_all_exit_conditions():
    report = _report()

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-six-acceptance-v1"
    assert report["phase"] == "Phase 6"
    assert report["passed"] is True
    assert report["fixture_count"] == 17
    assert report["chart_count"] == 68
    assert all(check["passed"] for check in report["checks"].values())


def test_phase_six_acceptance_fails_for_missing_metric_or_gate_regression():
    behavior = build_phase_six_behavior_matrix()
    evidence = _evidence()
    evidence["missing_metric_count"] = 1
    evidence["selected_gate_violation_count"] = 1

    report = build_phase_six_acceptance_report(
        [behavior, behavior],
        [evidence, evidence],
        network_blocked=True,
    )

    assert report["passed"] is False
    assert report["checks"]["fixture_ground_truth_metric_coverage"]["passed"] is False
    assert report["checks"]["selected_ai_repair_gate_scope"]["passed"] is False


def test_render_phase_six_acceptance_markdown_lists_exit_conditions():
    markdown = render_phase_six_acceptance_markdown(_report())

    assert "# Phase 6 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "same QualityReport contract" in markdown
    assert "Equivalent 16/24/48 and 12/18/36" in markdown
    assert "deterministic A/B calibration report" in markdown
    assert "Only calibrated extreme metrics enter AI repair" in markdown
    assert "no unified quality score" in markdown


def test_run_phase_six_acceptance_runs_twice_with_network_blocked(monkeypatch):
    baseline = _baseline()
    original_connect = socket.socket.connect
    network_states: list[bool] = []

    def fake_benchmark(_fixture_dir: Path):
        network_states.append(socket.socket.connect is not original_connect)
        return baseline

    monkeypatch.setattr(verify_phase_six, "benchmark_charts", fake_benchmark)

    report = verify_phase_six.run_phase_six_acceptance(FIXTURE_DIR)

    assert report["passed"] is True
    assert network_states == [True, True]
    assert socket.socket.connect is original_connect


def test_phase_six_main_writes_json_and_markdown(tmp_path, monkeypatch):
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(
        verify_phase_six,
        "run_phase_six_acceptance",
        lambda _fixture_dir: _report(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_six.py",
            "--fixture-dir",
            str(FIXTURE_DIR),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_six.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_six_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-six-acceptance-v1"
    assert acceptance["phase"] == "Phase 6"
    assert acceptance["passed"] is True
    assert acceptance["fixture_count"] == 17
    assert acceptance["chart_count"] == 68
    assert acceptance["scenario_count"] == 6
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["checks"]["selected_ai_repair_gate_scope"][
        "baseline_violation_count"
    ] == 0

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "`ai-rhythm-repair-gate-v1`" in markdown
    assert "no unified quality score" in markdown
