from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_three_acceptance import (
    build_phase_three_acceptance_report,
    build_phase_three_behavior_matrix,
    render_phase_three_acceptance_markdown,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_three_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_three_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_three.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_three", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_three = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_three)


def _chart_benchmark() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_version": "chart-alignment-v1",
        "fixture_count": 17,
        "chart_count": 68,
        "summary": {
            "deterministic_rate": 1.0,
            "strong_onset_response": {"recall": 0.98},
            "downbeat_response": {"recall": 0.99},
        },
    }


def _phase_two_acceptance() -> dict[str, object]:
    return {
        "acceptance_version": "phase-two-acceptance-v1",
        "passed": True,
        "current_summary": {
            "strong_onset_response": {"recall": 0.97},
            "downbeat_response": {"recall": 0.98},
        },
    }


def _fill_evidence() -> dict[str, object]:
    return {
        "fixture": "fill_burst_120.wav",
        "expected_burst_indexes": [3, 7],
        "reliable_burst_indexes": [3, 7],
        "special_note_indexes": [3, 7],
        "late_range_count": 2,
        "special_note_durations_seconds": [0.75, 0.75],
        "passed": True,
    }


def _report() -> dict[str, object]:
    benchmark = _chart_benchmark()
    behavior = build_phase_three_behavior_matrix()
    fill = _fill_evidence()
    return build_phase_three_acceptance_report(
        [benchmark, benchmark],
        [behavior, behavior],
        [fill, fill],
        phase_two_acceptance=_phase_two_acceptance(),
        network_blocked=True,
    )


def test_phase_three_behavior_matrix_covers_accent_color_and_special_notes():
    matrix = build_phase_three_behavior_matrix()

    assert matrix["passed"] is True
    assert matrix["accent"]["accent_response_rate"] == 1.0
    assert matrix["accent"]["adjacent_big_note_violation_count"] == 0
    assert matrix["accent"]["big_note_isolation_violation_count"] == 0
    assert matrix["color"]["low_attack_don_response_rate"] >= 0.75
    assert matrix["color"]["high_attack_ka_response_rate"] >= 0.75
    assert matrix["special_notes"]["configuration_count"] == 72
    assert matrix["special_notes"]["drumroll_count"] == 36
    assert matrix["special_notes"]["balloon_count"] == 36
    assert matrix["special_notes"]["no_burst_special_note_count"] == 0
    assert matrix["special_notes"]["violation_count"] == 0


def test_build_phase_three_acceptance_report_passes_all_exit_conditions():
    report = _report()

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-three-acceptance-v1"
    assert report["phase"] == "Phase 3"
    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"].values())


def test_phase_three_acceptance_fails_on_response_drift_and_fill_false_positive():
    benchmark = _chart_benchmark()
    benchmark["summary"]["strong_onset_response"]["recall"] = 0.5
    fill = _fill_evidence()
    fill["special_note_indexes"] = [1, 3, 7]
    fill["passed"] = False
    behavior = build_phase_three_behavior_matrix()

    report = build_phase_three_acceptance_report(
        [benchmark, benchmark],
        [behavior, behavior],
        [fill, fill],
        phase_two_acceptance=_phase_two_acceptance(),
        network_blocked=True,
    )

    assert report["passed"] is False
    assert report["checks"]["strong_downbeat_and_cadence_response"]["passed"] is False
    assert report["checks"]["fill_burst_fixture_alignment"]["passed"] is False


def test_render_phase_three_acceptance_markdown_lists_exit_conditions():
    markdown = render_phase_three_acceptance_markdown(_report())

    assert "# Phase 3 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "Strong-onset, downbeat, and cadence response" in markdown
    assert "real-time isolation violations" in markdown
    assert "Don/ka response remains driven by frequency evidence" in markdown
    assert "Phrase ends without burst evidence" in markdown
    assert "deterministic and offline" in markdown


def test_run_phase_three_acceptance_runs_twice_with_network_blocked(
    tmp_path, monkeypatch
):
    original_connect = socket.socket.connect
    network_states: list[bool] = []
    benchmark = _chart_benchmark()
    fill = _fill_evidence()
    (tmp_path / "phase_two_acceptance.json").write_text(
        json.dumps(_phase_two_acceptance()),
        encoding="utf-8",
    )

    def fake_benchmark(_fixture_dir: Path):
        network_states.append(socket.socket.connect is not original_connect)
        return benchmark

    monkeypatch.setattr(verify_phase_three, "benchmark_charts", fake_benchmark)
    monkeypatch.setattr(
        verify_phase_three,
        "_benchmark_fill_fixture",
        lambda _fixture_dir: fill,
    )

    report = verify_phase_three.run_phase_three_acceptance(tmp_path)

    assert report["passed"] is True
    assert network_states == [True, True]
    assert socket.socket.connect is original_connect


def test_phase_three_main_writes_json_and_markdown(tmp_path, monkeypatch):
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(
        verify_phase_three,
        "run_phase_three_acceptance",
        lambda _fixture_dir: _report(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_three.py",
            "--fixture-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_three.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_three_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-three-acceptance-v1"
    assert acceptance["phase"] == "Phase 3"
    assert acceptance["passed"] is True
    assert acceptance["fixture_count"] == 17
    assert acceptance["chart_count"] == 68
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["behavior_matrix"]["accent"]["big_note_count"] == 11
    assert (
        acceptance["behavior_matrix"]["accent"][
            "big_note_isolation_violation_count"
        ]
        == 0
    )
    assert acceptance["behavior_matrix"]["special_notes"]["configuration_count"] == 72
    assert acceptance["behavior_matrix"]["special_notes"]["violation_count"] == 0
    assert acceptance["fill_fixture_evidence"]["reliable_burst_indexes"] == [3, 7]
    assert acceptance["fill_fixture_evidence"]["special_note_indexes"] == [3, 7]

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "72 configurations" in markdown
