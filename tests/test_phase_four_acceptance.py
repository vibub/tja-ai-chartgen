from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_four_acceptance import (
    FOCUSED_FIXTURES,
    build_phase_four_acceptance_report,
    build_phase_four_behavior_matrix,
    render_phase_four_acceptance_markdown,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_four_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_four_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_four.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_four", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_four = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_four)


def _fixture_evidence() -> dict[str, object]:
    fixtures = []
    for audio, category in FOCUSED_FIXTURES:
        fixtures.append(
            {
                "audio": audio,
                "category": category,
                "expected_bpm": 120.0,
                "expected_time_signature": (
                    "3/4"
                    if category == "meter-3/4"
                    else "6/8"
                    if category == "meter-6/8"
                    else "4/4"
                ),
                "baseline": {
                    "analyzer": "librosa+onset-grid",
                    "estimated_bpm": 120.0,
                    "bpm_absolute_error": 0.0,
                    "estimated_time_signature": "4/4",
                    "meter_correct": category not in {"meter-3/4", "meter-6/8"},
                    "first_downbeat_absolute_error_seconds": 0.0,
                    "beat_f1": 0.96,
                    "downbeat_f1": 0.8,
                    "pickup_recall": 1.0 if category == "pickup" else None,
                    "half_time_error": False,
                    "double_time_error": False,
                },
                "enhanced": {
                    "analyzer": "librosa+onset-grid",
                    "estimated_bpm": 120.0,
                    "bpm_absolute_error": 0.0,
                    "estimated_time_signature": "4/4",
                    "meter_correct": category not in {"meter-3/4", "meter-6/8"},
                    "first_downbeat_absolute_error_seconds": 0.0,
                    "beat_f1": 0.96,
                    "downbeat_f1": 0.8,
                    "pickup_recall": 1.0 if category == "pickup" else None,
                    "half_time_error": False,
                    "double_time_error": False,
                },
                "beatnet_analysis_status": "complete",
                "decision": {
                    "decision_version": "tempo-arbitration-v3",
                    "selected_source": "librosa+onset-grid",
                    "tempo_source": "librosa+onset-grid",
                    "meter_source": "librosa+onset-grid",
                    "partial_adoption": False,
                    "ambiguous": False,
                    "accepted": False,
                    "reason": "incomplete-beat-numbers",
                    "candidate_rejections": {
                        "beatnet": "incomplete-beat-numbers"
                    },
                    "candidate_sources": [
                        "librosa",
                        "librosa+onset-grid",
                        "beatnet",
                        "beatnet+onset-grid",
                    ],
                    "selected_candidate_count": 1,
                    "tempo_variation_version": "tempo-variation-v1",
                    "tempo_variation_source": "beatnet",
                    "tempo_variation_classification": "stable",
                    "fixed_bpm_constrained": False,
                },
            }
        )
    return {
        "schema_version": 1,
        "fixture_set_version": "phase-four-fixtures-v1",
        "fixture_count": len(fixtures),
        "categories": [category for _, category in FOCUSED_FIXTURES],
        "summary": {
            "beatnet_complete_count": len(fixtures),
            "rejected_fixture_count": len(fixtures),
            "full_selection_count": 0,
            "partial_adoption_count": 0,
            "bpm_regression_count": 0,
            "beat_f1_regression_count": 0,
            "downbeat_f1_regression_count": 0,
            "pickup_regression_count": 0,
            "half_time_error_count": 0,
            "double_time_error_count": 0,
            "stable_four_four_preserved": True,
        },
        "fixtures": fixtures,
    }


def _default_evidence() -> dict[str, object]:
    return {
        "audio": "click_4_4.wav",
        "passed": True,
        "analyzer": "librosa+onset-grid",
        "bpm": 120.0,
        "beatnet_analysis_status": "unavailable",
        "candidate_sources": ["librosa", "librosa+onset-grid"],
        "selected_source": "librosa+onset-grid",
        "decision_version": "tempo-arbitration-v3",
    }


def _report() -> dict[str, object]:
    fixture = _fixture_evidence()
    behavior = build_phase_four_behavior_matrix()
    default = _default_evidence()
    return build_phase_four_acceptance_report(
        [fixture, fixture],
        [behavior, behavior],
        [default, default],
        network_blocked=True,
    )


def test_phase_four_behavior_matrix_covers_all_arbitration_modes():
    matrix = build_phase_four_behavior_matrix()

    assert matrix["passed"] is True
    assert matrix["scenarios"]["rejected_invalid_beatnet"]["passed"] is True
    assert matrix["scenarios"]["selected_stronger_beatnet"]["passed"] is True
    assert matrix["scenarios"]["partial_meter_downbeat_adoption"]["passed"] is True
    assert matrix["scenarios"]["ambiguous_alias_fallback"]["passed"] is True
    assert matrix["scenarios"]["manual_override_precedence"]["passed"] is True
    assert matrix["scenarios"]["tempo_variation_classification"]["passed"] is True


def test_build_phase_four_acceptance_report_passes_all_exit_conditions():
    report = _report()

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-four-acceptance-v1"
    assert report["phase"] == "Phase 4"
    assert report["passed"] is True
    assert report["fixture_count"] == 6
    assert all(check["passed"] for check in report["checks"].values())


def test_phase_four_acceptance_fails_on_regression_and_missing_diagnostics():
    fixture = _fixture_evidence()
    fixture["summary"]["beat_f1_regression_count"] = 1
    fixture["fixtures"][0]["decision"]["tempo_variation_version"] = None
    behavior = build_phase_four_behavior_matrix()
    default = _default_evidence()

    report = build_phase_four_acceptance_report(
        [fixture, fixture],
        [behavior, behavior],
        [default, default],
        network_blocked=True,
    )

    assert report["passed"] is False
    assert report["checks"]["stable_four_four_non_regression"]["passed"] is False
    assert report["checks"]["analysis_explainability"]["passed"] is False


def test_render_phase_four_acceptance_markdown_lists_exit_conditions():
    markdown = render_phase_four_acceptance_markdown(_report())

    assert "# Phase 4 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "Adaptive arbitration rejects weak candidates" in markdown
    assert "Default analysis remains valid without loading BeatNet" in markdown
    assert "tempo-variation classifications remain report-only" in markdown
    assert "deterministic and offline" in markdown


def test_run_phase_four_acceptance_runs_twice_with_network_blocked(
    tmp_path, monkeypatch
):
    original_connect = socket.socket.connect
    network_states: list[bool] = []
    fixture = _fixture_evidence()
    default = _default_evidence()

    def fake_fixture(_fixture_dir: Path):
        network_states.append(socket.socket.connect is not original_connect)
        return fixture

    monkeypatch.setattr(
        verify_phase_four,
        "_benchmark_focused_fixtures",
        fake_fixture,
    )
    monkeypatch.setattr(
        verify_phase_four,
        "_benchmark_default_path",
        lambda _fixture_dir: default,
    )

    report = verify_phase_four.run_phase_four_acceptance(tmp_path)

    assert report["passed"] is True
    assert network_states == [True, True]
    assert socket.socket.connect is original_connect


def test_phase_four_main_writes_json_and_markdown(tmp_path, monkeypatch):
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(
        verify_phase_four,
        "run_phase_four_acceptance",
        lambda _fixture_dir: _report(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_four.py",
            "--fixture-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_four.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_four_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-four-acceptance-v1"
    assert acceptance["phase"] == "Phase 4"
    assert acceptance["passed"] is True
    assert acceptance["fixture_count"] == 6
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["behavior_matrix"]["passed"] is True
    assert acceptance["checks"]["default_path_without_beatnet"]["passed"] is True
    assert acceptance["checks"]["analysis_explainability"]["passed"] is True

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "6 focused fixtures" not in markdown
    assert "Focused fixture count: 6" in markdown
