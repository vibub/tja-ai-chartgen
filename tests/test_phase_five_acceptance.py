from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_five_acceptance import (
    build_phase_five_acceptance_report,
    build_phase_five_behavior_matrix,
    build_phase_five_payload_evidence,
    render_phase_five_acceptance_markdown,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_five_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_five_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_five.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_five", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_five = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_five)


def _model_evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "passed": True,
        "benchmark_version": "instrument-model-benchmark-v1",
        "network_attempt_blocked": True,
        "offline_environment_applied": True,
        "profile_mismatch_rejected": True,
        "performance_threshold_enforced": False,
        "full": {
            "profile": "full",
            "feature_version": "instrument-v1",
            "hashes_verified": True,
            "file_count": 5,
            "ast_bytes": 7,
            "classifier_model": "ast",
        },
        "stem_role": {
            "profile": "stem-role",
            "feature_version": "stem-role-v1",
            "hashes_verified": True,
            "file_count": 2,
            "total_bytes": 22,
            "demucs_bytes": 22,
            "ast_bytes": 0,
            "classifier_model": None,
            "status": "complete",
            "stem_frame_count": 1,
            "classification_window_count": 0,
            "requested_device": "auto",
            "resolved_device": "cpu",
            "elapsed_seconds": 1.5,
            "realtime_factor": 0.75,
            "peak_rss_bytes": 1_000_000,
            "peak_rss_delta_bytes": 0,
        },
    }


def _web_evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "local_allowed": True,
        "remote_default_allowed": False,
        "remote_explicit_allowed": True,
    }


def _report() -> dict[str, object]:
    behavior = build_phase_five_behavior_matrix()
    model = _model_evidence()
    payload = build_phase_five_payload_evidence()
    web = _web_evidence()
    return build_phase_five_acceptance_report(
        [behavior, behavior],
        [model, model],
        [payload, payload],
        [web, web],
        network_blocked=True,
    )


def test_phase_five_behavior_matrix_covers_bounded_stem_roles():
    matrix = build_phase_five_behavior_matrix()

    assert matrix["matrix_version"] == "phase-five-stem-role-matrix-v1"
    assert matrix["passed"] is True
    assert matrix["scenario_count"] == 7
    assert all(scenario["passed"] for scenario in matrix["scenarios"].values())
    assert matrix["scenarios"]["stem_artifacts_do_not_create_hits"]["point_count"] == 0
    assert matrix["scenarios"]["bass_reinforces_beat_and_don_only"][
        "offbeat_created"
    ] is False


def test_phase_five_payload_removes_concrete_instrument_taxonomy():
    evidence = build_phase_five_payload_evidence()

    assert evidence["passed"] is True
    assert evidence["payload_schema"] == "tja-ai-chartgen-compact-v7"
    assert "dominant_instrument" not in evidence["instrument_bar_columns"]
    assert "active_instruments" not in evidence["instrument_bar_columns"]
    assert evidence["taxonomy_row"] == evidence["role_only_row"]


def test_build_phase_five_acceptance_report_passes_all_exit_conditions():
    report = _report()

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-five-acceptance-v1"
    assert report["phase"] == "Phase 5"
    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"].values())


def test_phase_five_acceptance_fails_without_offline_or_remote_gate():
    behavior = build_phase_five_behavior_matrix()
    model = _model_evidence()
    model["network_attempt_blocked"] = False
    payload = build_phase_five_payload_evidence()
    web = _web_evidence()
    web["remote_default_allowed"] = True

    report = build_phase_five_acceptance_report(
        [behavior, behavior],
        [model, model],
        [payload, payload],
        [web, web],
        network_blocked=True,
    )

    assert report["passed"] is False
    assert report["checks"]["offline_loading_and_cost_report"]["passed"] is False
    assert report["checks"]["remote_web_admin_gate"]["passed"] is False


def test_render_phase_five_acceptance_markdown_lists_exit_conditions():
    markdown = render_phase_five_acceptance_markdown(_report())

    assert "# Phase 5 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "htdemucs-only stem-role models without AST" in markdown
    assert "AI compact payload no longer depends" in markdown
    assert "Remote Web instrument analysis" in markdown
    assert "report-only" in markdown


def test_verify_model_profiles_exercises_preparation_hashes_and_offline_benchmark():
    evidence = verify_phase_five._verify_model_profiles()

    assert evidence["passed"] is True
    assert evidence["full"]["ast_bytes"] > 0
    assert evidence["stem_role"]["ast_bytes"] == 0
    assert evidence["stem_role"]["classification_window_count"] == 0
    assert evidence["network_attempt_blocked"] is True
    assert evidence["offline_environment_applied"] is True


def test_run_phase_five_acceptance_runs_twice_with_network_blocked(monkeypatch):
    original_connect = socket.socket.connect
    network_states: list[bool] = []
    behavior = build_phase_five_behavior_matrix()
    model = _model_evidence()
    payload = build_phase_five_payload_evidence()
    web = _web_evidence()

    def fake_behavior():
        network_states.append(socket.socket.connect is not original_connect)
        return behavior

    monkeypatch.setattr(verify_phase_five, "build_phase_five_behavior_matrix", fake_behavior)
    monkeypatch.setattr(verify_phase_five, "_verify_model_profiles", lambda: model)
    monkeypatch.setattr(verify_phase_five, "build_phase_five_payload_evidence", lambda: payload)
    monkeypatch.setattr(verify_phase_five, "_verify_web_permissions", lambda: web)

    report = verify_phase_five.run_phase_five_acceptance()

    assert report["passed"] is True
    assert network_states == [True, True]
    assert socket.socket.connect is original_connect


def test_phase_five_main_writes_json_and_markdown(tmp_path, monkeypatch):
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(verify_phase_five, "run_phase_five_acceptance", _report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_five.py",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_five.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_five_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-five-acceptance-v1"
    assert acceptance["phase"] == "Phase 5"
    assert acceptance["passed"] is True
    assert acceptance["scenario_count"] == 7
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["payload_evidence"]["payload_schema"] == (
        "tja-ai-chartgen-compact-v7"
    )
    assert acceptance["model_profile_evidence"]["stem_role"]["ast_bytes"] == 0
    assert acceptance["web_permission_evidence"]["remote_default_allowed"] is False

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "7 deterministic scenarios" not in markdown
    assert "Deterministic scenarios: 7" in markdown
