from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_seven_acceptance import (
    build_phase_seven_acceptance_report,
    build_phase_seven_behavior_matrix,
    build_phase_seven_compatibility_evidence,
    build_phase_seven_interface_evidence,
    build_phase_seven_no_model_pipeline_evidence,
    render_phase_seven_acceptance_markdown,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
COMPATIBILITY_DIR = Path(__file__).parent / "fixtures" / "compatibility"
AUDIO_FIXTURE = FIXTURE_DIR / "click_4_4.wav"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_seven_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_seven_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_seven.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_seven", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_seven = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_seven)


def _pipeline_evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "passed": True,
        "audio_fixture": AUDIO_FIXTURE.name,
        "bar_count": 4,
        "chart_bar_count": 4,
        "quality_bar_count": 4,
        "instrument_feature_version": None,
        "instrument_analysis_status": "unavailable",
        "used_rule_generator": True,
        "ai_failure": None,
        "new_heavy_modules": [],
        "ogg_created": True,
    }


def _report() -> dict[str, object]:
    behavior = build_phase_seven_behavior_matrix()
    pipeline = _pipeline_evidence()
    compatibility = build_phase_seven_compatibility_evidence(COMPATIBILITY_DIR)
    interface = build_phase_seven_interface_evidence()
    return build_phase_seven_acceptance_report(
        [behavior, behavior],
        [pipeline, pipeline],
        [compatibility, compatibility],
        [interface, interface],
        network_blocked=True,
    )


def test_phase_seven_behavior_matrix_covers_consumer_convergence():
    matrix = build_phase_seven_behavior_matrix()

    assert matrix["matrix_version"] == "phase-seven-consumer-convergence-matrix-v1"
    assert matrix["passed"] is True
    assert matrix["scenario_count"] == 3
    assert all(scenario["passed"] for scenario in matrix["scenarios"].values())
    taxonomy = matrix["scenarios"][
        "concrete_taxonomy_does_not_change_core_consumers"
    ]
    assert taxonomy["structure_equal"] is True
    assert taxonomy["fallback_equal"] is True
    assert taxonomy["payload_equal"] is True
    assert taxonomy["quality_core_equal"] is True


def test_phase_seven_no_model_pipeline_runs_real_audio_without_heavy_models(tmp_path):
    evidence = build_phase_seven_no_model_pipeline_evidence(
        AUDIO_FIXTURE,
        tmp_path,
    )

    assert evidence["passed"] is True
    assert evidence["bar_count"] == 4
    assert evidence["chart_bar_count"] == 4
    assert evidence["quality_bar_count"] == 4
    assert evidence["instrument_analysis_status"] == "unavailable"
    assert evidence["new_heavy_modules"] == []
    assert evidence["ogg_created"] is True


def test_phase_seven_compatibility_evidence_reads_legacy_and_compacts_payload():
    evidence = build_phase_seven_compatibility_evidence(COMPATIBILITY_DIR)

    assert evidence["passed"] is True
    legacy = evidence["legacy_read_compatibility"]
    compact = evidence["compact_payload"]
    assert legacy["analysis_schema_version"] == 5
    assert legacy["config_default_profile"] == "full"
    assert legacy["input_schema"] == "tja-ai-chartgen-compact-v4"
    assert compact["schema"] == "tja-ai-chartgen-compact-v7"
    assert compact["current_column_count"] < compact["legacy_column_count"]
    assert compact["current_row_bytes"] < compact["legacy_row_bytes"]
    assert compact["has_compact_salience"] is True


def test_phase_seven_interface_evidence_matches_recommended_capabilities():
    evidence = build_phase_seven_interface_evidence()

    assert evidence["passed"] is True
    assert evidence["cli"] == {
        "prepare_default_profile": "stem-role",
        "generate_default_profile": "stem-role",
        "generate_analysis_enabled_by_default": False,
    }
    assert evidence["web"]["recommended_option_selected"] is True
    assert evidence["web"]["legacy_option_labeled"] is True
    assert evidence["web"]["remote_default_allowed"] is False
    assert evidence["web"]["remote_explicit_allowed"] is True
    assert "核心生成仍继续使用" in evidence["notices"]["classifier_fallback"][
        "message"
    ]


def test_build_phase_seven_acceptance_report_passes_all_exit_conditions():
    report = _report()

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-seven-acceptance-v1"
    assert report["phase"] == "Phase 7"
    assert report["passed"] is True
    assert report["scenario_count"] == 3
    assert all(check["passed"] for check in report["checks"].values())


def test_phase_seven_acceptance_fails_for_compatibility_or_interface_regression():
    behavior = build_phase_seven_behavior_matrix()
    pipeline = _pipeline_evidence()
    compatibility = build_phase_seven_compatibility_evidence(COMPATIBILITY_DIR)
    interface = build_phase_seven_interface_evidence()
    compatibility["legacy_read_compatibility"]["passed"] = False
    interface["passed"] = False

    report = build_phase_seven_acceptance_report(
        [behavior, behavior],
        [pipeline, pipeline],
        [compatibility, compatibility],
        [interface, interface],
        network_blocked=True,
    )

    assert report["passed"] is False
    assert report["checks"]["legacy_persistence_compatibility"]["passed"] is False
    assert report["checks"]["cli_web_notice_capability_alignment"]["passed"] is False


def test_render_phase_seven_acceptance_markdown_lists_exit_conditions():
    markdown = render_phase_seven_acceptance_markdown(_report())

    assert "# Phase 7 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "complete rhythm-first path runs without heavyweight models" in markdown
    assert "Missing concrete taxonomy does not reduce core generation" in markdown
    assert "Legacy analysis, config, and AI sidecars remain readable" in markdown
    assert "CLI/Web defaults, progress, summaries, notices" in markdown


def test_run_phase_seven_acceptance_runs_twice_with_network_blocked(monkeypatch):
    original_connect = socket.socket.connect
    network_states: list[bool] = []
    original_behavior = verify_phase_seven.build_phase_seven_behavior_matrix

    def tracked_behavior():
        network_states.append(socket.socket.connect is not original_connect)
        return original_behavior()

    monkeypatch.setattr(
        verify_phase_seven,
        "build_phase_seven_behavior_matrix",
        tracked_behavior,
    )

    report = verify_phase_seven.run_phase_seven_acceptance(COMPATIBILITY_DIR)

    assert report["passed"] is True
    assert network_states == [True, True]
    assert socket.socket.connect is original_connect


def test_phase_seven_main_writes_json_and_markdown(tmp_path, monkeypatch):
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(
        verify_phase_seven,
        "run_phase_seven_acceptance",
        lambda _compatibility_dir, _audio_fixture: _report(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_seven.py",
            "--compatibility-dir",
            str(COMPATIBILITY_DIR),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_seven.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_seven_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-seven-acceptance-v1"
    assert acceptance["phase"] == "Phase 7"
    assert acceptance["passed"] is True
    assert acceptance["scenario_count"] == 3
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["checks"]["compact_ai_payload"]["legacy_column_count"] == 9
    assert acceptance["checks"]["compact_ai_payload"]["current_column_count"] == 7

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "`quality-report-rhythm-alignment-v1`" in markdown
    assert "stem-role recommended" in markdown
