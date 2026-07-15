from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_one_acceptance import (
    build_fixture_salience_metrics,
    build_phase_one_acceptance_report,
    build_salience_benchmark,
    render_phase_one_acceptance_markdown,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarRhythmicSalience,
    RhythmicSaliencePoint,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_one_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_one_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_one.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_one", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_one = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_one)


def _fixture_metrics() -> dict[str, object]:
    silent = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.0,
        grids_per_bar=48,
    )
    active = BarFeature(
        index=1,
        start_time=2.0,
        end_time=4.0,
        energy=0.6,
        grids_per_bar=48,
    )
    salience = [
        BarRhythmicSalience(fallback_reason="edge-silence"),
        BarRhythmicSalience(
            points=[
                RhythmicSaliencePoint(
                    grid=12,
                    hit=0.8,
                    confidence=0.9,
                    reasons=["onset"],
                )
            ],
            confidence=0.8,
        ),
    ]
    return build_fixture_salience_metrics(
        {"audio": "sample.wav", "onsets": [2.5]},
        [silent, active],
        salience,
    )


def _benchmark() -> dict[str, object]:
    return build_salience_benchmark([_fixture_metrics()])


def test_build_fixture_salience_metrics_checks_canonical_alignment_and_silence():
    metrics = _fixture_metrics()

    assert metrics["canonical_grid_count"] == 96
    assert metrics["salience_point_count"] == 1
    assert metrics["invalid_grid_count"] == 0
    assert metrics["duplicate_grid_count"] == 0
    assert metrics["edge_silent_bar_count"] == 1
    assert metrics["edge_silence_violation_count"] == 0
    assert metrics["confidence_reason_violation_count"] == 0
    assert metrics["onset_peak_alignment"]["precision"] == 1.0
    assert metrics["onset_peak_alignment"]["recall"] == 1.0


def test_build_phase_one_acceptance_report_passes_all_checks():
    benchmark = _benchmark()

    report = build_phase_one_acceptance_report(
        [benchmark, benchmark],
        network_blocked=True,
    )

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-one-acceptance-v1"
    assert report["passed"] is True
    assert report["fixture_count"] == 1
    assert all(check["passed"] for check in report["checks"].values())
    assert report["checks"]["compact_sparse_output"]["ai_payload_delta_fields"] == 0


def test_phase_one_acceptance_fails_on_drift_and_contract_violations():
    benchmark = _benchmark()
    drifted = json.loads(json.dumps(benchmark))
    drifted["summary"]["invalid_grid_count"] = 1

    report = build_phase_one_acceptance_report(
        [benchmark, drifted],
        network_blocked=True,
    )

    assert report["passed"] is False
    assert report["checks"]["deterministic_output"]["passed"] is False

    invalid_report = build_phase_one_acceptance_report(
        [drifted, drifted],
        network_blocked=True,
    )
    assert invalid_report["checks"]["canonical_grid_contract"]["passed"] is False


def test_render_phase_one_acceptance_markdown_lists_exit_conditions():
    benchmark = _benchmark()
    report = build_phase_one_acceptance_report(
        [benchmark, benchmark],
        network_blocked=True,
    )

    markdown = render_phase_one_acceptance_markdown(report)

    assert "# Phase 1 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "All salience points use canonical grids" in markdown
    assert "Known fixture onsets align with hit-salience peaks" in markdown
    assert "fallback generator, AI prompt, and quality report" in markdown


def test_run_phase_one_acceptance_runs_twice_with_network_blocked(
    tmp_path, monkeypatch
):
    original_connect = socket.socket.connect
    network_states: list[bool] = []
    benchmark = _benchmark()

    def fake_benchmark(_fixture_dir: Path, **_kwargs):
        network_states.append(socket.socket.connect is not original_connect)
        return benchmark

    monkeypatch.setattr(
        verify_phase_one,
        "benchmark_fixture_directory",
        fake_benchmark,
    )

    report = verify_phase_one.run_phase_one_acceptance(tmp_path)

    assert report["passed"] is True
    assert network_states == [True, True]
    assert socket.socket.connect is original_connect


def test_phase_one_main_writes_json_and_markdown(tmp_path, monkeypatch):
    benchmark = _benchmark()
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(
        verify_phase_one,
        "benchmark_fixture_directory",
        lambda _fixture_dir, **_kwargs: benchmark,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_one.py",
            "--fixture-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_one.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_one_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-one-acceptance-v1"
    assert acceptance["phase"] == "Phase 1"
    assert acceptance["passed"] is True
    assert acceptance["fixture_count"] == 17
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["checks"]["canonical_grid_contract"]["invalid_grid_count"] == 0
    assert acceptance["checks"]["edge_silence_zero"]["violation_count"] == 0
    assert acceptance["checks"]["onset_peak_alignment"]["recall"] >= 0.95
    assert acceptance["checks"]["compact_sparse_output"]["ai_payload_delta_fields"] == 0

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "17" in markdown
    assert "Integration into the fallback generator" in markdown
