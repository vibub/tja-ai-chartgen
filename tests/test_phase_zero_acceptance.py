import importlib.util
import json
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.evaluation.phase_zero_acceptance import (
    build_phase_zero_acceptance_report,
    render_phase_zero_acceptance_markdown,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
ACCEPTANCE_PATH = FIXTURE_DIR / "phase_zero_acceptance.json"
ACCEPTANCE_REPORT_PATH = FIXTURE_DIR / "phase_zero_acceptance.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "verify_phase_zero.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("verify_phase_zero", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
verify_phase_zero = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(verify_phase_zero)


def _audio_benchmark() -> dict[str, object]:
    return {
        "schema_version": 2,
        "benchmark_version": "audio-alignment-v2",
        "fixture_ground_truth_schema_version": 1,
        "fixture_count": 1,
        "settings": {"use_beatnet": False},
        "fixtures": [{"audio": "sample.wav"}],
    }


def _chart_benchmark() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_version": "chart-alignment-v1",
        "fixture_ground_truth_schema_version": 1,
        "fixture_count": 1,
        "chart_count": 4,
        "settings": {"use_beatnet": False},
        "charts": [
            {"audio": "sample.wav", "course": course}
            for course in ("Easy", "Normal", "Hard", "Oni")
        ],
    }


def _write_fixture_artifacts(directory: Path, *, audio: bytes = b"audio") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ground_truth.schema.json").write_text("{}\n", encoding="utf-8")
    (directory / "sample.events.json").write_text(
        json.dumps({"schema_version": 1, "audio": "sample.wav"}) + "\n",
        encoding="utf-8",
    )
    (directory / "sample.wav").write_bytes(audio)


def _acceptance_report(tmp_path: Path):
    fixture_dir = tmp_path / "fixtures"
    rebuild_one = tmp_path / "rebuild-1"
    rebuild_two = tmp_path / "rebuild-2"
    for directory in (fixture_dir, rebuild_one, rebuild_two):
        _write_fixture_artifacts(directory)
    audio = _audio_benchmark()
    chart = _chart_benchmark()
    return build_phase_zero_acceptance_report(
        fixture_dir=fixture_dir,
        rebuilt_directories=[rebuild_one, rebuild_two],
        audio_benchmark_runs=[audio, audio],
        chart_benchmark_runs=[chart, chart],
        committed_audio_baseline=audio,
        committed_chart_baseline=chart,
        network_blocked=True,
    )


def test_build_phase_zero_acceptance_report_passes_all_exit_conditions(tmp_path):
    report = _acceptance_report(tmp_path)

    assert report["schema_version"] == 1
    assert report["acceptance_version"] == "phase-zero-acceptance-v1"
    assert report["passed"] is True
    assert report["fixture_count"] == 1
    assert report["checks"]["fixture_ground_truth_rebuild"]["artifact_count"] == 3
    assert report["checks"]["audio_analysis_baseline"]["runs_stable"] is True
    assert report["checks"]["chart_alignment_baseline"]["chart_count"] == 4
    assert report["checks"]["repository_privacy"]["passed"] is True


def test_phase_zero_acceptance_fails_when_rebuild_content_drifts(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    rebuild_one = tmp_path / "rebuild-1"
    rebuild_two = tmp_path / "rebuild-2"
    _write_fixture_artifacts(fixture_dir)
    _write_fixture_artifacts(rebuild_one)
    _write_fixture_artifacts(rebuild_two, audio=b"changed")
    audio = _audio_benchmark()
    chart = _chart_benchmark()

    report = build_phase_zero_acceptance_report(
        fixture_dir=fixture_dir,
        rebuilt_directories=[rebuild_one, rebuild_two],
        audio_benchmark_runs=[audio, audio],
        chart_benchmark_runs=[chart, chart],
        committed_audio_baseline=audio,
        committed_chart_baseline=chart,
        network_blocked=True,
    )

    assert report["passed"] is False
    rebuild = report["checks"]["fixture_ground_truth_rebuild"]
    assert rebuild["passed"] is False
    assert rebuild["mismatches"] == ["content-mismatch-2:sample.wav"]


def test_phase_zero_acceptance_fails_on_private_or_external_values(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    rebuild_one = tmp_path / "rebuild-1"
    rebuild_two = tmp_path / "rebuild-2"
    for directory in (fixture_dir, rebuild_one, rebuild_two):
        _write_fixture_artifacts(directory)
    audio = _audio_benchmark() | {"title": "Real Song"}
    chart = _chart_benchmark() | {"debug_path": "C:\\private\\song.wav"}

    report = build_phase_zero_acceptance_report(
        fixture_dir=fixture_dir,
        rebuilt_directories=[rebuild_one, rebuild_two],
        audio_benchmark_runs=[audio, audio],
        chart_benchmark_runs=[chart, chart],
        committed_audio_baseline=audio,
        committed_chart_baseline=chart,
        network_blocked=True,
    )

    privacy = report["checks"]["repository_privacy"]
    assert report["passed"] is False
    assert privacy["forbidden_key_count"] == 1
    assert privacy["forbidden_string_count"] == 1


def test_render_phase_zero_acceptance_markdown_lists_each_exit_condition(tmp_path):
    markdown = render_phase_zero_acceptance_markdown(_acceptance_report(tmp_path))

    assert "# Phase 0 acceptance report" in markdown
    assert "Result: **PASS**" in markdown
    assert "Fixture and ground truth rebuild from one source" in markdown
    assert "Benchmarks run without network access" in markdown
    assert "No real-song identity" in markdown


def test_run_phase_zero_acceptance_rebuilds_twice_and_blocks_network(
    tmp_path, monkeypatch
):
    fixture_dir = tmp_path / "fixtures"
    _write_fixture_artifacts(fixture_dir)
    audio = _audio_benchmark()
    chart = _chart_benchmark()
    (fixture_dir / "audio_benchmark_baseline.json").write_text(
        json.dumps(audio), encoding="utf-8"
    )
    (fixture_dir / "chart_alignment_baseline.json").write_text(
        json.dumps(chart), encoding="utf-8"
    )
    rebuild_calls: list[Path] = []
    network_states: list[bool] = []
    original_connect = socket.socket.connect

    def fake_rebuild(output_dir: Path, *, rebuild_script: Path) -> None:
        assert rebuild_script == fixture_dir / "rebuild_click_fixtures.py"
        rebuild_calls.append(output_dir)
        _write_fixture_artifacts(output_dir)

    def fake_audio(_fixture_dir: Path):
        network_states.append(socket.socket.connect is not original_connect)
        return audio

    def fake_chart(_fixture_dir: Path):
        network_states.append(socket.socket.connect is not original_connect)
        return chart

    monkeypatch.setattr(verify_phase_zero, "_rebuild_fixtures", fake_rebuild)
    monkeypatch.setattr(verify_phase_zero, "benchmark_audio", fake_audio)
    monkeypatch.setattr(verify_phase_zero, "benchmark_charts", fake_chart)

    report = verify_phase_zero.run_phase_zero_acceptance(fixture_dir)

    assert report["passed"] is True
    assert len(rebuild_calls) == 2
    assert network_states == [True, True, True, True]
    assert socket.socket.connect is original_connect


def test_phase_zero_main_writes_json_and_markdown(tmp_path, monkeypatch):
    result = _acceptance_report(tmp_path)
    output_path = tmp_path / "acceptance.json"
    report_path = tmp_path / "acceptance.md"
    monkeypatch.setattr(
        verify_phase_zero,
        "run_phase_zero_acceptance",
        lambda _fixture_dir: result,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_phase_zero.py",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert verify_phase_zero.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert "Result: **PASS**" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_phase_zero_acceptance_report_passes_all_checks():
    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    assert acceptance["schema_version"] == 1
    assert acceptance["acceptance_version"] == "phase-zero-acceptance-v1"
    assert acceptance["phase"] == "Phase 0"
    assert acceptance["passed"] is True
    assert acceptance["fixture_count"] == 17
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["checks"]["fixture_ground_truth_rebuild"] == {
        "passed": True,
        "fixture_count": 17,
        "audio_files": sorted(path.name for path in FIXTURE_DIR.glob("*.wav")),
        "artifact_count": 35,
        "rebuild_count": 2,
        "mismatches": [],
    }
    assert acceptance["checks"]["audio_analysis_baseline"]["runs_stable"] is True
    assert acceptance["checks"]["chart_alignment_baseline"]["runs_stable"] is True
    assert acceptance["checks"]["repository_privacy"]["passed"] is True

    markdown = ACCEPTANCE_REPORT_PATH.read_text(encoding="utf-8")
    assert "Result: **PASS**" in markdown
    assert "35 artifacts, 2 independent rebuilds" in markdown
    assert "`audio-alignment-v2`, 17 fixtures" in markdown
    assert "`chart-alignment-v1`, 68 charts" in markdown
