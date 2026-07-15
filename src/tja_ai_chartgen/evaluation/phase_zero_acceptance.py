from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any


PHASE_ZERO_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_ZERO_ACCEPTANCE_VERSION = "phase-zero-acceptance-v1"
AUDIO_BENCHMARK_VERSION = "audio-alignment-v2"
CHART_BENCHMARK_VERSION = "chart-alignment-v1"
FORBIDDEN_PERSISTED_KEYS = {
    "api_key",
    "artist",
    "base_url",
    "input_audio",
    "output_dir",
    "title",
    "wave",
}
_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def build_phase_zero_acceptance_report(
    *,
    fixture_dir: Path,
    rebuilt_directories: list[Path],
    audio_benchmark_runs: list[dict[str, Any]],
    chart_benchmark_runs: list[dict[str, Any]],
    committed_audio_baseline: dict[str, Any],
    committed_chart_baseline: dict[str, Any],
    network_blocked: bool,
) -> dict[str, Any]:
    artifact_check = _fixture_artifact_check(fixture_dir, rebuilt_directories)
    audio_check = _benchmark_check(
        runs=audio_benchmark_runs,
        committed=committed_audio_baseline,
        expected_version=AUDIO_BENCHMARK_VERSION,
        expected_fixture_count=artifact_check["fixture_count"],
        expected_chart_count=None,
    )
    chart_check = _benchmark_check(
        runs=chart_benchmark_runs,
        committed=committed_chart_baseline,
        expected_version=CHART_BENCHMARK_VERSION,
        expected_fixture_count=artifact_check["fixture_count"],
        expected_chart_count=artifact_check["fixture_count"] * 4,
    )
    privacy_check = _privacy_check(
        committed_audio_baseline,
        committed_chart_baseline,
        allowed_audio_names=set(artifact_check["audio_files"]),
    )
    checks = {
        "fixture_ground_truth_rebuild": artifact_check,
        "offline_benchmark_execution": {
            "passed": network_blocked,
            "socket_connections_blocked": network_blocked,
            "audio_use_beatnet": committed_audio_baseline.get("settings", {}).get(
                "use_beatnet"
            ),
            "chart_use_beatnet": committed_chart_baseline.get("settings", {}).get(
                "use_beatnet"
            ),
        },
        "audio_analysis_baseline": audio_check,
        "chart_alignment_baseline": chart_check,
        "repository_privacy": privacy_check,
    }
    return {
        "schema_version": PHASE_ZERO_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_ZERO_ACCEPTANCE_VERSION,
        "phase": "Phase 0",
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "fixture_count": artifact_check["fixture_count"],
        "checks": checks,
    }


def render_phase_zero_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    artifact = checks["fixture_ground_truth_rebuild"]
    audio = checks["audio_analysis_baseline"]
    chart = checks["chart_alignment_baseline"]
    privacy = checks["repository_privacy"]
    lines = [
        "# Phase 0 acceptance report",
        "",
        f"- Acceptance: `{report['acceptance_version']}`",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Fixture count: {report['fixture_count']}",
        "",
        "## Exit conditions",
        "",
        "| Exit condition | Result | Evidence |",
        "| --- | :---: | --- |",
        _check_row(
            "Fixture and ground truth rebuild from one source",
            artifact["passed"],
            f"{artifact['artifact_count']} artifacts, {artifact['rebuild_count']} independent rebuilds",
        ),
        _check_row(
            "Benchmarks run without network access",
            checks["offline_benchmark_execution"]["passed"],
            "socket connect/connect_ex blocked during both benchmark passes",
        ),
        _check_row(
            "Audio analysis baseline exists",
            audio["passed"],
            f"`{audio['benchmark_version']}`, {audio['fixture_count']} fixtures",
        ),
        _check_row(
            "Chart alignment baseline exists",
            chart["passed"],
            f"`{chart['benchmark_version']}`, {chart['chart_count']} charts",
        ),
        _check_row(
            "Repeated runs are stable",
            audio["runs_stable"] and chart["runs_stable"],
            "audio and chart benchmark dictionaries are identical across two runs",
        ),
        _check_row(
            "No real-song identity, absolute path, or service secret is persisted",
            privacy["passed"],
            f"{privacy['scanned_string_count']} strings scanned, no forbidden values",
        ),
        "",
        "## Baseline verification",
        "",
        "| Baseline | Schema | Fixture/chart coverage | Stable | Matches committed baseline |",
        "| --- | ---: | ---: | :---: | :---: |",
        f"| Audio analysis | {audio['schema_version']} | {audio['fixture_count']} fixtures | "
        f"{_mark(audio['runs_stable'])} | {_mark(audio['matches_committed_baseline'])} |",
        f"| Chart alignment | {chart['schema_version']} | {chart['chart_count']} charts | "
        f"{_mark(chart['runs_stable'])} | {_mark(chart['matches_committed_baseline'])} |",
        "",
        "## Privacy and provenance",
        "",
        f"- Synthetic audio files: {artifact['fixture_count']}",
        f"- Rebuilt artifacts: {artifact['artifact_count']}",
        f"- Forbidden persisted keys: {privacy['forbidden_key_count']}",
        f"- Absolute paths or URLs: {privacy['forbidden_string_count']}",
        f"- Unknown audio references: {privacy['unknown_audio_reference_count']}",
        "",
        "The acceptance runner regenerates fixtures in temporary directories, blocks socket connections while running both benchmarks twice, and does not access real-song datasets.",
        "",
    ]
    return "\n".join(lines)


def _fixture_artifact_check(
    fixture_dir: Path,
    rebuilt_directories: list[Path],
) -> dict[str, Any]:
    event_paths = sorted(fixture_dir.glob("*.events.json"))
    audio_files = sorted(
        path.name.removesuffix(".events.json") + ".wav" for path in event_paths
    )
    artifact_names = ["ground_truth.schema.json"] + [
        name
        for event_path, audio_name in zip(event_paths, audio_files, strict=True)
        for name in (event_path.name, audio_name)
    ]
    mismatches: list[str] = []
    for name in artifact_names:
        expected_path = fixture_dir / name
        if not expected_path.is_file():
            mismatches.append(f"missing-committed:{name}")
            continue
        expected_hash = _file_hash(expected_path)
        for index, rebuilt_dir in enumerate(rebuilt_directories, start=1):
            rebuilt_path = rebuilt_dir / name
            if not rebuilt_path.is_file():
                mismatches.append(f"missing-rebuild-{index}:{name}")
            elif _file_hash(rebuilt_path) != expected_hash:
                mismatches.append(f"content-mismatch-{index}:{name}")
    return {
        "passed": bool(event_paths) and len(rebuilt_directories) >= 2 and not mismatches,
        "fixture_count": len(event_paths),
        "audio_files": audio_files,
        "artifact_count": len(artifact_names),
        "rebuild_count": len(rebuilt_directories),
        "mismatches": mismatches,
    }


def _benchmark_check(
    *,
    runs: list[dict[str, Any]],
    committed: dict[str, Any],
    expected_version: str,
    expected_fixture_count: int,
    expected_chart_count: int | None,
) -> dict[str, Any]:
    first = runs[0] if runs else {}
    stable = len(runs) >= 2 and all(run == first for run in runs[1:])
    fixture_count = int(committed.get("fixture_count", 0))
    chart_count = int(committed.get("chart_count", 0)) if expected_chart_count is not None else None
    valid = (
        committed.get("schema_version") in {1, 2}
        and committed.get("benchmark_version") == expected_version
        and committed.get("fixture_ground_truth_schema_version") == 1
        and fixture_count == expected_fixture_count
        and committed.get("settings", {}).get("use_beatnet") is False
        and (expected_chart_count is None or chart_count == expected_chart_count)
    )
    return {
        "passed": valid and stable,
        "schema_version": committed.get("schema_version"),
        "benchmark_version": committed.get("benchmark_version"),
        "fixture_count": fixture_count,
        "chart_count": chart_count,
        "run_count": len(runs),
        "runs_stable": stable,
        "matches_committed_baseline": bool(runs) and first == committed,
    }


def _privacy_check(
    *payloads: dict[str, Any],
    allowed_audio_names: set[str],
) -> dict[str, Any]:
    forbidden_keys: list[str] = []
    forbidden_strings: list[str] = []
    audio_references: list[str] = []
    scanned_string_count = 0

    def visit(value: Any, path: str = "$") -> None:
        nonlocal scanned_string_count
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = str(key).casefold()
                child_path = f"{path}.{key}"
                if normalized_key in FORBIDDEN_PERSISTED_KEYS:
                    forbidden_keys.append(child_path)
                if normalized_key == "audio" and isinstance(item, str):
                    audio_references.append(item)
                visit(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            scanned_string_count += 1
            if (
                _ABSOLUTE_WINDOWS_PATH.match(value)
                or value.startswith(("/", "\\\\"))
                or "://" in value
            ):
                forbidden_strings.append(f"{path}:{value}")

    for payload in payloads:
        visit(payload)
    unknown_audio = sorted(set(audio_references) - allowed_audio_names)
    return {
        "passed": not forbidden_keys and not forbidden_strings and not unknown_audio,
        "scanned_string_count": scanned_string_count,
        "forbidden_key_count": len(forbidden_keys),
        "forbidden_string_count": len(forbidden_strings),
        "unknown_audio_reference_count": len(unknown_audio),
        "forbidden_keys": forbidden_keys,
        "forbidden_strings": forbidden_strings,
        "unknown_audio_references": unknown_audio,
    }


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {_mark(passed)} | {evidence} |"


def _mark(value: bool) -> str:
    return "✓" if value else "✗"
