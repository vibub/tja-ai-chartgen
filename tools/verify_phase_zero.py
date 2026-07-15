from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.evaluation.phase_zero_acceptance import (
    build_phase_zero_acceptance_report,
    render_phase_zero_acceptance_markdown,
)
from tja_ai_chartgen.utils.paths import write_json


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_zero_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_zero_acceptance.md"
REBUILD_SCRIPT = DEFAULT_FIXTURE_DIR / "rebuild_click_fixtures.py"
AUDIO_BASELINE_FILENAME = "audio_benchmark_baseline.json"
CHART_BASELINE_FILENAME = "chart_alignment_baseline.json"


def _load_benchmark_function(filename: str, module_name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.benchmark_fixture_directory


benchmark_audio = _load_benchmark_function(
    "benchmark_audio_fixtures.py",
    "phase_zero_benchmark_audio",
)
benchmark_charts = _load_benchmark_function(
    "benchmark_chart_alignment.py",
    "phase_zero_benchmark_charts",
)


def run_phase_zero_acceptance(fixture_dir: Path) -> dict[str, Any]:
    committed_audio = _read_json(fixture_dir / AUDIO_BASELINE_FILENAME)
    committed_chart = _read_json(fixture_dir / CHART_BASELINE_FILENAME)

    with TemporaryDirectory(prefix="tja-phase-zero-") as temporary:
        temporary_root = Path(temporary)
        rebuild_directories = [temporary_root / "rebuild-1", temporary_root / "rebuild-2"]
        rebuild_script = fixture_dir / REBUILD_SCRIPT.name
        for output_dir in rebuild_directories:
            _rebuild_fixtures(output_dir, rebuild_script=rebuild_script)

        with _blocked_network_connections():
            audio_runs = [benchmark_audio(rebuild_directories[0]) for _ in range(2)]
            chart_runs = [benchmark_charts(rebuild_directories[0]) for _ in range(2)]

        return build_phase_zero_acceptance_report(
            fixture_dir=fixture_dir,
            rebuilt_directories=rebuild_directories,
            audio_benchmark_runs=audio_runs,
            chart_benchmark_runs=chart_runs,
            committed_audio_baseline=committed_audio,
            committed_chart_baseline=committed_chart,
            network_blocked=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic, offline Phase 0 acceptance checks."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        result = run_phase_zero_acceptance(args.fixture_dir)
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_zero_acceptance_markdown(result),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 0 acceptance failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Phase 0 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['fixture_count']} fixtures)"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


def _rebuild_fixtures(output_dir: Path, *, rebuild_script: Path = REBUILD_SCRIPT) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(rebuild_script),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"fixture rebuild failed: {detail}")


@contextmanager
def _blocked_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during Phase 0 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 0 acceptance")

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 0 baseline not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Phase 0 baseline must contain a JSON object: {path}")
    return value


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
