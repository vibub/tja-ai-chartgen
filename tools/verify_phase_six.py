from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import socket
import sys
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.evaluation.phase_six_acceptance import (
    build_phase_six_acceptance_report,
    build_phase_six_behavior_matrix,
    build_phase_six_benchmark_evidence,
    render_phase_six_acceptance_markdown,
)
from tja_ai_chartgen.utils.paths import write_json

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_six_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_six_acceptance.md"


def _load_chart_benchmark() -> Any:
    path = Path(__file__).with_name("benchmark_chart_alignment.py")
    spec = importlib.util.spec_from_file_location("phase_six_benchmark_charts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load chart benchmark tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.benchmark_fixture_directory


benchmark_charts = _load_chart_benchmark()


def run_phase_six_acceptance(fixture_dir: Path) -> dict[str, Any]:
    behavior_runs: list[dict[str, Any]] = []
    benchmark_runs: list[dict[str, Any]] = []
    with _blocked_network_connections():
        for _run_index in range(2):
            behavior_runs.append(build_phase_six_behavior_matrix())
            benchmark_runs.append(
                build_phase_six_benchmark_evidence(benchmark_charts(fixture_dir))
            )
    return build_phase_six_acceptance_report(
        behavior_runs,
        benchmark_runs,
        network_blocked=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline Phase 6 quality-report acceptance checks."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        result = run_phase_six_acceptance(args.fixture_dir)
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_six_acceptance_markdown(result),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 6 acceptance failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Phase 6 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['fixture_count']} fixtures, {result['chart_count']} charts, "
        f"{result['scenario_count']} deterministic scenarios)"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


@contextmanager
def _blocked_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during Phase 6 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 6 acceptance")

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


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
