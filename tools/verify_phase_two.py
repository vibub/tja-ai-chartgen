from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import socket
import sys
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.evaluation.phase_two_acceptance import (
    build_phase_two_acceptance_report,
    build_phase_two_constraint_matrix,
    build_shared_salience_metric_evidence,
    render_phase_two_acceptance_markdown,
)
from tja_ai_chartgen.utils.paths import write_json

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_two_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_two_acceptance.md"
CHART_BASELINE_FILENAME = "chart_alignment_baseline.json"


def _load_chart_benchmark() -> Any:
    path = Path(__file__).with_name("benchmark_chart_alignment.py")
    spec = importlib.util.spec_from_file_location("phase_two_benchmark_charts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load chart benchmark tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.benchmark_fixture_directory


benchmark_charts = _load_chart_benchmark()


def run_phase_two_acceptance(fixture_dir: Path) -> dict[str, Any]:
    committed_baseline = _read_json(fixture_dir / CHART_BASELINE_FILENAME)
    with _blocked_network_connections():
        chart_runs = [benchmark_charts(fixture_dir) for _ in range(2)]
        constraint_matrix = build_phase_two_constraint_matrix()
        shared_metric_evidence = build_shared_salience_metric_evidence()
    return build_phase_two_acceptance_report(
        chart_runs,
        committed_chart_baseline=committed_baseline,
        constraint_matrix=constraint_matrix,
        shared_metric_evidence=shared_metric_evidence,
        network_blocked=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline Phase 2 salience-consumer acceptance checks."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        result = run_phase_two_acceptance(args.fixture_dir)
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_two_acceptance_markdown(result),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 2 acceptance failed: {error}", file=sys.stderr)
        return 1

    alignment = result["checks"]["note_onset_alignment_improved"]
    unsupported = result["checks"]["unsupported_note_ratio_reduced"]
    print(
        f"Phase 2 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['fixture_count']} fixtures, {result['chart_count']} charts, "
        f"precision={alignment['current_precision']:.3f}, "
        f"unsupported={unsupported['current_ratio']:.3f})"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


@contextmanager
def _blocked_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during Phase 2 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 2 acceptance")

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
