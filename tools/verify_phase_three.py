from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.audio.analyze import analyze_audio
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.evaluation.phase_three_acceptance import (
    build_fill_burst_fixture_evidence,
    build_phase_three_acceptance_report,
    build_phase_three_behavior_matrix,
    render_phase_three_acceptance_markdown,
)
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.salience import build_burst_salience
from tja_ai_chartgen.features.structure import analyze_song_structure
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.utils.paths import write_json

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_three_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_three_acceptance.md"
PHASE_TWO_ACCEPTANCE_FILENAME = "phase_two_acceptance.json"
FILL_FIXTURE_FILENAME = "fill_burst_120.wav"
FILL_GROUND_TRUTH_FILENAME = "fill_burst_120.events.json"


def _load_chart_benchmark() -> Any:
    path = Path(__file__).with_name("benchmark_chart_alignment.py")
    spec = importlib.util.spec_from_file_location("phase_three_benchmark_charts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load chart benchmark tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.benchmark_fixture_directory


benchmark_charts = _load_chart_benchmark()


def run_phase_three_acceptance(fixture_dir: Path) -> dict[str, Any]:
    phase_two_acceptance = _read_json(
        fixture_dir / PHASE_TWO_ACCEPTANCE_FILENAME
    )
    chart_runs: list[dict[str, Any]] = []
    behavior_runs: list[dict[str, Any]] = []
    fill_runs: list[dict[str, Any]] = []
    with _blocked_network_connections():
        for _run_index in range(2):
            chart_runs.append(benchmark_charts(fixture_dir))
            behavior_runs.append(build_phase_three_behavior_matrix())
            fill_runs.append(_benchmark_fill_fixture(fixture_dir))
    return build_phase_three_acceptance_report(
        chart_runs,
        behavior_runs,
        fill_runs,
        phase_two_acceptance=phase_two_acceptance,
        network_blocked=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline Phase 3 accent/color/fill acceptance checks."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        result = run_phase_three_acceptance(args.fixture_dir)
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_three_acceptance_markdown(result),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 3 acceptance failed: {error}", file=sys.stderr)
        return 1

    color = result["checks"]["don_ka_balance"]
    fill = result["checks"]["fill_burst_fixture_alignment"]
    print(
        f"Phase 3 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['fixture_count']} fixtures, {result['chart_count']} charts, "
        f"ka_ratio={color['ka_ratio']:.3f}, bursts={fill['reliable_burst_indexes']})"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


def _benchmark_fill_fixture(fixture_dir: Path) -> dict[str, Any]:
    fixture_path = fixture_dir / FILL_FIXTURE_FILENAME
    ground_truth = _read_json(fixture_dir / FILL_GROUND_TRUTH_FILENAME)
    with TemporaryDirectory(prefix="tja-phase-three-") as temporary_dir:
        ogg_path = convert_to_ogg(
            fixture_path,
            Path(temporary_dir) / "fill_burst_120.ogg",
        )
        raw = analyze_audio(ogg_path)
        bars = analyze_song_structure(build_bar_features(raw)).bars
    burst_salience = build_burst_salience(bars)
    chart_bars = generate_fallback_chart_bars(
        bars,
        density="high",
        special_notes=True,
        course="Oni",
        level=10,
    )
    return build_fill_burst_fixture_evidence(
        ground_truth,
        bars,
        burst_salience,
        chart_bars,
    )


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
        raise RuntimeError("network access is disabled during Phase 3 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 3 acceptance")

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
