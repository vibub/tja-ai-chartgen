from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import sys
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.audio.analyze import analyze_audio
from tja_ai_chartgen.evaluation.phase_one_acceptance import (
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    DEFAULT_PEAK_HIT_THRESHOLD,
    build_fixture_salience_metrics,
    build_phase_one_acceptance_report,
    build_salience_benchmark,
    render_phase_one_acceptance_markdown,
)
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.salience import build_don_ka_salience
from tja_ai_chartgen.features.structure import analyze_song_structure
from tja_ai_chartgen.utils.paths import write_json


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_one_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_one_acceptance.md"


def benchmark_fixture_directory(
    fixture_dir: Path,
    *,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    peak_hit_threshold: float = DEFAULT_PEAK_HIT_THRESHOLD,
) -> dict[str, Any]:
    event_paths = sorted(fixture_dir.glob("*.events.json"))
    if not event_paths:
        raise ValueError(f"No fixture event JSON files found in {fixture_dir}")

    fixture_metrics: list[dict[str, Any]] = []
    for event_path in event_paths:
        ground_truth = json.loads(event_path.read_text(encoding="utf-8"))
        audio_path = fixture_dir / str(ground_truth["audio"])
        if not audio_path.is_file():
            raise ValueError(f"Missing fixture audio for {event_path.name}: {audio_path.name}")
        analysis = analyze_audio(audio_path, use_beatnet=False)
        bars = analyze_song_structure(build_bar_features(analysis)).bars
        salience = build_don_ka_salience(bars)
        fixture_metrics.append(
            build_fixture_salience_metrics(
                ground_truth,
                bars,
                salience,
                onset_tolerance_seconds=onset_tolerance_seconds,
                peak_hit_threshold=peak_hit_threshold,
            )
        )

    return build_salience_benchmark(
        fixture_metrics,
        onset_tolerance_seconds=onset_tolerance_seconds,
        peak_hit_threshold=peak_hit_threshold,
    )


def run_phase_one_acceptance(
    fixture_dir: Path,
    *,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    peak_hit_threshold: float = DEFAULT_PEAK_HIT_THRESHOLD,
) -> dict[str, Any]:
    with _blocked_network_connections():
        runs = [
            benchmark_fixture_directory(
                fixture_dir,
                onset_tolerance_seconds=onset_tolerance_seconds,
                peak_hit_threshold=peak_hit_threshold,
            )
            for _ in range(2)
        ]
    return build_phase_one_acceptance_report(runs, network_blocked=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline Phase 1 rhythmic-salience acceptance checks."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=DEFAULT_ONSET_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--peak-hit-threshold",
        type=float,
        default=DEFAULT_PEAK_HIT_THRESHOLD,
    )
    args = parser.parse_args()

    try:
        result = run_phase_one_acceptance(
            args.fixture_dir,
            onset_tolerance_seconds=args.onset_tolerance,
            peak_hit_threshold=args.peak_hit_threshold,
        )
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_one_acceptance_markdown(result),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 1 acceptance failed: {error}", file=sys.stderr)
        return 1

    onset = result["checks"]["onset_peak_alignment"]
    print(
        f"Phase 1 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['fixture_count']} fixtures, "
        f"peak precision={onset['precision']:.3f}, recall={onset['recall']:.3f})"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


@contextmanager
def _blocked_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during Phase 1 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 1 acceptance")

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
