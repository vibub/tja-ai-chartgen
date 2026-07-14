from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from tja_ai_chartgen.audio.analyze import analyze_audio
from tja_ai_chartgen.evaluation.audio_benchmark import (
    DEFAULT_BAND_ONSET_THRESHOLD,
    DEFAULT_BEAT_TOLERANCE_SECONDS,
    DEFAULT_DOWNBEAT_TOLERANCE_SECONDS,
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    build_audio_benchmark,
    build_fixture_audio_metrics,
    render_audio_benchmark_markdown,
)
from tja_ai_chartgen.utils.paths import write_json


DEFAULT_FIXTURE_DIR = Path(__file__).parents[1] / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "audio_benchmark_baseline.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "audio_benchmark_baseline.md"


def benchmark_fixture_directory(
    fixture_dir: Path,
    *,
    use_beatnet: bool = False,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    beat_tolerance_seconds: float = DEFAULT_BEAT_TOLERANCE_SECONDS,
    downbeat_tolerance_seconds: float = DEFAULT_DOWNBEAT_TOLERANCE_SECONDS,
    band_onset_threshold: float = DEFAULT_BAND_ONSET_THRESHOLD,
) -> dict[str, object]:
    fixture_metrics: list[dict[str, object]] = []
    event_paths = sorted(fixture_dir.glob("*.events.json"))
    if not event_paths:
        raise ValueError(f"No fixture event JSON files found in {fixture_dir}")

    for event_path in event_paths:
        ground_truth = json.loads(event_path.read_text(encoding="utf-8"))
        audio_path = fixture_dir / str(ground_truth["audio"])
        if not audio_path.is_file():
            raise ValueError(f"Missing fixture audio for {event_path.name}: {audio_path.name}")
        analysis = analyze_audio(audio_path, use_beatnet=use_beatnet)
        fixture_metrics.append(
            build_fixture_audio_metrics(
                ground_truth,
                analysis,
                onset_tolerance_seconds=onset_tolerance_seconds,
                beat_tolerance_seconds=beat_tolerance_seconds,
                downbeat_tolerance_seconds=downbeat_tolerance_seconds,
                band_onset_threshold=band_onset_threshold,
            )
        )

    return build_audio_benchmark(
        fixture_metrics,
        use_beatnet=use_beatnet,
        onset_tolerance_seconds=onset_tolerance_seconds,
        beat_tolerance_seconds=beat_tolerance_seconds,
        downbeat_tolerance_seconds=downbeat_tolerance_seconds,
        band_onset_threshold=band_onset_threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark onset, beat, and downbeat alignment on synthetic fixtures."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--use-beatnet", action="store_true")
    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=DEFAULT_ONSET_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--beat-tolerance",
        type=float,
        default=DEFAULT_BEAT_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--downbeat-tolerance",
        type=float,
        default=DEFAULT_DOWNBEAT_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--band-onset-threshold",
        type=float,
        default=DEFAULT_BAND_ONSET_THRESHOLD,
    )
    args = parser.parse_args()
    benchmark = benchmark_fixture_directory(
        args.fixture_dir,
        use_beatnet=args.use_beatnet,
        onset_tolerance_seconds=args.onset_tolerance,
        beat_tolerance_seconds=args.beat_tolerance,
        downbeat_tolerance_seconds=args.downbeat_tolerance,
        band_onset_threshold=args.band_onset_threshold,
    )
    output_path = write_json(args.output, benchmark)
    report_path = _write_text_atomic(
        args.report,
        render_audio_benchmark_markdown(benchmark),
    )
    summary = benchmark["summary"]
    print(
        f"Benchmarked {benchmark['fixture_count']} fixtures: "
        f"onset F1={summary['onset']['f1']:.3f}, "
        f"beat F1={summary['beat']['f1']:.3f}, "
        f"downbeat F1={summary['downbeat']['f1']:.3f}"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0


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
