from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from tja_ai_chartgen.audio.analyze import analyze_audio
from tja_ai_chartgen.evaluation.chart_alignment import (
    DEFAULT_BEAT_EVIDENCE_TOLERANCE_SECONDS,
    DEFAULT_NOTE_ONSET_TOLERANCE_SECONDS,
    build_chart_alignment_benchmark,
    build_chart_alignment_metrics,
    render_chart_alignment_markdown,
)
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.resolution import build_resolution_plan
from tja_ai_chartgen.features.structure import analyze_song_structure
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.quality import build_quality_report
from tja_ai_chartgen.utils.paths import write_json


DEFAULT_FIXTURE_DIR = Path(__file__).parents[1] / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "chart_alignment_baseline.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "chart_alignment_baseline.md"
DEFAULT_STYLE = "technical"
DEFAULT_DENSITY = "auto"
DEFAULT_SPECIAL_NOTES = False
COURSE_PROFILES = (
    ("Easy", 3),
    ("Normal", 5),
    ("Hard", 7),
    ("Oni", 10),
)


def benchmark_fixture_directory(
    fixture_dir: Path,
    *,
    use_beatnet: bool = False,
    style: str = DEFAULT_STYLE,
    density: str = DEFAULT_DENSITY,
    special_notes: bool = DEFAULT_SPECIAL_NOTES,
    note_onset_tolerance_seconds: float = DEFAULT_NOTE_ONSET_TOLERANCE_SECONDS,
    beat_evidence_tolerance_seconds: float = DEFAULT_BEAT_EVIDENCE_TOLERANCE_SECONDS,
) -> dict[str, object]:
    event_paths = sorted(fixture_dir.glob("*.events.json"))
    if not event_paths:
        raise ValueError(f"No fixture event JSON files found in {fixture_dir}")

    chart_metrics: list[dict[str, object]] = []
    for event_path in event_paths:
        ground_truth = json.loads(event_path.read_text(encoding="utf-8"))
        audio_path = fixture_dir / str(ground_truth["audio"])
        if not audio_path.is_file():
            raise ValueError(f"Missing fixture audio for {event_path.name}: {audio_path.name}")

        analysis = analyze_audio(audio_path, use_beatnet=use_beatnet)
        structure = analyze_song_structure(build_bar_features(analysis))
        feature_bars = structure.bars
        resolution_plan = build_resolution_plan(analysis, feature_bars)
        for course, level in COURSE_PROFILES:
            first = generate_fallback_chart_bars(
                feature_bars,
                style=style,
                density=density,
                special_notes=special_notes,
                course=course,
                level=level,
                resolution_plan=resolution_plan,
            )
            second = generate_fallback_chart_bars(
                feature_bars,
                style=style,
                density=density,
                special_notes=special_notes,
                course=course,
                level=level,
                resolution_plan=resolution_plan,
            )
            metrics = build_chart_alignment_metrics(
                ground_truth,
                feature_bars,
                first,
                course=course,
                level=level,
                deterministic=first == second,
                quality_report=build_quality_report(
                    first,
                    feature_bars,
                    resolution_plan,
                ),
                note_onset_tolerance_seconds=note_onset_tolerance_seconds,
                beat_evidence_tolerance_seconds=beat_evidence_tolerance_seconds,
            )
            metrics["analyzer"] = analysis.analyzer
            metrics["analyzed_bpm"] = analysis.bpm
            metrics["analyzed_offset"] = analysis.offset
            metrics["analyzed_time_signature"] = analysis.time_signature
            metrics["resolution_policy"] = resolution_plan.policy_version
            metrics["base_resolution"] = resolution_plan.base_resolution
            chart_metrics.append(metrics)

    return build_chart_alignment_benchmark(
        chart_metrics,
        fixture_count=len(event_paths),
        use_beatnet=use_beatnet,
        style=style,
        density=density,
        special_notes=special_notes,
        note_onset_tolerance_seconds=note_onset_tolerance_seconds,
        beat_evidence_tolerance_seconds=beat_evidence_tolerance_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic rule-chart alignment on synthetic fixtures."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--use-beatnet", action="store_true")
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--density", default=DEFAULT_DENSITY)
    parser.add_argument("--special-notes", action="store_true")
    parser.add_argument(
        "--note-onset-tolerance",
        type=float,
        default=DEFAULT_NOTE_ONSET_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--beat-evidence-tolerance",
        type=float,
        default=DEFAULT_BEAT_EVIDENCE_TOLERANCE_SECONDS,
    )
    args = parser.parse_args()

    benchmark = benchmark_fixture_directory(
        args.fixture_dir,
        use_beatnet=args.use_beatnet,
        style=args.style,
        density=args.density,
        special_notes=args.special_notes,
        note_onset_tolerance_seconds=args.note_onset_tolerance,
        beat_evidence_tolerance_seconds=args.beat_evidence_tolerance,
    )
    output_path = write_json(args.output, benchmark)
    report_path = _write_text_atomic(
        args.report,
        render_chart_alignment_markdown(benchmark),
    )
    summary = benchmark["summary"]
    print(
        f"Benchmarked {benchmark['chart_count']} charts across "
        f"{benchmark['fixture_count']} fixtures: "
        f"note/onset precision={summary['note_onset']['precision']:.3f}, "
        f"strong response={summary['strong_onset_response']['recall']:.3f}, "
        f"downbeat response={summary['downbeat_response']['recall']:.3f}"
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
