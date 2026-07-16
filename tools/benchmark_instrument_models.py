from __future__ import annotations

import argparse
from pathlib import Path
import sys
from uuid import uuid4

from tja_ai_chartgen.audio.instrument_models import (
    InstrumentModelError,
    resolve_instrument_model_dir,
)
from tja_ai_chartgen.evaluation.instrument_benchmark import (
    render_instrument_benchmark_markdown,
    run_instrument_benchmark,
)
from tja_ai_chartgen.utils.paths import write_json

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_AUDIO_PATH = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "click_4_4.wav"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate prepared instrument models offline and report model, device, "
            "latency, and memory costs."
        )
    )
    parser.add_argument("--profile", choices=("full", "stem-role"), default="stem-role")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    profile = args.profile
    model_dir = resolve_instrument_model_dir(
        args.model_dir,
        profile=profile,
        start=PROJECT_ROOT,
    )
    slug = profile.replace("-", "_")
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"instrument_benchmark_{slug}.json"
    report_path = args.report or DEFAULT_OUTPUT_DIR / f"instrument_benchmark_{slug}.md"

    try:
        result = run_instrument_benchmark(
            args.audio,
            model_dir=model_dir,
            profile=profile,
            device=args.device,
            max_duration=args.max_duration,
        )
        written_output = write_json(output_path, result)
        written_report = _write_text_atomic(
            report_path,
            render_instrument_benchmark_markdown(result),
        )
    except (FileNotFoundError, InstrumentModelError, OSError, RuntimeError, ValueError) as error:
        reason = getattr(error, "reason", str(error))
        print(f"Instrument benchmark failed: {reason}", file=sys.stderr)
        return 1

    runtime = result["runtime"]
    print(
        f"Instrument benchmark: {'PASS' if result['passed'] else 'FAIL'} "
        f"(profile={profile}, device={runtime['resolved_device']}, "
        f"elapsed={runtime['elapsed_seconds']:.3f}s, "
        f"rtf={runtime['realtime_factor']})"
    )
    print(f"Wrote {written_output}")
    print(f"Wrote {written_report}")
    return 0 if result["passed"] else 1


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
