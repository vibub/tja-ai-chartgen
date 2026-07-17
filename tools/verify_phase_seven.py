from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.evaluation.phase_seven_acceptance import (
    build_phase_seven_acceptance_report,
    build_phase_seven_behavior_matrix,
    build_phase_seven_compatibility_evidence,
    build_phase_seven_interface_evidence,
    build_phase_seven_no_model_pipeline_evidence,
    render_phase_seven_acceptance_markdown,
)
from tja_ai_chartgen.utils.paths import write_json

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_COMPATIBILITY_DIR = PROJECT_ROOT / "tests" / "fixtures" / "compatibility"
DEFAULT_AUDIO_FIXTURE = DEFAULT_FIXTURE_DIR / "click_4_4.wav"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_seven_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_seven_acceptance.md"


def run_phase_seven_acceptance(
    compatibility_dir: Path = DEFAULT_COMPATIBILITY_DIR,
    audio_fixture: Path = DEFAULT_AUDIO_FIXTURE,
) -> dict[str, Any]:
    behavior_runs: list[dict[str, Any]] = []
    pipeline_runs: list[dict[str, Any]] = []
    compatibility_runs: list[dict[str, Any]] = []
    interface_runs: list[dict[str, Any]] = []
    with _blocked_network_connections():
        for run_index in range(2):
            behavior_runs.append(build_phase_seven_behavior_matrix())
            with TemporaryDirectory(
                prefix=f"tja-phase-seven-{run_index}-"
            ) as temporary_dir:
                pipeline_runs.append(
                    build_phase_seven_no_model_pipeline_evidence(
                        audio_fixture,
                        Path(temporary_dir),
                    )
                )
            compatibility_runs.append(
                build_phase_seven_compatibility_evidence(compatibility_dir)
            )
            interface_runs.append(build_phase_seven_interface_evidence())
    return build_phase_seven_acceptance_report(
        behavior_runs,
        pipeline_runs,
        compatibility_runs,
        interface_runs,
        network_blocked=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline Phase 7 consumer-convergence acceptance checks."
    )
    parser.add_argument(
        "--compatibility-dir",
        type=Path,
        default=DEFAULT_COMPATIBILITY_DIR,
    )
    parser.add_argument("--audio-fixture", type=Path, default=DEFAULT_AUDIO_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        result = run_phase_seven_acceptance(
            args.compatibility_dir,
            args.audio_fixture,
        )
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_seven_acceptance_markdown(result),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 7 acceptance failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Phase 7 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['scenario_count']} deterministic consumer scenarios)"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


@contextmanager
def _blocked_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during Phase 7 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 7 acceptance")

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
