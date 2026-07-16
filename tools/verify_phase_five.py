from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from uuid import uuid4

from tja_ai_chartgen.audio import instrument_models
from tja_ai_chartgen.audio.instrument_models import (
    InstrumentModelError,
    prepare_instrument_models,
    validate_instrument_model_dir,
)
from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw, StemActivityFrame
from tja_ai_chartgen.evaluation.instrument_benchmark import run_instrument_benchmark
from tja_ai_chartgen.evaluation.phase_five_acceptance import (
    build_phase_five_acceptance_report,
    build_phase_five_behavior_matrix,
    build_phase_five_payload_evidence,
    render_phase_five_acceptance_markdown,
)
from tja_ai_chartgen.utils.paths import write_json
from tja_ai_chartgen.web import create_app

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio"
DEFAULT_OUTPUT_PATH = DEFAULT_FIXTURE_DIR / "phase_five_acceptance.json"
DEFAULT_REPORT_PATH = DEFAULT_FIXTURE_DIR / "phase_five_acceptance.md"


def run_phase_five_acceptance() -> dict[str, Any]:
    behavior_runs: list[dict[str, Any]] = []
    model_runs: list[dict[str, Any]] = []
    payload_runs: list[dict[str, Any]] = []
    web_runs: list[dict[str, Any]] = []
    with _blocked_network_connections():
        for _run_index in range(2):
            behavior_runs.append(build_phase_five_behavior_matrix())
            model_runs.append(_verify_model_profiles())
            payload_runs.append(build_phase_five_payload_evidence())
            web_runs.append(_verify_web_permissions())
    return build_phase_five_acceptance_report(
        behavior_runs,
        model_runs,
        payload_runs,
        web_runs,
        network_blocked=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline Phase 5 stem-role acceptance checks."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        result = run_phase_five_acceptance()
        output_path = write_json(args.output, result)
        report_path = _write_text_atomic(
            args.report,
            render_phase_five_acceptance_markdown(result),
        )
    except (InstrumentModelError, OSError, RuntimeError, ValueError) as error:
        reason = getattr(error, "reason", str(error))
        print(f"Phase 5 acceptance failed: {reason}", file=sys.stderr)
        return 1

    print(
        f"Phase 5 acceptance: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['scenario_count']} deterministic scenarios, "
        f"payload={result['payload_evidence']['payload_schema']})"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if result["passed"] else 1


def _verify_model_profiles() -> dict[str, Any]:
    network_attempt_blocked = False
    offline_environment_applied = False
    with TemporaryDirectory(prefix="tja-phase-five-") as temporary_dir:
        root = Path(temporary_dir)
        full_dir = root / "instrument-v1"
        stem_dir = root / "stem-role-v1"
        audio_path = root / "fixture.wav"
        audio_path.write_bytes(b"phase-five")

        with _deterministic_model_preparation():
            full_manifest = prepare_instrument_models(full_dir, profile="full")
            stem_manifest = prepare_instrument_models(stem_dir, profile="stem-role")

        full_validated = validate_instrument_model_dir(
            full_dir,
            profile="full",
            verify_hashes=True,
        )
        stem_validated = validate_instrument_model_dir(
            stem_dir,
            profile="stem-role",
            verify_hashes=True,
        )
        profile_mismatch_rejected = False
        try:
            validate_instrument_model_dir(stem_dir, profile="full")
        except InstrumentModelError as error:
            profile_mismatch_rejected = error.reason == "model-profile-mismatch"

        def fake_analysis(_audio_path: Path, **kwargs: Any) -> InstrumentAnalysisRaw:
            nonlocal network_attempt_blocked, offline_environment_applied
            offline_environment_applied = (
                os.environ.get("HF_HUB_OFFLINE") == "1"
                and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            )
            try:
                with socket.socket() as connection:
                    connection.connect(("127.0.0.1", 9))
            except RuntimeError:
                network_attempt_blocked = True
            return InstrumentAnalysisRaw(
                feature_version="stem-role-v1",
                status="complete",
                demucs_model="htdemucs",
                classifier_model=None,
                device="cpu",
                analyzed_duration=2.0,
                stem_frames=[StemActivityFrame(time=0.0, drums=0.8, drum_onset=0.9)],
            )

        full_ast_bytes = sum(
            (full_dir / relative).stat().st_size
            for relative in full_manifest.files
            if relative.startswith("ast/")
        )
        clock_values = iter((10.0, 11.5))
        benchmark = run_instrument_benchmark(
            audio_path,
            model_dir=stem_dir,
            profile="stem-role",
            device="auto",
            max_duration=2.0,
            _analysis_runner=fake_analysis,
            _clock=lambda: next(clock_values),
            _rss_probe=lambda: 1_000_000,
        )

    stem_runtime = benchmark["runtime"]
    stem_analysis = benchmark["analysis"]
    stem_model = benchmark["model"]
    return {
        "schema_version": 1,
        "passed": (
            full_validated == full_manifest
            and stem_validated == stem_manifest
            and stem_manifest.classifier_model is None
            and not any(relative.startswith("ast/") for relative in stem_manifest.files)
            and profile_mismatch_rejected
            and benchmark["passed"]
            and network_attempt_blocked
            and offline_environment_applied
        ),
        "benchmark_version": benchmark["benchmark_version"],
        "network_attempt_blocked": network_attempt_blocked,
        "offline_environment_applied": offline_environment_applied,
        "profile_mismatch_rejected": profile_mismatch_rejected,
        "performance_threshold_enforced": False,
        "full": {
            "profile": full_manifest.profile,
            "feature_version": full_manifest.feature_version,
            "hashes_verified": True,
            "file_count": len(full_manifest.files),
            "ast_bytes": full_ast_bytes,
            "classifier_model": full_manifest.classifier_model,
        },
        "stem_role": {
            "profile": stem_manifest.profile,
            "feature_version": stem_manifest.feature_version,
            "hashes_verified": stem_model["hashes_verified"],
            "file_count": stem_model["file_count"],
            "total_bytes": stem_model["total_bytes"],
            "demucs_bytes": stem_model["component_bytes"]["demucs"],
            "ast_bytes": stem_model["component_bytes"]["ast"],
            "classifier_model": stem_manifest.classifier_model,
            "status": stem_analysis["status"],
            "stem_frame_count": stem_analysis["stem_frame_count"],
            "classification_window_count": stem_analysis[
                "classification_window_count"
            ],
            "requested_device": stem_runtime["requested_device"],
            "resolved_device": stem_runtime["resolved_device"],
            "elapsed_seconds": stem_runtime["elapsed_seconds"],
            "realtime_factor": stem_runtime["realtime_factor"],
            "peak_rss_bytes": stem_runtime["peak_rss_bytes"],
            "peak_rss_delta_bytes": stem_runtime["peak_rss_delta_bytes"],
        },
    }


@contextmanager
def _deterministic_model_preparation() -> Iterator[None]:
    original_demucs = instrument_models._prepare_demucs
    original_ast = instrument_models._prepare_ast
    original_offline_validation = instrument_models._validate_offline_loading

    def prepare_demucs(path: Path) -> str:
        path.mkdir(parents=True)
        (path / "htdemucs.yaml").write_text("models: [test]\n", encoding="utf-8")
        (path / "test-deadbeef.th").write_bytes(b"demucs")
        return "test-deadbeef.th"

    def prepare_ast(path: Path) -> None:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"ast")

    instrument_models._prepare_demucs = prepare_demucs
    instrument_models._prepare_ast = prepare_ast
    instrument_models._validate_offline_loading = lambda _path: None
    try:
        yield
    finally:
        instrument_models._prepare_demucs = original_demucs
        instrument_models._prepare_ast = original_ast
        instrument_models._validate_offline_loading = original_offline_validation


def _verify_web_permissions() -> dict[str, Any]:
    with TemporaryDirectory(prefix="tja-phase-five-web-") as temporary_dir:
        root = Path(temporary_dir)
        local = create_app(root / "local")
        remote_default = create_app(root / "remote-default", remote_mode=True)
        remote_explicit = create_app(
            root / "remote-explicit",
            remote_mode=True,
            allow_instrument_analysis=True,
        )
        return {
            "schema_version": 1,
            "local_allowed": bool(local.state.allow_instrument_analysis),
            "remote_default_allowed": bool(
                remote_default.state.allow_instrument_analysis
            ),
            "remote_explicit_allowed": bool(
                remote_explicit.state.allow_instrument_analysis
            ),
        }


@contextmanager
def _blocked_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during Phase 5 acceptance")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during Phase 5 acceptance")

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
