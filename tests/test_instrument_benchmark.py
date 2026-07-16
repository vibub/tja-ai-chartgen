from __future__ import annotations

import hashlib
import os
from pathlib import Path
import socket
import sys

from tja_ai_chartgen.audio.instrument_models import InstrumentModelManifest
from tja_ai_chartgen.audio.instruments import (
    InstrumentAnalysisRaw,
    InstrumentClassificationWindow,
    StemActivityFrame,
)
from tja_ai_chartgen.evaluation.instrument_benchmark import (
    build_model_inventory,
    current_process_rss_bytes,
    render_instrument_benchmark_markdown,
    run_instrument_benchmark,
)
from tja_ai_chartgen.utils.paths import write_json


def _write_model_tree(path: Path, *, profile: str = "stem-role") -> InstrumentModelManifest:
    demucs_dir = path / "demucs"
    demucs_dir.mkdir(parents=True)
    yaml_path = demucs_dir / "htdemucs.yaml"
    weight_path = demucs_dir / "test-deadbeef.th"
    yaml_path.write_text("models: [test]\n", encoding="utf-8")
    weight_path.write_bytes(b"demucs")
    files = {
        candidate.relative_to(path).as_posix(): hashlib.sha256(candidate.read_bytes()).hexdigest()
        for candidate in (yaml_path, weight_path)
    }
    stem_role = profile == "stem-role"
    if not stem_role:
        ast_dir = path / "ast"
        ast_dir.mkdir()
        for name, content in {
            "config.json": b"{}",
            "preprocessor_config.json": b"{}",
            "model.safetensors": b"ast",
        }.items():
            candidate = ast_dir / name
            candidate.write_bytes(content)
            files[candidate.relative_to(path).as_posix()] = hashlib.sha256(content).hexdigest()
    manifest = InstrumentModelManifest(
        feature_version="stem-role-v1" if stem_role else "instrument-v1",
        profile=profile,
        demucs_revision="test-deadbeef.th",
        classifier_model=None if stem_role else "MIT/ast-finetuned-audioset-10-10-0.4593",
        classifier_revision=(
            None if stem_role else "f826b80d28226b62986cc218e5cec390b1096902"
        ),
        code_licenses={"demucs": "MIT"},
        weight_licenses={"htdemucs": "CC-BY-NC-4.0"},
        files=files,
    )
    write_json(path / "instrument_models.json", manifest)
    return manifest


def test_model_inventory_reports_component_sizes_without_paths(tmp_path):
    manifest = _write_model_tree(tmp_path, profile="full")

    inventory = build_model_inventory(tmp_path, manifest)

    assert inventory["file_count"] == 5
    assert inventory["component_bytes"]["demucs"] > 0
    assert inventory["component_bytes"]["ast"] > 0
    assert inventory["total_bytes"] == sum(inventory["component_bytes"].values())
    assert inventory["hashes_verified"] is True
    assert "model_dir" not in inventory


def test_instrument_benchmark_enforces_offline_runtime_and_reports_cost(tmp_path):
    model_dir = tmp_path / "models"
    _write_model_tree(model_dir)
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"audio")
    calls = []

    def fake_analysis(path, **kwargs):
        calls.append((path, kwargs))
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        try:
            with socket.socket() as connection:
                connection.connect(("127.0.0.1", 9))
        except RuntimeError as error:
            assert "network access is disabled" in str(error)
        else:
            raise AssertionError("benchmark must block socket connections")
        return InstrumentAnalysisRaw(
            feature_version="stem-role-v1",
            status="complete",
            demucs_model="htdemucs",
            device="cpu",
            analyzed_duration=2.0,
            stem_frames=[StemActivityFrame(time=0.0, drums=0.8)],
        )

    clock_values = iter((10.0, 11.5))
    report = run_instrument_benchmark(
        audio_path,
        model_dir=model_dir,
        profile="stem-role",
        device="auto",
        max_duration=2.0,
        _analysis_runner=fake_analysis,
        _clock=lambda: next(clock_values),
        _rss_probe=lambda: 1000,
    )

    assert report["passed"] is True
    assert report["offline"] == {
        "network_blocked": True,
        "hf_hub_offline": True,
        "transformers_offline": True,
    }
    assert report["runtime"]["elapsed_seconds"] == 1.5
    assert report["runtime"]["realtime_factor"] == 0.75
    assert report["runtime"]["peak_rss_delta_bytes"] == 0
    assert report["analysis"]["stem_frame_count"] == 1
    assert report["analysis"]["classification_window_count"] == 0
    assert calls[0][1]["profile"] == "stem-role"


def test_full_benchmark_requires_classifier_evidence(tmp_path):
    model_dir = tmp_path / "models"
    _write_model_tree(model_dir, profile="full")
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"audio")

    def fake_analysis(*_args, **_kwargs):
        return InstrumentAnalysisRaw(
            feature_version="instrument-v1",
            status="complete",
            demucs_model="htdemucs",
            classifier_model="MIT/ast-finetuned-audioset-10-10-0.4593",
            device="cpu",
            analyzed_duration=1.0,
            stem_frames=[StemActivityFrame(time=0.0, drums=0.8)],
            classification_windows=[
                InstrumentClassificationWindow(start_time=0.0, end_time=1.0)
            ],
        )

    report = run_instrument_benchmark(
        audio_path,
        model_dir=model_dir,
        profile="full",
        _analysis_runner=fake_analysis,
        _clock=iter((0.0, 1.0)).__next__,
        _rss_probe=lambda: None,
    )

    assert report["passed"] is True
    assert report["model"]["component_bytes"]["ast"] > 0
    assert report["analysis"]["classification_window_count"] == 1


def test_benchmark_markdown_renders_unavailable_memory():
    report = {
        "passed": True,
        "profile": "stem-role",
        "model": {
            "file_count": 2,
            "total_bytes": 1024,
            "component_bytes": {"demucs": 1024, "ast": 0, "other": 0},
        },
        "runtime": {
            "audio_file": "song.wav",
            "requested_device": "auto",
            "resolved_device": "cpu",
            "analyzed_duration_seconds": 2.0,
            "elapsed_seconds": 1.0,
            "realtime_factor": 0.5,
            "peak_rss_bytes": None,
            "peak_rss_delta_bytes": None,
        },
        "analysis": {
            "feature_version": "stem-role-v1",
            "status": "complete",
            "reason": None,
            "stem_frame_count": 10,
            "classification_window_count": 0,
        },
    }

    markdown = render_instrument_benchmark_markdown(report)

    assert "结果：PASS" in markdown
    assert "Real-time factor：0.500" in markdown
    assert "峰值 RSS：unavailable" in markdown


def test_current_process_rss_probe_is_non_negative_when_available():
    result = current_process_rss_bytes()

    if sys.platform == "win32":
        assert result is not None
        assert result > 0
    else:
        assert result is None or result >= 0
