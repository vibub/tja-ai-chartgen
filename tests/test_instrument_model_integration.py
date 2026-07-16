from pathlib import Path

import pytest

from tja_ai_chartgen.audio.instrument_models import (
    AST_MODEL_ID,
    DEMUCS_MODEL_NAME,
    InstrumentModelError,
    resolve_instrument_model_dir,
    validate_instrument_model_dir,
)
from tja_ai_chartgen.audio.instruments import analyze_instruments
from tja_ai_chartgen.evaluation.instrument_benchmark import run_instrument_benchmark


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"


@pytest.mark.instrument_model
def test_real_local_demucs_and_ast_models_smoke():
    model_dir = resolve_instrument_model_dir()
    try:
        validate_instrument_model_dir(model_dir)
    except InstrumentModelError as error:
        pytest.skip(f"prepared instrument models are required: {error.reason}")

    pytest.importorskip("torch")
    pytest.importorskip("demucs")
    pytest.importorskip("transformers")

    audio_path = FIXTURE_DIR / "click_4_4.wav"
    assert audio_path.is_file(), f"Missing golden audio fixture: {audio_path}"

    result = analyze_instruments(
        audio_path,
        model_dir=model_dir,
        device="cpu",
        analysis_sample_rate=22_050,
        analysis_hop_length=512,
        max_duration=4.0,
    )

    assert result.status == "complete", result.reason
    assert result.reason is None
    assert result.demucs_model == DEMUCS_MODEL_NAME
    assert result.classifier_model == AST_MODEL_ID
    assert result.device == "cpu"
    assert result.analyzed_duration == pytest.approx(4.0, abs=0.1)
    assert result.stem_frames
    assert result.classification_windows
    assert result.classification_windows[0].start_time == 0.0
    assert result.classification_windows[0].end_time > 0.0


@pytest.mark.instrument_model
def test_real_local_stem_role_model_offline_cost_smoke():
    model_dir = resolve_instrument_model_dir(profile="stem-role")
    try:
        validate_instrument_model_dir(model_dir, profile="stem-role", verify_hashes=True)
    except InstrumentModelError as error:
        pytest.skip(f"prepared stem-role models are required: {error.reason}")

    pytest.importorskip("torch")
    pytest.importorskip("demucs")

    audio_path = FIXTURE_DIR / "click_4_4.wav"
    assert audio_path.is_file(), f"Missing golden audio fixture: {audio_path}"

    report = run_instrument_benchmark(
        audio_path,
        model_dir=model_dir,
        profile="stem-role",
        device="cpu",
        max_duration=4.0,
    )

    assert report["passed"] is True, report["analysis"]["reason"]
    assert report["offline"]["network_blocked"] is True
    assert report["model"]["hashes_verified"] is True
    assert report["model"]["component_bytes"]["ast"] == 0
    assert report["runtime"]["resolved_device"] == "cpu"
    assert report["runtime"]["elapsed_seconds"] > 0.0
    assert report["runtime"]["realtime_factor"] > 0.0
    assert report["analysis"]["feature_version"] == "stem-role-v1"
    assert report["analysis"]["stem_frame_count"] > 0
    assert report["analysis"]["classification_window_count"] == 0
