from pathlib import Path

from typer.testing import CliRunner

from tja_ai_chartgen.audio.instrument_models import InstrumentModelError, InstrumentModelManifest
from tja_ai_chartgen.cli import app


runner = CliRunner()


def test_prepare_instrument_models_uses_resolved_default_directory(tmp_path, monkeypatch):
    target = tmp_path / "models" / "instrument-v1"
    calls: list[Path] = []
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.resolve_instrument_model_dir",
        lambda value: target if value is None else value,
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.prepare_local_instrument_models",
        lambda value: calls.append(value) or InstrumentModelManifest(demucs_revision="test"),
    )

    result = runner.invoke(app, ["prepare-instrument-models"])

    assert result.exit_code == 0, result.output
    assert calls == [target]
    assert "Preparing instrument models" in result.output
    assert "instrument-v1" in result.output
    assert "htdemucs" in result.output


def test_prepare_instrument_models_accepts_explicit_directory(tmp_path, monkeypatch):
    target = tmp_path / "custom"
    calls: list[Path] = []
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.prepare_local_instrument_models",
        lambda value: calls.append(value) or InstrumentModelManifest(demucs_revision="test"),
    )

    result = runner.invoke(
        app,
        ["prepare-instrument-models", "--model-dir", str(target)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [target.resolve()]


def test_prepare_instrument_models_reports_stable_failure(tmp_path, monkeypatch):
    target = tmp_path / "models"

    def fail(_value: Path):
        raise InstrumentModelError("missing-dependency:demucs")

    monkeypatch.setattr("tja_ai_chartgen.cli.prepare_local_instrument_models", fail)

    result = runner.invoke(
        app,
        ["prepare-instrument-models", "--model-dir", str(target)],
    )

    assert result.exit_code == 1
    assert "missing-dependency:demucs" in result.output
    assert "Traceback" not in result.output
