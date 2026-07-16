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
        lambda value, **_kwargs: target if value is None else value,
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.prepare_local_instrument_models",
        lambda value, **_kwargs: calls.append(value) or InstrumentModelManifest(demucs_revision="test"),
    )

    result = runner.invoke(app, ["prepare-instrument-models"])

    assert result.exit_code == 0, result.output
    assert calls == [target]
    assert "Preparing instrument models" in result.output
    compact_output = "".join(result.output.split())
    assert str(target) in compact_output
    assert "htdemucs" in result.output


def test_prepare_instrument_models_accepts_explicit_directory(tmp_path, monkeypatch):
    target = tmp_path / "custom"
    calls: list[Path] = []
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.prepare_local_instrument_models",
        lambda value, **_kwargs: calls.append(value) or InstrumentModelManifest(demucs_revision="test"),
    )

    result = runner.invoke(
        app,
        ["prepare-instrument-models", "--model-dir", str(target)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [target.resolve()]


def test_prepare_instrument_models_supports_stem_role_profile(tmp_path, monkeypatch):
    target = tmp_path / "models" / "stem-role-v1"
    calls = []
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.resolve_instrument_model_dir",
        lambda value, **kwargs: calls.append(("resolve", value, kwargs)) or target,
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.prepare_local_instrument_models",
        lambda value, **kwargs: calls.append(("prepare", value, kwargs))
        or InstrumentModelManifest(
            feature_version="stem-role-v1",
            profile="stem-role",
            demucs_revision="test",
            classifier_model=None,
            classifier_revision=None,
        ),
    )

    result = runner.invoke(app, ["prepare-instrument-models", "--profile", "stem-role"])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("resolve", None, {"profile": "stem-role"}),
        ("prepare", target, {"profile": "stem-role"}),
    ]
    assert "stem-role" in result.output
    assert "MIT/ast" not in result.output


def test_prepare_instrument_models_reports_stable_failure(tmp_path, monkeypatch):
    target = tmp_path / "models"

    def fail(_value: Path, **_kwargs):
        raise InstrumentModelError("missing-dependency:demucs")

    monkeypatch.setattr("tja_ai_chartgen.cli.prepare_local_instrument_models", fail)

    result = runner.invoke(
        app,
        ["prepare-instrument-models", "--model-dir", str(target)],
    )

    assert result.exit_code == 1
    assert "missing-dependency:demucs" in result.output
    assert "Traceback" not in result.output
