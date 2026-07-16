from pathlib import Path

from typer.testing import CliRunner

from tja_ai_chartgen.audio.instrument_models import InstrumentModelError, InstrumentModelManifest
from tja_ai_chartgen.cli import app


runner = CliRunner()


def test_prepare_instrument_models_defaults_to_recommended_stem_role(tmp_path, monkeypatch):
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

    result = runner.invoke(app, ["prepare-instrument-models"])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("resolve", None, {"profile": "stem-role"}),
        ("prepare", target, {"profile": "stem-role"}),
    ]
    assert "recommended stem-role rhythm enhancement" in result.output
    compact_output = "".join(result.output.split())
    assert str(target) in compact_output
    assert "htdemucs" in result.output
    assert "MIT/ast" not in result.output


def test_prepare_instrument_models_accepts_explicit_directory(tmp_path, monkeypatch):
    target = tmp_path / "custom"
    calls: list[Path] = []
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.prepare_local_instrument_models",
        lambda value, **_kwargs: calls.append(value)
        or InstrumentModelManifest(
            feature_version="stem-role-v1",
            profile="stem-role",
            demucs_revision="test",
            classifier_model=None,
            classifier_revision=None,
        ),
    )

    result = runner.invoke(
        app,
        ["prepare-instrument-models", "--model-dir", str(target)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [target.resolve()]


def test_prepare_instrument_models_marks_full_profile_as_legacy(tmp_path, monkeypatch):
    target = tmp_path / "models" / "instrument-v1"
    calls = []
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.resolve_instrument_model_dir",
        lambda value, **kwargs: calls.append(("resolve", value, kwargs)) or target,
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.cli.prepare_local_instrument_models",
        lambda value, **kwargs: calls.append(("prepare", value, kwargs))
        or InstrumentModelManifest(demucs_revision="test"),
    )

    result = runner.invoke(app, ["prepare-instrument-models", "--profile", "full"])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("resolve", None, {"profile": "full"}),
        ("prepare", target, {"profile": "full"}),
    ]
    assert "legacy full taxonomy diagnostics" in result.output
    assert "AST taxonomy is diagnostic-only" in result.output
    assert "MIT/ast" in result.output


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
