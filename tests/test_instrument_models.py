from pathlib import Path
from types import SimpleNamespace

import pytest

from tja_ai_chartgen.audio.instrument_models import (
    AST_MODEL_ID,
    AST_MODEL_REVISION,
    DEMUCS_MODEL_NAME,
    InstrumentModelError,
    InstrumentModelManifest,
    MODEL_VALIDATION_ERROR_PREFIX,
    STEM_ROLE_FEATURE_VERSION,
    _parse_demucs_remote_files,
    _publish_staging_directory,
    _replace_path_with_retry,
    _validate_offline_loading,
    prepare_instrument_models,
    resolve_instrument_model_dir,
    resolve_project_root,
    validate_instrument_model_dir,
)
from tja_ai_chartgen.utils.paths import write_json


def _write_model_tree(path: Path) -> InstrumentModelManifest:
    demucs_dir = path / "demucs"
    ast_dir = path / "ast"
    demucs_dir.mkdir(parents=True)
    ast_dir.mkdir()
    (demucs_dir / "htdemucs.yaml").write_text("models: [955717e8]\n", encoding="utf-8")
    (demucs_dir / "955717e8-deadbeef.th").write_bytes(b"demucs")
    (ast_dir / "config.json").write_text("{}", encoding="utf-8")
    (ast_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (ast_dir / "model.safetensors").write_bytes(b"ast")
    manifest = InstrumentModelManifest(
        demucs_revision="955717e8-deadbeef.th",
        files={},
    )
    write_json(path / "instrument_models.json", manifest)
    return manifest


def test_resolve_project_root_and_default_models_directory(tmp_path):
    project = tmp_path / "project"
    nested = project / "output" / "jobs"
    (project / "src" / "tja_ai_chartgen").mkdir(parents=True)
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert resolve_project_root(nested) == project
    assert resolve_instrument_model_dir(start=nested, environ={}) == (
        project / "models" / "instrument-v1"
    )


def test_model_directory_defaults_to_stem_role_profile_directory(tmp_path):
    project = tmp_path / "project"
    (project / "src" / "tja_ai_chartgen").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert resolve_instrument_model_dir(
        start=project,
        environ={},
        profile="stem-role",
    ) == (project / "models" / STEM_ROLE_FEATURE_VERSION)


def test_model_directory_explicit_value_precedes_environment(tmp_path):
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"

    result = resolve_instrument_model_dir(
        explicit,
        start=tmp_path,
        environ={"TJA_AI_CHARTGEN_MODEL_DIR": str(configured)},
    )

    assert result == explicit.resolve()


def test_model_directory_reads_environment_override(tmp_path):
    configured = tmp_path / "configured"

    result = resolve_instrument_model_dir(
        start=tmp_path,
        environ={"TJA_AI_CHARTGEN_MODEL_DIR": str(configured)},
    )

    assert result == configured.resolve()


def test_validate_instrument_model_dir_accepts_complete_local_tree(tmp_path):
    expected = _write_model_tree(tmp_path)

    result = validate_instrument_model_dir(tmp_path)

    assert result == expected
    assert result.demucs_model == DEMUCS_MODEL_NAME
    assert result.classifier_model == AST_MODEL_ID
    assert result.classifier_revision == AST_MODEL_REVISION


def test_validate_instrument_model_dir_accepts_stem_role_tree_without_ast(tmp_path):
    demucs_dir = tmp_path / "demucs"
    demucs_dir.mkdir()
    (demucs_dir / "htdemucs.yaml").write_text("models: [test]\n", encoding="utf-8")
    (demucs_dir / "test-deadbeef.th").write_bytes(b"demucs")
    manifest = InstrumentModelManifest(
        feature_version="stem-role-v1",
        profile="stem-role",
        demucs_revision="test-deadbeef.th",
        classifier_model=None,
        classifier_revision=None,
        files={},
    )
    write_json(tmp_path / "instrument_models.json", manifest)

    result = validate_instrument_model_dir(tmp_path, profile="stem-role")

    assert result == manifest
    assert not (tmp_path / "ast").exists()
    with pytest.raises(InstrumentModelError) as captured:
        validate_instrument_model_dir(tmp_path, profile="full")
    assert captured.value.reason == "model-profile-mismatch"


def test_validate_instrument_model_dir_reports_stable_missing_manifest(tmp_path):
    with pytest.raises(InstrumentModelError) as captured:
        validate_instrument_model_dir(tmp_path)

    assert captured.value.reason == "missing-model:manifest"
    assert str(tmp_path) not in str(captured.value)


def test_validate_instrument_model_dir_rejects_manifest_path_escape(tmp_path):
    _write_model_tree(tmp_path)
    manifest = InstrumentModelManifest(
        demucs_revision="test",
        files={"../outside.bin": "abc"},
    )
    write_json(tmp_path / "instrument_models.json", manifest)

    with pytest.raises(InstrumentModelError) as captured:
        validate_instrument_model_dir(tmp_path)

    assert captured.value.reason == "invalid-model-manifest"


def test_parse_demucs_remote_files_keeps_signature_filename_and_url():
    result = _parse_demucs_remote_files(
        '# htdemucs\nroot: "hybrid_transformer/"\n955717e8-8726e21a.th\n',
        root_url="https://models.example/",
    )

    assert result == {
        "955717e8": (
            "955717e8-8726e21a.th",
            "https://models.example/hybrid_transformer/955717e8-8726e21a.th",
        )
    }


def test_offline_model_validation_runs_in_disposable_subprocess(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tja_ai_chartgen.audio.instrument_models.subprocess.run", fake_run)

    _validate_offline_loading(tmp_path)

    assert captured["command"][-1] == str(tmp_path)
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["kwargs"]["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_offline_model_validation_preserves_stable_child_error(tmp_path, monkeypatch):
    reason = "invalid-model:ast:OSError"
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout=f"{MODEL_VALIDATION_ERROR_PREFIX}{reason}\n",
            stderr="",
        ),
    )

    with pytest.raises(InstrumentModelError) as captured:
        _validate_offline_loading(tmp_path)

    assert captured.value.reason == reason


def test_publish_staging_copies_manifest_last_when_windows_rename_stays_locked(
    tmp_path,
    monkeypatch,
):
    staging = tmp_path / ".instrument-v1.prepare-test"
    target = tmp_path / "instrument-v1"
    expected = _write_model_tree(staging)

    def fail_rename(*_args):
        raise PermissionError("locked")

    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models._replace_path_with_retry",
        fail_rename,
    )

    _publish_staging_directory(staging, target)

    assert not staging.exists()
    assert validate_instrument_model_dir(target) == expected


def test_replace_path_retries_temporary_windows_permission_error(monkeypatch):
    calls = 0

    class FlakyPath:
        def replace(self, _target):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise PermissionError("file is temporarily locked")

    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models.MODEL_PUBLISH_RETRY_DELAY_SECONDS",
        0.0,
    )

    _replace_path_with_retry(FlakyPath(), object())

    assert calls == 3


def test_replace_path_reraises_persistent_permission_error(monkeypatch):
    calls = 0

    class LockedPath:
        def replace(self, _target):
            nonlocal calls
            calls += 1
            raise PermissionError("file remains locked")

    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models.MODEL_PUBLISH_ATTEMPTS",
        2,
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models.MODEL_PUBLISH_RETRY_DELAY_SECONDS",
        0.0,
    )

    with pytest.raises(PermissionError):
        _replace_path_with_retry(LockedPath(), object())

    assert calls == 2


def test_prepare_instrument_models_replaces_target_only_after_validation(tmp_path, monkeypatch):
    target = tmp_path / "models" / "instrument-v1"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("old", encoding="utf-8")

    def fake_demucs(path: Path) -> str:
        path.mkdir(parents=True)
        (path / "htdemucs.yaml").write_text("models: [test]\n", encoding="utf-8")
        (path / "test-deadbeef.th").write_bytes(b"demucs")
        return "test-deadbeef.th"

    def fake_ast(path: Path) -> None:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"ast")

    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models._prepare_demucs",
        fake_demucs,
    )
    monkeypatch.setattr("tja_ai_chartgen.audio.instrument_models._prepare_ast", fake_ast)
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models._validate_offline_loading",
        lambda _path: None,
    )

    manifest = prepare_instrument_models(target)

    assert manifest.demucs_revision == "test-deadbeef.th"
    assert not (target / "old.txt").exists()
    assert (target / "instrument_models.json").is_file()
    assert validate_instrument_model_dir(target, verify_hashes=True) == manifest


def test_prepare_stem_role_models_skips_ast_and_records_separate_licenses(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "models" / "stem-role-v1"
    ast_called = False

    def fake_demucs(path: Path) -> str:
        path.mkdir(parents=True)
        (path / "htdemucs.yaml").write_text("models: [test]\n", encoding="utf-8")
        (path / "test-deadbeef.th").write_bytes(b"demucs")
        return "test-deadbeef.th"

    def fail_ast(_path: Path) -> None:
        nonlocal ast_called
        ast_called = True

    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models._prepare_demucs",
        fake_demucs,
    )
    monkeypatch.setattr("tja_ai_chartgen.audio.instrument_models._prepare_ast", fail_ast)
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instrument_models._validate_offline_loading",
        lambda _path: None,
    )

    manifest = prepare_instrument_models(target, profile="stem-role")

    assert ast_called is False
    assert manifest.profile == "stem-role"
    assert manifest.feature_version == "stem-role-v1"
    assert manifest.classifier_model is None
    assert manifest.code_licenses == {"demucs": "MIT"}
    assert manifest.weight_licenses == {"htdemucs": "CC-BY-NC-4.0"}
    assert validate_instrument_model_dir(
        target,
        profile="stem-role",
        verify_hashes=True,
    ) == manifest


def test_prepare_instrument_models_preserves_existing_target_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "models" / "instrument-v1"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("old", encoding="utf-8")

    def fail(_path: Path) -> str:
        raise InstrumentModelError("model-download-error:TimeoutError")

    monkeypatch.setattr("tja_ai_chartgen.audio.instrument_models._prepare_demucs", fail)

    with pytest.raises(InstrumentModelError):
        prepare_instrument_models(target)

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not list((target.parent).glob(".instrument-v1.prepare-*"))


def test_gitignore_excludes_root_models_directory():
    project_root = Path(__file__).resolve().parents[1]
    patterns = (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "models/" in patterns
