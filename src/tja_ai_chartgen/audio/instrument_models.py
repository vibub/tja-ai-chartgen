from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from tja_ai_chartgen.utils.paths import write_json


INSTRUMENT_FEATURE_VERSION = "instrument-v1"
DEMUCS_MODEL_NAME = "htdemucs"
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
MODEL_DIR_ENV = "TJA_AI_CHARTGEN_MODEL_DIR"
MODEL_MANIFEST_NAME = "instrument_models.json"
MODEL_PUBLISH_ATTEMPTS = 6
MODEL_PUBLISH_RETRY_DELAY_SECONDS = 0.25
MODEL_VALIDATION_TIMEOUT_SECONDS = 300
MODEL_VALIDATION_ERROR_PREFIX = "instrument-model-validation-error:"


class InstrumentModelManifest(BaseModel):
    schema_version: Literal[1] = 1
    feature_version: Literal["instrument-v1"] = INSTRUMENT_FEATURE_VERSION
    demucs_model: str = DEMUCS_MODEL_NAME
    demucs_revision: str
    classifier_model: str = AST_MODEL_ID
    classifier_revision: str = AST_MODEL_REVISION
    licenses: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)


class InstrumentModelError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "tja_ai_chartgen"
        ).is_dir():
            return candidate
    return current


def resolve_instrument_model_dir(
    explicit: Path | None = None,
    *,
    start: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    values = os.environ if environ is None else environ
    configured = values.get(MODEL_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return resolve_project_root(start) / "models" / INSTRUMENT_FEATURE_VERSION


def load_instrument_model_manifest(model_dir: Path) -> InstrumentModelManifest:
    manifest_path = model_dir / MODEL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise InstrumentModelError("missing-model:manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = InstrumentModelManifest.model_validate(payload)
    except Exception as error:  # noqa: BLE001 - public boundary returns a stable reason.
        raise InstrumentModelError("invalid-model-manifest") from error
    if manifest.demucs_model != DEMUCS_MODEL_NAME:
        raise InstrumentModelError("invalid-model-manifest")
    if manifest.classifier_model != AST_MODEL_ID:
        raise InstrumentModelError("invalid-model-manifest")
    return manifest


def validate_instrument_model_dir(
    model_dir: Path,
    *,
    verify_hashes: bool = False,
) -> InstrumentModelManifest:
    manifest = load_instrument_model_manifest(model_dir)
    demucs_dir = model_dir / "demucs"
    ast_dir = model_dir / "ast"
    if not demucs_dir.is_dir() or not (demucs_dir / f"{DEMUCS_MODEL_NAME}.yaml").is_file():
        raise InstrumentModelError("missing-model:htdemucs")
    if not any(demucs_dir.glob("*.th")):
        raise InstrumentModelError("missing-model:htdemucs")
    if not ast_dir.is_dir() or not (ast_dir / "config.json").is_file():
        raise InstrumentModelError("missing-model:ast")
    if not (ast_dir / "preprocessor_config.json").is_file():
        raise InstrumentModelError("missing-model:ast")
    if not any((ast_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise InstrumentModelError("missing-model:ast")

    for relative, expected_hash in manifest.files.items():
        candidate = _manifest_file(model_dir, relative)
        if not candidate.is_file():
            raise InstrumentModelError("missing-model:file")
        if verify_hashes and _sha256(candidate) != expected_hash:
            raise InstrumentModelError("invalid-model-checksum")
    return manifest


def prepare_instrument_models(model_dir: Path) -> InstrumentModelManifest:
    target = model_dir.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.prepare-{uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        demucs_revision = _prepare_demucs(staging / "demucs")
        _prepare_ast(staging / "ast")
        files = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and MODEL_MANIFEST_NAME not in path.parts
        }
        manifest = InstrumentModelManifest(
            demucs_revision=demucs_revision,
            licenses={"demucs": "MIT", "ast": "BSD-3-Clause"},
            files=files,
        )
        write_json(staging / MODEL_MANIFEST_NAME, manifest)
        validate_instrument_model_dir(staging, verify_hashes=True)
        _validate_offline_loading(staging)
        try:
            _replace_directory(staging, target)
        except PermissionError as error:
            raise InstrumentModelError("model-publish-error:permission-denied") from error
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _prepare_demucs(target: Path) -> str:
    try:
        from demucs import pretrained
        import yaml
    except ImportError as error:
        raise InstrumentModelError("missing-dependency:demucs") from error

    remote_root = Path(pretrained.REMOTE_ROOT)
    bag_path = remote_root / f"{DEMUCS_MODEL_NAME}.yaml"
    files_path = remote_root / "files.txt"
    if not bag_path.is_file() or not files_path.is_file():
        raise InstrumentModelError("missing-model-source:htdemucs")
    try:
        bag = yaml.safe_load(bag_path.read_text(encoding="utf-8"))
        signatures = [str(value) for value in bag["models"]]
    except Exception as error:  # noqa: BLE001 - dependency data must be validated.
        raise InstrumentModelError("invalid-model-source:htdemucs") from error
    remote_files = _parse_demucs_remote_files(
        files_path.read_text(encoding="utf-8"),
        root_url=str(pretrained.ROOT_URL),
    )

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bag_path, target / bag_path.name)
    downloaded: list[str] = []
    for signature in signatures:
        item = remote_files.get(signature)
        if item is None:
            raise InstrumentModelError("missing-model-source:htdemucs")
        filename, url = item
        _download_file(url, target / filename)
        downloaded.append(filename)
    return ",".join(downloaded)


def _prepare_ast(target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise InstrumentModelError("missing-dependency:huggingface-hub") from error
    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=AST_MODEL_ID,
            revision=AST_MODEL_REVISION,
            local_dir=target,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "pytorch_model.bin",
            ],
        )
    except Exception as error:  # noqa: BLE001 - command boundary reports a stable reason.
        raise InstrumentModelError(f"model-download-error:{type(error).__name__}") from error


def _validate_offline_loading(model_dir: Path) -> None:
    validation_script = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "from tja_ai_chartgen.audio.instrument_models import (",
            "    InstrumentModelError,",
            "    MODEL_VALIDATION_ERROR_PREFIX,",
            "    _validate_offline_loading_in_process,",
            ")",
            "try:",
            "    _validate_offline_loading_in_process(Path(sys.argv[1]))",
            "except InstrumentModelError as error:",
            "    print(f'{MODEL_VALIDATION_ERROR_PREFIX}{error.reason}')",
            "    raise SystemExit(2)",
            "except Exception as error:",
            "    print(f'{MODEL_VALIDATION_ERROR_PREFIX}validation-process-error:'",
            "          f'{type(error).__name__}')",
            "    raise SystemExit(3)",
        ]
    )
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", validation_script, str(model_dir)],
            capture_output=True,
            text=True,
            timeout=MODEL_VALIDATION_TIMEOUT_SECONDS,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise InstrumentModelError("model-validation-timeout") from error
    except OSError as error:
        raise InstrumentModelError(
            f"model-validation-process-error:{type(error).__name__}"
        ) from error
    if result.returncode == 0:
        return
    reason = next(
        (
            line.removeprefix(MODEL_VALIDATION_ERROR_PREFIX)
            for line in result.stdout.splitlines()
            if line.startswith(MODEL_VALIDATION_ERROR_PREFIX)
        ),
        "model-validation-process-failed",
    )
    raise InstrumentModelError(reason)


def _validate_offline_loading_in_process(model_dir: Path) -> None:
    try:
        from demucs.api import Separator
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError as error:
        module = str(getattr(error, "name", "optional"))
        raise InstrumentModelError(f"missing-dependency:{module}") from error

    separator = None
    feature_extractor = None
    classifier = None
    try:
        try:
            separator = Separator(
                model=DEMUCS_MODEL_NAME,
                repo=model_dir / "demucs",
                device="cpu",
            )
        except Exception as error:  # noqa: BLE001 - model validation must be contained.
            raise InstrumentModelError(
                f"invalid-model:htdemucs:{type(error).__name__}"
            ) from error
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_dir / "ast",
                local_files_only=True,
            )
            classifier = AutoModelForAudioClassification.from_pretrained(
                model_dir / "ast",
                local_files_only=True,
            )
        except Exception as error:  # noqa: BLE001 - model validation must be contained.
            raise InstrumentModelError(f"invalid-model:ast:{type(error).__name__}") from error
    finally:
        del classifier, feature_extractor, separator
        gc.collect()


def _parse_demucs_remote_files(
    content: str,
    *,
    root_url: str,
) -> dict[str, tuple[str, str]]:
    root = ""
    result: dict[str, tuple[str, str]] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("root:"):
            root = line.split(":", 1)[1].strip().strip('"')
            continue
        signature = line.split("-", 1)[0]
        result[signature] = (line, f"{root_url}{root}{line}")
    return result


def _download_file(url: str, target: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as error:  # noqa: BLE001 - command boundary reports a stable reason.
        raise InstrumentModelError(f"model-download-error:{type(error).__name__}") from error


def _manifest_file(model_dir: Path, relative: str) -> Path:
    candidate = (model_dir / relative).resolve()
    try:
        candidate.relative_to(model_dir.resolve())
    except ValueError as error:
        raise InstrumentModelError("invalid-model-manifest") from error
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_directory(staging: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
    had_target = target.exists()
    if had_target:
        _replace_path_with_retry(target, backup)
    try:
        _publish_staging_directory(staging, target)
    except Exception:
        if had_target and backup.exists() and not target.exists():
            _replace_path_with_retry(backup, target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _publish_staging_directory(staging: Path, target: Path) -> None:
    try:
        _replace_path_with_retry(staging, target)
        return
    except PermissionError:
        pass

    manifest_source = staging / MODEL_MANIFEST_NAME

    def ignore_manifest(directory: str, names: list[str]) -> list[str]:
        if Path(directory) == staging and MODEL_MANIFEST_NAME in names:
            return [MODEL_MANIFEST_NAME]
        return []

    try:
        shutil.copytree(staging, target, ignore=ignore_manifest)
        shutil.copy2(manifest_source, target / MODEL_MANIFEST_NAME)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)


def _replace_path_with_retry(source: Path, target: Path) -> None:
    for attempt in range(MODEL_PUBLISH_ATTEMPTS):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= MODEL_PUBLISH_ATTEMPTS:
                raise
            gc.collect()
            time.sleep(MODEL_PUBLISH_RETRY_DELAY_SECONDS * (attempt + 1))
