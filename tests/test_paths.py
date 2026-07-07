import json
from pathlib import Path

import pytest

from tja_ai_chartgen.utils.paths import write_json


def test_write_json_preserves_existing_file_when_write_is_interrupted(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    original = {"status": "running", "step": "upload"}
    path.write_text(json.dumps(original), encoding="utf-8")

    def interrupted_write_text(self, data, encoding=None):
        self.write_bytes(b"")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(Path, "write_text", interrupted_write_text)

    with pytest.raises(OSError, match="simulated interrupted write"):
        write_json(path, {"status": "done"})

    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_write_json_retries_when_atomic_replace_is_temporarily_blocked(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    original_replace = Path.replace
    attempts = {"count": 0}

    def temporarily_blocked_replace(self, target):
        if Path(target) == path and attempts["count"] == 0:
            attempts["count"] += 1
            raise PermissionError("simulated locked target")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", temporarily_blocked_replace)

    write_json(path, {"status": "done"})

    assert attempts["count"] == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "done"}
