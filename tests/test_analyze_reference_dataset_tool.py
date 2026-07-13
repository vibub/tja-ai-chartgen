import json
from pathlib import Path
import sys

import pytest

from tools import analyze_reference_dataset


VALID_TJA = """\
TITLE:Reference Song
BPM:120
WAVE:sample.ogg
COURSE:Oni
LEVEL:7
#START
1000,
#END
"""


def _write_reference_pair(directory: Path) -> None:
    (directory / "sample.tja").write_text(VALID_TJA, encoding="utf-8")
    (directory / "sample.ogg").write_bytes(b"reference-audio-placeholder")


def test_analyze_reference_directory_requires_same_stem_audio(tmp_path: Path):
    (tmp_path / "missing-audio.tja").write_text(VALID_TJA, encoding="utf-8")

    with pytest.raises(ValueError, match="missing-audio.tja"):
        analyze_reference_dataset.analyze_reference_directory(tmp_path)


def test_main_writes_benchmark_with_atomic_replace(tmp_path: Path, monkeypatch):
    _write_reference_pair(tmp_path)
    output_path = tmp_path / "benchmark.json"
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = analyze_reference_dataset.os.replace

    def track_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(analyze_reference_dataset.os, "replace", track_replace)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_reference_dataset.py",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    assert analyze_reference_dataset.main() == 0

    benchmark = json.loads(output_path.read_text(encoding="utf-8"))
    assert benchmark["file_count"] == 1
    assert benchmark["paired_audio_count"] == 1
    assert len(replace_calls) == 1
    temporary_path, replaced_path = replace_calls[0]
    assert temporary_path.parent == output_path.parent
    assert temporary_path.name.startswith(f".{output_path.name}.")
    assert temporary_path.suffix == ".tmp"
    assert replaced_path == output_path
    assert not temporary_path.exists()


def test_main_does_not_overwrite_existing_output_when_analysis_fails(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "missing-audio.tja").write_text(VALID_TJA, encoding="utf-8")
    output_path = tmp_path / "benchmark.json"
    output_path.write_text("existing-result\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_reference_dataset.py",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    assert analyze_reference_dataset.main() == 1
    assert output_path.read_text(encoding="utf-8") == "existing-result\n"
