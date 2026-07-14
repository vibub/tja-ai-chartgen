import json
from pathlib import Path
import subprocess
import sys


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
REBUILD_SCRIPT = FIXTURE_DIR / "rebuild_click_fixtures.py"
SCHEMA_PATH = FIXTURE_DIR / "ground_truth.schema.json"
EXPECTED_AUDIO_FILES = {
    "click_4_4.wav",
    "click_4_4_leadin.wav",
    "sparse_120.wav",
    "dense_180.wav",
    "transient_noise_intro_120.wav",
    "straight_120.wav",
    "triplet_120.wav",
    "mixed_120.wav",
    "structure_build_up_120.wav",
}
REQUIRED_FIELDS = {
    "schema_version",
    "audio",
    "bpm",
    "time_signature",
    "duration",
    "first_downbeat",
    "onsets",
    "strong_onsets",
    "beats",
    "downbeats",
    "low_band_onsets",
    "high_band_onsets",
    "silent_ranges",
    "fill_ranges",
    "sections",
}


def test_fixture_ground_truth_files_follow_schema_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    event_paths = sorted(FIXTURE_DIR.glob("*.events.json"))

    assert schema["properties"]["schema_version"]["const"] == 1
    assert set(schema["required"]) == REQUIRED_FIELDS
    assert {path.name.removesuffix(".events.json") + ".wav" for path in event_paths} == (
        EXPECTED_AUDIO_FILES
    )

    for event_path in event_paths:
        ground_truth = json.loads(event_path.read_text(encoding="utf-8"))

        assert set(ground_truth) == REQUIRED_FIELDS
        assert ground_truth["schema_version"] == 1
        assert ground_truth["audio"] in EXPECTED_AUDIO_FILES
        assert (FIXTURE_DIR / ground_truth["audio"]).is_file()
        assert ground_truth["bpm"] > 0
        assert ground_truth["time_signature"] in {"4/4", "3/4", "6/8"}
        assert ground_truth["duration"] > ground_truth["first_downbeat"] >= 0
        assert ground_truth["onsets"] == sorted(ground_truth["onsets"])
        assert ground_truth["beats"] == sorted(ground_truth["beats"])
        assert ground_truth["downbeats"] == sorted(ground_truth["downbeats"])
        assert ground_truth["first_downbeat"] == ground_truth["downbeats"][0]
        assert set(ground_truth["strong_onsets"]).issubset(ground_truth["onsets"])
        assert set(ground_truth["downbeats"]).issubset(ground_truth["onsets"])

        for field in ("silent_ranges", "fill_ranges"):
            assert all(
                0 <= start < end <= ground_truth["duration"]
                for start, end in ground_truth[field]
            )
        assert all(
            0 <= section["start"] < section["end"] <= ground_truth["duration"]
            and section["role"]
            for section in ground_truth["sections"]
        )


def test_rebuild_script_regenerates_audio_and_ground_truth_deterministically(tmp_path):
    subprocess.run(
        [sys.executable, str(REBUILD_SCRIPT), "--output-dir", str(tmp_path)],
        check=True,
    )

    expected_names = EXPECTED_AUDIO_FILES | {
        "ground_truth.schema.json",
        *(f"{Path(filename).stem}.events.json" for filename in EXPECTED_AUDIO_FILES),
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names

    for filename in expected_names:
        assert (tmp_path / filename).read_bytes() == (FIXTURE_DIR / filename).read_bytes()
