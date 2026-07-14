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
    "syncopated_120.wav",
    "pickup_120.wav",
    "band_attacks_120.wav",
    "harmonic_sparse_120.wav",
    "fill_burst_120.wav",
    "meter_3_4_120.wav",
    "meter_6_8_120.wav",
    "tempo_ambiguity_120.wav",
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
        assert ground_truth["strong_onsets"] == sorted(ground_truth["strong_onsets"])
        assert set(ground_truth["strong_onsets"]).issubset(ground_truth["onsets"])
        assert set(ground_truth["downbeats"]).issubset(ground_truth["onsets"])
        assert set(ground_truth["low_band_onsets"]).issubset(ground_truth["onsets"])
        assert set(ground_truth["high_band_onsets"]).issubset(ground_truth["onsets"])

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


def _load_ground_truth(stem: str) -> dict[str, object]:
    path = FIXTURE_DIR / f"{stem}.events.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_extended_fixture_ground_truth_covers_phase_zero_rhythm_cases():
    syncopated = _load_ground_truth("syncopated_120")
    assert any(onset not in syncopated["beats"] for onset in syncopated["onsets"])

    pickup = _load_ground_truth("pickup_120")
    assert pickup["onsets"][0] < pickup["first_downbeat"]

    band_attacks = _load_ground_truth("band_attacks_120")
    assert band_attacks["low_band_onsets"]
    assert band_attacks["high_band_onsets"]
    assert set(band_attacks["low_band_onsets"]).isdisjoint(
        band_attacks["high_band_onsets"]
    )

    harmonic_sparse = _load_ground_truth("harmonic_sparse_120")
    assert harmonic_sparse["sections"] == [
        {"start": 0.5, "end": 16.5, "role": "stable"}
    ]
    assert len(harmonic_sparse["onsets"]) < len(harmonic_sparse["beats"])

    fill_burst = _load_ground_truth("fill_burst_120")
    assert len(fill_burst["fill_ranges"]) == 2
    assert all(
        any(start <= onset < end for onset in fill_burst["onsets"])
        for start, end in fill_burst["fill_ranges"]
    )

    three_four = _load_ground_truth("meter_3_4_120")
    six_eight = _load_ground_truth("meter_6_8_120")
    assert three_four["time_signature"] == "3/4"
    assert six_eight["time_signature"] == "6/8"
    assert len(three_four["beats"]) == len(six_eight["beats"]) == 24
    assert len(six_eight["onsets"]) == 2 * len(six_eight["beats"])

    tempo_ambiguity = _load_ground_truth("tempo_ambiguity_120")
    assert len(tempo_ambiguity["strong_onsets"]) == 2 * len(
        tempo_ambiguity["downbeats"]
    )


def test_rebuild_script_can_generate_one_selected_fixture(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(REBUILD_SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--fixture",
            "pickup_120.wav",
        ],
        check=True,
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "ground_truth.schema.json",
        "pickup_120.wav",
        "pickup_120.events.json",
    }


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
