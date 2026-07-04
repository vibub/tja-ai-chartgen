from pathlib import Path

from tja_ai_chartgen.ai.examples import build_reference_examples
from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw


def test_build_reference_examples_pairs_tja_and_processed_audio(tmp_path, monkeypatch):
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    tja_path = examples_dir / "song.tja"
    audio_path = examples_dir / "song.mp3"
    tja_path.write_text(
        """TITLE:Reference Song
SUBTITLE:--Reference Artist
BPM:120
OFFSET:0
COURSE:Oni
LEVEL:8
BALLOON:8

#START
1000100010001000,
7000000080000000,
#END
""",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"fake audio")
    work_dir = tmp_path / "work"

    def fake_convert_to_ogg(input_path: Path, output_path: Path):
        assert input_path == audio_path
        assert output_path == work_dir / "song.ogg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake ogg")
        return output_path

    def fake_analyze_audio(input_path: Path, use_beatnet=False):
        assert input_path == work_dir / "song.ogg"
        assert use_beatnet is True
        return AudioAnalysisRaw(
            bpm=120,
            beat_times=[0.0, 0.5, 1.0, 1.5],
            onset_times=[0.0, 0.5, 1.0, 1.5],
            onset_strengths=[],
            duration=2.0,
            offset=0.0,
        )

    monkeypatch.setattr("tja_ai_chartgen.ai.examples.convert_to_ogg", fake_convert_to_ogg)
    monkeypatch.setattr("tja_ai_chartgen.ai.examples.analyze_audio", fake_analyze_audio)

    examples = build_reference_examples(
        examples_dir,
        max_bars_per_example=1,
        use_beatnet=True,
        work_dir=work_dir,
    )

    assert len(examples) == 1
    assert examples[0]["title"] == "Reference Song"
    assert examples[0]["artist"] == "Reference Artist"
    assert examples[0]["processed_audio"] == str(work_dir / "song.ogg")
    assert examples[0]["bars"] == [
        {
            "bar": 1,
            "audio_features": examples[0]["bars"][0]["audio_features"],
            "reference_notes": "1000100010001000",
            "reference_note_resolution": 16,
            "balloon_counts": [],
        }
    ]
    assert examples[0]["bars"][0]["audio_features"]["grids_per_bar"] == 16
