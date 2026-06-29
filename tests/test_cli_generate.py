from typer.testing import CliRunner

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.cli import app

runner = CliRunner()


def test_generate_without_ai_writes_outputs(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    tja_path = output_dir / "song.tja"
    assert tja_path.exists()
    assert "#START" in tja_path.read_text(encoding="utf-8")
    assert "#END" in tja_path.read_text(encoding="utf-8")
    assert (output_dir / "analysis.json").exists()
    assert (output_dir / "report.txt").exists()


def test_generate_with_ai_failure_falls_back_to_rules(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch)

    def fake_generate_chart_bars_with_ai(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--use-ai",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "AI generation failed" in result.output
    assert (output_dir / "song.tja").exists()
    assert "network unavailable" in (output_dir / "report.txt").read_text(encoding="utf-8")
    assert (output_dir / "ai_output.json").exists()


def _patch_audio_pipeline(monkeypatch):
    def fake_convert_to_ogg(input_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake ogg")
        return output_path

    def fake_analyze_audio(input_path):
        return AudioAnalysisRaw(
            bpm=120,
            beat_times=[0, 0.5, 1.0, 1.5, 2.0],
            onset_times=[0.0, 0.5, 1.0, 1.5],
            onset_strengths=[],
            duration=2.0,
            offset=0.0,
        )

    monkeypatch.setattr("tja_ai_chartgen.cli.convert_to_ogg", fake_convert_to_ogg)
    monkeypatch.setattr("tja_ai_chartgen.cli.analyze_audio", fake_analyze_audio)
