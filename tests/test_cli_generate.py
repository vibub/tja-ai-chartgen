import json

from typer.testing import CliRunner

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.cli import app
from tja_ai_chartgen.tja.model import ChartBar
from tja_ai_chartgen.tja.writer import TJA_FILE_ENCODING

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
            "迷っちゃうわ",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    tja_path = output_dir / "song.tja"
    assert tja_path.exists()
    assert "#START" in tja_path.read_text(encoding=TJA_FILE_ENCODING)
    assert "#END" in tja_path.read_text(encoding=TJA_FILE_ENCODING)
    assert tja_path.read_bytes().startswith("TITLE:迷っちゃうわ".encode(TJA_FILE_ENCODING))
    assert (output_dir / "analysis.json").exists()
    assert (output_dir / "generation_config.json").exists()
    assert (output_dir / "report.txt").exists()
    assert "Generation config:" in (output_dir / "report.txt").read_text(encoding="utf-8")


def test_generate_writes_generation_config(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch, duration=4.0)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--artist",
            "Artist Name",
            "--output-dir",
            str(output_dir),
            "--course",
            "Oni",
            "--level",
            "10",
            "--style",
            "technical",
            "--density",
            "low",
            "--max-bars",
            "2",
            "--bpm",
            "240.1234",
            "--offset",
            "0.25",
        ],
    )

    assert result.exit_code == 0, result.output
    config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert config == {
        "input_audio": str(input_audio),
        "title": "Song Title",
        "artist": "Artist Name",
        "output_dir": str(output_dir),
        "course": "Oni",
        "level": 10,
        "all_courses": False,
        "style": "technical",
        "density": "low",
        "max_bars": 2,
        "bpm_override": 240.1234,
        "offset_override": 0.25,
        "time_signature": None,
        "use_beatnet": False,
        "special_notes": False,
        "use_ai": False,
        "model": None,
        "ai_repair_retries": 2,
    }
    assert f"Generation config: {output_dir / 'generation_config.json'}" in (
        output_dir / "report.txt"
    ).read_text(encoding="utf-8")


def test_generate_with_beatnet_passes_flag_and_records_config(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"

    def fake_convert_to_ogg(input_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake ogg")
        return output_path

    def fake_analyze_audio(input_path, use_beatnet=False):
        assert use_beatnet is True
        return AudioAnalysisRaw(
            bpm=120,
            beat_times=[0, 0.5, 1.0, 1.5],
            onset_times=[0.0, 0.5, 1.0, 1.5],
            onset_strengths=[],
            duration=2.0,
            offset=0.0,
            downbeat_times=[0.0],
            beat_numbers=[1, 2, 3, 4],
            analyzer="beatnet+librosa",
        )

    monkeypatch.setattr("tja_ai_chartgen.cli.convert_to_ogg", fake_convert_to_ogg)
    monkeypatch.setattr("tja_ai_chartgen.cli.analyze_audio", fake_analyze_audio)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--use-beatnet",
        ],
    )

    assert result.exit_code == 0, result.output
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    analysis = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    assert saved_config["use_beatnet"] is True
    assert analysis["time_signature"] == "4/4"


def test_generate_with_time_signature_outputs_measure_and_records_config(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch, duration=3.0)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--time-signature",
            "3/4",
            "--max-bars",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    tja_text = (output_dir / "song.tja").read_text(encoding=TJA_FILE_ENCODING)
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    analysis = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    assert "#MEASURE 3/4" in tja_text
    assert "100010001000," in tja_text
    assert saved_config["time_signature"] == "3/4"
    assert analysis["time_signature"] == "3/4"
    assert analysis["bars"][0]["grids_per_bar"] == 12


def test_generate_from_config_replays_saved_parameters(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "input_audio": str(input_audio),
                "title": "Song Title",
                "artist": None,
                "output_dir": str(output_dir),
                "course": "Oni",
                "level": 10,
                "style": "technical",
                "density": "low",
                "max_bars": 2,
                "bpm_override": 240.1234,
                "offset_override": 0.25,
                "time_signature": "3/4",
                "use_ai": False,
                "model": None,
                "ai_base_url": "https://llm.example.com/v1",
                "ai_repair_retries": 1,
            }
        ),
        encoding="utf-8",
    )
    _patch_audio_pipeline(monkeypatch, duration=4.0)

    result = runner.invoke(app, ["generate-from-config", str(config_path)])

    assert result.exit_code == 0, result.output
    tja_text = (output_dir / "song.tja").read_text(encoding=TJA_FILE_ENCODING)
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert "BPM:240.123" in tja_text
    assert "OFFSET:0.25" in tja_text
    assert "#MEASURE 3/4" in tja_text
    assert "100010001000," in tja_text
    assert saved_config["density"] == "low"
    assert saved_config["max_bars"] == 2
    assert saved_config["time_signature"] == "3/4"
    assert "ai_base_url" not in saved_config
    assert saved_config["ai_repair_retries"] == 1


def test_generate_from_config_reports_missing_file(tmp_path):
    result = runner.invoke(app, ["generate-from-config", str(tmp_path / "missing.json")])

    assert result.exit_code == 1
    assert "Generation config not found" in result.output
    assert "Traceback" not in result.output


def test_generate_from_config_reports_invalid_json(tmp_path):
    config_path = tmp_path / "generation_config.json"
    config_path.write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["generate-from-config", str(config_path)])

    assert result.exit_code == 1
    assert "Invalid generation config JSON" in result.output
    assert "Traceback" not in result.output


def test_generate_from_config_reports_missing_required_field(tmp_path):
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(json.dumps({"title": "Song Title"}), encoding="utf-8")

    result = runner.invoke(app, ["generate-from-config", str(config_path)])

    assert result.exit_code == 1
    assert "missing required field: input_audio" in result.output
    assert "Traceback" not in result.output


def test_generate_with_max_bars_limits_output_bars(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch, duration=6.0)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--max-bars",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    tja_text = (output_dir / "song.tja").read_text(encoding=TJA_FILE_ENCODING)
    chart_lines = [line for line in tja_text.splitlines() if line.endswith(",")]
    assert len(chart_lines) == 2
    assert '"index": 1' in (output_dir / "analysis.json").read_text(encoding="utf-8")
    assert '"index": 2' not in (output_dir / "analysis.json").read_text(encoding="utf-8")


def test_generate_with_density_controls_fallback_patterns(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch, duration=4.0)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--max-bars",
            "2",
            "--density",
            "low",
        ],
    )

    assert result.exit_code == 0, result.output
    tja_text = (output_dir / "song.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "1000100010001000," in tja_text
    assert "1000200010002000," in tja_text


def test_generate_with_special_notes_outputs_balloon_header(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch, duration=16.0)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--density",
            "high",
            "--special-notes",
            "--max-bars",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    tja_text = (output_dir / "song.tja").read_text(encoding=TJA_FILE_ENCODING)
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert "5000000080000000," in tja_text
    assert "BALLOON:8" in tja_text
    assert "7000000080000000," in tja_text
    assert saved_config["special_notes"] is True


def test_generate_all_courses_writes_four_tja_files(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch, duration=4.0)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--all-courses",
            "--max-bars",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert saved_config["all_courses"] is True
    for course, level in [("easy", 3), ("normal", 5), ("hard", 7), ("oni", 10)]:
        tja_text = (output_dir / f"song_{course}.tja").read_text(encoding=TJA_FILE_ENCODING)
        assert f"COURSE:{course.title() if course != 'oni' else 'Oni'}" in tja_text
        assert f"LEVEL:{level}" in tja_text


def test_generate_rejects_invalid_density(tmp_path):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--density",
            "extreme",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid density: extreme" in result.output


def test_generate_rejects_invalid_style(tmp_path):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--style",
            "random",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid style: random" in result.output


def test_generate_with_bpm_and_offset_overrides_outputs_metadata(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch, duration=4.0)

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--output-dir",
            str(output_dir),
            "--max-bars",
            "2",
            "--bpm",
            "240.1234",
            "--offset",
            "0.25",
        ],
    )

    assert result.exit_code == 0, result.output
    tja_text = (output_dir / "song.tja").read_text(encoding=TJA_FILE_ENCODING)
    analysis_text = (output_dir / "analysis.json").read_text(encoding="utf-8")
    assert "BPM:240.123" in tja_text
    assert "OFFSET:0.25" in tja_text
    assert '"bpm": 240.123' in analysis_text
    assert '"offset": 0.25' in analysis_text


def test_generate_reports_missing_input_without_traceback(tmp_path):
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "generate",
            str(tmp_path / "missing.mp3"),
            "--title",
            "Missing Song",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Error: Input audio file not found" in result.output
    assert "Traceback" not in result.output
    assert (output_dir / "generation_config.json").exists()
    report_text = (output_dir / "report.txt").read_text(encoding="utf-8")
    assert "Generation config:" in report_text
    assert "Error: Input audio file not found" in report_text


def test_generate_rejects_non_positive_max_bars(tmp_path):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--max-bars",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "--max-bars must be greater than or equal to 1" in result.output


def test_generate_rejects_non_positive_bpm(tmp_path):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--bpm",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "--bpm must be greater than 0" in result.output


def test_generate_with_ai_passes_openai_compatible_options_without_saving_key(
    tmp_path,
    monkeypatch,
):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch)

    def fake_generate_chart_bars_with_ai(
        analysis,
        course,
        level,
        style,
        density,
        model,
        *,
        api_base,
        api_key,
        max_repair_attempts,
        special_notes,
    ):
        assert model == "openai/custom-model"
        assert api_base == "https://llm.example.com/v1"
        assert api_key == "secret-key"
        assert max_repair_attempts == 1
        return [ChartBar(index=0, notes="1000100010001000")], {"final": {"bars": []}}

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
            "--model",
            "openai/custom-model",
            "--ai-base-url",
            "https://llm.example.com/v1",
            "--ai-api-key",
            "secret-key",
            "--ai-repair-retries",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert saved_config["model"] == "openai/custom-model"
    assert "ai_base_url" not in saved_config
    assert saved_config["ai_repair_retries"] == 1
    assert "ai_api_key" not in saved_config
    assert "secret-key" not in (output_dir / "generation_config.json").read_text(encoding="utf-8")


def test_generate_with_ai_records_default_model(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch)
    monkeypatch.setattr("tja_ai_chartgen.cli.load_dotenv", lambda: None)
    monkeypatch.delenv("MODEL", raising=False)

    def fake_generate_chart_bars_with_ai(
        analysis,
        course,
        level,
        style,
        density,
        model,
        *,
        api_base,
        api_key,
        max_repair_attempts,
        special_notes,
    ):
        assert model == "openai/gpt-4o-mini"
        return [ChartBar(index=0, notes="1000100010001000")], {"final": {"bars": []}}

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
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert saved_config["model"] == "openai/gpt-4o-mini"



def test_generate_with_ai_records_model_from_environment(tmp_path, monkeypatch):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")
    output_dir = tmp_path / "output"
    _patch_audio_pipeline(monkeypatch)
    monkeypatch.setenv("MODEL", "openai/env-model")

    def fake_generate_chart_bars_with_ai(
        analysis,
        course,
        level,
        style,
        density,
        model,
        *,
        api_base,
        api_key,
        max_repair_attempts,
        special_notes,
    ):
        assert model == "openai/env-model"
        return [ChartBar(index=0, notes="1000100010001000")], {"final": {"bars": []}}

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
    saved_config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert saved_config["model"] == "openai/env-model"



def test_generate_rejects_negative_ai_repair_retries(tmp_path):
    input_audio = tmp_path / "song.mp3"
    input_audio.write_bytes(b"fake audio")

    result = runner.invoke(
        app,
        [
            "generate",
            str(input_audio),
            "--title",
            "Song Title",
            "--use-ai",
            "--ai-repair-retries",
            "-1",
        ],
    )

    assert result.exit_code == 1
    assert "--ai-repair-retries must be greater than or equal to 0" in result.output


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


def _patch_audio_pipeline(monkeypatch, duration=2.0):
    def fake_convert_to_ogg(input_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake ogg")
        return output_path

    def fake_analyze_audio(input_path, use_beatnet=False):
        return AudioAnalysisRaw(
            bpm=120,
            beat_times=[0, 0.5, 1.0, 1.5, 2.0],
            onset_times=[0.0, 0.5, 1.0, 1.5],
            onset_strengths=[],
            duration=duration,
            offset=0.0,
        )

    monkeypatch.setattr("tja_ai_chartgen.cli.convert_to_ogg", fake_convert_to_ogg)
    monkeypatch.setattr("tja_ai_chartgen.cli.analyze_audio", fake_analyze_audio)
