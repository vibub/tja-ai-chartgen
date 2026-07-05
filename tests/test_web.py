import time

from fastapi.testclient import TestClient

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.tja.model import ChartBar
from tja_ai_chartgen.tja.writer import TJA_FILE_ENCODING
from tja_ai_chartgen.web import create_app


def test_web_index_shows_upload_form(tmp_path):
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert "上传并分析" in response.text
    assert "data-analyze-form" in response.text
    assert "交换歌名和歌手" in response.text
    assert "直接播放 TJA" in response.text
    assert "multipart/form-data" in response.text
    assert 'accept="audio/*,video/mp4,.mp4,.m4s"' in response.text
    assert "m4s" in response.text
    assert 'name="use_beatnet" type="checkbox" value="true" checked' in response.text
    assert '<select name="course">' in response.text
    assert '魔王（Oni）' in response.text
    assert 'name="use_ai" type="checkbox" value="true" data-role="ai-toggle" checked' in response.text
    assert '<details class="advanced-panel field-wide" data-role="ai-options">' in response.text
    assert 'name="ai_model"' in response.text


def test_web_analyze_upload_opens_game_preview(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1", "bpm": "180", "offset": "0.25"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert "正在制谱" in response.text
    assert "data-progress-page" in response.text
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert status["status"] == "done"
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert result.status_code == 200
    assert "游玩预览" in result.text
    assert "data-game-preview" in result.text
    assert 'name="start_bar" type="number" min="1" value="1"' in result.text
    assert 'name="end_bar" type="number" min="1" value="1"' in result.text
    assert '<select name="course">' in result.text
    assert '魔王（Oni）' in result.text
    assert '技巧（technical）' in result.text
    assert 'name="special_notes" type="checkbox" value="true" checked' in result.text
    assert 'name="use_ai" type="checkbox" value="true" checked' in result.text
    assert '<details class="advanced-panel field-wide">' in result.text
    assert '<summary>AI 参数</summary>' in result.text
    assert '<details class="advanced-panel field-wide" open>' not in result.text
    assert "BPM: 180.0" not in result.text
    assert "OFFSET: 0.25" not in result.text
    assert "小节预览" not in result.text
    assert (job_dir / "analysis.json").exists()
    assert (job_dir / "preview.tja").exists()


def test_web_export_chart_saves_ogg_and_tja_with_matching_names(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path / "jobs"))
    analyze_response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert analyze_response.status_code == 200
    job_id = next((tmp_path / "jobs").iterdir()).name
    _wait_for_job_done(client, job_id)
    analyze_result = client.get(f"/jobs/{job_id}/result")
    assert "保存 OGG 和 TJA" in analyze_result.text
    export_dir = tmp_path / "exported"

    response = client.post(
        "/export-chart",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
        },
    )

    assert response.status_code == 200
    assert "已保存" in response.text
    assert (export_dir / "song.ogg").read_bytes() == b"fake ogg"
    assert (export_dir / "song.tja").exists()
    assert "WAVE:song.ogg" in (export_dir / "song.tja").read_text(encoding=TJA_FILE_ENCODING)


def test_web_progress_status_reports_failed_stage(tmp_path, monkeypatch):
    def fake_convert_to_ogg(input_path, output_path):
        raise RuntimeError("ffmpeg missing")

    monkeypatch.setattr("tja_ai_chartgen.web.convert_to_ogg", fake_convert_to_ogg)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert "正在制谱" in response.text
    job_id = next(tmp_path.iterdir()).name
    status = _wait_for_job_done(client, job_id, final_status="error")
    assert status["status"] == "error"
    assert status["step"] == "convert"
    assert "ffmpeg missing" in status["error"]


def test_web_regenerate_selected_bars(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch, duration=4.0)
    client = TestClient(create_app(output_dir=tmp_path))
    analyze_response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "2"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert analyze_response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    _wait_for_job_done(client, job_id)
    analyze_result = client.get(f"/jobs/{job_id}/result")
    assert 'name="start_bar" type="number" min="1" value="1"' in analyze_result.text
    assert 'name="end_bar" type="number" min="1" value="2"' in analyze_result.text

    response = client.post(
        "/regenerate",
        data={
            "job_id": job_id,
            "start_bar": "1",
            "end_bar": "2",
            "course": "Oni",
            "level": "10",
            "style": "performance",
            "density": "high",
            "special_notes": "true",
        },
    )

    assert response.status_code == 200
    assert "Regenerated bars" in response.text
    assert "游玩预览" in response.text
    assert "谱面时间线" in response.text
    assert "data-game-preview" in response.text
    assert "TJA preview" not in response.text
    assert (tmp_path / job_id / "regenerated_1_2.tja").exists()


def test_web_analyze_can_use_ai_for_full_chart(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch, duration=4.0)

    def fake_generate_chart_bars_with_ai(
        analysis,
        course,
        level,
        style,
        density,
        model,
        *,
        api_base=None,
        api_key=None,
        max_repair_attempts=2,
        special_notes=False,
    ):
        assert course == "Hard"
        assert level == 8
        assert style == "performance"
        assert density == "max"
        assert model == "openai/test-model"
        assert api_base == "https://llm.example.com/v1"
        assert api_key == "secret-key"
        assert max_repair_attempts == 1
        assert special_notes is True
        assert [bar.index for bar in analysis.bars] == [0, 1]
        return [
            ChartBar(index=0, notes="111100000000", time_signature="3/4"),
            ChartBar(index=1, notes="222200000000", time_signature="3/4"),
        ], {"final": {"bars": []}, "api_key_provided": True}

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={
            "title": "Song Title",
            "max_bars": "2",
            "time_signature": "3/4",
            "course": "Hard",
            "level": "8",
            "style": "performance",
            "density": "max",
            "special_notes": "true",
            "use_ai": "true",
            "ai_model": "openai/test-model",
            "ai_base_url": "https://llm.example.com/v1",
            "ai_api_key": "secret-key",
            "ai_repair_retries": "1",
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert "正在制谱" in response.text
    job_dir = next(tmp_path.iterdir())
    _wait_for_job_done(client, job_dir.name)
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert "AI 增强已完成" in result.text
    assert (job_dir / "ai_input_1_2.json").exists()
    assert (job_dir / "ai_output_1_2.json").exists()
    generated_text = (job_dir / "preview.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "COURSE:Hard" in generated_text
    assert "LEVEL:8" in generated_text
    assert "111100000000," in generated_text
    assert "222200000000," in generated_text



def test_web_regenerate_can_use_ai_enhancement(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch, duration=4.0)

    def fake_generate_chart_bars_with_ai(
        analysis,
        course,
        level,
        style,
        density,
        model,
        *,
        api_base=None,
        api_key=None,
        max_repair_attempts=2,
        special_notes=False,
    ):
        assert course == "Oni"
        assert level == 10
        assert style == "hybrid"
        assert density == "high"
        assert model == "openai/test-model"
        assert api_base == "https://llm.example.com/v1"
        assert api_key == "secret-key"
        assert max_repair_attempts == 1
        assert special_notes is True
        assert len(analysis.bars) == 2
        return [
            ChartBar(index=0, notes="1111000000000000", time_signature="4/4"),
            ChartBar(index=1, notes="2222000000000000", time_signature="4/4"),
        ], {"final": {"bars": []}, "api_key_provided": True}

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )
    client = TestClient(create_app(output_dir=tmp_path))
    analyze_response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "2"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert analyze_response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    _wait_for_job_done(client, job_id)

    response = client.post(
        "/regenerate",
        data={
            "job_id": job_id,
            "start_bar": "1",
            "end_bar": "2",
            "course": "Oni",
            "level": "10",
            "style": "hybrid",
            "density": "high",
            "special_notes": "true",
            "use_ai": "true",
            "ai_model": "openai/test-model",
            "ai_base_url": "https://llm.example.com/v1",
            "ai_api_key": "secret-key",
            "ai_repair_retries": "1",
        },
    )

    assert response.status_code == 200
    assert "AI 增强已完成" in response.text
    job_dir = tmp_path / job_id
    assert (job_dir / "ai_input_1_2.json").exists()
    assert (job_dir / "ai_output_1_2.json").exists()
    generated_text = (job_dir / "regenerated_1_2.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "1111000000000000," in generated_text
    assert "2222000000000000," in generated_text


def test_web_regenerate_shows_ai_option(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    _wait_for_job_done(client, job_id)
    result = client.get(f"/jobs/{job_id}/result")
    assert "使用 AI 增强" in result.text
    assert "name=\"ai_model\"" in result.text


def test_web_save_chart_edits(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch, duration=4.0)
    client = TestClient(create_app(output_dir=tmp_path))
    analyze_response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "2"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert analyze_response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    _wait_for_job_done(client, job_id)

    response = client.post(
        "/save-chart",
        data={
            "job_id": job_id,
            "course": "Oni",
            "level": "10",
            "bar_count": "2",
            "bar_index_0": "0",
            "time_signature_0": "4/4",
            "grids_per_bar_0": "16",
            "notes_0": "1000000000000000",
            "balloon_counts_0": "",
            "bar_index_1": "1",
            "time_signature_1": "4/4",
            "grids_per_bar_1": "16",
            "notes_1": "7000000080000000",
            "balloon_counts_1": "8",
        },
    )

    assert response.status_code == 200
    assert "Saved chart edits" in response.text
    assert "游玩预览" in response.text
    assert "BALLOON:8" not in response.text
    edited_text = (tmp_path / job_id / "edited_1_2.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "BALLOON:8" in edited_text
    assert edited_text.count("1000000000000000,") == 1


def test_web_preview_tja_upload(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))
    tja_text = """TITLE:Debug Song
BPM:120
OFFSET:-0.5
COURSE:Oni
LEVEL:10

#START
1000200030004000,
#END
"""

    response = client.post(
        "/preview-tja",
        files={
            "tja": ("debug.tja", tja_text.encode("utf-8"), "text/plain"),
            "audio": ("song.ogg", b"fake ogg", "audio/ogg"),
        },
    )

    assert response.status_code == 200
    assert "游玩预览" in response.text
    assert "Debug Song" in response.text
    assert "0.500s" in response.text
    assert "data-game-preview" in response.text
    assert '"startTime": 0.5' in response.text
    assert '"time": 0.5, "type": "1"' in response.text
    assert '"time": 1.0, "type": "2"' in response.text
    assert "taiko_don_16bit_44100.wav" in response.text
    assert "taiko_ka_16bit_44100.wav" in response.text
    assert (next(tmp_path.iterdir()) / "debug.tja").exists()


def test_web_serves_taiko_hit_sounds(tmp_path):
    client = TestClient(create_app(output_dir=tmp_path))

    don_response = client.get("/assets/taiko_don_16bit_44100.wav")
    ka_response = client.get("/assets/taiko_ka_16bit_44100.wav")

    assert don_response.status_code == 200
    assert ka_response.status_code == 200
    assert don_response.content.startswith(b"RIFF")
    assert ka_response.content.startswith(b"RIFF")


def test_web_serves_job_audio(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))
    analyze_response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert analyze_response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    _wait_for_job_done(client, job_id)

    response = client.get(f"/jobs/{job_id}/song.ogg")

    assert response.status_code == 200
    assert response.content == b"fake ogg"


def _wait_for_job_done(client, job_id, *, final_status="done"):
    deadline = time.monotonic() + 3
    last_status = None
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}/status")
        assert response.status_code == 200
        last_status = response.json()
        if last_status["status"] == final_status:
            return last_status
        if last_status["status"] == "error" and final_status != "error":
            raise AssertionError(last_status)
        time.sleep(0.02)
    raise AssertionError(last_status)


def _patch_web_audio_pipeline(monkeypatch, duration=2.0):
    def fake_convert_to_ogg(input_path, output_path):
        output_path.write_bytes(b"fake ogg")
        return output_path

    def fake_analyze_audio(input_path, use_beatnet=False):
        beat_times = [index * 0.5 for index in range(int(duration / 0.5) + 1)]
        return AudioAnalysisRaw(
            bpm=120,
            beat_times=beat_times,
            onset_times=beat_times[:-1],
            onset_strengths=[],
            duration=duration,
            offset=0.0,
        )

    monkeypatch.setattr("tja_ai_chartgen.web.convert_to_ogg", fake_convert_to_ogg)
    monkeypatch.setattr("tja_ai_chartgen.web.analyze_audio", fake_analyze_audio)
