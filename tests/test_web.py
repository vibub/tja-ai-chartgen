from fastapi.testclient import TestClient

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
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


def test_web_analyze_upload_opens_game_preview(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1", "bpm": "180", "offset": "0.25"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert "游玩预览" in response.text
    assert "data-game-preview" in response.text
    assert "BPM: 180.0" not in response.text
    assert "OFFSET: 0.25" not in response.text
    assert "小节预览" not in response.text
    job_dir = next(tmp_path.iterdir())
    assert (job_dir / "analysis.json").exists()
    assert (job_dir / "preview.tja").exists()


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
    edited_text = (tmp_path / job_id / "edited_1_2.tja").read_text(encoding="utf-8")
    assert "BALLOON:8" in edited_text
    assert edited_text.count("1000000000000000,") == 1


def test_web_preview_tja_upload(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))
    tja_text = """TITLE:Debug Song
BPM:120
OFFSET:0.5
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
    assert "data-game-preview" in response.text
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

    response = client.get(f"/jobs/{job_id}/song.ogg")

    assert response.status_code == 200
    assert response.content == b"fake ogg"


def _patch_web_audio_pipeline(monkeypatch, duration=2.0):
    def fake_convert_to_ogg(input_path, output_path):
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

    monkeypatch.setattr("tja_ai_chartgen.web.convert_to_ogg", fake_convert_to_ogg)
    monkeypatch.setattr("tja_ai_chartgen.web.analyze_audio", fake_analyze_audio)
