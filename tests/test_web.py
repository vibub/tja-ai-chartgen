from fastapi.testclient import TestClient

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.web import create_app


def test_web_index_shows_upload_form(tmp_path):
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert "Upload and analyze" in response.text
    assert "multipart/form-data" in response.text


def test_web_analyze_upload_previews_bars(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1", "bpm": "180", "offset": "0.25"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert "Analysis preview" in response.text
    assert "BPM: 180.0" in response.text
    assert "OFFSET: 0.25" in response.text
    assert "Regenerate selected bars" in response.text
    assert (next(tmp_path.iterdir()) / "analysis.json").exists()


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
    assert "COURSE:Oni" in response.text
    assert (tmp_path / job_id / "regenerated_1_2.tja").exists()


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
