import json
import time
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from tja_ai_chartgen.ai.client import AiProviderError
from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw
from tja_ai_chartgen.audio.spectral import SpectralAnalysisRaw
from tja_ai_chartgen.tja.model import ChartBar, SongAnalysis, TempoAnalysisDecision
from tja_ai_chartgen.tja.writer import TJA_FILE_ENCODING
from tja_ai_chartgen.web import (
    WEB_UPLOAD_MAX_BYTES,
    UploadTooLargeError,
    _export_chart_files,
    _read_progress,
    _save_upload,
    create_app,
)


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
    assert 'name="use_instrument_analysis" type="checkbox" value="true"' in response.text
    assert 'select name="instrument_device"' in response.text
    assert "prepare-instrument-models" in response.text
    assert '<select name="course">' in response.text
    assert '魔王（Oni）' in response.text
    assert 'name="use_ai" type="checkbox" value="true" data-role="ai-toggle" checked' in response.text
    assert '<details class="advanced-panel field-wide" data-role="ai-options">' in response.text
    assert 'name="ai_model"' in response.text
    assert 'name="ai_request_timeout" type="number" min="1" max="600" value="300"' in response.text
    assert 'name="ai_transport_retries" type="number" min="0" max="1" value="1"' in response.text


def test_web_remote_mode_disables_instrument_analysis_without_server_opt_in(tmp_path):
    client = TestClient(create_app(output_dir=tmp_path, remote_mode=True))

    response = client.get("/")

    assert response.status_code == 200
    assert 'name="use_instrument_analysis" type="checkbox" value="true" disabled' in response.text
    assert 'select name="instrument_device" disabled' in response.text
    assert "远程模式未由服务器管理员开放重型分析" in response.text


def test_web_upload_limit_is_100_mib():
    assert WEB_UPLOAD_MAX_BYTES == 100 * 1024 * 1024


def test_save_upload_reads_in_bounded_chunks(tmp_path):
    upload = UploadFile(filename="song.mp3", file=BytesIO(b"abcdef"))

    path = _save_upload(tmp_path, upload, max_bytes=6, chunk_size=3)

    assert path.read_bytes() == b"abcdef"


def test_save_upload_removes_partial_file_when_limit_is_exceeded(tmp_path):
    upload = UploadFile(filename="song.mp3", file=BytesIO(b"12345"))

    with pytest.raises(UploadTooLargeError):
        _save_upload(tmp_path, upload, max_bytes=4, chunk_size=2)

    assert not (tmp_path / "song.mp3").exists()


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
    assert status["notices"] == []
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
    assert 'name="ai_request_timeout" type="number" min="1" max="600" value="300"' in result.text
    assert 'name="ai_transport_retries" type="number" min="0" max="1" value="1"' in result.text
    assert "BPM: 180.0" not in result.text
    assert "OFFSET: 0.25" not in result.text
    assert "小节预览" not in result.text
    analysis = json.loads((job_dir / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["analyzer"] == "librosa+manual-override"
    assert analysis["tempo_analysis"]["accepted"] is False
    assert analysis["tempo_analysis"]["selected_source"] == "librosa"
    assert analysis["tempo_analysis"]["reason"] == "insufficient_onsets"
    quality_report = json.loads((job_dir / "quality_report_1_1.json").read_text(encoding="utf-8"))
    notices = json.loads((job_dir / "generation_notices.json").read_text(encoding="utf-8"))
    assert quality_report["bar_count"] == 1
    assert notices == []
    assert (job_dir / "preview.tja").exists()


def test_web_analysis_fallback_notice_is_persisted_and_rendered(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert [notice["code"] for notice in status["notices"]] == [
        "tempo-refinement-fallback"
    ]
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert "onset-grid 节拍校正置信度不足" in result.text
    notices = json.loads((job_dir / "generation_notices.json").read_text(encoding="utf-8"))
    assert [notice["code"] for notice in notices] == ["tempo-refinement-fallback"]


def test_web_spectral_fallback_notice_is_persisted_and_rendered(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch, spectral_fallback=True)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1", "bpm": "120"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert [notice["code"] for notice in status["notices"]] == [
        "spectral-analysis-fallback"
    ]
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert "频谱语义增强未生效" in result.text
    notices = json.loads((job_dir / "generation_notices.json").read_text(encoding="utf-8"))
    assert notices[0]["detail"] == "reason=extractor-error:RuntimeError"


def test_web_instrument_analysis_success_is_persisted_and_rendered(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch, instrument_status="complete")
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={
            "title": "Song Title",
            "max_bars": "1",
            "bpm": "120",
            "use_instrument_analysis": "true",
            "instrument_device": "cpu",
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert any(
        notice["code"] == "instrument-analysis-succeeded"
        for notice in status["notices"]
    )
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert "人声与乐器分析" in result.text
    assert "状态：完整" in result.text
    analysis = json.loads((job_dir / "analysis.json").read_text(encoding="utf-8"))
    options = json.loads((job_dir / "chart_options.json").read_text(encoding="utf-8"))
    assert analysis["instrument_feature_version"] == "instrument-v1"
    assert analysis["instrument_analysis_status"] == "complete"
    assert options["use_instrument_analysis"] is True
    assert options["instrument_device"] == "cpu"


def test_web_instrument_model_fallback_notice_is_persisted_and_rendered(
    tmp_path,
    monkeypatch,
):
    _patch_web_audio_pipeline(
        monkeypatch,
        instrument_status="fallback",
        instrument_reason="missing-model:ast",
    )
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={
            "title": "Song Title",
            "max_bars": "1",
            "bpm": "120",
            "use_instrument_analysis": "true",
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert any(
        notice["code"] == "instrument-models-missing"
        for notice in status["notices"]
    )
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert "人声与乐器模型未准备完整" in result.text
    assert "状态：已降级" in result.text
    notices = json.loads((job_dir / "generation_notices.json").read_text(encoding="utf-8"))
    instrument_notice = next(
        item for item in notices if item["code"] == "instrument-models-missing"
    )
    assert instrument_notice["detail"] == "reason=missing-model:ast"


def test_web_remote_mode_rejects_instrument_analysis_without_server_opt_in(
    tmp_path,
    monkeypatch,
):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path, remote_mode=True))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "use_instrument_analysis": "true"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert "Remote instrument analysis is disabled" in response.text
    assert not list(tmp_path.iterdir())


def test_web_analyze_oversized_upload_returns_413_and_removes_job(tmp_path, monkeypatch):
    def fake_save_upload(job_dir, upload):
        (job_dir / "partial.upload").write_bytes(b"partial")
        raise UploadTooLargeError("Uploaded file exceeds the 100 MiB limit")

    monkeypatch.setattr("tja_ai_chartgen.web._save_upload", fake_save_upload)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Large"},
        files={"audio": ("large.mp3", b"12345", "audio/mpeg")},
    )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_web_preview_oversized_second_upload_removes_whole_job(tmp_path, monkeypatch):
    save_count = 0

    def fake_save_upload(job_dir, upload):
        nonlocal save_count
        save_count += 1
        path = job_dir / (upload.filename or "upload")
        path.write_bytes(b"partial")
        if save_count == 2:
            raise UploadTooLargeError("Uploaded file exceeds the 100 MiB limit")
        return path

    monkeypatch.setattr("tja_ai_chartgen.web._save_upload", fake_save_upload)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/preview-tja",
        files={
            "tja": ("debug.tja", b"TITLE:Test", "text/plain"),
            "audio": ("song.ogg", b"12345", "audio/ogg"),
        },
    )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


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


def test_web_export_conflict_shows_four_actions_and_suggested_name(tmp_path, monkeypatch):
    client, job_id = _create_export_job(tmp_path, monkeypatch)
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    (export_dir / "song.ogg").write_bytes(b"old ogg")
    (export_dir / "song.tja").write_bytes(b"old tja")

    response = client.post(
        "/export-chart",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
        },
    )

    assert response.status_code == 409
    assert "覆盖原文件" in response.text
    assert "重新命名保存" in response.text
    assert "重命名原文件并保存" in response.text
    assert "打开文件所在目录" in response.text
    assert 'name="output_stem" value="song (1)"' in response.text
    assert (export_dir / "song.ogg").read_bytes() == b"old ogg"
    assert (export_dir / "song.tja").read_bytes() == b"old tja"


def test_web_export_chart_can_overwrite_existing_pair(tmp_path, monkeypatch):
    client, job_id = _create_export_job(tmp_path, monkeypatch)
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    (export_dir / "song.ogg").write_bytes(b"old ogg")
    (export_dir / "song.tja").write_bytes(b"old tja")

    response = client.post(
        "/export-chart",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
            "conflict_action": "overwrite",
        },
    )

    assert response.status_code == 200
    assert (export_dir / "song.ogg").read_bytes() == b"fake ogg"
    assert "WAVE:song.ogg" in (export_dir / "song.tja").read_text(
        encoding=TJA_FILE_ENCODING
    )


def test_web_export_chart_can_save_with_new_name_and_updates_wave(tmp_path, monkeypatch):
    client, job_id = _create_export_job(tmp_path, monkeypatch)
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    (export_dir / "song.ogg").write_bytes(b"old ogg")
    (export_dir / "song.tja").write_bytes(b"old tja")

    response = client.post(
        "/export-chart",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
            "conflict_action": "rename",
            "output_stem": "song remix.tja",
        },
    )

    assert response.status_code == 200
    assert (export_dir / "song remix.ogg").read_bytes() == b"fake ogg"
    renamed_tja = (export_dir / "song remix.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "WAVE:song remix.ogg" in renamed_tja
    assert "WAVE:song.ogg" not in renamed_tja
    assert (export_dir / "song.ogg").read_bytes() == b"old ogg"
    assert (export_dir / "song.tja").read_bytes() == b"old tja"


def test_web_export_chart_rejects_invalid_renamed_stem(tmp_path, monkeypatch):
    client, job_id = _create_export_job(tmp_path, monkeypatch)
    export_dir = tmp_path / "exported"
    export_dir.mkdir()

    response = client.post(
        "/export-chart",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
            "conflict_action": "rename",
            "output_stem": "../outside",
        },
    )

    assert response.status_code == 400
    assert "重新命名保存" in response.text
    assert "invalid characters" in response.text
    assert not (tmp_path / "outside.ogg").exists()
    assert not (tmp_path / "outside.tja").exists()


def test_web_export_chart_can_backup_existing_pair_before_saving(tmp_path, monkeypatch):
    client, job_id = _create_export_job(tmp_path, monkeypatch)
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    (export_dir / "song.ogg").write_bytes(b"old ogg")
    (export_dir / "song.tja").write_bytes(b"old tja")
    (export_dir / "song.ogg.bak").write_bytes(b"older ogg")
    (export_dir / "song.tja.bak").write_bytes(b"older tja")

    response = client.post(
        "/export-chart",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
            "conflict_action": "backup",
        },
    )

    assert response.status_code == 200
    assert (export_dir / "song.ogg.bak.1").read_bytes() == b"old ogg"
    assert (export_dir / "song.tja.bak.1").read_bytes() == b"old tja"
    assert (export_dir / "song.ogg").read_bytes() == b"fake ogg"
    assert "WAVE:song.ogg" in (export_dir / "song.tja").read_text(
        encoding=TJA_FILE_ENCODING
    )


def test_export_backup_restores_original_pair_when_install_fails(tmp_path, monkeypatch):
    _client, job_id = _create_export_job(tmp_path, monkeypatch)
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_id
    analysis = SongAnalysis.model_validate_json(
        (job_dir / "analysis.json").read_text(encoding="utf-8")
    )
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    old_ogg = export_dir / "song.ogg"
    old_tja = export_dir / "song.tja"
    old_ogg.write_bytes(b"old ogg")
    old_tja.write_bytes(b"old tja")
    real_replace = Path.replace

    def fail_installing_tja(path, target):
        target = Path(target)
        if path.name.startswith(".song.tja.") and target == old_tja:
            raise OSError("simulated TJA install failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_installing_tja)

    with pytest.raises(OSError, match="simulated TJA install failure"):
        _export_chart_files(
            tja_path=job_dir / "preview.tja",
            analysis=analysis,
            output_dir=export_dir,
            conflict_action="backup",
        )

    assert old_ogg.read_bytes() == b"old ogg"
    assert old_tja.read_bytes() == b"old tja"
    assert not (export_dir / "song.ogg.bak").exists()
    assert not (export_dir / "song.tja.bak").exists()
    assert not list(export_dir.glob(".*.tmp"))


def test_web_export_conflict_can_open_output_directory(tmp_path, monkeypatch):
    client, job_id = _create_export_job(tmp_path, monkeypatch)
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    opened = []
    monkeypatch.setattr("tja_ai_chartgen.web._open_directory", opened.append)

    response = client.post(
        "/open-export-directory",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
        },
    )

    assert response.status_code == 200
    assert opened == [export_dir]
    assert "已打开文件所在目录" in response.text
    assert "重新命名保存" in response.text


def test_web_remote_mode_rejects_server_side_export(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    jobs_dir = tmp_path / "jobs"
    client = TestClient(create_app(output_dir=jobs_dir, remote_mode=True))
    analyze_response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert analyze_response.status_code == 200
    job_id = next(jobs_dir.iterdir()).name
    _wait_for_job_done(client, job_id)
    export_dir = tmp_path / "must-not-exist"

    response = client.post(
        "/export-chart",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
        },
    )

    assert response.status_code == 403
    open_response = client.post(
        "/open-export-directory",
        data={
            "job_id": job_id,
            "tja_filename": "preview.tja",
            "output_dir": str(export_dir),
        },
    )
    assert open_response.status_code == 403
    assert not export_dir.exists()


def test_web_progress_status_reports_failed_stage(tmp_path, monkeypatch):
    def fake_convert_to_ogg(input_path, output_path):
        raise RuntimeError("ffmpeg missing")

    monkeypatch.setattr("tja_ai_chartgen.generation.convert_to_ogg", fake_convert_to_ogg)
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


def test_web_remote_progress_hides_background_error_details(tmp_path, monkeypatch):
    private_detail = r"D:\private\audio\ffmpeg missing"

    def fake_convert_to_ogg(input_path, output_path):
        raise RuntimeError(private_detail)

    monkeypatch.setattr("tja_ai_chartgen.generation.convert_to_ogg", fake_convert_to_ogg)
    client = TestClient(create_app(output_dir=tmp_path, remote_mode=True))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    status = _wait_for_job_done(client, job_id, final_status="error")
    serialized = json.dumps(status, ensure_ascii=False)
    assert private_detail not in serialized
    assert "RuntimeError" in status["error"]
    assert "远程模式已隐藏" in status["error"]
    assert status["notices"][0]["code"] == "convert-failed"
    assert "detail" not in status["notices"][0]


def test_web_encoding_error_can_retry_metadata_without_regeneration(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "你的歌", "artist": "歌手", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    status = _wait_for_job_done(client, job_id, final_status="error")
    assert status["step"] == "render"
    assert status["can_retry_metadata"] is True
    assert status["title"] == "你的歌"
    assert "cannot be encoded with cp932" in status["error"]
    progress = client.get(f"/jobs/{job_id}/progress")
    assert "重试写入预览" in progress.text
    assert (tmp_path / job_id / "chart_bars.json").exists()

    retry = client.post(
        f"/jobs/{job_id}/retry-metadata",
        data={"title": "Song Title", "artist": "Artist"},
    )

    assert retry.status_code == 200
    assert "游玩预览" in retry.text
    assert "已复用现有分析与谱面" in retry.text
    assert "任务停止，请根据错误信息调整参数后重试" not in retry.text
    assert (tmp_path / job_id / "preview.tja").exists()
    generated_text = (tmp_path / job_id / "preview.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "TITLE:Song Title" in generated_text
    assert "SUBTITLE:-- Artist" in generated_text
    status_after_retry = client.get(f"/jobs/{job_id}/status").json()
    assert status_after_retry["status"] == "done"


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
    quality_report = json.loads(
        (tmp_path / job_id / "quality_report_1_2.json").read_text(encoding="utf-8")
    )
    assert quality_report["bar_count"] == 2


def test_web_job_file_download_uses_public_artifact_allowlist(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))
    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    _wait_for_job_done(client, job_dir.name)
    (job_dir / "ai_input_1_1.json").write_text("{}", encoding="utf-8")
    (job_dir / "ai_output_1_1.json").write_text("{}", encoding="utf-8")
    (job_dir / "ai_attempts_1_1.json").write_text("{}", encoding="utf-8")

    assert client.get(f"/jobs/{job_dir.name}/song.ogg").status_code == 200
    assert client.get(f"/jobs/{job_dir.name}/preview.tja").status_code == 200
    for filename in (
        "song.mp3",
        "analysis.json",
        "progress.json",
        "chart_bars.json",
        "chart_options.json",
        "generation_notices.json",
        "result.html",
        "ai_input_1_1.json",
        "ai_output_1_1.json",
        "ai_attempts_1_1.json",
        "quality_report_1_1.json",
    ):
        assert client.get(f"/jobs/{job_dir.name}/{filename}").status_code == 404


def test_web_analyze_uses_admin_ai_environment_when_request_credentials_are_empty(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    _patch_web_threads_to_run_synchronously(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")

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
        attempt_log_path=None,
        request_timeout=300.0,
        max_transport_retries=1,
    ):
        assert api_base == "https://env.example.com/v1"
        assert api_key == "env-secret"
        assert [bar.index for bar in analysis.bars] == [0]
        return [ChartBar(index=0, notes="1000000000000000")], {
            "final": {"bars": []},
            "api_key_provided": True,
        }

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={
            "title": "Song Title",
            "max_bars": "1",
            "use_ai": "true",
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert status["status"] == "done"
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert "AI 增强已完成" in result.text


def test_web_analyze_can_use_ai_for_full_chart(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch, duration=4.0)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://server.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")

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
        attempt_log_path=None,
        request_timeout=300.0,
        max_transport_retries=1,
    ):
        assert course == "Hard"
        assert level == 8
        assert style == "performance"
        assert density == "max"
        assert model == "openai/test-model"
        assert api_base == "https://llm.example.com/v1"
        assert api_key == "secret-key"
        assert max_repair_attempts == 1
        assert request_timeout == 42.5
        assert max_transport_retries == 0
        assert special_notes is True
        assert attempt_log_path is not None
        assert attempt_log_path.name == "ai_attempts_1_2.json"
        attempt_log_path.write_text(
            json.dumps(
                {
                    "request_api_key": "secret-key",
                    "attempts": [{"error": {"api_key": "secret-key"}}],
                    "secret-key": "provider-echo-key",
                }
            ),
            encoding="utf-8",
        )
        assert [bar.index for bar in analysis.bars] == [0, 1]
        return [
            ChartBar(index=0, notes="111100000000", time_signature="3/4"),
            ChartBar(index=1, notes="222200000000", time_signature="3/4"),
        ], {
            "final": {"bars": []},
            "api_key_provided": True,
            "request_api_key": "secret-key",
            "attempts": [{"response": {"api_key": "secret-key"}}],
        }

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
            "ai_request_timeout": "42.5",
            "ai_transport_retries": "0",
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert "正在制谱" in response.text
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert status["status"] == "done"
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert "AI 增强已完成" in result.text
    assert (job_dir / "ai_input_1_2.json").exists()
    assert (job_dir / "ai_output_1_2.json").exists()
    generated_text = (job_dir / "preview.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "COURSE:Hard" in generated_text
    assert "LEVEL:8" in generated_text
    assert "111100000000," in generated_text
    assert "222200000000," in generated_text
    chart_options = json.loads((job_dir / "chart_options.json").read_text(encoding="utf-8"))
    assert chart_options["ai_request_timeout"] == 42.5
    assert chart_options["ai_transport_retries"] == 0
    assert 'name="ai_request_timeout" type="number" min="1" max="600" value="42.5"' in result.text
    assert 'name="ai_transport_retries" type="number" min="0" max="1" value="0"' in result.text

    for artifact_path in job_dir.rglob("*"):
        if artifact_path.is_file() and artifact_path.suffix in {".json", ".html", ".tja", ".txt"}:
            encoding = TJA_FILE_ENCODING if artifact_path.suffix == ".tja" else "utf-8"
            artifact_text = artifact_path.read_text(encoding=encoding)
            assert "secret-key" not in artifact_text
            assert "server-key" not in artifact_text


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ai_request_timeout", "0", "AI request timeout must be between 1 and 600 seconds"),
        ("ai_request_timeout", "601", "AI request timeout must be between 1 and 600 seconds"),
        ("ai_transport_retries", "-1", "AI transport retries must be 0 or 1"),
        ("ai_transport_retries", "2", "AI transport retries must be 0 or 1"),
    ],
)
def test_web_analyze_rejects_invalid_ai_transport_settings(
    tmp_path, monkeypatch, field, value, message
):
    _patch_web_audio_pipeline(monkeypatch)
    _patch_web_threads_to_run_synchronously(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", field: value},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert message in response.text
    assert list(tmp_path.iterdir()) == []


def test_web_analyze_rejects_custom_ai_url_without_request_key(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    _patch_web_threads_to_run_synchronously(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://server.example.com/v1")

    def fake_generate_chart_bars_with_ai(*args, **kwargs):
        return [ChartBar(index=0, notes="1000000000000000")], {"final": {"bars": []}}

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={
            "title": "Song Title",
            "use_ai": "true",
            "ai_base_url": "https://request.example.com/v1",
            "ai_api_key": "",
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    created_jobs = list(tmp_path.iterdir())
    if response.status_code == 200 and created_jobs:
        _wait_for_job_done(client, created_jobs[0].name)

    assert response.status_code == 400
    assert "Custom AI base URL and API key must be provided together" in response.text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("environment_name", "environment_value"),
    [
        ("OPENAI_API_KEY", "server-secret"),
        ("OPENAI_BASE_URL", "https://server.example.com/v1"),
    ],
)
def test_web_analyze_rejects_incomplete_server_ai_credentials(
    tmp_path,
    monkeypatch,
    environment_name,
    environment_value,
):
    _patch_web_audio_pipeline(monkeypatch)
    _patch_web_threads_to_run_synchronously(monkeypatch)
    monkeypatch.setattr("tja_ai_chartgen.web.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv(environment_name, environment_value)

    def fake_generate_chart_bars_with_ai(*args, **kwargs):
        return [ChartBar(index=0, notes="1000000000000000")], {"final": {"bars": []}}

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={"title": "Song Title", "use_ai": "true"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert "Server OPENAI_BASE_URL and OPENAI_API_KEY must be configured together" in response.text
    assert list(tmp_path.iterdir()) == []


def test_web_ai_failure_sidecars_redact_api_key(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    _patch_web_threads_to_run_synchronously(monkeypatch)
    secret = "request-secret-value"

    def fake_generate_chart_bars_with_ai(*args, **kwargs):
        api_key = kwargs["api_key"]
        raise AiProviderError(
            f"provider timed out for {api_key}",
            {
                "api_key": api_key,
                "provider": {"authorization": f"Bearer {api_key}"},
                "attempts": [],
                "transport_attempts": [
                    {
                        "content_attempt": 1,
                        "transport_attempt": 2,
                        "status": "error",
                        "elapsed_seconds": 300.0,
                        "error_type": "Timeout",
                        "error": f"provider timed out for {api_key}",
                    }
                ],
                "fallback_reason": "transport_retries_exhausted",
            },
        )

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )
    client = TestClient(create_app(output_dir=tmp_path))

    response = client.post(
        "/analyze",
        data={
            "title": "Song Title",
            "max_bars": "1",
            "use_ai": "true",
            "ai_base_url": "https://request.example.com/v1",
            "ai_api_key": secret,
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    assert status["status"] == "done"
    ai_notice = next(notice for notice in status["notices"] if notice["code"] == "ai-fallback")
    assert ai_notice["level"] == "warning"
    assert secret not in ai_notice["detail"]
    assert "[REDACTED]" in ai_notice["detail"]
    result = client.get(f"/jobs/{job_dir.name}/result")
    assert result.status_code == 200
    assert "AI 增强失败，已自动回退到规则生成" in result.text
    assert (job_dir / "ai_attempts_1_1.json").exists()
    assert (job_dir / "ai_output_1_1.json").exists()
    assert (job_dir / "quality_report_1_1.json").exists()
    attempts = json.loads((job_dir / "ai_attempts_1_1.json").read_text(encoding="utf-8"))
    assert attempts["fallback_reason"] == "transport_retries_exhausted"

    artifact_texts = [result.text]
    for artifact_path in job_dir.rglob("*"):
        if artifact_path.is_file() and artifact_path.suffix in {".json", ".html", ".tja", ".txt"}:
            encoding = TJA_FILE_ENCODING if artifact_path.suffix == ".tja" else "utf-8"
            artifact_texts.append(artifact_path.read_text(encoding=encoding))

    assert all(secret not in artifact_text for artifact_text in artifact_texts)
    assert any("[REDACTED]" in artifact_text for artifact_text in artifact_texts)


def test_web_remote_ai_fallback_hides_provider_details(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    _patch_web_threads_to_run_synchronously(monkeypatch)
    secret = "remote-request-secret"
    private_detail = "provider response from https://private-provider.example/v1"

    def fake_generate_chart_bars_with_ai(*args, **kwargs):
        raise AiProviderError(
            f"{private_detail} with {kwargs['api_key']}",
            {"fallback_reason": "transport_retries_exhausted"},
        )

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fake_generate_chart_bars_with_ai,
    )
    client = TestClient(create_app(output_dir=tmp_path, remote_mode=True))

    response = client.post(
        "/analyze",
        data={
            "title": "Song Title",
            "max_bars": "1",
            "use_ai": "true",
            "ai_base_url": "https://request.example.com/v1",
            "ai_api_key": secret,
        },
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    job_dir = next(tmp_path.iterdir())
    status = _wait_for_job_done(client, job_dir.name)
    result = client.get(f"/jobs/{job_dir.name}/result")
    public_text = json.dumps(status, ensure_ascii=False) + result.text
    assert "AI 增强失败，已自动回退到规则生成" in result.text
    assert private_detail not in public_text
    assert secret not in public_text
    ai_notice = next(notice for notice in status["notices"] if notice["code"] == "ai-fallback")
    assert "detail" not in ai_notice


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
        attempt_log_path=None,
        request_timeout=300.0,
        max_transport_retries=1,
    ):
        assert course == "Oni"
        assert level == 10
        assert style == "hybrid"
        assert density == "high"
        assert model == "openai/test-model"
        assert api_base == "https://llm.example.com/v1"
        assert api_key == "secret-key"
        assert max_repair_attempts == 1
        assert request_timeout == 42.5
        assert max_transport_retries == 0
        assert special_notes is True
        assert attempt_log_path is not None
        assert attempt_log_path.name == "ai_attempts_1_2.json"
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
            "ai_request_timeout": "42.5",
            "ai_transport_retries": "0",
        },
    )

    assert response.status_code == 200
    assert "AI 增强已完成" in response.text
    job_dir = tmp_path / job_id
    assert (job_dir / "ai_input_1_2.json").exists()
    assert (job_dir / "ai_output_1_2.json").exists()
    notices = json.loads(
        (job_dir / "generation_notices_1_2.json").read_text(encoding="utf-8")
    )
    assert any(notice["code"] == "ai-generation-succeeded" for notice in notices)
    generated_text = (job_dir / "regenerated_1_2.tja").read_text(encoding=TJA_FILE_ENCODING)
    assert "1111000000000000," in generated_text
    assert "2222000000000000," in generated_text
    assert 'name="ai_request_timeout" type="number" min="1" max="600" value="42.5"' in response.text
    assert 'name="ai_transport_retries" type="number" min="0" max="1" value="0"' in response.text


def test_web_regenerate_rejects_custom_ai_url_without_request_key(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    _patch_web_threads_to_run_synchronously(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://server.example.com/v1")
    client = TestClient(create_app(output_dir=tmp_path))
    analyze_response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert analyze_response.status_code == 200
    job_id = next(tmp_path.iterdir()).name
    _wait_for_job_done(client, job_id)

    def fail_if_ai_is_called(*args, **kwargs):
        pytest.fail("AI generation must not run for an invalid request")

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        fail_if_ai_is_called,
    )

    response = client.post(
        "/regenerate",
        data={
            "job_id": job_id,
            "start_bar": "1",
            "end_bar": "1",
            "course": "Oni",
            "level": "10",
            "style": "hybrid",
            "density": "high",
            "use_ai": "true",
            "ai_base_url": "https://request.example.com/v1",
            "ai_api_key": "",
        },
    )

    assert response.status_code == 400
    assert "Custom AI base URL and API key must be provided together" in response.text


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ai_request_timeout", "601", "AI request timeout must be between 1 and 600 seconds"),
        ("ai_transport_retries", "2", "AI transport retries must be 0 or 1"),
    ],
)
def test_web_regenerate_rejects_invalid_ai_transport_settings(
    tmp_path, monkeypatch, field, value, message
):
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

    response = client.post(
        "/regenerate",
        data={
            "job_id": job_id,
            "start_bar": "1",
            "end_bar": "1",
            "course": "Oni",
            "level": "10",
            "style": "technical",
            "density": "auto",
            field: value,
        },
    )

    assert response.status_code == 400
    assert message in response.text


def test_web_regenerate_runs_ai_helper_via_asyncio_to_thread(tmp_path, monkeypatch):
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
    calls = []

    def fake_generate_ai(**kwargs):
        from tja_ai_chartgen.generation import generate_chart_bars as real_generate_chart_bars

        return real_generate_chart_bars(**{**kwargs, "use_ai": False})

    async def fake_to_thread(function, *args, **kwargs):
        calls.append((function, args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr("tja_ai_chartgen.web.generate_chart_bars", fake_generate_ai)
    monkeypatch.setattr("tja_ai_chartgen.web.asyncio.to_thread", fake_to_thread)

    response = client.post(
        "/regenerate",
        data={
            "job_id": job_id,
            "start_bar": "1",
            "end_bar": "1",
            "course": "Oni",
            "level": "10",
            "style": "technical",
            "density": "auto",
            "use_ai": "true",
            "ai_request_timeout": "25",
            "ai_transport_retries": "0",
        },
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][0] is fake_generate_ai
    assert calls[0][2]["ai_request_timeout"] == 25
    assert calls[0][2]["ai_transport_retries"] == 0


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


def test_web_preview_tja_supports_variable_bar_resolutions(tmp_path, monkeypatch):
    _patch_web_audio_pipeline(monkeypatch)
    client = TestClient(create_app(output_dir=tmp_path))
    notes_24 = "1" + ("0" * 11) + "2" + ("0" * 11)
    notes_48 = "1" + ("0" * 11) + "2" + ("0" * 11) + "1" + ("0" * 23)
    tja_text = f"""TITLE:Variable Resolution
BPM:120
OFFSET:-0.5
COURSE:Oni
LEVEL:10

#START
1,
{notes_24},
{notes_48},
#END
"""

    response = client.post(
        "/preview-tja",
        files={
            "tja": ("variable.tja", tja_text.encode("utf-8"), "text/plain"),
            "audio": ("song.ogg", b"fake ogg", "audio/ogg"),
        },
    )

    assert response.status_code == 200
    assert '"time": 0.5, "type": "1"' in response.text
    assert '"time": 2.5, "type": "1"' in response.text
    assert '"time": 3.5, "type": "2"' in response.text
    assert '"time": 4.5, "type": "1"' in response.text
    assert '"time": 5.0, "type": "2"' in response.text
    assert '"time": 5.5, "type": "1"' in response.text


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


def test_read_progress_retries_when_progress_file_is_temporarily_locked(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        '{"status": "done", "step": "render", "message": "ok"}',
        encoding="utf-8",
    )
    original_read_text = type(progress_path).read_text
    attempts = {"count": 0}

    def temporarily_locked_read_text(self, encoding=None):
        if self == progress_path and attempts["count"] == 0:
            attempts["count"] += 1
            raise PermissionError("simulated locked progress")
        return original_read_text(self, encoding=encoding)

    monkeypatch.setattr(type(progress_path), "read_text", temporarily_locked_read_text)

    assert _read_progress(tmp_path)["status"] == "done"
    assert attempts["count"] == 1


def _create_export_job(tmp_path, monkeypatch, *, remote_mode=False):
    _patch_web_audio_pipeline(monkeypatch)
    jobs_dir = tmp_path / "jobs"
    client = TestClient(create_app(output_dir=jobs_dir, remote_mode=remote_mode))
    response = client.post(
        "/analyze",
        data={"title": "Song Title", "max_bars": "1"},
        files={"audio": ("song.mp3", b"fake audio", "audio/mpeg")},
    )
    assert response.status_code == 200
    job_id = next(jobs_dir.iterdir()).name
    _wait_for_job_done(client, job_id)
    return client, job_id


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


def _patch_web_threads_to_run_synchronously(monkeypatch):
    class SynchronousThread:
        def __init__(self, *, target, kwargs=None, **thread_options):
            self.target = target
            self.kwargs = kwargs or {}

        def start(self):
            self.target(**self.kwargs)

    monkeypatch.setattr("tja_ai_chartgen.web.Thread", SynchronousThread)


def _patch_web_audio_pipeline(
    monkeypatch,
    duration=2.0,
    spectral_fallback=False,
    instrument_status=None,
    instrument_reason=None,
):
    def fake_convert_to_ogg(input_path, output_path):
        output_path.write_bytes(b"fake ogg")
        return output_path

    def fake_analyze_audio(input_path, use_beatnet=False):
        beat_times = [index * 0.5 for index in range(int(duration / 0.5) + 1)]
        onset_times = beat_times[:-1]
        time_span = onset_times[-1] - onset_times[0] if len(onset_times) >= 2 else 0.0
        reason = "insufficient_onsets" if len(onset_times) < 8 else "insufficient_time_span"
        return AudioAnalysisRaw(
            bpm=120,
            beat_times=beat_times,
            onset_times=onset_times,
            onset_strengths=[],
            duration=duration,
            offset=0.0,
            analyzer="librosa",
            spectral=SpectralAnalysisRaw(
                feature_version="spectral-v1" if spectral_fallback else None,
                status="fallback" if spectral_fallback else "unavailable",
                reason="extractor-error:RuntimeError" if spectral_fallback else None,
            ),
            tempo_analysis=TempoAnalysisDecision(
                fallback_source="librosa",
                selected_source="librosa",
                estimated_bpm=120,
                estimated_offset=0.0,
                normalized_support=0.0,
                onset_count=len(onset_times),
                time_coverage=(time_span / duration) if duration > 0 else 0.0,
                accepted=False,
                reason=reason,
            ),
        )

    def fake_analyze_instruments(*_args, **_kwargs):
        return InstrumentAnalysisRaw(
            feature_version="instrument-v1",
            status=instrument_status or "complete",
            reason=instrument_reason,
            demucs_model="htdemucs",
            classifier_model="ast",
            device="cpu",
        )

    monkeypatch.setattr("tja_ai_chartgen.generation.convert_to_ogg", fake_convert_to_ogg)
    monkeypatch.setattr("tja_ai_chartgen.generation.analyze_audio", fake_analyze_audio)
    monkeypatch.setattr(
        "tja_ai_chartgen.generation.analyze_instruments",
        fake_analyze_instruments,
    )
