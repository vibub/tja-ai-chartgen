import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.generation import (
    GenerationConfig,
    build_song_analysis,
    generate_chart_bars,
    load_generation_config,
)
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, SongAnalysis


def _bar(index: int = 0) -> BarFeature:
    return BarFeature(
        index=index,
        start_time=float(index * 2),
        end_time=float((index + 1) * 2),
        energy=0.5,
        onset_16=[0, 4, 8, 12],
        activity_16=[0.5] * 16,
        section="verse",
    )


def _analysis(bars: list[BarFeature] | None = None) -> SongAnalysis:
    return SongAnalysis(
        title="Song",
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120.0,
        offset=0.0,
        bars=bars or [_bar()],
    )


def _config_data(tmp_path: Path) -> dict[str, object]:
    return {
        "input_audio": str(tmp_path / "song.mp3"),
        "title": "Song",
        "output_dir": str(tmp_path / "output"),
    }


def test_generation_config_writes_schema_version_and_excludes_secrets(tmp_path):
    config = GenerationConfig.model_validate(_config_data(tmp_path))

    payload = config.model_dump(mode="json")

    assert payload["schema_version"] == 1
    assert "ai_api_key" not in payload
    assert "ai_base_url" not in payload

    with pytest.raises(ValidationError):
        GenerationConfig.model_validate({**_config_data(tmp_path), "ai_api_key": "secret"})


def test_load_generation_config_accepts_legacy_config_without_schema_version(tmp_path):
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(json.dumps(_config_data(tmp_path)), encoding="utf-8")

    config = load_generation_config(config_path)

    assert config.schema_version == 1
    assert config.title == "Song"


def test_load_generation_config_rejects_unknown_schema_version(tmp_path):
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps({**_config_data(tmp_path), "schema_version": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported generation config schema_version: 2"):
        load_generation_config(config_path)


def test_build_song_analysis_applies_overrides_and_max_bars(tmp_path, monkeypatch):
    input_path = tmp_path / "song.mp3"
    ogg_path = tmp_path / "song.ogg"
    input_path.write_bytes(b"audio")
    raw = AudioAnalysisRaw(
        duration=8.0,
        bpm=120.0,
        offset=0.0,
        time_signature="4/4",
        beat_times=[],
        onset_times=[],
        onset_strengths=[],
    )
    converted: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        "tja_ai_chartgen.generation.convert_to_ogg",
        lambda source, target: converted.append((source, target)),
    )
    monkeypatch.setattr("tja_ai_chartgen.generation.analyze_audio", lambda *_args, **_kwargs: raw)

    analysis = build_song_analysis(
        input_audio=input_path,
        ogg_path=ogg_path,
        title="Song",
        artist="Artist",
        max_bars=2,
        bpm_override=240.0,
        offset_override=0.25,
        time_signature_override="3/4",
        use_beatnet=True,
    )

    assert converted == [(input_path, ogg_path)]
    assert analysis.bpm == 240.0
    assert analysis.offset == 0.25
    assert analysis.time_signature == "3/4"
    assert len(analysis.bars) == 2
    assert all(bar.time_signature == "3/4" for bar in analysis.bars)


def test_generate_chart_bars_falls_back_and_builds_quality_report(monkeypatch):
    analysis = _analysis()
    expected = [ChartBar(index=0, notes="1000100010001000")]
    fallback_calls = []

    def fake_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        "tja_ai_chartgen.generation.generate_fallback_chart_bars",
        fake_fallback,
    )

    result = generate_chart_bars(
        analysis=analysis,
        selected_bars=analysis.bars,
        course="Oni",
        level=10,
        style="technical",
        density="high",
        special_notes=False,
        use_ai=False,
    )

    assert result.chart_bars == expected
    assert result.used_fallback is True
    assert result.ai_failure is None
    assert result.quality_report.bar_count == 1
    assert fallback_calls == [
        (
            (analysis.bars,),
            {
                "style": "technical",
                "density": "high",
                "special_notes": False,
                "course": "Oni",
                "level": 10,
            },
        )
    ]


def test_generate_chart_bars_writes_ai_sidecars_and_sanitizes(tmp_path, monkeypatch):
    analysis = _analysis([_bar(index=3)])
    ai_input_path = tmp_path / "input.json"
    ai_output_path = tmp_path / "output.json"
    ai_attempts_path = tmp_path / "attempts.json"

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.generate_chart_bars_with_ai",
        lambda *_args, **_kwargs: (
            [ChartBar(index=0, notes="1111000000000000")],
            {"bars": [{"index": 0, "notes": "1111000000000000"}]},
        ),
    )

    result = generate_chart_bars(
        analysis=analysis,
        selected_bars=analysis.bars,
        course="Oni",
        level=10,
        style="technical",
        density="high",
        special_notes=False,
        use_ai=True,
        model="openai/test-model",
        ai_input_path=ai_input_path,
        ai_output_path=ai_output_path,
        ai_attempts_path=ai_attempts_path,
    )

    assert ai_input_path.exists()
    assert ai_output_path.exists()
    assert result.used_fallback is False
    assert result.chart_bars[0].index == 3
    assert result.chart_bars[0].notes == "1111000000000000"
