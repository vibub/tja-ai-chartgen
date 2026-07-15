import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw, StemActivityFrame
from tja_ai_chartgen.audio.spectral import SpectralAnalysisRaw
from tja_ai_chartgen.generation import (
    GenerationConfig,
    build_analysis_notices,
    build_song_analysis,
    generate_chart_bars,
    load_generation_config,
)
from tja_ai_chartgen.tja.model import (
    BarFeature,
    ChartBar,
    ResolutionDecision,
    ResolutionPlan,
    SongAnalysis,
    TempoAnalysisDecision,
    TempoMeterCandidate,
)


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


def test_generation_config_accepts_transport_retry_upper_bound(tmp_path):
    config = GenerationConfig.model_validate(
        {**_config_data(tmp_path), "ai_transport_retries": 5}
    )

    assert config.ai_transport_retries == 5

    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(
            {**_config_data(tmp_path), "ai_transport_retries": 6}
        )


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
        spectral=SpectralAnalysisRaw(
            feature_version="spectral-v1",
            status="complete",
        ),
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
    assert analysis.analysis_schema_version == 8
    assert analysis.spectral_feature_version == "spectral-v1"
    assert analysis.spectral_analysis_status == "complete"
    assert analysis.instrument_feature_version is None
    assert analysis.instrument_analysis_status == "unavailable"
    assert analysis.structure_feature_version == "structure-v1"
    assert analysis.structure_confidence is not None
    assert len(analysis.bar_structures) == 2
    assert len(analysis.phrase_plan) == 1
    assert analysis.phrase_plan[0].resolution == 12
    assert analysis.resolution_policy_version == "song-global-v1"
    assert analysis.resolution_plan is not None
    assert analysis.resolution_plan.canonical_grids_per_bar == 36
    assert analysis.resolution_plan.bar_resolutions == [12, 12]


def test_build_song_analysis_runs_optional_instrument_analysis_after_overrides(
    tmp_path,
    monkeypatch,
):
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
        sample_rate=22_050,
        tempo_candidates=[
            TempoMeterCandidate(
                source="beatnet",
                bpm=120,
                offset=0.0,
                time_signature="4/4",
                beat_times=[0.0, 0.5, 1.0, 1.5],
                downbeat_times=[0.0],
                accepted=True,
            )
        ],
    )
    instrument_result = InstrumentAnalysisRaw(
        feature_version="instrument-v1",
        status="complete",
        demucs_model="htdemucs",
        classifier_model="ast",
        device="cpu",
        stem_frames=[
            StemActivityFrame(
                time=0.0,
                vocals=0.7,
                drum_onset=0.8,
                bass_onset=0.6,
            )
        ],
    )
    calls = []
    stages = []
    monkeypatch.setattr("tja_ai_chartgen.generation.convert_to_ogg", lambda *_args: None)
    monkeypatch.setattr("tja_ai_chartgen.generation.analyze_audio", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(
        "tja_ai_chartgen.generation.resolve_instrument_model_dir",
        lambda value: tmp_path / "models" if value is None else value,
    )

    def fake_analyze(path, **kwargs):
        calls.append((path, kwargs))
        return instrument_result

    monkeypatch.setattr("tja_ai_chartgen.generation.analyze_instruments", fake_analyze)

    analysis = build_song_analysis(
        input_audio=input_path,
        ogg_path=ogg_path,
        title="Song",
        max_bars=2,
        bpm_override=240.0,
        offset_override=0.25,
        time_signature_override="3/4",
        use_instrument_analysis=True,
        instrument_device="cpu",
        stage_callback=stages.append,
    )

    assert stages == ["convert", "analyze", "instruments", "features"]
    assert calls == [
        (
            ogg_path,
            {
                "model_dir": tmp_path / "models",
                "device": "cpu",
                "analysis_sample_rate": 22_050,
                "analysis_hop_length": 512,
                "max_duration": 5.75,
            },
        )
    ]
    assert analysis.analysis_schema_version == 8
    assert analysis.instrument_feature_version == "instrument-v1"
    assert analysis.instrument_analysis_status == "complete"
    assert analysis.instrument_demucs_model == "htdemucs"
    assert analysis.instrument_classifier_model == "ast"
    assert analysis.instrument_analysis_device == "cpu"
    assert analysis.tempo_candidates[0].evidence.auxiliary_drum_onset_support == 0.8
    assert analysis.tempo_candidates[0].evidence.auxiliary_bass_onset_support == 0.6


def test_build_analysis_notices_reports_optional_analysis_fallbacks():
    bars = [_bar(index=index) for index in range(4)]
    analysis = _analysis(bars).model_copy(
        update={
            "analyzer": "librosa",
            "beatnet_analysis_status": "fallback",
            "beatnet_analysis_reason": "inference-error:RuntimeError",
            "spectral_feature_version": "spectral-v1",
            "spectral_analysis_status": "fallback",
            "spectral_analysis_reason": "extractor-error:RuntimeError",
            "structure_confidence": 0.2,
            "tempo_analysis": TempoAnalysisDecision(
                fallback_source="librosa",
                selected_source="librosa",
                estimated_bpm=120.0,
                estimated_offset=0.0,
                normalized_support=0.15,
                onset_count=3,
                time_coverage=0.2,
                accepted=False,
                reason="insufficient_onsets",
            ),
            "resolution_plan": ResolutionPlan(
                canonical_grids_per_bar=48,
                base_resolution=16,
                bar_resolutions=[16] * 4,
                decision=ResolutionDecision(
                    selected_resolution=16,
                    evidence_count=0,
                    reason="insufficient reliable onset evidence",
                ),
            ),
        }
    )

    notices = build_analysis_notices(analysis, requested_beatnet=True)

    assert {notice.code for notice in notices} == {
        "beatnet-fallback",
        "spectral-analysis-fallback",
        "tempo-refinement-fallback",
        "structure-low-confidence",
        "resolution-evidence-fallback",
    }
    assert all(notice.level == "warning" for notice in notices)
    assert all(notice.stage == "analysis" for notice in notices)
    beatnet_notice = next(notice for notice in notices if notice.code == "beatnet-fallback")
    assert beatnet_notice.detail == "reason=inference-error:RuntimeError"


@pytest.mark.parametrize(
    ("status", "reason", "expected_code", "expected_level"),
    [
        ("complete", None, "instrument-analysis-succeeded", "info"),
        ("partial", "classifier-error:ValueError", "instrument-analysis-partial", "warning"),
        (
            "fallback",
            "missing-dependency:demucs",
            "instrument-dependencies-unavailable",
            "warning",
        ),
        ("fallback", "missing-model:ast", "instrument-models-missing", "warning"),
        ("fallback", "device-unavailable:cuda", "instrument-device-unavailable", "warning"),
        ("fallback", "cuda-out-of-memory", "instrument-analysis-fallback", "warning"),
    ],
)
def test_build_analysis_notices_reports_instrument_analysis_status(
    status,
    reason,
    expected_code,
    expected_level,
):
    analysis = _analysis().model_copy(
        update={
            "instrument_feature_version": "instrument-v1",
            "instrument_analysis_status": status,
            "instrument_analysis_reason": reason,
        }
    )

    notices = build_analysis_notices(
        analysis,
        requested_instrument_analysis=True,
    )

    instrument_notice = next(notice for notice in notices if notice.code == expected_code)
    assert instrument_notice.level == expected_level
    assert instrument_notice.stage == "analysis"
    if reason is not None:
        assert instrument_notice.detail == f"reason={reason}"


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
                "resolution_plan": None,
            },
        )
    ]


def test_generate_chart_bars_exposes_structured_ai_fallback_notice(monkeypatch):
    analysis = _analysis()

    def fail_ai_generation(**kwargs):
        raise RuntimeError(f"provider rejected {kwargs['api_key']}")

    monkeypatch.setattr(
        "tja_ai_chartgen.generation._generate_ai_bars",
        fail_ai_generation,
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
        api_key="request-secret",
    )

    assert result.used_fallback is True
    assert result.ai_failure == "provider rejected [REDACTED]"
    assert [notice.code for notice in result.notices] == ["ai-fallback"]
    assert result.notices[0].detail == result.ai_failure
    assert result.notices[0].scope == "Oni"


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
    assert [notice.code for notice in result.notices] == ["ai-generation-succeeded"]
    assert result.notices[0].scope == "Oni"
