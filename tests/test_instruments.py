from types import SimpleNamespace

import numpy as np
import pytest

from tja_ai_chartgen.audio.instrument_models import InstrumentModelError
from tja_ai_chartgen.audio.instruments import (
    INSTRUMENT_FEATURE_VERSION,
    InstrumentAnalysisError,
    InstrumentAnalysisRaw,
    InstrumentClassificationWindow,
    StemActivityFrame,
    _build_stem_frames,
    _select_device,
    _taxonomy_label_indexes,
    _window_audio,
    analyze_instruments,
    instrument_fallback,
)
from tja_ai_chartgen.tja.model import BarFeature, InstrumentBarFeature, SongAnalysis


def test_instrument_models_have_stable_compatible_defaults():
    raw = InstrumentAnalysisRaw()
    bar = BarFeature(index=0, start_time=0.0, end_time=2.0, energy=0.5)
    analysis = SongAnalysis(
        title="Song",
        audio_file="song.wav",
        ogg_file="song.ogg",
        bpm=120.0,
        offset=0.0,
        bars=[bar],
    )

    assert raw.status == "unavailable"
    assert raw.stem_frames == []
    assert bar.instrument == InstrumentBarFeature()
    assert bar.instrument_grid_features == []
    assert analysis.instrument_feature_version is None
    assert analysis.instrument_analysis_status == "unavailable"
    assert analysis.instrument_analysis_reason is None


def test_song_analysis_preserves_legacy_instrument_v1_status_fields():
    legacy_payload = SongAnalysis(
        title="Song",
        audio_file="song.wav",
        ogg_file="song.ogg",
        bpm=120.0,
        offset=0.0,
        instrument_feature_version="instrument-v1",
        instrument_analysis_status="partial",
        instrument_analysis_reason="classifier-load-error:OSError",
        instrument_demucs_model="htdemucs",
        instrument_classifier_model="MIT/ast-finetuned-audioset-10-10-0.4593",
        instrument_analysis_device="cpu",
        bars=[],
    ).model_dump(mode="json")

    restored = SongAnalysis.model_validate(legacy_payload)

    assert restored.instrument_feature_version == "instrument-v1"
    assert restored.instrument_analysis_status == "partial"
    assert restored.instrument_analysis_reason == "classifier-load-error:OSError"
    assert restored.instrument_demucs_model == "htdemucs"
    assert restored.instrument_classifier_model == (
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    assert restored.instrument_analysis_device == "cpu"


def test_instrument_raw_serializes_stable_frames_and_windows():
    raw = InstrumentAnalysisRaw(
        feature_version=INSTRUMENT_FEATURE_VERSION,
        status="complete",
        demucs_model="htdemucs",
        classifier_model="ast",
        device="cpu",
        analyzed_duration=4.0,
        stem_frames=[StemActivityFrame(time=0.5, vocals=0.8, drum_onset=0.6)],
        classification_windows=[
            InstrumentClassificationWindow(
                start_time=0.0,
                end_time=4.0,
                mix_scores={"guitar": 0.4},
                other_scores={"guitar": 0.8},
            )
        ],
    )

    payload = raw.model_dump(mode="json")

    assert payload["feature_version"] == "instrument-v1"
    assert payload["stem_frames"][0]["vocals"] == 0.8
    assert payload["classification_windows"][0]["other_scores"] == {"guitar": 0.8}
    assert raw.has_stem_evidence is True
    assert raw.has_classifier_evidence is True
    assert raw.uses_legacy_instrument_semantics is True


def test_instrument_raw_reads_legacy_instrument_v1_partial_json_without_rewriting_status():
    legacy = InstrumentAnalysisRaw.model_validate(
        {
            "feature_version": "instrument-v1",
            "status": "partial",
            "reason": "classifier-load-error:OSError",
            "demucs_model": "htdemucs",
            "classifier_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "device": "cpu",
            "analyzed_duration": 2.0,
            "stem_frames": [{"time": 0.0, "drums": 0.8, "drum_onset": 0.7}],
            "classification_windows": [],
        }
    )

    assert legacy.feature_version == "instrument-v1"
    assert legacy.status == "partial"
    assert legacy.reason == "classifier-load-error:OSError"
    assert legacy.has_stem_evidence is True
    assert legacy.has_classifier_evidence is False
    assert legacy.uses_legacy_instrument_semantics is True
    assert legacy.model_dump(mode="json")["status"] == "partial"


def test_instrument_fallback_uses_stable_feature_version_and_reason():
    result = instrument_fallback("missing-model:htdemucs")

    assert result.feature_version == "instrument-v1"
    assert result.status == "fallback"
    assert result.reason == "missing-model:htdemucs"
    assert result.has_stem_evidence is False
    assert result.has_classifier_evidence is False


class _Backend:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Cuda(_Backend):
    class OutOfMemoryError(RuntimeError):
        pass

    def empty_cache(self) -> None:
        return None


def _fake_torch(*, cuda: bool = False, mps: bool = False):
    return SimpleNamespace(
        cuda=_Cuda(cuda),
        backends=SimpleNamespace(mps=_Backend(mps)),
    )


def test_select_device_prefers_cuda_then_mps_then_cpu():
    assert _select_device(_fake_torch(cuda=True, mps=True), "auto") == "cuda"
    assert _select_device(_fake_torch(cuda=False, mps=True), "auto") == "mps"
    assert _select_device(_fake_torch(), "auto") == "cpu"


def test_select_device_rejects_unavailable_explicit_backend():
    with pytest.raises(InstrumentAnalysisError) as captured:
        _select_device(_fake_torch(), "cuda")

    assert captured.value.reason == "device-unavailable:cuda"


def test_window_audio_uses_deterministic_four_second_windows_and_two_second_hop():
    samples = np.arange(5 * 10, dtype=np.float32)

    windows = _window_audio(samples, 10)

    assert [(start, end) for start, end, _ in windows] == [
        (0.0, 4.0),
        (2.0, 5.0),
        (4.0, 5.0),
    ]
    assert all(window.shape == (40,) for _, _, window in windows)
    assert np.all(windows[-1][2][10:] == 0.0)


def test_taxonomy_label_indexes_uses_fixed_audioset_names():
    indexes = _taxonomy_label_indexes(
        {
            "140": "Guitar",
            "153": "Piano",
            "190": "String section",
            "155": "Organ",
            "999": "Unrelated label",
        }
    )

    assert indexes["guitar"] == [140]
    assert indexes["piano_keyboard"] == [153]
    assert indexes["strings"] == [190]
    assert indexes["organ"] == [155]
    assert all(999 not in values for values in indexes.values())


def test_build_stem_frames_returns_aligned_activity_and_onsets():
    sample_rate = 800
    timeline = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    pulse = (np.sin(2 * np.pi * 4 * timeline) > 0.8).astype(np.float32)
    mix = np.vstack([pulse, pulse])
    stems = {
        "vocals": np.vstack([pulse * 0.7, pulse * 0.7]),
        "drums": np.vstack([pulse, pulse]),
        "bass": np.vstack([pulse * 0.4, pulse * 0.4]),
        "other": np.vstack([pulse * 0.2, pulse * 0.2]),
    }

    frames = _build_stem_frames(
        mix,
        stems,
        source_sample_rate=sample_rate,
        analysis_sample_rate=sample_rate,
        hop_length=100,
    )

    assert len(frames) == 17
    assert any(frame.drums > 0.0 for frame in frames)
    assert any(frame.drum_onset > 0.0 for frame in frames)
    assert all(0.0 <= frame.vocals <= 1.0 for frame in frames)
    assert [frame.time for frame in frames] == sorted(frame.time for frame in frames)


def test_analyze_instruments_returns_complete_result_from_both_models(tmp_path, monkeypatch):
    audio_path = tmp_path / "song.ogg"
    audio_path.write_bytes(b"audio")
    frame = StemActivityFrame(time=0.0, vocals=0.5)
    window = InstrumentClassificationWindow(
        start_time=0.0,
        end_time=1.0,
        mix_scores={"guitar": 0.3},
        other_scores={"guitar": 0.7},
    )
    mix = np.zeros((2, 800), dtype=np.float32)
    stems = {name: mix.copy() for name in ("vocals", "drums", "bass", "other")}
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments.validate_instrument_model_dir",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr("tja_ai_chartgen.audio.instruments._import_torch", _fake_torch)
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._separate_stems",
        lambda *_args, **_kwargs: (mix, stems, 800),
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._build_stem_frames",
        lambda *_args, **_kwargs: [frame],
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._classify_stems",
        lambda *_args, **_kwargs: [window],
    )

    result = analyze_instruments(
        audio_path,
        model_dir=tmp_path / "models",
        analysis_sample_rate=800,
        analysis_hop_length=100,
    )

    assert result.status == "complete"
    assert result.device == "cpu"
    assert result.analyzed_duration == 1.0
    assert result.stem_frames == [frame]
    assert result.classification_windows == [window]


def test_analyze_instruments_stem_role_skips_classifier(tmp_path, monkeypatch):
    audio_path = tmp_path / "song.ogg"
    audio_path.write_bytes(b"audio")
    frame = StemActivityFrame(time=0.0, drums=0.8)
    mix = np.zeros((2, 800), dtype=np.float32)
    stems = {name: mix.copy() for name in ("vocals", "drums", "bass", "other")}
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments.validate_instrument_model_dir",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr("tja_ai_chartgen.audio.instruments._import_torch", _fake_torch)
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._separate_stems",
        lambda *_args, **_kwargs: (mix, stems, 800),
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._build_stem_frames",
        lambda *_args, **_kwargs: [frame],
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._classify_stems",
        lambda *_args, **_kwargs: pytest.fail("stem-role must not load AST"),
    )

    result = analyze_instruments(
        audio_path,
        model_dir=tmp_path / "models",
        profile="stem-role",
        analysis_sample_rate=800,
        analysis_hop_length=100,
    )

    assert result.feature_version == "stem-role-v1"
    assert result.status == "complete"
    assert result.classifier_model is None
    assert result.has_stem_evidence is True
    assert result.has_classifier_evidence is False
    assert result.uses_legacy_instrument_semantics is False
    assert result.stem_frames == [frame]
    assert result.classification_windows == []


def test_analyze_instruments_degrades_full_profile_when_ast_files_are_missing(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "song.ogg"
    audio_path.write_bytes(b"audio")
    frame = StemActivityFrame(time=0.0, drums=0.8)
    mix = np.zeros((2, 800), dtype=np.float32)
    stems = {name: mix.copy() for name in ("vocals", "drums", "bass", "other")}
    validation_calls = []

    def fake_validate(_path, **kwargs):
        validation_calls.append(kwargs)
        if kwargs.get("require_classifier", True):
            raise InstrumentModelError("missing-model:ast")
        return None

    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments.validate_instrument_model_dir",
        fake_validate,
    )
    monkeypatch.setattr("tja_ai_chartgen.audio.instruments._import_torch", _fake_torch)
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._separate_stems",
        lambda *_args, **_kwargs: (mix, stems, 800),
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._build_stem_frames",
        lambda *_args, **_kwargs: [frame],
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._classify_stems",
        lambda *_args, **_kwargs: pytest.fail("missing AST must skip classifier loading"),
    )

    result = analyze_instruments(
        audio_path,
        model_dir=tmp_path / "models",
        profile="full",
        analysis_sample_rate=800,
        analysis_hop_length=100,
    )

    assert validation_calls == [
        {"profile": "full"},
        {"profile": "full", "require_classifier": False},
    ]
    assert result.feature_version == "stem-role-v1"
    assert result.status == "complete"
    assert result.reason == "missing-model:ast"
    assert result.has_stem_evidence is True
    assert result.classifier_model is None


def test_analyze_instruments_keeps_demucs_evidence_when_classifier_fails(tmp_path, monkeypatch):
    audio_path = tmp_path / "song.ogg"
    audio_path.write_bytes(b"audio")
    frame = StemActivityFrame(time=0.0, drums=0.8)
    mix = np.zeros((2, 800), dtype=np.float32)
    stems = {name: mix.copy() for name in ("vocals", "drums", "bass", "other")}
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments.validate_instrument_model_dir",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr("tja_ai_chartgen.audio.instruments._import_torch", _fake_torch)
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._separate_stems",
        lambda *_args, **_kwargs: (mix, stems, 800),
    )
    monkeypatch.setattr(
        "tja_ai_chartgen.audio.instruments._build_stem_frames",
        lambda *_args, **_kwargs: [frame],
    )

    def fail(*_args, **_kwargs):
        raise InstrumentAnalysisError("classifier-load-error:OSError")

    monkeypatch.setattr("tja_ai_chartgen.audio.instruments._classify_stems", fail)

    result = analyze_instruments(
        audio_path,
        model_dir=tmp_path / "models",
        analysis_sample_rate=800,
        analysis_hop_length=100,
    )

    assert result.feature_version == "stem-role-v1"
    assert result.status == "complete"
    assert result.reason == "classifier-load-error:OSError"
    assert result.classifier_model is None
    assert result.has_stem_evidence is True
    assert result.has_classifier_evidence is False
    assert result.uses_legacy_instrument_semantics is False
    assert result.stem_frames == [frame]
    assert result.classification_windows == []


def test_analyze_instruments_falls_back_before_importing_models_when_manifest_missing(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "song.ogg"
    audio_path.write_bytes(b"audio")
    imported = False

    def should_not_import():
        nonlocal imported
        imported = True
        return _fake_torch()

    monkeypatch.setattr("tja_ai_chartgen.audio.instruments._import_torch", should_not_import)

    result = analyze_instruments(
        audio_path,
        model_dir=tmp_path / "missing",
        analysis_sample_rate=800,
        analysis_hop_length=100,
    )

    assert result.status == "fallback"
    assert result.reason == "missing-model:manifest"
    assert imported is False
