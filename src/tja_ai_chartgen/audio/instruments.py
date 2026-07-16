from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
from pydantic import BaseModel, Field

from tja_ai_chartgen.audio.instrument_models import (
    AST_MODEL_ID,
    DEMUCS_MODEL_NAME,
    INSTRUMENT_FEATURE_VERSION,
    STEM_ROLE_FEATURE_VERSION,
    InstrumentModelError,
    InstrumentModelProfile,
    validate_instrument_model_dir,
)


AST_SAMPLE_RATE = 16_000
AST_WINDOW_SECONDS = 4.0
AST_HOP_SECONDS = 2.0
AST_BATCH_SIZE = 8
DEMUCS_SEGMENT_SECONDS = 7.8
DEMUCS_RETRY_SEGMENT_SECONDS = 4.0
DEMUCS_OVERLAP = 0.25
SILENCE_EPSILON = 1e-8
STEM_PRESENCE_THRESHOLD = 0.08

INSTRUMENT_TAXONOMY = {
    "guitar": (
        "Guitar",
        "Electric guitar",
        "Acoustic guitar",
        "Steel guitar, slide guitar",
    ),
    "piano_keyboard": (
        "Piano",
        "Electric piano",
        "Keyboard (musical)",
    ),
    "strings": (
        "String section",
        "Violin, fiddle",
        "Cello",
        "Orchestra",
        "Bowed string instrument",
        "Pizzicato",
    ),
    "brass": (
        "Brass instrument",
        "Trumpet",
        "Trombone",
        "French horn",
    ),
    "woodwind": (
        "Wind instrument, woodwind instrument",
        "Flute",
        "Saxophone",
        "Clarinet",
        "Oboe",
        "Bassoon",
        "Piccolo",
    ),
    "synth": (
        "Synthesizer",
        "Sampler",
    ),
    "organ": (
        "Organ",
        "Hammond organ",
    ),
    "other_instrument": (
        "Musical instrument",
        "Plucked string instrument",
        "Harp",
        "Accordion",
        "Harmonica",
    ),
}


class StemActivityFrame(BaseModel):
    time: float = Field(ge=0.0)
    vocals: float = Field(default=0.0, ge=0.0, le=1.0)
    drums: float = Field(default=0.0, ge=0.0, le=1.0)
    bass: float = Field(default=0.0, ge=0.0, le=1.0)
    other: float = Field(default=0.0, ge=0.0, le=1.0)
    vocal_onset: float = Field(default=0.0, ge=0.0, le=1.0)
    drum_onset: float = Field(default=0.0, ge=0.0, le=1.0)
    bass_onset: float = Field(default=0.0, ge=0.0, le=1.0)
    accompaniment_onset: float = Field(default=0.0, ge=0.0, le=1.0)


class InstrumentClassificationWindow(BaseModel):
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    mix_scores: dict[str, float] = Field(default_factory=dict)
    other_scores: dict[str, float] = Field(default_factory=dict)


class InstrumentAnalysisRaw(BaseModel):
    feature_version: str | None = None
    status: Literal["unavailable", "complete", "partial", "fallback"] = "unavailable"
    reason: str | None = None
    demucs_model: str | None = None
    classifier_model: str | None = None
    device: str | None = None
    analyzed_duration: float = Field(default=0.0, ge=0.0)
    stem_frames: list[StemActivityFrame] = Field(default_factory=list)
    classification_windows: list[InstrumentClassificationWindow] = Field(default_factory=list)

    @property
    def has_stem_evidence(self) -> bool:
        return bool(self.stem_frames) and self.status in {"complete", "partial"}

    @property
    def has_classifier_evidence(self) -> bool:
        return bool(self.classification_windows)

    @property
    def uses_legacy_instrument_semantics(self) -> bool:
        return self.feature_version == INSTRUMENT_FEATURE_VERSION


class InstrumentAnalysisError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def analyze_instruments(
    input_path: Path,
    *,
    model_dir: Path,
    profile: InstrumentModelProfile = "full",
    device: str = "auto",
    analysis_sample_rate: int,
    analysis_hop_length: int,
    max_duration: float | None = None,
) -> InstrumentAnalysisRaw:
    if not input_path.is_file():
        return instrument_fallback("audio-not-found", profile=profile)
    if analysis_sample_rate <= 0:
        return instrument_fallback("invalid-sample-rate", profile=profile)
    if analysis_hop_length <= 0:
        return instrument_fallback("invalid-hop-length", profile=profile)
    classifier_preflight_reason: str | None = None
    try:
        validate_instrument_model_dir(model_dir, profile=profile)
    except InstrumentModelError as error:
        if profile != "full":
            return instrument_fallback(error.reason, profile=profile)
        try:
            validate_instrument_model_dir(
                model_dir,
                profile=profile,
                require_classifier=False,
            )
        except InstrumentModelError as stem_error:
            return instrument_fallback(stem_error.reason, profile=profile)
        classifier_preflight_reason = error.reason

    try:
        torch = _import_torch()
        selected_device = _select_device(torch, device)
    except InstrumentAnalysisError as error:
        return instrument_fallback(error.reason, profile=profile)

    try:
        mix, stems, model_sample_rate = _separate_stems(
            input_path,
            model_dir=model_dir,
            device=selected_device,
            max_duration=max_duration,
            torch=torch,
        )
        frames = _build_stem_frames(
            mix,
            stems,
            source_sample_rate=model_sample_rate,
            analysis_sample_rate=analysis_sample_rate,
            hop_length=analysis_hop_length,
        )
        if not frames:
            raise InstrumentAnalysisError("invalid-model-output")
    except InstrumentAnalysisError as error:
        return instrument_fallback(error.reason, profile=profile, device=selected_device)
    except Exception as error:  # noqa: BLE001 - optional enhancement must not block generation.
        return instrument_fallback(
            f"demucs-error:{type(error).__name__}",
            profile=profile,
            device=selected_device,
        )

    analyzed_duration = mix.shape[-1] / model_sample_rate
    if profile == "stem-role":
        return InstrumentAnalysisRaw(
            feature_version=STEM_ROLE_FEATURE_VERSION,
            status="complete",
            demucs_model=DEMUCS_MODEL_NAME,
            device=selected_device,
            analyzed_duration=round(float(analyzed_duration), 6),
            stem_frames=frames,
        )
    if classifier_preflight_reason is not None:
        return _classifier_fallback_result(
            reason=classifier_preflight_reason,
            device=selected_device,
            analyzed_duration=analyzed_duration,
            frames=frames,
        )

    try:
        windows = _classify_stems(
            mix,
            stems["other"],
            sample_rate=model_sample_rate,
            model_dir=model_dir,
            device=selected_device,
            torch=torch,
        )
    except InstrumentAnalysisError as error:
        return _classifier_fallback_result(
            reason=error.reason,
            device=selected_device,
            analyzed_duration=analyzed_duration,
            frames=frames,
        )
    except Exception as error:  # noqa: BLE001 - Demucs evidence remains usable.
        return _classifier_fallback_result(
            reason=f"classifier-error:{type(error).__name__}",
            device=selected_device,
            analyzed_duration=analyzed_duration,
            frames=frames,
        )

    return InstrumentAnalysisRaw(
        feature_version=INSTRUMENT_FEATURE_VERSION,
        status="complete",
        demucs_model=DEMUCS_MODEL_NAME,
        classifier_model=AST_MODEL_ID,
        device=selected_device,
        analyzed_duration=round(float(analyzed_duration), 6),
        stem_frames=frames,
        classification_windows=windows,
    )


def _classifier_fallback_result(
    *,
    reason: str,
    device: str,
    analyzed_duration: float,
    frames: list[StemActivityFrame],
) -> InstrumentAnalysisRaw:
    """AST 不可用时降级为完整 stem-role 能力，而不是部分 stem 结果。"""
    return InstrumentAnalysisRaw(
        feature_version=STEM_ROLE_FEATURE_VERSION,
        status="complete",
        reason=reason,
        demucs_model=DEMUCS_MODEL_NAME,
        classifier_model=None,
        device=device,
        analyzed_duration=round(float(analyzed_duration), 6),
        stem_frames=frames,
    )


def instrument_fallback(
    reason: str,
    *,
    profile: InstrumentModelProfile = "full",
    device: str | None = None,
) -> InstrumentAnalysisRaw:
    stem_role = profile == "stem-role"
    return InstrumentAnalysisRaw(
        feature_version=STEM_ROLE_FEATURE_VERSION if stem_role else INSTRUMENT_FEATURE_VERSION,
        status="fallback",
        reason=reason,
        demucs_model=DEMUCS_MODEL_NAME,
        classifier_model=None if stem_role else AST_MODEL_ID,
        device=device,
    )


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise InstrumentAnalysisError("missing-dependency:torch") from error
    return torch


def _select_device(torch: Any, requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise InstrumentAnalysisError(f"device-unavailable:{normalized or 'unknown'}")
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if normalized == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if normalized == "cuda" and not cuda_available:
        raise InstrumentAnalysisError("device-unavailable:cuda")
    if normalized == "mps" and not mps_available:
        raise InstrumentAnalysisError("device-unavailable:mps")
    return normalized


def _separate_stems(
    input_path: Path,
    *,
    model_dir: Path,
    device: str,
    max_duration: float | None,
    torch: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    try:
        from demucs.api import Separator
    except ImportError as error:
        raise InstrumentAnalysisError("missing-dependency:demucs") from error
    try:
        separator = Separator(
            model=DEMUCS_MODEL_NAME,
            repo=model_dir / "demucs",
            device=device,
            segment=DEMUCS_SEGMENT_SECONDS,
            shifts=0,
            split=True,
            overlap=DEMUCS_OVERLAP,
            jobs=0,
            progress=False,
        )
    except Exception as error:  # noqa: BLE001 - dependency errors become stable reasons.
        raise InstrumentAnalysisError(f"demucs-load-error:{type(error).__name__}") from error

    try:
        waveform, source_rate = librosa.load(
            str(input_path),
            sr=None,
            mono=False,
            duration=max_duration,
        )
    except Exception as error:  # noqa: BLE001 - audio loading is contained.
        raise InstrumentAnalysisError(f"audio-load-error:{type(error).__name__}") from error
    audio = _channels_first(waveform)
    if audio.shape[-1] == 0 or not np.all(np.isfinite(audio)):
        raise InstrumentAnalysisError("invalid-waveform")
    tensor = torch.as_tensor(audio, dtype=torch.float32)

    try:
        origin, separated = separator.separate_tensor(tensor, sr=int(source_rate))
    except Exception as error:  # noqa: BLE001 - retry only recognized CUDA OOM failures.
        if device != "cuda" or not _is_cuda_oom(torch, error):
            raise InstrumentAnalysisError(f"demucs-error:{type(error).__name__}") from error
        try:
            torch.cuda.empty_cache()
            separator.update_parameter(segment=DEMUCS_RETRY_SEGMENT_SECONDS)
            origin, separated = separator.separate_tensor(tensor, sr=int(source_rate))
        except Exception as retry_error:  # noqa: BLE001 - second failure ends enhancement.
            reason = (
                "cuda-out-of-memory"
                if _is_cuda_oom(torch, retry_error)
                else f"demucs-error:{type(retry_error).__name__}"
            )
            raise InstrumentAnalysisError(reason) from retry_error

    mix = _tensor_to_numpy(origin)
    stem_arrays = {name: _tensor_to_numpy(value) for name, value in separated.items()}
    required = {"vocals", "drums", "bass", "other"}
    if not required.issubset(stem_arrays):
        raise InstrumentAnalysisError("invalid-model-output")
    expected_length = mix.shape[-1]
    for name in required:
        value = stem_arrays[name]
        if value.ndim != 2 or value.shape[-1] != expected_length or not np.all(np.isfinite(value)):
            raise InstrumentAnalysisError("invalid-model-output")
    return mix, {name: stem_arrays[name] for name in required}, int(separator.samplerate)


def _build_stem_frames(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    *,
    source_sample_rate: int,
    analysis_sample_rate: int,
    hop_length: int,
) -> list[StemActivityFrame]:
    if source_sample_rate <= 0 or analysis_sample_rate <= 0 or hop_length <= 0:
        return []
    mix_mono = _resample_mono(mix, source_sample_rate, analysis_sample_rate)
    stem_mono = {
        name: _resample_mono(value, source_sample_rate, analysis_sample_rate)
        for name, value in stems.items()
    }
    frame_count = 1 + mix_mono.size // hop_length
    if frame_count <= 0:
        return []
    mix_rms = _aligned_rms(mix_mono, hop_length, frame_count)
    stem_rms = {
        name: _aligned_rms(value, hop_length, frame_count)
        for name, value in stem_mono.items()
    }
    total_stem_rms = np.sum(np.vstack(list(stem_rms.values())), axis=0)
    mix_gate = _activity_gate(mix_rms)
    activities: dict[str, np.ndarray] = {}
    onsets: dict[str, np.ndarray] = {}
    for name, values in stem_rms.items():
        normalized = _normalize_envelope(values)
        share = np.divide(
            values,
            total_stem_rms,
            out=np.zeros_like(values),
            where=total_stem_rms > SILENCE_EPSILON,
        )
        activities[name] = np.clip(mix_gate * (normalized * 0.6 + share * 0.4), 0.0, 1.0)
        onset = librosa.onset.onset_strength(
            y=stem_mono[name],
            sr=analysis_sample_rate,
            hop_length=hop_length,
            n_fft=_frame_length(stem_mono[name]),
        )
        onsets[name] = _align_values(_normalize_envelope(onset), frame_count) * mix_gate

    times = librosa.frames_to_time(
        np.arange(frame_count),
        sr=analysis_sample_rate,
        hop_length=hop_length,
    )
    return [
        StemActivityFrame(
            time=round(float(times[index]), 6),
            vocals=_rounded(activities["vocals"][index]),
            drums=_rounded(activities["drums"][index]),
            bass=_rounded(activities["bass"][index]),
            other=_rounded(activities["other"][index]),
            vocal_onset=_rounded(onsets["vocals"][index]),
            drum_onset=_rounded(onsets["drums"][index]),
            bass_onset=_rounded(onsets["bass"][index]),
            accompaniment_onset=_rounded(onsets["other"][index]),
        )
        for index in range(frame_count)
    ]


def _classify_stems(
    mix: np.ndarray,
    other: np.ndarray,
    *,
    sample_rate: int,
    model_dir: Path,
    device: str,
    torch: Any,
) -> list[InstrumentClassificationWindow]:
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError as error:
        raise InstrumentAnalysisError("missing-dependency:transformers") from error
    try:
        extractor = AutoFeatureExtractor.from_pretrained(
            model_dir / "ast",
            local_files_only=True,
        )
        model = AutoModelForAudioClassification.from_pretrained(
            model_dir / "ast",
            local_files_only=True,
        )
        model.to(device)
        model.eval()
    except Exception as error:  # noqa: BLE001 - local model failures become stable reasons.
        raise InstrumentAnalysisError(f"classifier-load-error:{type(error).__name__}") from error

    mix_mono = _resample_mono(mix, sample_rate, AST_SAMPLE_RATE)
    other_mono = _resample_mono(other, sample_rate, AST_SAMPLE_RATE)
    mix_windows = _window_audio(mix_mono, AST_SAMPLE_RATE)
    other_windows = _window_audio(other_mono, AST_SAMPLE_RATE)
    if len(mix_windows) != len(other_windows) or not mix_windows:
        raise InstrumentAnalysisError("invalid-model-output")
    label_indexes = _taxonomy_label_indexes(getattr(model.config, "id2label", {}))
    if not any(label_indexes.values()):
        raise InstrumentAnalysisError("invalid-model-output")

    mix_scores = _predict_taxonomy_scores(
        [window for _, _, window in mix_windows],
        extractor=extractor,
        model=model,
        label_indexes=label_indexes,
        device=device,
        torch=torch,
    )
    other_scores = _predict_taxonomy_scores(
        [window for _, _, window in other_windows],
        extractor=extractor,
        model=model,
        label_indexes=label_indexes,
        device=device,
        torch=torch,
    )
    result: list[InstrumentClassificationWindow] = []
    for (start, end, mix_window), (_, _, other_window), mix_item, other_item in zip(
        mix_windows,
        other_windows,
        mix_scores,
        other_scores,
        strict=True,
    ):
        mix_gate = _window_gate(mix_window)
        other_gate = _relative_window_gate(other_window, mix_window)
        result.append(
            InstrumentClassificationWindow(
                start_time=round(start, 6),
                end_time=round(end, 6),
                mix_scores={key: _rounded(value * mix_gate) for key, value in mix_item.items()},
                other_scores={
                    key: _rounded(value * other_gate) for key, value in other_item.items()
                },
            )
        )
    return result


def _predict_taxonomy_scores(
    windows: list[np.ndarray],
    *,
    extractor: Any,
    model: Any,
    label_indexes: dict[str, list[int]],
    device: str,
    torch: Any,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for start in range(0, len(windows), AST_BATCH_SIZE):
        batch = windows[start : start + AST_BATCH_SIZE]
        inputs = extractor(
            batch,
            sampling_rate=AST_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        if probabilities.ndim != 2 or probabilities.shape[0] != len(batch):
            raise InstrumentAnalysisError("invalid-model-output")
        for row in probabilities:
            if not np.all(np.isfinite(row)):
                raise InstrumentAnalysisError("invalid-model-output")
            results.append(
                {
                    category: float(max((row[index] for index in indexes), default=0.0))
                    for category, indexes in label_indexes.items()
                }
            )
    return results


def _taxonomy_label_indexes(id2label: Any) -> dict[str, list[int]]:
    normalized = {int(index): str(label) for index, label in dict(id2label).items()}
    reverse = {label.casefold(): index for index, label in normalized.items()}
    return {
        category: [reverse[label.casefold()] for label in labels if label.casefold() in reverse]
        for category, labels in INSTRUMENT_TAXONOMY.items()
    }


def _window_audio(
    samples: np.ndarray,
    sample_rate: int,
) -> list[tuple[float, float, np.ndarray]]:
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or waveform.size == 0:
        return []
    window_size = max(1, round(AST_WINDOW_SECONDS * sample_rate))
    hop_size = max(1, round(AST_HOP_SECONDS * sample_rate))
    result: list[tuple[float, float, np.ndarray]] = []
    for start in range(0, waveform.size, hop_size):
        end = min(waveform.size, start + window_size)
        window = np.zeros(window_size, dtype=np.float32)
        window[: end - start] = waveform[start:end]
        result.append((start / sample_rate, end / sample_rate, window))
    return result


def _channels_first(samples: np.ndarray) -> np.ndarray:
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim == 1:
        return np.vstack([waveform, waveform])
    if waveform.ndim != 2:
        raise InstrumentAnalysisError("invalid-waveform")
    if waveform.shape[0] > waveform.shape[1] and waveform.shape[1] <= 8:
        waveform = waveform.T
    if waveform.shape[0] == 1:
        waveform = np.vstack([waveform[0], waveform[0]])
    if waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def _tensor_to_numpy(value: Any) -> np.ndarray:
    tensor = value.detach().cpu() if hasattr(value, "detach") else value
    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    result = np.asarray(array, dtype=np.float32)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    return result


def _resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    waveform = np.asarray(samples, dtype=np.float32)
    mono = np.mean(waveform, axis=0) if waveform.ndim == 2 else waveform.reshape(-1)
    if source_rate == target_rate:
        return mono
    return librosa.resample(mono, orig_sr=source_rate, target_sr=target_rate)


def _aligned_rms(samples: np.ndarray, hop_length: int, frame_count: int) -> np.ndarray:
    values = librosa.feature.rms(
        y=samples,
        frame_length=_frame_length(samples),
        hop_length=hop_length,
    )[0]
    return _align_values(np.maximum(0.0, np.nan_to_num(values, nan=0.0)), frame_count)


def _frame_length(samples: np.ndarray) -> int:
    return max(2, min(2048, int(np.asarray(samples).size)))


def _align_values(values: np.ndarray, frame_count: int) -> np.ndarray:
    result = np.zeros(frame_count, dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    copied = min(frame_count, array.size)
    result[:copied] = np.clip(
        np.nan_to_num(array[:copied], nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )
    return result


def _normalize_envelope(values: np.ndarray) -> np.ndarray:
    finite = np.maximum(0.0, np.nan_to_num(np.asarray(values, dtype=float), nan=0.0))
    positive = finite[finite > SILENCE_EPSILON]
    if positive.size == 0:
        return np.zeros_like(finite)
    scale = float(np.percentile(positive, 95))
    if scale <= SILENCE_EPSILON:
        return np.zeros_like(finite)
    return np.clip(finite / scale, 0.0, 1.0)


def _activity_gate(mix_rms: np.ndarray) -> np.ndarray:
    normalized = _normalize_envelope(mix_rms)
    return np.where(mix_rms > SILENCE_EPSILON, np.clip(normalized / 0.15, 0.0, 1.0), 0.0)


def _window_gate(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(np.asarray(samples, dtype=float)))))
    return float(np.clip(rms / 0.02, 0.0, 1.0))


def _relative_window_gate(candidate: np.ndarray, mix: np.ndarray) -> float:
    candidate_rms = float(np.sqrt(np.mean(np.square(np.asarray(candidate, dtype=float)))))
    mix_rms = float(np.sqrt(np.mean(np.square(np.asarray(mix, dtype=float)))))
    if mix_rms <= SILENCE_EPSILON:
        return 0.0
    return float(np.clip(candidate_rms / mix_rms, 0.0, 1.0))


def _is_cuda_oom(torch: Any, error: Exception) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(error, oom_type):
        return True
    return "out of memory" in str(error).casefold()


def _rounded(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 4)
