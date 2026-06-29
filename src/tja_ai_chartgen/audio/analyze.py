from pathlib import Path
from typing import Any

import librosa
import numpy as np
from pydantic import BaseModel, Field


class AudioAnalysisRaw(BaseModel):
    bpm: float
    beat_times: list[float]
    onset_times: list[float]
    onset_strengths: list[float]
    duration: float
    offset: float
    sample_rate: int | None = None
    hop_length: int = 512
    downbeat_times: list[float] = Field(default_factory=list)
    beat_numbers: list[int] = Field(default_factory=list)
    time_signature: str = "4/4"
    analyzer: str = "librosa"


def normalize_bpm(bpm: float) -> float:
    if bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm}")

    while bpm < 100:
        bpm *= 2
    while bpm > 220:
        bpm /= 2
    return round(bpm, 3)


def apply_analysis_overrides(
    raw: AudioAnalysisRaw,
    bpm: float | None = None,
    offset: float | None = None,
) -> AudioAnalysisRaw:
    updates: dict[str, float] = {}

    if bpm is not None:
        if bpm <= 0:
            raise ValueError(f"BPM must be positive, got {bpm}")
        updates["bpm"] = round(bpm, 3)

    if offset is not None:
        updates["offset"] = offset

    if not updates:
        return raw

    return raw.model_copy(update=updates)


def analyze_audio(input_path: Path, use_beatnet: bool = False) -> AudioAnalysisRaw:
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    y, sr = librosa.load(str(input_path), sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    onset_times_list = [float(value) for value in onset_times]

    offset = float(beat_times[0]) if beat_times else 0.0
    raw = AudioAnalysisRaw(
        bpm=normalize_bpm(tempo_value),
        beat_times=[float(value) for value in beat_times],
        onset_times=onset_times_list,
        onset_strengths=[float(value) for value in onset_env.tolist()],
        duration=duration,
        offset=offset,
        sample_rate=int(sr),
    )

    if not use_beatnet:
        return raw

    return enhance_with_beatnet(input_path, raw)


def enhance_with_beatnet(input_path: Path, raw: AudioAnalysisRaw) -> AudioAnalysisRaw:
    try:
        from BeatNet.BeatNet import BeatNet
    except ImportError:
        return raw

    try:
        estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
        output = estimator.process(str(input_path))
    except Exception:  # noqa: BLE001 - BeatNet is an optional enhancement.
        return raw

    return merge_beatnet_output(raw, output)


def merge_beatnet_output(raw: AudioAnalysisRaw, output: Any) -> AudioAnalysisRaw:
    rows = np.asarray(output)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
        return raw

    beat_times: list[float] = []
    beat_numbers: list[int] = []
    for row in rows:
        time = float(row[0])
        beat_number = int(round(float(row[1])))
        if not np.isfinite(time) or beat_number < 1:
            continue
        beat_times.append(time)
        beat_numbers.append(beat_number)

    if not beat_times:
        return raw

    downbeat_times = [time for time, number in zip(beat_times, beat_numbers, strict=True) if number == 1]
    updates: dict[str, Any] = {
        "beat_times": beat_times,
        "beat_numbers": beat_numbers,
        "downbeat_times": downbeat_times,
        "offset": downbeat_times[0] if downbeat_times else beat_times[0],
        "time_signature": estimate_time_signature(beat_numbers),
        "analyzer": "beatnet+librosa",
    }

    positive_intervals = [
        later - earlier
        for earlier, later in zip(beat_times, beat_times[1:], strict=False)
        if later > earlier
    ]
    if positive_intervals:
        updates["bpm"] = normalize_bpm(60.0 / float(np.mean(positive_intervals)))

    return raw.model_copy(update=updates)


def estimate_time_signature(beat_numbers: list[int]) -> str:
    if not beat_numbers:
        return "4/4"

    max_beat_number = max(beat_numbers)
    if max_beat_number == 3:
        return "3/4"
    if max_beat_number == 6:
        return "6/8"
    return "4/4"
