from pathlib import Path

import librosa
import numpy as np
from pydantic import BaseModel


class AudioAnalysisRaw(BaseModel):
    bpm: float
    beat_times: list[float]
    onset_times: list[float]
    onset_strengths: list[float]
    duration: float
    offset: float


def normalize_bpm(bpm: float) -> float:
    if bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm}")

    while bpm < 100:
        bpm *= 2
    while bpm > 220:
        bpm /= 2
    return round(bpm, 3)


def analyze_audio(input_path: Path) -> AudioAnalysisRaw:
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

    return AudioAnalysisRaw(
        bpm=normalize_bpm(tempo_value),
        beat_times=[float(value) for value in beat_times],
        onset_times=onset_times_list,
        onset_strengths=[float(value) for value in onset_env.tolist()],
        duration=duration,
        offset=offset,
    )
