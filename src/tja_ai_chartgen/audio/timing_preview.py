"""Short, offline audio/beat/chart overlays for timing A/B inspection."""

import subprocess
import sys
import wave
from array import array
from math import ceil, pi, sin
from pathlib import Path

from tja_ai_chartgen.tja.model import ChartBar, SongAnalysis

SAMPLE_RATE = 22050
PREVIEW_SECONDS = 12.0


def write_timing_previews(
    audio: Path, output_dir: Path, analysis: SongAnalysis, report: dict,
    chart: list[ChartBar] | None = None,
) -> list[dict]:
    if chart is not None and len(chart) != len(analysis.bars):
        raise ValueError("Preview chart must have the same bar count as analysis.")
    duration = report["duration_seconds"]
    starts = sorted({0.0, max(0.0, (duration - PREVIEW_SECONDS) / 2),
                     max(0.0, duration - PREVIEW_SECONDS)})
    clocks = [("current", analysis.bpm, analysis.offset)]
    if report["calibration"] is not None:
        fit = report["calibration"]
        clocks.append(("calibrated", fit["bpm"], fit["offset"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    for index, start in enumerate(starts):
        decoded = subprocess.run([
            "ffmpeg", "-v", "error", "-ss", str(start), "-i", str(audio),
            "-t", str(PREVIEW_SECONDS), "-vn", "-f", "s16le", "-ac", "1",
            "-ar", str(SAMPLE_RATE), "pipe:1",
        ], capture_output=True)
        if decoded.returncode:
            raise RuntimeError("Preview audio decoding failed: " + decoded.stderr.decode(errors="replace"))
        source = array("h", decoded.stdout)
        if sys.byteorder != "little":
            source.byteswap()
        if not source:
            continue  # Analysis may include a padded trailing bar after the audio ends.
        end = start + len(source) / SAMPLE_RATE
        for name, bpm, offset in clocks:
            samples = [sample * .5 for sample in source]
            first = ceil((start - offset) * bpm / 60)
            last = ceil((end - offset) * bpm / 60)
            for beat in range(first, last):
                _click(samples, offset + beat * 60 / bpm - start, 1400, .12)
            if chart is not None:
                for feature, bar in zip(analysis.bars, chart):
                    if not bar.notes:
                        continue
                    first_beat = (feature.start_time - analysis.offset) * analysis.bpm / 60
                    beat_span = (feature.end_time - feature.start_time) * analysis.bpm / 60
                    for position, note in enumerate(bar.notes):
                        if note not in "1234":
                            continue
                        time = offset + (first_beat + position / len(bar.notes) * beat_span) * 60 / bpm
                        _click(samples, time - start, 500 if note in "13" else 850, .22)
            encoded = array("h", (max(-32768, min(32767, round(value))) for value in samples))
            if sys.byteorder != "little":
                encoded.byteswap()
            path = output_dir / f"timing_{index + 1}_{name}.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(SAMPLE_RATE)
                stream.writeframes(encoded.tobytes())
            previews.append({"file": path.name, "clock": name, "start_seconds": start,
                             "end_seconds": end, "includes_chart": chart is not None})
    return previews


def _click(samples: list[float], time: float, frequency: float, gain: float) -> None:
    start = round(time * SAMPLE_RATE)
    if start < 0 or start >= len(samples):
        return
    length = round(.02 * SAMPLE_RATE)
    for i in range(min(length, len(samples) - start)):
        samples[start + i] += 32767 * gain * (1 - i / length) ** 2 * sin(2 * pi * frequency * i / SAMPLE_RATE)
