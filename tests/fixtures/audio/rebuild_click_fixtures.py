from pathlib import Path
import math
import struct
import wave


SAMPLE_RATE = 22050
BPM = 120.0
BEAT_INTERVAL_SECONDS = 60.0 / BPM
BEAT_COUNT = 16
CLICK_DURATION_SECONDS = 0.04
TAIL_SECONDS = 0.5
FIXTURES = {
    "click_4_4.wav": 0.25,
    "click_4_4_leadin.wav": 1.25,
}


def build_click_track(path: Path, *, first_beat_seconds: float) -> None:
    duration = first_beat_seconds + ((BEAT_COUNT - 1) * BEAT_INTERVAL_SECONDS) + TAIL_SECONDS
    samples = [0.0] * round(duration * SAMPLE_RATE)
    click_sample_count = round(CLICK_DURATION_SECONDS * SAMPLE_RATE)

    for beat_index in range(BEAT_COUNT):
        start = round(
            (first_beat_seconds + (beat_index * BEAT_INTERVAL_SECONDS)) * SAMPLE_RATE
        )
        downbeat = beat_index % 4 == 0
        amplitude = 0.9 if downbeat else 0.55
        frequency = 1760.0 if downbeat else 1100.0

        for click_index in range(click_sample_count):
            time = click_index / SAMPLE_RATE
            envelope = math.exp(-80.0 * time)
            value = amplitude * envelope * math.cos(2.0 * math.pi * frequency * time)
            sample_index = start + click_index
            if sample_index < len(samples):
                samples[sample_index] += value

    pcm = b"".join(
        struct.pack("<h", round(max(-1.0, min(1.0, sample)) * 32767)) for sample in samples
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def main() -> None:
    output_dir = Path(__file__).parent
    for filename, first_beat_seconds in FIXTURES.items():
        build_click_track(output_dir / filename, first_beat_seconds=first_beat_seconds)


if __name__ == "__main__":
    main()
