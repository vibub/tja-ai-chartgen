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
QUALITY_FIXTURES = {
    "sparse_120.wav": (120.0, 0.5, "sparse"),
    "dense_180.wav": (180.0, 0.5, "dense"),
}
TRANSIENT_NOISE_FIXTURE = "transient_noise_intro_120.wav"


def build_click_track(path: Path, *, first_beat_seconds: float) -> None:
    duration = first_beat_seconds + ((BEAT_COUNT - 1) * BEAT_INTERVAL_SECONDS) + TAIL_SECONDS
    events = [
        (first_beat_seconds + beat_index * BEAT_INTERVAL_SECONDS, beat_index % 4 == 0)
        for beat_index in range(BEAT_COUNT)
    ]
    _write_click_events(path, duration=duration, events=events)


def build_quality_track(
    path: Path,
    *,
    bpm: float,
    first_beat_seconds: float,
    pattern: str,
) -> None:
    beat_interval = 60.0 / bpm
    bar_duration = beat_interval * 4
    bar_count = 8
    events: list[tuple[float, bool]] = []

    for bar_index in range(bar_count):
        bar_start = first_beat_seconds + bar_index * bar_duration
        if pattern == "sparse":
            beat_indexes = (0,) if bar_index in {2, 4} else (0, 1, 2, 3)
            for beat_index in beat_indexes:
                events.append((bar_start + beat_index * beat_interval, beat_index == 0))
            continue

        if pattern != "dense":
            raise ValueError(f"Unknown quality fixture pattern: {pattern}")
        if bar_index == 4:
            events.append((bar_start, True))
            continue
        subdivision = 4 if bar_index in {2, 6} else 2
        for subdivision_index in range(4 * subdivision):
            event_time = bar_start + subdivision_index * beat_interval / subdivision
            events.append((event_time, subdivision_index == 0))

    duration = first_beat_seconds + bar_count * bar_duration + TAIL_SECONDS
    _write_click_events(path, duration=duration, events=events)


def build_transient_noise_intro_track(path: Path) -> None:
    first_music_seconds = 2.25
    events = [
        (first_music_seconds + beat_index * BEAT_INTERVAL_SECONDS, beat_index % 4 == 0)
        for beat_index in range(8)
    ]
    duration = first_music_seconds + (8 * BEAT_INTERVAL_SECONDS) + TAIL_SECONDS
    _write_click_events(
        path,
        duration=duration,
        events=events,
        impulses=[(0.5, 0.1), (1.5, 0.1)],
    )


def _write_click_events(
    path: Path,
    *,
    duration: float,
    events: list[tuple[float, bool]],
    impulses: list[tuple[float, float]] | None = None,
) -> None:
    samples = [0.0] * round(duration * SAMPLE_RATE)
    click_sample_count = round(CLICK_DURATION_SECONDS * SAMPLE_RATE)

    for event_time, downbeat in events:
        start = round(event_time * SAMPLE_RATE)
        amplitude = 0.9 if downbeat else 0.55
        frequency = 1760.0 if downbeat else 1100.0

        for click_index in range(click_sample_count):
            time = click_index / SAMPLE_RATE
            envelope = math.exp(-80.0 * time)
            value = amplitude * envelope * math.cos(2.0 * math.pi * frequency * time)
            sample_index = start + click_index
            if sample_index < len(samples):
                samples[sample_index] += value

    for event_time, amplitude in impulses or []:
        sample_index = round(event_time * SAMPLE_RATE)
        if 0 <= sample_index < len(samples):
            samples[sample_index] += amplitude

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
    for filename, (bpm, first_beat_seconds, pattern) in QUALITY_FIXTURES.items():
        build_quality_track(
            output_dir / filename,
            bpm=bpm,
            first_beat_seconds=first_beat_seconds,
            pattern=pattern,
        )
    build_transient_noise_intro_track(output_dir / TRANSIENT_NOISE_FIXTURE)


if __name__ == "__main__":
    main()
