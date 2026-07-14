from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
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
GROUND_TRUTH_SCHEMA_VERSION = 1
GROUND_TRUTH_SCHEMA_FILENAME = "ground_truth.schema.json"
FIXTURES = {
    "click_4_4.wav": 0.25,
    "click_4_4_leadin.wav": 1.25,
}
QUALITY_FIXTURES = {
    "sparse_120.wav": (120.0, 0.5, "sparse"),
    "dense_180.wav": (180.0, 0.5, "dense"),
}
TRANSIENT_NOISE_FIXTURE = "transient_noise_intro_120.wav"
RESOLUTION_FIXTURES = {
    "straight_120.wav": "straight",
    "triplet_120.wav": "triplet",
    "mixed_120.wav": "mixed",
}
STRUCTURE_FIXTURE = "structure_build_up_120.wav"
GROUND_TRUTH_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": GROUND_TRUTH_SCHEMA_FILENAME,
    "title": "Synthetic audio fixture ground truth",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "audio",
        "bpm",
        "time_signature",
        "duration",
        "first_downbeat",
        "onsets",
        "strong_onsets",
        "beats",
        "downbeats",
        "low_band_onsets",
        "high_band_onsets",
        "silent_ranges",
        "fill_ranges",
        "sections",
    ],
    "properties": {
        "schema_version": {"const": GROUND_TRUTH_SCHEMA_VERSION},
        "audio": {"type": "string", "pattern": "^[^/\\\\]+\\.wav$"},
        "bpm": {"type": "number", "exclusiveMinimum": 0},
        "time_signature": {"enum": ["4/4", "3/4", "6/8"]},
        "duration": {"type": "number", "exclusiveMinimum": 0},
        "first_downbeat": {"type": "number", "minimum": 0},
        "onsets": {"$ref": "#/$defs/times"},
        "strong_onsets": {"$ref": "#/$defs/times"},
        "beats": {"$ref": "#/$defs/times"},
        "downbeats": {"$ref": "#/$defs/times"},
        "low_band_onsets": {"$ref": "#/$defs/times"},
        "high_band_onsets": {"$ref": "#/$defs/times"},
        "silent_ranges": {"$ref": "#/$defs/ranges"},
        "fill_ranges": {"$ref": "#/$defs/ranges"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "role"],
                "properties": {
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "exclusiveMinimum": 0},
                    "role": {"type": "string", "minLength": 1},
                },
            },
        },
    },
    "$defs": {
        "times": {
            "type": "array",
            "items": {"type": "number", "minimum": 0},
        },
        "ranges": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "number", "minimum": 0},
                    {"type": "number", "exclusiveMinimum": 0},
                ],
                "items": False,
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
}


@dataclass(frozen=True)
class FixtureSpec:
    filename: str
    bpm: float
    duration: float
    events: tuple[tuple[float, bool], ...]
    beats: tuple[float, ...]
    time_signature: str = "4/4"
    impulses: tuple[tuple[float, float], ...] = ()
    frequency_overrides: tuple[tuple[float, float, float], ...] = ()
    sustained_tones: tuple[tuple[float, float, float, float], ...] = ()
    strong_onsets: tuple[float, ...] = ()
    low_band_onsets: tuple[float, ...] = ()
    high_band_onsets: tuple[float, ...] = ()
    silent_ranges: tuple[tuple[float, float], ...] = ()
    fill_ranges: tuple[tuple[float, float], ...] = ()
    sections: tuple[dict[str, object], ...] = field(default_factory=tuple)


def _rounded(value: float) -> float:
    return round(value, 9)


def _click_track_spec(filename: str, *, first_beat_seconds: float) -> FixtureSpec:
    duration = first_beat_seconds + ((BEAT_COUNT - 1) * BEAT_INTERVAL_SECONDS) + TAIL_SECONDS
    events = tuple(
        (first_beat_seconds + beat_index * BEAT_INTERVAL_SECONDS, beat_index % 4 == 0)
        for beat_index in range(BEAT_COUNT)
    )
    return FixtureSpec(
        filename=filename,
        bpm=BPM,
        duration=duration,
        events=events,
        beats=tuple(event_time for event_time, _ in events),
        silent_ranges=((0.0, first_beat_seconds),),
    )


def _quality_track_spec(
    filename: str,
    *,
    bpm: float,
    first_beat_seconds: float,
    pattern: str,
) -> FixtureSpec:
    beat_interval = 60.0 / bpm
    bar_duration = beat_interval * 4
    bar_count = 8
    events: list[tuple[float, bool]] = []
    beats: list[float] = []

    for bar_index in range(bar_count):
        bar_start = first_beat_seconds + bar_index * bar_duration
        beats.extend(bar_start + beat_index * beat_interval for beat_index in range(4))
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
    return FixtureSpec(
        filename=filename,
        bpm=bpm,
        duration=duration,
        events=tuple(events),
        beats=tuple(beats),
        silent_ranges=((0.0, first_beat_seconds),),
    )


def _resolution_track_spec(filename: str, *, pattern: str) -> FixtureSpec:
    first_beat_seconds = 0.5
    beat_interval = 60.0 / 120.0
    bar_count = 4
    events: list[tuple[float, bool]] = []
    beats: list[float] = []
    for bar_index in range(bar_count):
        bar_start = first_beat_seconds + bar_index * beat_interval * 4
        subdivisions = (
            4
            if pattern == "straight" or (pattern == "mixed" and bar_index % 2 == 0)
            else 3
        )
        if pattern not in {"straight", "triplet", "mixed"}:
            raise ValueError(f"Unknown resolution fixture pattern: {pattern}")
        for beat_index in range(4):
            beat_start = bar_start + beat_index * beat_interval
            beats.append(beat_start)
            for subdivision_index in range(subdivisions):
                events.append(
                    (
                        beat_start + subdivision_index * beat_interval / subdivisions,
                        beat_index == 0 and subdivision_index == 0,
                    )
                )
    duration = first_beat_seconds + bar_count * beat_interval * 4 + TAIL_SECONDS
    return FixtureSpec(
        filename=filename,
        bpm=120.0,
        duration=duration,
        events=tuple(events),
        beats=tuple(beats),
        silent_ranges=((0.0, first_beat_seconds),),
    )


def _structure_track_spec() -> FixtureSpec:
    first_beat_seconds = 0.5
    beat_interval = 60.0 / 120.0
    bar_duration = beat_interval * 4
    subdivisions_by_bar = [1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 1, 1]
    events: list[tuple[float, bool]] = []
    beats: list[float] = []
    for bar_index, subdivisions in enumerate(subdivisions_by_bar):
        bar_start = first_beat_seconds + bar_index * bar_duration
        for beat_index in range(4):
            beat_start = bar_start + beat_index * beat_interval
            beats.append(beat_start)
            for subdivision_index in range(subdivisions):
                events.append(
                    (
                        beat_start + subdivision_index * beat_interval / subdivisions,
                        beat_index == 0 and subdivision_index == 0,
                    )
                )
    duration = first_beat_seconds + len(subdivisions_by_bar) * bar_duration + TAIL_SECONDS
    sections = (
        {"start": first_beat_seconds, "end": first_beat_seconds + 5 * bar_duration, "role": "stable"},
        {
            "start": first_beat_seconds + 5 * bar_duration,
            "end": first_beat_seconds + 8 * bar_duration,
            "role": "build_up",
        },
        {
            "start": first_beat_seconds + 8 * bar_duration,
            "end": first_beat_seconds + 10 * bar_duration,
            "role": "peak",
        },
        {
            "start": first_beat_seconds + 10 * bar_duration,
            "end": first_beat_seconds + 12 * bar_duration,
            "role": "drop",
        },
    )
    return FixtureSpec(
        filename=STRUCTURE_FIXTURE,
        bpm=120.0,
        duration=duration,
        events=tuple(events),
        beats=tuple(beats),
        silent_ranges=((0.0, first_beat_seconds),),
        sections=sections,
    )


def _transient_noise_intro_track_spec() -> FixtureSpec:
    first_music_seconds = 2.25
    events = tuple(
        (first_music_seconds + beat_index * BEAT_INTERVAL_SECONDS, beat_index % 4 == 0)
        for beat_index in range(8)
    )
    duration = first_music_seconds + (8 * BEAT_INTERVAL_SECONDS) + TAIL_SECONDS
    return FixtureSpec(
        filename=TRANSIENT_NOISE_FIXTURE,
        bpm=BPM,
        duration=duration,
        events=events,
        beats=tuple(event_time for event_time, _ in events),
        impulses=((0.5, 0.1), (1.5, 0.1)),
        silent_ranges=((0.0, first_music_seconds),),
    )


def _syncopated_track_spec() -> FixtureSpec:
    first_downbeat = 0.5
    beat_interval = 0.5
    bar_duration = beat_interval * 4
    bar_count = 8
    offsets = (0.0, 0.375, 0.75, 1.25, 1.75)
    events = tuple(
        (first_downbeat + bar_index * bar_duration + offset, offset == 0.0)
        for bar_index in range(bar_count)
        for offset in offsets
    )
    beats = tuple(
        first_downbeat + beat_index * beat_interval for beat_index in range(bar_count * 4)
    )
    return FixtureSpec(
        filename="syncopated_120.wav",
        bpm=120.0,
        duration=first_downbeat + bar_count * bar_duration + TAIL_SECONDS,
        events=events,
        beats=beats,
        silent_ranges=((0.0, first_downbeat),),
    )


def _pickup_track_spec() -> FixtureSpec:
    first_downbeat = 1.0
    beat_interval = 0.5
    bar_count = 8
    beats = tuple(
        first_downbeat + beat_index * beat_interval for beat_index in range(bar_count * 4)
    )
    pickup_events = ((0.5, False), (0.75, False))
    metered_events = tuple(
        (beat_time, beat_index % 4 == 0) for beat_index, beat_time in enumerate(beats)
    )
    return FixtureSpec(
        filename="pickup_120.wav",
        bpm=120.0,
        duration=first_downbeat + bar_count * 4 * beat_interval + TAIL_SECONDS,
        events=pickup_events + metered_events,
        beats=beats,
        silent_ranges=((0.0, pickup_events[0][0]),),
    )


def _band_attack_track_spec() -> FixtureSpec:
    first_downbeat = 0.5
    beat_interval = 0.5
    bar_count = 8
    beats = tuple(
        first_downbeat + beat_index * beat_interval for beat_index in range(bar_count * 4)
    )
    low_onsets = beats
    high_onsets = tuple(beat_time + beat_interval / 2.0 for beat_time in beats)
    events = tuple(
        sorted(
            [
                *((time, index % 4 == 0) for index, time in enumerate(low_onsets)),
                *((time, False) for time in high_onsets),
            ]
        )
    )
    frequency_overrides = tuple(
        (time, 180.0, 0.9 if index % 4 == 0 else 0.65)
        for index, time in enumerate(low_onsets)
    ) + tuple((time, 4200.0, 0.55) for time in high_onsets)
    return FixtureSpec(
        filename="band_attacks_120.wav",
        bpm=120.0,
        duration=first_downbeat + bar_count * 4 * beat_interval + TAIL_SECONDS,
        events=events,
        beats=beats,
        frequency_overrides=frequency_overrides,
        low_band_onsets=low_onsets,
        high_band_onsets=high_onsets,
        silent_ranges=((0.0, first_downbeat),),
    )


def _harmonic_sparse_track_spec() -> FixtureSpec:
    first_downbeat = 0.5
    beat_interval = 0.5
    bar_duration = beat_interval * 4
    bar_count = 8
    beats = tuple(
        first_downbeat + beat_index * beat_interval for beat_index in range(bar_count * 4)
    )
    events = tuple(
        (
            first_downbeat + bar_index * bar_duration + beat_index * beat_interval,
            beat_index == 0,
        )
        for bar_index in range(bar_count)
        for beat_index in (0, 2)
    )
    music_end = first_downbeat + bar_count * bar_duration
    return FixtureSpec(
        filename="harmonic_sparse_120.wav",
        bpm=120.0,
        duration=music_end + TAIL_SECONDS,
        events=events,
        beats=beats,
        sustained_tones=(
            (first_downbeat, music_end, 220.0, 0.08),
            (first_downbeat, music_end, 330.0, 0.05),
        ),
        silent_ranges=((0.0, first_downbeat),),
        sections=({"start": first_downbeat, "end": music_end, "role": "stable"},),
    )


def _fill_burst_track_spec() -> FixtureSpec:
    first_downbeat = 0.5
    beat_interval = 0.5
    bar_duration = beat_interval * 4
    bar_count = 8
    beats = tuple(
        first_downbeat + beat_index * beat_interval for beat_index in range(bar_count * 4)
    )
    event_map = {
        _rounded(beat_time): beat_index % 4 == 0
        for beat_index, beat_time in enumerate(beats)
    }
    fill_ranges: list[tuple[float, float]] = []
    for bar_index in (3, 7):
        fill_start = first_downbeat + bar_index * bar_duration + 3 * beat_interval
        fill_end = fill_start + beat_interval
        fill_ranges.append((fill_start, fill_end))
        for subdivision_index in range(4):
            event_map[_rounded(fill_start + subdivision_index * beat_interval / 4)] = False
    events = tuple((time, event_map[time]) for time in sorted(event_map))
    return FixtureSpec(
        filename="fill_burst_120.wav",
        bpm=120.0,
        duration=first_downbeat + bar_count * bar_duration + TAIL_SECONDS,
        events=events,
        beats=beats,
        silent_ranges=((0.0, first_downbeat),),
        fill_ranges=tuple(fill_ranges),
    )


def _meter_track_spec(time_signature: str) -> FixtureSpec:
    first_downbeat = 0.5
    beat_interval = 0.5
    bar_count = 8
    if time_signature == "3/4":
        quarter_notes_per_bar = 3
        onset_interval = beat_interval
        onsets_per_bar = 3
        filename = "meter_3_4_120.wav"
    elif time_signature == "6/8":
        quarter_notes_per_bar = 3
        onset_interval = beat_interval / 2.0
        onsets_per_bar = 6
        filename = "meter_6_8_120.wav"
    else:
        raise ValueError(f"Unsupported meter fixture: {time_signature}")

    bar_duration = quarter_notes_per_bar * beat_interval
    beats = tuple(
        first_downbeat + beat_index * beat_interval
        for beat_index in range(bar_count * quarter_notes_per_bar)
    )
    events = tuple(
        (
            first_downbeat + bar_index * bar_duration + onset_index * onset_interval,
            onset_index == 0,
        )
        for bar_index in range(bar_count)
        for onset_index in range(onsets_per_bar)
    )
    downbeats = tuple(event_time for event_time, downbeat in events if downbeat)
    compound_accents = (
        tuple(
            first_downbeat + bar_index * bar_duration + 3 * onset_interval
            for bar_index in range(bar_count)
        )
        if time_signature == "6/8"
        else ()
    )
    strong_onsets = tuple(sorted(downbeats + compound_accents))
    frequency_overrides = tuple(
        (event_time, 1100.0, 0.75) for event_time in compound_accents
    )
    return FixtureSpec(
        filename=filename,
        bpm=120.0,
        duration=first_downbeat + bar_count * bar_duration + TAIL_SECONDS,
        events=events,
        beats=beats,
        time_signature=time_signature,
        frequency_overrides=frequency_overrides,
        strong_onsets=strong_onsets,
        silent_ranges=((0.0, first_downbeat),),
    )


def _tempo_ambiguity_track_spec() -> FixtureSpec:
    first_downbeat = 0.5
    beat_interval = 0.5
    bar_count = 8
    beats = tuple(
        first_downbeat + beat_index * beat_interval for beat_index in range(bar_count * 4)
    )
    events = tuple(
        (beat_time, beat_index % 4 == 0) for beat_index, beat_time in enumerate(beats)
    )
    strong_onsets = tuple(
        beat_time for beat_index, beat_time in enumerate(beats) if beat_index % 2 == 0
    )
    frequency_overrides = tuple(
        (
            beat_time,
            1760.0 if beat_index % 4 == 0 else 1100.0,
            0.9 if beat_index % 4 == 0 else 0.75 if beat_index % 2 == 0 else 0.22,
        )
        for beat_index, beat_time in enumerate(beats)
    )
    return FixtureSpec(
        filename="tempo_ambiguity_120.wav",
        bpm=120.0,
        duration=first_downbeat + bar_count * 4 * beat_interval + TAIL_SECONDS,
        events=events,
        beats=beats,
        frequency_overrides=frequency_overrides,
        strong_onsets=strong_onsets,
        silent_ranges=((0.0, first_downbeat),),
    )


def build_fixture_specs() -> list[FixtureSpec]:
    specs = [
        _click_track_spec(filename, first_beat_seconds=first_beat_seconds)
        for filename, first_beat_seconds in FIXTURES.items()
    ]
    specs.extend(
        _quality_track_spec(
            filename,
            bpm=bpm,
            first_beat_seconds=first_beat_seconds,
            pattern=pattern,
        )
        for filename, (bpm, first_beat_seconds, pattern) in QUALITY_FIXTURES.items()
    )
    specs.append(_transient_noise_intro_track_spec())
    specs.extend(
        _resolution_track_spec(filename, pattern=pattern)
        for filename, pattern in RESOLUTION_FIXTURES.items()
    )
    specs.append(_structure_track_spec())
    specs.extend(
        [
            _syncopated_track_spec(),
            _pickup_track_spec(),
            _band_attack_track_spec(),
            _harmonic_sparse_track_spec(),
            _fill_burst_track_spec(),
            _meter_track_spec("3/4"),
            _meter_track_spec("6/8"),
            _tempo_ambiguity_track_spec(),
        ]
    )
    return specs


def _ground_truth(spec: FixtureSpec) -> dict[str, object]:
    onsets = [_rounded(event_time) for event_time, _ in spec.events]
    downbeats = [_rounded(event_time) for event_time, downbeat in spec.events if downbeat]
    strong_onsets = (
        [_rounded(value) for value in spec.strong_onsets]
        if spec.strong_onsets
        else downbeats
    )
    return {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "audio": spec.filename,
        "bpm": spec.bpm,
        "time_signature": spec.time_signature,
        "duration": _rounded(spec.duration),
        "first_downbeat": downbeats[0],
        "onsets": onsets,
        "strong_onsets": strong_onsets,
        "beats": [_rounded(value) for value in spec.beats],
        "downbeats": downbeats,
        "low_band_onsets": [_rounded(value) for value in spec.low_band_onsets],
        "high_band_onsets": [_rounded(value) for value in spec.high_band_onsets],
        "silent_ranges": [
            [_rounded(start), _rounded(end)] for start, end in spec.silent_ranges
        ],
        "fill_ranges": [
            [_rounded(start), _rounded(end)] for start, end in spec.fill_ranges
        ],
        "sections": [
            {
                "start": _rounded(float(section["start"])),
                "end": _rounded(float(section["end"])),
                "role": section["role"],
            }
            for section in spec.sections
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _selected_fixture_specs(fixture_names: set[str] | None) -> list[FixtureSpec]:
    specs = build_fixture_specs()
    if fixture_names is None:
        return specs
    available_names = {spec.filename for spec in specs}
    unknown_names = fixture_names - available_names
    if unknown_names:
        raise ValueError(f"Unknown fixture names: {', '.join(sorted(unknown_names))}")
    return [spec for spec in specs if spec.filename in fixture_names]


def write_ground_truth_files(
    output_dir: Path,
    *,
    fixture_names: set[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / GROUND_TRUTH_SCHEMA_FILENAME
    if fixture_names is None or not schema_path.exists():
        _write_json(schema_path, GROUND_TRUTH_SCHEMA)
    for spec in _selected_fixture_specs(fixture_names):
        _write_json(output_dir / f"{Path(spec.filename).stem}.events.json", _ground_truth(spec))


def build_all_fixtures(
    output_dir: Path,
    *,
    fixture_names: set[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in _selected_fixture_specs(fixture_names):
        _write_click_events(
            output_dir / spec.filename,
            duration=spec.duration,
            events=list(spec.events),
            impulses=list(spec.impulses),
            frequency_overrides=list(spec.frequency_overrides),
            sustained_tones=list(spec.sustained_tones),
        )
    write_ground_truth_files(output_dir, fixture_names=fixture_names)


def _write_click_events(
    path: Path,
    *,
    duration: float,
    events: list[tuple[float, bool]],
    impulses: list[tuple[float, float]] | None = None,
    frequency_overrides: list[tuple[float, float, float]] | None = None,
    sustained_tones: list[tuple[float, float, float, float]] | None = None,
) -> None:
    samples = [0.0] * round(duration * SAMPLE_RATE)
    click_sample_count = round(CLICK_DURATION_SECONDS * SAMPLE_RATE)
    overrides = {
        _rounded(event_time): (frequency, amplitude)
        for event_time, frequency, amplitude in frequency_overrides or []
    }

    for start_time, end_time, frequency, amplitude in sustained_tones or []:
        start = max(0, round(start_time * SAMPLE_RATE))
        end = min(len(samples), round(end_time * SAMPLE_RATE))
        fade_sample_count = max(1, round(0.05 * SAMPLE_RATE))
        for sample_index in range(start, end):
            relative_index = sample_index - start
            remaining_index = end - sample_index - 1
            envelope = min(
                1.0,
                relative_index / fade_sample_count,
                remaining_index / fade_sample_count,
            )
            time = relative_index / SAMPLE_RATE
            samples[sample_index] += (
                amplitude * envelope * math.sin(2.0 * math.pi * frequency * time)
            )

    for event_time, downbeat in events:
        start = round(event_time * SAMPLE_RATE)
        amplitude = 0.9 if downbeat else 0.55
        frequency = 1760.0 if downbeat else 1100.0
        frequency, amplitude = overrides.get(
            _rounded(event_time),
            (frequency, amplitude),
        )

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--ground-truth-only", action="store_true")
    parser.add_argument("--fixture", action="append", dest="fixture_names")
    args = parser.parse_args()
    fixture_names = set(args.fixture_names) if args.fixture_names else None
    if args.ground_truth_only:
        write_ground_truth_files(args.output_dir, fixture_names=fixture_names)
    else:
        build_all_fixtures(args.output_dir, fixture_names=fixture_names)


if __name__ == "__main__":
    main()
