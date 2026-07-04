from pathlib import Path
from typing import Any

from tja_ai_chartgen.audio.analyze import analyze_audio
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.meter import validate_time_signature
from tja_ai_chartgen.features.sections import assign_sections
from tja_ai_chartgen.tja.model import ChartBar
from tja_ai_chartgen.tja.writer import read_tja_text

DEFAULT_REFERENCE_EXAMPLES_DIR = Path("D:/Downloads/examples")
DEFAULT_REFERENCE_BARS_PER_EXAMPLE = 32
REFERENCE_AUDIO_EXTENSIONS = (".ogg", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus")
REFERENCE_ALLOWED_NOTES = set("01234578")


def build_reference_examples(
    examples_dir: Path = DEFAULT_REFERENCE_EXAMPLES_DIR,
    *,
    max_bars_per_example: int = DEFAULT_REFERENCE_BARS_PER_EXAMPLE,
    use_beatnet: bool = False,
    work_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if max_bars_per_example < 1 or not examples_dir.exists():
        return []

    examples: list[dict[str, Any]] = []
    for tja_path in sorted(examples_dir.glob("*.tja")):
        audio_path = _matching_audio_path(tja_path)
        if audio_path is None:
            continue
        try:
            examples.append(
                _build_reference_example(
                    tja_path=tja_path,
                    audio_path=audio_path,
                    max_bars=max_bars_per_example,
                    use_beatnet=use_beatnet,
                    work_dir=work_dir,
                )
            )
        except Exception:  # noqa: BLE001 - Reference examples should not block chart generation.
            continue

    return examples


def _build_reference_example(
    *,
    tja_path: Path,
    audio_path: Path,
    max_bars: int,
    use_beatnet: bool,
    work_dir: Path | None,
) -> dict[str, Any]:
    metadata, chart_bars = _parse_reference_tja(tja_path)
    processed_audio = _prepare_reference_audio(audio_path, work_dir=work_dir)
    raw = analyze_audio(processed_audio, use_beatnet=use_beatnet)
    bars = assign_sections(build_bar_features(raw, max_bars=max_bars))
    paired_count = min(len(bars), len(chart_bars), max_bars)

    return {
        "title": metadata.get("TITLE", tja_path.stem),
        "artist": metadata.get("SUBTITLE", "").removeprefix("--") or None,
        "course": metadata.get("COURSE"),
        "level": _optional_int(metadata.get("LEVEL")),
        "bpm": _optional_float(metadata.get("BPM")),
        "offset": _optional_float(metadata.get("OFFSET")),
        "source_tja": str(tja_path),
        "source_audio": str(audio_path),
        "processed_audio": str(processed_audio),
        "bars": [
            {
                "bar": index + 1,
                "audio_features": bars[index].model_dump(),
                "reference_notes": chart_bars[index].notes,
                "reference_note_resolution": len(chart_bars[index].notes),
                "balloon_counts": chart_bars[index].balloon_counts,
            }
            for index in range(paired_count)
        ],
    }


def _prepare_reference_audio(audio_path: Path, *, work_dir: Path | None) -> Path:
    if work_dir is None:
        return audio_path

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / f"{audio_path.stem}.ogg"
    return convert_to_ogg(audio_path, output_path)


def _matching_audio_path(tja_path: Path) -> Path | None:
    for suffix in REFERENCE_AUDIO_EXTENSIONS:
        candidate = tja_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _parse_reference_tja(tja_path: Path) -> tuple[dict[str, str], list[ChartBar]]:
    metadata: dict[str, str] = {}
    chart_bars: list[ChartBar] = []
    in_chart = False
    time_signature = "4/4"
    balloon_counts = _parse_balloon_header(metadata)
    balloon_index = 0

    for raw_line in read_tja_text(tja_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        upper_line = line.upper()
        if upper_line == "#START":
            in_chart = True
            balloon_counts = _parse_balloon_header(metadata)
            continue
        if upper_line == "#END":
            break
        if not in_chart:
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().upper()] = value.strip()
            continue
        if upper_line.startswith("#MEASURE"):
            time_signature = _time_signature_from_measure(upper_line)
            continue
        if upper_line.startswith("#"):
            continue

        for raw_notes in line.split(","):
            notes = "".join(character for character in raw_notes.strip() if not character.isspace())
            if not notes:
                continue
            if sorted(set(notes) - REFERENCE_ALLOWED_NOTES):
                continue
            balloon_count = notes.count("7")
            current_balloon_counts = balloon_counts[balloon_index : balloon_index + balloon_count]
            balloon_index += balloon_count
            chart_bars.append(
                ChartBar(
                    index=len(chart_bars),
                    notes=notes,
                    time_signature=time_signature,
                    balloon_counts=current_balloon_counts,
                )
            )

    if not chart_bars:
        raise ValueError(f"Reference TJA does not contain playable bars: {tja_path}")

    return metadata, chart_bars


def _parse_balloon_header(metadata: dict[str, str]) -> list[int]:
    raw_value = metadata.get("BALLOON", "")
    counts: list[int] = []
    for raw_count in raw_value.split(","):
        raw_count = raw_count.strip()
        if not raw_count:
            continue
        value = _optional_int(raw_count)
        if value is not None and value > 0:
            counts.append(value)
    return counts


def _time_signature_from_measure(measure_line: str) -> str:
    parts = measure_line.split(maxsplit=1)
    if len(parts) < 2:
        return "4/4"
    ratio = parts[1].strip()
    if ratio == "1/1":
        return "4/4"
    if ratio == "3/4":
        return "3/4"
    if ratio == "2/3":
        return "6/8"
    try:
        validate_time_signature(ratio)
    except ValueError:
        return "4/4"
    return ratio


def _optional_float(value: str | None) -> float | None:
    if value is None or not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(value: str | None) -> int | None:
    if value is None or not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
