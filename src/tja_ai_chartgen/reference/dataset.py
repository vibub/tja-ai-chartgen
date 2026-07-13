from pathlib import Path


AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def paired_reference_tja_paths(reference_dir: Path) -> list[Path]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

    tja_paths = sorted(reference_dir.glob("*.tja"))
    if not tja_paths:
        raise ValueError("Reference directory contains no .tja files")

    audio_by_stem = {
        path.stem.casefold()
        for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES
    }
    missing_audio = [
        path.name for path in tja_paths if path.stem.casefold() not in audio_by_stem
    ]
    if missing_audio:
        raise ValueError(
            "Reference TJA files are missing same-stem audio pairs: "
            + ", ".join(missing_audio)
        )
    return tja_paths
