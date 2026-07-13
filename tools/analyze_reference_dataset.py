from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from tja_ai_chartgen.reference.benchmark import build_reference_benchmark
from tja_ai_chartgen.reference.dataset import paired_reference_tja_paths
from tja_ai_chartgen.reference.tja_parser import ReferenceTjaParseError, parse_tja_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse paired reference OGG/TJA files and emit anonymous aggregate metrics."
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        benchmark = analyze_reference_directory(args.reference_dir)
        payload = json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            _write_text_atomically(args.output, payload)
    except (FileNotFoundError, ValueError, ReferenceTjaParseError) as error:
        print(f"Reference analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


def analyze_reference_directory(reference_dir: Path) -> dict[str, object]:
    tja_paths = paired_reference_tja_paths(reference_dir)
    parsed_files = [parse_tja_file(path) for path in tja_paths]
    benchmark = build_reference_benchmark(parsed_files)
    benchmark["paired_audio_count"] = len(tja_paths)
    return benchmark


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
