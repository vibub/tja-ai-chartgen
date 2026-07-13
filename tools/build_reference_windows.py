from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tja_ai_chartgen.reference.dataset import paired_reference_tja_paths
from tja_ai_chartgen.reference.tja_parser import ReferenceTjaParseError, parse_tja_file
from tja_ai_chartgen.reference.windows import DEFAULT_WINDOW_SIZE, build_reference_windows
from tja_ai_chartgen.utils.paths import write_json


DEFAULT_OUTPUT = Path("src/tja_ai_chartgen/ai/reference_windows.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build anonymous continuous reference windows from paired audio/TJA files."
        )
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    args = parser.parse_args()

    try:
        parsed_files = [
            parse_tja_file(path) for path in paired_reference_tja_paths(args.reference_dir)
        ]
        payload = build_reference_windows(parsed_files, window_size=args.window_size)
        write_json(args.output, payload)
    except (FileNotFoundError, ValueError, ReferenceTjaParseError) as error:
        print(f"Reference window build failed: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_count": payload["source_count"],
                "course_count": payload["course_count"],
                "window_count": len(payload["windows"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
