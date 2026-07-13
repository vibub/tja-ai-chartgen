from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from tja_ai_chartgen.reference.model import (
    ReferenceBar,
    ReferenceCourse,
    ReferenceTempoChange,
    ReferenceTja,
)


SUPPORTED_NOTE_CHARACTERS = frozenset("0123456789")
_UNSUPPORTED_COMMAND_PREFIXES = (
    "#BRANCHSTART",
    "#BRANCHEND",
    "#DELAY",
    "#SCROLL",
)
_UNSUPPORTED_COMMANDS = {"#N", "#E", "#M"}


class ReferenceTjaParseError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        line: int | None = None,
        course: str | None = None,
        bar: int | None = None,
    ) -> None:
        self.code = code
        self.line = line
        self.course = course
        self.bar = bar
        context: list[str] = []
        if line is not None:
            context.append(f"line {line}")
        if course is not None:
            context.append(f"course {course}")
        if bar is not None:
            context.append(f"bar {bar + 1}")
        suffix = f" ({', '.join(context)})" if context else ""
        super().__init__(f"{code}: {message}{suffix}")


def parse_tja_file(path: Path) -> ReferenceTja:
    data = path.read_bytes()
    text = _decode_tja(data)
    return parse_tja_text(text, source_id=sha256(data).hexdigest()[:16])


def parse_tja_text(text: str, *, source_id: str) -> ReferenceTja:
    lines = text.splitlines()
    global_metadata: dict[str, str] = {}
    course_metadata: dict[str, str] = {}
    courses: list[ReferenceCourse] = []

    in_chart = False
    pending_notes = ""
    pending_start_bpm: float | None = None
    pending_tempo_changes: list[ReferenceTempoChange] = []
    current_bpm = 0.0
    current_measure = "1/1"
    current_time = 0.0
    current_gogo = False
    current_barline = True
    current_course = ""
    current_level = 0
    current_bars: list[ReferenceBar] = []
    balloon_values: list[int] = []
    balloon_index = 0

    for line_number, raw_line in enumerate(lines, start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        upper = line.upper()

        if not in_chart:
            if upper.startswith("#START"):
                bpm = _parse_positive_float(
                    global_metadata.get("BPM"),
                    code="invalid-bpm",
                    message="BPM metadata must be a positive number",
                    line=line_number,
                )
                current_course = course_metadata.get("COURSE", "").strip()
                if not current_course:
                    raise ReferenceTjaParseError(
                        "missing-course",
                        "COURSE metadata is required before #START",
                        line=line_number,
                    )
                current_level = _parse_level(course_metadata.get("LEVEL"), line=line_number)
                balloon_values = _parse_balloon_values(
                    course_metadata.get("BALLOON", ""), line=line_number
                )
                balloon_index = 0
                current_bpm = bpm
                current_measure = "1/1"
                current_time = -_parse_float(
                    global_metadata.get("OFFSET", "0"),
                    code="invalid-offset",
                    message="OFFSET metadata must be numeric",
                    line=line_number,
                )
                current_gogo = False
                current_barline = True
                current_bars = []
                pending_notes = ""
                pending_start_bpm = None
                pending_tempo_changes = []
                in_chart = True
                continue
            if upper.startswith("#"):
                raise ReferenceTjaParseError(
                    "unexpected-command",
                    f"command {line.split()[0]} is not valid outside a chart",
                    line=line_number,
                )
            if ":" not in line:
                raise ReferenceTjaParseError(
                    "invalid-metadata",
                    "metadata lines must contain ':'",
                    line=line_number,
                )
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "COURSE":
                course_metadata = {"COURSE": value}
            elif key in {"LEVEL", "BALLOON", "SCOREINIT", "SCOREDIFF"}:
                course_metadata[key] = value
            else:
                global_metadata[key] = value
            continue

        if upper.startswith("#END"):
            if pending_notes:
                raise ReferenceTjaParseError(
                    "unclosed-bar",
                    "notes before #END are missing a trailing comma",
                    line=line_number,
                    course=current_course,
                    bar=len(current_bars),
                )
            if current_gogo:
                raise ReferenceTjaParseError(
                    "unclosed-gogo",
                    "#GOGOSTART has no matching #GOGOEND",
                    line=line_number,
                    course=current_course,
                )
            if balloon_index != len(balloon_values):
                raise ReferenceTjaParseError(
                    "balloon-count-mismatch",
                    f"consumed {balloon_index} BALLOON value(s), expected {len(balloon_values)}",
                    line=line_number,
                    course=current_course,
                )
            courses.append(
                ReferenceCourse(
                    course=current_course,
                    level=current_level,
                    bars=current_bars,
                )
            )
            in_chart = False
            continue

        if upper.startswith("#"):
            if upper in _UNSUPPORTED_COMMANDS or upper.startswith(_UNSUPPORTED_COMMAND_PREFIXES):
                raise ReferenceTjaParseError(
                    "unsupported-command",
                    f"command {line.split()[0]} is not supported by the reference parser",
                    line=line_number,
                    course=current_course,
                )
            if upper.startswith("#BPMCHANGE"):
                value = _command_argument(line, "#BPMCHANGE", line_number, current_course)
                next_bpm = _parse_positive_float(
                    value,
                    code="invalid-bpm",
                    message="#BPMCHANGE must use a positive number",
                    line=line_number,
                    course=current_course,
                )
                if pending_notes:
                    if pending_start_bpm is None:
                        pending_start_bpm = current_bpm
                    pending_tempo_changes.append(
                        ReferenceTempoChange(position=len(pending_notes), bpm=next_bpm)
                    )
                current_bpm = next_bpm
                continue
            if pending_notes:
                raise ReferenceTjaParseError(
                    "command-inside-bar",
                    "only #BPMCHANGE may split an unfinished notes measure",
                    line=line_number,
                    course=current_course,
                    bar=len(current_bars),
                )
            if upper.startswith("#MEASURE"):
                value = _command_argument(line, "#MEASURE", line_number, current_course)
                _measure_quarter_notes(value, line=line_number, course=current_course)
                current_measure = value
            elif upper == "#GOGOSTART":
                if current_gogo:
                    raise ReferenceTjaParseError(
                        "nested-gogo",
                        "#GOGOSTART cannot be nested",
                        line=line_number,
                        course=current_course,
                    )
                current_gogo = True
            elif upper == "#GOGOEND":
                if not current_gogo:
                    raise ReferenceTjaParseError(
                        "orphan-gogo-end",
                        "#GOGOEND has no matching #GOGOSTART",
                        line=line_number,
                        course=current_course,
                    )
                current_gogo = False
            elif upper == "#BARLINEOFF":
                current_barline = False
            elif upper == "#BARLINEON":
                current_barline = True
            elif upper.startswith("#START"):
                raise ReferenceTjaParseError(
                    "nested-chart",
                    "#START cannot appear inside a chart",
                    line=line_number,
                    course=current_course,
                )
            else:
                raise ReferenceTjaParseError(
                    "unsupported-command",
                    f"command {line.split()[0]} is not supported by the reference parser",
                    line=line_number,
                    course=current_course,
                )
            continue

        for character in line:
            if character.isspace():
                continue
            if character == ",":
                if not pending_notes:
                    raise ReferenceTjaParseError(
                        "empty-bar",
                        "a comma must close a non-empty notes measure",
                        line=line_number,
                        course=current_course,
                        bar=len(current_bars),
                    )
                illegal = sorted(set(pending_notes) - SUPPORTED_NOTE_CHARACTERS)
                if illegal:
                    raise ReferenceTjaParseError(
                        "unsupported-note-character",
                        f"unsupported note character(s): {''.join(illegal)}",
                        line=line_number,
                        course=current_course,
                        bar=len(current_bars),
                    )
                bar_balloon_counts: list[int] = []
                for _ in range(sum(character in "79" for character in pending_notes)):
                    if balloon_index >= len(balloon_values):
                        raise ReferenceTjaParseError(
                            "balloon-count-mismatch",
                            "chart contains more balloon notes than BALLOON values",
                            line=line_number,
                            course=current_course,
                            bar=len(current_bars),
                        )
                    bar_balloon_counts.append(balloon_values[balloon_index])
                    balloon_index += 1
                quarter_notes = _measure_quarter_notes(
                    current_measure, line=line_number, course=current_course
                )
                start_bpm = pending_start_bpm or current_bpm
                duration = _bar_duration(
                    resolution=len(pending_notes),
                    quarter_notes=quarter_notes,
                    start_bpm=start_bpm,
                    tempo_changes=pending_tempo_changes,
                )
                start_time = current_time
                end_time = current_time + duration
                current_bars.append(
                    ReferenceBar(
                        index=len(current_bars),
                        notes=pending_notes,
                        resolution=len(pending_notes),
                        start_time=round(start_time, 6),
                        end_time=round(end_time, 6),
                        bpm=start_bpm,
                        tempo_changes=list(pending_tempo_changes),
                        measure_ratio=current_measure,
                        gogo=current_gogo,
                        barline_visible=current_barline,
                        balloon_counts=bar_balloon_counts,
                    )
                )
                current_time = end_time
                pending_notes = ""
                pending_start_bpm = None
                pending_tempo_changes = []
                continue
            if not pending_notes:
                pending_start_bpm = current_bpm
            pending_notes += character

    if in_chart:
        raise ReferenceTjaParseError(
            "unclosed-chart",
            "#START has no matching #END",
            line=max(1, len(lines)),
            course=current_course or None,
        )
    if not courses:
        raise ReferenceTjaParseError(
            "missing-chart",
            "TJA contains no completed chart courses",
            line=max(1, len(lines)),
        )

    bpm = _parse_positive_float(
        global_metadata.get("BPM"),
        code="invalid-bpm",
        message="BPM metadata must be a positive number",
        line=1,
    )
    return ReferenceTja(
        source_id=source_id,
        title=global_metadata.get("TITLE") or None,
        artist=(global_metadata.get("SUBTITLE") or "").removeprefix("--").strip() or None,
        wave=global_metadata.get("WAVE") or None,
        bpm=bpm,
        internal_offset=-_parse_float(
            global_metadata.get("OFFSET", "0"),
            code="invalid-offset",
            message="OFFSET metadata must be numeric",
            line=1,
        ),
        courses=courses,
    )


def _decode_tja(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ReferenceTjaParseError(
        "decode-failed",
        "TJA could not be decoded as UTF-8 or CP932",
        line=1,
    )


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0]


def _parse_level(value: str | None, *, line: int) -> int:
    try:
        level = int(float(value or ""))
    except ValueError as error:
        raise ReferenceTjaParseError(
            "invalid-level",
            "LEVEL metadata must be numeric",
            line=line,
        ) from error
    if level < 0:
        raise ReferenceTjaParseError(
            "invalid-level",
            "LEVEL metadata must not be negative",
            line=line,
        )
    return level


def _parse_balloon_values(value: str, *, line: int) -> list[int]:
    if not value.strip():
        return []
    counts: list[int] = []
    for raw_count in value.split(","):
        try:
            count = int(raw_count.strip())
        except ValueError as error:
            raise ReferenceTjaParseError(
                "invalid-balloon-count",
                "BALLOON values must be positive integers",
                line=line,
            ) from error
        if count <= 0:
            raise ReferenceTjaParseError(
                "invalid-balloon-count",
                "BALLOON values must be positive integers",
                line=line,
            )
        counts.append(count)
    return counts


def _command_argument(line: str, command: str, line_number: int, course: str) -> str:
    parts = line.split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        raise ReferenceTjaParseError(
            "missing-command-argument",
            f"{command} requires an argument",
            line=line_number,
            course=course,
        )
    return parts[1].strip()


def _bar_duration(
    *,
    resolution: int,
    quarter_notes: float,
    start_bpm: float,
    tempo_changes: list[ReferenceTempoChange],
) -> float:
    duration = 0.0
    previous_position = 0
    active_bpm = start_bpm
    for change in tempo_changes:
        position = min(resolution, max(previous_position, change.position))
        segment_quarters = quarter_notes * (position - previous_position) / resolution
        duration += segment_quarters * 60.0 / active_bpm
        previous_position = position
        active_bpm = change.bpm
    remaining_quarters = quarter_notes * (resolution - previous_position) / resolution
    duration += remaining_quarters * 60.0 / active_bpm
    return duration


def _measure_quarter_notes(value: str, *, line: int, course: str | None = None) -> float:
    parts = value.split("/", 1)
    if len(parts) != 2:
        raise ReferenceTjaParseError(
            "invalid-measure",
            "MEASURE must use numerator/denominator syntax",
            line=line,
            course=course,
        )
    try:
        numerator = float(parts[0])
        denominator = float(parts[1])
    except ValueError as error:
        raise ReferenceTjaParseError(
            "invalid-measure",
            "MEASURE numerator and denominator must be numeric",
            line=line,
            course=course,
        ) from error
    if numerator <= 0 or denominator <= 0:
        raise ReferenceTjaParseError(
            "invalid-measure",
            "MEASURE numerator and denominator must be positive",
            line=line,
            course=course,
        )
    return 4.0 * numerator / denominator


def _parse_positive_float(
    value: str | None,
    *,
    code: str,
    message: str,
    line: int,
    course: str | None = None,
) -> float:
    result = _parse_float(value or "", code=code, message=message, line=line, course=course)
    if result <= 0:
        raise ReferenceTjaParseError(code, message, line=line, course=course)
    return result


def _parse_float(
    value: str,
    *,
    code: str,
    message: str,
    line: int,
    course: str | None = None,
) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ReferenceTjaParseError(code, message, line=line, course=course) from error
