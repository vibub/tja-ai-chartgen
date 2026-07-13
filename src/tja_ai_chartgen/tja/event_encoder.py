from dataclasses import dataclass

from tja_ai_chartgen.tja.model import ChartBar, ChartBarEvents, ChartLongNoteEvent


SUPPORTED_HIT_NOTES = frozenset("1234")
SUPPORTED_LONG_NOTE_KINDS = frozenset({"drumroll", "balloon"})


@dataclass(frozen=True)
class EventEncodingIssue:
    code: str
    message: str
    tick: int | None = None


class EventEncodingError(ValueError):
    def __init__(self, issues: list[EventEncodingIssue]) -> None:
        self.issues = issues
        lines = [f"Event encoding failed with {len(issues)} issue(s):"]
        lines.extend(f"- {issue.code}: {issue.message}" for issue in issues)
        super().__init__("\n".join(lines))


def encode_chart_bar_events(
    events: ChartBarEvents,
    *,
    canonical_grids_per_bar: int,
    output_resolution: int,
    time_signature: str,
) -> ChartBar:
    if (
        canonical_grids_per_bar <= 0
        or output_resolution <= 0
        or canonical_grids_per_bar % output_resolution != 0
    ):
        raise EventEncodingError(
            [
                EventEncodingIssue(
                    code="incompatible-resolution",
                    message=(
                        f"canonical grid {canonical_grids_per_bar} must be a positive multiple "
                        f"of output resolution {output_resolution}"
                    ),
                )
            ]
        )

    step = canonical_grids_per_bar // output_resolution
    issues: list[EventEncodingIssue] = []
    hits_by_tick: dict[int, str] = {}
    seen_hits: set[tuple[int, str]] = set()

    for hit in events.hits:
        if not _valid_tick(hit.tick, canonical_grids_per_bar):
            issues.append(
                EventEncodingIssue(
                    code="tick-out-of-range",
                    message=f"hit tick {hit.tick!r} is outside the canonical bar",
                    tick=hit.tick if isinstance(hit.tick, int) else None,
                )
            )
            continue
        if hit.tick % step:
            issues.append(
                EventEncodingIssue(
                    code="tick-not-representable",
                    message=(
                        f"hit tick {hit.tick} is not representable at output resolution "
                        f"{output_resolution}"
                    ),
                    tick=hit.tick,
                )
            )
            continue
        if hit.note not in SUPPORTED_HIT_NOTES:
            issues.append(
                EventEncodingIssue(
                    code="unsupported-hit-note",
                    message=f"unsupported normal hit note {hit.note!r}",
                    tick=hit.tick,
                )
            )
            continue
        key = (hit.tick, hit.note)
        if key in seen_hits:
            continue
        seen_hits.add(key)
        previous = hits_by_tick.get(hit.tick)
        if previous is not None and previous != hit.note:
            issues.append(
                EventEncodingIssue(
                    code="conflicting-hit",
                    message=f"tick {hit.tick} contains both {previous!r} and {hit.note!r}",
                    tick=hit.tick,
                )
            )
            continue
        hits_by_tick[hit.tick] = hit.note

    valid_long_notes: list[ChartLongNoteEvent] = []
    seen_long_notes: set[tuple[int, int, str, int | None]] = set()
    for long_note in events.long_notes:
        key = (
            long_note.start_tick,
            long_note.end_tick,
            long_note.kind,
            long_note.balloon_count,
        )
        if key in seen_long_notes:
            continue
        seen_long_notes.add(key)
        valid = True
        if long_note.kind not in SUPPORTED_LONG_NOTE_KINDS:
            issues.append(
                EventEncodingIssue(
                    code="unsupported-long-note-kind",
                    message=f"unsupported long note kind {long_note.kind!r}",
                    tick=long_note.start_tick,
                )
            )
            valid = False
        if not _valid_tick(long_note.start_tick, canonical_grids_per_bar) or not _valid_tick(
            long_note.end_tick, canonical_grids_per_bar
        ):
            issues.append(
                EventEncodingIssue(
                    code="tick-out-of-range",
                    message=(
                        f"long note range {long_note.start_tick!r}-{long_note.end_tick!r} "
                        "is outside the canonical bar"
                    ),
                    tick=(
                        long_note.start_tick if isinstance(long_note.start_tick, int) else None
                    ),
                )
            )
            valid = False
        elif long_note.start_tick % step or long_note.end_tick % step:
            issues.append(
                EventEncodingIssue(
                    code="tick-not-representable",
                    message=(
                        f"long note range {long_note.start_tick}-{long_note.end_tick} is not "
                        f"representable at output resolution {output_resolution}"
                    ),
                    tick=long_note.start_tick,
                )
            )
            valid = False
        if (
            isinstance(long_note.start_tick, int)
            and isinstance(long_note.end_tick, int)
            and long_note.end_tick <= long_note.start_tick
        ):
            issues.append(
                EventEncodingIssue(
                    code="invalid-long-note-range",
                    message="long note end tick must be later than its start tick",
                    tick=long_note.start_tick,
                )
            )
            valid = False
        if long_note.kind == "balloon":
            if not _positive_int(long_note.balloon_count):
                issues.append(
                    EventEncodingIssue(
                        code="invalid-balloon-count",
                        message="balloon notes require a positive integer balloon_count",
                        tick=long_note.start_tick,
                    )
                )
                valid = False
        elif long_note.balloon_count is not None:
            issues.append(
                EventEncodingIssue(
                    code="unexpected-balloon-count",
                    message="only balloon notes may define balloon_count",
                    tick=long_note.start_tick,
                )
            )
            valid = False
        if valid:
            valid_long_notes.append(long_note)

    ordered_long_notes = sorted(valid_long_notes, key=lambda item: (item.start_tick, item.end_tick))
    for previous, current in zip(ordered_long_notes[:-1], ordered_long_notes[1:], strict=True):
        if current.start_tick <= previous.end_tick:
            issues.append(
                EventEncodingIssue(
                    code="overlapping-long-notes",
                    message=(
                        f"long note {current.start_tick}-{current.end_tick} overlaps "
                        f"{previous.start_tick}-{previous.end_tick}"
                    ),
                    tick=current.start_tick,
                )
            )

    for tick in sorted(hits_by_tick):
        if any(note.start_tick <= tick <= note.end_tick for note in ordered_long_notes):
            issues.append(
                EventEncodingIssue(
                    code="hit-inside-long-note",
                    message=f"normal hit at tick {tick} overlaps a long note",
                    tick=tick,
                )
            )

    if issues:
        raise EventEncodingError(issues)

    notes = ["0"] * output_resolution
    for tick, note in hits_by_tick.items():
        notes[tick // step] = note

    balloon_counts: list[int] = []
    for long_note in ordered_long_notes:
        start_index = long_note.start_tick // step
        end_index = long_note.end_tick // step
        notes[start_index] = "7" if long_note.kind == "balloon" else "5"
        notes[end_index] = "8"
        if long_note.kind == "balloon":
            balloon_counts.append(int(long_note.balloon_count))

    return ChartBar(
        index=events.index,
        notes="".join(notes),
        time_signature=time_signature,
        balloon_counts=balloon_counts,
    )


def _valid_tick(value: object, canonical_grids_per_bar: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < canonical_grids_per_bar
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
