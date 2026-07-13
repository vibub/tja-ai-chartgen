from dataclasses import dataclass

from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.tja.model import TjaChart

SUPPORTED_NOTE_CHARACTERS = frozenset("01234578")
MAX_NOTE_GRIDS_PER_BAR = 192


@dataclass(frozen=True, slots=True)
class ChartValidationIssue:
    code: str
    message: str
    bar_index: int
    grid_index: int | None = None

    def format(self) -> str:
        location = f"bar {self.bar_index}"
        if self.grid_index is not None:
            location += f", grid {self.grid_index}"
        return f"{location}: {self.message}"


class ChartValidationError(ValueError):
    """Raised when a structured chart cannot be rendered safely."""

    def __init__(self, issues: list[ChartValidationIssue]) -> None:
        self.issues = tuple(issues)
        issue_word = "issue" if len(issues) == 1 else "issues"
        details = "\n".join(f"- {issue.format()}" for issue in issues)
        super().__init__(f"Chart preflight failed with {len(issues)} {issue_word}:\n{details}")


def validate_chart(chart: TjaChart) -> None:
    issues: list[ChartValidationIssue] = []

    for bar in chart.bars:
        get_meter_spec(bar.time_signature)
        if not 1 <= len(bar.notes) <= MAX_NOTE_GRIDS_PER_BAR:
            issues.append(
                ChartValidationIssue(
                    code="invalid-note-count",
                    message=(
                        f"expected between 1 and {MAX_NOTE_GRIDS_PER_BAR} note grids, "
                        f"got {len(bar.notes)}"
                    ),
                    bar_index=bar.index,
                )
            )

        for grid_index, note in enumerate(bar.notes):
            if note not in SUPPORTED_NOTE_CHARACTERS:
                issues.append(
                    ChartValidationIssue(
                        code="unsupported-note-character",
                        message=f"unsupported note character {note!r}",
                        bar_index=bar.index,
                        grid_index=grid_index,
                    )
                )

        balloon_grids = [grid_index for grid_index, note in enumerate(bar.notes) if note == "7"]
        for grid_index in balloon_grids[len(bar.balloon_counts) :]:
            issues.append(
                ChartValidationIssue(
                    code="missing-balloon-count",
                    message="balloon note has no matching balloon count",
                    bar_index=bar.index,
                    grid_index=grid_index,
                )
            )
        for count_index in range(len(balloon_grids), len(bar.balloon_counts)):
            issues.append(
                ChartValidationIssue(
                    code="extra-balloon-count",
                    message=f"balloon count at index {count_index} has no matching balloon note",
                    bar_index=bar.index,
                )
            )

        for count_index, count in enumerate(bar.balloon_counts):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                issues.append(
                    ChartValidationIssue(
                        code="invalid-balloon-count",
                        message=f"balloon count at index {count_index} must be a positive integer, got {count!r}",
                        bar_index=bar.index,
                    )
                )

    issues.extend(_validate_sustained_notes(chart))

    if issues:
        bar_order = {bar.index: position for position, bar in enumerate(chart.bars)}
        issues.sort(
            key=lambda issue: (
                bar_order[issue.bar_index],
                -1 if issue.grid_index is None else issue.grid_index,
            )
        )
        raise ChartValidationError(issues)


def _validate_sustained_notes(chart: TjaChart) -> list[ChartValidationIssue]:
    issues: list[ChartValidationIssue] = []
    active_start: tuple[int, int, str] | None = None

    for bar in chart.bars:
        for grid_index, note in enumerate(bar.notes):
            if note in {"5", "7"}:
                if active_start is None:
                    active_start = (bar.index, grid_index, note)
                else:
                    issues.append(
                        ChartValidationIssue(
                            code="overlapping-roll-start",
                            message=f"sustained note {note!r} starts before the active sustained note is closed",
                            bar_index=bar.index,
                            grid_index=grid_index,
                        )
                    )
            elif note == "8":
                if active_start is None:
                    issues.append(
                        ChartValidationIssue(
                            code="orphan-roll-end",
                            message="sustained note end has no active start",
                            bar_index=bar.index,
                            grid_index=grid_index,
                        )
                    )
                else:
                    active_start = None

    if active_start is not None:
        bar_index, grid_index, note = active_start
        issues.append(
            ChartValidationIssue(
                code="unclosed-roll",
                message=f"sustained note {note!r} is not closed before the chart ends",
                bar_index=bar_index,
                grid_index=grid_index,
            )
        )

    return issues
