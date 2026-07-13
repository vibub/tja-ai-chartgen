import pytest

from tja_ai_chartgen.tja.chart_validator import (
    MAX_NOTE_GRIDS_PER_BAR,
    ChartValidationError,
    validate_chart,
)
from tja_ai_chartgen.tja.model import ChartBar, ChartMetadata, TjaChart


def _chart(*bars: ChartBar) -> TjaChart:
    return TjaChart(
        metadata=ChartMetadata(
            title="Song Title",
            wave="song.ogg",
            bpm=120.0,
            offset=0.0,
        ),
        bars=list(bars),
    )


@pytest.mark.parametrize(
    ("time_signature", "notes"),
    [
        ("4/4", "1"),
        ("4/4", "1000"),
        ("4/4", "1000100010001000"),
        ("4/4", "1" + ("0" * 23)),
        ("4/4", "1" + ("0" * 47)),
        ("3/4", "100010001000"),
        ("3/4", "1" + ("0" * 17)),
        ("3/4", "1" + ("0" * 35)),
        ("6/8", "100000100000"),
        ("6/8", "1" + ("0" * 35)),
    ],
)
def test_validate_chart_accepts_supported_bar_lengths(time_signature, notes):
    validate_chart(_chart(ChartBar(index=0, notes=notes, time_signature=time_signature)))


def test_validate_chart_reports_all_unsupported_note_characters():
    chart = _chart(ChartBar(index=3, notes="90001000X0001000"))

    with pytest.raises(ChartValidationError) as error:
        validate_chart(chart)

    issues = error.value.issues
    assert [(issue.code, issue.bar_index, issue.grid_index) for issue in issues] == [
        ("unsupported-note-character", 3, 0),
        ("unsupported-note-character", 3, 8),
    ]
    assert "unsupported note character '9'" in str(error.value)
    assert "unsupported note character 'X'" in str(error.value)


@pytest.mark.parametrize("notes", ["", "0" * (MAX_NOTE_GRIDS_PER_BAR + 1)])
def test_validate_chart_rejects_unsafe_note_count(notes):
    with pytest.raises(ChartValidationError) as error:
        validate_chart(_chart(ChartBar(index=1, notes=notes)))

    issue = next(issue for issue in error.value.issues if issue.code == "invalid-note-count")
    assert issue.bar_index == 1
    assert "between 1 and" in issue.message


@pytest.mark.parametrize(
    ("notes", "balloon_counts", "code"),
    [
        ("7000000080000000", [], "missing-balloon-count"),
        ("1000100010001000", [8], "extra-balloon-count"),
        ("7000700080008000", [8], "missing-balloon-count"),
        ("7000000080000000", [8, 12], "extra-balloon-count"),
    ],
)
def test_validate_chart_requires_exact_balloon_count_mapping(notes, balloon_counts, code):
    with pytest.raises(ChartValidationError) as error:
        validate_chart(_chart(ChartBar(index=2, notes=notes, balloon_counts=balloon_counts)))

    assert any(issue.code == code for issue in error.value.issues)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "8"])
def test_validate_chart_rejects_invalid_balloon_count_values(count):
    bar = ChartBar.model_construct(
        index=0,
        notes="7000000080000000",
        time_signature="4/4",
        balloon_counts=[count],
    )

    with pytest.raises(ChartValidationError) as error:
        validate_chart(_chart(bar))

    assert any(issue.code == "invalid-balloon-count" for issue in error.value.issues)


@pytest.mark.parametrize(
    "notes",
    [
        "5000000080000000",
        "7000000080000000",
    ],
)
def test_validate_chart_accepts_closed_sustained_notes(notes):
    balloon_counts = [8] if "7" in notes else []
    validate_chart(_chart(ChartBar(index=0, notes=notes, balloon_counts=balloon_counts)))


def test_validate_chart_accepts_sustained_note_closed_in_later_bar():
    validate_chart(
        _chart(
            ChartBar(index=0, notes="5000000000000000"),
            ChartBar(index=1, notes="8000100010001000"),
        )
    )


@pytest.mark.parametrize(
    ("notes", "code"),
    [
        ("8000100010001000", "orphan-roll-end"),
        ("5000100010001000", "unclosed-roll"),
        ("7000100010001000", "unclosed-roll"),
        ("5000700080000000", "overlapping-roll-start"),
    ],
)
def test_validate_chart_rejects_invalid_sustained_note_structure(notes, code):
    balloon_counts = [8] if "7" in notes else []

    with pytest.raises(ChartValidationError) as error:
        validate_chart(_chart(ChartBar(index=0, notes=notes, balloon_counts=balloon_counts)))

    assert any(issue.code == code for issue in error.value.issues)


def test_validate_chart_aggregates_multiple_issue_types_in_stable_order():
    bar = ChartBar.model_construct(
        index=4,
        notes="590000000000000",
        time_signature="4/4",
        balloon_counts=[0],
    )

    with pytest.raises(ChartValidationError) as error:
        validate_chart(_chart(bar))

    assert [issue.code for issue in error.value.issues] == [
        "extra-balloon-count",
        "invalid-balloon-count",
        "unclosed-roll",
        "unsupported-note-character",
    ]
    assert str(error.value).startswith("Chart preflight failed with 4 issues:")
