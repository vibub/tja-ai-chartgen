import pytest

from tja_ai_chartgen.tja.event_encoder import EventEncodingError, encode_chart_bar_events
from tja_ai_chartgen.tja.model import ChartBarEvents, ChartHitEvent, ChartLongNoteEvent


@pytest.mark.parametrize(
    ("canonical_grids", "resolution", "ticks", "expected_indexes"),
    [
        (48, 16, [0, 12, 24, 36], [0, 4, 8, 12]),
        (48, 24, [0, 2, 46], [0, 1, 23]),
        (48, 48, [0, 1, 47], [0, 1, 47]),
        (36, 12, [0, 9, 18, 27], [0, 3, 6, 9]),
        (36, 18, [0, 2, 34], [0, 1, 17]),
        (36, 36, [0, 1, 35], [0, 1, 35]),
    ],
)
def test_encode_chart_bar_events_maps_canonical_ticks(
    canonical_grids, resolution, ticks, expected_indexes
):
    events = ChartBarEvents(
        index=3,
        hits=[ChartHitEvent(tick=tick, note=str((index % 4) + 1)) for index, tick in enumerate(ticks)],
    )

    chart_bar = encode_chart_bar_events(
        events,
        canonical_grids_per_bar=canonical_grids,
        output_resolution=resolution,
        time_signature="4/4" if canonical_grids == 48 else "3/4",
    )

    assert chart_bar.index == 3
    assert len(chart_bar.notes) == resolution
    assert [index for index, note in enumerate(chart_bar.notes) if note != "0"] == expected_indexes
    assert chart_bar.balloon_counts == []


def test_encode_chart_bar_events_sorts_and_deduplicates_identical_hits():
    events = ChartBarEvents(
        index=0,
        hits=[
            ChartHitEvent(tick=12, note="2"),
            ChartHitEvent(tick=0, note="1"),
            ChartHitEvent(tick=12, note="2"),
        ],
    )

    chart_bar = encode_chart_bar_events(
        events,
        canonical_grids_per_bar=48,
        output_resolution=16,
        time_signature="4/4",
    )

    assert chart_bar.notes == "1000200000000000"


def test_encode_chart_bar_events_encodes_drumroll_and_balloon():
    events = ChartBarEvents(
        index=0,
        long_notes=[
            ChartLongNoteEvent(start_tick=6, end_tick=18, kind="drumroll"),
            ChartLongNoteEvent(
                start_tick=24,
                end_tick=45,
                kind="balloon",
                balloon_count=12,
            ),
        ],
    )

    chart_bar = encode_chart_bar_events(
        events,
        canonical_grids_per_bar=48,
        output_resolution=16,
        time_signature="4/4",
    )

    assert chart_bar.notes[2] == "5"
    assert chart_bar.notes[6] == "8"
    assert chart_bar.notes[8] == "7"
    assert chart_bar.notes[15] == "8"
    assert chart_bar.balloon_counts == [12]


@pytest.mark.parametrize(
    ("events", "canonical_grids", "resolution", "expected_code"),
    [
        (
            ChartBarEvents(index=0, hits=[ChartHitEvent(tick=-1, note="1")]),
            48,
            16,
            "tick-out-of-range",
        ),
        (
            ChartBarEvents(index=0, hits=[ChartHitEvent(tick=48, note="1")]),
            48,
            16,
            "tick-out-of-range",
        ),
        (
            ChartBarEvents(index=0, hits=[ChartHitEvent(tick=1, note="1")]),
            48,
            16,
            "tick-not-representable",
        ),
        (
            ChartBarEvents(index=0, hits=[ChartHitEvent(tick=0, note="9")]),
            48,
            16,
            "unsupported-hit-note",
        ),
        (
            ChartBarEvents(
                index=0,
                hits=[ChartHitEvent(tick=0, note="1"), ChartHitEvent(tick=0, note="2")],
            ),
            48,
            16,
            "conflicting-hit",
        ),
        (
            ChartBarEvents(
                index=0,
                long_notes=[ChartLongNoteEvent(start_tick=12, end_tick=12, kind="drumroll")],
            ),
            48,
            16,
            "invalid-long-note-range",
        ),
        (
            ChartBarEvents(
                index=0,
                long_notes=[ChartLongNoteEvent(start_tick=12, end_tick=24, kind="laser")],
            ),
            48,
            16,
            "unsupported-long-note-kind",
        ),
        (
            ChartBarEvents(
                index=0,
                long_notes=[
                    ChartLongNoteEvent(start_tick=12, end_tick=24, kind="balloon")
                ],
            ),
            48,
            16,
            "invalid-balloon-count",
        ),
        (
            ChartBarEvents(
                index=0,
                long_notes=[
                    ChartLongNoteEvent(
                        start_tick=12,
                        end_tick=24,
                        kind="drumroll",
                        balloon_count=8,
                    )
                ],
            ),
            48,
            16,
            "unexpected-balloon-count",
        ),
    ],
)
def test_encode_chart_bar_events_rejects_invalid_events(
    events, canonical_grids, resolution, expected_code
):
    with pytest.raises(EventEncodingError) as error:
        encode_chart_bar_events(
            events,
            canonical_grids_per_bar=canonical_grids,
            output_resolution=resolution,
            time_signature="4/4",
        )

    assert any(issue.code == expected_code for issue in error.value.issues)


def test_encode_chart_bar_events_rejects_overlapping_long_notes_and_hits():
    events = ChartBarEvents(
        index=2,
        hits=[ChartHitEvent(tick=18, note="1")],
        long_notes=[
            ChartLongNoteEvent(start_tick=6, end_tick=18, kind="drumroll"),
            ChartLongNoteEvent(start_tick=12, end_tick=24, kind="balloon", balloon_count=8),
        ],
    )

    with pytest.raises(EventEncodingError) as error:
        encode_chart_bar_events(
            events,
            canonical_grids_per_bar=48,
            output_resolution=16,
            time_signature="4/4",
        )

    codes = {issue.code for issue in error.value.issues}
    assert "overlapping-long-notes" in codes
    assert "hit-inside-long-note" in codes


def test_encode_chart_bar_events_rejects_incompatible_resolution():
    with pytest.raises(EventEncodingError) as error:
        encode_chart_bar_events(
            ChartBarEvents(index=0),
            canonical_grids_per_bar=48,
            output_resolution=20,
            time_signature="4/4",
        )

    assert [issue.code for issue in error.value.issues] == ["incompatible-resolution"]
