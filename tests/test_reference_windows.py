import json

from tja_ai_chartgen.reference.tja_parser import parse_tja_text
from tja_ai_chartgen.reference.windows import BAR_COLUMNS, build_reference_windows

from test_reference_tja_parser import MULTI_COURSE_TJA


def test_build_reference_windows_emits_anonymous_continuous_sparse_events():
    parsed = parse_tja_text(MULTI_COURSE_TJA, source_id="anonymous-01")

    payload = build_reference_windows([parsed], window_size=3)

    assert payload["schema_version"] == 1
    assert payload["source_count"] == 1
    assert payload["course_count"] == 2
    assert payload["bar_columns"] == BAR_COLUMNS
    assert payload["windows"]
    for window in payload["windows"]:
        assert window["source_id"] == "anonymous-01"
        assert window["end_bar"] - window["start_bar"] + 1 == len(window["bars"])
        assert [bar[0] for bar in window["bars"]] == list(range(len(window["bars"])))
        for bar in window["bars"]:
            events = bar[BAR_COLUMNS.index("events")]
            resolution = bar[BAR_COLUMNS.index("resolution")]
            assert all(0 <= event[0] < resolution and event[1] != "0" for event in events)

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Reference Song" not in serialized
    assert "reference.ogg" not in serialized
    assert '"notes"' not in serialized


def test_build_reference_windows_selects_intro_peak_and_cadence_without_duplicates():
    parsed = parse_tja_text(MULTI_COURSE_TJA, source_id="anonymous-01")

    payload = build_reference_windows([parsed], window_size=2)
    oni_windows = [window for window in payload["windows"] if window["course"] == "Oni"]

    assert {window["role"] for window in oni_windows} >= {"intro", "cadence"}
    assert len({window["start_bar"] for window in oni_windows}) == len(oni_windows)
    assert all(window["average_notes_per_second"] >= 0 for window in oni_windows)
