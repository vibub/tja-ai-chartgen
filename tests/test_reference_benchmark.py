import json

from tja_ai_chartgen.reference.benchmark import build_reference_benchmark
from tja_ai_chartgen.reference.tja_parser import parse_tja_text

from test_reference_tja_parser import MULTI_COURSE_TJA


def test_build_reference_benchmark_reports_anonymous_resolution_and_load_metrics():
    parsed = parse_tja_text(MULTI_COURSE_TJA, source_id="anonymous-01")

    benchmark = build_reference_benchmark([parsed])

    assert benchmark["schema_version"] == 1
    assert benchmark["file_count"] == 1
    assert benchmark["course_count"] == 2
    assert benchmark["bar_count"] == 6
    assert benchmark["resolution_counts"] == {"4": 5, "12": 1}

    oni = next(item for item in benchmark["courses"] if item["course"] == "Oni")
    assert oni["source_id"] == "anonymous-01"
    assert oni["level"] == 8
    assert oni["bar_count"] == 4
    assert oni["resolution_counts"] == {"4": 3, "12": 1}
    assert oni["resolution_change_count"] == 2
    assert oni["normal_note_count"] == 6
    assert oni["average_notes_per_second"] > 0
    assert oni["peak_bar_notes_per_second"] >= oni["average_notes_per_second"]
    assert oni["gogo_bar_count"] == 1
    assert oni["gogo_average_notes_per_second"] > 0

    serialized = json.dumps(benchmark, ensure_ascii=False)
    assert "Reference Song" not in serialized
    assert "reference.ogg" not in serialized
    assert "1000" not in serialized


def test_build_reference_benchmark_aggregates_course_level_counts():
    parsed = parse_tja_text(MULTI_COURSE_TJA, source_id="anonymous-01")

    benchmark = build_reference_benchmark([parsed])

    assert benchmark["course_level_counts"] == {"Easy:3": 1, "Oni:8": 1}
    assert benchmark["gogo_bar_count"] == 1
    assert benchmark["resolution_change_count"] == 2
