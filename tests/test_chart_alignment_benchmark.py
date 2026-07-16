import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from tja_ai_chartgen.evaluation.chart_alignment import (
    build_chart_alignment_benchmark,
    build_chart_alignment_metrics,
    render_chart_alignment_markdown,
)
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, ResolutionPlan
from tja_ai_chartgen.tja.quality import build_quality_report


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
BASELINE_PATH = FIXTURE_DIR / "chart_alignment_baseline.json"
BASELINE_REPORT_PATH = FIXTURE_DIR / "chart_alignment_baseline.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "benchmark_chart_alignment.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("benchmark_chart_alignment", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
benchmark_chart_alignment = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(benchmark_chart_alignment)


def _ground_truth() -> dict[str, object]:
    return {
        "schema_version": 1,
        "audio": "sample.wav",
        "bpm": 120.0,
        "time_signature": "4/4",
        "duration": 2.0,
        "first_downbeat": 0.0,
        "onsets": [0.0, 0.5, 1.5],
        "strong_onsets": [0.0],
        "beats": [0.0, 0.5, 1.5],
        "downbeats": [0.0],
        "low_band_onsets": [],
        "high_band_onsets": [],
        "silent_ranges": [[0.9, 1.1]],
        "fill_ranges": [[1.4, 1.6]],
        "sections": [],
    }


def _feature_bar() -> BarFeature:
    return BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.8,
        grids_per_bar=48,
        activity_grids=[0.5] * 48,
    )


def _chart_metrics(*, course: str = "Oni", deterministic: bool = True):
    feature_bars = [_feature_bar()]
    chart_bars = [ChartBar(index=0, notes="1111")]
    return build_chart_alignment_metrics(
        _ground_truth(),
        feature_bars,
        chart_bars,
        course=course,
        level=10,
        deterministic=deterministic,
        quality_report=build_quality_report(chart_bars, feature_bars),
    )


def test_build_chart_alignment_metrics_covers_phase_zero_chart_metrics():
    metrics = _chart_metrics()

    assert metrics["note_count"] == 4
    assert metrics["note_onset"]["precision"] == 0.75
    assert metrics["note_onset"]["recall"] == 1.0
    assert metrics["strong_onset_response"]["recall"] == 1.0
    assert metrics["downbeat_response"]["recall"] == 1.0
    assert metrics["fill_onset_response"]["recall"] == 1.0
    assert metrics["unsupported_note_count"] == 1
    assert metrics["unsupported_note_ratio"] == 0.25
    assert metrics["silent_violation_count"] == 1
    assert metrics["silent_violation_ratio"] == 0.25
    assert metrics["deterministic"] is True


def test_build_chart_alignment_metrics_rejects_mismatched_bar_counts():
    try:
        build_chart_alignment_metrics(
            _ground_truth(),
            [_feature_bar()],
            [],
            course="Oni",
            level=10,
            deterministic=True,
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Expected mismatched chart bar count to fail")


def test_build_chart_alignment_benchmark_aggregates_courses_and_determinism():
    metrics = [
        _chart_metrics(course="Easy"),
        _chart_metrics(course="Oni", deterministic=False),
    ]

    benchmark = build_chart_alignment_benchmark(
        metrics,
        fixture_count=1,
        use_beatnet=False,
        style="technical",
        density="auto",
        special_notes=False,
    )

    assert benchmark["schema_version"] == 1
    assert benchmark["benchmark_version"] == "chart-alignment-v1"
    assert benchmark["fixture_count"] == 1
    assert benchmark["chart_count"] == 2
    assert benchmark["summary"]["note_count"] == 8
    assert benchmark["summary"]["unsupported_note_ratio"] == 0.25
    assert benchmark["summary"]["deterministic_rate"] == 0.5
    assert set(benchmark["course_summaries"]) == {"Easy", "Oni"}
    calibration = benchmark["report_only_calibration"]
    assert calibration["chart_count"] == 2
    assert set(calibration["course_summaries"]) == {"Easy", "Oni"}
    assert calibration["summary"]["note_onset_alignment"]["evaluated_count"] == 8
    assert 0.0 <= calibration["summary"]["note_onset_alignment"]["value"] <= 1.0
    assert set(calibration["summary"]["salience_coverage_by_density"]) == {
        "silent",
        "rest",
        "sparse",
        "normal",
        "dense",
        "fill",
    }


def test_render_chart_alignment_markdown_includes_summary_and_chart_rows():
    benchmark = build_chart_alignment_benchmark(
        [_chart_metrics()],
        fixture_count=1,
        use_beatnet=False,
        style="technical",
        density="auto",
        special_notes=False,
    )

    report = render_chart_alignment_markdown(benchmark)

    assert "# Synthetic chart alignment benchmark" in report
    assert "## Aggregate metrics" in report
    assert "## Per-course metrics" in report
    assert "`sample.wav`" in report
    assert "Unsupported-note ratio" in report
    assert "## Report-only QualityReport calibration" in report
    assert "### Salience coverage by density hint" in report
    assert "### Ground-truth comparison" in report
    assert "### Report-only metrics by course" in report


def test_benchmark_fixture_directory_generates_all_course_profiles(tmp_path, monkeypatch):
    ground_truth = _ground_truth()
    (tmp_path / "sample.wav").write_bytes(b"fixture")
    (tmp_path / "sample.events.json").write_text(
        json.dumps(ground_truth),
        encoding="utf-8",
    )
    feature_bars = [_feature_bar()]
    plan = ResolutionPlan(
        canonical_grids_per_bar=48,
        base_resolution=4,
        bar_resolutions=[4],
    )
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        benchmark_chart_alignment,
        "analyze_audio",
        lambda *_args, **_kwargs: SimpleNamespace(
            analyzer="test-analyzer",
            bpm=120.0,
            offset=0.0,
            time_signature="4/4",
        ),
    )
    monkeypatch.setattr(
        benchmark_chart_alignment,
        "build_bar_features",
        lambda _analysis: feature_bars,
    )
    monkeypatch.setattr(
        benchmark_chart_alignment,
        "analyze_song_structure",
        lambda bars: SimpleNamespace(bars=bars),
    )
    monkeypatch.setattr(
        benchmark_chart_alignment,
        "build_resolution_plan",
        lambda _analysis, _bars: plan,
    )

    def fake_generate(_bars, *, course, level, **_kwargs):
        calls.append((course, level))
        return [ChartBar(index=0, notes="1111")]

    monkeypatch.setattr(
        benchmark_chart_alignment,
        "generate_fallback_chart_bars",
        fake_generate,
    )

    benchmark = benchmark_chart_alignment.benchmark_fixture_directory(tmp_path)

    assert benchmark["fixture_count"] == 1
    assert benchmark["chart_count"] == 4
    assert calls == [
        ("Easy", 3),
        ("Easy", 3),
        ("Normal", 5),
        ("Normal", 5),
        ("Hard", 7),
        ("Hard", 7),
        ("Oni", 10),
        ("Oni", 10),
    ]
    assert all(item["deterministic"] for item in benchmark["charts"])
    assert benchmark["report_only_calibration"]["chart_count"] == 4
    assert all("report_only" in item for item in benchmark["charts"])


def test_benchmark_main_writes_json_and_markdown_reports(tmp_path, monkeypatch):
    benchmark = build_chart_alignment_benchmark(
        [_chart_metrics()],
        fixture_count=1,
        use_beatnet=False,
        style="technical",
        density="auto",
        special_notes=False,
    )
    output_path = tmp_path / "baseline.json"
    report_path = tmp_path / "baseline.md"
    monkeypatch.setattr(
        benchmark_chart_alignment,
        "benchmark_fixture_directory",
        lambda *_args, **_kwargs: benchmark,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_chart_alignment.py",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert benchmark_chart_alignment.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "# Synthetic chart alignment benchmark" in report_path.read_text(
        encoding="utf-8"
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_chart_alignment_baseline_covers_all_fixtures_and_courses():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    event_files = sorted(FIXTURE_DIR.glob("*.events.json"))

    assert baseline["schema_version"] == 1
    assert baseline["benchmark_version"] == "chart-alignment-v1"
    assert baseline["fixture_ground_truth_schema_version"] == 1
    assert baseline["fixture_count"] == len(event_files) == 17
    assert baseline["chart_count"] == len(event_files) * 4 == 68
    assert baseline["settings"] == {
        "use_beatnet": False,
        "style": "technical",
        "density": "auto",
        "special_notes": False,
        "note_onset_tolerance_seconds": 0.05,
        "beat_evidence_tolerance_seconds": 0.07,
    }
    assert set(baseline["course_summaries"]) == {"Easy", "Normal", "Hard", "Oni"}
    assert {item["audio"] for item in baseline["charts"]} == {
        path.name.removesuffix(".events.json") + ".wav" for path in event_files
    }
    for key in (
        "note_onset",
        "strong_onset_response",
        "downbeat_response",
        "fill_onset_response",
    ):
        metric = baseline["summary"][key]
        assert 0.0 <= metric["precision"] <= 1.0
        assert 0.0 <= metric["recall"] <= 1.0
        assert 0.0 <= metric["f1"] <= 1.0
    assert 0.0 <= baseline["summary"]["unsupported_note_ratio"] <= 1.0
    assert baseline["summary"]["deterministic_rate"] == 1.0
    calibration = baseline["report_only_calibration"]
    assert calibration["chart_count"] == 68
    assert all("report_only" in item for item in baseline["charts"])
    assert set(calibration["course_summaries"]) == {"Easy", "Normal", "Hard", "Oni"}
    for key in (
        "note_onset_alignment",
        "strong_onset_response",
        "downbeat_response",
        "unsupported_note_rate",
        "silent_range_violation_rate",
        "fill_burst_alignment",
    ):
        assert 0.0 <= calibration["summary"][key]["value"] <= 1.0
    assert calibration["summary"]["rhythmic_quantization_error"]["value"] is not None
    assert set(calibration["summary"]["salience_coverage_by_density"]) == {
        "silent",
        "rest",
        "sparse",
        "normal",
        "dense",
        "fill",
    }

    report = BASELINE_REPORT_PATH.read_text(encoding="utf-8")
    assert "# Synthetic chart alignment benchmark" in report
    assert "chart-alignment-v1" in report
    assert "Unsupported-note ratio" in report
    assert "## Report-only QualityReport calibration" in report
    assert "### Salience coverage by density hint" in report
    assert "### Ground-truth comparison" in report
    assert all(
        f"`{path.name.removesuffix('.events.json')}.wav`" in report
        for path in event_files
    )
