import importlib.util
import json
from pathlib import Path
import sys

import pytest

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.audio.spectral import SpectralAnalysisRaw
from tja_ai_chartgen.evaluation.audio_benchmark import (
    build_audio_benchmark,
    build_fixture_audio_metrics,
    match_events,
    render_audio_benchmark_markdown,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
BASELINE_PATH = FIXTURE_DIR / "audio_benchmark_baseline.json"
BASELINE_REPORT_PATH = FIXTURE_DIR / "audio_benchmark_baseline.md"
TOOL_PATH = Path(__file__).parents[1] / "tools" / "benchmark_audio_fixtures.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("benchmark_audio_fixtures", TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
benchmark_audio_fixtures = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(benchmark_audio_fixtures)


def _ground_truth() -> dict[str, object]:
    return {
        "schema_version": 1,
        "audio": "sample.wav",
        "bpm": 120.0,
        "time_signature": "4/4",
        "duration": 4.5,
        "first_downbeat": 0.5,
        "onsets": [0.5, 1.0, 1.5, 2.0, 2.5],
        "strong_onsets": [0.5, 2.5],
        "beats": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "downbeats": [0.5, 2.5],
        "low_band_onsets": [],
        "high_band_onsets": [],
        "silent_ranges": [[0.0, 0.5]],
        "fill_ranges": [],
        "sections": [],
    }


def _analysis() -> AudioAnalysisRaw:
    return AudioAnalysisRaw(
        bpm=120.0,
        beat_times=[0.51, 1.01, 1.51, 2.01, 2.51, 3.01, 3.51, 4.01],
        onset_times=[0.51, 1.01, 1.49, 2.01, 2.51],
        onset_strengths=[],
        duration=4.5,
        offset=0.5,
        time_signature="4/4",
        analyzer="test-analyzer",
    )


def test_match_events_uses_one_to_one_tolerance_matching():
    metrics = match_events(
        [0.5, 1.0, 1.5],
        [0.48, 1.04, 1.8],
        tolerance_seconds=0.05,
    )

    assert metrics == {
        "reference_count": 3,
        "estimated_count": 3,
        "true_positive_count": 2,
        "false_positive_count": 1,
        "false_negative_count": 1,
        "precision": pytest.approx(2 / 3, abs=1e-6),
        "recall": pytest.approx(2 / 3, abs=1e-6),
        "f1": pytest.approx(2 / 3, abs=1e-6),
        "mean_absolute_error_seconds": 0.03,
        "max_absolute_error_seconds": 0.04,
    }


def test_build_fixture_audio_metrics_derives_downbeats_from_offset_and_meter():
    metrics = build_fixture_audio_metrics(_ground_truth(), _analysis())

    assert metrics["bpm_absolute_error"] == 0.0
    assert metrics["meter_correct"] is True
    assert metrics["downbeat_source"] == "derived-offset-meter"
    assert metrics["first_downbeat_absolute_error_seconds"] == 0.0
    assert metrics["onset"]["true_positive_count"] == 5
    assert metrics["strong_onset"]["recall"] == 1.0
    assert metrics["beat"]["f1"] == 1.0
    assert metrics["downbeat"]["f1"] == 1.0
    assert metrics["silent_onset_false_positive_count"] == 0
    assert metrics["resolution"]["base_resolution"] == 16
    assert metrics["resolution"]["expressible_event_ratio"] == 1.0
    assert metrics["resolution"]["under_resolved_bar_count"] == 0
    assert metrics["resolution"]["unnecessary_high_resolution_bar_count"] == 0


def test_resolution_metrics_identify_minimum_grid_without_unnecessary_upgrade():
    ground_truth = _ground_truth()
    analysis = _analysis().model_copy(
        update={
            "beat_times": ground_truth["beats"],
            "onset_times": ground_truth["onsets"],
        }
    )

    resolution = build_fixture_audio_metrics(ground_truth, analysis)["resolution"]

    assert resolution["base_resolution"] == 16
    assert resolution["expressible_event_ratio"] == 1.0
    assert resolution["expected_minimum_resolution_counts"] == {"16": 2}
    assert resolution["unnecessary_high_resolution_bar_count"] == 0


def test_build_fixture_audio_metrics_covers_band_pickup_fill_and_resolution():
    ground_truth = _ground_truth() | {
        "first_downbeat": 1.0,
        "onsets": [0.5, 1.0, 1.5, 2.0],
        "strong_onsets": [1.0],
        "beats": [1.0, 1.5, 2.0, 2.5],
        "downbeats": [1.0],
        "low_band_onsets": [0.5],
        "high_band_onsets": [1.5],
        "fill_ranges": [[1.5, 2.1]],
    }
    low_envelope = [0.0] * 46
    high_envelope = [0.0] * 46
    low_envelope[5] = 1.0
    high_envelope[15] = 1.0
    analysis = _analysis().model_copy(
        update={
            "beat_times": [1.01, 1.51, 2.01, 2.51],
            "onset_times": [0.51, 1.01, 1.49, 2.01],
            "offset": 1.0,
            "sample_rate": 100,
            "hop_length": 10,
            "spectral": SpectralAnalysisRaw(
                status="complete",
                frame_count=46,
                low_onset_envelope=low_envelope,
                high_onset_envelope=high_envelope,
            ),
        }
    )

    metrics = build_fixture_audio_metrics(ground_truth, analysis)

    assert metrics["low_band_onset"]["f1"] == 1.0
    assert metrics["high_band_onset"]["f1"] == 1.0
    assert metrics["pickup_onset"]["recall"] == 1.0
    assert metrics["fill_onset"]["recall"] == 1.0
    assert metrics["resolution"]["pre_downbeat_event_count"] == 1
    assert metrics["resolution"]["evaluated_event_count"] == 3


def test_build_audio_benchmark_aggregates_fixture_counts():
    fixture_metrics = [build_fixture_audio_metrics(_ground_truth(), _analysis())]

    benchmark = build_audio_benchmark(fixture_metrics, use_beatnet=False)

    assert benchmark["schema_version"] == 2
    assert benchmark["benchmark_version"] == "audio-alignment-v2"
    assert benchmark["fixture_count"] == 1
    assert benchmark["summary"]["onset"]["true_positive_count"] == 5
    assert benchmark["summary"]["beat"]["f1"] == 1.0
    assert benchmark["summary"]["meter_accuracy"] == 1.0
    assert benchmark["summary"]["resolution"]["expressible_event_ratio"] == 1.0


def test_render_audio_benchmark_markdown_includes_summary_and_fixture_rows():
    fixture_metrics = [build_fixture_audio_metrics(_ground_truth(), _analysis())]
    benchmark = build_audio_benchmark(fixture_metrics, use_beatnet=False)

    report = render_audio_benchmark_markdown(benchmark)

    assert "# Synthetic audio fixture benchmark" in report
    assert "## Aggregate metrics" in report
    assert "## Tempo, meter, and resolution" in report
    assert "`sample.wav`" in report
    assert "Resolution expressible-event ratio" in report


def test_benchmark_fixture_directory_analyzes_ground_truth_audio(tmp_path, monkeypatch):
    ground_truth = _ground_truth()
    (tmp_path / "sample.wav").write_bytes(b"fixture")
    (tmp_path / "sample.events.json").write_text(
        json.dumps(ground_truth),
        encoding="utf-8",
    )
    calls: list[tuple[Path, bool]] = []

    def fake_analyze_audio(path: Path, use_beatnet: bool = False) -> AudioAnalysisRaw:
        calls.append((path, use_beatnet))
        return _analysis()

    monkeypatch.setattr(benchmark_audio_fixtures, "analyze_audio", fake_analyze_audio)

    benchmark = benchmark_audio_fixtures.benchmark_fixture_directory(tmp_path)

    assert calls == [(tmp_path / "sample.wav", False)]
    assert benchmark["fixture_count"] == 1
    assert benchmark["fixtures"][0]["audio"] == "sample.wav"


def test_benchmark_main_writes_json_and_markdown_reports(tmp_path, monkeypatch):
    benchmark = build_audio_benchmark(
        [build_fixture_audio_metrics(_ground_truth(), _analysis())],
        use_beatnet=False,
    )
    output_path = tmp_path / "baseline.json"
    report_path = tmp_path / "baseline.md"
    monkeypatch.setattr(
        benchmark_audio_fixtures,
        "benchmark_fixture_directory",
        lambda *_args, **_kwargs: benchmark,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_audio_fixtures.py",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert benchmark_audio_fixtures.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert "# Synthetic audio fixture benchmark" in report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_audio_benchmark_baseline_covers_all_ground_truth_files():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    event_files = sorted(FIXTURE_DIR.glob("*.events.json"))

    assert baseline["schema_version"] == 2
    assert baseline["benchmark_version"] == "audio-alignment-v2"
    assert baseline["fixture_ground_truth_schema_version"] == 1
    assert baseline["settings"]["use_beatnet"] is False
    assert baseline["settings"]["band_onset_threshold"] == 0.25
    assert baseline["fixture_count"] == len(event_files) == 17
    assert {item["audio"] for item in baseline["fixtures"]} == {
        path.name.removesuffix(".events.json") + ".wav" for path in event_files
    }
    for metric_name in (
        "onset",
        "strong_onset",
        "beat",
        "downbeat",
        "low_band_onset",
        "high_band_onset",
        "pickup_onset",
        "fill_onset",
    ):
        metric = baseline["summary"][metric_name]
        assert 0.0 <= metric["precision"] <= 1.0
        assert 0.0 <= metric["recall"] <= 1.0
        assert 0.0 <= metric["f1"] <= 1.0

    resolution = baseline["summary"]["resolution"]
    assert 0.0 <= resolution["expressible_event_ratio"] <= 1.0
    assert resolution["evaluated_event_count"] > 0
    assert resolution["selected_resolution_counts"]
    assert resolution["expected_minimum_resolution_counts"]

    report = BASELINE_REPORT_PATH.read_text(encoding="utf-8")
    assert "# Synthetic audio fixture benchmark" in report
    assert "audio-alignment-v2" in report
    assert "Resolution expressible-event ratio" in report
    assert all(f"`{item['audio']}`" in report for item in baseline["fixtures"])
