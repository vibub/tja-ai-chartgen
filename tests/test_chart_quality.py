import pytest

from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import BarFeature, ChartBar
from tja_ai_chartgen.tja.quality import build_quality_report


def test_quality_report_records_density_compliance_and_silent_notes():
    features = [
        _feature(
            0,
            energy=0.068,
            onset_count=3,
            rms_dbfs=-59.7,
            peak_rms_dbfs=-46.7,
            relative_rms_db=-53.5,
            sustained_activity_ratio=0.098,
        ),
        _feature(1, energy=0.5, onset_count=8),
        _feature(2, energy=0.0),
    ]
    bars = [
        _chart_bar(0, "1000000000000000"),
        _chart_bar(1, "1000000000000000"),
        _chart_bar(2, "0000000000000000"),
    ]

    report = build_quality_report(bars, features)

    assert report.density_evaluated_bars == 3
    assert report.density_compliant_bars == 1
    assert report.density_compliance_rate == pytest.approx(1 / 3)
    assert report.silent_bar_note_count == 1


def test_quality_report_records_empty_runs_and_exact_pattern_repetition():
    features = [_feature(index, energy=0.3) for index in range(7)]
    bars = [
        _chart_bar(0, "1000100010001000"),
        _chart_bar(1, "0000000000000000"),
        _chart_bar(2, "0000000000000000"),
        _chart_bar(3, "2000200020002000"),
        _chart_bar(4, "1000100010001000"),
        _chart_bar(5, "1000100010001000"),
        _chart_bar(6, "0000000000000000"),
    ]

    report = build_quality_report(bars, features)

    assert report.longest_empty_bar_run == 2
    assert report.repeated_bar_count == 2
    assert report.repeated_bar_rate == pytest.approx(2 / 4)


def test_quality_report_records_color_ratio_and_cross_bar_monochrome_run():
    features = [_feature(0, grids=12), _feature(1, grids=12), _feature(2, grids=12)]
    bars = [
        _chart_bar(0, "110011000000", time_signature="3/4"),
        _chart_bar(1, "001100200000", time_signature="3/4"),
        _chart_bar(2, "220052200000", time_signature="3/4"),
    ]

    report = build_quality_report(bars, features)

    assert report.don_count == 6
    assert report.ka_count == 5
    assert report.ka_ratio == pytest.approx(5 / 11)
    assert report.longest_monochrome_run == 6


def test_quality_report_handles_empty_chart_without_division_by_zero():
    report = build_quality_report([], [])

    assert report.bar_count == 0
    assert report.density_evaluated_bars == 0
    assert report.density_compliance_rate == 1.0
    assert report.repeated_bar_rate == 0.0
    assert report.ka_ratio == 0.0
    assert report.longest_empty_bar_run == 0
    assert report.longest_monochrome_run == 0
    assert report.playable_note_count == 0
    assert report.playable_duration_seconds == 0.0
    assert report.average_notes_per_second == 0.0
    assert report.peak_bar_notes_per_second == 0.0
    assert report.active_duration_seconds == 0.0
    assert report.active_average_notes_per_second == 0.0
    assert report.longest_note_stream_count == 0
    assert report.longest_note_stream_seconds == 0.0
    assert report.accent_candidate_count == 0
    assert report.accent_hit_count == 0
    assert report.accent_coverage_rate == 1.0


def test_quality_report_records_time_normalized_load():
    features = [
        _feature(0, energy=0.5, start_time=0.0, end_time=2.0),
        _feature(1, energy=0.5, start_time=2.0, end_time=3.0),
    ]
    bars = [
        _chart_bar(0, "1000100010001000"),
        _chart_bar(1, "1111000000000000"),
    ]

    report = build_quality_report(bars, features)

    assert report.playable_note_count == 8
    assert report.playable_duration_seconds == pytest.approx(3.0)
    assert report.average_notes_per_second == pytest.approx(8 / 3)
    assert report.peak_bar_notes_per_second == pytest.approx(4.0)


@pytest.mark.parametrize("end_time", [0.0, -1.0, float("inf"), float("nan")])
def test_quality_report_ignores_invalid_bar_durations(end_time):
    features = [_feature(0, energy=0.5, start_time=0.0, end_time=end_time)]
    bars = [_chart_bar(0, "1111000000000000")]

    report = build_quality_report(bars, features)

    assert report.playable_note_count == 0
    assert report.playable_duration_seconds == 0.0
    assert report.average_notes_per_second == 0.0
    assert report.peak_bar_notes_per_second == 0.0


def test_quality_report_records_special_note_duration_and_balloon_load():
    features = [
        _feature(0, energy=0.8, start_time=0.0, end_time=2.0),
        _feature(1, energy=0.9, start_time=2.0, end_time=3.0),
    ]
    bars = [
        _chart_bar(0, "0000500000008000"),
        ChartBar(
            index=1,
            notes="0000000700000080",
            balloon_counts=[6],
        ),
    ]

    report = build_quality_report(bars, features)

    assert report.drumroll_count == 1
    assert report.balloon_count == 1
    assert report.special_note_count == 2
    assert report.drumroll_duration_seconds == pytest.approx(1.0)
    assert report.balloon_duration_seconds == pytest.approx(7 / 16)
    assert report.special_note_duration_seconds == pytest.approx(23 / 16)
    assert report.balloon_required_hits == 6
    assert report.balloon_hits_per_second == pytest.approx(96 / 7)
    assert report.playable_note_count == 0
    assert report.average_notes_per_second == 0.0


def test_quality_report_excludes_sustained_markers_from_normal_note_count():
    features = [_feature(0, energy=0.8, start_time=0.0, end_time=2.0)]
    bars = [_chart_bar(0, "1234578000000000")]

    report = build_quality_report(bars, features)

    assert report.playable_note_count == 4
    assert report.average_notes_per_second == pytest.approx(2.0)


def test_quality_report_special_metrics_default_to_zero():
    report = build_quality_report([], [])

    assert report.drumroll_count == 0
    assert report.balloon_count == 0
    assert report.special_note_count == 0
    assert report.drumroll_duration_seconds == 0.0
    assert report.balloon_duration_seconds == 0.0
    assert report.special_note_duration_seconds == 0.0
    assert report.balloon_required_hits == 0
    assert report.balloon_hits_per_second == 0.0


def test_quality_report_uses_same_metrics_for_identical_ai_and_fallback_bars():
    features = [_feature(index, energy=0.3) for index in range(2)]
    ai_bars = [
        _chart_bar(0, "1000100010002000"),
        _chart_bar(1, "2000200010001000"),
    ]
    fallback_bars = [bar.model_copy() for bar in ai_bars]

    ai_report = build_quality_report(ai_bars, features)
    fallback_report = build_quality_report(fallback_bars, features)

    assert ai_report.model_dump() == fallback_report.model_dump()


def test_quality_report_records_active_load_and_cross_bar_note_stream():
    features = [
        _feature(0, energy=0.8, start_time=0.0, end_time=1.0),
        _feature(1, energy=0.8, start_time=1.0, end_time=2.0),
        _feature(2, energy=0.0, start_time=2.0, end_time=3.0),
    ]
    bars = [
        _chart_bar(0, "1000100010001000"),
        _chart_bar(1, "1000100010001000"),
        _chart_bar(2, "0000000000000000"),
    ]

    report = build_quality_report(bars, features)

    assert report.active_duration_seconds == pytest.approx(2.0)
    assert report.active_average_notes_per_second == pytest.approx(4.0)
    assert report.longest_note_stream_count == 8
    assert report.longest_note_stream_seconds == pytest.approx(1.75)


def test_quality_report_breaks_note_stream_at_long_gap_and_ignores_special_notes():
    features = [
        _feature(0, energy=0.8, start_time=0.0, end_time=1.0),
        _feature(1, energy=0.8, start_time=1.0, end_time=2.0),
    ]
    bars = [
        _chart_bar(0, "1111500000008000"),
        _chart_bar(1, "0000000000001111"),
    ]

    report = build_quality_report(bars, features)

    assert report.longest_note_stream_count == 4
    assert report.longest_note_stream_seconds == pytest.approx(3 / 16)


def test_quality_report_records_accent_coverage_for_supported_grids():
    features = [
        BarFeature(
            index=0,
            start_time=0.0,
            end_time=2.0,
            energy=0.8,
            onset_16=[0, 4, 8, 12],
            accent_16=[4, 8, 20],
            beat_grids=[0, 4, 8, 12],
            downbeat_grid=0,
        ),
        BarFeature(
            index=1,
            start_time=2.0,
            end_time=3.5,
            energy=0.8,
            time_signature="3/4",
            grids_per_bar=12,
            onset_16=[0, 4, 8],
            accent_16=[4, 8],
            beat_grids=[0, 4, 8],
            downbeat_grid=0,
        ),
    ]
    bars = [
        _chart_bar(0, "1000100000000000"),
        _chart_bar(1, "100000001000", time_signature="3/4"),
    ]

    report = build_quality_report(bars, features)

    assert report.accent_candidate_count == 6
    assert report.accent_hit_count == 4
    assert report.accent_coverage_rate == pytest.approx(2 / 3)


def test_feature_driven_fallback_quality_avoids_silence_and_exact_repetition():
    features = [
        _feature(0, energy=0.0),
        BarFeature(
            index=1,
            start_time=2.0,
            end_time=4.0,
            energy=0.5,
            onset_16=[0, 4, 8, 12],
            accent_16=[0, 8],
            beat_grids=[0, 4, 8, 12],
            downbeat_grid=0,
        ),
        BarFeature(
            index=2,
            start_time=4.0,
            end_time=6.0,
            energy=0.5,
            onset_16=[2, 6, 10, 14],
            accent_16=[2, 10],
            beat_grids=[0, 4, 8, 12],
            downbeat_grid=0,
        ),
        _feature(3, energy=0.0),
    ]

    chart_bars = generate_fallback_chart_bars(features, density="medium")
    report = build_quality_report(chart_bars, features)

    assert report.density_compliance_rate == 1.0
    assert report.silent_bar_note_count == 0
    assert report.repeated_bar_count == 0
    assert report.longest_monochrome_run <= 4


def _feature(
    index: int,
    *,
    energy: float = 0.0,
    onset_count: int = 0,
    grids: int = 16,
    start_time: float | None = None,
    end_time: float | None = None,
    rms_dbfs: float | None = None,
    peak_rms_dbfs: float | None = None,
    relative_rms_db: float | None = None,
    sustained_activity_ratio: float | None = None,
) -> BarFeature:
    return BarFeature(
        index=index,
        start_time=float(index if start_time is None else start_time),
        end_time=float(index + 1 if end_time is None else end_time),
        energy=energy,
        rms_dbfs=rms_dbfs,
        peak_rms_dbfs=peak_rms_dbfs,
        relative_rms_db=relative_rms_db,
        sustained_activity_ratio=sustained_activity_ratio,
        time_signature="4/4" if grids == 16 else "3/4",
        grids_per_bar=grids,
        onset_16=list(range(min(onset_count, grids))),
        section="unknown",
    )


def _chart_bar(index: int, notes: str, *, time_signature: str = "4/4") -> ChartBar:
    return ChartBar(index=index, notes=notes, time_signature=time_signature)
