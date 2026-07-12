import pytest

from tja_ai_chartgen.tja.model import BarFeature, ChartBar
from tja_ai_chartgen.tja.quality import build_quality_report


def test_quality_report_records_density_compliance_and_silent_notes():
    features = [
        _feature(0, energy=0.0),
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


def _feature(
    index: int,
    *,
    energy: float = 0.0,
    onset_count: int = 0,
    grids: int = 16,
) -> BarFeature:
    return BarFeature(
        index=index,
        start_time=float(index),
        end_time=float(index + 1),
        energy=energy,
        time_signature="4/4" if grids == 16 else "3/4",
        grids_per_bar=grids,
        onset_16=list(range(min(onset_count, grids))),
        section="unknown",
    )


def _chart_bar(index: int, notes: str, *, time_signature: str = "4/4") -> ChartBar:
    return ChartBar(index=index, notes=notes, time_signature=time_signature)
