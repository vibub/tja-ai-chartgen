import pytest

from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import BarFeature


def test_generate_fallback_chart_bars_outputs_valid_16_char_notes():
    bars = [
        BarFeature(index=0, start_time=0, end_time=2, energy=0.1),
        BarFeature(index=1, start_time=2, end_time=4, energy=0.5),
        BarFeature(index=2, start_time=4, end_time=6, energy=0.7),
        BarFeature(index=3, start_time=6, end_time=8, energy=0.9),
    ]

    chart_bars = generate_fallback_chart_bars(bars)

    assert len(chart_bars) == len(bars)
    for expected_index, chart_bar in enumerate(chart_bars):
        assert chart_bar.index == expected_index
        assert len(chart_bar.notes) == 16
        assert set(chart_bar.notes) <= set("01234")


def test_generate_fallback_chart_bars_density_overrides_energy():
    bars = [
        BarFeature(index=0, start_time=0, end_time=2, energy=0.95),
        BarFeature(index=1, start_time=2, end_time=4, energy=0.95),
    ]

    low_bars = generate_fallback_chart_bars(bars, density="low")
    max_bars = generate_fallback_chart_bars(bars, density="max")

    assert [bar.notes for bar in low_bars] == ["1000100010001000", "1000200010002000"]
    assert [bar.notes for bar in max_bars] == ["1122112233441122", "1211221211223344"]


def test_generate_fallback_chart_bars_applies_style_templates():
    bars = [
        BarFeature(index=0, start_time=0, end_time=2, energy=0.95),
        BarFeature(index=1, start_time=2, end_time=4, energy=0.95),
    ]

    performance_bars = generate_fallback_chart_bars(bars, style="performance", density="low")
    stamina_bars = generate_fallback_chart_bars(bars, style="stamina", density="medium")

    assert [bar.notes for bar in performance_bars] == ["3000400030004000", "1000300010004000"]
    assert [bar.notes for bar in stamina_bars] == ["2220222022202220", "1110111011101110"]


def test_generate_fallback_chart_bars_rejects_invalid_density():
    bars = [BarFeature(index=0, start_time=0, end_time=2, energy=0.5)]

    with pytest.raises(ValueError, match="Invalid density"):
        generate_fallback_chart_bars(bars, density="extreme")


def test_generate_fallback_chart_bars_trims_patterns_for_three_four():
    bars = [
        BarFeature(
            index=0,
            start_time=0,
            end_time=1.5,
            energy=0.5,
            time_signature="3/4",
            grids_per_bar=12,
        )
    ]

    chart_bars = generate_fallback_chart_bars(bars, density="low")

    assert chart_bars[0].notes == "100010001000"
    assert chart_bars[0].time_signature == "3/4"


def test_generate_fallback_chart_bars_can_emit_rolls_and_balloons():
    bars = [
        BarFeature(index=index, start_time=index * 2, end_time=(index + 1) * 2, energy=0.9)
        for index in range(8)
    ]

    chart_bars = generate_fallback_chart_bars(bars, density="high", special_notes=True)

    assert chart_bars[3].notes == "5000000080000000"
    assert chart_bars[7].notes == "7000000080000000"
    assert chart_bars[7].balloon_counts == [8]
