import pytest

from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import (
    BarFeature,
    ChartBar,
    GridFeature,
    InstrumentBarFeature,
    InstrumentGridFeature,
    ResolutionDecision,
    ResolutionPlan,
)
from tja_ai_chartgen.tja.quality import build_quality_report, pattern_counts


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
    assert report.note_onset_aligned_count == 0
    assert report.note_onset_evaluated_count == 0
    assert report.note_onset_alignment == 1.0
    assert report.strong_onset_responded_count == 0
    assert report.strong_onset_evaluated_count == 0
    assert report.strong_onset_response == 1.0
    assert report.downbeat_responded_count == 0
    assert report.downbeat_evaluated_count == 0
    assert report.downbeat_response == 1.0
    assert report.unsupported_note_count == 0
    assert report.unsupported_note_evaluated_count == 0
    assert report.unsupported_note_rate == 0.0
    assert report.silent_range_evaluated_bar_count == 0
    assert report.silent_range_violation_count == 0
    assert report.silent_range_violation_rate == 0.0


def test_quality_report_is_independent_of_notes_resolution():
    feature = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.5,
        time_signature="4/4",
        grids_per_bar=48,
        onset_grids=[0, 12, 24, 36],
        accent_grids=[0, 12, 24, 36],
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
    )

    reports = []
    for resolution in (16, 24, 48):
        notes = ["0"] * resolution
        for index, note in zip(
            (0, resolution // 4, resolution // 2, resolution * 3 // 4),
            "1212",
            strict=True,
        ):
            notes[index] = note
        reports.append(build_quality_report([_chart_bar(0, "".join(notes))], [feature]))

    comparable = [
        (
            report.playable_note_count,
            report.average_notes_per_second,
            report.peak_bar_notes_per_second,
            report.accent_candidate_count,
            report.accent_hit_count,
            report.accent_coverage_rate,
            report.note_onset_aligned_count,
            report.note_onset_evaluated_count,
            report.note_onset_alignment,
            report.strong_onset_responded_count,
            report.strong_onset_evaluated_count,
            report.strong_onset_response,
            report.downbeat_responded_count,
            report.downbeat_evaluated_count,
            report.downbeat_response,
            report.unsupported_note_count,
            report.unsupported_note_evaluated_count,
            report.unsupported_note_rate,
            report.silent_range_evaluated_bar_count,
            report.silent_range_violation_count,
            report.silent_range_violation_rate,
            report.don_count,
            report.ka_count,
        )
        for report in reports
    ]
    assert comparable[0] == comparable[1] == comparable[2]


def test_quality_report_records_note_onset_alignment_with_time_tolerance():
    feature = _reliable_onset_feature(0, start_time=0.0, end_time=2.0)
    notes = ["0"] * 48
    notes[11] = "1"  # 距 grid 12 的可靠瞬态约 42 ms。
    notes[14] = "2"  # 距 grid 12 的可靠瞬态约 83 ms。
    notes[24] = "5"  # 特殊音符不属于普通 note 指标。

    report = build_quality_report([_chart_bar(0, "".join(notes))], [feature])

    assert report.note_onset_aligned_count == 1
    assert report.note_onset_evaluated_count == 2
    assert report.note_onset_alignment == 0.5


def test_quality_report_note_onset_alignment_uses_cross_bar_tolerance():
    features = [
        _reliable_onset_feature(0, start_time=0.0, end_time=1.0),
        _reliable_onset_feature(1, start_time=1.0, end_time=2.0),
    ]
    first_notes = ["0"] * 48
    first_notes[47] = "1"

    report = build_quality_report(
        [_chart_bar(0, "".join(first_notes)), _chart_bar(1, "0" * 48)],
        features,
    )

    assert report.note_onset_aligned_count == 1
    assert report.note_onset_evaluated_count == 1
    assert report.note_onset_alignment == 1.0


def test_quality_report_note_onset_alignment_requires_reliable_transient():
    feature = _feature(0, energy=0.8).model_copy(
        update={
            "grid_features": [
                GridFeature(grid=0, beat=0, downbeat=True, activity=0.8),
            ],
            "beat_grids": [0, 4, 8, 12],
            "downbeat_grid": 0,
        }
    )

    report = build_quality_report([_chart_bar(0, "1000000000000000")], [feature])

    assert report.note_onset_aligned_count == 0
    assert report.note_onset_evaluated_count == 1
    assert report.note_onset_alignment == 0.0


def test_quality_report_records_strong_onset_response_from_notes_and_roll_ranges():
    feature = _reliable_onset_feature(0, start_time=0.0, end_time=2.0)
    notes = ["0"] * 48
    notes[0] = "1"
    notes[10] = "5"
    notes[26] = "8"

    report = build_quality_report([_chart_bar(0, "".join(notes))], [feature])

    assert report.strong_onset_evaluated_count == 4
    assert report.strong_onset_responded_count == 3
    assert report.strong_onset_response == 0.75
    assert report.downbeat_evaluated_count == 1
    assert report.downbeat_responded_count == 1
    assert report.downbeat_response == 1.0
    assert report.unsupported_note_evaluated_count == 1
    assert report.unsupported_note_count == 0


def test_quality_report_matches_clustered_strong_onsets_one_to_one():
    feature = _reliable_onset_feature(
        0,
        start_time=0.0,
        end_time=1.0,
        onset_grids=[0, 12, 13, 24, 36],
    )
    notes = ["0"] * 48
    notes[12] = "1"

    report = build_quality_report([_chart_bar(0, "".join(notes))], [feature])

    assert report.strong_onset_evaluated_count == 5
    assert report.strong_onset_responded_count == 1
    assert report.strong_onset_response == 0.2


def test_quality_report_strong_onset_response_accepts_cross_bar_roll_range():
    features = [
        _reliable_onset_feature(0, start_time=0.0, end_time=1.0),
        _reliable_onset_feature(1, start_time=1.0, end_time=2.0),
    ]
    first_notes = ["0"] * 48
    second_notes = ["0"] * 48
    first_notes[40] = "5"
    second_notes[8] = "8"

    report = build_quality_report(
        [
            _chart_bar(0, "".join(first_notes)),
            _chart_bar(1, "".join(second_notes)),
        ],
        features,
    )

    assert report.strong_onset_evaluated_count == 8
    assert report.strong_onset_responded_count == 1
    assert report.strong_onset_response == 0.125
    assert report.downbeat_responded_count == 0


def test_quality_report_downbeat_response_is_cross_bar_and_normal_note_only():
    features = [
        _reliable_onset_feature(0, start_time=0.0, end_time=1.0),
        _reliable_onset_feature(1, start_time=1.0, end_time=2.0),
    ]
    first_notes = ["0"] * 48
    first_notes[0] = "5"
    first_notes[4] = "8"
    first_notes[47] = "1"

    report = build_quality_report(
        [_chart_bar(0, "".join(first_notes)), _chart_bar(1, "0" * 48)],
        features,
    )

    assert report.strong_onset_evaluated_count == 8
    assert report.strong_onset_responded_count == 2
    assert report.strong_onset_response == 0.25
    assert report.downbeat_evaluated_count == 2
    assert report.downbeat_responded_count == 1
    assert report.downbeat_response == 0.5


def test_quality_report_records_supported_and_unsupported_normal_notes():
    feature = _unsupported_evidence_feature()
    notes = ["0"] * 48
    for grid in (0, 24, 30, 31, 40):
        notes[grid] = "1"

    report = build_quality_report([_chart_bar(0, "".join(notes))], [feature])

    assert report.unsupported_note_evaluated_count == 5
    assert report.unsupported_note_count == 1
    assert report.unsupported_note_rate == 0.2


def test_quality_report_rhythm_connection_does_not_chain_indefinitely():
    feature = _unsupported_evidence_feature()
    notes = ["0"] * 48
    for grid in (30, 36, 42):
        notes[grid] = "1"

    report = build_quality_report([_chart_bar(0, "".join(notes))], [feature])

    assert report.unsupported_note_evaluated_count == 3
    assert report.unsupported_note_count == 1
    assert report.unsupported_note_rate == pytest.approx(1 / 3, abs=1e-6)


def test_quality_report_records_internal_and_edge_silence_violations():
    features = [
        _feature(0, energy=0.0),
        _feature(1, energy=0.8),
        _feature(2, energy=0.0),
        _feature(3, energy=0.8),
        _feature(4, energy=0.0).model_copy(update={"phrase_position": "song_end"}),
    ]
    bars = [
        _chart_bar(0, "1000000000000000"),
        _chart_bar(1, "1000000000000000"),
        _chart_bar(2, "5000000080000000"),
        _chart_bar(3, "0000000000000000"),
        _chart_bar(4, "2000000000000000"),
    ]

    report = build_quality_report(bars, features)

    assert report.silent_bar_note_count == 2
    assert report.silent_range_evaluated_bar_count == 3
    assert report.silent_range_violation_count == 3
    assert report.silent_range_violation_rate == 0.75


def test_pattern_counts_normalizes_equivalent_resolutions():
    notes_16 = "1000200010002000"
    notes_48 = ["0"] * 48
    for index, note in zip((0, 12, 24, 36), "1212", strict=True):
        notes_48[index] = note

    chart_bars = [_chart_bar(0, notes_16), _chart_bar(1, "".join(notes_48))]
    counts = pattern_counts(chart_bars)
    report = build_quality_report(chart_bars, [_feature(0), _feature(1)])

    assert list(counts.values()) == [2]
    assert report.repeated_bar_count == 1
    assert report.repeated_bar_rate == 0.5


def test_quality_report_records_structure_and_resolution_metrics():
    features = [
        _feature(
            index,
            energy=0.2 + index * 0.15,
            energy_percentile=0.2 + index * 0.2,
            grids=48,
            phrase_id=0,
            phrase_progress=index / 3,
            transition_role="build_up",
            section_id="section-1",
        )
        for index in range(4)
    ]
    features.extend(
        [
            _feature(
                4,
                energy=0.95,
                energy_percentile=1.0,
                grids=48,
                phrase_id=1,
                transition_role="peak",
                section_id="section-2",
                fill_candidate=True,
            ),
            _feature(
                5,
                energy=0.25,
                energy_percentile=0.1,
                grids=48,
                phrase_id=1,
                phrase_progress=1.0,
                transition_role="stable",
                section_id="section-2",
            ),
        ]
    )
    chart_bars = [
        _chart_bar(index, _notes_with_hits(hit_count, 48))
        for index, hit_count in enumerate([2, 4, 6, 8, 10, 3])
    ]
    plan = ResolutionPlan(
        canonical_grids_per_bar=48,
        base_resolution=48,
        bar_resolutions=[48] * 6,
        decision=ResolutionDecision(
            selected_resolution=48,
            candidate_errors={"16": 0.5, "24": 0.25, "48": 0.05},
            evidence_count=24,
            confidence=0.8,
            reason="mixed subdivisions",
        ),
    )

    report = build_quality_report(chart_bars, features, plan)

    assert report.structure_density_correlation > 0
    assert report.build_up_slope_agreement == pytest.approx(1.0)
    assert report.peak_contrast > 0
    assert report.highlight_note_contrast > 0
    assert report.fill_candidate_precision == 1.0
    assert report.base_resolution == 48
    assert report.resolution_change_count == 0
    assert report.resolution_changes_per_100_bars == 0.0
    assert report.high_resolution_bar_ratio == 1.0
    assert report.resolution_quantization_error == 0.05
    assert report.avoided_resolution_changes == 1


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

    assert report.accent_candidate_count == 7
    assert report.accent_hit_count == 4
    assert report.accent_coverage_rate == pytest.approx(4 / 7)


def test_quality_report_caps_unified_accent_candidates_by_meter():
    feature = BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.9,
        onset_16=list(range(0, 16, 2)),
        beat_grids=[0, 4, 8, 12],
        downbeat_grid=0,
    )

    report = build_quality_report([_chart_bar(0, "1010101010101010")], [feature])

    assert report.accent_candidate_count == 4
    assert report.accent_hit_count == 4
    assert report.accent_coverage_rate == 1.0


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


def test_quality_report_records_instrument_alignment_metrics():
    first = _feature(0, energy=0.5, phrase_id=0).model_copy(
        update={
            "phrase_position": "phrase_start",
            "instrument": InstrumentBarFeature(
                vocal_activity=0.8,
                dominant_source="vocals",
                confidence=0.8,
            ),
            "instrument_grid_features": [
                InstrumentGridFeature(grid=0, vocal_onset=0.8),
            ],
            "beat_grids": [0, 4, 8, 12],
            "downbeat_grid": 0,
        }
    )
    second = _feature(1, energy=0.8, phrase_id=1, fill_candidate=True).model_copy(
        update={
            "instrument": InstrumentBarFeature(
                drum_activity=0.9,
                bass_activity=0.7,
                other_activity=0.5,
                dominant_source="drums",
                dominant_instrument="guitar",
                confidence=0.9,
            ),
            "instrument_grid_features": [
                InstrumentGridFeature(grid=0, bass_onset=0.8),
                InstrumentGridFeature(grid=4, drum_onset=0.9),
                InstrumentGridFeature(grid=12, drum_onset=0.8, accompaniment_onset=0.7),
            ],
            "beat_grids": [0, 4, 8, 12],
            "downbeat_grid": 0,
        }
    )
    chart_bars = [
        _chart_bar(0, "1000000000000000"),
        _chart_bar(1, "1000100000001000"),
    ]

    report = build_quality_report(chart_bars, [first, second])

    assert report.drum_onset_hit_coverage == 1.0
    assert report.bass_downbeat_alignment == 1.0
    assert report.vocal_phrase_response == 1.0
    assert report.instrument_transition_response == 1.0
    assert report.instrument_confident_bar_ratio == 1.0
    assert report.instrument_fill_support == 1.0


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
    energy_percentile: float = 0.0,
    phrase_id: int | None = None,
    phrase_progress: float = 0.0,
    transition_role: str = "stable",
    section_id: str | None = None,
    fill_candidate: bool = False,
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
        time_signature="4/4" if grids in {16, 24, 48} else "3/4",
        grids_per_bar=grids,
        onset_16=list(range(min(onset_count, grids))),
        section="unknown",
        energy_percentile=energy_percentile,
        phrase_id=phrase_id,
        phrase_progress=phrase_progress,
        transition_role=transition_role,
        section_id=section_id,
        fill_candidate=fill_candidate,
    )


def _unsupported_evidence_feature() -> BarFeature:
    onset_grids = [0, 4, 8, 16]
    activity_grids = [0.0] * 48
    activity_grids[30] = 0.9
    return BarFeature(
        index=0,
        start_time=0.0,
        end_time=2.0,
        energy=0.9,
        time_signature="4/4",
        grids_per_bar=48,
        onset_grids=onset_grids,
        activity_grids=activity_grids,
        grid_features=[
            GridFeature(
                grid=grid,
                onset=True,
                strength=1.0,
                activity=0.9 if grid == 0 else 0.0,
                downbeat=grid == 0,
            )
            for grid in onset_grids
        ],
        beat_grids=[0, 24],
        downbeat_grid=0,
        onset_count=len(onset_grids),
    )


def _reliable_onset_feature(
    index: int,
    *,
    start_time: float,
    end_time: float,
    onset_grids: list[int] | None = None,
) -> BarFeature:
    resolved_onset_grids = onset_grids or [0, 12, 24, 36]
    return BarFeature(
        index=index,
        start_time=start_time,
        end_time=end_time,
        energy=0.9,
        time_signature="4/4",
        grids_per_bar=48,
        onset_grids=resolved_onset_grids,
        activity_grids=list(range(48)),
        grid_features=[
            GridFeature(
                grid=grid,
                onset=True,
                accent=grid in {0, 24},
                beat=grid // 12 if grid % 12 == 0 else None,
                downbeat=grid == 0,
                strength=1.0,
                activity=0.9,
            )
            for grid in resolved_onset_grids
        ],
        beat_grids=[0, 12, 24, 36],
        downbeat_grid=0,
        onset_count=len(resolved_onset_grids),
    )


def _chart_bar(index: int, notes: str, *, time_signature: str = "4/4") -> ChartBar:
    return ChartBar(index=index, notes=notes, time_signature=time_signature)


def _notes_with_hits(hit_count: int, resolution: int = 16) -> str:
    notes = ["0"] * resolution
    for index in range(hit_count):
        notes[index] = "1" if index % 2 == 0 else "2"
    return "".join(notes)
