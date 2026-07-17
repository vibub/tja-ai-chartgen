import pytest

from tja_ai_chartgen.audio.analyze import AudioAnalysisRaw
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.meter import get_meter_spec
from tja_ai_chartgen.features.resolution import build_resolution_plan


def _raw_for_ticks(
    ticks: list[int],
    *,
    time_signature: str = "4/4",
    strengths: dict[int, float] | None = None,
) -> AudioAnalysisRaw:
    meter = get_meter_spec(time_signature)
    bpm = 120.0
    bar_length = meter.beats_per_bar * 60.0 / bpm
    onset_times = [bar_length * tick / meter.grids_per_bar for tick in ticks]
    sample_rate = 480
    hop_length = 1
    onset_strengths: list[float] = []
    if strengths is not None:
        onset_strengths = [0.0] * (round(bar_length * sample_rate) + 1)
        for tick, time in zip(ticks, onset_times, strict=True):
            onset_strengths[round(time * sample_rate)] = strengths.get(tick, 1.0)
    return AudioAnalysisRaw(
        bpm=bpm,
        beat_times=[],
        onset_times=onset_times,
        onset_strengths=onset_strengths,
        duration=bar_length,
        offset=0.0,
        sample_rate=sample_rate,
        hop_length=hop_length,
        time_signature=time_signature,
    )


@pytest.mark.parametrize(
    ("ticks", "expected_resolution"),
    [
        (list(range(0, 48, 3)), 16),
        (list(range(0, 48, 4)), 24),
        (sorted(set(range(0, 48, 3)) | set(range(0, 48, 4))), 48),
    ],
)
def test_build_resolution_plan_selects_straight_triplet_or_mixed_grid(
    ticks, expected_resolution
):
    raw = _raw_for_ticks(ticks)
    bars = build_bar_features(raw)

    plan = build_resolution_plan(raw, bars)

    assert plan.canonical_grids_per_bar == 48
    assert plan.base_resolution == expected_resolution
    assert plan.bar_resolutions == [expected_resolution]
    assert plan.change_points == []
    assert plan.policy_version == "song-global-v1"
    assert plan.decision is not None
    assert plan.decision.selected_resolution == expected_resolution
    assert plan.decision.evidence_count == len(ticks)


@pytest.mark.parametrize(
    ("time_signature", "triplet_resolution", "mixed_resolution"),
    [("3/4", 18, 36), ("6/8", 18, 36)],
)
def test_build_resolution_plan_supports_three_four_and_six_eight(
    time_signature, triplet_resolution, mixed_resolution
):
    meter = get_meter_spec(time_signature)
    triplet_ticks = list(range(0, meter.grids_per_bar, 4))
    mixed_ticks = sorted(set(triplet_ticks) | set(range(0, meter.grids_per_bar, 3)))

    triplet_raw = _raw_for_ticks(triplet_ticks, time_signature=time_signature)
    mixed_raw = _raw_for_ticks(mixed_ticks, time_signature=time_signature)

    assert build_resolution_plan(
        triplet_raw, build_bar_features(triplet_raw)
    ).base_resolution == triplet_resolution
    assert build_resolution_plan(
        mixed_raw, build_bar_features(mixed_raw)
    ).base_resolution == mixed_resolution


def test_build_resolution_plan_uses_legacy_resolution_without_reliable_onsets():
    raw = _raw_for_ticks([])
    bars = build_bar_features(raw)

    plan = build_resolution_plan(raw, bars)

    assert plan.base_resolution == 16
    assert plan.bar_resolutions == [16]
    assert plan.decision is not None
    assert plan.decision.evidence_count == 0
    assert plan.decision.confidence == 0.0
    assert "insufficient" in plan.decision.reason


def test_build_resolution_plan_prefers_stable_low_resolution_when_all_candidates_are_uncertain():
    raw = _raw_for_ticks([0.4, 12.4, 24.4, 36.4])
    bars = build_bar_features(raw)

    plan = build_resolution_plan(raw, bars)

    assert plan.base_resolution == 16
    assert plan.decision is not None
    assert plan.decision.confidence == 0.0
    assert "no candidate met" in plan.decision.reason
    assert "lowest stable resolution" in plan.decision.reason


def test_build_resolution_plan_ignores_weak_off_grid_noise():
    raw = _raw_for_ticks(
        [0, 12, 24, 36, 1],
        strengths={0: 1.0, 12: 1.0, 24: 1.0, 36: 1.0, 1: 0.05},
    )
    bars = build_bar_features(raw)

    plan = build_resolution_plan(raw, bars)

    assert plan.base_resolution == 16
    assert plan.decision is not None
    assert plan.decision.evidence_count == 4


def test_build_resolution_plan_upgrades_complete_high_confidence_phrase_only():
    raw = _raw_for_bar_patterns(["straight"] * 4 + ["triplet"] * 4 + ["straight"] * 4)
    bars = build_bar_features(raw)
    structured_bars = [
        bar.model_copy(
            update={
                "phrase_id": position // 4,
                "section_id": "section-a" if position // 4 != 1 else "section-b",
                "boundary_confidence": 0.8 if position in {3, 7} else 0.0,
            }
        )
        for position, bar in enumerate(bars)
    ]

    plan = build_resolution_plan(raw, structured_bars)

    assert plan.base_resolution == 16
    assert plan.bar_resolutions == [16] * 4 + [24] * 4 + [16] * 4
    assert plan.change_points == [4, 8]
    assert plan.policy_version == "phrase-stable-v2"
    assert all(change_point % 4 == 0 for change_point in plan.change_points)


def test_build_resolution_plan_repeats_one_resolution_for_all_bars():
    raw = AudioAnalysisRaw(
        bpm=120.0,
        beat_times=[],
        onset_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        onset_strengths=[],
        duration=4.0,
        offset=0.0,
    )
    bars = build_bar_features(raw)

    plan = build_resolution_plan(raw, bars)

    assert len(plan.bar_resolutions) == len(bars) == 2
    assert plan.bar_resolutions == [plan.base_resolution, plan.base_resolution]
    assert plan.change_points == []


def _raw_for_bar_patterns(patterns: list[str]) -> AudioAnalysisRaw:
    bpm = 120.0
    bar_length = 2.0
    canonical_grids = 48
    onset_times: list[float] = []
    for bar_index, pattern in enumerate(patterns):
        step = 3 if pattern == "straight" else 4
        onset_times.extend(
            bar_index * bar_length + bar_length * tick / canonical_grids
            for tick in range(0, canonical_grids, step)
        )
    return AudioAnalysisRaw(
        bpm=bpm,
        beat_times=[],
        onset_times=onset_times,
        onset_strengths=[],
        duration=len(patterns) * bar_length,
        offset=0.0,
        time_signature="4/4",
    )
