from __future__ import annotations

from typing import Any

from tja_ai_chartgen.audio.analyze import (
    AudioAnalysisRaw,
    _apply_tempo_arbitration,
    _candidate_interval_statistics,
    _fixed_bpm_fit_metrics,
    _tempo_variation_diagnostic,
    apply_analysis_overrides,
    arbitrate_tempo_candidates,
)
from tja_ai_chartgen.tja.model import TempoMeterCandidate, TempoMeterEvidence

PHASE_FOUR_ACCEPTANCE_SCHEMA_VERSION = 1
PHASE_FOUR_ACCEPTANCE_VERSION = "phase-four-acceptance-v1"
PHASE_FOUR_FIXTURE_SET_VERSION = "phase-four-fixtures-v1"
FOCUSED_FIXTURES = (
    ("click_4_4.wav", "stable-4/4"),
    ("meter_3_4_120.wav", "meter-3/4"),
    ("meter_6_8_120.wav", "meter-6/8"),
    ("pickup_120.wav", "pickup"),
    ("tempo_ambiguity_120.wav", "half-double-ambiguity"),
    ("harmonic_sparse_120.wav", "weak-rhythm"),
)


def build_phase_four_behavior_matrix() -> dict[str, Any]:
    baseline = _candidate("librosa+onset-grid", onset_support=0.95)
    incomplete = _candidate("beatnet", onset_support=0.9).model_copy(
        update={
            "evidence": _candidate("beatnet").evidence.model_copy(
                update={"beat_number_completeness": 0.3}
            )
        }
    )
    rejected_candidates, rejected_decision = arbitrate_tempo_candidates(
        [baseline, incomplete],
        fallback_source=baseline.source,
    )
    rejected = next(item for item in rejected_candidates if item.source == "beatnet")

    weak_baseline = _candidate(
        "librosa",
        onset_support=0.35,
        onset_count=4,
    ).model_copy(update={"time_coverage": 0.4, "interval_stability": 0.5})
    strong_beatnet = _candidate(
        "beatnet",
        time_signature="3/4",
        onset_count=4,
        downbeat_support=0.8,
    )
    full_raw = _raw(weak_baseline)
    full = _apply_tempo_arbitration(
        full_raw,
        [weak_baseline, strong_beatnet],
        fallback_source=weak_baseline.source,
    )

    partial_beatnet = _candidate(
        "beatnet",
        time_signature="3/4",
        onset_support=0.7,
        downbeat_support=0.8,
    )
    partial = _apply_tempo_arbitration(
        _raw(baseline),
        [baseline, partial_beatnet],
        fallback_source=baseline.source,
    )

    alias_beatnet = _candidate("beatnet", bpm=240.0)
    _, alias_decision = arbitrate_tempo_candidates(
        [baseline, alias_beatnet],
        fallback_source=baseline.source,
    )
    overridden = apply_analysis_overrides(partial, bpm=128.0, offset=0.4)

    variation = {
        name: _variation(intervals).model_dump(mode="json")
        for name, intervals in {
            "stable": [0.5] * 16,
            "tempo_change": ([0.5] * 8) + ([0.65] * 8),
            "rubato": [0.46 + index * (0.08 / 17) for index in range(18)],
            "live_performance": [
                0.45,
                0.47,
                0.5,
                0.53,
                0.55,
                0.53,
                0.5,
                0.47,
                0.45,
            ]
            * 2,
        }.items()
    }
    scenarios = {
        "rejected_invalid_beatnet": {
            "passed": (
                rejected.accepted is False
                and rejected.reason == "incomplete-beat-numbers"
                and rejected_decision.selected_source == baseline.source
            ),
            "selected_source": rejected_decision.selected_source,
            "rejection": rejected.reason,
        },
        "selected_stronger_beatnet": {
            "passed": (
                full.tempo_analysis is not None
                and full.tempo_analysis.selected_source == "beatnet"
                and full.analyzer == "beatnet"
                and full.time_signature == "3/4"
            ),
            "selected_source": (
                full.tempo_analysis.selected_source if full.tempo_analysis else None
            ),
        },
        "partial_meter_downbeat_adoption": {
            "passed": (
                partial.tempo_analysis is not None
                and partial.tempo_analysis.partial_adoption
                and partial.bpm == baseline.bpm
                and partial.time_signature == "3/4"
                and partial.tempo_analysis.tempo_source == baseline.source
                and partial.tempo_analysis.meter_source == "beatnet"
            ),
            "analyzer": partial.analyzer,
            "bpm": partial.bpm,
            "time_signature": partial.time_signature,
        },
        "ambiguous_alias_fallback": {
            "passed": (
                alias_decision.ambiguous
                and alias_decision.selected_source == baseline.source
                and alias_decision.reason == "ambiguous-candidates"
            ),
            "selected_source": alias_decision.selected_source,
            "reason": alias_decision.reason,
        },
        "manual_override_precedence": {
            "passed": (
                overridden.bpm == 128.0
                and overridden.offset == 0.4
                and overridden.analyzer.endswith("+manual-override")
                and overridden.tempo_analysis == partial.tempo_analysis
            ),
            "analyzer": overridden.analyzer,
            "bpm": overridden.bpm,
            "offset": overridden.offset,
        },
        "tempo_variation_classification": {
            "passed": (
                variation["stable"]["classification"] == "stable"
                and variation["stable"]["suspected"] is False
                and variation["tempo_change"]["classification"]
                == "possible-tempo-change"
                and variation["rubato"]["classification"] == "possible-rubato"
                and variation["live_performance"]["classification"]
                == "possible-live-performance"
                and all(
                    variation[name]["fixed_bpm_constrained"]
                    for name in ("tempo_change", "rubato", "live_performance")
                )
            ),
            "classifications": {
                name: diagnostic["classification"]
                for name, diagnostic in variation.items()
            },
        },
    }
    return {
        "schema_version": 1,
        "matrix_version": "phase-four-behavior-matrix-v1",
        "passed": all(bool(item["passed"]) for item in scenarios.values()),
        "scenarios": scenarios,
    }


def build_phase_four_acceptance_report(
    fixture_runs: list[dict[str, Any]],
    behavior_runs: list[dict[str, Any]],
    default_runs: list[dict[str, Any]],
    *,
    network_blocked: bool,
) -> dict[str, Any]:
    fixture = fixture_runs[0] if fixture_runs else {}
    behavior = behavior_runs[0] if behavior_runs else {}
    default = default_runs[0] if default_runs else {}
    summary = fixture.get("summary", {})
    fixtures = fixture.get("fixtures", [])
    categories = set(fixture.get("categories", []))
    required_categories = {item[1] for item in FOCUSED_FIXTURES}
    decisions = [item.get("decision", {}) for item in fixtures]

    checks = {
        "adaptive_arbitration": {
            "passed": bool(behavior.get("passed")),
            "scenario_count": len(behavior.get("scenarios", {})),
            "scenarios": behavior.get("scenarios", {}),
        },
        "stable_four_four_non_regression": {
            "passed": (
                _int(summary.get("bpm_regression_count")) == 0
                and _int(summary.get("beat_f1_regression_count")) == 0
                and _int(summary.get("downbeat_f1_regression_count")) == 0
                and bool(summary.get("stable_four_four_preserved"))
            ),
            "bpm_regression_count": summary.get("bpm_regression_count"),
            "beat_f1_regression_count": summary.get("beat_f1_regression_count"),
            "downbeat_f1_regression_count": summary.get(
                "downbeat_f1_regression_count"
            ),
            "stable_four_four_preserved": summary.get(
                "stable_four_four_preserved"
            ),
        },
        "meter_pickup_alias_coverage": {
            "passed": (
                required_categories.issubset(categories)
                and _int(summary.get("half_time_error_count")) == 0
                and _int(summary.get("double_time_error_count")) == 0
                and _int(summary.get("pickup_regression_count")) == 0
            ),
            "required_categories": sorted(required_categories),
            "covered_categories": sorted(categories),
            "half_time_error_count": summary.get("half_time_error_count"),
            "double_time_error_count": summary.get("double_time_error_count"),
            "pickup_regression_count": summary.get("pickup_regression_count"),
        },
        "beatnet_rejection_or_contribution": {
            "passed": (
                _int(summary.get("beatnet_complete_count")) == len(fixtures)
                and _int(summary.get("rejected_fixture_count")) > 0
                and bool(
                    behavior.get("scenarios", {})
                    .get("partial_meter_downbeat_adoption", {})
                    .get("passed")
                )
            ),
            "beatnet_complete_count": summary.get("beatnet_complete_count"),
            "rejected_fixture_count": summary.get("rejected_fixture_count"),
            "full_selection_count": summary.get("full_selection_count"),
            "partial_adoption_count": summary.get("partial_adoption_count"),
            "synthetic_partial_adoption": behavior.get("scenarios", {}).get(
                "partial_meter_downbeat_adoption", {}
            ),
        },
        "analysis_explainability": {
            "passed": bool(fixtures)
            and all(
                decision.get("decision_version") == "tempo-arbitration-v3"
                and decision.get("selected_source")
                and decision.get("selected_candidate_count") == 1
                and decision.get("tempo_variation_version")
                == "tempo-variation-v1"
                and decision.get("tempo_variation_classification")
                in {
                    "stable",
                    "possible-tempo-change",
                    "possible-rubato",
                    "possible-live-performance",
                    "insufficient-evidence",
                }
                for decision in decisions
            ),
            "explained_fixture_count": sum(
                bool(decision.get("selected_source")) for decision in decisions
            ),
            "fixture_count": len(fixtures),
        },
        "default_path_without_beatnet": {
            "passed": (
                bool(default.get("passed"))
                and default.get("beatnet_analysis_status") == "unavailable"
                and not any(
                    str(source).startswith("beatnet")
                    for source in default.get("candidate_sources", [])
                )
            ),
            **default,
        },
        "fixed_bpm_limitation_diagnostic": {
            "passed": bool(
                behavior.get("scenarios", {})
                .get("tempo_variation_classification", {})
                .get("passed")
            )
            and all(
                decision.get("tempo_variation_classification") == "stable"
                and decision.get("fixed_bpm_constrained") is False
                for decision in decisions
            ),
            "fixture_classifications": {
                item.get("audio", "unknown"): item.get("decision", {}).get(
                    "tempo_variation_classification"
                )
                for item in fixtures
            },
        },
        "deterministic_offline_acceptance": {
            "passed": (
                network_blocked
                and _runs_stable(fixture_runs)
                and _runs_stable(behavior_runs)
                and _runs_stable(default_runs)
                and len(fixture_runs) == 2
                and len(behavior_runs) == 2
                and len(default_runs) == 2
            ),
            "network_blocked": network_blocked,
            "fixture_runs_stable": _runs_stable(fixture_runs),
            "behavior_runs_stable": _runs_stable(behavior_runs),
            "default_runs_stable": _runs_stable(default_runs),
            "run_count": min(
                len(fixture_runs), len(behavior_runs), len(default_runs)
            ),
        },
    }
    return {
        "schema_version": PHASE_FOUR_ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_version": PHASE_FOUR_ACCEPTANCE_VERSION,
        "phase": "Phase 4",
        "passed": bool(fixture)
        and bool(behavior)
        and bool(default)
        and all(bool(check["passed"]) for check in checks.values()),
        "fixture_set_version": PHASE_FOUR_FIXTURE_SET_VERSION,
        "fixture_count": int(fixture.get("fixture_count", 0)),
        "checks": checks,
        "fixture_evidence": fixture,
        "behavior_matrix": behavior,
        "default_path_evidence": default,
    }


def render_phase_four_acceptance_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    stable = checks["stable_four_four_non_regression"]
    coverage = checks["meter_pickup_alias_coverage"]
    contribution = checks["beatnet_rejection_or_contribution"]
    lines = [
        "# Phase 4 acceptance report",
        "",
        f"- Acceptance: `{report['acceptance_version']}`",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Focused fixture count: {report['fixture_count']}",
        f"- Fixture set: `{report['fixture_set_version']}`",
        "",
        "## Exit conditions",
        "",
        "| Exit condition | Result | Evidence |",
        "| --- | :---: | --- |",
        _check_row(
            "Adaptive arbitration rejects weak candidates, selects strong candidates, and supports partial adoption",
            checks["adaptive_arbitration"]["passed"],
            f"{checks['adaptive_arbitration']['scenario_count']} deterministic scenarios",
        ),
        _check_row(
            "Stable 4/4 and focused beat/downbeat metrics do not regress",
            stable["passed"],
            f"BPM {stable['bpm_regression_count']}, beat {stable['beat_f1_regression_count']}, "
            f"downbeat {stable['downbeat_f1_regression_count']} regressions",
        ),
        _check_row(
            "3/4, 4/4, 6/8, pickup, weak-rhythm, and half/double ambiguity fixtures are covered",
            coverage["passed"],
            f"{', '.join(coverage['covered_categories'])}; "
            f"half {coverage['half_time_error_count']}, double {coverage['double_time_error_count']}",
        ),
        _check_row(
            "BeatNet can be rejected or contribute only meter/downbeat",
            contribution["passed"],
            f"{contribution['rejected_fixture_count']} rejected fixtures; "
            f"{contribution['partial_adoption_count']} fixture partial adoptions; "
            "synthetic partial-adoption contract covered",
        ),
        _check_row(
            "analysis.json diagnostics explain the final tempo and meter decision",
            checks["analysis_explainability"]["passed"],
            f"{checks['analysis_explainability']['explained_fixture_count']}/"
            f"{checks['analysis_explainability']['fixture_count']} fixtures explained",
        ),
        _check_row(
            "Default analysis remains valid without loading BeatNet",
            checks["default_path_without_beatnet"]["passed"],
            f"status {checks['default_path_without_beatnet'].get('beatnet_analysis_status')}",
        ),
        _check_row(
            "Fixed-BPM limitation and tempo-variation classifications remain report-only",
            checks["fixed_bpm_limitation_diagnostic"]["passed"],
            "stable fixtures plus synthetic tempo-change/rubato/live-performance cases",
        ),
        _check_row(
            "Acceptance is deterministic and offline",
            checks["deterministic_offline_acceptance"]["passed"],
            f"{checks['deterministic_offline_acceptance']['run_count']} blocked-network runs",
        ),
        "",
        "## Focused fixture outcomes",
        "",
        "| Audio | Category | Baseline analyzer | Enhanced analyzer | Decision | Meter | Beat F1 | Downbeat F1 |",
        "| --- | --- | --- | --- | --- | :---: | ---: | ---: |",
    ]
    for item in report["fixture_evidence"].get("fixtures", []):
        decision = item["decision"]
        lines.append(
            f"| `{item['audio']}` | {item['category']} | `{item['baseline']['analyzer']}` | "
            f"`{item['enhanced']['analyzer']}` | `{decision['reason']}` | "
            f"{'✓' if item['enhanced']['meter_correct'] else '✗'} | "
            f"{item['enhanced']['beat_f1']:.6f} | {item['enhanced']['downbeat_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Phase 4 acceptance keeps BeatNet opt-in. The focused synthetic fixtures verify conservative arbitration and explainability; enabling BeatNet by default still requires broader real-music evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate(
    source: str,
    *,
    bpm: float = 120.0,
    time_signature: str = "4/4",
    onset_support: float = 0.9,
    onset_count: int = 16,
    downbeat_support: float = 0.5,
) -> TempoMeterCandidate:
    interval = 60.0 / bpm
    beat_times = [0.2 + index * interval for index in range(17)]
    meter_size = {"3/4": 3, "4/4": 4, "6/8": 6}[time_signature]
    downbeat_times = [
        beat_times[index]
        for index in range(
            0,
            len(beat_times),
            3 if time_signature == "6/8" else meter_size,
        )
    ]
    return TempoMeterCandidate(
        source=source,
        bpm=bpm,
        offset=0.2,
        time_signature=time_signature,
        beat_times=beat_times,
        downbeat_times=downbeat_times if source.startswith("beatnet") else [],
        onset_support=onset_support,
        time_coverage=1.0,
        interval_stability=1.0,
        evidence=TempoMeterEvidence(
            onset_count=onset_count,
            onset_support_margin=0.1,
            interval_count=len(beat_times) - 1,
            beat_number_completeness=1.0 if source.startswith("beatnet") else 0.0,
            meter_stability=1.0 if source.startswith("beatnet") else 0.0,
            downbeat_support=downbeat_support if source.startswith("beatnet") else 0.0,
            meter_length_score=1.0 if source.startswith("beatnet") else 0.0,
        ),
        accepted=True,
        reason="candidate",
    )


def _raw(candidate: TempoMeterCandidate) -> AudioAnalysisRaw:
    return AudioAnalysisRaw(
        bpm=candidate.bpm,
        beat_times=candidate.beat_times,
        onset_times=[],
        onset_strengths=[],
        duration=8.0,
        offset=candidate.offset,
        time_signature=candidate.time_signature,
        analyzer=candidate.source,
        tempo_candidates=[candidate],
    )


def _variation(intervals: list[float]) -> Any:
    beat_times = [0.2]
    for interval in intervals:
        beat_times.append(beat_times[-1] + interval)
    interval_count, mean_interval, interval_cv = _candidate_interval_statistics(
        beat_times
    )
    fit_metrics = _fixed_bpm_fit_metrics(
        beat_times,
        fixed_interval_seconds=0.5,
    )
    candidate = TempoMeterCandidate(
        source="beatnet",
        bpm=120.0,
        offset=0.2,
        time_signature="4/4",
        beat_times=beat_times,
        time_coverage=1.0,
        interval_stability=max(0.0, 1.0 - (interval_cv or 0.0)),
        evidence=TempoMeterEvidence(
            interval_count=interval_count,
            mean_interval_seconds=mean_interval,
            interval_coefficient_of_variation=interval_cv,
            **fit_metrics,
        ),
        accepted=True,
    )
    return _tempo_variation_diagnostic([candidate], selected_source="beatnet")


def _runs_stable(runs: list[dict[str, Any]]) -> bool:
    return bool(runs) and all(run == runs[0] for run in runs[1:])


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _check_row(label: str, passed: bool, evidence: str) -> str:
    return f"| {label} | {'PASS' if passed else 'FAIL'} | {evidence} |"
