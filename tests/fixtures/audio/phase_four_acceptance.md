# Phase 4 acceptance report

- Acceptance: `phase-four-acceptance-v1`
- Result: **PASS**
- Focused fixture count: 6
- Fixture set: `phase-four-fixtures-v1`

## Exit conditions

| Exit condition | Result | Evidence |
| --- | :---: | --- |
| Adaptive arbitration rejects weak candidates, selects strong candidates, and supports partial adoption | PASS | 6 deterministic scenarios |
| Stable 4/4 and focused beat/downbeat metrics do not regress | PASS | BPM 0, beat 0, downbeat 0 regressions |
| 3/4, 4/4, 6/8, pickup, weak-rhythm, and half/double ambiguity fixtures are covered | PASS | half-double-ambiguity, meter-3/4, meter-6/8, pickup, stable-4/4, weak-rhythm; half 0, double 0 |
| BeatNet can be rejected or contribute only meter/downbeat | PASS | 6 rejected fixtures; 0 fixture partial adoptions; synthetic partial-adoption contract covered |
| analysis.json diagnostics explain the final tempo and meter decision | PASS | 6/6 fixtures explained |
| Default analysis remains valid without loading BeatNet | PASS | status unavailable |
| Fixed-BPM limitation and tempo-variation classifications remain report-only | PASS | stable fixtures plus synthetic tempo-change/rubato/live-performance cases |
| Acceptance is deterministic and offline | PASS | 2 blocked-network runs |

## Focused fixture outcomes

| Audio | Category | Baseline analyzer | Enhanced analyzer | Decision | Meter | Beat F1 | Downbeat F1 |
| --- | --- | --- | --- | --- | :---: | ---: | ---: |
| `click_4_4.wav` | stable-4/4 | `librosa+onset-grid` | `librosa+onset-grid` | `incomplete-beat-numbers` | ✓ | 0.969697 | 0.888889 |
| `meter_3_4_120.wav` | meter-3/4 | `librosa+onset-grid` | `librosa+onset-grid` | `incomplete-beat-numbers` | ✗ | 0.960000 | 0.266667 |
| `meter_6_8_120.wav` | meter-6/8 | `librosa+onset-grid` | `librosa+onset-grid` | `incomplete-beat-numbers` | ✗ | 0.960000 | 0.266667 |
| `pickup_120.wav` | pickup | `librosa+onset-grid` | `librosa+onset-grid` | `incomplete-beat-numbers` | ✓ | 0.955224 | 0.000000 |
| `tempo_ambiguity_120.wav` | half-double-ambiguity | `librosa+onset-grid` | `librosa+onset-grid` | `incomplete-beat-numbers` | ✓ | 0.969697 | 0.941176 |
| `harmonic_sparse_120.wav` | weak-rhythm | `librosa` | `librosa` | `ambiguous-candidates` | ✓ | 0.638298 | 0.000000 |

Phase 4 acceptance keeps BeatNet opt-in. The focused synthetic fixtures verify conservative arbitration and explainability; enabling BeatNet by default still requires broader real-music evidence.
