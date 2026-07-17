# Phase 7 acceptance report

- Acceptance: `phase-seven-acceptance-v1`
- Result: **PASS**
- Deterministic scenarios: 3

## Exit conditions

| Exit condition | Result | Evidence |
| --- | :---: | --- |
| The complete rhythm-first path runs without heavyweight models | PASS | audio fixture `click_4_4.wav`; 4 bars; new heavy modules=[] |
| htdemucs adds only bounded stem activity/onset evidence | PASS | drum agreement strengthens an existing canonical transient |
| Missing concrete taxonomy does not reduce core generation | PASS | structure, fallback, compact payload, and core QualityReport are unchanged |
| The AI instrument projection is smaller than the legacy instrument-v1 route | PASS | 9→7 columns; 73→34 bytes |
| Legacy analysis, config, and AI sidecars remain readable | PASS | analysis v5, config v0, compact-v4 input, legacy output/attempts |
| QualityReport prioritizes rhythm alignment and keeps instrument metrics diagnostic-only | PASS | `quality-report-rhythm-alignment-v1` |
| CLI/Web defaults, progress, summaries, notices, and remote permissions match capabilities | PASS | stem-role recommended; full explicitly labeled legacy diagnostics |
| Phase 7 acceptance is deterministic and offline | PASS | 2 blocked-network runs |

## Consumer convergence matrix

| Scenario | Result |
| --- | :---: |
| `no_heavy_model_rhythm_path` | PASS |
| `stem_role_adds_bounded_activity_onset_evidence` | PASS |
| `concrete_taxonomy_does_not_change_core_consumers` | PASS |

Phase 7 makes stem-role salience the recommended optional enhancement while preserving the legacy instrument-v1 read path. Concrete instrument taxonomy remains compatibility diagnostics only and is excluded from structure decisions, fallback output, compact AI input, and the QualityReport mainline.
