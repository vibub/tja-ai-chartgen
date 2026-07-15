# Phase 1 acceptance report

- Acceptance: `phase-one-acceptance-v1`
- Result: **PASS**
- Fixture count: 17

## Exit conditions

| Exit condition | Result | Evidence |
| --- | :---: | --- |
| All salience points use canonical grids | PASS | 0 invalid, 0 duplicate, 0 unordered |
| Repeated runs are deterministic | PASS | 2 offline runs |
| Edge silence produces no salience points | PASS | 10 edge-silent bars, 0 violations |
| Known fixture onsets align with hit-salience peaks | PASS | precision 0.999, recall 0.989, max error 0.052s |
| Confidence and fallback reasons are internally consistent | PASS | 0 violations |
| Persisted acceptance evidence remains sparse and AI payload is unchanged | PASS | 1028/6240 points (0.165), 0 AI payload fields added |
| Acceptance runs without network access | PASS | socket connect/connect_ex blocked during both passes |

## Handoff boundary

Phase 1 validates the shared hit/accent/don-ka salience pipeline itself. Integration into the fallback generator, AI prompt, and quality report remains explicitly deferred to later roadmap phases.

## Fixture summary

| Fixture | Bars | Points | Peaks | Onset precision | Onset recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `band_attacks_120.wav` | 9 | 83 | 64 | 1.000 | 1.000 |
| `click_4_4.wav` | 5 | 27 | 16 | 1.000 | 1.000 |
| `click_4_4_leadin.wav` | 5 | 31 | 16 | 1.000 | 1.000 |
| `dense_180.wav` | 9 | 98 | 70 | 1.000 | 0.959 |
| `fill_burst_120.wav` | 9 | 60 | 38 | 1.000 | 1.000 |
| `harmonic_sparse_120.wav` | 8 | 43 | 15 | 1.000 | 0.938 |
| `meter_3_4_120.wav` | 7 | 38 | 24 | 1.000 | 1.000 |
| `meter_6_8_120.wav` | 7 | 61 | 48 | 1.000 | 1.000 |
| `mixed_120.wav` | 5 | 69 | 56 | 1.000 | 1.000 |
| `pickup_120.wav` | 9 | 53 | 34 | 1.000 | 1.000 |
| `sparse_120.wav` | 9 | 43 | 26 | 1.000 | 1.000 |
| `straight_120.wav` | 4 | 62 | 62 | 1.000 | 0.969 |
| `structure_build_up_120.wav` | 12 | 138 | 96 | 1.000 | 1.000 |
| `syncopated_120.wav` | 12 | 79 | 40 | 1.000 | 1.000 |
| `tempo_ambiguity_120.wav` | 9 | 54 | 30 | 1.000 | 0.938 |
| `transient_noise_intro_120.wav` | 4 | 16 | 9 | 0.889 | 1.000 |
| `triplet_120.wav` | 7 | 73 | 48 | 1.000 | 1.000 |
