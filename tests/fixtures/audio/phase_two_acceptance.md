# Phase 2 acceptance report

- Acceptance: `phase-two-acceptance-v1`
- Result: **PASS**
- Fixture count: 17
- Chart count: 68

## Exit conditions

| Exit condition | Result | Evidence |
| --- | :---: | --- |
| Synthetic note/onset alignment improves over the Phase 2 baseline | PASS | 0.626563 → 0.638240 (+0.011677) |
| Unsupported-note ratio decreases | PASS | 0.059857 → 0.045524 (-0.014333) |
| Strong-onset and downbeat response do not regress | PASS | strong 0.923077 → 0.973077; downbeat 0.934211 → 0.982456 |
| Easy/Normal/Hard/Oni load remains monotonic | PASS | {'Easy': 360, 'Normal': 605, 'Hard': 1052, 'Oni': 1278} |
| Silent/rest/sparse behavior does not regress | PASS | silent violations 65 → 60; matrix violations 0 |
| 16/24/48 and 12/18/36 preserve equivalent load and hard constraints | PASS | 360 configurations, 1080 bar results |
| Rule and AI results use the same salience-alignment metric | PASS | ai-salience-validation-v1, report-only |
| Acceptance is deterministic and offline | PASS | 2 blocked-network runs, deterministic rate 1.000000 |

## Aggregate comparison

| Metric | Baseline | Current |
| --- | ---: | ---: |
| Note/onset precision | 0.626563 | 0.638240 |
| Unsupported-note ratio | 0.059857 | 0.045524 |
| Strong-onset recall | 0.923077 | 0.973077 |
| Downbeat recall | 0.934211 | 0.982456 |

## Constraint matrix

- Load configurations: 180
- Silence/rest/sparse configurations: 180
- Total bar results: 1080
- Violations: 0

Phase 2 acceptance keeps AI salience diagnostics report-only. Repair thresholds remain deferred until the metrics are compared with listening results.
