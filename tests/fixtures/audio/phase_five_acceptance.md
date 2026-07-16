# Phase 5 acceptance report

- Acceptance: `phase-five-acceptance-v1`
- Result: **PASS**
- Deterministic scenarios: 7

## Exit conditions

| Exit condition | Result | Evidence |
| --- | :---: | --- |
| Users can prepare htdemucs-only stem-role models without AST | PASS | stem AST 0 bytes; full AST 7 bytes |
| Stem-role loading is offline and performance costs remain report-only | PASS | network blocked=True; benchmark `instrument-model-benchmark-v1` |
| Model fallback preserves default rhythmic salience | PASS | baseline and fallback salience are identical |
| Drum, bass, vocal, and accompaniment roles improve their bounded salience targets | PASS | drums, bass, vocals, accompaniment |
| Stem artifacts and non-drum onsets cannot create unconditional hits | PASS | 0 generated points |
| AI compact payload no longer depends on concrete instrument taxonomy | PASS | schema `tja-ai-chartgen-compact-v7` |
| Remote Web instrument analysis still requires explicit administrator permission | PASS | local allowed; remote default denied; remote explicit allowed |
| Phase 5 acceptance is deterministic and offline | PASS | 2 blocked-network runs |

## Stem-role behavior matrix

| Scenario | Result |
| --- | :---: |
| `fallback_preserves_default_salience` | PASS |
| `drum_agreement_strengthens_transient` | PASS |
| `bass_reinforces_beat_and_don_only` | PASS |
| `vocal_supports_phrase_context` | PASS |
| `accompaniment_supports_highlight` | PASS |
| `stem_artifacts_do_not_create_hits` | PASS |
| `drum_onset_rise_supports_burst` | PASS |

Phase 5 keeps htdemucs opt-in. Performance measurements remain environment-specific report-only diagnostics; no cross-machine latency or memory threshold is enforced.
