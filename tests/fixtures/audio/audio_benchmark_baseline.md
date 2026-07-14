# Synthetic audio fixture benchmark

- Benchmark: `audio-alignment-v2`
- Fixture count: 17
- BeatNet: `false`
- Tolerances: onset 0.050s, beat 0.070s, downbeat 0.070s
- Band onset threshold: 0.250

## Aggregate metrics

| Metric | Precision | Recall | F1 | Mean error |
| --- | ---: | ---: | ---: | ---: |
| Onset | 0.997135 | 0.995708 | 0.996421 | 0.016850s |
| Strong onset | 0.186246 | 1.000000 | 0.314010 | 0.015434s |
| Beat | 0.783262 | 0.829545 | 0.805740 | 0.012736s |
| Downbeat | 0.392308 | 0.447368 | 0.418033 | 0.012861s |
| Low-band onset | 0.500000 | 1.000000 | 0.666667 | 0.017029s |
| High-band onset | 0.500000 | 1.000000 | 0.666667 | 0.020998s |
| Pickup onset | 0.666667 | 1.000000 | 0.800000 | 0.013549s |
| Fill onset | 0.888889 | 1.000000 | 0.941176 | 0.017489s |

## Tempo, meter, and resolution

| Metric | Value |
| --- | ---: |
| Mean BPM absolute error | 8.008412 |
| Mean BPM relative error | 0.065990 |
| Meter accuracy | 0.882353 |
| Mean first-downbeat error | 0.308080s |
| Resolution expressible-event ratio | 1.000000 |
| Mean resolution quantization error | 0.000000s |
| Under-resolved bars | 0 |
| Unnecessary high-resolution bars | 114 |
| Resolution changes | 0 |
| Silent-range onset false positives | 2 |

## Per-fixture metrics

| Audio | BPM error | Meter | Onset F1 | Beat F1 | Downbeat F1 | Band F1 | Pickup recall | Fill recall | Base res. | Expressible | Quant. error |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `band_attacks_120.wav` | 0.000000 | ✓ | 1.000000 | 0.969697 | 0.941176 | 0.666667 | — | — | 48 | 1.000000 | 0.000000s |
| `click_4_4.wav` | 0.000000 | ✓ | 1.000000 | 0.969697 | 0.888889 | — | — | — | 48 | 1.000000 | 0.000000s |
| `click_4_4_leadin.wav` | 0.000000 | ✓ | 1.000000 | 0.914286 | 0.000000 | — | — | — | 48 | 1.000000 | 0.000000s |
| `dense_180.wav` | 4.570000 | ✓ | 0.979021 | 0.584615 | 0.352941 | — | — | — | 48 | 1.000000 | 0.000000s |
| `fill_burst_120.wav` | 0.000000 | ✓ | 1.000000 | 0.969697 | 0.941176 | — | — | 1.000000 | 48 | 1.000000 | 0.000000s |
| `harmonic_sparse_120.wav` | 0.185000 | ✓ | 1.000000 | 0.638298 | 0.000000 | — | — | — | 48 | 1.000000 | 0.000000s |
| `meter_3_4_120.wav` | 0.000000 | ✗ | 1.000000 | 0.960000 | 0.266667 | — | — | — | 48 | 1.000000 | 0.000000s |
| `meter_6_8_120.wav` | 0.000000 | ✗ | 1.000000 | 0.960000 | 0.266667 | — | — | — | 48 | 1.000000 | 0.000000s |
| `mixed_120.wav` | 2.546000 | ✓ | 1.000000 | 1.000000 | 0.444444 | — | — | — | 48 | 1.000000 | 0.000000s |
| `pickup_120.wav` | 0.000000 | ✓ | 1.000000 | 0.955224 | 0.000000 | — | 1.000000 | — | 48 | 1.000000 | 0.000000s |
| `sparse_120.wav` | 0.000000 | ✓ | 1.000000 | 0.969697 | 0.941176 | — | — | — | 48 | 1.000000 | 0.000000s |
| `straight_120.wav` | 24.297000 | ✓ | 1.000000 | 0.142857 | 0.000000 | — | — | — | 48 | 1.000000 | 0.000000s |
| `structure_build_up_120.wav` | 2.546000 | ✓ | 1.000000 | 0.989474 | 0.166667 | — | — | — | 48 | 1.000000 | 0.000000s |
| `syncopated_120.wav` | 41.499000 | ✓ | 1.000000 | 0.270270 | 0.200000 | — | — | — | 48 | 1.000000 | 0.000000s |
| `tempo_ambiguity_120.wav` | 0.000000 | ✓ | 1.000000 | 0.969697 | 0.941176 | — | — | — | 48 | 1.000000 | 0.000000s |
| `transient_noise_intro_120.wav` | 0.500000 | ✓ | 0.888889 | 0.727273 | 0.666667 | — | — | — | 48 | 1.000000 | 0.000000s |
| `triplet_120.wav` | 60.000000 | ✓ | 1.000000 | 0.363636 | 0.000000 | — | — | — | 48 | 1.000000 | 0.000000s |

This report is a deterministic regression baseline. Metric values may vary across librosa, BeatNet, and platform versions; CI validates structure and ranges rather than requiring cross-environment byte equality.
