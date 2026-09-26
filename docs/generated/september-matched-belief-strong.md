<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-26` using selection `range` (2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 223.94 s; process peak working set: 138.3 MiB; enforced budget: 600 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | 100.0% (30/30) | 100.0% (30/30) | 3.2000 [3.0000, 3.4000] | 3.2000 [3.0000, 3.4000] | 76.7% | 96.7% | +0.0000 [+0.0000, +0.0000] | 0/30/0 | 7.5789 | 0.9990 | 334.71 ms | n/a/n/a |
| `finite_strong` | 100.0% (30/30) | 100.0% (30/30) | 3.5667 [3.3000, 3.8000] | 3.5667 [3.3000, 3.8000] | 50.0% | 93.3% | +0.3667 [+0.1667, +0.5667] | 4/13/13 | 7.5789 | 0.9990 | 2016.33 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 48/8/0/40/0 | 0/96 | 0/0 |
| `finite_strong` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/107 | 0/107 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | all | 1 | 30/30 | 0.0011 | 7.5789 | 0.9990 |
| `staged_fixed_belief` | all | 2 | 30/30 | 0.1475 | 3.4767 | 0.8588 |
| `staged_fixed_belief` | all | 3 | 28/28 | 0.7529 | 0.6331 | 0.2387 |
| `staged_fixed_belief` | all | 4 | 7/7 | 0.8489 | 0.8053 | 0.2705 |
| `staged_fixed_belief` | all | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `staged_fixed_belief` | never_used | 1 | 25/25 | 0.0011 | 7.7276 | 0.9990 |
| `staged_fixed_belief` | never_used | 2 | 25/25 | 0.1545 | 3.5558 | 0.8537 |
| `staged_fixed_belief` | never_used | 3 | 23/23 | 0.7733 | 0.6250 | 0.2214 |
| `staged_fixed_belief` | never_used | 4 | 5/5 | 0.7980 | 1.1177 | 0.3783 |
| `staged_fixed_belief` | never_used | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `staged_fixed_belief` | reused | 1 | 5/5 | 0.0011 | 6.8352 | 0.9990 |
| `staged_fixed_belief` | reused | 2 | 5/5 | 0.1127 | 3.0809 | 0.8839 |
| `staged_fixed_belief` | reused | 3 | 5/5 | 0.6591 | 0.6705 | 0.3184 |
| `staged_fixed_belief` | reused | 4 | 2/2 | 0.9762 | 0.0243 | 0.0011 |
| `staged_fixed_belief` | out_of_core | 1 | 5/5 | 0.0000 | 12.1442 | 1.0011 |
| `staged_fixed_belief` | out_of_core | 2 | 5/5 | 0.0007 | 8.3919 | 1.1408 |
| `staged_fixed_belief` | out_of_core | 3 | 5/5 | 0.4124 | 2.1759 | 0.6592 |
| `staged_fixed_belief` | out_of_core | 4 | 4/4 | 0.7509 | 1.3937 | 0.4728 |
| `staged_fixed_belief` | out_of_core | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_strong` | all | 1 | 30/30 | 0.0011 | 7.5789 | 0.9990 |
| `finite_strong` | all | 2 | 30/30 | 0.0704 | 4.4241 | 0.9280 |
| `finite_strong` | all | 3 | 29/29 | 0.5024 | 1.2554 | 0.5169 |
| `finite_strong` | all | 4 | 15/15 | 0.8726 | 0.2780 | 0.1167 |
| `finite_strong` | all | 5 | 2/2 | 0.7500 | 0.3466 | 0.2500 |
| `finite_strong` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_strong` | never_used | 1 | 25/25 | 0.0011 | 7.7276 | 0.9990 |
| `finite_strong` | never_used | 2 | 25/25 | 0.0773 | 4.5818 | 0.9205 |
| `finite_strong` | never_used | 3 | 24/24 | 0.4923 | 1.3316 | 0.5327 |
| `finite_strong` | never_used | 4 | 13/13 | 0.8531 | 0.3208 | 0.1346 |
| `finite_strong` | never_used | 5 | 2/2 | 0.7500 | 0.3466 | 0.2500 |
| `finite_strong` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_strong` | reused | 1 | 5/5 | 0.0011 | 6.8352 | 0.9990 |
| `finite_strong` | reused | 2 | 5/5 | 0.0358 | 3.6356 | 0.9659 |
| `finite_strong` | reused | 3 | 5/5 | 0.5508 | 0.8897 | 0.4409 |
| `finite_strong` | reused | 4 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `finite_strong` | out_of_core | 1 | 5/5 | 0.0000 | 12.1442 | 1.0011 |
| `finite_strong` | out_of_core | 2 | 5/5 | 0.0001 | 9.3655 | 1.0285 |
| `finite_strong` | out_of_core | 3 | 5/5 | 0.2512 | 3.2609 | 0.9689 |
| `finite_strong` | out_of_core | 4 | 4/4 | 0.5996 | 0.9578 | 0.4144 |
| `finite_strong` | out_of_core | 5 | 2/2 | 0.7500 | 0.3466 | 0.2500 |
| `finite_strong` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `staged_fixed_belief` all-game mean sensitivity: penalty 6 = 3.2000 [3.0000, 3.4000]; penalty 7 = 3.2000 [3.0000, 3.4000]; penalty 8 = 3.2000 [3.0000, 3.4000].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
