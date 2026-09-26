<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-26` using selection `range` (2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 85.85 s; process peak working set: 161.0 MiB; enforced budget: 1200 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | 100.0% (30/30) | 100.0% (30/30) | 3.2000 [3.0000, 3.4000] | 3.2000 [3.0000, 3.4000] | 76.7% | 96.7% | +0.0000 [+0.0000, +0.0000] | 0/30/0 | 7.5789 | 0.9990 | 315.74 ms | n/a/n/a |
| `finite_fixed_work` | 100.0% (30/30) | 100.0% (30/30) | 3.5000 [3.2667, 3.7333] | 3.5000 [3.2667, 3.7333] | 60.0% | 90.0% | +0.3000 [+0.1667, +0.4667] | 2/18/10 | 7.5789 | 0.9990 | 161.12 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 48/8/0/40/0 | 0/96 | 0/0 |
| `finite_fixed_work` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/105 | 0/105 | 0/0 |

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
| `finite_fixed_work` | all | 1 | 30/30 | 0.0011 | 7.5789 | 0.9990 |
| `finite_fixed_work` | all | 2 | 30/30 | 0.1298 | 3.3265 | 0.8794 |
| `finite_fixed_work` | all | 3 | 29/29 | 0.6241 | 1.0179 | 0.3941 |
| `finite_fixed_work` | all | 4 | 12/12 | 0.8400 | 0.3023 | 0.1656 |
| `finite_fixed_work` | all | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_fixed_work` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fixed_work` | never_used | 1 | 25/25 | 0.0011 | 7.7276 | 0.9990 |
| `finite_fixed_work` | never_used | 2 | 25/25 | 0.1367 | 3.5057 | 0.8719 |
| `finite_fixed_work` | never_used | 3 | 24/24 | 0.6077 | 1.1199 | 0.4113 |
| `finite_fixed_work` | never_used | 4 | 10/10 | 0.8084 | 0.3623 | 0.1987 |
| `finite_fixed_work` | never_used | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_fixed_work` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fixed_work` | reused | 1 | 5/5 | 0.0011 | 6.8352 | 0.9990 |
| `finite_fixed_work` | reused | 2 | 5/5 | 0.0949 | 2.4310 | 0.9165 |
| `finite_fixed_work` | reused | 3 | 5/5 | 0.7024 | 0.5283 | 0.3119 |
| `finite_fixed_work` | reused | 4 | 2/2 | 0.9978 | 0.0022 | 0.0000 |
| `finite_fixed_work` | out_of_core | 1 | 5/5 | 0.0000 | 12.1442 | 1.0011 |
| `finite_fixed_work` | out_of_core | 2 | 5/5 | 0.0005 | 8.0167 | 1.1165 |
| `finite_fixed_work` | out_of_core | 3 | 5/5 | 0.0536 | 3.7314 | 1.0907 |
| `finite_fixed_work` | out_of_core | 4 | 5/5 | 0.6219 | 0.7194 | 0.3973 |
| `finite_fixed_work` | out_of_core | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_fixed_work` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `staged_fixed_belief` all-game mean sensitivity: penalty 6 = 3.2000 [3.0000, 3.4000]; penalty 7 = 3.2000 [3.0000, 3.4000]; penalty 8 = 3.2000 [3.0000, 3.4000].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
