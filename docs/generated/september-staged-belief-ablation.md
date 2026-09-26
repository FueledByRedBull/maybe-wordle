<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-26` using selection `range` (2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 165.15 s; process peak working set: 178.8 MiB; enforced budget: 1200 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_dynamic_belief` | 100.0% (30/30) | 100.0% (30/30) | 3.3333 [3.1000, 3.6000] | 3.3333 [3.1000, 3.6000] | 73.3% | 93.3% | +0.0000 [+0.0000, +0.0000] | 0/30/0 | 6.5971 | 0.9986 | 21.59 ms | n/a/n/a |
| `staged_fixed_belief` | 100.0% (30/30) | 100.0% (30/30) | 3.2000 [3.0000, 3.4000] | 3.2000 [3.0000, 3.4000] | 76.7% | 96.7% | -0.1333 [-0.4000, +0.1333] | 8/16/6 | 7.5789 | 0.9990 | 324.47 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_dynamic_belief` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 34/12/0/54/0 | 5/33 | 0/0 |
| `staged_fixed_belief` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 48/8/0/40/0 | 0/96 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `staged_dynamic_belief` | all | 1 | 25/30 | 0.0014 | 6.5971 | 0.9986 |
| `staged_dynamic_belief` | all | 2 | 25/30 | 0.1400 | 2.6978 | 0.8557 |
| `staged_dynamic_belief` | all | 3 | 26/28 | 0.7759 | 0.5355 | 0.2333 |
| `staged_dynamic_belief` | all | 4 | 8/8 | 0.7005 | 0.7125 | 0.3081 |
| `staged_dynamic_belief` | all | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_dynamic_belief` | all | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `staged_dynamic_belief` | never_used | 1 | 20/25 | 0.0014 | 6.5547 | 0.9985 |
| `staged_dynamic_belief` | never_used | 2 | 20/25 | 0.1352 | 2.7316 | 0.8562 |
| `staged_dynamic_belief` | never_used | 3 | 21/23 | 0.7912 | 0.5505 | 0.2233 |
| `staged_dynamic_belief` | never_used | 4 | 6/6 | 0.6094 | 0.9412 | 0.4105 |
| `staged_dynamic_belief` | never_used | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_dynamic_belief` | never_used | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `staged_dynamic_belief` | reused | 1 | 5/5 | 0.0012 | 6.7664 | 0.9990 |
| `staged_dynamic_belief` | reused | 2 | 5/5 | 0.1591 | 2.5630 | 0.8540 |
| `staged_dynamic_belief` | reused | 3 | 5/5 | 0.7114 | 0.4725 | 0.2753 |
| `staged_dynamic_belief` | reused | 4 | 2/2 | 0.9741 | 0.0262 | 0.0008 |
| `staged_dynamic_belief` | out_of_core | 1 | 0/5 | n/a | n/a | n/a |
| `staged_dynamic_belief` | out_of_core | 2 | 0/5 | n/a | n/a | n/a |
| `staged_dynamic_belief` | out_of_core | 3 | 3/5 | 0.3808 | 2.6524 | 0.8362 |
| `staged_dynamic_belief` | out_of_core | 4 | 4/4 | 0.4190 | 1.4069 | 0.6157 |
| `staged_dynamic_belief` | out_of_core | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_dynamic_belief` | out_of_core | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
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

Reference `staged_dynamic_belief` all-game mean sensitivity: penalty 6 = 3.3333 [3.1000, 3.6000]; penalty 7 = 3.3333 [3.1000, 3.6000]; penalty 8 = 3.3333 [3.1000, 3.6000].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
