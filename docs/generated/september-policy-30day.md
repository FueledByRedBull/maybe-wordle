<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-26` using selection `range` (2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 111.50 s; process peak working set: 143.5 MiB; enforced budget: 1100 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | 100.0% (30/30) | 93.3% (28/30) | 3.4000 [3.1000, 3.7667] | 3.3333 [3.1000, 3.6000] | 73.3% | 93.3% | +0.0000 [+0.0000, +0.0000] | 0/30/0 | 6.5971 | 0.9986 | 27.53 ms | n/a/n/a |
| `finite_same_prior` | 100.0% (30/30) | 100.0% (30/30) | 3.4667 [3.2000, 3.7667] | 3.4667 [3.2000, 3.7667] | 56.7% | 93.3% | +0.0667 [-0.4333, +0.5333] | 5/15/10 | 7.5789 | 0.9990 | 268.35 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 34/12/0/54/0 | 5/33 | 0/0 |
| `finite_same_prior` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/104 | 0/104 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | all | 1 | 25/30 | 0.0014 | 6.5971 | 0.9986 |
| `staged_incumbent` | all | 2 | 25/30 | 0.1400 | 2.6978 | 0.8557 |
| `staged_incumbent` | all | 3 | 26/28 | 0.7759 | 0.5355 | 0.2333 |
| `staged_incumbent` | all | 4 | 8/8 | 0.7005 | 0.7125 | 0.3081 |
| `staged_incumbent` | all | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_incumbent` | all | 6 | 2/2 | 0.5000 | 0.6931 | 0.5000 |
| `staged_incumbent` | never_used | 1 | 20/25 | 0.0014 | 6.5547 | 0.9985 |
| `staged_incumbent` | never_used | 2 | 20/25 | 0.1352 | 2.7316 | 0.8562 |
| `staged_incumbent` | never_used | 3 | 21/23 | 0.7912 | 0.5505 | 0.2233 |
| `staged_incumbent` | never_used | 4 | 6/6 | 0.6094 | 0.9412 | 0.4105 |
| `staged_incumbent` | never_used | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_incumbent` | never_used | 6 | 2/2 | 0.5000 | 0.6931 | 0.5000 |
| `staged_incumbent` | reused | 1 | 5/5 | 0.0012 | 6.7664 | 0.9990 |
| `staged_incumbent` | reused | 2 | 5/5 | 0.1591 | 2.5630 | 0.8540 |
| `staged_incumbent` | reused | 3 | 5/5 | 0.7114 | 0.4725 | 0.2753 |
| `staged_incumbent` | reused | 4 | 2/2 | 0.9741 | 0.0262 | 0.0008 |
| `staged_incumbent` | out_of_core | 1 | 0/5 | n/a | n/a | n/a |
| `staged_incumbent` | out_of_core | 2 | 0/5 | n/a | n/a | n/a |
| `staged_incumbent` | out_of_core | 3 | 3/5 | 0.3808 | 2.6524 | 0.8362 |
| `staged_incumbent` | out_of_core | 4 | 4/4 | 0.4190 | 1.4069 | 0.6157 |
| `staged_incumbent` | out_of_core | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_incumbent` | out_of_core | 6 | 2/2 | 0.5000 | 0.6931 | 0.5000 |
| `finite_same_prior` | all | 1 | 30/30 | 0.0011 | 7.5789 | 0.9990 |
| `finite_same_prior` | all | 2 | 30/30 | 0.0799 | 4.0427 | 0.9240 |
| `finite_same_prior` | all | 3 | 29/29 | 0.5702 | 1.2654 | 0.4855 |
| `finite_same_prior` | all | 4 | 13/13 | 0.8605 | 0.2191 | 0.1348 |
| `finite_same_prior` | all | 5 | 2/2 | 0.9980 | 0.0020 | 0.0000 |
| `finite_same_prior` | never_used | 1 | 25/25 | 0.0011 | 7.7276 | 0.9990 |
| `finite_same_prior` | never_used | 2 | 25/25 | 0.0834 | 4.2568 | 0.9201 |
| `finite_same_prior` | never_used | 3 | 24/24 | 0.5382 | 1.4342 | 0.5232 |
| `finite_same_prior` | never_used | 4 | 11/11 | 0.8377 | 0.2563 | 0.1592 |
| `finite_same_prior` | never_used | 5 | 2/2 | 0.9980 | 0.0020 | 0.0000 |
| `finite_same_prior` | reused | 1 | 5/5 | 0.0011 | 6.8352 | 0.9990 |
| `finite_same_prior` | reused | 2 | 5/5 | 0.0626 | 2.9722 | 0.9434 |
| `finite_same_prior` | reused | 3 | 5/5 | 0.7238 | 0.4548 | 0.3043 |
| `finite_same_prior` | reused | 4 | 2/2 | 0.9854 | 0.0147 | 0.0005 |
| `finite_same_prior` | out_of_core | 1 | 5/5 | 0.0000 | 12.1442 | 1.0011 |
| `finite_same_prior` | out_of_core | 2 | 5/5 | 0.0003 | 8.2974 | 1.0664 |
| `finite_same_prior` | out_of_core | 3 | 5/5 | 0.0293 | 4.6839 | 1.3534 |
| `finite_same_prior` | out_of_core | 4 | 4/4 | 0.6875 | 0.5199 | 0.3125 |
| `finite_same_prior` | out_of_core | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `staged_incumbent` all-game mean sensitivity: penalty 6 = 3.3333 [3.1000, 3.6000]; penalty 7 = 3.4000 [3.1000, 3.7667]; penalty 8 = 3.4667 [3.1000, 3.9667].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
