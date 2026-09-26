<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-08-20` through `2026-08-26` using selection `range` (2026-08-20..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 50.13 s; process peak working set: 92.3 MiB; enforced budget: 1100 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | 100.0% (7/7) | 71.4% (5/7) | 4.1429 [4.1429, 4.1429] | 3.8571 [3.8571, 3.8571] | 57.1% | 71.4% | +0.0000 [+0.0000, +0.0000] | 0/7/0 | 6.6429 | 0.9987 | 18.98 ms | n/a/n/a |
| `finite_same_prior` | 100.0% (7/7) | 100.0% (7/7) | 3.7143 [3.7143, 3.7143] | 3.7143 [3.7143, 3.7143] | 28.6% | 85.7% | -0.4286 [-0.4286, -0.4286] | 2/3/2 | 8.2638 | 0.9994 | 275.07 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 9/1/0/17/0 | 2/10 | 0/0 |
| `finite_same_prior` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/26 | 0/26 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | all | 1 | 5/7 | 0.0013 | 6.6429 | 0.9987 |
| `staged_incumbent` | all | 2 | 5/7 | 0.2904 | 2.2298 | 0.7116 |
| `staged_incumbent` | all | 3 | 4/6 | 0.7246 | 0.3804 | 0.2511 |
| `staged_incumbent` | all | 4 | 3/3 | 0.3880 | 1.6488 | 0.6543 |
| `staged_incumbent` | all | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_incumbent` | all | 6 | 2/2 | 0.5000 | 0.6931 | 0.5000 |
| `staged_incumbent` | never_used | 1 | 3/5 | 0.0014 | 6.5569 | 0.9985 |
| `staged_incumbent` | never_used | 2 | 3/5 | 0.3390 | 2.4499 | 0.6471 |
| `staged_incumbent` | never_used | 3 | 2/4 | 0.4843 | 0.7251 | 0.5007 |
| `staged_incumbent` | never_used | 4 | 3/3 | 0.3880 | 1.6488 | 0.6543 |
| `staged_incumbent` | never_used | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_incumbent` | never_used | 6 | 2/2 | 0.5000 | 0.6931 | 0.5000 |
| `staged_incumbent` | reused | 1 | 2/2 | 0.0012 | 6.7718 | 0.9990 |
| `staged_incumbent` | reused | 2 | 2/2 | 0.2175 | 1.8997 | 0.8084 |
| `staged_incumbent` | reused | 3 | 2/2 | 0.9649 | 0.0358 | 0.0016 |
| `staged_incumbent` | out_of_core | 1 | 0/2 | n/a | n/a | n/a |
| `staged_incumbent` | out_of_core | 2 | 0/2 | n/a | n/a | n/a |
| `staged_incumbent` | out_of_core | 3 | 0/2 | n/a | n/a | n/a |
| `staged_incumbent` | out_of_core | 4 | 2/2 | 0.0880 | 2.4672 | 0.9814 |
| `staged_incumbent` | out_of_core | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `staged_incumbent` | out_of_core | 6 | 2/2 | 0.5000 | 0.6931 | 0.5000 |
| `finite_same_prior` | all | 1 | 7/7 | 0.0009 | 8.2638 | 0.9994 |
| `finite_same_prior` | all | 2 | 7/7 | 0.0723 | 4.6887 | 0.9579 |
| `finite_same_prior` | all | 3 | 6/6 | 0.4653 | 1.1131 | 0.5141 |
| `finite_same_prior` | all | 4 | 5/5 | 0.8967 | 0.1419 | 0.1000 |
| `finite_same_prior` | all | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_same_prior` | never_used | 1 | 5/5 | 0.0008 | 8.8331 | 0.9996 |
| `finite_same_prior` | never_used | 2 | 5/5 | 0.0874 | 5.1910 | 0.9540 |
| `finite_same_prior` | never_used | 3 | 4/4 | 0.3847 | 1.3611 | 0.5949 |
| `finite_same_prior` | never_used | 4 | 4/4 | 0.8720 | 0.1763 | 0.1250 |
| `finite_same_prior` | never_used | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_same_prior` | reused | 1 | 2/2 | 0.0011 | 6.8405 | 0.9990 |
| `finite_same_prior` | reused | 2 | 2/2 | 0.0346 | 3.4331 | 0.9675 |
| `finite_same_prior` | reused | 3 | 2/2 | 0.6266 | 0.6171 | 0.3525 |
| `finite_same_prior` | reused | 4 | 1/1 | 0.9956 | 0.0044 | 0.0000 |
| `finite_same_prior` | out_of_core | 1 | 2/2 | 0.0000 | 12.1441 | 1.0011 |
| `finite_same_prior` | out_of_core | 2 | 2/2 | 0.0006 | 7.8712 | 1.1295 |
| `finite_same_prior` | out_of_core | 3 | 2/2 | 0.1082 | 2.2237 | 0.9123 |
| `finite_same_prior` | out_of_core | 4 | 2/2 | 0.7500 | 0.3466 | 0.2500 |
| `finite_same_prior` | out_of_core | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `staged_incumbent` all-game mean sensitivity: penalty 6 = 3.8571 [3.8571, 3.8571]; penalty 7 = 4.1429 [4.1429, 4.1429]; penalty 8 = 4.4286 [4.4286, 4.4286].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
