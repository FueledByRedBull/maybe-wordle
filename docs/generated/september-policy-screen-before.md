<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-08-20` through `2026-08-26` using selection `range` (2026-08-20..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 47.92 s; process peak working set: 92.4 MiB; enforced budget: 1100 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | 100.0% (7/7) | 71.4% (5/7) | 4.1429 [4.1429, 4.1429] | 3.8571 [3.8571, 3.8571] | 57.1% | 71.4% | +0.0000 [+0.0000, +0.0000] | 0/7/0 | 6.6429 | 0.9987 | 19.61 ms | n/a/n/a |
| `finite_same_prior` | 100.0% (7/7) | 100.0% (7/7) | 3.5714 [3.5714, 3.5714] | 3.5714 [3.5714, 3.5714] | 42.9% | 85.7% | -0.5714 [-0.5714, -0.5714] | 2/4/1 | 8.2638 | 0.9994 | 264.63 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_incumbent` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 9/1/0/17/0 | 2/10 | 0/0 |
| `finite_same_prior` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/25 | 0/25 | 0/0 |

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
| `finite_same_prior` | all | 2 | 7/7 | 0.0684 | 5.4655 | 0.9269 |
| `finite_same_prior` | all | 3 | 6/6 | 0.4837 | 1.7799 | 0.5284 |
| `finite_same_prior` | all | 4 | 4/4 | 0.7827 | 0.4895 | 0.2143 |
| `finite_same_prior` | all | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_same_prior` | never_used | 1 | 5/5 | 0.0008 | 8.8331 | 0.9996 |
| `finite_same_prior` | never_used | 2 | 5/5 | 0.0872 | 6.1152 | 0.9055 |
| `finite_same_prior` | never_used | 3 | 4/4 | 0.3642 | 2.4824 | 0.6727 |
| `finite_same_prior` | never_used | 4 | 4/4 | 0.7827 | 0.4895 | 0.2143 |
| `finite_same_prior` | never_used | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_same_prior` | reused | 1 | 2/2 | 0.0011 | 6.8405 | 0.9990 |
| `finite_same_prior` | reused | 2 | 2/2 | 0.0215 | 3.8412 | 0.9801 |
| `finite_same_prior` | reused | 3 | 2/2 | 0.7225 | 0.3748 | 0.2397 |
| `finite_same_prior` | out_of_core | 1 | 2/2 | 0.0000 | 12.1441 | 1.0011 |
| `finite_same_prior` | out_of_core | 2 | 2/2 | 0.0000 | 10.1817 | 1.0083 |
| `finite_same_prior` | out_of_core | 3 | 2/2 | 0.0673 | 4.4664 | 1.0679 |
| `finite_same_prior` | out_of_core | 4 | 2/2 | 0.5714 | 0.9730 | 0.4286 |
| `finite_same_prior` | out_of_core | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `staged_incumbent` all-game mean sensitivity: penalty 6 = 3.8571 [3.8571, 3.8571]; penalty 7 = 4.1429 [4.1429, 4.1429]; penalty 8 = 4.4286 [4.4286, 4.4286].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
