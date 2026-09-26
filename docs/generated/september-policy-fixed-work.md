<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-26` using selection `range` (2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 30.34 s; process peak working set: 84.6 MiB; enforced budget: 900 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `finite_fast_250ms` | 100.0% (30/30) | 100.0% (30/30) | 3.6333 [3.5000, 3.7667] | 3.6333 [3.5000, 3.7667] | 43.3% | 90.0% | +0.0000 [+0.0000, +0.0000] | 0/30/0 | 7.5789 | 0.9990 | 262.71 ms | n/a/n/a |
| `finite_fast_fixed_work` | 100.0% (30/30) | 100.0% (30/30) | 3.5000 [3.2667, 3.7333] | 3.5000 [3.2667, 3.7333] | 60.0% | 90.0% | -0.1333 [-0.4000, +0.1667] | 12/10/8 | 7.5789 | 0.9990 | 154.46 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `finite_fast_250ms` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/109 | 0/109 | 0/0 |
| `finite_fast_fixed_work` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/105 | 0/105 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `finite_fast_250ms` | all | 1 | 30/30 | 0.0011 | 7.5789 | 0.9990 |
| `finite_fast_250ms` | all | 2 | 30/30 | 0.0442 | 4.6361 | 0.9577 |
| `finite_fast_250ms` | all | 3 | 29/29 | 0.4716 | 1.3545 | 0.5617 |
| `finite_fast_250ms` | all | 4 | 17/17 | 0.8770 | 0.2249 | 0.1191 |
| `finite_fast_250ms` | all | 5 | 3/3 | 0.9973 | 0.0027 | 0.0000 |
| `finite_fast_250ms` | never_used | 1 | 25/25 | 0.0011 | 7.7276 | 0.9990 |
| `finite_fast_250ms` | never_used | 2 | 25/25 | 0.0478 | 4.7842 | 0.9547 |
| `finite_fast_250ms` | never_used | 3 | 24/24 | 0.4758 | 1.4410 | 0.5662 |
| `finite_fast_250ms` | never_used | 4 | 15/15 | 0.8629 | 0.2526 | 0.1350 |
| `finite_fast_250ms` | never_used | 5 | 3/3 | 0.9973 | 0.0027 | 0.0000 |
| `finite_fast_250ms` | reused | 1 | 5/5 | 0.0011 | 6.8352 | 0.9990 |
| `finite_fast_250ms` | reused | 2 | 5/5 | 0.0266 | 3.8958 | 0.9728 |
| `finite_fast_250ms` | reused | 3 | 5/5 | 0.4512 | 0.9390 | 0.5399 |
| `finite_fast_250ms` | reused | 4 | 2/2 | 0.9830 | 0.0172 | 0.0005 |
| `finite_fast_250ms` | out_of_core | 1 | 5/5 | 0.0000 | 12.1442 | 1.0011 |
| `finite_fast_250ms` | out_of_core | 2 | 5/5 | 0.0002 | 9.1012 | 1.0381 |
| `finite_fast_250ms` | out_of_core | 3 | 5/5 | 0.0575 | 4.0571 | 1.2434 |
| `finite_fast_250ms` | out_of_core | 4 | 5/5 | 0.8286 | 0.3892 | 0.1714 |
| `finite_fast_250ms` | out_of_core | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fast_fixed_work` | all | 1 | 30/30 | 0.0011 | 7.5789 | 0.9990 |
| `finite_fast_fixed_work` | all | 2 | 30/30 | 0.1298 | 3.3265 | 0.8794 |
| `finite_fast_fixed_work` | all | 3 | 29/29 | 0.6241 | 1.0179 | 0.3941 |
| `finite_fast_fixed_work` | all | 4 | 12/12 | 0.8400 | 0.3023 | 0.1656 |
| `finite_fast_fixed_work` | all | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_fast_fixed_work` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fast_fixed_work` | never_used | 1 | 25/25 | 0.0011 | 7.7276 | 0.9990 |
| `finite_fast_fixed_work` | never_used | 2 | 25/25 | 0.1367 | 3.5057 | 0.8719 |
| `finite_fast_fixed_work` | never_used | 3 | 24/24 | 0.6077 | 1.1199 | 0.4113 |
| `finite_fast_fixed_work` | never_used | 4 | 10/10 | 0.8084 | 0.3623 | 0.1987 |
| `finite_fast_fixed_work` | never_used | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_fast_fixed_work` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fast_fixed_work` | reused | 1 | 5/5 | 0.0011 | 6.8352 | 0.9990 |
| `finite_fast_fixed_work` | reused | 2 | 5/5 | 0.0949 | 2.4310 | 0.9165 |
| `finite_fast_fixed_work` | reused | 3 | 5/5 | 0.7024 | 0.5283 | 0.3119 |
| `finite_fast_fixed_work` | reused | 4 | 2/2 | 0.9978 | 0.0022 | 0.0000 |
| `finite_fast_fixed_work` | out_of_core | 1 | 5/5 | 0.0000 | 12.1442 | 1.0011 |
| `finite_fast_fixed_work` | out_of_core | 2 | 5/5 | 0.0005 | 8.0167 | 1.1165 |
| `finite_fast_fixed_work` | out_of_core | 3 | 5/5 | 0.0536 | 3.7314 | 1.0907 |
| `finite_fast_fixed_work` | out_of_core | 4 | 5/5 | 0.6219 | 0.7194 | 0.3973 |
| `finite_fast_fixed_work` | out_of_core | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_fast_fixed_work` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `finite_fast_250ms` all-game mean sensitivity: penalty 6 = 3.6333 [3.5000, 3.7667]; penalty 7 = 3.6333 [3.5000, 3.7667]; penalty 8 = 3.6333 [3.5000, 3.7667].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
