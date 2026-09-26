<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-26` using selection `range` (2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 21.67 s; process peak working set: 85.6 MiB; enforced budget: 900 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `finite_selected_tail_fixed` | 100.0% (30/30) | 100.0% (30/30) | 3.5000 [3.2667, 3.7333] | 3.5000 [3.2667, 3.7333] | 60.0% | 90.0% | +0.0000 [+0.0000, +0.0000] | 0/30/0 | 7.5789 | 0.9990 | 213.08 ms | n/a/n/a |
| `finite_near_core_fixed` | 100.0% (30/30) | 100.0% (30/30) | 3.6333 [3.4000, 3.8667] | 3.6333 [3.4000, 3.8667] | 50.0% | 90.0% | +0.1333 [-0.0333, +0.3000] | 3/20/7 | 8.6048 | 0.9990 | 211.31 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `finite_selected_tail_fixed` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/105 | 0/105 | 0/0 |
| `finite_near_core_fixed` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 0/0/0/0/109 | 0/109 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `finite_selected_tail_fixed` | all | 1 | 30/30 | 0.0011 | 7.5789 | 0.9990 |
| `finite_selected_tail_fixed` | all | 2 | 30/30 | 0.1298 | 3.3265 | 0.8794 |
| `finite_selected_tail_fixed` | all | 3 | 29/29 | 0.6241 | 1.0179 | 0.3941 |
| `finite_selected_tail_fixed` | all | 4 | 12/12 | 0.8400 | 0.3023 | 0.1656 |
| `finite_selected_tail_fixed` | all | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_selected_tail_fixed` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_selected_tail_fixed` | never_used | 1 | 25/25 | 0.0011 | 7.7276 | 0.9990 |
| `finite_selected_tail_fixed` | never_used | 2 | 25/25 | 0.1367 | 3.5057 | 0.8719 |
| `finite_selected_tail_fixed` | never_used | 3 | 24/24 | 0.6077 | 1.1199 | 0.4113 |
| `finite_selected_tail_fixed` | never_used | 4 | 10/10 | 0.8084 | 0.3623 | 0.1987 |
| `finite_selected_tail_fixed` | never_used | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_selected_tail_fixed` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_selected_tail_fixed` | reused | 1 | 5/5 | 0.0011 | 6.8352 | 0.9990 |
| `finite_selected_tail_fixed` | reused | 2 | 5/5 | 0.0949 | 2.4310 | 0.9165 |
| `finite_selected_tail_fixed` | reused | 3 | 5/5 | 0.7024 | 0.5283 | 0.3119 |
| `finite_selected_tail_fixed` | reused | 4 | 2/2 | 0.9978 | 0.0022 | 0.0000 |
| `finite_selected_tail_fixed` | out_of_core | 1 | 5/5 | 0.0000 | 12.1442 | 1.0011 |
| `finite_selected_tail_fixed` | out_of_core | 2 | 5/5 | 0.0005 | 8.0167 | 1.1165 |
| `finite_selected_tail_fixed` | out_of_core | 3 | 5/5 | 0.0536 | 3.7314 | 1.0907 |
| `finite_selected_tail_fixed` | out_of_core | 4 | 5/5 | 0.6219 | 0.7194 | 0.3973 |
| `finite_selected_tail_fixed` | out_of_core | 5 | 3/3 | 0.8333 | 0.2310 | 0.1667 |
| `finite_selected_tail_fixed` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_near_core_fixed` | all | 1 | 30/30 | 0.0011 | 8.6048 | 0.9990 |
| `finite_near_core_fixed` | all | 2 | 30/30 | 0.0862 | 4.5242 | 0.9238 |
| `finite_near_core_fixed` | all | 3 | 30/30 | 0.5804 | 1.9586 | 0.4775 |
| `finite_near_core_fixed` | all | 4 | 15/15 | 0.7967 | 1.4012 | 0.3319 |
| `finite_near_core_fixed` | all | 5 | 3/3 | 0.7500 | 0.4621 | 0.2500 |
| `finite_near_core_fixed` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_near_core_fixed` | never_used | 1 | 25/25 | 0.0011 | 8.9725 | 0.9991 |
| `finite_near_core_fixed` | never_used | 2 | 25/25 | 0.0878 | 4.8519 | 0.9212 |
| `finite_near_core_fixed` | never_used | 3 | 25/25 | 0.5850 | 2.1596 | 0.4812 |
| `finite_near_core_fixed` | never_used | 4 | 13/13 | 0.7655 | 1.6168 | 0.3829 |
| `finite_near_core_fixed` | never_used | 5 | 3/3 | 0.7500 | 0.4621 | 0.2500 |
| `finite_near_core_fixed` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_near_core_fixed` | reused | 1 | 5/5 | 0.0012 | 6.7665 | 0.9990 |
| `finite_near_core_fixed` | reused | 2 | 5/5 | 0.0781 | 2.8860 | 0.9364 |
| `finite_near_core_fixed` | reused | 3 | 5/5 | 0.5570 | 0.9539 | 0.4593 |
| `finite_near_core_fixed` | reused | 4 | 2/2 | 1.0000 | 0.0000 | 0.0000 |
| `finite_near_core_fixed` | out_of_core | 1 | 5/5 | 0.0000 | 18.6433 | 1.0013 |
| `finite_near_core_fixed` | out_of_core | 2 | 5/5 | 0.0000 | 14.1411 | 1.0738 |
| `finite_near_core_fixed` | out_of_core | 3 | 5/5 | 0.0667 | 8.8635 | 1.2961 |
| `finite_near_core_fixed` | out_of_core | 4 | 5/5 | 0.5001 | 4.0551 | 0.8954 |
| `finite_near_core_fixed` | out_of_core | 5 | 3/3 | 0.7500 | 0.4621 | 0.2500 |
| `finite_near_core_fixed` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `finite_selected_tail_fixed` all-game mean sensitivity: penalty 6 = 3.5000 [3.2667, 3.7333]; penalty 7 = 3.5000 [3.2667, 3.7333]; penalty 8 = 3.5000 [3.2667, 3.7333].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
