<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-26` using selection `range` (2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 52.10 s; process peak working set: 133.3 MiB; enforced budget: 1200 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | 100.0% (30/30) | 100.0% (30/30) | 3.3333 [3.1000, 3.6000] | 3.3333 [3.1000, 3.6000] | 73.3% | 93.3% | +0.0000 [+0.0000, +0.0000] | 0/30/0 | 6.5971 | 0.9986 | 21.24 ms | n/a/n/a |
| `finite_fast_dynamic` | 100.0% (30/30) | 100.0% (30/30) | 3.4000 [3.2000, 3.5667] | 3.4000 [3.2000, 3.5667] | 73.3% | 86.7% | +0.0667 [-0.1333, +0.2667] | 4/21/5 | 6.5971 | 0.9986 | 256.62 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 34/12/0/54/0 | 5/33 | 0/0 |
| `finite_fast_dynamic` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 0/0/0/0/102 | 5/29 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | all | 1 | 25/30 | 0.0014 | 6.5971 | 0.9986 |
| `selected_staged` | all | 2 | 25/30 | 0.1400 | 2.6978 | 0.8557 |
| `selected_staged` | all | 3 | 26/28 | 0.7759 | 0.5355 | 0.2333 |
| `selected_staged` | all | 4 | 8/8 | 0.7005 | 0.7125 | 0.3081 |
| `selected_staged` | all | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `selected_staged` | all | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `selected_staged` | never_used | 1 | 20/25 | 0.0014 | 6.5547 | 0.9985 |
| `selected_staged` | never_used | 2 | 20/25 | 0.1352 | 2.7316 | 0.8562 |
| `selected_staged` | never_used | 3 | 21/23 | 0.7912 | 0.5505 | 0.2233 |
| `selected_staged` | never_used | 4 | 6/6 | 0.6094 | 0.9412 | 0.4105 |
| `selected_staged` | never_used | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `selected_staged` | never_used | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `selected_staged` | reused | 1 | 5/5 | 0.0012 | 6.7664 | 0.9990 |
| `selected_staged` | reused | 2 | 5/5 | 0.1591 | 2.5630 | 0.8540 |
| `selected_staged` | reused | 3 | 5/5 | 0.7114 | 0.4725 | 0.2753 |
| `selected_staged` | reused | 4 | 2/2 | 0.9741 | 0.0262 | 0.0008 |
| `selected_staged` | out_of_core | 1 | 0/5 | n/a | n/a | n/a |
| `selected_staged` | out_of_core | 2 | 0/5 | n/a | n/a | n/a |
| `selected_staged` | out_of_core | 3 | 3/5 | 0.3808 | 2.6524 | 0.8362 |
| `selected_staged` | out_of_core | 4 | 4/4 | 0.4190 | 1.4069 | 0.6157 |
| `selected_staged` | out_of_core | 5 | 2/2 | 0.2024 | 1.6027 | 0.9514 |
| `selected_staged` | out_of_core | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fast_dynamic` | all | 1 | 25/30 | 0.0014 | 6.5971 | 0.9986 |
| `finite_fast_dynamic` | all | 2 | 25/30 | 0.1400 | 2.6978 | 0.8557 |
| `finite_fast_dynamic` | all | 3 | 27/29 | 0.7623 | 0.6888 | 0.2672 |
| `finite_fast_dynamic` | all | 4 | 7/8 | 0.6228 | 0.7228 | 0.4276 |
| `finite_fast_dynamic` | all | 5 | 4/4 | 0.8057 | 0.3754 | 0.2403 |
| `finite_fast_dynamic` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fast_dynamic` | never_used | 1 | 20/25 | 0.0014 | 6.5547 | 0.9985 |
| `finite_fast_dynamic` | never_used | 2 | 20/25 | 0.1352 | 2.7316 | 0.8562 |
| `finite_fast_dynamic` | never_used | 3 | 22/24 | 0.7482 | 0.7759 | 0.2934 |
| `finite_fast_dynamic` | never_used | 4 | 6/7 | 0.6439 | 0.7266 | 0.4152 |
| `finite_fast_dynamic` | never_used | 5 | 3/3 | 0.7409 | 0.5005 | 0.3204 |
| `finite_fast_dynamic` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fast_dynamic` | reused | 1 | 5/5 | 0.0012 | 6.7664 | 0.9990 |
| `finite_fast_dynamic` | reused | 2 | 5/5 | 0.1591 | 2.5630 | 0.8540 |
| `finite_fast_dynamic` | reused | 3 | 5/5 | 0.8244 | 0.3057 | 0.1516 |
| `finite_fast_dynamic` | reused | 4 | 1/1 | 0.4963 | 0.7005 | 0.5024 |
| `finite_fast_dynamic` | reused | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `finite_fast_dynamic` | out_of_core | 1 | 0/5 | n/a | n/a | n/a |
| `finite_fast_dynamic` | out_of_core | 2 | 0/5 | n/a | n/a | n/a |
| `finite_fast_dynamic` | out_of_core | 3 | 3/5 | 0.0442 | 4.3681 | 1.4139 |
| `finite_fast_dynamic` | out_of_core | 4 | 4/5 | 0.5941 | 0.9135 | 0.4981 |
| `finite_fast_dynamic` | out_of_core | 5 | 3/3 | 0.7409 | 0.5005 | 0.3204 |
| `finite_fast_dynamic` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |

Reference `selected_staged` all-game mean sensitivity: penalty 6 = 3.3333 [3.1000, 3.6000]; penalty 7 = 3.3333 [3.1000, 3.6000]; penalty 8 = 3.3333 [3.1000, 3.6000].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
