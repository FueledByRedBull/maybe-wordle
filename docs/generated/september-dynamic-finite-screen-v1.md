<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-01` using selection `range` (2026-07-28..2026-08-01) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 13.98 s; process peak working set: 91.5 MiB; enforced budget: 1200 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | 100.0% (5/5) | 100.0% (5/5) | 3.0000 [3.0000, 3.0000] | 3.0000 [3.0000, 3.0000] | 100.0% | 100.0% | +0.0000 [+0.0000, +0.0000] | 0/5/0 | 6.5587 | 0.9985 | 21.17 ms | n/a/n/a |
| `finite_fast_dynamic` | 100.0% (5/5) | 100.0% (5/5) | 3.2000 [3.2000, 3.2000] | 3.2000 [3.2000, 3.2000] | 80.0% | 100.0% | +0.2000 [+0.2000, +0.2000] | 0/4/1 | 6.5587 | 0.9985 | 257.33 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 5/1/0/9/0 | 0/5 | 0/0 |
| `finite_fast_dynamic` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 0/0/0/0/16 | 0/3 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | all | 1 | 5/5 | 0.0014 | 6.5587 | 0.9985 |
| `selected_staged` | all | 2 | 5/5 | 0.1840 | 2.1542 | 0.8029 |
| `selected_staged` | all | 3 | 5/5 | 0.9652 | 0.0355 | 0.0017 |
| `selected_staged` | never_used | 1 | 4/4 | 0.0014 | 6.5533 | 0.9985 |
| `selected_staged` | never_used | 2 | 4/4 | 0.1497 | 2.4087 | 0.8340 |
| `selected_staged` | never_used | 3 | 4/4 | 0.9701 | 0.0304 | 0.0013 |
| `selected_staged` | reused | 1 | 1/1 | 0.0014 | 6.5799 | 0.9986 |
| `selected_staged` | reused | 2 | 1/1 | 0.3210 | 1.1364 | 0.6784 |
| `selected_staged` | reused | 3 | 1/1 | 0.9456 | 0.0559 | 0.0033 |
| `finite_fast_dynamic` | all | 1 | 5/5 | 0.0014 | 6.5587 | 0.9985 |
| `finite_fast_dynamic` | all | 2 | 5/5 | 0.1840 | 2.1542 | 0.8029 |
| `finite_fast_dynamic` | all | 3 | 5/5 | 0.8246 | 0.3184 | 0.1529 |
| `finite_fast_dynamic` | all | 4 | 1/1 | 0.5009 | 0.6913 | 0.4982 |
| `finite_fast_dynamic` | never_used | 1 | 4/4 | 0.0014 | 6.5533 | 0.9985 |
| `finite_fast_dynamic` | never_used | 2 | 4/4 | 0.1497 | 2.4087 | 0.8340 |
| `finite_fast_dynamic` | never_used | 3 | 4/4 | 0.7943 | 0.3840 | 0.1903 |
| `finite_fast_dynamic` | never_used | 4 | 1/1 | 0.5009 | 0.6913 | 0.4982 |
| `finite_fast_dynamic` | reused | 1 | 1/1 | 0.0014 | 6.5799 | 0.9986 |
| `finite_fast_dynamic` | reused | 2 | 1/1 | 0.3210 | 1.1364 | 0.6784 |
| `finite_fast_dynamic` | reused | 3 | 1/1 | 0.9456 | 0.0559 | 0.0033 |

Reference `selected_staged` all-game mean sensitivity: penalty 6 = 3.0000 [3.0000, 3.0000]; penalty 7 = 3.0000 [3.0000, 3.0000]; penalty 8 = 3.0000 [3.0000, 3.0000].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
