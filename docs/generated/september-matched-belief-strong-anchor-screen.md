<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-01` using selection `range` (2026-07-28..2026-08-01) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 74.08 s; process peak working set: 91.1 MiB; enforced budget: 180 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | 100.0% (5/5) | 100.0% (5/5) | 3.0000 [3.0000, 3.0000] | 3.0000 [3.0000, 3.0000] | 80.0% | 100.0% | +0.0000 [+0.0000, +0.0000] | 0/5/0 | 6.6274 | 0.9985 | 329.76 ms | n/a/n/a |
| `finite_strong` | 100.0% (5/5) | 100.0% (5/5) | 3.6000 [3.6000, 3.6000] | 3.6000 [3.6000, 3.6000] | 40.0% | 100.0% | +0.6000 [+0.6000, +0.6000] | 1/1/3 | 6.6274 | 0.9985 | 2016.74 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 8/0/0/7/0 | 0/15 | 0/0 |
| `finite_strong` | 0.0% | 0.0% | 0.0% | 0.0013 [0.0013, 0.0013] | 0/0/0/0/18 | 0/18 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `staged_fixed_belief` | all | 1 | 5/5 | 0.0013 | 6.6274 | 0.9985 |
| `staged_fixed_belief` | all | 2 | 5/5 | 0.2177 | 2.4042 | 0.7498 |
| `staged_fixed_belief` | all | 3 | 4/4 | 0.7720 | 0.4694 | 0.2071 |
| `staged_fixed_belief` | all | 4 | 1/1 | 0.9568 | 0.0442 | 0.0022 |
| `staged_fixed_belief` | never_used | 1 | 4/4 | 0.0013 | 6.6221 | 0.9985 |
| `staged_fixed_belief` | never_used | 2 | 4/4 | 0.2650 | 2.1137 | 0.6952 |
| `staged_fixed_belief` | never_used | 3 | 3/3 | 0.9741 | 0.0266 | 0.0015 |
| `staged_fixed_belief` | reused | 1 | 1/1 | 0.0013 | 6.6487 | 0.9986 |
| `staged_fixed_belief` | reused | 2 | 1/1 | 0.0283 | 3.5662 | 0.9685 |
| `staged_fixed_belief` | reused | 3 | 1/1 | 0.1657 | 1.7978 | 0.8237 |
| `staged_fixed_belief` | reused | 4 | 1/1 | 0.9568 | 0.0442 | 0.0022 |
| `finite_strong` | all | 1 | 5/5 | 0.0013 | 6.6274 | 0.9985 |
| `finite_strong` | all | 2 | 5/5 | 0.0891 | 3.1959 | 0.9047 |
| `finite_strong` | all | 3 | 5/5 | 0.5433 | 0.8327 | 0.4359 |
| `finite_strong` | all | 4 | 3/3 | 0.9212 | 0.0887 | 0.0301 |
| `finite_strong` | never_used | 1 | 4/4 | 0.0013 | 6.6221 | 0.9985 |
| `finite_strong` | never_used | 2 | 4/4 | 0.0920 | 3.3558 | 0.9024 |
| `finite_strong` | never_used | 3 | 4/4 | 0.4390 | 1.0308 | 0.5444 |
| `finite_strong` | never_used | 4 | 3/3 | 0.9212 | 0.0887 | 0.0301 |
| `finite_strong` | reused | 1 | 1/1 | 0.0013 | 6.6487 | 0.9986 |
| `finite_strong` | reused | 2 | 1/1 | 0.0776 | 2.5565 | 0.9141 |
| `finite_strong` | reused | 3 | 1/1 | 0.9606 | 0.0402 | 0.0018 |

Reference `staged_fixed_belief` all-game mean sensitivity: penalty 6 = 3.0000 [3.0000, 3.0000]; penalty 7 = 3.0000 [3.0000, 3.0000]; penalty 8 = 3.0000 [3.0000, 3.0000].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
