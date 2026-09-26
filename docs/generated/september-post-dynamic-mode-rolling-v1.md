<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2025-07-03` through `2026-08-26` using selection `rolling_folds` (2025-07-03..2025-08-01, 2025-08-02..2025-08-31, 2025-09-01..2025-09-30, 2025-10-01..2025-10-30, 2025-10-31..2025-11-29, 2025-11-30..2025-12-29, 2025-12-30..2026-01-28, 2026-01-29..2026-02-27, 2026-02-28..2026-03-29, 2026-03-30..2026-04-28, 2026-04-29..2026-05-28, 2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 582.94 s; process peak working set: 162.6 MiB; enforced budget: 1200 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | 100.0% (360/360) | 100.0% (360/360) | 3.1944 [3.1194, 3.2722] | 3.1944 [3.1194, 3.2722] | 76.4% | 96.7% | +0.0000 [+0.0000, +0.0000] | 0/360/0 | 6.6703 | 0.9987 | 20.84 ms | n/a/n/a |
| `v19b_staged` | 100.0% (360/360) | 100.0% (360/360) | 3.1917 [3.1194, 3.2694] | 3.1917 [3.1194, 3.2694] | 76.1% | 96.9% | -0.0028 [-0.0222, +0.0167] | 6/347/7 | 6.6703 | 0.9987 | 20.97 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | 0.3% | 0.6% | 0.6% | 0.0017 [0.0013, 0.0077] | 386/139/0/625/0 | 36/292 | 0/0 |
| `v19b_staged` | 0.3% | 0.6% | 0.6% | 0.0017 [0.0013, 0.0077] | 386/131/0/632/0 | 36/297 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `selected_staged` | all | 1 | 332/360 | 0.0013 | 6.6703 | 0.9987 |
| `selected_staged` | all | 2 | 334/360 | 0.1201 | 2.8215 | 0.8792 |
| `selected_staged` | all | 3 | 320/329 | 0.7421 | 0.5525 | 0.2577 |
| `selected_staged` | all | 4 | 84/85 | 0.8448 | 0.3250 | 0.1448 |
| `selected_staged` | all | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `selected_staged` | all | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `selected_staged` | never_used | 1 | 315/343 | 0.0013 | 6.6155 | 0.9986 |
| `selected_staged` | never_used | 2 | 317/343 | 0.1219 | 2.7757 | 0.8748 |
| `selected_staged` | never_used | 3 | 303/312 | 0.7527 | 0.5150 | 0.2428 |
| `selected_staged` | never_used | 4 | 76/77 | 0.8403 | 0.3450 | 0.1540 |
| `selected_staged` | never_used | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `selected_staged` | never_used | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `selected_staged` | reused | 1 | 17/17 | 0.0008 | 7.6872 | 0.9998 |
| `selected_staged` | reused | 2 | 17/17 | 0.0855 | 3.6756 | 0.9605 |
| `selected_staged` | reused | 3 | 17/17 | 0.5519 | 1.2222 | 0.5237 |
| `selected_staged` | reused | 4 | 8/8 | 0.8881 | 0.1358 | 0.0569 |
| `selected_staged` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `selected_staged` | out_of_core | 2 | 2/28 | 0.0170 | 4.6022 | 1.3337 |
| `selected_staged` | out_of_core | 3 | 19/28 | 0.2065 | 3.1306 | 1.0928 |
| `selected_staged` | out_of_core | 4 | 23/24 | 0.5470 | 1.0525 | 0.4781 |
| `selected_staged` | out_of_core | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `selected_staged` | out_of_core | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `v19b_staged` | all | 1 | 332/360 | 0.0013 | 6.6703 | 0.9987 |
| `v19b_staged` | all | 2 | 334/360 | 0.1195 | 2.7936 | 0.8798 |
| `v19b_staged` | all | 3 | 320/329 | 0.7348 | 0.5578 | 0.2643 |
| `v19b_staged` | all | 4 | 85/86 | 0.8612 | 0.2915 | 0.1276 |
| `v19b_staged` | all | 5 | 11/11 | 0.6832 | 0.5067 | 0.3280 |
| `v19b_staged` | all | 6 | 3/3 | 1.0000 | -0.0000 | 0.0000 |
| `v19b_staged` | never_used | 1 | 315/343 | 0.0013 | 6.6155 | 0.9986 |
| `v19b_staged` | never_used | 2 | 317/343 | 0.1210 | 2.7527 | 0.8756 |
| `v19b_staged` | never_used | 3 | 303/312 | 0.7447 | 0.5220 | 0.2501 |
| `v19b_staged` | never_used | 4 | 77/78 | 0.8585 | 0.3077 | 0.1349 |
| `v19b_staged` | never_used | 5 | 11/11 | 0.6832 | 0.5067 | 0.3280 |
| `v19b_staged` | never_used | 6 | 3/3 | 1.0000 | -0.0000 | 0.0000 |
| `v19b_staged` | reused | 1 | 17/17 | 0.0008 | 7.6872 | 0.9998 |
| `v19b_staged` | reused | 2 | 17/17 | 0.0906 | 3.5568 | 0.9588 |
| `v19b_staged` | reused | 3 | 17/17 | 0.5589 | 1.1963 | 0.5171 |
| `v19b_staged` | reused | 4 | 8/8 | 0.8881 | 0.1358 | 0.0569 |
| `v19b_staged` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `v19b_staged` | out_of_core | 2 | 2/28 | 0.0170 | 4.6022 | 1.3337 |
| `v19b_staged` | out_of_core | 3 | 19/28 | 0.2068 | 3.0853 | 1.0860 |
| `v19b_staged` | out_of_core | 4 | 23/24 | 0.5784 | 0.9746 | 0.4427 |
| `v19b_staged` | out_of_core | 5 | 11/11 | 0.6832 | 0.5067 | 0.3280 |
| `v19b_staged` | out_of_core | 6 | 3/3 | 1.0000 | -0.0000 | 0.0000 |

Reference `selected_staged` all-game mean sensitivity: penalty 6 = 3.1944 [3.1194, 3.2722]; penalty 7 = 3.1944 [3.1194, 3.2722]; penalty 8 = 3.1944 [3.1194, 3.2722].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
