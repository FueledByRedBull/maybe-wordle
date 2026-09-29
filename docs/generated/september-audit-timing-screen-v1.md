<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2026-07-28` through `2026-08-05` using selection `range` (2026-07-28..2026-08-05) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 19.58 s; process peak working set: 122.1 MiB; enforced budget: 180 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `previous_release_790ec2d` | 100.0% (9/9) | 100.0% (9/9) | 3.2222 [3.1111, 3.3333] | 3.2222 [3.1111, 3.3333] (modeled_games=9) | 77.8% | 100.0% | +0.2222 [+0.1111, +0.3333] | 0/7/2 | 7.1629 | 0.9992 | 23.15 ms | n/a/n/a |
| `uniform_entropy` | 100.0% (9/9) | 100.0% (9/9) | 3.4444 [3.2222, 3.6667] | 3.4444 [3.2222, 3.6667] (modeled_games=9) | 55.6% | 100.0% | +0.4444 [+0.2222, +0.6667] | 0/5/4 | 7.7664 | 0.9996 | 24.62 ms | n/a/n/a |
| `cooldown_entropy` | 100.0% (9/9) | 100.0% (9/9) | 3.2222 [3.1111, 3.3333] | 3.2222 [3.1111, 3.3333] (modeled_games=9) | 77.8% | 100.0% | +0.2222 [+0.1111, +0.3333] | 0/7/2 | 7.5965 | 0.9995 | 25.02 ms | n/a/n/a |
| `weighted_proxy_only` | 100.0% (9/9) | 100.0% (9/9) | 3.1111 [3.0000, 3.2222] | 3.1111 [3.0000, 3.2222] (modeled_games=9) | 88.9% | 100.0% | +0.1111 [+0.0000, +0.2222] | 0/8/1 | 6.5562 | 0.9985 | 26.73 ms | n/a/n/a |
| `weighted_proxy_exact_endgame` | 100.0% (9/9) | 100.0% (9/9) | 3.0000 [3.0000, 3.0000] | 3.0000 [3.0000, 3.0000] (modeled_games=9) | 100.0% | 100.0% | +0.0000 [+0.0000, +0.0000] | 0/9/0 | 6.5562 | 0.9985 | 26.78 ms | n/a/n/a |
| `weighted_staged_no_artifacts` | 100.0% (9/9) | 100.0% (9/9) | 3.0000 [3.0000, 3.0000] | 3.0000 [3.0000, 3.0000] (modeled_games=9) | 100.0% | 100.0% | +0.0000 [+0.0000, +0.0000] | 0/9/0 | 6.5562 | 0.9985 | 27.69 ms | n/a/n/a |
| `selected_default_disk_artifacts` | 100.0% (9/9) | 100.0% (9/9) | 3.0000 [3.0000, 3.0000] | 3.0000 [3.0000, 3.0000] (modeled_games=9) | 100.0% | 100.0% | +0.0000 [+0.0000, +0.0000] | 0/9/0 | 6.5562 | 0.9985 | 19.65 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F/T | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `previous_release_790ec2d` | 0.0% | 0.0% | 0.0% | 0.0008 [0.0008, 0.0008] | 9/3/0/17/0/0 | 0/7 | 0/0 |
| `uniform_entropy` | 0.0% | 0.0% | 0.0% | 0.0004 [0.0004, 0.0004] | 31/0/0/0/0/0 | 0/10 | 0/0 |
| `cooldown_entropy` | 0.0% | 0.0% | 0.0% | 0.0005 [0.0005, 0.0005] | 29/0/0/0/0/0 | 0/9 | 0/0 |
| `weighted_proxy_only` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 28/0/0/0/0/0 | 0/6 | 0/0 |
| `weighted_proxy_exact_endgame` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 12/0/0/15/0/0 | 0/7 | 0/0 |
| `weighted_staged_no_artifacts` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 9/3/0/15/0/0 | 0/7 | 0/0 |
| `selected_default_disk_artifacts` | 0.0% | 0.0% | 0.0% | 0.0014 [0.0014, 0.0014] | 9/3/0/15/0/0 | 0/7 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `previous_release_790ec2d` | all | 1 | 9/9 | 0.0008 | 7.1629 | 0.9992 |
| `previous_release_790ec2d` | all | 2 | 9/9 | 0.0715 | 3.0110 | 0.9259 |
| `previous_release_790ec2d` | all | 3 | 9/9 | 0.8482 | 0.2243 | 0.1294 |
| `previous_release_790ec2d` | all | 4 | 2/2 | 0.7500 | 0.3466 | 0.2500 |
| `previous_release_790ec2d` | never_used | 1 | 8/8 | 0.0008 | 7.1629 | 0.9992 |
| `previous_release_790ec2d` | never_used | 2 | 8/8 | 0.0495 | 3.2128 | 0.9482 |
| `previous_release_790ec2d` | never_used | 3 | 8/8 | 0.8392 | 0.2420 | 0.1448 |
| `previous_release_790ec2d` | never_used | 4 | 2/2 | 0.7500 | 0.3466 | 0.2500 |
| `previous_release_790ec2d` | reused | 1 | 1/1 | 0.0008 | 7.1630 | 0.9992 |
| `previous_release_790ec2d` | reused | 2 | 1/1 | 0.2474 | 1.3967 | 0.7480 |
| `previous_release_790ec2d` | reused | 3 | 1/1 | 0.9207 | 0.0827 | 0.0069 |
| `uniform_entropy` | all | 1 | 9/9 | 0.0004 | 7.7664 | 0.9996 |
| `uniform_entropy` | all | 2 | 9/9 | 0.0458 | 3.5389 | 0.9542 |
| `uniform_entropy` | all | 3 | 9/9 | 0.6350 | 0.5995 | 0.3491 |
| `uniform_entropy` | all | 4 | 4/4 | 0.9967 | 0.0033 | 0.0001 |
| `uniform_entropy` | never_used | 1 | 8/8 | 0.0004 | 7.7664 | 0.9996 |
| `uniform_entropy` | never_used | 2 | 8/8 | 0.0501 | 3.4216 | 0.9499 |
| `uniform_entropy` | never_used | 3 | 8/8 | 0.6894 | 0.4732 | 0.2927 |
| `uniform_entropy` | never_used | 4 | 3/3 | 0.9956 | 0.0045 | 0.0001 |
| `uniform_entropy` | reused | 1 | 1/1 | 0.0004 | 7.7664 | 0.9996 |
| `uniform_entropy` | reused | 2 | 1/1 | 0.0114 | 4.4773 | 0.9886 |
| `uniform_entropy` | reused | 3 | 1/1 | 0.2000 | 1.6094 | 0.8000 |
| `uniform_entropy` | reused | 4 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `cooldown_entropy` | all | 1 | 9/9 | 0.0005 | 7.5965 | 0.9995 |
| `cooldown_entropy` | all | 2 | 9/9 | 0.0489 | 3.4098 | 0.9510 |
| `cooldown_entropy` | all | 3 | 9/9 | 0.7600 | 0.3307 | 0.2230 |
| `cooldown_entropy` | all | 4 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `cooldown_entropy` | never_used | 1 | 8/8 | 0.0005 | 7.5965 | 0.9995 |
| `cooldown_entropy` | never_used | 2 | 8/8 | 0.0533 | 3.3017 | 0.9467 |
| `cooldown_entropy` | never_used | 3 | 8/8 | 0.7934 | 0.2835 | 0.1884 |
| `cooldown_entropy` | never_used | 4 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `cooldown_entropy` | reused | 1 | 1/1 | 0.0005 | 7.5965 | 0.9995 |
| `cooldown_entropy` | reused | 2 | 1/1 | 0.0139 | 4.2743 | 0.9860 |
| `cooldown_entropy` | reused | 3 | 1/1 | 0.4926 | 0.7080 | 0.5002 |
| `weighted_proxy_only` | all | 1 | 9/9 | 0.0014 | 6.5562 | 0.9985 |
| `weighted_proxy_only` | all | 2 | 9/9 | 0.1360 | 2.3725 | 0.8547 |
| `weighted_proxy_only` | all | 3 | 9/9 | 0.8546 | 0.2155 | 0.1302 |
| `weighted_proxy_only` | all | 4 | 1/1 | 0.5009 | 0.6913 | 0.4982 |
| `weighted_proxy_only` | never_used | 1 | 8/8 | 0.0014 | 6.5532 | 0.9985 |
| `weighted_proxy_only` | never_used | 2 | 8/8 | 0.1129 | 2.5270 | 0.8767 |
| `weighted_proxy_only` | never_used | 3 | 8/8 | 0.8432 | 0.2354 | 0.1461 |
| `weighted_proxy_only` | never_used | 4 | 1/1 | 0.5009 | 0.6913 | 0.4982 |
| `weighted_proxy_only` | reused | 1 | 1/1 | 0.0014 | 6.5799 | 0.9986 |
| `weighted_proxy_only` | reused | 2 | 1/1 | 0.3210 | 1.1364 | 0.6784 |
| `weighted_proxy_only` | reused | 3 | 1/1 | 0.9456 | 0.0559 | 0.0033 |
| `weighted_proxy_exact_endgame` | all | 1 | 9/9 | 0.0014 | 6.5562 | 0.9985 |
| `weighted_proxy_exact_endgame` | all | 2 | 9/9 | 0.1360 | 2.3725 | 0.8547 |
| `weighted_proxy_exact_endgame` | all | 3 | 9/9 | 0.9212 | 0.1011 | 0.0566 |
| `weighted_proxy_exact_endgame` | never_used | 1 | 8/8 | 0.0014 | 6.5532 | 0.9985 |
| `weighted_proxy_exact_endgame` | never_used | 2 | 8/8 | 0.1129 | 2.5270 | 0.8767 |
| `weighted_proxy_exact_endgame` | never_used | 3 | 8/8 | 0.9182 | 0.1068 | 0.0633 |
| `weighted_proxy_exact_endgame` | reused | 1 | 1/1 | 0.0014 | 6.5799 | 0.9986 |
| `weighted_proxy_exact_endgame` | reused | 2 | 1/1 | 0.3210 | 1.1364 | 0.6784 |
| `weighted_proxy_exact_endgame` | reused | 3 | 1/1 | 0.9456 | 0.0559 | 0.0033 |
| `weighted_staged_no_artifacts` | all | 1 | 9/9 | 0.0014 | 6.5562 | 0.9985 |
| `weighted_staged_no_artifacts` | all | 2 | 9/9 | 0.1360 | 2.3725 | 0.8547 |
| `weighted_staged_no_artifacts` | all | 3 | 9/9 | 0.9740 | 0.0265 | 0.0012 |
| `weighted_staged_no_artifacts` | never_used | 1 | 8/8 | 0.0014 | 6.5532 | 0.9985 |
| `weighted_staged_no_artifacts` | never_used | 2 | 8/8 | 0.1129 | 2.5270 | 0.8767 |
| `weighted_staged_no_artifacts` | never_used | 3 | 8/8 | 0.9775 | 0.0228 | 0.0009 |
| `weighted_staged_no_artifacts` | reused | 1 | 1/1 | 0.0014 | 6.5799 | 0.9986 |
| `weighted_staged_no_artifacts` | reused | 2 | 1/1 | 0.3210 | 1.1364 | 0.6784 |
| `weighted_staged_no_artifacts` | reused | 3 | 1/1 | 0.9456 | 0.0559 | 0.0033 |
| `selected_default_disk_artifacts` | all | 1 | 9/9 | 0.0014 | 6.5562 | 0.9985 |
| `selected_default_disk_artifacts` | all | 2 | 9/9 | 0.1360 | 2.3725 | 0.8547 |
| `selected_default_disk_artifacts` | all | 3 | 9/9 | 0.9740 | 0.0265 | 0.0012 |
| `selected_default_disk_artifacts` | never_used | 1 | 8/8 | 0.0014 | 6.5532 | 0.9985 |
| `selected_default_disk_artifacts` | never_used | 2 | 8/8 | 0.1129 | 2.5270 | 0.8767 |
| `selected_default_disk_artifacts` | never_used | 3 | 8/8 | 0.9775 | 0.0228 | 0.0009 |
| `selected_default_disk_artifacts` | reused | 1 | 1/1 | 0.0014 | 6.5799 | 0.9986 |
| `selected_default_disk_artifacts` | reused | 2 | 1/1 | 0.3210 | 1.1364 | 0.6784 |
| `selected_default_disk_artifacts` | reused | 3 | 1/1 | 0.9456 | 0.0559 | 0.0033 |

Reference `selected_default_disk_artifacts` all-game mean sensitivity: penalty 6 = 3.0000 [3.0000, 3.0000]; penalty 7 = 3.0000 [3.0000, 3.0000]; penalty 8 = 3.0000 [3.0000, 3.0000].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
