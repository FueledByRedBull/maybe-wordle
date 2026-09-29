<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2025-07-03` through `2026-08-26` using selection `rolling_folds` (2025-07-03..2025-08-01, 2025-08-02..2025-08-31, 2025-09-01..2025-09-30, 2025-10-01..2025-10-30, 2025-10-31..2025-11-29, 2025-11-30..2025-12-29, 2025-12-30..2026-01-28, 2026-01-29..2026-02-27, 2026-02-28..2026-03-29, 2026-03-30..2026-04-28, 2026-04-29..2026-05-28, 2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 665.15 s; process peak working set: 191.5 MiB; enforced budget: 1200 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `previous_release_790ec2d` | 100.0% (360/360) | 100.0% (360/360) | 3.3306 [3.2667, 3.3972] | 3.3306 [3.2667, 3.3972] | 67.5% | 96.4% | +0.1361 [+0.0861, +0.1834] | 18/276/66 | 7.1652 | 0.9992 | 22.94 ms | n/a/n/a |
| `uniform_entropy` | 100.0% (360/360) | 100.0% (360/360) | 3.5694 [3.5055, 3.6389] | 3.5694 [3.5055, 3.6389] | 47.2% | 94.4% | +0.3750 [+0.3110, +0.4389] | 35/172/153 | 7.7603 | 0.9996 | 22.78 ms | n/a/n/a |
| `cooldown_entropy` | 100.0% (360/360) | 100.0% (360/360) | 3.4889 [3.4250, 3.5556] | 3.4889 [3.4250, 3.5556] | 53.6% | 95.6% | +0.2944 [+0.2250, +0.3639] | 46/172/142 | 7.5911 | 0.9995 | 21.86 ms | n/a/n/a |
| `weighted_proxy_only` | 100.0% (360/360) | 100.0% (360/360) | 3.2444 [3.1750, 3.3250] | 3.2444 [3.1750, 3.3250] | 73.1% | 96.4% | +0.0500 [+0.0167, +0.0833] | 15/313/32 | 6.6703 | 0.9987 | 23.29 ms | n/a/n/a |
| `weighted_proxy_exact_endgame` | 100.0% (360/360) | 100.0% (360/360) | 3.2000 [3.1305, 3.2750] | 3.2000 [3.1305, 3.2750] | 75.3% | 97.2% | +0.0056 [-0.0222, +0.0306] | 14/330/16 | 6.6703 | 0.9987 | 25.85 ms | n/a/n/a |
| `weighted_staged_no_artifacts` | 100.0% (360/360) | 100.0% (360/360) | 3.1944 [3.1194, 3.2722] | 3.1944 [3.1194, 3.2722] | 76.4% | 96.7% | +0.0000 [+0.0000, +0.0000] | 0/360/0 | 6.6703 | 0.9987 | 22.58 ms | n/a/n/a |
| `selected_default_disk_artifacts` | 100.0% (360/360) | 100.0% (360/360) | 3.1944 [3.1194, 3.2722] | 3.1944 [3.1194, 3.2722] | 76.4% | 96.7% | +0.0000 [+0.0000, +0.0000] | 0/360/0 | 6.6703 | 0.9987 | 21.83 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `previous_release_790ec2d` | 0.3% | 0.6% | 0.6% | 0.0022 [0.0008, 0.0083] | 386/141/0/672/0 | 36/329 | 0/0 |
| `uniform_entropy` | 0.0% | 0.0% | 0.3% | 0.0004 [0.0004, 0.0004] | 1285/0/0/0/0 | 42/410 | 0/0 |
| `cooldown_entropy` | 0.0% | 0.0% | 0.3% | 0.0005 [0.0005, 0.0005] | 1256/0/0/0/0 | 41/383 | 0/0 |
| `weighted_proxy_only` | 0.3% | 0.6% | 0.6% | 0.0017 [0.0013, 0.0077] | 1168/0/0/0/0 | 38/290 | 0/0 |
| `weighted_proxy_exact_endgame` | 0.3% | 0.6% | 0.6% | 0.0017 [0.0013, 0.0077] | 529/0/0/623/0 | 36/280 | 0/0 |
| `weighted_staged_no_artifacts` | 0.3% | 0.6% | 0.6% | 0.0017 [0.0013, 0.0077] | 386/139/0/625/0 | 36/292 | 0/0 |
| `selected_default_disk_artifacts` | 0.3% | 0.6% | 0.6% | 0.0017 [0.0013, 0.0077] | 386/139/0/625/0 | 36/292 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `previous_release_790ec2d` | all | 1 | 332/360 | 0.0008 | 7.1652 | 0.9992 |
| `previous_release_790ec2d` | all | 2 | 334/360 | 0.0779 | 3.2928 | 0.9186 |
| `previous_release_790ec2d` | all | 3 | 338/345 | 0.6187 | 0.8341 | 0.3816 |
| `previous_release_790ec2d` | all | 4 | 116/117 | 0.8715 | 0.2854 | 0.1315 |
| `previous_release_790ec2d` | all | 5 | 13/13 | 0.7260 | 0.6247 | 0.3401 |
| `previous_release_790ec2d` | all | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `previous_release_790ec2d` | never_used | 1 | 315/343 | 0.0008 | 7.1510 | 0.9992 |
| `previous_release_790ec2d` | never_used | 2 | 317/343 | 0.0786 | 3.2788 | 0.9178 |
| `previous_release_790ec2d` | never_used | 3 | 321/328 | 0.6244 | 0.8202 | 0.3769 |
| `previous_release_790ec2d` | never_used | 4 | 108/109 | 0.8649 | 0.3034 | 0.1409 |
| `previous_release_790ec2d` | never_used | 5 | 13/13 | 0.7260 | 0.6247 | 0.3401 |
| `previous_release_790ec2d` | never_used | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `previous_release_790ec2d` | reused | 1 | 17/17 | 0.0007 | 7.4286 | 0.9993 |
| `previous_release_790ec2d` | reused | 2 | 17/17 | 0.0638 | 3.5545 | 0.9339 |
| `previous_release_790ec2d` | reused | 3 | 17/17 | 0.5111 | 1.0958 | 0.4707 |
| `previous_release_790ec2d` | reused | 4 | 8/8 | 0.9600 | 0.0424 | 0.0047 |
| `previous_release_790ec2d` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `previous_release_790ec2d` | out_of_core | 2 | 2/28 | 0.0177 | 4.5486 | 1.2124 |
| `previous_release_790ec2d` | out_of_core | 3 | 21/28 | 0.1657 | 3.8138 | 1.2385 |
| `previous_release_790ec2d` | out_of_core | 4 | 25/26 | 0.5818 | 1.1005 | 0.5158 |
| `previous_release_790ec2d` | out_of_core | 5 | 11/11 | 0.6850 | 0.7291 | 0.4014 |
| `previous_release_790ec2d` | out_of_core | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `uniform_entropy` | all | 1 | 332/360 | 0.0004 | 7.7603 | 0.9996 |
| `uniform_entropy` | all | 2 | 335/360 | 0.0498 | 3.7361 | 0.9529 |
| `uniform_entropy` | all | 3 | 346/351 | 0.4985 | 1.1149 | 0.5138 |
| `uniform_entropy` | all | 4 | 189/190 | 0.8893 | 0.1861 | 0.1015 |
| `uniform_entropy` | all | 5 | 20/20 | 0.8809 | 0.3644 | 0.1355 |
| `uniform_entropy` | all | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `uniform_entropy` | never_used | 1 | 315/343 | 0.0004 | 7.7601 | 0.9996 |
| `uniform_entropy` | never_used | 2 | 318/343 | 0.0470 | 3.7387 | 0.9561 |
| `uniform_entropy` | never_used | 3 | 331/336 | 0.5021 | 1.1140 | 0.5116 |
| `uniform_entropy` | never_used | 4 | 180/181 | 0.8899 | 0.1872 | 0.1010 |
| `uniform_entropy` | never_used | 5 | 20/20 | 0.8809 | 0.3644 | 0.1355 |
| `uniform_entropy` | never_used | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `uniform_entropy` | reused | 1 | 17/17 | 0.0004 | 7.7638 | 0.9996 |
| `uniform_entropy` | reused | 2 | 17/17 | 0.1025 | 3.6863 | 0.8929 |
| `uniform_entropy` | reused | 3 | 15/15 | 0.4188 | 1.1353 | 0.5644 |
| `uniform_entropy` | reused | 4 | 9/9 | 0.8788 | 0.1643 | 0.1115 |
| `uniform_entropy` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `uniform_entropy` | out_of_core | 2 | 3/28 | 0.0059 | 5.3324 | 1.3853 |
| `uniform_entropy` | out_of_core | 3 | 23/28 | 0.1828 | 3.6252 | 1.2329 |
| `uniform_entropy` | out_of_core | 4 | 24/25 | 0.6697 | 0.6723 | 0.3674 |
| `uniform_entropy` | out_of_core | 5 | 11/11 | 0.8415 | 0.5854 | 0.2000 |
| `uniform_entropy` | out_of_core | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `cooldown_entropy` | all | 1 | 332/360 | 0.0005 | 7.5911 | 0.9995 |
| `cooldown_entropy` | all | 2 | 335/360 | 0.0590 | 3.5784 | 0.9433 |
| `cooldown_entropy` | all | 3 | 346/351 | 0.5472 | 0.9868 | 0.4596 |
| `cooldown_entropy` | all | 4 | 167/167 | 0.8828 | 0.2295 | 0.1135 |
| `cooldown_entropy` | all | 5 | 16/16 | 0.9518 | 0.0880 | 0.0469 |
| `cooldown_entropy` | all | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `cooldown_entropy` | never_used | 1 | 315/343 | 0.0005 | 7.5910 | 0.9995 |
| `cooldown_entropy` | never_used | 2 | 318/343 | 0.0557 | 3.5967 | 0.9470 |
| `cooldown_entropy` | never_used | 3 | 331/336 | 0.5494 | 0.9879 | 0.4586 |
| `cooldown_entropy` | never_used | 4 | 161/161 | 0.8819 | 0.2334 | 0.1146 |
| `cooldown_entropy` | never_used | 5 | 16/16 | 0.9518 | 0.0880 | 0.0469 |
| `cooldown_entropy` | never_used | 6 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `cooldown_entropy` | reused | 1 | 17/17 | 0.0005 | 7.5944 | 0.9995 |
| `cooldown_entropy` | reused | 2 | 17/17 | 0.1214 | 3.2357 | 0.8745 |
| `cooldown_entropy` | reused | 3 | 15/15 | 0.4989 | 0.9637 | 0.4819 |
| `cooldown_entropy` | reused | 4 | 6/6 | 0.9077 | 0.1246 | 0.0836 |
| `cooldown_entropy` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `cooldown_entropy` | out_of_core | 2 | 3/28 | 0.0051 | 5.4884 | 1.4004 |
| `cooldown_entropy` | out_of_core | 3 | 23/28 | 0.2068 | 3.5512 | 1.1435 |
| `cooldown_entropy` | out_of_core | 4 | 24/24 | 0.5381 | 1.1050 | 0.5344 |
| `cooldown_entropy` | out_of_core | 5 | 12/12 | 0.9375 | 0.1155 | 0.0625 |
| `cooldown_entropy` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_proxy_only` | all | 1 | 332/360 | 0.0013 | 6.6703 | 0.9987 |
| `weighted_proxy_only` | all | 2 | 334/360 | 0.1201 | 2.8215 | 0.8792 |
| `weighted_proxy_only` | all | 3 | 325/333 | 0.7163 | 0.6076 | 0.2788 |
| `weighted_proxy_only` | all | 4 | 96/97 | 0.8678 | 0.2575 | 0.1246 |
| `weighted_proxy_only` | all | 5 | 13/13 | 0.6680 | 0.5473 | 0.3507 |
| `weighted_proxy_only` | all | 6 | 5/5 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_proxy_only` | never_used | 1 | 315/343 | 0.0013 | 6.6155 | 0.9986 |
| `weighted_proxy_only` | never_used | 2 | 317/343 | 0.1219 | 2.7757 | 0.8748 |
| `weighted_proxy_only` | never_used | 3 | 308/316 | 0.7225 | 0.5762 | 0.2707 |
| `weighted_proxy_only` | never_used | 4 | 90/91 | 0.8664 | 0.2647 | 0.1286 |
| `weighted_proxy_only` | never_used | 5 | 13/13 | 0.6680 | 0.5473 | 0.3507 |
| `weighted_proxy_only` | never_used | 6 | 5/5 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_proxy_only` | reused | 1 | 17/17 | 0.0008 | 7.6872 | 0.9998 |
| `weighted_proxy_only` | reused | 2 | 17/17 | 0.0855 | 3.6756 | 0.9605 |
| `weighted_proxy_only` | reused | 3 | 17/17 | 0.6039 | 1.1757 | 0.4264 |
| `weighted_proxy_only` | reused | 4 | 6/6 | 0.8892 | 0.1502 | 0.0655 |
| `weighted_proxy_only` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `weighted_proxy_only` | out_of_core | 2 | 2/28 | 0.0170 | 4.6022 | 1.3337 |
| `weighted_proxy_only` | out_of_core | 3 | 20/28 | 0.1523 | 3.4711 | 1.0978 |
| `weighted_proxy_only` | out_of_core | 4 | 24/25 | 0.6044 | 0.8615 | 0.4171 |
| `weighted_proxy_only` | out_of_core | 5 | 11/11 | 0.6129 | 0.6414 | 0.4142 |
| `weighted_proxy_only` | out_of_core | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_proxy_exact_endgame` | all | 1 | 332/360 | 0.0013 | 6.6703 | 0.9987 |
| `weighted_proxy_exact_endgame` | all | 2 | 334/360 | 0.1201 | 2.8215 | 0.8792 |
| `weighted_proxy_exact_endgame` | all | 3 | 321/329 | 0.7279 | 0.5867 | 0.2676 |
| `weighted_proxy_exact_endgame` | all | 4 | 88/89 | 0.8747 | 0.2702 | 0.1171 |
| `weighted_proxy_exact_endgame` | all | 5 | 10/10 | 0.5738 | 0.7077 | 0.4569 |
| `weighted_proxy_exact_endgame` | all | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_proxy_exact_endgame` | never_used | 1 | 315/343 | 0.0013 | 6.6155 | 0.9986 |
| `weighted_proxy_exact_endgame` | never_used | 2 | 317/343 | 0.1219 | 2.7757 | 0.8748 |
| `weighted_proxy_exact_endgame` | never_used | 3 | 304/312 | 0.7332 | 0.5571 | 0.2601 |
| `weighted_proxy_exact_endgame` | never_used | 4 | 82/83 | 0.8736 | 0.2789 | 0.1208 |
| `weighted_proxy_exact_endgame` | never_used | 5 | 10/10 | 0.5738 | 0.7077 | 0.4569 |
| `weighted_proxy_exact_endgame` | never_used | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_proxy_exact_endgame` | reused | 1 | 17/17 | 0.0008 | 7.6872 | 0.9998 |
| `weighted_proxy_exact_endgame` | reused | 2 | 17/17 | 0.0855 | 3.6756 | 0.9605 |
| `weighted_proxy_exact_endgame` | reused | 3 | 17/17 | 0.6323 | 1.1157 | 0.4019 |
| `weighted_proxy_exact_endgame` | reused | 4 | 6/6 | 0.8892 | 0.1502 | 0.0655 |
| `weighted_proxy_exact_endgame` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `weighted_proxy_exact_endgame` | out_of_core | 2 | 2/28 | 0.0170 | 4.6022 | 1.3337 |
| `weighted_proxy_exact_endgame` | out_of_core | 3 | 20/28 | 0.2010 | 3.3243 | 1.0559 |
| `weighted_proxy_exact_endgame` | out_of_core | 4 | 23/24 | 0.5971 | 0.9450 | 0.4271 |
| `weighted_proxy_exact_endgame` | out_of_core | 5 | 10/10 | 0.5738 | 0.7077 | 0.4569 |
| `weighted_proxy_exact_endgame` | out_of_core | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_staged_no_artifacts` | all | 1 | 332/360 | 0.0013 | 6.6703 | 0.9987 |
| `weighted_staged_no_artifacts` | all | 2 | 334/360 | 0.1201 | 2.8215 | 0.8792 |
| `weighted_staged_no_artifacts` | all | 3 | 320/329 | 0.7421 | 0.5525 | 0.2577 |
| `weighted_staged_no_artifacts` | all | 4 | 84/85 | 0.8448 | 0.3250 | 0.1448 |
| `weighted_staged_no_artifacts` | all | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `weighted_staged_no_artifacts` | all | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_staged_no_artifacts` | never_used | 1 | 315/343 | 0.0013 | 6.6155 | 0.9986 |
| `weighted_staged_no_artifacts` | never_used | 2 | 317/343 | 0.1219 | 2.7757 | 0.8748 |
| `weighted_staged_no_artifacts` | never_used | 3 | 303/312 | 0.7527 | 0.5150 | 0.2428 |
| `weighted_staged_no_artifacts` | never_used | 4 | 76/77 | 0.8403 | 0.3450 | 0.1540 |
| `weighted_staged_no_artifacts` | never_used | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `weighted_staged_no_artifacts` | never_used | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `weighted_staged_no_artifacts` | reused | 1 | 17/17 | 0.0008 | 7.6872 | 0.9998 |
| `weighted_staged_no_artifacts` | reused | 2 | 17/17 | 0.0855 | 3.6756 | 0.9605 |
| `weighted_staged_no_artifacts` | reused | 3 | 17/17 | 0.5519 | 1.2222 | 0.5237 |
| `weighted_staged_no_artifacts` | reused | 4 | 8/8 | 0.8881 | 0.1358 | 0.0569 |
| `weighted_staged_no_artifacts` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `weighted_staged_no_artifacts` | out_of_core | 2 | 2/28 | 0.0170 | 4.6022 | 1.3337 |
| `weighted_staged_no_artifacts` | out_of_core | 3 | 19/28 | 0.2065 | 3.1306 | 1.0928 |
| `weighted_staged_no_artifacts` | out_of_core | 4 | 23/24 | 0.5470 | 1.0525 | 0.4781 |
| `weighted_staged_no_artifacts` | out_of_core | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `weighted_staged_no_artifacts` | out_of_core | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `selected_default_disk_artifacts` | all | 1 | 332/360 | 0.0013 | 6.6703 | 0.9987 |
| `selected_default_disk_artifacts` | all | 2 | 334/360 | 0.1201 | 2.8215 | 0.8792 |
| `selected_default_disk_artifacts` | all | 3 | 320/329 | 0.7421 | 0.5525 | 0.2577 |
| `selected_default_disk_artifacts` | all | 4 | 84/85 | 0.8448 | 0.3250 | 0.1448 |
| `selected_default_disk_artifacts` | all | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `selected_default_disk_artifacts` | all | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `selected_default_disk_artifacts` | never_used | 1 | 315/343 | 0.0013 | 6.6155 | 0.9986 |
| `selected_default_disk_artifacts` | never_used | 2 | 317/343 | 0.1219 | 2.7757 | 0.8748 |
| `selected_default_disk_artifacts` | never_used | 3 | 303/312 | 0.7527 | 0.5150 | 0.2428 |
| `selected_default_disk_artifacts` | never_used | 4 | 76/77 | 0.8403 | 0.3450 | 0.1540 |
| `selected_default_disk_artifacts` | never_used | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `selected_default_disk_artifacts` | never_used | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `selected_default_disk_artifacts` | reused | 1 | 17/17 | 0.0008 | 7.6872 | 0.9998 |
| `selected_default_disk_artifacts` | reused | 2 | 17/17 | 0.0855 | 3.6756 | 0.9605 |
| `selected_default_disk_artifacts` | reused | 3 | 17/17 | 0.5519 | 1.2222 | 0.5237 |
| `selected_default_disk_artifacts` | reused | 4 | 8/8 | 0.8881 | 0.1358 | 0.0569 |
| `selected_default_disk_artifacts` | out_of_core | 1 | 0/28 | n/a | n/a | n/a |
| `selected_default_disk_artifacts` | out_of_core | 2 | 2/28 | 0.0170 | 4.6022 | 1.3337 |
| `selected_default_disk_artifacts` | out_of_core | 3 | 19/28 | 0.2065 | 3.1306 | 1.0928 |
| `selected_default_disk_artifacts` | out_of_core | 4 | 23/24 | 0.5470 | 1.0525 | 0.4781 |
| `selected_default_disk_artifacts` | out_of_core | 5 | 12/12 | 0.6448 | 0.5897 | 0.3808 |
| `selected_default_disk_artifacts` | out_of_core | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |

Reference `selected_default_disk_artifacts` all-game mean sensitivity: penalty 6 = 3.1944 [3.1194, 3.2722]; penalty 7 = 3.1944 [3.1194, 3.2722]; penalty 8 = 3.1944 [3.1194, 3.2722].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
