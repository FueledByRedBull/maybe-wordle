<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->
## Predictive solver evidence

Development-only diagnostic for `2025-07-03` through `2026-08-26` using selection `rolling_folds` (2025-07-03..2025-08-01, 2025-08-02..2025-08-31, 2025-09-01..2025-09-30, 2025-10-01..2025-10-30, 2025-10-31..2025-11-29, 2025-11-30..2025-12-29, 2025-12-30..2026-01-28, 2026-01-29..2026-02-27, 2026-02-28..2026-03-29, 2026-03-30..2026-04-28, 2026-04-29..2026-05-28, 2026-07-28..2026-08-26) and history through `2026-08-26`. The sealed test was **not** evaluated.

Measured generation compute time: 1124.76 s; process peak working set: 87.8 MiB; enforced budget: 1800 s / 4096 MiB.

| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic_prior` | 100.0% (360/360) | 100.0% (360/360) | 3.5083 [3.4444, 3.5778] | 3.5083 [3.4444, 3.5778] | 51.4% | 95.6% | +0.0000 [+0.0000, +0.0000] | 0/360/0 | 7.1596 | 0.9989 | 262.28 ms | n/a/n/a |
| `uniform_prior` | 100.0% (360/360) | 100.0% (360/360) | 3.7778 [3.7056, 3.8500] | 3.7778 [3.7056, 3.8500] | 35.0% | 87.8% | +0.2694 [+0.1944, +0.3417] | 59/164/137 | 8.1648 | 0.9996 | 262.28 ms | n/a/n/a |
| `used_unused_prior` | 100.0% (360/360) | 100.0% (360/360) | 3.5194 [3.4500, 3.5972] | 3.5194 [3.4500, 3.5972] | 50.6% | 95.3% | +0.0111 [-0.0361, +0.0583] | 27/304/29 | 7.2677 | 0.9989 | 262.14 ms | n/a/n/a |
| `recency_buckets_prior` | 100.0% (360/360) | 100.0% (360/360) | 3.6417 [3.5667, 3.7222] | 3.6417 [3.5667, 3.7222] | 43.6% | 90.8% | +0.1333 [+0.0610, +0.2056] | 68/182/110 | 7.6615 | 0.9993 | 263.48 ms | n/a/n/a |
| `regularized_frequency_prior` | 100.0% (360/360) | 99.7% (359/360) | 3.9444 [3.8722, 4.0167] | 3.9417 [3.8694, 4.0111] | 26.4% | 81.7% | +0.4361 [+0.3667, +0.5056] | 30/170/160 | 8.6248 | 0.9999 | 262.42 ms | n/a/n/a |

Session-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.

Measured artifact sizes: `pattern_table` = 35132187 bytes; `answer_history` = 64472 bytes; `modeled_answers` = 177428 bytes; `predictive_books` = 812321 bytes.

| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic_prior` | 0.3% | 0.6% | 0.6% | 0.0015 [0.0012, 0.0071] | 0/0/0/0/1263 | 0/1263 | 0/0 |
| `uniform_prior` | 0.0% | 0.0% | 0.3% | 0.0004 [0.0004, 0.0004] | 0/0/0/0/1360 | 0/1360 | 0/0 |
| `used_unused_prior` | 0.3% | 0.6% | 0.6% | 0.0014 [0.0013, 0.0070] | 0/0/0/0/1267 | 0/1267 | 0/0 |
| `recency_buckets_prior` | 0.0% | 0.0% | 0.6% | 0.0007 [0.0007, 0.0007] | 0/0/0/0/1311 | 0/1311 | 0/0 |
| `regularized_frequency_prior` | 0.0% | 0.0% | 0.0% | 0.0006 [0.0005, 0.0006] | 0/0/0/0/1419 | 0/1419 | 0/0 |

Post-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):

| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `logistic_prior` | all | 1 | 360/360 | 0.0011 | 7.1596 | 0.9989 |
| `logistic_prior` | all | 2 | 360/360 | 0.0883 | 3.6018 | 0.9079 |
| `logistic_prior` | all | 3 | 351/351 | 0.5946 | 0.9580 | 0.3990 |
| `logistic_prior` | all | 4 | 175/175 | 0.9106 | 0.1758 | 0.0876 |
| `logistic_prior` | all | 5 | 16/16 | 0.8854 | 0.1986 | 0.1146 |
| `logistic_prior` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `logistic_prior` | never_used | 1 | 343/343 | 0.0012 | 7.1300 | 0.9988 |
| `logistic_prior` | never_used | 2 | 343/343 | 0.0896 | 3.5781 | 0.9055 |
| `logistic_prior` | never_used | 3 | 334/334 | 0.5939 | 0.9679 | 0.3983 |
| `logistic_prior` | never_used | 4 | 169/169 | 0.9084 | 0.1811 | 0.0906 |
| `logistic_prior` | never_used | 5 | 16/16 | 0.8854 | 0.1986 | 0.1146 |
| `logistic_prior` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `logistic_prior` | reused | 1 | 17/17 | 0.0007 | 7.7560 | 0.9998 |
| `logistic_prior` | reused | 2 | 17/17 | 0.0634 | 4.0803 | 0.9571 |
| `logistic_prior` | reused | 3 | 17/17 | 0.6090 | 0.7630 | 0.4129 |
| `logistic_prior` | reused | 4 | 6/6 | 0.9737 | 0.0271 | 0.0027 |
| `logistic_prior` | out_of_core | 1 | 28/28 | 0.0000 | 12.1455 | 1.0011 |
| `logistic_prior` | out_of_core | 2 | 28/28 | 0.0038 | 8.6720 | 1.0345 |
| `logistic_prior` | out_of_core | 3 | 28/28 | 0.2036 | 4.0365 | 0.9868 |
| `logistic_prior` | out_of_core | 4 | 25/25 | 0.6498 | 0.8547 | 0.3973 |
| `logistic_prior` | out_of_core | 5 | 10/10 | 0.8167 | 0.3178 | 0.1833 |
| `logistic_prior` | out_of_core | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `uniform_prior` | all | 1 | 360/360 | 0.0004 | 8.1648 | 0.9996 |
| `uniform_prior` | all | 2 | 360/360 | 0.0278 | 4.7810 | 0.9727 |
| `uniform_prior` | all | 3 | 354/354 | 0.3719 | 1.6992 | 0.6307 |
| `uniform_prior` | all | 4 | 234/234 | 0.8323 | 0.3925 | 0.1884 |
| `uniform_prior` | all | 5 | 44/44 | 0.8464 | 0.2283 | 0.1516 |
| `uniform_prior` | all | 6 | 8/8 | 1.0000 | -0.0000 | 0.0000 |
| `uniform_prior` | never_used | 1 | 343/343 | 0.0004 | 8.1812 | 0.9996 |
| `uniform_prior` | never_used | 2 | 343/343 | 0.0285 | 4.7798 | 0.9720 |
| `uniform_prior` | never_used | 3 | 337/337 | 0.3794 | 1.6866 | 0.6236 |
| `uniform_prior` | never_used | 4 | 221/221 | 0.8344 | 0.3962 | 0.1883 |
| `uniform_prior` | never_used | 5 | 42/42 | 0.8391 | 0.2391 | 0.1588 |
| `uniform_prior` | never_used | 6 | 8/8 | 1.0000 | -0.0000 | 0.0000 |
| `uniform_prior` | reused | 1 | 17/17 | 0.0004 | 7.8326 | 0.9996 |
| `uniform_prior` | reused | 2 | 17/17 | 0.0124 | 4.8053 | 0.9870 |
| `uniform_prior` | reused | 3 | 17/17 | 0.2220 | 1.9499 | 0.7710 |
| `uniform_prior` | reused | 4 | 13/13 | 0.7980 | 0.3308 | 0.1902 |
| `uniform_prior` | reused | 5 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `uniform_prior` | out_of_core | 1 | 28/28 | 0.0000 | 12.1455 | 1.0004 |
| `uniform_prior` | out_of_core | 2 | 28/28 | 0.0004 | 8.9158 | 1.0255 |
| `uniform_prior` | out_of_core | 3 | 28/28 | 0.0461 | 5.1692 | 1.1436 |
| `uniform_prior` | out_of_core | 4 | 28/28 | 0.4401 | 1.9638 | 0.8059 |
| `uniform_prior` | out_of_core | 5 | 18/18 | 0.7222 | 0.4142 | 0.2778 |
| `uniform_prior` | out_of_core | 6 | 5/5 | 1.0000 | -0.0000 | 0.0000 |
| `used_unused_prior` | all | 1 | 360/360 | 0.0012 | 7.2677 | 0.9989 |
| `used_unused_prior` | all | 2 | 360/360 | 0.0938 | 3.6913 | 0.9083 |
| `used_unused_prior` | all | 3 | 351/351 | 0.5815 | 1.0539 | 0.4258 |
| `used_unused_prior` | all | 4 | 178/178 | 0.8935 | 0.1911 | 0.1036 |
| `used_unused_prior` | all | 5 | 17/17 | 0.9700 | 0.0414 | 0.0294 |
| `used_unused_prior` | all | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `used_unused_prior` | never_used | 1 | 343/343 | 0.0012 | 7.0796 | 0.9988 |
| `used_unused_prior` | never_used | 2 | 343/343 | 0.0984 | 3.5147 | 0.8973 |
| `used_unused_prior` | never_used | 3 | 334/334 | 0.5943 | 0.9847 | 0.4058 |
| `used_unused_prior` | never_used | 4 | 169/169 | 0.9124 | 0.1627 | 0.0867 |
| `used_unused_prior` | never_used | 5 | 15/15 | 0.9660 | 0.0469 | 0.0333 |
| `used_unused_prior` | never_used | 6 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `used_unused_prior` | reused | 1 | 17/17 | 0.0000 | 11.0633 | 1.0014 |
| `used_unused_prior` | reused | 2 | 17/17 | 0.0016 | 7.2531 | 1.1295 |
| `used_unused_prior` | reused | 3 | 17/17 | 0.3300 | 2.4125 | 0.8177 |
| `used_unused_prior` | reused | 4 | 9/9 | 0.5377 | 0.7241 | 0.4212 |
| `used_unused_prior` | reused | 5 | 2/2 | 1.0000 | -0.0000 | 0.0000 |
| `used_unused_prior` | out_of_core | 1 | 28/28 | 0.0000 | 12.1455 | 1.0013 |
| `used_unused_prior` | out_of_core | 2 | 28/28 | 0.0038 | 8.6745 | 1.0391 |
| `used_unused_prior` | out_of_core | 3 | 28/28 | 0.1386 | 4.4462 | 1.1152 |
| `used_unused_prior` | out_of_core | 4 | 26/26 | 0.6473 | 0.7510 | 0.3889 |
| `used_unused_prior` | out_of_core | 5 | 10/10 | 0.9500 | 0.0693 | 0.0500 |
| `recency_buckets_prior` | all | 1 | 360/360 | 0.0006 | 7.6615 | 0.9993 |
| `recency_buckets_prior` | all | 2 | 360/360 | 0.0485 | 4.4530 | 0.9440 |
| `recency_buckets_prior` | all | 3 | 350/350 | 0.4167 | 1.5111 | 0.5625 |
| `recency_buckets_prior` | all | 4 | 203/203 | 0.8141 | 0.4008 | 0.1862 |
| `recency_buckets_prior` | all | 5 | 33/33 | 0.8986 | 0.1594 | 0.0961 |
| `recency_buckets_prior` | all | 6 | 5/5 | 1.0000 | -0.0000 | 0.0000 |
| `recency_buckets_prior` | never_used | 1 | 343/343 | 0.0006 | 7.6743 | 0.9993 |
| `recency_buckets_prior` | never_used | 2 | 343/343 | 0.0473 | 4.4592 | 0.9450 |
| `recency_buckets_prior` | never_used | 3 | 334/334 | 0.4185 | 1.5174 | 0.5610 |
| `recency_buckets_prior` | never_used | 4 | 192/192 | 0.8075 | 0.4184 | 0.1941 |
| `recency_buckets_prior` | never_used | 5 | 33/33 | 0.8986 | 0.1594 | 0.0961 |
| `recency_buckets_prior` | never_used | 6 | 5/5 | 1.0000 | -0.0000 | 0.0000 |
| `recency_buckets_prior` | reused | 1 | 17/17 | 0.0006 | 7.4032 | 0.9992 |
| `recency_buckets_prior` | reused | 2 | 17/17 | 0.0722 | 4.3284 | 0.9243 |
| `recency_buckets_prior` | reused | 3 | 16/16 | 0.3783 | 1.3804 | 0.5942 |
| `recency_buckets_prior` | reused | 4 | 11/11 | 0.9281 | 0.0922 | 0.0489 |
| `recency_buckets_prior` | out_of_core | 1 | 28/28 | 0.0000 | 12.1455 | 1.0005 |
| `recency_buckets_prior` | out_of_core | 2 | 28/28 | 0.0003 | 8.8660 | 1.0288 |
| `recency_buckets_prior` | out_of_core | 3 | 28/28 | 0.1378 | 4.7328 | 1.0902 |
| `recency_buckets_prior` | out_of_core | 4 | 25/25 | 0.4254 | 1.9246 | 0.7749 |
| `recency_buckets_prior` | out_of_core | 5 | 16/16 | 0.8073 | 0.3106 | 0.1927 |
| `recency_buckets_prior` | out_of_core | 6 | 4/4 | 1.0000 | -0.0000 | 0.0000 |
| `regularized_frequency_prior` | all | 1 | 360/360 | 0.0002 | 8.6248 | 0.9999 |
| `regularized_frequency_prior` | all | 2 | 360/360 | 0.0259 | 4.9090 | 0.9876 |
| `regularized_frequency_prior` | all | 3 | 358/358 | 0.3312 | 1.8342 | 0.7237 |
| `regularized_frequency_prior` | all | 4 | 265/265 | 0.8039 | 0.3957 | 0.2172 |
| `regularized_frequency_prior` | all | 5 | 66/66 | 0.8923 | 0.2051 | 0.1289 |
| `regularized_frequency_prior` | all | 6 | 10/10 | 0.9500 | 0.0693 | 0.0500 |
| `regularized_frequency_prior` | never_used | 1 | 343/343 | 0.0002 | 8.6705 | 1.0000 |
| `regularized_frequency_prior` | never_used | 2 | 343/343 | 0.0253 | 4.9556 | 0.9891 |
| `regularized_frequency_prior` | never_used | 3 | 341/341 | 0.3231 | 1.8752 | 0.7362 |
| `regularized_frequency_prior` | never_used | 4 | 256/256 | 0.8029 | 0.4014 | 0.2198 |
| `regularized_frequency_prior` | never_used | 5 | 65/65 | 0.8906 | 0.2083 | 0.1309 |
| `regularized_frequency_prior` | never_used | 6 | 10/10 | 0.9500 | 0.0693 | 0.0500 |
| `regularized_frequency_prior` | reused | 1 | 17/17 | 0.0005 | 7.7032 | 0.9995 |
| `regularized_frequency_prior` | reused | 2 | 17/17 | 0.0379 | 3.9677 | 0.9573 |
| `regularized_frequency_prior` | reused | 3 | 17/17 | 0.4943 | 1.0106 | 0.4717 |
| `regularized_frequency_prior` | reused | 4 | 9/9 | 0.8343 | 0.2322 | 0.1435 |
| `regularized_frequency_prior` | reused | 5 | 1/1 | 1.0000 | -0.0000 | 0.0000 |
| `regularized_frequency_prior` | out_of_core | 1 | 28/28 | 0.0000 | 12.1455 | 1.0004 |
| `regularized_frequency_prior` | out_of_core | 2 | 28/28 | 0.0006 | 8.6773 | 1.0401 |
| `regularized_frequency_prior` | out_of_core | 3 | 28/28 | 0.2120 | 4.3038 | 0.9718 |
| `regularized_frequency_prior` | out_of_core | 4 | 23/23 | 0.5727 | 1.6061 | 0.6066 |
| `regularized_frequency_prior` | out_of_core | 5 | 12/12 | 0.7655 | 0.5715 | 0.2959 |
| `regularized_frequency_prior` | out_of_core | 6 | 3/3 | 0.8333 | 0.2310 | 0.1667 |

Reference `logistic_prior` all-game mean sensitivity: penalty 6 = 3.5083 [3.4444, 3.5778]; penalty 7 = 3.5083 [3.4444, 3.5778]; penalty 8 = 3.5083 [3.4444, 3.5778].

The old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.

The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.
<!-- END GENERATED PREDICTIVE EVIDENCE -->
