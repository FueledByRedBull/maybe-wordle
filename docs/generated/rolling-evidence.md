<!-- BEGIN GENERATED ROLLING EVIDENCE -->
### Rolling-origin promotion guard

Across 12 non-overlapping development folds (360 scheduled games), the sealed test was **not** evaluated. Coverage gaps and six-guess failures are hard constraints before mean score.

| Configuration | Solved | All-game mean | Delta vs baseline | W/T/L | Latency p95 | Guard decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `audit_selected_staged` | 360/360 | 3.1944 [3.1194, 3.2722] | reference | -- | 28.29 ms | retained |
| `audit_v19b_staged` | 360/360 | 3.1917 [3.1194, 3.2694] | -0.0028 [-0.0222, +0.0167] | 6/347/7 | 27.25 ms | not promoted: improvement uncertain |

| Configuration | Prior top-1/3/5 | Confidence ECE | Search steps P/L/XE/X/F/T | Recovery/fallback steps |
| --- | ---: | ---: | ---: | ---: |
| `audit_selected_staged` | 0.3%/0.6%/0.6% | 0.0017 [0.0013, 0.0077] | 386/139/0/609/0/16 | 36/292 |
| `audit_v19b_staged` | 0.3%/0.6%/0.6% | 0.0017 [0.0013, 0.0077] | 386/131/0/618/0/14 | 36/297 |

Development decisions:

- `audit_v19b_staged` is retained as a development finalist, not promoted, because the observed improvement's paired interval includes zero.

This development comparison did not access the sealed window and does not establish prospective performance. Any later sealed evaluation requires separate evidence.
<!-- END GENERATED ROLLING EVIDENCE -->
