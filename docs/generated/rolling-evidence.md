<!-- BEGIN GENERATED ROLLING EVIDENCE -->
### Rolling-origin promotion guard

Across 12 non-overlapping development folds (360 scheduled games), the sealed test was **not** evaluated. Coverage gaps and six-guess failures are hard constraints before mean score.

| Configuration | Solved | All-game mean | Delta vs baseline | W/T/L | Latency p95 | Guard decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `finite_baseline_250ms` | 354/360 | 3.5389 [3.4472, 3.6333] | reference | -- | 29.79 ms | retained |
| `finite_preordered_250ms` | 360/360 | 3.5833 [3.5055, 3.6694] | +0.0444 [-0.0639, +0.1556] | 95/156/109 | 263.15 ms | rejected: no solve-quality gain |

| Configuration | Prior top-1/3/5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps |
| --- | ---: | ---: | ---: | ---: |
| `finite_baseline_250ms` | 0.3%/0.6%/0.6% | 0.0015 [0.0012, 0.0071] | 0/0/0/0/1268 | 0/1268 |
| `finite_preordered_250ms` | 0.3%/0.6%/0.6% | 0.0015 [0.0012, 0.0071] | 0/0/0/0/1290 | 0/1290 |

Development decisions:

- `finite_preordered_250ms` is rejected because it did not improve solve quality.

This development comparison did not access the sealed window and does not establish prospective performance. Any later sealed evaluation requires separate evidence.
<!-- END GENERATED ROLLING EVIDENCE -->
