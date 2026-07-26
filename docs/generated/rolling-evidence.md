<!-- BEGIN GENERATED ROLLING EVIDENCE -->
### Rolling-origin promotion guard

Across 12 non-overlapping development folds (360 scheduled games), the sealed test was **not** evaluated. Coverage gaps and six-guess failures are hard constraints before mean score.

| Configuration | Solved | All-game mean | Delta vs default | W/T/L | Latency p95 | Guard decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `current_default` | 358/360 | 3.3000 [3.2306, 3.3778] | reference | -- | 55.61 ms | retained |
| `selected-predictive-v20` | 360/360 | 3.1778 [3.1056, 3.2556] | -0.1222 [-0.1722, -0.0722] | 69/265/26 | 41.60 ms | eligible on solve quality |

| Configuration | Prior top-1/3/5 | Confidence ECE | Search steps P/L/XE/X | Recovery/fallback steps |
| --- | ---: | ---: | ---: | ---: |
| `previous-default` | 0.3%/0.6%/0.6% | 0.0022 [0.0008, 0.0081] | 383/137/0/666 | 32/157 |
| `selected-predictive-v20` | 0.3%/0.6%/0.6% | 0.0016 [0.0013, 0.0076] | 383/139/0/622 | 30/281 |

Development decisions:

- `selected-predictive-v20` is eligible on solve quality because the paired interval is entirely below zero.

This comparison did not access the sealed window; the release summary records its subsequent once-only evaluation.
<!-- END GENERATED ROLLING EVIDENCE -->
