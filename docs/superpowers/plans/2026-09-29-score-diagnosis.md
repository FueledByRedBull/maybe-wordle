# Predictive score diagnosis: 29 September 2026

## Decision

Keep selected v20 unchanged. The retained 360-game development mean is 3.194444;
four new bounded experiments produced no qualifying improvement. This is a
practical plateau for the tested hypotheses, not proof that a sub-three mean is
impossible. No production source, configuration or executable changed.

Research used source revision `d279a1d205e9721bc4807f1300b8c3ec8ebc1033` and the
existing evaluation policy: development through 2026-08-26, excluding
2026-06-18 through 2026-07-17 validation targets. The 2026-08-28 through
2026-09-26 seal was not evaluated.
Earlier consumed answers remained usable only as chronological training history.
No later missed answers were retroactively added to earlier answer sets.

## Where the guesses go

The retained selected-policy results contain 1,150 guesses over 360 solved games,
with no failures or coverage gaps. Strictly below three requires at least 71 net
guesses saved.

| Initial modeled support | Games | Two | Three | Four | Five | Six | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| In core | 332 | 31 | 240 | 61 | 0 | 0 | 3.090361 |
| Out of core | 28 | 0 | 4 | 12 | 8 | 4 | 4.428571 |

All twelve five/six-guess games are out of core and concern never-used answers.
The target first enters active support on turn two in two tail games, turn three
in 17, turn four in eight, and turn five in one. Missing calibration observations
are not zero-loss observations.

The staged policy keeps fallback answers dormant until the active set reaches
four or is exhausted. Its configured 6.6449% fallback mass supplies recovery
weights, not positive root probability for dormant answers. This explains a
mechanism, not the causal number of guesses it costs: these words may also be
harder. Even reducing every tail game to three leaves the overall mean at
3.083333, still needing 31 further net saves to go below three.

In all 61 in-core four-guess games, target probability before turn three is at
most 50% (mean 0.331329). That rejects the narrow explanation that the policy
ignored an almost-certain answer; it does not rule out other ranking/prior errors.

## Why solve-by-three coverage is not mean guesses

For a second guess, let `C2` be the sum of the largest normalized answer
probability in each feedback bucket, including green. When all answers are
legal guesses, this is the maximum fixed-belief probability of solving by
total turn three with one final answer guess.

If unresolved games after turn three receive score four, the expected capped
score is `4 - C2 - P(green)`. Thus maximizing coverage alone can lose more
immediate solves than it gains later. This identity is not a six-turn dynamic
policy value.

An exhaustive local scan evaluated all 14,855 legal guesses at each of 360
retained second-turn states: 5,347,800 roots in 8.16 seconds. Current feedback
paths and retained target-posterior observations were reconstructed; this was
not a fresh action ranking at every historical state.

| Mean fixed-belief quantity | Selected move | Coverage-first maximum |
| --- | ---: | ---: |
| Solve by total turn three | 0.800426 | 0.818629 |
| Immediate solve at turn two | 0.079646 | 0.011799 |

Coverage-first changes 203 actions but worsens mean capped expected score by
0.049645. Maximizing `C2 + P(green)` instead offers only 0.012612 mean capped
headroom (4.5404 modeled score units across the 360 states). This must not be
subtracted from the published mean or called a bound on six-turn improvement.
For 26 of 28 tail games, the target is still dormant at turn two and contributes
no mass to this calculation.

An initial empirical coverage-first replay stopped after 15 games: two wins,
11 ties, two losses, zero net guess change. It is not a full benchmark. A later
timed reproduction located its slow next game: the alternate second guess leaves
63 active candidates for turn-three exact search versus 17 for selected staged.
The alternate request cancels at ten seconds; selected staged completes the
whole game in about 0.4 seconds. The unfinished alternate outcome remains unknown.

## Bounded follow-up screens

Dates were selected before the screen by position 0, 14 and 29 in each existing
30-game fold: 36 dates, not selected by outcomes. Each arm used top 5, the
unchanged opener, full staged turn context and six-turn rules. Per-request/
comparison cancellation was ten seconds, with a 20-minute overall run cap.
These are development screens, not a new 360-game or prospective benchmark.

| Policy | Completed games | Candidate / matched control guesses | Result |
| --- | ---: | ---: | --- |
| Selected staged | 36/36 | 121 / 121 | All retained paths reproduced |
| Early uniform tail | 33/36 | 113 / 112 | Three staged-request timeouts |
| Early lexical tail | 35/36 | 118 / 118 | One staged-request timeout |
| Maximize C2 + P(green) at turn two | 36/36 | 121 / 121 | Three wins offset three losses |
| Guarded staged-continuation rollout | 36/36 | 121 / 121 | No completed differing-root comparison; retained staged |

All completed games solved. Unfinished games are unknown, not successes or
failures; partial totals must not be compared with 121/36. Uniform activation
saves two tail guesses but loses three core guesses among its completed pairs.
Lexical activation saves two tail guesses but loses two core guesses. Capped
ranking saves one core guess but loses one tail guess. Exploratory paired
intervals include zero for all three; none earned full-360 expansion.

### Belief experiment

The lexical prior uses add-one-smoothed positional answer-versus-dictionary
letter likelihood ratios, combined by their geometric mean. It fits only answers
dated before each target, preserves initial raw tail mass and conditions that
mass after feedback. This is expanding-history fitting, not a frozen-per-fold fit.
Uniform recovery on core exhaustion remains unchanged.

On 36 identical post-opener full-support states, lexical mean log loss is
3.690262 versus uniform tail's 3.703379; Brier is 0.913936 versus 0.914110, with
no missing or mismatched pairs. This is one calibration dataset, not independent
replication across the arms, not production's active-only posterior and not
calibrated editorial probability. Its small improvement did not yield a
qualifying game-score improvement.

### Continuation-cost experiment

A challenger maximizes `C2 + P(green)` over all legal roots. It is compared with
the staged move using unchanged staged continuation through six turns. A separate
fixed core/tail distribution scores every branch while dynamic recovery chooses
future moves. One extra unit is charged for an unsolved sixth turn, matching
all-game penalty seven. A change requires complete values, no higher modeled
failure and strictly lower penalized expected cost.

Twenty nominations already match staged. All 16 differing nominations exhaust
their ten-second comparison budget: 15 during the incumbent tree, one after
starting the challenger tree. There are zero complete comparisons and zero
accepted changes. The replay completes by retaining staged; its unchanged score
does not establish that the uncomputed challengers are inferior.

Sequential process wall times are 75.04s control, 79.11s uniform, 75.21s lexical,
73.93s capped and 234.36s rollout. Completed whole-policy turn p95 is 3.371s control
versus 12.676s rollout (3.76x); the other arms are 3.352s, 3.441s and 3.318s,
excluding their separately reported unfinished requests. Cooperative caps are
not hard real-time guarantees. Process-lifetime peak working sets are 72.5-77.3
MiB, shared across each process's games and stages, not per-candidate rankings.

## Verification and reproduction limits

Thirteen focused research tests passed, covering chronology, tail mass, immediate
solve value, the sixth-turn penalty, fixed scoring versus dynamic recovery and
cancellation. Independent review checked mathematics, completed-result arithmetic,
calibration denominators and the distinction between completed replay and
incomplete value comparisons. No new full application-suite result is claimed.

Word-bearing results, scratch helpers and execution notes remain private under
`target/score-diagnosis/` and `target/tail-cost-research/`. They are not public
release artifacts; this document alone does not provide an independently runnable
benchmark. Reproduction requires those retained local inputs and helpers.

Key SHA-256 identities:

| Input | Digest |
| --- | --- |
| Production prior | `62ab0bdf49b29e48e40a6ecd71c43260cfc1b0d63449d53e62bea780ffe7397c` |
| Evaluation policy | `85f6c83d82cc82796d75877a409b4e7ade2b2eb1b328d6be237d5b234ca39acb` |
| Private retained 360-game matrix | `0c847dce4f41642045a0bb25f4f0019ecfaac9cab41ec48b9300a9d2c40d34de` |
| Raw history | `c9dd24b3ad29c88c1c7ed1fad0d7fafa84bfcfc8ac67b19408ab146e7930685c` |

The follow-up's verified domain-separated input digest is
`04193948ba1b0f42a6a139c0b7ef26f6e721de03f546165558b3845f5af24c0f`;
it binds config, policy, history, seed inputs, retained matrix and both Rust
helpers. Local execution notes record individual helper digests and commands.

The temporary in-crate test hook was removed and rebuildable binaries/cache
cleaned. Production source/config/dependencies and both dist executables remain
unchanged. README's generated benchmark evidence is deliberately unchanged.

## When to reopen research

Require evidence for a substantially stronger chronological answer-likelihood
signal or a cheaper trustworthy continuation-value comparison. Preserve complete
paired outcomes, core/tail strata, calibration, latency and failure guards before
promotion. More parameter sweeps, blanket tail activation or a larger timeout
alone are not justified by these results. Global optimality, the causal cost of
delayed activation and achievable unseen-data mean remain open questions.
