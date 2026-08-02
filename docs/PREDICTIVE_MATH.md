# Predictive solver mathematical contract

This document specifies the predictive pipeline implemented by Maybe Wordle. It separates exact identities, admissible bounds, and heuristic ranking terms. A quantity described as a heuristic is not an estimate of true NYT editorial probability or globally optimal expected guesses.

The authoritative implementation lives in `src/scoring.rs`, `src/model.rs`, `src/solver/`, and `src/experiments/`. Tests are part of the contract; this document does not turn an untested heuristic into a proof.

## 1. Date-bounded support and prior

For a game on date `d`, the solver uses information available through `d - 1 day`.

The primary answer support is the union of pinned candidate answers, historically observed answers whose first observation is not in the future, and manual additions. Every syntactically valid guess is also retained as dormant fallback support. A future history-only word is not eligible before its first observed date.

For primary candidate `a`, the unnormalized prior is

```text
w(a, d) = base(a) * recency(a, d) * manual(a)
```

where `base(a)` is `base_seed_weight` or `base_history_only_weight`, and a missing manual multiplier is `1`.

If `a` has never appeared before `d`, `recency(a, d) = 1`. Otherwise let `t` be whole days since its most recent appearance:

```text
recency(t) = cooldown_floor,                                      t < cooldown_days
recency(t) = cooldown_floor
             + (1 - cooldown_floor)
               / (1 + exp(-logistic_k * (t - midpoint_days))),   otherwise
```

The second branch is monotone increasing when `logistic_k > 0`, tends to `cooldown_floor` as its argument tends to negative infinity, and tends to `1` as it tends to positive infinity. There can be a jump at `cooldown_days`; continuity is not assumed and must be evaluated as a model choice.

At state `S`, positive weights are normalized:

```text
P(a | S, d) = w(a, d) / sum_{x in S} w(x, d)
```

All masses must be finite and non-negative, and the normalized total must be within `1e-9` of one. If modeled mass is zero but supported candidates remain, the declared recovery policy is applied; the runtime never silently deletes date-supported candidates.

Dormant valid guesses receive a total raw fallback mass controlled by `fallback_prior_mass`. They are filtered by every observation but become active only under the declared `fallback_activation_threshold`/inconsistency rule. This provides full historical coverage without constructing a full guess-by-all-valid-answers pattern table.

## 2. Wordle feedback

Feedback uses two passes so repeated letters cannot consume the same target occurrence twice:

1. mark exact-position matches green and remove those target occurrences;
2. scan remaining guess positions left-to-right, marking yellow only when an unused matching target occurrence remains;
3. otherwise mark gray.

Trits are `0 = gray`, `1 = yellow`, `2 = green`, encoded as

```text
pattern = sum_{i=0..4} trit[i] * 3^i
```

so every pattern is in `[0, 242]` and fits in one byte. Filtering retains answer `a` exactly when `score(guess, a)` equals the observed pattern.

Duplicate-letter fixtures and encode/decode round trips are tested in the scoring and integration suites.

## 3. Partition statistics

For guess `g`, state `S`, and feedback bucket `S_p`:

```text
mass[p]  = sum_{a in S_p} P(a | S, d)
count[p] = |S_p|
```

The implementation computes the following exact statistics for the declared state distribution:

```text
entropy(g) = -sum_p mass[p] * log2(mass[p])
expected_remaining(g) = sum_p mass[p] * count[p]
solve_probability(g) = mass[all_green]
```

Zero-mass buckets contribute zero to entropy. `force_in_two` means every non-green bucket has at most one answer; it is a structural property of the modeled state, not a guarantee about candidates outside support.

The solver also records the largest non-green bucket mass and size, counts of buckets above declared size/mass thresholds, concentration, and mass in large buckets. These are diagnostics and heuristic features.

For positive-mass non-green bucket probabilities `q_i`, the normalized concentration penalty is

```text
C = (sum_i q_i^2 - 1/k) / (1 - 1/k)
```

clamped to `[0, 1]`, where `k` is the number of positive-mass non-green buckets. It is zero for one or fewer buckets. Structural zero-mass buckets remain visible to coverage/trap diagnostics but do not dilute this probability concentration. This term measures unevenness; it is not an expected-guess value.

## 4. Proxy continuation score

For each non-green child bucket, proxy continuation cost is:

```text
0                                      all green
1                                      singleton
1 + (1 - largest_mass / bucket_mass)   2 <= count <= proxy_small_state_lower_bound_threshold
max(count / 243, entropy_bits / log2(243), 1) otherwise
```

The guess proxy cost starts at one for the current guess and adds the probability-weighted child costs. Concentration is not embedded here; it is an independent registered feature in the large-state score below.

The large-state ranking score is a linear heuristic:

```text
 score = + entropy_w * entropy
         - bucket_mass_w * largest_non_green_mass
         - bucket_size_w * largest_non_green_size
         - ambiguous_w * high_mass_ambiguous_bucket_count
         - proxy_w * proxy_cost
         + solve_prob_w * solve_probability
         + posterior_w * posterior_answer_probability
         - smoothness_w * concentration_penalty
         - gray_reuse_w * known_absent_letter_hits
         - large_bucket_count_w * large_bucket_count
         - dangerous_mass_count_w * dangerous_mass_bucket_count
         - large_bucket_mass_w * mass_in_large_buckets
```

Feature signs are explicit. Several features are correlated (largest mass, bucket count, concentration, and mass in large buckets), so their coefficients cannot be interpreted causally. Out-of-fold ablation/regression evidence is required before simplifying or promoting weights.

### Coupling audit result (2026-07-19)

`exact_exhaustive_threshold` formerly selected both the exact-search budget and the proxy child-cost formula. This made an exact-search knob silently change broad-state ranking. The formula now uses the independent, registered `proxy_small_state_lower_bound_threshold`. The small-state branch also formerly read a uniform-count table under a weighted prior; it now uses the weight-aware one-step cost above. A threshold of zero selects the large-state analytic heuristic for every non-singleton.

## 5. Second-turn three-solve coverage

For selected second-guess candidate `g`, the coverage analysis asks for each non-green child whether the child is a singleton or any allowed reply partitions it into singleton non-green buckets. It records covered posterior mass, uncovered answer count, and uncovered bucket count.

This is an exact structural check within capped child states, but the root candidate scan is bounded. Therefore it proves coverage only for the scanned candidate and modeled child support; it does not prove the globally best second guess was scanned.

The feature is active only when:

```text
observation_count == 1
and second_guess_coverage_min_survivors <= |S|
and |S| <= second_guess_coverage_max_survivors
```

`second_guess_coverage_max_survivors = 0` disables it. The number of proxy-ranked roots scanned is exactly `second_guess_coverage_pool`; child buckets larger than `second_guess_coverage_child_cap` are conservatively classified as uncovered.

### Coupling audit result (2026-07-18)

Activation and pool size formerly depended on exact/lookahead thresholds. This made tuning exact search silently alter a second-turn objective. Activation min/max and pool size are now independent registered parameters, with tests proving that changing `exact_threshold` does not change activation.

## 6. Search allocation

`search_policy_mode` is an explicit categorical policy:

- `staged`: exact at or below `exact_threshold`, danger-triggered pooled exact where eligible, lookahead at or below its thresholds, proxy otherwise;
- `proxy_with_exact_endgame`: exact at or below `exact_threshold`, proxy otherwise;
- `proxy_only`: proxy ranking at every state.

Within exact mode, states at or below `exact_exhaustive_threshold` scan every allowed guess; larger eligible states use a bounded candidate pool. Candidate-pool exact search is exact only over that pool. The pool is a deduplicated mixture of the primary proxy, entropy, worst-bucket, worst-mass, solve-probability, and posterior-answer rankings. Each source fraction is registered independently. Tight and medium score-gap expansion multipliers are also registered rather than fixed in code.

After five observations, only one guess remains. The runtime therefore overrides unlimited-horizon proxy/lookahead/exact ordering and ranks guesses by immediate solve probability, then posterior answer probability, then lexical order for deterministic ties. Information gain has no value after the final guess. This does not remove irreducible failures when several unseen, cutoff-safe dormant candidates have identical mass; recovery activation must expose those candidates early enough for previous guesses to separate them.

The danger score is a normalized weighted combination of top-posterior concentration, largest-bucket mass, largest-bucket size ratio, ambiguity pressure, and top-candidate disagreement. Registry v6 makes the posterior and candidate windows, mass and size disagreement cutoffs, and ambiguity saturation count explicit alongside the five feature weights and allocation thresholds. Its thresholds allocate computation; it is not a probability of failure. Lookahead starts with one for the root guess and adds each branch probability times the selected child reply's complete proxy cost. That proxy cost already includes the reply guess. The previous heuristic path added another unit at the child, double-counting the reply only above `exact_exhaustive_threshold` and creating a discontinuity with exact children; a hand-computed regression now prevents it. A later domain audit found that bounded child-reply pools could still admit a guess whose largest non-green bucket was the entire child state. Such a reply has no well-founded finite continuation value, so child metrics now apply the same strict-progress predicate used at the root and by exact recursion. The slow reference and a deliberately inert-guess fixture enforce the same domain. Proxy continuation cost no longer embeds an extra concentration surcharge because concentration already has its own registered score weight. Root lookahead penalties apply to four separate inputs—worst-branch posterior mass, large-bucket count, dangerous-mass count, and mass in large buckets. Approximate replies add candidate-count ratio through its own registered coefficient instead of merging it into the posterior-mass coefficient. These penalties remain heuristic and must never be labeled exact expected guesses.

### Tractable-state regret audit (2026-07-26)

`search-regret` follows deterministic historical targets with the artifact-free proxy policy until a requested posterior-size band is reached. It then evaluates the production choice, forced proxy choice, and configured bounded-lookahead choice on that identical posterior. The reference scans every allowed root guess and uses exhaustive Bellman continuation. Reports bind the executable, source, config, and data identities; record the exact observation path; enforce a wall-clock cap between games and states; and are diagnostic development evidence, not sealed-test solve scores.

The first run found that proxy ranking could return a guess whose only non-green bucket contained the entire state, giving it infinite continuation cost. Predictive ranking now removes every non-progressing guess before any proxy, lookahead, or exact candidate selection, with a regression test using a deliberately inert extra guess.

After that correction:

- [`search-regret-v1.json`](../benchmarks/predictive/search-regret-v1.json) evaluated 16 time-spread states with 3–8 survivors. Production, proxy, and lookahead all matched exhaustive cost on every state, apart from sub-`1e-15` floating-point noise reported as zero positive-regret states.
- [`search-regret-9-16-v1.json`](../benchmarks/predictive/search-regret-9-16-v1.json) evaluated 14 available states with 9–16 survivors. Proxy-only ranking was suboptimal on 6/14 states, with mean regret `0.340952` and maximum regret `0.899180` expected guesses. Bounded lookahead was suboptimal on 3/14 states, with mean regret `0.000154` and maximum regret `0.001724`. Production matched exhaustive cost on all 14 states.

Across both reports, bounded lookahead matched exhaustive cost on 27/30 states and its combined mean regret was about `0.000072`, versus about `0.159111` for proxy-only ranking. This evidence supports retaining the bounded lookahead candidate mixture and exact escalation: the added search nearly eliminates the large proxy error. It does not establish that each individual penalty or pool source has held-out solve benefit; those terms still require the registered ablations and rolling studies before simplification or promotion.

### Learned continuation-cost experiment (2026-08-02)

The learned proxy is a residual model, not a replacement recurrence. For row features `x`, existing proxy cost `C_proxy`, and exhaustive Bellman label `C_exact`, training minimizes

```text
sum_i (C_exact,i - C_proxy,i - beta_0 - x_i beta)^2 + lambda ||beta||_2^2
```

after train-only population standardization. The intercept is not regularized. Deterministic pivoted elimination solves the regularized normal equations and rejects a solution whose backward residual exceeds tolerance. Artifact validation binds the ordered feature schema, scaling, dataset/replay identities, and coefficients. Invalid dimensions, non-finite inputs, invalid models, or negative/non-finite predictions return the explicit baseline score with a machine-readable fallback reason.

The native adapter samples reachable 3–12-survivor states from three contiguous development-only windows: training, validation, and an inner development test. State/trajectory identities must be disjoint and every held-out date must follow every training date. Each row records the complete survivor IDs and weights, date, turn, deterministic candidate guess, baseline features, and exact continuation cost. Candidate guesses mix primary proxy, entropy, worst-bucket/mass, immediate-solve, and surviving-answer coverage. This is exact for every recorded `(state, guess)` row, but it is a bounded diagnostic candidate set—not a proof that all allowed guesses were materialized.

The recorded artifact has 401 rows across 24 states. Ridge regularization was selected only on validation. Learned and baseline rankings both selected a zero-regret row on all validation/test states, but learned pairwise accuracy was worse (`0.9419` versus `0.9527` validation; `0.9712` versus `0.9773` inner test), and inner-test MAE rose from `0.002990` to `0.005397`. A same-window five-state exhaustive reference gave production, proxy, and lookahead zero regret, but that sample is diagnostic rather than a solve-quality gate. Evidence floats are canonicalized to `10^-12` absolute precision before checkpointing so semantically equal Bellman results survive JSON replay with a stable digest; this does not alter the live solver recurrence. The artifact is therefore non-promotable. Even a ranking win would still require full rolling solve, coverage, failure, latency, and memory guards before production use.

### Policy-era survival experiment (2026-08-02)

Reuse intervals are represented as half-open daily risk intervals `[entry, exit)`. Reuse contributes an event on the final risk day; a right-censored interval contributes no event. First observed appearances are left-truncated because the last use before the history origin is unknown and therefore are not labeled as reuse events. Never-used support mass is tracked separately and never converted into a censored reuse event.

Intervals crossing an editor-policy boundary are split by era while retaining an `elapsed_offset_days` value, so `t` remains days since the original last use rather than resetting at the boundary. Fold construction clips exposure to `[training_start, training_end + 1)`, converts future events to censoring, and rejects date/era mismatches. Identical `(era, elapsed-day)` rows are aggregated into weighted binomial observations; this preserves the logistic likelihood exactly while reducing allocation and fit time.

The daily hazard is a regularized logistic model over a low-degree polynomial in `log(1 + t / scale)` plus policy-era indicators. Ridge and second-difference penalties control coefficient size and time curvature. Dated inference integrates daily hazards using the era active on each risk date. This differs from the selected hand-set recovery curve and remains an experiment; output weights are heuristic prior scores, not calibrated editorial probabilities.

Across the canonical 12 development folds, only 26 fold-local reuse events were available. On 360 validation games, the survival score had log loss `6.750447` and Brier `0.998692788`, versus `6.690665` and `0.998682995` for the selected logistic curve; both had 23 support gaps. The paired proxy-only solve diagnostic held the search policy and no-book condition fixed across all folds. Logistic had conditional/failure-penalized mean guesses `3.4171/3.4250` with 3 unsolved games; survival had `3.4415/3.4444` with 1 unsolved game. Survival's maximum fold p95 was `1982.4` ms versus `1689.0` ms; process peak was recorded as `166080512` bytes. Sparse event count, worse probability scores, coverage gaps, worse penalized guesses, higher latency, and the still-unrun production-search solve gate block promotion. The sealed `2026-06-18` through `2026-07-17` outcomes were not read as development targets.

## 7. Exact expected cost and lower bounds

For normalized state `S`, exact expected cost is the Bellman recurrence

```text
C(S) = min_g [1 + sum_{p != green} P(p | g, S) * C(S_p)]
```

with singleton cost `1`. A guess that leaves a non-green child equal to `S` is non-progressing and excluded.

Memo keys contain the sorted answer subset; weights are date-fixed for one solver evaluation. Branch-and-bound uses the weight-aware admissible one-step bound

```text
LB(S) = 1 + (1 - max_a P(a | S))
```

because one guess can solve at most one distinct answer and every non-green outcome requires at least one more guess. The previous count-only abstract-partition bound assumed uniform mass and was not admissible for a skewed predictive prior; a `0.40/0.59/0.01` regression demonstrates the premature-prune case and cross-checks the corrected result against exhaustive root evaluation. Zero-mass branches contribute zero to the expected recurrence and are not recursively evaluated, preventing zero-mass errors and `0 * infinity` NaNs; runtime recovery still handles such a branch if it is actually observed. Pool-limited exact search is reported as candidate-pool exact cost, not a global optimum.

## 8. Probability scores

For normalized class probabilities `p_i` and observed target class `y`:

```text
log_loss = -ln(max(p_y, 1e-12))
Brier = sum_i (p_i - 1[i = y])^2
```

The multiclass Brier score is unhalved, so its range is `[0, 2]`. Input is rejected unless probabilities are finite, non-negative, and sum to one within `1e-9`.

Current log loss around `7.23` and Brier around `0.9992` indicate a weakly calibrated editorial prior. GUI/CLI probabilities therefore remain labeled heuristic scores. Calibration claims require rolling-origin reliability/ECE evidence, not only a lower log loss.

## 9. Evaluation contract

Every scheduled date has exactly one status:

- solved in `1..=6` guesses;
- unsolved after six guesses;
- coverage gap (no eligible target in support).

Coverage gaps never disappear from denominators. The canonical score uses penalty `L = 7`:

```text
all_game_score = mean(
    guesses,                  solved
    L,                        unsolved or coverage gap
)
```

`conditional_mean_guesses` is named explicitly and averages modeled games only. It must not be compared with an all-game score when coverage differs.

Reported distribution statistics include the 1–6 histogram, median, p90, p95, maximum, and solved-within-three/four rates. Coverage and solve-rate intervals use Wilson score intervals.

Prior ranking evidence reports top-1, top-3, and top-5 recall with Wilson intervals. Ten-bin confidence expected calibration error uses the maximum prior probability as confidence and whether that top-ranked word is the observed answer as correctness. Its interval uses the same deterministic chronological moving-block bootstrap as other standalone statistics.

Standalone mean-like intervals use a deterministic moving-block bootstrap (`2,000` resamples, block length `7`, recorded seed). Paired comparisons resample chronological candidate-minus-baseline per-game differences, preserving dates. Negative delta favors the candidate. Win/tie/loss counts use per-game penalized values.

Overlapping standalone intervals are not a paired significance test. A candidate is promoted only when hard constraints pass first: no added coverage gaps or failures, then solve quality/calibration, then latency/memory.

## 10. Rolling-origin and sealed test

The default evaluation plan uses expanding chronological training and 30-day, non-overlapping validation folds. Features, eligibility, and artifacts for target date `d` are bounded to information before `d`.

The final 30-day window is excluded from optimizer measurements and may be evaluated exactly once only after configuration and artifacts are frozen. Evidence generation, rolling comparison, all study/tuning paths, and live-config evaluation derive their boundary from one canonical plan helper and reject ranges that intersect it.

The final v20 candidate passed the complete 12-fold development guard at `360/360` and `3.1778`, versus `358/360` and `3.3000` for the previous default. The paired candidate-minus-default delta was `-0.1222` with chronological block-bootstrap interval `[-0.1722, -0.0722]` and win/tie/loss counts `69/265/26`.

After the configuration and artifacts were frozen under `sha256-v1:15cb4c86c7548dbcdf94624a8a80b93009a8ebbdcc57d228617977765d4a543a`, the sealed 2026-06-18 through 2026-07-17 window was evaluated once. The candidate solved `30/30` games with all-game mean `3.3000`, interval `[3.1333, 3.4667]`, no coverage gaps, no failures, median 3, p95 5, and maximum 5. The result does not support a `<= 3.0` claim, and this sealed window must not be reused for tuning.

## 11. Study fidelity and promotion

The common study runner evaluates only rolling-development folds. Given initial fold count `F0`, reduction factor `eta >= 2`, and maximum `Fmax`, its deterministic fidelity schedule is

```text
F_r = min(Fmax, F0 * eta^r)
```

with the final `Fmax` rung inserted when multiplication would skip it. A candidate resumes from its saved additive fold accumulators; an already-recorded fold cannot be merged twice. At each non-final rung, candidates are ordered lexicographically by coverage gaps, failures, all-game failure-penalized mean when that stage measures solves, log loss, Brier score, latency, peak memory, and stable candidate number. The best `ceil(n / eta)` advance. Candidate zero is retained as the explicit reference baseline even when it would otherwise be pruned.

The remaining roadmap studies use `F0 = 3`, `eta = 2`, and `Fmax = 12`, so their fidelity rungs are `3 -> 6 -> 12`. Each rung is a deterministic nested, time-spread subset of the same canonical 12 outer development folds; it is not a new resample or additional independent evidence. Spreading low-fidelity measurements across the development span avoids pruning solely on the oldest consecutive months. Every promoted finalist is evaluated on all 12 outer folds. Multiple optimization seeds alter candidate generation only, while `BookPolicy` rebuilds cutoff-safe artifacts for those same outer folds. Any inner chronological out-of-fold rows used to fit continuation-cost models belong exclusively to an outer fold's training data and never replace or augment its held-out validation score.

The declared process-memory cap is enforced against peak working-set bytes on Windows, Linux, and macOS at solver construction, fold/game checkpoints where available, and final latency measurement. Peak bytes are checkpointed and enter guarded/Pareto ordering after latency. Exceeding the cap fails the trial; an unmeasured value ranks behind a measured value. Windows is release-tested; the macOS sampler still requires native-hardware validation. Wall-clock time is cumulative across resumed fidelity rungs and includes contention while a candidate is active. The default is 7,200 seconds: the earlier 3,600-second v11 proxy screen could complete six folds but not the final 12-fold rung, so that partial state is screening evidence only.

Calibration-only studies deliberately leave guess means and latency null: zero is not a valid stand-in for an unmeasured solve objective. Their results can nominate prior candidates for later solve-policy evaluation, but cannot directly promote a solver configuration. The equal-compute strategy evidence in `benchmarks/predictive/study-strategy-comparison-v8.json` therefore reports calibration convergence and seed sensitivity, not a mean-guesses improvement.

Static grid, low-discrepancy, random, and local-refinement suggestions are deterministic functions of the declared seed, registry, and budget. They evaluate independent candidates in parallel and may use successive halving. Fold scoring intentionally omits latency while candidates share the worker pool; after the pool joins, complete 12-fold finalists receive serialized latency measurements. This prevents scheduler contention from being mistaken for candidate latency. Model-based mode first evaluates a deterministic global startup pool, then separates completed trials into the guarded best quartile and remainder. It draws kernel proposals near the elite values and maximizes the log density ratio `log l(x) - log g(x)`. Suggestions are sequential and atomically checkpointed before evaluation, so resume preserves the ask/tell sequence. The five-seed evidence shows that both startup exploration and TPE refinement matter; neither is sufficient promotion evidence without solve outcomes.

Study domains and cohorts are explicit. `Calibration` changes prior parameters only and measures prior scores; recovery knobs are isolated in `CoverageRecovery`, where they can affect the objective. Proxy work is partitioned into `ProxyCore`, `ProxyRisk`, and `ProxySmallState`; search allocation is partitioned into `SearchRouting`, `SearchExact`, `SearchCoverage`, `SearchLookahead`, `SearchPool`, `SearchDanger`, and `SearchPenalty`. These granular stages measure rolling solve outcomes without books. Registry validation requires each cohort's domain and optimizer role to agree, and tests compare all 85 registry entries with every serialized `PriorConfig` leaf, verify that every entry changes config identity, and prove that all 79 optimizer-controlled knobs occur in exactly one granular stage. The registered values include the opener holdout shortlist, artifact freshness/rebuild cadence, reply-book candidate pool, exact second-guess coverage root pool, its force-in-two child cap, ambiguity cutoff, all danger weights/windows/cutoffs, two pool-expansion multipliers, six exact-pool source fractions, and the separate reply bucket-ratio penalty. Static and model-based granular studies begin with a deterministic valid one-factor perturbation for every eligible setting and reject trial counts that cannot include this sweep plus the baseline; wider proposals begin only after this coverage prelude. `ProxyRanker` and `SolvePolicy` retain their aggregate semantics for compatibility, while `Joint` is the deliberate prior/recovery/proxy/search cross-domain refinement for finalists. `--base-config` carries an exact frozen TOML result into the next stage and that canonical base is part of study identity. `BookPolicy` rebuilds isolated candidate/fold artifacts at chronological cutoffs and evaluates them in disk-only mode. The legacy proxy fitter's greedy per-field objective on one 80/20 state split is no longer an optimization path. Its calibration-row builder is retained only as input preparation for a future leakage-safe, out-of-fold continuation-cost model.

Diagnostic search variants are serialized registry-validated profiles under `config/profiles/`. The offline-book migration corrected a previously hidden inconsistency: its root candidate/reply pools exceeded the declared ranges and were larger than the corresponding medium-state pools. The profile now satisfies `root_pool <= medium_pool` for both candidates and replies; this is a correctness/configuration fix, not evidence that the new values improve guesses.

Fixed experiment cohorts use the same typed values in format-v1 matrices under `config/experiments/`. Optimizer domains retain strictly positive minima for log-scaled weights. A separate diagnostic application rule permits exactly zero for float parameters so a term can be removed in an ablation; it does not permit negative values, arbitrary out-of-range nonzero values, operational/safety parameters, or configs that violate cross-field validation.

## 12. Artifact and identity contract

Model, word-list/pattern-table, predictive-book, formal-proof, rolling/study, and benchmark inputs use SHA-256 with domain separation and unsigned 64-bit little-endian field lengths. File fields are hashed in bounded streaming chunks and are tested against one-shot encoding. Rolling/study/benchmark provenance covers the exact current executable as well as launch-time source, tests, Cargo manifests, and data inputs; phase-boundary rechecks prevent a long command from publishing results after those inputs change. Text identities use the explicit `sha256-v1:` prefix; filename-safe predictive hashes use the same 256-bit digest under manifest version 2. Pattern tables and formal binary artifacts have new magic values, studies use format v16, the parameter registry uses format v6, benchmark evidence uses schema v4, and rolling comparisons use schema v3. Old or mixed formats cannot be resumed/reused as current evidence and produce a regenerate/rebuild error where they cross a persisted boundary.

## 13. Verification map

Relevant automated evidence includes:

- duplicate-letter and feedback encoding fixtures in scoring tests;
- date-bounded dormant-support and recovery tests in solver/model tests;
- hand-computed entropy, concentration, multiclass Brier/log-loss, bootstrap, Wilson, and paired-comparison tests;
- randomized tractable-state exact/pruning comparisons under positive and zero masses, an independent raw-partition heuristic-lookahead ranking reference, and hand-computed lookahead/proxy-cost fixtures;
- finite/non-negative/normalized-mass boundaries plus registry uniqueness, type/bounds/constraint, config round-trip, per-field identity, danger-definition behavior, and search-mode tests;
- rolling-plan sealed-window and future-only leakage tests;
- versioned JSON evidence under `benchmarks/predictive/`, including exhaustive tractable-state search-regret reports, and generated README fragments under `docs/generated/`.

Release CPU, cycle, allocation, working-set, page-fault, and cold/warm measurements are recorded in [`PERFORMANCE.md`](./PERFORMANCE.md) and its machine-readable artifact. Hardware cache-miss counters were unavailable in the installed Windows toolchain, so the documentation does not invent them.
