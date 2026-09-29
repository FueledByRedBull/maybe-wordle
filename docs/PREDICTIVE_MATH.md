# Predictive solver mathematical contract

This document specifies the predictive pipeline implemented by Maybe Wordle. It separates exact identities, admissible bounds, and heuristic ranking terms. A quantity described as a heuristic is not an estimate of true NYT editorial probability or globally optimal expected guesses.

The authoritative implementation lives in `src/scoring.rs`, `src/model.rs`, `src/solver/`, and `src/experiments/`. Tests are part of the contract; this document does not turn an untested heuristic into a proof.

## 1. Date-bounded support and prior

For a game on date `d`, the solver uses information available through `d - 1 day`.

The live request calls this `puzzle_date`; the response exposes both `puzzle_date`
and the derived inclusive `history_cutoff`. CLI and GUI defaults use the local calendar.
Lower-level snapshot and book-building APIs retain explicitly inclusive `as_of` dates.
Effective model identities sort words canonically and omit future history; raw source
provenance can still change when new records arrive. Existing book hash domains were
advanced so artifacts made under the old identity semantics are not reused.

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

Dormant valid guesses receive a total raw fallback mass controlled by `fallback_prior_mass`. They are filtered by every observation but become active only under the declared `fallback_activation_threshold`/inconsistency rule. This extends coverage without constructing a full guess-by-all-valid-answers pattern table; it does not guarantee coverage of future editorial answers. A future history-only word outside the guess dictionary is excluded from dormant support until date-eligible, so it cannot dilute earlier fallback weights.

### Experimental finite-horizon policy

The September `finite_fast` and `finite_strong` policies are under integration and
validation, not validated replacements for the selected production configuration.
They freeze a single distribution at the puzzle cutoff. Positive modeled answers
form the core, with normalized total mass `1 - fallback_prior_mass`; remaining
supported answers share the tail mass uniformly. A zero modeled weight (including
a manual zero) means tail membership, not a hard vocabulary exclusion. If there is
no positive core, the supported universe is uniform; if there is no tail, the core
has total mass one. An explicitly zero tail mass excludes that tail from planning.
Feedback filters this distribution without changing individual weights or invoking
another recovery rule. Prior-only model scores and effective planning probabilities
must remain distinguishable.

The controlled finite API can also run experimentally on the selected staged
*dynamic* belief. In that case `S` below includes active weights, dormant
support, activation/recovery flags and, in hard mode, the observation history;
it is not merely a survivor subset. Branch probability uses the parent state's
current mass, then each non-green child applies the same fallback and recovery
transition as live `apply_feedback`. Dormant words have zero *current* mass
until activated, so their eventual empirical coverage risk is not a positive
root failure probability under this model. Memo keys include the dynamic
weights/support/flags as well as the horizon and hard-mode history. This path
is opt-in, not a validated replacement for selected `staged`.
Dormant fallback answers are partitioned once per evaluated guess and reused
for its positive-mass non-green children. This leaves completed child beliefs
and their values unchanged, but can reduce charged work units and can change which
bounded-search roots finish before a deadline; a returned upper bound remains
only a feasible policy value, not an exact global optimum.
The raw benchmark path below is a local-only diagnostic; it is shown as inline code and is not part of a public clean checkout.
`../benchmarks/predictive/september-dynamic-finite-30day-v1.json`
Its score/latency rejection is recorded in [the release ledger](SEPTEMBER_RELEASE.md).

Let `h` be remaining turns, and let `L(S,H)` be guesses legal under the simulated
hard-mode history `H` (all allowed guesses in normal mode). For an unresolved state:

```text
V(S, H, 0) = (1, 0)
V(S, H, h) = lex_min over g in L(S,H) of
  ( sum_{p != green} P(p | g,S) * F(S_p, H+(g,p), h-1),
    1 + sum_{p != green} P(p | g,S) * T(S_p, H+(g,p), h-1) )
```

`F` is modeled failure probability; `T` is expected attempts actually taken before
solving or exhausting the horizon, including attempts in failed games. Minimize `F`
first, then `T`. Failure/gap=7 is a separate reporting metric, not the terminal cost
in this recurrence. Finite-horizon non-progress actions are well-founded because
`h` decreases, although a useful baseline should not waste turns repeating them.
Memoization includes survivors, horizon and the full simulated hard-mode history.
When the initial root shortlist and its endgame refinement finish before the
deadline, the finite search spends remaining budget on all legal roots with exact
continuations. Previously completed policies remain usable if widening is
interrupted. Exact memo entries remain valid across this widening because only
exact values are stored there; baseline memo entries describe a threshold-independent
policy. Widening does not guarantee exhaustive completion within the budget.
For a fixed belief, or a dynamic belief with no dormant support, recursive exact
minimization skips a non-solving action whose feedback is identical for every
survivor once an incumbent exists: it consumes a turn without changing support.
This pruning is not valid in general for a dynamic belief: a probe that leaves the
active set unchanged can remove dormant answers and alter later activation.
Those actions retain their exact child transitions. Fixed-root evaluation still
evaluates such actions, so its reported value remains comparable to the independent
oracle rather than pretending the requested move was replaced.
For other recursive actions, a completed feasible incumbent permits branch-level
pruning. The partial lower bound uses exact failure and attempt contributions from
completed feedback children, compulsory failure for unexamined one-turn misses,
and the minimum further attempts compatible with the incumbent's quantized failure
bucket. It prunes only when that lexicographic lower bound cannot improve the
incumbent; a heuristic or upper-bound child disables this proof. Normal-mode
recursion also reuses an already-evaluated action when its full per-answer feedback
signature (including green) matches. Hard-mode recursion does not deduplicate,
because the guessed word can change later legal moves even with the same survivor
partition. These lower bounds are internal dominance proofs, not feasible action
values or exact child costs. They are not cached or displayed as candidate values;
public quality labels remain `Heuristic`, `UpperBound` and `Exact`.
Public predictive requests validate the supplied hard-mode sequence, not just the
next recommended guess, and reject turns after a solved row. The current hard-mode
contract preserves green positions, forbids yellow positions, and requires the
maximum revealed positive count of each letter. Gray tiles filter answer support;
they do not impose a hard-mode upper-count restriction on probe guesses.
Recursive dictionary scans compile these existing constraints once per state and
use the same structured violation check as public validation, without formatting
rejection messages for internal boolean decisions. The rule set and memo identity
are unchanged; the public API still validates inputs and formats the same errors.

This recursive legality contract applies to the finite policies. Legacy staged
search filters the next suggestion for hard-mode legality, but its unlimited-horizon
continuation estimates use normal-mode replies and subset-only memoization. Those
legacy costs are not exact hard-mode policy values; use a finite profile when
recursive hard-mode planning is required.

The staged policy has a direct late-turn route before unlimited-horizon refinement.
After four observations it evaluates the legal roots for the two remaining turns: structural
`force_in_two` witnesses first only when every survivor is a legal final guess,
then the exact modeled two-turn success mass
`sum_p max_{a in S_p and a in G} P(a)` (including the all-green bucket), then
immediate solve probability to minimize attempts among equally successful moves.
For a dynamic belief, each non-green contribution is the parent branch probability
times the best legal answer probability after the exact child transition; using
unmodified parent weights would ignore recovery and support activation.
History/manual answer words absent from the guess dictionary `G` are rejected at
the input boundary; malformed internal terminal states also error. Hard-mode
filtering removes illegal probes. After five observations it ranks immediate
solve probability first. The earlier staged turns still use the unlimited-horizon
heuristic/exact-cost machinery; this exception is not a claim that the whole
staged policy optimizes the six-turn objective or that modeled support covers
every possible answer.

With one turn left, choose the highest-probability legal answer: `F=1-p_max`,
`T=1`. With two turns left and a legal final guess in every child, a proposed `g`
has `F=1-P(g)-sum_{p != green} max_{a legal in S_p} P(a)` and `T=2-P(g)`, where
the probabilities in the sum are measured against the parent distribution. The
unlimited-horizon `2-p_max` bound must not be used as an attempts bound at `h=1`.
For `h>=2`, a feasible zero-failure policy attaining `T=2-p_max` proves the
lexicographic optimum: failure cannot be negative, and every first-guess miss
needs at least one further attempt. Exact endgames may stop on this attainment.

A cheaper sufficient witness for one proposed root guess is possible when the
root has no dormant fallback support. For every non-green feedback bucket,
apply the exact dynamic state transition and require
that the resulting child has no dormant fallback support, at most `h-1` active
answers, and a guess-dictionary entry for every active answer that is legal
under the full child hard-mode history. The sequential-answer continuation
must also keep each exact child transition usable (for example, strict recovery
can reject a branch with only zero modeled mass). Every wrong sequential reply
to an active answer removes that answer; no dormant support can be activated
later, and every still-consistent answer remains legal. Thus the proposed
root has modeled failure probability zero within that declared support. This
does not prove that the root minimizes expected attempts or is
globally optimal, and it says nothing about an answer outside modeled support.
Checking only that fallback does not activate on the root transition is
insufficient: a subsequent wrong reply could trigger it.
The opt-in `staged-zero-failure-certificate` diagnostic uses the stricter
sufficient condition that every active child answer has strictly positive
modeled weight, so later nonempty subsets retain positive mass. It checks
dictionary membership and the complete hard-mode history, rejects dormant
root/child support, and reports only aggregate selected-root counts. It does
not store a per-root proof witness, certify a globally optimal action, or
validate outcomes outside the modeled active support. Missing history dates,
unsupported historical targets, failed path replay, state caps and deadlines
make the range report incomplete rather than a negative certificate result.
With one legal active answer, guessing it solves immediately. With two legal
active answers, guessing the heavier answer first attains the bound only when
no dormant fallback answer can activate after a miss. The direct two-answer
shortcut therefore requires empty dormant fallback support in a dynamic
belief; otherwise the search evaluates the actual child transition. This
condition is conservative when activation is disabled, but prevents an
incorrect zero-failure `Exact` value at low activation thresholds.

Complete rollout values describe feasible policies. They upper-bound the optimum
in lexicographic order, not each coordinate independently. An exact continuation
value for a shortlisted action does not prove that its root action is globally
optimal. Unevaluated deadline fallbacks have no numeric value claim; proposal
sampling is heuristic and must not remove answers from actual rollout evaluation.
Broad roots first compare grouped baseline-policy rollouts across the root shortlist
without recursive exact search, then attempt exact endgame refinement. A completed baseline value
survives an interrupted refinement; incomplete values never enter the exact cache.
The cheap baseline rescoring pool consists of the highest-mass legal surviving
answers, limited by `reply_shortlist`, with lexical mass ties. It does not inherit
the calling root's probe shortlist: the same conditioned state must give the same
baseline action in a rollout and a fresh request. Root improvement still considers
non-answer probes. The state-local change introduced finite policy identity v2; earlier
root-anchor latency results do not establish its solve quality. A matched-budget
empirical comparison remains required before promotion.
Proposal partitions use the same proxy-score finalization as the shared ranking
path, not a separate expected-bucket-count replacement formula.
The experimental v3 policy schedules root proposals by summed unique-letter
Bernoulli variance `sum_l p_l * (1 - p_l)` before computing full proxy partitions.
This is a scheduling heuristic, not an entropy identity, bound, or changed rollout
objective. Proposal work soft-stops at one quarter of its budget to reserve time
for completed values; `proposal_sampled` also records this truncation. Even a
`Complete` status with that flag is not an exhaustive root optimum. Deadlines are
cooperative: support/proposal sorting and request setup are not individually
interruptible. The v3 development comparison did not establish a mean-guess gain.
Execution telemetry records these steps as `finite`, independently of each
candidate's value quality. Evidence tables use `P/L/XE/X/F` for proxy, lookahead,
escalated exact, unlimited-horizon exact and finite-policy steps. Legacy pool-size
ratios do not describe finite search; finite steps leave those fields zero and
report their completed root candidates through the finite-search result instead.

Finite values use floating-point arithmetic. Comparisons round each coordinate to
bins of `64 * f64::EPSILON` (about `1.42e-14`) before lexicographic comparison;
quality and word order resolve remaining ties. This transitive ordering prevents
accumulation noise from deciding mathematical ties. Model-exact labels mean exhaustive
continuation search under this numerical contract, not exact rational arithmetic.
The independent small-state oracle instead accumulates integer probability masses
and attempt totals before normalization to check both coordinates.

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

Zero-mass buckets contribute zero to entropy. In live suggestions,
`force_in_two` means every non-green bucket has at most one answer and every
survivor is in the legal guess dictionary. It is a structural property of the
modeled state, not a guarantee about candidates outside support.

The solver also records the largest non-green bucket mass and size, counts of buckets above declared size/mass thresholds, concentration, and mass in large buckets. These are diagnostics and heuristic features.

For positive-mass non-green bucket probabilities `q_i`, the normalized concentration penalty is

```text
C = 0,                                      k <= 1
C = (sum_i q_i^2 - 1/k) / (1 - 1/k),         k > 1
```

clamped to `[0, 1]`, where `k` is the number of positive-mass non-green buckets. The implementation returns the first case before evaluating the ratio, so it never evaluates `0/0`. Structural zero-mass buckets remain visible to coverage/trap diagnostics but do not dilute this probability concentration. This term measures unevenness; it is not an expected-guess value.

## 4. Proxy continuation score

For each non-green child bucket, proxy continuation cost is:

```text
0                                      all green
1                                      singleton
1 + (1 - largest_mass / bucket_mass)   2 <= count <= proxy_small_state_lower_bound_threshold
max(2 - largest_mass / bucket_mass, count / 243, 1) otherwise
```

The weight-aware floor applies on both sides of the threshold, preventing the previous
uniform 12-to-13-answer cost drop. The count term remains a heuristic estimate;
they are not certified branch-and-bound lower bounds. The guess proxy cost starts at
one and adds probability-weighted child costs. Concentration is a separate feature.

The former child-entropy floor is redundant: `H <= log2(count)`, and
`H/log2(243) <= max(1, count/243)` for every positive count. It and its per-answer
weighted-log accumulation are removed. This does not remove feedback-partition
entropy from the ranking features. Old/new numerical equivalence is regression-tested
across uniform/skewed distributions and the threshold/243/486 count boundaries.

The weighted floor alone cannot increase under partition refinement: its mass-weighted
sum is `2M - sum(child maxima)`, and splitting cannot decrease the sum of maxima.
The full count-based heuristic does not have that property. For example, a 300-answer
unit-mass bucket with maximum mass 0.8 has proxy 1.234568; splitting into a 298-answer
bucket of mass 0.9 (maximum 0.8) and two answers of mass 0.05 each gives 1.253704.
Regression tests preserve this distinction; the full proxy must not be used for pruning.

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

`exact_exhaustive_threshold` formerly selected both exact-search budget and proxy formula.
The independent `proxy_small_state_lower_bound_threshold` selects where the broader
analytic heuristic participates. The weighted floor applies throughout, including at
threshold zero. Neither branch reads the old uniform-count table.

## 5. Second-turn three-solve coverage

For a selected second guess `g`, success by total turn three includes an immediate
win and one final answer attempt in each non-green child. Using probabilities in the
current state (before child normalization), the exact quantity is:

```text
P(T <= 3 | g, turn 2) = P(answer = g) + sum_non_green_B max_(a in B) P(answer = a)
```

The root candidate scan is bounded. This calculation assumes each supported answer is
a legal final guess. Positive-mass answers beyond that one final choice count as
uncovered; immediate green success is included. It is a success-probability diagnostic,
not expected attempts or a guarantee over unmodeled answers.

The former structural check could permit completion on turn four and omitted green
mass. Its evidence must not be labeled success by turn three.

The feature is active only when:

```text
observation_count == 1
and second_guess_coverage_min_survivors <= |S|
and |S| <= second_guess_coverage_max_survivors
```

`second_guess_coverage_max_survivors = 0` disables it. The number of proxy-ranked roots
scanned is exactly `second_guess_coverage_pool`. The obsolete
`second_guess_coverage_child_cap` has been removed from current configuration and
registry v7; legacy TOML inputs may still contain it, but it has no effect and is not
written back. Historical configs remain archived without rewriting their identities.

### Coupling audit result (2026-07-18)

Activation and pool size formerly depended on exact/lookahead thresholds. This made tuning exact search silently alter a second-turn objective. Activation min/max and pool size are now independent registered parameters, with tests proving that changing `exact_threshold` does not change activation.

## 6. Search allocation

`search_policy_mode` is an explicit categorical policy:

- `staged`: exact at or below `exact_threshold`, danger-triggered pooled exact where eligible, lookahead at or below its thresholds, proxy otherwise;
- `proxy_with_exact_endgame`: exact at or below `exact_threshold`, proxy otherwise;
- `proxy_only`: proxy ranking at every state.

Within exact mode, states at or below `exact_exhaustive_threshold` scan every allowed guess; larger eligible states use a bounded candidate pool. Candidate-pool exact search is exact only over that pool. The pool is a deduplicated mixture of the primary proxy, entropy, worst-bucket, worst-mass, solve-probability, and posterior-answer rankings. Each source fraction is registered independently. Tight and medium score-gap expansion multipliers are also registered rather than fixed in code.

For a pooled exact request for the top `K` suggestions, the implementation can
avoid recursive evaluation of roots that provably cannot enter that prefix. For
root guess `g`, let `M` be the current total mass and, for each non-green
feedback bucket `b`, let `m_b` be its mass and `w_b` its largest answer mass.
The admissible root bound is

`LB(g) = 1 + sum_b (2 m_b - w_b) / M`.

The root guess costs one turn. In each non-green bucket, at least one reply is
needed, and every answer other than the heaviest needs at least one more.
Roots are evaluated in ascending bound order; once `K` exact costs are known,
later roots with bounds strictly above the `K`th cost (plus a floating-point
guard) are skipped. Equal or near-equal bounds remain eligible. This preserves
the requested exact-cost prefix within the *same candidate pool*, not global
optimality, a six-turn objective, or exact costs for unrequested rows. The
shortcut is disabled when coverage is the primary ranking key or exhaustive
mode is selected.

Common positive scaling of answer masses leaves normalized probabilities unchanged
(apart from floating-point rounding); this is different from scaling score weights.
Multiplying every linear large-state coefficient by a positive constant preserves
its ordering, but scales the absolute top-to-pool-edge score gap. Legacy pool
expansion compares that gap with `pool_tight_gap_threshold` and
`pool_medium_gap_threshold`, so coefficient scale is not a redundant policy dimension.
For example, a gap of 0.04 is below the default tight threshold 0.05, whereas the
same ordering scaled tenfold has gap 0.4 and receives no gap expansion. Scaling both
thresholds restores that split-score comparison, but also changes the separate
proxy-cost branch, whose costs were not scaled. Do not normalize away this degree of
freedom or remove correlated coefficients without a matched policy ablation.

After five observations, only one guess remains. The runtime therefore overrides unlimited-horizon proxy/lookahead/exact ordering and ranks guesses by immediate solve probability, then posterior answer probability, then lexical order for deterministic ties. Information gain has no value after the final guess. This does not remove irreducible failures when several unseen, cutoff-safe dormant candidates have identical mass; recovery activation must expose those candidates early enough for previous guesses to separate them.

The danger score is a normalized weighted combination of top-posterior concentration, largest-bucket mass, largest-bucket size ratio, ambiguity pressure, and top-candidate disagreement. Registry v7 makes the posterior and candidate windows, mass and size disagreement cutoffs, and ambiguity saturation count explicit alongside the five feature weights and allocation thresholds. Its thresholds allocate computation; it is not a probability of failure. Lookahead starts with one for the root guess and adds each branch probability times the selected child reply's complete proxy cost. That proxy cost already includes the reply guess. The previous heuristic path added another unit at the child, double-counting the reply only above `exact_exhaustive_threshold` and creating a discontinuity with exact children; a hand-computed regression now prevents it. A later domain audit found that bounded child-reply pools could still admit a guess whose largest non-green bucket was the entire child state. Such a reply has no well-founded finite continuation value, so child metrics now apply the same strict-progress predicate used at the root and by exact recursion. The slow reference and a deliberately inert-guess fixture enforce the same domain. Proxy continuation cost no longer embeds an extra concentration surcharge because concentration already has its own registered score weight. Root lookahead penalties apply to four separate inputs—worst-branch posterior mass, large-bucket count, dangerous-mass count, and mass in large buckets. Approximate replies add candidate-count ratio through its own registered coefficient instead of merging it into the posterior-mass coefficient. These penalties remain heuristic and must never be labeled exact expected guesses.

### Tractable-state regret audit (2026-07-26)

The legacy scalar-cost reports below do not validate the new six-turn objective.
`search-regret --finite` accepts only the fixed-belief Fast/Strong modes and
compares their actions with a
finite exact reference under the same posterior, remaining turns and hard-mode
history. An exact global reference requires all legal root actions, exact values,
completed search and no proposal sampling. Fixed-root references independently
partition that action with raw feedback, then reuse the finite kernel for each
child at one fewer turn. This is a shared-kernel cross-check; independent exhaustive
verification remains confined to the tiny-state tests. The fixed-root reference
does not validate `finite_fast_dynamic`; use the separate same-state dynamic
reference below for bounded, explicitly selected states.

Failure regret is reported first. Expected-attempt regret is reported only when
failure values tie after quantization at `64 * f64::EPSILON`, matching the kernel's
comparison convention. An exact fixed-root value that beats the purported global
optimum is rejected as contradictory evidence. Interrupted or incomplete references
retain an explicit status and null regret. These bounded diagnostics cannot support
a claim about unexamined larger states or prospective games.

### Same-state dynamic reference (2026-09-27)

The opt-in `same-state-dynamic-regret` command requires a staged config, one
explicit development date, and a turn from one to six. It reconstructs the
artifact-free staged path through the preceding turns, then evaluates the
staged and `finite_fast_dynamic` choices at the same reconstructed dynamic
belief. Each choice is compared with a full-legal-root, dynamic finite-horizon
reference. Exact references are limited to at most six combined active and
dormant-fallback survivors. Larger states are unresolved; path reconstruction
and reference search share one wall-clock budget, so a deadline can also leave
the run without an exact result. Unresolved regret is not zero. This is a
normal-mode small-state decision diagnostic: both the choices and reference
are evaluated with `hard_mode=false`. It does not validate recursive
hard-mode routing, a whole-policy score, prospective performance, or an
independent formal proof; the fixed-belief `search-regret` diagnostic remains
separate.

`search-regret` follows deterministic historical targets with the artifact-free proxy policy until a requested posterior-size band is reached. It then evaluates the production choice, forced proxy choice, and configured bounded-lookahead choice on that identical posterior. The reference scans every allowed root guess and uses exhaustive Bellman continuation. Reports bind the executable, source, config, and data identities and record the exact observation path. Schema 2 checks the cumulative wall-clock cap cooperatively inside path collection and exhaustive search, as well as at boundaries. `complete` and `stop_reason` expose interruption; `planned_states` counts selected states and `sampled_states` counts fully evaluated rows. Summaries cover retained complete rows only, never partial values or unevaluated states. If no state completed, mean and maximum regret are JSON `null` and display as unavailable (`sampled_states=0`), not measured zero. These are diagnostic development results, not sealed-test solve scores.

The first run found that proxy ranking could return a guess whose only non-green bucket contained the entire state, giving it infinite continuation cost under fixed weights. Unlimited-horizon fixed-weight ranking excludes such non-progressing guesses, with a deliberately inert extra-guess regression. Dynamic finite search retains probes that change dormant support, even if the active set does not shrink; its transition and oracle tests cover that exception.

After that correction:

- [`search-regret-v1.json`](../benchmarks/predictive/search-regret-v1.json) evaluated 16 time-spread states with 3–8 survivors. Production, proxy, and lookahead all matched exhaustive cost on every state, apart from sub-`1e-15` floating-point noise reported as zero positive-regret states.
- [`search-regret-9-16-v1.json`](../benchmarks/predictive/search-regret-9-16-v1.json) evaluated 14 available states with 9–16 survivors. Proxy-only ranking was suboptimal on 6/14 states, with mean regret `0.340952` and maximum regret `0.899180` expected guesses. Bounded lookahead was suboptimal on 3/14 states, with mean regret `0.000154` and maximum regret `0.001724`. Production matched exhaustive cost on all 14 states.

Across both reports, bounded lookahead matched exhaustive cost on 27/30 states and its combined mean regret was about `0.000072`, versus about `0.159111` for proxy-only ranking. This evidence supports retaining the bounded lookahead candidate mixture and exact escalation: the added search nearly eliminates the large proxy error. It does not establish that each individual penalty or pool source has held-out solve benefit; those terms still require the registered ablations and rolling studies before simplification or promotion.

### Learned continuation-cost experiment (2026-08-02)

The learned proxy is a residual model, not a replacement recurrence. For row features `x`, existing proxy cost `C_proxy`, and exhaustive Bellman label `C_exact`, training minimizes

```text
sum_i (C_exact,i - C_proxy,i - beta_0 - x_i beta)^2 + lambda ||beta||_2^2
```

with optional train-only population standardization. Fitting an intercept centers the features and response even when variance scaling is disabled; the intercept is not regularized. Deterministic pivoted elimination solves the regularized normal equations and rejects a solution whose backward residual exceeds tolerance. Artifact validation binds the ordered feature schema, scaling, dataset/replay identities, and coefficients. Invalid dimensions, non-finite inputs, invalid models, or negative/non-finite predictions return an error; callers cannot silently turn invalid inference into a plausible baseline score. For `x=[1,2,3]`, `y=[3,5,7]`, `lambda=1`, the unscaled intercept fit has slope `4/3` and intercept `7/3`.

The native adapter samples reachable 3–12-survivor states from three contiguous development-only windows: training, validation, and an inner development test. State/trajectory identities must be disjoint and every held-out date must follow every training date. Each row records the complete survivor IDs and weights, date, turn, deterministic candidate guess, baseline features, and exact continuation cost. Candidate guesses mix primary proxy, entropy, worst-bucket/mass, immediate-solve, and surviving-answer coverage. This is exact for every recorded `(state, guess)` row, but it is a bounded diagnostic candidate set—not a proof that all allowed guesses were materialized.

The recorded artifact has 401 rows across 24 states. Ridge regularization was selected only on validation. Learned and baseline rankings both selected a zero-regret row on all validation/test states, but learned pairwise accuracy was worse (`0.9419` versus `0.9527` validation; `0.9712` versus `0.9773` inner test), and inner-test MAE rose from `0.002990` to `0.005397`. A same-window five-state exhaustive reference gave production, proxy, and lookahead zero regret, but that sample is diagnostic rather than a solve-quality gate. Evidence floats are canonicalized to `10^-12` absolute precision before checkpointing so semantically equal Bellman results survive JSON replay with a stable digest; this does not alter the live solver recurrence. The artifact is therefore non-promotable. Even a ranking win would still require full rolling solve, coverage, failure, latency, and memory guards before production use.

### Policy-era survival experiment (2026-08-02)

Reuse intervals are represented as half-open daily risk intervals `[entry, exit)`. Reuse contributes an event on the final risk day; a right-censored interval contributes no event. First observed appearances are left-truncated because the last use before the history origin is unknown and therefore are not labeled as reuse events. Never-used support mass is tracked separately and never converted into a censored reuse event.

Intervals crossing an editor-policy boundary are split by era while retaining an `elapsed_offset_days` value, so `t` remains days since the original last use rather than resetting at the boundary. Fold construction clips exposure to `[training_start, training_end + 1)`, converts future events to censoring, and rejects date/era mismatches. Identical `(era, elapsed-day)` rows are aggregated into weighted binomial observations; this preserves the logistic likelihood exactly while reducing allocation and fit time.

The daily hazard is a regularized logistic model over a low-degree polynomial in `log(1 + t / scale)` plus policy-era indicators. Ridge and second-difference penalties control coefficient size and time curvature. Dated inference integrates daily hazards using the era active on each risk date. The consumed June 18-July 17 outcomes may be chronological training history for later dates, but are excluded as new tuning or validation targets. This differs from the selected hand-set recovery curve and remains an experiment; output weights are heuristic prior scores, not calibrated editorial probabilities.

The historical run across the canonical 12 development folds recorded 26 fold-local reuse-event exposures; this is not a count of unique reuse events. Current reports count unique events separately, reject non-converged/zero-risk fits, and bind a canonical ordered era basis. On 360 validation games, the survival score had log loss `6.750447` and Brier `0.998692788`, versus `6.690665` and `0.998682995` for the selected logistic curve; both had 23 support gaps. The paired proxy-only solve diagnostic held the search policy and no-book condition fixed across all folds. Logistic had conditional/failure-penalized mean guesses `3.4171/3.4250` with 3 unsolved games; survival had `3.4415/3.4444` with 1 unsolved game. Survival's maximum fold p95 was `1982.4` ms versus `1689.0` ms; process peak was recorded as `166080512` bytes. Sparse event count, worse probability scores, coverage gaps, worse penalized guesses, higher latency, and the still-unrun production-search solve gate block promotion. The sealed `2026-06-18` through `2026-07-17` outcomes were not read as development targets.

## 7. Exact expected cost and lower bounds

Live API, CLI and GUI labels come from execution metadata, not a threshold guess
or the presence of a numeric cost. They separately report route, actual objective,
normal/root-only-hard/recursive-hard action scope, root coverage, selected-action
value quality, completion reason and whether root selection was established optimal
for that model. An exact value for one action is not a globally optimal root, and
an exhaustive root set with pooled child continuations is not an exact Bellman
result. Coverage overrides and penalized lookahead are labeled by those objectives.
These are predictive model claims, never formal certificates.

For normalized state `S`, exact expected cost is the Bellman recurrence

```text
G_progress(S) = {g : every non-green child S_p is a strict subset of S}
C(S) = min_{g in G_progress(S)} [1 + sum_{p != green} P(p | g, S) * C(S_p)]
```

with singleton cost `1`. The explicit action domain excludes self-recursion. This is the fixed-weight, normal-mode, unlimited-horizon recurrence, not the dynamic finite-horizon recurrence above.

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

Historical log loss around `7.23` and Brier around `0.9992` did not establish a calibrated editorial prior. GUI/CLI probabilities remain labeled heuristic scores. Calibration claims require rolling-origin reliability/ECE evidence, not only a lower log loss; old results do not validate changed code.

Benchmark `average_log_loss` and `average_brier` score each target against the
date-bounded *initial* state, before gameplay. They are not posterior-path
averages. The staged policy starts from its modeled support with separate
recovery, while fixed-belief finite modes freeze a positive-mass core/tail state.
The opt-in `finite_fast_dynamic` mode instead starts from staged's dynamic belief.
Thus two
profiles with the same TOML can have different initial probability vectors;
their score comparison is an end-to-end policy-bundle comparison unless the
effective support and weights are explicitly matched.

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

In backtest `PredictiveMetrics`, `conditional_mean_guesses` averages modeled games, including failed attempts. In `StudyMeasurement`, the same field is solved-only: its numerator is the exact sum of the 1–6 solved histogram, divided by `solved_games`. These distinct conditional populations must not be compared with each other or with an all-game score when coverage/failures differ. Study format v17 fixes the earlier conversion that multiplied the modeled-game mean by the solved-game count. The study's all-game numerator is the solved total plus `L * (unsolved_games + coverage_gaps)`.

Reported distribution statistics include the 1–6 histogram, median, p90, p95, maximum, and solved-within-three/four rates. Coverage and solve-rate intervals use Wilson score intervals.

Undefined conditional means, intervals and quantiles are `null`/`n/a`, never zero.
An empty scheduled population is an error; an all-gap population still has the
all-game penalty and explicit zero coverage, but no modeled/prior conditional
score. Canonical metrics retain additive modeled-attempt and penalized totals,
so merging unequal folds never reconstructs a numerator from a rounded mean.

Prior ranking evidence reports top-1, top-3, and top-5 recall with Wilson intervals. Ten-bin confidence expected calibration error uses the maximum prior probability as confidence and whether that top-ranked word is the observed answer as correctness. Its interval uses the same deterministic chronological moving-block bootstrap as other standalone statistics.

Benchmark evidence also replays the recorded feedback path to score the posterior
before each actual guess (turns 1–6), without rerunning search. Per-turn summaries
report both scored and unscored states: they condition on games still being played,
not all scheduled games. The all-games group and overlapping never-used, reused,
historical-only and out-of-core strata use only history before the puzzle date.
Coverage gaps remain unscored observations rather than disappearing. Persisted
rows and aggregate scores are checked for arithmetic and path consistency.
Session-book latency is null/n/a when that path is not used or benchmarked; it is
not a measured zero-latency result.

Standalone mean-like intervals use a deterministic moving-block bootstrap (`2,000` resamples, block length `7`, recorded seed). Paired comparisons resample chronological candidate-minus-baseline per-game differences, preserving dates. Negative delta favors the candidate. Win/tie/loss counts use per-game penalized values.

Overlapping standalone intervals are not a paired significance test. A candidate is promoted only when hard constraints pass first: no added coverage gaps or failures, then solve quality/calibration, then latency/memory.

## 10. Rolling-origin and sealed test

The default evaluation plan uses expanding chronological training and 30-day, non-overlapping validation folds. Features, eligibility, and artifacts for target date `d` are bounded to information before `d`.

Tuning and exhaustive-label collection use the explicit union of eligible fold
dates, not the continuous envelope between the first and last fold. Excluded
consumed windows and gaps stay out of target selection. Earlier consumed answers
may still be chronological training history for later dates; target eligibility
and feature-history availability are separate rules.

The final 30-day window is excluded from optimizer measurements and may be evaluated exactly once only after configuration and artifacts are frozen. Evidence generation, rolling comparison, all study/tuning paths, and live-config evaluation derive their boundary from one canonical plan helper and reject ranges that intersect it.

New sealed evaluations require complete unique calendar coverage before claiming
ownership, a window-scoped exclusive ledger that rejects overlaps across
candidates, and private source/window digest rechecks before publication. The
legacy consumed-window record is preserved. See [persistence](PERSISTENCE.md).

The final v20 candidate passed the complete 12-fold development guard at `360/360` and `3.1778`, versus `358/360` and `3.3000` for the previous default. The paired candidate-minus-default delta was `-0.1222` with chronological block-bootstrap interval `[-0.1722, -0.0722]` and win/tie/loss counts `69/265/26`.

After the configuration and artifacts were frozen under `sha256-v1:15cb4c86c7548dbcdf94624a8a80b93009a8ebbdcc57d228617977765d4a543a`, the sealed 2026-06-18 through 2026-07-17 window was evaluated once. The candidate solved `30/30` games with all-game mean `3.3000`, interval `[3.1333, 3.4667]`, no coverage gaps, no failures, median 3, p95 5, and maximum 5. The result does not support a `<= 3.0` claim, and this sealed window must not be reused for tuning.

The September 26 development comparisons are separate from that historical
sealed result. Across 12 allowed folds, selected staged scored `3.2000` with
`358/360` solves. An earlier-build finite Fast scored `3.5556` and `3.5639`
with `360/360` solves in two runs, but its effective initial support differs
from staged, so this is an end-to-end policy comparison, not a search-only
ablation. A final-build rerun reproduced all staged and entropy-weight
candidate game paths: the candidate scored `3.1944` with `359/360` solves;
its paired difference from selected was `-0.0056` with interval
`[-0.0278, +0.0139]`. The result does not establish an improvement or meet
the zero-failure gate. All three remain development evidence; see the
[September release ledger](SEPTEMBER_RELEASE.md) for identities and dates.

## 11. Study fidelity and promotion

The common study runner evaluates only rolling-development folds. Given initial fold count `F0`, reduction factor `eta >= 2`, and maximum `Fmax`, its deterministic fidelity schedule is

```text
F_r = min(Fmax, F0 * eta^r)
```

with the final `Fmax` rung inserted when multiplication would skip it. A candidate resumes from its saved additive fold accumulators; an already-recorded fold cannot be merged twice. At each non-final rung, candidates are ordered lexicographically by coverage gaps, failures, all-game failure-penalized mean when that stage measures solves, log loss, Brier score, latency when available, and stable candidate number. Cumulative wall-clock and process-lifetime peak memory are cooperative stop limits, not allocator-enforced hard caps. Expensive search, book building and label work poll cancellation internally; completed folds/states/profiles are checkpointed, while interrupted work still consumes elapsed budget. Process-lifetime peaks cannot rank candidates in a shared process. The best `ceil(n / eta)` advance. Candidate zero is retained as the explicit reference baseline even when it would otherwise be pruned.

The remaining roadmap studies use `F0 = 3`, `eta = 2`, and `Fmax = 12`, so their fidelity rungs are `3 -> 6 -> 12`. Each rung is a deterministic nested, time-spread subset of the same canonical 12 outer development folds; it is not a new resample or additional independent evidence. Spreading low-fidelity measurements across the development span avoids pruning solely on the oldest consecutive months. Every promoted finalist is evaluated on all 12 outer folds. Multiple optimization seeds alter candidate generation only, while `BookPolicy` rebuilds cutoff-safe artifacts for those same outer folds. Any inner chronological out-of-fold rows used to fit continuation-cost models belong exclusively to an outer fold's training data and never replace or augment its held-out validation score.

The declared process-memory cap is enforced against peak working-set bytes on Windows, Linux, and macOS at solver construction, fold/game checkpoints where available, and final latency measurement. Peak bytes are checkpointed as shared-process budget diagnostics and are excluded from guarded/Pareto ordering: later trials inherit the process high-water mark. Exceeding the cap fails the trial; candidate-specific memory comparisons require isolated matched processes. Windows is release-tested; the native macOS memory-sampler CI job passed on revision `23a73e0`, without validating the macOS GUI. Wall-clock time is cumulative across resumed fidelity rungs and includes contention while a candidate is active. The default is 7,200 seconds: the earlier 3,600-second v11 proxy screen could complete six folds but not the final 12-fold rung, so that partial state is screening evidence only.

Calibration-only studies deliberately leave guess means and latency null: zero is not a valid stand-in for an unmeasured solve objective. Their results can nominate prior candidates for later solve-policy evaluation, but cannot directly promote a solver configuration. The equal-compute strategy evidence in `benchmarks/predictive/study-strategy-comparison-v8.json` therefore reports calibration convergence and seed sensitivity, not a mean-guesses improvement.

Static grid, low-discrepancy, random, and local-refinement suggestions are deterministic functions of the declared seed, registry, and budget. They evaluate independent candidates in parallel and may use successive halving. Fold scoring intentionally omits latency while candidates share the worker pool; after the pool joins, complete 12-fold finalists receive serialized latency measurements. This prevents scheduler contention from being mistaken for candidate latency. Model-based mode first evaluates a deterministic global startup pool, then separates completed trials into the guarded best quartile and remainder. At least two completed observations are required for that split; a two-trial aggregate study therefore uses random startup for its second trial. It draws kernel proposals near the elite values and maximizes the log density ratio `log l(x) - log g(x)`. Suggestions are sequential and atomically checkpointed before evaluation, so resume preserves the ask/tell sequence. The five-seed evidence shows that both startup exploration and TPE refinement matter; neither is sufficient promotion evidence without solve outcomes.

Study domains and cohorts are explicit. `Calibration` changes prior parameters only and measures prior scores; recovery knobs are isolated in `CoverageRecovery`, where they can affect the objective. Proxy work is partitioned into `ProxyCore`, `ProxyRisk`, and `ProxySmallState`; search allocation is partitioned into `SearchRouting`, `SearchExact`, `SearchCoverage`, `SearchLookahead`, `SearchPool`, `SearchDanger`, and `SearchPenalty`. These granular stages measure rolling solve outcomes without books. Registry validation requires each cohort's domain and optimizer role to agree, and tests compare all 84 registry entries with every serialized `PriorConfig` leaf, verify that every entry changes config identity, and prove that all 78 optimizer-controlled knobs occur in exactly one granular stage. The registered values include the opener holdout shortlist, artifact freshness/rebuild cadence, reply-book candidate pool, exact second-guess coverage root pool, ambiguity cutoff, all danger weights/windows/cutoffs, two pool-expansion multipliers, six exact-pool source fractions, and the separate reply bucket-ratio penalty. Static and model-based granular studies begin with a deterministic valid one-factor perturbation for every eligible setting and reject trial counts that cannot include this sweep plus the baseline; wider proposals begin only after this coverage prelude. `ProxyRanker` and `SolvePolicy` retain their aggregate semantics for compatibility, while `Joint` is the deliberate prior/recovery/proxy/search cross-domain refinement for finalists. `--base-config` carries an exact frozen TOML result into the next stage and that canonical base is part of study identity. `BookPolicy` rebuilds isolated candidate/fold artifacts at chronological cutoffs and evaluates them in disk-only mode. The legacy proxy fitter's greedy per-field objective on one 80/20 state split is no longer an optimization path. Its unused calibration-row builder has been removed. The format-2 exhaustive-cost pipeline instead uses an independent iterative teacher that considers every legal continuation action for its fixed-weight normal-mode contract. Recorded root actions remain a declared sample. Production thresholds/pools cannot change teacher labels; row limits are checked before labeling, and resumed complete-state rows are reconstructed and verified before reuse.

Diagnostic search variants are serialized registry-validated profiles under `config/profiles/`. The offline-book migration corrected a previously hidden inconsistency: its root candidate/reply pools exceeded the declared ranges and were larger than the corresponding medium-state pools. The profile now satisfies `root_pool <= medium_pool` for both candidates and replies; this is a correctness/configuration fix, not evidence that the new values improve guesses.

Fixed experiment cohorts use the same typed values in format-v1 matrices under `config/experiments/`. Optimizer domains retain strictly positive minima for log-scaled weights. A separate diagnostic application rule permits exactly zero for float parameters so a term can be removed in an ablation; it does not permit negative values, arbitrary out-of-range nonzero values, operational/safety parameters, or configs that violate cross-field validation.

Finite-policy studies are an exception to legacy parallel trial execution: they
require `--jobs 1`, and their backtests execute games sequentially. Competing
wall-clock-limited searches would change the amount of search completed and hence
the policy being evaluated, not merely its reported latency. This guard also applies
to finite calibration studies, which may run live latency diagnostics. The diagnostic
`finite_baseline` mode is available to comparisons but excluded from optimizer choices.
The registry classifies `fallback_prior_mass` as prior calibration for fixed-belief
finite modes, giving that stage seven parameters. Reactive recovery controls are
inactive under a fixed core/tail posterior, so a fixed-belief finite recovery-only
study has no tunable parameters and is rejected. Dynamic-belief modes retain the
recovery cohort.

## 12. Artifact and identity contract

Model, word-list/pattern-table, predictive-book, formal-proof, rolling/study, and benchmark inputs use SHA-256 with domain separation and unsigned 64-bit little-endian field lengths. File fields are hashed in bounded streaming chunks and are tested against one-shot encoding. Rolling/study/benchmark provenance covers the exact current executable as well as launch-time source, tests, Cargo manifests, and data inputs; phase-boundary rechecks prevent a long command from publishing results after those inputs change. Text identities use the explicit `sha256-v1:` prefix; filename-safe predictive hashes use the same 256-bit digest under book manifest version 3. Pattern tables and formal binary artifacts have new magic values, studies use format v19, the parameter registry uses format v7, benchmark evidence uses schema v8, and rolling comparisons use schema v5. Format-2 exhaustive datasets require cross-field state/action counts and replay identity; study v19 records examined calendar games separately from recovery-cohort games. Formal objective/state contracts are version 3 and certificates version 8; generation pointers bind a coherent immutable set. Rolling final artifacts require the evaluated `top` setting and reject baseline reuse with a different value. Old or mixed formats cannot be resumed/reused as current evidence and produce a regenerate/rebuild error where they cross a persisted boundary.

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
