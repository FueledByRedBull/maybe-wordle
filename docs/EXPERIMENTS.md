# Experiments, studies and evidence

This document holds the detail that used to sit in the README: the September
2026 audit and experiment notes, the experiment and evidence tooling
reference, and the GUI behaviour and verification history. The notes are dated
history written at the time. Current status is in the
[README](../README.md), [TODO.md](../TODO.md) and the
[research close-out](./superpowers/plans/2026-10-03-subthree-research.md).

Many `benchmarks/predictive/september-*.json` artifacts named below contain
answer words and are local-only; they are not in a clean checkout.

## Retained evidence index

These tracked artifacts are kept as the audit trail of how the selected
configuration was reached. No other document links them by name.

| Path | What it is |
| --- | --- |
| `benchmarks/predictive/studies/*.json` | July 2026 `study-run` records for the calibration, coverage-recovery, proxy-core, proxy-risk and proxy-ranker stages, versions v11 through v22 |
| `benchmarks/predictive/rolling-*.json` | 12-fold rolling comparisons of earlier candidates against their baselines, March to July 2026; `rolling-selected-v20-final-20260726.json` is the promotion guard cited in the README |
| `benchmarks/predictive/study-strategy-comparison-v5.json` | The earlier format of the search-strategy comparison; `-v8.json` is the one described below |
| `config/candidates/*.toml` | Candidate configurations evaluated by those studies and comparisons |
| `config/profiles/proxy-only.toml`, `config/profiles/proxy-exact-endgame.toml` | Configurations named for the two proxy-versus-default rolling comparisons |

## September 2026 audit and experiment notes

The [September 29 audit remediation](./superpowers/plans/2026-09-29-audit-remediation.md)
corrects formal horizon optimization, dynamic-support pruning, input/persistence
boundaries, GUI request/terminal handling, and evaluation/learning contracts.
The final Windows gate passes 603 tests, 15 benchmark smoke workloads, formatting
and warning-denied Clippy. Rebuilt GUI/CLI executables are in `dist`, with checksums.
The audit rolling comparison retains a 3.1944 mean and 360/360 solves for selected
v20; it establishes no score gain or flat-three result. Measurements and historical
tables retain their recorded executable identities; the final reporting-only
regret fix is identified separately in the release ledger.
The production configuration has not been promoted or retuned. Hosted Windows,
Linux and macOS tests, binary builds and CLI smoke
[passed for `d2024bd`](https://github.com/FueledByRedBull/maybe-wordle/actions/runs/36591624931).
Native interactive GUI and real Windows UNC-share acceptance remain open;
the GUI was left closed.
Dependency checks and their limited maintenance exceptions are documented in
[DEPENDENCIES.md](./DEPENDENCIES.md).

Production remains v20; no newer configuration has passed every release guard. The declared evaluation policy in [`config/evaluation.toml`](../config/evaluation.toml) freezes development through 2026-08-26 and reserves 2026-08-28 through 2026-09-26 as an untouched seal. The consumed June 18–July 17 seal is excluded from new tuning and validation targets, but remains usable as chronological training history for later dates.

The latest bounded policy investigation fixes the known nine- and 22-survivor
hard-mode choices; all six audited states now match the exhaustive reference to
floating-point precision. It has not established a lower overall average score.
See the [screen results and limitations](./SEPTEMBER_RELEASE.md#bounded-oldnew-policy-investigation).
Two earlier validated 30-game development runs (first (`benchmarks/predictive/september-policy-30day-validated-v1.json`),
repeat (`benchmarks/predictive/september-policy-30day-validated-repeat-v1.json`)) solved
30/30 with finite versus 28/30 with staged, but finite scored 3.6000 and
3.6667 against staged's repeatable 3.4000 (failures penalized as seven).
The same executable/config/data identities produced eight different finite
paths; every divergent choice hit the 250 ms deadline. The [generated tables](./generated/september-policy-30day-validated.md)
and [repeat table](./generated/september-policy-30day-validated-repeat.md)
record the paired results. Earlier-source runs are retained; none
establishes a lower mean or justifies promotion.

An experiment-only fixed-work comparison (`benchmarks/predictive/september-policy-fixed-work-v1.json`)
and repeat (`benchmarks/predictive/september-policy-fixed-work-repeat-v1.json`)
gave identical 30-game finite paths and per-move traces with no deadline exits.
Both scored 3.5000 with 30/30 solves, against 3.6333 and 3.6667 for 250 ms
finite Fast. The paired improvement intervals include zero, and staged's earlier
3.4000 remains lower. This control improves repeatability, not release status.
The preceding-binary staged-versus-fixed comparison (`benchmarks/predictive/september-staged-fixed-work-v1.json`)
and repeat (`benchmarks/predictive/september-staged-fixed-work-repeat-v1.json`)
confirmed those scores on the same 30 dates: staged 3.4000 with 28 solves,
fixed work 3.5000 with 30. Fixed work was slower and had worse initial-state
probability scores; its paired guess-score interval includes zero. The two modes
also use different effective support/recovery semantics despite sharing one
configuration file, so this is an end-to-end policy comparison, not an isolated
search-algorithm win.

A September 26, equal-initial-belief 12-fold finite baseline versus Fast
comparison (`benchmarks/predictive/september-finite-matched-budget-current-v1.json`)
and fresh repeat (`benchmarks/predictive/september-finite-matched-budget-current-repeat-v1.json`)
covered 360 development games each at the same 250 ms budget. The baseline
repeated at 3.5389 with 354/360 solves; Fast solved 360/360 but scored 3.5694
and 3.5528 (penalty seven). Its paired mean differences were +0.0306 and
+0.0139 guesses, with both intervals spanning zero. Fast changed 20 game
paths and eight game scores between runs, while baseline paths repeated.
This isolates the finite rollout choice more cleanly than the staged comparison,
but shows no mean-score win or deterministic deadline behavior. It does not
justify changing production.

An earlier-build selected-policy 12-fold development comparison (`benchmarks/predictive/september-staged-finite-current-source-v1.json`)
and fresh Fast repeat (`benchmarks/predictive/september-staged-finite-current-source-repeat-v1.json`)
put staged at **3.2000 all-game guesses** with 358/360 solves and no coverage
gaps. Finite Fast solved 360/360 but scored 3.5556 and 3.5639, with paired
Fast-minus-staged intervals entirely above zero. Staged paths repeated; Fast
changed 13 paths and eight game scores. This is development evidence from the
preceding executable identity,
not a sealed or prospective score, and the policies start from different
effective beliefs. It supports keeping staged as the selected policy, not a
flat-three claim or a finite promotion.

A one-coefficient staged-policy 30-game screen (`benchmarks/predictive/september-v19b-staged-screen-v1.json`)
and full 12-fold follow-up (`benchmarks/predictive/september-v19b-staged-final-build-v1.json`)
tested the older v19b entropy weight. A preceding rebuild
reproduced all 360 baseline and candidate game paths and scores. It recovered one of
staged's two development failures and scored 3.1944 versus 3.2000 across
360 games, but the paired interval [-0.0278, +0.0139] includes no change and
one failure remains. This is a useful negative/inconclusive screen, not a
selected config change or a three-guess result.

An earlier-binary 12-fold terminal-rule replay (`benchmarks/predictive/september-v19b-staged-terminal-legal-current-binary-v1.json`)
uses a direct failure-first decision with two turns remaining, counting only
answers that can legally be guessed on the final turn. Selected staged
now solves **360/360 at 3.1944** all-game guesses; v19b solves 360/360 at
3.1917. Selected staged solved 275/360 by turn three (31 in two, 244 in
three); its p95 was four guesses and maximum was six. Reaching an exact
3.0000 on these same games would require 70 fewer total guesses. Only two
selected paths changed from the preceding build, both
previously unsolved. The candidate-minus-selected paired difference is
-0.0028 guesses, 95% interval [-0.0222, +0.0167], so v19b is still not
promoted. This is allowed development evidence, not a sealed or prospective
result and not a flat-three claim. Its 720 development-game paths exactly
reproduced the preceding-build replay (`benchmarks/predictive/september-v19b-staged-terminal-legal-v1.json`);
the current run took 16.2 minutes and did not evaluate the reserved seal.

A follow-up 12-fold replay (`benchmarks/predictive/september-staged-root-bound-rolling-v1.json`)
of the same selected and v19b policies reproduced all 720 game paths and
scores after an admissible pooled-exact pruning change. The final build took 8.2 minutes
on this machine; a matched first-feedback top-ten CLI request fell from
55.15 to 3.92 seconds with identical suggestions and exact costs. This is a
performance improvement, not a lower mean-guess score or proof of the
still-open six-turn objective. See the [measurement details](./PERFORMANCE.md).

The completed pre-validator-build [12-fold replay table](./generated/september-post-layout-tests-rolling-v1.md)
used CLI SHA-256 `ECFEE91B8199956E327954BD5E5BE233F07129A8C2CDE3D6F4612CA740B46B72`
over the 12 allowed development folds and recorded `sealed_test_evaluated=false`.
Selected staged and v19b each solved 360/360 with zero coverage gaps/failures at
3.194444 and 3.191667 all-game guesses; per-move p95 was 20.7277 and 20.7633 ms.
V19b-minus-selected was -0.0027778 guesses, 95% CI [-0.0222222, +0.0166667],
with 6/347/7 wins/ties/losses. The run took 508.3 seconds and peaked at
168898560 bytes (about 161.1 MiB). All 720 outcomes and paths exactly matched the
[previous replay table](./generated/september-post-dynamic-mode-rolling-v1.md).
The word-bearing per-game artifact remains local; a [redacted public copy](./evidence/september-post-layout-tests-rolling-public-v1.json)
preserves the reported outcomes and calibration numbers without answer or guess words.
This is retrospective development evidence; no promotion or flat-three claim follows.

An earlier-binary matched-belief 30-game comparison (`benchmarks/predictive/september-matched-belief-fixed-work-v1.json`)
and exact repeat (`benchmarks/predictive/september-matched-belief-fixed-work-repeat-v1.json`)
isolated search on the same frozen posterior: staged ranked at 3.2000 versus
finite fixed-work at 3.5000, with 30/30 solves and no coverage gaps for both.
A matched-belief Strong follow-up (`benchmarks/predictive/september-matched-belief-strong-v1.json`)
also lost: 3.5667 versus staged's 3.2000 on the same 30 dates, despite 30/30
solves, and used roughly six times the per-move p95 latency. Increasing the
finite search budget alone does not close the quality gap.
A separate belief ablation (`benchmarks/predictive/september-staged-belief-ablation-v1.json`)
found a lower frozen-belief point mean than dynamic staged, but the paired
interval crossed zero and calibration and latency worsened. Neither diagnostic
changes the selected policy or establishes a prospective improvement.
A later opt-in dynamic-belief finite trial also failed its 30-game screen:
3.4333 versus selected staged's 3.3333, with 30/30 solves but about 12 times
the per-step p95 latency. The selected route was restored and reproduced all
30 earlier paths; the detailed [negative result and provenance limits](./SEPTEMBER_RELEASE.md#dynamic-belief-six-turn-experiment-and-rejection)
are recorded separately from release evidence.

A reproducible opt-in [dynamic-belief finite comparison](../config/experiments/september-dynamic-finite-matched.json)
now supersedes that trial for auditability. Its five-day screen (`benchmarks/predictive/september-dynamic-finite-screen-v1.json`)
([generated table](./generated/september-dynamic-finite-screen-v1.md))
had staged and `finite_fast_dynamic` at 3.0000 and 3.2000 with 5/5 solves;
latency p95 was 21.17 and 257.33 ms. The 30-day artifact (`benchmarks/predictive/september-dynamic-finite-30day-v1.json`)
([generated table](./generated/september-dynamic-finite-30day-v1.md))
covered July 28-August 26: both solved 30/30 with zero gaps/failures, while
staged scored 3.3333 versus 3.4000 for dynamic finite. The paired
dynamic-minus-staged result was +0.0667 guesses, 95% CI [-0.1333, +0.2667],
with 4/21/5 wins/ties/losses; latency p95 was 21.24 versus 256.62 ms and
initial log loss/Brier were identical at 6.5971/0.9986. Generation took 52.10 s
and peaked at 133.3 MiB. Dynamic roots chose `olate` on all 30 games, averaging
0.004110 modeled root risk and 3.3002 expected attempts; every root hit its
deadline, and both profiles recorded five recovery steps. This retrospective
development run has `sealed_test_evaluated=false`, passes neither score nor
latency guard, and does not promote the policy. The fixed-belief-only finite
`search-regret` diagnostic is not validation of dynamic mode; production remains
unchanged. A separate opt-in `same-state-dynamic-regret` check compares both
choices against a normal-mode dynamic reference on small, explicit development
states; it is exact only up to six combined active/dormant survivors and does
not validate hard-mode routing or whole-policy performance
([scope and examples](./PREDICTIVE_MATH.md#same-state-dynamic-reference-2026-09-27)).

A later one-pass fallback-partition optimization was checked three times on
the same 30 allowed development dates. Dynamic finite scored 3.3000, 3.2667
and 3.3667 against staged's 3.3333, with 30/30 solves in every run. All
paired intervals crossed zero and finite per-move p95 stayed above 260 ms.
The first two changed-binary runs matched each other on only 27/30 paths
under the deadline;
the final rebuilt binary did not retain their apparent point-score lead.
This reduced charged finite work in those runs but is not a selected-policy change or a prospective
score claim; see the [release ledger](./SEPTEMBER_RELEASE.md).

The opt-in `staged-zero-failure-certificate` command checks a conservative
six-turn witness for selected staged moves on explicit development dates.
It does not change live play or prove the best move. A complete one-day smoke
checked four roots and certified none; a nine-day run stopped at its four-minute
cap after seven complete games and 22 roots, also with no certificates. The
partial run is not a nine-day rate or a score comparison; see the
[scope and evidence](./SEPTEMBER_RELEASE.md#september-27-staged-zero-failure-certificate-screen).

The [September acceptance plan](./SEPTEMBER_RELEASE.md) puts shared game semantics
and bounded search before further tuning. Live requests now name the puzzle date and
derive history through the preceding day; effective model identities exclude future
history. Current source has correctness changes and is not validated by the old v20
scores. The audit executables were rebuilt; the local `dist/BUILD-INFO.txt` and
`dist/SHA256SUMS.txt` identify the retained distribution. Rebuilding does not make
the experimental finite policy a release candidate.
The GUI starts without a console window. A native accessibility-tree check
found the Play controls and separate board/suggestions panels, while a locked
desktop prevented a visual/keyboard pass; that gate remains open. The finite
policy has not passed the score or release
gates.

Private benchmark diagnostics are local-only and are not included in a clean checkout.

Study v19 adds complete-fold calendar accounting and cumulative interruption checks. It retains v18's grid endpoints, unique dimensions and complete elite inheritance;
registry v7 removes the obsolete coverage child cap. It retains v17's solved-histogram
totals and exclusion of shared-process memory from ranking. Older studies are historical
evidence and cannot be resumed against the changed policy. New pattern caches use
`MWORDPT3` with a SHA-256 payload digest and regenerate older caches automatically.
Fresh evaluation and release verification remain in [`TODO.md`](../TODO.md).

## Experiment, study and evidence tooling

Predictive mode now has a separate experiment and cache surface:

- `cargo run -- predictive-ablations --from YYYY-MM-DD --to YYYY-MM-DD`
- `cargo run -- evaluate-live-config --config path/to/prior.toml --from YYYY-MM-DD --to YYYY-MM-DD`
- `cargo run -- three-guess-gap --from YYYY-MM-DD --to YYYY-MM-DD`
- `cargo run -- four-guess-openers --from YYYY-MM-DD --to YYYY-MM-DD --opener crane`
- `cargo run -- build-predictive-opener --date YYYY-MM-DD`
- `cargo run -- build-predictive-replies --date YYYY-MM-DD`

The opener and reply caches are predictive-only artifacts under `data/derived/predictive/`. Filenames and serialized identity include the predictive manifest version and full model-manifest hash.

Opener artifacts are date-specific. The predictive suggestion API now exposes three modes:

1. `LiveOnly`: no artifact or session promotion
2. `FastDiskOnly`: disk artifacts only
3. `Full`: disk artifacts plus live session fallback

For `Full` root suggestions the solver uses this fallback chain:

1. exact-date opener artifact
2. newest earlier opener artifact within 14 days
3. live session opener computation

For `FastDiskOnly`, step 3 is skipped. Reply-book and third-turn artifacts use the same newest-earlier, configurable freshness rule (`session_artifact_freshness_days`, default 14) while still requiring the same model/context identity.

`build-predictive-opener` is heavier than ordinary suggestion commands: it evaluates a bounded opener pool on a recent 30-day window, tracks four-guess tails explicitly, and validates opener switches against a previous-window holdout. If you want fast predictive GUI/root suggestions for a specific date, build the opener artifact for that date ahead of time.

The config loads from [`config/prior.toml`](../config/prior.toml). Predictive artifact identity binds that configuration together with a versioned policy id (`predictive-v1`).

Recovery behavior is also explicit. Every date-supported candidate remains in feedback filtering even when its modeled weight is zero. If feedback isolates only zero-mass candidates, predictive mode can fail loudly (`Strict`) or repair that branch with `UniformOverSupport` or `EpsilonRepair`. Future history-only words are not date-supported before their first appearance. The default, and the selected configuration, is `UniformOverSupport`.

`evaluation-plan` emits the canonical expanding-window rolling-origin folds and sealed final-test window as JSON. `study-run` runs deterministic domain studies over development folds with typed parameters, grid/low-discrepancy/random/local-refinement/model-based sampling, atomic per-fold and per-suggestion checkpoints, cooperative cancellation, safe resume, hard-constraint violations, and Pareto ranks. `--base-config <TOML>` lets each stage start from a frozen finalist instead of the mutable default. Static strategies parallelize independent candidates and use serialized, nested time-spread successive-halving rungs so early pruning sees early, middle, and late development periods; finalists still evaluate all 12 folds. Fold scoring runs without latency measurement, then complete finalists receive serialized latency measurements after the parallel pool joins, preventing CPU contention from corrupting the promotion metric. Observation-driven TPE-style search is sequential so every suggestion consumes the preceding completed trial. Trial identity binds strategy, parallelism, fold selection, fold/time/peak-working-set budgets, canonical base config, registry, evaluation plan, data cutoff, launch-time source/data content, and the exact running executable. Long evidence and study commands recheck that identity at phase boundaries and fail instead of publishing a mixed-input run. Windows, Linux, and macOS studies sample the process working set at checkpoints, store the peak in trial measurements, fail a trial that crosses `--maximum-memory-mb`, but exclude the shared-process peak from candidate ordering; isolated runs are needed for candidate-specific memory comparisons. `tune-prior` uses the common prior-only calibration runner and applies an additional solve-quality guard before returning a complete TOML config. `fit-proxy-weights` is a compatibility shortcut for the common `proxy-ranker` stage; it changes only registered proxy-domain knobs and scores them on rolling all-game solve quality instead of the removed greedy 80/20 coordinate search. Evidence, rolling comparison, studies, tuning, and `evaluate-live-config` share the same canonical development/sealed boundary; ordinary development commands cannot evaluate the sealed window. `parameter-registry` emits all current predictive, book, recovery, operational, safety, and manual settings; only declared hyperparameters are optimizer-controlled.

Study format v19 and registry format v7 bind typed cohorts and canonical SHA-256 config/registry/data/code identities into provenance. Prefer the coherent stages `proxy-core`, `proxy-risk`, `proxy-small-state`, `search-routing`, `search-exact`, `search-coverage`, `search-lookahead`, `search-pool`, `search-danger`, and `search-penalty`; `proxy-ranker` and `solve-policy` remain aggregate compatibility stages. Registry tests compare all 84 entries against every serialized `PriorConfig` leaf, prove that every entry changes cryptographic config identity, and prove that all 78 optimizer-controlled knobs occur in exactly one granular stage. This includes formerly hidden opener-holdout, artifact-freshness and danger posterior/candidate windows, mass/size disagreement cutoffs, and ambiguity saturation; `session_reply_pool` controls reply-book construction, and `second_guess_coverage_pool` is no longer clamped to 24. The ambiguity cutoff, normalized danger features, two candidate-pool expansion multipliers, six exact-pool source fractions, and separate reply bucket-ratio penalty are explicit study parameters. Static and model-based granular studies first generate one deterministic, config-valid perturbation for every eligible knob and reject a trial count too small to include that sweep plus the baseline; wider proposals begin only after this coverage prelude. Solver work runs on explicitly sized 8 MiB-stack threads, including the custom Rayon study pool, so deep exact branches do not inherit platform-default worker stacks. The default cumulative per-candidate wall-clock cap is two hours; pre-v19 study checkpoints are historical because measurement semantics or earlier search/latency protocols changed.

The exact predictive recurrence prunes with a weight-aware admissible lower bound rather than the former uniform-count bound. Skewed-prior and zero-mass-branch fixtures protect the correction, and probability concentration ignores zero-mass-only buckets while structural coverage diagnostics retain them. The 2026-07-19 audit also removed an extra unit that double-counted heuristic lookahead replies above the exact threshold and replaced the small-state proxy's uniform-count table with a weighted one-step cost. Those ranking changes require fresh rolling evidence; older generated scores remain audit history until regeneration completes. See [`docs/PREDICTIVE_MATH.md`](./PREDICTIVE_MATH.md) for the formulas and scope.

`search-regret` provides a separate tractable-state check against exhaustive Bellman search. Its versioned reports follow deterministic artifact-free proxy paths, bind source/executable/data/config identity, and retain the exact observations for replay. The first audit exposed a proxy choice that could leave the entire state unchanged; unlimited-horizon fixed-weight ranking excludes non-progressing guesses. Dynamic finite search must retain probes that can change dormant support even without shrinking the active set. After that fix, bounded lookahead matched exhaustive cost on 27/30 sampled states across the 3–16 survivor bands, with combined mean regret about `0.000072`; proxy-only ranking had combined mean regret about `0.159111` and reached `0.899180` on one state. This supports keeping bounded lookahead, but it is a math diagnostic—not a sealed-test or mean-guesses claim. See [`search-regret-v1.json`](../benchmarks/predictive/search-regret-v1.json) and [`search-regret-9-16-v1.json`](../benchmarks/predictive/search-regret-9-16-v1.json).

Current regret schema 2 checks its cumulative deadline inside path collection and
exhaustive search. `complete` and `stop_reason` distinguish a finished diagnostic
from interruption; `planned_states` counts selected states and `sampled_states`
counts fully evaluated rows. Interrupted reports retain those complete rows only,
and their summaries do not stand in for the unfinished population.

For the experimental six-turn policy, use `search-regret --finite --config
config/candidates/september-finite-fast.toml --from 2026-07-28 --to 2026-08-26
--output benchmarks/predictive/september-finite-regret.json` (one command). Add
`--hard-mode` for recursive hard-mode legality. The default audit is limited to six
reachable states and 60 seconds. It separates the runtime estimate from exact
fixed-move and global reference values; unresolved references have null regret,
not an invented zero. These are shared-kernel development diagnostics, not an
independent proof or a new solve-quality score. Its finite reference is
fixed-belief-only and must not be treated as validation of `finite_fast_dynamic`;
use the matched dynamic-belief evidence above for that mode.

Both commands described next, and the code only they used, were removed on 2026-10-12 after research closed; they last exist in commit `7539f30`. The retained result files are unchanged.

The 2026-08-02 learned-model audit added two development-only commands. `learned-proxy-experiment` samples three chronological, trajectory-disjoint splits, records exact continuation costs for a deterministic proxy/entropy/worst-bucket/solve-probability guess mixture, fits a standardized residual ridge model on train rows only, selects its regularization on validation, and reports an untouched inner development test. Exact-state work is atomically checkpointed after every state; resume rejects different source/config/executable/sampling identities, reports ETA, and preserves a stable semantic dataset digest by canonicalizing evidence precision. The fresh Rust 1.97 release run produced 401 exact rows across 24 states in `8.275` seconds, plus about `1.4` seconds for the same-window production/proxy/lookahead reference audit. Baseline and learned rows both had zero top-choice regret on validation and test, but pairwise accuracy fell from `0.9527` to `0.9419` on validation and from `0.9773` to `0.9712` on test; inner-test MAE also rose from `0.002990` to `0.005397`. The five-state production/proxy/lookahead reference happened to have zero regret for all three policies, which is reassuring but far too small for promotion. The learned model is therefore not promoted. See [`learned-proxy-experiment-v1.json`](../benchmarks/predictive/learned-proxy-experiment-v1.json) and its replayable [`learned-proxy-dataset-v1.json`](../benchmarks/predictive/learned-proxy-dataset-v1.json).

`survival-experiment` fits a fold-local, policy-era-aware discrete-time reuse model with right censoring, explicit never-used mass, left truncation, regularized smooth time effects, and era-preserving elapsed-time offsets. Identical day/era exposures are aggregated as weighted binomial rows without changing the likelihood. Across all 12 development folds (360 games), the Rust 1.97 release run completed in `81.768` seconds and found only 26 fold-local reuse events. Survival log loss was `6.750447` versus `6.690665` for the selected logistic curve; Brier was `0.998692788` versus `0.998682995`, with the same 23 prior-support gaps. A bounded paired proxy-policy solve audit covered all 360 games without predictive books: the logistic baseline scored `3.4250` failure-penalized mean guesses with 3 unsolved games, while survival scored `3.4444` with 1 unsolved game; conditional means were `3.4171` and `3.4415`. The survival path also had the higher maximum fold p95 (`1982.4` ms versus `1689.0` ms). This audit isolates the prior while staying bounded; it does not replace the required production lookahead/exact gate. The survival model is not promoted. See [`survival-experiment-v1.json`](../benchmarks/predictive/survival-experiment-v1.json).

Both commands explicitly exclude the sealed `2026-06-18` through `2026-07-17` window, emit periodic progress, bind inputs and model artifacts cryptographically, and leave production v20 unchanged. A learned artifact cannot become production evidence from ranking or prior calibration alone: it still needs zero-gap/zero-failure rolling solve improvement plus latency and memory gates.

The release performance profile covers the full 2,358-answer proxy root plus replayable 15-answer lookahead and pooled-exact states. Warm suggestion latency was `36.724 ms`, `94.153 ms`, and `189.173 ms` respectively on the recorded Windows/AMD system; process peak working set reached `83.0 MiB`. The dedicated allocator benchmark also records CPU time, process cycles, allocation calls/bytes, page faults, cold/warm ratios, executable/config/input identity, and explicit measurement limitations. See [`docs/PERFORMANCE.md`](./PERFORMANCE.md) and [`release-performance-v1.json`](../benchmarks/predictive/release-performance-v1.json).

`book-policy` performs cutoff-safe optimization: each candidate/fold gets an isolated artifact namespace, opener/reply artifacts are rebuilt from history available at the training cutoff and each 14-day freshness boundary, and validation runs in disk-only mode. Cancellation and cumulative time/process-memory guards reach the inner book and simulated-game searches; completed work is retained on interruption. These are cooperative stops, not OS-enforced hard caps. `joint` remains artifact-free because its registered space excludes book parameters; book finalists enter only an explicit final refinement cohort.

Special diagnostic configurations are data, not hidden code branches. [`config/profiles/aggressive-three-guess.json`](../config/profiles/aggressive-three-guess.json), [`config/profiles/offline-book.json`](../config/profiles/offline-book.json), and [`config/profiles/wide-pools.json`](../config/profiles/wide-pools.json) are versioned parameter overlays parsed and validated by the same registry used for studies. The migration exposed invalid legacy pool ordering; serialized profiles now keep root candidate/reply pools within declared bounds and no larger than their medium-state counterparts. The old flattened ablations were removed because they silently rewrote `manual_weights`; manual word overrides remain a separate auditable layer.

Fixed benchmark and ablation cohorts are also declarative. [`config/experiments/development-evidence.json`](../config/experiments/development-evidence.json) defines seven generated-README baselines, including an immutable previous-release config, and can bind a safe repository-relative base config before typed overlays. [`config/experiments/predictive-ablations.json`](../config/experiments/predictive-ablations.json) defines the baseline/wide-pool combinations, including weight mode, model variant, artifact policy, and typed parameter overlays. Exact-zero float values are accepted only through the diagnostic-profile path so entropy ablations can disable terms without changing the strictly positive log-search domains.

Non-optimizer search diagnostics are declarative too. [`config/experiments/diagnostic-suite.json`](../config/experiments/diagnostic-suite.json) owns the three-guess rescue profile and root/reply limits, default four-guess opener tournament, hard-case category count and scan/cutoff values, and evidence/evaluation/study latency sample budgets. Suite v2 removes the inert book `forced_suggestion_top` setting; forced simulations always execute the selected policy's next action. The shipped suite is schema-validated and tested. These settings no longer survive as disconnected constants in solver code; all promotable parameter search remains in the typed Rust study runner.

The current equal-compute prior-calibration diagnostic gives every strategy eight candidates × twelve folds (96 candidate-fold evaluations). Lower is better:

| Strategy | Best log loss | Best Brier | Coverage gaps | Interpretation |
| --- | ---: | ---: | ---: | --- |
| grid | 7.169984 | 0.999187423 | 23/360 | tiny improvement |
| low discrepancy | 7.170026 | 0.999187426 | 23/360 | tiny improvement |
| random | 7.166967 | 0.999144443 | 23/360 | best static strategy at this seed |
| local refinement | 7.168935 | 0.999173901 | 23/360 | useful after global exploration |
| model-based portfolio | **7.063526** | **0.998947748** | 23/360 | best shared-seed result; sequential |

Across five model-based seeds, best log loss had median 7.063526 and range 6.695866–7.121388. The TPE suggestion itself won three seeds; on the other two, its deterministic global startup pool won. The selected route is therefore multi-seed global startup plus observation-driven refinement, followed by hard-constraint-safe rolling solve evaluation and local refinement—not promotion from calibration alone. These runs did not measure guesses, change the selected solver config, or open the sealed test. The machine-readable record is [`benchmarks/predictive/study-strategy-comparison-v8.json`](../benchmarks/predictive/study-strategy-comparison-v8.json); the earlier format-v5 diagnostic is retained only as an audit trail.

The strongest calibration-only candidate was rejected for failures, and a follow-on `CoverageRecovery` study found a zero-failure threshold-4 candidate. Later feature-algebra and parameterization audits superseded that screening result. The final v20 configuration passed the replacement 12-fold rolling guard at `360/360` and `3.1778`, then scored `30/30` and `3.3000` on the once-only sealed test. The older [`rolling-prior-recovery-threshold4-lookahead-audit-20260719-v2.json`](../benchmarks/predictive/rolling-prior-recovery-threshold4-lookahead-audit-20260719-v2.json) remains an audit record, not current promotion evidence.

The previous Python/Optuna path is not an evidence source for promotion. [`benchmarks/predictive/legacy-optuna-archive.json`](../benchmarks/predictive/legacy-optuna-archive.json) deterministically preserves 33 completed historical trials from three local SQLite databases, including source SHA-256, parameters, and reported metrics; six unfinished trials are counted and ignored. Those runs lack current provenance, guarded objectives, and resource budgets. Rebuild or verify the archive with `py -3 scripts/import_optuna_archive.py [--check]` on Windows (or `python3` elsewhere), and re-evaluate any interesting configuration through `study-run`/`rolling-compare`.

`rolling-compare` evaluates a named candidate over every development fold and can safely reuse a prior default baseline only when the complete plan and canonical default TOML match. `benchmark-evidence-docs` and `rolling-evidence-docs` update or verify the generated README sections from their JSON artifacts.

`freeze-prospective --config <candidate.toml> --comparison <rolling.json>`
creates an immutable freeze only for a full-coverage, zero-failure development
winner whose paired 95% interval is entirely below zero. It records the next
30-day UTC window beginning after the freeze date, separately from the
reserved, untouched August 28-September 26 seal. Freezing requires complete
daily history from the development cutoff through the UTC freeze date and
binds a digest of the later training rows.
`evaluate-prospective --frozen <freeze.json>`
can run only after that window's final UTC day has elapsed and exact history
coverage, source/config and pre-window-history identity, and once-only marker
preflight pass. The first evaluation attempt acquires the one-window
reservation for this release;
interrupted attempts cannot be rerun as fresh tests. The raw report defaults
to ignored `target/diagnostics/`; keep it and the markers local. On 2026-10-12
unchanged selected v20 was frozen for the window 2026-10-13 through 2026-11-11
([freeze record](../benchmarks/predictive/prospective-frozen-v1.json)); it has
not been evaluated.

`benchmark-evidence` writes versioned JSON plus a generated Markdown fragment and rejects excluded evaluation dates. Use `--rolling-folds` instead of `--from/--to` for the exact noncontiguous development folds, and `--matrix` to select a separate experiment matrix. Checkpoint v5 binds the selected dates, matrix contents, resolved profile configurations, time/memory ceilings, and effective Rayon worker count; older checkpoints cannot resume. Long runs emit flushed `profile-start`, per-game, and `profile-complete` records with completed/total work, elapsed time, and an evolving ETA. Profiles run sequentially; finite-policy games also run sequentially so their wall-clock search budgets do not compete. Legacy games may use Rayon. Cumulative resource limits are polled within games, search, books and latency probes as well as at profile boundaries; partial profiles are not published as completed evidence. The selected v20 configuration remains the production default; its historical sealed score is `3.3000`, not a three-guess claim or validation of the experimental finite policy.

Backtests keep coverage gaps in all-game denominators and report both an explicitly conditional mean over modeled games and a failure-penalized all-game mean. Mean intervals and paired comparisons use deterministic chronological block bootstrap samples; coverage and solve rates use Wilson intervals. Experiment output also includes log loss and multiclass Brier score. These are diagnostics for a heuristic prior, not evidence that its probabilities are calibrated.

Set `MAYBE_WORDLE_EVIDENCE_TIMING=1` for stderr-only per-profile phase
timings during `benchmark-evidence`; these do not alter saved evidence or
checkpoints. The September 29 full seven-profile matrix reached its 20-minute
ceiling and retained only complete profiles; the separate two-policy rolling
comparison completed. Earlier successful full-matrix timings belong to their
recorded builds, not the audit build (see the [release ledger](./SEPTEMBER_RELEASE.md)).

The equations, domains, exact-versus-heuristic boundaries, coupling audit, and verification map are documented in [`docs/PREDICTIVE_MATH.md`](./PREDICTIVE_MATH.md).

## GUI behaviour and verification history

The notes in this section were written in September 2026 about the builds of
that time. Where they say native keyboard, visual or accessibility acceptance
"remains open", the October 12 acceptance at the end of the section is the
later state.

The Rust GUI is a predictive-first workspace with Play, Policy, Diagnostics, and secondary Formal panels. One worker shares a replaceable pending slot: obsolete queued work is discarded and predictive searches cooperate with cancellation when a newer request arrives. Pending requests clear old recommendations; stale responses cannot replace the current state. Each request uses the explicitly selected configured, Fast, or Strong policy, without an automatic alternate-policy preview. Results report the actual route, value quality, action scope, and stop reason. Play includes keyboard feedback codes (`0/1/2` or `b/y/g`), Enter-to-apply, a six-row board, compact history, a suggestion inspector, and a filterable/exportable full candidate list. Board tiles use contrast-safe, font-safe `A`, `P`, and `C` markers for absent, present, and correct feedback; the earlier Unicode marks rendered as empty squares with some Windows fonts. Text scaling and a stacked layout support narrow windows. The puzzle date, history cutoff, and persistent incomplete-history notice make the information boundary explicit. Native keyboard and assistive-technology acceptance remains open.

The guess/feedback controls now wrap at the minimum window width; a regression
checks that all five feedback buttons stay visible. Earlier builds received
limited native checks, but the audit build has deliberately remained closed.
Its fresh native visual, keyboard and accessibility acceptance remains open.

Missing derived data no longer prevents the window from opening. The setup surface offers explicit public-history sync, local build, retry, progress/error reporting, and cooperative cancellation at phase/request boundaries. Formal artifacts remain optional; their absence does not block predictive play.

The earlier native usability pass covered initial suggestions, symbolic feedback normalization, Enter-to-apply, alternative inspection, undo, wide and 713-pixel compact layouts, full-workspace scrolling, diagnostics/provenance layout, and an isolated missing-data recovery fixture. The fixture also verified that a failed local rebuild remains recoverable and reports an actionable error instead of closing the app. This is historical UI evidence; the September worker/profile changes still require a fresh native visual pass.

Finite profiles enforce hard-mode legality throughout simulated continuations. Legacy
staged search filters legal next moves but estimates later replies in normal mode;
its legacy costs are not exact hard-mode policy values.

CLI and GUI predictive mode are intentionally conservative:

- the GUI uses the unified predictive suggestion API in `FastDiskOnly` mode by default
- the CLI predictive path uses `FastDiskOnly` by default
- pass `--live-fallback` to CLI `suggest` or `solve-interactive` to opt into `Full`
- `FastDiskOnly` means "use disk artifacts, do not promote a live session fallback"
- `Full` means "use disk artifacts first, then allow live-session fallback if the artifact chain is missing"
- `recovery mode` means the solver had to repair a zero-mass state before it could keep ranking guesses
- if predictive artifacts are missing, the GUI should make that explicit instead of implying a richer artifact-backed result exists

### October 12 native acceptance

Run through UI Automation and sent keystrokes against the packaged Windows
build after the `eframe` 0.36.2 update (GUI SHA-256 `c954cf6e7219ac07...`),
with the desktop unlocked. The same pass on the earlier `eframe` 0.32.3 build
of that day gave the same results.

- All 38 buttons, text fields, combo boxes and sliders in the default window
  expose a non-empty name. The five
  feedback tiles are named by position (`Letter 1: Absent` and so on). Before
  the fixes nine inputs had no name and the five tiles shared one.
- No control extends past the window edge at 1180 by 820 or at the 560 by 620
  minimum, at 100% and at 135% text.
- The sort indicator reads `asc` or `desc`; the bundled fonts have no arrow
  glyphs, so the earlier arrows rendered as boxes.
- Keyboard: typing a guess and feedback and pressing Enter applies the row and
  returns focus to the Guess field, so the next row can be typed without
  navigating. Tab moved through the first 21 controls in visual order, and
  Space activates a focused button.
- By design, Ctrl+Z (undo a row) and Esc (clear the row being typed) act only
  when no text field has focus; the on-screen hint says so. Esc inside a text
  field leaves the field, after which Ctrl+Z undid a row.
- Window captures at each size show the light theme and legible text with the
  wgpu renderer.

Not done: a screen-reader pass with Narrator or NVDA.
