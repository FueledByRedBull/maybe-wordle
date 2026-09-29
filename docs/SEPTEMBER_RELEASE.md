# September predictive release acceptance plan

Status: audit remediation locally verified; release/promotion acceptance below
remains open. This plan consolidates the six source-audit and
rollout notes supplied on September 7, 2026, against revision `1d607151` and the
existing uncommitted August work. Historical v20 scores do not validate changed code.
The release goal includes every requirement below. A checkbox requires recorded
implementation and verification evidence, not merely a proposed fix.

The September 29 comprehensive audit is tracked in the
[48-item remediation ledger](superpowers/plans/2026-09-29-audit-remediation.md).
Its corrections supersede older source-level assumptions here, not the identities
or outcomes of historical measurements. In particular, formal depth/expectation
optimization, dynamic dormant-support pruning, shared terminal handling, explicit
metric populations, sealed-window ownership and independent exhaustive labels
have changed. Current verification and bounded measurements are recorded below.
The native GUI remains closed at the user's request; headless geometry
and state tests do not close the visual/keyboard acceptance gate.

Unlinked `benchmarks/predictive/*.json` filenames below identify local-only raw
diagnostics, not files guaranteed in a clean checkout. Public redacted evidence
is linked where available; its aggregate arithmetic can be checked without
redistributing per-game answer and guess words.

## September 29 audit delivery

The 48-item audit ledger records the implemented corrections and their regression
evidence. Production remains selected staged v20; this audit does not authorize
an experimental policy promotion or claim a flat-three average.

The final Windows gate passed Rust 1.97 format, warning-denied locked Clippy,
603 tests and all 15 benchmark smoke workloads. Final review reproduced empty
search-regret aggregates displayed as numeric zero; optional JSON values and
population-labeled CLI output now pass zero/mid/final-deadline regressions and
the refreshed full gate. The native GUI has not been opened. Hosted three-platform CI,
native Unix persistence/FIFO and Windows UNC acceptance are not local passes.

The selected-policy first-feedback profiler completed all six fresh/warm top
1/5/10 calls in 844-1364 ms; each solver clone allocated 290 bytes, and a 10-ms
cooperative cancellation stopped at 16.49 ms. These single-call diagnostics,
allocation definitions and executable identity are in [PERFORMANCE.md](PERFORMANCE.md).
They do not establish a population p95, an OS-cold startup time, or a matched
speedup against the old executable.

The nine-date seven-profile [timing screen](generated/september-audit-timing-screen-v1.md)
finished in 19.58 seconds. Its selected-policy 3.0000 mean is not a full-fold
score. The subsequent 12-fold seven-profile run stopped at the configured
1,200,021-ms cumulative cap, with a process peak of 229,695,488 bytes. Six complete
profiles were retained in the local checkpoint; the selected disk-artifact profile
was incomplete, and no final full-matrix report was published. The checkpoint and
timeout log remain at `target/evidence-checkpoints/september-audit-seven-profile-v1.json`
and `target/audit-work/seven-profile.log`. This is an incomplete experiment, not a
successful release matrix or a reason to bypass its resource limit.

The retained complete artifact-free staged profile solved all 360 games at
3.194444 guesses. The proxy-only and proxy-with-exact-endgame profiles scored
3.244444 and 3.200000 respectively, also with 360 solves; those are within-run
diagnostics, not causal attribution to a particular audit fix. A separately
completed [paired rolling comparison](generated/rolling-evidence.md) supplies
the published full-fold selected-policy evidence: selected v20 scored 3.194444
and exploratory v19b 3.191667, both 360/360 with no gaps or failures. The paired
v19b-minus-selected interval is [-0.022222, +0.016667], with 6/347/7 wins/ties/losses;
v19b is not promoted. All 720 paths and outcomes match the earlier post-layout
replay. Current measured per-move p95 was 28.29/27.25 ms, not a matched speedup
claim. No consumed or reserved seal was evaluated by these audit runs.

These measurements bind audit CLI SHA-256
`9490EAA144801EEDBD0809016100C135EFB635047CCFA5B73893D16A48D5454D`.
They precede the final reporting-only correction to empty search-regret summaries;
no gameplay ranking, data, configuration or benchmark calculation changed in that
correction. The delivered executable's separate identity is in `dist/SHA256SUMS.txt`.
Do not relabel the measured binary as that later build or infer a new speedup.

The final release GUI and CLI are retained in `dist` with checksums and build
notes; their Windows subsystems are 2 (GUI, no console) and 3 (CLI). CLI help,
both exact CI documentation commands, public redaction and local Markdown links
pass. Verified Cargo profile cleanup removed 8.2 GiB of debug and 1.1 GiB of
release output while preserving dist, data, evidence and checkpoints. Publication
was authorized after this local handoff; hosted acceptance is recorded by the CI
run for the published commit, not inferred from these Windows checks.

## Decision contract

The primary product is online, artifact-free six-turn predictive Wordle. Choose legal
moves by modeled failure probability first, then expected attempts. Report failures,
coverage gaps, solved-guess distribution, probability of solving by turn three, and
all-game mean separately; score failures and gaps as seven in that last metric.
Empirical release guards require full coverage and zero observed failures, and do not
establish a guarantee about future editorial choices. Fast and Strong budgets must
evaluate the same objective. Formal proof remains optional research.

## 1. Correctness before studies

- [x] Define one public puzzle date and exclusive usable-history boundary. Align
  live CLI/GUI, evaluator, model support/weights, local-date default and book identities.
  Test that appending/changing same-day or future answers cannot affect effective
  prior, support or artifact-free recommendations; yesterday may affect them. Raw
  provenance hashes may change. Audit vocabulary/order effects, not only date filters.
- [x] Repair coverage ordering so lexical/proxy ties cannot shadow deeper values.
  Include immediate wins in success probability and distinguish success by total
  turn three from structural completion within three additional moves.
- [x] Remove the small-child proxy discontinuity across threshold sizes, test uniform
  and skewed distributions and partition refinement. Keep heuristic estimates distinct
  from proved pruning bounds.
- [x] Make generated, partial and shipped configuration defaults agree and preserve
  all serialized leaves on round-trip. Preserve existing complete configurations.
- [x] Share normal/forced-prefix/opening/book simulation semantics, including dormant
  support, failures/gaps and exact additive outcome totals. Keep target information
  private from the policy; only the feedback generator knows the target.
- [x] Verify existing solved-histogram aggregation and canonical fallback ordering fixes;
  invalidate incompatible older evidence/checkpoints explicitly.

## 2. Search the actual game

- [x] Finite policies carry remaining turns and updated hard-mode constraints through
  every child, candidate pool and memo key. Same survivors with different legal
  actions or horizons must not share invalid values. Final-turn selection bypasses
  unnecessary recursion.
- [x] Make finite-policy modeled transitions agree with live support/weight updates.
  Evaluate a fixed positive core/tail mixture with conditioning, or explicitly
  represent dormant-support activation. Cover historical-only, out-of-core,
  manual-zero and zero-mass cases.
- [x] Implement finite one/two-turn terminal decisions for the declared objective and
  compare with an independent tiny-state exhaustive normal/hard-mode oracle, including
  duplicate-letter legality and deliberately different optimal moves by horizon.
  The dynamic two-answer direct shortcut now requires empty dormant fallback
  support; a lower activation threshold can otherwise reactivate answers after
  the first miss. A failing-before/passing-after threshold-one oracle test guards it.
- [x] Finite policies evaluate shortlisted moves by grouped rollout of a complete
  inexpensive baseline policy; include its own move. Compare with the corrected
  baseline at identical prior, rules, dates, book policy and time budget. Keep
  exact manageable endgames.
- [ ] Make the selected staged policy choose by the six-turn failure-first objective
  before the final two turns, or select a finite policy that meets all paired
  quality, calibration and resource guards. The current staged terminal rule
  corrects late decisions only; its earlier unlimited-horizon ranks and
  normal-mode continuation estimates do not satisfy the declared objective
  or recursive hard-mode contract. Its two-turn shortcut maximizes surviving
  answer mass per feedback bucket; a surviving answer itself satisfies all
  earlier and new hard-mode clues, so no separate final-reply filter is
  mathematically needed. A hard-mode final-reply regression now checks this
  with duplicate-letter feedback, but does not repair the earlier-turn
  continuation. Historical finite Fast
  comparisons were
  worse on mean guesses, so merely switching the default is not accepted.
- [x] Use feasible incumbents and admissible lower bounds to prune unfinished exact
  branches. Lower bounds are internal dominance proofs, never candidate action values;
  public/cache labels distinguish exact values, feasible upper bounds and unevaluated
  heuristics. Deduplicate full-feedback-equivalent partitions only in normal mode,
  where future legality is unchanged; hard mode does not deduplicate.
- [x] Add inner-loop/recursive cancellation and deadlines, returning a valid completed
  incumbent and an explicit budget status. Obsolete GUI work releases worker capacity.
  Cancellation is cooperative; setup/sorting and OS scheduling are not strict real-time
  guarantees.

## 3. Runtime evidence and practical UI

- [x] Create a fixed-date reachable-state suite: opener, OLATE/10001 stall, coverage/proxy
  threshold bands, ambiguous clusters, fallback, hard mode and late turns. Measure first
  usable result and completed refinement separately, distribution/max latency, CPU and
  isolated finalist memory. The bounded suite has eight states and three samples per profile;
  its p95 is the maximum of three samples, not a population p95, and it excludes native GUI
  paint latency. See [`PERFORMANCE.md`](PERFORMANCE.md).
- [ ] Profile before kernel changes. Remove unused predictive small-state-table work and
  false bound claims. Measure invariant weighted-log precomputation, immutable geometry
  sharing, ID-based rows, scratch reuse and partial selection where applicable. Record
  rejected optimizations; do not claim translated C++ timings as Rust measurements.
  A September 26 in-process profile isolated the first-feedback `OLATE/10001`
  stall in pooled exact ranking (about 9 ms preview versus 54-55 s full CLI).
  A root partial-cost shortcut did not help and was removed. The retained
  admissible bucket bound reduced a matched top-ten full CLI request to 3.92 s
  from 55.15 s with identical words/costs. A fresh 12-fold replay matched all
  720 preceding paths and scores and took 491.2 s versus 969.5 s. See
  [`PERFORMANCE.md`](PERFORMANCE.md) and the
  new evidence (`benchmarks/predictive/september-staged-root-bound-rolling-v1.json`).
  This is a latency optimization of the existing unlimited-horizon route, not
  completion of the selected policy's six-turn objective.
- [x] Keep one resource budget across trials and solver parallelism. Preserve the measured
  constant-liar rejection unless a new matched experiment warrants reopening it.
- [x] Implement clear puzzle date/history cutoff, next guess and alternatives, feedback,
  candidates, undo and hard mode. Add useful time-budget profiles and honest provisional,
  bounded and model-exact status; retain keyboard/accessibility and separate research views.
  The legacy preview now says "configured ranking"; CLI and GUI expose that legacy
  continuation costs relax future hard-mode legality, with targeted regressions.
- [ ] Verify native visual, keyboard and accessibility behavior in the desktop app. This
  remains unverified. On September 23 the Windows helper recovered and exposed the
  running distribution app's accessibility tree: the main controls had names, but
  30 empty, noninteractive board cells appeared as unnamed buttons. Source now
  renders board cells as display-only labels; an AccessKit regression confirms
  that filled/draft cells remain readable and no board cell is a button. The
  visual capture was obscured by another active window, and the app window
  disappeared before a keyboard pass. A later first-guess geometry regression
  reproduced a 564-pixel tile overflowing its column; fixed-width tiles now
  pass the regression. A minimum-window regression then reproduced the separate
  guess/feedback control row clipping its fifth button; wrapping that row fixed
  it and all 20 GUI tests passed. The reported wide-window overlap remains
  unconfirmed without a native visual/keyboard pass.

## 4. One conditional optimization and evaluation pipeline

- [x] Preserve grid endpoints and select unique dimensions without stride collisions.
  TPE refinement inherits a complete elite configuration before perturbation. Test actual
  proposed values and combinations, not merely hashes.
- [x] Keep Rust authoritative for config/registry, serialization, simulator, metrics and
  promotion. Remove exact duplicate parameters with explicit migration; condition inactive
  recovery/search/book controls. Audit weight/score scaling and absolute-gap interactions.
  Use ablations before removing merely correlated features. Do not add another optimizer.
- [x] Establish a corrected baseline before new tuning. Compare simple uniform,
  used/unused, recency-bucket, logistic and regularized-prior families under the same policy.
  Report proper-score baselines and never-used/reused/historical-only/out-of-core strata,
  including useful posterior calibration after feedback.
- [x] Diagnose regret on reachable states where production can lose quality. Treat larger
  bounded reference values as intervals or approximations, never fabricated exact labels.
- [ ] Run conditional cohorts in cost order, only Pareto finalists in small multi-seed
  joint refinement, then all common development folds and seven-profile evidence. Freeze
  exact executable/config/data/rules/date/book/budget identities and per-game outcomes.
- [x] Complete the seven-profile incumbent/baseline development matrix under the
  roughly 20-minute ceiling. The rebuilt current-executable 12-fold rerun covered
  2,520/2,520 profile-games in about 11.1 minutes with zero failures and gaps; it does not
  substitute for a future qualified finalist's paired promotion gate.
- [x] Validate atomic identity-bound per-profile checkpoints and progress/ETA with fresh,
  resumed and isolated-result agreement plus rejection of changed sources/config/data,
  evaluation plans, policies and overlapping completed units.

## 5. Information boundaries and maintenance

- [ ] Preserve development cutoff August 26 and exclude both consumed June 18-July 17
  and declared August 28-September 26 windows as tuning/validation targets;
  consumed answers may remain chronological training history for later dates.
  Record frozen-candidate dates
  and window consumption. A candidate selected after a window starts must not be called
  prospectively frozen before it. Declare a later prospective confirmation if necessary.
  Future outcomes are an external dependency, never a reason to fabricate completion.
- [x] Validate upstream history words/dates before persistence; preserve useful fetch/decode
  error categories. Audit cache payload integrity and the independent oracle trust boundary;
  input hashes and in-range bytes alone do not prove an uncorrupted feedback payload.
- [x] Verify removal of the obsolete Python entry point with deterministic archive retained,
  untracked reproducible formal table, Rust 1.97 manifest/CI/docs contract, official checkout
  upgrade and dependency-alert status. Resolve actionable findings without exposing secrets.
- [x] Validate the resident-set sampler on native macOS before claiming platform validation.
  The `macos-memory` CI job passed on `23a73e0`; this does not validate the macOS GUI.
- [x] Inspect whether supported headless platforms need CI coverage/GUI feature separation.
  The README explicitly packages Windows GUI/CLI executables but does not
  promise a GUI-free Linux or macOS build. CLI arguments avoid launching
  `eframe`, although the dependency and GUI module remain unconditional.
  The added Linux CI job checks Rust compilation with that dependency; it
  does not link or run a headless CLI, and its hosted result awaits a
  user-authorized push. No feature split is justified by the current
  documented contract; this inspection is not a Linux runtime claim.
- [x] Keep the small independent formal oracle and proof tests. Larger proof work requires
  a concrete feasibility improvement; tiny-universe exponential extrapolation is not a
  theorem about Wordle strategy construction.

## 6. Release gate

- [x] Rust 1.97 formatting, warnings-denied Clippy, all-target tests and focused semantic,
  model/config/checkpoint tests pass; review failures rather than weakening assertions.
- [x] Release-decision evidence arithmetic, generated docs, Markdown links and
  historical archive verify. TODO lists only unfinished work; README, PLAN
  and math/performance docs distinguish current-build from historical evidence.
  The current-build staged/v19b pair was independently replayed; other retained
  historical JSON is not claimed to validate changed code.
- [x] Complete whole-repository Ponytail audit before git actions; integrate only
  justified minimal cleanup, preserve attribution/user changes and remove stale
  process narration. The September 26 audit found no release-relevant cut worth
  invalidating current evidence; serialized/public-schema candidates are deferred
  pending an explicit compatibility decision.
- [x] Rebuild both Windows executables into ignored `dist`, verify metadata, hashes and
  GUI/console subsystems, smoke-test the runnable distribution, then safely remove
  rebuildable release artifacts after required evidence/checkpoints are preserved.
  Native visual/keyboard acceptance remains a separate unchecked gate above.
- [x] Inspect the entire intentional diff and repository hygiene. Do not stage, commit or
  push until explicitly authorized. Record any prospective-holdout/native-platform blocker
  honestly; neither a negative optimization result nor an unmet 3.0 aspiration is a defect.

## Evaluation readiness (September 8)

Acceptance re-audit: upstream normalization is shared by read, write and fetch paths;
requested-date validation precedes persistence, and serialization is revalidated before
atomic replacement. The predictive cache binds both ordered vocabularies and its
payload digest. Malformed-history, requested-date and in-range-payload mutation tests
pass. Fetch, HTTP-status, decode and validation errors retain separate context.
This closes the upstream/cache requirement, not authentication of arbitrary artifacts
or automatic verification of formal runtime binaries.

The regret-diagnosis requirement is also complete: bounded normal/hard reachable-state
reports below expose concrete shortlist loss and distinguish shared-kernel references
from the independent toy oracle. Fixing or explicitly accepting those measured search
limitations remains a promotion decision in TODO.md; diagnosis is not a claim that
the optimizer is globally exact. The current 320-test full gate closes the code-check
item for this worktree and must be rerun if implementation changes.

The matrix integration regression now also resumes a genuinely partial two-profile
checkpoint through `build_development_evidence_with_selection`: it retains the first
profile byte-equivalently (including timings), computes the missing profile, and
matches fresh-run canonical totals and per-game outcomes without duplicate units.
The existing complete replay, standalone-profile agreement and changed-matrix
rejection remain in the same passing test. These deterministic toy checks validate
resume mechanics; they do not replace the pending seven-profile empirical run.

The pipeline acceptance items are backed by the shared Rayon study pool and the
finite-policy `--jobs 1` guard, finite inactive-parameter conditioning, complete
registry/serializer tests, explicit obsolete-child-cap migration, and the absolute-gap
score-scaling regression. The constant-liar decision record (`benchmarks/predictive/constant-liar-decision-v1.json`)
preserves the rejected parallelization experiment; its conclusion remains
unchanged. Evidence checkpoint identity binds executable/source, effective configurations, matrix and
evaluation selection; mutation/prefix/overlap tests reject incompatible reuse. These
closures concern implementation and bounded verification, not completion of the
seven-profile release measurements or a new production promotion.

Checked search requirements are backed by the passing finite-kernel and predictive
characterization suites: `finite_values_match_independent_recursive_normal_and_hard_oracle`,
`two_turn_objective_can_choose_a_different_move`,
`fixed_posterior_conditions_without_repair_or_tail_activation`,
`baseline_is_state_local_and_conditioning_invariant`, and
`finite_baseline_replay_matches_grouped_normal_and_hard_policy_value`.
Coverage ordering is covered by `equal_coverage_defers_to_appended_search_cost` and
`solve_by_three_coverage_counts_green_and_one_final_guess_per_bucket`.
All six ranking tests and 13 finite-kernel tests pass after removing the dominated
child-entropy calculation. The new all-leaf nondefault serialization round-trip
also passes, alongside the existing canonical/partial-default checks.
The public predictive suite now passes all 11 tests, including
`predictive_public_books_ignore_same_day_future_history_and_storage_order` and
`puzzle_replay_ignores_same_day_and_future_history_including_new_primary_words`.
These exercise eligible vocabulary, metadata, recommendations and exact-date
opener/reply identities across the puzzle-date boundary. Inclusive `as_of` helpers
remain explicit adapters; the public puzzle date uses the previous day's cutoff.
The new finite force-in-two/top-limit and post-search cancellation tests also pass.
Cancellation consolidation and recursive-incumbent tests passed the integrated
320-test gate recorded below; later changes require fresh verification.
Sampling repairs are covered by `bounded_samplers_keep_dimensions_unique`,
`grid_sampler_retains_the_upper_endpoint`,
`effective_parameter_deduplication_ignores_explicit_defaults`, and
`tpe_refinement_retains_one_elite_full_effective_configuration`. No truly duplicate
configuration parameter was removed: the obsolete coverage child cap belonged to
the removed recursive coverage algorithm. Existing TOMLs can still be read, while
current serialization omits that obsolete field. This does not require a new
configuration migration framework.
`obsolete_coverage_child_cap_is_dropped_without_changing_active_settings` now
verifies the explicit read/drop/write rule: an old child-cap key is accepted,
active settings remain identical, current serialization omits the key, and reload
preserves the effective config. The focused Rust 1.97 regression and formatter pass.
The matched chronological result below completes the baseline comparison requirement,
not the separate promotion gate. All release checks must run again after integration.

The canonical `evaluation-plan` command was checked against the local data: it
produces 12 chronological 30-day validation folds. The first 11 cover July 3, 2025
through May 28, 2026; the final fold is July 28 through August 26, 2026. The consumed
June 18-July 17 window and August 28-September 26 seal remain excluded from validation.
No new solve-quality result is implied by this plan check.

Before study execution, distinguish three comparisons: corrected legacy policy
(unbounded diagnostic), finite Fast versus Strong (different compute budgets), and
baseline-policy versus rollout improvement at a matched finite budget. The current
seven-profile matrix is a legacy/book comparison and does not substitute for that
last experiment. The runner now supports `--rolling-folds`, selecting the exact
noncontiguous validation ranges, and `--matrix` for a separate family comparison.
Checkpoint identities bind those ranges, matrix contents and resolved profile configs.
The outer wall-clock/memory budget is checked between profiles, not a preemptive
deadline for a legacy search. Finite requests additionally enforce their own budgets.

The inexpensive rollout baseline is now state-local: it rescans the highest-mass
legal surviving answers instead of retaining the originating root's probe list.
The conditioning-invariance regression and all 13 finite kernel tests pass. The
state-local latency artifact is preserved separately; it does not replace solve
evidence. Experimental comparison configs are
`config/candidates/september-finite-baseline.toml` and
`config/candidates/september-finite-fast.toml`. Both inherit the same tested
canonical defaults; resolved configurations must be bound in the comparison
artifact. Baseline-only execution and raw-feedback replay checks now pass for both
normal/hard mode at two- and three-turn horizons. It uses the same 250 ms/reply-8
cap as Fast, bypassing root proposal and exact refinement. Incomplete values remain
heuristic. The first chronological comparison is preserved as
`benchmarks/predictive/september-finite-matched-budget-v1.json`: 3.4889 versus 3.5389
penalized guesses, zero versus six failures, paired delta CI -0.1417 to +0.0417.
It ran games concurrently, so wall-clock-limited searches contended for CPU. It is
a contention diagnostic, not interactive-play release evidence. The shared finite
backtester now runs games sequentially; a calling-thread regression passes. The fresh
`september-finite-matched-budget-sequential-v1.json` comparison records 3.5028 versus
3.5389 penalized guesses across 360 games, zero versus six failures, and paired delta
CI -0.1333 to +0.0611. Its mean improvement is not established by this interval;
neither the production default nor a prospective seal was promoted/consumed.
Active fallback telemetry in finite runs means
the fixed core/tail mixture, not reactive dormant-support activation.
The legacy `weighted_staged_no_artifacts` matrix entry now explicitly selects
`staged`, preventing a finite base config from silently changing its labeled policy.
Final rolling artifacts now reuse checkpoint structural/arithmetic validation before
reuse, freeze and rendering. Fractional-metric JSON round-trip testing exposed a
parser precision mismatch; enabling `serde_json`'s existing `float_roundtrip` feature
preserves exact recomputation checks without introducing permissive float tolerances.

The requested read-only Ponytail audit found two small consolidation opportunities:
duplicated CLI/GUI root discovery and thread-pool bootstrap (about 30-35 lines), and
the two README generated-section replacement functions (about 13-15 lines). It found
no supported dependency removal. These are optional simplifications, not release
correctness blockers; the audit made no deletions or unrelated bootstrap changes.

## Dependency alert triage (September 8)

The official checkout repository reports
[v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) as its latest
release (July 20, 2026); its `v7` action manifest selects `node24`. The pending CI
update therefore removes the old Node 20 action runtime. The deterministic Python
optimizer archive also passes `py -3 scripts/import_optuna_archive.py --check`.

GitHub exposes seven open Dependabot alerts at the time of this check; automatic security updates
are disabled. Secret scanning and push protection are enabled. No GitHub settings
or alert dispositions were changed. Static triage before the approved lockfile updates:

| Alert | Locked dependency | Verdict / evidence |
| --- | --- | --- |
| [7](https://github.com/FueledByRedBull/maybe-wordle/security/dependabot/7) | quinn-proto 0.11.14 | Not actionable in current build: no active dependency path across all targets; HTTP/3 is not enabled. |
| [6](https://github.com/FueledByRedBull/maybe-wordle/security/dependabot/6) | rustls-webpki 0.103.9 | Not actionable in current configuration: the production HTTP client does not load CRLs. |
| [5](https://github.com/FueledByRedBull/maybe-wordle/security/dependabot/5) | rand 0.8.5 | Not actionable with current features: required `log` feature is absent. |
| [4](https://github.com/FueledByRedBull/maybe-wordle/security/dependabot/4) | rand 0.9.2 | Not actionable in current build: no active dependency path across all targets. |
| [3](https://github.com/FueledByRedBull/maybe-wordle/security/dependabot/3) | rustls-webpki 0.103.9 | Needs review, priority 1: active TLS verification path; a suitably misissued signed wildcard certificate is an unresolved exploit precondition. |
| [2](https://github.com/FueledByRedBull/maybe-wordle/security/dependabot/2) | rustls-webpki 0.103.9 | Needs review, priority 2: active TLS verification path; a suitably misissued signed certificate with URI constraints is an unresolved exploit precondition. |
| [1](https://github.com/FueledByRedBull/maybe-wordle/security/dependabot/1) | rustls-webpki 0.103.9 | Not actionable in current configuration: CRL matching is not reached. |

Evidence: `Cargo.toml` enables reqwest blocking/json/rustls-tls, not HTTP/3;
`src/data.rs` constructs the only production client without CRL configuration.
The installed reqwest 0.12.28 source defaults to an empty CRL list and selects
ordinary root verification for that case. Offline locked all-target dependency and
feature graphs establish the inactive Quinn/rand 0.9 paths and rand 0.8 features.
No applicable SECURITY.md was found. The remote TLS boundary is supported by the
sync implementation, but exploitability of alerts 2/3 was not dynamically tested.
Verdict confidence is high for the excluded paths and medium for the two review
items. These are configuration-specific dispositions, not a claim that the locked
packages are patched. Approved precise updates now select rustls-webpki 0.103.13,
quinn-proto 0.11.15, rand 0.8.6 and rand 0.9.3. The lockfile diff contains only these
versions/checksums and their version-qualified dependency references. Locked metadata
and `cargo +1.97.0 check --locked -p rustls-webpki` pass. A standalone reproducer
linked against the retained 0.103.9 library reproduces the malformed-CRL panic
(exit 101); the same source linked against 0.103.13 rejects that CRL, truncated DER
and empty DER without panic (exit 0). All nine `data::tests` pass with the patched
dependency. The upstream dependency's own unit suite cannot run as a non-workspace
package with unavailable dev-dependencies; no dependencies were added to work around
that. Local HTTP fixtures do not independently validate every TLS/name-constraint
advisory. Independent source/feature review found no supported-path regression; the
rand 0.8 patch changes a rare entropy-reseed failure to a panic on its active Unix
accessibility path. After final status/test integration, locked full gates passed (275 library,
26 CLI, five integration and 11 predictive-characterization tests, plus benchmark
smoke checks), including expanded source/config/data identity mutations and standalone
profile agreement. Formatting and warnings-denied all-target Clippy passed again.
Open remote alerts are not locally closed.

## Additional development diagnostics (September 8)

Integrated regressions now verify shared normal/forced-prefix/opening outcome totals,
including six-turn failures and coverage gaps. The policy receives feedback, not the
target. `forced_prefix_and_normal_simulation_share_dormant_support_and_turn_limit`
checks the additive classifications and failure/gap penalties. GUI tests cover a
superseding spawned-worker request and cancellation after configured ranking starts.
`recursive_exact_incumbent_survives_budget_without_poisoning_next_search` checks
that an interrupted completed incumbent is an upper bound, not an exact-cache entry.
The score-scaling regression demonstrates that absolute-gap expansion makes a
global coefficient scale nonredundant; no arbitrary normalization was introduced.

`metrics_keep_gaps_in_all_game_denominators`,
`solve_totals_exclude_failures_and_penalize_every_unsolved_game` and
`fold_measurements_resume_without_double_counting` cover histogram/additive accounting.
`dormant_primary_and_secondary_answers_remain_index_sorted` covers fallback ordering.
Schema/identity rejection tests pass; the real older
`development-2026-06-17.json` artifact was also rejected by the CLI documentation
consumer with a missing `config_fingerprint` error, without rewriting its evidence.
History read/write/request-date tests and `in_range_payload_corruption_rebuilds_cache`
cover the upstream-data and predictive pattern-cache boundaries. SHA-256 detects
payload changes; it is not authentication of externally supplied artifacts.

The independent formal oracle and certificate mutation tests remain passing and
limited to tractable universes. Formal runtime binary loading is a separate trust
boundary: loading `state_values.bin`/`policy_table.bin` alone does not run independent
certificate verification. These binaries must not be treated as independently
verified merely because parsing succeeds. Expanding formal runtime binary hardening
is separate from the predictive pattern-cache checksum work.

Fresh finite-regret diagnostics are recorded in
`benchmarks/predictive/september-finite-regret-normal-v1.json` and
`benchmarks/predictive/september-finite-regret-hard-v1.json`. Each scanned 12 of the
30 permitted July 28-August 26 games, yielding one normal-mode and three hard-mode
states with three or four survivors. All four references resolved with zero failure
and attempts regret. These small, shared-kernel checks are not independent proofs,
large-state coverage or prospective solve-quality evidence.

The eight-trial finite calibration screen is recorded in
`benchmarks/predictive/september-finite-calibration-v1.json`. Two trials completed
and six were pruned across the declared expanding-history folds. The best trial
reduced log loss from 7.1596044169 to 7.1594918424 and Brier from 0.9988634674 to
0.9988634655. These are tiny prior-only changes: solve metrics were not measured,
and the candidate is not promoted. Subsequent source changes make this historical
screening evidence, not validation of the final executable.

The first five-family run was interrupted during its second profile. Its completed
logistic profile solved 360/360 development games with mean 3.525 in 222.6 seconds
of game evaluation, but an unrelated cold session-book probe added 55.274 seconds.
The partial identity-bound checkpoint is archived as
`benchmarks/predictive/september-prior-family-interrupted-v1.json` before target
cleanup. This is not a completed five-family comparison. Artifact-free and finite
profiles now omit this unused book probe and report its timings as null/n/a, not zero.
Fresh evidence must use the changed code identity and a new checkpoint.

The fresh five-family screen completed in 1,124.76 seconds (18 minutes 45 seconds),
using a new identity-bound checkpoint and all 12 allowed development folds.
`september-prior-family-evidence-v1.json`
and its [generated report](generated/september-prior-family-evidence.md) retain
per-game outcomes, paired comparisons and per-turn/stratum calibration. Logistic
scored 3.5083 penalized guesses with 360/360 solved; uniform 3.7778, used/unused
3.5194, recency buckets 3.6417 and regularized frequency 3.9444 (one failure).
Every profile covered all games. Used/unused's paired interval includes zero;
its small measured latency difference is not established as a resource improvement.
None of these alternatives displaced logistic. This is a five-prior development
screen, not the pending seven-profile release matrix or a production promotion.
Subsequent status-label and isolated-profile regression changes are not represented
by this screen's captured executable identity; final release checks remain separate.

Final integration also tests standalone-profile agreement against the combined
matrix: effective config/fingerprint, per-game outcomes and canonical metrics match.
Timing and shared-process memory are intentionally not equality assertions.
Proposal-reservation truncation now sets `proposal_sampled` even below the support
sampling cap; a deterministic node-budget test rejects an exact-reference claim.
The CLI reports this flag alongside the completion reason.
Published benchmark JSON is compact to avoid tens of thousands of formatting-only
diff lines; all records remain present and the generated Markdown stays readable.
Fresh isolated v2 baseline/preview/Fast/Strong profiles now share verified executable,
input and eight-workload identities. See [the timing table](PERFORMANCE.md#isolated-september-phases)
for full-request medians, including OLATE/10001, and explicit small-sample/GUI-paint
limitations. They do not replace chronological solve-quality evidence.

The fresh seven-profile matrix probe was stopped under the user's roughly 20-minute
ceiling. At 43/2,520 games (43/360 in `previous_release_790ec2d`), elapsed time was
48.1 seconds and the throughput ETA was another 2,768.4 seconds: approximately
47 minutes total. This early projection is not a measured complete-run duration;
later profiles have different costs. The owned process was explicitly stopped before
any complete profile/checkpoint or final evidence artifact was published.
The final drained progress was 60 games at 69.2 seconds, still projecting about
48 minutes total. The intentional termination returned exit 1; no final files exist.
Completing this matrix requires a larger authorized runtime or measured runtime improvements;
the five-prior screen must not be substituted for it.

The v2 isolated root traces exposed a proposal-order bottleneck: the baseline chose
`abuse`, Fast chose `abled`, and Strong chose `ablet` in all three samples, while
the existing full proxy preview at the same August 1 cutoff ranked `olate`, `reais`
and `aiery`. The bounded proposal scan visits dictionary order before its reserve
cutoff, biasing partially scanned roots toward early alphabetical guesses. A bounded
state-local proposal-order experiment now schedules unique-letter Bernoulli variance
before full proxy partitions (finite policy identity v3). Existing v2 timings and
five-family outcomes are retained as pre-change evidence, not final validation.

The fresh [v3 matched-budget comparison](evidence/september-finite-preordered-public-v1.json)
completed all twelve development folds, 360 games per policy, without opening the seal.
Baseline scored 3.5389 with six failures; preordered Fast scored 3.5833 with zero failures,
both with full coverage. Candidate-minus-baseline was +0.0444, paired 95% interval
[-0.0639, +0.1556], with 95 wins, 156 ties and 109 losses. Candidate folds took about
18-20 seconds each. This is not a mean-guess improvement and does not justify promotion.
The older Fast result of 3.5028 used another executable, so it is historical context,
not a controlled ordering-only comparison. At the August 1 root, Fast now evaluates
`raise` but still selects `abled`: the former has lower modeled attempts but higher
modeled failure probability. Lexical-looking selected words alone do not prove a bug.

Rust 1.97 verification after this experiment passed formatting, warnings-denied Clippy,
276 library, 26 CLI, five integration and eleven predictive characterization tests,
plus benchmark smoke checks. Independent read-only review found no objective or
hard-mode legality change; it identified the existing uninterruptible support sort
as a limit on strict wall-clock cancellation claims. Native UI and final release
acceptance remain pending.

The September development binaries are now copied into ignored `dist` with fresh
`BUILD-INFO.txt`; these are not a completed release or predictive promotion.
GUI size is 10,857,472 bytes, SHA-256
`E0BC355C62A83A19CDA3109788E2AF0FAD6466378B4C4231828477CD0068A89A`, PE subsystem 2.
CLI size is 15,103,488 bytes, SHA-256
`92D87182445B3AE047B8862DBC05E9E872530059EAEC6C0567A3D4590A409FEC`, PE subsystem 3.
CLI help and a finite Fast suggestion succeeded from `dist`, resolving repository
inputs correctly. The GUI created a responding, input-idle window, but the hidden
process smoke did not close within ten seconds; only the owned test process was
terminated. This does not validate visual interaction or clean user-initiated exit.
Repeating the lifecycle check after explicitly waiting for the `Maybe Wordle` main
window title succeeded: responding window after 0.27 seconds, close request accepted,
normal exit code 0 within five seconds. The earlier input-idle-only check was not
sufficient readiness evidence; no exit bug is established. Visual interaction remains
unverified.
Keep repository `config` and `data` alongside the distribution. Cleanup of `target`
is deferred until remaining experiments no longer need its executables/checkpoints.

## Conditional study and larger-state audit

The finite small-state cohort (`benchmarks/predictive/september-finite-small-state-v1.json`)
evaluated thresholds 12 and 4 serially on all twelve development folds (720 games).
The original run completed both trials in 455.0 seconds combined: threshold 12
scored 3.5861, threshold 4 scored 3.5833, both with full coverage and zero failures.
Prior scores were identical; p95 latency was 262.09 versus 262.02 ms. Threshold 4
ranked Pareto 0 in this screen, but saving one guess total with wall-clock search
does not establish a reproducible gain. No production config was changed.

An immediate completed-study replay exposed a status bug: the early halving rungs
reset completed trials to `running`, despite all twelve fold measurements and latency
remaining present. The persisted artifact retains those measurements but its current
statuses reflect that failed replay. No best-config file was emitted. Preserve this
negative resume evidence; repair the state transition and verify replay on a multi-rung
fixture without hand-editing measured outcomes or bypassing source identities.
The replay regression now reproduces the pre-fix failure (zero completed trials
instead of four), and passes after excluding `Complete` trials from early-rung
halving transitions. It compares the full saved trial state, measurements, elapsed
times, selected config and summary across a two-fold static-study replay. The old
real checkpoint remains untouched as negative evidence and is not eligible for
cross-source resume. Parent rerun and warnings-denied Clippy pass.
The full post-fix gate passes 319 tests (276 library, 26 CLI, six integration,
eleven predictive), benchmark smoke checks, formatting and generated-document
verification. Rebuilt development `dist` now contains GUI 10,855,936 bytes/SHA-256
`7332B68E0526F3301BB7FDD7DF977981FDFC13CA85EFDF42B095ACF3BAE7CB2B` and CLI
15,101,952 bytes/SHA-256
`5D8AE3264F0C97EE2B1A8089D667BD09B9FD9878161E6A7346FAE4E2EC34925E`, retaining
subsystems 2/3. These supersede the earlier distribution hashes above; empirical
study/profile artifacts still identify their pre-fix executables.

Fresh bounded regret audits use the same development-only July 28-August 26 window:
normal mode (`benchmarks/predictive/september-finite-regret-medium-normal-v1.json`)
resolved three states with 7, 9 and 11 survivors in 6.6 seconds, all with zero regret.
Hard mode (`benchmarks/predictive/september-finite-regret-medium-hard-v1.json`)
resolved six states with 8-22 survivors in 24.6 seconds. Its maximum failure regret
was 0.0012701; among the five states with equal optimal failure risk, maximum attempts
regret was 0.95482 (mean 0.19096). A nine-survivor state had an exact fixed-root value
but omitted a better root from the shortlist; a 22-survivor state also lost modeled
failure quality. These are concrete search-quality gaps, not deadline crashes.
The references share the finite kernel; only toy tests supply an independent oracle.
No large-state proof, sealed evaluation or promotion follows from these diagnostics.
Manual Strong-mode replay on the nine-survivor state attains the reference attempts
value (2.01768 with zero modeled failure). On the 22-survivor state it still selects
`spike` with modeled failure 0.0038104 and an upper-bound continuation value, despite
reporting `Complete`. This confirms a shortlist/continuation limitation rather than
merely insufficient deadline duration; `Complete` never implies a global optimum.

The five-trial proxy-risk cohort (`benchmarks/predictive/september-finite-risk-v1.json`)
then completed 1,170 total games in 732.8 seconds of recorded trial time. Baseline
and one finalist completed all twelve folds; two variants were pruned after three
folds and one after nine. All observed games had full coverage and zero failures.
The finalist reduces only `proxy_weights.large_bucket_count_w` from 0.1782 to
0.06890947643930216 relative to finite Fast. It scored 3.5444 versus 3.5889 baseline
(16 fewer guesses across 360 games), with identical prior scores and serialized
p95 latency 261.87 versus 262.27 ms. It is Pareto rank 0 in this study, not a
statistically confirmed or production-promoted winner. Shared process peak memory
is diagnostic, not a candidate-specific advantage.

Replaying the completed real study preserved the checkpoint byte-for-byte,
retained two completed/zero running trials and the same selected result, and
exported [the experimental config](../config/candidates/september-finite-risk-v1.toml).
This provides real-run confirmation of the multi-rung replay fix. Only this
provisional finalist is eligible to seed later joint refinement; production
`config/prior.toml` remains unchanged.

The bounded joint follow-up used two seeded-random searches from that sole Pareto
finalist: seed 20260909 (`benchmarks/predictive/september-finite-joint-random-20260909-v1.json`)
changed `trap_size_threshold` to 17; seed 20260910 (`benchmarks/predictive/september-finite-joint-random-20260910-v1.json`)
changed `proxy_small_state_lower_bound_threshold` to 29. Each compared its proposal
with the risk finalist, retaining Fast mode and the same development plan. Both
proposals were pruned after three folds (90 games), with means 3.5556 and 3.5111;
these partial means must not be compared directly with twelve-fold scores.
The risk finalist completed all twelve folds in both runs at 3.5417 and 3.5444,
with full coverage and zero failures. Combined recorded trial time was 567.5 seconds
for 900 games. Neither proposal justified changing the saved finalist.

The initial local-refinement dry run generated identical first proposals for both
seeds, so it was not counted as multi-seed evidence; its unevaluated checkpoints
are rebuildable inspection files under ignored `target/studies`. Seeded-random
proposals were inspected before timing and no outcome-based reseeding occurred.
Both completed real joint checkpoints replayed byte-identically, with unchanged
selected result and terminal counts, before JSON compaction. This is a small
development screen, not an exhaustive joint search or prospective confirmation.
The fresh paired finalist comparison (`benchmarks/predictive/september-finite-risk-paired-v1.json`)
completed twelve development folds (360 games per policy): Fast baseline 3.5833,
risk finalist 3.5417, with full coverage and zero failures for both. The paired
delta is -0.0417 guesses (95% bootstrap interval -0.0889 to +0.0083), with
52 wins, 270 ties and 38 losses. This interval includes no improvement; the
development reuse and wall-clock-sensitive search also prevent a prospective
claim. Isolated candidate resource measurements are recorded in
[the performance report](PERFORMANCE.md): matched Fast root/OLATE medians remain
about 261/256 ms and process peak about 115 MiB; finalist Strong is about 2 seconds.
All eight workload identities and non-config inputs match across separate processes.
This is bounded fixed-state evidence, not population p95 or a memory improvement.
The remaining release guards still apply; production is unchanged.

The latest verification passes 320 tests (277 library, 26 CLI, six integration,
eleven predictive), benchmark smoke checks, Rust 1.97 formatting and warnings-denied
Clippy. The config-override profiler was also built and exercised in release mode,
including default identity and missing-path rejection without report publication.

Development `dist` was refreshed after the config-migration regression build:
GUI 10,857,472 bytes/SHA-256
`C2B734E473EE69EC828B6076BB8382C012AAE37D177337A26B542D329F66A0AC`; CLI
15,103,488 bytes/SHA-256
`E8F919610FFF63ACF30EFCCEA346B782DA22D288E85B0625BBC49D3E9D641399`.
PE subsystems remain GUI=2 / CLI=3. CLI help and responsive GUI main-window
startup/normal close pass. The first CLI smoke truncated its output pipe and
returned nonzero; capturing output before truncation passed. This is lifecycle
verification, not native visual interaction or final release acceptance.

The follow-up cancellation audit reproduced a forced-poll bug at the proposal
reservation boundary: the poll returned before observing cancellation and left
the reason unset. The shared control now checks scheduled/forced cancellation
before reservation and node-budget exits. The regression failed before the fix
and passes afterward, together with all seventeen finite-kernel tests. An explicit
memo-key regression also separates horizons, hard-mode histories and rule modes,
while confirming safe history reuse in normal mode. Cooperative sorting/setup
limitations remain; this does not establish a strict wall-clock deadline.
The deterministic GUI regression now holds a dequeued request inside its cancellation
callback before enqueueing the replacement. It verifies an explicit `Cancelled`
response followed by successful completion of the newer request on the single worker.
The synchronization hook is compiled only for tests; no production callback branch
or worker abstraction was added. Together with the recursive-incumbent and zero-budget
tests, this verifies capacity handoff and honest fallback status, not strict real-time
scheduling. The integrated gate now passes 327 tests (282 library, 28 CLI, six
integration and eleven predictive), formatting, warnings-denied Clippy and benchmark
smokes. Both release binaries were rebuilt and the distributed CLI help/GUI lifecycle
smokes pass. Current GUI: 10,852,352 bytes, SHA-256
`8D2D5E611AFF8D559B0B42276D38FBC4624A958D52D67CDE95B9BBA7A5F798B9`;
CLI: 15,099,904 bytes, SHA-256
`943D335C68AF26AD1EB5AD4F6F57979958FF117244A8A1F11E8036B66239A72E`.
PE subsystems remain 2 and 3. Earlier empirical artifacts retain their own executable
identities; this cancellation correction is not a new solve-quality measurement.

Repository hygiene pass: the 48 new JSON evidence files were compacted with the
standard JSON formatter, reducing their total from 3,520,716 to 2,996,866 bytes.
Parsed JSON values were compared before/after for every file; no measurement,
configuration, provenance or negative outcome was dropped. Generated rolling and
prior-family documentation still verifies. `dist` and `target` remain ignored;
the intentionally removed formal pattern table is ignored after its pending Git
deletion. No files have been staged, committed or pushed.

Final diff review is partial, not yet a release sign-off. Locally reviewed groups:
manifest/lockfile/CI, evaluation and experiment matrices, the production config
deletion, predictive public date/response types, book call-site adaptations,
research date-bounding, artifact-identity changes and the removed small-state
benchmark. These match the intended migration and retain the production tuning
values; CI's referenced rolling artifact and the Rust 1.97 contract are present.
The independent reviewer handles disappeared before results were delivered, so
their intended larger runtime/evaluator reviews are not counted as completed.
Native macOS execution and visual GUI acceptance remain unverified.

The local runtime review additionally covered the model-weight and ranking diffs
and legacy search cancellation/bound propagation. Experimental prior modes use
date-filtered observations; weighted smoothing and manual weights are explicit.
The removed entropy term is dominated by the retained count heuristic, with an
equivalence regression; that count heuristic remains explicitly non-admissible.
Coverage-only comparison now defers ties to the subsequent search-cost comparator.
Recursive scratch ownership is restored before propagating cancellation errors,
and incomplete recursive costs are not inserted into memo caches. No new defect
was identified in these reviewed changes. Legacy continuation still relaxes future
hard-mode legality and is labeled accordingly; it is not the finite policy contract.

State/online integration review confirms that finite requests freeze eligible core
and tail mass before feedback, preserve dormant/manual-zero support, and condition
without later repair. Public puzzle dates become exclusive history cutoffs; older
inclusive-cutoff helpers explicitly advance the request date. Force-in-two filtering
precedes the requested result limit. Fast/Strong use the documented 250 ms/2 s
controls and retain one honestly labeled legal action when evaluation is cancelled.
This review found no new defect in these paths; it does not remove the documented
setup/sorting cancellation limits or substitute for the pending UI/evaluator review.

The bounded evaluator review found one additional sampler defect: a valid two-trial
model-based aggregate study could enter TPE with one observation, creating an empty
elite set and panicking on modulo zero. The startup threshold now requires at least
two observations before fitting elite/remainder densities, reusing the existing
deterministic random sampler for the second trial. The regression reproduced the
panic before the one-line fix and now passes for proxy, search and joint aggregate
stages; all 29 study tests pass. No other defect was established in that evaluator
review. The UI review separately identified puzzle-date/cutoff warning/display
inconsistencies and stale registry counts. History-range warnings now compare the
preceding-day cutoff (a puzzle after the last synced answer is not falsely stale),
interactive predictive output names both dates, and the GUI Policy panel labels
the puzzle date correctly and derives version/leaf/tunable counts from the registry.
Focused warning-boundary, date-formatting and registry-summary regressions pass.

Cleanup remains incomplete: the guarded removal of rebuildable `target/debug`
was rejected by the execution policy before execution. Subsequent inspection
confirmed the directory, release binaries and study checkpoints remain present;
the two `dist` SHA-256 values above are unchanged. No alternate deletion mechanism
was attempted. Final artifact cleanup therefore requires an allowed execution path
or manual user action, in addition to the pending benchmark and external validation.

### Bounded old/new policy investigation

The seven-day August 20-26 development screen uses
`config/experiments/september-policy-screen.json`: both profiles load the selected
configuration, disable books, and differ only in policy selection. This matches
configuration and dates, not posterior semantics: staged recovery and finite
core/tail conditioning differ. Staged has no equivalent hard per-move deadline;
this is not an equal-time comparison or a replay of the historical executable.
The single chronological block also gives degenerate bootstrap intervals, which
must not be interpreted as statistical certainty.

The before screen (`benchmarks/predictive/september-policy-screen-before-v1.json`)
finished in 47.9 seconds: staged solved 5/7 with two failures and a 4.1429 penalized
mean; finite solved 7/7 at 3.5714. Spare-budget widening now retains initial
incumbents while evaluating additional legal roots and exact continuations.
The widening-only screen (`benchmarks/predictive/september-policy-screen-after-v1.json`)
regressed to 3.7143, still 7/7; this negative outcome is preserved.
Exact recursive minimization now also excludes non-solving, non-informative
actions once an incumbent exists; fixed-root values remain unchanged.
The final screen (`benchmarks/predictive/september-policy-screen-progress-v1.json`)
finished in 48.0 seconds and returned to 3.5714, 7/7, versus staged 4.1429, 5/7.
No net finite gameplay-score improvement or production promotion is established.

**Superseded by the 22-survivor hard-mode resolution below.** This earlier replay is
preserved as historical evidence of the pre-constraint-reuse result.

The six-state hard-mode replay (`benchmarks/predictive/september-finite-regret-progress-hard-v1.json`)
completed in 15.3 seconds, versus 25.3 for the retained
widening-only hard-mode record (`benchmarks/predictive/september-finite-regret-widened-hard-v1.json`).
The nine-survivor state now selects `blype` and matches the exhaustive attempts
value to floating-point precision. The 22-survivor Fast state still selects `spike`
and retains failure
regret 0.0012701243. Strong's manual replay improves the fixed-root value to exact
but still selects `spike`; neither budget proves the global optimum there.
The references share the finite kernel; the normal/hard toy oracle remains independent.
New tests check widened values against that oracle and deterministic preservation
of completed roots when widening exhausts its work budget.
One preliminary regret run was discarded by the identity guard after a concurrent
source edit; only stable-snapshot successful runs are retained as evidence.

Final integration passes Rust 1.97 formatting, warnings-denied Clippy, all 329 tests
(284 library, 28 CLI, 6 integration, 11 predictive), and benchmark smokes. Generated
screen and existing rolling-evidence checks pass. Both Windows executables were
rebuilt and copied into ignored `dist`: GUI 10,865,664 bytes, subsystem 2, SHA-256
`8784E093FA211E65E9C6614978AEE244A25FD7983B58B4A8E8ABBE7A775E8221`;
CLI 15,116,288 bytes, subsystem 3, SHA-256
`DAB05A4FD4B4070DCA6EECB13A6B58FDB5395BB1FE5552902B21EBADB0A190D2`.
These supersede earlier hashes. CLI help works from `dist`; no new native visual
claim is made. Production selection is unchanged, no full matrix was run, and
changes remain uncommitted. Previously blocked cleanup was not retried.

### 22-survivor hard-mode resolution

The remaining bottleneck was repeated hard-mode constraint construction and
formatted rejection allocation inside recursive dictionary scans. Scans now
compile the existing constraints once per node; public diagnostics and internal
boolean filtering share one structured violation check. No legality rule, search
budget, prior, production selection, or memo key changed.

The fresh six-state audit (`benchmarks/predictive/september-finite-regret-legality-hard-v1.json`)
completed in 5.4 seconds. All six choices match the exhaustive reference within
floating-point precision (maximum failure regret 2.17e-19; attempts regret 8.89e-16).
Fast now selects `smite` for the 22-survivor state at modeled failure probability
0.0012701243 and expected remaining attempts 3.3021156; five separate manual Fast
replays repeated this choice. Strong also selects `smite` and completes the search.
Fast may still report Deadline: a completed fixed-root value is not a blanket
guarantee that all root values were resolved under every machine load.

The repeat seven-day screen (`benchmarks/predictive/september-policy-screen-legality-v1.json`)
completed in 52.5 seconds: finite 7/7 at 3.5714 versus staged 5/7 at 4.1429.
Normal-mode mean is unchanged; this fixes the measured hard-mode gap, not the
overall three-guess target. The existing independent toy oracle, hard-mode error
fixtures, cancellation checks, all 329 tests, warnings-denied Clippy and formatting
pass. Earlier negative evidence remains intact; no full-matrix promotion follows.

Refreshed ignored Windows distribution: GUI 10,864,640 bytes, SHA-256
`91CE717877B55C4A77570941BF3D86A93701DA4683C36107EB095A50314378A1`;
CLI 15,116,800 bytes, SHA-256
`04BA4F3DF350F789E2A981D1ACAE353F84195EBA67A53B445C5063218DA445F7`.
These supersede the prior hashes; production remains staged and changes uncommitted.

### Thirty-game decision and release handoff

The 30-game chronological comparison (`benchmarks/predictive/september-policy-30day-v1.json`)
ran July 28-August 26 in 111.5 seconds with the same two-profile configuration
screen. Its [generated table](generated/september-policy-30day.md) preserves full
coverage, outcomes, distributions, latency and paired uncertainty. Finite solved
30/30 at 3.4667; staged solved 28/30 at 3.4000 with failure/gap penalty seven.
Candidate-minus-baseline was +0.0667, paired interval [-0.4333, +0.5333], with
5 wins, 15 ties and 10 losses. Latency p95 was 268.35 ms versus 27.53 ms; staged
has no equivalent hard deadline. Support/recovery semantics still differ despite
matching configuration, dates and disabled books. This is development evidence,
not a new sealed test or a replay of the old release binary.

The finite policy spent 12 additional guesses across its ten losses and saved ten
penalized guesses across its five wins, including repairing both staged failures.
August 11 and 14 contributed two extra guesses each; their paths diverged from
the opener. These are candidates for broad-state policy analysis, not evidence
that a particular move is globally suboptimal. With failure penalty eight the
two policies tie at 3.4667, demonstrating the reliability/mean-score tradeoff;
the declared penalty-seven metric was not changed to manufacture a win.

Decision: retain the incumbent production selection; no credible lower-mean
improvement was established, so do not launch the longer release matrix or claim
promotion. Current `dist` still matches the verified source and hashes above;
no source changes or rebuild were needed for this comparison. Existing rolling
docs, deterministic historical archive, local Markdown file targets (15 files)
and whitespace checks pass. The requested guarded removal of `target/debug` was
again rejected before execution; no files were deleted and no bypass was attempted.
Native visual/macOS validation, future confirmation, full acceptance and cleanup
remain incomplete. No staging, commit or push was performed.

### Current review status (2026-09-22)

The restored Ponytail whole-repository audit found one unreachable post-check and three
duplicate digest encoders. The unreachable check was removed, and all three encoders now
use the existing `identity::hex` helper. Public forwarding/type aliases were retained
because this crate has a library surface and external compatibility cannot be ruled out.
Rust 1.97 formatting, all-target tests (329 passed) and warnings-denied Clippy pass after
these changes.

After Rust verification, the generated project-local `target/` directory was removed
through Cargo's clean command (its dry run identified only that directory). The runnable
Windows executables and build metadata in `dist/` are preserved. This supersedes the earlier
cleanup status without erasing the historical record of prior cleanup-command rejections.
No files were staged, committed, or pushed.

### September 23 dependency refresh and GUI observation

The local RustSec database now reports seven vulnerabilities against the previous
lockfile. Four compatible, targeted lockfile updates were applied without changing
`Cargo.toml`: crossbeam-epoch 0.9.20, rustls 0.23.45 (which also selects
rustls-webpki 0.103.15), webbrowser 1.2.2, and wayland-scanner 0.31.11 (which
selects quick-xml 0.41.0 for Wayland). Four further compatible patch updates
select anyhow 1.0.103, event-listener 5.4.2, memmap2 0.9.11 and uds_windows
1.2.1. `cargo audit --no-fetch` now exits 1 with only RUSTSEC-2026-0194 and
RUSTSEC-2026-0195 against quick-xml 0.30.0; its two remaining warnings are
unmaintained paste and ttf-parser, for which no compatible patch exists. The
remaining dependency is pinned by `zbus_xml ^4.0` through the Linux accessibility
stack; it is absent from both Windows and macOS normal dependency trees and
present on Linux. This is a platform-scoped unresolved alert, not a clean global
audit. A major GUI/accessibility-stack change or upstream release is needed to
remove it without disabling Linux accessibility. No remote alerts were dismissed.

The Windows computer-use helper also recovered enough to expose the distribution
app's accessibility tree. The main controls were named, but the 30 blank board
cells were exposed as unnamed buttons. The source now uses noninteractive labels,
and a Rust 1.97 AccessKit test passes for both blank and filled/draft cells.
The capture was obscured by another active window and the app disappeared before
keyboard interaction; rebuild and native visual acceptance remain open.

The exact finite argmin now prunes branches only against a completed feasible
incumbent with conservative failure-first/attempts-second bounds; upper-bound
children cannot act as exact lower-bound contributions. Normal-mode duplicate
feedback signatures are reused, while hard mode and fixed-root evaluations
remain unchanged. An independent source review found no concrete value-soundness
defect; 22 finite tests include an additional skewed/zero-mass oracle case.
Rust 1.97 formatting, warnings-denied all-target Clippy and all-target tests
(333 tests plus benchmark smoke checks) pass against the refreshed lockfile.
The then-current-source fixed-state profile (`benchmarks/predictive/release-gameplay-latency-bound-pruning-v1.json`)
completed eight workloads in 33.7 seconds excluding compilation. Its broad-state
results are feasible upper bounds, not a new mean-guess or promotion result.

On September 26, a two-column egui regression reproduced the first-guess
board/recommendations overlap: the first nominal 48-pixel tile painted 564
pixels wide. Constraining the tile width restored 48-pixel geometry. Rust 1.97
formatting, warnings-denied Clippy and all-target tests pass (334 tests plus
benchmark smoke checks). Both `dist` executables were rebuilt and hash-checked:
GUI SHA-256 `D8281054F1F504D54F55C94979F35FEAE20E18B3A9328CD9EC92D9FAE35CA7E3`
(subsystem 2), CLI SHA-256
`42E878E4CD913B122E5D079442065974BEF98F8A1F28E5AB488C13D501A5046E`
(subsystem 3). The native capture helper twice failed to activate the GUI
window, so this remains a geometry-test result, not a visual or keyboard pass.
Cargo's dry run identified only rebuildable `target` artifacts (11,811 files,
11.0 GiB); after checking for reparse points, `cargo clean` removed them.
Both ignored `dist` executables retained their hashes, and CLI `--help`
still runs without `target`.

### Current-source development score rerun (September 26)

The rebuilt executable completed the same July 28-August 26 two-profile
development comparison twice, each in under 90 seconds and below the 900-second
cap: first JSON (`benchmarks/predictive/september-policy-30day-current-v1.json`)
and repeat JSON (`benchmarks/predictive/september-policy-30day-current-repeat-v1.json`),
with their [first](generated/september-policy-30day-current.md) and
[repeat](generated/september-policy-30day-current-repeat.md) generated tables.
The seal was not evaluated. Independent sums of per-game penalty-seven scores
match the reported means: staged 102/30 = 3.4000 with 28 solves in both runs;
finite 109/30 = 3.6333 and 111/30 = 3.7000 with 30 solves in both runs.
The paired deltas were +0.2333 [-0.1000, +0.5000] and +0.3000
[-0.1667, +0.6667]. This remains a negative mean-score result, not a
promotion or a stable single-number estimate.

The two runs have identical executable/input, effective-config, matrix and
date identities. Staged paths were identical; eight finite paths changed,
and its opener mix changed from ABLED 23 / RAISE 7 to ABLED 28 / ABORE 2.
The earlier 3.4667 finite run had the same configs and dates but a different
executable identity; 27 finite paths changed, while staged paths did not.
Source tracing narrows the likely mechanism: the 250 ms finite Fast search
inserts a state-local baseline first, then retains completed feasible
candidates from the root-evaluation prefix when the deadline fires. An
`UpperBound` ABLED result is a completed rollout incumbent, not the unscored
heuristic fallback. Different completed prefixes can therefore change the
selected root. The artifacts do not record per-game search status, work units
or candidate values, so this is a plausible explanation, not proof that no
other source of variation exists. Capture those diagnostics and compare a
fixed-work-unit control before changing selection policy. Do not ascribe the
regression to bound pruning alone or average these time-budgeted runs into a
prospective performance claim.

Handoff: `master` at `1d60715` remains dirty; nothing was staged, committed or
pushed. The two completed benchmark JSON artifacts and generated tables are
retained above. A checked `cargo clean --dry-run` identified only three
rebuildable files (230.5 KiB) in `target`; `cargo clean` removed them. Both
ignored `dist` executables retained their release hashes, and CLI `--help`
still runs without `target`. Next actions:
trace the finite root-choice variation, confirm native GUI visual/keyboard
behavior, declare a genuinely prospective holdout after candidate freeze,
obtain native macOS evidence and decide whether the longer seven-profile
matrix is justified. The finite policy remains experimental.

### Per-move finite trace and repeatability check (September 26)

The experimental benchmark now records an optional `finite_search_steps` entry
for each finite move: turn, stop reason, visited nodes, cooperative work units,
proposal-sampling flag, returned candidate count and up to eight ranked
candidate words/qualities/values. Historical JSON without this optional field
remains readable. An unscored heuristic fallback has null modeled values;
completed candidates must have finite valid values. A failing-then-passing
backtest test and a fallback/deadline serialization test cover the new trace.
Rust 1.97 formatting, warnings-denied Clippy and all-target tests pass (335
tests plus benchmark smoke checks). The rebuilt ignored Windows GUI and CLI
hashes are respectively
`3B943A6BA11C56ED9254640B39E617957A0B73D0FC1EA3E9169EB12F283E5232`
(subsystem 2) and
`EFC5CC500F965F14A342FC84938A5E22E2F762FBFA4FDD76A366D4B46BDE33B9`
(subsystem 3). Native visual/keyboard acceptance remains open.

The same July 28-August 26 development matrix completed twice in 87.1 and
87.9 seconds, under the 900-second cap: first traced JSON (`benchmarks/predictive/september-policy-30day-traced-v1.json`)
and repeat traced JSON (`benchmarks/predictive/september-policy-30day-traced-repeat-v1.json`),
with their [first](generated/september-policy-30day-traced.md) and
[repeat](generated/september-policy-30day-traced-repeat.md) generated tables.
Both artifacts declare the same input fingerprint and `sealed_test_evaluated=false`.
Independent per-game sums match reported penalty-seven means: staged 102/30 =
3.4000 with 28/30 solves in both; finite 110/30 = 3.6667 and 109/30 =
3.6333 with 30/30 solves. Paired finite-minus-staged deltas are +0.2667
[-0.1000, +0.5667] and +0.2333 [-0.1675, +0.5342]. There is still no
lower-mean evidence or production promotion.

Staged paths were identical; two finite paths differed. On July 28, the
second-turn `marts` candidate was ranked first in one run but absent from the
other run's top eight, which chose `tarns` and finished one guess sooner. On
August 11, the third-turn `weigh` candidate's completed upper-bound failure
value changed from about 0.001176 to 0, and it displaced `wingy` as the next
guess; both paths took five guesses. Both divergent moves reported `deadline`,
with different work-unit totals and completed candidate rankings. This is
direct evidence of deadline-sensitive partial evaluation, not proof that
wall-clock timing is the only influence. A deterministic work-unit control
and matched score/latency/memory comparison are required before altering
selection policy. The older current-source artifacts remain retained as
negative and repeatability evidence; source/executable identities differ from
these traced artifacts.

Handoff: `master` at `1d60715` remains dirty and uncommitted; no push occurred.
The four traced JSON/Markdown outputs are retained. The exact CI rolling-docs
check, fresh-render comparisons for both generated tables, legacy archive
check, 110 local Markdown links and changed-file whitespace check pass. Cargo
confirmed `target` was project-local with no reparse points; its dry run found
7,138 rebuildable files (5.5 GiB), and `cargo clean` removed them after evidence
capture. Both ignored `dist` executables retained their hashes, and CLI
`--help` runs without `target`. Native GUI visual/keyboard validation, the
fixed-work-unit score/resource experiment, prospective holdout, native macOS
checks and remaining release guards are still open.

### Trace review and final-source development repeat (September 26)

Independent read-only review found two trace safeguards missing. The producer
now validates every completed candidate and guess index before truncating to
eight serialized entries, and finite backtests reject missing, out-of-order or
path-mismatched step traces. Both defects had failing-then-passing regression
tests; the optional field remains backward-compatible with historical JSON.
Rust 1.97 formatting, warnings-denied Clippy and all-target tests pass again
(336 tests plus benchmark smoke checks). The rebuilt ignored GUI and CLI hashes
are respectively
`F1CD3B1E4F685CE87FE8D5203E7AAE4787DD2441E073B687D9593FB563F6BA03`
(subsystem 2) and
`F1A48521170B9D09E274908D8D55C3FD7B1C62FF6A78960FCAB8335839B11D33`
(subsystem 3). These supersede the earlier `dist` hashes without deleting
the earlier evidence.

The same July 28-August 26 development matrix finished twice with this exact
binary in 123.4 and 94.4 seconds, below the 900-second cap:
validated JSON (`benchmarks/predictive/september-policy-30day-validated-v1.json`)
and identical-input repeat (`benchmarks/predictive/september-policy-30day-validated-repeat-v1.json`),
plus their [first](generated/september-policy-30day-validated.md) and
[repeat](generated/september-policy-30day-validated-repeat.md) generated tables.
Both declare the same input fingerprint and no sealed-test evaluation.
Independent per-game sums match reported penalty-seven means: staged 102/30
= 3.4000 with 28 solves and no coverage gaps in both; finite 108/30 = 3.6000
and 110/30 = 3.6667 with 30 solves and no gaps. Paired finite-minus-staged
deltas were +0.2000 [-0.3000, +0.6000] and +0.2667 [-0.1000, +0.5667].
Staged paths were identical; eight finite paths differed. Every first divergent
move recorded `deadline`; on several root moves the returned candidate count
varied from two to ten as cooperative work varied. This establishes a
deadline-sensitive partial-evaluation cause for the observed path differences,
but does not prove that timing is the only source of variation. The finite
policy still fails the lower-mean guard, so there is no promotion or flat-three
claim. Next: a deterministic work-unit control under matched development
rules, then paired score and resource checks if it is competitive. The declared
seal remains untouched; genuinely prospective confirmation and native macOS
validation are still external/future gates.

Handoff: `master` at `1d60715` is dirty and uncommitted; no staging or push.
The four final-source JSON/Markdown outputs are retained alongside earlier
historical runs. The exact CI rolling-docs check, fresh render of both new
tables, historical JSON parsing, legacy archive check, 116 local Markdown
links and tracked/untracked whitespace checks pass. Cargo confirmed `target`
was project-local with no reparse points; its dry run found 7,138 rebuildable
files (5.5 GiB), and `cargo clean` removed them after evidence capture.
Both ignored `dist` executables retained their hashes and CLI `--help` runs
without `target`. Native GUI visual/keyboard acceptance remains open because
capture could not activate the window.

### Fixed-work repeatability control (September 26)

An experiment-only `finite_fast_fixed_work` profile preserves finite Fast's
shortlists and exact-state threshold, but stops at four million cooperative
work units with a five-second safety deadline. The selected production config
and 250 ms finite Fast profile are unchanged. The first two-profile run (`benchmarks/predictive/september-policy-fixed-work-v1.json`)
and repeat (`benchmarks/predictive/september-policy-fixed-work-repeat-v1.json`)
used July 28-August 26, one input fingerprint, no artifacts and no sealed
evaluation; their generated [first](generated/september-policy-fixed-work.md)
and [repeat](generated/september-policy-fixed-work-repeat.md) tables are retained.
Both completed in about 30 seconds, below the 900-second cap.

Fixed work solved 30/30 with a penalty-seven mean of 3.5000 in both runs.
All 30 paths and every finite-search trace matched; 74 moves reached the node
budget, 31 completed, and none hit the safety deadline. Timed finite Fast also
solved 30/30 and scored 3.6333/3.6667, but three paths and traces on all
30 games differed. The fixed-minus-timed paired deltas were -0.1333 and
-0.1667, with block-bootstrap intervals [-0.4000, +0.1667] and
[-0.5000, +0.1333], and wins/ties/losses 12/10/8 and 12/11/7.
Both intervals include no improvement. The already measured staged policy
scored 3.4000 on these same dates with 28 solves, so fixed work is not a
lower-mean release candidate. Its 30/30 solves are encouraging but too few
to establish a failure-rate advantage.

Recorded per-move p95 latency was 154.46/152.49 ms fixed versus
262.71/264.53 ms timed; whole-process peak working set was 88.69/90.35 MB
and cannot be assigned to one profile. This control isolates the observed
deadline sensitivity on this machine; it is not proof of cross-machine timing
stability or a reason to replace production defaults. Rust 1.97 formatting,
warnings-denied Clippy and all-target tests (338 tests plus benchmark smokes)
pass after the mode/registry change. Native GUI visual/keyboard acceptance,
matched staged calibration/resource guards, prospective confirmation and
native macOS evidence remain open. Branch `master` at `1d60715` stays dirty,
unstaged and unpushed.

Final artifact hygiene: the Windows GUI and CLI were rebuilt from the same
source used for the fixed-work reruns and copied to ignored `dist`. Their
SHA-256 values are `347EF387C4AB4383EBBE2F15570486A8306FAC0AEF4EFEE9007C67DF252EB554`
and `152DFDBBC264730AEAA1480EC5CA406A6BF4F14B4F8891546FAB21C5CD8B86E9`;
the PE subsystems remain 2 (GUI) and 3 (console). Cargo metadata confirmed
`target` inside this repository, with no reparse points; `cargo clean`
removed 7,171 rebuildable files (5.6 GiB). The ignored `dist` is retained.

### Same-binary staged versus fixed-work decision (September 26)

The final `dist` CLI ran the separate [staged-versus-fixed matrix](../config/experiments/september-staged-fixed-work.json)
on July 28-August 26 twice: first evidence (`benchmarks/predictive/september-staged-fixed-work-v1.json`)
and repeat (`benchmarks/predictive/september-staged-fixed-work-repeat-v1.json`),
with generated [first](generated/september-staged-fixed-work.md) and
[repeat](generated/september-staged-fixed-work-repeat.md) tables. Both artifacts
share one input fingerprint, declare no sealed-test evaluation, and took
73.1/72.2 seconds. Staged paths repeated exactly, as did fixed-work paths and
all fixed finite traces. The fixed configuration fingerprint and 30 paths also
match the preceding fixed-versus-timed experiment.

Staged solved 28/30 at a penalty-seven mean of 3.4000; fixed work solved
30/30 at 3.5000, with no coverage gaps in either. The paired fixed-minus-staged
delta is +0.1000, 95% block-bootstrap interval [-0.2000, +0.4000], with
5 wins, 16 ties and 9 losses. Initial-state log loss was 6.5971 staged versus
7.5789 fixed; unhalved Brier was 0.998579 versus 0.999017. Recorded per-move
p95 latency was 21.31/23.49 ms staged versus 139.29/137.16 ms fixed. Process
peak working set was 147.00/143.46 MiB across both profiles, not a
candidate-specific memory measurement. The two extra solves are a useful
development observation, but 30 games and a mean interval spanning zero do
not establish a quality advantage or justify promotion.

These profiles match the config file, dates, rules, executable and disabled
artifacts, but **not the effective initial probability distribution**:
`initial_state` freezes core/tail mass for finite modes and retains staged
recovery semantics otherwise. `average_log_loss` and Brier are scored at that
initial state, not along each posterior path. The comparison is an end-to-end
policy-bundle test, not a search-only ablation or equal-time experiment.
Effective-prior matching, isolated resource checks, genuinely prospective
confirmation, native GUI visual/keyboard and macOS checks remain open; the
declared seal was not used for selection. No production promotion, commit or
push follows from these development results.

Native GUI retry: the final `dist` executable launched and exposed exactly one
`Maybe Wordle` window. The Windows capture helper failed to activate both the
initial and freshly selected handle, so it provided no trustworthy screenshot
or keyboard interaction. The first-guess column geometry test remains green,
but visual/keyboard acceptance is still unverified.

After verifying the two final JSON artifacts and generated tables, the only
new `target` contents were two project-local completed-run checkpoints, with
no reparse points. `cargo clean` removed them (three entries, 695 KiB); the
ignored Windows `dist` binaries remain present and unchanged.

Continuation handoff: `master` at `1d60715` remains dirty, unstaged and
unpushed. The native helper again found exactly one running `Maybe Wordle`
window but failed to activate both the selected and refreshed handle, so
visual/keyboard acceptance is still open; the fixed-width tile regression
and previously verified release GUI remain the available evidence. The
read-only rolling-docs and legacy archive checks pass, all 67 then-existing untracked
benchmark JSON files parse with no byte-identical duplicates, both `dist`
hashes still match `BUILD-INFO.txt`, and `target` is absent. `TODO.md`
now lists only unresolved decisions and release gates. No source code or
production policy changed in this continuation. Next: native UI evidence
when the capture path works, plus the matched-belief search ablation and conditional finalist selection before
remaining release checks; do not use the reserved seal for tuning.

The release profiler was rebuilt on Rust 1.97 and ran its existing
baseline-only and Fast phases in separate processes over all eight fixed
August 1 states. Both artifacts have the same executable, input and config
fingerprints: baseline (`benchmarks/predictive/release-gameplay-latency-baseline-v1.json`)
and Fast (`benchmarks/predictive/release-gameplay-latency-fast-matched-v1.json`).
OLATE/10001 median full-request latency was 4.93 ms baseline-only versus
254.64 ms Fast; root was 29.82 versus 260.92 ms. This supports a bounded
refinement-cost diagnosis, not a score improvement or exact internal-stage
attribution. The [performance report](PERFORMANCE.md) records all eight
states and limitations. No production config or solver source changed.
The artifact audit checked schema v3, eight matching workload IDs, three
samples per phase, executable/input/config identity and normalized posterior
sums. Both GUI board-tile release tests pass on Rust 1.97. The read-only
Markdown audit found 138 local links across 25 files with no missing target.
Cargo metadata resolved `target` inside this repository; after a dry run
and reparse-point check, `cargo clean` removed 2,267 rebuildable files
(920.5 MiB). The ignored `dist` executables are retained. Source-level
Rust CI gates remain at their preceding 338-test pass because this
continuation changed only documentation and benchmark evidence.

### Matched finite rollout screen (September 26 source)

The final `dist` CLI ran the declared 12 development folds, 360 games per
profile, with the same 250 ms budget and initial finite belief in both modes.
The two candidate TOMLs differ only in `search_policy_mode`; their recorded
prior evidence is identical. The first comparison (`benchmarks/predictive/september-finite-matched-budget-current-v1.json`)
and fresh-label repeat (`benchmarks/predictive/september-finite-matched-budget-current-repeat-v1.json`)
have the same input fingerprint, code revision, fold plan and profile config
fingerprints. Their target dates exclude the consumed June 18-July 17 window and
the reserved August 28-September 26 seal. Both declare `sealed_test_evaluated:
false`. The artifact audit matched every date/target pair and recomputed the
penalty-seven means from all 360 game outcomes.

Finite baseline solved 354/360 and repeated exactly at 3.5389 mean with
identical game paths. Fast solved 360/360 in each run but scored 3.5694 first
and 3.5528 on repeat. The paired Fast-minus-baseline differences were +0.0306
with 95% interval [-0.0722, +0.1361], then +0.0139 with interval [-0.0890,
+0.1167]. Between Fast runs, 20 paths and eight game scores changed; 16
finite-step reason sequences changed. These outcomes support a solve-rate
tradeoff, not a mean-score win. The deadline-sensitive Fast search remains
non-repeatable under the matched wall-clock budget. This is a finite
baseline-versus-rollout ablation, **not** a staged-versus-finite search-only
comparison; the latter still needs matched effective beliefs. Neither mode
passes release promotion gates, and production v20 remains unchanged.

The first-guess GUI regression now checks draft and applied tile bounds
against the adjacent suggestions panel at 861px (two-column threshold) and
1180px. Both widths pass with the existing 48px tile frames; no new
production GUI change was needed. Rust 1.97 `cargo fmt --check`, warnings-denied all-target Clippy,
all-target tests and the 19 GUI unit tests pass. This is layout geometry,
not a native visual/keyboard acceptance check. At that phase, the `dist`
GUI/CLI hashes matched the then-current `BUILD-INFO.txt`; no production source
had changed, so the executables were not rebuilt then.
After the shared checks, Cargo metadata located `target` inside this repository,
the dry run counted 5,015 rebuildable files (4.6 GiB), and a recursive scope
check found no reparse points. `cargo clean` removed that output; ignored
`dist` and the two comparison artifacts remain intact. The worktree is still
dirty and uncommitted. Next gates are native UI interaction, a staged-versus-
finite matched-belief diagnostic if pursued, finalist selection and full
release arithmetic/hygiene review before any authorized git action.

### Checkpoint trace guard and historical-artifact compatibility

Current source now rejects a resumed finite rolling checkpoint whose stored
game path and per-move search traces are missing or misaligned. The guard
does not impose that newer requirement on final rolling artifacts generated
before optional traces existed; any traces present in those artifacts are
still checked. A missing-trace checkpoint regression failed before the guard
and passed afterward. A separate historical-finite-artifact regression and
the read-only `rolling-evidence-docs` command pass after correcting this
compatibility boundary. The first release build exposed the mistake by
rejecting the old evidence; that build was superseded before release.

Rust 1.97 formatting, warnings-denied all-target Clippy and all-target tests
pass on the corrected source (295 library tests plus other targets and bench
smokes). The corrected release build succeeded. The current `dist` CLI has
SHA-256 `699C5A37C45784E7EE0F80DE54A8523CB3D311E4B1ABF13F456EAAE4F3434723`
and PE console subsystem 3; it passes `--help` and the rolling-docs check.
At that checkpoint, the new GUI build had SHA-256
`CA1B2A2ECAEE43D31CDC1CB19FCB98A4B4C7C6F60A9E255DC000A3DED41851ED`
and GUI subsystem 2, but remained in `target/release` while the older
`dist/maybe-wordle.exe` was running. Closing that window could lose in-app
state, so replacement waited for user approval; the later hygiene handoff
records the completed replacement and release-profile cleanup.
The earlier 12-fold comparison artifacts were measured before this
checkpoint-validator change; no search-policy or scoring code changed, but
their code identity is historical rather than the final release build.

### Current selected-policy development comparison

The corrected `dist` CLI ran a fresh staged-versus-finite 12-fold comparison (`benchmarks/predictive/september-staged-finite-current-source-v1.json`)
and a fresh-label Fast repeat reusing its identity-checked staged baseline (`benchmarks/predictive/september-staged-finite-current-source-repeat-v1.json`).
Each profile covers the same 360 dated targets in the declared development
folds, from 2025-07-03 through 2026-08-26. The artifacts have matching
source/input, plan and config fingerprints, aligned date/target pairs, zero
coverage gaps and `sealed_test_evaluated: false`; neither evaluates the consumed
June 18-July 17 window or reserved August 28-September 26 seal as targets.
Earlier consumed answers may still be chronological training history. Independent
per-game penalty-seven sums, paired win/tie/loss counts and dates match the
reported aggregates. The staged baseline paths were identical on reuse.

Staged solved 358/360 and scored **3.2000** (1,152 penalty-seven guesses).
Finite Fast solved 360/360, but scored **3.5556** first (1,280) and **3.5639**
on repeat (1,283). Fast-minus-staged paired differences were +0.3556 with
95% interval [+0.2639, +0.4445], then +0.3639 with interval [+0.2694,
+0.4528]. The first comparison had 38/170/152 Fast wins/ties/losses; the
repeat had 38/169/153. Fast changed 13 paths and eight game scores between
runs; staged did not change. Per-move p95 was 22.48 ms staged versus
262.46/262.60 ms Fast. Initial prior evidence covered 332 staged targets
versus 360 finite targets, reflecting different effective support/recovery
semantics. Thus this is an end-to-end policy comparison, not an equal-belief
search-only ablation. It rejects finite Fast as a mean-score improvement,
while the two extra solves show why the empirical zero-failure guard remains
separate. Staged itself does not pass that guard. No candidate is promoted,
and 3.0 guesses remains unproven. The finite route is closed for this release
unless new matched evidence provides a substantially better tradeoff.

### Exploratory staged entropy-weight screen

An older staged v19b configuration was screened only as a low-cost failure
recheck after the corrected baseline, not as an expected quality winner.
The identity-bound [30-game matrix](../config/experiments/september-v19b-staged-screen.json)
produced JSON evidence (`benchmarks/predictive/september-v19b-staged-screen-v1.json`)
and a [generated table](generated/september-v19b-staged-screen.md) for
July 28-August 26. The effective configs differ only in
`proxy_weights.entropy_w` (selected 0.1463482484; v19b 0.3375), so prior
log loss and Brier are identical. Selected staged scored 3.4000 with 28/30
solves; v19b scored 3.3000 with 29/30. Exactly one game changed, giving
v19b-minus-selected -0.1000, 95% interval [-0.3000, 0], with 1/29/0
wins/ties/losses. Generation took 122.58 seconds and stayed within its
1,200-second/4,096-MiB limits. This fold had already been used in development,
so it is a screen, not independent validation.

The candidate then ran over all 12 allowed folds using the identical
source/input and an identity-checked reused selected baseline:
rolling comparison (`benchmarks/predictive/september-v19b-staged-rolling-v1.json`).
All 360 date/target pairs align, no excluded or sealed date appears, both
have zero coverage gaps, and independent penalty-seven sums reproduce the
reported values. Selected scored 3.2000 with 358/360 solves (1,152 total);
v19b scored 3.1944 with 359/360 (1,150 total). The paired delta is -0.0056,
95% interval [-0.0278, +0.0139], with 6/347/7 v19b wins/ties/losses.
Forty-one game paths changed; candidate per-move p95 was 23.08 ms versus
22.48 ms selected. The effective initial prior evidence is identical because
the only config change is a proxy score weight. The small mean shift is not
distinguishable from zero, the candidate still has one failure, and it does
not meet release-finalist or zero-failure gates. Production remains v20.
No multi-seed joint refinement or seven-profile release matrix is justified
from this screen alone; those conditional gates remain open for a genuinely
competitive candidate.

### September 26 hygiene and simplification handoff

The latest Rust 1.97 formatting, warnings-denied Clippy, and all-target test
gates passed after the checkpoint trace guard and expanded first-row GUI
geometry regression. The subsequent work changed only documentation and
development evidence. A scoped whole-repo simplification audit found no
rebuildable or duplicated source evidence to discard. It identified a small
GUI feedback-decoder duplication, a one-variant study fold-selection enum,
and public predictive-policy projection types whose only in-repo use is a
fixed identity string. The latter two affect serialized or public contracts,
so deletion is deferred pending an explicit compatibility decision. Removing
the small decoder duplication would change the executable identity for only
about nine lines of code and invalidate these latest source-bound comparisons;
it is also deferred. None changes the release-score decision. A dry-run and
path check preceded removal
of 6,550 rebuildable development-profile files (6.3 GiB). The old GUI
process then closed normally at the user's request. The verified current-source
GUI replaced it in `dist` with an exact SHA-256 match to the release build;
a subsequent startup exposed a window and closed normally. After a dry run,
workspace-path check and reparse-point check, `cargo clean --release` removed
another 2,118 rebuildable files (859.8 MiB). Both current-source executables
and benchmark evidence remain in place. Native visual/keyboard acceptance
is still unverified. No commit or push has been made.

### Formal cache and live dependency recheck (September 26)

`py -3 scripts/import_optuna_archive.py --check` verifies the retained
historical trial archive, and the obsolete Python optimizer entry point has
no remaining in-repo references. Offline Cargo metadata confirms Rust 2024
with `rust-version = 1.97`; both CI jobs pin 1.97.0. The official checkout
releases page still marks v7.0.1 latest, and the pending workflow uses the
`v7` major tag.

The current `dist` CLI regenerated the formal pattern cache twice from pinned
seed lists. Both copies had SHA-256
`E86B18AE50E529130C6ED9F17B0B962B8977028A55A984A31723D4C36872D6B7`,
`MWORDPT3` magic, 14,855 guesses, 2,315 answers and 34,389,437 bytes; each
run was stopped after cache creation and before full proof search. The temporary
cache was removed again. Git therefore still shows the intended deletion of
the formerly tracked table; it cannot become untracked in the index until an
authorized staging/commit action. This proves deterministic cache generation,
not feasibility of full-model formal construction.

A live read-only GitHub Dependabot query still found seven open alerts on the
pushed `Cargo.lock`: `quinn-proto`, `rustls-webpki` and both locked `rand`
versions. The local, uncommitted lockfile contains `quinn-proto 0.11.15`,
`rustls-webpki 0.103.15`, `rand 0.8.6` and `rand 0.9.3`, meeting each alert's
reported first-patched version. No alert was dismissed; remote status can
change only after an authorized push and GitHub re-analysis. Offline
`cargo audit --no-fetch` still reports two `quick-xml 0.30.0` advisories in
the Linux accessibility dependency path and unmaintained `paste` and
`ttf-parser` warnings. Those are distinct from the seven remote lockfile
alerts and remain recorded for platform review.

### Pre-terminal rebuild identity and paired replay (September 26)

The whole-diff audit noticed that `src/gui.rs` was newer than the packaged
GUI. A fresh Rust 1.97 `cargo build --locked --release --bins` produced
different, repeatable binary hashes; both `dist` executables were replaced
from that build. At this preceding checkpoint the GUI was
`75DE3F8A3D91B1873D1AFA499C75A9A6369B2D32B4281DDC9BBC6C6A845FA724`
(PE GUI subsystem 2), and the CLI was
`10C32BFA6172739DE95BC763FAD2B63A54BE9287F2B04DC95AF4AE98B036AD30`
(PE console subsystem 3). The new GUI opened a window and closed normally;
native visual/keyboard acceptance is still open. The CLI help and read-only
rolling-docs check pass.

The rebuilt CLI then completed a fresh 12-fold staged/v19b comparison (`benchmarks/predictive/september-v19b-staged-final-build-v1.json`)
without reusing a baseline artifact. The input fingerprint changed from
`sha256-v1:cb534cacea57613c1a666da594c0fca2c7ecf08de7105b1969ca49cc6f79e797`
to `sha256-v1:b38fdadf15baf594f91ca676ce292ec55f47c6409069b755b5d3318dd4b31c11`.
Independent replay checks found 360 aligned, unique date/target pairs, no
excluded or sealed date, zero coverage gaps, and exact reproduction of every
earlier baseline and candidate path. Penalty-seven sums were 1,152 staged
and 1,150 v19b, with 358 and 359 solves respectively. The paired v19b-minus-
staged mean is -0.0056, interval [-0.0278, +0.0139], and win/tie/loss
6/347/7. Per-move p95 was 21.81/21.65 ms staged/v19b in this run. This
closes the stale-binary evidence gap for the primary staged policy and v19b
screen, but not the candidate's failure or inconclusive paired interval.
The earlier finite Fast comparisons retain their preceding executable
identity and remain negative historical development evidence.

### Two-turn staged correction and fresh release replay (September 26)

An independent partition check found a concrete staged objective violation
in the allowed August 26 development game. With four observations and four
remaining dictionary candidates, the old exhaustive ranking preferred
`canon` for its lower unlimited-horizon expected cost. `junco` partitions all
four by distinct feedback and guarantees an answer on the sixth turn in
normal mode. It is not legal in hard mode, so the hard-mode filter still
applies. The staged path now uses exact modeled two-turn success mass and a
structural `force_in_two` guard after four observations; the final-turn rule
is unchanged. This is a terminal correction, not full finite search in
earlier turns. A failing-then-passing regression and an independent feedback-
bucket oracle test cover the ranking and mass formula, including an answer
that is not in the guess dictionary. Both staged and finite suggestions now
withhold `force_in_two` when such an answer survives, so the public
force-only filter cannot imply an impossible guarantee. The current CLI
recommends `junco` with no cached promotion for the four-clue state.

The rebuilt release CLI ran a new 12-fold paired comparison (`benchmarks/predictive/september-v19b-staged-terminal-legal-v1.json`)
with input fingerprint
`sha256-v1:9a1d4d413c3550e0515c6c175e5e5da2af84165b9f999fb735e78790841169d9`.
Independent replay found 360 unique aligned date/target pairs, no declared
sealed date, no coverage gap, and exact agreement with the preceding build
except two selected outcomes and one v19b outcome that were formerly unsolved
and now solve in six. One selected six-guess path also differs from the first,
pre-legality terminal replay (`benchmarks/predictive/september-v19b-staged-terminal-v1.json`)
while retaining its six-guess solve. Selected staged scored 1,150/360 =
**3.1944**, 360/360 solves; v19b scored 1,149/360 = **3.1917**, 360/360
solves. The paired v19b-minus-selected
difference was -0.0028, 95% interval [-0.0222, +0.0167], with 6/347/7
win/tie/loss. Per-move p95 was 21.66/22.20 ms. Both profiles meet the
development zero-failure observation, but the candidate advantage is
inconclusive and neither result is prospective or near three. Selected
production configuration remains v20.

Rust 1.97 formatting, warnings-denied Clippy and all-target tests passed
after the source correction. The new `dist` GUI is
`0D2370F7262FFF1FF046E73FA7CEA7318FF8B3A486C202D1EBC7B6D91A1B577D`
(PE subsystem 2) and CLI is
`DBCD2E0FA4580099FE352098702866FA9BB9FDBFC7EAB21A460BBA01D5214826`
(subsystem 3); both hashes match the release outputs. Windows UI Automation
exposed the live Play controls and distinct board/suggestions panels. A
first-row entry attempt did not commit, and GPU-window pixel capture on the
locked desktop showed the lock screen rather than the app. The passing egui
draft/applied geometry and AccessKit tests remain the strongest first-row
evidence; native visual/keyboard acceptance is still open. No result from
that failed capture is presented as a screenshot verification.

The final distributed CLI reproduced `junco` for the four-clue state without
cached promotion, and the read-only rolling-evidence documentation check
passed. After verifying the final binary hashes and Cargo dry-run scope,
`cargo clean --release` removed 2,118 rebuildable files (859.9 MiB) and
`cargo clean --profile dev` removed 6,550 (6.3 GiB). The verified `dist`
executables and benchmark JSON remain; resumable evidence checkpoints were
not touched. The whole-worktree diff and local-link checks found no broken
links or whitespace errors. The worktree remains unstaged and uncommitted;
native Windows interaction, macOS CI evidence, and a genuinely prospective
quality result remain open.

The release-scope hygiene inventory found 78 untracked predictive JSON files
(18.9 MiB). Sixty-five already had repository consumers; the constant-liar,
widened-hard-mode and pre-legality terminal records are now linked above as
otherwise easy-to-lose negative or historical evidence. The remaining ten
unreferenced `study-*.json` outputs use obsolete study formats 16/17, rejected
by the current format-18 loader. They remain local and untracked, but are not
proposed release files; they were not deleted because current code cannot
regenerate those exact historical runs. The unused finite-calibration TOML
likewise remains untracked pending a final release-scope decision. The CI
rolling-docs comparison JSON, experiment matrices and candidate TOMLs loaded
by tests must be included when a commit is authorized. The newest remote CI
success is still the August 2 run for the previously pushed revision; local
checks do not substitute for CI on these uncommitted changes.

### Fixed-work tail-mass control (September 26)

The final `dist` CLI ran the [tail-mass matrix](../config/experiments/september-tail-fixed-work.json)
on the allowed July 28-August 26 development dates. The first (`benchmarks/predictive/september-tail-fixed-work-v1.json`)
and repeat (`benchmarks/predictive/september-tail-fixed-work-repeat-v1.json`)
JSON, with generated [first](generated/september-tail-fixed-work.md) and
[repeat](generated/september-tail-fixed-work-repeat.md) tables, share the
current-source input/config/matrix fingerprints and declare no sealed-test
evaluation. All 30 game paths, scores and finite-search traces for each
profile reproduced exactly; neither profile recorded a deadline stop.

Both fixed-work profiles solved 30/30 without a coverage gap. The selected
tail mass scored 3.5000 all-game guesses; lowering `fallback_prior_mass` to
0.0001 scored 3.6333. The paired near-core-minus-selected delta is +0.1333,
95% block-bootstrap interval [-0.0333, +0.3000] (3/20/7 wins/ties/losses).
The low-tail profile also worsened initial-state log loss from 7.5789 to
8.6048. This is a deterministic, small, development-only ablation, not a
prospective or production improvement claim. The 0.0001 setting is an
approximation, not exact core-only support. Earlier 250 ms finite Fast runs
varied across repeats because of wall-clock budget stops; do not interpret
those timed results as a causal tail-mass comparison. Production v20 and
its staged policy are unchanged.

Release handoff: native Windows inspection launched the retained GUI and
confirmed its live controls through accessibility, but its capture returned
the Windows lock screen and window activation failed. No native visual or
keyboard pass is claimed. The full source-gate result predates this
documentation-only tail control; the read-only rolling-docs check, 171 local
links and diff whitespace check passed afterward. Cargo's dev-profile dry
run identified 3,191 rebuildable files (3.2 GiB), then `cargo clean --profile
dev` removed them; `target/release` was also absent at that checkpoint. Both retained `dist`
hashes still match the final-build values above. A 30-pair process-level
four-observation probe is recorded in `PERFORMANCE.md`; the next source work
needs an in-process staged-kernel profile and a matched-belief
six-turn-objective candidate; native UI acceptance needs an unlocked desktop,
and prospective/macOS/remote CI checks still need their declared conditions.
No files were staged or pushed.

### Matched-belief finite-search experiment boundary

The next causal policy comparison must separate search from the posterior.
Currently the `staged` mode starts with modeled core mass and potentially
dormant fallback, whereas finite modes freeze a positive core/tail mixture
before play. A shared TOML therefore does not imply a shared effective belief.
For the first experiment, construct one fixed posterior per development date,
run staged ranking and finite search from that same explicit state with books
disabled. On identical forced observation prefixes, fingerprint surviving
indices, normalized weights, total mass, fallback/conditioning flags and
hard-mode legal actions after each feedback; independently record diverging
policy paths rather than requiring their later states to match.
Use a deterministic fixed-work finite control first; separately report
deadline-limited Fast/Strong resource behavior. Keep dates, data, candidate
rules, config apart from policy, and failure-as-seven scoring aligned, then
require paired all-game outcomes, zero failures/gaps, calibration and latency
guards across the permitted folds before considering promotion.

This frozen-belief experiment does **not** make finite search implement
dynamic staged recovery. The finite recursion currently memoizes a subset,
horizon and hard-mode history against fixed weights; a branch that activates
dormant fallback would need a full belief transition and identity in the
memo key. Passing a staged dynamic state directly into that kernel would
invalidate its values. Simply setting the production default to `finite_fast`
also changes both search and support, and earlier finite development means
were worse. The production policy remains staged while this experiment and
its tests are outstanding.

### Matched-belief development results

The diagnostic `staged_fixed_belief` mode keeps staged ranking while using
finite modes' fixed core/tail posterior; the selected `staged` mode is
unchanged. A toy fixture checks equal root support/weights and conditioned
feedback against finite fixed-work. The [matched-policy matrix](../config/experiments/september-matched-belief-fixed-work.json)
then isolates search choice on July 28-August 26. Its first (`benchmarks/predictive/september-matched-belief-fixed-work-v1.json`)
and repeat (`benchmarks/predictive/september-matched-belief-fixed-work-repeat-v1.json`)
JSON, with generated [first](generated/september-matched-belief-fixed-work.md)
and [repeat](generated/september-matched-belief-fixed-work-repeat.md) tables,
share input/config/matrix identities and declare no sealed-test evaluation.
All 30 paths and finite traces repeated exactly. Both profiles solved 30/30
without a coverage gap and had identical initial log loss 7.5789/Brier
0.9990. On this fixed belief, staged scored 3.2000 and finite fixed-work
3.5000; finite-minus-staged was +0.3000 guesses, paired 95% interval
[+0.1667, +0.4667], with 2/18/10 wins/ties/losses. Per-move p95 was
316.45 versus 154.23 ms in the first run. Finite did not beat staged
ranking on this development window; its bounded values are not a global
optimum claim.

A separate [belief-ablation matrix](../config/experiments/september-staged-belief-ablation.json)
keeps staged ranking in both profiles and changes only dynamic recovery
versus the frozen posterior. Its JSON (`benchmarks/predictive/september-staged-belief-ablation-v1.json`)
and [generated table](generated/september-staged-belief-ablation.md) report
30/30 solves and no gaps for both: selected dynamic staged 3.3333 versus
fixed-belief staged 3.2000. Fixed-minus-dynamic was -0.1333, paired interval
[-0.4000, +0.1333] (8/16/6 wins/ties/losses). The fixed belief worsened
initial log loss 6.5971 to 7.5789 and Brier 0.99858 to 0.99902, while
per-move p95 rose from 21.59 to 324.47 ms. The interval, calibration and
latency guards reject promotion despite the lower point mean. All 30
selected dynamic paths exactly matched corresponding dates in the earlier
current-source 12-fold baseline. Both release-mode 30-day runs took under
three minutes; a debug smoke was interrupted after exceeding its useful
budget, and no debug timing is used for ranking. No production default or
prospective seal was changed.

### Diagnostic-build verification and cleanup

Rust 1.97 formatting, warnings-denied all-target Clippy and all-target tests
passed after adding the diagnostic belief mode; the selected production mode
and `config/prior.toml` were not changed for this comparison. The current
release binaries are retained in `dist/`: GUI SHA-256
`CE2F034D04931BDF0FD0EB80950DBC6D0D2A0A18041EEF99C5D6CA564008C059`
and CLI SHA-256
`15AA0817A552B9D7D823019226481B74665951A91F067CE516876816F4E0E715`.
The CLI rolling-evidence documentation check, edited-doc local links and Git
whitespace check passed. Cargo's verified project-local cleanup removed the
rebuildable debug and release outputs (6.0 GiB and 859.8 MiB); `dist/` and
completed evidence checkpoints remain. Eight disposable smoke/capture scratch
files were also removed. A subsequent current-binary 12-fold replay (`benchmarks/predictive/september-v19b-staged-terminal-legal-current-binary-v1.json`)
and [generated table](generated/september-v19b-staged-terminal-legal-current-binary.md)
reproduced all 720 prior selected/v19b game paths exactly. Each profile
solved 360/360 without a gap; means remained 3.1944/3.1917 and the paired
candidate-minus-selected interval remained [-0.0222, +0.0167]. The run took
969.5 seconds (16.2 minutes) and evaluated neither consumed nor reserved
dates. This is current-executable development evidence, not prospective
validation or a promotion. Native GUI first-row visual/keyboard acceptance
remains open: an egui geometry regression covers draft and applied tiles at
861 and 1180 px, but the Windows session exposed a lock screen rather than a
usable app window. No native interaction is inferred from that test.

### September 26 pooled-exact and narrow-window handoff

Repository `master` remains at `1d607151f4a9` with uncommitted September
work (46 tracked changes and 128 untracked files at this handoff); nothing
was staged or pushed. The selected `staged` configuration is unchanged. The
first-feedback `OLATE/10001` delay was isolated in pooled exact ranking:
proxy preview took about 9 ms, while the preceding distributed CLI took
55.15 s for a top-ten full request. `src/solver/search.rs` now uses an
admissible weighted bucket bound to skip roots outside that requested exact
prefix, preserving coverage-first and exhaustive routes. Skewed, zero and
near-maximum finite mass tests, ranked-prefix tests, and a final 3.92 s CLI
call passed with the same ten words and exact costs. The discarded root
partial-cost experiment gave no meaningful speedup and is not retained.

The final-build 12-fold evidence (`benchmarks/predictive/september-staged-root-bound-rolling-v1.json`)
and [generated table](generated/september-staged-root-bound-rolling.md) cover
the allowed 720 development games only: all paths/outcomes matched the prior
binary by date and answer; selected/v19b each solved 360/360 with no gaps,
at 3.1944/3.1917 all-game guesses. The two-profile run took 491.2 s versus
969.5 s before; this is a speed result, not score improvement or prospective
confirmation. `src/gui.rs` now wraps the guess/feedback controls at the
minimum window width; a failing-then-passing production-path test found the
formerly missing fifth button. The original reported 1210 px board overlap
remains unconfirmed: the rebuilt app launched and returned a window, but the
Windows UI helper failed to activate that window on two fresh attempts, so no
trustworthy native visual or keyboard pass was possible.

Rust 1.97 formatting, warnings-denied all-target Clippy, all-target tests,
the rolling-evidence documentation check, the deterministic Optuna archive
check, 196 local Markdown links, and the GUI startup smoke passed. Release
GUI/CLI outputs were copied to ignored `dist/` with matching SHA-256 hashes:
`7F8964D32E102B0E939BD7D6883779B679439CE64A91BBA7CEDFC8534183C308`
(PE subsystem 2) and
`BCFB3B17DC2323B16EE55F48DB99497DEB506691BB2868EBE57F8F78C8E7A859`
(PE subsystem 3). Cargo profile-clean dry runs identified 6.1 GiB debug
and 1.1 GiB release output, both removed; small identity-bound checkpoints
remain for recovery. `dist/BUILD-INFO.txt` records the runnable artifacts.

Next: obtain a native visual/keyboard pass, implement and validate the
selected policy's earlier-turn six-turn failure-first/hard-mode contract (or
find a guarded finite replacement), then pursue prospective and macOS/remote
CI gates under their declared conditions. Review the remaining quick-xml
advisories before any release claim. The 128 untracked files include 86
historical/development benchmark artifacts (about 20.8 MiB); do not stage
them wholesale or delete non-reproducible evidence during hygiene work.

### Matched-belief Strong search follow-up

The [Strong matrix](../config/experiments/september-matched-belief-strong.json)
compared the selected staged ranking and finite Strong with the same frozen
core/tail posterior and books disabled on July 28-August 26. Its
JSON (`benchmarks/predictive/september-matched-belief-strong-v1.json`) and
[generated table](generated/september-matched-belief-strong.md) report 30/30
solves and zero coverage gaps for both. Initial log loss/Brier match at
7.5789/0.99902. Staged scored 3.2000 all-game guesses and Strong 3.5667;
Strong-minus-staged was +0.3667 with paired 95% interval [+0.1667, +0.5667]
(4/13/13 wins/ties/losses). Per-move p95 was 334.71 versus 2016.33 ms.
The 223.9-second two-profile run avoided the consumed and reserved windows.
This rejects a simple higher-budget finite promotion. A direct July 28
Strong request reported `Deadline`, `proposal_sampled=true`, and 18 evaluated
roots; its top suggestion was `ablet`, while staged's first move was `seria`
on all 30 matched-belief games. That discrepancy is a proposal/value audit
target, not evidence that `seria` has a better modeled failure value. The
finite route remains opt-in; production `config/prior.toml` is unchanged.

A focused proposal diagnostic on that same July 28 state found that the
top-4,096-answer proposal sample contained 94.2783% of posterior mass, but
reversed the full-posterior proxy ordering of `seria` and `aesir`. An
experimental full-support root anchor therefore made `seria` available to
Strong. Strong still ranked it sixth: its modeled failure probability was
0.009573 versus `ablet`'s 0.007211. In the five-date
anchor screen (`benchmarks/predictive/september-matched-belief-strong-anchor-screen-v1.json`)
([generated table](generated/september-matched-belief-strong-anchor-screen.md)),
Strong's five paths were unchanged and its 3.6000 mean still trailed staged's
3.0000. The anchor was removed from source rather than promoted. This narrow
screen does not establish behavior on the full 30-date cohort; it does show
that proposal omission alone did not explain those five observed losses.
Further work must check belief/value alignment before spending more search.

The matched-belief root traces make the quality tradeoff explicit. Across
30 games, fixed-work finite chose feasible upper-bound roots with mean modeled
failure probability 0.010060 and expected attempts 3.3813; Strong's values
were 0.007230 and 3.4672. Every root sampled proposals; fixed-work exhausted
its node budget and Strong reached its deadline. Both solved 30/30 in this
window, so the sample cannot establish that spending extra guesses to reduce
roughly one-percent modeled failure risk is worthwhile or that the risk model
is wrong. The next bounded comparison should use the existing dynamic-belief
finite transition path on the same allowed dates, record chosen modeled risk
and attempts against realized outcomes, and stratify dormant-tail activation.
That diagnostic is not a production-policy change or a reason to reopen the
protected seal.

The native Windows helper was retried against the sole running `Maybe Wordle`
window after a fresh window-list selection. Activation failed twice again, so
this was not a visual or keyboard pass. The finite experiment was removed;
22 focused release-mode finite tests and Rust 1.97 formatting passed. A fresh
CLI release rebuild had the same length as the retained `dist` CLI and differed
in only 24 PE metadata bytes (timestamp and linker metadata), with no evidence
of a gameplay-code difference; the existing verified `dist` executables remain
untouched and runnable.

### Dynamic-belief six-turn experiment and rejection

An opt-in finite kernel now carries staged active weights, dormant support,
fallback activation and recovery state through recursion. Its memo identity
includes these values and complete hard-mode history. Twenty-six finite tests
pass, including an `apply_feedback`-based recovery/duplicate-clue oracle and a
dormant-filtering work-budget test. The ordinary selected `staged` route remains
the previous ranking policy; a controlled finite request can exercise the new
kernel without changing the shipped configuration.

A seven-survivor development diagnosis exposed an independent bounded-search
quality problem: Strong's exact-threshold path enumerated root guesses in
dictionary order, so a 2 s deadline could spend its budget on an inert probe.
The exhaustive action set is unchanged, but it now evaluates the highest-mass
legal answer first. A failing-then-passing node-budget regression covers the
fallback. On `OLATE/11002` for August 15, the opt-in Strong result changed
from inert `jetty` to candidate `boule`; both runs still hit the deadline.
This narrow check is not a score gain or promotion.

An interim uncommitted routing trial applied dynamic finite search to the first
four turns of `staged` and left the two terminal turns unchanged. The allowed
July 28-August 1 five-game screen solved 5/5 at 3.2000 guesses versus 3.0000
for the retained staged policy. The July 28-August 26 development run solved
30/30 with no coverage gaps but scored 3.4333 versus 3.3333; initial
log loss was identical (6.5971), while per-step p95 latency rose from 21.59 to
259.22 ms. Nine game lengths changed: five worsened and four improved, for a
net three extra guesses. The trial failed the mean-score and latency guards,
so the selected route was restored; this is not a release finalist or a
flat-three result. The interim matrix and generated run files were removed
from durable evidence locations because the restored source cannot reproduce
them. These figures are a rejected development diagnosis, not final-build
evidence or a reproducible benchmark artifact.

A separate release-mode replay after restoration solved the same 30/30 games
at 3.3333, with all 30 paths identical to the earlier staged baseline and
p95 latency 21.08 ms. Rust 1.97 formatting, warnings-denied Clippy, and all
target tests pass after restoration. The first-guess board/suggestions geometry
regression still passes; native pixel and keyboard verification remains open.
No files were staged, committed, or pushed. Both `dist` executables were
rebuilt from the current uncommitted source after verification; copied hashes
match release outputs. The GUI/CLI SHA-256 values are
`23BF486FE9D1DBD1C525A4EFE6658F9F20B930AC1B7C886887B8571B8D1F5601`
and `73EF3D80D4F673E26AF8BEB94E4D3C179F930B29E47AEB4699D27BBB071110B7`.
The PE subsystems are GUI 2 and console 3. GUI startup and CLI help passed;
this does not close native visual/keyboard acceptance. After the hashes and
startup checks, Cargo profile-clean dry runs confirmed the scope; release
cleanup removed 2,118 rebuildable files (860.5 MiB) and dev cleanup removed
6,568 (6.4 GiB). `dist/` and identity-bound evidence checkpoints remain.

### Reproducible dynamic-belief finite comparison

The new opt-in `finite_fast_dynamic` profile is now recorded by a reproducible
two-profile matrix, [`september-dynamic-finite-matched.json`](../config/experiments/september-dynamic-finite-matched.json),
with the selected `staged` profile as reference and books disabled. The
five-day JSON artifact (`benchmarks/predictive/september-dynamic-finite-screen-v1.json`)
and [generated table](generated/september-dynamic-finite-screen-v1.md) cover
July 28-August 1: staged and dynamic finite both solved 5/5 with no gaps or
failures; their all-game means were 3.0000 and 3.2000, and latency p95 was
21.17 and 257.33 ms respectively.

The 30-day JSON artifact (`benchmarks/predictive/september-dynamic-finite-30day-v1.json`)
and [generated table](generated/september-dynamic-finite-30day-v1.md) cover
July 28-August 26. Both profiles solved 30/30 with zero gaps and failures:
staged scored 3.3333 and `finite_fast_dynamic` 3.4000. The dynamic-minus-staged
paired difference was +0.0667 guesses, with 95% interval [-0.1333, +0.2667]
and 4/21/5 wins/ties/losses. Latency p95 was 21.24 versus 256.62 ms; initial
log loss and Brier were identical at 6.5971 and 0.9986. Generation took
52.10 seconds and the process peak was 133.3 MiB. The artifact records
`sealed_test_evaluated=false`, so these 30 dates are retrospective development
evidence, not prospective validation.

The dynamic finite root traces chose `olate` on all 30 games, with mean modeled
root failure risk 0.004110 and expected attempts 3.3002; every root search hit
its deadline. Both profiles recorded five uniform-recovery steps. The run did
not pass the score or latency guard and does not promote the dynamic policy;
production `config/prior.toml` remains unchanged. The fixed-belief-only
`search-regret --finite` diagnostic must not be described as validation of this
dynamic mode, because its reference values use frozen-belief semantics.

### Latest current-source distributed-binary development replay

The completed current `dist` CLI (SHA-256
`ECFEE91B8199956E327954BD5E5BE233F07129A8C2CDE3D6F4612CA740B46B72`) ran the
[12-fold paired comparison table](generated/september-post-layout-tests-rolling-v1.md)
over the 12 allowed development folds and 720 games. It records
`sealed_test_evaluated=false`, zero coverage gaps/failures, and exact agreement
of all 720 outcomes and paths with the earlier
[post-dynamic-mode replay table](generated/september-post-dynamic-mode-rolling-v1.md),
which remains historical. Selected staged solved 360/360 at 3.194444 all-game
guesses with 20.7277 ms p95; v19b solved 360/360 at 3.191667 with 20.7633 ms
p95. V19b-minus-selected was -0.0027778 guesses, 95% CI
[-0.0222222, +0.0166667], with 6/347/7 wins/ties/losses. The run took
508.3 seconds and peaked at 168898560 bytes (about 161.1 MiB). The full
per-game JSON remains local pending a publication decision. This is
retrospective development evidence only; no promotion or flat-three claim
follows. The current source/binary identity supersedes the prior “latest” link.

### Earlier distributed-binary development replay (historical)

The rebuilt `dist/maybe-wordle-cli.exe` ran a fresh
12-fold paired comparison (`benchmarks/predictive/september-post-dynamic-final-rolling-v1.json`)
with its [generated table](generated/september-post-dynamic-final-rolling-v1.md).
It covered the same 12 allowed development folds and no consumed/reserved seal
dates (`sealed_test_evaluated=false`). Selected staged solved 360/360 with no
coverage gaps at 3.1944 all-game guesses; exploratory v19b solved 360/360 at
3.1917. All 720 game paths and scores match the preceding root-bound replay
exactly. V19b-minus-selected was -0.0028 guesses, paired 95% interval
[-0.0222, +0.0167], with 6/347/7 wins/ties/losses; no candidate qualifies
for promotion or a flat-three claim. The two-profile replay took about
510 seconds, below the requested 20-minute cap. Its step-latency p95 values
were 23.36 ms selected and 22.11 ms v19b. This is retrospective development
evidence, not prospective or sealed validation. The identity-bound checkpoint
is retained under `target/evidence-checkpoints/` for recovery.

### September 26 dynamic-mode phase handoff

The worktree remains on `master` at `1d607151f4a9` with extensive prior
uncommitted September work; nothing from this phase was staged, committed or
pushed. The opt-in `finite_fast_dynamic` route, matched-config matrix and three
versioned development evidence runs are retained; selected `config/prior.toml`
remains staged v20. Independent read-only reviews found no critical or
important defect in the new route. The next public-path test should exercise a
non-empty hard-mode history through fallback activation at the six-turn
boundary before broadening this experimental policy.

Rust 1.97 formatting, warnings-denied all-target Clippy, all-target tests and
the release build passed on the changed source. The rolling-docs check,
historical archive check, 233 local links across 41 Markdown files, and Git
diff whitespace check passed after documentation updates. The final `dist`
CLI passed an allowed-date suggestion after Cargo removed 860.5 MiB release
and 4.7 GiB dev profile output; both executables and all identity-bound
checkpoints remain. Native GUI activation still failed twice, so the actual
first-guess visual/keyboard acceptance remains open.

Publication is not automatic. These new JSON evidence files include per-game
target and path words; confirm redistribution rights or prepare a sanitized
aggregate artifact before staging them or relying on public README links.
Their generated tables are not yet part of the CI evidence-docs check. The
review also noted that `dist/BUILD-INFO.txt` records binary hashes but no
cryptographic dirty-tree digest, so it is local build provenance rather than
proof of an exact source-to-binary match. Next work is a guarded policy
candidate or further in-process profile, plus the native GUI, prospective,
macOS and release-file-scope gates. Do not open the protected seal or claim a
flat-three result.

### September 26 public-path, GUI-layout, and replay handoff

The worktree remains on `master` at `1d607151f4a9`, with prior September edits
preserved and nothing staged, committed, or pushed. The selected configuration
is still staged v20. A new public-API regression in
`tests/predictive_characterization.rs` exercises dynamic finite search with a
non-empty hard-mode history, dormant fallback activation, legal root actions,
positive modeled failure mass after activation, and the six-turn boundary. A
production `WordleGuiApp::update` regression in `src/gui.rs` checks an applied
first row, populated next draft, and a rendered recommendation at 135% text
scaling across 1180/1240/1260px. It found no overflow and required no layout
change. Native Windows activation still failed after a fresh launch, so pixel,
keyboard, and accessibility acceptance remains open.

Rust 1.97 `fmt --check`, warnings-denied all-target Clippy, and all-target tests
passed (311 library, 28 binary, 6 integration, and 14 predictive tests, plus
benchmark targets). Both release binaries were rebuilt and copied to ignored
`dist/`; GUI/CLI SHA-256 hashes are
`6152F8AE79CE31096914B1CD8A8FF8339B0956B3287360B6B5B3B1DCFEB829F5` and
`ECFEE91B8199956E327954BD5E5BE233F07129A8C2CDE3D6F4612CA740B46B72`.
The PE subsystems remain 2/3, CLI help and an allowed-date suggestion passed,
and the refreshed GUI launched. The new 12-fold result above completed in
508.3 seconds; its private JSON and identity-bound checkpoint remain local.
The rolling-docs check, Optuna archive check, 231 local Markdown links across
42 files, and diff whitespace check passed. Cargo's profile-scoped clean
removed 5.2 GiB dev and 860.5 MiB release outputs while preserving `dist` and
checkpoints.

Publication remains unresolved: the full JSON contains per-game words, and
CI's input at this checkpoint was an untracked private artifact. Do not stage
these full artifacts or push documentation that depends on them without a
rights decision or aggregate-only publication path. The selected six-turn
early-turn objective, guarded improvement, native GUI check, genuinely later
prospective validation, and native macOS/remote CI remain open. Do not claim a
flat-three mean or release completion from this retrospective replay.

### September 26 publication, CI, and dependency handoff

The current worktree remains uncommitted on `master` at `1d607151f4a9`.
Read-only publication review found 125 documentation links to 75 distinct
untracked predictive JSON files; some already-tracked historical JSON also
contains per-game fields. Keep full JSON and checkpoints private unless rights
are established. Markdown-only publication is the smallest proposed route,
but the aggregate-public scope and link migration remain undecided.

The pending CI workflow no longer reads the untracked
`september-finite-preordered-v1.json`. Its Windows job now checks the tracked
rolling Markdown fragment against the README marker section; the local check
passed and rejected an in-memory altered fragment. This verifies presentation
consistency, not regeneration of metrics from private raw evidence. The
README predictive section currently matches the untracked
`september-prior-family-evidence.md`, not the older tracked canonical fragment,
so its clean-checkout publication/check remains open. The pushed August 2 CI
run is green; it does not validate these uncommitted changes.

GitHub still lists seven open Dependabot alerts against the pushed lockfile.
The local uncommitted lockfile resolves all their reported patched versions:
`quinn-proto` 0.11.15, `rustls-webpki` 0.103.15, and `rand` 0.8.6/0.9.3.
The separate RustSec `quick-xml` 0.30.0 findings remain in the Linux
accessibility dependency tree; static inspection found its use through
`zbus-lockstep` validation macros and bundled XML, not a known remote XML
runtime path. That narrows the apparent product exposure but does not make
the global dependency audit pass or justify dismissing the findings.

No source code changed in this handoff. The current-source 12-fold score and
release binaries remain valid. The next decision is privacy-safe evidence
publication and whether score/latency guards or exact early-turn failure-first
semantics govern a production policy choice; native GUI, macOS, and later
prospective checks remain open.

The Windows desktop remained locked during native acceptance. A `PrintWindow`
capture or UI Automation bounds would not establish the eframe GPU pixels or
keyboard behavior, so neither substitutes for an unlocked-session visual pass.

A provisional clean-checkout file-scope audit found that tracked Markdown has
97 links to 63 untracked targets (55 JSON, seven Markdown, and
`config/evaluation.toml`). The current CI no longer requires any untracked
benchmark JSON, but a public source checkout must include the untracked
`src/solver/finite.rs`, `src/solver/online.rs`, evaluation config and release
ledger. Resolve the remaining links through a privacy-safe publication choice;
do not mistake successful local link checks for a self-contained release.

The final read-only subsystem diff review found no confirmed solver-core
regression. A suspected zero-base seed issue was withdrawn after checking the
documented primary-support contract: seed/manual answers remain eligible even
at zero modeled weight; dormant fallback is the separate valid-guess tail.
It did find that the current CI only compares the two rendered rolling-evidence
copies. The original tracked comparison artifact fails the current CLI parser
with `missing field top`, while newer current-schema comparisons remain local
and untracked. A privacy-safe aggregate source or an explicit historical
migration is needed before this can again verify the published metrics from
source. The minor CLI `--proxy-preview` help text also omits that combining it
with `--search-budget` invokes a 30 ms finite search; defer this wording change
until a necessary source rebuild so a cosmetic binary change does not
invalidate current executable-identity evidence alone.

The conditional-cohort audit confirms that provisional Pareto-rank-0
small-state and proxy-risk candidates existed but failed downstream repeat or
paired validation; there is no qualified promotion finalist. The seven-profile
current-schema gate remains unrun after an earlier 60/2,520-game probe
projected about 48 minutes, beyond the requested roughly 20-minute ceiling.
That old projection predates the pooled root bound and is not a measurement
of the current executable. No new costly study was run here.

The later survival folds can include the consumed June 18-July 17 outcomes as
chronological training labels while excluding that interval from validation
targets. This is not future-target leakage. On September 27 the owner approved
this chronological-training-only interpretation: the interval may train
later-date models but must never become a new tuning or validation target.
The ruling confirms existing fold behavior; it does not promote the survival
model or change any historical evaluation result.

The current-source `dist` CLI (SHA-256
`ECFEE91B8199956E327954BD5E5BE233F07129A8C2CDE3D6F4612CA740B46B72`)
completed a seven-profile runtime probe over July 28-August 5: 63/63
profile-games in 69.8 seconds, `sealed_test_evaluated=false`, with separate
scratch output and checkpoint under `target/`. Those three completed-probe
scratch files were removed after recording the aggregate timing. Multiplying that short-window
rate by 40 projects roughly 46.5 minutes for all 2,520 profile-games. This
is only a linear runtime estimate: dates and profile work vary, and the
nine-day scores are not promotion evidence. It corroborates that the full
seven-profile gate is above the user's roughly 20-minute ceiling, so the full
run was not started. The ordinary `rolling-evidence-docs` verifier passed
locally against the matching schema-4 `september-finite-preordered-v1.json`;
that raw source remains untracked/private, so the clean-checkout CI check
still verifies presentation consistency only.

A read-only paired-path audit of the existing July 28-August 26
`september-matched-belief-fixed-work-v1.json` compared only profile moves,
not target words. Both profiles had 30 unique dates and nonempty paths; their
first guesses differed on all 30. The finite runner proposes its own
proxy-ranked shortlist and state-local baseline move, not the matched-belief
staged chooser's move or continuation. This is a concrete reason to test a
staged-seeded finite proposal/rollout as an opt-in diagnostic; it is not
evidence that such seeding improves paired score, latency, calibration, or
the exactness of deadline-limited finite values.

An independent read-only audit of the allowed fixed-state latency evidence
found a six-turn root with 14,855 survivors in
`release-gameplay-latency-bound-pruning-v1.json`. The finite Fast/Strong
routes sample proposals and hit deadlines there; their values are not exact
global six-turn action proofs. The root is too broad for the proposed
horizon-aware bucket-size certificate, while the retained pooled root bound
optimizes the separate unlimited-horizon scorer. This does not prove an exact
early-turn implementation impossible, but no measured route currently meets
the selected staged policy's roughly 21 ms p95 and its score guard. Keep the
selected-policy objective checkbox open rather than silently changing the
objective or promoting a bounded diagnostic.

### September 26 first-guess and fixed-work follow-up

The full production-layout test now includes the actual empty-history first
guess at 1180 and 1210 pixels, plus applied-row cases at 1240 and 1260 pixels,
with a populated recommendation at 135% text scaling. The focused regression,
Rust 1.97 `cargo fmt --check`, warnings-denied all-target Clippy, and all-target
tests passed (311 library, 28 CLI, six integration, and 14 predictive
characterization tests). This verifies egui geometry, not native GPU paint or
keyboard behavior. The retained `dist` GUI launched again, but the Windows
helper failed to activate its window on two fresh attempts; native acceptance
remains open.

The matched-belief fixed-work artifact records 30/30 finite root searches with
sampled proposals, seven completed first-turn candidates, and a 4,000,000-work-
unit stop. None of the recorded first-turn candidate lists includes the matched
staged opener. This strengthens the case for a bounded staged-root
counterfactual, but neither its finite value nor its paired play outcome has
been measured. Production staged v20 remains selected.

Both Rust 1.97 release binaries rebuilt successfully. The GUI and CLI outputs
have the same lengths as the retained `dist` copies and each differs in 24 PE
metadata bytes only; this turn changed a test, not product code. The retained
`dist` executables and their existing `BUILD-INFO.txt` hashes remain intact and
runnable. Cargo's profile-specific dry runs identified 4,761 dev files
(4.3 GiB) and 2,118 release files (860.5 MiB); `cargo clean --profile dev`
and `cargo clean --release` removed only those rebuildable outputs. Evidence
and rolling checkpoints under `target/` remain. No files were staged,
committed, or pushed.

### September 26 continuation: diagnostic and release-gate limits

A staged-root seed for finite search is still only a hypothesis. The existing
matched-belief 30-day comparison scored staged 3.2000 at 316.45 ms per-move
p95 and fixed-work finite 3.5000 at 154.23 ms; the staged opener was absent
from every finite first-turn candidate list. Recomputing the staged choice
inside a 250 ms finite request cannot be assumed to meet that wall-clock
budget. A proposed new configurable mode was stopped after its expected
test-first parse failure, then removed; the seven existing config tests and
`cargo +1.97.0 fmt --check` pass. No production search mode or selected
config changed. A future diagnostic must count staged proposal time in the
same request budget, retain hard-mode legality, and compare paired play on
allowed dates before any score claim.
This records the September 26 status; the bounded September 28 seed
counterfactual below was negative and did not become a production mode.

The seven-profile 2,520-game run still projects about 46.5 minutes from the
current-source 63-game probe, requiring at least a 2.33x speedup to fit the
requested roughly 20-minute ceiling. Profiles currently run sequentially
while eligible games and ranking already use Rayon; the checkpoint requires
an ordered profile prefix and has no concurrent-writer merge. Parallel CLI
copies with a shared checkpoint are unsafe, and independent scratch matrices
would change evidence identity. No full run or unsupported speedup claim was
made.

The native GUI window was discoverable again, but the Windows helper returned
`failed to activate captured window` after refreshing its window selection
and retrying once. Pixel/keyboard acceptance remains open. A release-hygiene
audit also found that current CI only compares the README rolling fragment
with its Markdown copy; it no longer recomputes that fragment from the JSON
outcomes. This is a real evidence-gate gap before push. The comparison JSON
and newer benchmark JSON contain per-game target/path words, so restoring a
source-backed public gate requires an explicit privacy-safe publication path
or redistribution-rights decision; matching text alone is not arithmetic
verification. Local links, `git diff --check`, and retained `dist` hashes pass,
but the worktree is not a blanket-stage release set. No files were staged,
committed, or pushed.

The exploratory config test left 2,754 rebuildable dev-profile files
(2.1 GiB). A verified `cargo +1.97.0 clean --profile dev` removed only that
profile; `dist` hashes still match `BUILD-INFO.txt`, and the evidence and
rolling checkpoint directories remain under `target/`.

The CI evidence gap above was then repaired locally without publishing raw
answer/path words. At this checkpoint,
`scripts/redact_public_evidence.ps1` produced two public
copies under `docs/evidence/` for the current selected-policy benchmark and
the earlier finite rolling comparison. The script preserves dates, outcomes,
numeric calibration, path lengths and source/config identity, replaces each
game target and path word with `[redacted]`, and adds an explicit marker. Its
generation, deterministic regeneration and read-only structural check passed;
the privacy scan found no quoted raw target/path words in the copies. Both
existing Rust documentation verifiers pass on the public copies. README's
generated predictive section now comes from the current-source 360-game
selected/v19b comparison, rather than the older prior-ablation table; the
separate generated fragment matched byte-for-byte before the README update.
CI now checks redaction plus JSON-backed rolling and predictive arithmetic,
not merely two equal Markdown fragments. These checks cannot authenticate
the original private data or replay Wordle from the redacted guesses. The
public copies, redaction script and current generated fragment are still
untracked until a deliberate release file-scope decision. The historical
documentation-link cleanup and remaining intentional bundle dependencies are
recorded in the handoff below. No raw JSON was staged or published.

### September 27 handoff: evidence validation and reviewed docs scope

On `master`, the worktree remains dirty and unstaged. The private benchmark
JSONs remain local. Tracked documentation no longer links to untracked raw
predictive JSON; those filenames are plain local-only provenance. Links to the
new release ledger, aggregate generated summaries, experiment configs, and the
two redacted evidence copies then available are intentional prospective
release dependencies and
must be included together if a commit is authorized. The 32 generated
September summaries contain aggregate metrics, not per-game target/path words.
Two explicit target-answer names were removed from this ledger; remaining
five-letter examples describe solver guesses or candidates. No public-rights
decision or staging occurred.

Independent review found two more evidence-gate defects, both fixed with
failing-then-passing regressions: benchmark artifacts now reject a true
`sealed_test_evaluated` flag before rendering a no-seal claim, and rolling
games now require path length to match reported guesses even when posterior
calibration is empty. Empty coverage-gap paths remain valid. Rust 1.97
formatting, warnings-denied all-target Clippy, all-target tests (314 library,
28 CLI, 6 integration, 14 predictive-characterization, plus benchmark smokes),
the redaction structural check, and both JSON-backed documentation verifiers
pass. The public checks recalculate internal arithmetic but do not authenticate
the private raw source or every reported telemetry value; their scope is stated
in README.

Both Windows release executables were rebuilt from the validator-fixed source
and copied to ignored `dist/`. GUI SHA-256 is
`D0A16F9E13A025AA187AA74AB011129821B81FE39A9D665B70DFFED7F53A3B96`
(PE subsystem 2); CLI SHA-256 is
`2DC983689CA19C63ECAD82D48D8D68BC9322198D6200904A0DF0367E72DBE3BA`
(subsystem 3). Source/output hashes matched, CLI help worked, and the GUI
stayed running through a three-second startup smoke. This source change only
affects evidence validation; the 3.1944/3.1917 rolling scores belong to the
preceding binary, and no new full gameplay replay was run under these hashes.
After verifying Cargo's target directory was a non-reparse child of the repo,
profile dry runs scoped 6,097 dev files (6.0 GiB) and 2,118 release files
(860.4 MiB); both rebuildable profiles were removed. `dist/`, evidence and
rolling checkpoints, and other non-profile target contents remain. The latter
include 42 Criterion files (17.8 KiB) and two temporary documentation previews
(64.4 KiB). Their paths and lack of reparse points were inspected, but the
recursive removal command was blocked by execution policy; no alternate
deletion route was used.

Open for full release acceptance: native GUI visual/keyboard interaction on an
unlocked desktop, CI and dependency-alert inspection after the authorized
`master` integration push, a seven-profile 2,520-game gate that currently
projects about 46.5 minutes (above the user's roughly 20-minute ceiling), and
genuinely prospective evidence from a later window. The reviewed push must
include linked public dependencies while excluding raw word-bearing JSON. Do
not turn the integration push into a production-policy promotion or a
flat-three claim.

### September 27 pre-push dependency and Windows build verification

The final pre-push audit found two high-severity RustSec advisories against
`quick-xml` 0.30.0 in the Linux GUI accessibility dependency chain. The
approved bounded fix updates `eframe` from 0.31 to 0.32.3 without disabling
AccessKit or changing product code. Cargo now resolves `zbus_xml` 5.2.1,
which does not depend on the old XML parser. The refreshed
`cargo audit --file Cargo.lock` exits successfully with no vulnerabilities and two
unmaintained warnings (`paste` and `ttf-parser`). Linux-target dependency
inspection confirms the vulnerable parser is absent and accessibility remains
enabled; an actual Linux build was not run on this Windows host.

After that update, Rust 1.97 all-target check, formatting, warnings-denied
Clippy, and all-target tests pass (314 library, 28 CLI, six integration, 14
predictive characterization, plus benchmark smokes). The structural public
evidence check and both JSON-backed README verifiers pass. A new locked
Windows release build supplies ignored local `dist/maybe-wordle.exe` (GUI
subsystem, SHA-256
`ACC42C2446E0FF72E90C5DF24A5198590A838A2AFACB96C1C5473D8D2419388A`)
and `dist/maybe-wordle-cli.exe` (console subsystem, SHA-256
`1172F40705EE4DEBFB719DD91FE41A4C95F050F1E70D6E67313BFA7076178810`).
Both hashes match the release outputs; CLI help and a three-second GUI startup
smoke pass. This is not a native visual/keyboard acceptance test, and no new
gameplay replay was run after the dependency-only change. The selected policy
and development scores are unchanged; neither production promotion nor a
three-guess claim follows from this build.

### September 27 clean-checkout CI correction and native interaction

The first integration push (`7d799719`) passed the macOS native memory job,
but its Windows CI job failed only at the predictive documentation verifier:
the checked-out Markdown used CRLF while the generated fragment used LF. The
rolling verifier already normalized this platform difference. The predictive
verifier now does the same for both its generated fragment and README check,
with regression tests for equivalent CRLF/LF text and genuinely stale content.
Rust 1.97 formatting, warnings-denied Clippy, all-target tests, public
redaction, and both source-backed documentation verifiers passed locally after
the correction. The next clean-checkout CI result is the remote acceptance
check; the initial failure is not erased by the local pass. The Dependabot API
reported zero open alerts after the first push.

An unlocked-desktop pass of the earlier, behavior-identical GUI build showed
first-row draft and applied tiles inside the board column at about 1180px,
beside visible recommendations. Enter-to-apply, Undo, Hard Mode, suggestion
inspection, and Reset worked. This does not establish narrow-window or
enlarged-text native behavior, nor a timed first-feedback latency result. The
verifier-only source correction was rebuilt into ignored local `dist/`; GUI
and CLI SHA-256 are respectively
`0653CF34922CA765ECE19E9EF46DD9B621A214BE270CD6E069F92A60BE9F5750`
and `42D3FB6846CF7ECB22647F7692109AABE9678040644870A015B2ACF13CC37D81`.
Both copied hashes matched the release outputs, CLI help passed, and the GUI
remained running through a three-second startup smoke. No candidate policy,
score, or prospective-validation claim changed.

### September 27 current-executable dynamic-policy disagreement check

The retained Windows CLI at revision `23a73e0` repeated the selected-staged
versus `finite_fast_dynamic` matrix on July 28-August 26 development dates
with books disabled. It finished 60 profile-games in 67.48 seconds, within the
20-minute ceiling, and did not evaluate the declared seal. The full word-bearing
report is local-only at
`benchmarks/predictive/september-dynamic-finite-current-30day-v1.json`
(SHA-256
`281F1EA0D4FD9109FF241165C9F88DFF2B7A6859364D5B755BFCE2C337D25BEC`).
Its source revision is recorded with `code_dirty=true`, so this is a diagnostic,
not a clean-source release artifact. No production config changed.

Both profiles solved 30/30 without coverage gaps. Staged scored 3.3333
all-game guesses at 21.52 ms per-move p95; dynamic finite scored 3.4000 at
256.20 ms. The paired finite-minus-staged difference was +0.0667 guesses,
95% block-bootstrap interval [-0.1333,+0.2667], with 4/21/5
wins/ties/losses. All 60 per-game paths and outcomes matched the
earlier-source artifact exactly.
All first guesses matched; 25 games first disagreed at guess two and five
never disagreed. Each divergent finite choice had `deadline` status and only
an `upper_bound` top-candidate value; those 25 games split 4 finite wins,
16 ties, and five staged wins. Later turns after a divergent guess do not
represent shared states. This replay does not establish a turn/state class
where a live finite router improves the guarded outcome on the observed paths.
The fixed-belief regret tool still cannot serve as an exact dynamic-belief
reference; the separate opt-in same-state diagnostic and its narrow scope are
recorded below. No switch is implemented or claimed from this result.

### September 27 remaining performance and seven-profile gate check

The focused Rust 1.97 Criterion four-feedback benchmark passed on the current
source at revision `23a73e0`: its slope estimates were 0.302 ms for history,
13.399 ms for preview and 18.275 ms for full staged suggestions. The 4.88 ms
full-minus-preview difference is not native paint time, and removing that
refinement without replacement would remove exact-cost details exposed in
the CLI and GUI.
No solver code or selected config changed.

A current-executable `benchmark-evidence` run covered the seven-profile matrix
on July 28-August 5 only. It completed 63/63 profile-games in 64.57 seconds,
with `sealed_test_evaluated=false`. The word-bearing raw diagnostic remains
local at `target/evidence-checkpoints/september-seven-profile-runtime-current-v1.json`
(SHA-256 `616AFF4DB0E830147D25959EA618E837F40BEB0D55DAD30FC6F163152D6A956F`).
The two staged profiles took 21.5 and 23.6 seconds, about 70% of the total;
a linear full-matrix projection remains above 40 minutes and is not a full
run. Their nine-game paths matched, but the artifact modes differ and the
current checkpoint protocol cannot merge independently run profile reports.
An eight-Rayon-worker repeat of the same nine-day slice took 64.17 seconds
versus 64.57 seconds with 16 workers; input/matrix identities and all game
paths matched. Thread-count tuning therefore provides no material reduction.
At this earlier probe, the effective Rayon count was logged but omitted from
checkpoint identity and artifact metadata. Time/memory ceilings appear in the
final artifact but are also omitted from checkpoint identity, so resumed
profiles could have run under different ceilings. Keep all three values fixed
across a resumed run; the checkpoint format must reject changed-resource
resume explicitly.
The seven-profile acceptance gate remains open under the 20-minute ceiling.

The focused GUI regression suite passed 22/22 tests. A retained `dist` GUI
startup smoke succeeded, but this Windows session's computer-control service
did not expose any native windows, so compact-width, 135% text, keyboard and
end-to-end accessibility acceptance remain unverified here.

### September 27 prospective-freeze workflow audit

`freeze-candidate` freezes the candidate in a current-schema rolling
comparison only with full coverage, zero failures and a paired upper
confidence bound below zero. The current v19b comparison does not qualify
and cannot freeze the selected staged incumbent; a fresh eligible comparison
with staged as its candidate would be required. `evaluate-sealed` is tied to
the prior global once-only marker, already consumed for the June 18-July 17
test. The current implementation preflights source/plan identity, solver setup,
and exact date coverage before atomically reserving that marker with exclusive
creation. It still must not be repurposed for a later window as-is.
No held-out outcomes were inspected or evaluated in this audit.

A genuinely later window requires a distinct dated freeze and consumption
record, the same complete-date preflight and irreversible exclusive reservation
for a distinct per-window marker, and fresh plan/source identities. If a valid
candidate is frozen by September 27, before the September 28 target is
available, September 28-October 27 is the conservative earliest 30-day
window; a later freeze shifts it. The development cutoff remains August 26;
the old August 28-September 26 seal must not become development tuning data.
No candidate has been frozen and no prospective window has been consumed.

Phase handoff: `master` at `23a73e0` has only documentation tracked edits in
`TODO.md`, `docs/PERFORMANCE.md`, and this ledger. The focused Rust 1.97 GUI
suite (22 tests), four-feedback Criterion run, both nine-day runtime probes,
artifact identity/path comparison and `git diff --check` passed. The new
runtime-probe reports remain private under `target/`; both ignored `dist`
executables remain. Next:
resolve the consumed-window training-history rule, design a complete-date
prospective preflight and dated freeze record, and profile the staged exact
metric rescans before changing solver behavior. The native UI gate and full
seven-profile run remain open; no commit or push was made.

### September 27 checkpoint resource-identity correction

Evidence checkpoint schema/hash domain v4 now records and binds the effective
Rayon worker count and maximum time/memory ceilings. A resumed run rejects
changed resource settings or a v3 checkpoint before accepting its completed
profile prefix. A raw v3 JSON regression checks the explicit unsupported-schema
error, and a malformed v4 JSON lacking resource fields cannot resume. The
final benchmark artifact schema remains v7; its resource budget is recorded,
but its effective Rayon worker count still resides only in the run log and
checkpoint. Existing v3 checkpoints remain private evidence but cannot resume
under v4; this is not a seven-profile completion or a solver-score change.

After the change, `cargo +1.97.0 fmt --check`, warnings-denied all-target
Clippy, `cargo +1.97.0 test --all-targets` (318 library, 30 CLI, 6 integration,
14 predictive-characterization tests and benchmark smokes), and
`git diff --check` passed. The source change is confined to
`src/solver/eval.rs`; the prior runtime probes and retained `dist` executables
still reflect the preceding binary. No rebuild, freeze, sealed evaluation,
commit, or push occurred. The next prospective design must require complete
date coverage before marker creation and exclusive per-window acquisition;
the existing global marker remains completed for the old test.

### September 27 maintenance and native-window recheck

The obsolete Python optimizer is absent, and the deterministic archive check
passes. The formal pattern table is untracked and reproducible. Cargo declares
Rust 1.97, CI pins 1.97.0 and uses `actions/checkout@v7`. A local
`cargo audit --no-fetch` found no vulnerabilities (only unmaintained-crate
warnings). A fresh read-only GitHub check reported zero open Dependabot alerts;
CI for `23a73e0` completed successfully in both the Windows `rust` and native
`macos-memory` jobs. This closes those maintenance and macOS-sampler checks,
not macOS GUI validation or the separate Linux/headless build-contract question.

The retained Windows GUI launched and exposed one `Maybe Wordle` window. Its
accessibility tree named the controls and represented board cells as text, not
buttons. The Windows helper could capture that tree but could not activate the
window; the screenshot showed the desktop. A refreshed window selection and
one raise/retry also failed. Pixel layout and keyboard behavior therefore
remain unverified. No in-app controls were changed; no new benchmark, sealed
evaluation, commit or push occurred. Branch `master` remains at `23a73e0`
with the earlier checkpoint/source and documentation edits; `dist` is preserved
but predates the checkpoint-v4 source change. Next: decide the headless support contract,
obtain a usable native GUI interaction session, and continue the candidate
and prospective gates without opening protected dates.

### September 27 sealed-marker preflight hardening

The existing, consumed once-only evaluator now builds its solver and checks
for exactly one history date on every day of the declared inclusive window
before creating a marker. Marker acquisition uses exclusive `create_new`,
followed by a file sync and, on Unix, a parent-directory sync; a failed
post-acquisition write or sync remains fail-closed. Toy regressions cover
missing/duplicate dates and two simultaneous acquisition attempts. No real
sealed or prospective evaluation ran. This fixes the old evaluator's preflight
and race defects, but does not introduce a dated prospective freeze/marker or
change the incumbent promotion rule. Windows new-file pathname durability
across power loss remains unproven; see [`PERSISTENCE.md`](PERSISTENCE.md).

With this source change, Rust 1.97 formatting, warnings-denied all-target
Clippy, and all-target tests passed (320 library, 30 CLI, 6 integration,
14 predictive-characterization tests plus benchmark smokes). Public evidence
redaction and both CI documentation verifiers passed, as did `git diff --check`.
The source edit is still only `src/solver/eval.rs`; `dist` still contains the
earlier runnable executables and must be rebuilt after source work is final.
The full seven-profile gate, selected-policy candidate, native GUI acceptance
and prospective freeze/evaluation remain open. No staging, commit or push.

The proposed bucket-size zero-failure certificate was audited but not run as
a study. The existing CLI does not report eligibility; a temporary diagnostic
could count selected staged moves on allowed development states, but that would
not enumerate alternative legal roots or prove a better choice. The 14,855-
survivor opening state already exceeds the certificate's 1,211-survivor
six-turn ceiling. No production routing change follows from this audit.

### September 27 dynamic-finite fallback partition experiment

The opt-in finite kernel now groups dormant fallback answers by feedback once
per candidate guess and passes the corresponding bucket to each positive-mass
non-green child. It preserves the existing activation, recovery, hard-mode and
memo transitions; direct child calls still perform their own checked fallback
scan. The TDD duplicate-letter regression failed on the preceding code at
work unit 31 under a 30-unit cap, then passed with three distinct matching
fallback branches and a 42-unit cap. All 27 finite-module tests passed on Rust
1.97. An independent read-only review found no correctness defect. It did
flag that eager bucketing may waste work if an early branch prunes; a
completed value is unchanged, but the bounded search may stop at a different
root because work and elapsed time have changed.

The matched retrospective matrix was
`config/experiments/september-dynamic-finite-matched.json`, July 28-August 26,
30 development dates, a 300-second evidence ceiling, and unchanged
config/matrix/history snapshot/resource settings. The old release CLI SHA-256
was `3F6409143FF9545DCBA54741E90F5EBDE19D736FFCF55A6F6386D4F616C2DDAD`;
the changed CLI was
`5824C5643F1D70A76670069DCB01B22973B79FCFFE15D8B381C9C07FC37C5897`.
After a Clippy-only annotation, the final rebuilt CLI was
`BA31A05334B38A77A567DAA027109D29068A89B37B45C9D29133C9535DF78C0A`;
it received its own matched replay. All four accepted reports are private
under `target/diagnostics/` with `fallback-partition-` filenames; they
contain word-bearing paths and are not public artifacts.
An initial old-binary run exited 1 after all games because its source
identity changed while a test fixture was edited; it was discarded. The
source was then held fixed and the clean old-binary run completed before
the changed binary was built. All four accepted reports have
`sealed_test_evaluated=false`, 30/30 solves and zero coverage gaps/failures
for both profiles. The old and first changed run had 30/30 matching dates,
targets, and staged paths.

| Executable/run | Staged mean | Dynamic finite mean | Finite minus staged, 95% paired interval | Finite per-move p95 | Mean finite work units/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old, clean baseline | 3.3333 | 3.4000 | +0.0667 [-0.1333,+0.2667] | 258.56 ms | 8.40M |
| One-pass partition | 3.3333 | 3.3000 | -0.0333 [-0.2333,+0.1333] | 260.68 ms | 7.78M |
| Same changed binary, repeat | 3.3333 | 3.2667 | -0.0667 [-0.3333,+0.1667] | 264.54 ms | 7.65M |
| Final rebuilt CLI | 3.3333 | 3.3667 | +0.0333 [-0.2000,+0.2333] | 263.28 ms | 7.53M |

The changed run's finite first move differed from the old run in all 30
games, and its paired old-to-new outcome was seven improved, 17 tied, six
worse. The two changed-binary repeats matched 27/30 finite paths and first
moves; the final rebuild matched 28/30 paths against the first changed run
and did not retain its point-score lead. This is deadline-sensitive search,
not a changed belief formula; all paired intervals still cross zero. The
finite p95 remains roughly ten times
the staged p95 and near the unchanged 250 ms root deadline. This retains a
justified opt-in work reduction, but does not establish a selected-policy
score or latency win, permit promotion, or approach a validated flat-three
mean. The same-state dynamic-regret diagnostic is now available for small
development states; the seven-profile and prospective release gates remain
open.

After the code change, `cargo +1.97.0 fmt --check`, warnings-denied all-target
Clippy, and `cargo +1.97.0 test --all-targets --quiet` passed (321 library,
30 CLI, 6 integration and 14 predictive-characterization tests plus benchmark
smokes). Clippy initially exited 1 for the new eighth transition argument;
a targeted lint explanation resolved it, without changing runtime behavior.
Both Windows release executables were rebuilt from the final source and copied
to ignored `dist/`; copy hashes match the release outputs (GUI
`DE3BA0B5F339F8DED4F1B8F8206F5FFCD04D19441F90447FBF62AF75EDEA4D8F`,
CLI `BA31A05334B38A77A567DAA027109D29068A89B37B45C9D29133C9535DF78C0A`).
Their PE subsystems are GUI 2 and console 3, and the retained CLI help runs.
The native GUI attempt was stopped by the user before a trustworthy layout
capture, so its remaining narrow/enlarged-text/accessibility acceptance stays
open. After dry-run scope and reparse-point checks, profile-scoped Cargo
cleanup removed 2,556 release files (1.1 GiB) and 6,975 dev files (6.8
GiB). Ignored `dist/`, diagnostics, Criterion data and identity-bound
checkpoints were retained. No staging, commit, push, seal evaluation, or
candidate promotion occurred.

Earlier phase handoff (at that checkpoint): `master` was at `23a73e0` with eight modified tracked
files, no staged files, and local `AGENTS.md` plus 94 private predictive JSON
reports still untracked. The independent final checkpoint/sealed-evaluator
review found no correctness issue. The last code change passed Rust 1.97
formatting, warnings-denied Clippy and all-target tests; after the final
documentation edits, both retained-CLI evidence verifiers, public redaction,
105 local documentation links and `git diff --check` passed. Cleanup preserved
`dist/`, diagnostics, rolling/evidence checkpoints and Criterion data. Next:
obtain an uninterrupted native narrow/enlarged-text GUI pass, resolve the
consumed-window training-history rule and prospective-control choice, find a
candidate that clears the paired quality/resource guards, then run the
seven-profile gate only if its projected wall time meets the user's limit.

### September 27 same-state dynamic-regret spot checks

The opt-in `same-state-dynamic-regret` command requires a staged config and an
explicit development date/turn. It replays the artifact-free staged path to
that turn, then compares the staged and `finite_fast_dynamic` choices at the
same dynamic belief against an exact full-legal-root reference. Exact
references are available only when combined active plus dormant-fallback
support is at most six; the path and reference share one wall-clock budget.
These isolated decisions are diagnostic, not whole-policy, prospective, or
promotion evidence. Raw reports and target words are omitted here.

| Development date / turn | Active survivors | Choice comparison | Exact-state result |
| --- | ---: | --- | --- |
| 2026-08-24 / 4 | 6 (dormant: 0) | Different | Staged: failure 0, attempts +0.1114125; dynamic: both 0 |
| 2026-08-24 / 5 | 3 | Different | Zero regret |
| 2026-08-26 / 5 | 4 | Same | Zero regret |
| 2026-08-06 / 4 | 5 | Same | Zero regret |

An August 7 turn-4 invocation with a five-second shared budget exited with a
global-deadline error before producing a usable report. It is not a solver-
quality result and contributes no regret evidence. The observed exact cases do
not justify a production change. States above the six-survivor limit and
broader score, latency, seven-profile, and prospective gates remain unresolved.

An evenly spaced turn-four screen used the retained CLI SHA-256
`2F5EA98E232BA52AF2F12964C0233F446153CD92B6B10D480830B7EB4022E4F7`
and `config/prior.toml` on ten allowed development dates, July 30 through
August 26 at three-day intervals. Each invocation had a 15-second shared
budget; the two deadline-only replays were retried at 45 seconds without
changing source or config. Eight staged paths had already solved before turn
four, August 20 had four combined survivors and both choices had zero exact
regret, and August 26 had eleven survivors, above the exact-reference limit.
The screen therefore found no additional exact disagreement to justify a
router. Solved-before-turn and oversized cases are not zero-regret samples;
the raw summary-only reports remain under ignored `target/diagnostics/`.

Current phase handoff: the opt-in same-state diagnostic and documentation are
uncommitted. Rust 1.97 all-target tests passed (325 library, 31 CLI, 6
integration, and 14 predictive-characterization tests plus benchmark smokes).
The retained GUI and CLI hashes are
`33C13CAD8AEB218C909D35D1F46ED47FCA114FCDA7842F9C73B028BCB136F98A` and
`2F5EA98E232BA52AF2F12964C0233F446153CD92B6B10D480830B7EB4022E4F7`.
Documentation checks passed: public-evidence redaction, both CI evidence-docs
verifiers, and `git diff --check`. The first unpinned verifier invocation used
Rust 1.94 and failed the crate's Rust 1.97 minimum; both checks passed when
rerun with `+1.97.0`. Cargo's checked, profile-scoped cleanup removed 6,585
dev files (6.4 GiB) and 2,115 release files (851.5 MiB), while retaining both
`dist` executables, diagnostics, checkpoints and Criterion data. The new Linux
CLI CI job has not been validated by hosted CI and remains pending push. The
six-support limit, seven-profile and prospective gates remain open; no candidate
 promotion, commit, or push occurred.

### September 27 consumed-history decision and legacy CLI guard

The owner approved using the consumed June 18-July 17 outcomes only as
chronological training history for later target dates. They remain excluded
from new tuning and validation targets; the existing rolling-fold construction
already follows this distinction. The older `backtest` and `experiments` CLI
commands now require both date bounds and validate the declared development
policy before constructing a solver or loading answer history. Shared policy
regressions reject overlap with the consumed interval and the reserved seal,
while asserting that the latest allowed fold includes the consumed dates in
its earlier training range. No protected target run was made.

The changed source passed Rust 1.97 warnings-denied Clippy, formatting, and
all-target tests (326 library, 32 CLI, 6 integration, 14 predictive
characterization, and benchmark smokes). Public evidence redaction and both
CI generated-document verifiers passed. This protects legacy evaluation
entry points; it does not create a later prospective freeze, qualify a new
policy, or change the selected solver.

### September 27 staged zero-failure certificate screen

The new opt-in `staged-zero-failure-certificate` command replays artifact-free
selected staged decisions on explicit allowed development dates. For a chosen
root with remaining horizon `h`, it rejects dormant fallback at the root and
in every non-green exact child; each child must have at most `h-1` active
answers, all dictionary-guessable and hard-mode legal, with positive modeled
weight so later sequential replies remain usable. This is a sufficient
modeled-failure-zero witness for that selected root only, not a global action
optimum, a compact stored proof, or coverage of out-of-support answers. The
diagnostic leaves production routing and prior configuration unchanged.

The current-source August 20 one-day smoke had 1/1 history date and game
replayed, four selected/evaluated roots, zero certified, three dormant-support
and one unstable-modeled-support rejections, no coverage/replay failures, and
`complete=true` in 2.2 seconds. A July 28-August 5 screen declared nine dates
and found all nine in history but reached its four-minute budget after seven
games were fully replayed (eighth started). It checked 22 roots: zero
certified, 15 dormant-support and seven unstable-modeled-support rejections;
`deadline_reached=true`, `complete=false`, no history gaps, duplicates,
unsupported targets or replay failures. These are local aggregate diagnostics
under ignored `target/diagnostics/`; no target/path words are published here.
The partial screen is not a nine-day rate or a quality comparison, and provides
no case for production routing or a flat-three claim. The seven-profile,
prospective and native UI release gates remain open.

### September 27 distinct prospective-window workflow

`freeze-prospective` and `evaluate-prospective` are separate from the
historical `freeze-candidate` / `evaluate-sealed` path and leave the declared
August 28-September 26 seal unchanged. The new freeze reuses the development
winner guard: complete coverage, zero failures, and a paired 95% upper bound
strictly below zero. It creates an immutable, versioned record containing the
validated inner freeze identity, UTC freeze timestamp, and the next 30-day
window. Freeze schema v2 also requires exact daily history after the
development cutoff through the UTC freeze date and binds a digest of those
chronological training rows. At freeze time, the history must not already
contain a target in the future window. The UTC day after the final window date
must begin before evaluation.

Evaluation checks the source/executable, pre-window history, and canonical development plan,
reconstructs the frozen configuration, verifies exactly one history row per
window date, and digests those rows before acquiring markers. A global
one-window reservation prevents overlapping or repeated future windows; a
date-named `create_new` marker records the selected window and digest without
target words. The source and window data are checked again immediately before
report publication. Any failure after acquisition leaves the window reserved;
the private report defaults to ignored `target/diagnostics/`, and output
aliases of the registry or marker, including their directory, reject before
acquisition. The report and
completed marker are atomically replaced. The
synthetic identity, date-boundary, marker-race, and mutation regressions pass
on Rust 1.97; the pre-window digest rejects changed, missing, or duplicate
same-day training rows. No protected answers were used in tests, no real window was
frozen or evaluated, and no prospective score exists. The current v19b
comparison's paired upper interval remains above zero, so eligibility—not
the mechanism—is the next blocker.

### September 27 timed gate probe and integration checkpoint

The opt-in `MAYBE_WORDLE_EVIDENCE_TIMING=1` release build, before the
prospective-history identity fix, repeated the seven
profiles on the same nine allowed July 28-August 5 development dates with 16
Rayon workers, a 180-second cap, and private outputs under
`target/diagnostics/` and `target/evidence-checkpoints/`. All 63 profile-games
completed in 73.2 seconds; no seal or prospective target was evaluated. About
71.7 seconds were game simulation, including roughly 49 seconds in the two
staged profiles. Setup, reporting, validation, and checkpoint I/O were small.
The earlier 64.57-second repeat and this timed repeat vary, but both project
well above the 20-minute full-matrix ceiling; neither is a measured full run
or score improvement. A nine-game `3.0000` staged mean is not a flat-three
release result.

After the prospective history-identity correction, Rust 1.97 formatting,
warning-denied all-target Clippy, and all-target tests passed: 349 library,
34 CLI, six integration, and 14 predictive-characterization tests, plus the
benchmark harnesses. The historical Optuna archive, public redaction, and
both generated-evidence verifiers passed after the prose alignment. The worktree remains
uncommitted, and no real prospective freeze/consumption or production promotion
has occurred. Native GUI interaction, a qualifying candidate, the full
seven-profile gate, and hosted Linux CI remain open.

Final local handoff for this checkpoint: `master` is still `23a73e0`, with
13 modified tracked files, no staged files, and 95 untracked files (94
private predictive JSON reports and the local `AGENTS.md`). The final locked
Rust 1.97 release build produced `dist/maybe-wordle.exe` (GUI subsystem 2,
SHA-256 `2B8E238C6833F08D4EE68613A7927CD07C7C4D3C24B4862091C03B8761A8655A`)
and `dist/maybe-wordle-cli.exe` (console subsystem 3, SHA-256
`4F9CEF8D39083EBC5E146084BF5154064BB44F3270768D1924321CFD90F1EF7A`).
The CLI help launched; no native GUI acceptance was claimed. Profile-scoped
Cargo cleanup removed 6,981 dev files (6.9 GiB) and 2,115 release files
(853.4 MiB), retaining `dist/`, private diagnostics, and checkpoints. The
public documentation verifiers, historical archive check, local link review,
and diff whitespace check passed. No staging, commit, push, candidate
promotion, or protected-window evaluation occurred.

### September 27 final diff and test-isolation review

Independent read-only reviews covered the tracked source, CI, test, and
documentation diff. They found no confirmed remaining source regression. One
prospective marker-race test reused a fixed scratch pathname: a stale marker
reproduced a failure, then a process/timestamp-unique root made the same test
pass. The test fixture was removed. The historical consumed sealed report's
missing separate sealed-window source digest/recheck is now distinguished from
the stronger prospective workflow in [`PERSISTENCE.md`](PERSISTENCE.md); the
consumed test was not rerun. Documentation wording now distinguishes
future-window recording from evaluation-time reservation, training history
from forbidden validation targets, and measured from universal work savings.

Rust 1.97 formatting, warnings-denied all-target Clippy, and all-target tests
pass (349 library, 34 CLI, six integration, 14 predictive-characterization,
plus benchmark harnesses). Both public evidence-document verifiers, public
redaction, the legacy Optuna archive check, and `git diff --check` pass after
the documentation corrections. The rebuilt ignored Windows distribution has
GUI/console subsystem 2/3 and SHA-256
`8EADC4FC363A35A90A44E2CCEBE25DB541B7F99998461AC5EBC762B1841A7768` /
`BA03DBEF5F3A930FEE84E385200E3376E9FE054F683FEDAA6468BAD9933B7D23`;
CLI help starts. Checked profile-scoped cleanup removed 5,727 debug files
(5.7 GiB) and 2,115 release files (853.4 MiB), preserving `dist/` and private
evidence/checkpoints. `master` remains at `23a73e0` with 13 modified tracked
files, no staged files, and 95 untracked items: 94 word-bearing raw reports
plus local `AGENTS.md`. An eventual release commit must exclude those local
items. No selected-policy promotion, full seven-profile gate, real prospective
freeze/evaluation, native compact/accessibility GUI acceptance, hosted Linux
CI, commit, or push is claimed.

### September 27 matched per-turn search probe

The current-source release CLI repeated all seven profiles over the permitted
July 28-August 5 development slice with 16 Rayon workers and identical
180-second/4,096-MiB budgets. Timing-off and opt-in per-turn-timing runs
completed 63/63 profile-games in 80.62 and 85.42 seconds respectively. Their
input, matrix and config identities, all seven backtest summaries and every
per-game payload matched exactly. The selected staged profile took about
29.2 seconds in the timed run. Its 27 suggestion calls summed to 90.99 seconds
across parallel games; three second-turn lookahead calls at 71, 71 and 103
survivors accounted for 71.70 seconds of that summed call time. The slowest
single call took 28.62 seconds. These overlapping call durations cannot be
added into profile wall time; the pair also does not isolate logging overhead.
The nine-game staged mean of 3.0000 is not a release mean or new candidate
result. The timing-off slice projects about 54 minutes for the full 2,520-game
matrix, still above the user's roughly 20-minute ceiling. Per-turn logs contain
turn, survivor count, regime and duration but no target or guess words; raw
reports remain ignored and private under `target/diagnostics/`. The next
performance question is which lookahead child computation dominates; no
kernel change or full-matrix run is justified by this coarse probe alone.

A subsequent one-profile selected-staged diagnostic, using a different
single-profile matrix and newer timing-only source, took 28.31 seconds. Its
config fingerprint, backtest summary and all nine game payloads matched the
selected row of the seven-profile report; the overall matrix/input identity
necessarily differs. The 32 emitted search-stage lines summed to 69.76
seconds in 176 lookahead roots, of which 69.37 seconds were inside 4,678
small-child exact-search calls. Larger-child metric scans totaled 84 ms;
coverage rounded to 0 ms at the log's millisecond resolution. These are
overlapping call times, not additive wall time, but they identify recursive
exact child search as the dominant measured bottleneck. A recursive-bound
trial is pending path-equivalence and speed checks, with no production
promotion or full-matrix claim.

The first recursive-bound trial reused the existing admissible per-root
bucket bound and `1e-10` margin after a finite incumbent. The same nine-date
selected-profile run fell from 28.31 to 15.68 seconds, with identical
configuration, backtest summary and all nine game payloads. Summed lookahead
and exact-child times fell from 69.76/69.37 to 39.45/39.00 seconds. A
timing-off seven-profile repeat completed in 46.52 seconds versus the
80.62-second pre-bound pair, with unchanged matrix/config fingerprints, all
seven summaries and all 63 game payloads. This is one local before/after
slice, not a new solver score or proof of population latency. Its linear
full-matrix projection is about 31 minutes, still above the roughly 20-minute
ceiling; the full 2,520-game gate was not launched.

### September 27-28 exact-search speedup and complete development gate

The later recursive exact-search candidate ordering tries the highest-positive-
mass answer first (stable lowest-index tie), while retaining all candidates.
On the same nine allowed July 28-August 5 dates, the selected profile fell
from 15.68 to 5.69 seconds after ordering. Seven profiles took 17.98 and
17.96 seconds on two separate repeats, versus 80.62 seconds before the bound
and ordering changes. Matrix/config fingerprints, all seven summaries and all
63 game payloads matched. This demonstrates speed without changed decisions
on the matched slice, not an improved mean score.

With that measured runtime below the user's roughly 20-minute full-run ceiling,
the current-schema development gate ran to completion using the final source
and retained release CLI: 12 allowed folds, seven distinct profiles, and
2,520/2,520 profile-games. The runner reported 834.39 seconds generation
compute and 834.57 seconds wall time (13.91 minutes); peak process working
set was 203,927,552 bytes (194.5 MiB) under the 4,096-MiB cap. Every profile
had 360/360 scheduled and modeled games, 360/360 solves, zero failures and
zero coverage gaps. The all-game means were previous-release 3.3306, uniform
entropy 3.5694, cooldown entropy 3.4889, weighted proxy-only 3.2444,
proxy-plus-exact-endgame 3.2000, staged without artifacts 3.1944, and selected
staged with disk artifacts 3.1944. The latter two had identical 360 game paths
on this run but remain separate artifact-mode profiles. Selected p95 suggestion
latency was 30.57 ms, and its initial log loss/Brier were 6.6703/0.9987.
The report's `sealed_test_evaluated` field is false; the consumed June
validation targets and reserved August-September seal were not evaluated.
The private source and checkpoint are under `target/diagnostics/` and
`target/evidence-checkpoints/`. This is retrospective development evidence,
not a new prospective result or a demonstrated score improvement over the
selected policy. It does not qualify v19b or any other candidate for a freeze.

Rust 1.97 formatting, warning-denied all-target Clippy, and all-target tests
passed after the final search edit (351 library, 34 CLI, six integration and
14 predictive-characterization tests, plus benchmark harnesses). The Windows
distribution was rebuilt from that source; its GUI and CLI SHA-256 digests are
`B20B0468D8F0AA023721BC1104753A384FD438853D5B30E2E6B3577C543201A9`
and `EF6639CFF025848E8F6C5B3296012B6444440E1E2B82C2F6110765002EF34BEA`.
Independent review found the recursive bound admissible and the answer-first
ordering traversal-only; no blocking correctness or word-privacy issue was
found. It noted that the pre-existing pooled prefix-bound scan checks
cancellation only after its candidate-by-survivor pass, a nonblocking delay
under the current production limits. The new
[redacted public artifact](evidence/september-seven-profile-current-full-public-v1.json)
and [source-backed table](generated/september-seven-profile-current-full-v1.md)
pass their local redaction and documentation verifiers. Final hygiene and
native compact/accessibility GUI acceptance are
separate checks; this paragraph does not claim they passed. No commit or push
was made.

Final local handoff for this checkpoint: on `master` at `23a73e0`, the
worktree has 15 modified tracked files, 97 untracked entries and nothing
staged. The new public JSON (about 1.65 MB) and generated Markdown are the
only new publication inputs; the word-bearing report and checkpoint remain
ignored under `target/`, alongside older local raw diagnostics. Before
cleanup, Rust 1.97 full code gates passed on the final source. After the docs
update, redaction checked all three public artifacts, both CI-equivalent
source-backed document verifiers passed, all 112 relative links in the five
edited Markdown documents resolved, and `git diff --check` passed. The
rebuilt CLI still launches, and the GUI/CLI binaries retain subsystem 2/3
and the hashes above. Verified profile-scoped Cargo cleanup removed 6,817
debug files (6.7 GiB) and 2,115 release files (853.6 MiB), preserving
`dist/`, private evidence and checkpoints. Hosted CI, native compact/enlarged-
text and accessibility GUI acceptance, a qualifying six-turn candidate,
and real prospective confirmation remain open. No files were staged,
committed or pushed.

### September 28 same-state turn-four development diagnostic

The retained CLI ran `same-state-dynamic-regret` on each July 28-August 26
development date at turn four, with a five-second per-date shared budget and
private summary-only output under `target/diagnostics/`. Eighteen paths had
already solved before that turn; four initially exceeded the short budget,
then all four reported already solved when retried at 30 seconds. The final
classification is therefore 22 solved-before-turn-four, six exact-reference
states with one to six combined active/dormant survivors, and two unresolved
states above the six-answer reference limit (supports seven and eleven).
No consumed June validation targets or reserved August-September seal were
used. Each report binds the current source/config/data identity.

Only one exact state yielded different choices: August 24, support six and
three turns remaining. Both choices had zero exact-reference modeled failure
regret; staged had +0.1114125 expected-attempt regret and dynamic finite had
zero. The other five exact-state choices agreed and had zero regret. This is
one local decision-quality signal, not an observed all-game improvement, a
general turn-four routing class, a runtime pass, or a reason to change the
selected policy. The eight not-yet-solved states are too few, with two lacking
exact reference, to infer a score gain. The staged-root-seed hypothesis was
tested separately below; it did not qualify for a policy rollout.

The same cohort's two surviving turn-five paths were also checked at the
30-second cap. Six of the eight turn-four survivors had already solved by
turn five. August 24 had three active answers and different staged/dynamic
actions, but both matched the exact two-turn optimum; August 26 had four
active answers and the actions agreed at zero regret. This tiny conditional
sample does not validate the terminal rule generally, but it supplies no
new reason to replace it. The unresolved objective gap is earlier in play.

### September 28 staged-root-seed counterfactual

A temporary, opt-in finite-search diagnostic inserted the selected staged
action after the state-local baseline without removing original proposals.
It compared unseeded and seeded dynamic-belief finite choices on the same
staged-replayed state, with 4,000,000 work units and a five-second safety
deadline per finite arm. The shared report cap was 30 seconds. A unit test
checked legal insertion, completed evaluation and original-root retention;
another rejected invalid seeds. Both passed before the screen.

The development screen covered five spaced turn-one dates and all 30
July 28-August 26 dates at turn two. At turn three, 28 dates remained
eligible and two had already solved. At turn four, eight remained eligible
and 22 had already solved. Thus 71 eligible state evaluations completed the
seeded root. Unseeded and seeded finite choices were identical in all 71;
their result quality, stopping reason, and modeled failure/attempt values
also matched to 1e-12. Eighteen states had exact small-state references;
53 exceeded the six-answer reference support limit. The raw summary-only
experimental reports remain ignored under `target/diagnostics/`; no target
words were published. The diagnostic-only code was removed after the
negative result, leaving the selected policy unchanged. The retained `dist`
binaries were subsequently rebuilt from the restored source, as recorded
below.

This is a bounded proposal-screen result, not a whole-game score comparison,
proof of global optimality on the larger states, or evidence about a higher
work budget. The staged seed was supplied for free after its separate
calculation, so the screen does not establish a viable full-request latency.
It gives no reason to promote a seeded finite route or spend
another seven-profile run on that route. Earlier-turn six-turn optimization
remains open.

### September 28 rebuilt-executable evidence and local handoff

After removing the negative seed trial, Rust 1.97 formatting, warning-denied
all-target Clippy, and all-target tests passed (351 library, 34 CLI, six
integration, 14 predictive-characterization, plus benchmark harnesses). Both
Windows release binaries were rebuilt from that source and copied to the
ignored `dist/`: GUI subsystem 2, SHA-256
`CF680A4F478A0CCFB2A435A853A788A7E613A10499D32897597C63E31880A691`;
CLI subsystem 3, SHA-256
`50BD7AB8B54312F476C428084D667606679310C62F95EA2E5E61E548F4DEDA97`.
Both copied hashes matched their release outputs and the CLI help launched.
`dist/BUILD-INFO.txt` records this build. Native compact/enlarged-text and
accessibility GUI acceptance is still open; no visual acceptance is inferred
from the PE subsystem check.

Because executable bytes changed, a new private seven-profile development
run was made with the retained CLI and a separate checkpoint. It completed
all 2,520/2,520 profile-games over the same 12 allowed folds in about 15.12
minutes (907.38 seconds generation compute), within the 1,200-second and
4,096-MiB caps. Peak process working set was 203,755,520 bytes (194.3 MiB).
All seven profiles solved 360/360 with zero failures and coverage gaps;
selected staged remained 3.1944 all-game mean and 29.72 ms shared-process
suggestion p95. The old and new reports agreed on all 2,520 targets,
outcomes, guess paths, prior strata and posterior-calibration rows. Config and
matrix fingerprints and selected ranges agreed; the input fingerprint changed
because it includes the executable. The preceding private report was copied
to a separate archive before the new report became the canonical redaction
source. The new public redaction and generated README fragment were refreshed;
the three-artifact redaction check, both source-backed document checks and
112 relative Markdown links passed. The consumed validation and declared
seal remain unused as new targets; no candidate was frozen.

The Cargo profile clean dry runs scoped 6,497 dev files (6.5 GiB) and 2,115
release files (853.6 MiB) to rebuildable outputs. No reparse points were
found under those profile directories. The corresponding clean commands
removed only those files while preserving `dist/`, the private reports and
checkpoints under `target/`, and the redacted public evidence. This is the
current local checkpoint, not a release or permission to commit or push.
The selected staged policy still lacks an early-turn six-turn objective
qualification; native compact/accessibility GUI acceptance and genuinely
prospective confirmation remain open. Hosted CI for this uncommitted work is
also unavailable until a user-authorized push.

### September 28 turn-three same-state development screen

The retained CLI (SHA-256
`50BD7AB8B54312F476C428084D667606679310C62F95EA2E5E61E548F4DEDA97`)
ran the opt-in dynamic-regret diagnostic at turn three on every July
28-August 26 development date, with a 15-second per-date shared cap. The 28
summary-only private reports have one input/config/source identity and a
combined 67.38 seconds of measured generation time. Two dates were already
solved before turn three; the CLI returned exit 1 with
`selected turn occurs after the staged path was solved` and produced no report.
These are not solver failures or zero-regret samples.

Twelve states had exact full-legal-root references at one to five combined
active/dormant answers. Staged and `finite_fast_dynamic` chose the same action
in all twelve, with zero modeled failure and expected-attempt regret for each.
Sixteen states exceeded the six-answer exact-reference limit. Four of those
had different actions, all without an exact regret value; the disagreements
include dormant fallback support or a larger active set. This complete
normal-mode turn-three screen narrows the unresolved objective gap to larger
states, but does not show a score gain, qualify a router, validate recursive
hard-mode behavior, or authorize production promotion. No consumed validation
or reserved seal targets were evaluated. The raw reports remain under ignored
`target/diagnostics/`.

### September 28 dynamic exact-shortcut correction and current build

An independent source audit found that the finite kernel's direct two-active-
answer `Exact` shortcut could claim zero modeled failure after a first miss
even when that miss would activate dormant fallback answers. A threshold-one
dynamic-oracle regression failed before the correction: the shortcut returned
zero failure risk where the oracle returned 0.0586543. The shortcut now runs
for a single legal active answer, or for two legal active answers only when
the dynamic belief has no dormant fallback survivors. The regression passes;
Rust 1.97 formatting, warning-denied all-target Clippy and all-target tests
also pass (352 library, 34 CLI, six integration and 14 predictive-
characterization tests, plus benchmark harnesses). This repairs the modeled
finite continuation in low-threshold configurations. It does not change the
selected staged policy. Under the selected threshold-four configuration,
ordinary and recursive feedback transitions activate matching dormant words
before a two-active-plus-dormant node is reached, so the prior same-state
references known to use that config are not semantically invalidated. Other
historical reports retain their recorded executable identity; no blanket
fresh-source claim is made for them.

The rebuilt Windows GUI/CLI binaries were copied to retained `dist/` from the
release outputs. Their SHA-256 digests are respectively
`D83A91D9D1F76F8594473837DB4F2C8DCD8BBD87271C3C8F817D67E5D61E3675`
and `D2F3DF1B9B5D53FC6B49BA017CE5C4AEE985584DE34237C6B7FF407C928D39CD`;
both copies match their release outputs. The PE subsystems are GUI 2 and
console 3, and CLI help launched. Per the user's request, the GUI was left
closed, so narrow/enlarged-text and accessibility acceptance remain open.

A fresh current-CLI matched July 28-August 26 dynamic-finite diagnostic
solved 30/30 with zero coverage gaps for each profile: selected staged
3.3333, opt-in `finite_fast_dynamic` 3.3667, finite-minus-staged paired
interval [-0.1333,+0.1667] around +0.0333. Their shared-process p95
suggestion latencies were 22.68 and 265.31 ms. Compared with the earlier
pre-fix executable's final matched report, staged kept all 30 paths; finite
kept only one path while six outcomes improved, five worsened and 19 retained
the same guess count. Both finite reports had the same 3.3667 point mean.
The bounded root deadline and changed executable preclude attributing those
individual path differences to the shortcut fix; there is no demonstrated
score or latency win and no promotion.

The current-binary seven-profile matrix completed all 2,520 profile-games
over the 12 permitted development folds in 665.16 seconds (about 11.1
minutes), under the 1,200-second and 4,096-MiB caps, with a 191.5-MiB peak
process working set. Every profile solved 360/360 with zero failures and
coverage gaps. Selected staged remained 3.1944 with 21.83-ms shared-process
suggestion p95. Relative to the preceding 15.12-minute executable, all
2,520 inputs, game paths, outcomes, prior strata and posterior-calibration
rows matched; config/matrix identities and selected ranges also matched.
The input fingerprint changed with the executable. None of the seven
profiles uses dynamic finite search, so neither the unchanged score nor
the shorter elapsed time demonstrates benefit from this correctness fix.
The previous private source report was archived separately under `target/`
before the current report became the canonical redaction input. The public
copy and README table were regenerated from it; all three redactions and
both CI-equivalent source-backed documentation checks pass. The consumed
validation targets and reserved seal remain untouched as new targets.
Native GUI acceptance, a qualifying early-turn six-turn candidate, genuinely
prospective confirmation and hosted CI remain open. No commit or push was made.
