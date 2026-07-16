# Maybe Wordle review backlog

This file captures the read-only engineering review completed on 2026-07-10 and the implementation pass completed on 2026-07-17. Priorities reflect impact on correctness, reproducibility, and product claims. Completed boxes are retained as a historical engineering record.

## 2026-07-17 implementation record

- Formal verification now starts from an empty independent cache, validates complete persisted-state coverage, recomputes certificate partitions/children/masses, independently resolves every decision, and uses certificate format v4. Unsafe refinement pruning was removed; deterministic randomized 13–40-answer universe tests compare builder decisions with the exhaustive reference solver.
- History sync retries all gaps, honors `429 Retry-After`, uses bounded exponential backoff, and preserves the last contiguous archive when a repair remains partial. History/model/pattern/predictive/formal writes are crash-safe sibling-temp atomic replacements.
- Model loading rejects non-contiguous history unless `allow_history_gaps = true`; date weights no longer admit future history-only words.
- Predictive artifact identity now hashes ordered guesses, answer metadata, full history, effective weights, policy, config, variant, and date. Mixed positive/zero support remains filterable and all recovery modes are regression-tested.
- Tuning now uses non-overlapping train/validation/untouched-test windows. Backtests report coverage and 95% confidence intervals; experiment surfaces retain log-loss and Brier diagnostics. Predictive probabilities remain explicitly labeled heuristic.
- CLI/GUI output now exposes model/manifest/history/artifact/cache-promotion status and labels predictive deep scores as candidate-pool exact. Artifact identity was split from book search, and atomic persistence was centralized.
- Additional logic/performance weakness found during implementation: recomputing artifact identity from only list counts was both stale-prone and made cache lookup semantics opaque. The full manifest fixes correctness; its canonical length-delimited hash avoids large serialization allocations while retaining deterministic invalidation.

Post-fix untouched-period baseline (kept separate from pre-fix results): release build, weighted `seed_plus_history`, 2026-03-28 through 2026-04-26, predictive policy `predictive-v1`, model manifest `bab9c2f13ee4dd32`, history snapshot 2026-04-26 / hash `8e97e7d493ecc539`. Results: 30 games, 27 covered, mean guesses 3.2222 (95% CI 3.0624–3.3820), p95 4, max 4, 3 failures (Wilson 95% failure-rate CI 0.034599–0.256214), mean log loss 7.327027, mean Brier score 0.999241, mean target probability 0.000751, and mean target rank 360.44. This baseline is diagnostic evidence for a heuristic prior, not a calibration claim.

## P0 - Formal-mode claims and verification

- [x] Make formal verification genuinely independent before describing formal mode as an "exact policy builder" or its output as a proof.

  Evidence:

  - `FormalPolicyRuntime::solve_state_independent` supplies the persisted policy as a seed cache (`src/formal.rs:1033-1035`).
  - `IndependentExactSolver::solve` returns a seed-cache entry before doing any independent work (`src/formal.rs:498-503`).
  - The oracle verifier invokes that path for states which are already in the persisted policy (`src/formal.rs:677-688`), making the comparison tautological.
  - The certificate verifier checks only states included in the certificate; it does not require coverage of all persisted/reachable policy states (`src/formal.rs:2032-2040`). It also does not recompute the listed guess partitions and child masses from the pattern table.

  Acceptance criteria:

  - An oracle verifier runs with an empty policy cache and is exercised on tractable models.
  - The certificate verifier recomputes every claimed child state, feedback pattern, and probability mass from the model.
  - The certificate proves coverage of every required reachable state and records a witness for pruned candidates/bounds.
  - Tests prove that deleting a certificate state, changing a child state, changing a child mass, or changing a policy decision all fail verification.

- [x] Audit or remove refinement pruning until its dominance direction is proved.

  `plan_refined_by` (`src/formal.rs:2510-2517`) discards a candidate when every candidate bucket is a subset of a retained bucket. That appears to discard the more informative partition, which is the opposite of ordinary information dominance. The optimization may be rare because it only runs under plan-count thresholds, but it weakens the exactness claim until demonstrated correct.

  Acceptance criteria:

  - Add randomized 13-40-answer toy universes that compare the policy builder against a no-pruning exhaustive solver.
  - Document the dominance argument, including all-green branches and the lexicographic objective.

## P1 - Data lineage and model correctness

- [x] Prevent permanent holes in NYT history.

  Sync records a failed date and continues (`src/data.rs:209-249`), then persists the partial archive. A later sync starts from the last existing date minus only the reverify window (`src/data.rs:182-189`), so a missing middle date followed by later successes is never retried.

  Acceptance criteria:

  - Derive and retry every missing date between launch and the requested end date, or persist a durable pending-gap ledger.
  - Reject model-building, artifact generation, and backtesting when history is non-contiguous unless an explicit override is supplied.
  - Add a test: first sync fails on a middle date, next sync succeeds, and the archive becomes contiguous.

- [x] Retry rate limiting and make sync writes crash-safe.

  Retry logic accepts server errors but not HTTP 429 (`src/data.rs:290-300`). History and several generated artifacts are written directly with `File::create`/`fs::write` (`src/data.rs:150-165`, `src/pattern_table.rs:119-141`, `src/solver/books.rs:468-475`), leaving files vulnerable to interruption or concurrent generation.

  Acceptance criteria:

  - Respect `Retry-After` and retry 429 responses with bounded exponential backoff.
  - Write to a sibling temporary file, flush/sync it, then atomically rename it into place.
  - Validate archive continuity and JSON readability before replacing an existing history file.

- [x] Remove future-history leakage from date-based modeling and backtests.

  The model adds all historical entries (`src/model.rs:140-149`), but the date-aware weight calculation uses the full history to decide eligibility and history-only base weight (`src/model.rs:224-240`). A word first seen after `as_of` can therefore be admitted with positive mass during an earlier backtest.

  Acceptance criteria:

  - Construct the effective answer universe from information available on or before `as_of`, or explicitly call evaluations retrospective and exclude them from predictive claims.
  - Add a date-boundary test with a history-only word whose first appearance is after `as_of`; it must not be eligible at that date.
  - Re-run all backtests and record the post-fix baseline separately from previous results.

- [x] Treat the predictive prior as a heuristic until it is calibrated with leakage-free rolling evaluation.

  The cooldown curve, seed/history base weights, and proxy score are sensible declared assumptions, but they are not yet evidence of calibrated probabilities. Tuning and evaluation should be separated in time.

  Acceptance criteria:

  - Use rolling-origin train/validation/test windows.
  - Report coverage, log loss, Brier score, mean guesses, failure rate, and confidence intervals on an untouched final period.
  - Version the input history snapshot and publish the exact model/config manifest with every result.

## P1 - Predictive artifacts and recovery behavior

- [x] Fix predictive artifact identity so stale advice cannot be accepted.

  The identity payload fingerprints policy/config plus only `guesses.len()` and `answers.len()` (`src/solver/books.rs:4-24`). Artifact loading trusts that identity (`src/solver/books.rs:51-65`, `src/solver/books.rs:84-94`). Replacing a word while preserving list length, changing historical data, or changing answer metadata can silently reuse an old opener/reply book.

  Acceptance criteria:

  - Create a versioned model manifest hash from ordered guesses, ordered answer records, history, effective weights, and policy/config.
  - Include the manifest hash in artifact names and serialized identity.
  - Add regression tests for same-count guess/answer substitutions and history mutations.

- [x] Repair the recovery-mode contract and restore the failing test gate.

  The working tree currently changes `cooldown_floor` from `0.00` to `0.01` (`config/prior.toml:4`, `src/config.rs:97`). The existing zero-mass fixture is therefore positive, and `recovery_modes_are_explicit_in_predictive_api` fails at `tests/predictive_characterization.rs:252`.

  Independently, `initial_state` omits zero-weight candidates whenever any candidate has positive mass (`src/solver/state.rs:15-39`). A later feedback branch cannot recover those omitted candidates even though recovery mode is intended to repair a zero-mass state.

  Acceptance criteria:

  - Decide whether the nonzero floor intentionally eliminates normal recovery use; update the fixture to create genuine zero mass rather than weakening the assertion.
  - Keep every supported candidate through feedback filtering, with modeled and effective weights represented separately.
  - Add mixed-support tests: positive candidates initially exist, feedback isolates only zero-mass candidates, and each recovery mode behaves as documented.
  - Restore a green `cargo test` run.

## P2 - Maintainability and product clarity

- [x] Keep the distinction between heuristic predictive mode and formal mode unmistakable in the UI, CLI, and README.

  Predictive search uses bounded pools, proxy ranking, and selective lookahead; it is high-quality heuristic search, not a global optimum. The formal mode is a fixed, pinned model, not a prediction of NYT editorial choices.

  Acceptance criteria:

  - Show the model version, history snapshot date/hash, artifact status, and whether a suggestion was promoted from a cache.
  - Label pooled "exact" scores as candidate-pool exact, or equivalent wording.
  - Do not market a certificate as proof until P0 is complete.

- [x] Split the largest algorithm files along stable boundaries.

  `src/formal.rs`, `src/solver.rs`, and `src/solver/eval.rs` contain large amounts of coupled policy, persistence, evaluation, and test logic. The recent internal-module split is a good start; continue it around formal verification, artifact identity, backtesting, and search primitives.

  Acceptance criteria:

  - Each module has one dominant responsibility.
  - Search/partition logic can be tested with tiny in-memory models without filesystem artifacts.
  - Persistence and verification code are independently testable.

## Regression-test checklist

- [x] Full history sync with a transient middle-date failure followed by successful repair.
- [x] HTTP 429 retry and `Retry-After` handling.
- [x] Atomic-write interruption leaves the previous valid artifact readable.
- [x] Same-count word-list/history mutation invalidates predictive artifacts.
- [x] Future history does not change an earlier `as_of` state.
- [x] Mixed positive/zero-mass recovery branch.
- [x] Formal certificate rejects missing state coverage, wrong pattern, wrong child state, wrong mass, and altered policy decision.
- [x] Random small-universe formal policy agrees with an uncached exhaustive reference solver.

## Original review validation (2026-07-10)

- `cargo clippy --all-targets -- -D warnings` passed.
- `cargo test` had one failure and 105 passing tests: `recovery_modes_are_explicit_in_predictive_api` in `tests/predictive_characterization.rs`.

## Implementation validation (2026-07-17)

- `cargo test` passed: 115 tests across library, binary, integration, and predictive characterization targets; no failures.
- `cargo clippy --all-targets -- -D warnings` passed.
- `cargo build --release` passed; the verified Windows package is `dist/maybe-wordle.exe` (11,507,200 bytes, SHA-256 `9221B2B2EEBECE7BE6E14C90EC14D0E0C053104235CBE11B199A2DF5240CCF1E`).
- `cargo clean` removed 7,206 rebuildable files (3.9 GiB) after packaging; the runnable executable was preserved outside `target/`.

## Existing strengths to preserve

- Duplicate-letter feedback scoring is correct and explicitly tested.
- Predictive and formal objectives are intentionally separated.
- The pattern table, compact state structures, Rayon parallelism, and small-state exact fallback are appropriate for the workload.
- The CLI/GUI expose useful artifact and recovery state instead of silently hiding model limitations.
- The project's central idea - declaring the answer universe and prior rather than assuming a frozen uniform Wordle list - is excellent.
