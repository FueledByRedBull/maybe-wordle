# Plan: Remaining Formal Solver Work After the Current Phase 1/Partial Phase 2-6 Landing

## Summary

The repo already has the Phase 1 core plus part of the later roadmap:
- progress reporting and diagnostics
- predictive Gap 2 split with `exact_exhaustive_threshold`
- canonical predictive exact memo keys
- direct bucket construction and quick-plan bucket reuse
- root-level iterative-deepening entry
- entropy-based expected-cost lower-bound pruning
- cached `StateKey` count/hash
- initial artifact/version metadata
- a proof-certificate sidecar metadata file and a fast cache-only verification scan
- meaningful Criterion benches

What still remains is the deeper architectural work that was planned but is not yet implemented in the code.

## Remaining Changes

### 1. Complete the formal search redesign
- Replace the current residual two-pass structure in [src/formal.rs](C:/Users/ancha/Documents/Projects/Maybe%20Wordle/src/formal.rs) with one unified evaluator that returns the full lexicographic optimum in a single recursive traversal.
- Stop using `solve_depth` only to discover a depth and then `solve_expected` to rewalk the same state space.
- Carry the incumbent full objective `(worst_case_depth, expected_guesses)` during recursion so depth pruning and expected-cost pruning cooperate in one pass.
- Keep the current iterative-deepening entry, but drive it through the unified evaluator rather than the old two-function split.
- Preserve the current default objective and tie-breaking semantics.

### 2. Add real small-state math tables
- Implement `min_expected[n]` as a real offline retrograde table, not the current analytical entropy floor.
- Build all states up to a benchmark-tuned size `K` bottom-up from singleton states.
- Reuse the same exact small-state table in both:
  - formal search base cases
  - predictive exhaustive tiny-subset search
- Version the table and include its version in the manifest hash inputs.
- Keep the current entropy-based lower bound as a secondary admissible floor, not the primary one.

### 3. Add sound refinement pruning
- Add root-level refinement pruning over the initial guess set using subset relations between corresponding child states.
- Add state-local refinement pruning only when the subset-check cost is below a configured threshold.
- Keep pruning provably sound; do not introduce entropy/max-bucket heuristic dominance rules.
- Record pruning statistics in the existing progress/diagnostic output.

### 4. Finish the state and memo architecture
- Replace the current cached-`Vec<u64>` `StateKey` with the planned hybrid representation:
  - compact sorted-index form for small states
  - fixed-capacity inline bitset form for larger states
- Replace the current hash-by-recompute finalize path with deterministic Zobrist hashing.
- Add answer-local dense reindexing for small/medium states so partitioning can scan contiguous local answer IDs.
- Introduce the bounded two-level transposition table with depth/objective-preferred replacement for the hot formal search path.
- Keep a separate exact persisted memo for correctness-critical stored results where needed.
- Define and enforce explicit memory-budget targets for the bounded tables and small-state cache.

### 5. Add the parallel and low-level acceleration phase
- Implement adaptive Rayon subtree dispatch for expensive states, not just root-only parallelism.
- Base dispatch on estimated subtree cost from survivor count, lower bounds, and branch structure.
- Use task-local memos and deterministic merge rules.
- Add scalar/SIMD split for the partitioning hot loop, with the scalar path kept as the correctness oracle.
- Keep SIMD behind one isolated abstraction layer and benchmark it independently from the full build.
- Extend the existing Criterion bench suite in [benches/solver_bench.rs](C:/Users/ancha/Documents/Projects/Maybe%20Wordle/benches/solver_bench.rs) to cover:
  - unified formal evaluation
  - retrograde-table construction
  - refinement pruning
  - scalar vs SIMD partitioning
  - bounded-transposition behavior

### 6. Turn the certificate sidecar into a real proof system
- Replace the current certificate metadata sidecar with actual witness data sufficient for linear-time verification.
- For each stored state, persist the data needed to prove:
  - the chosen guess’s objective matches its child states
  - no rival guess can beat that objective
- Make `verify-optimal-policy` use certificate verification by default.
- Keep the current independent re-search as an explicit oracle/debug verification mode.
- Version the certificate format and fail fast on stale or mismatched side assets.

### 7. Finish artifact hardening and experimental tracks
- Complete manifest hashing/versioning for:
  - objective ID/version
  - `StateKey` format
  - small-state table
  - certificate format
  - any bounded-memo or auxiliary side assets that affect correctness
- Add the experimental expected-cost-only formal objective with distinct model IDs and artifacts.
- Add state-isomorphism canonicalization behind an internal non-default flag only.
- Keep both experimental tracks isolated so they cannot be loaded as the default formal policy.

## Test Plan

- Keep the current solver path as an oracle during the unified-search migration and cross-check old vs new on toy and medium fixtures before removal.
- Validate retrograde tables exhaustively up to `K` and cross-check sampled states against uncached exhaustive search.
- Validate refinement pruning by proving dropped guesses are refined by retained guesses and confirming the optimum is unchanged.
- Validate hybrid `StateKey` correctness by asserting small/large representations compare and hash identically for the same logical state.
- Validate bounded transposition behavior with fixed memory ceilings and confirm replacement policy does not change final results.
- Validate adaptive parallel search by comparing serial vs parallel artifacts byte-for-byte.
- Validate SIMD with bit-for-bit equality against the scalar path.
- Validate full proof certificates by checking certificate verification and oracle re-search agree on generated artifacts.
- Extend the current Criterion suite so every newly added subsystem has a before/after benchmark.

## Assumptions and Defaults

- The already-landed progress reporting, predictive exact split, quick-plan reuse, cached `StateKey` count/hash, and initial artifact metadata remain the base to build on rather than being redesigned again.
- `exact_exhaustive_threshold` stays at `12` unless new benchmarks justify changing it.
- Hybrid `StateKey` switching still targets roughly `30` survivors unless measurement says otherwise.
- Adaptive parallel dispatch still targets states around `50+` survivors as the first tuning point.
- Small-state retrograde tables are bounded by an explicit memory budget and versioned as formal assets.
- Proof certificates are not considered “done” until they can replace default verification without any re-search.
- Experimental expected-cost-only mode and state-isomorphism canonicalization remain non-default even after implementation.
- Checkpoint/resume remains out of scope unless new artifact work makes it cheap enough to add without distorting the main solver changes.
