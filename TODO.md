# Maybe Wordle remaining roadmap

The predictive solver is the primary product; formal mode is a secondary research tool. Completed work and current evidence are documented in `README.md` and `PLAN.md`.

The current release did not reach the aspirational flat-three target. Its guarded 12-fold development mean is `3.1778`, and its once-only 30-game sealed mean is `3.3000` with full coverage and no failures. The sealed window must not be reused for further tuning.

## Predictive research after this release

### [ ] Establish a new future holdout before further optimization

- Extend the history and declare a new chronological development/sealed boundary before making another promotion claim.
- Never tune against the 2026-06-18 through 2026-07-17 sealed outcomes now recorded in `benchmarks/predictive/sealed-selected-v20-20260726.json`.

### [ ] Run the remaining high-cost cohort research

- Run each typed proxy, search, and book cohort with enough candidates for its deterministic one-factor sweep, then pass only Pareto finalists to a small multi-seed joint refinement.
- Fit leakage-safe out-of-fold continuation costs with a regularized baseline before deciding whether a learned ranker is better than the current explicit proxy.
- Compare empirical-frequency, recency-bucket, survival/hazard, and regularized prior models under the same coverage-first rolling objective.
- Promote only configurations that retain full coverage and no failures while improving paired all-game score within declared latency and memory budgets.

### [ ] Improve long-run study ergonomics

- Add resumable per-profile checkpoints so a cancelled evidence matrix can retain verified completed profiles.
- Consider batched constant-liar model-based suggestions only if a measured wall-clock win justifies weaker within-batch feedback; sequential ask/tell remains the reproducible reference.

### [ ] Validate platform telemetry

- Exercise the macOS resident-set sampler on native macOS hardware before claiming the hard memory cap is cross-platform validated. Windows is the release-tested platform.

### [ ] Retire the archived optimizer source

The Rust study runner is the only promotable optimizer. The deterministic archive at `benchmarks/predictive/legacy-optuna-archive.json` preserves the 33 completed legacy trials and identifies six unfinished trials.

- Delete `scripts/optimize_live_config.py` only after explicit file-deletion approval. Its scalar objective is historical and its candidates remain non-promotable without current guarded evaluation.

## Formal research boundary

Certificate v7 is independently verified on tractable universes with explicit witnesses and mutation tests. The scale-v2 benchmark projects the complete 2,358-answer proof far beyond practical resource limits.

- Attempt a larger formal universe only if a new algorithm or declared budget materially changes those projections.
