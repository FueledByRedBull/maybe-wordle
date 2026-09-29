# September Code Audit Implementation Plan

> Workers receive bounded task briefs and exclusive file ownership. The parent owns this ledger, integration, review and final validation.

**Goal:** Resolve all 48 findings from the September 28 audit against current source, verify the resulting application, and retain a clean runnable Windows distribution.

**Architecture:** Repair existing shared invariants first: horizon-conditioned formal decisions, transactional history, typed request/result and artifact outcomes, strict persistence and evaluation boundaries. Introduce only module boundaries that give these responsibilities one owner. Preserve selected v20 until matched evidence supports another policy.

**Tech stack:** Rust 2024, Rust 1.97.0, eframe 0.32.3, existing Cargo.lock, PowerShell 7.

**Spec:** externally supplied `maybe-wordle-code-review.md`, SHA-256 `FDCE754B5BCFB36D21815E7B877422AA007A6C188F5783E05829D88041D3C22D`. Its findings and acceptance cases are reference material under the user's implementation request.

## Delivery - September 29

All 48 findings below have implemented, reviewed dispositions. Authorized local
audit work is complete; the remaining external acceptance and separate research
items are explicitly retained, not counted as passing checks.

- Final Rust 1.97.0 format and locked warning-denied Clippy pass. Locked all-target
  tests pass: 536 library, 43 CLI, one GUI-entry, seven integration and 16 predictive
  characterization tests (603 total), plus 15 benchmark smoke workloads.
- Independent reviews covered formal mathematics/publication, evaluation ownership,
  cancellation and final reporting. The final empty-regret finding was reproduced
  RED, fixed and included in those full gates. Concurrent library runs passed
  533+533 before the final formal/reporting additions; those are not relabeled as
  concurrent coverage of the final build.
- First-feedback top 1/5/10, clone allocation and cancellation measurements are in
  PERFORMANCE.md. The full 12-fold selected/v19b comparison solves 360/360 per
  profile at 3.194444/3.191667, with all 720 paths/outcomes unchanged from the prior
  replay. The paired interval crosses zero; production remains v20. The full
  seven-profile attempt stopped at 1,200,021 ms with only six complete profiles;
  its negative evidence/checkpoint is retained, not published as a full success.
- Generated rolling/timing-screen documentation passes both exact CI commands;
  all five public redacted artifacts pass the checker. Markdown local links and
  `git diff --check` pass. Measurement identities predate only the final empty-regret
  reporting correction and remain distinct from the final dist hashes.
- Final release build passes; GUI/CLI copies match source executables, have Windows
  subsystem 2/3 respectively, and CLI help passes. Checksums are in `dist/SHA256SUMS.txt`.
  Verified Cargo profile cleanup removed 7,511 debug files (8.2 GiB) and 2,556 release
  files (1.1 GiB). Dist, raw/runtime data, seeds, source, unrelated changes, private
  evidence, checkpoints and audit verification records are preserved.
- Hosted Windows/Linux/macOS tests, binary builds and CLI smoke pass for `d2024bd`,
  including Unix persistence/symlink/FIFO regressions (run linked below). Native
  GUI visual/keyboard/accessibility/high-DPI and real Windows UNC-share acceptance
  remain unrun. The GUI stayed closed.
  The dependency audit found zero known vulnerabilities with two specifically
  documented maintenance exceptions, not a blanket security guarantee.
- At the local delivery checkpoint, the branch was `master` at `23a73e0` plus
  existing/audit uncommitted changes; publication had not yet been requested.
  The subsequent authorization is recorded below. Future promotion/prospective work
  remains in TODO; no reserved seal was opened and no future evaluation invented.

## Global constraints

- Base: `master` at `23a73e092678a2ca5c396640780f72581cfdae33`, with existing uncommitted source/docs and private evidence. Preserve that work.
- Development cutoff: 2026-08-26. Exclude 2026-06-18 through 2026-07-17 from new targets. Leave the 2026-08-28 through 2026-09-26 seal untouched.
- Consumed outcomes may be chronological training history, never new tuning/validation targets.
- No automatic policy promotion, fabricated fallback results, or historical-score claims for changed code.
- Each worker has exclusive files; outside interfaces go through the parent. Workers do not delegate or commit. Shared integration runs after edits stabilize.
- Keep the GUI closed under the user's preference; headless tests proceed. Native interaction acceptance remains explicit.
- Preserve dist, raw history, seeds and evidence/checkpoints. Delete only verified rebuildable output at delivery. Implementation and subsequent publication were separately authorized.

## Review focus

- Formal children may use spare depth to reduce expectation; persisted/reloaded decisions must preserve that horizon.
- Invalid history edits retain committed state and the visible draft.
- Cancellation, stale replies and worker exit leave truthful GUI state.
- Malformed files/interrupted publication cannot truncate authoritative data or invent values.
- Empty/partial/excluded target populations cannot become successful evidence or consume a seal.

## Task sequence

1. **Formal mathematics:** exclusive `src/formal.rs`, `src/formal/verifier.rs`, `src/small_state.rs` as needed. Reproduce depth 4 / expected cost `235/132` for the nine-answer weighted audit fixture. Implement `E(S,d)` or a full frontier, independent verification, ExpectedOnly without a minimum-depth constraint, explicit objective config and incompatible artifact versioning. Parent wires CLI when the interface is settled.
2. **Atomic persistence:** exclusive `src/atomic_file.rs` initially. Resolve bare/dot relative paths, Unicode/traversal/new destinations and Windows verbatim paths; retain pre/post-replacement guarantees. Add a streaming write entry point for later seed/config/table work. Run path and injected-stage tests.
3. **Ridge mathematics:** exclusive `src/predictive/learned_proxy.rs` initially. Separate intercept centering from variance scaling; test all four fit modes, constant/multiple columns and serialized prediction. Audit fixture must yield slope `4/3`, intercept `7/3` for x=[1,2,3], y=[3,5,7], lambda=1.
4. **Input/history:** `data.rs`, `model.rs`, `seed.rs`, `config.rs`. Strict nonempty/guessable inputs, atomic authoritative writes and overlap protection, bounded exports, complete and cancellable sync with truthful applied/attempted counters. Use local fake HTTP tests, never live seal targets.
5. **GUI/shared state:** shared history transition first; then `gui.rs`, GUI entry point and focused runtime/view modules. Fix draft validity, transactional apply/undo/terminal/EOF, stale result identity, typed status, background loading, cancellation/disconnection, labels, candidate reachability, rank and contrast. Test domain/worker logic and egui geometry without launching the app.
6. **Evaluation:** explicit eligible target sets, exhaustive teacher, complete window-scoped exclusive sealed ownership, cross-field checkpoint validation, cooperative inner budgets and additive optional metrics. Correct semantics before moving offline orchestration/formatting out of `solver/eval.rs`.
7. **Predictive runtime:** broaden dynamic oracle coverage and nonprogress audit; direct final-turn routing; propagate book/holdout errors; typed artifact provenance and actual search quality; share immutable patterns/identity work. Measure first-feedback top 1/5/10 with cancellation and allocation evidence before optimizing.
8. **Research/config:** fallible inference, canonical survival eras/convergence/unique events, finite aggregation and sampling-domain validation; consolidate mechanical parameter metadata and feature names without changing feature order.
9. **Formal integration:** bounded parsers, coherent generations, stored primary decision, shared bounded alternatives, cancellation, truthful scale target/timing and resume checks after task 1 stabilizes.
10. **Cleanup/CI/delivery:** remove confirmed unused generator/aliases/ignored arguments, share feedback/README helpers, isolate test/bench scratch, remove late environment mutation, pin toolchain/actions and add modest platform/advisory checks. Review, verify, regenerate relevant evidence/docs, rebuild dist and clean outputs.

## Item ledger

Open items require implementation, relevant checks and review. Already-correct or replaced findings need an evidence-backed disposition.

| ID | Required result | Status |
| --- | --- | --- |
| P1-01 | Horizon-conditioned formal lexicographic/expected-only objective | Implemented; weighted fixture 235/132, exact-depth frontier/ExpectedOnly tests and independent review pass |
| P1-02 | Dynamic shortcut/nonprogress and all-root oracle coverage | Implemented; threshold 0/1/2/4 all-root, hard-mode, horizon 1-4, dormant-removal and memo tests pass; new dynamic nonprogress counterexample fixed |
| P1-03 | Atomic relative/dot/Unicode/Windows paths | Implemented; local Windows and hosted Unix path regressions pass; real UNC-share durability remains unrun |
| P1-04 | Atomic seed/config and overlapping edits | Implemented; injected interruption, overlapping-edit regressions and final integration pass |
| P1-05 | Strict words, nonempty lists and guessability invariant | Implemented in shared loader, predictive/formal boundary and seed mutation; focused tests pass |
| P1-06 | Empty/gapped sync, completeness/counters/cancellation | Implemented; 22 isolated native sync tests pass; CLI separates attempted/applied/completeness. Closure review reproduced false GUI Ready with incomplete history; persistent exact-date coverage notice fixed, GUI 39 tests pass |
| P1-07 | Immediate Loading GUI; independent optional formal state | Implemented; asynchronous loading and corrupt optional formal-state tests pass |
| P1-08 | Current-request results; explicit policy only | Implemented; full request identity rejects stale replies and explicit profile preserves selected policy |
| P1-09 | Authoritative valid feedback draft | Implemented; raw draft remains authoritative, invalid edits preserve input; regression passes |
| P1-10 | Transactional history, terminal state and EOF | Implemented shared transactional game transition and CLI/GUI terminal handling; focused tests pass |
| P1-11 | Cancellation, worker exit/restart and spawn errors | Implemented GUI cancellation/disconnect/retry/spawn handling and formal cooperative cancellation; focused tests pass |
| P1-12 | Stored formal primary action; shared bounded alternatives | Implemented bounded/cancellable API. Independent review reproduced public suggest discarding its primary at the cap; exact primary now survives, CLI exposes incomplete alternatives, valid 10,001-probe regression passes |
| P1-13 | Bounded formal parsers/certificates and corruption checks | Implemented bounded parsing/allocation and corruption tests. Independent review added regular-file checks before open/after handle acquisition; native formal group 49 and hosted Unix FIFO regression pass |
| P1-14 | Complete exclusive window-scoped sealed ownership | Implemented complete-date/output-path preflight, exclusive overlapping-window ledger, preserved legacy records and private snapshot rechecks; eight seal tests and independent review pass |
| P1-15 | Eligible target sets for tuning and labels | Explicit eligible target sets implemented for tuning and labels; excluded-hole/history test and reconstructed dataset resume checks pass |
| P1-16 | Explicit candidate/required-holdout errors | Implemented contextual candidate and required-holdout error propagation; primary/holdout failure regressions pass |
| P1-17 | Typed artifact status, actual date and promotion | Implemented typed Missing/Invalid/Valid lookup, strict corruption handling, actual artifact dates and conditional promotion; five book regressions pass |
| P1-18 | Direct late-turn objective; measured first-feedback work | Direct one/two-turn routing implemented, dynamic/cancellation/unguessable regressions pass; current selected-policy top 1/5/10 first-feedback measurements complete in 844-1364 ms, 10 ms cooperative cancellation stops at 16.49 ms; see PERFORMANCE.md |
| P2-01 | Actual route/objective/scope/quality metadata | Implemented actual route/objective/action-scope/root-coverage/value-quality/stop metadata across API, CLI and GUI; five matrix tests, finite 30, terminal three and GUI 36 pass |
| P2-02 | Typed GUI severity and completion state | Implemented typed GUI states/severity separate from bounded stop reason; tests pass |
| P2-03 | Shared immutable table/vocabulary | Arc-owned immutable inputs and private per-clone caches implemented; pointer/state-isolation tests pass; release measurement records four allocations / 290 bytes per clone rather than copying the pattern table |
| P2-04 | Flat parallel table and streaming persistence | Implemented; six tests pass; 2048x512 toy build/load peak allocations approximately halved |
| P2-05 | Immutable identity reuse and per-request key | Bounded 32-entry identity/snapshot reuse implemented; repeated requests, config/data changes and clone isolation regression passes |
| P2-06 | Cumulative cooperative budgets; retained complete chunks | Inner guards cover teacher, replay, studies, evidence, books, session computation and regret; interrupted work charged, whole chunks retained. Slow-final-checkpoint and regret deadline regressions pass with final integration |
| P2-07 | Independent exhaustive label teacher | Independent iterative all-action continuation teacher implemented without production pools/thresholds; oracle tests and native dataset/resume/tamper/interruption integration pass |
| P2-08 | Correct ridge/intercept without standardization | Implemented; four fit modes, exact 4/3 slope and 7/3 intercept fixture pass |
| P2-09 | Explicit experimental inference errors | Implemented fallible learned/survival inference and promotion validation; focused tests pass |
| P2-10 | Additive survival solve aggregation | Implemented additive solve totals and unique reuse-event accounting; unequal-fold regression passes |
| P2-11 | Explicit undefined metrics and finite aggregation | Canonical, survival, experiment, tuning, prior-ablation and search-regret metrics preserve absent populations explicitly. Final review reproduced zero-state regret as numeric zero; optional mean/max and population-labeled CLI now pass RED/green empty/mid/final-deadline round-trip tests. Additive totals, all-gap tests and strict prior-population validation pass |
| P2-12 | Canonical eras, convergence and unique events | Implemented canonical era basis, convergence/zero-risk rejection and unique event counts; focused tests pass |
| P2-13 | Cross-field dataset/checkpoint/fold completeness | Format-2 exhaustive artifact and reconstructed-resume validation plus format-19 calendar/fold validation implemented; native eval group 78 passes |
| P2-14 | Indexed action lookup and early row budget | Indexed materialization and pre-work row caps implemented; 19 artifact tests pass; teacher avoids constructing a whole graph |
| P2-15 | Explicit formal objective independent of name | Implemented; explicit enum and versioned identity, exact built-in migration only |
| P2-16 | Coherent formal generations and end-to-end timing | Immutable checksum-bound generations published by pointer last, phase timings separated; independent generation/runtime review complete; formal 49 tests and earlier two integration fixtures pass |
| P2-17 | Shared live/replay rules and offline boundaries | Shared replay selection/transition exercised by two normal/hard public parity tests; early-green bug reproduced/fixed. Study, sealed orchestration and formatters extracted with explicit imports and byte-preservation check; eval 78 passes |
| P2-18 | GUI runtime/views and real CLI workflow boundaries | GUI runtime/views and CLI evidence publication module extracted; current GUI 39 and CLI 42 tests pass in final integration |
| P2-19 | Authoritative metadata and valid sampling domains | Registry cohort/domain source consolidated, invalid generic bounds rejected, canonical feature names shared; registry 8/config 9/learned 15 tests pass |
| P2-20 | Date-bounded history export | Implemented; future-only/repeated dates, inclusive cutoff and raw/effective summary regressions pass |
| P2-21 | Platform core/persistence/CLI/GUI CI; pinned toolchain | Three-platform locked matrix and local toolchain pin implemented; hosted tests/builds/CLI smoke pass for d2024bd; native GUI acceptance remains unrun |
| P2-22 | Dependency policy and pinned workflow revisions | Advisory workflow, immutable action pins and reviewed maintenance exceptions implemented; cargo-audit 0.22.2 found zero known vulnerabilities on current lockfile |
| P2-23 | Owned scratch and no late unsafe env mutation | Owned test/benchmark fixtures and removal of late env mutation implemented; concurrent integration 7+7 and characterization 16+16 pass. Closure review migrated remaining fixed library fixtures to exclusive RAII roots with shared clone lifetime; full concurrent library runs 533+533 pass. All-target benchmark smoke executes formal build/verify after predictive/Rayon work |
| P3-01 | Full reachable candidate list and consistent export | Virtualized full candidate list and CSV/filter parity implemented; row-height geometry regression fixed; GUI 34 tests pass |
| P3-02 | Stable policy rank while sorting | Stable policy rank retained when sorting; headless regression passes |
| P3-03 | Central contrast-safe feedback palette | Central black-on-feedback palette meets normal-text contrast; regression passes |
| P3-04 | Remove unused calibration generator/constants | Removed unreferenced calibration generator, row type and constants; historical evidence untouched; final all-target gate passes |
| P3-05 | Remove aliases/ignored args; share small helpers | Unreferenced aliases/forwarders and ignored forced-evaluation arguments removed; canonical feedback/feature/README helpers shared. All 243 GUI feedback encodings pass; diagnostic suite v2 removes inert forced-top knob. Final all-target gates pass |
| P3-06 | Positive scale prefixes, derived target, valid resume | Positive scale prefixes, source-derived target and strict resume implemented; formal tests pass |
| P3-07 | Visible contextual Windows startup errors/log | Windows startup message/log preserves error context; focused test passes; native app remains closed |

## Completion checks

- Formal weighted/uniform and horizon-slack fixtures, ExpectedOnly beyond small-state threshold, independent reference and persisted reload parity.
- Dynamic five-word threshold 0/1/2/4 all-root oracle, hard mode, one-turn, dormant removal and memo reuse.
- Atomic/input corruption and interruption tests; fake HTTP sync; concurrent seed/ownership/temporary fixtures.
- GUI invalid/partial drafts, contradictory histories, terminal undo, stale replies, dropped worker, cancellation, layout/palette and startup error behavior.
- Excluded-date sentinel gaps, exact unique seal dates, concurrent seal consumers, teacher independence, corrupted checkpoints, empty/all-gap metrics and unequal/nonconverged survival folds.
- Rust 1.97 format, warning-denied locked all-target Clippy/tests, parser mutation/fuzz coverage, exact CI doc checks, redaction and local links.
- Bounded relevant development measurements after final code; release build, matching dist hashes, PE subsystem/CLI smoke, docs and verified artifact cleanup.

## Carry-over and decisions

- Retain prior correctness tests, source-bound evidence, date guards, GUI overlap checks, demonstrated performance improvements and runnable dist requirements.
- Earlier-turn policy qualification and prospective confirmation remain research gates; they do not justify delaying concrete audit corrections or promising three guesses.
- Avoid intermediate full-matrix rebuild/rebenchmark cycles while audit changes remain in flight. Historical results retain their source identity.
- Ruling: user-authorized implementation used disjoint owners; the subsequent commit/push authorization applies to reviewed release content. Keep the native GUI closed.

## Publication handoff

The user authorized cleanup, commit and publication to the repository's existing
default `master` after the local delivery checkpoint. Private word-bearing research
outputs remain local; only reviewed redacted evidence is included. Intermediate
worker/checkpoint notes are retained in ignored task evidence rather than release
history. Hosted acceptance belongs to the CI run for the published commit; local
Windows checks above do not imply a cross-platform pass.

The first published-source run (`3ef8978`) passed Windows/macOS and the separate
dependency audit, but Linux failed four library tests with exhausted fixture
deadlines. The teacher fixture spent 32.4 seconds before its tiny search and only
0.1 seconds collecting/solving states; identity rechecking then took another
14.3 seconds. Both preflight paths hash the complete test executable. CI now
tests with line-table-only debug information to avoid embedded-DWARF hashing
overhead while retaining file/line backtraces and every original assertion,
deadline and executable-content check. The
[failing run](https://github.com/FueledByRedBull/maybe-wordle/actions/runs/36590240153) is retained;
the [follow-up run for `d2024bd`](https://github.com/FueledByRedBull/maybe-wordle/actions/runs/36591624931)
passed all three platforms. Linux ran all 603 tests, including the four previously
failing fixtures, with zero failures or ignored tests; its library suite took
88.17 seconds versus 243.73 seconds in the failing run. Unix symlink, directory-sync
and FIFO regressions also passed on Linux and macOS. The unchanged lockfile passed
the separate [dependency audit](https://github.com/FueledByRedBull/maybe-wordle/actions/runs/36590284316)
with only the two documented maintenance exceptions. These checks do not establish
native GUI interaction, real UNC-share durability or the separate research gates.

The documentation-only `ef9c13c` [repeat run](https://github.com/FueledByRedBull/maybe-wordle/actions/runs/36593982576)
passed Windows/macOS but exposed remaining Linux timing instability: 534 library
tests passed, and two staged-certificate fixtures completed their expected replay
counters but exceeded their 10-second budgets during final executable rehashing.
The smaller debug payload alone was therefore insufficient. The CI test command
now also applies `profile.test.package.sha2.opt-level=3` to the existing non-generic
SHA-256 compression implementation. Application code remains unoptimized; all
assertions, overflow checks, concurrency, content rechecks and deadlines remain
unchanged. No dependency version or release-build setting changed. Every published
head still requires its own successful matrix; a prior pass does not erase this failure.

### Local artifact recovery

The owner confirmed removing ignored `dist` and `target` during local cleanup
after their contents had been verified. Tracked source, public redacted evidence,
raw data and all 95 explicitly excluded local paths remained present. This
supersedes earlier preservation statements for local `target` contents; it does
not erase the recorded checks or CI results.

An offline locked release rebuild from `d2024bd` and documentation-only changes
passed in 1m 39s. Both executables were restored to `dist`, with fresh checksums
and build notes; CLI help passed and the GUI stayed closed. Hosted logs were
downloaded again into `target/audit-work`. Earlier local-only checkpoints, detailed
verification logs and intermediate worker notes were not recovered, including the
interrupted seven-profile checkpoint. Do not claim that experiment is resumable.
Benchmark identities remain unchanged and distinct from the recovered distribution.
A checked profile-only Cargo clean then removed 2,115 rebuild files (859.3 MiB),
preserving the restored dist and recovered verification logs.
