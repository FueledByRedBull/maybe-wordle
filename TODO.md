# Maybe Wordle remaining work

The [September 29 audit remediation ledger](docs/superpowers/plans/2026-09-29-audit-remediation.md)
records all 48 implemented findings, regression evidence and platform limitations.
The research/release carry-over below remains separate from that audit delivery.

The final Windows Rust gates pass 603 tests and 15 benchmark smoke workloads.
First-feedback/allocation/cancellation measurements, fresh redacted development
evidence and rebuilt `dist` are recorded in the ledger. The audit rolling replay
reproduces all 720 prior selected/v19b paths and outcomes: 3.194444/3.191667, both
360/360 solved, paired interval crossing zero. The full seven-profile matrix hit
its 20-minute cap and is explicitly incomplete; its nine-date timing screen is
not a substitute. Hosted Windows/Linux/macOS tests, builds and CLI smoke
[passed for `d2024bd`](https://github.com/FueledByRedBull/maybe-wordle/actions/runs/36591624931).
Native interactive GUI and real Windows UNC-share acceptance remain open.
The ledger also records local cleanup of ignored `dist`/`target` during
publication: executables and hosted logs were restored, but earlier local-only
checkpoints were not. The interrupted seven-profile attempt cannot be resumed locally.

The predictive solver is the primary product. [September release acceptance](docs/SEPTEMBER_RELEASE.md)
is the detailed requirement and evidence ledger. Keep the declared August 28-September 26
seal untouched during development; the consumed June 18-July 17 window is not
reusable for validation or tuning. It may remain chronological training
history for later dates, but never a new validation target.
Historical production scores (12-fold development mean 3.1778; once-only sealed mean
3.3000) do not validate the changed solver or promise a flat-three mean.

## Solver and evaluation

- [ ] Find a predictive candidate that improves the guarded outcome before
  changing selected v20. The earlier 12-fold terminal-rule replay gives
  selected staged 360/360 solves at 3.1944 and exploratory v19b 360/360 at
  3.1917 on the 12 allowed development folds, both with zero coverage gaps.
  Selected staged still uses unlimited-horizon ranking before the final two
  turns; the six-turn failure-first objective and recursive hard-mode policy
  are implemented in finite modes, not yet across the selected policy.
  The two-turn shortcut uses surviving answer words as final replies; those
  words satisfy all observed hard-mode clues by construction. An explicit
  hard-mode final-reply regression now checks this with duplicate-letter
  feedback; it does not provide recursive hard-mode search in earlier turns.
  A root-bound optimization then reproduced all 720 paths again in 8.2 minutes
  versus the preceding 16.2-minute run, and avoided consumed and reserved
  dates. The earlier post-dynamic-final and post-dynamic-mode replays remain
  historical. The completed [pre-validator-build replay table](docs/generated/september-post-layout-tests-rolling-v1.md)
  used CLI SHA-256 ECFEE91B8199956E327954BD5E5BE233F07129A8C2CDE3D6F4612CA740B46B72
  over the 12 allowed folds. All 720 outcomes and paths matched the preceding
  replay exactly; selected staged and v19b both solved 360/360 with zero gaps/
  failures at 3.194444/3.191667, p95 20.7277/20.7633 ms. The paired
  v19b-minus-selected delta was -0.0027778, 95% CI [-0.0222222,+0.0166667],
  with 6/347/7 wins/ties/losses. It took 508.3 s and peaked at 168898560 bytes
  (about 161.1 MiB); `sealed_test_evaluated=false`. The full per-game JSON is
  local pending publication review. This supersedes the earlier source identity
  but remains retrospective development evidence, with no promotion or flat-three claim.
  Earlier-build finite
  Fast scored 3.5556/3.5639 with 360/360. Preserve the negative evidence in
  the release ledger. Fixed-belief matched comparisons are recorded, and the
  finite kernel now also has an explicit dynamic-belief entry point whose
  recursive recovery and hard-mode values match a small independent oracle.
  A rejected interim selected-policy trial used it on turns one through four:
  July 28-August 26 scored 3.4333 versus the retained staged policy's 3.3333,
  both 30/30 solved, with per-step p95 259.22 versus 21.59 ms. The selected
  path was restored and reproduced all 30 earlier game paths; future trials
  need matched belief/value diagnostics before promotion. A repeatable 30-game fixed-work
  tail-mass ablation found no reason to reduce the finite tail: 3.5000 selected
  versus 3.6333 near-core, paired interval crossing zero. The matched-belief
  30-day replay then scored staged 3.2000 versus finite fixed-work 3.5000,
  with a positive paired interval; frozen-belief staged's gain over dynamic
  staged was inconclusive and failed calibration/latency guards. A read-only
  paired-path audit of that allowed 30-day artifact found 30 unique dates,
  no empty paths, and 0/30 matching first guesses between its matched-belief
  staged and finite fixed-work profiles. The finite root shortlist includes
  its own baseline move, not necessarily staged's move; explicitly seeding
  staged's move is a bounded hypothesis to test, not a demonstrated score gain
  or an exact global six-turn certificate. In that same artifact, the staged
  opener appeared in none of the finite first-turn top lists; all 30 finite
  roots sampled proposals, reported seven completed candidates, and stopped
  at the 4,000,000-work-unit limit. This makes root seeding a useful
  counterfactual, not evidence of a better finite value. Preserve
  both negative results in the release ledger. A same-belief finite Strong
  follow-up also failed: staged 3.2000 versus Strong 3.5667 over 30/30 solved
  games, paired Strong-minus-staged interval [+0.1667, +0.5667], with per-move
  p95 334.71 versus 2016.33 ms. More finite search alone is not a promotion
  path. A top-4,096-answer proposal sample covered 94.2783% of mass and
  reversed two full-support proxy ranks, but adding a full-support anchor
  left all five screened Strong game paths unchanged; the experimental code
  was removed. Matched-belief finite roots modeled about 1% failure risk;
  30/30 actual solves do not establish whether the extra caution pays off.
  A reproducible opt-in [dynamic-belief finite comparison](config/experiments/september-dynamic-finite-matched.json)
  now covers the same allowed dates: the five-day screen (`benchmarks/predictive/september-dynamic-finite-screen-v1.json`)
  scored staged/dynamic 3.0000/3.2000 with 5/5 solves, and the 30-day artifact (`benchmarks/predictive/september-dynamic-finite-30day-v1.json`)
  scored 3.3333/3.4000 with 30/30 solves, zero gaps/failures, paired
  dynamic-minus-staged +0.0667 [-0.1333,+0.2667], 4/21/5 wins/ties/losses,
  and p95 21.24/256.62 ms. Initial log loss/Brier were identical at
  6.5971/0.9986; generation took 52.10 s and peaked at 133.3 MiB. Dynamic
  roots chose `olate` all 30 times, averaging 0.004110 modeled risk and 3.3002
  expected attempts, with every root at the deadline; both profiles recorded
  five recovery steps. It passes neither score nor latency guard, is retrospective
  development evidence (`sealed_test_evaluated=false`), and does not authorize
  promotion. The fixed-belief-only finite regret diagnostic is not validation
  of dynamic mode. A September 27 current-executable repeat on those same 30
  allowed development dates preserved all 60 prior paths/outcomes and again
  scored staged/dynamic 3.3333/3.4000 with 30/30 solves. All first guesses
  matched; 25 games first diverged on guess two. Every divergent finite choice
  was deadline-limited with only an upper-bound value, and those 25 games gave
  finite/staged 4/5 wins (16 ties). Per-move p95 was 21.52/256.20 ms. The raw
  local report is `benchmarks/predictive/september-dynamic-finite-current-30day-v1.json`;
  it is not public or prospective. No stage trigger is justified by this
  replay. An opt-in same-state dynamic-belief regret check is now available for
  explicit development states, but its exact reference is limited to six
  combined active/dormant survivors and is not policy-level validation; see
  the [spot checks](docs/SEPTEMBER_RELEASE.md#september-27-same-state-dynamic-regret-spot-checks).
  An evenly spaced ten-date development screen at turn four yielded eight
  already-solved paths, one exact four-survivor tie, and one eleven-survivor
  unresolved state; it found no additional exact disagreement or promotion
  evidence. Two initial deadline cases resolved as already solved with a
  larger bounded budget.
  A September 27 one-pass dormant-fallback partition preserves the dynamic
  child transition in duplicate-feedback/fallback-activation regressions and
  lowers mean finite work units on the same 30 development dates from 8.40M
  to 7.78M/7.65M/7.53M in three post-change runs. Dynamic finite scored
  3.3000, 3.2667 and 3.3667 versus staged's 3.3333, all 30/30 solved; every
  paired interval includes zero and finite p95 remains 260-265 ms. The two
  initial post-change repeats matched only 27/30 finite paths, reflecting
  deadline-sensitive search; the final rebuilt executable also failed to
  reproduce their apparent point-score lead. These results are neither
  prospective nor a score/latency promotion decision.
  Keep the production policy staged and require a new guarded
  comparison before any further bounded-policy decision. A staged-top-eight
  finite-value overlay is not a production shortcut: sampled roots and
  deadline-limited upper bounds cannot certify the global six-turn action.
  It may be used only as a counterfactual diagnostic unless an exact,
  all-legal-root route clears the same quality and resource guards.
  A cheap horizon-aware zero-failure certificate could cover roots with no
  dormant fallback support when every non-green child has at most
  `remaining_turns - 1` guessable answers under the full hard-mode history
  and no dormant fallback left. Merely avoiding immediate fallback activation
  is insufficient: a wrong sequential reply might activate dormant support
  later. Subsequent sequential-reply transitions must also remain usable under
  recovery rules. This certifies only the selected root under modeled support,
  not the globally best action; root states above 1,211 survivors cannot
  meet its six-turn bucket-size condition. An opt-in development-only
  `staged-zero-failure-certificate` diagnostic now applies a stricter safe
  check: no dormant root/child support and strictly positive modeled child
  weights, plus dictionary/hard-mode legality. Its complete August 20 smoke
  checked four roots and certified none. A July 28-August 5 screen hit its
  four-minute deadline after seven complete games and 22 roots, with zero
  certificates; it is incomplete and cannot establish a nine-day eligibility
  rate. Broader eligibility remains open before any production route.
  No current evidence
  shows an exact early-turn route at the selected policy's roughly 21 ms p95.
  A read-only audit of the allowed fixed-state latency artifact found a
  14,855-survivor six-turn root whose finite searches sample proposals, hit
  their deadlines, and label their returned values heuristic or bounded.
  That root cannot meet the cheap bucket-size certificate. Exact all-legal-root
  optimization at the current latency remains unproven; keep the finite result
  labeled as a usable move, not a certified global optimum.
- [x] Add a public-path hard-mode regression for the opt-in dynamic finite
  route. `finite_fast_dynamic_public_history_preserves_hard_mode_fallback_and_six_turn_boundary`
  exercises non-empty feedback history, dormant-to-active fallback, legal
  candidates, positive failure mass after activation, and the six-turn limit.
  The full predictive-characterization suite passed on Rust 1.97. This closes
  the test gap, not the candidate's score or latency release guards.
- [x] Guard the finite two-active-answer exact shortcut against dormant
  reactivation. A threshold-one oracle regression failed on the old zero-risk
  value (actual zero versus 0.0586543 expected), then passed with recursive
  child evaluation. One active legal answer still resolves immediately. The
  selected threshold-four configuration activates dormant support before this
  state is reachable; this fixes alternate low-threshold configurations, not
  a demonstrated production score gain.
- [ ] Finish native paint/resource acceptance; keep further kernel changes measured.
  A same-executable fixed-state contrast shows the baseline is much cheaper
  than Fast refinement, but does not identify every internal hot stage.
  Profile the four-observation staged path in-process before changing it
  again: it still runs unlimited-horizon refinement before the terminal-success
  re-sort. A 30-pair process-level comparison on the `junco` state found
  identical top five words with proxy preview and a 6.05 ms paired median
  full-minus-preview difference, but startup/model loading confound attribution.
  Predictive small-state precomputation is already removed; do not reopen the
  rejected constant-liar or learned-model routes without new evidence.
  The first-feedback pooled-exact stall is now isolated and its admissible
  top-prefix root bound reduced a matched top-ten CLI call from 55.15 to
  3.92 seconds without changing the 720 development paths. The four-observation
  staged kernel has an in-process profile. A September 27 current-source
  Criterion repeat estimated 13.40 ms for preview and 18.27 ms for full
  suggestions on that state. Its roughly 4.88 ms refinement delta does not
  justify removing exact-cost metadata consumed by the GUI/CLI. Still finish
  native paint/resource checks and do not mistake this bound for the six-turn
  policy objective. The September 29 audit additionally measured actual selected
  first-feedback top 1/5/10, warm calls, shared-table clone allocations and
  cooperative cancellation; see [PERFORMANCE.md](docs/PERFORMANCE.md). Those
  measurements close the audit's kernel-measurement item, not native UI acceptance.
- [ ] Complete conditional cohorts in cost order and send only Pareto finalists
  through small multi-seed joint refinement, paired validation and seven-profile
  development evidence. The prior-family screen, small-state/proxy-risk cohorts,
  bounded two-seed joint screen, and fixed-state resource checks are recorded;
  none authorizes production promotion. The staged v19b entropy screen
  improved the last 30-day fold by one recovered game, but its full
  preceding-build 12-fold result was 3.1944/359 solves versus selected
  3.2000/358. The current terminal rule fixes the remaining late failures in
  both profiles, but v19b's 3.1917 versus selected 3.1944 paired interval
  still spans zero.
  This does not yet qualify as a release finalist. Keep dates, rules, effective beliefs,
  budgets, executable and data identities matched for causal comparisons.
  The small-state and proxy-risk screens did produce provisional Pareto-rank-0
  candidates, but their downstream repeats/paired comparisons did not qualify
  either for promotion. Do not call this a screen with no Pareto candidates.
  The current-schema seven-profile incumbent/baseline gate is now complete:
  12 allowed development folds, 2,520/2,520 profile-games, zero failures and
  coverage gaps, with no sealed evaluation. The rebuilt current-executable
  rerun took 665.16 seconds of generation compute (about 11.1 minutes), under
  the user's roughly 20-minute ceiling. All 2,520 game outcomes and guess
  paths matched the preceding 15.12-minute build. Selected staged and
  staged without artifacts each scored 3.1944 over 360 games; every selected
  game path matched, but their separate artifact modes remain separate profiles.
  The previous-release baseline scored 3.3306, proxy-only 3.2444, and
  proxy-plus-exact-endgame 3.2000. These are retrospective development scores,
  not evidence that the changed solver improved mean guesses or reached 3.0.
  The full run is retained privately under `target/diagnostics/`; only its
  redacted copy may be published. The bound and deterministic answer-first
  ordering in recursive exact search reduced a matched nine-date seven-profile
  slice from 80.62 to 17.98/17.96 seconds on two repeats with identical 63
  game payloads. This is a runtime gain, not a score gain. A new candidate still
  needs its own matched paired validation and full release guards before any
  promotion or prospective freeze; no current candidate qualifies.
  A current-CLI same-state dynamic audit on all 30 July 28-August 26 dates at
  turn four found 22 games already solved, six exact references at support
  one through six, and two states above the six-answer reference limit. Of
  the six exact states, one (August 24) preferred the dynamic finite action
  by 0.1114 modeled expected attempts at equal zero failure probability;
  the other five choices agreed. This does not establish a favorable routing
  class or all-game score gain. The broad-root finite shortlist omitted the
  staged action in the existing matched-belief screen, but a temporary
  diagnostic seed did not help: 71 eligible state evaluations across sampled
  turn-one and complete turn-two through turn-four development cohorts gave
  the staged root a completed bounded evaluation. None changed the finite
  choice, result quality/stopping reason, or modeled failure/attempt value.
  Eighteen small states had exact references; the larger states did not.
  The trial code was removed; retain the negative private reports under
  `target/diagnostics/`. This does not justify another full policy comparison.
  At turn five in this cohort, only two games remained; both staged choices
  had zero exact regret, so the evidence still points to earlier turns rather
  than replacing the terminal rule.
  A complete July 28-August 26 turn-three same-state screen on the rebuilt
  CLI found two paths already solved before that turn, 12 exact-reference
  states, and 16 states above the six-answer combined-support limit. Both
  policies chose the same exact-optimal action in all 12 exact states.
  Four of the 16 larger states had different actions, but none has an exact
  reference; this does not establish a favorable early-turn route. The
  private summary-only reports are under `target/diagnostics/`.
- [ ] Freeze a selected candidate before a genuinely later prospective window;
  record the freeze and consumption dates. Do not reuse the seal or call a
  retrospective result prospective. Three guesses is an aspiration, not a gate.
  The existing v19b comparison cannot freeze the staged incumbent: the command
  freezes only the comparison's eligible candidate, and v19b's paired upper
  confidence bound is not below zero. The old once-only marker remains global
  to the consumed seal. A distinct `freeze-prospective` /
  `evaluate-prospective` workflow now binds an eligible development winner to
  the next 30-day UTC window, with exact-date preflight, a global one-window
  reservation, and a dated exclusive marker. Freeze schema v2 requires complete
  daily history after the development cutoff through the UTC freeze date and
  binds that pre-window training-history digest; report output cannot share
  the marker directory. Its synthetic tests do not
  consume any real target. A September 27 UTC freeze would select September
  28-October 27; a later freeze shifts the window. No candidate has cleared
  the eligibility guard, so no real freeze or evaluation has occurred.

## Release and maintenance

- [ ] Verify the rebuilt Windows GUI visually and with keyboard/accessibility
  interaction at the first-guess board and recommendations, narrow layout,
  feedback, undo and hard mode. A failing-then-passing egui regression caught
  a 564-pixel tile overflowing the first column; fixed-width tiles are in the
  retained dist GUI, but the native capture helper still cannot activate its
  window. A stronger test now compares both draft and applied first-guess
  tile rectangles with the right suggestions panel at 861px and 1180px; it
  passes. A native UI Automation retry exposed the live Play controls and
  separate board/suggestions headings, but its first-row entry did not
  commit and the locked desktop yielded no trustworthy pixel capture.
  A separate minimum-window regression reproduced the feedback controls losing
  their fifth button; wrapping the row fixed it. A production-path layout
  regression now renders both the true first-guess draft at 1180/1210px and
  an applied first row with a draft next row at 1240/1260px, each with a real
  recommendation at 135% text scaling; its board and recommendation bounds
  pass. Rust 1.97 formatting, warnings-denied Clippy, and all-target tests
  pass after the added first-guess cases.
  The reported wide-window overlap is not reproduced by geometry tests.
  On September 27 an unlocked-desktop native pass showed the first-guess draft
  and applied tiles beside recommendations without overlap at about 1180px;
  Enter, Undo, Hard Mode, suggestion inspection, and Reset worked. A narrow
  native window and enlarged text were not verified. Keep this gate open for
  those cases and accessibility acceptance; geometry tests alone are not enough.
  A focused September 27 GUI regression run passed 22/22 tests. A later
  Windows helper session launched the retained app and read its accessibility
  tree, but failed to activate the window or capture trustworthy pixels;
  the remaining narrow/enlarged-text interaction pass still needs a usable session.
- [x] Obtain native macOS memory-sampler evidence from CI. The September 27
  `macos-memory` job passed. After the integration push, the Dependabot API
  reported zero open alerts; the approved `eframe` 0.32.3 update removed the
  vulnerable `quick-xml` 0.30.0 Linux accessibility chain. A local advisory
  scan passed with two unmaintained warnings (`paste`, `ttf-parser`), which
  remain maintenance items rather than cleared warnings.
- [x] Finish the whole-repository diff, proposed release file-scope and
  hygiene review for the integration push. Current
  release-decision arithmetic, generated-doc check, local links and historical
  archive pass; do not treat older artifacts as changed-code validation.
  Rust 1.97 format, warnings-denied Clippy and all-target tests were rerun
  after the CI correction; the small final diff was reviewed before publication.
  Both retained `dist` executables are runnable, and the GUI passed the
  unlocked-desktop interaction subset above. Keep `dist` and evidence
  checkpoints.
- [x] Resolve evidence publication for the integration push: the new benchmark JSON
  contains per-game target and path words; some already-tracked historical JSON
  has the same issue. Local public copies of the current selected-policy and
  earlier rolling artifacts now replace each game target/path word with a
  redaction marker while retaining outcome and calibration arithmetic. The
  existing Rust documentation verifiers and a structural redaction check pass
  locally; CI now invokes those source-backed checks, and README's generated
  predictive table presented the then-current 3.1944/3.1917 development comparison.
  This validates internal arithmetic and presentation, not replay from raw
  history or rights to publish the word-bearing sources. The two public JSONs,
  redaction script, current generated fragment, linked aggregate summaries,
  release ledger, experiment configs, and required solver modules were included
  in the reviewed push. Historical links to untracked raw predictive JSON are
  plain local-only paths. Private raw JSON remains untracked; redaction does
  not establish source authenticity. The later seven-profile run has its own
  additional redacted public JSON and generated table; it supersedes the
  two-profile table as README's current evidence without rewriting that
  historical artifact.
- [x] Clarify the consumed June 18-July 17 policy. The owner permits these
  outcomes as chronological training history for later-date models, including
  the survival experiment, but not as new validation or tuning targets. This
  confirms the existing fold behavior; it does not promote the survival model
  or permit future-target leakage.
- [x] Require explicit, development-only target dates in the legacy `backtest`
  and `experiments` CLI commands. Both now reject the consumed validation
  interval and reserved seal before solver/history loading. The shared policy
  test confirms the latest fold may still train chronologically on the consumed
  interval. This is leakage prevention, not a new score result.
- [x] Inspect the integration push's CI and dependency-alert results. The earlier
  Windows CRLF-only documentation failure was corrected. The September 29 audit
  push then exposed Linux fixture deadlines exhausted by test-executable hashing;
  line-table-only test debug information reduced the overhead, and the three-platform
  matrix passed for `d2024bd` (linked above). A later `ef9c13c` run still hit two
  Linux completion deadlines after replay, so CI additionally optimizes only its
  SHA-256 dependency; assertions, solver optimization and deadlines are unchanged.
  Every published head must pass the matrix. The [dependency audit](https://github.com/FueledByRedBull/maybe-wordle/actions/runs/36590284316)
  passed on the unchanged lockfile with its two documented maintenance exceptions.
  Keep native GUI, real UNC-share, prospective and candidate-specific paired/promotion
  validation gates open rather than treating the integration push as full release acceptance.

Formal expansion remains conditional on a concrete feasibility improvement;
it is not a prerequisite for a useful predictive release.
