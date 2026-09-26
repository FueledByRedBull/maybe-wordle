# Maybe Wordle remaining work

The predictive solver is the primary product. [September release acceptance](docs/SEPTEMBER_RELEASE.md)
is the detailed requirement and evidence ledger. Keep the declared August 28-September 26
seal untouched during development; the consumed June 18-July 17 window is not reusable.
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
  of dynamic mode. Keep the production policy staged and require a new guarded
  comparison before any further bounded-policy decision. A staged-top-eight
  finite-value overlay is not a production shortcut: sampled roots and
  deadline-limited upper bounds cannot certify the global six-turn action.
  It may be used only as a counterfactual diagnostic unless an exact,
  all-legal-root route clears the same quality and resource guards.
  A cheap horizon-aware zero-failure certificate could cover roots whose
  every non-green bucket has at most `remaining_turns - 1` guessable answers
  and cannot activate dormant fallback. It would certify only those roots,
  not the globally best action; root states above 1,211 survivors cannot
  meet its six-turn bucket-size condition. Count eligibility on allowed
  development states before adding it to production. No current evidence
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
- [ ] Finish the measured kernel profile and only justified simplifications.
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
  staged kernel has an in-process profile; still finish broader paint/resource
  checks and do not mistake this bound for the six-turn policy objective.
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
  The required seven-profile current-schema gate is still incomplete: its
  earlier 60/2,520-game probe projected roughly 48 minutes, above the user's
  roughly 20-minute benchmark ceiling. A fresh current-executable probe on
  nine allowed July 28-August 5 dates completed all 63 profile-games in
  69.8 seconds with no sealed evaluation. Linear extrapolation to 2,520 games
  is about 46.5 minutes; that is a runtime estimate, not a full-run result.
  Do not launch the full gate under the stated ceiling or claim it passed.
- [ ] Freeze a selected candidate before a genuinely later prospective window;
  record the freeze and consumption dates. Do not reuse the seal or call a
  retrospective result prospective. Three guesses is an aspiration, not a gate.

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
  A fresh native launch exposed the rebuilt window, but the Windows UI helper
  again failed to activate it twice on September 26, so native visual/keyboard acceptance is
  still open. The desktop was locked; `PrintWindow` and accessibility bounds
  cannot establish what the eframe GPU surface painted or verify keyboard
  interaction in that session. Retry the native pass only on an unlocked desktop.
  Do not mark native
  verification complete from geometry tests alone.
- [ ] Obtain native macOS memory-sampler evidence from CI after an authorized
  push before claiming platform validation. The last recorded remote check
  found seven Dependabot alerts against the pushed lockfile; the uncommitted
  lockfile meets their reported patched versions. The approved `eframe` 0.32.3
  update also removes the vulnerable `quick-xml` 0.30.0 Linux accessibility
  chain; a local advisory scan now passes with two unmaintained warnings.
  Recheck alert status and macOS CI after push, and assess those warnings
  separately.
- [ ] Finish the final whole-repository diff, proposed release file-scope and
  hygiene review after any remaining implementation changes. Current
  release-decision arithmetic, generated-doc check, local links and historical
  archive pass; do not treat older artifacts as changed-code validation.
  Recheck Rust 1.97 format, warnings-denied Clippy and all-target tests after
  any source change. Both retained `dist` executables now match the latest
  uncommitted source build and are runnable; the GUI passed a startup smoke,
  but not native interaction. Keep `dist` runnable and retain evidence
  checkpoints.
- [ ] Resolve evidence publication before staging: the new benchmark JSON
  contains per-game target and path words; some already-tracked historical JSON
  has the same issue. Local public copies of the current selected-policy and
  earlier rolling artifacts now replace each game target/path word with a
  redaction marker while retaining outcome and calibration arithmetic. The
  existing Rust documentation verifiers and a structural redaction check pass
  locally; CI now invokes those source-backed checks, and README's generated
  predictive table presents the current 3.1944/3.1917 development comparison.
  This validates internal arithmetic and presentation, not replay from raw
  history or rights to publish the word-bearing sources. Include the two
  public JSONs, redaction script, and current generated fragment in the
  reviewed release scope. Historical links to untracked raw predictive JSON
  are now plain local-only paths. Include the linked aggregate generated
  summaries, release ledger, and experiment configs deliberately as one docs
  bundle so a clean checkout does not have broken links. Also include untracked
  `src/solver/finite.rs` and `src/solver/online.rs`; omitting them breaks
  compilation. Do not blanket-stage
  private raw JSON or assume redaction establishes source authenticity.
- [ ] Clarify whether the consumed June 18-July 17 window is excluded only
  from new validation/tuning targets or also from later-date supervised model
  fitting. The survival experiment uses it as chronological training history
  for later folds but never as their validation target. Do not change model
  semantics or claim a leak resolution until that policy decision is explicit.
- [ ] After the authorized `master` push, inspect CI and dependency-alert
  results. Keep any remaining native GUI, prospective, and seven-profile
  validation gates open rather than treating the integration push as release
  acceptance.

Formal expansion remains conditional on a concrete feasibility improvement;
it is not a prerequisite for a useful predictive release.
