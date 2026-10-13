# Predictive sub-three research: close-out

Status: stopped at the owner's request on October 5, 2026. The under-three goal
was not achieved; the app goal is paused rather than falsely marked successful.
The owner requested keeping the best verified score/latency balance and closing
the speculative work. No new source screen, fit or gameplay run is queued.

October 12 update: an oracle kill-test confirmed the close-out and set the
reopening bar, and the owner asked for the open product items to be finished.
Unchanged selected v20 is frozen for one prospective window, 2026-10-13
through 2026-11-11. See the
[kill-test](#october-12-oracle-membership-kill-test) and the
[October 12 close-out pass](#october-12-close-out-pass).

## Product decision

Keep selected staged v20 in `config/prior.toml`. No research candidate met all
promotion requirements. Keep the already-integrated admissible exact-search
bounds and deterministic answer-first ordering: they have matched correctness
and timing evidence and do not require a new policy or research dependency.
Do not promote a faster research-data generator as a faster playing solver.

The four uncommitted Rust research changes were removed after preservation and
independent review. They were test-only penalty/hybrid/rollout/partition/tail
experiments, plus neutral production plumbing, not a new production fix. Normal
tests no longer import ignored `target/subthree-research` source or substitute
research behavior for production transitions. Application source and selected
configuration again match `042cb857e5444374ef1a9e55b936a42429397f89`.

## What the evidence actually says

Populations and executable identities differ across rows. Do not rank every row
as if it came from one matched tournament. Failures are scored seven in all-game
means; historical research is not untouched qualification.

| Comparison | Population and outcome | Decision |
| --- | --- | --- |
| Selected v20 vs v19b | 360 games each: 3.194444 vs 3.191667, both all solved; paired difference CI [-0.022222, +0.016667] | v20 retained; no qualified v19b win |
| Best hybrid research vs its control | 360 games each: 1,133 vs 1,143 guesses (3.147222 vs 3.175000), both all solved; paired CI [-0.094514, +0.036111] | Interesting point estimate, not a validated replacement |
| Actual-loss opener | 240 earlier training dates, 117 later development dates: 383 vs 389 guesses (3.273504 vs 3.324786), all solved | Missed the fixed twelve-save and uncertainty gates |
| Decision-belief / likelihood fits | 117 reused dates: control 356 guesses; decision 397 with one failure; likelihood 370, all solved | Both worse; no promotion |
| Participant-group familiarity | 360 reused calibration dates: post-opener log-loss gain 0.007012 nats vs required 0.03; fallback Brier worse by 0.000428 | Reject; no gameplay expansion |
| Matched 0.25/1/2/5/10-second curve | Twelve middle dates: 39 guesses at every tier | More time alone was not a useful lever |
| MASC spoken/written source | Metadata/protocol work only; source-helper implementation interrupted before a corpus screen | Unevaluated, not a failed or successful model |

The best 360-game research score still needs another 54 guesses saved to reach
at most 1,079, strictly below three. The older 71-save number belonged to the
original 1,150-guess baseline. Neither calculation proves the target feasible.

Human familiarity was the strongest prediction lead, but subsequent richer
static, semantic, phonological, source-contrast and decision-fitting variants
did not establish a deployable benefit. The approximately 49x acceleration of
28,521,600 fixed-policy training costs is a research-generation result, not a
GUI latency or gameplay-score improvement. Repeated negative trials and flat
time curves motivated this close-out, not a proof that under three is impossible.

## October 12 oracle-membership kill-test

Question: would the selected policy average at most three guesses if it knew
which pool each answer came from (never-used list word, reused answer or new
word) but not the word itself?

A temporary in-crate test hook replayed the same 360 development games
artifact-free with top 5. It was removed after the run; application source
again matches the committed tree. The control arm reproduced the retained
seven-profile selected outcomes on all 360 games. Two oracle arms were scored:
a state restricted to the target's pool, and the production state shape with
out-of-pool in-list words demoted to the cooldown-floor weight. For new-word
games the pool-only state is the only oracle arm. History on disk ended on
2026-08-26 and the hook asserted fold membership, so the reserved seal was not
touched.

| Answer pool | Games | Production | Oracle, same state shape | Oracle, pool-only state |
| --- | ---: | ---: | ---: | ---: |
| Never-used list word | 315 | 967 (3.0698) | 971 (3.0825) | 1,000 (3.1746) |
| Reused answer | 17 | 59 (3.4706) | 50 (2.9412) | 49 (2.8824) |
| New word | 28 | 124 (4.4286) | 116 (4.1429) | 116 (4.1429) |
| All | 360 | 1,150 (3.1944) | 1,137 (3.1583) | 1,165 (3.2361) |

A flat three needs at most 1,080 guesses. The same-shape oracle is 57 short and
its paired difference from production is -0.0361 guesses, 95% interval
[-0.0778, +0.0056]. Taking the best arm in every row still leaves 1,132.

- Pool membership is worth about 13 of the 70 guesses needed. For never-used
  answers, production's cooldown floor already acts as the membership oracle.
- The pool-only state loses 0.1048 guesses per game on never-used answers,
  interval [0.0254, 0.1841], with nearly the same belief. Production plays one
  opener in all 332 in-list games; the pool-only arm chose eight different
  openers, none equal to it, and left more in-pool candidates (mean 26.75
  versus 24.59, median 22 versus 18). The selected configuration is coupled to
  near-zero-weight used words remaining in the state. This is a robustness
  note, not a route to a lower score.
- New-word days stay at 4.1429 even with membership known: the dormant pool is
  about 12,500 words.
- The 17 reuse days show no weekday pattern. Fifteen of the reused answers
  first appeared within the first 190 puzzles; ages were 1,408-1,856 days with
  one at 524.
- The never-used pool fell from 857 words on 2025-07-03 to 494 on 2026-08-23,
  about 24-28 words per 30 days. Never-used days scored 3.0765 before reuse
  began and 3.0588 afterwards, within noise, so a pool-drain improvement is a
  hypothesis to track, not an observed trend.

Limits: this is the production policy under oracle beliefs, not the optimal
policy for them; the folds are in-sample for the selected configuration; the
intervals use a simple game-level bootstrap, not the chronological block
bootstrap; the same-shape arm keeps the dormant new-word hedge.

### Reopening bar

The remaining gap can only come from word-level prediction inside the
never-used pool. Across the 332 in-list games, guesses rise by about 0.283 per
nat of post-opener log loss; the uniform-to-weighted baseline ladder implies
about 0.36. Reaching three therefore needs roughly 0.5-0.75 nats, against
0.007-0.013 nats for the best signals found. Reopen only for a date-safe
signal showing at least about 0.3 nats of post-opener log-loss improvement on
a chronological holdout. The earlier 0.03-nat gate is superseded.

For scale, the published exact optimum for the static game is 3.421 average
guesses with opener `salet`
([Bertsimas and Paskov, Operations Research 73(3), 2025](https://doi.org/10.1287/opre.2022.0434)).
The selected policy's lower means come from conditioning on date-bounded answer
history, a different information set, not from better search on the same game.

Word-free per-game rows, the hook source and the analysis script are retained
privately under `target/oracle-membership-killtest/`.

## Retained product performance

The existing pooled root bound reduced a matched top-ten request from 55.15 to
3.92 seconds with identical words/costs; a 720-path replay fell from 969.5 to
491.2 seconds with identical paths/scores. Recursive admissible bounds and
highest-positive-mass-first ordering reduced a matched seven-profile slice
from 80.62 to 17.98/17.96 seconds with identical 63 game payloads.

The last complete seven-profile development matrix (September 28, pre-audit
executable) covered 2,520 profile-games
in 665.16 seconds, with zero failures/gaps; selected staged scored 3.194444 and
had 21.83-ms shared-process suggestion p95. These are retained measurements of
their recorded executables, not new close-out measurements or population-wide
real-time guarantees. See [PERFORMANCE.md](../../PERFORMANCE.md) and the
[September release ledger](../../SEPTEMBER_RELEASE.md).

## Evidence and recovery

The full 630,432-byte chronological narrative is retained privately at
`target/subthree-research/closeout/docs/superpowers/plans/2026-10-03-subthree-research.md`,
SHA256 `2f09b6fc8b1e55b69672a3182c2777530d98259e15b69193879839bac7baa6d0`.
The close-out folder also contains the original four source files, old TODO,
`research-working-tree.patch` and `preservation.json`. All snapshots were
hash-compared before removal from the live source. The patch is against
`042cb857`; replay only in an isolated research checkout with the retained private
helpers and inputs, never as a production update.

The existing private source/model/report/process freezes remain intact, including
failed, capped and superseded attempts. The final participant record is
`participant-choice-postrun.json`, SHA256
`4ecebfe3021709041733b43b7349eca03e1296d4a926391bd5fc66143063546e`.
Its independent reconstruction covered all 4,380 training rows, 4,320 baseline
scalars, 720 support pairs and 14,855 feature values; metrics and gradients agreed.
Research-only verification passed 931 tests before the hooks were retired. That
count includes private experiments and is not the final product test count.

Private evidence is not a build cache. Do not delete all `target`, run unqualified
`cargo clean`, or publish licensed/word-bearing artifacts. Profile-scoped cleanup
must preserve this material, `dist`, runtime data and unrelated files.

## Close-out verification and remaining limits

This section is the October 5 state. The
[October 12 pass](#october-12-close-out-pass) supersedes its open items and its
statements that nothing was frozen and the GUI stayed closed.

Final local verification on the restored product source passed:

- `cargo fmt --check`, locked/offline all-target Clippy with warnings denied,
  and 603 all-target tests plus 15 benchmark smokes with the CI test-profile
  settings. This is not a new full gameplay benchmark.
- Five public evidence redaction checks, both generated evidence documentation
  checks, and all 136 local Markdown links.
- `cargo audit` with warnings denied and only the two documented maintenance
  exceptions; see [DEPENDENCIES.md](../../DEPENDENCIES.md).
- A locked/offline release build of both binaries in 42.52 seconds. The retained
  `dist` copies match the build outputs by SHA256. Windows PE subsystem checks
  identify the GUI as subsystem 2 and CLI as subsystem 3; the packaged CLI's
  `--help` succeeds before and after cleanup. This does not validate native GUI
  interaction, and no fresh hosted CI run is claimed.

Profile-scoped Cargo cleanup removed 12,884 dev files (12.5 GiB) and 4,172 release
files (1.4 GiB). The four protected evidence-tree file counts and sizes remained
unchanged; original source/diary/patch hashes and distribution checksums still
match. Previous distribution copies remain in the close-out backup. Nothing
under the private evidence trees was discarded as a build cache.

Publication is documentation-only, excluding the local TODO rewrite, binaries
and private artifacts. No binary release is needed: application source, selected
configuration and dependencies are unchanged. The reserved seal remains
untouched; no prospective freeze was consumed. Narrow/enlarged-text native GUI,
accessibility and real UNC-share acceptance remain explicitly open in
`TODO.md`, not silently passed. The GUI remains closed.

## October 12 close-out pass

Requested by the owner on 2026-10-12. Nothing here reopens research, tunes
anything or changes the selected configuration.

- **Kill-test and reopening bar.** Recorded above.
- **Prospective check.** Ordinary commands cannot score dates after the
  development cutoff, so the forward score uses the once-only prospective
  workflow. On the final build, selected v20 against the previous release
  (`config/baselines/previous-release-790ec2d.toml`) reproduced 1,150 guesses
  over the 360 development games (3.1944, all solved) versus 1,199 (3.3306):
  paired difference -0.1361, 95% interval [-0.1834, -0.0861], win/tie/loss
  66/276/18. That met the eligibility rule, and unchanged v20 was frozen at
  2026-10-12T04:17Z for 2026-10-13 through 2026-11-11
  ([freeze record](../../../benchmarks/predictive/prospective-frozen-v1.json)).
  It can be evaluated once, on or after 2026-11-12 UTC, with the same
  executable. Until then any change to `src/`, `tests/`, `Cargo.toml`,
  `Cargo.lock`, `config/evaluation.toml`, `data/seed/` or the packaged CLI
  makes the evaluation refuse to run. The comparison is a retrospective
  eligibility check, not new evidence; its per-game file contains answer words
  and stays local.
- **Native GUI acceptance.** Through UI Automation on the packaged Windows
  build: all 38 input controls expose a name (nine did not before the fixes),
  the five feedback tiles have distinct names, and no control leaves the window
  at 1180 by 820 or at the 560 by 620 minimum, at 100% and 135% text. Keyboard
  use was checked on the build before the fixes; it found that focus was lost
  after Enter, which is now fixed. The keyboard pass and screenshots could not
  be repeated on the final build because the desktop session was locked. See
  [EXPERIMENTS.md](../../EXPERIMENTS.md#october-12-native-acceptance).
- **Network-path persistence.** The packaged CLI built the model, wrote and
  reused the pattern table and persisted a manual addition with its working
  directory on a loopback SMB path, leaving no temporary files. A share on
  another machine is untested. See [PERSISTENCE.md](../../PERSISTENCE.md).
- **Dependencies.** The audit passes with the same two exceptions; a fix path
  for one of them now exists. See [DEPENDENCIES.md](../../DEPENDENCIES.md).
- **Dead-code removal.** An over-engineering audit removed a net 2,200 lines of
  unused or duplicated code, mostly retired research scaffolding
  (the Bellman dataset builder, the survival artifact layer and an unused
  policy view). Two visible changes: the GUI loading status no longer claims
  "no predictive artifact available" before the answer is known, and an
  invalid date in `evaluate-live-config`, `three-guess-gap` and
  `four-guess-openers` is now reported with the same wording as other
  commands. Formatting, warning-denied Clippy, 591 tests and 15 benchmark
  smokes pass. The count is lower than October 5 because tests of deleted code
  went with it. The final build reproduces the published September audit
  outcome for selected v20 on all 360 development games (status, guess count
  and path length).
- **Repository hygiene.** One duplicate and one superseded benchmark artifact
  were deleted. The README was split into a short README and
  [EXPERIMENTS.md](../../EXPERIMENTS.md), with the generated evidence blocks
  unchanged.

## October 12 follow-up and re-freeze

Later on 2026-10-12 the owner asked for further source and dependency
changes. Any such change voids a prospective freeze, so the 04:17Z freeze
above was replaced before its window opened. No window data existed yet, and
the configuration, the comparison baseline and the window are the same, so
this is the same test made on a different build. Where this section differs
from the close-out pass above, this section is current.

- **Typed CLI arguments.** Dates are parsed by the argument parser, and solver
  mode, merge strategy, weight mode, model variant, study stage and study
  strategy are enumerated values. Every spelling accepted before still works
  (`_` or `-`, any letter case, the documented aliases). An invalid value is
  now rejected by the parser with its own wording and exit code 2 instead
  of 1.
- **Two experiment commands removed.** `learned-proxy-experiment` and
  `survival-experiment` are gone together with the code only they used. Both
  experiments had closed as not promoted; their result files and write-ups
  stay, and the code last exists in commit `7539f30`. The other research
  commands are kept because they reproduce the retained evidence. After all
  of this section's changes `src/` is a net 6,654 lines smaller.
- **Dependencies.** The three open Dependabot updates (`rayon` 1.12.0, `serde`
  1.0.229, `toml` 1.0.6) and `eframe` 0.36.2 are in, and `quinn-proto`, which
  is locked but never compiled, moved to 0.11.19 after five advisories against
  it were published during the day. The audit passes with one exception
  instead of two. See [DEPENDENCIES.md](../../DEPENDENCIES.md).
- **Native GUI acceptance.** Repeated on the final build with the desktop
  unlocked, including the keyboard pass that was missing above: 38 named
  controls, five distinct tile names, no control outside the window in the
  four layouts, and focus back in the Guess field after Enter. The new
  renderer and text stack showed no visible difference from the morning build
  beyond default widget styling. See
  [EXPERIMENTS.md](../../EXPERIMENTS.md#october-12-native-acceptance).
- **Gates.** Formatting, warning-denied Clippy, 535 tests and 15 benchmark
  smokes pass on the final source; the count fell from 591 because tests of
  the removed code went with it. One characterization test failed once on a
  Windows file-replace error while the machine was busy with other work and
  passed on the rerun.
- **Re-freeze.** On the final CLI (SHA-256 `858956b9c06c8b8a...`), selected
  v20 against the previous release again gave 1,150 guesses over the 360
  development games versus 1,199, with the same paired difference and
  interval, and matched the published September audit outcome on every date.
  The configuration fingerprint is unchanged. Unchanged v20 was frozen again
  at 2026-10-12T22:26Z for 2026-10-13 through 2026-11-11; the
  [freeze record](../../../benchmarks/predictive/prospective-frozen-v1.json)
  at the same path replaces the earlier one. The rules are as stated above:
  one evaluation, on or after 2026-11-12 UTC, with that executable and with
  `src/`, `tests/`, `Cargo.toml`, `Cargo.lock`, `config/evaluation.toml` and
  `data/seed/` untouched until then.

Still open:

- Evaluate the prospective freeze once, on or after 2026-11-12.
- Run a screen reader (Narrator or NVDA) over the GUI.
- Test persistence on a share hosted by another machine.
- Reassess the remaining dependency exception on the next GUI or lockfile
  update and no later than 2027-02-01.

`TODO.md` is the owner's local working list and is no longer published; older
mentions of it in this repository refer to that file.

On 2026-10-12 the directory `target/subthree-research` named under
"Evidence and recovery" was not present on the owner's machine, so the private
close-out evidence and its hashes could not be rechecked. The owner believes it
was deleted by accident. It was not in the Recycle Bin and no moved copy was
found under the owner's projects folder, so treat the private narrative, the
four source snapshots, the old TODO, `research-working-tree.patch` and
`preservation.json` as lost; the recovery steps above can no longer be
followed. This ledger and the committed history are the surviving record. The
kill-test evidence is under `target/oracle-membership-killtest/`.
