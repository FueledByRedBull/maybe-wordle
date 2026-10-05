# Predictive sub-three research: close-out

Status: stopped at the owner's request on October 5, 2026. The under-three goal
was not achieved; the app goal is paused rather than falsely marked successful.
The owner requested keeping the best verified score/latency balance and closing
the speculative work. No new source screen, fit or gameplay run is queued.

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

## Retained product performance

The existing pooled root bound reduced a matched top-ten request from 55.15 to
3.92 seconds with identical words/costs; a 720-path replay fell from 969.5 to
491.2 seconds with identical paths/scores. Recursive admissible bounds and
highest-positive-mass-first ordering reduced a matched seven-profile slice
from 80.62 to 17.98/17.96 seconds with identical 63 game payloads.

The later complete seven-profile development matrix covered 2,520 profile-games
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
[TODO.md](../../../TODO.md), not silently passed. The GUI remains closed.
