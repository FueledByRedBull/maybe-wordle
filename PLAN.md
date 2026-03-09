# Maybe Wordle Implementation Plan

## Goal

Build a Wordle solver that remains strong after the New York Times started reusing past answers in February 2026.

The solver should:

- accept standard Wordle feedback for each guess
- rank next guesses using a weighted probability model instead of assuming all remaining answers are equally likely
- backtest itself against historical NYT answers
- make its assumptions explicit so "optimal" always means "optimal for the model we chose", not "perfectly predicts the editor"

## Product Decision

Version 1 should be a Rust CLI, not a GUI.

Why:

- the hard part is the data model, solver core, and backtesting
- a GUI adds surface area before the scoring model is validated
- a CLI makes it easy to benchmark, regression-test, and later expose the same engine to `egui`, Tauri, web, or Python

GUI work is Phase 5, after the solver is correct and fast.

## What Changes From The Original Draft

These are deliberate replacements, not minor edits:

- Do not assume there is a public, official, current NYT answer list. There is not.
- Do not hardcode a strict "~2300 answer list" as the full truth for 2026 and beyond.
- Do not claim the engine is mathematically proven to be globally optimal for modern NYT Wordle. That claim only makes sense once the answer universe and prior are fixed.
- Do not start with `egui`, SIMD, or bit-packing. First win on data quality, algorithm choice, cache-friendly layout, and benchmarks.

## Research Summary

### 1. Historical played answers can be fetched directly from NYT

Primary source:

- `https://www.nytimes.com/svc/wordle/v2/YYYY-MM-DD.json`

Verified examples:

- `https://www.nytimes.com/svc/wordle/v2/2021-06-19.json` returns puzzle 1 with solution `cigar`
- `https://www.nytimes.com/svc/wordle/v2/2026-03-08.json` returns solution `lobby`

This endpoint is the best source for "already played games" because it is direct, date-addressable, and current.

### 2. Accepted guess words are available from community mirrors of the original game data

Practical seed source:

- [tabatkins/wordle-list](https://github.com/tabatkins/wordle-list)

Verified locally during research:

- the raw list currently contains `14855` accepted words

This is suitable for the guess space, but it is not an official NYT artifact.

### 3. A current official NYT answer list is not publicly maintained

That means the solver cannot honestly rely on a single "true" modern solution list. Instead, it needs a modeled answer universe built from:

- a seed candidate-answer list from well-known Wordle mirrors
- all historical NYT answers fetched from the daily endpoint
- optional manual additions if NYT introduces answers outside the seed set

### 4. Exact-search results from the classic Wordle setting are still useful

Useful references:

- [Bertsimas and Paskov, An Exact and Interpretable Solution to Wordle](https://auction-upload-files.s3.amazonaws.com/Wordle_Paper_Final.pdf)
- [Lokshtanov and Subercaseaux, Wordle Is NP-Hard, But We Can Still Win in 3.421 Guesses](https://www.ijcai.org/proceedings/2022/783)

These references are useful for:

- exact-search structure
- pruning strategies
- baseline performance targets
- sanity-checking starter words under a fixed, uniform prior

They are not proof that the modern NYT-weighted problem has the same optimum.

## Explicit Modeling Assumptions

The solver needs three sets, not two:

### `G`: Guess space

All accepted guesses. Initial source: `tabatkins/wordle-list`.

### `A_seed`: Seed answer universe

A curated candidate-answer list bootstrapped from one or more community-maintained solution lists derived from the original Wordle data and later NYT edits.

Initial implementation rule:

- vendor one pinned seed list into the repo
- record its source URL and commit hash in a metadata file
- never fetch it at runtime

### `H`: Historical NYT answers

All answers actually observed from the NYT daily endpoint.

### `A_model`: Modeled answer universe

`A_model = A_seed U H U manual_additions`

This is the actual universe the solver reasons over.

This is better than the original strict two-tier design because the modern NYT answer policy is curated and mutable. A rigid "~2300 answers only" approach will eventually break.

## Data Sources And Repo Layout

Use this layout from the start:

```text
data/
  raw/
    nyt_daily_answers.jsonl
  seed/
    valid_guesses.txt
    candidate_answers.txt
    sources.toml
  derived/
    answer_history.csv
    modeled_answers.csv
    pattern_table.bin
config/
  prior.toml
```

### `data/raw/nyt_daily_answers.jsonl`

One JSON object per day pulled from the NYT endpoint:

```json
{"print_date":"2026-03-08","solution":"lobby","editor":"Tracy Bennett"}
```

Fetch rule:

- on every sync, start from the last stored `print_date + 1 day`
- re-fetch and compare the most recent 3 stored days on every sync to catch same-day or retroactive NYT edits
- stop at today
- append only new rows
- if a re-fetched day changed, overwrite the stored row and emit a warning
- if a date fails, log it and stop the sync with a non-zero exit code

### `data/seed/valid_guesses.txt`

Primary source:

- [tabatkins/wordle-list](https://github.com/tabatkins/wordle-list)

Rule:

- vendor a pinned snapshot into the repo
- keep source URL and commit hash in `sources.toml`

### `data/seed/candidate_answers.txt`

Source strategy:

- start from one pinned community-maintained candidate-answer list
- compare it against at least one second maintained source during review
- store the final deduplicated result in-repo

Practical starting sources:

- [joshstephenson/Wordle-Solver](https://github.com/joshstephenson/Wordle-Solver) for a classic `nyt-answers.txt` style seed list
- [LaurentLessard/wordlesolver](https://github.com/LaurentLessard/wordlesolver) as a second maintained reference
- [tjstankus gist on NYT list changes](https://gist.github.com/tjstankus/14e255af52a5855a4f44d4d4a5a668ac) as supporting context on post-2022 drift

Do not fetch this list dynamically at runtime.

Reason:

- this list is a modeling choice, not an official API result
- reproducibility matters more than pretending the source is canonical

### `data/derived/answer_history.csv`

Derived from the NYT JSONL archive with columns:

```text
word,first_seen,last_seen,times_seen,days_since_last_seen
```

### `data/derived/modeled_answers.csv`

Derived table used by the solver:

```text
word,in_seed,is_historical,first_seen,last_seen,times_seen,base_weight,recency_weight,final_weight
```

## Weight Model

The original idea of time-decay weights is correct, but it needs to be parameterized and testable.

### Base weight

Start with:

- `1.0` if word is in `A_seed`
- `0.25` if word only exists because it appeared historically in NYT but is not in `A_seed`
- configurable manual multiplier for promoted or demoted words

Rationale:

- seed answers are the main prior
- historically observed out-of-seed answers must stay possible
- they should not dominate the distribution without evidence

### Recency weight

Use a cooldown plus recovery curve:

```text
if word has never been played:
    recency_weight = 1.0
else if days_since_last_seen < cooldown_days:
    recency_weight = cooldown_floor
else:
    recency_weight = cooldown_floor +
        (1.0 - cooldown_floor) /
        (1.0 + exp(-k * (days_since_last_seen - midpoint_days)))
```

Initial defaults:

- `cooldown_days = 180`
- `cooldown_floor = 0.01`
- `midpoint_days = 720`
- `k = 0.01`

These are starting values, not truths. They must be tuned by backtesting.

### Final prior

For each answer `a` in `A_model`:

```text
weight(a) = base_weight(a) * recency_weight(a) * manual_weight(a)
```

At solve time, normalize weights over the currently surviving answer set.

## Solver Core

### Feedback encoding

Implement standard Wordle scoring exactly, including duplicate-letter behavior.

Representation:

- each feedback pattern is one value in `[0, 242]`
- encode as a base-3 integer over 5 positions
- use a two-pass scoring function to match Wordle behavior

Encoding convention:

- `0 = gray`
- `1 = yellow`
- `2 = green`
- `pattern = sum(feedback[i] * 3^i)` for `i in 0..4`

This fits in `u8` because the maximum encoded value is `242`.

This must be unit-tested heavily because repeated letters are the easiest way to build a fast but wrong solver.

### Pattern table

Precompute:

```text
pattern_table[guess_index][answer_index] -> u8
```

Properties:

- one byte per pair
- if `|G| = 14855` and `|A_model| ~= 2315`, memory is about 34 MB
- if `A_model` grows, memory still stays practical on desktop

This table is the main performance win for Version 1.

### Posterior update

After each guess and observed pattern:

1. filter surviving answers in `A_model`
2. recompute normalized weights on the surviving set
3. score all allowed next guesses

### Guess scoring

For a guess `g` and surviving weighted answer set `S`:

```text
bucket_mass[p] = sum(weight(a)) for all a in S where pattern(g, a) = p
total_mass = sum(weight(a)) over S
P(p | g, S) = bucket_mass[p] / total_mass
H(g, S) = -sum(P(p | g, S) * log2(P(p | g, S))) over non-empty buckets
```

Primary ranking metric above the exact-search threshold:

- maximize weighted entropy

Tie-breakers, in order:

1. maximize immediate solve probability
2. minimize expected remaining candidate count
3. prefer guesses inside the surviving answer set
4. prefer lexicographically stable output for reproducibility

### Exact search

Use a hybrid strategy.

### Stage 1: Weighted entropy

When the surviving answer set is large, use weighted entropy.

### Stage 2: Exact expected solve depth

When the surviving answer set is small, switch to exact search with memoization and pruning.

Initial threshold:

- `exact_threshold = 64`

This threshold is intentionally conservative. Benchmark `64`, `96`, and `128` before changing it.

Exact-search objective:

```text
C(S) = min over guesses g of
    1 + sum(P(p | g, S) * C(S_p))
```

Where:

- `S` is the surviving answer set
- `S_p` is the subset consistent with pattern `p`
- probabilities use the weighted prior, not a uniform count

## Performance Targets

Do not add SIMD or bit-packing in Version 1 unless benchmarks demand it.

Target release-build latency on a typical desktop:

- cold load with pattern table generation: acceptable in seconds
- warm load with cached `pattern_table.bin`: less than 1 second
- full next-guess scoring on turn 1: less than 150 ms
- full next-guess scoring on later turns: less than 100 ms
- exact-search decision at threshold 64: less than 2 seconds

Implementation choices:

- Rust
- `rayon` for data-parallel guess scoring
- contiguous arrays and `Vec`-backed storage
- fixed `[f64; 243]` or `[f32; 243]` bucket arrays per worker
- binary cache for the pattern table

Only consider SIMD after:

- the pattern table exists
- release benchmarks exist
- flamegraphs show the bottleneck is still in the inner scoring loop

## Correctness And Validation

### Unit tests

Must cover:

- duplicate-letter feedback cases
- pattern encode/decode round-trips
- posterior filtering correctness
- entropy calculation on hand-checkable toy examples

Required duplicate-letter fixtures:

- target `ALLEY`, guess `LILLY` must score as yellow, gray, green, gray, green
- target `DREAD`, guess `ADDED` must verify that extra guessed `D` instances do not all score as present
- target `BANAL`, guess `LLAMA` must verify mixed gray/yellow behavior when the guess overuses a repeated target letter

### Backtests

Backtest on the historical NYT answer archive generated from the daily endpoint.

Metrics:

- average guesses to solve
- 95th percentile guesses
- max guesses
- fail count under 6 guesses, if any
- calibration metrics for the prior if we compare predicted answer mass against later observed answers

Required experiments:

1. Uniform prior over surviving answers
2. Cooldown-only prior
3. Cooldown plus recovery curve
4. Alternative `A_model` variants with and without out-of-seed historical answers

Acceptance criteria for Version 1:

- no correctness failures in unit tests
- no failed solves on the historical archive used for evaluation
- weighted model performs at least as well as the uniform baseline on post-February-2026 games

## CLI Scope

Version 1 commands:

```text
maybe-wordle sync-data
maybe-wordle build-model
maybe-wordle suggest --guess CRANE --feedback 01020
maybe-wordle solve-interactive
maybe-wordle backtest --from 2026-02-01 --to 2026-03-09
maybe-wordle benchmark
```

Notes:

- feedback encoding should support both numeric trits and a human-friendly form such as `bgybb`
- `solve-interactive` should print top suggestions, expected entropy, solve probability, and surviving candidates

## Implementation Milestones

### Phase 0: Data ingestion

- vendor `valid_guesses.txt`
- vendor `candidate_answers.txt`
- implement NYT history sync
- generate `answer_history.csv`
- build `modeled_answers.csv`

### Phase 1: Correct solver core

- implement Wordle feedback scoring
- add unit tests for repeated letters
- implement posterior filtering
- implement weighted entropy scoring

### Phase 2: Performance layer

- precompute and cache `pattern_table.bin`
- parallelize guess scoring with `rayon`
- add benchmarks

### Phase 3: Exact search

- implement memoized exact search below threshold
- add branch-and-bound pruning
- benchmark thresholds

### Phase 4: Evaluation and tuning

- run backtests
- tune prior parameters
- compare against uniform baseline

### Phase 5: Optional UI

- add `egui` only after the CLI model is validated

## Risks

### Risk: seed answer list drift

Mitigation:

- use `A_model = A_seed U H U manual_additions`
- treat the seed list as a prior, not absolute truth

### Risk: overfitting the recency curve

Mitigation:

- keep the prior configurable
- backtest against rolling historical windows
- compare against the uniform baseline

### Risk: incorrect duplicate-letter handling

Mitigation:

- unit tests first
- lock down known examples before optimizing

### Risk: premature low-level optimization

Mitigation:

- benchmark before adding SIMD or custom packed encodings

## Source Register

Primary and supporting sources used for this plan:

- NYT daily answer endpoint: `https://www.nytimes.com/svc/wordle/v2/YYYY-MM-DD.json`
- Accepted guesses seed: [tabatkins/wordle-list](https://github.com/tabatkins/wordle-list)
- Candidate-answer seed options: [joshstephenson/Wordle-Solver](https://github.com/joshstephenson/Wordle-Solver), [LaurentLessard/wordlesolver](https://github.com/LaurentLessard/wordlesolver)
- NYT drift context after the original static list era: [tjstankus gist](https://gist.github.com/tjstankus/14e255af52a5855a4f44d4d4a5a668ac)
- Solver theory and exact-search structure: [Bertsimas and Paskov, An Exact and Interpretable Solution to Wordle](https://auction-upload-files.s3.amazonaws.com/Wordle_Paper_Final.pdf)
- Complexity and optimal-play baseline: [Lokshtanov and Subercaseaux, IJCAI 2022](https://www.ijcai.org/proceedings/2022/783)

## Immediate Next Step

Implement Phase 0 first.

If you want me to continue, the next concrete task should be:

1. scaffold the Rust project
2. add the data sync command for the NYT endpoint
3. vendor the pinned guess and candidate-answer lists
4. write the feedback-scoring tests before any solver optimization
