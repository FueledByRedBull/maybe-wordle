use std::{
    array,
    collections::{HashMap, HashSet},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{Days, NaiveDate, Utc};
use rayon::prelude::*;

use crate::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, read_history_jsonl},
    model::{
        AnswerRecord, ModelVariant, WeightMode, load_model, load_model_with_variant,
        weight_snapshot_for_mode,
    },
    pattern_table::PatternTable,
    scoring::{
        ALL_GREEN_PATTERN, PATTERN_SPACE, format_feedback_letters, parse_feedback, score_guess,
    },
    small_state::SmallStateTable,
};

#[derive(Clone, Debug)]
pub struct Suggestion {
    pub word: String,
    pub entropy: f64,
    pub solve_probability: f64,
    pub expected_remaining: f64,
    pub proxy_cost: Option<f64>,
    pub posterior_answer_probability: f64,
    pub lookahead_cost: Option<f64>,
    pub exact_cost: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SolveState {
    pub surviving: Vec<usize>,
    pub weights: Vec<f64>,
    pub total_weight: f64,
}

#[derive(Clone, Debug)]
pub struct SolveStep {
    pub guess: String,
    pub feedback: u8,
}

#[derive(Clone, Debug)]
pub struct SolveRun {
    pub target: String,
    pub date: NaiveDate,
    pub steps: Vec<SolveStep>,
    pub solved: bool,
}

#[derive(Clone, Debug)]
pub struct BacktestStats {
    pub games: usize,
    pub average_guesses: f64,
    pub p95_guesses: usize,
    pub max_guesses: usize,
    pub failures: usize,
    pub coverage_gaps: usize,
}

#[derive(Clone, Debug)]
pub struct ExperimentResult {
    pub config_id: String,
    pub mode: WeightMode,
    pub variant: ModelVariant,
    pub backtest: BacktestStats,
    pub average_log_loss: f64,
    pub average_brier: f64,
    pub average_target_probability: f64,
    pub average_target_rank: f64,
    pub latency_p95_ms: f64,
}

#[derive(Clone, Debug)]
pub struct TuningEvaluation {
    pub config: PriorConfig,
    pub average_guesses: f64,
    pub failures: usize,
    pub coverage_gaps: usize,
    pub average_log_loss: f64,
    pub average_target_rank: f64,
    pub latency_p95_ms: f64,
}

#[derive(Clone, Debug)]
pub struct TunePriorSummary {
    pub search_window_start: NaiveDate,
    pub search_window_end: NaiveDate,
    pub validation_window_start: NaiveDate,
    pub validation_window_end: NaiveDate,
    pub current: TuningEvaluation,
    pub best: TuningEvaluation,
    pub replacement_toml: String,
}

#[derive(Clone, Debug)]
struct PriorSearchEvaluation {
    average_log_loss: f64,
    average_target_rank: f64,
    average_target_probability: f64,
}

pub struct Solver {
    pub config: PriorConfig,
    pub mode: WeightMode,
    pub variant: ModelVariant,
    pub guesses: Vec<String>,
    pub answers: Vec<AnswerRecord>,
    pub history_dates: Vec<NytDailyEntry>,
    exact_small_state_table: SmallStateTable,
    pattern_table: PatternTable,
    guess_index: HashMap<String, usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExactSuggestionMode {
    Exhaustive,
    Pooled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PredictiveSearchMode {
    ProxyOnly,
    Lookahead,
    Exact(ExactSuggestionMode),
}

#[derive(Clone, Copy, Debug)]
struct GuessMetrics {
    guess_index: usize,
    entropy: f64,
    solve_probability: f64,
    expected_remaining: f64,
    proxy_cost: f64,
    posterior_answer_probability: f64,
}

const EXACT_SUBSET_INLINE_CAPACITY: usize = 16;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactSubsetKey(ExactSubsetStorage);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ExactSubsetStorage {
    Inline {
        len: u8,
        ids: [u16; EXACT_SUBSET_INLINE_CAPACITY],
    },
    Heap(Box<[u16]>),
}

impl ExactSubsetKey {
    fn from_sorted_subset(subset: &[usize]) -> Self {
        debug_assert!(subset.windows(2).all(|window| window[0] < window[1]));
        if subset.len() <= EXACT_SUBSET_INLINE_CAPACITY {
            let mut ids = [0u16; EXACT_SUBSET_INLINE_CAPACITY];
            for (slot, value) in ids.iter_mut().zip(subset.iter().copied()) {
                *slot = u16::try_from(value).expect("predictive exact subset index exceeds u16");
            }
            return Self(ExactSubsetStorage::Inline {
                len: subset.len() as u8,
                ids,
            });
        }
        Self(ExactSubsetStorage::Heap(
            subset
                .iter()
                .copied()
                .map(|value| {
                    u16::try_from(value).expect("predictive exact subset index exceeds u16")
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ))
    }
}

struct ExactPartitionFrame {
    masses: [f64; PATTERN_SPACE],
    touched_patterns: Vec<u8>,
    child_subsets: [Vec<usize>; PATTERN_SPACE],
}

impl ExactPartitionFrame {
    fn new() -> Self {
        Self {
            masses: [0.0; PATTERN_SPACE],
            touched_patterns: Vec::with_capacity(PATTERN_SPACE),
            child_subsets: array::from_fn(|_| Vec::new()),
        }
    }

    fn reset(&mut self) {
        for pattern in self.touched_patterns.drain(..) {
            self.masses[pattern as usize] = 0.0;
            self.child_subsets[pattern as usize].clear();
        }
    }
}

struct ExactSearchScratch {
    frames: Vec<ExactPartitionFrame>,
}

impl ExactSearchScratch {
    fn new() -> Self {
        Self { frames: Vec::new() }
    }

    fn frame_mut(&mut self, depth: usize) -> &mut ExactPartitionFrame {
        while self.frames.len() <= depth {
            self.frames.push(ExactPartitionFrame::new());
        }
        let frame = &mut self.frames[depth];
        frame.reset();
        frame
    }
}

struct GuessMetricScratch {
    masses: [f64; PATTERN_SPACE],
    counts: [usize; PATTERN_SPACE],
    weighted_log_sums: [f64; PATTERN_SPACE],
    touched_patterns: Vec<u8>,
}

impl GuessMetricScratch {
    fn new() -> Self {
        Self {
            masses: [0.0; PATTERN_SPACE],
            counts: [0; PATTERN_SPACE],
            weighted_log_sums: [0.0; PATTERN_SPACE],
            touched_patterns: Vec::with_capacity(PATTERN_SPACE),
        }
    }

    fn reset(&mut self) {
        for pattern in self.touched_patterns.drain(..) {
            self.masses[pattern as usize] = 0.0;
            self.counts[pattern as usize] = 0;
            self.weighted_log_sums[pattern as usize] = 0.0;
        }
    }
}

impl Solver {
    pub fn from_paths(paths: &ProjectPaths, config: &PriorConfig) -> Result<Self> {
        Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )
    }

    pub fn from_paths_with_mode(
        paths: &ProjectPaths,
        config: &PriorConfig,
        mode: WeightMode,
    ) -> Result<Self> {
        Self::from_paths_with_settings(paths, config, mode, ModelVariant::SeedPlusHistory)
    }

    pub fn from_paths_with_settings(
        paths: &ProjectPaths,
        config: &PriorConfig,
        mode: WeightMode,
        variant: ModelVariant,
    ) -> Result<Self> {
        let model = if variant == ModelVariant::SeedPlusHistory {
            load_model(paths, config)?
        } else {
            load_model_with_variant(paths, config, variant)?
        };
        let pattern_table = PatternTable::load_or_build(paths, &model.guesses, &model.answers)?;
        let guess_index = model
            .guesses
            .iter()
            .enumerate()
            .map(|(index, guess)| (guess.clone(), index))
            .collect::<HashMap<_, _>>();

        Ok(Self {
            config: config.clone(),
            mode,
            variant: model.variant,
            guesses: model.guesses,
            answers: model.answers,
            history_dates: model.history,
            exact_small_state_table: SmallStateTable::build(
                config.exact_exhaustive_threshold.max(2),
            ),
            pattern_table,
            guess_index,
        })
    }

    pub fn today() -> NaiveDate {
        Utc::now().date_naive()
    }

    pub fn initial_state(&self, as_of: NaiveDate) -> SolveState {
        let mut weights = vec![0.0; self.answers.len()];
        let mut surviving = Vec::new();
        let mut total_weight = 0.0;

        for (index, answer) in self.answers.iter().enumerate() {
            let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
            if snapshot.final_weight > 0.0 {
                weights[index] = snapshot.final_weight;
                total_weight += snapshot.final_weight;
                surviving.push(index);
            }
        }

        SolveState {
            surviving,
            weights,
            total_weight,
        }
    }

    pub fn apply_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
    ) -> Result<SolveState> {
        let mut state = self.initial_state(as_of);
        for (guess, pattern) in observations {
            self.apply_feedback(&mut state, guess, *pattern)?;
        }
        Ok(state)
    }

    pub fn apply_feedback(&self, state: &mut SolveState, guess: &str, pattern: u8) -> Result<()> {
        let guess_index = self
            .guess_index
            .get(&guess.to_ascii_lowercase())
            .copied()
            .ok_or_else(|| anyhow!("unknown guess: {}", guess))?;
        state
            .surviving
            .retain(|answer_index| self.pattern_table.get(guess_index, *answer_index) == pattern);
        state.total_weight = state
            .surviving
            .iter()
            .map(|index| state.weights[*index])
            .sum::<f64>();

        if state.surviving.is_empty() {
            bail!(
                "no answers remain after applying {} {}",
                guess,
                format_feedback_letters(pattern)
            );
        }
        Ok(())
    }

    pub fn suggestions(&self, state: &SolveState, top: usize) -> Result<Vec<Suggestion>> {
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        let search_mode = predictive_search_mode(&self.config, state.surviving.len());
        let surviving_words = state
            .surviving
            .iter()
            .map(|index| self.answers[*index].word.as_str())
            .collect::<HashSet<_>>();
        let metrics = self.score_guess_metrics_for_subset(
            &state.surviving,
            &state.weights,
            &self.exact_small_state_table,
        );
        let mut suggestions = metrics
            .into_iter()
            .map(|metric| Suggestion {
                word: self.guesses[metric.guess_index].clone(),
                entropy: metric.entropy,
                solve_probability: metric.solve_probability,
                expected_remaining: metric.expected_remaining,
                proxy_cost: Some(metric.proxy_cost),
                posterior_answer_probability: metric.posterior_answer_probability,
                lookahead_cost: None,
                exact_cost: None,
            })
            .collect::<Vec<_>>();

        suggestions.sort_by(|left, right| compare_suggestions(left, right, &surviving_words));

        if let PredictiveSearchMode::Lookahead = search_mode {
            let root_candidates = self.collect_lookahead_candidates(&suggestions)?;
            let mut exact_memo = HashMap::new();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut lookahead_memo = HashMap::new();
            let mut lookahead_costs = vec![None; self.guesses.len()];

            for guess_index in root_candidates {
                let cost = self.lookahead_cost_for_guess(
                    guess_index,
                    &state.surviving,
                    &state.weights,
                    &mut exact_memo,
                    &mut exact_scratch,
                    &mut lookahead_memo,
                )?;
                lookahead_costs[guess_index] = Some(cost);
            }

            for suggestion in &mut suggestions {
                suggestion.lookahead_cost = self
                    .guess_index
                    .get(&suggestion.word)
                    .and_then(|guess_index| lookahead_costs[*guess_index]);
            }
            suggestions.sort_by(|left, right| compare_lookahead(left, right, &surviving_words));
        }

        if let PredictiveSearchMode::Exact(exact_mode) = search_mode {
            let exact_candidates = match exact_mode {
                ExactSuggestionMode::Exhaustive => (0..self.guesses.len()).collect::<Vec<_>>(),
                ExactSuggestionMode::Pooled => {
                    self.collect_exact_candidates(state, &suggestions)?
                }
            };
            let mut memo = HashMap::new();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut exact_costs = vec![None; self.guesses.len()];

            for guess_index in exact_candidates {
                let cost = self.exact_cost_for_guess(
                    guess_index,
                    &state.surviving,
                    &state.weights,
                    &self.exact_small_state_table,
                    &mut memo,
                    f64::INFINITY,
                    &mut exact_scratch,
                    0,
                )?;
                exact_costs[guess_index] = Some(cost);
            }

            match exact_mode {
                ExactSuggestionMode::Exhaustive => {
                    for suggestion in &mut suggestions {
                        suggestion.exact_cost = self
                            .guess_index
                            .get(&suggestion.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
                    }
                    suggestions.sort_by(|left, right| compare_exact(left, right, &surviving_words));
                }
                ExactSuggestionMode::Pooled => {
                    suggestions.sort_by(|left, right| {
                        let left_cost = self
                            .guess_index
                            .get(&left.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
                        let right_cost = self
                            .guess_index
                            .get(&right.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
                        compare_exact_costs(left, right, left_cost, right_cost, &surviving_words)
                    });
                }
            }
        }

        suggestions.truncate(top);
        Ok(suggestions)
    }

    pub fn solve_target(&self, target: &str, date: NaiveDate, top: usize) -> Result<SolveRun> {
        let target = target.to_ascii_lowercase();
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let mut state = self.initial_state(as_of);

        if !state
            .surviving
            .iter()
            .any(|index| self.answers[*index].word == target)
        {
            return Ok(SolveRun {
                target,
                date,
                steps: Vec::new(),
                solved: false,
            });
        }

        let mut steps = Vec::new();
        while steps.len() < 6 {
            let suggestions = self.suggestions(&state, top.max(1))?;
            let guess = suggestions
                .first()
                .ok_or_else(|| anyhow!("solver returned no suggestions"))?
                .word
                .clone();
            let feedback = score_guess(&guess, &target);
            steps.push(SolveStep {
                guess: guess.clone(),
                feedback,
            });
            if feedback == ALL_GREEN_PATTERN {
                return Ok(SolveRun {
                    target,
                    date,
                    steps,
                    solved: true,
                });
            }
            self.apply_feedback(&mut state, &guess, feedback)?;
        }

        Ok(SolveRun {
            target,
            date,
            steps,
            solved: false,
        })
    }

    pub fn backtest(&self, from: NaiveDate, to: NaiveDate, top: usize) -> Result<BacktestStats> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested backtest range");
        }

        let mut guess_counts = Vec::new();
        let mut failures = 0usize;
        let mut coverage_gaps = 0usize;

        for entry in games {
            let run = self.solve_target(&entry.solution, entry.print_date, top)?;
            if run.steps.is_empty() {
                coverage_gaps += 1;
                failures += 1;
                continue;
            }
            if !run.solved {
                failures += 1;
            }
            guess_counts.push(run.steps.len());
        }

        guess_counts.sort_unstable();
        let games_played = guess_counts.len() + coverage_gaps;
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        let p95_guesses = guess_counts
            .get(p95_index.saturating_sub(1))
            .copied()
            .unwrap_or_default();
        let max_guesses = guess_counts.last().copied().unwrap_or_default();

        Ok(BacktestStats {
            games: games_played,
            average_guesses,
            p95_guesses,
            max_guesses,
            failures,
            coverage_gaps,
        })
    }

    pub fn experiment_report(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<ExperimentResult> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let backtest = self.backtest(from, to, top)?;
        let mut total_log_loss = 0.0;
        let mut total_brier = 0.0;
        let mut total_target_probability = 0.0;
        let mut total_rank = 0.0;
        let mut measured = 0usize;

        for entry in games {
            if let Some(metrics) = self.initial_prior_metrics(&entry.solution, entry.print_date) {
                total_log_loss += metrics.log_loss;
                total_brier += metrics.brier;
                total_target_probability += metrics.target_probability;
                total_rank += metrics.target_rank as f64;
                measured += 1;
            }
        }

        let divisor = measured.max(1) as f64;
        Ok(ExperimentResult {
            config_id: format!(
                "et{}-ee{}-cp{}-lt{}-lc{}-lr{}",
                self.config.exact_threshold,
                self.config.exact_exhaustive_threshold,
                self.config.exact_candidate_pool,
                self.config.lookahead_threshold,
                self.config.lookahead_candidate_pool,
                self.config.lookahead_reply_pool
            ),
            mode: self.mode,
            variant: self.variant,
            backtest,
            average_log_loss: total_log_loss / divisor,
            average_brier: total_brier / divisor,
            average_target_probability: total_target_probability / divisor,
            average_target_rank: total_rank / divisor,
            latency_p95_ms: self.benchmark_predictive_latency(5)?,
        })
    }

    pub fn parse_observations(
        guesses: &[String],
        feedbacks: &[String],
    ) -> Result<Vec<(String, u8)>> {
        if guesses.len() != feedbacks.len() {
            bail!("--guess and --feedback must appear the same number of times");
        }

        guesses
            .iter()
            .zip(feedbacks)
            .map(|(guess, feedback)| {
                Ok((guess.trim().to_ascii_lowercase(), parse_feedback(feedback)?))
            })
            .collect()
    }

    pub fn latest_history_range(paths: &ProjectPaths) -> Result<Option<(NaiveDate, NaiveDate)>> {
        let history = read_history_jsonl(&paths.raw_history)?;
        Ok(history
            .first()
            .zip(history.last())
            .map(|(first, last)| (first.print_date, last.print_date)))
    }

    pub fn pattern_table_bytes(&self) -> usize {
        self.pattern_table.bytes_len()
    }

    pub fn tune_prior(paths: &ProjectPaths, config: &PriorConfig) -> Result<TunePriorSummary> {
        let (history_start, history_end) = Self::latest_history_range(paths)?
            .ok_or_else(|| anyhow!("run sync-data before tune-prior"))?;
        let window_end = history_end;
        let window_start = history_end
            .checked_sub_days(Days::new(364))
            .map_or(history_start, |date| date.max(history_start));
        let mut best_prior_config = config.clone();
        let mut best_prior = Self::evaluate_prior_search_candidate(
            paths,
            &best_prior_config,
            window_start,
            window_end,
        )?;

        macro_rules! search_dimension {
            ($field:ident, $values:expr) => {{
                loop {
                    let mut improved = false;
                    for value in $values {
                        if best_prior_config.$field == value {
                            continue;
                        }
                        let mut candidate_config = best_prior_config.clone();
                        candidate_config.$field = value;
                        let candidate = Self::evaluate_prior_search_candidate(
                            paths,
                            &candidate_config,
                            window_start,
                            window_end,
                        )?;
                        if Self::better_prior_search_evaluation(&candidate, &best_prior) {
                            best_prior_config = candidate_config;
                            best_prior = candidate;
                            improved = true;
                        }
                    }
                    if !improved {
                        break;
                    }
                }
            }};
        }

        search_dimension!(base_seed_weight, [0.75, 1.0, 1.25]);
        search_dimension!(base_history_only_weight, [0.10, 0.20, 0.25, 0.33, 0.50]);
        search_dimension!(cooldown_days, [90_i64, 120, 180, 240, 365]);
        search_dimension!(cooldown_floor, [0.0, 0.01, 0.02, 0.05]);
        search_dimension!(midpoint_days, [365.0, 540.0, 720.0, 900.0, 1080.0]);
        search_dimension!(logistic_k, [0.005, 0.01, 0.015, 0.02]);

        let validation_start = window_end
            .checked_sub_days(Days::new(6))
            .map_or(window_start, |date| date.max(window_start));
        let current = Self::evaluate_tuning_candidate(paths, config, validation_start, window_end)?;
        let best = Self::evaluate_tuning_candidate(
            paths,
            &best_prior_config,
            validation_start,
            window_end,
        )?;

        let replacement_toml = format!(
            concat!(
                "base_seed_weight = {:.2}\n",
                "base_history_only_weight = {:.2}\n",
                "cooldown_days = {}\n",
                "cooldown_floor = {:.2}\n",
                "midpoint_days = {:.1}\n",
                "logistic_k = {:.3}\n",
                "exact_threshold = {}\n",
                "exact_exhaustive_threshold = {}\n",
                "exact_candidate_pool = {}\n",
                "lookahead_threshold = {}\n",
                "lookahead_candidate_pool = {}\n",
                "lookahead_reply_pool = {}\n",
            ),
            best.config.base_seed_weight,
            best.config.base_history_only_weight,
            best.config.cooldown_days,
            best.config.cooldown_floor,
            best.config.midpoint_days,
            best.config.logistic_k,
            best.config.exact_threshold,
            best.config.exact_exhaustive_threshold,
            best.config.exact_candidate_pool,
            best.config.lookahead_threshold,
            best.config.lookahead_candidate_pool,
            best.config.lookahead_reply_pool,
        );

        Ok(TunePriorSummary {
            search_window_start: window_start,
            search_window_end: window_end,
            validation_window_start: validation_start,
            validation_window_end: window_end,
            current,
            best,
            replacement_toml,
        })
    }

    fn initial_prior_metrics(&self, target: &str, date: NaiveDate) -> Option<PriorMetrics> {
        let as_of = date.checked_sub_days(Days::new(1))?;
        let state = self.initial_state(as_of);
        let target = target.to_ascii_lowercase();
        let target_index = state
            .surviving
            .iter()
            .find(|index| self.answers[**index].word == target)
            .copied()?;

        let target_probability = state.weights[target_index] / state.total_weight;
        let mut ordered = state
            .surviving
            .iter()
            .map(|index| (*index, state.weights[*index] / state.total_weight))
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| right.1.total_cmp(&left.1));
        let target_rank = ordered
            .iter()
            .position(|(index, _)| *index == target_index)
            .map(|rank| rank + 1)?;
        let probability_square_sum = ordered
            .iter()
            .map(|(_, probability)| probability * probability)
            .sum::<f64>();

        Some(PriorMetrics {
            target_probability,
            target_rank,
            log_loss: -(target_probability.max(1e-12)).ln(),
            brier: probability_square_sum - (2.0 * target_probability) + 1.0,
        })
    }

    fn benchmark_predictive_latency(&self, runs: usize) -> Result<f64> {
        let run_count = runs.max(1);
        let state = self.initial_state(Self::today());
        let mut samples = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            let start = Instant::now();
            let _ = self.suggestions(&state, 10)?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|left, right| left.total_cmp(right));
        let p95_index = ((samples.len() as f64) * 0.95).ceil() as usize;
        Ok(samples[p95_index.saturating_sub(1)].max(0.0))
    }

    fn evaluate_tuning_candidate(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<TuningEvaluation> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let report = solver.experiment_report(from, to, 5)?;
        Ok(TuningEvaluation {
            config: config.clone(),
            average_guesses: report.backtest.average_guesses,
            failures: report.backtest.failures,
            coverage_gaps: report.backtest.coverage_gaps,
            average_log_loss: report.average_log_loss,
            average_target_rank: report.average_target_rank,
            latency_p95_ms: report.latency_p95_ms,
        })
    }

    fn evaluate_prior_search_candidate(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<PriorSearchEvaluation> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let games = solver
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let mut total_log_loss = 0.0;
        let mut total_target_rank = 0.0;
        let mut total_target_probability = 0.0;
        let mut measured = 0usize;
        for entry in games {
            if let Some(metrics) = solver.initial_prior_metrics(&entry.solution, entry.print_date) {
                total_log_loss += metrics.log_loss;
                total_target_rank += metrics.target_rank as f64;
                total_target_probability += metrics.target_probability;
                measured += 1;
            }
        }

        let divisor = measured.max(1) as f64;
        Ok(PriorSearchEvaluation {
            average_log_loss: total_log_loss / divisor,
            average_target_rank: total_target_rank / divisor,
            average_target_probability: total_target_probability / divisor,
        })
    }

    fn better_prior_search_evaluation(
        candidate: &PriorSearchEvaluation,
        incumbent: &PriorSearchEvaluation,
    ) -> bool {
        candidate
            .average_log_loss
            .total_cmp(&incumbent.average_log_loss)
            .then_with(|| {
                candidate
                    .average_target_rank
                    .total_cmp(&incumbent.average_target_rank)
            })
            .then_with(|| {
                incumbent
                    .average_target_probability
                    .total_cmp(&candidate.average_target_probability)
            })
            == std::cmp::Ordering::Less
    }

    fn score_guess_metrics_for_subset(
        &self,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
    ) -> Vec<GuessMetrics> {
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut posterior_answer_probability = vec![0.0; self.guesses.len()];
        if total_weight > 0.0 {
            for answer_index in subset {
                if let Some(guess_index) = self.guess_index.get(&self.answers[*answer_index].word) {
                    posterior_answer_probability[*guess_index] =
                        weights[*answer_index] / total_weight;
                }
            }
        }

        (0..self.guesses.len())
            .into_par_iter()
            .map_init(GuessMetricScratch::new, |scratch, guess_index| {
                self.score_guess_metrics(
                    guess_index,
                    subset,
                    weights,
                    total_weight,
                    small_state_table,
                    posterior_answer_probability[guess_index],
                    scratch,
                )
            })
            .collect()
    }

    fn score_guess_metrics(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        total_weight: f64,
        small_state_table: &SmallStateTable,
        posterior_answer_probability: f64,
        scratch: &mut GuessMetricScratch,
    ) -> GuessMetrics {
        scratch.reset();
        for answer_index in subset {
            let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
            if scratch.counts[pattern] == 0 {
                scratch.touched_patterns.push(pattern as u8);
            }
            let weight = weights[*answer_index];
            scratch.masses[pattern] += weight;
            scratch.counts[pattern] += 1;
            if weight > 0.0 {
                scratch.weighted_log_sums[pattern] += weight * weight.log2();
            }
        }

        let pattern_space_log = (PATTERN_SPACE as f64).log2();
        let mut entropy = 0.0;
        let mut expected_remaining = 0.0;
        let mut solve_probability = 0.0;
        let mut proxy_cost = 1.0;

        for pattern in scratch.touched_patterns.iter().copied() {
            let index = pattern as usize;
            let mass = scratch.masses[index];
            let probability = if total_weight > 0.0 {
                mass / total_weight
            } else {
                0.0
            };
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
            expected_remaining += probability * scratch.counts[index] as f64;
            if pattern == ALL_GREEN_PATTERN {
                solve_probability = probability;
            }
            let child_proxy = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else if scratch.counts[index] == 1 {
                1.0
            } else if scratch.counts[index] <= self.config.exact_exhaustive_threshold {
                small_state_table.lower_bound(scratch.counts[index])
            } else {
                let expected_remaining_floor =
                    (scratch.counts[index] as f64 / PATTERN_SPACE as f64).max(1.0);
                let entropy_bits = if mass > 0.0 {
                    mass.log2() - (scratch.weighted_log_sums[index] / mass)
                } else {
                    0.0
                };
                let entropy_floor = (entropy_bits / pattern_space_log).max(1.0);
                expected_remaining_floor.max(entropy_floor)
            };
            proxy_cost += probability * child_proxy;
        }

        GuessMetrics {
            guess_index,
            entropy,
            solve_probability,
            expected_remaining,
            proxy_cost,
            posterior_answer_probability,
        }
    }

    fn collect_exact_candidates(
        &self,
        state: &SolveState,
        suggestions: &[Suggestion],
    ) -> Result<Vec<usize>> {
        let pool = self.config.exact_candidate_pool.max(1);
        let surviving_guess_indexes = state
            .surviving
            .iter()
            .filter_map(|answer_index| self.guess_index.get(&self.answers[*answer_index].word))
            .copied()
            .collect::<HashSet<_>>();
        let mut candidate_indexes = Vec::new();
        let mut seen = HashSet::new();
        let mut push_candidate = |guess_index: usize| {
            if seen.insert(guess_index) {
                candidate_indexes.push(guess_index);
            }
        };

        for suggestion in suggestions.iter().take(pool / 2) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_entropy = suggestions.iter().collect::<Vec<_>>();
        by_entropy.sort_by(|left, right| {
            right
                .entropy
                .total_cmp(&left.entropy)
                .then_with(|| left.word.cmp(&right.word))
        });
        for suggestion in by_entropy.into_iter().take(pool / 4) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_solve_prob = suggestions.iter().collect::<Vec<_>>();
        by_solve_prob.sort_by(|left, right| {
            right
                .solve_probability
                .total_cmp(&left.solve_probability)
                .then_with(|| left.word.cmp(&right.word))
        });
        for suggestion in by_solve_prob.into_iter().take(pool / 8) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_posterior = suggestions
            .iter()
            .filter(|suggestion| suggestion.posterior_answer_probability > 0.0)
            .collect::<Vec<_>>();
        by_posterior.sort_by(|left, right| {
            right
                .posterior_answer_probability
                .total_cmp(&left.posterior_answer_probability)
                .then_with(|| left.word.cmp(&right.word))
        });
        for suggestion in by_posterior.into_iter().take(pool / 8) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        for answer_index in &state.surviving {
            if let Some(guess_index) = self.guess_index.get(&self.answers[*answer_index].word) {
                push_candidate(*guess_index);
            }
        }

        let extra_limit = self.config.exact_candidate_pool + surviving_guess_indexes.len();
        if candidate_indexes.len() > extra_limit {
            let mut trimmed = Vec::with_capacity(extra_limit);
            let mut extra_count = 0usize;
            for guess_index in candidate_indexes {
                if surviving_guess_indexes.contains(&guess_index) {
                    trimmed.push(guess_index);
                    continue;
                }
                if extra_count < self.config.exact_candidate_pool {
                    trimmed.push(guess_index);
                    extra_count += 1;
                }
            }
            return Ok(trimmed);
        }

        Ok(candidate_indexes)
    }

    fn lookahead_cost_for_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        exact_memo: &mut HashMap<ExactSubsetKey, f64>,
        exact_scratch: &mut ExactSearchScratch,
        lookahead_memo: &mut HashMap<ExactSubsetKey, f64>,
    ) -> Result<f64> {
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        if total_weight <= 0.0 {
            bail!("cannot evaluate lookahead on a zero-mass subset");
        }

        let mut ordered_patterns = [0u8; PATTERN_SPACE];
        let ordered_len = {
            let frame = exact_scratch.frame_mut(0);
            for answer_index in subset {
                let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
                if frame.child_subsets[pattern].is_empty() {
                    frame.touched_patterns.push(pattern as u8);
                }
                frame.masses[pattern] += weights[*answer_index];
                frame.child_subsets[pattern].push(*answer_index);
            }
            let len = frame.touched_patterns.len();
            ordered_patterns[..len].copy_from_slice(&frame.touched_patterns);
            len
        };

        let mut total_cost = 1.0;
        for pattern in ordered_patterns[..ordered_len].iter().copied() {
            let mass = exact_scratch.frames[0].masses[pattern as usize];
            let probability = mass / total_weight;
            let child_value = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else {
                let child_subset =
                    std::mem::take(&mut exact_scratch.frames[0].child_subsets[pattern as usize]);
                let result =
                    if child_subset.len() == subset.len() && child_subset.as_slice() == subset {
                        f64::INFINITY
                    } else {
                        self.lookahead_child_value(
                            &child_subset,
                            weights,
                            exact_memo,
                            exact_scratch,
                            lookahead_memo,
                        )?
                    };
                exact_scratch.frames[0].child_subsets[pattern as usize] = child_subset;
                result
            };
            total_cost += probability * child_value;
        }
        Ok(total_cost)
    }

    fn lookahead_child_value(
        &self,
        subset: &[usize],
        weights: &[f64],
        exact_memo: &mut HashMap<ExactSubsetKey, f64>,
        exact_scratch: &mut ExactSearchScratch,
        lookahead_memo: &mut HashMap<ExactSubsetKey, f64>,
    ) -> Result<f64> {
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() <= self.config.exact_threshold {
            return self.exact_best_cost(
                subset,
                weights,
                &self.exact_small_state_table,
                exact_memo,
                exact_scratch,
                1,
            );
        }

        let key = ExactSubsetKey::from_sorted_subset(subset);
        if let Some(cached) = lookahead_memo.get(&key) {
            return Ok(*cached);
        }

        let mut metrics =
            self.score_guess_metrics_for_subset(subset, weights, &self.exact_small_state_table);
        metrics.sort_by(|left, right| compare_guess_metrics(left, right, &self.guesses));
        let best_reply = metrics
            .into_iter()
            .take(self.config.lookahead_reply_pool.max(1))
            .map(|metric| metric.proxy_cost)
            .fold(f64::INFINITY, f64::min);
        let child_value = 1.0 + best_reply;
        lookahead_memo.insert(key, child_value);
        Ok(child_value)
    }

    fn collect_lookahead_candidates(&self, suggestions: &[Suggestion]) -> Result<Vec<usize>> {
        suggestions
            .iter()
            .take(self.config.lookahead_candidate_pool.max(1))
            .map(|suggestion| {
                self.guess_index
                    .get(&suggestion.word)
                    .copied()
                    .with_context(|| format!("missing guess {}", suggestion.word))
            })
            .collect()
    }

    fn exact_cost_for_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
        memo: &mut HashMap<ExactSubsetKey, f64>,
        best_bound: f64,
        scratch: &mut ExactSearchScratch,
        depth: usize,
    ) -> Result<f64> {
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() == 1 && self.guesses[guess_index] == self.answers[subset[0]].word {
            return Ok(1.0);
        }

        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut ordered_patterns = [0u8; PATTERN_SPACE];
        let ordered_len = {
            let frame = scratch.frame_mut(depth);
            for answer_index in subset {
                let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
                if frame.child_subsets[pattern].is_empty() {
                    frame.touched_patterns.push(pattern as u8);
                }
                frame.masses[pattern] += weights[*answer_index];
                frame.child_subsets[pattern].push(*answer_index);
            }
            let len = frame.touched_patterns.len();
            ordered_patterns[..len].copy_from_slice(&frame.touched_patterns);
            let masses = &frame.masses;
            ordered_patterns[..len]
                .sort_by(|left, right| masses[*right as usize].total_cmp(&masses[*left as usize]));
            len
        };

        let mut cost = 1.0;
        for pattern in ordered_patterns[..ordered_len].iter().copied() {
            let mass = scratch.frames[depth].masses[pattern as usize];
            let branch_probability = mass / total_weight;
            let child_cost = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else {
                let child_subset =
                    std::mem::take(&mut scratch.frames[depth].child_subsets[pattern as usize]);
                let result =
                    if child_subset.len() == subset.len() && child_subset.as_slice() == subset {
                        f64::INFINITY
                    } else {
                        self.exact_best_cost(
                            &child_subset,
                            weights,
                            small_state_table,
                            memo,
                            scratch,
                            depth + 1,
                        )?
                    };
                scratch.frames[depth].child_subsets[pattern as usize] = child_subset;
                result
            };
            cost += branch_probability * child_cost;
            if cost >= best_bound {
                return Ok(cost);
            }
        }

        Ok(cost)
    }

    fn exact_best_cost(
        &self,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
        memo: &mut HashMap<ExactSubsetKey, f64>,
        scratch: &mut ExactSearchScratch,
        depth: usize,
    ) -> Result<f64> {
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() == 1 {
            return Ok(1.0);
        }

        let key = ExactSubsetKey::from_sorted_subset(subset);
        if let Some(cached) = memo.get(&key) {
            return Ok(*cached);
        }

        let suggestion_mode = exact_suggestion_mode(&self.config, subset.len());
        let scores = match suggestion_mode {
            Some(ExactSuggestionMode::Exhaustive) => (0..self.guesses.len()).collect::<Vec<_>>(),
            Some(ExactSuggestionMode::Pooled) | None => {
                self.top_guess_indexes_for_subset(subset, weights, self.config.exact_candidate_pool)
            }
        };
        let lower_bound = small_state_table.lower_bound(subset.len());
        for answer_index in subset {
            debug_assert!(u16::try_from(*answer_index).is_ok());
        }
        let mut best_cost = f64::INFINITY;
        for guess_index in scores.iter().copied() {
            let cost = self.exact_cost_for_guess(
                guess_index,
                subset,
                weights,
                small_state_table,
                memo,
                best_cost,
                scratch,
                depth,
            )?;
            if cost < best_cost {
                best_cost = cost;
                if best_cost <= lower_bound {
                    break;
                }
            }
        }
        if !best_cost.is_finite()
            && matches!(suggestion_mode, Some(ExactSuggestionMode::Pooled) | None)
        {
            let shortlisted = scores.into_iter().collect::<HashSet<_>>();
            for guess_index in 0..self.guesses.len() {
                if shortlisted.contains(&guess_index) {
                    continue;
                }
                let cost = self.exact_cost_for_guess(
                    guess_index,
                    subset,
                    weights,
                    small_state_table,
                    memo,
                    best_cost,
                    scratch,
                    depth,
                )?;
                if cost < best_cost {
                    best_cost = cost;
                    if best_cost <= lower_bound {
                        break;
                    }
                }
            }
        }
        if !best_cost.is_finite() {
            bail!(
                "no valid exact guess found for subset of size {}",
                subset.len()
            );
        }
        memo.insert(key, best_cost);
        Ok(best_cost)
    }

    fn top_guess_indexes_for_subset(
        &self,
        subset: &[usize],
        weights: &[f64],
        count: usize,
    ) -> Vec<usize> {
        let mut metrics =
            self.score_guess_metrics_for_subset(subset, weights, &self.exact_small_state_table);
        metrics.sort_by(|left, right| compare_guess_metrics(left, right, &self.guesses));
        metrics.truncate(count);
        metrics
            .into_iter()
            .map(|metric| metric.guess_index)
            .collect()
    }
}

#[derive(Clone, Debug)]
struct PriorMetrics {
    target_probability: f64,
    target_rank: usize,
    log_loss: f64,
    brier: f64,
}

fn exact_suggestion_mode(
    config: &PriorConfig,
    surviving_answers: usize,
) -> Option<ExactSuggestionMode> {
    if surviving_answers > config.exact_threshold {
        return None;
    }
    if surviving_answers
        <= config
            .exact_exhaustive_threshold
            .min(config.exact_threshold)
    {
        Some(ExactSuggestionMode::Exhaustive)
    } else {
        Some(ExactSuggestionMode::Pooled)
    }
}

fn predictive_search_mode(config: &PriorConfig, surviving_answers: usize) -> PredictiveSearchMode {
    if let Some(mode) = exact_suggestion_mode(config, surviving_answers) {
        PredictiveSearchMode::Exact(mode)
    } else if surviving_answers <= config.lookahead_threshold {
        PredictiveSearchMode::Lookahead
    } else {
        PredictiveSearchMode::ProxyOnly
    }
}

fn compare_guess_metrics(
    left: &GuessMetrics,
    right: &GuessMetrics,
    guesses: &[String],
) -> std::cmp::Ordering {
    left.proxy_cost
        .total_cmp(&right.proxy_cost)
        .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
        .then_with(|| right.entropy.total_cmp(&left.entropy))
        .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
        .then_with(|| {
            right
                .posterior_answer_probability
                .total_cmp(&left.posterior_answer_probability)
        })
        .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
}

fn compare_suggestions(
    left: &Suggestion,
    right: &Suggestion,
    _surviving_words: &HashSet<&str>,
) -> std::cmp::Ordering {
    left.proxy_cost
        .unwrap_or(f64::INFINITY)
        .total_cmp(&right.proxy_cost.unwrap_or(f64::INFINITY))
        .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
        .then_with(|| right.entropy.total_cmp(&left.entropy))
        .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
        .then_with(|| {
            right
                .posterior_answer_probability
                .total_cmp(&left.posterior_answer_probability)
        })
        .then_with(|| left.word.cmp(&right.word))
}

fn compare_lookahead(
    left: &Suggestion,
    right: &Suggestion,
    surviving_words: &HashSet<&str>,
) -> std::cmp::Ordering {
    match (left.lookahead_cost, right.lookahead_cost) {
        (Some(left_cost), Some(right_cost)) => left_cost
            .total_cmp(&right_cost)
            .then_with(|| compare_suggestions(left, right, surviving_words)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right, surviving_words),
    }
}

fn compare_exact(
    left: &Suggestion,
    right: &Suggestion,
    surviving_words: &HashSet<&str>,
) -> std::cmp::Ordering {
    match (left.exact_cost, right.exact_cost) {
        (Some(left_cost), Some(right_cost)) => left_cost
            .total_cmp(&right_cost)
            .then_with(|| compare_suggestions(left, right, surviving_words)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right, surviving_words),
    }
}

fn compare_exact_costs(
    left: &Suggestion,
    right: &Suggestion,
    left_cost: Option<f64>,
    right_cost: Option<f64>,
    surviving_words: &HashSet<&str>,
) -> std::cmp::Ordering {
    match (left_cost, right_cost) {
        (Some(left_cost), Some(right_cost)) => left_cost
            .total_cmp(&right_cost)
            .then_with(|| compare_suggestions(left, right, surviving_words)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right, surviving_words),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use chrono::NaiveDate;

    use crate::{
        config::PriorConfig,
        model::{AnswerRecord, ModelVariant, WeightMode},
        pattern_table::PatternTable,
        scoring::{format_feedback_letters, score_guess},
        small_state::SmallStateTable,
    };

    use super::{
        ExactSearchScratch, ExactSubsetKey, ExactSubsetStorage, ExactSuggestionMode, GuessMetrics,
        Solver, compare_guess_metrics, exact_suggestion_mode,
    };

    fn test_solver(words: &[&str]) -> Solver {
        let guesses = words
            .iter()
            .map(|word| (*word).to_string())
            .collect::<Vec<_>>();
        let answers = words
            .iter()
            .map(|word| AnswerRecord {
                word: (*word).to_string(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect::<Vec<_>>();
        let pattern_root: PathBuf =
            std::env::temp_dir().join(format!("maybe-wordle-solver-test-{}", words.join("-")));
        let _ = std::fs::remove_dir_all(&pattern_root);
        std::fs::create_dir_all(&pattern_root).expect("pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&pattern_root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        Solver {
            config: PriorConfig::default(),
            mode: WeightMode::Uniform,
            variant: ModelVariant::SeedPlusHistory,
            guesses: guesses.clone(),
            answers,
            history_dates: Vec::new(),
            exact_small_state_table: SmallStateTable::build(4),
            pattern_table,
            guess_index: guesses
                .iter()
                .enumerate()
                .map(|(index, guess)| (guess.clone(), index))
                .collect::<HashMap<_, _>>(),
        }
    }

    #[test]
    fn parse_observations_rejects_length_mismatch() {
        let error = Solver::parse_observations(&["crane".into()], &[]).expect_err("must fail");
        assert!(error.to_string().contains("same number"));
    }

    #[test]
    fn target_feedback_matches_expected_fixture() {
        assert_eq!(
            format_feedback_letters(score_guess("lilly", "alley")),
            "ybgbg"
        );
    }

    #[test]
    fn initial_state_includes_seed_words() {
        let config = PriorConfig::default();
        let answer = AnswerRecord {
            word: "cigar".into(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid")],
        };
        let state_weight = crate::model::weight_snapshot(
            &answer,
            &config,
            NaiveDate::from_ymd_opt(2026, 3, 1).expect("valid"),
        );
        assert!(state_weight.final_weight > 0.0);
    }

    #[test]
    fn exact_mode_uses_exhaustive_search_for_tiny_states() {
        let config = PriorConfig::default();
        assert_eq!(
            exact_suggestion_mode(&config, config.exact_exhaustive_threshold),
            Some(ExactSuggestionMode::Exhaustive)
        );
    }

    #[test]
    fn exact_mode_keeps_pooled_search_between_thresholds() {
        let mut config = PriorConfig::default();
        config.exact_threshold = 16;
        config.exact_exhaustive_threshold = 8;
        assert_eq!(
            exact_suggestion_mode(&config, 12),
            Some(ExactSuggestionMode::Pooled)
        );
        assert_eq!(exact_suggestion_mode(&config, 17), None);
    }

    #[test]
    fn exact_subset_key_inlines_small_subsets() {
        let key = ExactSubsetKey::from_sorted_subset(&[1, 4, 9, 15]);
        assert!(matches!(
            key,
            ExactSubsetKey(ExactSubsetStorage::Inline { len: 4, .. })
        ));
    }

    #[test]
    fn exact_subset_key_boxes_large_subsets() {
        let subset = (0..17).collect::<Vec<_>>();
        let key = ExactSubsetKey::from_sorted_subset(&subset);
        assert!(matches!(key, ExactSubsetKey(ExactSubsetStorage::Heap(_))));
    }

    #[test]
    fn proxy_ordering_beats_raw_entropy() {
        let guesses = vec!["alpha".to_string(), "bravo".to_string()];
        let better_proxy = GuessMetrics {
            guess_index: 0,
            entropy: 3.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            proxy_cost: 1.8,
            posterior_answer_probability: 0.0,
        };
        let worse_proxy = GuessMetrics {
            guess_index: 1,
            entropy: 4.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            proxy_cost: 2.2,
            posterior_answer_probability: 0.0,
        };
        assert_eq!(
            compare_guess_metrics(&better_proxy, &worse_proxy, &guesses),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn pooled_exact_candidates_keep_surviving_answers() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let state = super::SolveState {
            surviving: vec![0, 1],
            weights: vec![1.0; solver.answers.len()],
            total_weight: 2.0,
        };
        let suggestions = vec![
            super::Suggestion {
                word: "humph".into(),
                entropy: 5.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                proxy_cost: Some(1.5),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            super::Suggestion {
                word: "awake".into(),
                entropy: 4.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                proxy_cost: Some(1.6),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
        ];

        let candidates = solver
            .collect_exact_candidates(&state, &suggestions)
            .expect("candidates");
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&1));
    }

    #[test]
    fn lookahead_uses_exact_recursion_for_small_children() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        solver.config.exact_threshold = 2;
        solver.config.lookahead_threshold = 4;
        let subset = vec![0, 1];
        let weights = vec![1.0; solver.answers.len()];
        let mut exact_memo = HashMap::new();
        let mut exact_scratch = ExactSearchScratch::new();
        let mut lookahead_memo = HashMap::new();

        let lookahead_value = solver
            .lookahead_child_value(
                &subset,
                &weights,
                &mut exact_memo,
                &mut exact_scratch,
                &mut lookahead_memo,
            )
            .expect("lookahead value");
        let exact_value = solver
            .exact_best_cost(
                &subset,
                &weights,
                &solver.exact_small_state_table,
                &mut HashMap::new(),
                &mut ExactSearchScratch::new(),
                0,
            )
            .expect("exact value");
        assert!((lookahead_value - exact_value).abs() < 1e-9);
    }

    #[test]
    fn exact_search_errors_when_no_guess_shrinks_subset() {
        let corpus = ["cigar", "rebut", "sissy", "humph", "awake", "blush"];
        let mut witness = None;
        'outer: for guess in corpus {
            for left_index in 0..corpus.len() {
                for right_index in (left_index + 1)..corpus.len() {
                    if score_guess(guess, corpus[left_index])
                        == score_guess(guess, corpus[right_index])
                    {
                        witness = Some((
                            guess.to_string(),
                            corpus[left_index].to_string(),
                            corpus[right_index].to_string(),
                        ));
                        break 'outer;
                    }
                }
            }
        }
        let witness = witness.expect("need a non-splitting witness");
        let guesses = vec![witness.0.clone()];
        let answers = vec![
            AnswerRecord {
                word: witness.1.clone(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            },
            AnswerRecord {
                word: witness.2.clone(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            },
        ];
        let pattern_root: PathBuf = std::env::temp_dir().join("maybe-wordle-exact-no-shrink");
        let _ = std::fs::remove_dir_all(&pattern_root);
        std::fs::create_dir_all(&pattern_root).expect("pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&pattern_root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        let solver = Solver {
            config: PriorConfig::default(),
            mode: WeightMode::Uniform,
            variant: ModelVariant::SeedPlusHistory,
            guesses: guesses.clone(),
            answers,
            history_dates: Vec::new(),
            exact_small_state_table: SmallStateTable::build(2),
            pattern_table,
            guess_index: guesses
                .iter()
                .enumerate()
                .map(|(index, guess)| (guess.clone(), index))
                .collect::<HashMap<_, _>>(),
        };
        let mut memo = HashMap::new();
        let mut scratch = ExactSearchScratch::new();
        let error = solver
            .exact_best_cost(
                &[0, 1],
                &[1.0, 1.0],
                &solver.exact_small_state_table,
                &mut memo,
                &mut scratch,
                0,
            )
            .expect_err("no shrinking guess should error");
        assert!(error.to_string().contains("no valid exact guess found"));
        assert!(memo.is_empty());
        let _ = std::fs::remove_dir_all(&pattern_root);
    }
}
