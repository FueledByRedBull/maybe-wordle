use std::{
    array,
    collections::{HashMap, HashSet},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{Days, NaiveDate, Utc};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

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
    pub force_in_two: bool,
    pub worst_non_green_bucket_size: usize,
    pub largest_non_green_bucket_mass: f64,
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
pub struct SuggestionSnapshot {
    pub word: String,
    pub force_in_two: bool,
    pub worst_non_green_bucket_size: usize,
    pub largest_non_green_bucket_mass: f64,
    pub proxy_cost: Option<f64>,
    pub lookahead_cost: Option<f64>,
    pub exact_cost: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct DetailedSolveStep {
    pub guess: String,
    pub feedback: u8,
    pub surviving_before: usize,
    pub surviving_after: usize,
    pub chosen_force_in_two: bool,
    pub alternative_force_in_two: bool,
    pub danger_score: f64,
    pub danger_escalated: bool,
    pub regime_used: PredictiveRegime,
    pub top_suggestions: Vec<SuggestionSnapshot>,
}

#[derive(Clone, Debug)]
pub struct DetailedSolveRun {
    pub target: String,
    pub date: NaiveDate,
    pub steps: Vec<DetailedSolveStep>,
    pub solved: bool,
}

impl From<DetailedSolveRun> for SolveRun {
    fn from(value: DetailedSolveRun) -> Self {
        Self {
            target: value.target,
            date: value.date,
            steps: value
                .steps
                .into_iter()
                .map(|step| SolveStep {
                    guess: step.guess,
                    feedback: step.feedback,
                })
                .collect(),
            solved: value.solved,
        }
    }
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
    pub proxy_step_pct: f64,
    pub lookahead_step_pct: f64,
    pub escalated_exact_step_pct: f64,
    pub exact_step_pct: f64,
}

#[derive(Clone, Debug)]
pub struct DetailedBacktestReport {
    pub summary: BacktestStats,
    pub runs: Vec<DetailedSolveRun>,
}

#[derive(Clone, Debug)]
pub struct HardCaseResult {
    pub label: String,
    pub run: DetailedSolveRun,
}

#[derive(Clone, Debug)]
pub struct HardCaseReport {
    pub average_guesses: f64,
    pub failures: usize,
    pub cases: Vec<HardCaseResult>,
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
    pub hard_case_average_guesses: f64,
    pub hard_case_failures: usize,
    pub proxy_step_pct: f64,
    pub lookahead_step_pct: f64,
    pub escalated_exact_step_pct: f64,
    pub exact_step_pct: f64,
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
    EscalatedExact,
    Exact(ExactSuggestionMode),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictiveRegime {
    Proxy,
    Lookahead,
    EscalatedExact,
    Exact,
}

impl PredictiveRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::Proxy => "proxy",
            Self::Lookahead => "lookahead",
            Self::EscalatedExact => "escalated_exact",
            Self::Exact => "exact",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GuessMetrics {
    guess_index: usize,
    entropy: f64,
    solve_probability: f64,
    expected_remaining: f64,
    force_in_two: bool,
    worst_non_green_bucket_size: usize,
    largest_non_green_bucket_mass: f64,
    high_mass_ambiguous_bucket_count: usize,
    proxy_cost: f64,
    posterior_answer_probability: f64,
}

#[derive(Clone, Copy, Debug)]
struct StateDangerAssessment {
    danger_score: f64,
    dangerous_lookahead: bool,
    dangerous_exact: bool,
}

#[derive(Clone, Debug)]
struct SuggestionBatch {
    suggestions: Vec<Suggestion>,
    danger_score: f64,
    danger_escalated: bool,
    regime_used: PredictiveRegime,
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

type PredictiveMemoMap<K, V> = FxHashMap<K, V>;

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
        Ok(self.suggestion_batch(state, top)?.suggestions)
    }

    fn suggestion_batch(&self, state: &SolveState, top: usize) -> Result<SuggestionBatch> {
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        let mut metrics = self.score_guess_metrics_for_subset(
            &state.surviving,
            &state.weights,
            &self.exact_small_state_table,
        );
        metrics.sort_by(|left, right| compare_guess_metrics(left, right, &self.guesses));
        let assessment = self.assess_state_danger(state, &metrics);
        let search_mode = predictive_search_mode(&self.config, state.surviving.len(), assessment);
        let mut suggestions = metrics
            .into_iter()
            .map(|metric| Suggestion {
                word: self.guesses[metric.guess_index].clone(),
                entropy: metric.entropy,
                solve_probability: metric.solve_probability,
                expected_remaining: metric.expected_remaining,
                force_in_two: metric.force_in_two,
                worst_non_green_bucket_size: metric.worst_non_green_bucket_size,
                largest_non_green_bucket_mass: metric.largest_non_green_bucket_mass,
                proxy_cost: Some(metric.proxy_cost),
                posterior_answer_probability: metric.posterior_answer_probability,
                lookahead_cost: None,
                exact_cost: None,
            })
            .collect::<Vec<_>>();

        if let PredictiveSearchMode::Lookahead = search_mode {
            let root_candidates =
                self.collect_lookahead_candidates(&suggestions, assessment.dangerous_lookahead)?;
            let mut exact_memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut lookahead_memo = PredictiveMemoMap::default();
            let mut lookahead_costs = vec![None; self.guesses.len()];

            for guess_index in root_candidates {
                let cost = self.lookahead_cost_for_guess(
                    guess_index,
                    &state.surviving,
                    &state.weights,
                    assessment.dangerous_lookahead,
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
            suggestions.sort_by(compare_lookahead);
        }

        if let PredictiveSearchMode::Exact(exact_mode) = search_mode {
            let exact_candidates = match exact_mode {
                ExactSuggestionMode::Exhaustive => (0..self.guesses.len()).collect::<Vec<_>>(),
                ExactSuggestionMode::Pooled => self.collect_exact_candidates(
                    state,
                    &suggestions,
                    self.config.exact_candidate_pool,
                )?,
            };
            let mut memo = PredictiveMemoMap::default();
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
                    suggestions.sort_by(compare_exact);
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
                        compare_exact_costs(left, right, left_cost, right_cost)
                    });
                }
            }
        }

        if let PredictiveSearchMode::EscalatedExact = search_mode {
            let exact_candidates = self.collect_exact_candidates(
                state,
                &suggestions,
                self.config.danger_exact_root_pool.max(1),
            )?;
            let mut memo = PredictiveMemoMap::default();
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

            for suggestion in &mut suggestions {
                suggestion.exact_cost = self
                    .guess_index
                    .get(&suggestion.word)
                    .and_then(|guess_index| exact_costs[*guess_index]);
            }
            suggestions.sort_by(|left, right| {
                let left_cost = self
                    .guess_index
                    .get(&left.word)
                    .and_then(|guess_index| exact_costs[*guess_index]);
                let right_cost = self
                    .guess_index
                    .get(&right.word)
                    .and_then(|guess_index| exact_costs[*guess_index]);
                compare_exact_costs(left, right, left_cost, right_cost)
            });
        }

        suggestions.truncate(top);
        Ok(SuggestionBatch {
            suggestions,
            danger_score: assessment.danger_score,
            danger_escalated: matches!(search_mode, PredictiveSearchMode::EscalatedExact)
                || (matches!(search_mode, PredictiveSearchMode::Lookahead)
                    && assessment.dangerous_lookahead),
            regime_used: regime_from_search_mode(search_mode),
        })
    }

    pub fn solve_target(&self, target: &str, date: NaiveDate, top: usize) -> Result<SolveRun> {
        Ok(self.solve_target_detailed(target, date, top)?.into())
    }

    pub fn solve_target_detailed(
        &self,
        target: &str,
        date: NaiveDate,
        top: usize,
    ) -> Result<DetailedSolveRun> {
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        self.solve_target_from_state_detailed(target, as_of, date, top)
    }

    fn solve_target_from_state_detailed(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        top: usize,
    ) -> Result<DetailedSolveRun> {
        let target = target.to_ascii_lowercase();
        let mut state = self.initial_state(as_of);

        if !state
            .surviving
            .iter()
            .any(|index| self.answers[*index].word == target)
        {
            return Ok(DetailedSolveRun {
                target,
                date,
                steps: Vec::new(),
                solved: false,
            });
        }

        let mut steps = Vec::new();
        while steps.len() < 6 {
            let surviving_before = state.surviving.len();
            let batch = self.suggestion_batch(&state, top.max(1))?;
            let chosen = batch
                .suggestions
                .first()
                .ok_or_else(|| anyhow!("solver returned no suggestions"))?
                .clone();
            let feedback = score_guess(&chosen.word, &target);
            let surviving_after = if feedback == ALL_GREEN_PATTERN {
                1
            } else {
                let mut next_state = state.clone();
                self.apply_feedback(&mut next_state, &chosen.word, feedback)?;
                next_state.surviving.len()
            };
            steps.push(DetailedSolveStep {
                guess: chosen.word.clone(),
                feedback,
                surviving_before,
                surviving_after,
                chosen_force_in_two: chosen.force_in_two,
                alternative_force_in_two: batch
                    .suggestions
                    .iter()
                    .skip(1)
                    .any(|suggestion| suggestion.force_in_two),
                danger_score: batch.danger_score,
                danger_escalated: batch.danger_escalated,
                regime_used: batch.regime_used,
                top_suggestions: batch
                    .suggestions
                    .iter()
                    .take(top.max(1))
                    .map(Self::snapshot_suggestion)
                    .collect(),
            });
            if feedback == ALL_GREEN_PATTERN {
                return Ok(DetailedSolveRun {
                    target,
                    date,
                    steps,
                    solved: true,
                });
            }
            self.apply_feedback(&mut state, &chosen.word, feedback)?;
        }

        Ok(DetailedSolveRun {
            target,
            date,
            steps,
            solved: false,
        })
    }

    pub fn backtest(&self, from: NaiveDate, to: NaiveDate, top: usize) -> Result<BacktestStats> {
        Ok(self.backtest_detailed(from, to, top)?.summary)
    }

    pub fn backtest_detailed(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<DetailedBacktestReport> {
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
        let mut runs = Vec::new();

        for entry in games {
            let run = self.solve_target_detailed(&entry.solution, entry.print_date, top)?;
            if run.steps.is_empty() {
                coverage_gaps += 1;
                failures += 1;
                runs.push(run);
                continue;
            }
            if !run.solved {
                failures += 1;
            }
            guess_counts.push(run.steps.len());
            runs.push(run);
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

        Ok(DetailedBacktestReport {
            summary: BacktestStats {
                games: games_played,
                average_guesses,
                p95_guesses,
                max_guesses,
                failures,
                coverage_gaps,
            },
            runs,
        })
    }

    pub fn hard_case_report(&self, top: usize) -> Result<HardCaseReport> {
        let as_of = Self::today();
        let cases = self.select_hard_case_targets(as_of, top)?;
        let mut results = Vec::new();
        let mut failures = 0usize;
        let mut guess_total = 0usize;

        for (label, target) in cases {
            let run = self.solve_target_from_state_detailed(&target, as_of, as_of, top)?;
            if !run.solved {
                failures += 1;
            }
            guess_total += run.steps.len();
            results.push(HardCaseResult { label, run });
        }

        let average_guesses = if results.is_empty() {
            0.0
        } else {
            guess_total as f64 / results.len() as f64
        };
        Ok(HardCaseReport {
            average_guesses,
            failures,
            cases: results,
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

        let detailed = self.backtest_detailed(from, to, top)?;
        let backtest = detailed.summary.clone();
        let (proxy_step_pct, lookahead_step_pct, escalated_exact_step_pct, exact_step_pct) =
            Self::regime_mix(&detailed.runs);
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
            proxy_step_pct,
            lookahead_step_pct,
            escalated_exact_step_pct,
            exact_step_pct,
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

    pub fn has_guess(&self, guess: &str) -> bool {
        self.guess_index.contains_key(&guess.to_ascii_lowercase())
    }

    fn snapshot_suggestion(suggestion: &Suggestion) -> SuggestionSnapshot {
        SuggestionSnapshot {
            word: suggestion.word.clone(),
            force_in_two: suggestion.force_in_two,
            worst_non_green_bucket_size: suggestion.worst_non_green_bucket_size,
            largest_non_green_bucket_mass: suggestion.largest_non_green_bucket_mass,
            proxy_cost: suggestion.proxy_cost,
            lookahead_cost: suggestion.lookahead_cost,
            exact_cost: suggestion.exact_cost,
        }
    }

    fn assess_state_danger(
        &self,
        state: &SolveState,
        metrics: &[GuessMetrics],
    ) -> StateDangerAssessment {
        self.assess_subset_danger(
            &state.surviving,
            &state.weights,
            state.total_weight,
            metrics,
        )
    }

    fn assess_subset_danger(
        &self,
        subset: &[usize],
        weights: &[f64],
        total_weight: f64,
        metrics: &[GuessMetrics],
    ) -> StateDangerAssessment {
        if metrics.is_empty() || total_weight <= 0.0 {
            return StateDangerAssessment {
                danger_score: 0.0,
                dangerous_lookahead: false,
                dangerous_exact: false,
            };
        }

        let mut posterior = subset
            .iter()
            .map(|index| weights[*index] / total_weight)
            .collect::<Vec<_>>();
        posterior.sort_by(|left, right| right.total_cmp(left));
        let top_concentration = posterior.iter().take(3).sum::<f64>();
        let best = metrics[0];
        let top_window = metrics.iter().take(3).copied().collect::<Vec<_>>();
        let disagreement = top_window.iter().skip(1).any(|metric| {
            metric.force_in_two != best.force_in_two
                || (metric.largest_non_green_bucket_mass - best.largest_non_green_bucket_mass).abs()
                    >= 0.10
                || metric
                    .worst_non_green_bucket_size
                    .abs_diff(best.worst_non_green_bucket_size)
                    >= 2
        });
        let worst_bucket_ratio =
            best.worst_non_green_bucket_size as f64 / subset.len().max(1) as f64;
        let ambiguous_bucket_pressure =
            (best.high_mass_ambiguous_bucket_count as f64 / 4.0).min(1.0);
        let danger_score = (0.30 * top_concentration)
            + (0.25 * best.largest_non_green_bucket_mass)
            + (0.20 * worst_bucket_ratio)
            + (0.15 * ambiguous_bucket_pressure)
            + if disagreement { 0.10 } else { 0.0 };
        StateDangerAssessment {
            danger_score,
            dangerous_lookahead: danger_score >= self.config.danger_lookahead_threshold,
            dangerous_exact: danger_score >= self.config.danger_exact_threshold,
        }
    }

    fn regime_mix(runs: &[DetailedSolveRun]) -> (f64, f64, f64, f64) {
        let mut proxy_steps = 0usize;
        let mut lookahead_steps = 0usize;
        let mut escalated_exact_steps = 0usize;
        let mut exact_steps = 0usize;
        let mut total_steps = 0usize;

        for run in runs {
            for step in &run.steps {
                total_steps += 1;
                match step.regime_used {
                    PredictiveRegime::Proxy => proxy_steps += 1,
                    PredictiveRegime::Lookahead => lookahead_steps += 1,
                    PredictiveRegime::EscalatedExact => escalated_exact_steps += 1,
                    PredictiveRegime::Exact => exact_steps += 1,
                }
            }
        }

        if total_steps == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let divisor = total_steps as f64;
        (
            proxy_steps as f64 / divisor,
            lookahead_steps as f64 / divisor,
            escalated_exact_steps as f64 / divisor,
            exact_steps as f64 / divisor,
        )
    }

    fn select_hard_case_targets(
        &self,
        as_of: NaiveDate,
        top: usize,
    ) -> Result<Vec<(String, String)>> {
        let state = self.initial_state(as_of);
        let weighted_answers = state
            .surviving
            .iter()
            .map(|answer_index| {
                (
                    *answer_index,
                    self.answers[*answer_index].word.clone(),
                    state.weights[*answer_index],
                )
            })
            .collect::<Vec<_>>();
        let repeated_letters = weighted_answers
            .iter()
            .find(|(_, word, _)| has_repeated_letters(word))
            .map(|(_, word, _)| word.clone());
        let dense_cluster = weighted_answers
            .iter()
            .max_by_key(|(answer_index, _, _)| {
                weighted_answers
                    .iter()
                    .filter(|(other_index, _, _)| {
                        *answer_index != *other_index
                            && hamming_distance(
                                &self.answers[*answer_index].word,
                                &self.answers[*other_index].word,
                            ) <= 1
                    })
                    .count()
            })
            .map(|(_, word, _)| word.clone());
        let low_prior_outlier = weighted_answers
            .iter()
            .filter(|(_, _, weight)| *weight > 0.0)
            .min_by(|left, right| left.2.total_cmp(&right.2))
            .map(|(_, word, _)| word.clone());
        let high_posterior_trap = {
            let mut ranked = weighted_answers.clone();
            ranked.sort_by(|left, right| right.2.total_cmp(&left.2));
            ranked
                .iter()
                .take(96)
                .filter_map(|(answer_index, word, weight)| {
                    let cluster_mass = ranked
                        .iter()
                        .take(96)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= 1
                        })
                        .map(|(_, _, other_weight)| *other_weight)
                        .sum::<f64>();
                    let neighbors = ranked
                        .iter()
                        .take(96)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= 1
                        })
                        .count();
                    (neighbors >= 2).then_some((cluster_mass + *weight, word.clone()))
                })
                .max_by(|left, right| left.0.total_cmp(&right.0))
                .map(|(_, word)| word)
        };

        let opener = self
            .suggestions(&state, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("missing predictive opener"))?;
        let mut non_answer_splitter_needed = None;
        let mut candidate_answers = weighted_answers;
        candidate_answers.sort_by(|left, right| left.2.total_cmp(&right.2));
        for (_, target, _) in candidate_answers.into_iter().take(128) {
            let feedback = score_guess(&opener.word, &target);
            if feedback == ALL_GREEN_PATTERN {
                continue;
            }
            let mut child_state = state.clone();
            self.apply_feedback(&mut child_state, &opener.word, feedback)?;
            if child_state.surviving.len() <= 1 {
                continue;
            }
            let reply = self
                .suggestions(&child_state, top.max(1))?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("missing predictive reply"))?;
            let surviving_words = child_state
                .surviving
                .iter()
                .map(|index| self.answers[*index].word.as_str())
                .collect::<HashSet<_>>();
            if !surviving_words.contains(reply.word.as_str()) {
                non_answer_splitter_needed = Some(target);
                break;
            }
        }

        let mut selected = Vec::new();
        for (label, target) in [
            ("repeated_letters", repeated_letters),
            ("dense_cluster", dense_cluster),
            ("low_prior_outlier", low_prior_outlier),
            ("non_answer_splitter_needed", non_answer_splitter_needed),
            ("high_posterior_trap", high_posterior_trap),
        ] {
            if let Some(target) = target {
                if label == "high_posterior_trap"
                    || selected
                        .iter()
                        .all(|(_, existing): &(String, String)| existing != &target)
                {
                    selected.push((label.to_string(), target));
                }
            }
        }
        if selected.is_empty() {
            bail!("unable to construct hard-case suite from current model");
        }
        Ok(selected)
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
        let candidate = Self::evaluate_tuning_candidate(
            paths,
            &best_prior_config,
            validation_start,
            window_end,
        )?;
        let best = if candidate.average_guesses < current.average_guesses
            && candidate.failures <= current.failures
            && candidate.hard_case_failures <= current.hard_case_failures
            && candidate.latency_p95_ms
                <= (current.latency_p95_ms * 3.0).max(current.latency_p95_ms)
        {
            candidate
        } else {
            current.clone()
        };

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
                "lookahead_root_force_in_two_scan = {}\n",
                "danger_lookahead_threshold = {:.2}\n",
                "danger_exact_threshold = {:.2}\n",
                "danger_reply_pool_bonus = {}\n",
                "danger_exact_root_pool = {}\n",
                "danger_exact_survivor_cap = {}\n",
                "lookahead_trap_penalty = {:.2}\n",
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
            best.config.lookahead_root_force_in_two_scan,
            best.config.danger_lookahead_threshold,
            best.config.danger_exact_threshold,
            best.config.danger_reply_pool_bonus,
            best.config.danger_exact_root_pool,
            best.config.danger_exact_survivor_cap,
            best.config.lookahead_trap_penalty,
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
        let hard_cases = solver.hard_case_report(5)?;
        Ok(TuningEvaluation {
            config: config.clone(),
            average_guesses: report.backtest.average_guesses,
            failures: report.backtest.failures,
            coverage_gaps: report.backtest.coverage_gaps,
            average_log_loss: report.average_log_loss,
            average_target_rank: report.average_target_rank,
            latency_p95_ms: report.latency_p95_ms,
            hard_case_average_guesses: hard_cases.average_guesses,
            hard_case_failures: hard_cases.failures,
            proxy_step_pct: report.proxy_step_pct,
            lookahead_step_pct: report.lookahead_step_pct,
            escalated_exact_step_pct: report.escalated_exact_step_pct,
            exact_step_pct: report.exact_step_pct,
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
        let mut sum_mass_log_mass = 0.0;
        let mut expected_remaining = 0.0;
        let mut solve_probability = 0.0;
        let mut force_in_two = true;
        let mut worst_non_green_bucket_size = 0usize;
        let mut largest_non_green_bucket_mass = 0.0_f64;
        let mut high_mass_ambiguous_bucket_count = 0usize;
        let mut proxy_cost = 1.0;

        for pattern in scratch.touched_patterns.iter().copied() {
            let index = pattern as usize;
            let mass = scratch.masses[index];
            let probability = if total_weight > 0.0 {
                mass / total_weight
            } else {
                0.0
            };
            if mass > 0.0 {
                sum_mass_log_mass += mass * mass.log2();
            }
            expected_remaining += probability * scratch.counts[index] as f64;
            if pattern == ALL_GREEN_PATTERN {
                solve_probability = probability;
                worst_non_green_bucket_size =
                    worst_non_green_bucket_size.max(scratch.counts[index]);
            } else {
                worst_non_green_bucket_size =
                    worst_non_green_bucket_size.max(scratch.counts[index]);
                largest_non_green_bucket_mass = largest_non_green_bucket_mass.max(probability);
                if scratch.counts[index] > 1 {
                    force_in_two = false;
                    if probability >= 0.10 {
                        high_mass_ambiguous_bucket_count += 1;
                    }
                }
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
        let entropy = if total_weight > 0.0 {
            total_weight.log2() - (sum_mass_log_mass / total_weight)
        } else {
            0.0
        };

        GuessMetrics {
            guess_index,
            entropy,
            solve_probability,
            expected_remaining,
            force_in_two,
            worst_non_green_bucket_size,
            largest_non_green_bucket_mass,
            high_mass_ambiguous_bucket_count,
            proxy_cost,
            posterior_answer_probability,
        }
    }

    fn collect_exact_candidates(
        &self,
        state: &SolveState,
        suggestions: &[Suggestion],
        non_surviving_limit: usize,
    ) -> Result<Vec<usize>> {
        let pool = non_surviving_limit.max(1);
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

        let mut by_worst_bucket = suggestions.iter().collect::<Vec<_>>();
        by_worst_bucket.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_worst_bucket.into_iter().take(pool / 6) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_mass_reducer = suggestions.iter().collect::<Vec<_>>();
        by_mass_reducer.sort_by(|left, right| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_mass_reducer.into_iter().take(pool / 6) {
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

        for suggestion in suggestions
            .iter()
            .take(self.config.lookahead_root_force_in_two_scan.max(pool))
            .filter(|suggestion| suggestion.force_in_two)
        {
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

        let extra_limit = non_surviving_limit + surviving_guess_indexes.len();
        if candidate_indexes.len() > extra_limit {
            let mut trimmed = Vec::with_capacity(extra_limit);
            let mut extra_count = 0usize;
            for guess_index in candidate_indexes {
                if surviving_guess_indexes.contains(&guess_index) {
                    trimmed.push(guess_index);
                    continue;
                }
                if extra_count < non_surviving_limit {
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
        expanded: bool,
        exact_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        exact_scratch: &mut ExactSearchScratch,
        lookahead_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
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
        let mut worst_child_probability = 0.0_f64;
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
                            expanded,
                            exact_memo,
                            exact_scratch,
                            lookahead_memo,
                        )?
                    };
                exact_scratch.frames[0].child_subsets[pattern as usize] = child_subset;
                result
            };
            total_cost += probability * child_value;
            if pattern != ALL_GREEN_PATTERN {
                worst_child_probability = worst_child_probability.max(probability);
            }
        }
        Ok(total_cost + (self.config.lookahead_trap_penalty * worst_child_probability))
    }

    fn lookahead_child_value(
        &self,
        subset: &[usize],
        weights: &[f64],
        expanded: bool,
        exact_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        exact_scratch: &mut ExactSearchScratch,
        lookahead_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
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
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let assessment = self.assess_subset_danger(subset, weights, total_weight, &metrics);
        metrics.sort_by(|left, right| compare_guess_metrics(left, right, &self.guesses));
        let reply_pool = self.config.lookahead_reply_pool.max(1)
            + if expanded || assessment.dangerous_lookahead {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let mut reply_candidates = Vec::new();
        let mut seen = HashSet::new();
        for metric in metrics.iter().take(reply_pool) {
            if seen.insert(metric.guess_index) {
                reply_candidates.push(*metric);
            }
        }
        let mut by_worst_bucket = metrics.iter().collect::<Vec<_>>();
        by_worst_bucket.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| compare_guess_metrics(left, right, &self.guesses))
        });
        for metric in by_worst_bucket.into_iter().take(reply_pool) {
            if seen.insert(metric.guess_index) {
                reply_candidates.push(*metric);
            }
        }
        let mut by_mass_reducer = metrics.iter().collect::<Vec<_>>();
        by_mass_reducer.sort_by(|left, right| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
                .then_with(|| compare_guess_metrics(left, right, &self.guesses))
        });
        for metric in by_mass_reducer.into_iter().take(reply_pool) {
            if seen.insert(metric.guess_index) {
                reply_candidates.push(*metric);
            }
        }
        let best_reply = reply_candidates
            .into_iter()
            .map(|metric| {
                metric.proxy_cost
                    + self.config.lookahead_trap_penalty
                        * ((metric.worst_non_green_bucket_size as f64 / subset.len().max(1) as f64)
                            + metric.largest_non_green_bucket_mass
                            + (metric.high_mass_ambiguous_bucket_count as f64 / 4.0).min(1.0))
            })
            .fold(f64::INFINITY, f64::min);
        let child_value = 1.0 + best_reply;
        lookahead_memo.insert(key, child_value);
        Ok(child_value)
    }

    fn collect_lookahead_candidates(
        &self,
        suggestions: &[Suggestion],
        expanded: bool,
    ) -> Result<Vec<usize>> {
        let pool = self.config.lookahead_candidate_pool.max(1)
            + if expanded {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let force_scan = self.config.lookahead_root_force_in_two_scan.max(1)
            + if expanded {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        for suggestion in suggestions.iter().take(pool) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        for suggestion in suggestions
            .iter()
            .take(force_scan)
            .filter(|suggestion| suggestion.force_in_two)
        {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        Ok(candidates)
    }

    fn exact_cost_for_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
        memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
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
        memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
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

fn predictive_search_mode(
    config: &PriorConfig,
    surviving_answers: usize,
    assessment: StateDangerAssessment,
) -> PredictiveSearchMode {
    if let Some(mode) = exact_suggestion_mode(config, surviving_answers) {
        PredictiveSearchMode::Exact(mode)
    } else if surviving_answers <= config.danger_exact_survivor_cap
        && surviving_answers > config.exact_threshold
        && assessment.dangerous_exact
    {
        PredictiveSearchMode::EscalatedExact
    } else if surviving_answers <= config.lookahead_threshold
        || (surviving_answers <= config.danger_exact_survivor_cap && assessment.dangerous_lookahead)
    {
        PredictiveSearchMode::Lookahead
    } else {
        PredictiveSearchMode::ProxyOnly
    }
}

fn regime_from_search_mode(search_mode: PredictiveSearchMode) -> PredictiveRegime {
    match search_mode {
        PredictiveSearchMode::ProxyOnly => PredictiveRegime::Proxy,
        PredictiveSearchMode::Lookahead => PredictiveRegime::Lookahead,
        PredictiveSearchMode::EscalatedExact => PredictiveRegime::EscalatedExact,
        PredictiveSearchMode::Exact(_) => PredictiveRegime::Exact,
    }
}

fn approx_tie(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9
}

fn compare_force_in_two(left: bool, right: bool) -> std::cmp::Ordering {
    right.cmp(&left)
}

fn has_repeated_letters(word: &str) -> bool {
    let mut seen = [false; 26];
    for byte in word.bytes() {
        let index = (byte - b'a') as usize;
        if seen[index] {
            return true;
        }
        seen[index] = true;
    }
    false
}

fn hamming_distance(left: &str, right: &str) -> usize {
    left.bytes()
        .zip(right.bytes())
        .filter(|(left, right)| left != right)
        .count()
}

fn compare_guess_metrics(
    left: &GuessMetrics,
    right: &GuessMetrics,
    guesses: &[String],
) -> std::cmp::Ordering {
    let proxy_cmp = left.proxy_cost.total_cmp(&right.proxy_cost);
    if proxy_cmp != std::cmp::Ordering::Equal && !approx_tie(left.proxy_cost, right.proxy_cost) {
        return proxy_cmp;
    }
    compare_force_in_two(left.force_in_two, right.force_in_two)
        .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
        .then_with(|| right.entropy.total_cmp(&left.entropy))
        .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
        .then_with(|| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
        })
        .then_with(|| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
        })
        .then_with(|| {
            right
                .posterior_answer_probability
                .total_cmp(&left.posterior_answer_probability)
        })
        .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
}

fn compare_suggestions(left: &Suggestion, right: &Suggestion) -> std::cmp::Ordering {
    let left_proxy = left.proxy_cost.unwrap_or(f64::INFINITY);
    let right_proxy = right.proxy_cost.unwrap_or(f64::INFINITY);
    let proxy_cmp = left_proxy.total_cmp(&right_proxy);
    if proxy_cmp != std::cmp::Ordering::Equal && !approx_tie(left_proxy, right_proxy) {
        return proxy_cmp;
    }
    compare_force_in_two(left.force_in_two, right.force_in_two)
        .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
        .then_with(|| right.entropy.total_cmp(&left.entropy))
        .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
        .then_with(|| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
        })
        .then_with(|| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
        })
        .then_with(|| {
            right
                .posterior_answer_probability
                .total_cmp(&left.posterior_answer_probability)
        })
        .then_with(|| left.word.cmp(&right.word))
}

fn compare_lookahead(left: &Suggestion, right: &Suggestion) -> std::cmp::Ordering {
    match (left.lookahead_cost, right.lookahead_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal && !approx_tie(left_cost, right_cost) {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions(left, right))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right),
    }
}

fn compare_exact(left: &Suggestion, right: &Suggestion) -> std::cmp::Ordering {
    match (left.exact_cost, right.exact_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal && !approx_tie(left_cost, right_cost) {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions(left, right))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right),
    }
}

fn compare_exact_costs(
    left: &Suggestion,
    right: &Suggestion,
    left_cost: Option<f64>,
    right_cost: Option<f64>,
) -> std::cmp::Ordering {
    match (left_cost, right_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal && !approx_tie(left_cost, right_cost) {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions(left, right))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right),
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
        PredictiveMemoMap, PredictiveSearchMode, Solver, StateDangerAssessment, Suggestion,
        compare_exact_costs, compare_guess_metrics, compare_lookahead, compare_suggestions,
        exact_suggestion_mode, predictive_search_mode,
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
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let pattern_root: PathBuf = std::env::temp_dir().join(format!(
            "maybe-wordle-solver-test-{}-{unique}",
            words.join("-")
        ));
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
            force_in_two: false,
            worst_non_green_bucket_size: 2,
            largest_non_green_bucket_mass: 0.25,
            high_mass_ambiguous_bucket_count: 1,
            proxy_cost: 1.8,
            posterior_answer_probability: 0.0,
        };
        let worse_proxy = GuessMetrics {
            guess_index: 1,
            entropy: 4.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            force_in_two: false,
            worst_non_green_bucket_size: 3,
            largest_non_green_bucket_mass: 0.35,
            high_mass_ambiguous_bucket_count: 2,
            proxy_cost: 2.2,
            posterior_answer_probability: 0.0,
        };
        assert_eq!(
            compare_guess_metrics(&better_proxy, &worse_proxy, &guesses),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn force_in_two_wins_proxy_ties_only() {
        let force = Suggestion {
            word: "alpha".into(),
            entropy: 3.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            force_in_two: true,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            proxy_cost: Some(2.0),
            posterior_answer_probability: 0.0,
            lookahead_cost: None,
            exact_cost: None,
        };
        let non_force = Suggestion {
            word: "bravo".into(),
            entropy: 3.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            force_in_two: false,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            proxy_cost: Some(2.0),
            posterior_answer_probability: 0.0,
            lookahead_cost: None,
            exact_cost: None,
        };
        assert_eq!(
            compare_suggestions(&force, &non_force),
            std::cmp::Ordering::Less
        );

        let clearly_better = Suggestion {
            proxy_cost: Some(1.9),
            ..non_force.clone()
        };
        assert_eq!(
            compare_suggestions(&clearly_better, &force),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn force_in_two_breaks_exact_cost_ties_only() {
        let force = Suggestion {
            word: "alpha".into(),
            entropy: 3.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            force_in_two: true,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            proxy_cost: Some(2.0),
            posterior_answer_probability: 0.0,
            lookahead_cost: Some(2.0),
            exact_cost: Some(3.0),
        };
        let non_force = Suggestion {
            word: "bravo".into(),
            force_in_two: false,
            ..force.clone()
        };
        assert_eq!(
            compare_exact_costs(&force, &non_force, force.exact_cost, non_force.exact_cost),
            std::cmp::Ordering::Less
        );

        let better_exact = Suggestion {
            exact_cost: Some(2.5),
            ..non_force.clone()
        };
        assert_eq!(
            compare_exact_costs(
                &better_exact,
                &force,
                better_exact.exact_cost,
                force.exact_cost
            ),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn force_in_two_does_not_beat_better_lookahead_score() {
        let force = Suggestion {
            word: "alpha".into(),
            entropy: 3.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            force_in_two: true,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            proxy_cost: Some(2.0),
            posterior_answer_probability: 0.0,
            lookahead_cost: Some(3.0),
            exact_cost: None,
        };
        let better = Suggestion {
            word: "bravo".into(),
            force_in_two: false,
            lookahead_cost: Some(2.5),
            ..force.clone()
        };
        assert_eq!(compare_lookahead(&better, &force), std::cmp::Ordering::Less);
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
                force_in_two: false,
                worst_non_green_bucket_size: 2,
                largest_non_green_bucket_mass: 0.40,
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
                force_in_two: false,
                worst_non_green_bucket_size: 2,
                largest_non_green_bucket_mass: 0.35,
                proxy_cost: Some(1.6),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
        ];

        let candidates = solver
            .collect_exact_candidates(&state, &suggestions, solver.config.exact_candidate_pool)
            .expect("candidates");
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&1));
    }

    #[test]
    fn pooled_exact_candidates_include_force_and_worst_bucket_guesses() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let state = super::SolveState {
            surviving: vec![0, 1],
            weights: vec![1.0; solver.answers.len()],
            total_weight: 2.0,
        };
        let suggestions = vec![
            Suggestion {
                word: "humph".into(),
                entropy: 5.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: false,
                worst_non_green_bucket_size: 4,
                largest_non_green_bucket_mass: 0.45,
                proxy_cost: Some(1.5),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "awake".into(),
                entropy: 4.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: true,
                worst_non_green_bucket_size: 3,
                largest_non_green_bucket_mass: 0.25,
                proxy_cost: Some(1.6),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "blush".into(),
                entropy: 3.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: false,
                worst_non_green_bucket_size: 1,
                largest_non_green_bucket_mass: 0.05,
                proxy_cost: Some(1.7),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
        ];

        let candidates = solver
            .collect_exact_candidates(&state, &suggestions, solver.config.exact_candidate_pool)
            .expect("candidates");
        assert!(candidates.contains(solver.guess_index.get("awake").expect("awake")));
        assert!(candidates.contains(solver.guess_index.get("blush").expect("blush")));
    }

    #[test]
    fn lookahead_uses_exact_recursion_for_small_children() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        solver.config.exact_threshold = 2;
        solver.config.lookahead_threshold = 4;
        let subset = vec![0, 1];
        let weights = vec![1.0; solver.answers.len()];
        let mut exact_memo = PredictiveMemoMap::default();
        let mut exact_scratch = ExactSearchScratch::new();
        let mut lookahead_memo = PredictiveMemoMap::default();

        let lookahead_value = solver
            .lookahead_child_value(
                &subset,
                &weights,
                false,
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
                &mut PredictiveMemoMap::default(),
                &mut ExactSearchScratch::new(),
                0,
            )
            .expect("exact value");
        assert!((lookahead_value - exact_value).abs() < 1e-9);
    }

    #[test]
    fn dangerous_states_escalate_but_safe_states_do_not() {
        let config = PriorConfig::default();
        let safe = StateDangerAssessment {
            danger_score: 0.30,
            dangerous_lookahead: false,
            dangerous_exact: false,
        };
        let dangerous = StateDangerAssessment {
            danger_score: 0.90,
            dangerous_lookahead: true,
            dangerous_exact: true,
        };
        assert!(matches!(
            predictive_search_mode(&config, config.lookahead_threshold + 32, safe),
            PredictiveSearchMode::ProxyOnly
        ));
        assert!(matches!(
            predictive_search_mode(&config, config.exact_threshold + 8, dangerous),
            PredictiveSearchMode::EscalatedExact
        ));
    }

    #[test]
    fn force_in_two_detects_unique_non_green_partition() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let subset = (0..solver.answers.len()).collect::<Vec<_>>();
        let weights = vec![1.0; solver.answers.len()];
        let metrics = solver.score_guess_metrics_for_subset(
            &subset,
            &weights,
            &solver.exact_small_state_table,
        );
        assert!(
            metrics.iter().any(|metric| metric.force_in_two),
            "expected at least one force-in-two witness"
        );
    }

    #[test]
    fn force_in_two_rejects_split_with_multi_answer_non_green_bucket() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let subset = (0..solver.answers.len()).collect::<Vec<_>>();
        let weights = vec![1.0; solver.answers.len()];
        let metrics = solver.score_guess_metrics_for_subset(
            &subset,
            &weights,
            &solver.exact_small_state_table,
        );
        assert!(
            metrics.iter().any(|metric| !metric.force_in_two),
            "expected at least one non-force-in-two witness"
        );
    }

    #[test]
    fn entropy_algebra_matches_probability_formula() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let subset = vec![0, 1, 2, 3];
        let weights = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let metric = solver.score_guess_metrics(
            0,
            &subset,
            &weights,
            total_weight,
            &solver.exact_small_state_table,
            0.0,
            &mut super::GuessMetricScratch::new(),
        );
        let mut masses = HashMap::<u8, f64>::new();
        for answer_index in &subset {
            let pattern = solver.pattern_table.get(0, *answer_index);
            *masses.entry(pattern).or_insert(0.0) += weights[*answer_index];
        }
        let reference = masses
            .values()
            .map(|mass| {
                let probability = *mass / total_weight;
                -(probability * probability.log2())
            })
            .sum::<f64>();
        assert!((metric.entropy - reference).abs() <= 1e-12);
    }

    #[test]
    fn detailed_run_tracks_step_diagnostics() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let run = solver
            .solve_target_from_state_detailed("cigar", Solver::today(), Solver::today(), 3)
            .expect("detailed run");
        assert!(!run.steps.is_empty());
        assert!(!run.steps[0].top_suggestions.is_empty());
        assert!(run.steps[0].danger_score >= 0.0);
    }

    #[test]
    fn hard_case_selection_includes_high_posterior_trap_when_available() {
        let solver = test_solver(&["cigar", "cigap", "cigam", "rebut", "sissy", "humph"]);
        let cases = solver
            .select_hard_case_targets(Solver::today(), 3)
            .expect("cases");
        assert!(
            cases
                .iter()
                .any(|(label, _)| label == "high_posterior_trap")
        );
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
        let mut memo = PredictiveMemoMap::default();
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
