use std::{
    array,
    collections::{HashMap, HashSet},
    fs,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{Days, NaiveDate, Utc};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, read_history_jsonl},
    model::{
        AnswerRecord, ModelVariant, WeightMode, load_model, load_model_with_variant,
        weight_snapshot_for_mode,
    },
    pattern_table::PatternTable,
    scoring::{
        ALL_GREEN_PATTERN, PATTERN_SPACE, decode_feedback, format_feedback_letters, parse_feedback,
        score_guess,
    },
    small_state::SmallStateTable,
};

const PROXY_CALIBRATION_MAX_STEPS: usize = 3;
const PROXY_CALIBRATION_MAX_CANDIDATES_PER_STATE: usize = 10;
const PROXY_CALIBRATION_MAX_SURVIVORS_FOR_FORCED_ROWS: usize = 192;
const PROXY_CALIBRATION_MAX_GAME_SECONDS: f64 = 20.0;
const HARD_MODE_WORD_LENGTH: usize = 5;

#[derive(Clone, Debug)]
pub struct Suggestion {
    pub word: String,
    pub entropy: f64,
    pub solve_probability: f64,
    pub expected_remaining: f64,
    pub force_in_two: bool,
    pub known_absent_letter_hits: usize,
    pub worst_non_green_bucket_size: usize,
    pub largest_non_green_bucket_mass: f64,
    pub large_non_green_bucket_count: usize,
    pub dangerous_mass_bucket_count: usize,
    pub non_green_mass_in_large_buckets: f64,
    pub proxy_cost: Option<f64>,
    pub large_state_score: Option<f64>,
    pub posterior_answer_probability: f64,
    pub lookahead_cost: Option<f64>,
    pub exact_cost: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct AbsurdleSuggestion {
    pub word: String,
    pub entropy: f64,
    pub largest_bucket_size: usize,
    pub second_largest_bucket_size: usize,
    pub multi_answer_bucket_count: usize,
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
    pub large_non_green_bucket_count: usize,
    pub dangerous_mass_bucket_count: usize,
    pub non_green_mass_in_large_buckets: f64,
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
    pub lookahead_pool_base: usize,
    pub lookahead_pool_size: usize,
    pub exact_pool_base: usize,
    pub exact_pool_size: usize,
    pub root_candidate_count: usize,
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
    pub session_fallback_cold_ms: f64,
    pub session_fallback_warm_ms: f64,
    pub proxy_step_pct: f64,
    pub lookahead_step_pct: f64,
    pub escalated_exact_step_pct: f64,
    pub exact_step_pct: f64,
    pub average_lookahead_pool_ratio: f64,
    pub average_exact_pool_ratio: f64,
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

#[derive(Clone, Debug)]
pub struct PredictiveAblationResult {
    pub label: String,
    pub result: ExperimentResult,
}

#[derive(Clone, Debug)]
pub struct PredictiveOpenerBuildSummary {
    pub path: PathBuf,
    pub opener: String,
    pub as_of: NaiveDate,
    pub config_fingerprint: String,
    pub games: usize,
    pub average_guesses: f64,
    pub failures: usize,
}

#[derive(Clone, Debug)]
pub struct PredictiveReplyBuildSummary {
    pub path: PathBuf,
    pub opener: String,
    pub reply_count: usize,
    pub third_reply_count: usize,
    pub as_of: NaiveDate,
    pub config_fingerprint: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct PredictiveBookIdentity {
    mode: String,
    variant: String,
    config_fingerprint: String,
    as_of: NaiveDate,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PredictiveOpenerArtifact {
    identity: PredictiveBookIdentity,
    opener: String,
    search_window_start: NaiveDate,
    search_window_end: NaiveDate,
    games: usize,
    average_guesses: f64,
    failures: usize,
    proxy_cost: Option<f64>,
    lookahead_cost: Option<f64>,
    exact_cost: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PredictiveThirdReplyEntry {
    second_feedback_pattern: u8,
    reply: String,
    surviving_answers: usize,
    proxy_cost: Option<f64>,
    lookahead_cost: Option<f64>,
    exact_cost: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PredictiveReplyEntry {
    feedback_pattern: u8,
    reply: String,
    surviving_answers: usize,
    proxy_cost: Option<f64>,
    lookahead_cost: Option<f64>,
    exact_cost: Option<f64>,
    #[serde(default)]
    third_replies: Vec<PredictiveThirdReplyEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PredictiveReplyBookArtifact {
    identity: PredictiveBookIdentity,
    opener: String,
    replies: Vec<PredictiveReplyEntry>,
}

type SessionReplyCacheKey = (PredictiveBookIdentity, String, u8);
type SessionThirdCacheKey = (PredictiveBookIdentity, String, u8, String, u8);
type SessionReplyCache = Arc<Mutex<HashMap<SessionReplyCacheKey, Option<String>>>>;
type SessionThirdCache = Arc<Mutex<HashMap<SessionThirdCacheKey, Option<String>>>>;

#[derive(Clone, Copy, Debug)]
struct ForcedOpenerEvaluation {
    guess_index: usize,
    games: usize,
    average_guesses: f64,
    p95_guesses: usize,
    max_guesses: usize,
    failures: usize,
}

#[derive(Clone, Debug)]
pub struct ProxyCalibrationRow {
    pub state_id: String,
    pub date: NaiveDate,
    pub step_index: usize,
    pub surviving_answers: usize,
    pub guess: String,
    pub entropy: f64,
    pub largest_non_green_bucket_mass: f64,
    pub worst_non_green_bucket_size: usize,
    pub high_mass_ambiguous_bucket_count: usize,
    pub proxy_cost: f64,
    pub solve_probability: f64,
    pub posterior_answer_probability: f64,
    pub smoothness_penalty: f64,
    pub known_absent_letter_hits: usize,
    pub large_non_green_bucket_count: usize,
    pub dangerous_mass_bucket_count: usize,
    pub non_green_mass_in_large_buckets: f64,
    pub realized_cost: f64,
}

#[derive(Clone, Debug)]
pub struct ProxyWeightFitSummary {
    pub row_count: usize,
    pub state_count: usize,
    pub training_average_guesses: f64,
    pub validation_average_guesses: f64,
    pub replacement_toml: String,
}

#[derive(Clone, Debug)]
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
    artifact_dir: PathBuf,
    session_opener_cache: Arc<Mutex<HashMap<PredictiveBookIdentity, Option<String>>>>,
    session_reply_cache: SessionReplyCache,
    session_third_cache: SessionThirdCache,
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
    known_absent_letter_hits: usize,
    worst_non_green_bucket_size: usize,
    largest_non_green_bucket_mass: f64,
    high_mass_ambiguous_bucket_count: usize,
    smoothness_penalty: f64,
    large_non_green_bucket_count: usize,
    dangerous_mass_bucket_count: usize,
    non_green_mass_in_large_buckets: f64,
    proxy_cost: f64,
    large_state_score: f64,
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
    lookahead_pool_base: usize,
    lookahead_pool_size: usize,
    exact_pool_base: usize,
    exact_pool_size: usize,
    root_candidate_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct PredictiveContext<'a> {
    as_of: NaiveDate,
    observations: &'a [(String, u8)],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PredictiveBookUsage {
    None,
    DiskOnly,
    Full,
}

const EXACT_SUBSET_INLINE_CAPACITY: usize = 16;
const ZERO_WEIGHT_FALLBACK_SCALE: f64 = 1e-6;

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

const REPLY_BOOK_CANDIDATE_LIMIT: usize = 24;

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
            artifact_dir: paths.derived_predictive.clone(),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
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
            let weight = if snapshot.final_weight > 0.0 {
                snapshot.final_weight
            } else if snapshot.base_weight > 0.0 {
                (snapshot.base_weight * snapshot.manual_weight * ZERO_WEIGHT_FALLBACK_SCALE)
                    .max(f64::MIN_POSITIVE)
            } else {
                0.0
            };
            if weight > 0.0 {
                weights[index] = weight;
                total_weight += weight;
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

    pub fn absurdle_initial_state(&self) -> SolveState {
        SolveState {
            surviving: (0..self.answers.len()).collect(),
            weights: vec![1.0; self.answers.len()],
            total_weight: self.answers.len() as f64,
        }
    }

    pub fn absurdle_apply_history(&self, observations: &[(String, u8)]) -> Result<SolveState> {
        let mut state = self.absurdle_initial_state();
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
        Ok(self
            .suggestion_batch_internal(state, top, None, PredictiveBookUsage::None)?
            .suggestions)
    }

    pub fn suggestions_for_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        let state = self.apply_history(as_of, observations)?;
        Ok(self
            .suggestion_batch_internal(
                &state,
                top,
                Some(PredictiveContext {
                    as_of,
                    observations,
                }),
                PredictiveBookUsage::Full,
            )?
            .suggestions)
    }

    pub fn suggestions_for_history_hard_mode(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        self.filtered_suggestions_for_history(
            as_of,
            observations,
            top,
            PredictiveBookUsage::Full,
            true,
            false,
        )
    }

    pub fn suggestions_for_history_disk_books_only(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        let state = self.apply_history(as_of, observations)?;
        Ok(self
            .suggestion_batch_internal(
                &state,
                top,
                Some(PredictiveContext {
                    as_of,
                    observations,
                }),
                PredictiveBookUsage::DiskOnly,
            )?
            .suggestions)
    }

    pub fn suggestions_for_history_disk_books_only_with_filters(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
        hard_mode: bool,
        force_in_two_only: bool,
    ) -> Result<Vec<Suggestion>> {
        self.filtered_suggestions_for_history(
            as_of,
            observations,
            top,
            PredictiveBookUsage::DiskOnly,
            hard_mode,
            force_in_two_only,
        )
    }

    pub fn force_in_two_suggestions_for_history_disk_books_only(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        self.suggestions_for_history_disk_books_only_with_filters(
            as_of,
            observations,
            top,
            false,
            true,
        )
    }

    pub fn absurdle_suggestions(
        &self,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<AbsurdleSuggestion>> {
        let state = self.absurdle_apply_history(observations)?;
        self.absurdle_suggestions_for_state(&state, top)
    }

    pub fn absurdle_suggestions_for_state(
        &self,
        state: &SolveState,
        top: usize,
    ) -> Result<Vec<AbsurdleSuggestion>> {
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        let total = state.surviving.len() as f64;
        let mut suggestions = (0..self.guesses.len())
            .into_par_iter()
            .map(|guess_index| self.absurdle_score_guess(guess_index, &state.surviving, total))
            .collect::<Vec<_>>();
        suggestions.sort_by(|left, right| compare_absurdle_suggestions(left, right));
        suggestions.truncate(top.min(suggestions.len()));
        Ok(suggestions)
    }

    pub fn hard_mode_violation(
        &self,
        observations: &[(String, u8)],
        guess: &str,
    ) -> Option<String> {
        hard_mode_violation(observations, guess)
    }

    fn suggestion_batch_for_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        state: &SolveState,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<SuggestionBatch> {
        self.suggestion_batch_internal(
            state,
            top,
            Some(PredictiveContext {
                as_of,
                observations,
            }),
            book_usage,
        )
    }

    fn filtered_suggestions_for_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
        book_usage: PredictiveBookUsage,
        hard_mode: bool,
        force_in_two_only: bool,
    ) -> Result<Vec<Suggestion>> {
        let state = self.apply_history(as_of, observations)?;
        let limit = if hard_mode || force_in_two_only {
            self.guesses.len()
        } else {
            top
        };
        let mut suggestions = self
            .suggestion_batch_internal(
                &state,
                limit,
                Some(PredictiveContext {
                    as_of,
                    observations,
                }),
                book_usage,
            )?
            .suggestions;
        if hard_mode {
            suggestions.retain(|suggestion| {
                self.hard_mode_violation(observations, &suggestion.word)
                    .is_none()
            });
        }
        if force_in_two_only {
            suggestions.retain(|suggestion| suggestion.force_in_two);
        }
        suggestions.truncate(top.min(suggestions.len()));
        Ok(suggestions)
    }

    fn suggestion_batch_internal(
        &self,
        state: &SolveState,
        top: usize,
        context: Option<PredictiveContext<'_>>,
        book_usage: PredictiveBookUsage,
    ) -> Result<SuggestionBatch> {
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        let split_first = state.surviving.len() > self.config.large_state_split_threshold;
        let known_absent_mask = context
            .as_ref()
            .map(|context| known_absent_letter_mask(context.observations))
            .unwrap_or(0);
        let mut metrics = self.score_guess_metrics_for_subset(
            &state.surviving,
            &state.weights,
            &self.exact_small_state_table,
        );
        if known_absent_mask != 0 {
            for metric in &mut metrics {
                metric.known_absent_letter_hits =
                    count_masked_letters(&self.guesses[metric.guess_index], known_absent_mask);
                metric.large_state_score = proxy_row_score_from_weights(
                    &self.config.proxy_weights,
                    metric.entropy,
                    metric.largest_non_green_bucket_mass,
                    metric.worst_non_green_bucket_size,
                    metric.high_mass_ambiguous_bucket_count,
                    metric.proxy_cost,
                    metric.solve_probability,
                    metric.posterior_answer_probability,
                    metric.smoothness_penalty,
                    metric.known_absent_letter_hits,
                    metric.large_non_green_bucket_count,
                    metric.dangerous_mass_bucket_count,
                    metric.non_green_mass_in_large_buckets,
                );
            }
        }
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
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
                known_absent_letter_hits: metric.known_absent_letter_hits,
                worst_non_green_bucket_size: metric.worst_non_green_bucket_size,
                largest_non_green_bucket_mass: metric.largest_non_green_bucket_mass,
                large_non_green_bucket_count: metric.large_non_green_bucket_count,
                dangerous_mass_bucket_count: metric.dangerous_mass_bucket_count,
                non_green_mass_in_large_buckets: metric.non_green_mass_in_large_buckets,
                proxy_cost: Some(metric.proxy_cost),
                large_state_score: Some(metric.large_state_score),
                posterior_answer_probability: metric.posterior_answer_probability,
                lookahead_cost: None,
                exact_cost: None,
            })
            .collect::<Vec<_>>();
        let lookahead_pool_base = self.lookahead_candidate_pool_for_state(state.surviving.len());
        let lookahead_pool = self.expanded_pool_size(
            &suggestions,
            lookahead_pool_base,
            split_first,
            matches!(search_mode, PredictiveSearchMode::Lookahead)
                && state.surviving.len() > self.config.large_state_split_threshold,
            assessment,
        );
        let exact_pool = self.expanded_pool_size(
            &suggestions,
            self.config.exact_candidate_pool,
            split_first,
            matches!(search_mode, PredictiveSearchMode::EscalatedExact)
                && state.surviving.len() > self.config.exact_threshold,
            assessment,
        );
        let mut root_candidate_count = 0usize;

        if let PredictiveSearchMode::Lookahead = search_mode {
            let root_candidates = self.collect_lookahead_candidates(
                &suggestions,
                state.surviving.len(),
                assessment.dangerous_lookahead,
                lookahead_pool,
            )?;
            root_candidate_count = root_candidates.len();
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
            suggestions.sort_by(|left, right| compare_lookahead(left, right, split_first));
        }

        if let PredictiveSearchMode::Exact(exact_mode) = search_mode {
            let exact_candidates = match exact_mode {
                ExactSuggestionMode::Exhaustive => (0..self.guesses.len()).collect::<Vec<_>>(),
                ExactSuggestionMode::Pooled => {
                    self.collect_exact_candidates(state, &suggestions, exact_pool)?
                }
            };
            root_candidate_count = exact_candidates.len();
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
                    suggestions.sort_by(|left, right| compare_exact(left, right, split_first));
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
                        compare_exact_costs(left, right, left_cost, right_cost, split_first)
                    });
                }
            }
        }

        if let PredictiveSearchMode::EscalatedExact = search_mode {
            let exact_candidates = self.collect_exact_candidates(
                state,
                &suggestions,
                self.config.danger_exact_root_pool.max(1).max(exact_pool),
            )?;
            root_candidate_count = exact_candidates.len();
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
                compare_exact_costs(left, right, left_cost, right_cost, split_first)
            });
        }

        if book_usage != PredictiveBookUsage::None {
            if let Some(context) = context {
                if let Some(cached_word) = self.cached_predictive_choice(
                    context.as_of,
                    context.observations,
                    book_usage == PredictiveBookUsage::Full,
                ) {
                    promote_cached_suggestion(&mut suggestions, &cached_word);
                }
            }
        }

        suggestions.truncate(top);
        Ok(SuggestionBatch {
            suggestions,
            danger_score: assessment.danger_score,
            danger_escalated: matches!(search_mode, PredictiveSearchMode::EscalatedExact)
                || (matches!(search_mode, PredictiveSearchMode::Lookahead)
                    && assessment.dangerous_lookahead),
            regime_used: regime_from_search_mode(search_mode),
            lookahead_pool_base,
            lookahead_pool_size: lookahead_pool,
            exact_pool_base: self.config.exact_candidate_pool,
            exact_pool_size: exact_pool,
            root_candidate_count,
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
        self.solve_target_from_state_detailed(target, as_of, date, top, PredictiveBookUsage::Full)
    }

    fn solve_target_from_state_detailed(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedSolveRun> {
        let target = target.to_ascii_lowercase();
        let mut state = self.initial_state(as_of);
        let mut observations = Vec::new();

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
            let batch = self.suggestion_batch_for_history(
                as_of,
                &observations,
                &state,
                top.max(1),
                book_usage,
            )?;
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
                lookahead_pool_base: batch.lookahead_pool_base,
                lookahead_pool_size: batch.lookahead_pool_size,
                exact_pool_base: batch.exact_pool_base,
                exact_pool_size: batch.exact_pool_size,
                root_candidate_count: batch.root_candidate_count,
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
            observations.push((chosen.word.clone(), feedback));
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
            let as_of = entry
                .print_date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
            let run = self.solve_target_from_state_detailed(
                &entry.solution,
                as_of,
                entry.print_date,
                top,
                PredictiveBookUsage::DiskOnly,
            )?;
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
            let run = self.solve_target_from_state_detailed(
                &target,
                as_of,
                as_of,
                top,
                PredictiveBookUsage::DiskOnly,
            )?;
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
        let mut lookahead_pool_ratio_sum = 0.0;
        let mut lookahead_pool_ratio_count = 0usize;
        let mut exact_pool_ratio_sum = 0.0;
        let mut exact_pool_ratio_count = 0usize;
        for run in &detailed.runs {
            for step in &run.steps {
                if step.lookahead_pool_base > 0 && step.lookahead_pool_size > 0 {
                    lookahead_pool_ratio_sum +=
                        step.lookahead_pool_size as f64 / step.lookahead_pool_base as f64;
                    lookahead_pool_ratio_count += 1;
                }
                if step.exact_pool_base > 0 && step.exact_pool_size > 0 {
                    exact_pool_ratio_sum +=
                        step.exact_pool_size as f64 / step.exact_pool_base as f64;
                    exact_pool_ratio_count += 1;
                }
            }
        }
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
                "et{}-ee{}-cp{}-lt{}-lc{}-lr{}-ls{}",
                self.config.exact_threshold,
                self.config.exact_exhaustive_threshold,
                self.config.exact_candidate_pool,
                self.config.lookahead_threshold,
                self.config.lookahead_candidate_pool,
                self.config.lookahead_reply_pool,
                self.config.large_state_split_threshold,
            ),
            mode: self.mode,
            variant: self.variant,
            backtest,
            average_log_loss: total_log_loss / divisor,
            average_brier: total_brier / divisor,
            average_target_probability: total_target_probability / divisor,
            average_target_rank: total_rank / divisor,
            latency_p95_ms: self.benchmark_predictive_latency(5)?,
            session_fallback_cold_ms: 0.0,
            session_fallback_warm_ms: 0.0,
            proxy_step_pct,
            lookahead_step_pct,
            escalated_exact_step_pct,
            exact_step_pct,
            average_lookahead_pool_ratio: if lookahead_pool_ratio_count == 0 {
                0.0
            } else {
                lookahead_pool_ratio_sum / lookahead_pool_ratio_count as f64
            },
            average_exact_pool_ratio: if exact_pool_ratio_count == 0 {
                0.0
            } else {
                exact_pool_ratio_sum / exact_pool_ratio_count as f64
            },
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

    pub fn build_predictive_opener_cache(
        &self,
        as_of: NaiveDate,
    ) -> Result<PredictiveOpenerBuildSummary> {
        let offline = self.offline_book_solver();
        let (window_start, window_end, targets) =
            offline.recent_history_targets_for_books(as_of)?;
        let state = offline.initial_state(as_of);
        let opener_guess = offline
            .suggestion_batch_internal(
                &state,
                1,
                Some(PredictiveContext {
                    as_of,
                    observations: &[],
                }),
                PredictiveBookUsage::None,
            )?
            .suggestions
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("missing predictive opener candidate"))?;
        let opener_guess_index = offline
            .guess_index
            .get(&opener_guess.word)
            .copied()
            .ok_or_else(|| anyhow!("missing opener guess {}", opener_guess.word))?;
        let opener_eval = offline.evaluate_forced_opener(as_of, &targets, opener_guess_index, 5)?;
        let opener = offline.guesses[opener_eval.guess_index].clone();
        let artifact = PredictiveOpenerArtifact {
            identity: self.predictive_book_identity(as_of),
            opener: opener.clone(),
            search_window_start: window_start,
            search_window_end: window_end,
            games: opener_eval.games,
            average_guesses: opener_eval.average_guesses,
            failures: opener_eval.failures,
            proxy_cost: None,
            lookahead_cost: None,
            exact_cost: None,
        };
        let path = self.opener_artifact_path(as_of);
        write_predictive_artifact(&path, &artifact)?;
        Ok(PredictiveOpenerBuildSummary {
            path,
            opener: artifact.opener,
            as_of,
            config_fingerprint: artifact.identity.config_fingerprint,
            games: artifact.games,
            average_guesses: artifact.average_guesses,
            failures: artifact.failures,
        })
    }

    pub fn build_predictive_reply_book(
        &self,
        as_of: NaiveDate,
    ) -> Result<PredictiveReplyBuildSummary> {
        let opener_artifact = self
            .load_predictive_opener_artifact(as_of)?
            .ok_or_else(|| anyhow!("build the predictive opener cache first"))?;
        let opener_index = self
            .guess_index
            .get(&opener_artifact.opener)
            .copied()
            .ok_or_else(|| anyhow!("cached opener is not in the current guess list"))?;
        let offline = self.offline_book_solver();
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let root = offline.initial_state(as_of);
        let mut seen_patterns = HashSet::new();
        let mut replies = Vec::new();

        for answer_index in &root.surviving {
            let pattern = offline.pattern_table.get(opener_index, *answer_index);
            if pattern == ALL_GREEN_PATTERN || !seen_patterns.insert(pattern) {
                continue;
            }
            let mut child = root.clone();
            offline.apply_feedback(&mut child, &opener_artifact.opener, pattern)?;
            if child.surviving.len() <= 1 {
                continue;
            }
            let scoped_targets = targets
                .iter()
                .filter(|(_, target)| score_guess(&opener_artifact.opener, target) == pattern)
                .cloned()
                .collect::<Vec<_>>();
            if scoped_targets.is_empty() {
                continue;
            }
            let observation = [(opener_artifact.opener.clone(), pattern)];
            let batch = offline.suggestion_batch_internal(
                &child,
                REPLY_BOOK_CANDIDATE_LIMIT,
                Some(PredictiveContext {
                    as_of,
                    observations: &observation,
                }),
                PredictiveBookUsage::None,
            )?;
            let mut best_reply: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
            for suggestion in batch
                .suggestions
                .into_iter()
                .take(REPLY_BOOK_CANDIDATE_LIMIT)
            {
                let guess_index = offline
                    .guess_index
                    .get(&suggestion.word)
                    .copied()
                    .ok_or_else(|| anyhow!("missing reply guess {}", suggestion.word))?;
                let evaluation = offline.evaluate_forced_reply(
                    &opener_artifact.opener,
                    pattern,
                    &scoped_targets,
                    guess_index,
                    5,
                )?;
                if best_reply.as_ref().is_none_or(|(_, current)| {
                    compare_forced_openers(&evaluation, current, &offline.guesses)
                        == std::cmp::Ordering::Less
                }) {
                    best_reply = Some((suggestion, evaluation));
                }
            }
            if let Some((reply, _)) = best_reply {
                let reply_word = reply.word.clone();
                let reply_index = offline
                    .guess_index
                    .get(&reply_word)
                    .copied()
                    .ok_or_else(|| anyhow!("missing reply guess {}", reply_word))?;
                let mut seen_second_patterns = HashSet::new();
                let mut grandchild = child.clone();
                let mut third_replies = Vec::new();
                for target_index in &child.surviving {
                    let second_feedback = offline.pattern_table.get(reply_index, *target_index);
                    if second_feedback == ALL_GREEN_PATTERN
                        || !seen_second_patterns.insert(second_feedback)
                    {
                        continue;
                    }
                    grandchild.clone_from(&child);
                    offline.apply_feedback(&mut grandchild, &reply_word, second_feedback)?;
                    if grandchild.surviving.len() <= 1 {
                        continue;
                    }
                    let grand_targets = scoped_targets
                        .iter()
                        .filter(|(_, target)| score_guess(&reply_word, target) == second_feedback)
                        .cloned()
                        .collect::<Vec<_>>();
                    if grand_targets.is_empty() {
                        continue;
                    }
                    let grand_observations = [
                        (opener_artifact.opener.clone(), pattern),
                        (reply_word.clone(), second_feedback),
                    ];
                    let grand_batch = offline.suggestion_batch_internal(
                        &grandchild,
                        REPLY_BOOK_CANDIDATE_LIMIT,
                        Some(PredictiveContext {
                            as_of,
                            observations: &grand_observations,
                        }),
                        PredictiveBookUsage::None,
                    )?;
                    let mut best_third: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
                    for suggestion in grand_batch
                        .suggestions
                        .into_iter()
                        .take(REPLY_BOOK_CANDIDATE_LIMIT)
                    {
                        let guess_index = offline
                            .guess_index
                            .get(&suggestion.word)
                            .copied()
                            .ok_or_else(|| anyhow!("missing third guess {}", suggestion.word))?;
                        let evaluation = offline.evaluate_forced_continuation(
                            &[opener_artifact.opener.clone(), reply_word.clone()],
                            &grand_targets,
                            guess_index,
                            5,
                        )?;
                        if best_third.as_ref().is_none_or(|(_, current)| {
                            compare_forced_openers(&evaluation, current, &offline.guesses)
                                == std::cmp::Ordering::Less
                        }) {
                            best_third = Some((suggestion, evaluation));
                        }
                    }
                    if let Some((third, _)) = best_third {
                        third_replies.push(PredictiveThirdReplyEntry {
                            second_feedback_pattern: second_feedback,
                            reply: third.word,
                            surviving_answers: grandchild.surviving.len(),
                            proxy_cost: third.proxy_cost,
                            lookahead_cost: third.lookahead_cost,
                            exact_cost: third.exact_cost,
                        });
                    }
                }
                third_replies.sort_by(|left, right| {
                    left.second_feedback_pattern
                        .cmp(&right.second_feedback_pattern)
                });
                replies.push(PredictiveReplyEntry {
                    feedback_pattern: pattern,
                    reply: reply_word,
                    surviving_answers: child.surviving.len(),
                    proxy_cost: reply.proxy_cost,
                    lookahead_cost: reply.lookahead_cost,
                    exact_cost: reply.exact_cost,
                    third_replies,
                });
            }
        }
        replies.sort_by(|left, right| left.feedback_pattern.cmp(&right.feedback_pattern));
        let artifact = PredictiveReplyBookArtifact {
            identity: self.predictive_book_identity(as_of),
            opener: opener_artifact.opener.clone(),
            replies,
        };
        let path = self.reply_book_artifact_path(as_of);
        write_predictive_artifact(&path, &artifact)?;
        let third_reply_count = artifact
            .replies
            .iter()
            .map(|entry| entry.third_replies.len())
            .sum();
        Ok(PredictiveReplyBuildSummary {
            path,
            opener: artifact.opener,
            reply_count: artifact.replies.len(),
            third_reply_count,
            as_of,
            config_fingerprint: artifact.identity.config_fingerprint,
        })
    }

    pub fn predictive_ablation_report(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<Vec<PredictiveAblationResult>> {
        let mut rows = Vec::new();
        let expanded = |mut candidate: PriorConfig| {
            candidate.lookahead_candidate_pool = candidate.lookahead_candidate_pool.max(150);
            candidate.lookahead_reply_pool = candidate.lookahead_reply_pool.max(50);
            candidate.exact_candidate_pool = candidate.exact_candidate_pool.max(160);
            candidate
        };
        let flattened_50 = flatten_weighted_config(config, 0.50);
        let flattened_25 = flatten_weighted_config(config, 0.25);
        for (label, mode, candidate) in [
            ("uniform_baseline", WeightMode::Uniform, config.clone()),
            (
                "uniform_wide_pools",
                WeightMode::Uniform,
                expanded(config.clone()),
            ),
            (
                "cooldown_baseline",
                WeightMode::CooldownOnly,
                config.clone(),
            ),
            (
                "cooldown_wide_pools",
                WeightMode::CooldownOnly,
                expanded(config.clone()),
            ),
            ("weighted_baseline", WeightMode::Weighted, config.clone()),
            (
                "weighted_wide_pools",
                WeightMode::Weighted,
                expanded(config.clone()),
            ),
            (
                "weighted_flat_50",
                WeightMode::Weighted,
                flattened_50.clone(),
            ),
            (
                "weighted_flat_50_wide_pools",
                WeightMode::Weighted,
                expanded(flattened_50),
            ),
            (
                "weighted_flat_25",
                WeightMode::Weighted,
                flattened_25.clone(),
            ),
            (
                "weighted_flat_25_wide_pools",
                WeightMode::Weighted,
                expanded(flattened_25),
            ),
        ] {
            let solver = Self::from_paths_with_settings(
                paths,
                &candidate,
                mode,
                ModelVariant::SeedPlusHistory,
            )?;
            rows.push(PredictiveAblationResult {
                label: label.to_string(),
                result: solver.experiment_report(from, to, top)?,
            });
        }
        Ok(rows)
    }

    pub fn build_proxy_calibration_set(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<Vec<ProxyCalibrationRow>> {
        let started = Instant::now();
        let emit_progress = |message: String| {
            eprintln!("{message}");
            let _ = std::io::stderr().flush();
        };
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no games found in the requested calibration range");
        }

        let mut rows = Vec::new();
        let total_games = games.len();
        emit_progress(format!(
            "fit-proxy-weights phase=calibration-start games={} from={} to={} elapsed_s=0.0",
            total_games, from, to
        ));
        for (game_index, entry) in games.into_iter().enumerate() {
            let date = entry.print_date;
            let as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot calibrate before launch date"))?;
            let target = entry.solution.to_ascii_lowercase();
            let mut state = self.initial_state(as_of);
            if !state
                .surviving
                .iter()
                .any(|index| self.answers[*index].word == target)
            {
                continue;
            }

            let mut observations = Vec::new();
            let mut step_index = 0usize;
            let game_started = Instant::now();
            while step_index < PROXY_CALIBRATION_MAX_STEPS
                && state.surviving.len() > self.config.large_state_split_threshold
            {
                if game_started.elapsed().as_secs_f64() > PROXY_CALIBRATION_MAX_GAME_SECONDS {
                    emit_progress(format!(
                        "fit-proxy-weights phase=calibration-skip game={}/{} date={} reason=budget rows={} elapsed_s={:.1}",
                        game_index + 1,
                        total_games,
                        date,
                        rows.len(),
                        started.elapsed().as_secs_f64(),
                    ));
                    break;
                }
                let mut metrics = self.score_guess_metrics_for_subset(
                    &state.surviving,
                    &state.weights,
                    &self.exact_small_state_table,
                );
                let known_absent_mask = known_absent_letter_mask(&observations);
                for metric in &mut metrics {
                    metric.known_absent_letter_hits =
                        count_masked_letters(&self.guesses[metric.guess_index], known_absent_mask);
                    metric.large_state_score = proxy_row_score_from_weights(
                        &self.config.proxy_weights,
                        metric.entropy,
                        metric.largest_non_green_bucket_mass,
                        metric.worst_non_green_bucket_size,
                        metric.high_mass_ambiguous_bucket_count,
                        metric.proxy_cost,
                        metric.solve_probability,
                        metric.posterior_answer_probability,
                        metric.smoothness_penalty,
                        metric.known_absent_letter_hits,
                        metric.large_non_green_bucket_count,
                        metric.dangerous_mass_bucket_count,
                        metric.non_green_mass_in_large_buckets,
                    );
                }
                metrics.sort_by(|left, right| {
                    compare_guess_metrics_for_state(left, right, &self.guesses, true)
                });

                let state_id = format!("{date}:{step_index}");
                let candidate_limit =
                    if state.surviving.len() <= PROXY_CALIBRATION_MAX_SURVIVORS_FOR_FORCED_ROWS {
                        PROXY_CALIBRATION_MAX_CANDIDATES_PER_STATE.min(metrics.len())
                    } else {
                        0
                    };
                if candidate_limit == 0 {
                    emit_progress(format!(
                        "fit-proxy-weights phase=calibration-step game={}/{} date={} step={} survivors={} candidates=0 reason=survivor-cap elapsed_s={:.1}",
                        game_index + 1,
                        total_games,
                        date,
                        step_index,
                        state.surviving.len(),
                        started.elapsed().as_secs_f64(),
                    ));
                } else {
                    emit_progress(format!(
                        "fit-proxy-weights phase=calibration-step game={}/{} date={} step={} survivors={} candidates={} elapsed_s={:.1}",
                        game_index + 1,
                        total_games,
                        date,
                        step_index,
                        state.surviving.len(),
                        candidate_limit,
                        started.elapsed().as_secs_f64(),
                    ));
                }
                for metric in metrics.iter().take(candidate_limit) {
                    if game_started.elapsed().as_secs_f64() > PROXY_CALIBRATION_MAX_GAME_SECONDS {
                        emit_progress(format!(
                            "fit-proxy-weights phase=calibration-skip game={}/{} date={} reason=budget rows={} elapsed_s={:.1}",
                            game_index + 1,
                            total_games,
                            date,
                            rows.len(),
                            started.elapsed().as_secs_f64(),
                        ));
                        break;
                    }
                    let guess = self.guesses[metric.guess_index].clone();
                    let mut forced = observations.clone();
                    forced.push((guess.clone(), 0));
                    let run =
                        self.solve_target_with_forced_prefix(&target, as_of, date, &forced, 3)?;
                    let realized_cost = if run.solved {
                        run.steps.len().saturating_sub(observations.len()) as f64
                    } else {
                        7.0
                    };
                    rows.push(ProxyCalibrationRow {
                        state_id: state_id.clone(),
                        date,
                        step_index,
                        surviving_answers: state.surviving.len(),
                        guess,
                        entropy: metric.entropy,
                        largest_non_green_bucket_mass: metric.largest_non_green_bucket_mass,
                        worst_non_green_bucket_size: metric.worst_non_green_bucket_size,
                        high_mass_ambiguous_bucket_count: metric.high_mass_ambiguous_bucket_count,
                        proxy_cost: metric.proxy_cost,
                        solve_probability: metric.solve_probability,
                        posterior_answer_probability: metric.posterior_answer_probability,
                        smoothness_penalty: metric.smoothness_penalty,
                        known_absent_letter_hits: metric.known_absent_letter_hits,
                        large_non_green_bucket_count: metric.large_non_green_bucket_count,
                        dangerous_mass_bucket_count: metric.dangerous_mass_bucket_count,
                        non_green_mass_in_large_buckets: metric.non_green_mass_in_large_buckets,
                        realized_cost,
                    });
                }

                let chosen = metrics
                    .first()
                    .ok_or_else(|| anyhow!("missing top calibration guess"))?;
                let guess = self.guesses[chosen.guess_index].clone();
                let feedback = score_guess(&guess, &target);
                observations.push((guess.clone(), feedback));
                if feedback == ALL_GREEN_PATTERN {
                    break;
                }
                self.apply_feedback(&mut state, &guess, feedback)?;
                step_index += 1;
            }

            if game_index < 3 || (game_index + 1) % 10 == 0 || game_index + 1 == total_games {
                emit_progress(format!(
                    "fit-proxy-weights phase=calibration games={}/{} rows={} elapsed_s={:.1}",
                    game_index + 1,
                    total_games,
                    rows.len(),
                    started.elapsed().as_secs_f64(),
                ));
            }
        }
        Ok(rows)
    }

    pub fn fit_proxy_weights(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<ProxyWeightFitSummary> {
        let started = Instant::now();
        let emit_progress = |message: String| {
            eprintln!("{message}");
            let _ = std::io::stderr().flush();
        };
        emit_progress(format!(
            "fit-proxy-weights phase=start from={} to={} elapsed_s=0.0",
            from, to
        ));
        let rows = self.build_proxy_calibration_set(from, to)?;
        if rows.is_empty() {
            bail!("no large-state calibration rows found in the requested range");
        }
        emit_progress(format!(
            "fit-proxy-weights phase=grouping rows={} elapsed_s={:.1}",
            rows.len(),
            started.elapsed().as_secs_f64(),
        ));

        let mut grouped = std::collections::BTreeMap::<String, Vec<&ProxyCalibrationRow>>::new();
        for row in &rows {
            grouped.entry(row.state_id.clone()).or_default().push(row);
        }
        let groups = grouped.into_iter().collect::<Vec<_>>();
        let split = ((groups.len() as f64) * 0.8).floor() as usize;
        let split = split.clamp(1, groups.len());
        let (train_groups, validation_groups) = groups.split_at(split);

        let evaluate = |weights: &crate::config::ProxyWeights,
                        sample: &[(String, Vec<&ProxyCalibrationRow>)]|
         -> f64 {
            if sample.is_empty() {
                return 0.0;
            }
            sample
                .iter()
                .map(|(_, rows)| {
                    rows.iter()
                        .max_by(|left, right| {
                            let left_score = proxy_row_score_from_weights(
                                weights,
                                left.entropy,
                                left.largest_non_green_bucket_mass,
                                left.worst_non_green_bucket_size,
                                left.high_mass_ambiguous_bucket_count,
                                left.proxy_cost,
                                left.solve_probability,
                                left.posterior_answer_probability,
                                left.smoothness_penalty,
                                left.known_absent_letter_hits,
                                left.large_non_green_bucket_count,
                                left.dangerous_mass_bucket_count,
                                left.non_green_mass_in_large_buckets,
                            );
                            let right_score = proxy_row_score_from_weights(
                                weights,
                                right.entropy,
                                right.largest_non_green_bucket_mass,
                                right.worst_non_green_bucket_size,
                                right.high_mass_ambiguous_bucket_count,
                                right.proxy_cost,
                                right.solve_probability,
                                right.posterior_answer_probability,
                                right.smoothness_penalty,
                                right.known_absent_letter_hits,
                                right.large_non_green_bucket_count,
                                right.dangerous_mass_bucket_count,
                                right.non_green_mass_in_large_buckets,
                            );
                            right_score.total_cmp(&left_score)
                        })
                        .map(|row| row.realized_cost)
                        .unwrap_or(7.0)
                })
                .sum::<f64>()
                / sample.len() as f64
        };

        emit_progress(format!(
            "fit-proxy-weights phase=search train_states={} validation_states={} baseline_train_avg={:.4} elapsed_s={:.1}",
            train_groups.len(),
            validation_groups.len(),
            evaluate(&self.config.proxy_weights, train_groups),
            started.elapsed().as_secs_f64(),
        ));

        let mut best = self.config.proxy_weights.clone();
        let mut best_train = evaluate(&best, train_groups);
        macro_rules! search_weight {
            ($field:ident) => {{
                emit_progress(format!(
                    "fit-proxy-weights phase=field name={} current={:.4} train_avg={:.4} elapsed_s={:.1}",
                    stringify!($field),
                    best.$field,
                    best_train,
                    started.elapsed().as_secs_f64(),
                ));
                for _ in 0..2 {
                    let mut improved = false;
                    let base = best.$field;
                    for scale in [0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00] {
                        let candidate_value = (base * scale).max(0.001);
                        if (candidate_value - base).abs() <= 1e-9 {
                            continue;
                        }
                        let mut candidate = best.clone();
                        candidate.$field = candidate_value;
                        let train_score = evaluate(&candidate, train_groups);
                        if train_score + 1e-9 < best_train {
                            best = candidate;
                            best_train = train_score;
                            improved = true;
                            emit_progress(format!(
                                "fit-proxy-weights phase=improved name={} value={:.4} train_avg={:.4} elapsed_s={:.1}",
                                stringify!($field),
                                best.$field,
                                best_train,
                                started.elapsed().as_secs_f64(),
                            ));
                        }
                    }
                    if !improved {
                        break;
                    }
                }
            }};
        }

        search_weight!(entropy_w);
        search_weight!(bucket_mass_w);
        search_weight!(bucket_size_w);
        search_weight!(ambiguous_w);
        search_weight!(proxy_w);
        search_weight!(solve_prob_w);
        search_weight!(posterior_w);
        search_weight!(smoothness_w);
        search_weight!(gray_reuse_w);
        search_weight!(large_bucket_count_w);
        search_weight!(dangerous_mass_count_w);
        search_weight!(large_bucket_mass_w);

        let validation_average = evaluate(&best, validation_groups);
        emit_progress(format!(
            "fit-proxy-weights phase=done train_avg={:.4} validation_avg={:.4} elapsed_s={:.1}",
            best_train,
            validation_average,
            started.elapsed().as_secs_f64(),
        ));
        let replacement_toml = format!(
            concat!(
                "[proxy_weights]\n",
                "entropy_w = {:.4}\n",
                "bucket_mass_w = {:.4}\n",
                "bucket_size_w = {:.4}\n",
                "ambiguous_w = {:.4}\n",
                "proxy_w = {:.4}\n",
                "solve_prob_w = {:.4}\n",
                "posterior_w = {:.4}\n",
                "smoothness_w = {:.4}\n",
                "gray_reuse_w = {:.4}\n",
                "large_bucket_count_w = {:.4}\n",
                "dangerous_mass_count_w = {:.4}\n",
                "large_bucket_mass_w = {:.4}\n",
            ),
            best.entropy_w,
            best.bucket_mass_w,
            best.bucket_size_w,
            best.ambiguous_w,
            best.proxy_w,
            best.solve_prob_w,
            best.posterior_w,
            best.smoothness_w,
            best.gray_reuse_w,
            best.large_bucket_count_w,
            best.dangerous_mass_count_w,
            best.large_bucket_mass_w,
        );

        Ok(ProxyWeightFitSummary {
            row_count: rows.len(),
            state_count: train_groups.len() + validation_groups.len(),
            training_average_guesses: best_train,
            validation_average_guesses: validation_average,
            replacement_toml,
        })
    }

    fn snapshot_suggestion(suggestion: &Suggestion) -> SuggestionSnapshot {
        SuggestionSnapshot {
            word: suggestion.word.clone(),
            force_in_two: suggestion.force_in_two,
            worst_non_green_bucket_size: suggestion.worst_non_green_bucket_size,
            largest_non_green_bucket_mass: suggestion.largest_non_green_bucket_mass,
            large_non_green_bucket_count: suggestion.large_non_green_bucket_count,
            dangerous_mass_bucket_count: suggestion.dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets: suggestion.non_green_mass_in_large_buckets,
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
        use std::io::Write;

        let (history_start, history_end) = Self::latest_history_range(paths)?
            .ok_or_else(|| anyhow!("run sync-data before tune-prior"))?;
        let window_end = history_end;
        let window_start = history_end
            .checked_sub_days(Days::new(364))
            .map_or(history_start, |date| date.max(history_start));
        let started = Instant::now();
        eprintln!(
            "tune-prior phase=start search_window={}..{} elapsed_s=0.0",
            window_start, window_end
        );
        let _ = std::io::stderr().flush();
        let mut best_prior_config = config.clone();
        let mut best_prior = Self::evaluate_prior_search_candidate(
            paths,
            &best_prior_config,
            window_start,
            window_end,
        )?;

        macro_rules! search_dimension {
            ($field:ident, $label:literal, $values:expr) => {{
                eprintln!(
                    "tune-prior phase=field name={} current={} elapsed_s={:.1}",
                    $label,
                    best_prior_config.$field,
                    started.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
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
                            eprintln!(
                                "tune-prior phase=improved name={} value={} log_loss={:.6} target_rank={:.2} target_probability={:.6} elapsed_s={:.1}",
                                $label,
                                best_prior_config.$field,
                                best_prior.average_log_loss,
                                best_prior.average_target_rank,
                                best_prior.average_target_probability,
                                started.elapsed().as_secs_f64()
                            );
                            let _ = std::io::stderr().flush();
                            improved = true;
                        }
                    }
                    if !improved {
                        break;
                    }
                }
            }};
        }

        search_dimension!(base_seed_weight, "base_seed_weight", [0.75, 1.0, 1.25]);
        search_dimension!(
            base_history_only_weight,
            "base_history_only_weight",
            [0.10, 0.20, 0.25, 0.33, 0.50]
        );
        search_dimension!(cooldown_days, "cooldown_days", [90_i64, 120, 180, 240, 365]);
        search_dimension!(cooldown_floor, "cooldown_floor", [0.0, 0.01, 0.02, 0.05]);
        search_dimension!(
            midpoint_days,
            "midpoint_days",
            [365.0, 540.0, 720.0, 900.0, 1080.0]
        );
        search_dimension!(logistic_k, "logistic_k", [0.005, 0.01, 0.015, 0.02]);
        search_dimension!(
            pool_tight_gap_threshold,
            "pool_tight_gap_threshold",
            [0.03, 0.05, 0.07]
        );
        search_dimension!(
            pool_medium_gap_threshold,
            "pool_medium_gap_threshold",
            [0.10, 0.15, 0.20]
        );
        search_dimension!(
            danger_lookahead_threshold,
            "danger_lookahead_threshold",
            [0.52, 0.58, 0.64]
        );
        search_dimension!(
            danger_exact_threshold,
            "danger_exact_threshold",
            [0.68, 0.72, 0.76]
        );
        search_dimension!(
            lookahead_trap_penalty,
            "lookahead_trap_penalty",
            [0.25, 0.35, 0.45]
        );
        search_dimension!(
            lookahead_large_bucket_penalty,
            "lookahead_large_bucket_penalty",
            [0.08, 0.12, 0.16]
        );
        search_dimension!(
            lookahead_dangerous_mass_penalty,
            "lookahead_dangerous_mass_penalty",
            [0.05, 0.08, 0.12]
        );
        search_dimension!(
            lookahead_large_bucket_mass_penalty,
            "lookahead_large_bucket_mass_penalty",
            [0.06, 0.10, 0.14]
        );
        search_dimension!(
            medium_state_lookahead_candidate_pool,
            "medium_state_lookahead_candidate_pool",
            [32, 48, 64]
        );
        search_dimension!(
            medium_state_lookahead_reply_pool,
            "medium_state_lookahead_reply_pool",
            [16, 20, 28]
        );
        search_dimension!(
            medium_state_force_in_two_scan,
            "medium_state_force_in_two_scan",
            [96, 160, 224]
        );

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
        eprintln!(
            "tune-prior phase=done current_avg_guesses={:.4} best_avg_guesses={:.4} current_latency_p95_ms={:.3} best_latency_p95_ms={:.3} elapsed_s={:.1}",
            current.average_guesses,
            best.average_guesses,
            current.latency_p95_ms,
            best.latency_p95_ms,
            started.elapsed().as_secs_f64()
        );
        let _ = std::io::stderr().flush();

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
                "medium_state_lookahead_threshold = {}\n",
                "lookahead_candidate_pool = {}\n",
                "medium_state_lookahead_candidate_pool = {}\n",
                "lookahead_reply_pool = {}\n",
                "medium_state_lookahead_reply_pool = {}\n",
                "lookahead_root_force_in_two_scan = {}\n",
                "medium_state_force_in_two_scan = {}\n",
                "large_state_split_threshold = {}\n",
                "pool_tight_gap_threshold = {:.2}\n",
                "pool_medium_gap_threshold = {:.2}\n",
                "danger_lookahead_threshold = {:.2}\n",
                "danger_exact_threshold = {:.2}\n",
                "danger_reply_pool_bonus = {}\n",
                "danger_exact_root_pool = {}\n",
                "danger_exact_survivor_cap = {}\n",
                "lookahead_trap_penalty = {:.2}\n",
                "lookahead_large_bucket_penalty = {:.2}\n",
                "lookahead_dangerous_mass_penalty = {:.2}\n",
                "lookahead_large_bucket_mass_penalty = {:.2}\n",
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
            best.config.medium_state_lookahead_threshold,
            best.config.lookahead_candidate_pool,
            best.config.medium_state_lookahead_candidate_pool,
            best.config.lookahead_reply_pool,
            best.config.medium_state_lookahead_reply_pool,
            best.config.lookahead_root_force_in_two_scan,
            best.config.medium_state_force_in_two_scan,
            best.config.large_state_split_threshold,
            best.config.pool_tight_gap_threshold,
            best.config.pool_medium_gap_threshold,
            best.config.danger_lookahead_threshold,
            best.config.danger_exact_threshold,
            best.config.danger_reply_pool_bonus,
            best.config.danger_exact_root_pool,
            best.config.danger_exact_survivor_cap,
            best.config.lookahead_trap_penalty,
            best.config.lookahead_large_bucket_penalty,
            best.config.lookahead_dangerous_mass_penalty,
            best.config.lookahead_large_bucket_mass_penalty,
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

    fn offline_book_solver(&self) -> Self {
        let mut config = self.config.clone();
        config.medium_state_lookahead_threshold = config.medium_state_lookahead_threshold.max(96);
        config.lookahead_candidate_pool = config.lookahead_candidate_pool.max(150);
        config.medium_state_lookahead_candidate_pool =
            config.medium_state_lookahead_candidate_pool.max(96);
        config.lookahead_reply_pool = config.lookahead_reply_pool.max(50);
        config.medium_state_lookahead_reply_pool = config.medium_state_lookahead_reply_pool.max(32);
        config.medium_state_force_in_two_scan = config.medium_state_force_in_two_scan.max(192);
        config.exact_candidate_pool = config.exact_candidate_pool.max(160);
        config.lookahead_threshold = config.lookahead_threshold.max(224);
        config.danger_exact_root_pool = config.danger_exact_root_pool.max(32);
        config.large_state_split_threshold = config.large_state_split_threshold.min(50);
        self.clone_with_config(config)
    }

    fn clone_with_config(&self, config: PriorConfig) -> Self {
        let mut cloned = self.clone();
        cloned.config = config.clone();
        cloned.exact_small_state_table =
            SmallStateTable::build(config.exact_exhaustive_threshold.max(2));
        cloned
    }

    fn is_medium_state_lookahead(&self, surviving_answers: usize) -> bool {
        surviving_answers > self.config.exact_threshold
            && surviving_answers <= self.config.medium_state_lookahead_threshold
    }

    fn lookahead_candidate_pool_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_lookahead_candidate_pool
        } else {
            self.config.lookahead_candidate_pool
        }
    }

    fn lookahead_reply_pool_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_lookahead_reply_pool
        } else {
            self.config.lookahead_reply_pool
        }
    }

    fn force_in_two_scan_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_force_in_two_scan
        } else {
            self.config.lookahead_root_force_in_two_scan
        }
    }

    fn recent_history_targets_for_books(
        &self,
        as_of: NaiveDate,
    ) -> Result<(NaiveDate, NaiveDate, Vec<(NaiveDate, String)>)> {
        let mut entries = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date <= as_of)
            .collect::<Vec<_>>();
        if entries.is_empty() {
            bail!("run sync-data before building predictive books");
        }
        entries.sort_by_key(|entry| entry.print_date);
        let window_end = entries
            .last()
            .map(|entry| entry.print_date)
            .ok_or_else(|| anyhow!("missing recent history"))?;
        let window_days = self.config.session_window_days.saturating_sub(1) as u64;
        let window_start = window_end
            .checked_sub_days(Days::new(window_days))
            .map_or(entries[0].print_date, |date| {
                date.max(entries[0].print_date)
            });
        let targets = entries
            .into_iter()
            .filter(|entry| entry.print_date >= window_start)
            .map(|entry| (entry.print_date, entry.solution.clone()))
            .collect::<Vec<_>>();
        Ok((window_start, window_end, targets))
    }

    fn evaluate_forced_opener(
        &self,
        _as_of: NaiveDate,
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        let opener = self.guesses[guess_index].clone();
        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut failures = 0usize;
        for (date, target) in targets {
            let target_as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate opener before launch date"))?;
            let run =
                self.solve_target_with_forced_opening(target, target_as_of, *date, &opener, top)?;
            guess_counts.push(run.steps.len());
            if !run.solved {
                failures += 1;
            }
        }
        guess_counts.sort_unstable();
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(ForcedOpenerEvaluation {
            guess_index,
            games: guess_counts.len(),
            average_guesses,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
            failures,
        })
    }

    fn evaluate_forced_reply(
        &self,
        opener: &str,
        _opener_feedback: u8,
        targets: &[(NaiveDate, String)],
        reply_guess_index: usize,
        top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        self.evaluate_forced_continuation(&[opener.to_string()], targets, reply_guess_index, top)
    }

    fn evaluate_forced_continuation(
        &self,
        forced_prefix: &[String],
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        let guess = self.guesses[guess_index].clone();
        let forced_prefix = forced_prefix
            .iter()
            .cloned()
            .map(|word| (word, 0))
            .collect::<Vec<_>>();
        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut failures = 0usize;
        for (date, target) in targets {
            let target_as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate reply before launch date"))?;
            let mut forced = forced_prefix.clone();
            forced.push((guess.clone(), 0));
            let run =
                self.solve_target_with_forced_prefix(target, target_as_of, *date, &forced, top)?;
            guess_counts.push(run.steps.len());
            if !run.solved {
                failures += 1;
            }
        }
        guess_counts.sort_unstable();
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(ForcedOpenerEvaluation {
            guess_index,
            games: guess_counts.len(),
            average_guesses,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
            failures,
        })
    }

    fn solve_target_with_forced_opening(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        opener: &str,
        top: usize,
    ) -> Result<DetailedSolveRun> {
        let forced = [(opener.to_string(), 0)];
        self.solve_target_with_forced_prefix(target, as_of, date, &forced, top)
    }

    fn solve_target_with_forced_prefix(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        forced: &[(String, u8)],
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
        let mut observations = Vec::new();
        for (position, (guess, expected_feedback)) in forced.iter().enumerate() {
            let feedback = score_guess(guess, &target);
            if position == 0 && *expected_feedback != 0 && *expected_feedback != feedback {
                bail!(
                    "forced opener feedback mismatch for {}: expected {}, got {}",
                    guess,
                    format_feedback_letters(*expected_feedback),
                    format_feedback_letters(feedback)
                );
            }
            let surviving_before = state.surviving.len();
            let surviving_after = if feedback == ALL_GREEN_PATTERN {
                1
            } else {
                let mut next_state = state.clone();
                self.apply_feedback(&mut next_state, guess, feedback)?;
                next_state.surviving.len()
            };
            steps.push(DetailedSolveStep {
                guess: guess.clone(),
                feedback,
                surviving_before,
                surviving_after,
                chosen_force_in_two: false,
                alternative_force_in_two: false,
                danger_score: 0.0,
                danger_escalated: false,
                regime_used: PredictiveRegime::Proxy,
                lookahead_pool_base: 0,
                lookahead_pool_size: 0,
                exact_pool_base: 0,
                exact_pool_size: 0,
                root_candidate_count: 0,
                top_suggestions: Vec::new(),
            });
            if feedback == ALL_GREEN_PATTERN {
                return Ok(DetailedSolveRun {
                    target,
                    date,
                    steps,
                    solved: true,
                });
            }
            observations.push((guess.clone(), feedback));
            self.apply_feedback(&mut state, guess, feedback)?;
        }

        while steps.len() < 6 {
            let surviving_before = state.surviving.len();
            let batch = self.suggestion_batch_internal(
                &state,
                top.max(1),
                Some(PredictiveContext {
                    as_of,
                    observations: &observations,
                }),
                PredictiveBookUsage::None,
            )?;
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
                lookahead_pool_base: batch.lookahead_pool_base,
                lookahead_pool_size: batch.lookahead_pool_size,
                exact_pool_base: batch.exact_pool_base,
                exact_pool_size: batch.exact_pool_size,
                root_candidate_count: batch.root_candidate_count,
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
            observations.push((chosen.word.clone(), feedback));
            self.apply_feedback(&mut state, &chosen.word, feedback)?;
        }

        Ok(DetailedSolveRun {
            target,
            date,
            steps,
            solved: false,
        })
    }

    fn predictive_book_identity(&self, as_of: NaiveDate) -> PredictiveBookIdentity {
        let config_toml =
            toml::to_string(&self.config).expect("predictive config serialization must succeed");
        let payload = format!(
            "mode={};variant={};as_of={};guesses={};answers={};config={}",
            self.mode.label(),
            self.variant.label(),
            as_of,
            self.guesses.len(),
            self.answers.len(),
            config_toml
        );
        PredictiveBookIdentity {
            mode: self.mode.label().to_string(),
            variant: self.variant.label().to_string(),
            config_fingerprint: stable_fingerprint(&payload),
            as_of,
        }
    }

    fn opener_artifact_path(&self, as_of: NaiveDate) -> PathBuf {
        let identity = self.predictive_book_identity(as_of);
        self.artifact_dir.join(format!(
            "opener-{}-{}-{}-{}.json",
            identity.mode, identity.variant, identity.config_fingerprint, identity.as_of
        ))
    }

    fn reply_book_artifact_path(&self, as_of: NaiveDate) -> PathBuf {
        let identity = self.predictive_book_identity(as_of);
        self.artifact_dir.join(format!(
            "reply-book-{}-{}-{}-{}.json",
            identity.mode, identity.variant, identity.config_fingerprint, identity.as_of
        ))
    }

    fn load_predictive_opener_artifact(
        &self,
        as_of: NaiveDate,
    ) -> Result<Option<PredictiveOpenerArtifact>> {
        let path = self.opener_artifact_path(as_of);
        if !path.exists() {
            return Ok(None);
        }
        let artifact: PredictiveOpenerArtifact = read_predictive_artifact(&path)?;
        let valid = artifact.identity == self.predictive_book_identity(as_of)
            && artifact.games > 0
            && artifact.average_guesses.is_finite()
            && artifact.average_guesses > 0.0
            && artifact.failures < artifact.games;
        Ok(valid.then_some(artifact))
    }

    fn load_predictive_reply_book(
        &self,
        as_of: NaiveDate,
    ) -> Result<Option<PredictiveReplyBookArtifact>> {
        let path = self.reply_book_artifact_path(as_of);
        if !path.exists() {
            return Ok(None);
        }
        let artifact: PredictiveReplyBookArtifact = read_predictive_artifact(&path)?;
        Ok((artifact.identity == self.predictive_book_identity(as_of)).then_some(artifact))
    }

    fn session_root_guess(&self, as_of: NaiveDate) -> Result<Option<String>> {
        let identity = self.predictive_book_identity(as_of);
        if let Some(cached) = self
            .session_opener_cache
            .lock()
            .expect("session opener cache")
            .get(&identity)
            .cloned()
        {
            return Ok(cached);
        }

        let computed = self.evaluate_session_opener(as_of)?;
        self.session_opener_cache
            .lock()
            .expect("session opener cache")
            .insert(identity, computed.clone());
        Ok(computed)
    }

    fn evaluate_session_opener(&self, as_of: NaiveDate) -> Result<Option<String>> {
        let offline = self.offline_book_solver();
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        if targets.is_empty() {
            return Ok(None);
        }
        let state = offline.initial_state(as_of);
        let candidates = offline
            .suggestion_batch_internal(
                &state,
                offline.config.session_opener_pool.max(1),
                Some(PredictiveContext {
                    as_of,
                    observations: &[],
                }),
                PredictiveBookUsage::None,
            )?
            .suggestions;
        let best = candidates
            .par_iter()
            .filter_map(|suggestion| {
                let guess_index = offline.guess_index.get(&suggestion.word).copied()?;
                let evaluation = offline
                    .evaluate_forced_opener(as_of, &targets, guess_index, 3)
                    .ok()?;
                Some((suggestion.word.clone(), evaluation))
            })
            .min_by(|left, right| compare_forced_openers(&left.1, &right.1, &offline.guesses));
        Ok(best.map(|(word, _)| word))
    }

    fn session_reply_guess(
        &self,
        as_of: NaiveDate,
        opener: &str,
        pattern: u8,
    ) -> Result<Option<String>> {
        let identity = self.predictive_book_identity(as_of);
        let key = (identity, opener.to_string(), pattern);
        if let Some(cached) = self
            .session_reply_cache
            .lock()
            .expect("session reply cache")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let computed = self.evaluate_session_reply(as_of, opener, pattern)?;
        self.session_reply_cache
            .lock()
            .expect("session reply cache")
            .insert(key, computed.clone());
        Ok(computed)
    }

    fn session_third_guess(
        &self,
        as_of: NaiveDate,
        opener: &str,
        opener_pattern: u8,
        reply: &str,
        reply_pattern: u8,
    ) -> Result<Option<String>> {
        let identity = self.predictive_book_identity(as_of);
        let key = (
            identity,
            opener.to_string(),
            opener_pattern,
            reply.to_string(),
            reply_pattern,
        );
        if let Some(cached) = self
            .session_third_cache
            .lock()
            .expect("session third cache")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let computed =
            self.evaluate_session_third(as_of, opener, opener_pattern, reply, reply_pattern)?;
        self.session_third_cache
            .lock()
            .expect("session third cache")
            .insert(key, computed.clone());
        Ok(computed)
    }

    fn evaluate_session_reply(
        &self,
        as_of: NaiveDate,
        opener: &str,
        pattern: u8,
    ) -> Result<Option<String>> {
        let offline = self.offline_book_solver();
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let scoped_targets = targets
            .into_iter()
            .filter(|(_, target)| score_guess(opener, target) == pattern)
            .collect::<Vec<_>>();
        if scoped_targets.is_empty() {
            return Ok(None);
        }
        let root = offline.initial_state(as_of);
        let mut child = root.clone();
        offline.apply_feedback(&mut child, opener, pattern)?;
        if child.surviving.len() <= 1 {
            return Ok(None);
        }
        let observations = vec![(opener.to_string(), pattern)];
        offline.evaluate_session_branch_guess(as_of, &child, &observations, &scoped_targets)
    }

    fn evaluate_session_third(
        &self,
        as_of: NaiveDate,
        opener: &str,
        opener_pattern: u8,
        reply: &str,
        reply_pattern: u8,
    ) -> Result<Option<String>> {
        let offline = self.offline_book_solver();
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let scoped_targets = targets
            .into_iter()
            .filter(|(_, target)| {
                score_guess(opener, target) == opener_pattern
                    && score_guess(reply, target) == reply_pattern
            })
            .collect::<Vec<_>>();
        if scoped_targets.is_empty() {
            return Ok(None);
        }
        let mut state = offline.initial_state(as_of);
        offline.apply_feedback(&mut state, opener, opener_pattern)?;
        if state.surviving.len() <= 1 {
            return Ok(None);
        }
        offline.apply_feedback(&mut state, reply, reply_pattern)?;
        if state.surviving.len() <= 1 {
            return Ok(None);
        }
        let observations = vec![
            (opener.to_string(), opener_pattern),
            (reply.to_string(), reply_pattern),
        ];
        offline.evaluate_session_branch_guess(as_of, &state, &observations, &scoped_targets)
    }

    fn evaluate_session_branch_guess(
        &self,
        as_of: NaiveDate,
        state: &SolveState,
        observations: &[(String, u8)],
        scoped_targets: &[(NaiveDate, String)],
    ) -> Result<Option<String>> {
        let batch = self.suggestion_batch_internal(
            state,
            self.config.session_reply_pool.max(1),
            Some(PredictiveContext {
                as_of,
                observations,
            }),
            PredictiveBookUsage::None,
        )?;
        let split_first = state.surviving.len() > self.config.large_state_split_threshold;
        let mut metrics = self.score_guess_metrics_for_subset(
            &state.surviving,
            &state.weights,
            &self.exact_small_state_table,
        );
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let total_weight = state
            .surviving
            .iter()
            .map(|index| state.weights[*index])
            .sum::<f64>();
        let assessment =
            self.assess_subset_danger(&state.surviving, &state.weights, total_weight, &metrics);
        let lookahead_pool = self.expanded_pool_size(
            &batch.suggestions,
            self.config.session_reply_pool.max(1),
            split_first,
            state.surviving.len() > self.config.large_state_split_threshold,
            assessment,
        );
        let mut candidate_indexes = self.collect_lookahead_candidates(
            &batch.suggestions,
            state.surviving.len(),
            assessment.dangerous_lookahead,
            lookahead_pool,
        )?;
        if assessment.dangerous_exact
            && state.surviving.len() <= self.config.danger_exact_survivor_cap
        {
            let exact_pool = self.expanded_pool_size(
                &batch.suggestions,
                self.config.session_reply_pool.max(1),
                split_first,
                state.surviving.len() > self.config.exact_threshold,
                assessment,
            );
            let mut seen = candidate_indexes.iter().copied().collect::<HashSet<_>>();
            for guess_index in
                self.collect_exact_candidates(state, &batch.suggestions, exact_pool)?
            {
                if seen.insert(guess_index) {
                    candidate_indexes.push(guess_index);
                }
            }
        }
        let forced_prefix = observations
            .iter()
            .map(|(guess, _)| guess.clone())
            .collect::<Vec<_>>();
        let best = candidate_indexes
            .par_iter()
            .filter_map(|guess_index| {
                let evaluation = self
                    .evaluate_forced_continuation(&forced_prefix, scoped_targets, *guess_index, 3)
                    .ok()?;
                Some((self.guesses[*guess_index].clone(), evaluation))
            })
            .min_by(|left, right| compare_forced_openers(&left.1, &right.1, &self.guesses));
        Ok(best.map(|(word, _)| word))
    }

    fn cached_predictive_choice(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        allow_session_fallback: bool,
    ) -> Option<String> {
        match observations {
            [] => self
                .load_predictive_opener_artifact(as_of)
                .ok()
                .flatten()
                .map(|artifact| artifact.opener)
                .or_else(|| {
                    allow_session_fallback
                        .then(|| self.session_root_guess(as_of).ok().flatten())
                        .flatten()
                }),
            [(guess, pattern)] => self
                .load_predictive_reply_book(as_of)
                .ok()
                .flatten()
                .filter(|artifact| artifact.opener == guess.as_str())
                .and_then(|artifact| {
                    artifact
                        .replies
                        .into_iter()
                        .find(|entry| entry.feedback_pattern == *pattern)
                        .map(|entry| entry.reply)
                })
                .or_else(|| {
                    allow_session_fallback
                        .then(|| {
                            self.session_reply_guess(as_of, guess, *pattern)
                                .ok()
                                .flatten()
                        })
                        .flatten()
                }),
            [(opener, opener_pattern), (reply, reply_pattern)] => self
                .load_predictive_reply_book(as_of)
                .ok()
                .flatten()
                .filter(|artifact| artifact.opener == opener.as_str())
                .and_then(|artifact| {
                    artifact
                        .replies
                        .into_iter()
                        .find(|entry| {
                            entry.feedback_pattern == *opener_pattern
                                && entry.reply == reply.as_str()
                        })
                        .and_then(|entry| {
                            entry
                                .third_replies
                                .into_iter()
                                .find(|entry| entry.second_feedback_pattern == *reply_pattern)
                                .map(|entry| entry.reply)
                        })
                })
                .or_else(|| {
                    allow_session_fallback
                        .then(|| {
                            self.session_third_guess(
                                as_of,
                                opener,
                                *opener_pattern,
                                reply,
                                *reply_pattern,
                            )
                            .ok()
                            .flatten()
                        })
                        .flatten()
                }),
            _ => None,
        }
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

    fn absurdle_score_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        total: f64,
    ) -> AbsurdleSuggestion {
        let mut counts = [0usize; PATTERN_SPACE];
        let mut touched_patterns = Vec::new();
        for answer_index in subset {
            let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
            if counts[pattern] == 0 {
                touched_patterns.push(pattern as u8);
            }
            counts[pattern] += 1;
        }

        let mut largest_bucket_size = 0usize;
        let mut second_largest_bucket_size = 0usize;
        let mut multi_answer_bucket_count = 0usize;
        let mut entropy = 0.0;

        for pattern in touched_patterns {
            if pattern == ALL_GREEN_PATTERN {
                continue;
            }
            let count = counts[pattern as usize];
            if count > 1 {
                multi_answer_bucket_count += 1;
            }
            if count > largest_bucket_size {
                second_largest_bucket_size = largest_bucket_size;
                largest_bucket_size = count;
            } else if count > second_largest_bucket_size {
                second_largest_bucket_size = count;
            }
            let probability = count as f64 / total;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        AbsurdleSuggestion {
            word: self.guesses[guess_index].clone(),
            entropy,
            largest_bucket_size,
            second_largest_bucket_size,
            multi_answer_bucket_count,
        }
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
        let mut large_non_green_bucket_count = 0usize;
        let mut dangerous_mass_bucket_count = 0usize;
        let mut non_green_mass_in_large_buckets = 0.0_f64;
        let mut non_green_bucket_count = 0usize;
        let mut non_green_mass = 0.0_f64;
        let mut non_green_mass_square_sum = 0.0_f64;
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
            } else {
                non_green_bucket_count += 1;
                non_green_mass += probability;
                non_green_mass_square_sum += probability * probability;
                worst_non_green_bucket_size =
                    worst_non_green_bucket_size.max(scratch.counts[index]);
                largest_non_green_bucket_mass = largest_non_green_bucket_mass.max(probability);
                if scratch.counts[index] >= self.config.trap_size_threshold {
                    large_non_green_bucket_count += 1;
                    non_green_mass_in_large_buckets += probability;
                }
                if probability >= self.config.trap_mass_threshold {
                    dangerous_mass_bucket_count += 1;
                }
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
        let smoothness_penalty = normalized_concentration_penalty(
            non_green_mass,
            non_green_mass_square_sum,
            non_green_bucket_count,
        );
        proxy_cost += 0.25 * smoothness_penalty;
        let entropy = if total_weight > 0.0 {
            total_weight.log2() - (sum_mass_log_mass / total_weight)
        } else {
            0.0
        };
        let large_state_score = proxy_row_score_from_weights(
            &self.config.proxy_weights,
            entropy,
            largest_non_green_bucket_mass,
            worst_non_green_bucket_size,
            high_mass_ambiguous_bucket_count,
            proxy_cost,
            solve_probability,
            posterior_answer_probability,
            smoothness_penalty,
            0,
            large_non_green_bucket_count,
            dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets,
        );

        GuessMetrics {
            guess_index,
            entropy,
            solve_probability,
            expected_remaining,
            force_in_two,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size,
            largest_non_green_bucket_mass,
            high_mass_ambiguous_bucket_count,
            smoothness_penalty,
            large_non_green_bucket_count,
            dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets,
            proxy_cost,
            large_state_score,
            posterior_answer_probability,
        }
    }

    fn expanded_pool_size(
        &self,
        suggestions: &[Suggestion],
        base_pool: usize,
        split_first: bool,
        allow_expansion: bool,
        assessment: StateDangerAssessment,
    ) -> usize {
        let base = base_pool.max(1).min(suggestions.len().max(1));
        if suggestions.is_empty() || !allow_expansion {
            return base;
        }
        let kth = base
            .saturating_sub(1)
            .min(suggestions.len().saturating_sub(1));
        let gap = if split_first {
            let top = suggestions[0]
                .large_state_score
                .unwrap_or(f64::NEG_INFINITY);
            let kth_score = suggestions[kth]
                .large_state_score
                .unwrap_or(f64::NEG_INFINITY);
            (top - kth_score).abs()
        } else {
            let top = suggestions[0].proxy_cost.unwrap_or(f64::INFINITY);
            let kth_score = suggestions[kth].proxy_cost.unwrap_or(f64::INFINITY);
            (kth_score - top).abs()
        };
        let expanded = if gap < self.config.pool_tight_gap_threshold {
            base.saturating_mul(5) / 2
        } else if gap < self.config.pool_medium_gap_threshold {
            base.saturating_mul(3) / 2
        } else {
            base
        };
        let expanded = if assessment.dangerous_lookahead || assessment.dangerous_exact {
            expanded.saturating_add(self.config.danger_reply_pool_bonus)
        } else {
            expanded
        };
        expanded.min(suggestions.len())
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

        let mut diversity_ranked = suggestions.iter().collect::<Vec<_>>();
        diversity_ranked.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| {
                    left.largest_non_green_bucket_mass
                        .total_cmp(&right.largest_non_green_bucket_mass)
                })
                .then_with(|| compare_suggestions(left, right))
        });
        let stride = self.config.pool_diversity_stride.max(1);
        for suggestion in diversity_ranked.into_iter().step_by(stride) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
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
        let mut large_bucket_count = 0usize;
        let mut dangerous_mass_bucket_count = 0usize;
        let mut non_green_mass_in_large_buckets = 0.0_f64;
        let mut high_mass_ambiguous_bucket_count = 0usize;
        for pattern in ordered_patterns[..ordered_len].iter().copied() {
            let mass = exact_scratch.frames[0].masses[pattern as usize];
            let probability = mass / total_weight;
            let child_len = exact_scratch.frames[0].child_subsets[pattern as usize].len();
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
                if child_len >= self.config.trap_size_threshold {
                    large_bucket_count += 1;
                    non_green_mass_in_large_buckets += probability;
                }
                if probability >= self.config.trap_mass_threshold {
                    dangerous_mass_bucket_count += 1;
                    if child_len > 1 {
                        high_mass_ambiguous_bucket_count += 1;
                    }
                }
            }
        }
        Ok(total_cost
            + self.aggregate_lookahead_trap_penalty(
                worst_child_probability,
                large_bucket_count,
                dangerous_mass_bucket_count,
                non_green_mass_in_large_buckets,
                high_mass_ambiguous_bucket_count,
            ))
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
        if subset.len() <= self.config.exact_exhaustive_threshold {
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
        let split_first = subset.len() > self.config.large_state_split_threshold;
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let assessment = self.assess_subset_danger(subset, weights, total_weight, &metrics);
        let reply_pool = self.lookahead_reply_pool_for_state(subset.len()).max(1)
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
            .map(|metric| metric.proxy_cost + self.lookahead_reply_penalty(&metric, subset.len()))
            .fold(f64::INFINITY, f64::min);
        let child_value = 1.0 + best_reply;
        lookahead_memo.insert(key, child_value);
        Ok(child_value)
    }

    fn aggregate_lookahead_trap_penalty(
        &self,
        worst_branch_mass: f64,
        large_bucket_count: usize,
        dangerous_mass_bucket_count: usize,
        non_green_mass_in_large_buckets: f64,
        high_mass_ambiguous_bucket_count: usize,
    ) -> f64 {
        let compounded_large_mass = non_green_mass_in_large_buckets
            * (1.0
                + (large_bucket_count as f64 * 0.25)
                + (dangerous_mass_bucket_count as f64 * 0.35));
        (self.config.lookahead_trap_penalty
            * (worst_branch_mass + (high_mass_ambiguous_bucket_count as f64 / 4.0).min(1.0)))
            + (self.config.lookahead_large_bucket_penalty
                * (large_bucket_count as f64 + (high_mass_ambiguous_bucket_count as f64 * 0.5)))
            + (self.config.lookahead_dangerous_mass_penalty
                * (dangerous_mass_bucket_count as f64 + high_mass_ambiguous_bucket_count as f64))
            + (self.config.lookahead_large_bucket_mass_penalty * compounded_large_mass)
    }

    fn lookahead_reply_penalty(&self, metric: &GuessMetrics, subset_len: usize) -> f64 {
        let bucket_ratio = metric.worst_non_green_bucket_size as f64 / subset_len.max(1) as f64;
        self.aggregate_lookahead_trap_penalty(
            bucket_ratio + metric.largest_non_green_bucket_mass,
            metric.large_non_green_bucket_count,
            metric.dangerous_mass_bucket_count,
            metric.non_green_mass_in_large_buckets,
            metric.high_mass_ambiguous_bucket_count,
        )
    }

    fn collect_lookahead_candidates(
        &self,
        suggestions: &[Suggestion],
        surviving_answers: usize,
        expanded: bool,
        base_pool: usize,
    ) -> Result<Vec<usize>> {
        let pool = base_pool.max(1)
            + if expanded {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let force_scan = self.force_in_two_scan_for_state(surviving_answers).max(1)
            + if expanded {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let diversity_take = pool.div_ceil(self.config.pool_diversity_stride.max(1));
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
        let stride = self.config.pool_diversity_stride.max(1);
        let mut by_entropy = suggestions.iter().collect::<Vec<_>>();
        by_entropy.sort_by(|left, right| {
            right
                .entropy
                .total_cmp(&left.entropy)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_entropy.into_iter().step_by(stride).take(diversity_take) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        let mut by_worst_bucket = suggestions.iter().collect::<Vec<_>>();
        by_worst_bucket.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_worst_bucket
            .into_iter()
            .step_by(stride)
            .take(diversity_take)
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
        let mut by_mass_reducer = suggestions.iter().collect::<Vec<_>>();
        by_mass_reducer.sort_by(|left, right| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_mass_reducer
            .into_iter()
            .step_by(stride)
            .take(diversity_take)
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
        for suggestion in suggestions
            .iter()
            .skip(stride - 1)
            .step_by(stride)
            .take(diversity_take)
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
        let split_first = subset.len() > self.config.large_state_split_threshold;
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let surviving_guess_indexes = subset
            .iter()
            .filter_map(|answer_index| self.guess_index.get(&self.answers[*answer_index].word))
            .copied()
            .collect::<HashSet<_>>();
        let mut selected = Vec::new();
        let mut seen = HashSet::new();
        for metric in metrics.iter().take(count) {
            if seen.insert(metric.guess_index) {
                selected.push(metric.guess_index);
            }
        }
        for metric in metrics.into_iter().skip(count) {
            if surviving_guess_indexes.contains(&metric.guess_index)
                && seen.insert(metric.guess_index)
            {
                selected.push(metric.guess_index);
            }
        }
        selected
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

fn stable_fingerprint(input: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn write_predictive_artifact<T: Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let raw =
        serde_json::to_vec_pretty(value).context("failed to serialize predictive artifact")?;
    fs::write(path, raw).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn read_predictive_artifact<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Result<T> {
    let raw = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn promote_cached_suggestion(suggestions: &mut [Suggestion], cached_word: &str) {
    if let Some(position) = suggestions
        .iter()
        .position(|suggestion| suggestion.word == cached_word)
    {
        suggestions[..=position].rotate_right(1);
    }
}

fn compare_forced_openers(
    left: &ForcedOpenerEvaluation,
    right: &ForcedOpenerEvaluation,
    guesses: &[String],
) -> std::cmp::Ordering {
    left.failures
        .cmp(&right.failures)
        .then_with(|| left.average_guesses.total_cmp(&right.average_guesses))
        .then_with(|| left.p95_guesses.cmp(&right.p95_guesses))
        .then_with(|| left.max_guesses.cmp(&right.max_guesses))
        .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
}

fn known_absent_letter_mask(observations: &[(String, u8)]) -> u32 {
    let mut gray_mask = 0u32;
    let mut present_mask = 0u32;
    for (guess, pattern) in observations {
        let mut value = *pattern;
        for byte in guess.bytes().take(5) {
            let letter_bit = 1u32 << ((byte - b'a') as u32);
            let trit = value % 3;
            value /= 3;
            if trit == 0 {
                gray_mask |= letter_bit;
            } else {
                present_mask |= letter_bit;
            }
        }
    }
    gray_mask & !present_mask
}

fn count_masked_letters(word: &str, mask: u32) -> usize {
    word.bytes()
        .filter(|byte| (mask & (1u32 << ((byte - b'a') as u32))) != 0)
        .count()
}

fn hard_mode_violation(observations: &[(String, u8)], guess: &str) -> Option<String> {
    if observations.is_empty() {
        return None;
    }
    if guess.len() != HARD_MODE_WORD_LENGTH || !guess.bytes().all(|byte| byte.is_ascii_lowercase())
    {
        return Some("hard mode guess must be exactly 5 lowercase letters".to_string());
    }

    let constraints = build_hard_mode_constraints(observations);
    let guess_bytes = guess.as_bytes();
    let mut guess_counts = [0u8; 26];
    for (index, &byte) in guess_bytes.iter().enumerate() {
        let letter_index = (byte - b'a') as usize;
        guess_counts[letter_index] += 1;

        if let Some(expected) = constraints.greens[index] {
            if byte != expected {
                return Some(format!(
                    "hard mode requires {} in position {}",
                    char::from(expected).to_ascii_uppercase(),
                    index + 1
                ));
            }
        }

        if (constraints.yellow_forbidden[index] & (1u32 << letter_index)) != 0 {
            return Some(format!(
                "hard mode forbids {} in position {}",
                char::from(byte).to_ascii_uppercase(),
                index + 1
            ));
        }
    }

    for (letter_index, &required) in constraints.required_counts.iter().enumerate() {
        if required > 0 && guess_counts[letter_index] < required {
            let letter = char::from(b'a' + letter_index as u8).to_ascii_uppercase();
            return Some(format!(
                "hard mode requires {} occurrence{} of {}",
                required,
                if required == 1 { "" } else { "s" },
                letter
            ));
        }
    }

    None
}

struct HardModeConstraints {
    greens: [Option<u8>; HARD_MODE_WORD_LENGTH],
    yellow_forbidden: [u32; HARD_MODE_WORD_LENGTH],
    required_counts: [u8; 26],
}

fn build_hard_mode_constraints(observations: &[(String, u8)]) -> HardModeConstraints {
    let mut constraints = HardModeConstraints {
        greens: [None; HARD_MODE_WORD_LENGTH],
        yellow_forbidden: [0; HARD_MODE_WORD_LENGTH],
        required_counts: [0; 26],
    };

    for (guess, pattern) in observations {
        let feedback = decode_feedback(*pattern);
        let guess_bytes = guess.as_bytes();
        let mut positive_counts = [0u8; 26];

        for index in 0..HARD_MODE_WORD_LENGTH {
            let byte = guess_bytes[index];
            let letter_index = (byte - b'a') as usize;
            match feedback[index] {
                2 => {
                    constraints.greens[index] = Some(byte);
                    positive_counts[letter_index] += 1;
                }
                1 => {
                    constraints.yellow_forbidden[index] |= 1u32 << letter_index;
                    positive_counts[letter_index] += 1;
                }
                _ => {}
            }
        }

        for (letter_index, &count) in positive_counts.iter().enumerate() {
            constraints.required_counts[letter_index] =
                constraints.required_counts[letter_index].max(count);
        }
    }

    constraints
}

fn normalized_concentration_penalty(
    total_mass: f64,
    mass_square_sum: f64,
    bucket_count: usize,
) -> f64 {
    if total_mass <= 0.0 || bucket_count <= 1 {
        return 0.0;
    }
    let concentration = mass_square_sum / (total_mass * total_mass);
    let uniform = 1.0 / bucket_count as f64;
    ((concentration - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
}

fn proxy_row_score_from_weights(
    weights: &crate::config::ProxyWeights,
    entropy: f64,
    largest_non_green_bucket_mass: f64,
    worst_non_green_bucket_size: usize,
    high_mass_ambiguous_bucket_count: usize,
    proxy_cost: f64,
    solve_probability: f64,
    posterior_answer_probability: f64,
    smoothness_penalty: f64,
    known_absent_letter_hits: usize,
    large_non_green_bucket_count: usize,
    dangerous_mass_bucket_count: usize,
    non_green_mass_in_large_buckets: f64,
) -> f64 {
    (weights.entropy_w * entropy)
        - (weights.bucket_mass_w * largest_non_green_bucket_mass)
        - (weights.bucket_size_w * worst_non_green_bucket_size as f64)
        - (weights.ambiguous_w * high_mass_ambiguous_bucket_count as f64)
        - (weights.proxy_w * proxy_cost)
        + (weights.solve_prob_w * solve_probability)
        + (weights.posterior_w * posterior_answer_probability)
        - (weights.smoothness_w * smoothness_penalty)
        - (weights.gray_reuse_w * known_absent_letter_hits as f64)
        - (weights.large_bucket_count_w * large_non_green_bucket_count as f64)
        - (weights.dangerous_mass_count_w * dangerous_mass_bucket_count as f64)
        - (weights.large_bucket_mass_w * non_green_mass_in_large_buckets)
}

fn flatten_weighted_config(config: &PriorConfig, keep_factor: f64) -> PriorConfig {
    let mut flattened = config.clone();
    let blend = |value: f64| 1.0 + ((value - 1.0) * keep_factor);
    flattened.base_seed_weight = blend(flattened.base_seed_weight).max(0.01);
    flattened.base_history_only_weight = blend(flattened.base_history_only_weight).max(0.01);
    flattened.cooldown_floor = blend(flattened.cooldown_floor).clamp(0.0, 1.0);
    flattened.manual_weights = flattened
        .manual_weights
        .into_iter()
        .map(|(word, weight)| (word, blend(weight).max(0.01)))
        .collect();
    flattened
}

fn compare_guess_metrics(
    left: &GuessMetrics,
    right: &GuessMetrics,
    guesses: &[String],
) -> std::cmp::Ordering {
    compare_guess_metrics_for_state(left, right, guesses, false)
}

fn compare_absurdle_suggestions(
    left: &AbsurdleSuggestion,
    right: &AbsurdleSuggestion,
) -> std::cmp::Ordering {
    left.largest_bucket_size
        .cmp(&right.largest_bucket_size)
        .then_with(|| {
            left.second_largest_bucket_size
                .cmp(&right.second_largest_bucket_size)
        })
        .then_with(|| {
            left.multi_answer_bucket_count
                .cmp(&right.multi_answer_bucket_count)
        })
        .then_with(|| right.entropy.total_cmp(&left.entropy))
        .then_with(|| left.word.cmp(&right.word))
}

fn compare_guess_metrics_for_state(
    left: &GuessMetrics,
    right: &GuessMetrics,
    guesses: &[String],
    split_first: bool,
) -> std::cmp::Ordering {
    if split_first {
        let score_cmp = right.large_state_score.total_cmp(&left.large_state_score);
        if score_cmp != std::cmp::Ordering::Equal {
            return score_cmp;
        }
        left.known_absent_letter_hits
            .cmp(&right.known_absent_letter_hits)
            .then_with(|| {
                left.largest_non_green_bucket_mass
                    .total_cmp(&right.largest_non_green_bucket_mass)
            })
            .then_with(|| {
                left.worst_non_green_bucket_size
                    .cmp(&right.worst_non_green_bucket_size)
            })
            .then_with(|| {
                left.large_non_green_bucket_count
                    .cmp(&right.large_non_green_bucket_count)
            })
            .then_with(|| {
                left.dangerous_mass_bucket_count
                    .cmp(&right.dangerous_mass_bucket_count)
            })
            .then_with(|| right.entropy.total_cmp(&left.entropy))
            .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
            .then_with(|| left.proxy_cost.total_cmp(&right.proxy_cost))
            .then_with(|| compare_force_in_two(left.force_in_two, right.force_in_two))
            .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
            .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
    } else {
        let proxy_cmp = left.proxy_cost.total_cmp(&right.proxy_cost);
        if proxy_cmp != std::cmp::Ordering::Equal {
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
}

fn compare_suggestions(left: &Suggestion, right: &Suggestion) -> std::cmp::Ordering {
    compare_suggestions_for_state(left, right, false)
}

fn compare_suggestions_for_state(
    left: &Suggestion,
    right: &Suggestion,
    split_first: bool,
) -> std::cmp::Ordering {
    if split_first {
        let left_score = left.large_state_score.unwrap_or(f64::NEG_INFINITY);
        let right_score = right.large_state_score.unwrap_or(f64::NEG_INFINITY);
        let score_cmp = right_score.total_cmp(&left_score);
        if score_cmp != std::cmp::Ordering::Equal {
            return score_cmp;
        }
        left.known_absent_letter_hits
            .cmp(&right.known_absent_letter_hits)
            .then_with(|| {
                left.largest_non_green_bucket_mass
                    .total_cmp(&right.largest_non_green_bucket_mass)
            })
            .then_with(|| {
                left.worst_non_green_bucket_size
                    .cmp(&right.worst_non_green_bucket_size)
            })
            .then_with(|| {
                left.large_non_green_bucket_count
                    .cmp(&right.large_non_green_bucket_count)
            })
            .then_with(|| {
                left.dangerous_mass_bucket_count
                    .cmp(&right.dangerous_mass_bucket_count)
            })
            .then_with(|| {
                left.non_green_mass_in_large_buckets
                    .total_cmp(&right.non_green_mass_in_large_buckets)
            })
            .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
            .then_with(|| {
                left.proxy_cost
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&right.proxy_cost.unwrap_or(f64::INFINITY))
            })
            .then_with(|| compare_force_in_two(left.force_in_two, right.force_in_two))
            .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
            .then_with(|| left.word.cmp(&right.word))
    } else {
        let left_proxy = left.proxy_cost.unwrap_or(f64::INFINITY);
        let right_proxy = right.proxy_cost.unwrap_or(f64::INFINITY);
        let proxy_cmp = left_proxy.total_cmp(&right_proxy);
        if proxy_cmp != std::cmp::Ordering::Equal {
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
                left.large_non_green_bucket_count
                    .cmp(&right.large_non_green_bucket_count)
            })
            .then_with(|| {
                left.dangerous_mass_bucket_count
                    .cmp(&right.dangerous_mass_bucket_count)
            })
            .then_with(|| {
                left.non_green_mass_in_large_buckets
                    .total_cmp(&right.non_green_mass_in_large_buckets)
            })
            .then_with(|| {
                right
                    .posterior_answer_probability
                    .total_cmp(&left.posterior_answer_probability)
            })
            .then_with(|| left.word.cmp(&right.word))
    }
}

fn compare_lookahead(
    left: &Suggestion,
    right: &Suggestion,
    split_first: bool,
) -> std::cmp::Ordering {
    match (left.lookahead_cost, right.lookahead_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions_for_state(left, right, split_first))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions_for_state(left, right, split_first),
    }
}

fn compare_exact(left: &Suggestion, right: &Suggestion, split_first: bool) -> std::cmp::Ordering {
    match (left.exact_cost, right.exact_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions_for_state(left, right, split_first))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions_for_state(left, right, split_first),
    }
}

fn compare_exact_costs(
    left: &Suggestion,
    right: &Suggestion,
    left_cost: Option<f64>,
    right_cost: Option<f64>,
    split_first: bool,
) -> std::cmp::Ordering {
    match (left_cost, right_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions_for_state(left, right, split_first))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions_for_state(left, right, split_first),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use chrono::NaiveDate;

    use crate::{
        config::PriorConfig,
        data::NytDailyEntry,
        model::{AnswerRecord, ModelVariant, WeightMode},
        pattern_table::PatternTable,
        scoring::{format_feedback_letters, score_guess},
        small_state::SmallStateTable,
    };

    use super::{
        AbsurdleSuggestion, ExactSearchScratch, ExactSubsetKey, ExactSubsetStorage,
        ExactSuggestionMode, GuessMetrics, PredictiveBookUsage, PredictiveMemoMap,
        PredictiveReplyBookArtifact, PredictiveReplyEntry, PredictiveSearchMode,
        PredictiveThirdReplyEntry, Solver, StateDangerAssessment, Suggestion,
        compare_absurdle_suggestions, compare_exact_costs, compare_guess_metrics,
        compare_guess_metrics_for_state, compare_lookahead, compare_suggestions,
        compare_suggestions_for_state, count_masked_letters, exact_suggestion_mode,
        hard_mode_violation, known_absent_letter_mask, predictive_search_mode,
        write_predictive_artifact,
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
            artifact_dir: pattern_root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    #[test]
    fn parse_observations_rejects_length_mismatch() {
        let error = Solver::parse_observations(&["crane".into()], &[]).expect_err("must fail");
        assert!(error.to_string().contains("same number"));
    }

    #[test]
    fn absurdle_comparator_prefers_smaller_worst_bucket() {
        let better = AbsurdleSuggestion {
            word: "crane".into(),
            entropy: 2.0,
            largest_bucket_size: 4,
            second_largest_bucket_size: 2,
            multi_answer_bucket_count: 1,
        };
        let worse = AbsurdleSuggestion {
            word: "slate".into(),
            entropy: 3.5,
            largest_bucket_size: 5,
            second_largest_bucket_size: 1,
            multi_answer_bucket_count: 1,
        };
        assert_eq!(
            compare_absurdle_suggestions(&better, &worse),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn absurdle_comparator_breaks_ties_with_second_largest_bucket() {
        let better = AbsurdleSuggestion {
            word: "crane".into(),
            entropy: 2.0,
            largest_bucket_size: 4,
            second_largest_bucket_size: 1,
            multi_answer_bucket_count: 1,
        };
        let worse = AbsurdleSuggestion {
            word: "slate".into(),
            entropy: 2.0,
            largest_bucket_size: 4,
            second_largest_bucket_size: 2,
            multi_answer_bucket_count: 1,
        };
        assert_eq!(
            compare_absurdle_suggestions(&better, &worse),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn absurdle_apply_history_matches_wordle_filtering() {
        let solver = test_solver(&["cigar", "rebut", "sissy"]);
        let pattern = score_guess("cigar", "rebut");
        let absurdle = solver
            .absurdle_apply_history(&[("cigar".to_string(), pattern)])
            .expect("state");
        let wordle = solver
            .apply_history(
                NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid"),
                &[("cigar".to_string(), pattern)],
            )
            .expect("state");
        assert_eq!(absurdle.surviving, wordle.surviving);
    }

    #[test]
    fn target_feedback_matches_expected_fixture() {
        assert_eq!(
            format_feedback_letters(score_guess("lilly", "alley")),
            "ybgbg"
        );
    }

    #[test]
    fn known_absent_letters_ignore_letters_seen_as_present() {
        let observations = vec![("slate".to_string(), 0)];
        let mask = known_absent_letter_mask(&observations);
        assert_eq!(count_masked_letters("crony", mask), 0);
        assert_eq!(count_masked_letters("stare", mask), 4);

        let observations = vec![
            ("slate".to_string(), 0),
            ("crony".to_string(), score_guess("crony", "cigar")),
        ];
        let mask = known_absent_letter_mask(&observations);
        assert_eq!(count_masked_letters("crony", mask), 3);
        assert_eq!(count_masked_letters("cigar", mask), 1);
    }

    #[test]
    fn hard_mode_requires_green_positions_and_yellow_letters() {
        let observations = vec![("crane".to_string(), score_guess("crane", "cigar"))];
        assert_eq!(
            hard_mode_violation(&observations, "chair").expect("must fail"),
            "hard mode forbids A in position 3"
        );
        assert!(hard_mode_violation(&observations, "cigar").is_none());
    }

    #[test]
    fn hard_mode_requires_repeated_revealed_letters() {
        let observations = vec![("added".to_string(), score_guess("added", "dread"))];
        let error = hard_mode_violation(&observations, "tread").expect("must fail");
        assert!(error.contains("2 occurrences of D"));
        assert!(hard_mode_violation(&observations, "dread").is_none());
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
    fn initial_state_keeps_recent_zero_weight_answers_as_tiny_fallbacks() {
        let guesses = vec!["noisy".to_string()];
        let answers = vec![AnswerRecord {
            word: "noisy".to_string(),
            in_seed: false,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![NaiveDate::from_ymd_opt(2025, 9, 14).expect("valid")],
        }];
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let pattern_root: PathBuf =
            std::env::temp_dir().join(format!("maybe-wordle-zero-weight-test-{unique}"));
        let _ = std::fs::remove_dir_all(&pattern_root);
        std::fs::create_dir_all(&pattern_root).expect("pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&pattern_root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        let mut config = PriorConfig::default();
        config.cooldown_days = 365;
        config.cooldown_floor = 0.0;
        let solver = Solver {
            config,
            mode: WeightMode::Weighted,
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
            artifact_dir: pattern_root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        };
        let state = solver.initial_state(NaiveDate::from_ymd_opt(2026, 3, 10).expect("valid"));
        assert_eq!(state.surviving.len(), 1);
        assert!(state.total_weight > 0.0);
        assert!(state.weights[0] > 0.0);
        assert!(state.weights[0] < 0.001);
    }

    #[test]
    fn suggestions_for_history_populates_session_opener_cache_without_disk_books() {
        let guesses = vec![
            "cigar".to_string(),
            "rebut".to_string(),
            "sissy".to_string(),
        ];
        let answers = guesses
            .iter()
            .map(|word| AnswerRecord {
                word: word.clone(),
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
        let pattern_root =
            std::env::temp_dir().join(format!("maybe-wordle-session-cache-test-{unique}"));
        let _ = std::fs::remove_dir_all(&pattern_root);
        std::fs::create_dir_all(&pattern_root).expect("pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&pattern_root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        let mut config = PriorConfig::default();
        config.session_window_days = 1;
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        let solver = Solver {
            config,
            mode: WeightMode::Weighted,
            variant: ModelVariant::SeedPlusHistory,
            guesses: guesses.clone(),
            answers,
            history_dates: vec![NytDailyEntry {
                id: Some(1),
                solution: "cigar".to_string(),
                print_date: as_of,
                days_since_launch: Some(1),
                editor: None,
            }],
            exact_small_state_table: SmallStateTable::build(4),
            pattern_table,
            guess_index: guesses
                .iter()
                .enumerate()
                .map(|(index, guess)| (guess.clone(), index))
                .collect::<HashMap<_, _>>(),
            artifact_dir: pattern_root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        };
        assert_eq!(
            solver
                .session_opener_cache
                .lock()
                .expect("session opener cache")
                .len(),
            0
        );
        let suggestions = solver
            .suggestions_for_history(as_of, &[], 1)
            .expect("session suggestions");
        assert!(!suggestions.is_empty());
        assert_eq!(
            solver
                .session_opener_cache
                .lock()
                .expect("session opener cache")
                .len(),
            1
        );
    }

    #[test]
    fn suggestions_for_history_populates_session_reply_cache_without_disk_books() {
        let guesses = vec![
            "cigar".to_string(),
            "rebut".to_string(),
            "sissy".to_string(),
            "humph".to_string(),
        ];
        let answers = guesses
            .iter()
            .map(|word| AnswerRecord {
                word: word.clone(),
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
        let pattern_root =
            std::env::temp_dir().join(format!("maybe-wordle-session-reply-cache-test-{unique}"));
        let _ = std::fs::remove_dir_all(&pattern_root);
        std::fs::create_dir_all(&pattern_root).expect("pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&pattern_root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        let solver = Solver {
            config: PriorConfig::default(),
            mode: WeightMode::Weighted,
            variant: ModelVariant::SeedPlusHistory,
            guesses: guesses.clone(),
            answers,
            history_dates: vec![NytDailyEntry {
                id: Some(1),
                solution: "rebut".to_string(),
                print_date: as_of,
                days_since_launch: Some(1),
                editor: None,
            }],
            exact_small_state_table: SmallStateTable::build(4),
            pattern_table,
            guess_index: guesses
                .iter()
                .enumerate()
                .map(|(index, guess)| (guess.clone(), index))
                .collect::<HashMap<_, _>>(),
            artifact_dir: pattern_root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        };
        let feedback = score_guess("cigar", "rebut");
        let suggestions = solver
            .suggestions_for_history(as_of, &[("cigar".to_string(), feedback)], 1)
            .expect("session reply suggestions");
        assert!(!suggestions.is_empty());
        assert_eq!(
            solver
                .session_reply_cache
                .lock()
                .expect("session reply cache")
                .len(),
            1
        );
    }

    #[test]
    fn suggestions_for_history_populates_session_third_cache_without_disk_books() {
        let guesses = vec![
            "cigar".to_string(),
            "rebut".to_string(),
            "sissy".to_string(),
            "humph".to_string(),
            "awake".to_string(),
        ];
        let answers = guesses
            .iter()
            .map(|word| AnswerRecord {
                word: word.clone(),
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
        let pattern_root =
            std::env::temp_dir().join(format!("maybe-wordle-session-third-cache-test-{unique}"));
        let _ = std::fs::remove_dir_all(&pattern_root);
        std::fs::create_dir_all(&pattern_root).expect("pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&pattern_root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        let solver = Solver {
            config: PriorConfig::default(),
            mode: WeightMode::Weighted,
            variant: ModelVariant::SeedPlusHistory,
            guesses: guesses.clone(),
            answers,
            history_dates: vec![NytDailyEntry {
                id: Some(1),
                solution: "humph".to_string(),
                print_date: as_of,
                days_since_launch: Some(1),
                editor: None,
            }],
            exact_small_state_table: SmallStateTable::build(4),
            pattern_table,
            guess_index: guesses
                .iter()
                .enumerate()
                .map(|(index, guess)| (guess.clone(), index))
                .collect::<HashMap<_, _>>(),
            artifact_dir: pattern_root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        };
        let first_feedback = score_guess("cigar", "humph");
        let second_feedback = score_guess("rebut", "humph");
        let suggestions = solver
            .suggestions_for_history(
                as_of,
                &[
                    ("cigar".to_string(), first_feedback),
                    ("rebut".to_string(), second_feedback),
                ],
                1,
            )
            .expect("session third suggestions");
        assert!(!suggestions.is_empty());
        assert_eq!(
            solver
                .session_third_cache
                .lock()
                .expect("session third cache")
                .len(),
            1
        );
    }

    #[test]
    fn cached_predictive_choice_reads_third_turn_from_disk_book() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        let opener_pattern = score_guess("cigar", "humph");
        let reply_pattern = score_guess("rebut", "humph");
        let artifact = PredictiveReplyBookArtifact {
            identity: solver.predictive_book_identity(as_of),
            opener: "cigar".to_string(),
            replies: vec![PredictiveReplyEntry {
                feedback_pattern: opener_pattern,
                reply: "rebut".to_string(),
                surviving_answers: 2,
                proxy_cost: None,
                lookahead_cost: None,
                exact_cost: None,
                third_replies: vec![PredictiveThirdReplyEntry {
                    second_feedback_pattern: reply_pattern,
                    reply: "sissy".to_string(),
                    surviving_answers: 2,
                    proxy_cost: None,
                    lookahead_cost: None,
                    exact_cost: None,
                }],
            }],
        };
        write_predictive_artifact(&solver.reply_book_artifact_path(as_of), &artifact)
            .expect("write reply book");

        let choice = solver.cached_predictive_choice(
            as_of,
            &[
                ("cigar".to_string(), opener_pattern),
                ("rebut".to_string(), reply_pattern),
            ],
            false,
        );
        assert_eq!(choice.as_deref(), Some("sissy"));
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
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 2,
            largest_non_green_bucket_mass: 0.25,
            high_mass_ambiguous_bucket_count: 1,
            smoothness_penalty: 0.10,
            large_non_green_bucket_count: 1,
            dangerous_mass_bucket_count: 1,
            non_green_mass_in_large_buckets: 0.25,
            proxy_cost: 1.8,
            large_state_score: 1.0,
            posterior_answer_probability: 0.0,
        };
        let worse_proxy = GuessMetrics {
            guess_index: 1,
            entropy: 4.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            force_in_two: false,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 3,
            largest_non_green_bucket_mass: 0.35,
            high_mass_ambiguous_bucket_count: 2,
            smoothness_penalty: 0.35,
            large_non_green_bucket_count: 2,
            dangerous_mass_bucket_count: 2,
            non_green_mass_in_large_buckets: 0.35,
            proxy_cost: 2.2,
            large_state_score: 0.2,
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
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            large_non_green_bucket_count: 0,
            dangerous_mass_bucket_count: 0,
            non_green_mass_in_large_buckets: 0.0,
            proxy_cost: Some(2.0),
            large_state_score: Some(1.0),
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
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            large_non_green_bucket_count: 0,
            dangerous_mass_bucket_count: 0,
            non_green_mass_in_large_buckets: 0.0,
            proxy_cost: Some(2.0),
            large_state_score: Some(0.8),
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
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            large_non_green_bucket_count: 0,
            dangerous_mass_bucket_count: 0,
            non_green_mass_in_large_buckets: 0.0,
            proxy_cost: Some(2.0),
            large_state_score: Some(1.0),
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
            compare_exact_costs(
                &force,
                &non_force,
                force.exact_cost,
                non_force.exact_cost,
                false
            ),
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
                force.exact_cost,
                false,
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
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.20,
            large_non_green_bucket_count: 0,
            dangerous_mass_bucket_count: 0,
            non_green_mass_in_large_buckets: 0.0,
            proxy_cost: Some(2.0),
            large_state_score: Some(1.0),
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
        assert_eq!(
            compare_lookahead(&better, &force, false),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn large_state_split_ordering_deemphasizes_solve_probability() {
        let guesses = vec!["alpha".to_string(), "bravo".to_string()];
        let safer_split = GuessMetrics {
            guess_index: 0,
            entropy: 4.8,
            solve_probability: 0.0,
            expected_remaining: 3.0,
            force_in_two: false,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 4,
            largest_non_green_bucket_mass: 0.18,
            high_mass_ambiguous_bucket_count: 1,
            smoothness_penalty: 0.08,
            large_non_green_bucket_count: 1,
            dangerous_mass_bucket_count: 1,
            non_green_mass_in_large_buckets: 0.18,
            proxy_cost: 1.8,
            large_state_score: 1.2,
            posterior_answer_probability: 0.0,
        };
        let gambler = GuessMetrics {
            guess_index: 1,
            entropy: 4.2,
            solve_probability: 0.3,
            expected_remaining: 3.0,
            force_in_two: false,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 6,
            largest_non_green_bucket_mass: 0.32,
            high_mass_ambiguous_bucket_count: 2,
            smoothness_penalty: 0.28,
            large_non_green_bucket_count: 2,
            dangerous_mass_bucket_count: 2,
            non_green_mass_in_large_buckets: 0.32,
            proxy_cost: 1.8,
            large_state_score: 0.5,
            posterior_answer_probability: 0.4,
        };
        assert_eq!(
            compare_guess_metrics_for_state(&safer_split, &gambler, &guesses, true),
            std::cmp::Ordering::Less
        );
        let safer_suggestion = Suggestion {
            word: "alpha".into(),
            entropy: safer_split.entropy,
            solve_probability: safer_split.solve_probability,
            expected_remaining: safer_split.expected_remaining,
            force_in_two: safer_split.force_in_two,
            known_absent_letter_hits: safer_split.known_absent_letter_hits,
            worst_non_green_bucket_size: safer_split.worst_non_green_bucket_size,
            largest_non_green_bucket_mass: safer_split.largest_non_green_bucket_mass,
            large_non_green_bucket_count: safer_split.large_non_green_bucket_count,
            dangerous_mass_bucket_count: safer_split.dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets: safer_split.non_green_mass_in_large_buckets,
            proxy_cost: Some(safer_split.proxy_cost),
            large_state_score: Some(safer_split.large_state_score),
            posterior_answer_probability: safer_split.posterior_answer_probability,
            lookahead_cost: None,
            exact_cost: None,
        };
        let gambler_suggestion = Suggestion {
            word: "bravo".into(),
            entropy: gambler.entropy,
            solve_probability: gambler.solve_probability,
            expected_remaining: gambler.expected_remaining,
            force_in_two: gambler.force_in_two,
            known_absent_letter_hits: gambler.known_absent_letter_hits,
            worst_non_green_bucket_size: gambler.worst_non_green_bucket_size,
            largest_non_green_bucket_mass: gambler.largest_non_green_bucket_mass,
            large_non_green_bucket_count: gambler.large_non_green_bucket_count,
            dangerous_mass_bucket_count: gambler.dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets: gambler.non_green_mass_in_large_buckets,
            proxy_cost: Some(gambler.proxy_cost),
            large_state_score: Some(gambler.large_state_score),
            posterior_answer_probability: gambler.posterior_answer_probability,
            lookahead_cost: None,
            exact_cost: None,
        };
        assert_eq!(
            compare_suggestions_for_state(&safer_suggestion, &gambler_suggestion, true),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn large_state_split_ordering_prefers_fewer_known_absent_letters_on_entropy_ties() {
        let guesses = vec!["alpha".to_string(), "bravo".to_string()];
        let cleaner = GuessMetrics {
            guess_index: 0,
            entropy: 4.5,
            solve_probability: 0.0,
            expected_remaining: 3.0,
            force_in_two: false,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 4,
            largest_non_green_bucket_mass: 0.20,
            high_mass_ambiguous_bucket_count: 1,
            smoothness_penalty: 0.10,
            large_non_green_bucket_count: 1,
            dangerous_mass_bucket_count: 1,
            non_green_mass_in_large_buckets: 0.20,
            proxy_cost: 1.8,
            large_state_score: 1.0,
            posterior_answer_probability: 0.0,
        };
        let grayer = GuessMetrics {
            guess_index: 1,
            known_absent_letter_hits: 2,
            ..cleaner
        };
        assert_eq!(
            compare_guess_metrics_for_state(&cleaner, &grayer, &guesses, true),
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
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 2,
                largest_non_green_bucket_mass: 0.40,
                large_non_green_bucket_count: 1,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.40,
                proxy_cost: Some(1.5),
                large_state_score: Some(0.6),
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
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 2,
                largest_non_green_bucket_mass: 0.35,
                large_non_green_bucket_count: 1,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.35,
                proxy_cost: Some(1.6),
                large_state_score: Some(0.7),
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
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 4,
                largest_non_green_bucket_mass: 0.45,
                large_non_green_bucket_count: 2,
                dangerous_mass_bucket_count: 2,
                non_green_mass_in_large_buckets: 0.45,
                proxy_cost: Some(1.5),
                large_state_score: Some(0.5),
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
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 3,
                largest_non_green_bucket_mass: 0.25,
                large_non_green_bucket_count: 1,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.25,
                proxy_cost: Some(1.6),
                large_state_score: Some(0.8),
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
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 1,
                largest_non_green_bucket_mass: 0.05,
                large_non_green_bucket_count: 0,
                dangerous_mass_bucket_count: 0,
                non_green_mass_in_large_buckets: 0.0,
                proxy_cost: Some(1.7),
                large_state_score: Some(0.9),
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
    fn top_guess_indexes_for_subset_appends_surviving_answers_after_cutoff() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let subset = vec![0, 1];
        let weights = vec![1.0; solver.answers.len()];

        let full = solver.top_guess_indexes_for_subset(&subset, &weights, solver.guesses.len());
        let shortlisted = solver.top_guess_indexes_for_subset(&subset, &weights, 1);

        assert_eq!(shortlisted.first(), full.first());
        assert!(shortlisted.contains(solver.guess_index.get("cigar").expect("cigar")));
        assert!(shortlisted.contains(solver.guess_index.get("rebut").expect("rebut")));
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
    fn lookahead_candidates_include_secondary_rankings() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let suggestions = vec![
            Suggestion {
                word: "cigar".into(),
                entropy: 3.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 5,
                largest_non_green_bucket_mass: 0.40,
                large_non_green_bucket_count: 2,
                dangerous_mass_bucket_count: 2,
                non_green_mass_in_large_buckets: 0.40,
                proxy_cost: Some(1.0),
                large_state_score: Some(1.0),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "rebut".into(),
                entropy: 5.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 4,
                largest_non_green_bucket_mass: 0.35,
                large_non_green_bucket_count: 1,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.35,
                proxy_cost: Some(1.2),
                large_state_score: Some(0.9),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "sissy".into(),
                entropy: 2.5,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 1,
                largest_non_green_bucket_mass: 0.10,
                large_non_green_bucket_count: 0,
                dangerous_mass_bucket_count: 0,
                non_green_mass_in_large_buckets: 0.0,
                proxy_cost: Some(1.3),
                large_state_score: Some(0.8),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "humph".into(),
                entropy: 2.0,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 3,
                largest_non_green_bucket_mass: 0.15,
                large_non_green_bucket_count: 1,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.15,
                proxy_cost: Some(1.4),
                large_state_score: Some(0.7),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
        ];

        let candidates = solver
            .collect_lookahead_candidates(&suggestions, 32, false, 1)
            .expect("candidates");

        assert!(candidates.contains(solver.guess_index.get("cigar").expect("cigar")));
        assert!(candidates.contains(solver.guess_index.get("rebut").expect("rebut")));
        assert!(candidates.contains(solver.guess_index.get("sissy").expect("sissy")));
    }

    #[test]
    fn suggestion_tie_breaks_keep_trap_signals() {
        let safer = Suggestion {
            word: "alpha".into(),
            entropy: 3.0,
            solve_probability: 0.0,
            expected_remaining: 2.0,
            force_in_two: false,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 3,
            largest_non_green_bucket_mass: 0.20,
            large_non_green_bucket_count: 1,
            dangerous_mass_bucket_count: 1,
            non_green_mass_in_large_buckets: 0.20,
            proxy_cost: Some(2.0),
            large_state_score: Some(1.0),
            posterior_answer_probability: 0.0,
            lookahead_cost: Some(3.0),
            exact_cost: Some(3.0),
        };
        let trap_heavier = Suggestion {
            word: "bravo".into(),
            large_non_green_bucket_count: 3,
            dangerous_mass_bucket_count: 2,
            non_green_mass_in_large_buckets: 0.45,
            ..safer.clone()
        };
        assert_eq!(
            compare_suggestions_for_state(&safer, &trap_heavier, false),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_lookahead(&safer, &trap_heavier, false),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_exact_costs(
                &safer,
                &trap_heavier,
                safer.exact_cost,
                trap_heavier.exact_cost,
                false
            ),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn medium_state_uses_deeper_force_in_two_scan_and_pools() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake"]);
        solver.config.lookahead_root_force_in_two_scan = 2;
        solver.config.medium_state_force_in_two_scan = 5;
        solver.config.medium_state_lookahead_threshold = 80;
        solver.config.lookahead_candidate_pool = 12;
        solver.config.medium_state_lookahead_candidate_pool = 36;
        solver.config.lookahead_reply_pool = 6;
        solver.config.medium_state_lookahead_reply_pool = 18;

        assert_eq!(solver.force_in_two_scan_for_state(120), 2);
        assert_eq!(solver.force_in_two_scan_for_state(72), 5);
        assert_eq!(solver.lookahead_candidate_pool_for_state(120), 12);
        assert_eq!(solver.lookahead_candidate_pool_for_state(72), 36);
        assert_eq!(solver.lookahead_reply_pool_for_state(120), 6);
        assert_eq!(solver.lookahead_reply_pool_for_state(72), 18);
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
    fn pool_expansion_only_applies_when_enabled() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let suggestions = vec![
            Suggestion {
                word: "cigar".into(),
                entropy: 3.2,
                solve_probability: 0.0,
                expected_remaining: 2.0,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 4,
                largest_non_green_bucket_mass: 0.30,
                large_non_green_bucket_count: 2,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.30,
                proxy_cost: Some(1.00),
                large_state_score: Some(1.00),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "rebut".into(),
                entropy: 3.1,
                solve_probability: 0.0,
                expected_remaining: 2.1,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 4,
                largest_non_green_bucket_mass: 0.31,
                large_non_green_bucket_count: 2,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.31,
                proxy_cost: Some(1.01),
                large_state_score: Some(0.99),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "sissy".into(),
                entropy: 3.0,
                solve_probability: 0.0,
                expected_remaining: 2.2,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 4,
                largest_non_green_bucket_mass: 0.32,
                large_non_green_bucket_count: 2,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.32,
                proxy_cost: Some(1.02),
                large_state_score: Some(0.98),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
            Suggestion {
                word: "humph".into(),
                entropy: 2.9,
                solve_probability: 0.0,
                expected_remaining: 2.3,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 4,
                largest_non_green_bucket_mass: 0.33,
                large_non_green_bucket_count: 2,
                dangerous_mass_bucket_count: 1,
                non_green_mass_in_large_buckets: 0.33,
                proxy_cost: Some(1.03),
                large_state_score: Some(0.97),
                posterior_answer_probability: 0.0,
                lookahead_cost: None,
                exact_cost: None,
            },
        ];
        let assessment = StateDangerAssessment {
            danger_score: 0.9,
            dangerous_lookahead: true,
            dangerous_exact: true,
        };

        assert_eq!(
            solver.expanded_pool_size(&suggestions, 2, true, false, assessment),
            2
        );
        assert!(solver.expanded_pool_size(&suggestions, 2, true, true, assessment) > 2);
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
    fn normalized_concentration_penalty_prefers_smoother_partitions() {
        let smooth = super::normalized_concentration_penalty(1.0, 0.25 * 0.25 * 4.0, 4);
        let spiky =
            super::normalized_concentration_penalty(1.0, 0.70 * 0.70 + 0.10 * 0.10 * 3.0, 4);
        assert_eq!(smooth, 0.0);
        assert!(spiky > smooth);
        assert!(spiky <= 1.0);
    }

    #[test]
    fn aggregated_lookahead_trap_penalty_grows_with_compound_traps() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let mild = solver.aggregate_lookahead_trap_penalty(0.20, 1, 1, 0.20, 0);
        let severe = solver.aggregate_lookahead_trap_penalty(0.20, 3, 2, 0.45, 2);
        assert!(severe > mild);
    }

    #[test]
    fn detailed_run_tracks_step_diagnostics() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let run = solver
            .solve_target_from_state_detailed(
                "cigar",
                Solver::today(),
                Solver::today(),
                3,
                PredictiveBookUsage::Full,
            )
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
            artifact_dir: pattern_root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
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
