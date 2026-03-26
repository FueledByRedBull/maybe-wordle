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
    predictive::{
        PredictivePromotionSource, PredictiveRegime, PredictiveStateSummary,
        PredictiveSuggestRequest, PredictiveSuggestResponse, PredictiveSuggestionMode,
        RecoveryMode,
    },
    scoring::{
        ALL_GREEN_PATTERN, PATTERN_SPACE, decode_feedback, format_feedback_letters, parse_feedback,
        score_guess,
    },
    small_state::SmallStateTable,
};

mod books;
mod eval;
mod ranking;
mod search;
mod state;

use self::books::write_predictive_artifact;
#[allow(unused_imports)]
use self::state::{hard_mode_violation_message as hard_mode_violation, *};
#[allow(unused_imports)]
use self::{eval::*, ranking::*, search::*};

const PROXY_CALIBRATION_MAX_STEPS: usize = 3;
const PROXY_CALIBRATION_MAX_CANDIDATES_PER_STATE: usize = 10;
const PROXY_CALIBRATION_MAX_SURVIVORS_FOR_FORCED_ROWS: usize = 192;
const PROXY_CALIBRATION_MAX_GAME_SECONDS: f64 = 20.0;
const HARD_MODE_WORD_LENGTH: usize = 5;
const THREE_GUESS_ROOT_CANDIDATE_LIMIT: usize = 1;
const THREE_GUESS_REPLY_CANDIDATE_LIMIT: usize = 4;
const MEDIUM_SECOND_GUESS_COVERAGE_POOL: usize = 24;
const THREE_SOLVE_CHILD_CAP: usize = 24;
const OPENER_HOLDOUT_SHORTLIST: usize = 4;
const OPENER_ARTIFACT_FRESHNESS_DAYS: u64 = 14;

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
    pub modeled_weights: Vec<f64>,
    pub recovery_weights: Vec<f64>,
    pub weights: Vec<f64>,
    pub modeled_total_weight: f64,
    pub total_weight: f64,
    pub recovery_mode_used: Option<RecoveryMode>,
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

#[derive(Clone, Debug, Serialize)]
pub struct LiveConfigEvaluation {
    pub config: PriorConfig,
    pub average_guesses: f64,
    pub failures: usize,
    pub coverage_gaps: usize,
    pub latency_p95_ms: f64,
    pub hard_case_average_guesses: f64,
    pub hard_case_failures: usize,
}

#[derive(Clone, Debug)]
pub struct ThreeGuessGapCase {
    pub target: String,
    pub date: NaiveDate,
    pub base_guesses: usize,
    pub aggressive_guesses: usize,
    pub best_forced_guesses: usize,
    pub converted_by_aggressive: bool,
    pub converted_by_targeted_search: bool,
    pub base_path: Vec<String>,
    pub aggressive_path: Vec<String>,
    pub best_forced_path: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ThreeGuessGapReport {
    pub games: usize,
    pub base_average_guesses: f64,
    pub aggressive_case_average_guesses: f64,
    pub base_four_guess_cases: usize,
    pub aggressive_four_guess_cases: usize,
    pub converted_by_aggressive: usize,
    pub converted_by_targeted_search: usize,
    pub cases: Vec<ThreeGuessGapCase>,
}

#[derive(Clone, Debug)]
pub struct FourGuessTarget {
    pub target: String,
    pub date: NaiveDate,
    pub base_path: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct FourGuessOpenerEvaluation {
    pub opener: String,
    pub average_guesses: f64,
    pub three_guess_solves: usize,
    pub failures: usize,
    pub p95_guesses: usize,
    pub max_guesses: usize,
}

#[derive(Clone, Debug)]
pub struct FourGuessOpenerReport {
    pub games: usize,
    pub targets: Vec<FourGuessTarget>,
    pub evaluations: Vec<FourGuessOpenerEvaluation>,
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
    pub four_guess_games: usize,
    pub average_guesses: f64,
    pub failures: usize,
    pub holdout_games: usize,
    pub holdout_four_guess_games: usize,
    pub holdout_average_guesses: f64,
    pub holdout_failures: usize,
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
    policy_id: String,
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
    #[serde(default)]
    four_guess_games: usize,
    average_guesses: f64,
    failures: usize,
    #[serde(default)]
    holdout_window_start: Option<NaiveDate>,
    #[serde(default)]
    holdout_window_end: Option<NaiveDate>,
    #[serde(default)]
    holdout_games: usize,
    #[serde(default)]
    holdout_four_guess_games: usize,
    #[serde(default)]
    holdout_average_guesses: f64,
    #[serde(default)]
    holdout_failures: usize,
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
    four_guess_games: usize,
    average_guesses: f64,
    p95_guesses: usize,
    max_guesses: usize,
    failures: usize,
}

#[derive(Clone, Debug)]
struct ValidatedOpenerEvaluation {
    word: String,
    primary: ForcedOpenerEvaluation,
    holdout: Option<ForcedOpenerEvaluation>,
}

#[derive(Clone, Copy, Debug)]
struct ForcedSolveScore {
    guesses: usize,
    solved: bool,
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

#[derive(Clone, Copy, Debug, Default)]
struct ThreeSolveCoverage {
    mass: f64,
    uncovered_answers: usize,
    uncovered_buckets: usize,
}

type BookTargetWindow = (NaiveDate, NaiveDate, Vec<(NaiveDate, String)>);

#[derive(Clone, Copy, Debug)]
struct GuessMetricContext<'a> {
    subset: &'a [usize],
    weights: &'a [f64],
    total_weight: f64,
    small_state_table: &'a SmallStateTable,
    posterior_answer_probability: f64,
}

struct LookaheadCostContext<'a> {
    subset: &'a [usize],
    weights: &'a [f64],
    expanded: bool,
    exact_memo: &'a mut PredictiveMemoMap<ExactSubsetKey, f64>,
    exact_scratch: &'a mut ExactSearchScratch,
    lookahead_memo: &'a mut PredictiveMemoMap<ExactSubsetKey, f64>,
}

struct ExactCostContext<'a> {
    subset: &'a [usize],
    weights: &'a [f64],
    small_state_table: &'a SmallStateTable,
    memo: &'a mut PredictiveMemoMap<ExactSubsetKey, f64>,
    best_bound: f64,
    scratch: &'a mut ExactSearchScratch,
    depth: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProxyRowStats {
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
}

impl ProxyRowStats {
    fn from_metric(metric: &GuessMetrics) -> Self {
        Self {
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
        }
    }

    fn from_calibration_row(row: &ProxyCalibrationRow) -> Self {
        Self {
            entropy: row.entropy,
            largest_non_green_bucket_mass: row.largest_non_green_bucket_mass,
            worst_non_green_bucket_size: row.worst_non_green_bucket_size,
            high_mass_ambiguous_bucket_count: row.high_mass_ambiguous_bucket_count,
            proxy_cost: row.proxy_cost,
            solve_probability: row.solve_probability,
            posterior_answer_probability: row.posterior_answer_probability,
            smoothness_penalty: row.smoothness_penalty,
            known_absent_letter_hits: row.known_absent_letter_hits,
            large_non_green_bucket_count: row.large_non_green_bucket_count,
            dangerous_mass_bucket_count: row.dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets: row.non_green_mass_in_large_buckets,
        }
    }
}

#[derive(Clone, Debug)]
struct SuggestionBatch {
    suggestions: Vec<Suggestion>,
    promoted_word: Option<String>,
    promotion_source: Option<PredictivePromotionSource>,
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

#[derive(Clone, Debug)]
struct PromotedPredictiveChoice {
    word: String,
    source: PredictivePromotionSource,
}

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
}

#[derive(Clone, Debug)]
struct PriorMetrics {
    target_probability: f64,
    target_rank: usize,
    log_loss: f64,
    brier: f64,
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
        ExactSuggestionMode, ForcedOpenerEvaluation, GuessMetrics, PredictiveBookUsage,
        PredictiveMemoMap, PredictiveOpenerArtifact, PredictiveReplyBookArtifact,
        PredictiveReplyEntry, PredictiveSearchMode, PredictiveThirdReplyEntry, Solver,
        StateDangerAssessment, Suggestion, compare_absurdle_suggestions, compare_exact_costs,
        compare_forced_openers, compare_guess_metrics, compare_guess_metrics_for_state,
        compare_lookahead, compare_suggestions, compare_suggestions_for_state,
        count_masked_letters, exact_suggestion_mode, hard_mode_violation, known_absent_letter_mask,
        predictive_search_mode, should_replace_forced_opener,
    };

    use super::books::write_predictive_artifact;

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
    fn forced_opener_comparator_penalizes_four_guess_paths_before_average() {
        let guesses = vec!["crane".to_string(), "slate".to_string()];
        let safer = ForcedOpenerEvaluation {
            guess_index: 0,
            games: 30,
            four_guess_games: 2,
            average_guesses: 3.20,
            p95_guesses: 4,
            max_guesses: 4,
            failures: 0,
        };
        let riskier = ForcedOpenerEvaluation {
            guess_index: 1,
            games: 30,
            four_guess_games: 4,
            average_guesses: 3.18,
            p95_guesses: 4,
            max_guesses: 4,
            failures: 0,
        };
        assert_eq!(
            compare_forced_openers(&safer, &riskier, &guesses),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn stable_opener_switch_rejects_holdout_regressions() {
        let guesses = vec!["crane".to_string(), "slate".to_string()];
        let incumbent_primary = ForcedOpenerEvaluation {
            guess_index: 0,
            games: 30,
            four_guess_games: 4,
            average_guesses: 3.30,
            p95_guesses: 4,
            max_guesses: 4,
            failures: 0,
        };
        let candidate_primary = ForcedOpenerEvaluation {
            guess_index: 1,
            games: 30,
            four_guess_games: 2,
            average_guesses: 3.20,
            p95_guesses: 4,
            max_guesses: 4,
            failures: 0,
        };
        let incumbent_holdout = ForcedOpenerEvaluation {
            guess_index: 0,
            games: 30,
            four_guess_games: 2,
            average_guesses: 3.25,
            p95_guesses: 4,
            max_guesses: 4,
            failures: 0,
        };
        let candidate_holdout = ForcedOpenerEvaluation {
            guess_index: 1,
            games: 30,
            four_guess_games: 5,
            average_guesses: 3.35,
            p95_guesses: 4,
            max_guesses: 4,
            failures: 0,
        };
        assert!(!should_replace_forced_opener(
            &candidate_primary,
            Some(&candidate_holdout),
            &incumbent_primary,
            Some(&incumbent_holdout),
            &guesses,
        ));
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
        let config = PriorConfig {
            cooldown_days: 365,
            cooldown_floor: 0.0,
            ..PriorConfig::default()
        };
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
        let config = PriorConfig {
            session_window_days: 1,
            ..PriorConfig::default()
        };
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
        assert_eq!(choice.map(|choice| choice.word), Some("sissy".to_string()));
    }

    #[test]
    fn cached_predictive_choice_uses_recent_opener_artifact_when_exact_date_is_missing() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let artifact_date = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        let request_date = NaiveDate::from_ymd_opt(2026, 3, 16).expect("valid");
        let artifact = PredictiveOpenerArtifact {
            identity: solver.predictive_book_identity(artifact_date),
            opener: "cigar".to_string(),
            search_window_start: NaiveDate::from_ymd_opt(2026, 2, 8).expect("valid"),
            search_window_end: artifact_date,
            games: 30,
            four_guess_games: 10,
            average_guesses: 3.3,
            failures: 0,
            holdout_window_start: None,
            holdout_window_end: None,
            holdout_games: 0,
            holdout_four_guess_games: 0,
            holdout_average_guesses: 0.0,
            holdout_failures: 0,
            proxy_cost: None,
            lookahead_cost: None,
            exact_cost: None,
        };
        write_predictive_artifact(&solver.opener_artifact_path(artifact_date), &artifact)
            .expect("write opener artifact");

        let choice = solver.cached_predictive_choice(request_date, &[], false);
        assert_eq!(choice.map(|choice| choice.word), Some("cigar".to_string()));
    }

    #[test]
    fn cached_predictive_choice_prefers_exact_date_opener_artifact_over_recent_one() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let older_date = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        let exact_date = NaiveDate::from_ymd_opt(2026, 3, 16).expect("valid");
        let older = PredictiveOpenerArtifact {
            identity: solver.predictive_book_identity(older_date),
            opener: "cigar".to_string(),
            search_window_start: NaiveDate::from_ymd_opt(2026, 2, 8).expect("valid"),
            search_window_end: older_date,
            games: 30,
            four_guess_games: 10,
            average_guesses: 3.3,
            failures: 0,
            holdout_window_start: None,
            holdout_window_end: None,
            holdout_games: 0,
            holdout_four_guess_games: 0,
            holdout_average_guesses: 0.0,
            holdout_failures: 0,
            proxy_cost: None,
            lookahead_cost: None,
            exact_cost: None,
        };
        let exact = PredictiveOpenerArtifact {
            identity: solver.predictive_book_identity(exact_date),
            opener: "rebut".to_string(),
            search_window_start: NaiveDate::from_ymd_opt(2026, 2, 15).expect("valid"),
            search_window_end: exact_date,
            games: 30,
            four_guess_games: 8,
            average_guesses: 3.2,
            failures: 0,
            holdout_window_start: None,
            holdout_window_end: None,
            holdout_games: 0,
            holdout_four_guess_games: 0,
            holdout_average_guesses: 0.0,
            holdout_failures: 0,
            proxy_cost: None,
            lookahead_cost: None,
            exact_cost: None,
        };
        write_predictive_artifact(&solver.opener_artifact_path(older_date), &older)
            .expect("write older opener artifact");
        write_predictive_artifact(&solver.opener_artifact_path(exact_date), &exact)
            .expect("write exact opener artifact");

        let choice = solver.cached_predictive_choice(exact_date, &[], false);
        assert_eq!(choice.map(|choice| choice.word), Some("rebut".to_string()));
    }

    #[test]
    fn live_backtest_works_without_disk_books() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        solver.history_dates = vec![NytDailyEntry {
            id: Some(1),
            solution: "cigar".to_string(),
            print_date: date,
            days_since_launch: Some(1),
            editor: None,
        }];

        let report = solver
            .backtest_detailed_with_book_usage(date, date, 3, PredictiveBookUsage::None)
            .expect("live backtest");
        assert_eq!(report.summary.games, 1);
        assert_eq!(report.summary.coverage_gaps, 0);
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
        let config = PriorConfig {
            exact_threshold: 16,
            exact_exhaustive_threshold: 8,
            ..PriorConfig::default()
        };
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
    fn medium_second_guess_coverage_overrides_weaker_proxy() {
        let guesses = vec!["alpha".to_string(), "bravo".to_string()];
        let baseline = GuessMetrics {
            guess_index: 0,
            entropy: 3.5,
            solve_probability: 0.1,
            expected_remaining: 2.5,
            force_in_two: false,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 3,
            largest_non_green_bucket_mass: 0.24,
            high_mass_ambiguous_bucket_count: 1,
            smoothness_penalty: 0.12,
            large_non_green_bucket_count: 1,
            dangerous_mass_bucket_count: 1,
            non_green_mass_in_large_buckets: 0.24,
            proxy_cost: 1.8,
            large_state_score: 0.9,
            posterior_answer_probability: 0.0,
        };
        let stronger_proxy = GuessMetrics {
            guess_index: 1,
            proxy_cost: 1.6,
            ..baseline
        };
        let mut coverage = super::FxHashMap::default();
        coverage.insert(
            0,
            super::ThreeSolveCoverage {
                mass: 0.90,
                uncovered_answers: 1,
                uncovered_buckets: 1,
            },
        );
        coverage.insert(
            1,
            super::ThreeSolveCoverage {
                mass: 0.40,
                uncovered_answers: 4,
                uncovered_buckets: 2,
            },
        );
        assert_eq!(
            super::compare_guess_metrics_with_coverage(
                &baseline,
                &stronger_proxy,
                &guesses,
                false,
                &coverage
            ),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn medium_second_guess_suggestion_coverage_overrides_proxy_tie_break() {
        let mut coverage = super::FxHashMap::default();
        coverage.insert(
            0,
            super::ThreeSolveCoverage {
                mass: 0.85,
                uncovered_answers: 1,
                uncovered_buckets: 1,
            },
        );
        coverage.insert(
            1,
            super::ThreeSolveCoverage {
                mass: 0.20,
                uncovered_answers: 5,
                uncovered_buckets: 3,
            },
        );
        let guess_index =
            HashMap::from([("alpha".to_string(), 0usize), ("bravo".to_string(), 1usize)]);
        let better_coverage = Suggestion {
            word: "alpha".into(),
            entropy: 3.0,
            solve_probability: 0.1,
            expected_remaining: 2.0,
            force_in_two: false,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 2,
            largest_non_green_bucket_mass: 0.20,
            large_non_green_bucket_count: 1,
            dangerous_mass_bucket_count: 1,
            non_green_mass_in_large_buckets: 0.20,
            proxy_cost: Some(2.2),
            large_state_score: Some(0.8),
            posterior_answer_probability: 0.0,
            lookahead_cost: Some(3.0),
            exact_cost: None,
        };
        let stronger_proxy = Suggestion {
            word: "bravo".into(),
            proxy_cost: Some(1.8),
            lookahead_cost: Some(2.8),
            ..better_coverage.clone()
        };
        assert_eq!(
            super::compare_suggestions_with_coverage(
                &better_coverage,
                &stronger_proxy,
                false,
                &guess_index,
                &coverage,
            ),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn pooled_exact_candidates_keep_surviving_answers() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let state = super::SolveState {
            surviving: vec![0, 1],
            modeled_weights: vec![1.0; solver.answers.len()],
            recovery_weights: vec![1.0; solver.answers.len()],
            weights: vec![1.0; solver.answers.len()],
            modeled_total_weight: 2.0,
            total_weight: 2.0,
            recovery_mode_used: None,
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
            modeled_weights: vec![1.0; solver.answers.len()],
            recovery_weights: vec![1.0; solver.answers.len()],
            weights: vec![1.0; solver.answers.len()],
            modeled_total_weight: 2.0,
            total_weight: 2.0,
            recovery_mode_used: None,
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
            &mut super::GuessMetricScratch::new(),
            super::GuessMetricContext {
                subset: &subset,
                weights: &weights,
                total_weight,
                small_state_table: &solver.exact_small_state_table,
                posterior_answer_probability: 0.0,
            },
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
