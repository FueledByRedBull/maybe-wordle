use std::{
    array,
    collections::{BTreeSet, HashMap, HashSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
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
    experiments::exhaustive_cost::{
        ChronologicalSplitMetadata, DatasetProvenance, DatasetSplit, DatasetSplitMetadata,
        ExactState, ExhaustiveCostCheckpoint, ExhaustiveCostDatasetArtifact, ExhaustiveCostRow,
        ExhaustiveProgress, ReplayIdentityInput, ResourceBudget,
    },
    experiments::{
        BootstrapConfig, DateRange, EvaluationPlan, ExperimentArtifactMode, GameOutcome,
        PairedDifference, ParameterRegistry, PredictiveConfigProfile, PredictiveExperimentMatrix,
        PredictiveMetrics, PriorEvidenceMetrics, RankedProbabilityObservation, RollingOriginConfig,
        StudyMeasurement, StudyProvenance, StudySearchStrategy, StudySpec, StudyStage, StudyState,
        StudyTrial, TrialStatus, build_rolling_origin_plan, default_diagnostic_suite,
        generate_candidates, predictive_parameter_registry, score_multiclass_probabilities,
        summarize_predictive_outcomes, summarize_ranked_probability_observations,
    },
    model::{
        AnswerRecord, ModelVariant, WeightMode, load_model, load_model_with_variant,
        weight_snapshot_for_mode,
    },
    pattern_table::PatternTable,
    predictive::{
        PredictiveCandidateSummary, PredictivePromotionSource, PredictiveRegime,
        PredictiveStateSummary, PredictiveSuggestRequest, PredictiveSuggestResponse,
        PredictiveSuggestionMode, RecoveryMode,
    },
    scoring::{
        ALL_GREEN_PATTERN, PATTERN_SPACE, decode_feedback, format_feedback_letters, parse_feedback,
        score_guess,
    },
    small_state::SmallStateTable,
};

mod artifact_identity;
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

pub(crate) fn predictive_source_identity(paths: &ProjectPaths) -> Result<String> {
    rolling_source_identity(paths)
}

pub(crate) fn predictive_executable_fingerprint() -> Result<String> {
    current_executable_fingerprint()
}

pub(crate) fn ensure_predictive_source_identity(
    paths: &ProjectPaths,
    expected: &str,
) -> Result<()> {
    ensure_rolling_source_identity(paths, expected)
}

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
    pub fallback_surviving: Vec<usize>,
    pub fallback_active: bool,
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
    pub promotion_source: Option<PredictivePromotionSource>,
    pub recovery_mode_used: Option<RecoveryMode>,
    pub fallback_active: bool,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestStats {
    pub canonical: PredictiveMetrics,
    /// Compatibility alias for `canonical.scheduled_games`.
    pub games: usize,
    /// Compatibility alias for `canonical.conditional_mean_guesses`.
    pub average_guesses: f64,
    pub p95_guesses: usize,
    pub max_guesses: usize,
    pub failures: usize,
    pub coverage_gaps: usize,
    pub average_guesses_ci95: (f64, f64),
    pub failure_rate_ci95: (f64, f64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolvePolicyEvidence {
    pub summary: BacktestStats,
    pub elapsed_ms: u64,
    pub latency_p95_ms: f64,
    pub peak_memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalSolveComparison {
    pub baseline: SolvePolicyEvidence,
    pub survival: SolvePolicyEvidence,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub config_id: String,
    pub mode: WeightMode,
    pub variant: ModelVariant,
    pub backtest: BacktestStats,
    pub average_log_loss: f64,
    pub average_brier: f64,
    pub average_target_probability: f64,
    pub average_target_rank: f64,
    pub prior_evidence: Option<PriorEvidenceMetrics>,
    pub execution: ExecutionTelemetry,
    pub failure_penalty_sensitivity: Vec<FailurePenaltyEvidence>,
    pub latency_p95_ms: f64,
    pub session_fallback_cold_ms: f64,
    pub session_fallback_warm_ms: f64,
    pub proxy_step_pct: f64,
    pub lookahead_step_pct: f64,
    pub escalated_exact_step_pct: f64,
    pub exact_step_pct: f64,
    pub average_lookahead_pool_ratio: f64,
    pub average_exact_pool_ratio: f64,
    pub games: Vec<ExperimentGameResult>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    pub total_steps: usize,
    pub proxy_steps: usize,
    pub lookahead_steps: usize,
    pub escalated_exact_steps: usize,
    pub exact_steps: usize,
    pub danger_escalated_steps: usize,
    pub strict_recovery_steps: usize,
    pub uniform_recovery_steps: usize,
    pub epsilon_repair_steps: usize,
    pub dormant_fallback_steps: usize,
    pub exact_date_opener_artifact_hits: usize,
    pub recent_opener_artifact_hits: usize,
    pub reply_book_hits: usize,
    pub session_fallback_hits: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FailurePenaltyEvidence {
    pub penalty_guesses: f64,
    pub all_game_mean_guesses: f64,
    pub ci95: crate::experiments::MetricInterval,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentGameResult {
    pub target: String,
    pub outcome: GameOutcome,
    pub path: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceBaseline {
    pub id: String,
    pub description: String,
    pub artifacts: String,
    pub effective_config_toml: String,
    pub config_fingerprint: String,
    pub paired_vs_selected_default: Option<PairedDifference>,
    pub result: ExperimentResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoricalDiagnosticBaseline {
    pub date_range: String,
    pub scheduled_games: usize,
    pub modeled_games: usize,
    pub coverage_gaps: usize,
    pub conditional_mean_guesses: f64,
    pub average_log_loss: f64,
    pub average_brier_score: f64,
    pub interpretation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictiveEvidenceArtifact {
    pub schema_version: u32,
    pub identity_format: String,
    pub input_fingerprint: String,
    pub config_fingerprint: String,
    pub scope: String,
    pub sealed_test_evaluated: bool,
    pub evaluation_from: NaiveDate,
    pub evaluation_to: NaiveDate,
    pub history_snapshot_start: NaiveDate,
    pub history_snapshot_end: NaiveDate,
    pub code_revision: Option<String>,
    pub code_dirty: Option<bool>,
    pub platform: String,
    pub cpu: Option<String>,
    pub release_command: String,
    pub config_toml: String,
    #[serde(default)]
    pub resource_budget: EvidenceResourceBudget,
    #[serde(default)]
    pub resources: EvidenceResourceTelemetry,
    pub historical_diagnostic: HistoricalDiagnosticBaseline,
    pub baselines: Vec<EvidenceBaseline>,
    pub limitations: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EvidenceResourceBudget {
    pub maximum_seconds: u64,
    pub maximum_memory_mb: u64,
}

impl Default for EvidenceResourceBudget {
    fn default() -> Self {
        Self {
            maximum_seconds: 3_600,
            maximum_memory_mb: 4_096,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvidenceResourceTelemetry {
    pub generation_compute_ms: u64,
    pub current_working_set_bytes: Option<u64>,
    pub peak_working_set_bytes: Option<u64>,
    pub artifact_sizes: Vec<EvidenceArtifactSize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceArtifactSize {
    pub name: String,
    pub path: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollingFoldEvidence {
    pub fold_index: usize,
    pub validation: DateRange,
    pub metrics: PredictiveMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollingConfigEvidence {
    pub label: String,
    pub config_toml: String,
    pub config_fingerprint: String,
    pub folds: Vec<RollingFoldEvidence>,
    pub aggregate: PredictiveMetrics,
    pub prior_evidence: Option<PriorEvidenceMetrics>,
    pub execution: ExecutionTelemetry,
    pub failure_penalty_sensitivity: Vec<FailurePenaltyEvidence>,
    pub games: Vec<ExperimentGameResult>,
    pub latency_p95_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollingComparisonArtifact {
    pub schema_version: u32,
    pub identity_format: String,
    pub input_fingerprint: String,
    pub evaluation_plan: EvaluationPlan,
    pub sealed_test_evaluated: bool,
    pub code_revision: Option<String>,
    pub code_dirty: Option<bool>,
    pub baseline: RollingConfigEvidence,
    pub candidate: RollingConfigEvidence,
    pub candidate_minus_baseline: PairedDifference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenPredictiveCandidate {
    pub schema_version: u32,
    pub identity_format: String,
    pub input_fingerprint: String,
    pub freeze_fingerprint: String,
    pub evaluation_plan: EvaluationPlan,
    pub config_toml: String,
    pub config_fingerprint: String,
    pub candidate_label: String,
    pub development_comparison_fingerprint: String,
    pub development_metrics: PredictiveMetrics,
    pub development_paired_difference: PairedDifference,
    pub evaluation_artifact_policy: String,
    pub sealed_test_evaluated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SealedTestReport {
    pub schema_version: u32,
    pub identity_format: String,
    pub freeze_fingerprint: String,
    pub input_fingerprint: String,
    pub config_fingerprint: String,
    pub evaluation_plan: EvaluationPlan,
    pub evaluation_artifact_policy: String,
    pub sealed_test_evaluated: bool,
    pub evaluated_once: bool,
    pub metrics: PredictiveMetrics,
    pub prior_evidence: Option<PriorEvidenceMetrics>,
    pub execution: ExecutionTelemetry,
    pub games: Vec<ExperimentGameResult>,
    pub latency_p95_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SealedTestMarker {
    schema_version: u32,
    freeze_fingerprint: String,
    output_path: String,
    status: String,
}

impl PredictiveEvidenceArtifact {
    pub fn validate_identity(&self) -> Result<()> {
        if self.schema_version != 4
            || self.identity_format != crate::identity::IDENTITY_FORMAT
            || !crate::identity::is_tagged_digest(&self.input_fingerprint)
            || !crate::identity::is_tagged_digest(&self.config_fingerprint)
        {
            bail!("benchmark evidence uses an unsupported or mixed identity format; regenerate it");
        }
        let expected = crate::identity::digest_bytes_tagged(
            "maybe-wordle-benchmark-root-config-v1",
            self.config_toml.as_bytes(),
        );
        if self.config_fingerprint != expected {
            bail!("benchmark evidence config fingerprint mismatch; regenerate it");
        }
        for baseline in &self.baselines {
            let expected = crate::identity::digest_bytes_tagged(
                "maybe-wordle-benchmark-config-v1",
                baseline.effective_config_toml.as_bytes(),
            );
            if baseline.config_fingerprint != expected {
                bail!(
                    "benchmark baseline {} config fingerprint mismatch; regenerate it",
                    baseline.id
                );
            }
        }
        Ok(())
    }
}

impl RollingComparisonArtifact {
    pub fn validate_identity(&self) -> Result<()> {
        if self.schema_version != 3
            || self.identity_format != crate::identity::IDENTITY_FORMAT
            || !crate::identity::is_tagged_digest(&self.input_fingerprint)
        {
            bail!("rolling evidence uses an unsupported or mixed identity format; regenerate it");
        }
        for config in [&self.baseline, &self.candidate] {
            let expected = crate::identity::digest_bytes_tagged(
                "maybe-wordle-rolling-config-v1",
                config.config_toml.as_bytes(),
            );
            if config.config_fingerprint != expected {
                bail!(
                    "rolling config {} fingerprint mismatch; regenerate the evidence",
                    config.label
                );
            }
        }
        Ok(())
    }
}

impl FrozenPredictiveCandidate {
    pub fn validate_identity(&self) -> Result<()> {
        if self.schema_version != 1
            || self.identity_format != crate::identity::IDENTITY_FORMAT
            || !crate::identity::is_tagged_digest(&self.input_fingerprint)
            || !crate::identity::is_tagged_digest(&self.freeze_fingerprint)
            || !crate::identity::is_tagged_digest(&self.config_fingerprint)
            || !crate::identity::is_tagged_digest(&self.development_comparison_fingerprint)
            || self.sealed_test_evaluated
            || self.evaluation_artifact_policy != "artifact_free"
        {
            bail!("frozen candidate identity or policy is invalid; freeze a new candidate");
        }
        let config_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-rolling-config-v1",
            self.config_toml.as_bytes(),
        );
        if self.config_fingerprint != config_fingerprint {
            bail!("frozen candidate config fingerprint mismatch");
        }
        let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-frozen-candidate-v1");
        hash.field(self.input_fingerprint.as_bytes())
            .field(self.config_fingerprint.as_bytes())
            .field(self.development_comparison_fingerprint.as_bytes())
            .field(self.candidate_label.as_bytes())
            .field(self.evaluation_artifact_policy.as_bytes())
            .field(
                &serde_json::to_vec(&self.evaluation_plan)
                    .context("serialize frozen evaluation plan identity")?,
            )
            .field(
                &serde_json::to_vec(&self.development_metrics)
                    .context("serialize frozen development metrics identity")?,
            )
            .field(
                &serde_json::to_vec(&self.development_paired_difference)
                    .context("serialize frozen paired difference identity")?,
            );
        if self.freeze_fingerprint != hash.finish_tagged() {
            bail!("frozen candidate fingerprint mismatch");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RollingEvaluationCheckpoint {
    schema_version: u32,
    source_identity: String,
    evaluation_plan: EvaluationPlan,
    label: String,
    config_toml: String,
    folds: Vec<RollingFoldEvidence>,
    games: Vec<ExperimentGameResult>,
    prior_observations: Vec<RankedProbabilityObservation>,
    execution: ExecutionTelemetry,
}

#[derive(Clone, Debug)]
pub struct DetailedBacktestReport {
    pub summary: BacktestStats,
    pub runs: Vec<DetailedSolveRun>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRegretChoice {
    pub word: String,
    pub exact_cost: f64,
    pub regret: f64,
    pub matches_optimum: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRegretObservation {
    pub guess: String,
    pub feedback: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRegretState {
    pub date: NaiveDate,
    pub target: String,
    pub turn: usize,
    pub surviving_answers: usize,
    pub observations: Vec<SearchRegretObservation>,
    pub production_regime: String,
    pub optimal_word: String,
    pub optimal_exact_cost: f64,
    pub production: SearchRegretChoice,
    pub proxy: SearchRegretChoice,
    pub lookahead: SearchRegretChoice,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRegretSummary {
    pub states: usize,
    pub exact_matches: usize,
    pub positive_regret_states: usize,
    pub mean_regret: f64,
    pub maximum_regret: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchRegretRequest {
    pub from: NaiveDate,
    pub to: NaiveDate,
    pub minimum_survivors: usize,
    pub maximum_survivors: usize,
    pub maximum_states: usize,
    pub maximum_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnedProxyDatasetRequest {
    pub minimum_survivors: usize,
    pub maximum_survivors: usize,
    pub maximum_states_per_split: usize,
    pub guesses_per_state: usize,
    pub maximum_seconds: u64,
    pub maximum_memory_mb: u64,
    #[serde(default)]
    pub checkpoint_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRegretReport {
    pub schema_version: u32,
    pub identity_format: String,
    pub input_fingerprint: String,
    pub config_fingerprint: String,
    pub code_revision: Option<String>,
    pub code_dirty: Option<bool>,
    pub evaluation_from: NaiveDate,
    pub evaluation_to: NaiveDate,
    pub state_path_policy: String,
    pub minimum_survivors: usize,
    pub maximum_survivors: usize,
    pub maximum_states: usize,
    pub maximum_seconds: u64,
    pub historical_games: usize,
    pub scanned_games: usize,
    pub available_states: usize,
    pub sampled_states: usize,
    pub generation_elapsed_ms: u64,
    pub production: SearchRegretSummary,
    pub proxy: SearchRegretSummary,
    pub lookahead: SearchRegretSummary,
    pub states: Vec<SearchRegretState>,
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
    pub all_game_penalized_mean_guesses: f64,
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
    pub evaluation_plan: EvaluationPlan,
    pub search_window_start: NaiveDate,
    pub search_window_end: NaiveDate,
    pub validation_window_start: NaiveDate,
    pub validation_window_end: NaiveDate,
    pub test_window_start: NaiveDate,
    pub test_window_end: NaiveDate,
    pub current: TuningEvaluation,
    pub best: TuningEvaluation,
    pub replacement_toml: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct LiveConfigEvaluation {
    pub config: PriorConfig,
    pub predictive_metrics: PredictiveMetrics,
    pub average_guesses: f64,
    pub all_game_penalized_mean_guesses: f64,
    pub failures: usize,
    pub coverage_gaps: usize,
    pub latency_p95_ms: f64,
    pub hard_case_average_guesses: f64,
    pub hard_case_failures: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct StudyRunSummary {
    pub state_path: PathBuf,
    pub requested_parallelism: usize,
    pub effective_parallelism: usize,
    pub compute_threads: usize,
    pub completed_trials: usize,
    pub pending_trials: usize,
    pub running_trials: usize,
    pub pruned_trials: usize,
    pub rejected_trials: usize,
    pub failed_trials: usize,
    pub best_trial_number: Option<usize>,
    pub best_measurement: Option<StudyMeasurement>,
    pub best_config: Option<PriorConfig>,
    pub sealed_test_evaluated: bool,
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
    manifest_version: u32,
    model_manifest_hash: String,
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
pub struct Solver {
    pub config: PriorConfig,
    pub mode: WeightMode,
    pub variant: ModelVariant,
    pub guesses: Vec<String>,
    pub answers: Vec<AnswerRecord>,
    pub primary_answer_count: usize,
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

#[derive(Clone, Copy, Debug)]
struct SolveExecutionPolicy {
    book_usage: PredictiveBookUsage,
    search_mode: Option<PredictiveSearchMode>,
}

#[derive(Clone, Copy, Debug)]
struct PredictiveSuggestionFilters {
    mode: PredictiveSuggestionMode,
    hard_mode: bool,
    force_in_two_only: bool,
    forced_search_mode: Option<PredictiveSearchMode>,
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
    largest_weights: [f64; PATTERN_SPACE],
    counts: [usize; PATTERN_SPACE],
    weighted_log_sums: [f64; PATTERN_SPACE],
    touched_patterns: Vec<u8>,
}

type PredictiveMemoMap<K, V> = FxHashMap<K, V>;

impl GuessMetricScratch {
    fn new() -> Self {
        Self {
            masses: [0.0; PATTERN_SPACE],
            largest_weights: [0.0; PATTERN_SPACE],
            counts: [0; PATTERN_SPACE],
            weighted_log_sums: [0.0; PATTERN_SPACE],
            touched_patterns: Vec::with_capacity(PATTERN_SPACE),
        }
    }

    fn reset(&mut self) {
        for pattern in self.touched_patterns.drain(..) {
            self.masses[pattern as usize] = 0.0;
            self.largest_weights[pattern as usize] = 0.0;
            self.counts[pattern as usize] = 0;
            self.weighted_log_sums[pattern as usize] = 0.0;
        }
    }
}

impl Solver {
    fn answer_pattern(&self, guess_index: usize, answer_index: usize) -> u8 {
        if answer_index < self.primary_answer_count {
            self.pattern_table.get(guess_index, answer_index)
        } else {
            score_guess(&self.guesses[guess_index], &self.answers[answer_index].word)
        }
    }

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
        crate::experiments::validate_predictive_config(config)
            .context("invalid predictive solver config")?;
        let model = if variant == ModelVariant::SeedPlusHistory {
            load_model(paths, config)?
        } else {
            load_model_with_variant(paths, config, variant)?
        };
        let pattern_table = PatternTable::load_or_build(
            paths,
            &model.guesses,
            &model.answers[..model.primary_answer_count],
        )?;
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
            primary_answer_count: model.primary_answer_count,
            history_dates: model.history,
            exact_small_state_table: SmallStateTable::build(
                config
                    .exact_exhaustive_threshold
                    .max(config.proxy_small_state_lower_bound_threshold)
                    .max(2),
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
    top_probability: f64,
    top_prediction_correct: bool,
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
        predictive::{PredictiveSuggestRequest, PredictiveSuggestionMode, RecoveryMode},
        scoring::{ALL_GREEN_PATTERN, format_feedback_letters, score_guess},
        small_state::SmallStateTable,
    };

    use super::{
        AbsurdleSuggestion, ExactSearchScratch, ExactSubsetKey, ExactSubsetStorage,
        ExactSuggestionMode, ForcedOpenerEvaluation, GuessMetrics, PredictiveBookUsage,
        PredictiveMemoMap, PredictiveOpenerArtifact, PredictiveReplyBookArtifact,
        PredictiveReplyEntry, PredictiveSearchMode, PredictiveThirdReplyEntry, Solver,
        StateDangerAssessment, Suggestion, compare_absurdle_suggestions, compare_exact_costs,
        compare_final_turn, compare_forced_openers, compare_guess_metrics,
        compare_guess_metrics_for_state, compare_lookahead, compare_suggestions,
        compare_suggestions_for_state, count_masked_letters, exact_suggestion_mode,
        hard_mode_violation, known_absent_letter_mask, predictive_search_mode,
        should_replace_forced_opener, should_use_final_turn_objective,
        should_use_second_guess_coverage,
    };

    use super::books::write_predictive_artifact;

    fn test_solver(words: &[&str]) -> Solver {
        test_solver_with_answer_count(words, words.len())
    }

    fn test_solver_with_answer_count(words: &[&str], answer_count: usize) -> Solver {
        assert!(answer_count > 0 && answer_count <= words.len());
        let guesses = words
            .iter()
            .map(|word| (*word).to_string())
            .collect::<Vec<_>>();
        let answers = words[..answer_count]
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
            primary_answer_count: answer_count,
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
    fn predictive_suggestions_exclude_non_progressing_guesses() {
        let mut solver =
            test_solver_with_answer_count(&["cigar", "rebut", "sissy", "humph", "zzzzz"], 4);
        solver.config.search_policy_mode = crate::config::SearchPolicyMode::ProxyOnly;
        let state = solver.initial_state(NaiveDate::from_ymd_opt(2026, 1, 1).expect("date"));
        let suggestions = solver
            .suggestions(&state, solver.guesses.len())
            .expect("suggestions");

        assert!(
            suggestions
                .iter()
                .all(|suggestion| suggestion.word != "zzzzz")
        );
        assert!(suggestions.iter().all(|suggestion| {
            let guess_index = solver.guess_index[&suggestion.word];
            state
                .surviving
                .iter()
                .filter(|answer_index| {
                    solver.answer_pattern(guess_index, **answer_index) != ALL_GREEN_PATTERN
                })
                .fold(HashMap::<u8, usize>::new(), |mut counts, answer_index| {
                    *counts
                        .entry(solver.answer_pattern(guess_index, *answer_index))
                        .or_default() += 1;
                    counts
                })
                .values()
                .copied()
                .max()
                .unwrap_or(0)
                < state.surviving.len()
        }));
    }

    #[test]
    fn pooled_exact_suggestions_expose_the_cost_used_for_ranking() {
        let mut solver = test_solver(&[
            "cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade",
        ]);
        solver.config.exact_threshold = 8;
        solver.config.exact_exhaustive_threshold = 2;
        solver.config.exact_candidate_pool = 4;
        let state = solver.initial_state(NaiveDate::from_ymd_opt(2026, 1, 1).expect("date"));
        let suggestions = solver.suggestions(&state, 5).expect("suggestions");

        assert!(
            suggestions
                .first()
                .expect("top suggestion")
                .exact_cost
                .is_some()
        );
    }

    #[test]
    fn proxy_preview_returns_immediately_rankable_results_without_exact_refinement() {
        let mut solver = test_solver(&[
            "cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade",
        ]);
        solver.config.exact_threshold = 8;
        solver.config.exact_exhaustive_threshold = 2;
        solver.config.exact_candidate_pool = 4;
        let observations = Vec::new();
        let request = PredictiveSuggestRequest {
            as_of: NaiveDate::from_ymd_opt(2026, 1, 1).expect("date"),
            observations: &observations,
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        };
        let preview = solver
            .suggest_predictive_proxy_preview(request)
            .expect("proxy preview");

        assert_eq!(preview.state.surviving, solver.answers.len());
        assert!(!preview.suggestions.is_empty());
        assert!(preview.suggestions.iter().all(
            |suggestion| suggestion.exact_cost.is_none() && suggestion.lookahead_cost.is_none()
        ));
        assert!(preview.candidates.len() == solver.answers.len());
    }

    fn slow_exact_best_cost(
        solver: &Solver,
        subset: &[usize],
        weights: &[f64],
        memo: &mut HashMap<Vec<usize>, f64>,
    ) -> f64 {
        if subset.is_empty() {
            return 0.0;
        }
        if subset.len() == 1 {
            return 1.0;
        }
        if let Some(value) = memo.get(subset) {
            return *value;
        }
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut best = f64::INFINITY;
        for guess_index in 0..solver.guesses.len() {
            let mut children = HashMap::<u8, Vec<usize>>::new();
            for answer_index in subset {
                children
                    .entry(solver.answer_pattern(guess_index, *answer_index))
                    .or_default()
                    .push(*answer_index);
            }
            if children
                .iter()
                .any(|(pattern, child)| *pattern != ALL_GREEN_PATTERN && child == subset)
            {
                continue;
            }
            let mut cost = 1.0;
            for (pattern, child) in children {
                if pattern == ALL_GREEN_PATTERN {
                    continue;
                }
                let mass = child.iter().map(|index| weights[*index]).sum::<f64>();
                if mass == 0.0 {
                    continue;
                }
                cost += (mass / total_weight) * slow_exact_best_cost(solver, &child, weights, memo);
            }
            best = best.min(cost);
        }
        assert!(
            best.is_finite(),
            "slow exact reference found no progressing guess"
        );
        memo.insert(subset.to_vec(), best);
        best
    }

    fn slow_heuristic_reply_cost(solver: &Solver, subset: &[usize], weights: &[f64]) -> f64 {
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        assert!(total_weight.is_finite() && total_weight > 0.0);
        let pattern_space_log = (super::PATTERN_SPACE as f64).log2();
        let mut best = f64::INFINITY;

        for guess_index in 0..solver.guesses.len() {
            let mut counts = [0usize; super::PATTERN_SPACE];
            let mut masses = [0.0_f64; super::PATTERN_SPACE];
            let mut largest_weights = [0.0_f64; super::PATTERN_SPACE];
            let mut weighted_log_sums = [0.0_f64; super::PATTERN_SPACE];
            for answer_index in subset {
                let pattern = solver.answer_pattern(guess_index, *answer_index) as usize;
                let weight = weights[*answer_index];
                counts[pattern] += 1;
                masses[pattern] += weight;
                largest_weights[pattern] = largest_weights[pattern].max(weight);
                if weight > 0.0 {
                    weighted_log_sums[pattern] += weight * weight.log2();
                }
            }

            let mut proxy_cost = 1.0;
            let mut worst_non_green_bucket_size = 0usize;
            let mut largest_non_green_bucket_mass = 0.0_f64;
            let mut large_non_green_bucket_count = 0usize;
            let mut dangerous_mass_bucket_count = 0usize;
            let mut non_green_mass_in_large_buckets = 0.0_f64;
            for pattern in 0..super::PATTERN_SPACE {
                if counts[pattern] == 0 {
                    continue;
                }
                let mass = masses[pattern];
                let probability = mass / total_weight;
                let child_proxy = if pattern as u8 == ALL_GREEN_PATTERN {
                    0.0
                } else if counts[pattern] == 1 {
                    1.0
                } else if counts[pattern] <= solver.config.proxy_small_state_lower_bound_threshold {
                    if mass <= 0.0 {
                        0.0
                    } else {
                        1.0 + ((mass - largest_weights[pattern]) / mass)
                    }
                } else {
                    let expected_remaining_floor =
                        (counts[pattern] as f64 / super::PATTERN_SPACE as f64).max(1.0);
                    let entropy_bits = if mass > 0.0 {
                        mass.log2() - (weighted_log_sums[pattern] / mass)
                    } else {
                        0.0
                    };
                    expected_remaining_floor.max((entropy_bits / pattern_space_log).max(1.0))
                };
                proxy_cost += probability * child_proxy;

                if pattern as u8 != ALL_GREEN_PATTERN {
                    worst_non_green_bucket_size = worst_non_green_bucket_size.max(counts[pattern]);
                    largest_non_green_bucket_mass = largest_non_green_bucket_mass.max(probability);
                    if counts[pattern] >= solver.config.trap_size_threshold {
                        large_non_green_bucket_count += 1;
                        non_green_mass_in_large_buckets += probability;
                    }
                    if probability >= solver.config.trap_mass_threshold {
                        dangerous_mass_bucket_count += 1;
                    }
                }
            }

            let bucket_ratio = worst_non_green_bucket_size as f64 / subset.len().max(1) as f64;
            if worst_non_green_bucket_size == subset.len() {
                continue;
            }
            let penalty = (solver.config.lookahead_trap_penalty * largest_non_green_bucket_mass)
                + (solver.config.lookahead_large_bucket_penalty
                    * large_non_green_bucket_count as f64)
                + (solver.config.lookahead_dangerous_mass_penalty
                    * dangerous_mass_bucket_count as f64)
                + (solver.config.lookahead_large_bucket_mass_penalty
                    * non_green_mass_in_large_buckets)
                + (solver.config.lookahead_worst_bucket_ratio_penalty * bucket_ratio);
            best = best.min(proxy_cost + penalty);
        }
        best
    }

    fn slow_lookahead_child_cost(solver: &Solver, subset: &[usize], weights: &[f64]) -> f64 {
        if subset.is_empty() {
            return 0.0;
        }
        if subset.len() <= solver.config.exact_exhaustive_threshold {
            return slow_exact_best_cost(solver, subset, weights, &mut HashMap::new());
        }
        slow_heuristic_reply_cost(solver, subset, weights)
    }

    fn slow_lookahead_root_cost(
        solver: &Solver,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
    ) -> f64 {
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        assert!(total_weight.is_finite() && total_weight > 0.0);
        let mut children = vec![Vec::<usize>::new(); super::PATTERN_SPACE];
        let mut masses = [0.0_f64; super::PATTERN_SPACE];
        for answer_index in subset {
            let pattern = solver.answer_pattern(guess_index, *answer_index) as usize;
            children[pattern].push(*answer_index);
            masses[pattern] += weights[*answer_index];
        }

        let mut total_cost = 1.0;
        let mut worst_child_probability = 0.0_f64;
        let mut large_bucket_count = 0usize;
        let mut dangerous_mass_bucket_count = 0usize;
        let mut non_green_mass_in_large_buckets = 0.0_f64;
        for pattern in 0..super::PATTERN_SPACE {
            if children[pattern].is_empty() {
                continue;
            }
            let probability = masses[pattern] / total_weight;
            if masses[pattern] > 0.0 && pattern as u8 != ALL_GREEN_PATTERN {
                let child_value = if children[pattern].as_slice() == subset {
                    f64::INFINITY
                } else {
                    slow_lookahead_child_cost(solver, &children[pattern], weights)
                };
                total_cost += probability * child_value;
            }
            if pattern as u8 != ALL_GREEN_PATTERN {
                worst_child_probability = worst_child_probability.max(probability);
                if children[pattern].len() >= solver.config.trap_size_threshold {
                    large_bucket_count += 1;
                    non_green_mass_in_large_buckets += probability;
                }
                if probability >= solver.config.trap_mass_threshold {
                    dangerous_mass_bucket_count += 1;
                }
            }
        }
        total_cost
            + (solver.config.lookahead_trap_penalty * worst_child_probability)
            + (solver.config.lookahead_large_bucket_penalty * large_bucket_count as f64)
            + (solver.config.lookahead_dangerous_mass_penalty * dangerous_mass_bucket_count as f64)
            + (solver.config.lookahead_large_bucket_mass_penalty * non_green_mass_in_large_buckets)
    }

    fn next_test_u64(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
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
            primary_answer_count: guesses.len(),
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
    fn mixed_support_feedback_can_recover_a_zero_mass_candidate() {
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 10).expect("date");
        let mut solver = test_solver(&["cigar", "rebut"]);
        solver.mode = WeightMode::Weighted;
        solver.config.cooldown_floor = 0.0;
        solver.config.cooldown_days = 365;
        solver.answers[0].history_dates = vec![as_of];

        let initial = solver.initial_state(as_of);
        assert_eq!(initial.surviving.len(), 2);
        assert!(initial.modeled_weights[0] == 0.0);
        assert!(initial.modeled_weights[1] > 0.0);

        let mut repaired = initial.clone();
        solver
            .apply_feedback(&mut repaired, "cigar", score_guess("cigar", "cigar"))
            .expect("epsilon repair");
        assert_eq!(repaired.surviving, vec![0]);
        assert_eq!(
            repaired.recovery_mode_used,
            Some(RecoveryMode::EpsilonRepair)
        );
        assert!(repaired.total_weight > 0.0);

        solver.config.recovery.mode = RecoveryMode::UniformOverSupport;
        let mut uniform = initial.clone();
        solver
            .apply_feedback(&mut uniform, "cigar", score_guess("cigar", "cigar"))
            .expect("uniform repair");
        assert_eq!(
            uniform.recovery_mode_used,
            Some(RecoveryMode::UniformOverSupport)
        );

        solver.config.recovery.mode = RecoveryMode::Strict;
        let mut strict = initial;
        assert!(
            solver
                .apply_feedback(&mut strict, "cigar", score_guess("cigar", "cigar"))
                .is_err()
        );
    }

    #[test]
    fn inconsistent_primary_feedback_activates_dormant_fallback_support() {
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 10).expect("date");
        let mut solver = test_solver(&["cigar", "rebut"]);
        solver.primary_answer_count = 1;
        let mut state = solver.initial_state(as_of);
        assert_eq!(state.surviving, vec![0]);
        assert_eq!(state.fallback_surviving, vec![1]);

        solver
            .apply_feedback(&mut state, "cigar", score_guess("cigar", "rebut"))
            .expect("activate fallback");
        assert_eq!(state.surviving, vec![1]);
        assert!(state.fallback_surviving.is_empty());
        assert_eq!(state.recovery_mode_used, Some(RecoveryMode::EpsilonRepair));
        assert!(state.total_weight > 0.0);
    }

    #[test]
    fn future_history_only_primary_answer_is_dormant_support_before_first_seen() {
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 10).expect("date");
        let mut solver = test_solver(&["cigar", "rebut"]);
        solver.answers[1].in_seed = false;
        solver.answers[1].history_dates =
            vec![NaiveDate::from_ymd_opt(2026, 3, 11).expect("future date")];

        let state = solver.initial_state(as_of);
        assert_eq!(state.surviving, vec![0]);
        assert_eq!(state.fallback_surviving, vec![1]);
        assert_eq!(state.modeled_weights[1], 0.0);
    }

    #[test]
    fn predictive_manifest_changes_for_same_count_inputs_and_history_mutations() {
        let as_of = NaiveDate::from_ymd_opt(2026, 3, 10).expect("date");
        let base = test_solver(&["cigar", "rebut", "sissy"]);
        let base_identity = base.predictive_book_identity(as_of);

        let mut changed_guess = base.clone();
        changed_guess.guesses[0] = "humph".to_string();
        assert_ne!(
            base_identity.model_manifest_hash,
            changed_guess
                .predictive_book_identity(as_of)
                .model_manifest_hash
        );

        let mut changed_answer = base.clone();
        changed_answer.answers[0].word = "humph".to_string();
        assert_ne!(
            base_identity.model_manifest_hash,
            changed_answer
                .predictive_book_identity(as_of)
                .model_manifest_hash
        );

        let mut changed_history = base;
        changed_history.history_dates.push(NytDailyEntry {
            id: Some(99),
            solution: "cigar".to_string(),
            print_date: as_of,
            days_since_launch: Some(99),
            editor: Some("fixture".to_string()),
        });
        assert_ne!(
            base_identity.model_manifest_hash,
            changed_history
                .predictive_book_identity(as_of)
                .model_manifest_hash
        );
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
            primary_answer_count: guesses.len(),
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
            primary_answer_count: guesses.len(),
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
            primary_answer_count: guesses.len(),
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
    fn cached_predictive_choice_reads_recent_reply_and_third_turn_book() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let artifact_date = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
        let request_date = NaiveDate::from_ymd_opt(2026, 3, 16).expect("valid");
        let opener_pattern = score_guess("cigar", "humph");
        let reply_pattern = score_guess("rebut", "humph");
        let artifact = PredictiveReplyBookArtifact {
            identity: solver.predictive_book_identity(artifact_date),
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
        write_predictive_artifact(&solver.reply_book_artifact_path(artifact_date), &artifact)
            .expect("write recent reply book");

        let reply = solver.cached_predictive_choice(
            request_date,
            &[("cigar".to_string(), opener_pattern)],
            false,
        );
        assert_eq!(reply.map(|choice| choice.word), Some("rebut".to_string()));
        let third = solver.cached_predictive_choice(
            request_date,
            &[
                ("cigar".to_string(), opener_pattern),
                ("rebut".to_string(), reply_pattern),
            ],
            false,
        );
        assert_eq!(third.map(|choice| choice.word), Some("sissy".to_string()));
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
    fn final_turn_prefers_immediate_solve_probability_over_future_information() {
        let informative = Suggestion {
            word: "alpha".into(),
            entropy: 4.0,
            solve_probability: 0.0,
            expected_remaining: 1.1,
            force_in_two: true,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.1,
            large_non_green_bucket_count: 0,
            dangerous_mass_bucket_count: 0,
            non_green_mass_in_large_buckets: 0.0,
            proxy_cost: Some(1.0),
            large_state_score: Some(4.0),
            posterior_answer_probability: 0.0,
            lookahead_cost: Some(1.0),
            exact_cost: Some(1.0),
        };
        let likely_answer = Suggestion {
            word: "bravo".into(),
            entropy: 0.1,
            solve_probability: 0.6,
            posterior_answer_probability: 0.6,
            force_in_two: false,
            proxy_cost: Some(9.0),
            large_state_score: Some(-9.0),
            lookahead_cost: Some(9.0),
            exact_cost: Some(9.0),
            ..informative.clone()
        };

        assert_eq!(
            compare_final_turn(&likely_answer, &informative),
            std::cmp::Ordering::Less
        );
        assert!(!should_use_final_turn_objective(4));
        assert!(should_use_final_turn_objective(5));
        assert!(should_use_final_turn_objective(6));
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
            fallback_surviving: Vec::new(),
            fallback_active: false,
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
    fn registered_pool_scalars_control_expansion_and_source_quotas() {
        assert_eq!(super::scaled_pool_size(10, 2.5), 25);
        assert_eq!(super::scaled_pool_size(10, 1.5), 15);
        assert_eq!(super::fractional_pool_take(96, 0.5), 48);
        assert_eq!(super::fractional_pool_take(96, 0.25), 24);
        assert_eq!(super::fractional_pool_take(96, 1.0 / 6.0), 16);
        assert_eq!(super::fractional_pool_take(96, 0.125), 12);
    }

    #[test]
    fn pooled_exact_candidates_include_force_and_worst_bucket_guesses() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let state = super::SolveState {
            surviving: vec![0, 1],
            fallback_surviving: Vec::new(),
            fallback_active: false,
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
    fn weighted_exact_pruning_matches_exhaustive_root_cost_for_skewed_mass() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy"]);
        solver.config.exact_threshold = 3;
        solver.config.exact_exhaustive_threshold = 3;
        let subset = vec![0, 1, 2];
        let weights = vec![0.40, 0.59, 0.01];

        let mut root_costs = Vec::new();
        for guess_index in 0..solver.guesses.len() {
            root_costs.push(
                solver
                    .exact_cost_for_guess(
                        guess_index,
                        super::ExactCostContext {
                            subset: &subset,
                            weights: &weights,
                            small_state_table: &solver.exact_small_state_table,
                            memo: &mut PredictiveMemoMap::default(),
                            best_bound: f64::INFINITY,
                            scratch: &mut ExactSearchScratch::new(),
                            depth: 0,
                        },
                    )
                    .expect("root cost"),
            );
        }
        let exhaustive = root_costs.iter().copied().fold(f64::INFINITY, f64::min);
        let old_uniform_count_bound = solver.exact_small_state_table.lower_bound(subset.len());
        assert!(root_costs[0] <= old_uniform_count_bound);
        assert!(root_costs[0] > exhaustive);

        let admissible = super::weighted_exact_lower_bound(&subset, &weights).expect("bound");
        assert!(admissible <= exhaustive + 1e-12);
        let exact = solver
            .exact_best_cost(
                &subset,
                &weights,
                &solver.exact_small_state_table,
                &mut PredictiveMemoMap::default(),
                &mut ExactSearchScratch::new(),
                0,
            )
            .expect("exact value");
        assert!((exact - exhaustive).abs() <= 1e-12);
    }

    #[test]
    fn weighted_search_mass_validation_covers_numeric_boundaries() {
        assert_eq!(
            super::weighted_exact_lower_bound(&[], &[]).expect("empty bound"),
            0.0
        );
        assert_eq!(
            super::weighted_exact_lower_bound(&[0, 1], &[2.0, 0.0]).expect("one positive mass"),
            1.0
        );
        for (weights, expected_message) in [
            (vec![0.0, 0.0], "zero-mass"),
            (vec![1.0, -0.1], "finite and non-negative"),
            (vec![1.0, f64::NAN], "finite and non-negative"),
            (vec![1.0, f64::INFINITY], "finite and non-negative"),
        ] {
            let error = super::weighted_exact_lower_bound(&[0, 1], &weights)
                .expect_err("invalid mass must fail");
            assert!(
                error.to_string().contains(expected_message),
                "weights={weights:?} error={error}"
            );
        }
        let error = super::weighted_exact_lower_bound(&[0, 2], &[1.0, 1.0])
            .expect_err("out-of-range answer index must fail");
        assert!(error.to_string().contains("out of range"));
    }

    #[test]
    fn exact_pruning_matches_independent_slow_reference_on_tractable_states() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        solver.config.exact_threshold = solver.answers.len();
        solver.config.exact_exhaustive_threshold = solver.answers.len();

        for seed in 0..13usize {
            let weights = (0..solver.answers.len())
                .map(|index| {
                    if (index + seed) % 5 == 0 {
                        0.0
                    } else {
                        (((index + 1) * (seed + 3)) % 17 + 1) as f64
                    }
                })
                .collect::<Vec<_>>();
            for mask in 1usize..(1usize << solver.answers.len()) {
                let subset = (0..solver.answers.len())
                    .filter(|index| mask & (1usize << index) != 0)
                    .collect::<Vec<_>>();
                if !(2..=5).contains(&subset.len())
                    || subset.iter().all(|index| weights[*index] == 0.0)
                {
                    continue;
                }
                let expected =
                    slow_exact_best_cost(&solver, &subset, &weights, &mut HashMap::new());
                let lower_bound =
                    super::weighted_exact_lower_bound(&subset, &weights).expect("lower bound");
                assert!(
                    lower_bound <= expected + 1e-12,
                    "inadmissible bound: seed={seed} subset={subset:?} bound={lower_bound} expected={expected}"
                );
                let actual = solver
                    .exact_best_cost(
                        &subset,
                        &weights,
                        &solver.exact_small_state_table,
                        &mut PredictiveMemoMap::default(),
                        &mut ExactSearchScratch::new(),
                        0,
                    )
                    .expect("optimized exact cost");
                assert!(
                    (actual - expected).abs() <= 1e-12,
                    "seed={seed} subset={subset:?} expected={expected} actual={actual}"
                );
            }
        }
    }

    #[test]
    fn heuristic_lookahead_ranking_matches_randomized_slow_reference() {
        let mut solver = test_solver(&[
            "cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade",
        ]);
        solver.config.exact_threshold = 2;
        solver.config.exact_exhaustive_threshold = 2;
        solver.config.lookahead_reply_pool = solver.guesses.len();
        solver.config.medium_state_lookahead_reply_pool = solver.guesses.len();
        solver.config.danger_reply_pool_bonus = 0;

        let mut random_state = 0x6a09_e667_f3bc_c909_u64;
        for case_index in 0..24 {
            let mut weights = Vec::with_capacity(solver.answers.len());
            for _ in 0..solver.answers.len() {
                let sample = next_test_u64(&mut random_state);
                weights.push(if sample.is_multiple_of(7) {
                    0.0
                } else {
                    ((sample >> 16) % 97 + 1) as f64
                });
            }
            let mut subset = (0..solver.answers.len())
                .filter(|_| next_test_u64(&mut random_state) & 3 != 0)
                .collect::<Vec<_>>();
            if subset.len() < 4 {
                subset = vec![0, 1, 2, 3, 4];
            }
            if subset.iter().all(|index| weights[*index] == 0.0) {
                weights[subset[0]] = 1.0;
            }

            let mut expected_ranking = (0..solver.guesses.len())
                .map(|guess_index| {
                    (
                        guess_index,
                        slow_lookahead_root_cost(&solver, guess_index, &subset, &weights),
                    )
                })
                .collect::<Vec<_>>();
            expected_ranking.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| solver.guesses[left.0].cmp(&solver.guesses[right.0]))
            });

            let mut exact_memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut lookahead_memo = PredictiveMemoMap::default();
            let mut actual_ranking = (0..solver.guesses.len())
                .map(|guess_index| {
                    let cost = solver
                        .lookahead_cost_for_guess(
                            guess_index,
                            super::LookaheadCostContext {
                                subset: &subset,
                                weights: &weights,
                                expanded: false,
                                exact_memo: &mut exact_memo,
                                exact_scratch: &mut exact_scratch,
                                lookahead_memo: &mut lookahead_memo,
                            },
                        )
                        .expect("optimized lookahead cost");
                    (guess_index, cost)
                })
                .collect::<Vec<_>>();
            actual_ranking.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| solver.guesses[left.0].cmp(&solver.guesses[right.0]))
            });

            for ((actual_index, actual_cost), (expected_index, expected_cost)) in
                actual_ranking.iter().zip(&expected_ranking)
            {
                assert_eq!(
                    actual_index, expected_index,
                    "case={case_index} subset={subset:?} weights={weights:?}"
                );
                assert!(
                    (actual_cost - expected_cost).abs() <= 1e-12
                        || (actual_cost.is_infinite() && expected_cost.is_infinite()),
                    "case={case_index} guess={} expected={expected_cost} actual={actual_cost}",
                    solver.guesses[*actual_index]
                );
            }
        }
    }

    #[test]
    fn predictive_recursion_skips_zero_mass_branches_without_nan() {
        let corpus = ["cigar", "rebut", "sissy", "humph", "awake", "blush"];
        let mut witness = None;
        'outer: for guess in corpus {
            for left_index in 0..corpus.len() {
                for right_index in (left_index + 1)..corpus.len() {
                    if corpus[left_index] != guess
                        && corpus[right_index] != guess
                        && score_guess(guess, corpus[left_index])
                            == score_guess(guess, corpus[right_index])
                    {
                        witness = Some((guess, corpus[left_index], corpus[right_index]));
                        break 'outer;
                    }
                }
            }
        }
        let (guess, zero_left, zero_right) = witness.expect("zero-mass branch witness");
        let mut solver = test_solver(&[guess, zero_left, zero_right]);
        solver.config.exact_threshold = 3;
        solver.config.exact_exhaustive_threshold = 3;
        let subset = vec![0, 1, 2];
        let weights = vec![1.0, 0.0, 0.0];

        let exact = solver
            .exact_best_cost(
                &subset,
                &weights,
                &solver.exact_small_state_table,
                &mut PredictiveMemoMap::default(),
                &mut ExactSearchScratch::new(),
                0,
            )
            .expect("zero-mass branches do not affect expected exact cost");
        assert_eq!(exact, 1.0);

        let lookahead = solver
            .lookahead_cost_for_guess(
                0,
                super::LookaheadCostContext {
                    subset: &subset,
                    weights: &weights,
                    expanded: false,
                    exact_memo: &mut PredictiveMemoMap::default(),
                    exact_scratch: &mut ExactSearchScratch::new(),
                    lookahead_memo: &mut PredictiveMemoMap::default(),
                },
            )
            .expect("zero-mass branches do not fail lookahead");
        assert!(lookahead.is_finite());
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
    fn registered_danger_feature_definitions_change_the_declared_score() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let subset = vec![0, 1, 2, 3];
        let weights = vec![0.4, 0.3, 0.2, 0.1];
        let mut metrics = solver.score_guess_metrics_for_subset(
            &subset,
            &weights,
            &solver.exact_small_state_table,
        );
        metrics[0].force_in_two = false;
        metrics[1].force_in_two = false;
        metrics[0].largest_non_green_bucket_mass = 0.20;
        metrics[1].largest_non_green_bucket_mass = 0.29;
        metrics[0].worst_non_green_bucket_size = 2;
        metrics[1].worst_non_green_bucket_size = 4;
        metrics[0].high_mass_ambiguous_bucket_count = 2;

        solver.config.danger_top_concentration_w = 1.0;
        solver.config.danger_bucket_mass_w = 0.0;
        solver.config.danger_bucket_ratio_w = 0.0;
        solver.config.danger_ambiguous_w = 0.0;
        solver.config.danger_disagreement_w = 0.0;
        solver.config.danger_posterior_window = 1;
        let top_one = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        solver.config.danger_posterior_window = 3;
        let top_three = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        assert!((top_one - 0.4).abs() <= 1e-12);
        assert!((top_three - 0.9).abs() <= 1e-12);

        solver.config.danger_top_concentration_w = 0.0;
        solver.config.danger_disagreement_w = 1.0;
        solver.config.danger_candidate_window = 1;
        let no_comparison = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        solver.config.danger_candidate_window = 2;
        solver.config.danger_mass_disagreement_threshold = 0.10;
        solver.config.danger_size_disagreement_threshold = 3;
        let below_cutoffs = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        solver.config.danger_mass_disagreement_threshold = 0.08;
        let mass_disagreement = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        solver.config.danger_mass_disagreement_threshold = 0.10;
        solver.config.danger_size_disagreement_threshold = 2;
        let size_disagreement = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        assert_eq!(no_comparison, 0.0);
        assert_eq!(below_cutoffs, 0.0);
        assert_eq!(mass_disagreement, 1.0);
        assert_eq!(size_disagreement, 1.0);

        solver.config.danger_disagreement_w = 0.0;
        solver.config.danger_ambiguous_w = 1.0;
        solver.config.danger_ambiguity_saturation_count = 4;
        let half_saturated = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        solver.config.danger_ambiguity_saturation_count = 2;
        let saturated = solver
            .assess_subset_danger(&subset, &weights, 1.0, &metrics)
            .danger_score;
        assert!((half_saturated - 0.5).abs() <= 1e-12);
        assert_eq!(saturated, 1.0);
    }

    #[test]
    fn declared_search_policy_modes_override_staged_escalation() {
        let dangerous = StateDangerAssessment {
            danger_score: 0.90,
            dangerous_lookahead: true,
            dangerous_exact: true,
        };
        let mut config = PriorConfig {
            search_policy_mode: crate::config::SearchPolicyMode::ProxyOnly,
            ..PriorConfig::default()
        };
        assert!(matches!(
            predictive_search_mode(&config, 8, dangerous),
            PredictiveSearchMode::ProxyOnly
        ));
        config.search_policy_mode = crate::config::SearchPolicyMode::ProxyWithExactEndgame;
        assert!(matches!(
            predictive_search_mode(&config, config.exact_threshold + 8, dangerous),
            PredictiveSearchMode::ProxyOnly
        ));
        assert!(matches!(
            predictive_search_mode(&config, config.exact_exhaustive_threshold, dangerous),
            PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive)
        ));
    }

    #[test]
    fn second_guess_coverage_is_independent_of_exact_search_threshold() {
        let mut config = PriorConfig {
            exact_threshold: 64,
            second_guess_coverage_min_survivors: 65,
            second_guess_coverage_max_survivors: 80,
            ..PriorConfig::default()
        };
        assert!(!should_use_second_guess_coverage(&config, 12, 1));
        assert!(should_use_second_guess_coverage(&config, 72, 1));
        assert!(!should_use_second_guess_coverage(&config, 81, 1));
        assert!(!should_use_second_guess_coverage(&config, 72, 2));
        config.exact_threshold = 0;
        assert!(should_use_second_guess_coverage(&config, 72, 1));
    }

    #[test]
    fn second_guess_coverage_pool_is_an_exact_cap() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        solver.config.second_guess_coverage_pool = 2;
        let subset = (0..solver.answers.len()).collect::<Vec<_>>();
        let weights = vec![1.0; solver.answers.len()];
        let metrics = solver.score_guess_metrics_for_subset(
            &subset,
            &weights,
            &solver.exact_small_state_table,
        );
        assert!(metrics.len() > solver.config.second_guess_coverage_pool);
        let coverage = solver
            .medium_second_guess_coverage(&subset, &weights, &metrics)
            .expect("coverage");
        assert_eq!(coverage.len(), solver.config.second_guess_coverage_pool);
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
                posterior_answer_probability: 0.0,
            },
        );
        let mut masses = HashMap::<u8, f64>::new();
        for answer_index in &subset {
            let pattern = solver.answer_pattern(0, *answer_index);
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
        assert_eq!(super::normalized_concentration_penalty(0.0, 0.0, 0), 0.0);
        assert_eq!(super::normalized_concentration_penalty(1.0, 1.0, 1), 0.0);
        let smooth = super::normalized_concentration_penalty(1.0, 0.25 * 0.25 * 4.0, 4);
        let spiky =
            super::normalized_concentration_penalty(1.0, 0.70 * 0.70 + 0.10 * 0.10 * 3.0, 4);
        assert_eq!(smooth, 0.0);
        assert!(spiky > smooth);
        assert!(spiky <= 1.0);
    }

    #[test]
    fn weighted_proxy_child_floor_respects_skewed_mass() {
        let skewed = super::weighted_proxy_child_floor(1.0, 0.99);
        let uniform = super::weighted_proxy_child_floor(1.0, 0.5);
        assert!((skewed - 1.01).abs() <= 1e-12);
        assert!((uniform - 1.5).abs() <= 1e-12);
        assert!(skewed < uniform);
    }

    #[test]
    fn rolling_checkpoint_namespace_changes_with_source_identity() {
        let first = super::rolling_checkpoint_fingerprint("same-config", "source-a");
        let second = super::rolling_checkpoint_fingerprint("same-config", "source-b");
        let repeat = super::rolling_checkpoint_fingerprint("same-config", "source-a");
        assert_ne!(first, second);
        assert_eq!(first, repeat);
    }

    #[test]
    fn zero_mass_buckets_do_not_change_probability_concentration() {
        let solver = test_solver(&["cigar", "rebut", "sissy"]);
        assert_ne!(score_guess("cigar", "rebut"), score_guess("cigar", "sissy"));
        let weights = vec![1.0, 1.0, 0.0];
        let positive_only = solver.score_guess_metrics_for_subset(
            &[0, 1],
            &weights,
            &solver.exact_small_state_table,
        );
        let with_zero_mass_bucket = solver.score_guess_metrics_for_subset(
            &[0, 1, 2],
            &weights,
            &solver.exact_small_state_table,
        );
        assert_eq!(
            positive_only[0].smoothness_penalty,
            with_zero_mass_bucket[0].smoothness_penalty
        );
    }

    #[test]
    fn orthogonal_lookahead_penalties_grow_with_multiple_risk_signals() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let mild = solver.aggregate_lookahead_trap_penalty(0.20, 1, 1, 0.20);
        let severe = solver.aggregate_lookahead_trap_penalty(0.20, 3, 2, 0.45);
        let expected_mild = (solver.config.lookahead_trap_penalty * 0.20)
            + solver.config.lookahead_large_bucket_penalty
            + solver.config.lookahead_dangerous_mass_penalty
            + (solver.config.lookahead_large_bucket_mass_penalty * 0.20);
        assert!((mild - expected_mild).abs() <= 1e-12);
        assert!(severe > mild);
    }

    #[test]
    fn reply_bucket_ratio_penalty_is_independent_of_branch_mass() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let subset = (0..solver.answers.len()).collect::<Vec<_>>();
        let weights = vec![1.0; solver.answers.len()];
        let metric = solver
            .score_guess_metrics_for_subset(&subset, &weights, &solver.exact_small_state_table)
            .into_iter()
            .next()
            .expect("metric");
        solver.config.lookahead_trap_penalty = 0.0;
        solver.config.lookahead_large_bucket_penalty = 0.0;
        solver.config.lookahead_dangerous_mass_penalty = 0.0;
        solver.config.lookahead_large_bucket_mass_penalty = 0.0;
        solver.config.lookahead_worst_bucket_ratio_penalty = 0.75;

        let expected = 0.75 * metric.worst_non_green_bucket_size as f64 / subset.len() as f64;
        assert!((solver.lookahead_reply_penalty(&metric, subset.len()) - expected).abs() <= 1e-12);
    }

    #[test]
    fn ambiguous_mass_threshold_controls_proxy_feature() {
        let mut solver = test_solver(&["bakes", "cakes", "fakes", "lakes", "makes"]);
        let subset = (0..solver.answers.len()).collect::<Vec<_>>();
        let weights = vec![0.60, 0.10, 0.10, 0.10, 0.10];

        solver.config.ambiguous_mass_threshold = 0.10;
        let low_threshold = solver.score_guess_metrics_for_subset(
            &subset,
            &weights,
            &solver.exact_small_state_table,
        );
        solver.config.ambiguous_mass_threshold = 0.50;
        let high_threshold = solver.score_guess_metrics_for_subset(
            &subset,
            &weights,
            &solver.exact_small_state_table,
        );

        assert!(
            low_threshold[0].high_mass_ambiguous_bucket_count
                > high_threshold[0].high_mass_ambiguous_bucket_count
        );
    }

    #[test]
    fn danger_score_is_invariant_to_common_weight_scale() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake"]);
        let state =
            solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 1, 1).expect("date"));
        let metrics = solver.score_guess_metrics_for_subset(
            &state.surviving,
            &state.weights,
            &solver.exact_small_state_table,
        );
        let baseline = solver.assess_state_danger(&state, &metrics).danger_score;
        solver.config.danger_top_concentration_w *= 2.0;
        solver.config.danger_bucket_mass_w *= 2.0;
        solver.config.danger_bucket_ratio_w *= 2.0;
        solver.config.danger_ambiguous_w *= 2.0;
        solver.config.danger_disagreement_w *= 2.0;
        let scaled = solver.assess_state_danger(&state, &metrics).danger_score;
        assert!((baseline - scaled).abs() <= 1e-12);
    }

    #[test]
    fn heuristic_lookahead_child_does_not_double_count_reply_guess() {
        let mut solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake"]);
        solver.config.exact_exhaustive_threshold = 0;
        solver.config.lookahead_reply_pool = solver.guesses.len();
        solver.config.medium_state_lookahead_reply_pool = solver.guesses.len();
        solver.config.danger_reply_pool_bonus = 0;
        solver.config.lookahead_trap_penalty = 0.0;
        solver.config.lookahead_worst_bucket_ratio_penalty = 0.0;
        solver.config.lookahead_large_bucket_penalty = 0.0;
        solver.config.lookahead_dangerous_mass_penalty = 0.0;
        solver.config.lookahead_large_bucket_mass_penalty = 0.0;

        let subset = (0..solver.answers.len()).collect::<Vec<_>>();
        let weights = vec![1.0; solver.answers.len()];
        let expected = solver
            .score_guess_metrics_for_subset(&subset, &weights, &solver.exact_small_state_table)
            .into_iter()
            .filter(|metric| super::search::reply_guess_makes_progress(metric, subset.len()))
            .map(|metric| metric.proxy_cost)
            .fold(f64::INFINITY, f64::min);
        let mut exact_memo = PredictiveMemoMap::default();
        let mut exact_scratch = ExactSearchScratch::new();
        let mut lookahead_memo = PredictiveMemoMap::default();
        let actual = solver
            .lookahead_child_value(
                &subset,
                &weights,
                false,
                &mut exact_memo,
                &mut exact_scratch,
                &mut lookahead_memo,
            )
            .expect("heuristic child value");

        assert!((actual - expected).abs() <= 1e-12);
    }

    #[test]
    fn heuristic_lookahead_excludes_inert_child_reply() {
        let mut solver =
            test_solver_with_answer_count(&["cigar", "rebut", "sissy", "humph", "zzzzz"], 4);
        solver.config.exact_exhaustive_threshold = 0;
        solver.config.lookahead_reply_pool = solver.guesses.len();
        solver.config.medium_state_lookahead_reply_pool = solver.guesses.len();
        solver.config.danger_reply_pool_bonus = 0;
        solver.config.lookahead_trap_penalty = 0.0;
        solver.config.lookahead_worst_bucket_ratio_penalty = 0.0;
        solver.config.lookahead_large_bucket_penalty = 0.0;
        solver.config.lookahead_dangerous_mass_penalty = 0.0;
        solver.config.lookahead_large_bucket_mass_penalty = 0.0;

        let subset = (0..solver.answers.len()).collect::<Vec<_>>();
        let weights = vec![1.0; solver.answers.len()];
        let metrics = solver.score_guess_metrics_for_subset(
            &subset,
            &weights,
            &solver.exact_small_state_table,
        );
        let inert = metrics
            .iter()
            .find(|metric| solver.guesses[metric.guess_index] == "zzzzz")
            .expect("inert metric");
        assert!(!super::search::reply_guess_makes_progress(
            inert,
            subset.len()
        ));
        let expected = metrics
            .iter()
            .filter(|metric| super::search::reply_guess_makes_progress(metric, subset.len()))
            .map(|metric| metric.proxy_cost)
            .fold(f64::INFINITY, f64::min);
        let actual = solver
            .lookahead_child_value(
                &subset,
                &weights,
                false,
                &mut PredictiveMemoMap::default(),
                &mut ExactSearchScratch::new(),
                &mut PredictiveMemoMap::default(),
            )
            .expect("heuristic child value");
        assert!((actual - expected).abs() <= 1e-12);
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
    fn parallel_backtest_preserves_serial_game_order_and_results() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let games = [
            NytDailyEntry {
                id: Some(1),
                solution: "cigar".to_string(),
                print_date: NaiveDate::from_ymd_opt(2026, 1, 2).expect("date"),
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(2),
                solution: "rebut".to_string(),
                print_date: NaiveDate::from_ymd_opt(2026, 1, 3).expect("date"),
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(3),
                solution: "sissy".to_string(),
                print_date: NaiveDate::from_ymd_opt(2026, 1, 4).expect("date"),
                days_since_launch: None,
                editor: None,
            },
        ];
        let references = games.iter().collect::<Vec<_>>();
        let parallel = solver
            .backtest_selected_games(&references, 3, PredictiveBookUsage::None)
            .expect("parallel backtest");
        let progress = std::sync::Mutex::new(Vec::new());
        let report_with_progress = solver
            .backtest_selected_games_with_progress(
                &references,
                3,
                PredictiveBookUsage::None,
                Some(&|completed, total| {
                    progress
                        .lock()
                        .expect("progress lock")
                        .push((completed, total));
                }),
            )
            .expect("parallel backtest with progress");
        let serial = games
            .iter()
            .map(|entry| solver.solve_backtest_entry(entry, 3, PredictiveBookUsage::None))
            .collect::<Result<Vec<_>, _>>()
            .expect("serial games");

        assert_eq!(
            report_with_progress.summary.canonical,
            parallel.summary.canonical
        );
        let mut progress = progress.into_inner().expect("progress values");
        progress.sort_unstable();
        assert_eq!(progress, vec![(1, 3), (2, 3), (3, 3)]);
        assert_eq!(parallel.runs.len(), serial.len());
        for (parallel_run, (serial_outcome, serial_run)) in parallel.runs.iter().zip(serial.iter())
        {
            assert_eq!(parallel_run.target, serial_run.target);
            assert_eq!(parallel_run.date, serial_run.date);
            assert_eq!(parallel_run.solved, serial_run.solved);
            assert_eq!(
                parallel_run
                    .steps
                    .iter()
                    .map(|step| (&step.guess, step.feedback))
                    .collect::<Vec<_>>(),
                serial_run
                    .steps
                    .iter()
                    .map(|step| (&step.guess, step.feedback))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                serial_outcome.guesses,
                (!serial_run.steps.is_empty()).then_some(serial_run.steps.len())
            );
        }
    }

    #[test]
    fn hard_case_selection_includes_high_posterior_trap_when_available() {
        let solver = test_solver(&["cigar", "cigap", "cigam", "rebut", "sissy", "humph"]);
        let spec = crate::experiments::default_diagnostic_suite()
            .expect("diagnostic suite")
            .hard_cases;
        let cases = solver
            .select_hard_case_targets(Solver::today(), 3, &spec)
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
            primary_answer_count: guesses.len(),
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
