use super::search::check_predictive_search_cancelled;
mod formatting;
mod sealed;
mod study;
use super::*;
use std::{
    sync::OnceLock,
    time::{Duration, Instant},
};

const EVIDENCE_CHECKPOINT_SCHEMA_VERSION: u32 = 5;
const ROLLING_CHECKPOINT_SCHEMA_VERSION: u32 = 3;
pub(super) const BENCHMARK_EVIDENCE_SCHEMA_VERSION: u32 = 8;
const FINITE_SEARCH_REGRET_SCHEMA_VERSION: u32 = 1;
const FINITE_REGRET_VALUE_RESOLUTION: f64 = 64.0 * f64::EPSILON;
const SAME_STATE_DYNAMIC_MAX_REFERENCE_SUPPORT: usize = 6;
const STAGED_ZERO_FAILURE_CERTIFICATE_SCHEMA_VERSION: u32 = 1;
const PROSPECTIVE_FROZEN_CANDIDATE_SCHEMA_VERSION: u32 = 3;
const PROSPECTIVE_EVALUATION_SCHEMA_VERSION: u32 = 3;
const PROSPECTIVE_MARKER_SCHEMA_VERSION: u32 = 2;
const PROSPECTIVE_REGISTRY_SCHEMA_VERSION: u32 = 2;
const PROSPECTIVE_WINDOW_DAYS: u64 = 30;

const POSTERIOR_CALIBRATION_STRATA: [&str; 5] = [
    "all",
    "never_used",
    "reused",
    "historical_only",
    "out_of_core",
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct EvidenceTimingEvent {
    profile: Option<String>,
    phase: &'static str,
    elapsed_ms: u64,
}

trait EvidenceTimingSink {
    fn enabled(&self) -> bool;
    fn emit(&mut self, event: EvidenceTimingEvent);
}

struct EvaluationControl<'a> {
    progress: Option<&'a (dyn Fn(usize, usize) + Sync)>,
    cancelled: &'a (dyn Fn() -> bool + Sync),
}

struct StderrEvidenceTiming {
    enabled: bool,
}

impl StderrEvidenceTiming {
    fn from_env() -> Self {
        Self {
            enabled: evidence_timing_enabled(
                std::env::var("MAYBE_WORDLE_EVIDENCE_TIMING")
                    .ok()
                    .as_deref(),
            ),
        }
    }
}

impl EvidenceTimingSink for StderrEvidenceTiming {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn emit(&mut self, event: EvidenceTimingEvent) {
        eprintln!(
            "benchmark-evidence timing phase={} profile={} elapsed_ms={}",
            event.phase,
            event.profile.as_deref().unwrap_or("-"),
            event.elapsed_ms,
        );
        let _ = std::io::stderr().flush();
    }
}

#[derive(Default)]
struct NoopEvidenceTiming;

impl EvidenceTimingSink for NoopEvidenceTiming {
    fn enabled(&self) -> bool {
        false
    }

    fn emit(&mut self, _event: EvidenceTimingEvent) {}
}

fn evidence_timing_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true"))
}

fn record_evidence_timing(
    sink: &mut dyn EvidenceTimingSink,
    profile: Option<&str>,
    phase: &'static str,
    elapsed: Duration,
) {
    if sink.enabled() {
        sink.emit(EvidenceTimingEvent {
            profile: profile.map(str::to_owned),
            phase,
            elapsed_ms: elapsed.as_millis().min(u64::MAX as u128) as u64,
        });
    }
}

fn per_turn_search_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        evidence_timing_enabled(
            std::env::var("MAYBE_WORDLE_EVIDENCE_TIMING")
                .ok()
                .as_deref(),
        )
    })
}

fn format_search_timing(
    turn: usize,
    survivors: usize,
    regime: PredictiveRegime,
    elapsed_ms: u64,
) -> String {
    format!(
        "benchmark-evidence timing turn={turn} survivors={survivors} regime={} elapsed_ms={elapsed_ms}",
        regime.label()
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProspectiveWindowMarker {
    schema_version: u32,
    freeze_fingerprint: String,
    window_fingerprint: String,
    input_fingerprint: String,
    config_fingerprint: String,
    pre_window_history_fingerprint: String,
    window: DateRange,
    window_data_fingerprint: String,
    output_path: String,
    consumed_at_utc: DateTime<Utc>,
    status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProspectiveRegistryMarker {
    schema_version: u32,
    freeze_fingerprint: String,
    window_fingerprint: String,
    pre_window_history_fingerprint: String,
    window: DateRange,
    reserved_at_utc: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvidenceMatrixCheckpoint {
    schema_version: u32,
    identity: String,
    #[serde(default)]
    resource_budget: Option<EvidenceResourceBudget>,
    #[serde(default)]
    rayon_threads: Option<usize>,
    elapsed_ms: u64,
    peak_working_set_bytes: u64,
    baselines: Vec<EvidenceBaseline>,
}

impl EvidenceMatrixCheckpoint {
    fn validate(
        &self,
        identity: &str,
        profile_ids: &[String],
        resource_budget: EvidenceResourceBudget,
        rayon_threads: usize,
    ) -> Result<()> {
        if self.schema_version != EVIDENCE_CHECKPOINT_SCHEMA_VERSION {
            bail!("unsupported evidence checkpoint schema");
        }
        if self.resource_budget != Some(resource_budget)
            || self.rayon_threads != Some(rayon_threads)
        {
            bail!("evidence checkpoint uses a different resource budget or Rayon worker count");
        }
        if self.identity != identity {
            bail!(
                "evidence checkpoint belongs to different source, config, plan, or matrix inputs"
            );
        }
        let completed_ids = self
            .baselines
            .iter()
            .map(|baseline| baseline.id.clone())
            .collect::<Vec<_>>();
        if !is_profile_prefix(&completed_ids, profile_ids) {
            bail!("evidence checkpoint profiles are not a valid completed prefix");
        }
        for baseline in &self.baselines {
            validate_evidence_baseline(baseline)?;
        }
        Ok(())
    }
}

fn is_profile_prefix(completed: &[String], expected: &[String]) -> bool {
    completed.len() <= expected.len()
        && completed
            .iter()
            .zip(expected)
            .all(|(completed, expected)| completed == expected)
}

pub(super) fn validate_evidence_baseline(baseline: &EvidenceBaseline) -> Result<()> {
    let expected_config_fingerprint = crate::identity::digest_bytes_tagged(
        "maybe-wordle-benchmark-config-v1",
        baseline.effective_config_toml.as_bytes(),
    );
    if baseline.config_fingerprint != expected_config_fingerprint {
        bail!(
            "evidence baseline {} config fingerprint mismatch",
            baseline.id
        );
    }

    let mut outcomes = baseline
        .result
        .games
        .iter()
        .map(|game| game.outcome)
        .collect::<Vec<_>>();
    validate_unique_game_dates(&outcomes, "evidence baseline")?;
    validate_predictive_metrics(
        &baseline.result.backtest.canonical,
        &mut outcomes,
        "evidence baseline",
    )?;

    let canonical = &baseline.result.backtest.canonical;
    let summary = &baseline.result.backtest;
    let measured_prior_games = baseline
        .result
        .prior_evidence
        .as_ref()
        .map_or(0, |prior| prior.measured_games);
    let prior_means = [
        baseline.result.average_log_loss,
        baseline.result.average_brier,
        baseline.result.average_target_probability,
        baseline.result.average_target_rank,
    ];
    if baseline.result.prior_evidence.is_some() && measured_prior_games == 0
        || measured_prior_games > canonical.scheduled_games
        || prior_means.iter().any(|mean| {
            mean.is_some() != (measured_prior_games > 0)
                || mean.is_some_and(|value| !value.is_finite())
        })
        || baseline
            .result
            .average_log_loss
            .is_some_and(|value| value < 0.0)
        || baseline
            .result
            .average_brier
            .is_some_and(|value| value < 0.0)
        || baseline
            .result
            .average_target_probability
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        || baseline
            .result
            .average_target_rank
            .is_some_and(|value| value < 1.0)
    {
        bail!(
            "evidence baseline {} prior means do not match a finite measured population",
            baseline.id
        );
    }
    let expected_failures = canonical
        .unsolved_games
        .checked_add(canonical.coverage_gaps)
        .ok_or_else(|| anyhow!("evidence baseline {} failure count overflowed", baseline.id))?;
    if summary.games != canonical.scheduled_games
        || summary.p95_guesses != canonical.p95_guesses
        || summary.max_guesses != canonical.max_guesses
        || summary.failures != expected_failures
        || summary.coverage_gaps != canonical.coverage_gaps
        || summary.average_guesses.map(f64::to_bits)
            != canonical.conditional_mean_guesses.map(f64::to_bits)
        || summary
            .average_guesses_ci95
            .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
            != canonical
                .conditional_mean_guesses_ci95
                .map(|interval| (interval.lower.to_bits(), interval.upper.to_bits()))
        || summary.failure_rate_ci95.0.to_bits()
            != (1.0 - canonical.solve_rate_ci95.upper).to_bits()
        || summary.failure_rate_ci95.1.to_bits()
            != (1.0 - canonical.solve_rate_ci95.lower).to_bits()
    {
        bail!(
            "evidence baseline {} compatibility metrics do not match canonical metrics",
            baseline.id
        );
    }
    validate_posterior_calibration_evidence(
        &baseline.result.games,
        &baseline.result.posterior_calibration,
        true,
        &format!("evidence baseline {}", baseline.id),
    )?;
    Ok(())
}

fn posterior_calibration_stratum_matches(
    stratum_index: usize,
    prior_strata: Option<PriorStrata>,
) -> bool {
    match stratum_index {
        0 => true,
        1 => prior_strata.is_some_and(|strata| strata.never_used),
        2 => prior_strata.is_some_and(|strata| strata.reused),
        3 => prior_strata.is_some_and(|strata| strata.historical_only),
        4 => prior_strata.is_some_and(|strata| strata.out_of_core),
        _ => false,
    }
}

fn summarize_posterior_calibration(
    games: &[ExperimentGameResult],
) -> Vec<PosteriorCalibrationSummary> {
    let mut summaries = Vec::with_capacity(POSTERIOR_CALIBRATION_STRATA.len() * 6);
    for (stratum_index, stratum) in POSTERIOR_CALIBRATION_STRATA.iter().enumerate() {
        for turn in 1_u8..=6 {
            let mut total_states = 0usize;
            let mut scores = Vec::new();
            for game in games {
                if !posterior_calibration_stratum_matches(stratum_index, game.prior_strata) {
                    continue;
                }
                for observation in &game.posterior_calibration {
                    if observation.turn != turn {
                        continue;
                    }
                    total_states += 1;
                    if let Some(score) = observation.score {
                        scores.push(score);
                    }
                }
            }
            let scored_states = scores.len();
            let mean_score = if scored_states == 0 {
                None
            } else {
                let divisor = scored_states as f64;
                Some(crate::experiments::ProbabilityScore {
                    target_probability: scores
                        .iter()
                        .map(|score| score.target_probability)
                        .sum::<f64>()
                        / divisor,
                    log_loss: scores.iter().map(|score| score.log_loss).sum::<f64>() / divisor,
                    brier: scores.iter().map(|score| score.brier).sum::<f64>() / divisor,
                })
            };
            summaries.push(PosteriorCalibrationSummary {
                stratum: (*stratum).to_string(),
                turn,
                total_states,
                scored_states,
                mean_score,
            });
        }
    }
    summaries
}

fn validate_probability_score(
    score: crate::experiments::ProbabilityScore,
    context: &str,
) -> Result<()> {
    if !score.target_probability.is_finite()
        || !(0.0..=1.0).contains(&score.target_probability)
        || !score.log_loss.is_finite()
        || score.log_loss < 0.0
        || !score.brier.is_finite()
        || score.brier < 0.0
        || score.brier > 2.0 + 1e-9
    {
        bail!("{context} contains an invalid posterior probability score");
    }
    let residual = 1.0 - score.target_probability;
    let binary = score_multiclass_probabilities(&[score.target_probability, residual], 0)?;
    if score.log_loss != binary.log_loss
        || score.brier < residual.powi(2) - 1e-9
        || score.brier > binary.brier + 1e-9
    {
        bail!("{context} posterior score is inconsistent with its target probability");
    }
    Ok(())
}

fn validate_posterior_calibration_game(game: &ExperimentGameResult, context: &str) -> Result<()> {
    validate_game_path(game, context)?;
    if game.posterior_calibration.is_empty() {
        return Ok(());
    }
    if let Some(strata) = game.prior_strata {
        if strata.never_used == strata.reused {
            bail!(
                "{context} game {} must mark exactly one of never-used and reused",
                game.outcome.date
            );
        }
        if strata.historical_only && !strata.reused {
            bail!(
                "{context} game {} marks a non-reused target historical-only",
                game.outcome.date
            );
        }
    }
    let expected_states = game.path.len().max(1);
    if game.posterior_calibration.len() != expected_states {
        bail!(
            "{context} game {} posterior calibration has {} states; expected {} from its path",
            game.outcome.date,
            game.posterior_calibration.len(),
            expected_states
        );
    }
    for (index, observation) in game.posterior_calibration.iter().enumerate() {
        let expected_turn = u8::try_from(index + 1).expect("at most six calibration states");
        if observation.turn != expected_turn {
            bail!(
                "{context} game {} posterior calibration turns are not contiguous",
                game.outcome.date
            );
        }
        if game.path.is_empty() && observation.score.is_some() {
            bail!(
                "{context} coverage-gap game {} has a scored turn-1 posterior",
                game.outcome.date
            );
        }
        if game.prior_strata.is_none() && observation.score.is_some() {
            bail!(
                "{context} game {} has a scored posterior without target strata metadata",
                game.outcome.date
            );
        }
        if let Some(score) = observation.score {
            validate_probability_score(
                score,
                &format!(
                    "{context} game {} turn {}",
                    game.outcome.date, observation.turn
                ),
            )?;
        }
    }
    Ok(())
}

fn validate_game_path(game: &ExperimentGameResult, context: &str) -> Result<()> {
    if game.path.len() > 6 {
        bail!(
            "{context} game {} has more than six guesses",
            game.outcome.date
        );
    }
    if game.path.is_empty() {
        if game.outcome.status != crate::experiments::GameOutcomeStatus::CoverageGap
            || game.outcome.guesses.is_some()
        {
            bail!(
                "{context} game {} has an empty path but is not a coverage gap",
                game.outcome.date
            );
        }
    } else if game.outcome.guesses != Some(game.path.len()) {
        bail!(
            "{context} game {} outcome guess count does not match its path",
            game.outcome.date
        );
    }
    Ok(())
}

fn validate_posterior_calibration_evidence(
    games: &[ExperimentGameResult],
    summaries: &[PosteriorCalibrationSummary],
    require_calibration: bool,
    context: &str,
) -> Result<()> {
    for game in games {
        validate_posterior_calibration_game(game, context)?;
    }
    if summaries.is_empty() {
        if require_calibration {
            bail!("{context} is missing posterior calibration summaries");
        }
        if games
            .iter()
            .any(|game| !game.posterior_calibration.is_empty())
        {
            bail!("{context} contains observations without posterior summaries");
        }
        return Ok(());
    }
    if games
        .iter()
        .any(|game| game.posterior_calibration.is_empty())
    {
        bail!("{context} contains a game without posterior calibration observations");
    }
    let expected = summarize_posterior_calibration(games);
    if expected != summaries {
        bail!("{context} posterior calibration summaries do not match per-game observations");
    }
    Ok(())
}

fn validate_predictive_metrics(
    metrics: &PredictiveMetrics,
    outcomes: &mut [GameOutcome],
    context: &str,
) -> Result<()> {
    outcomes.sort_by_key(|outcome| outcome.date);
    let expected =
        summarize_predictive_outcomes(outcomes, metrics.failure_penalty_guesses, metrics.bootstrap)
            .with_context(|| format!("invalid {context} game outcomes"))?;
    if expected != *metrics {
        bail!("{context} metrics do not match their per-game outcomes");
    }
    Ok(())
}

fn validate_unique_game_dates(outcomes: &[GameOutcome], context: &str) -> Result<()> {
    let mut dates = HashSet::new();
    for outcome in outcomes {
        if !dates.insert(outcome.date) {
            bail!("{context} contains duplicate game date {}", outcome.date);
        }
    }
    Ok(())
}

fn validate_rolling_checkpoint(
    checkpoint: &RollingEvaluationCheckpoint,
    source_identity: &str,
    label: &str,
    config_toml: &str,
    evaluation_plan: &EvaluationPlan,
    require_finite_traces: bool,
) -> Result<()> {
    if checkpoint.schema_version != ROLLING_CHECKPOINT_SCHEMA_VERSION {
        bail!("unsupported rolling checkpoint schema");
    }
    if checkpoint.source_identity != source_identity
        || checkpoint.evaluation_plan != *evaluation_plan
        || checkpoint.label != label
        || checkpoint.config_toml != config_toml
    {
        bail!(
            "rolling checkpoint does not match the current source/config/plan; remove the rebuildable checkpoint and retry"
        );
    }
    let config: PriorConfig =
        toml::from_str(config_toml).context("parse rolling checkpoint config")?;
    for game in &checkpoint.games {
        validate_game_path(game, "rolling checkpoint")?;
        if require_finite_traces || !game.finite_search_steps.is_empty() {
            let expected_prefix = if config.search_policy_mode.is_finite() {
                Some(game.path.len())
            } else {
                None
            };
            if let Some(expected_prefix) = expected_prefix {
                validate_finite_game_trace(game, expected_prefix)?;
            }
        }
    }
    let mut seen_fold_ids = HashSet::new();
    for stored_fold in &checkpoint.folds {
        if !seen_fold_ids.insert(stored_fold.fold_index) {
            bail!(
                "rolling checkpoint contains duplicate fold id {}",
                stored_fold.fold_index
            );
        }
        let planned_fold = evaluation_plan
            .folds
            .iter()
            .find(|fold| fold.index == stored_fold.fold_index)
            .ok_or_else(|| {
                anyhow!(
                    "rolling checkpoint contains unknown fold id {}",
                    stored_fold.fold_index
                )
            })?;
        if stored_fold.validation != planned_fold.validation {
            bail!(
                "rolling checkpoint fold {} validation range does not match the evaluation plan",
                stored_fold.fold_index
            );
        }
    }
    if checkpoint.folds.len() > evaluation_plan.folds.len() {
        bail!("rolling checkpoint contains more folds than the evaluation plan");
    }

    validate_unique_game_dates(
        &checkpoint
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>(),
        "rolling checkpoint",
    )?;
    for game in &checkpoint.games {
        let matching_planned_folds = evaluation_plan
            .folds
            .iter()
            .filter(|fold| fold.validation.contains(game.outcome.date))
            .count();
        if matching_planned_folds != 1 {
            bail!(
                "rolling checkpoint game {} does not belong to exactly one planned validation range",
                game.outcome.date
            );
        }
        let matching_completed_folds = checkpoint
            .folds
            .iter()
            .filter(|fold| fold.validation.contains(game.outcome.date))
            .count();
        if matching_completed_folds != 1 {
            bail!(
                "rolling checkpoint game {} does not belong to exactly one completed fold",
                game.outcome.date
            );
        }
    }

    for stored_fold in &checkpoint.folds {
        let mut fold_outcomes = checkpoint
            .games
            .iter()
            .filter(|game| stored_fold.validation.contains(game.outcome.date))
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        if fold_outcomes.is_empty() {
            bail!(
                "rolling checkpoint fold {} has no games in its validation range",
                stored_fold.fold_index
            );
        }
        validate_unique_game_dates(&fold_outcomes, "rolling checkpoint fold")?;
        validate_predictive_metrics(
            &stored_fold.metrics,
            &mut fold_outcomes,
            &format!("rolling checkpoint fold {}", stored_fold.fold_index),
        )?;
    }
    Ok(())
}

fn validate_rolling_comparison_artifact(artifact: &RollingComparisonArtifact) -> Result<()> {
    artifact.validate_identity()?;
    if artifact.top == 0 {
        bail!("rolling comparison top must be positive");
    }
    if artifact.sealed_test_evaluated {
        bail!("rolling comparison must not evaluate the sealed test");
    }

    for (context, evidence) in [
        ("baseline", &artifact.baseline),
        ("candidate", &artifact.candidate),
    ] {
        let checkpoint = RollingEvaluationCheckpoint {
            schema_version: ROLLING_CHECKPOINT_SCHEMA_VERSION,
            source_identity: artifact.input_fingerprint.clone(),
            evaluation_plan: artifact.evaluation_plan.clone(),
            label: evidence.label.clone(),
            config_toml: evidence.config_toml.clone(),
            folds: evidence.folds.clone(),
            games: evidence.games.clone(),
            prior_observations: Vec::new(),
            execution: evidence.execution.clone(),
        };
        validate_rolling_checkpoint(
            &checkpoint,
            &artifact.input_fingerprint,
            &evidence.label,
            &evidence.config_toml,
            &artifact.evaluation_plan,
            false,
        )?;
        if evidence.folds.len() != artifact.evaluation_plan.folds.len() {
            bail!("rolling {context} evidence does not contain every planned validation fold");
        }

        for fold in &evidence.folds {
            if fold.metrics.failure_penalty_guesses.to_bits() != 7.0f64.to_bits()
                || fold.metrics.bootstrap != BootstrapConfig::default()
            {
                bail!(
                    "rolling {context} fold {} uses an unsupported metric penalty or bootstrap",
                    fold.fold_index
                );
            }
        }
        if evidence.aggregate.failure_penalty_guesses.to_bits() != 7.0f64.to_bits()
            || evidence.aggregate.bootstrap != BootstrapConfig::default()
        {
            bail!("rolling {context} aggregate uses an unsupported metric penalty or bootstrap");
        }

        let mut outcomes = evidence
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        validate_predictive_metrics(
            &evidence.aggregate,
            &mut outcomes,
            &format!("rolling {context} aggregate"),
        )?;
    }

    let mut baseline_outcomes = artifact
        .baseline
        .games
        .iter()
        .map(|game| game.outcome)
        .collect::<Vec<_>>();
    let mut candidate_outcomes = artifact
        .candidate
        .games
        .iter()
        .map(|game| game.outcome)
        .collect::<Vec<_>>();
    baseline_outcomes.sort_by_key(|outcome| outcome.date);
    candidate_outcomes.sort_by_key(|outcome| outcome.date);
    let expected = PairedDifference::all_game_penalized(
        &baseline_outcomes,
        &candidate_outcomes,
        7.0,
        BootstrapConfig::default(),
    )?;
    if expected != artifact.candidate_minus_baseline {
        bail!("rolling paired difference does not match per-game outcomes");
    }
    Ok(())
}

struct StudyEvaluationRequest<'a> {
    paths: &'a ProjectPaths,
    config: &'a PriorConfig,
    stage: StudyStage,
    artifact_namespace: &'a str,
    evaluation_plan: &'a EvaluationPlan,
    top: usize,
    target_validation_folds: usize,
    validation_fold_indices: &'a [usize],
    maximum_trial_seconds: u64,
    maximum_memory_mb: u64,
    measure_latency: bool,
    measurement: StudyMeasurement,
    prior_elapsed_ms: u64,
    cancellation_path: Option<&'a Path>,
}

#[derive(Clone, Debug)]
struct SearchRegretCandidateState {
    date: NaiveDate,
    target: String,
    turn: usize,
    observations: Vec<(String, u8)>,
    surviving_answers: usize,
}

#[cfg(test)]
thread_local! {
    // Deterministic deadline fault after a validated whole-state result.
    static REGRET_STOP_AFTER_STATES: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StagedZeroFailureRootCheck {
    certified: bool,
    reason: &'static str,
}

fn evenly_spaced_indices(total: usize, maximum: usize) -> Vec<usize> {
    let take = total.min(maximum);
    match take {
        0 => Vec::new(),
        1 => vec![total / 2],
        _ => (0..take)
            .map(|index| index * (total - 1) / (take - 1))
            .collect(),
    }
}

use crate::predictive::learned_proxy::learned_proxy_feature_names;

fn learned_proxy_features(metric: &GuessMetrics) -> Vec<f64> {
    vec![
        metric.entropy,
        metric.solve_probability,
        metric.expected_remaining,
        if metric.force_in_two { 1.0 } else { 0.0 },
        metric.worst_non_green_bucket_size as f64,
        metric.largest_non_green_bucket_mass,
        metric.high_mass_ambiguous_bucket_count as f64,
        metric.smoothness_penalty,
        metric.large_non_green_bucket_count as f64,
        metric.dangerous_mass_bucket_count as f64,
        metric.non_green_mass_in_large_buckets,
        metric.posterior_answer_probability,
    ]
}

fn learned_proxy_guess_pool(
    solver: &Solver,
    metrics: &[GuessMetrics],
    survivors: &[usize],
    count: usize,
) -> Vec<usize> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |metric: &GuessMetrics| {
        if seen.insert(metric.guess_index) {
            selected.push(metric.guess_index);
        }
    };
    let primary_take = (count / 2).max(1);
    let secondary_take = (count.saturating_sub(primary_take) / 3).max(1);
    let mut primary = metrics.iter().collect::<Vec<_>>();
    primary.sort_by(|left, right| {
        compare_guess_metrics_for_state(left, right, &solver.guesses, false)
    });
    for metric in primary.into_iter().take(primary_take) {
        push(metric);
    }
    let mut by_entropy = metrics.iter().collect::<Vec<_>>();
    by_entropy.sort_by(|left, right| {
        right
            .entropy
            .total_cmp(&left.entropy)
            .then_with(|| solver.guesses[left.guess_index].cmp(&solver.guesses[right.guess_index]))
    });
    for metric in by_entropy.into_iter().take(secondary_take) {
        push(metric);
    }
    let mut by_worst = metrics.iter().collect::<Vec<_>>();
    by_worst.sort_by(|left, right| {
        left.worst_non_green_bucket_size
            .cmp(&right.worst_non_green_bucket_size)
            .then_with(|| {
                left.largest_non_green_bucket_mass
                    .total_cmp(&right.largest_non_green_bucket_mass)
            })
    });
    for metric in by_worst.into_iter().take(secondary_take) {
        push(metric);
    }
    let mut by_solve = metrics.iter().collect::<Vec<_>>();
    by_solve.sort_by(|left, right| {
        right
            .solve_probability
            .total_cmp(&left.solve_probability)
            .then_with(|| solver.guesses[left.guess_index].cmp(&solver.guesses[right.guess_index]))
    });
    for metric in by_solve.into_iter().take(secondary_take) {
        push(metric);
    }
    for answer_index in survivors {
        if let Some(guess_index) = solver.guess_index.get(&solver.answers[*answer_index].word)
            && seen.insert(*guess_index)
        {
            selected.push(*guess_index);
        }
    }
    selected
}

fn summarize_search_regret(
    states: &[SearchRegretState],
    choice: impl Fn(&SearchRegretState) -> &SearchRegretChoice,
) -> SearchRegretSummary {
    let regrets = states
        .iter()
        .map(|state| choice(state).regret)
        .collect::<Vec<_>>();
    SearchRegretSummary {
        states: states.len(),
        exact_matches: states
            .iter()
            .filter(|state| choice(state).matches_optimum)
            .count(),
        positive_regret_states: regrets.iter().filter(|regret| **regret > 1e-9).count(),
        mean_regret: (!regrets.is_empty())
            .then(|| regrets.iter().sum::<f64>() / regrets.len() as f64),
        maximum_regret: regrets.into_iter().reduce(f64::max),
    }
}

fn finite_quality_name(quality: FiniteSearchQuality) -> &'static str {
    match quality {
        FiniteSearchQuality::Heuristic => "heuristic",
        FiniteSearchQuality::UpperBound => "upper_bound",
        FiniteSearchQuality::Exact => "exact",
    }
}

fn finite_reason_name(reason: FiniteSearchReason) -> &'static str {
    match reason {
        FiniteSearchReason::Complete => "complete",
        FiniteSearchReason::Deadline => "deadline",
        FiniteSearchReason::Cancelled => "cancelled",
        FiniteSearchReason::NodeBudget => "node_budget",
    }
}

fn finite_regret_value(candidate: FiniteSearchCandidate) -> FiniteSearchRegretValue {
    FiniteSearchRegretValue {
        failure_probability: candidate.failure_probability,
        expected_attempts: candidate.expected_attempts,
    }
}

fn finite_value_is_valid(value: FiniteSearchRegretValue) -> bool {
    value.failure_probability.is_finite()
        && (0.0..=1.0).contains(&value.failure_probability)
        && value.expected_attempts.is_finite()
        && value.expected_attempts >= 0.0
}

fn validate_finite_trace_step(
    guess: &str,
    expected_turn: u8,
    trace: &FiniteSearchStepEvidence,
) -> Result<()> {
    if trace.turn != expected_turn
        || trace.candidate_count < trace.top_candidates.len()
        || trace.top_candidates.is_empty()
        || trace.top_candidates.len() > 8
        || trace.top_candidates[0].word != guess
    {
        bail!("finite search trace does not match its game path");
    }
    Ok(())
}

fn validate_finite_game_trace(game: &ExperimentGameResult, expected_prefix: usize) -> Result<()> {
    if game.finite_search_steps.len() != expected_prefix {
        bail!("finite search trace count does not match its game path");
    }
    for (index, (guess, trace)) in game.path.iter().zip(&game.finite_search_steps).enumerate() {
        validate_finite_trace_step(guess, (index + 1) as u8, trace)?;
    }
    Ok(())
}

fn finite_step_evidence(
    guesses: &[String],
    turn: u8,
    search: &FiniteSearchResult,
) -> Result<FiniteSearchStepEvidence> {
    for candidate in &search.candidates {
        if candidate.guess_index >= guesses.len() {
            bail!("finite candidate guess index is out of range");
        }
        if candidate.quality != FiniteSearchQuality::Heuristic
            && !finite_value_is_valid(finite_regret_value(*candidate))
        {
            bail!("completed finite candidate has an invalid value");
        }
    }
    let top_candidates = search
        .candidates
        .iter()
        .take(8)
        .map(|candidate| {
            let completed = candidate.quality != FiniteSearchQuality::Heuristic;
            FiniteSearchCandidateEvidence {
                word: guesses[candidate.guess_index].clone(),
                quality: finite_quality_name(candidate.quality).to_string(),
                modeled_failure_probability: completed.then_some(candidate.failure_probability),
                expected_attempts_remaining: completed.then_some(candidate.expected_attempts),
            }
        })
        .collect();
    Ok(FiniteSearchStepEvidence {
        turn,
        reason: finite_reason_name(search.reason).to_string(),
        nodes_visited: search.nodes_visited,
        work_units: search.work_units,
        proposal_sampled: search.proposal_sampled,
        candidate_count: search.candidates.len(),
        top_candidates,
    })
}

fn validate_finite_run_trace(run: &DetailedSolveRun, expected_prefix: usize) -> Result<()> {
    for (index, step) in run.steps.iter().enumerate() {
        if index < expected_prefix {
            let trace = step
                .finite_search
                .as_ref()
                .ok_or_else(|| anyhow!("finite backtest step is missing its search trace"))?;
            validate_finite_trace_step(&step.guess, (index + 1) as u8, trace)?;
        } else if step.finite_search.is_some() {
            bail!("finite backtest trace appears after its expected prefix");
        }
    }
    Ok(())
}

fn finite_reference_from_result(
    result: &FiniteSearchResult,
    expected_legal_guesses: usize,
    guesses: &[String],
) -> FiniteSearchRegretReference {
    if result.reason != FiniteSearchReason::Complete {
        return FiniteSearchRegretReference {
            status: format!("unresolved_{}", finite_reason_name(result.reason)),
            word: None,
            value: None,
        };
    }
    if result.proposal_sampled {
        return FiniteSearchRegretReference {
            status: "incomplete_proposal_sample".to_string(),
            word: None,
            value: None,
        };
    }
    if result.candidates.len() != expected_legal_guesses || result.candidates.is_empty() {
        return FiniteSearchRegretReference {
            status: "incomplete_root_set".to_string(),
            word: None,
            value: None,
        };
    }
    let mut seen = HashSet::with_capacity(result.candidates.len());
    if result.candidates.iter().any(|candidate| {
        candidate.guess_index >= guesses.len()
            || !seen.insert(candidate.guess_index)
            || candidate.quality != FiniteSearchQuality::Exact
            || !finite_value_is_valid(finite_regret_value(*candidate))
    }) {
        return FiniteSearchRegretReference {
            status: "incomplete_candidate_set".to_string(),
            word: None,
            value: None,
        };
    }
    let candidate = result.candidates[0];
    FiniteSearchRegretReference {
        status: "exact".to_string(),
        word: guesses.get(candidate.guess_index).cloned(),
        value: Some(finite_regret_value(candidate)),
    }
}

fn finite_quantized_value(value: f64) -> f64 {
    (value / FINITE_REGRET_VALUE_RESOLUTION).round()
}

fn finite_quantized_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && finite_quantized_value(left).total_cmp(&finite_quantized_value(right))
            == std::cmp::Ordering::Equal
}

fn finite_regrets(
    fixed_root: &FiniteSearchRegretReference,
    global: &FiniteSearchRegretReference,
) -> Result<(Option<f64>, Option<f64>, Option<bool>)> {
    let (Some(fixed), Some(optimal)) = (fixed_root.value, global.value) else {
        return Ok((None, None, None));
    };
    if fixed_root.status != "exact" || global.status != "exact" {
        return Ok((None, None, None));
    }
    let fixed_failure = finite_quantized_value(fixed.failure_probability);
    let optimal_failure = finite_quantized_value(optimal.failure_probability);
    let fixed_is_better = fixed_failure < optimal_failure
        || (fixed_failure == optimal_failure
            && finite_quantized_value(fixed.expected_attempts)
                < finite_quantized_value(optimal.expected_attempts));
    if fixed_is_better {
        bail!("finite reference contradiction: fixed root beats purported global optimum");
    }
    let failure_regret = (fixed.failure_probability - optimal.failure_probability).max(0.0);
    let attempts_regret =
        finite_quantized_equal(fixed.failure_probability, optimal.failure_probability)
            .then(|| (fixed.expected_attempts - optimal.expected_attempts).max(0.0));
    let matches_optimum = Some(
        finite_quantized_equal(fixed.failure_probability, optimal.failure_probability)
            && finite_quantized_equal(fixed.expected_attempts, optimal.expected_attempts),
    );
    Ok((Some(failure_regret), attempts_regret, matches_optimum))
}

fn same_state_dynamic_regret_choice(
    runtime: FiniteSearchRegretRuntimeChoice,
    selected: &FiniteSearchRegretReference,
    global: &FiniteSearchRegretReference,
) -> Result<SameStateDynamicRegretChoice> {
    let (failure_regret, attempts_regret, matches_optimum) = finite_regrets(selected, global)?;
    let reference_status = if failure_regret.is_some() {
        "exact".to_string()
    } else if selected.status != "exact" {
        selected.status.clone()
    } else if global.status != "exact" {
        global.status.clone()
    } else {
        "unresolved_missing_reference_value".to_string()
    };
    Ok(SameStateDynamicRegretChoice {
        runtime: SameStateDynamicRegretRuntime {
            value: runtime.value,
            quality: runtime.quality,
            reason: runtime.reason,
        },
        reference_status,
        failure_regret,
        attempts_regret,
        matches_optimum,
    })
}

fn same_state_replay_guess(runtime: FiniteSearchRegretRuntimeChoice) -> Result<String> {
    let Some(word) = runtime.word else {
        if runtime.reason == "global_deadline" {
            bail!(
                "shared deadline elapsed before the staged artifact-free path reached the selected turn"
            );
        }
        bail!(
            "staged artifact-free path returned no suggestion before the selected turn (reason={})",
            runtime.reason
        );
    };
    Ok(word)
}

fn same_state_dynamic_identity(
    date: NaiveDate,
    turn: usize,
    observations: &[(String, u8)],
    state: &SolveState,
) -> String {
    let mut hash =
        crate::identity::CanonicalSha256::new("maybe-wordle-same-state-dynamic-regret-v1");
    hash.field(date.to_string().as_bytes())
        .field(&(turn as u64).to_le_bytes())
        .field(&[
            u8::from(state.condition_only),
            u8::from(state.fallback_active),
        ])
        .field(format!("{:?}", state.recovery_mode_used).as_bytes())
        .field(&state.modeled_total_weight.to_bits().to_le_bytes())
        .field(&state.total_weight.to_bits().to_le_bytes());
    for (guess, feedback) in observations {
        hash.field(guess.as_bytes()).field(&[*feedback]);
    }
    for survivors in [&state.surviving, &state.fallback_surviving] {
        hash.field(&(survivors.len() as u64).to_le_bytes());
        for answer_index in survivors {
            hash.field(&(*answer_index as u64).to_le_bytes());
        }
    }
    for weights in [
        &state.weights,
        &state.modeled_weights,
        &state.recovery_weights,
    ] {
        hash.field(&(weights.len() as u64).to_le_bytes());
        for weight in weights {
            hash.field(&weight.to_bits().to_le_bytes());
        }
    }
    hash.finish_tagged()
}

fn finite_mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn summarize_finite_search_regret(states: &[FiniteSearchRegretState]) -> FiniteSearchRegretSummary {
    let resolved_states = states
        .iter()
        .filter(|state| {
            state.exact_fixed_root.status == "exact" && state.global_optimum.status == "exact"
        })
        .count();
    let failure_regrets = states
        .iter()
        .filter_map(|state| state.failure_regret)
        .collect::<Vec<_>>();
    let attempts_regrets = states
        .iter()
        .filter_map(|state| state.attempts_regret)
        .collect::<Vec<_>>();
    FiniteSearchRegretSummary {
        states: states.len(),
        resolved_states,
        unresolved_states: states.len().saturating_sub(resolved_states),
        failure_regret_states: failure_regrets.len(),
        mean_failure_regret: finite_mean(&failure_regrets),
        maximum_failure_regret: failure_regrets.into_iter().reduce(f64::max),
        attempts_regret_states: attempts_regrets.len(),
        mean_attempts_regret: finite_mean(&attempts_regrets),
        maximum_attempts_regret: attempts_regrets.into_iter().reduce(f64::max),
    }
}

fn validate_exact_date_coverage(
    range: DateRange,
    dates: impl Iterator<Item = NaiveDate>,
) -> Result<()> {
    let mut dates = dates
        .filter(|date| range.contains(*date))
        .collect::<Vec<_>>();
    dates.sort_unstable();
    if dates.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("sealed-test history contains duplicate dates");
    }
    if dates.len() as u64 != range.days() {
        bail!("sealed-test history does not exactly cover the inclusive date range");
    }
    Ok(())
}

impl Solver {
    pub fn solve_target(&self, target: &str, date: NaiveDate, top: usize) -> Result<SolveRun> {
        Ok(self.solve_target_detailed(target, date, top)?.into())
    }

    pub fn solve_target_detailed(
        &self,
        target: &str,
        date: NaiveDate,
        top: usize,
    ) -> Result<DetailedSolveRun> {
        let as_of = crate::predictive::history_cutoff(date)?;
        self.solve_target_from_state_detailed(target, as_of, date, top, PredictiveBookUsage::Full)
    }

    pub(super) fn solve_target_from_state_detailed(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedSolveRun> {
        let state = self.initial_state(as_of);
        self.solve_target_from_initial_state_detailed(
            target,
            as_of,
            date,
            top,
            state,
            SolveExecutionPolicy {
                cancelled: &|| false,
                book_usage,
                search_mode: None,
                forced: &[],
            },
        )
    }

    fn solve_target_from_initial_state_detailed(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        top: usize,
        mut state: SolveState,
        policy: SolveExecutionPolicy<'_>,
    ) -> Result<DetailedSolveRun> {
        let target = target.to_ascii_lowercase();
        let mut observations = Vec::new();
        if policy.forced.len() > 6 {
            bail!("forced prefix exceeds the six-turn limit");
        }
        for (guess, pattern) in policy.forced {
            if !self.guess_index.contains_key(guess) || *pattern as usize >= PATTERN_SPACE {
                bail!("invalid forced guess or feedback: {guess}");
            }
        }

        if !state
            .surviving
            .iter()
            .chain(state.fallback_surviving.iter())
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
        let turn_timing_enabled = per_turn_search_timing_enabled();
        while steps.len() < 6 {
            let surviving_before = state.surviving.len();
            if let Some((guess, expected_feedback)) = policy.forced.get(steps.len()) {
                let feedback = score_guess(guess, &target);
                // Zero is the existing forced-opener API's unspecified-pattern sentinel.
                if *expected_feedback != 0 && *expected_feedback != feedback {
                    bail!(
                        "forced feedback mismatch for {guess}: expected {}, got {}",
                        format_feedback_letters(*expected_feedback),
                        format_feedback_letters(feedback)
                    );
                }
                let recovery_mode_used = state.recovery_mode_used;
                let fallback_active = state.fallback_active;
                if feedback != ALL_GREEN_PATTERN {
                    self.apply_feedback(&mut state, guess, feedback)?;
                }
                steps.push(DetailedSolveStep {
                    guess: guess.clone(),
                    feedback,
                    surviving_before,
                    surviving_after: if feedback == ALL_GREEN_PATTERN {
                        1
                    } else {
                        state.surviving.len()
                    },
                    chosen_force_in_two: false,
                    alternative_force_in_two: false,
                    danger_score: 0.0,
                    danger_escalated: false,
                    regime_used: PredictiveRegime::Proxy,
                    promotion_source: None,
                    recovery_mode_used,
                    fallback_active,
                    lookahead_pool_base: 0,
                    lookahead_pool_size: 0,
                    exact_pool_base: 0,
                    exact_pool_size: 0,
                    root_candidate_count: 0,
                    top_suggestions: Vec::new(),
                    finite_search: None,
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
                continue;
            }
            let search_started = turn_timing_enabled.then(Instant::now);
            check_predictive_search_cancelled(policy.cancelled)?;
            let batch = self.suggestion_batch_internal_with_search_mode_controlled(
                &state,
                top.max(1),
                Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &observations,
                }),
                policy.book_usage,
                policy.search_mode,
                policy.cancelled,
            )?;
            check_predictive_search_cancelled(policy.cancelled)?;
            if let Some(search_started) = search_started {
                let elapsed_ms = search_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                eprintln!(
                    "{}",
                    format_search_timing(
                        steps.len() + 1,
                        surviving_before,
                        batch.regime_used,
                        elapsed_ms,
                    )
                );
            }
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
            let finite_search = batch
                .finite_search
                .as_ref()
                .map(|search| finite_step_evidence(&self.guesses, (steps.len() + 1) as u8, search))
                .transpose()?;
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
                promotion_source: batch.promotion_source,
                recovery_mode_used: state.recovery_mode_used,
                fallback_active: state.fallback_active,
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
                finite_search,
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
        self.backtest_detailed_with_book_usage(from, to, top, PredictiveBookUsage::DiskOnly)
    }

    pub fn learned_proxy_dataset(
        &self,
        paths: &ProjectPaths,
        request: LearnedProxyDatasetRequest,
    ) -> Result<ExhaustiveCostDatasetArtifact> {
        if request.minimum_survivors < 2 || request.minimum_survivors > request.maximum_survivors {
            bail!("learned-proxy survivor range is invalid");
        }
        if request.maximum_states_per_split == 0 || request.guesses_per_state < 2 {
            bail!("learned-proxy state and guess budgets must be positive");
        }
        if request.maximum_seconds == 0 || request.maximum_memory_mb == 0 {
            bail!("learned-proxy time and memory budgets must be positive");
        }
        let plan = canonical_development_evaluation_plan(paths, "building learned-proxy data")?;
        if plan.folds.len() < 2 {
            bail!("learned-proxy data requires at least two development folds");
        }
        let validation_fold = &plan.folds[plan.folds.len() - 2];
        let test_fold = &plan.folds[plan.folds.len() - 1];
        let split = DatasetSplitMetadata::chronological(ChronologicalSplitMetadata {
            train_end: validation_fold.training.end,
            validation_start: validation_fold.validation.start,
            validation_end: validation_fold.validation.end,
            test_start: test_fold.validation.start,
            test_end: test_fold.validation.end,
        })?;
        let started = Instant::now();
        let budget = std::time::Duration::from_secs(request.maximum_seconds);
        let maximum_memory_bytes = request.maximum_memory_mb.saturating_mul(1024 * 1024);
        let input_fingerprint = development_source_identity(paths, plan.development.end)?;
        let executable_fingerprint = current_executable_fingerprint()?;
        let config_toml = toml::to_string_pretty(&self.config)?;
        let config_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-learned-proxy-config-v1",
            config_toml.as_bytes(),
        );
        let feature_names = learned_proxy_feature_names();
        let feature_digest =
            crate::predictive::learned_proxy::feature_schema_digest(&feature_names);
        let teacher_contract = super::exhaustive_teacher::CONTRACT;
        let replay_identity = ReplayIdentityInput {
            format_version: crate::experiments::exhaustive_cost::REPLAY_IDENTITY_FORMAT_VERSION,
            algorithm_version: teacher_contract.to_string(),
            solver_identity: crate::identity::digest_bytes_tagged(
                "maybe-wordle-independent-teacher-identity-v1",
                format!(
                    "{input_fingerprint}:{teacher_contract}:{feature_digest}:{}:{}:{}:{}:{}:{}",
                    self.mode.label(),
                    self.variant.label(),
                    request.minimum_survivors,
                    request.maximum_survivors,
                    request.maximum_states_per_split,
                    request.guesses_per_state,
                )
                .as_bytes(),
            ),
            source_data_fingerprint: input_fingerprint.clone(),
            config_fingerprint: config_fingerprint.clone(),
            feedback_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-feedback-contract-v1",
                b"wordle-two-pass-base3-all-green-242",
            ),
            state_encoding_version: 1,
            weighting_fingerprint: config_fingerprint.clone(),
        };
        let replay_identity_digest = replay_identity.digest_hex()?;
        let resource_budget = ResourceBudget {
            maximum_states: request.maximum_states_per_split.saturating_mul(3),
            maximum_rows: request
                .maximum_states_per_split
                .saturating_mul(3)
                .saturating_mul(
                    request
                        .guesses_per_state
                        .saturating_add(request.maximum_survivors),
                ),
            maximum_seconds: request.maximum_seconds,
            maximum_memory_bytes: Some(maximum_memory_bytes),
            checkpoint_every_rows: request.guesses_per_state.max(1),
        };
        resource_budget.validate()?;
        let ranges = [
            (
                DatasetSplit::Train,
                DateRange::new(plan.history.start, validation_fold.training.end)?,
            ),
            (DatasetSplit::Validation, validation_fold.validation),
            (DatasetSplit::Test, test_fold.validation),
        ];
        let mut rows = Vec::new();
        let mut completed_state_ids = BTreeSet::new();
        let mut completed_state_row_counts = std::collections::BTreeMap::new();
        let mut prior_elapsed_ms = 0_u64;
        let mut prior_peak_memory_bytes = 0_u64;
        if let Some(checkpoint_path) = request
            .checkpoint_path
            .as_ref()
            .filter(|path| path.exists())
        {
            let raw = fs::read_to_string(checkpoint_path).with_context(|| {
                format!(
                    "read learned-proxy checkpoint {}",
                    checkpoint_path.display()
                )
            })?;
            let checkpoint: ExhaustiveCostCheckpoint =
                serde_json::from_str(&raw).with_context(|| {
                    format!(
                        "decode learned-proxy checkpoint {}",
                        checkpoint_path.display()
                    )
                })?;
            checkpoint.validate(&split)?;
            if checkpoint.replay_identity_digest != replay_identity_digest {
                bail!(
                    "learned-proxy checkpoint {} belongs to different source/config/code inputs",
                    checkpoint_path.display()
                );
            }
            if checkpoint.budget != resource_budget {
                bail!(
                    "learned-proxy checkpoint {} uses a different resource or sampling budget",
                    checkpoint_path.display()
                );
            }
            prior_elapsed_ms = checkpoint.progress.elapsed_ms;
            prior_peak_memory_bytes = checkpoint.progress.peak_memory_bytes.unwrap_or(0);
            completed_state_ids.extend(checkpoint.completed_state_ids);
            completed_state_row_counts = checkpoint.completed_state_row_counts;
            rows = checkpoint.rows;
            eprintln!(
                "learned-proxy phase=resume states={} rows={} prior_elapsed_s={:.1} checkpoint={}",
                completed_state_ids.len(),
                rows.len(),
                prior_elapsed_ms as f64 / 1_000.0,
                checkpoint_path.display()
            );
        }
        let mut work_budget = super::exhaustive_teacher::WorkBudget::new(
            started,
            prior_elapsed_ms,
            budget,
            Some(maximum_memory_bytes),
        );
        let mut resume_validated = completed_state_ids.is_empty();
        let result = (|| -> Result<ExhaustiveCostDatasetArtifact> {
            if prior_peak_memory_bytes > maximum_memory_bytes {
                return Err(super::exhaustive_teacher::BudgetExceeded(
                    "offline evaluation previously exceeded its process memory budget",
                )
                .into());
            }
            work_budget.check_now()?;
            let mut selected_states = Vec::new();
            for (row_split, range) in ranges {
                work_budget.check()?;
                let collection = self.collect_learned_proxy_states(
                    &plan,
                    range,
                    request.minimum_survivors,
                    request.maximum_survivors,
                    request.maximum_states_per_split.saturating_mul(6).max(12),
                    started,
                    work_budget.remaining_run_time(),
                    false,
                    false,
                );
                work_budget.check_now()?;
                let (candidates, _) = collection?;
                if candidates.is_empty() {
                    bail!(
                        "learned-proxy split {:?} has no reachable states in {} through {}",
                        row_split,
                        range.start,
                        range.end
                    );
                }
                for index in
                    evenly_spaced_indices(candidates.len(), request.maximum_states_per_split)
                {
                    selected_states.push((row_split, candidates[index].clone()));
                }
            }
            let total_states = selected_states.len();
            let selected_state_ids = selected_states
                .iter()
                .map(|(_, candidate)| {
                    format!(
                        "{}-turn-{}-{}",
                        candidate.date, candidate.turn, candidate.target
                    )
                })
                .collect::<BTreeSet<_>>();
            if !completed_state_ids.is_subset(&selected_state_ids) {
                bail!("learned-proxy checkpoint contains states outside the deterministic sample");
            }
            let resumed_states = completed_state_ids.len();
            // Validate every resumed state before generating or publishing any new row.
            selected_states.sort_by_key(|(_, candidate)| {
                !completed_state_ids.contains(&format!(
                    "{}-turn-{}-{}",
                    candidate.date, candidate.turn, candidate.target,
                ))
            });
            let mut reconstructed_states = 0;
            let exact_started = Instant::now();
            for (row_split, candidate) in selected_states {
                let state_id = format!(
                    "{}-turn-{}-{}",
                    candidate.date, candidate.turn, candidate.target
                );
                work_budget.check()?;
                let already_completed = completed_state_ids.contains(&state_id);
                if !already_completed && rows.len() >= resource_budget.maximum_rows {
                    return Err(super::exhaustive_teacher::BudgetExceeded(
                        "learned-proxy dataset has no remaining row budget",
                    )
                    .into());
                }
                let as_of = candidate
                    .date
                    .checked_sub_days(Days::new(1))
                    .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
                let state = self.apply_history(as_of, &candidate.observations)?;
                if state.surviving.len() != candidate.surviving_answers {
                    bail!("learned-proxy state reconstruction changed survivor count");
                }
                let mut metrics =
                    self.score_guess_metrics_for_subset(&state.surviving, &state.weights);
                metrics.retain(|metric| reply_guess_makes_progress(metric, state.surviving.len()));
                let metric_by_guess = metrics
                    .iter()
                    .map(|metric| (metric.guess_index, *metric))
                    .collect::<HashMap<_, _>>();
                let guess_indexes = learned_proxy_guess_pool(
                    self,
                    &metrics,
                    &state.surviving,
                    request.guesses_per_state,
                );
                let trajectory_id = format!("{}-{}", candidate.date, candidate.target);
                let exact_state = ExactState {
                    state_id: state_id.clone(),
                    trajectory_id,
                    date: Some(candidate.date),
                    step_index: candidate.turn.saturating_sub(1),
                    survivor_ids: state
                        .surviving
                        .iter()
                        .map(|index| u32::try_from(*index).context("answer index exceeds u32"))
                        .collect::<Result<Vec<_>>>()?,
                    survivor_weights: state
                        .surviving
                        .iter()
                        .map(|index| state.weights[*index])
                        .collect(),
                };
                let requested_guesses = guess_indexes
                    .iter()
                    .map(|&guess| self.guesses[guess].clone())
                    .collect::<Vec<_>>();
                if already_completed {
                    let first = rows
                        .partition_point(|row: &ExhaustiveCostRow| row.state.state_id < state_id);
                    let count = completed_state_row_counts[&state_id];
                    crate::experiments::exhaustive_cost::validate_completed_state_rows(
                        &rows[first..first + count],
                        &exact_state,
                        row_split,
                        &requested_guesses,
                    )?;
                    reconstructed_states += 1;
                    resume_validated = reconstructed_states == resumed_states;
                    work_budget.check()?;
                    continue;
                }
                let labels = super::exhaustive_teacher::label_root_actions(
                    &state.surviving,
                    &state.weights,
                    self.guesses.len(),
                    &guess_indexes,
                    resource_budget.maximum_rows - rows.len(),
                    |guess, answer| self.pattern_table.get(guess, answer),
                    &mut work_budget,
                )?;
                let first_state_row = rows.len();
                for (guess_index, exact_cost) in labels {
                    work_budget.check()?;
                    let metric = metric_by_guess
                        .get(&guess_index)
                        .ok_or_else(|| anyhow!("teacher action has no predictive features"))?;
                    let mut row = ExhaustiveCostRow {
                        state: exact_state.clone(),
                        guess: self.guesses[guess_index].clone(),
                        exact_continuation_cost: exact_cost,
                        feature_values: learned_proxy_features(metric),
                        baseline_proxy_cost: Some(metric.proxy_cost),
                        split: row_split,
                    };
                    row.canonicalize_numeric_values();
                    rows.push(row);
                }
                crate::experiments::exhaustive_cost::validate_completed_state_rows(
                    &rows[first_state_row..],
                    &exact_state,
                    row_split,
                    &requested_guesses,
                )?;
                work_budget.check_now()?;
                let state_row_count = rows.len() - first_state_row;
                if state_row_count == 0 {
                    bail!("learned-proxy completed state {state_id} has no action labels");
                }
                completed_state_row_counts.insert(state_id.clone(), state_row_count);
                completed_state_ids.insert(state_id.clone());
                rows.sort_by_key(ExhaustiveCostRow::key);
                let elapsed_ms = work_budget.elapsed_ms();
                let completed = completed_state_ids.len();
                let completed_this_run = completed.saturating_sub(resumed_states);
                let eta_seconds = if completed_this_run == 0 {
                    0.0
                } else {
                    exact_started.elapsed().as_secs_f64() * (total_states - completed) as f64
                        / completed_this_run as f64
                };
                if let Some(checkpoint_path) = &request.checkpoint_path {
                    let checkpoint = ExhaustiveCostCheckpoint {
                        format_version:
                            crate::experiments::exhaustive_cost::EXHAUSTIVE_COST_FORMAT_VERSION,
                        replay_identity_digest: replay_identity_digest.clone(),
                        budget: resource_budget,
                        progress: ExhaustiveProgress {
                            phase: "exact".to_string(),
                            states_evaluated: completed,
                            rows_emitted: rows.len(),
                            elapsed_ms,
                            peak_memory_bytes: work_budget
                                .peak_memory_bytes()
                                .map(|peak| peak.max(prior_peak_memory_bytes)),
                            last_state_id: completed_state_ids.iter().next_back().cloned(),
                            complete: false,
                            stop_reason: None,
                        },
                        completed_state_ids: completed_state_ids.iter().cloned().collect(),
                        completed_state_row_counts: completed_state_row_counts.clone(),
                        rows: rows.clone(),
                    };
                    checkpoint.validate(&split)?;
                    work_budget.check_now()?;
                    crate::atomic_file::atomic_write(
                        checkpoint_path,
                        &serde_json::to_vec_pretty(&checkpoint)?,
                    )?;
                }
                eprintln!(
                    "learned-proxy phase=exact states={}/{} rows={} survivors={} elapsed_s={:.1} eta_s={:.1}",
                    completed,
                    total_states,
                    rows.len(),
                    state.surviving.len(),
                    elapsed_ms as f64 / 1_000.0,
                    eta_seconds
                );
                let _ = std::io::stderr().flush();
            }
            if completed_state_ids.len() != total_states {
                bail!(
                    "learned-proxy checkpoint/sample mismatch: completed {} of {} states",
                    completed_state_ids.len(),
                    total_states
                );
            }
            rows.sort_by_key(ExhaustiveCostRow::key);
            ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
            work_budget.check_now()?;
            let elapsed_ms = work_budget.elapsed_ms();
            let artifact = ExhaustiveCostDatasetArtifact {
                format_version: crate::experiments::exhaustive_cost::EXHAUSTIVE_COST_FORMAT_VERSION,
                provenance: DatasetProvenance {
                    dataset_id: crate::identity::digest_bytes_tagged(
                        "maybe-wordle-learned-proxy-dataset-v2",
                        format!("{}:{}", replay_identity_digest, rows.len()).as_bytes(),
                    ),
                    generator_version: teacher_contract.to_string(),
                    source_identity: input_fingerprint.clone(),
                    source_data_fingerprint: input_fingerprint.clone(),
                    config_fingerprint,
                    executable_fingerprint: Some(executable_fingerprint),
                    cutoff_start: plan.history.start,
                    cutoff_end: plan.development.end,
                    replay_identity,
                },
                split: split.clone(),
                budget: resource_budget,
                progress: ExhaustiveProgress {
                    phase: "complete".to_string(),
                    states_evaluated: total_states,
                    rows_emitted: rows.len(),
                    elapsed_ms,
                    peak_memory_bytes: work_budget
                        .peak_memory_bytes()
                        .map(|peak| peak.max(prior_peak_memory_bytes)),
                    last_state_id: rows.last().map(|row| row.state.state_id.clone()),
                    complete: true,
                    stop_reason: None,
                },
                completed_state_row_counts: completed_state_row_counts.clone(),
                rows: rows.clone(),
                checkpoint: None,
            };
            artifact.validate()?;
            work_budget.check_now()?;
            if let Some(checkpoint_path) = &request.checkpoint_path {
                let checkpoint = ExhaustiveCostCheckpoint {
                    format_version:
                        crate::experiments::exhaustive_cost::EXHAUSTIVE_COST_FORMAT_VERSION,
                    replay_identity_digest: replay_identity_digest.clone(),
                    budget: resource_budget,
                    progress: artifact.progress.clone(),
                    completed_state_ids: completed_state_ids.iter().cloned().collect(),
                    completed_state_row_counts: completed_state_row_counts.clone(),
                    rows: artifact.rows.clone(),
                };
                checkpoint.validate(&artifact.split)?;
                work_budget.check_now()?;
                crate::atomic_file::atomic_write(
                    checkpoint_path,
                    &serde_json::to_vec_pretty(&checkpoint)?,
                )?;
            }
            work_budget.check_now()?;
            eprintln!(
                "learned-proxy phase=complete features={} rows={} elapsed_s={:.1}",
                feature_names.len(),
                artifact.rows.len(),
                elapsed_ms as f64 / 1_000.0
            );
            Ok(artifact)
        })();
        if let Err(error) = &result
            && error.is::<super::exhaustive_teacher::BudgetExceeded>()
            && resume_validated
            && let Some(checkpoint_path) = &request.checkpoint_path
        {
            // Keep only validated whole states. Interrupted state work is charged
            // even though its unfinished labels are deliberately not retained.
            rows.retain(|row| completed_state_ids.contains(&row.state.state_id));
            rows.sort_by_key(ExhaustiveCostRow::key);
            let checkpoint = ExhaustiveCostCheckpoint {
                format_version: crate::experiments::exhaustive_cost::EXHAUSTIVE_COST_FORMAT_VERSION,
                replay_identity_digest,
                budget: resource_budget,
                progress: ExhaustiveProgress {
                    phase: "interrupted".into(),
                    states_evaluated: completed_state_ids.len(),
                    rows_emitted: rows.len(),
                    elapsed_ms: work_budget.elapsed_ms(),
                    peak_memory_bytes: crate::process_memory::process_memory_snapshot()
                        .map(|snapshot| {
                            snapshot.peak_working_set_bytes.max(prior_peak_memory_bytes)
                        })
                        .or(work_budget.peak_memory_bytes()),
                    last_state_id: completed_state_ids.iter().next_back().cloned(),
                    complete: false,
                    stop_reason: Some(error.to_string()),
                },
                completed_state_ids: completed_state_ids.into_iter().collect(),
                completed_state_row_counts,
                rows,
            };
            checkpoint.validate(&split)?;
            crate::atomic_file::atomic_write(
                checkpoint_path,
                &serde_json::to_vec_pretty(&checkpoint)?,
            )?;
        }
        result
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "shared offline collector keeps sampling rules and deadline behavior explicit"
    )]
    fn collect_learned_proxy_states(
        &self,
        plan: &EvaluationPlan,
        range: DateRange,
        minimum_survivors: usize,
        maximum_survivors: usize,
        maximum_games: usize,
        started: Instant,
        budget: std::time::Duration,
        hard_mode: bool,
        allow_partial_on_deadline: bool,
    ) -> Result<(Vec<SearchRegretCandidateState>, usize)> {
        let eligible_dates = plan.eligible_target_dates(
            range,
            self.history_dates.iter().map(|entry| entry.print_date),
        )?;
        let games = self
            .history_dates
            .iter()
            .filter(|entry| eligible_dates.contains(&entry.print_date))
            .collect::<Vec<_>>();
        let indices = evenly_spaced_indices(games.len(), maximum_games);
        let total = indices.len();
        let mut candidates = Vec::new();
        let mut scanned_games = 0usize;
        'games: for (game_number, index) in indices.into_iter().enumerate() {
            if started.elapsed() >= budget {
                if allow_partial_on_deadline {
                    break;
                }
                bail!("learned-proxy collection exceeded its wall-clock budget");
            }
            let entry = games[index];
            scanned_games += 1;
            let as_of = entry
                .print_date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
            let target = entry.solution.to_ascii_lowercase();
            let mut state = self.initial_state(as_of);
            let mut observations = Vec::new();
            for step_index in 0..5 {
                if started.elapsed() >= budget {
                    if allow_partial_on_deadline {
                        break 'games;
                    }
                    bail!("learned-proxy collection exceeded its wall-clock budget");
                }
                if (minimum_survivors..=maximum_survivors).contains(&state.surviving.len()) {
                    candidates.push(SearchRegretCandidateState {
                        date: entry.print_date,
                        target: target.clone(),
                        turn: step_index + 1,
                        observations,
                        surviving_answers: state.surviving.len(),
                    });
                    break;
                }
                if state.surviving.len() < minimum_survivors {
                    break;
                }
                let cancelled = || started.elapsed() >= budget;
                let batch = self.suggestion_batch_internal_with_search_mode_controlled(
                    &state,
                    if hard_mode { self.guesses.len() } else { 1 },
                    Some(PredictiveContext {
                        hard_mode,
                        as_of,
                        observations: &observations,
                    }),
                    PredictiveBookUsage::None,
                    Some(PredictiveSearchMode::ProxyOnly),
                    &cancelled,
                );
                let mut batch = match batch {
                    Ok(batch) => batch,
                    Err(_) if cancelled() && allow_partial_on_deadline => break 'games,
                    Err(_) if cancelled() => {
                        bail!("learned-proxy collection exceeded its wall-clock budget")
                    }
                    Err(error) => return Err(error),
                };
                if started.elapsed() >= budget {
                    if allow_partial_on_deadline {
                        break 'games;
                    }
                    bail!("learned-proxy collection exceeded its wall-clock budget");
                }
                let chosen = batch
                    .suggestions
                    .drain(..)
                    .find(|suggestion| {
                        !hard_mode
                            || self
                                .hard_mode_violation(&observations, &suggestion.word)
                                .is_none()
                    })
                    .ok_or_else(|| anyhow!("proxy path returned no suggestion"))?;
                let feedback = score_guess(&chosen.word, &target);
                if feedback == ALL_GREEN_PATTERN {
                    break;
                }
                observations.push((chosen.word.clone(), feedback));
                self.apply_feedback(&mut state, &chosen.word, feedback)?;
            }
            if started.elapsed() >= budget {
                if allow_partial_on_deadline {
                    break;
                }
                bail!("learned-proxy collection exceeded its wall-clock budget");
            }
            if game_number < 2 || (game_number + 1) % 10 == 0 || game_number + 1 == total {
                eprintln!(
                    "learned-proxy phase=collect games={}/{} states={} elapsed_s={:.1}",
                    game_number + 1,
                    total,
                    candidates.len(),
                    started.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
            }
        }
        Ok((candidates, scanned_games))
    }

    pub fn search_regret_report(
        &self,
        paths: &ProjectPaths,
        request: SearchRegretRequest,
    ) -> Result<SearchRegretReport> {
        let SearchRegretRequest {
            from,
            to,
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
        } = request;
        if from > to {
            bail!("search-regret start date cannot be after end date");
        }
        if minimum_survivors < 2 {
            bail!("search-regret minimum survivors must be at least 2");
        }
        if minimum_survivors > maximum_survivors {
            bail!("search-regret minimum survivors cannot exceed maximum survivors");
        }
        if maximum_states == 0 {
            bail!("search-regret maximum states must be greater than zero");
        }
        if maximum_seconds == 0 {
            bail!("search-regret maximum seconds must be greater than zero");
        }

        let plan = ensure_development_target_range(paths, from, to, "search-regret")?;
        let started = Instant::now();
        let input_fingerprint = development_source_identity(paths, plan.development.end)?;
        let budget = std::time::Duration::from_secs(maximum_seconds);
        let cancelled = || {
            #[cfg(test)]
            if REGRET_STOP_AFTER_STATES.with(|remaining| remaining.get() == Some(0)) {
                return true;
            }
            started.elapsed() >= budget
        };
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .cloned()
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no games found in the requested search-regret range");
        }

        let historical_games = games.len();
        let game_scan_limit = historical_games.min(maximum_states.saturating_mul(2).max(4));
        let game_indices = evenly_spaced_indices(historical_games, game_scan_limit);
        let games = game_indices
            .into_iter()
            .map(|index| games[index].clone())
            .collect::<Vec<_>>();
        let total_games = games.len();
        let mut candidates = Vec::new();
        let mut scanned_games = 0;
        let mut stop_reason = None;
        'collect: for (game_index, entry) in games.into_iter().enumerate() {
            if cancelled() {
                stop_reason = Some("global_deadline_during_collection".to_string());
                break;
            }
            scanned_games += 1;
            let as_of = entry
                .print_date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
            let target = entry.solution.to_ascii_lowercase();
            let mut state = self.initial_state(as_of);
            let mut observations = Vec::new();
            if state
                .surviving
                .iter()
                .chain(state.fallback_surviving.iter())
                .any(|index| self.answers[*index].word == target)
            {
                for step_index in 0..5 {
                    if cancelled() {
                        stop_reason = Some("global_deadline_during_collection".to_string());
                        break 'collect;
                    }
                    if (minimum_survivors..=maximum_survivors).contains(&state.surviving.len()) {
                        candidates.push(SearchRegretCandidateState {
                            date: entry.print_date,
                            target: target.clone(),
                            turn: step_index + 1,
                            observations: observations.clone(),
                            surviving_answers: state.surviving.len(),
                        });
                        break;
                    }
                    if state.surviving.len() < minimum_survivors {
                        break;
                    }
                    let batch = self.suggestion_batch_internal_with_search_mode_controlled(
                        &state,
                        1,
                        Some(PredictiveContext {
                            hard_mode: false,
                            as_of,
                            observations: &observations,
                        }),
                        PredictiveBookUsage::None,
                        Some(PredictiveSearchMode::ProxyOnly),
                        &cancelled,
                    );
                    let batch = match batch {
                        Ok(batch) => batch,
                        Err(_) if cancelled() => {
                            stop_reason = Some("global_deadline_during_collection".to_string());
                            break 'collect;
                        }
                        Err(error) => return Err(error),
                    };
                    let chosen = batch
                        .suggestions
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow!("solver returned no suggestion during audit"))?;
                    let feedback = score_guess(&chosen.word, &target);
                    if feedback == ALL_GREEN_PATTERN {
                        break;
                    }
                    observations.push((chosen.word.clone(), feedback));
                    self.apply_feedback(&mut state, &chosen.word, feedback)?;
                }
            }
            if game_index < 2 || (game_index + 1) % 5 == 0 || game_index + 1 == total_games {
                eprintln!(
                    "search-regret phase=collect games={}/{} eligible_states={} elapsed_s={:.1}",
                    game_index + 1,
                    total_games,
                    candidates.len(),
                    started.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
            }
        }
        if candidates.is_empty() && stop_reason.is_none() {
            bail!(
                "no reachable states had between {} and {} survivors",
                minimum_survivors,
                maximum_survivors
            );
        }

        let available_states = candidates.len();
        let selected_indices = evenly_spaced_indices(available_states, maximum_states);
        let mut exhaustive_solver = self.clone();
        exhaustive_solver.config.exact_threshold = maximum_survivors;
        exhaustive_solver.config.exact_exhaustive_threshold = maximum_survivors;
        let mut states = Vec::with_capacity(selected_indices.len());
        let sampled_states = selected_indices.len();
        for (sample_index, candidate_index) in selected_indices.into_iter().enumerate() {
            if cancelled() {
                stop_reason = Some("global_deadline_during_state_audit".to_string());
                break;
            }
            let candidate = &candidates[candidate_index];
            let audited = (|| -> Result<SearchRegretState> {
                let as_of = candidate
                    .date
                    .checked_sub_days(Days::new(1))
                    .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
                let state = self.apply_history(as_of, &candidate.observations)?;
                if state.surviving.len() != candidate.surviving_answers {
                    bail!(
                        "search-regret state reconstruction mismatch for {} turn {}: expected {} survivors, reconstructed {}",
                        candidate.date,
                        candidate.turn,
                        candidate.surviving_answers,
                        state.surviving.len()
                    );
                }

                let context = Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &candidate.observations,
                });
                let proxy_guess = self
                    .suggestion_batch_internal_with_search_mode_controlled(
                        &state,
                        1,
                        context,
                        PredictiveBookUsage::None,
                        Some(PredictiveSearchMode::ProxyOnly),
                        &cancelled,
                    )?
                    .suggestions
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("proxy audit returned no suggestion"))?
                    .word;
                let lookahead_guess = self
                    .suggestion_batch_internal_with_search_mode_controlled(
                        &state,
                        1,
                        context,
                        PredictiveBookUsage::None,
                        Some(PredictiveSearchMode::Lookahead),
                        &cancelled,
                    )?
                    .suggestions
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("lookahead audit returned no suggestion"))?
                    .word;
                let production_is_exhaustive = self.config.search_policy_mode
                    != crate::config::SearchPolicyMode::ProxyOnly
                    && matches!(
                        exact_suggestion_mode(&self.config, state.surviving.len()),
                        Some(ExactSuggestionMode::Exhaustive)
                    );
                let (production_guess, production_regime) = if production_is_exhaustive {
                    (None, PredictiveRegime::Exact)
                } else {
                    let production = self.suggestion_batch_internal_with_search_mode_controlled(
                        &state,
                        1,
                        context,
                        PredictiveBookUsage::None,
                        None,
                        &cancelled,
                    )?;
                    (
                        Some(
                            production
                                .suggestions
                                .into_iter()
                                .next()
                                .ok_or_else(|| anyhow!("production audit returned no suggestion"))?
                                .word,
                        ),
                        production.regime_used,
                    )
                };
                exhaustive_solver.audit_search_regret_state(
                    candidate,
                    &state,
                    production_guess.as_deref(),
                    production_regime,
                    &proxy_guess,
                    &lookahead_guess,
                    &cancelled,
                )
            })();
            match audited {
                Ok(state) => states.push(state),
                Err(_) if cancelled() => {
                    stop_reason = Some("global_deadline_during_state_audit".to_string());
                    break;
                }
                Err(error) => return Err(error),
            }
            #[cfg(test)]
            REGRET_STOP_AFTER_STATES.with(|remaining| {
                if let Some(count) = remaining.get() {
                    remaining.set(Some(count.saturating_sub(1)));
                }
            });
            eprintln!(
                "search-regret phase=exhaustive states={}/{} survivors={} elapsed_s={:.1}",
                sample_index + 1,
                sampled_states,
                candidate.surviving_answers,
                started.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }

        let config_toml =
            toml::to_string_pretty(&self.config).context("failed to serialize audit config")?;
        ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        if cancelled() && stop_reason.is_none() {
            stop_reason = Some("global_deadline_during_finalization".to_string());
        }
        Ok(SearchRegretReport {
            schema_version: 2,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-search-regret-config-v1",
                config_toml.as_bytes(),
            ),
            code_revision,
            code_dirty,
            evaluation_from: from,
            evaluation_to: to,
            state_path_policy: "forced_proxy_without_artifacts".to_string(),
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
            historical_games,
            scanned_games,
            available_states,
            planned_states: sampled_states,
            sampled_states: states.len(),
            complete: stop_reason.is_none() && states.len() == sampled_states,
            stop_reason,
            generation_elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            production: summarize_search_regret(&states, |state| &state.production),
            proxy: summarize_search_regret(&states, |state| &state.proxy),
            lookahead: summarize_search_regret(&states, |state| &state.lookahead),
            states,
        })
    }

    pub fn finite_search_regret_report(
        &self,
        paths: &ProjectPaths,
        request: FiniteSearchRegretRequest,
    ) -> Result<FiniteSearchRegretReport> {
        let FiniteSearchRegretRequest {
            from,
            to,
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
            hard_mode,
        } = request;
        if !matches!(
            self.config.search_policy_mode,
            crate::config::SearchPolicyMode::FiniteFast
                | crate::config::SearchPolicyMode::FiniteStrong
        ) {
            bail!(
                "finite search-regret requires a finite_fast or finite_strong config; got {}",
                self.config.search_policy_mode.label()
            );
        }
        if from > to {
            bail!("finite search-regret start date cannot be after end date");
        }
        if minimum_survivors < 2 {
            bail!("finite search-regret minimum survivors must be at least 2");
        }
        if minimum_survivors > maximum_survivors {
            bail!("finite search-regret minimum survivors cannot exceed maximum survivors");
        }
        if maximum_states == 0 {
            bail!("finite search-regret maximum states must be greater than zero");
        }
        if maximum_seconds == 0 {
            bail!("finite search-regret maximum seconds must be greater than zero");
        }

        let plan = ensure_development_target_range(paths, from, to, "finite search-regret")?;
        let started = Instant::now();
        let budget = std::time::Duration::from_secs(maximum_seconds);
        let input_fingerprint = development_source_identity(paths, plan.development.end)?;
        let range = DateRange::new(from, to)?;
        let historical_games = self
            .history_dates
            .iter()
            .filter(|entry| range.contains(entry.print_date))
            .count();
        let (candidates, scanned_games) = self.collect_learned_proxy_states(
            &plan,
            range,
            minimum_survivors,
            maximum_survivors,
            maximum_states.saturating_mul(2).max(4),
            started,
            budget,
            hard_mode,
            true,
        )?;
        if candidates.is_empty() {
            bail!(
                "no reachable states had between {} and {} survivors",
                minimum_survivors,
                maximum_survivors
            );
        }

        let available_states = candidates.len();
        let selected_indices = evenly_spaced_indices(available_states, maximum_states);
        let sampled_states = selected_indices.len();
        let mut states = Vec::with_capacity(sampled_states);
        for (sample_index, candidate_index) in selected_indices.into_iter().enumerate() {
            let candidate = &candidates[candidate_index];
            let as_of = candidate
                .date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
            let state = self.apply_history(as_of, &candidate.observations)?;
            if state.surviving.len() != candidate.surviving_answers {
                bail!(
                    "finite search-regret state reconstruction mismatch for {} turn {}: expected {} survivors, reconstructed {}",
                    candidate.date,
                    candidate.turn,
                    candidate.surviving_answers,
                    state.surviving.len()
                );
            }
            states.push(
                self.finite_search_regret_state(candidate, &state, hard_mode, started, budget)?,
            );
            eprintln!(
                "finite-search-regret phase=reference states={}/{} survivors={} elapsed_s={:.1}",
                sample_index + 1,
                sampled_states,
                candidate.surviving_answers,
                started.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }

        let config_toml =
            toml::to_string_pretty(&self.config).context("failed to serialize audit config")?;
        let config_identity = format!("hard_mode={hard_mode}\n{config_toml}");
        ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        Ok(FiniteSearchRegretReport {
            schema_version: FINITE_SEARCH_REGRET_SCHEMA_VERSION,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-finite-search-regret-config-v1",
                config_identity.as_bytes(),
            ),
            code_revision,
            code_dirty,
            evaluation_from: from,
            evaluation_to: to,
            state_path_policy: format!("forced_proxy_without_artifacts_hard_mode={hard_mode}"),
            reference_kernel: "finite_horizon_search/shared_kernel".to_string(),
            independent_cross_check: "toy-only independent oracle; not used for report values"
                .to_string(),
            rules: if hard_mode {
                "wordle_feedback_v1+hard_mode".to_string()
            } else {
                "wordle_feedback_v1+normal_mode".to_string()
            },
            search_policy_mode: self.config.search_policy_mode.label().to_string(),
            hard_mode,
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
            historical_games,
            scanned_games,
            available_states,
            sampled_states: states.len(),
            generation_elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            summary: summarize_finite_search_regret(&states),
            states,
        })
    }

    pub fn same_state_dynamic_regret_report(
        &self,
        paths: &ProjectPaths,
        date: NaiveDate,
        turn: usize,
        maximum_seconds: u64,
    ) -> Result<SameStateDynamicRegretReport> {
        if self.config.search_policy_mode != crate::config::SearchPolicyMode::Staged {
            bail!("same-state dynamic regret requires a staged input config");
        }
        if !(1..=6).contains(&turn) {
            bail!("same-state dynamic regret turn must be between one and six");
        }
        if maximum_seconds == 0 {
            bail!("same-state dynamic regret maximum seconds must be greater than zero");
        }
        let plan = ensure_development_target_range(paths, date, date, "same-state dynamic regret")?;
        let started = Instant::now();
        let budget = std::time::Duration::from_secs(maximum_seconds);
        let input_fingerprint = development_source_identity(paths, plan.development.end)?;
        let mut matching_dates = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date == date);
        let entry = matching_dates
            .next()
            .ok_or_else(|| anyhow!("selected development date has no historical game"))?;
        if matching_dates.next().is_some() {
            bail!("selected development date has duplicate historical games");
        }
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("selected game precedes the supported history"))?;
        let target = entry.solution.to_ascii_lowercase();
        let mut staged_state = self.initial_state(as_of);
        if !staged_state
            .surviving
            .iter()
            .chain(&staged_state.fallback_surviving)
            .any(|index| self.answers[*index].word == target)
        {
            bail!("selected staged path is outside the modeled answer support");
        }

        let mut observations = Vec::with_capacity(turn.saturating_sub(1));
        for _ in 1..turn {
            let runtime = self.same_state_runtime_choice(
                &staged_state,
                as_of,
                &observations,
                started,
                budget,
            )?;
            let guess = same_state_replay_guess(runtime)?;
            let feedback = score_guess(&guess, &target);
            if feedback == ALL_GREEN_PATTERN {
                bail!("selected turn occurs after the staged path was solved");
            }
            observations.push((guess.clone(), feedback));
            self.apply_feedback(&mut staged_state, &guess, feedback)
                .map_err(|_| anyhow!("staged artifact-free path could not be reconstructed"))?;
        }

        let mut dynamic_solver = self.clone();
        dynamic_solver.config.search_policy_mode =
            crate::config::SearchPolicyMode::FiniteFastDynamic;
        let mut dynamic_state = dynamic_solver.initial_state(as_of);
        if !dynamic_state
            .surviving
            .iter()
            .chain(&dynamic_state.fallback_surviving)
            .any(|index| dynamic_solver.answers[*index].word == target)
        {
            bail!("selected path is outside the dynamic answer support");
        }
        for (guess, feedback) in &observations {
            dynamic_solver
                .apply_feedback(&mut dynamic_state, guess, *feedback)
                .map_err(|_| {
                    anyhow!("dynamic belief could not be reconstructed from the selected path")
                })?;
        }
        if !dynamic_state
            .surviving
            .iter()
            .chain(&dynamic_state.fallback_surviving)
            .any(|index| dynamic_solver.answers[*index].word == target)
        {
            bail!("selected path left the dynamic answer support");
        }

        let staged_runtime =
            self.same_state_runtime_choice(&dynamic_state, as_of, &observations, started, budget)?;
        let dynamic_runtime = dynamic_solver.same_state_runtime_choice(
            &dynamic_state,
            as_of,
            &observations,
            started,
            budget,
        )?;
        let selected_roots = [
            staged_runtime
                .word
                .as_ref()
                .and_then(|word| dynamic_solver.guess_index.get(word).copied()),
            dynamic_runtime
                .word
                .as_ref()
                .and_then(|word| dynamic_solver.guess_index.get(word).copied()),
        ];
        let selected_root_indices = selected_roots.iter().flatten().copied().collect::<Vec<_>>();
        let choices_differ = staged_runtime
            .word
            .as_ref()
            .zip(dynamic_runtime.word.as_ref())
            .map(|(staged, dynamic)| staged != dynamic);
        let (global_reference, selected_references) = dynamic_solver
            .same_state_dynamic_root_references(
                &dynamic_state,
                &observations,
                (7 - turn) as u8,
                &selected_root_indices,
                started,
                budget,
            )?;
        let mut selected_references = selected_references.into_iter();
        let mut next_selected_reference = || {
            selected_references
                .next()
                .unwrap_or(FiniteSearchRegretReference {
                    status: "unresolved_selected_root".to_string(),
                    word: None,
                    value: None,
                })
        };
        let staged_reference = if selected_roots[0].is_some() {
            next_selected_reference()
        } else {
            FiniteSearchRegretReference {
                status: "unresolved_runtime_choice".to_string(),
                word: None,
                value: None,
            }
        };
        let dynamic_reference = if selected_roots[1].is_some() {
            next_selected_reference()
        } else {
            FiniteSearchRegretReference {
                status: "unresolved_runtime_choice".to_string(),
                word: None,
                value: None,
            }
        };
        let staged =
            same_state_dynamic_regret_choice(staged_runtime, &staged_reference, &global_reference)?;
        let finite_fast_dynamic = same_state_dynamic_regret_choice(
            dynamic_runtime,
            &dynamic_reference,
            &global_reference,
        )?;
        let config_toml = toml::to_string_pretty(&self.config)
            .context("failed to serialize staged audit config")?;
        let config_identity = format!(
            "staged_mode={}\ndynamic_mode=finite_fast_dynamic\n{config_toml}",
            self.config.search_policy_mode.label()
        );
        ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        Ok(SameStateDynamicRegretReport {
            schema_version: 1,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-same-state-dynamic-regret-config-v1",
                config_identity.as_bytes(),
            ),
            code_revision,
            code_dirty,
            date,
            turn,
            horizon: (7 - turn) as u8,
            state_path_policy:
                "explicit_date_turn_staged_artifact_free_dynamic_replay_same_observations"
                    .to_string(),
            state_identity: same_state_dynamic_identity(date, turn, &observations, &dynamic_state),
            active_survivors: dynamic_state.surviving.len(),
            dormant_fallback_survivors: dynamic_state.fallback_surviving.len(),
            reference_support_limit: SAME_STATE_DYNAMIC_MAX_REFERENCE_SUPPORT,
            staged_search_policy_mode: self.config.search_policy_mode.label().to_string(),
            choices_differ,
            reference_status: global_reference.status,
            reference_value: global_reference.value,
            legal_root_count: dynamic_solver.guesses.len(),
            maximum_seconds,
            generation_elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            staged,
            finite_fast_dynamic,
        })
    }

    fn staged_zero_failure_root_check(
        &self,
        state: &SolveState,
        observations: &[(String, u8)],
        root_index: usize,
        horizon: u8,
        hard_mode: bool,
    ) -> Result<StagedZeroFailureRootCheck> {
        if root_index >= self.guesses.len() || horizon == 0 {
            bail!("staged zero-failure root check received an invalid root or horizon");
        }
        let root = &self.guesses[root_index];
        if hard_mode && self.hard_mode_violation(observations, root).is_some() {
            return Ok(StagedZeroFailureRootCheck {
                certified: false,
                reason: "hard_mode",
            });
        }
        if !state.fallback_surviving.is_empty() {
            return Ok(StagedZeroFailureRootCheck {
                certified: false,
                reason: "dormant_support",
            });
        }

        let mut partitions = (0..PATTERN_SPACE)
            .map(|_| Vec::new())
            .collect::<Vec<Vec<usize>>>();
        for &answer_index in &state.surviving {
            if answer_index >= self.answers.len() {
                return Ok(StagedZeroFailureRootCheck {
                    certified: false,
                    reason: "transition",
                });
            }
            partitions[self.answer_pattern(root_index, answer_index) as usize].push(answer_index);
        }

        for (pattern, bucket) in partitions.into_iter().enumerate() {
            if bucket.is_empty() || pattern == ALL_GREEN_PATTERN as usize {
                continue;
            }
            let mut child = state.clone();
            if self
                .apply_feedback(&mut child, root, pattern as u8)
                .is_err()
            {
                return Ok(StagedZeroFailureRootCheck {
                    certified: false,
                    reason: "transition",
                });
            }
            if !child.fallback_surviving.is_empty() {
                return Ok(StagedZeroFailureRootCheck {
                    certified: false,
                    reason: "dormant_support",
                });
            }
            if child.surviving.iter().any(|&answer_index| {
                child
                    .modeled_weights
                    .get(answer_index)
                    .is_none_or(|weight| !weight.is_finite() || *weight <= 0.0)
            }) {
                return Ok(StagedZeroFailureRootCheck {
                    certified: false,
                    reason: "unstable_modeled_support",
                });
            }
            if child.surviving.len() > usize::from(horizon.saturating_sub(1)) {
                return Ok(StagedZeroFailureRootCheck {
                    certified: false,
                    reason: "bucket_cardinality",
                });
            }
            let mut child_observations = observations.to_vec();
            child_observations.push((root.clone(), pattern as u8));
            for &answer_index in &child.surviving {
                let Some(answer) = self.answers.get(answer_index) else {
                    return Ok(StagedZeroFailureRootCheck {
                        certified: false,
                        reason: "transition",
                    });
                };
                if !self.guess_index.contains_key(&answer.word) {
                    return Ok(StagedZeroFailureRootCheck {
                        certified: false,
                        reason: "missing_dictionary_answer",
                    });
                }
                if hard_mode
                    && self
                        .hard_mode_violation(&child_observations, &answer.word)
                        .is_some()
                {
                    return Ok(StagedZeroFailureRootCheck {
                        certified: false,
                        reason: "hard_mode",
                    });
                }
            }
        }
        Ok(StagedZeroFailureRootCheck {
            certified: true,
            reason: "certified",
        })
    }

    fn staged_target_index_if_supported(&self, state: &SolveState, target: &str) -> Option<usize> {
        state.surviving.iter().copied().find(|&answer_index| {
            self.answers
                .get(answer_index)
                .is_some_and(|answer| answer.word == target)
                && state
                    .modeled_weights
                    .get(answer_index)
                    .is_some_and(|weight| weight.is_finite() && *weight > 0.0)
        })
    }

    pub fn staged_zero_failure_certificate_report(
        &self,
        paths: &ProjectPaths,
        request: StagedZeroFailureCertificateRequest,
    ) -> Result<StagedZeroFailureCertificateReport> {
        if self.config.search_policy_mode != crate::config::SearchPolicyMode::Staged {
            bail!("staged zero-failure certificate requires a staged input config");
        }
        if request.from > request.to {
            bail!("staged zero-failure certificate start date cannot be after end date");
        }
        if request.maximum_states == 0 {
            bail!("staged zero-failure certificate maximum states must be greater than zero");
        }
        if request.maximum_seconds == 0 {
            bail!("staged zero-failure certificate maximum seconds must be greater than zero");
        }
        let started = Instant::now();
        let plan = ensure_development_target_range(
            paths,
            request.from,
            request.to,
            "staged zero-failure certificate",
        )?;
        let budget = Duration::from_secs(request.maximum_seconds);
        let input_fingerprint = development_source_identity(paths, plan.development.end)?;
        let range = DateRange::new(request.from, request.to)?;
        let games = self
            .history_dates
            .iter()
            .filter(|entry| range.contains(entry.print_date))
            .collect::<Vec<_>>();
        let mut date_counts = HashMap::new();
        for entry in &games {
            *date_counts.entry(entry.print_date).or_insert(0usize) += 1;
        }
        let scheduled_days = range.days();
        let coverage_gaps = scheduled_days.saturating_sub(date_counts.len() as u64);
        let duplicate_history_dates = date_counts
            .values()
            .map(|count| count.saturating_sub(1))
            .sum();
        let historical_games = games.len();
        let mut scanned_games = 0;
        let mut selected_roots = 0;
        let mut evaluated_roots = 0;
        let mut certified_roots = 0;
        let mut bucket_cardinality_rejections = 0;
        let mut dormant_support_rejections = 0;
        let mut unstable_modeled_support_rejections = 0;
        let mut missing_dictionary_rejections = 0;
        let mut hard_mode_rejections = 0;
        let mut transition_rejections = 0;
        let mut state_cap_reached = false;
        let mut deadline_reached = false;
        let mut unsupported_target_games = 0;
        let mut replayed_games = 0;
        let mut path_replay_failures = 0;

        'games: for entry in games {
            if started.elapsed() >= budget {
                deadline_reached = true;
                break;
            }
            scanned_games += 1;
            let as_of = entry
                .print_date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
            let target = entry.solution.to_ascii_lowercase();
            let mut state = self.initial_state(as_of);
            let mut observations = Vec::new();
            let mut unsupported_target = false;
            let mut replayed_game = false;
            let mut path_replay_failed = false;
            for turn in 1..=6usize {
                if started.elapsed() >= budget {
                    deadline_reached = true;
                    break 'games;
                }
                if selected_roots >= request.maximum_states {
                    state_cap_reached = true;
                    break 'games;
                }
                if self
                    .staged_target_index_if_supported(&state, &target)
                    .is_none()
                {
                    unsupported_target_games += 1;
                    unsupported_target = true;
                    break;
                }
                if state.surviving.is_empty() || state.total_weight <= 0.0 {
                    transition_rejections += 1;
                    path_replay_failed = true;
                    break;
                }
                let Some(root) = self.staged_artifact_free_root_guess(
                    &state,
                    as_of,
                    &observations,
                    request.hard_mode,
                    started,
                    budget,
                )?
                else {
                    if started.elapsed() >= budget {
                        deadline_reached = true;
                        break 'games;
                    }
                    transition_rejections += 1;
                    path_replay_failed = true;
                    break;
                };
                if started.elapsed() >= budget {
                    deadline_reached = true;
                    break 'games;
                }
                let Some(&root_index) = self.guess_index.get(&root) else {
                    transition_rejections += 1;
                    path_replay_failed = true;
                    break;
                };
                selected_roots += 1;
                let horizon = (7 - turn) as u8;
                let outcome = self.staged_zero_failure_root_check(
                    &state,
                    &observations,
                    root_index,
                    horizon,
                    request.hard_mode,
                )?;
                if started.elapsed() >= budget {
                    deadline_reached = true;
                    break 'games;
                }
                evaluated_roots += 1;
                match outcome.reason {
                    "bucket_cardinality" => bucket_cardinality_rejections += 1,
                    "dormant_support" => dormant_support_rejections += 1,
                    "unstable_modeled_support" => unstable_modeled_support_rejections += 1,
                    "missing_dictionary_answer" => missing_dictionary_rejections += 1,
                    "hard_mode" => hard_mode_rejections += 1,
                    "transition" => transition_rejections += 1,
                    "certified" => {}
                    other => bail!("unknown staged zero-failure root result {other}"),
                }

                let feedback = score_guess(&root, &target);
                if feedback == ALL_GREEN_PATTERN {
                    if outcome.certified {
                        certified_roots += 1;
                    }
                    replayed_game = true;
                    break;
                }
                if turn == 6 {
                    // A non-green sixth guess is a valid completed replay, but it has no
                    // remaining turn in which to certify the final child.
                    replayed_game = true;
                    break;
                }
                let mut next_state = state.clone();
                if self
                    .apply_feedback(&mut next_state, &root, feedback)
                    .is_err()
                {
                    if outcome.reason != "transition" {
                        transition_rejections += 1;
                    }
                    path_replay_failed = true;
                    // A failed actual transition invalidates this otherwise local witness and
                    // prevents replaying later roots from a stale state.
                    break;
                }
                if outcome.certified {
                    certified_roots += 1;
                }
                observations.push((root, feedback));
                state = next_state;
            }
            if !replayed_game && !unsupported_target && !path_replay_failed {
                path_replay_failed = true;
            }
            if replayed_game {
                replayed_games += 1;
            } else if path_replay_failed {
                path_replay_failures += 1;
            }
        }

        let config_toml = toml::to_string_pretty(&self.config)
            .context("failed to serialize staged certificate config")?;
        let config_identity = format!(
            "hard_mode={}\nmaximum_states={}\nmaximum_seconds={}\n{}",
            request.hard_mode, request.maximum_states, request.maximum_seconds, config_toml
        );
        ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        if started.elapsed() >= budget {
            deadline_reached = true;
        }
        let complete = !state_cap_reached
            && !deadline_reached
            && coverage_gaps == 0
            && duplicate_history_dates == 0
            && scanned_games == historical_games
            && unsupported_target_games == 0
            && replayed_games == historical_games
            && path_replay_failures == 0;
        Ok(StagedZeroFailureCertificateReport {
            schema_version: STAGED_ZERO_FAILURE_CERTIFICATE_SCHEMA_VERSION,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-staged-zero-failure-certificate-config-v1",
                config_identity.as_bytes(),
            ),
            code_revision,
            code_dirty,
            evaluation_from: request.from,
            evaluation_to: request.to,
            state_path_policy: "artifact_free_staged_replay".to_string(),
            certificate_scope:
                "selected root under modeled active support with no dormant root or child support; not global optimum or out-of-support outcomes".to_string(),
            hard_mode: request.hard_mode,
            maximum_states: request.maximum_states,
            maximum_seconds: request.maximum_seconds,
            scheduled_days,
            historical_games,
            coverage_gaps,
            duplicate_history_dates,
            scanned_games,
            unsupported_target_games,
            replayed_games,
            path_replay_failures,
            selected_roots,
            evaluated_roots,
            certified_roots,
            bucket_cardinality_rejections,
            dormant_support_rejections,
            unstable_modeled_support_rejections,
            missing_dictionary_rejections,
            hard_mode_rejections,
            transition_rejections,
            state_cap_reached,
            deadline_reached,
            complete,
            generation_elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        })
    }

    fn staged_artifact_free_root_guess(
        &self,
        state: &SolveState,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        hard_mode: bool,
        started: Instant,
        budget: Duration,
    ) -> Result<Option<String>> {
        if started.elapsed() >= budget {
            return Ok(None);
        }
        let cancelled = || started.elapsed() >= budget;
        let batch = match self.suggestion_batch_internal_with_search_mode_controlled(
            state,
            if hard_mode { self.guesses.len() } else { 1 },
            Some(PredictiveContext {
                hard_mode,
                as_of,
                observations,
            }),
            PredictiveBookUsage::None,
            None,
            &cancelled,
        ) {
            Ok(batch) => batch,
            Err(_error) if started.elapsed() >= budget => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(batch
            .suggestions
            .into_iter()
            .find(|suggestion| {
                !hard_mode
                    || self
                        .hard_mode_violation(observations, &suggestion.word)
                        .is_none()
            })
            .map(|suggestion| suggestion.word))
    }

    fn same_state_runtime_choice(
        &self,
        state: &SolveState,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        started: Instant,
        budget: std::time::Duration,
    ) -> Result<FiniteSearchRegretRuntimeChoice> {
        if budget
            .checked_sub(started.elapsed())
            .is_none_or(|remaining| remaining.is_zero())
        {
            return Ok(FiniteSearchRegretRuntimeChoice {
                word: None,
                value: None,
                quality: None,
                reason: "global_deadline".to_string(),
            });
        }
        let cancelled = || started.elapsed() >= budget;
        let batch = match self.suggestion_batch_internal_with_search_mode_controlled(
            state,
            1,
            Some(PredictiveContext {
                hard_mode: false,
                as_of,
                observations,
            }),
            PredictiveBookUsage::None,
            None,
            &cancelled,
        ) {
            Ok(batch) => batch,
            Err(_) if started.elapsed() >= budget => {
                return Ok(FiniteSearchRegretRuntimeChoice {
                    word: None,
                    value: None,
                    quality: None,
                    reason: "global_deadline".to_string(),
                });
            }
            Err(error) => {
                return Err(error).context("same-state dynamic regret runtime search failed");
            }
        };
        let finite_search = batch.finite_search;
        let suggestion = batch.suggestions.into_iter().next();
        let (word, value, quality) = suggestion.map_or((None, None, None), |suggestion| {
            let value = suggestion
                .finite_value
                .map(finite_regret_value)
                .filter(|value| finite_value_is_valid(*value));
            (
                Some(suggestion.word),
                value,
                suggestion.finite_value.and_then(|candidate| {
                    finite_value_is_valid(finite_regret_value(candidate))
                        .then(|| finite_quality_name(candidate.quality).to_string())
                }),
            )
        });
        let reason = if started.elapsed() >= budget {
            "global_deadline".to_string()
        } else {
            finite_search.map_or_else(
                || "not_finite".to_string(),
                |search| finite_reason_name(search.reason).to_string(),
            )
        };
        Ok(FiniteSearchRegretRuntimeChoice {
            word,
            value,
            quality,
            reason,
        })
    }

    fn same_state_dynamic_root_references(
        &self,
        state: &SolveState,
        observations: &[(String, u8)],
        horizon: u8,
        selected_roots: &[usize],
        started: Instant,
        budget: std::time::Duration,
    ) -> Result<(
        FiniteSearchRegretReference,
        Vec<FiniteSearchRegretReference>,
    )> {
        let unresolved = |status: &str| FiniteSearchRegretReference {
            status: status.to_string(),
            word: None,
            value: None,
        };
        let support_size = state
            .surviving
            .len()
            .saturating_add(state.fallback_surviving.len());
        let unresolved_status = if support_size > SAME_STATE_DYNAMIC_MAX_REFERENCE_SUPPORT {
            Some("unresolved_state_too_large")
        } else if budget
            .checked_sub(started.elapsed())
            .is_none_or(|remaining| remaining.is_zero())
        {
            Some("unresolved_global_deadline")
        } else {
            None
        };
        if let Some(status) = unresolved_status {
            return Ok((
                unresolved(status),
                selected_roots.iter().map(|_| unresolved(status)).collect(),
            ));
        }
        let remaining = budget.checked_sub(started.elapsed()).unwrap_or_default();
        let options = FiniteSearchOptions {
            root_shortlist: self.guesses.len(),
            reply_shortlist: self.guesses.len(),
            exact_state_threshold: support_size.max(1),
            budget: remaining,
            node_limit: None,
            baseline_only: false,
        };
        let cancelled = || started.elapsed() >= budget;
        let result = match self.finite_horizon_search_dynamic(
            state,
            observations,
            horizon,
            false,
            options,
            &cancelled,
        ) {
            Ok(result) => result,
            Err(_) if started.elapsed() >= budget => {
                return Ok((
                    unresolved("unresolved_global_deadline"),
                    selected_roots
                        .iter()
                        .map(|_| unresolved("unresolved_global_deadline"))
                        .collect(),
                ));
            }
            Err(error) => return Err(error),
        };
        let global = if started.elapsed() >= budget {
            unresolved("unresolved_global_deadline")
        } else {
            finite_reference_from_result(&result, self.guesses.len(), &self.guesses)
        };
        let selected = selected_roots
            .iter()
            .map(|root_index| {
                if global.status != "exact" {
                    return unresolved(&global.status);
                }
                let Some(candidate) = result
                    .candidates
                    .iter()
                    .find(|candidate| candidate.guess_index == *root_index)
                else {
                    return unresolved("incomplete_selected_root");
                };
                let value = finite_regret_value(*candidate);
                if candidate.quality != FiniteSearchQuality::Exact || !finite_value_is_valid(value)
                {
                    return unresolved("incomplete_selected_root");
                }
                FiniteSearchRegretReference {
                    status: "exact".to_string(),
                    word: self.guesses.get(*root_index).cloned(),
                    value: Some(value),
                }
            })
            .collect();
        Ok((global, selected))
    }

    fn finite_exact_options(
        &self,
        state_size: usize,
        budget: std::time::Duration,
    ) -> FiniteSearchOptions {
        FiniteSearchOptions {
            root_shortlist: self.guesses.len(),
            reply_shortlist: self.guesses.len(),
            exact_state_threshold: state_size.max(1),
            budget,
            node_limit: None,
            baseline_only: false,
        }
    }

    fn finite_legal_guess_count(&self, observations: &[(String, u8)], hard_mode: bool) -> usize {
        (0..self.guesses.len())
            .filter(|guess_index| {
                !hard_mode
                    || self
                        .hard_mode_violation(observations, &self.guesses[*guess_index])
                        .is_none()
            })
            .count()
    }

    fn finite_runtime_choice(
        &self,
        state: &SolveState,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        hard_mode: bool,
        started: Instant,
        budget: std::time::Duration,
    ) -> Result<FiniteSearchRegretRuntimeChoice> {
        let Some(remaining) = budget.checked_sub(started.elapsed()) else {
            return Ok(FiniteSearchRegretRuntimeChoice {
                word: None,
                value: None,
                quality: None,
                reason: "global_deadline".to_string(),
            });
        };
        if remaining.is_zero() {
            return Ok(FiniteSearchRegretRuntimeChoice {
                word: None,
                value: None,
                quality: None,
                reason: "global_deadline".to_string(),
            });
        }
        let mut options = self.finite_search_options();
        options.budget = options.budget.min(remaining);
        let cancelled = || started.elapsed() >= budget;
        let batch = self.finite_suggestion_batch(
            state,
            1,
            Some(PredictiveContext {
                hard_mode,
                as_of,
                observations,
            }),
            options,
            &cancelled,
        )?;
        let finite_search = batch.finite_search;
        let suggestion = batch.suggestions.into_iter().next();
        let (word, value, quality) = suggestion.map_or((None, None, None), |suggestion| {
            let value = suggestion
                .finite_value
                .map(finite_regret_value)
                .filter(|value| finite_value_is_valid(*value));
            (
                Some(suggestion.word),
                value,
                suggestion.finite_value.and_then(|candidate| {
                    finite_value_is_valid(finite_regret_value(candidate))
                        .then(|| finite_quality_name(candidate.quality).to_string())
                }),
            )
        });
        let reason = if started.elapsed() >= budget {
            "global_deadline".to_string()
        } else {
            finite_search.as_ref().map_or_else(
                || "not_finite".to_string(),
                |search| finite_reason_name(search.reason).to_string(),
            )
        };
        Ok(FiniteSearchRegretRuntimeChoice {
            word,
            value,
            quality,
            reason,
        })
    }

    fn finite_global_reference(
        &self,
        state: &SolveState,
        observations: &[(String, u8)],
        horizon: u8,
        hard_mode: bool,
        started: Instant,
        budget: std::time::Duration,
    ) -> Result<FiniteSearchRegretReference> {
        let Some(remaining) = budget.checked_sub(started.elapsed()) else {
            return Ok(FiniteSearchRegretReference {
                status: "unresolved_global_deadline".to_string(),
                word: None,
                value: None,
            });
        };
        if remaining.is_zero() {
            return Ok(FiniteSearchRegretReference {
                status: "unresolved_global_deadline".to_string(),
                word: None,
                value: None,
            });
        }
        let options = self.finite_exact_options(state.surviving.len(), remaining);
        let cancelled = || started.elapsed() >= budget;
        let result = self.finite_horizon_search(
            &state.surviving,
            &state.weights,
            observations,
            horizon,
            hard_mode,
            options,
            &cancelled,
        )?;
        if started.elapsed() >= budget {
            return Ok(FiniteSearchRegretReference {
                status: "unresolved_global_deadline".to_string(),
                word: None,
                value: None,
            });
        }
        Ok(finite_reference_from_result(
            &result,
            self.finite_legal_guess_count(observations, hard_mode),
            &self.guesses,
        ))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "explicit fixed action, posterior, history, horizon and shared audit budget"
    )]
    fn finite_fixed_root_reference(
        &self,
        state: &SolveState,
        observations: &[(String, u8)],
        horizon: u8,
        hard_mode: bool,
        root_index: usize,
        started: Instant,
        budget: std::time::Duration,
    ) -> Result<FiniteSearchRegretReference> {
        let deadline = || FiniteSearchRegretReference {
            status: "unresolved_global_deadline".to_string(),
            word: None,
            value: None,
        };
        if started.elapsed() >= budget {
            return Ok(deadline());
        }
        if root_index >= self.guesses.len() {
            return Ok(FiniteSearchRegretReference {
                status: "invalid_root".to_string(),
                word: None,
                value: None,
            });
        }
        if hard_mode
            && self
                .hard_mode_violation(observations, &self.guesses[root_index])
                .is_some()
        {
            return Ok(FiniteSearchRegretReference {
                status: "illegal_root".to_string(),
                word: None,
                value: None,
            });
        }
        let mut total_weight = 0.0;
        let mut partitions = (0..PATTERN_SPACE)
            .map(|_| Vec::new())
            .collect::<Vec<Vec<usize>>>();
        for &answer_index in &state.surviving {
            if started.elapsed() >= budget {
                return Ok(deadline());
            }
            let weight = state.weights.get(answer_index).copied().ok_or_else(|| {
                anyhow!("finite reference answer index {answer_index} is out of range")
            })?;
            if !weight.is_finite() || weight < 0.0 {
                bail!("finite reference weights must be finite and non-negative");
            }
            total_weight += weight;
            let answer = self.answers.get(answer_index).ok_or_else(|| {
                anyhow!("finite reference answer index {answer_index} is out of range")
            })?;
            let pattern = score_guess(&self.guesses[root_index], &answer.word) as usize;
            partitions[pattern].push(answer_index);
        }
        if !total_weight.is_finite() || total_weight <= 0.0 {
            bail!("finite reference requires positive answer mass");
        }

        let mut failure_probability = 0.0;
        let mut expected_attempts = 1.0;
        for (pattern, child_subset) in partitions.into_iter().enumerate() {
            if started.elapsed() >= budget {
                return Ok(deadline());
            }
            if child_subset.is_empty() {
                continue;
            }
            let mass = child_subset
                .iter()
                .map(|answer_index| state.weights[*answer_index])
                .sum::<f64>();
            if mass <= 0.0 {
                continue;
            }
            let probability = mass / total_weight;
            if pattern == ALL_GREEN_PATTERN as usize {
                continue;
            }
            if horizon <= 1 {
                failure_probability += probability;
                continue;
            }
            let Some(remaining) = budget.checked_sub(started.elapsed()) else {
                return Ok(FiniteSearchRegretReference {
                    status: "unresolved_global_deadline".to_string(),
                    word: None,
                    value: None,
                });
            };
            if remaining.is_zero() {
                return Ok(FiniteSearchRegretReference {
                    status: "unresolved_global_deadline".to_string(),
                    word: None,
                    value: None,
                });
            }
            let child_observations = if hard_mode {
                let mut next = observations.to_vec();
                next.push((self.guesses[root_index].clone(), pattern as u8));
                next
            } else {
                Vec::new()
            };
            let options = self.finite_exact_options(child_subset.len(), remaining);
            let cancelled = || started.elapsed() >= budget;
            let result = self.finite_horizon_search(
                &child_subset,
                &state.weights,
                &child_observations,
                horizon - 1,
                hard_mode,
                options,
                &cancelled,
            )?;
            if started.elapsed() >= budget {
                return Ok(FiniteSearchRegretReference {
                    status: "unresolved_global_deadline".to_string(),
                    word: None,
                    value: None,
                });
            }
            let child = finite_reference_from_result(
                &result,
                self.finite_legal_guess_count(&child_observations, hard_mode),
                &self.guesses,
            );
            let Some(child_value) = child.value else {
                return Ok(FiniteSearchRegretReference {
                    status: format!("child_{}", child.status),
                    word: None,
                    value: None,
                });
            };
            if child.status != "exact" {
                return Ok(FiniteSearchRegretReference {
                    status: format!("child_{}", child.status),
                    word: None,
                    value: None,
                });
            }
            failure_probability += probability * child_value.failure_probability;
            expected_attempts += probability * child_value.expected_attempts;
        }
        let value = FiniteSearchRegretValue {
            failure_probability: failure_probability.clamp(0.0, 1.0),
            expected_attempts: expected_attempts.max(0.0),
        };
        if !finite_value_is_valid(value) {
            bail!("finite fixed-root reference produced an invalid value");
        }
        if started.elapsed() >= budget {
            return Ok(deadline());
        }
        Ok(FiniteSearchRegretReference {
            status: "exact".to_string(),
            word: Some(self.guesses[root_index].clone()),
            value: Some(value),
        })
    }

    fn finite_search_regret_state(
        &self,
        candidate: &SearchRegretCandidateState,
        state: &SolveState,
        hard_mode: bool,
        started: Instant,
        budget: std::time::Duration,
    ) -> Result<FiniteSearchRegretState> {
        let horizon = 6usize
            .checked_sub(candidate.observations.len())
            .ok_or_else(|| anyhow!("finite search-regret state exceeds six turns"))?
            as u8;
        if horizon == 0 {
            bail!("finite search-regret cannot audit a terminal state");
        }
        let as_of = candidate
            .date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
        let runtime = self.finite_runtime_choice(
            state,
            as_of,
            &candidate.observations,
            hard_mode,
            started,
            budget,
        )?;
        let exact_fixed_root = runtime
            .word
            .as_deref()
            .and_then(|word| self.guess_index.get(word).copied())
            .map_or(
                Ok(FiniteSearchRegretReference {
                    status: "unresolved_runtime_choice".to_string(),
                    word: None,
                    value: None,
                }),
                |root_index| {
                    self.finite_fixed_root_reference(
                        state,
                        &candidate.observations,
                        horizon,
                        hard_mode,
                        root_index,
                        started,
                        budget,
                    )
                },
            )?;
        let global_optimum = self.finite_global_reference(
            state,
            &candidate.observations,
            horizon,
            hard_mode,
            started,
            budget,
        )?;
        let (failure_regret, attempts_regret, matches_optimum) =
            finite_regrets(&exact_fixed_root, &global_optimum)?;
        Ok(FiniteSearchRegretState {
            date: candidate.date,
            target: candidate.target.clone(),
            turn: candidate.turn,
            horizon,
            surviving_answers: candidate.surviving_answers,
            hard_mode,
            observations: candidate
                .observations
                .iter()
                .map(|(guess, feedback)| SearchRegretObservation {
                    guess: guess.clone(),
                    feedback: format_feedback_letters(*feedback),
                })
                .collect(),
            production_regime: self.config.search_policy_mode.label().to_string(),
            runtime,
            exact_fixed_root,
            global_optimum,
            failure_regret,
            attempts_regret,
            matches_optimum,
        })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "state choices and shared cancellation remain explicit"
    )]
    fn audit_search_regret_state(
        &self,
        candidate: &SearchRegretCandidateState,
        state: &SolveState,
        production_guess: Option<&str>,
        production_regime: PredictiveRegime,
        proxy_guess: &str,
        lookahead_guess: &str,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<SearchRegretState> {
        check_predictive_search_cancelled(cancelled)?;
        let production_index = production_guess
            .map(|guess| {
                self.guess_index
                    .get(guess)
                    .copied()
                    .with_context(|| format!("unknown production guess {guess}"))
            })
            .transpose()?;
        let proxy_index = self
            .guess_index
            .get(proxy_guess)
            .copied()
            .with_context(|| format!("unknown proxy guess {proxy_guess}"))?;
        let lookahead_index = self
            .guess_index
            .get(lookahead_guess)
            .copied()
            .with_context(|| format!("unknown lookahead guess {lookahead_guess}"))?;

        let selected = production_index
            .into_iter()
            .chain([proxy_index, lookahead_index])
            .collect::<Vec<_>>();
        let mut memo = PredictiveMemoMap::default();
        let mut scratch = ExactSearchScratch::new();
        let lower_bound = weighted_exact_lower_bound(&state.surviving, &state.weights)?;
        let mut optimal_index = proxy_index;
        let mut optimal_cost = f64::INFINITY;
        for guess_index in 0..self.guesses.len() {
            let cost = self.exact_cost_for_guess_controlled(
                guess_index,
                ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    memo: &mut memo,
                    best_bound: optimal_cost,
                    scratch: &mut scratch,
                    depth: 0,
                },
                cancelled,
            )?;
            if cost.total_cmp(&optimal_cost).is_lt() {
                optimal_index = guess_index;
                optimal_cost = cost;
                if optimal_cost <= lower_bound + 1e-12 {
                    break;
                }
            }
        }
        if !optimal_cost.is_finite() {
            bail!(
                "search-regret found no exhaustive optimum for {} turn {}",
                candidate.date,
                candidate.turn
            );
        }

        let mut selected_costs = HashMap::new();
        for guess_index in selected.iter().copied() {
            if selected_costs.contains_key(&guess_index) {
                continue;
            }
            let cost = self.exact_cost_for_guess_controlled(
                guess_index,
                ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    memo: &mut memo,
                    best_bound: f64::INFINITY,
                    scratch: &mut scratch,
                    depth: 0,
                },
                cancelled,
            )?;
            if !cost.is_finite() {
                bail!(
                    "search-regret selected non-progressing guess {} for {} turn {}",
                    self.guesses[guess_index],
                    candidate.date,
                    candidate.turn
                );
            }
            selected_costs.insert(guess_index, cost);
        }

        let choice = |guess_index: usize| {
            let exact_cost = selected_costs[&guess_index];
            let regret = (exact_cost - optimal_cost).max(0.0);
            SearchRegretChoice {
                word: self.guesses[guess_index].clone(),
                exact_cost,
                regret,
                matches_optimum: regret <= 1e-9,
            }
        };
        check_predictive_search_cancelled(cancelled)?;
        Ok(SearchRegretState {
            date: candidate.date,
            target: candidate.target.clone(),
            turn: candidate.turn,
            surviving_answers: candidate.surviving_answers,
            observations: candidate
                .observations
                .iter()
                .map(|(guess, feedback)| SearchRegretObservation {
                    guess: guess.clone(),
                    feedback: format_feedback_letters(*feedback),
                })
                .collect(),
            production_regime: production_regime.label().to_string(),
            optimal_word: self.guesses[optimal_index].clone(),
            optimal_exact_cost: optimal_cost,
            production: production_index.map_or_else(
                || SearchRegretChoice {
                    word: self.guesses[optimal_index].clone(),
                    exact_cost: optimal_cost,
                    regret: 0.0,
                    matches_optimum: true,
                },
                choice,
            ),
            proxy: choice(proxy_index),
            lookahead: choice(lookahead_index),
        })
    }

    pub(super) fn backtest_detailed_with_book_usage(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedBacktestReport> {
        self.backtest_detailed_with_book_usage_and_progress(from, to, top, book_usage, None)
    }

    fn backtest_detailed_with_book_usage_and_progress(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<DetailedBacktestReport> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested backtest range");
        }

        self.backtest_selected_games_with_progress(&games, top, book_usage, progress)
    }

    #[cfg(test)]
    pub(super) fn backtest_selected_games(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedBacktestReport> {
        self.backtest_selected_games_with_progress(games, top, book_usage, None)
    }

    fn recovery_target_games(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<&NytDailyEntry>> {
        let mut games = Vec::new();
        for entry in self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
        {
            check_predictive_search_cancelled(cancelled)?;
            let as_of = crate::predictive::history_cutoff(entry.print_date)?;
            let target = entry.solution.to_ascii_lowercase();
            let state = self.initial_state(as_of);
            if !state
                .surviving
                .iter()
                .any(|index| self.answers[*index].word == target)
                && state
                    .fallback_surviving
                    .iter()
                    .any(|index| self.answers[*index].word == target)
            {
                games.push(entry);
            }
        }
        Ok(games)
    }

    pub fn compare_survival_model_on_games(
        &self,
        games: &[NytDailyEntry],
        model: &crate::predictive::survival::SurvivalModel,
        top: usize,
    ) -> Result<SurvivalSolveComparison> {
        if games.is_empty() {
            bail!("survival solve comparison requires at least one game");
        }
        model.validate()?;
        let baseline = self.evaluate_solve_policy(games, "logistic", |entry| {
            let started = Instant::now();
            let (outcome, _) = self.solve_backtest_entry_proxy(entry, top)?;
            Ok((outcome, started.elapsed().as_secs_f64() * 1_000.0))
        })?;
        let survival = self.evaluate_solve_policy(games, "survival", |entry| {
            let started = Instant::now();
            let (outcome, _) = self.solve_backtest_entry_with_survival(entry, model, top)?;
            Ok((outcome, started.elapsed().as_secs_f64() * 1_000.0))
        })?;
        Ok(SurvivalSolveComparison { baseline, survival })
    }

    fn evaluate_solve_policy<F>(
        &self,
        games: &[NytDailyEntry],
        label: &str,
        evaluate: F,
    ) -> Result<SolvePolicyEvidence>
    where
        F: Fn(&NytDailyEntry) -> Result<(GameOutcome, f64)> + Sync,
    {
        let started = Instant::now();
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let total = games.len();
        let evaluated = games
            .par_iter()
            .map(|entry| {
                let result = evaluate(entry);
                let current = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if current == 1 || current.is_multiple_of(10) || current == total {
                    let elapsed = started.elapsed().as_secs_f64();
                    let eta = if current == 0 {
                        0.0
                    } else {
                        elapsed * (total - current) as f64 / current as f64
                    };
                    eprintln!(
                        "survival phase=solve policy={} games={}/{} elapsed_s={:.1} eta_s={:.1}",
                        label, current, total, elapsed, eta
                    );
                    let _ = std::io::stderr().flush();
                }
                result
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let (outcomes, mut latencies_ms) = evaluated.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        latencies_ms.sort_by(f64::total_cmp);
        let p95_index = ((latencies_ms.len() as f64 * 0.95).ceil() as usize)
            .saturating_sub(1)
            .min(latencies_ms.len().saturating_sub(1));
        let canonical = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let failure_rate_ci95 = (
            1.0 - canonical.solve_rate_ci95.upper,
            1.0 - canonical.solve_rate_ci95.lower,
        );
        Ok(SolvePolicyEvidence {
            summary: BacktestStats {
                games: canonical.scheduled_games,
                average_guesses: canonical.conditional_mean_guesses,
                p95_guesses: canonical.p95_guesses,
                max_guesses: canonical.max_guesses,
                failures: canonical.unsolved_games + canonical.coverage_gaps,
                coverage_gaps: canonical.coverage_gaps,
                average_guesses_ci95: canonical
                    .conditional_mean_guesses_ci95
                    .map(|interval| (interval.lower, interval.upper)),
                failure_rate_ci95,
                canonical,
            },
            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            latency_p95_ms: latencies_ms[p95_index],
            peak_memory_bytes: crate::process_memory::process_memory_snapshot()
                .map(|snapshot| snapshot.peak_working_set_bytes),
        })
    }

    fn solve_backtest_entry_with_survival(
        &self,
        entry: &NytDailyEntry,
        model: &crate::predictive::survival::SurvivalModel,
        top: usize,
    ) -> Result<(GameOutcome, DetailedSolveRun)> {
        let as_of = entry
            .print_date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let modeled_weights = self
            .answers
            .iter()
            .take(self.primary_answer_count)
            .map(|answer| {
                let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
                let recency_weight = match snapshot.last_seen {
                    Some(last_seen) => (1.0
                        - model.try_predict_interval(last_seen, as_of)?.survival)
                        .max(self.config.cooldown_floor),
                    None => 1.0,
                };
                Ok(snapshot.base_weight * recency_weight * snapshot.manual_weight)
            })
            .collect::<Result<Vec<_>>>()?;
        let state = self.initial_state_with_modeled_weights(as_of, Some(&modeled_weights))?;
        let run = self.solve_target_from_initial_state_detailed(
            &entry.solution,
            as_of,
            entry.print_date,
            top,
            state,
            SolveExecutionPolicy {
                cancelled: &|| false,
                book_usage: PredictiveBookUsage::None,
                search_mode: Some(PredictiveSearchMode::ProxyOnly),
                forced: &[],
            },
        )?;
        let outcome = if run.steps.is_empty() {
            GameOutcome::coverage_gap(entry.print_date)
        } else if run.solved {
            GameOutcome::solved(entry.print_date, run.steps.len())
        } else {
            GameOutcome::unsolved(entry.print_date, run.steps.len())
        };
        Ok((outcome, run))
    }

    fn solve_backtest_entry_proxy(
        &self,
        entry: &NytDailyEntry,
        top: usize,
    ) -> Result<(GameOutcome, DetailedSolveRun)> {
        let as_of = entry
            .print_date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let state = self.initial_state(as_of);
        let run = self.solve_target_from_initial_state_detailed(
            &entry.solution,
            as_of,
            entry.print_date,
            top,
            state,
            SolveExecutionPolicy {
                cancelled: &|| false,
                book_usage: PredictiveBookUsage::None,
                search_mode: Some(PredictiveSearchMode::ProxyOnly),
                forced: &[],
            },
        )?;
        let outcome = if run.steps.is_empty() {
            GameOutcome::coverage_gap(entry.print_date)
        } else if run.solved {
            GameOutcome::solved(entry.print_date, run.steps.len())
        } else {
            GameOutcome::unsolved(entry.print_date, run.steps.len())
        };
        Ok((outcome, run))
    }

    pub(super) fn backtest_selected_games_with_progress(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<DetailedBacktestReport> {
        self.backtest_selected_games_controlled(games, top, book_usage, progress, &|| false)
    }

    fn backtest_selected_games_controlled(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<DetailedBacktestReport> {
        check_predictive_search_cancelled(cancelled)?;
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let total = games.len();
        let evaluate = |entry: &&NytDailyEntry| {
            let result = self.solve_backtest_entry_controlled(entry, top, book_usage, cancelled);
            let current = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if let Some(progress) = progress {
                progress(current, total);
            }
            result
        };
        // Wall-clock-limited policies must not compete with other games for
        // their search budget. Unlimited legacy policies retain parallelism.
        let bounded_search = self.config.search_policy_mode.is_finite();
        let evaluated = if bounded_search {
            games.iter().map(evaluate).collect::<Vec<_>>()
        } else {
            games.par_iter().map(evaluate).collect::<Vec<_>>()
        }
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        // Both iterator paths preserve the canonical chronological input order.
        let (outcomes, runs) = evaluated.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        if bounded_search {
            for run in &runs {
                validate_finite_run_trace(run, run.steps.len())?;
            }
        }

        let canonical = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let failure_rate_ci95 = (
            1.0 - canonical.solve_rate_ci95.upper,
            1.0 - canonical.solve_rate_ci95.lower,
        );

        Ok(DetailedBacktestReport {
            summary: BacktestStats {
                games: canonical.scheduled_games,
                average_guesses: canonical.conditional_mean_guesses,
                p95_guesses: canonical.p95_guesses,
                max_guesses: canonical.max_guesses,
                failures: canonical.unsolved_games + canonical.coverage_gaps,
                coverage_gaps: canonical.coverage_gaps,
                average_guesses_ci95: canonical
                    .conditional_mean_guesses_ci95
                    .map(|interval| (interval.lower, interval.upper)),
                failure_rate_ci95,
                canonical,
            },
            runs,
        })
    }

    #[cfg(test)]
    pub(super) fn solve_backtest_entry(
        &self,
        entry: &NytDailyEntry,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<(GameOutcome, DetailedSolveRun)> {
        self.solve_backtest_entry_controlled(entry, top, book_usage, &|| false)
    }

    fn solve_backtest_entry_controlled(
        &self,
        entry: &NytDailyEntry,
        top: usize,
        book_usage: PredictiveBookUsage,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<(GameOutcome, DetailedSolveRun)> {
        check_predictive_search_cancelled(cancelled)?;
        let as_of = entry
            .print_date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let run = self.solve_target_from_initial_state_detailed(
            &entry.solution,
            as_of,
            entry.print_date,
            top,
            self.initial_state(as_of),
            SolveExecutionPolicy {
                book_usage,
                search_mode: None,
                forced: &[],
                cancelled,
            },
        )?;
        let outcome = if run.steps.is_empty() {
            GameOutcome::coverage_gap(entry.print_date)
        } else if run.solved {
            GameOutcome::solved(entry.print_date, run.steps.len())
        } else {
            GameOutcome::unsolved(entry.print_date, run.steps.len())
        };
        Ok((outcome, run))
    }

    pub fn hard_case_report(&self, top: usize) -> Result<HardCaseReport> {
        self.hard_case_report_with_book_usage(Self::today(), top, PredictiveBookUsage::DiskOnly)
    }

    pub(super) fn hard_case_report_with_book_usage(
        &self,
        puzzle_date: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<HardCaseReport> {
        let as_of = crate::predictive::history_cutoff(puzzle_date)?;
        let hard_case_spec = default_diagnostic_suite()?.hard_cases;
        let cases = self.select_hard_case_targets(as_of, top, &hard_case_spec)?;
        let mut results = Vec::new();
        let mut failures = 0usize;
        let mut guess_total = 0usize;

        for (label, target) in cases {
            let run = self.solve_target_from_state_detailed(
                &target,
                as_of,
                puzzle_date,
                top,
                book_usage,
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
        self.experiment_report_with_book_usage(from, to, top, PredictiveBookUsage::DiskOnly)
    }

    fn experiment_report_with_book_usage(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<ExperimentResult> {
        self.experiment_report_with_book_usage_and_progress(from, to, top, book_usage, None)
    }

    fn experiment_report_with_book_usage_and_progress(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<ExperimentResult> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        self.experiment_report_for_selected_games_with_book_usage_and_progress(
            &games, top, book_usage, progress,
        )
    }

    fn experiment_report_for_selected_games_with_book_usage_and_progress(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<ExperimentResult> {
        let mut timing = NoopEvidenceTiming;
        self.experiment_report_for_selected_games_with_book_usage_and_progress_and_timing(
            games,
            top,
            book_usage,
            EvaluationControl {
                progress,
                cancelled: &|| false,
            },
            None,
            &mut timing,
        )
    }

    fn experiment_report_for_selected_games_with_book_usage_and_progress_and_timing(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
        control: EvaluationControl<'_>,
        timing_profile: Option<&str>,
        timing: &mut dyn EvidenceTimingSink,
    ) -> Result<ExperimentResult> {
        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let simulation_started = Instant::now();
        let detailed = self.backtest_selected_games_controlled(
            games,
            top,
            book_usage,
            control.progress,
            control.cancelled,
        )?;
        record_evidence_timing(
            timing,
            timing_profile,
            "game_simulation",
            simulation_started.elapsed(),
        );
        let report_started = Instant::now();
        let backtest = detailed.summary.clone();
        let (
            proxy_step_pct,
            lookahead_step_pct,
            escalated_exact_step_pct,
            exact_step_pct,
            finite_step_pct,
            terminal_step_pct,
        ) = Self::regime_mix(&detailed.runs);
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
        let mut prior_observations = Vec::new();

        for entry in games {
            check_predictive_search_cancelled(control.cancelled)?;
            if let Some(metrics) = self.initial_prior_metrics(&entry.solution, entry.print_date) {
                total_log_loss += metrics.log_loss;
                total_brier += metrics.brier;
                total_target_probability += metrics.target_probability;
                total_rank += metrics.target_rank as f64;
                measured += 1;
                prior_observations.push(RankedProbabilityObservation {
                    target_rank: metrics.target_rank,
                    top_probability: metrics.top_probability,
                    top_prediction_correct: metrics.top_prediction_correct,
                });
            }
        }

        let game_results = detailed
            .runs
            .iter()
            .map(|run| {
                let (prior_strata, posterior_calibration) =
                    self.posterior_calibration_for_run(run)?;
                Ok(ExperimentGameResult {
                    target: run.target.clone(),
                    outcome: if run.steps.is_empty() {
                        GameOutcome::coverage_gap(run.date)
                    } else if run.solved {
                        GameOutcome::solved(run.date, run.steps.len())
                    } else {
                        GameOutcome::unsolved(run.date, run.steps.len())
                    },
                    path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                    finite_search_steps: run
                        .steps
                        .iter()
                        .filter_map(|step| step.finite_search.clone())
                        .collect(),
                    prior_strata,
                    posterior_calibration,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let posterior_calibration = summarize_posterior_calibration(&game_results);
        validate_posterior_calibration_evidence(
            &game_results,
            &posterior_calibration,
            true,
            "experiment result",
        )?;

        let outcomes = detailed
            .runs
            .iter()
            .map(|run| {
                if run.steps.is_empty() {
                    GameOutcome::coverage_gap(run.date)
                } else if run.solved {
                    GameOutcome::solved(run.date, run.steps.len())
                } else {
                    GameOutcome::unsolved(run.date, run.steps.len())
                }
            })
            .collect::<Vec<_>>();
        let failure_penalty_sensitivity = [6.0, 7.0, 8.0]
            .into_iter()
            .map(|penalty_guesses| {
                let metrics = summarize_predictive_outcomes(
                    &outcomes,
                    penalty_guesses,
                    BootstrapConfig::default(),
                )?;
                Ok(FailurePenaltyEvidence {
                    penalty_guesses,
                    all_game_mean_guesses: metrics.all_game_penalized_mean_guesses,
                    ci95: metrics.all_game_penalized_mean_guesses_ci95,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let evaluation_to = games
            .iter()
            .map(|entry| entry.print_date)
            .max()
            .expect("non-empty selected games");
        let fallback_as_of = evaluation_to
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("session-fallback benchmark cutoff underflowed"))?;
        let (session_fallback_cold_ms, session_fallback_warm_ms) = if book_usage
            == PredictiveBookUsage::Full
            && !self.config.search_policy_mode.is_finite()
        {
            let (cold, warm) =
                self.benchmark_session_fallback_latency(fallback_as_of, control.cancelled)?;
            (Some(cold), Some(warm))
        } else {
            (None, None)
        };
        let result = ExperimentResult {
            config_id: format!(
                "{}-et{}-ee{}-cp{}-lt{}-lc{}-lr{}-ls{}",
                self.config.search_policy_mode.label(),
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
            average_log_loss: (measured > 0).then(|| total_log_loss / measured as f64),
            average_brier: (measured > 0).then(|| total_brier / measured as f64),
            average_target_probability: (measured > 0)
                .then(|| total_target_probability / measured as f64),
            average_target_rank: (measured > 0).then(|| total_rank / measured as f64),
            prior_evidence: (!prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            posterior_calibration,
            execution: Self::execution_telemetry(&detailed.runs),
            failure_penalty_sensitivity,
            latency_p95_ms: self.benchmark_predictive_latency_controlled(
                evaluation_to,
                default_diagnostic_suite()?.latency.evidence_runs,
                control.cancelled,
            )?,
            session_fallback_cold_ms,
            session_fallback_warm_ms,
            proxy_step_pct,
            lookahead_step_pct,
            escalated_exact_step_pct,
            exact_step_pct,
            finite_step_pct,
            terminal_step_pct,
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
            games: game_results,
        };
        record_evidence_timing(
            timing,
            timing_profile,
            "report_assembly",
            report_started.elapsed(),
        );
        Ok(result)
    }

    pub fn build_development_evidence(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<PredictiveEvidenceArtifact> {
        Self::build_development_evidence_with_budget(
            paths,
            config,
            from,
            to,
            top,
            EvidenceResourceBudget::default(),
        )
    }

    pub fn build_development_evidence_with_budget(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        resource_budget: EvidenceResourceBudget,
    ) -> Result<PredictiveEvidenceArtifact> {
        Self::build_development_evidence_with_checkpoint(
            paths,
            config,
            from,
            to,
            top,
            resource_budget,
            None,
        )
    }

    pub fn build_development_evidence_with_checkpoint(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        resource_budget: EvidenceResourceBudget,
        checkpoint_path: Option<&Path>,
    ) -> Result<PredictiveEvidenceArtifact> {
        if from > to {
            bail!("evidence start date cannot be after end date");
        }
        Self::build_development_evidence_with_selection(
            paths,
            config,
            EvidenceDateSelection::Range(DateRange::new(from, to)?),
            top,
            resource_budget,
            None,
            checkpoint_path,
        )
    }

    pub fn build_development_evidence_with_selection(
        paths: &ProjectPaths,
        config: &PriorConfig,
        selection: EvidenceDateSelection,
        top: usize,
        resource_budget: EvidenceResourceBudget,
        matrix_path: Option<&Path>,
        checkpoint_path: Option<&Path>,
    ) -> Result<PredictiveEvidenceArtifact> {
        if resource_budget.maximum_seconds == 0 || resource_budget.maximum_memory_mb == 0 {
            bail!("evidence time and memory budgets must be positive");
        }
        let mut timing = StderrEvidenceTiming::from_env();
        let rayon_threads = rayon::current_num_threads();
        let generation_started = Instant::now();
        let (plan, selected_ranges, selection_label, from, to) = match selection {
            EvidenceDateSelection::Range(range) => {
                let plan = ensure_development_target_range(
                    paths,
                    range.start,
                    range.end,
                    "evidence generation",
                )?;
                (plan, vec![range], "range", range.start, range.end)
            }
            EvidenceDateSelection::RollingFolds => {
                let plan = canonical_development_evaluation_plan(paths, "evidence generation")?;
                let selected_ranges = plan
                    .folds
                    .iter()
                    .map(|fold| fold.validation)
                    .collect::<Vec<_>>();
                let first = selected_ranges
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("canonical development plan has no validation folds"))?;
                let last = selected_ranges
                    .last()
                    .copied()
                    .expect("non-empty selected ranges");
                (
                    plan,
                    selected_ranges,
                    "rolling_folds",
                    first.start,
                    last.end,
                )
            }
        };
        let input_fingerprint = development_source_identity(paths, plan.development.end)?;

        let (matrix, matrix_source, matrix_fingerprint) = load_evidence_matrix(paths, matrix_path)?;
        let profile_ids = matrix
            .profiles
            .iter()
            .map(|profile| profile.id.clone())
            .collect::<Vec<_>>();
        let resolved_profile_base_configs =
            resolved_evidence_profile_base_configs(paths, config, &matrix)?;
        let config_toml =
            toml::to_string_pretty(config).context("failed to serialize evidence config")?;
        let checkpoint_identity = evidence_checkpoint_identity(
            &input_fingerprint,
            &config_toml,
            &plan,
            selection_label,
            &selected_ranges,
            &matrix_source,
            &matrix_fingerprint,
            &profile_ids,
            &resolved_profile_base_configs,
            DateRange::new(from, to)?,
            top,
            resource_budget,
            rayon_threads,
        )?;
        let checkpoint_path = checkpoint_path.map(|path| {
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                paths.root.join(path)
            }
        });
        let mut prior_elapsed_ms = 0_u64;
        let mut prior_peak_working_set_bytes = 0_u64;
        let mut baselines = Vec::new();
        if let Some(path) = checkpoint_path.as_deref().filter(|path| path.exists()) {
            let raw = fs::read(path)
                .with_context(|| format!("read evidence checkpoint {}", path.display()))?;
            let checkpoint: EvidenceMatrixCheckpoint = serde_json::from_slice(&raw)
                .with_context(|| format!("parse evidence checkpoint {}", path.display()))?;
            checkpoint.validate(
                &checkpoint_identity,
                &profile_ids,
                resource_budget,
                rayon_threads,
            )?;
            prior_elapsed_ms = checkpoint.elapsed_ms;
            prior_peak_working_set_bytes = checkpoint.peak_working_set_bytes;
            baselines = checkpoint.baselines;
            eprintln!(
                "benchmark-evidence phase=resume profiles={}/{} prior_elapsed_s={:.1} checkpoint={}",
                baselines.len(),
                profile_ids.len(),
                prior_elapsed_ms as f64 / 1_000.0,
                path.display()
            );
        }
        enforce_evidence_resource_budget(
            generation_started,
            prior_elapsed_ms,
            prior_peak_working_set_bytes,
            resource_budget,
        )?;
        let inner_budget = Mutex::new(super::exhaustive_teacher::WorkBudget::new(
            generation_started,
            prior_elapsed_ms,
            Duration::from_secs(resource_budget.maximum_seconds),
            Some(
                resource_budget
                    .maximum_memory_mb
                    .saturating_mul(1024 * 1024),
            ),
        ));
        let cancelled = || {
            inner_budget
                .lock()
                .expect("evidence budget")
                .check()
                .is_err()
        };
        let total_profiles = matrix.profiles.len();
        let completed_games = std::sync::atomic::AtomicUsize::new(
            baselines
                .iter()
                .map(|baseline| baseline.result.backtest.canonical.scheduled_games)
                .sum(),
        );
        baselines.reserve(total_profiles.saturating_sub(baselines.len()));
        eprintln!(
            "benchmark-evidence phase=start profiles={} rayon_threads={} from={} to={} elapsed_s=0.0",
            total_profiles, rayon_threads, from, to,
        );
        let _ = std::io::stderr().flush();
        for (profile_index, profile) in matrix.profiles.iter().cloned().enumerate() {
            if profile_index < baselines.len() {
                continue;
            }
            let profile_id = profile.id.clone();
            eprintln!(
                "benchmark-evidence phase=profile-start profile={}/{} id={} elapsed_s={:.1}",
                profile_index + 1,
                total_profiles,
                profile_id,
                generation_started.elapsed().as_secs_f64(),
            );
            let _ = std::io::stderr().flush();
            let profile_started = Instant::now();
            let solver_started = Instant::now();
            let profile_base = profile.load_base_config(&paths.root, config)?;
            let profile_config =
                profile.apply(&predictive_parameter_registry(&profile_base), &profile_base)?;
            let book_usage = match profile.artifact_mode {
                ExperimentArtifactMode::Disabled => PredictiveBookUsage::None,
                ExperimentArtifactMode::DiskOnly => PredictiveBookUsage::DiskOnly,
            };
            let solver = Self::from_paths_with_settings(
                paths,
                &profile_config,
                profile.weight_mode,
                profile.model_variant,
            )?;
            record_evidence_timing(
                &mut timing,
                Some(&profile_id),
                "solver_setup",
                solver_started.elapsed(),
            );
            let effective_config_toml = toml::to_string_pretty(&profile_config)
                .context("failed to serialize evidence profile config")?;
            let progress = |profile_completed: usize, profile_total: usize| {
                let total_games = profile_total.saturating_mul(total_profiles);
                let global_completed =
                    completed_games.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let elapsed = cumulative_evidence_elapsed_ms(generation_started, prior_elapsed_ms)
                    as f64
                    / 1_000.0;
                let eta = if global_completed == 0 {
                    0.0
                } else {
                    elapsed / global_completed as f64
                        * total_games.saturating_sub(global_completed) as f64
                };
                eprintln!(
                    "benchmark-evidence phase=games id={} profile_games={}/{} total_games={}/{} elapsed_s={:.1} eta_s={:.1}",
                    profile_id,
                    profile_completed,
                    profile_total,
                    global_completed,
                    total_games,
                    elapsed,
                    eta,
                );
                let _ = std::io::stderr().flush();
            };
            let game_selection_started = Instant::now();
            let games = selected_evidence_games(&solver.history_dates, &selected_ranges)?;
            record_evidence_timing(
                &mut timing,
                Some(&profile_id),
                "game_selection",
                game_selection_started.elapsed(),
            );
            let result = solver
                .experiment_report_for_selected_games_with_book_usage_and_progress_and_timing(
                    &games,
                    top,
                    book_usage,
                    EvaluationControl {
                        progress: Some(&progress),
                        cancelled: &cancelled,
                    },
                    Some(&profile_id),
                    &mut timing,
                );
            if let Err(error) = result {
                if let Some(path) = checkpoint_path.as_deref() {
                    // Charge interrupted work on resume, retaining only complete profiles.
                    let checkpoint = EvidenceMatrixCheckpoint {
                        schema_version: EVIDENCE_CHECKPOINT_SCHEMA_VERSION,
                        identity: checkpoint_identity.clone(),
                        resource_budget: Some(resource_budget),
                        rayon_threads: Some(rayon_threads),
                        elapsed_ms: cumulative_evidence_elapsed_ms(
                            generation_started,
                            prior_elapsed_ms,
                        ),
                        peak_working_set_bytes: inner_budget
                            .lock()
                            .expect("evidence budget")
                            .peak_memory_bytes()
                            .unwrap_or(prior_peak_working_set_bytes)
                            .max(prior_peak_working_set_bytes),
                        baselines: baselines.clone(),
                    };
                    checkpoint.validate(
                        &checkpoint_identity,
                        &profile_ids,
                        resource_budget,
                        rayon_threads,
                    )?;
                    crate::atomic_file::atomic_write(
                        path,
                        &serde_json::to_vec_pretty(&checkpoint)?,
                    )?;
                }
                inner_budget.lock().expect("evidence budget").check()?;
                return Err(error);
            }
            let result = result?;
            eprintln!(
                "benchmark-evidence phase=profile-complete profile={}/{} id={} solved={}/{} failures={} mean_guesses={:.4} elapsed_s={:.1}",
                profile_index + 1,
                total_profiles,
                profile_id,
                result.backtest.canonical.solved_games,
                result.backtest.canonical.scheduled_games,
                result.backtest.failures,
                result.backtest.canonical.all_game_penalized_mean_guesses,
                generation_started.elapsed().as_secs_f64(),
            );
            let _ = std::io::stderr().flush();
            baselines.push(EvidenceBaseline {
                id: profile.id,
                description: profile.description,
                artifacts: match book_usage {
                    PredictiveBookUsage::None => "disabled",
                    PredictiveBookUsage::DiskOnly => "valid_disk_only",
                    PredictiveBookUsage::Full => "disk_then_live",
                }
                .to_string(),
                config_fingerprint: crate::identity::digest_bytes_tagged(
                    "maybe-wordle-benchmark-config-v1",
                    effective_config_toml.as_bytes(),
                ),
                effective_config_toml,
                paired_vs_selected_default: None,
                result,
            });
            let validation_started = Instant::now();
            let memory = crate::process_memory::process_memory_snapshot()
                .ok_or_else(|| anyhow!("evidence process memory measurement unavailable"))?;
            prior_peak_working_set_bytes =
                prior_peak_working_set_bytes.max(memory.peak_working_set_bytes);
            ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
            let (current_matrix, current_matrix_source, current_matrix_fingerprint) =
                load_evidence_matrix(paths, matrix_path)?;
            let current_profile_ids = current_matrix
                .profiles
                .iter()
                .map(|profile| profile.id.clone())
                .collect::<Vec<_>>();
            let current_profile_base_configs =
                resolved_evidence_profile_base_configs(paths, config, &current_matrix)?;
            let current_identity = evidence_checkpoint_identity(
                &input_fingerprint,
                &config_toml,
                &plan,
                selection_label,
                &selected_ranges,
                &current_matrix_source,
                &current_matrix_fingerprint,
                &current_profile_ids,
                &current_profile_base_configs,
                DateRange::new(from, to)?,
                top,
                resource_budget,
                rayon_threads,
            )?;
            if current_identity != checkpoint_identity {
                bail!(
                    "evidence source, config, selection, or matrix changed during evaluation; discard the partial checkpoint and retry"
                );
            }
            record_evidence_timing(
                &mut timing,
                Some(&profile_id),
                "validation",
                validation_started.elapsed(),
            );
            if let Some(path) = checkpoint_path.as_deref() {
                let checkpoint = EvidenceMatrixCheckpoint {
                    schema_version: EVIDENCE_CHECKPOINT_SCHEMA_VERSION,
                    identity: checkpoint_identity.clone(),
                    resource_budget: Some(resource_budget),
                    rayon_threads: Some(rayon_threads),
                    elapsed_ms: cumulative_evidence_elapsed_ms(
                        generation_started,
                        prior_elapsed_ms,
                    ),
                    peak_working_set_bytes: prior_peak_working_set_bytes,
                    baselines: baselines.clone(),
                };
                checkpoint.validate(
                    &checkpoint_identity,
                    &profile_ids,
                    resource_budget,
                    rayon_threads,
                )?;
                let checkpoint_serialization_started = Instant::now();
                let checkpoint_bytes = serde_json::to_vec_pretty(&checkpoint)?;
                record_evidence_timing(
                    &mut timing,
                    Some(&profile_id),
                    "checkpoint_serialization",
                    checkpoint_serialization_started.elapsed(),
                );
                let checkpoint_write_started = Instant::now();
                crate::atomic_file::atomic_write(path, &checkpoint_bytes)?;
                record_evidence_timing(
                    &mut timing,
                    Some(&profile_id),
                    "checkpoint_write",
                    checkpoint_write_started.elapsed(),
                );
            }
            enforce_evidence_resource_budget(
                generation_started,
                prior_elapsed_ms,
                prior_peak_working_set_bytes,
                resource_budget,
            )?;
            record_evidence_timing(
                &mut timing,
                Some(&profile_id),
                "profile_total",
                profile_started.elapsed(),
            );
        }
        for baseline in &baselines {
            validate_evidence_baseline(baseline)?;
        }
        let reference_profile_id = baselines
            .iter()
            .find(|baseline| baseline.id == "selected_default_disk_artifacts")
            .or_else(|| baselines.first())
            .map(|baseline| baseline.id.clone())
            .ok_or_else(|| anyhow!("evidence matrix has no reference profile"))?;
        let reference_outcomes = baselines
            .iter()
            .find(|baseline| baseline.id == reference_profile_id)
            .expect("reference profile was selected")
            .result
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        for baseline in &mut baselines {
            let candidate = baseline
                .result
                .games
                .iter()
                .map(|game| game.outcome)
                .collect::<Vec<_>>();
            baseline.paired_vs_selected_default = Some(PairedDifference::all_game_penalized(
                &reference_outcomes,
                &candidate,
                7.0,
                BootstrapConfig::default(),
            )?);
        }

        let (code_revision, code_dirty) = git_provenance(&paths.root);
        let generation_compute_ms =
            cumulative_evidence_elapsed_ms(generation_started, prior_elapsed_ms);
        let memory = enforce_evidence_resource_budget(
            generation_started,
            prior_elapsed_ms,
            prior_peak_working_set_bytes,
            resource_budget,
        )?;
        let config_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-benchmark-root-config-v1",
            config_toml.as_bytes(),
        );
        ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
        let (final_matrix, final_matrix_source, final_matrix_fingerprint) =
            load_evidence_matrix(paths, matrix_path)?;
        let final_profile_ids = final_matrix
            .profiles
            .iter()
            .map(|profile| profile.id.clone())
            .collect::<Vec<_>>();
        let final_profile_base_configs =
            resolved_evidence_profile_base_configs(paths, config, &final_matrix)?;
        let final_identity = evidence_checkpoint_identity(
            &input_fingerprint,
            &config_toml,
            &plan,
            selection_label,
            &selected_ranges,
            &final_matrix_source,
            &final_matrix_fingerprint,
            &final_profile_ids,
            &final_profile_base_configs,
            DateRange::new(from, to)?,
            top,
            resource_budget,
            rayon_threads,
        )?;
        if final_identity != checkpoint_identity {
            bail!(
                "evidence source, config, selection, or matrix changed during evaluation; discard the partial checkpoint and retry"
            );
        }
        let selection_args = match selection {
            EvidenceDateSelection::Range(_) => format!("--from {from} --to {to}"),
            EvidenceDateSelection::RollingFolds => "--rolling-folds".to_string(),
        };
        let matrix_args = matrix_path
            .map(|path| format!(" --matrix {}", path.display()))
            .unwrap_or_default();
        Ok(PredictiveEvidenceArtifact {
            schema_version: BENCHMARK_EVIDENCE_SCHEMA_VERSION,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            config_fingerprint,
            scope: "rolling-development-diagnostic; not sealed-test evidence".to_string(),
            sealed_test_evaluated: false,
            evaluation_from: from,
            evaluation_to: to,
            evaluation_selection: selection_label.to_string(),
            selected_ranges: selected_ranges.clone(),
            matrix_source,
            matrix_fingerprint,
            profile_ids,
            reference_profile_id: reference_profile_id.clone(),
            history_snapshot_start: plan.history.start,
            history_snapshot_end: plan.history.end,
            code_revision,
            code_dirty,
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            cpu: std::env::var("PROCESSOR_IDENTIFIER").ok(),
            release_command: format!(
                "cargo run --release -- benchmark-evidence {selection_args}{matrix_args} --maximum-seconds {} --maximum-memory-mb {} --output <json> --markdown-output <md> --checkpoint <checkpoint>",
                resource_budget.maximum_seconds,
                resource_budget.maximum_memory_mb
            ),
            config_toml,
            resource_budget,
            resources: EvidenceResourceTelemetry {
                generation_compute_ms,
                current_working_set_bytes: Some(memory.current_working_set_bytes),
                peak_working_set_bytes: Some(
                    prior_peak_working_set_bytes.max(memory.peak_working_set_bytes),
                ),
                artifact_sizes: evidence_artifact_sizes(paths)?,
            },
            historical_diagnostic: HistoricalDiagnosticBaseline {
                date_range: "historical 30-game diagnostic before dormant-support repair"
                    .to_string(),
                scheduled_games: 30,
                modeled_games: 27,
                coverage_gaps: 3,
                conditional_mean_guesses: 3.2222,
                average_log_loss: 7.327027,
                average_brier_score: 0.999241,
                interpretation: "Attribution baseline only: its guess mean excluded three coverage gaps and is not comparable to an all-game score".to_string(),
            },
            baselines,
            limitations: vec![
                "This artifact evaluates development dates only; the sealed final window remains unopened.".to_string(),
                format!("The `{selection_label}` selection uses these exact validation ranges: {}.", selected_ranges.iter().map(|range| format!("{}..{}", range.start, range.end)).collect::<Vec<_>>().join(", ")),
                "Prior probabilities remain heuristic until calibration improves on rolling-origin validation.".to_string(),
                "SHA-256 identity fields detect changed inputs but do not prove statistical validity or authenticate an external artifact producer.".to_string(),
            ],
        })
    }

    pub fn build_rolling_config_comparison(
        paths: &ProjectPaths,
        baseline_config: &PriorConfig,
        baseline_label: &str,
        candidate_config: &PriorConfig,
        candidate_label: &str,
        top: usize,
        reusable_baseline: Option<&RollingComparisonArtifact>,
    ) -> Result<RollingComparisonArtifact> {
        let evaluation_plan = canonical_development_evaluation_plan(paths, "rolling comparison")?;
        let input_fingerprint =
            development_source_identity(paths, evaluation_plan.development.end)?;
        let baseline_toml = toml::to_string_pretty(baseline_config)
            .context("failed to serialize baseline config")?;
        let baseline = if let Some(reusable) = reusable_baseline {
            validate_rolling_comparison_artifact(reusable)?;
            if reusable.top != top {
                bail!("reusable baseline uses a different top setting; regenerate it");
            }
            if reusable.input_fingerprint != input_fingerprint {
                bail!("reusable baseline input fingerprint is stale; regenerate it");
            }
            if reusable.sealed_test_evaluated {
                bail!("cannot reuse a baseline artifact that evaluated the sealed test");
            }
            if reusable.evaluation_plan != evaluation_plan {
                bail!("reusable baseline uses a different rolling evaluation plan");
            }
            if reusable.baseline.config_toml != baseline_toml {
                bail!("reusable baseline uses a different default config");
            }
            reusable.baseline.clone()
        } else {
            Self::evaluate_config_on_rolling_folds(
                paths,
                baseline_config,
                baseline_label,
                &evaluation_plan,
                top,
            )?
        };
        let candidate = Self::evaluate_config_on_rolling_folds(
            paths,
            candidate_config,
            candidate_label,
            &evaluation_plan,
            top,
        )?;
        let baseline_outcomes = baseline
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        let candidate_outcomes = candidate
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        let comparison = PairedDifference::all_game_penalized(
            &baseline_outcomes,
            &candidate_outcomes,
            7.0,
            BootstrapConfig::default(),
        )?;
        ensure_development_source_identity(
            paths,
            evaluation_plan.development.end,
            &input_fingerprint,
        )?;
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        let comparison = RollingComparisonArtifact {
            schema_version: 5,
            top,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            evaluation_plan,
            sealed_test_evaluated: false,
            code_revision,
            code_dirty,
            baseline,
            candidate,
            candidate_minus_baseline: comparison,
        };
        validate_rolling_comparison_artifact(&comparison)?;
        Ok(comparison)
    }

    fn evaluate_config_on_rolling_folds(
        paths: &ProjectPaths,
        config: &PriorConfig,
        label: &str,
        evaluation_plan: &EvaluationPlan,
        top: usize,
    ) -> Result<RollingConfigEvidence> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let config_toml = toml::to_string_pretty(config)
            .context("failed to serialize rolling comparison config")?;
        let source_identity = development_source_identity(paths, evaluation_plan.development.end)?;
        let checkpoint_path =
            rolling_checkpoint_path(paths, label, &config_toml, &source_identity, top);
        let mut checkpoint = if checkpoint_path.exists() {
            let raw = fs::read(&checkpoint_path)
                .with_context(|| format!("failed to read {}", checkpoint_path.display()))?;
            let checkpoint: RollingEvaluationCheckpoint = serde_json::from_slice(&raw)
                .with_context(|| format!("failed to parse {}", checkpoint_path.display()))?;
            validate_rolling_checkpoint(
                &checkpoint,
                &source_identity,
                label,
                &config_toml,
                evaluation_plan,
                true,
            )
            .with_context(|| format!("invalid rolling checkpoint {}", checkpoint_path.display()))?;
            checkpoint
        } else {
            RollingEvaluationCheckpoint {
                schema_version: ROLLING_CHECKPOINT_SCHEMA_VERSION,
                source_identity: source_identity.clone(),
                evaluation_plan: evaluation_plan.clone(),
                label: label.to_string(),
                config_toml: config_toml.clone(),
                folds: Vec::with_capacity(evaluation_plan.folds.len()),
                games: Vec::new(),
                prior_observations: Vec::new(),
                execution: ExecutionTelemetry::default(),
            }
        };
        validate_rolling_checkpoint(
            &checkpoint,
            &source_identity,
            label,
            &config_toml,
            evaluation_plan,
            true,
        )?;
        let completed_folds = checkpoint
            .folds
            .iter()
            .map(|fold| fold.fold_index)
            .collect::<HashSet<_>>();
        for fold in &evaluation_plan.folds {
            if completed_folds.contains(&fold.index) {
                eprintln!(
                    "rolling-compare label={} fold={}/{} resumed_from_checkpoint=true",
                    label,
                    fold.index + 1,
                    evaluation_plan.folds.len()
                );
                continue;
            }
            let started = Instant::now();
            let report = solver.backtest_detailed_with_book_usage(
                fold.validation.start,
                fold.validation.end,
                top,
                PredictiveBookUsage::None,
            )?;
            checkpoint.folds.push(RollingFoldEvidence {
                fold_index: fold.index,
                validation: fold.validation,
                metrics: report.summary.canonical.clone(),
            });
            checkpoint.games.extend(report.runs.iter().map(|run| {
                ExperimentGameResult {
                    target: run.target.clone(),
                    outcome: if run.steps.is_empty() {
                        GameOutcome::coverage_gap(run.date)
                    } else if run.solved {
                        GameOutcome::solved(run.date, run.steps.len())
                    } else {
                        GameOutcome::unsolved(run.date, run.steps.len())
                    },
                    path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                    finite_search_steps: run
                        .steps
                        .iter()
                        .filter_map(|step| step.finite_search.clone())
                        .collect(),
                    prior_strata: None,
                    posterior_calibration: Vec::new(),
                }
            }));
            merge_execution_telemetry(
                &mut checkpoint.execution,
                &Self::execution_telemetry(&report.runs),
            );
            for entry in solver.history_dates.iter().filter(|entry| {
                entry.print_date >= fold.validation.start && entry.print_date <= fold.validation.end
            }) {
                if let Some(metrics) =
                    solver.initial_prior_metrics(&entry.solution, entry.print_date)
                {
                    checkpoint
                        .prior_observations
                        .push(RankedProbabilityObservation {
                            target_rank: metrics.target_rank,
                            top_probability: metrics.top_probability,
                            top_prediction_correct: metrics.top_prediction_correct,
                        });
                }
            }
            checkpoint.folds.sort_by_key(|fold| fold.fold_index);
            checkpoint.games.sort_by_key(|game| game.outcome.date);
            validate_rolling_checkpoint(
                &checkpoint,
                &source_identity,
                label,
                &config_toml,
                evaluation_plan,
                true,
            )?;
            crate::atomic_file::atomic_write(
                &checkpoint_path,
                &serde_json::to_vec_pretty(&checkpoint)
                    .context("failed to serialize rolling checkpoint")?,
            )?;
            eprintln!(
                "rolling-compare label={} fold={}/{} validation={}..{} all_game_mean={:.4} failures={} elapsed_ms={}",
                label,
                fold.index + 1,
                evaluation_plan.folds.len(),
                fold.validation.start,
                fold.validation.end,
                report.summary.canonical.all_game_penalized_mean_guesses,
                report.summary.canonical.unsolved_games + report.summary.canonical.coverage_gaps,
                started.elapsed().as_millis()
            );
        }
        validate_rolling_checkpoint(
            &checkpoint,
            &source_identity,
            label,
            &config_toml,
            evaluation_plan,
            true,
        )?;
        if checkpoint.folds.len() != evaluation_plan.folds.len() {
            bail!("rolling evaluation ended before every planned fold completed");
        }
        checkpoint.games.sort_by_key(|game| game.outcome.date);
        let outcomes = checkpoint
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        let aggregate = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let failure_penalty_sensitivity = [6.0, 7.0, 8.0]
            .into_iter()
            .map(|penalty_guesses| {
                let metrics = summarize_predictive_outcomes(
                    &outcomes,
                    penalty_guesses,
                    BootstrapConfig::default(),
                )?;
                Ok(FailurePenaltyEvidence {
                    penalty_guesses,
                    all_game_mean_guesses: metrics.all_game_penalized_mean_guesses,
                    ci95: metrics.all_game_penalized_mean_guesses_ci95,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let config_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-rolling-config-v1",
            config_toml.as_bytes(),
        );
        Ok(RollingConfigEvidence {
            label: label.to_string(),
            config_toml,
            config_fingerprint,
            folds: checkpoint.folds,
            aggregate,
            prior_evidence: (!checkpoint.prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &checkpoint.prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            execution: checkpoint.execution,
            failure_penalty_sensitivity,
            games: checkpoint.games,
            latency_p95_ms: solver.benchmark_predictive_latency(
                evaluation_plan.development.end,
                default_diagnostic_suite()?.latency.evidence_runs,
            )?,
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

    pub fn development_evaluation_plan(paths: &ProjectPaths) -> Result<EvaluationPlan> {
        let (history_start, history_end) = Self::latest_history_range(paths)?
            .ok_or_else(|| anyhow!("run sync-data before development evaluation"))?;
        let history = DateRange::new(history_start, history_end)?;
        let policy =
            crate::experiments::EvaluationPolicy::load(&paths.root.join("config/evaluation.toml"))?;
        crate::experiments::build_declared_rolling_origin_plan(
            history,
            rolling_origin_config_for_history(history)?,
            &policy,
        )
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
        self.build_predictive_opener_cache_controlled(as_of, &|| false)
    }

    fn build_predictive_opener_cache_controlled(
        &self,
        as_of: NaiveDate,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<PredictiveOpenerBuildSummary> {
        check_predictive_search_cancelled(cancelled)?;
        let offline = self.offline_book_solver()?;
        let (window_start, window_end, targets) =
            offline.recent_history_targets_for_books(as_of)?;
        let holdout = offline.previous_history_targets_for_books(window_start)?;
        let state = offline.initial_state(as_of);
        let candidates = offline
            .suggestion_batch_internal_with_search_mode_controlled(
                &state,
                offline.config.session_opener_pool.max(1),
                Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &[],
                }),
                PredictiveBookUsage::None,
                None,
                cancelled,
            )?
            .suggestions;
        let selected = offline
            .select_validated_opener(
                as_of,
                &candidates,
                &targets,
                holdout.as_ref().map(|(_, _, entries)| entries.as_slice()),
                cancelled,
            )?
            .ok_or_else(|| anyhow!("missing predictive opener candidate"))?;
        let opener = selected.word.clone();
        let artifact = PredictiveOpenerArtifact {
            identity: self.predictive_book_identity(as_of),
            opener: opener.clone(),
            search_window_start: window_start,
            search_window_end: window_end,
            games: selected.primary.games,
            four_guess_games: selected.primary.four_guess_games,
            average_guesses: selected.primary.average_guesses,
            failures: selected.primary.failures,
            holdout_window_start: holdout.as_ref().map(|(start, _, _)| *start),
            holdout_window_end: holdout.as_ref().map(|(_, end, _)| *end),
            holdout_games: selected.holdout.as_ref().map_or(0, |eval| eval.games),
            holdout_four_guess_games: selected
                .holdout
                .as_ref()
                .map_or(0, |eval| eval.four_guess_games),
            holdout_average_guesses: selected
                .holdout
                .as_ref()
                .map_or(0.0, |eval| eval.average_guesses),
            holdout_failures: selected.holdout.as_ref().map_or(0, |eval| eval.failures),
            proxy_cost: None,
            lookahead_cost: None,
            exact_cost: None,
        };
        let path = self.opener_artifact_path(as_of);
        check_predictive_search_cancelled(cancelled)?;
        write_predictive_artifact(&path, &artifact)?;
        Ok(PredictiveOpenerBuildSummary {
            path,
            opener: artifact.opener,
            as_of,
            config_fingerprint: artifact.identity.config_fingerprint,
            games: artifact.games,
            four_guess_games: artifact.four_guess_games,
            average_guesses: artifact.average_guesses,
            failures: artifact.failures,
            holdout_games: artifact.holdout_games,
            holdout_four_guess_games: artifact.holdout_four_guess_games,
            holdout_average_guesses: artifact.holdout_average_guesses,
            holdout_failures: artifact.holdout_failures,
        })
    }

    pub fn build_predictive_reply_book(
        &self,
        as_of: NaiveDate,
    ) -> Result<PredictiveReplyBuildSummary> {
        self.build_predictive_reply_book_controlled(as_of, &|| false)
    }

    fn build_predictive_reply_book_controlled(
        &self,
        as_of: NaiveDate,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<PredictiveReplyBuildSummary> {
        check_predictive_search_cancelled(cancelled)?;
        let opener_artifact = self
            .load_predictive_opener_artifact(as_of)?
            .ok_or_else(|| anyhow!("build the predictive opener cache first"))?;
        let opener_index = self
            .guess_index
            .get(&opener_artifact.opener)
            .copied()
            .ok_or_else(|| anyhow!("cached opener is not in the current guess list"))?;
        let offline = self.offline_book_solver()?;
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let root = offline.initial_state(as_of);
        let mut seen_patterns = HashSet::new();
        let mut replies = Vec::new();
        let reply_candidate_limit = offline.config.session_reply_pool;

        for answer_index in &root.surviving {
            check_predictive_search_cancelled(cancelled)?;
            let pattern = offline.answer_pattern(opener_index, *answer_index);
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
            let batch = offline.suggestion_batch_internal_with_search_mode_controlled(
                &child,
                reply_candidate_limit,
                Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &observation,
                }),
                PredictiveBookUsage::None,
                None,
                cancelled,
            )?;
            let mut best_reply: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
            for suggestion in batch.suggestions.into_iter().take(reply_candidate_limit) {
                let guess_index = offline
                    .guess_index
                    .get(&suggestion.word)
                    .copied()
                    .ok_or_else(|| anyhow!("missing reply guess {}", suggestion.word))?;
                let evaluation = offline.evaluate_forced_continuation(
                    std::slice::from_ref(&opener_artifact.opener),
                    &scoped_targets,
                    guess_index,
                    cancelled,
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
                    check_predictive_search_cancelled(cancelled)?;
                    let second_feedback = offline.answer_pattern(reply_index, *target_index);
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
                    let grand_batch = offline
                        .suggestion_batch_internal_with_search_mode_controlled(
                            &grandchild,
                            reply_candidate_limit,
                            Some(PredictiveContext {
                                hard_mode: false,
                                as_of,
                                observations: &grand_observations,
                            }),
                            PredictiveBookUsage::None,
                            None,
                            cancelled,
                        )?;
                    let mut best_third: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
                    for suggestion in grand_batch
                        .suggestions
                        .into_iter()
                        .take(reply_candidate_limit)
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
                            cancelled,
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
        replies.sort_by_key(|reply| reply.feedback_pattern);
        let artifact = PredictiveReplyBookArtifact {
            identity: self.predictive_book_identity(as_of),
            opener: opener_artifact.opener.clone(),
            replies,
        };
        let path = self.reply_book_artifact_path(as_of);
        check_predictive_search_cancelled(cancelled)?;
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
        Self::predictive_ablation_report_filtered(paths, config, from, to, top, None)
    }

    pub fn predictive_ablation_report_filtered(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        profile_filter: Option<&str>,
    ) -> Result<Vec<PredictiveAblationResult>> {
        ensure_development_target_range(paths, from, to, "predictive ablation")?;
        let registry = predictive_parameter_registry(config);
        let mut matrix = PredictiveExperimentMatrix::parse_json(include_str!(
            "../../config/experiments/predictive-ablations.json"
        ))?;
        if let Some(profile_id) = profile_filter {
            matrix.profiles.retain(|profile| profile.id == profile_id);
            if matrix.profiles.is_empty() {
                bail!("unknown predictive ablation profile {profile_id}");
            }
        }
        let started = Instant::now();
        let profile_count = matrix.profiles.len();
        let mut rows = Vec::with_capacity(matrix.profiles.len());
        for (profile_index, profile) in matrix.profiles.into_iter().enumerate() {
            eprintln!(
                "predictive-ablation phase=profile-start profile={}/{} id={} elapsed_s={:.1}",
                profile_index + 1,
                profile_count,
                profile.id,
                started.elapsed().as_secs_f64()
            );
            let candidate = profile.apply(&registry, config)?;
            let book_usage = match profile.artifact_mode {
                ExperimentArtifactMode::Disabled => PredictiveBookUsage::None,
                ExperimentArtifactMode::DiskOnly => PredictiveBookUsage::DiskOnly,
            };
            let solver = Self::from_paths_with_settings(
                paths,
                &candidate,
                profile.weight_mode,
                profile.model_variant,
            )?;
            let result = solver.experiment_report_with_book_usage(from, to, top, book_usage)?;
            rows.push(PredictiveAblationResult {
                label: profile.id,
                result,
            });
            let elapsed = started.elapsed().as_secs_f64();
            let remaining = profile_count - profile_index - 1;
            let eta = elapsed / (profile_index + 1) as f64 * remaining as f64;
            eprintln!(
                "predictive-ablation phase=profile-done profile={}/{} id={} elapsed_s={elapsed:.1} eta_s={eta:.1}",
                profile_index + 1,
                profile_count,
                rows.last().expect("just pushed").label
            );
        }
        Ok(rows)
    }

    pub fn predictive_prior_ablation_report(
        paths: &ProjectPaths,
        config: &PriorConfig,
        profile_filter: &[String],
    ) -> Result<PredictivePriorAblationReport> {
        let started = Instant::now();
        let evaluation_plan = Self::development_evaluation_plan(paths)?;
        let input_fingerprint =
            development_source_identity(paths, evaluation_plan.development.end)?;
        let matrix_source = include_str!("../../config/experiments/predictive-ablations.json");
        let matrix_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-prior-ablation-matrix-v1",
            matrix_source.as_bytes(),
        );
        let registry = predictive_parameter_registry(config);
        let mut matrix = PredictiveExperimentMatrix::parse_json(matrix_source)?;
        if profile_filter.is_empty() {
            matrix
                .profiles
                .retain(|profile| profile.id.ends_with("_baseline"));
        } else {
            matrix
                .profiles
                .retain(|profile| profile_filter.contains(&profile.id));
            if matrix.profiles.len() != profile_filter.len() {
                bail!("one or more requested predictive prior profiles are unknown or duplicated");
            }
        }

        let mut profiles = Vec::with_capacity(matrix.profiles.len());
        for profile in matrix.profiles {
            let candidate = profile.apply(&registry, config)?;
            let config_toml = toml::to_string_pretty(&candidate)?;
            let solver = Self::from_paths_with_settings(
                paths,
                &candidate,
                profile.weight_mode,
                profile.model_variant,
            )?;
            let mut folds = Vec::with_capacity(evaluation_plan.folds.len());
            let mut scheduled_games = 0usize;
            let mut measured_games = 0usize;
            let mut log_loss_sum = 0.0;
            let mut brier_sum = 0.0;
            for fold in &evaluation_plan.folds {
                let mut fold_scheduled = 0usize;
                let mut fold_measured = 0usize;
                let mut fold_log_loss = 0.0;
                let mut fold_brier = 0.0;
                for entry in solver
                    .history_dates
                    .iter()
                    .filter(|entry| fold.validation.contains(entry.print_date))
                {
                    fold_scheduled += 1;
                    if let Some(metrics) =
                        solver.initial_prior_metrics(&entry.solution, entry.print_date)
                    {
                        fold_measured += 1;
                        fold_log_loss += metrics.log_loss;
                        fold_brier += metrics.brier;
                    }
                }
                scheduled_games += fold_scheduled;
                measured_games += fold_measured;
                log_loss_sum += fold_log_loss;
                brier_sum += fold_brier;
                folds.push(PredictivePriorAblationFold {
                    fold_index: fold.index,
                    validation: fold.validation,
                    scheduled_games: fold_scheduled,
                    measured_games: fold_measured,
                    coverage_gaps: fold_scheduled.saturating_sub(fold_measured),
                    average_log_loss: (fold_measured > 0)
                        .then(|| fold_log_loss / fold_measured as f64),
                    average_brier: (fold_measured > 0).then(|| fold_brier / fold_measured as f64),
                });
            }
            profiles.push(PredictivePriorAblationProfile {
                label: profile.id,
                mode: profile.weight_mode,
                variant: profile.model_variant,
                config_fingerprint: crate::identity::digest_bytes_tagged(
                    "maybe-wordle-prior-ablation-config-v1",
                    config_toml.as_bytes(),
                ),
                scheduled_games,
                measured_games,
                coverage_gaps: scheduled_games.saturating_sub(measured_games),
                average_log_loss: (measured_games > 0)
                    .then(|| log_loss_sum / measured_games as f64),
                average_brier: (measured_games > 0).then(|| brier_sum / measured_games as f64),
                folds,
                promotable: false,
                promotion_blockers: Vec::new(),
            });
        }

        let baseline = profiles
            .iter()
            .find(|profile| profile.label == "weighted_baseline")
            .map(|profile| {
                (
                    profile.coverage_gaps,
                    profile.average_log_loss,
                    profile.average_brier,
                )
            });
        for profile in &mut profiles {
            if profile.label == "weighted_baseline" {
                profile
                    .promotion_blockers
                    .push("Reference weighted baseline; not a replacement candidate.".to_string());
                continue;
            }
            if profile.coverage_gaps > 0 {
                profile.promotion_blockers.push(format!(
                    "The prior has {} rolling-development coverage gaps.",
                    profile.coverage_gaps
                ));
            }
            if let Some((baseline_gaps, baseline_log_loss, baseline_brier)) = baseline {
                if profile.coverage_gaps > baseline_gaps {
                    profile
                        .promotion_blockers
                        .push("Coverage is worse than the weighted reference prior.".to_string());
                }
                match (profile.average_log_loss, profile.average_brier, baseline_log_loss, baseline_brier) {
                    (Some(log_loss), Some(brier), Some(reference_log_loss), Some(reference_brier)) => {
                        if log_loss >= reference_log_loss {
                            profile.promotion_blockers.push(
                                "Log loss does not improve on the weighted reference prior.".to_string(),
                            );
                        }
                        if brier >= reference_brier {
                            profile.promotion_blockers.push(
                                "Brier score does not improve on the weighted reference prior.".to_string(),
                            );
                        }
                    }
                    _ => profile.promotion_blockers.push(
                        "Candidate or weighted reference prior scores are unavailable; no measured comparison can justify promotion.".to_string(),
                    ),
                }
            } else {
                profile.promotion_blockers.push(
                    "The weighted reference profile was not included in this run.".to_string(),
                );
            }
            profile.promotable = profile.promotion_blockers.is_empty();
        }
        ensure_development_source_identity(
            paths,
            evaluation_plan.development.end,
            &input_fingerprint,
        )?;
        Ok(PredictivePriorAblationReport {
            schema_version: 2,
            input_fingerprint,
            matrix_fingerprint,
            evaluation_plan,
            profiles,
            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        })
    }

    pub(super) fn snapshot_suggestion(suggestion: &Suggestion) -> SuggestionSnapshot {
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

    pub(super) fn assess_state_danger(
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

    pub(super) fn assess_subset_danger(
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
        let top_concentration = posterior
            .iter()
            .take(self.config.danger_posterior_window)
            .sum::<f64>();
        let best = metrics[0];
        let top_window = metrics
            .iter()
            .take(self.config.danger_candidate_window)
            .copied()
            .collect::<Vec<_>>();
        let disagreement = top_window.iter().skip(1).any(|metric| {
            metric.force_in_two != best.force_in_two
                || (metric.largest_non_green_bucket_mass - best.largest_non_green_bucket_mass).abs()
                    >= self.config.danger_mass_disagreement_threshold
                || metric
                    .worst_non_green_bucket_size
                    .abs_diff(best.worst_non_green_bucket_size)
                    >= self.config.danger_size_disagreement_threshold
        });
        let worst_bucket_ratio =
            best.worst_non_green_bucket_size as f64 / subset.len().max(1) as f64;
        let ambiguous_bucket_pressure = (best.high_mass_ambiguous_bucket_count as f64
            / self.config.danger_ambiguity_saturation_count as f64)
            .min(1.0);
        let weight_sum = self.config.danger_top_concentration_w
            + self.config.danger_bucket_mass_w
            + self.config.danger_bucket_ratio_w
            + self.config.danger_ambiguous_w
            + self.config.danger_disagreement_w;
        let danger_score = ((self.config.danger_top_concentration_w * top_concentration)
            + (self.config.danger_bucket_mass_w * best.largest_non_green_bucket_mass)
            + (self.config.danger_bucket_ratio_w * worst_bucket_ratio)
            + (self.config.danger_ambiguous_w * ambiguous_bucket_pressure)
            + if disagreement {
                self.config.danger_disagreement_w
            } else {
                0.0
            })
            / weight_sum;
        StateDangerAssessment {
            danger_score,
            dangerous_lookahead: danger_score >= self.config.danger_lookahead_threshold,
            dangerous_exact: danger_score >= self.config.danger_exact_threshold,
        }
    }

    pub(super) fn regime_mix(runs: &[DetailedSolveRun]) -> (f64, f64, f64, f64, f64, f64) {
        let mut proxy_steps = 0usize;
        let mut lookahead_steps = 0usize;
        let mut escalated_exact_steps = 0usize;
        let mut exact_steps = 0usize;
        let mut finite_steps = 0usize;
        let mut terminal_steps = 0usize;
        let mut total_steps = 0usize;

        for run in runs {
            for step in &run.steps {
                total_steps += 1;
                match step.regime_used {
                    PredictiveRegime::Proxy => proxy_steps += 1,
                    PredictiveRegime::Lookahead => lookahead_steps += 1,
                    PredictiveRegime::EscalatedExact => escalated_exact_steps += 1,
                    PredictiveRegime::Exact => exact_steps += 1,
                    PredictiveRegime::Finite => finite_steps += 1,
                    PredictiveRegime::Terminal => terminal_steps += 1,
                }
            }
        }

        if total_steps == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let divisor = total_steps as f64;
        (
            proxy_steps as f64 / divisor,
            lookahead_steps as f64 / divisor,
            escalated_exact_steps as f64 / divisor,
            exact_steps as f64 / divisor,
            finite_steps as f64 / divisor,
            terminal_steps as f64 / divisor,
        )
    }

    pub(super) fn execution_telemetry(runs: &[DetailedSolveRun]) -> ExecutionTelemetry {
        let mut telemetry = ExecutionTelemetry::default();
        for step in runs.iter().flat_map(|run| &run.steps) {
            telemetry.total_steps += 1;
            match step.regime_used {
                PredictiveRegime::Proxy => telemetry.proxy_steps += 1,
                PredictiveRegime::Lookahead => telemetry.lookahead_steps += 1,
                PredictiveRegime::EscalatedExact => telemetry.escalated_exact_steps += 1,
                PredictiveRegime::Exact => telemetry.exact_steps += 1,
                PredictiveRegime::Finite => telemetry.finite_steps += 1,
                PredictiveRegime::Terminal => telemetry.terminal_steps += 1,
            }
            telemetry.danger_escalated_steps += usize::from(step.danger_escalated);
            telemetry.dormant_fallback_steps += usize::from(step.fallback_active);
            match step.recovery_mode_used {
                Some(RecoveryMode::Strict) => telemetry.strict_recovery_steps += 1,
                Some(RecoveryMode::UniformOverSupport) => telemetry.uniform_recovery_steps += 1,
                Some(RecoveryMode::EpsilonRepair) => telemetry.epsilon_repair_steps += 1,
                None => {}
            }
            match step.promotion_source {
                Some(PredictivePromotionSource::ExactDateOpenerArtifact) => {
                    telemetry.exact_date_opener_artifact_hits += 1;
                }
                Some(PredictivePromotionSource::RecentOpenerArtifact) => {
                    telemetry.recent_opener_artifact_hits += 1;
                }
                Some(
                    PredictivePromotionSource::ReplyBook
                    | PredictivePromotionSource::RecentReplyBook,
                ) => telemetry.reply_book_hits += 1,
                Some(
                    PredictivePromotionSource::SessionRootFallback
                    | PredictivePromotionSource::SessionReplyFallback
                    | PredictivePromotionSource::SessionThirdFallback,
                ) => telemetry.session_fallback_hits += 1,
                None => {}
            }
        }
        telemetry
    }

    pub(super) fn select_hard_case_targets(
        &self,
        as_of: NaiveDate,
        top: usize,
        spec: &crate::experiments::HardCaseDiagnosticSpec,
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
                            ) <= spec.maximum_cluster_hamming_distance
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
                .take(spec.top_posterior_scan)
                .filter_map(|(answer_index, word, weight)| {
                    let cluster_mass = ranked
                        .iter()
                        .take(spec.top_posterior_scan)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= spec.maximum_cluster_hamming_distance
                        })
                        .map(|(_, _, other_weight)| *other_weight)
                        .sum::<f64>();
                    let neighbors = ranked
                        .iter()
                        .take(spec.top_posterior_scan)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= spec.maximum_cluster_hamming_distance
                        })
                        .count();
                    (neighbors >= spec.minimum_trap_neighbors)
                        .then_some((cluster_mass + *weight, word.clone()))
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
        for (_, target, _) in candidate_answers
            .into_iter()
            .take(spec.low_prior_splitter_scan)
        {
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
            if let Some(target) = target
                && (label == "high_posterior_trap"
                    || selected
                        .iter()
                        .all(|(_, existing): &(String, String)| existing != &target))
            {
                selected.push((label.to_string(), target));
            }
        }
        if selected.is_empty() {
            bail!("unable to construct hard-case suite from current model");
        }
        selected.truncate(spec.target_count);
        Ok(selected)
    }

    pub fn tune_prior(paths: &ProjectPaths, config: &PriorConfig) -> Result<TunePriorSummary> {
        let evaluation_plan = canonical_development_evaluation_plan(paths, "tune-prior")?;
        let first_fold = evaluation_plan
            .folds
            .first()
            .ok_or_else(|| anyhow!("rolling-origin plan contains no folds"))?;
        let last_fold = evaluation_plan
            .folds
            .last()
            .ok_or_else(|| anyhow!("rolling-origin plan contains no folds"))?;
        let window_start = first_fold.training.start;
        let window_end = last_fold.training.end;
        let validation_start = first_fold.validation.start;
        let validation_end = last_fold.validation.end;
        let test_start = evaluation_plan.sealed_test.start;
        let test_end = evaluation_plan.sealed_test.end;
        let study_state_path = paths.root.join("target/studies/tune-prior-v18.json");
        let study_summary = Self::run_predictive_study(
            paths,
            config,
            StudySpec {
                name: "tune-prior".to_string(),
                stage: StudyStage::Calibration,
                seed: 20_260_315,
                trial_count: 24,
                parallelism: std::thread::available_parallelism()
                    .map_or(1, |count| count.get().min(4)),
                strategy: crate::experiments::StudySearchStrategy::LowDiscrepancy,
                maximum_validation_folds: evaluation_plan.folds.len(),
                initial_validation_folds: evaluation_plan.folds.len().min(3),
                reduction_factor: 3,
                fold_selection: crate::experiments::StudyFoldSelection::NestedTimeSpread,
                maximum_trial_seconds: 7_200,
                maximum_memory_mb: 4_096,
            },
            &study_state_path,
            5,
            None,
        )?;
        let best_prior_config = study_summary.best_config.unwrap_or_else(|| config.clone());

        let validation_current = Self::evaluate_tuning_candidate(paths, config, &evaluation_plan)?;
        let candidate =
            Self::evaluate_tuning_candidate(paths, &best_prior_config, &evaluation_plan)?;
        let selected_config = if candidate.all_game_penalized_mean_guesses
            < validation_current.all_game_penalized_mean_guesses
            && candidate.failures <= validation_current.failures
            && candidate.coverage_gaps <= validation_current.coverage_gaps
            && candidate.latency_p95_ms
                <= (validation_current.latency_p95_ms * 3.0).max(validation_current.latency_p95_ms)
        {
            candidate.config
        } else {
            config.clone()
        };
        // The sealed final period is intentionally not evaluated by tuning.
        let current = validation_current;
        let best = Self::evaluate_tuning_candidate(paths, &selected_config, &evaluation_plan)?;
        let replacement_toml = toml::to_string_pretty(&best.config)
            .context("failed to serialize selected prior-study config")?;

        Ok(TunePriorSummary {
            evaluation_plan,
            search_window_start: window_start,
            search_window_end: window_end,
            validation_window_start: validation_start,
            validation_window_end: validation_end,
            test_window_start: test_start,
            test_window_end: test_end,
            current,
            best,
            replacement_toml,
        })
    }

    pub fn evaluate_live_config(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<LiveConfigEvaluation> {
        if from > to {
            bail!("live-config evaluation start date cannot be after end date");
        }
        let _evaluation_plan =
            ensure_development_target_range(paths, from, to, "live-config evaluation")?;
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let backtest =
            solver.backtest_detailed_with_book_usage(from, to, top, PredictiveBookUsage::None)?;
        let hard_cases =
            solver.hard_case_report_with_book_usage(to, top, PredictiveBookUsage::None)?;
        let latency_p95_ms = solver.benchmark_predictive_latency(
            to,
            default_diagnostic_suite()?.latency.evaluation_runs,
        )?;
        Ok(LiveConfigEvaluation {
            config: config.clone(),
            predictive_metrics: backtest.summary.canonical.clone(),
            average_guesses: backtest.summary.average_guesses,
            all_game_penalized_mean_guesses: backtest
                .summary
                .canonical
                .all_game_penalized_mean_guesses,
            failures: backtest.summary.failures,
            coverage_gaps: backtest.summary.coverage_gaps,
            latency_p95_ms,
            hard_case_average_guesses: hard_cases.average_guesses,
            hard_case_failures: hard_cases.failures,
        })
    }

    pub fn three_guess_gap_report(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<ThreeGuessGapReport> {
        let base_solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let aggressive_solver =
            base_solver.clone_with_config(aggressive_early_exact_config(config)?);
        let diagnostic = default_diagnostic_suite()?.three_guess_rescue;
        if diagnostic.profile != "aggressive-three-guess" {
            bail!(
                "unsupported three-guess diagnostic profile: {}",
                diagnostic.profile
            );
        }
        let (base_backtest, four_guess_runs) = base_solver.four_guess_runs(from, to, top)?;
        let mut cases = four_guess_runs
            .par_iter()
            .map(|run| {
                let as_of = run
                    .date
                    .checked_sub_days(Days::new(1))
                    .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
                let solver = aggressive_solver.clone();
                let aggressive_run = solver.solve_target_from_state_detailed(
                    &run.target,
                    as_of,
                    run.date,
                    top,
                    PredictiveBookUsage::None,
                )?;
                let best_forced = solver.best_three_guess_attempt_for_target(
                    &run.target,
                    run.date,
                    top,
                    diagnostic.root_candidate_limit,
                    diagnostic.reply_candidate_limit,
                )?;
                let converted_aggressive = aggressive_run.solved && aggressive_run.steps.len() <= 3;
                let converted_targeted = best_forced.solved && best_forced.steps.len() <= 3;
                Ok(ThreeGuessGapCase {
                    target: run.target.clone(),
                    date: run.date,
                    base_guesses: run.steps.len(),
                    aggressive_guesses: aggressive_run.steps.len(),
                    best_forced_guesses: best_forced.steps.len(),
                    converted_by_aggressive: converted_aggressive,
                    converted_by_targeted_search: converted_targeted,
                    base_path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                    aggressive_path: aggressive_run
                        .steps
                        .iter()
                        .map(|step| step.guess.clone())
                        .collect(),
                    best_forced_path: best_forced
                        .steps
                        .iter()
                        .map(|step| step.guess.clone())
                        .collect(),
                })
            })
            .collect::<Vec<Result<ThreeGuessGapCase>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        cases.sort_by(|left, right| {
            left.date
                .cmp(&right.date)
                .then_with(|| left.target.cmp(&right.target))
        });
        let converted_by_aggressive = cases
            .iter()
            .filter(|case| case.converted_by_aggressive)
            .count();
        let converted_by_targeted_search = cases
            .iter()
            .filter(|case| case.converted_by_targeted_search)
            .count();
        let aggressive_four_guess_cases = cases
            .iter()
            .filter(|case| case.aggressive_guesses == 4)
            .count();
        let aggressive_guess_total = cases
            .iter()
            .map(|case| case.aggressive_guesses)
            .sum::<usize>();

        Ok(ThreeGuessGapReport {
            games: base_backtest.summary.games,
            base_average_guesses: base_backtest.summary.average_guesses,
            aggressive_case_average_guesses: if cases.is_empty() {
                0.0
            } else {
                aggressive_guess_total as f64 / cases.len() as f64
            },
            base_four_guess_cases: base_backtest
                .runs
                .iter()
                .filter(|run| run.solved && run.steps.len() == 4)
                .count(),
            aggressive_four_guess_cases,
            converted_by_aggressive,
            converted_by_targeted_search,
            cases,
        })
    }

    pub fn four_guess_opener_report(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        openers: &[String],
    ) -> Result<FourGuessOpenerReport> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let (_, four_guess_runs) = solver.four_guess_runs(from, to, top)?;
        let targets = four_guess_runs
            .iter()
            .map(|run| (run.date, run.target.clone()))
            .collect::<Vec<_>>();
        let opener_list = if openers.is_empty() {
            default_diagnostic_suite()?
                .default_four_guess_openers
                .into_iter()
                .filter(|opener| solver.has_guess(opener))
                .collect::<Vec<_>>()
        } else {
            openers
                .iter()
                .map(|opener| opener.trim().to_ascii_lowercase())
                .collect::<Vec<_>>()
        };
        for opener in &opener_list {
            if !solver.has_guess(opener) {
                bail!("unknown opener: {}", opener);
            }
        }
        let evaluations = opener_list
            .into_par_iter()
            .map(|opener| solver.evaluate_named_opener_on_targets(&targets, &opener, top))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let mut evaluations = evaluations;
        evaluations.sort_by(|left, right| {
            left.average_guesses
                .total_cmp(&right.average_guesses)
                .then_with(|| right.three_guess_solves.cmp(&left.three_guess_solves))
                .then_with(|| left.failures.cmp(&right.failures))
                .then_with(|| left.opener.cmp(&right.opener))
        });
        Ok(FourGuessOpenerReport {
            games: targets.len(),
            targets: four_guess_runs
                .into_iter()
                .map(|run| FourGuessTarget {
                    target: run.target,
                    date: run.date,
                    base_path: run.steps.into_iter().map(|step| step.guess).collect(),
                })
                .collect(),
            evaluations,
        })
    }

    pub(super) fn initial_prior_metrics(
        &self,
        target: &str,
        date: NaiveDate,
    ) -> Option<PriorMetrics> {
        let as_of = crate::predictive::history_cutoff(date).ok()?;
        let state = self.initial_state(as_of);
        self.posterior_metrics_for_state(&state, target)
    }

    fn posterior_metrics_for_state(
        &self,
        state: &SolveState,
        target: &str,
    ) -> Option<PriorMetrics> {
        if state.total_weight <= 0.0 || !state.total_weight.is_finite() {
            return None;
        }
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
        let target_position = ordered
            .iter()
            .position(|(index, _)| *index == target_index)?;
        let probability_score = score_multiclass_probabilities(
            &ordered
                .iter()
                .map(|(_, probability)| *probability)
                .collect::<Vec<_>>(),
            target_position,
        )
        .ok()?;

        Some(PriorMetrics {
            target_probability,
            target_rank,
            log_loss: probability_score.log_loss,
            brier: probability_score.brier,
            top_probability: ordered.first()?.1,
            top_prediction_correct: target_position == 0,
        })
    }

    fn prior_strata_for_answer(
        answer: &AnswerRecord,
        modeled_weight: f64,
        as_of: NaiveDate,
    ) -> PriorStrata {
        let prior_count = answer.history_dates.partition_point(|date| *date <= as_of);
        PriorStrata {
            never_used: prior_count == 0,
            reused: prior_count > 0,
            historical_only: !answer.in_seed && prior_count > 0,
            out_of_core: modeled_weight <= 0.0,
        }
    }

    fn posterior_calibration_for_run(
        &self,
        run: &DetailedSolveRun,
    ) -> Result<(Option<PriorStrata>, Vec<PosteriorCalibrationObservation>)> {
        if run.steps.len() > 6 {
            bail!(
                "posterior calibration run {} has more than six guesses",
                run.date
            );
        }
        let as_of = crate::predictive::history_cutoff(run.date)?;
        let mut state = self.initial_state(as_of);
        let target = run.target.to_ascii_lowercase();
        let target_index = self.answers.iter().position(|answer| answer.word == target);
        let prior_strata = target_index.map(|target_index| {
            let answer = &self.answers[target_index];
            Self::prior_strata_for_answer(answer, state.modeled_weights[target_index], as_of)
        });

        // A coverage-gap run has no actual guess state to score. Keep its
        // first-turn denominator visible, but never manufacture a score.
        if run.steps.is_empty() {
            return Ok((
                prior_strata,
                vec![PosteriorCalibrationObservation {
                    turn: 1,
                    score: None,
                }],
            ));
        }

        let mut observations = Vec::with_capacity(run.steps.len());
        for (step_index, step) in run.steps.iter().enumerate() {
            let score = self
                .posterior_metrics_for_state(&state, &target)
                .map(|metrics| crate::experiments::ProbabilityScore {
                    target_probability: metrics.target_probability,
                    log_loss: metrics.log_loss,
                    brier: metrics.brier,
                });
            observations.push(PosteriorCalibrationObservation {
                turn: u8::try_from(step_index + 1).expect("calibration has at most six turns"),
                score,
            });

            // The all-green state has no next decision. Likewise, do not
            // create a turn seven row after the final allowed guess.
            if step.feedback == ALL_GREEN_PATTERN || step_index + 1 == run.steps.len() {
                break;
            }
            self.apply_feedback(&mut state, &step.guess, step.feedback)?;
        }
        Ok((prior_strata, observations))
    }

    pub(super) fn benchmark_predictive_latency(
        &self,
        puzzle_date: NaiveDate,
        runs: usize,
    ) -> Result<f64> {
        self.benchmark_predictive_latency_controlled(puzzle_date, runs, &|| false)
    }

    fn benchmark_predictive_latency_controlled(
        &self,
        puzzle_date: NaiveDate,
        runs: usize,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<f64> {
        let run_count = runs.max(1);
        let top = default_diagnostic_suite()?.latency.top_suggestions;
        let mut samples = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            let start = Instant::now();
            check_predictive_search_cancelled(cancelled)?;
            let _ = self.suggest_predictive_cancellable(
                PredictiveSuggestRequest {
                    puzzle_date,
                    observations: &[],
                    top,
                    hard_mode: false,
                    force_in_two_only: false,
                    mode: PredictiveSuggestionMode::LiveOnly,
                },
                cancelled,
            )?;
            check_predictive_search_cancelled(cancelled)?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|left, right| left.total_cmp(right));
        let p95_index = ((samples.len() as f64) * 0.95).ceil() as usize;
        Ok(samples[p95_index.saturating_sub(1)].max(0.0))
    }

    pub(super) fn benchmark_session_fallback_latency(
        &self,
        as_of: NaiveDate,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<(f64, f64)> {
        let mut benchmark = self.clone();
        benchmark.session_opener_cache = Arc::new(Mutex::new(HashMap::new()));
        benchmark.session_reply_cache = Arc::new(Mutex::new(HashMap::new()));
        benchmark.session_third_cache = Arc::new(Mutex::new(HashMap::new()));

        let cold_started = Instant::now();
        let _ = benchmark.session_root_guess(as_of, cancelled)?;
        let cold_ms = cold_started.elapsed().as_secs_f64() * 1_000.0;
        let warm_started = Instant::now();
        let _ = benchmark.session_root_guess(as_of, cancelled)?;
        let warm_ms = warm_started.elapsed().as_secs_f64() * 1_000.0;
        Ok((cold_ms.max(0.0), warm_ms.max(0.0)))
    }

    pub(super) fn four_guess_runs(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<(DetailedBacktestReport, Vec<DetailedSolveRun>)> {
        let backtest =
            self.backtest_detailed_with_book_usage(from, to, top, PredictiveBookUsage::None)?;
        let runs = backtest
            .runs
            .iter()
            .filter(|run| run.solved && run.steps.len() == 4)
            .cloned()
            .collect::<Vec<_>>();
        Ok((backtest, runs))
    }

    pub(super) fn best_three_guess_attempt_for_target(
        &self,
        target: &str,
        date: NaiveDate,
        top: usize,
        root_candidate_limit: usize,
        reply_candidate_limit: usize,
    ) -> Result<DetailedSolveRun> {
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let root = self.initial_state(as_of);
        let root_batch = self.suggestion_batch_internal(
            &root,
            root_candidate_limit.max(top),
            Some(PredictiveContext {
                hard_mode: false,
                as_of,
                observations: &[],
            }),
            PredictiveBookUsage::None,
        )?;
        let mut best = self.solve_target_from_state_detailed(
            target,
            as_of,
            date,
            top,
            PredictiveBookUsage::None,
        )?;

        for opener in root_batch
            .suggestions
            .iter()
            .take(root_candidate_limit.max(top))
            .map(|suggestion| suggestion.word.clone())
        {
            let opener_run =
                self.solve_target_with_forced_opening(target, as_of, date, &opener, top)?;
            if better_targeted_run(&opener_run, &best) {
                best = opener_run.clone();
            }
            if opener_run.solved && opener_run.steps.len() <= 3 {
                return Ok(opener_run);
            }

            let opener_feedback = score_guess(&opener, target);
            if opener_feedback == ALL_GREEN_PATTERN {
                continue;
            }
            let mut child = root.clone();
            self.apply_feedback(&mut child, &opener, opener_feedback)?;
            let observations = [(opener.clone(), opener_feedback)];
            let reply_batch = self.suggestion_batch_internal(
                &child,
                reply_candidate_limit.max(top),
                Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &observations,
                }),
                PredictiveBookUsage::None,
            )?;
            for reply in reply_batch
                .suggestions
                .iter()
                .take(reply_candidate_limit.max(top))
                .map(|suggestion| suggestion.word.clone())
            {
                let forced = [(opener.clone(), opener_feedback), (reply, 0)];
                let run =
                    self.solve_target_with_forced_prefix(target, as_of, date, &forced, top)?;
                if better_targeted_run(&run, &best) {
                    best = run.clone();
                }
                if run.solved && run.steps.len() <= 3 {
                    return Ok(run);
                }
            }
        }

        Ok(best)
    }

    #[cfg(test)]
    pub(super) fn medium_second_guess_coverage(
        &self,
        subset: &[usize],
        weights: &[f64],
        metrics: &[GuessMetrics],
    ) -> Result<FxHashMap<usize, ThreeSolveCoverage>> {
        self.medium_second_guess_coverage_controlled(subset, weights, metrics, &|| false)
    }

    pub(super) fn medium_second_guess_coverage_controlled(
        &self,
        subset: &[usize],
        weights: &[f64],
        metrics: &[GuessMetrics],
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<FxHashMap<usize, ThreeSolveCoverage>> {
        check_predictive_search_cancelled(cancelled)?;
        let limit = metrics.len().min(self.config.second_guess_coverage_pool);
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut coverage = FxHashMap::default();
        for metric in metrics.iter().take(limit) {
            check_predictive_search_cancelled(cancelled)?;
            coverage.insert(
                metric.guess_index,
                self.three_solve_coverage_for_guess_controlled(
                    metric.guess_index,
                    subset,
                    weights,
                    total_weight,
                    cancelled,
                )?,
            );
        }
        check_predictive_search_cancelled(cancelled)?;
        Ok(coverage)
    }

    /// At turn two, one guess remains after this move to finish by total turn three.
    #[cfg(test)]
    pub(super) fn three_solve_coverage_for_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        total_weight: f64,
    ) -> Result<ThreeSolveCoverage> {
        self.three_solve_coverage_for_guess_controlled(
            guess_index,
            subset,
            weights,
            total_weight,
            &|| false,
        )
    }

    fn three_solve_coverage_for_guess_controlled(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        total_weight: f64,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<ThreeSolveCoverage> {
        check_predictive_search_cancelled(cancelled)?;
        if !total_weight.is_finite() || total_weight <= 0.0 {
            bail!("coverage requires finite positive answer mass");
        }
        let mut largest = [0.0_f64; PATTERN_SPACE];
        let mut counts = [0usize; PATTERN_SPACE];
        for (position, &answer_index) in subset.iter().enumerate() {
            if position % 64 == 0 {
                check_predictive_search_cancelled(cancelled)?;
            }
            let pattern = self.answer_pattern(guess_index, answer_index) as usize;
            let weight = weights[answer_index];
            if !weight.is_finite() || weight < 0.0 {
                bail!("coverage requires finite non-negative weights");
            }
            if weight > 0.0 {
                largest[pattern] = largest[pattern].max(weight);
                counts[pattern] += 1;
            }
        }
        let mut result = ThreeSolveCoverage {
            mass: largest.iter().sum::<f64>() / total_weight,
            ..ThreeSolveCoverage::default()
        };
        for count in counts {
            if count > 1 {
                result.uncovered_buckets += 1;
                result.uncovered_answers += count - 1;
            }
        }
        check_predictive_search_cancelled(cancelled)?;
        Ok(result)
    }

    pub(super) fn evaluate_named_opener_on_targets(
        &self,
        targets: &[(NaiveDate, String)],
        opener: &str,
        top: usize,
    ) -> Result<FourGuessOpenerEvaluation> {
        if targets.is_empty() {
            bail!("forced evaluation requires at least one target");
        }

        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut failures = 0usize;
        let mut three_guess_solves = 0usize;
        for (date, target) in targets {
            let as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate opener before launch date"))?;
            let run = self.solve_target_with_forced_opening(target, as_of, *date, opener, top)?;
            guess_counts.push(run.steps.len());
            failures += usize::from(!run.solved);
            three_guess_solves += usize::from(run.solved && run.steps.len() <= 3);
        }
        guess_counts.sort_unstable();
        let average_guesses = guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64;
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(FourGuessOpenerEvaluation {
            opener: opener.to_string(),
            average_guesses,
            three_guess_solves,
            failures,
            p95_guesses: guess_counts[p95_index - 1],
            max_guesses: guess_counts[guess_counts.len() - 1],
        })
    }

    pub(super) fn evaluate_tuning_candidate(
        paths: &ProjectPaths,
        config: &PriorConfig,
        plan: &EvaluationPlan,
    ) -> Result<TuningEvaluation> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let dates = plan.validation_target_dates()?;
        for fold in &plan.folds {
            validate_exact_date_coverage(
                fold.validation,
                solver.history_dates.iter().map(|entry| entry.print_date),
            )?;
        }
        let games = solver
            .history_dates
            .iter()
            .filter(|entry| dates.contains(&entry.print_date))
            .collect::<Vec<_>>();
        let report = solver.experiment_report_for_selected_games_with_book_usage_and_progress(
            &games,
            5,
            PredictiveBookUsage::None,
            None,
        )?;
        let to = *dates
            .last()
            .ok_or_else(|| anyhow!("tuning validation dates are empty"))?;
        let hard_cases =
            solver.hard_case_report_with_book_usage(to, 5, PredictiveBookUsage::None)?;
        Ok(TuningEvaluation {
            config: config.clone(),
            scheduled_games: report.backtest.canonical.scheduled_games,
            measured_prior_games: report
                .prior_evidence
                .as_ref()
                .map_or(0, |prior| prior.measured_games),
            average_guesses: report.backtest.average_guesses,
            all_game_penalized_mean_guesses: report
                .backtest
                .canonical
                .all_game_penalized_mean_guesses,
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
            finite_step_pct: report.finite_step_pct,
            terminal_step_pct: report.terminal_step_pct,
        })
    }

    pub(super) fn offline_book_solver(&self) -> Result<Self> {
        let config = apply_embedded_profile(
            &self.config,
            include_str!("../../config/profiles/offline-book.json"),
        )?;
        Ok(self.clone_with_config(config))
    }

    pub(super) fn clone_with_config(&self, config: PriorConfig) -> Self {
        let mut cloned = self.clone();
        cloned.config = config.clone();
        cloned
    }

    pub(super) fn is_medium_state_lookahead(&self, surviving_answers: usize) -> bool {
        surviving_answers > self.config.exact_threshold
            && surviving_answers <= self.config.medium_state_lookahead_threshold
    }

    pub(super) fn lookahead_candidate_pool_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_lookahead_candidate_pool
        } else {
            self.config.lookahead_candidate_pool
        }
    }

    pub(super) fn lookahead_reply_pool_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_lookahead_reply_pool
        } else {
            self.config.lookahead_reply_pool
        }
    }

    pub(super) fn force_in_two_scan_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_force_in_two_scan
        } else {
            self.config.lookahead_root_force_in_two_scan
        }
    }

    pub(super) fn recent_history_targets_for_books(
        &self,
        as_of: NaiveDate,
    ) -> Result<BookTargetWindow> {
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

    pub(super) fn previous_history_targets_for_books(
        &self,
        current_window_start: NaiveDate,
    ) -> Result<Option<BookTargetWindow>> {
        let mut entries = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date < current_window_start)
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Ok(None);
        }
        entries.sort_by_key(|entry| entry.print_date);
        let holdout_end = entries
            .last()
            .map(|entry| entry.print_date)
            .ok_or_else(|| anyhow!("missing holdout history"))?;
        let window_days = self.config.session_window_days.saturating_sub(1) as u64;
        let holdout_start = holdout_end
            .checked_sub_days(Days::new(window_days))
            .map_or(entries[0].print_date, |date| {
                date.max(entries[0].print_date)
            });
        let targets = entries
            .into_iter()
            .filter(|entry| entry.print_date >= holdout_start)
            .map(|entry| (entry.print_date, entry.solution.clone()))
            .collect::<Vec<_>>();
        Ok(Some((holdout_start, holdout_end, targets)))
    }

    pub(super) fn evaluate_forced_opener(
        &self,
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<ForcedOpenerEvaluation> {
        if targets.is_empty() {
            bail!("forced evaluation requires at least one target");
        }

        let opener = self.guesses[guess_index].clone();
        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut four_guess_games = 0usize;
        let mut failures = 0usize;
        for (date, target) in targets {
            let target_as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate opener before launch date"))?;
            let score = self.score_target_with_forced_prefix_controlled(
                target,
                target_as_of,
                *date,
                &[(opener.clone(), 0)],
                cancelled,
            )?;
            if score.guesses >= 4 {
                four_guess_games += 1;
            }
            guess_counts.push(score.guesses);
            if !score.solved {
                failures += 1;
            }
        }
        guess_counts.sort_unstable();
        let average_guesses = guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64;
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(ForcedOpenerEvaluation {
            guess_index,
            games: guess_counts.len(),
            four_guess_games,
            average_guesses,
            p95_guesses: guess_counts[p95_index - 1],
            max_guesses: guess_counts[guess_counts.len() - 1],
            failures,
        })
    }

    pub(super) fn evaluate_forced_continuation(
        &self,
        forced_prefix: &[String],
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<ForcedOpenerEvaluation> {
        if targets.is_empty() {
            bail!("forced evaluation requires at least one target");
        }

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
            let score = self.score_target_with_forced_prefix_controlled(
                target,
                target_as_of,
                *date,
                &forced,
                cancelled,
            )?;
            guess_counts.push(score.guesses);
            if !score.solved {
                failures += 1;
            }
        }
        guess_counts.sort_unstable();
        let average_guesses = guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64;
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(ForcedOpenerEvaluation {
            guess_index,
            games: guess_counts.len(),
            four_guess_games: guess_counts.iter().filter(|count| **count >= 4).count(),
            average_guesses,
            p95_guesses: guess_counts[p95_index - 1],
            max_guesses: guess_counts[guess_counts.len() - 1],
            failures,
        })
    }

    pub(super) fn select_validated_opener(
        &self,
        as_of: NaiveDate,
        candidates: &[Suggestion],
        primary_targets: &[(NaiveDate, String)],
        holdout_targets: Option<&[(NaiveDate, String)]>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<ValidatedOpenerEvaluation>> {
        let mut evaluations = candidates
            .par_iter()
            .map(|suggestion| {
                let guess_index =
                    self.guess_index
                        .get(&suggestion.word)
                        .copied()
                        .ok_or_else(|| {
                            anyhow!(
                                "book primary candidate {} at {as_of} is not a legal guess",
                                suggestion.word
                            )
                        })?;
                let primary = self
                    .evaluate_forced_opener(primary_targets, guess_index, cancelled)
                    .with_context(|| {
                        format!("evaluate primary opener {} at {as_of}", suggestion.word)
                    })?;
                Ok(ValidatedOpenerEvaluation {
                    word: suggestion.word.clone(),
                    primary,
                    holdout: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        evaluations.sort_by(|left, right| {
            compare_forced_openers(&left.primary, &right.primary, &self.guesses)
        });
        let shortlist_len = if holdout_targets.is_some() {
            self.config
                .session_opener_holdout_shortlist
                .min(evaluations.len())
        } else {
            evaluations.len()
        };
        let mut best: Option<ValidatedOpenerEvaluation> = None;
        for mut evaluation in evaluations.into_iter().take(shortlist_len) {
            if let Some(targets) = holdout_targets {
                evaluation.holdout = Some(
                    self.evaluate_forced_opener(targets, evaluation.primary.guess_index, cancelled)
                        .with_context(|| {
                            format!(
                                "evaluate required holdout for opener {} at {as_of}",
                                evaluation.word
                            )
                        })?,
                );
            }
            if best.as_ref().is_none_or(|current| {
                should_replace_forced_opener(
                    &evaluation.primary,
                    evaluation.holdout.as_ref(),
                    &current.primary,
                    current.holdout.as_ref(),
                    &self.guesses,
                )
            }) {
                best = Some(evaluation);
            }
        }
        Ok(best)
    }

    pub(super) fn solve_target_with_forced_opening(
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

    pub(super) fn solve_target_with_forced_prefix(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        forced: &[(String, u8)],
        top: usize,
    ) -> Result<DetailedSolveRun> {
        self.solve_target_from_initial_state_detailed(
            target,
            as_of,
            date,
            top,
            self.initial_state(as_of),
            SolveExecutionPolicy {
                cancelled: &|| false,
                book_usage: PredictiveBookUsage::None,
                search_mode: None,
                forced,
            },
        )
    }

    #[cfg(test)]
    pub(super) fn score_target_with_forced_prefix(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        forced: &[(String, u8)],
    ) -> Result<ForcedSolveScore> {
        self.score_target_with_forced_prefix_controlled(target, as_of, date, forced, &|| false)
    }

    fn score_target_with_forced_prefix_controlled(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        forced: &[(String, u8)],
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<ForcedSolveScore> {
        let run = self.solve_target_from_initial_state_detailed(
            target,
            as_of,
            date,
            1,
            self.initial_state(as_of),
            SolveExecutionPolicy {
                cancelled,
                book_usage: PredictiveBookUsage::None,
                search_mode: None,
                forced,
            },
        )?;
        Ok(ForcedSolveScore {
            guesses: run.steps.len(),
            solved: run.solved,
        })
    }
}

fn canonical_development_evaluation_plan(
    paths: &ProjectPaths,
    operation: &str,
) -> Result<EvaluationPlan> {
    Solver::development_evaluation_plan(paths)
        .with_context(|| format!("cannot build development plan for {operation}"))
}

fn ensure_development_target_range(
    paths: &ProjectPaths,
    from: NaiveDate,
    to: NaiveDate,
    operation: &str,
) -> Result<EvaluationPlan> {
    let plan = canonical_development_evaluation_plan(paths, operation)?;
    let policy =
        crate::experiments::EvaluationPolicy::load(&paths.root.join("config/evaluation.toml"))?;
    let requested = DateRange::new(from, to)?;
    if from < plan.development.start || to > plan.development.end {
        bail!(
            "{operation} range {from}..{to} is outside declared development {}..{}",
            plan.development.start,
            plan.development.end
        );
    }
    policy
        .validate_development_target_range(requested)
        .with_context(|| format!("cannot validate {operation} target range"))?;
    Ok(plan)
}

fn rolling_origin_config_for_history(history: DateRange) -> Result<RollingOriginConfig> {
    if history.days() >= 425 {
        return Ok(RollingOriginConfig::default());
    }
    if history.days() < 3 {
        bail!(
            "history has {} days but development evaluation requires at least 3",
            history.days()
        );
    }
    Ok(RollingOriginConfig {
        minimum_training_days: history.days() - 2,
        validation_days: 1,
        step_days: 1,
        sealed_test_days: 1,
        maximum_folds: 1,
    })
}

fn rolling_checkpoint_path(
    paths: &ProjectPaths,
    label: &str,
    config_toml: &str,
    source_identity: &str,
    top: usize,
) -> PathBuf {
    let safe_label = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let fingerprint = rolling_checkpoint_fingerprint_with_top(config_toml, source_identity, top);
    paths.root.join(format!(
        "target/rolling-checkpoints/{safe_label}-{fingerprint}.json"
    ))
}

#[cfg(test)]
pub(super) fn rolling_checkpoint_fingerprint(config_toml: &str, source_identity: &str) -> String {
    rolling_checkpoint_fingerprint_with_top(config_toml, source_identity, 0)
}

fn rolling_checkpoint_fingerprint_with_top(
    config_toml: &str,
    source_identity: &str,
    top: usize,
) -> String {
    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-rolling-checkpoint-v3");
    hash.field(config_toml.as_bytes())
        .field(source_identity.as_bytes())
        .field(&(top as u64).to_le_bytes());
    hash.finish_hex()
}

fn evidence_artifact_sizes(paths: &ProjectPaths) -> Result<Vec<EvidenceArtifactSize>> {
    let declared = [
        ("pattern_table", paths.pattern_table.as_path()),
        ("answer_history", paths.derived_answer_history.as_path()),
        ("modeled_answers", paths.derived_modeled_answers.as_path()),
        ("predictive_books", paths.derived_predictive.as_path()),
    ];
    declared
        .into_iter()
        .filter(|(_, path)| path.exists())
        .map(|(name, path)| {
            Ok(EvidenceArtifactSize {
                name: name.to_string(),
                path: path
                    .strip_prefix(&paths.root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .replace('\\', "/"),
                bytes: filesystem_tree_bytes(path)?,
            })
        })
        .collect()
}

fn cumulative_evidence_elapsed_ms(started: Instant, prior_elapsed_ms: u64) -> u64 {
    prior_elapsed_ms.saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64)
}

fn enforce_evidence_resource_budget(
    started: Instant,
    prior_elapsed_ms: u64,
    prior_peak_working_set_bytes: u64,
    budget: EvidenceResourceBudget,
) -> Result<crate::process_memory::ProcessMemorySnapshot> {
    let elapsed_ms = cumulative_evidence_elapsed_ms(started, prior_elapsed_ms);
    if elapsed_ms > budget.maximum_seconds.saturating_mul(1_000) {
        bail!(
            "evidence generation took {:.3} seconds and exceeded the {} second budget",
            elapsed_ms as f64 / 1_000.0,
            budget.maximum_seconds
        );
    }
    let memory = crate::process_memory::process_memory_snapshot().ok_or_else(|| {
        anyhow!(
            "hard evidence memory budgets are unsupported on this platform; supported platforms are Windows, Linux, and macOS"
        )
    })?;
    let memory_budget_bytes = budget.maximum_memory_mb.saturating_mul(1024 * 1024);
    let peak_working_set_bytes = prior_peak_working_set_bytes.max(memory.peak_working_set_bytes);
    if peak_working_set_bytes > memory_budget_bytes {
        bail!(
            "evidence peak working set {} MiB exceeded the {} MiB budget",
            peak_working_set_bytes.div_ceil(1024 * 1024),
            budget.maximum_memory_mb
        );
    }
    Ok(memory)
}

// Keep each resume-identity input explicit; grouping them would hide which
// changing source invalidates a partial evidence checkpoint.
#[allow(clippy::too_many_arguments)]
fn evidence_checkpoint_identity(
    input_fingerprint: &str,
    config_toml: &str,
    plan: &EvaluationPlan,
    selection_label: &str,
    selected_ranges: &[DateRange],
    matrix_source: &str,
    matrix_fingerprint: &str,
    profile_ids: &[String],
    resolved_profile_base_configs: &[(String, String)],
    dates: DateRange,
    top: usize,
    resource_budget: EvidenceResourceBudget,
    rayon_threads: usize,
) -> Result<String> {
    // Disk artifacts are validated by the solver when they are used and are reported in the
    // baseline policy. Their mutable bytes are not available as a single checkpoint input, so
    // this identity deliberately makes no authentication claim about those artifacts.
    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-evidence-checkpoint-v4");
    hash.field(input_fingerprint.as_bytes())
        .field(config_toml.as_bytes())
        .field(&serde_json::to_vec(plan)?)
        .field(selection_label.as_bytes())
        .field(&serde_json::to_vec(selected_ranges)?)
        .field(matrix_source.as_bytes())
        .field(matrix_fingerprint.as_bytes())
        .field(&serde_json::to_vec(profile_ids)?)
        .field(&(resolved_profile_base_configs.len() as u64).to_le_bytes());
    for (profile_id, base_config_source) in resolved_profile_base_configs {
        hash.field(profile_id.as_bytes())
            .field(base_config_source.as_bytes());
    }
    hash.field(dates.start.to_string().as_bytes())
        .field(dates.end.to_string().as_bytes())
        .field(&(top as u64).to_le_bytes())
        .field(&resource_budget.maximum_seconds.to_le_bytes())
        .field(&resource_budget.maximum_memory_mb.to_le_bytes())
        .field(&(rayon_threads as u64).to_le_bytes());
    Ok(hash.finish_tagged())
}

fn load_evidence_matrix(
    paths: &ProjectPaths,
    matrix_path: Option<&Path>,
) -> Result<(PredictiveExperimentMatrix, String, String)> {
    let (source, source_label) = match matrix_path {
        Some(path) => {
            let resolved = if path.is_absolute() {
                path.to_path_buf()
            } else {
                paths.root.join(path)
            };
            let source = fs::read_to_string(&resolved)
                .with_context(|| format!("read evidence matrix {}", resolved.display()))?;
            let label = path
                .strip_prefix(&paths.root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            (source, label)
        }
        None => (
            include_str!("../../config/experiments/development-evidence.json").to_string(),
            "config/experiments/development-evidence.json".to_string(),
        ),
    };
    let matrix = PredictiveExperimentMatrix::parse_json(&source)
        .with_context(|| format!("parse evidence matrix {source_label}"))?;
    let fingerprint =
        crate::identity::digest_bytes_tagged("maybe-wordle-evidence-matrix-v1", source.as_bytes());
    Ok((matrix, source_label, fingerprint))
}

fn selected_evidence_games<'a>(
    history: &'a [NytDailyEntry],
    selected_ranges: &[DateRange],
) -> Result<Vec<&'a NytDailyEntry>> {
    if selected_ranges.is_empty() {
        bail!("evidence selection has no date ranges");
    }
    for (index, range) in selected_ranges.iter().enumerate() {
        DateRange::new(range.start, range.end)?;
        if index > 0 && selected_ranges[index - 1].end >= range.start {
            bail!("evidence selection ranges must be sorted and non-overlapping");
        }
    }
    let mut games = history
        .iter()
        .filter(|entry| {
            selected_ranges
                .iter()
                .any(|range| range.contains(entry.print_date))
        })
        .collect::<Vec<_>>();
    games.sort_by_key(|entry| entry.print_date);
    if games.is_empty() {
        bail!("no games found in the selected evidence dates");
    }
    Ok(games)
}

fn resolved_evidence_profile_base_configs(
    paths: &ProjectPaths,
    fallback: &PriorConfig,
    matrix: &PredictiveExperimentMatrix,
) -> Result<Vec<(String, String)>> {
    matrix
        .profiles
        .iter()
        .map(|profile| {
            let base = profile.load_base_config(&paths.root, fallback)?;
            let source = profile
                .base_config_path
                .as_deref()
                .map(|relative| {
                    fs::read_to_string(paths.root.join(relative))
                        .with_context(|| format!("read evidence profile base config {relative}"))
                })
                .transpose()?
                .unwrap_or(
                    toml::to_string_pretty(&base)
                        .context("failed to serialize evidence profile fallback base config")?,
                );
            Ok((profile.id.clone(), source))
        })
        .collect()
}

fn filesystem_tree_bytes(path: &Path) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect artifact size for {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Ok(0);
    }
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }
    let mut bytes = 0u64;
    for entry in fs::read_dir(path)
        .with_context(|| format!("failed to enumerate artifact directory {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", path.display()))?;
        bytes = bytes.saturating_add(filesystem_tree_bytes(&entry.path())?);
    }
    Ok(bytes)
}

pub(super) fn development_source_identity(
    paths: &ProjectPaths,
    development_cutoff: NaiveDate,
) -> Result<String> {
    rolling_source_identity_with_history_cutoff(paths, development_cutoff)
}

fn rolling_source_identity_with_history_cutoff(
    paths: &ProjectPaths,
    development_cutoff: NaiveDate,
) -> Result<String> {
    let mut files = Vec::new();
    collect_regular_files(&paths.root.join("src"), &mut files)?;
    collect_regular_files(&paths.root.join("tests"), &mut files)?;
    files.extend([
        paths.root.join("Cargo.toml"),
        paths.root.join("Cargo.lock"),
        paths.root.join("config/evaluation.toml"),
        paths.raw_history.clone(),
        paths.seed_guesses.clone(),
        paths.seed_answers.clone(),
        paths.seed_reference_answers.clone(),
        paths.manual_additions.clone(),
    ]);
    files.sort();
    files.dedup();

    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-rolling-inputs-v3");
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    hash.field(b"current_executable");
    hash_identity_file(&mut hash, &executable)?;
    for path in files {
        let relative = path.strip_prefix(&paths.root).unwrap_or(&path);
        let relative = relative.to_string_lossy().replace('\\', "/");
        hash.field(relative.as_bytes());
        if path.is_file() {
            hash.field(&[1]);
            if path == paths.raw_history {
                let cutoff = development_cutoff;
                let history = read_history_jsonl(&path)?
                    .into_iter()
                    .filter(|entry| entry.print_date <= cutoff)
                    .collect::<Vec<_>>();
                if history.last().map(|entry| entry.print_date) != Some(cutoff) {
                    bail!("history does not contain the declared development cutoff {cutoff}");
                }
                let mut canonical_history = Vec::new();
                for entry in history {
                    serde_json::to_writer(&mut canonical_history, &entry)?;
                    canonical_history.push(b'\n');
                }
                hash.field(&canonical_history);
            } else {
                hash_identity_file(&mut hash, &path)?;
            }
        } else {
            hash.field(&[0]);
        }
    }
    Ok(hash.finish_tagged())
}

pub(super) fn current_executable_fingerprint() -> Result<String> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-executable-v1");
    hash_identity_file(&mut hash, &executable)?;
    Ok(hash.finish_tagged())
}

fn hash_identity_file(hash: &mut crate::identity::CanonicalSha256, path: &Path) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to inspect {}", path.display()))?;
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    hash.field_reader(&mut file, metadata.len())
        .with_context(|| format!("failed to fingerprint {}", path.display()))?;
    Ok(())
}

pub(super) fn ensure_development_source_identity(
    paths: &ProjectPaths,
    development_cutoff: NaiveDate,
    expected: &str,
) -> Result<()> {
    if development_source_identity(paths, development_cutoff)? != expected {
        bail!(
            "source, executable, or declared development inputs changed during evaluation; discard this run and restart from a consistent snapshot"
        );
    }
    Ok(())
}

fn collect_regular_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to enumerate {}", directory.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", directory.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_regular_files(&path, output)?;
        } else if path.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

fn merge_execution_telemetry(target: &mut ExecutionTelemetry, addition: &ExecutionTelemetry) {
    target.total_steps += addition.total_steps;
    target.proxy_steps += addition.proxy_steps;
    target.lookahead_steps += addition.lookahead_steps;
    target.escalated_exact_steps += addition.escalated_exact_steps;
    target.exact_steps += addition.exact_steps;
    target.finite_steps += addition.finite_steps;
    target.terminal_steps += addition.terminal_steps;
    target.danger_escalated_steps += addition.danger_escalated_steps;
    target.strict_recovery_steps += addition.strict_recovery_steps;
    target.uniform_recovery_steps += addition.uniform_recovery_steps;
    target.epsilon_repair_steps += addition.epsilon_repair_steps;
    target.dormant_fallback_steps += addition.dormant_fallback_steps;
    target.exact_date_opener_artifact_hits += addition.exact_date_opener_artifact_hits;
    target.recent_opener_artifact_hits += addition.recent_opener_artifact_hits;
    target.reply_book_hits += addition.reply_book_hits;
    target.session_fallback_hits += addition.session_fallback_hits;
}

fn git_provenance(root: &Path) -> (Option<String>, Option<bool>) {
    let revision = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let dirty = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty());
    (revision, dirty)
}

#[cfg(test)]
mod tests {
    fn parity_solver(
        mode: crate::config::SearchPolicyMode,
        date: NaiveDate,
    ) -> (crate::test_support::TestDirectory, Solver) {
        let directory = crate::test_support::TestDirectory::new("policy-parity");
        let paths = ProjectPaths::new(directory.path());
        paths.ensure_layout().unwrap();
        for path in [&paths.seed_guesses, &paths.seed_answers] {
            fs::write(path, "cigar\nrebut\nsissy\nhumph\nawake\nblush\n").unwrap();
        }
        for path in [
            &paths.seed_reference_answers,
            &paths.seed_sources,
            &paths.manual_additions,
            &paths.raw_history,
        ] {
            fs::write(path, "").unwrap();
        }
        crate::data::write_history_jsonl(
            &paths.raw_history,
            &[NytDailyEntry {
                id: None,
                solution: "humph".to_string(),
                print_date: date,
                days_since_launch: None,
                editor: None,
            }],
        )
        .unwrap();
        let config = PriorConfig {
            search_policy_mode: mode,
            ..PriorConfig::default()
        };
        let solver = Solver::from_paths(&paths, &config).unwrap();
        (directory, solver)
    }

    #[test]
    fn public_live_and_normal_replay_preserve_policy_roots_and_beliefs() {
        use crate::config::SearchPolicyMode;

        let date = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        let cutoff = crate::predictive::history_cutoff(date).unwrap();
        for mode in [
            SearchPolicyMode::ProxyOnly,
            SearchPolicyMode::Staged,
            SearchPolicyMode::FiniteBaseline,
        ] {
            let (_directory, solver) = parity_solver(mode, date);
            let replay = solver
                .backtest_detailed(date, date, 3)
                .unwrap()
                .runs
                .into_iter()
                .next()
                .unwrap();
            assert!(replay.solved);
            let mut observations = Vec::new();
            let mut incremental = solver.initial_state(cutoff);
            for step in &replay.steps {
                let live = solver
                    .suggest_predictive(PredictiveSuggestRequest {
                        puzzle_date: date,
                        observations: &observations,
                        top: 3,
                        hard_mode: false,
                        force_in_two_only: false,
                        mode: PredictiveSuggestionMode::LiveOnly,
                    })
                    .unwrap();
                assert!(live.promotion_source.is_none() && step.promotion_source.is_none());
                assert_eq!(live.suggestions[0].word, step.guess, "{mode:?}");
                assert_eq!(live.state.surviving, step.surviving_before);
                assert_eq!(live.state.effective_total_weight, incremental.total_weight);
                assert_eq!(live.execution.route, step.regime_used);
                assert!(solver.has_guess(&step.guess));
                assert_eq!(score_guess(&step.guess, "humph"), step.feedback);
                solver
                    .apply_feedback(&mut incremental, &step.guess, step.feedback)
                    .unwrap();
                observations.push((step.guess.clone(), step.feedback));
                let rebuilt = solver
                    .validate_game_history(date, &observations, false)
                    .unwrap();
                assert_eq!(incremental.surviving, rebuilt.surviving);
                assert_eq!(incremental.weights, rebuilt.weights);
                assert_eq!(rebuilt.surviving.len(), step.surviving_after);
            }
        }
    }

    #[test]
    fn public_live_and_shared_replay_selection_match_normal_and_hard_terminal_rules() {
        use crate::config::SearchPolicyMode;

        let date = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        let cutoff = crate::predictive::history_cutoff(date).unwrap();
        for mode in [
            SearchPolicyMode::ProxyOnly,
            SearchPolicyMode::Staged,
            SearchPolicyMode::FiniteBaseline,
        ] {
            let (_directory, solver) = parity_solver(mode, date);
            for hard_mode in [false, true] {
                let mut observations = Vec::new();
                for _ in 0..6 {
                    let request = PredictiveSuggestRequest {
                        puzzle_date: date,
                        observations: &observations,
                        top: 3,
                        hard_mode,
                        force_in_two_only: false,
                        mode: PredictiveSuggestionMode::LiveOnly,
                    };
                    let live = solver.suggest_predictive(request).unwrap();
                    let state = solver
                        .validate_game_history(date, &observations, hard_mode)
                        .unwrap();
                    assert_eq!(live.state.surviving, state.surviving.len());
                    assert_eq!(live.state.effective_total_weight, state.total_weight);
                    let replay = if mode.is_finite() {
                        solver
                            .finite_suggestion_batch(
                                &state,
                                3,
                                Some(PredictiveContext {
                                    as_of: cutoff,
                                    observations: &observations,
                                    hard_mode,
                                }),
                                solver.finite_search_options(),
                                &|| false,
                            )
                            .unwrap()
                    } else {
                        solver
                            .filtered_suggestion_batch_for_history_with_search_mode_controlled(
                                cutoff,
                                &observations,
                                3,
                                PredictiveSuggestionFilters {
                                    mode: PredictiveSuggestionMode::LiveOnly,
                                    hard_mode,
                                    force_in_two_only: false,
                                    forced_search_mode: None,
                                },
                                &|| false,
                            )
                            .unwrap()
                    };
                    assert_eq!(
                        live.suggestions
                            .iter()
                            .map(|row| &row.word)
                            .collect::<Vec<_>>(),
                        replay
                            .suggestions
                            .iter()
                            .map(|row| &row.word)
                            .collect::<Vec<_>>()
                    );
                    let chosen = &live.suggestions[0].word;
                    assert!(solver.has_guess(chosen));
                    assert!(
                        !hard_mode || solver.hard_mode_violation(&observations, chosen).is_none()
                    );
                    let feedback = score_guess(chosen, "humph");
                    observations.push((chosen.clone(), feedback));
                    if feedback == ALL_GREEN_PATTERN {
                        break;
                    }
                }
                assert_eq!(observations.last().unwrap().1, ALL_GREEN_PATTERN);
                let terminal = solver
                    .suggest_predictive(PredictiveSuggestRequest {
                        puzzle_date: date,
                        observations: &observations,
                        top: 3,
                        hard_mode,
                        force_in_two_only: false,
                        mode: PredictiveSuggestionMode::LiveOnly,
                    })
                    .unwrap();
                assert!(
                    terminal.suggestions.is_empty(),
                    "solved {mode:?}, hard={hard_mode}"
                );
                observations.push(("humph".to_string(), ALL_GREEN_PATTERN));
                assert!(
                    solver
                        .validate_game_history(date, &observations, hard_mode)
                        .is_err()
                );
                assert!(
                    solver
                        .suggest_predictive(PredictiveSuggestRequest {
                            puzzle_date: date,
                            observations: &observations,
                            top: 3,
                            hard_mode,
                            force_in_two_only: false,
                            mode: PredictiveSuggestionMode::LiveOnly,
                        })
                        .is_err()
                );
                let exhausted = vec![("cigar".to_string(), score_guess("cigar", "humph")); 6];
                let terminal = solver
                    .suggest_predictive(PredictiveSuggestRequest {
                        puzzle_date: date,
                        observations: &exhausted,
                        top: 3,
                        hard_mode,
                        force_in_two_only: false,
                        mode: PredictiveSuggestionMode::LiveOnly,
                    })
                    .unwrap();
                assert!(
                    terminal.suggestions.is_empty(),
                    "exhausted {mode:?}, hard={hard_mode}"
                );
            }
        }
    }

    use super::sealed::{
        acquire_sealed_window, create_once_marker, create_prospective_window_marker,
        create_sealed_test_marker, ensure_prospective_window_elapsed,
        preflight_prospective_output_path, prospective_freeze_fingerprint,
        prospective_history_fingerprint, prospective_pre_window_history_fingerprint,
        prospective_pre_window_history_start, prospective_registry_marker_path,
        prospective_window_for_freeze, prospective_window_marker_path,
    };
    use super::study::needs_serial_study_latency;
    use std::{
        collections::HashMap,
        fs,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use chrono::{Days, NaiveDate, TimeZone, Utc};

    use super::*;
    use crate::{
        config::PriorConfig,
        data::NytDailyEntry,
        experiments::{EvaluationPolicy, RollingOriginFold, TrialStatus},
        model::{AnswerRecord, ModelVariant, WeightMode},
        pattern_table::PatternTable,
    };

    struct RecordingEvidenceTiming {
        enabled: bool,
        events: Vec<EvidenceTimingEvent>,
    }

    impl RecordingEvidenceTiming {
        fn new(enabled: bool) -> Self {
            Self {
                enabled,
                events: Vec::new(),
            }
        }
    }

    impl EvidenceTimingSink for RecordingEvidenceTiming {
        fn enabled(&self) -> bool {
            self.enabled
        }

        fn emit(&mut self, event: EvidenceTimingEvent) {
            self.events.push(event);
        }
    }

    #[test]
    fn evidence_timing_flag_is_explicitly_opt_in() {
        assert!(!evidence_timing_enabled(None));
        assert!(!evidence_timing_enabled(Some("")));
        assert!(!evidence_timing_enabled(Some("0")));
        assert!(!evidence_timing_enabled(Some("TRUE")));
        assert!(evidence_timing_enabled(Some("1")));
        assert!(evidence_timing_enabled(Some("true")));
    }

    #[test]
    fn evidence_timing_sink_suppresses_disabled_events() {
        let mut disabled = RecordingEvidenceTiming::new(false);
        record_evidence_timing(
            &mut disabled,
            Some("profile"),
            "solver_setup",
            Duration::from_millis(7),
        );
        assert!(disabled.events.is_empty());

        let mut enabled = RecordingEvidenceTiming::new(true);
        record_evidence_timing(
            &mut enabled,
            Some("profile"),
            "solver_setup",
            Duration::from_millis(7),
        );
        assert_eq!(
            enabled.events,
            vec![EvidenceTimingEvent {
                profile: Some("profile".to_string()),
                phase: "solver_setup",
                elapsed_ms: 7,
            }]
        );
        assert_eq!(enabled.events[0].elapsed_ms, 7);
    }

    #[test]
    fn search_timing_line_is_compact_and_word_free() {
        assert_eq!(
            format_search_timing(3, 42, PredictiveRegime::Lookahead, 17),
            "benchmark-evidence timing turn=3 survivors=42 regime=lookahead elapsed_ms=17"
        );
    }

    #[test]
    fn sealed_test_date_preflight_rejects_missing_and_duplicate_dates() {
        let range = DateRange::new(
            NaiveDate::from_ymd_opt(2026, 1, 1).expect("date"),
            NaiveDate::from_ymd_opt(2026, 1, 3).expect("date"),
        )
        .expect("range");
        let day_one = NaiveDate::from_ymd_opt(2026, 1, 1).expect("date");
        let day_two = NaiveDate::from_ymd_opt(2026, 1, 2).expect("date");
        let day_three = NaiveDate::from_ymd_opt(2026, 1, 3).expect("date");

        assert!(validate_exact_date_coverage(range, [day_one, day_three].into_iter()).is_err());
        assert!(validate_exact_date_coverage(range, [day_two, day_three].into_iter()).is_err());
        assert!(validate_exact_date_coverage(range, [day_one, day_two].into_iter()).is_err());
        assert!(
            validate_exact_date_coverage(range, [day_one, day_two, day_two, day_three].into_iter())
                .is_err()
        );
        validate_exact_date_coverage(range, [day_one, day_two, day_three].into_iter())
            .expect("one entry per inclusive date");
    }

    #[test]
    fn sealed_output_preflight_rejects_ledger_aliases_before_claiming_a_window() {
        let directory = crate::test_support::TestDirectory::new("sealed-output-paths");
        let root = directory.path();
        let window = DateRange::new(
            NaiveDate::from_ymd_opt(2032, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2032, 1, 3).unwrap(),
        )
        .unwrap();
        let ledger = root.join("benchmarks/predictive/sealed-windows");
        let marker = ledger.join("2032-01-01-2032-01-03-once.json");
        let paths = [
            marker.clone(),
            ledger.join("report.json"),
            ledger.join("nested/report.json"),
            ledger.join("registry"),
            ledger.join(".registry.mwedit.lock"),
            root.join("benchmarks/predictive/sealed-test-once.json"),
            root.join("benchmarks/predictive/prospective-window-once.json"),
            root.join("benchmarks/predictive/./sealed-windows/report.json"),
            root.join("benchmarks/predictive/other/../sealed-windows/report.json"),
        ];
        for output in paths {
            assert!(
                super::sealed::preflight_sealed_output_path(root, &output, window).is_err(),
                "must reject {} before acquiring the seal",
                output.display()
            );
            assert!(!marker.exists(), "path rejection must not consume the seal");
            assert!(
                !output.is_file(),
                "path rejection must not create a report or lock"
            );
        }
        for normal in [
            root.join("benchmarks/predictive/sealed-test-v1.json"),
            root.join("benchmarks/predictive/sealed-windows-copy/report.json"),
            root.join("benchmarks/predictive/結果.json"),
        ] {
            super::sealed::preflight_sealed_output_path(root, &normal, window).unwrap();
            assert!(!normal.exists());
        }
        #[cfg(windows)]
        {
            for path in [
                root.join("benchmarks/predictive/SEALED-TEST-ONCE.JSON"),
                root.join("benchmarks/predictive/sealed-test-once.json."),
                root.join("benchmarks/predictive/sealed-test-once.json:report"),
                fs::canonicalize(root)
                    .unwrap()
                    .join("benchmarks/predictive/sealed-windows/report.json"),
            ] {
                assert!(super::sealed::preflight_sealed_output_path(root, &path, window).is_err());
            }
        }
    }

    #[test]
    fn replay_cancellation_reaches_inner_search_and_does_not_emit_partial_games() {
        let solver =
            super::super::tests::test_solver(&["tower", "power", "bower", "rower", "sower"]);
        let date = NaiveDate::from_ymd_opt(2030, 3, 9).unwrap();
        let entry = NytDailyEntry {
            solution: "power".to_string(),
            print_date: date,
            id: None,
            days_since_launch: None,
            editor: None,
        };
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let stopped = || calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) > 8;
        let error = solver
            .backtest_selected_games_controlled(
                &[&entry],
                3,
                PredictiveBookUsage::None,
                None,
                &stopped,
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("cancelled"));
        assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 8);
        let resumed = solver
            .backtest_selected_games_controlled(
                &[&entry],
                3,
                PredictiveBookUsage::None,
                None,
                &|| false,
            )
            .unwrap();
        assert_eq!(resumed.summary.games, 1);
        assert!(resumed.runs[0].solved);
    }

    #[test]
    fn sealed_ownership_is_window_scoped_and_preserves_historical_records() {
        let directory = crate::test_support::TestDirectory::new("sealed-windows");
        let root = directory.path();
        fs::create_dir_all(root.join("benchmarks/predictive")).unwrap();
        let old = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2030, 1, 3).unwrap(),
        )
        .unwrap();
        let next = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 2, 1).unwrap(),
            NaiveDate::from_ymd_opt(2030, 2, 3).unwrap(),
        )
        .unwrap();
        let legacy_path = root.join("benchmarks/predictive/sealed-test-once.json");
        let legacy = br#"{"output_path":"benchmarks/predictive/old-report.json","freeze_fingerprint":"old","status":"completed"}"#;
        fs::write(&legacy_path, legacy).unwrap();
        fs::write(
            root.join("benchmarks/predictive/old-report.json"),
            serde_json::to_vec(&serde_json::json!({"evaluation_plan":{"sealed_test":old}}))
                .unwrap(),
        )
        .unwrap();
        let mut marker = SealedTestMarker {
            schema_version: 2,
            window: old,
            evaluation_contract_fingerprint: "contract".to_string(),
            window_data_fingerprint: "snapshot".to_string(),
            freeze_fingerprint: "candidate-a".to_string(),
            output_path: "new-report.json".to_string(),
            status: "started_irreversible".to_string(),
        };
        assert!(acquire_sealed_window(root, &marker).is_err());
        marker.window = next;
        let published = acquire_sealed_window(root, &marker).unwrap();
        assert!(published.exists());
        marker.freeze_fingerprint = "candidate-b".to_string();
        marker.evaluation_contract_fingerprint = "different-contract".to_string();
        assert!(
            acquire_sealed_window(root, &marker).is_err(),
            "another candidate or contract cannot reopen the same window"
        );
        marker.window.start = next.start.succ_opt().unwrap();
        assert!(
            acquire_sealed_window(root, &marker).is_err(),
            "shifting the endpoints cannot reopen outcomes"
        );
        assert_eq!(fs::read(&legacy_path).unwrap(), legacy);
    }

    #[test]
    fn sealed_window_claims_remain_exclusive_under_two_openers() {
        let directory = crate::test_support::TestDirectory::new("sealed-claim-race");
        let window = DateRange::new(
            NaiveDate::from_ymd_opt(2031, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2031, 1, 2).unwrap(),
        )
        .unwrap();
        let marker = SealedTestMarker {
            schema_version: 2,
            window,
            evaluation_contract_fingerprint: "contract".to_string(),
            window_data_fingerprint: "snapshot".to_string(),
            freeze_fingerprint: "a".to_string(),
            output_path: "report.json".to_string(),
            status: "started_irreversible".to_string(),
        };
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles = (0..2)
            .map(|index| {
                let root = directory.path().to_owned();
                let barrier = barrier.clone();
                let mut marker = marker.clone();
                marker.freeze_fingerprint = index.to_string();
                std::thread::spawn(move || {
                    barrier.wait();
                    acquire_sealed_window(&root, &marker)
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            handles
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .filter(Result::is_ok)
                .count(),
            1
        );
    }

    #[test]
    fn sealed_test_marker_acquisition_is_exclusive_under_race() {
        let root = std::env::current_dir()
            .expect("test working directory")
            .join("target")
            .join("test-scratch");
        fs::create_dir_all(&root).expect("test marker directory");
        let marker_path = root.join(format!(
            "maybe-wordle-sealed-marker-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let contenders = [b"first".to_vec(), b"second".to_vec()]
            .into_iter()
            .map(|contents| {
                let marker_path = marker_path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    create_sealed_test_marker(&marker_path, &contents)
                })
            })
            .collect::<Vec<_>>();
        let results = contenders
            .into_iter()
            .map(|contender| contender.join().expect("marker contender"))
            .collect::<Vec<_>>();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        let acquisition_error = results
            .iter()
            .find_map(|result| result.as_ref().err())
            .expect("losing contender must receive an error");
        assert!(acquisition_error.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::AlreadyExists)
        }));
        let marker_contents = fs::read(&marker_path).expect("winning marker remains");
        assert!(marker_contents == b"first" || marker_contents == b"second");
        fs::remove_file(marker_path).expect("remove test marker");
    }

    fn synthetic_prospective_policy() -> EvaluationPolicy {
        EvaluationPolicy {
            format_version: crate::experiments::EVALUATION_POLICY_FORMAT_VERSION,
            development_cutoff: NaiveDate::from_ymd_opt(2030, 1, 1).expect("date"),
            sealed_test: DateRange::new(
                NaiveDate::from_ymd_opt(2030, 1, 3).expect("date"),
                NaiveDate::from_ymd_opt(2030, 2, 1).expect("date"),
            )
            .expect("seal"),
            excluded_validation: vec![
                DateRange::new(
                    NaiveDate::from_ymd_opt(2029, 11, 1).expect("date"),
                    NaiveDate::from_ymd_opt(2029, 11, 30).expect("date"),
                )
                .expect("consumed range"),
            ],
        }
    }

    #[test]
    fn prospective_window_uses_utc_freeze_date_and_requires_complete_future_range() {
        let policy = synthetic_prospective_policy();
        let frozen_at = Utc
            .with_ymd_and_hms(2030, 2, 2, 23, 59, 59)
            .single()
            .expect("timestamp");
        let window = prospective_window_for_freeze(&policy, frozen_at, [].into_iter())
            .expect("later window");
        assert_eq!(
            window.start,
            NaiveDate::from_ymd_opt(2030, 2, 3).expect("date")
        );
        assert_eq!(window.days(), PROSPECTIVE_WINDOW_DAYS);
        assert!(window.start > frozen_at.date_naive());
        assert!(window.start > policy.sealed_test.end);

        let error = prospective_window_for_freeze(&policy, frozen_at, [window.start].into_iter())
            .expect_err("existing target date must block freezing");
        assert!(
            error.to_string().contains("history already has"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn prospective_window_rejects_freeze_before_declared_seal_or_consumed_range() {
        let policy = synthetic_prospective_policy();
        let before_seal = Utc
            .with_ymd_and_hms(2030, 1, 2, 12, 0, 0)
            .single()
            .expect("timestamp");
        let error = prospective_window_for_freeze(&policy, before_seal, [].into_iter())
            .expect_err("window overlapping declared seal");
        assert!(error.to_string().contains("declared sealed test"));

        let mut consumed_policy = policy;
        consumed_policy.excluded_validation = vec![
            DateRange::new(
                NaiveDate::from_ymd_opt(2030, 2, 3).expect("date"),
                NaiveDate::from_ymd_opt(2030, 2, 10).expect("date"),
            )
            .expect("consumed range"),
        ];
        let frozen_at = Utc
            .with_ymd_and_hms(2030, 2, 2, 12, 0, 0)
            .single()
            .expect("timestamp");
        let error = prospective_window_for_freeze(&consumed_policy, frozen_at, [].into_iter())
            .expect_err("window overlapping consumed range");
        assert!(error.to_string().contains("excluded validation"));
    }

    #[test]
    fn prospective_window_requires_the_first_utc_day_after_end() {
        let window = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 2, 3).expect("date"),
            NaiveDate::from_ymd_opt(2030, 3, 4).expect("date"),
        )
        .expect("window");
        for hour in [0, 12] {
            let consumed_at = Utc
                .with_ymd_and_hms(2030, 3, 4, hour, 0, 0)
                .single()
                .expect("timestamp");
            let error = ensure_prospective_window_elapsed(window, consumed_at)
                .expect_err("final UTC day must still be protected");
            assert!(error.to_string().contains("not fully elapsed"));
        }
        ensure_prospective_window_elapsed(
            window,
            Utc.with_ymd_and_hms(2030, 3, 5, 0, 0, 0)
                .single()
                .expect("timestamp"),
        )
        .expect("next UTC day is eligible");
    }

    #[test]
    fn prospective_marker_is_exclusive_and_scoped_to_window() {
        let root = std::env::current_dir()
            .expect("test working directory")
            .join("target")
            .join("test-scratch")
            .join(format!(
                "prospective-marker-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
        fs::create_dir_all(&root).expect("marker directory");
        let first_window = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 2, 3).expect("date"),
            NaiveDate::from_ymd_opt(2030, 3, 4).expect("date"),
        )
        .expect("window");
        let second_window = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 3, 5).expect("date"),
            NaiveDate::from_ymd_opt(2030, 4, 3).expect("date"),
        )
        .expect("window");
        let first_path = prospective_window_marker_path(&root, first_window);
        let second_path = prospective_window_marker_path(&root, second_window);
        assert_ne!(first_path, second_path);
        fs::create_dir_all(first_path.parent().expect("marker parent")).expect("marker parent");

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let contenders = [b"first".to_vec(), b"second".to_vec()]
            .into_iter()
            .map(|contents| {
                let marker_path = first_path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    create_prospective_window_marker(&marker_path, &contents)
                })
            })
            .collect::<Vec<_>>();
        let results = contenders
            .into_iter()
            .map(|contender| contender.join().expect("marker contender"))
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(results.iter().any(|result| {
            result.as_ref().err().is_some_and(|error| {
                error.chain().any(|cause| {
                    cause
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|error| error.kind() == std::io::ErrorKind::AlreadyExists)
                })
            })
        }));
        fs::remove_file(first_path).expect("remove marker");
    }

    #[test]
    fn prospective_global_registry_reserves_only_one_window() {
        let root = std::env::current_dir()
            .expect("test working directory")
            .join("target")
            .join("test-scratch")
            .join(format!(
                "prospective-registry-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
        let registry_path = prospective_registry_marker_path(&root);
        fs::create_dir_all(registry_path.parent().expect("registry parent"))
            .expect("registry parent");
        let first_window = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 2, 3).expect("date"),
            NaiveDate::from_ymd_opt(2030, 3, 4).expect("date"),
        )
        .expect("window");
        let second_window = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 4, 5).expect("date"),
            NaiveDate::from_ymd_opt(2030, 5, 4).expect("date"),
        )
        .expect("window");
        let marker = |window: DateRange| ProspectiveRegistryMarker {
            schema_version: PROSPECTIVE_REGISTRY_SCHEMA_VERSION,
            freeze_fingerprint: "sha256-v1:freeze".to_string(),
            window_fingerprint: format!("sha256-v1:{}", window.start),
            pre_window_history_fingerprint: "sha256-v1:pre-window".to_string(),
            window,
            reserved_at_utc: Utc
                .with_ymd_and_hms(2030, 5, 5, 0, 0, 0)
                .single()
                .expect("timestamp"),
        };
        let first = serde_json::to_vec(&marker(first_window)).expect("first marker");
        let second = serde_json::to_vec(&marker(second_window)).expect("second marker");
        create_once_marker(&registry_path, &first, "prospective registry")
            .expect("first window reserves registry");
        let error = create_once_marker(&registry_path, &second, "prospective registry")
            .expect_err("second window must not reserve a second prospective run");
        assert!(error.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::AlreadyExists)
        }));
        fs::remove_file(registry_path).expect("remove registry marker");
    }

    #[test]
    fn prospective_freeze_output_is_immutable_once_file() {
        let root = std::env::current_dir()
            .expect("test working directory")
            .join("target")
            .join("test-scratch")
            .join(format!(
                "prospective-freeze-output-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
        let output_path = root.join("prospective-frozen-v1.json");
        fs::create_dir_all(&root).expect("output parent");
        create_once_marker(
            &output_path,
            br#"{"schema_version":1,"identity_format":"test"}"#,
            "prospective frozen candidate",
        )
        .expect("first freeze output");
        let error = create_once_marker(
            &output_path,
            br#"{"schema_version":1,"identity_format":"replacement"}"#,
            "prospective frozen candidate",
        )
        .expect_err("a freeze output must never be replaced");
        assert!(error.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::AlreadyExists)
        }));
        assert_eq!(
            fs::read_to_string(&output_path).expect("read freeze output"),
            r#"{"schema_version":1,"identity_format":"test"}"#
        );
        fs::remove_file(output_path).expect("remove freeze output");
    }

    #[test]
    fn prospective_output_aliases_reject_before_marker_acquisition() {
        let root = std::env::current_dir()
            .expect("test working directory")
            .join("target")
            .join("test-scratch")
            .join(format!(
                "prospective-output-alias-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
        let window = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 2, 3).expect("date"),
            NaiveDate::from_ymd_opt(2030, 3, 4).expect("date"),
        )
        .expect("window");
        let marker_path = prospective_window_marker_path(&root, window);
        let registry_path = prospective_registry_marker_path(&root);
        let mut aliases = vec![
            root.join("benchmarks/predictive/../predictive/prospective-window-once.json"),
            root.join("benchmarks/predictive/./prospective-2030-02-03-2030-03-04-once.json"),
            root.join("benchmarks/predictive/prospective-report.json"),
            root.join("benchmarks/predictive/prospective-window-once."),
        ];
        #[cfg(windows)]
        aliases.push(root.join("BENCHMARKS/PREDICTIVE/PROSPECTIVE-WINDOW-ONCE.JSON"));

        for output_path in aliases {
            let error =
                preflight_prospective_output_path(&output_path, &marker_path, &registry_path)
                    .expect_err("output aliases must be rejected before acquisition");
            assert!(error.to_string().contains("aliases"));
            assert!(!marker_path.exists());
            assert!(!registry_path.exists());
        }
    }

    #[test]
    fn prospective_identity_binds_all_freeze_inputs() {
        let frozen_at = Utc
            .with_ymd_and_hms(2030, 2, 2, 12, 0, 0)
            .single()
            .expect("timestamp");
        let window = DateRange::new(
            NaiveDate::from_ymd_opt(2030, 2, 3).expect("date"),
            NaiveDate::from_ymd_opt(2030, 3, 4).expect("date"),
        )
        .expect("window");
        let base = prospective_freeze_fingerprint(
            "freeze-a",
            "input-a",
            "config-a",
            "comparison-a",
            "history-a",
            frozen_at,
            window,
        );
        assert_ne!(
            base,
            prospective_freeze_fingerprint(
                "freeze-b",
                "input-a",
                "config-a",
                "comparison-a",
                "history-a",
                frozen_at,
                window
            )
        );
        assert_ne!(
            base,
            prospective_freeze_fingerprint(
                "freeze-a",
                "input-b",
                "config-a",
                "comparison-a",
                "history-a",
                frozen_at,
                window
            )
        );
        assert_ne!(
            base,
            prospective_freeze_fingerprint(
                "freeze-a",
                "input-a",
                "config-b",
                "comparison-a",
                "history-a",
                frozen_at,
                window
            )
        );
        assert_ne!(
            base,
            prospective_freeze_fingerprint(
                "freeze-a",
                "input-a",
                "config-a",
                "comparison-b",
                "history-a",
                frozen_at,
                window
            )
        );
        assert_ne!(
            base,
            prospective_freeze_fingerprint(
                "freeze-a",
                "input-a",
                "config-a",
                "comparison-a",
                "history-b",
                frozen_at,
                window
            )
        );
        assert_ne!(
            base,
            prospective_freeze_fingerprint(
                "freeze-a",
                "input-a",
                "config-a",
                "comparison-a",
                "history-a",
                frozen_at,
                DateRange::new(
                    NaiveDate::from_ymd_opt(2030, 2, 4).expect("date"),
                    NaiveDate::from_ymd_opt(2030, 3, 5).expect("date"),
                )
                .expect("window")
            )
        );
    }

    #[test]
    fn prospective_pre_window_history_digest_requires_complete_coverage() {
        let development_cutoff = NaiveDate::from_ymd_opt(2029, 12, 31).expect("date");
        let start = prospective_pre_window_history_start(development_cutoff)
            .expect("history starts after development cutoff");
        assert_eq!(start, NaiveDate::from_ymd_opt(2030, 1, 1).expect("date"));
        let freeze_date = NaiveDate::from_ymd_opt(2030, 1, 3).expect("date");
        let entries = vec![
            NytDailyEntry {
                id: Some(1),
                solution: "aaaaa".to_string(),
                print_date: start,
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(2),
                solution: "bbbbb".to_string(),
                print_date: start + Days::new(1),
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(3),
                solution: "ccccc".to_string(),
                print_date: freeze_date,
                days_since_launch: None,
                editor: None,
            },
        ];
        let digest = prospective_pre_window_history_fingerprint(&entries, start, freeze_date)
            .expect("complete pre-window history");
        let mut changed = entries.clone();
        changed[2].solution = "ddddd".to_string();
        assert_ne!(
            digest,
            prospective_pre_window_history_fingerprint(&changed, start, freeze_date)
                .expect("changed history digest")
        );
        let missing = entries[..2].to_vec();
        let error = prospective_pre_window_history_fingerprint(&missing, start, freeze_date)
            .expect_err("missing freeze-date row");
        assert!(error.to_string().contains("exactly cover"));
        let mut same_date_append = entries.clone();
        same_date_append.push(NytDailyEntry {
            id: Some(4),
            solution: "eeeee".to_string(),
            print_date: freeze_date,
            days_since_launch: None,
            editor: None,
        });
        let error =
            prospective_pre_window_history_fingerprint(&same_date_append, start, freeze_date)
                .expect_err("same-date append must not replay a frozen history digest");
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn prospective_history_digest_binds_window_rows_without_serializing_marker_words() {
        let first_date = NaiveDate::from_ymd_opt(2030, 2, 3).expect("date");
        let entries = vec![
            NytDailyEntry {
                id: Some(1),
                solution: "aaaaa".to_string(),
                print_date: first_date,
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(2),
                solution: "bbbbb".to_string(),
                print_date: first_date + Days::new(1),
                days_since_launch: None,
                editor: None,
            },
        ];
        let window = DateRange::new(first_date, first_date + Days::new(1)).expect("window");
        let digest = prospective_history_fingerprint(&entries, window).expect("digest");
        let mut changed = entries.clone();
        changed[1].solution = "ccccc".to_string();
        assert_ne!(
            digest,
            prospective_history_fingerprint(&changed, window).expect("digest")
        );
        let marker = ProspectiveWindowMarker {
            schema_version: PROSPECTIVE_MARKER_SCHEMA_VERSION,
            freeze_fingerprint: "sha256-v1:freeze".to_string(),
            window_fingerprint: "sha256-v1:window".to_string(),
            input_fingerprint: "sha256-v1:input".to_string(),
            config_fingerprint: "sha256-v1:config".to_string(),
            pre_window_history_fingerprint: "sha256-v1:pre-window".to_string(),
            window,
            window_data_fingerprint: digest,
            output_path: "report.json".to_string(),
            consumed_at_utc: Utc
                .with_ymd_and_hms(2030, 2, 4, 0, 0, 0)
                .single()
                .expect("timestamp"),
            status: "started_irreversible".to_string(),
        };
        let serialized = serde_json::to_string(&marker).expect("marker");
        assert!(!serialized.contains("aaaaa"));
        assert!(!serialized.contains("bbbbb"));
    }

    #[test]
    fn finite_step_evidence_keeps_deadline_and_does_not_serialize_heuristic_values() {
        let guesses = vec!["cigar".to_string(), "rebut".to_string()];
        let search = FiniteSearchResult {
            root_candidates_considered: 2,
            all_legal_roots_evaluated: false,
            candidates: vec![
                FiniteSearchCandidate {
                    guess_index: 0,
                    failure_probability: 1.0,
                    expected_attempts: f64::INFINITY,
                    quality: FiniteSearchQuality::Heuristic,
                },
                FiniteSearchCandidate {
                    guess_index: 1,
                    failure_probability: 0.25,
                    expected_attempts: 1.75,
                    quality: FiniteSearchQuality::UpperBound,
                },
            ],
            reason: FiniteSearchReason::Deadline,
            nodes_visited: 17,
            work_units: 42,
            cache_hits: 0,
            proposal_sampled: true,
        };
        let trace = finite_step_evidence(&guesses, 2, &search).expect("trace");
        assert_eq!(trace.reason, "deadline");
        assert_eq!(trace.work_units, 42);
        assert_eq!(trace.candidate_count, 2);
        assert_eq!(trace.top_candidates[0].quality, "heuristic");
        assert_eq!(trace.top_candidates[0].modeled_failure_probability, None);
        assert_eq!(trace.top_candidates[0].expected_attempts_remaining, None);
        assert_eq!(trace.top_candidates[1].word, "rebut");
        assert_eq!(
            trace.top_candidates[1].modeled_failure_probability,
            Some(0.25)
        );
        serde_json::to_string(&trace).expect("finite JSON");

        let mut invalid = search;
        invalid.candidates[0].quality = FiniteSearchQuality::UpperBound;
        assert!(finite_step_evidence(&guesses, 2, &invalid).is_err());

        invalid.candidates = vec![
            FiniteSearchCandidate {
                guess_index: 1,
                failure_probability: 0.25,
                expected_attempts: 1.75,
                quality: FiniteSearchQuality::UpperBound,
            };
            9
        ];
        invalid.candidates[8].expected_attempts = f64::INFINITY;
        assert!(finite_step_evidence(&guesses, 2, &invalid).is_err());
    }

    #[test]
    fn finite_backtest_trace_rejects_missing_or_misaligned_steps() {
        let mut solver = finite_audit_test_solver(&["cigar", "rebut", "sissy"]);
        solver.config.search_policy_mode = crate::config::SearchPolicyMode::FiniteFast;
        let date = NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let run = solver
            .solve_target_detailed("cigar", date, 1)
            .expect("finite run");
        validate_finite_run_trace(&run, run.steps.len()).expect("aligned trace");

        let mut missing = run.clone();
        missing.steps[0].finite_search = None;
        assert!(validate_finite_run_trace(&missing, missing.steps.len()).is_err());

        let mut misaligned = run;
        misaligned.steps[0]
            .finite_search
            .as_mut()
            .expect("trace")
            .top_candidates[0]
            .word = "rebut".to_string();
        assert!(validate_finite_run_trace(&misaligned, misaligned.steps.len()).is_err());
    }

    fn finite_audit_test_solver(words: &[&str]) -> Solver {
        finite_audit_test_solver_with_guesses_and_answers(words, words)
    }

    fn finite_audit_test_solver_with_guesses_and_answers(
        guess_words: &[&str],
        answer_words: &[&str],
    ) -> Solver {
        let guesses = guess_words
            .iter()
            .map(|word| (*word).to_string())
            .collect::<Vec<_>>();
        let answers = answer_words
            .iter()
            .map(|word| AnswerRecord {
                word: (*word).to_string(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect::<Vec<_>>();
        let fixture = crate::test_support::TestDirectory::new("solver-fixture");
        let root = fixture.path().to_path_buf();
        fs::create_dir_all(&root).expect("test pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        fs::remove_file(root.join("pattern.bin")).expect("remove in-memory fixture's backing file");
        fs::remove_dir(&root).expect("remove empty fixture directory");
        Solver {
            config: PriorConfig::default(),
            mode: WeightMode::Uniform,
            variant: ModelVariant::SeedPlusHistory,
            data: std::sync::Arc::new(crate::solver::SolverData {
                guesses: guesses.clone(),
                answers,
                primary_answer_count: answer_words.len(),
                history_dates: Vec::new(),
                pattern_table,
                guess_index: guesses
                    .iter()
                    .enumerate()
                    .map(|(index, word)| (word.clone(), index))
                    .collect::<HashMap<_, _>>(),
            }),
            artifact_dir: root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            identity_cache: Default::default(),
            test_fixture: Some(std::sync::Arc::new(fixture)),
        }
    }

    #[test]
    fn learned_proxy_collection_skips_consumed_targets_but_preserves_history_features() {
        let mut solver = finite_audit_test_solver(&["aaaaa", "bbbbb", "ccccc"]);
        let date = |day| NaiveDate::from_ymd_opt(2024, 1, day).unwrap();
        solver.data_mut().history_dates = (1..=4)
            .map(|day| NytDailyEntry {
                id: None,
                solution: "aaaaa".into(),
                print_date: date(day),
                days_since_launch: None,
                editor: None,
            })
            .collect();
        solver.data_mut().answers[0].history_dates = vec![date(2)];
        let range = DateRange::new(date(1), date(4)).unwrap();
        let plan = EvaluationPlan {
            history: DateRange::new(date(1), date(5)).unwrap(),
            development: range,
            sealed_test: DateRange::new(date(5), date(5)).unwrap(),
            folds: Vec::new(),
            config: crate::experiments::RollingOriginConfig::default(),
            excluded_target_ranges: vec![DateRange::new(date(2), date(2)).unwrap()],
        };
        let (states, scanned) = solver
            .collect_learned_proxy_states(
                &plan,
                range,
                3,
                3,
                10,
                Instant::now(),
                Duration::from_secs(5),
                false,
                false,
            )
            .unwrap();
        assert_eq!(scanned, 3);
        assert_eq!(
            states.iter().map(|state| state.date).collect::<Vec<_>>(),
            vec![date(1), date(3), date(4)]
        );
        assert_eq!(solver.history_dates.len(), 4);
        let snapshot = crate::model::weight_snapshot_for_mode(
            &solver.answers[0],
            &solver.config,
            date(3),
            WeightMode::Uniform,
        );
        assert_eq!(snapshot.seen_count, 1);
        assert_eq!(snapshot.last_seen, Some(date(2)));
    }

    #[test]
    fn learned_proxy_teacher_is_unchanged_by_production_exact_thresholds_and_pools() {
        let words = [
            "tower", "power", "bower", "rower", "sware", "crare", "beare", "urare", "blare",
            "mesne",
        ];
        let mut solver = finite_audit_test_solver_with_guesses_and_answers(&words, &words[..9]);
        let weights = [43.0, 43.0, 19.0, 4.0, 4.0, 5.0, 6.0, 6.0, 2.0];
        let survivors = (0..9).collect::<Vec<_>>();
        let requested = (0..words.len()).collect::<Vec<_>>();
        let mut outputs = Vec::new();
        for (threshold, pool) in [(1, 1), (50, words.len())] {
            solver.config.exact_exhaustive_threshold = threshold;
            solver.config.exact_candidate_pool = pool;
            let mut budget = super::super::exhaustive_teacher::WorkBudget::new(
                Instant::now(),
                0,
                Duration::from_secs(5),
                None,
            );
            outputs.push(
                super::super::exhaustive_teacher::label_root_actions(
                    &survivors,
                    &weights,
                    solver.guesses.len(),
                    &requested,
                    requested.len(),
                    |guess, answer| solver.pattern_table.get(guess, answer),
                    &mut budget,
                )
                .unwrap(),
            );
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[0].len(), requested.len());
    }

    #[test]
    fn learned_proxy_dataset_resume_revalidates_completed_rows_and_elapsed_budget() {
        let root = crate::test_support::TestDirectory::new("teacher-resume");
        let paths = ProjectPaths::new(root.path());
        fs::create_dir_all(root.path().join("config")).unwrap();
        fs::create_dir_all(root.path().join("data/raw")).unwrap();
        let mut solver = finite_audit_test_solver(&["aaaaa", "bbbbb"]);
        let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let cutoff = start.checked_add_days(Days::new(424)).unwrap();
        let policy = EvaluationPolicy {
            format_version: 1,
            development_cutoff: cutoff,
            sealed_test: DateRange::new(
                cutoff.checked_add_days(Days::new(1)).unwrap(),
                cutoff.checked_add_days(Days::new(30)).unwrap(),
            )
            .unwrap(),
            excluded_validation: Vec::new(),
        };
        fs::write(
            root.path().join("config/evaluation.toml"),
            toml::to_string(&policy).unwrap(),
        )
        .unwrap();
        let history = (0..425)
            .map(|day| NytDailyEntry {
                id: None,
                solution: "aaaaa".into(),
                print_date: start.checked_add_days(Days::new(day)).unwrap(),
                days_since_launch: None,
                editor: None,
            })
            .collect::<Vec<_>>();
        let source = history
            .iter()
            .map(|entry| serde_json::to_string(entry).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&paths.raw_history, source).unwrap();
        solver.data_mut().history_dates = history;
        let checkpoint_path = root.path().join("checkpoint.json");
        let request = LearnedProxyDatasetRequest {
            minimum_survivors: 2,
            maximum_survivors: 2,
            maximum_states_per_split: 1,
            guesses_per_state: 2,
            maximum_seconds: 60,
            maximum_memory_mb: 8_192,
            checkpoint_path: Some(checkpoint_path.clone()),
        };
        let artifact = solver
            .learned_proxy_dataset(&paths, request.clone())
            .unwrap();
        assert_eq!(artifact.rows.len(), 6);
        assert_eq!(artifact.completed_state_row_counts.len(), 3);
        assert!(
            artifact
                .rows
                .iter()
                .all(|row| row.exact_continuation_cost == 1.5)
        );
        assert_eq!(
            artifact.provenance.replay_identity.algorithm_version,
            super::super::exhaustive_teacher::CONTRACT
        );
        let mut changed_sampling = request.clone();
        changed_sampling.maximum_survivors = 3;
        let original_checkpoint = fs::read(&checkpoint_path).unwrap();
        assert!(
            solver
                .learned_proxy_dataset(&paths, changed_sampling)
                .unwrap_err()
                .to_string()
                .contains("different source/config/code inputs")
        );
        assert_eq!(fs::read(&checkpoint_path).unwrap(), original_checkpoint);
        let resumed = solver
            .learned_proxy_dataset(&paths, request.clone())
            .unwrap();
        assert_eq!(resumed.rows, artifact.rows);
        let mut interrupted_request = request.clone();
        let interrupted_path = root.path().join("interrupted.json");
        interrupted_request.checkpoint_path = Some(interrupted_path.clone());
        super::super::exhaustive_teacher::EXHAUST_AFTER_STATES
            .with(|remaining| remaining.set(Some(1)));
        let interrupted = solver.learned_proxy_dataset(&paths, interrupted_request.clone());
        super::super::exhaustive_teacher::EXHAUST_AFTER_STATES
            .with(|remaining| remaining.set(None));
        assert!(
            interrupted
                .unwrap_err()
                .to_string()
                .contains("wall-clock budget")
        );
        let interrupted_bytes = fs::read(&interrupted_path).unwrap();
        let stopped: ExhaustiveCostCheckpoint = serde_json::from_slice(&interrupted_bytes).unwrap();
        stopped.validate(&artifact.split).unwrap();
        assert_eq!(stopped.rows.len(), 2);
        assert_eq!(stopped.completed_state_ids.len(), 1);
        assert!(!stopped.progress.complete);
        assert!(
            stopped
                .progress
                .stop_reason
                .as_ref()
                .unwrap()
                .contains("wall-clock budget")
        );
        assert!(stopped.progress.elapsed_ms >= interrupted_request.maximum_seconds * 1_000);
        assert!(
            solver
                .learned_proxy_dataset(&paths, interrupted_request)
                .unwrap_err()
                .to_string()
                .contains("wall-clock budget")
        );
        assert_eq!(fs::read(interrupted_path).unwrap(), interrupted_bytes);
        let checkpoint: ExhaustiveCostCheckpoint =
            serde_json::from_slice(&fs::read(&checkpoint_path).unwrap()).unwrap();

        // Cross-field-consistent tampering still has to match the reconstructed state/actions.
        let mut changed = checkpoint.clone();
        let state_id = changed.rows[0].state.state_id.clone();
        for row in changed
            .rows
            .iter_mut()
            .filter(|row| row.state.state_id == state_id)
        {
            row.state.survivor_ids = vec![0, 2];
        }
        changed.validate(&artifact.split).unwrap();
        fs::write(&checkpoint_path, serde_json::to_vec(&changed).unwrap()).unwrap();
        let changed_bytes = fs::read(&checkpoint_path).unwrap();
        let error = solver
            .learned_proxy_dataset(&paths, request.clone())
            .unwrap_err();
        assert!(
            error.to_string().contains("reconstructed state"),
            "{error:#}"
        );
        assert_eq!(fs::read(&checkpoint_path).unwrap(), changed_bytes);

        let mut missing = checkpoint.clone();
        missing.rows.remove(0);
        *missing
            .completed_state_row_counts
            .get_mut(&state_id)
            .unwrap() -= 1;
        missing.progress.rows_emitted -= 1;
        missing.validate(&artifact.split).unwrap();
        fs::write(&checkpoint_path, serde_json::to_vec(&missing).unwrap()).unwrap();
        let missing_bytes = fs::read(&checkpoint_path).unwrap();
        let error = solver
            .learned_proxy_dataset(&paths, request.clone())
            .unwrap_err();
        assert!(error.to_string().contains("action identities"), "{error:#}");
        assert_eq!(fs::read(&checkpoint_path).unwrap(), missing_bytes);

        let mut expired = checkpoint;
        expired.progress.elapsed_ms = request.maximum_seconds * 1_000;
        expired.validate(&artifact.split).unwrap();
        let bytes = serde_json::to_vec(&expired).unwrap();
        fs::write(&checkpoint_path, &bytes).unwrap();
        let error = solver.learned_proxy_dataset(&paths, request).unwrap_err();
        assert!(
            error.to_string().contains("cumulative wall-clock budget"),
            "{error:#}"
        );
        assert_eq!(fs::read(checkpoint_path).unwrap(), bytes);
    }

    fn finite_audit_test_state(answer_count: usize) -> SolveState {
        let weights = vec![1.0; answer_count];
        SolveState {
            condition_only: true,
            surviving: (0..answer_count).collect(),
            fallback_surviving: Vec::new(),
            fallback_active: false,
            modeled_weights: weights.clone(),
            recovery_weights: weights.clone(),
            weights,
            modeled_total_weight: answer_count as f64,
            total_weight: answer_count as f64,
            recovery_mode_used: None,
        }
    }

    fn staged_certificate_test_paths() -> (crate::test_support::TestDirectory, PathBuf, ProjectPaths)
    {
        let fixture = crate::test_support::TestDirectory::new("staged-certificate");
        let root = fixture.path().to_path_buf();
        fs::create_dir_all(root.join("config")).expect("config directory");
        fs::create_dir_all(root.join("data/raw")).expect("raw data directory");
        fs::write(
            root.join("config/evaluation.toml"),
            include_str!("../../config/evaluation.toml"),
        )
        .expect("evaluation policy");
        let end = NaiveDate::from_ymd_opt(2026, 8, 26).expect("date");
        let start = end.checked_sub_days(Days::new(424)).expect("history start");
        let mut history = String::new();
        for offset in 0..425u64 {
            let entry = NytDailyEntry {
                id: None,
                solution: "aaaaa".to_string(),
                print_date: start
                    .checked_add_days(Days::new(offset))
                    .expect("history date"),
                days_since_launch: None,
                editor: None,
            };
            history.push_str(&serde_json::to_string(&entry).expect("history entry"));
            history.push('\n');
        }
        fs::write(root.join("data/raw/nyt_daily_answers.jsonl"), history).expect("history");
        (fixture, root.clone(), ProjectPaths::new(&root))
    }

    fn exact_finite_reference(
        word: &str,
        failure_probability: f64,
        expected_attempts: f64,
    ) -> FiniteSearchRegretReference {
        FiniteSearchRegretReference {
            status: "exact".to_string(),
            word: Some(word.to_string()),
            value: Some(FiniteSearchRegretValue {
                failure_probability,
                expected_attempts,
            }),
        }
    }

    #[test]
    fn finite_search_regret_rejects_fixed_root_better_than_global_reference() {
        let fixed = exact_finite_reference("fixed", 0.1, 1.0);
        let global = exact_finite_reference("global", 0.2, 1.5);
        let error = finite_regrets(&fixed, &global).expect_err("contradictory references");
        assert!(error.to_string().contains("reference contradiction"));
    }

    #[test]
    fn finite_search_regret_finds_known_two_turn_optimum() {
        let solver = finite_audit_test_solver(&["aaaaa", "bbbbb", "ccccc", "abcde"]);
        let state = finite_audit_test_state(solver.answers.len());
        let reference = solver
            .finite_global_reference(
                &state,
                &[],
                2,
                false,
                Instant::now(),
                Duration::from_secs(10),
            )
            .expect("finite global reference");
        assert_eq!(reference.status, "exact");
        assert_eq!(reference.word.as_deref(), Some("abcde"));
        assert_eq!(
            reference
                .value
                .expect("reference value")
                .failure_probability,
            0.0
        );
    }

    #[test]
    fn staged_zero_failure_rejects_a_non_green_bucket_above_remaining_turns() {
        let solver = finite_audit_test_solver(&["aaaaa", "aaaab", "aaabb", "zzzzz"]);
        let state = finite_audit_test_state(solver.answers.len());
        let result = solver
            .staged_zero_failure_root_check(&state, &[], solver.guess_index["zzzzz"], 3, false)
            .expect("root check");
        assert_eq!(result.reason, "bucket_cardinality");
        assert!(!result.certified);
    }

    #[test]
    fn staged_zero_failure_rejects_dormant_support_that_can_activate_later() {
        let mut solver = finite_audit_test_solver(&["aaaaa", "abbbb", "acccc", "zzzzz"]);
        solver.data_mut().primary_answer_count = 1;
        solver.config.fallback_activation_threshold = 0;
        let mut state = finite_audit_test_state(solver.answers.len());
        state.condition_only = false;
        state.surviving = vec![0];
        state.fallback_surviving = vec![1, 2];
        let result = solver
            .staged_zero_failure_root_check(&state, &[], solver.guess_index["zzzzz"], 3, false)
            .expect("root check");
        assert_eq!(result.reason, "dormant_support");
        assert!(!result.certified);
    }

    #[test]
    fn staged_zero_failure_rejects_a_zero_modeled_child_answer() {
        let solver = finite_audit_test_solver(&["aaaaa", "abbbb", "zzzzz"]);
        let mut state = finite_audit_test_state(solver.answers.len());
        state.condition_only = false;
        state.surviving = vec![0, 1];
        state.fallback_surviving.clear();
        state.modeled_weights[1] = 0.0;
        let result = solver
            .staged_zero_failure_root_check(&state, &[], solver.guess_index["zzzzz"], 3, false)
            .expect("root check");
        assert_eq!(result.reason, "unstable_modeled_support");
        assert!(!result.certified);
    }

    #[test]
    fn staged_zero_failure_rejects_dormant_root_support_when_active_root_is_green() {
        let solver = finite_audit_test_solver(&["aaaaa", "zzzzz"]);
        let mut state = finite_audit_test_state(solver.answers.len());
        state.condition_only = false;
        state.surviving = vec![0];
        state.fallback_surviving = vec![1];
        let result = solver
            .staged_zero_failure_root_check(&state, &[], solver.guess_index["aaaaa"], 3, false)
            .expect("root check");
        assert_eq!(result.reason, "dormant_support");
        assert!(!result.certified);
    }

    #[test]
    fn staged_zero_failure_target_support_rejects_absent_and_zero_modeled_targets() {
        let solver = finite_audit_test_solver(&["aaaaa", "aaaab"]);
        let state = finite_audit_test_state(solver.answers.len());
        assert!(
            solver
                .staged_target_index_if_supported(&state, "bbbbb")
                .is_none()
        );

        let mut zero_modeled = state.clone();
        zero_modeled.modeled_weights[0] = 0.0;
        assert!(
            solver
                .staged_target_index_if_supported(&zero_modeled, "aaaaa")
                .is_none()
        );
    }

    #[test]
    fn staged_zero_failure_target_support_keeps_colliding_feedback_target() {
        let solver = finite_audit_test_solver(&["aaaaa", "aaaab", "ccccc"]);
        let state = finite_audit_test_state(solver.answers.len());
        let target_index = solver
            .staged_target_index_if_supported(&state, "aaaab")
            .expect("colliding target remains supported");
        let root_index = solver.guess_index["ccccc"];
        assert_eq!(
            solver.answer_pattern(root_index, target_index),
            solver.answer_pattern(root_index, solver.guess_index["aaaaa"])
        );
    }

    #[test]
    fn staged_zero_failure_stops_a_game_after_a_green_root() {
        let (_fixture, root, paths) = staged_certificate_test_paths();
        let date = NaiveDate::from_ymd_opt(2026, 8, 26).expect("date");
        let mut solver = finite_audit_test_solver(&["aaaaa"]);
        solver.data_mut().history_dates = vec![NytDailyEntry {
            id: None,
            solution: "aaaaa".to_string(),
            print_date: date,
            days_since_launch: None,
            editor: None,
        }];
        let report = solver
            .staged_zero_failure_certificate_report(
                &paths,
                StagedZeroFailureCertificateRequest {
                    from: date,
                    to: date,
                    maximum_states: 2,
                    maximum_seconds: 10,
                    hard_mode: false,
                },
            )
            .expect("certificate report");
        assert_eq!(report.selected_roots, 1);
        assert_eq!(report.evaluated_roots, 1);
        assert_eq!(report.certified_roots, 1);
        assert_eq!(report.unsupported_target_games, 0);
        assert_eq!(report.replayed_games, 1);
        assert_eq!(report.path_replay_failures, 0);
        assert!(report.complete);
        assert!(!report.state_cap_reached);
        assert!(!report.deadline_reached);
        assert_eq!(report.scheduled_days, 1);
        assert_eq!(report.coverage_gaps, 0);
        assert_eq!(report.duplicate_history_dates, 0);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn staged_zero_failure_report_skips_an_unsupported_historical_target() {
        let (_fixture, root, paths) = staged_certificate_test_paths();
        let date = NaiveDate::from_ymd_opt(2026, 8, 26).expect("date");
        let mut solver = finite_audit_test_solver(&["aaaaa"]);
        solver.data_mut().history_dates = vec![NytDailyEntry {
            id: None,
            solution: "bbbbb".to_string(),
            print_date: date,
            days_since_launch: None,
            editor: None,
        }];
        let report = solver
            .staged_zero_failure_certificate_report(
                &paths,
                StagedZeroFailureCertificateRequest {
                    from: date,
                    to: date,
                    maximum_states: 1,
                    maximum_seconds: 10,
                    hard_mode: false,
                },
            )
            .expect("certificate report");
        assert_eq!(report.unsupported_target_games, 1);
        assert_eq!(report.selected_roots, 0);
        assert_eq!(report.evaluated_roots, 0);
        assert_eq!(report.replayed_games, 0);
        assert_eq!(report.path_replay_failures, 0);
        assert!(!report.complete);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn staged_zero_failure_counts_a_six_turn_unsolved_replay() {
        let guesses = [
            "aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff", "zzzzz",
        ];
        let target = "zzzzz";
        let answers = ["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff", target];
        let solver = finite_audit_test_solver_with_guesses_and_answers(&guesses, &answers);
        let date = NaiveDate::from_ymd_opt(2026, 8, 26).expect("date");
        let (_fixture, root, paths) = staged_certificate_test_paths();
        let mut solver = solver;
        solver.data_mut().history_dates = vec![NytDailyEntry {
            id: None,
            solution: target.to_string(),
            print_date: date,
            days_since_launch: None,
            editor: None,
        }];
        let report = solver
            .staged_zero_failure_certificate_report(
                &paths,
                StagedZeroFailureCertificateRequest {
                    from: date,
                    to: date,
                    maximum_states: 6,
                    maximum_seconds: 10,
                    hard_mode: false,
                },
            )
            .expect("certificate report");
        assert_eq!(report.selected_roots, 6);
        assert_eq!(report.evaluated_roots, 6);
        assert_eq!(report.unsupported_target_games, 0);
        assert_eq!(report.replayed_games, 1);
        assert_eq!(report.path_replay_failures, 0);
        assert!(report.certified_roots < report.evaluated_roots);
        assert!(report.complete);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn staged_zero_failure_rejects_an_answer_missing_from_the_guess_dictionary() {
        let mut solver = finite_audit_test_solver(&["aaaaa", "bbbbb", "ccccc"]);
        solver.data_mut().guess_index.remove("bbbbb");
        let state = finite_audit_test_state(solver.answers.len());
        let result = solver
            .staged_zero_failure_root_check(&state, &[], solver.guess_index["aaaaa"], 3, false)
            .expect("root check");
        assert_eq!(result.reason, "missing_dictionary_answer");
        assert!(!result.certified);
    }

    #[test]
    fn staged_zero_failure_uses_complete_duplicate_letter_hard_history() {
        let solver = finite_audit_test_solver(&["allee", "llama", "cigar", "caper"]);
        let observations = vec![
            ("allee".to_string(), score_guess("allee", "llama")),
            ("cigar".to_string(), score_guess("cigar", "llama")),
        ];
        let mut state = finite_audit_test_state(solver.answers.len());
        state.surviving = vec![3];
        let result = solver
            .staged_zero_failure_root_check(
                &state,
                &observations,
                solver.guess_index["llama"],
                3,
                true,
            )
            .expect("root check");
        assert_eq!(result.reason, "hard_mode");
        assert!(!result.certified);
    }

    #[test]
    fn same_state_dynamic_regret_uses_the_selected_dynamic_root_row() {
        let mut solver = finite_audit_test_solver(&["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee"]);
        solver.data_mut().primary_answer_count = 3;
        solver.config.fallback_activation_threshold = 2;
        solver.config.search_policy_mode = crate::config::SearchPolicyMode::FiniteFastDynamic;
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).expect("date");
        let state = solver.initial_state(date);
        let started = Instant::now();
        let budget = Duration::from_secs(10);
        let selected_root = solver.guess_index["ddddd"];

        let (global, selected) = solver
            .same_state_dynamic_root_references(&state, &[], 3, &[selected_root], started, budget)
            .expect("dynamic same-state references");
        let direct = solver
            .finite_horizon_search_dynamic(
                &state,
                &[],
                3,
                false,
                solver.finite_exact_options(state.surviving.len(), budget),
                &|| false,
            )
            .expect("direct dynamic reference");
        let frozen = solver
            .finite_horizon_search(
                &state.surviving,
                &state.weights,
                &[],
                3,
                false,
                solver.finite_exact_options(state.surviving.len(), budget),
                &|| false,
            )
            .expect("frozen-basis comparison");
        let expected = direct
            .candidates
            .iter()
            .find(|candidate| candidate.guess_index == selected_root)
            .copied()
            .expect("selected root row");
        let frozen_selected = frozen
            .candidates
            .iter()
            .find(|candidate| candidate.guess_index == selected_root)
            .copied()
            .expect("frozen selected root row");
        assert_eq!(global.status, "exact");
        assert_eq!(selected[0].status, "exact");
        assert_eq!(selected[0].word.as_deref(), Some("ddddd"));
        assert_eq!(selected[0].value, Some(finite_regret_value(expected)));
        assert_ne!(
            finite_regret_value(expected),
            finite_regret_value(frozen_selected)
        );
        let (failure_regret, _, _) = finite_regrets(&selected[0], &global).expect("regret");
        assert!(failure_regret.is_some_and(|regret| regret > 0.0));
    }

    #[test]
    fn same_state_dynamic_reference_counts_dormant_fallback_support() {
        let solver = finite_audit_test_solver(&[
            "aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff", "ggggg",
        ]);
        let mut state = finite_audit_test_state(solver.answers.len());
        state.surviving = vec![0];
        state.fallback_surviving = (1..solver.answers.len()).collect();

        let (global, selected) = solver
            .same_state_dynamic_root_references(
                &state,
                &[],
                2,
                &[0],
                Instant::now(),
                Duration::from_secs(10),
            )
            .expect("bounded dynamic reference");

        assert_eq!(state.surviving.len(), 1);
        assert_eq!(global.status, "unresolved_state_too_large");
        assert_eq!(selected[0].status, "unresolved_state_too_large");
        assert!(global.value.is_none());
        assert!(selected[0].value.is_none());
    }

    #[test]
    fn same_state_dynamic_regret_rejects_non_staged_policy() {
        for mode in [
            crate::config::SearchPolicyMode::ProxyOnly,
            crate::config::SearchPolicyMode::FiniteFastDynamic,
        ] {
            let mut solver = finite_audit_test_solver(&["aaaaa", "bbbbb"]);
            solver.config.search_policy_mode = mode;
            let error = solver
                .same_state_dynamic_regret_report(
                    &ProjectPaths::new("unused"),
                    NaiveDate::from_ymd_opt(2026, 3, 10).expect("date"),
                    1,
                    1,
                )
                .expect_err("non-staged config must be rejected");
            assert!(error.to_string().contains("requires a staged input config"));
        }
    }

    #[test]
    fn same_state_replay_missing_guess_reports_its_runtime_reason() {
        let deadline = same_state_replay_guess(FiniteSearchRegretRuntimeChoice {
            word: None,
            value: None,
            quality: None,
            reason: "global_deadline".to_string(),
        })
        .expect_err("a deadline without a guess must be reported");
        assert!(deadline.to_string().contains("shared deadline elapsed"));

        let missing = same_state_replay_guess(FiniteSearchRegretRuntimeChoice {
            word: None,
            value: None,
            quality: None,
            reason: "not_finite".to_string(),
        })
        .expect_err("a missing suggestion must be reported");
        assert!(missing.to_string().contains("returned no suggestion"));
        assert!(missing.to_string().contains("reason=not_finite"));
    }

    #[test]
    fn finite_search_regret_reports_unresolved_global_deadline() {
        let solver = finite_audit_test_solver(&["aaaaa", "bbbbb", "ccccc"]);
        let state = finite_audit_test_state(solver.answers.len());
        let reference = solver
            .finite_global_reference(&state, &[], 2, false, Instant::now(), Duration::ZERO)
            .expect("bounded reference");
        assert_eq!(reference.status, "unresolved_global_deadline");
        assert!(reference.value.is_none());
    }

    #[test]
    fn search_regret_cancels_inside_exhaustive_state_without_partial_values() {
        let solver =
            finite_audit_test_solver(&["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "abcde"]);
        let state = finite_audit_test_state(solver.answers.len());
        let candidate = SearchRegretCandidateState {
            date: NaiveDate::from_ymd_opt(2032, 1, 2).unwrap(),
            target: "aaaaa".into(),
            turn: 1,
            observations: Vec::new(),
            surviving_answers: state.surviving.len(),
        };
        let polls = std::sync::atomic::AtomicUsize::new(0);
        let cancelled = || {
            let poll = polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if poll == 8 {
                std::thread::sleep(Duration::from_millis(20));
            }
            poll >= 8
        };
        let error = solver
            .audit_search_regret_state(
                &candidate,
                &state,
                Some("aaaaa"),
                PredictiveRegime::Exact,
                "bbbbb",
                "abcde",
                &cancelled,
            )
            .expect_err("slow inner unit must stop the exact audit");
        assert!(format!("{error:#}").contains("cancelled"));
        assert_eq!(polls.load(std::sync::atomic::Ordering::Relaxed), 9);
        let completed = solver
            .audit_search_regret_state(
                &candidate,
                &state,
                Some("aaaaa"),
                PredictiveRegime::Exact,
                "bbbbb",
                "abcde",
                &|| false,
            )
            .unwrap();
        assert!(completed.optimal_exact_cost.is_finite());
        assert!(completed.production.exact_cost >= completed.optimal_exact_cost);
    }

    #[test]
    fn finite_search_regret_expired_final_turn_is_unresolved() {
        let solver = finite_audit_test_solver(&["aaaaa", "bbbbb"]);
        let state = finite_audit_test_state(solver.answers.len());
        let reference = solver
            .finite_fixed_root_reference(&state, &[], 1, false, 0, Instant::now(), Duration::ZERO)
            .unwrap();
        assert_eq!(reference.status, "unresolved_global_deadline");
        assert!(reference.value.is_none());
    }

    #[test]
    fn search_regret_preserves_complete_rows_on_mid_report_and_final_deadlines() {
        let (_fixture, _root, paths) = staged_certificate_test_paths();
        let mut solver = finite_audit_test_solver(&["aaaaa", "bbbbb", "ccccc"]);
        let from = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let to = from.checked_add_days(Days::new(1)).unwrap();
        solver.data_mut().history_dates = [from, to]
            .into_iter()
            .map(|print_date| NytDailyEntry {
                id: None,
                solution: "aaaaa".into(),
                print_date,
                days_since_launch: None,
                editor: None,
            })
            .collect();
        for (completed_states, maximum_states, planned_states) in [(0, 1, 0), (1, 1, 1), (1, 2, 2)]
        {
            let request = SearchRegretRequest {
                from,
                to,
                minimum_survivors: 2,
                maximum_survivors: 3,
                maximum_states,
                maximum_seconds: 60,
            };
            REGRET_STOP_AFTER_STATES.with(|remaining| remaining.set(Some(completed_states)));
            let report = solver.search_regret_report(&paths, request);
            REGRET_STOP_AFTER_STATES.with(|remaining| remaining.set(None));
            let report = report.unwrap();
            assert_eq!(report.schema_version, 2);
            assert!(!report.complete);
            assert!(
                report
                    .stop_reason
                    .as_deref()
                    .unwrap()
                    .contains("global_deadline")
            );
            assert_eq!(report.planned_states, planned_states);
            assert_eq!(report.sampled_states, completed_states);
            assert_eq!(report.states.len(), completed_states);
            assert_eq!(report.production.states, completed_states);
            assert!(
                report
                    .states
                    .iter()
                    .all(|state| state.optimal_exact_cost.is_finite())
            );
            let encoded = serde_json::to_value(&report).unwrap();
            assert_eq!(encoded["complete"], false);
            assert_eq!(
                encoded["states"].as_array().unwrap().len(),
                completed_states
            );
            for policy in ["production", "proxy", "lookahead"] {
                for metric in ["mean_regret", "maximum_regret"] {
                    if completed_states == 0 {
                        assert!(
                            encoded[policy][metric].is_null(),
                            "unmeasured {policy}.{metric}: {}",
                            encoded[policy][metric]
                        );
                    } else {
                        assert_eq!(encoded[policy][metric].as_f64(), Some(0.0));
                    }
                }
            }
            let decoded: SearchRegretReport = serde_json::from_value(encoded.clone()).unwrap();
            assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
        }
        let complete = solver
            .search_regret_report(
                &paths,
                SearchRegretRequest {
                    from,
                    to,
                    minimum_survivors: 2,
                    maximum_survivors: 3,
                    maximum_states: 2,
                    maximum_seconds: 60,
                },
            )
            .unwrap();
        assert!(complete.complete);
        assert_eq!(complete.stop_reason, None);
        assert_eq!(complete.planned_states, complete.states.len());
    }

    #[test]
    fn proposal_reservation_cutoff_is_reported_and_not_exact() {
        let solver = finite_audit_test_solver(&[
            "aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff", "ggggg", "hhhhh", "iiiii",
        ]);
        let state = finite_audit_test_state(solver.answers.len());
        let result = solver
            .finite_horizon_search(
                &state.surviving,
                &state.weights,
                &[],
                3,
                false,
                FiniteSearchOptions {
                    root_shortlist: 1,
                    reply_shortlist: 1,
                    exact_state_threshold: 8,
                    budget: Duration::from_secs(10),
                    node_limit: Some(1),
                    baseline_only: false,
                },
                &|| false,
            )
            .expect("bounded finite search");
        assert!(result.proposal_sampled);
        assert_eq!(result.reason, FiniteSearchReason::NodeBudget);
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.quality != FiniteSearchQuality::Exact)
        );
        let reference =
            finite_reference_from_result(&result, solver.guesses.len(), &solver.guesses);
        assert_eq!(reference.status, "unresolved_node_budget");
        assert!(reference.value.is_none());
    }

    #[test]
    fn finite_search_regret_fixed_root_uses_hard_history_in_children() {
        let solver =
            finite_audit_test_solver(&["allee", "llama", "llava", "llaza", "apple", "ample"]);
        let mut state = finite_audit_test_state(solver.answers.len());
        let observations = vec![("allee".to_string(), score_guess("allee", "llama"))];
        solver
            .apply_feedback(&mut state, &observations[0].0, observations[0].1)
            .expect("condition the reachable hard-mode state");
        assert_eq!(state.surviving.len(), 3);
        let root_index = solver.guess_index["llama"];
        let reference = solver
            .finite_fixed_root_reference(
                &state,
                &observations,
                2,
                true,
                root_index,
                Instant::now(),
                Duration::from_secs(10),
            )
            .expect("hard-mode fixed-root reference");
        assert_eq!(reference.status, "exact");
        assert_eq!(reference.word.as_deref(), Some("llama"));
        let value = reference.value.expect("completed fixed-root value");
        assert!((value.failure_probability - 1.0 / 3.0).abs() < 1e-12);
        assert!((value.expected_attempts - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn posterior_calibration_summary_preserves_overlapping_strata() {
        let first_date = NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let second_date = first_date.checked_add_days(Days::new(1)).expect("date");
        let score = |target_probability, log_loss, brier| crate::experiments::ProbabilityScore {
            target_probability,
            log_loss,
            brier,
        };
        let games = vec![
            ExperimentGameResult {
                target: "alpha".to_string(),
                outcome: GameOutcome::solved(first_date, 2),
                path: vec!["probe".to_string(), "alpha".to_string()],
                finite_search_steps: Vec::new(),
                prior_strata: Some(PriorStrata {
                    never_used: false,
                    reused: true,
                    historical_only: true,
                    out_of_core: true,
                }),
                posterior_calibration: vec![
                    PosteriorCalibrationObservation {
                        turn: 1,
                        score: Some(score(0.5, std::f64::consts::LN_2, 0.5)),
                    },
                    PosteriorCalibrationObservation {
                        turn: 2,
                        score: Some(score(1.0, 0.0, 0.0)),
                    },
                ],
            },
            ExperimentGameResult {
                target: "bravo".to_string(),
                outcome: GameOutcome::coverage_gap(second_date),
                path: Vec::new(),
                finite_search_steps: Vec::new(),
                prior_strata: Some(PriorStrata {
                    never_used: true,
                    reused: false,
                    historical_only: false,
                    out_of_core: false,
                }),
                posterior_calibration: vec![PosteriorCalibrationObservation {
                    turn: 1,
                    score: None,
                }],
            },
        ];
        let summaries = summarize_posterior_calibration(&games);
        validate_posterior_calibration_evidence(&games, &summaries, true, "toy")
            .expect("valid toy calibration");

        let summary = |stratum: &str, turn: u8| {
            summaries
                .iter()
                .find(|summary| summary.stratum == stratum && summary.turn == turn)
                .expect("summary row")
        };
        assert_eq!(summary("all", 1).total_states, 2);
        assert_eq!(summary("all", 1).scored_states, 1);
        assert_eq!(summary("all", 2).total_states, 1);
        assert_eq!(summary("reused", 1).total_states, 1);
        assert_eq!(summary("historical_only", 1).total_states, 1);
        assert_eq!(summary("out_of_core", 1).total_states, 1);
        assert_eq!(summary("never_used", 1).total_states, 1);
        assert_eq!(summary("never_used", 1).scored_states, 0);
        assert_eq!(summary("never_used", 1).mean_score, None);
        assert_eq!(summary("reused", 2).mean_score, Some(score(1.0, 0.0, 0.0)));
    }

    #[test]
    fn posterior_calibration_validation_rejects_bad_turns_and_arithmetic() {
        let date = NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let game = ExperimentGameResult {
            target: "alpha".to_string(),
            outcome: GameOutcome::solved(date, 2),
            path: vec!["probe".to_string(), "alpha".to_string()],
            finite_search_steps: Vec::new(),
            prior_strata: Some(PriorStrata {
                never_used: true,
                reused: false,
                historical_only: false,
                out_of_core: false,
            }),
            posterior_calibration: vec![
                PosteriorCalibrationObservation {
                    turn: 1,
                    score: None,
                },
                PosteriorCalibrationObservation {
                    turn: 2,
                    score: None,
                },
            ],
        };
        let games = vec![game];
        let summaries = summarize_posterior_calibration(&games);
        let mut bad_game = games.clone();
        bad_game[0].posterior_calibration[1].turn = 3;
        let error = validate_posterior_calibration_evidence(&bad_game, &summaries, true, "toy")
            .expect_err("non-contiguous turns");
        assert!(error.to_string().contains("not contiguous"));

        let mut bad_summaries = summaries.clone();
        bad_summaries[0].total_states += 1;
        let error = validate_posterior_calibration_evidence(&games, &bad_summaries, true, "toy")
            .expect_err("summary arithmetic");
        assert!(error.to_string().contains("do not match"));
        for invalid in [
            crate::experiments::ProbabilityScore {
                target_probability: 0.5,
                log_loss: 0.0,
                brier: 0.5,
            },
            crate::experiments::ProbabilityScore {
                target_probability: 1.0,
                log_loss: 0.0,
                brier: 0.5,
            },
        ] {
            assert!(validate_probability_score(invalid, "corrupt row").is_err());
        }
    }

    fn benchmark_evidence_test_fixture() -> PredictiveEvidenceArtifact {
        let date = NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let game = ExperimentGameResult {
            target: "cigar".to_string(),
            outcome: GameOutcome::solved(date, 1),
            path: vec!["cigar".to_string()],
            finite_search_steps: Vec::new(),
            prior_strata: Some(PriorStrata {
                never_used: true,
                reused: false,
                historical_only: false,
                out_of_core: false,
            }),
            posterior_calibration: vec![PosteriorCalibrationObservation {
                turn: 1,
                score: None,
            }],
        };
        let outcome = game.outcome;
        let games = vec![game];
        let canonical = summarize_predictive_outcomes(&[outcome], 7.0, BootstrapConfig::default())
            .expect("metrics");
        let backtest = BacktestStats {
            canonical: canonical.clone(),
            games: canonical.scheduled_games,
            average_guesses: canonical.conditional_mean_guesses,
            p95_guesses: canonical.p95_guesses,
            max_guesses: canonical.max_guesses,
            failures: canonical.unsolved_games + canonical.coverage_gaps,
            coverage_gaps: canonical.coverage_gaps,
            average_guesses_ci95: canonical
                .conditional_mean_guesses_ci95
                .map(|interval| (interval.lower, interval.upper)),
            failure_rate_ci95: (
                1.0 - canonical.solve_rate_ci95.upper,
                1.0 - canonical.solve_rate_ci95.lower,
            ),
        };
        let config_toml = "test = true\n".to_string();
        let baseline = EvidenceBaseline {
            id: "test".to_string(),
            description: "test".to_string(),
            artifacts: "disabled".to_string(),
            effective_config_toml: config_toml.clone(),
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-benchmark-config-v1",
                config_toml.as_bytes(),
            ),
            paired_vs_selected_default: None,
            result: ExperimentResult {
                config_id: "test".to_string(),
                mode: WeightMode::Uniform,
                variant: ModelVariant::SeedPlusHistory,
                backtest,
                average_log_loss: Some(0.0),
                average_brier: Some(0.0),
                average_target_probability: Some(1.0),
                average_target_rank: Some(1.0),
                prior_evidence: Some(
                    summarize_ranked_probability_observations(
                        &[RankedProbabilityObservation {
                            target_rank: 1,
                            top_probability: 1.0,
                            top_prediction_correct: true,
                        }],
                        10,
                        BootstrapConfig::default(),
                    )
                    .unwrap(),
                ),
                posterior_calibration: summarize_posterior_calibration(&games),
                execution: ExecutionTelemetry::default(),
                failure_penalty_sensitivity: Vec::new(),
                latency_p95_ms: 0.0,
                session_fallback_cold_ms: None,
                session_fallback_warm_ms: None,
                proxy_step_pct: 0.0,
                lookahead_step_pct: 0.0,
                escalated_exact_step_pct: 0.0,
                exact_step_pct: 0.0,
                finite_step_pct: 0.0,
                terminal_step_pct: 0.0,
                average_lookahead_pool_ratio: 0.0,
                average_exact_pool_ratio: 0.0,
                games,
            },
        };
        let range = DateRange::new(date, date).expect("range");
        PredictiveEvidenceArtifact {
            schema_version: BENCHMARK_EVIDENCE_SCHEMA_VERSION,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint: crate::identity::digest_bytes_tagged("test", b"source"),
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-benchmark-root-config-v1",
                config_toml.as_bytes(),
            ),
            scope: "test".to_string(),
            sealed_test_evaluated: false,
            evaluation_from: date,
            evaluation_to: date,
            evaluation_selection: "range".to_string(),
            selected_ranges: vec![range],
            matrix_source: "test".to_string(),
            matrix_fingerprint: crate::identity::digest_bytes_tagged("test", b"matrix"),
            profile_ids: vec!["test".to_string()],
            reference_profile_id: "test".to_string(),
            history_snapshot_start: date,
            history_snapshot_end: date,
            code_revision: None,
            code_dirty: None,
            platform: "test".to_string(),
            cpu: None,
            release_command: "test".to_string(),
            config_toml,
            resource_budget: EvidenceResourceBudget::default(),
            resources: EvidenceResourceTelemetry::default(),
            historical_diagnostic: HistoricalDiagnosticBaseline {
                date_range: "test".to_string(),
                scheduled_games: 0,
                modeled_games: 0,
                coverage_gaps: 0,
                conditional_mean_guesses: 0.0,
                average_log_loss: 0.0,
                average_brier_score: 0.0,
                interpretation: "test".to_string(),
            },
            baselines: vec![baseline],
            limitations: Vec::new(),
        }
    }

    #[test]
    fn empty_forced_target_evaluations_are_not_successful_zero_costs() {
        let solver = super::super::tests::test_solver(&["cigar", "rebut"]);
        for result in [
            solver.evaluate_forced_opener(&[], 0, &|| false).map(|_| ()),
            solver
                .evaluate_named_opener_on_targets(&[], "cigar", 1)
                .map(|_| ()),
            solver
                .evaluate_forced_continuation(&["cigar".to_string()], &[], 0, &|| false)
                .map(|_| ()),
        ] {
            assert!(
                result
                    .expect_err("empty target population")
                    .to_string()
                    .contains("at least one target")
            );
        }
    }

    #[test]
    fn terminal_search_telemetry_and_merge_have_their_own_counter() {
        let step = DetailedSolveStep {
            guess: "cigar".into(),
            feedback: 242,
            surviving_before: 1,
            surviving_after: 1,
            chosen_force_in_two: false,
            alternative_force_in_two: false,
            danger_score: 0.0,
            danger_escalated: false,
            regime_used: PredictiveRegime::Terminal,
            promotion_source: None,
            recovery_mode_used: None,
            fallback_active: false,
            lookahead_pool_base: 0,
            lookahead_pool_size: 0,
            exact_pool_base: 0,
            exact_pool_size: 0,
            root_candidate_count: 1,
            top_suggestions: Vec::new(),
            finite_search: None,
        };
        let runs = [DetailedSolveRun {
            target: "cigar".into(),
            date: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            steps: vec![step],
            solved: true,
        }];
        let measured = Solver::execution_telemetry(&runs);
        assert_eq!(measured.total_steps, 1);
        assert_eq!(measured.terminal_steps, 1);
        assert_eq!(
            measured.proxy_steps + measured.exact_steps + measured.finite_steps,
            0
        );
        assert_eq!(Solver::regime_mix(&runs), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0));
        let mut merged = ExecutionTelemetry::default();
        merge_execution_telemetry(&mut merged, &measured);
        assert_eq!(merged.terminal_steps, 1);
    }

    #[test]
    fn all_gap_benchmark_markdown_reports_unavailable_modeled_metrics() {
        let mut artifact = benchmark_evidence_test_fixture();
        let baseline = &mut artifact.baselines[0];
        baseline.result.average_log_loss = None;
        baseline.result.average_brier = None;
        baseline.result.average_target_probability = None;
        baseline.result.average_target_rank = None;
        baseline.result.prior_evidence = None;
        let outcome = GameOutcome::coverage_gap(artifact.evaluation_from);
        let metrics = summarize_predictive_outcomes(&[outcome], 7.0, BootstrapConfig::default())
            .expect("gap metrics");
        let summary = &mut baseline.result.backtest;
        summary.canonical = metrics.clone();
        summary.average_guesses = None;
        summary.p95_guesses = None;
        summary.max_guesses = None;
        summary.average_guesses_ci95 = None;
        summary.failures = 1;
        summary.coverage_gaps = 1;
        summary.failure_rate_ci95 = (
            1.0 - metrics.solve_rate_ci95.upper,
            1.0 - metrics.solve_rate_ci95.lower,
        );
        let game = &mut baseline.result.games[0];
        game.outcome = outcome;
        game.path.clear();
        game.prior_strata = None;
        game.posterior_calibration = vec![PosteriorCalibrationObservation {
            turn: 1,
            score: None,
        }];
        baseline.result.posterior_calibration =
            summarize_posterior_calibration(&baseline.result.games);
        baseline.paired_vs_selected_default = Some(
            PairedDifference::all_game_penalized(
                &[outcome],
                &[outcome],
                7.0,
                BootstrapConfig::default(),
            )
            .expect("paired gap"),
        );
        let markdown = Solver::render_development_evidence_markdown(&artifact)
            .expect("render all-gap evidence");
        assert!(markdown.contains("unavailable (modeled_games=0)"));
        assert!(markdown.contains("unavailable (measured_prior_games=0/1)"));
        assert!(markdown.contains("7.0000 [7.0000, 7.0000]"));
        assert!(!markdown.contains("NaN"));
        assert!(!markdown.contains("inf"));
        let json = serde_json::to_string(&artifact).expect("finite JSON");
        let decoded: PredictiveEvidenceArtifact =
            serde_json::from_str(&json).expect("artifact round trip");
        decoded.validate_identity().expect("valid all-gap evidence");
    }

    #[test]
    fn all_gap_experiment_prior_averages_are_null_and_round_trip() {
        let date = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        let (_directory, solver) = parity_solver(crate::config::SearchPolicyMode::ProxyOnly, date);
        let gap = NytDailyEntry {
            id: None,
            solution: "zzzzz".to_string(),
            print_date: date,
            days_since_launch: None,
            editor: None,
        };
        let report = solver
            .experiment_report_for_selected_games_with_book_usage_and_progress(
                &[&gap],
                3,
                PredictiveBookUsage::None,
                None,
            )
            .unwrap();
        assert_eq!(report.backtest.canonical.scheduled_games, 1);
        assert_eq!(report.backtest.canonical.coverage_gaps, 1);
        assert!(report.prior_evidence.is_none());
        let json = serde_json::to_value(&report).unwrap();
        for metric in [
            "average_log_loss",
            "average_brier",
            "average_target_probability",
            "average_target_rank",
        ] {
            assert!(
                json[metric].is_null(),
                "unmeasured {metric}: {}",
                json[metric]
            );
        }
        let decoded: ExperimentResult = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json);
        let measured = NytDailyEntry {
            solution: "cigar".to_string(),
            print_date: date.succ_opt().unwrap(),
            ..gap.clone()
        };
        let expected = solver
            .initial_prior_metrics(&measured.solution, measured.print_date)
            .unwrap();
        let mixed = solver
            .experiment_report_for_selected_games_with_book_usage_and_progress(
                &[&gap, &measured],
                3,
                PredictiveBookUsage::None,
                None,
            )
            .unwrap();
        assert_eq!(mixed.backtest.canonical.scheduled_games, 2);
        assert_eq!(mixed.prior_evidence.unwrap().measured_games, 1);
        assert_eq!(mixed.average_log_loss, Some(expected.log_loss));
        assert_eq!(mixed.average_brier, Some(expected.brier));
        assert_eq!(
            mixed.average_target_probability,
            Some(expected.target_probability)
        );
        assert_eq!(mixed.average_target_rank, Some(expected.target_rank as f64));
    }

    #[test]
    fn benchmark_prior_means_require_a_measured_population() {
        let artifact = benchmark_evidence_test_fixture();
        let mut baseline = artifact.baselines[0].clone();
        baseline.result.prior_evidence = None;
        assert!(validate_evidence_baseline(&baseline).is_err());
        let mut baseline = artifact.baselines[0].clone();
        baseline.result.average_log_loss = None;
        assert!(validate_evidence_baseline(&baseline).is_err());
        for invalid in [f64::NAN, f64::INFINITY, -1.0] {
            let mut baseline = artifact.baselines[0].clone();
            baseline.result.average_log_loss = Some(invalid);
            assert!(validate_evidence_baseline(&baseline).is_err());
        }
        let mut baseline = artifact.baselines[0].clone();
        baseline
            .result
            .prior_evidence
            .as_mut()
            .unwrap()
            .measured_games = 0;
        assert!(validate_evidence_baseline(&baseline).is_err());
    }

    #[test]
    fn all_gap_prior_ablation_is_null_and_cannot_promote() {
        let date = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        let (directory, _solver) = parity_solver(crate::config::SearchPolicyMode::ProxyOnly, date);
        let paths = ProjectPaths::new(directory.path());
        fs::write(&paths.seed_answers, "cigar\n").unwrap();
        fs::write(paths.root.join("config/evaluation.toml"),
            "format_version = 1\ndevelopment_cutoff = '2030-01-02'\n[sealed_test]\nstart = '2030-01-03'\nend = '2030-01-03'\n").unwrap();
        let history = ["cigar", "humph", "rebut"]
            .into_iter()
            .enumerate()
            .map(|(offset, word)| NytDailyEntry {
                id: None,
                solution: word.to_string(),
                print_date: date.checked_add_days(Days::new(offset as u64)).unwrap(),
                days_since_launch: None,
                editor: None,
            })
            .collect::<Vec<_>>();
        crate::data::write_history_jsonl(&paths.raw_history, &history).unwrap();
        let report = Solver::predictive_prior_ablation_report(
            &paths,
            &PriorConfig::default(),
            &[
                "weighted_baseline".to_string(),
                "empirical_frequency_baseline".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(report.schema_version, 2);
        for profile in &report.profiles {
            assert_eq!(profile.scheduled_games, 1);
            assert_eq!(profile.measured_games, 0);
            assert_eq!(profile.coverage_gaps, 1);
            assert_eq!(profile.average_log_loss, None);
            assert_eq!(profile.average_brier, None);
            assert_eq!(profile.folds[0].average_log_loss, None);
            assert_eq!(profile.folds[0].average_brier, None);
            assert!(!profile.promotable && !profile.promotion_blockers.is_empty());
        }
        let json = serde_json::to_value(&report).unwrap();
        assert!(json["profiles"][0]["average_log_loss"].is_null());
        let decoded: PredictivePriorAblationReport = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json);
    }

    #[test]
    fn benchmark_evidence_rejects_sealed_test_flag() {
        let mut artifact = benchmark_evidence_test_fixture();
        artifact.sealed_test_evaluated = true;
        let error = artifact
            .validate_identity()
            .expect_err("sealed-test evidence must be rejected");
        assert!(
            error
                .to_string()
                .contains("benchmark evidence must not evaluate the sealed test"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn prior_strata_do_not_look_through_future_history() {
        let as_of = NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let answer = AnswerRecord {
            word: "alpha".to_string(),
            in_seed: false,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![
                as_of.checked_add_days(Days::new(1)).expect("date"),
                as_of.checked_add_days(Days::new(2)).expect("date"),
            ],
        };
        assert_eq!(
            Solver::prior_strata_for_answer(&answer, 0.0, as_of),
            PriorStrata {
                never_used: true,
                reused: false,
                historical_only: false,
                out_of_core: true,
            }
        );
        let answer = AnswerRecord {
            history_dates: vec![
                as_of.checked_sub_days(Days::new(1)).expect("date"),
                as_of.checked_add_days(Days::new(1)).expect("date"),
            ],
            ..answer
        };
        assert_eq!(
            Solver::prior_strata_for_answer(&answer, 1.0, as_of),
            PriorStrata {
                never_used: false,
                reused: true,
                historical_only: true,
                out_of_core: false,
            }
        );
    }

    #[test]
    fn finite_studies_reject_competing_trials_before_loading_data() {
        let spec: StudySpec = serde_json::from_str(
            r#"{"name":"finite-budget","stage":"calibration","seed":1,"trial_count":2,"parallelism":2}"#,
        ).unwrap();
        for mode in [
            crate::config::SearchPolicyMode::FiniteBaseline,
            crate::config::SearchPolicyMode::FiniteFast,
            crate::config::SearchPolicyMode::FiniteStrong,
        ] {
            let config = PriorConfig {
                search_policy_mode: mode,
                ..PriorConfig::default()
            };
            let root = Path::new("unused-finite-study-fixture");
            let error = Solver::run_predictive_study(
                &ProjectPaths::new(root),
                &config,
                spec.clone(),
                &root.join("study.json"),
                5,
                None,
            )
            .unwrap_err();
            assert!(error.to_string().contains("require --jobs 1"));
        }
    }

    fn rolling_checkpoint_test_fixture() -> (RollingEvaluationCheckpoint, EvaluationPlan, NaiveDate)
    {
        let validation_date = NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let training_end = validation_date
            .checked_sub_days(chrono::Days::new(1))
            .expect("training end");
        let training_start = training_end
            .checked_sub_days(chrono::Days::new(1))
            .expect("training start");
        let history = DateRange::new(training_start, validation_date).expect("history");
        let validation = DateRange::new(validation_date, validation_date).expect("validation");
        let plan = EvaluationPlan {
            history,
            development: history,
            sealed_test: validation,
            excluded_target_ranges: Vec::new(),
            folds: vec![RollingOriginFold {
                index: 0,
                training: DateRange::new(training_start, training_end).expect("training"),
                validation,
            }],
            config: RollingOriginConfig {
                minimum_training_days: 2,
                validation_days: 1,
                step_days: 1,
                sealed_test_days: 1,
                maximum_folds: 1,
            },
        };
        let outcomes = [GameOutcome::solved(validation_date, 1)];
        let metrics = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())
            .expect("metrics");
        let checkpoint = RollingEvaluationCheckpoint {
            schema_version: ROLLING_CHECKPOINT_SCHEMA_VERSION,
            source_identity: "source".to_string(),
            evaluation_plan: plan.clone(),
            label: "test".to_string(),
            config_toml: toml::to_string_pretty(&PriorConfig {
                search_policy_mode: crate::config::SearchPolicyMode::ProxyOnly,
                ..PriorConfig::default()
            })
            .expect("config"),
            folds: vec![RollingFoldEvidence {
                fold_index: 0,
                validation,
                metrics,
            }],
            games: vec![ExperimentGameResult {
                target: "cigar".to_string(),
                outcome: outcomes[0],
                path: vec!["cigar".to_string()],
                finite_search_steps: Vec::new(),
                prior_strata: None,
                posterior_calibration: Vec::new(),
            }],
            prior_observations: Vec::new(),
            execution: ExecutionTelemetry::default(),
        };
        (checkpoint, plan, validation_date)
    }

    fn validate_test_rolling_checkpoint(
        checkpoint: &RollingEvaluationCheckpoint,
        plan: &EvaluationPlan,
    ) -> Result<()> {
        validate_rolling_checkpoint(
            checkpoint,
            "source",
            "test",
            &checkpoint.config_toml,
            plan,
            true,
        )
    }

    #[test]
    fn serialized_latency_only_selects_complete_unmeasured_finalists() {
        assert!(needs_serial_study_latency(
            TrialStatus::Complete,
            12,
            12,
            false
        ));
        assert!(!needs_serial_study_latency(
            TrialStatus::Running,
            12,
            12,
            false
        ));
        assert!(!needs_serial_study_latency(
            TrialStatus::Complete,
            11,
            12,
            false
        ));
        assert!(!needs_serial_study_latency(
            TrialStatus::Complete,
            12,
            12,
            true
        ));
    }

    #[test]
    fn evidence_checkpoint_rejects_another_identity() {
        let resource_budget = EvidenceResourceBudget {
            maximum_seconds: 10,
            maximum_memory_mb: 20,
        };
        let checkpoint = EvidenceMatrixCheckpoint {
            schema_version: EVIDENCE_CHECKPOINT_SCHEMA_VERSION,
            identity: "first".into(),
            resource_budget: Some(resource_budget),
            rayon_threads: Some(8),
            elapsed_ms: 10,
            peak_working_set_bytes: 20,
            baselines: Vec::new(),
        };
        assert!(
            checkpoint
                .validate("first", &[], resource_budget, 8)
                .is_ok()
        );
        assert!(
            checkpoint
                .validate("second", &[], resource_budget, 8)
                .is_err()
        );
        assert!(
            checkpoint
                .validate(
                    "first",
                    &[],
                    EvidenceResourceBudget {
                        maximum_seconds: 11,
                        ..resource_budget
                    },
                    8,
                )
                .is_err()
        );
        assert!(
            checkpoint
                .validate(
                    "first",
                    &[],
                    EvidenceResourceBudget {
                        maximum_memory_mb: 21,
                        ..resource_budget
                    },
                    8,
                )
                .is_err()
        );
        assert!(
            checkpoint
                .validate("first", &[], resource_budget, 9)
                .is_err()
        );
        let mut old = checkpoint;
        old.schema_version = EVIDENCE_CHECKPOINT_SCHEMA_VERSION - 1;
        assert!(old.validate("first", &[], resource_budget, 8).is_err());
    }

    #[test]
    fn evidence_checkpoint_serialization_round_trips_resource_inputs() {
        let resource_budget = EvidenceResourceBudget {
            maximum_seconds: 10,
            maximum_memory_mb: 20,
        };
        let checkpoint = EvidenceMatrixCheckpoint {
            schema_version: EVIDENCE_CHECKPOINT_SCHEMA_VERSION,
            identity: "first".into(),
            resource_budget: Some(resource_budget),
            rayon_threads: Some(8),
            elapsed_ms: 10,
            peak_working_set_bytes: 20,
            baselines: Vec::new(),
        };
        let decoded: EvidenceMatrixCheckpoint =
            serde_json::from_slice(&serde_json::to_vec(&checkpoint).expect("serialize checkpoint"))
                .expect("deserialize checkpoint");
        assert_eq!(decoded.schema_version, checkpoint.schema_version);
        assert_eq!(decoded.identity, checkpoint.identity);
        assert_eq!(decoded.resource_budget, checkpoint.resource_budget);
        assert_eq!(decoded.rayon_threads, checkpoint.rayon_threads);
        assert_eq!(decoded.elapsed_ms, checkpoint.elapsed_ms);
        assert_eq!(
            decoded.peak_working_set_bytes,
            checkpoint.peak_working_set_bytes
        );
        assert_eq!(decoded.baselines.len(), checkpoint.baselines.len());
    }

    #[test]
    fn legacy_v3_checkpoint_reaches_explicit_schema_rejection() {
        let raw = r#"{
            "schema_version": 3,
            "identity": "first",
            "elapsed_ms": 10,
            "peak_working_set_bytes": 20,
            "baselines": []
        }"#;
        let decoded = serde_json::from_str::<EvidenceMatrixCheckpoint>(raw);
        assert!(
            decoded.is_ok(),
            "legacy v3 JSON should deserialize before schema validation: {decoded:?}"
        );
        let checkpoint = decoded.expect("legacy checkpoint");
        let error = checkpoint
            .validate(
                "first",
                &[],
                EvidenceResourceBudget {
                    maximum_seconds: 10,
                    maximum_memory_mb: 20,
                },
                8,
            )
            .expect_err("legacy schema must be rejected explicitly");
        assert_eq!(error.to_string(), "unsupported evidence checkpoint schema");
    }

    #[test]
    fn malformed_v4_checkpoint_cannot_resume_without_resource_inputs() {
        let raw = r#"{
            "schema_version": 4,
            "identity": "first",
            "elapsed_ms": 10,
            "peak_working_set_bytes": 20,
            "baselines": []
        }"#;
        let checkpoint = serde_json::from_str::<EvidenceMatrixCheckpoint>(raw)
            .expect("malformed v4 checkpoint should deserialize for validation");
        assert!(
            checkpoint
                .validate(
                    "first",
                    &[],
                    EvidenceResourceBudget {
                        maximum_seconds: 10,
                        maximum_memory_mb: 20,
                    },
                    8,
                )
                .is_err(),
            "v4 checkpoint missing resource inputs must not resume"
        );
    }

    #[test]
    fn v4_checkpoint_missing_budget_cannot_resume_under_default_budget() {
        let raw = r#"{
            "schema_version": 4,
            "identity": "first",
            "rayon_threads": 8,
            "elapsed_ms": 10,
            "peak_working_set_bytes": 20,
            "baselines": []
        }"#;
        let checkpoint = serde_json::from_str::<EvidenceMatrixCheckpoint>(raw)
            .expect("v4 checkpoint should deserialize for validation");
        assert!(
            checkpoint
                .validate("first", &[], EvidenceResourceBudget::default(), 8)
                .is_err(),
            "v4 checkpoint missing resource budget must not resume under the default budget"
        );
    }

    #[test]
    fn evidence_checkpoint_profiles_must_be_an_ordered_prefix() {
        let expected = vec!["first".to_string(), "second".to_string()];
        assert!(is_profile_prefix(&[], &expected));
        assert!(is_profile_prefix(&["first".to_string()], &expected));
        assert!(!is_profile_prefix(&["second".to_string()], &expected));
        assert!(!is_profile_prefix(
            &[
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ],
            &expected
        ));
    }

    #[test]
    fn evidence_checkpoint_identity_changes_for_resolved_base_config_content() {
        let date = NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let range = DateRange::new(date, date).expect("range");
        let plan = EvaluationPlan {
            history: range,
            development: range,
            sealed_test: range,
            excluded_target_ranges: Vec::new(),
            folds: Vec::new(),
            config: RollingOriginConfig {
                minimum_training_days: 1,
                validation_days: 1,
                step_days: 1,
                sealed_test_days: 1,
                maximum_folds: 1,
            },
        };
        let base = vec![("previous".to_string(), "base-a".to_string())];
        let changed = vec![("previous".to_string(), "base-b".to_string())];
        let profiles = vec!["profile".to_string()];
        let budget = EvidenceResourceBudget {
            maximum_seconds: 10,
            maximum_memory_mb: 20,
        };
        let first = evidence_checkpoint_identity(
            "input",
            "config",
            &plan,
            "range",
            &[range],
            "matrix",
            "matrix-fingerprint",
            &profiles,
            &base,
            range,
            3,
            budget,
            8,
        )
        .expect("identity");
        let second = evidence_checkpoint_identity(
            "input",
            "config",
            &plan,
            "range",
            &[range],
            "matrix",
            "matrix-fingerprint",
            &profiles,
            &changed,
            range,
            3,
            budget,
            8,
        )
        .expect("identity");
        assert_ne!(first, second);

        let rolling = evidence_checkpoint_identity(
            "input",
            "config",
            &plan,
            "rolling_folds",
            &[range],
            "matrix",
            "matrix-fingerprint",
            &profiles,
            &base,
            range,
            3,
            budget,
            8,
        )
        .expect("identity");
        assert_ne!(first, rolling);

        let changed_matrix = evidence_checkpoint_identity(
            "input",
            "config",
            &plan,
            "range",
            &[range],
            "matrix",
            "changed-matrix-fingerprint",
            &profiles,
            &base,
            range,
            3,
            budget,
            8,
        )
        .expect("identity");
        assert_ne!(first, changed_matrix);

        let changed_seconds = evidence_checkpoint_identity(
            "input",
            "config",
            &plan,
            "range",
            &[range],
            "matrix",
            "matrix-fingerprint",
            &profiles,
            &base,
            range,
            3,
            EvidenceResourceBudget {
                maximum_seconds: 11,
                ..budget
            },
            8,
        )
        .expect("identity");
        assert_ne!(first, changed_seconds);

        let changed_memory = evidence_checkpoint_identity(
            "input",
            "config",
            &plan,
            "range",
            &[range],
            "matrix",
            "matrix-fingerprint",
            &profiles,
            &base,
            range,
            3,
            EvidenceResourceBudget {
                maximum_memory_mb: 21,
                ..budget
            },
            8,
        )
        .expect("identity");
        assert_ne!(first, changed_memory);

        let changed_threads = evidence_checkpoint_identity(
            "input",
            "config",
            &plan,
            "range",
            &[range],
            "matrix",
            "matrix-fingerprint",
            &profiles,
            &base,
            range,
            3,
            budget,
            9,
        )
        .expect("identity");
        assert_ne!(first, changed_threads);
    }

    #[test]
    fn selected_evidence_games_uses_exact_ranges_without_gap_or_seal_leakage() {
        let entry = |raw: &str, solution: &str| NytDailyEntry {
            id: None,
            solution: solution.to_string(),
            print_date: NaiveDate::parse_from_str(raw, "%Y-%m-%d").expect("date"),
            days_since_launch: None,
            editor: None,
        };
        let history = vec![
            entry("2026-06-17", "cigar"),
            entry("2026-06-18", "rebut"),
            entry("2026-07-18", "sissy"),
            entry("2026-08-26", "humph"),
            entry("2026-08-28", "awake"),
        ];
        let selected_ranges = vec![
            DateRange::new(
                NaiveDate::from_ymd_opt(2026, 6, 17).expect("date"),
                NaiveDate::from_ymd_opt(2026, 6, 17).expect("date"),
            )
            .expect("range"),
            DateRange::new(
                NaiveDate::from_ymd_opt(2026, 7, 18).expect("date"),
                NaiveDate::from_ymd_opt(2026, 7, 18).expect("date"),
            )
            .expect("range"),
            DateRange::new(
                NaiveDate::from_ymd_opt(2026, 8, 26).expect("date"),
                NaiveDate::from_ymd_opt(2026, 8, 26).expect("date"),
            )
            .expect("range"),
        ];
        let selected = selected_evidence_games(&history, &selected_ranges)
            .expect("selected games")
            .into_iter()
            .map(|entry| entry.print_date)
            .collect::<Vec<_>>();
        assert_eq!(
            selected,
            [
                NaiveDate::from_ymd_opt(2026, 6, 17).expect("date"),
                NaiveDate::from_ymd_opt(2026, 7, 18).expect("date"),
                NaiveDate::from_ymd_opt(2026, 8, 26).expect("date"),
            ]
        );
    }

    fn rolling_comparison_test_fixture() -> RollingComparisonArtifact {
        let (checkpoint, plan, _) = rolling_checkpoint_test_fixture();
        let config = RollingConfigEvidence {
            label: checkpoint.label,
            config_toml: checkpoint.config_toml.clone(),
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-rolling-config-v1",
                checkpoint.config_toml.as_bytes(),
            ),
            folds: checkpoint.folds.clone(),
            aggregate: checkpoint.folds[0].metrics.clone(),
            prior_evidence: None,
            execution: checkpoint.execution,
            failure_penalty_sensitivity: Vec::new(),
            games: checkpoint.games.clone(),
            latency_p95_ms: 0.0,
        };
        let outcomes = checkpoint
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        RollingComparisonArtifact {
            schema_version: 5,
            top: 5,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint: crate::identity::digest_bytes_tagged("test", b"source"),
            evaluation_plan: plan,
            sealed_test_evaluated: false,
            code_revision: None,
            code_dirty: None,
            baseline: config.clone(),
            candidate: config,
            candidate_minus_baseline: PairedDifference::all_game_penalized(
                &outcomes,
                &outcomes,
                7.0,
                BootstrapConfig::default(),
            )
            .unwrap(),
        }
    }

    #[test]
    fn rolling_metric_validation_survives_fractional_json_roundtrips() {
        let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        for count in [3, 7, 17, 30] {
            let mut baseline = (0..count)
                .map(|index| {
                    GameOutcome::solved(start + chrono::Days::new(index), (index as usize % 6) + 1)
                })
                .collect::<Vec<_>>();
            let candidate = baseline
                .iter()
                .enumerate()
                .map(|(index, outcome)| GameOutcome::solved(outcome.date, (index % 5) + 1))
                .collect::<Vec<_>>();
            let metrics =
                summarize_predictive_outcomes(&baseline, 7.0, BootstrapConfig::default()).unwrap();
            let decoded = serde_json::from_slice(&serde_json::to_vec(&metrics).unwrap()).unwrap();
            validate_predictive_metrics(&decoded, &mut baseline, "roundtrip").unwrap();
            let paired = PairedDifference::all_game_penalized(
                &baseline,
                &candidate,
                7.0,
                BootstrapConfig::default(),
            )
            .unwrap();
            let decoded: PairedDifference =
                serde_json::from_slice(&serde_json::to_vec(&paired).unwrap()).unwrap();
            assert_eq!(paired, decoded);
        }
    }

    #[test]
    fn rolling_final_artifact_requires_top_and_current_schema() {
        let artifact = rolling_comparison_test_fixture();
        let value = serde_json::to_value(&artifact).unwrap();
        let decoded: RollingComparisonArtifact = serde_json::from_value(value.clone()).unwrap();
        validate_rolling_comparison_artifact(&decoded).unwrap();
        assert_eq!(decoded.top, 5);
        let rendered =
            Solver::render_rolling_comparison_markdown(std::slice::from_ref(&decoded)).unwrap();
        assert!(rendered.contains("| `test` |"));
        assert!(!rendered.contains("`current_default`"));
        assert!(!rendered.contains("subsequent once-only evaluation"));
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("top");
        assert!(serde_json::from_value::<RollingComparisonArtifact>(missing).is_err());
        let mut old: RollingComparisonArtifact = serde_json::from_value(value).unwrap();
        old.schema_version = 3;
        assert!(old.validate_identity().is_err());
    }

    #[test]
    fn rolling_final_artifact_accepts_historical_finite_games_without_traces() {
        let mut artifact = rolling_comparison_test_fixture();
        let mut config = PriorConfig::default();
        config.search_policy_mode = crate::config::SearchPolicyMode::FiniteFast;
        let config_toml = toml::to_string_pretty(&config).expect("finite config");
        let fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-rolling-config-v1",
            config_toml.as_bytes(),
        );
        for evidence in [&mut artifact.baseline, &mut artifact.candidate] {
            evidence.config_toml = config_toml.clone();
            evidence.config_fingerprint = fingerprint.clone();
        }
        assert!(artifact.baseline.games[0].finite_search_steps.is_empty());
        validate_rolling_comparison_artifact(&artifact)
            .expect("historical finite artifact without optional traces remains readable");
    }

    #[test]
    fn rolling_final_artifact_rejects_path_guess_mismatch_without_posterior_calibration() {
        let mut artifact = rolling_comparison_test_fixture();
        artifact.baseline.games[0].path = vec!["cigar".to_string(), "rebut".to_string()];
        let error = validate_rolling_comparison_artifact(&artifact)
            .expect_err("path count must match a solved outcome");
        assert!(
            error
                .to_string()
                .contains("outcome guess count does not match its path"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rolling_checkpoint_accepts_unsolved_games_and_coverage_gaps() {
        let (mut checkpoint, plan, validation_date) = rolling_checkpoint_test_fixture();
        checkpoint.games[0].outcome = GameOutcome::unsolved(validation_date, 6);
        checkpoint.games[0].path = vec!["cigar".to_string(); 6];
        checkpoint.folds[0].metrics = summarize_predictive_outcomes(
            &[checkpoint.games[0].outcome],
            7.0,
            BootstrapConfig::default(),
        )
        .expect("unsolved metrics");
        validate_test_rolling_checkpoint(&checkpoint, &plan)
            .expect("unsolved path count should remain valid");

        checkpoint.games[0].outcome = GameOutcome::coverage_gap(validation_date);
        checkpoint.games[0].path.clear();
        checkpoint.folds[0].metrics = summarize_predictive_outcomes(
            &[checkpoint.games[0].outcome],
            7.0,
            BootstrapConfig::default(),
        )
        .expect("coverage-gap metrics");
        validate_test_rolling_checkpoint(&checkpoint, &plan)
            .expect("coverage gaps should retain empty paths");
    }

    #[test]
    fn rolling_final_artifact_rejects_corrupt_structure_and_arithmetic() {
        let artifact = rolling_comparison_test_fixture();
        validate_rolling_comparison_artifact(&artifact).expect("valid artifact");

        let mut no_top = artifact.clone();
        no_top.top = 0;
        let error = validate_rolling_comparison_artifact(&no_top).expect_err("zero top");
        assert!(error.to_string().contains("top must be positive"));

        let mut duplicate_fold = artifact.clone();
        duplicate_fold
            .candidate
            .folds
            .push(duplicate_fold.candidate.folds[0].clone());
        let error = validate_rolling_comparison_artifact(&duplicate_fold)
            .expect_err("duplicate candidate fold");
        assert!(
            error.to_string().contains("duplicate fold id"),
            "unexpected error: {error}"
        );

        let mut duplicate_game = artifact.clone();
        duplicate_game
            .candidate
            .games
            .push(duplicate_game.candidate.games[0].clone());
        let error = validate_rolling_comparison_artifact(&duplicate_game)
            .expect_err("duplicate candidate game");
        assert!(error.to_string().contains("duplicate game date"));

        let mut out_of_range = artifact.clone();
        out_of_range.candidate.games[0].outcome =
            GameOutcome::solved(NaiveDate::from_ymd_opt(2026, 8, 2).expect("date"), 1);
        let error = validate_rolling_comparison_artifact(&out_of_range)
            .expect_err("out-of-range candidate game");
        assert!(error.to_string().contains("planned validation range"));

        let mut bad_aggregate = artifact.clone();
        bad_aggregate.candidate.aggregate.solved_games = 0;
        let error =
            validate_rolling_comparison_artifact(&bad_aggregate).expect_err("aggregate arithmetic");
        assert!(error.to_string().contains("metrics do not match"));

        let mut bad_paired = artifact;
        bad_paired.candidate_minus_baseline.candidate_wins = 1;
        let error =
            validate_rolling_comparison_artifact(&bad_paired).expect_err("paired arithmetic");
        assert!(error.to_string().contains("paired difference"));
    }

    #[test]
    fn rolling_checkpoint_identity_changes_with_top() {
        assert_ne!(
            rolling_checkpoint_fingerprint_with_top("config", "source", 1),
            rolling_checkpoint_fingerprint_with_top("config", "source", 2)
        );
    }

    #[test]
    fn rolling_checkpoint_rejects_malformed_fold_and_game_overlaps() {
        let (checkpoint, plan, _) = rolling_checkpoint_test_fixture();
        validate_test_rolling_checkpoint(&checkpoint, &plan).expect("valid checkpoint");

        let mut old_schema = checkpoint.clone();
        old_schema.schema_version = 1;
        let error = validate_test_rolling_checkpoint(&old_schema, &plan).expect_err("old schema");
        assert!(
            error
                .to_string()
                .contains("unsupported rolling checkpoint schema")
        );

        let mut duplicate_fold = checkpoint.clone();
        duplicate_fold.folds.push(duplicate_fold.folds[0].clone());
        let error =
            validate_test_rolling_checkpoint(&duplicate_fold, &plan).expect_err("duplicate fold");
        assert!(error.to_string().contains("duplicate fold id"));

        let mut duplicate_game = checkpoint.clone();
        duplicate_game.games.push(duplicate_game.games[0].clone());
        let error = validate_test_rolling_checkpoint(&duplicate_game, &plan)
            .expect_err("duplicate game date");
        assert!(error.to_string().contains("duplicate game date"));

        let mut wrong_range = checkpoint.clone();
        wrong_range.folds[0].validation = DateRange::new(
            wrong_range.folds[0]
                .validation
                .start
                .checked_sub_days(chrono::Days::new(1))
                .expect("date"),
            wrong_range.folds[0].validation.end,
        )
        .expect("range");
        let error = validate_test_rolling_checkpoint(&wrong_range, &plan).expect_err("wrong range");
        assert!(error.to_string().contains("validation range"));

        let mut out_of_range = checkpoint;
        out_of_range.games[0].outcome =
            GameOutcome::solved(NaiveDate::from_ymd_opt(2026, 8, 2).expect("date"), 1);
        let error =
            validate_test_rolling_checkpoint(&out_of_range, &plan).expect_err("out-of-range game");
        assert!(
            error
                .to_string()
                .contains("does not belong to exactly one planned validation range")
        );
    }

    #[test]
    fn finite_rolling_checkpoint_rejects_missing_or_misaligned_search_traces() {
        let (mut checkpoint, plan, _) = rolling_checkpoint_test_fixture();
        let mut config = PriorConfig::default();
        config.search_policy_mode = crate::config::SearchPolicyMode::FiniteFast;
        checkpoint.config_toml = toml::to_string(&config).expect("finite config");
        let trace = FiniteSearchStepEvidence {
            turn: 1,
            reason: "complete".to_string(),
            nodes_visited: 1,
            work_units: 1,
            proposal_sampled: false,
            candidate_count: 1,
            top_candidates: vec![FiniteSearchCandidateEvidence {
                word: "cigar".to_string(),
                quality: "exact".to_string(),
                modeled_failure_probability: Some(0.0),
                expected_attempts_remaining: Some(0.0),
            }],
        };
        checkpoint.games[0].finite_search_steps = vec![trace.clone()];
        validate_rolling_checkpoint(
            &checkpoint,
            "source",
            "test",
            &checkpoint.config_toml,
            &plan,
            true,
        )
        .expect("aligned finite trace");

        checkpoint.games[0].finite_search_steps.clear();
        let missing = validate_rolling_checkpoint(
            &checkpoint,
            "source",
            "test",
            &checkpoint.config_toml,
            &plan,
            true,
        )
        .expect_err("missing finite trace");
        assert!(missing.to_string().contains("finite search trace"));

        checkpoint.games[0].finite_search_steps = vec![trace.clone()];
        checkpoint.games[0].finite_search_steps[0].turn = 2;
        let misaligned = validate_rolling_checkpoint(
            &checkpoint,
            "source",
            "test",
            &checkpoint.config_toml,
            &plan,
            true,
        )
        .expect_err("misaligned finite trace");
        assert!(misaligned.to_string().contains("finite search trace"));

        checkpoint.games[0].finite_search_steps = vec![trace];
        checkpoint.games[0].finite_search_steps[0].top_candidates[0].word = "rebut".to_string();
        let wrong_guess = validate_rolling_checkpoint(
            &checkpoint,
            "source",
            "test",
            &checkpoint.config_toml,
            &plan,
            true,
        )
        .expect_err("trace guess does not match path");
        assert!(wrong_guess.to_string().contains("finite search trace"));
    }

    #[test]
    fn rolling_checkpoint_fresh_partial_resume_preserves_results() {
        let (partial, plan, _) = rolling_checkpoint_test_fixture();
        let mut fresh = partial.clone();
        fresh.folds.clear();
        fresh.games.clear();
        validate_test_rolling_checkpoint(&fresh, &plan).expect("fresh checkpoint");
        validate_test_rolling_checkpoint(&partial, &plan).expect("partial checkpoint");

        let encoded = serde_json::to_vec(&partial).expect("serialize");
        let resumed: RollingEvaluationCheckpoint =
            serde_json::from_slice(&encoded).expect("deserialize");
        validate_test_rolling_checkpoint(&resumed, &plan).expect("resumed checkpoint");
        assert_eq!(serde_json::to_vec(&resumed).expect("reserialize"), encoded);
    }

    #[test]
    fn rolling_checkpoint_rejects_metric_arithmetic_mismatch() {
        let (mut checkpoint, plan, _) = rolling_checkpoint_test_fixture();
        checkpoint.folds[0].metrics.solved_games = 0;
        let error =
            validate_test_rolling_checkpoint(&checkpoint, &plan).expect_err("metric mismatch");
        assert!(error.to_string().contains("metrics do not match"));
    }

    #[test]
    fn development_identity_ignores_history_after_the_declared_cutoff() {
        let fixture = crate::test_support::TestDirectory::new("development-identity");
        let root = fixture.path().to_path_buf();
        let _ = fs::remove_dir_all(&root);
        for directory in ["src", "tests", "config", "data/raw", "data/seed"] {
            fs::create_dir_all(root.join(directory)).expect("directory");
        }
        fs::write(root.join("Cargo.toml"), "[package]\nname='identity-test'\n").expect("manifest");
        fs::write(root.join("Cargo.lock"), "").expect("lock");
        fs::write(
            root.join("config/evaluation.toml"),
            include_str!("../../config/evaluation.toml"),
        )
        .expect("policy");
        for file in [
            "valid_guesses.txt",
            "candidate_answers.txt",
            "reference_candidate_answers.txt",
            "manual_additions.txt",
        ] {
            fs::write(root.join("data/seed").join(file), "cigar\n").expect("seed");
        }
        let cutoff = NaiveDate::from_ymd_opt(2026, 8, 26).expect("date");
        let entry = NytDailyEntry {
            id: Some(1),
            solution: "cigar".into(),
            print_date: cutoff,
            days_since_launch: None,
            editor: None,
        };
        let history_path = root.join("data/raw/nyt_daily_answers.jsonl");
        fs::write(
            &history_path,
            format!("{}\n", serde_json::to_string(&entry).expect("entry")),
        )
        .expect("history");
        let paths = ProjectPaths::new(&root);
        let before = development_source_identity(&paths, cutoff).expect("identity");
        let later = NytDailyEntry {
            id: Some(2),
            solution: "rebut".into(),
            print_date: NaiveDate::from_ymd_opt(2026, 8, 27).expect("date"),
            days_since_launch: None,
            editor: None,
        };
        fs::write(
            &history_path,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&entry).expect("entry"),
                serde_json::to_string(&later).expect("later")
            ),
        )
        .expect("extended history");
        let after = development_source_identity(&paths, cutoff).expect("identity");
        assert_eq!(before, after);
        for relative in [
            "src/lib.rs",
            "tests/replay.rs",
            "Cargo.toml",
            "Cargo.lock",
            "config/evaluation.toml",
            "data/seed/candidate_answers.txt",
        ] {
            let path = root.join(relative);
            let original = path
                .is_file()
                .then(|| fs::read(&path).expect("original input"));
            fs::write(&path, b"changed identity input\n").expect("mutate owned fixture input");
            assert_ne!(
                before,
                development_source_identity(&paths, cutoff).expect("changed identity"),
                "identity ignored {relative}"
            );
            if let Some(bytes) = original {
                fs::write(&path, bytes).expect("restore fixture input");
            } else {
                fs::remove_file(&path).expect("remove added fixture input");
            }
            assert_eq!(
                before,
                development_source_identity(&paths, cutoff).expect("restored identity")
            );
        }
        let _ = fs::remove_dir_all(root);
    }
}
