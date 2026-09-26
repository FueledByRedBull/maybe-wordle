use super::search::check_predictive_search_cancelled;
use super::*;

const EVIDENCE_CHECKPOINT_SCHEMA_VERSION: u32 = 3;
const ROLLING_CHECKPOINT_SCHEMA_VERSION: u32 = 2;
pub(super) const BENCHMARK_EVIDENCE_SCHEMA_VERSION: u32 = 7;
const FINITE_SEARCH_REGRET_SCHEMA_VERSION: u32 = 1;
const FINITE_REGRET_VALUE_RESOLUTION: f64 = 64.0 * f64::EPSILON;

const POSTERIOR_CALIBRATION_STRATA: [&str; 5] = [
    "all",
    "never_used",
    "reused",
    "historical_only",
    "out_of_core",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvidenceMatrixCheckpoint {
    schema_version: u32,
    identity: String,
    elapsed_ms: u64,
    peak_working_set_bytes: u64,
    baselines: Vec<EvidenceBaseline>,
}

impl EvidenceMatrixCheckpoint {
    fn validate(&self, identity: &str, profile_ids: &[String]) -> Result<()> {
        if self.schema_version != EVIDENCE_CHECKPOINT_SCHEMA_VERSION {
            bail!("unsupported evidence checkpoint schema");
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
    let expected_failures = canonical
        .unsolved_games
        .checked_add(canonical.coverage_gaps)
        .ok_or_else(|| anyhow!("evidence baseline {} failure count overflowed", baseline.id))?;
    if summary.games != canonical.scheduled_games
        || summary.p95_guesses != canonical.p95_guesses
        || summary.max_guesses != canonical.max_guesses
        || summary.failures != expected_failures
        || summary.coverage_gaps != canonical.coverage_gaps
        || summary.average_guesses.to_bits() != canonical.conditional_mean_guesses.to_bits()
        || summary.average_guesses_ci95.0.to_bits()
            != canonical.conditional_mean_guesses_ci95.lower.to_bits()
        || summary.average_guesses_ci95.1.to_bits()
            != canonical.conditional_mean_guesses_ci95.upper.to_bits()
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

fn needs_serial_study_latency(
    status: TrialStatus,
    completed_folds: usize,
    maximum_folds: usize,
    has_latency: bool,
) -> bool {
    status == TrialStatus::Complete && completed_folds >= maximum_folds && !has_latency
}

#[derive(Clone, Debug)]
struct SearchRegretCandidateState {
    date: NaiveDate,
    target: String,
    turn: usize,
    observations: Vec<(String, u8)>,
    surviving_answers: usize,
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

fn learned_proxy_feature_names() -> Vec<String> {
    [
        "entropy",
        "solve_probability",
        "expected_remaining",
        "force_in_two",
        "worst_non_green_bucket_size",
        "largest_non_green_bucket_mass",
        "high_mass_ambiguous_bucket_count",
        "smoothness_penalty",
        "large_non_green_bucket_count",
        "dangerous_mass_bucket_count",
        "non_green_mass_in_large_buckets",
        "posterior_answer_probability",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

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
        mean_regret: if regrets.is_empty() {
            0.0
        } else {
            regrets.iter().sum::<f64>() / regrets.len() as f64
        },
        maximum_regret: regrets.into_iter().fold(0.0, f64::max),
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
            let batch = match policy.search_mode {
                Some(mode) => self.suggestion_batch_internal_with_search_mode(
                    &state,
                    top.max(1),
                    Some(PredictiveContext {
                        hard_mode: false,
                        as_of,
                        observations: &observations,
                    }),
                    policy.book_usage,
                    Some(mode),
                )?,
                None => self.suggestion_batch_for_history(
                    as_of,
                    &observations,
                    &state,
                    top.max(1),
                    policy.book_usage,
                )?,
            };
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
        let replay_identity = ReplayIdentityInput {
            format_version: crate::experiments::exhaustive_cost::REPLAY_IDENTITY_FORMAT_VERSION,
            algorithm_version: "native-weighted-exact-v1".to_string(),
            solver_identity: input_fingerprint.clone(),
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
        let mut selected_states = Vec::new();
        for (row_split, range) in ranges {
            let (candidates, _) = self.collect_learned_proxy_states(
                range,
                request.minimum_survivors,
                request.maximum_survivors,
                request.maximum_states_per_split.saturating_mul(6).max(12),
                started,
                budget,
                false,
                false,
            )?;
            if candidates.is_empty() {
                bail!(
                    "learned-proxy split {:?} has no reachable states in {} through {}",
                    row_split,
                    range.start,
                    range.end
                );
            }
            for index in evenly_spaced_indices(candidates.len(), request.maximum_states_per_split) {
                selected_states.push((row_split, candidates[index].clone()));
            }
        }

        let feature_names = learned_proxy_feature_names();
        let mut rows = Vec::new();
        let mut completed_state_ids = BTreeSet::new();
        let mut prior_elapsed_ms = 0_u64;
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
            completed_state_ids.extend(checkpoint.completed_state_ids);
            rows = checkpoint.rows;
            eprintln!(
                "learned-proxy phase=resume states={} rows={} prior_elapsed_s={:.1} checkpoint={}",
                completed_state_ids.len(),
                rows.len(),
                prior_elapsed_ms as f64 / 1_000.0,
                checkpoint_path.display()
            );
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
        let exact_started = Instant::now();
        for (row_split, candidate) in selected_states {
            let state_id = format!(
                "{}-turn-{}-{}",
                candidate.date, candidate.turn, candidate.target
            );
            if completed_state_ids.contains(&state_id) {
                continue;
            }
            let cumulative_elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            if cumulative_elapsed_ms > request.maximum_seconds.saturating_mul(1_000) {
                bail!(
                    "learned-proxy dataset exceeded its {} second budget",
                    request.maximum_seconds
                );
            }
            if let Some(snapshot) = crate::process_memory::process_memory_snapshot()
                && snapshot.peak_working_set_bytes > maximum_memory_bytes
            {
                bail!(
                    "learned-proxy dataset exceeded its {} MiB memory budget",
                    request.maximum_memory_mb
                );
            }
            let as_of = candidate
                .date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
            let state = self.apply_history(as_of, &candidate.observations)?;
            if state.surviving.len() != candidate.surviving_answers {
                bail!("learned-proxy state reconstruction changed survivor count");
            }
            let mut metrics = self.score_guess_metrics_for_subset(&state.surviving, &state.weights);
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
            let mut memo = PredictiveMemoMap::default();
            let mut scratch = ExactSearchScratch::new();
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
            for guess_index in guess_indexes {
                let metric = metric_by_guess[&guess_index];
                let exact_cost = self.exact_cost_for_guess(
                    guess_index,
                    ExactCostContext {
                        subset: &state.surviving,
                        weights: &state.weights,
                        memo: &mut memo,
                        best_bound: f64::INFINITY,
                        scratch: &mut scratch,
                        depth: 0,
                    },
                )?;
                if exact_cost.is_finite() {
                    let mut row = ExhaustiveCostRow {
                        state: exact_state.clone(),
                        guess: self.guesses[guess_index].clone(),
                        exact_continuation_cost: exact_cost,
                        feature_values: learned_proxy_features(&metric),
                        baseline_proxy_cost: Some(metric.proxy_cost),
                        split: row_split,
                    };
                    row.canonicalize_numeric_values();
                    rows.push(row);
                }
            }
            completed_state_ids.insert(state_id.clone());
            rows.sort_by_key(ExhaustiveCostRow::key);
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
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
                        peak_memory_bytes: crate::process_memory::process_memory_snapshot()
                            .map(|snapshot| snapshot.peak_working_set_bytes),
                        last_state_id: completed_state_ids.iter().next_back().cloned(),
                        complete: false,
                        stop_reason: None,
                    },
                    completed_state_ids: completed_state_ids.iter().cloned().collect(),
                    rows: rows.clone(),
                };
                checkpoint.validate(&split)?;
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
        let elapsed_ms = prior_elapsed_ms
            .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        let artifact = ExhaustiveCostDatasetArtifact {
            format_version: crate::experiments::exhaustive_cost::EXHAUSTIVE_COST_FORMAT_VERSION,
            provenance: DatasetProvenance {
                dataset_id: crate::identity::digest_bytes_tagged(
                    "maybe-wordle-learned-proxy-dataset-v1",
                    format!("{}:{}", input_fingerprint, rows.len()).as_bytes(),
                ),
                generator_version: "native-weighted-exact-v1".to_string(),
                source_identity: input_fingerprint.clone(),
                source_data_fingerprint: input_fingerprint.clone(),
                config_fingerprint,
                executable_fingerprint: Some(executable_fingerprint),
                cutoff_start: plan.history.start,
                cutoff_end: plan.development.end,
                replay_identity,
            },
            split,
            budget: resource_budget,
            progress: ExhaustiveProgress {
                phase: "complete".to_string(),
                states_evaluated: total_states,
                rows_emitted: rows.len(),
                elapsed_ms,
                peak_memory_bytes: crate::process_memory::process_memory_snapshot()
                    .map(|snapshot| snapshot.peak_working_set_bytes),
                last_state_id: rows.last().map(|row| row.state.state_id.clone()),
                complete: true,
                stop_reason: None,
            },
            rows,
            checkpoint: None,
        };
        artifact.validate()?;
        if let Some(checkpoint_path) = &request.checkpoint_path {
            let checkpoint = ExhaustiveCostCheckpoint {
                format_version: crate::experiments::exhaustive_cost::EXHAUSTIVE_COST_FORMAT_VERSION,
                replay_identity_digest,
                budget: resource_budget,
                progress: artifact.progress.clone(),
                completed_state_ids: completed_state_ids.into_iter().collect(),
                rows: artifact.rows.clone(),
            };
            checkpoint.validate(&artifact.split)?;
            crate::atomic_file::atomic_write(
                checkpoint_path,
                &serde_json::to_vec_pretty(&checkpoint)?,
            )?;
        }
        ensure_development_source_identity(paths, plan.development.end, &input_fingerprint)?;
        eprintln!(
            "learned-proxy phase=complete features={} rows={} elapsed_s={:.1}",
            feature_names.len(),
            artifact.rows.len(),
            elapsed_ms as f64 / 1_000.0
        );
        Ok(artifact)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "shared legacy/finite collector keeps sampling rules and deadline behavior explicit"
    )]
    fn collect_learned_proxy_states(
        &self,
        range: DateRange,
        minimum_survivors: usize,
        maximum_survivors: usize,
        maximum_games: usize,
        started: Instant,
        budget: std::time::Duration,
        hard_mode: bool,
        allow_partial_on_deadline: bool,
    ) -> Result<(Vec<SearchRegretCandidateState>, usize)> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| range.contains(entry.print_date))
            .collect::<Vec<_>>();
        let indices = evenly_spaced_indices(games.len(), maximum_games);
        let total = indices.len();
        let mut candidates = Vec::new();
        let mut scanned_games = 0usize;
        for (game_number, index) in indices.into_iter().enumerate() {
            if started.elapsed() > budget {
                if allow_partial_on_deadline {
                    break;
                }
                bail!("learned-proxy collection exceeded its wall-clock budget");
            }
            if allow_partial_on_deadline && started.elapsed() >= budget {
                break;
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
                let mut batch = self.suggestion_batch_internal_with_search_mode(
                    &state,
                    if hard_mode { self.guesses.len() } else { 1 },
                    Some(PredictiveContext {
                        hard_mode,
                        as_of,
                        observations: &observations,
                    }),
                    PredictiveBookUsage::None,
                    Some(PredictiveSearchMode::ProxyOnly),
                )?;
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
        for (game_index, entry) in games.into_iter().enumerate() {
            if started.elapsed() > budget {
                bail!(
                    "search-regret exceeded its {} second budget while collecting reachable states",
                    maximum_seconds
                );
            }
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
                    let chosen = self
                        .suggestion_batch_internal_with_search_mode(
                            &state,
                            1,
                            Some(PredictiveContext {
                                hard_mode: false,
                                as_of,
                                observations: &observations,
                            }),
                            PredictiveBookUsage::None,
                            Some(PredictiveSearchMode::ProxyOnly),
                        )?
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
        if candidates.is_empty() {
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
            if started.elapsed() > budget {
                bail!(
                    "search-regret exceeded its {} second budget before completing all sampled states",
                    maximum_seconds
                );
            }
            let candidate = &candidates[candidate_index];
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
                .suggestion_batch_internal_with_search_mode(
                    &state,
                    1,
                    context,
                    PredictiveBookUsage::None,
                    Some(PredictiveSearchMode::ProxyOnly),
                )?
                .suggestions
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("proxy audit returned no suggestion"))?
                .word;
            let lookahead_guess = self
                .suggestion_batch_internal_with_search_mode(
                    &state,
                    1,
                    context,
                    PredictiveBookUsage::None,
                    Some(PredictiveSearchMode::Lookahead),
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
                let production = self.suggestion_batch_for_history(
                    as_of,
                    &candidate.observations,
                    &state,
                    1,
                    PredictiveBookUsage::None,
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
            states.push(exhaustive_solver.audit_search_regret_state(
                candidate,
                &state,
                production_guess.as_deref(),
                production_regime,
                &proxy_guess,
                &lookahead_guess,
            )?);
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
        Ok(SearchRegretReport {
            schema_version: 1,
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
            scanned_games: total_games,
            available_states,
            sampled_states: states.len(),
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

    fn audit_search_regret_state(
        &self,
        candidate: &SearchRegretCandidateState,
        state: &SolveState,
        production_guess: Option<&str>,
        production_regime: PredictiveRegime,
        proxy_guess: &str,
        lookahead_guess: &str,
    ) -> Result<SearchRegretState> {
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
            let cost = self.exact_cost_for_guess(
                guess_index,
                ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    memo: &mut memo,
                    best_bound: optimal_cost,
                    scratch: &mut scratch,
                    depth: 0,
                },
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
            let cost = self.exact_cost_for_guess(
                guess_index,
                ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    memo: &mut memo,
                    best_bound: f64::INFINITY,
                    scratch: &mut scratch,
                    depth: 0,
                },
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

    pub(super) fn recovery_backtest_detailed_with_book_usage(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedBacktestReport> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .filter(|entry| {
                let Some(as_of) = entry.print_date.checked_sub_days(Days::new(1)) else {
                    return false;
                };
                let target = entry.solution.to_ascii_lowercase();
                let state = self.initial_state(as_of);
                !state
                    .surviving
                    .iter()
                    .any(|index| self.answers[*index].word == target)
                    && state
                        .fallback_surviving
                        .iter()
                        .any(|index| self.answers[*index].word == target)
            })
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no out-of-primary recovery games found in the requested range");
        }
        self.backtest_selected_games(&games, top, book_usage)
    }

    pub(super) fn backtest_selected_games(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedBacktestReport> {
        self.backtest_selected_games_with_progress(games, top, book_usage, None)
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
                average_guesses_ci95: (
                    canonical.conditional_mean_guesses_ci95.lower,
                    canonical.conditional_mean_guesses_ci95.upper,
                ),
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
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let total = games.len();
        let evaluate = |entry: &&NytDailyEntry| {
            let result = self.solve_backtest_entry(entry, top, book_usage);
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
                average_guesses_ci95: (
                    canonical.conditional_mean_guesses_ci95.lower,
                    canonical.conditional_mean_guesses_ci95.upper,
                ),
                failure_rate_ci95,
                canonical,
            },
            runs,
        })
    }

    pub(super) fn solve_backtest_entry(
        &self,
        entry: &NytDailyEntry,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<(GameOutcome, DetailedSolveRun)> {
        let as_of = entry
            .print_date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let run = self.solve_target_from_state_detailed(
            &entry.solution,
            as_of,
            entry.print_date,
            top,
            book_usage,
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
        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let detailed =
            self.backtest_selected_games_with_progress(games, top, book_usage, progress)?;
        let backtest = detailed.summary.clone();
        let (
            proxy_step_pct,
            lookahead_step_pct,
            escalated_exact_step_pct,
            exact_step_pct,
            finite_step_pct,
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

        let divisor = measured.max(1) as f64;
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
            let (cold, warm) = self.benchmark_session_fallback_latency(fallback_as_of)?;
            (Some(cold), Some(warm))
        } else {
            (None, None)
        };
        Ok(ExperimentResult {
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
            average_log_loss: total_log_loss / divisor,
            average_brier: total_brier / divisor,
            average_target_probability: total_target_probability / divisor,
            average_target_rank: total_rank / divisor,
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
            latency_p95_ms: self.benchmark_predictive_latency(
                evaluation_to,
                default_diagnostic_suite()?.latency.evidence_runs,
            )?,
            session_fallback_cold_ms,
            session_fallback_warm_ms,
            proxy_step_pct,
            lookahead_step_pct,
            escalated_exact_step_pct,
            exact_step_pct,
            finite_step_pct,
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
        })
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
            checkpoint.validate(&checkpoint_identity, &profile_ids)?;
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
            total_profiles,
            rayon::current_num_threads(),
            from,
            to,
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
            let games = selected_evidence_games(&solver.history_dates, &selected_ranges)?;
            let result = solver.experiment_report_for_selected_games_with_book_usage_and_progress(
                &games,
                top,
                book_usage,
                Some(&progress),
            )?;
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
            let memory = enforce_evidence_resource_budget(
                generation_started,
                prior_elapsed_ms,
                prior_peak_working_set_bytes,
                resource_budget,
            )?;
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
            )?;
            if current_identity != checkpoint_identity {
                bail!(
                    "evidence source, config, selection, or matrix changed during evaluation; discard the partial checkpoint and retry"
                );
            }
            if let Some(path) = checkpoint_path.as_deref() {
                let checkpoint = EvidenceMatrixCheckpoint {
                    schema_version: EVIDENCE_CHECKPOINT_SCHEMA_VERSION,
                    identity: checkpoint_identity.clone(),
                    elapsed_ms: cumulative_evidence_elapsed_ms(
                        generation_started,
                        prior_elapsed_ms,
                    ),
                    peak_working_set_bytes: prior_peak_working_set_bytes,
                    baselines: baselines.clone(),
                };
                checkpoint.validate(&checkpoint_identity, &profile_ids)?;
                crate::atomic_file::atomic_write(path, &serde_json::to_vec_pretty(&checkpoint)?)?;
            }
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

    pub fn render_development_evidence_markdown(
        artifact: &PredictiveEvidenceArtifact,
    ) -> Result<String> {
        artifact.validate_identity()?;
        let mut output = String::new();
        output.push_str("<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->\n");
        output.push_str("## Predictive solver evidence\n\n");
        let selected_ranges = artifact
            .selected_ranges
            .iter()
            .map(|range| format!("{}..{}", range.start, range.end))
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&format!(
            "Development-only diagnostic for `{}` through `{}` using selection `{}` ({}) and history through `{}`. The sealed test was **not** evaluated.\n\n",
            artifact.evaluation_from,
            artifact.evaluation_to,
            artifact.evaluation_selection,
            selected_ranges,
            artifact.history_snapshot_end
        ));
        if let Some(peak_bytes) = artifact.resources.peak_working_set_bytes {
            output.push_str(&format!(
                "Measured generation compute time: {:.2} s; process peak working set: {:.1} MiB; enforced budget: {} s / {} MiB.\n\n",
                artifact.resources.generation_compute_ms as f64 / 1_000.0,
                peak_bytes as f64 / (1024.0 * 1024.0),
                artifact.resource_budget.maximum_seconds,
                artifact.resource_budget.maximum_memory_mb
            ));
        }
        output.push_str("| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for baseline in &artifact.baselines {
            let metrics = &baseline.result.backtest.canonical;
            let paired = baseline
                .paired_vs_selected_default
                .expect("generated evidence always has paired comparisons");
            output.push_str(&format!(
                "| `{}` | {:.1}% ({}/{}) | {:.1}% ({}/{}) | {:.4} [{:.4}, {:.4}] | {:.4} [{:.4}, {:.4}] | {:.1}% | {:.1}% | {:+.4} [{:+.4}, {:+.4}] | {}/{}/{} | {:.4} | {:.4} | {:.2} ms | {}/{} |\n",
                baseline.id,
                metrics.coverage_rate * 100.0,
                metrics.modeled_games,
                metrics.scheduled_games,
                metrics.solve_rate * 100.0,
                metrics.solved_games,
                metrics.scheduled_games,
                metrics.all_game_penalized_mean_guesses,
                metrics.all_game_penalized_mean_guesses_ci95.lower,
                metrics.all_game_penalized_mean_guesses_ci95.upper,
                metrics.conditional_mean_guesses,
                metrics.conditional_mean_guesses_ci95.lower,
                metrics.conditional_mean_guesses_ci95.upper,
                metrics.solved_in_guess_counts[..3].iter().sum::<usize>() as f64
                    / metrics.scheduled_games.max(1) as f64
                    * 100.0,
                metrics.solved_in_guess_counts[..4].iter().sum::<usize>() as f64
                    / metrics.scheduled_games.max(1) as f64
                    * 100.0,
                paired.candidate_minus_baseline,
                paired.ci95.lower,
                paired.ci95.upper,
                paired.candidate_wins,
                paired.ties,
                paired.baseline_wins,
                baseline.result.average_log_loss,
                baseline.result.average_brier,
                baseline.result.latency_p95_ms,
                baseline.result.session_fallback_cold_ms.map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
                baseline.result.session_fallback_warm_ms.map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
            ));
        }
        output.push_str("\nSession-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.\n");
        if !artifact.resources.artifact_sizes.is_empty() {
            output.push_str("\nMeasured artifact sizes: ");
            output.push_str(
                &artifact
                    .resources
                    .artifact_sizes
                    .iter()
                    .map(|artifact| format!("`{}` = {} bytes", artifact.name, artifact.bytes))
                    .collect::<Vec<_>>()
                    .join("; "),
            );
            output.push_str(".\n");
        }
        output.push_str("\n| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps | Artifact/session hits |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for baseline in &artifact.baselines {
            let prior = baseline.result.prior_evidence.as_ref();
            let telemetry = &baseline.result.execution;
            let recall = |value: Option<f64>| {
                value.map_or_else(
                    || "n/a".to_string(),
                    |value| format!("{:.1}%", value * 100.0),
                )
            };
            let ece = prior.map_or_else(
                || "n/a".to_string(),
                |metrics| {
                    format!(
                        "{:.4} [{:.4}, {:.4}]",
                        metrics.expected_calibration_error,
                        metrics.expected_calibration_error_ci95.lower,
                        metrics.expected_calibration_error_ci95.upper
                    )
                },
            );
            output.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {}/{}/{}/{}/{} | {}/{} | {}/{} |\n",
                baseline.id,
                recall(prior.map(|metrics| metrics.top_1_recall)),
                recall(prior.map(|metrics| metrics.top_3_recall)),
                recall(prior.map(|metrics| metrics.top_5_recall)),
                ece,
                telemetry.proxy_steps,
                telemetry.lookahead_steps,
                telemetry.escalated_exact_steps,
                telemetry.exact_steps,
                telemetry.finite_steps,
                telemetry.strict_recovery_steps
                    + telemetry.uniform_recovery_steps
                    + telemetry.epsilon_repair_steps,
                telemetry.dormant_fallback_steps,
                telemetry.exact_date_opener_artifact_hits
                    + telemetry.recent_opener_artifact_hits
                    + telemetry.reply_book_hits,
                telemetry.session_fallback_hits,
            ));
        }
        if artifact
            .baselines
            .iter()
            .any(|baseline| !baseline.result.posterior_calibration.is_empty())
        {
            output.push_str(
                "\nPost-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):\n\n",
            );
            output.push_str(
                "| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |\n",
            );
            output.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
            for baseline in &artifact.baselines {
                for summary in &baseline.result.posterior_calibration {
                    if summary.total_states == 0 {
                        continue;
                    }
                    let (target_probability, log_loss, brier) = summary.mean_score.map_or(
                        ("n/a".to_string(), "n/a".to_string(), "n/a".to_string()),
                        |score| {
                            (
                                format!("{:.4}", score.target_probability),
                                format!("{:.4}", score.log_loss),
                                format!("{:.4}", score.brier),
                            )
                        },
                    );
                    output.push_str(&format!(
                        "| `{}` | {} | {} | {}/{} | {} | {} | {} |\n",
                        baseline.id,
                        summary.stratum,
                        summary.turn,
                        summary.scored_states,
                        summary.total_states,
                        target_probability,
                        log_loss,
                        brier,
                    ));
                }
            }
        }
        if let Some(reference) = artifact
            .baselines
            .iter()
            .find(|baseline| baseline.id == artifact.reference_profile_id)
        {
            output.push_str(&format!(
                "\nReference `{}` all-game mean sensitivity: ",
                artifact.reference_profile_id
            ));
            for (index, metric) in reference
                .result
                .failure_penalty_sensitivity
                .iter()
                .enumerate()
            {
                if index > 0 {
                    output.push_str("; ");
                }
                output.push_str(&format!(
                    "penalty {:.0} = {:.4} [{:.4}, {:.4}]",
                    metric.penalty_guesses,
                    metric.all_game_mean_guesses,
                    metric.ci95.lower,
                    metric.ci95.upper
                ));
            }
            output.push_str(".\n");
        }
        output.push_str("\nThe old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.\n\n");
        output.push_str("The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.\n");
        output.push_str("<!-- END GENERATED PREDICTIVE EVIDENCE -->\n");
        Ok(output)
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
            schema_version: 4,
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

    pub fn freeze_predictive_candidate(
        paths: &ProjectPaths,
        config_path: &Path,
        comparison_path: &Path,
    ) -> Result<FrozenPredictiveCandidate> {
        let config = PriorConfig::load(config_path)?;
        let config_toml =
            toml::to_string_pretty(&config).context("serialize frozen candidate config")?;
        let comparison_bytes = fs::read(comparison_path)
            .with_context(|| format!("failed to read {}", comparison_path.display()))?;
        let comparison: RollingComparisonArtifact = serde_json::from_slice(&comparison_bytes)
            .with_context(|| format!("failed to parse {}", comparison_path.display()))?;
        validate_rolling_comparison_artifact(&comparison)?;
        let evaluation_plan =
            canonical_development_evaluation_plan(paths, "freezing predictive candidate")?;
        let input_fingerprint =
            development_source_identity(paths, evaluation_plan.development.end)?;
        if comparison.input_fingerprint != input_fingerprint
            || comparison.evaluation_plan != evaluation_plan
            || comparison.sealed_test_evaluated
        {
            bail!("development comparison is stale, uses another plan, or touched the sealed test");
        }
        if comparison.candidate.config_toml != config_toml {
            bail!("frozen config does not match the comparison candidate");
        }
        let candidate_failures = comparison.candidate.aggregate.unsolved_games
            + comparison.candidate.aggregate.coverage_gaps;
        if comparison.candidate.aggregate.scheduled_games == 0
            || comparison.candidate.aggregate.modeled_games
                != comparison.candidate.aggregate.scheduled_games
            || candidate_failures != 0
        {
            bail!("candidate cannot be frozen without full coverage and zero failures");
        }
        if comparison.candidate_minus_baseline.candidate_minus_baseline >= 0.0
            || comparison.candidate_minus_baseline.ci95.upper >= 0.0
        {
            bail!(
                "candidate cannot be frozen unless its paired development interval is entirely below zero"
            );
        }
        let development_comparison_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-development-comparison-v1",
            &comparison_bytes,
        );
        let config_fingerprint = comparison.candidate.config_fingerprint.clone();
        let evaluation_artifact_policy = "artifact_free".to_string();
        let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-frozen-candidate-v1");
        hash.field(input_fingerprint.as_bytes())
            .field(config_fingerprint.as_bytes())
            .field(development_comparison_fingerprint.as_bytes())
            .field(comparison.candidate.label.as_bytes())
            .field(evaluation_artifact_policy.as_bytes())
            .field(
                &serde_json::to_vec(&evaluation_plan)
                    .context("serialize frozen evaluation plan identity")?,
            )
            .field(
                &serde_json::to_vec(&comparison.candidate.aggregate)
                    .context("serialize frozen development metrics identity")?,
            )
            .field(
                &serde_json::to_vec(&comparison.candidate_minus_baseline)
                    .context("serialize frozen paired difference identity")?,
            );
        let frozen = FrozenPredictiveCandidate {
            schema_version: 1,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            freeze_fingerprint: hash.finish_tagged(),
            evaluation_plan,
            config_toml,
            config_fingerprint,
            candidate_label: comparison.candidate.label,
            development_comparison_fingerprint,
            development_metrics: comparison.candidate.aggregate,
            development_paired_difference: comparison.candidate_minus_baseline,
            evaluation_artifact_policy,
            sealed_test_evaluated: false,
        };
        frozen.validate_identity()?;
        Ok(frozen)
    }

    pub fn evaluate_frozen_candidate_on_sealed_test(
        paths: &ProjectPaths,
        frozen: &FrozenPredictiveCandidate,
        output_path: &Path,
    ) -> Result<SealedTestReport> {
        frozen.validate_identity()?;
        if development_source_identity(paths, frozen.evaluation_plan.development.end)?
            != frozen.input_fingerprint
        {
            bail!(
                "source, executable, or data changed after candidate freeze; the sealed test remains closed"
            );
        }
        let plan = canonical_development_evaluation_plan(paths, "opening the sealed final test")?;
        if plan != frozen.evaluation_plan {
            bail!("evaluation plan changed after candidate freeze");
        }
        let output_path = if output_path.is_absolute() {
            output_path.to_path_buf()
        } else {
            paths.root.join(output_path)
        };
        let marker_path = paths
            .root
            .join("benchmarks/predictive/sealed-test-once.json");
        if output_path.exists() || marker_path.exists() {
            bail!(
                "sealed test has already been started or completed; marker={} output={}",
                marker_path.display(),
                output_path.display()
            );
        }
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        if let Some(parent) = marker_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let relative_output = output_path
            .strip_prefix(&paths.root)
            .unwrap_or(&output_path)
            .to_string_lossy()
            .replace('\\', "/");
        let mut marker = SealedTestMarker {
            schema_version: 1,
            freeze_fingerprint: frozen.freeze_fingerprint.clone(),
            output_path: relative_output,
            status: "started_irreversible".to_string(),
        };
        crate::atomic_file::atomic_write(
            &marker_path,
            &serde_json::to_vec_pretty(&marker).context("serialize sealed-test marker")?,
        )?;

        let config: PriorConfig =
            toml::from_str(&frozen.config_toml).context("parse frozen candidate config")?;
        let solver = Self::from_paths_with_settings(
            paths,
            &config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let report = solver.backtest_detailed_with_book_usage(
            plan.sealed_test.start,
            plan.sealed_test.end,
            5,
            PredictiveBookUsage::None,
        )?;
        let games = report
            .runs
            .iter()
            .map(|run| ExperimentGameResult {
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
            })
            .collect::<Vec<_>>();
        let outcomes = games.iter().map(|game| game.outcome).collect::<Vec<_>>();
        let metrics = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let prior_observations = solver
            .history_dates
            .iter()
            .filter(|entry| {
                entry.print_date >= plan.sealed_test.start
                    && entry.print_date <= plan.sealed_test.end
            })
            .filter_map(|entry| {
                solver
                    .initial_prior_metrics(&entry.solution, entry.print_date)
                    .map(|metrics| RankedProbabilityObservation {
                        target_rank: metrics.target_rank,
                        top_probability: metrics.top_probability,
                        top_prediction_correct: metrics.top_prediction_correct,
                    })
            })
            .collect::<Vec<_>>();
        let sealed = SealedTestReport {
            schema_version: 1,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            freeze_fingerprint: frozen.freeze_fingerprint.clone(),
            input_fingerprint: frozen.input_fingerprint.clone(),
            config_fingerprint: frozen.config_fingerprint.clone(),
            evaluation_plan: plan,
            evaluation_artifact_policy: frozen.evaluation_artifact_policy.clone(),
            sealed_test_evaluated: true,
            evaluated_once: true,
            metrics,
            prior_evidence: (!prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            execution: Self::execution_telemetry(&report.runs),
            games,
            latency_p95_ms: solver.benchmark_predictive_latency(
                frozen.evaluation_plan.development.end,
                default_diagnostic_suite()?.latency.evidence_runs,
            )?,
        };
        crate::atomic_file::atomic_write(
            &output_path,
            &serde_json::to_vec_pretty(&sealed).context("serialize sealed-test report")?,
        )?;
        marker.status = "completed".to_string();
        crate::atomic_file::atomic_write(
            &marker_path,
            &serde_json::to_vec_pretty(&marker).context("serialize sealed-test marker")?,
        )?;
        Ok(sealed)
    }

    pub fn render_rolling_comparison_markdown(
        comparisons: &[RollingComparisonArtifact],
    ) -> Result<String> {
        let first = comparisons
            .first()
            .ok_or_else(|| anyhow!("at least one rolling comparison is required"))?;
        for comparison in comparisons {
            validate_rolling_comparison_artifact(comparison)?;
        }
        if comparisons.iter().any(|comparison| {
            comparison.evaluation_plan != first.evaluation_plan
                || comparison.baseline.label != first.baseline.label
                || comparison.baseline.config_toml != first.baseline.config_toml
                || comparison.baseline.aggregate != first.baseline.aggregate
        }) {
            bail!("rolling comparisons must share one development plan and baseline");
        }
        let baseline = &first.baseline;
        let mut output = String::new();
        output.push_str("<!-- BEGIN GENERATED ROLLING EVIDENCE -->\n");
        output.push_str("### Rolling-origin promotion guard\n\n");
        output.push_str(&format!(
            "Across {} non-overlapping development folds ({} scheduled games), the sealed test was **not** evaluated. Coverage gaps and six-guess failures are hard constraints before mean score.\n\n",
            first.evaluation_plan.folds.len(),
            baseline.aggregate.scheduled_games
        ));
        output.push_str("| Configuration | Solved | All-game mean | Delta vs baseline | W/T/L | Latency p95 | Guard decision |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n");
        output.push_str(&format!(
            "| `{}` | {}/{} | {:.4} [{:.4}, {:.4}] | reference | -- | {:.2} ms | retained |\n",
            baseline.label,
            baseline.aggregate.solved_games,
            baseline.aggregate.scheduled_games,
            baseline.aggregate.all_game_penalized_mean_guesses,
            baseline
                .aggregate
                .all_game_penalized_mean_guesses_ci95
                .lower,
            baseline
                .aggregate
                .all_game_penalized_mean_guesses_ci95
                .upper,
            baseline.latency_p95_ms,
        ));
        for comparison in comparisons {
            let candidate = &comparison.candidate;
            let paired = comparison.candidate_minus_baseline;
            let baseline_failures =
                baseline.aggregate.unsolved_games + baseline.aggregate.coverage_gaps;
            let candidate_failures =
                candidate.aggregate.unsolved_games + candidate.aggregate.coverage_gaps;
            let decision = if candidate_failures > baseline_failures {
                "rejected: added failures"
            } else if paired.ci95.upper < 0.0 {
                "eligible on solve quality"
            } else if paired.candidate_minus_baseline < 0.0 {
                "not promoted: improvement uncertain"
            } else {
                "rejected: no solve-quality gain"
            };
            output.push_str(&format!(
                "| `{}` | {}/{} | {:.4} [{:.4}, {:.4}] | {:+.4} [{:+.4}, {:+.4}] | {}/{}/{} | {:.2} ms | {} |\n",
                candidate.label,
                candidate.aggregate.solved_games,
                candidate.aggregate.scheduled_games,
                candidate.aggregate.all_game_penalized_mean_guesses,
                candidate.aggregate.all_game_penalized_mean_guesses_ci95.lower,
                candidate.aggregate.all_game_penalized_mean_guesses_ci95.upper,
                paired.candidate_minus_baseline,
                paired.ci95.lower,
                paired.ci95.upper,
                paired.candidate_wins,
                paired.ties,
                paired.baseline_wins,
                candidate.latency_p95_ms,
                decision,
            ));
        }
        output.push_str("\n| Configuration | Prior top-1/3/5 | Confidence ECE | Search steps P/L/XE/X/F | Recovery/fallback steps |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: |\n");
        for evidence in std::iter::once(baseline)
            .chain(comparisons.iter().map(|comparison| &comparison.candidate))
        {
            let prior = evidence.prior_evidence.as_ref();
            let telemetry = &evidence.execution;
            output.push_str(&format!(
                "| `{}` | {} | {} | {}/{}/{}/{}/{} | {}/{} |\n",
                evidence.label,
                prior.map_or_else(
                    || "n/a".to_string(),
                    |metrics| format!(
                        "{:.1}%/{:.1}%/{:.1}%",
                        metrics.top_1_recall * 100.0,
                        metrics.top_3_recall * 100.0,
                        metrics.top_5_recall * 100.0
                    )
                ),
                prior.map_or_else(
                    || "n/a".to_string(),
                    |metrics| format!(
                        "{:.4} [{:.4}, {:.4}]",
                        metrics.expected_calibration_error,
                        metrics.expected_calibration_error_ci95.lower,
                        metrics.expected_calibration_error_ci95.upper
                    )
                ),
                telemetry.proxy_steps,
                telemetry.lookahead_steps,
                telemetry.escalated_exact_steps,
                telemetry.exact_steps,
                telemetry.finite_steps,
                telemetry.strict_recovery_steps
                    + telemetry.uniform_recovery_steps
                    + telemetry.epsilon_repair_steps,
                telemetry.dormant_fallback_steps,
            ));
        }
        output.push_str("\nDevelopment decisions:\n\n");
        for comparison in comparisons {
            let candidate = &comparison.candidate;
            let paired = comparison.candidate_minus_baseline;
            let baseline_failures =
                baseline.aggregate.unsolved_games + baseline.aggregate.coverage_gaps;
            let candidate_failures =
                candidate.aggregate.unsolved_games + candidate.aggregate.coverage_gaps;
            let explanation = if candidate_failures > baseline_failures {
                format!(
                    "rejected because it added {} failure(s)",
                    candidate_failures - baseline_failures
                )
            } else if paired.ci95.upper < 0.0 {
                "eligible on solve quality because the paired interval is entirely below zero"
                    .to_string()
            } else if paired.candidate_minus_baseline < 0.0 {
                "retained as a development finalist, not promoted, because the observed improvement's paired interval includes zero"
                    .to_string()
            } else {
                "rejected because it did not improve solve quality".to_string()
            };
            output.push_str(&format!("- `{}` is {}.\n", candidate.label, explanation));
        }
        output.push_str(
            "\nThis development comparison did not access the sealed window and does not establish prospective performance. Any later sealed evaluation requires separate evidence.\n",
        );
        output.push_str("<!-- END GENERATED ROLLING EVIDENCE -->\n");
        Ok(output)
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
        let offline = self.offline_book_solver()?;
        let (window_start, window_end, targets) =
            offline.recent_history_targets_for_books(as_of)?;
        let holdout = offline.previous_history_targets_for_books(window_start)?;
        let state = offline.initial_state(as_of);
        let candidates = offline
            .suggestion_batch_internal(
                &state,
                offline.config.session_opener_pool.max(1),
                Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &[],
                }),
                PredictiveBookUsage::None,
            )?
            .suggestions;
        let selected = offline
            .select_validated_opener(
                as_of,
                &candidates,
                &targets,
                holdout.as_ref().map(|(_, _, entries)| entries.as_slice()),
                default_diagnostic_suite()?.book_build.forced_suggestion_top,
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
        let forced_suggestion_top = default_diagnostic_suite()?.book_build.forced_suggestion_top;

        for answer_index in &root.surviving {
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
            let batch = offline.suggestion_batch_internal(
                &child,
                reply_candidate_limit,
                Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &observation,
                }),
                PredictiveBookUsage::None,
            )?;
            let mut best_reply: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
            for suggestion in batch.suggestions.into_iter().take(reply_candidate_limit) {
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
                    forced_suggestion_top,
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
                    let grand_batch = offline.suggestion_batch_internal(
                        &grandchild,
                        reply_candidate_limit,
                        Some(PredictiveContext {
                            hard_mode: false,
                            as_of,
                            observations: &grand_observations,
                        }),
                        PredictiveBookUsage::None,
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
        replies.sort_by_key(|reply| reply.feedback_pattern);
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
                    average_log_loss: fold_log_loss / fold_measured.max(1) as f64,
                    average_brier: fold_brier / fold_measured.max(1) as f64,
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
                average_log_loss: log_loss_sum / measured_games.max(1) as f64,
                average_brier: brier_sum / measured_games.max(1) as f64,
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
                if profile.average_log_loss >= baseline_log_loss {
                    profile.promotion_blockers.push(
                        "Log loss does not improve on the weighted reference prior.".to_string(),
                    );
                }
                if profile.average_brier >= baseline_brier {
                    profile.promotion_blockers.push(
                        "Brier score does not improve on the weighted reference prior.".to_string(),
                    );
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
            schema_version: 1,
            input_fingerprint,
            matrix_fingerprint,
            evaluation_plan,
            profiles,
            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        })
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
                let mut metrics =
                    self.score_guess_metrics_for_subset(&state.surviving, &state.weights);
                let known_absent_mask = known_absent_letter_mask(&observations);
                for metric in &mut metrics {
                    metric.known_absent_letter_hits =
                        count_masked_letters(&self.guesses[metric.guess_index], known_absent_mask);
                    metric.large_state_score = proxy_row_score_from_weights(
                        &self.config.proxy_weights,
                        ProxyRowStats::from_metric(metric),
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

    pub(super) fn regime_mix(runs: &[DetailedSolveRun]) -> (f64, f64, f64, f64, f64) {
        let mut proxy_steps = 0usize;
        let mut lookahead_steps = 0usize;
        let mut escalated_exact_steps = 0usize;
        let mut exact_steps = 0usize;
        let mut finite_steps = 0usize;
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
                }
            }
        }

        if total_steps == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let divisor = total_steps as f64;
        (
            proxy_steps as f64 / divisor,
            lookahead_steps as f64 / divisor,
            escalated_exact_steps as f64 / divisor,
            exact_steps as f64 / divisor,
            finite_steps as f64 / divisor,
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
                Some(PredictivePromotionSource::ReplyBook) => telemetry.reply_book_hits += 1,
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

    pub fn run_predictive_study(
        paths: &ProjectPaths,
        base_config: &PriorConfig,
        spec: StudySpec,
        state_path: &Path,
        top: usize,
        cancellation_path: Option<&Path>,
    ) -> Result<StudyRunSummary> {
        spec.validate()?;
        if base_config.search_policy_mode.is_finite() && spec.parallelism != 1 {
            bail!(
                "bounded-search studies require --jobs 1 so wall-clock search budgets do not contend across trials"
            );
        }
        if top == 0 {
            bail!("study top-suggestion count must be positive");
        }
        let evaluation_plan = canonical_development_evaluation_plan(paths, "running a study")?;
        if spec.maximum_validation_folds > evaluation_plan.folds.len() {
            bail!(
                "study requests {} validation folds but the development plan contains only {}",
                spec.maximum_validation_folds,
                evaluation_plan.folds.len()
            );
        }
        let registry = predictive_parameter_registry(base_config);
        let base_config_toml =
            toml::to_string_pretty(base_config).context("failed to serialize study base config")?;
        let registry_json = serde_json::to_vec(&registry)
            .context("failed to serialize study parameter registry")?;
        let registry_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-parameter-registry-v6",
            &registry_json,
        );
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        let compute_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let provenance = crate::experiments::StudyProvenance {
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            base_config_toml,
            registry_format_version: registry.format_version,
            registry_fingerprint,
            input_fingerprint: development_source_identity(paths, evaluation_plan.development.end)?,
            operating_system: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            compute_threads,
            code_revision,
            code_dirty,
            history_snapshot_start: evaluation_plan.history.start,
            history_snapshot_end: evaluation_plan.history.end,
            development_cutoff: evaluation_plan.development.end,
            top_suggestions: top,
        };
        if spec.strategy == StudySearchStrategy::ModelBased {
            return Self::run_model_based_study(
                paths,
                base_config,
                &registry,
                spec,
                evaluation_plan,
                provenance,
                state_path,
                top,
                cancellation_path,
            );
        }
        let candidates = generate_candidates(&registry, base_config, &spec)?;
        let mut state = if state_path.exists() {
            let state = StudyState::load(state_path)?;
            if state.spec != spec {
                bail!("existing study state does not match the requested study specification");
            }
            if state.evaluation_plan != evaluation_plan {
                bail!("existing study state uses a different evaluation plan");
            }
            if state.provenance != provenance {
                bail!(
                    "existing study state provenance differs from the current base config, registry, source/data snapshot, cutoff, or evaluation settings"
                );
            }
            state
        } else {
            StudyState::new(spec.clone(), evaluation_plan.clone(), provenance.clone())?
        };
        let mut normalized_checkpoint = false;
        for trial in &mut state.trials {
            let Some((generated, _)) = candidates.get(trial.candidate.number) else {
                bail!(
                    "checkpoint contains out-of-range trial {}",
                    trial.candidate.number
                );
            };
            if !trial.candidate.equivalent_to(generated) {
                bail!(
                    "checkpoint trial {} differs from deterministic generation",
                    trial.candidate.number
                );
            }
            let canonical_identity = generated.identity(&spec, &provenance)?;
            if trial.candidate != *generated || trial.identity != canonical_identity {
                trial.candidate = generated.clone();
                trial.identity = canonical_identity;
                normalized_checkpoint = true;
            }
        }
        if normalized_checkpoint {
            state.save(state_path)?;
        }

        let existing = state
            .trials
            .iter()
            .map(|trial| trial.identity.clone())
            .collect::<HashSet<_>>();
        for (candidate, _) in &candidates {
            let identity = candidate.identity(&spec, &provenance)?;
            if !existing.contains(identity.as_str()) {
                state.trials.push(StudyTrial {
                    candidate: candidate.clone(),
                    identity,
                    status: TrialStatus::Pending,
                    measurement: None,
                    reason: None,
                    elapsed_ms: Some(0),
                    pareto_rank: None,
                    hard_constraint_violations: Vec::new(),
                });
            }
        }
        state.trials.sort_by_key(|trial| trial.candidate.number);
        state.save(state_path)?;

        let effective_parallelism = spec.parallelism.min(candidates.len()).max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(compute_threads)
            .stack_size(crate::SOLVER_THREAD_STACK_BYTES)
            .build()
            .context("failed to create the study worker pool")?;
        let shared_state = Arc::new(Mutex::new(state));
        'fidelity: for target_folds in spec.fidelity_schedule() {
            ensure_development_source_identity(
                paths,
                provenance.development_cutoff,
                &provenance.input_fingerprint,
            )?;
            if cancellation_path.is_some_and(Path::exists) {
                break;
            }
            let validation_fold_indices =
                spec.fidelity_fold_indices(evaluation_plan.folds.len(), target_folds)?;
            let runnable = {
                let state = shared_state
                    .lock()
                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                candidates
                    .iter()
                    .filter_map(|(candidate, config)| {
                        let trial = state
                            .trials
                            .iter()
                            .find(|trial| trial.candidate.number == candidate.number)?;
                        let completed_folds = trial
                            .measurement
                            .as_ref()
                            .map_or(0, |measurement| measurement.validation_fold_indices.len());
                        (!matches!(
                            trial.status,
                            TrialStatus::Complete
                                | TrialStatus::Failed
                                | TrialStatus::Rejected
                                | TrialStatus::Pruned
                        ) && completed_folds < target_folds)
                            .then(|| (candidate.clone(), config.clone(), trial.identity.clone()))
                    })
                    .collect::<Vec<_>>()
            };
            for batch in runnable.chunks(effective_parallelism) {
                if cancellation_path.is_some_and(Path::exists) {
                    break 'fidelity;
                }
                let batch_results = pool.install(|| {
                    batch
                        .par_iter()
                        .map(|(_candidate, config, identity)| -> Result<()> {
                        let (initial_measurement, prior_elapsed_ms) = {
                            let mut state = shared_state
                                .lock()
                                .map_err(|_| anyhow!("study state lock poisoned"))?;
                            let trial = state
                                .trials
                                .iter_mut()
                                .find(|trial| trial.identity == *identity)
                                .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                            trial.status = TrialStatus::Running;
                            trial.reason = None;
                            let measurement = trial.measurement.clone().unwrap_or_default();
                            let elapsed_ms = trial.elapsed_ms.unwrap_or_default();
                            state.save(state_path)?;
                            (measurement, elapsed_ms)
                        };
                        let checkpoint_state = Arc::clone(&shared_state);
                        let result = Self::evaluate_study_candidate(
                            StudyEvaluationRequest {
                                paths,
                                config,
                                stage: spec.stage,
                                artifact_namespace: identity,
                                evaluation_plan: &evaluation_plan,
                                top,
                                target_validation_folds: target_folds,
                                validation_fold_indices: &validation_fold_indices,
                                maximum_trial_seconds: spec.maximum_trial_seconds,
                                maximum_memory_mb: spec.maximum_memory_mb,
                                measure_latency: false,
                                measurement: initial_measurement,
                                prior_elapsed_ms,
                                cancellation_path,
                            },
                            |measurement, elapsed_ms| {
                                let mut state = checkpoint_state
                                    .lock()
                                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                                let trial = state
                                    .trials
                                    .iter_mut()
                                    .find(|trial| trial.identity == *identity)
                                    .ok_or_else(|| {
                                        anyhow!("study trial disappeared during evaluation")
                                    })?;
                                trial.status = TrialStatus::Running;
                                trial.measurement = Some(measurement.clone());
                                trial.elapsed_ms = Some(elapsed_ms);
                                state.save(state_path)
                            },
                        );
                        let mut state = shared_state
                            .lock()
                            .map_err(|_| anyhow!("study state lock poisoned"))?;
                        let trial = state
                            .trials
                            .iter_mut()
                            .find(|trial| trial.identity == *identity)
                            .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                        match result {
                            Ok(Some(measurement)) => {
                                trial.status = if target_folds == spec.maximum_validation_folds {
                                    TrialStatus::Complete
                                } else {
                                    TrialStatus::Running
                                };
                                trial.measurement = Some(measurement);
                                trial.reason = (target_folds < spec.maximum_validation_folds)
                                    .then(|| {
                                        format!(
                                            "completed fidelity rung {target_folds}; awaiting promotion"
                                        )
                                    });
                            }
                            Ok(None) => {
                                trial.status = TrialStatus::Running;
                                trial.reason = Some(
                                    "paused by cooperative cancellation file; safe to resume"
                                        .to_string(),
                                );
                            }
                            Err(error) => {
                                trial.status = TrialStatus::Failed;
                                trial.reason = Some(format!("{error:#}"));
                            }
                        }
                        state.save(state_path)
                    })
                        .collect::<Vec<_>>()
                });
                for result in batch_results {
                    result?;
                }
                if cancellation_path.is_some_and(Path::exists) {
                    break 'fidelity;
                }
            }

            if target_folds == spec.maximum_validation_folds
                && !spec.stage.evaluates_prior_only()
                && !cancellation_path.is_some_and(Path::exists)
            {
                // Every parallel fold worker has joined above. Time finalists one at a time so
                // latency is comparable rather than a measurement of study-worker contention.
                for (candidate, config) in &candidates {
                    if cancellation_path.is_some_and(Path::exists) {
                        break;
                    }
                    let identity = candidate.identity(&spec, &provenance)?;
                    let latency_request = {
                        let mut state = shared_state
                            .lock()
                            .map_err(|_| anyhow!("study state lock poisoned"))?;
                        let trial = state
                            .trials
                            .iter_mut()
                            .find(|trial| trial.identity == identity)
                            .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                        let Some(measurement) = trial.measurement.clone() else {
                            continue;
                        };
                        if !needs_serial_study_latency(
                            trial.status,
                            measurement.validation_fold_indices.len(),
                            spec.maximum_validation_folds,
                            measurement.latency_p95_ms.is_some(),
                        ) {
                            continue;
                        }
                        trial.status = TrialStatus::Running;
                        trial.reason =
                            Some("awaiting serialized contention-free latency measurement".into());
                        let prior_elapsed_ms = trial.elapsed_ms.unwrap_or_default();
                        state.save(state_path)?;
                        Some((measurement, prior_elapsed_ms))
                    };
                    let Some((measurement, prior_elapsed_ms)) = latency_request else {
                        continue;
                    };
                    ensure_development_source_identity(
                        paths,
                        provenance.development_cutoff,
                        &provenance.input_fingerprint,
                    )?;
                    let result = Self::measure_study_candidate_latency(
                        paths,
                        config,
                        measurement,
                        prior_elapsed_ms,
                        spec.maximum_trial_seconds,
                        spec.maximum_memory_mb,
                    );
                    let mut state = shared_state
                        .lock()
                        .map_err(|_| anyhow!("study state lock poisoned"))?;
                    let trial = state
                        .trials
                        .iter_mut()
                        .find(|trial| trial.identity == identity)
                        .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                    match result {
                        Ok((measurement, elapsed_ms)) => {
                            trial.status = TrialStatus::Complete;
                            trial.measurement = Some(measurement);
                            trial.elapsed_ms = Some(elapsed_ms);
                            trial.reason = None;
                        }
                        Err(error) => {
                            trial.status = TrialStatus::Failed;
                            trial.reason = Some(format!("{error:#}"));
                        }
                    }
                    state.save(state_path)?;
                }
            }

            {
                let mut state = shared_state
                    .lock()
                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                crate::experiments::annotate_trial_outcomes(&mut state.trials, target_folds);
                state.save(state_path)?;
            }

            if target_folds < spec.maximum_validation_folds {
                let mut state = shared_state
                    .lock()
                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                let survivors = crate::experiments::successive_halving_survivors(
                    &state.trials,
                    target_folds,
                    spec.reduction_factor,
                );
                for trial in &mut state.trials {
                    let completed_folds = trial
                        .measurement
                        .as_ref()
                        .map_or(0, |measurement| measurement.validation_fold_indices.len());
                    if completed_folds >= target_folds
                        && !matches!(
                            trial.status,
                            TrialStatus::Complete
                                | TrialStatus::Failed
                                | TrialStatus::Rejected
                                | TrialStatus::Pruned
                        )
                    {
                        if survivors.contains(&trial.candidate.number) {
                            trial.status = TrialStatus::Running;
                            trial.reason = Some(format!(
                                "promoted after {target_folds} folds by reduction factor {}",
                                spec.reduction_factor
                            ));
                        } else {
                            trial.status = TrialStatus::Pruned;
                            trial.reason = Some(format!(
                                "pruned after {target_folds} folds by reduction factor {}",
                                spec.reduction_factor
                            ));
                        }
                    }
                }
                state.save(state_path)?;
            }
        }

        let state = Arc::try_unwrap(shared_state)
            .map_err(|_| anyhow!("study state still has active worker references"))?
            .into_inner()
            .map_err(|_| anyhow!("study state lock poisoned"))?;
        ensure_development_source_identity(
            paths,
            provenance.development_cutoff,
            &provenance.input_fingerprint,
        )?;

        Self::summarize_study_run(
            state,
            &registry,
            base_config,
            state_path,
            spec.parallelism,
            effective_parallelism,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_model_based_study(
        paths: &ProjectPaths,
        base_config: &PriorConfig,
        registry: &ParameterRegistry,
        spec: StudySpec,
        evaluation_plan: EvaluationPlan,
        provenance: StudyProvenance,
        state_path: &Path,
        top: usize,
        cancellation_path: Option<&Path>,
    ) -> Result<StudyRunSummary> {
        let mut state = if state_path.exists() {
            let state = StudyState::load(state_path)?;
            if state.spec != spec {
                bail!("existing study state does not match the requested study specification");
            }
            if state.evaluation_plan != evaluation_plan {
                bail!("existing study state uses a different evaluation plan");
            }
            if state.provenance != provenance {
                bail!(
                    "existing study state provenance differs from the current base config, registry, source/data snapshot, cutoff, or evaluation settings"
                );
            }
            state
        } else {
            StudyState::new(spec.clone(), evaluation_plan.clone(), provenance.clone())?
        };
        let mut numbers = state
            .trials
            .iter()
            .map(|trial| trial.candidate.number)
            .collect::<Vec<_>>();
        numbers.sort_unstable();
        if numbers.iter().copied().ne(0..numbers.len()) || state.trials.len() > spec.trial_count {
            bail!("model-based checkpoint candidates are not a valid contiguous prefix");
        }
        for trial in &state.trials {
            registry.apply_tunable_values(base_config, &trial.candidate.parameters)?;
        }
        state.save(state_path)?;

        loop {
            ensure_development_source_identity(
                paths,
                provenance.development_cutoff,
                &provenance.input_fingerprint,
            )?;
            if cancellation_path.is_some_and(Path::exists) {
                break;
            }
            let active_number = state
                .trials
                .iter()
                .find(|trial| {
                    !matches!(
                        trial.status,
                        TrialStatus::Complete
                            | TrialStatus::Failed
                            | TrialStatus::Rejected
                            | TrialStatus::Pruned
                    )
                })
                .map(|trial| trial.candidate.number);
            let candidate_number = if let Some(number) = active_number {
                number
            } else if state.trials.len() < spec.trial_count {
                let (candidate, _) = crate::experiments::generate_model_based_candidate(
                    registry,
                    base_config,
                    &spec,
                    &state.trials,
                )?;
                let identity = candidate.identity(&spec, &provenance)?;
                let number = candidate.number;
                state.trials.push(StudyTrial {
                    candidate,
                    identity,
                    status: TrialStatus::Pending,
                    measurement: None,
                    reason: Some(
                        "checkpointed deterministic observation-driven suggestion".to_string(),
                    ),
                    elapsed_ms: Some(0),
                    pareto_rank: None,
                    hard_constraint_violations: Vec::new(),
                });
                state.save(state_path)?;
                number
            } else {
                break;
            };

            let trial = state
                .trials
                .iter()
                .find(|trial| trial.candidate.number == candidate_number)
                .ok_or_else(|| anyhow!("model-based trial disappeared"))?;
            let config = registry
                .apply_tunable_values(base_config, &trial.candidate.parameters)
                .context("failed to apply model-based candidate")?;
            let identity = trial.identity.clone();
            let measurement = trial.measurement.clone().unwrap_or_default();
            let prior_elapsed_ms = trial.elapsed_ms.unwrap_or_default();
            {
                let trial = state
                    .trials
                    .iter_mut()
                    .find(|trial| trial.identity == identity)
                    .ok_or_else(|| anyhow!("model-based trial disappeared"))?;
                trial.status = TrialStatus::Running;
                trial.reason = None;
            }
            state.save(state_path)?;
            let result = Self::evaluate_study_candidate(
                StudyEvaluationRequest {
                    paths,
                    config: &config,
                    stage: spec.stage,
                    artifact_namespace: &identity,
                    evaluation_plan: &evaluation_plan,
                    top,
                    target_validation_folds: spec.maximum_validation_folds,
                    validation_fold_indices: &spec.fidelity_fold_indices(
                        evaluation_plan.folds.len(),
                        spec.maximum_validation_folds,
                    )?,
                    maximum_trial_seconds: spec.maximum_trial_seconds,
                    maximum_memory_mb: spec.maximum_memory_mb,
                    measure_latency: true,
                    measurement,
                    prior_elapsed_ms,
                    cancellation_path,
                },
                |measurement, elapsed_ms| {
                    let trial = state
                        .trials
                        .iter_mut()
                        .find(|trial| trial.identity == identity)
                        .ok_or_else(|| {
                            anyhow!("model-based trial disappeared during checkpoint")
                        })?;
                    trial.status = TrialStatus::Running;
                    trial.measurement = Some(measurement.clone());
                    trial.elapsed_ms = Some(elapsed_ms);
                    state.save(state_path)
                },
            );
            let trial = state
                .trials
                .iter_mut()
                .find(|trial| trial.identity == identity)
                .ok_or_else(|| anyhow!("model-based trial disappeared"))?;
            match result {
                Ok(Some(measurement)) => {
                    trial.status = TrialStatus::Complete;
                    trial.measurement = Some(measurement);
                    trial.reason = Some("completed observation-driven evaluation".to_string());
                }
                Ok(None) => {
                    trial.status = TrialStatus::Running;
                    trial.reason =
                        Some("paused by cooperative cancellation file; safe to resume".to_string());
                }
                Err(error) => {
                    trial.status = TrialStatus::Failed;
                    trial.reason = Some(format!("{error:#}"));
                }
            }
            crate::experiments::annotate_trial_outcomes(
                &mut state.trials,
                spec.maximum_validation_folds,
            );
            state.save(state_path)?;
            if cancellation_path.is_some_and(Path::exists) {
                break;
            }
        }
        ensure_development_source_identity(
            paths,
            provenance.development_cutoff,
            &provenance.input_fingerprint,
        )?;

        Self::summarize_study_run(
            state,
            registry,
            base_config,
            state_path,
            spec.parallelism,
            1,
        )
    }

    fn summarize_study_run(
        state: StudyState,
        registry: &ParameterRegistry,
        base_config: &PriorConfig,
        state_path: &Path,
        requested_parallelism: usize,
        effective_parallelism: usize,
    ) -> Result<StudyRunSummary> {
        let best = state.best_completed();
        let best_trial_number = best.map(|trial| trial.candidate.number);
        let best_measurement = best.and_then(|trial| trial.measurement.clone());
        let best_config = best
            .map(|trial| registry.apply_tunable_values(base_config, &trial.candidate.parameters))
            .transpose()?;
        Ok(StudyRunSummary {
            state_path: state_path.to_path_buf(),
            requested_parallelism,
            effective_parallelism,
            compute_threads: state.provenance.compute_threads,
            completed_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Complete)
                .count(),
            pending_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Pending)
                .count(),
            running_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Running)
                .count(),
            pruned_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Pruned)
                .count(),
            rejected_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Rejected)
                .count(),
            failed_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Failed)
                .count(),
            best_trial_number,
            best_measurement,
            best_config,
            sealed_test_evaluated: false,
        })
    }

    fn evaluate_study_candidate<F>(
        request: StudyEvaluationRequest<'_>,
        mut checkpoint: F,
    ) -> Result<Option<StudyMeasurement>>
    where
        F: FnMut(&StudyMeasurement, u64) -> Result<()>,
    {
        let StudyEvaluationRequest {
            paths,
            config,
            stage,
            artifact_namespace,
            evaluation_plan,
            top,
            target_validation_folds,
            validation_fold_indices,
            maximum_trial_seconds,
            maximum_memory_mb,
            measure_latency,
            mut measurement,
            prior_elapsed_ms,
            cancellation_path,
        } = request;
        let started = Instant::now();
        let time_budget_ms = maximum_trial_seconds.saturating_mul(1_000);
        let memory_budget_bytes = maximum_memory_mb.saturating_mul(1024 * 1024);
        let observe_memory = |measurement: &mut StudyMeasurement| -> Result<()> {
            let snapshot = crate::process_memory::process_memory_snapshot().ok_or_else(|| {
                anyhow!(
                    "hard memory budgets are unsupported on this platform; supported platforms are Windows, Linux, and macOS"
                )
            })?;
            measurement.peak_memory_bytes = Some(
                measurement
                    .peak_memory_bytes
                    .unwrap_or_default()
                    .max(snapshot.peak_working_set_bytes),
            );
            if snapshot.peak_working_set_bytes > memory_budget_bytes {
                bail!(
                    "study process peak working set {} MiB exceeded the {} MiB hard budget",
                    snapshot.peak_working_set_bytes.div_ceil(1024 * 1024),
                    maximum_memory_mb
                );
            }
            Ok(())
        };
        if validation_fold_indices.len() != target_validation_folds {
            bail!(
                "study fidelity selected {} folds but target is {target_validation_folds}",
                validation_fold_indices.len()
            );
        }
        let selected_folds = validation_fold_indices
            .iter()
            .map(|index| {
                evaluation_plan
                    .folds
                    .iter()
                    .find(|fold| fold.index == *index)
                    .ok_or_else(|| anyhow!("study selected unknown validation fold {index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        observe_memory(&mut measurement)?;
        for fold in selected_folds {
            if measurement.validation_fold_indices.contains(&fold.index) {
                continue;
            }
            if cancellation_path.is_some_and(Path::exists) {
                return Ok(None);
            }
            observe_memory(&mut measurement)?;
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            if elapsed_ms > time_budget_ms {
                bail!(
                    "study candidate exceeded {} second wall-clock budget before fold {}",
                    maximum_trial_seconds,
                    fold.index
                );
            }

            let mut fold_measurement = StudyMeasurement {
                validation_fold_indices: vec![fold.index],
                ..StudyMeasurement::default()
            };
            if stage.evaluates_prior_only() {
                for entry in solver
                    .history_dates
                    .iter()
                    .filter(|entry| fold.validation.contains(entry.print_date))
                {
                    observe_memory(&mut fold_measurement)?;
                    if cancellation_path.is_some_and(Path::exists) {
                        return Ok(None);
                    }
                    let elapsed_ms = prior_elapsed_ms
                        .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
                    if elapsed_ms > time_budget_ms {
                        bail!(
                            "study candidate exceeded {} second wall-clock budget",
                            maximum_trial_seconds
                        );
                    }
                    fold_measurement.scheduled_games += 1;
                    if let Some(metrics) =
                        solver.initial_prior_metrics(&entry.solution, entry.print_date)
                    {
                        fold_measurement.measured_prior_games += 1;
                        fold_measurement.log_loss_sum += metrics.log_loss;
                        fold_measurement.brier_score_sum += metrics.brier;
                    }
                }
                fold_measurement.coverage_gaps = fold_measurement
                    .scheduled_games
                    .saturating_sub(fold_measurement.measured_prior_games);
            } else {
                let book_usage = if stage.uses_predictive_books() {
                    PredictiveBookUsage::DiskOnly
                } else {
                    PredictiveBookUsage::None
                };
                let fold_solver = if stage.uses_predictive_books() {
                    let mut fold_solver = solver.clone();
                    fold_solver.artifact_dir =
                        study_book_artifact_dir(paths, artifact_namespace, fold.index);
                    fs::create_dir_all(&fold_solver.artifact_dir).with_context(|| {
                        format!(
                            "failed to create study book directory {}",
                            fold_solver.artifact_dir.display()
                        )
                    })?;
                    let last_artifact_cutoff =
                        fold.validation
                            .end
                            .checked_sub_days(Days::new(1))
                            .ok_or_else(|| anyhow!("book-study cutoff underflowed"))?;
                    let mut artifact_cutoff = fold.training.end;
                    loop {
                        fold_solver.build_predictive_opener_cache(artifact_cutoff)?;
                        fold_solver.build_predictive_reply_book(artifact_cutoff)?;
                        observe_memory(&mut fold_measurement)?;
                        if cancellation_path.is_some_and(Path::exists) {
                            return Ok(None);
                        }
                        let elapsed_ms = prior_elapsed_ms.saturating_add(
                            started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                        );
                        if elapsed_ms > time_budget_ms {
                            bail!(
                                "study candidate exceeded {} second wall-clock budget while building fold {} books",
                                maximum_trial_seconds,
                                fold.index
                            );
                        }
                        let Some(next_cutoff) = artifact_cutoff.checked_add_days(Days::new(
                            fold_solver.config.session_artifact_freshness_days as u64,
                        )) else {
                            break;
                        };
                        if next_cutoff > last_artifact_cutoff {
                            break;
                        }
                        artifact_cutoff = next_cutoff;
                    }
                    Some(fold_solver)
                } else {
                    None
                };
                let active_solver = fold_solver.as_ref().unwrap_or(&solver);
                let report = if stage.evaluates_recovery_only() {
                    match active_solver.recovery_backtest_detailed_with_book_usage(
                        fold.validation.start,
                        fold.validation.end,
                        top,
                        book_usage,
                    ) {
                        Ok(report) => Some(report),
                        Err(error)
                            if error
                                .to_string()
                                .contains("no out-of-primary recovery games") =>
                        {
                            None
                        }
                        Err(error) => return Err(error),
                    }
                } else {
                    Some(active_solver.backtest_detailed_with_book_usage(
                        fold.validation.start,
                        fold.validation.end,
                        top,
                        book_usage,
                    )?)
                };
                if let Some(report) = report {
                    fold_measurement.record_solve_metrics(&report.summary.canonical);
                    for entry in active_solver
                        .history_dates
                        .iter()
                        .filter(|entry| fold.validation.contains(entry.print_date))
                    {
                        if let Some(prior) =
                            active_solver.initial_prior_metrics(&entry.solution, entry.print_date)
                        {
                            fold_measurement.measured_prior_games += 1;
                            fold_measurement.log_loss_sum += prior.log_loss;
                            fold_measurement.brier_score_sum += prior.brier;
                        }
                    }
                }
                observe_memory(&mut fold_measurement)?;
            }
            fold_measurement.refresh_derived();
            measurement.merge_fold(&fold_measurement)?;
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            checkpoint(&measurement, elapsed_ms)?;
        }
        if measurement.scheduled_games == 0 && !stage.evaluates_recovery_only() {
            bail!("no games were evaluated by the requested study stage");
        }
        if !stage.evaluates_prior_only() && measure_latency {
            measurement.latency_p95_ms = Some(solver.benchmark_predictive_latency(
                evaluation_plan.development.end,
                default_diagnostic_suite()?.latency.study_runs,
            )?);
            observe_memory(&mut measurement)?;
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            if elapsed_ms > time_budget_ms {
                bail!(
                    "study candidate exceeded {} second wall-clock budget during latency measurement",
                    maximum_trial_seconds
                );
            }
            checkpoint(&measurement, elapsed_ms)?;
        }
        measurement.refresh_derived();
        let elapsed_ms = prior_elapsed_ms
            .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        checkpoint(&measurement, elapsed_ms)?;
        Ok(Some(measurement))
    }

    fn measure_study_candidate_latency(
        paths: &ProjectPaths,
        config: &PriorConfig,
        mut measurement: StudyMeasurement,
        prior_elapsed_ms: u64,
        maximum_trial_seconds: u64,
        maximum_memory_mb: u64,
    ) -> Result<(StudyMeasurement, u64)> {
        let started = Instant::now();
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        measurement.latency_p95_ms = Some(
            solver.benchmark_predictive_latency(
                canonical_development_evaluation_plan(paths, "study latency")?
                    .development
                    .end,
                default_diagnostic_suite()?.latency.study_runs,
            )?,
        );
        let snapshot = crate::process_memory::process_memory_snapshot().ok_or_else(|| {
            anyhow!(
                "hard memory budgets are unsupported on this platform; supported platforms are Windows, Linux, and macOS"
            )
        })?;
        measurement.peak_memory_bytes = Some(
            measurement
                .peak_memory_bytes
                .unwrap_or_default()
                .max(snapshot.peak_working_set_bytes),
        );
        if snapshot.peak_working_set_bytes > maximum_memory_mb.saturating_mul(1024 * 1024) {
            bail!(
                "study process peak working set {} MiB exceeded the {} MiB hard budget during latency measurement",
                snapshot.peak_working_set_bytes.div_ceil(1024 * 1024),
                maximum_memory_mb
            );
        }
        let elapsed_ms = prior_elapsed_ms
            .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        if elapsed_ms > maximum_trial_seconds.saturating_mul(1_000) {
            bail!(
                "study candidate exceeded {} second wall-clock budget during latency measurement",
                maximum_trial_seconds
            );
        }
        measurement.refresh_derived();
        Ok((measurement, elapsed_ms))
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

        let validation_current =
            Self::evaluate_tuning_candidate(paths, config, validation_start, validation_end)?;
        let candidate = Self::evaluate_tuning_candidate(
            paths,
            &best_prior_config,
            validation_start,
            validation_end,
        )?;
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
        let best = Self::evaluate_tuning_candidate(
            paths,
            &selected_config,
            validation_start,
            validation_end,
        )?;
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
        let run_count = runs.max(1);
        let top = default_diagnostic_suite()?.latency.top_suggestions;
        let mut samples = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            let start = Instant::now();
            let _ = self.suggest_predictive(PredictiveSuggestRequest {
                puzzle_date,
                observations: &[],
                top,
                hard_mode: false,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::LiveOnly,
            })?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|left, right| left.total_cmp(right));
        let p95_index = ((samples.len() as f64) * 0.95).ceil() as usize;
        Ok(samples[p95_index.saturating_sub(1)].max(0.0))
    }

    pub(super) fn benchmark_session_fallback_latency(
        &self,
        as_of: NaiveDate,
    ) -> Result<(f64, f64)> {
        let mut benchmark = self.clone();
        benchmark.session_opener_cache = Arc::new(Mutex::new(HashMap::new()));
        benchmark.session_reply_cache = Arc::new(Mutex::new(HashMap::new()));
        benchmark.session_third_cache = Arc::new(Mutex::new(HashMap::new()));

        let cold_started = Instant::now();
        let _ = benchmark.session_root_guess(as_of)?;
        let cold_ms = cold_started.elapsed().as_secs_f64() * 1_000.0;
        let warm_started = Instant::now();
        let _ = benchmark.session_root_guess(as_of)?;
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
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(FourGuessOpenerEvaluation {
            opener: opener.to_string(),
            average_guesses,
            three_guess_solves,
            failures,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
        })
    }

    pub(super) fn evaluate_tuning_candidate(
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
        _as_of: NaiveDate,
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        _top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        let opener = self.guesses[guess_index].clone();
        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut four_guess_games = 0usize;
        let mut failures = 0usize;
        for (date, target) in targets {
            let target_as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate opener before launch date"))?;
            let score =
                self.score_target_with_forced_opening(target, target_as_of, *date, &opener)?;
            if score.guesses >= 4 {
                four_guess_games += 1;
            }
            guess_counts.push(score.guesses);
            if !score.solved {
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
            four_guess_games,
            average_guesses,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
            failures,
        })
    }

    pub(super) fn evaluate_forced_reply(
        &self,
        opener: &str,
        _opener_feedback: u8,
        targets: &[(NaiveDate, String)],
        reply_guess_index: usize,
        top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        self.evaluate_forced_continuation(&[opener.to_string()], targets, reply_guess_index, top)
    }

    pub(super) fn evaluate_forced_continuation(
        &self,
        forced_prefix: &[String],
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        _top: usize,
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
            let score =
                self.score_target_with_forced_prefix(target, target_as_of, *date, &forced)?;
            guess_counts.push(score.guesses);
            if !score.solved {
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
            four_guess_games: guess_counts.iter().filter(|count| **count >= 4).count(),
            average_guesses,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
            failures,
        })
    }

    pub(super) fn select_validated_opener(
        &self,
        as_of: NaiveDate,
        candidates: &[Suggestion],
        primary_targets: &[(NaiveDate, String)],
        holdout_targets: Option<&[(NaiveDate, String)]>,
        top: usize,
    ) -> Result<Option<ValidatedOpenerEvaluation>> {
        let mut evaluations = candidates
            .par_iter()
            .filter_map(|suggestion| {
                let guess_index = self.guess_index.get(&suggestion.word).copied()?;
                let primary = self
                    .evaluate_forced_opener(as_of, primary_targets, guess_index, top)
                    .ok()?;
                Some(ValidatedOpenerEvaluation {
                    word: suggestion.word.clone(),
                    primary,
                    holdout: None,
                })
            })
            .collect::<Vec<_>>();
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
                evaluation.holdout = self
                    .evaluate_forced_opener(as_of, targets, evaluation.primary.guess_index, top)
                    .ok();
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

    pub(super) fn score_target_with_forced_opening(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        opener: &str,
    ) -> Result<ForcedSolveScore> {
        let forced = [(opener.to_string(), 0)];
        self.score_target_with_forced_prefix(target, as_of, date, &forced)
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
                book_usage: PredictiveBookUsage::None,
                search_mode: None,
                forced,
            },
        )
    }

    pub(super) fn score_target_with_forced_prefix(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        forced: &[(String, u8)],
    ) -> Result<ForcedSolveScore> {
        let run = self.solve_target_with_forced_prefix(target, as_of, date, forced, 1)?;
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
    if policy
        .excluded_validation
        .iter()
        .any(|excluded| requested.start <= excluded.end && excluded.start <= requested.end)
    {
        bail!("{operation} range {from}..{to} intersects a consumed validation window");
    }
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

fn study_book_artifact_dir(
    paths: &ProjectPaths,
    artifact_namespace: &str,
    fold_index: usize,
) -> PathBuf {
    let namespace_hash = crate::identity::digest_bytes_hex(
        "maybe-wordle-study-book-namespace-v2",
        artifact_namespace.as_bytes(),
    );
    paths
        .root
        .join("target/studies/predictive-books")
        .join(format!("trial-{namespace_hash}"))
        .join(format!("fold-{fold_index:02}"))
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
) -> Result<String> {
    // Disk artifacts are validated by the solver when they are used and are reported in the
    // baseline policy. Their mutable bytes are not available as a single checkpoint input, so
    // this identity deliberately makes no authentication claim about those artifacts.
    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-evidence-checkpoint-v3");
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
        .field(&(top as u64).to_le_bytes());
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
    use std::{
        collections::HashMap,
        fs,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use chrono::{Days, NaiveDate};

    use super::*;
    use crate::{
        config::PriorConfig,
        experiments::RollingOriginFold,
        model::{AnswerRecord, ModelVariant, WeightMode},
        pattern_table::PatternTable,
    };

    #[test]
    fn finite_step_evidence_keeps_deadline_and_does_not_serialize_heuristic_values() {
        let guesses = vec!["cigar".to_string(), "rebut".to_string()];
        let search = FiniteSearchResult {
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
        static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
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
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "maybe-wordle-finite-audit-test-{}-{unique}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
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
            guesses: guesses.clone(),
            answers,
            primary_answer_count: words.len(),
            history_dates: Vec::new(),
            pattern_table,
            guess_index: guesses
                .iter()
                .enumerate()
                .map(|(index, word)| (word.clone(), index))
                .collect::<HashMap<_, _>>(),
            artifact_dir: root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
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
            average_guesses_ci95: (
                canonical.conditional_mean_guesses_ci95.lower,
                canonical.conditional_mean_guesses_ci95.upper,
            ),
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
                average_log_loss: 0.0,
                average_brier: 0.0,
                average_target_probability: 1.0,
                average_target_rank: 1.0,
                prior_evidence: None,
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

    fn identity_test_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "maybe-wordle-development-identity-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
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
        let checkpoint = EvidenceMatrixCheckpoint {
            schema_version: EVIDENCE_CHECKPOINT_SCHEMA_VERSION,
            identity: "first".into(),
            elapsed_ms: 10,
            peak_working_set_bytes: 20,
            baselines: Vec::new(),
        };
        assert!(checkpoint.validate("first", &[]).is_ok());
        assert!(checkpoint.validate("second", &[]).is_err());
        let mut old = checkpoint;
        old.schema_version = 1;
        assert!(old.validate("first", &[]).is_err());
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
        )
        .expect("identity");
        assert_ne!(first, changed_matrix);
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
            schema_version: 4,
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
        let root = identity_test_root();
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
