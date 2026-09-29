use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use chrono::{Days, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, read_history_jsonl, read_word_list},
    experiments::{
        DateRange, EvaluationPlan,
        exhaustive_cost::{DatasetSplit, ExhaustiveCostDatasetArtifact},
    },
    predictive::learned_proxy::{
        ProxyModelArtifact, ProxyModelProvenance, ProxyRankingMetrics, ProxySplit,
        ProxyTrainingRow, RidgeConfig, evaluate_proxy_ranking, feature_schema_digest,
        fit_ridge_residual, validate_holdout_disjointness,
    },
    predictive::survival::{
        FoldSpec, LeftTruncationMetadata, PolicyEra, SurvivalConfig, SurvivalModel,
        SurvivalObservation, build_fold_training_inputs,
    },
    solver::{SearchRegretReport, Solver, SurvivalSolveComparison},
};

pub const LEARNED_PROXY_EXPERIMENT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnedProxyLambdaResult {
    pub lambda: f64,
    pub validation: ProxyRankingMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnedProxyExperimentReport {
    pub schema_version: u32,
    pub dataset_identity: String,
    pub dataset_digest: String,
    pub feature_names: Vec<String>,
    pub lambda_results: Vec<LearnedProxyLambdaResult>,
    pub selected_lambda: f64,
    pub baseline_validation: ProxyRankingMetrics,
    pub learned_validation: ProxyRankingMetrics,
    pub baseline_test: ProxyRankingMetrics,
    pub learned_test: ProxyRankingMetrics,
    #[serde(default)]
    pub reference_search_regret: Option<SearchRegretReport>,
    pub promotable: bool,
    pub promotion_blockers: Vec<String>,
    pub model_artifact: ProxyModelArtifact,
}

pub fn fit_learned_proxy_experiment(
    dataset: &ExhaustiveCostDatasetArtifact,
) -> Result<LearnedProxyExperimentReport> {
    dataset.validate()?;
    ensure!(
        dataset.progress.complete && dataset.progress.stop_reason.is_none(),
        "learned proxy refuses an incomplete exhaustive-cost dataset"
    );
    let feature_count = dataset
        .rows
        .first()
        .map(|row| row.feature_values.len())
        .ok_or_else(|| anyhow::anyhow!("learned proxy dataset has no rows"))?;
    let canonical_names = learned_proxy_feature_names();
    ensure!(
        feature_count == canonical_names.len(),
        "learned proxy dataset feature count does not match the canonical schema"
    );
    let rows = dataset
        .rows
        .iter()
        .map(|row| {
            Ok(ProxyTrainingRow {
                state_id: row.state.state_id.clone(),
                trajectory_id: row.state.trajectory_id.clone(),
                date: row
                    .state
                    .date
                    .ok_or_else(|| anyhow::anyhow!("proxy row has no date"))?,
                step_index: row.state.step_index,
                guess: row.guess.clone(),
                features: row.feature_values.clone(),
                baseline_proxy_cost: row
                    .baseline_proxy_cost
                    .ok_or_else(|| anyhow::anyhow!("proxy row has no baseline cost"))?,
                exact_continuation_cost: row.exact_continuation_cost,
                split: match row.split {
                    DatasetSplit::Train => ProxySplit::Train,
                    DatasetSplit::Validation => ProxySplit::Validation,
                    DatasetSplit::Test => ProxySplit::Test,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let training = rows
        .iter()
        .filter(|row| row.split == ProxySplit::Train)
        .cloned()
        .collect::<Vec<_>>();
    let validation = rows
        .iter()
        .filter(|row| row.split == ProxySplit::Validation)
        .cloned()
        .collect::<Vec<_>>();
    let test = rows
        .iter()
        .filter(|row| row.split == ProxySplit::Test)
        .cloned()
        .collect::<Vec<_>>();
    validate_holdout_disjointness(&training, &validation)?;
    validate_holdout_disjointness(&training, &test)?;
    let dataset_digest = dataset.digest_hex()?;
    let provenance = ProxyModelProvenance {
        dataset_identity: dataset.provenance.dataset_id.clone(),
        replay_identity: dataset.provenance.replay_digest()?,
        split_policy: "chronological-development-train-validation-test-v1".to_string(),
        feature_schema_digest: feature_schema_digest(&canonical_names),
    };
    let lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0];
    let mut candidates = Vec::new();
    for lambda in lambdas {
        let model = fit_ridge_residual(
            &training,
            &canonical_names,
            RidgeConfig {
                lambda,
                ..RidgeConfig::default()
            },
            provenance.clone(),
        )?;
        let metrics = evaluate_proxy_ranking(&model, &validation, ProxySplit::Validation)?;
        candidates.push((model, metrics));
    }
    candidates.sort_by(|left, right| {
        left.1
            .mean_regret
            .total_cmp(&right.1.mean_regret)
            .then_with(|| left.1.maximum_regret.total_cmp(&right.1.maximum_regret))
            .then_with(|| {
                right
                    .1
                    .pairwise_accuracy
                    .total_cmp(&left.1.pairwise_accuracy)
            })
            .then_with(|| left.0.ridge_lambda.total_cmp(&right.0.ridge_lambda))
    });
    let (model, learned_validation) = candidates
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("ridge search produced no model"))?;
    let mut baseline_model = model.clone();
    baseline_model.coefficients.fill(0.0);
    baseline_model.intercept = 0.0;
    baseline_model.target_mean_residual = 0.0;
    let baseline_validation =
        evaluate_proxy_ranking(&baseline_model, &validation, ProxySplit::Validation)?;
    let baseline_test = evaluate_proxy_ranking(&baseline_model, &test, ProxySplit::Test)?;
    let learned_test = evaluate_proxy_ranking(&model, &test, ProxySplit::Test)?;
    let lambda_results = candidates
        .iter()
        .map(|(model, validation)| LearnedProxyLambdaResult {
            lambda: model.ridge_lambda,
            validation: validation.clone(),
        })
        .collect();
    let promotion_blockers = vec![
        "Development-only continuation-cost ranking evidence is not a full rolling solve-quality gate."
            .to_string(),
        "The once-opened 2026-06-18 through 2026-07-17 sealed window was not reused."
            .to_string(),
        "Production integration remains disabled until coverage, failures, paired solve quality, latency, and memory all pass."
            .to_string(),
    ];
    Ok(LearnedProxyExperimentReport {
        schema_version: LEARNED_PROXY_EXPERIMENT_VERSION,
        dataset_identity: dataset.provenance.dataset_id.clone(),
        dataset_digest,
        feature_names: canonical_names,
        lambda_results,
        selected_lambda: model.ridge_lambda,
        baseline_validation,
        learned_validation: learned_validation.clone(),
        baseline_test,
        learned_test: learned_test.clone(),
        reference_search_regret: None,
        promotable: false,
        promotion_blockers,
        model_artifact: ProxyModelArtifact {
            format_version: crate::predictive::learned_proxy::LEARNED_PROXY_FORMAT_VERSION,
            model,
            evaluation: Some(learned_test),
        },
    })
}

pub use crate::predictive::learned_proxy::learned_proxy_feature_names;

pub const SURVIVAL_EXPERIMENT_VERSION: u32 = 2;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PriorScoreMetrics {
    pub games: usize,
    pub coverage_gaps: usize,
    pub covered_games: usize,
    pub log_loss_sum: f64,
    pub brier_sum: f64,
    pub target_rank_sum: f64,
    pub mean_log_loss: Option<f64>,
    pub mean_brier: Option<f64>,
    pub mean_target_rank: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalFoldEvidence {
    pub fold_id: String,
    pub training: DateRange,
    pub validation: DateRange,
    pub training_observations: usize,
    pub reuse_events: usize,
    pub training_rows: usize,
    pub converged: bool,
    pub training_fingerprint: String,
    pub logistic: PriorScoreMetrics,
    pub survival: PriorScoreMetrics,
    pub solve_search_policy: String,
    pub solve_comparison: SurvivalSolveComparison,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SolvePolicyAggregate {
    pub scheduled_games: usize,
    pub modeled_games: usize,
    pub solved_games: usize,
    pub unsolved_games: usize,
    pub coverage_gaps: usize,
    pub modeled_guess_total: usize,
    pub all_game_penalized_guess_total: f64,
    pub conditional_mean_guesses: Option<f64>,
    pub all_game_penalized_mean_guesses: Option<f64>,
    pub maximum_fold_latency_p95_ms: f64,
    pub peak_memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SurvivalSolveAggregateComparison {
    pub search_policy: String,
    pub baseline: SolvePolicyAggregate,
    pub survival: SolvePolicyAggregate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalExperimentReport {
    pub schema_version: u32,
    pub input_identity: String,
    pub source_identity: String,
    pub executable_fingerprint: String,
    pub evaluation_plan: EvaluationPlan,
    pub policy_eras: Vec<PolicyEra>,
    pub left_truncation: LeftTruncationMetadata,
    pub folds: Vec<SurvivalFoldEvidence>,
    pub logistic: PriorScoreMetrics,
    pub survival: PriorScoreMetrics,
    pub solve_quality: SurvivalSolveAggregateComparison,
    pub total_reuse_events: usize,
    pub fold_reuse_event_exposures: usize,
    pub elapsed_ms: u64,
    pub peak_memory_bytes: Option<u64>,
    pub promotable: bool,
    pub promotion_blockers: Vec<String>,
}

pub fn run_survival_experiment(
    paths: &ProjectPaths,
    config: &PriorConfig,
    solver: &Solver,
) -> Result<SurvivalExperimentReport> {
    let started = Instant::now();
    let executable_fingerprint = crate::solver::predictive_executable_fingerprint()?;
    let mut history = read_history_jsonl(&paths.raw_history)?;
    history.sort_by_key(|entry| entry.print_date);
    ensure!(
        !history.is_empty(),
        "survival experiment requires synced history"
    );
    let evaluation_plan = Solver::development_evaluation_plan(paths)?;
    history.retain(|entry| entry.print_date <= evaluation_plan.development.end);
    ensure!(
        !history.is_empty(),
        "survival experiment has no development history"
    );
    let history_range = DateRange::new(
        history.first().expect("non-empty").print_date,
        history.last().expect("non-empty").print_date,
    )?;
    let source_identity = crate::solver::predictive_development_source_identity(
        paths,
        evaluation_plan.development.end,
    )?;
    let support = read_word_list(&paths.seed_answers)?;
    ensure!(
        !support.is_empty(),
        "survival experiment requires seed answers"
    );
    let policy_eras = policy_eras_from_history(&history)?;
    let config_toml = toml::to_string_pretty(config)?;
    let mut identity = crate::identity::CanonicalSha256::new("maybe-wordle-survival-input-v1");
    identity
        .field(&serde_json::to_vec(&history)?)
        .field(support.join("\n").as_bytes())
        .field(config_toml.as_bytes())
        .field(&serde_json::to_vec(&evaluation_plan)?)
        .field(source_identity.as_bytes())
        .field(executable_fingerprint.as_bytes());
    let input_identity = identity.finish_tagged();
    let left_truncation = LeftTruncationMetadata {
        origin: history_range.start,
        retained_pre_origin: false,
        description: "First observed appearances are excluded as reuse events because the pre-history last-use date is unknown."
            .to_string(),
    };
    let mut folds = Vec::new();
    let mut unique_reuse_events = BTreeSet::new();
    for fold in &evaluation_plan.folds {
        let observations =
            survival_observations_at_cutoff(&history, &support, &policy_eras, fold.training.end)?;
        let fold_spec = FoldSpec::new(
            format!("rolling-{:02}", fold.index + 1),
            Some(fold.training.start),
            fold.training.end,
            fold.validation.start,
            fold.validation.end,
        );
        let training = build_fold_training_inputs(&observations, fold_spec)?;
        let model = SurvivalModel::fit_fold(&training, &policy_eras, &SurvivalConfig::default())
            .with_context(|| format!("survival fit is ineligible for fold {}", training.fold.id))?;
        let (logistic, survival) = evaluate_prior_curves(
            &history,
            &support,
            fold.training.end,
            fold.validation,
            config,
            &model,
        )?;
        let validation_games = history
            .iter()
            .filter(|entry| fold.validation.contains(entry.print_date))
            .cloned()
            .collect::<Vec<_>>();
        let solve_comparison =
            solver.compare_survival_model_on_games(&validation_games, &model, 1)?;
        let fold_events = reuse_event_identities(&training.observations);
        let reuse_events = fold_events.len();
        unique_reuse_events.extend(fold_events);
        eprintln!(
            "survival phase=fold fold={}/{} reuse_events={} logistic_logloss={} survival_logloss={} elapsed_s={:.1}",
            fold.index + 1,
            evaluation_plan.folds.len(),
            reuse_events,
            format_prior_mean(logistic.mean_log_loss),
            format_prior_mean(survival.mean_log_loss),
            started.elapsed().as_secs_f64()
        );
        folds.push(SurvivalFoldEvidence {
            fold_id: format!("rolling-{:02}", fold.index + 1),
            training: fold.training,
            validation: fold.validation,
            training_observations: model.training_observations,
            reuse_events,
            training_rows: model.training_rows,
            converged: model.converged,
            training_fingerprint: model.training_fingerprint,
            logistic,
            survival,
            solve_search_policy: "proxy_only_without_predictive_books".to_string(),
            solve_comparison,
        });
    }
    let logistic = aggregate_prior_metrics(folds.iter().map(|fold| &fold.logistic))?;
    let survival = aggregate_prior_metrics(folds.iter().map(|fold| &fold.survival))?;
    let total_reuse_events = unique_reuse_events.len();
    let fold_reuse_event_exposures = folds.iter().map(|fold| fold.reuse_events).sum();
    let solve_quality = aggregate_survival_solves(&folds)?;
    let mut promotion_blockers = Vec::new();
    if total_reuse_events < 100 {
        promotion_blockers.push(format!(
            "Only {total_reuse_events} unique reuse events are available ({fold_reuse_event_exposures} fold exposures); the hazard fit is too sparse for promotion."
        ));
    }
    if survival.coverage_gaps > 0 {
        promotion_blockers.push(format!(
            "The survival evaluation had {} support coverage gaps.",
            survival.coverage_gaps
        ));
    }
    if !prior_metrics_improve(&survival, &logistic) {
        promotion_blockers.push(
            "The survival curve did not beat the logistic baseline on both development log loss and Brier score."
                .to_string(),
        );
    }
    if solve_quality.survival.coverage_gaps > solve_quality.baseline.coverage_gaps
        || solve_quality.survival.unsolved_games > solve_quality.baseline.unsolved_games
        || !matches!((solve_quality.survival.all_game_penalized_mean_guesses,
            solve_quality.baseline.all_game_penalized_mean_guesses),
            (Some(candidate), Some(baseline)) if candidate < baseline)
    {
        promotion_blockers.push(
            "The survival prior did not strictly improve paired rolling solve quality without increasing failures or coverage gaps."
                .to_string(),
        );
    }
    if solve_quality.survival.maximum_fold_latency_p95_ms
        > solve_quality.baseline.maximum_fold_latency_p95_ms * 1.10
    {
        promotion_blockers.push(
            "The survival prior exceeded the declared 10% per-fold p95 latency guard.".to_string(),
        );
    }
    if matches!(
        (
            solve_quality.baseline.peak_memory_bytes,
            solve_quality.survival.peak_memory_bytes
        ),
        (Some(baseline), Some(survival)) if survival > baseline + 64 * 1024 * 1024
    ) {
        promotion_blockers.push(
            "The survival prior exceeded the declared 64 MiB process-peak memory allowance."
                .to_string(),
        );
    }
    promotion_blockers.push(
        "The paired solve audit deliberately uses the bounded proxy-only policy; production lookahead/exact solve quality remains a separate promotion gate."
            .to_string(),
    );
    promotion_blockers.push(
        "The sealed 2026-06-18 through 2026-07-17 outcomes remain excluded from tuning."
            .to_string(),
    );
    crate::solver::ensure_predictive_development_source_identity(
        paths,
        evaluation_plan.development.end,
        &source_identity,
    )?;
    Ok(SurvivalExperimentReport {
        schema_version: SURVIVAL_EXPERIMENT_VERSION,
        input_identity,
        source_identity,
        executable_fingerprint,
        evaluation_plan,
        policy_eras,
        left_truncation,
        folds,
        logistic,
        survival,
        solve_quality,
        total_reuse_events,
        fold_reuse_event_exposures,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        peak_memory_bytes: crate::process_memory::process_memory_snapshot()
            .map(|snapshot| snapshot.peak_working_set_bytes),
        promotable: false,
        promotion_blockers,
    })
}

fn reuse_event_identities(observations: &[SurvivalObservation]) -> BTreeSet<(String, NaiveDate)> {
    // The exclusive interval end identifies the same event across truncated or overlapping folds.
    observations
        .iter()
        .filter(|observation| observation.reused)
        .map(|observation| (observation.word.clone(), observation.exit_date))
        .collect()
}

fn format_prior_mean(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_string(), |value| format!("{value:.6}"))
}

fn prior_metrics_improve(candidate: &PriorScoreMetrics, baseline: &PriorScoreMetrics) -> bool {
    candidate.covered_games > 0
        && baseline.covered_games > 0
        && matches!((candidate.mean_log_loss, baseline.mean_log_loss, candidate.mean_brier, baseline.mean_brier),
        (Some(candidate_loss), Some(baseline_loss), Some(candidate_brier), Some(baseline_brier))
        if [candidate_loss, baseline_loss, candidate_brier, baseline_brier].iter().all(|value| value.is_finite())
            && candidate_loss < baseline_loss && candidate_brier < baseline_brier)
}

fn aggregate_survival_solves(
    folds: &[SurvivalFoldEvidence],
) -> Result<SurvivalSolveAggregateComparison> {
    Ok(SurvivalSolveAggregateComparison {
        search_policy: "proxy_only_without_predictive_books".to_string(),
        baseline: aggregate_solve_metrics(folds.iter().map(|fold| {
            let item = &fold.solve_comparison.baseline;
            (
                &item.summary.canonical,
                item.latency_p95_ms,
                item.peak_memory_bytes,
            )
        }))?,
        survival: aggregate_solve_metrics(folds.iter().map(|fold| {
            let item = &fold.solve_comparison.survival;
            (
                &item.summary.canonical,
                item.latency_p95_ms,
                item.peak_memory_bytes,
            )
        }))?,
    })
}

fn aggregate_solve_metrics<'a>(
    evidence: impl Iterator<Item = (&'a crate::experiments::PredictiveMetrics, f64, Option<u64>)>,
) -> Result<SolvePolicyAggregate> {
    let mut aggregate = SolvePolicyAggregate::default();
    for (metrics, latency_p95_ms, peak_memory_bytes) in evidence {
        aggregate.scheduled_games = aggregate
            .scheduled_games
            .checked_add(metrics.scheduled_games)
            .context("scheduled game count overflow")?;
        aggregate.modeled_games = aggregate
            .modeled_games
            .checked_add(metrics.modeled_games)
            .context("modeled game count overflow")?;
        aggregate.solved_games = aggregate
            .solved_games
            .checked_add(metrics.solved_games)
            .context("solved game count overflow")?;
        aggregate.unsolved_games = aggregate
            .unsolved_games
            .checked_add(metrics.unsolved_games)
            .context("unsolved game count overflow")?;
        aggregate.coverage_gaps = aggregate
            .coverage_gaps
            .checked_add(metrics.coverage_gaps)
            .context("coverage gap count overflow")?;
        aggregate.modeled_guess_total = aggregate
            .modeled_guess_total
            .checked_add(metrics.modeled_guess_total)
            .context("modeled guess total overflow")?;
        ensure!(
            metrics.all_game_penalized_guess_total.is_finite()
                && metrics.all_game_penalized_guess_total >= 0.0,
            "penalized guess total must be finite and non-negative"
        );
        aggregate.all_game_penalized_guess_total += metrics.all_game_penalized_guess_total;
        ensure!(
            aggregate.all_game_penalized_guess_total.is_finite(),
            "penalized guess total overflow"
        );
        ensure!(
            latency_p95_ms.is_finite() && latency_p95_ms >= 0.0,
            "fold latency must be finite and non-negative"
        );
        aggregate.maximum_fold_latency_p95_ms =
            aggregate.maximum_fold_latency_p95_ms.max(latency_p95_ms);
        aggregate.peak_memory_bytes = match (aggregate.peak_memory_bytes, peak_memory_bytes) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
    }
    aggregate.conditional_mean_guesses = (aggregate.modeled_games > 0)
        .then(|| aggregate.modeled_guess_total as f64 / aggregate.modeled_games as f64);
    aggregate.all_game_penalized_mean_guesses = (aggregate.scheduled_games > 0)
        .then(|| aggregate.all_game_penalized_guess_total / aggregate.scheduled_games as f64);
    Ok(aggregate)
}

fn policy_eras_from_history(history: &[NytDailyEntry]) -> Result<Vec<PolicyEra>> {
    let start = history
        .first()
        .map(|entry| entry.print_date)
        .ok_or_else(|| anyhow::anyhow!("history is empty"))?;
    let transition = history
        .iter()
        .find(|entry| {
            entry
                .editor
                .as_deref()
                .is_some_and(|editor| !editor.trim().is_empty())
        })
        .map(|entry| entry.print_date);
    let eras = match transition {
        Some(transition) if transition > start => vec![
            PolicyEra::new("pre_tracy", start, Some(transition)),
            PolicyEra::new("tracy_bennett", transition, None),
        ],
        _ => vec![PolicyEra::new("observed_policy", start, None)],
    };
    crate::predictive::survival::validate_policy_eras(&eras)?;
    Ok(eras)
}

fn survival_observations_at_cutoff(
    history: &[NytDailyEntry],
    support: &[String],
    eras: &[PolicyEra],
    cutoff: NaiveDate,
) -> Result<Vec<SurvivalObservation>> {
    let mut dates_by_word = BTreeMap::<String, Vec<NaiveDate>>::new();
    for entry in history.iter().filter(|entry| entry.print_date <= cutoff) {
        dates_by_word
            .entry(entry.solution.to_ascii_lowercase())
            .or_default()
            .push(entry.print_date);
    }
    let cutoff_exclusive = cutoff
        .checked_add_days(Days::new(1))
        .ok_or_else(|| anyhow::anyhow!("survival cutoff overflowed"))?;
    let mut observations = Vec::new();
    for (word, dates) in &dates_by_word {
        for pair in dates.windows(2) {
            let entry = pair[0]
                .checked_add_days(Days::new(1))
                .ok_or_else(|| anyhow::anyhow!("reuse interval overflowed"))?;
            let exit = pair[1]
                .checked_add_days(Days::new(1))
                .ok_or_else(|| anyhow::anyhow!("reuse interval overflowed"))?;
            append_policy_split_interval(&mut observations, word, entry, exit, true, eras)?;
        }
        let entry = dates
            .last()
            .expect("word has a date")
            .checked_add_days(Days::new(1))
            .ok_or_else(|| anyhow::anyhow!("censor interval overflowed"))?;
        if entry < cutoff_exclusive {
            append_policy_split_interval(
                &mut observations,
                word,
                entry,
                cutoff_exclusive,
                false,
                eras,
            )?;
        }
    }
    let used = dates_by_word.keys().cloned().collect::<BTreeSet<_>>();
    let era = era_for_date(eras, cutoff)?;
    for word in support {
        if !used.contains(word) {
            observations.push(SurvivalObservation::never_used(word, cutoff, era));
        }
    }
    Ok(observations)
}

fn append_policy_split_interval(
    output: &mut Vec<SurvivalObservation>,
    word: &str,
    entry: NaiveDate,
    exit: NaiveDate,
    reused: bool,
    eras: &[PolicyEra],
) -> Result<()> {
    let mut cursor = entry;
    while cursor < exit {
        let era = eras
            .iter()
            .find(|era| era.contains(cursor))
            .ok_or_else(|| anyhow::anyhow!("no policy era covers {cursor}"))?;
        let segment_end = era.end.map_or(exit, |end| end.min(exit));
        ensure!(
            segment_end > cursor,
            "policy-era split did not make progress"
        );
        let is_event_segment = reused && segment_end == exit;
        let elapsed_offset = (cursor - entry).num_days() as usize;
        let observation = if is_event_segment {
            SurvivalObservation::reused_observation(word, cursor, segment_end, &era.id)
        } else {
            SurvivalObservation::right_censored(word, cursor, segment_end, &era.id)
        };
        output.push(observation.with_elapsed_offset(elapsed_offset));
        cursor = segment_end;
    }
    Ok(())
}

fn evaluate_prior_curves(
    history: &[NytDailyEntry],
    seed_support: &[String],
    training_end: NaiveDate,
    validation: DateRange,
    config: &PriorConfig,
    model: &SurvivalModel,
) -> Result<(PriorScoreMetrics, PriorScoreMetrics)> {
    model.validate()?;
    crate::experiments::validate_predictive_config(config)?;
    let mut support = seed_support.iter().cloned().collect::<BTreeSet<_>>();
    let mut last_seen = BTreeMap::<String, NaiveDate>::new();
    for entry in history
        .iter()
        .filter(|entry| entry.print_date <= training_end)
    {
        let word = entry.solution.to_ascii_lowercase();
        support.insert(word.clone());
        last_seen.insert(word, entry.print_date);
    }
    let validation_games = history
        .iter()
        .filter(|entry| validation.contains(entry.print_date))
        .collect::<Vec<_>>();
    let mut logistic = ScoreAccumulator::default();
    let mut survival = ScoreAccumulator::default();
    for entry in validation_games {
        let target = entry.solution.to_ascii_lowercase();
        let mut logistic_scores = Vec::with_capacity(support.len());
        let mut survival_scores = Vec::with_capacity(support.len());
        let mut target_index = None;
        for (index, word) in support.iter().enumerate() {
            if word == &target {
                target_index = Some(index);
            }
            if let Some(last) = last_seen.get(word) {
                let elapsed = (entry.print_date - *last).num_days().max(0) as f64;
                logistic_scores.push(logistic_recency_score(elapsed, config).max(1e-12));
                survival_scores.push(
                    (1.0 - model
                        .try_predict_interval(*last, entry.print_date)?
                        .survival)
                        .max(1e-12),
                );
            } else {
                logistic_scores.push(1.0);
                survival_scores.push(1.0);
            }
        }
        logistic.observe(&logistic_scores, target_index)?;
        survival.observe(&survival_scores, target_index)?;
        support.insert(target.clone());
        last_seen.insert(target, entry.print_date);
    }
    Ok((logistic.finish(), survival.finish()))
}

fn logistic_recency_score(elapsed: f64, config: &PriorConfig) -> f64 {
    if elapsed < config.cooldown_days as f64 {
        return config.cooldown_floor;
    }
    let linear = (-config.logistic_k * (elapsed - config.midpoint_days)).clamp(-40.0, 40.0);
    config.cooldown_floor + (1.0 - config.cooldown_floor) / (1.0 + linear.exp())
}

fn era_for_date(eras: &[PolicyEra], date: NaiveDate) -> Result<&str> {
    eras.iter()
        .find(|era| era.contains(date))
        .map(|era| era.id.as_str())
        .with_context(|| format!("no policy era covers {date}"))
}

#[derive(Default)]
struct ScoreAccumulator {
    games: usize,
    coverage_gaps: usize,
    log_loss: f64,
    brier: f64,
    target_rank: f64,
}

impl ScoreAccumulator {
    fn observe(&mut self, scores: &[f64], target_index: Option<usize>) -> Result<()> {
        ensure!(
            scores
                .iter()
                .all(|score| score.is_finite() && *score >= 0.0),
            "prior scores must be finite and non-negative"
        );
        let total = scores.iter().sum::<f64>();
        ensure!(
            total.is_finite() && (total > 0.0 || (scores.is_empty() && target_index.is_none())),
            "prior scores must have finite positive total mass"
        );
        let Some(target_index) = target_index else {
            self.games += 1;
            self.coverage_gaps += 1;
            return Ok(());
        };
        ensure!(
            target_index < scores.len(),
            "prior target index is outside score support"
        );
        self.games += 1;
        let target_probability = scores[target_index] / total;
        self.log_loss += -target_probability.max(1e-15).ln();
        self.brier += scores
            .iter()
            .enumerate()
            .map(|(index, score)| {
                let probability = score / total;
                let outcome = if index == target_index { 1.0 } else { 0.0 };
                (probability - outcome) * (probability - outcome)
            })
            .sum::<f64>();
        self.target_rank += 1.0
            + scores
                .iter()
                .enumerate()
                .filter(|(index, score)| {
                    *index != target_index && score.total_cmp(&scores[target_index]).is_gt()
                })
                .count() as f64;
        Ok(())
    }

    fn finish(self) -> PriorScoreMetrics {
        let covered = self.games - self.coverage_gaps;
        PriorScoreMetrics {
            games: self.games,
            coverage_gaps: self.coverage_gaps,
            covered_games: covered,
            log_loss_sum: self.log_loss,
            brier_sum: self.brier,
            target_rank_sum: self.target_rank,
            mean_log_loss: (covered > 0).then(|| self.log_loss / covered as f64),
            mean_brier: (covered > 0).then(|| self.brier / covered as f64),
            mean_target_rank: (covered > 0).then(|| self.target_rank / covered as f64),
        }
    }
}

fn aggregate_prior_metrics<'a>(
    metrics: impl Iterator<Item = &'a PriorScoreMetrics>,
) -> Result<PriorScoreMetrics> {
    let mut aggregate = ScoreAccumulator::default();
    for metric in metrics {
        ensure!(
            metric.games.checked_sub(metric.coverage_gaps) == Some(metric.covered_games),
            "prior metric coverage counts are inconsistent"
        );
        let sums = [
            metric.log_loss_sum,
            metric.brier_sum,
            metric.target_rank_sum,
        ];
        ensure!(
            sums.iter().all(|sum| sum.is_finite() && *sum >= 0.0),
            "prior metric sums must be finite and non-negative"
        );
        ensure!(
            metric.covered_games > 0 || sums.iter().all(|sum| *sum == 0.0),
            "unmeasured prior metrics must have zero additive sums"
        );
        aggregate.games = aggregate
            .games
            .checked_add(metric.games)
            .context("prior game count overflow")?;
        aggregate.coverage_gaps = aggregate
            .coverage_gaps
            .checked_add(metric.coverage_gaps)
            .context("prior gap count overflow")?;
        aggregate.log_loss += metric.log_loss_sum;
        aggregate.brier += metric.brier_sum;
        aggregate.target_rank += metric.target_rank_sum;
        ensure!(
            [aggregate.log_loss, aggregate.brier, aggregate.target_rank]
                .iter()
                .all(|sum| sum.is_finite()),
            "prior metric sum overflow"
        );
    }
    Ok(aggregate.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, day).expect("date")
    }

    #[test]
    fn uncovered_fold_does_not_poison_measured_prior_metrics() {
        let mut uncovered = ScoreAccumulator::default();
        uncovered.observe(&[1.0, 1.0], None).expect("uncovered");
        let mut measured = ScoreAccumulator::default();
        measured.observe(&[1.0, 1.0], Some(0)).expect("measured");
        let metrics = [uncovered.finish(), measured.finish()];
        let aggregate = aggregate_prior_metrics(metrics.iter()).expect("aggregate");
        let json = serde_json::to_value(aggregate).expect("metrics json");
        assert_eq!(json["games"], 2);
        assert_eq!(json["coverage_gaps"], 1);
        assert_eq!(json["mean_log_loss"].as_f64(), Some(2.0_f64.ln()));
        assert_eq!(json["mean_brier"].as_f64(), Some(0.5));
        assert_eq!(json["mean_target_rank"].as_f64(), Some(1.0));
        assert_eq!(json["covered_games"], 1);
    }

    #[test]
    fn empty_and_all_gap_priors_have_zero_sums_and_unavailable_means() {
        let mut uncovered = ScoreAccumulator::default();
        uncovered.observe(&[], None).expect("gap");
        let metrics = [ScoreAccumulator::default().finish(), uncovered.finish()];
        let aggregate = aggregate_prior_metrics(metrics.iter()).expect("aggregate");
        for metric in metrics.iter().chain(std::iter::once(&aggregate)) {
            assert_eq!(metric.covered_games, 0);
            assert_eq!(
                (
                    metric.log_loss_sum,
                    metric.brier_sum,
                    metric.target_rank_sum
                ),
                (0.0, 0.0, 0.0)
            );
            assert_eq!(
                (
                    metric.mean_log_loss,
                    metric.mean_brier,
                    metric.mean_target_rank
                ),
                (None, None, None)
            );
            let json = serde_json::to_value(metric).expect("json");
            assert!(json["mean_log_loss"].is_null());
            assert_eq!(json["log_loss_sum"], 0.0);
            assert_eq!(format_prior_mean(metric.mean_log_loss), "unavailable");
        }
        let mut measured = ScoreAccumulator::default();
        measured.observe(&[1.0], Some(0)).expect("measured");
        let measured = measured.finish();
        assert!(!prior_metrics_improve(&aggregate, &measured));
        assert!(!prior_metrics_improve(&measured, &aggregate));
        assert_eq!(
            aggregate_prior_metrics(std::iter::empty())
                .expect("empty")
                .mean_log_loss,
            None
        );
    }

    #[test]
    fn invalid_prior_scores_are_errors_not_coverage_gaps() {
        let mut accumulator = ScoreAccumulator::default();
        for scores in [
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![-1.0],
            vec![0.0],
            vec![f64::MAX, f64::MAX],
        ] {
            assert!(accumulator.observe(&scores, Some(0)).is_err());
            assert!(accumulator.observe(&scores, None).is_err());
        }
        assert!(accumulator.observe(&[1.0], Some(1)).is_err());
        assert_eq!(accumulator.finish().games, 0);
        let mut invalid = PriorScoreMetrics {
            games: 1,
            covered_games: 1,
            log_loss_sum: f64::INFINITY,
            ..PriorScoreMetrics::default()
        };
        assert!(aggregate_prior_metrics([&invalid].into_iter()).is_err());
        invalid.log_loss_sum = f64::MAX;
        assert!(aggregate_prior_metrics([&invalid, &invalid].into_iter()).is_err());
        invalid.covered_games = 0;
        assert!(aggregate_prior_metrics([&invalid].into_iter()).is_err());
    }

    #[test]
    fn reuse_events_are_unique_across_overlapping_training_folds() {
        let first = vec![SurvivalObservation::reused_observation(
            "word",
            date(1),
            date(5),
            "a",
        )];
        let second = vec![
            SurvivalObservation::reused_observation("word", date(3), date(5), "a")
                .with_elapsed_offset(2),
            SurvivalObservation::reused_observation("word", date(6), date(9), "a"),
            SurvivalObservation::right_censored("other", date(3), date(5), "a"),
        ];
        let first = reuse_event_identities(&first);
        let second = reuse_event_identities(&second);
        assert_eq!(first.len() + second.len(), 3);
        assert_eq!(first.union(&second).count(), 2);
    }

    #[test]
    fn prior_evaluation_rejects_invalid_model_even_without_seen_words() {
        let mut model = SurvivalModel::fit_with_policy_eras(
            &[SurvivalObservation::reused_observation(
                "word",
                date(1),
                date(3),
                "a",
            )],
            &[PolicyEra::new("a", date(1), None)],
            &SurvivalConfig::default(),
        )
        .expect("model");
        model.converged = false;
        assert!(
            evaluate_prior_curves(
                &[],
                &["word".into()],
                date(3),
                DateRange::new(date(4), date(5)).expect("range"),
                &PriorConfig::default(),
                &model
            )
            .is_err()
        );
    }

    #[test]
    fn solve_aggregation_weights_all_modeled_games_including_failures() {
        use crate::experiments::{BootstrapConfig, GameOutcome, summarize_predictive_outcomes};
        let bootstrap = BootstrapConfig {
            resamples: 1,
            block_length: 1,
            seed: 1,
        };
        let first = summarize_predictive_outcomes(
            &[
                GameOutcome::solved(date(1), 1),
                GameOutcome::unsolved(date(2), 6),
            ],
            7.0,
            bootstrap,
        )
        .expect("first fold");
        let second =
            summarize_predictive_outcomes(&[GameOutcome::solved(date(3), 2)], 7.0, bootstrap)
                .expect("second fold");
        let aggregate =
            aggregate_solve_metrics([(&first, 1.0, None), (&second, 2.0, None)].into_iter())
                .expect("aggregate");
        assert_eq!(
            serde_json::to_value(aggregate).expect("json")["conditional_mean_guesses"].as_f64(),
            Some(3.0)
        );
    }

    #[test]
    fn solve_aggregates_match_flattened_games_for_every_partition() {
        use crate::experiments::{BootstrapConfig, GameOutcome, summarize_predictive_outcomes};
        let bootstrap = BootstrapConfig {
            resamples: 1,
            block_length: 1,
            seed: 1,
        };
        let games = [
            GameOutcome::coverage_gap(date(1)),
            GameOutcome::unsolved(date(2), 6),
            GameOutcome::solved(date(3), 2),
            GameOutcome::solved(date(4), 1),
            GameOutcome::coverage_gap(date(5)),
            GameOutcome::unsolved(date(6), 5),
        ];
        let flat = summarize_predictive_outcomes(&games, 7.0, bootstrap).expect("flat");
        for width in 1..=games.len() {
            let folds = games
                .chunks(width)
                .map(|games| summarize_predictive_outcomes(games, 7.0, bootstrap).expect("fold"))
                .collect::<Vec<_>>();
            let aggregate =
                aggregate_solve_metrics(folds.iter().map(|metrics| (metrics, 1.0, None)))
                    .expect("aggregate");
            assert_eq!(aggregate.scheduled_games, flat.scheduled_games);
            assert_eq!(aggregate.modeled_games, flat.modeled_games);
            assert_eq!(aggregate.modeled_guess_total, flat.modeled_guess_total);
            assert_eq!(
                aggregate.all_game_penalized_guess_total,
                flat.all_game_penalized_guess_total
            );
            assert_eq!(
                aggregate.conditional_mean_guesses,
                flat.conditional_mean_guesses
            );
            assert_eq!(
                aggregate.all_game_penalized_mean_guesses,
                Some(flat.all_game_penalized_mean_guesses)
            );
        }
        let gap = summarize_predictive_outcomes(&games[..1], 7.0, bootstrap).expect("gap");
        let gap = aggregate_solve_metrics([(&gap, 0.0, None)].into_iter()).expect("gap aggregate");
        assert_eq!(gap.conditional_mean_guesses, None);
        assert_eq!(gap.all_game_penalized_mean_guesses, Some(7.0));
        let empty = aggregate_solve_metrics(std::iter::empty()).expect("empty");
        assert_eq!(empty.conditional_mean_guesses, None);
        assert_eq!(empty.all_game_penalized_mean_guesses, None);
    }

    #[test]
    fn policy_split_keeps_the_original_elapsed_clock() {
        let eras = vec![
            PolicyEra::new("a", date(1), Some(date(5))),
            PolicyEra::new("b", date(5), None),
        ];
        let mut observations = Vec::new();
        append_policy_split_interval(&mut observations, "word", date(2), date(8), true, &eras)
            .expect("split");
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].elapsed_offset_days, 0);
        assert_eq!(observations[1].elapsed_offset_days, 3);
        assert!(!observations[0].reused);
        assert!(observations[1].reused);
    }
}
