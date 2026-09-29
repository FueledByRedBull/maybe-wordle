use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Days;
use rayon::prelude::*;

use crate::{
    config::PriorConfig,
    data::ProjectPaths,
    experiments::{
        EvaluationPlan, ParameterRegistry, StudyMeasurement, StudyProvenance, StudySearchStrategy,
        StudySpec, StudyState, StudyTrial, TrialStatus, default_diagnostic_suite,
        generate_candidates, predictive_parameter_registry,
    },
    model::{ModelVariant, WeightMode},
    solver::{PredictiveBookUsage, Solver, StudyRunSummary, exhaustive_teacher::WorkBudget},
};

use super::{
    StudyEvaluationRequest, canonical_development_evaluation_plan, development_source_identity,
    ensure_development_source_identity, git_provenance, validate_exact_date_coverage,
};

pub(super) fn needs_serial_study_latency(
    status: TrialStatus,
    completed_folds: usize,
    maximum_folds: usize,
    has_latency: bool,
) -> bool {
    status == TrialStatus::Complete && completed_folds >= maximum_folds && !has_latency
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

impl Solver {
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
                    let latency_started = Instant::now();
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
                            trial.elapsed_ms = Some(prior_elapsed_ms.saturating_add(
                                latency_started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                            ));
                            if let (Some(measurement), Some(snapshot)) = (
                                trial.measurement.as_mut(),
                                crate::process_memory::process_memory_snapshot(),
                            ) {
                                measurement.peak_memory_bytes = Some(
                                    measurement
                                        .peak_memory_bytes
                                        .unwrap_or(0)
                                        .max(snapshot.peak_working_set_bytes),
                                );
                            }
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
        let inner_budget = Mutex::new(WorkBudget::new(
            started,
            prior_elapsed_ms,
            Duration::from_millis(time_budget_ms),
            Some(memory_budget_bytes),
        ));
        let cancelled = || {
            cancellation_path.is_some_and(Path::exists)
                || inner_budget.lock().expect("study budget").check().is_err()
        };
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
        let result = (|| -> Result<Option<StudyMeasurement>> {
            inner_budget.lock().expect("study budget").check_now()?;
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

                validate_exact_date_coverage(
                    fold.validation,
                    solver.history_dates.iter().map(|entry| entry.print_date),
                )
                .with_context(|| format!("study fold {} calendar coverage", fold.index))?;
                let mut fold_measurement = StudyMeasurement {
                    validation_fold_indices: vec![fold.index],
                    examined_calendar_games: fold.validation.days() as usize,
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
                        let elapsed_ms = prior_elapsed_ms.saturating_add(
                            started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                        );
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
                        let last_artifact_cutoff = fold
                            .validation
                            .end
                            .checked_sub_days(Days::new(1))
                            .ok_or_else(|| anyhow!("book-study cutoff underflowed"))?;
                        let mut artifact_cutoff = fold.training.end;
                        loop {
                            fold_solver.build_predictive_opener_cache_controlled(
                                artifact_cutoff,
                                &cancelled,
                            )?;
                            fold_solver.build_predictive_reply_book_controlled(
                                artifact_cutoff,
                                &cancelled,
                            )?;
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
                    let games = if stage.evaluates_recovery_only() {
                        active_solver.recovery_target_games(
                            fold.validation.start,
                            fold.validation.end,
                            &cancelled,
                        )?
                    } else {
                        active_solver
                            .history_dates
                            .iter()
                            .filter(|entry| fold.validation.contains(entry.print_date))
                            .collect()
                    };
                    let report = if games.is_empty() {
                        None
                    } else {
                        let evaluated = active_solver.backtest_selected_games_controlled(
                            &games, top, book_usage, None, &cancelled,
                        );
                        if cancellation_path.is_some_and(Path::exists) {
                            return Ok(None);
                        }
                        if evaluated.is_err() {
                            inner_budget.lock().expect("study budget").check()?;
                        }
                        Some(evaluated?)
                    };
                    if let Some(report) = report {
                        fold_measurement.record_solve_metrics(&report.summary.canonical);
                        for entry in active_solver
                            .history_dates
                            .iter()
                            .filter(|entry| fold.validation.contains(entry.print_date))
                        {
                            if let Some(prior) = active_solver
                                .initial_prior_metrics(&entry.solution, entry.print_date)
                            {
                                fold_measurement.measured_prior_games += 1;
                                fold_measurement.log_loss_sum += prior.log_loss;
                                fold_measurement.brier_score_sum += prior.brier;
                            }
                        }
                    }
                }
                let completed_fold_memory = observe_memory(&mut fold_measurement);
                fold_measurement.refresh_derived();
                measurement.merge_fold(&fold_measurement)?;
                let elapsed_ms = prior_elapsed_ms
                    .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
                checkpoint(&measurement, elapsed_ms)?;
                // Preserve the complete fold before stopping on a late resource overrun.
                completed_fold_memory?;
                inner_budget.lock().expect("study budget").check_now()?;
            }
            if measurement.scheduled_games == 0 && !stage.evaluates_recovery_only() {
                bail!("no games were evaluated by the requested study stage");
            }
            if !stage.evaluates_prior_only() && measure_latency {
                measurement.latency_p95_ms = Some(solver.benchmark_predictive_latency_controlled(
                    evaluation_plan.development.end,
                    default_diagnostic_suite()?.latency.study_runs,
                    &cancelled,
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
            Ok(Some(measurement.clone()))
        })();
        // Failed or cancelled work consumes budget too; only complete folds enter metrics.
        if let Some(snapshot) = crate::process_memory::process_memory_snapshot() {
            measurement.peak_memory_bytes = Some(
                measurement
                    .peak_memory_bytes
                    .unwrap_or(0)
                    .max(snapshot.peak_working_set_bytes),
            );
        }
        let elapsed_ms = prior_elapsed_ms
            .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        checkpoint(&measurement, elapsed_ms)?;
        let final_budget = inner_budget.lock().expect("study budget").check_now();
        if final_budget.is_err() {
            // Account for the final checkpoint itself before persisting a late stop.
            if let Some(snapshot) = crate::process_memory::process_memory_snapshot() {
                measurement.peak_memory_bytes = Some(
                    measurement
                        .peak_memory_bytes
                        .unwrap_or(0)
                        .max(snapshot.peak_working_set_bytes),
                );
            }
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            checkpoint(&measurement, elapsed_ms)?;
        }
        if cancellation_path.is_some_and(Path::exists) {
            return Ok(None);
        }
        final_budget?;
        match result {
            Ok(Some(_)) => Ok(Some(measurement)),
            other => other,
        }
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
        let inner_budget = Mutex::new(WorkBudget::new(
            started,
            prior_elapsed_ms,
            Duration::from_secs(maximum_trial_seconds),
            Some(maximum_memory_mb.saturating_mul(1024 * 1024)),
        ));
        inner_budget
            .lock()
            .expect("study latency budget")
            .check_now()?;
        let cancelled = || {
            inner_budget
                .lock()
                .expect("study latency budget")
                .check()
                .is_err()
        };
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        measurement.latency_p95_ms = Some(
            solver.benchmark_predictive_latency_controlled(
                canonical_development_evaluation_plan(paths, "study latency")?
                    .development
                    .end,
                default_diagnostic_suite()?.latency.study_runs,
                &cancelled,
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
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use crate::experiments::{
        DateRange, RollingOriginConfig, StudyStage, build_rolling_origin_plan,
    };
    use chrono::NaiveDate;

    #[test]
    fn exhausted_study_resume_is_charged_before_loading_or_scoring() {
        let fixture = crate::test_support::TestDirectory::new("study-budget");
        let paths = ProjectPaths::new(fixture.path());
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let plan = build_rolling_origin_plan(
            DateRange::new(start, start.checked_add_days(Days::new(2)).unwrap()).unwrap(),
            RollingOriginConfig {
                minimum_training_days: 1,
                validation_days: 1,
                step_days: 1,
                sealed_test_days: 1,
                maximum_folds: 1,
            },
        )
        .unwrap();
        let mut saved = None;
        let error = Solver::evaluate_study_candidate(
            StudyEvaluationRequest {
                paths: &paths,
                config: &PriorConfig::default(),
                stage: StudyStage::Calibration,
                artifact_namespace: "budget-test",
                evaluation_plan: &plan,
                top: 1,
                target_validation_folds: 1,
                validation_fold_indices: &[plan.folds[0].index],
                maximum_trial_seconds: 1,
                maximum_memory_mb: u64::MAX,
                measure_latency: false,
                measurement: StudyMeasurement::default(),
                prior_elapsed_ms: 1_000,
                cancellation_path: None,
            },
            |measurement, elapsed| {
                saved = Some((measurement.clone(), elapsed));
                Ok(())
            },
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("cumulative wall-clock budget"));
        let (measurement, elapsed) = saved.expect("charge failed resume");
        assert!(elapsed >= 1_000);
        assert_eq!(measurement.examined_calendar_games, 0);
        assert!(measurement.validation_fold_indices.is_empty());
        assert!(!paths.root.join("data/derived").exists());
    }

    #[test]
    fn slow_final_checkpoint_is_charged_without_reporting_a_complete_trial() {
        let fixture = crate::test_support::TestDirectory::new("study-final-checkpoint");
        let paths = ProjectPaths::new(fixture.path());
        paths.ensure_layout().unwrap();
        fs::write(&paths.seed_guesses, "aaaaa\nbbbbb\n").unwrap();
        fs::write(&paths.seed_answers, "aaaaa\nbbbbb\n").unwrap();
        fs::write(&paths.manual_additions, "").unwrap();
        fs::write(&paths.raw_history, "").unwrap();
        let start = NaiveDate::from_ymd_opt(2032, 1, 1).unwrap();
        let plan = build_rolling_origin_plan(
            DateRange::new(start, start.checked_add_days(Days::new(2)).unwrap()).unwrap(),
            RollingOriginConfig {
                minimum_training_days: 1,
                validation_days: 1,
                step_days: 1,
                sealed_test_days: 1,
                maximum_folds: 1,
            },
        )
        .unwrap();
        let mut measurement = StudyMeasurement {
            validation_fold_indices: vec![plan.folds[0].index],
            examined_calendar_games: 1,
            scheduled_games: 1,
            measured_prior_games: 1,
            log_loss_sum: 0.5,
            brier_score_sum: 0.25,
            ..StudyMeasurement::default()
        };
        measurement.refresh_derived();
        let config = PriorConfig::default();
        let request = || StudyEvaluationRequest {
            paths: &paths,
            config: &config,
            stage: StudyStage::Calibration,
            artifact_namespace: "final-checkpoint-test",
            evaluation_plan: &plan,
            top: 1,
            target_validation_folds: 1,
            validation_fold_indices: &measurement.validation_fold_indices,
            maximum_trial_seconds: 1,
            maximum_memory_mb: u64::MAX,
            measure_latency: false,
            measurement: measurement.clone(),
            prior_elapsed_ms: 0,
            cancellation_path: None,
        };
        let mut calls = 0;
        let mut saved = None;
        let error = Solver::evaluate_study_candidate(request(), |measurement, elapsed| {
            calls += 1;
            saved = Some((measurement.clone(), elapsed));
            if calls == 2 {
                std::thread::sleep(Duration::from_millis(1_100));
            }
            Ok(())
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("wall-clock budget"));
        let (saved_measurement, elapsed) = saved.unwrap();
        assert!(elapsed >= 1_000, "late checkpoint time must survive resume");
        assert_eq!(
            saved_measurement.validation_fold_indices,
            measurement.validation_fold_indices
        );
        assert_eq!(saved_measurement.scheduled_games, 1);
        assert_eq!(saved_measurement.log_loss_sum, 0.5);
        assert!(saved_measurement.peak_memory_bytes.is_some());

        let mut final_saved = None;
        let completed = Solver::evaluate_study_candidate(request(), |measurement, _| {
            final_saved = Some(measurement.clone());
            Ok(())
        })
        .unwrap()
        .unwrap();
        assert_eq!(completed, final_saved.unwrap());
    }
}
