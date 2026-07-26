use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{atomic_file::atomic_write, config::PriorConfig};

use super::{
    EvaluationPlan, ParameterCohort, ParameterDefinition, ParameterKind, ParameterRegistry,
    ParameterScale, ParameterValue,
};

pub const STUDY_FORMAT_VERSION: u32 = 16;

fn default_maximum_validation_folds() -> usize {
    12
}

fn default_maximum_trial_seconds() -> u64 {
    7_200
}

fn default_maximum_memory_mb() -> u64 {
    4_096
}

fn default_initial_validation_folds() -> usize {
    3
}

fn default_reduction_factor() -> usize {
    3
}

fn default_study_strategy() -> StudySearchStrategy {
    StudySearchStrategy::LowDiscrepancy
}

fn default_fold_selection() -> StudyFoldSelection {
    StudyFoldSelection::NestedTimeSpread
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StudySearchStrategy {
    Grid,
    #[default]
    LowDiscrepancy,
    Random,
    LocalRefinement,
    ModelBased,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StudyFoldSelection {
    #[default]
    NestedTimeSpread,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StudyProvenance {
    pub identity_format: String,
    pub base_config_toml: String,
    pub registry_format_version: u32,
    pub registry_fingerprint: String,
    pub input_fingerprint: String,
    pub operating_system: String,
    pub architecture: String,
    pub compute_threads: usize,
    pub code_revision: Option<String>,
    pub code_dirty: Option<bool>,
    pub history_snapshot_start: chrono::NaiveDate,
    pub history_snapshot_end: chrono::NaiveDate,
    pub development_cutoff: chrono::NaiveDate,
    pub top_suggestions: usize,
}

impl StudyProvenance {
    fn identity(&self) -> Result<String> {
        serde_json::to_string(self).context("failed to serialize study provenance identity")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StudyStage {
    Calibration,
    CoverageRecovery,
    ProxyCore,
    ProxyRisk,
    ProxySmallState,
    ProxyRanker,
    SearchRouting,
    SearchExact,
    SearchCoverage,
    SearchLookahead,
    SearchPool,
    SearchDanger,
    SearchPenalty,
    SolvePolicy,
    BookPolicy,
    Joint,
}

impl StudyStage {
    fn includes(self, cohort: ParameterCohort) -> bool {
        match self {
            Self::Calibration => cohort == ParameterCohort::PriorCalibration,
            Self::CoverageRecovery => cohort == ParameterCohort::CoverageRecovery,
            Self::ProxyCore => cohort == ParameterCohort::ProxyCore,
            Self::ProxyRisk => cohort == ParameterCohort::ProxyRisk,
            Self::ProxySmallState => cohort == ParameterCohort::ProxySmallState,
            Self::ProxyRanker => matches!(
                cohort,
                ParameterCohort::ProxyCore
                    | ParameterCohort::ProxyRisk
                    | ParameterCohort::ProxySmallState
            ),
            Self::SearchRouting => cohort == ParameterCohort::SearchRouting,
            Self::SearchExact => cohort == ParameterCohort::SearchExact,
            Self::SearchCoverage => cohort == ParameterCohort::SearchCoverage,
            Self::SearchLookahead => cohort == ParameterCohort::SearchLookahead,
            Self::SearchPool => cohort == ParameterCohort::SearchPool,
            Self::SearchDanger => cohort == ParameterCohort::SearchDanger,
            Self::SearchPenalty => cohort == ParameterCohort::SearchPenalty,
            Self::SolvePolicy => matches!(
                cohort,
                ParameterCohort::SearchRouting
                    | ParameterCohort::SearchExact
                    | ParameterCohort::SearchCoverage
                    | ParameterCohort::SearchLookahead
                    | ParameterCohort::SearchPool
                    | ParameterCohort::SearchDanger
                    | ParameterCohort::SearchPenalty
            ),
            Self::BookPolicy => cohort == ParameterCohort::BookPolicy,
            Self::Joint => matches!(
                cohort,
                ParameterCohort::PriorCalibration
                    | ParameterCohort::CoverageRecovery
                    | ParameterCohort::ProxyCore
                    | ParameterCohort::ProxyRisk
                    | ParameterCohort::ProxySmallState
                    | ParameterCohort::SearchRouting
                    | ParameterCohort::SearchExact
                    | ParameterCohort::SearchCoverage
                    | ParameterCohort::SearchLookahead
                    | ParameterCohort::SearchPool
                    | ParameterCohort::SearchDanger
                    | ParameterCohort::SearchPenalty
            ),
        }
    }

    pub fn evaluates_prior_only(self) -> bool {
        self == Self::Calibration
    }

    pub fn evaluates_recovery_only(self) -> bool {
        self == Self::CoverageRecovery
    }

    pub fn uses_predictive_books(self) -> bool {
        self == Self::BookPolicy
    }

    fn requires_complete_one_factor_sweep(self) -> bool {
        !matches!(self, Self::ProxyRanker | Self::SolvePolicy | Self::Joint)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StudySpec {
    pub name: String,
    pub stage: StudyStage,
    pub seed: u64,
    pub trial_count: usize,
    pub parallelism: usize,
    #[serde(default = "default_study_strategy")]
    pub strategy: StudySearchStrategy,
    #[serde(default = "default_maximum_validation_folds")]
    pub maximum_validation_folds: usize,
    #[serde(default = "default_initial_validation_folds")]
    pub initial_validation_folds: usize,
    #[serde(default = "default_reduction_factor")]
    pub reduction_factor: usize,
    #[serde(default = "default_fold_selection")]
    pub fold_selection: StudyFoldSelection,
    #[serde(default = "default_maximum_trial_seconds")]
    pub maximum_trial_seconds: u64,
    #[serde(default = "default_maximum_memory_mb")]
    pub maximum_memory_mb: u64,
}

impl StudySpec {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            bail!("study name must not be empty");
        }
        if self.trial_count == 0 {
            bail!("trial_count must be positive");
        }
        if self.parallelism == 0 {
            bail!("parallelism must be positive");
        }
        if self.maximum_validation_folds == 0 {
            bail!("maximum_validation_folds must be positive");
        }
        if self.initial_validation_folds == 0
            || self.initial_validation_folds > self.maximum_validation_folds
        {
            bail!("initial_validation_folds must be in 1..=maximum_validation_folds");
        }
        if self.reduction_factor < 2 {
            bail!("reduction_factor must be at least two");
        }
        if self.maximum_trial_seconds == 0 {
            bail!("maximum_trial_seconds must be positive");
        }
        if self.maximum_memory_mb == 0 {
            bail!("maximum_memory_mb must be positive");
        }
        Ok(())
    }

    pub fn fidelity_schedule(&self) -> Vec<usize> {
        let mut schedule = vec![
            self.initial_validation_folds
                .min(self.maximum_validation_folds),
        ];
        while *schedule.last().expect("fidelity schedule is non-empty")
            < self.maximum_validation_folds
        {
            let current = *schedule.last().expect("fidelity schedule is non-empty");
            let next = current
                .saturating_mul(self.reduction_factor)
                .min(self.maximum_validation_folds);
            if next == current {
                break;
            }
            schedule.push(next);
        }
        schedule
    }

    pub fn fidelity_fold_indices(
        &self,
        available_folds: usize,
        target_folds: usize,
    ) -> Result<Vec<usize>> {
        if self.maximum_validation_folds > available_folds {
            bail!(
                "study requests {} validation folds but only {available_folds} are available",
                self.maximum_validation_folds
            );
        }
        if target_folds == 0 || target_folds > self.maximum_validation_folds {
            bail!(
                "target fidelity must be in 1..={}, got {target_folds}",
                self.maximum_validation_folds
            );
        }
        let order = match self.fold_selection {
            StudyFoldSelection::NestedTimeSpread => {
                nested_time_spread_order(available_folds, self.initial_validation_folds)
            }
        };
        Ok(order.into_iter().take(target_folds).collect())
    }
}

fn nested_time_spread_order(available_folds: usize, initial_folds: usize) -> Vec<usize> {
    let strata = initial_folds.min(available_folds);
    let mut per_stratum = Vec::with_capacity(strata);
    for stratum in 0..strata {
        let start = stratum * available_folds / strata;
        let end = (stratum + 1) * available_folds / strata;
        per_stratum.push(farthest_point_order(start, end));
    }

    let mut order = Vec::with_capacity(available_folds);
    let mut round = 0usize;
    while order.len() < available_folds {
        for stratum in &per_stratum {
            if let Some(index) = stratum.get(round) {
                order.push(*index);
            }
        }
        round += 1;
    }
    order
}

fn farthest_point_order(start: usize, end: usize) -> Vec<usize> {
    let mut selected = vec![start + (end - start - 1) / 2];
    while selected.len() < end - start {
        let next = (start..end)
            .filter(|candidate| !selected.contains(candidate))
            .max_by_key(|candidate| {
                let nearest = selected
                    .iter()
                    .map(|chosen| candidate.abs_diff(*chosen))
                    .min()
                    .expect("a stratum always has an initial point");
                (nearest, std::cmp::Reverse(*candidate))
            })
            .expect("an incomplete stratum has an unselected point");
        selected.push(next);
    }
    selected
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StudyCandidate {
    pub number: usize,
    pub seed: u64,
    pub parameters: BTreeMap<String, ParameterValue>,
}

impl StudyCandidate {
    pub fn equivalent_to(&self, other: &Self) -> bool {
        self.number == other.number
            && self.seed == other.seed
            && self.parameters.len() == other.parameters.len()
            && self.parameters.iter().all(|(name, left)| {
                other
                    .parameters
                    .get(name)
                    .is_some_and(|right| parameter_values_equivalent(left, right))
            })
    }

    pub fn identity(&self, spec: &StudySpec, provenance: &StudyProvenance) -> Result<String> {
        #[derive(Serialize)]
        struct Identity<'a> {
            format_version: u32,
            study: &'a str,
            stage: StudyStage,
            study_seed: u64,
            trial_seed: u64,
            maximum_validation_folds: usize,
            initial_validation_folds: usize,
            reduction_factor: usize,
            maximum_trial_seconds: u64,
            maximum_memory_mb: u64,
            parallelism: usize,
            strategy: StudySearchStrategy,
            provenance: String,
            parameters: BTreeMap<&'a str, String>,
        }

        let parameters = self
            .parameters
            .iter()
            .map(|(name, value)| (name.as_str(), canonical_parameter_value(value)))
            .collect();

        serde_json::to_string(&Identity {
            format_version: STUDY_FORMAT_VERSION,
            study: &spec.name,
            stage: spec.stage,
            study_seed: spec.seed,
            trial_seed: self.seed,
            maximum_validation_folds: spec.maximum_validation_folds,
            initial_validation_folds: spec.initial_validation_folds,
            reduction_factor: spec.reduction_factor,
            maximum_trial_seconds: spec.maximum_trial_seconds,
            maximum_memory_mb: spec.maximum_memory_mb,
            parallelism: spec.parallelism,
            strategy: spec.strategy,
            provenance: provenance.identity()?,
            parameters,
        })
        .context("failed to serialize trial identity")
    }
}

fn canonical_parameter_value(value: &ParameterValue) -> String {
    match value {
        ParameterValue::Float(value) => {
            let stable = serde_json::to_string(value)
                .ok()
                .and_then(|encoded| serde_json::from_str::<f64>(&encoded).ok())
                .unwrap_or(*value);
            format!("float:{:016x}", stable.to_bits())
        }
        ParameterValue::Integer(value) => format!("integer:{value}"),
        ParameterValue::Categorical(value) => format!("categorical:{value}"),
        ParameterValue::FloatMap => "float_map".to_string(),
    }
}

fn parameter_set_key(parameters: &BTreeMap<String, ParameterValue>) -> String {
    parameters
        .iter()
        .map(|(name, value)| {
            format!(
                "{}:{}:{}",
                name.len(),
                name,
                canonical_parameter_value(value)
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn parameter_values_equivalent(left: &ParameterValue, right: &ParameterValue) -> bool {
    match (left, right) {
        (ParameterValue::Float(left), ParameterValue::Float(right)) => {
            left == right
                || (left - right).abs()
                    <= f64::EPSILON * left.abs().max(right.abs()).max(f64::MIN_POSITIVE) * 4.0
        }
        (ParameterValue::Integer(left), ParameterValue::Integer(right)) => left == right,
        (ParameterValue::Categorical(left), ParameterValue::Categorical(right)) => left == right,
        (ParameterValue::FloatMap, ParameterValue::FloatMap) => true,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Pending,
    Running,
    Complete,
    Rejected,
    Failed,
    Pruned,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StudyMeasurement {
    pub validation_fold_indices: Vec<usize>,
    pub scheduled_games: usize,
    pub solve_metrics_recorded: bool,
    pub solved_games: usize,
    pub coverage_gaps: usize,
    pub failures: usize,
    pub measured_prior_games: usize,
    pub penalized_guess_sum: f64,
    pub solved_guess_sum: f64,
    pub log_loss_sum: f64,
    pub brier_score_sum: f64,
    pub all_game_penalized_mean_guesses: Option<f64>,
    pub conditional_mean_guesses: Option<f64>,
    pub average_log_loss: Option<f64>,
    pub average_brier_score: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
}

impl StudyMeasurement {
    pub fn merge_fold(&mut self, fold: &Self) -> Result<()> {
        if fold.validation_fold_indices.is_empty() {
            bail!("study fold measurement must identify at least one validation fold");
        }
        if fold
            .validation_fold_indices
            .iter()
            .any(|index| self.validation_fold_indices.contains(index))
        {
            bail!("study fold measurement overlaps an already evaluated fold");
        }
        self.validation_fold_indices
            .extend(fold.validation_fold_indices.iter().copied());
        self.validation_fold_indices.sort_unstable();
        self.scheduled_games += fold.scheduled_games;
        self.solve_metrics_recorded |= fold.solve_metrics_recorded;
        self.solved_games += fold.solved_games;
        self.coverage_gaps += fold.coverage_gaps;
        self.failures += fold.failures;
        self.measured_prior_games += fold.measured_prior_games;
        self.penalized_guess_sum += fold.penalized_guess_sum;
        self.solved_guess_sum += fold.solved_guess_sum;
        self.log_loss_sum += fold.log_loss_sum;
        self.brier_score_sum += fold.brier_score_sum;
        self.peak_memory_bytes = match (self.peak_memory_bytes, fold.peak_memory_bytes) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
        self.refresh_derived();
        Ok(())
    }

    pub fn refresh_derived(&mut self) {
        self.all_game_penalized_mean_guesses = (self.solve_metrics_recorded
            && self.scheduled_games > 0)
            .then(|| self.penalized_guess_sum / self.scheduled_games as f64);
        self.conditional_mean_guesses =
            (self.solved_games > 0).then(|| self.solved_guess_sum / self.solved_games as f64);
        self.average_log_loss = (self.measured_prior_games > 0)
            .then(|| self.log_loss_sum / self.measured_prior_games as f64);
        self.average_brier_score = (self.measured_prior_games > 0)
            .then(|| self.brier_score_sum / self.measured_prior_games as f64);
    }

    pub fn compare_guarded(&self, other: &Self) -> Ordering {
        self.coverage_gaps
            .cmp(&other.coverage_gaps)
            .then_with(|| self.failures.cmp(&other.failures))
            .then_with(|| {
                compare_optional(
                    self.all_game_penalized_mean_guesses,
                    other.all_game_penalized_mean_guesses,
                )
            })
            .then_with(|| compare_optional(self.average_log_loss, other.average_log_loss))
            .then_with(|| compare_optional(self.average_brier_score, other.average_brier_score))
            .then_with(|| compare_optional(self.latency_p95_ms, other.latency_p95_ms))
            .then_with(|| compare_optional_u64(self.peak_memory_bytes, other.peak_memory_bytes))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StudyConstraintViolation {
    pub constraint: String,
    pub observed: f64,
    pub maximum: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StudyTrial {
    pub candidate: StudyCandidate,
    pub identity: String,
    pub status: TrialStatus,
    pub measurement: Option<StudyMeasurement>,
    pub reason: Option<String>,
    pub elapsed_ms: Option<u64>,
    #[serde(default)]
    pub pareto_rank: Option<usize>,
    #[serde(default)]
    pub hard_constraint_violations: Vec<StudyConstraintViolation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StudyState {
    pub format_version: u32,
    pub spec: StudySpec,
    pub evaluation_plan: EvaluationPlan,
    pub provenance: StudyProvenance,
    pub sealed_test_evaluated: bool,
    pub trials: Vec<StudyTrial>,
}

impl StudyState {
    pub fn new(
        spec: StudySpec,
        evaluation_plan: EvaluationPlan,
        provenance: StudyProvenance,
    ) -> Result<Self> {
        spec.validate()?;
        Ok(Self {
            format_version: STUDY_FORMAT_VERSION,
            spec,
            evaluation_plan,
            provenance,
            sealed_test_evaluated: false,
            trials: Vec::new(),
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.format_version != STUDY_FORMAT_VERSION {
            bail!("unsupported study format version: {}", self.format_version);
        }
        if self.provenance.identity_format != crate::identity::IDENTITY_FORMAT
            || !crate::identity::is_tagged_digest(&self.provenance.registry_fingerprint)
            || !crate::identity::is_tagged_digest(&self.provenance.input_fingerprint)
        {
            bail!(
                "study uses an unsupported or mixed identity format; start a new format-v{} state",
                STUDY_FORMAT_VERSION
            );
        }
        self.spec.validate()?;
        if self.sealed_test_evaluated {
            bail!("optimizer state must not contain sealed-test evaluation");
        }
        let mut identities = HashSet::new();
        for trial in &self.trials {
            if !identities.insert(trial.identity.as_str()) {
                bail!("duplicate trial identity");
            }
            let expected_identity = trial.candidate.identity(&self.spec, &self.provenance)?;
            if trial.identity != expected_identity {
                bail!(
                    "trial {} identity does not match its candidate: stored={} expected={}",
                    trial.candidate.number,
                    trial.identity,
                    expected_identity
                );
            }
            match trial.status {
                TrialStatus::Complete if trial.measurement.is_none() => {
                    bail!("completed trial is missing its measurement")
                }
                TrialStatus::Rejected | TrialStatus::Failed | TrialStatus::Pruned
                    if trial.reason.as_deref().is_none_or(str::is_empty) =>
                {
                    bail!("terminal non-complete trial is missing its reason")
                }
                _ => {}
            }
            if trial.pareto_rank.is_some() && trial.measurement.is_none() {
                bail!("Pareto-ranked trial is missing its measurement");
            }
            for violation in &trial.hard_constraint_violations {
                if violation.constraint.trim().is_empty()
                    || !violation.observed.is_finite()
                    || !violation.maximum.is_finite()
                    || violation.observed <= violation.maximum
                {
                    bail!("trial contains an invalid hard-constraint violation");
                }
            }
        }
        Ok(())
    }

    pub fn completed_identities(&self) -> HashSet<&str> {
        self.trials
            .iter()
            .filter(|trial| trial.status == TrialStatus::Complete)
            .map(|trial| trial.identity.as_str())
            .collect()
    }

    pub fn best_completed(&self) -> Option<&StudyTrial> {
        self.trials
            .iter()
            .filter(|trial| trial.status == TrialStatus::Complete)
            .filter(|trial| trial.measurement.is_some())
            .min_by(|left, right| {
                left.measurement
                    .as_ref()
                    .expect("filtered")
                    .compare_guarded(right.measurement.as_ref().expect("filtered"))
                    .then_with(|| left.candidate.number.cmp(&right.candidate.number))
            })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self).context("failed to serialize study state")?;
        atomic_write(path, &bytes)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let state: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        if state.format_version < STUDY_FORMAT_VERSION {
            bail!(
                "study state format {} lacks complete provenance; start a new format-v{} state instead of resuming it",
                state.format_version,
                STUDY_FORMAT_VERSION
            );
        }
        state.validate()?;
        Ok(state)
    }
}

pub fn annotate_trial_outcomes(trials: &mut [StudyTrial], completed_folds: usize) {
    let baseline = trials
        .iter()
        .find(|trial| {
            trial.candidate.number == 0
                && trial.measurement.as_ref().is_some_and(|measurement| {
                    measurement.validation_fold_indices.len() >= completed_folds
                })
        })
        .and_then(|trial| trial.measurement.clone());
    let eligible = trials
        .iter()
        .enumerate()
        .filter_map(|(index, trial)| {
            (!matches!(trial.status, TrialStatus::Failed | TrialStatus::Rejected)
                && trial.measurement.as_ref().is_some_and(|measurement| {
                    measurement.validation_fold_indices.len() >= completed_folds
                }))
            .then_some(index)
        })
        .collect::<Vec<_>>();

    for &index in &eligible {
        let trial = &mut trials[index];
        trial.pareto_rank = None;
        trial.hard_constraint_violations.clear();
        if let (Some(measurement), Some(baseline)) = (&trial.measurement, &baseline) {
            if measurement.coverage_gaps > baseline.coverage_gaps {
                trial
                    .hard_constraint_violations
                    .push(StudyConstraintViolation {
                        constraint: "coverage_gaps_vs_baseline".to_string(),
                        observed: measurement.coverage_gaps as f64,
                        maximum: baseline.coverage_gaps as f64,
                    });
            }
            if measurement.failures > baseline.failures {
                trial
                    .hard_constraint_violations
                    .push(StudyConstraintViolation {
                        constraint: "failures_vs_baseline".to_string(),
                        observed: measurement.failures as f64,
                        maximum: baseline.failures as f64,
                    });
            }
        }
    }

    let mut remaining = eligible;
    let mut rank = 0usize;
    while !remaining.is_empty() {
        let front = remaining
            .iter()
            .copied()
            .filter(|&candidate| {
                !remaining.iter().copied().any(|other| {
                    other != candidate && trial_dominates(&trials[other], &trials[candidate])
                })
            })
            .collect::<Vec<_>>();
        if front.is_empty() {
            break;
        }
        for &index in &front {
            trials[index].pareto_rank = Some(rank);
        }
        let front = front.into_iter().collect::<HashSet<_>>();
        remaining.retain(|index| !front.contains(index));
        rank += 1;
    }
}

fn trial_dominates(left: &StudyTrial, right: &StudyTrial) -> bool {
    match left
        .hard_constraint_violations
        .len()
        .cmp(&right.hard_constraint_violations.len())
    {
        Ordering::Less => return true,
        Ordering::Greater => return false,
        Ordering::Equal => {}
    }
    measurement_dominates(
        left.measurement
            .as_ref()
            .expect("eligible trial has a measurement"),
        right
            .measurement
            .as_ref()
            .expect("eligible trial has a measurement"),
    )
}

fn measurement_dominates(left: &StudyMeasurement, right: &StudyMeasurement) -> bool {
    let mut strictly_better = false;
    for ordering in [
        left.coverage_gaps.cmp(&right.coverage_gaps),
        left.failures.cmp(&right.failures),
    ] {
        if ordering == Ordering::Greater {
            return false;
        }
        strictly_better |= ordering == Ordering::Less;
    }
    for (left, right) in [
        (
            left.all_game_penalized_mean_guesses,
            right.all_game_penalized_mean_guesses,
        ),
        (left.average_log_loss, right.average_log_loss),
        (left.average_brier_score, right.average_brier_score),
        (left.latency_p95_ms, right.latency_p95_ms),
        (
            left.peak_memory_bytes.map(|value| value as f64),
            right.peak_memory_bytes.map(|value| value as f64),
        ),
    ] {
        if let (Some(left), Some(right)) = (left, right) {
            let ordering = left.total_cmp(&right);
            if ordering == Ordering::Greater {
                return false;
            }
            strictly_better |= ordering == Ordering::Less;
        }
    }
    strictly_better
}

pub fn successive_halving_survivors(
    trials: &[StudyTrial],
    completed_folds: usize,
    reduction_factor: usize,
) -> HashSet<usize> {
    let mut eligible = trials
        .iter()
        .filter(|trial| {
            !matches!(
                trial.status,
                TrialStatus::Failed | TrialStatus::Rejected | TrialStatus::Pruned
            ) && trial.measurement.as_ref().is_some_and(|measurement| {
                measurement.validation_fold_indices.len() >= completed_folds
            })
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| {
        left.measurement
            .as_ref()
            .expect("eligible trial has a measurement")
            .compare_guarded(
                right
                    .measurement
                    .as_ref()
                    .expect("eligible trial has a measurement"),
            )
            .then_with(|| left.candidate.number.cmp(&right.candidate.number))
    });
    let survivor_count = eligible.len().div_ceil(reduction_factor).max(1);
    let mut survivors = eligible
        .iter()
        .take(survivor_count)
        .map(|trial| trial.candidate.number)
        .collect::<HashSet<_>>();
    if let Some(baseline) = eligible.iter().find(|trial| trial.candidate.number == 0)
        && !survivors.contains(&0)
        && let Some(worst) = eligible
            .iter()
            .take(survivor_count)
            .next_back()
            .map(|trial| trial.candidate.number)
    {
        survivors.remove(&worst);
        survivors.insert(baseline.candidate.number);
    }
    survivors
}

pub fn generate_candidates(
    registry: &ParameterRegistry,
    base: &PriorConfig,
    spec: &StudySpec,
) -> Result<Vec<(StudyCandidate, PriorConfig)>> {
    spec.validate()?;
    registry.validate()?;
    if spec.strategy == StudySearchStrategy::ModelBased {
        bail!("model-based candidates require completed observations from the study runner");
    }
    let dimensions = registry
        .parameters
        .iter()
        .filter(|parameter| parameter.tunable() && spec.stage.includes(parameter.cohort))
        .collect::<Vec<_>>();
    if dimensions.is_empty() {
        bail!("study stage contains no tunable parameters");
    }
    if spec.stage.requires_complete_one_factor_sweep()
        && spec.trial_count < dimensions.len().saturating_add(1)
    {
        bail!(
            "study stage {:?} has {} parameters and requires at least {} trials including the baseline for complete one-factor coverage",
            spec.stage,
            dimensions.len(),
            dimensions.len().saturating_add(1)
        );
    }

    let mut candidates = Vec::with_capacity(spec.trial_count);
    let mut seen_parameters = HashSet::from([String::new()]);
    candidates.push((
        StudyCandidate {
            number: 0,
            seed: mix_seed(spec.seed, 0),
            parameters: BTreeMap::new(),
        },
        base.clone(),
    ));
    if spec.stage.requires_complete_one_factor_sweep() {
        for definition in &dimensions {
            let (parameters, config) = valid_one_factor_candidate(registry, base, definition)?;
            seen_parameters.insert(parameter_set_key(&parameters));
            let number = candidates.len();
            candidates.push((
                StudyCandidate {
                    number,
                    seed: mix_seed(spec.seed, number as u64),
                    parameters,
                },
                config,
            ));
        }
    }
    let mut attempt = 1usize;
    let attempt_limit = spec.trial_count.saturating_mul(200).max(200);
    while candidates.len() < spec.trial_count && attempt <= attempt_limit {
        let parameters = sample_parameters(&dimensions, spec, attempt);
        let parameter_key = parameter_set_key(&parameters);
        if seen_parameters.insert(parameter_key)
            && let Ok(config) = registry.apply_tunable_values(base, &parameters)
        {
            let candidate = StudyCandidate {
                number: candidates.len(),
                seed: mix_seed(spec.seed, attempt as u64),
                parameters,
            };
            candidates.push((candidate, config));
        }
        attempt += 1;
    }
    if candidates.len() != spec.trial_count {
        bail!(
            "could only generate {} valid candidates for {} requested trials",
            candidates.len(),
            spec.trial_count
        );
    }
    Ok(candidates)
}

pub fn generate_model_based_candidate(
    registry: &ParameterRegistry,
    base: &PriorConfig,
    spec: &StudySpec,
    trials: &[StudyTrial],
) -> Result<(StudyCandidate, PriorConfig)> {
    spec.validate()?;
    registry.validate()?;
    if spec.strategy != StudySearchStrategy::ModelBased {
        bail!("observation-driven generation requires the model-based strategy");
    }
    if trials.len() >= spec.trial_count {
        bail!("the requested model-based study already has every candidate");
    }
    let dimensions = registry
        .parameters
        .iter()
        .filter(|parameter| parameter.tunable() && spec.stage.includes(parameter.cohort))
        .collect::<Vec<_>>();
    if dimensions.is_empty() {
        bail!("study stage contains no tunable parameters");
    }
    if spec.stage.requires_complete_one_factor_sweep()
        && spec.trial_count < dimensions.len().saturating_add(1)
    {
        bail!(
            "study stage {:?} has {} parameters and requires at least {} trials including the baseline for complete one-factor coverage",
            spec.stage,
            dimensions.len(),
            dimensions.len().saturating_add(1)
        );
    }
    let mut numbers = trials
        .iter()
        .map(|trial| trial.candidate.number)
        .collect::<Vec<_>>();
    numbers.sort_unstable();
    if numbers.iter().copied().ne(0..numbers.len()) {
        bail!("model-based checkpoint candidate numbers must be contiguous from zero");
    }
    let number = trials.len();
    if number == 0 {
        return Ok((
            StudyCandidate {
                number: 0,
                seed: mix_seed(spec.seed, 0),
                parameters: BTreeMap::new(),
            },
            base.clone(),
        ));
    }
    if spec.stage.requires_complete_one_factor_sweep() && number <= dimensions.len() {
        let (parameters, config) =
            valid_one_factor_candidate(registry, base, dimensions[number - 1])?;
        return Ok((
            StudyCandidate {
                number,
                seed: mix_seed(spec.seed, number as u64),
                parameters,
            },
            config,
        ));
    }
    let seen = trials
        .iter()
        .map(|trial| parameter_set_key(&trial.candidate.parameters))
        .collect::<HashSet<_>>();
    let observations = trials
        .iter()
        .filter(|trial| trial.status == TrialStatus::Complete && trial.measurement.is_some())
        .collect::<Vec<_>>();
    let startup_trials = spec.trial_count.saturating_sub(1).min(8);
    let attempt_limit = spec.trial_count.saturating_mul(200).max(200);
    for attempt in 1..=attempt_limit {
        let parameters = if observations.len() < startup_trials {
            sample_random(
                &dimensions,
                spec.seed ^ number as u64,
                number.saturating_mul(attempt_limit) + attempt,
            )
        } else {
            sample_tpe_parameters(&dimensions, spec.seed, number, attempt, &observations)
        };
        if seen.contains(&parameter_set_key(&parameters)) {
            continue;
        }
        if let Ok(config) = registry.apply_tunable_values(base, &parameters) {
            return Ok((
                StudyCandidate {
                    number,
                    seed: mix_seed(spec.seed, number as u64),
                    parameters,
                },
                config,
            ));
        }
    }
    bail!("could not generate a unique valid model-based candidate")
}

fn valid_one_factor_candidate(
    registry: &ParameterRegistry,
    base: &PriorConfig,
    definition: &ParameterDefinition,
) -> Result<(BTreeMap<String, ParameterValue>, PriorConfig)> {
    const OFFSETS: [f64; 10] = [
        -0.125, 0.125, -0.25, 0.25, -0.375, 0.375, -0.5, 0.5, -1.0, 1.0,
    ];
    let center = value_to_unit(definition, &definition.default).unwrap_or(0.5);
    for offset in OFFSETS {
        let value = sample_value(definition, (center + offset).clamp(0.0, 1.0));
        if parameter_values_equivalent(&value, &definition.default) {
            continue;
        }
        let parameters = BTreeMap::from([(definition.name.clone(), value)]);
        if let Ok(config) = registry.apply_tunable_values(base, &parameters) {
            return Ok((parameters, config));
        }
    }
    bail!(
        "parameter {} has no valid one-factor perturbation around the supplied base config",
        definition.name
    )
}

fn sample_tpe_parameters(
    dimensions: &[&ParameterDefinition],
    seed: u64,
    number: usize,
    attempt: usize,
    observations: &[&StudyTrial],
) -> BTreeMap<String, ParameterValue> {
    let mut ranked = observations.to_vec();
    ranked.sort_by(|left, right| {
        left.measurement
            .as_ref()
            .expect("observation has a measurement")
            .compare_guarded(
                right
                    .measurement
                    .as_ref()
                    .expect("observation has a measurement"),
            )
            .then_with(|| left.candidate.number.cmp(&right.candidate.number))
    });
    let elite_count = ranked.len().div_ceil(4).max(2).min(ranked.len() - 1);
    let (elite, remainder) = ranked.split_at(elite_count);
    let width = (1 + (number.saturating_sub(1) / dimensions.len())).min(4);
    let proposal_count = 64usize;
    (0..proposal_count)
        .map(|proposal| {
            let proposal_seed = mix_seed(
                seed ^ number as u64,
                (attempt.saturating_mul(proposal_count) + proposal) as u64,
            );
            let first = proposal_seed as usize % dimensions.len();
            let mut score = 0.0;
            let parameters = (0..width)
                .map(|offset| {
                    let index = (first + offset * 17) % dimensions.len();
                    let definition = dimensions[index];
                    let center_trial = elite[(proposal_seed as usize + offset) % elite.len()];
                    let center = trial_parameter_unit(center_trial, definition);
                    let left = mix_seed(proposal_seed ^ index as u64, offset as u64);
                    let right = mix_seed(left, (proposal + 1) as u64);
                    let left_unit = (left >> 11) as f64 / ((1u64 << 53) as f64);
                    let right_unit = (right >> 11) as f64 / ((1u64 << 53) as f64);
                    let bandwidth = (0.30 / (elite.len() as f64).sqrt()).clamp(0.05, 0.20);
                    let unit = (center + (left_unit - right_unit) * bandwidth).clamp(0.0, 1.0);
                    let elite_units = elite
                        .iter()
                        .map(|trial| trial_parameter_unit(trial, definition))
                        .collect::<Vec<_>>();
                    let remainder_units = remainder
                        .iter()
                        .map(|trial| trial_parameter_unit(trial, definition))
                        .collect::<Vec<_>>();
                    score += (kernel_density(unit, &elite_units, bandwidth) + 1e-12).ln()
                        - (kernel_density(unit, &remainder_units, bandwidth) + 1e-12).ln();
                    (definition.name.clone(), sample_value(definition, unit))
                })
                .collect::<BTreeMap<_, _>>();
            (score, proposal, parameters)
        })
        .max_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| right.1.cmp(&left.1))
        })
        .expect("TPE proposal count is positive")
        .2
}

fn trial_parameter_unit(trial: &StudyTrial, definition: &ParameterDefinition) -> f64 {
    trial
        .candidate
        .parameters
        .get(&definition.name)
        .and_then(|value| value_to_unit(definition, value))
        .or_else(|| value_to_unit(definition, &definition.default))
        .unwrap_or(0.5)
}

fn kernel_density(unit: f64, samples: &[f64], bandwidth: f64) -> f64 {
    if samples.is_empty() {
        return 1.0;
    }
    samples
        .iter()
        .map(|sample| {
            let standardized = (unit - sample) / bandwidth;
            (-0.5 * standardized * standardized).exp()
        })
        .sum::<f64>()
        / samples.len() as f64
}

fn sample_parameters(
    dimensions: &[&ParameterDefinition],
    spec: &StudySpec,
    attempt: usize,
) -> BTreeMap<String, ParameterValue> {
    match spec.strategy {
        StudySearchStrategy::Grid => sample_grid(dimensions, attempt),
        StudySearchStrategy::LowDiscrepancy => sample_neighborhood(dimensions, attempt),
        StudySearchStrategy::Random => sample_random(dimensions, spec.seed, attempt),
        StudySearchStrategy::LocalRefinement => sample_local_refinement(dimensions, attempt),
        StudySearchStrategy::ModelBased => unreachable!("handled by the observation-driven runner"),
    }
}

fn sample_grid(
    dimensions: &[&ParameterDefinition],
    attempt: usize,
) -> BTreeMap<String, ParameterValue> {
    const LEVELS: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
    let dimension_count = dimensions.len();
    let zero_based = attempt - 1;
    let first = zero_based % dimension_count;
    let level = LEVELS[(zero_based / dimension_count) % LEVELS.len()];
    let width = 1 + (zero_based / (dimension_count * LEVELS.len())).min(2);
    (0..width)
        .map(|offset| {
            let index = (first + offset * 17) % dimension_count;
            let shifted = (level + offset as f64 / LEVELS.len() as f64).fract();
            (
                dimensions[index].name.clone(),
                sample_value(dimensions[index], shifted),
            )
        })
        .collect()
}

fn sample_random(
    dimensions: &[&ParameterDefinition],
    seed: u64,
    attempt: usize,
) -> BTreeMap<String, ParameterValue> {
    let dimension_count = dimensions.len();
    let width = 1 + ((attempt - 1) / dimension_count).min(dimension_count - 1);
    let first = mix_seed(seed, attempt as u64) as usize % dimension_count;
    (0..width)
        .map(|offset| {
            let index = (first + offset * 17) % dimension_count;
            let bits = mix_seed(seed ^ index as u64, (attempt + offset) as u64);
            let unit = (bits >> 11) as f64 / ((1u64 << 53) as f64);
            (
                dimensions[index].name.clone(),
                sample_value(dimensions[index], unit),
            )
        })
        .collect()
}

fn sample_local_refinement(
    dimensions: &[&ParameterDefinition],
    attempt: usize,
) -> BTreeMap<String, ParameterValue> {
    let dimension_count = dimensions.len();
    let zero_based = attempt - 1;
    let index = zero_based % dimension_count;
    let ring = zero_based / dimension_count;
    let magnitude = (0.08 * (1 + ring / 2) as f64).min(0.40);
    let direction = if ring.is_multiple_of(2) { -1.0 } else { 1.0 };
    let center = value_to_unit(dimensions[index], &dimensions[index].default).unwrap_or(0.5);
    let unit = (center + direction * magnitude).clamp(0.0, 1.0);
    BTreeMap::from([(
        dimensions[index].name.clone(),
        sample_value(dimensions[index], unit),
    )])
}

fn value_to_unit(definition: &ParameterDefinition, value: &ParameterValue) -> Option<f64> {
    match (&definition.kind, value) {
        (
            ParameterKind::Float {
                minimum,
                maximum,
                scale,
                ..
            },
            ParameterValue::Float(value),
        ) => Some(match scale {
            ParameterScale::Linear => (value - minimum) / (maximum - minimum),
            ParameterScale::Log => (value.ln() - minimum.ln()) / (maximum.ln() - minimum.ln()),
        }),
        (
            ParameterKind::Integer {
                minimum, maximum, ..
            },
            ParameterValue::Integer(value),
        ) => Some((*value - *minimum) as f64 / (*maximum - *minimum).max(1) as f64),
        (ParameterKind::Categorical { choices }, ParameterValue::Categorical(value)) => choices
            .iter()
            .position(|choice| choice == value)
            .map(|index| (index as f64 + 0.5) / choices.len() as f64),
        _ => None,
    }
    .map(|unit| unit.clamp(0.0, 1.0))
}

fn sample_neighborhood(
    dimensions: &[&ParameterDefinition],
    attempt: usize,
) -> BTreeMap<String, ParameterValue> {
    let dimension_count = dimensions.len();
    let round = (attempt - 1) / dimension_count;
    let first = (attempt - 1) % dimension_count;
    let width = if round < 2 {
        1
    } else {
        (2 + (round - 2) / 2).min(dimension_count)
    };
    (0..width)
        .map(|offset| {
            let index = (first + offset * 17) % dimension_count;
            let definition = dimensions[index];
            let unit = radical_inverse((round + offset + 1) as u64, prime(index));
            (definition.name.clone(), sample_value(definition, unit))
        })
        .collect()
}

fn sample_value(definition: &ParameterDefinition, unit: f64) -> ParameterValue {
    match &definition.kind {
        ParameterKind::Float {
            minimum,
            maximum,
            step,
            scale,
        } => {
            let raw = match scale {
                ParameterScale::Linear => minimum + unit * (maximum - minimum),
                ParameterScale::Log => (minimum.ln() + unit * (maximum.ln() - minimum.ln())).exp(),
            };
            let value = step.map_or(raw, |step| {
                (minimum + ((raw - minimum) / step).round() * step).clamp(*minimum, *maximum)
            });
            ParameterValue::Float(value)
        }
        ParameterKind::Integer {
            minimum,
            maximum,
            step,
        } => {
            let slots = (maximum - minimum) / step;
            let slot = ((unit * (slots + 1) as f64).floor() as i64).min(slots);
            ParameterValue::Integer(minimum + slot * step)
        }
        ParameterKind::Categorical { choices } => {
            let index = ((unit * choices.len() as f64).floor() as usize).min(choices.len() - 1);
            ParameterValue::Categorical(choices[index].clone())
        }
        ParameterKind::FloatMap => ParameterValue::FloatMap,
    }
}

fn compare_optional(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_optional_u64(left: Option<u64>, right: Option<u64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let inverse = 1.0 / base as f64;
    let mut factor = inverse;
    let mut value = 0.0;
    while index > 0 {
        value += (index % base) as f64 * factor;
        index /= base;
        factor *= inverse;
    }
    value
}

fn prime(index: usize) -> u64 {
    const PRIMES: [u64; 63] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
        191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307,
    ];
    PRIMES[index % PRIMES.len()]
}

fn mix_seed(seed: u64, value: u64) -> u64 {
    let mut mixed = seed ^ value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;
    use crate::experiments::{
        DateRange, ParameterDomain, RollingOriginConfig, build_rolling_origin_plan,
        predictive_parameter_registry,
    };

    fn spec(stage: StudyStage, trial_count: usize) -> StudySpec {
        StudySpec {
            name: "test-study".to_string(),
            stage,
            seed: 42,
            trial_count,
            parallelism: 2,
            strategy: StudySearchStrategy::LowDiscrepancy,
            maximum_validation_folds: 12,
            initial_validation_folds: 3,
            reduction_factor: 3,
            fold_selection: StudyFoldSelection::NestedTimeSpread,
            maximum_trial_seconds: 3_600,
            maximum_memory_mb: 4_096,
        }
    }

    fn provenance() -> StudyProvenance {
        StudyProvenance {
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            base_config_toml: "base=true".to_string(),
            registry_format_version: 6,
            registry_fingerprint: crate::identity::digest_bytes_tagged(
                "test-registry",
                b"registry",
            ),
            input_fingerprint: crate::identity::digest_bytes_tagged("test-inputs", b"inputs"),
            operating_system: "test-os".to_string(),
            architecture: "test-arch".to_string(),
            compute_threads: 4,
            code_revision: Some("revision".to_string()),
            code_dirty: Some(false),
            history_snapshot_start: NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
            history_snapshot_end: NaiveDate::from_ymd_opt(2025, 6, 1).expect("date"),
            development_cutoff: NaiveDate::from_ymd_opt(2025, 5, 2).expect("date"),
            top_suggestions: 5,
        }
    }

    #[test]
    fn candidate_generation_is_deterministic_and_stage_scoped() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let first = generate_candidates(&registry, &base, &spec(StudyStage::Calibration, 12))
            .expect("first");
        let second = generate_candidates(&registry, &base, &spec(StudyStage::Calibration, 12))
            .expect("second");
        assert_eq!(
            first
                .iter()
                .map(|(candidate, _)| candidate)
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|(candidate, _)| candidate)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            first
                .iter()
                .map(|(_, config)| toml::to_string(config).expect("serialize"))
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|(_, config)| toml::to_string(config).expect("serialize"))
                .collect::<Vec<_>>()
        );
        assert!(first[0].0.parameters.is_empty());
        assert!(
            first
                .iter()
                .skip(1)
                .all(
                    |(candidate, _)| candidate.parameters.keys().all(|name| matches!(
                        name.as_str(),
                        "base_seed_weight"
                            | "base_history_only_weight"
                            | "cooldown_days"
                            | "cooldown_floor"
                            | "midpoint_days"
                            | "logistic_k"
                            | "fallback_prior_mass"
                            | "fallback_activation_threshold"
                    ))
                )
        );
    }

    #[test]
    fn proxy_ranker_stage_changes_only_registered_proxy_parameters() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let candidates = generate_candidates(&registry, &base, &spec(StudyStage::ProxyRanker, 16))
            .expect("proxy candidates");
        assert!(candidates.iter().skip(1).all(|(candidate, _)| {
            !candidate.parameters.is_empty()
                && candidate.parameters.keys().all(|name| {
                    name.starts_with("proxy_weights.")
                        || name == "proxy_small_state_lower_bound_threshold"
                        || name == "ambiguous_mass_threshold"
                })
        }));
    }

    #[test]
    fn granular_stages_cover_every_tunable_parameter_exactly_once() {
        let registry = predictive_parameter_registry(&PriorConfig::default());
        let granular_stages = [
            StudyStage::Calibration,
            StudyStage::CoverageRecovery,
            StudyStage::ProxyCore,
            StudyStage::ProxyRisk,
            StudyStage::ProxySmallState,
            StudyStage::SearchRouting,
            StudyStage::SearchExact,
            StudyStage::SearchCoverage,
            StudyStage::SearchLookahead,
            StudyStage::SearchPool,
            StudyStage::SearchDanger,
            StudyStage::SearchPenalty,
            StudyStage::BookPolicy,
        ];
        let expected = registry
            .parameters
            .iter()
            .filter(|parameter| parameter.tunable())
            .map(|parameter| parameter.name.as_str())
            .collect::<HashSet<_>>();
        let mut covered = HashSet::new();
        for stage in granular_stages {
            let stage_parameters = registry
                .parameters
                .iter()
                .filter(|parameter| parameter.tunable() && stage.includes(parameter.cohort))
                .collect::<Vec<_>>();
            assert!(
                !stage_parameters.is_empty(),
                "empty granular stage {stage:?}"
            );
            for parameter in stage_parameters {
                assert!(
                    covered.insert(parameter.name.as_str()),
                    "{} appears in more than one granular stage",
                    parameter.name
                );
            }
        }
        assert_eq!(covered, expected);
    }

    #[test]
    fn one_factor_static_sweeps_reach_every_parameter_in_each_granular_stage() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        for stage in [
            StudyStage::Calibration,
            StudyStage::CoverageRecovery,
            StudyStage::ProxyCore,
            StudyStage::ProxyRisk,
            StudyStage::ProxySmallState,
            StudyStage::SearchRouting,
            StudyStage::SearchExact,
            StudyStage::SearchCoverage,
            StudyStage::SearchLookahead,
            StudyStage::SearchPool,
            StudyStage::SearchDanger,
            StudyStage::SearchPenalty,
            StudyStage::BookPolicy,
        ] {
            let expected = registry
                .parameters
                .iter()
                .filter(|parameter| parameter.tunable() && stage.includes(parameter.cohort))
                .map(|parameter| parameter.name.as_str())
                .collect::<HashSet<_>>();
            let candidates = generate_candidates(
                &registry,
                &base,
                &spec(stage, expected.len().saturating_add(1)),
            )
            .unwrap_or_else(|error| panic!("failed to generate {stage:?} sweep: {error:#}"));
            let covered = candidates
                .iter()
                .flat_map(|(candidate, _)| candidate.parameters.keys().map(String::as_str))
                .collect::<HashSet<_>>();
            assert_eq!(
                covered, expected,
                "incomplete one-factor sweep for {stage:?}"
            );
        }
    }

    #[test]
    fn granular_studies_reject_partial_budgets_and_model_based_startup_covers_every_knob() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let too_small = spec(StudyStage::SearchCoverage, 4);
        assert!(
            generate_candidates(&registry, &base, &too_small)
                .expect_err("four knobs plus a baseline require five trials")
                .to_string()
                .contains("requires at least 5 trials")
        );

        let mut study = spec(StudyStage::SearchCoverage, 5);
        study.strategy = StudySearchStrategy::ModelBased;
        let mut trials = Vec::new();
        for number in 0..study.trial_count {
            let (candidate, _) = generate_model_based_candidate(&registry, &base, &study, &trials)
                .expect("model-based coverage candidate");
            assert_eq!(candidate.number, number);
            trials.push(StudyTrial {
                candidate,
                identity: number.to_string(),
                status: TrialStatus::Complete,
                measurement: Some(StudyMeasurement {
                    validation_fold_indices: (0..12).collect(),
                    all_game_penalized_mean_guesses: Some(3.5),
                    ..StudyMeasurement::default()
                }),
                reason: None,
                elapsed_ms: Some(1),
                pareto_rank: None,
                hard_constraint_violations: Vec::new(),
            });
        }
        let covered = trials
            .iter()
            .flat_map(|trial| trial.candidate.parameters.keys().map(String::as_str))
            .collect::<HashSet<_>>();
        assert_eq!(
            covered,
            HashSet::from([
                "second_guess_coverage_min_survivors",
                "second_guess_coverage_max_survivors",
                "second_guess_coverage_pool",
                "second_guess_coverage_child_cap",
            ])
        );
    }

    #[test]
    fn solve_policy_stage_changes_only_registered_search_parameters() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let candidates = generate_candidates(&registry, &base, &spec(StudyStage::SolvePolicy, 16))
            .expect("solve-policy candidates");
        assert!(candidates.iter().skip(1).all(|(candidate, _)| {
            !candidate.parameters.is_empty()
                && candidate.parameters.keys().all(|name| {
                    registry
                        .parameters
                        .iter()
                        .find(|definition| definition.name == *name)
                        .is_some_and(|definition| {
                            definition.domain == ParameterDomain::SearchPolicy
                        })
                })
        }));
    }

    #[test]
    fn joint_stage_excludes_book_parameters_by_design() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let candidates =
            generate_candidates(&registry, &base, &spec(StudyStage::Joint, 24)).expect("joint");
        assert!(candidates.iter().all(|(candidate, _)| {
            candidate.parameters.keys().all(|name| {
                registry
                    .parameters
                    .iter()
                    .find(|definition| definition.name == *name)
                    .is_some_and(|definition| definition.domain != ParameterDomain::BookPolicy)
            })
        }));
    }

    #[test]
    fn every_declared_static_strategy_is_deterministic_and_unique() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        for strategy in [
            StudySearchStrategy::Grid,
            StudySearchStrategy::LowDiscrepancy,
            StudySearchStrategy::Random,
            StudySearchStrategy::LocalRefinement,
        ] {
            let mut study = spec(StudyStage::SolvePolicy, 10);
            study.strategy = strategy;
            let left = generate_candidates(&registry, &base, &study).expect("left");
            let right = generate_candidates(&registry, &base, &study).expect("right");
            assert_eq!(
                left.iter()
                    .map(|(candidate, _)| candidate)
                    .collect::<Vec<_>>(),
                right
                    .iter()
                    .map(|(candidate, _)| candidate)
                    .collect::<Vec<_>>(),
                "strategy {strategy:?}"
            );
            assert_eq!(
                left.iter()
                    .map(|(_, config)| toml::to_string(config).expect("serialize"))
                    .collect::<Vec<_>>(),
                right
                    .iter()
                    .map(|(_, config)| toml::to_string(config).expect("serialize"))
                    .collect::<Vec<_>>(),
                "strategy {strategy:?}"
            );
            let keys = left
                .iter()
                .map(|(candidate, _)| parameter_set_key(&candidate.parameters))
                .collect::<HashSet<_>>();
            assert_eq!(keys.len(), left.len(), "strategy {strategy:?}");
        }
    }

    #[test]
    fn model_based_generation_is_deterministic_observation_driven_and_unique() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let mut study = spec(StudyStage::ProxyRanker, 12);
        study.strategy = StudySearchStrategy::ModelBased;
        study.parallelism = 4;
        let mut trials = Vec::new();
        for number in 0..8 {
            let (candidate, _) = generate_model_based_candidate(&registry, &base, &study, &trials)
                .expect("startup candidate");
            assert_eq!(candidate.number, number);
            trials.push(StudyTrial {
                candidate,
                identity: number.to_string(),
                status: TrialStatus::Complete,
                measurement: Some(StudyMeasurement {
                    validation_fold_indices: (0..12).collect(),
                    all_game_penalized_mean_guesses: Some(3.8 - number as f64 * 0.05),
                    ..StudyMeasurement::default()
                }),
                reason: None,
                elapsed_ms: Some(1),
                pareto_rank: None,
                hard_constraint_violations: Vec::new(),
            });
        }
        let left = generate_model_based_candidate(&registry, &base, &study, &trials)
            .expect("model-based left");
        let right = generate_model_based_candidate(&registry, &base, &study, &trials)
            .expect("model-based right");
        assert_eq!(left.0, right.0);
        assert_eq!(
            toml::to_string(&left.1).expect("left config"),
            toml::to_string(&right.1).expect("right config")
        );
        assert_eq!(left.0.number, 8);
        assert!(!left.0.parameters.is_empty());
        assert!(trials.iter().all(|trial| {
            parameter_set_key(&trial.candidate.parameters) != parameter_set_key(&left.0.parameters)
        }));
    }

    #[test]
    fn trial_identity_binds_strategy_and_provenance() {
        let candidate = StudyCandidate {
            number: 0,
            seed: 7,
            parameters: BTreeMap::new(),
        };
        let base_spec = spec(StudyStage::Calibration, 2);
        let base_provenance = provenance();
        let identity = candidate
            .identity(&base_spec, &base_provenance)
            .expect("identity");

        let mut different_strategy = base_spec.clone();
        different_strategy.strategy = StudySearchStrategy::Random;
        assert_ne!(
            identity,
            candidate
                .identity(&different_strategy, &base_provenance)
                .expect("strategy identity")
        );

        let mut different_inputs = base_provenance;
        different_inputs.input_fingerprint = "different-inputs".to_string();
        assert_ne!(
            identity,
            candidate
                .identity(&base_spec, &different_inputs)
                .expect("provenance identity")
        );
    }

    #[test]
    fn fidelity_schedule_reaches_maximum_without_duplicates() {
        let mut study = spec(StudyStage::SolvePolicy, 10);
        study.initial_validation_folds = 2;
        study.maximum_validation_folds = 12;
        study.reduction_factor = 3;
        assert_eq!(study.fidelity_schedule(), vec![2, 6, 12]);
    }

    #[test]
    fn fidelity_fold_order_is_nested_and_spans_development_time() {
        let mut study = spec(StudyStage::SolvePolicy, 10);
        study.initial_validation_folds = 3;
        study.maximum_validation_folds = 12;
        study.reduction_factor = 2;

        let first = study.fidelity_fold_indices(12, 3).expect("first rung");
        let second = study.fidelity_fold_indices(12, 6).expect("second rung");
        let final_rung = study.fidelity_fold_indices(12, 12).expect("final rung");

        assert_eq!(first, vec![1, 5, 9]);
        assert_eq!(second, vec![1, 5, 9, 3, 7, 11]);
        assert_eq!(final_rung.len(), 12);
        assert!(first.iter().all(|fold| second.contains(fold)));
        assert!(second.iter().all(|fold| final_rung.contains(fold)));
        let mut unique = final_rung.clone();
        unique.sort_unstable();
        assert_eq!(unique, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn successive_halving_is_guarded_deterministic_and_keeps_baseline() {
        let trials = (0..4)
            .map(|number| {
                let candidate = StudyCandidate {
                    number,
                    seed: number as u64,
                    parameters: BTreeMap::new(),
                };
                StudyTrial {
                    identity: number.to_string(),
                    candidate,
                    status: TrialStatus::Running,
                    measurement: Some(StudyMeasurement {
                        validation_fold_indices: vec![0, 1, 2],
                        coverage_gaps: usize::from(number == 3),
                        all_game_penalized_mean_guesses: Some(match number {
                            0 => 3.6,
                            1 => 3.1,
                            2 => 3.2,
                            _ => 2.9,
                        }),
                        ..StudyMeasurement::default()
                    }),
                    reason: None,
                    elapsed_ms: Some(1),
                    pareto_rank: None,
                    hard_constraint_violations: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let survivors = successive_halving_survivors(&trials, 3, 2);
        assert_eq!(survivors, HashSet::from([0, 1]));
    }

    #[test]
    fn guarded_comparison_prioritizes_coverage_and_failures() {
        let lower_mean_with_gap = StudyMeasurement {
            coverage_gaps: 1,
            all_game_penalized_mean_guesses: Some(2.8),
            ..StudyMeasurement::default()
        };
        let complete = StudyMeasurement {
            all_game_penalized_mean_guesses: Some(3.2),
            ..StudyMeasurement::default()
        };
        assert_eq!(
            complete.compare_guarded(&lower_mean_with_gap),
            Ordering::Less
        );

        let measured_memory = StudyMeasurement {
            peak_memory_bytes: Some(100),
            ..StudyMeasurement::default()
        };
        assert_eq!(
            measured_memory.compare_guarded(&StudyMeasurement::default()),
            Ordering::Less
        );
    }

    #[test]
    fn outcome_annotation_stores_constraint_violations_and_pareto_ranks() {
        let measurements = [
            StudyMeasurement {
                validation_fold_indices: vec![0, 1, 2],
                all_game_penalized_mean_guesses: Some(3.5),
                average_log_loss: Some(7.0),
                ..StudyMeasurement::default()
            },
            StudyMeasurement {
                validation_fold_indices: vec![0, 1, 2],
                coverage_gaps: 1,
                all_game_penalized_mean_guesses: Some(3.0),
                average_log_loss: Some(6.0),
                ..StudyMeasurement::default()
            },
            StudyMeasurement {
                validation_fold_indices: vec![0, 1, 2],
                all_game_penalized_mean_guesses: Some(3.2),
                average_log_loss: Some(6.5),
                ..StudyMeasurement::default()
            },
        ];
        let mut trials = measurements
            .into_iter()
            .enumerate()
            .map(|(number, measurement)| StudyTrial {
                candidate: StudyCandidate {
                    number,
                    seed: number as u64,
                    parameters: BTreeMap::new(),
                },
                identity: number.to_string(),
                status: TrialStatus::Running,
                measurement: Some(measurement),
                reason: None,
                elapsed_ms: Some(1),
                pareto_rank: None,
                hard_constraint_violations: Vec::new(),
            })
            .collect::<Vec<_>>();

        annotate_trial_outcomes(&mut trials, 3);

        assert!(trials[0].hard_constraint_violations.is_empty());
        assert_eq!(trials[0].pareto_rank, Some(1));
        assert_eq!(trials[1].hard_constraint_violations.len(), 1);
        assert_eq!(
            trials[1].hard_constraint_violations[0].constraint,
            "coverage_gaps_vs_baseline"
        );
        assert_eq!(trials[1].pareto_rank, Some(2));
        assert!(trials[2].hard_constraint_violations.is_empty());
        assert_eq!(trials[2].pareto_rank, Some(0));
    }

    #[test]
    fn fold_measurements_resume_without_double_counting() {
        let mut aggregate = StudyMeasurement::default();
        let first = StudyMeasurement {
            validation_fold_indices: vec![0],
            scheduled_games: 2,
            solve_metrics_recorded: true,
            solved_games: 2,
            measured_prior_games: 2,
            penalized_guess_sum: 6.0,
            solved_guess_sum: 6.0,
            log_loss_sum: 1.0,
            brier_score_sum: 0.4,
            peak_memory_bytes: Some(100),
            ..StudyMeasurement::default()
        };
        let second = StudyMeasurement {
            validation_fold_indices: vec![1],
            scheduled_games: 2,
            solve_metrics_recorded: true,
            solved_games: 1,
            failures: 1,
            measured_prior_games: 2,
            penalized_guess_sum: 10.0,
            solved_guess_sum: 4.0,
            log_loss_sum: 2.0,
            brier_score_sum: 0.8,
            peak_memory_bytes: Some(140),
            ..StudyMeasurement::default()
        };
        aggregate.merge_fold(&first).expect("first");
        aggregate.merge_fold(&second).expect("second");

        assert_eq!(aggregate.validation_fold_indices, vec![0, 1]);
        assert_eq!(aggregate.scheduled_games, 4);
        assert_eq!(aggregate.solved_games, 3);
        assert_eq!(aggregate.failures, 1);
        assert_eq!(aggregate.all_game_penalized_mean_guesses, Some(4.0));
        assert_eq!(aggregate.conditional_mean_guesses, Some(10.0 / 3.0));
        assert_eq!(aggregate.average_log_loss, Some(0.75));
        assert!((aggregate.average_brier_score.expect("brier") - 0.3).abs() <= f64::EPSILON);
        assert_eq!(aggregate.peak_memory_bytes, Some(140));
        assert!(aggregate.merge_fold(&second).is_err());
    }

    #[test]
    fn calibration_measurement_does_not_report_a_zero_guess_score() {
        let mut calibration = StudyMeasurement {
            scheduled_games: 30,
            measured_prior_games: 27,
            log_loss_sum: 100.0,
            brier_score_sum: 20.0,
            ..StudyMeasurement::default()
        };
        calibration.refresh_derived();
        assert_eq!(calibration.all_game_penalized_mean_guesses, None);
        assert_eq!(calibration.conditional_mean_guesses, None);
        assert!(calibration.average_log_loss.is_some());
    }

    #[test]
    fn resource_budget_is_validated_and_changes_trial_identity() {
        let candidate = StudyCandidate {
            number: 0,
            seed: 7,
            parameters: BTreeMap::new(),
        };
        let baseline = spec(StudyStage::SolvePolicy, 2);
        let mut lower_fidelity = baseline.clone();
        lower_fidelity.maximum_validation_folds = 3;
        assert_ne!(
            candidate
                .identity(&baseline, &provenance())
                .expect("baseline"),
            candidate
                .identity(&lower_fidelity, &provenance())
                .expect("lower fidelity")
        );
        lower_fidelity.maximum_validation_folds = 0;
        assert!(lower_fidelity.validate().is_err());
        let mut zero_time = baseline;
        zero_time.maximum_trial_seconds = 0;
        assert!(zero_time.validate().is_err());

        let mut zero_memory = spec(StudyStage::SolvePolicy, 2);
        zero_memory.maximum_memory_mb = 0;
        assert!(zero_memory.validate().is_err());

        let mut different_parallelism = spec(StudyStage::SolvePolicy, 2);
        let original = candidate
            .identity(&different_parallelism, &provenance())
            .expect("parallel baseline");
        different_parallelism.parallelism += 1;
        assert_ne!(
            original,
            candidate
                .identity(&different_parallelism, &provenance())
                .expect("parallel change")
        );
    }

    #[test]
    fn candidate_equivalence_accepts_json_round_trip_float_drift() {
        let left = StudyCandidate {
            number: 4,
            seed: 7,
            parameters: BTreeMap::from([(
                "epsilon".to_string(),
                ParameterValue::Float(2.682_695_795_279_731e-11),
            )]),
        };
        let right = StudyCandidate {
            parameters: BTreeMap::from([(
                "epsilon".to_string(),
                ParameterValue::Float(2.682_695_795_279_730_8e-11),
            )]),
            ..left.clone()
        };
        assert!(left.equivalent_to(&right));
        assert_eq!(
            left.identity(&spec(StudyStage::Calibration, 5), &provenance())
                .expect("left"),
            right
                .identity(&spec(StudyStage::Calibration, 5), &provenance())
                .expect("right")
        );
    }

    #[test]
    fn state_round_trip_preserves_sealed_boundary_and_identity() {
        let evaluation_plan = build_rolling_origin_plan(
            DateRange::new(
                NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
                NaiveDate::from_ymd_opt(2025, 6, 1).expect("date"),
            )
            .expect("range"),
            RollingOriginConfig::default(),
        )
        .expect("plan");
        let spec = spec(StudyStage::Calibration, 2);
        let provenance = provenance();
        let mut state =
            StudyState::new(spec.clone(), evaluation_plan, provenance.clone()).expect("state");
        let candidate = StudyCandidate {
            number: 0,
            seed: mix_seed(spec.seed, 0),
            parameters: BTreeMap::new(),
        };
        state.trials.push(StudyTrial {
            identity: candidate.identity(&spec, &provenance).expect("identity"),
            candidate,
            status: TrialStatus::Complete,
            measurement: Some(StudyMeasurement::default()),
            reason: None,
            elapsed_ms: Some(1),
            pareto_rank: None,
            hard_constraint_violations: Vec::new(),
        });
        state.validate().expect("valid state");
        assert!(!state.sealed_test_evaluated);
        assert_eq!(state.best_completed().expect("best").candidate.number, 0);
    }
}
