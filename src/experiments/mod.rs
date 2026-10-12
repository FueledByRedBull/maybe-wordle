mod diagnostic;
pub mod exhaustive_cost;
mod folds;
mod matrix;
mod metrics;
mod parameter;
mod profile;
mod study;

pub use diagnostic::{HardCaseDiagnosticSpec, default_diagnostic_suite};
#[cfg(test)]
pub use folds::build_rolling_origin_plan;
pub use folds::{
    DateRange, EVALUATION_POLICY_FORMAT_VERSION, EvaluationPlan, EvaluationPolicy,
    RollingOriginConfig, RollingOriginFold, build_declared_rolling_origin_plan,
};
pub use matrix::{ExperimentArtifactMode, PredictiveExperimentMatrix};
pub use metrics::{
    BootstrapConfig, GameOutcome, GameOutcomeStatus, MetricInterval, PairedDifference,
    PredictiveMetrics, PriorEvidenceMetrics, ProbabilityScore, RankedProbabilityObservation,
    score_multiclass_probabilities, summarize_predictive_outcomes,
    summarize_ranked_probability_observations,
};
pub use parameter::{
    ParameterCohort, ParameterDefinition, ParameterDomain, ParameterKind, ParameterRegistry,
    ParameterScale, ParameterValue, predictive_parameter_registry, validate_predictive_config,
    validate_registered_predictive_config,
};
pub use profile::PredictiveConfigProfile;
pub use study::{
    StudyFoldSelection, StudyMeasurement, StudyProvenance, StudySearchStrategy, StudySpec,
    StudyStage, StudyState, StudyTrial, TrialStatus, annotate_trial_outcomes, generate_candidates,
    generate_model_based_candidate, successive_halving_survivors,
};
