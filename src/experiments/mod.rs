mod diagnostic;
mod folds;
mod matrix;
mod metrics;
mod parameter;
mod profile;
mod study;

pub use diagnostic::{
    BookDiagnosticSpec, DIAGNOSTIC_SUITE_FORMAT_VERSION, DiagnosticExperimentSuite,
    HardCaseDiagnosticSpec, LatencyDiagnosticSpec, ThreeGuessDiagnosticSpec,
    default_diagnostic_suite,
};
pub use folds::{
    DateRange, EvaluationPlan, RollingOriginConfig, RollingOriginFold, build_rolling_origin_plan,
};
pub use matrix::{
    EXPERIMENT_MATRIX_FORMAT_VERSION, ExperimentArtifactMode, PredictiveExperimentMatrix,
    PredictiveExperimentProfile,
};
pub use metrics::{
    BootstrapConfig, GameOutcome, GameOutcomeStatus, MetricInterval, PairedDifference,
    PredictiveMetrics, PriorEvidenceMetrics, ProbabilityScore, RankedProbabilityObservation,
    score_multiclass_probabilities, summarize_predictive_outcomes,
    summarize_ranked_probability_observations,
};
pub use parameter::{
    ObjectiveKind, ParameterCohort, ParameterDefinition, ParameterDomain, ParameterKind,
    ParameterRegistry, ParameterRole, ParameterScale, ParameterValue,
    predictive_parameter_registry, validate_predictive_config,
    validate_registered_predictive_config,
};
pub use profile::{PROFILE_FORMAT_VERSION, PredictiveConfigProfile};
pub use study::{
    STUDY_FORMAT_VERSION, StudyCandidate, StudyConstraintViolation, StudyFoldSelection,
    StudyMeasurement, StudyProvenance, StudySearchStrategy, StudySpec, StudyStage, StudyState,
    StudyTrial, TrialStatus, annotate_trial_outcomes, generate_candidates,
    generate_model_based_candidate, successive_halving_survivors,
};
