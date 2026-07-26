use std::collections::{BTreeMap, HashSet};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::PriorConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterDomain {
    Prior,
    Proxy,
    SearchPolicy,
    BookPolicy,
    Recovery,
    Operational,
    Safety,
    ManualOverride,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterRole {
    Hyperparameter,
    Operational,
    Safety,
    ManualOverride,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterCohort {
    PriorCalibration,
    CoverageRecovery,
    ProxyCore,
    ProxyRisk,
    ProxySmallState,
    SearchRouting,
    SearchExact,
    SearchCoverage,
    SearchLookahead,
    SearchPool,
    SearchDanger,
    SearchPenalty,
    BookPolicy,
    Operational,
    Safety,
    ManualOverride,
}

impl ParameterCohort {
    fn domain(self) -> ParameterDomain {
        match self {
            Self::PriorCalibration => ParameterDomain::Prior,
            Self::CoverageRecovery => ParameterDomain::Recovery,
            Self::ProxyCore | Self::ProxyRisk | Self::ProxySmallState => ParameterDomain::Proxy,
            Self::SearchRouting
            | Self::SearchExact
            | Self::SearchCoverage
            | Self::SearchLookahead
            | Self::SearchPool
            | Self::SearchDanger
            | Self::SearchPenalty => ParameterDomain::SearchPolicy,
            Self::BookPolicy => ParameterDomain::BookPolicy,
            Self::Operational => ParameterDomain::Operational,
            Self::Safety => ParameterDomain::Safety,
            Self::ManualOverride => ParameterDomain::ManualOverride,
        }
    }

    fn role(self) -> ParameterRole {
        match self {
            Self::Operational => ParameterRole::Operational,
            Self::Safety => ParameterRole::Safety,
            Self::ManualOverride => ParameterRole::ManualOverride,
            _ => ParameterRole::Hyperparameter,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveKind {
    Coverage,
    Calibration,
    SolveQuality,
    Latency,
    Memory,
    ArtifactQuality,
    NetworkReliability,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterScale {
    Linear,
    Log,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParameterKind {
    Float {
        minimum: f64,
        maximum: f64,
        step: Option<f64>,
        scale: ParameterScale,
    },
    Integer {
        minimum: i64,
        maximum: i64,
        step: i64,
    },
    Categorical {
        choices: Vec<String>,
    },
    FloatMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    Categorical(String),
    FloatMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterDefinition {
    pub name: String,
    pub domain: ParameterDomain,
    pub cohort: ParameterCohort,
    pub role: ParameterRole,
    pub default: ParameterValue,
    pub kind: ParameterKind,
    pub objectives: Vec<ObjectiveKind>,
    pub description: String,
}

impl ParameterDefinition {
    pub fn tunable(&self) -> bool {
        self.role == ParameterRole::Hyperparameter
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterRegistry {
    pub format_version: u32,
    pub parameters: Vec<ParameterDefinition>,
    pub constraints: Vec<String>,
}

impl ParameterRegistry {
    pub fn validate(&self) -> Result<()> {
        let mut names = HashSet::new();
        for parameter in &self.parameters {
            if parameter.name.trim().is_empty() {
                bail!("parameter names must not be empty");
            }
            if !names.insert(parameter.name.as_str()) {
                bail!("duplicate parameter: {}", parameter.name);
            }
            if parameter.cohort.domain() != parameter.domain {
                bail!(
                    "parameter {} cohort {:?} belongs to domain {:?}, not {:?}",
                    parameter.name,
                    parameter.cohort,
                    parameter.cohort.domain(),
                    parameter.domain
                );
            }
            if parameter.cohort.role() != parameter.role {
                bail!(
                    "parameter {} cohort {:?} requires role {:?}, not {:?}",
                    parameter.name,
                    parameter.cohort,
                    parameter.cohort.role(),
                    parameter.role
                );
            }
            validate_definition(parameter)?;
        }
        if self
            .constraints
            .iter()
            .any(|constraint| constraint.trim().is_empty())
        {
            bail!("parameter constraints must not be empty");
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ParameterDefinition> {
        self.parameters
            .iter()
            .find(|parameter| parameter.name == name)
    }

    pub fn apply_tunable_values(
        &self,
        base: &PriorConfig,
        values: &BTreeMap<String, ParameterValue>,
    ) -> Result<PriorConfig> {
        let mut document = toml::Value::try_from(base.clone())?;
        for (name, value) in values {
            let definition = self
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("unknown parameter: {name}"))?;
            if !definition.tunable() {
                bail!("parameter is not optimizer-controlled: {name}");
            }
            validate_value(definition, value)?;
            set_toml_value(&mut document, name, parameter_to_toml(value)?)?;
        }
        let config: PriorConfig = document.try_into()?;
        validate_predictive_config(&config)?;
        Ok(config)
    }

    pub fn apply_diagnostic_values(
        &self,
        base: &PriorConfig,
        values: &BTreeMap<String, ParameterValue>,
    ) -> Result<PriorConfig> {
        let mut document = toml::Value::try_from(base.clone())?;
        for (name, value) in values {
            let definition = self
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("unknown parameter: {name}"))?;
            if !definition.tunable() {
                bail!("parameter is not diagnostic-profile controlled: {name}");
            }
            let exact_zero_ablation = matches!(
                (&definition.kind, value),
                (ParameterKind::Float { minimum, .. }, ParameterValue::Float(value))
                    if *minimum > 0.0 && *value == 0.0
            );
            if !exact_zero_ablation {
                validate_value(definition, value)?;
            }
            set_toml_value(&mut document, name, parameter_to_toml(value)?)?;
        }
        let config: PriorConfig = document.try_into()?;
        validate_predictive_config(&config)?;
        Ok(config)
    }
}

pub fn validate_predictive_config(config: &PriorConfig) -> Result<()> {
    let finite_non_negative = [
        ("base_seed_weight", config.base_seed_weight),
        ("base_history_only_weight", config.base_history_only_weight),
        ("cooldown_floor", config.cooldown_floor),
        ("midpoint_days", config.midpoint_days),
        ("logistic_k", config.logistic_k),
        ("fallback_prior_mass", config.fallback_prior_mass),
        ("pool_tight_gap_threshold", config.pool_tight_gap_threshold),
        (
            "pool_medium_gap_threshold",
            config.pool_medium_gap_threshold,
        ),
        (
            "pool_tight_expansion_multiplier",
            config.pool_tight_expansion_multiplier,
        ),
        (
            "pool_medium_expansion_multiplier",
            config.pool_medium_expansion_multiplier,
        ),
        (
            "exact_pool_primary_fraction",
            config.exact_pool_primary_fraction,
        ),
        (
            "exact_pool_entropy_fraction",
            config.exact_pool_entropy_fraction,
        ),
        (
            "exact_pool_worst_bucket_fraction",
            config.exact_pool_worst_bucket_fraction,
        ),
        (
            "exact_pool_mass_reducer_fraction",
            config.exact_pool_mass_reducer_fraction,
        ),
        (
            "exact_pool_solve_probability_fraction",
            config.exact_pool_solve_probability_fraction,
        ),
        (
            "exact_pool_posterior_fraction",
            config.exact_pool_posterior_fraction,
        ),
        ("ambiguous_mass_threshold", config.ambiguous_mass_threshold),
        (
            "danger_top_concentration_w",
            config.danger_top_concentration_w,
        ),
        ("danger_bucket_mass_w", config.danger_bucket_mass_w),
        ("danger_bucket_ratio_w", config.danger_bucket_ratio_w),
        ("danger_ambiguous_w", config.danger_ambiguous_w),
        ("danger_disagreement_w", config.danger_disagreement_w),
        (
            "danger_mass_disagreement_threshold",
            config.danger_mass_disagreement_threshold,
        ),
        (
            "danger_lookahead_threshold",
            config.danger_lookahead_threshold,
        ),
        ("danger_exact_threshold", config.danger_exact_threshold),
        ("lookahead_trap_penalty", config.lookahead_trap_penalty),
        (
            "lookahead_worst_bucket_ratio_penalty",
            config.lookahead_worst_bucket_ratio_penalty,
        ),
        (
            "lookahead_large_bucket_penalty",
            config.lookahead_large_bucket_penalty,
        ),
        (
            "lookahead_dangerous_mass_penalty",
            config.lookahead_dangerous_mass_penalty,
        ),
        (
            "lookahead_large_bucket_mass_penalty",
            config.lookahead_large_bucket_mass_penalty,
        ),
        ("trap_mass_threshold", config.trap_mass_threshold),
        ("proxy_weights.entropy_w", config.proxy_weights.entropy_w),
        (
            "proxy_weights.bucket_mass_w",
            config.proxy_weights.bucket_mass_w,
        ),
        (
            "proxy_weights.bucket_size_w",
            config.proxy_weights.bucket_size_w,
        ),
        (
            "proxy_weights.ambiguous_w",
            config.proxy_weights.ambiguous_w,
        ),
        ("proxy_weights.proxy_w", config.proxy_weights.proxy_w),
        (
            "proxy_weights.solve_prob_w",
            config.proxy_weights.solve_prob_w,
        ),
        (
            "proxy_weights.posterior_w",
            config.proxy_weights.posterior_w,
        ),
        (
            "proxy_weights.smoothness_w",
            config.proxy_weights.smoothness_w,
        ),
        (
            "proxy_weights.gray_reuse_w",
            config.proxy_weights.gray_reuse_w,
        ),
        (
            "proxy_weights.large_bucket_count_w",
            config.proxy_weights.large_bucket_count_w,
        ),
        (
            "proxy_weights.dangerous_mass_count_w",
            config.proxy_weights.dangerous_mass_count_w,
        ),
        (
            "proxy_weights.large_bucket_mass_w",
            config.proxy_weights.large_bucket_mass_w,
        ),
        ("recovery.epsilon_scale", config.recovery.epsilon_scale),
    ];
    for (name, value) in finite_non_negative {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} must be finite and non-negative");
        }
    }
    if config.base_seed_weight == 0.0 && config.base_history_only_weight == 0.0 {
        bail!("at least one primary base weight must be positive");
    }
    if config.cooldown_days < 0 {
        bail!("cooldown_days must be non-negative");
    }
    if config.cooldown_floor > 1.0 {
        bail!("cooldown_floor must be <= 1");
    }
    if !config.fallback_prior_mass.is_finite()
        || config.fallback_prior_mass <= 0.0
        || config.fallback_prior_mass >= 1.0
    {
        bail!("fallback_prior_mass must be finite and strictly between zero and one");
    }
    if config.exact_exhaustive_threshold > config.exact_threshold {
        bail!("exact_exhaustive_threshold must be <= exact_threshold");
    }
    for (name, value) in [
        ("exact_threshold", config.exact_threshold),
        (
            "exact_exhaustive_threshold",
            config.exact_exhaustive_threshold,
        ),
        ("exact_candidate_pool", config.exact_candidate_pool),
        ("session_opener_pool", config.session_opener_pool),
        (
            "session_opener_holdout_shortlist",
            config.session_opener_holdout_shortlist,
        ),
        ("session_reply_pool", config.session_reply_pool),
        ("session_window_days", config.session_window_days),
        (
            "session_artifact_freshness_days",
            config.session_artifact_freshness_days,
        ),
        ("lookahead_threshold", config.lookahead_threshold),
        (
            "medium_state_lookahead_threshold",
            config.medium_state_lookahead_threshold,
        ),
        (
            "second_guess_coverage_pool",
            config.second_guess_coverage_pool,
        ),
        (
            "second_guess_coverage_child_cap",
            config.second_guess_coverage_child_cap,
        ),
        ("lookahead_candidate_pool", config.lookahead_candidate_pool),
        (
            "medium_state_lookahead_candidate_pool",
            config.medium_state_lookahead_candidate_pool,
        ),
        ("lookahead_reply_pool", config.lookahead_reply_pool),
        (
            "medium_state_lookahead_reply_pool",
            config.medium_state_lookahead_reply_pool,
        ),
        (
            "lookahead_root_force_in_two_scan",
            config.lookahead_root_force_in_two_scan,
        ),
        (
            "medium_state_force_in_two_scan",
            config.medium_state_force_in_two_scan,
        ),
        (
            "large_state_split_threshold",
            config.large_state_split_threshold,
        ),
        ("pool_diversity_stride", config.pool_diversity_stride),
        ("danger_posterior_window", config.danger_posterior_window),
        ("danger_candidate_window", config.danger_candidate_window),
        (
            "danger_size_disagreement_threshold",
            config.danger_size_disagreement_threshold,
        ),
        (
            "danger_ambiguity_saturation_count",
            config.danger_ambiguity_saturation_count,
        ),
        ("danger_exact_root_pool", config.danger_exact_root_pool),
        (
            "danger_exact_survivor_cap",
            config.danger_exact_survivor_cap,
        ),
        ("trap_size_threshold", config.trap_size_threshold),
    ] {
        if value == 0 {
            bail!("{name} must be positive");
        }
    }
    if config.second_guess_coverage_max_survivors != 0
        && config.second_guess_coverage_min_survivors > config.second_guess_coverage_max_survivors
    {
        bail!(
            "second_guess_coverage_min_survivors must be <= second_guess_coverage_max_survivors unless the maximum is zero (disabled)"
        );
    }
    if config.session_opener_holdout_shortlist > config.session_opener_pool {
        bail!("session_opener_holdout_shortlist must be <= session_opener_pool");
    }
    if !(config.exact_threshold < config.medium_state_lookahead_threshold
        && config.medium_state_lookahead_threshold <= config.lookahead_threshold)
    {
        bail!("expected exact_threshold < medium_state_lookahead_threshold <= lookahead_threshold");
    }
    if config.lookahead_candidate_pool > config.medium_state_lookahead_candidate_pool {
        bail!("lookahead_candidate_pool must be <= medium_state_lookahead_candidate_pool");
    }
    if config.lookahead_reply_pool > config.medium_state_lookahead_reply_pool {
        bail!("lookahead_reply_pool must be <= medium_state_lookahead_reply_pool");
    }
    if config.lookahead_root_force_in_two_scan > config.medium_state_force_in_two_scan {
        bail!("lookahead_root_force_in_two_scan must be <= medium_state_force_in_two_scan");
    }
    if config.pool_tight_gap_threshold > config.pool_medium_gap_threshold {
        bail!("pool_tight_gap_threshold must be <= pool_medium_gap_threshold");
    }
    if config.pool_medium_expansion_multiplier < 1.0
        || config.pool_tight_expansion_multiplier < config.pool_medium_expansion_multiplier
    {
        bail!("expected 1 <= pool_medium_expansion_multiplier <= pool_tight_expansion_multiplier");
    }
    let exact_pool_fractions = [
        config.exact_pool_primary_fraction,
        config.exact_pool_entropy_fraction,
        config.exact_pool_worst_bucket_fraction,
        config.exact_pool_mass_reducer_fraction,
        config.exact_pool_solve_probability_fraction,
        config.exact_pool_posterior_fraction,
    ];
    if exact_pool_fractions.iter().any(|fraction| *fraction > 1.0) {
        bail!("exact candidate-pool fractions must be <= 1");
    }
    if exact_pool_fractions.iter().all(|fraction| *fraction == 0.0) {
        bail!("at least one exact candidate-pool fraction must be positive");
    }
    if config.danger_mass_disagreement_threshold > 1.0
        || config.danger_lookahead_threshold > 1.0
        || config.danger_exact_threshold > 1.0
    {
        bail!("danger thresholds must be <= 1");
    }
    if config.danger_lookahead_threshold > config.danger_exact_threshold {
        bail!("danger_lookahead_threshold must be <= danger_exact_threshold");
    }
    if config.danger_top_concentration_w
        + config.danger_bucket_mass_w
        + config.danger_bucket_ratio_w
        + config.danger_ambiguous_w
        + config.danger_disagreement_w
        <= 0.0
    {
        bail!("at least one danger feature weight must be positive");
    }
    if config.exact_threshold >= config.danger_exact_survivor_cap {
        bail!("exact_threshold must be < danger_exact_survivor_cap");
    }
    if config.recovery.mode == crate::predictive::RecoveryMode::EpsilonRepair
        && (!config.recovery.epsilon_scale.is_finite() || config.recovery.epsilon_scale <= 0.0)
    {
        bail!("recovery.epsilon_scale must be positive in epsilon_repair mode");
    }
    if config.sync_reverify_days < 0 {
        bail!("sync_reverify_days must be non-negative");
    }
    if config.sync_request_timeout_seconds == 0 {
        bail!("sync_request_timeout_seconds must be positive");
    }
    for (word, weight) in &config.manual_weights {
        if word.len() != 5 || !word.bytes().all(|byte| byte.is_ascii_lowercase()) {
            bail!("manual weight key must be a normalized five-letter word: {word}");
        }
        if !weight.is_finite() || *weight < 0.0 {
            bail!("manual weight for {word} must be finite and non-negative");
        }
    }
    Ok(())
}

pub fn validate_registered_predictive_config(config: &PriorConfig) -> Result<()> {
    predictive_parameter_registry(config).validate()?;
    validate_predictive_config(config)
}

fn validate_value(definition: &ParameterDefinition, value: &ParameterValue) -> Result<()> {
    let valid = match (&definition.kind, value) {
        (
            ParameterKind::Float {
                minimum,
                maximum,
                step,
                ..
            },
            ParameterValue::Float(value),
        ) => {
            value.is_finite()
                && value >= minimum
                && value <= maximum
                && step.is_none_or(|step| {
                    let offset = (value - minimum) / step;
                    (offset - offset.round()).abs() <= 1e-8
                })
        }
        (
            ParameterKind::Integer {
                minimum,
                maximum,
                step,
            },
            ParameterValue::Integer(value),
        ) => value >= minimum && value <= maximum && (value - minimum) % step == 0,
        (ParameterKind::Categorical { choices }, ParameterValue::Categorical(value)) => {
            choices.contains(value)
        }
        (ParameterKind::FloatMap, ParameterValue::FloatMap) => true,
        _ => false,
    };
    if !valid {
        bail!("invalid value for {}", definition.name);
    }
    Ok(())
}

fn parameter_to_toml(value: &ParameterValue) -> Result<toml::Value> {
    match value {
        ParameterValue::Float(value) => toml::Value::try_from(*value).map_err(Into::into),
        ParameterValue::Integer(value) => Ok(toml::Value::Integer(*value)),
        ParameterValue::Categorical(value) => Ok(toml::Value::String(value.clone())),
        ParameterValue::FloatMap => bail!("float maps cannot be assigned by the optimizer"),
    }
}

fn set_toml_value(document: &mut toml::Value, path: &str, value: toml::Value) -> Result<()> {
    let mut parts = path.split('.').peekable();
    let mut cursor = document;
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            let table = cursor
                .as_table_mut()
                .ok_or_else(|| anyhow::anyhow!("parameter parent is not a table: {path}"))?;
            table.insert(part.to_string(), value);
            return Ok(());
        }
        cursor = cursor
            .as_table_mut()
            .and_then(|table| table.get_mut(part))
            .ok_or_else(|| anyhow::anyhow!("parameter path does not exist: {path}"))?;
    }
    bail!("parameter path must not be empty")
}

pub fn predictive_parameter_registry(config: &PriorConfig) -> ParameterRegistry {
    let mut parameters = vec![
        float(
            "base_seed_weight",
            ParameterDomain::Prior,
            config.base_seed_weight,
            0.05,
            2.0,
            None,
            ParameterScale::Log,
            &[
                ObjectiveKind::Coverage,
                ObjectiveKind::Calibration,
                ObjectiveKind::SolveQuality,
            ],
            "Base mass assigned to date-supported seed candidates.",
        ),
        float(
            "base_history_only_weight",
            ParameterDomain::Prior,
            config.base_history_only_weight,
            0.01,
            1.0,
            None,
            ParameterScale::Log,
            &[ObjectiveKind::Calibration, ObjectiveKind::SolveQuality],
            "Base mass assigned to words supported only by prior history.",
        ),
        integer(
            "cooldown_days",
            ParameterDomain::Prior,
            config.cooldown_days,
            0,
            730,
            15,
            &[ObjectiveKind::Calibration, ObjectiveKind::SolveQuality],
            "Days after an appearance that remain in the cooldown region.",
        ),
        float(
            "cooldown_floor",
            ParameterDomain::Prior,
            config.cooldown_floor,
            0.0,
            0.25,
            Some(0.005),
            ParameterScale::Linear,
            &[
                ObjectiveKind::Coverage,
                ObjectiveKind::Calibration,
                ObjectiveKind::SolveQuality,
            ],
            "Minimum recency multiplier for a recently used answer.",
        ),
        float(
            "midpoint_days",
            ParameterDomain::Prior,
            config.midpoint_days,
            30.0,
            1_825.0,
            Some(15.0),
            ParameterScale::Linear,
            &[ObjectiveKind::Calibration, ObjectiveKind::SolveQuality],
            "Logistic recovery midpoint after cooldown.",
        ),
        float(
            "logistic_k",
            ParameterDomain::Prior,
            config.logistic_k,
            0.001,
            0.1,
            None,
            ParameterScale::Log,
            &[ObjectiveKind::Calibration, ObjectiveKind::SolveQuality],
            "Logistic recovery slope after cooldown.",
        ),
        float(
            "fallback_prior_mass",
            ParameterDomain::Recovery,
            config.fallback_prior_mass,
            0.0001,
            0.25,
            None,
            ParameterScale::Log,
            &[ObjectiveKind::Coverage, ObjectiveKind::SolveQuality],
            "Prior probability reserved for valid-guess answers outside date-safe primary support.",
        ),
        integer(
            "fallback_activation_threshold",
            ParameterDomain::Recovery,
            config.fallback_activation_threshold as i64,
            0,
            256,
            1,
            &[
                ObjectiveKind::Coverage,
                ObjectiveKind::SolveQuality,
                ObjectiveKind::Latency,
            ],
            "Largest primary state where filtered dormant answer support joins live search; zero activates only after primary contradiction.",
        ),
    ];

    parameters.extend(proxy_parameters(config));
    parameters.extend(search_parameters(config));
    parameters.extend(book_parameters(config));
    parameters.extend(recovery_parameters(config));
    parameters.extend(operational_parameters(config));
    parameters.push(ParameterDefinition {
        name: "manual_weights".to_string(),
        domain: ParameterDomain::ManualOverride,
        cohort: ParameterCohort::ManualOverride,
        role: ParameterRole::ManualOverride,
        default: ParameterValue::FloatMap,
        kind: ParameterKind::FloatMap,
        objectives: vec![ObjectiveKind::Coverage, ObjectiveKind::Calibration],
        description: "Human-curated per-word multipliers; excluded from automatic tuning."
            .to_string(),
    });

    ParameterRegistry {
        format_version: 6,
        parameters,
        constraints: vec![
            "exact_exhaustive_threshold <= exact_threshold".to_string(),
            "session_opener_holdout_shortlist <= session_opener_pool".to_string(),
            "exact_threshold < medium_state_lookahead_threshold <= lookahead_threshold".to_string(),
            "second_guess_coverage_max_survivors = 0 or second_guess_coverage_min_survivors <= second_guess_coverage_max_survivors".to_string(),
            "lookahead_candidate_pool <= medium_state_lookahead_candidate_pool".to_string(),
            "lookahead_reply_pool <= medium_state_lookahead_reply_pool".to_string(),
            "lookahead_root_force_in_two_scan <= medium_state_force_in_two_scan".to_string(),
            "pool_tight_gap_threshold <= pool_medium_gap_threshold".to_string(),
            "1 <= pool_medium_expansion_multiplier <= pool_tight_expansion_multiplier".to_string(),
            "each exact candidate-pool fraction is in [0, 1], with at least one positive"
                .to_string(),
            "danger_lookahead_threshold <= danger_exact_threshold".to_string(),
            "at least one danger feature weight must be positive".to_string(),
            "exact_threshold < danger_exact_survivor_cap".to_string(),
            "recovery.epsilon_scale > 0 when recovery.mode = epsilon_repair".to_string(),
            "operational and safety parameters are never optimizer-controlled".to_string(),
            "manual_weights are excluded from automatic tuning".to_string(),
        ],
    }
}

fn proxy_parameters(config: &PriorConfig) -> Vec<ParameterDefinition> {
    let weights = &config.proxy_weights;
    let mut parameters = [
        (
            "proxy_weights.entropy_w",
            weights.entropy_w,
            0.01,
            8.0,
            "Weighted entropy reward.",
        ),
        (
            "proxy_weights.bucket_mass_w",
            weights.bucket_mass_w,
            0.01,
            8.0,
            "Largest non-green mass penalty.",
        ),
        (
            "proxy_weights.bucket_size_w",
            weights.bucket_size_w,
            0.001,
            2.0,
            "Worst non-green bucket-size penalty.",
        ),
        (
            "proxy_weights.ambiguous_w",
            weights.ambiguous_w,
            0.001,
            4.0,
            "High-mass ambiguous-bucket penalty.",
        ),
        (
            "proxy_weights.proxy_w",
            weights.proxy_w,
            0.01,
            6.0,
            "Proxy continuation-cost penalty.",
        ),
        (
            "proxy_weights.solve_prob_w",
            weights.solve_prob_w,
            0.001,
            2.0,
            "Immediate solve-probability reward.",
        ),
        (
            "proxy_weights.posterior_w",
            weights.posterior_w,
            0.001,
            2.0,
            "Posterior answer-probability reward.",
        ),
        (
            "proxy_weights.smoothness_w",
            weights.smoothness_w,
            0.001,
            4.0,
            "Partition concentration penalty.",
        ),
        (
            "proxy_weights.gray_reuse_w",
            weights.gray_reuse_w,
            0.001,
            2.0,
            "Known-absent letter reuse penalty.",
        ),
        (
            "proxy_weights.large_bucket_count_w",
            weights.large_bucket_count_w,
            0.001,
            2.0,
            "Large-bucket count penalty.",
        ),
        (
            "proxy_weights.dangerous_mass_count_w",
            weights.dangerous_mass_count_w,
            0.001,
            2.0,
            "Dangerous-mass bucket count penalty.",
        ),
        (
            "proxy_weights.large_bucket_mass_w",
            weights.large_bucket_mass_w,
            0.001,
            4.0,
            "Mass contained in large buckets penalty.",
        ),
    ]
    .into_iter()
    .map(|(name, default, minimum, maximum, description)| {
        float(
            name,
            ParameterDomain::Proxy,
            default,
            minimum,
            maximum,
            None,
            ParameterScale::Log,
            &[ObjectiveKind::SolveQuality, ObjectiveKind::Latency],
            description,
        )
    })
    .collect::<Vec<_>>();
    parameters.push(integer(
        "proxy_small_state_lower_bound_threshold",
        ParameterDomain::Proxy,
        config.proxy_small_state_lower_bound_threshold as i64,
        0,
        64,
        1,
        &[ObjectiveKind::SolveQuality, ObjectiveKind::Latency],
        "Largest child bucket where proxy cost uses the small-state lower-bound table; zero uses the analytic floor for every non-singleton.",
    ));
    parameters.push(float(
        "ambiguous_mass_threshold",
        ParameterDomain::Proxy,
        config.ambiguous_mass_threshold,
        0.01,
        0.50,
        Some(0.01),
        ParameterScale::Linear,
        &[ObjectiveKind::SolveQuality, ObjectiveKind::Latency],
        "Minimum posterior bucket mass counted as a high-mass ambiguity.",
    ));
    parameters
}

fn search_parameters(config: &PriorConfig) -> Vec<ParameterDefinition> {
    let objectives = &[
        ObjectiveKind::SolveQuality,
        ObjectiveKind::Latency,
        ObjectiveKind::Memory,
    ];
    vec![
        ParameterDefinition {
            name: "search_policy_mode".to_string(),
            domain: ParameterDomain::SearchPolicy,
            cohort: ParameterCohort::SearchRouting,
            role: ParameterRole::Hyperparameter,
            default: ParameterValue::Categorical(config.search_policy_mode.label().to_string()),
            kind: ParameterKind::Categorical {
                choices: vec![
                    "staged".to_string(),
                    "proxy_with_exact_endgame".to_string(),
                    "proxy_only".to_string(),
                ],
            },
            objectives: objectives.to_vec(),
            description: "Declared allocation policy for proxy, lookahead, and exact search."
                .to_string(),
        },
        integer(
            "exact_threshold",
            ParameterDomain::SearchPolicy,
            config.exact_threshold as i64,
            16,
            256,
            8,
            objectives,
            "Largest state eligible for predictive exact escalation.",
        ),
        integer(
            "exact_exhaustive_threshold",
            ParameterDomain::SearchPolicy,
            config.exact_exhaustive_threshold as i64,
            2,
            32,
            1,
            objectives,
            "Largest state that scans every allowed guess exactly.",
        ),
        integer(
            "exact_candidate_pool",
            ParameterDomain::SearchPolicy,
            config.exact_candidate_pool as i64,
            16,
            320,
            8,
            objectives,
            "Base shortlist for pooled exact search.",
        ),
        integer(
            "second_guess_coverage_min_survivors",
            ParameterDomain::SearchPolicy,
            config.second_guess_coverage_min_survivors as i64,
            0,
            512,
            1,
            objectives,
            "Smallest second-turn state where forced three-solve coverage is evaluated.",
        ),
        integer(
            "second_guess_coverage_max_survivors",
            ParameterDomain::SearchPolicy,
            config.second_guess_coverage_max_survivors as i64,
            0,
            512,
            1,
            objectives,
            "Largest second-turn state where forced three-solve coverage is evaluated; zero disables it.",
        ),
        integer(
            "second_guess_coverage_pool",
            ParameterDomain::SearchPolicy,
            config.second_guess_coverage_pool as i64,
            8,
            256,
            8,
            objectives,
            "Number of proxy-ranked second guesses checked by the three-solve coverage objective.",
        ),
        integer(
            "second_guess_coverage_child_cap",
            ParameterDomain::SearchPolicy,
            config.second_guess_coverage_child_cap as i64,
            4,
            64,
            1,
            objectives,
            "Largest child bucket scanned for a force-in-two continuation.",
        ),
        integer(
            "lookahead_threshold",
            ParameterDomain::SearchPolicy,
            config.lookahead_threshold as i64,
            32,
            512,
            8,
            objectives,
            "Largest state eligible for predictive lookahead.",
        ),
        integer(
            "medium_state_lookahead_threshold",
            ParameterDomain::SearchPolicy,
            config.medium_state_lookahead_threshold as i64,
            24,
            256,
            8,
            objectives,
            "Boundary for the deeper medium-state lookahead profile.",
        ),
        integer(
            "lookahead_candidate_pool",
            ParameterDomain::SearchPolicy,
            config.lookahead_candidate_pool as i64,
            4,
            128,
            4,
            objectives,
            "Base root lookahead candidate count.",
        ),
        integer(
            "medium_state_lookahead_candidate_pool",
            ParameterDomain::SearchPolicy,
            config.medium_state_lookahead_candidate_pool as i64,
            4,
            192,
            4,
            objectives,
            "Root candidate count for medium states.",
        ),
        integer(
            "lookahead_reply_pool",
            ParameterDomain::SearchPolicy,
            config.lookahead_reply_pool as i64,
            2,
            96,
            2,
            objectives,
            "Base reply count in approximate lookahead.",
        ),
        integer(
            "medium_state_lookahead_reply_pool",
            ParameterDomain::SearchPolicy,
            config.medium_state_lookahead_reply_pool as i64,
            2,
            128,
            2,
            objectives,
            "Reply count for medium states.",
        ),
        integer(
            "lookahead_root_force_in_two_scan",
            ParameterDomain::SearchPolicy,
            config.lookahead_root_force_in_two_scan as i64,
            8,
            512,
            8,
            objectives,
            "Root force-in-two scan budget.",
        ),
        integer(
            "medium_state_force_in_two_scan",
            ParameterDomain::SearchPolicy,
            config.medium_state_force_in_two_scan as i64,
            8,
            768,
            8,
            objectives,
            "Medium-state force-in-two scan budget.",
        ),
        integer(
            "large_state_split_threshold",
            ParameterDomain::SearchPolicy,
            config.large_state_split_threshold as i64,
            8,
            160,
            2,
            objectives,
            "State size where ranking emphasizes splitting over solve probability.",
        ),
        float(
            "pool_tight_gap_threshold",
            ParameterDomain::SearchPolicy,
            config.pool_tight_gap_threshold,
            0.0,
            0.25,
            Some(0.005),
            ParameterScale::Linear,
            objectives,
            "Score gap that triggers maximum pool expansion.",
        ),
        float(
            "pool_medium_gap_threshold",
            ParameterDomain::SearchPolicy,
            config.pool_medium_gap_threshold,
            0.0,
            0.5,
            Some(0.005),
            ParameterScale::Linear,
            objectives,
            "Score gap that triggers moderate pool expansion.",
        ),
        float(
            "pool_tight_expansion_multiplier",
            ParameterDomain::SearchPolicy,
            config.pool_tight_expansion_multiplier,
            1.0,
            4.0,
            Some(0.05),
            ParameterScale::Linear,
            objectives,
            "Multiplier applied to the root candidate pool when ranking scores are tightly clustered.",
        ),
        float(
            "pool_medium_expansion_multiplier",
            ParameterDomain::SearchPolicy,
            config.pool_medium_expansion_multiplier,
            1.0,
            3.0,
            Some(0.05),
            ParameterScale::Linear,
            objectives,
            "Multiplier applied to the root candidate pool for a moderately small score gap.",
        ),
        integer(
            "pool_diversity_stride",
            ParameterDomain::SearchPolicy,
            config.pool_diversity_stride as i64,
            1,
            16,
            1,
            objectives,
            "Stride used to diversify candidate rankings.",
        ),
        float(
            "exact_pool_primary_fraction",
            ParameterDomain::SearchPolicy,
            config.exact_pool_primary_fraction,
            0.0,
            1.0,
            Some(0.025),
            ParameterScale::Linear,
            objectives,
            "Fraction of the exact-search pool drawn from the primary proxy ranking.",
        ),
        float(
            "exact_pool_entropy_fraction",
            ParameterDomain::SearchPolicy,
            config.exact_pool_entropy_fraction,
            0.0,
            1.0,
            Some(0.025),
            ParameterScale::Linear,
            objectives,
            "Fraction of the exact-search pool drawn from entropy ranking.",
        ),
        float(
            "exact_pool_worst_bucket_fraction",
            ParameterDomain::SearchPolicy,
            config.exact_pool_worst_bucket_fraction,
            0.0,
            1.0,
            Some(0.025),
            ParameterScale::Linear,
            objectives,
            "Fraction of the exact-search pool drawn from smallest worst-bucket ranking.",
        ),
        float(
            "exact_pool_mass_reducer_fraction",
            ParameterDomain::SearchPolicy,
            config.exact_pool_mass_reducer_fraction,
            0.0,
            1.0,
            Some(0.025),
            ParameterScale::Linear,
            objectives,
            "Fraction of the exact-search pool drawn from smallest worst-mass ranking.",
        ),
        float(
            "exact_pool_solve_probability_fraction",
            ParameterDomain::SearchPolicy,
            config.exact_pool_solve_probability_fraction,
            0.0,
            1.0,
            Some(0.025),
            ParameterScale::Linear,
            objectives,
            "Fraction of the exact-search pool drawn from immediate solve probability.",
        ),
        float(
            "exact_pool_posterior_fraction",
            ParameterDomain::SearchPolicy,
            config.exact_pool_posterior_fraction,
            0.0,
            1.0,
            Some(0.025),
            ParameterScale::Linear,
            objectives,
            "Fraction of the exact-search pool drawn from posterior-answer probability.",
        ),
        float(
            "danger_top_concentration_w",
            ParameterDomain::SearchPolicy,
            config.danger_top_concentration_w,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Weight of top-three posterior concentration in the normalized danger score.",
        ),
        float(
            "danger_bucket_mass_w",
            ParameterDomain::SearchPolicy,
            config.danger_bucket_mass_w,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Weight of the largest non-green bucket mass in the normalized danger score.",
        ),
        float(
            "danger_bucket_ratio_w",
            ParameterDomain::SearchPolicy,
            config.danger_bucket_ratio_w,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Weight of the largest non-green bucket size ratio in the normalized danger score.",
        ),
        float(
            "danger_ambiguous_w",
            ParameterDomain::SearchPolicy,
            config.danger_ambiguous_w,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Weight of high-mass ambiguous-bucket pressure in the normalized danger score.",
        ),
        float(
            "danger_disagreement_w",
            ParameterDomain::SearchPolicy,
            config.danger_disagreement_w,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Weight of top-ranked candidate disagreement in the normalized danger score.",
        ),
        integer(
            "danger_posterior_window",
            ParameterDomain::SearchPolicy,
            config.danger_posterior_window as i64,
            1,
            12,
            1,
            objectives,
            "Number of highest posterior answer masses included in concentration.",
        ),
        integer(
            "danger_candidate_window",
            ParameterDomain::SearchPolicy,
            config.danger_candidate_window as i64,
            1,
            12,
            1,
            objectives,
            "Number of top-ranked guesses inspected for disagreement.",
        ),
        float(
            "danger_mass_disagreement_threshold",
            ParameterDomain::SearchPolicy,
            config.danger_mass_disagreement_threshold,
            0.0,
            0.5,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Largest-bucket posterior-mass gap classified as candidate disagreement.",
        ),
        integer(
            "danger_size_disagreement_threshold",
            ParameterDomain::SearchPolicy,
            config.danger_size_disagreement_threshold as i64,
            1,
            20,
            1,
            objectives,
            "Largest-bucket candidate-count gap classified as candidate disagreement.",
        ),
        integer(
            "danger_ambiguity_saturation_count",
            ParameterDomain::SearchPolicy,
            config.danger_ambiguity_saturation_count as i64,
            1,
            20,
            1,
            objectives,
            "Ambiguous-bucket count at which normalized ambiguity pressure reaches one.",
        ),
        float(
            "danger_lookahead_threshold",
            ParameterDomain::SearchPolicy,
            config.danger_lookahead_threshold,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Danger score that expands lookahead.",
        ),
        float(
            "danger_exact_threshold",
            ParameterDomain::SearchPolicy,
            config.danger_exact_threshold,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Danger score that escalates to pooled exact search.",
        ),
        integer(
            "danger_reply_pool_bonus",
            ParameterDomain::SearchPolicy,
            config.danger_reply_pool_bonus as i64,
            0,
            64,
            2,
            objectives,
            "Extra replies considered in dangerous states.",
        ),
        integer(
            "danger_exact_root_pool",
            ParameterDomain::SearchPolicy,
            config.danger_exact_root_pool as i64,
            4,
            128,
            4,
            objectives,
            "Minimum exact root pool in dangerous states.",
        ),
        integer(
            "danger_exact_survivor_cap",
            ParameterDomain::SearchPolicy,
            config.danger_exact_survivor_cap as i64,
            17,
            512,
            1,
            objectives,
            "Largest dangerous state eligible for exact escalation.",
        ),
        float(
            "lookahead_trap_penalty",
            ParameterDomain::SearchPolicy,
            config.lookahead_trap_penalty,
            0.0,
            2.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Largest non-green posterior-branch mass penalty.",
        ),
        float(
            "lookahead_worst_bucket_ratio_penalty",
            ParameterDomain::SearchPolicy,
            config.lookahead_worst_bucket_ratio_penalty,
            0.0,
            2.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Largest non-green candidate-count ratio penalty for approximate replies.",
        ),
        float(
            "lookahead_large_bucket_penalty",
            ParameterDomain::SearchPolicy,
            config.lookahead_large_bucket_penalty,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Large-bucket count penalty.",
        ),
        float(
            "lookahead_dangerous_mass_penalty",
            ParameterDomain::SearchPolicy,
            config.lookahead_dangerous_mass_penalty,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Dangerous-mass bucket penalty.",
        ),
        float(
            "lookahead_large_bucket_mass_penalty",
            ParameterDomain::SearchPolicy,
            config.lookahead_large_bucket_mass_penalty,
            0.0,
            1.0,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Compounded large-bucket mass penalty.",
        ),
        integer(
            "trap_size_threshold",
            ParameterDomain::SearchPolicy,
            config.trap_size_threshold as i64,
            2,
            20,
            1,
            objectives,
            "Bucket size classified as a trap.",
        ),
        float(
            "trap_mass_threshold",
            ParameterDomain::SearchPolicy,
            config.trap_mass_threshold,
            0.01,
            0.75,
            Some(0.01),
            ParameterScale::Linear,
            objectives,
            "Bucket mass classified as dangerous.",
        ),
    ]
}

fn book_parameters(config: &PriorConfig) -> Vec<ParameterDefinition> {
    let objectives = &[
        ObjectiveKind::ArtifactQuality,
        ObjectiveKind::SolveQuality,
        ObjectiveKind::Latency,
    ];
    vec![
        integer(
            "session_opener_pool",
            ParameterDomain::BookPolicy,
            config.session_opener_pool as i64,
            4,
            128,
            4,
            objectives,
            "Candidate openers evaluated for an artifact.",
        ),
        integer(
            "session_opener_holdout_shortlist",
            ParameterDomain::BookPolicy,
            config.session_opener_holdout_shortlist as i64,
            1,
            32,
            1,
            objectives,
            "Best primary-window openers re-evaluated on the holdout window.",
        ),
        integer(
            "session_reply_pool",
            ParameterDomain::BookPolicy,
            config.session_reply_pool as i64,
            4,
            128,
            4,
            objectives,
            "Candidate replies evaluated for an artifact branch.",
        ),
        integer(
            "session_window_days",
            ParameterDomain::BookPolicy,
            config.session_window_days as i64,
            7,
            180,
            1,
            objectives,
            "Historical target window used for book selection.",
        ),
        integer(
            "session_artifact_freshness_days",
            ParameterDomain::BookPolicy,
            config.session_artifact_freshness_days as i64,
            1,
            60,
            1,
            objectives,
            "Maximum artifact reuse age and rebuild cadence in days.",
        ),
    ]
}

fn recovery_parameters(config: &PriorConfig) -> Vec<ParameterDefinition> {
    vec![
        ParameterDefinition {
            name: "recovery.mode".to_string(),
            domain: ParameterDomain::Recovery,
            cohort: ParameterCohort::CoverageRecovery,
            role: ParameterRole::Hyperparameter,
            default: ParameterValue::Categorical(config.recovery.mode.label().to_string()),
            kind: ParameterKind::Categorical {
                choices: vec![
                    "strict".to_string(),
                    "uniform_over_support".to_string(),
                    "epsilon_repair".to_string(),
                ],
            },
            objectives: vec![ObjectiveKind::Coverage, ObjectiveKind::Calibration],
            description: "Declared response to a supported state with zero modeled mass."
                .to_string(),
        },
        float(
            "recovery.epsilon_scale",
            ParameterDomain::Recovery,
            config.recovery.epsilon_scale,
            1e-12,
            1e-2,
            None,
            ParameterScale::Log,
            &[ObjectiveKind::Coverage, ObjectiveKind::Calibration],
            "Fallback scale used by epsilon repair.",
        ),
    ]
}

fn operational_parameters(config: &PriorConfig) -> Vec<ParameterDefinition> {
    vec![
        non_tunable_integer(
            "sync_reverify_days",
            ParameterDomain::Operational,
            ParameterRole::Operational,
            config.sync_reverify_days,
            0,
            365,
            "Recent dates rechecked during sync.",
        ),
        non_tunable_integer(
            "sync_request_timeout_seconds",
            ParameterDomain::Operational,
            ParameterRole::Operational,
            config.sync_request_timeout_seconds as i64,
            1,
            300,
            "Per-request network timeout.",
        ),
        non_tunable_integer(
            "sync_retry_attempts",
            ParameterDomain::Operational,
            ParameterRole::Operational,
            config.sync_retry_attempts as i64,
            0,
            20,
            "Network retries after the first attempt.",
        ),
        non_tunable_integer(
            "sync_retry_backoff_millis",
            ParameterDomain::Operational,
            ParameterRole::Operational,
            config.sync_retry_backoff_millis as i64,
            0,
            60_000,
            "Initial network retry backoff.",
        ),
        ParameterDefinition {
            name: "allow_history_gaps".to_string(),
            domain: ParameterDomain::Safety,
            cohort: ParameterCohort::Safety,
            role: ParameterRole::Safety,
            default: ParameterValue::Categorical(config.allow_history_gaps.to_string()),
            kind: ParameterKind::Categorical {
                choices: vec!["false".to_string(), "true".to_string()],
            },
            objectives: vec![ObjectiveKind::Coverage],
            description: "Explicit retrospective override for non-contiguous history; never tuned."
                .to_string(),
        },
    ]
}

fn parameter_cohort(name: &str, domain: ParameterDomain, role: ParameterRole) -> ParameterCohort {
    match (domain, role) {
        (ParameterDomain::Prior, ParameterRole::Hyperparameter) => {
            ParameterCohort::PriorCalibration
        }
        (ParameterDomain::Recovery, ParameterRole::Hyperparameter) => {
            ParameterCohort::CoverageRecovery
        }
        (ParameterDomain::BookPolicy, ParameterRole::Hyperparameter) => ParameterCohort::BookPolicy,
        (ParameterDomain::Operational, ParameterRole::Operational) => ParameterCohort::Operational,
        (ParameterDomain::Safety, ParameterRole::Safety) => ParameterCohort::Safety,
        (ParameterDomain::ManualOverride, ParameterRole::ManualOverride) => {
            ParameterCohort::ManualOverride
        }
        (ParameterDomain::Proxy, ParameterRole::Hyperparameter) => match name {
            "proxy_weights.entropy_w"
            | "proxy_weights.bucket_mass_w"
            | "proxy_weights.bucket_size_w"
            | "proxy_weights.ambiguous_w"
            | "proxy_weights.proxy_w"
            | "proxy_weights.solve_prob_w"
            | "proxy_weights.posterior_w"
            | "proxy_weights.smoothness_w"
            | "proxy_weights.gray_reuse_w" => ParameterCohort::ProxyCore,
            "proxy_weights.large_bucket_count_w"
            | "proxy_weights.dangerous_mass_count_w"
            | "proxy_weights.large_bucket_mass_w"
            | "ambiguous_mass_threshold" => ParameterCohort::ProxyRisk,
            "proxy_small_state_lower_bound_threshold" => ParameterCohort::ProxySmallState,
            _ => panic!("proxy parameter {name} has no typed cohort"),
        },
        (ParameterDomain::SearchPolicy, ParameterRole::Hyperparameter) => match name {
            "search_policy_mode" | "large_state_split_threshold" => ParameterCohort::SearchRouting,
            "exact_threshold" | "exact_exhaustive_threshold" | "exact_candidate_pool" => {
                ParameterCohort::SearchExact
            }
            "second_guess_coverage_min_survivors"
            | "second_guess_coverage_max_survivors"
            | "second_guess_coverage_pool"
            | "second_guess_coverage_child_cap" => ParameterCohort::SearchCoverage,
            "lookahead_threshold"
            | "medium_state_lookahead_threshold"
            | "lookahead_candidate_pool"
            | "medium_state_lookahead_candidate_pool"
            | "lookahead_reply_pool"
            | "medium_state_lookahead_reply_pool"
            | "lookahead_root_force_in_two_scan"
            | "medium_state_force_in_two_scan" => ParameterCohort::SearchLookahead,
            "pool_tight_gap_threshold"
            | "pool_medium_gap_threshold"
            | "pool_tight_expansion_multiplier"
            | "pool_medium_expansion_multiplier"
            | "pool_diversity_stride"
            | "exact_pool_primary_fraction"
            | "exact_pool_entropy_fraction"
            | "exact_pool_worst_bucket_fraction"
            | "exact_pool_mass_reducer_fraction"
            | "exact_pool_solve_probability_fraction"
            | "exact_pool_posterior_fraction" => ParameterCohort::SearchPool,
            "danger_lookahead_threshold"
            | "danger_exact_threshold"
            | "danger_top_concentration_w"
            | "danger_bucket_mass_w"
            | "danger_bucket_ratio_w"
            | "danger_ambiguous_w"
            | "danger_disagreement_w"
            | "danger_posterior_window"
            | "danger_candidate_window"
            | "danger_mass_disagreement_threshold"
            | "danger_size_disagreement_threshold"
            | "danger_ambiguity_saturation_count"
            | "danger_reply_pool_bonus"
            | "danger_exact_root_pool"
            | "danger_exact_survivor_cap" => ParameterCohort::SearchDanger,
            "lookahead_trap_penalty"
            | "lookahead_worst_bucket_ratio_penalty"
            | "lookahead_large_bucket_penalty"
            | "lookahead_dangerous_mass_penalty"
            | "lookahead_large_bucket_mass_penalty"
            | "trap_size_threshold"
            | "trap_mass_threshold" => ParameterCohort::SearchPenalty,
            _ => panic!("search-policy parameter {name} has no typed cohort"),
        },
        _ => panic!("parameter {name} has unsupported domain/role pair {domain:?}/{role:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn float(
    name: &str,
    domain: ParameterDomain,
    default: f64,
    minimum: f64,
    maximum: f64,
    step: Option<f64>,
    scale: ParameterScale,
    objectives: &[ObjectiveKind],
    description: &str,
) -> ParameterDefinition {
    ParameterDefinition {
        name: name.to_string(),
        domain,
        cohort: parameter_cohort(name, domain, ParameterRole::Hyperparameter),
        role: ParameterRole::Hyperparameter,
        default: ParameterValue::Float(default),
        kind: ParameterKind::Float {
            minimum,
            maximum,
            step,
            scale,
        },
        objectives: objectives.to_vec(),
        description: description.to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn integer(
    name: &str,
    domain: ParameterDomain,
    default: i64,
    minimum: i64,
    maximum: i64,
    step: i64,
    objectives: &[ObjectiveKind],
    description: &str,
) -> ParameterDefinition {
    ParameterDefinition {
        name: name.to_string(),
        domain,
        cohort: parameter_cohort(name, domain, ParameterRole::Hyperparameter),
        role: ParameterRole::Hyperparameter,
        default: ParameterValue::Integer(default),
        kind: ParameterKind::Integer {
            minimum,
            maximum,
            step,
        },
        objectives: objectives.to_vec(),
        description: description.to_string(),
    }
}

fn non_tunable_integer(
    name: &str,
    domain: ParameterDomain,
    role: ParameterRole,
    default: i64,
    minimum: i64,
    maximum: i64,
    description: &str,
) -> ParameterDefinition {
    ParameterDefinition {
        name: name.to_string(),
        domain,
        cohort: parameter_cohort(name, domain, role),
        role,
        default: ParameterValue::Integer(default),
        kind: ParameterKind::Integer {
            minimum,
            maximum,
            step: 1,
        },
        objectives: vec![ObjectiveKind::NetworkReliability],
        description: description.to_string(),
    }
}

fn validate_definition(parameter: &ParameterDefinition) -> Result<()> {
    match (&parameter.kind, &parameter.default) {
        (
            ParameterKind::Float {
                minimum,
                maximum,
                step,
                ..
            },
            ParameterValue::Float(default),
        ) => {
            if !minimum.is_finite()
                || !maximum.is_finite()
                || !default.is_finite()
                || minimum > maximum
                || default < minimum
                || default > maximum
                || step.is_some_and(|step| !step.is_finite() || step <= 0.0)
            {
                bail!("invalid float definition for {}", parameter.name);
            }
        }
        (
            ParameterKind::Integer {
                minimum,
                maximum,
                step,
            },
            ParameterValue::Integer(default),
        ) if minimum <= maximum && default >= minimum && default <= maximum && *step > 0 => {}
        (ParameterKind::Categorical { choices }, ParameterValue::Categorical(default))
            if !choices.is_empty() && choices.contains(default) => {}
        (ParameterKind::FloatMap, ParameterValue::FloatMap) => {}
        _ => bail!("kind/default mismatch for {}", parameter.name),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn collect_leaf_paths(prefix: &str, value: &toml::Value, paths: &mut BTreeSet<String>) {
        if let toml::Value::Table(table) = value {
            if table.is_empty() && !prefix.is_empty() {
                paths.insert(prefix.to_string());
            } else {
                for (name, child) in table {
                    let child_prefix = if prefix.is_empty() {
                        name.clone()
                    } else {
                        format!("{prefix}.{name}")
                    };
                    collect_leaf_paths(&child_prefix, child, paths);
                }
            }
        } else {
            paths.insert(prefix.to_string());
        }
    }

    #[test]
    fn predictive_registry_is_unique_complete_and_valid() {
        let config = PriorConfig::default();
        let registry = predictive_parameter_registry(&config);
        registry.validate().expect("registry");
        assert_eq!(registry.format_version, 6);

        let config_value = toml::Value::try_from(config).expect("serialize config");
        let mut config_paths = BTreeSet::new();
        collect_leaf_paths("", &config_value, &mut config_paths);
        let registry_paths = registry
            .parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(registry_paths, config_paths);

        let mut cohort_counts = BTreeMap::new();
        for parameter in &registry.parameters {
            *cohort_counts.entry(parameter.cohort).or_insert(0usize) += 1;
        }
        assert_eq!(
            cohort_counts,
            BTreeMap::from([
                (ParameterCohort::PriorCalibration, 6),
                (ParameterCohort::CoverageRecovery, 4),
                (ParameterCohort::ProxyCore, 9),
                (ParameterCohort::ProxyRisk, 4),
                (ParameterCohort::ProxySmallState, 1),
                (ParameterCohort::SearchRouting, 2),
                (ParameterCohort::SearchExact, 3),
                (ParameterCohort::SearchCoverage, 4),
                (ParameterCohort::SearchLookahead, 8),
                (ParameterCohort::SearchPool, 11),
                (ParameterCohort::SearchDanger, 15),
                (ParameterCohort::SearchPenalty, 7),
                (ParameterCohort::BookPolicy, 5),
                (ParameterCohort::Operational, 4),
                (ParameterCohort::Safety, 1),
                (ParameterCohort::ManualOverride, 1),
            ])
        );

        assert!(registry.get("exact_threshold").expect("search").tunable());
        assert!(
            !registry
                .get("sync_reverify_days")
                .expect("operational")
                .tunable()
        );
        assert!(
            !registry
                .get("allow_history_gaps")
                .expect("safety")
                .tunable()
        );
    }

    #[test]
    fn every_registered_parameter_changes_serialized_config_identity() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let base_serialized = toml::to_string(&base).expect("serialize base");
        let base_identity = crate::identity::digest_bytes_tagged(
            "test-predictive-config",
            base_serialized.as_bytes(),
        );

        for definition in &registry.parameters {
            let alternate = match (&definition.kind, &definition.default) {
                (
                    ParameterKind::Float {
                        minimum, maximum, ..
                    },
                    ParameterValue::Float(default),
                ) => ParameterValue::Float(if default.total_cmp(minimum).is_eq() {
                    *maximum
                } else {
                    *minimum
                }),
                (
                    ParameterKind::Integer {
                        minimum, maximum, ..
                    },
                    ParameterValue::Integer(default),
                ) => ParameterValue::Integer(if default == minimum {
                    *maximum
                } else {
                    *minimum
                }),
                (ParameterKind::Categorical { choices }, ParameterValue::Categorical(default)) => {
                    ParameterValue::Categorical(
                        choices
                            .iter()
                            .find(|choice| *choice != default)
                            .expect("categorical parameter needs an alternate")
                            .clone(),
                    )
                }
                (ParameterKind::FloatMap, ParameterValue::FloatMap) => ParameterValue::FloatMap,
                _ => panic!("registry kind/default mismatch for {}", definition.name),
            };

            let mut document = toml::Value::try_from(base.clone()).expect("serialize document");
            let alternate_toml = if definition.name == "allow_history_gaps" {
                toml::Value::Boolean(!base.allow_history_gaps)
            } else if matches!(definition.kind, ParameterKind::FloatMap) {
                toml::Value::Table(toml::map::Map::from_iter([(
                    "cigar".to_string(),
                    toml::Value::Float(1.0),
                )]))
            } else {
                parameter_to_toml(&alternate).expect("alternate TOML value")
            };
            set_toml_value(&mut document, &definition.name, alternate_toml)
                .expect("set alternate parameter");
            let candidate: PriorConfig = document
                .try_into()
                .unwrap_or_else(|error| panic!("deserialize {}: {error}", definition.name));
            let serialized = toml::to_string(&candidate).expect("serialize candidate");
            let identity = crate::identity::digest_bytes_tagged(
                "test-predictive-config",
                serialized.as_bytes(),
            );
            assert_ne!(
                serialized, base_serialized,
                "{} did not change serialized config",
                definition.name
            );
            assert_ne!(
                identity, base_identity,
                "{} did not change config identity",
                definition.name
            );
        }
    }

    #[test]
    fn registry_applies_nested_and_scalar_tunable_values() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let values = BTreeMap::from([
            ("base_seed_weight".to_string(), ParameterValue::Float(1.0)),
            (
                "proxy_weights.entropy_w".to_string(),
                ParameterValue::Float(2.0),
            ),
            (
                "recovery.mode".to_string(),
                ParameterValue::Categorical("strict".to_string()),
            ),
        ]);

        let candidate = registry
            .apply_tunable_values(&base, &values)
            .expect("candidate");
        assert_eq!(candidate.base_seed_weight, 1.0);
        assert_eq!(candidate.proxy_weights.entropy_w, 2.0);
        assert_eq!(candidate.recovery.mode.label(), "strict");
    }

    #[test]
    fn registry_rejects_operational_and_cross_field_changes() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let operational = BTreeMap::from([(
            "sync_reverify_days".to_string(),
            ParameterValue::Integer(30),
        )]);
        assert!(
            registry
                .apply_tunable_values(&base, &operational)
                .expect_err("operational values are not tunable")
                .to_string()
                .contains("not optimizer-controlled")
        );

        let invalid = BTreeMap::from([(
            "exact_exhaustive_threshold".to_string(),
            ParameterValue::Integer(32),
        )]);
        let mut constrained_base = base;
        constrained_base.exact_threshold = 24;
        assert!(
            registry
                .apply_tunable_values(&constrained_base, &invalid)
                .expect_err("cross-field constraint")
                .to_string()
                .contains("exact_exhaustive_threshold")
        );
    }

    #[test]
    fn diagnostic_values_allow_exact_zero_but_not_out_of_range_nonzero_weights() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let zero = BTreeMap::from([(
            "proxy_weights.bucket_mass_w".to_string(),
            ParameterValue::Float(0.0),
        )]);
        assert_eq!(
            registry
                .apply_diagnostic_values(&base, &zero)
                .expect("zero ablation")
                .proxy_weights
                .bucket_mass_w,
            0.0
        );

        let negative = BTreeMap::from([(
            "proxy_weights.bucket_mass_w".to_string(),
            ParameterValue::Float(-0.1),
        )]);
        assert!(registry.apply_diagnostic_values(&base, &negative).is_err());
    }
}
