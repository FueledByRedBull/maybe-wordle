use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    atomic_file::atomic_write,
    predictive::{PredictivePolicy, RecoveryPolicy},
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchPolicyMode {
    #[default]
    Staged,
    /// Diagnostic staged ranking on the finite modes' frozen core/tail belief.
    StagedFixedBelief,
    ProxyWithExactEndgame,
    ProxyOnly,
    FiniteFast,
    FiniteFastDynamic,
    FiniteFastFixedWork,
    FiniteStrong,
    FiniteBaseline,
}

impl SearchPolicyMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Staged => "staged",
            Self::StagedFixedBelief => "staged_fixed_belief",
            Self::ProxyWithExactEndgame => "proxy_with_exact_endgame",
            Self::ProxyOnly => "proxy_only",
            Self::FiniteFast => "finite_fast",
            Self::FiniteFastDynamic => "finite_fast_dynamic",
            Self::FiniteFastFixedWork => "finite_fast_fixed_work",
            Self::FiniteStrong => "finite_strong",
            Self::FiniteBaseline => "finite_baseline",
        }
    }

    pub fn is_finite(self) -> bool {
        matches!(
            self,
            Self::FiniteFast
                | Self::FiniteFastDynamic
                | Self::FiniteFastFixedWork
                | Self::FiniteStrong
                | Self::FiniteBaseline
        )
    }

    pub fn uses_fixed_belief(self) -> bool {
        matches!(
            self,
            Self::StagedFixedBelief
                | Self::FiniteFast
                | Self::FiniteFastFixedWork
                | Self::FiniteStrong
                | Self::FiniteBaseline
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ProxyWeights {
    pub entropy_w: f64,
    pub bucket_mass_w: f64,
    pub bucket_size_w: f64,
    pub ambiguous_w: f64,
    pub proxy_w: f64,
    pub solve_prob_w: f64,
    pub posterior_w: f64,
    pub smoothness_w: f64,
    pub gray_reuse_w: f64,
    pub large_bucket_count_w: f64,
    pub dangerous_mass_count_w: f64,
    pub large_bucket_mass_w: f64,
}

impl Default for ProxyWeights {
    fn default() -> Self {
        Self {
            entropy_w: 0.14634824837897314,
            bucket_mass_w: 2.1000,
            bucket_size_w: 0.12,
            ambiguous_w: 0.7500,
            proxy_w: 1.00,
            solve_prob_w: 0.10,
            posterior_w: 0.05,
            smoothness_w: 0.45,
            gray_reuse_w: 0.08,
            large_bucket_count_w: 0.1782,
            dangerous_mass_count_w: 0.22,
            large_bucket_mass_w: 0.40,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PriorConfig {
    pub base_seed_weight: f64,
    pub base_history_only_weight: f64,
    pub cooldown_days: i64,
    pub cooldown_floor: f64,
    pub midpoint_days: f64,
    pub logistic_k: f64,
    pub fallback_prior_mass: f64,
    pub fallback_activation_threshold: usize,
    pub search_policy_mode: SearchPolicyMode,
    pub exact_threshold: usize,
    pub exact_exhaustive_threshold: usize,
    pub exact_candidate_pool: usize,
    pub session_opener_pool: usize,
    pub session_opener_holdout_shortlist: usize,
    pub session_reply_pool: usize,
    pub session_window_days: usize,
    pub session_artifact_freshness_days: usize,
    pub lookahead_threshold: usize,
    pub medium_state_lookahead_threshold: usize,
    pub second_guess_coverage_min_survivors: usize,
    pub second_guess_coverage_max_survivors: usize,
    pub second_guess_coverage_pool: usize,
    pub lookahead_candidate_pool: usize,
    pub medium_state_lookahead_candidate_pool: usize,
    pub lookahead_reply_pool: usize,
    pub medium_state_lookahead_reply_pool: usize,
    pub lookahead_root_force_in_two_scan: usize,
    pub medium_state_force_in_two_scan: usize,
    pub large_state_split_threshold: usize,
    pub pool_tight_gap_threshold: f64,
    pub pool_medium_gap_threshold: f64,
    pub pool_tight_expansion_multiplier: f64,
    pub pool_medium_expansion_multiplier: f64,
    pub pool_diversity_stride: usize,
    pub exact_pool_primary_fraction: f64,
    pub exact_pool_entropy_fraction: f64,
    pub exact_pool_worst_bucket_fraction: f64,
    pub exact_pool_mass_reducer_fraction: f64,
    pub exact_pool_solve_probability_fraction: f64,
    pub exact_pool_posterior_fraction: f64,
    pub ambiguous_mass_threshold: f64,
    pub danger_top_concentration_w: f64,
    pub danger_bucket_mass_w: f64,
    pub danger_bucket_ratio_w: f64,
    pub danger_ambiguous_w: f64,
    pub danger_disagreement_w: f64,
    pub danger_posterior_window: usize,
    pub danger_candidate_window: usize,
    pub danger_mass_disagreement_threshold: f64,
    pub danger_size_disagreement_threshold: usize,
    pub danger_ambiguity_saturation_count: usize,
    pub danger_lookahead_threshold: f64,
    pub danger_exact_threshold: f64,
    pub danger_reply_pool_bonus: usize,
    pub danger_exact_root_pool: usize,
    pub danger_exact_survivor_cap: usize,
    pub lookahead_trap_penalty: f64,
    pub lookahead_worst_bucket_ratio_penalty: f64,
    pub lookahead_large_bucket_penalty: f64,
    pub lookahead_dangerous_mass_penalty: f64,
    pub lookahead_large_bucket_mass_penalty: f64,
    pub trap_size_threshold: usize,
    pub trap_mass_threshold: f64,
    pub sync_reverify_days: i64,
    pub sync_request_timeout_seconds: u64,
    pub sync_retry_attempts: usize,
    pub sync_retry_backoff_millis: u64,
    pub allow_history_gaps: bool,
    pub proxy_weights: ProxyWeights,
    pub proxy_small_state_lower_bound_threshold: usize,
    pub recovery: RecoveryPolicy,
    pub manual_weights: BTreeMap<String, f64>,
}

impl Default for PriorConfig {
    fn default() -> Self {
        Self {
            base_seed_weight: 0.75,
            base_history_only_weight: 0.50,
            cooldown_days: 365,
            cooldown_floor: 0.01,
            midpoint_days: 1680.0,
            logistic_k: 0.030542173853450913,
            fallback_prior_mass: 0.06644939273961506,
            fallback_activation_threshold: 4,
            search_policy_mode: SearchPolicyMode::Staged,
            exact_threshold: 64,
            exact_exhaustive_threshold: 12,
            exact_candidate_pool: 96,
            session_opener_pool: 32,
            session_opener_holdout_shortlist: 4,
            session_reply_pool: 20,
            session_window_days: 30,
            session_artifact_freshness_days: 14,
            lookahead_threshold: 160,
            medium_state_lookahead_threshold: 80,
            second_guess_coverage_min_survivors: 65,
            second_guess_coverage_max_survivors: 80,
            second_guess_coverage_pool: 24,
            lookahead_candidate_pool: 24,
            medium_state_lookahead_candidate_pool: 48,
            lookahead_reply_pool: 12,
            medium_state_lookahead_reply_pool: 20,
            lookahead_root_force_in_two_scan: 64,
            medium_state_force_in_two_scan: 160,
            large_state_split_threshold: 50,
            pool_tight_gap_threshold: 0.05,
            pool_medium_gap_threshold: 0.15,
            pool_tight_expansion_multiplier: 2.5,
            pool_medium_expansion_multiplier: 1.5,
            pool_diversity_stride: 4,
            exact_pool_primary_fraction: 0.5,
            exact_pool_entropy_fraction: 0.25,
            exact_pool_worst_bucket_fraction: 1.0 / 6.0,
            exact_pool_mass_reducer_fraction: 1.0 / 6.0,
            exact_pool_solve_probability_fraction: 0.125,
            exact_pool_posterior_fraction: 0.125,
            ambiguous_mass_threshold: 0.10,
            danger_top_concentration_w: 0.30,
            danger_bucket_mass_w: 0.25,
            danger_bucket_ratio_w: 0.20,
            danger_ambiguous_w: 0.15,
            danger_disagreement_w: 0.10,
            danger_posterior_window: 3,
            danger_candidate_window: 3,
            danger_mass_disagreement_threshold: 0.10,
            danger_size_disagreement_threshold: 2,
            danger_ambiguity_saturation_count: 4,
            danger_lookahead_threshold: 0.58,
            danger_exact_threshold: 0.72,
            danger_reply_pool_bonus: 8,
            danger_exact_root_pool: 24,
            danger_exact_survivor_cap: 192,
            lookahead_trap_penalty: 0.35,
            lookahead_worst_bucket_ratio_penalty: 0.35,
            lookahead_large_bucket_penalty: 0.12,
            lookahead_dangerous_mass_penalty: 0.08,
            lookahead_large_bucket_mass_penalty: 0.10,
            trap_size_threshold: 6,
            trap_mass_threshold: 0.15,
            sync_reverify_days: 3,
            sync_request_timeout_seconds: 10,
            sync_retry_attempts: 3,
            sync_retry_backoff_millis: 500,
            allow_history_gaps: false,
            proxy_weights: ProxyWeights::default(),
            proxy_small_state_lower_bound_threshold: 12,
            recovery: RecoveryPolicy::default(),
            manual_weights: BTreeMap::new(),
        }
    }
}

impl PriorConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut config: Self =
            toml::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))?;
        config.normalize_manual_keys();
        crate::experiments::validate_registered_predictive_config(&config)
            .with_context(|| format!("invalid predictive config in {}", path.display()))?;
        Ok(config)
    }

    pub fn load_or_create(path: &Path) -> Result<Self> {
        if path.exists() {
            return Self::load(path);
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        let config = Self::default();
        let raw = toml::to_string_pretty(&config).context("failed to serialize default config")?;
        fs::write(path, raw).with_context(|| format!("failed to write {}", path.display()))?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        crate::experiments::validate_predictive_config(self)
            .with_context(|| format!("refusing to save invalid config to {}", path.display()))?;
        let raw = toml::to_string_pretty(self).context("failed to serialize prior config")?;
        atomic_write(path, raw.as_bytes())
    }

    pub fn predictive_policy(&self) -> PredictivePolicy {
        PredictivePolicy::from(self)
    }

    fn normalize_manual_keys(&mut self) {
        self.manual_weights = self
            .manual_weights
            .iter()
            .map(|(key, value)| (key.trim().to_ascii_lowercase(), *value))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{PriorConfig, SearchPolicyMode};
    use crate::predictive::RecoveryMode;

    fn shipped_toml() -> toml::Value {
        toml::from_str(include_str!("../config/prior.toml")).expect("parse shipped config")
    }

    fn effective_toml(config: &PriorConfig) -> toml::Value {
        toml::from_str(&toml::to_string(config).expect("serialize config"))
            .expect("parse serialized config")
    }

    #[test]
    fn shipped_default_is_canonical_for_empty_and_new_configs() {
        let shipped = shipped_toml();
        assert_eq!(effective_toml(&PriorConfig::default()), shipped);

        let empty: PriorConfig = toml::from_str("").expect("deserialize empty config");
        assert_eq!(effective_toml(&empty), shipped);

        let path = std::env::temp_dir().join(format!(
            "maybe-wordle-shipped-default-{}-{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let created = PriorConfig::load_or_create(&path).expect("create default config");
        assert_eq!(effective_toml(&created), shipped);
        let written = std::fs::read_to_string(&path).expect("read created config");
        let written: toml::Value = toml::from_str(&written).expect("parse created config");
        assert_eq!(written, shipped);
        std::fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn partial_nested_deserialization_uses_canonical_defaults() {
        let config: PriorConfig = toml::from_str(
            r#"
base_seed_weight = 1.0

[proxy_weights]
entropy_w = 1.0

[recovery]
mode = "strict"
"#,
        )
        .expect("deserialize partial config");
        let defaults = PriorConfig::default();

        assert_eq!(config.base_seed_weight, 1.0);
        assert_eq!(
            config.base_history_only_weight,
            defaults.base_history_only_weight
        );
        assert_eq!(config.midpoint_days, defaults.midpoint_days);
        assert_eq!(config.fallback_prior_mass, defaults.fallback_prior_mass);
        assert_eq!(config.proxy_weights.entropy_w, 1.0);
        assert_eq!(
            config.proxy_weights.bucket_mass_w,
            defaults.proxy_weights.bucket_mass_w
        );
        assert_eq!(config.recovery.mode, RecoveryMode::Strict);
        assert_eq!(
            config.recovery.epsilon_scale,
            defaults.recovery.epsilon_scale
        );
    }

    #[test]
    fn obsolete_coverage_child_cap_is_dropped_without_changing_active_settings() {
        let current = "second_guess_coverage_pool = 17\nbase_seed_weight = 1.25\n";
        let historical = format!("second_guess_coverage_child_cap = 24\n{current}");
        let expected: PriorConfig = toml::from_str(current).expect("current config");
        let migrated: PriorConfig = toml::from_str(&historical).expect("historical config");
        assert_eq!(effective_toml(&migrated), effective_toml(&expected));
        let serialized = toml::to_string(&migrated).expect("serialize migrated config");
        assert!(!serialized.contains("second_guess_coverage_child_cap"));
        let reloaded: PriorConfig = toml::from_str(&serialized).expect("reload migrated config");
        assert_eq!(effective_toml(&reloaded), effective_toml(&expected));
    }

    #[test]
    fn every_serialized_leaf_preserves_nondefault_values_on_roundtrip() {
        fn perturb(value: &mut toml::Value) {
            match value {
                toml::Value::Table(table) => table.iter_mut().for_each(|(_, value)| perturb(value)),
                toml::Value::Array(array) => array.iter_mut().for_each(perturb),
                toml::Value::Integer(value) => *value += 1,
                toml::Value::Float(value) => *value = *value * 1.03125 + 0.125,
                toml::Value::Boolean(value) => *value = !*value,
                toml::Value::String(value) => {
                    *value = match value.as_str() {
                        "staged" => "proxy_only",
                        "uniform_over_support" => "strict",
                        other => {
                            panic!("add a valid alternate value for new categorical leaf: {other}")
                        }
                    }
                    .to_string()
                }
                toml::Value::Datetime(_) => panic!("add a nondefault date fixture"),
            }
        }
        // This checks serialization, independently of cross-field search bounds.
        let mut expected = effective_toml(&PriorConfig::default());
        perturb(&mut expected);
        expected["manual_weights"]
            .as_table_mut()
            .unwrap()
            .insert("cigar".to_string(), toml::Value::Float(0.375));
        let config: PriorConfig = expected.clone().try_into().unwrap();
        let encoded = toml::to_string_pretty(&config).unwrap();
        let decoded: PriorConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(effective_toml(&decoded), expected);
    }

    #[test]
    fn finite_diagnostic_candidates_only_change_search_policy_mode() {
        assert_eq!(
            PriorConfig::default().search_policy_mode,
            SearchPolicyMode::Staged
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let baseline =
            PriorConfig::load(&root.join("config/candidates/september-finite-baseline.toml"))
                .expect("load baseline candidate");
        let fast = PriorConfig::load(&root.join("config/candidates/september-finite-fast.toml"))
            .expect("load fast candidate");
        assert_eq!(
            baseline.search_policy_mode,
            SearchPolicyMode::FiniteBaseline
        );
        assert_eq!(fast.search_policy_mode, SearchPolicyMode::FiniteFast);

        let mut baseline_effective = effective_toml(&baseline);
        let mut fast_effective = effective_toml(&fast);
        baseline_effective
            .as_table_mut()
            .expect("baseline table")
            .remove("search_policy_mode");
        fast_effective
            .as_table_mut()
            .expect("fast table")
            .remove("search_policy_mode");
        assert_eq!(baseline_effective, fast_effective);
    }

    #[test]
    fn prior_config_round_trips_lookahead_fields() {
        let mut config = PriorConfig {
            exact_exhaustive_threshold: 14,
            session_opener_pool: 36,
            session_opener_holdout_shortlist: 6,
            session_reply_pool: 24,
            session_window_days: 45,
            session_artifact_freshness_days: 12,
            lookahead_threshold: 144,
            medium_state_lookahead_threshold: 88,
            second_guess_coverage_min_survivors: 20,
            second_guess_coverage_max_survivors: 72,
            second_guess_coverage_pool: 56,
            proxy_small_state_lower_bound_threshold: 7,
            lookahead_candidate_pool: 18,
            medium_state_lookahead_candidate_pool: 40,
            lookahead_reply_pool: 9,
            medium_state_lookahead_reply_pool: 18,
            lookahead_root_force_in_two_scan: 72,
            medium_state_force_in_two_scan: 144,
            large_state_split_threshold: 48,
            pool_tight_gap_threshold: 0.04,
            pool_medium_gap_threshold: 0.11,
            pool_tight_expansion_multiplier: 2.75,
            pool_medium_expansion_multiplier: 1.75,
            pool_diversity_stride: 6,
            exact_pool_primary_fraction: 0.45,
            exact_pool_entropy_fraction: 0.30,
            exact_pool_worst_bucket_fraction: 0.20,
            exact_pool_mass_reducer_fraction: 0.18,
            exact_pool_solve_probability_fraction: 0.14,
            exact_pool_posterior_fraction: 0.12,
            ambiguous_mass_threshold: 0.12,
            danger_top_concentration_w: 0.28,
            danger_bucket_mass_w: 0.24,
            danger_bucket_ratio_w: 0.22,
            danger_ambiguous_w: 0.16,
            danger_disagreement_w: 0.10,
            danger_posterior_window: 4,
            danger_candidate_window: 5,
            danger_mass_disagreement_threshold: 0.08,
            danger_size_disagreement_threshold: 3,
            danger_ambiguity_saturation_count: 5,
            danger_lookahead_threshold: 0.61,
            danger_exact_threshold: 0.77,
            danger_reply_pool_bonus: 6,
            danger_exact_root_pool: 28,
            danger_exact_survivor_cap: 176,
            lookahead_trap_penalty: 0.42,
            lookahead_worst_bucket_ratio_penalty: 0.31,
            lookahead_large_bucket_penalty: 0.16,
            lookahead_dangerous_mass_penalty: 0.09,
            lookahead_large_bucket_mass_penalty: 0.13,
            trap_size_threshold: 7,
            trap_mass_threshold: 0.18,
            ..PriorConfig::default()
        };
        config.proxy_weights.entropy_w = 1.1;
        let encoded = toml::to_string_pretty(&config).expect("encode");
        assert!(encoded.contains("exact_exhaustive_threshold = 14"));
        assert!(encoded.contains("session_opener_pool = 36"));
        assert!(encoded.contains("session_opener_holdout_shortlist = 6"));
        assert!(encoded.contains("session_reply_pool = 24"));
        assert!(encoded.contains("session_window_days = 45"));
        assert!(encoded.contains("session_artifact_freshness_days = 12"));
        assert!(encoded.contains("lookahead_threshold = 144"));
        assert!(encoded.contains("medium_state_lookahead_threshold = 88"));
        assert!(encoded.contains("second_guess_coverage_min_survivors = 20"));
        assert!(encoded.contains("second_guess_coverage_max_survivors = 72"));
        assert!(encoded.contains("second_guess_coverage_pool = 56"));
        assert!(encoded.contains("proxy_small_state_lower_bound_threshold = 7"));
        assert!(encoded.contains("lookahead_candidate_pool = 18"));
        assert!(encoded.contains("medium_state_lookahead_candidate_pool = 40"));
        assert!(encoded.contains("lookahead_reply_pool = 9"));
        assert!(encoded.contains("medium_state_lookahead_reply_pool = 18"));
        assert!(encoded.contains("lookahead_root_force_in_two_scan = 72"));
        assert!(encoded.contains("medium_state_force_in_two_scan = 144"));
        assert!(encoded.contains("large_state_split_threshold = 48"));
        assert!(encoded.contains("pool_tight_gap_threshold = 0.04"));
        assert!(encoded.contains("pool_medium_gap_threshold = 0.11"));
        assert!(encoded.contains("pool_tight_expansion_multiplier = 2.75"));
        assert!(encoded.contains("pool_medium_expansion_multiplier = 1.75"));
        assert!(encoded.contains("pool_diversity_stride = 6"));
        assert!(encoded.contains("exact_pool_primary_fraction = 0.45"));
        assert!(encoded.contains("exact_pool_entropy_fraction = 0.3"));
        assert!(encoded.contains("exact_pool_worst_bucket_fraction = 0.2"));
        assert!(encoded.contains("exact_pool_mass_reducer_fraction = 0.18"));
        assert!(encoded.contains("exact_pool_solve_probability_fraction = 0.14"));
        assert!(encoded.contains("exact_pool_posterior_fraction = 0.12"));
        assert!(encoded.contains("ambiguous_mass_threshold = 0.12"));
        assert!(encoded.contains("danger_top_concentration_w = 0.28"));
        assert!(encoded.contains("danger_bucket_mass_w = 0.24"));
        assert!(encoded.contains("danger_bucket_ratio_w = 0.22"));
        assert!(encoded.contains("danger_ambiguous_w = 0.16"));
        assert!(encoded.contains("danger_disagreement_w = 0.1"));
        assert!(encoded.contains("danger_posterior_window = 4"));
        assert!(encoded.contains("danger_candidate_window = 5"));
        assert!(encoded.contains("danger_mass_disagreement_threshold = 0.08"));
        assert!(encoded.contains("danger_size_disagreement_threshold = 3"));
        assert!(encoded.contains("danger_ambiguity_saturation_count = 5"));
        assert!(encoded.contains("danger_lookahead_threshold = 0.61"));
        assert!(encoded.contains("danger_exact_threshold = 0.77"));
        assert!(encoded.contains("danger_reply_pool_bonus = 6"));
        assert!(encoded.contains("danger_exact_root_pool = 28"));
        assert!(encoded.contains("danger_exact_survivor_cap = 176"));
        assert!(encoded.contains("lookahead_trap_penalty = 0.42"));
        assert!(encoded.contains("lookahead_worst_bucket_ratio_penalty = 0.31"));
        assert!(encoded.contains("lookahead_large_bucket_penalty = 0.16"));
        assert!(encoded.contains("lookahead_dangerous_mass_penalty = 0.09"));
        assert!(encoded.contains("lookahead_large_bucket_mass_penalty = 0.13"));
        assert!(encoded.contains("trap_size_threshold = 7"));
        assert!(encoded.contains("trap_mass_threshold = 0.18"));
        assert!(encoded.contains("entropy_w = 1.1"));
        assert!(encoded.contains("mode = \"uniform_over_support\""));

        let decoded: PriorConfig = toml::from_str(&encoded).expect("decode");
        assert_eq!(decoded.exact_exhaustive_threshold, 14);
        assert_eq!(decoded.session_opener_pool, 36);
        assert_eq!(decoded.session_opener_holdout_shortlist, 6);
        assert_eq!(decoded.session_reply_pool, 24);
        assert_eq!(decoded.session_window_days, 45);
        assert_eq!(decoded.session_artifact_freshness_days, 12);
        assert_eq!(decoded.lookahead_threshold, 144);
        assert_eq!(decoded.medium_state_lookahead_threshold, 88);
        assert_eq!(decoded.second_guess_coverage_min_survivors, 20);
        assert_eq!(decoded.second_guess_coverage_max_survivors, 72);
        assert_eq!(decoded.second_guess_coverage_pool, 56);
        assert_eq!(decoded.proxy_small_state_lower_bound_threshold, 7);
        assert_eq!(decoded.lookahead_candidate_pool, 18);
        assert_eq!(decoded.medium_state_lookahead_candidate_pool, 40);
        assert_eq!(decoded.lookahead_reply_pool, 9);
        assert_eq!(decoded.medium_state_lookahead_reply_pool, 18);
        assert_eq!(decoded.lookahead_root_force_in_two_scan, 72);
        assert_eq!(decoded.medium_state_force_in_two_scan, 144);
        assert_eq!(decoded.large_state_split_threshold, 48);
        assert_eq!(decoded.pool_tight_gap_threshold, 0.04);
        assert_eq!(decoded.pool_medium_gap_threshold, 0.11);
        assert_eq!(decoded.pool_tight_expansion_multiplier, 2.75);
        assert_eq!(decoded.pool_medium_expansion_multiplier, 1.75);
        assert_eq!(decoded.pool_diversity_stride, 6);
        assert_eq!(decoded.exact_pool_primary_fraction, 0.45);
        assert_eq!(decoded.exact_pool_entropy_fraction, 0.30);
        assert_eq!(decoded.exact_pool_worst_bucket_fraction, 0.20);
        assert_eq!(decoded.exact_pool_mass_reducer_fraction, 0.18);
        assert_eq!(decoded.exact_pool_solve_probability_fraction, 0.14);
        assert_eq!(decoded.exact_pool_posterior_fraction, 0.12);
        assert_eq!(decoded.ambiguous_mass_threshold, 0.12);
        assert_eq!(decoded.danger_top_concentration_w, 0.28);
        assert_eq!(decoded.danger_bucket_mass_w, 0.24);
        assert_eq!(decoded.danger_bucket_ratio_w, 0.22);
        assert_eq!(decoded.danger_ambiguous_w, 0.16);
        assert_eq!(decoded.danger_disagreement_w, 0.10);
        assert_eq!(decoded.danger_posterior_window, 4);
        assert_eq!(decoded.danger_candidate_window, 5);
        assert_eq!(decoded.danger_mass_disagreement_threshold, 0.08);
        assert_eq!(decoded.danger_size_disagreement_threshold, 3);
        assert_eq!(decoded.danger_ambiguity_saturation_count, 5);
        assert_eq!(decoded.danger_lookahead_threshold, 0.61);
        assert_eq!(decoded.danger_exact_threshold, 0.77);
        assert_eq!(decoded.danger_reply_pool_bonus, 6);
        assert_eq!(decoded.danger_exact_root_pool, 28);
        assert_eq!(decoded.danger_exact_survivor_cap, 176);
        assert_eq!(decoded.lookahead_trap_penalty, 0.42);
        assert_eq!(decoded.lookahead_worst_bucket_ratio_penalty, 0.31);
        assert_eq!(decoded.lookahead_large_bucket_penalty, 0.16);
        assert_eq!(decoded.lookahead_dangerous_mass_penalty, 0.09);
        assert_eq!(decoded.lookahead_large_bucket_mass_penalty, 0.13);
        assert_eq!(decoded.trap_size_threshold, 7);
        assert_eq!(decoded.trap_mass_threshold, 0.18);
        assert_eq!(decoded.sync_request_timeout_seconds, 10);
        assert_eq!(decoded.sync_retry_attempts, 3);
        assert_eq!(decoded.sync_retry_backoff_millis, 500);
        assert_eq!(decoded.proxy_weights.entropy_w, 1.1);
        assert_eq!(decoded.recovery.mode.label(), "uniform_over_support");
    }

    #[test]
    fn loading_a_hand_edited_config_enforces_registered_bounds() {
        let root = std::env::temp_dir().join(format!(
            "maybe-wordle-invalid-prior-config-{}.toml",
            std::process::id()
        ));
        let invalid = toml::to_string_pretty(&PriorConfig {
            fallback_prior_mass: -0.1,
            ..PriorConfig::default()
        })
        .expect("encode invalid config");
        std::fs::write(&root, invalid).expect("write invalid config");
        let error = PriorConfig::load(&root).expect_err("invalid config must fail");
        assert!(format!("{error:#}").contains("fallback_prior_mass"));
        std::fs::remove_file(root).expect("remove fixture");
    }
}
