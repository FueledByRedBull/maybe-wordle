use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::config::{PriorConfig, ProxyWeights, SearchPolicyMode};

use super::recovery::RecoveryPolicy;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PredictivePolicy {
    pub policy_id: String,
    pub description: String,
    pub prior: PriorPolicy,
    pub search: SearchPolicy,
    pub proxy: ProxyPolicy,
    pub recovery: RecoveryPolicy,
}

impl Default for PredictivePolicy {
    fn default() -> Self {
        Self::from(&PriorConfig::default())
    }
}

impl From<&PriorConfig> for PredictivePolicy {
    fn from(config: &PriorConfig) -> Self {
        Self {
            policy_id: "predictive-v1".to_string(),
            description: "Predictive solver policy derived from config/prior.toml".to_string(),
            prior: PriorPolicy::from(config),
            search: SearchPolicy::from(config),
            proxy: ProxyPolicy::from(config),
            recovery: config.recovery.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PriorPolicy {
    pub base_seed_weight: f64,
    pub base_history_only_weight: f64,
    pub cooldown_days: i64,
    pub cooldown_floor: f64,
    pub midpoint_days: f64,
    pub logistic_k: f64,
    pub manual_weights: BTreeMap<String, f64>,
}

impl Default for PriorPolicy {
    fn default() -> Self {
        Self::from(&PriorConfig::default())
    }
}

impl From<&PriorConfig> for PriorPolicy {
    fn from(config: &PriorConfig) -> Self {
        Self {
            base_seed_weight: config.base_seed_weight,
            base_history_only_weight: config.base_history_only_weight,
            cooldown_days: config.cooldown_days,
            cooldown_floor: config.cooldown_floor,
            midpoint_days: config.midpoint_days,
            logistic_k: config.logistic_k,
            manual_weights: config.manual_weights.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchPolicy {
    pub mode: SearchPolicyMode,
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
    pub second_guess_coverage_child_cap: usize,
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
}

impl Default for SearchPolicy {
    fn default() -> Self {
        Self::from(&PriorConfig::default())
    }
}

impl From<&PriorConfig> for SearchPolicy {
    fn from(config: &PriorConfig) -> Self {
        Self {
            mode: config.search_policy_mode,
            exact_threshold: config.exact_threshold,
            exact_exhaustive_threshold: config.exact_exhaustive_threshold,
            exact_candidate_pool: config.exact_candidate_pool,
            session_opener_pool: config.session_opener_pool,
            session_opener_holdout_shortlist: config.session_opener_holdout_shortlist,
            session_reply_pool: config.session_reply_pool,
            session_window_days: config.session_window_days,
            session_artifact_freshness_days: config.session_artifact_freshness_days,
            lookahead_threshold: config.lookahead_threshold,
            medium_state_lookahead_threshold: config.medium_state_lookahead_threshold,
            second_guess_coverage_min_survivors: config.second_guess_coverage_min_survivors,
            second_guess_coverage_max_survivors: config.second_guess_coverage_max_survivors,
            second_guess_coverage_pool: config.second_guess_coverage_pool,
            second_guess_coverage_child_cap: config.second_guess_coverage_child_cap,
            lookahead_candidate_pool: config.lookahead_candidate_pool,
            medium_state_lookahead_candidate_pool: config.medium_state_lookahead_candidate_pool,
            lookahead_reply_pool: config.lookahead_reply_pool,
            medium_state_lookahead_reply_pool: config.medium_state_lookahead_reply_pool,
            lookahead_root_force_in_two_scan: config.lookahead_root_force_in_two_scan,
            medium_state_force_in_two_scan: config.medium_state_force_in_two_scan,
            large_state_split_threshold: config.large_state_split_threshold,
            pool_tight_gap_threshold: config.pool_tight_gap_threshold,
            pool_medium_gap_threshold: config.pool_medium_gap_threshold,
            pool_tight_expansion_multiplier: config.pool_tight_expansion_multiplier,
            pool_medium_expansion_multiplier: config.pool_medium_expansion_multiplier,
            pool_diversity_stride: config.pool_diversity_stride,
            exact_pool_primary_fraction: config.exact_pool_primary_fraction,
            exact_pool_entropy_fraction: config.exact_pool_entropy_fraction,
            exact_pool_worst_bucket_fraction: config.exact_pool_worst_bucket_fraction,
            exact_pool_mass_reducer_fraction: config.exact_pool_mass_reducer_fraction,
            exact_pool_solve_probability_fraction: config.exact_pool_solve_probability_fraction,
            exact_pool_posterior_fraction: config.exact_pool_posterior_fraction,
            ambiguous_mass_threshold: config.ambiguous_mass_threshold,
            danger_top_concentration_w: config.danger_top_concentration_w,
            danger_bucket_mass_w: config.danger_bucket_mass_w,
            danger_bucket_ratio_w: config.danger_bucket_ratio_w,
            danger_ambiguous_w: config.danger_ambiguous_w,
            danger_disagreement_w: config.danger_disagreement_w,
            danger_posterior_window: config.danger_posterior_window,
            danger_candidate_window: config.danger_candidate_window,
            danger_mass_disagreement_threshold: config.danger_mass_disagreement_threshold,
            danger_size_disagreement_threshold: config.danger_size_disagreement_threshold,
            danger_ambiguity_saturation_count: config.danger_ambiguity_saturation_count,
            danger_lookahead_threshold: config.danger_lookahead_threshold,
            danger_exact_threshold: config.danger_exact_threshold,
            danger_reply_pool_bonus: config.danger_reply_pool_bonus,
            danger_exact_root_pool: config.danger_exact_root_pool,
            danger_exact_survivor_cap: config.danger_exact_survivor_cap,
            lookahead_trap_penalty: config.lookahead_trap_penalty,
            lookahead_worst_bucket_ratio_penalty: config.lookahead_worst_bucket_ratio_penalty,
            lookahead_large_bucket_penalty: config.lookahead_large_bucket_penalty,
            lookahead_dangerous_mass_penalty: config.lookahead_dangerous_mass_penalty,
            lookahead_large_bucket_mass_penalty: config.lookahead_large_bucket_mass_penalty,
            trap_size_threshold: config.trap_size_threshold,
            trap_mass_threshold: config.trap_mass_threshold,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ProxyPolicy {
    pub weights: ProxyWeights,
    pub small_state_lower_bound_threshold: usize,
}

impl Default for ProxyPolicy {
    fn default() -> Self {
        Self::from(&PriorConfig::default())
    }
}

impl From<&PriorConfig> for ProxyPolicy {
    fn from(config: &PriorConfig) -> Self {
        Self {
            weights: config.proxy_weights.clone(),
            small_state_lower_bound_threshold: config.proxy_small_state_lower_bound_threshold,
        }
    }
}
