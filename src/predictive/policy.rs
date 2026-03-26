use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::config::{PriorConfig, ProxyWeights};

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
    pub exact_threshold: usize,
    pub exact_exhaustive_threshold: usize,
    pub exact_candidate_pool: usize,
    pub session_opener_pool: usize,
    pub session_reply_pool: usize,
    pub session_window_days: usize,
    pub lookahead_threshold: usize,
    pub medium_state_lookahead_threshold: usize,
    pub lookahead_candidate_pool: usize,
    pub medium_state_lookahead_candidate_pool: usize,
    pub lookahead_reply_pool: usize,
    pub medium_state_lookahead_reply_pool: usize,
    pub lookahead_root_force_in_two_scan: usize,
    pub medium_state_force_in_two_scan: usize,
    pub large_state_split_threshold: usize,
    pub pool_tight_gap_threshold: f64,
    pub pool_medium_gap_threshold: f64,
    pub pool_diversity_stride: usize,
    pub danger_lookahead_threshold: f64,
    pub danger_exact_threshold: f64,
    pub danger_reply_pool_bonus: usize,
    pub danger_exact_root_pool: usize,
    pub danger_exact_survivor_cap: usize,
    pub lookahead_trap_penalty: f64,
    pub lookahead_large_bucket_penalty: f64,
    pub lookahead_dangerous_mass_penalty: f64,
    pub lookahead_large_bucket_mass_penalty: f64,
    pub trap_size_threshold: usize,
    pub trap_mass_threshold: f64,
    pub sync_reverify_days: i64,
}

impl Default for SearchPolicy {
    fn default() -> Self {
        Self::from(&PriorConfig::default())
    }
}

impl From<&PriorConfig> for SearchPolicy {
    fn from(config: &PriorConfig) -> Self {
        Self {
            exact_threshold: config.exact_threshold,
            exact_exhaustive_threshold: config.exact_exhaustive_threshold,
            exact_candidate_pool: config.exact_candidate_pool,
            session_opener_pool: config.session_opener_pool,
            session_reply_pool: config.session_reply_pool,
            session_window_days: config.session_window_days,
            lookahead_threshold: config.lookahead_threshold,
            medium_state_lookahead_threshold: config.medium_state_lookahead_threshold,
            lookahead_candidate_pool: config.lookahead_candidate_pool,
            medium_state_lookahead_candidate_pool: config.medium_state_lookahead_candidate_pool,
            lookahead_reply_pool: config.lookahead_reply_pool,
            medium_state_lookahead_reply_pool: config.medium_state_lookahead_reply_pool,
            lookahead_root_force_in_two_scan: config.lookahead_root_force_in_two_scan,
            medium_state_force_in_two_scan: config.medium_state_force_in_two_scan,
            large_state_split_threshold: config.large_state_split_threshold,
            pool_tight_gap_threshold: config.pool_tight_gap_threshold,
            pool_medium_gap_threshold: config.pool_medium_gap_threshold,
            pool_diversity_stride: config.pool_diversity_stride,
            danger_lookahead_threshold: config.danger_lookahead_threshold,
            danger_exact_threshold: config.danger_exact_threshold,
            danger_reply_pool_bonus: config.danger_reply_pool_bonus,
            danger_exact_root_pool: config.danger_exact_root_pool,
            danger_exact_survivor_cap: config.danger_exact_survivor_cap,
            lookahead_trap_penalty: config.lookahead_trap_penalty,
            lookahead_large_bucket_penalty: config.lookahead_large_bucket_penalty,
            lookahead_dangerous_mass_penalty: config.lookahead_dangerous_mass_penalty,
            lookahead_large_bucket_mass_penalty: config.lookahead_large_bucket_mass_penalty,
            trap_size_threshold: config.trap_size_threshold,
            trap_mass_threshold: config.trap_mass_threshold,
            sync_reverify_days: config.sync_reverify_days,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ProxyPolicy {
    pub weights: ProxyWeights,
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
        }
    }
}
