use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
            entropy_w: 1.35,
            bucket_mass_w: 1.40,
            bucket_size_w: 0.12,
            ambiguous_w: 0.30,
            proxy_w: 1.00,
            solve_prob_w: 0.10,
            posterior_w: 0.05,
            smoothness_w: 0.45,
            gray_reuse_w: 0.08,
            large_bucket_count_w: 0.198,
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
    pub proxy_weights: ProxyWeights,
    pub manual_weights: BTreeMap<String, f64>,
}

impl Default for PriorConfig {
    fn default() -> Self {
        Self {
            base_seed_weight: 0.75,
            base_history_only_weight: 0.50,
            cooldown_days: 365,
            cooldown_floor: 0.0,
            midpoint_days: 1080.0,
            logistic_k: 0.02,
            exact_threshold: 64,
            exact_exhaustive_threshold: 12,
            exact_candidate_pool: 96,
            session_opener_pool: 32,
            session_reply_pool: 20,
            session_window_days: 30,
            lookahead_threshold: 160,
            medium_state_lookahead_threshold: 80,
            lookahead_candidate_pool: 24,
            medium_state_lookahead_candidate_pool: 48,
            lookahead_reply_pool: 12,
            medium_state_lookahead_reply_pool: 20,
            lookahead_root_force_in_two_scan: 64,
            medium_state_force_in_two_scan: 160,
            large_state_split_threshold: 50,
            pool_tight_gap_threshold: 0.05,
            pool_medium_gap_threshold: 0.15,
            pool_diversity_stride: 4,
            danger_lookahead_threshold: 0.58,
            danger_exact_threshold: 0.72,
            danger_reply_pool_bonus: 8,
            danger_exact_root_pool: 24,
            danger_exact_survivor_cap: 192,
            lookahead_trap_penalty: 0.35,
            lookahead_large_bucket_penalty: 0.12,
            lookahead_dangerous_mass_penalty: 0.08,
            lookahead_large_bucket_mass_penalty: 0.10,
            trap_size_threshold: 6,
            trap_mass_threshold: 0.15,
            sync_reverify_days: 3,
            proxy_weights: ProxyWeights::default(),
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
    use super::PriorConfig;

    #[test]
    fn prior_config_round_trips_lookahead_fields() {
        let mut config = PriorConfig {
            exact_exhaustive_threshold: 14,
            session_opener_pool: 36,
            session_reply_pool: 24,
            session_window_days: 45,
            lookahead_threshold: 144,
            medium_state_lookahead_threshold: 88,
            lookahead_candidate_pool: 18,
            medium_state_lookahead_candidate_pool: 40,
            lookahead_reply_pool: 9,
            medium_state_lookahead_reply_pool: 18,
            lookahead_root_force_in_two_scan: 72,
            medium_state_force_in_two_scan: 144,
            large_state_split_threshold: 48,
            pool_tight_gap_threshold: 0.04,
            pool_medium_gap_threshold: 0.11,
            pool_diversity_stride: 6,
            danger_lookahead_threshold: 0.61,
            danger_exact_threshold: 0.77,
            danger_reply_pool_bonus: 6,
            danger_exact_root_pool: 28,
            danger_exact_survivor_cap: 176,
            lookahead_trap_penalty: 0.42,
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
        assert!(encoded.contains("session_reply_pool = 24"));
        assert!(encoded.contains("session_window_days = 45"));
        assert!(encoded.contains("lookahead_threshold = 144"));
        assert!(encoded.contains("medium_state_lookahead_threshold = 88"));
        assert!(encoded.contains("lookahead_candidate_pool = 18"));
        assert!(encoded.contains("medium_state_lookahead_candidate_pool = 40"));
        assert!(encoded.contains("lookahead_reply_pool = 9"));
        assert!(encoded.contains("medium_state_lookahead_reply_pool = 18"));
        assert!(encoded.contains("lookahead_root_force_in_two_scan = 72"));
        assert!(encoded.contains("medium_state_force_in_two_scan = 144"));
        assert!(encoded.contains("large_state_split_threshold = 48"));
        assert!(encoded.contains("pool_tight_gap_threshold = 0.04"));
        assert!(encoded.contains("pool_medium_gap_threshold = 0.11"));
        assert!(encoded.contains("pool_diversity_stride = 6"));
        assert!(encoded.contains("danger_lookahead_threshold = 0.61"));
        assert!(encoded.contains("danger_exact_threshold = 0.77"));
        assert!(encoded.contains("danger_reply_pool_bonus = 6"));
        assert!(encoded.contains("danger_exact_root_pool = 28"));
        assert!(encoded.contains("danger_exact_survivor_cap = 176"));
        assert!(encoded.contains("lookahead_trap_penalty = 0.42"));
        assert!(encoded.contains("lookahead_large_bucket_penalty = 0.16"));
        assert!(encoded.contains("lookahead_dangerous_mass_penalty = 0.09"));
        assert!(encoded.contains("lookahead_large_bucket_mass_penalty = 0.13"));
        assert!(encoded.contains("trap_size_threshold = 7"));
        assert!(encoded.contains("trap_mass_threshold = 0.18"));
        assert!(encoded.contains("entropy_w = 1.1"));

        let decoded: PriorConfig = toml::from_str(&encoded).expect("decode");
        assert_eq!(decoded.exact_exhaustive_threshold, 14);
        assert_eq!(decoded.session_opener_pool, 36);
        assert_eq!(decoded.session_reply_pool, 24);
        assert_eq!(decoded.session_window_days, 45);
        assert_eq!(decoded.lookahead_threshold, 144);
        assert_eq!(decoded.medium_state_lookahead_threshold, 88);
        assert_eq!(decoded.lookahead_candidate_pool, 18);
        assert_eq!(decoded.medium_state_lookahead_candidate_pool, 40);
        assert_eq!(decoded.lookahead_reply_pool, 9);
        assert_eq!(decoded.medium_state_lookahead_reply_pool, 18);
        assert_eq!(decoded.lookahead_root_force_in_two_scan, 72);
        assert_eq!(decoded.medium_state_force_in_two_scan, 144);
        assert_eq!(decoded.large_state_split_threshold, 48);
        assert_eq!(decoded.pool_tight_gap_threshold, 0.04);
        assert_eq!(decoded.pool_medium_gap_threshold, 0.11);
        assert_eq!(decoded.pool_diversity_stride, 6);
        assert_eq!(decoded.danger_lookahead_threshold, 0.61);
        assert_eq!(decoded.danger_exact_threshold, 0.77);
        assert_eq!(decoded.danger_reply_pool_bonus, 6);
        assert_eq!(decoded.danger_exact_root_pool, 28);
        assert_eq!(decoded.danger_exact_survivor_cap, 176);
        assert_eq!(decoded.lookahead_trap_penalty, 0.42);
        assert_eq!(decoded.lookahead_large_bucket_penalty, 0.16);
        assert_eq!(decoded.lookahead_dangerous_mass_penalty, 0.09);
        assert_eq!(decoded.lookahead_large_bucket_mass_penalty, 0.13);
        assert_eq!(decoded.trap_size_threshold, 7);
        assert_eq!(decoded.trap_mass_threshold, 0.18);
        assert_eq!(decoded.proxy_weights.entropy_w, 1.1);
    }
}
