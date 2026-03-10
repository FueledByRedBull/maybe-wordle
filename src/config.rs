use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
    pub lookahead_threshold: usize,
    pub lookahead_candidate_pool: usize,
    pub lookahead_reply_pool: usize,
    pub lookahead_root_force_in_two_scan: usize,
    pub danger_lookahead_threshold: f64,
    pub danger_exact_threshold: f64,
    pub danger_reply_pool_bonus: usize,
    pub danger_exact_root_pool: usize,
    pub danger_exact_survivor_cap: usize,
    pub lookahead_trap_penalty: f64,
    pub sync_reverify_days: i64,
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
            lookahead_threshold: 160,
            lookahead_candidate_pool: 24,
            lookahead_reply_pool: 12,
            lookahead_root_force_in_two_scan: 64,
            danger_lookahead_threshold: 0.58,
            danger_exact_threshold: 0.72,
            danger_reply_pool_bonus: 8,
            danger_exact_root_pool: 24,
            danger_exact_survivor_cap: 192,
            lookahead_trap_penalty: 0.35,
            sync_reverify_days: 3,
            manual_weights: BTreeMap::new(),
        }
    }
}

impl PriorConfig {
    pub fn load_or_create(path: &Path) -> Result<Self> {
        if path.exists() {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let mut config: Self = toml::from_str(&raw)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            config.normalize_manual_keys();
            return Ok(config);
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
        let mut config = PriorConfig::default();
        config.exact_exhaustive_threshold = 14;
        config.lookahead_threshold = 144;
        config.lookahead_candidate_pool = 18;
        config.lookahead_reply_pool = 9;
        config.lookahead_root_force_in_two_scan = 72;
        config.danger_lookahead_threshold = 0.61;
        config.danger_exact_threshold = 0.77;
        config.danger_reply_pool_bonus = 6;
        config.danger_exact_root_pool = 28;
        config.danger_exact_survivor_cap = 176;
        config.lookahead_trap_penalty = 0.42;
        let encoded = toml::to_string_pretty(&config).expect("encode");
        assert!(encoded.contains("exact_exhaustive_threshold = 14"));
        assert!(encoded.contains("lookahead_threshold = 144"));
        assert!(encoded.contains("lookahead_candidate_pool = 18"));
        assert!(encoded.contains("lookahead_reply_pool = 9"));
        assert!(encoded.contains("lookahead_root_force_in_two_scan = 72"));
        assert!(encoded.contains("danger_lookahead_threshold = 0.61"));
        assert!(encoded.contains("danger_exact_threshold = 0.77"));
        assert!(encoded.contains("danger_reply_pool_bonus = 6"));
        assert!(encoded.contains("danger_exact_root_pool = 28"));
        assert!(encoded.contains("danger_exact_survivor_cap = 176"));
        assert!(encoded.contains("lookahead_trap_penalty = 0.42"));

        let decoded: PriorConfig = toml::from_str(&encoded).expect("decode");
        assert_eq!(decoded.exact_exhaustive_threshold, 14);
        assert_eq!(decoded.lookahead_threshold, 144);
        assert_eq!(decoded.lookahead_candidate_pool, 18);
        assert_eq!(decoded.lookahead_reply_pool, 9);
        assert_eq!(decoded.lookahead_root_force_in_two_scan, 72);
        assert_eq!(decoded.danger_lookahead_threshold, 0.61);
        assert_eq!(decoded.danger_exact_threshold, 0.77);
        assert_eq!(decoded.danger_reply_pool_bonus, 6);
        assert_eq!(decoded.danger_exact_root_pool, 28);
        assert_eq!(decoded.danger_exact_survivor_cap, 176);
        assert_eq!(decoded.lookahead_trap_penalty, 0.42);
    }
}
