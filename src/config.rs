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
    pub sync_reverify_days: i64,
    pub manual_weights: BTreeMap<String, f64>,
}

impl Default for PriorConfig {
    fn default() -> Self {
        Self {
            base_seed_weight: 1.0,
            base_history_only_weight: 0.25,
            cooldown_days: 180,
            cooldown_floor: 0.01,
            midpoint_days: 720.0,
            logistic_k: 0.01,
            exact_threshold: 64,
            exact_exhaustive_threshold: 12,
            exact_candidate_pool: 96,
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
