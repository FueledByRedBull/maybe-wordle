use std::{
    collections::{BTreeMap, HashSet},
    path::{Component, Path},
};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    config::PriorConfig,
    model::{ModelVariant, WeightMode},
};

use super::{ParameterRegistry, ParameterValue};

pub const EXPERIMENT_MATRIX_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentArtifactMode {
    Disabled,
    DiskOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictiveExperimentProfile {
    pub id: String,
    pub description: String,
    pub weight_mode: WeightMode,
    pub model_variant: ModelVariant,
    pub artifact_mode: ExperimentArtifactMode,
    #[serde(default)]
    pub base_config_path: Option<String>,
    #[serde(default)]
    pub parameters: BTreeMap<String, ParameterValue>,
}

impl PredictiveExperimentProfile {
    pub fn apply(&self, registry: &ParameterRegistry, base: &PriorConfig) -> Result<PriorConfig> {
        registry.apply_diagnostic_values(base, &self.parameters)
    }

    pub fn load_base_config(&self, root: &Path, fallback: &PriorConfig) -> Result<PriorConfig> {
        let Some(relative) = self.base_config_path.as_deref() else {
            return Ok(fallback.clone());
        };
        let path = Path::new(relative);
        if path.is_absolute()
            || path
                .components()
                .any(|component| component == Component::ParentDir)
        {
            bail!("experiment base config path must stay inside the repository: {relative}");
        }
        PriorConfig::load(&root.join(path))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictiveExperimentMatrix {
    pub format_version: u32,
    pub name: String,
    pub profiles: Vec<PredictiveExperimentProfile>,
}

impl PredictiveExperimentMatrix {
    pub fn parse_json(source: &str) -> Result<Self> {
        let matrix: Self = serde_json::from_str(source)?;
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format_version != EXPERIMENT_MATRIX_FORMAT_VERSION {
            bail!(
                "unsupported experiment matrix format {}; expected {}",
                self.format_version,
                EXPERIMENT_MATRIX_FORMAT_VERSION
            );
        }
        if self.name.trim().is_empty() || self.profiles.is_empty() {
            bail!("experiment matrix name and profiles must not be empty");
        }
        let mut ids = HashSet::new();
        for profile in &self.profiles {
            if profile.id.trim().is_empty() || profile.description.trim().is_empty() {
                bail!("experiment profile id and description must not be empty");
            }
            if !ids.insert(profile.id.as_str()) {
                bail!("duplicate experiment profile id: {}", profile.id);
            }
            if profile
                .base_config_path
                .as_deref()
                .is_some_and(str::is_empty)
            {
                bail!("experiment base config path must not be empty");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::predictive_parameter_registry;

    #[test]
    fn shipped_development_matrix_is_unique_and_registry_validated() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let matrix = PredictiveExperimentMatrix::parse_json(include_str!(
            "../../config/experiments/development-evidence.json"
        ))
        .expect("matrix");
        assert_eq!(matrix.profiles.len(), 7);
        for profile in &matrix.profiles {
            let profile_base = profile
                .load_base_config(Path::new(env!("CARGO_MANIFEST_DIR")), &base)
                .expect("profile base");
            profile
                .apply(&predictive_parameter_registry(&profile_base), &profile_base)
                .expect("applied profile");
        }
        let entropy = matrix
            .profiles
            .iter()
            .find(|profile| profile.id == "uniform_entropy")
            .expect("entropy profile")
            .apply(&registry, &base)
            .expect("entropy config");
        assert_eq!(entropy.proxy_weights.entropy_w, 1.0);
        assert_eq!(entropy.proxy_weights.bucket_mass_w, 0.0);

        let ablations = PredictiveExperimentMatrix::parse_json(include_str!(
            "../../config/experiments/predictive-ablations.json"
        ))
        .expect("ablations");
        assert_eq!(ablations.profiles.len(), 6);
        for profile in &ablations.profiles {
            profile.apply(&registry, &base).expect("ablation profile");
        }
    }
}
