use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::PriorConfig;

use super::{ParameterRegistry, ParameterValue};

pub const PROFILE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictiveConfigProfile {
    pub format_version: u32,
    pub name: String,
    pub description: String,
    pub parameters: BTreeMap<String, ParameterValue>,
}

impl PredictiveConfigProfile {
    pub fn parse_json(source: &str) -> Result<Self> {
        let profile: Self = serde_json::from_str(source)?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format_version != PROFILE_FORMAT_VERSION {
            bail!(
                "unsupported predictive profile format {}; expected {}",
                self.format_version,
                PROFILE_FORMAT_VERSION
            );
        }
        if self.name.trim().is_empty() || self.description.trim().is_empty() {
            bail!("predictive profile name and description must not be empty");
        }
        if self.parameters.is_empty() {
            bail!("predictive profile must declare at least one parameter");
        }
        Ok(())
    }

    pub fn apply(&self, registry: &ParameterRegistry, base: &PriorConfig) -> Result<PriorConfig> {
        self.validate()?;
        registry.apply_tunable_values(base, &self.parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::predictive_parameter_registry;

    #[test]
    fn serialized_profile_is_validated_through_the_registry() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        let profile = PredictiveConfigProfile::parse_json(
            r#"{
                "format_version": 1,
                "name": "test",
                "description": "test profile",
                "parameters": {
                    "exact_threshold": {"type": "integer", "value": 72}
                }
            }"#,
        )
        .expect("profile");
        assert_eq!(
            profile
                .apply(&registry, &base)
                .expect("apply")
                .exact_threshold,
            72
        );

        let mut invalid = profile;
        invalid.parameters.insert(
            "sync_retry_attempts".to_string(),
            ParameterValue::Integer(9),
        );
        assert!(invalid.apply(&registry, &base).is_err());
    }

    #[test]
    fn shipped_profiles_parse_and_apply_to_the_default_config() {
        let base = PriorConfig::default();
        let registry = predictive_parameter_registry(&base);
        for source in [
            include_str!("../../config/profiles/aggressive-three-guess.json"),
            include_str!("../../config/profiles/offline-book.json"),
            include_str!("../../config/profiles/wide-pools.json"),
        ] {
            let profile = PredictiveConfigProfile::parse_json(source).expect("profile");
            let applied = profile.apply(&registry, &base).expect("apply profile");
            assert_ne!(
                toml::to_string(&applied).expect("applied"),
                toml::to_string(&base).expect("base")
            );
        }
    }
}
