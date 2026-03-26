use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryMode {
    Strict,
    UniformOverSupport,
    EpsilonRepair,
}

impl RecoveryMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::UniformOverSupport => "uniform_over_support",
            Self::EpsilonRepair => "epsilon_repair",
        }
    }
}

impl Default for RecoveryMode {
    fn default() -> Self {
        Self::EpsilonRepair
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RecoveryPolicy {
    pub mode: RecoveryMode,
    pub epsilon_scale: f64,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            mode: RecoveryMode::default(),
            epsilon_scale: 1e-6,
        }
    }
}

impl RecoveryPolicy {
    pub fn label(&self) -> &'static str {
        self.mode.label()
    }

    pub fn repair_weight(&self, seed_weight: f64, support_size: usize) -> f64 {
        match self.mode {
            RecoveryMode::Strict => 0.0,
            RecoveryMode::UniformOverSupport => {
                if support_size == 0 {
                    0.0
                } else {
                    1.0 / support_size as f64
                }
            }
            RecoveryMode::EpsilonRepair => {
                if support_size == 0 {
                    0.0
                } else {
                    (seed_weight * self.epsilon_scale).max(f64::MIN_POSITIVE)
                }
            }
        }
    }

    pub fn needs_repair(&self, total_weight: f64) -> bool {
        total_weight <= 0.0
    }
}
