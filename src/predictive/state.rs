use super::RecoveryMode;

#[derive(Clone, Debug)]
pub struct PredictiveStateSummary {
    pub surviving: usize,
    pub modeled_total_weight: f64,
    pub effective_total_weight: f64,
    pub recovery_mode_used: Option<RecoveryMode>,
}
