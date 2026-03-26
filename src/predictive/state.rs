use super::{PredictivePromotionSource, RecoveryMode};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictiveArtifactState {
    ExactDateArtifact,
    RecentOpenerArtifact,
    LiveSessionFallback,
    NoPredictiveArtifactAvailable,
}

impl PredictiveArtifactState {
    pub fn banner_text(self) -> &'static str {
        match self {
            Self::ExactDateArtifact => "Using exact-date predictive artifact",
            Self::RecentOpenerArtifact => "Using recent opener artifact",
            Self::LiveSessionFallback => "Using live session fallback",
            Self::NoPredictiveArtifactAvailable => "No predictive artifact available",
        }
    }

    pub fn compute_text(self) -> &'static str {
        match self {
            Self::ExactDateArtifact | Self::RecentOpenerArtifact => "disk-backed",
            Self::LiveSessionFallback => "live session fallback",
            Self::NoPredictiveArtifactAvailable => "no predictive artifact available",
        }
    }

    pub fn from_promotion_source(source: Option<PredictivePromotionSource>) -> Self {
        match source {
            Some(PredictivePromotionSource::ExactDateOpenerArtifact)
            | Some(PredictivePromotionSource::ReplyBook) => Self::ExactDateArtifact,
            Some(PredictivePromotionSource::RecentOpenerArtifact) => Self::RecentOpenerArtifact,
            Some(PredictivePromotionSource::SessionRootFallback)
            | Some(PredictivePromotionSource::SessionReplyFallback)
            | Some(PredictivePromotionSource::SessionThirdFallback) => Self::LiveSessionFallback,
            None => Self::NoPredictiveArtifactAvailable,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PredictiveStateSummary {
    pub surviving: usize,
    pub modeled_total_weight: f64,
    pub effective_total_weight: f64,
    pub recovery_mode_used: Option<RecoveryMode>,
}
