use chrono::NaiveDate;

use crate::solver::Suggestion;

use super::state::PredictiveArtifactState;
use super::{PredictivePromotionSource, PredictiveStateSummary};

#[derive(Clone, Debug)]
pub struct PredictiveCandidateSummary {
    pub word: String,
    pub probability: f64,
    pub modeled_weight: f64,
    pub fallback_support: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictiveSuggestionMode {
    LiveOnly,
    FastDiskOnly,
    Full,
}

#[derive(Clone, Copy, Debug)]
pub struct PredictiveSuggestRequest<'a> {
    pub as_of: NaiveDate,
    pub observations: &'a [(String, u8)],
    pub top: usize,
    pub hard_mode: bool,
    pub force_in_two_only: bool,
    pub mode: PredictiveSuggestionMode,
}

#[derive(Clone, Debug)]
pub struct PredictiveSuggestResponse {
    pub state: PredictiveStateSummary,
    pub suggestions: Vec<Suggestion>,
    pub candidates: Vec<PredictiveCandidateSummary>,
    pub promoted_word: Option<String>,
    pub promotion_source: Option<PredictivePromotionSource>,
    pub artifact_state: PredictiveArtifactState,
    pub model_version: String,
    pub model_manifest_hash: String,
    pub history_snapshot_date: Option<NaiveDate>,
    pub history_snapshot_hash: String,
}

impl PredictiveSuggestResponse {
    pub fn artifact_state(&self) -> PredictiveArtifactState {
        self.artifact_state
    }
}
