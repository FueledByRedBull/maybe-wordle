use chrono::NaiveDate;

use crate::solver::Suggestion;

use super::state::PredictiveArtifactState;
use super::{PredictivePromotionSource, PredictiveStateSummary};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuggestionValueKind {
    Proxy,
    Lookahead,
    ContinuationEstimate,
    ExactAction,
    Finite(crate::solver::FiniteSearchQuality),
    Terminal,
}

impl SuggestionValueKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Proxy => "heuristic proxy",
            Self::Lookahead => "lookahead estimate",
            Self::ContinuationEstimate => "pooled continuation estimate",
            Self::ExactAction => "model-exact action value",
            Self::Finite(crate::solver::FiniteSearchQuality::Heuristic) => "finite unevaluated",
            Self::Finite(crate::solver::FiniteSearchQuality::UpperBound) => {
                "finite completed rollout"
            }
            Self::Finite(crate::solver::FiniteSearchQuality::Exact) => {
                "finite model-exact action value"
            }
            Self::Terminal => "terminal solve probability",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchObjective {
    ProxyRanking,
    PenalizedLookahead,
    ExpectedGuesses,
    ThreeSolveCoverageThenCost,
    FailureThenAttempts,
    TerminalSolveProbability,
}

impl SearchObjective {
    pub fn label(self) -> &'static str {
        match self {
            Self::ProxyRanking => "heuristic ranking",
            Self::PenalizedLookahead => "penalized lookahead estimate",
            Self::ExpectedGuesses => "fixed-belief expected guesses",
            Self::ThreeSolveCoverageThenCost => "three-solve coverage then route-specific ranking",
            Self::FailureThenAttempts => "failure probability then attempts",
            Self::TerminalSolveProbability => "solve probability within remaining turns",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchActionScope {
    Normal,
    HardRootNormalContinuation,
    HardRecursive,
}

impl SearchActionScope {
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal-mode actions",
            Self::HardRootNormalContinuation => "hard-mode root; normal-mode future replies",
            Self::HardRecursive => "hard-mode actions at every turn",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchCandidateScope {
    AllActions,
    CandidatePool,
    BoundedSubset,
}

impl SearchCandidateScope {
    pub fn label(self) -> &'static str {
        match self {
            Self::AllActions => "all root actions",
            Self::CandidatePool => "candidate-pool roots",
            Self::BoundedSubset => "bounded root subset",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchExecution {
    pub route: super::PredictiveRegime,
    pub objective: SearchObjective,
    pub action_scope: SearchActionScope,
    pub candidate_scope: SearchCandidateScope,
    pub roots_considered: usize,
    pub roots_evaluated: usize,
    pub selected_value_kind: Option<SuggestionValueKind>,
    /// Optimal only for the stated model, objective and action scope; never a formal proof.
    pub root_selection_optimal: bool,
    pub stop_reason: Option<crate::solver::FiniteSearchReason>,
}

impl SearchExecution {
    pub fn route_label(&self) -> &'static str {
        match (self.route, self.candidate_scope) {
            (super::PredictiveRegime::Exact, SearchCandidateScope::AllActions) => {
                "exhaustive-root continuation"
            }
            (super::PredictiveRegime::Exact, _) => "pooled-root continuation",
            (route, _) => route.label(),
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "route={} objective={} actions={} coverage={} roots_evaluated={}/{} selected_value={} model_root_optimal={} stop={}",
            self.route_label(),
            self.objective.label(),
            self.action_scope.label(),
            self.candidate_scope.label(),
            self.roots_evaluated,
            self.roots_considered,
            self.selected_value_kind
                .map_or("unavailable", SuggestionValueKind::label),
            self.root_selection_optimal,
            match self.stop_reason {
                None | Some(crate::solver::FiniteSearchReason::Complete) => "complete",
                Some(crate::solver::FiniteSearchReason::Deadline) => "deadline",
                Some(crate::solver::FiniteSearchReason::NodeBudget) => "node budget",
                Some(crate::solver::FiniteSearchReason::Cancelled) => "cancelled",
            }
        )
    }
}

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
    pub puzzle_date: NaiveDate,
    pub observations: &'a [(String, u8)],
    pub top: usize,
    pub hard_mode: bool,
    pub force_in_two_only: bool,
    pub mode: PredictiveSuggestionMode,
}

/// Inclusive snapshot cutoff for information available before a puzzle.
pub fn history_cutoff(puzzle_date: NaiveDate) -> anyhow::Result<NaiveDate> {
    puzzle_date
        .pred_opt()
        .ok_or_else(|| anyhow::anyhow!("puzzle date has no preceding history date"))
}

#[derive(Clone, Debug)]
pub struct PredictiveSuggestResponse {
    pub execution: SearchExecution,
    pub finite_search: Option<crate::solver::FiniteSearchResult>,
    pub puzzle_date: NaiveDate,
    pub history_cutoff: NaiveDate,
    pub state: PredictiveStateSummary,
    pub suggestions: Vec<Suggestion>,
    pub candidates: Vec<PredictiveCandidateSummary>,
    pub promoted_word: Option<String>,
    pub promotion_source: Option<PredictivePromotionSource>,
    pub promoted_artifact_date: Option<NaiveDate>,
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
