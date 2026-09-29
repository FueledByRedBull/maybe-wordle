use std::time::Duration;

use super::*;
use crate::predictive::types::{
    SearchActionScope, SearchCandidateScope, SearchExecution, SearchObjective, SuggestionValueKind,
};

impl FiniteSearchOptions {
    pub fn fast() -> Self {
        Self {
            root_shortlist: 8,
            reply_shortlist: 8,
            exact_state_threshold: 6,
            budget: Duration::from_millis(250),
            node_limit: None,
            baseline_only: false,
        }
    }

    pub fn strong() -> Self {
        Self {
            root_shortlist: 16,
            reply_shortlist: 16,
            exact_state_threshold: 8,
            budget: Duration::from_secs(2),
            node_limit: None,
            baseline_only: false,
        }
    }

    /// Diagnostic control for repeatability experiments. The wall clock is
    /// only a safety stop; results that hit it are not fixed-work evidence.
    pub fn fast_fixed_work() -> Self {
        let mut options = Self::fast();
        options.budget = Duration::from_secs(5);
        options.node_limit = Some(4_000_000);
        options
    }
}

impl Solver {
    pub fn finite_search_options(&self) -> FiniteSearchOptions {
        match self.config.search_policy_mode {
            crate::config::SearchPolicyMode::FiniteStrong => FiniteSearchOptions::strong(),
            crate::config::SearchPolicyMode::FiniteFastFixedWork => {
                FiniteSearchOptions::fast_fixed_work()
            }
            crate::config::SearchPolicyMode::FiniteBaseline => {
                let mut options = FiniteSearchOptions::fast();
                options.baseline_only = true;
                options
            }
            _ => FiniteSearchOptions::fast(),
        }
    }

    pub(super) fn finite_suggestion_batch(
        &self,
        state: &SolveState,
        top: usize,
        context: Option<PredictiveContext<'_>>,
        options: FiniteSearchOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<SuggestionBatch> {
        let observations = context.map_or(&[][..], |context| context.observations);
        let remaining_turns = 6usize
            .checked_sub(observations.len())
            .ok_or_else(|| anyhow!("a Wordle game has at most six turns"))?
            as u8;
        let mut search = if remaining_turns == 0
            || observations
                .last()
                .is_some_and(|(_, pattern)| *pattern == ALL_GREEN_PATTERN)
        {
            FiniteSearchResult {
                candidates: Vec::new(),
                root_candidates_considered: 0,
                all_legal_roots_evaluated: false,
                reason: FiniteSearchReason::Complete,
                nodes_visited: 0,
                work_units: 0,
                cache_hits: 0,
                proposal_sampled: false,
            }
        } else if state.condition_only {
            self.finite_horizon_search(
                &state.surviving,
                &state.weights,
                observations,
                remaining_turns,
                context.is_some_and(|context| context.hard_mode),
                options,
                cancelled,
            )?
        } else {
            self.finite_horizon_search_dynamic(
                state,
                observations,
                remaining_turns,
                context.is_some_and(|context| context.hard_mode),
                options,
                cancelled,
            )?
        };
        let absent_mask = known_absent_letter_mask(observations);
        let all_answers_guessable = state.surviving.iter().all(|answer_index| {
            self.guess_index
                .contains_key(&self.answers[*answer_index].word)
        });
        let mut scratch = GuessMetricScratch::new();
        let suggestions = search
            .candidates
            .iter()
            .take(top)
            .enumerate()
            // Preserve one usable action, but do not format a large obsolete result set.
            .take_while(|(index, _)| *index == 0 || !cancelled())
            .map(|(_, candidate)| {
                let metric = self.score_guess_metrics(
                    candidate.guess_index,
                    &mut scratch,
                    GuessMetricContext {
                        subset: &state.surviving,
                        weights: &state.weights,
                        total_weight: state.total_weight,
                        posterior_answer_probability: candidate_probability(
                            self,
                            state,
                            candidate.guess_index,
                        ),
                    },
                );
                Suggestion {
                    value_kind: SuggestionValueKind::Finite(candidate.quality),
                    finite_value: Some(*candidate),
                    word: self.guesses[candidate.guess_index].clone(),
                    entropy: metric.entropy,
                    solve_probability: metric.solve_probability,
                    expected_remaining: metric.expected_remaining,
                    force_in_two: metric.force_in_two && all_answers_guessable,
                    known_absent_letter_hits: count_masked_letters(
                        &self.guesses[candidate.guess_index],
                        absent_mask,
                    ),
                    worst_non_green_bucket_size: metric.worst_non_green_bucket_size,
                    largest_non_green_bucket_mass: metric.largest_non_green_bucket_mass,
                    large_non_green_bucket_count: metric.large_non_green_bucket_count,
                    dangerous_mass_bucket_count: metric.dangerous_mass_bucket_count,
                    non_green_mass_in_large_buckets: metric.non_green_mass_in_large_buckets,
                    proxy_cost: Some(metric.proxy_cost),
                    large_state_score: Some(metric.large_state_score),
                    posterior_answer_probability: metric.posterior_answer_probability,
                    lookahead_cost: None,
                    exact_cost: None,
                }
            })
            .collect::<Vec<_>>();
        if cancelled() {
            search.reason = FiniteSearchReason::Cancelled;
        }
        Ok(SuggestionBatch {
            execution: SearchExecution {
                route: PredictiveRegime::Finite,
                objective: SearchObjective::FailureThenAttempts,
                action_scope: if context.is_some_and(|context| context.hard_mode) {
                    SearchActionScope::HardRecursive
                } else {
                    SearchActionScope::Normal
                },
                candidate_scope: if search.all_legal_roots_evaluated {
                    SearchCandidateScope::AllActions
                } else {
                    SearchCandidateScope::BoundedSubset
                },
                roots_considered: search.root_candidates_considered,
                roots_evaluated: search
                    .candidates
                    .iter()
                    .filter(|row| row.quality != FiniteSearchQuality::Heuristic)
                    .count(),
                selected_value_kind: suggestions.first().map(|row| row.value_kind),
                root_selection_optimal: search.all_legal_roots_evaluated
                    && search.reason == FiniteSearchReason::Complete
                    && !suggestions.is_empty()
                    && search
                        .candidates
                        .iter()
                        .all(|row| row.quality == FiniteSearchQuality::Exact),
                stop_reason: Some(search.reason),
            },
            root_candidate_count: search.candidates.len(),
            finite_search: Some(search),
            suggestions,
            promoted_word: None,
            promotion_source: None,
            promoted_artifact_date: None,
            danger_score: 0.0,
            danger_escalated: false,
            regime_used: PredictiveRegime::Finite,
            lookahead_pool_base: 0,
            lookahead_pool_size: 0,
            exact_pool_base: 0,
            exact_pool_size: 0,
        })
    }
}

fn candidate_probability(solver: &Solver, state: &SolveState, guess_index: usize) -> f64 {
    state
        .surviving
        .iter()
        .find(|index| solver.answers[**index].word == solver.guesses[guess_index])
        .map_or(0.0, |index| state.weights[*index] / state.total_weight)
}

#[cfg(test)]
mod execution_metadata_tests {
    use super::*;

    #[test]
    fn finite_budget_stops_and_complete_search_report_actual_quality_and_scope() {
        let solver = crate::solver::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let state = solver.initial_state(date);
        let context = Some(PredictiveContext {
            hard_mode: true,
            as_of: date,
            observations: &[],
        });
        let complete = FiniteSearchOptions {
            budget: Duration::from_secs(5),
            ..FiniteSearchOptions::fast()
        };
        for (options, reason) in [
            (
                FiniteSearchOptions {
                    budget: Duration::ZERO,
                    ..complete
                },
                FiniteSearchReason::Deadline,
            ),
            (
                FiniteSearchOptions {
                    node_limit: Some(0),
                    ..complete
                },
                FiniteSearchReason::NodeBudget,
            ),
            (complete, FiniteSearchReason::Complete),
        ] {
            let batch = solver
                .finite_suggestion_batch(&state, 3, context, options, &|| false)
                .unwrap();
            assert_eq!(batch.execution.route, PredictiveRegime::Finite);
            assert_eq!(
                batch.execution.objective,
                SearchObjective::FailureThenAttempts
            );
            assert_eq!(
                batch.execution.action_scope,
                SearchActionScope::HardRecursive
            );
            assert_eq!(batch.execution.stop_reason, Some(reason));
            assert_eq!(
                batch.execution.selected_value_kind,
                batch.suggestions.first().map(|row| row.value_kind)
            );
            if reason == FiniteSearchReason::Complete {
                assert_eq!(
                    batch.execution.candidate_scope,
                    SearchCandidateScope::AllActions
                );
                assert_eq!(batch.execution.roots_evaluated, 3);
                assert_eq!(batch.execution.roots_considered, 3);
                assert_eq!(
                    batch.execution.selected_value_kind,
                    Some(SuggestionValueKind::Finite(FiniteSearchQuality::Exact))
                );
                assert!(batch.execution.root_selection_optimal);
            } else {
                assert_eq!(
                    batch.execution.candidate_scope,
                    SearchCandidateScope::BoundedSubset
                );
                assert_eq!(batch.execution.roots_evaluated, 0);
                assert_eq!(batch.execution.roots_considered, 1);
                assert_eq!(
                    batch.execution.selected_value_kind,
                    Some(SuggestionValueKind::Finite(FiniteSearchQuality::Heuristic))
                );
                assert!(!batch.execution.root_selection_optimal);
            }
        }
        let cancelled = solver
            .finite_suggestion_batch(&state, 3, context, complete, &|| true)
            .unwrap();
        assert_eq!(
            cancelled.execution.stop_reason,
            Some(FiniteSearchReason::Cancelled)
        );
        assert!(!cancelled.execution.root_selection_optimal);
    }

    #[test]
    fn finite_baseline_action_value_is_not_global_root_selection() {
        let solver = crate::solver::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let state = solver.initial_state(date);
        let history = vec![("xxxxx".into(), 0); 5];
        let batch = solver
            .finite_suggestion_batch(
                &state,
                3,
                Some(PredictiveContext {
                    hard_mode: true,
                    as_of: date,
                    observations: &history,
                }),
                FiniteSearchOptions {
                    baseline_only: true,
                    budget: Duration::from_secs(5),
                    ..FiniteSearchOptions::fast()
                },
                &|| false,
            )
            .unwrap();
        assert_eq!(
            batch.execution.stop_reason,
            Some(FiniteSearchReason::Complete)
        );
        assert_eq!(
            batch.execution.selected_value_kind,
            Some(SuggestionValueKind::Finite(FiniteSearchQuality::Exact))
        );
        assert_eq!(
            batch.execution.candidate_scope,
            SearchCandidateScope::BoundedSubset
        );
        assert_eq!(batch.execution.roots_evaluated, 1);
        assert!(!batch.execution.root_selection_optimal);
        let rollout = solver
            .finite_suggestion_batch(
                &state,
                3,
                None,
                FiniteSearchOptions {
                    baseline_only: true,
                    budget: Duration::from_secs(5),
                    ..FiniteSearchOptions::fast()
                },
                &|| false,
            )
            .unwrap();
        assert_eq!(
            rollout.execution.selected_value_kind,
            Some(SuggestionValueKind::Finite(FiniteSearchQuality::UpperBound))
        );
        assert_eq!(
            rollout.execution.stop_reason,
            Some(FiniteSearchReason::Complete)
        );
        assert_eq!(
            rollout.execution.candidate_scope,
            SearchCandidateScope::BoundedSubset
        );
        assert!(!rollout.execution.root_selection_optimal);
    }
}

#[cfg(test)]
mod tests {
    use super::FiniteSearchOptions;

    #[test]
    fn fixed_work_profile_keeps_fast_search_geometry_and_a_safety_deadline() {
        let fast = FiniteSearchOptions::fast();
        let fixed = FiniteSearchOptions::fast_fixed_work();
        assert_eq!(fixed.root_shortlist, fast.root_shortlist);
        assert_eq!(fixed.reply_shortlist, fast.reply_shortlist);
        assert_eq!(fixed.exact_state_threshold, fast.exact_state_threshold);
        assert!(!fixed.baseline_only);
        assert_eq!(fixed.node_limit, Some(4_000_000));
        assert!(fixed.budget > fast.budget);
    }
}
