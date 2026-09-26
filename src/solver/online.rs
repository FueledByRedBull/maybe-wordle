use std::time::Duration;

use super::*;

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
            .collect();
        if cancelled() {
            search.reason = FiniteSearchReason::Cancelled;
        }
        Ok(SuggestionBatch {
            root_candidate_count: search.candidates.len(),
            finite_search: Some(search),
            suggestions,
            promoted_word: None,
            promotion_source: None,
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
