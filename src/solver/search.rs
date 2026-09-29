use super::*;
use crate::predictive::types::{
    SearchActionScope, SearchCandidateScope, SearchExecution, SearchObjective, SuggestionValueKind,
};
use std::{
    sync::OnceLock,
    time::{Duration, Instant},
};

#[derive(Default)]
struct SearchTiming {
    base_metrics: Duration,
    second_guess_coverage: Duration,
    lookahead_root: Duration,
    lookahead_root_count: usize,
    exact_child: Duration,
    exact_child_count: usize,
    large_child_metric_scan: Duration,
    large_child_metric_scan_count: usize,
}

impl SearchTiming {
    fn from_env() -> Option<Self> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        ENABLED
            .get_or_init(|| {
                matches!(
                    std::env::var("MAYBE_WORDLE_EVIDENCE_TIMING")
                        .ok()
                        .as_deref(),
                    Some("1" | "true")
                )
            })
            .then_some(Self::default())
    }

    fn format_line(&self) -> String {
        format!(
            "benchmark-evidence search-timing base_ms={} coverage_ms={} lookahead_root_ms={} lookahead_root_count={} exact_child_ms={} exact_child_count={} large_child_metric_scan_ms={} large_child_metric_scan_count={}",
            self.base_metrics.as_millis().min(u64::MAX as u128),
            self.second_guess_coverage.as_millis().min(u64::MAX as u128),
            self.lookahead_root.as_millis().min(u64::MAX as u128),
            self.lookahead_root_count,
            self.exact_child.as_millis().min(u64::MAX as u128),
            self.exact_child_count,
            self.large_child_metric_scan
                .as_millis()
                .min(u64::MAX as u128),
            self.large_child_metric_scan_count,
        )
    }

    fn emit(&self) {
        eprintln!("{}", self.format_line());
        let _ = std::io::stderr().flush();
    }
}

pub(crate) fn check_predictive_search_cancelled(
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<()> {
    if cancelled() {
        bail!("predictive search cancelled");
    }
    Ok(())
}

impl Solver {
    pub(super) fn filtered_suggestion_batch_for_history_with_search_mode_controlled(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
        filters: PredictiveSuggestionFilters,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<SuggestionBatch> {
        check_predictive_search_cancelled(cancelled)?;
        let state = self.apply_history(as_of, observations)?;
        check_predictive_search_cancelled(cancelled)?;
        let limit = if filters.hard_mode || filters.force_in_two_only {
            self.guesses.len()
        } else {
            top
        };
        let mut batch = self.suggestion_batch_internal_with_search_mode_controlled(
            &state,
            limit,
            Some(PredictiveContext {
                hard_mode: filters.hard_mode,
                as_of,
                observations,
            }),
            book_usage_for_mode(filters.mode),
            filters.forced_search_mode,
            cancelled,
        )?;
        check_predictive_search_cancelled(cancelled)?;
        if filters.hard_mode {
            batch.suggestions.retain(|suggestion| {
                self.hard_mode_violation(observations, &suggestion.word)
                    .is_none()
            });
        }
        if filters.force_in_two_only {
            batch
                .suggestions
                .retain(|suggestion| suggestion.force_in_two);
            batch.execution.root_selection_optimal = false;
        }
        batch.suggestions.truncate(top.min(batch.suggestions.len()));
        batch.execution.selected_value_kind = batch.suggestions.first().map(|row| row.value_kind);
        if batch.promoted_word.as_ref().is_some_and(|word| {
            batch
                .suggestions
                .first()
                .is_none_or(|row| &row.word != word)
        }) {
            batch.promoted_word = None;
            batch.promotion_source = None;
            batch.promoted_artifact_date = None;
        }
        Ok(batch)
    }

    pub(super) fn suggestion_batch_internal(
        &self,
        state: &SolveState,
        top: usize,
        context: Option<PredictiveContext<'_>>,
        book_usage: PredictiveBookUsage,
    ) -> Result<SuggestionBatch> {
        self.suggestion_batch_internal_with_search_mode_controlled(
            state,
            top,
            context,
            book_usage,
            None,
            &|| false,
        )
    }

    pub(super) fn suggestion_batch_internal_with_search_mode_controlled(
        &self,
        state: &SolveState,
        top: usize,
        context: Option<PredictiveContext<'_>>,
        book_usage: PredictiveBookUsage,
        forced_search_mode: Option<PredictiveSearchMode>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<SuggestionBatch> {
        check_predictive_search_cancelled(cancelled)?;
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        if state.total_weight <= 0.0 {
            bail!("cannot score guesses when no positive answer mass remains");
        }
        if let Some(context) = context
            && (context.observations.len() >= 6
                || context
                    .observations
                    .last()
                    .is_some_and(|(_, pattern)| *pattern == ALL_GREEN_PATTERN))
        {
            return self.terminal_suggestion_batch(state, top, context, cancelled);
        }
        if self.config.search_policy_mode.is_finite() && forced_search_mode.is_none() {
            return self.finite_suggestion_batch(
                state,
                top,
                context,
                self.finite_search_options(),
                cancelled,
            );
        }
        if let Some(context) = context
            && context.observations.len() >= 4
        {
            return self.terminal_suggestion_batch(state, top, context, cancelled);
        }
        let mut search_timing = SearchTiming::from_env();
        let split_first = state.surviving.len() > self.config.large_state_split_threshold;
        let use_second_guess_coverage = should_use_second_guess_coverage(
            &self.config,
            state.surviving.len(),
            context.map_or(0, |context| context.observations.len()),
        );
        let known_absent_mask = context
            .as_ref()
            .map(|context| known_absent_letter_mask(context.observations))
            .unwrap_or(0);
        let base_metrics_started = search_timing.as_ref().map(|_| Instant::now());
        let mut metrics = self.score_guess_metrics_for_subset_controlled(
            &state.surviving,
            &state.weights,
            cancelled,
        )?;
        if let (Some(timing), Some(started)) = (search_timing.as_mut(), base_metrics_started) {
            timing.base_metrics = started.elapsed();
        }
        check_predictive_search_cancelled(cancelled)?;
        if state.surviving.iter().any(|answer_index| {
            !self
                .guess_index
                .contains_key(&self.answers[*answer_index].word)
        }) {
            for metric in &mut metrics {
                metric.force_in_two = false;
            }
        }
        if known_absent_mask != 0 {
            for metric in &mut metrics {
                metric.known_absent_letter_hits =
                    count_masked_letters(&self.guesses[metric.guess_index], known_absent_mask);
                metric.large_state_score = proxy_row_score_from_weights(
                    &self.config.proxy_weights,
                    ProxyRowStats::from_metric(metric),
                );
            }
        }
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let three_solve_coverage = if use_second_guess_coverage {
            let coverage_started = search_timing.as_ref().map(|_| Instant::now());
            let coverage = self.medium_second_guess_coverage_controlled(
                &state.surviving,
                &state.weights,
                &metrics,
                cancelled,
            )?;
            if let (Some(timing), Some(started)) = (search_timing.as_mut(), coverage_started) {
                timing.second_guess_coverage = started.elapsed();
            }
            check_predictive_search_cancelled(cancelled)?;
            Some(coverage)
        } else {
            None
        };
        if let Some(coverage) = three_solve_coverage.as_ref() {
            metrics.sort_by(|left, right| {
                compare_guess_metrics_with_coverage(
                    left,
                    right,
                    &self.guesses,
                    split_first,
                    coverage,
                )
            });
        }
        let assessment = self.assess_state_danger(state, &metrics);
        let search_mode = forced_search_mode.unwrap_or_else(|| {
            predictive_search_mode(&self.config, state.surviving.len(), assessment)
        });
        let mut suggestions = metrics
            .into_iter()
            .map(|metric| self.suggestion_from_metric(metric))
            .collect::<Vec<_>>();
        suggestions
            .retain(|suggestion| suggestion.worst_non_green_bucket_size < state.surviving.len());
        if suggestions.is_empty() {
            bail!(
                "no predictive guess makes progress on a state with {} survivors",
                state.surviving.len()
            );
        }
        let lookahead_pool_base = self.lookahead_candidate_pool_for_state(state.surviving.len());
        let lookahead_pool = self.expanded_pool_size(
            &suggestions,
            lookahead_pool_base,
            split_first,
            matches!(search_mode, PredictiveSearchMode::Lookahead)
                && state.surviving.len() > self.config.large_state_split_threshold,
            assessment,
        );
        let exact_pool = self.expanded_pool_size(
            &suggestions,
            self.config.exact_candidate_pool,
            split_first,
            matches!(search_mode, PredictiveSearchMode::EscalatedExact)
                && state.surviving.len() > self.config.exact_threshold,
            assessment,
        );
        let mut root_candidate_count = 0usize;
        let mut roots_evaluated = self.guesses.len();
        let mut exact_actions = false;

        if let PredictiveSearchMode::Lookahead = search_mode {
            check_predictive_search_cancelled(cancelled)?;
            let root_candidates = self.collect_lookahead_candidates(
                &suggestions,
                state.surviving.len(),
                assessment.dangerous_lookahead,
                lookahead_pool,
            )?;
            check_predictive_search_cancelled(cancelled)?;
            root_candidate_count = root_candidates.len();
            let root_started = search_timing.as_ref().map(|_| Instant::now());
            if let Some(timing) = search_timing.as_mut() {
                timing.lookahead_root_count = root_candidate_count;
            }
            let mut exact_memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut lookahead_memo = PredictiveMemoMap::default();
            let mut lookahead_costs = vec![None; self.guesses.len()];

            for guess_index in root_candidates {
                check_predictive_search_cancelled(cancelled)?;
                let context = LookaheadCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    expanded: assessment.dangerous_lookahead,
                    exact_memo: &mut exact_memo,
                    exact_scratch: &mut exact_scratch,
                    lookahead_memo: &mut lookahead_memo,
                };
                let cost = self.lookahead_cost_for_guess_controlled(
                    guess_index,
                    context,
                    search_timing.as_mut().map(|timing| &mut *timing),
                    cancelled,
                )?;
                lookahead_costs[guess_index] = Some(cost);
            }
            if let (Some(timing), Some(started)) = (search_timing.as_mut(), root_started) {
                timing.lookahead_root = started.elapsed();
            }

            for suggestion in &mut suggestions {
                suggestion.lookahead_cost = self
                    .guess_index
                    .get(&suggestion.word)
                    .and_then(|guess_index| lookahead_costs[*guess_index]);
                if suggestion.lookahead_cost.is_some() {
                    suggestion.value_kind = SuggestionValueKind::Lookahead;
                }
            }
            roots_evaluated = root_candidate_count;
            suggestions.sort_by(|left, right| {
                if let Some(coverage) = three_solve_coverage.as_ref() {
                    compare_suggestions_with_coverage(
                        left,
                        right,
                        split_first,
                        &self.guess_index,
                        coverage,
                    )
                    .then_with(|| compare_lookahead(left, right, split_first))
                } else {
                    compare_lookahead(left, right, split_first)
                }
            });
        }

        if let PredictiveSearchMode::Exact(exact_mode) = search_mode {
            check_predictive_search_cancelled(cancelled)?;
            let exact_candidates = match exact_mode {
                ExactSuggestionMode::Exhaustive => (0..self.guesses.len()).collect::<Vec<_>>(),
                ExactSuggestionMode::Pooled => {
                    self.collect_exact_candidates(state, &suggestions, exact_pool)?
                }
            };
            check_predictive_search_cancelled(cancelled)?;
            root_candidate_count = exact_candidates.len();
            let mut memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut exact_costs = vec![None; self.guesses.len()];
            // Coverage is the primary ranking key when active, so a cost-only
            // bound must not skip any of its candidates.
            let bound_ranked_prefix = exact_mode == ExactSuggestionMode::Pooled
                && three_solve_coverage.is_none()
                && top > 0
                && top < exact_candidates.len();
            let mut ordered_candidates = exact_candidates
                .into_iter()
                .map(|guess_index| (guess_index, 0.0))
                .collect::<Vec<_>>();
            if bound_ranked_prefix {
                for (guess_index, bound) in &mut ordered_candidates {
                    *bound = self.exact_root_cost_lower_bound(
                        *guess_index,
                        &state.surviving,
                        &state.weights,
                    )?;
                }
                ordered_candidates.sort_by(|left, right| left.1.total_cmp(&right.1));
            }
            let mut best_exact_costs = Vec::with_capacity(top.min(ordered_candidates.len()));

            for (guess_index, lower_bound) in ordered_candidates {
                check_predictive_search_cancelled(cancelled)?;
                if bound_ranked_prefix
                    && best_exact_costs.len() == top
                    && lower_bound > best_exact_costs[top - 1] + 1e-10
                {
                    break;
                }
                let context = ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    memo: &mut memo,
                    best_bound: f64::INFINITY,
                    scratch: &mut exact_scratch,
                    depth: 0,
                };
                let cost = self.exact_cost_for_guess_controlled(guess_index, context, cancelled)?;
                exact_costs[guess_index] = Some(cost);
                if bound_ranked_prefix {
                    best_exact_costs.push(cost);
                    best_exact_costs.sort_by(f64::total_cmp);
                    best_exact_costs.truncate(top);
                }
            }

            roots_evaluated = exact_costs.iter().filter(|cost| cost.is_some()).count();
            exact_actions = !exact_scratch.used_candidate_pool;
            match exact_mode {
                ExactSuggestionMode::Exhaustive => {
                    for suggestion in &mut suggestions {
                        suggestion.exact_cost = self
                            .guess_index
                            .get(&suggestion.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
                        if suggestion.exact_cost.is_some() {
                            suggestion.value_kind = if exact_actions {
                                SuggestionValueKind::ExactAction
                            } else {
                                SuggestionValueKind::ContinuationEstimate
                            };
                        }
                    }
                    suggestions.sort_by(|left, right| {
                        if let Some(coverage) = three_solve_coverage.as_ref() {
                            compare_suggestions_with_coverage(
                                left,
                                right,
                                split_first,
                                &self.guess_index,
                                coverage,
                            )
                            .then_with(|| compare_exact(left, right, split_first))
                        } else {
                            compare_exact(left, right, split_first)
                        }
                    });
                }
                ExactSuggestionMode::Pooled => {
                    for suggestion in &mut suggestions {
                        suggestion.exact_cost = self
                            .guess_index
                            .get(&suggestion.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
                        if suggestion.exact_cost.is_some() {
                            suggestion.value_kind = if exact_actions {
                                SuggestionValueKind::ExactAction
                            } else {
                                SuggestionValueKind::ContinuationEstimate
                            };
                        }
                    }
                    suggestions.sort_by(|left, right| {
                        let left_cost = self
                            .guess_index
                            .get(&left.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
                        let right_cost = self
                            .guess_index
                            .get(&right.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
                        if let Some(coverage) = three_solve_coverage.as_ref() {
                            compare_suggestions_with_coverage(
                                left,
                                right,
                                split_first,
                                &self.guess_index,
                                coverage,
                            )
                            .then_with(|| {
                                compare_exact_costs(left, right, left_cost, right_cost, split_first)
                            })
                        } else {
                            compare_exact_costs(left, right, left_cost, right_cost, split_first)
                        }
                    });
                }
            }
        }

        if let PredictiveSearchMode::EscalatedExact = search_mode {
            check_predictive_search_cancelled(cancelled)?;
            let exact_candidates = self.collect_exact_candidates(
                state,
                &suggestions,
                self.config.danger_exact_root_pool.max(1).max(exact_pool),
            )?;
            check_predictive_search_cancelled(cancelled)?;
            root_candidate_count = exact_candidates.len();
            let mut memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut exact_costs = vec![None; self.guesses.len()];

            for guess_index in exact_candidates {
                check_predictive_search_cancelled(cancelled)?;
                let context = ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    memo: &mut memo,
                    best_bound: f64::INFINITY,
                    scratch: &mut exact_scratch,
                    depth: 0,
                };
                let cost = self.exact_cost_for_guess_controlled(guess_index, context, cancelled)?;
                exact_costs[guess_index] = Some(cost);
            }

            roots_evaluated = exact_costs.iter().filter(|cost| cost.is_some()).count();
            exact_actions = !exact_scratch.used_candidate_pool;
            for suggestion in &mut suggestions {
                suggestion.exact_cost = self
                    .guess_index
                    .get(&suggestion.word)
                    .and_then(|guess_index| exact_costs[*guess_index]);
                if suggestion.exact_cost.is_some() {
                    suggestion.value_kind = if exact_actions {
                        SuggestionValueKind::ExactAction
                    } else {
                        SuggestionValueKind::ContinuationEstimate
                    };
                }
            }
            suggestions.sort_by(|left, right| {
                let left_cost = self
                    .guess_index
                    .get(&left.word)
                    .and_then(|guess_index| exact_costs[*guess_index]);
                let right_cost = self
                    .guess_index
                    .get(&right.word)
                    .and_then(|guess_index| exact_costs[*guess_index]);
                if let Some(coverage) = three_solve_coverage.as_ref() {
                    compare_suggestions_with_coverage(
                        left,
                        right,
                        split_first,
                        &self.guess_index,
                        coverage,
                    )
                    .then_with(|| {
                        compare_exact_costs(left, right, left_cost, right_cost, split_first)
                    })
                } else {
                    compare_exact_costs(left, right, left_cost, right_cost, split_first)
                }
            });
        }

        let mut promoted_word = None;
        let mut promotion_source = None;
        let mut promoted_artifact_date = None;
        check_predictive_search_cancelled(cancelled)?;
        if top > 0
            && book_usage != PredictiveBookUsage::None
            && let Some(context) = context
            && let Some(choice) = self.cached_predictive_choice(
                context.as_of,
                context.observations,
                book_usage == PredictiveBookUsage::Full,
                cancelled,
            )?
            && promote_cached_suggestion(&mut suggestions, &choice.word)
        {
            promoted_word = Some(choice.word);
            promotion_source = Some(choice.source);
            promoted_artifact_date = choice.artifact_date;
        }

        check_predictive_search_cancelled(cancelled)?;
        suggestions.truncate(top);
        let execution = SearchExecution {
            route: regime_from_search_mode(search_mode),
            objective: if three_solve_coverage.is_some() {
                SearchObjective::ThreeSolveCoverageThenCost
            } else if matches!(search_mode, PredictiveSearchMode::ProxyOnly) {
                SearchObjective::ProxyRanking
            } else if matches!(search_mode, PredictiveSearchMode::Lookahead) {
                SearchObjective::PenalizedLookahead
            } else {
                SearchObjective::ExpectedGuesses
            },
            action_scope: if context.is_some_and(|context| context.hard_mode) {
                SearchActionScope::HardRootNormalContinuation
            } else {
                SearchActionScope::Normal
            },
            candidate_scope: match search_mode {
                PredictiveSearchMode::ProxyOnly
                | PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive) => {
                    SearchCandidateScope::AllActions
                }
                _ => SearchCandidateScope::CandidatePool,
            },
            roots_considered: if matches!(search_mode, PredictiveSearchMode::ProxyOnly) {
                self.guesses.len()
            } else {
                root_candidate_count
            },
            roots_evaluated,
            selected_value_kind: suggestions.first().map(|row| row.value_kind),
            root_selection_optimal: exact_actions
                && matches!(
                    search_mode,
                    PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive)
                )
                && three_solve_coverage.is_none()
                && promoted_word.is_none()
                && !suggestions.is_empty()
                && !context.is_some_and(|context| context.hard_mode)
                && state
                    .surviving
                    .iter()
                    .all(|index| self.guess_index.contains_key(&self.answers[*index].word)),
            stop_reason: None,
        };
        let batch = SuggestionBatch {
            execution,
            finite_search: None,
            suggestions,
            promoted_word,
            promotion_source,
            promoted_artifact_date,
            danger_score: assessment.danger_score,
            danger_escalated: matches!(search_mode, PredictiveSearchMode::EscalatedExact)
                || (matches!(search_mode, PredictiveSearchMode::Lookahead)
                    && assessment.dangerous_lookahead),
            regime_used: regime_from_search_mode(search_mode),
            lookahead_pool_base,
            lookahead_pool_size: lookahead_pool,
            exact_pool_base: self.config.exact_candidate_pool,
            exact_pool_size: exact_pool,
            root_candidate_count,
        };
        if let Some(timing) = search_timing {
            timing.emit();
        }
        Ok(batch)
    }

    fn terminal_suggestion_batch(
        &self,
        state: &SolveState,
        top: usize,
        context: PredictiveContext<'_>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<SuggestionBatch> {
        let mut batch = SuggestionBatch {
            execution: SearchExecution {
                route: PredictiveRegime::Terminal,
                objective: SearchObjective::TerminalSolveProbability,
                action_scope: if context.hard_mode {
                    SearchActionScope::HardRecursive
                } else {
                    SearchActionScope::Normal
                },
                candidate_scope: SearchCandidateScope::AllActions,
                roots_considered: 0,
                roots_evaluated: 0,
                selected_value_kind: None,
                root_selection_optimal: false,
                stop_reason: None,
            },
            finite_search: None,
            suggestions: Vec::new(),
            promoted_word: None,
            promotion_source: None,
            promoted_artifact_date: None,
            danger_score: 0.0,
            danger_escalated: false,
            regime_used: PredictiveRegime::Terminal,
            lookahead_pool_base: 0,
            lookahead_pool_size: 0,
            exact_pool_base: 0,
            exact_pool_size: 0,
            root_candidate_count: 0,
        };
        let remaining_turns = 6usize.saturating_sub(context.observations.len());
        if remaining_turns == 0
            || context
                .observations
                .last()
                .is_some_and(|(_, p)| *p == ALL_GREEN_PATTERN)
        {
            return Ok(batch);
        }
        if state
            .surviving
            .iter()
            .any(|answer| !self.guess_index.contains_key(&self.answers[*answer].word))
        {
            bail!("terminal search requires every surviving answer to be a legal guess");
        }
        let absent_mask = known_absent_letter_mask(context.observations);
        let metrics = self.score_guess_metrics_for_subset_controlled(
            &state.surviving,
            &state.weights,
            cancelled,
        )?;
        let mut suggestions = metrics
            .into_iter()
            .filter(|metric| {
                !context.hard_mode
                    || self
                        .hard_mode_violation(
                            context.observations,
                            &self.guesses[metric.guess_index],
                        )
                        .is_none()
            })
            .map(|mut metric| {
                metric.known_absent_letter_hits =
                    count_masked_letters(&self.guesses[metric.guess_index], absent_mask);
                let mut row = self.suggestion_from_metric(metric);
                row.value_kind = SuggestionValueKind::Terminal;
                row
            })
            .collect::<Vec<_>>();
        if should_use_final_turn_objective(context.observations.len()) {
            suggestions.sort_by(compare_final_turn);
        } else {
            let success = self.terminal_two_turn_success(state, context, cancelled)?;
            for row in &mut suggestions {
                row.force_in_two = success[self.guess_index[&row.word]].1;
            }
            suggestions.sort_by(|left, right| {
                compare_two_turn(
                    left,
                    right,
                    success[self.guess_index[&left.word]].0,
                    success[self.guess_index[&right.word]].0,
                )
            });
        }
        check_predictive_search_cancelled(cancelled)?;
        batch.root_candidate_count = suggestions.len();
        batch.execution.roots_considered = suggestions.len();
        batch.execution.roots_evaluated = suggestions.len();
        suggestions.truncate(top);
        batch.execution.selected_value_kind = suggestions.first().map(|row| row.value_kind);
        batch.execution.root_selection_optimal = !suggestions.is_empty();
        batch.suggestions = suggestions;
        Ok(batch)
    }

    fn terminal_two_turn_success(
        &self,
        state: &SolveState,
        context: PredictiveContext<'_>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<(f64, bool)>> {
        // With fixed support, every answer in a feedback bucket is itself a legal
        // final guess. Dynamic recovery instead requires the actual child belief.
        if state.condition_only || state.fallback_surviving.is_empty() {
            let scores = self.two_turn_success_by_guess_controlled(state, cancelled)?;
            let mut result = Vec::with_capacity(scores.len());
            for (guess, score) in scores.into_iter().enumerate() {
                check_predictive_search_cancelled(cancelled)?;
                let mut counts = [0usize; PATTERN_SPACE];
                for answer in &state.surviving {
                    counts[self.answer_pattern(guess, *answer) as usize] += 1;
                }
                result.push((score, counts.iter().all(|count| *count <= 1)));
            }
            return Ok(result);
        }
        let mut scores = vec![(0.0, false); self.guesses.len()];
        let mut masses = [0.0; PATTERN_SPACE];
        for (guess_index, guess) in self.guesses.iter().enumerate() {
            check_predictive_search_cancelled(cancelled)?;
            if context.hard_mode
                && self
                    .hard_mode_violation(context.observations, guess)
                    .is_some()
            {
                continue;
            }
            masses.fill(0.0);
            for answer in &state.surviving {
                masses[self.answer_pattern(guess_index, *answer) as usize] +=
                    state.weights[*answer];
            }
            let mut success = masses[ALL_GREEN_PATTERN as usize] / state.total_weight;
            let mut guaranteed = true;
            for (pattern, mass) in masses
                .iter()
                .enumerate()
                .filter(|(p, mass)| *p != ALL_GREEN_PATTERN as usize && **mass > 0.0)
            {
                check_predictive_search_cancelled(cancelled)?;
                let mut child = state.clone();
                self.apply_feedback(&mut child, guess, pattern as u8)?;
                let best_mass = child
                    .surviving
                    .iter()
                    .map(|answer| child.weights[*answer])
                    .fold(0.0, f64::max);
                success += mass / state.total_weight * best_mass / child.total_weight;
                guaranteed &= child.surviving.len() == 1;
            }
            scores[guess_index] = (success, guaranteed);
        }
        Ok(scores)
    }

    pub(super) fn expanded_pool_size(
        &self,
        suggestions: &[Suggestion],
        base_pool: usize,
        split_first: bool,
        allow_expansion: bool,
        assessment: StateDangerAssessment,
    ) -> usize {
        let base = base_pool.max(1).min(suggestions.len().max(1));
        if suggestions.is_empty() || !allow_expansion {
            return base;
        }
        let kth = base
            .saturating_sub(1)
            .min(suggestions.len().saturating_sub(1));
        let gap = if split_first {
            let top = suggestions[0]
                .large_state_score
                .unwrap_or(f64::NEG_INFINITY);
            let kth_score = suggestions[kth]
                .large_state_score
                .unwrap_or(f64::NEG_INFINITY);
            (top - kth_score).abs()
        } else {
            let top = suggestions[0].proxy_cost.unwrap_or(f64::INFINITY);
            let kth_score = suggestions[kth].proxy_cost.unwrap_or(f64::INFINITY);
            (kth_score - top).abs()
        };
        let expanded = if gap < self.config.pool_tight_gap_threshold {
            scaled_pool_size(base, self.config.pool_tight_expansion_multiplier)
        } else if gap < self.config.pool_medium_gap_threshold {
            scaled_pool_size(base, self.config.pool_medium_expansion_multiplier)
        } else {
            base
        };
        let expanded = if assessment.dangerous_lookahead || assessment.dangerous_exact {
            expanded.saturating_add(self.config.danger_reply_pool_bonus)
        } else {
            expanded
        };
        expanded.min(suggestions.len())
    }

    pub(super) fn collect_exact_candidates(
        &self,
        state: &SolveState,
        suggestions: &[Suggestion],
        non_surviving_limit: usize,
    ) -> Result<Vec<usize>> {
        let pool = non_surviving_limit.max(1);
        let surviving_guess_indexes = state
            .surviving
            .iter()
            .filter_map(|answer_index| self.guess_index.get(&self.answers[*answer_index].word))
            .copied()
            .collect::<HashSet<_>>();
        let mut candidate_indexes = Vec::new();
        let mut seen = HashSet::new();
        let mut push_candidate = |guess_index: usize| {
            if seen.insert(guess_index) {
                candidate_indexes.push(guess_index);
            }
        };

        for suggestion in suggestions.iter().take(fractional_pool_take(
            pool,
            self.config.exact_pool_primary_fraction,
        )) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_entropy = suggestions.iter().collect::<Vec<_>>();
        by_entropy.sort_by(|left, right| {
            right
                .entropy
                .total_cmp(&left.entropy)
                .then_with(|| left.word.cmp(&right.word))
        });
        for suggestion in by_entropy.into_iter().take(fractional_pool_take(
            pool,
            self.config.exact_pool_entropy_fraction,
        )) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_worst_bucket = suggestions.iter().collect::<Vec<_>>();
        by_worst_bucket.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_worst_bucket.into_iter().take(fractional_pool_take(
            pool,
            self.config.exact_pool_worst_bucket_fraction,
        )) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_mass_reducer = suggestions.iter().collect::<Vec<_>>();
        by_mass_reducer.sort_by(|left, right| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_mass_reducer.into_iter().take(fractional_pool_take(
            pool,
            self.config.exact_pool_mass_reducer_fraction,
        )) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_solve_prob = suggestions.iter().collect::<Vec<_>>();
        by_solve_prob.sort_by(|left, right| {
            right
                .solve_probability
                .total_cmp(&left.solve_probability)
                .then_with(|| left.word.cmp(&right.word))
        });
        for suggestion in by_solve_prob.into_iter().take(fractional_pool_take(
            pool,
            self.config.exact_pool_solve_probability_fraction,
        )) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let mut by_posterior = suggestions
            .iter()
            .filter(|suggestion| suggestion.posterior_answer_probability > 0.0)
            .collect::<Vec<_>>();
        by_posterior.sort_by(|left, right| {
            right
                .posterior_answer_probability
                .total_cmp(&left.posterior_answer_probability)
                .then_with(|| left.word.cmp(&right.word))
        });
        for suggestion in by_posterior.into_iter().take(fractional_pool_take(
            pool,
            self.config.exact_pool_posterior_fraction,
        )) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        for suggestion in suggestions
            .iter()
            .take(self.config.lookahead_root_force_in_two_scan.max(pool))
            .filter(|suggestion| suggestion.force_in_two)
        {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        for answer_index in &state.surviving {
            if let Some(guess_index) = self.guess_index.get(&self.answers[*answer_index].word) {
                push_candidate(*guess_index);
            }
        }

        let mut diversity_ranked = suggestions.iter().collect::<Vec<_>>();
        diversity_ranked.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| {
                    left.largest_non_green_bucket_mass
                        .total_cmp(&right.largest_non_green_bucket_mass)
                })
                .then_with(|| compare_suggestions(left, right))
        });
        let stride = self.config.pool_diversity_stride.max(1);
        for suggestion in diversity_ranked.into_iter().step_by(stride) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            push_candidate(guess_index);
        }

        let extra_limit = non_surviving_limit + surviving_guess_indexes.len();
        if candidate_indexes.len() > extra_limit {
            let mut trimmed = Vec::with_capacity(extra_limit);
            let mut extra_count = 0usize;
            for guess_index in candidate_indexes {
                if surviving_guess_indexes.contains(&guess_index) {
                    trimmed.push(guess_index);
                    continue;
                }
                if extra_count < non_surviving_limit {
                    trimmed.push(guess_index);
                    extra_count += 1;
                }
            }
            return Ok(trimmed);
        }

        Ok(candidate_indexes)
    }

    fn lookahead_cost_for_guess_controlled(
        &self,
        guess_index: usize,
        context: LookaheadCostContext<'_>,
        mut timing: Option<&mut SearchTiming>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<f64> {
        let LookaheadCostContext {
            subset,
            weights,
            expanded,
            exact_memo,
            exact_scratch,
            lookahead_memo,
        } = context;
        check_predictive_search_cancelled(cancelled)?;
        let (total_weight, _) = validated_subset_mass(subset, weights)?;

        let mut ordered_patterns = [0u8; PATTERN_SPACE];
        let ordered_len = {
            let frame = exact_scratch.frame_mut(0);
            for answer_index in subset {
                let pattern = self.answer_pattern(guess_index, *answer_index) as usize;
                if frame.child_subsets[pattern].is_empty() {
                    frame.touched_patterns.push(pattern as u8);
                }
                frame.masses[pattern] += weights[*answer_index];
                frame.child_subsets[pattern].push(*answer_index);
            }
            let len = frame.touched_patterns.len();
            ordered_patterns[..len].copy_from_slice(&frame.touched_patterns);
            len
        };

        let mut total_cost = 1.0;
        let mut worst_child_probability = 0.0_f64;
        let mut large_bucket_count = 0usize;
        let mut dangerous_mass_bucket_count = 0usize;
        let mut non_green_mass_in_large_buckets = 0.0_f64;
        for pattern in ordered_patterns[..ordered_len].iter().copied() {
            check_predictive_search_cancelled(cancelled)?;
            let mass = exact_scratch.frames[0].masses[pattern as usize];
            let probability = mass / total_weight;
            let child_len = exact_scratch.frames[0].child_subsets[pattern as usize].len();
            let child_value = if mass == 0.0 || pattern == ALL_GREEN_PATTERN {
                0.0
            } else {
                let child_subset =
                    std::mem::take(&mut exact_scratch.frames[0].child_subsets[pattern as usize]);
                let result =
                    if child_subset.len() == subset.len() && child_subset.as_slice() == subset {
                        Ok(f64::INFINITY)
                    } else {
                        self.lookahead_child_value_controlled(
                            &child_subset,
                            weights,
                            expanded,
                            exact_memo,
                            exact_scratch,
                            lookahead_memo,
                            timing.as_deref_mut(),
                            cancelled,
                        )
                    };
                exact_scratch.frames[0].child_subsets[pattern as usize] = child_subset;
                result?
            };
            total_cost += probability * child_value;
            if pattern != ALL_GREEN_PATTERN {
                worst_child_probability = worst_child_probability.max(probability);
                if child_len >= self.config.trap_size_threshold {
                    large_bucket_count += 1;
                    non_green_mass_in_large_buckets += probability;
                }
                if probability >= self.config.trap_mass_threshold {
                    dangerous_mass_bucket_count += 1;
                }
            }
        }
        check_predictive_search_cancelled(cancelled)?;
        Ok(total_cost
            + self.aggregate_lookahead_trap_penalty(
                worst_child_probability,
                large_bucket_count,
                dangerous_mass_bucket_count,
                non_green_mass_in_large_buckets,
            ))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "explicit existing recursion state plus cooperative cancellation"
    )]
    fn lookahead_child_value_controlled(
        &self,
        subset: &[usize],
        weights: &[f64],
        expanded: bool,
        exact_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        exact_scratch: &mut ExactSearchScratch,
        lookahead_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        mut timing: Option<&mut SearchTiming>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<f64> {
        check_predictive_search_cancelled(cancelled)?;
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() <= self.config.exact_exhaustive_threshold {
            let exact_started = timing.as_ref().map(|_| Instant::now());
            let result = self.exact_best_cost_controlled(
                subset,
                weights,
                exact_memo,
                exact_scratch,
                1,
                cancelled,
            );
            if let (Some(timing), Some(started)) = (timing.as_mut(), exact_started) {
                timing.exact_child += started.elapsed();
                timing.exact_child_count += 1;
            }
            return result;
        }

        let key = ExactSubsetKey::from_sorted_subset(subset);
        if let Some(cached) = lookahead_memo.get(&key) {
            return Ok(*cached);
        }

        let metric_scan_started = timing.as_ref().map(|_| Instant::now());
        let metrics_result =
            self.score_guess_metrics_for_subset_controlled(subset, weights, cancelled);
        if let (Some(timing), Some(started)) = (timing.as_mut(), metric_scan_started) {
            timing.large_child_metric_scan += started.elapsed();
            timing.large_child_metric_scan_count += 1;
        }
        let mut metrics = metrics_result?;
        metrics.retain(|metric| reply_guess_makes_progress(metric, subset.len()));
        if metrics.is_empty() {
            bail!("bounded lookahead found no progressing reply guess");
        }
        let split_first = subset.len() > self.config.large_state_split_threshold;
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let (total_weight, _) = validated_subset_mass(subset, weights)?;
        let assessment = self.assess_subset_danger(subset, weights, total_weight, &metrics);
        let reply_pool = self.lookahead_reply_pool_for_state(subset.len()).max(1)
            + if expanded || assessment.dangerous_lookahead {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let mut reply_candidates = Vec::new();
        let mut seen = HashSet::new();
        for metric in metrics.iter().take(reply_pool) {
            check_predictive_search_cancelled(cancelled)?;
            if seen.insert(metric.guess_index) {
                reply_candidates.push(*metric);
            }
        }
        let mut by_worst_bucket = metrics.iter().collect::<Vec<_>>();
        by_worst_bucket.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| compare_guess_metrics(left, right, &self.guesses))
        });
        for metric in by_worst_bucket.into_iter().take(reply_pool) {
            check_predictive_search_cancelled(cancelled)?;
            if seen.insert(metric.guess_index) {
                reply_candidates.push(*metric);
            }
        }
        let mut by_mass_reducer = metrics.iter().collect::<Vec<_>>();
        by_mass_reducer.sort_by(|left, right| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
                .then_with(|| compare_guess_metrics(left, right, &self.guesses))
        });
        for metric in by_mass_reducer.into_iter().take(reply_pool) {
            check_predictive_search_cancelled(cancelled)?;
            if seen.insert(metric.guess_index) {
                reply_candidates.push(*metric);
            }
        }
        let mut best_reply = f64::INFINITY;
        for metric in reply_candidates {
            check_predictive_search_cancelled(cancelled)?;
            best_reply = best_reply
                .min(metric.proxy_cost + self.lookahead_reply_penalty(&metric, subset.len()));
        }
        // `proxy_cost` already includes the reply guess (`1 + E[child]`).
        // Adding another unit here would count that reply twice and make the
        // heuristic branch discontinuous with the exact branch above.
        check_predictive_search_cancelled(cancelled)?;
        lookahead_memo.insert(key, best_reply);
        Ok(best_reply)
    }

    pub(super) fn exact_root_cost_lower_bound(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
    ) -> Result<f64> {
        if subset.is_empty() {
            return Ok(0.0);
        }
        let (total_weight, _) = validated_subset_mass(subset, weights)?;
        let mut masses = [0.0; PATTERN_SPACE];
        let mut largest_weights = [0.0_f64; PATTERN_SPACE];
        for answer_index in subset {
            let pattern = self.answer_pattern(guess_index, *answer_index) as usize;
            let weight = weights[*answer_index];
            masses[pattern] += weight;
            largest_weights[pattern] = largest_weights[pattern].max(weight);
        }
        // Each non-green child needs at least one reply; all its mass except
        // the heaviest answer needs at least one further guess after that.
        // Normalize before summing so valid masses near f64::MAX cannot overflow.
        let remaining_cost = (0..PATTERN_SPACE)
            .filter(|pattern| *pattern != ALL_GREEN_PATTERN as usize)
            .map(|pattern| {
                masses[pattern] / total_weight
                    + (masses[pattern] - largest_weights[pattern]) / total_weight
            })
            .sum::<f64>();
        Ok(1.0 + remaining_cost)
    }

    pub(super) fn exact_cost_for_guess_controlled(
        &self,
        guess_index: usize,
        context: ExactCostContext<'_>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<f64> {
        let ExactCostContext {
            subset,
            weights,
            memo,
            best_bound,
            scratch,
            depth,
        } = context;
        check_predictive_search_cancelled(cancelled)?;
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() == 1 && self.guesses[guess_index] == self.answers[subset[0]].word {
            return Ok(1.0);
        }

        let (total_weight, _) = validated_subset_mass(subset, weights)?;
        let mut ordered_patterns = [0u8; PATTERN_SPACE];
        let ordered_len = {
            let frame = scratch.frame_mut(depth);
            for answer_index in subset {
                let pattern = self.answer_pattern(guess_index, *answer_index) as usize;
                if frame.child_subsets[pattern].is_empty() {
                    frame.touched_patterns.push(pattern as u8);
                }
                frame.masses[pattern] += weights[*answer_index];
                frame.child_subsets[pattern].push(*answer_index);
            }
            let len = frame.touched_patterns.len();
            ordered_patterns[..len].copy_from_slice(&frame.touched_patterns);
            let masses = &frame.masses;
            ordered_patterns[..len]
                .sort_by(|left, right| masses[*right as usize].total_cmp(&masses[*left as usize]));
            len
        };

        let mut cost = 1.0;
        for pattern in ordered_patterns[..ordered_len].iter().copied() {
            check_predictive_search_cancelled(cancelled)?;
            let mass = scratch.frames[depth].masses[pattern as usize];
            if mass == 0.0 {
                continue;
            }
            let branch_probability = mass / total_weight;
            let child_cost = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else {
                let child_subset =
                    std::mem::take(&mut scratch.frames[depth].child_subsets[pattern as usize]);
                let result =
                    if child_subset.len() == subset.len() && child_subset.as_slice() == subset {
                        Ok(f64::INFINITY)
                    } else {
                        self.exact_best_cost_controlled(
                            &child_subset,
                            weights,
                            memo,
                            scratch,
                            depth + 1,
                            cancelled,
                        )
                    };
                scratch.frames[depth].child_subsets[pattern as usize] = child_subset;
                result?
            };
            cost += branch_probability * child_cost;
            if cost >= best_bound {
                return Ok(cost);
            }
        }

        check_predictive_search_cancelled(cancelled)?;
        Ok(cost)
    }

    fn exact_best_cost_controlled(
        &self,
        subset: &[usize],
        weights: &[f64],
        memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        scratch: &mut ExactSearchScratch,
        depth: usize,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<f64> {
        check_predictive_search_cancelled(cancelled)?;
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() == 1 {
            return Ok(1.0);
        }

        let key = ExactSubsetKey::from_sorted_subset(subset);
        if let Some(cached) = memo.get(&key) {
            return Ok(*cached);
        }

        let suggestion_mode = exact_suggestion_mode(&self.config, subset.len());
        let scores = match suggestion_mode {
            Some(ExactSuggestionMode::Exhaustive) => {
                let mut scores = (0..self.guesses.len()).collect::<Vec<_>>();
                // Seed the incumbent with a legal answer guess carrying the most mass.
                if let Some(guess_index) = subset
                    .iter()
                    .copied()
                    .filter_map(|answer_index| {
                        let weight = weights.get(answer_index).copied()?;
                        if !weight.is_finite() || weight <= 0.0 {
                            return None;
                        }
                        let guess_index = self
                            .answers
                            .get(answer_index)
                            .and_then(|answer| self.guess_index.get(&answer.word))
                            .copied()?;
                        Some((weight, guess_index))
                    })
                    .max_by(|left, right| {
                        left.0
                            .total_cmp(&right.0)
                            .then_with(|| right.1.cmp(&left.1))
                    })
                    .map(|(_, guess_index)| guess_index)
                    && let Some(position) = scores.iter().position(|index| *index == guess_index)
                {
                    scores.swap(0, position);
                }
                scores
            }
            Some(ExactSuggestionMode::Pooled) | None => {
                scratch.used_candidate_pool = true;
                self.top_guess_indexes_for_subset_controlled(
                    subset,
                    weights,
                    self.config.exact_candidate_pool,
                    cancelled,
                )?
            }
        };
        let lower_bound = weighted_exact_lower_bound(subset, weights)?;
        for answer_index in subset {
            debug_assert!(u16::try_from(*answer_index).is_ok());
        }
        let mut best_cost = f64::INFINITY;
        for guess_index in scores.iter().copied() {
            check_predictive_search_cancelled(cancelled)?;
            if best_cost.is_finite()
                && self.exact_root_cost_lower_bound(guess_index, subset, weights)?
                    > best_cost + 1e-10
            {
                continue;
            }
            let cost = self.exact_cost_for_guess_controlled(
                guess_index,
                ExactCostContext {
                    subset,
                    weights,
                    memo,
                    best_bound: best_cost,
                    scratch,
                    depth,
                },
                cancelled,
            )?;
            if cost < best_cost {
                best_cost = cost;
                if best_cost <= lower_bound {
                    break;
                }
            }
        }
        if !best_cost.is_finite()
            && matches!(suggestion_mode, Some(ExactSuggestionMode::Pooled) | None)
        {
            let shortlisted = scores.into_iter().collect::<HashSet<_>>();
            for guess_index in 0..self.guesses.len() {
                check_predictive_search_cancelled(cancelled)?;
                if shortlisted.contains(&guess_index) {
                    continue;
                }
                if best_cost.is_finite()
                    && self.exact_root_cost_lower_bound(guess_index, subset, weights)?
                        > best_cost + 1e-10
                {
                    continue;
                }
                let cost = self.exact_cost_for_guess_controlled(
                    guess_index,
                    ExactCostContext {
                        subset,
                        weights,
                        memo,
                        best_bound: best_cost,
                        scratch,
                        depth,
                    },
                    cancelled,
                )?;
                if cost < best_cost {
                    best_cost = cost;
                    if best_cost <= lower_bound {
                        break;
                    }
                }
            }
        }
        if !best_cost.is_finite() {
            bail!(
                "no valid exact guess found for subset of size {}",
                subset.len()
            );
        }
        check_predictive_search_cancelled(cancelled)?;
        memo.insert(key, best_cost);
        Ok(best_cost)
    }

    fn top_guess_indexes_for_subset_controlled(
        &self,
        subset: &[usize],
        weights: &[f64],
        count: usize,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<usize>> {
        let mut metrics =
            self.score_guess_metrics_for_subset_controlled(subset, weights, cancelled)?;
        let split_first = subset.len() > self.config.large_state_split_threshold;
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let surviving_guess_indexes = subset
            .iter()
            .filter_map(|answer_index| self.guess_index.get(&self.answers[*answer_index].word))
            .copied()
            .collect::<HashSet<_>>();
        let mut selected = Vec::new();
        let mut seen = HashSet::new();
        for metric in metrics.iter().take(count) {
            check_predictive_search_cancelled(cancelled)?;
            if seen.insert(metric.guess_index) {
                selected.push(metric.guess_index);
            }
        }
        for metric in metrics.into_iter().skip(count) {
            check_predictive_search_cancelled(cancelled)?;
            if surviving_guess_indexes.contains(&metric.guess_index)
                && seen.insert(metric.guess_index)
            {
                selected.push(metric.guess_index);
            }
        }
        Ok(selected)
    }

    #[cfg(test)]
    pub(super) fn lookahead_cost_for_guess(
        &self,
        guess_index: usize,
        context: LookaheadCostContext<'_>,
    ) -> Result<f64> {
        self.lookahead_cost_for_guess_controlled(guess_index, context, None, &|| false)
    }

    #[cfg(test)]
    pub(super) fn lookahead_child_value(
        &self,
        subset: &[usize],
        weights: &[f64],
        expanded: bool,
        exact_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        exact_scratch: &mut ExactSearchScratch,
        lookahead_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
    ) -> Result<f64> {
        self.lookahead_child_value_controlled(
            subset,
            weights,
            expanded,
            exact_memo,
            exact_scratch,
            lookahead_memo,
            None,
            &|| false,
        )
    }

    pub(super) fn aggregate_lookahead_trap_penalty(
        &self,
        worst_branch_mass: f64,
        large_bucket_count: usize,
        dangerous_mass_bucket_count: usize,
        non_green_mass_in_large_buckets: f64,
    ) -> f64 {
        (self.config.lookahead_trap_penalty * worst_branch_mass)
            + (self.config.lookahead_large_bucket_penalty * large_bucket_count as f64)
            + (self.config.lookahead_dangerous_mass_penalty * dangerous_mass_bucket_count as f64)
            + (self.config.lookahead_large_bucket_mass_penalty * non_green_mass_in_large_buckets)
    }

    pub(super) fn lookahead_reply_penalty(&self, metric: &GuessMetrics, subset_len: usize) -> f64 {
        let bucket_ratio = metric.worst_non_green_bucket_size as f64 / subset_len.max(1) as f64;
        self.aggregate_lookahead_trap_penalty(
            metric.largest_non_green_bucket_mass,
            metric.large_non_green_bucket_count,
            metric.dangerous_mass_bucket_count,
            metric.non_green_mass_in_large_buckets,
        ) + (self.config.lookahead_worst_bucket_ratio_penalty * bucket_ratio)
    }

    pub(super) fn collect_lookahead_candidates(
        &self,
        suggestions: &[Suggestion],
        surviving_answers: usize,
        expanded: bool,
        base_pool: usize,
    ) -> Result<Vec<usize>> {
        let pool = base_pool.max(1)
            + if expanded {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let force_scan = self.force_in_two_scan_for_state(surviving_answers).max(1)
            + if expanded {
                self.config.danger_reply_pool_bonus
            } else {
                0
            };
        let diversity_take = pool.div_ceil(self.config.pool_diversity_stride.max(1));
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        for suggestion in suggestions.iter().take(pool) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        for suggestion in suggestions
            .iter()
            .take(force_scan)
            .filter(|suggestion| suggestion.force_in_two)
        {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        let stride = self.config.pool_diversity_stride.max(1);
        let mut by_entropy = suggestions.iter().collect::<Vec<_>>();
        by_entropy.sort_by(|left, right| {
            right
                .entropy
                .total_cmp(&left.entropy)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_entropy.into_iter().step_by(stride).take(diversity_take) {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        let mut by_worst_bucket = suggestions.iter().collect::<Vec<_>>();
        by_worst_bucket.sort_by(|left, right| {
            left.worst_non_green_bucket_size
                .cmp(&right.worst_non_green_bucket_size)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_worst_bucket
            .into_iter()
            .step_by(stride)
            .take(diversity_take)
        {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        let mut by_mass_reducer = suggestions.iter().collect::<Vec<_>>();
        by_mass_reducer.sort_by(|left, right| {
            left.largest_non_green_bucket_mass
                .total_cmp(&right.largest_non_green_bucket_mass)
                .then_with(|| compare_suggestions(left, right))
        });
        for suggestion in by_mass_reducer
            .into_iter()
            .step_by(stride)
            .take(diversity_take)
        {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        for suggestion in suggestions
            .iter()
            .skip(stride - 1)
            .step_by(stride)
            .take(diversity_take)
        {
            let guess_index = self
                .guess_index
                .get(&suggestion.word)
                .copied()
                .with_context(|| format!("missing guess {}", suggestion.word))?;
            if seen.insert(guess_index) {
                candidates.push(guess_index);
            }
        }
        Ok(candidates)
    }

    #[cfg(test)]
    pub(super) fn exact_cost_for_guess(
        &self,
        guess_index: usize,
        context: ExactCostContext<'_>,
    ) -> Result<f64> {
        self.exact_cost_for_guess_controlled(guess_index, context, &|| false)
    }

    #[cfg(test)]
    pub(super) fn exact_best_cost(
        &self,
        subset: &[usize],
        weights: &[f64],
        memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        scratch: &mut ExactSearchScratch,
        depth: usize,
    ) -> Result<f64> {
        self.exact_best_cost_controlled(subset, weights, memo, scratch, depth, &|| false)
    }

    #[cfg(test)]
    pub(super) fn top_guess_indexes_for_subset(
        &self,
        subset: &[usize],
        weights: &[f64],
        count: usize,
    ) -> Vec<usize> {
        self.top_guess_indexes_for_subset_controlled(subset, weights, count, &|| false)
            .expect("non-cancellable subset ranking cannot fail")
    }
}

pub(super) fn reply_guess_makes_progress(metric: &GuessMetrics, subset_len: usize) -> bool {
    metric.worst_non_green_bucket_size < subset_len
}

pub(super) fn scaled_pool_size(base: usize, multiplier: f64) -> usize {
    ((base as f64) * multiplier).floor() as usize
}

pub(super) fn fractional_pool_take(pool: usize, fraction: f64) -> usize {
    ((pool as f64) * fraction).floor() as usize
}

pub(super) fn weighted_exact_lower_bound(subset: &[usize], weights: &[f64]) -> Result<f64> {
    if subset.is_empty() {
        return Ok(0.0);
    }
    let (total_weight, largest_weight) = validated_subset_mass(subset, weights)?;
    // A guess can solve at most one distinct answer. Every other outcome needs at
    // least one additional guess, so 1 + P(not solved immediately) is admissible
    // for arbitrary non-uniform predictive masses.
    Ok(1.0 + ((total_weight - largest_weight) / total_weight))
}

fn validated_subset_mass(subset: &[usize], weights: &[f64]) -> Result<(f64, f64)> {
    let mut total_weight = 0.0;
    let mut largest_weight = 0.0_f64;
    for answer_index in subset {
        let weight = weights.get(*answer_index).copied().ok_or_else(|| {
            anyhow!("predictive-search answer index {answer_index} is out of range")
        })?;
        if !weight.is_finite() || weight < 0.0 {
            bail!("predictive-search weights must be finite and non-negative");
        }
        total_weight += weight;
        largest_weight = largest_weight.max(weight);
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        bail!("cannot evaluate predictive search on a zero-mass subset");
    }
    Ok((total_weight, largest_weight))
}

pub(super) fn exact_suggestion_mode(
    config: &PriorConfig,
    surviving_answers: usize,
) -> Option<ExactSuggestionMode> {
    if surviving_answers > config.exact_threshold {
        return None;
    }
    if surviving_answers
        <= config
            .exact_exhaustive_threshold
            .min(config.exact_threshold)
    {
        Some(ExactSuggestionMode::Exhaustive)
    } else {
        Some(ExactSuggestionMode::Pooled)
    }
}

pub(super) fn should_use_second_guess_coverage(
    config: &PriorConfig,
    surviving_answers: usize,
    observation_count: usize,
) -> bool {
    observation_count == 1
        && surviving_answers >= config.second_guess_coverage_min_survivors
        && surviving_answers <= config.second_guess_coverage_max_survivors
}

pub(super) fn should_use_final_turn_objective(observation_count: usize) -> bool {
    observation_count >= 5
}

pub(super) fn book_usage_for_mode(mode: PredictiveSuggestionMode) -> PredictiveBookUsage {
    match mode {
        PredictiveSuggestionMode::LiveOnly => PredictiveBookUsage::None,
        PredictiveSuggestionMode::FastDiskOnly => PredictiveBookUsage::DiskOnly,
        PredictiveSuggestionMode::Full => PredictiveBookUsage::Full,
    }
}

pub(super) fn predictive_search_mode(
    config: &PriorConfig,
    surviving_answers: usize,
    assessment: StateDangerAssessment,
) -> PredictiveSearchMode {
    match config.search_policy_mode {
        crate::config::SearchPolicyMode::ProxyOnly => return PredictiveSearchMode::ProxyOnly,
        crate::config::SearchPolicyMode::ProxyWithExactEndgame => {
            return exact_suggestion_mode(config, surviving_answers)
                .map(PredictiveSearchMode::Exact)
                .unwrap_or(PredictiveSearchMode::ProxyOnly);
        }
        crate::config::SearchPolicyMode::Staged
        | crate::config::SearchPolicyMode::StagedFixedBelief => {}
        crate::config::SearchPolicyMode::FiniteFast
        | crate::config::SearchPolicyMode::FiniteFastDynamic
        | crate::config::SearchPolicyMode::FiniteFastFixedWork
        | crate::config::SearchPolicyMode::FiniteBaseline
        | crate::config::SearchPolicyMode::FiniteStrong => {
            return PredictiveSearchMode::Lookahead;
        }
    }
    if let Some(mode) = exact_suggestion_mode(config, surviving_answers) {
        PredictiveSearchMode::Exact(mode)
    } else if surviving_answers <= config.danger_exact_survivor_cap
        && surviving_answers > config.exact_threshold
        && assessment.dangerous_exact
    {
        PredictiveSearchMode::EscalatedExact
    } else if surviving_answers <= config.lookahead_threshold
        || (surviving_answers <= config.danger_exact_survivor_cap && assessment.dangerous_lookahead)
    {
        PredictiveSearchMode::Lookahead
    } else {
        PredictiveSearchMode::ProxyOnly
    }
}

#[cfg(test)]
mod terminal_routing_tests {
    use super::*;

    #[test]
    fn last_two_turns_skip_unlimited_horizon_refinement() {
        let solver =
            crate::solver::tests::test_solver(&["tower", "power", "bower", "rower", "sower"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let state = solver.initial_state(date);
        for turns in [1, 2] {
            let history = vec![("xxxxx".to_owned(), 0); 6 - turns];
            for hard_mode in [false, true] {
                let batch = solver
                    .suggestion_batch_internal_with_search_mode_controlled(
                        &state,
                        5,
                        Some(PredictiveContext {
                            hard_mode,
                            as_of: date,
                            observations: &history,
                        }),
                        PredictiveBookUsage::None,
                        Some(PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive)),
                        &|| false,
                    )
                    .unwrap();
                assert_eq!(batch.suggestions.len(), 5);
                assert_eq!(batch.root_candidate_count, 5);
                assert!(
                    batch
                        .suggestions
                        .iter()
                        .all(|row| row.exact_cost.is_none() && row.lookahead_cost.is_none())
                );
                assert_eq!(batch.exact_pool_size, 0);
                assert_eq!(batch.lookahead_pool_size, 0);
                assert_eq!(batch.execution.route, PredictiveRegime::Terminal);
                assert_eq!(
                    batch.execution.objective,
                    SearchObjective::TerminalSolveProbability
                );
                assert_eq!(
                    batch.execution.action_scope,
                    if hard_mode {
                        SearchActionScope::HardRecursive
                    } else {
                        SearchActionScope::Normal
                    }
                );
                assert_eq!(
                    batch.execution.candidate_scope,
                    SearchCandidateScope::AllActions
                );
                assert_eq!(
                    batch.execution.selected_value_kind,
                    Some(SuggestionValueKind::Terminal)
                );
                assert!(batch.execution.root_selection_optimal);
            }
        }
    }

    #[test]
    fn terminal_success_matches_dynamic_finite_values_and_is_cancellable() {
        let mut solver = crate::solver::tests::test_solver_with_answer_count(
            &["aaaaa", "bbbbb", "cdddd", "cffff", "cgggg", "ccccc"],
            5,
        );
        solver.data_mut().primary_answer_count = 2;
        solver.config.fallback_activation_threshold = 1;
        solver.config.fallback_prior_mass = 0.4;
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let state = solver.initial_state(date);
        let history = vec![("xxxxx".to_owned(), 0); 4];
        for hard_mode in [false, true] {
            let context = PredictiveContext {
                hard_mode,
                as_of: date,
                observations: &history,
            };
            let scores = solver
                .terminal_two_turn_success(&state, context, &|| false)
                .unwrap();
            let exact = solver
                .finite_horizon_search_dynamic(
                    &state,
                    &history,
                    2,
                    hard_mode,
                    FiniteSearchOptions {
                        root_shortlist: 6,
                        reply_shortlist: 6,
                        exact_state_threshold: 8,
                        budget: Duration::from_secs(5),
                        node_limit: None,
                        baseline_only: false,
                    },
                    &|| false,
                )
                .unwrap();
            assert_eq!(exact.reason, FiniteSearchReason::Complete);
            for row in exact.candidates {
                assert!(
                    (scores[row.guess_index].0 - (1.0 - row.failure_probability)).abs() < 1e-12
                );
            }
            assert!(scores[solver.guess_index["aaaaa"]].0 < 1.0);
            assert!(!scores[solver.guess_index["aaaaa"]].1);
            let calls = std::sync::atomic::AtomicUsize::new(0);
            let error = solver
                .terminal_two_turn_success(&state, context, &|| {
                    calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 3
                })
                .expect_err("cancel inside the child scan");
            assert!(error.to_string().contains("cancel"));
            assert_eq!(
                scores,
                solver
                    .terminal_two_turn_success(&state, context, &|| false)
                    .unwrap()
            );
        }
    }

    #[test]
    fn terminal_search_rejects_unguessable_answers() {
        let mut solver = crate::solver::tests::test_solver(&["cigar", "rebut"]);
        solver.data_mut().guess_index.remove("rebut");
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let state = solver.initial_state(date);
        let history = vec![("xxxxx".to_owned(), 0); 5];
        let error = solver
            .terminal_suggestion_batch(
                &state,
                2,
                PredictiveContext {
                    hard_mode: false,
                    as_of: date,
                    observations: &history,
                },
                &|| false,
            )
            .unwrap_err();
        assert!(error.to_string().contains("legal guess"));
    }
}

pub(super) fn regime_from_search_mode(search_mode: PredictiveSearchMode) -> PredictiveRegime {
    match search_mode {
        PredictiveSearchMode::ProxyOnly => PredictiveRegime::Proxy,
        PredictiveSearchMode::Lookahead => PredictiveRegime::Lookahead,
        PredictiveSearchMode::EscalatedExact => PredictiveRegime::EscalatedExact,
        PredictiveSearchMode::Exact(_) => PredictiveRegime::Exact,
    }
}

#[cfg(test)]
mod execution_metadata_tests {
    use super::*;

    fn run(
        solver: &Solver,
        mode: PredictiveSearchMode,
        hard: bool,
        observations: &[(String, u8)],
    ) -> SuggestionBatch {
        let date = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        solver
            .suggestion_batch_internal_with_search_mode_controlled(
                &solver.initial_state(date),
                solver.guesses.len(),
                Some(PredictiveContext {
                    hard_mode: hard,
                    as_of: date,
                    observations,
                }),
                PredictiveBookUsage::None,
                Some(mode),
                &|| false,
            )
            .unwrap()
    }

    #[test]
    fn forced_routes_are_reported_instead_of_inferred_from_small_state_size() {
        let solver =
            crate::solver::tests::test_solver(&["tower", "power", "bower", "rower", "sower"]);
        for (mode, route, objective, kind, scope, optimal) in [
            (
                PredictiveSearchMode::ProxyOnly,
                PredictiveRegime::Proxy,
                SearchObjective::ProxyRanking,
                SuggestionValueKind::Proxy,
                SearchCandidateScope::AllActions,
                false,
            ),
            (
                PredictiveSearchMode::Lookahead,
                PredictiveRegime::Lookahead,
                SearchObjective::PenalizedLookahead,
                SuggestionValueKind::Lookahead,
                SearchCandidateScope::CandidatePool,
                false,
            ),
            (
                PredictiveSearchMode::EscalatedExact,
                PredictiveRegime::EscalatedExact,
                SearchObjective::ExpectedGuesses,
                SuggestionValueKind::ExactAction,
                SearchCandidateScope::CandidatePool,
                false,
            ),
            (
                PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive),
                PredictiveRegime::Exact,
                SearchObjective::ExpectedGuesses,
                SuggestionValueKind::ExactAction,
                SearchCandidateScope::AllActions,
                true,
            ),
            (
                PredictiveSearchMode::Exact(ExactSuggestionMode::Pooled),
                PredictiveRegime::Exact,
                SearchObjective::ExpectedGuesses,
                SuggestionValueKind::ExactAction,
                SearchCandidateScope::CandidatePool,
                false,
            ),
        ] {
            let batch = run(&solver, mode, false, &[]);
            assert_eq!(batch.execution.route, route);
            assert_eq!(batch.execution.objective, objective);
            assert_eq!(batch.execution.selected_value_kind, Some(kind));
            assert_eq!(batch.execution.candidate_scope, scope);
            assert_eq!(batch.execution.root_selection_optimal, optimal);
            assert_eq!(batch.execution.action_scope, SearchActionScope::Normal);
            assert_eq!(
                batch.execution.roots_evaluated,
                batch.execution.roots_considered
            );
            assert!(batch.execution.roots_evaluated > 0);
            assert_eq!(
                batch.execution.selected_value_kind,
                batch.suggestions.first().map(|row| row.value_kind)
            );
        }
    }

    #[test]
    fn exhaustive_roots_do_not_certify_pooled_continuations() {
        let mut solver =
            crate::solver::tests::test_solver(&["tower", "power", "bower", "rower", "sower"]);
        solver.config.exact_exhaustive_threshold = 1;
        solver.config.exact_candidate_pool = 1;
        let batch = run(
            &solver,
            PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive),
            false,
            &[],
        );
        assert_eq!(
            batch.execution.candidate_scope,
            SearchCandidateScope::AllActions
        );
        assert_eq!(
            batch.execution.selected_value_kind,
            Some(SuggestionValueKind::ContinuationEstimate)
        );
        assert!(batch.suggestions.iter().all(|row| row.exact_cost.is_some()
            && row.value_kind == SuggestionValueKind::ContinuationEstimate));
        assert!(!batch.execution.root_selection_optimal);
        assert!(
            batch
                .execution
                .summary()
                .contains("pooled continuation estimate")
        );
    }

    #[test]
    fn hard_root_and_coverage_objectives_do_not_claim_global_expected_cost_optimality() {
        let mut solver =
            crate::solver::tests::test_solver(&["tower", "power", "bower", "rower", "sower"]);
        let hard = run(
            &solver,
            PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive),
            true,
            &[],
        );
        assert_eq!(
            hard.execution.action_scope,
            SearchActionScope::HardRootNormalContinuation
        );
        assert_eq!(
            hard.execution.selected_value_kind,
            Some(SuggestionValueKind::ExactAction)
        );
        assert!(!hard.execution.root_selection_optimal);
        solver.config.second_guess_coverage_min_survivors = 1;
        solver.config.second_guess_coverage_max_survivors = 10;
        let coverage = run(
            &solver,
            PredictiveSearchMode::Exact(ExactSuggestionMode::Exhaustive),
            false,
            &[("xxxxx".into(), 0)],
        );
        assert_eq!(
            coverage.execution.objective,
            SearchObjective::ThreeSolveCoverageThenCost
        );
        assert!(!coverage.execution.root_selection_optimal);
    }
}

#[cfg(test)]
mod search_timing_tests {
    use super::*;

    #[test]
    fn formats_numeric_search_timing_without_words() {
        let timing = SearchTiming {
            base_metrics: std::time::Duration::from_millis(2),
            second_guess_coverage: std::time::Duration::from_millis(3),
            lookahead_root: std::time::Duration::from_millis(5),
            lookahead_root_count: 7,
            exact_child: std::time::Duration::from_millis(11),
            exact_child_count: 13,
            large_child_metric_scan: std::time::Duration::from_millis(17),
            large_child_metric_scan_count: 19,
        };

        assert_eq!(
            timing.format_line(),
            "benchmark-evidence search-timing base_ms=2 coverage_ms=3 lookahead_root_ms=5 lookahead_root_count=7 exact_child_ms=11 exact_child_count=13 large_child_metric_scan_ms=17 large_child_metric_scan_count=19"
        );
    }
}
