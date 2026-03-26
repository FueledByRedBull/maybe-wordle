use super::*;

impl Solver {
    pub(super) fn suggestion_batch_for_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        state: &SolveState,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<SuggestionBatch> {
        self.suggestion_batch_internal(
            state,
            top,
            Some(PredictiveContext {
                as_of,
                observations,
            }),
            book_usage,
        )
    }

    pub(super) fn filtered_suggestion_batch_for_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
        mode: PredictiveSuggestionMode,
        hard_mode: bool,
        force_in_two_only: bool,
    ) -> Result<SuggestionBatch> {
        let state = self.apply_history(as_of, observations)?;
        let limit = if hard_mode || force_in_two_only {
            self.guesses.len()
        } else {
            top
        };
        let mut batch = self.suggestion_batch_internal(
            &state,
            limit,
            Some(PredictiveContext {
                as_of,
                observations,
            }),
            book_usage_for_mode(mode),
        )?;
        if hard_mode {
            batch.suggestions.retain(|suggestion| {
                self.hard_mode_violation(observations, &suggestion.word)
                    .is_none()
            });
        }
        if force_in_two_only {
            batch
                .suggestions
                .retain(|suggestion| suggestion.force_in_two);
        }
        batch.suggestions.truncate(top.min(batch.suggestions.len()));
        Ok(batch)
    }

    pub(super) fn suggestion_batch_internal(
        &self,
        state: &SolveState,
        top: usize,
        context: Option<PredictiveContext<'_>>,
        book_usage: PredictiveBookUsage,
    ) -> Result<SuggestionBatch> {
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        if state.total_weight <= 0.0 {
            bail!("cannot score guesses when no positive answer mass remains");
        }
        let split_first = state.surviving.len() > self.config.large_state_split_threshold;
        let medium_second_guess = context
            .as_ref()
            .is_some_and(|context| context.observations.len() == 1)
            && self.is_medium_state_lookahead(state.surviving.len());
        let known_absent_mask = context
            .as_ref()
            .map(|context| known_absent_letter_mask(context.observations))
            .unwrap_or(0);
        let mut metrics = self.score_guess_metrics_for_subset(
            &state.surviving,
            &state.weights,
            &self.exact_small_state_table,
        );
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
        let three_solve_coverage = if medium_second_guess {
            Some(self.medium_second_guess_coverage(&state.surviving, &state.weights, &metrics)?)
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
        let search_mode = predictive_search_mode(&self.config, state.surviving.len(), assessment);
        let mut suggestions = metrics
            .into_iter()
            .map(|metric| Suggestion {
                word: self.guesses[metric.guess_index].clone(),
                entropy: metric.entropy,
                solve_probability: metric.solve_probability,
                expected_remaining: metric.expected_remaining,
                force_in_two: metric.force_in_two,
                known_absent_letter_hits: metric.known_absent_letter_hits,
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
            })
            .collect::<Vec<_>>();
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

        if let PredictiveSearchMode::Lookahead = search_mode {
            let root_candidates = self.collect_lookahead_candidates(
                &suggestions,
                state.surviving.len(),
                assessment.dangerous_lookahead,
                lookahead_pool,
            )?;
            root_candidate_count = root_candidates.len();
            let mut exact_memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut lookahead_memo = PredictiveMemoMap::default();
            let mut lookahead_costs = vec![None; self.guesses.len()];

            for guess_index in root_candidates {
                let cost = self.lookahead_cost_for_guess(
                    guess_index,
                    LookaheadCostContext {
                        subset: &state.surviving,
                        weights: &state.weights,
                        expanded: assessment.dangerous_lookahead,
                        exact_memo: &mut exact_memo,
                        exact_scratch: &mut exact_scratch,
                        lookahead_memo: &mut lookahead_memo,
                    },
                )?;
                lookahead_costs[guess_index] = Some(cost);
            }

            for suggestion in &mut suggestions {
                suggestion.lookahead_cost = self
                    .guess_index
                    .get(&suggestion.word)
                    .and_then(|guess_index| lookahead_costs[*guess_index]);
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
                    .then_with(|| compare_lookahead(left, right, split_first))
                } else {
                    compare_lookahead(left, right, split_first)
                }
            });
        }

        if let PredictiveSearchMode::Exact(exact_mode) = search_mode {
            let exact_candidates = match exact_mode {
                ExactSuggestionMode::Exhaustive => (0..self.guesses.len()).collect::<Vec<_>>(),
                ExactSuggestionMode::Pooled => {
                    self.collect_exact_candidates(state, &suggestions, exact_pool)?
                }
            };
            root_candidate_count = exact_candidates.len();
            let mut memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut exact_costs = vec![None; self.guesses.len()];

            for guess_index in exact_candidates {
                let cost = self.exact_cost_for_guess(
                    guess_index,
                    ExactCostContext {
                        subset: &state.surviving,
                        weights: &state.weights,
                        small_state_table: &self.exact_small_state_table,
                        memo: &mut memo,
                        best_bound: f64::INFINITY,
                        scratch: &mut exact_scratch,
                        depth: 0,
                    },
                )?;
                exact_costs[guess_index] = Some(cost);
            }

            match exact_mode {
                ExactSuggestionMode::Exhaustive => {
                    for suggestion in &mut suggestions {
                        suggestion.exact_cost = self
                            .guess_index
                            .get(&suggestion.word)
                            .and_then(|guess_index| exact_costs[*guess_index]);
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
            let exact_candidates = self.collect_exact_candidates(
                state,
                &suggestions,
                self.config.danger_exact_root_pool.max(1).max(exact_pool),
            )?;
            root_candidate_count = exact_candidates.len();
            let mut memo = PredictiveMemoMap::default();
            let mut exact_scratch = ExactSearchScratch::new();
            let mut exact_costs = vec![None; self.guesses.len()];

            for guess_index in exact_candidates {
                let cost = self.exact_cost_for_guess(
                    guess_index,
                    ExactCostContext {
                        subset: &state.surviving,
                        weights: &state.weights,
                        small_state_table: &self.exact_small_state_table,
                        memo: &mut memo,
                        best_bound: f64::INFINITY,
                        scratch: &mut exact_scratch,
                        depth: 0,
                    },
                )?;
                exact_costs[guess_index] = Some(cost);
            }

            for suggestion in &mut suggestions {
                suggestion.exact_cost = self
                    .guess_index
                    .get(&suggestion.word)
                    .and_then(|guess_index| exact_costs[*guess_index]);
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
        if book_usage != PredictiveBookUsage::None
            && let Some(context) = context
            && let Some(choice) = self.cached_predictive_choice(
                context.as_of,
                context.observations,
                book_usage == PredictiveBookUsage::Full,
            )
        {
            promote_cached_suggestion(&mut suggestions, &choice.word);
            promoted_word = Some(choice.word);
            promotion_source = Some(choice.source);
        }

        suggestions.truncate(top);
        Ok(SuggestionBatch {
            suggestions,
            promoted_word,
            promotion_source,
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
        })
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
            base.saturating_mul(5) / 2
        } else if gap < self.config.pool_medium_gap_threshold {
            base.saturating_mul(3) / 2
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

        for suggestion in suggestions.iter().take(pool / 2) {
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
        for suggestion in by_entropy.into_iter().take(pool / 4) {
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
        for suggestion in by_worst_bucket.into_iter().take(pool / 6) {
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
        for suggestion in by_mass_reducer.into_iter().take(pool / 6) {
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
        for suggestion in by_solve_prob.into_iter().take(pool / 8) {
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
        for suggestion in by_posterior.into_iter().take(pool / 8) {
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

    pub(super) fn lookahead_cost_for_guess(
        &self,
        guess_index: usize,
        context: LookaheadCostContext<'_>,
    ) -> Result<f64> {
        let LookaheadCostContext {
            subset,
            weights,
            expanded,
            exact_memo,
            exact_scratch,
            lookahead_memo,
        } = context;
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        if total_weight <= 0.0 {
            bail!("cannot evaluate lookahead on a zero-mass subset");
        }

        let mut ordered_patterns = [0u8; PATTERN_SPACE];
        let ordered_len = {
            let frame = exact_scratch.frame_mut(0);
            for answer_index in subset {
                let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
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
        let mut high_mass_ambiguous_bucket_count = 0usize;
        for pattern in ordered_patterns[..ordered_len].iter().copied() {
            let mass = exact_scratch.frames[0].masses[pattern as usize];
            let probability = mass / total_weight;
            let child_len = exact_scratch.frames[0].child_subsets[pattern as usize].len();
            let child_value = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else {
                let child_subset =
                    std::mem::take(&mut exact_scratch.frames[0].child_subsets[pattern as usize]);
                let result =
                    if child_subset.len() == subset.len() && child_subset.as_slice() == subset {
                        f64::INFINITY
                    } else {
                        self.lookahead_child_value(
                            &child_subset,
                            weights,
                            expanded,
                            exact_memo,
                            exact_scratch,
                            lookahead_memo,
                        )?
                    };
                exact_scratch.frames[0].child_subsets[pattern as usize] = child_subset;
                result
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
                    if child_len > 1 {
                        high_mass_ambiguous_bucket_count += 1;
                    }
                }
            }
        }
        Ok(total_cost
            + self.aggregate_lookahead_trap_penalty(
                worst_child_probability,
                large_bucket_count,
                dangerous_mass_bucket_count,
                non_green_mass_in_large_buckets,
                high_mass_ambiguous_bucket_count,
            ))
    }

    pub(super) fn lookahead_child_value(
        &self,
        subset: &[usize],
        weights: &[f64],
        expanded: bool,
        exact_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        exact_scratch: &mut ExactSearchScratch,
        lookahead_memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
    ) -> Result<f64> {
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() <= self.config.exact_exhaustive_threshold {
            return self.exact_best_cost(
                subset,
                weights,
                &self.exact_small_state_table,
                exact_memo,
                exact_scratch,
                1,
            );
        }

        let key = ExactSubsetKey::from_sorted_subset(subset);
        if let Some(cached) = lookahead_memo.get(&key) {
            return Ok(*cached);
        }

        let mut metrics =
            self.score_guess_metrics_for_subset(subset, weights, &self.exact_small_state_table);
        let split_first = subset.len() > self.config.large_state_split_threshold;
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
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
            if seen.insert(metric.guess_index) {
                reply_candidates.push(*metric);
            }
        }
        let best_reply = reply_candidates
            .into_iter()
            .map(|metric| metric.proxy_cost + self.lookahead_reply_penalty(&metric, subset.len()))
            .fold(f64::INFINITY, f64::min);
        let child_value = 1.0 + best_reply;
        lookahead_memo.insert(key, child_value);
        Ok(child_value)
    }

    pub(super) fn aggregate_lookahead_trap_penalty(
        &self,
        worst_branch_mass: f64,
        large_bucket_count: usize,
        dangerous_mass_bucket_count: usize,
        non_green_mass_in_large_buckets: f64,
        high_mass_ambiguous_bucket_count: usize,
    ) -> f64 {
        let compounded_large_mass = non_green_mass_in_large_buckets
            * (1.0
                + (large_bucket_count as f64 * 0.25)
                + (dangerous_mass_bucket_count as f64 * 0.35));
        (self.config.lookahead_trap_penalty
            * (worst_branch_mass + (high_mass_ambiguous_bucket_count as f64 / 4.0).min(1.0)))
            + (self.config.lookahead_large_bucket_penalty
                * (large_bucket_count as f64 + (high_mass_ambiguous_bucket_count as f64 * 0.5)))
            + (self.config.lookahead_dangerous_mass_penalty
                * (dangerous_mass_bucket_count as f64 + high_mass_ambiguous_bucket_count as f64))
            + (self.config.lookahead_large_bucket_mass_penalty * compounded_large_mass)
    }

    pub(super) fn lookahead_reply_penalty(&self, metric: &GuessMetrics, subset_len: usize) -> f64 {
        let bucket_ratio = metric.worst_non_green_bucket_size as f64 / subset_len.max(1) as f64;
        self.aggregate_lookahead_trap_penalty(
            bucket_ratio + metric.largest_non_green_bucket_mass,
            metric.large_non_green_bucket_count,
            metric.dangerous_mass_bucket_count,
            metric.non_green_mass_in_large_buckets,
            metric.high_mass_ambiguous_bucket_count,
        )
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

    pub(super) fn exact_cost_for_guess(
        &self,
        guess_index: usize,
        context: ExactCostContext<'_>,
    ) -> Result<f64> {
        let ExactCostContext {
            subset,
            weights,
            small_state_table,
            memo,
            best_bound,
            scratch,
            depth,
        } = context;
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() == 1 && self.guesses[guess_index] == self.answers[subset[0]].word {
            return Ok(1.0);
        }

        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut ordered_patterns = [0u8; PATTERN_SPACE];
        let ordered_len = {
            let frame = scratch.frame_mut(depth);
            for answer_index in subset {
                let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
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
            let mass = scratch.frames[depth].masses[pattern as usize];
            let branch_probability = mass / total_weight;
            let child_cost = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else {
                let child_subset =
                    std::mem::take(&mut scratch.frames[depth].child_subsets[pattern as usize]);
                let result =
                    if child_subset.len() == subset.len() && child_subset.as_slice() == subset {
                        f64::INFINITY
                    } else {
                        self.exact_best_cost(
                            &child_subset,
                            weights,
                            small_state_table,
                            memo,
                            scratch,
                            depth + 1,
                        )?
                    };
                scratch.frames[depth].child_subsets[pattern as usize] = child_subset;
                result
            };
            cost += branch_probability * child_cost;
            if cost >= best_bound {
                return Ok(cost);
            }
        }

        Ok(cost)
    }

    pub(super) fn exact_best_cost(
        &self,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
        memo: &mut PredictiveMemoMap<ExactSubsetKey, f64>,
        scratch: &mut ExactSearchScratch,
        depth: usize,
    ) -> Result<f64> {
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
            Some(ExactSuggestionMode::Exhaustive) => (0..self.guesses.len()).collect::<Vec<_>>(),
            Some(ExactSuggestionMode::Pooled) | None => {
                self.top_guess_indexes_for_subset(subset, weights, self.config.exact_candidate_pool)
            }
        };
        let lower_bound = small_state_table.lower_bound(subset.len());
        for answer_index in subset {
            debug_assert!(u16::try_from(*answer_index).is_ok());
        }
        let mut best_cost = f64::INFINITY;
        for guess_index in scores.iter().copied() {
            let cost = self.exact_cost_for_guess(
                guess_index,
                ExactCostContext {
                    subset,
                    weights,
                    small_state_table,
                    memo,
                    best_bound: best_cost,
                    scratch,
                    depth,
                },
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
                if shortlisted.contains(&guess_index) {
                    continue;
                }
                let cost = self.exact_cost_for_guess(
                    guess_index,
                    ExactCostContext {
                        subset,
                        weights,
                        small_state_table,
                        memo,
                        best_bound: best_cost,
                        scratch,
                        depth,
                    },
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
        memo.insert(key, best_cost);
        Ok(best_cost)
    }

    pub(super) fn top_guess_indexes_for_subset(
        &self,
        subset: &[usize],
        weights: &[f64],
        count: usize,
    ) -> Vec<usize> {
        let mut metrics =
            self.score_guess_metrics_for_subset(subset, weights, &self.exact_small_state_table);
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
            if seen.insert(metric.guess_index) {
                selected.push(metric.guess_index);
            }
        }
        for metric in metrics.into_iter().skip(count) {
            if surviving_guess_indexes.contains(&metric.guess_index)
                && seen.insert(metric.guess_index)
            {
                selected.push(metric.guess_index);
            }
        }
        selected
    }
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

pub(super) fn regime_from_search_mode(search_mode: PredictiveSearchMode) -> PredictiveRegime {
    match search_mode {
        PredictiveSearchMode::ProxyOnly => PredictiveRegime::Proxy,
        PredictiveSearchMode::Lookahead => PredictiveRegime::Lookahead,
        PredictiveSearchMode::EscalatedExact => PredictiveRegime::EscalatedExact,
        PredictiveSearchMode::Exact(_) => PredictiveRegime::Exact,
    }
}
