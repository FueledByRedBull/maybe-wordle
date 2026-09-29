use super::*;

impl Solver {
    pub(super) fn suggestion_from_metric(&self, metric: GuessMetrics) -> Suggestion {
        Suggestion {
            value_kind: crate::predictive::types::SuggestionValueKind::Proxy,
            finite_value: None,
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
        }
    }

    pub(super) fn score_guess_metrics_for_subset_controlled(
        &self,
        subset: &[usize],
        weights: &[f64],
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<GuessMetrics>> {
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut posterior_answer_probability = vec![0.0; self.guesses.len()];
        if total_weight > 0.0 {
            for answer_index in subset {
                if let Some(guess_index) = self.guess_index.get(&self.answers[*answer_index].word) {
                    posterior_answer_probability[*guess_index] =
                        weights[*answer_index] / total_weight;
                }
            }
        }

        (0..self.guesses.len())
            .into_par_iter()
            .map_init(GuessMetricScratch::new, |scratch, guess_index| {
                if cancelled() {
                    return Err(anyhow!("predictive search cancelled"));
                }
                Ok(self.score_guess_metrics(
                    guess_index,
                    scratch,
                    GuessMetricContext {
                        subset,
                        weights,
                        total_weight,
                        posterior_answer_probability: posterior_answer_probability[guess_index],
                    },
                ))
            })
            .collect()
    }

    pub(super) fn score_guess_metrics_for_subset(
        &self,
        subset: &[usize],
        weights: &[f64],
    ) -> Vec<GuessMetrics> {
        self.score_guess_metrics_for_subset_controlled(subset, weights, &|| false)
            .expect("no-op cancellation cannot fail")
    }

    pub(super) fn absurdle_score_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        total: f64,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<AbsurdleSuggestion> {
        super::search::check_predictive_search_cancelled(cancelled)?;
        let mut counts = [0usize; PATTERN_SPACE];
        let mut touched_patterns = Vec::new();
        for (index, answer_index) in subset.iter().enumerate() {
            if index % 128 == 0 {
                super::search::check_predictive_search_cancelled(cancelled)?;
            }
            let pattern = self.answer_pattern(guess_index, *answer_index) as usize;
            if counts[pattern] == 0 {
                touched_patterns.push(pattern as u8);
            }
            counts[pattern] += 1;
        }

        let mut largest_bucket_size = 0usize;
        let mut second_largest_bucket_size = 0usize;
        let mut multi_answer_bucket_count = 0usize;
        let mut entropy = 0.0;

        for pattern in touched_patterns {
            if pattern == ALL_GREEN_PATTERN {
                continue;
            }
            let count = counts[pattern as usize];
            if count > 1 {
                multi_answer_bucket_count += 1;
            }
            if count > largest_bucket_size {
                second_largest_bucket_size = largest_bucket_size;
                largest_bucket_size = count;
            } else if count > second_largest_bucket_size {
                second_largest_bucket_size = count;
            }
            let probability = count as f64 / total;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        Ok(AbsurdleSuggestion {
            word: self.guesses[guess_index].clone(),
            entropy,
            largest_bucket_size,
            second_largest_bucket_size,
            multi_answer_bucket_count,
        })
    }

    pub(super) fn score_guess_metrics(
        &self,
        guess_index: usize,
        scratch: &mut GuessMetricScratch,
        context: GuessMetricContext<'_>,
    ) -> GuessMetrics {
        let GuessMetricContext {
            subset,
            weights,
            total_weight,
            posterior_answer_probability,
        } = context;
        scratch.reset();
        for answer_index in subset {
            let pattern = self.answer_pattern(guess_index, *answer_index) as usize;
            if scratch.counts[pattern] == 0 {
                scratch.touched_patterns.push(pattern as u8);
            }
            let weight = weights[*answer_index];
            scratch.masses[pattern] += weight;
            scratch.largest_weights[pattern] = scratch.largest_weights[pattern].max(weight);
            scratch.counts[pattern] += 1;
        }

        self.score_partition_metrics(
            guess_index,
            scratch,
            total_weight,
            posterior_answer_probability,
        )
    }

    pub(super) fn score_partition_metrics(
        &self,
        guess_index: usize,
        scratch: &GuessMetricScratch,
        total_weight: f64,
        posterior_answer_probability: f64,
    ) -> GuessMetrics {
        let mut sum_mass_log_mass = 0.0;
        let mut expected_remaining = 0.0;
        let mut solve_probability = 0.0;
        let mut force_in_two = true;
        let mut worst_non_green_bucket_size = 0usize;
        let mut largest_non_green_bucket_mass = 0.0_f64;
        let mut high_mass_ambiguous_bucket_count = 0usize;
        let mut large_non_green_bucket_count = 0usize;
        let mut dangerous_mass_bucket_count = 0usize;
        let mut non_green_mass_in_large_buckets = 0.0_f64;
        let mut non_green_bucket_count = 0usize;
        let mut non_green_mass = 0.0_f64;
        let mut non_green_mass_square_sum = 0.0_f64;
        let mut proxy_cost = 1.0;

        for pattern in scratch.touched_patterns.iter().copied() {
            let index = pattern as usize;
            let mass = scratch.masses[index];
            let probability = if total_weight > 0.0 {
                mass / total_weight
            } else {
                0.0
            };
            if mass > 0.0 {
                sum_mass_log_mass += mass * mass.log2();
            }
            expected_remaining += probability * scratch.counts[index] as f64;
            if pattern == ALL_GREEN_PATTERN {
                solve_probability = probability;
            } else {
                if probability > 0.0 {
                    non_green_bucket_count += 1;
                    non_green_mass += probability;
                    non_green_mass_square_sum += probability * probability;
                }
                worst_non_green_bucket_size =
                    worst_non_green_bucket_size.max(scratch.counts[index]);
                largest_non_green_bucket_mass = largest_non_green_bucket_mass.max(probability);
                if scratch.counts[index] >= self.config.trap_size_threshold {
                    large_non_green_bucket_count += 1;
                    non_green_mass_in_large_buckets += probability;
                }
                if probability >= self.config.trap_mass_threshold {
                    dangerous_mass_bucket_count += 1;
                }
                if scratch.counts[index] > 1 {
                    force_in_two = false;
                    if probability >= self.config.ambiguous_mass_threshold {
                        high_mass_ambiguous_bucket_count += 1;
                    }
                }
            }
            let child_proxy = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else {
                proxy_child_cost(
                    scratch.counts[index],
                    mass,
                    scratch.largest_weights[index],
                    self.config.proxy_small_state_lower_bound_threshold,
                )
            };
            proxy_cost += probability * child_proxy;
        }
        let smoothness_penalty = normalized_concentration_penalty(
            non_green_mass,
            non_green_mass_square_sum,
            non_green_bucket_count,
        );
        let entropy = if total_weight > 0.0 {
            total_weight.log2() - (sum_mass_log_mass / total_weight)
        } else {
            0.0
        };
        let large_state_score = proxy_row_score_from_weights(
            &self.config.proxy_weights,
            ProxyRowStats {
                entropy,
                largest_non_green_bucket_mass,
                worst_non_green_bucket_size,
                high_mass_ambiguous_bucket_count,
                proxy_cost,
                solve_probability,
                posterior_answer_probability,
                smoothness_penalty,
                known_absent_letter_hits: 0,
                large_non_green_bucket_count,
                dangerous_mass_bucket_count,
                non_green_mass_in_large_buckets,
            },
        );

        GuessMetrics {
            guess_index,
            entropy,
            solve_probability,
            expected_remaining,
            force_in_two,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size,
            largest_non_green_bucket_mass,
            high_mass_ambiguous_bucket_count,
            smoothness_penalty,
            large_non_green_bucket_count,
            dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets,
            proxy_cost,
            large_state_score,
            posterior_answer_probability,
        }
    }

    pub(super) fn two_turn_success_by_guess_controlled(
        &self,
        state: &SolveState,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<f64>> {
        let guessable_answers = state
            .surviving
            .iter()
            .copied()
            .filter(|answer_index| {
                self.guess_index
                    .contains_key(&self.answers[*answer_index].word)
            })
            .collect::<Vec<_>>();
        let mut scores = vec![0.0; self.guesses.len()];
        let mut best_by_feedback = [0.0_f64; PATTERN_SPACE];
        for (guess_index, score) in scores.iter_mut().enumerate() {
            if guess_index % 64 == 0 {
                check_predictive_search_cancelled(cancelled)?;
            }
            best_by_feedback.fill(0.0);
            for answer_index in &guessable_answers {
                let pattern = self.answer_pattern(guess_index, *answer_index) as usize;
                best_by_feedback[pattern] =
                    best_by_feedback[pattern].max(state.weights[*answer_index]);
            }
            *score = best_by_feedback.iter().sum::<f64>() / state.total_weight;
        }
        Ok(scores)
    }
}

pub(super) fn weighted_proxy_child_floor(total_mass: f64, largest_mass: f64) -> f64 {
    if total_mass <= 0.0 {
        return 0.0;
    }
    1.0 + ((total_mass - largest_mass) / total_mass)
}

fn proxy_child_cost(
    count: usize,
    mass: f64,
    largest_mass: f64,
    small_state_threshold: usize,
) -> f64 {
    let weighted_floor = weighted_proxy_child_floor(mass, largest_mass);
    if count <= small_state_threshold {
        return weighted_floor;
    }

    // The count floor is a broad-state heuristic; the weight-aware floor keeps
    // its transition from lowering the proxy for a larger child. The normalized
    // entropy floor is dominated because H <= log2(count) and
    // log2(count) / log2(PATTERN_SPACE) <= max(1, count / PATTERN_SPACE).
    let expected_remaining_floor = (count as f64 / PATTERN_SPACE as f64).max(1.0);
    weighted_floor.max(expected_remaining_floor)
}

pub(super) fn compare_force_in_two(left: bool, right: bool) -> std::cmp::Ordering {
    right.cmp(&left)
}

pub(super) fn compare_three_solve_coverage(
    left: ThreeSolveCoverage,
    right: ThreeSolveCoverage,
) -> std::cmp::Ordering {
    right
        .mass
        .total_cmp(&left.mass)
        .then_with(|| left.uncovered_answers.cmp(&right.uncovered_answers))
        .then_with(|| left.uncovered_buckets.cmp(&right.uncovered_buckets))
}

pub(super) fn compare_guess_metrics_with_coverage(
    left: &GuessMetrics,
    right: &GuessMetrics,
    guesses: &[String],
    split_first: bool,
    coverage: &FxHashMap<usize, ThreeSolveCoverage>,
) -> std::cmp::Ordering {
    let left_coverage = coverage.get(&left.guess_index).copied().unwrap_or_default();
    let right_coverage = coverage
        .get(&right.guess_index)
        .copied()
        .unwrap_or_default();
    compare_three_solve_coverage(left_coverage, right_coverage)
        .then_with(|| compare_guess_metrics_for_state(left, right, guesses, split_first))
}

pub(super) fn compare_suggestions_with_coverage(
    left: &Suggestion,
    right: &Suggestion,
    _split_first: bool,
    guess_index: &HashMap<String, usize>,
    coverage: &FxHashMap<usize, ThreeSolveCoverage>,
) -> std::cmp::Ordering {
    let left_coverage = guess_index
        .get(&left.word)
        .and_then(|index| coverage.get(index))
        .copied()
        .unwrap_or_default();
    let right_coverage = guess_index
        .get(&right.word)
        .and_then(|index| coverage.get(index))
        .copied()
        .unwrap_or_default();
    // Callers append the mode-specific comparator (lookahead or exact). Keep
    // this stage coverage-only so proxy/lexical ties cannot shadow that cost.
    compare_three_solve_coverage(left_coverage, right_coverage)
}

pub(super) fn has_repeated_letters(word: &str) -> bool {
    let mut seen = [false; 26];
    for byte in word.bytes() {
        let index = (byte - b'a') as usize;
        if seen[index] {
            return true;
        }
        seen[index] = true;
    }
    false
}

pub(super) fn aggressive_early_exact_config(config: &PriorConfig) -> Result<PriorConfig> {
    apply_embedded_profile(
        config,
        include_str!("../../config/profiles/aggressive-three-guess.json"),
    )
}

pub(super) fn apply_embedded_profile(config: &PriorConfig, source: &str) -> Result<PriorConfig> {
    let profile = PredictiveConfigProfile::parse_json(source)?;
    profile.apply(
        &predictive_parameter_registry(&PriorConfig::default()),
        config,
    )
}

pub(super) fn better_targeted_run(
    candidate: &DetailedSolveRun,
    incumbent: &DetailedSolveRun,
) -> bool {
    if candidate.solved != incumbent.solved {
        return candidate.solved;
    }
    if candidate.steps.len() != incumbent.steps.len() {
        return candidate.steps.len() < incumbent.steps.len();
    }
    let candidate_path = candidate
        .steps
        .iter()
        .map(|step| step.guess.as_str())
        .collect::<Vec<_>>();
    let incumbent_path = incumbent
        .steps
        .iter()
        .map(|step| step.guess.as_str())
        .collect::<Vec<_>>();
    candidate_path < incumbent_path
}

pub(super) fn hamming_distance(left: &str, right: &str) -> usize {
    left.bytes()
        .zip(right.bytes())
        .filter(|(left, right)| left != right)
        .count()
}

pub(super) fn promote_cached_suggestion(suggestions: &mut [Suggestion], cached_word: &str) -> bool {
    if let Some(position) = suggestions
        .iter()
        .position(|suggestion| suggestion.word == cached_word)
    {
        suggestions[..=position].rotate_right(1);
        true
    } else {
        false
    }
}

pub(super) fn compare_forced_openers(
    left: &ForcedOpenerEvaluation,
    right: &ForcedOpenerEvaluation,
    guesses: &[String],
) -> std::cmp::Ordering {
    left.failures
        .cmp(&right.failures)
        .then_with(|| left.four_guess_games.cmp(&right.four_guess_games))
        .then_with(|| left.average_guesses.total_cmp(&right.average_guesses))
        .then_with(|| left.p95_guesses.cmp(&right.p95_guesses))
        .then_with(|| left.max_guesses.cmp(&right.max_guesses))
        .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
}

pub(super) fn should_replace_forced_opener(
    candidate_primary: &ForcedOpenerEvaluation,
    candidate_holdout: Option<&ForcedOpenerEvaluation>,
    incumbent_primary: &ForcedOpenerEvaluation,
    incumbent_holdout: Option<&ForcedOpenerEvaluation>,
    guesses: &[String],
) -> bool {
    if compare_forced_openers(candidate_primary, incumbent_primary, guesses)
        != std::cmp::Ordering::Less
    {
        return false;
    }
    match (candidate_holdout, incumbent_holdout) {
        (Some(candidate), Some(incumbent)) => {
            compare_forced_openers(candidate, incumbent, guesses) != std::cmp::Ordering::Greater
        }
        _ => true,
    }
}

pub(super) fn known_absent_letter_mask(observations: &[(String, u8)]) -> u32 {
    let mut gray_mask = 0u32;
    let mut present_mask = 0u32;
    for (guess, pattern) in observations {
        let mut value = *pattern;
        for byte in guess.bytes().take(5) {
            let letter_bit = 1u32 << ((byte - b'a') as u32);
            let trit = value % 3;
            value /= 3;
            if trit == 0 {
                gray_mask |= letter_bit;
            } else {
                present_mask |= letter_bit;
            }
        }
    }
    gray_mask & !present_mask
}

pub(super) fn count_masked_letters(word: &str, mask: u32) -> usize {
    word.bytes()
        .filter(|byte| (mask & (1u32 << ((byte - b'a') as u32))) != 0)
        .count()
}

pub(super) fn normalized_concentration_penalty(
    total_mass: f64,
    mass_square_sum: f64,
    bucket_count: usize,
) -> f64 {
    if total_mass <= 0.0 || bucket_count <= 1 {
        return 0.0;
    }
    let concentration = mass_square_sum / (total_mass * total_mass);
    let uniform = 1.0 / bucket_count as f64;
    ((concentration - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
}

pub(super) fn proxy_row_score_from_weights(
    weights: &crate::config::ProxyWeights,
    row: ProxyRowStats,
) -> f64 {
    (weights.entropy_w * row.entropy)
        - (weights.bucket_mass_w * row.largest_non_green_bucket_mass)
        - (weights.bucket_size_w * row.worst_non_green_bucket_size as f64)
        - (weights.ambiguous_w * row.high_mass_ambiguous_bucket_count as f64)
        - (weights.proxy_w * row.proxy_cost)
        + (weights.solve_prob_w * row.solve_probability)
        + (weights.posterior_w * row.posterior_answer_probability)
        - (weights.smoothness_w * row.smoothness_penalty)
        - (weights.gray_reuse_w * row.known_absent_letter_hits as f64)
        - (weights.large_bucket_count_w * row.large_non_green_bucket_count as f64)
        - (weights.dangerous_mass_count_w * row.dangerous_mass_bucket_count as f64)
        - (weights.large_bucket_mass_w * row.non_green_mass_in_large_buckets)
}

pub(super) fn compare_guess_metrics(
    left: &GuessMetrics,
    right: &GuessMetrics,
    guesses: &[String],
) -> std::cmp::Ordering {
    compare_guess_metrics_for_state(left, right, guesses, false)
}

pub(super) fn compare_absurdle_suggestions(
    left: &AbsurdleSuggestion,
    right: &AbsurdleSuggestion,
) -> std::cmp::Ordering {
    left.largest_bucket_size
        .cmp(&right.largest_bucket_size)
        .then_with(|| {
            left.second_largest_bucket_size
                .cmp(&right.second_largest_bucket_size)
        })
        .then_with(|| {
            left.multi_answer_bucket_count
                .cmp(&right.multi_answer_bucket_count)
        })
        .then_with(|| right.entropy.total_cmp(&left.entropy))
        .then_with(|| left.word.cmp(&right.word))
}

pub(super) fn compare_guess_metrics_for_state(
    left: &GuessMetrics,
    right: &GuessMetrics,
    guesses: &[String],
    split_first: bool,
) -> std::cmp::Ordering {
    if split_first {
        let score_cmp = right.large_state_score.total_cmp(&left.large_state_score);
        if score_cmp != std::cmp::Ordering::Equal {
            return score_cmp;
        }
        left.known_absent_letter_hits
            .cmp(&right.known_absent_letter_hits)
            .then_with(|| {
                left.largest_non_green_bucket_mass
                    .total_cmp(&right.largest_non_green_bucket_mass)
            })
            .then_with(|| {
                left.worst_non_green_bucket_size
                    .cmp(&right.worst_non_green_bucket_size)
            })
            .then_with(|| {
                left.large_non_green_bucket_count
                    .cmp(&right.large_non_green_bucket_count)
            })
            .then_with(|| {
                left.dangerous_mass_bucket_count
                    .cmp(&right.dangerous_mass_bucket_count)
            })
            .then_with(|| right.entropy.total_cmp(&left.entropy))
            .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
            .then_with(|| left.proxy_cost.total_cmp(&right.proxy_cost))
            .then_with(|| compare_force_in_two(left.force_in_two, right.force_in_two))
            .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
            .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
    } else {
        let proxy_cmp = left.proxy_cost.total_cmp(&right.proxy_cost);
        if proxy_cmp != std::cmp::Ordering::Equal {
            return proxy_cmp;
        }
        compare_force_in_two(left.force_in_two, right.force_in_two)
            .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
            .then_with(|| right.entropy.total_cmp(&left.entropy))
            .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
            .then_with(|| {
                left.worst_non_green_bucket_size
                    .cmp(&right.worst_non_green_bucket_size)
            })
            .then_with(|| {
                left.largest_non_green_bucket_mass
                    .total_cmp(&right.largest_non_green_bucket_mass)
            })
            .then_with(|| {
                right
                    .posterior_answer_probability
                    .total_cmp(&left.posterior_answer_probability)
            })
            .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
    }
}

pub(super) fn compare_suggestions(left: &Suggestion, right: &Suggestion) -> std::cmp::Ordering {
    compare_suggestions_for_state(left, right, false)
}

pub(super) fn compare_suggestions_for_state(
    left: &Suggestion,
    right: &Suggestion,
    split_first: bool,
) -> std::cmp::Ordering {
    if split_first {
        let left_score = left.large_state_score.unwrap_or(f64::NEG_INFINITY);
        let right_score = right.large_state_score.unwrap_or(f64::NEG_INFINITY);
        let score_cmp = right_score.total_cmp(&left_score);
        if score_cmp != std::cmp::Ordering::Equal {
            return score_cmp;
        }
        left.known_absent_letter_hits
            .cmp(&right.known_absent_letter_hits)
            .then_with(|| {
                left.largest_non_green_bucket_mass
                    .total_cmp(&right.largest_non_green_bucket_mass)
            })
            .then_with(|| {
                left.worst_non_green_bucket_size
                    .cmp(&right.worst_non_green_bucket_size)
            })
            .then_with(|| {
                left.large_non_green_bucket_count
                    .cmp(&right.large_non_green_bucket_count)
            })
            .then_with(|| {
                left.dangerous_mass_bucket_count
                    .cmp(&right.dangerous_mass_bucket_count)
            })
            .then_with(|| {
                left.non_green_mass_in_large_buckets
                    .total_cmp(&right.non_green_mass_in_large_buckets)
            })
            .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
            .then_with(|| {
                left.proxy_cost
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&right.proxy_cost.unwrap_or(f64::INFINITY))
            })
            .then_with(|| compare_force_in_two(left.force_in_two, right.force_in_two))
            .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
            .then_with(|| left.word.cmp(&right.word))
    } else {
        let left_proxy = left.proxy_cost.unwrap_or(f64::INFINITY);
        let right_proxy = right.proxy_cost.unwrap_or(f64::INFINITY);
        let proxy_cmp = left_proxy.total_cmp(&right_proxy);
        if proxy_cmp != std::cmp::Ordering::Equal {
            return proxy_cmp;
        }
        compare_force_in_two(left.force_in_two, right.force_in_two)
            .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
            .then_with(|| right.entropy.total_cmp(&left.entropy))
            .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
            .then_with(|| {
                left.worst_non_green_bucket_size
                    .cmp(&right.worst_non_green_bucket_size)
            })
            .then_with(|| {
                left.largest_non_green_bucket_mass
                    .total_cmp(&right.largest_non_green_bucket_mass)
            })
            .then_with(|| {
                left.large_non_green_bucket_count
                    .cmp(&right.large_non_green_bucket_count)
            })
            .then_with(|| {
                left.dangerous_mass_bucket_count
                    .cmp(&right.dangerous_mass_bucket_count)
            })
            .then_with(|| {
                left.non_green_mass_in_large_buckets
                    .total_cmp(&right.non_green_mass_in_large_buckets)
            })
            .then_with(|| {
                right
                    .posterior_answer_probability
                    .total_cmp(&left.posterior_answer_probability)
            })
            .then_with(|| left.word.cmp(&right.word))
    }
}

pub(super) fn compare_lookahead(
    left: &Suggestion,
    right: &Suggestion,
    split_first: bool,
) -> std::cmp::Ordering {
    match (left.lookahead_cost, right.lookahead_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions_for_state(left, right, split_first))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions_for_state(left, right, split_first),
    }
}

pub(super) fn compare_exact(
    left: &Suggestion,
    right: &Suggestion,
    split_first: bool,
) -> std::cmp::Ordering {
    match (left.exact_cost, right.exact_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions_for_state(left, right, split_first))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions_for_state(left, right, split_first),
    }
}

pub(super) fn compare_exact_costs(
    left: &Suggestion,
    right: &Suggestion,
    left_cost: Option<f64>,
    right_cost: Option<f64>,
    split_first: bool,
) -> std::cmp::Ordering {
    match (left_cost, right_cost) {
        (Some(left_cost), Some(right_cost)) => {
            let cost_cmp = left_cost.total_cmp(&right_cost);
            if cost_cmp != std::cmp::Ordering::Equal {
                return cost_cmp;
            }
            compare_force_in_two(left.force_in_two, right.force_in_two)
                .then_with(|| compare_suggestions_for_state(left, right, split_first))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions_for_state(left, right, split_first),
    }
}

pub(super) fn compare_final_turn(left: &Suggestion, right: &Suggestion) -> std::cmp::Ordering {
    right
        .solve_probability
        .total_cmp(&left.solve_probability)
        .then_with(|| {
            right
                .posterior_answer_probability
                .total_cmp(&left.posterior_answer_probability)
        })
        .then_with(|| left.word.cmp(&right.word))
}

pub(super) fn compare_two_turn(
    left: &Suggestion,
    right: &Suggestion,
    left_success: f64,
    right_success: f64,
) -> std::cmp::Ordering {
    compare_force_in_two(left.force_in_two, right.force_in_two)
        .then_with(|| right_success.total_cmp(&left_success))
        .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
        .then_with(|| left.word.cmp(&right.word))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn weighted_child_floor_cannot_increase_under_partition_refinement() {
        for buckets in [
            vec![vec![0.8, 0.1], vec![0.05, 0.05]],
            vec![vec![1.0], vec![1.0, 1.0]],
        ] {
            let total: f64 = buckets.iter().flatten().sum();
            let largest = buckets.iter().flatten().copied().fold(0.0, f64::max);
            let coarse = total * weighted_proxy_child_floor(total, largest);
            let refined: f64 = buckets
                .iter()
                .map(|bucket| {
                    let mass = bucket.iter().sum();
                    let largest = bucket.iter().copied().fold(0.0, f64::max);
                    mass * weighted_proxy_child_floor(mass, largest)
                })
                .sum();
            assert!(refined <= coarse + 1e-12);
        }
    }

    #[test]
    fn broad_state_count_heuristic_is_not_a_refinement_monotone_bound() {
        let coarse = proxy_child_cost(300, 1.0, 0.8, 12);
        let refined =
            0.9 * proxy_child_cost(298, 0.9, 0.8, 12) + 0.1 * proxy_child_cost(2, 0.1, 0.05, 12);
        assert!(refined > coarse);
    }

    #[test]
    fn proxy_child_cost_has_no_uniform_threshold_drop() {
        let threshold = 12;
        let at_threshold = proxy_child_cost(threshold, threshold as f64, 1.0, threshold);
        let above_threshold =
            proxy_child_cost(threshold + 1, (threshold + 1) as f64, 1.0, threshold);

        assert!((at_threshold - (2.0 - 1.0 / threshold as f64)).abs() <= 1e-12);
        assert!(above_threshold > at_threshold);
    }

    #[test]
    fn proxy_child_cost_keeps_skewed_mass_floor_across_threshold() {
        let threshold = 12;
        let at_threshold = proxy_child_cost(threshold, 1.0, 0.99, threshold);
        let above_threshold = proxy_child_cost(threshold + 1, 1.0, 0.99, threshold);

        assert!((at_threshold - 1.01).abs() <= 1e-12);
        assert!((above_threshold - at_threshold).abs() <= 1e-12);
    }

    #[test]
    fn proxy_child_cost_matches_the_dominated_entropy_floor() {
        fn old_proxy_child_cost(
            count: usize,
            mass: f64,
            largest_mass: f64,
            weighted_log_sum: f64,
            pattern_space_log: f64,
            small_state_threshold: usize,
        ) -> f64 {
            let weighted_floor = weighted_proxy_child_floor(mass, largest_mass);
            if count <= small_state_threshold {
                return weighted_floor;
            }
            let expected_remaining_floor = (count as f64 / PATTERN_SPACE as f64).max(1.0);
            let entropy_bits = if mass > 0.0 {
                mass.log2() - (weighted_log_sum / mass)
            } else {
                0.0
            };
            let entropy_floor = (entropy_bits / pattern_space_log).max(1.0);
            weighted_floor.max(expected_remaining_floor.max(entropy_floor))
        }

        let pattern_space_log = (PATTERN_SPACE as f64).log2();
        let threshold = 12;
        for count in [1, threshold, threshold + 1, 242, 243, 244, 486] {
            for weights in [
                vec![1.0; count],
                if count == 1 {
                    vec![1.0]
                } else {
                    let mut weights = vec![0.2 / (count - 1) as f64; count];
                    weights[0] = 0.8;
                    weights
                },
            ] {
                let mass = weights.iter().sum::<f64>();
                let largest_mass = weights.iter().copied().fold(0.0, f64::max);
                let weighted_log_sum = weights
                    .iter()
                    .filter(|weight| **weight > 0.0)
                    .map(|weight| weight * weight.log2())
                    .sum::<f64>();
                let old = old_proxy_child_cost(
                    count,
                    mass,
                    largest_mass,
                    weighted_log_sum,
                    pattern_space_log,
                    threshold,
                );
                let new = proxy_child_cost(count, mass, largest_mass, threshold);
                assert!(
                    (new - old).abs() <= 1e-12,
                    "count={count} weights={weights:?}"
                );
            }
        }
    }

    #[test]
    fn equal_coverage_defers_to_appended_search_cost() {
        fn suggestion(word: &str, proxy_cost: f64, lookahead_cost: f64) -> Suggestion {
            Suggestion {
                value_kind: crate::predictive::types::SuggestionValueKind::Proxy,
                finite_value: None,
                word: word.to_string(),
                entropy: 0.0,
                solve_probability: 0.0,
                expected_remaining: 0.0,
                force_in_two: false,
                known_absent_letter_hits: 0,
                worst_non_green_bucket_size: 0,
                largest_non_green_bucket_mass: 0.0,
                large_non_green_bucket_count: 0,
                dangerous_mass_bucket_count: 0,
                non_green_mass_in_large_buckets: 0.0,
                proxy_cost: Some(proxy_cost),
                large_state_score: None,
                posterior_answer_probability: 0.0,
                lookahead_cost: Some(lookahead_cost),
                exact_cost: Some(lookahead_cost),
            }
        }

        let left = suggestion("zebra", 2.0, 1.0);
        let right = suggestion("alpha", 1.0, 2.0);
        let guess_index =
            HashMap::from([(left.word.clone(), 0usize), (right.word.clone(), 1usize)]);
        let mut coverage = FxHashMap::default();
        let equal = ThreeSolveCoverage {
            mass: 0.5,
            uncovered_answers: 2,
            uncovered_buckets: 1,
        };
        coverage.insert(0, equal);
        coverage.insert(1, equal);

        assert_eq!(
            compare_suggestions_with_coverage(&left, &right, false, &guess_index, &coverage,),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_suggestions_with_coverage(&left, &right, false, &guess_index, &coverage,)
                .then_with(|| compare_lookahead(&left, &right, false)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_suggestions_with_coverage(&left, &right, false, &guess_index, &coverage)
                .then_with(|| compare_exact(&left, &right, false)),
            std::cmp::Ordering::Less
        );
    }
}
