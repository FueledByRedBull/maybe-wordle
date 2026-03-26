use super::*;

impl Solver {
    pub(super) fn score_guess_metrics_for_subset(
        &self,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
    ) -> Vec<GuessMetrics> {
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
                self.score_guess_metrics(
                    guess_index,
                    scratch,
                    GuessMetricContext {
                        subset,
                        weights,
                        total_weight,
                        small_state_table,
                        posterior_answer_probability: posterior_answer_probability[guess_index],
                    },
                )
            })
            .collect()
    }

    pub(super) fn absurdle_score_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        total: f64,
    ) -> AbsurdleSuggestion {
        let mut counts = [0usize; PATTERN_SPACE];
        let mut touched_patterns = Vec::new();
        for answer_index in subset {
            let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
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

        AbsurdleSuggestion {
            word: self.guesses[guess_index].clone(),
            entropy,
            largest_bucket_size,
            second_largest_bucket_size,
            multi_answer_bucket_count,
        }
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
            small_state_table,
            posterior_answer_probability,
        } = context;
        scratch.reset();
        for answer_index in subset {
            let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
            if scratch.counts[pattern] == 0 {
                scratch.touched_patterns.push(pattern as u8);
            }
            let weight = weights[*answer_index];
            scratch.masses[pattern] += weight;
            scratch.counts[pattern] += 1;
            if weight > 0.0 {
                scratch.weighted_log_sums[pattern] += weight * weight.log2();
            }
        }

        let pattern_space_log = (PATTERN_SPACE as f64).log2();
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
                non_green_bucket_count += 1;
                non_green_mass += probability;
                non_green_mass_square_sum += probability * probability;
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
                    if probability >= 0.10 {
                        high_mass_ambiguous_bucket_count += 1;
                    }
                }
            }
            let child_proxy = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else if scratch.counts[index] == 1 {
                1.0
            } else if scratch.counts[index] <= self.config.exact_exhaustive_threshold {
                small_state_table.lower_bound(scratch.counts[index])
            } else {
                let expected_remaining_floor =
                    (scratch.counts[index] as f64 / PATTERN_SPACE as f64).max(1.0);
                let entropy_bits = if mass > 0.0 {
                    mass.log2() - (scratch.weighted_log_sums[index] / mass)
                } else {
                    0.0
                };
                let entropy_floor = (entropy_bits / pattern_space_log).max(1.0);
                expected_remaining_floor.max(entropy_floor)
            };
            proxy_cost += probability * child_proxy;
        }
        let smoothness_penalty = normalized_concentration_penalty(
            non_green_mass,
            non_green_mass_square_sum,
            non_green_bucket_count,
        );
        proxy_cost += 0.25 * smoothness_penalty;
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
    split_first: bool,
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
    compare_three_solve_coverage(left_coverage, right_coverage)
        .then_with(|| compare_suggestions_for_state(left, right, split_first))
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

pub(super) fn aggressive_early_exact_config(config: &PriorConfig) -> PriorConfig {
    let mut aggressive = config.clone();
    aggressive.exact_threshold = aggressive.exact_threshold.max(80);
    aggressive.exact_exhaustive_threshold = aggressive.exact_exhaustive_threshold.max(16);
    aggressive.exact_candidate_pool = aggressive.exact_candidate_pool.max(160);
    aggressive.lookahead_threshold = aggressive.lookahead_threshold.max(192);
    aggressive.medium_state_lookahead_threshold =
        aggressive.medium_state_lookahead_threshold.max(96);
    aggressive.lookahead_candidate_pool = aggressive.lookahead_candidate_pool.max(48);
    aggressive.medium_state_lookahead_candidate_pool =
        aggressive.medium_state_lookahead_candidate_pool.max(64);
    aggressive.lookahead_reply_pool = aggressive.lookahead_reply_pool.max(24);
    aggressive.medium_state_lookahead_reply_pool =
        aggressive.medium_state_lookahead_reply_pool.max(24);
    aggressive.lookahead_root_force_in_two_scan =
        aggressive.lookahead_root_force_in_two_scan.max(96);
    aggressive.medium_state_force_in_two_scan = aggressive.medium_state_force_in_two_scan.max(160);
    aggressive.danger_lookahead_threshold = aggressive.danger_lookahead_threshold.min(0.52);
    aggressive.danger_exact_threshold = aggressive.danger_exact_threshold.min(0.64);
    aggressive.danger_reply_pool_bonus = aggressive.danger_reply_pool_bonus.max(10);
    aggressive.danger_exact_root_pool = aggressive.danger_exact_root_pool.max(48);
    aggressive.danger_exact_survivor_cap = aggressive.danger_exact_survivor_cap.max(224);
    aggressive.large_state_split_threshold = aggressive.large_state_split_threshold.min(48);
    aggressive
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

pub(super) fn promote_cached_suggestion(suggestions: &mut [Suggestion], cached_word: &str) {
    if let Some(position) = suggestions
        .iter()
        .position(|suggestion| suggestion.word == cached_word)
    {
        suggestions[..=position].rotate_right(1);
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

pub(super) fn flatten_weighted_config(config: &PriorConfig, keep_factor: f64) -> PriorConfig {
    let mut flattened = config.clone();
    let blend = |value: f64| 1.0 + ((value - 1.0) * keep_factor);
    flattened.base_seed_weight = blend(flattened.base_seed_weight).max(0.01);
    flattened.base_history_only_weight = blend(flattened.base_history_only_weight).max(0.01);
    flattened.cooldown_floor = blend(flattened.cooldown_floor).clamp(0.0, 1.0);
    flattened.manual_weights = flattened
        .manual_weights
        .into_iter()
        .map(|(word, weight)| (word, blend(weight).max(0.01)))
        .collect();
    flattened
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
