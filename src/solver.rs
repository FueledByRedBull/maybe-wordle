use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{Days, NaiveDate, Utc};
use rayon::prelude::*;

use crate::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, read_history_jsonl},
    model::{
        AnswerRecord, ModelVariant, WeightMode, load_model, load_model_with_variant,
        weight_snapshot_for_mode,
    },
    pattern_table::PatternTable,
    scoring::{
        ALL_GREEN_PATTERN, PATTERN_SPACE, format_feedback_letters, parse_feedback, score_guess,
    },
    small_state::SmallStateTable,
};

#[derive(Clone, Debug)]
pub struct Suggestion {
    pub word: String,
    pub entropy: f64,
    pub solve_probability: f64,
    pub expected_remaining: f64,
    pub exact_cost: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SolveState {
    pub surviving: Vec<usize>,
    pub weights: Vec<f64>,
    pub total_weight: f64,
}

#[derive(Clone, Debug)]
pub struct SolveStep {
    pub guess: String,
    pub feedback: u8,
}

#[derive(Clone, Debug)]
pub struct SolveRun {
    pub target: String,
    pub date: NaiveDate,
    pub steps: Vec<SolveStep>,
    pub solved: bool,
}

#[derive(Clone, Debug)]
pub struct BacktestStats {
    pub games: usize,
    pub average_guesses: f64,
    pub p95_guesses: usize,
    pub max_guesses: usize,
    pub failures: usize,
    pub coverage_gaps: usize,
}

#[derive(Clone, Debug)]
pub struct ExperimentResult {
    pub mode: WeightMode,
    pub variant: ModelVariant,
    pub backtest: BacktestStats,
    pub average_log_loss: f64,
    pub average_brier: f64,
    pub average_target_probability: f64,
    pub average_target_rank: f64,
}

pub struct Solver {
    pub config: PriorConfig,
    pub mode: WeightMode,
    pub variant: ModelVariant,
    pub guesses: Vec<String>,
    pub answers: Vec<AnswerRecord>,
    pub history_dates: Vec<NytDailyEntry>,
    pattern_table: PatternTable,
    guess_index: HashMap<String, usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExactSuggestionMode {
    Exhaustive,
    Pooled,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactSubsetKey(Box<[usize]>);

impl ExactSubsetKey {
    fn from_sorted_subset(subset: &[usize]) -> Self {
        debug_assert!(subset.windows(2).all(|window| window[0] < window[1]));
        Self(subset.to_vec().into_boxed_slice())
    }
}

impl Solver {
    pub fn from_paths(paths: &ProjectPaths, config: &PriorConfig) -> Result<Self> {
        Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )
    }

    pub fn from_paths_with_mode(
        paths: &ProjectPaths,
        config: &PriorConfig,
        mode: WeightMode,
    ) -> Result<Self> {
        Self::from_paths_with_settings(paths, config, mode, ModelVariant::SeedPlusHistory)
    }

    pub fn from_paths_with_settings(
        paths: &ProjectPaths,
        config: &PriorConfig,
        mode: WeightMode,
        variant: ModelVariant,
    ) -> Result<Self> {
        let model = if variant == ModelVariant::SeedPlusHistory {
            load_model(paths, config)?
        } else {
            load_model_with_variant(paths, config, variant)?
        };
        let pattern_table = PatternTable::load_or_build(paths, &model.guesses, &model.answers)?;
        let guess_index = model
            .guesses
            .iter()
            .enumerate()
            .map(|(index, guess)| (guess.clone(), index))
            .collect::<HashMap<_, _>>();

        Ok(Self {
            config: config.clone(),
            mode,
            variant: model.variant,
            guesses: model.guesses,
            answers: model.answers,
            history_dates: model.history,
            pattern_table,
            guess_index,
        })
    }

    pub fn today() -> NaiveDate {
        Utc::now().date_naive()
    }

    pub fn initial_state(&self, as_of: NaiveDate) -> SolveState {
        let mut weights = vec![0.0; self.answers.len()];
        let mut surviving = Vec::new();
        let mut total_weight = 0.0;

        for (index, answer) in self.answers.iter().enumerate() {
            let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
            if snapshot.final_weight > 0.0 {
                weights[index] = snapshot.final_weight;
                total_weight += snapshot.final_weight;
                surviving.push(index);
            }
        }

        SolveState {
            surviving,
            weights,
            total_weight,
        }
    }

    pub fn apply_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
    ) -> Result<SolveState> {
        let mut state = self.initial_state(as_of);
        for (guess, pattern) in observations {
            self.apply_feedback(&mut state, guess, *pattern)?;
        }
        Ok(state)
    }

    pub fn apply_feedback(&self, state: &mut SolveState, guess: &str, pattern: u8) -> Result<()> {
        let guess_index = self
            .guess_index
            .get(&guess.to_ascii_lowercase())
            .copied()
            .ok_or_else(|| anyhow!("unknown guess: {}", guess))?;
        state
            .surviving
            .retain(|answer_index| self.pattern_table.get(guess_index, *answer_index) == pattern);
        state.total_weight = state
            .surviving
            .iter()
            .map(|index| state.weights[*index])
            .sum::<f64>();

        if state.surviving.is_empty() {
            bail!(
                "no answers remain after applying {} {}",
                guess,
                format_feedback_letters(pattern)
            );
        }
        Ok(())
    }

    pub fn suggestions(&self, state: &SolveState, top: usize) -> Result<Vec<Suggestion>> {
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        let surviving_words = state
            .surviving
            .iter()
            .map(|index| self.answers[*index].word.as_str())
            .collect::<HashSet<_>>();

        let mut suggestions = (0..self.guesses.len())
            .into_par_iter()
            .map(|guess_index| {
                let mut masses = [0.0f64; PATTERN_SPACE];
                let mut counts = [0usize; PATTERN_SPACE];
                for answer_index in &state.surviving {
                    let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
                    masses[pattern] += state.weights[*answer_index];
                    counts[pattern] += 1;
                }

                let entropy = masses
                    .iter()
                    .filter(|mass| **mass > 0.0)
                    .map(|mass| {
                        let probability = mass / state.total_weight;
                        -probability * probability.log2()
                    })
                    .sum::<f64>();

                let expected_remaining = masses
                    .iter()
                    .zip(counts.iter())
                    .filter(|(mass, _)| **mass > 0.0)
                    .map(|(mass, count)| (mass / state.total_weight) * (*count as f64))
                    .sum::<f64>();

                Suggestion {
                    word: self.guesses[guess_index].clone(),
                    entropy,
                    solve_probability: masses[ALL_GREEN_PATTERN as usize] / state.total_weight,
                    expected_remaining,
                    exact_cost: None,
                }
            })
            .collect::<Vec<_>>();

        suggestions.sort_by(|left, right| compare_suggestions(left, right, &surviving_words));

        if let Some(exact_mode) = exact_suggestion_mode(&self.config, state.surviving.len()) {
            let small_state_table =
                SmallStateTable::build(self.config.exact_exhaustive_threshold.max(2));
            let exact_candidates = match exact_mode {
                ExactSuggestionMode::Exhaustive => (0..self.guesses.len()).collect::<Vec<_>>(),
                ExactSuggestionMode::Pooled => {
                    self.collect_exact_candidates(state, &suggestions)?
                }
            };
            let mut memo = HashMap::new();
            let mut exact_costs = vec![None; self.guesses.len()];

            for guess_index in exact_candidates {
                let cost = self.exact_cost_for_guess(
                    guess_index,
                    &state.surviving,
                    &state.weights,
                    &small_state_table,
                    &mut memo,
                    f64::INFINITY,
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
                    suggestions.sort_by(|left, right| compare_exact(left, right, &surviving_words));
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
                        compare_exact_costs(left, right, left_cost, right_cost, &surviving_words)
                    });
                }
            }
        }

        suggestions.truncate(top);
        Ok(suggestions)
    }

    pub fn solve_target(&self, target: &str, date: NaiveDate, top: usize) -> Result<SolveRun> {
        let target = target.to_ascii_lowercase();
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let mut state = self.initial_state(as_of);

        if !state
            .surviving
            .iter()
            .any(|index| self.answers[*index].word == target)
        {
            return Ok(SolveRun {
                target,
                date,
                steps: Vec::new(),
                solved: false,
            });
        }

        let mut steps = Vec::new();
        while steps.len() < 6 {
            let suggestions = self.suggestions(&state, top.max(1))?;
            let guess = suggestions
                .first()
                .ok_or_else(|| anyhow!("solver returned no suggestions"))?
                .word
                .clone();
            let feedback = score_guess(&guess, &target);
            steps.push(SolveStep {
                guess: guess.clone(),
                feedback,
            });
            if feedback == ALL_GREEN_PATTERN {
                return Ok(SolveRun {
                    target,
                    date,
                    steps,
                    solved: true,
                });
            }
            self.apply_feedback(&mut state, &guess, feedback)?;
        }

        Ok(SolveRun {
            target,
            date,
            steps,
            solved: false,
        })
    }

    pub fn backtest(&self, from: NaiveDate, to: NaiveDate, top: usize) -> Result<BacktestStats> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested backtest range");
        }

        let mut guess_counts = Vec::new();
        let mut failures = 0usize;
        let mut coverage_gaps = 0usize;

        for entry in games {
            let run = self.solve_target(&entry.solution, entry.print_date, top)?;
            if run.steps.is_empty() {
                coverage_gaps += 1;
                failures += 1;
                continue;
            }
            if !run.solved {
                failures += 1;
            }
            guess_counts.push(run.steps.len());
        }

        guess_counts.sort_unstable();
        let games_played = guess_counts.len() + coverage_gaps;
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        let p95_guesses = guess_counts
            .get(p95_index.saturating_sub(1))
            .copied()
            .unwrap_or_default();
        let max_guesses = guess_counts.last().copied().unwrap_or_default();

        Ok(BacktestStats {
            games: games_played,
            average_guesses,
            p95_guesses,
            max_guesses,
            failures,
            coverage_gaps,
        })
    }

    pub fn experiment_report(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<ExperimentResult> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let backtest = self.backtest(from, to, top)?;
        let mut total_log_loss = 0.0;
        let mut total_brier = 0.0;
        let mut total_target_probability = 0.0;
        let mut total_rank = 0.0;
        let mut measured = 0usize;

        for entry in games {
            if let Some(metrics) = self.initial_prior_metrics(&entry.solution, entry.print_date) {
                total_log_loss += metrics.log_loss;
                total_brier += metrics.brier;
                total_target_probability += metrics.target_probability;
                total_rank += metrics.target_rank as f64;
                measured += 1;
            }
        }

        let divisor = measured.max(1) as f64;
        Ok(ExperimentResult {
            mode: self.mode,
            variant: self.variant,
            backtest,
            average_log_loss: total_log_loss / divisor,
            average_brier: total_brier / divisor,
            average_target_probability: total_target_probability / divisor,
            average_target_rank: total_rank / divisor,
        })
    }

    pub fn parse_observations(
        guesses: &[String],
        feedbacks: &[String],
    ) -> Result<Vec<(String, u8)>> {
        if guesses.len() != feedbacks.len() {
            bail!("--guess and --feedback must appear the same number of times");
        }

        guesses
            .iter()
            .zip(feedbacks)
            .map(|(guess, feedback)| {
                Ok((guess.trim().to_ascii_lowercase(), parse_feedback(feedback)?))
            })
            .collect()
    }

    pub fn latest_history_range(paths: &ProjectPaths) -> Result<Option<(NaiveDate, NaiveDate)>> {
        let history = read_history_jsonl(&paths.raw_history)?;
        Ok(history
            .first()
            .zip(history.last())
            .map(|(first, last)| (first.print_date, last.print_date)))
    }

    pub fn pattern_table_bytes(&self) -> usize {
        self.pattern_table.bytes_len()
    }

    fn initial_prior_metrics(&self, target: &str, date: NaiveDate) -> Option<PriorMetrics> {
        let as_of = date.checked_sub_days(Days::new(1))?;
        let state = self.initial_state(as_of);
        let target = target.to_ascii_lowercase();
        let target_index = state
            .surviving
            .iter()
            .find(|index| self.answers[**index].word == target)
            .copied()?;

        let target_probability = state.weights[target_index] / state.total_weight;
        let mut ordered = state
            .surviving
            .iter()
            .map(|index| (*index, state.weights[*index] / state.total_weight))
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| right.1.total_cmp(&left.1));
        let target_rank = ordered
            .iter()
            .position(|(index, _)| *index == target_index)
            .map(|rank| rank + 1)?;
        let probability_square_sum = ordered
            .iter()
            .map(|(_, probability)| probability * probability)
            .sum::<f64>();

        Some(PriorMetrics {
            target_probability,
            target_rank,
            log_loss: -(target_probability.max(1e-12)).ln(),
            brier: probability_square_sum - (2.0 * target_probability) + 1.0,
        })
    }

    fn collect_exact_candidates(
        &self,
        state: &SolveState,
        suggestions: &[Suggestion],
    ) -> Result<Vec<usize>> {
        let mut candidate_indexes = suggestions
            .iter()
            .take(self.config.exact_candidate_pool)
            .map(|suggestion| {
                self.guess_index
                    .get(&suggestion.word)
                    .copied()
                    .with_context(|| format!("missing guess {}", suggestion.word))
            })
            .collect::<Result<Vec<_>>>()?;

        for answer_index in &state.surviving {
            if let Some(guess_index) = self.guess_index.get(&self.answers[*answer_index].word) {
                candidate_indexes.push(*guess_index);
            }
        }

        candidate_indexes.sort_unstable();
        candidate_indexes.dedup();
        Ok(candidate_indexes)
    }

    fn exact_cost_for_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
        memo: &mut HashMap<ExactSubsetKey, f64>,
        best_bound: f64,
    ) -> Result<f64> {
        if subset.is_empty() {
            return Ok(0.0);
        }
        if subset.len() == 1 && self.guesses[guess_index] == self.answers[subset[0]].word {
            return Ok(1.0);
        }

        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut partitions: HashMap<u8, (f64, Vec<usize>)> = HashMap::new();

        for answer_index in subset {
            let pattern = self.pattern_table.get(guess_index, *answer_index);
            let entry = partitions.entry(pattern).or_insert((0.0, Vec::new()));
            entry.0 += weights[*answer_index];
            entry.1.push(*answer_index);
        }

        let mut ordered = partitions.into_iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| right.1.0.total_cmp(&left.1.0));

        let mut cost = 1.0;
        for (pattern, (mass, child_subset)) in ordered {
            let branch_probability = mass / total_weight;
            let child_cost = if pattern == ALL_GREEN_PATTERN {
                0.0
            } else if child_subset.len() == subset.len() && child_subset == subset {
                return Ok(f64::INFINITY);
            } else {
                self.exact_best_cost(&child_subset, weights, small_state_table, memo)?
            };
            cost += branch_probability * child_cost;
            if cost >= best_bound {
                return Ok(cost);
            }
        }

        Ok(cost)
    }

    fn exact_best_cost(
        &self,
        subset: &[usize],
        weights: &[f64],
        small_state_table: &SmallStateTable,
        memo: &mut HashMap<ExactSubsetKey, f64>,
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

        let scores = match exact_suggestion_mode(&self.config, subset.len()) {
            Some(ExactSuggestionMode::Exhaustive) => (0..self.guesses.len()).collect::<Vec<_>>(),
            Some(ExactSuggestionMode::Pooled) | None => {
                self.top_guess_indexes_for_subset(subset, weights, self.config.exact_candidate_pool)
            }
        };
        let lower_bound = small_state_table.lower_bound(subset.len());
        let mut best_cost = f64::INFINITY;
        for guess_index in scores {
            let cost = self.exact_cost_for_guess(
                guess_index,
                subset,
                weights,
                small_state_table,
                memo,
                best_cost,
            )?;
            if cost < best_cost {
                best_cost = cost;
                if best_cost <= lower_bound {
                    break;
                }
            }
        }
        memo.insert(key, best_cost);
        Ok(best_cost)
    }

    fn top_guess_indexes_for_subset(
        &self,
        subset: &[usize],
        weights: &[f64],
        count: usize,
    ) -> Vec<usize> {
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut scores = (0..self.guesses.len())
            .into_par_iter()
            .map(|guess_index| {
                let mut masses = [0.0f64; PATTERN_SPACE];
                for answer_index in subset {
                    let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
                    masses[pattern] += weights[*answer_index];
                }
                let entropy = masses
                    .iter()
                    .filter(|mass| **mass > 0.0)
                    .map(|mass| {
                        let probability = mass / total_weight;
                        -probability * probability.log2()
                    })
                    .sum::<f64>();
                (guess_index, entropy)
            })
            .collect::<Vec<_>>();
        scores.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        scores.truncate(count);
        scores.into_iter().map(|(index, _)| index).collect()
    }
}

#[derive(Clone, Debug)]
struct PriorMetrics {
    target_probability: f64,
    target_rank: usize,
    log_loss: f64,
    brier: f64,
}

fn exact_suggestion_mode(
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

fn compare_suggestions(
    left: &Suggestion,
    right: &Suggestion,
    surviving_words: &HashSet<&str>,
) -> std::cmp::Ordering {
    right
        .entropy
        .total_cmp(&left.entropy)
        .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
        .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
        .then_with(|| {
            let left_surviving = surviving_words.contains(left.word.as_str());
            let right_surviving = surviving_words.contains(right.word.as_str());
            right_surviving.cmp(&left_surviving)
        })
        .then_with(|| left.word.cmp(&right.word))
}

fn compare_exact(
    left: &Suggestion,
    right: &Suggestion,
    surviving_words: &HashSet<&str>,
) -> std::cmp::Ordering {
    match (left.exact_cost, right.exact_cost) {
        (Some(left_cost), Some(right_cost)) => left_cost
            .total_cmp(&right_cost)
            .then_with(|| compare_suggestions(left, right, surviving_words)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right, surviving_words),
    }
}

fn compare_exact_costs(
    left: &Suggestion,
    right: &Suggestion,
    left_cost: Option<f64>,
    right_cost: Option<f64>,
    surviving_words: &HashSet<&str>,
) -> std::cmp::Ordering {
    match (left_cost, right_cost) {
        (Some(left_cost), Some(right_cost)) => left_cost
            .total_cmp(&right_cost)
            .then_with(|| compare_suggestions(left, right, surviving_words)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => compare_suggestions(left, right, surviving_words),
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use crate::{
        config::PriorConfig,
        model::AnswerRecord,
        scoring::{format_feedback_letters, score_guess},
    };

    use super::{ExactSuggestionMode, Solver, exact_suggestion_mode};

    #[test]
    fn parse_observations_rejects_length_mismatch() {
        let error = Solver::parse_observations(&["crane".into()], &[]).expect_err("must fail");
        assert!(error.to_string().contains("same number"));
    }

    #[test]
    fn target_feedback_matches_expected_fixture() {
        assert_eq!(
            format_feedback_letters(score_guess("lilly", "alley")),
            "ybgbg"
        );
    }

    #[test]
    fn initial_state_includes_seed_words() {
        let config = PriorConfig::default();
        let answer = AnswerRecord {
            word: "cigar".into(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid")],
        };
        let state_weight = crate::model::weight_snapshot(
            &answer,
            &config,
            NaiveDate::from_ymd_opt(2026, 3, 1).expect("valid"),
        );
        assert!(state_weight.final_weight > 0.0);
    }

    #[test]
    fn exact_mode_uses_exhaustive_search_for_tiny_states() {
        let config = PriorConfig::default();
        assert_eq!(
            exact_suggestion_mode(&config, config.exact_exhaustive_threshold),
            Some(ExactSuggestionMode::Exhaustive)
        );
    }

    #[test]
    fn exact_mode_keeps_pooled_search_between_thresholds() {
        let mut config = PriorConfig::default();
        config.exact_threshold = 16;
        config.exact_exhaustive_threshold = 8;
        assert_eq!(
            exact_suggestion_mode(&config, 12),
            Some(ExactSuggestionMode::Pooled)
        );
        assert_eq!(exact_suggestion_mode(&config, 17), None);
    }
}
