use anyhow::{Result, bail};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameOutcomeStatus {
    Solved,
    Unsolved,
    CoverageGap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GameOutcome {
    pub date: NaiveDate,
    pub status: GameOutcomeStatus,
    pub guesses: Option<usize>,
}

impl GameOutcome {
    pub fn solved(date: NaiveDate, guesses: usize) -> Self {
        Self {
            date,
            status: GameOutcomeStatus::Solved,
            guesses: Some(guesses),
        }
    }

    pub fn unsolved(date: NaiveDate, guesses: usize) -> Self {
        Self {
            date,
            status: GameOutcomeStatus::Unsolved,
            guesses: Some(guesses),
        }
    }

    pub fn coverage_gap(date: NaiveDate) -> Self {
        Self {
            date,
            status: GameOutcomeStatus::CoverageGap,
            guesses: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricInterval {
    pub lower: f64,
    pub upper: f64,
}

impl MetricInterval {
    fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BootstrapConfig {
    pub resamples: usize,
    pub block_length: usize,
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            resamples: 2_000,
            block_length: 7,
            seed: 0x4d_57_4f_52_44_4c_45,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictiveMetrics {
    pub scheduled_games: usize,
    pub modeled_games: usize,
    pub solved_games: usize,
    pub unsolved_games: usize,
    pub coverage_gaps: usize,
    pub coverage_rate: f64,
    pub solve_rate: f64,
    pub conditional_mean_guesses: f64,
    pub conditional_mean_guesses_ci95: MetricInterval,
    pub failure_penalty_guesses: f64,
    pub all_game_penalized_mean_guesses: f64,
    pub all_game_penalized_mean_guesses_ci95: MetricInterval,
    pub median_guesses: f64,
    pub p90_guesses: usize,
    pub p95_guesses: usize,
    pub max_guesses: usize,
    pub solved_in_guess_counts: [usize; 6],
    pub coverage_rate_ci95: MetricInterval,
    pub solve_rate_ci95: MetricInterval,
    pub bootstrap: BootstrapConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedDifference {
    pub candidate_minus_baseline: f64,
    pub ci95: MetricInterval,
    pub candidate_wins: usize,
    pub ties: usize,
    pub baseline_wins: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbabilityScore {
    pub target_probability: f64,
    pub log_loss: f64,
    pub brier: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedProbabilityObservation {
    pub target_rank: usize,
    pub top_probability: f64,
    pub top_prediction_correct: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriorEvidenceMetrics {
    pub measured_games: usize,
    pub top_1_recall: f64,
    pub top_1_recall_ci95: MetricInterval,
    pub top_3_recall: f64,
    pub top_3_recall_ci95: MetricInterval,
    pub top_5_recall: f64,
    pub top_5_recall_ci95: MetricInterval,
    pub calibration_bins: usize,
    pub expected_calibration_error: f64,
    pub expected_calibration_error_ci95: MetricInterval,
    pub bootstrap: BootstrapConfig,
}

pub fn summarize_ranked_probability_observations(
    observations: &[RankedProbabilityObservation],
    calibration_bins: usize,
    bootstrap: BootstrapConfig,
) -> Result<PriorEvidenceMetrics> {
    if observations.is_empty() {
        bail!("prior evidence requires at least one observation");
    }
    if calibration_bins == 0 {
        bail!("calibration bin count must be positive");
    }
    if bootstrap.resamples == 0 || bootstrap.block_length == 0 {
        bail!("bootstrap resamples and block length must be positive");
    }
    if observations.iter().any(|observation| {
        observation.target_rank == 0
            || !observation.top_probability.is_finite()
            || !(0.0..=1.0).contains(&observation.top_probability)
    }) {
        bail!(
            "ranked probability observations must have a positive rank and finite [0, 1] confidence"
        );
    }

    let measured_games = observations.len();
    let top_1 = observations
        .iter()
        .filter(|observation| observation.target_rank <= 1)
        .count();
    let top_3 = observations
        .iter()
        .filter(|observation| observation.target_rank <= 3)
        .count();
    let top_5 = observations
        .iter()
        .filter(|observation| observation.target_rank <= 5)
        .count();
    let expected_calibration_error = calibration_error(observations, calibration_bins);
    let expected_calibration_error_ci95 =
        block_bootstrap_ranked_interval(observations, bootstrap, |sample| {
            calibration_error(sample, calibration_bins)
        });

    Ok(PriorEvidenceMetrics {
        measured_games,
        top_1_recall: top_1 as f64 / measured_games as f64,
        top_1_recall_ci95: wilson_interval(top_1, measured_games),
        top_3_recall: top_3 as f64 / measured_games as f64,
        top_3_recall_ci95: wilson_interval(top_3, measured_games),
        top_5_recall: top_5 as f64 / measured_games as f64,
        top_5_recall_ci95: wilson_interval(top_5, measured_games),
        calibration_bins,
        expected_calibration_error,
        expected_calibration_error_ci95,
        bootstrap,
    })
}

pub fn score_multiclass_probabilities(
    probabilities: &[f64],
    target_index: usize,
) -> Result<ProbabilityScore> {
    if probabilities.is_empty() {
        bail!("probability vector must not be empty");
    }
    if target_index >= probabilities.len() {
        bail!("target index is outside the probability vector");
    }
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        bail!("probabilities must be finite and non-negative");
    }
    let total = probabilities.iter().sum::<f64>();
    if (total - 1.0).abs() > 1e-9 {
        bail!("probabilities must sum to one, got {total:.12}");
    }

    let target_probability = probabilities[target_index];
    let brier = probabilities
        .iter()
        .enumerate()
        .map(|(index, probability)| {
            let observed = f64::from(index == target_index);
            (probability - observed).powi(2)
        })
        .sum();
    Ok(ProbabilityScore {
        target_probability,
        log_loss: -(target_probability.max(1e-12)).ln(),
        brier,
    })
}

pub fn summarize_predictive_outcomes(
    outcomes: &[GameOutcome],
    failure_penalty_guesses: f64,
    bootstrap: BootstrapConfig,
) -> Result<PredictiveMetrics> {
    validate_inputs(outcomes, failure_penalty_guesses, bootstrap)?;

    let scheduled_games = outcomes.len();
    let modeled = outcomes
        .iter()
        .filter_map(|outcome| outcome.guesses.map(|guesses| guesses as f64))
        .collect::<Vec<_>>();
    let modeled_games = modeled.len();
    let solved_games = outcomes
        .iter()
        .filter(|outcome| outcome.status == GameOutcomeStatus::Solved)
        .count();
    let unsolved_games = outcomes
        .iter()
        .filter(|outcome| outcome.status == GameOutcomeStatus::Unsolved)
        .count();
    let coverage_gaps = outcomes
        .iter()
        .filter(|outcome| outcome.status == GameOutcomeStatus::CoverageGap)
        .count();
    let conditional_mean_guesses = mean(&modeled);
    let all_game_values = outcomes
        .iter()
        .map(|outcome| match outcome.status {
            GameOutcomeStatus::Solved => outcome.guesses.unwrap_or_default() as f64,
            GameOutcomeStatus::Unsolved | GameOutcomeStatus::CoverageGap => failure_penalty_guesses,
        })
        .collect::<Vec<_>>();
    let all_game_penalized_mean_guesses = mean(&all_game_values);
    let mut sorted_modeled = modeled.clone();
    sorted_modeled.sort_by(f64::total_cmp);
    let solved_in_guess_counts = solved_distribution(outcomes);

    Ok(PredictiveMetrics {
        scheduled_games,
        modeled_games,
        solved_games,
        unsolved_games,
        coverage_gaps,
        coverage_rate: modeled_games as f64 / scheduled_games as f64,
        solve_rate: solved_games as f64 / scheduled_games as f64,
        conditional_mean_guesses,
        conditional_mean_guesses_ci95: block_bootstrap_interval(outcomes, bootstrap, |sample| {
            let values = sample
                .iter()
                .filter_map(|outcome| outcome.guesses.map(|guesses| guesses as f64))
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| mean(&values))
        }),
        failure_penalty_guesses,
        all_game_penalized_mean_guesses,
        all_game_penalized_mean_guesses_ci95: block_bootstrap_interval(
            outcomes,
            bootstrap,
            |sample| {
                Some(mean(
                    &sample
                        .iter()
                        .map(|outcome| match outcome.status {
                            GameOutcomeStatus::Solved => outcome.guesses.unwrap_or_default() as f64,
                            GameOutcomeStatus::Unsolved | GameOutcomeStatus::CoverageGap => {
                                failure_penalty_guesses
                            }
                        })
                        .collect::<Vec<_>>(),
                ))
            },
        ),
        median_guesses: median(&sorted_modeled),
        p90_guesses: integer_quantile(&sorted_modeled, 0.90),
        p95_guesses: integer_quantile(&sorted_modeled, 0.95),
        max_guesses: sorted_modeled.last().copied().unwrap_or_default() as usize,
        solved_in_guess_counts,
        coverage_rate_ci95: wilson_interval(modeled_games, scheduled_games),
        solve_rate_ci95: wilson_interval(solved_games, scheduled_games),
        bootstrap,
    })
}

impl PairedDifference {
    pub fn all_game_penalized(
        baseline: &[GameOutcome],
        candidate: &[GameOutcome],
        failure_penalty_guesses: f64,
        bootstrap: BootstrapConfig,
    ) -> Result<Self> {
        if baseline.len() != candidate.len() || baseline.is_empty() {
            bail!("paired outcomes must have the same non-zero length");
        }
        if baseline
            .iter()
            .zip(candidate)
            .any(|(left, right)| left.date != right.date)
        {
            bail!("paired outcomes must use identical dates in identical order");
        }
        validate_inputs(baseline, failure_penalty_guesses, bootstrap)?;
        validate_inputs(candidate, failure_penalty_guesses, bootstrap)?;

        let differences = baseline
            .iter()
            .zip(candidate)
            .map(|(baseline, candidate)| {
                penalized_value(candidate, failure_penalty_guesses)
                    - penalized_value(baseline, failure_penalty_guesses)
            })
            .collect::<Vec<_>>();
        let candidate_wins = differences
            .iter()
            .filter(|difference| **difference < 0.0)
            .count();
        let ties = differences
            .iter()
            .filter(|difference| **difference == 0.0)
            .count();
        let baseline_wins = differences
            .iter()
            .filter(|difference| **difference > 0.0)
            .count();
        let ci95 = block_bootstrap_numeric_interval(&differences, bootstrap);

        Ok(Self {
            candidate_minus_baseline: mean(&differences),
            ci95,
            candidate_wins,
            ties,
            baseline_wins,
        })
    }
}

fn validate_inputs(
    outcomes: &[GameOutcome],
    failure_penalty_guesses: f64,
    bootstrap: BootstrapConfig,
) -> Result<()> {
    if outcomes.is_empty() {
        bail!("predictive metrics require at least one scheduled game");
    }
    if !failure_penalty_guesses.is_finite() || failure_penalty_guesses <= 0.0 {
        bail!("failure penalty must be finite and positive");
    }
    if bootstrap.resamples == 0 || bootstrap.block_length == 0 {
        bail!("bootstrap resamples and block length must be positive");
    }
    if outcomes.windows(2).any(|pair| pair[0].date >= pair[1].date) {
        bail!("predictive outcomes must be strictly chronological");
    }
    for outcome in outcomes {
        match (outcome.status, outcome.guesses) {
            (GameOutcomeStatus::Solved, Some(1..=6))
            | (GameOutcomeStatus::Unsolved, Some(1..=6))
            | (GameOutcomeStatus::CoverageGap, None) => {}
            _ => bail!("invalid outcome for {}", outcome.date),
        }
    }
    Ok(())
}

fn solved_distribution(outcomes: &[GameOutcome]) -> [usize; 6] {
    let mut counts = [0usize; 6];
    for outcome in outcomes {
        if outcome.status == GameOutcomeStatus::Solved
            && let Some(guesses @ 1..=6) = outcome.guesses
        {
            counts[guesses - 1] += 1;
        }
    }
    counts
}

fn penalized_value(outcome: &GameOutcome, failure_penalty_guesses: f64) -> f64 {
    match outcome.status {
        GameOutcomeStatus::Solved => outcome.guesses.unwrap_or_default() as f64,
        GameOutcomeStatus::Unsolved | GameOutcomeStatus::CoverageGap => failure_penalty_guesses,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(sorted: &[f64]) -> f64 {
    match sorted.len() {
        0 => 0.0,
        len if len % 2 == 1 => sorted[len / 2],
        len => (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0,
    }
}

fn integer_quantile(sorted: &[f64], probability: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64) * probability).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)] as usize
}

fn block_bootstrap_interval<F>(
    outcomes: &[GameOutcome],
    config: BootstrapConfig,
    statistic: F,
) -> MetricInterval
where
    F: Fn(&[GameOutcome]) -> Option<f64>,
{
    if outcomes.len() == 1 {
        return statistic(outcomes).map_or(MetricInterval::point(0.0), MetricInterval::point);
    }
    let mut rng = SplitMix64::new(config.seed);
    let mut sample = Vec::with_capacity(outcomes.len());
    let mut estimates = Vec::with_capacity(config.resamples);
    for _ in 0..config.resamples {
        sample.clear();
        fill_block_sample(outcomes, config.block_length, &mut rng, &mut sample);
        if let Some(value) = statistic(&sample)
            && value.is_finite()
        {
            estimates.push(value);
        }
    }
    percentile_interval(estimates)
}

fn block_bootstrap_numeric_interval(values: &[f64], config: BootstrapConfig) -> MetricInterval {
    if values.len() == 1 {
        return MetricInterval::point(values[0]);
    }
    let mut rng = SplitMix64::new(config.seed);
    let mut estimates = Vec::with_capacity(config.resamples);
    let block_length = config.block_length.min(values.len());
    for _ in 0..config.resamples {
        let mut total = 0.0;
        let mut sampled = 0usize;
        while sampled < values.len() {
            let start = rng.index(values.len());
            for offset in 0..block_length {
                if sampled == values.len() {
                    break;
                }
                total += values[(start + offset) % values.len()];
                sampled += 1;
            }
        }
        estimates.push(total / values.len() as f64);
    }
    percentile_interval(estimates)
}

fn block_bootstrap_ranked_interval<F>(
    observations: &[RankedProbabilityObservation],
    config: BootstrapConfig,
    statistic: F,
) -> MetricInterval
where
    F: Fn(&[RankedProbabilityObservation]) -> f64,
{
    if observations.len() == 1 {
        return MetricInterval::point(statistic(observations));
    }
    let mut rng = SplitMix64::new(config.seed);
    let mut sample = Vec::with_capacity(observations.len());
    let mut estimates = Vec::with_capacity(config.resamples);
    let block_length = config.block_length.min(observations.len());
    for _ in 0..config.resamples {
        sample.clear();
        while sample.len() < observations.len() {
            let start = rng.index(observations.len());
            for offset in 0..block_length {
                if sample.len() == observations.len() {
                    break;
                }
                sample.push(observations[(start + offset) % observations.len()]);
            }
        }
        estimates.push(statistic(&sample));
    }
    percentile_interval(estimates)
}

fn calibration_error(
    observations: &[RankedProbabilityObservation],
    calibration_bins: usize,
) -> f64 {
    let mut counts = vec![0usize; calibration_bins];
    let mut confidence_sums = vec![0.0; calibration_bins];
    let mut correct_counts = vec![0usize; calibration_bins];
    for observation in observations {
        let bin = ((observation.top_probability * calibration_bins as f64).floor() as usize)
            .min(calibration_bins - 1);
        counts[bin] += 1;
        confidence_sums[bin] += observation.top_probability;
        correct_counts[bin] += usize::from(observation.top_prediction_correct);
    }
    counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(bin, count)| {
            let weight = *count as f64 / observations.len() as f64;
            let confidence = confidence_sums[bin] / *count as f64;
            let accuracy = correct_counts[bin] as f64 / *count as f64;
            weight * (confidence - accuracy).abs()
        })
        .sum()
}

fn fill_block_sample(
    outcomes: &[GameOutcome],
    block_length: usize,
    rng: &mut SplitMix64,
    sample: &mut Vec<GameOutcome>,
) {
    let block_length = block_length.min(outcomes.len());
    while sample.len() < outcomes.len() {
        let start = rng.index(outcomes.len());
        for offset in 0..block_length {
            if sample.len() == outcomes.len() {
                break;
            }
            sample.push(outcomes[(start + offset) % outcomes.len()]);
        }
    }
}

fn percentile_interval(mut estimates: Vec<f64>) -> MetricInterval {
    if estimates.is_empty() {
        return MetricInterval::point(0.0);
    }
    estimates.sort_by(f64::total_cmp);
    let lower = percentile(&estimates, 0.025);
    let upper = percentile(&estimates, 0.975);
    MetricInterval { lower, upper }
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len().saturating_sub(1)) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn wilson_interval(successes: usize, trials: usize) -> MetricInterval {
    if trials == 0 {
        return MetricInterval::point(0.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let n = trials as f64;
    let probability = successes as f64 / n;
    let denominator = 1.0 + z * z / n;
    let center = (probability + z * z / (2.0 * n)) / denominator;
    let margin =
        z * ((probability * (1.0 - probability) / n + z * z / (4.0 * n * n)).sqrt()) / denominator;
    MetricInterval {
        lower: (center - margin).max(0.0),
        upper: (center + margin).min(1.0),
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next() % length as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Days;

    fn outcomes() -> Vec<GameOutcome> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).expect("date");
        [
            GameOutcomeStatus::Solved,
            GameOutcomeStatus::Solved,
            GameOutcomeStatus::Solved,
            GameOutcomeStatus::Unsolved,
            GameOutcomeStatus::CoverageGap,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, status)| {
            let date = start
                .checked_add_days(Days::new(index as u64))
                .expect("date");
            match status {
                GameOutcomeStatus::Solved => GameOutcome::solved(date, index + 2),
                GameOutcomeStatus::Unsolved => GameOutcome::unsolved(date, 6),
                GameOutcomeStatus::CoverageGap => GameOutcome::coverage_gap(date),
            }
        })
        .collect()
    }

    #[test]
    fn metrics_keep_gaps_in_all_game_denominators() {
        let metrics = summarize_predictive_outcomes(
            &outcomes(),
            7.0,
            BootstrapConfig {
                resamples: 200,
                block_length: 2,
                seed: 7,
            },
        )
        .expect("metrics");

        assert_eq!(metrics.scheduled_games, 5);
        assert_eq!(metrics.modeled_games, 4);
        assert_eq!(metrics.solved_games, 3);
        assert_eq!(metrics.unsolved_games, 1);
        assert_eq!(metrics.coverage_gaps, 1);
        assert_eq!(metrics.coverage_rate, 0.8);
        assert_eq!(metrics.solve_rate, 0.6);
        assert_eq!(metrics.conditional_mean_guesses, 3.75);
        assert_eq!(metrics.all_game_penalized_mean_guesses, 4.6);
        assert_eq!(metrics.solved_in_guess_counts, [0, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let config = BootstrapConfig {
            resamples: 250,
            block_length: 3,
            seed: 42,
        };
        let first = summarize_predictive_outcomes(&outcomes(), 7.0, config).expect("first");
        let second = summarize_predictive_outcomes(&outcomes(), 7.0, config).expect("second");
        assert_eq!(first, second);
    }

    #[test]
    fn ranked_probability_summary_reports_recall_and_calibration() {
        let observations = [
            RankedProbabilityObservation {
                target_rank: 1,
                top_probability: 0.8,
                top_prediction_correct: true,
            },
            RankedProbabilityObservation {
                target_rank: 2,
                top_probability: 0.6,
                top_prediction_correct: false,
            },
            RankedProbabilityObservation {
                target_rank: 5,
                top_probability: 0.4,
                top_prediction_correct: false,
            },
            RankedProbabilityObservation {
                target_rank: 8,
                top_probability: 0.2,
                top_prediction_correct: false,
            },
        ];
        let metrics = summarize_ranked_probability_observations(
            &observations,
            5,
            BootstrapConfig {
                resamples: 100,
                block_length: 2,
                seed: 19,
            },
        )
        .expect("ranked metrics");

        assert_eq!(metrics.measured_games, 4);
        assert_eq!(metrics.top_1_recall, 0.25);
        assert_eq!(metrics.top_3_recall, 0.5);
        assert_eq!(metrics.top_5_recall, 0.75);
        assert!((metrics.expected_calibration_error - 0.35).abs() <= 1e-12);
        assert!(metrics.expected_calibration_error_ci95.lower >= 0.0);
        assert!(metrics.expected_calibration_error_ci95.upper <= 1.0);
    }

    #[test]
    fn ranked_probability_summary_rejects_invalid_confidence() {
        let error = summarize_ranked_probability_observations(
            &[RankedProbabilityObservation {
                target_rank: 1,
                top_probability: 1.1,
                top_prediction_correct: true,
            }],
            10,
            BootstrapConfig::default(),
        )
        .expect_err("invalid confidence");
        assert!(error.to_string().contains("finite [0, 1] confidence"));
    }

    #[test]
    fn paired_comparison_requires_matching_dates() {
        let baseline = outcomes();
        let mut candidate = baseline.clone();
        candidate[0].date = candidate[0]
            .date
            .checked_add_days(Days::new(1))
            .expect("date");
        let error = PairedDifference::all_game_penalized(
            &baseline,
            &candidate,
            7.0,
            BootstrapConfig::default(),
        )
        .expect_err("dates must match");
        assert!(error.to_string().contains("identical dates"));
    }

    #[test]
    fn paired_comparison_preserves_date_pairing_and_direction() {
        let baseline = outcomes();
        let mut candidate = baseline.clone();
        candidate[0].guesses = Some(1);
        let comparison = PairedDifference::all_game_penalized(
            &baseline,
            &candidate,
            7.0,
            BootstrapConfig {
                resamples: 200,
                block_length: 2,
                seed: 99,
            },
        )
        .expect("comparison");

        assert_eq!(comparison.candidate_minus_baseline, -0.2);
        assert_eq!(comparison.candidate_wins, 1);
        assert_eq!(comparison.ties, 4);
        assert_eq!(comparison.baseline_wins, 0);
        assert!(comparison.ci95.lower <= comparison.candidate_minus_baseline);
        assert!(comparison.ci95.upper >= comparison.candidate_minus_baseline);
    }

    #[test]
    fn multiclass_scores_match_hand_computed_example() {
        let score = score_multiclass_probabilities(&[0.7, 0.2, 0.1], 1).expect("score");
        assert_eq!(score.target_probability, 0.2);
        assert!((score.log_loss - (-0.2_f64.ln())).abs() <= 1e-12);
        assert!((score.brier - 1.14).abs() <= 1e-12);
    }

    #[test]
    fn multiclass_scores_reject_invalid_probability_mass() {
        assert!(
            score_multiclass_probabilities(&[], 0)
                .expect_err("empty vector must fail")
                .to_string()
                .contains("must not be empty")
        );
        assert!(
            score_multiclass_probabilities(&[1.0], 1)
                .expect_err("target outside vector must fail")
                .to_string()
                .contains("outside")
        );
        let error = score_multiclass_probabilities(&[0.6, 0.3], 0)
            .expect_err("unnormalized probabilities must fail");
        assert!(error.to_string().contains("sum to one"));
        for probabilities in [
            vec![1.1, -0.1],
            vec![f64::NAN, 0.0],
            vec![f64::INFINITY, 0.0],
        ] {
            let error = score_multiclass_probabilities(&probabilities, 0)
                .expect_err("non-finite or negative probabilities must fail");
            assert!(error.to_string().contains("finite and non-negative"));
        }
        let near_normalized = score_multiclass_probabilities(&[0.5, 0.500_000_000_5], 1)
            .expect("mass within tolerance");
        assert!(near_normalized.log_loss.is_finite());
        assert!(near_normalized.brier.is_finite());
    }
}
