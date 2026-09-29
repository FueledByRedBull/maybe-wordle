//! Leakage-safe learned proxy ranking.
//!
//! The production solver already computes an inexpensive proxy continuation
//! score.  This module fits a deterministic ridge model to the *residual* between
//! that score and exhaustive Bellman cost.  Training rows are explicitly marked
//! with a split and the fitter accepts training rows only; callers must build those
//! rows from a chronological/grouped holdout plan before invoking this module.
//! Inference and evaluation reject malformed features, invalid artifacts, and
//! non-finite model output rather than substituting baseline scores.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const LEARNED_PROXY_FORMAT_VERSION: u32 = 1;

pub fn learned_proxy_feature_names() -> Vec<String> {
    [
        "entropy",
        "solve_probability",
        "expected_remaining",
        "force_in_two",
        "worst_non_green_bucket_size",
        "largest_non_green_bucket_mass",
        "high_mass_ambiguous_bucket_count",
        "smoothness_penalty",
        "large_non_green_bucket_count",
        "dangerous_mass_bucket_count",
        "non_green_mass_in_large_buckets",
        "posterior_answer_probability",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProxySplit {
    #[default]
    Train,
    Validation,
    Test,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyTrainingRow {
    pub state_id: String,
    pub trajectory_id: String,
    pub date: NaiveDate,
    pub step_index: usize,
    pub guess: String,
    pub features: Vec<f64>,
    /// Existing (hand-designed) proxy cost.  Lower is better.
    pub baseline_proxy_cost: f64,
    /// Exhaustive Bellman cost, including the current guess.
    pub exact_continuation_cost: f64,
    #[serde(default)]
    pub split: ProxySplit,
}

impl ProxyTrainingRow {
    pub fn validate(&self, expected_features: Option<usize>) -> Result<()> {
        for (label, value) in [
            ("state id", self.state_id.as_str()),
            ("trajectory id", self.trajectory_id.as_str()),
            ("guess", self.guess.as_str()),
        ] {
            ensure!(
                !value.trim().is_empty(),
                "proxy row {} must not be empty",
                label
            );
        }
        if let Some(expected) = expected_features {
            ensure!(
                self.features.len() == expected,
                "proxy row {} has {} features, expected {}",
                self.state_id,
                self.features.len(),
                expected
            );
        }
        ensure!(
            self.features.iter().all(|value| value.is_finite()),
            "proxy row {} has non-finite feature",
            self.state_id
        );
        ensure!(
            self.baseline_proxy_cost.is_finite() && self.baseline_proxy_cost >= 0.0,
            "proxy row {} has invalid baseline cost {}",
            self.state_id,
            self.baseline_proxy_cost
        );
        ensure!(
            self.exact_continuation_cost.is_finite() && self.exact_continuation_cost >= 1.0,
            "proxy row {} has invalid exact cost {}",
            self.state_id,
            self.exact_continuation_cost
        );
        Ok(())
    }

    pub fn key(&self) -> String {
        format!(
            "{}\u{001f}{:010}\u{001f}{}",
            self.state_id, self.step_index, self.guess
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RidgeConfig {
    /// L2 penalty.  The intercept is not regularized.
    pub lambda: f64,
    pub standardize: bool,
    pub fit_intercept: bool,
    pub minimum_scale: f64,
}

impl Default for RidgeConfig {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            standardize: true,
            fit_intercept: true,
            minimum_scale: 1e-12,
        }
    }
}

impl RidgeConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.lambda.is_finite() && self.lambda > 0.0,
            "ridge lambda must be positive"
        );
        ensure!(
            self.minimum_scale.is_finite() && self.minimum_scale > 0.0,
            "ridge minimum scale must be positive"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureScaling {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
}

impl FeatureScaling {
    pub fn validate(&self, feature_count: usize) -> Result<()> {
        ensure!(
            self.means.len() == feature_count && self.scales.len() == feature_count,
            "feature scaling dimension does not match schema"
        );
        ensure!(
            self.means.iter().all(|value| value.is_finite())
                && self
                    .scales
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
            "feature scaling means/scales must be finite and scales positive"
        );
        Ok(())
    }

    pub fn transform(&self, values: &[f64]) -> Result<Vec<f64>> {
        self.validate(values.len())?;
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "proxy feature vector contains non-finite values"
        );
        let transformed = values
            .iter()
            .zip(self.means.iter().zip(&self.scales))
            .map(|(value, (mean, scale))| (value - mean) / scale)
            .collect::<Vec<_>>();
        ensure!(
            transformed.iter().all(|value| value.is_finite()),
            "transformed proxy feature vector contains non-finite values"
        );
        Ok(transformed)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyModelProvenance {
    pub dataset_identity: String,
    pub replay_identity: String,
    pub split_policy: String,
    pub feature_schema_digest: String,
}

impl Default for ProxyModelProvenance {
    fn default() -> Self {
        Self {
            dataset_identity: "unspecified".to_string(),
            replay_identity: "unspecified".to_string(),
            split_policy: "train_only".to_string(),
            feature_schema_digest: "unspecified".to_string(),
        }
    }
}

impl ProxyModelProvenance {
    pub fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("dataset identity", self.dataset_identity.as_str()),
            ("replay identity", self.replay_identity.as_str()),
            ("split policy", self.split_policy.as_str()),
            ("feature schema digest", self.feature_schema_digest.as_str()),
        ] {
            ensure!(
                !value.trim().is_empty(),
                "proxy {} must not be empty",
                label
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnedProxyModel {
    pub format_version: u32,
    pub feature_names: Vec<String>,
    pub scaling: FeatureScaling,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub ridge_lambda: f64,
    pub standardize: bool,
    pub fit_intercept: bool,
    pub target_mean_residual: f64,
    pub training_rows: usize,
    pub training_states: usize,
    pub training_trajectories: usize,
    pub provenance: ProxyModelProvenance,
}

impl LearnedProxyModel {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == LEARNED_PROXY_FORMAT_VERSION,
            "unsupported learned proxy format {}; expected {}",
            self.format_version,
            LEARNED_PROXY_FORMAT_VERSION
        );
        ensure!(
            !self.feature_names.is_empty(),
            "learned proxy feature schema is empty"
        );
        let mut names = BTreeSet::new();
        for name in &self.feature_names {
            ensure!(
                !name.trim().is_empty(),
                "learned proxy feature name is empty"
            );
            ensure!(
                names.insert(name),
                "duplicate learned proxy feature name {}",
                name
            );
        }
        self.scaling.validate(self.feature_names.len())?;
        ensure!(
            self.coefficients.len() == self.feature_names.len(),
            "learned proxy coefficient dimension does not match schema"
        );
        ensure!(
            self.coefficients.iter().all(|value| value.is_finite())
                && self.intercept.is_finite()
                && self.ridge_lambda.is_finite()
                && self.ridge_lambda > 0.0
                && self.target_mean_residual.is_finite(),
            "learned proxy parameters must be finite and ridge lambda positive"
        );
        ensure!(self.training_rows > 0, "learned proxy has no training rows");
        ensure!(
            self.training_states > 0,
            "learned proxy has no training states"
        );
        ensure!(
            self.training_trajectories > 0,
            "learned proxy has no training trajectories"
        );
        if !self.standardize {
            ensure!(
                self.scaling.means.iter().all(|value| *value == 0.0)
                    && self.scaling.scales.iter().all(|value| *value == 1.0),
                "unstandardized learned proxy must use identity feature scaling"
            );
        }
        if !self.fit_intercept {
            ensure!(
                self.intercept == 0.0 && self.target_mean_residual == 0.0,
                "learned proxy without an intercept must store zero intercept/target mean"
            );
        }
        ensure!(
            self.provenance.feature_schema_digest == feature_schema_digest(&self.feature_names),
            "learned proxy feature schema digest does not match feature names"
        );
        self.provenance.validate()
    }

    /// Return a learned prediction, rejecting invalid models, features, baselines,
    /// or scores. Lower scores rank earlier; errors never become baseline scores.
    pub fn predict(&self, features: &[f64], baseline: f64) -> Result<ProxyPrediction> {
        self.validate()?;
        ensure!(
            baseline.is_finite() && baseline >= 0.0,
            "proxy baseline must be finite and non-negative"
        );
        let transformed = self.scaling.transform(features)?;
        let residual = self.intercept
            + self
                .coefficients
                .iter()
                .zip(transformed)
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>();
        ensure!(residual.is_finite(), "learned proxy residual is non-finite");
        let score = baseline + residual;
        ensure!(
            score.is_finite() && score >= 0.0,
            "learned proxy score must be finite and non-negative"
        );
        Ok(ProxyPrediction { score, residual })
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(source: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(source)?;
        model.validate()?;
        Ok(model)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyPrediction {
    pub score: f64,
    pub residual: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyModelArtifact {
    pub format_version: u32,
    pub model: LearnedProxyModel,
    #[serde(default)]
    pub evaluation: Option<ProxyRankingMetrics>,
}

impl ProxyModelArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == LEARNED_PROXY_FORMAT_VERSION,
            "unsupported proxy artifact format {}; expected {}",
            self.format_version,
            LEARNED_PROXY_FORMAT_VERSION
        );
        self.model.validate()
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(source: &str) -> Result<Self> {
        let artifact: Self = serde_json::from_str(source)?;
        artifact.validate()?;
        Ok(artifact)
    }
}

/// Fit a deterministic residual ridge model with optional standardization. All
/// supplied rows must be training rows; validation/test rows are rejected to make
/// accidental leakage visible at the API boundary.
pub fn fit_ridge_residual(
    rows: &[ProxyTrainingRow],
    feature_names: &[String],
    config: RidgeConfig,
    provenance: ProxyModelProvenance,
) -> Result<LearnedProxyModel> {
    config.validate()?;
    provenance.validate()?;
    ensure!(!rows.is_empty(), "ridge training requires at least one row");
    ensure!(
        !feature_names.is_empty(),
        "ridge feature schema must not be empty"
    );
    let mut names = BTreeSet::new();
    for name in feature_names {
        ensure!(
            !name.trim().is_empty(),
            "ridge feature name must not be empty"
        );
        ensure!(names.insert(name), "duplicate ridge feature name {}", name);
    }
    ensure!(
        provenance.feature_schema_digest == feature_schema_digest(feature_names),
        "proxy provenance feature schema digest does not match the training schema"
    );
    for row in rows {
        ensure!(
            row.split == ProxySplit::Train,
            "ridge training received {:?} row {}; fit only on train rows",
            row.split,
            row.state_id
        );
        row.validate(Some(feature_names.len()))?;
    }

    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|row| row.key());
    ensure!(
        ordered
            .windows(2)
            .all(|pair| pair[0].key() != pair[1].key()),
        "ridge training contains duplicate state/step/guess rows"
    );
    let feature_count = feature_names.len();
    let mut scaling = FeatureScaling {
        means: vec![0.0; feature_count],
        scales: vec![1.0; feature_count],
    };
    let mut feature_means = vec![0.0; feature_count];
    if config.standardize || config.fit_intercept {
        for row in &ordered {
            for (index, value) in row.features.iter().enumerate() {
                feature_means[index] += *value;
            }
        }
        for mean in &mut feature_means {
            *mean /= ordered.len() as f64;
        }
    }
    if config.standardize {
        scaling.means.clone_from(&feature_means);
        for row in &ordered {
            for (index, value) in row.features.iter().enumerate() {
                scaling.scales[index] += (*value - scaling.means[index]).powi(2);
            }
        }
        for scale in &mut scaling.scales {
            // The initial one keeps a constant feature well-conditioned; for a
            // varying feature use population standard deviation.
            let variance = (*scale - 1.0) / ordered.len() as f64;
            *scale = variance.sqrt().max(config.minimum_scale);
        }
    }
    let transformed = ordered
        .iter()
        .map(|row| {
            row.features
                .iter()
                .enumerate()
                .map(|(index, value)| (*value - feature_means[index]) / scaling.scales[index])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let targets = ordered
        .iter()
        .map(|row| row.exact_continuation_cost - row.baseline_proxy_cost)
        .collect::<Vec<_>>();
    let target_mean = if config.fit_intercept {
        targets.iter().sum::<f64>() / targets.len() as f64
    } else {
        0.0
    };
    let dimension = feature_count;
    let mut normal = vec![vec![0.0; dimension]; dimension];
    let mut rhs = vec![0.0; dimension];
    for (row, target) in transformed.iter().zip(&targets) {
        let centered = *target - target_mean;
        for left in 0..dimension {
            rhs[left] += row[left] * centered;
            for right in 0..dimension {
                normal[left][right] += row[left] * row[right];
            }
        }
    }
    for (diagonal, row) in normal.iter_mut().enumerate() {
        row[diagonal] += config.lambda;
    }
    let coefficients = solve_positive_definite(normal, rhs)?;
    // Unstandardized artifacts predict in raw coordinates, even though fitting
    // an unpenalized intercept requires centered features in the normal equations.
    let intercept = if !config.standardize && config.fit_intercept {
        target_mean
            - coefficients
                .iter()
                .zip(&feature_means)
                .map(|(coefficient, mean)| coefficient * mean)
                .sum::<f64>()
    } else {
        target_mean
    };
    let model = LearnedProxyModel {
        format_version: LEARNED_PROXY_FORMAT_VERSION,
        feature_names: feature_names.to_vec(),
        scaling,
        coefficients,
        intercept,
        ridge_lambda: config.lambda,
        standardize: config.standardize,
        fit_intercept: config.fit_intercept,
        target_mean_residual: target_mean,
        training_rows: ordered.len(),
        training_states: ordered
            .iter()
            .map(|row| row.state_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        training_trajectories: ordered
            .iter()
            .map(|row| row.trajectory_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        provenance,
    };
    model.validate()?;
    Ok(model)
}

fn solve_positive_definite(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let dimension = rhs.len();
    ensure!(
        matrix.len() == dimension && matrix.iter().all(|row| row.len() == dimension),
        "ridge normal equation dimensions do not agree"
    );
    let original_matrix = matrix.clone();
    let original_rhs = rhs.clone();
    // Deterministic Gauss-Jordan elimination with a stable largest-pivot choice.
    for column in 0..dimension {
        let mut pivot = column;
        for row in (column + 1)..dimension {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        ensure!(
            matrix[pivot][column].is_finite() && matrix[pivot][column].abs() > 1e-15,
            "ridge normal equations are singular at feature {}",
            column
        );
        if pivot != column {
            matrix.swap(column, pivot);
            rhs.swap(column, pivot);
        }
        let divisor = matrix[column][column];
        for value in &mut matrix[column][column..] {
            *value /= divisor;
        }
        rhs[column] /= divisor;
        let pivot_tail = matrix[column][column..].to_vec();
        for row in 0..dimension {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in matrix[row][column..].iter_mut().zip(&pivot_tail) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    ensure!(
        rhs.iter().all(|value| value.is_finite()),
        "ridge coefficients are non-finite"
    );
    let maximum_residual = original_matrix
        .iter()
        .zip(&original_rhs)
        .map(|(row, target)| {
            (row.iter()
                .zip(&rhs)
                .map(|(value, coefficient)| value * coefficient)
                .sum::<f64>()
                - target)
                .abs()
        })
        .fold(0.0_f64, f64::max);
    let scale = original_rhs
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    ensure!(
        maximum_residual <= 1e-8 * scale,
        "ridge normal-equation residual {maximum_residual} exceeds tolerance"
    );
    Ok(rhs)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyRankingMetrics {
    pub split: ProxySplit,
    pub states: usize,
    pub rows: usize,
    pub pair_count: usize,
    pub concordant_pairs: usize,
    pub discordant_pairs: usize,
    pub tied_pairs: usize,
    pub pairwise_accuracy: f64,
    pub kendall_tau: f64,
    pub top1_matches: usize,
    pub top1_accuracy: f64,
    pub mean_regret: f64,
    pub maximum_regret: f64,
    pub mean_absolute_error: f64,
}

/// Prove that a chronological holdout cannot share states or trajectories with
/// training and that every held-out row occurs strictly after every training row.
pub fn validate_holdout_disjointness(
    training: &[ProxyTrainingRow],
    held_out: &[ProxyTrainingRow],
) -> Result<()> {
    ensure!(!training.is_empty(), "proxy training split is empty");
    ensure!(!held_out.is_empty(), "proxy held-out split is empty");
    let mut training_states = BTreeSet::new();
    let mut training_trajectories = BTreeSet::new();
    let mut training_end = training[0].date;
    for row in training {
        ensure!(
            row.split == ProxySplit::Train,
            "proxy training split contains a {:?} row",
            row.split
        );
        row.validate(None)?;
        training_states.insert(row.state_id.as_str());
        training_trajectories.insert(row.trajectory_id.as_str());
        training_end = training_end.max(row.date);
    }
    let mut held_out_start = held_out[0].date;
    for row in held_out {
        ensure!(
            row.split != ProxySplit::Train,
            "proxy held-out split contains a training row"
        );
        row.validate(None)?;
        ensure!(
            !training_states.contains(row.state_id.as_str()),
            "proxy state {} occurs in training and held-out data",
            row.state_id
        );
        ensure!(
            !training_trajectories.contains(row.trajectory_id.as_str()),
            "proxy trajectory {} occurs in training and held-out data",
            row.trajectory_id
        );
        held_out_start = held_out_start.min(row.date);
    }
    ensure!(
        training_end < held_out_start,
        "proxy chronological holdout starts {held_out_start} before training ends {training_end}"
    );
    Ok(())
}

/// Evaluate ranking and regret within each state, preserving state grouping so a
/// row from one trajectory cannot be compared as if it were an independent state.
pub fn evaluate_proxy_ranking(
    model: &LearnedProxyModel,
    rows: &[ProxyTrainingRow],
    split: ProxySplit,
) -> Result<ProxyRankingMetrics> {
    model.validate()?;
    ensure!(
        !rows.is_empty(),
        "proxy evaluation requires at least one row"
    );
    let mut groups = BTreeMap::<String, Vec<&ProxyTrainingRow>>::new();
    let mut state_metadata = BTreeMap::<String, (&str, NaiveDate, usize)>::new();
    for row in rows {
        ensure!(
            row.split == split,
            "evaluation row {} has wrong split",
            row.state_id
        );
        row.validate(Some(model.feature_names.len()))?;
        let metadata = (row.trajectory_id.as_str(), row.date, row.step_index);
        if let Some(previous) = state_metadata.insert(row.state_id.clone(), metadata) {
            ensure!(
                previous == metadata,
                "state id {} spans conflicting trajectory/date/step metadata",
                row.state_id
            );
        }
        groups.entry(row.state_id.clone()).or_default().push(row);
    }
    let mut pair_count = 0usize;
    let mut concordant_pairs = 0usize;
    let mut discordant_pairs = 0usize;
    let mut tied_pairs = 0usize;
    let mut top1_matches = 0usize;
    let mut total_regret = 0.0;
    let mut maximum_regret: f64 = 0.0;
    let mut absolute_error = 0.0;
    for mut group in groups.into_values() {
        group.sort_by(|left, right| left.guess.cmp(&right.guess));
        let predictions = group
            .iter()
            .map(|row| {
                let prediction = model
                    .predict(&row.features, row.baseline_proxy_cost)
                    .with_context(|| {
                        format!(
                            "predicting proxy state {} guess {}",
                            row.state_id, row.guess
                        )
                    })?;
                Ok((row, prediction))
            })
            .collect::<Result<Vec<_>>>()?;
        let best_exact = group
            .iter()
            .map(|row| row.exact_continuation_cost)
            .fold(f64::INFINITY, f64::min);
        let selected = predictions
            .iter()
            .min_by(|left, right| {
                left.1
                    .score
                    .total_cmp(&right.1.score)
                    .then_with(|| left.0.guess.cmp(&right.0.guess))
            })
            .expect("non-empty group");
        let regret = (selected.0.exact_continuation_cost - best_exact).max(0.0);
        total_regret += regret;
        maximum_regret = maximum_regret.max(regret);
        if regret <= 1e-12 {
            top1_matches += 1;
        }
        for (row, prediction) in &predictions {
            absolute_error += (prediction.score - row.exact_continuation_cost).abs();
        }
        for left in 0..predictions.len() {
            for right in (left + 1)..predictions.len() {
                let exact_order = predictions[left]
                    .0
                    .exact_continuation_cost
                    .total_cmp(&predictions[right].0.exact_continuation_cost);
                if exact_order == std::cmp::Ordering::Equal {
                    continue;
                }
                pair_count += 1;
                let predicted_order = predictions[left]
                    .1
                    .score
                    .total_cmp(&predictions[right].1.score);
                match predicted_order {
                    std::cmp::Ordering::Equal => tied_pairs += 1,
                    order if order == exact_order => concordant_pairs += 1,
                    _ => discordant_pairs += 1,
                }
            }
        }
    }
    let states = rows
        .iter()
        .map(|row| row.state_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let pairwise_accuracy = if pair_count == 0 {
        0.0
    } else {
        (concordant_pairs as f64 + 0.5 * tied_pairs as f64) / pair_count as f64
    };
    let kendall_tau = if pair_count == 0 {
        0.0
    } else {
        (concordant_pairs as f64 - discordant_pairs as f64) / pair_count as f64
    };
    Ok(ProxyRankingMetrics {
        split,
        states,
        rows: rows.len(),
        pair_count,
        concordant_pairs,
        discordant_pairs,
        tied_pairs,
        pairwise_accuracy,
        kendall_tau,
        top1_matches,
        top1_accuracy: top1_matches as f64 / states as f64,
        mean_regret: total_regret / states as f64,
        maximum_regret,
        mean_absolute_error: absolute_error / rows.len() as f64,
    })
}

pub fn feature_schema_digest(feature_names: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"maybe-wordle-learned-proxy-features-v1");
    for name in feature_names {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
    }
    let digest = hasher.finalize();
    crate::identity::hex(&digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_feature_names_preserve_the_existing_order_and_digest() {
        assert_eq!(learned_proxy_feature_names().len(), 12);
        assert_eq!(
            feature_schema_digest(&learned_proxy_feature_names()),
            "c21028b9a78484dd3ff9d4612d6ab74fa86b1c32cdf082bbd121f91033447425"
        );
    }

    fn provenance(feature_names: &[String]) -> ProxyModelProvenance {
        ProxyModelProvenance {
            dataset_identity: "dataset".to_string(),
            replay_identity: "replay".to_string(),
            split_policy: "chronological".to_string(),
            feature_schema_digest: feature_schema_digest(feature_names),
        }
    }

    fn row(state: &str, guess: &str, x: f64, baseline: f64, exact: f64) -> ProxyTrainingRow {
        ProxyTrainingRow {
            state_id: state.to_string(),
            trajectory_id: format!("trajectory-{state}"),
            date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
            step_index: 0,
            guess: guess.to_string(),
            features: vec![x],
            baseline_proxy_cost: baseline,
            exact_continuation_cost: exact,
            split: ProxySplit::Train,
        }
    }

    #[test]
    fn residual_ridge_is_deterministic_and_standardized() {
        let rows = vec![
            row("s1", "a", 0.0, 2.0, 2.5),
            row("s1", "b", 1.0, 2.0, 1.5),
            row("s2", "a", 0.0, 3.0, 3.5),
            row("s2", "b", 1.0, 3.0, 2.5),
        ];
        let names = vec!["entropy".to_string()];
        let first = fit_ridge_residual(&rows, &names, RidgeConfig::default(), provenance(&names))
            .expect("fit");
        let second = fit_ridge_residual(
            &rows.iter().rev().cloned().collect::<Vec<_>>(),
            &names,
            RidgeConfig::default(),
            provenance(&names),
        )
        .expect("fit");
        assert_eq!(first, second);
        assert!(first.coefficients[0].is_finite());
    }

    #[test]
    fn unstandardized_ridge_centers_features_for_an_unpenalized_intercept() {
        let rows = vec![
            row("s1", "a", 1.0, 2.0, 5.0),
            row("s2", "a", 2.0, 2.0, 7.0),
            row("s3", "a", 3.0, 2.0, 9.0),
        ];
        let names = vec!["x".to_string()];
        let model = fit_ridge_residual(
            &rows,
            &names,
            RidgeConfig {
                lambda: 1.0,
                standardize: false,
                fit_intercept: true,
                ..RidgeConfig::default()
            },
            provenance(&names),
        )
        .expect("fit");
        // Centered sums are Sxx=2 and Sxy=4; only the slope is penalized.
        assert!((model.coefficients[0] - 4.0 / 3.0).abs() < 1e-12);
        assert!((model.intercept - 7.0 / 3.0).abs() < 1e-12);
        let prediction = model.predict(&[4.0], 2.0).expect("predict");
        assert!((prediction.score - 29.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn ridge_modes_and_json_round_trips_match_scalar_normal_equations() {
        let rows = vec![
            row("s1", "a", 1.0, 2.0, 5.0),
            row("s2", "a", 2.0, 2.0, 7.0),
            row("s3", "a", 3.0, 2.0, 9.0),
        ];
        let names = vec!["x".to_string()];
        // Raw sums: Sxx=14, Sxy=34. Centered sums: Sxx=2, Sxy=4,
        // variance=2/3. Standardization retains its centered-feature contract
        // even without an intercept; lambda penalizes the scaled coefficient.
        for (lambda, standardize, fit_intercept, slope, intercept) in [
            (1.0, false, false, 34.0 / 15.0, 0.0),
            (1.0, false, true, 4.0 / 3.0, 7.0 / 3.0),
            (1.0, true, false, 3.0 / 2.0, -3.0),
            (1.0, true, true, 3.0 / 2.0, 2.0),
            (2.0, false, false, 17.0 / 8.0, 0.0),
            (2.0, false, true, 1.0, 3.0),
            (2.0, true, false, 6.0 / 5.0, -12.0 / 5.0),
            (2.0, true, true, 6.0 / 5.0, 13.0 / 5.0),
        ] {
            let model = fit_ridge_residual(
                &rows,
                &names,
                RidgeConfig {
                    lambda,
                    standardize,
                    fit_intercept,
                    ..RidgeConfig::default()
                },
                provenance(&names),
            )
            .expect("fit");
            let model_json = model.to_json().expect("serialize model");
            let reloaded = LearnedProxyModel::from_json(&model_json).expect("reload model");
            let artifact = ProxyModelArtifact {
                format_version: LEARNED_PROXY_FORMAT_VERSION,
                model: model.clone(),
                evaluation: None,
            };
            let artifact_json = artifact.to_json().expect("serialize artifact");
            let reloaded_artifact =
                ProxyModelArtifact::from_json(&artifact_json).expect("reload artifact");
            for fitted in [&model, &reloaded, &reloaded_artifact.model] {
                assert!((fitted.coefficients[0] / fitted.scaling.scales[0] - slope).abs() < 1e-12);
                for x in [0.0, 2.0, 4.0] {
                    let prediction = fitted.predict(&[x], 10.0).expect("predict");
                    assert!(
                        (prediction.score - (10.0 + intercept + slope * x)).abs() < 1e-12,
                        "lambda={lambda}, standardize={standardize}, fit_intercept={fit_intercept}, x={x}"
                    );
                }
            }
        }
    }

    #[test]
    fn ridge_modes_match_two_column_normal_equations() {
        let rows = [
            (1.0, 4.0, 9.0),
            (2.0, 2.0, 8.0),
            (3.0, 8.0, 19.0),
            (4.0, 6.0, 18.0),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (x, z, residual))| {
            let mut sample = row(&format!("s{index}"), "a", x, 2.0, 2.0 + residual);
            sample.features.push(z);
            sample
        })
        .collect::<Vec<_>>();
        let names = vec!["x".to_string(), "z".to_string()];
        // Raw Gram/rhs: [[30,56],[56,120]], [154,312]. Centered:
        // [[5,6],[6,20]], [19,42]. Variances are 5/4 and 5; invert the
        // corresponding 2x2 matrix after adding the diagonal penalty.
        for (standardize, fit_intercept, slopes, intercept) in [
            (false, false, [1162.0 / 615.0, 1048.0 / 615.0], 0.0),
            (false, true, [49.0 / 30.0, 23.0 / 15.0], 7.0 / 4.0),
            (true, false, [892.0 / 481.0, 594.0 / 481.0], -5200.0 / 481.0),
            (true, true, [892.0 / 481.0, 594.0 / 481.0], 2587.0 / 962.0),
        ] {
            let model = fit_ridge_residual(
                &rows,
                &names,
                RidgeConfig {
                    lambda: 1.0,
                    standardize,
                    fit_intercept,
                    ..RidgeConfig::default()
                },
                provenance(&names),
            )
            .expect("fit");
            for (index, slope) in slopes.iter().enumerate() {
                assert!(
                    (model.coefficients[index] / model.scaling.scales[index] - slope).abs() < 1e-12
                );
            }
            for features in [[0.0, 0.0], [5.0, 12.0]] {
                let prediction = model.predict(&features, 100.0).expect("predict");
                let expected =
                    100.0 + intercept + slopes[0] * features[0] + slopes[1] * features[1];
                assert!((prediction.score - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ridge_modes_handle_constant_features() {
        let rows = vec![
            row("s1", "a", 2.0, 2.0, 5.0),
            row("s2", "a", 2.0, 2.0, 7.0),
            row("s3", "a", 2.0, 2.0, 9.0),
        ];
        let names = vec!["constant".to_string()];
        for (standardize, fit_intercept, slope, intercept) in [
            (false, false, 30.0 / 13.0, 0.0),
            (false, true, 0.0, 5.0),
            (true, false, 0.0, 0.0),
            (true, true, 0.0, 5.0),
        ] {
            let model = fit_ridge_residual(
                &rows,
                &names,
                RidgeConfig {
                    lambda: 1.0,
                    standardize,
                    fit_intercept,
                    ..RidgeConfig::default()
                },
                provenance(&names),
            )
            .expect("constant feature fit");
            assert!((model.coefficients[0] - slope).abs() < 1e-12);
            assert!((model.intercept - intercept).abs() < 1e-12);
            let prediction = model.predict(&[2.0], 2.0).expect("predict");
            assert!((prediction.score - (2.0 + intercept + 2.0 * slope)).abs() < 1e-12);
        }
    }

    #[test]
    fn ridge_rejects_zero_regularization_in_every_mode() {
        let names = vec!["x".to_string()];
        for standardize in [false, true] {
            for fit_intercept in [false, true] {
                let error = fit_ridge_residual(
                    &[row("s1", "a", 1.0, 2.0, 5.0), row("s2", "a", 2.0, 2.0, 7.0)],
                    &names,
                    RidgeConfig {
                        lambda: 0.0,
                        standardize,
                        fit_intercept,
                        ..RidgeConfig::default()
                    },
                    provenance(&names),
                )
                .expect_err("zero regularization is outside the ridge contract");
                assert!(error.to_string().contains("ridge lambda must be positive"));
            }
        }
    }

    #[test]
    fn fitter_rejects_holdout_leakage() {
        let mut holdout = row("s", "a", 0.0, 2.0, 2.0);
        holdout.split = ProxySplit::Validation;
        let error = fit_ridge_residual(
            &[holdout],
            &["x".to_string()],
            RidgeConfig::default(),
            provenance(&["x".to_string()]),
        )
        .expect_err("holdout must not train");
        assert!(error.to_string().contains("fit only on train"));
    }

    #[test]
    fn inference_rejects_invalid_features_and_baselines() {
        let model = fit_ridge_residual(
            &[row("s", "a", 0.0, 2.0, 2.0), row("s", "b", 1.0, 2.0, 2.0)],
            &["x".to_string()],
            RidgeConfig::default(),
            provenance(&["x".to_string()]),
        )
        .expect("fit");
        for features in [vec![], vec![0.0, 1.0], vec![f64::NAN], vec![f64::INFINITY]] {
            assert!(model.predict(&features, 2.0).is_err());
        }
        for baseline in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(model.predict(&[0.0], baseline).is_err());
        }
        assert!(
            model
                .predict(&[f64::MAX], 2.0)
                .expect_err("feature scaling overflow must fail")
                .to_string()
                .contains("transformed proxy feature vector")
        );
    }

    #[test]
    fn ranking_rejects_invalid_model_instead_of_scoring_the_baseline() {
        let rows = vec![row("s", "a", 0.0, 2.0, 2.0), row("s", "b", 1.0, 2.0, 3.0)];
        let names = vec!["x".to_string()];
        let model = fit_ridge_residual(&rows, &names, RidgeConfig::default(), provenance(&names))
            .expect("fit");
        let mut wrong_dimension = model.clone();
        wrong_dimension.coefficients.clear();
        let mut wrong_schema = model.clone();
        wrong_schema.feature_names[0] = "unrelated-feature".to_string();
        let mut nonfinite_coefficient = model;
        nonfinite_coefficient.coefficients[0] = f64::INFINITY;
        for invalid in [wrong_dimension, wrong_schema, nonfinite_coefficient] {
            assert!(invalid.predict(&[1.0], 2.0).is_err());
            assert!(evaluate_proxy_ranking(&invalid, &rows, ProxySplit::Train).is_err());
        }
    }

    #[test]
    fn ranking_rejects_invalid_prediction_instead_of_scoring_the_baseline() {
        let rows = vec![row("s", "a", 2.0, 2.0, 2.0), row("s", "b", 3.0, 2.0, 3.0)];
        let names = vec!["x".to_string()];
        let mut model = fit_ridge_residual(
            &rows,
            &names,
            RidgeConfig {
                standardize: false,
                fit_intercept: false,
                ..RidgeConfig::default()
            },
            provenance(&names),
        )
        .expect("fit");
        for coefficient in [f64::MAX, -10.0] {
            model.coefficients[0] = coefficient;
            model
                .validate()
                .expect("finite coefficient is a valid artifact");
            assert!(evaluate_proxy_ranking(&model, &rows, ProxySplit::Train).is_err());
        }
        model.coefficients[0] = f64::MAX / 2.0;
        assert!(model.predict(&[1.0], f64::MAX).is_err());
    }

    #[test]
    fn ranking_rejects_invalid_feature_rows() {
        let rows = vec![row("s", "a", 0.0, 2.0, 2.0), row("s", "b", 1.0, 2.0, 3.0)];
        let names = vec!["x".to_string()];
        let model = fit_ridge_residual(&rows, &names, RidgeConfig::default(), provenance(&names))
            .expect("fit");
        for features in [vec![], vec![0.0, 1.0], vec![f64::NAN], vec![f64::INFINITY]] {
            let mut invalid = rows.clone();
            invalid[0].features = features;
            assert!(evaluate_proxy_ranking(&model, &invalid, ProxySplit::Train).is_err());
        }
    }

    #[test]
    fn model_and_artifact_json_reject_malformed_parameters() {
        let names = vec!["x".to_string()];
        let model = fit_ridge_residual(
            &[row("s", "a", 0.0, 2.0, 2.0), row("s", "b", 1.0, 2.0, 3.0)],
            &names,
            RidgeConfig::default(),
            provenance(&names),
        )
        .expect("fit");
        let valid = serde_json::to_value(&model).expect("model value");
        for (field, value) in [
            ("coefficients", serde_json::json!([])),
            ("feature_names", serde_json::json!(["unrelated-feature"])),
            ("intercept", serde_json::Value::Null),
            (
                "scaling",
                serde_json::json!({"means": [0.0], "scales": [0.0]}),
            ),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            assert!(LearnedProxyModel::from_json(&invalid.to_string()).is_err());
            let artifact = serde_json::json!({
                "format_version": LEARNED_PROXY_FORMAT_VERSION,
                "model": invalid,
                "evaluation": null,
            });
            assert!(ProxyModelArtifact::from_json(&artifact.to_string()).is_err());
        }
    }

    #[test]
    fn ranking_metrics_measure_regret_and_pairs() {
        let rows = vec![row("s", "a", 0.0, 2.0, 2.0), row("s", "b", 1.0, 2.0, 3.0)];
        let model = fit_ridge_residual(
            &rows,
            &["x".to_string()],
            RidgeConfig::default(),
            provenance(&["x".to_string()]),
        )
        .expect("fit");
        let metrics = evaluate_proxy_ranking(&model, &rows, ProxySplit::Train).expect("metrics");
        assert_eq!(metrics.states, 1);
        assert_eq!(metrics.pair_count, 1);
        assert_eq!(metrics.top1_matches, 1);
        assert!(metrics.mean_regret.abs() <= 1e-12);
    }

    #[test]
    fn schema_digest_is_stable() {
        assert_eq!(
            feature_schema_digest(&["a".to_string(), "b".to_string()]),
            feature_schema_digest(&["a".to_string(), "b".to_string()])
        );
    }
}
