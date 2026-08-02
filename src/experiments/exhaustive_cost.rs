//! Deterministic exhaustive continuation-cost data and Bellman utilities.
//!
//! This module deliberately does not depend on the private [`crate::solver::Solver`]
//! implementation.  A caller supplies a materialized state/action graph (usually
//! produced by a solver adapter), and this module performs the same weighted Bellman
//! recurrence for every supplied action.  Keeping the graph boundary explicit makes
//! generated rows replayable and prevents a training run from silently using a
//! different solver policy.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::scoring::ALL_GREEN_PATTERN;

pub const EXHAUSTIVE_COST_FORMAT_VERSION: u32 = 1;
pub const REPLAY_IDENTITY_FORMAT_VERSION: u32 = 1;

/// A state partition used by the exhaustive graph.  Survivor ids and weights are
/// retained in the row so that a row can be independently replayed without loading
/// an opaque in-memory solver state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExactState {
    pub state_id: String,
    #[serde(default)]
    pub trajectory_id: String,
    #[serde(default)]
    pub date: Option<NaiveDate>,
    #[serde(default)]
    pub step_index: usize,
    pub survivor_ids: Vec<u32>,
    pub survivor_weights: Vec<f64>,
}

impl ExactState {
    pub fn new(
        state_id: impl Into<String>,
        survivor_ids: Vec<u32>,
        survivor_weights: Vec<f64>,
    ) -> Self {
        Self {
            state_id: state_id.into(),
            trajectory_id: String::new(),
            date: None,
            step_index: 0,
            survivor_ids,
            survivor_weights,
        }
    }

    pub fn with_trajectory(mut self, trajectory_id: impl Into<String>, date: NaiveDate) -> Self {
        self.trajectory_id = trajectory_id.into();
        self.date = Some(date);
        self
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.state_id.trim().is_empty(),
            "state id must not be empty"
        );
        ensure!(
            !self.survivor_ids.is_empty(),
            "state {} must have at least one survivor",
            self.state_id
        );
        ensure!(
            self.survivor_ids.len() == self.survivor_weights.len(),
            "state {} survivor ids and weights have different lengths",
            self.state_id
        );
        ensure!(
            self.survivor_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "state {} survivor ids must be strictly sorted and unique",
            self.state_id
        );
        ensure!(
            self.survivor_ids
                .iter()
                .all(|id| u16::try_from(*id).is_ok()),
            "state {} survivor id exceeds the solver memo-key range",
            self.state_id
        );
        ensure!(
            self.survivor_weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0),
            "state {} survivor weights must be finite and non-negative",
            self.state_id
        );
        ensure!(
            self.survivor_weights.iter().any(|weight| *weight > 0.0),
            "state {} must have positive survivor mass",
            self.state_id
        );
        Ok(())
    }

    /// A stable textual encoding used for replay and cache keys.  Floating-point
    /// values are represented by their IEEE-754 bits, avoiding locale/format drift.
    pub fn canonical_key(&self) -> String {
        let ids = self
            .survivor_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let weights = self
            .survivor_weights
            .iter()
            .map(|weight| format!("{:016x}", weight.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        format!("{}|{}|{}", self.state_id, ids, weights)
    }
}

/// One feedback branch of an action.  `solved` branches terminate with zero
/// continuation cost; all other positive-mass branches must identify a child state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BellmanOutcome {
    pub pattern: u32,
    pub probability: f64,
    #[serde(default)]
    pub child_state_id: Option<String>,
    #[serde(default)]
    pub solved: bool,
}

impl BellmanOutcome {
    pub fn solved(pattern: u32, probability: f64) -> Self {
        Self {
            pattern,
            probability,
            child_state_id: None,
            solved: true,
        }
    }

    pub fn child(pattern: u32, probability: f64, child_state_id: impl Into<String>) -> Self {
        Self {
            pattern,
            probability,
            child_state_id: Some(child_state_id.into()),
            solved: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BellmanAction {
    pub guess: String,
    pub outcomes: Vec<BellmanOutcome>,
}

impl BellmanAction {
    pub fn validate(&self, state_id: &str) -> Result<()> {
        ensure!(
            !self.guess.trim().is_empty(),
            "state {} contains an empty guess",
            state_id
        );
        ensure!(
            !self.outcomes.is_empty(),
            "action {} has no outcomes",
            self.guess
        );
        let mut patterns = BTreeSet::new();
        let mut probability_sum = 0.0;
        for outcome in &self.outcomes {
            ensure!(
                outcome.pattern <= u32::from(ALL_GREEN_PATTERN),
                "action {} has out-of-range feedback pattern {}",
                self.guess,
                outcome.pattern
            );
            ensure!(
                patterns.insert(outcome.pattern),
                "action {} repeats feedback pattern {}",
                self.guess,
                outcome.pattern
            );
            ensure!(
                outcome.probability.is_finite() && (0.0..=1.0).contains(&outcome.probability),
                "action {} has invalid probability {}",
                self.guess,
                outcome.probability
            );
            probability_sum += outcome.probability;
            if outcome.solved {
                ensure!(
                    outcome.pattern == u32::from(ALL_GREEN_PATTERN),
                    "only the all-green feedback pattern may be solved"
                );
                ensure!(
                    outcome.child_state_id.is_none(),
                    "solved branch {} of action {} must not have a child",
                    outcome.pattern,
                    self.guess
                );
            } else if outcome.probability > 0.0 {
                ensure!(
                    outcome.pattern != u32::from(ALL_GREEN_PATTERN),
                    "all-green feedback must be marked solved"
                );
                ensure!(
                    outcome
                        .child_state_id
                        .as_deref()
                        .is_some_and(|id| !id.trim().is_empty()),
                    "positive-mass branch {} of action {} must identify a child",
                    outcome.pattern,
                    self.guess
                );
            }
        }
        ensure!(
            (probability_sum - 1.0).abs() <= 1e-9,
            "action {} probabilities sum to {:.12}, expected one",
            self.guess,
            probability_sum
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BellmanStateNode {
    pub state: ExactState,
    pub actions: Vec<BellmanAction>,
}

impl BellmanStateNode {
    pub fn validate(&self) -> Result<()> {
        self.state.validate()?;
        ensure!(
            !self.actions.is_empty(),
            "state {} must have at least one action",
            self.state.state_id
        );
        let mut guesses = BTreeSet::new();
        for action in &self.actions {
            ensure!(
                guesses.insert(action.guess.as_str()),
                "state {} repeats guess {}",
                self.state.state_id,
                action.guess
            );
            action.validate(&self.state.state_id)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BellmanActionCost {
    pub state_id: String,
    pub guess: String,
    /// Cost includes the current guess and expected future guesses.
    pub exact_continuation_cost: f64,
    pub optimal: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BellmanSolution {
    pub state_costs: BTreeMap<String, f64>,
    pub action_costs: Vec<BellmanActionCost>,
    pub optimal_guesses: BTreeMap<String, String>,
}

/// Solve all supplied actions with the exact weighted Bellman recurrence.
///
/// For a state `s` and action `a`,
/// `C(s,a) = 1 + sum_p P(p | s,a) * C(child(s,a,p))`; solved branches have
/// continuation cost zero.  A branch that leaves a state unchanged is treated as
/// an invalid (infinite) action and is excluded from the finite training rows.
pub fn exhaustive_bellman(nodes: &[BellmanStateNode]) -> Result<BellmanSolution> {
    ensure!(!nodes.is_empty(), "Bellman graph must not be empty");
    let mut graph = BTreeMap::new();
    for node in nodes {
        node.validate()?;
        ensure!(
            graph.insert(node.state.state_id.clone(), node).is_none(),
            "duplicate Bellman state id {}",
            node.state.state_id
        );
    }
    validate_graph_partitions(&graph)?;
    for node in nodes {
        for action in &node.actions {
            for outcome in &action.outcomes {
                if outcome.probability > 0.0 && !outcome.solved {
                    let child = outcome.child_state_id.as_deref().unwrap_or_default();
                    ensure!(
                        graph.contains_key(child),
                        "state {} action {} references missing child {}",
                        node.state.state_id,
                        action.guess,
                        child
                    );
                }
            }
        }
    }

    let mut memo = BTreeMap::new();
    let mut visiting = BTreeSet::new();
    let state_ids = graph.keys().cloned().collect::<Vec<_>>();
    for state_id in &state_ids {
        let _ = state_cost(state_id, &graph, &mut memo, &mut visiting)?;
    }

    let mut action_costs = Vec::new();
    let mut optimal_guesses = BTreeMap::new();
    for state_id in state_ids {
        let node = graph[&state_id];
        let mut finite = Vec::new();
        for action in &node.actions {
            let cost = action_cost(&state_id, action, &graph, &mut memo, &mut visiting)?;
            if cost.is_finite() {
                finite.push((action.guess.clone(), cost));
            }
        }
        ensure!(
            !finite.is_empty(),
            "state {} has no finite Bellman action",
            state_id
        );
        finite.sort_by(|left, right| left.0.cmp(&right.0));
        let best = finite
            .iter()
            .map(|(_, cost)| *cost)
            .fold(f64::INFINITY, f64::min);
        let optimal = finite
            .iter()
            .filter(|(_, cost)| (*cost - best).abs() <= 1e-12)
            .map(|(guess, _)| guess.clone())
            .min()
            .expect("finite Bellman action");
        optimal_guesses.insert(state_id.clone(), optimal);
        for (guess, cost) in finite {
            action_costs.push(BellmanActionCost {
                state_id: state_id.clone(),
                guess,
                exact_continuation_cost: cost,
                optimal: (cost - best).abs() <= 1e-12,
            });
        }
    }

    Ok(BellmanSolution {
        state_costs: memo,
        action_costs,
        optimal_guesses,
    })
}

fn validate_graph_partitions(graph: &BTreeMap<String, &BellmanStateNode>) -> Result<()> {
    for node in graph.values() {
        let parent_mass = node.state.survivor_weights.iter().sum::<f64>();
        let parent_weights = node
            .state
            .survivor_ids
            .iter()
            .copied()
            .zip(node.state.survivor_weights.iter().copied())
            .collect::<BTreeMap<_, _>>();
        for action in &node.actions {
            let mut assigned = BTreeSet::new();
            let mut child_mass = 0.0;
            let mut solved_probability = 0.0;
            for outcome in &action.outcomes {
                if outcome.probability == 0.0 {
                    continue;
                }
                if outcome.solved {
                    solved_probability += outcome.probability;
                    continue;
                }
                let child_id = outcome.child_state_id.as_deref().expect("validated child");
                let child = graph
                    .get(child_id)
                    .with_context(|| format!("missing Bellman child state {child_id}"))?;
                if child.state.survivor_ids == node.state.survivor_ids {
                    ensure!(
                        child_id == node.state.state_id,
                        "state {} action {} aliases its non-progressing subset as child {}",
                        node.state.state_id,
                        action.guess,
                        child_id
                    );
                } else {
                    ensure!(
                        child.state.survivor_ids.len() < node.state.survivor_ids.len(),
                        "state {} action {} child {} is not a strict survivor subset",
                        node.state.state_id,
                        action.guess,
                        child_id
                    );
                }
                let mut branch_mass = 0.0;
                for (id, weight) in child
                    .state
                    .survivor_ids
                    .iter()
                    .copied()
                    .zip(child.state.survivor_weights.iter().copied())
                {
                    let parent_weight = parent_weights.get(&id).with_context(|| {
                        format!(
                            "child {} contains survivor {} absent from parent {}",
                            child_id, id, node.state.state_id
                        )
                    })?;
                    ensure!(
                        parent_weight.to_bits() == weight.to_bits(),
                        "child {} changes survivor {} weight",
                        child_id,
                        id
                    );
                    ensure!(
                        assigned.insert(id),
                        "state {} action {} assigns survivor {} to multiple feedback branches",
                        node.state.state_id,
                        action.guess,
                        id
                    );
                    branch_mass += weight;
                }
                ensure!(
                    (outcome.probability - branch_mass / parent_mass).abs() <= 1e-9,
                    "state {} action {} branch {} probability disagrees with child mass",
                    node.state.state_id,
                    action.guess,
                    outcome.pattern
                );
                child_mass += branch_mass;
            }
            ensure!(
                (solved_probability - (parent_mass - child_mass) / parent_mass).abs() <= 1e-9,
                "state {} action {} solved probability disagrees with unassigned parent mass",
                node.state.state_id,
                action.guess
            );
        }
    }
    Ok(())
}

fn state_cost(
    state_id: &str,
    graph: &BTreeMap<String, &BellmanStateNode>,
    memo: &mut BTreeMap<String, f64>,
    visiting: &mut BTreeSet<String>,
) -> Result<f64> {
    if let Some(cost) = memo.get(state_id) {
        return Ok(*cost);
    }
    ensure!(
        visiting.insert(state_id.to_string()),
        "Bellman graph contains a non-progressing cycle at state {}",
        state_id
    );
    let node = graph
        .get(state_id)
        .with_context(|| format!("missing Bellman state {state_id}"))?;
    let mut best = f64::INFINITY;
    for action in &node.actions {
        let cost = action_cost(state_id, action, graph, memo, visiting)?;
        if cost < best {
            best = cost;
        }
    }
    visiting.remove(state_id);
    ensure!(
        best.is_finite(),
        "state {} has no finite Bellman action",
        state_id
    );
    memo.insert(state_id.to_string(), best);
    Ok(best)
}

fn action_cost(
    state_id: &str,
    action: &BellmanAction,
    graph: &BTreeMap<String, &BellmanStateNode>,
    memo: &mut BTreeMap<String, f64>,
    visiting: &mut BTreeSet<String>,
) -> Result<f64> {
    let mut cost = 1.0;
    let mut outcomes = action.outcomes.iter().collect::<Vec<_>>();
    outcomes.sort_by_key(|outcome| outcome.pattern);
    for outcome in outcomes {
        if outcome.probability == 0.0 || outcome.solved {
            continue;
        }
        let child = outcome.child_state_id.as_deref().unwrap_or_default();
        if child == state_id || visiting.contains(child) {
            return Ok(f64::INFINITY);
        }
        let child_cost = state_cost(child, graph, memo, visiting)?;
        cost += outcome.probability * child_cost;
        if !cost.is_finite() {
            return Ok(f64::INFINITY);
        }
    }
    Ok(cost)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChronologicalSplitMetadata {
    pub train_end: NaiveDate,
    pub validation_start: NaiveDate,
    pub validation_end: NaiveDate,
    pub test_start: NaiveDate,
    pub test_end: NaiveDate,
}

impl ChronologicalSplitMetadata {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.train_end < self.validation_start
                && self.validation_start <= self.validation_end
                && self.validation_end < self.test_start
                && self.test_start <= self.test_end,
            "chronological split windows must be ordered and non-empty"
        );
        Ok(())
    }

    pub fn classify(&self, date: NaiveDate) -> Result<DatasetSplit> {
        self.validate()?;
        if date <= self.train_end {
            Ok(DatasetSplit::Train)
        } else if date >= self.validation_start && date <= self.validation_end {
            Ok(DatasetSplit::Validation)
        } else if date >= self.test_start && date <= self.test_end {
            Ok(DatasetSplit::Test)
        } else {
            bail!("date {date} lies outside chronological split windows")
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupedStateTrajectorySplitMetadata {
    pub train_trajectory_ids: Vec<String>,
    pub validation_trajectory_ids: Vec<String>,
    pub test_trajectory_ids: Vec<String>,
}

impl GroupedStateTrajectorySplitMetadata {
    pub fn validate(&self) -> Result<()> {
        let mut seen = BTreeMap::new();
        for (split, ids) in [
            (DatasetSplit::Train, &self.train_trajectory_ids),
            (DatasetSplit::Validation, &self.validation_trajectory_ids),
            (DatasetSplit::Test, &self.test_trajectory_ids),
        ] {
            ensure!(
                !ids.is_empty(),
                "grouped split {:?} must not be empty",
                split
            );
            for id in ids {
                ensure!(!id.trim().is_empty(), "trajectory id must not be empty");
                ensure!(
                    seen.insert(id.clone(), split).is_none(),
                    "trajectory {} occurs in multiple grouped splits",
                    id
                );
            }
        }
        Ok(())
    }

    pub fn classify(&self, trajectory_id: &str) -> Result<DatasetSplit> {
        self.validate()?;
        if self
            .train_trajectory_ids
            .iter()
            .any(|id| id == trajectory_id)
        {
            Ok(DatasetSplit::Train)
        } else if self
            .validation_trajectory_ids
            .iter()
            .any(|id| id == trajectory_id)
        {
            Ok(DatasetSplit::Validation)
        } else if self
            .test_trajectory_ids
            .iter()
            .any(|id| id == trajectory_id)
        {
            Ok(DatasetSplit::Test)
        } else {
            bail!("trajectory {} is not assigned to a split", trajectory_id)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitStrategy {
    Chronological,
    GroupedStateTrajectory,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetSplitMetadata {
    pub strategy: SplitStrategy,
    #[serde(default)]
    pub chronological: Option<ChronologicalSplitMetadata>,
    #[serde(default)]
    pub grouped: Option<GroupedStateTrajectorySplitMetadata>,
}

impl DatasetSplitMetadata {
    pub fn chronological(windows: ChronologicalSplitMetadata) -> Result<Self> {
        windows.validate()?;
        Ok(Self {
            strategy: SplitStrategy::Chronological,
            chronological: Some(windows),
            grouped: None,
        })
    }

    pub fn grouped(groups: GroupedStateTrajectorySplitMetadata) -> Result<Self> {
        groups.validate()?;
        Ok(Self {
            strategy: SplitStrategy::GroupedStateTrajectory,
            chronological: None,
            grouped: Some(groups),
        })
    }

    pub fn validate(&self) -> Result<()> {
        match self.strategy {
            SplitStrategy::Chronological => {
                ensure!(
                    self.chronological.is_some() && self.grouped.is_none(),
                    "chronological split requires only chronological metadata"
                );
                self.chronological
                    .as_ref()
                    .expect("checked above")
                    .validate()
            }
            SplitStrategy::GroupedStateTrajectory => {
                ensure!(
                    self.grouped.is_some() && self.chronological.is_none(),
                    "grouped split requires only grouped metadata"
                );
                self.grouped.as_ref().expect("checked above").validate()
            }
        }
    }

    pub fn classify(&self, state: &ExactState) -> Result<DatasetSplit> {
        self.validate()?;
        match self.strategy {
            SplitStrategy::Chronological => {
                self.chronological.as_ref().expect("validated").classify(
                    state
                        .date
                        .ok_or_else(|| anyhow::anyhow!("state {} has no date", state.state_id))?,
                )
            }
            SplitStrategy::GroupedStateTrajectory => self
                .grouped
                .as_ref()
                .expect("validated")
                .classify(&state.trajectory_id),
        }
    }
}

/// A single action row.  The split assignment is stored redundantly so consumers
/// can train/evaluate without recomputing date/group routing; validation proves the
/// value agrees with [`DatasetSplitMetadata`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExhaustiveCostRow {
    pub state: ExactState,
    pub guess: String,
    pub exact_continuation_cost: f64,
    #[serde(default)]
    pub feature_values: Vec<f64>,
    #[serde(default)]
    pub baseline_proxy_cost: Option<f64>,
    pub split: DatasetSplit,
}

impl ExhaustiveCostRow {
    pub(crate) fn canonicalize_numeric_values(&mut self) {
        for weight in &mut self.state.survivor_weights {
            *weight = canonical_evidence_float(*weight);
        }
        self.exact_continuation_cost = canonical_evidence_float(self.exact_continuation_cost);
        for value in &mut self.feature_values {
            *value = canonical_evidence_float(*value);
        }
        self.baseline_proxy_cost = self.baseline_proxy_cost.map(canonical_evidence_float);
    }

    pub fn validate(&self, splits: &DatasetSplitMetadata) -> Result<()> {
        self.state.validate()?;
        ensure!(
            !self.state.trajectory_id.trim().is_empty(),
            "row trajectory id must not be empty"
        );
        ensure!(
            self.state.date.is_some(),
            "row state {} has no date",
            self.state.state_id
        );
        ensure!(!self.guess.trim().is_empty(), "row guess must not be empty");
        ensure!(
            self.exact_continuation_cost.is_finite() && self.exact_continuation_cost >= 1.0,
            "row {} / {} has invalid exact cost {}",
            self.state.state_id,
            self.guess,
            self.exact_continuation_cost
        );
        ensure!(
            self.feature_values.iter().all(|value| value.is_finite()),
            "row {} / {} has non-finite feature",
            self.state.state_id,
            self.guess
        );
        if let Some(cost) = self.baseline_proxy_cost {
            ensure!(
                cost.is_finite() && cost >= 0.0,
                "row baseline proxy cost must be finite and non-negative"
            );
        }
        ensure!(
            self.state
                .survivor_weights
                .iter()
                .chain(std::iter::once(&self.exact_continuation_cost))
                .chain(self.feature_values.iter())
                .chain(self.baseline_proxy_cost.iter())
                .all(|value| canonical_evidence_float(*value) == *value),
            "row {} / {} contains non-canonical evidence precision",
            self.state.state_id,
            self.guess
        );
        ensure!(
            splits.classify(&self.state)? == self.split,
            "row {} / {} split assignment disagrees with metadata",
            self.state.state_id,
            self.guess
        );
        Ok(())
    }

    pub fn key(&self) -> String {
        format!("{}\u{001f}{}", self.state.state_id, self.guess)
    }
}

fn canonical_evidence_float(value: f64) -> f64 {
    const SCALE: f64 = 1_000_000_000_000.0;
    let canonical = (value * SCALE).round() / SCALE;
    if canonical == 0.0 { 0.0 } else { canonical }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayIdentityInput {
    pub format_version: u32,
    pub algorithm_version: String,
    pub solver_identity: String,
    pub source_data_fingerprint: String,
    pub config_fingerprint: String,
    pub feedback_fingerprint: String,
    pub state_encoding_version: u32,
    pub weighting_fingerprint: String,
}

impl ReplayIdentityInput {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == REPLAY_IDENTITY_FORMAT_VERSION,
            "unsupported replay identity format {}; expected {}",
            self.format_version,
            REPLAY_IDENTITY_FORMAT_VERSION
        );
        for (label, value) in [
            ("algorithm version", self.algorithm_version.as_str()),
            ("solver identity", self.solver_identity.as_str()),
            (
                "source data fingerprint",
                self.source_data_fingerprint.as_str(),
            ),
            ("config fingerprint", self.config_fingerprint.as_str()),
            ("feedback fingerprint", self.feedback_fingerprint.as_str()),
            ("weighting fingerprint", self.weighting_fingerprint.as_str()),
        ] {
            ensure!(
                !value.trim().is_empty(),
                "replay {} must not be empty",
                label
            );
        }
        ensure!(
            self.state_encoding_version > 0,
            "state encoding version must be positive"
        );
        Ok(())
    }

    pub fn digest_hex(&self) -> Result<String> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(b"maybe-wordle-exhaustive-replay-v1");
        for value in [
            self.format_version.to_string(),
            self.algorithm_version.clone(),
            self.solver_identity.clone(),
            self.source_data_fingerprint.clone(),
            self.config_fingerprint.clone(),
            self.feedback_fingerprint.clone(),
            self.state_encoding_version.to_string(),
            self.weighting_fingerprint.clone(),
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        Ok(hex_digest(&hasher.finalize()))
    }
}

pub type ReplayIdentityInputs = ReplayIdentityInput;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetProvenance {
    pub dataset_id: String,
    pub generator_version: String,
    pub source_identity: String,
    pub source_data_fingerprint: String,
    pub config_fingerprint: String,
    #[serde(default)]
    pub executable_fingerprint: Option<String>,
    pub cutoff_start: NaiveDate,
    pub cutoff_end: NaiveDate,
    pub replay_identity: ReplayIdentityInput,
}

impl DatasetProvenance {
    pub fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("dataset id", self.dataset_id.as_str()),
            ("generator version", self.generator_version.as_str()),
            ("source identity", self.source_identity.as_str()),
            (
                "source data fingerprint",
                self.source_data_fingerprint.as_str(),
            ),
            ("config fingerprint", self.config_fingerprint.as_str()),
        ] {
            ensure!(!value.trim().is_empty(), "{} must not be empty", label);
        }
        if let Some(value) = &self.executable_fingerprint {
            ensure!(
                !value.trim().is_empty(),
                "executable fingerprint must not be empty"
            );
        }
        ensure!(
            self.cutoff_start <= self.cutoff_end,
            "provenance cutoff range is inverted"
        );
        self.replay_identity.validate()
    }

    pub fn replay_digest(&self) -> Result<String> {
        self.replay_identity.digest_hex()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub maximum_states: usize,
    pub maximum_rows: usize,
    pub maximum_seconds: u64,
    #[serde(default)]
    pub maximum_memory_bytes: Option<u64>,
    pub checkpoint_every_rows: usize,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            maximum_states: 1_000_000,
            maximum_rows: 10_000_000,
            maximum_seconds: 7_200,
            maximum_memory_bytes: None,
            checkpoint_every_rows: 10_000,
        }
    }
}

impl ResourceBudget {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.maximum_states > 0, "maximum states must be positive");
        ensure!(self.maximum_rows > 0, "maximum rows must be positive");
        ensure!(self.maximum_seconds > 0, "maximum seconds must be positive");
        ensure!(
            self.checkpoint_every_rows > 0,
            "checkpoint interval must be positive"
        );
        if let Some(bytes) = self.maximum_memory_bytes {
            ensure!(bytes > 0, "maximum memory bytes must be positive");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExhaustiveProgress {
    pub phase: String,
    pub states_evaluated: usize,
    pub rows_emitted: usize,
    pub elapsed_ms: u64,
    pub peak_memory_bytes: Option<u64>,
    #[serde(default)]
    pub last_state_id: Option<String>,
    pub complete: bool,
    #[serde(default)]
    pub stop_reason: Option<String>,
}

impl Default for ExhaustiveProgress {
    fn default() -> Self {
        Self {
            phase: "not_started".to_string(),
            states_evaluated: 0,
            rows_emitted: 0,
            elapsed_ms: 0,
            peak_memory_bytes: None,
            last_state_id: None,
            complete: false,
            stop_reason: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExhaustiveCostCheckpoint {
    pub format_version: u32,
    pub replay_identity_digest: String,
    pub budget: ResourceBudget,
    pub progress: ExhaustiveProgress,
    pub completed_state_ids: Vec<String>,
    pub rows: Vec<ExhaustiveCostRow>,
}

impl ExhaustiveCostCheckpoint {
    pub fn validate(&self, splits: &DatasetSplitMetadata) -> Result<()> {
        ensure!(
            self.format_version == EXHAUSTIVE_COST_FORMAT_VERSION,
            "unsupported checkpoint format {}; expected {}",
            self.format_version,
            EXHAUSTIVE_COST_FORMAT_VERSION
        );
        ensure!(
            !self.replay_identity_digest.trim().is_empty(),
            "checkpoint replay digest is empty"
        );
        self.budget.validate()?;
        ensure!(
            self.completed_state_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "checkpoint state ids must be strictly sorted"
        );
        ensure!(
            self.progress.rows_emitted == self.rows.len(),
            "checkpoint progress row count disagrees with rows"
        );
        ensure!(
            self.progress.states_evaluated == self.completed_state_ids.len(),
            "checkpoint progress state count disagrees with completed state ids"
        );
        ensure!(
            self.rows.len() <= self.budget.maximum_rows,
            "checkpoint exceeds row budget"
        );
        for row in &self.rows {
            row.validate(splits)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExhaustiveCostDatasetArtifact {
    pub format_version: u32,
    pub provenance: DatasetProvenance,
    pub split: DatasetSplitMetadata,
    pub budget: ResourceBudget,
    pub progress: ExhaustiveProgress,
    pub rows: Vec<ExhaustiveCostRow>,
    #[serde(default)]
    pub checkpoint: Option<ExhaustiveCostCheckpoint>,
}

impl ExhaustiveCostDatasetArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == EXHAUSTIVE_COST_FORMAT_VERSION,
            "unsupported exhaustive-cost format {}; expected {}",
            self.format_version,
            EXHAUSTIVE_COST_FORMAT_VERSION
        );
        self.provenance.validate()?;
        self.split.validate()?;
        self.budget.validate()?;
        ensure!(
            self.rows.len() <= self.budget.maximum_rows,
            "dataset exceeds row budget"
        );
        ensure!(
            self.progress.rows_emitted == self.rows.len(),
            "progress row count disagrees with dataset"
        );
        let mut keys = BTreeSet::new();
        let mut previous = None::<String>;
        for row in &self.rows {
            row.validate(&self.split)?;
            ensure!(
                keys.insert(row.key()),
                "duplicate dataset row {}",
                row.key()
            );
            let key = row.key();
            if let Some(previous) = previous {
                ensure!(
                    previous < key,
                    "dataset rows must be deterministically sorted by state/guess"
                );
            }
            previous = Some(key);
        }
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint.validate(&self.split)?;
            ensure!(
                checkpoint.replay_identity_digest == self.provenance.replay_digest()?,
                "checkpoint replay identity does not match dataset provenance"
            );
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(source: &str) -> Result<Self> {
        let artifact: Self =
            serde_json::from_str(source).context("decode exhaustive-cost artifact")?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn digest_hex(&self) -> Result<String> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.progress.elapsed_ms = 0;
        canonical.progress.peak_memory_bytes = None;
        canonical.checkpoint = None;
        let bytes = serde_json::to_vec(&canonical)?;
        let mut hasher = Sha256::new();
        hasher.update(b"maybe-wordle-exhaustive-cost-artifact-v1");
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
        Ok(hex_digest(&hasher.finalize()))
    }

    pub fn rows_for_split(&self, split: DatasetSplit) -> impl Iterator<Item = &ExhaustiveCostRow> {
        self.rows.iter().filter(move |row| row.split == split)
    }
}

/// Materialize all finite action costs from a solved Bellman graph.  This is the
/// integration hook for a solver adapter: construct graph nodes from private solver
/// state/feedback partitions, then call this function before fitting a proxy.
pub fn build_exhaustive_cost_dataset(
    nodes: &[BellmanStateNode],
    solution: &BellmanSolution,
    provenance: DatasetProvenance,
    split: DatasetSplitMetadata,
    budget: ResourceBudget,
) -> Result<ExhaustiveCostDatasetArtifact> {
    let started = Instant::now();
    budget.validate()?;
    provenance.validate()?;
    split.validate()?;
    ensure!(
        nodes.len() <= budget.maximum_states,
        "Bellman graph exceeds state budget"
    );
    let recomputed = exhaustive_bellman(nodes)?;
    ensure!(
        &recomputed == solution,
        "supplied Bellman solution does not match the graph"
    );
    let mut rows = Vec::new();
    let node_map = nodes
        .iter()
        .map(|node| (node.state.state_id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    for node in nodes {
        ensure!(
            started.elapsed().as_secs() <= budget.maximum_seconds,
            "Bellman dataset generation exceeded its wall-clock budget"
        );
        if let (Some(limit), Some(snapshot)) = (
            budget.maximum_memory_bytes,
            crate::process_memory::process_memory_snapshot(),
        ) {
            ensure!(
                snapshot.peak_working_set_bytes <= limit,
                "Bellman dataset generation exceeded its memory budget"
            );
        }
        let row_split = split.classify(&node.state)?;
        for action in &node.actions {
            let Some(cost) = recomputed
                .action_costs
                .iter()
                .find(|entry| entry.state_id == node.state.state_id && entry.guess == action.guess)
                .map(|entry| entry.exact_continuation_cost)
            else {
                continue;
            };
            let mut row = ExhaustiveCostRow {
                state: node.state.clone(),
                guess: action.guess.clone(),
                exact_continuation_cost: cost,
                feature_values: Vec::new(),
                baseline_proxy_cost: None,
                split: row_split,
            };
            row.canonicalize_numeric_values();
            rows.push(row);
        }
    }
    rows.sort_by_key(|row| row.key());
    ensure!(
        rows.len() <= budget.maximum_rows,
        "Bellman rows exceed row budget"
    );
    ensure!(
        node_map.len() == nodes.len(),
        "Bellman graph contains duplicate state ids"
    );
    let progress = ExhaustiveProgress {
        phase: "complete".to_string(),
        states_evaluated: nodes.len(),
        rows_emitted: rows.len(),
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        peak_memory_bytes: crate::process_memory::process_memory_snapshot()
            .map(|snapshot| snapshot.peak_working_set_bytes),
        last_state_id: nodes
            .iter()
            .map(|node| node.state.state_id.as_str())
            .max()
            .map(str::to_string),
        complete: true,
        stop_reason: None,
    };
    let artifact = ExhaustiveCostDatasetArtifact {
        format_version: EXHAUSTIVE_COST_FORMAT_VERSION,
        provenance,
        split,
        budget,
        progress,
        rows,
        checkpoint: None,
    };
    artifact.validate()?;
    Ok(artifact)
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(id: &str, date: &str, trajectory: &str, step: usize, ids: &[u32]) -> ExactState {
        ExactState {
            state_id: id.to_string(),
            trajectory_id: trajectory.to_string(),
            date: Some(NaiveDate::parse_from_str(date, "%Y-%m-%d").expect("date")),
            step_index: step,
            survivor_ids: ids.to_vec(),
            survivor_weights: vec![1.0; ids.len()],
        }
    }

    fn identity() -> ReplayIdentityInput {
        ReplayIdentityInput {
            format_version: REPLAY_IDENTITY_FORMAT_VERSION,
            algorithm_version: "bellman-v1".to_string(),
            solver_identity: "toy".to_string(),
            source_data_fingerprint: "data".to_string(),
            config_fingerprint: "config".to_string(),
            feedback_fingerprint: "patterns".to_string(),
            state_encoding_version: 1,
            weighting_fingerprint: "weights".to_string(),
        }
    }

    fn provenance() -> DatasetProvenance {
        DatasetProvenance {
            dataset_id: "test".to_string(),
            generator_version: "test-v1".to_string(),
            source_identity: "source".to_string(),
            source_data_fingerprint: "data".to_string(),
            config_fingerprint: "config".to_string(),
            executable_fingerprint: None,
            cutoff_start: NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
            cutoff_end: NaiveDate::from_ymd_opt(2024, 1, 3).expect("date"),
            replay_identity: identity(),
        }
    }

    #[test]
    fn weighted_bellman_solves_and_excludes_inert_actions() {
        let leaf = BellmanStateNode {
            state: state("leaf", "2024-01-01", "t", 1, &[1]),
            actions: vec![BellmanAction {
                guess: "a".to_string(),
                outcomes: vec![BellmanOutcome::solved(242, 1.0)],
            }],
        };
        let root = BellmanStateNode {
            state: state("root", "2024-01-01", "t", 0, &[1, 2]),
            actions: vec![
                BellmanAction {
                    guess: "bad".to_string(),
                    outcomes: vec![BellmanOutcome::child(0, 1.0, "root")],
                },
                BellmanAction {
                    guess: "good".to_string(),
                    outcomes: vec![
                        BellmanOutcome::solved(242, 0.5),
                        BellmanOutcome::child(1, 0.5, "leaf"),
                    ],
                },
            ],
        };
        let solution = exhaustive_bellman(&[root, leaf]).expect("solution");
        assert_eq!(solution.state_costs["leaf"], 1.0);
        assert_eq!(solution.state_costs["root"], 1.5);
        assert!(solution.action_costs.iter().all(|row| row.guess != "bad"));
    }

    #[test]
    fn grouped_split_rejects_trajectory_leakage() {
        let split = DatasetSplitMetadata::grouped(GroupedStateTrajectorySplitMetadata {
            train_trajectory_ids: vec!["a".to_string()],
            validation_trajectory_ids: vec!["b".to_string()],
            test_trajectory_ids: vec!["c".to_string()],
        })
        .expect("split");
        let mut row_state = state("s", "2024-01-01", "a", 0, &[1]);
        let row = ExhaustiveCostRow {
            state: row_state.clone(),
            guess: "a".to_string(),
            exact_continuation_cost: 1.0,
            feature_values: Vec::new(),
            baseline_proxy_cost: None,
            split: DatasetSplit::Train,
        };
        row.validate(&split).expect("train row");
        row_state.trajectory_id = "b".to_string();
        let leaked = ExhaustiveCostRow {
            state: row_state,
            ..row
        };
        assert!(leaked.validate(&split).is_err());
    }

    #[test]
    fn artifact_round_trip_preserves_identity_and_rows() {
        let split = DatasetSplitMetadata::chronological(ChronologicalSplitMetadata {
            train_end: NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
            validation_start: NaiveDate::from_ymd_opt(2024, 1, 2).expect("date"),
            validation_end: NaiveDate::from_ymd_opt(2024, 1, 2).expect("date"),
            test_start: NaiveDate::from_ymd_opt(2024, 1, 3).expect("date"),
            test_end: NaiveDate::from_ymd_opt(2024, 1, 3).expect("date"),
        })
        .expect("split");
        let row = ExhaustiveCostRow {
            state: state("s", "2024-01-01", "t", 0, &[1]),
            guess: "a".to_string(),
            exact_continuation_cost: 1.0,
            feature_values: vec![0.5],
            baseline_proxy_cost: Some(1.0),
            split: DatasetSplit::Train,
        };
        let artifact = ExhaustiveCostDatasetArtifact {
            format_version: EXHAUSTIVE_COST_FORMAT_VERSION,
            provenance: provenance(),
            split,
            budget: ResourceBudget::default(),
            progress: ExhaustiveProgress {
                phase: "complete".to_string(),
                states_evaluated: 1,
                rows_emitted: 1,
                elapsed_ms: 0,
                peak_memory_bytes: None,
                last_state_id: Some("s".to_string()),
                complete: true,
                stop_reason: None,
            },
            rows: vec![row],
            checkpoint: None,
        };
        let json = artifact.to_json().expect("json");
        let decoded = ExhaustiveCostDatasetArtifact::from_json(&json).expect("decode");
        assert_eq!(decoded, artifact);
        assert_eq!(
            decoded.digest_hex().expect("digest"),
            artifact.digest_hex().expect("digest")
        );
        let mut different_runtime = artifact.clone();
        different_runtime.progress.elapsed_ms = 99_999;
        different_runtime.progress.peak_memory_bytes = Some(123_456);
        assert_eq!(
            different_runtime
                .digest_hex()
                .expect("runtime-independent digest"),
            artifact.digest_hex().expect("digest")
        );
    }

    #[test]
    fn evidence_precision_is_stable_across_json_round_trips() {
        let mut row = ExhaustiveCostRow {
            state: state("s", "2024-01-01", "t", 0, &[1]),
            guess: "a".to_string(),
            exact_continuation_cost: 1.999_999_999_999_999_8,
            feature_values: vec![3.469_446_951_953_614e-17],
            baseline_proxy_cost: Some(1.999_999_999_999_999_8),
            split: DatasetSplit::Train,
        };
        row.canonicalize_numeric_values();
        assert_eq!(row.exact_continuation_cost, 2.0);
        assert_eq!(row.feature_values, vec![0.0]);
        assert_eq!(row.baseline_proxy_cost, Some(2.0));
        let decoded: ExhaustiveCostRow =
            serde_json::from_str(&serde_json::to_string(&row).expect("encode canonical row"))
                .expect("decode canonical row");
        assert_eq!(decoded, row);
    }
}
