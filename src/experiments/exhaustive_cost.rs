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

pub const EXHAUSTIVE_COST_FORMAT_VERSION: u32 = 2;
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
    exhaustive_bellman_with_row_limit(nodes, usize::MAX)
}

fn exhaustive_bellman_with_row_limit(
    nodes: &[BellmanStateNode],
    maximum_rows: usize,
) -> Result<BellmanSolution> {
    ensure!(!nodes.is_empty(), "Bellman graph must not be empty");
    ensure!(
        nodes.len() <= maximum_rows,
        "Bellman states require more finite rows than the row budget"
    );
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
                ensure!(
                    action_costs.len().saturating_add(finite.len()) < maximum_rows,
                    "Bellman actions exceed row budget"
                );
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
        Ok(crate::identity::hex(&hasher.finalize()))
    }
}

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
        self.replay_identity.validate()?;
        ensure!(
            self.source_data_fingerprint == self.replay_identity.source_data_fingerprint,
            "provenance source data fingerprint disagrees with replay identity"
        );
        ensure!(
            self.config_fingerprint == self.replay_identity.config_fingerprint,
            "provenance config fingerprint disagrees with replay identity"
        );
        Ok(())
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
    /// Recorded only after the teacher finishes all selected actions for a state.
    pub completed_state_row_counts: BTreeMap<String, usize>,
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
            self.replay_identity_digest.len() == 64
                && self
                    .replay_identity_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "checkpoint replay digest must be a lowercase SHA-256 hex digest"
        );
        ensure!(
            self.completed_state_ids
                .iter()
                .eq(self.completed_state_row_counts.keys()),
            "checkpoint completed ids must exactly match sorted completion-count keys"
        );
        validate_completed_rows(
            &self.rows,
            splits,
            &self.completed_state_row_counts,
            &self.budget,
            &self.progress,
        )?;
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
    pub completed_state_row_counts: BTreeMap<String, usize>,
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
        ensure!(
            self.progress.complete,
            "a final dataset must be complete; retain partial work as a checkpoint"
        );
        validate_completed_rows(
            &self.rows,
            &self.split,
            &self.completed_state_row_counts,
            &self.budget,
            &self.progress,
        )?;
        for row in &self.rows {
            let date = row.state.date.expect("validated dated row");
            ensure!(
                date >= self.provenance.cutoff_start && date <= self.provenance.cutoff_end,
                "row {} date lies outside provenance cutoffs",
                row.state.state_id
            );
        }
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint.validate(&self.split)?;
            ensure!(
                checkpoint.replay_identity_digest == self.provenance.replay_digest()?,
                "checkpoint replay identity does not match dataset provenance"
            );
            ensure!(
                checkpoint.budget == self.budget
                    && checkpoint.progress == self.progress
                    && checkpoint.rows == self.rows
                    && checkpoint.completed_state_row_counts == self.completed_state_row_counts,
                "attached checkpoint does not exactly describe the dataset"
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
        hasher.update(b"maybe-wordle-exhaustive-cost-artifact-v2");
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
        Ok(crate::identity::hex(&hasher.finalize()))
    }
}

fn validate_completed_rows(
    rows: &[ExhaustiveCostRow],
    splits: &DatasetSplitMetadata,
    completed_counts: &BTreeMap<String, usize>,
    budget: &ResourceBudget,
    progress: &ExhaustiveProgress,
) -> Result<()> {
    splits.validate()?;
    budget.validate()?;
    ensure!(rows.len() <= budget.maximum_rows, "rows exceed row budget");
    ensure!(
        completed_counts.len() <= budget.maximum_states,
        "completed states exceed state budget"
    );
    ensure!(
        completed_counts.values().all(|count| *count > 0),
        "completed states must have at least one action row"
    );
    let mut actual_counts = BTreeMap::new();
    let mut previous: Option<&ExhaustiveCostRow> = None;
    for row in rows {
        row.validate(splits)?;
        if let Some(previous) = previous {
            ensure!(
                (&previous.state.state_id, &previous.guess) < (&row.state.state_id, &row.guess),
                "rows must have unique keys and be sorted by state/guess"
            );
            if previous.state.state_id == row.state.state_id {
                ensure!(
                    previous.state == row.state && previous.split == row.split,
                    "rows disagree on state description or split for {}",
                    row.state.state_id
                );
            }
        }
        *actual_counts
            .entry(row.state.state_id.clone())
            .or_insert(0usize) += 1;
        previous = Some(row);
    }
    ensure!(
        &actual_counts == completed_counts,
        "rows do not exactly match completed-state row counts"
    );
    ensure!(
        progress.rows_emitted == rows.len() && progress.states_evaluated == completed_counts.len(),
        "progress counts disagree with completed states and rows"
    );
    ensure!(
        progress.last_state_id.as_ref() == completed_counts.keys().next_back(),
        "progress last state disagrees with completed states"
    );
    ensure!(
        !progress.phase.trim().is_empty() && progress.complete == (progress.phase == "complete"),
        "progress phase disagrees with completion status"
    );
    ensure!(
        !progress.complete || (!rows.is_empty() && progress.stop_reason.is_none()),
        "complete progress requires rows and no stop reason"
    );
    ensure!(
        progress
            .stop_reason
            .as_ref()
            .is_none_or(|reason| !reason.trim().is_empty()),
        "stop reason must not be blank"
    );
    let explicitly_stopped = !progress.complete && progress.stop_reason.is_some();
    ensure!(
        explicitly_stopped || progress.elapsed_ms <= budget.maximum_seconds.saturating_mul(1_000),
        "progress exceeds elapsed-time budget"
    );
    if let Some(limit) = budget.maximum_memory_bytes {
        ensure!(
            rows.is_empty() || progress.peak_memory_bytes.is_some(),
            "memory-budgeted progress must include measured peak memory"
        );
        ensure!(
            explicitly_stopped || progress.peak_memory_bytes.is_none_or(|peak| peak <= limit),
            "progress exceeds memory budget"
        );
    }
    Ok(())
}

/// Validate one completed state's rows against the independently reconstructed
/// state and finite teacher action identities. Pass only that state's rows;
/// checkpoint validation alone cannot establish which actions the teacher selected.
/// Validate the checkpoint first for schema, numeric and cross-state invariants.
pub fn validate_completed_state_rows(
    rows: &[ExhaustiveCostRow],
    expected_state: &ExactState,
    expected_split: DatasetSplit,
    expected_guesses: &[String],
) -> Result<()> {
    expected_state.validate()?;
    let expected = expected_guesses.iter().collect::<BTreeSet<_>>();
    ensure!(
        !expected.is_empty()
            && expected.len() == expected_guesses.len()
            && expected.iter().all(|guess| !guess.trim().is_empty()),
        "expected teacher actions must be nonempty and unique"
    );
    let mut canonical_state = expected_state.clone();
    for weight in &mut canonical_state.survivor_weights {
        *weight = canonical_evidence_float(*weight);
    }
    canonical_state.validate()?;
    let actual = rows.iter().map(|row| &row.guess).collect::<BTreeSet<_>>();
    ensure!(
        rows.len() == expected.len() && actual == expected,
        "completed state action identities disagree with teacher"
    );
    ensure!(
        rows.iter()
            .all(|row| row.state == canonical_state && row.split == expected_split),
        "completed state description or split disagrees with reconstructed state"
    );
    Ok(())
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
    #[cfg(test)]
    MATERIALIZED_ACTION_VISITS.set(0);
    let started = Instant::now();
    budget.validate()?;
    provenance.validate()?;
    split.validate()?;
    ensure!(
        nodes.len() <= budget.maximum_states,
        "Bellman graph exceeds state budget"
    );
    ensure!(
        solution.action_costs.len() <= budget.maximum_rows,
        "supplied Bellman actions exceed row budget"
    );
    let recomputed = exhaustive_bellman_with_row_limit(nodes, budget.maximum_rows)?;
    ensure!(
        &recomputed == solution,
        "supplied Bellman solution does not match the graph"
    );
    let mut rows = Vec::with_capacity(recomputed.action_costs.len());
    let mut node_map = BTreeMap::new();
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
        node_map.insert(node.state.state_id.as_str(), (&node.state, row_split));
    }
    let mut completed_state_row_counts = BTreeMap::new();
    // exhaustive_bellman has already established unique, sorted action keys.
    for action in &recomputed.action_costs {
        ensure!(
            rows.len() < budget.maximum_rows,
            "Bellman rows exceed row budget"
        );
        #[cfg(test)]
        MATERIALIZED_ACTION_VISITS.set(MATERIALIZED_ACTION_VISITS.get() + 1);
        let (state, row_split) = node_map[action.state_id.as_str()];
        let mut row = ExhaustiveCostRow {
            state: state.clone(),
            guess: action.guess.clone(),
            exact_continuation_cost: action.exact_continuation_cost,
            feature_values: Vec::new(),
            baseline_proxy_cost: None,
            split: row_split,
        };
        row.canonicalize_numeric_values();
        rows.push(row);
        *completed_state_row_counts
            .entry(action.state_id.clone())
            .or_insert(0usize) += 1;
    }
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
        completed_state_row_counts,
        rows,
        checkpoint: None,
    };
    artifact.validate()?;
    Ok(artifact)
}

#[cfg(test)]
thread_local! {
    static MATERIALIZED_ACTION_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

    fn dataset() -> ExhaustiveCostDatasetArtifact {
        let split = DatasetSplitMetadata::chronological(ChronologicalSplitMetadata {
            train_end: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            validation_start: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            validation_end: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            test_start: NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
            test_end: NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
        })
        .unwrap();
        let rows = [("s", "a"), ("s", "b"), ("t", "a")]
            .into_iter()
            .map(|(id, guess)| ExhaustiveCostRow {
                state: state(id, "2024-01-01", "trajectory", 0, &[1]),
                guess: guess.to_string(),
                exact_continuation_cost: 1.0,
                feature_values: vec![0.5],
                baseline_proxy_cost: Some(1.0),
                split: DatasetSplit::Train,
            })
            .collect::<Vec<_>>();
        ExhaustiveCostDatasetArtifact {
            format_version: EXHAUSTIVE_COST_FORMAT_VERSION,
            provenance: provenance(),
            split,
            budget: ResourceBudget::default(),
            progress: ExhaustiveProgress {
                phase: "complete".into(),
                states_evaluated: 2,
                rows_emitted: 3,
                last_state_id: Some("t".into()),
                complete: true,
                ..ExhaustiveProgress::default()
            },
            completed_state_row_counts: BTreeMap::from([("s".into(), 2), ("t".into(), 1)]),
            rows,
            checkpoint: None,
        }
    }

    fn checkpoint(artifact: &ExhaustiveCostDatasetArtifact) -> ExhaustiveCostCheckpoint {
        ExhaustiveCostCheckpoint {
            format_version: EXHAUSTIVE_COST_FORMAT_VERSION,
            replay_identity_digest: artifact.provenance.replay_digest().unwrap(),
            budget: artifact.budget,
            progress: artifact.progress.clone(),
            completed_state_row_counts: artifact.completed_state_row_counts.clone(),
            completed_state_ids: vec!["s".into(), "t".into()],
            rows: artifact.rows.clone(),
        }
    }

    #[test]
    fn provenance_rejects_disagreement_with_replay_identity() {
        let mut artifact = dataset();
        artifact.validate().unwrap();
        artifact.provenance.source_data_fingerprint = "other-data".into();
        assert!(artifact.validate().is_err());
        let mut artifact = dataset();
        artifact.provenance.config_fingerprint = "other-config".into();
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn rows_for_one_state_must_describe_the_identical_state() {
        let mutations: [fn(&mut ExhaustiveCostRow); 4] = [
            |row| row.state.survivor_weights[0] = 2.0,
            |row| row.state.trajectory_id = "different".into(),
            |row| row.state.step_index += 1,
            |row| {
                row.state.date = Some(NaiveDate::from_ymd_opt(2024, 1, 2).unwrap());
                row.split = DatasetSplit::Validation;
            },
        ];
        for mutate in mutations {
            let mut artifact = dataset();
            mutate(&mut artifact.rows[1]);
            assert!(artifact.validate().is_err());
        }
    }

    #[test]
    fn checkpoint_rows_must_exactly_cover_completed_states() {
        let artifact = dataset();
        let mut checkpoint = checkpoint(&artifact);
        checkpoint.validate(&artifact.split).unwrap();
        checkpoint.rows.pop();
        checkpoint.progress.rows_emitted = checkpoint.rows.len();
        assert!(checkpoint.validate(&artifact.split).is_err());
    }

    #[test]
    fn checkpoint_rejects_duplicate_rows() {
        let artifact = dataset();
        let mut checkpoint = checkpoint(&artifact);
        checkpoint.rows.insert(1, checkpoint.rows[0].clone());
        checkpoint.progress.rows_emitted = checkpoint.rows.len();
        assert!(checkpoint.validate(&artifact.split).is_err());
    }

    #[test]
    fn dataset_rejects_false_progress_and_attached_checkpoint() {
        let mut artifact = dataset();
        artifact.progress.states_evaluated = 99;
        assert!(artifact.validate().is_err());
        let mut artifact = dataset();
        artifact.budget.maximum_states = 1;
        assert!(artifact.validate().is_err());
        let mut artifact = dataset();
        let mut attached = checkpoint(&artifact);
        attached.rows[0].exact_continuation_cost = 2.0;
        artifact.checkpoint = Some(attached);
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn row_budget_is_checked_before_bellman_recomputation() {
        let mut nodes = vec![BellmanStateNode {
            state: state("s", "2024-01-01", "t", 0, &[1]),
            actions: ["a", "b"]
                .map(|guess| BellmanAction {
                    guess: guess.into(),
                    outcomes: vec![BellmanOutcome::solved(242, 1.0)],
                })
                .to_vec(),
        }];
        let solution = exhaustive_bellman(&nodes).unwrap();
        nodes[0].state.survivor_weights[0] = f64::NAN;
        let artifact = dataset();
        let error = build_exhaustive_cost_dataset(
            &nodes,
            &solution,
            artifact.provenance,
            artifact.split,
            ResourceBudget {
                maximum_rows: 1,
                ..ResourceBudget::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("row budget"), "{error}");
        assert_eq!(MATERIALIZED_ACTION_VISITS.get(), 0);
    }

    #[test]
    fn checkpoint_rejects_partial_state_deletion_even_with_updated_total() {
        let artifact = dataset();
        let mut checkpoint = checkpoint(&artifact);
        checkpoint.rows.remove(0);
        checkpoint.progress.rows_emitted = checkpoint.rows.len();
        assert!(checkpoint.validate(&artifact.split).is_err());
        let mut artifact = dataset();
        artifact.rows.remove(0);
        artifact.progress.rows_emitted = artifact.rows.len();
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn checkpoint_ids_counts_and_rows_cannot_name_different_states() {
        let artifact = dataset();
        let original = checkpoint(&artifact);
        for ids in [
            vec!["s".into()],
            vec!["s".into(), "unknown".into()],
            vec!["t".into(), "s".into()],
            vec!["s".into(), "s".into()],
        ] {
            let mut checkpoint = original.clone();
            checkpoint.completed_state_ids = ids;
            assert!(checkpoint.validate(&artifact.split).is_err());
        }
        let mut checkpoint = original.clone();
        checkpoint.completed_state_row_counts.insert("s".into(), 0);
        assert!(checkpoint.validate(&artifact.split).is_err());
        let mut checkpoint = original;
        checkpoint.rows.last_mut().unwrap().state.state_id = "unknown".into();
        assert!(checkpoint.validate(&artifact.split).is_err());
    }

    #[test]
    fn progress_and_provenance_bounds_are_checked_at_load() {
        let mutations: [fn(&mut ExhaustiveCostDatasetArtifact); 7] = [
            |artifact| artifact.progress.last_state_id = Some("unknown".into()),
            |artifact| artifact.progress.stop_reason = Some("stopped".into()),
            |artifact| artifact.progress.complete = false,
            |artifact| artifact.progress.elapsed_ms = artifact.budget.maximum_seconds * 1_000 + 1,
            |artifact| artifact.budget.maximum_memory_bytes = Some(1),
            |artifact| {
                artifact.budget.maximum_memory_bytes = Some(1);
                artifact.progress.peak_memory_bytes = Some(2);
            },
            |artifact| {
                artifact.provenance.cutoff_start = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()
            },
        ];
        for mutate in mutations {
            let mut artifact = dataset();
            mutate(&mut artifact);
            let serialized = serde_json::to_string(&artifact).unwrap();
            assert!(ExhaustiveCostDatasetArtifact::from_json(&serialized).is_err());
        }
        let artifact = dataset();
        let mut checkpoint = checkpoint(&artifact);
        checkpoint.rows.clear();
        checkpoint.completed_state_ids.clear();
        checkpoint.completed_state_row_counts.clear();
        checkpoint.progress = ExhaustiveProgress::default();
        checkpoint.validate(&artifact.split).unwrap();
    }

    #[test]
    fn schema_two_requires_authoritative_completion_counts() {
        let artifact = dataset();
        let mut value = serde_json::to_value(&artifact).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("completed_state_row_counts");
        assert!(ExhaustiveCostDatasetArtifact::from_json(&value.to_string()).is_err());
        let mut artifact = artifact;
        artifact.format_version = 1;
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn stopped_incomplete_checkpoint_retains_actual_overrun_without_becoming_complete() {
        let artifact = dataset();
        let mut stopped = checkpoint(&artifact);
        stopped.progress.phase = "interrupted".into();
        stopped.progress.complete = false;
        stopped.progress.elapsed_ms = stopped.budget.maximum_seconds * 1_000 + 1;
        stopped.budget.maximum_memory_bytes = Some(1);
        stopped.progress.peak_memory_bytes = Some(2);
        stopped.progress.stop_reason =
            Some("offline evaluation exceeded its resource budget".into());
        stopped.validate(&artifact.split).unwrap();
        stopped.progress.stop_reason = None;
        assert!(stopped.validate(&artifact.split).is_err());
        stopped.progress.complete = true;
        stopped.progress.phase = "complete".into();
        stopped.progress.stop_reason = Some("exhausted".into());
        assert!(stopped.validate(&artifact.split).is_err());
    }

    #[test]
    fn matching_attached_checkpoint_round_trips_without_changing_digest() {
        let mut artifact = dataset();
        let digest = artifact.digest_hex().unwrap();
        artifact.checkpoint = Some(checkpoint(&artifact));
        let decoded =
            ExhaustiveCostDatasetArtifact::from_json(&artifact.to_json().unwrap()).unwrap();
        assert_eq!(artifact, decoded);
        assert_eq!(decoded.digest_hex().unwrap(), digest);
    }

    #[test]
    fn reconstructed_state_and_teacher_action_set_are_required_for_resume() {
        let artifact = dataset();
        let rows = &artifact.rows[..2];
        let state = &rows[0].state;
        let actions = vec!["a".into(), "b".into()];
        validate_completed_state_rows(rows, state, DatasetSplit::Train, &actions).unwrap();
        for actions in [
            vec!["a".into()],
            vec!["a".into(), "other".into()],
            vec!["a".into(), "a".into()],
        ] {
            assert!(
                validate_completed_state_rows(rows, state, DatasetSplit::Train, &actions).is_err()
            );
        }
        let mut changed = state.clone();
        changed.survivor_weights[0] = 2.0;
        assert!(
            validate_completed_state_rows(rows, &changed, DatasetSplit::Train, &actions).is_err()
        );
        assert!(
            validate_completed_state_rows(rows, state, DatasetSplit::Validation, &actions).is_err()
        );
        assert!(
            validate_completed_state_rows(&artifact.rows, state, DatasetSplit::Train, &actions)
                .is_err()
        );
    }

    fn terminal_graph(states: usize, actions: usize) -> Vec<BellmanStateNode> {
        (0..states)
            .map(|index| BellmanStateNode {
                state: state(
                    &format!("s{index:04}"),
                    "2024-01-01",
                    "trajectory",
                    0,
                    &[index as u32],
                ),
                actions: (0..actions)
                    .map(|guess| BellmanAction {
                        guess: format!("g{guess:04}"),
                        outcomes: vec![BellmanOutcome::solved(242, 1.0)],
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn keyed_materialization_visits_each_action_once_and_is_order_independent() {
        for states in [16, 32, 64, 128] {
            let mut nodes = terminal_graph(states, 8);
            let solution = exhaustive_bellman(&nodes).unwrap();
            let fixture = dataset();
            let artifact = build_exhaustive_cost_dataset(
                &nodes,
                &solution,
                fixture.provenance.clone(),
                fixture.split.clone(),
                fixture.budget,
            )
            .unwrap();
            assert_eq!(MATERIALIZED_ACTION_VISITS.get(), states * 8);
            assert_eq!(artifact.rows.len(), states * 8);
            assert!(
                artifact
                    .rows
                    .iter()
                    .all(|row| row.exact_continuation_cost == 1.0)
            );
            let expected = nodes
                .iter()
                .flat_map(|node| {
                    node.actions
                        .iter()
                        .map(|action| (node.state.state_id.clone(), action.guess.clone()))
                })
                .collect::<Vec<_>>();
            assert_eq!(
                artifact
                    .rows
                    .iter()
                    .map(|row| (row.state.state_id.clone(), row.guess.clone()))
                    .collect::<Vec<_>>(),
                expected
            );
            nodes.reverse();
            for node in &mut nodes {
                node.actions.reverse();
            }
            let reordered = build_exhaustive_cost_dataset(
                &nodes,
                &solution,
                fixture.provenance,
                fixture.split,
                fixture.budget,
            )
            .unwrap();
            assert_eq!(MATERIALIZED_ACTION_VISITS.get(), states * 8);
            assert_eq!(artifact.rows, reordered.rows);
            assert_eq!(
                artifact.digest_hex().unwrap(),
                reordered.digest_hex().unwrap()
            );
        }
    }

    #[test]
    fn bounded_recomputation_cannot_trust_an_underreported_solution() {
        let nodes = terminal_graph(2, 8);
        let mut solution = exhaustive_bellman(&nodes).unwrap();
        solution.action_costs.clear();
        let fixture = dataset();
        let error = build_exhaustive_cost_dataset(
            &nodes,
            &solution,
            fixture.provenance,
            fixture.split,
            ResourceBudget {
                maximum_rows: 4,
                ..ResourceBudget::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("row budget"), "{error}");
        assert_eq!(MATERIALIZED_ACTION_VISITS.get(), 0);
    }

    #[test]
    fn graph_verification_still_rejects_duplicate_and_changed_actions() {
        let nodes = terminal_graph(2, 2);
        let solution = exhaustive_bellman(&nodes).unwrap();
        let mut duplicate = nodes.clone();
        duplicate[0].actions.push(nodes[0].actions[0].clone());
        assert!(exhaustive_bellman(&duplicate).is_err());
        let mut duplicate = nodes.clone();
        duplicate.push(nodes[0].clone());
        assert!(exhaustive_bellman(&duplicate).is_err());
        let mut wrong = solution;
        wrong.action_costs[0].exact_continuation_cost = 2.0;
        let fixture = dataset();
        assert!(
            build_exhaustive_cost_dataset(
                &nodes,
                &wrong,
                fixture.provenance,
                fixture.split,
                fixture.budget
            )
            .is_err()
        );
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
            completed_state_row_counts: BTreeMap::from([("s".into(), 1)]),
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
