use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};

use super::*;

const FINITE_POLL_STRIDE: usize = 64;
const FINITE_PROPOSAL_SAMPLE_CAP: usize = 4096;
const FINITE_VALUE_RESOLUTION: f64 = 64.0 * f64::EPSILON;

/// Bounded finite-horizon search controls.  The two shortlist sizes affect only
/// broad-state proposal/rollout work; tractable states enumerate every legal
/// guess.
#[derive(Clone, Copy, Debug)]
pub struct FiniteSearchOptions {
    pub root_shortlist: usize,
    pub reply_shortlist: usize,
    pub exact_state_threshold: usize,
    pub budget: Duration,
    /// Deterministic cap on cooperative work units, including partition and
    /// proposal work; despite the field name this is not a recursive-node cap.
    pub node_limit: Option<usize>,
    /// Evaluate only the deterministic state-local baseline policy.
    pub baseline_only: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FiniteSearchQuality {
    /// A legal fallback selected before any bounded evaluation completed.
    Heuristic,
    /// A complete feasible rollout.  It is an upper bound on the optimum for
    /// the lexicographic failure/attempt objective, not an exact optimum.
    UpperBound,
    /// Exhaustive over the configured finite state and legal action set.
    Exact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FiniteSearchReason {
    Complete,
    Deadline,
    Cancelled,
    NodeBudget,
}

#[derive(Clone, Copy, Debug)]
pub struct FiniteSearchCandidate {
    pub guess_index: usize,
    pub failure_probability: f64,
    pub expected_attempts: f64,
    pub quality: FiniteSearchQuality,
}

#[derive(Clone, Debug)]
pub struct FiniteSearchResult {
    pub candidates: Vec<FiniteSearchCandidate>,
    pub root_candidates_considered: usize,
    pub all_legal_roots_evaluated: bool,
    pub reason: FiniteSearchReason,
    /// Recursive calls, including memo hits, not unique expanded states.
    pub nodes_visited: usize,
    pub work_units: usize,
    pub cache_hits: usize,
    /// Proposal geometry or scanning was truncated; not an exhaustive proposal set.
    pub proposal_sampled: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FiniteNodeKey {
    subset: Vec<usize>,
    dynamic_belief: Option<FiniteDynamicKey>,
    observations: Vec<(String, u8)>,
    remaining_turns: u8,
    hard_mode: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FiniteDynamicKey {
    active_weight_bits: Vec<u64>,
    fallback_surviving: Vec<usize>,
    condition_only: bool,
    fallback_active: bool,
    recovery_mode_used: Option<u8>,
}

#[derive(Clone, Copy, Debug)]
struct FiniteValue {
    failure_probability: f64,
    expected_attempts: f64,
}

#[derive(Clone, Copy, Debug)]
struct CachedFiniteValue {
    value: FiniteValue,
    quality: FiniteSearchQuality,
}

enum FiniteMoveEvaluation {
    Complete(CachedFiniteValue),
    Pruned,
}

struct FinitePartition {
    patterns: Vec<u8>,
    signature: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
struct ProposalMetric {
    guess_index: usize,
    solve_probability: f64,
    entropy: f64,
    expected_remaining: f64,
    largest_non_green_mass: f64,
    worst_non_green_bucket_size: usize,
    proxy_score: f64,
}

#[derive(Clone, Debug)]
struct FiniteNode {
    subset: Vec<usize>,
    observations: Vec<(String, u8)>,
    dynamic_belief: Option<FiniteDynamicBelief>,
}

#[derive(Clone, Debug)]
struct FiniteDynamicBelief {
    /// Active weights aligned with `FiniteNode::subset`, which is index-sorted
    /// for dynamic searches.
    weights: Vec<f64>,
    fallback_surviving: Vec<usize>,
    condition_only: bool,
    fallback_active: bool,
    recovery_mode_used: Option<RecoveryMode>,
}

fn finite_node_weight(weights: &[f64], node: &FiniteNode, answer_index: usize) -> f64 {
    let Some(belief) = &node.dynamic_belief else {
        return weights[answer_index];
    };
    let position = node
        .subset
        .binary_search(&answer_index)
        .expect("dynamic answer weights align with index-sorted support");
    belief.weights[position]
}

struct FiniteSearchControl<'a> {
    started: Instant,
    budget: Duration,
    node_limit: Option<usize>,
    cancelled: &'a dyn Fn() -> bool,
    reason: Option<FiniteSearchReason>,
    work_units: usize,
    nodes_visited: usize,
}

impl<'a> FiniteSearchControl<'a> {
    fn new(options: FiniteSearchOptions, cancelled: &'a dyn Fn() -> bool) -> Self {
        Self {
            started: Instant::now(),
            budget: options.budget,
            node_limit: options.node_limit,
            cancelled,
            reason: None,
            work_units: 0,
            nodes_visited: 0,
        }
    }

    fn poll(&mut self, force: bool) -> bool {
        self.poll_inner(force, false)
    }

    fn poll_proposal(&mut self, force: bool) -> bool {
        self.poll_inner(force, true)
    }

    fn poll_inner(&mut self, force: bool, reserve_incumbent: bool) -> bool {
        if self.reason.is_some() {
            return false;
        }
        self.work_units = self.work_units.saturating_add(1);
        let poll_external = force || self.work_units.is_multiple_of(FINITE_POLL_STRIDE);
        if poll_external && (self.cancelled)() {
            self.reason = Some(FiniteSearchReason::Cancelled);
            return false;
        }
        if reserve_incumbent
            && self
                .node_limit
                .is_some_and(|limit| self.work_units > limit / 4)
        {
            return false;
        }
        if self.node_limit.is_some_and(|limit| self.work_units > limit) {
            self.reason = Some(FiniteSearchReason::NodeBudget);
            return false;
        }
        if poll_external {
            let deadline = if reserve_incumbent {
                self.budget / 4
            } else {
                self.budget
            };
            if self.started.elapsed() >= deadline {
                if reserve_incumbent {
                    return false;
                }
                self.reason = Some(FiniteSearchReason::Deadline);
                return false;
            }
        }
        true
    }

    fn visit_node(&mut self) -> bool {
        if !self.poll(true) {
            return false;
        }
        self.nodes_visited = self.nodes_visited.saturating_add(1);
        true
    }
}

struct FinitePartitionFrame {
    masses: [f64; PATTERN_SPACE],
    touched_patterns: Vec<u8>,
    child_subsets: [Vec<usize>; PATTERN_SPACE],
}

impl FinitePartitionFrame {
    fn new() -> Self {
        Self {
            masses: [0.0; PATTERN_SPACE],
            touched_patterns: Vec::with_capacity(PATTERN_SPACE),
            child_subsets: array::from_fn(|_| Vec::new()),
        }
    }

    fn reset(&mut self) {
        for pattern in self.touched_patterns.drain(..) {
            self.masses[pattern as usize] = 0.0;
            self.child_subsets[pattern as usize].clear();
        }
    }
}

struct FinitePartitionScratch {
    frames: Vec<FinitePartitionFrame>,
}

impl FinitePartitionScratch {
    fn new() -> Self {
        Self { frames: Vec::new() }
    }

    fn frame_mut(&mut self, depth: usize) -> &mut FinitePartitionFrame {
        while self.frames.len() <= depth {
            self.frames.push(FinitePartitionFrame::new());
        }
        let frame = &mut self.frames[depth];
        frame.reset();
        frame
    }
}

struct FiniteSearchRunner<'a> {
    solver: &'a Solver,
    weights: &'a [f64],
    dynamic_basis: Option<&'a SolveState>,
    proposal_scratch: GuessMetricScratch,
    options: FiniteSearchOptions,
    hard_mode: bool,
    control: FiniteSearchControl<'a>,
    partition: FinitePartitionScratch,
    exact_memo: HashMap<FiniteNodeKey, CachedFiniteValue>,
    baseline_memo: HashMap<FiniteNodeKey, CachedFiniteValue>,
    cache_hits: usize,
    proposal_sampled: bool,
}

impl<'a> FiniteSearchRunner<'a> {
    fn new(
        solver: &'a Solver,
        weights: &'a [f64],
        options: FiniteSearchOptions,
        hard_mode: bool,
        cancelled: &'a dyn Fn() -> bool,
    ) -> Self {
        Self::with_dynamic_basis(solver, weights, None, options, hard_mode, cancelled)
    }

    fn with_dynamic_basis(
        solver: &'a Solver,
        weights: &'a [f64],
        dynamic_basis: Option<&'a SolveState>,
        options: FiniteSearchOptions,
        hard_mode: bool,
        cancelled: &'a dyn Fn() -> bool,
    ) -> Self {
        Self {
            solver,
            weights,
            dynamic_basis,
            proposal_scratch: GuessMetricScratch::new(),
            options,
            hard_mode,
            control: FiniteSearchControl::new(options, cancelled),
            partition: FinitePartitionScratch::new(),
            exact_memo: HashMap::new(),
            baseline_memo: HashMap::new(),
            cache_hits: 0,
            proposal_sampled: false,
        }
    }

    fn weight_for_index(&self, node: &FiniteNode, answer_index: usize) -> f64 {
        finite_node_weight(self.weights, node, answer_index)
    }

    fn dynamic_key(&self, node: &FiniteNode) -> Option<FiniteDynamicKey> {
        node.dynamic_belief.as_ref().map(|belief| FiniteDynamicKey {
            active_weight_bits: belief
                .weights
                .iter()
                .map(|weight| weight.to_bits())
                .collect(),
            fallback_surviving: belief.fallback_surviving.clone(),
            condition_only: belief.condition_only,
            fallback_active: belief.fallback_active,
            recovery_mode_used: belief.recovery_mode_used.map(recovery_mode_code),
        })
    }

    fn node_key(&self, node: &FiniteNode, remaining_turns: u8) -> FiniteNodeKey {
        FiniteNodeKey {
            subset: node.subset.clone(),
            dynamic_belief: self.dynamic_key(node),
            observations: if self.hard_mode {
                node.observations.clone()
            } else {
                Vec::new()
            },
            remaining_turns,
            hard_mode: self.hard_mode,
        }
    }

    fn legal_guess(&self, node: &FiniteNode, guess_index: usize) -> bool {
        !self.hard_mode
            || self
                .solver
                .hard_mode_violation(&node.observations, &self.solver.guesses[guess_index])
                .is_none()
    }

    fn quick_legal_candidate(&self, node: &FiniteNode) -> Option<usize> {
        let mut answers = node.subset.clone();
        answers.sort_by(|left, right| {
            self.weight_for_index(node, *right)
                .total_cmp(&self.weight_for_index(node, *left))
                .then_with(|| {
                    self.solver.answers[*left]
                        .word
                        .cmp(&self.solver.answers[*right].word)
                })
        });
        for answer_index in &answers {
            let Some(guess_index) = self
                .solver
                .guess_index
                .get(&self.solver.answers[*answer_index].word)
                .copied()
            else {
                continue;
            };
            if self.legal_guess(node, guess_index) {
                return Some(guess_index);
            }
        }
        (0..self.solver.guesses.len()).find(|guess_index| self.legal_guess(node, *guess_index))
    }

    fn all_legal_guesses(&mut self, node: &FiniteNode) -> Vec<usize> {
        let mut guesses = Vec::new();
        let constraints = self
            .hard_mode
            .then(|| build_hard_mode_constraints(&node.observations));
        for guess_index in 0..self.solver.guesses.len() {
            if !self.control.poll(false) {
                break;
            }
            if constraints
                .as_ref()
                .is_none_or(|rules| rules.allows(&self.solver.guesses[guess_index]))
            {
                guesses.push(guess_index);
            }
        }
        guesses
    }

    fn partition_fallback_by_pattern(
        &mut self,
        fallback_surviving: &[usize],
        guess_index: usize,
        patterns: &[u8],
    ) -> Option<[Vec<usize>; PATTERN_SPACE]> {
        let mut selected = [false; PATTERN_SPACE];
        for pattern in patterns {
            selected[*pattern as usize] = true;
        }
        let mut partitions = std::array::from_fn(|_| Vec::new());
        for answer_index in fallback_surviving {
            if !self.control.poll(false) {
                return None;
            }
            let pattern = self.solver.answer_pattern(guess_index, *answer_index) as usize;
            if selected[pattern] {
                partitions[pattern].push(*answer_index);
            }
        }
        Some(partitions)
    }

    fn child_node(
        &mut self,
        node: &FiniteNode,
        guess_index: usize,
        pattern: u8,
        subset: Vec<usize>,
        fallback_subset: Option<Vec<usize>>,
    ) -> Result<Option<FiniteNode>> {
        let mut observations = if self.hard_mode {
            node.observations.clone()
        } else {
            Vec::new()
        };
        if self.hard_mode {
            observations.push((self.solver.guesses[guess_index].clone(), pattern));
        }
        let (subset, dynamic_belief) =
            if let (Some(parent), Some(basis)) = (&node.dynamic_belief, self.dynamic_basis) {
                let Some((subset, belief)) = self.dynamic_child_belief(
                    node,
                    parent,
                    basis,
                    guess_index,
                    pattern,
                    subset,
                    fallback_subset,
                )?
                else {
                    return Ok(None);
                };
                (subset, Some(belief))
            } else {
                (subset, None)
            };
        Ok(Some(FiniteNode {
            subset,
            observations,
            dynamic_belief,
        }))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "explicit dynamic transition inputs keep the optional prepartitioned fallback distinct"
    )]
    fn dynamic_child_belief(
        &mut self,
        node: &FiniteNode,
        parent: &FiniteDynamicBelief,
        basis: &SolveState,
        guess_index: usize,
        pattern: u8,
        active_subset: Vec<usize>,
        fallback_subset: Option<Vec<usize>>,
    ) -> Result<Option<(Vec<usize>, FiniteDynamicBelief)>> {
        let mut active = Vec::with_capacity(active_subset.len());
        for answer_index in active_subset {
            if !self.control.poll(false) {
                return Ok(None);
            }
            let position = node.subset.binary_search(&answer_index).map_err(|_| {
                anyhow::anyhow!("finite-horizon dynamic active partition is inconsistent")
            })?;
            active.push((answer_index, parent.weights[position]));
        }
        let mut fallback_surviving = if let Some(fallback_subset) = fallback_subset {
            fallback_subset
        } else {
            let mut fallback_surviving = Vec::new();
            for answer_index in &parent.fallback_surviving {
                if !self.control.poll(false) {
                    return Ok(None);
                }
                if self.solver.answer_pattern(guess_index, *answer_index) == pattern {
                    fallback_surviving.push(*answer_index);
                }
            }
            fallback_surviving
        };

        let mut fallback_active = parent.fallback_active;
        if !parent.condition_only {
            if active.is_empty() && !fallback_surviving.is_empty() {
                active.extend(
                    fallback_surviving
                        .drain(..)
                        .map(|answer_index| (answer_index, 0.0)),
                );
                fallback_active = true;
            } else if self.solver.config.fallback_activation_threshold > 0
                && active.len() <= self.solver.config.fallback_activation_threshold
                && !fallback_surviving.is_empty()
            {
                active.extend(
                    fallback_surviving
                        .drain(..)
                        .map(|answer_index| (answer_index, 0.0)),
                );
                fallback_active = true;
            }
        }
        active.sort_unstable_by_key(|(answer_index, _)| *answer_index);
        fallback_surviving.sort_unstable();

        let mut recovery_mode_used = parent.recovery_mode_used;
        if !parent.condition_only {
            let mut modeled_total_weight = 0.0;
            for (answer_index, _) in &active {
                if !self.control.poll(false) {
                    return Ok(None);
                }
                modeled_total_weight += basis.modeled_weights[*answer_index];
            }
            if modeled_total_weight > 0.0 {
                for (answer_index, weight) in &mut active {
                    if !self.control.poll(false) {
                        return Ok(None);
                    }
                    *weight = if basis.modeled_weights[*answer_index] > 0.0 {
                        basis.modeled_weights[*answer_index]
                    } else if fallback_active {
                        basis.recovery_weights[*answer_index]
                    } else {
                        0.0
                    };
                }
                recovery_mode_used = None;
            } else {
                let active_count = active.len();
                recovery_mode_used = match self.solver.config.recovery.mode {
                    RecoveryMode::Strict => {
                        for (_, weight) in &mut active {
                            if !self.control.poll(false) {
                                return Ok(None);
                            }
                            *weight = 0.0;
                        }
                        None
                    }
                    mode => {
                        for (answer_index, weight) in &mut active {
                            if !self.control.poll(false) {
                                return Ok(None);
                            }
                            *weight = self
                                .solver
                                .config
                                .recovery
                                .repair_weight(basis.recovery_weights[*answer_index], active_count);
                        }
                        Some(mode)
                    }
                };
            }
        }

        let total_weight = active.iter().map(|(_, weight)| *weight).sum::<f64>();
        if active.is_empty()
            || (total_weight <= 0.0 && self.solver.config.recovery.mode == RecoveryMode::Strict)
        {
            bail!("finite-horizon dynamic transition has no positive answer mass");
        }
        let subset = active
            .iter()
            .map(|(answer_index, _)| *answer_index)
            .collect();
        Ok(Some((
            subset,
            FiniteDynamicBelief {
                weights: active.into_iter().map(|(_, weight)| weight).collect(),
                fallback_surviving,
                condition_only: parent.condition_only,
                fallback_active,
                recovery_mode_used,
            },
        )))
    }

    fn total_weight(&self, node: &FiniteNode) -> Result<f64> {
        self.subset_weight(node, &node.subset)
    }

    fn subset_weight(&self, node: &FiniteNode, subset: &[usize]) -> Result<f64> {
        let mut total = 0.0;
        for answer_index in subset {
            let weight = self.weight_for_index(node, *answer_index);
            if !weight.is_finite() || weight < 0.0 {
                bail!("finite-horizon weights must be finite and non-negative");
            }
            total += weight;
        }
        if !total.is_finite() || total <= 0.0 {
            bail!("finite-horizon search requires positive answer mass");
        }
        Ok(total)
    }

    fn partition_patterns(
        &mut self,
        node: &FiniteNode,
        guess_index: usize,
        depth: usize,
        capture_signature: bool,
    ) -> Result<Option<FinitePartition>> {
        let weights = self.weights;
        let frame = self.partition.frame_mut(depth);
        let mut signature = capture_signature.then(|| Vec::with_capacity(node.subset.len()));
        for answer_index in &node.subset {
            if !self.control.poll(false) {
                return Ok(None);
            }
            let weight = finite_node_weight(weights, node, *answer_index);
            let pattern = self.solver.answer_pattern(guess_index, *answer_index) as usize;
            if let Some(signature) = &mut signature {
                signature.push(pattern as u8);
            }
            if frame.child_subsets[pattern].is_empty() {
                frame.touched_patterns.push(pattern as u8);
            }
            frame.masses[pattern] += weight;
            frame.child_subsets[pattern].push(*answer_index);
        }
        Ok(Some(FinitePartition {
            patterns: frame.touched_patterns.clone(),
            signature,
        }))
    }

    fn cooperative_proposal_metric(
        &mut self,
        node: &FiniteNode,
        subset: &[usize],
        total_weight: f64,
        guess_index: usize,
        reserve_incumbent: bool,
    ) -> Option<ProposalMetric> {
        let weights = self.weights;
        let scratch = &mut self.proposal_scratch;
        scratch.reset();
        for answer_index in subset {
            if !(if reserve_incumbent {
                self.control.poll_proposal(false)
            } else {
                self.control.poll(false)
            }) {
                if reserve_incumbent && self.control.reason.is_none() {
                    self.proposal_sampled = true;
                }
                return None;
            }
            let pattern = self.solver.answer_pattern(guess_index, *answer_index) as usize;
            if scratch.counts[pattern] == 0 {
                scratch.touched_patterns.push(pattern as u8);
            }
            let weight = finite_node_weight(weights, node, *answer_index);
            scratch.counts[pattern] += 1;
            scratch.masses[pattern] += weight;
            scratch.largest_weights[pattern] = scratch.largest_weights[pattern].max(weight);
        }
        let metric = self.solver.score_partition_metrics(
            guess_index,
            scratch,
            total_weight,
            scratch.masses[ALL_GREEN_PATTERN as usize] / total_weight,
        );
        Some(ProposalMetric {
            guess_index,
            solve_probability: metric.solve_probability,
            entropy: metric.entropy,
            expected_remaining: metric.expected_remaining,
            largest_non_green_mass: metric.largest_non_green_bucket_mass,
            worst_non_green_bucket_size: metric.worst_non_green_bucket_size,
            proxy_score: metric.large_state_score,
        })
    }

    fn proposal_subset(&mut self, node: &FiniteNode) -> Vec<usize> {
        if node.subset.len() <= FINITE_PROPOSAL_SAMPLE_CAP {
            return node.subset.clone();
        }
        self.proposal_sampled = true;
        // ponytail: truncate only proposal geometry; evaluate policies on full support.
        // A stratified proposal sample can replace this if measured regret warrants it.
        let mut sampled = node.subset.clone();
        sampled.sort_by(|left, right| {
            self.weight_for_index(node, *right)
                .total_cmp(&self.weight_for_index(node, *left))
                .then_with(|| {
                    self.solver.answers[*left]
                        .word
                        .cmp(&self.solver.answers[*right].word)
                })
        });
        sampled.truncate(FINITE_PROPOSAL_SAMPLE_CAP);
        sampled.sort_unstable_by(|left, right| {
            self.solver.answers[*left]
                .word
                .cmp(&self.solver.answers[*right].word)
        });
        sampled
    }

    fn preorder_root_guesses(
        &mut self,
        node: &FiniteNode,
        subset: &[usize],
        total_weight: f64,
    ) -> Vec<usize> {
        // Unique-letter Bernoulli variance only schedules proposal work;
        // full partition scores and rollout values still choose the move.
        let mut letter_masses = [0.0_f64; 26];
        for answer_index in subset {
            if !self.control.poll_proposal(false) {
                if self.control.reason.is_none() {
                    self.proposal_sampled = true;
                }
                return Vec::new();
            }
            let mut seen = 0_u32;
            for byte in self.solver.answers[*answer_index].word.bytes() {
                debug_assert!(byte.is_ascii_lowercase());
                let letter = usize::from(byte - b'a');
                let bit = 1_u32 << letter;
                if seen & bit == 0 {
                    seen |= bit;
                    letter_masses[letter] += self.weight_for_index(node, *answer_index);
                }
            }
        }

        let letter_information = letter_masses.map(|mass| {
            let probability = (mass / total_weight).clamp(0.0, 1.0);
            probability * (1.0 - probability)
        });
        let mut ordered = Vec::new();
        for guess_index in 0..self.solver.guesses.len() {
            if !self.control.poll_proposal(false) {
                if self.control.reason.is_none() {
                    self.proposal_sampled = true;
                }
                return Vec::new();
            }
            if !self.legal_guess(node, guess_index) {
                continue;
            }
            let mut seen = 0_u32;
            let mut information = 0.0_f64;
            for byte in self.solver.guesses[guess_index].bytes() {
                debug_assert!(byte.is_ascii_lowercase());
                let letter = usize::from(byte - b'a');
                let bit = 1_u32 << letter;
                if seen & bit == 0 {
                    seen |= bit;
                    information += letter_information[letter];
                }
            }
            ordered.push((guess_index, information));
        }
        ordered.sort_by(
            |(left_index, left_information), (right_index, right_information)| {
                right_information.total_cmp(left_information).then_with(|| {
                    self.solver.guesses[*left_index].cmp(&self.solver.guesses[*right_index])
                })
            },
        );
        ordered
            .into_iter()
            .map(|(guess_index, _)| guess_index)
            .collect()
    }

    fn propose_root_guesses(&mut self, node: &FiniteNode, quick: usize) -> Result<Vec<usize>> {
        let proposal_subset = self.proposal_subset(node);
        // Proposal geometry is sampled only for broad states; normalize its
        // proxy scores to sampled mass just as the fixed-belief runner does.
        let total_weight = self.subset_weight(node, &proposal_subset)?;
        let mut metrics = Vec::new();
        let proposal_order = self.preorder_root_guesses(node, &proposal_subset, total_weight);
        for guess_index in proposal_order {
            if !self.control.poll_proposal(false) {
                if self.control.reason.is_none() {
                    self.proposal_sampled = true;
                }
                break;
            }
            if let Some(metric) = self.cooperative_proposal_metric(
                node,
                &proposal_subset,
                total_weight,
                guess_index,
                true,
            ) {
                metrics.push(metric);
            }
        }
        metrics.sort_by(|left, right| {
            right
                .proxy_score
                .total_cmp(&left.proxy_score)
                .then_with(|| right.solve_probability.total_cmp(&left.solve_probability))
                .then_with(|| right.entropy.total_cmp(&left.entropy))
                .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
                .then_with(|| {
                    left.largest_non_green_mass
                        .total_cmp(&right.largest_non_green_mass)
                })
                .then_with(|| {
                    left.worst_non_green_bucket_size
                        .cmp(&right.worst_non_green_bucket_size)
                })
                .then_with(|| {
                    self.solver.guesses[left.guess_index]
                        .cmp(&self.solver.guesses[right.guess_index])
                })
        });

        let mut roots = metrics
            .into_iter()
            .take(self.options.root_shortlist.max(1))
            .map(|metric| metric.guess_index)
            .collect::<Vec<_>>();
        if !roots.contains(&quick) {
            roots.push(quick);
        }
        Ok(roots)
    }

    fn baseline_guess(&mut self, node: &FiniteNode) -> Result<Option<usize>> {
        let reply_limit = self.options.reply_shortlist.max(1);
        let total_weight = self.total_weight(node)?;
        let mut candidate_indexes = Vec::new();
        // A state-local baseline must choose the same action when reached by
        // rollout or by a fresh request. Root proposal anchors violate that
        // contract. Rescore the highest-mass legal answers at every node;
        // unrestricted probe guesses remain available to root improvement.
        let mut survivor_answers = node
            .subset
            .iter()
            .filter_map(|answer_index| {
                let guess_index = self
                    .solver
                    .guess_index
                    .get(&self.solver.answers[*answer_index].word)
                    .copied()?;
                self.legal_guess(node, guess_index).then_some((
                    guess_index,
                    self.weight_for_index(node, *answer_index),
                    self.solver.answers[*answer_index].word.as_str(),
                ))
            })
            .collect::<Vec<_>>();
        survivor_answers
            .sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.2.cmp(right.2)));
        // Every surviving answer is retained as a legal/progressing fallback;
        // only the bounded top reply shortlist receives the more expensive
        // proxy rescore below.
        let fallback_answer = survivor_answers.first().map(|entry| entry.0);
        for (guess_index, _, _) in survivor_answers.iter().take(reply_limit) {
            candidate_indexes.push(*guess_index);
        }

        let mut best_progressing = None;
        for guess_index in candidate_indexes {
            if !self.control.poll(false) {
                return Ok(None);
            }
            let Some(metric) = self.cooperative_proposal_metric(
                node,
                &node.subset,
                total_weight,
                guess_index,
                false,
            ) else {
                return Ok(None);
            };
            if metric.worst_non_green_bucket_size >= node.subset.len() {
                continue;
            }
            let better = best_progressing.is_none_or(|incumbent: ProposalMetric| {
                self.compare_reply_metrics(&metric, &incumbent, node.subset.len())
                    == std::cmp::Ordering::Less
            });
            if better {
                best_progressing = Some(metric);
            }
        }
        if let Some(metric) = best_progressing {
            return Ok(Some(metric.guess_index));
        }
        if let Some(guess_index) = fallback_answer {
            return Ok(Some(guess_index));
        }
        for guess_index in 0..self.solver.guesses.len() {
            if !self.control.poll(false) {
                return Ok(None);
            }
            if self.legal_guess(node, guess_index) {
                return Ok(Some(guess_index));
            }
        }
        Ok(None)
    }

    fn max_mass_legal_guess(&mut self, node: &FiniteNode) -> Result<Option<usize>> {
        let mut best = None;
        for answer_index in &node.subset {
            if !self.control.poll(false) {
                return Ok(None);
            }
            let Some(guess_index) = self
                .solver
                .guess_index
                .get(&self.solver.answers[*answer_index].word)
                .copied()
            else {
                continue;
            };
            if !self.legal_guess(node, guess_index) {
                continue;
            }
            let candidate = (
                guess_index,
                self.weight_for_index(node, *answer_index),
                self.solver.answers[*answer_index].word.as_str(),
            );
            if best.is_none_or(|incumbent: (usize, f64, &str)| {
                candidate.1 > incumbent.1
                    || (candidate.1 == incumbent.1 && candidate.2 < incumbent.2)
            }) {
                best = Some(candidate);
            }
        }
        Ok(best.map(|candidate| candidate.0))
    }

    fn compare_reply_metrics(
        &self,
        left: &ProposalMetric,
        right: &ProposalMetric,
        subset_len: usize,
    ) -> std::cmp::Ordering {
        (right.worst_non_green_bucket_size < subset_len)
            .cmp(&(left.worst_non_green_bucket_size < subset_len))
            .then_with(|| right.proxy_score.total_cmp(&left.proxy_score))
            .then_with(|| right.entropy.total_cmp(&left.entropy))
            .then_with(|| left.expected_remaining.total_cmp(&right.expected_remaining))
            .then_with(|| {
                self.solver.guesses[left.guess_index].cmp(&self.solver.guesses[right.guess_index])
            })
    }

    fn compare_values(left: FiniteValue, right: FiniteValue) -> std::cmp::Ordering {
        Self::value_key(left.failure_probability)
            .total_cmp(&Self::value_key(right.failure_probability))
            .then_with(|| {
                Self::value_key(left.expected_attempts)
                    .total_cmp(&Self::value_key(right.expected_attempts))
            })
    }

    fn value_key(value: f64) -> f64 {
        // Quantized keys define a transitive order; pairwise epsilon ties do
        // not. Precision is part of the finite policy identity/contract.
        (value / FINITE_VALUE_RESOLUTION).round()
    }

    fn join_quality(left: FiniteSearchQuality, right: FiniteSearchQuality) -> FiniteSearchQuality {
        match (left, right) {
            (FiniteSearchQuality::Heuristic, _) | (_, FiniteSearchQuality::Heuristic) => {
                FiniteSearchQuality::Heuristic
            }
            (FiniteSearchQuality::UpperBound, _) | (_, FiniteSearchQuality::UpperBound) => {
                FiniteSearchQuality::UpperBound
            }
            _ => FiniteSearchQuality::Exact,
        }
    }

    fn downgrade_to_upper_bound(mut value: CachedFiniteValue) -> CachedFiniteValue {
        value.quality = FiniteSearchQuality::UpperBound;
        value
    }

    fn direct_one_turn(&mut self, node: &FiniteNode) -> Result<Option<CachedFiniteValue>> {
        let total_weight = self.total_weight(node)?;
        let mut best_green_mass = 0.0_f64;
        let mut legal_answer_count = 0usize;
        for answer_index in &node.subset {
            if !self.control.poll(false) {
                return Ok(None);
            }
            let Some(guess_index) = self
                .solver
                .guess_index
                .get(&self.solver.answers[*answer_index].word)
                .copied()
            else {
                continue;
            };
            if self.legal_guess(node, guess_index) {
                legal_answer_count += 1;
                best_green_mass = best_green_mass.max(self.weight_for_index(node, *answer_index));
            }
        }
        if legal_answer_count == 0 {
            // A valid hard-mode state normally retains its answer action.  If
            // it does not, no answer guess can solve the state; retain one
            // attempt for an available legal probe and leave failure at one.
            return Ok(Some(CachedFiniteValue {
                value: FiniteValue {
                    failure_probability: 1.0,
                    expected_attempts: if !(0..self.solver.guesses.len())
                        .any(|guess| self.legal_guess(node, guess))
                    {
                        0.0
                    } else {
                        1.0
                    },
                },
                quality: FiniteSearchQuality::Exact,
            }));
        }
        Ok(Some(CachedFiniteValue {
            value: FiniteValue {
                failure_probability: (1.0 - (best_green_mass / total_weight)).clamp(0.0, 1.0),
                expected_attempts: 1.0,
            },
            quality: FiniteSearchQuality::Exact,
        }))
    }

    fn evaluate_move(
        &mut self,
        node: &FiniteNode,
        guess_index: usize,
        remaining_turns: u8,
        depth: usize,
        exact_endgames: bool,
    ) -> Result<Option<CachedFiniteValue>> {
        let total_weight = self.total_weight(node)?;
        let Some(partition) = self.partition_patterns(node, guess_index, depth, false)? else {
            return Ok(None);
        };
        let Some(FiniteMoveEvaluation::Complete(result)) = self.evaluate_partitioned_move(
            node,
            guess_index,
            remaining_turns,
            depth,
            exact_endgames,
            total_weight,
            &partition.patterns,
            None,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(result))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "partition context and admissible incumbent stay explicit"
    )]
    fn evaluate_partitioned_move(
        &mut self,
        node: &FiniteNode,
        guess_index: usize,
        remaining_turns: u8,
        depth: usize,
        exact_endgames: bool,
        total_weight: f64,
        patterns: &[u8],
        incumbent: Option<FiniteValue>,
    ) -> Result<Option<FiniteMoveEvaluation>> {
        let mut failure_probability = 0.0;
        let mut expected_attempts = 1.0;
        let mut quality = FiniteSearchQuality::Exact;
        let mut exact_bounds = true;

        let mut fallback_patterns = if remaining_turns > 1 {
            if let Some(belief) = node.dynamic_belief.as_ref()
                && !belief.fallback_surviving.is_empty()
            {
                let child_patterns = patterns
                    .iter()
                    .copied()
                    .filter(|pattern| {
                        *pattern != ALL_GREEN_PATTERN
                            && self.partition.frames[depth].masses[*pattern as usize] > 0.0
                    })
                    .collect::<Vec<_>>();
                if child_patterns.is_empty() {
                    None
                } else {
                    let Some(partitions) = self.partition_fallback_by_pattern(
                        &belief.fallback_surviving,
                        guess_index,
                        &child_patterns,
                    ) else {
                        return Ok(None);
                    };
                    Some(partitions)
                }
            } else {
                None
            }
        } else {
            None
        };

        for (pattern_index, pattern) in patterns.iter().copied().enumerate() {
            if !self.control.poll(false) {
                return Ok(None);
            }
            let mass = self.partition.frames[depth].masses[pattern as usize];
            if mass <= 0.0 {
                continue;
            }
            let probability = mass / total_weight;
            let child_value = if pattern == ALL_GREEN_PATTERN {
                CachedFiniteValue {
                    value: FiniteValue {
                        failure_probability: 0.0,
                        expected_attempts: 0.0,
                    },
                    quality: FiniteSearchQuality::Exact,
                }
            } else if remaining_turns <= 1 {
                CachedFiniteValue {
                    value: FiniteValue {
                        failure_probability: 1.0,
                        expected_attempts: 0.0,
                    },
                    quality: FiniteSearchQuality::Exact,
                }
            } else {
                let child_subset = std::mem::take(
                    &mut self.partition.frames[depth].child_subsets[pattern as usize],
                );
                let fallback_subset = fallback_patterns
                    .as_mut()
                    .map(|partitions| std::mem::take(&mut partitions[pattern as usize]));
                let Some(child) =
                    self.child_node(node, guess_index, pattern, child_subset, fallback_subset)?
                else {
                    return Ok(None);
                };
                let result =
                    if exact_endgames && child.subset.len() <= self.options.exact_state_threshold {
                        self.evaluate_exact_node(&child, remaining_turns - 1, depth + 1)?
                    } else {
                        self.evaluate_baseline_node(&child, remaining_turns - 1, depth + 1)?
                    };
                self.partition.frames[depth].child_subsets[pattern as usize] = child.subset;
                let Some(result) = result else {
                    return Ok(None);
                };
                result
            };
            failure_probability += probability * child_value.value.failure_probability;
            expected_attempts += probability * child_value.value.expected_attempts;
            exact_bounds &= child_value.quality == FiniteSearchQuality::Exact;
            quality = Self::join_quality(quality, child_value.quality);

            if exact_bounds
                && let Some(incumbent) = incumbent
                && self.move_lower_bound_dominates(
                    failure_probability,
                    expected_attempts,
                    &patterns[pattern_index + 1..],
                    depth,
                    total_weight,
                    remaining_turns,
                    incumbent,
                )
            {
                return Ok(Some(FiniteMoveEvaluation::Pruned));
            }
        }

        Ok(Some(FiniteMoveEvaluation::Complete(CachedFiniteValue {
            value: FiniteValue {
                failure_probability: failure_probability.clamp(0.0, 1.0),
                expected_attempts: expected_attempts.max(0.0),
            },
            quality,
        })))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "partial value and remaining branch context stay explicit"
    )]
    fn move_lower_bound_dominates(
        &self,
        accumulated_failure: f64,
        accumulated_attempts: f64,
        remaining_patterns: &[u8],
        depth: usize,
        total_weight: f64,
        remaining_turns: u8,
        incumbent: FiniteValue,
    ) -> bool {
        let mut failure_lower_bound = accumulated_failure;
        let mut future_attempt_mass = 0.0;
        for pattern in remaining_patterns {
            let mass = self.partition.frames[depth].masses[*pattern as usize];
            if mass <= 0.0 {
                continue;
            }
            let probability = mass / total_weight;
            if remaining_turns <= 1 {
                if *pattern != ALL_GREEN_PATTERN {
                    // Keep the same left-to-right accumulation order as the
                    // complete move value, so the floating-point bound cannot
                    // exceed the value of the corresponding minimum branch.
                    failure_lower_bound += probability;
                }
            } else if *pattern != ALL_GREEN_PATTERN {
                // Any successful non-green branch needs at least one more
                // guess.  This is only used after the failure lower bound
                // reaches the incumbent, so every remaining branch must
                // succeed for the candidate to tie it.
                future_attempt_mass += probability;
            }
        }

        // The complete value clamps failure at one; preserve that same cap
        // when floating-point mass additions round a probability sum upward.
        let failure_lower_bound = failure_lower_bound.min(1.0);
        let failure_key = Self::value_key(failure_lower_bound);
        let incumbent_failure_key = Self::value_key(incumbent.failure_probability);
        if failure_key > incumbent_failure_key {
            return true;
        }
        if failure_key < incumbent_failure_key {
            return false;
        }

        // Allow all additional failure that can remain in the incumbent's
        // quantized failure bucket before bounding attempts. This keeps the
        // lexicographic pruning sound even when values differ within the
        // configured resolution.
        let failure_bucket_ceiling =
            (incumbent_failure_key + 0.5) * FINITE_VALUE_RESOLUTION + FINITE_VALUE_RESOLUTION;
        let allowed_future_failure = (failure_bucket_ceiling - failure_lower_bound)
            .max(0.0)
            .min(future_attempt_mass);
        let attempts_lower_bound =
            accumulated_attempts + future_attempt_mass - allowed_future_failure;
        Self::value_key(attempts_lower_bound) >= Self::value_key(incumbent.expected_attempts)
    }

    fn evaluate_exact_node(
        &mut self,
        node: &FiniteNode,
        remaining_turns: u8,
        depth: usize,
    ) -> Result<Option<CachedFiniteValue>> {
        if !self.control.visit_node() {
            return Ok(None);
        }
        if remaining_turns == 0 {
            return Ok(Some(CachedFiniteValue {
                value: FiniteValue {
                    failure_probability: 1.0,
                    expected_attempts: 0.0,
                },
                quality: FiniteSearchQuality::Exact,
            }));
        }
        let key = self.node_key(node, remaining_turns);
        if let Some(cached) = self.exact_memo.get(&key).copied() {
            self.cache_hits = self.cache_hits.saturating_add(1);
            return Ok(Some(cached));
        }
        // One legal answer wins immediately. Two can be solved in order only
        // when a miss cannot activate dormant fallback answers.
        if node.subset.len() <= 2
            && (node.subset.len() == 1
                || node
                    .dynamic_belief
                    .as_ref()
                    .is_none_or(|belief| belief.fallback_surviving.is_empty()))
            && node.subset.iter().all(|index| {
                self.solver
                    .guess_index
                    .get(&self.solver.answers[*index].word)
                    .is_some_and(|guess| self.legal_guess(node, *guess))
            })
        {
            let result = self.direct_one_turn(node)?.map(|mut result| {
                if remaining_turns >= 2 {
                    result.value.expected_attempts += result.value.failure_probability;
                    result.value.failure_probability = 0.0;
                }
                result
            });
            if let Some(result) = result {
                self.exact_memo.insert(key, result);
            }
            return Ok(result);
        }
        if remaining_turns == 1 {
            let result = self.direct_one_turn(node)?;
            if let Some(result) = result {
                self.exact_memo.insert(key, result);
            }
            return Ok(result);
        }
        if node.subset.len() > self.options.exact_state_threshold {
            return self.evaluate_baseline_node(node, remaining_turns, depth);
        }

        let mut legal_guesses = self.all_legal_guesses(node);
        if self.control.reason.is_some() {
            return Ok(None);
        }
        let quick = self.quick_legal_candidate(node);
        if let Some(quick) = quick {
            legal_guesses.retain(|guess| *guess != quick);
            legal_guesses.insert(0, quick);
        }
        // For h >= 2, any zero-failure policy needs at least a second
        // attempt unless its first guess solves. At most p_max solves now.
        // Attaining (0, 2-p_max) proves the best value; no other action can
        // improve it. Do not use this attempts bound at h == 1.
        let total_weight = self.total_weight(node)?;
        let p_max = node
            .subset
            .iter()
            .map(|index| self.weight_for_index(node, *index) / total_weight)
            .fold(0.0_f64, f64::max);
        let lower_bound = FiniteValue {
            failure_probability: 0.0,
            expected_attempts: 2.0 - p_max,
        };
        let mut best = None;
        let mut equivalent_partitions = HashSet::new();
        for guess_index in legal_guesses {
            if !self.control.poll(false) {
                return Ok(best.map(Self::downgrade_to_upper_bound));
            }
            if best.is_some()
                && node.dynamic_belief.as_ref().is_none_or(|belief| {
                    belief.condition_only || belief.fallback_surviving.is_empty()
                })
            {
                let first = self.solver.answer_pattern(guess_index, node.subset[0]);
                let mut non_progressing = first != ALL_GREEN_PATTERN;
                for answer in node.subset.iter().skip(1) {
                    if !non_progressing {
                        break;
                    }
                    if !self.control.poll(false) {
                        return Ok(best.map(Self::downgrade_to_upper_bound));
                    }
                    non_progressing = self.solver.answer_pattern(guess_index, *answer) == first;
                }
                // With fixed support, no information and no solve cannot improve
                // the optimum; hard-mode history can only restrict replies.
                // A dynamic probe may instead remove dormant support, so the
                // active partition alone cannot prove it is non-progressing.
                // Keep fixed-root evaluation unchanged; prune only the argmin.
                if non_progressing {
                    continue;
                }
            }
            let capture_signature = !self.hard_mode && node.dynamic_belief.is_none();
            let Some(partition) =
                self.partition_patterns(node, guess_index, depth, capture_signature)?
            else {
                return Ok(best.map(Self::downgrade_to_upper_bound));
            };
            if let Some(signature) = partition.signature
                && !equivalent_partitions.insert(signature)
            {
                continue;
            }
            let Some(evaluation) = self.evaluate_partitioned_move(
                node,
                guess_index,
                remaining_turns,
                depth,
                true,
                total_weight,
                &partition.patterns,
                best.map(|incumbent: CachedFiniteValue| incumbent.value),
            )?
            else {
                return Ok(best.map(Self::downgrade_to_upper_bound));
            };
            let FiniteMoveEvaluation::Complete(candidate) = evaluation else {
                continue;
            };
            let better = best.is_none_or(|incumbent: CachedFiniteValue| {
                Self::compare_values(candidate.value, incumbent.value) == std::cmp::Ordering::Less
            });
            if better {
                best = Some(candidate);
            }
            if candidate.quality == FiniteSearchQuality::Exact
                && candidate.value.failure_probability == 0.0
                && Self::compare_values(candidate.value, lower_bound) != std::cmp::Ordering::Greater
            {
                break;
            }
        }
        let Some(best) = best else {
            return Ok(Some(CachedFiniteValue {
                value: FiniteValue {
                    failure_probability: 1.0,
                    expected_attempts: 0.0,
                },
                quality: FiniteSearchQuality::Exact,
            }));
        };
        if self.control.reason.is_some() || best.quality != FiniteSearchQuality::Exact {
            return Ok(Some(Self::downgrade_to_upper_bound(best)));
        }
        self.exact_memo.insert(key, best);
        Ok(Some(best))
    }

    fn evaluate_baseline_node(
        &mut self,
        node: &FiniteNode,
        remaining_turns: u8,
        depth: usize,
    ) -> Result<Option<CachedFiniteValue>> {
        if !self.control.visit_node() {
            return Ok(None);
        }
        if remaining_turns == 0 {
            return Ok(Some(CachedFiniteValue {
                value: FiniteValue {
                    failure_probability: 1.0,
                    expected_attempts: 0.0,
                },
                quality: FiniteSearchQuality::Exact,
            }));
        }
        let key = self.node_key(node, remaining_turns);
        if let Some(cached) = self.baseline_memo.get(&key).copied() {
            self.cache_hits = self.cache_hits.saturating_add(1);
            return Ok(Some(cached));
        }
        if remaining_turns == 1 {
            let result = self.direct_one_turn(node)?;
            if let Some(result) = result {
                self.baseline_memo.insert(key, result);
            }
            return Ok(result);
        }
        let Some(guess_index) = self.baseline_guess(node)? else {
            return Ok(None);
        };
        let Some(mut result) =
            self.evaluate_move(node, guess_index, remaining_turns, depth, false)?
        else {
            return Ok(None);
        };
        result.quality = Self::join_quality(FiniteSearchQuality::UpperBound, result.quality);
        self.baseline_memo.insert(key, result);
        Ok(Some(result))
    }

    fn heuristic_candidate(&self, guess_index: usize) -> FiniteSearchCandidate {
        FiniteSearchCandidate {
            guess_index,
            // The value is intentionally conservative and explicitly marked
            // heuristic; it must never outrank a completed value on its own.
            failure_probability: 1.0,
            expected_attempts: f64::INFINITY,
            quality: FiniteSearchQuality::Heuristic,
        }
    }

    fn evaluate_roots(
        &mut self,
        node: &FiniteNode,
        roots: &[usize],
        remaining_turns: u8,
    ) -> Result<Vec<FiniteSearchCandidate>> {
        if remaining_turns == 1 {
            return self.evaluate_final_turn_roots(node, roots);
        }
        let mut candidates = Vec::new();
        // Compare complete cheap policies across the shortlist before allowing
        // one expensive exact continuation to consume the remaining budget.
        for exact_endgames in [false, true] {
            if !exact_endgames
                && (node.subset.len() <= self.options.exact_state_threshold || remaining_turns <= 2)
            {
                continue;
            }
            for guess_index in roots.iter().copied() {
                if !self.control.poll(true) {
                    return Ok(candidates);
                }
                if !self.legal_guess(node, guess_index) {
                    continue;
                }
                let Some(result) =
                    self.evaluate_move(node, guess_index, remaining_turns, 0, exact_endgames)?
                else {
                    return Ok(candidates);
                };
                candidates.push(FiniteSearchCandidate {
                    guess_index,
                    failure_probability: result.value.failure_probability,
                    expected_attempts: result.value.expected_attempts,
                    quality: result.quality,
                });
            }
        }
        Ok(candidates)
    }

    fn evaluate_final_turn_roots(
        &mut self,
        node: &FiniteNode,
        roots: &[usize],
    ) -> Result<Vec<FiniteSearchCandidate>> {
        let total_weight = self.total_weight(node)?;
        let mut green_mass_by_guess = vec![0.0_f64; self.solver.guesses.len()];
        for answer_index in &node.subset {
            if !self.control.poll(false) {
                return Ok(Vec::new());
            }
            let Some(guess_index) = self
                .solver
                .guess_index
                .get(&self.solver.answers[*answer_index].word)
                .copied()
            else {
                continue;
            };
            if self.legal_guess(node, guess_index) {
                green_mass_by_guess[guess_index] += self.weight_for_index(node, *answer_index);
            }
        }
        let mut candidates = Vec::with_capacity(roots.len());
        for guess_index in roots.iter().copied() {
            if !self.control.poll(false) {
                return Ok(Vec::new());
            }
            if !self.legal_guess(node, guess_index) {
                continue;
            }
            let green_probability = green_mass_by_guess[guess_index] / total_weight;
            candidates.push(FiniteSearchCandidate {
                guess_index,
                failure_probability: (1.0 - green_probability).clamp(0.0, 1.0),
                expected_attempts: 1.0,
                quality: FiniteSearchQuality::Exact,
            });
        }
        Ok(candidates)
    }
}

impl Solver {
    /// Evaluate a finite-horizon policy without target information.  Feedback
    /// transitions are generated from the authoritative answer-pattern table;
    /// the target is never passed to this method or any recursive helper.
    #[allow(
        clippy::too_many_arguments,
        reason = "explicit search state, rules and cooperative controls; no extra wrapper type"
    )]
    pub(super) fn finite_horizon_search(
        &self,
        subset: &[usize],
        weights: &[f64],
        observations: &[(String, u8)],
        remaining_turns: u8,
        hard_mode: bool,
        options: FiniteSearchOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<FiniteSearchResult> {
        if subset.is_empty() {
            bail!("finite-horizon search requires a non-empty state");
        }
        if !(1..=6).contains(&remaining_turns) {
            bail!("finite-horizon search requires between one and six remaining turns");
        }
        if weights.len() < self.answers.len() {
            bail!(
                "finite-horizon weight vector has {} values; expected at least {}",
                weights.len(),
                self.answers.len()
            );
        }
        for answer_index in subset {
            if *answer_index >= self.answers.len() {
                bail!("finite-horizon answer index {answer_index} is out of range");
            }
        }
        if hard_mode {
            for (guess, pattern) in observations {
                if guess.len() != HARD_MODE_WORD_LENGTH
                    || !guess.bytes().all(|byte| byte.is_ascii_lowercase())
                {
                    bail!("hard-mode history guess must be exactly 5 lowercase letters");
                }
                if *pattern as usize >= PATTERN_SPACE {
                    bail!("hard-mode history contains an invalid feedback pattern");
                }
            }
        }

        let mut root_subset = subset.to_vec();
        root_subset.sort_unstable_by(|left, right| {
            self.answers[*left].word.cmp(&self.answers[*right].word)
        });
        root_subset.dedup();
        let root = FiniteNode {
            subset: root_subset,
            observations: if hard_mode {
                observations.to_vec()
            } else {
                Vec::new()
            },
            dynamic_belief: None,
        };
        self.finite_horizon_search_from_root(
            root,
            weights,
            None,
            remaining_turns,
            hard_mode,
            options,
            cancelled,
        )
    }

    pub(super) fn finite_horizon_search_dynamic(
        &self,
        state: &SolveState,
        observations: &[(String, u8)],
        remaining_turns: u8,
        hard_mode: bool,
        options: FiniteSearchOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<FiniteSearchResult> {
        if state.surviving.is_empty() {
            bail!("finite-horizon search requires a non-empty state");
        }
        if !(1..=6).contains(&remaining_turns) {
            bail!("finite-horizon search requires between one and six remaining turns");
        }
        if state.weights.len() < self.answers.len()
            || state.modeled_weights.len() < self.answers.len()
            || state.recovery_weights.len() < self.answers.len()
        {
            bail!("finite-horizon dynamic state weight vectors are incomplete");
        }
        if hard_mode {
            for (guess, pattern) in observations {
                if guess.len() != HARD_MODE_WORD_LENGTH
                    || !guess.bytes().all(|byte| byte.is_ascii_lowercase())
                {
                    bail!("hard-mode history guess must be exactly 5 lowercase letters");
                }
                if *pattern as usize >= PATTERN_SPACE {
                    bail!("hard-mode history contains an invalid feedback pattern");
                }
            }
        }
        for answer_index in state.surviving.iter().chain(&state.fallback_surviving) {
            if *answer_index >= self.answers.len() {
                bail!("finite-horizon answer index {answer_index} is out of range");
            }
        }
        let mut root_subset = state.surviving.clone();
        root_subset.sort_unstable();
        if root_subset.windows(2).any(|pair| pair[0] == pair[1]) {
            bail!("finite-horizon dynamic state contains duplicate active answers");
        }
        let mut fallback_surviving = state.fallback_surviving.clone();
        fallback_surviving.sort_unstable();
        if fallback_surviving.windows(2).any(|pair| pair[0] == pair[1]) {
            bail!("finite-horizon dynamic state contains duplicate fallback answers");
        }
        let root = FiniteNode {
            dynamic_belief: Some(FiniteDynamicBelief {
                weights: root_subset
                    .iter()
                    .map(|answer_index| state.weights[*answer_index])
                    .collect(),
                fallback_surviving,
                condition_only: state.condition_only,
                fallback_active: state.fallback_active,
                recovery_mode_used: state.recovery_mode_used,
            }),
            subset: root_subset,
            observations: if hard_mode {
                observations.to_vec()
            } else {
                Vec::new()
            },
        };
        self.finite_horizon_search_from_root(
            root,
            &state.weights,
            Some(state),
            remaining_turns,
            hard_mode,
            options,
            cancelled,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "explicit recursive search configuration remains visible"
    )]
    fn finite_horizon_search_from_root(
        &self,
        root: FiniteNode,
        weights: &[f64],
        dynamic_basis: Option<&SolveState>,
        remaining_turns: u8,
        hard_mode: bool,
        options: FiniteSearchOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<FiniteSearchResult> {
        let mut runner = if let Some(dynamic_basis) = dynamic_basis {
            FiniteSearchRunner::with_dynamic_basis(
                self,
                weights,
                Some(dynamic_basis),
                options,
                hard_mode,
                cancelled,
            )
        } else {
            FiniteSearchRunner::new(self, weights, options, hard_mode, cancelled)
        };
        runner.total_weight(&root)?;
        let Some(quick) = runner.quick_legal_candidate(&root) else {
            bail!("finite-horizon search found no legal guess");
        };

        // Discovering a legal answer guess is intentionally outside the
        // bounded ranking work so a zero-budget request still returns a valid
        // action and an honest heuristic status.
        if !runner.control.poll(true) {
            return Ok(FiniteSearchResult {
                candidates: vec![runner.heuristic_candidate(quick)],
                root_candidates_considered: 1,
                all_legal_roots_evaluated: false,
                reason: runner
                    .control
                    .reason
                    .unwrap_or(FiniteSearchReason::Deadline),
                nodes_visited: runner.control.nodes_visited,
                work_units: runner.control.work_units,
                cache_hits: runner.cache_hits,
                proposal_sampled: runner.proposal_sampled,
            });
        }

        if options.baseline_only {
            let selected = if remaining_turns == 1 {
                runner.max_mass_legal_guess(&root)?
            } else {
                runner.baseline_guess(&root)?
            }
            .unwrap_or(quick);
            if runner.control.reason.is_none() {
                let evaluated = if remaining_turns == 1 {
                    runner.direct_one_turn(&root)?
                } else {
                    runner.evaluate_move(&root, selected, remaining_turns, 0, false)?
                };
                if let Some(mut result) = evaluated {
                    if remaining_turns > 1 {
                        result.quality = FiniteSearchRunner::join_quality(
                            FiniteSearchQuality::UpperBound,
                            result.quality,
                        );
                    }
                    return Ok(FiniteSearchResult {
                        root_candidates_considered: 1,
                        all_legal_roots_evaluated: false,
                        candidates: vec![FiniteSearchCandidate {
                            guess_index: selected,
                            failure_probability: result.value.failure_probability,
                            expected_attempts: result.value.expected_attempts,
                            quality: result.quality,
                        }],
                        reason: runner
                            .control
                            .reason
                            .unwrap_or(FiniteSearchReason::Complete),
                        nodes_visited: runner.control.nodes_visited,
                        work_units: runner.control.work_units,
                        cache_hits: runner.cache_hits,
                        proposal_sampled: runner.proposal_sampled,
                    });
                }
            }
            return Ok(FiniteSearchResult {
                candidates: vec![runner.heuristic_candidate(selected)],
                root_candidates_considered: 1,
                all_legal_roots_evaluated: false,
                reason: runner
                    .control
                    .reason
                    .unwrap_or(FiniteSearchReason::Deadline),
                nodes_visited: runner.control.nodes_visited,
                work_units: runner.control.work_units,
                cache_hits: runner.cache_hits,
                proposal_sampled: runner.proposal_sampled,
            });
        }

        let enumerate_all =
            root.subset.len() <= options.exact_state_threshold || remaining_turns <= 2;
        let mut all_legal_roots_enumerated = false;
        let mut roots = if enumerate_all {
            let mut legal = runner.all_legal_guesses(&root);
            all_legal_roots_enumerated = runner.control.reason.is_none();
            legal.retain(|guess| *guess != quick);
            legal.insert(0, quick);
            legal
        } else {
            runner.propose_root_guesses(&root, quick)?
        };
        if runner.control.reason.is_none() && roots.is_empty() {
            bail!("finite-horizon search found no legal root guess");
        }
        if !enumerate_all
            && runner.control.reason.is_none()
            && let Some(baseline) = runner.baseline_guess(&root)?
        {
            roots.retain(|guess| *guess != baseline);
            roots.insert(0, baseline);
        }
        let mut candidates = runner.evaluate_roots(&root, &roots, remaining_turns)?;
        if !enumerate_all && runner.control.reason.is_none() {
            // The shortlist is an initial policy, not a stopping condition.
            // Preserve its completed incumbents while spending spare budget
            // on all legal roots and exact continuations.
            let additional_roots = runner.all_legal_guesses(&root);
            all_legal_roots_enumerated = runner.control.reason.is_none();
            let initial_roots = roots.clone();
            roots.extend(
                additional_roots
                    .into_iter()
                    .filter(|guess| !initial_roots.contains(guess)),
            );
            runner.options.exact_state_threshold = root.subset.len();
            candidates.extend(runner.evaluate_roots(&root, &roots, remaining_turns)?);
        }
        if candidates.is_empty() {
            candidates.push(runner.heuristic_candidate(roots.first().copied().unwrap_or(quick)));
        }
        candidates.sort_by(|left, right| {
            FiniteSearchRunner::compare_values(
                FiniteValue {
                    failure_probability: left.failure_probability,
                    expected_attempts: left.expected_attempts,
                },
                FiniteValue {
                    failure_probability: right.failure_probability,
                    expected_attempts: right.expected_attempts,
                },
            )
            .then_with(|| quality_order(left.quality).cmp(&quality_order(right.quality)))
            .then_with(|| self.guesses[left.guess_index].cmp(&self.guesses[right.guess_index]))
        });
        let mut seen = HashSet::new();
        candidates.retain(|candidate| seen.insert(candidate.guess_index));

        Ok(FiniteSearchResult {
            root_candidates_considered: roots.len(),
            all_legal_roots_evaluated: all_legal_roots_enumerated
                && runner.control.reason.is_none(),
            candidates,
            reason: runner
                .control
                .reason
                .unwrap_or(FiniteSearchReason::Complete),
            nodes_visited: runner.control.nodes_visited,
            work_units: runner.control.work_units,
            cache_hits: runner.cache_hits,
            proposal_sampled: runner.proposal_sampled,
        })
    }
}

fn quality_order(quality: FiniteSearchQuality) -> u8 {
    match quality {
        FiniteSearchQuality::Exact => 0,
        FiniteSearchQuality::UpperBound => 1,
        FiniteSearchQuality::Heuristic => 2,
    }
}

fn recovery_mode_code(mode: RecoveryMode) -> u8 {
    match mode {
        RecoveryMode::Strict => 0,
        RecoveryMode::UniformOverSupport => 1,
        RecoveryMode::EpsilonRepair => 2,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs, time::Duration};

    use crate::{
        config::PriorConfig,
        model::{AnswerRecord, ModelVariant, WeightMode},
        pattern_table::PatternTable,
        scoring::{ALL_GREEN_PATTERN, score_guess},
    };

    use super::{FiniteSearchOptions, FiniteSearchQuality, FiniteSearchReason, Solver};

    fn test_solver(words: &[&str]) -> Solver {
        test_solver_with_guesses(words, words)
    }

    fn test_solver_with_guesses(words: &[&str], guess_words: &[&str]) -> Solver {
        let guesses = guess_words
            .iter()
            .map(|word| (*word).to_string())
            .collect::<Vec<_>>();
        let answers = words
            .iter()
            .map(|word| AnswerRecord {
                word: (*word).to_string(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect::<Vec<_>>();
        let fixture = crate::test_support::TestDirectory::new("solver-fixture");
        let root = fixture.path().to_path_buf();
        fs::create_dir_all(&root).expect("test pattern root");
        let pattern_table =
            PatternTable::load_or_build_at(&root.join("pattern.bin"), &guesses, &answers)
                .expect("pattern table");
        Solver {
            config: PriorConfig::default(),
            mode: WeightMode::Uniform,
            variant: ModelVariant::SeedPlusHistory,
            data: std::sync::Arc::new(crate::solver::SolverData {
                guesses: guesses.clone(),
                answers,
                primary_answer_count: words.len(),
                history_dates: Vec::new(),
                pattern_table,
                guess_index: guesses
                    .iter()
                    .enumerate()
                    .map(|(index, word)| (word.clone(), index))
                    .collect::<HashMap<_, _>>(),
            }),
            artifact_dir: root.join("predictive"),
            session_opener_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_reply_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            session_third_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            identity_cache: Default::default(),
            test_fixture: Some(std::sync::Arc::new(fixture)),
        }
    }

    fn options(exact_state_threshold: usize) -> FiniteSearchOptions {
        FiniteSearchOptions {
            root_shortlist: 16,
            reply_shortlist: 16,
            exact_state_threshold,
            budget: Duration::from_secs(10),
            node_limit: None,
            baseline_only: false,
        }
    }

    fn oracle_feedback(guess: &str, answer: &str) -> u8 {
        let mut used = [false; 5];
        let mut marks = [0u8; 5];
        for i in 0..5 {
            if guess.as_bytes()[i] == answer.as_bytes()[i] {
                marks[i] = 2;
                used[i] = true;
            }
        }
        for (i, mark) in marks.iter_mut().enumerate() {
            if *mark == 0
                && let Some(j) =
                    (0..5).find(|j| !used[*j] && guess.as_bytes()[i] == answer.as_bytes()[*j])
            {
                *mark = 1;
                used[j] = true;
            }
        }
        marks.iter().rev().fold(0, |value, mark| value * 3 + mark)
    }

    fn oracle_legal(guess: &str, history: &[(String, u8)]) -> bool {
        history.iter().all(|(previous, pattern)| {
            let marks = (0..5)
                .map(|i| (*pattern / 3u8.pow(i)) % 3)
                .collect::<Vec<_>>();
            (0..5).all(|i| {
                let letter = previous.as_bytes()[i];
                let required = (0..5)
                    .filter(|j| marks[*j] != 0 && previous.as_bytes()[*j] == letter)
                    .count();
                (marks[i] != 2 || guess.as_bytes()[i] == letter)
                    && (marks[i] != 1 || guess.as_bytes()[i] != letter)
                    && guess.bytes().filter(|byte| *byte == letter).count() >= required
            })
        })
    }

    fn oracle_move(
        solver: &Solver,
        subset: &[usize],
        weights: &[u64],
        history: &[(String, u8)],
        h: u8,
        hard: bool,
        guess: usize,
    ) -> (u64, u64) {
        let total = subset.iter().map(|index| weights[*index]).sum::<u64>();
        let mut buckets = std::collections::BTreeMap::<u8, Vec<usize>>::new();
        for index in subset {
            if weights[*index] > 0 {
                buckets
                    .entry(oracle_feedback(
                        &solver.guesses[guess],
                        &solver.answers[*index].word,
                    ))
                    .or_default()
                    .push(*index);
            }
        }
        // Integer mass makes comparisons exact; no shared floating comparator.
        let mut value = (0, total);
        for (pattern, child) in buckets {
            if pattern == 242 {
                continue;
            }
            let mass = child.iter().map(|index| weights[*index]).sum::<u64>();
            let mut next = history.to_vec();
            next.push((solver.guesses[guess].clone(), pattern));
            let continuation = if h == 1 {
                (mass, 0)
            } else {
                (0..solver.guesses.len())
                    .filter(|guess| !hard || oracle_legal(&solver.guesses[*guess], &next))
                    .map(|guess| oracle_move(solver, &child, weights, &next, h - 1, hard, guess))
                    .min()
                    .unwrap_or((mass, 0))
            };
            value.0 += continuation.0;
            value.1 += continuation.1;
        }
        value
    }

    fn dynamic_oracle_value(
        solver: &Solver,
        state: &crate::solver::SolveState,
        history: &[(String, u8)],
        turns: u8,
        hard_mode: bool,
    ) -> (f64, f64) {
        (0..solver.guesses.len())
            .filter(|guess| !hard_mode || oracle_legal(&solver.guesses[*guess], history))
            .map(|guess| dynamic_oracle_move(solver, state, history, turns, hard_mode, guess))
            .min_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.total_cmp(&right.1))
            })
            .unwrap_or((1.0, 0.0))
    }

    fn dynamic_oracle_move(
        solver: &Solver,
        state: &crate::solver::SolveState,
        history: &[(String, u8)],
        turns: u8,
        hard_mode: bool,
        guess_index: usize,
    ) -> (f64, f64) {
        let total_weight = state
            .surviving
            .iter()
            .map(|index| state.weights[*index])
            .sum::<f64>();
        let mut buckets = std::collections::BTreeMap::<u8, Vec<usize>>::new();
        for answer_index in &state.surviving {
            if state.weights[*answer_index] > 0.0 {
                buckets
                    .entry(oracle_feedback(
                        &solver.guesses[guess_index],
                        &solver.answers[*answer_index].word,
                    ))
                    .or_default()
                    .push(*answer_index);
            }
        }

        let mut failure_probability = 0.0;
        let mut expected_attempts = 1.0;
        for (pattern, child_answers) in buckets {
            if pattern == 242 {
                continue;
            }
            let mass = child_answers
                .iter()
                .map(|index| state.weights[*index])
                .sum::<f64>();
            let probability = mass / total_weight;
            if turns == 1 {
                failure_probability += probability;
                continue;
            }

            let mut child = state.clone();
            solver
                .apply_feedback(&mut child, &solver.guesses[guess_index], pattern)
                .expect("oracle feedback transition");
            let mut child_history = history.to_vec();
            child_history.push((solver.guesses[guess_index].clone(), pattern));
            let continuation =
                dynamic_oracle_value(solver, &child, &child_history, turns - 1, hard_mode);
            failure_probability += probability * continuation.0;
            expected_attempts += probability * continuation.1;
        }
        (failure_probability, expected_attempts)
    }

    #[test]
    fn baseline_is_state_local_and_conditioning_invariant() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let weights = [3.0, 1.0, 2.0, 1.0, 1.0, 0.0];
        let root = super::FiniteNode {
            subset: vec![0, 1, 2, 3, 4],
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let child = super::FiniteNode {
            subset: vec![1, 2, 3],
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let mut rollout =
            super::FiniteSearchRunner::new(&solver, &weights, options(3), false, &|| false);
        rollout.baseline_guess(&root).unwrap().unwrap();
        let continuation = rollout.baseline_guess(&child).unwrap().unwrap();
        let conditioned = [0.0, 0.25, 0.5, 0.25, 0.0, 0.0];
        let mut fresh =
            super::FiniteSearchRunner::new(&solver, &conditioned, options(3), false, &|| false);
        assert_eq!(fresh.baseline_guess(&child).unwrap(), Some(continuation));
        assert!(
            child
                .subset
                .iter()
                .any(|answer| { solver.answers[*answer].word == solver.guesses[continuation] })
        );
    }

    #[test]
    fn baseline_shortlist_completes_before_exact_refinement() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let weights = [3.0, 1.0, 2.0, 1.0, 1.0, 0.0];
        let node = super::FiniteNode {
            subset: vec![0, 1, 2, 3, 4],
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let roots = vec![0, 5];
        let mut dry =
            super::FiniteSearchRunner::new(&solver, &weights, options(3), false, &|| false);
        for guess in &roots {
            assert!(dry.control.poll(true));
            dry.evaluate_move(&node, *guess, 4, 0, false)
                .unwrap()
                .unwrap();
        }
        let mut bounded_options = options(3);
        bounded_options.node_limit = Some(dry.control.work_units);
        let mut bounded =
            super::FiniteSearchRunner::new(&solver, &weights, bounded_options, false, &|| false);
        let result = bounded.evaluate_roots(&node, &roots, 4).unwrap();
        assert_eq!(bounded.control.reason, Some(FiniteSearchReason::NodeBudget));
        assert_eq!(
            result.iter().map(|row| row.guess_index).collect::<Vec<_>>(),
            roots
        );
        assert!(
            result
                .iter()
                .all(|row| row.quality == FiniteSearchQuality::UpperBound)
        );
        assert!(bounded.exact_memo.is_empty());
    }

    #[test]
    fn spare_budget_expands_beyond_the_root_shortlist() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let weights = [3.0, 1.0, 2.0, 1.0, 1.0, 0.0];
        let oracle_weights = [3_u64, 1, 2, 1, 1, 0];
        let total_mass = oracle_weights.iter().sum::<u64>() as f64;
        for hard in [false, true] {
            let mut bounded = options(2);
            bounded.root_shortlist = 1;
            let result = solver
                .finite_horizon_search(&[0, 1, 2, 3, 4], &weights, &[], 4, hard, bounded, &|| false)
                .unwrap();
            assert_eq!(result.reason, FiniteSearchReason::Complete);
            assert_eq!(result.candidates.len(), solver.guesses.len());
            for candidate in &result.candidates {
                let counts = oracle_move(
                    &solver,
                    &[0, 1, 2, 3, 4],
                    &oracle_weights,
                    &[],
                    4,
                    hard,
                    candidate.guess_index,
                );
                assert_eq!(candidate.quality, FiniteSearchQuality::Exact);
                assert!(
                    (candidate.failure_probability - counts.0 as f64 / total_mass).abs() < 1e-12
                );
                assert!((candidate.expected_attempts - counts.1 as f64 / total_mass).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn widening_interruption_preserves_completed_roots() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let weights = [3.0, 1.0, 2.0, 1.0, 1.0, 0.0];
        let mut bounded = options(2);
        bounded.root_shortlist = 1;
        let node = super::FiniteNode {
            subset: vec![0, 1, 2, 3, 4],
            observations: Vec::new(),
            dynamic_belief: None,
        };

        // Account for the complete initial shortlist, then stop on the first
        // work unit of the spare-budget root scan.
        let mut probe =
            super::FiniteSearchRunner::new(&solver, &weights, bounded, false, &|| false);
        probe.total_weight(&node).unwrap();
        let quick = probe.quick_legal_candidate(&node).expect("quick candidate");
        assert!(probe.control.poll(true));
        let mut roots = probe.propose_root_guesses(&node, quick).unwrap();
        assert!(probe.control.reason.is_none());
        if let Some(baseline) = probe.baseline_guess(&node).unwrap() {
            roots.retain(|guess| *guess != baseline);
            roots.insert(0, baseline);
        }
        let initial = probe.evaluate_roots(&node, &roots, 4).unwrap();
        assert!(probe.control.reason.is_none());
        assert!(!initial.is_empty());
        let initial_root_count = initial
            .iter()
            .map(|candidate| candidate.guess_index)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(initial_root_count < solver.guesses.len());

        bounded.node_limit = Some(probe.control.work_units);
        let result = solver
            .finite_horizon_search(&[0, 1, 2, 3, 4], &weights, &[], 4, false, bounded, &|| {
                false
            })
            .unwrap();
        assert_eq!(result.reason, FiniteSearchReason::NodeBudget);
        assert_eq!(result.work_units, probe.control.work_units + 1);
        assert!(!result.candidates.is_empty());
        assert!(result.candidates.iter().all(|candidate| {
            initial
                .iter()
                .any(|incumbent| incumbent.guess_index == candidate.guess_index)
        }));
    }

    #[test]
    fn completed_baseline_survives_refinement_exhaustion() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let weights = [3.0, 1.0, 2.0, 1.0, 1.0, 0.0];
        let node = super::FiniteNode {
            subset: vec![0, 1, 2, 3, 4],
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let roots = vec![0, 5];
        let mut dry =
            super::FiniteSearchRunner::new(&solver, &weights, options(3), false, &|| false);
        let baseline = dry.evaluate_move(&node, 0, 4, 0, false).unwrap().unwrap();
        assert!(dry.exact_memo.is_empty());
        let mut bounded_options = options(3);
        bounded_options.node_limit = Some(dry.control.work_units + 1);
        let mut bounded =
            super::FiniteSearchRunner::new(&solver, &weights, bounded_options, false, &|| false);
        let result = bounded.evaluate_roots(&node, &roots, 4).unwrap();
        assert_eq!(bounded.control.reason, Some(FiniteSearchReason::NodeBudget));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].quality, FiniteSearchQuality::UpperBound);
        assert_eq!(
            result[0].failure_probability,
            baseline.value.failure_probability
        );
        assert_eq!(
            result[0].expected_attempts,
            baseline.value.expected_attempts
        );
    }

    #[test]
    fn cooperative_proposals_use_the_shared_proxy_formula() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let weights = [3.0, 1.0, 2.0, 1.0, 1.0, 0.0];
        let subset = [0, 1, 2, 3, 4];
        let mut runner =
            super::FiniteSearchRunner::new(&solver, &weights, options(8), false, &|| false);
        let node = super::FiniteNode {
            subset: subset.to_vec(),
            observations: Vec::new(),
            dynamic_belief: None,
        };
        for guess in 0..solver.guesses.len() {
            let proposal = runner
                .cooperative_proposal_metric(&node, &subset, 8.0, guess, false)
                .unwrap();
            let reference = solver.score_guess_metrics(
                guess,
                &mut super::GuessMetricScratch::new(),
                super::GuessMetricContext {
                    subset: &subset,
                    weights: &weights,
                    total_weight: 8.0,
                    posterior_answer_probability: weights[guess] / 8.0,
                },
            );
            assert_eq!(proposal.proxy_score, reference.large_state_score);
            assert_eq!(proposal.entropy, reference.entropy);
            assert_eq!(proposal.solve_probability, reference.solve_probability);
        }
    }

    #[test]
    fn forced_cancellation_is_checked_before_proposal_reservation() {
        let mut bounded = options(8);
        bounded.node_limit = Some(4);
        let cancelled = std::cell::Cell::new(false);
        let is_cancelled = || cancelled.get();
        let mut control = super::FiniteSearchControl::new(bounded, &is_cancelled);
        assert!(control.poll_proposal(true));
        cancelled.set(true);
        assert!(!control.poll_proposal(true));
        assert_eq!(control.reason, Some(FiniteSearchReason::Cancelled));
    }

    #[test]
    fn bounded_root_preparation_prioritizes_unique_letter_probe() {
        let solver = test_solver(&[
            "aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff", "ggggg", "hhhhh", "zebra",
        ]);
        let weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let node = super::FiniteNode {
            subset: (0..solver.answers.len()).collect(),
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let mut bounded_options = options(8);
        bounded_options.root_shortlist = 1;
        // Preparation plus one complete partition fits; the next scan does not.
        let one_proposal_work =
            1 + node.subset.len() + solver.guesses.len() + 1 + node.subset.len();
        bounded_options.node_limit = Some(one_proposal_work * 4);
        let mut runner =
            super::FiniteSearchRunner::new(&solver, &weights, bounded_options, false, &|| false);
        assert!(runner.control.visit_node());
        let quick = runner
            .quick_legal_candidate(&node)
            .expect("quick candidate");
        assert_eq!(quick, 0);

        let roots = runner
            .propose_root_guesses(&node, quick)
            .expect("bounded proposal roots");
        assert_eq!(roots.first().copied(), Some(8));
        assert!(roots.contains(&quick));
        assert!(runner.proposal_sampled);
    }

    #[test]
    fn exact_search_stops_when_a_feasible_policy_attains_the_bound() {
        let solver = test_solver(&["cigar", "rebut", "sissy", "humph", "awake", "blush"]);
        let weights = [3.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        for hard in [false, true] {
            let mut runner =
                super::FiniteSearchRunner::new(&solver, &weights, options(8), hard, &|| false);
            let node = super::FiniteNode {
                subset: vec![0, 1, 2],
                observations: Vec::new(),
                dynamic_belief: None,
            };
            let result = runner.evaluate_exact_node(&node, 4, 0).unwrap().unwrap();
            assert_eq!(result.value.failure_probability, 0.0);
            assert!((result.value.expected_attempts - 1.4).abs() < 1e-12);
            assert!(runner.control.nodes_visited <= 3);
        }
        for horizon in [0, 7, u8::MAX] {
            assert!(
                solver
                    .finite_horizon_search(&[0], &weights, &[], horizon, false, options(8), &|| {
                        false
                    })
                    .is_err()
            );
        }
    }

    #[test]
    fn recursive_exact_argmin_uses_partial_branch_bounds_before_budget_expires() {
        let answers = ["aaaaa", "bbbbb", "ccccc", "ddddd"];
        let guesses = ["aaaaa", "bbbbb", "ccccc", "ddddd", "aaaxy"];
        let solver = test_solver_with_guesses(&answers, &guesses);
        let weights = [1.0; 4];
        let node = super::FiniteNode {
            subset: (0..answers.len()).collect(),
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let mut bounded_options = options(8);
        // Each later answer guess becomes unable to improve after one known
        // non-green branch. A full evaluation of every move exceeds this cap.
        bounded_options.node_limit = Some(67);
        let mut runner =
            super::FiniteSearchRunner::new(&solver, &weights, bounded_options, false, &|| false);

        let value = runner
            .evaluate_exact_node(&node, 2, 0)
            .expect("bounded search")
            .expect("feasible exact incumbent");

        assert_eq!(runner.control.reason, None);
        assert_eq!(value.quality, FiniteSearchQuality::Exact);
        assert_eq!(value.value.failure_probability, 0.5);
        assert_eq!(value.value.expected_attempts, 1.75);
    }

    #[test]
    fn equivalent_partitions_keep_green_semantics_and_fixed_root_oracle_values() {
        let answers = ["aaaaa", "bbbbb", "ccccc", "ddddd"];
        let guesses = [
            "aaaaa", "bbbbb", "ccccc", "ddddd", "aaaxy", "aaazq", "fffff", "ggggg",
        ];
        let solver = test_solver_with_guesses(&answers, &guesses);
        let weights = [1.0; 4];
        let subset = (0..answers.len()).collect::<Vec<_>>();
        let result = solver
            .finite_horizon_search(&subset, &weights, &[], 3, false, options(8), &|| false)
            .expect("finite search");

        assert_eq!(result.reason, FiniteSearchReason::Complete);
        assert_eq!(result.candidates.len(), guesses.len());
        for candidate in result.candidates {
            let expected = oracle_move(
                &solver,
                &subset,
                &[1, 1, 1, 1],
                &[],
                3,
                false,
                candidate.guess_index,
            );
            assert_eq!(candidate.quality, FiniteSearchQuality::Exact);
            assert_eq!(candidate.failure_probability, expected.0 as f64 / 4.0);
            assert_eq!(candidate.expected_attempts, expected.1 as f64 / 4.0);
        }

        let mut runner =
            super::FiniteSearchRunner::new(&solver, &weights, options(8), false, &|| false);
        let first_probe = super::FiniteNode {
            subset: subset.clone(),
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let second_probe = super::FiniteNode {
            subset: subset.clone(),
            observations: Vec::new(),
            dynamic_belief: None,
        };
        let first = runner
            .evaluate_move(&first_probe, 4, 2, 0, true)
            .expect("first probe")
            .expect("complete first probe")
            .value;
        let second = runner
            .evaluate_move(&second_probe, 5, 2, 0, true)
            .expect("second probe")
            .expect("complete second probe")
            .value;
        assert_eq!(first.failure_probability, second.failure_probability);
        assert_eq!(first.expected_attempts, second.expected_attempts);
    }

    #[test]
    fn one_and_two_answer_endgames_do_not_scan_the_dictionary() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let weights = [3.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        for hard in [false, true] {
            for (subset, expected_attempts) in [(vec![0], 1.0), (vec![0, 1], 1.25)] {
                let mut runner =
                    super::FiniteSearchRunner::new(&solver, &weights, options(8), hard, &|| false);
                let node = super::FiniteNode {
                    subset,
                    observations: Vec::new(),
                    dynamic_belief: None,
                };
                let result = runner.evaluate_exact_node(&node, 6, 0).unwrap().unwrap();
                assert_eq!(result.value.failure_probability, 0.0);
                assert_eq!(result.value.expected_attempts, expected_attempts);
                assert!(runner.control.work_units <= 3);
            }
        }
    }

    #[test]
    fn finite_values_match_independent_recursive_normal_and_hard_oracle() {
        for words in [
            ["bound", "hound", "mound", "pound", "wound", "whomp"],
            ["allee", "llama", "apple", "ample", "eerie", "cigar"],
        ] {
            let solver = test_solver(&words);
            let weights = [0.35, 0.1, 0.2, 0.15, 0.2, 0.0];
            let subset = [0, 1, 2, 3, 4];
            for hard in [false, true] {
                for h in 1..=4 {
                    let result = solver
                        .finite_horizon_search(&subset, &weights, &[], h, hard, options(8), &|| {
                            false
                        })
                        .expect("finite");
                    assert_eq!(result.reason, FiniteSearchReason::Complete);
                    assert_eq!(result.candidates.len(), solver.guesses.len());
                    for row in result.candidates {
                        let counts = oracle_move(
                            &solver,
                            &subset,
                            &[7, 2, 4, 3, 4, 0],
                            &[],
                            h,
                            hard,
                            row.guess_index,
                        );
                        let expected = (counts.0 as f64 / 20.0, counts.1 as f64 / 20.0);
                        assert!(
                            (row.failure_probability - expected.0).abs() < 1e-12,
                            "failure mismatch h={h} hard={hard}"
                        );
                        assert!(
                            (row.expected_attempts - expected.1).abs() < 1e-12,
                            "attempt mismatch h={h} hard={hard} word={} actual=({}, {}) expected={expected:?}",
                            solver.guesses[row.guess_index],
                            row.failure_probability,
                            row.expected_attempts
                        );
                        assert_eq!(row.quality, FiniteSearchQuality::Exact);
                    }
                }
            }
        }
    }

    #[test]
    fn dynamic_finite_values_match_recovery_and_duplicate_clue_oracles() {
        let mut solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee"]);
        solver.data_mut().primary_answer_count = 3;
        solver.config.fallback_activation_threshold = 2;
        let state = solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 3, 10).unwrap());
        assert_eq!(state.surviving, vec![0, 1, 2]);
        assert_eq!(state.fallback_surviving, vec![3, 4]);

        let result = solver
            .finite_horizon_search_dynamic(&state, &[], 3, false, options(8), &|| false)
            .expect("finite search");
        let gray_child = state.clone();
        let mut activated = gray_child;
        solver
            .apply_feedback(&mut activated, "aaaaa", 0)
            .expect("activate dormant fallback");
        assert_eq!(activated.surviving, vec![1, 2, 3, 4]);
        assert!(activated.fallback_active);

        let expected = dynamic_oracle_move(&solver, &state, &[], 3, false, 0);
        assert!(expected.0 > 0.0, "recovered child must expose finite risk");
        let actual = result
            .candidates
            .iter()
            .find(|candidate| solver.guesses[candidate.guess_index] == "aaaaa")
            .expect("probe candidate");
        assert!(
            (actual.failure_probability - expected.0).abs() < 1e-12,
            "static continuation missed the activated fallback: actual={}, expected={}",
            actual.failure_probability,
            expected.0
        );
        assert!((actual.expected_attempts - expected.1).abs() < 1e-12);
        for candidate in &result.candidates {
            let expected =
                dynamic_oracle_move(&solver, &state, &[], 3, false, candidate.guess_index);
            assert!((candidate.failure_probability - expected.0).abs() < 1e-12);
            assert!((candidate.expected_attempts - expected.1).abs() < 1e-12);
        }

        let duplicate_solver = test_solver(&["allee", "llama", "apple", "ample"]);
        let duplicate_state =
            duplicate_solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 3, 10).unwrap());
        let history = Vec::new();
        let duplicate_result = duplicate_solver
            .finite_horizon_search_dynamic(&duplicate_state, &history, 3, true, options(8), &|| {
                false
            })
            .expect("hard-mode finite search");
        let clue = oracle_feedback("allee", "llama");
        let duplicate_history = [("allee".to_string(), clue)];
        let legal_after_duplicate_clue = duplicate_solver
            .guesses
            .iter()
            .filter(|guess| oracle_legal(guess, &duplicate_history))
            .map(String::as_str)
            .collect::<Vec<_>>();
        assert_eq!(legal_after_duplicate_clue, ["llama"]);
        for candidate in duplicate_result.candidates {
            let expected = dynamic_oracle_move(
                &duplicate_solver,
                &duplicate_state,
                &history,
                3,
                true,
                candidate.guess_index,
            );
            assert!((candidate.failure_probability - expected.0).abs() < 1e-12);
            assert!((candidate.expected_attempts - expected.1).abs() < 1e-12);
        }
    }

    #[test]
    fn dormant_five_word_all_roots_match_oracle_across_thresholds_and_horizons() {
        for threshold in [0, 1, 2, 4] {
            let mut solver = test_solver(&["tower", "power", "bower", "rower", "sower"]);
            solver.data_mut().primary_answer_count = 3;
            solver.config.fallback_activation_threshold = threshold;
            solver.config.fallback_prior_mass = 0.4;
            let initial =
                solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 3, 10).unwrap());
            let clue = oracle_feedback("rower", "tower");
            let mut filtered = initial.clone();
            solver.apply_feedback(&mut filtered, "rower", clue).unwrap();
            assert!(!filtered.fallback_surviving.contains(&3));
            for (state, history) in [
                (initial, Vec::new()),
                (filtered, vec![("rower".to_string(), clue)]),
            ] {
                for hard in [false, true] {
                    for turns in 1..=4 {
                        let result = solver
                            .finite_horizon_search_dynamic(
                                &state,
                                &history,
                                turns,
                                hard,
                                options(8),
                                &|| false,
                            )
                            .unwrap();
                        assert_eq!(result.reason, FiniteSearchReason::Complete);
                        let legal = solver
                            .guesses
                            .iter()
                            .filter(|guess| !hard || oracle_legal(guess, &history))
                            .count();
                        assert_eq!(result.candidates.len(), legal);
                        for candidate in &result.candidates {
                            let expected = dynamic_oracle_move(
                                &solver,
                                &state,
                                &history,
                                turns,
                                hard,
                                candidate.guess_index,
                            );
                            assert!(
                                (candidate.failure_probability - expected.0).abs() < 1e-12,
                                "threshold={threshold} turns={turns} hard={hard} history={history:?} guess={} actual={} expected={}",
                                solver.guesses[candidate.guess_index],
                                candidate.failure_probability,
                                expected.0
                            );
                            assert!((candidate.expected_attempts - expected.1).abs() < 1e-12);
                            assert_eq!(candidate.quality, FiniteSearchQuality::Exact);
                            if threshold == 1
                                && turns == 3
                                && history.is_empty()
                                && candidate.guess_index == 0
                            {
                                assert!((candidate.failure_probability - 2.0 / 9.0).abs() < 1e-12);
                                assert!((candidate.expected_attempts - 2.0).abs() < 1e-12);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nonprogressing_active_probe_can_remove_dormant_support_and_improve_risk() {
        let words = ["aaaaa", "bbbbb", "cdddd", "cffff", "cgggg"];
        let guesses = ["aaaaa", "bbbbb", "cdddd", "cffff", "cgggg", "ccccc"];
        let mut solver = test_solver_with_guesses(&words, &guesses);
        solver.data_mut().primary_answer_count = 2;
        solver.config.fallback_activation_threshold = 1;
        solver.config.fallback_prior_mass = 0.4;
        let state = solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 3, 10).unwrap());
        assert_eq!(state.surviving, [0, 1]);
        let expected = dynamic_oracle_value(&solver, &state, &[], 3, false);
        assert_eq!(expected.0, 0.0);
        assert_eq!(expected.1, 2.5);
        let node = super::FiniteNode {
            subset: state.surviving.clone(),
            observations: Vec::new(),
            dynamic_belief: Some(super::FiniteDynamicBelief {
                weights: state
                    .surviving
                    .iter()
                    .map(|index| state.weights[*index])
                    .collect(),
                fallback_surviving: state.fallback_surviving.clone(),
                condition_only: state.condition_only,
                fallback_active: state.fallback_active,
                recovery_mode_used: state.recovery_mode_used,
            }),
        };
        let mut runner = super::FiniteSearchRunner::with_dynamic_basis(
            &solver,
            &state.weights,
            Some(&state),
            options(8),
            false,
            &|| false,
        );
        for _ in 0..2 {
            let actual = runner.evaluate_exact_node(&node, 3, 0).unwrap().unwrap();
            assert!((actual.value.failure_probability - expected.0).abs() < 1e-12);
            assert!((actual.value.expected_attempts - expected.1).abs() < 1e-12);
            assert_eq!(actual.quality, FiniteSearchQuality::Exact);
        }
        assert!(
            runner.cache_hits > 0,
            "the repeated evaluation must reuse its exact memo"
        );
    }

    #[test]
    fn dynamic_two_answer_continuation_counts_dormant_reactivation() {
        let mut solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "ddddd"]);
        solver.data_mut().primary_answer_count = 3;
        solver.config.fallback_activation_threshold = 1;
        let state = solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 3, 10).unwrap());
        assert_eq!(state.surviving, vec![0, 1, 2]);
        assert_eq!(state.fallback_surviving, vec![3]);

        let guess_index = solver.guess_index["aaaaa"];
        let expected = dynamic_oracle_move(&solver, &state, &[], 3, false, guess_index);
        assert!(
            expected.0 > 0.0,
            "a later miss must activate dormant support"
        );

        let result = solver
            .finite_horizon_search_dynamic(&state, &[], 3, false, options(8), &|| false)
            .expect("finite search");
        let actual = result
            .candidates
            .iter()
            .find(|candidate| candidate.guess_index == guess_index)
            .expect("first-guess candidate");
        assert_eq!(actual.quality, FiniteSearchQuality::Exact);
        assert!(
            (actual.failure_probability - expected.0).abs() < 1e-12,
            "dormant activation missed: actual={}, expected={}",
            actual.failure_probability,
            expected.0
        );
        assert!((actual.expected_attempts - expected.1).abs() < 1e-12);
    }

    #[test]
    fn dynamic_fallback_partition_preserves_duplicate_feedback_transitions_within_work_cap() {
        let words = [
            "allee", "llama", "cigar", "eerie", "llapy", "party", "ebcde", "apple", "ample",
            "selle", "level",
        ];
        let mut solver = test_solver_with_guesses(&words, &["allee"]);
        solver.data_mut().primary_answer_count = 4;
        solver.config.fallback_activation_threshold = 2;
        let state = solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 3, 10).unwrap());
        let guess_index = solver.guess_index["allee"];
        let expected_patterns = [
            oracle_feedback("allee", "llama"),
            oracle_feedback("allee", "cigar"),
            oracle_feedback("allee", "eerie"),
        ];
        let fallback_patterns = words[4..]
            .iter()
            .map(|word| oracle_feedback("allee", word))
            .collect::<Vec<_>>();
        let mut distinct_patterns = expected_patterns.to_vec();
        distinct_patterns.sort_unstable();
        distinct_patterns.dedup();
        assert_eq!(distinct_patterns.len(), expected_patterns.len());
        assert!(
            expected_patterns
                .iter()
                .all(|pattern| fallback_patterns.contains(pattern))
        );

        let options = FiniteSearchOptions {
            node_limit: Some(42),
            ..options(usize::MAX)
        };
        let mut runner = super::FiniteSearchRunner::with_dynamic_basis(
            &solver,
            &state.weights,
            Some(&state),
            options,
            false,
            &|| false,
        );
        let root = super::FiniteNode {
            subset: state.surviving.clone(),
            observations: Vec::new(),
            dynamic_belief: Some(super::FiniteDynamicBelief {
                weights: state
                    .surviving
                    .iter()
                    .map(|index| state.weights[*index])
                    .collect(),
                fallback_surviving: state.fallback_surviving.clone(),
                condition_only: state.condition_only,
                fallback_active: state.fallback_active,
                recovery_mode_used: state.recovery_mode_used,
            }),
        };
        let total_weight = runner.total_weight(&root).expect("root weight");
        let Some(partition) = runner
            .partition_patterns(&root, guess_index, 0, false)
            .expect("root partition")
        else {
            panic!("root partition should fit the work cap");
        };
        assert!(
            expected_patterns
                .iter()
                .all(|pattern| partition.patterns.contains(pattern))
        );

        let result = runner
            .evaluate_partitioned_move(
                &root,
                guess_index,
                2,
                0,
                true,
                total_weight,
                &partition.patterns,
                None,
            )
            .expect("candidate evaluation");
        assert!(
            matches!(result, Some(super::FiniteMoveEvaluation::Complete(_))),
            "duplicate-feedback children should complete within the work cap; reason={:?}, units={}",
            runner.control.reason,
            runner.control.work_units
        );

        for pattern in expected_patterns {
            let mut expected_state = state.clone();
            solver
                .apply_feedback(&mut expected_state, "allee", pattern)
                .expect("oracle child transition");
            if fallback_patterns.contains(&pattern) {
                assert!(expected_state.fallback_active);
            }
            expected_state.surviving.sort_unstable();
            let expected_child = super::FiniteNode {
                subset: expected_state.surviving.clone(),
                observations: Vec::new(),
                dynamic_belief: Some(super::FiniteDynamicBelief {
                    weights: expected_state
                        .surviving
                        .iter()
                        .map(|index| expected_state.weights[*index])
                        .collect(),
                    fallback_surviving: expected_state.fallback_surviving,
                    condition_only: expected_state.condition_only,
                    fallback_active: expected_state.fallback_active,
                    recovery_mode_used: expected_state.recovery_mode_used,
                }),
            };
            let expected_key = runner.node_key(&expected_child, 1);
            assert!(
                runner.exact_memo.contains_key(&expected_key),
                "the evaluated child for duplicate feedback pattern {pattern} must match the state transition"
            );
        }
    }

    #[test]
    fn dynamic_fallback_transition_obeys_the_work_budget() {
        let mut solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee"]);
        solver.data_mut().primary_answer_count = 3;
        solver.config.fallback_activation_threshold = 2;
        let state = solver.initial_state(chrono::NaiveDate::from_ymd_opt(2026, 3, 10).unwrap());
        let mut bounded = options(8);
        bounded.node_limit = Some(2);
        let mut runner = super::FiniteSearchRunner::with_dynamic_basis(
            &solver,
            &state.weights,
            Some(&state),
            bounded,
            false,
            &|| false,
        );
        let subset = state.surviving.clone();
        let root = super::FiniteNode {
            dynamic_belief: Some(super::FiniteDynamicBelief {
                weights: subset
                    .iter()
                    .map(|answer_index| state.weights[*answer_index])
                    .collect(),
                fallback_surviving: state.fallback_surviving.clone(),
                condition_only: state.condition_only,
                fallback_active: state.fallback_active,
                recovery_mode_used: state.recovery_mode_used,
            }),
            subset,
            observations: Vec::new(),
        };

        let child = runner
            .child_node(&root, 0, 0, vec![1, 2], None)
            .expect("dynamic child transition");
        assert!(
            child.is_none(),
            "dormant filtering must stop at the work cap"
        );
        assert_eq!(runner.control.reason, Some(FiniteSearchReason::NodeBudget));
        assert_eq!(runner.control.work_units, 3);
    }

    #[test]
    fn skewed_and_zero_mass_exact_values_match_independent_oracle() {
        let solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "ddddd"]);
        let subset = [0, 1, 2, 3];
        let weights = [0.40, 0.59, 0.01, 0.0];
        for hard in [false, true] {
            for turns in [2, 3] {
                let result = solver
                    .finite_horizon_search(&subset, &weights, &[], turns, hard, options(8), &|| {
                        false
                    })
                    .expect("finite search");
                assert_eq!(result.reason, FiniteSearchReason::Complete);
                for candidate in result.candidates {
                    let expected = oracle_move(
                        &solver,
                        &subset,
                        &[40, 59, 1, 0],
                        &[],
                        turns,
                        hard,
                        candidate.guess_index,
                    );
                    assert_eq!(candidate.quality, FiniteSearchQuality::Exact);
                    assert!(
                        (candidate.failure_probability - expected.0 as f64 / 100.0).abs() < 1e-12
                    );
                    assert!(
                        (candidate.expected_attempts - expected.1 as f64 / 100.0).abs() < 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn cancellation_inside_search_returns_unevaluated_legal_action() {
        let solver = test_solver(&["bound", "hound", "mound", "pound", "wound", "whomp"]);
        let calls = std::cell::Cell::new(0usize);
        let result = solver
            .finite_horizon_search(
                &[0, 1, 2, 3, 4],
                &[1.0; 6],
                &[],
                4,
                false,
                options(8),
                &|| {
                    calls.set(calls.get() + 1);
                    calls.get() >= 5
                },
            )
            .expect("cancelled search");
        assert_eq!(result.reason, FiniteSearchReason::Cancelled);
        assert!(result.work_units > 1);
        assert!(!result.candidates.is_empty());
        assert!(
            result
                .candidates
                .iter()
                .all(|row| row.guess_index < solver.guesses.len())
        );
    }

    #[test]
    fn recursive_exact_incumbent_survives_budget_without_poisoning_next_search() {
        let solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "abcde"]);
        let weights = [1.0; 4];
        let node = super::FiniteNode {
            subset: (0..4).collect(),
            observations: Vec::new(),
            dynamic_belief: None,
        };

        // Mirror the first candidate's deterministic work to place the limit
        // immediately after its complete policy is available.
        let mut probe =
            super::FiniteSearchRunner::new(&solver, &weights, options(8), false, &|| false);
        assert!(probe.control.visit_node());
        let mut legal_guesses = probe.all_legal_guesses(&node);
        assert!(probe.control.reason.is_none());
        let quick = probe.quick_legal_candidate(&node).expect("quick candidate");
        legal_guesses.retain(|guess| *guess != quick);
        legal_guesses.insert(0, quick);
        assert!(probe.control.poll(false));
        let first = probe
            .evaluate_move(&node, legal_guesses[0], 2, 0, true)
            .expect("first move")
            .expect("complete first move");
        assert!(first.value.failure_probability > 0.0);
        let cutoff = probe.control.work_units;

        let mut bounded_options = options(8);
        bounded_options.node_limit = Some(cutoff);
        let mut bounded =
            super::FiniteSearchRunner::new(&solver, &weights, bounded_options, false, &|| false);
        let incumbent = bounded
            .evaluate_exact_node(&node, 2, 0)
            .expect("bounded exact search")
            .expect("completed incumbent");
        assert_eq!(bounded.control.reason, Some(FiniteSearchReason::NodeBudget));
        assert_eq!(incumbent.quality, FiniteSearchQuality::UpperBound);
        assert_eq!(
            incumbent.value.failure_probability,
            first.value.failure_probability
        );
        assert_eq!(
            incumbent.value.expected_attempts,
            first.value.expected_attempts
        );
        let root_key = bounded.node_key(&node, 2);
        assert!(!bounded.exact_memo.contains_key(&root_key));
        assert!(
            bounded
                .exact_memo
                .values()
                .all(|cached| cached.quality == FiniteSearchQuality::Exact)
        );

        // A fresh complete run must still agree with the independent integer
        // oracle for every action after the interrupted recursive run.
        let complete = solver
            .finite_horizon_search(&[0, 1, 2, 3], &weights, &[], 2, false, options(8), &|| {
                false
            })
            .expect("complete follow-up search");
        assert_eq!(complete.reason, FiniteSearchReason::Complete);
        assert_eq!(complete.candidates.len(), solver.guesses.len());
        let oracle_weights = [1_u64; 4];
        for row in complete.candidates {
            let counts = oracle_move(
                &solver,
                &[0, 1, 2, 3],
                &oracle_weights,
                &[],
                2,
                false,
                row.guess_index,
            );
            assert_eq!(row.quality, FiniteSearchQuality::Exact);
            assert!((row.failure_probability - counts.0 as f64 / 4.0).abs() < 1e-12);
            assert!((row.expected_attempts - counts.1 as f64 / 4.0).abs() < 1e-12);
        }
    }

    #[test]
    fn one_turn_matches_independent_failure_oracle() {
        let solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "abcde"]);
        let weights = vec![1.0; solver.answers.len()];
        let result = solver
            .finite_horizon_search(
                &(0..solver.answers.len()).collect::<Vec<_>>(),
                &weights,
                &[],
                1,
                false,
                options(8),
                &|| false,
            )
            .expect("finite search");
        let mut oracle = Vec::new();
        for guess_index in 0..solver.guesses.len() {
            let green = (0..solver.answers.len())
                .filter(|answer_index| {
                    score_guess(
                        &solver.guesses[guess_index],
                        &solver.answers[*answer_index].word,
                    ) == ALL_GREEN_PATTERN
                })
                .count();
            oracle.push((
                1.0 - green as f64 / solver.answers.len() as f64,
                solver.guesses[guess_index].clone(),
            ));
        }
        oracle.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        assert_eq!(result.reason, FiniteSearchReason::Complete);
        assert_eq!(result.candidates[0].quality, FiniteSearchQuality::Exact);
        assert!((result.candidates[0].failure_probability - oracle[0].0).abs() <= 1e-12);
        assert_eq!(
            solver.guesses[result.candidates[0].guess_index],
            oracle[0].1
        );
    }

    #[test]
    fn two_turn_objective_can_choose_a_different_move() {
        let solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "abcde"]);
        let weights = vec![1.0; solver.answers.len()];
        let state = (0..solver.answers.len()).collect::<Vec<_>>();
        let one = solver
            .finite_horizon_search(&state, &weights, &[], 1, false, options(8), &|| false)
            .expect("one-turn search");
        let two = solver
            .finite_horizon_search(&state, &weights, &[], 2, false, options(8), &|| false)
            .expect("two-turn search");
        assert_eq!(solver.guesses[one.candidates[0].guess_index], "aaaaa");
        assert_eq!(solver.guesses[two.candidates[0].guess_index], "abcde");
        assert_eq!(two.candidates[0].failure_probability, 0.0);
    }

    #[test]
    fn memo_keys_separate_horizons_and_hard_mode_histories() {
        let solver = test_solver(&["allee", "llama", "apple", "ample"]);
        let weights = vec![1.0; solver.answers.len()];
        let mut runner =
            super::FiniteSearchRunner::new(&solver, &weights, options(8), true, &|| false);
        let node = super::FiniteNode {
            subset: vec![1],
            observations: vec![("allee".to_string(), score_guess("allee", "llama"))],
            dynamic_belief: None,
        };
        let mut different_history = node.clone();
        different_history.observations.clear();
        let key = runner.node_key(&node, 2);
        assert_ne!(key, runner.node_key(&node, 1));
        assert_ne!(key, runner.node_key(&different_history, 2));
        runner.hard_mode = false;
        assert_ne!(key, runner.node_key(&node, 2));
        assert_eq!(
            runner.node_key(&node, 2),
            runner.node_key(&different_history, 2),
            "normal-mode legal actions depend on survivors, not past hints"
        );
    }

    #[test]
    fn memo_keys_include_full_dynamic_belief_identity() {
        let solver = test_solver(&["aaaaa", "bbbbb", "ccccc", "ddddd"]);
        let weights = vec![1.0; solver.answers.len()];
        let runner = super::FiniteSearchRunner::new(&solver, &weights, options(8), true, &|| false);
        let node = super::FiniteNode {
            subset: vec![0, 1],
            observations: vec![("aaaaa".to_string(), 0)],
            dynamic_belief: Some(super::FiniteDynamicBelief {
                weights: vec![0.25, 0.75],
                fallback_surviving: vec![2, 3],
                condition_only: false,
                fallback_active: false,
                recovery_mode_used: None,
            }),
        };
        let key = runner.node_key(&node, 3);

        let mut different_weight = node.clone();
        different_weight.dynamic_belief.as_mut().unwrap().weights[0] = 0.5;
        assert_ne!(key, runner.node_key(&different_weight, 3));

        let mut different_dormant_support = node.clone();
        different_dormant_support
            .dynamic_belief
            .as_mut()
            .unwrap()
            .fallback_surviving
            .pop();
        assert_ne!(key, runner.node_key(&different_dormant_support, 3));

        let mut activated_fallback = node.clone();
        activated_fallback
            .dynamic_belief
            .as_mut()
            .unwrap()
            .fallback_active = true;
        assert_ne!(key, runner.node_key(&activated_fallback, 3));

        let mut condition_only = node.clone();
        condition_only
            .dynamic_belief
            .as_mut()
            .unwrap()
            .condition_only = true;
        assert_ne!(key, runner.node_key(&condition_only, 3));

        let mut recovered = node.clone();
        recovered
            .dynamic_belief
            .as_mut()
            .unwrap()
            .recovery_mode_used = Some(super::RecoveryMode::UniformOverSupport);
        assert_ne!(key, runner.node_key(&recovered, 3));

        let mut different_history = node.clone();
        different_history.observations[0].1 = 1;
        assert_ne!(key, runner.node_key(&different_history, 3));
    }

    #[test]
    fn hard_mode_candidates_use_full_history_and_duplicate_constraints() {
        let solver = test_solver(&["allee", "llama", "apple", "ample"]);
        let observations = vec![("allee".to_string(), score_guess("allee", "llama"))];
        let weights = vec![1.0; solver.answers.len()];
        let result = solver
            .finite_horizon_search(
                &(0..solver.answers.len()).collect::<Vec<_>>(),
                &weights,
                &observations,
                1,
                true,
                options(8),
                &|| false,
            )
            .expect("hard-mode search");
        assert!(result.candidates.iter().all(|candidate| {
            solver
                .hard_mode_violation(&observations, &solver.guesses[candidate.guess_index])
                .is_none()
        }));
        assert!(!result.candidates.is_empty());
    }

    #[test]
    fn zero_budget_returns_legal_heuristic_and_does_not_poison_next_search() {
        let solver = test_solver(&["cigar", "rebut", "sissy"]);
        let weights = vec![1.0; solver.answers.len()];
        let state = (0..solver.answers.len()).collect::<Vec<_>>();
        let mut immediate = options(8);
        immediate.budget = Duration::ZERO;
        let bounded = solver
            .finite_horizon_search(&state, &weights, &[], 2, false, immediate, &|| false)
            .expect("bounded search");
        assert_eq!(bounded.reason, FiniteSearchReason::Deadline);
        assert_eq!(bounded.candidates.len(), 1);
        assert_eq!(
            bounded.candidates[0].quality,
            FiniteSearchQuality::Heuristic
        );

        let complete = solver
            .finite_horizon_search(&state, &weights, &[], 2, false, options(8), &|| false)
            .expect("unbounded search");
        assert_eq!(complete.reason, FiniteSearchReason::Complete);
        assert_eq!(complete.candidates[0].quality, FiniteSearchQuality::Exact);
    }

    #[test]
    fn interrupted_exhaustive_root_scan_keeps_the_high_mass_answer() {
        let solver = test_solver(&["aaaaa", "loose", "solve"]);
        let weights = vec![0.01, 4.0, 1.0];
        let state = (0..solver.answers.len()).collect::<Vec<_>>();
        let mut bounded = options(8);
        bounded.node_limit = Some(2);
        let result = solver
            .finite_horizon_search(&state, &weights, &[], 3, false, bounded, &|| false)
            .expect("budgeted exhaustive search");
        assert_eq!(result.reason, FiniteSearchReason::NodeBudget);
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].quality, FiniteSearchQuality::Heuristic);
        assert_eq!(solver.guesses[result.candidates[0].guess_index], "loose");
    }

    #[test]
    fn baseline_only_deadline_returns_the_selected_action_as_heuristic() {
        let solver = test_solver(&["cigar", "rebut", "sissy"]);
        let weights = vec![1.0; solver.answers.len()];
        let state = (0..solver.answers.len()).collect::<Vec<_>>();
        let mut immediate = options(8);
        immediate.baseline_only = true;
        immediate.budget = Duration::ZERO;
        let result = solver
            .finite_horizon_search(&state, &weights, &[], 2, false, immediate, &|| false)
            .expect("bounded baseline search");
        assert_eq!(result.reason, FiniteSearchReason::Deadline);
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].quality, FiniteSearchQuality::Heuristic);
        assert!(result.candidates[0].guess_index < solver.guesses.len());
    }
}
