use std::{cmp::Ordering, collections::HashMap};

use anyhow::{Result, anyhow, bail};

use super::{
    ALL_GREEN_PATTERN, AnswerId, FormalObjectiveKind, FormalPolicyRuntime, PATTERN_SPACE,
    PersistedCandidateWitness, PolicyObjective, ProofCertificate, StateKey,
};

#[derive(Clone, Debug)]
struct IndependentPartition {
    answer_indices: Vec<AnswerId>,
    mass: f64,
}

#[cfg(test)]
pub(super) fn verify_certificate_witnesses(
    runtime: &FormalPolicyRuntime,
    certificate: &ProofCertificate,
) -> Result<()> {
    verify_certificate_witnesses_controlled(runtime, certificate, &|| false)
}

pub(super) fn verify_certificate_witnesses_controlled(
    runtime: &FormalPolicyRuntime,
    certificate: &ProofCertificate,
    cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    super::check_cancelled(cancelled)?;
    validate_structure(runtime, certificate)?;
    runtime.validate_metadata()?;
    validate_header(runtime, certificate)?;
    let model = &runtime.model;
    let mut keys = Vec::with_capacity(certificate.states.len());
    let mut state_ids = HashMap::with_capacity(certificate.states.len());
    for (position, state) in certificate.states.iter().enumerate() {
        if state.state_id as usize != position {
            bail!(
                "certificate state ids must be contiguous and ordered: row {} has id {}",
                position,
                state.state_id
            );
        }
        validate_answer_indices(state.state_id, &state.answer_indices, model.answers.len())?;
        let key = StateKey::from_indices_with_tokens(
            model.answers.len(),
            state.answer_indices.iter().map(|index| *index as usize),
            &model.zobrist,
        )
        .with_horizon(state.horizon);
        if state_ids.insert(key.clone(), state.state_id).is_some() {
            bail!("certificate repeats an answer state");
        }
        keys.push(key);
    }

    let root_indices = (0..model.answers.len())
        .map(|index| index as AnswerId)
        .collect::<Vec<_>>();
    let root = certificate
        .states
        .get(certificate.root_state_id as usize)
        .ok_or_else(|| anyhow!("certificate root state id is out of range"))?;
    if root.answer_indices != root_indices {
        bail!("certificate root does not contain the complete answer universe");
    }
    if keys[certificate.root_state_id as usize] != runtime.root_state {
        bail!("certificate root horizon does not match the persisted policy root");
    }
    let mut feasibility = HashMap::new();
    match model.objective_spec.kind {
        FormalObjectiveKind::Lexicographic => {
            let horizon = root
                .horizon
                .ok_or_else(|| anyhow!("lexicographic root needs a horizon"))?;
            if horizon == 0
                || root.best_objective.worst_case_depth != horizon
                || can_finish(
                    runtime,
                    &root.answer_indices,
                    horizon - 1,
                    &mut feasibility,
                    cancelled,
                )?
            {
                bail!("certificate root horizon is not minimum feasible depth");
            }
        }
        FormalObjectiveKind::ExpectedOnly if root.horizon.is_some() => {
            bail!("expected-only certificate must be unconstrained");
        }
        FormalObjectiveKind::ExpectedOnly => {}
    }

    for state in &certificate.states {
        super::check_cancelled(cancelled)?;
        validate_objective(state.state_id, "best", &state.best_objective)?;
        if state.horizon.is_some() != root.horizon.is_some()
            || state.horizon.is_some_and(|horizon| {
                horizon == 0 || state.best_objective.worst_case_depth > horizon
            })
        {
            bail!("certificate state horizon is inconsistent with objective");
        }
        if state.best_guess >= model.guesses.len() {
            bail!(
                "certificate best guess is out of range for state {}",
                state.state_id
            );
        }
        if state.candidates.len() != model.guesses.len() {
            bail!(
                "certificate state {} has {} candidate witnesses; expected {}",
                state.state_id,
                state.candidates.len(),
                model.guesses.len()
            );
        }
        let total_mass = state
            .answer_indices
            .iter()
            .map(|index| model.prior[*index as usize])
            .sum::<f64>();
        if !total_mass.is_finite() || total_mass <= 0.0 {
            bail!(
                "certificate state {} has non-positive or non-finite mass",
                state.state_id
            );
        }

        let mut covered_guesses = vec![false; model.guesses.len()];
        let mut candidate_objectives = vec![None; model.guesses.len()];
        let mut independently_best: Option<(usize, PolicyObjective)> = None;
        for (candidate_position, candidate) in state.candidates.iter().enumerate() {
            super::check_cancelled(cancelled)?;
            if candidate.guess_index >= model.guesses.len() {
                bail!(
                    "certificate candidate guess is out of range for state {}",
                    state.state_id
                );
            }
            if std::mem::replace(&mut covered_guesses[candidate.guess_index], true) {
                bail!(
                    "certificate repeats guess {} for state {}",
                    candidate.guess_index,
                    state.state_id
                );
            }
            if candidate.guess_index != candidate_position {
                bail!(
                    "certificate candidates must be ordered by guess index for state {}",
                    state.state_id
                );
            }
            let partitions =
                independent_partition(runtime, &state.answer_indices, candidate.guess_index);
            match &candidate.witness {
                PersistedCandidateWitness::Infeasible => {
                    let horizon = state.horizon.ok_or_else(|| {
                        anyhow!("unconstrained candidate cannot be horizon-infeasible")
                    })?;
                    let mut has_infeasible_child = false;
                    for (pattern, child) in partitions.iter().enumerate() {
                        if pattern != ALL_GREEN_PATTERN as usize
                            && let Some(child) = child
                            && !can_finish(
                                runtime,
                                &child.answer_indices,
                                horizon - 1,
                                &mut feasibility,
                                cancelled,
                            )?
                        {
                            has_infeasible_child = true;
                            break;
                        }
                    }
                    if !has_infeasible_child {
                        bail!("invalid infeasible witness for state {}", state.state_id);
                    }
                }
                PersistedCandidateWitness::NonProgress { pattern } => {
                    if *pattern as usize >= PATTERN_SPACE
                        || *pattern == ALL_GREEN_PATTERN
                        || partitions[*pattern as usize]
                            .as_ref()
                            .is_none_or(|partition| {
                                partition.answer_indices != state.answer_indices
                            })
                    {
                        bail!(
                            "invalid non-progress witness for state {} guess {}",
                            state.state_id,
                            candidate.guess_index
                        );
                    }
                }
                PersistedCandidateWitness::Equivalent {
                    representative_guess,
                } => {
                    if *representative_guess >= candidate.guess_index {
                        bail!(
                            "equivalent witness must reference an earlier guess for state {}",
                            state.state_id
                        );
                    }
                    let representative_partitions = independent_partition(
                        runtime,
                        &state.answer_indices,
                        *representative_guess,
                    );
                    if !same_partitions_independent(&partitions, &representative_partitions) {
                        bail!(
                            "equivalent witness partition mismatch for state {} guess {}",
                            state.state_id,
                            candidate.guess_index
                        );
                    }
                    if let Some(objective) = candidate_objectives[*representative_guess].clone() {
                        update_independent_best(
                            &mut independently_best,
                            candidate.guess_index,
                            &objective,
                            &model.guesses,
                            FormalObjectiveKind::ExpectedOnly,
                        );
                        candidate_objectives[candidate.guess_index] = Some(objective);
                    }
                }
                PersistedCandidateWitness::Exact {
                    objective,
                    children,
                } => {
                    validate_objective(state.state_id, "candidate", objective)?;
                    if partitions.iter().enumerate().any(|(pattern, partition)| {
                        pattern != ALL_GREEN_PATTERN as usize
                            && partition.as_ref().is_some_and(|partition| {
                                partition.answer_indices == state.answer_indices
                            })
                    }) {
                        bail!(
                            "exact witness is non-progressing for state {} guess {}",
                            state.state_id,
                            candidate.guess_index
                        );
                    }
                    let actual_child_count = partitions
                        .iter()
                        .enumerate()
                        .filter(|(pattern, partition)| {
                            *pattern != ALL_GREEN_PATTERN as usize && partition.is_some()
                        })
                        .count();
                    if actual_child_count != children.len() {
                        bail!(
                            "certificate child coverage mismatch for state {} guess {}: expected {}, found {}",
                            state.state_id,
                            candidate.guess_index,
                            actual_child_count,
                            children.len()
                        );
                    }
                    let mut seen_patterns = [false; PATTERN_SPACE];
                    let mut recomputed = PolicyObjective {
                        worst_case_depth: 1,
                        expected_guesses: 1.0,
                    };
                    for child in children {
                        if child.pattern as usize >= PATTERN_SPACE
                            || child.pattern == ALL_GREEN_PATTERN
                            || std::mem::replace(&mut seen_patterns[child.pattern as usize], true)
                        {
                            bail!(
                                "certificate child pattern is green or repeated for state {}",
                                state.state_id
                            );
                        }
                        let partition =
                            partitions[child.pattern as usize].as_ref().ok_or_else(|| {
                                anyhow!(
                                    "certificate pattern {} is absent for state {} guess {}",
                                    child.pattern,
                                    state.state_id,
                                    candidate.guess_index
                                )
                            })?;
                        let child_state = certificate
                            .states
                            .get(child.child_state_id as usize)
                            .ok_or_else(|| {
                                anyhow!(
                                    "certificate child id {} is out of range",
                                    child.child_state_id
                                )
                            })?;
                        if child.child_state_id >= state.state_id
                            || child_state.answer_indices != partition.answer_indices
                            || child_state.horizon != state.horizon.map(|horizon| horizon - 1)
                        {
                            bail!(
                                "certificate child state mismatch for state {} pattern {}",
                                state.state_id,
                                child.pattern
                            );
                        }
                        if !approximately_equal(child.mass, partition.mass, 1e-12) {
                            bail!(
                                "certificate child mass mismatch for state {} pattern {}",
                                state.state_id,
                                child.pattern
                            );
                        }
                        if !same_objective_independent(
                            &child.objective,
                            &child_state.best_objective,
                        ) {
                            bail!(
                                "certificate child objective mismatch for state {} pattern {}",
                                state.state_id,
                                child.pattern
                            );
                        }
                        recomputed.worst_case_depth = recomputed.worst_case_depth.max(
                            child
                                .objective
                                .worst_case_depth
                                .checked_add(1)
                                .ok_or_else(|| anyhow!("certificate depth overflows"))?,
                        );
                        recomputed.expected_guesses +=
                            (partition.mass / total_mass) * child.objective.expected_guesses;
                    }
                    if !same_objective_independent(&recomputed, objective) {
                        bail!(
                            "certificate candidate objective mismatch for state {} guess {}",
                            state.state_id,
                            candidate.guess_index
                        );
                    }
                    if state
                        .horizon
                        .is_some_and(|horizon| objective.worst_case_depth > horizon)
                    {
                        bail!("certificate candidate exceeds its horizon");
                    }
                    update_independent_best(
                        &mut independently_best,
                        candidate.guess_index,
                        objective,
                        &model.guesses,
                        FormalObjectiveKind::ExpectedOnly,
                    );
                    candidate_objectives[candidate.guess_index] = Some(objective.clone());
                }
            }
        }
        if covered_guesses.iter().any(|covered| !covered) {
            bail!(
                "certificate is missing one or more guesses for state {}",
                state.state_id
            );
        }
        let (best_guess, best_objective) = independently_best.ok_or_else(|| {
            anyhow!(
                "certificate state {} has no progressing candidate",
                state.state_id
            )
        })?;
        if best_guess != state.best_guess
            || !same_objective_independent(&best_objective, &state.best_objective)
        {
            bail!(
                "certificate winning decision mismatch for state {}",
                state.state_id
            );
        }
    }

    for (policy_state, stored) in &runtime.policy {
        let certificate_id = state_ids.get(policy_state).copied().ok_or_else(|| {
            anyhow!(
                "certificate does not cover persisted policy state {}",
                policy_state.state_hash()
            )
        })?;
        let certificate_state = &certificate.states[certificate_id as usize];
        if stored.best_guess != certificate_state.best_guess
            || !same_objective_independent(&stored.objective, &certificate_state.best_objective)
        {
            bail!(
                "certificate disagrees with persisted policy state {}",
                policy_state.state_hash()
            );
        }
    }
    Ok(())
}

fn validate_header(runtime: &FormalPolicyRuntime, certificate: &ProofCertificate) -> Result<()> {
    let manifest = &runtime.model.manifest;
    if certificate.states.len() != certificate.state_count {
        bail!(
            "proof certificate state count mismatch: header={} payload={}",
            certificate.state_count,
            certificate.states.len()
        );
    }
    if certificate.policy_state_count != runtime.policy.len()
        || runtime.ordered_states.len() != runtime.policy.len()
        || runtime.state_ids.len() != runtime.policy.len()
    {
        bail!(
            "proof certificate policy coverage mismatch: certificate={} policy={}",
            certificate.policy_state_count,
            runtime.policy.len()
        );
    }
    if certificate.model_id != manifest.model_id
        || certificate.manifest_hash != manifest.manifest_hash
        || certificate.objective_id != manifest.objective_id
        || certificate.objective_version != manifest.objective_version
        || certificate.state_format_version != manifest.state_format_version
        || certificate.aux_table_version != manifest.aux_table_version
        || certificate.certificate_format_version != manifest.certificate_format_version
        || certificate.small_state_table_hash != manifest.small_state_table_hash
    {
        bail!("proof certificate header does not match the formal manifest");
    }
    Ok(())
}

fn validate_answer_indices(state_id: u32, indices: &[AnswerId], answer_count: usize) -> Result<()> {
    if indices.is_empty() {
        bail!("certificate state {} is empty", state_id);
    }
    let mut previous = None;
    for index in indices {
        if *index as usize >= answer_count || previous.is_some_and(|value| value >= *index) {
            bail!(
                "certificate state {} has invalid or unordered answer indices",
                state_id
            );
        }
        previous = Some(*index);
    }
    Ok(())
}

fn validate_objective(state_id: u32, label: &str, objective: &PolicyObjective) -> Result<()> {
    if objective.worst_case_depth == 0
        || !objective.expected_guesses.is_finite()
        || objective.expected_guesses < 1.0
        || objective.expected_guesses > f64::from(objective.worst_case_depth) + 1e-9
    {
        bail!(
            "certificate {} objective is invalid for state {}",
            label,
            state_id
        );
    }
    Ok(())
}

// This checks identities and finite, acyclic references only. It does not establish optimality.
pub(super) fn validate_structure(
    runtime: &FormalPolicyRuntime,
    certificate: &ProofCertificate,
) -> Result<()> {
    runtime.validate_metadata()?;
    validate_header(runtime, certificate)?;
    if certificate.states.is_empty() || certificate.states.len() > super::MAX_FORMAL_STATES {
        bail!("certificate state count exceeds formal resource limits");
    }
    let model = &runtime.model;
    let mut keys = HashMap::with_capacity(certificate.states.len());
    for (position, state) in certificate.states.iter().enumerate() {
        if state.state_id as usize != position
            || state.best_guess >= model.guesses.len()
            || state.candidates.len() != model.guesses.len()
        {
            bail!("certificate state id, guess, or candidate coverage is invalid");
        }
        validate_answer_indices(state.state_id, &state.answer_indices, model.answers.len())?;
        validate_objective(state.state_id, "best", &state.best_objective)?;
        let key = StateKey::from_indices_with_tokens(
            model.answers.len(),
            state.answer_indices.iter().map(|index| *index as usize),
            &model.zobrist,
        )
        .with_horizon(state.horizon);
        super::validate_loaded_state(&key, model)?;
        super::validate_loaded_objective(&key, &state.best_objective)?;
        if keys.insert(key, state).is_some() {
            bail!("certificate repeats an answer state/horizon");
        }
        for (guess_index, candidate) in state.candidates.iter().enumerate() {
            if candidate.guess_index != guess_index {
                bail!("certificate candidate ids must be ordered and unique");
            }
            match &candidate.witness {
                PersistedCandidateWitness::Infeasible if state.horizon.is_none() => {
                    bail!("unconstrained infeasible witness")
                }
                PersistedCandidateWitness::Infeasible => {}
                PersistedCandidateWitness::NonProgress { pattern } => {
                    if *pattern as usize >= PATTERN_SPACE || *pattern == ALL_GREEN_PATTERN {
                        bail!("invalid non-progress pattern");
                    }
                }
                PersistedCandidateWitness::Equivalent {
                    representative_guess,
                } => {
                    if *representative_guess >= guess_index {
                        bail!("invalid equivalent candidate reference");
                    }
                }
                PersistedCandidateWitness::Exact {
                    objective,
                    children,
                } => {
                    super::validate_loaded_objective(
                        &StateKey::from_indices_with_tokens(
                            model.answers.len(),
                            state.answer_indices.iter().map(|index| *index as usize),
                            &model.zobrist,
                        )
                        .with_horizon(state.horizon),
                        objective,
                    )?;
                    if children.len() >= PATTERN_SPACE {
                        bail!("too many certificate children");
                    }
                    let mut seen = [false; PATTERN_SPACE];
                    for child in children {
                        if child.pattern as usize >= PATTERN_SPACE
                            || child.pattern == ALL_GREEN_PATTERN
                            || std::mem::replace(&mut seen[child.pattern as usize], true)
                            || child.child_state_id >= state.state_id
                            || !child.mass.is_finite()
                            || child.mass <= 0.0
                            || child.mass > 1.0 + 1e-9
                        {
                            bail!("invalid certificate child reference, pattern, or mass");
                        }
                        let target = certificate
                            .states
                            .get(child.child_state_id as usize)
                            .ok_or_else(|| anyhow!("certificate child id out of range"))?;
                        if target.answer_indices.len() >= state.answer_indices.len()
                            || target.horizon
                                != state.horizon.and_then(|horizon| horizon.checked_sub(1))
                            || !same_objective_independent(&child.objective, &target.best_objective)
                        {
                            bail!("certificate child is not a matching, strictly smaller state");
                        }
                        validate_objective(state.state_id, "child", &child.objective)?;
                    }
                }
            }
        }
    }
    let root = certificate
        .states
        .get(certificate.root_state_id as usize)
        .ok_or_else(|| anyhow!("certificate root id out of range"))?;
    if keys
        .get(&runtime.root_state)
        .is_none_or(|state| state.state_id != root.state_id)
    {
        bail!("certificate root does not match the persisted root");
    }
    for (key, decision) in &runtime.policy {
        let state = keys
            .get(key)
            .ok_or_else(|| anyhow!("certificate omits a policy state"))?;
        if state.best_guess != decision.best_guess
            || !same_objective_independent(&state.best_objective, &decision.objective)
        {
            bail!("certificate disagrees with persisted decision");
        }
    }
    Ok(())
}

fn independent_partition(
    runtime: &FormalPolicyRuntime,
    state: &[AnswerId],
    guess_index: usize,
) -> [Option<IndependentPartition>; PATTERN_SPACE] {
    let mut partitions: [Option<IndependentPartition>; PATTERN_SPACE] =
        std::array::from_fn(|_| None);
    for answer_index in state {
        let pattern = runtime
            .model
            .pattern_table
            .get(guess_index, *answer_index as usize) as usize;
        let partition = partitions[pattern].get_or_insert_with(|| IndependentPartition {
            answer_indices: Vec::new(),
            mass: 0.0,
        });
        partition.answer_indices.push(*answer_index);
        partition.mass += runtime.model.prior[*answer_index as usize];
    }
    partitions
}

// Feasibility is a Boolean AND/OR search, independent of cost minimization and
// optimizer pruning. It proves rejected horizons without trusting cached values.
fn can_finish(
    runtime: &FormalPolicyRuntime,
    state: &[AnswerId],
    turns: u8,
    memo: &mut HashMap<(Vec<AnswerId>, u8), bool>,
    cancelled: &dyn Fn() -> bool,
) -> Result<bool> {
    super::check_cancelled(cancelled)?;
    if turns == 0 {
        return Ok(false);
    }
    let key = (state.to_vec(), turns);
    if let Some(feasible) = memo.get(&key) {
        return Ok(*feasible);
    }
    let mut feasible = false;
    for guess in 0..runtime.model.guesses.len() {
        super::check_cancelled(cancelled)?;
        let mut all_finish = true;
        for (pattern, partition) in independent_partition(runtime, state, guess)
            .iter()
            .enumerate()
        {
            if pattern != ALL_GREEN_PATTERN as usize
                && let Some(child) = partition
                && (child.answer_indices.len() >= state.len()
                    || !can_finish(runtime, &child.answer_indices, turns - 1, memo, cancelled)?)
            {
                all_finish = false;
                break;
            }
        }
        if all_finish {
            feasible = true;
            break;
        }
    }
    memo.insert(key, feasible);
    Ok(feasible)
}

fn same_partitions_independent(
    left: &[Option<IndependentPartition>; PATTERN_SPACE],
    right: &[Option<IndependentPartition>; PATTERN_SPACE],
) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| match (left, right) {
            (Some(left), Some(right)) => left.answer_indices == right.answer_indices,
            (None, None) => true,
            _ => false,
        })
}

fn update_independent_best(
    best: &mut Option<(usize, PolicyObjective)>,
    guess: usize,
    objective: &PolicyObjective,
    guesses: &[String],
    kind: FormalObjectiveKind,
) {
    let replace = best.as_ref().is_none_or(|(best_guess, best_objective)| {
        compare_decisions_independent(guess, objective, *best_guess, best_objective, guesses, kind)
            .is_lt()
    });
    if replace {
        *best = Some((guess, objective.clone()));
    }
}

fn compare_decisions_independent(
    left_guess: usize,
    left: &PolicyObjective,
    right_guess: usize,
    right: &PolicyObjective,
    guesses: &[String],
    kind: FormalObjectiveKind,
) -> Ordering {
    compare_objectives_independent(left, right, kind)
        .then_with(|| guesses[left_guess].cmp(&guesses[right_guess]))
}

fn compare_objectives_independent(
    left: &PolicyObjective,
    right: &PolicyObjective,
    kind: FormalObjectiveKind,
) -> Ordering {
    match kind {
        FormalObjectiveKind::Lexicographic => left
            .worst_case_depth
            .cmp(&right.worst_case_depth)
            .then_with(|| left.expected_guesses.total_cmp(&right.expected_guesses)),
        FormalObjectiveKind::ExpectedOnly => left
            .expected_guesses
            .total_cmp(&right.expected_guesses)
            .then_with(|| left.worst_case_depth.cmp(&right.worst_case_depth)),
    }
}

fn same_objective_independent(left: &PolicyObjective, right: &PolicyObjective) -> bool {
    left.worst_case_depth == right.worst_case_depth
        && approximately_equal(left.expected_guesses, right.expected_guesses, 1e-9)
}

fn approximately_equal(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}
