use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    atomic_file::atomic_write,
    data::{ProjectPaths, read_word_list_from, validate_answer_universe},
    identity::{CanonicalSha256, digest_bytes_tagged, tag},
    model::AnswerRecord,
    pattern_table::{PatternTable, hash_word_list},
    scoring::{ALL_GREEN_PATTERN, PATTERN_SPACE, format_feedback_letters, parse_feedback},
    small_state::{SMALL_STATE_TABLE_VERSION, SmallStateTable},
};

#[cfg(test)]
mod artifact_tests;
mod generation;
#[cfg(test)]
mod horizon_tests;
mod parsing;
#[cfg(test)]
mod runtime_tests;
mod scale;
mod verifier;

pub use scale::{
    FormalScalePoint, FormalScaleProjection, FormalScaleReport, FormalScaleRequest,
    benchmark_formal_scale,
};

pub const DEFAULT_FORMAL_MODEL_ID: &str = "formal-v1";
pub const DEFAULT_EXPECTED_ONLY_MODEL_ID: &str = "formal-expected-v1";
pub const DEFAULT_FORMAL_ALTERNATIVE_PARTITIONS: usize = 10_000;
const PRIOR_SPEC_NAME: &str = "prior.toml";
const MANIFEST_NAME: &str = "manifest.json";
const VALUES_NAME: &str = "state_values.bin";
const POLICY_NAME: &str = "policy_table.bin";
const METADATA_NAME: &str = "proof_metadata.json";
const CERTIFICATE_NAME: &str = "proof_certificate.json";
const SMALL_STATE_TABLE_NAME: &str = "small_state_table.json";
const FORMAL_PATTERN_TABLE_NAME: &str = "pattern_table.bin";
const POLICY_MAGIC: &[u8; 8] = b"MWORDPV3";
const VALUES_MAGIC: &[u8; 8] = b"MWORDVV3";
const TAGGED_DIGEST_LENGTH: usize = 74;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const OBJECTIVE_VERSION: u32 = 3;
const STATE_FORMAT_VERSION: u32 = 3;
const AUX_TABLE_VERSION: u32 = 4;
const CERTIFICATE_FORMAT_VERSION: u32 = 8;
const SMALL_STATE_LIMIT: usize = 12;
const MAX_FORMAL_STATES: usize = 100_000;
const MAX_FORMAL_BINARY_BYTES: u64 = 256 * 1024 * 1024;
const BINARY_HEADER_BYTES: u64 = 8 + TAGGED_DIGEST_LENGTH as u64 + 8 + 4;

type AnswerId = u16;
const INLINE_STATE_THRESHOLD: usize = 30;
const LOCAL_REINDEX_THRESHOLD: usize = 256;
const STATE_INLINE_CAPACITY: usize = INLINE_STATE_THRESHOLD;
const HOT_TT_BYTES: usize = 128 * 1024 * 1024;
const HOT_TT_ASSOCIATIVITY: usize = 4;
const STATE_TAG_INLINE: u8 = 0;
const STATE_TAG_BITSET: u8 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FormalObjectiveKind {
    Lexicographic,
    ExpectedOnly,
}

#[derive(Clone, Copy, Debug)]
struct FormalObjectiveSpec {
    id: &'static str,
    kind: FormalObjectiveKind,
    version: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FormalVerificationMode {
    Certificate,
    Oracle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCertificateState {
    state_id: u32,
    #[serde(deserialize_with = "parsing::bounded_vec::<_, _, { parsing::MAX_WORDS }>")]
    answer_indices: Vec<AnswerId>,
    horizon: Option<u8>,
    best_guess: usize,
    best_objective: PolicyObjective,
    #[serde(deserialize_with = "parsing::bounded_vec::<_, _, { parsing::MAX_WORDS }>")]
    candidates: Vec<PersistedCertificateCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCertificateCandidate {
    guess_index: usize,
    witness: PersistedCandidateWitness,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistedCandidateWitness {
    Infeasible,
    NonProgress {
        pattern: u8,
    },
    Equivalent {
        representative_guess: usize,
    },
    Exact {
        objective: PolicyObjective,
        #[serde(deserialize_with = "parsing::bounded_vec::<_, _, PATTERN_SPACE>")]
        children: Vec<PersistedCertificateChild>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCertificateChild {
    pattern: u8,
    child_state_id: u32,
    objective: PolicyObjective,
    mass: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormalManifest {
    pub model_id: String,
    pub objective_kind: FormalObjectiveKind,
    pub objective_id: String,
    pub objective: String,
    pub objective_version: u32,
    pub normal_mode_only: bool,
    pub guess_count: usize,
    pub answer_count: usize,
    pub guess_hash: String,
    pub answer_hash: String,
    pub prior_hash: String,
    pub state_format_version: u32,
    pub aux_table_version: u32,
    pub certificate_format_version: u32,
    pub small_state_table_version: u32,
    pub small_state_table_hash: String,
    pub manifest_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyObjective {
    pub worst_case_depth: u8,
    pub expected_guesses: f64,
}

#[derive(Clone, Debug)]
pub struct FormalSuggestion {
    pub word: String,
    pub objective: PolicyObjective,
    pub bucket_sizes: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    pub model_id: String,
    pub manifest_hash: String,
    pub solved_states: usize,
    pub deduped_signatures: u64,
    pub bound_hits: u64,
    pub root_refinement_pruned: u64,
    pub local_refinement_pruned: u64,
    pub solve_millis: u128,
    pub certificate_millis: u128,
    /// Elapsed time sampled before writing metadata and publishing the generation pointer.
    pub pre_publish_millis: u128,
    pub root_objective: PolicyObjective,
}

#[derive(Clone, Debug)]
pub struct BuildOptimalSummary {
    pub model_id: String,
    pub manifest_hash: String,
    pub solved_states: usize,
    pub deduped_signatures: u64,
    pub bound_hits: u64,
    pub root_refinement_pruned: u64,
    pub local_refinement_pruned: u64,
    pub build_millis: u128,
    pub solve_millis: u128,
    pub certificate_millis: u128,
    pub persistence_millis: u128,
    pub root_best_guess: String,
    pub root_objective: PolicyObjective,
}

#[derive(Clone, Debug)]
pub struct VerifySummary {
    pub mode: FormalVerificationMode,
    pub certificate_format_version: u32,
    pub certificate_state_count: usize,
    pub verified_cached_states: usize,
    pub verified_small_states: usize,
    pub verified_medium_states: usize,
    pub model_id: String,
    pub manifest_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofCertificate {
    pub model_id: String,
    pub manifest_hash: String,
    pub objective_id: String,
    pub objective_version: u32,
    pub state_format_version: u32,
    pub aux_table_version: u32,
    pub certificate_format_version: u32,
    pub small_state_table_hash: String,
    pub policy_state_count: usize,
    pub state_count: usize,
    pub root_state_id: u32,
    #[serde(deserialize_with = "parsing::bounded_vec::<_, _, MAX_FORMAL_STATES>")]
    states: Vec<PersistedCertificateState>,
}

#[derive(Clone, Debug)]
pub struct FormalStateExplanation {
    pub model_id: String,
    pub manifest_hash: String,
    pub surviving_answers: usize,
    pub best_guess: String,
    pub objective: PolicyObjective,
    pub bucket_sizes: Vec<usize>,
    pub tied_moves: Vec<FormalSuggestion>,
    pub alternatives_status: FormalAlternativesStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FormalAlternativesStatus {
    NotRequested,
    Complete,
    WorkLimitReached,
}

#[derive(Debug)]
enum FormalSearchStop {
    Cancelled,
    WorkLimit,
}

impl std::fmt::Display for FormalSearchStop {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Cancelled => "formal search cancelled",
            Self::WorkLimit => "formal search partition budget exhausted",
        })
    }
}

impl std::error::Error for FormalSearchStop {}

#[derive(Clone, Debug)]
pub struct PolicyArtifactSet {
    pub model_dir: PathBuf,
    pub prior_spec: PathBuf,
    pub manifest: PathBuf,
    pub values: PathBuf,
    pub policy: PathBuf,
    pub metadata: PathBuf,
    pub certificate: PathBuf,
    pub small_state_table: PathBuf,
    pub pattern_table: PathBuf,
}

impl PolicyArtifactSet {
    pub fn for_model(paths: &ProjectPaths, model_id: &str) -> Self {
        let model_dir = paths.root.join("data/formal").join(model_id);
        Self::in_directory(model_dir)
    }

    fn in_directory(model_dir: PathBuf) -> Self {
        Self {
            prior_spec: model_dir.join(PRIOR_SPEC_NAME),
            manifest: model_dir.join(MANIFEST_NAME),
            values: model_dir.join(VALUES_NAME),
            policy: model_dir.join(POLICY_NAME),
            metadata: model_dir.join(METADATA_NAME),
            certificate: model_dir.join(CERTIFICATE_NAME),
            small_state_table: model_dir.join(SMALL_STATE_TABLE_NAME),
            pattern_table: model_dir.join(FORMAL_PATTERN_TABLE_NAME),
            model_dir,
        }
    }

    /// Resolve and validate one immutable generation. Old flat artifacts require rebuilding.
    pub fn published(paths: &ProjectPaths, model_id: &str) -> Result<Self> {
        generation::resolve(paths, model_id)
    }

    pub fn exists(&self) -> bool {
        self.manifest.exists()
            && self.values.exists()
            && self.policy.exists()
            && self.metadata.exists()
            && self.certificate.exists()
            && self.small_state_table.exists()
            && self.pattern_table.exists()
            && self.prior_spec.exists()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FormalPriorSpec {
    Uniform,
    Explicit { weights: HashMap<String, f64> },
}

#[derive(Deserialize)]
struct FormalConfig {
    objective: Option<FormalObjectiveKind>,
    #[serde(flatten)]
    prior: FormalPriorSpec,
}

#[derive(Clone, Debug)]
pub struct FormalModel {
    pub manifest: FormalManifest,
    pub guesses: Vec<String>,
    pub answers: Vec<String>,
    pub prior: Vec<f64>,
    zobrist: Vec<u64>,
    small_state_table: SmallStateTable,
    objective_spec: FormalObjectiveSpec,
    pattern_table: PatternTable,
    guess_index: HashMap<String, usize>,
    raw_prior: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct FormalPolicyRuntime {
    model: FormalModel,
    policy: HashMap<StateKey, StoredState>,
    root_state: StateKey,
    ordered_states: Vec<StateKey>,
    state_ids: HashMap<StateKey, u32>,
    metadata: ProofMetadata,
    artifacts: PolicyArtifactSet,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateKey {
    storage: StateStorage,
    count: usize,
    hash: u64,
    // None is unconstrained expected cost; Some(d) is E(S, d).
    horizon: Option<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum StateStorage {
    Inline {
        len: u8,
        indices: [AnswerId; STATE_INLINE_CAPACITY],
    },
    Bitset(Box<[u64]>),
}

impl Hash for StateKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
        state.write_usize(self.count);
        self.horizon.hash(state);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PartitionFingerprint {
    bucket_count: u16,
    mix_a: u64,
    mix_b: u64,
}

#[derive(Clone, Debug)]
struct GuessEvaluation {
    guess_index: usize,
    objective: PolicyObjective,
    bucket_sizes: Vec<usize>,
}

#[derive(Clone, Debug)]
struct GuessQuickPlan {
    guess_index: usize,
    lower_bound: u8,
    max_bucket: usize,
    entropy: f64,
    solve_mass: f64,
    buckets: Vec<PartitionBucket>,
}

#[derive(Clone, Debug)]
struct StoredState {
    objective: PolicyObjective,
    best_guess: usize,
}

#[derive(Clone)]
struct FormalPolicyBuilder<'a> {
    model: FormalModel,
    memo: HashMap<StateKey, StoredState>,
    hot_tt: HotTranspositionTable,
    deduped_signatures: u64,
    bound_hits: u64,
    root_refinement_pruned: u64,
    local_refinement_pruned: u64,
    partition_calls: u64,
    quick_plan_calls: u64,
    started: Instant,
    last_progress: Instant,
    cancelled: Option<&'a (dyn Fn() -> bool + Sync)>,
}

#[derive(Clone, Debug)]
struct PartitionBucket {
    pattern: u8,
    state: StateKey,
    mass: f64,
    count: usize,
    entropy_bits: f64,
}

#[derive(Clone, Debug)]
struct PartitionScratch {
    words: Vec<u64>,
    masses: [f64; PATTERN_SPACE],
    counts: [usize; PATTERN_SPACE],
    weighted_log_sums: [f64; PATTERN_SPACE],
    offsets: [usize; PATTERN_SPACE + 1],
    positions: Vec<u16>,
}

impl Default for PartitionScratch {
    fn default() -> Self {
        Self {
            words: Vec::new(),
            masses: [0.0; PATTERN_SPACE],
            counts: [0; PATTERN_SPACE],
            weighted_log_sums: [0.0; PATTERN_SPACE],
            offsets: [0; PATTERN_SPACE + 1],
            positions: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct StateFrame {
    global_ids: Box<[AnswerId]>,
}

#[derive(Clone, Debug)]
struct HotTranspositionTable {
    sets: Vec<[Option<HotTtEntry>; HOT_TT_ASSOCIATIVITY]>,
    victims: Vec<Option<HotTtEntry>>,
    generation: u64,
}

#[derive(Clone, Debug)]
struct HotTtEntry {
    state: StateKey,
    stored: StoredState,
    generation: u64,
}

#[derive(Clone)]
struct IndependentExactSolver<'a> {
    model: &'a FormalModel,
    local_memo: HashMap<StateKey, StoredState>,
    infeasible: HashSet<StateKey>,
    scratch: PartitionScratch,
    cancelled: Option<&'a dyn Fn() -> bool>,
    remaining_partitions: Option<usize>,
}

impl HotTranspositionTable {
    fn new(bytes: usize) -> Self {
        let entry_bytes = std::mem::size_of::<HotTtEntry>().max(1);
        let target_entries = (bytes / entry_bytes).max(HOT_TT_ASSOCIATIVITY);
        let set_count = (target_entries / HOT_TT_ASSOCIATIVITY).max(1);
        Self {
            sets: (0..set_count)
                .map(|_| std::array::from_fn(|_| None))
                .collect(),
            victims: vec![None; set_count],
            generation: 0,
        }
    }

    fn get(&mut self, state: &StateKey) -> Option<StoredState> {
        let set_index = self.set_index(state.state_hash());
        if let Some(entry) = self.sets[set_index]
            .iter_mut()
            .flatten()
            .find(|entry| entry.state == *state)
        {
            self.generation += 1;
            entry.generation = self.generation;
            return Some(entry.stored.clone());
        }
        if let Some(entry) = self.victims[set_index]
            .as_mut()
            .filter(|entry| entry.state == *state)
        {
            self.generation += 1;
            entry.generation = self.generation;
            return Some(entry.stored.clone());
        }
        None
    }

    fn insert(&mut self, state: StateKey, stored: StoredState) {
        let set_index = self.set_index(state.state_hash());
        self.generation += 1;
        let new_entry = HotTtEntry {
            state,
            stored,
            generation: self.generation,
        };
        if let Some(slot) = self.sets[set_index].iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(new_entry);
            return;
        }
        let replace_index = self.sets[set_index]
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                compare_hot_tt_entry(left.as_ref().unwrap(), right.as_ref().unwrap())
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let evicted = self.sets[set_index][replace_index].replace(new_entry);
        if let Some(entry) = evicted {
            self.victims[set_index] = Some(entry);
        }
    }

    fn set_index(&self, hash: u64) -> usize {
        (hash as usize) % self.sets.len()
    }
}

fn compare_hot_tt_entry(left: &HotTtEntry, right: &HotTtEntry) -> std::cmp::Ordering {
    left.stored
        .objective
        .worst_case_depth
        .cmp(&right.stored.objective.worst_case_depth)
        .then_with(|| left.state.count().cmp(&right.state.count()))
        .then_with(|| left.generation.cmp(&right.generation))
}

impl<'a> IndependentExactSolver<'a> {
    fn new(model: &'a FormalModel) -> Self {
        Self {
            model,
            local_memo: HashMap::new(),
            infeasible: HashSet::new(),
            scratch: PartitionScratch::default(),
            cancelled: None,
            remaining_partitions: None,
        }
    }

    fn check_cancelled(&self) -> Result<()> {
        if self.cancelled.is_some_and(|cancelled| cancelled()) {
            return Err(FormalSearchStop::Cancelled.into());
        }
        Ok(())
    }

    fn consume_partition(&mut self) -> Result<()> {
        self.check_cancelled()?;
        if let Some(remaining) = &mut self.remaining_partitions {
            if *remaining == 0 {
                return Err(FormalSearchStop::WorkLimit.into());
            }
            *remaining -= 1;
        }
        Ok(())
    }

    fn solve(&mut self, state: &StateKey) -> Result<StoredState> {
        if state.horizon.is_none()
            && self.model.objective_spec.kind == FormalObjectiveKind::Lexicographic
        {
            for horizon in 1..=u8::MAX {
                if let Some(best) = self.solve_conditioned(&state.with_horizon(Some(horizon)))? {
                    return Ok(best);
                }
            }
            bail!("state exceeds the representable formal depth");
        }
        self.solve_conditioned(state)?
            .ok_or_else(|| anyhow!("state has no policy within its remaining horizon"))
    }

    fn solve_conditioned(&mut self, state: &StateKey) -> Result<Option<StoredState>> {
        #[cfg(test)]
        runtime_tests::SOLVE_CALLS.with(|calls| calls.set(calls.get() + 1));
        self.check_cancelled()?;
        if state.horizon == Some(0) || self.infeasible.contains(state) {
            return Ok(None);
        }
        if let Some(existing) = self.local_memo.get(state) {
            return Ok(Some(existing.clone()));
        }
        if state.count() == 1 {
            let stored = singleton_state_for_model(self.model, state)?;
            self.local_memo.insert(state.clone(), stored.clone());
            return Ok(Some(stored));
        }

        let state_indices = state.indices();
        let total_mass = state_indices
            .iter()
            .map(|index| self.model.prior[*index])
            .sum::<f64>();
        let mut best: Option<StoredState> = None;
        for guess_index in 0..self.model.guesses.len() {
            self.consume_partition()?;
            let buckets = partition_guess_with_scratch(
                self.model.answers.len(),
                state,
                guess_index,
                &self.model.pattern_table,
                &self.model.prior,
                &self.model.zobrist,
                &mut self.scratch,
            )?;
            if buckets
                .iter()
                .any(|bucket| bucket.pattern != ALL_GREEN_PATTERN && bucket.count == state.count())
            {
                continue;
            }
            let mut worst_case = 1u8;
            let mut expected = 1.0;
            let mut feasible = true;
            for bucket in buckets {
                if bucket.pattern == ALL_GREEN_PATTERN {
                    continue;
                }
                let child_state = bucket.state.with_horizon(state.child_horizon());
                let Some(child) = self.solve_conditioned(&child_state)? else {
                    feasible = false;
                    break;
                };
                worst_case = worst_case.max(
                    child
                        .objective
                        .worst_case_depth
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("formal depth exceeds u8 representation"))?,
                );
                expected += (bucket.mass / total_mass) * child.objective.expected_guesses;
            }
            if !feasible {
                continue;
            }
            let candidate = StoredState {
                objective: PolicyObjective {
                    worst_case_depth: worst_case,
                    expected_guesses: expected,
                },
                best_guess: guess_index,
            };
            if best.as_ref().is_none_or(|current| {
                compare_stored_with_kind(
                    &candidate,
                    current,
                    &self.model.guesses,
                    FormalObjectiveKind::ExpectedOnly,
                )
                .is_lt()
            }) {
                best = Some(candidate);
            }
        }
        if let Some(stored) = &best {
            self.local_memo.insert(state.clone(), stored.clone());
        } else {
            self.infeasible.insert(state.clone());
        }
        Ok(best)
    }
}

pub fn build_optimal_policy(paths: &ProjectPaths, model_id: &str) -> Result<BuildOptimalSummary> {
    build_optimal_policy_controlled(paths, model_id, &|| false)
}

fn check_cancelled(cancelled: &dyn Fn() -> bool) -> Result<()> {
    if cancelled() {
        return Err(FormalSearchStop::Cancelled.into());
    }
    Ok(())
}

fn build_optimal_policy_controlled(
    paths: &ProjectPaths,
    model_id: &str,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<BuildOptimalSummary> {
    let total_started = Instant::now();
    check_cancelled(cancelled)?;
    let model = FormalModel::load(paths, model_id)?;
    check_cancelled(cancelled)?;
    let root = StateKey::full(model.answers.len(), &model.zobrist);
    let started = Instant::now();
    let mut builder = FormalPolicyBuilder {
        model,
        memo: HashMap::new(),
        hot_tt: HotTranspositionTable::new(HOT_TT_BYTES),
        deduped_signatures: 0,
        bound_hits: 0,
        root_refinement_pruned: 0,
        local_refinement_pruned: 0,
        partition_calls: 0,
        quick_plan_calls: 0,
        started,
        last_progress: started,
        cancelled: Some(cancelled),
    };
    let root_decision = builder.solve_state(&root)?;
    let root = root.with_horizon(match builder.model.objective_spec.kind {
        FormalObjectiveKind::Lexicographic => Some(root_decision.objective.worst_case_depth),
        FormalObjectiveKind::ExpectedOnly => None,
    });
    builder.force_report_progress("root_complete");
    builder.materialize_policy_reachable_states(&root)?;
    builder.force_report_progress("policy_materialized");
    let root_state = builder
        .memo
        .get(&root)
        .cloned()
        .ok_or_else(|| anyhow!("root state missing after materialization"))?;
    let solve_millis = started.elapsed().as_millis();
    let certificate_started = Instant::now();
    let certificate = build_exhaustive_proof_certificate(&builder.model, &builder.memo, cancelled)?;
    let certificate_millis = certificate_started.elapsed().as_millis();
    let metadata = ProofMetadata {
        model_id: builder.model.manifest.model_id.clone(),
        manifest_hash: builder.model.manifest.manifest_hash.clone(),
        solved_states: builder.memo.len(),
        deduped_signatures: builder.deduped_signatures,
        bound_hits: builder.bound_hits,
        root_refinement_pruned: builder.root_refinement_pruned,
        local_refinement_pruned: builder.local_refinement_pruned,
        solve_millis,
        certificate_millis,
        pre_publish_millis: total_started.elapsed().as_millis(),
        root_objective: root_state.objective.clone(),
    };
    let persistence_started = Instant::now();
    generation::publish(
        &builder.model,
        &builder.memo,
        metadata,
        &certificate,
        paths,
        total_started,
        cancelled,
    )?;
    let persistence_millis = persistence_started.elapsed().as_millis();
    Ok(BuildOptimalSummary {
        model_id: builder.model.manifest.model_id.clone(),
        manifest_hash: builder.model.manifest.manifest_hash.clone(),
        solved_states: builder.memo.len(),
        deduped_signatures: builder.deduped_signatures,
        bound_hits: builder.bound_hits,
        root_refinement_pruned: builder.root_refinement_pruned,
        local_refinement_pruned: builder.local_refinement_pruned,
        build_millis: total_started.elapsed().as_millis(),
        solve_millis,
        certificate_millis,
        persistence_millis,
        root_best_guess: builder.model.guesses[root_state.best_guess].clone(),
        root_objective: root_state.objective,
    })
}

pub fn verify_optimal_policy(paths: &ProjectPaths, model_id: &str) -> Result<VerifySummary> {
    verify_optimal_policy_with_mode(paths, model_id, FormalVerificationMode::Certificate)
}

pub fn verify_optimal_policy_with_mode(
    paths: &ProjectPaths,
    model_id: &str,
    mode: FormalVerificationMode,
) -> Result<VerifySummary> {
    verify_optimal_policy_controlled(paths, model_id, mode, &|| false)
}

fn verify_optimal_policy_controlled(
    paths: &ProjectPaths,
    model_id: &str,
    mode: FormalVerificationMode,
    cancelled: &dyn Fn() -> bool,
) -> Result<VerifySummary> {
    check_cancelled(cancelled)?;
    let runtime = FormalPolicyRuntime::load(paths, model_id)?;
    check_cancelled(cancelled)?;
    let certificate: ProofCertificate = parsing::read_json(
        &runtime.artifacts.certificate,
        parsing::MAX_CERTIFICATE_BYTES,
    )?;
    if certificate.manifest_hash != runtime.model.manifest.manifest_hash {
        bail!(
            "proof certificate is stale for {}: expected manifest {}, found {}",
            model_id,
            runtime.model.manifest.manifest_hash,
            certificate.manifest_hash
        );
    }
    if certificate.certificate_format_version != CERTIFICATE_FORMAT_VERSION {
        bail!(
            "proof certificate format mismatch for {}: expected {}, found {}",
            model_id,
            CERTIFICATE_FORMAT_VERSION,
            certificate.certificate_format_version
        );
    }
    if certificate.small_state_table_hash != runtime.model.manifest.small_state_table_hash {
        bail!(
            "proof certificate small-state hash mismatch for {}",
            model_id
        );
    }
    let mut cached_states = 0usize;
    if mode == FormalVerificationMode::Certificate {
        verifier::verify_certificate_witnesses_controlled(&runtime, &certificate, cancelled)?;
        cached_states = runtime.policy.len();
    }
    let mut small_states = 0usize;
    let mut medium_states = 0usize;
    if mode == FormalVerificationMode::Oracle {
        for (state, stored) in &runtime.policy {
            check_cancelled(cancelled)?;
            let cached = runtime.evaluate_state_exact(state)?;
            if !same_decision(&cached, stored) {
                bail!(
                    "cached verification failed for state {}",
                    state.state_hash()
                );
            }
            cached_states += 1;
            let size = state.count();
            if size <= 6 {
                let independent = runtime.solve_state_independent(state)?;
                if !same_decision(&independent, stored) {
                    bail!(
                        "independent verification failed for size {} state {}",
                        size,
                        state.state_hash()
                    );
                }
                small_states += 1;
            } else if size <= 10 && medium_states < 12 {
                let exact = runtime.evaluate_state_exact(state)?;
                if !same_decision(&exact, stored) {
                    bail!(
                        "medium-state verification failed for size {} state {}",
                        size,
                        state.state_hash()
                    );
                }
                medium_states += 1;
            }
        }
    }

    Ok(VerifySummary {
        mode,
        certificate_format_version: certificate.certificate_format_version,
        certificate_state_count: certificate.state_count,
        verified_cached_states: cached_states,
        verified_small_states: small_states,
        verified_medium_states: medium_states,
        model_id: runtime.model.manifest.model_id.clone(),
        manifest_hash: runtime.model.manifest.manifest_hash,
    })
}

pub fn artifacts_exist(paths: &ProjectPaths, model_id: &str) -> bool {
    if parsing::validate_component(model_id).is_err() {
        return false;
    }
    let artifacts = PolicyArtifactSet::for_model(paths, model_id);
    let present = |path: &Path| match fs::symlink_metadata(path) {
        Ok(_) => true,
        Err(error) => error.kind() != std::io::ErrorKind::NotFound,
    };
    present(&artifacts.model_dir.join("current.json"))
        || [
            &artifacts.manifest,
            &artifacts.values,
            &artifacts.policy,
            &artifacts.metadata,
            &artifacts.certificate,
            &artifacts.small_state_table,
        ]
        .iter()
        .any(|path| present(path))
}

fn objective_spec(kind: FormalObjectiveKind) -> FormalObjectiveSpec {
    if kind == FormalObjectiveKind::ExpectedOnly {
        FormalObjectiveSpec {
            id: "expected_guesses_only",
            kind: FormalObjectiveKind::ExpectedOnly,
            version: OBJECTIVE_VERSION,
        }
    } else {
        FormalObjectiveSpec {
            id: "worst_case_depth_then_expected_guesses",
            kind: FormalObjectiveKind::Lexicographic,
            version: OBJECTIVE_VERSION,
        }
    }
}

impl FormalModel {
    pub fn load(paths: &ProjectPaths, model_id: &str) -> Result<Self> {
        let artifacts = PolicyArtifactSet::for_model(paths, model_id);
        Self::load_artifacts(paths, model_id, &artifacts, false)
    }

    fn load_artifacts(
        paths: &ProjectPaths,
        model_id: &str,
        artifacts: &PolicyArtifactSet,
        immutable: bool,
    ) -> Result<Self> {
        parsing::validate_component(model_id)?;
        let guesses = read_word_list_from(
            std::io::Cursor::new(parsing::read_bounded(
                &paths.seed_guesses,
                parsing::MAX_INPUT_BYTES,
            )?),
            &paths.seed_guesses,
        )
        .with_context(|| format!("failed to load {}", paths.seed_guesses.display()))?;
        let answers = read_word_list_from(
            std::io::Cursor::new(parsing::read_bounded(
                &paths.seed_answers,
                parsing::MAX_INPUT_BYTES,
            )?),
            &paths.seed_answers,
        )
        .with_context(|| format!("failed to load {}", paths.seed_answers.display()))?;
        validate_answer_universe(&guesses, answers.iter().map(String::as_str))
            .context("invalid formal answer universe")?;
        if guesses.len() > parsing::MAX_WORDS
            || answers.len() > parsing::MAX_WORDS
            || guesses
                .len()
                .checked_mul(answers.len())
                .is_none_or(|cells| cells > parsing::MAX_PATTERN_CELLS)
        {
            bail!("formal model exceeds word-id or pattern-cell resource limits");
        }
        let raw_prior = parsing::read_bounded(&artifacts.prior_spec, parsing::MAX_INPUT_BYTES)?;
        if immutable
            && raw_prior
                != parsing::read_bounded(
                    &PolicyArtifactSet::for_model(paths, model_id).prior_spec,
                    parsing::MAX_INPUT_BYTES,
                )?
        {
            bail!(
                "published formal prior differs from current prior.toml; rebuild formal artifacts"
            );
        }
        let config: FormalConfig = toml::from_str(
            std::str::from_utf8(&raw_prior).context("formal prior spec must be valid UTF-8")?,
        )
        .with_context(|| format!("failed to parse {}", artifacts.prior_spec.display()))?;
        let guess_hash = tag(&hash_word_list(guesses.iter().map(String::as_str)));
        let answer_hash = tag(&hash_word_list(answers.iter().map(String::as_str)));
        let prior_hash = digest_bytes_tagged("maybe-wordle-formal-prior-v2", &raw_prior);
        // Only the two historical built-ins have an unambiguous migration.
        let kind = match config.objective {
            Some(kind) => kind,
            None => match model_id {
                DEFAULT_FORMAL_MODEL_ID => FormalObjectiveKind::Lexicographic,
                DEFAULT_EXPECTED_ONLY_MODEL_ID => FormalObjectiveKind::ExpectedOnly,
                _ => bail!(
                    "custom formal model {model_id} requires an explicit objective in prior.toml"
                ),
            },
        };
        let objective_spec = objective_spec(kind);
        let canonical_small_state_table = SmallStateTable::build(SMALL_STATE_LIMIT);
        if immutable {
            let persisted = parsing::read_small_table(&artifacts.small_state_table)?;
            validate_small_state_table(&persisted, &canonical_small_state_table)?;
        }
        let small_state_table = canonical_small_state_table;
        let small_state_table_hash = hash_small_state_table(&small_state_table);
        let manifest_hash = combine_hashes(
            &guess_hash,
            &answer_hash,
            &prior_hash,
            objective_spec,
            &small_state_table_hash,
        );
        let manifest = FormalManifest {
            model_id: model_id.to_string(),
            objective_kind: kind,
            objective_id: objective_spec.id.to_string(),
            objective: objective_spec.id.to_string(),
            objective_version: objective_spec.version,
            normal_mode_only: objective_spec.kind == FormalObjectiveKind::Lexicographic,
            guess_count: guesses.len(),
            answer_count: answers.len(),
            guess_hash,
            answer_hash,
            prior_hash,
            state_format_version: STATE_FORMAT_VERSION,
            aux_table_version: AUX_TABLE_VERSION,
            certificate_format_version: CERTIFICATE_FORMAT_VERSION,
            small_state_table_version: SMALL_STATE_TABLE_VERSION,
            small_state_table_hash,
            manifest_hash,
        };
        let prior = build_prior(&answers, config.prior)?;
        let answer_records = answers
            .iter()
            .map(|word| AnswerRecord {
                word: word.clone(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect::<Vec<_>>();
        let pattern_table = if immutable {
            PatternTable::load_existing_at(&artifacts.pattern_table, &guesses, &answer_records)?
        } else {
            PatternTable::load_or_build_at(&artifacts.pattern_table, &guesses, &answer_records)?
        };
        let guess_index = guesses
            .iter()
            .enumerate()
            .map(|(index, guess)| (guess.clone(), index))
            .collect::<HashMap<_, _>>();
        let zobrist = build_zobrist_tokens(answers.len());

        Ok(Self {
            manifest,
            guesses,
            answers,
            prior,
            zobrist,
            small_state_table,
            objective_spec,
            pattern_table,
            guess_index,
            raw_prior,
        })
    }
}

impl FormalPolicyRuntime {
    pub fn load(paths: &ProjectPaths, model_id: &str) -> Result<Self> {
        let artifacts = PolicyArtifactSet::published(paths, model_id)?;
        Self::load_generation(paths, model_id, artifacts)
    }

    fn load_generation(
        paths: &ProjectPaths,
        model_id: &str,
        artifacts: PolicyArtifactSet,
    ) -> Result<Self> {
        let model = FormalModel::load_artifacts(paths, model_id, &artifacts, true)?;
        let manifest: FormalManifest =
            parsing::read_json(&artifacts.manifest, parsing::MAX_JSON_BYTES)?;
        if serde_json::to_value(&manifest)? != serde_json::to_value(&model.manifest)? {
            bail!(
                "formal artifacts are stale for {}: expected manifest {}, found {}",
                model_id,
                model.manifest.manifest_hash,
                manifest.manifest_hash
            );
        }
        let metadata: ProofMetadata =
            parsing::read_json(&artifacts.metadata, parsing::MAX_JSON_BYTES)?;
        let values = read_values(&artifacts.values, &model)?;
        let policies = read_policy(&artifacts.policy, &model)?;
        if values.len() != policies.len() {
            bail!("formal values and policy tables have different state counts");
        }
        let mut ordered_states = values.keys().cloned().collect::<Vec<_>>();
        ordered_states.sort_by(|left, right| {
            left.state_hash()
                .cmp(&right.state_hash())
                .then_with(|| left.cmp_storage(right, model.answers.len()))
        });
        let state_ids = ordered_states
            .iter()
            .enumerate()
            .map(|(index, state)| (state.clone(), index as u32))
            .collect::<HashMap<_, _>>();
        let mut policy = HashMap::with_capacity(values.len());
        for (state, objective) in values {
            let best_guess = policies
                .get(&state)
                .copied()
                .ok_or_else(|| anyhow!("missing policy entry for state {}", state.state_hash()))?;
            policy.insert(
                state,
                StoredState {
                    objective,
                    best_guess,
                },
            );
        }

        let full = StateKey::full(model.answers.len(), &model.zobrist);
        let mut roots = policy
            .keys()
            .filter(|state| state.with_horizon(None) == full);
        let root_state = roots
            .next()
            .cloned()
            .ok_or_else(|| anyhow!("formal policy has no complete root state"))?;
        if roots.next().is_some() {
            bail!("formal policy has multiple root horizons");
        }
        let runtime = Self {
            model,
            policy,
            root_state,
            ordered_states,
            state_ids,
            metadata,
            artifacts,
        };
        runtime.validate_metadata()?;
        let certificate = parsing::read_json(
            &runtime.artifacts.certificate,
            parsing::MAX_CERTIFICATE_BYTES,
        )?;
        verifier::validate_structure(&runtime, &certificate)?;
        Ok(runtime)
    }

    pub fn initial_state(&self) -> StateKey {
        self.root_state.clone()
    }

    fn validate_metadata(&self) -> Result<()> {
        let root = self
            .policy
            .get(&self.root_state)
            .ok_or_else(|| anyhow!("formal root decision is missing"))?;
        let expected_horizon = match self.model.objective_spec.kind {
            FormalObjectiveKind::Lexicographic => Some(root.objective.worst_case_depth),
            FormalObjectiveKind::ExpectedOnly => None,
        };
        if self.root_state.horizon != expected_horizon
            || root.objective.worst_case_depth == 0
            || !root.objective.expected_guesses.is_finite()
            || root.objective.expected_guesses < 1.0
        {
            bail!("formal root decision has an invalid objective or horizon");
        }
        if self.metadata.model_id != self.model.manifest.model_id
            || self.metadata.manifest_hash != self.model.manifest.manifest_hash
            || self.metadata.solved_states != self.policy.len()
            || !same_objective(&self.metadata.root_objective, &root.objective)
        {
            bail!("formal proof metadata disagrees with the persisted policy");
        }
        Ok(())
    }

    pub fn apply_history(&self, observations: &[(String, u8)]) -> Result<StateKey> {
        let mut state = self.initial_state();
        for (guess, pattern) in observations {
            state = self.apply_feedback(&state, guess, *pattern)?;
        }
        Ok(state)
    }

    pub fn apply_feedback(&self, state: &StateKey, guess: &str, pattern: u8) -> Result<StateKey> {
        let guess_index = self
            .model
            .guess_index
            .get(&guess.to_ascii_lowercase())
            .copied()
            .ok_or_else(|| anyhow!("unknown guess: {}", guess))?;
        let next = StateKey::from_indices_with_tokens(
            self.model.answers.len(),
            state.indices().into_iter().filter(|answer_index| {
                self.model.pattern_table.get(guess_index, *answer_index) == pattern
            }),
            &self.model.zobrist,
        );
        if next.count() == 0 {
            bail!(
                "no answers remain after applying {} {}",
                guess,
                format_feedback_letters(pattern)
            );
        }
        Ok(next.with_horizon(state.child_horizon()))
    }

    pub fn has_guess(&self, guess: &str) -> bool {
        self.model
            .guess_index
            .contains_key(&guess.to_ascii_lowercase())
    }

    /// Returns the exact primary suggestion even if bounded alternative ranking
    /// stops early; use [`Self::explain_state`] to inspect completion status.
    pub fn suggest(&self, state: &StateKey, top: usize) -> Result<Vec<FormalSuggestion>> {
        if top == 0 {
            return Ok(Vec::new());
        }
        let explanation = self.explain_state(state, top)?;
        Ok(explanation.tied_moves)
    }

    /// Uses the default work limit. Interactive callers should supply cancellation
    /// and consume this single response for both explanation and suggestions.
    pub fn explain_state(&self, state: &StateKey, top: usize) -> Result<FormalStateExplanation> {
        self.explain_state_cancellable(state, top, DEFAULT_FORMAL_ALTERNATIVE_PARTITIONS, &|| false)
    }

    /// The work limit bounds additional partitions, including any uncached
    /// primary search. A stored primary costs one partition and no recursion.
    /// Incomplete alternatives are discarded, leaving only the exact primary.
    pub fn explain_state_cancellable(
        &self,
        state: &StateKey,
        top: usize,
        maximum_partitions: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<FormalStateExplanation> {
        let mut solver = IndependentExactSolver::new(&self.model);
        solver.cancelled = Some(cancelled);
        solver.remaining_partitions = Some(maximum_partitions);
        solver.check_cancelled()?;
        let stored = match self.policy.get(state) {
            Some(stored) => stored.clone(),
            None => solver.solve(state)?,
        };
        let state = if state.horizon.is_none()
            && self.model.objective_spec.kind == FormalObjectiveKind::Lexicographic
        {
            state.with_horizon(Some(stored.objective.worst_case_depth))
        } else {
            state.clone()
        };
        let buckets = partition_guess_with_scratch(
            self.model.answers.len(),
            &state,
            stored.best_guess,
            &self.model.pattern_table,
            &self.model.prior,
            &self.model.zobrist,
            &mut solver.scratch,
        )?;
        solver.check_cancelled()?;
        let mut bucket_sizes = buckets
            .iter()
            .map(|bucket| bucket.count)
            .collect::<Vec<_>>();
        bucket_sizes.sort_unstable_by(|left, right| right.cmp(left));
        let best = GuessEvaluation {
            guess_index: stored.best_guess,
            objective: stored.objective,
            bucket_sizes,
        };
        let (ranked, alternatives_status) = if top <= 1 {
            (vec![best.clone()], FormalAlternativesStatus::NotRequested)
        } else {
            match self.evaluate_state_ranked_with_solver(
                &state,
                &mut solver,
                Some((best.clone(), buckets)),
            ) {
                Ok(ranked) => (ranked, FormalAlternativesStatus::Complete),
                Err(error)
                    if matches!(
                        error.downcast_ref::<FormalSearchStop>(),
                        Some(FormalSearchStop::WorkLimit)
                    ) =>
                {
                    (
                        vec![best.clone()],
                        FormalAlternativesStatus::WorkLimitReached,
                    )
                }
                Err(error) => return Err(error),
            }
        };
        solver.check_cancelled()?;
        Ok(FormalStateExplanation {
            model_id: self.model.manifest.model_id.clone(),
            manifest_hash: self.model.manifest.manifest_hash.clone(),
            surviving_answers: state.count(),
            best_guess: self.model.guesses[best.guess_index].clone(),
            objective: best.objective.clone(),
            bucket_sizes: best.bucket_sizes.clone(),
            tied_moves: ranked
                .into_iter()
                .take(top)
                .map(|candidate| FormalSuggestion {
                    word: self.model.guesses[candidate.guess_index].clone(),
                    objective: candidate.objective,
                    bucket_sizes: candidate.bucket_sizes,
                })
                .collect(),
            alternatives_status,
        })
    }

    pub fn metadata(&self) -> &ProofMetadata {
        &self.metadata
    }

    pub fn manifest(&self) -> &FormalManifest {
        &self.model.manifest
    }

    fn evaluate_state_ranked(&self, state: &StateKey) -> Result<Vec<GuessEvaluation>> {
        let mut solver = IndependentExactSolver::new(&self.model);
        self.evaluate_state_ranked_with_solver(state, &mut solver, None)
    }

    fn evaluate_state_ranked_with_solver(
        &self,
        state: &StateKey,
        solver: &mut IndependentExactSolver<'_>,
        primary: Option<(GuessEvaluation, Vec<PartitionBucket>)>,
    ) -> Result<Vec<GuessEvaluation>> {
        let state = if state.horizon.is_none()
            && self.model.objective_spec.kind == FormalObjectiveKind::Lexicographic
        {
            let solved = solver.solve(state)?;
            state.with_horizon(Some(solved.objective.worst_case_depth))
        } else {
            state.clone()
        };
        if state.horizon == Some(0) {
            bail!("formal policy has exhausted its remaining horizon");
        }
        let total_mass = self.state_mass(&state);
        let mut scratch = PartitionScratch::default();
        let mut signature_map: HashMap<PartitionFingerprint, Vec<usize>> = HashMap::new();
        let mut evaluations = Vec::new();
        let mut evaluation_buckets: Vec<Vec<PartitionBucket>> = Vec::new();
        let primary_guess = primary
            .as_ref()
            .map(|(evaluation, _)| evaluation.guess_index);
        if let Some((evaluation, buckets)) = primary {
            signature_map.insert(partition_fingerprint_from_buckets(&buckets), vec![0]);
            evaluations.push(evaluation);
            evaluation_buckets.push(buckets);
        }
        let mut guess_indices = (0..self.model.guesses.len()).collect::<Vec<_>>();
        guess_indices.sort_unstable_by_key(|index| &self.model.guesses[*index]);
        for guess_index in guess_indices {
            if Some(guess_index) == primary_guess {
                continue;
            }
            solver.consume_partition()?;
            let buckets = partition_guess_with_scratch(
                self.model.answers.len(),
                &state,
                guess_index,
                &self.model.pattern_table,
                &self.model.prior,
                &self.model.zobrist,
                &mut scratch,
            )?;
            let signature = partition_fingerprint_from_buckets(&buckets);
            if signature_map.get(&signature).is_some_and(|indexes| {
                indexes
                    .iter()
                    .copied()
                    .any(|index| same_bucket_partition(&evaluation_buckets[index], &buckets))
            }) {
                continue;
            }
            let Some(built) =
                self.build_guess_evaluation(&state, guess_index, total_mass, &buckets, solver)?
            else {
                continue;
            };
            signature_map
                .entry(signature)
                .or_default()
                .push(evaluations.len());
            evaluation_buckets.push(buckets);
            evaluations.push(built);
        }
        evaluations[usize::from(primary_guess.is_some())..].sort_by(|left, right| {
            compare_evaluations_with_kind(
                left,
                right,
                &self.model.guesses,
                FormalObjectiveKind::ExpectedOnly,
            )
        });
        Ok(evaluations)
    }

    fn evaluate_state_exact(&self, state: &StateKey) -> Result<StoredState> {
        let ranked = self.evaluate_state_ranked(state)?;
        let best = ranked
            .first()
            .ok_or_else(|| anyhow!("state {} missing exact ranking", state.state_hash()))?;
        Ok(StoredState {
            objective: best.objective.clone(),
            best_guess: best.guess_index,
        })
    }

    fn solve_state_independent(&self, state: &StateKey) -> Result<StoredState> {
        IndependentExactSolver::new(&self.model).solve(state)
    }

    fn build_guess_evaluation(
        &self,
        state: &StateKey,
        guess_index: usize,
        total_mass: f64,
        buckets: &[PartitionBucket],
        solver: &mut IndependentExactSolver<'_>,
    ) -> Result<Option<GuessEvaluation>> {
        if buckets
            .iter()
            .any(|bucket| bucket.pattern != ALL_GREEN_PATTERN && bucket.count == state.count())
        {
            return Ok(None);
        }
        let mut worst_case = 1u8;
        let mut expected = 1.0;
        let mut bucket_sizes = buckets
            .iter()
            .map(|bucket| bucket.count)
            .collect::<Vec<_>>();
        bucket_sizes.sort_unstable_by(|left, right| right.cmp(left));
        for bucket in buckets {
            if bucket.pattern == ALL_GREEN_PATTERN {
                continue;
            }
            let child_state = bucket.state.with_horizon(state.child_horizon());
            let stored = if let Some(stored) = self.policy.get(&child_state) {
                stored.clone()
            } else {
                let Some(stored) = solver.solve_conditioned(&child_state)? else {
                    return Ok(None);
                };
                stored
            };
            worst_case = worst_case.max(
                stored
                    .objective
                    .worst_case_depth
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("formal depth exceeds u8 representation"))?,
            );
            expected += (bucket.mass / total_mass) * stored.objective.expected_guesses;
        }

        Ok(Some(GuessEvaluation {
            guess_index,
            objective: PolicyObjective {
                worst_case_depth: worst_case,
                expected_guesses: expected,
            },
            bucket_sizes,
        }))
    }

    fn state_mass(&self, state: &StateKey) -> f64 {
        let mut total = 0.0;
        state.for_each_index(|index| total += self.model.prior[index]);
        total
    }
}

impl FormalPolicyBuilder<'_> {
    fn check_cancelled(&self) -> Result<()> {
        if let Some(cancelled) = self.cancelled {
            check_cancelled(cancelled)?;
        }
        Ok(())
    }
    fn solve_state(&mut self, state: &StateKey) -> Result<StoredState> {
        if state.horizon.is_none()
            && self.model.objective_spec.kind == FormalObjectiveKind::Lexicographic
        {
            for horizon in 1..=u8::MAX {
                if let Some(best) = self.solve_conditioned(&state.with_horizon(Some(horizon)))? {
                    return Ok(best);
                }
            }
            bail!("state exceeds the representable formal depth");
        }
        self.solve_conditioned(state)?
            .ok_or_else(|| anyhow!("state has no policy within its remaining horizon"))
    }

    fn solve_conditioned(&mut self, state: &StateKey) -> Result<Option<StoredState>> {
        self.check_cancelled()?;
        if state.horizon == Some(0) {
            return Ok(None);
        }
        if let Some(existing) = self.memo.get(state) {
            return Ok(Some(existing.clone()));
        }
        if let Some(existing) = self.hot_tt.get(state) {
            return Ok(Some(existing));
        }
        if state.count() == 1 {
            let stored = singleton_state_for_model(&self.model, state)?;
            self.hot_tt.insert(state.clone(), stored.clone());
            return Ok(Some(stored));
        }
        if state.count() <= self.model.small_state_table.max_size {
            let mut solver = IndependentExactSolver::new(&self.model);
            solver.cancelled = self.cancelled.map(|callback| callback as &dyn Fn() -> bool);
            let exact = solver.solve_conditioned(state)?;
            if let Some(stored) = &exact {
                self.hot_tt.insert(state.clone(), stored.clone());
            }
            return Ok(exact);
        }

        let quick_plans = self.collect_quick_plans_for_state(state)?;
        let total_mass = self.state_mass(state);
        let mut best: Option<StoredState> = None;
        for plan in &quick_plans {
            self.check_cancelled()?;
            if state
                .horizon
                .is_some_and(|horizon| plan.lower_bound > horizon)
            {
                self.bound_hits += 1;
                continue;
            }
            let expected_lower_bound =
                guess_expected_lower_bound(&plan.buckets, total_mass, PATTERN_SPACE as f64);
            if best.as_ref().is_some_and(|stored| {
                expected_lower_bound > stored.objective.expected_guesses + 1e-12
            }) {
                self.bound_hits += 1;
                continue;
            }
            let mut remaining_lower = plan
                .buckets
                .iter()
                .filter(|bucket| bucket.pattern != ALL_GREEN_PATTERN)
                .map(|bucket| {
                    (bucket.mass / total_mass)
                        * child_expected_lower_bound(bucket, PATTERN_SPACE as f64)
                })
                .sum::<f64>();
            let mut worst_case = 1u8;
            let mut expected = 1.0;
            let mut valid = true;
            let mut unresolved = plan
                .buckets
                .iter()
                .filter(|bucket| bucket.pattern != ALL_GREEN_PATTERN)
                .cloned()
                .collect::<Vec<_>>();
            unresolved.sort_unstable_by_key(|bucket| std::cmp::Reverse(bucket.count));
            for bucket in unresolved {
                let probability = bucket.mass / total_mass;
                remaining_lower -=
                    probability * child_expected_lower_bound(&bucket, PATTERN_SPACE as f64);
                let child_state = bucket.state.with_horizon(state.child_horizon());
                let Some(child) = self.solve_conditioned(&child_state)? else {
                    valid = false;
                    self.bound_hits += 1;
                    break;
                };
                worst_case = worst_case.max(
                    child
                        .objective
                        .worst_case_depth
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("formal depth exceeds u8 representation"))?,
                );
                expected += probability * child.objective.expected_guesses;
                if best.as_ref().is_some_and(|stored| {
                    expected + remaining_lower > stored.objective.expected_guesses + 1e-12
                }) {
                    valid = false;
                    self.bound_hits += 1;
                    break;
                }
            }
            if !valid {
                continue;
            }
            let candidate = StoredState {
                objective: PolicyObjective {
                    worst_case_depth: worst_case,
                    expected_guesses: expected,
                },
                best_guess: plan.guess_index,
            };
            if best.as_ref().is_none_or(|current| {
                compare_stored_with_kind(
                    &candidate,
                    current,
                    &self.model.guesses,
                    FormalObjectiveKind::ExpectedOnly,
                )
                .is_lt()
            }) {
                best = Some(candidate);
            }
        }
        if let Some(stored) = &best {
            self.hot_tt.insert(state.clone(), stored.clone());
        }
        Ok(best)
    }

    fn materialize_policy_reachable_states(&mut self, root: &StateKey) -> Result<()> {
        let mut frontier = vec![root.clone()];
        let mut seen = HashSet::new();
        while let Some(state) = frontier.pop() {
            self.check_cancelled()?;
            if !seen.insert(state.clone()) || self.memo.contains_key(&state) {
                continue;
            }
            self.maybe_report_progress("materialization");
            let stored = self.solve_state(&state)?;
            self.memo.insert(state.clone(), stored.clone());
            let buckets = self.partition_guess(&state, stored.best_guess)?;
            for bucket in buckets {
                if bucket.pattern != ALL_GREEN_PATTERN {
                    frontier.push(bucket.state.with_horizon(state.child_horizon()));
                }
            }
        }
        Ok(())
    }

    fn collect_quick_plans_for_state(&mut self, state: &StateKey) -> Result<Vec<GuessQuickPlan>> {
        self.quick_plan_calls += 1;
        self.maybe_report_progress("search");
        let total_mass = self.state_mass(state);
        let mut raw_plans = (0..self.model.guesses.len())
            .into_par_iter()
            .map_init(
                PartitionScratch::default,
                |scratch, guess_index| -> Result<Option<(GuessQuickPlan, PartitionFingerprint)>> {
                    self.check_cancelled()?;
                    let buckets = partition_guess_with_scratch(
                        self.model.answers.len(),
                        state,
                        guess_index,
                        &self.model.pattern_table,
                        &self.model.prior,
                        &self.model.zobrist,
                        scratch,
                    )?;
                    if buckets.iter().any(|bucket| {
                        bucket.pattern != ALL_GREEN_PATTERN && bucket.count == state.count()
                    }) {
                        return Ok(None);
                    }
                    let max_bucket = buckets
                        .iter()
                        .filter(|bucket| bucket.pattern != ALL_GREEN_PATTERN)
                        .map(|bucket| bucket.count)
                        .max()
                        .unwrap_or(0);
                    let lower_bound = if max_bucket == 0 {
                        1
                    } else {
                        1 + depth_lower_bound(max_bucket)
                    };
                    let mut entropy = 0.0;
                    let mut solve_mass = 0.0;
                    for bucket in &buckets {
                        let probability = bucket.mass / total_mass;
                        if probability > 0.0 {
                            entropy -= probability * probability.log2();
                        }
                        if bucket.pattern == ALL_GREEN_PATTERN {
                            solve_mass = bucket.mass;
                        }
                    }
                    let signature = partition_fingerprint_from_buckets(&buckets);
                    Ok(Some((
                        GuessQuickPlan {
                            guess_index,
                            lower_bound,
                            max_bucket,
                            entropy,
                            solve_mass,
                            buckets,
                        },
                        signature,
                    )))
                },
            )
            .collect::<Result<Vec<_>>>()?;
        raw_plans.sort_by(|left, right| {
            left.as_ref()
                .map(|(plan, _)| &self.model.guesses[plan.guess_index])
                .cmp(
                    &right
                        .as_ref()
                        .map(|(plan, _)| &self.model.guesses[plan.guess_index]),
                )
        });
        let mut signatures: HashMap<PartitionFingerprint, Vec<usize>> = HashMap::new();
        let mut plans = Vec::new();
        self.partition_calls += self.model.guesses.len() as u64;
        for (plan, signature) in raw_plans.into_iter().flatten() {
            if partition_dedup_hit(&signatures, &plans, signature, &plan.buckets) {
                self.deduped_signatures += 1;
                continue;
            }
            signatures.entry(signature).or_default().push(plans.len());
            plans.push(plan);
        }
        plans.sort_by(|left, right| {
            left.lower_bound
                .cmp(&right.lower_bound)
                .then_with(|| left.max_bucket.cmp(&right.max_bucket))
                .then_with(|| right.solve_mass.total_cmp(&left.solve_mass))
                .then_with(|| right.entropy.total_cmp(&left.entropy))
                .then_with(|| {
                    self.model.guesses[left.guess_index].cmp(&self.model.guesses[right.guess_index])
                })
        });
        // Refinement pruning was intentionally removed. The previous subset direction could
        // discard the more informative partition, which is unsafe for both expected cost and
        // lexicographic worst-case/expected objectives.
        Ok(plans)
    }

    fn partition_guess(
        &mut self,
        state: &StateKey,
        guess_index: usize,
    ) -> Result<Vec<PartitionBucket>> {
        self.check_cancelled()?;
        self.partition_calls += 1;
        let mut scratch = PartitionScratch::default();
        partition_guess_with_scratch(
            self.model.answers.len(),
            state,
            guess_index,
            &self.model.pattern_table,
            &self.model.prior,
            &self.model.zobrist,
            &mut scratch,
        )
    }

    fn state_mass(&self, state: &StateKey) -> f64 {
        let mut total = 0.0;
        state.for_each_index(|index| total += self.model.prior[index]);
        total
    }

    fn maybe_report_progress(&mut self, phase: &str) {
        if !progress_enabled() {
            return;
        }
        if self.last_progress.elapsed() >= PROGRESS_INTERVAL {
            self.force_report_progress(phase);
        }
    }

    fn force_report_progress(&mut self, phase: &str) {
        if !progress_enabled() {
            return;
        }
        self.last_progress = Instant::now();
        eprintln!(
            "formal-progress phase={} elapsed_ms={} memo={} deduped_signatures={} bound_hits={} root_refinement_pruned={} local_refinement_pruned={} quick_plan_calls={} partition_calls={}",
            phase,
            self.started.elapsed().as_millis(),
            self.memo.len(),
            self.deduped_signatures,
            self.bound_hits,
            self.root_refinement_pruned,
            self.local_refinement_pruned,
            self.quick_plan_calls,
            self.partition_calls
        );
    }
}

fn progress_enabled() -> bool {
    !matches!(
        env::var("MAYBE_WORDLE_FORMAL_PROGRESS"),
        Ok(value) if matches!(value.trim(), "0" | "false" | "FALSE" | "False")
    )
}

impl StateKey {
    fn with_horizon(&self, horizon: Option<u8>) -> Self {
        Self {
            horizon,
            ..self.clone()
        }
    }

    fn child_horizon(&self) -> Option<u8> {
        self.horizon.map(|horizon| horizon.saturating_sub(1))
    }

    fn full(answer_count: usize, zobrist: &[u64]) -> Self {
        Self::from_indices_with_tokens(answer_count, 0..answer_count, zobrist)
    }

    #[cfg(test)]
    fn from_indices(answer_count: usize, indices: impl IntoIterator<Item = usize>) -> Self {
        let zobrist = build_zobrist_tokens(answer_count);
        Self::from_indices_with_tokens(answer_count, indices, &zobrist)
    }

    fn from_indices_with_tokens(
        answer_count: usize,
        indices: impl IntoIterator<Item = usize>,
        zobrist: &[u64],
    ) -> Self {
        let mut collected = indices.into_iter().collect::<Vec<_>>();
        collected.sort_unstable();
        if collected.len() <= INLINE_STATE_THRESHOLD {
            let mut inline = [0u16; STATE_INLINE_CAPACITY];
            let mut hash = 0u64;
            for (slot, index) in collected.iter().copied().enumerate() {
                inline[slot] = index as u16;
                hash ^= zobrist[index];
            }
            Self {
                storage: StateStorage::Inline {
                    len: collected.len() as u8,
                    indices: inline,
                },
                count: collected.len(),
                hash,
                horizon: None,
            }
        } else {
            let mut words = vec![0u64; answer_count.div_ceil(64)];
            let mut hash = 0u64;
            for index in collected {
                words[index / 64] |= 1u64 << (index % 64);
                hash ^= zobrist[index];
            }
            let count = words.iter().map(|word| word.count_ones() as usize).sum();
            Self {
                storage: StateStorage::Bitset(words.into_boxed_slice()),
                count,
                hash,
                horizon: None,
            }
        }
    }

    fn from_words_with_tokens(words: Vec<u64>, zobrist: &[u64]) -> Self {
        let count = words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        if count <= INLINE_STATE_THRESHOLD {
            let mut inline = [0u16; STATE_INLINE_CAPACITY];
            let mut slot = 0usize;
            let mut hash = 0u64;
            for (word_index, word) in words.iter().copied().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let index = (word_index * 64) + bit;
                    inline[slot] = index as u16;
                    slot += 1;
                    hash ^= zobrist[index];
                    bits &= bits - 1;
                }
            }
            Self {
                storage: StateStorage::Inline {
                    len: count as u8,
                    indices: inline,
                },
                count,
                hash,
                horizon: None,
            }
        } else {
            let mut hash = 0u64;
            for (word_index, word) in words.iter().copied().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    hash ^= zobrist[(word_index * 64) + bit];
                    bits &= bits - 1;
                }
            }
            Self {
                storage: StateStorage::Bitset(words.into_boxed_slice()),
                count,
                hash,
                horizon: None,
            }
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    fn state_hash(&self) -> u64 {
        self.hash
    }

    fn indices(&self) -> Vec<usize> {
        match &self.storage {
            StateStorage::Inline { len, indices } => indices[..*len as usize]
                .iter()
                .map(|index| *index as usize)
                .collect(),
            StateStorage::Bitset(words) => indices_from_words(words),
        }
    }

    fn for_each_index(&self, mut f: impl FnMut(usize)) {
        match &self.storage {
            StateStorage::Inline { len, indices } => {
                for index in &indices[..*len as usize] {
                    f(*index as usize);
                }
            }
            StateStorage::Bitset(words) => {
                for (word_index, word) in words.iter().copied().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        f((word_index * 64) + bit);
                        bits &= bits - 1;
                    }
                }
            }
        }
    }

    fn as_words(&self, answer_count: usize) -> Vec<u64> {
        match &self.storage {
            StateStorage::Inline { len, indices } => {
                let mut words = vec![0u64; answer_count.div_ceil(64)];
                for index in &indices[..*len as usize] {
                    let index = *index as usize;
                    words[index / 64] |= 1u64 << (index % 64);
                }
                words
            }
            StateStorage::Bitset(words) => words.to_vec(),
        }
    }

    fn write_tagged(&self, writer: &mut impl Write, answer_count: usize) -> Result<()> {
        writer.write_all(
            &self
                .horizon
                .map(u16::from)
                .unwrap_or(u16::MAX)
                .to_le_bytes(),
        )?;
        match &self.storage {
            StateStorage::Inline { len, indices } => {
                writer.write_all(&[STATE_TAG_INLINE])?;
                writer.write_all(&(*len as u16).to_le_bytes())?;
                for index in &indices[..*len as usize] {
                    writer.write_all(&index.to_le_bytes())?;
                }
            }
            StateStorage::Bitset(words) => {
                writer.write_all(&[STATE_TAG_BITSET])?;
                writer.write_all(&(answer_count.div_ceil(64) as u16).to_le_bytes())?;
                for word in words.iter() {
                    writer.write_all(&word.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    fn read_tagged(reader: &mut impl Read, answer_count: usize, zobrist: &[u64]) -> Result<Self> {
        if answer_count == 0 || answer_count > u16::MAX as usize || zobrist.len() != answer_count {
            bail!("unsupported formal answer dimensions");
        }
        let horizon = match read_u16(reader)? {
            u16::MAX => None,
            value if value > 0 && value <= u8::MAX as u16 => Some(value as u8),
            _ => bail!("invalid formal state horizon"),
        };
        let tag = read_u8(reader)?;
        let state = match tag {
            STATE_TAG_INLINE => {
                let len = read_u16(reader)? as usize;
                if len == 0 || len > INLINE_STATE_THRESHOLD || len > answer_count {
                    bail!("invalid inline formal state length");
                }
                let mut indices = Vec::with_capacity(len);
                for _ in 0..len {
                    let index = read_u16(reader)? as usize;
                    if index >= answer_count
                        || indices.last().is_some_and(|previous| *previous >= index)
                    {
                        bail!("formal state IDs must be in-range, unique and ordered");
                    }
                    indices.push(index);
                }
                Self::from_indices_with_tokens(answer_count, indices, zobrist)
            }
            STATE_TAG_BITSET => {
                let word_count = read_u16(reader)? as usize;
                if word_count != answer_count.div_ceil(64) {
                    bail!("unexpected tagged state word count");
                }
                let words = read_state_words(reader, word_count)?;
                let tail_bits = answer_count % 64;
                if tail_bits != 0 && words.last().is_some_and(|word| *word >> tail_bits != 0) {
                    bail!("formal bitset contains out-of-range answer IDs");
                }
                if words
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>()
                    <= INLINE_STATE_THRESHOLD
                {
                    bail!("noncanonical formal bitset; small states require inline encoding");
                }
                Self::from_words_with_tokens(words, zobrist)
            }
            _ => bail!("invalid state tag {}", tag),
        };
        Ok(state.with_horizon(horizon))
    }

    fn cmp_storage(&self, other: &Self, answer_count: usize) -> std::cmp::Ordering {
        let left = self.as_words(answer_count);
        let right = other.as_words(answer_count);
        left.cmp(&right)
            .then_with(|| self.horizon.cmp(&other.horizon))
    }
}

impl StateFrame {
    fn from_state(state: &StateKey) -> Option<Self> {
        (state.count() <= LOCAL_REINDEX_THRESHOLD).then(|| Self {
            global_ids: state
                .indices()
                .into_iter()
                .map(|index| index as u16)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }
}

fn build_zobrist_tokens(answer_count: usize) -> Vec<u64> {
    (0..answer_count)
        .map(|index| splitmix64((index as u64) + 0x9e37_79b9_7f4a_7c15))
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn indices_from_words(words: &[u64]) -> Vec<usize> {
    let mut indices = Vec::new();
    for (word_index, word) in words.iter().copied().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            indices.push((word_index * 64) + bit);
            bits &= bits - 1;
        }
    }
    indices
}

fn build_prior(answers: &[String], prior_spec: FormalPriorSpec) -> Result<Vec<f64>> {
    let mut weights = match prior_spec {
        FormalPriorSpec::Uniform => vec![1.0; answers.len()],
        FormalPriorSpec::Explicit { weights } => answers
            .iter()
            .map(|answer| {
                let weight = weights.get(answer).copied().ok_or_else(|| {
                    anyhow!("formal prior is missing a positive weight for answer {answer}")
                })?;
                if !weight.is_finite() || weight <= 0.0 {
                    bail!("formal prior weight for {answer} must be finite and positive");
                }
                Ok(weight)
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        bail!("formal prior must assign positive total probability mass");
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

fn build_exhaustive_proof_certificate(
    model: &FormalModel,
    policy: &HashMap<StateKey, StoredState>,
    cancelled: &dyn Fn() -> bool,
) -> Result<ProofCertificate> {
    let root = StateKey::full(model.answers.len(), &model.zobrist);
    let mut exhaustive = IndependentExactSolver::new(model);
    exhaustive.cancelled = Some(cancelled);
    let root_decision = exhaustive.solve(&root)?;
    if exhaustive.local_memo.len() > MAX_FORMAL_STATES {
        bail!("formal certificate closure exceeds the persisted state resource limit");
    }
    let root = root.with_horizon(match model.objective_spec.kind {
        FormalObjectiveKind::Lexicographic => Some(root_decision.objective.worst_case_depth),
        FormalObjectiveKind::ExpectedOnly => None,
    });
    let mut proof_entries = exhaustive.local_memo.iter().collect::<Vec<_>>();
    proof_entries.sort_by(|(left_key, _), (right_key, _)| {
        left_key
            .count()
            .cmp(&right_key.count())
            .then_with(|| left_key.state_hash().cmp(&right_key.state_hash()))
            .then_with(|| left_key.cmp_storage(right_key, model.answers.len()))
    });
    let state_ids = proof_entries
        .iter()
        .enumerate()
        .map(|(index, (state, _))| ((*state).clone(), index as u32))
        .collect::<HashMap<_, _>>();
    for (state, stored) in policy {
        let independently_solved = exhaustive.local_memo.get(state).ok_or_else(|| {
            anyhow!(
                "persisted policy state {} is absent from the exhaustive proof closure",
                state.state_hash()
            )
        })?;
        if !same_decision(stored, independently_solved) {
            bail!(
                "persisted policy state {} disagrees with the exhaustive proof closure",
                state.state_hash()
            );
        }
    }

    let mut scratch = PartitionScratch::default();
    let mut states = Vec::with_capacity(proof_entries.len());
    for (state_id, (state, stored)) in proof_entries.iter().enumerate() {
        check_cancelled(cancelled)?;
        let total_mass = state_total_mass(state, &model.prior);
        let mut candidates = Vec::with_capacity(model.guesses.len());
        let mut signature_map: HashMap<PartitionFingerprint, Vec<usize>> = HashMap::new();
        let mut representative_guesses = Vec::new();
        let mut representative_buckets: Vec<Vec<PartitionBucket>> = Vec::new();
        for guess_index in 0..model.guesses.len() {
            check_cancelled(cancelled)?;
            let buckets = partition_guess_with_scratch(
                model.answers.len(),
                state,
                guess_index,
                &model.pattern_table,
                &model.prior,
                &model.zobrist,
                &mut scratch,
            )?;
            let signature = partition_fingerprint_from_buckets(&buckets);
            let equivalent = signature_map.get(&signature).and_then(|indexes| {
                indexes
                    .iter()
                    .copied()
                    .find(|index| same_bucket_partition(&representative_buckets[*index], &buckets))
            });
            let witness = if let Some(representative_index) = equivalent {
                PersistedCandidateWitness::Equivalent {
                    representative_guess: representative_guesses[representative_index],
                }
            } else if let Some(bucket) = buckets
                .iter()
                .find(|bucket| bucket.pattern != ALL_GREEN_PATTERN && bucket.count == state.count())
            {
                PersistedCandidateWitness::NonProgress {
                    pattern: bucket.pattern,
                }
            } else if buckets.iter().any(|bucket| {
                bucket.pattern != ALL_GREEN_PATTERN
                    && !exhaustive
                        .local_memo
                        .contains_key(&bucket.state.with_horizon(state.child_horizon()))
            }) {
                PersistedCandidateWitness::Infeasible
            } else {
                let mut objective = PolicyObjective {
                    worst_case_depth: 1,
                    expected_guesses: 1.0,
                };
                let mut children = Vec::new();
                for bucket in buckets
                    .iter()
                    .filter(|bucket| bucket.pattern != ALL_GREEN_PATTERN)
                {
                    let child_state = bucket.state.with_horizon(state.child_horizon());
                    let child = exhaustive.local_memo.get(&child_state).ok_or_else(|| {
                        anyhow!(
                            "exact proof closure is missing child state {}",
                            bucket.state.state_hash()
                        )
                    })?;
                    objective.worst_case_depth = objective
                        .worst_case_depth
                        .max(1 + child.objective.worst_case_depth);
                    objective.expected_guesses +=
                        (bucket.mass / total_mass) * child.objective.expected_guesses;
                    children.push(PersistedCertificateChild {
                        pattern: bucket.pattern,
                        child_state_id: state_ids[&child_state],
                        objective: child.objective.clone(),
                        mass: bucket.mass,
                    });
                }
                PersistedCandidateWitness::Exact {
                    objective,
                    children,
                }
            };
            if equivalent.is_none() {
                let representative_index = representative_buckets.len();
                signature_map
                    .entry(signature)
                    .or_default()
                    .push(representative_index);
                representative_guesses.push(guess_index);
                representative_buckets.push(buckets);
            }
            candidates.push(PersistedCertificateCandidate {
                guess_index,
                witness,
            });
        }
        states.push(PersistedCertificateState {
            state_id: state_id as u32,
            answer_indices: state
                .indices()
                .into_iter()
                .map(|index| index as AnswerId)
                .collect(),
            horizon: state.horizon,
            best_guess: stored.best_guess,
            best_objective: stored.objective.clone(),
            candidates,
        });
    }

    Ok(ProofCertificate {
        model_id: model.manifest.model_id.clone(),
        manifest_hash: model.manifest.manifest_hash.clone(),
        objective_id: model.manifest.objective_id.clone(),
        objective_version: model.manifest.objective_version,
        state_format_version: model.manifest.state_format_version,
        aux_table_version: model.manifest.aux_table_version,
        certificate_format_version: model.manifest.certificate_format_version,
        small_state_table_hash: model.manifest.small_state_table_hash.clone(),
        policy_state_count: policy.len(),
        state_count: states.len(),
        root_state_id: state_ids[&root],
        states,
    })
}

fn write_values(
    path: &Path,
    model: &FormalModel,
    entries: &[(&StateKey, &StoredState)],
) -> Result<()> {
    let mut writer = Vec::new();
    writer.write_all(VALUES_MAGIC)?;
    writer.write_all(model.manifest.manifest_hash.as_bytes())?;
    writer.write_all(&(entries.len() as u64).to_le_bytes())?;
    writer.write_all(&(model.answers.len().div_ceil(64) as u32).to_le_bytes())?;
    for (state, stored) in entries {
        writer.write_all(&state.state_hash().to_le_bytes())?;
        state.write_tagged(&mut writer, model.answers.len())?;
        writer.write_all(&[stored.objective.worst_case_depth])?;
        writer.write_all(&stored.objective.expected_guesses.to_le_bytes())?;
    }
    atomic_write(path, &writer)
}

fn write_policy(
    path: &Path,
    model: &FormalModel,
    entries: &[(&StateKey, &StoredState)],
) -> Result<()> {
    let mut writer = Vec::new();
    writer.write_all(POLICY_MAGIC)?;
    writer.write_all(model.manifest.manifest_hash.as_bytes())?;
    writer.write_all(&(entries.len() as u64).to_le_bytes())?;
    writer.write_all(&(model.answers.len().div_ceil(64) as u32).to_le_bytes())?;
    for (state, stored) in entries {
        writer.write_all(&state.state_hash().to_le_bytes())?;
        state.write_tagged(&mut writer, model.answers.len())?;
        writer.write_all(&(stored.best_guess as u32).to_le_bytes())?;
    }
    atomic_write(path, &writer)
}

fn read_values(path: &Path, model: &FormalModel) -> Result<HashMap<StateKey, PolicyObjective>> {
    let (mut reader, file_length) = bounded_binary_reader(path)?;
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    if &header != VALUES_MAGIC {
        bail!(
            "unsupported values artifact format: {}; rebuild formal artifacts",
            path.display()
        );
    }
    let manifest_hash = read_tagged_digest(&mut reader)?;
    if manifest_hash != model.manifest.manifest_hash {
        bail!("stale values file: {}", path.display());
    }
    let count = bounded_record_count(read_u64(&mut reader)?, file_length, 24)?;
    let word_count = read_u32(&mut reader)? as usize;
    if word_count != model.answers.len().div_ceil(64) {
        bail!("unexpected state word count in {}", path.display());
    }
    let mut values = HashMap::with_capacity(count);
    for _ in 0..count {
        let state_hash = read_u64(&mut reader)?;
        let state = StateKey::read_tagged(&mut reader, model.answers.len(), &model.zobrist)?;
        let worst_case_depth = read_u8(&mut reader)?;
        let expected_guesses = read_f64(&mut reader)?;
        validate_loaded_state(&state, model)?;
        validate_loaded_objective(
            &state,
            &PolicyObjective {
                worst_case_depth,
                expected_guesses,
            },
        )?;
        if state.state_hash() != state_hash {
            bail!("state hash mismatch in {}", path.display());
        }
        if values
            .insert(
                state,
                PolicyObjective {
                    worst_case_depth,
                    expected_guesses,
                },
            )
            .is_some()
        {
            bail!("duplicate formal value state in {}", path.display());
        }
    }
    reject_trailing_bytes(&mut reader)?;
    Ok(values)
}

fn read_policy(path: &Path, model: &FormalModel) -> Result<HashMap<StateKey, usize>> {
    let (mut reader, file_length) = bounded_binary_reader(path)?;
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    if &header != POLICY_MAGIC {
        bail!(
            "unsupported policy artifact format: {}; rebuild formal artifacts",
            path.display()
        );
    }
    let manifest_hash = read_tagged_digest(&mut reader)?;
    if manifest_hash != model.manifest.manifest_hash {
        bail!("stale policy file: {}", path.display());
    }
    let count = bounded_record_count(read_u64(&mut reader)?, file_length, 19)?;
    let word_count = read_u32(&mut reader)? as usize;
    if word_count != model.answers.len().div_ceil(64) {
        bail!("unexpected state word count in {}", path.display());
    }
    let mut policies = HashMap::with_capacity(count);
    for _ in 0..count {
        let state_hash = read_u64(&mut reader)?;
        let state = StateKey::read_tagged(&mut reader, model.answers.len(), &model.zobrist)?;
        validate_loaded_state(&state, model)?;
        let best_guess = read_u32(&mut reader)? as usize;
        if best_guess >= model.guesses.len() {
            bail!(
                "policy references invalid guess index in {}",
                path.display()
            );
        }
        if state.state_hash() != state_hash {
            bail!("state hash mismatch in {}", path.display());
        }
        if policies.insert(state, best_guess).is_some() {
            bail!("duplicate formal policy state in {}", path.display());
        }
    }
    reject_trailing_bytes(&mut reader)?;
    Ok(policies)
}

fn bounded_binary_reader(path: &Path) -> Result<(BufReader<std::io::Take<File>>, u64)> {
    let file = parsing::open_regular_file(path)?;
    let length = file.metadata()?.len();
    if !(BINARY_HEADER_BYTES..=MAX_FORMAL_BINARY_BYTES).contains(&length) {
        bail!(
            "formal binary length exceeds parser limits: {}; rebuild formal artifacts",
            path.display()
        );
    }
    Ok((
        BufReader::new(file.take(MAX_FORMAL_BINARY_BYTES + 1)),
        length,
    ))
}

fn bounded_record_count(count: u64, file_length: u64, minimum_record_bytes: u64) -> Result<usize> {
    let available = file_length.saturating_sub(BINARY_HEADER_BYTES);
    if count == 0 || count > MAX_FORMAL_STATES as u64 || count > available / minimum_record_bytes {
        bail!("formal record count exceeds file length or parser limits; rebuild formal artifacts");
    }
    usize::try_from(count).context("formal record count exceeds address space")
}

fn validate_loaded_state(state: &StateKey, model: &FormalModel) -> Result<()> {
    if state.horizon == Some(0)
        || state.horizon.is_some()
            != (model.objective_spec.kind == FormalObjectiveKind::Lexicographic)
    {
        bail!("formal state horizon does not match the declared objective");
    }
    Ok(())
}

fn validate_loaded_objective(state: &StateKey, objective: &PolicyObjective) -> Result<()> {
    if objective.worst_case_depth == 0
        || objective.worst_case_depth as usize > state.count()
        || !objective.expected_guesses.is_finite()
        || objective.expected_guesses < 1.0
        || objective.expected_guesses > f64::from(objective.worst_case_depth) + 1e-9
        || state
            .horizon
            .is_some_and(|horizon| objective.worst_case_depth > horizon)
    {
        bail!("invalid formal objective; rebuild formal artifacts");
    }
    Ok(())
}

fn reject_trailing_bytes(reader: &mut impl Read) -> Result<()> {
    if reader.read(&mut [0u8; 1])? != 0 {
        bail!("formal artifact contains trailing data; rebuild formal artifacts");
    }
    Ok(())
}

fn read_state_words(reader: &mut impl Read, word_count: usize) -> Result<Vec<u64>> {
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        words.push(read_u64(reader)?);
    }
    Ok(words)
}

fn read_tagged_digest(reader: &mut impl Read) -> Result<String> {
    let mut bytes = [0u8; TAGGED_DIGEST_LENGTH];
    reader.read_exact(&mut bytes)?;
    let digest = std::str::from_utf8(&bytes).context("artifact digest must be UTF-8")?;
    if !crate::identity::is_tagged_digest(digest) {
        bail!("artifact uses an unsupported identity format; rebuild formal artifacts");
    }
    Ok(digest.to_string())
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64(reader: &mut impl Read) -> Result<f64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

#[cfg(test)]
fn read_proof_certificate(paths: &ProjectPaths, model_id: &str) -> Result<ProofCertificate> {
    let artifacts = PolicyArtifactSet::published(paths, model_id)?;
    parsing::read_json(&artifacts.certificate, parsing::MAX_CERTIFICATE_BYTES)
}

#[cfg(test)]
fn verify_certificate(runtime: &FormalPolicyRuntime, certificate: &ProofCertificate) -> Result<()> {
    verifier::verify_certificate_witnesses(runtime, certificate)
}

fn combine_hashes(
    left: &str,
    middle: &str,
    right: &str,
    objective_spec: FormalObjectiveSpec,
    small_state_table_hash: &str,
) -> String {
    let mut hash = CanonicalSha256::new("maybe-wordle-formal-manifest-v2");
    hash.field(left.as_bytes())
        .field(middle.as_bytes())
        .field(right.as_bytes())
        .field(objective_spec.id.as_bytes())
        .field(&objective_spec.version.to_le_bytes())
        .field(&STATE_FORMAT_VERSION.to_le_bytes())
        .field(&AUX_TABLE_VERSION.to_le_bytes())
        .field(&CERTIFICATE_FORMAT_VERSION.to_le_bytes())
        .field(&SMALL_STATE_TABLE_VERSION.to_le_bytes())
        .field(small_state_table_hash.as_bytes());
    hash.finish_tagged()
}

fn hash_small_state_table(table: &SmallStateTable) -> String {
    let mut hash = CanonicalSha256::new("maybe-wordle-small-state-table-v2");
    hash.field(&table.version.to_le_bytes())
        .field(&(table.max_size as u64).to_le_bytes())
        .field(&(table.expected_lower_bound_by_size.len() as u64).to_le_bytes());
    for value in &table.expected_lower_bound_by_size {
        hash.field(&value.to_bits().to_le_bytes());
    }
    hash.finish_tagged()
}

fn validate_small_state_table(
    persisted: &SmallStateTable,
    canonical: &SmallStateTable,
) -> Result<()> {
    if persisted.version != canonical.version
        || persisted.max_size != canonical.max_size
        || persisted.expected_lower_bound_by_size.len()
            != canonical.expected_lower_bound_by_size.len()
        || persisted
            .expected_lower_bound_by_size
            .iter()
            .zip(&canonical.expected_lower_bound_by_size)
            .any(|(left, right)| (left - right).abs() > 1e-12)
    {
        bail!("formal small-state table is stale or invalid; rebuild formal artifacts");
    }
    Ok(())
}

fn depth_lower_bound(count: usize) -> u8 {
    if count <= 1 {
        return 0;
    }
    let mut depth = 0u8;
    let mut capacity = 1usize;
    while capacity < count {
        depth += 1;
        capacity = capacity.saturating_mul(PATTERN_SPACE);
    }
    depth
}

fn guess_expected_lower_bound(
    buckets: &[PartitionBucket],
    total_mass: f64,
    pattern_space: f64,
) -> f64 {
    let mut lower_bound = 1.0;
    for bucket in buckets {
        if bucket.pattern == ALL_GREEN_PATTERN {
            continue;
        }
        let child_floor = child_expected_lower_bound(bucket, pattern_space);
        lower_bound += (bucket.mass / total_mass) * child_floor;
    }
    lower_bound
}

fn child_expected_lower_bound(bucket: &PartitionBucket, pattern_space: f64) -> f64 {
    if bucket.count <= 1 {
        return 1.0;
    }
    (bucket.entropy_bits / pattern_space.log2()).max(1.0)
}

fn partition_guess_with_scratch(
    answer_count: usize,
    state: &StateKey,
    guess_index: usize,
    pattern_table: &PatternTable,
    prior: &[f64],
    zobrist: &[u64],
    scratch: &mut PartitionScratch,
) -> Result<Vec<PartitionBucket>> {
    #[cfg(test)]
    runtime_tests::PARTITION_CALLS.with(|calls| calls.set(calls.get() + 1));
    scratch.masses.fill(0.0);
    scratch.counts.fill(0);
    scratch.weighted_log_sums.fill(0.0);
    if let Some(frame) = StateFrame::from_state(state) {
        scratch.positions.resize(frame.global_ids.len() * 2, 0);
        let (patterns, staged) = scratch.positions.split_at_mut(frame.global_ids.len());
        for (local_index, answer_id) in frame.global_ids.iter().copied().enumerate() {
            let answer_index = answer_id as usize;
            let pattern = pattern_table.get(guess_index, answer_index) as usize;
            scratch.counts[pattern] += 1;
            let weight = prior[answer_index];
            scratch.masses[pattern] += weight;
            if weight > 0.0 {
                scratch.weighted_log_sums[pattern] += weight * weight.log2();
            }
            patterns[local_index] = pattern as AnswerId;
        }
        let mut next = 0usize;
        for pattern in 0..PATTERN_SPACE {
            scratch.offsets[pattern] = next;
            next += scratch.counts[pattern];
        }
        scratch.offsets[PATTERN_SPACE] = next;
        let mut cursors = [0usize; PATTERN_SPACE];
        cursors.copy_from_slice(&scratch.offsets[..PATTERN_SPACE]);
        for (local_index, pattern) in patterns.iter().copied().enumerate() {
            let slot = &mut cursors[pattern as usize];
            staged[*slot] = local_index as AnswerId;
            *slot += 1;
        }
        let built = (0..PATTERN_SPACE)
            .filter(|pattern| scratch.counts[*pattern] > 0)
            .map(|pattern| {
                let start = scratch.offsets[pattern];
                let end = scratch.offsets[pattern + 1];
                let child = StateKey::from_indices_with_tokens(
                    answer_count,
                    staged[start..end]
                        .iter()
                        .map(|local_index| frame.global_ids[*local_index as usize] as usize),
                    zobrist,
                );
                PartitionBucket {
                    pattern: pattern as u8,
                    state: child,
                    mass: scratch.masses[pattern],
                    count: scratch.counts[pattern],
                    entropy_bits: if scratch.counts[pattern] <= 1 || scratch.masses[pattern] <= 0.0
                    {
                        0.0
                    } else {
                        scratch.masses[pattern].log2()
                            - (scratch.weighted_log_sums[pattern] / scratch.masses[pattern])
                    },
                }
            })
            .collect::<Vec<_>>();
        if built.is_empty() {
            bail!("guess partition unexpectedly empty");
        }
        return Ok(built);
    }

    let word_count = answer_count.div_ceil(64);
    scratch.words.resize(PATTERN_SPACE * word_count, 0);
    scratch.words.fill(0);
    state.for_each_index(|answer_index| {
        let pattern = pattern_table.get(guess_index, answer_index) as usize;
        let offset = (pattern * word_count) + (answer_index / 64);
        scratch.words[offset] |= 1u64 << (answer_index % 64);
        let weight = prior[answer_index];
        scratch.masses[pattern] += weight;
        scratch.counts[pattern] += 1;
        if weight > 0.0 {
            scratch.weighted_log_sums[pattern] += weight * weight.log2();
        }
    });

    let built = (0..PATTERN_SPACE)
        .filter(|pattern| scratch.counts[*pattern] > 0)
        .map(|pattern| {
            let offset = pattern * word_count;
            PartitionBucket {
                pattern: pattern as u8,
                state: state_key_from_words(
                    scratch.words[offset..offset + word_count].to_vec(),
                    zobrist,
                ),
                mass: scratch.masses[pattern],
                count: scratch.counts[pattern],
                entropy_bits: if scratch.counts[pattern] <= 1 || scratch.masses[pattern] <= 0.0 {
                    0.0
                } else {
                    scratch.masses[pattern].log2()
                        - (scratch.weighted_log_sums[pattern] / scratch.masses[pattern])
                },
            }
        })
        .collect::<Vec<_>>();
    if built.is_empty() {
        bail!("guess partition unexpectedly empty");
    }
    Ok(built)
}

fn state_key_from_words(words: Vec<u64>, zobrist: &[u64]) -> StateKey {
    StateKey::from_words_with_tokens(words, zobrist)
}

fn partition_fingerprint_from_buckets(buckets: &[PartitionBucket]) -> PartitionFingerprint {
    let mut mix_a = 0xcbf2_9ce4_8422_2325u64;
    let mut mix_b = 0x9e37_79b9_7f4a_7c15u64;
    for bucket in buckets {
        let encoded =
            ((bucket.pattern as u64) << 56) ^ bucket.state.state_hash() ^ (bucket.count as u64);
        mix_a = mix_a.rotate_left(7) ^ splitmix64(encoded ^ mix_b);
        mix_b = mix_b
            .wrapping_mul(0x1000_0000_01b3)
            .wrapping_add(splitmix64(encoded ^ mix_a));
    }
    PartitionFingerprint {
        bucket_count: buckets.len() as u16,
        mix_a,
        mix_b,
    }
}

fn partition_dedup_hit(
    fingerprints: &HashMap<PartitionFingerprint, Vec<usize>>,
    plans: &[GuessQuickPlan],
    fingerprint: PartitionFingerprint,
    buckets: &[PartitionBucket],
) -> bool {
    fingerprints.get(&fingerprint).is_some_and(|indexes| {
        indexes
            .iter()
            .copied()
            .any(|index| same_bucket_partition(&plans[index].buckets, buckets))
    })
}

fn same_bucket_partition(left: &[PartitionBucket], right: &[PartitionBucket]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left_bucket, right_bucket)| {
                left_bucket.pattern == right_bucket.pattern
                    && left_bucket.state == right_bucket.state
            })
}

fn state_total_mass(state: &StateKey, prior: &[f64]) -> f64 {
    let mut total = 0.0;
    state.for_each_index(|index| total += prior[index]);
    total
}

fn compare_stored_with_kind(
    left: &StoredState,
    right: &StoredState,
    guesses: &[String],
    kind: FormalObjectiveKind,
) -> std::cmp::Ordering {
    compare_objective_with_kind(&left.objective, &right.objective, kind)
        .then_with(|| guesses[left.best_guess].cmp(&guesses[right.best_guess]))
}

fn compare_objective_with_kind(
    left: &PolicyObjective,
    right: &PolicyObjective,
    kind: FormalObjectiveKind,
) -> std::cmp::Ordering {
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

fn compare_evaluations_with_kind(
    left: &GuessEvaluation,
    right: &GuessEvaluation,
    guesses: &[String],
    kind: FormalObjectiveKind,
) -> std::cmp::Ordering {
    compare_objective_with_kind(&left.objective, &right.objective, kind)
        .then_with(|| guesses[left.guess_index].cmp(&guesses[right.guess_index]))
}

fn singleton_state_for_model(model: &FormalModel, state: &StateKey) -> Result<StoredState> {
    let answer_index = state
        .indices()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("singleton state was empty"))?;
    let best_guess = model
        .guess_index
        .get(&model.answers[answer_index])
        .copied()
        .ok_or_else(|| {
            anyhow!(
                "answer {} is not a valid guess",
                model.answers[answer_index]
            )
        })?;
    Ok(StoredState {
        objective: PolicyObjective {
            worst_case_depth: 1,
            expected_guesses: 1.0,
        },
        best_guess,
    })
}

#[cfg(test)]
fn state_is_subset_of(left: &StateKey, right: &StateKey) -> bool {
    if left.count() > right.count() {
        return false;
    }
    match (&left.storage, &right.storage) {
        (
            StateStorage::Inline {
                len: left_len,
                indices: left_indices,
            },
            StateStorage::Inline {
                len: right_len,
                indices: right_indices,
            },
        ) => {
            let left_slice = &left_indices[..*left_len as usize];
            let right_slice = &right_indices[..*right_len as usize];
            let mut left_pos = 0usize;
            let mut right_pos = 0usize;
            while left_pos < left_slice.len() && right_pos < right_slice.len() {
                match left_slice[left_pos].cmp(&right_slice[right_pos]) {
                    std::cmp::Ordering::Less => return false,
                    std::cmp::Ordering::Equal => {
                        left_pos += 1;
                        right_pos += 1;
                    }
                    std::cmp::Ordering::Greater => right_pos += 1,
                }
            }
            left_pos == left_slice.len()
        }
        (
            StateStorage::Inline {
                len: left_len,
                indices: left_indices,
            },
            StateStorage::Bitset(right_words),
        ) => left_indices[..*left_len as usize].iter().all(|index| {
            let index = *index as usize;
            let word_index = index / 64;
            word_index < right_words.len()
                && (right_words[word_index] & (1u64 << (index % 64))) != 0
        }),
        (StateStorage::Bitset(left_words), StateStorage::Bitset(right_words)) => left_words
            .iter()
            .zip(right_words.iter())
            .all(|(left_word, right_word)| left_word & !right_word == 0),
        (StateStorage::Bitset(_), StateStorage::Inline { .. }) => false,
    }
}

fn same_decision(left: &StoredState, right: &StoredState) -> bool {
    left.best_guess == right.best_guess && same_objective(&left.objective, &right.objective)
}

fn same_objective(left: &PolicyObjective, right: &PolicyObjective) -> bool {
    left.worst_case_depth == right.worst_case_depth
        && (left.expected_guesses - right.expected_guesses).abs() < 1e-9
}

pub fn parse_observations(guesses: &[String], feedbacks: &[String]) -> Result<Vec<(String, u8)>> {
    if guesses.len() != feedbacks.len() {
        bail!("--guess and --feedback must appear the same number of times");
    }
    guesses
        .iter()
        .zip(feedbacks)
        .map(|(guess, feedback)| Ok((guess.trim().to_ascii_lowercase(), parse_feedback(feedback)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
        io::Cursor,
        path::Path,
    };

    use chrono::NaiveDate;

    use super::*;
    use crate::{data::ProjectPaths, model::AnswerRecord, pattern_table::PatternTable};

    fn write_fixture(path: &Path, contents: &str) {
        std::fs::write(path, contents).expect("write fixture");
    }

    fn synthetic_words(count: usize) -> Vec<String> {
        (0..count)
            .map(|mut value| {
                let mut chars = ['a'; 5];
                for slot in (0..5).rev() {
                    chars[slot] = char::from(b'a' + (value % 26) as u8);
                    value /= 26;
                }
                chars.into_iter().collect::<String>()
            })
            .collect()
    }

    fn synthetic_answers(words: &[String]) -> Vec<AnswerRecord> {
        words
            .iter()
            .map(|word| AnswerRecord {
                word: word.clone(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect()
    }

    fn state_hash_for_tests(state: &StateKey) -> u64 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        hasher.finish()
    }

    fn bucket_from_indices(
        answer_count: usize,
        pattern: u8,
        indices: impl IntoIterator<Item = usize>,
        zobrist: &[u64],
    ) -> PartitionBucket {
        let state = StateKey::from_indices_with_tokens(answer_count, indices, zobrist);
        PartitionBucket {
            pattern,
            count: state.count(),
            state,
            mass: 1.0,
            entropy_bits: 0.0,
        }
    }

    fn partition_guess_scalar(
        answer_count: usize,
        state: &StateKey,
        guess_index: usize,
        pattern_table: &PatternTable,
        prior: &[f64],
        zobrist: &[u64],
    ) -> Vec<PartitionBucket> {
        let word_count = answer_count.div_ceil(64);
        let mut words = vec![0u64; PATTERN_SPACE * word_count];
        let mut masses = [0.0f64; PATTERN_SPACE];
        let mut counts = [0usize; PATTERN_SPACE];
        let mut weighted_log_sums = [0.0f64; PATTERN_SPACE];
        state.for_each_index(|answer_index| {
            let pattern = pattern_table.get(guess_index, answer_index) as usize;
            let offset = (pattern * word_count) + (answer_index / 64);
            words[offset] |= 1u64 << (answer_index % 64);
            let weight = prior[answer_index];
            masses[pattern] += weight;
            counts[pattern] += 1;
            if weight > 0.0 {
                weighted_log_sums[pattern] += weight * weight.log2();
            }
        });
        (0..PATTERN_SPACE)
            .filter(|pattern| counts[*pattern] > 0)
            .map(|pattern| {
                let offset = pattern * word_count;
                PartitionBucket {
                    pattern: pattern as u8,
                    state: state_key_from_words(
                        words[offset..offset + word_count].to_vec(),
                        zobrist,
                    ),
                    mass: masses[pattern],
                    count: counts[pattern],
                    entropy_bits: if counts[pattern] <= 1 || masses[pattern] <= 0.0 {
                        0.0
                    } else {
                        masses[pattern].log2() - (weighted_log_sums[pattern] / masses[pattern])
                    },
                }
            })
            .collect()
    }

    #[test]
    fn state_key_counts_and_hashes_stably() {
        let state = StateKey::from_indices(10, [1, 3, 9]);
        assert_eq!(state.count(), 3);
        assert_eq!(state.indices(), vec![1, 3, 9]);
        assert_eq!(state.state_hash(), state.state_hash());
    }

    #[test]
    fn formal_expected_bound_does_not_apply_uniform_count_floor_to_skewed_mass() {
        let probabilities = [0.59_f64, 0.40, 0.01];
        let entropy_bits = probabilities
            .iter()
            .map(|probability| -probability * probability.log2())
            .sum::<f64>();
        let bucket = PartitionBucket {
            pattern: 0,
            state: StateKey::from_indices(3, 0..3),
            mass: 1.0,
            count: 3,
            entropy_bits,
        };
        assert!(SmallStateTable::build(3).lower_bound(3) > 1.0);
        assert_eq!(
            child_expected_lower_bound(&bucket, PATTERN_SPACE as f64),
            1.0
        );
    }

    #[test]
    fn explicit_formal_prior_requires_every_answer_to_have_positive_finite_mass() {
        let answers = vec!["cigar".to_string(), "rebut".to_string()];
        let missing = build_prior(
            &answers,
            FormalPriorSpec::Explicit {
                weights: HashMap::from([("cigar".to_string(), 1.0)]),
            },
        )
        .expect_err("missing formal mass must fail");
        assert!(missing.to_string().contains("missing a positive weight"));

        let negative = build_prior(
            &answers,
            FormalPriorSpec::Explicit {
                weights: HashMap::from([("cigar".to_string(), 1.0), ("rebut".to_string(), -0.1)]),
            },
        )
        .expect_err("negative formal mass must fail");
        assert!(negative.to_string().contains("finite and positive"));
    }

    #[test]
    fn reproducible_manifest_hash_uses_same_inputs() {
        let fixture = crate::test_support::TestDirectory::new("formal-audit");
        let root = fixture.path().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
        write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\n");
        write_fixture(&paths.seed_answers, "cigar\nrebut\n");
        write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");

        let left = FormalModel::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("model");
        let right = FormalModel::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("model");
        assert_eq!(left.manifest.manifest_hash, right.manifest.manifest_hash);
        atomic_write(
            &artifacts.small_state_table,
            &serde_json::to_vec_pretty(&left.small_state_table).expect("table"),
        )
        .expect("persist table");
        let reloaded = FormalModel::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("reloaded");
        assert_eq!(left.manifest.guess_hash, reloaded.manifest.guess_hash);
        assert_eq!(left.manifest.answer_hash, reloaded.manifest.answer_hash);
        assert_eq!(left.manifest.prior_hash, reloaded.manifest.prior_hash);
        assert_eq!(
            left.manifest.small_state_table_hash,
            reloaded.manifest.small_state_table_hash
        );
        assert_eq!(left.manifest.manifest_hash, reloaded.manifest.manifest_hash);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn tagged_state_round_trips_for_inline_and_bitset_states() {
        let answer_count = 96;
        let zobrist = build_zobrist_tokens(answer_count);
        for state in [
            StateKey::from_indices_with_tokens(answer_count, [1, 7, 15, 31], &zobrist),
            StateKey::from_indices_with_tokens(answer_count, 0..48, &zobrist),
        ] {
            let mut bytes = Vec::new();
            state
                .write_tagged(&mut bytes, answer_count)
                .expect("write tagged state");
            let restored = StateKey::read_tagged(&mut Cursor::new(bytes), answer_count, &zobrist)
                .expect("read tagged state");
            assert_eq!(restored, state);
            assert_eq!(restored.state_hash(), state.state_hash());
            assert_eq!(
                state_hash_for_tests(&restored),
                state_hash_for_tests(&state)
            );
        }
    }

    #[test]
    fn state_key_hashing_is_deterministic() {
        let answer_count = 64;
        let left_tokens = build_zobrist_tokens(answer_count);
        let right_tokens = build_zobrist_tokens(answer_count);
        let left = StateKey::from_indices_with_tokens(answer_count, [2, 9, 17, 33], &left_tokens);
        let right = StateKey::from_indices_with_tokens(answer_count, [2, 9, 17, 33], &right_tokens);
        assert_eq!(left, right);
        assert_eq!(left.state_hash(), right.state_hash());
        assert_eq!(state_hash_for_tests(&left), state_hash_for_tests(&right));
    }

    #[test]
    fn scratch_partition_matches_scalar_partition_for_local_frame_states() {
        let answer_count = 48;
        let words = synthetic_words(answer_count);
        let answers = synthetic_answers(&words);
        let fixture = crate::test_support::TestDirectory::new("formal-audit");
        let root = fixture.path().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("partition root");
        let table = PatternTable::load_or_build_at(&root.join("pattern.bin"), &words, &answers)
            .expect("pattern table");
        let prior = vec![1.0 / answer_count as f64; answer_count];
        let zobrist = build_zobrist_tokens(answer_count);
        let state = StateKey::from_indices_with_tokens(answer_count, 0..answer_count, &zobrist);
        let expected = partition_guess_scalar(answer_count, &state, 0, &table, &prior, &zobrist);
        let mut scratch = PartitionScratch::default();
        let actual = partition_guess_with_scratch(
            answer_count,
            &state,
            0,
            &table,
            &prior,
            &zobrist,
            &mut scratch,
        )
        .expect("scratch partition");
        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_eq!(left.pattern, right.pattern);
            assert_eq!(left.state, right.state);
            assert_eq!(left.count, right.count);
            assert!((left.mass - right.mass).abs() < 1e-12);
            assert!((left.entropy_bits - right.entropy_bits).abs() < 1e-12);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn subset_checks_cover_inline_and_bitset_storage_without_allocating_words() {
        let answer_count = 96;
        let zobrist = build_zobrist_tokens(answer_count);
        let inline_left = StateKey::from_indices_with_tokens(answer_count, [1, 3, 9], &zobrist);
        let inline_right =
            StateKey::from_indices_with_tokens(answer_count, [1, 3, 7, 9, 15], &zobrist);
        let inline_miss = StateKey::from_indices_with_tokens(answer_count, [1, 7, 55], &zobrist);
        let bitset_right = StateKey::from_indices_with_tokens(answer_count, 0..40, &zobrist);
        let bitset_left = StateKey::from_indices_with_tokens(answer_count, 10..45, &zobrist);

        assert!(state_is_subset_of(&inline_left, &inline_right));
        assert!(!state_is_subset_of(&inline_left, &inline_miss));
        assert!(state_is_subset_of(&inline_left, &bitset_right));
        assert!(!state_is_subset_of(&inline_miss, &bitset_right));
        assert!(state_is_subset_of(
            &StateKey::from_indices_with_tokens(answer_count, 10..20, &zobrist),
            &bitset_right,
        ));
        assert!(!state_is_subset_of(&bitset_left, &bitset_right));
        assert!(!state_is_subset_of(&bitset_right, &inline_right));
    }

    #[test]
    fn partition_dedup_requires_exact_bucket_match_within_fingerprint_bucket() {
        let answer_count = 96;
        let zobrist = build_zobrist_tokens(answer_count);
        let kept_buckets = vec![
            bucket_from_indices(answer_count, 1, [1, 3, 5], &zobrist),
            bucket_from_indices(answer_count, 2, [7, 9], &zobrist),
        ];
        let kept_plan = GuessQuickPlan {
            guess_index: 0,
            lower_bound: 1,
            max_bucket: 3,
            entropy: 0.0,
            solve_mass: 0.0,
            buckets: kept_buckets.clone(),
        };
        let fingerprint = PartitionFingerprint {
            bucket_count: kept_buckets.len() as u16,
            mix_a: 123,
            mix_b: 456,
        };
        let mut index = std::collections::HashMap::new();
        index.insert(fingerprint, vec![0]);

        assert!(partition_dedup_hit(
            &index,
            std::slice::from_ref(&kept_plan),
            fingerprint,
            &kept_buckets,
        ));
        assert!(!partition_dedup_hit(
            &index,
            &[kept_plan],
            fingerprint,
            &[
                bucket_from_indices(answer_count, 1, [1, 3, 5], &zobrist),
                bucket_from_indices(answer_count, 2, [7, 11], &zobrist),
            ],
        ));
    }

    #[test]
    fn certificate_verification_rejects_tampered_candidate_objective() {
        let fixture = crate::test_support::TestDirectory::new("formal-audit");
        let root = fixture.path().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
        write_fixture(
            &paths.seed_guesses,
            "cigar\nrebut\nsissy\nhumph\nzzzzz\nxxxxx\n",
        );
        write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
        write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");

        let _ = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("policy");
        let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("load");
        let mut certificate =
            read_proof_certificate(&paths, DEFAULT_FORMAL_MODEL_ID).expect("certificate");
        let exact = certificate
            .states
            .iter_mut()
            .flat_map(|state| &mut state.candidates)
            .find_map(|candidate| match &mut candidate.witness {
                PersistedCandidateWitness::Exact { objective, .. } => Some(objective),
                PersistedCandidateWitness::Infeasible
                | PersistedCandidateWitness::NonProgress { .. }
                | PersistedCandidateWitness::Equivalent { .. } => None,
            })
            .expect("exact candidate");
        exact.expected_guesses += 0.25;

        assert!(verify_certificate(&runtime, &certificate).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn certificate_verification_rejects_missing_or_tampered_structure() {
        let fixture = crate::test_support::TestDirectory::new("formal-audit");
        let root = fixture.path().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
        write_fixture(
            &paths.seed_guesses,
            "cigar\nrebut\nsissy\nhumph\nzzzzz\nxxxxx\n",
        );
        write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
        write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");
        build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("policy");
        let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("load");
        let certificate =
            read_proof_certificate(&paths, DEFAULT_FORMAL_MODEL_ID).expect("certificate");
        verify_certificate(&runtime, &certificate).expect("baseline certificate");

        let mut missing = certificate.clone();
        missing.states.pop();
        assert!(verify_certificate(&runtime, &missing).is_err());

        let state_with_child = certificate
            .states
            .iter()
            .position(|state| {
                state.candidates.iter().any(|candidate| {
                    matches!(
                        &candidate.witness,
                        PersistedCandidateWitness::Exact { children, .. }
                            if !children.is_empty()
                    )
                })
            })
            .expect("child state");
        let candidate_with_child = certificate.states[state_with_child]
            .candidates
            .iter()
            .position(|candidate| {
                matches!(
                    &candidate.witness,
                    PersistedCandidateWitness::Exact { children, .. } if !children.is_empty()
                )
            })
            .expect("candidate");

        let mut wrong_pattern = certificate.clone();
        let PersistedCandidateWitness::Exact { children, .. } =
            &mut wrong_pattern.states[state_with_child].candidates[candidate_with_child].witness
        else {
            unreachable!("selected exact candidate")
        };
        children[0].pattern = ALL_GREEN_PATTERN;
        assert!(verify_certificate(&runtime, &wrong_pattern).is_err());

        let mut wrong_child = certificate.clone();
        let PersistedCandidateWitness::Exact { children, .. } =
            &mut wrong_child.states[state_with_child].candidates[candidate_with_child].witness
        else {
            unreachable!("selected exact candidate")
        };
        children[0].child_state_id =
            ((children[0].child_state_id as usize + 1) % certificate.state_count) as u32;
        assert!(verify_certificate(&runtime, &wrong_child).is_err());

        let mut wrong_mass = certificate.clone();
        let PersistedCandidateWitness::Exact { children, .. } =
            &mut wrong_mass.states[state_with_child].candidates[candidate_with_child].witness
        else {
            unreachable!("selected exact candidate")
        };
        children[0].mass += 0.125;
        assert!(verify_certificate(&runtime, &wrong_mass).is_err());

        let mut wrong_child_objective = certificate.clone();
        let PersistedCandidateWitness::Exact { children, .. } =
            &mut wrong_child_objective.states[state_with_child].candidates[candidate_with_child]
                .witness
        else {
            unreachable!("selected exact candidate")
        };
        children[0].objective.expected_guesses += 0.25;
        assert!(verify_certificate(&runtime, &wrong_child_objective).is_err());

        let mut missing_candidate = certificate.clone();
        missing_candidate.states[state_with_child].candidates.pop();
        assert!(verify_certificate(&runtime, &missing_candidate).is_err());

        let mut duplicate_guess = certificate.clone();
        duplicate_guess.states[state_with_child].candidates[1].guess_index =
            duplicate_guess.states[state_with_child].candidates[0].guess_index;
        assert!(verify_certificate(&runtime, &duplicate_guess).is_err());

        let non_progress = certificate
            .states
            .iter()
            .enumerate()
            .find_map(|(state_index, state)| {
                state
                    .candidates
                    .iter()
                    .position(|candidate| {
                        matches!(
                            candidate.witness,
                            PersistedCandidateWitness::NonProgress { .. }
                        )
                    })
                    .map(|candidate_index| (state_index, candidate_index))
            })
            .expect("non-progress witness");
        let mut wrong_witness_type = certificate.clone();
        wrong_witness_type.states[non_progress.0].candidates[non_progress.1].witness =
            PersistedCandidateWitness::Exact {
                objective: PolicyObjective {
                    worst_case_depth: 1,
                    expected_guesses: 1.0,
                },
                children: Vec::new(),
            };
        assert!(verify_certificate(&runtime, &wrong_witness_type).is_err());

        let equivalent = certificate
            .states
            .iter()
            .enumerate()
            .find_map(|(state_index, state)| {
                state
                    .candidates
                    .iter()
                    .position(|candidate| {
                        matches!(
                            candidate.witness,
                            PersistedCandidateWitness::Equivalent { .. }
                        )
                    })
                    .map(|candidate_index| (state_index, candidate_index))
            })
            .expect("equivalent witness");
        let mut wrong_equivalent = certificate.clone();
        let candidate = &mut wrong_equivalent.states[equivalent.0].candidates[equivalent.1];
        candidate.witness = PersistedCandidateWitness::Equivalent {
            representative_guess: candidate.guess_index,
        };
        assert!(verify_certificate(&runtime, &wrong_equivalent).is_err());

        let mut wrong_state = certificate.clone();
        wrong_state.states[state_with_child]
            .answer_indices
            .reverse();
        assert!(verify_certificate(&runtime, &wrong_state).is_err());

        let mut wrong_decision = certificate;
        wrong_decision.states[0].best_guess =
            (wrong_decision.states[0].best_guess + 1) % runtime.model.guesses.len();
        assert!(verify_certificate(&runtime, &wrong_decision).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hot_tt_comparator_evicts_shallow_states_before_deeper_ones() {
        let answer_count = 96;
        let zobrist = build_zobrist_tokens(answer_count);
        let shallow = HotTtEntry {
            state: StateKey::from_indices_with_tokens(answer_count, 0..40, &zobrist),
            stored: StoredState {
                objective: PolicyObjective {
                    worst_case_depth: 2,
                    expected_guesses: 2.0,
                },
                best_guess: 0,
            },
            generation: 10,
        };
        let deep = HotTtEntry {
            state: StateKey::from_indices_with_tokens(answer_count, [1, 3, 5, 7, 9], &zobrist),
            stored: StoredState {
                objective: PolicyObjective {
                    worst_case_depth: 5,
                    expected_guesses: 5.0,
                },
                best_guess: 1,
            },
            generation: 1,
        };

        assert!(compare_hot_tt_entry(&shallow, &deep).is_lt());
        assert!(compare_hot_tt_entry(&deep, &shallow).is_gt());
    }

    #[test]
    fn toy_universe_matches_independent_solver() {
        let fixture = crate::test_support::TestDirectory::new("formal-audit");
        let root = fixture.path().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
        write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
        write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
        write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");

        let summary = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("policy");
        assert!(summary.solved_states > 0);
        let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("load");
        let root_state = runtime.initial_state();
        let exact = runtime.evaluate_state_exact(&root_state).expect("exact");
        let independent = runtime
            .solve_state_independent(&root_state)
            .expect("independent");
        assert!(same_decision(&exact, &independent));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn skewed_explicit_prior_builds_and_matches_independent_solver() {
        let fixture = crate::test_support::TestDirectory::new("formal-audit");
        let root = fixture.path().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
        write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
        write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
        write_fixture(
            &artifacts.prior_spec,
            "kind = \"explicit\"\n[weights]\ncigar = 0.40\nrebut = 0.59\nsissy = 0.01\n",
        );

        build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("policy");
        let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("load");
        let root_state = runtime.initial_state();
        let exact = runtime.evaluate_state_exact(&root_state).expect("exact");
        let independent = runtime
            .solve_state_independent(&root_state)
            .expect("independent");
        assert!(same_decision(&exact, &independent));
        let certificate =
            read_proof_certificate(&paths, DEFAULT_FORMAL_MODEL_ID).expect("certificate");
        verify_certificate(&runtime, &certificate).expect("verify skewed certificate");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn independently_verified_certificates_match_two_solvers_on_tractable_prefixes() {
        let words = [
            "cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade",
        ];
        for answer_count in 3..=7 {
            let fixture = crate::test_support::TestDirectory::new("formal-audit");
            let root = fixture.path().to_path_buf();
            let _ = std::fs::remove_dir_all(&root);
            let paths = ProjectPaths::new(&root);
            paths.ensure_layout().expect("layout");
            let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
            std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
            write_fixture(&paths.seed_guesses, &format!("{}\n", words.join("\n")));
            write_fixture(
                &paths.seed_answers,
                &format!("{}\n", words[..answer_count].join("\n")),
            );
            write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");

            build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("builder");
            let runtime =
                FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("runtime");
            let certificate =
                read_proof_certificate(&paths, DEFAULT_FORMAL_MODEL_ID).expect("certificate");
            verify_certificate(&runtime, &certificate).expect("witness verifier");
            let root_state = runtime.initial_state();
            let policy = runtime.policy.get(&root_state).expect("root policy");
            let slow = IndependentExactSolver::new(&runtime.model)
                .solve(&root_state)
                .expect("slow reference");
            assert!(same_decision(policy, &slow));
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn randomized_states_in_13_to_40_answer_universes_match_exhaustive_reference() {
        let words = "cigar rebut sissy humph awake blush focal evade naval serve heath dwarf model karma stink grade quiet bench abate feign major death fresh crust stool colon abase marry react batty pride floss helix croak staff paper unfed whelp trawl outdo adobe";
        let words = words.split_whitespace().collect::<Vec<_>>();
        for answer_count in [13usize, 24, 40] {
            let fixture = crate::test_support::TestDirectory::new("formal-audit");
            let root = fixture.path().to_path_buf();
            let _ = std::fs::remove_dir_all(&root);
            let paths = ProjectPaths::new(&root);
            paths.ensure_layout().expect("layout");
            let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
            std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
            write_fixture(&paths.seed_guesses, &format!("{}\n", words.join("\n")));
            write_fixture(
                &paths.seed_answers,
                &format!("{}\n", words[..answer_count].join("\n")),
            );
            write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");
            let model = FormalModel::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("model");
            let mut seed = answer_count as u64 * 0x9e37_79b9;
            let mut indices = HashSet::new();
            while indices.len() < 6 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                indices.insert((seed as usize) % answer_count);
            }
            let state = StateKey::from_indices_with_tokens(answer_count, indices, &model.zobrist);
            let started = Instant::now();
            let mut builder = FormalPolicyBuilder {
                model,
                memo: HashMap::new(),
                hot_tt: HotTranspositionTable::new(1024 * 1024),
                deduped_signatures: 0,
                bound_hits: 0,
                root_refinement_pruned: 0,
                local_refinement_pruned: 0,
                partition_calls: 0,
                quick_plan_calls: 0,
                started,
                last_progress: started,
                cancelled: None,
            };
            let built = builder.solve_state(&state).expect("builder");
            let exhaustive = IndependentExactSolver::new(&builder.model)
                .solve(&state)
                .expect("exhaustive");
            assert!(same_decision(&built, &exhaustive));
            assert_eq!(builder.root_refinement_pruned, 0);
            assert_eq!(builder.local_refinement_pruned, 0);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn parse_observations_rejects_length_mismatch() {
        let error = parse_observations(&["crane".into()], &[]).expect_err("must fail");
        assert!(error.to_string().contains("same number"));
    }

    #[test]
    fn apply_history_filters_answers() {
        let fixture = crate::test_support::TestDirectory::new("formal-audit");
        let root = fixture.path().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
        write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\n");
        write_fixture(&paths.seed_answers, "cigar\nrebut\n");
        write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");
        let _ = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("policy");
        let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("load");
        let observations = vec![(
            "cigar".to_string(),
            parse_feedback("ggggg").expect("feedback"),
        )];
        let state = runtime.apply_history(&observations).expect("state");
        assert_eq!(state.count(), 1);
        assert_eq!(state.indices(), vec![0]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn keeps_duplicate_letter_fixture_stable() {
        let pattern = crate::scoring::score_guess("lilly", "alley");
        assert_eq!(format_feedback_letters(pattern), "ybgbg");
        let _ = NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid");
    }
}
