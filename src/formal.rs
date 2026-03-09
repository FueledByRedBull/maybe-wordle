use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    data::{ProjectPaths, read_word_list},
    model::AnswerRecord,
    pattern_table::{PatternTable, hash_bytes, hash_word_list},
    scoring::{ALL_GREEN_PATTERN, PATTERN_SPACE, format_feedback_letters, parse_feedback},
    small_state::{SMALL_STATE_TABLE_VERSION, SmallStateTable},
};

pub const DEFAULT_FORMAL_MODEL_ID: &str = "formal-v1";
pub const DEFAULT_EXPECTED_ONLY_MODEL_ID: &str = "formal-expected-v1";
const PRIOR_SPEC_NAME: &str = "prior.toml";
const MANIFEST_NAME: &str = "manifest.json";
const VALUES_NAME: &str = "state_values.bin";
const POLICY_NAME: &str = "policy_table.bin";
const METADATA_NAME: &str = "proof_metadata.json";
const CERTIFICATE_NAME: &str = "proof_certificate.json";
const SMALL_STATE_TABLE_NAME: &str = "small_state_table.json";
const FORMAL_PATTERN_TABLE_NAME: &str = "pattern_table.bin";
const POLICY_MAGIC: &[u8; 8] = b"MWORDPV1";
const VALUES_MAGIC: &[u8; 8] = b"MWORDVV1";
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const OBJECTIVE_VERSION: u32 = 2;
const STATE_FORMAT_VERSION: u32 = 2;
const AUX_TABLE_VERSION: u32 = 2;
const CERTIFICATE_FORMAT_VERSION: u32 = 3;
const SMALL_STATE_LIMIT: usize = 12;
const ROOT_REFINEMENT_LIMIT: usize = 256;
const LOCAL_REFINEMENT_LIMIT: usize = 40;

type AnswerId = u16;
const INLINE_STATE_THRESHOLD: usize = 30;
const LOCAL_REINDEX_THRESHOLD: usize = 256;
const STATE_INLINE_CAPACITY: usize = INLINE_STATE_THRESHOLD;
const HOT_TT_BYTES: usize = 128 * 1024 * 1024;
const HOT_TT_ASSOCIATIVITY: usize = 4;
const STATE_TAG_INLINE: u8 = 0;
const STATE_TAG_BITSET: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FormalObjectiveKind {
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

#[derive(Clone, Debug)]
struct CertificateState {
    state: StateKey,
    best_guess: usize,
    best_objective: PolicyObjective,
    candidates: Vec<CertificateCandidate>,
}

#[derive(Clone, Debug)]
struct CertificateCandidate {
    guess_index: usize,
    objective: PolicyObjective,
    children: Vec<CertificateChild>,
}

#[derive(Clone, Debug)]
struct CertificateChild {
    pattern: u8,
    state: StateKey,
    objective: PolicyObjective,
    mass: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCertificateState {
    state_id: u32,
    best_guess: usize,
    best_objective: PolicyObjective,
    candidates: Vec<PersistedCertificateCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCertificateCandidate {
    guess_index: usize,
    objective: PolicyObjective,
    children: Vec<PersistedCertificateChild>,
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
    pub objective_id: String,
    pub objective: String,
    pub objective_version: u32,
    pub normal_mode_only: bool,
    pub guess_count: usize,
    pub answer_count: usize,
    pub guess_hash: u64,
    pub answer_hash: u64,
    pub prior_hash: u64,
    pub state_format_version: u32,
    pub aux_table_version: u32,
    pub certificate_format_version: u32,
    pub small_state_table_version: u32,
    pub small_state_table_hash: u64,
    pub manifest_hash: u64,
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
    pub manifest_hash: u64,
    pub solved_states: usize,
    pub deduped_signatures: u64,
    pub bound_hits: u64,
    pub root_refinement_pruned: u64,
    pub local_refinement_pruned: u64,
    pub build_millis: u128,
    pub root_objective: PolicyObjective,
}

#[derive(Clone, Debug)]
pub struct BuildOptimalSummary {
    pub model_id: String,
    pub manifest_hash: u64,
    pub solved_states: usize,
    pub deduped_signatures: u64,
    pub bound_hits: u64,
    pub root_refinement_pruned: u64,
    pub local_refinement_pruned: u64,
    pub build_millis: u128,
    pub root_best_guess: String,
    pub root_objective: PolicyObjective,
}

#[derive(Clone, Debug)]
pub struct VerifySummary {
    pub mode: FormalVerificationMode,
    pub verified_cached_states: usize,
    pub verified_small_states: usize,
    pub verified_medium_states: usize,
    pub model_id: String,
    pub manifest_hash: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofCertificate {
    pub model_id: String,
    pub manifest_hash: u64,
    pub objective_id: String,
    pub objective_version: u32,
    pub state_format_version: u32,
    pub aux_table_version: u32,
    pub certificate_format_version: u32,
    pub small_state_table_hash: u64,
    pub state_count: usize,
    states: Vec<PersistedCertificateState>,
}

#[derive(Clone, Debug)]
pub struct FormalStateExplanation {
    pub model_id: String,
    pub manifest_hash: u64,
    pub surviving_answers: usize,
    pub best_guess: String,
    pub objective: PolicyObjective,
    pub bucket_sizes: Vec<usize>,
    pub tied_moves: Vec<FormalSuggestion>,
}

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
}

#[derive(Clone, Debug)]
pub struct FormalPolicyRuntime {
    model: FormalModel,
    policy: HashMap<StateKey, StoredState>,
    ordered_states: Vec<StateKey>,
    state_ids: HashMap<StateKey, u32>,
    metadata: ProofMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateKey {
    storage: StateStorage,
    count: usize,
    hash: u64,
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

#[derive(Clone, Debug)]
struct FormalPolicyBuilder {
    model: FormalModel,
    memo: HashMap<StateKey, StoredState>,
    certificate_states: HashMap<StateKey, CertificateState>,
    hot_tt: HotTranspositionTable,
    deduped_signatures: u64,
    bound_hits: u64,
    root_refinement_pruned: u64,
    local_refinement_pruned: u64,
    partition_calls: u64,
    quick_plan_calls: u64,
    started: Instant,
    last_progress: Instant,
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

#[derive(Clone, Debug)]
struct IndependentExactSolver<'a> {
    model: &'a FormalModel,
    seed_cache: Option<&'a HashMap<StateKey, StoredState>>,
    local_memo: HashMap<StateKey, StoredState>,
    scratch: PartitionScratch,
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
    fn new(model: &'a FormalModel, seed_cache: Option<&'a HashMap<StateKey, StoredState>>) -> Self {
        Self {
            model,
            seed_cache,
            local_memo: HashMap::new(),
            scratch: PartitionScratch::default(),
        }
    }

    fn solve(&mut self, state: &StateKey) -> Result<StoredState> {
        if let Some(existing) = self.local_memo.get(state) {
            return Ok(existing.clone());
        }
        if let Some(existing) = self.seed_cache.and_then(|cache| cache.get(state)).cloned() {
            return Ok(existing);
        }
        if state.count() == 1 {
            let stored = singleton_state_for_model(self.model, state)?;
            self.local_memo.insert(state.clone(), stored.clone());
            return Ok(stored);
        }

        let state_indices = state.indices();
        let total_mass = state_indices
            .iter()
            .map(|index| self.model.prior[*index])
            .sum::<f64>();
        let mut best: Option<StoredState> = None;
        for guess_index in 0..self.model.guesses.len() {
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
                .any(|bucket| bucket.pattern != ALL_GREEN_PATTERN && bucket.state == *state)
            {
                continue;
            }
            let mut worst_case = 1u8;
            let mut expected = 1.0;
            for bucket in buckets {
                if bucket.pattern == ALL_GREEN_PATTERN {
                    continue;
                }
                let child = self.solve(&bucket.state)?;
                worst_case = worst_case.max(1 + child.objective.worst_case_depth);
                expected += (bucket.mass / total_mass) * child.objective.expected_guesses;
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
                    self.model.objective_spec.kind,
                )
                .is_lt()
            }) {
                best = Some(candidate);
            }
        }
        let best = best
            .ok_or_else(|| anyhow!("state {} had no independent candidates", state.state_hash()))?;
        self.local_memo.insert(state.clone(), best.clone());
        Ok(best)
    }
}

pub fn build_optimal_policy(paths: &ProjectPaths, model_id: &str) -> Result<BuildOptimalSummary> {
    let model = FormalModel::load(paths, model_id)?;
    let root = StateKey::full(model.answers.len(), &model.zobrist);
    let started = Instant::now();
    let mut builder = FormalPolicyBuilder {
        model,
        memo: HashMap::new(),
        certificate_states: HashMap::new(),
        hot_tt: HotTranspositionTable::new(HOT_TT_BYTES),
        deduped_signatures: 0,
        bound_hits: 0,
        root_refinement_pruned: 0,
        local_refinement_pruned: 0,
        partition_calls: 0,
        quick_plan_calls: 0,
        started,
        last_progress: started,
    };
    let _ = builder.solve_state(&root)?;
    builder.force_report_progress("root_complete");
    builder.materialize_policy_reachable_states(&root)?;
    builder.force_report_progress("policy_materialized");
    let root_state = builder
        .memo
        .get(&root)
        .cloned()
        .ok_or_else(|| anyhow!("root state missing after materialization"))?;
    let build_millis = started.elapsed().as_millis();
    let metadata = ProofMetadata {
        model_id: builder.model.manifest.model_id.clone(),
        manifest_hash: builder.model.manifest.manifest_hash,
        solved_states: builder.memo.len(),
        deduped_signatures: builder.deduped_signatures,
        bound_hits: builder.bound_hits,
        root_refinement_pruned: builder.root_refinement_pruned,
        local_refinement_pruned: builder.local_refinement_pruned,
        build_millis,
        root_objective: root_state.objective.clone(),
    };
    persist_policy(
        &builder.model,
        &builder.memo,
        &builder.certificate_states,
        &metadata,
        paths,
    )?;
    Ok(BuildOptimalSummary {
        model_id: builder.model.manifest.model_id.clone(),
        manifest_hash: builder.model.manifest.manifest_hash,
        solved_states: builder.memo.len(),
        deduped_signatures: builder.deduped_signatures,
        bound_hits: builder.bound_hits,
        root_refinement_pruned: builder.root_refinement_pruned,
        local_refinement_pruned: builder.local_refinement_pruned,
        build_millis,
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
    let runtime = FormalPolicyRuntime::load(paths, model_id)?;
    let certificate = read_proof_certificate(paths, model_id)?;
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
        verify_certificate(&runtime, &certificate)?;
        cached_states = certificate.state_count;
    }
    let mut small_states = 0usize;
    let mut medium_states = 0usize;
    if mode == FormalVerificationMode::Oracle {
        for (state, stored) in &runtime.policy {
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
        verified_cached_states: cached_states,
        verified_small_states: small_states,
        verified_medium_states: medium_states,
        model_id: runtime.model.manifest.model_id.clone(),
        manifest_hash: runtime.model.manifest.manifest_hash,
    })
}

pub fn artifacts_exist(paths: &ProjectPaths, model_id: &str) -> bool {
    PolicyArtifactSet::for_model(paths, model_id).exists()
}

fn objective_spec_for_model(model_id: &str) -> FormalObjectiveSpec {
    if model_id.contains("expected") {
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
        if let Some(parent) = artifacts.prior_spec.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let guesses = read_word_list(&paths.seed_guesses)
            .with_context(|| format!("failed to load {}", paths.seed_guesses.display()))?;
        let answers = read_word_list(&paths.seed_answers)
            .with_context(|| format!("failed to load {}", paths.seed_answers.display()))?;
        let raw_prior = fs::read(&artifacts.prior_spec)
            .with_context(|| format!("failed to read {}", artifacts.prior_spec.display()))?;
        let prior_spec: FormalPriorSpec = toml::from_str(
            std::str::from_utf8(&raw_prior).context("formal prior spec must be valid UTF-8")?,
        )
        .with_context(|| format!("failed to parse {}", artifacts.prior_spec.display()))?;
        let guess_hash = hash_word_list(guesses.iter().map(String::as_str));
        let answer_hash = hash_word_list(answers.iter().map(String::as_str));
        let prior_hash = hash_bytes(1469598103934665603, &raw_prior);
        let objective_spec = objective_spec_for_model(model_id);
        let (small_state_table, small_state_table_bytes) = if artifacts.small_state_table.exists() {
            let raw = fs::read(&artifacts.small_state_table).with_context(|| {
                format!("failed to read {}", artifacts.small_state_table.display())
            })?;
            let table = serde_json::from_slice(&raw).with_context(|| {
                format!("failed to parse {}", artifacts.small_state_table.display())
            })?;
            (table, raw)
        } else {
            let table = SmallStateTable::build(SMALL_STATE_LIMIT);
            let raw = serde_json::to_vec_pretty(&table).context("small-state table serialize")?;
            (table, raw)
        };
        let small_state_table_hash = hash_bytes(1469598103934665603, &small_state_table_bytes);
        let manifest_hash = combine_hashes(
            guess_hash,
            answer_hash,
            prior_hash,
            objective_spec,
            small_state_table_hash,
        );
        let manifest = FormalManifest {
            model_id: model_id.to_string(),
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
        let prior = build_prior(&answers, prior_spec)?;
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
        let pattern_table =
            PatternTable::load_or_build_at(&artifacts.pattern_table, &guesses, &answer_records)?;
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
        })
    }
}

impl FormalPolicyRuntime {
    pub fn load(paths: &ProjectPaths, model_id: &str) -> Result<Self> {
        let model = FormalModel::load(paths, model_id)?;
        let artifacts = PolicyArtifactSet::for_model(paths, model_id);
        let manifest: FormalManifest = serde_json::from_reader(BufReader::new(
            File::open(&artifacts.manifest)
                .with_context(|| format!("failed to open {}", artifacts.manifest.display()))?,
        ))
        .with_context(|| format!("failed to parse {}", artifacts.manifest.display()))?;
        if manifest.manifest_hash != model.manifest.manifest_hash {
            bail!(
                "formal artifacts are stale for {}: expected manifest {}, found {}",
                model_id,
                model.manifest.manifest_hash,
                manifest.manifest_hash
            );
        }
        let metadata: ProofMetadata = serde_json::from_reader(BufReader::new(
            File::open(&artifacts.metadata)
                .with_context(|| format!("failed to open {}", artifacts.metadata.display()))?,
        ))
        .with_context(|| format!("failed to parse {}", artifacts.metadata.display()))?;
        let values = read_values(&artifacts.values, &model)?;
        let policies = read_policy(&artifacts.policy, &model)?;
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

        Ok(Self {
            model,
            policy,
            ordered_states,
            state_ids,
            metadata,
        })
    }

    pub fn initial_state(&self) -> StateKey {
        StateKey::full(self.model.answers.len(), &self.model.zobrist)
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
        Ok(next)
    }

    pub fn suggest(&self, state: &StateKey, top: usize) -> Result<Vec<FormalSuggestion>> {
        let mut evaluations = self.evaluate_state_ranked(state)?;
        evaluations.truncate(top);
        Ok(evaluations
            .into_iter()
            .map(|evaluation| FormalSuggestion {
                word: self.model.guesses[evaluation.guess_index].clone(),
                objective: evaluation.objective,
                bucket_sizes: evaluation.bucket_sizes,
            })
            .collect())
    }

    pub fn explain_state(&self, state: &StateKey, top: usize) -> Result<FormalStateExplanation> {
        let ranked = self.evaluate_state_ranked(state)?;
        let best = ranked
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("state {} is missing evaluations", state.state_hash()))?;
        let tied_moves = ranked.into_iter().take(top).collect::<Vec<_>>();
        Ok(FormalStateExplanation {
            model_id: self.model.manifest.model_id.clone(),
            manifest_hash: self.model.manifest.manifest_hash,
            surviving_answers: state.count(),
            best_guess: self.model.guesses[best.guess_index].clone(),
            objective: best.objective.clone(),
            bucket_sizes: best.bucket_sizes.clone(),
            tied_moves: tied_moves
                .into_iter()
                .map(|candidate| FormalSuggestion {
                    word: self.model.guesses[candidate.guess_index].clone(),
                    objective: candidate.objective,
                    bucket_sizes: candidate.bucket_sizes,
                })
                .collect(),
        })
    }

    pub fn metadata(&self) -> &ProofMetadata {
        &self.metadata
    }

    pub fn manifest(&self) -> &FormalManifest {
        &self.model.manifest
    }

    fn evaluate_state_ranked(&self, state: &StateKey) -> Result<Vec<GuessEvaluation>> {
        let total_mass = self.state_mass(state);
        let mut scratch = PartitionScratch::default();
        let mut signature_map: HashMap<PartitionFingerprint, Vec<usize>> = HashMap::new();
        let mut evaluations = Vec::new();
        let mut evaluation_buckets: Vec<Vec<PartitionBucket>> = Vec::new();
        for guess_index in 0..self.model.guesses.len() {
            let buckets = partition_guess_with_scratch(
                self.model.answers.len(),
                state,
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
                self.build_guess_evaluation(state, guess_index, total_mass, &buckets, false)?
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
        evaluations.sort_by(|left, right| {
            compare_evaluations_with_kind(
                left,
                right,
                &self.model.guesses,
                self.model.objective_spec.kind,
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
        IndependentExactSolver::new(&self.model, Some(&self.policy)).solve(state)
    }

    fn build_guess_evaluation(
        &self,
        state: &StateKey,
        guess_index: usize,
        total_mass: f64,
        buckets: &[PartitionBucket],
        use_cache_only: bool,
    ) -> Result<Option<GuessEvaluation>> {
        if buckets
            .iter()
            .any(|bucket| bucket.pattern != ALL_GREEN_PATTERN && bucket.state == *state)
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
            let stored = if use_cache_only {
                match self.policy.get(&bucket.state).cloned() {
                    Some(stored) => stored,
                    None => return Ok(None),
                }
            } else {
                self.solve_state_independent(&bucket.state)?
            };
            worst_case = worst_case.max(1 + stored.objective.worst_case_depth);
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

impl FormalPolicyBuilder {
    fn solve_state(&mut self, state: &StateKey) -> Result<StoredState> {
        if let Some(existing) = self.memo.get(state) {
            return Ok(existing.clone());
        }
        let quick_plans = self.collect_quick_plans_for_state(state)?;
        let lower_bound = quick_plans
            .first()
            .map(|plan| plan.lower_bound)
            .unwrap_or(1);
        for target_depth in lower_bound..=u8::MAX {
            let upper = PolicyObjective {
                worst_case_depth: target_depth,
                expected_guesses: f64::INFINITY,
            };
            if let Some(best) = self.solve_state_with_bound(state, &quick_plans, &upper)? {
                return Ok(best);
            }
        }
        bail!("state {} exceeded bounded depth search", state.state_hash())
    }

    fn solve_state_with_upper(
        &mut self,
        state: &StateKey,
        upper: &PolicyObjective,
    ) -> Result<Option<StoredState>> {
        if let Some(existing) = self.memo.get(state) {
            return Ok(
                objective_le(&existing.objective, upper, self.model.objective_spec.kind)
                    .then_some(existing.clone()),
            );
        }
        if let Some(existing) = self.hot_tt.get(state) {
            return Ok(
                objective_le(&existing.objective, upper, self.model.objective_spec.kind)
                    .then_some(existing),
            );
        }
        let quick_plans = self.collect_quick_plans_for_state(state)?;
        self.solve_state_with_bound(state, &quick_plans, upper)
    }

    fn solve_state_with_bound(
        &mut self,
        state: &StateKey,
        quick_plans: &[GuessQuickPlan],
        upper: &PolicyObjective,
    ) -> Result<Option<StoredState>> {
        if let Some(existing) = self.memo.get(state) {
            return Ok(
                objective_le(&existing.objective, upper, self.model.objective_spec.kind)
                    .then_some(existing.clone()),
            );
        }
        if let Some(existing) = self.hot_tt.get(state) {
            return Ok(
                objective_le(&existing.objective, upper, self.model.objective_spec.kind)
                    .then_some(existing),
            );
        }
        if state.count() == 1 {
            let stored = singleton_state_for_model(&self.model, state)?;
            self.hot_tt.insert(state.clone(), stored.clone());
            return Ok(
                objective_le(&stored.objective, upper, self.model.objective_spec.kind)
                    .then_some(stored),
            );
        }
        if state.count() <= self.model.small_state_table.max_size {
            let exact = self.solve_small_state_exact(state)?;
            self.hot_tt.insert(state.clone(), exact.clone());
            return Ok(
                objective_le(&exact.objective, upper, self.model.objective_spec.kind)
                    .then_some(exact),
            );
        }

        let state_lower_bound = quick_plans
            .first()
            .map(|plan| plan.lower_bound)
            .unwrap_or(1);
        if state_lower_bound > upper.worst_case_depth {
            self.bound_hits += 1;
            return Ok(None);
        }
        let total_mass = self.state_mass(state);
        let mut best: Option<StoredState> = None;
        let mut best_certificate: Option<CertificateState> = None;
        for plan in quick_plans {
            let effective_upper = best
                .as_ref()
                .map(|stored| {
                    min_objective(upper, &stored.objective, self.model.objective_spec.kind)
                })
                .unwrap_or_else(|| upper.clone());
            if plan.lower_bound > effective_upper.worst_case_depth {
                self.bound_hits += 1;
                continue;
            }
            let expected_lower_bound = guess_expected_lower_bound(
                &plan.buckets,
                total_mass,
                PATTERN_SPACE as f64,
                &self.model.small_state_table,
            );
            let lower_objective = PolicyObjective {
                worst_case_depth: plan.lower_bound,
                expected_guesses: expected_lower_bound,
            };
            if objective_ge(
                &lower_objective,
                &effective_upper,
                self.model.objective_spec.kind,
            ) {
                self.bound_hits += 1;
                continue;
            }
            let mut children = Vec::new();
            let mut remaining_lower = plan
                .buckets
                .iter()
                .filter(|bucket| bucket.pattern != ALL_GREEN_PATTERN)
                .map(|bucket| {
                    (bucket.mass / total_mass)
                        * child_expected_lower_bound(
                            bucket,
                            PATTERN_SPACE as f64,
                            &self.model.small_state_table,
                        )
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
            unresolved.sort_unstable_by(|left, right| right.count.cmp(&left.count));
            for bucket in unresolved {
                let probability = bucket.mass / total_mass;
                remaining_lower -= probability
                    * child_expected_lower_bound(
                        &bucket,
                        PATTERN_SPACE as f64,
                        &self.model.small_state_table,
                    );
                let child_upper = PolicyObjective {
                    worst_case_depth: effective_upper.worst_case_depth.saturating_sub(1),
                    expected_guesses: f64::INFINITY,
                };
                let Some(child) = self.solve_state_with_upper(&bucket.state, &child_upper)? else {
                    valid = false;
                    self.bound_hits += 1;
                    break;
                };
                worst_case = worst_case.max(1 + child.objective.worst_case_depth);
                if worst_case > effective_upper.worst_case_depth {
                    valid = false;
                    self.bound_hits += 1;
                    break;
                }
                expected += probability * child.objective.expected_guesses;
                if worst_case == effective_upper.worst_case_depth
                    && expected + remaining_lower >= effective_upper.expected_guesses
                {
                    valid = false;
                    self.bound_hits += 1;
                    break;
                }
                children.push(CertificateChild {
                    pattern: bucket.pattern,
                    state: bucket.state.clone(),
                    objective: child.objective.clone(),
                    mass: bucket.mass,
                });
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
            let certificate = CertificateCandidate {
                guess_index: plan.guess_index,
                objective: candidate.objective.clone(),
                children,
            };
            if best.as_ref().is_none_or(|current| {
                compare_stored_with_kind(
                    &candidate,
                    current,
                    &self.model.guesses,
                    self.model.objective_spec.kind,
                )
                .is_lt()
            }) {
                best_certificate = Some(CertificateState {
                    state: state.clone(),
                    best_guess: plan.guess_index,
                    best_objective: candidate.objective.clone(),
                    candidates: vec![certificate],
                });
                best = Some(candidate);
            } else if let Some(existing) = &mut best_certificate {
                existing.candidates.push(certificate);
            }
        }
        if let (Some(stored), Some(certificate)) = (best, best_certificate) {
            self.hot_tt.insert(state.clone(), stored.clone());
            self.certificate_states.insert(state.clone(), certificate);
            return Ok(
                objective_le(&stored.objective, upper, self.model.objective_spec.kind)
                    .then_some(stored),
            );
        }
        Ok(None)
    }

    fn solve_small_state_exact(&mut self, state: &StateKey) -> Result<StoredState> {
        IndependentExactSolver::new(&self.model, Some(&self.memo)).solve(state)
    }

    fn materialize_policy_reachable_states(&mut self, root: &StateKey) -> Result<()> {
        let mut frontier = vec![root.clone()];
        let mut seen = HashSet::new();
        while let Some(state) = frontier.pop() {
            if !seen.insert(state.clone()) || self.memo.contains_key(&state) {
                continue;
            }
            self.maybe_report_progress("materialization");
            let stored = self.solve_state(&state)?;
            self.memo.insert(state.clone(), stored.clone());
            self.certificate_states
                .entry(state.clone())
                .or_insert_with(|| trivial_certificate_state(&state, &stored));
            let buckets = self.partition_guess(&state, stored.best_guess)?;
            for bucket in buckets {
                if bucket.pattern != ALL_GREEN_PATTERN {
                    frontier.push(bucket.state);
                }
            }
        }
        Ok(())
    }

    fn collect_quick_plans_for_state(&mut self, state: &StateKey) -> Result<Vec<GuessQuickPlan>> {
        self.quick_plan_calls += 1;
        self.maybe_report_progress("search");
        let total_mass = self.state_mass(state);
        let raw_plans = (0..self.model.guesses.len())
            .into_par_iter()
            .map_init(
                PartitionScratch::default,
                |scratch, guess_index| -> Result<Option<(GuessQuickPlan, PartitionFingerprint)>> {
                    let buckets = partition_guess_with_scratch(
                        self.model.answers.len(),
                        state,
                        guess_index,
                        &self.model.pattern_table,
                        &self.model.prior,
                        &self.model.zobrist,
                        scratch,
                    )?;
                    if buckets
                        .iter()
                        .any(|bucket| bucket.pattern != ALL_GREEN_PATTERN && bucket.state == *state)
                    {
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
        let (pruned, pruned_count) = prune_refined_guesses(
            &plans,
            if state.count() == self.model.answers.len() {
                ROOT_REFINEMENT_LIMIT
            } else {
                LOCAL_REFINEMENT_LIMIT
            },
        );
        if state.count() == self.model.answers.len() {
            self.root_refinement_pruned += pruned_count as u64;
        } else {
            self.local_refinement_pruned += pruned_count as u64;
        }
        Ok(pruned)
    }

    fn partition_guess(
        &mut self,
        state: &StateKey,
        guess_index: usize,
    ) -> Result<Vec<PartitionBucket>> {
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
        let tag = read_u8(reader)?;
        match tag {
            STATE_TAG_INLINE => {
                let len = read_u16(reader)? as usize;
                let mut indices = Vec::with_capacity(len);
                for _ in 0..len {
                    indices.push(read_u16(reader)? as usize);
                }
                Ok(Self::from_indices_with_tokens(
                    answer_count,
                    indices,
                    zobrist,
                ))
            }
            STATE_TAG_BITSET => {
                let word_count = read_u16(reader)? as usize;
                if word_count != answer_count.div_ceil(64) {
                    bail!("unexpected tagged state word count");
                }
                let words = read_state_words(reader, word_count)?;
                Ok(Self::from_words_with_tokens(words, zobrist))
            }
            _ => bail!("invalid state tag {}", tag),
        }
    }

    fn cmp_storage(&self, other: &Self, answer_count: usize) -> std::cmp::Ordering {
        let left = self.as_words(answer_count);
        let right = other.as_words(answer_count);
        left.cmp(&right)
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
            .map(|answer| weights.get(answer).copied().unwrap_or(0.0))
            .collect::<Vec<_>>(),
    };
    let total = weights.iter().sum::<f64>();
    if total <= 0.0 {
        bail!("formal prior must assign positive total probability mass");
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

fn persist_policy(
    model: &FormalModel,
    memo: &HashMap<StateKey, StoredState>,
    certificate_states: &HashMap<StateKey, CertificateState>,
    metadata: &ProofMetadata,
    paths: &ProjectPaths,
) -> Result<()> {
    let artifacts = PolicyArtifactSet::for_model(paths, &model.manifest.model_id);
    fs::create_dir_all(&artifacts.model_dir)
        .with_context(|| format!("failed to create {}", artifacts.model_dir.display()))?;
    serde_json::to_writer_pretty(
        BufWriter::new(
            File::create(&artifacts.manifest)
                .with_context(|| format!("failed to create {}", artifacts.manifest.display()))?,
        ),
        &model.manifest,
    )
    .with_context(|| format!("failed to write {}", artifacts.manifest.display()))?;
    serde_json::to_writer_pretty(
        BufWriter::new(
            File::create(&artifacts.metadata)
                .with_context(|| format!("failed to create {}", artifacts.metadata.display()))?,
        ),
        metadata,
    )
    .with_context(|| format!("failed to write {}", artifacts.metadata.display()))?;
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(&artifacts.small_state_table).with_context(|| {
            format!("failed to create {}", artifacts.small_state_table.display())
        })?),
        &model.small_state_table,
    )
    .with_context(|| format!("failed to write {}", artifacts.small_state_table.display()))?;
    let mut certificate_entries = certificate_states.iter().collect::<Vec<_>>();
    certificate_entries.sort_by(|(left_key, _), (right_key, _)| {
        left_key
            .state_hash()
            .cmp(&right_key.state_hash())
            .then_with(|| left_key.cmp_storage(right_key, model.answers.len()))
    });
    let mut entries = memo.iter().collect::<Vec<_>>();
    entries.sort_by(|(left_key, _), (right_key, _)| {
        left_key
            .state_hash()
            .cmp(&right_key.state_hash())
            .then_with(|| left_key.cmp_storage(right_key, model.answers.len()))
    });
    let state_ids = entries
        .iter()
        .enumerate()
        .map(|(index, (state, _))| ((*state).clone(), index as u32))
        .collect::<HashMap<_, _>>();
    let certificate = ProofCertificate {
        model_id: model.manifest.model_id.clone(),
        manifest_hash: model.manifest.manifest_hash,
        objective_id: model.manifest.objective_id.clone(),
        objective_version: model.manifest.objective_version,
        state_format_version: model.manifest.state_format_version,
        aux_table_version: model.manifest.aux_table_version,
        certificate_format_version: model.manifest.certificate_format_version,
        small_state_table_hash: model.manifest.small_state_table_hash,
        state_count: memo.len(),
        states: certificate_entries
            .into_iter()
            .map(|(_, state)| PersistedCertificateState {
                state_id: state_ids[&state.state],
                best_guess: state.best_guess,
                best_objective: state.best_objective.clone(),
                candidates: state
                    .candidates
                    .iter()
                    .map(|candidate| PersistedCertificateCandidate {
                        guess_index: candidate.guess_index,
                        objective: candidate.objective.clone(),
                        children: candidate
                            .children
                            .iter()
                            .map(|child| PersistedCertificateChild {
                                pattern: child.pattern,
                                child_state_id: state_ids[&child.state],
                                objective: child.objective.clone(),
                                mass: child.mass,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
    };
    serde_json::to_writer_pretty(
        BufWriter::new(
            File::create(&artifacts.certificate)
                .with_context(|| format!("failed to create {}", artifacts.certificate.display()))?,
        ),
        &certificate,
    )
    .with_context(|| format!("failed to write {}", artifacts.certificate.display()))?;

    write_values(&artifacts.values, model, &entries)?;
    write_policy(&artifacts.policy, model, &entries)?;
    Ok(())
}

fn write_values(
    path: &Path,
    model: &FormalModel,
    entries: &[(&StateKey, &StoredState)],
) -> Result<()> {
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    writer.write_all(VALUES_MAGIC)?;
    writer.write_all(&model.manifest.manifest_hash.to_le_bytes())?;
    writer.write_all(&(entries.len() as u64).to_le_bytes())?;
    writer.write_all(&(model.answers.len().div_ceil(64) as u32).to_le_bytes())?;
    for (state, stored) in entries {
        writer.write_all(&state.state_hash().to_le_bytes())?;
        state.write_tagged(&mut writer, model.answers.len())?;
        writer.write_all(&[stored.objective.worst_case_depth])?;
        writer.write_all(&stored.objective.expected_guesses.to_le_bytes())?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn write_policy(
    path: &Path,
    model: &FormalModel,
    entries: &[(&StateKey, &StoredState)],
) -> Result<()> {
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    writer.write_all(POLICY_MAGIC)?;
    writer.write_all(&model.manifest.manifest_hash.to_le_bytes())?;
    writer.write_all(&(entries.len() as u64).to_le_bytes())?;
    writer.write_all(&(model.answers.len().div_ceil(64) as u32).to_le_bytes())?;
    for (state, stored) in entries {
        writer.write_all(&state.state_hash().to_le_bytes())?;
        state.write_tagged(&mut writer, model.answers.len())?;
        writer.write_all(&(stored.best_guess as u32).to_le_bytes())?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn read_values(path: &Path, model: &FormalModel) -> Result<HashMap<StateKey, PolicyObjective>> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    if &header != VALUES_MAGIC {
        bail!("invalid values file magic: {}", path.display());
    }
    let manifest_hash = read_u64(&mut reader)?;
    if manifest_hash != model.manifest.manifest_hash {
        bail!("stale values file: {}", path.display());
    }
    let count = read_u64(&mut reader)? as usize;
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
        if state.state_hash() != state_hash {
            bail!("state hash mismatch in {}", path.display());
        }
        values.insert(
            state,
            PolicyObjective {
                worst_case_depth,
                expected_guesses,
            },
        );
    }
    Ok(values)
}

fn read_policy(path: &Path, model: &FormalModel) -> Result<HashMap<StateKey, usize>> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    if &header != POLICY_MAGIC {
        bail!("invalid policy file magic: {}", path.display());
    }
    let manifest_hash = read_u64(&mut reader)?;
    if manifest_hash != model.manifest.manifest_hash {
        bail!("stale policy file: {}", path.display());
    }
    let count = read_u64(&mut reader)? as usize;
    let word_count = read_u32(&mut reader)? as usize;
    if word_count != model.answers.len().div_ceil(64) {
        bail!("unexpected state word count in {}", path.display());
    }
    let mut policies = HashMap::with_capacity(count);
    for _ in 0..count {
        let state_hash = read_u64(&mut reader)?;
        let state = StateKey::read_tagged(&mut reader, model.answers.len(), &model.zobrist)?;
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
        policies.insert(state, best_guess);
    }
    Ok(policies)
}

fn read_state_words(reader: &mut impl Read, word_count: usize) -> Result<Vec<u64>> {
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        words.push(read_u64(reader)?);
    }
    Ok(words)
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

fn read_proof_certificate(paths: &ProjectPaths, model_id: &str) -> Result<ProofCertificate> {
    let artifacts = PolicyArtifactSet::for_model(paths, model_id);
    serde_json::from_reader(BufReader::new(
        File::open(&artifacts.certificate)
            .with_context(|| format!("failed to open {}", artifacts.certificate.display()))?,
    ))
    .with_context(|| format!("failed to parse {}", artifacts.certificate.display()))
}

fn verify_certificate(runtime: &FormalPolicyRuntime, certificate: &ProofCertificate) -> Result<()> {
    if certificate.states.len() != certificate.state_count {
        bail!(
            "proof certificate state count mismatch: header={} payload={}",
            certificate.state_count,
            certificate.states.len()
        );
    }
    for state in &certificate.states {
        let key = runtime
            .ordered_states
            .get(state.state_id as usize)
            .ok_or_else(|| anyhow!("certificate state id {} out of range", state.state_id))?;
        if runtime.state_ids.get(key).copied() != Some(state.state_id) {
            bail!(
                "runtime state row mapping mismatch for state {}",
                state.state_id
            );
        }
        let stored = runtime
            .policy
            .get(key)
            .ok_or_else(|| anyhow!("certificate references unknown state {}", state.state_id))?;
        if stored.best_guess != state.best_guess
            || !same_objective(&stored.objective, &state.best_objective)
        {
            bail!(
                "certificate best decision mismatch for state {}",
                state.state_id
            );
        }
        let total_mass = state_total_mass(key, &runtime.model.prior);
        if total_mass <= 0.0 {
            bail!(
                "certificate state {} has non-positive total mass",
                state.state_id
            );
        }
        let mut saw_best = false;
        for candidate in &state.candidates {
            if candidate.guess_index == state.best_guess
                && same_objective(&candidate.objective, &state.best_objective)
            {
                saw_best = true;
            }
            let mut seen_patterns = [false; PATTERN_SPACE];
            let mut recomputed = PolicyObjective {
                worst_case_depth: 1,
                expected_guesses: 1.0,
            };
            let should_rederive = !candidate.children.is_empty() || key.count() == 1;
            for child in &candidate.children {
                if child.pattern == ALL_GREEN_PATTERN {
                    bail!(
                        "certificate child unexpectedly uses all-green pattern for state {}",
                        state.state_id
                    );
                }
                let pattern_index = child.pattern as usize;
                if seen_patterns[pattern_index] {
                    bail!(
                        "certificate has duplicate child pattern {} for state {}",
                        child.pattern,
                        state.state_id
                    );
                }
                seen_patterns[pattern_index] = true;
                let child_key = runtime
                    .ordered_states
                    .get(child.child_state_id as usize)
                    .ok_or_else(|| {
                        anyhow!("certificate child id {} out of range", child.child_state_id)
                    })?;
                if runtime.state_ids.get(child_key).copied() != Some(child.child_state_id) {
                    bail!(
                        "runtime child row mapping mismatch for state {}",
                        child.child_state_id
                    );
                }
                let child_stored = runtime.policy.get(child_key).ok_or_else(|| {
                    anyhow!(
                        "certificate references unknown child state {}",
                        child.child_state_id
                    )
                })?;
                if !same_objective(&child_stored.objective, &child.objective) {
                    bail!("certificate child objective mismatch");
                }
                recomputed.worst_case_depth = recomputed
                    .worst_case_depth
                    .max(1 + child.objective.worst_case_depth);
                recomputed.expected_guesses +=
                    (child.mass / total_mass) * child.objective.expected_guesses;
            }
            if should_rederive && !same_objective(&recomputed, &candidate.objective) {
                bail!(
                    "certificate candidate objective mismatch for state {}",
                    state.state_id
                );
            }
            if compare_objective_with_kind(
                &candidate.objective,
                &state.best_objective,
                runtime.model.objective_spec.kind,
            )
            .is_lt()
            {
                bail!(
                    "certificate contains candidate better than best for state {}",
                    state.state_id
                );
            }
        }
        if !saw_best {
            bail!(
                "certificate is missing the winning candidate for state {}",
                state.state_id
            );
        }
    }
    Ok(())
}

fn combine_hashes(
    left: u64,
    middle: u64,
    right: u64,
    objective_spec: FormalObjectiveSpec,
    small_state_table_hash: u64,
) -> u64 {
    let mut hash = 1469598103934665603u64;
    hash = hash_bytes(hash, &left.to_le_bytes());
    hash = hash_bytes(hash, &middle.to_le_bytes());
    hash = hash_bytes(hash, &right.to_le_bytes());
    hash = hash_bytes(hash, objective_spec.id.as_bytes());
    hash = hash_bytes(hash, &objective_spec.version.to_le_bytes());
    hash = hash_bytes(hash, &STATE_FORMAT_VERSION.to_le_bytes());
    hash = hash_bytes(hash, &AUX_TABLE_VERSION.to_le_bytes());
    hash = hash_bytes(hash, &CERTIFICATE_FORMAT_VERSION.to_le_bytes());
    hash = hash_bytes(hash, &SMALL_STATE_TABLE_VERSION.to_le_bytes());
    hash = hash_bytes(hash, &small_state_table_hash.to_le_bytes());
    hash
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
    small_state_table: &SmallStateTable,
) -> f64 {
    let mut lower_bound = 1.0;
    for bucket in buckets {
        if bucket.pattern == ALL_GREEN_PATTERN {
            continue;
        }
        let child_floor = child_expected_lower_bound(bucket, pattern_space, small_state_table);
        lower_bound += (bucket.mass / total_mass) * child_floor;
    }
    lower_bound
}

fn child_expected_lower_bound(
    bucket: &PartitionBucket,
    pattern_space: f64,
    small_state_table: &SmallStateTable,
) -> f64 {
    if bucket.count <= 1 {
        return 1.0;
    }
    let table_floor = small_state_table.lower_bound(bucket.count);
    let entropy_floor = (bucket.entropy_bits / pattern_space.log2()).max(1.0);
    table_floor.max(entropy_floor)
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

fn objective_le(
    left: &PolicyObjective,
    right: &PolicyObjective,
    kind: FormalObjectiveKind,
) -> bool {
    !compare_objective_with_kind(left, right, kind).is_gt()
}

fn objective_ge(
    left: &PolicyObjective,
    right: &PolicyObjective,
    kind: FormalObjectiveKind,
) -> bool {
    !compare_objective_with_kind(left, right, kind).is_lt()
}

fn min_objective(
    left: &PolicyObjective,
    right: &PolicyObjective,
    kind: FormalObjectiveKind,
) -> PolicyObjective {
    if compare_objective_with_kind(left, right, kind).is_gt() {
        right.clone()
    } else {
        left.clone()
    }
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

fn trivial_certificate_state(state: &StateKey, stored: &StoredState) -> CertificateState {
    CertificateState {
        state: state.clone(),
        best_guess: stored.best_guess,
        best_objective: stored.objective.clone(),
        candidates: vec![CertificateCandidate {
            guess_index: stored.best_guess,
            objective: stored.objective.clone(),
            children: Vec::new(),
        }],
    }
}

fn prune_refined_guesses(
    plans: &[GuessQuickPlan],
    threshold: usize,
) -> (Vec<GuessQuickPlan>, usize) {
    if plans.len() > threshold {
        return (plans.to_vec(), 0);
    }
    let mut kept = Vec::with_capacity(plans.len());
    let mut pruned = 0usize;
    'candidate: for plan in plans {
        for retained in &kept {
            if plan_refined_by(plan, retained) {
                pruned += 1;
                continue 'candidate;
            }
        }
        kept.push(plan.clone());
    }
    (kept, pruned)
}

fn plan_refined_by(candidate: &GuessQuickPlan, retained: &GuessQuickPlan) -> bool {
    candidate.buckets.iter().all(|candidate_bucket| {
        retained.buckets.iter().any(|retained_bucket| {
            candidate_bucket.count <= retained_bucket.count
                && candidate_bucket.state == retained_bucket.state
                || state_is_subset_of(&candidate_bucket.state, &retained_bucket.state)
        })
    })
}

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
    fn reproducible_manifest_hash_uses_same_inputs() {
        let root = std::env::temp_dir().join("maybe-wordle-formal-manifest");
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
        let root = std::env::temp_dir().join("maybe-wordle-formal-partition-scratch");
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
        let root = std::env::temp_dir().join("maybe-wordle-formal-tampered-certificate");
        let _ = std::fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal dir");
        write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
        write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
        write_fixture(&artifacts.prior_spec, "kind = \"uniform\"\n");

        let _ = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("policy");
        let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("load");
        let mut certificate =
            read_proof_certificate(&paths, DEFAULT_FORMAL_MODEL_ID).expect("certificate");
        certificate.states[0].candidates[0]
            .objective
            .expected_guesses += 0.25;

        assert!(verify_certificate(&runtime, &certificate).is_err());
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
        let root = std::env::temp_dir().join("maybe-wordle-formal-toy");
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
    fn parse_observations_rejects_length_mismatch() {
        let error = parse_observations(&["crane".into()], &[]).expect_err("must fail");
        assert!(error.to_string().contains("same number"));
    }

    #[test]
    fn apply_history_filters_answers() {
        let root = std::env::temp_dir().join("maybe-wordle-formal-history");
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
