use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::HashSet,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use chrono::NaiveDate;
use maybe_wordle::{
    atomic_file::atomic_write,
    config::PriorConfig,
    data::ProjectPaths,
    identity::{CanonicalSha256, IDENTITY_FORMAT},
    predictive::{PredictiveSuggestRequest, PredictiveSuggestResponse, PredictiveSuggestionMode},
    scoring::{ALL_GREEN_PATTERN, format_feedback_trits, parse_feedback, score_guess},
    solver::{FiniteSearchOptions, FiniteSearchQuality, FiniteSearchReason, SolveState, Solver},
};
use serde::Serialize;

const PROFILE_DATE: NaiveDate = match NaiveDate::from_ymd_opt(2026, 8, 1) {
    Some(date) => date,
    None => panic!("invalid fixed profile date"),
};
const PROFILE_SAMPLES: usize = 3;
const PREVIEW_BUDGET: Duration = Duration::from_millis(30);
const SYNTHETIC_GUESSES: &[&str] = &[
    "olate", "embar", "crane", "slate", "audio", "raise", "adieu", "arose", "stare", "charm",
    "rebut", "nymph", "cloud", "pious", "reply", "ought", "tears", "soare", "cigar", "fuzzy",
    "jumpy", "slyly", "bally", "belly", "bully", "alley", "civic", "eerie", "added", "dread",
];

struct CountingAllocator;

static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: every operation delegates to the process `System` allocator with the
// original pointer and layout. Optional relaxed counters do not change
// allocation ownership or lifetime.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        // SAFETY: delegated with the caller-provided valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        // SAFETY: delegated with the caller-provided valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: delegated with the original pointer and layout.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation(new_size);
        // SAFETY: delegated with the original pointer/layout and requested size.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_allocation(bytes: usize) {
    if COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProcessSnapshot {
    cpu_100ns: Option<u64>,
    cycles: Option<u64>,
    current_working_set_bytes: Option<u64>,
    peak_working_set_bytes: Option<u64>,
    page_faults: Option<u64>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct InputDigest {
    path: String,
    bytes: u64,
    fingerprint: String,
}

#[derive(Debug, Serialize)]
struct ProfileSample {
    wall_ms: f64,
    top_guess: Option<String>,
    top_quality: Option<String>,
    finite_status: String,
    support_candidate_count: usize,
    fallback_candidate_count: usize,
    candidate_probability_sum: f64,
    finite_candidate_count: usize,
    suggestion_count: usize,
    work_units: usize,
    nodes_visited: usize,
    cache_hits: usize,
    proposal_sampled: bool,
}

#[derive(Debug, Serialize)]
struct ProfileMeasurement {
    runs: usize,
    samples: Vec<ProfileSample>,
    wall_ms_per_call: f64,
    wall_ms_median: f64,
    wall_ms_p95: f64,
    wall_ms_max: f64,
    process_cpu_ms_per_call: Option<f64>,
    process_cycles_per_call: Option<f64>,
    allocation_calls_per_call: f64,
    allocated_bytes_per_call: f64,
    page_faults_per_call: Option<f64>,
    current_working_set_bytes: Option<u64>,
    peak_working_set_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct BudgetIdentity {
    budget_ms: u64,
    root_shortlist: usize,
    reply_shortlist: usize,
    exact_state_threshold: usize,
    node_limit: Option<usize>,
    baseline_only: bool,
    fingerprint: String,
}

#[derive(Debug, Serialize)]
struct ProfilePhase {
    phase: String,
    budget: BudgetIdentity,
    measurement: ProfileMeasurement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProfilePhaseSelection {
    All,
    Preview,
    Fast,
    Strong,
    Baseline,
}

impl ProfilePhaseSelection {
    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Preview => "preview",
            Self::Fast => "fast",
            Self::Strong => "strong",
            Self::Baseline => "baseline",
        }
    }
}

#[derive(Debug, Serialize)]
struct ProfileWorkload {
    id: String,
    description: String,
    puzzle_date: NaiveDate,
    observations: Vec<String>,
    hard_mode: bool,
    remaining_turns: usize,
    surviving_answers: usize,
    fallback_support_candidates: usize,
    config_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preview: Option<ProfilePhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fast: Option<ProfilePhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strong: Option<ProfilePhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline: Option<ProfilePhase>,
}

#[derive(Debug, Serialize)]
struct PerformanceProfile {
    schema_version: u32,
    identity_format: String,
    suite_id: String,
    scope: String,
    build_command: String,
    platform: String,
    cpu: Option<String>,
    code_revision: Option<String>,
    code_dirty: Option<bool>,
    executable_fingerprint: String,
    inputs: Vec<InputDigest>,
    sample_runs_per_phase: usize,
    selected_phase: String,
    workloads: Vec<ProfileWorkload>,
    limitations: Vec<String>,
}

#[derive(Clone, Debug)]
struct WorkloadSpec {
    id: &'static str,
    description: &'static str,
    observations: Vec<(String, u8)>,
    hard_mode: bool,
}

fn parse_profile_phase(value: Option<&str>) -> Result<ProfilePhaseSelection> {
    match value {
        None => Ok(ProfilePhaseSelection::All),
        Some("preview") => Ok(ProfilePhaseSelection::Preview),
        Some("fast") => Ok(ProfilePhaseSelection::Fast),
        Some("strong") => Ok(ProfilePhaseSelection::Strong),
        Some("baseline") => Ok(ProfilePhaseSelection::Baseline),
        Some(value) => Err(anyhow!(
            "invalid MAYBE_WORDLE_PROFILE_PHASE {value:?}; expected preview, fast, strong, or baseline"
        )),
    }
}

fn profile_phase_from_env() -> Result<ProfilePhaseSelection> {
    match std::env::var_os("MAYBE_WORDLE_PROFILE_PHASE") {
        None => Ok(ProfilePhaseSelection::All),
        Some(value) => {
            let value = value
                .to_str()
                .ok_or_else(|| anyhow!("MAYBE_WORDLE_PROFILE_PHASE must contain valid UTF-8"))?;
            parse_profile_phase(Some(value))
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let paths = ProjectPaths::new(&root);
    let explicit_output = std::env::var_os("MAYBE_WORDLE_PROFILE_OUTPUT").map(PathBuf::from);
    if cfg!(debug_assertions) && explicit_output.is_none() {
        println!(
            "performance_profile=skipped reason=debug-build hint='run cargo bench --bench performance_profile'"
        );
        return Ok(());
    }
    let selected_phase = profile_phase_from_env()?;
    let config_path = std::env::var_os("MAYBE_WORDLE_PROFILE_CONFIG")
        .map(|path| root.join(PathBuf::from(path)))
        .unwrap_or_else(|| paths.config_prior.clone());
    let output = explicit_output
        .unwrap_or_else(|| root.join("benchmarks/predictive/release-gameplay-latency-v1.json"));
    let declared_inputs = [
        config_path.as_path(),
        paths.raw_history.as_path(),
        paths.seed_guesses.as_path(),
        paths.seed_answers.as_path(),
        paths.seed_reference_answers.as_path(),
        paths.manual_additions.as_path(),
    ];
    let capture_inputs = || {
        declared_inputs
            .iter()
            .copied()
            .filter(|path| path.is_file())
            .map(|path| input_digest(&root, path))
            .collect::<Result<Vec<_>>>()
    };
    let mut inputs = capture_inputs()?;
    let base = PriorConfig::load(&config_path)?;
    let solver = Solver::from_paths(&paths, &base)?;
    if capture_inputs()? != inputs {
        return Err(anyhow!(
            "profile inputs changed while loading the solver; restart"
        ));
    }
    // Construction may legitimately generate this cache; bind it after construction.
    inputs.push(input_digest(&root, &paths.pattern_table)?);
    let config_toml = toml::to_string_pretty(&base)?;
    let config_fingerprint = maybe_wordle::identity::digest_bytes_tagged(
        "maybe-wordle-gameplay-latency-config-v2",
        config_toml.as_bytes(),
    );
    let specs = fixed_workloads(&solver, PROFILE_DATE)?;
    let total_workloads = specs.len();
    let profile_started = Instant::now();
    let mut workloads = Vec::with_capacity(total_workloads);
    for (index, spec) in specs.into_iter().enumerate() {
        println!(
            "performance_profile workload_start={}/{} id={} puzzle_date={} observations={} hard_mode={}",
            index + 1,
            total_workloads,
            spec.id,
            PROFILE_DATE,
            spec.observations.len(),
            spec.hard_mode
        );
        let workload_started = Instant::now();
        let workload =
            profile_workload(&solver, &spec, config_fingerprint.clone(), selected_phase)?;
        workloads.push(workload);
        let elapsed_s = profile_started.elapsed().as_secs_f64();
        let completed = (index + 1) as f64;
        let eta_s = elapsed_s / completed * (total_workloads - index - 1) as f64;
        println!(
            "performance_profile workload_complete={}/{} id={} elapsed_s={:.1} workload_s={:.1} eta_s={:.1}",
            index + 1,
            total_workloads,
            spec.id,
            elapsed_s,
            workload_started.elapsed().as_secs_f64(),
            eta_s
        );
    }
    let executable = std::env::current_exe().context("failed to locate profile executable")?;
    let mut final_inputs = capture_inputs()?;
    final_inputs.push(input_digest(&root, &paths.pattern_table)?);
    if final_inputs != inputs {
        return Err(anyhow!(
            "profile inputs changed during measurement; discard this run and restart"
        ));
    }
    let (code_revision, code_dirty) = git_provenance(&root);
    let report = PerformanceProfile {
        schema_version: 3,
        identity_format: IDENTITY_FORMAT.to_string(),
        suite_id: "fixed-reachable-gameplay-latency-v1".to_string(),
        scope: "fixed-date bounded gameplay latency profile; no sealed-test evaluation".to_string(),
        build_command: "cargo bench --bench performance_profile".to_string(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        cpu: std::env::var("PROCESSOR_IDENTIFIER").ok(),
        code_revision,
        code_dirty,
        executable_fingerprint: fingerprint_file(
            "maybe-wordle-gameplay-latency-executable-v2",
            &executable,
        )?,
        inputs,
        sample_runs_per_phase: PROFILE_SAMPLES,
        selected_phase: selected_phase.label().to_string(),
        workloads,
        limitations: vec![
            "Allocation counts use a System-allocator wrapper in this dedicated benchmark executable; relaxed atomic counters add measurement overhead.".to_string(),
            "Process CPU time and cycle counts include Rayon worker threads and are process-wide.".to_string(),
            "Cold/warm ratios and page faults characterize application/data-cache behavior; hardware L1/L2/LLC miss counters were unavailable in the installed Windows profiling toolchain.".to_string(),
            "Working-set values are process-lifetime high-water marks shared by the fixed suite; candidate-specific memory comparisons require isolated matched processes.".to_string(),
            "The reported three-sample p95 is the maximum of three samples, not a population p95 or gameplay-distribution estimate.".to_string(),
            "Synthetic observations are generated from supported candidate words and validated as reachable states; they are not held-out answer outcomes.".to_string(),
        ],
    };
    let encoded =
        serde_json::to_vec_pretty(&report).context("failed to encode performance profile")?;
    atomic_write(&output, &encoded)?;
    println!(
        "performance_profile={} workloads={}",
        output.display(),
        report.workloads.len()
    );
    Ok(())
}

fn profile_workload(
    solver: &Solver,
    spec: &WorkloadSpec,
    config_fingerprint: String,
    selected_phase: ProfilePhaseSelection,
) -> Result<ProfileWorkload> {
    let state = reachable_state(solver, PROFILE_DATE, &spec.observations)?;
    let fallback_support_candidates = state
        .surviving
        .iter()
        .filter(|index| state.modeled_weights[**index] <= 0.0)
        .count();
    let mut preview_options = FiniteSearchOptions::fast();
    preview_options.budget = PREVIEW_BUDGET;
    let mut baseline_options = FiniteSearchOptions::fast();
    baseline_options.baseline_only = true;
    let phases = match selected_phase {
        ProfilePhaseSelection::All => vec![
            ("preview", preview_options),
            ("fast", FiniteSearchOptions::fast()),
            ("strong", FiniteSearchOptions::strong()),
        ],
        ProfilePhaseSelection::Preview => vec![("preview", preview_options)],
        ProfilePhaseSelection::Fast => vec![("fast", FiniteSearchOptions::fast())],
        ProfilePhaseSelection::Strong => vec![("strong", FiniteSearchOptions::strong())],
        ProfilePhaseSelection::Baseline => vec![("baseline", baseline_options)],
    };
    let mut preview = None;
    let mut fast = None;
    let mut strong = None;
    let mut baseline = None;
    for (phase_name, options) in phases {
        println!(
            "performance_profile phase_start workload={} phase={} budget_ms={} runs={}",
            spec.id,
            phase_name,
            options.budget.as_millis(),
            PROFILE_SAMPLES
        );
        let measurement = measure_controlled(solver, spec, options, PROFILE_SAMPLES)?;
        println!(
            "performance_profile phase_complete workload={} phase={} median_ms={:.3} p95_ms={:.3} max_ms={:.3}",
            spec.id,
            phase_name,
            measurement.wall_ms_median,
            measurement.wall_ms_p95,
            measurement.wall_ms_max
        );
        let phase = ProfilePhase {
            phase: phase_name.to_string(),
            budget: budget_identity(options),
            measurement,
        };
        match phase_name {
            "preview" => preview = Some(phase),
            "fast" => fast = Some(phase),
            "strong" => strong = Some(phase),
            "baseline" => baseline = Some(phase),
            _ => unreachable!("profile phase list contains an unknown phase"),
        }
    }
    Ok(ProfileWorkload {
        id: spec.id.to_string(),
        description: spec.description.to_string(),
        puzzle_date: PROFILE_DATE,
        observations: format_observations(&spec.observations),
        hard_mode: spec.hard_mode,
        remaining_turns: 6usize.saturating_sub(spec.observations.len()),
        surviving_answers: state.surviving.len(),
        fallback_support_candidates,
        config_fingerprint,
        preview,
        fast,
        strong,
        baseline,
    })
}

fn measure_controlled(
    solver: &Solver,
    spec: &WorkloadSpec,
    options: FiniteSearchOptions,
    runs: usize,
) -> Result<ProfileMeasurement> {
    if runs == 0 {
        return Err(anyhow!("profile sample count must be positive"));
    }
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    let before = process_snapshot();
    COUNT_ALLOCATIONS.store(true, Ordering::SeqCst);
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        let response = match solver.suggest_predictive_controlled(
            PredictiveSuggestRequest {
                puzzle_date: PROFILE_DATE,
                observations: &spec.observations,
                top: 5,
                hard_mode: spec.hard_mode,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::LiveOnly,
            },
            options,
            &|| false,
        ) {
            Ok(response) => response,
            Err(error) => {
                COUNT_ALLOCATIONS.store(false, Ordering::SeqCst);
                return Err(error).with_context(|| {
                    format!(
                        "controlled profile failed for workload {} with {} budget",
                        spec.id,
                        options.budget.as_millis()
                    )
                });
            }
        };
        let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
        COUNT_ALLOCATIONS.store(false, Ordering::SeqCst);
        let sample = match profile_sample(&response, wall_ms) {
            Ok(sample) => sample,
            Err(error) => {
                COUNT_ALLOCATIONS.store(false, Ordering::SeqCst);
                return Err(error).with_context(|| {
                    format!(
                        "controlled profile returned invalid metadata for {}",
                        spec.id
                    )
                });
            }
        };
        black_box(response);
        samples.push(sample);
        COUNT_ALLOCATIONS.store(true, Ordering::SeqCst);
    }
    COUNT_ALLOCATIONS.store(false, Ordering::SeqCst);
    let after = process_snapshot();
    let per_call = runs as f64;
    let delta = |after: Option<u64>, before: Option<u64>| {
        Some(after?.saturating_sub(before?) as f64 / per_call)
    };
    let mut wall_samples = samples
        .iter()
        .map(|sample| sample.wall_ms)
        .collect::<Vec<_>>();
    wall_samples.sort_by(f64::total_cmp);
    Ok(ProfileMeasurement {
        runs,
        wall_ms_per_call: wall_samples.iter().sum::<f64>() / per_call,
        wall_ms_median: percentile(&wall_samples, 0.50),
        wall_ms_p95: percentile(&wall_samples, 0.95),
        wall_ms_max: *wall_samples.last().expect("profile samples are non-empty"),
        samples,
        process_cpu_ms_per_call: delta(after.cpu_100ns, before.cpu_100ns)
            .map(|value| value / 10_000.0),
        process_cycles_per_call: delta(after.cycles, before.cycles),
        allocation_calls_per_call: ALLOCATION_CALLS.load(Ordering::Relaxed) as f64 / per_call,
        allocated_bytes_per_call: ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / per_call,
        page_faults_per_call: delta(after.page_faults, before.page_faults),
        current_working_set_bytes: after.current_working_set_bytes,
        peak_working_set_bytes: after.peak_working_set_bytes,
    })
}

fn profile_sample(response: &PredictiveSuggestResponse, wall_ms: f64) -> Result<ProfileSample> {
    let mut probability_sum = 0.0;
    let mut fallback_candidate_count = 0;
    for candidate in &response.candidates {
        if !candidate.probability.is_finite() || candidate.probability < 0.0 {
            return Err(anyhow!(
                "profile response contained an invalid candidate probability"
            ));
        }
        probability_sum += candidate.probability;
        fallback_candidate_count += usize::from(candidate.fallback_support);
    }
    if response.candidates.is_empty() || (probability_sum - 1.0).abs() > 1e-6 {
        return Err(anyhow!(
            "profile response candidate probabilities do not form a complete support posterior"
        ));
    }
    let search = response.finite_search.as_ref();
    let top = response.suggestions.first();
    Ok(ProfileSample {
        wall_ms,
        top_guess: top.map(|suggestion| suggestion.word.clone()),
        top_quality: top
            .and_then(|suggestion| suggestion.finite_value)
            .map(|value| finite_quality_label(value.quality).to_string()),
        finite_status: search
            .map(|search| finite_reason_label(search.reason))
            .unwrap_or("Missing")
            .to_string(),
        support_candidate_count: response.candidates.len(),
        fallback_candidate_count,
        candidate_probability_sum: probability_sum,
        finite_candidate_count: search.map_or(0, |search| search.candidates.len()),
        suggestion_count: response.suggestions.len(),
        work_units: search.map_or(0, |search| search.work_units),
        nodes_visited: search.map_or(0, |search| search.nodes_visited),
        cache_hits: search.map_or(0, |search| search.cache_hits),
        proposal_sampled: search.is_some_and(|search| search.proposal_sampled),
    })
}

fn finite_quality_label(quality: FiniteSearchQuality) -> &'static str {
    match quality {
        FiniteSearchQuality::Heuristic => "Heuristic",
        FiniteSearchQuality::UpperBound => "UpperBound",
        FiniteSearchQuality::Exact => "Exact",
    }
}

fn finite_reason_label(reason: FiniteSearchReason) -> &'static str {
    match reason {
        FiniteSearchReason::Complete => "Complete",
        FiniteSearchReason::Deadline => "Deadline",
        FiniteSearchReason::Cancelled => "Cancelled",
        FiniteSearchReason::NodeBudget => "NodeBudget",
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn budget_identity(options: FiniteSearchOptions) -> BudgetIdentity {
    let mut identity = CanonicalSha256::new("maybe-wordle-gameplay-latency-budget-v3");
    identity
        .field(&options.root_shortlist.to_le_bytes())
        .field(&options.reply_shortlist.to_le_bytes())
        .field(&options.exact_state_threshold.to_le_bytes())
        .field(&options.budget.as_nanos().to_le_bytes())
        .field(&[
            options.node_limit.is_some() as u8,
            options.baseline_only as u8,
        ])
        .field(&options.node_limit.unwrap_or_default().to_le_bytes());
    BudgetIdentity {
        budget_ms: options.budget.as_millis() as u64,
        root_shortlist: options.root_shortlist,
        reply_shortlist: options.reply_shortlist,
        exact_state_threshold: options.exact_state_threshold,
        node_limit: options.node_limit,
        baseline_only: options.baseline_only,
        fingerprint: identity.finish_tagged(),
    }
}

fn fixed_workloads(solver: &Solver, puzzle_date: NaiveDate) -> Result<Vec<WorkloadSpec>> {
    let history_cutoff = puzzle_date
        .pred_opt()
        .ok_or_else(|| anyhow!("fixed profile puzzle date has no history cutoff"))?;
    let root = solver.fixed_posterior_state(history_cutoff)?;
    if root.surviving.is_empty() {
        return Err(anyhow!("fixed profile posterior has no surviving answers"));
    }
    let olate_pattern = parse_feedback("10001")?;
    let olate = vec![("olate".to_string(), olate_pattern)];
    reachable_state(solver, puzzle_date, &olate).with_context(
        || "fixed OLATE/10001 profile state is not reachable from the permitted date",
    )?;
    let pool = benchmark_guess_pool(solver);
    let strong = FiniteSearchOptions::strong();
    let exact_max = strong.exact_state_threshold.max(2);
    let exact_band = find_reachable_band(solver, &root, 2, exact_max.min(12), false, &pool)?;
    let broad_band = find_reachable_band(
        solver,
        &root,
        exact_max.saturating_add(1),
        exact_max.saturating_add(8),
        false,
        &pool,
    )?;
    let ambiguous_band = find_reachable_band(solver, &root, 3, 6, false, &pool)?;
    let fallback_target = root
        .surviving
        .iter()
        .find(|index| root.modeled_weights[**index] <= 0.0)
        .map(|index| solver.answers[*index].word.clone())
        .ok_or_else(|| anyhow!("fixed profile posterior has no fallback-supported candidate"))?;
    let fallback_observations =
        target_observations(solver, &root, &fallback_target, 1, false, &pool)?;
    let fallback_state = reachable_state(solver, puzzle_date, &fallback_observations)?;
    if !fallback_state
        .surviving
        .iter()
        .any(|index| root.modeled_weights[*index] <= 0.0)
    {
        return Err(anyhow!(
            "fallback profile state lost its fallback-supported candidate"
        ));
    }
    let hard_target = root
        .surviving
        .iter()
        .find(|index| root.modeled_weights[**index] > 0.0)
        .map(|index| solver.answers[*index].word.clone())
        .ok_or_else(|| anyhow!("fixed profile posterior has no modeled candidate"))?;
    let hard_observations = target_observations(solver, &root, &hard_target, 2, true, &pool)?;
    for (index, (guess, _)) in hard_observations.iter().enumerate() {
        if solver
            .hard_mode_violation(&hard_observations[..index], guess)
            .is_some()
        {
            return Err(anyhow!("generated hard-mode profile guess is not legal"));
        }
    }
    let late_observations = target_observations(solver, &root, &hard_target, 5, false, &pool)?;
    Ok(vec![
        WorkloadSpec {
            id: "root",
            description: "Fixed-date root posterior with no gameplay observations.",
            observations: Vec::new(),
            hard_mode: false,
        },
        WorkloadSpec {
            id: "olate-10001",
            description: "The reported OLATE/10001 first-feedback state.",
            observations: olate,
            hard_mode: false,
        },
        WorkloadSpec {
            id: "threshold-exact-band",
            description: "Reachable state at or below the strong exact-state threshold.",
            observations: exact_band,
            hard_mode: false,
        },
        WorkloadSpec {
            id: "threshold-broad-band",
            description: "Reachable state immediately above the strong exact-state threshold.",
            observations: broad_band,
            hard_mode: false,
        },
        WorkloadSpec {
            id: "ambiguous-cluster",
            description: "Small reachable ambiguous answer cluster.",
            observations: ambiguous_band,
            hard_mode: false,
        },
        WorkloadSpec {
            id: "fallback-tail",
            description: "Reachable state retaining a fallback-supported candidate.",
            observations: fallback_observations,
            hard_mode: false,
        },
        WorkloadSpec {
            id: "hard-mode",
            description: "Reachable two-turn state with hard-mode legal observations.",
            observations: hard_observations,
            hard_mode: true,
        },
        WorkloadSpec {
            id: "late-turns",
            description: "Reachable late-turn state with one attempt remaining.",
            observations: late_observations,
            hard_mode: false,
        },
    ])
}

fn reachable_state(
    solver: &Solver,
    puzzle_date: NaiveDate,
    observations: &[(String, u8)],
) -> Result<SolveState> {
    let mut state = solver.fixed_posterior_state(
        puzzle_date
            .pred_opt()
            .ok_or_else(|| anyhow!("profile puzzle date has no history cutoff"))?,
    )?;
    for (guess, pattern) in observations {
        solver.apply_feedback(&mut state, guess, *pattern)?;
    }
    if state.surviving.is_empty() {
        return Err(anyhow!("profile state has no surviving candidates"));
    }
    Ok(state)
}

fn format_observations(observations: &[(String, u8)]) -> Vec<String> {
    observations
        .iter()
        .map(|(guess, pattern)| {
            format!(
                "{}:{}",
                guess.to_ascii_uppercase(),
                format_feedback_trits(*pattern)
            )
        })
        .collect()
}

fn benchmark_guess_pool(solver: &Solver) -> Vec<String> {
    let mut pool = Vec::new();
    let mut seen = HashSet::new();
    for word in SYNTHETIC_GUESSES {
        if solver.guesses.iter().any(|candidate| candidate == word) && seen.insert(*word) {
            pool.push((*word).to_string());
        }
    }
    for word in solver.guesses.iter().take(128) {
        if seen.insert(word.as_str()) {
            pool.push(word.clone());
        }
    }
    pool
}

fn find_reachable_band(
    solver: &Solver,
    initial: &SolveState,
    minimum: usize,
    maximum: usize,
    fallback_target: bool,
    pool: &[String],
) -> Result<Vec<(String, u8)>> {
    let targets = initial
        .surviving
        .iter()
        .filter(|index| (initial.modeled_weights[**index] <= 0.0) == fallback_target)
        .take(24)
        .copied()
        .collect::<Vec<_>>();
    for target_index in targets {
        let target = solver.answers[target_index].word.clone();
        let mut frontier = vec![(initial.clone(), Vec::<(String, u8)>::new())];
        for depth in 0..=3 {
            for (state, observations) in &frontier {
                if !observations.is_empty() && (minimum..=maximum).contains(&state.surviving.len())
                {
                    return Ok(observations.clone());
                }
            }
            if depth == 3 {
                break;
            }
            let mut next = Vec::new();
            for (state, observations) in frontier {
                for guess in pool {
                    if guess == &target || observations.iter().any(|(used, _)| used == guess) {
                        continue;
                    }
                    let pattern = score_guess(guess, &target);
                    if pattern == ALL_GREEN_PATTERN {
                        continue;
                    }
                    let mut next_state = state.clone();
                    if solver
                        .apply_feedback(&mut next_state, guess, pattern)
                        .is_err()
                    {
                        continue;
                    }
                    let mut next_observations = observations.clone();
                    next_observations.push((guess.clone(), pattern));
                    next.push((next_state, next_observations));
                    if next.len() >= 256 {
                        break;
                    }
                }
                if next.len() >= 256 {
                    break;
                }
            }
            frontier = next;
        }
    }
    Err(anyhow!(
        "could not construct a reachable state in survivor band {}..={}",
        minimum,
        maximum
    ))
}

fn target_observations(
    solver: &Solver,
    initial: &SolveState,
    target: &str,
    steps: usize,
    hard_mode: bool,
    pool: &[String],
) -> Result<Vec<(String, u8)>> {
    let mut state = initial.clone();
    let mut observations = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut selected = None;
        for guess in pool.iter().chain(solver.guesses.iter()) {
            if guess == target || observations.iter().any(|(used, _)| used == guess) {
                continue;
            }
            if hard_mode && solver.hard_mode_violation(&observations, guess).is_some() {
                continue;
            }
            let pattern = score_guess(guess, target);
            if pattern == ALL_GREEN_PATTERN {
                continue;
            }
            let mut next_state = state.clone();
            if solver
                .apply_feedback(&mut next_state, guess, pattern)
                .is_ok()
                && !next_state.surviving.is_empty()
            {
                selected = Some((guess.clone(), pattern, next_state));
                break;
            }
        }
        let (guess, pattern, next_state) = selected.ok_or_else(|| {
            anyhow!(
                "could not construct {} reachable observations for target {}",
                steps,
                target
            )
        })?;
        observations.push((guess, pattern));
        state = next_state;
    }
    Ok(observations)
}

fn input_digest(root: &Path, path: &Path) -> Result<InputDigest> {
    let metadata = fs::metadata(path)?;
    Ok(InputDigest {
        path: path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/"),
        bytes: metadata.len(),
        fingerprint: fingerprint_file("maybe-wordle-gameplay-latency-input-v2", path)?,
    })
}

fn fingerprint_file(domain: &str, path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)?;
    let mut file = fs::File::open(path)?;
    let mut digest = CanonicalSha256::new(domain);
    digest
        .field_reader(&mut file, metadata.len())
        .with_context(|| format!("failed to fingerprint {}", path.display()))?;
    Ok(digest.finish_tagged())
}

fn git_provenance(root: &Path) -> (Option<String>, Option<bool>) {
    let revision = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let dirty = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty());
    (revision, dirty)
}

#[cfg(windows)]
fn process_snapshot() -> ProcessSnapshot {
    use std::{ffi::c_void, mem::size_of};

    #[repr(C)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
        fn QueryProcessCycleTime(process: *mut c_void, cycles: *mut u64) -> i32;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCountersEx,
            size: u32,
        ) -> i32;
    }

    let process = unsafe { GetCurrentProcess() };
    let mut creation = FileTime { low: 0, high: 0 };
    let mut exit = FileTime { low: 0, high: 0 };
    let mut kernel = FileTime { low: 0, high: 0 };
    let mut user = FileTime { low: 0, high: 0 };
    let mut cycles = 0u64;
    let mut memory = ProcessMemoryCountersEx {
        cb: size_of::<ProcessMemoryCountersEx>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
        private_usage: 0,
    };
    // SAFETY: the pseudo-handle is process-local and every output pointer refers
    // to a correctly sized, live writable structure.
    let times_ok =
        unsafe { GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user) } != 0;
    // SAFETY: `cycles` is a live writable u64 for the duration of the call.
    let cycles_ok = unsafe { QueryProcessCycleTime(process, &mut cycles) } != 0;
    // SAFETY: `memory` has the Windows-declared layout and byte size.
    let memory_ok = unsafe {
        GetProcessMemoryInfo(
            process,
            &mut memory,
            size_of::<ProcessMemoryCountersEx>() as u32,
        )
    } != 0;
    let file_time = |value: FileTime| ((value.high as u64) << 32) | value.low as u64;
    ProcessSnapshot {
        cpu_100ns: times_ok.then(|| file_time(kernel) + file_time(user)),
        cycles: cycles_ok.then_some(cycles),
        current_working_set_bytes: memory_ok.then_some(memory.working_set_size as u64),
        peak_working_set_bytes: memory_ok.then_some(memory.peak_working_set_size as u64),
        page_faults: memory_ok.then_some(memory.page_fault_count as u64),
    }
}

#[cfg(not(windows))]
fn process_snapshot() -> ProcessSnapshot {
    ProcessSnapshot::default()
}

#[cfg(test)]
mod tests {
    #[test]
    fn profile_input_digest_detects_same_length_mutation() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("target").join(format!(
            "profile-input-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::write(&path, b"before").expect("write initial input");
        let before = super::input_digest(root, &path).expect("initial digest");
        std::fs::write(&path, b"after!").expect("mutate input");
        let after = super::input_digest(root, &path).expect("changed digest");
        std::fs::remove_file(&path).expect("remove test input");
        assert_eq!(before.bytes, after.bytes);
        assert_ne!(before, after);
    }

    #[test]
    fn invalid_profile_phase_is_rejected() {
        let error = super::parse_profile_phase(Some("not-a-phase")).expect_err("invalid phase");
        assert!(error.to_string().contains("MAYBE_WORDLE_PROFILE_PHASE"));
        assert_eq!(
            super::parse_profile_phase(None).expect("default phase"),
            super::ProfilePhaseSelection::All
        );
    }
}
