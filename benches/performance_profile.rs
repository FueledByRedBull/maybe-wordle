use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use chrono::NaiveDate;
use maybe_wordle::{
    atomic_file::atomic_write,
    config::PriorConfig,
    data::ProjectPaths,
    identity::{CanonicalSha256, IDENTITY_FORMAT},
    solver::{SolveState, Solver},
};
use serde::Serialize;

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

#[derive(Debug, Serialize)]
struct InputDigest {
    path: String,
    bytes: u64,
    fingerprint: String,
}

#[derive(Debug, Serialize)]
struct ProfileMeasurement {
    runs: usize,
    wall_ms_per_call: f64,
    process_cpu_ms_per_call: Option<f64>,
    process_cycles_per_call: Option<f64>,
    allocation_calls_per_call: f64,
    allocated_bytes_per_call: f64,
    page_faults_per_call: Option<f64>,
    current_working_set_bytes: Option<u64>,
    peak_working_set_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ProfileWorkload {
    id: String,
    description: String,
    as_of: NaiveDate,
    observations: Vec<String>,
    surviving_answers: usize,
    detected_regime: String,
    config_fingerprint: String,
    cold: ProfileMeasurement,
    warm: ProfileMeasurement,
    cold_to_warm_wall_ratio: f64,
}

#[derive(Debug, Serialize)]
struct PerformanceProfile {
    schema_version: u32,
    identity_format: String,
    scope: String,
    build_command: String,
    platform: String,
    cpu: Option<String>,
    code_revision: Option<String>,
    code_dirty: Option<bool>,
    executable_fingerprint: String,
    inputs: Vec<InputDigest>,
    workloads: Vec<ProfileWorkload>,
    limitations: Vec<String>,
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
    let output = explicit_output
        .unwrap_or_else(|| root.join("benchmarks/predictive/release-performance-v1.json"));
    let base = PriorConfig::load(&paths.config_prior)?;

    let root_proxy = PriorConfig {
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::ProxyOnly,
        ..base.clone()
    };
    let exact_observations = Solver::parse_observations(
        &["olate".to_string(), "embar".to_string()],
        &["bbgyb".to_string(), "bbbyb".to_string()],
    )?;
    let mut lookahead = base.clone();
    lookahead.exact_threshold = 8;
    lookahead.exact_exhaustive_threshold = 6;
    lookahead.medium_state_lookahead_threshold = 12;
    lookahead.lookahead_threshold = 16;
    lookahead.danger_exact_survivor_cap = 9;

    let workloads = vec![
        profile_workload(
            &paths,
            &root_proxy,
            "proxy-root",
            "Proxy-only ranking on the full modeled posterior.",
            NaiveDate::from_ymd_opt(2026, 7, 17).expect("date"),
            &[],
            12,
        )?,
        profile_workload(
            &paths,
            &lookahead,
            "bounded-lookahead-15",
            "Bounded lookahead on a replayable 15-answer posterior.",
            NaiveDate::from_ymd_opt(2026, 4, 18).expect("date"),
            &exact_observations,
            12,
        )?,
        profile_workload(
            &paths,
            &base,
            "pooled-exact-15",
            "Production pooled-exact search on the same 15-answer posterior.",
            NaiveDate::from_ymd_opt(2026, 4, 18).expect("date"),
            &exact_observations,
            3,
        )?,
    ];

    let executable = std::env::current_exe().context("failed to locate profile executable")?;
    let declared_inputs = [
        paths.config_prior.as_path(),
        paths.raw_history.as_path(),
        paths.seed_guesses.as_path(),
        paths.seed_answers.as_path(),
        paths.seed_reference_answers.as_path(),
        paths.manual_additions.as_path(),
        paths.pattern_table.as_path(),
    ];
    let inputs = declared_inputs
        .into_iter()
        .filter(|path| path.is_file())
        .map(|path| input_digest(&root, path))
        .collect::<Result<Vec<_>>>()?;
    let (code_revision, code_dirty) = git_provenance(&root);
    let report = PerformanceProfile {
        schema_version: 1,
        identity_format: IDENTITY_FORMAT.to_string(),
        scope: "release suggestion-kernel profile; no sealed-test evaluation".to_string(),
        build_command: "cargo bench --bench performance_profile".to_string(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        cpu: std::env::var("PROCESSOR_IDENTIFIER").ok(),
        code_revision,
        code_dirty,
        executable_fingerprint: fingerprint_file("maybe-wordle-profile-executable-v1", &executable)?,
        inputs,
        workloads,
        limitations: vec![
            "Allocation counts use a System-allocator wrapper in this dedicated benchmark executable; relaxed atomic counters add measurement overhead.".to_string(),
            "Process CPU time and cycle counts include Rayon worker threads and are process-wide.".to_string(),
            "Cold/warm ratios and page faults characterize application/data-cache behavior; hardware L1/L2/LLC miss counters were unavailable in the installed Windows profiling toolchain.".to_string(),
            "Peak working set is the process lifetime high-water mark and cannot be reset between workloads, so later workloads inherit earlier setup memory.".to_string(),
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
    paths: &ProjectPaths,
    config: &PriorConfig,
    id: &str,
    description: &str,
    as_of: NaiveDate,
    observations: &[(String, u8)],
    warm_runs: usize,
) -> Result<ProfileWorkload> {
    let solver = Solver::from_paths(paths, config)?;
    let state = solver.apply_history(as_of, observations)?;
    let (cold, regime) = measure_suggestions(&solver, &state, 1)?;
    let (warm, _) = measure_suggestions(&solver, &state, warm_runs)?;
    let config_toml = toml::to_string_pretty(config)?;
    Ok(ProfileWorkload {
        id: id.to_string(),
        description: description.to_string(),
        as_of,
        observations: observations
            .iter()
            .map(|(guess, feedback)| {
                format!(
                    "{}:{}",
                    guess,
                    maybe_wordle::scoring::format_feedback_letters(*feedback)
                )
            })
            .collect(),
        surviving_answers: state.surviving.len(),
        detected_regime: regime,
        config_fingerprint: maybe_wordle::identity::digest_bytes_tagged(
            "maybe-wordle-performance-config-v1",
            config_toml.as_bytes(),
        ),
        cold_to_warm_wall_ratio: cold.wall_ms_per_call / warm.wall_ms_per_call.max(f64::EPSILON),
        cold,
        warm,
    })
}

fn measure_suggestions(
    solver: &Solver,
    state: &SolveState,
    runs: usize,
) -> Result<(ProfileMeasurement, String)> {
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    let before = process_snapshot();
    COUNT_ALLOCATIONS.store(true, Ordering::SeqCst);
    let started = Instant::now();
    let mut regime = None;
    for _ in 0..runs {
        let suggestions = black_box(solver.suggestions(state, 5)?);
        let first = suggestions
            .first()
            .ok_or_else(|| anyhow!("profile workload returned no suggestions"))?;
        regime = Some(if first.exact_cost.is_some() {
            "exact"
        } else if first.lookahead_cost.is_some() {
            "lookahead"
        } else {
            "proxy"
        });
        black_box(suggestions);
    }
    let wall = started.elapsed();
    COUNT_ALLOCATIONS.store(false, Ordering::SeqCst);
    let after = process_snapshot();
    let per_call = runs as f64;
    let delta = |after: Option<u64>, before: Option<u64>| {
        Some(after?.saturating_sub(before?) as f64 / per_call)
    };
    Ok((
        ProfileMeasurement {
            runs,
            wall_ms_per_call: wall.as_secs_f64() * 1_000.0 / per_call,
            process_cpu_ms_per_call: delta(after.cpu_100ns, before.cpu_100ns)
                .map(|value| value / 10_000.0),
            process_cycles_per_call: delta(after.cycles, before.cycles),
            allocation_calls_per_call: ALLOCATION_CALLS.load(Ordering::Relaxed) as f64 / per_call,
            allocated_bytes_per_call: ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / per_call,
            page_faults_per_call: delta(after.page_faults, before.page_faults),
            current_working_set_bytes: after.current_working_set_bytes,
            peak_working_set_bytes: after.peak_working_set_bytes,
        },
        regime.unwrap_or("unknown").to_string(),
    ))
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
        fingerprint: fingerprint_file("maybe-wordle-performance-input-v1", path)?,
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
