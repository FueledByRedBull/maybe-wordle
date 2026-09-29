use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use super::{DEFAULT_FORMAL_MODEL_ID, FormalVerificationMode, PolicyArtifactSet, parsing};
use crate::{
    atomic_file::atomic_write,
    data::{ProjectPaths, read_word_list_from, validate_answer_universe},
    identity::{CanonicalSha256, IDENTITY_FORMAT, is_tagged_digest},
    process_memory::process_memory_snapshot,
};

const FORMAL_SCALE_FORMAT_VERSION: u32 = 3;
const MAXIMUM_SAFE_SCALE_PREFIX: usize = 16;

#[derive(Clone, Debug)]
pub struct FormalScaleRequest {
    pub answer_counts: Vec<usize>,
    pub guess_limit: usize,
    pub maximum_seconds: u64,
    pub maximum_memory_mb: u64,
    pub maximum_disk_mb: u64,
    pub output: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormalScalePoint {
    pub answer_count: usize,
    pub guess_count: usize,
    pub policy_states: usize,
    pub certificate_states: usize,
    pub build_millis: u128,
    pub verify_millis: u128,
    pub states_per_second: f64,
    pub certificate_bytes: u64,
    pub artifact_bytes: u64,
    pub scale_checkpoint_bytes: u64,
    pub process_peak_working_set_bytes: u64,
    pub manifest_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormalScaleProjection {
    pub target_answer_count: usize,
    pub method: String,
    pub source_points: usize,
    pub projected_log10_seconds: Option<f64>,
    pub projected_log10_certificate_bytes: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormalScaleReport {
    pub format_version: u32,
    pub identity_format: String,
    pub input_fingerprint: String,
    pub operating_system: String,
    pub architecture: String,
    pub logical_cpus: usize,
    #[serde(deserialize_with = "parsing::bounded_vec::<_, _, MAXIMUM_SAFE_SCALE_PREFIX>")]
    pub answer_counts: Vec<usize>,
    pub source_answer_count: usize,
    pub guess_limit: usize,
    pub maximum_seconds: u64,
    pub maximum_memory_mb: u64,
    pub maximum_disk_mb: u64,
    pub full_model_projection: FormalScaleProjection,
    #[serde(deserialize_with = "parsing::bounded_vec::<_, _, MAXIMUM_SAFE_SCALE_PREFIX>")]
    pub points: Vec<FormalScalePoint>,
    pub completed: bool,
    pub stopped_reason: Option<String>,
}

pub fn benchmark_formal_scale(
    paths: &ProjectPaths,
    request: &FormalScaleRequest,
) -> Result<FormalScaleReport> {
    let run_started = Instant::now();
    validate_request(request)?;
    process_memory_snapshot().ok_or_else(|| {
        anyhow!(
            "formal scale memory budgets are unsupported on this operating system; supported platforms are Windows, Linux, and macOS"
        )
    })?;
    let raw_guesses = parsing::read_bounded(&paths.seed_guesses, parsing::MAX_INPUT_BYTES)?;
    let raw_answers = parsing::read_bounded(&paths.seed_answers, parsing::MAX_INPUT_BYTES)?;
    let source_guesses =
        read_word_list_from(std::io::Cursor::new(&raw_guesses), &paths.seed_guesses)?;
    let source_answers =
        read_word_list_from(std::io::Cursor::new(&raw_answers), &paths.seed_answers)?;
    validate_answer_universe(&source_guesses, source_answers.iter().map(String::as_str))?;
    let effective_guess_limit = if request.guess_limit == 0 {
        source_guesses.len()
    } else {
        request.guess_limit.min(source_guesses.len())
    };
    let maximum_answer_count = *request.answer_counts.last().expect("validated counts");
    if maximum_answer_count > source_answers.len() {
        bail!(
            "formal scale requests {} answers but the pinned source has only {}",
            maximum_answer_count,
            source_answers.len()
        );
    }
    if effective_guess_limit < maximum_answer_count {
        bail!("formal scale guess limit must cover every selected answer");
    }
    if source_answers.len() > parsing::MAX_WORDS
        || effective_guess_limit > parsing::MAX_WORDS
        || effective_guess_limit
            .checked_mul(maximum_answer_count)
            .is_none_or(|cells| cells > parsing::MAX_PATTERN_CELLS)
    {
        bail!("formal scale dimensions exceed formal word-id or pattern-cell limits");
    }
    let minimum_build_bytes =
        super::HOT_TT_BYTES as u64 + (effective_guess_limit * maximum_answer_count) as u64;
    if minimum_build_bytes > mib(request.maximum_memory_mb) {
        bail!(
            "formal scale memory budget is smaller than its transposition table and selected pattern dimensions"
        );
    }
    let fingerprint =
        scale_input_fingerprint(request, effective_guess_limit, &raw_guesses, &raw_answers)?;
    let mut report = load_or_initialize_report(
        request,
        effective_guess_limit,
        source_answers.len(),
        &fingerprint,
    )?;
    let previous_millis = report.points.iter().fold(0u128, |total, point| {
        total
            .saturating_add(point.build_millis)
            .saturating_add(point.verify_millis)
    });
    let budget = ScaleBudget {
        started: run_started,
        previous_millis,
        maximum_millis: request.maximum_seconds as u128 * 1000,
        maximum_memory_bytes: mib(request.maximum_memory_mb),
        sample: std::sync::Mutex::new((None, None)),
    };
    let cancelled = || budget.cancelled();
    report.completed = false;
    report.stopped_reason = None;
    let namespace = fingerprint
        .strip_prefix("sha256-v1:")
        .unwrap_or(&fingerprint);
    let scratch_root = paths.root.join("target/formal-scale").join(namespace);
    fs::create_dir_all(&scratch_root)
        .with_context(|| format!("failed to create {}", scratch_root.display()))?;
    validate_resumed_artifacts(&report, &scratch_root, &source_guesses, &source_answers)?;

    for answer_count in request.answer_counts.iter().copied() {
        if report
            .points
            .iter()
            .any(|point| point.answer_count == answer_count)
        {
            continue;
        }
        if let Some(reason) = preflight_stop_reason(&report, request, answer_count) {
            report.stopped_reason = Some(reason);
            checkpoint_report(&request.output, &mut report)?;
            return Ok(report);
        }

        let point_root = scratch_root.join(format!("answers-{answer_count:02}"));
        let point_paths = ProjectPaths::new(&point_root);
        point_paths.ensure_layout()?;
        let artifacts = PolicyArtifactSet::for_model(&point_paths, DEFAULT_FORMAL_MODEL_ID);
        fs::create_dir_all(&artifacts.model_dir)
            .with_context(|| format!("failed to create {}", artifacts.model_dir.display()))?;
        let answers = source_answers[..answer_count].to_vec();
        let guesses = prefix_guess_space(&source_guesses, &answers, effective_guess_limit);
        write_word_list(&point_paths.seed_answers, &answers)?;
        write_word_list(&point_paths.seed_guesses, &guesses)?;
        atomic_write(
            &artifacts.prior_spec,
            b"objective = \"lexicographic\"\nkind = \"uniform\"\n",
        )?;

        let build_started = Instant::now();
        let build = match super::build_optimal_policy_controlled(
            &point_paths,
            DEFAULT_FORMAL_MODEL_ID,
            &cancelled,
        ) {
            Ok(build) => build,
            Err(error) if error.downcast_ref::<super::FormalSearchStop>().is_some() => {
                report.stopped_reason = Some(budget.reason());
                checkpoint_report(&request.output, &mut report)?;
                return Ok(report);
            }
            Err(error) => return Err(error),
        };
        let build_millis = build_started.elapsed().as_millis();
        let verify_started = Instant::now();
        let verify = match super::verify_optimal_policy_controlled(
            &point_paths,
            DEFAULT_FORMAL_MODEL_ID,
            FormalVerificationMode::Certificate,
            &cancelled,
        ) {
            Ok(verify) => verify,
            Err(error) if error.downcast_ref::<super::FormalSearchStop>().is_some() => {
                report.stopped_reason = Some(budget.reason());
                checkpoint_report(&request.output, &mut report)?;
                return Ok(report);
            }
            Err(error) => return Err(error),
        };
        let verify_millis = verify_started.elapsed().as_millis();
        let artifacts = PolicyArtifactSet::published(&point_paths, DEFAULT_FORMAL_MODEL_ID)?;
        let memory = process_memory_snapshot().ok_or_else(|| {
            anyhow!("formal scale memory sampler became unavailable during the run")
        })?;
        let certificate_bytes = file_bytes(&artifacts.certificate)?;
        let artifact_bytes = formal_artifact_bytes(&artifacts)?;
        let elapsed_seconds = (build_millis.max(1) as f64) / 1_000.0;
        report.points.push(FormalScalePoint {
            answer_count,
            guess_count: guesses.len(),
            policy_states: build.solved_states,
            certificate_states: verify.certificate_state_count,
            build_millis,
            verify_millis,
            states_per_second: verify.certificate_state_count as f64 / elapsed_seconds,
            certificate_bytes,
            artifact_bytes,
            scale_checkpoint_bytes: 0,
            process_peak_working_set_bytes: memory.peak_working_set_bytes,
            manifest_hash: build.manifest_hash,
        });
        report.full_model_projection = projection(&report.points, report.source_answer_count);
        checkpoint_report(&request.output, &mut report)?;
        if cancelled() {
            report.stopped_reason = Some(budget.reason());
            checkpoint_report(&request.output, &mut report)?;
            return Ok(report);
        }
        if report.points.iter().fold(0u64, |total, point| {
            total.saturating_add(point.artifact_bytes)
        }) > mib(request.maximum_disk_mb)
        {
            report.stopped_reason = Some("completed artifacts exceed the disk budget".to_string());
            checkpoint_report(&request.output, &mut report)?;
            return Ok(report);
        }
    }

    report.completed = true;
    report.stopped_reason = None;
    report.full_model_projection = projection(&report.points, report.source_answer_count);
    checkpoint_report(&request.output, &mut report)?;
    Ok(report)
}

struct ScaleBudget {
    started: Instant,
    previous_millis: u128,
    maximum_millis: u128,
    maximum_memory_bytes: u64,
    sample: std::sync::Mutex<(Option<Instant>, Option<String>)>,
}

impl ScaleBudget {
    fn cancelled(&self) -> bool {
        let mut sample = self
            .sample
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if sample.1.is_some() {
            return true;
        }
        if self
            .previous_millis
            .saturating_add(self.started.elapsed().as_millis())
            >= self.maximum_millis
        {
            sample.1 = Some(
                "cooperative time budget reached during formal build/verification".to_string(),
            );
        } else if sample
            .0
            .is_none_or(|last| last.elapsed() >= std::time::Duration::from_millis(50))
        {
            sample.0 = Some(Instant::now());
            sample.1 = match process_memory_snapshot() {
                Some(memory) if memory.peak_working_set_bytes > self.maximum_memory_bytes => Some(
                    "cooperative process-memory budget reached during formal build/verification"
                        .to_string(),
                ),
                None => Some("formal scale memory sampler became unavailable".to_string()),
                _ => None,
            };
        }
        sample.1.is_some()
    }

    fn reason(&self) -> String {
        self.sample
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .1
            .clone()
            .unwrap_or_else(|| "formal scale computation cancelled".to_string())
    }
}

fn validate_request(request: &FormalScaleRequest) -> Result<()> {
    if request.answer_counts.is_empty()
        || request.answer_counts.contains(&0)
        || request.maximum_seconds == 0
        || request.maximum_memory_mb == 0
        || request.maximum_disk_mb == 0
    {
        bail!(
            "formal scale answer counts and resource budgets must be positive (guess_limit=0 means all guesses)"
        );
    }
    if request.answer_counts[0] > 6
        || request
            .answer_counts
            .windows(2)
            .any(|pair| pair[0] >= pair[1] || pair[1] - pair[0] > 2)
    {
        bail!(
            "formal scale answer counts must increase strictly, start at six or fewer, and advance by at most two"
        );
    }
    let maximum = *request.answer_counts.last().expect("non-empty");
    if maximum > MAXIMUM_SAFE_SCALE_PREFIX {
        bail!(
            "formal scale prefixes are capped at {} answers; use the projection before authorizing a larger run",
            MAXIMUM_SAFE_SCALE_PREFIX
        );
    }
    if request.guess_limit != 0 && request.guess_limit < maximum {
        bail!("formal scale guess limit must cover every selected answer");
    }
    Ok(())
}

fn load_or_initialize_report(
    request: &FormalScaleRequest,
    effective_guess_limit: usize,
    source_answer_count: usize,
    fingerprint: &str,
) -> Result<FormalScaleReport> {
    if request.output.exists() {
        let report: FormalScaleReport =
            parsing::read_json(&request.output, parsing::MAX_JSON_BYTES)
                .context("invalid formal scale checkpoint; choose a new output path")?;
        if report.format_version != FORMAL_SCALE_FORMAT_VERSION
            || report.identity_format != IDENTITY_FORMAT
            || report.input_fingerprint != fingerprint
            || report.answer_counts != request.answer_counts
            || report.guess_limit != effective_guess_limit
            || report.source_answer_count != source_answer_count
            || report.operating_system != std::env::consts::OS
            || report.architecture != std::env::consts::ARCH
            || report.maximum_seconds != request.maximum_seconds
            || report.maximum_memory_mb != request.maximum_memory_mb
            || report.maximum_disk_mb != request.maximum_disk_mb
        {
            bail!(
                "formal scale checkpoint provenance does not match this run; choose a new output path"
            );
        }
        validate_checkpoint(&report)?;
        return Ok(report);
    }
    Ok(FormalScaleReport {
        format_version: FORMAL_SCALE_FORMAT_VERSION,
        identity_format: IDENTITY_FORMAT.to_string(),
        input_fingerprint: fingerprint.to_string(),
        operating_system: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        logical_cpus: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        answer_counts: request.answer_counts.clone(),
        source_answer_count,
        guess_limit: effective_guess_limit,
        maximum_seconds: request.maximum_seconds,
        maximum_memory_mb: request.maximum_memory_mb,
        maximum_disk_mb: request.maximum_disk_mb,
        full_model_projection: projection(&[], source_answer_count),
        points: Vec::new(),
        completed: false,
        stopped_reason: None,
    })
}

fn preflight_stop_reason(
    report: &FormalScaleReport,
    request: &FormalScaleRequest,
    next_answer_count: usize,
) -> Option<String> {
    let elapsed_millis = report.points.iter().fold(0u128, |sum, point| {
        sum.saturating_add(point.build_millis)
            .saturating_add(point.verify_millis)
    });
    if elapsed_millis >= request.maximum_seconds as u128 * 1_000 {
        return Some(format!(
            "time budget exhausted before {} answers",
            next_answer_count
        ));
    }
    if report
        .points
        .last()
        .is_some_and(|point| point.process_peak_working_set_bytes > mib(request.maximum_memory_mb))
    {
        return Some(format!(
            "memory budget exhausted before {} answers",
            next_answer_count
        ));
    }
    let disk_bytes = report
        .points
        .iter()
        .map(|point| point.artifact_bytes)
        .fold(0u64, u64::saturating_add);
    if disk_bytes > mib(request.maximum_disk_mb) {
        return Some(format!(
            "disk budget exhausted before {} answers",
            next_answer_count
        ));
    }
    if let Some(predicted_peak) = predict_next_metric(&report.points, next_answer_count, |point| {
        point.process_peak_working_set_bytes as f64
    }) && predicted_peak * 1.25 > mib(request.maximum_memory_mb) as f64
    {
        return Some(format!(
            "projected next-point peak memory ({:.1} MiB) exceeds the memory budget with safety margin",
            predicted_peak / (1024.0 * 1024.0)
        ));
    }
    if let Some(predicted_artifact_bytes) =
        predict_next_metric(&report.points, next_answer_count, |point| {
            point.artifact_bytes as f64
        })
        && (disk_bytes as f64 + predicted_artifact_bytes * 1.25)
            > mib(request.maximum_disk_mb) as f64
    {
        return Some(format!(
            "projected next-point artifacts ({:.1} MiB) exceed the remaining disk budget with safety margin",
            predicted_artifact_bytes / (1024.0 * 1024.0)
        ));
    }
    if let Some(predicted_millis) = predict_next_millis(&report.points, next_answer_count) {
        let remaining_millis = request.maximum_seconds as f64 * 1_000.0 - elapsed_millis as f64;
        if predicted_millis * 1.25 > remaining_millis {
            return Some(format!(
                "projected next point ({predicted_millis:.0} ms) does not fit the remaining time budget"
            ));
        }
    }
    None
}

fn validate_checkpoint(report: &FormalScaleReport) -> Result<()> {
    if report.source_answer_count == 0
        || report.logical_cpus == 0
        || report.points.len() > report.answer_counts.len()
        || report.completed
            && (report.points.len() != report.answer_counts.len()
                || report.stopped_reason.is_some())
    {
        bail!("formal scale checkpoint has inconsistent completion or source counts");
    }
    for (point, expected_count) in report.points.iter().zip(&report.answer_counts) {
        if point.answer_count != *expected_count
            || point.answer_count == 0
            || point.guess_count != report.guess_limit
            || point.guess_count < point.answer_count
            || point.policy_states == 0
            || point.policy_states > super::MAX_FORMAL_STATES
            || point.certificate_states < point.policy_states
            || point.certificate_states > super::MAX_FORMAL_STATES
            || !point.states_per_second.is_finite()
            || point.states_per_second <= 0.0
            || point.certificate_bytes == 0
            || point.artifact_bytes < point.certificate_bytes
            || point.scale_checkpoint_bytes == 0
            || point.scale_checkpoint_bytes > parsing::MAX_JSON_BYTES
            || point.process_peak_working_set_bytes == 0
            || !is_tagged_digest(&point.manifest_hash)
        {
            bail!(
                "formal scale checkpoint points must be a valid, unique ordered prefix of requested counts"
            );
        }
        let expected_rate =
            point.certificate_states as f64 / (point.build_millis.max(1) as f64 / 1000.0);
        if (expected_rate - point.states_per_second).abs() > expected_rate.max(1.0) * 1e-9 {
            bail!("formal scale checkpoint has inconsistent throughput");
        }
    }
    let expected = projection(&report.points, report.source_answer_count);
    if serde_json::to_value(&report.full_model_projection)? != serde_json::to_value(expected)? {
        bail!("formal scale checkpoint projection does not match its source and points");
    }
    Ok(())
}

fn validate_resumed_artifacts(
    report: &FormalScaleReport,
    scratch_root: &Path,
    source_guesses: &[String],
    source_answers: &[String],
) -> Result<()> {
    for point in &report.points {
        let paths =
            ProjectPaths::new(scratch_root.join(format!("answers-{:02}", point.answer_count)));
        let runtime = super::FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID)
            .context("formal scale checkpoint references an unavailable or invalid generation")?;
        let manifest = &runtime.model.manifest;
        let expected_answers = source_answers.get(..point.answer_count).ok_or_else(|| {
            anyhow!("formal scale resumed answer count exceeds the selected source")
        })?;
        let expected_guesses =
            prefix_guess_space(source_guesses, expected_answers, report.guess_limit);
        let artifacts = &runtime.artifacts;
        let certificate: super::ProofCertificate =
            parsing::read_json(&artifacts.certificate, parsing::MAX_CERTIFICATE_BYTES)?;
        if manifest.manifest_hash != point.manifest_hash
            || manifest.answer_count != point.answer_count
            || manifest.guess_count != point.guess_count
            || manifest.answer_hash
                != crate::identity::tag(&crate::pattern_table::hash_word_list(
                    expected_answers.iter().map(String::as_str),
                ))
            || manifest.guess_hash
                != crate::identity::tag(&crate::pattern_table::hash_word_list(
                    expected_guesses.iter().map(String::as_str),
                ))
            || runtime.policy.len() != point.policy_states
            || certificate.state_count != point.certificate_states
            || file_bytes(&artifacts.certificate)? != point.certificate_bytes
            || formal_artifact_bytes(artifacts)? != point.artifact_bytes
        {
            bail!(
                "formal scale checkpoint point identity/counts do not match its published artifacts"
            );
        }
    }
    Ok(())
}

fn predict_next_millis(points: &[FormalScalePoint], next_answer_count: usize) -> Option<f64> {
    predict_next_metric(points, next_answer_count, |point| {
        point.build_millis.max(1) as f64
    })
}

fn predict_next_metric(
    points: &[FormalScalePoint],
    next_answer_count: usize,
    metric: impl Fn(&FormalScalePoint) -> f64,
) -> Option<f64> {
    let last = points.last()?;
    let last_value = metric(last).max(1.0);
    let per_answer_growth = if points.len() >= 2 {
        let previous = &points[points.len() - 2];
        let delta = last.answer_count.checked_sub(previous.answer_count)?;
        if delta == 0 {
            return None;
        }
        let answer_delta = delta as f64;
        (last_value / metric(previous).max(1.0))
            .powf(1.0 / answer_delta)
            .max(1.0)
    } else {
        2.0
    };
    let delta = next_answer_count.checked_sub(last.answer_count)?;
    if delta == 0 {
        return None;
    }
    Some(last_value * per_answer_growth.powf(delta as f64))
}

fn projection(points: &[FormalScalePoint], target_answer_count: usize) -> FormalScaleProjection {
    FormalScaleProjection {
        target_answer_count,
        method: "least-squares log10(metric) fit on tiny prefixes; extrapolation is not a full-model runtime guarantee".to_string(),
        source_points: points.len(),
        projected_log10_seconds: log_linear_projection(points, target_answer_count, |point| {
            point.build_millis.max(1) as f64 / 1_000.0
        }),
        projected_log10_certificate_bytes: log_linear_projection(points, target_answer_count, |point| {
            point.certificate_bytes.max(1) as f64
        }),
    }
}

fn log_linear_projection(
    points: &[FormalScalePoint],
    target_answer_count: usize,
    metric: impl Fn(&FormalScalePoint) -> f64,
) -> Option<f64> {
    if points.len() < 3 {
        return None;
    }
    let selected = &points[points.len().saturating_sub(5)..];
    let mean_x = selected
        .iter()
        .map(|point| point.answer_count as f64)
        .sum::<f64>()
        / selected.len() as f64;
    let mean_y = selected
        .iter()
        .map(|point| metric(point).log10())
        .sum::<f64>()
        / selected.len() as f64;
    let denominator = selected
        .iter()
        .map(|point| {
            let centered = point.answer_count as f64 - mean_x;
            centered * centered
        })
        .sum::<f64>();
    if denominator <= 0.0 {
        return None;
    }
    let slope = selected
        .iter()
        .map(|point| (point.answer_count as f64 - mean_x) * (metric(point).log10() - mean_y))
        .sum::<f64>()
        / denominator;
    let projected = mean_y + slope * (target_answer_count as f64 - mean_x);
    projected.is_finite().then_some(projected)
}

fn prefix_guess_space(
    source_guesses: &[String],
    answers: &[String],
    guess_limit: usize,
) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut guesses = Vec::with_capacity(guess_limit);
    for word in answers.iter().chain(source_guesses) {
        if seen.insert(word.clone()) {
            guesses.push(word.clone());
            if guesses.len() == guess_limit {
                break;
            }
        }
    }
    guesses
}

fn write_word_list(path: &Path, words: &[String]) -> Result<()> {
    let mut bytes = words.join("\n").into_bytes();
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

fn scale_input_fingerprint(
    request: &FormalScaleRequest,
    effective_guess_limit: usize,
    raw_guesses: &[u8],
    raw_answers: &[u8],
) -> Result<String> {
    let mut hash = CanonicalSha256::new("maybe-wordle-formal-scale-v1");
    hash.field(&FORMAL_SCALE_FORMAT_VERSION.to_le_bytes())
        .field(&effective_guess_limit.to_le_bytes())
        .field(&request.maximum_seconds.to_le_bytes())
        .field(&request.maximum_memory_mb.to_le_bytes())
        .field(&request.maximum_disk_mb.to_le_bytes());
    for count in &request.answer_counts {
        hash.field(&count.to_le_bytes());
    }
    hash.field(raw_guesses).field(raw_answers);
    let executable = std::env::current_exe().context("failed to locate current executable")?;
    let mut file = fs::File::open(&executable)?;
    let length = file.metadata()?.len();
    hash.field_reader(&mut file, length)?;
    Ok(hash.finish_tagged())
}

fn formal_artifact_bytes(artifacts: &PolicyArtifactSet) -> Result<u64> {
    [
        &artifacts.manifest,
        &artifacts.values,
        &artifacts.policy,
        &artifacts.metadata,
        &artifacts.certificate,
        &artifacts.small_state_table,
        &artifacts.pattern_table,
        &artifacts.prior_spec,
    ]
    .into_iter()
    .try_fold(0u64, |total, path| {
        Ok(total.saturating_add(file_bytes(path)?))
    })
}

fn file_bytes(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?
        .len())
}

fn checkpoint_report(path: &Path, report: &mut FormalScaleReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    for _ in 0..4 {
        let bytes = serde_json::to_vec_pretty(report).context("serialize formal scale report")?;
        let length = bytes.len() as u64;
        if let Some(last) = report.points.last_mut() {
            if last.scale_checkpoint_bytes == length {
                return atomic_write(path, &bytes);
            }
            last.scale_checkpoint_bytes = length;
        } else {
            return atomic_write(path, &bytes);
        }
    }
    let bytes = serde_json::to_vec_pretty(report).context("serialize formal scale report")?;
    atomic_write(path, &bytes)
}

fn mib(value: u64) -> u64 {
    value.saturating_mul(1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_request_rejects_large_or_sparse_prefixes() {
        let mut request = FormalScaleRequest {
            answer_counts: vec![3, 4, 6],
            guess_limit: 8,
            maximum_seconds: 60,
            maximum_memory_mb: 512,
            maximum_disk_mb: 512,
            output: PathBuf::from("unused.json"),
        };
        validate_request(&request).expect("valid request");
        request.guess_limit = 0;
        validate_request(&request).expect("zero guess limit means all guesses");
        request.answer_counts = vec![0];
        assert!(validate_request(&request).is_err());
        request.answer_counts = vec![0, 2];
        assert!(validate_request(&request).is_err());
        request.answer_counts = vec![7, 8];
        assert!(validate_request(&request).is_err());
        request.answer_counts = vec![3, 6];
        assert!(validate_request(&request).is_err());
        request.answer_counts = vec![3, MAXIMUM_SAFE_SCALE_PREFIX + 1];
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn projection_uses_log_space_without_overflowing() {
        let points = [3usize, 4, 5]
            .into_iter()
            .map(|answer_count| FormalScalePoint {
                answer_count,
                guess_count: 8,
                policy_states: answer_count,
                certificate_states: answer_count * 2,
                build_millis: 10u128.pow(answer_count as u32 - 1),
                verify_millis: 1,
                states_per_second: 1.0,
                certificate_bytes: 10u64.pow(answer_count as u32),
                artifact_bytes: 1,
                scale_checkpoint_bytes: 1,
                process_peak_working_set_bytes: 1,
                manifest_hash: "test".to_string(),
            })
            .collect::<Vec<_>>();
        let projection = projection(&points, 47);
        assert_eq!(projection.target_answer_count, 47);
        assert!(
            projection
                .projected_log10_seconds
                .is_some_and(f64::is_finite)
        );
        assert!(
            projection
                .projected_log10_certificate_bytes
                .is_some_and(f64::is_finite)
        );
    }

    #[test]
    fn resumed_counts_and_projection_identity_are_validated_before_arithmetic() {
        let root = crate::test_support::TestDirectory::new("formal-scale-resume");
        let request = FormalScaleRequest {
            answer_counts: vec![2, 3, 4],
            guess_limit: 8,
            maximum_seconds: 60,
            maximum_memory_mb: 512,
            maximum_disk_mb: 512,
            output: root.path().join("report.json"),
        };
        let fingerprint = crate::identity::digest_bytes_tagged("scale-test", b"fixture");
        let mut baseline = load_or_initialize_report(&request, 8, 47, &fingerprint).unwrap();
        assert_eq!(baseline.full_model_projection.target_answer_count, 47);
        baseline.points = [2, 3]
            .into_iter()
            .map(|answer_count| FormalScalePoint {
                answer_count,
                guess_count: 8,
                policy_states: answer_count,
                certificate_states: answer_count,
                build_millis: 1000,
                verify_millis: 1,
                states_per_second: answer_count as f64,
                certificate_bytes: 10,
                artifact_bytes: 100,
                scale_checkpoint_bytes: 100,
                process_peak_working_set_bytes: 100,
                manifest_hash: fingerprint.clone(),
            })
            .collect();
        baseline.full_model_projection = projection(&baseline.points, 47);
        validate_checkpoint(&baseline).unwrap();
        for mutation in 0..8 {
            let mut changed = baseline.clone();
            match mutation {
                0 => changed.points[1].answer_count = 2,
                1 => changed.points.swap(0, 1),
                2 => changed.points[0].answer_count = 0,
                3 => changed.full_model_projection.target_answer_count = 2358,
                4 => changed.completed = true,
                5 => changed.points[0].manifest_hash = "stale".to_string(),
                6 => changed.points[0].guess_count = 0,
                _ => changed.points[0].states_per_second = -1.0,
            }
            fs::write(&request.output, serde_json::to_vec(&changed).unwrap()).unwrap();
            assert!(
                load_or_initialize_report(&request, 8, 47, &fingerprint).is_err(),
                "mutation {mutation}"
            );
        }
        let mut reversed = baseline.points.clone();
        reversed.reverse();
        assert!(predict_next_millis(&reversed, 4).is_none());
        assert!(predict_next_millis(&baseline.points, 2).is_none());
        assert!(validate_resumed_artifacts(&baseline, root.path(), &[], &[]).is_err());
    }

    #[test]
    fn cooperative_scale_budget_stops_and_checkpoint_retains_completed_points() {
        let budget = ScaleBudget {
            started: Instant::now(),
            previous_millis: 10,
            maximum_millis: 10,
            maximum_memory_bytes: u64::MAX,
            sample: std::sync::Mutex::new((None, None)),
        };
        assert!(budget.cancelled());
        assert!(budget.reason().contains("time budget"));
        assert!(budget.cancelled(), "cancellation is sticky");
        let root = crate::test_support::TestDirectory::new("formal-budget-checkpoint");
        let request = FormalScaleRequest {
            answer_counts: vec![2, 3],
            guess_limit: 8,
            maximum_seconds: 60,
            maximum_memory_mb: 512,
            maximum_disk_mb: 512,
            output: root.path().join("report.json"),
        };
        let fingerprint = crate::identity::digest_bytes_tagged("scale-budget", b"fixture");
        let mut report = load_or_initialize_report(&request, 8, 47, &fingerprint).unwrap();
        report.points.push(FormalScalePoint {
            answer_count: 2,
            guess_count: 8,
            policy_states: 2,
            certificate_states: 2,
            build_millis: 1000,
            verify_millis: 1,
            states_per_second: 2.0,
            certificate_bytes: 10,
            artifact_bytes: 20,
            scale_checkpoint_bytes: 0,
            process_peak_working_set_bytes: 100,
            manifest_hash: fingerprint.clone(),
        });
        report.full_model_projection = projection(&report.points, 47);
        report.stopped_reason = Some(budget.reason());
        checkpoint_report(&request.output, &mut report).unwrap();
        let reloaded = load_or_initialize_report(&request, 8, 47, &fingerprint).unwrap();
        assert_eq!(reloaded.points.len(), 1);
        assert_eq!(reloaded.points[0].answer_count, 2);
        assert!(!reloaded.completed);
        assert!(reloaded.stopped_reason.unwrap().contains("time budget"));
    }
}
