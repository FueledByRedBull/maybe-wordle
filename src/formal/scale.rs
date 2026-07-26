use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use super::{
    DEFAULT_FORMAL_MODEL_ID, FormalVerificationMode, PolicyArtifactSet, build_optimal_policy,
    verify_optimal_policy_with_mode,
};
use crate::{
    atomic_file::atomic_write,
    data::{ProjectPaths, read_word_list},
    identity::{CanonicalSha256, IDENTITY_FORMAT},
    process_memory::process_memory_snapshot,
};

const FORMAL_SCALE_FORMAT_VERSION: u32 = 2;
const MAXIMUM_SAFE_SCALE_PREFIX: usize = 16;
const FULL_MODEL_PROJECTION_ANSWERS: usize = 2_358;

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
    pub answer_counts: Vec<usize>,
    pub guess_limit: usize,
    pub maximum_seconds: u64,
    pub maximum_memory_mb: u64,
    pub maximum_disk_mb: u64,
    pub full_model_projection: FormalScaleProjection,
    pub points: Vec<FormalScalePoint>,
    pub completed: bool,
    pub stopped_reason: Option<String>,
}

pub fn benchmark_formal_scale(
    paths: &ProjectPaths,
    request: &FormalScaleRequest,
) -> Result<FormalScaleReport> {
    validate_request(request)?;
    let memory = process_memory_snapshot().ok_or_else(|| {
        anyhow!(
            "formal scale memory budgets are unsupported on this operating system; supported platforms are Windows, Linux, and macOS"
        )
    })?;
    let source_guesses = read_word_list(&paths.seed_guesses)?;
    let source_answers = read_word_list(&paths.seed_answers)?;
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
    let fingerprint = scale_input_fingerprint(paths, request, effective_guess_limit)?;
    let mut report = load_or_initialize_report(request, effective_guess_limit, &fingerprint)?;
    report.completed = false;
    report.stopped_reason = None;
    let namespace = fingerprint
        .strip_prefix("sha256-v1:")
        .unwrap_or(&fingerprint);
    let scratch_root = paths.root.join("target/formal-scale").join(namespace);
    fs::create_dir_all(&scratch_root)
        .with_context(|| format!("failed to create {}", scratch_root.display()))?;

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
        atomic_write(&artifacts.prior_spec, b"kind = \"uniform\"\n")?;

        let build_started = Instant::now();
        let build = build_optimal_policy(&point_paths, DEFAULT_FORMAL_MODEL_ID)?;
        let build_millis = build_started.elapsed().as_millis();
        let verify_started = Instant::now();
        let verify = verify_optimal_policy_with_mode(
            &point_paths,
            DEFAULT_FORMAL_MODEL_ID,
            FormalVerificationMode::Certificate,
        )?;
        let verify_millis = verify_started.elapsed().as_millis();
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
        report.full_model_projection = projection(&report.points);
        checkpoint_report(&request.output, &mut report)?;
    }

    report.completed = true;
    report.stopped_reason = None;
    report.full_model_projection = projection(&report.points);
    checkpoint_report(&request.output, &mut report)?;
    let final_memory = process_memory_snapshot().unwrap_or(memory);
    if final_memory.peak_working_set_bytes > mib(request.maximum_memory_mb) {
        bail!(
            "formal scale exceeded the {} MiB memory budget",
            request.maximum_memory_mb
        );
    }
    Ok(report)
}

fn validate_request(request: &FormalScaleRequest) -> Result<()> {
    if request.answer_counts.is_empty()
        || request.maximum_seconds == 0
        || request.maximum_memory_mb == 0
        || request.maximum_disk_mb == 0
    {
        bail!("formal scale counts, guess limit, and resource budgets must be positive");
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
    fingerprint: &str,
) -> Result<FormalScaleReport> {
    if request.output.exists() {
        let report: FormalScaleReport = serde_json::from_slice(
            &fs::read(&request.output)
                .with_context(|| format!("failed to read {}", request.output.display()))?,
        )
        .with_context(|| format!("failed to parse {}", request.output.display()))?;
        if report.format_version != FORMAL_SCALE_FORMAT_VERSION
            || report.identity_format != IDENTITY_FORMAT
            || report.input_fingerprint != fingerprint
            || report.answer_counts != request.answer_counts
            || report.guess_limit != effective_guess_limit
            || report.maximum_seconds != request.maximum_seconds
            || report.maximum_memory_mb != request.maximum_memory_mb
            || report.maximum_disk_mb != request.maximum_disk_mb
        {
            bail!(
                "formal scale checkpoint provenance does not match this run; choose a new output path"
            );
        }
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
        guess_limit: effective_guess_limit,
        maximum_seconds: request.maximum_seconds,
        maximum_memory_mb: request.maximum_memory_mb,
        maximum_disk_mb: request.maximum_disk_mb,
        full_model_projection: projection(&[]),
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
    let elapsed_millis = report
        .points
        .iter()
        .map(|point| point.build_millis + point.verify_millis)
        .sum::<u128>();
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
        .sum::<u64>();
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
        let answer_delta = (last.answer_count - previous.answer_count) as f64;
        (last_value / metric(previous).max(1.0))
            .powf(1.0 / answer_delta)
            .max(1.0)
    } else {
        2.0
    };
    Some(last_value * per_answer_growth.powf((next_answer_count - last.answer_count) as f64))
}

fn projection(points: &[FormalScalePoint]) -> FormalScaleProjection {
    FormalScaleProjection {
        target_answer_count: FULL_MODEL_PROJECTION_ANSWERS,
        method: "least-squares linear fit of log10(metric) against pinned answer count".to_string(),
        source_points: points.len(),
        projected_log10_seconds: log_linear_projection(points, |point| {
            point.build_millis.max(1) as f64 / 1_000.0
        }),
        projected_log10_certificate_bytes: log_linear_projection(points, |point| {
            point.certificate_bytes.max(1) as f64
        }),
    }
}

fn log_linear_projection(
    points: &[FormalScalePoint],
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
    Some(mean_y + slope * (FULL_MODEL_PROJECTION_ANSWERS as f64 - mean_x))
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
    paths: &ProjectPaths,
    request: &FormalScaleRequest,
    effective_guess_limit: usize,
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
    for path in [&paths.seed_guesses, &paths.seed_answers] {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        hash.field(&bytes);
    }
    let executable = std::env::current_exe().context("failed to locate current executable")?;
    let bytes = fs::read(&executable)
        .with_context(|| format!("failed to read {}", executable.display()))?;
    hash.field(&bytes);
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
        let projection = projection(&points);
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
}
