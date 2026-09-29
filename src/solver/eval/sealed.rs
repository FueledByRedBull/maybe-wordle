use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Days, NaiveDate, Utc};
use serde::Deserialize;

use crate::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, read_history_jsonl},
    experiments::{
        BootstrapConfig, DateRange, EvaluationPolicy, GameOutcome, RankedProbabilityObservation,
        default_diagnostic_suite, summarize_predictive_outcomes,
        summarize_ranked_probability_observations,
    },
    model::{ModelVariant, WeightMode},
    solver::{
        ExperimentGameResult, FrozenPredictiveCandidate, PredictiveBookUsage,
        ProspectiveEvaluationReport, ProspectiveFrozenCandidate, RollingComparisonArtifact,
        SealedTestMarker, SealedTestReport, Solver,
    },
};

use super::{
    PROSPECTIVE_EVALUATION_SCHEMA_VERSION, PROSPECTIVE_FROZEN_CANDIDATE_SCHEMA_VERSION,
    PROSPECTIVE_MARKER_SCHEMA_VERSION, PROSPECTIVE_REGISTRY_SCHEMA_VERSION,
    PROSPECTIVE_WINDOW_DAYS, ProspectiveRegistryMarker, ProspectiveWindowMarker,
    canonical_development_evaluation_plan, development_source_identity,
    ensure_development_source_identity, validate_exact_date_coverage,
    validate_rolling_comparison_artifact,
};

pub(super) fn create_once_marker(
    marker_path: &Path,
    serialized_marker: &[u8],
    label: &str,
) -> Result<()> {
    let mut marker_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(marker_path)
        .with_context(|| format!("failed to acquire {label} {}", marker_path.display()))?;
    std::io::Write::write_all(&mut marker_file, serialized_marker)
        .with_context(|| format!("failed to write {label} {}", marker_path.display()))?;
    marker_file
        .sync_all()
        .with_context(|| format!("failed to sync {label} {}", marker_path.display()))?;
    #[cfg(unix)]
    {
        let parent = marker_path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::File::open(parent)
            .with_context(|| {
                format!(
                    "failed to open {label} parent directory {}",
                    parent.display()
                )
            })?
            .sync_all()
            .with_context(|| {
                format!(
                    "failed to sync {label} parent directory {}",
                    parent.display()
                )
            })?;
    }
    Ok(())
}

pub(super) fn create_sealed_test_marker(
    marker_path: &Path,
    serialized_marker: &[u8],
) -> Result<()> {
    create_once_marker(marker_path, serialized_marker, "sealed-test marker")
}

fn sealed_window_marker_path(root: &Path, window: DateRange) -> PathBuf {
    root.join("benchmarks/predictive/sealed-windows")
        .join(format!("{}-{}-once.json", window.start, window.end))
}

pub(super) fn preflight_sealed_output_path(
    root: &Path,
    output: &Path,
    window: DateRange,
) -> Result<()> {
    let marker = sealed_window_marker_path(root, window);
    let ledger = marker.parent().expect("sealed ledger directory");
    let historical_marker = root.join("benchmarks/predictive/sealed-test-once.json");
    let prospective_registry = prospective_registry_marker_path(root);
    for path in [output, &marker, &historical_marker, &prospective_registry] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("create sealed preflight directory {}", parent.display())
            })?;
        }
    }
    #[cfg(windows)]
    if let Some(name) = output.file_name() {
        let name = name.to_string_lossy();
        if name.contains(':') || name.ends_with(['.', ' ']) {
            bail!("sealed output cannot use an alternate stream or ambiguous Windows filename");
        }
    }
    for path in [output, &marker] {
        match fs::symlink_metadata(path) {
            Ok(_) => bail!(
                "sealed output or irreversible marker already exists: {}",
                path.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect sealed path {}", path.display()));
            }
        }
    }
    let output_key = prospective_path_alias_key(output)?;
    for (label, protected) in [
        ("window marker", &marker),
        ("historical marker", &historical_marker),
        ("prospective registry", &prospective_registry),
    ] {
        if output_key == prospective_path_alias_key(protected)? {
            bail!("sealed output aliases the {label}");
        }
    }
    // Canonicalize existing parents so symlinks and `..` use filesystem semantics.
    // Reserve the whole ledger tree, including its cooperative-writer lock file.
    let ledger_key = prospective_path_key(&fs::canonicalize(ledger)?);
    if output_key == ledger_key || output_key.starts_with(&format!("{ledger_key}/")) {
        bail!("sealed output aliases the consumption ledger directory");
    }
    Ok(())
}

fn reject_overlapping_seal(requested: DateRange, consumed: DateRange) -> Result<()> {
    DateRange::new(consumed.start, consumed.end)?;
    if requested.start <= consumed.end && consumed.start <= requested.end {
        bail!(
            "sealed window {}..{} overlaps previously consumed {}..{}",
            requested.start,
            requested.end,
            consumed.start,
            consumed.end
        );
    }
    Ok(())
}

pub(super) fn acquire_sealed_window(root: &Path, marker: &SealedTestMarker) -> Result<PathBuf> {
    let marker_path = sealed_window_marker_path(root, marker.window);
    let directory = marker_path.parent().expect("window directory");
    fs::create_dir_all(directory)?;
    // Serialize claims, including overlapping windows with different date endpoints.
    let _lock = crate::atomic_file::acquire_edit_lock(&directory.join("registry"))?;
    let legacy_path = root.join("benchmarks/predictive/sealed-test-once.json");
    if legacy_path.exists() {
        #[derive(Deserialize)]
        struct LegacyMarker {
            output_path: String,
        }
        #[derive(Deserialize)]
        struct Window {
            sealed_test: DateRange,
        }
        #[derive(Deserialize)]
        struct LegacyReport {
            evaluation_plan: Window,
        }
        let legacy: LegacyMarker = serde_json::from_slice(&fs::read(&legacy_path)?)
            .context("unreadable historical sealed marker; keep the seal closed")?;
        let output = fs::canonicalize(root.join(legacy.output_path)).context(
            "historical sealed output is unavailable; cannot safely identify consumed window",
        )?;
        if !output.starts_with(fs::canonicalize(root)?) {
            bail!("historical sealed output must remain inside the repository");
        }
        let report: LegacyReport = serde_json::from_slice(&fs::read(output)?)
            .context("historical sealed report has no trustworthy window")?;
        reject_overlapping_seal(marker.window, report.evaluation_plan.sealed_test)?;
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            let prior: SealedTestMarker =
                serde_json::from_slice(&fs::read(&path)?).with_context(|| {
                    format!("invalid sealed ledger {}; keep seal closed", path.display())
                })?;
            if prior.schema_version != 2 {
                bail!("unsupported sealed ledger version");
            }
            reject_overlapping_seal(marker.window, prior.window)?;
        }
    }
    create_sealed_test_marker(&marker_path, &serde_json::to_vec_pretty(marker)?)?;
    Ok(marker_path)
}

pub(super) fn create_prospective_window_marker(
    marker_path: &Path,
    serialized_marker: &[u8],
) -> Result<()> {
    create_once_marker(marker_path, serialized_marker, "prospective-window marker")
}

pub(super) fn prospective_window_marker_path(root: &Path, window: DateRange) -> PathBuf {
    root.join("benchmarks/predictive").join(format!(
        "prospective-{}-{}-once.json",
        window.start, window.end
    ))
}

pub(super) fn prospective_registry_marker_path(root: &Path) -> PathBuf {
    // This release permits one prospective confirmation globally; the dated
    // window marker still records the exact window identity for that run.
    root.join("benchmarks/predictive/prospective-window-once.json")
}

fn prospective_canonical_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let file_name = absolute
        .file_name()
        .ok_or_else(|| anyhow!("prospective output path has no file name"))?;
    let parent = absolute
        .parent()
        .ok_or_else(|| anyhow!("prospective output path has no parent"))?;
    let canonical_parent = fs::canonicalize(parent).with_context(|| {
        format!(
            "failed to canonicalize prospective output parent {}",
            parent.display()
        )
    })?;
    Ok(canonical_parent.join(file_name))
}

fn prospective_path_key(path: &Path) -> String {
    let mut key = path.to_string_lossy().replace('\\', "/");
    if cfg!(windows) {
        key.make_ascii_lowercase();
    }
    key
}

fn prospective_path_alias_key(path: &Path) -> Result<String> {
    Ok(prospective_path_key(&prospective_canonical_path(path)?))
}

fn prospective_parent_alias_key(path: &Path) -> Result<String> {
    let canonical = prospective_canonical_path(path)?;
    let parent = canonical
        .parent()
        .ok_or_else(|| anyhow!("prospective output path has no canonical parent"))?;
    Ok(prospective_path_key(parent))
}

pub(super) fn preflight_prospective_output_path(
    output_path: &Path,
    marker_path: &Path,
    registry_path: &Path,
) -> Result<()> {
    for path in [output_path, marker_path, registry_path] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create prospective output parent {}",
                    parent.display()
                )
            })?;
        }
    }
    let output_key = prospective_path_alias_key(output_path)?;
    for (label, path) in [
        ("window marker", marker_path),
        ("global registry", registry_path),
    ] {
        if output_key == prospective_path_alias_key(path)? {
            bail!("prospective evaluation output aliases the {label} path");
        }
    }
    let output_parent_key = prospective_parent_alias_key(output_path)?;
    for (label, path) in [
        ("window marker", marker_path),
        ("global registry", registry_path),
    ] {
        if output_parent_key == prospective_parent_alias_key(path)? {
            bail!("prospective evaluation output aliases the {label} directory");
        }
    }
    if output_path.exists() || marker_path.exists() || registry_path.exists() {
        bail!(
            "prospective evaluation has already been reserved or completed; registry={} marker={} output={}",
            registry_path.display(),
            marker_path.display(),
            output_path.display()
        );
    }
    Ok(())
}

pub(super) fn prospective_window_for_freeze(
    policy: &EvaluationPolicy,
    frozen_at_utc: DateTime<Utc>,
    mut history_dates: impl Iterator<Item = NaiveDate>,
) -> Result<DateRange> {
    let freeze_date = frozen_at_utc.date_naive();
    let start = freeze_date
        .checked_add_days(Days::new(1))
        .ok_or_else(|| anyhow!("prospective window start overflowed"))?;
    let end = start
        .checked_add_days(Days::new(PROSPECTIVE_WINDOW_DAYS - 1))
        .ok_or_else(|| anyhow!("prospective window end overflowed"))?;
    let window = DateRange::new(start, end)?;
    if window.start <= freeze_date {
        bail!("prospective window must start strictly after UTC freeze date");
    }
    if window.start <= policy.sealed_test.end {
        bail!("prospective window overlaps the declared sealed test");
    }
    if policy
        .excluded_validation
        .iter()
        .any(|excluded| ranges_overlap(window, *excluded))
    {
        bail!("prospective window overlaps an excluded validation range");
    }
    if history_dates.any(|date| window.contains(date)) {
        bail!("history already has a target in the prospective window");
    }
    if window.days() != PROSPECTIVE_WINDOW_DAYS {
        bail!("prospective window must contain exactly 30 inclusive days");
    }
    Ok(window)
}

pub(super) fn prospective_freeze_fingerprint(
    freeze_fingerprint: &str,
    input_fingerprint: &str,
    config_fingerprint: &str,
    comparison_fingerprint: &str,
    pre_window_history_fingerprint: &str,
    frozen_at_utc: DateTime<Utc>,
    window: DateRange,
) -> String {
    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-prospective-freeze-v1");
    hash.field(freeze_fingerprint.as_bytes())
        .field(input_fingerprint.as_bytes())
        .field(config_fingerprint.as_bytes())
        .field(comparison_fingerprint.as_bytes())
        .field(pre_window_history_fingerprint.as_bytes())
        .field(frozen_at_utc.to_rfc3339().as_bytes())
        .field(window.start.to_string().as_bytes())
        .field(window.end.to_string().as_bytes());
    hash.finish_tagged()
}

fn prospective_history_fingerprint_with_tag(
    history: &[NytDailyEntry],
    window: DateRange,
    tag: &str,
) -> Result<String> {
    let mut entries = history
        .iter()
        .filter(|entry| window.contains(entry.print_date))
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.print_date);
    let mut bytes = Vec::new();
    for entry in entries {
        serde_json::to_writer(&mut bytes, entry).context("serialize prospective history row")?;
        bytes.push(b'\n');
    }
    Ok(crate::identity::digest_bytes_tagged(tag, &bytes))
}

pub(super) fn prospective_history_fingerprint(
    history: &[NytDailyEntry],
    window: DateRange,
) -> Result<String> {
    prospective_history_fingerprint_with_tag(
        history,
        window,
        "maybe-wordle-prospective-window-data-v1",
    )
}

fn prospective_window_data_fingerprint(
    history: &[NytDailyEntry],
    window: DateRange,
) -> Result<String> {
    validate_exact_date_coverage(window, history.iter().map(|entry| entry.print_date))?;
    prospective_history_fingerprint(history, window)
}

pub(super) fn prospective_pre_window_history_fingerprint(
    history: &[NytDailyEntry],
    start: NaiveDate,
    freeze_date: NaiveDate,
) -> Result<String> {
    let pre_window = DateRange::new(start, freeze_date)?;
    validate_exact_date_coverage(pre_window, history.iter().map(|entry| entry.print_date))?;
    prospective_history_fingerprint_with_tag(
        history,
        pre_window,
        "maybe-wordle-prospective-pre-window-data-v1",
    )
}

pub(super) fn prospective_pre_window_history_start(
    development_cutoff: NaiveDate,
) -> Result<NaiveDate> {
    development_cutoff
        .checked_add_days(Days::new(1))
        .ok_or_else(|| anyhow!("prospective pre-window history start overflowed"))
}

pub(super) fn ensure_prospective_window_elapsed(
    window: DateRange,
    consumed_at_utc: DateTime<Utc>,
) -> Result<()> {
    if consumed_at_utc.date_naive() <= window.end {
        bail!("prospective window has not fully elapsed in UTC");
    }
    Ok(())
}

fn ranges_overlap(left: DateRange, right: DateRange) -> bool {
    left.start <= right.end && right.start <= left.end
}

impl ProspectiveFrozenCandidate {
    pub fn validate_identity(&self) -> Result<()> {
        if self.schema_version != PROSPECTIVE_FROZEN_CANDIDATE_SCHEMA_VERSION
            || self.identity_format != crate::identity::IDENTITY_FORMAT
            || !crate::identity::is_tagged_digest(&self.pre_window_history_fingerprint)
        {
            bail!("prospective frozen candidate uses an unsupported identity format");
        }
        self.frozen.validate_identity()?;
        let expected_start = self
            .frozen_at_utc
            .date_naive()
            .checked_add_days(Days::new(1))
            .ok_or_else(|| anyhow!("prospective window start overflowed"))?;
        let expected_end = expected_start
            .checked_add_days(Days::new(PROSPECTIVE_WINDOW_DAYS - 1))
            .ok_or_else(|| anyhow!("prospective window end overflowed"))?;
        if self.window != DateRange::new(expected_start, expected_end)?
            || self.window.days() != PROSPECTIVE_WINDOW_DAYS
            || self.window.start <= self.frozen.evaluation_plan.sealed_test.end
        {
            bail!("prospective frozen candidate has an invalid later window");
        }
        let expected = prospective_freeze_fingerprint(
            &self.frozen.freeze_fingerprint,
            &self.frozen.input_fingerprint,
            &self.frozen.config_fingerprint,
            &self.frozen.development_comparison_fingerprint,
            &self.pre_window_history_fingerprint,
            self.frozen_at_utc,
            self.window,
        );
        if self.window_fingerprint != expected {
            bail!("prospective frozen candidate fingerprint mismatch");
        }
        Ok(())
    }
}

impl Solver {
    pub fn freeze_predictive_candidate(
        paths: &ProjectPaths,
        config_path: &Path,
        comparison_path: &Path,
    ) -> Result<FrozenPredictiveCandidate> {
        let config = PriorConfig::load(config_path)?;
        let config_toml =
            toml::to_string_pretty(&config).context("serialize frozen candidate config")?;
        let comparison_bytes = fs::read(comparison_path)
            .with_context(|| format!("failed to read {}", comparison_path.display()))?;
        let comparison: RollingComparisonArtifact = serde_json::from_slice(&comparison_bytes)
            .with_context(|| format!("failed to parse {}", comparison_path.display()))?;
        validate_rolling_comparison_artifact(&comparison)?;
        let evaluation_plan =
            canonical_development_evaluation_plan(paths, "freezing predictive candidate")?;
        let input_fingerprint =
            development_source_identity(paths, evaluation_plan.development.end)?;
        if comparison.input_fingerprint != input_fingerprint
            || comparison.evaluation_plan != evaluation_plan
            || comparison.sealed_test_evaluated
        {
            bail!("development comparison is stale, uses another plan, or touched the sealed test");
        }
        if comparison.candidate.config_toml != config_toml {
            bail!("frozen config does not match the comparison candidate");
        }
        let candidate_failures = comparison.candidate.aggregate.unsolved_games
            + comparison.candidate.aggregate.coverage_gaps;
        if comparison.candidate.aggregate.scheduled_games == 0
            || comparison.candidate.aggregate.modeled_games
                != comparison.candidate.aggregate.scheduled_games
            || candidate_failures != 0
        {
            bail!("candidate cannot be frozen without full coverage and zero failures");
        }
        if comparison.candidate_minus_baseline.candidate_minus_baseline >= 0.0
            || comparison.candidate_minus_baseline.ci95.upper >= 0.0
        {
            bail!(
                "candidate cannot be frozen unless its paired development interval is entirely below zero"
            );
        }
        let development_comparison_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-development-comparison-v1",
            &comparison_bytes,
        );
        let config_fingerprint = comparison.candidate.config_fingerprint.clone();
        let evaluation_artifact_policy = "artifact_free".to_string();
        let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-frozen-candidate-v1");
        hash.field(input_fingerprint.as_bytes())
            .field(config_fingerprint.as_bytes())
            .field(development_comparison_fingerprint.as_bytes())
            .field(comparison.candidate.label.as_bytes())
            .field(evaluation_artifact_policy.as_bytes())
            .field(
                &serde_json::to_vec(&evaluation_plan)
                    .context("serialize frozen evaluation plan identity")?,
            )
            .field(
                &serde_json::to_vec(&comparison.candidate.aggregate)
                    .context("serialize frozen development metrics identity")?,
            )
            .field(
                &serde_json::to_vec(&comparison.candidate_minus_baseline)
                    .context("serialize frozen paired difference identity")?,
            );
        let frozen = FrozenPredictiveCandidate {
            schema_version: 2,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            freeze_fingerprint: hash.finish_tagged(),
            evaluation_plan,
            config_toml,
            config_fingerprint,
            candidate_label: comparison.candidate.label,
            development_comparison_fingerprint,
            development_metrics: comparison.candidate.aggregate,
            development_paired_difference: comparison.candidate_minus_baseline,
            evaluation_artifact_policy,
            sealed_test_evaluated: false,
        };
        frozen.validate_identity()?;
        Ok(frozen)
    }

    pub fn freeze_prospective_candidate(
        paths: &ProjectPaths,
        config_path: &Path,
        comparison_path: &Path,
        frozen_at_utc: DateTime<Utc>,
    ) -> Result<ProspectiveFrozenCandidate> {
        let frozen = Self::freeze_predictive_candidate(paths, config_path, comparison_path)?;
        let policy = EvaluationPolicy::load(&paths.root.join("config/evaluation.toml"))?;
        let history = read_history_jsonl(&paths.raw_history)?;
        let window = prospective_window_for_freeze(
            &policy,
            frozen_at_utc,
            history.iter().map(|entry| entry.print_date),
        )?;
        let pre_window_history_fingerprint = prospective_pre_window_history_fingerprint(
            &history,
            prospective_pre_window_history_start(frozen.evaluation_plan.development.end)?,
            frozen_at_utc.date_naive(),
        )?;
        let window_fingerprint = prospective_freeze_fingerprint(
            &frozen.freeze_fingerprint,
            &frozen.input_fingerprint,
            &frozen.config_fingerprint,
            &frozen.development_comparison_fingerprint,
            &pre_window_history_fingerprint,
            frozen_at_utc,
            window,
        );
        let prospective = ProspectiveFrozenCandidate {
            schema_version: PROSPECTIVE_FROZEN_CANDIDATE_SCHEMA_VERSION,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            frozen,
            frozen_at_utc,
            window,
            pre_window_history_fingerprint,
            window_fingerprint,
        };
        prospective.validate_identity()?;
        Ok(prospective)
    }

    pub fn write_prospective_frozen_candidate(
        output_path: &Path,
        frozen: &ProspectiveFrozenCandidate,
    ) -> Result<()> {
        frozen.validate_identity()?;
        let serialized = serde_json::to_vec_pretty(frozen)
            .context("failed to serialize prospective frozen candidate")?;
        create_once_marker(output_path, &serialized, "prospective frozen candidate")
    }

    pub fn evaluate_prospective_candidate(
        paths: &ProjectPaths,
        frozen: &ProspectiveFrozenCandidate,
        output_path: &Path,
    ) -> Result<ProspectiveEvaluationReport> {
        Self::evaluate_prospective_candidate_at(paths, frozen, output_path, Utc::now())
    }

    fn evaluate_prospective_candidate_at(
        paths: &ProjectPaths,
        frozen: &ProspectiveFrozenCandidate,
        output_path: &Path,
        consumed_at_utc: DateTime<Utc>,
    ) -> Result<ProspectiveEvaluationReport> {
        frozen.validate_identity()?;
        if development_source_identity(paths, frozen.frozen.evaluation_plan.development.end)?
            != frozen.frozen.input_fingerprint
        {
            bail!("source, executable, or development data changed after prospective freeze");
        }
        let plan = canonical_development_evaluation_plan(paths, "opening prospective window")?;
        if plan != frozen.frozen.evaluation_plan {
            bail!("evaluation plan changed after prospective freeze");
        }
        let policy = EvaluationPolicy::load(&paths.root.join("config/evaluation.toml"))?;
        let expected_window =
            prospective_window_for_freeze(&policy, frozen.frozen_at_utc, [].into_iter())?;
        if expected_window != frozen.window {
            bail!("prospective window no longer matches the declared policy");
        }
        ensure_prospective_window_elapsed(frozen.window, consumed_at_utc)?;
        let output_path = if output_path.is_absolute() {
            output_path.to_path_buf()
        } else {
            paths.root.join(output_path)
        };
        let marker_path = prospective_window_marker_path(&paths.root, frozen.window);
        let registry_path = prospective_registry_marker_path(&paths.root);
        preflight_prospective_output_path(&output_path, &marker_path, &registry_path)?;
        let config: PriorConfig =
            toml::from_str(&frozen.frozen.config_toml).context("parse frozen candidate config")?;
        let solver = Self::from_paths_with_settings(
            paths,
            &config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let initial_history = read_history_jsonl(&paths.raw_history)?;
        let pre_window_history_fingerprint = prospective_pre_window_history_fingerprint(
            &initial_history,
            prospective_pre_window_history_start(frozen.frozen.evaluation_plan.development.end)?,
            frozen.frozen_at_utc.date_naive(),
        )?;
        if pre_window_history_fingerprint != frozen.pre_window_history_fingerprint {
            bail!("prospective pre-window history changed after freeze");
        }
        let window_data_fingerprint =
            prospective_window_data_fingerprint(&initial_history, frozen.window)?;
        let relative_output = output_path
            .strip_prefix(&paths.root)
            .unwrap_or(&output_path)
            .to_string_lossy()
            .replace('\\', "/");
        let registry = ProspectiveRegistryMarker {
            schema_version: PROSPECTIVE_REGISTRY_SCHEMA_VERSION,
            freeze_fingerprint: frozen.frozen.freeze_fingerprint.clone(),
            window_fingerprint: frozen.window_fingerprint.clone(),
            pre_window_history_fingerprint: frozen.pre_window_history_fingerprint.clone(),
            window: frozen.window,
            reserved_at_utc: consumed_at_utc,
        };
        create_once_marker(
            &registry_path,
            &serde_json::to_vec_pretty(&registry).context("serialize prospective registry")?,
            "prospective registry",
        )?;
        let mut marker = ProspectiveWindowMarker {
            schema_version: PROSPECTIVE_MARKER_SCHEMA_VERSION,
            freeze_fingerprint: frozen.frozen.freeze_fingerprint.clone(),
            window_fingerprint: frozen.window_fingerprint.clone(),
            input_fingerprint: frozen.frozen.input_fingerprint.clone(),
            config_fingerprint: frozen.frozen.config_fingerprint.clone(),
            pre_window_history_fingerprint: frozen.pre_window_history_fingerprint.clone(),
            window: frozen.window,
            window_data_fingerprint: window_data_fingerprint.clone(),
            output_path: relative_output,
            consumed_at_utc,
            status: "started_irreversible".to_string(),
        };
        create_prospective_window_marker(
            &marker_path,
            &serde_json::to_vec_pretty(&marker).context("serialize prospective marker")?,
        )?;

        let report = solver.backtest_detailed_with_book_usage(
            frozen.window.start,
            frozen.window.end,
            5,
            PredictiveBookUsage::None,
        )?;
        ensure_development_source_identity(
            paths,
            frozen.frozen.evaluation_plan.development.end,
            &frozen.frozen.input_fingerprint,
        )?;
        let current_history = read_history_jsonl(&paths.raw_history)?;
        if prospective_pre_window_history_fingerprint(
            &current_history,
            prospective_pre_window_history_start(frozen.frozen.evaluation_plan.development.end)?,
            frozen.frozen_at_utc.date_naive(),
        )? != frozen.pre_window_history_fingerprint
            || prospective_window_data_fingerprint(&current_history, frozen.window)?
                != window_data_fingerprint
        {
            bail!("prospective history changed during evaluation; marker remains reserved");
        }
        let games = report
            .runs
            .iter()
            .map(|run| ExperimentGameResult {
                target: run.target.clone(),
                outcome: if run.steps.is_empty() {
                    GameOutcome::coverage_gap(run.date)
                } else if run.solved {
                    GameOutcome::solved(run.date, run.steps.len())
                } else {
                    GameOutcome::unsolved(run.date, run.steps.len())
                },
                path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                finite_search_steps: run
                    .steps
                    .iter()
                    .filter_map(|step| step.finite_search.clone())
                    .collect(),
                prior_strata: None,
                posterior_calibration: Vec::new(),
            })
            .collect::<Vec<_>>();
        let outcomes = games.iter().map(|game| game.outcome).collect::<Vec<_>>();
        let metrics = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let prior_observations = solver
            .history_dates
            .iter()
            .filter(|entry| frozen.window.contains(entry.print_date))
            .filter_map(|entry| {
                solver
                    .initial_prior_metrics(&entry.solution, entry.print_date)
                    .map(|metrics| RankedProbabilityObservation {
                        target_rank: metrics.target_rank,
                        top_probability: metrics.top_probability,
                        top_prediction_correct: metrics.top_prediction_correct,
                    })
            })
            .collect::<Vec<_>>();
        let prospective = ProspectiveEvaluationReport {
            schema_version: PROSPECTIVE_EVALUATION_SCHEMA_VERSION,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            freeze_fingerprint: frozen.frozen.freeze_fingerprint.clone(),
            input_fingerprint: frozen.frozen.input_fingerprint.clone(),
            config_fingerprint: frozen.frozen.config_fingerprint.clone(),
            window_fingerprint: frozen.window_fingerprint.clone(),
            pre_window_history_fingerprint: frozen.pre_window_history_fingerprint.clone(),
            window: frozen.window,
            frozen_at_utc: frozen.frozen_at_utc,
            consumed_at_utc,
            evaluation_artifact_policy: "artifact_free".to_string(),
            prospective_evaluated: true,
            evaluated_once: true,
            window_data_fingerprint: window_data_fingerprint.clone(),
            metrics,
            prior_evidence: (!prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            execution: Self::execution_telemetry(&report.runs),
            games,
            latency_p95_ms: solver.benchmark_predictive_latency(
                frozen.frozen.evaluation_plan.development.end,
                default_diagnostic_suite()?.latency.evidence_runs,
            )?,
        };
        ensure_development_source_identity(
            paths,
            frozen.frozen.evaluation_plan.development.end,
            &frozen.frozen.input_fingerprint,
        )?;
        let final_history = read_history_jsonl(&paths.raw_history)?;
        if prospective_pre_window_history_fingerprint(
            &final_history,
            prospective_pre_window_history_start(frozen.frozen.evaluation_plan.development.end)?,
            frozen.frozen_at_utc.date_naive(),
        )? != frozen.pre_window_history_fingerprint
            || prospective_window_data_fingerprint(&final_history, frozen.window)?
                != window_data_fingerprint
        {
            bail!("prospective history changed before report publication; marker remains reserved");
        }
        crate::atomic_file::atomic_write(
            &output_path,
            &serde_json::to_vec_pretty(&prospective)
                .context("serialize prospective evaluation report")?,
        )?;
        marker.status = "completed".to_string();
        crate::atomic_file::atomic_write(
            &marker_path,
            &serde_json::to_vec_pretty(&marker).context("serialize prospective marker")?,
        )?;
        Ok(prospective)
    }

    pub fn evaluate_frozen_candidate_on_sealed_test(
        paths: &ProjectPaths,
        frozen: &FrozenPredictiveCandidate,
        output_path: &Path,
    ) -> Result<SealedTestReport> {
        frozen.validate_identity()?;
        if development_source_identity(paths, frozen.evaluation_plan.development.end)?
            != frozen.input_fingerprint
        {
            bail!(
                "source, executable, or data changed after candidate freeze; the sealed test remains closed"
            );
        }
        let plan = canonical_development_evaluation_plan(paths, "opening the sealed final test")?;
        if plan != frozen.evaluation_plan {
            bail!("evaluation plan changed after candidate freeze");
        }
        let output_path = if output_path.is_absolute() {
            output_path.to_path_buf()
        } else {
            paths.root.join(output_path)
        };
        preflight_sealed_output_path(&paths.root, &output_path, plan.sealed_test)?;
        let config: PriorConfig =
            toml::from_str(&frozen.config_toml).context("parse frozen candidate config")?;
        let solver = Self::from_paths_with_settings(
            paths,
            &config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        validate_exact_date_coverage(
            plan.sealed_test,
            solver.history_dates.iter().map(|entry| entry.print_date),
        )?;
        let window_data_fingerprint =
            prospective_window_data_fingerprint(&solver.history_dates, plan.sealed_test)?;
        let current_history = read_history_jsonl(&paths.raw_history)?;
        if prospective_window_data_fingerprint(&current_history, plan.sealed_test)?
            != window_data_fingerprint
        {
            bail!("sealed input changed during solver preflight; seal remains closed");
        }
        let relative_output = output_path
            .strip_prefix(&paths.root)
            .unwrap_or(&output_path)
            .to_string_lossy()
            .replace('\\', "/");
        let mut marker = SealedTestMarker {
            schema_version: 2,
            window: plan.sealed_test,
            evaluation_contract_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-sealed-evaluation-contract-v1",
                &serde_json::to_vec(&(&plan, &frozen.evaluation_artifact_policy))?,
            ),
            window_data_fingerprint: window_data_fingerprint.clone(),
            freeze_fingerprint: frozen.freeze_fingerprint.clone(),
            output_path: relative_output,
            status: "started_irreversible".to_string(),
        };
        let marker_path = acquire_sealed_window(&paths.root, &marker)?;

        let report = solver.backtest_detailed_with_book_usage(
            plan.sealed_test.start,
            plan.sealed_test.end,
            5,
            PredictiveBookUsage::None,
        )?;
        let games = report
            .runs
            .iter()
            .map(|run| ExperimentGameResult {
                target: run.target.clone(),
                outcome: if run.steps.is_empty() {
                    GameOutcome::coverage_gap(run.date)
                } else if run.solved {
                    GameOutcome::solved(run.date, run.steps.len())
                } else {
                    GameOutcome::unsolved(run.date, run.steps.len())
                },
                path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                finite_search_steps: run
                    .steps
                    .iter()
                    .filter_map(|step| step.finite_search.clone())
                    .collect(),
                prior_strata: None,
                posterior_calibration: Vec::new(),
            })
            .collect::<Vec<_>>();
        let outcomes = games.iter().map(|game| game.outcome).collect::<Vec<_>>();
        let metrics = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let prior_observations = solver
            .history_dates
            .iter()
            .filter(|entry| {
                entry.print_date >= plan.sealed_test.start
                    && entry.print_date <= plan.sealed_test.end
            })
            .filter_map(|entry| {
                solver
                    .initial_prior_metrics(&entry.solution, entry.print_date)
                    .map(|metrics| RankedProbabilityObservation {
                        target_rank: metrics.target_rank,
                        top_probability: metrics.top_probability,
                        top_prediction_correct: metrics.top_prediction_correct,
                    })
            })
            .collect::<Vec<_>>();
        let sealed = SealedTestReport {
            schema_version: 2,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            freeze_fingerprint: frozen.freeze_fingerprint.clone(),
            input_fingerprint: frozen.input_fingerprint.clone(),
            config_fingerprint: frozen.config_fingerprint.clone(),
            evaluation_plan: plan,
            evaluation_artifact_policy: frozen.evaluation_artifact_policy.clone(),
            sealed_test_evaluated: true,
            evaluated_once: true,
            metrics,
            prior_evidence: (!prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            execution: Self::execution_telemetry(&report.runs),
            games,
            latency_p95_ms: solver.benchmark_predictive_latency(
                frozen.evaluation_plan.development.end,
                default_diagnostic_suite()?.latency.evidence_runs,
            )?,
        };
        ensure_development_source_identity(
            paths,
            frozen.evaluation_plan.development.end,
            &frozen.input_fingerprint,
        )?;
        let final_history = read_history_jsonl(&paths.raw_history)?;
        if prospective_window_data_fingerprint(&final_history, frozen.evaluation_plan.sealed_test)?
            != window_data_fingerprint
        {
            bail!(
                "sealed input changed during evaluation; irreversible ownership remains reserved"
            );
        }
        create_once_marker(
            &output_path,
            &serde_json::to_vec_pretty(&sealed).context("serialize sealed-test report")?,
            "sealed-test output",
        )?;
        marker.status = "completed".to_string();
        crate::atomic_file::atomic_write(
            &marker_path,
            &serde_json::to_vec_pretty(&marker).context("serialize sealed-test marker")?,
        )?;
        Ok(sealed)
    }
}
