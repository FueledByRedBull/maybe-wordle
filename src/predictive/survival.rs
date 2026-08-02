//! Experimental, leakage-safe discrete-time survival model for answer reuse.
//!
//! This module is deliberately separate from the production logistic prior.  It
//! provides the data preparation, fit, scoring, and evidence containers needed
//! to evaluate a survival/reuse hypothesis without changing the current solver
//! behaviour.  Dates are represented as half-open risk intervals: an interval
//! `[entry_date, exit_date)` contributes one risk row per day and, when
//! `reused` is true, an event on the final row.  Right-censored intervals never
//! contribute an event.  `never_used` observations are tracked as a separate
//! mass and are not silently treated as censored reuse events.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use chrono::{Duration, NaiveDate};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SURVIVAL_SCHEMA_VERSION: u32 = 1;
const FINGERPRINT_PREFIX: &str = "sha256-v1:";
const DEFAULT_DATE: NaiveDate = NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid date");

/// Errors returned while validating, preparing, fitting, or scoring a model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SurvivalError {
    InvalidInput(String),
    Leakage(String),
    InvalidArtifact(String),
    Serialization(String),
    Numeric(String),
}

impl fmt::Display for SurvivalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid survival input: {message}"),
            Self::Leakage(message) => write!(f, "survival fold leakage: {message}"),
            Self::InvalidArtifact(message) => write!(f, "invalid survival artifact: {message}"),
            Self::Serialization(message) => write!(f, "survival serialization error: {message}"),
            Self::Numeric(message) => write!(f, "survival numeric error: {message}"),
        }
    }
}

impl Error for SurvivalError {}

pub type SurvivalResult<T> = Result<T, SurvivalError>;

/// The mutually exclusive interpretation of an interval observation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStatus {
    Reused,
    RightCensored,
    NeverUsed,
}

impl ObservationStatus {
    fn sort_key(self) -> u8 {
        match self {
            Self::Reused => 0,
            Self::RightCensored => 1,
            Self::NeverUsed => 2,
        }
    }
}

/// A right-censored interval or a reuse event for one word.
///
/// `entry_date` is the first date on risk and `exit_date` is exclusive.  A
/// reuse event therefore has at least one day of exposure.  For a word that
/// has never appeared, the dates are metadata only and may be equal; that mass
/// is explicitly kept out of the reuse hazard fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurvivalObservation {
    pub word: String,
    pub entry_date: NaiveDate,
    pub exit_date: NaiveDate,
    pub reused: bool,
    pub never_used: bool,
    pub policy_era: String,
    pub left_truncated: bool,
    #[serde(default)]
    pub elapsed_offset_days: usize,
    #[serde(default = "default_mass")]
    pub mass: f64,
}

fn default_mass() -> f64 {
    1.0
}

impl Default for SurvivalObservation {
    fn default() -> Self {
        Self {
            word: String::new(),
            entry_date: DEFAULT_DATE,
            exit_date: DEFAULT_DATE,
            reused: false,
            never_used: true,
            policy_era: "default".to_string(),
            left_truncated: false,
            elapsed_offset_days: 0,
            mass: 1.0,
        }
    }
}

impl SurvivalObservation {
    pub fn reused_observation(
        word: impl Into<String>,
        entry_date: NaiveDate,
        exit_date: NaiveDate,
        policy_era: impl Into<String>,
    ) -> Self {
        Self {
            word: word.into(),
            entry_date,
            exit_date,
            reused: true,
            never_used: false,
            policy_era: policy_era.into(),
            left_truncated: false,
            elapsed_offset_days: 0,
            mass: 1.0,
        }
    }

    pub fn right_censored(
        word: impl Into<String>,
        entry_date: NaiveDate,
        exit_date: NaiveDate,
        policy_era: impl Into<String>,
    ) -> Self {
        Self {
            word: word.into(),
            entry_date,
            exit_date,
            reused: false,
            never_used: false,
            policy_era: policy_era.into(),
            left_truncated: false,
            elapsed_offset_days: 0,
            mass: 1.0,
        }
    }

    pub fn never_used(
        word: impl Into<String>,
        as_of: NaiveDate,
        policy_era: impl Into<String>,
    ) -> Self {
        Self {
            word: word.into(),
            entry_date: as_of,
            exit_date: as_of,
            reused: false,
            never_used: true,
            policy_era: policy_era.into(),
            left_truncated: false,
            elapsed_offset_days: 0,
            mass: 1.0,
        }
    }

    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

    pub fn with_left_truncation(mut self, left_truncated: bool) -> Self {
        self.left_truncated = left_truncated;
        self
    }

    pub fn with_elapsed_offset(mut self, elapsed_offset_days: usize) -> Self {
        self.elapsed_offset_days = elapsed_offset_days;
        self
    }

    pub fn status(&self) -> ObservationStatus {
        if self.never_used {
            ObservationStatus::NeverUsed
        } else if self.reused {
            ObservationStatus::Reused
        } else {
            ObservationStatus::RightCensored
        }
    }

    pub fn duration_days(&self) -> i64 {
        (self.exit_date - self.entry_date).num_days()
    }

    pub fn validate(&self) -> SurvivalResult<()> {
        if self.word.trim().is_empty() {
            return Err(SurvivalError::InvalidInput(
                "observation word must not be empty".to_string(),
            ));
        }
        if self.policy_era.trim().is_empty() {
            return Err(SurvivalError::InvalidInput(format!(
                "observation {} has an empty policy era",
                self.word
            )));
        }
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err(SurvivalError::InvalidInput(format!(
                "observation {} has non-positive/non-finite mass {}",
                self.word, self.mass
            )));
        }
        if self.reused && self.never_used {
            return Err(SurvivalError::InvalidInput(format!(
                "observation {} cannot be both reused and never used",
                self.word
            )));
        }
        if !self.never_used && self.exit_date <= self.entry_date {
            return Err(SurvivalError::InvalidInput(format!(
                "observation {} must have a positive interval",
                self.word
            )));
        }
        Ok(())
    }
}

/// A policy-era interval.  `end` is exclusive; `None` means open-ended.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PolicyEra {
    pub id: String,
    pub start: NaiveDate,
    pub end: Option<NaiveDate>,
}

impl PolicyEra {
    pub fn new(id: impl Into<String>, start: NaiveDate, end: Option<NaiveDate>) -> Self {
        Self {
            id: id.into(),
            start,
            end,
        }
    }

    pub fn contains(&self, date: NaiveDate) -> bool {
        date >= self.start && self.end.is_none_or(|end| date < end)
    }

    pub fn validate(&self) -> SurvivalResult<()> {
        if self.id.trim().is_empty() {
            return Err(SurvivalError::InvalidInput(
                "policy-era id must not be empty".to_string(),
            ));
        }
        if self.end.is_some_and(|end| end <= self.start) {
            return Err(SurvivalError::InvalidInput(format!(
                "policy era {} has an end not after its start",
                self.id
            )));
        }
        Ok(())
    }
}

pub fn validate_policy_eras(eras: &[PolicyEra]) -> SurvivalResult<()> {
    if eras.is_empty() {
        return Err(SurvivalError::InvalidInput(
            "at least one policy era is required".to_string(),
        ));
    }
    let mut ordered = eras.to_vec();
    ordered.sort_by_key(|era| (era.start, era.id.clone()));
    let mut ids = BTreeSet::new();
    for era in &ordered {
        era.validate()?;
        if !ids.insert(era.id.clone()) {
            return Err(SurvivalError::InvalidInput(format!(
                "duplicate policy-era id {}",
                era.id
            )));
        }
    }
    for pair in ordered.windows(2) {
        if pair[0].end != Some(pair[1].start) {
            return Err(SurvivalError::InvalidInput(format!(
                "policy eras {} and {} must be contiguous and non-overlapping",
                pair[0].id, pair[1].id
            )));
        }
    }
    Ok(())
}

fn policy_era_for_date(eras: &[PolicyEra], date: NaiveDate) -> Option<&str> {
    eras.iter()
        .find(|era| era.contains(date))
        .map(|era| era.id.as_str())
}

/// Metadata describing what happened to histories before the training window.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LeftTruncationMetadata {
    pub origin: NaiveDate,
    pub retained_pre_origin: bool,
    pub description: String,
}

impl Default for LeftTruncationMetadata {
    fn default() -> Self {
        Self {
            origin: DEFAULT_DATE,
            retained_pre_origin: false,
            description: "No pre-origin rows retained".to_string(),
        }
    }
}

impl LeftTruncationMetadata {
    pub fn validate(&self) -> SurvivalResult<()> {
        if self.description.trim().is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "left-truncation description must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Chronological training/validation boundary used for deterministic folds.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FoldSpec {
    pub id: String,
    pub training_start: Option<NaiveDate>,
    pub training_end: NaiveDate,
    pub validation_start: NaiveDate,
    pub validation_end: NaiveDate,
}

impl FoldSpec {
    pub fn new(
        id: impl Into<String>,
        training_start: Option<NaiveDate>,
        training_end: NaiveDate,
        validation_start: NaiveDate,
        validation_end: NaiveDate,
    ) -> Self {
        Self {
            id: id.into(),
            training_start,
            training_end,
            validation_start,
            validation_end,
        }
    }

    pub fn validate(&self) -> SurvivalResult<()> {
        if self.id.trim().is_empty() {
            return Err(SurvivalError::InvalidInput(
                "fold id must not be empty".to_string(),
            ));
        }
        if self
            .training_start
            .is_some_and(|start| start > self.training_end)
        {
            return Err(SurvivalError::InvalidInput(
                "training_start must not be after training_end".to_string(),
            ));
        }
        if self.training_end >= self.validation_start {
            return Err(SurvivalError::Leakage(
                "training_end must be before validation_start".to_string(),
            ));
        }
        if self.validation_start > self.validation_end {
            return Err(SurvivalError::InvalidInput(
                "validation_start must not be after validation_end".to_string(),
            ));
        }
        Ok(())
    }
}

/// A deterministic, fold-local snapshot of the observations used for fitting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldTrainingData {
    pub fold: FoldSpec,
    pub observations: Vec<SurvivalObservation>,
    pub source_fingerprint: String,
}

impl FoldTrainingData {
    pub fn validate(&self) -> SurvivalResult<()> {
        self.fold.validate()?;
        let cutoff_exclusive = self
            .fold
            .training_end
            .checked_add_signed(Duration::days(1))
            .ok_or_else(|| SurvivalError::InvalidInput("training cutoff overflowed".to_string()))?;
        for observation in &self.observations {
            observation.validate()?;
            if !observation.never_used && observation.exit_date > cutoff_exclusive {
                return Err(SurvivalError::Leakage(format!(
                    "fold {} retains an observation after its training cutoff",
                    self.fold.id
                )));
            }
            if observation.entry_date >= self.fold.validation_start {
                return Err(SurvivalError::Leakage(format!(
                    "fold {} retains an observation entering during validation",
                    self.fold.id
                )));
            }
            if let Some(start) = self.fold.training_start {
                if !observation.never_used && observation.entry_date < start {
                    return Err(SurvivalError::Leakage(format!(
                        "fold {} retains pre-window risk exposure",
                        self.fold.id
                    )));
                }
                if observation.left_truncated && observation.entry_date != start {
                    return Err(SurvivalError::Leakage(format!(
                        "fold {} has inconsistent left-truncation metadata",
                        self.fold.id
                    )));
                }
            }
        }
        validate_fingerprint(&self.source_fingerprint, "fold source")
    }
}

/// Build deterministic training observations for one chronological fold.
///
/// Events after the training cutoff are clipped to the cutoff and converted to
/// right-censored rows.  This preserves safe pre-cutoff exposure without
/// allowing the future event label to leak into fitting.  Never-used rows that
/// enter after the cutoff are dropped rather than reclassified as reuse risk.
pub fn build_fold_training_inputs(
    observations: &[SurvivalObservation],
    fold: FoldSpec,
) -> SurvivalResult<FoldTrainingData> {
    fold.validate()?;
    for observation in observations {
        observation.validate()?;
    }

    let cutoff_exclusive = fold
        .training_end
        .checked_add_signed(Duration::days(1))
        .ok_or_else(|| SurvivalError::InvalidInput("training cutoff overflowed".to_string()))?;
    let mut selected = Vec::new();
    for observation in observations {
        if observation.entry_date >= fold.validation_start {
            continue;
        }
        if observation.never_used {
            if observation.entry_date <= fold.training_end {
                selected.push(observation.clone());
            }
            continue;
        }
        if observation.entry_date >= cutoff_exclusive {
            continue;
        }

        let mut clipped = observation.clone();
        if clipped.exit_date > cutoff_exclusive {
            clipped.exit_date = cutoff_exclusive;
            clipped.reused = false;
        }
        if let Some(start) = fold.training_start {
            if clipped.exit_date <= start {
                continue;
            }
            if clipped.entry_date < start {
                let removed_days = (start - clipped.entry_date).num_days() as usize;
                clipped.entry_date = start;
                clipped.left_truncated = true;
                clipped.elapsed_offset_days = clipped
                    .elapsed_offset_days
                    .checked_add(removed_days)
                    .ok_or_else(|| {
                        SurvivalError::InvalidInput(
                            "left-truncation elapsed offset overflowed".to_string(),
                        )
                    })?;
            }
        }
        if clipped.exit_date <= clipped.entry_date {
            continue;
        }
        selected.push(clipped);
    }

    selected.sort_by(|left, right| {
        left.word
            .cmp(&right.word)
            .then_with(|| left.policy_era.cmp(&right.policy_era))
            .then_with(|| left.entry_date.cmp(&right.entry_date))
            .then_with(|| left.exit_date.cmp(&right.exit_date))
            .then_with(|| left.status().sort_key().cmp(&right.status().sort_key()))
            .then_with(|| left.mass.total_cmp(&right.mass))
    });
    let source_fingerprint = fingerprint_observations(&fold, &selected);
    let data = FoldTrainingData {
        fold,
        observations: selected,
        source_fingerprint,
    };
    data.validate()?;
    Ok(data)
}

pub fn prepare_fold_training_inputs(
    observations: &[SurvivalObservation],
    fold: FoldSpec,
) -> SurvivalResult<FoldTrainingData> {
    build_fold_training_inputs(observations, fold)
}

fn fingerprint_observations(fold: &FoldSpec, observations: &[SurvivalObservation]) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, &fold.id);
    hash_field(&mut hasher, date_text(fold.training_end));
    hash_field(&mut hasher, date_text(fold.validation_start));
    hash_field(&mut hasher, date_text(fold.validation_end));
    for observation in observations {
        hash_field(&mut hasher, &observation.word);
        hash_field(&mut hasher, date_text(observation.entry_date));
        hash_field(&mut hasher, date_text(observation.exit_date));
        hash_field(&mut hasher, &observation.policy_era);
        hash_field(&mut hasher, observation.status().sort_key().to_string());
        hash_field(&mut hasher, observation.left_truncated.to_string());
        hash_field(&mut hasher, observation.elapsed_offset_days.to_le_bytes());
        hash_field(&mut hasher, observation.mass.to_bits().to_le_bytes());
    }
    format!(
        "{FINGERPRINT_PREFIX}{}",
        hex_digest(hasher.finalize().as_slice())
    )
}

fn hash_field(hasher: &mut Sha256, value: impl AsRef<[u8]>) {
    let value = value.as_ref();
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn date_text(date: NaiveDate) -> String {
    date.format("%Y-%m-%d").to_string()
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

fn validate_fingerprint(value: &str, label: &str) -> SurvivalResult<()> {
    if value.trim().is_empty() {
        return Err(SurvivalError::InvalidArtifact(format!(
            "{label} fingerprint must not be empty"
        )));
    }
    if !value.starts_with(FINGERPRINT_PREFIX) {
        return Err(SurvivalError::InvalidArtifact(format!(
            "{label} fingerprint must use {FINGERPRINT_PREFIX}"
        )));
    }
    let digest = &value[FINGERPRINT_PREFIX.len()..];
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SurvivalError::InvalidArtifact(format!(
            "{label} fingerprint must contain a 256-bit hexadecimal digest"
        )));
    }
    Ok(())
}

/// Configuration for the smoothed discrete-time hazard fit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SurvivalConfig {
    pub basis_degree: usize,
    pub time_scale_days: f64,
    pub ridge_lambda: f64,
    pub smoothness_lambda: f64,
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub min_probability: f64,
    pub max_interval_days: usize,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            basis_degree: 2,
            time_scale_days: 365.0,
            ridge_lambda: 0.25,
            smoothness_lambda: 0.05,
            max_iterations: 80,
            convergence_tolerance: 1e-8,
            min_probability: 1e-9,
            max_interval_days: 10_000,
        }
    }
}

impl SurvivalConfig {
    pub fn validate(&self) -> SurvivalResult<()> {
        if self.basis_degree > 8 {
            return Err(SurvivalError::InvalidInput(
                "time-basis degree must be at most 8".to_string(),
            ));
        }
        if !self.time_scale_days.is_finite() || self.time_scale_days <= 0.0 {
            return Err(SurvivalError::InvalidInput(
                "time_scale_days must be finite and positive".to_string(),
            ));
        }
        if !self.ridge_lambda.is_finite() || self.ridge_lambda < 0.0 {
            return Err(SurvivalError::InvalidInput(
                "ridge_lambda must be finite and non-negative".to_string(),
            ));
        }
        if !self.smoothness_lambda.is_finite() || self.smoothness_lambda < 0.0 {
            return Err(SurvivalError::InvalidInput(
                "smoothness_lambda must be finite and non-negative".to_string(),
            ));
        }
        if self.max_iterations == 0
            || !self.convergence_tolerance.is_finite()
            || self.convergence_tolerance <= 0.0
        {
            return Err(SurvivalError::InvalidInput(
                "fit iteration and tolerance settings must be positive".to_string(),
            ));
        }
        if !self.min_probability.is_finite()
            || self.min_probability <= 0.0
            || self.min_probability >= 0.5
        {
            return Err(SurvivalError::InvalidInput(
                "min_probability must lie strictly between zero and one-half".to_string(),
            ));
        }
        if self.max_interval_days == 0 {
            return Err(SurvivalError::InvalidInput(
                "max_interval_days must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct MassSummary {
    pub never_used: f64,
    pub reused: f64,
    pub right_censored: f64,
}

impl MassSummary {
    pub fn total(self) -> f64 {
        self.never_used + self.reused + self.right_censored
    }

    pub fn never_used_fraction(self) -> f64 {
        fraction(self.never_used, self.total())
    }

    /// Historically used mass includes both observed reuse and used-but-
    /// censored intervals; neither is confused with never-used support mass.
    pub fn historically_used_fraction(self) -> f64 {
        fraction(self.reused + self.right_censored, self.total())
    }
}

fn fraction(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        (numerator / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalPrediction {
    pub hazard: f64,
    pub survival: f64,
    pub reuse_probability: f64,
    pub reusable_weight: f64,
    pub never_used_weight: f64,
    pub historically_used_mass_fraction: f64,
}

/// Fitted discrete-time model.  It is an experimental artifact and does not
/// alter the existing logistic prior unless a caller explicitly integrates it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalModel {
    pub schema_version: u32,
    pub coefficients: Vec<f64>,
    pub config: SurvivalConfig,
    pub policy_eras: Vec<PolicyEra>,
    pub era_ids: Vec<String>,
    pub mass: MassSummary,
    pub training_observations: usize,
    pub training_rows: usize,
    pub converged: bool,
    pub training_fingerprint: String,
}

impl SurvivalModel {
    pub fn fit(
        observations: &[SurvivalObservation],
        config: &SurvivalConfig,
    ) -> SurvivalResult<Self> {
        config.validate()?;
        for observation in observations {
            observation.validate()?;
        }
        if observations.is_empty() {
            return Err(SurvivalError::InvalidInput(
                "cannot fit survival model with no observations".to_string(),
            ));
        }
        let eras = derive_policy_eras(observations)?;
        Self::fit_with_policy_eras(observations, &eras, config)
    }

    pub fn fit_with_policy_eras(
        observations: &[SurvivalObservation],
        policy_eras: &[PolicyEra],
        config: &SurvivalConfig,
    ) -> SurvivalResult<Self> {
        config.validate()?;
        validate_policy_eras(policy_eras)?;
        if observations.is_empty() {
            return Err(SurvivalError::InvalidInput(
                "cannot fit survival model with no observations".to_string(),
            ));
        }
        for observation in observations {
            observation.validate()?;
            let era = policy_eras
                .iter()
                .find(|era| era.id == observation.policy_era)
                .ok_or_else(|| {
                    SurvivalError::InvalidInput(format!(
                        "observation {} references unknown policy era {}",
                        observation.word, observation.policy_era
                    ))
                })?;
            if !era.contains(observation.entry_date)
                || (!observation.never_used
                    && !era.contains(
                        observation
                            .exit_date
                            .checked_sub_signed(Duration::days(1))
                            .ok_or_else(|| {
                                SurvivalError::InvalidInput(format!(
                                    "observation {} interval underflowed",
                                    observation.word
                                ))
                            })?,
                    ))
            {
                return Err(SurvivalError::InvalidInput(format!(
                    "observation {} spans dates outside policy era {}",
                    observation.word, observation.policy_era
                )));
            }
        }

        let mut ordered_eras = policy_eras.to_vec();
        ordered_eras.sort_by_key(|era| (era.start, era.id.clone()));
        let era_ids = ordered_eras
            .iter()
            .map(|era| era.id.clone())
            .collect::<Vec<_>>();
        let dimension = config.basis_degree + era_ids.len();
        let mut aggregates = BTreeMap::<(usize, usize), (f64, f64)>::new();
        let mut mass = MassSummary::default();

        for observation in observations {
            match observation.status() {
                ObservationStatus::NeverUsed => {
                    mass.never_used += observation.mass;
                    continue;
                }
                ObservationStatus::Reused => mass.reused += observation.mass,
                ObservationStatus::RightCensored => mass.right_censored += observation.mass,
            }
            let duration = observation.duration_days();
            if duration <= 0
                || observation
                    .elapsed_offset_days
                    .saturating_add(duration as usize)
                    > config.max_interval_days
            {
                return Err(SurvivalError::InvalidInput(format!(
                    "observation {} has duration {} outside configured range",
                    observation.word, duration
                )));
            }
            for elapsed in 0..duration as usize {
                let date = observation
                    .entry_date
                    .checked_add_signed(Duration::days(elapsed as i64))
                    .ok_or_else(|| {
                        SurvivalError::InvalidInput(format!(
                            "observation {} overflows date range",
                            observation.word
                        ))
                    })?;
                let era_id = policy_era_for_date(&ordered_eras, date).ok_or_else(|| {
                    SurvivalError::InvalidInput(format!(
                        "observation {} crosses an uncovered policy date {}",
                        observation.word, date
                    ))
                })?;
                if era_id != observation.policy_era {
                    return Err(SurvivalError::InvalidInput(format!(
                        "observation {} crosses policy eras; split it before fitting",
                        observation.word
                    )));
                }
                let elapsed_since_last_use = observation
                    .elapsed_offset_days
                    .checked_add(elapsed)
                    .ok_or_else(|| {
                    SurvivalError::InvalidInput(format!(
                        "observation {} elapsed duration overflowed",
                        observation.word
                    ))
                })?;
                let era_index = era_ids
                    .iter()
                    .position(|id| id == era_id)
                    .expect("validated era id");
                let target = observation.reused && elapsed + 1 == duration as usize;
                let aggregate = aggregates
                    .entry((era_index, elapsed_since_last_use))
                    .or_default();
                aggregate.1 += observation.mass;
                if target {
                    aggregate.0 += observation.mass;
                }
            }
        }

        let rows = aggregates
            .into_iter()
            .map(
                |((era_index, elapsed), (event_mass, total_mass))| TrainingRow {
                    feature: basis_feature(
                        config.basis_degree,
                        config.time_scale_days,
                        elapsed as f64,
                        &era_ids[era_index],
                        &era_ids,
                    ),
                    target: event_mass / total_mass,
                    weight: total_mass,
                },
            )
            .collect::<Vec<_>>();

        let (coefficients, converged) = fit_coefficients(&rows, dimension, config)?;
        let training_fingerprint = fingerprint_observations(
            &FoldSpec::new(
                "in-memory",
                None,
                observations
                    .iter()
                    .map(|observation| observation.exit_date)
                    .max()
                    .unwrap_or(DEFAULT_DATE),
                observations
                    .iter()
                    .map(|observation| observation.exit_date)
                    .max()
                    .unwrap_or(DEFAULT_DATE),
                observations
                    .iter()
                    .map(|observation| observation.exit_date)
                    .max()
                    .unwrap_or(DEFAULT_DATE),
            ),
            observations,
        );
        let model = Self {
            schema_version: SURVIVAL_SCHEMA_VERSION,
            coefficients,
            config: config.clone(),
            policy_eras: ordered_eras,
            era_ids,
            mass,
            training_observations: observations.len(),
            training_rows: rows.len(),
            converged,
            training_fingerprint,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn fit_fold(
        training: &FoldTrainingData,
        policy_eras: &[PolicyEra],
        config: &SurvivalConfig,
    ) -> SurvivalResult<Self> {
        training.validate()?;
        let mut model = Self::fit_with_policy_eras(&training.observations, policy_eras, config)?;
        model.training_fingerprint = training.source_fingerprint.clone();
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> SurvivalResult<()> {
        if self.schema_version != SURVIVAL_SCHEMA_VERSION {
            return Err(SurvivalError::InvalidArtifact(format!(
                "unsupported model schema version {}",
                self.schema_version
            )));
        }
        self.config.validate()?;
        validate_policy_eras(&self.policy_eras)?;
        if self.era_ids.is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "model must contain at least one era id".to_string(),
            ));
        }
        if self.era_ids.iter().any(|id| id.trim().is_empty())
            || self.era_ids.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(SurvivalError::InvalidArtifact(
                "model era ids must be non-empty and unique".to_string(),
            ));
        }
        let expected_dimension = self.config.basis_degree + self.era_ids.len();
        if self.coefficients.len() != expected_dimension
            || self.coefficients.iter().any(|value| !value.is_finite())
        {
            return Err(SurvivalError::InvalidArtifact(
                "model coefficient vector has the wrong shape or non-finite values".to_string(),
            ));
        }
        if !self.mass.never_used.is_finite()
            || !self.mass.reused.is_finite()
            || !self.mass.right_censored.is_finite()
            || self.mass.never_used < 0.0
            || self.mass.reused < 0.0
            || self.mass.right_censored < 0.0
            || self.mass.total() <= 0.0
        {
            return Err(SurvivalError::InvalidArtifact(
                "model mass summary must be finite and positive".to_string(),
            ));
        }
        if self.training_observations == 0 && self.training_rows != 0 {
            return Err(SurvivalError::InvalidArtifact(
                "training rows cannot exist without observations".to_string(),
            ));
        }
        validate_fingerprint(&self.training_fingerprint, "training")
    }

    pub fn mass_summary(&self) -> MassSummary {
        self.mass
    }

    pub fn hazard_score(&self, elapsed_days: f64, policy_era: &str) -> f64 {
        self.predict(elapsed_days, policy_era).hazard
    }

    pub fn reusable_weight(&self, elapsed_days: f64, policy_era: &str) -> f64 {
        self.predict(elapsed_days, policy_era).reusable_weight
    }

    pub fn try_predict(
        &self,
        elapsed_days: f64,
        policy_era: &str,
    ) -> SurvivalResult<SurvivalPrediction> {
        self.validate()?;
        if !elapsed_days.is_finite() || elapsed_days < 0.0 {
            return Err(SurvivalError::InvalidInput(
                "elapsed_days must be finite and non-negative".to_string(),
            ));
        }
        if !self.era_ids.iter().any(|id| id == policy_era) {
            return Err(SurvivalError::InvalidInput(format!(
                "unknown policy era {policy_era}"
            )));
        }
        let elapsed = elapsed_days.min(self.config.max_interval_days as f64);
        let hazard = sigmoid_clamped(
            dot(
                &self.coefficients,
                &basis_feature(
                    self.config.basis_degree,
                    self.config.time_scale_days,
                    elapsed,
                    policy_era,
                    &self.era_ids,
                ),
            ),
            self.config.min_probability,
        );
        let days = elapsed.ceil() as usize;
        let mut log_survival = 0.0;
        for day in 0..days {
            let day_hazard = sigmoid_clamped(
                dot(
                    &self.coefficients,
                    &basis_feature(
                        self.config.basis_degree,
                        self.config.time_scale_days,
                        day as f64,
                        policy_era,
                        &self.era_ids,
                    ),
                ),
                self.config.min_probability,
            );
            log_survival += (1.0 - day_hazard).ln();
            if log_survival < -745.0 {
                log_survival = -745.0;
                break;
            }
        }
        let survival = log_survival.exp().clamp(0.0, 1.0);
        let reuse_probability =
            (1.0 - survival).clamp(0.0, 1.0) * self.mass.historically_used_fraction();
        let historically_used_mass_fraction = self.mass.historically_used_fraction();
        let reusable_weight = (hazard * historically_used_mass_fraction).clamp(0.0, 1.0);
        let never_used_weight = self.mass.never_used_fraction().clamp(0.0, 1.0);
        Ok(SurvivalPrediction {
            hazard,
            survival,
            reuse_probability,
            reusable_weight,
            never_used_weight,
            historically_used_mass_fraction,
        })
    }

    /// Predict reuse across a dated interval, applying the policy era in force
    /// on each risk day instead of pretending the current era covered the
    /// entire time since the previous use.
    pub fn try_predict_interval(
        &self,
        last_use: NaiveDate,
        as_of: NaiveDate,
    ) -> SurvivalResult<SurvivalPrediction> {
        self.validate()?;
        if as_of < last_use {
            return Err(SurvivalError::InvalidInput(
                "survival prediction date precedes last use".to_string(),
            ));
        }
        let elapsed = (as_of - last_use).num_days() as usize;
        if elapsed > self.config.max_interval_days {
            return Err(SurvivalError::InvalidInput(
                "survival prediction interval exceeds configured range".to_string(),
            ));
        }
        let current_era = policy_era_for_date(&self.policy_eras, as_of).ok_or_else(|| {
            SurvivalError::InvalidInput(format!("no policy era covers prediction date {as_of}"))
        })?;
        let current_elapsed = elapsed.saturating_sub(1) as f64;
        let hazard = sigmoid_clamped(
            dot(
                &self.coefficients,
                &basis_feature(
                    self.config.basis_degree,
                    self.config.time_scale_days,
                    current_elapsed,
                    current_era,
                    &self.era_ids,
                ),
            ),
            self.config.min_probability,
        );
        let mut log_survival = 0.0;
        for offset in 0..elapsed {
            let risk_date = last_use
                .checked_add_signed(Duration::days(offset as i64 + 1))
                .ok_or_else(|| {
                    SurvivalError::InvalidInput("prediction risk date overflowed".to_string())
                })?;
            let era = policy_era_for_date(&self.policy_eras, risk_date).ok_or_else(|| {
                SurvivalError::InvalidInput(format!(
                    "no policy era covers prediction risk date {risk_date}"
                ))
            })?;
            let day_hazard = sigmoid_clamped(
                dot(
                    &self.coefficients,
                    &basis_feature(
                        self.config.basis_degree,
                        self.config.time_scale_days,
                        offset as f64,
                        era,
                        &self.era_ids,
                    ),
                ),
                self.config.min_probability,
            );
            log_survival += (1.0 - day_hazard).ln();
            if log_survival < -745.0 {
                log_survival = -745.0;
                break;
            }
        }
        let survival = log_survival.exp().clamp(0.0, 1.0);
        let historically_used_mass_fraction = self.mass.historically_used_fraction();
        Ok(SurvivalPrediction {
            hazard,
            survival,
            reuse_probability: (1.0 - survival).clamp(0.0, 1.0) * historically_used_mass_fraction,
            reusable_weight: (hazard * historically_used_mass_fraction).clamp(0.0, 1.0),
            never_used_weight: self.mass.never_used_fraction().clamp(0.0, 1.0),
            historically_used_mass_fraction,
        })
    }

    /// Panic-free convenience inference.  Invalid elapsed values are treated
    /// as time zero; callers that need diagnostics should use `try_predict`.
    pub fn predict(&self, elapsed_days: f64, policy_era: &str) -> SurvivalPrediction {
        let elapsed = if elapsed_days.is_finite() && elapsed_days >= 0.0 {
            elapsed_days
        } else {
            0.0
        };
        self.try_predict(elapsed, policy_era)
            .unwrap_or_else(|_| SurvivalPrediction {
                hazard: 0.5,
                survival: 1.0,
                reuse_probability: 0.0,
                reusable_weight: 0.0,
                never_used_weight: self.mass.never_used_fraction(),
                historically_used_mass_fraction: self.mass.historically_used_fraction(),
            })
    }

    pub fn score_for_status(
        &self,
        status: ObservationStatus,
        elapsed_days: f64,
        policy_era: &str,
    ) -> f64 {
        let prediction = self.predict(elapsed_days, policy_era);
        match status {
            ObservationStatus::NeverUsed => prediction.never_used_weight,
            ObservationStatus::Reused | ObservationStatus::RightCensored => {
                prediction.reusable_weight
            }
        }
    }
}

fn derive_policy_eras(observations: &[SurvivalObservation]) -> SurvivalResult<Vec<PolicyEra>> {
    let mut starts = BTreeMap::<String, NaiveDate>::new();
    for observation in observations {
        starts
            .entry(observation.policy_era.clone())
            .and_modify(|date| *date = (*date).min(observation.entry_date))
            .or_insert(observation.entry_date);
    }
    let mut ordered = starts
        .into_iter()
        .map(|(id, start)| (start, id))
        .collect::<Vec<_>>();
    ordered.sort();
    let mut eras = Vec::with_capacity(ordered.len());
    for index in 0..ordered.len() {
        let (start, id) = &ordered[index];
        let end = ordered.get(index + 1).map(|(next, _)| *next);
        eras.push(PolicyEra::new(id.clone(), *start, end));
    }
    validate_policy_eras(&eras)?;
    Ok(eras)
}

#[derive(Clone, Debug)]
struct TrainingRow {
    feature: Vec<f64>,
    target: f64,
    weight: f64,
}

fn basis_feature(
    basis_degree: usize,
    time_scale_days: f64,
    elapsed_days: f64,
    policy_era: &str,
    era_ids: &[String],
) -> Vec<f64> {
    let scaled = (elapsed_days.max(0.0) / time_scale_days).ln_1p();
    let mut feature = Vec::with_capacity(basis_degree + era_ids.len());
    feature.push(1.0);
    for degree in 1..=basis_degree {
        feature.push(scaled.powi(degree as i32));
    }
    // The first era is the baseline; the remaining era indicators avoid an
    // intercept/one-hot singularity while still allowing policy-era shifts.
    for era_id in era_ids.iter().skip(1) {
        feature.push(if policy_era == era_id { 1.0 } else { 0.0 });
    }
    feature
}

fn fit_coefficients(
    rows: &[TrainingRow],
    dimension: usize,
    config: &SurvivalConfig,
) -> SurvivalResult<(Vec<f64>, bool)> {
    let mut coefficients = vec![0.0; dimension];
    if rows.is_empty() {
        return Ok((coefficients, true));
    }
    let mut converged = false;
    for _ in 0..config.max_iterations {
        let mut hessian = vec![vec![0.0; dimension]; dimension];
        let mut gradient = vec![0.0; dimension];
        for row in rows {
            let eta = dot(&coefficients, &row.feature).clamp(-40.0, 40.0);
            let probability = sigmoid(eta);
            let variance = (probability * (1.0 - probability)).max(1e-12);
            let residual = row.weight * (row.target - probability);
            for (i, (gradient_value, hessian_row)) in
                gradient.iter_mut().zip(&mut hessian).enumerate()
            {
                *gradient_value += residual * row.feature[i];
                for (j, hessian_value) in hessian_row.iter_mut().enumerate().take(i + 1) {
                    *hessian_value += row.weight * variance * row.feature[i] * row.feature[j];
                }
            }
        }
        // Symmetrize the accumulated lower triangle and apply ridge plus a
        // second-difference penalty to the time basis for smoothness.
        for (i, (gradient_value, coefficient)) in gradient.iter_mut().zip(&coefficients).enumerate()
        {
            let (earlier_rows, current_and_later) = hessian.split_at_mut(i);
            let current_row = &mut current_and_later[0];
            for (j, earlier_row) in earlier_rows.iter_mut().enumerate() {
                earlier_row[i] = current_row[j];
            }
            current_row[i] += config.ridge_lambda;
            *gradient_value -= config.ridge_lambda * coefficient;
        }
        if config.smoothness_lambda > 0.0 && config.basis_degree >= 2 {
            for k in 2..=config.basis_degree {
                let contrast = [(k - 2, 1.0), (k - 1, -2.0), (k, 1.0)];
                let second_difference = contrast
                    .iter()
                    .map(|(index, weight)| coefficients[*index] * weight)
                    .sum::<f64>();
                for (i, weight_i) in contrast {
                    gradient[i] -= config.smoothness_lambda * second_difference * weight_i;
                    for (j, weight_j) in contrast {
                        hessian[i][j] += config.smoothness_lambda * weight_i * weight_j;
                    }
                }
            }
        }

        let delta = solve_linear_system(&hessian, &gradient).ok_or_else(|| {
            SurvivalError::Numeric("regularized hazard Hessian is singular".to_string())
        })?;
        let mut max_delta: f64 = 0.0;
        for (coefficient, step) in coefficients.iter_mut().zip(delta) {
            let bounded_step = step.clamp(-5.0, 5.0);
            *coefficient = (*coefficient + bounded_step).clamp(-40.0, 40.0);
            max_delta = max_delta.max(bounded_step.abs());
        }
        if !coefficients.iter().all(|value| value.is_finite()) {
            return Err(SurvivalError::Numeric(
                "non-finite coefficient encountered".to_string(),
            ));
        }
        if max_delta <= config.convergence_tolerance {
            converged = true;
            break;
        }
    }
    Ok((coefficients, converged))
}

fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    for pivot in 0..n {
        let (pivot_row, pivot_value) = (pivot..n)
            .map(|row| (row, a[row][pivot].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))?;
        if pivot_value < 1e-12 || !pivot_value.is_finite() {
            return None;
        }
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            b.swap(pivot, pivot_row);
        }
        let pivot_values = a[pivot].clone();
        let pivot_diagonal = pivot_values[pivot];
        let pivot_rhs = b[pivot];
        for (row_values, rhs_value) in a.iter_mut().zip(&mut b).skip(pivot + 1) {
            let factor = row_values[pivot] / pivot_diagonal;
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in row_values[pivot..].iter_mut().zip(&pivot_values[pivot..]) {
                *value -= factor * pivot_value;
            }
            *rhs_value -= factor * pivot_rhs;
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let remainder = (row + 1..n)
            .map(|column| a[row][column] * solution[column])
            .sum::<f64>();
        let pivot = a[row][row];
        if pivot.abs() < 1e-12 || !pivot.is_finite() {
            return None;
        }
        solution[row] = (b[row] - remainder) / pivot;
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn sigmoid_clamped(value: f64, floor: f64) -> f64 {
    let value = if value.is_finite() { value } else { 0.0 };
    sigmoid(value.clamp(-40.0, 40.0)).clamp(floor, 1.0 - floor)
}

/// Provenance required for an artifact to be auditable and fold-local.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SurvivalProvenance {
    pub input_fingerprint: String,
    pub code_revision: String,
    pub fold_id: String,
    pub training_cutoff: NaiveDate,
}

impl Default for SurvivalProvenance {
    fn default() -> Self {
        Self {
            input_fingerprint: String::new(),
            code_revision: String::new(),
            fold_id: String::new(),
            training_cutoff: DEFAULT_DATE,
        }
    }
}

impl SurvivalProvenance {
    pub fn validate(&self) -> SurvivalResult<()> {
        validate_fingerprint(&self.input_fingerprint, "artifact input")?;
        if self.code_revision.trim().is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "artifact code_revision must not be empty".to_string(),
            ));
        }
        if self.fold_id.trim().is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "artifact fold_id must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Evidence metadata is mandatory in the serialized schema.  The default is
/// intentionally not promotable; callers must provide held-out evidence.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvidenceGateMetadata {
    pub gate_id: String,
    pub validation_folds: usize,
    pub fold_ids: Vec<String>,
    pub source_identity: String,
    pub coverage_ok: bool,
    pub failure_count: usize,
    pub held_out: bool,
    pub paired_solve_quality_ok: bool,
    pub latency_ok: bool,
    pub memory_ok: bool,
    pub sealed_window_untouched: bool,
    pub approved_by: String,
}

impl EvidenceGateMetadata {
    pub fn validate_for_promotion(&self) -> SurvivalResult<()> {
        if self.gate_id.trim().is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "promotion evidence gate id is required".to_string(),
            ));
        }
        if self.validation_folds == 0 {
            return Err(SurvivalError::InvalidArtifact(
                "promotion evidence must include validation folds".to_string(),
            ));
        }
        let unique_folds = self.fold_ids.iter().collect::<BTreeSet<_>>();
        if self.fold_ids.len() != self.validation_folds
            || unique_folds.len() != self.fold_ids.len()
            || self.fold_ids.iter().any(|id| id.trim().is_empty())
        {
            return Err(SurvivalError::InvalidArtifact(
                "promotion evidence fold identities are missing or inconsistent".to_string(),
            ));
        }
        if self.source_identity.trim().is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "promotion evidence source identity is required".to_string(),
            ));
        }
        if !self.coverage_ok
            || self.failure_count != 0
            || !self.held_out
            || !self.paired_solve_quality_ok
            || !self.latency_ok
            || !self.memory_ok
            || !self.sealed_window_untouched
        {
            return Err(SurvivalError::InvalidArtifact(
                "promotion evidence must pass held-out solve-quality, coverage, failure, latency, memory, and sealed-window gates"
                    .to_string(),
            ));
        }
        if self.approved_by.trim().is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "promotion evidence approver is required".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PromotionMetadata {
    pub enabled: bool,
    pub evidence_gate: EvidenceGateMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SurvivalMetadata {
    pub schema_version: u32,
    pub model_version: String,
    pub policy_eras: Vec<PolicyEra>,
    pub left_truncation: LeftTruncationMetadata,
    pub provenance: SurvivalProvenance,
}

impl SurvivalMetadata {
    pub fn validate(&self) -> SurvivalResult<()> {
        if self.schema_version != SURVIVAL_SCHEMA_VERSION {
            return Err(SurvivalError::InvalidArtifact(format!(
                "unsupported artifact schema version {}",
                self.schema_version
            )));
        }
        if self.model_version.trim().is_empty() {
            return Err(SurvivalError::InvalidArtifact(
                "artifact model_version must not be empty".to_string(),
            ));
        }
        validate_policy_eras(&self.policy_eras)?;
        self.left_truncation.validate()?;
        self.provenance.validate()
    }
}

/// Persistable model + metadata.  `promotion.enabled` remains false unless a
/// caller explicitly opts in and supplies an evidence gate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalArtifact {
    pub schema_version: u32,
    pub model: SurvivalModel,
    pub metadata: SurvivalMetadata,
    pub promotion: PromotionMetadata,
}

impl SurvivalArtifact {
    pub fn new(
        model: SurvivalModel,
        model_version: impl Into<String>,
        left_truncation: LeftTruncationMetadata,
        provenance: SurvivalProvenance,
    ) -> Self {
        let policy_eras = model.policy_eras.clone();
        Self {
            schema_version: SURVIVAL_SCHEMA_VERSION,
            model,
            metadata: SurvivalMetadata {
                schema_version: SURVIVAL_SCHEMA_VERSION,
                model_version: model_version.into(),
                policy_eras,
                left_truncation,
                provenance,
            },
            promotion: PromotionMetadata::default(),
        }
    }

    pub fn validate(&self) -> SurvivalResult<()> {
        if self.schema_version != SURVIVAL_SCHEMA_VERSION {
            return Err(SurvivalError::InvalidArtifact(format!(
                "unsupported artifact schema version {}",
                self.schema_version
            )));
        }
        self.model.validate()?;
        self.metadata.validate()?;
        if self.metadata.policy_eras != self.model.policy_eras {
            return Err(SurvivalError::InvalidArtifact(
                "artifact metadata policy eras do not match model eras".to_string(),
            ));
        }
        if self.metadata.provenance.input_fingerprint != self.model.training_fingerprint {
            return Err(SurvivalError::InvalidArtifact(
                "artifact provenance fingerprint does not match fitted training inputs".to_string(),
            ));
        }
        if self.promotion.enabled {
            self.promotion.evidence_gate.validate_for_promotion()?;
        }
        Ok(())
    }

    pub fn promotion_allowed(&self) -> bool {
        self.promotion.enabled
            && self
                .promotion
                .evidence_gate
                .validate_for_promotion()
                .is_ok()
    }

    pub fn to_json(&self) -> SurvivalResult<String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| SurvivalError::Serialization(error.to_string()))
    }

    pub fn from_json(json: &str) -> SurvivalResult<Self> {
        let artifact = serde_json::from_str::<Self>(json)
            .map_err(|error| SurvivalError::Serialization(error.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }
}

/// A pair of probabilities for one held-out outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryComparisonRow {
    pub observed: bool,
    pub survival_probability: f64,
    pub logistic_probability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub index: usize,
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub mean_prediction: f64,
    pub observed_rate: f64,
    pub absolute_gap: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationSummary {
    pub bins: Vec<CalibrationBin>,
    pub expected_calibration_error: f64,
    pub maximum_calibration_gap: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryMetrics {
    pub count: usize,
    pub log_loss: f64,
    pub brier: f64,
    pub calibration: CalibrationSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelComparison {
    pub survival: BinaryMetrics,
    pub logistic_baseline: BinaryMetrics,
    pub delta_log_loss: f64,
    pub delta_brier: f64,
}

pub fn compare_against_logistic_baseline(
    rows: &[BinaryComparisonRow],
) -> SurvivalResult<ModelComparison> {
    compare_with_calibration_bins(rows, 10)
}

pub fn compare_with_calibration_bins(
    rows: &[BinaryComparisonRow],
    calibration_bins: usize,
) -> SurvivalResult<ModelComparison> {
    if rows.is_empty() {
        return Err(SurvivalError::InvalidInput(
            "cannot compare models with no outcomes".to_string(),
        ));
    }
    if calibration_bins == 0 {
        return Err(SurvivalError::InvalidInput(
            "calibration bin count must be positive".to_string(),
        ));
    }
    let survival = metrics_for(
        rows.iter()
            .map(|row| (row.survival_probability, row.observed))
            .collect::<Vec<_>>()
            .as_slice(),
        calibration_bins,
    )?;
    let logistic_baseline = metrics_for(
        rows.iter()
            .map(|row| (row.logistic_probability, row.observed))
            .collect::<Vec<_>>()
            .as_slice(),
        calibration_bins,
    )?;
    Ok(ModelComparison {
        delta_log_loss: survival.log_loss - logistic_baseline.log_loss,
        delta_brier: survival.brier - logistic_baseline.brier,
        survival,
        logistic_baseline,
    })
}

pub fn metrics_for_probabilities(
    probabilities: &[(f64, bool)],
    calibration_bins: usize,
) -> SurvivalResult<BinaryMetrics> {
    metrics_for(probabilities, calibration_bins)
}

fn metrics_for(
    probabilities: &[(f64, bool)],
    calibration_bins: usize,
) -> SurvivalResult<BinaryMetrics> {
    if probabilities.is_empty() {
        return Err(SurvivalError::InvalidInput(
            "cannot score an empty probability set".to_string(),
        ));
    }
    if calibration_bins == 0 {
        return Err(SurvivalError::InvalidInput(
            "calibration bin count must be positive".to_string(),
        ));
    }
    let mut log_loss = 0.0;
    let mut brier = 0.0;
    for (probability, observed) in probabilities {
        validate_probability(*probability)?;
        let target = if *observed { 1.0 } else { 0.0 };
        let bounded = probability.clamp(1e-12, 1.0 - 1e-12);
        log_loss += if *observed {
            -bounded.ln()
        } else {
            -(1.0 - bounded).ln()
        };
        brier += (bounded - target).powi(2);
    }
    let count = probabilities.len();
    let bins = calibration_primitives(probabilities, calibration_bins)?;
    let expected_calibration_error = bins
        .iter()
        .map(|bin| bin.count as f64 / count as f64 * bin.absolute_gap)
        .sum();
    let maximum_calibration_gap = bins.iter().map(|bin| bin.absolute_gap).fold(0.0, f64::max);
    Ok(BinaryMetrics {
        count,
        log_loss: log_loss / count as f64,
        brier: brier / count as f64,
        calibration: CalibrationSummary {
            bins,
            expected_calibration_error,
            maximum_calibration_gap,
        },
    })
}

pub fn calibration_primitives(
    probabilities: &[(f64, bool)],
    calibration_bins: usize,
) -> SurvivalResult<Vec<CalibrationBin>> {
    if calibration_bins == 0 {
        return Err(SurvivalError::InvalidInput(
            "calibration bin count must be positive".to_string(),
        ));
    }
    let mut counts = vec![0usize; calibration_bins];
    let mut prediction_sums = vec![0.0; calibration_bins];
    let mut outcome_sums = vec![0.0; calibration_bins];
    for (probability, observed) in probabilities {
        validate_probability(*probability)?;
        let index =
            ((*probability * calibration_bins as f64).floor() as usize).min(calibration_bins - 1);
        counts[index] += 1;
        prediction_sums[index] += *probability;
        outcome_sums[index] += if *observed { 1.0 } else { 0.0 };
    }
    let mut bins = Vec::with_capacity(calibration_bins);
    for index in 0..calibration_bins {
        let count = counts[index];
        let mean_prediction = if count == 0 {
            0.0
        } else {
            prediction_sums[index] / count as f64
        };
        let observed_rate = if count == 0 {
            0.0
        } else {
            outcome_sums[index] / count as f64
        };
        bins.push(CalibrationBin {
            index,
            lower: index as f64 / calibration_bins as f64,
            upper: (index + 1) as f64 / calibration_bins as f64,
            count,
            mean_prediction,
            observed_rate,
            absolute_gap: (mean_prediction - observed_rate).abs(),
        });
    }
    Ok(bins)
}

pub fn logistic_baseline_probability(
    elapsed_days: f64,
    midpoint_days: f64,
    slope: f64,
    floor: f64,
) -> SurvivalResult<f64> {
    if !elapsed_days.is_finite()
        || !midpoint_days.is_finite()
        || !slope.is_finite()
        || !floor.is_finite()
        || !(0.0..0.5).contains(&floor)
    {
        return Err(SurvivalError::InvalidInput(
            "logistic baseline parameters must be finite and have floor in [0, .5)".to_string(),
        ));
    }
    let linear = (slope * (elapsed_days - midpoint_days)).clamp(-40.0, 40.0);
    Ok((floor + (1.0 - floor) * sigmoid(linear)).clamp(floor, 1.0 - 1e-12))
}

fn validate_probability(probability: f64) -> SurvivalResult<()> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(SurvivalError::InvalidInput(format!(
            "probability must be finite and in [0, 1], got {probability}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;

    fn date(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, day).expect("valid date")
    }

    fn era() -> PolicyEra {
        PolicyEra::new("legacy", date(1), Some(date(20)))
    }

    fn provenance(fingerprint: String) -> SurvivalProvenance {
        SurvivalProvenance {
            input_fingerprint: fingerprint,
            code_revision: "test-revision".to_string(),
            fold_id: "fold-1".to_string(),
            training_cutoff: date(10),
        }
    }

    #[test]
    fn fold_clips_future_event_without_label_leakage() {
        let observations = vec![
            SurvivalObservation::reused_observation("a", date(1), date(10), "legacy"),
            SurvivalObservation::never_used("b", date(9), "legacy"),
        ];
        let fold = FoldSpec::new("f", None, date(5), date(6), date(9));
        let training = build_fold_training_inputs(&observations, fold).expect("fold");
        assert_eq!(training.observations.len(), 1);
        let clipped = &training.observations[0];
        assert_eq!(clipped.exit_date, date(6));
        assert!(!clipped.reused);
        assert_eq!(clipped.status(), ObservationStatus::RightCensored);
        assert!(training.source_fingerprint.starts_with(FINGERPRINT_PREFIX));
    }

    #[test]
    fn fold_applies_left_truncation_to_risk_exposure() {
        let observations = vec![SurvivalObservation::right_censored(
            "a",
            date(1),
            date(10),
            "legacy",
        )];
        let fold = FoldSpec::new("f", Some(date(4)), date(5), date(6), date(9));
        let training = build_fold_training_inputs(&observations, fold).expect("fold");
        assert_eq!(training.observations.len(), 1);
        let clipped = &training.observations[0];
        assert_eq!(clipped.entry_date, date(4));
        assert_eq!(clipped.exit_date, date(6));
        assert!(clipped.left_truncated);
    }

    #[test]
    fn never_used_mass_is_not_reuse_event_mass_and_scores_are_bounded() {
        let observations = vec![
            SurvivalObservation::reused_observation("a", date(1), date(3), "legacy"),
            SurvivalObservation::right_censored("b", date(1), date(4), "legacy"),
            SurvivalObservation::never_used("c", date(4), "legacy").with_mass(3.0),
        ];
        let model = SurvivalModel::fit_with_policy_eras(
            &observations,
            &[era()],
            &SurvivalConfig::default(),
        )
        .expect("model");
        assert_eq!(model.mass.never_used, 3.0);
        assert_eq!(model.mass.reused, 1.0);
        assert_eq!(model.mass.right_censored, 1.0);
        let prediction = model.try_predict(15.0, "legacy").expect("prediction");
        for value in [
            prediction.hazard,
            prediction.survival,
            prediction.reuse_probability,
            prediction.reusable_weight,
            prediction.never_used_weight,
        ] {
            assert!((0.0..=1.0).contains(&value));
            assert!(value.is_finite());
        }
        assert!(prediction.never_used_weight > 0.0);
        assert!(prediction.reusable_weight > 0.0);
    }

    #[test]
    fn policy_era_boundary_is_explicit_and_invalid_overlap_rejected() {
        let overlapping = vec![
            PolicyEra::new("a", date(1), Some(date(10))),
            PolicyEra::new("b", date(9), None),
        ];
        assert!(validate_policy_eras(&overlapping).is_err());
        let eras = vec![
            PolicyEra::new("a", date(1), Some(date(10))),
            PolicyEra::new("b", date(10), None),
        ];
        assert!(validate_policy_eras(&eras).is_ok());
        assert!(eras[0].contains(date(9)));
        assert!(!eras[0].contains(date(10)));
    }

    #[test]
    fn dated_prediction_preserves_elapsed_time_across_policy_eras() {
        let eras = vec![
            PolicyEra::new("a", date(1), Some(date(10))),
            PolicyEra::new("b", date(10), Some(date(20))),
        ];
        let observations = vec![
            SurvivalObservation::right_censored("word", date(2), date(10), "a"),
            SurvivalObservation::reused_observation("word", date(10), date(13), "b")
                .with_elapsed_offset(8),
        ];
        let model =
            SurvivalModel::fit_with_policy_eras(&observations, &eras, &SurvivalConfig::default())
                .expect("model");
        let prediction = model
            .try_predict_interval(date(1), date(12))
            .expect("dated prediction");
        assert!(prediction.hazard.is_finite());
        assert!((0.0..=1.0).contains(&prediction.survival));
    }

    #[test]
    fn artifact_is_not_promotable_without_evidence_gate() {
        let observations = vec![SurvivalObservation::reused_observation(
            "a",
            date(1),
            date(3),
            "legacy",
        )];
        let model = SurvivalModel::fit_with_policy_eras(
            &observations,
            &[era()],
            &SurvivalConfig::default(),
        )
        .expect("model");
        let fingerprint = model.training_fingerprint.clone();
        let artifact = SurvivalArtifact::new(
            model,
            "survival-experimental-v1",
            LeftTruncationMetadata::default(),
            provenance(fingerprint),
        );
        assert!(!artifact.promotion.enabled);
        assert!(!artifact.promotion_allowed());
        assert!(artifact.validate().is_ok());
        let json = artifact.to_json().expect("json");
        assert!(!json.contains("\"enabled\": true"));
        assert!(SurvivalArtifact::from_json(&json).is_ok());
    }

    #[test]
    fn comparison_reports_log_loss_brier_and_calibration() {
        let rows = vec![
            BinaryComparisonRow {
                observed: true,
                survival_probability: 0.8,
                logistic_probability: 0.6,
            },
            BinaryComparisonRow {
                observed: false,
                survival_probability: 0.2,
                logistic_probability: 0.4,
            },
        ];
        let comparison = compare_against_logistic_baseline(&rows).expect("metrics");
        assert_eq!(comparison.survival.count, 2);
        assert!(comparison.survival.log_loss < comparison.logistic_baseline.log_loss);
        assert!(comparison.survival.brier < comparison.logistic_baseline.brier);
        assert_eq!(comparison.survival.calibration.bins.len(), 10);
    }

    #[test]
    fn logistic_baseline_is_stable_at_extreme_inputs() {
        let probability =
            logistic_baseline_probability(1e9, 0.0, 100.0, 0.01).expect("probability");
        assert!((0.0..=1.0).contains(&probability));
        assert!(probability.is_finite());
    }
}
