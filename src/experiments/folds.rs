use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use chrono::{Days, NaiveDate};
use serde::{Deserialize, Serialize};

pub const EVALUATION_POLICY_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DateRange {
    pub start: NaiveDate,
    pub end: NaiveDate,
}

impl DateRange {
    pub fn new(start: NaiveDate, end: NaiveDate) -> Result<Self> {
        if start > end {
            bail!("date range start {start} is after end {end}");
        }
        Ok(Self { start, end })
    }

    pub fn days(self) -> u64 {
        (self.end - self.start).num_days() as u64 + 1
    }

    pub fn contains(self, date: NaiveDate) -> bool {
        date >= self.start && date <= self.end
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RollingOriginConfig {
    pub minimum_training_days: u64,
    pub validation_days: u64,
    pub step_days: u64,
    pub sealed_test_days: u64,
    pub maximum_folds: usize,
}

impl Default for RollingOriginConfig {
    fn default() -> Self {
        Self {
            minimum_training_days: 365,
            validation_days: 30,
            step_days: 30,
            sealed_test_days: 30,
            maximum_folds: 12,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RollingOriginFold {
    pub index: usize,
    pub training: DateRange,
    pub validation: DateRange,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvaluationPlan {
    pub history: DateRange,
    pub development: DateRange,
    pub sealed_test: DateRange,
    pub folds: Vec<RollingOriginFold>,
    pub config: RollingOriginConfig,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvaluationPolicy {
    pub format_version: u32,
    pub development_cutoff: NaiveDate,
    pub sealed_test: DateRange,
    #[serde(default)]
    pub excluded_validation: Vec<DateRange>,
}

impl EvaluationPolicy {
    pub fn parse_toml(source: &str) -> Result<Self> {
        let policy: Self = toml::from_str(source).context("parse evaluation policy")?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let source = fs::read_to_string(path)
            .with_context(|| format!("read evaluation policy {}", path.display()))?;
        Self::parse_toml(&source)
    }

    pub fn validate(&self) -> Result<()> {
        DateRange::new(self.sealed_test.start, self.sealed_test.end)?;
        if self.format_version != EVALUATION_POLICY_FORMAT_VERSION {
            bail!(
                "unsupported evaluation policy format {}; expected {}",
                self.format_version,
                EVALUATION_POLICY_FORMAT_VERSION
            );
        }
        if self.development_cutoff >= self.sealed_test.start {
            bail!("development cutoff must precede the sealed test");
        }
        for (index, range) in self.excluded_validation.iter().enumerate() {
            DateRange::new(range.start, range.end)?;
            if range.end > self.development_cutoff {
                bail!("excluded validation range extends beyond development cutoff");
            }
            if index > 0 && self.excluded_validation[index - 1].end >= range.start {
                bail!("excluded validation ranges must be sorted and non-overlapping");
            }
        }
        Ok(())
    }
}

pub fn build_rolling_origin_plan(
    history: DateRange,
    config: RollingOriginConfig,
) -> Result<EvaluationPlan> {
    DateRange::new(history.start, history.end)?;
    if config.minimum_training_days == 0 {
        bail!("minimum_training_days must be positive");
    }
    if config.validation_days == 0 {
        bail!("validation_days must be positive");
    }
    if config.step_days == 0 {
        bail!("step_days must be positive");
    }
    if config.sealed_test_days == 0 {
        bail!("sealed_test_days must be positive");
    }
    if config.maximum_folds == 0 {
        bail!("maximum_folds must be positive");
    }

    let required_days = config
        .minimum_training_days
        .checked_add(config.validation_days)
        .and_then(|days| days.checked_add(config.sealed_test_days))
        .ok_or_else(|| anyhow::anyhow!("rolling-origin day counts overflowed"))?;
    if history.days() < required_days {
        bail!(
            "history has {} days but rolling-origin evaluation requires at least {}",
            history.days(),
            required_days
        );
    }

    let sealed_test_start = history
        .end
        .checked_sub_days(Days::new(config.sealed_test_days - 1))
        .ok_or_else(|| anyhow::anyhow!("sealed test window underflowed"))?;
    let development_end = sealed_test_start
        .checked_sub_days(Days::new(1))
        .ok_or_else(|| anyhow::anyhow!("development window underflowed"))?;
    let development = DateRange::new(history.start, development_end)?;
    let sealed_test = DateRange::new(sealed_test_start, history.end)?;

    let mut validation_end = development.end;
    let mut newest_first = Vec::new();
    loop {
        let validation_start = validation_end
            .checked_sub_days(Days::new(config.validation_days - 1))
            .ok_or_else(|| anyhow::anyhow!("validation date underflowed"))?;
        let training_end = validation_start
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow::anyhow!("training date underflowed"))?;
        let training_days = (training_end - history.start).num_days() + 1;
        if training_days < config.minimum_training_days as i64 {
            break;
        }
        newest_first.push((
            DateRange::new(history.start, training_end)?,
            DateRange::new(validation_start, validation_end)?,
        ));
        if newest_first.len() == config.maximum_folds {
            break;
        }
        validation_end = validation_end
            .checked_sub_days(Days::new(config.step_days))
            .ok_or_else(|| anyhow::anyhow!("rolling-origin step underflowed"))?;
    }

    if newest_first.is_empty() {
        bail!("history cannot produce a complete rolling-origin validation fold");
    }
    newest_first.reverse();
    let folds = newest_first
        .into_iter()
        .enumerate()
        .map(|(index, (training, validation))| RollingOriginFold {
            index,
            training,
            validation,
        })
        .collect::<Vec<_>>();

    Ok(EvaluationPlan {
        history,
        development,
        sealed_test,
        folds,
        config,
    })
}

pub fn build_declared_rolling_origin_plan(
    available_history: DateRange,
    config: RollingOriginConfig,
    policy: &EvaluationPolicy,
) -> Result<EvaluationPlan> {
    policy.validate()?;
    DateRange::new(available_history.start, available_history.end)?;
    if available_history.end < policy.development_cutoff {
        bail!(
            "history ends on {} but declared development requires data through {}",
            available_history.end,
            policy.development_cutoff
        );
    }
    if policy.sealed_test.days() != config.sealed_test_days {
        bail!(
            "declared sealed test has {} days but rolling configuration requires {}",
            policy.sealed_test.days(),
            config.sealed_test_days
        );
    }
    if config.minimum_training_days == 0
        || config.validation_days == 0
        || config.step_days == 0
        || config.maximum_folds == 0
    {
        bail!("rolling-origin day counts and maximum folds must be positive");
    }

    let history = DateRange::new(available_history.start, policy.development_cutoff)?;
    let minimum_days = config
        .minimum_training_days
        .checked_add(config.validation_days)
        .ok_or_else(|| anyhow::anyhow!("rolling-origin day counts overflowed"))?;
    if history.days() < minimum_days {
        bail!(
            "development history has {} days but rolling-origin evaluation requires at least {}",
            history.days(),
            minimum_days
        );
    }

    let development = history;
    let mut validation_end = development.end;
    let mut newest_first = Vec::new();
    loop {
        let validation_start = validation_end
            .checked_sub_days(Days::new(config.validation_days - 1))
            .ok_or_else(|| anyhow::anyhow!("validation date underflowed"))?;
        let validation = DateRange::new(validation_start, validation_end)?;
        let training_end = validation_start
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow::anyhow!("training date underflowed"))?;
        let training_days = (training_end - history.start).num_days() + 1;
        if training_days < config.minimum_training_days as i64 {
            break;
        }
        if !policy
            .excluded_validation
            .iter()
            .any(|excluded| ranges_overlap(validation, *excluded))
        {
            newest_first.push((DateRange::new(history.start, training_end)?, validation));
            if newest_first.len() == config.maximum_folds {
                break;
            }
        }
        validation_end = validation_end
            .checked_sub_days(Days::new(config.step_days))
            .ok_or_else(|| anyhow::anyhow!("rolling-origin step underflowed"))?;
    }
    if newest_first.is_empty() {
        bail!("history cannot produce a complete permitted validation fold");
    }
    newest_first.reverse();
    let folds = newest_first
        .into_iter()
        .enumerate()
        .map(|(index, (training, validation))| RollingOriginFold {
            index,
            training,
            validation,
        })
        .collect();

    Ok(EvaluationPlan {
        history,
        development,
        sealed_test: policy.sealed_test,
        folds,
        config,
    })
}

fn ranges_overlap(left: DateRange, right: DateRange) -> bool {
    left.start <= right.end && right.start <= left.end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(raw: &str) -> NaiveDate {
        NaiveDate::parse_from_str(raw, "%Y-%m-%d").expect("date")
    }

    #[test]
    fn rolling_origin_plan_seals_final_window_and_never_leaks() {
        let history = DateRange::new(date("2024-01-01"), date("2026-01-30")).expect("range");
        let config = RollingOriginConfig {
            minimum_training_days: 365,
            validation_days: 30,
            step_days: 30,
            sealed_test_days: 30,
            maximum_folds: 4,
        };
        let plan = build_rolling_origin_plan(history, config).expect("plan");

        assert_eq!(plan.sealed_test.start, date("2026-01-01"));
        assert_eq!(plan.sealed_test.end, date("2026-01-30"));
        assert_eq!(plan.development.end, date("2025-12-31"));
        assert_eq!(plan.folds.len(), 4);
        assert_eq!(
            plan.folds.last().expect("last fold").validation.end,
            plan.development.end
        );
        for fold in &plan.folds {
            assert!(fold.training.end < fold.validation.start);
            assert!(fold.validation.end <= plan.development.end);
            assert!(fold.validation.end < plan.sealed_test.start);
        }
        assert!(
            plan.folds
                .windows(2)
                .all(|pair| pair[0].validation.start < pair[1].validation.start)
        );
    }

    #[test]
    fn rolling_origin_plan_rejects_insufficient_history() {
        let history = DateRange::new(date("2025-01-01"), date("2025-02-28")).expect("range");
        let error = build_rolling_origin_plan(history, RollingOriginConfig::default())
            .expect_err("short history must fail");
        assert!(error.to_string().contains("requires at least"));
    }

    #[test]
    fn declared_plan_preserves_new_seal_and_excludes_consumed_targets() {
        let history = DateRange::new(date("2021-06-19"), date("2026-08-26")).expect("range");
        let policy = EvaluationPolicy {
            format_version: EVALUATION_POLICY_FORMAT_VERSION,
            development_cutoff: date("2026-08-26"),
            sealed_test: DateRange::new(date("2026-08-28"), date("2026-09-26")).expect("seal"),
            excluded_validation: vec![
                DateRange::new(date("2026-06-18"), date("2026-07-17")).expect("old seal"),
            ],
        };
        let plan =
            build_declared_rolling_origin_plan(history, RollingOriginConfig::default(), &policy)
                .expect("plan");

        assert_eq!(plan.folds.len(), 12);
        assert_eq!(plan.development.end, date("2026-08-26"));
        assert_eq!(plan.sealed_test.start, date("2026-08-28"));
        assert_eq!(plan.sealed_test.end, date("2026-09-26"));
        assert_eq!(
            plan.folds.last().expect("latest fold").validation,
            DateRange::new(date("2026-07-28"), date("2026-08-26")).expect("latest")
        );
        assert!(plan.folds.iter().all(|fold| {
            !ranges_overlap(
                fold.validation,
                DateRange::new(date("2026-06-18"), date("2026-07-17")).expect("old seal"),
            )
        }));
    }

    #[test]
    fn shipped_evaluation_policy_is_valid() {
        let policy = EvaluationPolicy::parse_toml(include_str!("../../config/evaluation.toml"))
            .expect("policy");
        assert_eq!(policy.development_cutoff, date("2026-08-26"));
        let mut invalid = policy.clone();
        std::mem::swap(&mut invalid.sealed_test.start, &mut invalid.sealed_test.end);
        assert!(invalid.validate().is_err());
        let range = &mut invalid.excluded_validation[0];
        std::mem::swap(&mut range.start, &mut range.end);
        invalid.sealed_test = policy.sealed_test;
        assert!(invalid.validate().is_err());
    }
}
