use std::collections::{BTreeMap, BTreeSet, HashSet};

use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use csv::Writer;
use serde::{Deserialize, Serialize};

use crate::{
    atomic_file::atomic_write,
    config::PriorConfig,
    data::{
        NytDailyEntry, ProjectPaths, normalize_word, read_history_jsonl, read_word_list,
        validate_answer_universe, validate_history_continuity,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightMode {
    Weighted,
    Uniform,
    CooldownOnly,
    EmpiricalFrequency,
    RegularizedFrequency,
    UsedUnused,
    RecencyBuckets,
}

impl WeightMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::Uniform => "uniform",
            Self::CooldownOnly => "cooldown_only",
            Self::EmpiricalFrequency => "empirical_frequency",
            Self::RegularizedFrequency => "regularized_frequency",
            Self::UsedUnused => "used_unused",
            Self::RecencyBuckets => "recency_buckets",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVariant {
    SeedOnly,
    SeedPlusHistory,
}

impl ModelVariant {
    pub fn label(self) -> &'static str {
        match self {
            Self::SeedOnly => "seed_only",
            Self::SeedPlusHistory => "seed_plus_history",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnswerRecord {
    pub word: String,
    pub in_seed: bool,
    pub manual_entry: bool,
    pub manual_weight: f64,
    pub history_dates: Vec<NaiveDate>,
}

#[derive(Clone, Debug)]
pub struct ModelData {
    pub guesses: Vec<String>,
    pub answers: Vec<AnswerRecord>,
    pub primary_answer_count: usize,
    pub history: Vec<NytDailyEntry>,
    pub variant: ModelVariant,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnswerHistoryRow {
    pub word: String,
    pub first_seen: String,
    pub last_seen: String,
    pub times_seen: usize,
    pub days_since_last_seen: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModeledAnswerRow {
    pub word: String,
    pub in_seed: bool,
    pub is_historical: bool,
    pub first_seen: String,
    pub last_seen: String,
    pub times_seen: usize,
    pub base_weight: f64,
    pub recency_weight: f64,
    pub manual_weight: f64,
    pub final_weight: f64,
}

#[derive(Clone, Debug)]
pub struct WeightSnapshot {
    pub first_seen: Option<NaiveDate>,
    pub last_seen: Option<NaiveDate>,
    pub seen_count: usize,
    pub base_weight: f64,
    pub recency_weight: f64,
    pub manual_weight: f64,
    pub final_weight: f64,
}

#[derive(Clone, Debug)]
pub struct BuildSummary {
    pub guess_count: usize,
    pub answer_count: usize,
    pub fallback_answer_count: usize,
    /// Distinct primary answers observed on or before the requested snapshot date.
    pub historical_answers: usize,
    /// Daily observations on or before the requested snapshot date.
    pub history_rows: usize,
    /// Full source archive size, including observations after a backdated snapshot.
    pub source_history_rows: usize,
}

pub fn load_model(paths: &ProjectPaths, config: &PriorConfig) -> Result<ModelData> {
    load_model_with_variant(paths, config, ModelVariant::SeedPlusHistory)
}

pub fn load_model_with_variant(
    paths: &ProjectPaths,
    config: &PriorConfig,
    variant: ModelVariant,
) -> Result<ModelData> {
    let guesses = read_word_list(&paths.seed_guesses)
        .with_context(|| format!("failed to load {}", paths.seed_guesses.display()))?;
    let seed_answers = read_word_list(&paths.seed_answers)
        .with_context(|| format!("failed to load {}", paths.seed_answers.display()))?;
    validate_answer_universe(&guesses, seed_answers.iter().map(String::as_str))
        .with_context(|| format!("invalid seed universe in {}", paths.root.display()))?;
    let manual_additions = read_word_list(&paths.manual_additions)
        .with_context(|| format!("failed to load {}", paths.manual_additions.display()))?;
    let history = read_history_jsonl(&paths.raw_history)
        .with_context(|| format!("failed to load {}", paths.raw_history.display()))?;
    if !config.allow_history_gaps {
        validate_history_continuity(&history)?;
    }

    let seed_lookup = seed_answers.iter().cloned().collect::<HashSet<_>>();
    let manual_keys = config
        .manual_weights
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut builders: BTreeMap<String, AnswerRecordBuilder> = BTreeMap::new();

    for word in &seed_answers {
        builders
            .entry(word.clone())
            .or_insert_with(|| AnswerRecordBuilder::new(word.clone()))
            .in_seed = true;
    }

    for entry in &history {
        let word = normalize_word(&entry.solution);
        if variant == ModelVariant::SeedPlusHistory || seed_lookup.contains(&word) {
            builders
                .entry(word.clone())
                .or_insert_with(|| AnswerRecordBuilder::new(word))
                .history_dates
                .push(entry.print_date);
        }
    }

    for word in manual_keys.into_iter().chain(manual_additions) {
        if word.len() != 5 || !word.bytes().all(|byte| byte.is_ascii_lowercase()) {
            bail!("invalid manual answer {word:?}: expected five lowercase ASCII letters");
        }
        builders
            .entry(word.clone())
            .or_insert_with(|| AnswerRecordBuilder::new(word))
            .manual_entry = true;
    }

    let extra_words = builders
        .keys()
        .filter(|word| !seed_lookup.contains(*word))
        .cloned()
        .collect::<Vec<_>>();

    let ordered_words = seed_answers
        .iter()
        .cloned()
        .chain(extra_words)
        .collect::<Vec<_>>();

    let mut answers = ordered_words
        .into_iter()
        .filter_map(|word| builders.remove(&word))
        .map(|builder| builder.finish(config))
        .collect::<Vec<_>>();
    validate_answer_universe(&guesses, answers.iter().map(|answer| answer.word.as_str()))
        .with_context(|| {
            format!(
                "invalid modeled answer universe in {}",
                paths.root.display()
            )
        })?;
    let primary_answer_count = answers.len();
    let answer_lookup = answers
        .iter()
        .map(|answer| answer.word.as_str())
        .collect::<HashSet<_>>();
    let fallback_words = guesses
        .iter()
        .filter(|word| !answer_lookup.contains(word.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    answers.extend(fallback_words.into_iter().map(|word| AnswerRecord {
        word,
        in_seed: false,
        manual_entry: false,
        manual_weight: 1.0,
        history_dates: Vec::new(),
    }));

    Ok(ModelData {
        guesses,
        answers,
        primary_answer_count,
        history,
        variant,
    })
}

pub fn build_model_artifacts(
    paths: &ProjectPaths,
    config: &PriorConfig,
    as_of: NaiveDate,
) -> Result<BuildSummary> {
    let model = load_model(paths, config)?;
    let primary_answers = &model.answers[..model.primary_answer_count];
    let history_rows = build_history_rows(primary_answers, as_of);
    let modeled_rows = build_modeled_rows(primary_answers, config, as_of);

    write_csv(&paths.derived_answer_history, &history_rows)?;
    write_csv(&paths.derived_modeled_answers, &modeled_rows)?;

    Ok(BuildSummary {
        guess_count: model.guesses.len(),
        answer_count: model.primary_answer_count,
        fallback_answer_count: model.answers.len() - model.primary_answer_count,
        historical_answers: history_rows.len(),
        history_rows: model
            .history
            .iter()
            .filter(|entry| entry.print_date <= as_of)
            .count(),
        source_history_rows: model.history.len(),
    })
}

pub fn weight_snapshot(
    record: &AnswerRecord,
    config: &PriorConfig,
    as_of: NaiveDate,
) -> WeightSnapshot {
    weight_snapshot_for_mode(record, config, as_of, WeightMode::Weighted)
}

pub fn weight_snapshot_for_mode(
    record: &AnswerRecord,
    config: &PriorConfig,
    as_of: NaiveDate,
    mode: WeightMode,
) -> WeightSnapshot {
    let cutoff = record.history_dates.partition_point(|date| *date <= as_of);
    let seen_dates = &record.history_dates[..cutoff];
    let first_seen = seen_dates.first().copied();
    let last_seen = seen_dates.last().copied();
    let seen_count = seen_dates.len();
    let eligible = record.in_seed || record.manual_entry || !seen_dates.is_empty();

    let (base_weight, recency_weight, final_weight) = match mode {
        WeightMode::Uniform => {
            let base_weight = if eligible { 1.0 } else { 0.0 };
            (base_weight, 1.0, base_weight)
        }
        WeightMode::CooldownOnly => {
            let base_weight = if record.in_seed || record.manual_entry {
                config.base_seed_weight
            } else if !seen_dates.is_empty() {
                config.base_history_only_weight
            } else {
                0.0
            };
            let recency_weight = if let Some(last_seen) = last_seen {
                let days_since_last_seen = (as_of - last_seen).num_days();
                if days_since_last_seen < config.cooldown_days {
                    config.cooldown_floor
                } else {
                    1.0
                }
            } else {
                1.0
            };
            let final_weight = base_weight * recency_weight * record.manual_weight;
            (base_weight, recency_weight, final_weight)
        }
        WeightMode::EmpiricalFrequency | WeightMode::RegularizedFrequency => {
            let smoothing = usize::from(mode == WeightMode::RegularizedFrequency);
            let base_weight = if eligible {
                (seen_count + smoothing) as f64
            } else {
                0.0
            };
            (base_weight, 1.0, base_weight * record.manual_weight)
        }
        WeightMode::UsedUnused | WeightMode::RecencyBuckets => {
            let base_weight = if record.in_seed || record.manual_entry {
                config.base_seed_weight
            } else if !seen_dates.is_empty() {
                config.base_history_only_weight
            } else {
                0.0
            };
            let recency_weight = match mode {
                WeightMode::UsedUnused => {
                    if seen_dates.is_empty() {
                        1.0
                    } else {
                        config.cooldown_floor
                    }
                }
                WeightMode::RecencyBuckets => last_seen
                    .map(|last_seen| recency_bucket_weight(config, (as_of - last_seen).num_days()))
                    .unwrap_or(1.0),
                _ => unreachable!("combined arm only contains experimental modes"),
            };
            let final_weight = base_weight * recency_weight * record.manual_weight;
            (base_weight, recency_weight, final_weight)
        }
        WeightMode::Weighted => {
            let base_weight = if record.in_seed || record.manual_entry {
                config.base_seed_weight
            } else if !seen_dates.is_empty() {
                config.base_history_only_weight
            } else {
                0.0
            };
            let recency_weight = if let Some(last_seen) = last_seen {
                let days_since_last_seen = (as_of - last_seen).num_days();
                if days_since_last_seen < config.cooldown_days {
                    config.cooldown_floor
                } else {
                    config.cooldown_floor
                        + (1.0 - config.cooldown_floor)
                            / (1.0
                                + (-config.logistic_k
                                    * ((days_since_last_seen as f64) - config.midpoint_days))
                                    .exp())
                }
            } else {
                1.0
            };
            let final_weight = base_weight * recency_weight * record.manual_weight;
            (base_weight, recency_weight, final_weight)
        }
    };

    WeightSnapshot {
        first_seen,
        last_seen,
        seen_count,
        base_weight,
        recency_weight,
        manual_weight: record.manual_weight,
        final_weight,
    }
}

fn recency_bucket_weight(config: &PriorConfig, days_since_last_seen: i64) -> f64 {
    let bucket_level = if days_since_last_seen < config.cooldown_days {
        0.0
    } else if days_since_last_seen < config.cooldown_days.saturating_mul(2) {
        0.25
    } else if days_since_last_seen < config.cooldown_days.saturating_mul(4) {
        0.5
    } else {
        1.0
    };
    config.cooldown_floor + (1.0 - config.cooldown_floor) * bucket_level
}

fn build_history_rows(records: &[AnswerRecord], as_of: NaiveDate) -> Vec<AnswerHistoryRow> {
    records
        .iter()
        .filter_map(|record| {
            let cutoff = record.history_dates.partition_point(|date| *date <= as_of);
            let seen_dates = &record.history_dates[..cutoff];
            let first_seen = seen_dates.first().copied()?;
            let last_seen = seen_dates.last().copied()?;
            Some(AnswerHistoryRow {
                word: record.word.clone(),
                first_seen: first_seen.format("%Y-%m-%d").to_string(),
                last_seen: last_seen.format("%Y-%m-%d").to_string(),
                times_seen: seen_dates.len(),
                days_since_last_seen: (as_of - last_seen).num_days(),
            })
        })
        .collect()
}

fn build_modeled_rows(
    records: &[AnswerRecord],
    config: &PriorConfig,
    as_of: NaiveDate,
) -> Vec<ModeledAnswerRow> {
    records
        .iter()
        .map(|record| {
            let snapshot = weight_snapshot(record, config, as_of);
            ModeledAnswerRow {
                word: record.word.clone(),
                in_seed: record.in_seed,
                is_historical: snapshot.seen_count > 0,
                first_seen: snapshot
                    .first_seen
                    .map(|date| date.format("%Y-%m-%d").to_string())
                    .unwrap_or_default(),
                last_seen: snapshot
                    .last_seen
                    .map(|date| date.format("%Y-%m-%d").to_string())
                    .unwrap_or_default(),
                times_seen: snapshot.seen_count,
                base_weight: snapshot.base_weight,
                recency_weight: snapshot.recency_weight,
                manual_weight: snapshot.manual_weight,
                final_weight: snapshot.final_weight,
            }
        })
        .collect()
}

fn write_csv<T: Serialize>(path: &std::path::Path, rows: &[T]) -> Result<()> {
    let mut writer = Writer::from_writer(Vec::new());
    for row in rows {
        writer.serialize(row).context("failed to write csv row")?;
    }
    writer.flush().context("failed to flush csv writer")?;
    let bytes = writer
        .into_inner()
        .context("failed to finalize csv writer")?;
    atomic_write(path, &bytes)
}

#[derive(Clone, Debug)]
struct AnswerRecordBuilder {
    word: String,
    in_seed: bool,
    manual_entry: bool,
    history_dates: Vec<NaiveDate>,
}

impl AnswerRecordBuilder {
    fn new(word: String) -> Self {
        Self {
            word,
            in_seed: false,
            manual_entry: false,
            history_dates: Vec::new(),
        }
    }

    fn finish(mut self, config: &PriorConfig) -> AnswerRecord {
        self.history_dates.sort_unstable();
        self.history_dates.dedup();
        AnswerRecord {
            word: self.word.clone(),
            in_seed: self.in_seed,
            manual_entry: self.manual_entry,
            manual_weight: config
                .manual_weights
                .get(&self.word)
                .copied()
                .unwrap_or(1.0),
            history_dates: self.history_dates,
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, NaiveDate};

    use crate::config::PriorConfig;

    use super::{
        AnswerRecord, ModelVariant, WeightMode, build_history_rows, load_model_with_variant,
        weight_snapshot, weight_snapshot_for_mode,
    };

    #[test]
    fn weight_snapshot_uses_cooldown_and_seed_defaults() {
        let config = PriorConfig::default();
        let record = AnswerRecord {
            word: "cigar".into(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![NaiveDate::from_ymd_opt(2024, 3, 1).expect("valid")],
        };

        let snapshot = weight_snapshot(
            &record,
            &config,
            NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid"),
        );
        assert_eq!(snapshot.base_weight, config.base_seed_weight);
        assert!(snapshot.recency_weight > config.cooldown_floor);
        assert!(snapshot.final_weight > 0.0);
    }

    #[test]
    fn weighted_default_config_cooldown_boundary_is_finite_monotonic_and_tightly_bounded() {
        const MAX_DEFAULT_CONFIG_BOUNDARY_DISCONTINUITY: f64 = 1.0e-6;

        let config = PriorConfig::default();
        assert!(config.cooldown_days > 0);
        assert!(config.logistic_k > 0.0);
        let last_seen = NaiveDate::from_ymd_opt(2024, 1, 1).expect("date");
        let record = AnswerRecord {
            word: "cigar".to_string(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![last_seen],
        };
        let snapshot_at = |days_since_last_seen: i64| {
            weight_snapshot_for_mode(
                &record,
                &config,
                last_seen + Duration::days(days_since_last_seen),
                WeightMode::Weighted,
            )
        };

        let before = snapshot_at(config.cooldown_days - 1);
        let at = snapshot_at(config.cooldown_days);
        let after = snapshot_at(config.cooldown_days + 1);

        for (label, snapshot) in [("before", &before), ("at", &at), ("after", &after)] {
            assert!(
                snapshot.recency_weight.is_finite(),
                "{label} cooldown weight must be finite"
            );
            assert!(
                (config.cooldown_floor..=1.0).contains(&snapshot.recency_weight),
                "{label} cooldown weight must be in [floor, 1], got {}",
                snapshot.recency_weight
            );
        }

        assert_eq!(before.recency_weight, config.cooldown_floor);
        // Far below the midpoint, consecutive logistic values can round identically.
        assert!(before.recency_weight <= at.recency_weight);
        assert!(at.recency_weight <= after.recency_weight);
        let midpoint = config.midpoint_days as i64;
        assert!(snapshot_at(midpoint).recency_weight < snapshot_at(midpoint + 1).recency_weight);

        let boundary_discontinuity = at.recency_weight - before.recency_weight;
        assert!(
            boundary_discontinuity <= MAX_DEFAULT_CONFIG_BOUNDARY_DISCONTINUITY,
            "default config cooldown boundary jump is too large: {boundary_discontinuity}"
        );
    }

    #[test]
    fn model_loading_rejects_malformed_programmatic_manual_keys() {
        let root = crate::test_support::TestDirectory::new("model-manual-key");
        let paths = crate::data::ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        std::fs::write(&paths.seed_guesses, "cigar\n").expect("guesses");
        std::fs::write(&paths.seed_answers, "cigar\n").expect("answers");
        std::fs::write(&paths.manual_additions, "").expect("manual");
        for word in ["cig4r", "four", "CIGAR"] {
            let mut config = PriorConfig::default();
            config.manual_weights.insert(word.into(), 1.0);
            let error =
                load_model_with_variant(&paths, &config, ModelVariant::SeedOnly).expect_err(word);
            assert!(format!("{error:#}").contains(word));
        }
    }

    #[test]
    fn model_loading_rejects_empty_required_vocabularies() {
        let root = crate::test_support::TestDirectory::new("model-empty");
        let paths = crate::data::ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        std::fs::write(&paths.manual_additions, "").expect("manual");
        for (guesses, answers) in [("", "cigar\n"), ("cigar\n", "")] {
            std::fs::write(&paths.seed_guesses, guesses).expect("guesses");
            std::fs::write(&paths.seed_answers, answers).expect("answers");
            let error =
                load_model_with_variant(&paths, &PriorConfig::default(), ModelVariant::SeedOnly)
                    .expect_err("empty vocabulary");
            assert!(format!("{error:#}").contains("empty"));
        }
    }

    #[test]
    fn model_loading_rejects_unguessable_seed_manual_and_history_answers() {
        let root = crate::test_support::TestDirectory::new("model-guessability");
        let paths = crate::data::ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        std::fs::write(&paths.seed_guesses, "cigar\n").expect("guesses");
        for source in ["seed", "manual", "history", "manual-weight"] {
            std::fs::write(
                &paths.seed_answers,
                if source == "seed" {
                    "rebut\n"
                } else {
                    "cigar\n"
                },
            )
            .expect("answers");
            std::fs::write(
                &paths.manual_additions,
                if source == "manual" { "rebut\n" } else { "" },
            )
            .expect("manual");
            std::fs::write(
                &paths.raw_history,
                if source == "history" {
                    "{\"solution\":\"rebut\",\"print_date\":\"2024-02-01\"}\n"
                } else {
                    ""
                },
            )
            .expect("history");
            let mut config = PriorConfig::default();
            if source == "manual-weight" {
                config.manual_weights.insert("rebut".into(), 1.0);
            }
            let error = load_model_with_variant(&paths, &config, ModelVariant::SeedPlusHistory)
                .expect_err(source);
            let message = format!("{error:#}");
            assert!(
                message.contains("rebut") && message.contains("guess"),
                "{message}"
            );
        }
    }

    #[test]
    fn backdated_artifact_summary_reports_effective_history() {
        let root = crate::test_support::TestDirectory::new("model-as-of");
        let paths = crate::data::ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        std::fs::write(&paths.seed_guesses, "cigar\nrebut\n").expect("guesses");
        std::fs::write(&paths.seed_answers, "cigar\nrebut\n").expect("answers");
        std::fs::write(&paths.manual_additions, "").expect("manual");
        let raw = concat!(
            "{\"solution\":\"cigar\",\"print_date\":\"2024-02-01\"}\n",
            "{\"solution\":\"rebut\",\"print_date\":\"2024-02-02\"}\n"
        );
        std::fs::write(&paths.raw_history, raw).expect("history");
        let summary = super::build_model_artifacts(
            &paths,
            &PriorConfig::default(),
            NaiveDate::from_ymd_opt(2024, 2, 1).expect("date"),
        )
        .expect("backdated artifacts");
        assert_eq!(summary.historical_answers, 1);
        assert_eq!(summary.history_rows, 1);
        assert_eq!(summary.source_history_rows, 2);
        assert_eq!(
            std::fs::read_to_string(&paths.raw_history).expect("source"),
            raw
        );
    }

    #[test]
    fn history_export_excludes_future_occurrences_and_future_only_words() {
        let as_of = NaiveDate::from_ymd_opt(2024, 2, 1).expect("date");
        let past = as_of - Duration::days(10);
        let record = AnswerRecord {
            word: "cigar".into(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![past, as_of],
        };
        let mut with_future = record.clone();
        with_future.history_dates.push(as_of + Duration::days(1));
        let future_only = AnswerRecord {
            word: "rebut".into(),
            history_dates: vec![as_of + Duration::days(2)],
            ..record.clone()
        };
        let expected = build_history_rows(std::slice::from_ref(&record), as_of);
        let actual = build_history_rows(&[with_future.clone(), future_only], as_of);
        assert_eq!(
            serde_json::to_value(&actual).expect("serialize"),
            serde_json::to_value(&expected).expect("serialize")
        );
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].first_seen, past.to_string());
        assert_eq!(actual[0].last_seen, as_of.to_string());
        assert_eq!(actual[0].times_seen, 2);
        assert_eq!(actual[0].days_since_last_seen, 0);
        assert_eq!(
            weight_snapshot(&record, &PriorConfig::default(), as_of).final_weight,
            weight_snapshot(&with_future, &PriorConfig::default(), as_of).final_weight
        );
        // A historical export must not mutate the full source provenance.
        assert_eq!(with_future.history_dates.len(), 3);
    }

    #[test]
    fn future_history_only_answer_is_not_eligible_before_first_seen() {
        let as_of = NaiveDate::from_ymd_opt(2024, 2, 1).expect("date");
        let record = AnswerRecord {
            word: "cigar".to_string(),
            in_seed: false,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![NaiveDate::from_ymd_opt(2024, 3, 1).expect("future")],
        };
        for mode in [
            WeightMode::Weighted,
            WeightMode::Uniform,
            WeightMode::CooldownOnly,
            WeightMode::EmpiricalFrequency,
            WeightMode::RegularizedFrequency,
            WeightMode::UsedUnused,
            WeightMode::RecencyBuckets,
        ] {
            let snapshot = weight_snapshot_for_mode(&record, &PriorConfig::default(), as_of, mode);
            assert_eq!(snapshot.seen_count, 0);
            assert_eq!(snapshot.base_weight, 0.0);
            assert_eq!(snapshot.final_weight, 0.0);
        }
    }

    #[test]
    fn experimental_weight_modes_have_stable_labels_and_serde_names() {
        for (mode, label) in [
            (WeightMode::UsedUnused, "used_unused"),
            (WeightMode::RecencyBuckets, "recency_buckets"),
        ] {
            assert_eq!(mode.label(), label);
            let encoded = serde_json::to_string(&mode).expect("serialize weight mode");
            assert_eq!(encoded, format!("\"{label}\""));
            assert_eq!(
                serde_json::from_str::<WeightMode>(&encoded).expect("decode weight mode"),
                mode
            );
        }
    }

    #[test]
    fn used_unused_is_binary_and_recency_buckets_are_monotone_at_boundaries() {
        let config = PriorConfig {
            cooldown_days: 4,
            cooldown_floor: 0.1,
            ..PriorConfig::default()
        };
        let last_seen = NaiveDate::from_ymd_opt(2024, 1, 1).expect("date");
        let used = AnswerRecord {
            word: "cigar".to_string(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: vec![last_seen],
        };
        let never_used = AnswerRecord {
            history_dates: Vec::new(),
            ..used.clone()
        };

        let used_snapshot = weight_snapshot_for_mode(
            &used,
            &config,
            last_seen + Duration::days(20),
            WeightMode::UsedUnused,
        );
        assert_eq!(used_snapshot.recency_weight, config.cooldown_floor);
        let never_used_snapshot = weight_snapshot_for_mode(
            &never_used,
            &config,
            last_seen + Duration::days(20),
            WeightMode::UsedUnused,
        );
        assert_eq!(never_used_snapshot.recency_weight, 1.0);

        let snapshot_at = |days_since_last_seen: i64| {
            weight_snapshot_for_mode(
                &used,
                &config,
                last_seen + Duration::days(days_since_last_seen),
                WeightMode::RecencyBuckets,
            )
            .recency_weight
        };
        let expected = |level: f64| config.cooldown_floor + (1.0 - config.cooldown_floor) * level;
        let boundaries = [
            (0, expected(0.0)),
            (3, expected(0.0)),
            (4, expected(0.25)),
            (7, expected(0.25)),
            (8, expected(0.5)),
            (15, expected(0.5)),
            (16, expected(1.0)),
        ];
        let mut previous = 0.0;
        for (days, expected_weight) in boundaries {
            let weight = snapshot_at(days);
            assert!((weight - expected_weight).abs() < f64::EPSILON);
            assert!(
                weight >= previous,
                "recency bucket decreased at {days} days"
            );
            previous = weight;
        }
    }

    #[test]
    fn experimental_modes_ignore_history_after_as_of_date() {
        let as_of = NaiveDate::from_ymd_opt(2024, 2, 1).expect("date");
        let future = as_of + Duration::days(1);
        let without_future_history = AnswerRecord {
            word: "cigar".to_string(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.5,
            history_dates: Vec::new(),
        };
        let with_future_history = AnswerRecord {
            history_dates: vec![future],
            ..without_future_history.clone()
        };

        for mode in [WeightMode::UsedUnused, WeightMode::RecencyBuckets] {
            let before = weight_snapshot_for_mode(
                &without_future_history,
                &PriorConfig::default(),
                as_of,
                mode,
            );
            let after = weight_snapshot_for_mode(
                &with_future_history,
                &PriorConfig::default(),
                as_of,
                mode,
            );
            assert_eq!(after.seen_count, 0);
            assert_eq!(after.first_seen, None);
            assert_eq!(after.last_seen, None);
            assert_eq!(after.base_weight, before.base_weight);
            assert_eq!(after.recency_weight, before.recency_weight);
            assert_eq!(after.manual_weight, before.manual_weight);
            assert_eq!(after.final_weight, before.final_weight);
        }
    }

    #[test]
    fn frequency_modes_use_observed_counts_and_additive_smoothing() {
        let as_of = NaiveDate::from_ymd_opt(2024, 3, 1).expect("date");
        let record = AnswerRecord {
            word: "cigar".to_string(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 2.0,
            history_dates: vec![
                NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
                NaiveDate::from_ymd_opt(2024, 2, 1).expect("date"),
            ],
        };

        let empirical = weight_snapshot_for_mode(
            &record,
            &PriorConfig::default(),
            as_of,
            WeightMode::EmpiricalFrequency,
        );
        let regularized = weight_snapshot_for_mode(
            &record,
            &PriorConfig::default(),
            as_of,
            WeightMode::RegularizedFrequency,
        );

        assert_eq!(empirical.base_weight, 2.0);
        assert_eq!(empirical.final_weight, 4.0);
        assert_eq!(regularized.base_weight, 3.0);
        assert_eq!(regularized.final_weight, 6.0);
    }

    #[test]
    fn seed_only_variant_drops_history_only_answers() {
        let root = crate::test_support::TestDirectory::new("model-variant");
        let paths = crate::data::ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        std::fs::write(&paths.seed_guesses, "cigar\nrebut\n").expect("guesses");
        std::fs::write(&paths.seed_answers, "cigar\n").expect("seed");
        std::fs::write(&paths.manual_additions, "").expect("manual");
        let history = [
            r#"{"id":1,"solution":"cigar","print_date":"2021-06-19"}"#,
            r#"{"id":2,"solution":"rebut","print_date":"2021-06-20"}"#,
        ]
        .join("\n");
        std::fs::write(&paths.raw_history, history).expect("history");

        let config = PriorConfig::default();
        let seed_only =
            load_model_with_variant(&paths, &config, ModelVariant::SeedOnly).expect("model");
        let full =
            load_model_with_variant(&paths, &config, ModelVariant::SeedPlusHistory).expect("model");

        assert_eq!(seed_only.primary_answer_count, 1);
        assert_eq!(seed_only.answers.len(), 2);
        assert_eq!(seed_only.answers[1].word, "rebut");
        assert_eq!(full.primary_answer_count, 2);
        assert_eq!(full.answers.len(), 2);
    }
}
