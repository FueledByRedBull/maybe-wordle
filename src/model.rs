use std::collections::{BTreeMap, BTreeSet, HashSet};

use anyhow::{Context, Result};
use chrono::NaiveDate;
use csv::Writer;
use serde::{Deserialize, Serialize};

use crate::{
    atomic_file::atomic_write,
    config::PriorConfig,
    data::{
        NytDailyEntry, ProjectPaths, normalize_word, read_history_jsonl, read_word_list,
        validate_history_continuity,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightMode {
    Weighted,
    Uniform,
    CooldownOnly,
}

impl WeightMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::Uniform => "uniform",
            Self::CooldownOnly => "cooldown_only",
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
    pub historical_answers: usize,
    pub history_rows: usize,
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
        if word.len() == 5 && word.bytes().all(|byte| byte.is_ascii_lowercase()) {
            builders
                .entry(word.clone())
                .or_insert_with(|| AnswerRecordBuilder::new(word))
                .manual_entry = true;
        }
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
        historical_answers: model
            .answers
            .iter()
            .take(model.primary_answer_count)
            .filter(|answer| !answer.history_dates.is_empty())
            .count(),
        history_rows: model.history.len(),
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

fn build_history_rows(records: &[AnswerRecord], as_of: NaiveDate) -> Vec<AnswerHistoryRow> {
    records
        .iter()
        .filter_map(|record| {
            let first_seen = record.history_dates.first().copied()?;
            let last_seen = record.history_dates.last().copied()?;
            Some(AnswerHistoryRow {
                word: record.word.clone(),
                first_seen: first_seen.format("%Y-%m-%d").to_string(),
                last_seen: last_seen.format("%Y-%m-%d").to_string(),
                times_seen: record.history_dates.len(),
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
    use chrono::NaiveDate;

    use crate::config::PriorConfig;

    use super::{
        AnswerRecord, ModelVariant, WeightMode, load_model_with_variant, weight_snapshot,
        weight_snapshot_for_mode,
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
        ] {
            let snapshot = weight_snapshot_for_mode(&record, &PriorConfig::default(), as_of, mode);
            assert_eq!(snapshot.seen_count, 0);
            assert_eq!(snapshot.base_weight, 0.0);
            assert_eq!(snapshot.final_weight, 0.0);
        }
    }

    #[test]
    fn seed_only_variant_drops_history_only_answers() {
        let root = std::env::temp_dir().join("maybe-wordle-model-variant-test");
        let _ = std::fs::remove_dir_all(&root);
        let paths = crate::data::ProjectPaths::new(&root);
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
        let _ = std::fs::remove_dir_all(&root);
    }
}
