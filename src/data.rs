use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::config::PriorConfig;

pub const WORDLE_LAUNCH_DATE: &str = "2021-06-19";

#[derive(Clone, Debug)]
pub struct ProjectPaths {
    pub root: PathBuf,
    pub config_prior: PathBuf,
    pub raw_history: PathBuf,
    pub seed_guesses: PathBuf,
    pub seed_answers: PathBuf,
    pub seed_reference_answers: PathBuf,
    pub seed_sources: PathBuf,
    pub manual_additions: PathBuf,
    pub merged_seed_answers: PathBuf,
    pub derived_answer_history: PathBuf,
    pub derived_modeled_answers: PathBuf,
    pub derived_seed_reconciliation: PathBuf,
    pub derived_predictive: PathBuf,
    pub pattern_table: PathBuf,
}

impl ProjectPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            config_prior: root.join("config/prior.toml"),
            raw_history: root.join("data/raw/nyt_daily_answers.jsonl"),
            seed_guesses: root.join("data/seed/valid_guesses.txt"),
            seed_answers: root.join("data/seed/candidate_answers.txt"),
            seed_reference_answers: root.join("data/seed/reference_candidate_answers.txt"),
            seed_sources: root.join("data/seed/sources.toml"),
            manual_additions: root.join("data/seed/manual_additions.txt"),
            merged_seed_answers: root.join("data/seed/candidate_answers.merged.txt"),
            derived_answer_history: root.join("data/derived/answer_history.csv"),
            derived_modeled_answers: root.join("data/derived/modeled_answers.csv"),
            derived_seed_reconciliation: root.join("data/derived/seed_reconciliation.csv"),
            derived_predictive: root.join("data/derived/predictive"),
            pattern_table: root.join("data/derived/pattern_table.bin"),
            root,
        }
    }

    pub fn ensure_layout(&self) -> Result<()> {
        for path in [
            self.root.join("config"),
            self.root.join("data/raw"),
            self.root.join("data/seed"),
            self.root.join("data/derived"),
            self.derived_predictive.clone(),
            self.root.join("data/formal"),
            self.root.join("src"),
            self.root.join("tests"),
            self.root.join("benches"),
        ] {
            fs::create_dir_all(&path)
                .with_context(|| format!("failed to create {}", path.display()))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NytDailyEntry {
    pub id: Option<u32>,
    pub solution: String,
    #[serde(with = "date_format")]
    pub print_date: NaiveDate,
    pub days_since_launch: Option<u32>,
    pub editor: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SyncSummary {
    pub fetched: usize,
    pub reverified: usize,
    pub changed: usize,
    pub total: usize,
    pub first_date: NaiveDate,
    pub last_date: NaiveDate,
    pub changed_dates: Vec<NaiveDate>,
}

pub fn normalize_word(word: &str) -> String {
    word.trim().to_ascii_lowercase()
}

pub fn read_word_list(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut seen = HashSet::new();
    let mut words = Vec::new();

    for line in reader.lines() {
        let word =
            normalize_word(&line.with_context(|| format!("failed to read {}", path.display()))?);
        if word.len() != 5 || !word.bytes().all(|byte| byte.is_ascii_lowercase()) {
            continue;
        }
        if seen.insert(word.clone()) {
            words.push(word);
        }
    }

    Ok(words)
}

pub fn read_history_jsonl(path: &Path) -> Result<Vec<NytDailyEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut entry: NytDailyEntry = serde_json::from_str(&line)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        entry.solution = normalize_word(&entry.solution);
        entries.push(entry);
    }

    entries.sort_by_key(|entry| entry.print_date);
    Ok(entries)
}

pub fn write_history_jsonl(path: &Path, entries: &[NytDailyEntry]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for entry in entries {
        serde_json::to_writer(&mut writer, entry).context("failed to serialize history entry")?;
        writer.write_all(b"\n").context("failed to write newline")?;
    }
    writer.flush().context("failed to flush history file")?;
    Ok(())
}

pub fn sync_nyt_history(
    paths: &ProjectPaths,
    config: &PriorConfig,
    today: NaiveDate,
) -> Result<SyncSummary> {
    paths.ensure_layout()?;

    let existing = read_history_jsonl(&paths.raw_history)?;
    let launch_date =
        NaiveDate::parse_from_str(WORDLE_LAUNCH_DATE, "%Y-%m-%d").expect("launch date is valid");
    let last_existing = existing.last().map(|entry| entry.print_date);
    let start_date = last_existing
        .map(|date| date - chrono::Days::new(config.sync_reverify_days.saturating_sub(1) as u64))
        .unwrap_or(launch_date)
        .max(launch_date);

    let client = Client::builder()
        .user_agent("maybe-wordle/0.1")
        .build()
        .context("failed to build HTTP client")?;

    let mut entries_by_date: BTreeMap<NaiveDate, NytDailyEntry> = existing
        .into_iter()
        .map(|entry| (entry.print_date, entry))
        .collect();

    let mut fetched = 0usize;
    let mut reverified = 0usize;
    let mut changed = 0usize;
    let mut changed_dates = Vec::new();

    let mut current = start_date;
    while current <= today {
        let fetched_entry = fetch_nyt_entry(&client, current)?;
        fetched += 1;
        if last_existing.is_some_and(|last| current <= last) {
            reverified += 1;
        }
        match entries_by_date.get(&current) {
            Some(existing_entry) if existing_entry == &fetched_entry => {}
            Some(_) => {
                changed += 1;
                changed_dates.push(current);
                entries_by_date.insert(current, fetched_entry);
            }
            None => {
                entries_by_date.insert(current, fetched_entry);
            }
        }
        current = current
            .checked_add_days(chrono::Days::new(1))
            .expect("date increment stays in range");
    }

    let entries: Vec<NytDailyEntry> = entries_by_date.into_values().collect();
    if entries.is_empty() {
        bail!("NYT history sync produced no entries");
    }
    write_history_jsonl(&paths.raw_history, &entries)?;

    Ok(SyncSummary {
        fetched,
        reverified,
        changed,
        total: entries.len(),
        first_date: entries.first().expect("entries not empty").print_date,
        last_date: entries.last().expect("entries not empty").print_date,
        changed_dates,
    })
}

fn fetch_nyt_entry(client: &Client, date: NaiveDate) -> Result<NytDailyEntry> {
    let url = format!(
        "https://www.nytimes.com/svc/wordle/v2/{}.json",
        date.format("%Y-%m-%d")
    );
    let mut entry = client
        .get(&url)
        .send()
        .with_context(|| format!("failed to fetch {}", url))?
        .error_for_status()
        .with_context(|| format!("NYT returned error for {}", url))?
        .json::<NytDailyEntry>()
        .with_context(|| format!("failed to decode {}", url))?;
    entry.solution = normalize_word(&entry.solution);
    Ok(entry)
}

mod date_format {
    use chrono::NaiveDate;
    use serde::{self, Deserialize, Deserializer, Serializer};

    const FORMAT: &str = "%Y-%m-%d";

    pub fn serialize<S>(date: &NaiveDate, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&date.format(FORMAT).to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<NaiveDate, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        NaiveDate::parse_from_str(&raw, FORMAT).map_err(serde::de::Error::custom)
    }
}
