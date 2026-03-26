use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::config::PriorConfig;

pub const WORDLE_LAUNCH_DATE: &str = "2021-06-19";
const NYT_WORDLE_BASE_URL: &str = "https://www.nytimes.com/svc/wordle/v2";

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
    pub partial_sync: bool,
    pub failed_dates: Vec<NaiveDate>,
    pub last_successful_date: Option<NaiveDate>,
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
    sync_nyt_history_with_base_url(paths, config, today, NYT_WORDLE_BASE_URL)
}

fn sync_nyt_history_with_base_url(
    paths: &ProjectPaths,
    config: &PriorConfig,
    today: NaiveDate,
    base_url: &str,
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
        .timeout(Duration::from_secs(config.sync_request_timeout_seconds))
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
    let mut failed_dates = Vec::new();
    let mut last_successful_date = None;

    let mut current = start_date;
    while current <= today {
        match fetch_nyt_entry_with_retry(
            &client,
            current,
            base_url,
            config.sync_retry_attempts,
            config.sync_retry_backoff_millis,
        ) {
            Ok(fetched_entry) => {
                fetched += 1;
                last_successful_date = Some(current);
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
            }
            Err(_) => {
                failed_dates.push(current);
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
        partial_sync: !failed_dates.is_empty(),
        failed_dates,
        last_successful_date,
    })
}

fn fetch_nyt_entry_with_retry(
    client: &Client,
    date: NaiveDate,
    base_url: &str,
    retry_attempts: usize,
    retry_backoff_millis: u64,
) -> Result<NytDailyEntry> {
    let mut attempt = 0usize;
    loop {
        match fetch_nyt_entry(client, date, base_url) {
            Ok(entry) => return Ok(entry),
            Err(error) => {
                if attempt >= retry_attempts || !is_retryable_fetch_error(&error) {
                    return Err(error);
                }
                let backoff = retry_backoff_millis.saturating_mul((attempt + 1) as u64);
                if backoff > 0 {
                    thread::sleep(Duration::from_millis(backoff));
                }
                attempt += 1;
            }
        }
    }
}

fn is_retryable_fetch_error(error: &anyhow::Error) -> bool {
    let Some(reqwest_error) = error.downcast_ref::<reqwest::Error>() else {
        return false;
    };
    match reqwest_error.status() {
        Some(status) => status.is_server_error(),
        None => {
            reqwest_error.is_timeout()
                || reqwest_error.is_connect()
                || reqwest_error.is_request()
                || reqwest_error.is_body()
                || reqwest_error.is_decode()
        }
    }
}

fn fetch_nyt_entry(client: &Client, date: NaiveDate, base_url: &str) -> Result<NytDailyEntry> {
    let url = format!("{base_url}/{}.json", date.format("%Y-%m-%d"));
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

#[cfg(test)]
fn make_test_entry(date: NaiveDate, solution: &str) -> NytDailyEntry {
    NytDailyEntry {
        id: Some(1),
        solution: solution.to_string(),
        print_date: date,
        days_since_launch: Some(1),
        editor: None,
    }
}

#[cfg(test)]
fn test_json_response(entry: &NytDailyEntry) -> String {
    serde_json::to_string(entry).expect("serialize test entry")
}

#[cfg(test)]
fn read_request_path(stream: &mut std::net::TcpStream) -> Result<String> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .context("failed to read test request line")?;
    let mut parts = request_line.split_whitespace();
    let _method = parts.next().context("missing test request method")?;
    let path = parts.next().context("missing test request path")?;
    Ok(path.to_string())
}

#[cfg(test)]
fn write_response(stream: &mut std::net::TcpStream, status: u16, body: &str) -> Result<()> {
    let reason = match status {
        200 => "OK",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .context("failed to write test response")?;
    stream.flush().context("failed to flush test response")?;
    Ok(())
}

#[cfg(test)]
fn spawn_test_server<F>(expected_requests: usize, handler: F) -> (String, thread::JoinHandle<()>)
where
    F: Fn(&str, usize) -> (u16, String) + Send + Sync + 'static,
{
    use std::net::TcpListener;
    use std::sync::Arc;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("test server addr");
    let handler = Arc::new(handler);
    let join = thread::spawn(move || {
        let mut counts = std::collections::HashMap::<String, usize>::new();
        for _ in 0..expected_requests {
            let (mut stream, _) = listener.accept().expect("accept test request");
            let path = read_request_path(&mut stream).expect("request path");
            let count = counts.entry(path.clone()).or_insert(0);
            *count += 1;
            let (status, body) = handler(&path, *count);
            write_response(&mut stream, status, &body).expect("write test response");
        }
    });
    (format!("http://{}", addr), join)
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

#[cfg(test)]
mod tests {
    use super::{
        PriorConfig, ProjectPaths, make_test_entry, spawn_test_server,
        sync_nyt_history_with_base_url, test_json_response,
    };
    use chrono::NaiveDate;
    use std::{fs, path::PathBuf};

    fn temp_project_root(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("maybe-wordle-{name}-{unique}"));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn prior_config_round_trips_sync_fields() {
        let config = PriorConfig {
            sync_request_timeout_seconds: 7,
            sync_retry_attempts: 4,
            sync_retry_backoff_millis: 250,
            ..PriorConfig::default()
        };
        let encoded = toml::to_string_pretty(&config).expect("encode");
        assert!(encoded.contains("sync_request_timeout_seconds = 7"));
        assert!(encoded.contains("sync_retry_attempts = 4"));
        assert!(encoded.contains("sync_retry_backoff_millis = 250"));
        let decoded: PriorConfig = toml::from_str(&encoded).expect("decode");
        assert_eq!(decoded.sync_request_timeout_seconds, 7);
        assert_eq!(decoded.sync_retry_attempts, 4);
        assert_eq!(decoded.sync_retry_backoff_millis, 250);
    }

    #[test]
    fn sync_nyt_history_retries_before_succeeding() {
        let root = temp_project_root("retry-success");
        let paths = ProjectPaths::new(&root);
        let today = NaiveDate::from_ymd_opt(2021, 6, 19).expect("today");
        let (base_url, join) = spawn_test_server(2, |path, count| {
            if count == 1 {
                (500, String::new())
            } else {
                let date = path
                    .rsplit('/')
                    .next()
                    .expect("path segment")
                    .trim_end_matches(".json");
                let entry = make_test_entry(
                    NaiveDate::parse_from_str(date, "%Y-%m-%d").expect("date"),
                    "cigar",
                );
                (200, test_json_response(&entry))
            }
        });
        let config = PriorConfig {
            sync_request_timeout_seconds: 1,
            sync_retry_attempts: 1,
            sync_retry_backoff_millis: 0,
            sync_reverify_days: 1,
            ..PriorConfig::default()
        };

        let summary =
            sync_nyt_history_with_base_url(&paths, &config, today, &base_url).expect("sync");
        join.join().expect("server thread");

        assert_eq!(summary.fetched, 1);
        assert!(!summary.partial_sync);
        assert!(summary.failed_dates.is_empty());
        assert_eq!(summary.last_successful_date, Some(today));
        assert_eq!(summary.total, 1);
        assert_eq!(summary.first_date, today);
        assert_eq!(summary.last_date, today);
    }

    #[test]
    fn sync_nyt_history_preserves_existing_data_on_partial_sync() {
        let root = temp_project_root("partial-sync");
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let second = NaiveDate::from_ymd_opt(2021, 6, 20).expect("second");
        super::write_history_jsonl(&paths.raw_history, &[make_test_entry(first, "cigar")])
            .expect("seed history");

        let (base_url, join) = spawn_test_server(2, move |path, _count| {
            let date = path
                .rsplit('/')
                .next()
                .expect("path segment")
                .trim_end_matches(".json");
            if date == "2021-06-19" {
                let entry = make_test_entry(first, "cigar");
                (200, test_json_response(&entry))
            } else {
                (500, String::new())
            }
        });
        let config = PriorConfig {
            sync_request_timeout_seconds: 1,
            sync_retry_attempts: 0,
            sync_retry_backoff_millis: 0,
            sync_reverify_days: 1,
            ..PriorConfig::default()
        };

        let summary =
            sync_nyt_history_with_base_url(&paths, &config, second, &base_url).expect("sync");
        join.join().expect("server thread");

        assert_eq!(summary.fetched, 1);
        assert!(summary.partial_sync);
        assert_eq!(summary.failed_dates, vec![second]);
        assert_eq!(summary.last_successful_date, Some(first));
        assert_eq!(summary.total, 1);
        assert_eq!(summary.first_date, first);
        assert_eq!(summary.last_date, first);
        let rewritten = super::read_history_jsonl(&paths.raw_history).expect("read history");
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].print_date, first);
    }

    #[test]
    fn sync_nyt_history_errors_when_nothing_can_be_fetched() {
        let root = temp_project_root("no-success");
        let paths = ProjectPaths::new(&root);
        let today = NaiveDate::from_ymd_opt(2021, 6, 19).expect("today");
        let (base_url, join) = spawn_test_server(1, |_path, _count| (500, String::new()));
        let config = PriorConfig {
            sync_request_timeout_seconds: 1,
            sync_retry_attempts: 0,
            sync_retry_backoff_millis: 0,
            ..PriorConfig::default()
        };

        let error = sync_nyt_history_with_base_url(&paths, &config, today, &base_url)
            .expect_err("sync should fail");
        join.join().expect("server thread");

        let message = format!("{error:#}");
        assert!(message.contains("NYT history sync produced no entries"));
        assert!(
            super::read_history_jsonl(&paths.raw_history)
                .expect("read history")
                .is_empty()
        );
    }
}
