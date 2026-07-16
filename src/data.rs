use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fmt,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use reqwest::{StatusCode, blocking::Client, header::RETRY_AFTER};
use serde::{Deserialize, Serialize};

use crate::{atomic_file::atomic_write, config::PriorConfig};

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

pub fn validate_history_continuity(entries: &[NytDailyEntry]) -> Result<()> {
    for pair in entries.windows(2) {
        let expected = pair[0]
            .print_date
            .checked_add_days(chrono::Days::new(1))
            .ok_or_else(|| anyhow::anyhow!("history date overflow"))?;
        if pair[1].print_date != expected {
            bail!(
                "NYT history is non-contiguous: expected {}, found {}; run sync-data to repair gaps or set allow_history_gaps = true for an explicit retrospective override",
                expected,
                pair[1].print_date
            );
        }
    }
    Ok(())
}

pub fn write_history_jsonl(path: &Path, entries: &[NytDailyEntry]) -> Result<()> {
    validate_history_continuity(entries)?;
    let mut bytes = Vec::new();
    for entry in entries {
        serde_json::to_writer(&mut bytes, entry).context("failed to serialize history entry")?;
        bytes.write_all(b"\n").context("failed to write newline")?;
    }
    let decoded = read_history_jsonl_bytes(&bytes)?;
    validate_history_continuity(&decoded)?;
    atomic_write(path, &bytes)
}

fn read_history_jsonl_bytes(bytes: &[u8]) -> Result<Vec<NytDailyEntry>> {
    let mut entries = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let mut entry: NytDailyEntry =
            serde_json::from_slice(line).context("failed to validate serialized history entry")?;
        entry.solution = normalize_word(&entry.solution);
        entries.push(entry);
    }
    entries.sort_by_key(|entry| entry.print_date);
    Ok(entries)
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
    let reverify_start = last_existing
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

    let mut current = launch_date;
    while current <= today {
        let needs_fetch = !entries_by_date.contains_key(&current) || current >= reverify_start;
        if !needs_fetch {
            current = current
                .checked_add_days(chrono::Days::new(1))
                .expect("date increment stays in range");
            continue;
        }
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
    let persisted_entries = if let Err(error) = validate_history_continuity(&entries) {
        if paths.raw_history.exists() {
            let existing = read_history_jsonl(&paths.raw_history)?;
            validate_history_continuity(&existing).with_context(|| {
                "sync could not repair every history gap and the existing archive is also non-contiguous"
            })?;
            existing
        } else {
            return Err(error).context("sync could not create a contiguous history archive");
        }
    } else {
        write_history_jsonl(&paths.raw_history, &entries)?;
        entries
    };

    Ok(SyncSummary {
        fetched,
        reverified,
        changed,
        total: persisted_entries.len(),
        first_date: persisted_entries
            .first()
            .expect("entries not empty")
            .print_date,
        last_date: persisted_entries
            .last()
            .expect("entries not empty")
            .print_date,
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
                let exponent = 1u64.checked_shl(attempt.min(16) as u32).unwrap_or(u64::MAX);
                let exponential = retry_backoff_millis.saturating_mul(exponent).min(30_000);
                let backoff = retry_after_for_error(&error)
                    .map(|duration| duration.as_millis().min(60_000) as u64)
                    .unwrap_or(exponential);
                if backoff > 0 {
                    thread::sleep(Duration::from_millis(backoff));
                }
                attempt += 1;
            }
        }
    }
}

fn is_retryable_fetch_error(error: &anyhow::Error) -> bool {
    if let Some(status_error) = error.downcast_ref::<HttpStatusError>() {
        return status_error.status == StatusCode::TOO_MANY_REQUESTS
            || status_error.status.is_server_error();
    }
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

fn retry_after_for_error(error: &anyhow::Error) -> Option<Duration> {
    error
        .downcast_ref::<HttpStatusError>()
        .and_then(|status| status.retry_after)
}

#[derive(Debug)]
struct HttpStatusError {
    status: StatusCode,
    retry_after: Option<Duration>,
    url: String,
}

impl fmt::Display for HttpStatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "HTTP {} returned for {}", self.status, self.url)
    }
}

impl Error for HttpStatusError {}

fn parse_retry_after(value: &str) -> Option<Duration> {
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&Utc);
    let seconds = (retry_at - Utc::now()).num_seconds().max(0) as u64;
    Some(Duration::from_secs(seconds))
}

fn fetch_nyt_entry(client: &Client, date: NaiveDate, base_url: &str) -> Result<NytDailyEntry> {
    let url = format!("{base_url}/{}.json", date.format("%Y-%m-%d"));
    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("failed to fetch {}", url))?;
    if !response.status().is_success() {
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_retry_after);
        return Err(HttpStatusError {
            status: response.status(),
            retry_after,
            url,
        }
        .into());
    }
    let mut entry = response
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
    write_response_with_headers(stream, status, &[], body)
}

#[cfg(test)]
fn write_response_with_headers(
    stream: &mut std::net::TcpStream,
    status: u16,
    headers: &[(&str, &str)],
    body: &str,
) -> Result<()> {
    let reason = match status {
        200 => "OK",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let extra_headers = headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect::<String>();
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
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
        sync_nyt_history_with_base_url, test_json_response, write_response_with_headers,
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
    fn sync_retries_rate_limits_and_respects_retry_after() {
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;

        let root = temp_project_root("retry-rate-limit");
        let paths = ProjectPaths::new(&root);
        let today = NaiveDate::from_ymd_opt(2021, 6, 19).expect("today");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("address");
        let join = std::thread::spawn(move || {
            for request in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept");
                let mut line = String::new();
                BufReader::new(stream.try_clone().expect("clone"))
                    .read_line(&mut line)
                    .expect("request");
                if request == 0 {
                    write_response_with_headers(&mut stream, 429, &[("Retry-After", "0")], "")
                        .expect("429");
                } else {
                    write_response_with_headers(
                        &mut stream,
                        200,
                        &[],
                        &test_json_response(&make_test_entry(today, "cigar")),
                    )
                    .expect("200");
                }
            }
        });
        let config = PriorConfig {
            sync_retry_attempts: 1,
            sync_retry_backoff_millis: 10_000,
            sync_reverify_days: 1,
            ..PriorConfig::default()
        };
        let summary =
            sync_nyt_history_with_base_url(&paths, &config, today, &format!("http://{addr}"))
                .expect("rate-limit retry");
        join.join().expect("server");
        assert_eq!(summary.total, 1);
        assert!(!summary.partial_sync);
    }

    #[test]
    fn later_sync_repairs_a_transient_middle_date_gap() {
        let root = temp_project_root("repair-gap");
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let middle = first
            .checked_add_days(chrono::Days::new(1))
            .expect("middle");
        let last = middle.checked_add_days(chrono::Days::new(1)).expect("last");
        super::write_history_jsonl(&paths.raw_history, &[make_test_entry(first, "cigar")])
            .expect("seed");
        let config = PriorConfig {
            sync_retry_attempts: 0,
            sync_retry_backoff_millis: 0,
            sync_reverify_days: 1,
            ..PriorConfig::default()
        };

        let (first_url, first_join) = spawn_test_server(3, move |path, _| {
            let date = path
                .rsplit('/')
                .next()
                .expect("segment")
                .trim_end_matches(".json");
            let date = NaiveDate::parse_from_str(date, "%Y-%m-%d").expect("date");
            if date == middle {
                (500, String::new())
            } else {
                (200, test_json_response(&make_test_entry(date, "cigar")))
            }
        });
        let partial = sync_nyt_history_with_base_url(&paths, &config, last, &first_url)
            .expect("partial sync");
        first_join.join().expect("server");
        assert!(partial.partial_sync);
        assert_eq!(
            super::read_history_jsonl(&paths.raw_history)
                .expect("old")
                .len(),
            1
        );

        let (second_url, second_join) = spawn_test_server(3, move |path, _| {
            let date = path
                .rsplit('/')
                .next()
                .expect("segment")
                .trim_end_matches(".json");
            let date = NaiveDate::parse_from_str(date, "%Y-%m-%d").expect("date");
            (200, test_json_response(&make_test_entry(date, "rebut")))
        });
        let repaired = sync_nyt_history_with_base_url(&paths, &config, last, &second_url)
            .expect("repair sync");
        second_join.join().expect("server");
        assert!(!repaired.partial_sync);
        let history = super::read_history_jsonl(&paths.raw_history).expect("history");
        assert_eq!(history.len(), 3);
        super::validate_history_continuity(&history).expect("contiguous");
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
