use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fmt,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
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
    /// Distinct dates for which a fetch was started, including failed or cancelled fetches.
    pub attempted: usize,
    /// Successfully fetched responses, including responses discarded on partial failure.
    pub fetched: usize,
    pub reverified: usize,
    /// Newly inserted or changed records actually published in the archive.
    pub applied: usize,
    /// Previously persisted records whose decoded values remain unchanged.
    pub retained: usize,
    pub changed: usize,
    pub total: usize,
    pub first_date: NaiveDate,
    pub last_date: NaiveDate,
    pub changed_dates: Vec<NaiveDate>,
    pub partial_sync: bool,
    pub failed_dates: Vec<NaiveDate>,
    pub last_successful_date: Option<NaiveDate>,
    pub retained_existing_archive: bool,
    pub coverage_complete: bool,
    pub requested_first_date: NaiveDate,
    pub requested_last_date: NaiveDate,
    pub missing_dates: Vec<NaiveDate>,
    pub failures: Vec<SyncFailure>,
    pub cancelled: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncFailure {
    pub date: NaiveDate,
    pub message: String,
}

pub fn normalize_word(word: &str) -> String {
    word.trim().to_ascii_lowercase()
}

fn normalize_and_validate_entry(
    entry: &mut NytDailyEntry,
    expected_date: Option<NaiveDate>,
) -> Result<()> {
    entry.solution = normalize_word(&entry.solution);
    if entry.solution.len() != 5 || !entry.solution.bytes().all(|byte| byte.is_ascii_lowercase()) {
        bail!(
            "NYT solution {:?} must be exactly five lowercase ASCII letters",
            entry.solution
        );
    }
    if let Some(expected_date) = expected_date
        && entry.print_date != expected_date
    {
        bail!(
            "NYT response date mismatch: requested {}, returned {}",
            expected_date,
            entry.print_date
        );
    }
    Ok(())
}

pub fn read_word_list(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    read_word_list_from(BufReader::new(file), path)
}

pub(crate) fn read_word_list_from(reader: impl BufRead, path: &Path) -> Result<Vec<String>> {
    let mut seen = HashSet::new();
    let mut words = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("failed to read {}:{}", path.display(), index + 1))?;
        let word = line.trim();
        if word.is_empty() || word.starts_with('#') {
            continue;
        }
        if word.len() != 5 || !word.bytes().all(|byte| byte.is_ascii_lowercase()) {
            bail!(
                "{}:{}: word must be exactly five lowercase ASCII letters, found {word:?}",
                path.display(),
                index + 1
            );
        }
        if seen.insert(word.to_owned()) {
            words.push(word.to_owned());
        }
    }

    Ok(words)
}

/// Both solvers require every possible answer to be a legal submitted guess.
pub fn validate_answer_universe<'a>(
    guesses: &[String],
    answers: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    if guesses.is_empty() {
        bail!("guess vocabulary must not be empty");
    }
    for guess in guesses {
        if guess.len() != 5 || !guess.bytes().all(|byte| byte.is_ascii_lowercase()) {
            bail!("invalid guess {guess:?}: expected five lowercase ASCII letters");
        }
    }
    let guess_set = guesses.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut answer_count = 0;
    for answer in answers {
        if !guess_set.contains(answer) {
            bail!("answer {answer:?} is absent from the legal guess vocabulary");
        }
        answer_count += 1;
    }
    if answer_count == 0 {
        bail!("answer vocabulary must not be empty");
    }
    Ok(())
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
        normalize_and_validate_entry(&mut entry, None)
            .with_context(|| format!("invalid history entry in {}", path.display()))?;
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

/// Checks requested coverage in addition to adjacency within the stored archive.
pub fn validate_history_coverage(
    entries: &[NytDailyEntry],
    first_date: NaiveDate,
    last_date: NaiveDate,
) -> Result<()> {
    if first_date > last_date {
        bail!("history coverage range is reversed: {first_date}..{last_date}");
    }
    validate_history_continuity(entries)?;
    let (Some(first), Some(last)) = (entries.first(), entries.last()) else {
        bail!("history is empty; requested coverage is {first_date}..{last_date}");
    };
    if first.print_date > first_date || last.print_date < last_date {
        bail!(
            "history covers {}..{}, not the complete requested range {first_date}..{last_date}",
            first.print_date,
            last.print_date
        );
    }
    Ok(())
}

pub fn write_history_jsonl(path: &Path, entries: &[NytDailyEntry]) -> Result<()> {
    validate_history_continuity(entries)?;
    let mut bytes = Vec::new();
    for entry in entries {
        let mut entry = entry.clone();
        normalize_and_validate_entry(&mut entry, None)?;
        serde_json::to_writer(&mut bytes, &entry).context("failed to serialize history entry")?;
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
        normalize_and_validate_entry(&mut entry, None)
            .context("failed to validate serialized history entry")?;
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
    sync_nyt_history_cancellable(paths, config, today, &|| false)
}

/// Cancels between requests and during retry waits; an in-flight request remains
/// bounded by `sync_request_timeout_seconds`. Partial/cancelled syncs never publish.
pub fn sync_nyt_history_cancellable(
    paths: &ProjectPaths,
    config: &PriorConfig,
    today: NaiveDate,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<SyncSummary> {
    sync_nyt_history_with_base_url_cancellable(paths, config, today, NYT_WORDLE_BASE_URL, cancelled)
}

#[cfg(test)]
fn sync_nyt_history_with_base_url(
    paths: &ProjectPaths,
    config: &PriorConfig,
    today: NaiveDate,
    base_url: &str,
) -> Result<SyncSummary> {
    sync_nyt_history_with_base_url_cancellable(paths, config, today, base_url, &|| false)
}

fn sync_nyt_history_with_base_url_cancellable(
    paths: &ProjectPaths,
    config: &PriorConfig,
    today: NaiveDate,
    base_url: &str,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<SyncSummary> {
    paths.ensure_layout()?;

    let existing = read_history_jsonl(&paths.raw_history)?;
    let launch_date =
        NaiveDate::parse_from_str(WORDLE_LAUNCH_DATE, "%Y-%m-%d").expect("launch date is valid");
    if today < launch_date {
        bail!("sync end date {today} precedes Wordle launch {launch_date}");
    }
    if config.sync_reverify_days < 1 {
        bail!("sync_reverify_days must be at least 1");
    }
    let last_existing = existing.last().map(|entry| entry.print_date);
    let reverify_start = last_existing
        .and_then(|date| {
            date.checked_sub_days(chrono::Days::new((config.sync_reverify_days - 1) as u64))
        })
        .unwrap_or(launch_date)
        .max(launch_date);

    let client = Client::builder()
        .user_agent("maybe-wordle/0.1")
        .timeout(Duration::from_secs(config.sync_request_timeout_seconds))
        .build()
        .context("failed to build HTTP client")?;

    let mut entries_by_date: BTreeMap<NaiveDate, NytDailyEntry> = existing
        .iter()
        .cloned()
        .map(|entry| (entry.print_date, entry))
        .collect();

    let mut attempted = 0usize;
    let mut fetched = 0usize;
    let mut reverified = 0usize;
    let mut failures = Vec::new();
    let mut last_successful_date = None;
    let mut was_cancelled = false;

    for current in launch_date.iter_days().take_while(|date| *date <= today) {
        if cancelled() {
            was_cancelled = true;
            break;
        }
        let needs_fetch = !entries_by_date.contains_key(&current) || current >= reverify_start;
        if !needs_fetch {
            continue;
        }
        attempted += 1;
        match fetch_nyt_entry_with_retry(
            &client,
            current,
            base_url,
            config.sync_retry_attempts,
            config.sync_retry_backoff_millis,
            cancelled,
        ) {
            Ok(fetched_entry) => {
                fetched += 1;
                last_successful_date = Some(current);
                if entries_by_date.contains_key(&current) {
                    reverified += 1;
                }
                entries_by_date.insert(current, fetched_entry);
            }
            Err(error) => {
                was_cancelled = error.downcast_ref::<SyncCancelled>().is_some();
                failures.push(SyncFailure {
                    date: current,
                    message: format!("{error:#}"),
                });
                if was_cancelled {
                    break;
                }
            }
        }
    }

    let entries: Vec<NytDailyEntry> = entries_by_date.into_values().collect();
    was_cancelled |= cancelled();
    let retained_existing_archive = was_cancelled
        || !failures.is_empty()
        || validate_history_coverage(&entries, launch_date, today).is_err();
    let persisted_entries = if retained_existing_archive {
        if existing.is_empty() || validate_history_continuity(&existing).is_err() {
            let reason = if entries.is_empty() {
                "NYT history sync produced no entries"
            } else {
                "NYT history sync could not produce complete requested coverage"
            };
            let causes = failures
                .iter()
                .map(|failure| format!("{}: {}", failure.date, failure.message))
                .collect::<Vec<_>>()
                .join("; ");
            bail!(
                "{reason}; no usable nonempty contiguous prior archive; requested={launch_date}..{today} attempted={attempted} fetched={fetched} applied=0 cancelled={was_cancelled}; failures=[{causes}]"
            );
        }
        &existing
    } else {
        write_history_jsonl(&paths.raw_history, &entries)?;
        &entries
    };
    let original_by_date = existing
        .iter()
        .map(|entry| (entry.print_date, entry))
        .collect::<BTreeMap<_, _>>();
    let changed_dates = persisted_entries
        .iter()
        .filter_map(|entry| match original_by_date.get(&entry.print_date) {
            Some(original) if *original != entry => Some(entry.print_date),
            _ => None,
        })
        .collect::<Vec<_>>();
    let retained = persisted_entries
        .iter()
        .filter(|entry| {
            original_by_date
                .get(&entry.print_date)
                .is_some_and(|original| *original == *entry)
        })
        .count();
    let applied = persisted_entries.len() - retained;
    let persisted_dates = persisted_entries
        .iter()
        .map(|entry| entry.print_date)
        .collect::<HashSet<_>>();
    let missing_dates = launch_date
        .iter_days()
        .take_while(|date| *date <= today)
        .filter(|date| !persisted_dates.contains(date))
        .collect::<Vec<_>>();
    let coverage_complete = missing_dates.is_empty();
    let first_date = persisted_entries
        .first()
        .context("sync has no persisted first date")?
        .print_date;
    let last_date = persisted_entries
        .last()
        .context("sync has no persisted last date")?
        .print_date;

    Ok(SyncSummary {
        attempted,
        fetched,
        reverified,
        applied,
        retained,
        changed: changed_dates.len(),
        total: persisted_entries.len(),
        first_date,
        last_date,
        changed_dates,
        partial_sync: retained_existing_archive || !coverage_complete,
        failed_dates: failures.iter().map(|failure| failure.date).collect(),
        last_successful_date,
        retained_existing_archive,
        coverage_complete,
        requested_first_date: launch_date,
        requested_last_date: today,
        missing_dates,
        failures,
        cancelled: was_cancelled,
    })
}

fn fetch_nyt_entry_with_retry(
    client: &Client,
    date: NaiveDate,
    base_url: &str,
    retry_attempts: usize,
    retry_backoff_millis: u64,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<NytDailyEntry> {
    let mut attempt = 0usize;
    loop {
        if cancelled() {
            return Err(SyncCancelled.into());
        }
        match fetch_nyt_entry(client, date, base_url) {
            Ok(entry) => return Ok(entry),
            Err(error) => {
                if cancelled() {
                    return Err(error.context(SyncCancelled));
                }
                if attempt >= retry_attempts || !is_retryable_fetch_error(&error) {
                    return Err(error);
                }
                let exponent = 1u64.checked_shl(attempt.min(16) as u32).unwrap_or(u64::MAX);
                let exponential = retry_backoff_millis.saturating_mul(exponent).min(30_000);
                let backoff = retry_after_for_error(&error)
                    .map(|duration| duration.as_millis().min(60_000) as u64)
                    .unwrap_or(exponential);
                let duration = Duration::from_millis(backoff);
                let started = Instant::now();
                while started.elapsed() < duration {
                    if cancelled() {
                        return Err(error.context(SyncCancelled));
                    }
                    thread::sleep(
                        duration
                            .saturating_sub(started.elapsed())
                            .min(Duration::from_millis(50)),
                    );
                }
                attempt += 1;
            }
        }
    }
}

#[derive(Debug)]
struct SyncCancelled;

impl fmt::Display for SyncCancelled {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("history sync cancelled")
    }
}

impl Error for SyncCancelled {}

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
    normalize_and_validate_entry(&mut entry, Some(date))
        .with_context(|| format!("invalid response from {}", url))?;
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
        PriorConfig, ProjectPaths, fetch_nyt_entry, make_test_entry, read_history_jsonl,
        spawn_test_server, sync_nyt_history_with_base_url, test_json_response, write_history_jsonl,
        write_response_with_headers,
    };
    use crate::test_support::TestDirectory;
    use chrono::NaiveDate;
    use reqwest::blocking::Client;
    use std::fs;

    #[test]
    fn word_lists_reject_malformed_rows_with_file_and_line() {
        let directory = crate::test_support::TestDirectory::new("word-validation");
        let path = directory.path().join("words.txt");
        for invalid in [
            "four", "longer", "CIGAR", "ciGar", "cig4r", "cigár", "cig ar",
        ] {
            fs::write(&path, format!("# Header\n\ncigar\n{invalid}\n")).expect("fixture");
            let error = super::read_word_list(&path).expect_err(invalid);
            let message = format!("{error:#}");
            assert!(message.contains("words.txt:4"), "{message}");
        }
    }

    #[test]
    fn word_lists_keep_first_duplicate_and_accept_blank_comment_rows() {
        let directory = crate::test_support::TestDirectory::new("word-comments");
        let path = directory.path().join("words.txt");
        fs::write(
            &path,
            "# Header\n\n  # Comment\nrebut\ncigar\nrebut\n  cigar  \n",
        )
        .expect("fixture");
        assert_eq!(
            super::read_word_list(&path).expect("words"),
            ["rebut", "cigar"]
        );
    }

    #[test]
    fn history_read_normalizes_uppercase_solutions_and_rejects_malformed_ones() {
        let root = TestDirectory::new("history-validation");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        let date = NaiveDate::from_ymd_opt(2021, 6, 19).expect("date");
        fs::write(
            &paths.raw_history,
            test_json_response(&make_test_entry(date, "CIGAR")),
        )
        .expect("uppercase fixture");
        let history = read_history_jsonl(&paths.raw_history).expect("read uppercase fixture");
        assert_eq!(history[0].solution, "cigar");

        fs::write(
            &paths.raw_history,
            test_json_response(&make_test_entry(date, "abcd!")),
        )
        .expect("malformed fixture");
        let error = read_history_jsonl(&paths.raw_history).expect_err("malformed solution");
        assert!(format!("{error:#}").contains("five lowercase ASCII letters"));
    }

    #[test]
    fn history_write_rejects_malformed_solutions_before_persisting() {
        let root = TestDirectory::new("history-write-validation");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        let date = NaiveDate::from_ymd_opt(2021, 6, 19).expect("date");
        let error = write_history_jsonl(&paths.raw_history, &[make_test_entry(date, "four")])
            .expect_err("malformed solution");
        assert!(format!("{error:#}").contains("five lowercase ASCII letters"));
        assert!(!paths.raw_history.exists());
    }

    #[test]
    fn fetched_entry_must_match_requested_date() {
        let requested = NaiveDate::from_ymd_opt(2021, 6, 19).expect("requested date");
        let returned = requested.succ_opt().expect("returned date");
        let (base_url, join) = spawn_test_server(1, move |_path, _count| {
            (200, test_json_response(&make_test_entry(returned, "cigar")))
        });
        let client = Client::builder().build().expect("client");
        let error = fetch_nyt_entry(&client, requested, &base_url)
            .expect_err("date mismatch must be rejected");
        join.join().expect("server thread");
        assert!(format!("{error:#}").contains("response date mismatch"));
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
        let root = TestDirectory::new("retry-success");
        let paths = ProjectPaths::new(root.path());
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

        let root = TestDirectory::new("retry-rate-limit");
        let paths = ProjectPaths::new(root.path());
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
        let root = TestDirectory::new("repair-gap");
        let paths = ProjectPaths::new(root.path());
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
        let root = TestDirectory::new("partial-sync");
        let paths = ProjectPaths::new(root.path());
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
    fn empty_existing_archive_with_gapped_responses_is_not_published_or_panicked() {
        let root = TestDirectory::new("empty-sync-gap");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        fs::write(&paths.raw_history, "").expect("empty archive");
        let (base_url, join) = spawn_test_server(3, |path, _| {
            let date = NaiveDate::parse_from_str(
                path.rsplit('/')
                    .next()
                    .expect("date path")
                    .trim_end_matches(".json"),
                "%Y-%m-%d",
            )
            .expect("date");
            if date == NaiveDate::from_ymd_opt(2021, 6, 20).expect("middle") {
                (500, String::new())
            } else {
                (200, test_json_response(&make_test_entry(date, "cigar")))
            }
        });
        let config = PriorConfig {
            sync_retry_attempts: 0,
            ..PriorConfig::default()
        };
        let result = sync_nyt_history_with_base_url(
            &paths,
            &config,
            NaiveDate::from_ymd_opt(2021, 6, 21).expect("last"),
            &base_url,
        );
        join.join().expect("server");
        assert!(result.is_err(), "an empty retained archive is not usable");
        assert_eq!(fs::read(&paths.raw_history).expect("archive retained"), b"");
    }

    #[test]
    fn missing_requested_boundary_is_not_a_complete_new_archive() {
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let last = first.succ_opt().expect("last");
        for missing in [first, last] {
            let root = TestDirectory::new("boundary-sync-gap");
            let paths = ProjectPaths::new(root.path());
            let (base_url, join) = spawn_test_server(2, move |path, _| {
                let date = NaiveDate::parse_from_str(
                    path.rsplit('/')
                        .next()
                        .expect("date path")
                        .trim_end_matches(".json"),
                    "%Y-%m-%d",
                )
                .expect("date");
                if date == missing {
                    (500, String::new())
                } else {
                    (200, test_json_response(&make_test_entry(date, "cigar")))
                }
            });
            let config = PriorConfig {
                sync_retry_attempts: 0,
                ..PriorConfig::default()
            };
            let result = sync_nyt_history_with_base_url(&paths, &config, last, &base_url);
            join.join().expect("server");
            assert!(result.is_err(), "missing requested boundary {missing}");
            assert!(!paths.raw_history.exists());
        }
    }

    #[test]
    fn discarded_changed_fetches_are_not_reported_as_applied_changes() {
        let root = TestDirectory::new("discarded-sync-changes");
        let paths = ProjectPaths::new(root.path());
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        write_history_jsonl(&paths.raw_history, &[make_test_entry(first, "cigar")]).expect("seed");
        let before = fs::read(&paths.raw_history).expect("original archive");
        let last = NaiveDate::from_ymd_opt(2021, 6, 21).expect("last");
        let (base_url, join) = spawn_test_server(3, |path, _| {
            let date = NaiveDate::parse_from_str(
                path.rsplit('/')
                    .next()
                    .expect("date path")
                    .trim_end_matches(".json"),
                "%Y-%m-%d",
            )
            .expect("date");
            if date == NaiveDate::from_ymd_opt(2021, 6, 20).expect("middle") {
                (500, String::new())
            } else {
                (200, test_json_response(&make_test_entry(date, "rebut")))
            }
        });
        let config = PriorConfig {
            sync_retry_attempts: 0,
            ..PriorConfig::default()
        };
        let summary = sync_nyt_history_with_base_url(&paths, &config, last, &base_url)
            .expect("retained archive");
        join.join().expect("server");
        assert_eq!(summary.changed, 0);
        assert!(summary.changed_dates.is_empty());
        assert_eq!(
            (
                summary.attempted,
                summary.fetched,
                summary.applied,
                summary.retained
            ),
            (3, 2, 0, 1)
        );
        assert!(summary.retained_existing_archive);
        assert!(!summary.coverage_complete);
        assert_eq!(
            summary.missing_dates,
            vec![first.succ_opt().expect("middle"), last]
        );
        assert_eq!(summary.failures.len(), 1);
        assert!(summary.failures[0].message.contains("HTTP 500"));
        assert_eq!(
            fs::read(&paths.raw_history).expect("preserved archive"),
            before
        );
    }

    #[test]
    fn failed_reverification_retains_complete_coverage_without_claiming_sync_success() {
        let root = TestDirectory::new("complete-but-unverified");
        let paths = ProjectPaths::new(root.path());
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let last = first.succ_opt().expect("last");
        write_history_jsonl(
            &paths.raw_history,
            &[
                make_test_entry(first, "cigar"),
                make_test_entry(last, "rebut"),
            ],
        )
        .expect("archive");
        let before = fs::read(&paths.raw_history).expect("original");
        let (base_url, join) = spawn_test_server(2, move |path, _| {
            if path.ends_with("2021-06-19.json") {
                (200, test_json_response(&make_test_entry(first, "sissy")))
            } else {
                (500, String::new())
            }
        });
        let config = PriorConfig {
            sync_retry_attempts: 0,
            ..PriorConfig::default()
        };
        let summary = sync_nyt_history_with_base_url(&paths, &config, last, &base_url)
            .expect("retained archive");
        join.join().expect("server");
        assert!(summary.coverage_complete);
        assert!(summary.partial_sync);
        assert!(summary.retained_existing_archive);
        assert_eq!(
            (
                summary.attempted,
                summary.fetched,
                summary.applied,
                summary.retained,
                summary.changed
            ),
            (2, 1, 0, 2, 0)
        );
        assert!(summary.missing_dates.is_empty());
        assert_eq!(summary.failed_dates, vec![last]);
        assert_eq!(summary.failures[0].date, last);
        assert!(summary.failures[0].message.contains("HTTP 500"));
        assert_eq!(
            fs::read(&paths.raw_history).expect("unchanged archive"),
            before
        );
    }

    #[test]
    fn sync_cancellation_during_retry_wait_preserves_archive_promptly() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        let root = TestDirectory::new("cancel-sync-retry");
        let paths = ProjectPaths::new(root.path());
        let today = NaiveDate::from_ymd_opt(2021, 6, 19).expect("today");
        write_history_jsonl(&paths.raw_history, &[make_test_entry(today, "cigar")])
            .expect("archive");
        let before = fs::read(&paths.raw_history).expect("original");
        let responded = Arc::new(AtomicBool::new(false));
        let server_responded = Arc::clone(&responded);
        let (base_url, join) = spawn_test_server(1, move |_, _| {
            server_responded.store(true, Ordering::Release);
            (500, String::new())
        });
        let config = PriorConfig {
            sync_retry_attempts: 3,
            sync_retry_backoff_millis: 10_000,
            ..PriorConfig::default()
        };
        let retry_polls = AtomicUsize::new(0);
        let cancelled =
            || responded.load(Ordering::Acquire) && retry_polls.fetch_add(1, Ordering::AcqRel) >= 2;
        let started = std::time::Instant::now();
        let summary = super::sync_nyt_history_with_base_url_cancellable(
            &paths, &config, today, &base_url, &cancelled,
        )
        .expect("retained archive");
        join.join().expect("server");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "must not wait the ten-second backoff"
        );
        assert!(summary.cancelled && summary.partial_sync && summary.retained_existing_archive);
        assert!(summary.coverage_complete);
        assert_eq!(
            (
                summary.attempted,
                summary.fetched,
                summary.applied,
                summary.retained
            ),
            (1, 0, 0, 1)
        );
        assert_eq!(summary.failed_dates, vec![today]);
        assert!(summary.failures[0].message.contains("cancelled"));
        assert!(summary.failures[0].message.contains("HTTP 500"));
        assert_eq!(
            fs::read(&paths.raw_history).expect("unchanged archive"),
            before
        );
    }

    #[test]
    fn cancelled_sync_starts_no_fetch_and_reports_retained_state() {
        let root = TestDirectory::new("pre-cancelled-sync");
        let paths = ProjectPaths::new(root.path());
        let today = NaiveDate::from_ymd_opt(2021, 6, 19).expect("today");
        write_history_jsonl(&paths.raw_history, &[make_test_entry(today, "cigar")])
            .expect("archive");
        let (base_url, join) = spawn_test_server(0, |_, _| panic!("cancelled sync must not fetch"));
        let summary = super::sync_nyt_history_with_base_url_cancellable(
            &paths,
            &PriorConfig::default(),
            today,
            &base_url,
            &|| true,
        )
        .expect("retained archive");
        join.join().expect("server");
        assert!(summary.cancelled && summary.partial_sync && summary.coverage_complete);
        assert_eq!(
            (
                summary.attempted,
                summary.fetched,
                summary.applied,
                summary.retained
            ),
            (0, 0, 0, 1)
        );
    }

    #[test]
    fn incremental_sync_reports_only_published_insertions_and_retained_records() {
        let root = TestDirectory::new("incremental-sync-counters");
        let paths = ProjectPaths::new(root.path());
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let second = first.succ_opt().expect("second");
        let today = second.succ_opt().expect("third");
        write_history_jsonl(
            &paths.raw_history,
            &[
                make_test_entry(first, "cigar"),
                make_test_entry(second, "rebut"),
            ],
        )
        .expect("archive");
        let (base_url, join) = spawn_test_server(2, move |path, _| {
            assert!(!path.ends_with("2021-06-19.json"));
            let date = if path.ends_with("2021-06-20.json") {
                second
            } else {
                today
            };
            (200, test_json_response(&make_test_entry(date, "rebut")))
        });
        let config = PriorConfig {
            sync_reverify_days: 1,
            sync_retry_attempts: 0,
            ..PriorConfig::default()
        };
        let summary = sync_nyt_history_with_base_url(&paths, &config, today, &base_url)
            .expect("incremental sync");
        join.join().expect("server");
        assert_eq!(
            (
                summary.attempted,
                summary.fetched,
                summary.reverified,
                summary.applied,
                summary.retained,
                summary.changed
            ),
            (2, 2, 1, 1, 2, 0)
        );
        assert!(
            summary.coverage_complete
                && !summary.partial_sync
                && !summary.retained_existing_archive
                && !summary.cancelled
        );
        assert_eq!(
            (summary.requested_first_date, summary.requested_last_date),
            (first, today)
        );
        assert!(summary.failures.is_empty() && summary.missing_dates.is_empty());
        assert_eq!(
            read_history_jsonl(&paths.raw_history)
                .expect("published")
                .len(),
            3
        );
    }

    #[test]
    fn all_failed_requests_retain_prior_archive_with_date_specific_causes() {
        let root = TestDirectory::new("all-failed-sync-retains");
        let paths = ProjectPaths::new(root.path());
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let last = first.succ_opt().expect("last");
        write_history_jsonl(&paths.raw_history, &[make_test_entry(first, "cigar")])
            .expect("archive");
        let before = fs::read(&paths.raw_history).expect("original");
        let (base_url, join) = spawn_test_server(2, |_, _| (500, String::new()));
        let config = PriorConfig {
            sync_retry_attempts: 0,
            ..PriorConfig::default()
        };
        let summary = sync_nyt_history_with_base_url(&paths, &config, last, &base_url)
            .expect("retained archive");
        join.join().expect("server");
        assert_eq!(
            (
                summary.attempted,
                summary.fetched,
                summary.applied,
                summary.retained
            ),
            (2, 0, 0, 1)
        );
        assert_eq!(summary.failed_dates, vec![first, last]);
        assert!(
            summary
                .failures
                .iter()
                .all(|failure| failure.message.contains("HTTP 500")
                    && failure.message.contains(&failure.date.to_string()))
        );
        assert!(summary.partial_sync && !summary.coverage_complete);
        assert_eq!(summary.missing_dates, vec![last]);
        assert_eq!(summary.last_successful_date, None);
        assert_eq!(fs::read(&paths.raw_history).expect("unchanged"), before);
    }

    #[test]
    fn unrepairable_discontinuous_prior_archive_is_not_replaced() {
        let root = TestDirectory::new("discontinuous-sync-prior");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let last = NaiveDate::from_ymd_opt(2021, 6, 21).expect("last");
        let original = format!(
            "{}\n{}\n",
            test_json_response(&make_test_entry(first, "cigar")),
            test_json_response(&make_test_entry(last, "rebut"))
        );
        fs::write(&paths.raw_history, &original).expect("gapped archive");
        let (base_url, join) = spawn_test_server(3, move |path, _| {
            if path.ends_with("2021-06-20.json") {
                (500, String::new())
            } else {
                let date = if path.ends_with("2021-06-19.json") {
                    first
                } else {
                    last
                };
                (200, test_json_response(&make_test_entry(date, "sissy")))
            }
        });
        let config = PriorConfig {
            sync_retry_attempts: 0,
            ..PriorConfig::default()
        };
        let error = sync_nyt_history_with_base_url(&paths, &config, last, &base_url)
            .expect_err("no usable prior archive");
        join.join().expect("server");
        assert!(
            error
                .to_string()
                .contains("no usable nonempty contiguous prior archive")
        );
        assert_eq!(
            fs::read_to_string(&paths.raw_history).expect("unchanged"),
            original
        );
    }

    #[test]
    fn internal_continuity_does_not_imply_requested_coverage() {
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let last = first.succ_opt().expect("last");
        let entries = [make_test_entry(last, "rebut")];
        super::validate_history_continuity(&entries).expect("internally contiguous");
        super::validate_history_coverage(&entries, first, last).expect_err("missing first date");
        super::validate_history_coverage(&[], first, last).expect_err("empty archive");
        super::validate_history_coverage(&entries, last, last).expect("covered range");
    }

    #[test]
    fn invalid_sync_range_or_reverify_window_is_rejected_before_fetch() {
        let launch = NaiveDate::from_ymd_opt(2021, 6, 19).expect("launch");
        for (today, days, message) in [
            (
                launch.pred_opt().expect("before launch"),
                1,
                "precedes Wordle launch",
            ),
            (launch, 0, "sync_reverify_days must be at least 1"),
            (launch, -1, "sync_reverify_days must be at least 1"),
        ] {
            let root = TestDirectory::new("invalid-sync-input");
            let paths = ProjectPaths::new(root.path());
            let (base_url, join) =
                spawn_test_server(0, |_, _| panic!("invalid sync must not fetch"));
            let config = PriorConfig {
                sync_reverify_days: days,
                ..PriorConfig::default()
            };
            let error = sync_nyt_history_with_base_url(&paths, &config, today, &base_url)
                .expect_err("invalid input");
            join.join().expect("server");
            assert!(error.to_string().contains(message));
            assert!(!paths.raw_history.exists());
        }
    }

    #[test]
    fn sync_nyt_history_errors_when_nothing_can_be_fetched() {
        let root = TestDirectory::new("no-success");
        let paths = ProjectPaths::new(root.path());
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
        assert!(message.contains("2021-06-19: HTTP 500"));
        assert!(message.contains("attempted=1 fetched=0 applied=0"));
        assert!(
            super::read_history_jsonl(&paths.raw_history)
                .expect("read history")
                .is_empty()
        );
    }
}
