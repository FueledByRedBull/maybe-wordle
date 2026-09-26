use std::{fs::File, io::Read};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{
    atomic_file::atomic_write,
    identity::{CanonicalSha256, digest_bytes},
    model::AnswerRecord,
    scoring::{PATTERN_SPACE, score_guess},
};

const MAGIC: &[u8; 8] = b"MWORDPT3";
const PAYLOAD_DIGEST_DOMAIN: &str = "maybe-wordle-pattern-table-payload-v3";
const HEADER_SIZE: usize = 8 + 4 + 4 + 32 + 32 + 32;

#[derive(Clone, Debug)]
pub struct PatternTable {
    guess_count: usize,
    answer_count: usize,
    data: Vec<u8>,
}

impl PatternTable {
    pub fn load_or_build(
        paths: &crate::data::ProjectPaths,
        guesses: &[String],
        answers: &[AnswerRecord],
    ) -> Result<Self> {
        Self::load_or_build_at(&paths.pattern_table, guesses, answers)
    }

    pub fn load_or_build_at(
        path: &std::path::Path,
        guesses: &[String],
        answers: &[AnswerRecord],
    ) -> Result<Self> {
        if let Some(existing) = Self::try_load(path, guesses, answers)? {
            return Ok(existing);
        }

        let answer_words = answers
            .iter()
            .map(|answer| answer.word.as_str())
            .collect::<Vec<_>>();
        let rows = guesses
            .par_iter()
            .map(|guess| {
                answer_words
                    .iter()
                    .map(|answer| score_guess(guess, answer))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let data = rows.into_iter().flatten().collect::<Vec<_>>();

        let table = Self {
            guess_count: guesses.len(),
            answer_count: answer_words.len(),
            data,
        };
        table.persist(path, guesses, &answer_words)?;
        Ok(table)
    }

    pub fn get(&self, guess_index: usize, answer_index: usize) -> u8 {
        self.data[(guess_index * self.answer_count) + answer_index]
    }

    pub fn bytes_len(&self) -> usize {
        self.data.len()
    }

    fn try_load(
        path: &std::path::Path,
        guesses: &[String],
        answers: &[AnswerRecord],
    ) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }

        let mut bytes = Vec::new();
        File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read {}", path.display()))?;

        if bytes.len() < HEADER_SIZE || &bytes[..MAGIC.len()] != MAGIC {
            return Ok(None);
        }

        let guess_count =
            u32::from_le_bytes(bytes[8..12].try_into().expect("slice length")) as usize;
        let answer_count =
            u32::from_le_bytes(bytes[12..16].try_into().expect("slice length")) as usize;
        let guess_hash: [u8; 32] = bytes[16..48].try_into().expect("slice length");
        let answer_hash: [u8; 32] = bytes[48..80].try_into().expect("slice length");
        let payload_hash: [u8; 32] = bytes[80..112].try_into().expect("slice length");

        if guess_count != guesses.len()
            || answer_count != answers.len()
            || guess_hash != hash_word_list(guesses.iter().map(String::as_str))
            || answer_hash != hash_word_list(answers.iter().map(|answer| answer.word.as_str()))
        {
            return Ok(None);
        }

        let data = bytes[HEADER_SIZE..].to_vec();
        if data.len() != guess_count * answer_count {
            return Ok(None);
        }
        if data.iter().any(|value| *value as usize >= PATTERN_SPACE) {
            bail!("pattern table contains invalid pattern values");
        }
        if payload_hash != digest_bytes(PAYLOAD_DIGEST_DOMAIN, &data) {
            return Ok(None);
        }

        Ok(Some(Self {
            guess_count,
            answer_count,
            data,
        }))
    }

    fn persist(&self, path: &std::path::Path, guesses: &[String], answers: &[&str]) -> Result<()> {
        let mut bytes = Vec::with_capacity(HEADER_SIZE + self.data.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(self.guess_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.answer_count as u32).to_le_bytes());
        bytes.extend_from_slice(&hash_word_list(guesses.iter().map(String::as_str)));
        bytes.extend_from_slice(&hash_word_list(answers.iter().copied()));
        bytes.extend_from_slice(&digest_bytes(PAYLOAD_DIGEST_DOMAIN, &self.data));
        bytes.extend_from_slice(&self.data);

        atomic_write(path, &bytes)
    }
}

pub fn hash_word_list<'a>(words: impl IntoIterator<Item = &'a str>) -> [u8; 32] {
    let mut hash = CanonicalSha256::new("maybe-wordle-word-list-v2");
    for word in words {
        hash.field(word.as_bytes());
    }
    hash.finish()
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn test_inputs() -> (Vec<String>, Vec<AnswerRecord>) {
        let guesses = ["cigar", "rebut"].into_iter().map(str::to_string).collect();
        let answers = ["sissy", "humph"]
            .into_iter()
            .map(|word| AnswerRecord {
                word: word.to_string(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect();
        (guesses, answers)
    }

    fn test_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "maybe-wordle-pattern-table-{label}-{}-{unique}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn in_range_payload_corruption_rebuilds_cache() {
        let path = test_path("payload");
        let (guesses, answers) = test_inputs();
        let original =
            PatternTable::load_or_build_at(&path, &guesses, &answers).expect("build pattern table");
        let mut bytes = fs::read(&path).expect("read pattern table");
        let payload = &mut bytes[HEADER_SIZE..];
        payload[0] = (payload[0] + 1) % PATTERN_SPACE as u8;
        assert!((payload[0] as usize) < PATTERN_SPACE);
        fs::write(&path, bytes).expect("corrupt pattern table");

        assert!(
            PatternTable::try_load(&path, &guesses, &answers)
                .expect("inspect corrupted pattern table")
                .is_none()
        );
        let rebuilt = PatternTable::load_or_build_at(&path, &guesses, &answers)
            .expect("rebuild pattern table");
        assert_eq!(rebuilt.data, original.data);
        assert!(
            PatternTable::try_load(&path, &guesses, &answers)
                .expect("inspect rebuilt pattern table")
                .is_some()
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn legacy_pattern_table_header_is_rejected() {
        let path = test_path("legacy");
        let (guesses, answers) = test_inputs();
        let original =
            PatternTable::load_or_build_at(&path, &guesses, &answers).expect("build pattern table");
        let mut bytes = fs::read(&path).expect("read pattern table");
        bytes[..MAGIC.len()].copy_from_slice(b"MWORDPT2");
        fs::write(&path, bytes).expect("write legacy header");

        assert!(
            PatternTable::try_load(&path, &guesses, &answers)
                .expect("inspect legacy pattern table")
                .is_none()
        );
        let rebuilt = PatternTable::load_or_build_at(&path, &guesses, &answers)
            .expect("rebuild legacy pattern table");
        assert_eq!(rebuilt.data, original.data);
        let _ = fs::remove_file(path);
    }
}
