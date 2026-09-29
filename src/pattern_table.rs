use std::{
    fs::File,
    io::{Read, Write},
};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{
    atomic_file::atomic_write_with,
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
        let length = guesses
            .len()
            .checked_mul(answer_words.len())
            .context("pattern table size overflow")?;
        let mut data = vec![0; length];
        if !answer_words.is_empty() {
            data.par_chunks_mut(answer_words.len())
                .zip(guesses.par_iter())
                .for_each(|(row, guess)| {
                    for (value, answer) in row.iter_mut().zip(&answer_words) {
                        *value = score_guess(guess, answer);
                    }
                });
        }

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

    /// Read an immutable artifact without repairing or replacing it on failure.
    pub fn load_existing_at(
        path: &std::path::Path,
        guesses: &[String],
        answers: &[AnswerRecord],
    ) -> Result<Self> {
        Self::try_load(path, guesses, answers)?
            .with_context(|| format!("missing, corrupt or stale pattern table {}", path.display()))
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

        let mut file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let file_length = file
            .metadata()
            .with_context(|| format!("failed to inspect {}", path.display()))?
            .len();
        if file_length < HEADER_SIZE as u64 {
            return Ok(None);
        }
        let mut bytes = [0; HEADER_SIZE];
        file.read_exact(&mut bytes)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if &bytes[..MAGIC.len()] != MAGIC {
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

        let Some(length) = guess_count.checked_mul(answer_count) else {
            return Ok(None);
        };
        if file_length != (HEADER_SIZE as u64).saturating_add(length as u64) {
            return Ok(None);
        }
        let mut data = vec![0; length];
        file.read_exact(&mut data)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if file
            .read(&mut [0])
            .with_context(|| format!("failed to read {}", path.display()))?
            != 0
        {
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
        atomic_write_with(path, |file| {
            file.write_all(MAGIC)?;
            file.write_all(&(self.guess_count as u32).to_le_bytes())?;
            file.write_all(&(self.answer_count as u32).to_le_bytes())?;
            file.write_all(&hash_word_list(guesses.iter().map(String::as_str)))?;
            file.write_all(&hash_word_list(answers.iter().copied()))?;
            file.write_all(&digest_bytes(PAYLOAD_DIGEST_DOMAIN, &self.data))?;
            file.write_all(&self.data)?;
            Ok(())
        })
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
        PathBuf::from("target/audit-work/pattern-tests").join(format!(
            "maybe-wordle-pattern-table-{label}-{}-{unique}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn published_tables_are_never_rebuilt_by_read_only_loading() {
        let directory = crate::test_support::TestDirectory::new("published-pattern");
        let path = directory.path().join("patterns.bin");
        let (guesses, answers) = test_inputs();
        assert!(PatternTable::load_existing_at(&path, &guesses, &answers).is_err());
        assert!(!path.exists());
        PatternTable::load_or_build_at(&path, &guesses, &answers).unwrap();
        let original = fs::read(&path).unwrap();
        assert!(PatternTable::load_existing_at(&path, &guesses, &answers).is_ok());
        let mut corrupted = original.clone();
        corrupted[HEADER_SIZE] = (corrupted[HEADER_SIZE] + 1) % 243;
        fs::write(&path, &corrupted).unwrap();
        assert!(PatternTable::load_existing_at(&path, &guesses, &answers).is_err());
        assert_eq!(fs::read(&path).unwrap(), corrupted);
        fs::write(&path, &original).unwrap();
        assert!(PatternTable::load_existing_at(&path, &guesses[1..], &answers).is_err());
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn flat_table_matches_direct_scoring_and_preserves_serialized_payload() {
        let path = test_path("direct");
        let words = [
            "lilly", "alley", "added", "dread", "eerie", "sissy", "humph",
        ];
        let guesses: Vec<_> = words.iter().map(|word| word.to_string()).collect();
        let answers: Vec<_> = words
            .iter()
            .rev()
            .map(|word| AnswerRecord {
                word: word.to_string(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect();
        let table = PatternTable::load_or_build_at(&path, &guesses, &answers).expect("build");
        let loaded = PatternTable::try_load(&path, &guesses, &answers)
            .expect("load")
            .expect("valid table");
        let bytes = fs::read(&path).expect("persisted bytes");
        assert_eq!(&bytes[..8], b"MWORDPT3");
        assert_eq!(&bytes[8..16], &[7, 0, 0, 0, 7, 0, 0, 0]);
        assert_eq!(bytes.len(), HEADER_SIZE + words.len() * words.len());
        for (guess_index, guess) in guesses.iter().enumerate() {
            for (answer_index, answer) in answers.iter().enumerate() {
                let expected = score_guess(guess, &answer.word);
                assert_eq!(table.get(guess_index, answer_index), expected);
                assert_eq!(loaded.get(guess_index, answer_index), expected);
                assert_eq!(
                    bytes[HEADER_SIZE + guess_index * answers.len() + answer_index],
                    expected
                );
            }
        }
        fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn sampled_bundled_words_match_direct_scoring() {
        let path = test_path("bundled-sample");
        let guesses: Vec<_> = include_str!("../data/seed/valid_guesses.txt")
            .lines()
            .step_by(97)
            .take(64)
            .map(str::to_string)
            .collect();
        let answers: Vec<_> = include_str!("../data/seed/candidate_answers.txt")
            .lines()
            .step_by(37)
            .take(32)
            .map(|word| AnswerRecord {
                word: word.to_string(),
                in_seed: true,
                manual_entry: false,
                manual_weight: 1.0,
                history_dates: Vec::new(),
            })
            .collect();
        assert_eq!((guesses.len(), answers.len()), (64, 32));
        let table =
            PatternTable::load_or_build_at(&path, &guesses, &answers).expect("sampled build");
        for (g, guess) in guesses.iter().enumerate() {
            for (a, answer) in answers.iter().enumerate() {
                assert_eq!(table.get(g, a), score_guess(guess, &answer.word));
            }
        }
        fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn truncated_extended_and_wrong_identity_tables_are_rejected() {
        let path = test_path("invalid-length-or-identity");
        let (guesses, answers) = test_inputs();
        PatternTable::load_or_build_at(&path, &guesses, &answers).expect("build");
        let original = fs::read(&path).expect("bytes");
        for length in [0, 1, 7, HEADER_SIZE - 1, HEADER_SIZE, original.len() - 1] {
            fs::write(&path, &original[..length]).expect("truncated fixture");
            assert!(
                PatternTable::try_load(&path, &guesses, &answers)
                    .expect("reject truncation")
                    .is_none()
            );
        }
        let mut extended = original.clone();
        extended.push(0);
        fs::write(&path, extended).expect("extended fixture");
        assert!(
            PatternTable::try_load(&path, &guesses, &answers)
                .expect("reject trailing data")
                .is_none()
        );
        for offset in [8, 12, 16, 48, 80] {
            let mut corrupted = original.clone();
            corrupted[offset] ^= 1;
            fs::write(&path, corrupted).expect("identity fixture");
            assert!(
                PatternTable::try_load(&path, &guesses, &answers)
                    .expect("reject identity")
                    .is_none()
            );
        }
        let mut invalid_pattern = original;
        invalid_pattern[HEADER_SIZE] = 243;
        fs::write(&path, invalid_pattern).expect("invalid pattern");
        assert!(
            PatternTable::try_load(&path, &guesses, &answers)
                .expect_err("invalid pattern is an error")
                .to_string()
                .contains("invalid pattern")
        );
        fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn empty_dimensions_remain_valid_zero_byte_tables() {
        let (guesses, answers) = test_inputs();
        for (label, guesses, answers) in [
            ("empty-guesses", Vec::new(), answers),
            ("empty-answers", guesses, Vec::new()),
        ] {
            let path = test_path(label);
            let table =
                PatternTable::load_or_build_at(&path, &guesses, &answers).expect("empty table");
            assert_eq!(table.bytes_len(), 0);
            assert_eq!(
                PatternTable::try_load(&path, &guesses, &answers)
                    .expect("reload")
                    .expect("valid empty table")
                    .bytes_len(),
                0
            );
            fs::remove_file(path).expect("cleanup");
        }
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
