use std::{
    fs::{self, File},
    io::{Read, Write},
};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{
    model::AnswerRecord,
    scoring::{PATTERN_SPACE, score_guess},
};

const MAGIC: &[u8; 8] = b"MWORDPT1";
const HEADER_SIZE: usize = 8 + 4 + 4 + 8 + 8;

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
        let guess_hash = u64::from_le_bytes(bytes[16..24].try_into().expect("slice length"));
        let answer_hash = u64::from_le_bytes(bytes[24..32].try_into().expect("slice length"));

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

        Ok(Some(Self {
            guess_count,
            answer_count,
            data,
        }))
    }

    fn persist(&self, path: &std::path::Path, guesses: &[String], answers: &[&str]) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let mut bytes = Vec::with_capacity(HEADER_SIZE + self.data.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(self.guess_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.answer_count as u32).to_le_bytes());
        bytes.extend_from_slice(&hash_word_list(guesses.iter().map(String::as_str)).to_le_bytes());
        bytes.extend_from_slice(&hash_word_list(answers.iter().copied()).to_le_bytes());
        bytes.extend_from_slice(&self.data);

        let mut file =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }
}

pub fn hash_word_list<'a>(words: impl IntoIterator<Item = &'a str>) -> u64 {
    let mut hash = 1469598103934665603u64;
    for word in words {
        hash = hash_bytes(hash, word.as_bytes());
        hash = hash_bytes(hash, b"\n");
    }
    hash
}

pub fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}
