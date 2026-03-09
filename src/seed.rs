use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::Write,
    path::Path,
};

use anyhow::{Context, Result, bail};
use csv::Writer;
use serde::Serialize;

use crate::data::{ProjectPaths, read_word_list};

#[derive(Clone, Debug, Serialize)]
pub struct SeedReconciliationRow {
    pub word: String,
    pub in_primary: bool,
    pub in_reference: bool,
}

#[derive(Clone, Debug)]
pub struct SeedReconciliationSummary {
    pub primary_count: usize,
    pub reference_count: usize,
    pub shared_count: usize,
    pub primary_only_count: usize,
    pub reference_only_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeStrategy {
    KeepPrimary,
    Union,
}

impl MergeStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::KeepPrimary => "keep_primary",
            Self::Union => "union",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SeedMergeSummary {
    pub strategy: MergeStrategy,
    pub merged_count: usize,
    pub primary_count: usize,
    pub reference_count: usize,
    pub output_path: String,
    pub applied_to_primary: bool,
}

pub fn add_manual_addition(paths: &ProjectPaths, word: &str) -> Result<()> {
    let normalized = word.trim().to_ascii_lowercase();
    if normalized.len() != 5 || !normalized.bytes().all(|byte| byte.is_ascii_lowercase()) {
        bail!("manual additions must be lowercase five-letter words");
    }

    let mut words = if paths.manual_additions.exists() {
        read_word_list(&paths.manual_additions)?
    } else {
        Vec::new()
    };
    words.push(normalized);
    words.sort_unstable();
    words.dedup();

    if let Some(parent) = paths.manual_additions.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut file = File::create(&paths.manual_additions)
        .with_context(|| format!("failed to create {}", paths.manual_additions.display()))?;
    writeln!(file, "# One lowercase five-letter word per line.")
        .context("failed to write header")?;
    writeln!(
        file,
        "# Words listed here are added to the modeled answer universe even if they are not in the pinned seed lists."
    )
    .context("failed to write header")?;
    for word in words {
        writeln!(file, "{word}").context("failed to write manual addition")?;
    }

    Ok(())
}

pub fn reconcile_seed_lists(paths: &ProjectPaths) -> Result<SeedReconciliationSummary> {
    let primary = read_word_list(&paths.seed_answers)
        .with_context(|| format!("failed to load {}", paths.seed_answers.display()))?;
    let reference = read_word_list(&paths.seed_reference_answers)
        .with_context(|| format!("failed to load {}", paths.seed_reference_answers.display()))?;

    let primary_set = primary.iter().cloned().collect::<BTreeSet<_>>();
    let reference_set = reference.iter().cloned().collect::<BTreeSet<_>>();
    let words = primary_set
        .union(&reference_set)
        .cloned()
        .collect::<Vec<_>>();

    let rows = words
        .iter()
        .map(|word| SeedReconciliationRow {
            word: word.clone(),
            in_primary: primary_set.contains(word),
            in_reference: reference_set.contains(word),
        })
        .collect::<Vec<_>>();

    write_csv(&paths.derived_seed_reconciliation, &rows)?;

    let shared_count = primary_set.intersection(&reference_set).count();
    Ok(SeedReconciliationSummary {
        primary_count: primary_set.len(),
        reference_count: reference_set.len(),
        shared_count,
        primary_only_count: primary_set.len() - shared_count,
        reference_only_count: reference_set.len() - shared_count,
    })
}

pub fn merge_seed_lists(
    paths: &ProjectPaths,
    strategy: MergeStrategy,
    apply_to_primary: bool,
) -> Result<SeedMergeSummary> {
    let primary = read_word_list(&paths.seed_answers)
        .with_context(|| format!("failed to load {}", paths.seed_answers.display()))?;
    let reference = read_word_list(&paths.seed_reference_answers)
        .with_context(|| format!("failed to load {}", paths.seed_reference_answers.display()))?;

    let merged = match strategy {
        MergeStrategy::KeepPrimary => primary.clone(),
        MergeStrategy::Union => {
            let mut merged = primary.clone();
            merged.extend(reference.iter().cloned());
            merged.sort_unstable();
            merged.dedup();
            merged
        }
    };

    let output_path = if apply_to_primary {
        paths.seed_answers.clone()
    } else {
        paths.merged_seed_answers.clone()
    };
    write_word_list(&output_path, &merged)?;

    Ok(SeedMergeSummary {
        strategy,
        merged_count: merged.len(),
        primary_count: primary.len(),
        reference_count: reference.len(),
        output_path: output_path.display().to_string(),
        applied_to_primary: apply_to_primary,
    })
}

fn write_csv(path: &Path, rows: &[SeedReconciliationRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = Writer::from_writer(file);
    for row in rows {
        writer.serialize(row).context("failed to write csv row")?;
    }
    writer.flush().context("failed to flush csv writer")?;
    Ok(())
}

fn write_word_list(path: &Path, words: &[String]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    for word in words {
        writeln!(file, "{word}").context("failed to write merged word list")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::data::ProjectPaths;

    use super::{MergeStrategy, add_manual_addition, merge_seed_lists};

    #[test]
    fn add_manual_addition_deduplicates_words() {
        let root = std::env::temp_dir().join("maybe-wordle-seed-test");
        let _ = fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");

        add_manual_addition(&paths, "cigar").expect("write");
        add_manual_addition(&paths, "cigar").expect("dedupe");
        let contents = fs::read_to_string(&paths.manual_additions).expect("file");
        assert_eq!(contents.matches("cigar").count(), 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn union_merge_writes_reference_words() {
        let root = std::env::temp_dir().join("maybe-wordle-seed-merge-test");
        let _ = fs::remove_dir_all(&root);
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        fs::write(&paths.seed_answers, "cigar\n").expect("primary");
        fs::write(&paths.seed_reference_answers, "cigar\nrebut\n").expect("reference");

        let summary = merge_seed_lists(&paths, MergeStrategy::Union, false).expect("merge");
        let merged = fs::read_to_string(&paths.merged_seed_answers).expect("merged");

        assert_eq!(summary.merged_count, 2);
        assert!(merged.contains("rebut"));
        let _ = fs::remove_dir_all(&root);
    }
}
