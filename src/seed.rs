use std::{collections::BTreeSet, io::Write, path::Path};

use anyhow::{Context, Result, bail};
use csv::Writer;
use serde::Serialize;

use crate::{
    atomic_file::{acquire_edit_lock, atomic_write},
    data::{ProjectPaths, read_word_list, validate_answer_universe},
};

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
    let _edit = acquire_edit_lock(&paths.seed_answers)?;
    let guesses = read_word_list(&paths.seed_guesses)?;
    validate_answer_universe(&guesses, [normalized.as_str()])?;

    let mut words = if paths.manual_additions.exists() {
        read_word_list(&paths.manual_additions)?
    } else {
        Vec::new()
    };
    words.push(normalized);
    words.sort_unstable();
    words.dedup();

    validate_answer_universe(&guesses, words.iter().map(String::as_str))?;
    let mut contents = Vec::new();
    writeln!(contents, "# One lowercase five-letter word per line.")
        .context("failed to write header")?;
    writeln!(
        contents,
        "# Words listed here are added to the modeled answer universe even if they are not in the pinned seed lists."
    )
    .context("failed to write header")?;
    for word in words {
        writeln!(contents, "{word}").context("failed to write manual addition")?;
    }

    atomic_write(&paths.manual_additions, &contents)
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
    let _edit = acquire_edit_lock(&paths.seed_answers)?;
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
    let guesses = read_word_list(&paths.seed_guesses)?;
    validate_answer_universe(&guesses, merged.iter().map(String::as_str))?;

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
    let mut writer = Writer::from_writer(Vec::new());
    for row in rows {
        writer.serialize(row).context("failed to write csv row")?;
    }
    writer.flush().context("failed to flush csv writer")?;
    let contents = writer
        .into_inner()
        .context("failed to finish reconciliation CSV")?;
    atomic_write(path, &contents)
}

fn write_word_list(path: &Path, words: &[String]) -> Result<()> {
    let mut contents = Vec::new();
    for word in words {
        writeln!(contents, "{word}").context("failed to write merged word list")?;
    }
    atomic_write(path, &contents)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::data::ProjectPaths;

    use super::{MergeStrategy, add_manual_addition, merge_seed_lists};

    #[test]
    fn manual_addition_rejects_unguessable_word_before_writing() {
        let root = crate::test_support::TestDirectory::new("manual-unguessable");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        fs::write(&paths.seed_guesses, "cigar\n").expect("guesses");
        fs::write(&paths.manual_additions, "# preserved\n").expect("manual");
        let error = add_manual_addition(&paths, "rebut").expect_err("unguessable");
        assert!(format!("{error:#}").contains("guess"));
        assert_eq!(
            fs::read_to_string(&paths.manual_additions).expect("manual"),
            "# preserved\n"
        );
    }

    #[test]
    fn authoritative_seed_edits_survive_pre_replace_interruptions() {
        use crate::atomic_file::{AtomicWriteStage, test_hooks::with_failure};
        let root = crate::test_support::TestDirectory::new("seed-interruption");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        fs::write(&paths.seed_guesses, "cigar\nrebut\n").expect("guesses");
        fs::write(&paths.seed_answers, "cigar\n").expect("primary");
        fs::write(&paths.seed_reference_answers, "rebut\n").expect("reference");
        fs::write(&paths.manual_additions, "cigar\n").expect("manual");
        for stage in [
            AtomicWriteStage::TempCreated,
            AtomicWriteStage::DataWritten,
            AtomicWriteStage::BeforeReplace,
        ] {
            assert!(with_failure(stage, || add_manual_addition(&paths, "rebut")).is_err());
            assert_eq!(
                fs::read_to_string(&paths.manual_additions).expect("manual"),
                "cigar\n"
            );
            assert!(
                with_failure(stage, || merge_seed_lists(
                    &paths,
                    MergeStrategy::Union,
                    true
                ))
                .is_err()
            );
            assert_eq!(
                fs::read_to_string(&paths.seed_answers).expect("primary"),
                "cigar\n"
            );
        }
    }

    #[test]
    fn overlapping_seed_edits_are_rejected_without_losing_previous_additions() {
        let root = crate::test_support::TestDirectory::new("seed-overlap");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        fs::write(&paths.seed_guesses, "cigar\nrebut\n").expect("guesses");
        fs::write(&paths.manual_additions, "cigar\n").expect("manual");
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(
                paths
                    .seed_answers
                    .with_file_name(".candidate_answers.txt.mwedit.lock"),
            )
            .expect("lock file");
        lock.try_lock().expect("hold edit lock");
        let result = std::thread::scope(|scope| {
            scope
                .spawn(|| add_manual_addition(&paths, "rebut"))
                .join()
                .expect("edit thread")
        });
        assert!(result.is_err(), "overlapping edit must report busy");
        assert_eq!(
            fs::read_to_string(&paths.manual_additions).expect("manual"),
            "cigar\n"
        );
        drop(lock);
        add_manual_addition(&paths, "rebut").expect("retry after unlock");
        assert_eq!(
            crate::data::read_word_list(&paths.manual_additions).expect("both additions"),
            ["cigar", "rebut"]
        );
    }

    #[test]
    fn add_manual_addition_deduplicates_words() {
        let root = crate::test_support::TestDirectory::new("seed-deduplicate");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");

        fs::write(&paths.seed_guesses, "cigar\n").expect("guesses");

        add_manual_addition(&paths, "cigar").expect("write");
        add_manual_addition(&paths, "cigar").expect("dedupe");
        let contents = fs::read_to_string(&paths.manual_additions).expect("file");
        assert_eq!(contents.matches("cigar").count(), 1);
    }

    #[test]
    fn union_merge_writes_reference_words() {
        let root = crate::test_support::TestDirectory::new("seed-merge");
        let paths = ProjectPaths::new(root.path());
        paths.ensure_layout().expect("layout");
        fs::write(&paths.seed_guesses, "cigar\nrebut\n").expect("guesses");
        fs::write(&paths.seed_answers, "cigar\n").expect("primary");
        fs::write(&paths.seed_reference_answers, "cigar\nrebut\n").expect("reference");

        let summary = merge_seed_lists(&paths, MergeStrategy::Union, false).expect("merge");
        let merged = fs::read_to_string(&paths.merged_seed_answers).expect("merged");

        assert_eq!(summary.merged_count, 2);
        assert!(merged.contains("rebut"));
    }
}
