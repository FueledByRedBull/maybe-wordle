use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

const TEMP_FORMAT: &str = "mwatomic-v1";
const STALE_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Durably writes bytes to a sibling temporary file and atomically replaces `path`.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_with_hook(path, bytes, |_| Ok(()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AtomicWriteStage {
    TempCreated,
    DataWritten,
    TempSynced,
    BeforeReplace,
    BeforePlatformReplace,
    AfterPlatformReplace,
    #[cfg(unix)]
    BeforeParentDirectorySync,
    #[cfg(unix)]
    AfterParentDirectorySync,
}

fn atomic_write_with_hook(
    path: &Path,
    bytes: &[u8],
    mut stage_hook: impl FnMut(AtomicWriteStage) -> Result<()>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    cleanup_stale_atomic_temps(path, STALE_TEMP_AGE)?;

    let temp = sibling_temp_path(path);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .with_context(|| format!("failed to create {}", temp.display()))?;
        stage_hook(AtomicWriteStage::TempCreated)?;
        file.write_all(bytes)
            .with_context(|| format!("failed to write {}", temp.display()))?;
        stage_hook(AtomicWriteStage::DataWritten)?;
        file.flush()
            .with_context(|| format!("failed to flush {}", temp.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temp.display()))?;
        stage_hook(AtomicWriteStage::TempSynced)?;
        drop(file);
        stage_hook(AtomicWriteStage::BeforeReplace)?;
        replace_file_with_hook(&temp, path, &mut stage_hook)
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn sibling_temp_path(path: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(
        ".{name}.{TEMP_FORMAT}.{}.{}.tmp",
        std::process::id(),
        nonce
    ))
}

fn cleanup_stale_atomic_temps(path: &Path, minimum_age: Duration) -> Result<usize> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.exists() {
        return Ok(0);
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact");
    let prefix = format!(".{name}.{TEMP_FORMAT}.");
    let now = SystemTime::now();
    let mut removed = 0usize;
    for entry in fs::read_dir(parent).with_context(|| {
        format!(
            "failed to inspect atomic temporaries in {}",
            parent.display()
        )
    })? {
        let entry =
            entry.with_context(|| format!("failed to inspect entry in {}", parent.display()))?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if !owned_temp_name(file_name, &prefix) {
            continue;
        }
        let metadata = entry.metadata().with_context(|| {
            format!(
                "failed to inspect stale temporary {}",
                entry.path().display()
            )
        })?;
        if !metadata.is_file()
            || now
                .duration_since(metadata.modified().with_context(|| {
                    format!(
                        "failed to inspect modification time for {}",
                        entry.path().display()
                    )
                })?)
                .ok()
                .is_none_or(|age| age < minimum_age)
        {
            continue;
        }
        match fs::remove_file(entry.path()) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to remove stale temporary {}",
                        entry.path().display()
                    )
                });
            }
        }
    }
    Ok(removed)
}

fn owned_temp_name(file_name: &str, prefix: &str) -> bool {
    let Some(body) = file_name
        .strip_prefix(prefix)
        .and_then(|value| value.strip_suffix(".tmp"))
    else {
        return false;
    };
    let mut parts = body.split('.');
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(process), Some(nonce), None)
            if !process.is_empty()
                && !nonce.is_empty()
                && process.bytes().all(|byte| byte.is_ascii_digit())
                && nonce.bytes().all(|byte| byte.is_ascii_digit())
    )
}

#[cfg(unix)]
fn replace_file_with_hook(
    source: &Path,
    destination: &Path,
    stage_hook: &mut impl FnMut(AtomicWriteStage) -> Result<()>,
) -> Result<()> {
    stage_hook(AtomicWriteStage::BeforePlatformReplace)?;
    fs::rename(source, destination).with_context(|| {
        format!(
            "failed to atomically replace {} with {}",
            destination.display(),
            source.display()
        )
    })?;
    stage_hook(AtomicWriteStage::AfterPlatformReplace)?;
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    stage_hook(AtomicWriteStage::BeforeParentDirectorySync)?;
    fs::File::open(parent)
        .with_context(|| format!("failed to open parent directory {}", parent.display()))?
        .sync_all()
        .with_context(|| format!("failed to sync parent directory {}", parent.display()))?;
    stage_hook(AtomicWriteStage::AfterParentDirectorySync)
}

#[cfg(all(not(windows), not(unix)))]
fn replace_file_with_hook(
    source: &Path,
    destination: &Path,
    stage_hook: &mut impl FnMut(AtomicWriteStage) -> Result<()>,
) -> Result<()> {
    stage_hook(AtomicWriteStage::BeforePlatformReplace)?;
    fs::rename(source, destination).with_context(|| {
        format!(
            "failed to atomically replace {} with {}",
            destination.display(),
            source.display()
        )
    })?;
    stage_hook(AtomicWriteStage::AfterPlatformReplace)
}

#[cfg(windows)]
fn replace_file_with_hook(
    source: &Path,
    destination: &Path,
    stage_hook: &mut impl FnMut(AtomicWriteStage) -> Result<()>,
) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    let extended_path = |path: &Path| -> Result<Vec<u16>> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .context("failed to resolve current directory for atomic replacement")?
                .join(path)
        };
        let mut raw = absolute.as_os_str().encode_wide().collect::<Vec<_>>();
        for unit in &mut raw {
            if *unit == b'/' as u16 {
                *unit = b'\\' as u16;
            }
        }
        const VERBATIM: &[u16] = &[b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16];
        let mut extended = if raw.starts_with(VERBATIM) {
            raw
        } else if raw.starts_with(&[b'\\' as u16, b'\\' as u16]) {
            VERBATIM
                .iter()
                .copied()
                .chain("UNC\\".encode_utf16())
                .chain(raw.into_iter().skip(2))
                .collect()
        } else {
            VERBATIM.iter().copied().chain(raw).collect()
        };
        extended.push(0);
        Ok(extended)
    };
    let source_wide = extended_path(source)?;
    let destination_wide = extended_path(destination)?;
    stage_hook(AtomicWriteStage::BeforePlatformReplace)?;
    // SAFETY: both pointers reference NUL-terminated buffers for the duration of the call.
    let replaced = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "failed to atomically replace {} with {}",
                destination.display(),
                source.display()
            )
        });
    }
    stage_hook(AtomicWriteStage::AfterPlatformReplace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_replacement_leaves_no_temporary_file() {
        let root = std::env::temp_dir().join("maybe-wordle-atomic-write");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("root");
        let path = root.join("artifact.json");
        fs::write(&path, b"old").expect("seed");

        atomic_write(&path, b"new").expect("replace");
        assert_eq!(fs::read(&path).expect("read"), b"new");
        assert!(!fs::read_dir(&root).expect("dir").any(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn interruption_before_replace_preserves_existing_file() {
        let root = std::env::temp_dir().join("maybe-wordle-atomic-interruption");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("root");
        let path = root.join("artifact.json");
        fs::write(&path, b"valid-old").expect("seed");
        let temp = sibling_temp_path(&path);
        fs::write(&temp, b"partial-new").expect("interrupted temp write");

        assert_eq!(fs::read(&path).expect("read old"), b"valid-old");
        fs::remove_file(temp).expect("cleanup temp");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn every_injected_pre_replace_failure_preserves_old_file_and_cleans_temp() {
        for stage in [
            AtomicWriteStage::TempCreated,
            AtomicWriteStage::DataWritten,
            AtomicWriteStage::TempSynced,
            AtomicWriteStage::BeforeReplace,
        ] {
            let root = std::env::temp_dir()
                .join(format!("maybe-wordle-atomic-failure-{}", stage_name(stage)));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("root");
            let path = root.join("artifact.json");
            fs::write(&path, b"valid-old").expect("seed");

            let error = atomic_write_with_hook(&path, b"new", |current| {
                if current == stage {
                    anyhow::bail!("injected failure at {}", stage_name(stage));
                }
                Ok(())
            })
            .expect_err("injected failure");
            assert!(error.to_string().contains("injected failure"));
            assert_eq!(fs::read(&path).expect("read old"), b"valid-old");
            assert!(!fs::read_dir(&root).expect("dir").any(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".tmp")
            }));
            let _ = fs::remove_dir_all(root);
        }
    }

    #[cfg(any(windows, unix))]
    #[test]
    fn platform_replace_injections_distinguish_pre_and_post_replace_state() {
        for (stage, expected) in [
            (
                AtomicWriteStage::BeforePlatformReplace,
                b"valid-old".as_slice(),
            ),
            (AtomicWriteStage::AfterPlatformReplace, b"new".as_slice()),
        ] {
            let root = std::env::temp_dir().join(format!(
                "maybe-wordle-platform-replace-failure-{}",
                stage_name(stage)
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("root");
            let path = root.join("artifact.json");
            fs::write(&path, b"valid-old").expect("seed");

            let error = atomic_write_with_hook(&path, b"new", |current| {
                if current == stage {
                    anyhow::bail!("injected failure at {}", stage_name(stage));
                }
                Ok(())
            })
            .expect_err("injected failure");
            assert!(error.to_string().contains("injected failure"));
            assert_eq!(fs::read(&path).expect("read artifact"), expected);
            assert!(!fs::read_dir(&root).expect("dir").any(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".tmp")
            }));
            let _ = fs::remove_dir_all(root);
        }
    }

    #[cfg(unix)]
    #[test]
    fn unix_directory_sync_injections_report_post_rename_uncertainty() {
        for stage in [
            AtomicWriteStage::BeforeParentDirectorySync,
            AtomicWriteStage::AfterParentDirectorySync,
        ] {
            let root = std::env::temp_dir().join(format!(
                "maybe-wordle-directory-sync-failure-{}",
                stage_name(stage)
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("root");
            let path = root.join("artifact.json");
            fs::write(&path, b"valid-old").expect("seed");

            atomic_write_with_hook(&path, b"new", |current| {
                if current == stage {
                    anyhow::bail!("injected failure at {}", stage_name(stage));
                }
                Ok(())
            })
            .expect_err("injected failure");
            assert_eq!(fs::read(&path).expect("read new"), b"new");
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn stale_cleanup_removes_only_exact_owned_versioned_siblings() {
        let root = std::env::temp_dir().join("maybe-wordle-owned-stale-temp-cleanup");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("root");
        let path = root.join("artifact.json");
        let owned = root.join(format!(".artifact.json.{TEMP_FORMAT}.123.456.tmp"));
        let malformed = root.join(format!(".artifact.json.{TEMP_FORMAT}.123.bad.tmp"));
        let other_target = root.join(format!(".other.json.{TEMP_FORMAT}.123.456.tmp"));
        let old_format = root.join(".artifact.json.123.456.tmp");
        let unrelated = root.join(".artifact.json.user.tmp");
        for candidate in [&owned, &malformed, &other_target, &old_format, &unrelated] {
            fs::write(candidate, b"temporary").expect("temporary");
        }

        assert_eq!(
            cleanup_stale_atomic_temps(&path, Duration::ZERO).expect("cleanup"),
            1
        );
        assert!(!owned.exists());
        for preserved in [malformed, other_target, old_format, unrelated] {
            assert!(preserved.exists(), "preserved {}", preserved.display());
        }
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(windows)]
    #[test]
    fn windows_atomic_replacement_supports_extended_length_paths() {
        let root = std::env::temp_dir().join(format!(
            "maybe-wordle-atomic-long-path-{}",
            std::process::id()
        ));
        let nested = root
            .join("a".repeat(80))
            .join("b".repeat(80))
            .join("c".repeat(80));
        fs::create_dir_all(&nested).expect("long root");
        let path = nested.join("artifact.json");
        assert!(path.as_os_str().to_string_lossy().encode_utf16().count() > 260);

        atomic_write(&path, b"first").expect("create long artifact");
        atomic_write(&path, b"second").expect("replace long artifact");
        assert_eq!(fs::read(&path).expect("read long artifact"), b"second");
        let _ = fs::remove_dir_all(root);
    }

    fn stage_name(stage: AtomicWriteStage) -> &'static str {
        match stage {
            AtomicWriteStage::TempCreated => "temp-created",
            AtomicWriteStage::DataWritten => "data-written",
            AtomicWriteStage::TempSynced => "temp-synced",
            AtomicWriteStage::BeforeReplace => "before-replace",
            AtomicWriteStage::BeforePlatformReplace => "before-platform-replace",
            AtomicWriteStage::AfterPlatformReplace => "after-platform-replace",
            #[cfg(unix)]
            AtomicWriteStage::BeforeParentDirectorySync => "before-parent-directory-sync",
            #[cfg(unix)]
            AtomicWriteStage::AfterParentDirectorySync => "after-parent-directory-sync",
        }
    }
}
