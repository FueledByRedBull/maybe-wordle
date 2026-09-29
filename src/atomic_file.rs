use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

const TEMP_FORMAT: &str = "mwatomic-v1";
const STALE_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Holds a cooperating writer's exclusive lock until the returned file is dropped.
/// The sidecar remains on disk: unlinking it could split concurrent lock owners.
pub(crate) fn acquire_edit_lock(destination: &Path) -> Result<fs::File> {
    let destination = std::path::absolute(destination)?;
    let parent = parent_directory(&destination);
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let mut name = std::ffi::OsString::from(".");
    name.push(
        destination
            .file_name()
            .context("edit destination needs a filename")?,
    );
    name.push(".mwedit.lock");
    let lock_path = destination.with_file_name(name);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open edit lock {}", lock_path.display()))?;
    file.try_lock().with_context(|| {
        format!(
            "cannot acquire edit lock for {}; another edit may be in progress",
            destination.display()
        )
    })?;
    Ok(file)
}

/// Durably writes bytes to a sibling temporary file and atomically replaces `path`.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_with(path, |file| Ok(file.write_all(bytes)?))
}

/// Streams into a sibling temporary file, then durably replaces `path`.
/// The callback must finish writing (including flushing its own buffers) before returning.
pub fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut fs::File) -> Result<()>,
) -> Result<()> {
    atomic_write_with_hook(path, write, |stage| {
        #[cfg(test)]
        test_hooks::check(stage)?;
        #[cfg(not(test))]
        let _ = stage;
        Ok(())
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomicWriteStage {
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

#[cfg(test)]
pub(crate) mod test_hooks {
    use super::AtomicWriteStage;
    use std::cell::Cell;

    thread_local! {
        static FAILURE: Cell<Option<AtomicWriteStage>> = const { Cell::new(None) };
        static TEMP_NONCE: Cell<Option<u128>> = const { Cell::new(None) };
    }

    pub(crate) fn with_failure<T>(stage: AtomicWriteStage, operation: impl FnOnce() -> T) -> T {
        struct Restore(Option<AtomicWriteStage>);
        impl Drop for Restore {
            fn drop(&mut self) {
                FAILURE.set(self.0);
            }
        }
        let _restore = Restore(FAILURE.replace(Some(stage)));
        operation()
    }

    pub(super) fn check(stage: AtomicWriteStage) -> anyhow::Result<()> {
        if FAILURE.get() == Some(stage) {
            anyhow::bail!("injected atomic write failure at {stage:?}");
        }
        Ok(())
    }

    pub(super) fn with_temp_nonce<T>(nonce: u128, operation: impl FnOnce() -> T) -> T {
        struct Restore(Option<u128>);
        impl Drop for Restore {
            fn drop(&mut self) {
                TEMP_NONCE.set(self.0);
            }
        }
        let _restore = Restore(TEMP_NONCE.replace(Some(nonce)));
        operation()
    }

    pub(super) fn temp_nonce() -> Option<u128> {
        TEMP_NONCE.get()
    }
}

fn atomic_write_with_hook(
    path: &Path,
    write: impl FnOnce(&mut fs::File) -> Result<()>,
    mut stage_hook: impl FnMut(AtomicWriteStage) -> Result<()>,
) -> Result<()> {
    // Use native path rules: Windows ordinary paths resolve dot components,
    // while POSIX keeps `..` so directory symlinks retain their meaning.
    let path = std::path::absolute(path)
        .with_context(|| format!("failed to resolve atomic destination {}", path.display()))?;
    let parent = parent_directory(&path);
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    cleanup_stale_atomic_temps(&path, STALE_TEMP_AGE)?;

    let temp = sibling_temp_path(&path);
    let mut created = false;
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .with_context(|| format!("failed to create {}", temp.display()))?;
        created = true;
        stage_hook(AtomicWriteStage::TempCreated)?;
        write(&mut file).with_context(|| format!("failed to write {}", temp.display()))?;
        stage_hook(AtomicWriteStage::DataWritten)?;
        file.flush()
            .with_context(|| format!("failed to flush {}", temp.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temp.display()))?;
        stage_hook(AtomicWriteStage::TempSynced)?;
        drop(file);
        stage_hook(AtomicWriteStage::BeforeReplace)?;
        replace_file_with_hook(&temp, &path, &mut stage_hook)
    })();

    if result.is_err() && created {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn sibling_temp_path(path: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    #[cfg(test)]
    let nonce = test_hooks::temp_nonce().unwrap_or(nonce);
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
    let parent = parent_directory(path);
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
    let parent = parent_directory(destination);
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
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    let source_wide = windows_extended_path(source);
    let destination_wide = windows_extended_path(destination);
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

#[cfg(windows)]
fn windows_extended_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    // Callers already applied native absolute-path rules; do not reinterpret
    // separators or dot components in an explicitly supplied verbatim path.
    let raw = path.as_os_str().encode_wide().collect::<Vec<_>>();
    const VERBATIM: &[u16] = &[b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16];
    const DEVICE: &[u16] = &[b'\\' as u16, b'\\' as u16, b'.' as u16, b'\\' as u16];
    let mut extended = if raw.starts_with(VERBATIM) || raw.starts_with(DEVICE) {
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
    extended
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_isolated_directory(name: &str, test: impl FnOnce()) {
        const CHILD_TEST: &str = "MAYBE_WORDLE_ATOMIC_CHILD_TEST";
        if std::env::var(CHILD_TEST).as_deref() == Ok(name) {
            test();
            return;
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::current_dir()
            .expect("cwd")
            .join("target/audit-work/atomic-tests")
            .join(format!("{name}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&root).expect("isolated directory");
        let thread = std::thread::current();
        let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", thread.name().expect("test name"), "--nocapture"])
            .current_dir(&root)
            .env(CHILD_TEST, name)
            .output()
            .expect("run isolated test");
        fs::remove_dir_all(&root).expect("remove isolated test directory");
        assert!(
            output.status.success(),
            "isolated test failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn bare_filename_cleanup_inspects_current_directory() {
        in_isolated_directory("bare_filename_cleanup_inspects_current_directory", || {
            let owned = format!(".out.json.{TEMP_FORMAT}.123.456.tmp");
            fs::write(&owned, b"interrupted").expect("stale temporary");
            assert_eq!(
                cleanup_stale_atomic_temps(Path::new("out.json"), Duration::ZERO)
                    .expect("cleanup bare filename"),
                1
            );
            assert!(!Path::new(&owned).exists());
        });
    }

    #[test]
    fn colliding_temporary_is_not_owned_or_removed_by_failed_writer() {
        in_isolated_directory(
            "colliding_temporary_is_not_owned_or_removed_by_failed_writer",
            || {
                for existing in [false, true] {
                    let path = PathBuf::from(format!("artifact-{existing}.json"));
                    if existing {
                        fs::write(&path, b"old").expect("existing destination");
                    }
                    test_hooks::with_temp_nonce(123, || {
                        let temp = sibling_temp_path(&path);
                        fs::write(&temp, b"other writer's data").expect("colliding temporary");
                        let error = atomic_write_with(&path, |_| {
                            panic!("collision must not start writing")
                        })
                        .expect_err("create_new must reject collision");
                        assert_eq!(
                            error
                                .downcast_ref::<std::io::Error>()
                                .expect("I/O error")
                                .kind(),
                            std::io::ErrorKind::AlreadyExists
                        );
                        assert_eq!(
                            fs::read(&temp).expect("other writer's temporary retained"),
                            b"other writer's data"
                        );
                        if existing {
                            assert_eq!(fs::read(&path).expect("original destination"), b"old");
                        } else {
                            assert!(!path.exists());
                        }
                    });
                }
            },
        );
    }

    #[test]
    fn relative_destinations_create_and_replace_with_correct_status() {
        in_isolated_directory(
            "relative_destinations_create_and_replace_with_correct_status",
            || {
                fs::create_dir("traverse").expect("real traversal directory");
                for path in [
                    "out.json",
                    "./dot.json",
                    "nested/out.json",
                    "traverse/../parent.json",
                    "\u{3bd}\u{3ad}\u{3bf}/\u{3ba}\u{3cc}\u{3c3}\u{3bc}\u{3bf}\u{3c2}.json",
                ] {
                    let path = Path::new(path);
                    atomic_write(path, b"first").expect("create relative artifact");
                    assert_eq!(fs::read(path).expect("read new artifact"), b"first");
                    atomic_write(path, b"second").expect("replace relative artifact");
                    assert_eq!(fs::read(path).expect("read replaced artifact"), b"second");
                }
            },
        );
    }

    #[cfg(unix)]
    #[test]
    fn parent_traversal_retains_directory_symlink_semantics() {
        in_isolated_directory(
            "parent_traversal_retains_directory_symlink_semantics",
            || {
                fs::create_dir_all("real/nested").expect("real directories");
                std::os::unix::fs::symlink("real/nested", "link").expect("directory symlink");
                atomic_write(Path::new("link/../out.json"), b"new").expect("symlink traversal");
                assert_eq!(
                    fs::read("real/out.json").expect("actual parent destination"),
                    b"new"
                );
                assert!(!Path::new("out.json").exists());
            },
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_namespace_conversion_preserves_verbatim_and_unc_paths() {
        for (input, expected) in [
            (r"C:\dir\..\new.json", "\\\\?\\C:\\new.json\0"),
            (
                r"\\server\share\dir\..\new.json",
                "\\\\?\\UNC\\server\\share\\new.json\0",
            ),
            (
                r"\\?\UNC\server\share\dir\..\new.json",
                "\\\\?\\UNC\\server\\share\\dir\\..\\new.json\0",
            ),
            (r"\\?\C:\dir\..\new.json", "\\\\?\\C:\\dir\\..\\new.json\0"),
            (r"\\.\C:\new.json", "\\\\.\\C:\\new.json\0"),
        ] {
            let absolute = std::path::absolute(input).expect("native absolute path");
            assert_eq!(
                windows_extended_path(&absolute),
                expected.encode_utf16().collect::<Vec<_>>()
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_verbatim_destinations_create_and_replace() {
        in_isolated_directory("windows_verbatim_destinations_create_and_replace", || {
            let path = fs::canonicalize(".")
                .expect("verbatim root")
                .join("artifact.json");
            assert!(path.as_os_str().to_string_lossy().starts_with(r"\\?\"));
            atomic_write(&path, b"first").expect("create verbatim destination");
            assert_eq!(fs::read(&path).expect("new artifact"), b"first");
            atomic_write(&path, b"second").expect("replace verbatim destination");
            assert_eq!(fs::read(&path).expect("replacement"), b"second");
        });
    }

    #[test]
    fn streaming_write_publishes_all_chunks_and_preserves_old_file_on_error() {
        in_isolated_directory(
            "streaming_write_publishes_all_chunks_and_preserves_old_file_on_error",
            || {
                let path = Path::new("streamed.bin");
                let unrelated = Path::new(".streamed.bin.user.tmp");
                fs::write(unrelated, b"unrelated").expect("unrelated temporary");
                atomic_write_with(path, |file| {
                    file.write_all(b"header")?;
                    file.write_all(b"payload")?;
                    Ok(())
                })
                .expect("streamed write");
                assert_eq!(
                    fs::read(path).expect("read streamed artifact"),
                    b"headerpayload"
                );
                let error = atomic_write_with(path, |file| {
                    file.write_all(b"partial")?;
                    anyhow::bail!("injected stream failure");
                })
                .expect_err("failed streaming write");
                assert!(format!("{error:#}").contains("injected stream failure"));
                assert_eq!(
                    fs::read(path).expect("preserved artifact"),
                    b"headerpayload"
                );
                atomic_write_with(Path::new("new.bin"), |file| {
                    file.write_all(b"partial")?;
                    anyhow::bail!("injected stream failure");
                })
                .expect_err("failed new streaming write");
                assert!(!Path::new("new.bin").exists());
                assert_eq!(
                    fs::read(unrelated).expect("unrelated temporary preserved"),
                    b"unrelated"
                );
                assert_eq!(fs::read_dir(".").expect("directory").count(), 2);
            },
        );
    }

    #[test]
    fn relative_failures_preserve_pre_and_post_replace_state() {
        in_isolated_directory(
            "relative_failures_preserve_pre_and_post_replace_state",
            || {
                for stage in [
                    AtomicWriteStage::TempCreated,
                    AtomicWriteStage::DataWritten,
                    AtomicWriteStage::TempSynced,
                    AtomicWriteStage::BeforeReplace,
                    AtomicWriteStage::BeforePlatformReplace,
                    AtomicWriteStage::AfterPlatformReplace,
                    #[cfg(unix)]
                    AtomicWriteStage::BeforeParentDirectorySync,
                    #[cfg(unix)]
                    AtomicWriteStage::AfterParentDirectorySync,
                ] {
                    for existing in [false, true] {
                        let path = PathBuf::from(format!("{}-{existing}.json", stage_name(stage)));
                        if existing {
                            fs::write(&path, b"old").expect("seed existing artifact");
                        }
                        let error = atomic_write_with_hook(
                            &path,
                            |file| Ok(file.write_all(b"new")?),
                            |current| {
                                if current == stage {
                                    anyhow::bail!("injected failure");
                                }
                                Ok(())
                            },
                        )
                        .expect_err("injected write must report failure");
                        assert!(error.to_string().contains("injected failure"));
                        let replaced = match stage {
                            AtomicWriteStage::AfterPlatformReplace => true,
                            #[cfg(unix)]
                            AtomicWriteStage::BeforeParentDirectorySync
                            | AtomicWriteStage::AfterParentDirectorySync => true,
                            _ => false,
                        };
                        if replaced {
                            assert_eq!(fs::read(&path).expect("new artifact"), b"new");
                        } else if existing {
                            assert_eq!(fs::read(&path).expect("old artifact"), b"old");
                        } else {
                            assert!(!path.exists());
                        }
                        assert!(!fs::read_dir(".").expect("directory").any(|entry| {
                            entry
                                .expect("entry")
                                .file_name()
                                .to_string_lossy()
                                .ends_with(".tmp")
                        }));
                    }
                }
            },
        );
    }

    #[test]
    fn atomic_replacement_leaves_no_temporary_file() {
        let fixture = crate::test_support::TestDirectory::new("atomic-audit");
        let root = fixture.path().to_path_buf();
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
        let fixture = crate::test_support::TestDirectory::new("atomic-audit");
        let root = fixture.path().to_path_buf();
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
            let fixture = crate::test_support::TestDirectory::new("atomic-audit");
            let root = fixture.path().to_path_buf();
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("root");
            let path = root.join("artifact.json");
            fs::write(&path, b"valid-old").expect("seed");

            let error = atomic_write_with_hook(
                &path,
                |file| Ok(file.write_all(b"new")?),
                |current| {
                    if current == stage {
                        anyhow::bail!("injected failure at {}", stage_name(stage));
                    }
                    Ok(())
                },
            )
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
            let fixture = crate::test_support::TestDirectory::new("atomic-audit");
            let root = fixture.path().to_path_buf();
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("root");
            let path = root.join("artifact.json");
            fs::write(&path, b"valid-old").expect("seed");

            let error = atomic_write_with_hook(
                &path,
                |file| Ok(file.write_all(b"new")?),
                |current| {
                    if current == stage {
                        anyhow::bail!("injected failure at {}", stage_name(stage));
                    }
                    Ok(())
                },
            )
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
            let fixture = crate::test_support::TestDirectory::new("atomic-audit");
            let root = fixture.path().to_path_buf();
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("root");
            let path = root.join("artifact.json");
            fs::write(&path, b"valid-old").expect("seed");

            atomic_write_with_hook(
                &path,
                |file| Ok(file.write_all(b"new")?),
                |current| {
                    if current == stage {
                        anyhow::bail!("injected failure at {}", stage_name(stage));
                    }
                    Ok(())
                },
            )
            .expect_err("injected failure");
            assert_eq!(fs::read(&path).expect("read new"), b"new");
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn stale_cleanup_removes_only_exact_owned_versioned_siblings() {
        let fixture = crate::test_support::TestDirectory::new("atomic-audit");
        let root = fixture.path().to_path_buf();
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
        let fixture = crate::test_support::TestDirectory::new("atomic-audit");
        let root = fixture.path().to_path_buf();
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
