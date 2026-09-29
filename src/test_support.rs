use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

/// An exclusively created, project-local fixture, removed when its owner drops.
#[derive(Debug)]
pub(crate) struct TestDirectory(PathBuf);

impl TestDirectory {
    pub(crate) fn new(label: &str) -> Self {
        assert!(
            label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        );
        let parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test-fixtures");
        fs::create_dir_all(&parent).expect("create fixture parent");
        loop {
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!("{label}-{}-{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create owned fixture {}: {error}", path.display()),
            }
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        // Never clean a replacement link or anything outside the owned fixture.
        if fs::symlink_metadata(&self.0)
            .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
        {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

#[test]
fn fixtures_with_the_same_label_are_independent_and_owned() {
    let first = TestDirectory::new("ownership");
    let first_path = first.path().to_owned();
    let second = TestDirectory::new("ownership");
    assert_ne!(first.path(), second.path());
    fs::write(first.path().join("first.txt"), "first").expect("first fixture");
    fs::write(second.path().join("second.txt"), "second").expect("second fixture");
    drop(first);
    assert!(!first_path.exists());
    assert_eq!(
        fs::read_to_string(second.path().join("second.txt")).expect("second kept"),
        "second"
    );
}
