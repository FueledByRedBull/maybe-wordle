#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use maybe_wordle::{SOLVER_THREAD_STACK_BYTES, gui::run_gui};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        report_startup_error(&error);
        std::process::exit(1);
    }
}

fn report_startup_error(error: &anyhow::Error) {
    let log_path = env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .map(|directory| directory.join("MaybeWordle/startup-error.log"))
        .or_else(|| {
            env::current_dir()
                .ok()
                .map(|directory| directory.join("maybe-wordle-startup-error.log"))
        });
    let log_result = log_path.as_ref().map(|path| -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, format!("{error:#}\n"))?;
        Ok(())
    });
    let diagnostic = match (log_path, log_result) {
        (Some(path), Some(Ok(()))) => format!("Diagnostic log: {}", path.display()),
        (Some(path), Some(Err(log_error))) => format!(
            "Could not write diagnostic log {}: {log_error:#}",
            path.display()
        ),
        _ => "Could not resolve a diagnostic log directory.".to_string(),
    };
    let message = startup_error_message(error, &diagnostic);
    eprintln!("{diagnostic}");
    show_startup_error(&message);
}

fn startup_error_message(error: &anyhow::Error, diagnostic: &str) -> String {
    format!("Maybe Wordle could not start.\n\n{error:#}\n\n{diagnostic}")
}

#[cfg(target_os = "windows")]
fn show_startup_error(message: &str) {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn MessageBoxW(
            window: *mut std::ffi::c_void,
            text: *const u16,
            caption: *const u16,
            flags: u32,
        ) -> i32;
    }
    let text: Vec<u16> = message
        .replace('\0', " ")
        .encode_utf16()
        .chain(Some(0))
        .collect();
    let title: Vec<u16> = "Maybe Wordle — startup error"
        .encode_utf16()
        .chain(Some(0))
        .collect();
    // Both UTF-16 buffers are NUL-terminated and remain alive for this synchronous call.
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), 0x10);
    }
}

#[cfg(not(target_os = "windows"))]
fn show_startup_error(_message: &str) {}

fn run() -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .stack_size(SOLVER_THREAD_STACK_BYTES)
        .build_global()
        .context("failed to configure the global solver worker pool")?;
    run_gui(resolve_project_root()?)
}

fn resolve_project_root() -> Result<PathBuf> {
    let current_dir = env::current_dir().context("failed to resolve current directory")?;
    if let Some(root) = find_project_root(&current_dir) {
        return Ok(root);
    }
    if let Ok(current_exe) = env::current_exe()
        && let Some(root) = find_project_root(&current_exe)
    {
        return Ok(root);
    }
    Ok(current_dir)
}

fn find_project_root(start: &Path) -> Option<PathBuf> {
    let anchor = if start.is_dir() {
        start
    } else {
        start.parent()?
    };
    anchor
        .ancestors()
        .find(|candidate| {
            candidate.join("config/prior.toml").is_file()
                && candidate.join("data/seed/valid_guesses.txt").is_file()
                && candidate.join("data/seed/candidate_answers.txt").is_file()
        })
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    #[test]
    fn startup_error_keeps_context_and_diagnostic_location() {
        let error = anyhow::anyhow!("seed file is missing").context("failed to load workspace");
        let message =
            super::startup_error_message(&error, "Diagnostic log: local/startup-error.log");
        assert!(message.contains("failed to load workspace: seed file is missing"));
        assert!(message.contains("local/startup-error.log"));
    }
}
