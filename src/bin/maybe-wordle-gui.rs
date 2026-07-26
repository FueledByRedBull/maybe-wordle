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
        std::process::exit(1);
    }
}

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
