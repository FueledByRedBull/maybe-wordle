pub mod atomic_file;
pub mod config;
pub mod data;
pub mod experiments;
pub mod formal;
pub mod game;
pub mod gui;
pub mod identity;
pub mod model;
pub mod pattern_table;
pub mod predictive;
pub(crate) mod process_memory;
pub mod research;
pub mod scoring;
pub mod seed;
pub mod small_state;
pub mod solver;

#[cfg(test)]
pub(crate) mod test_support;

pub const SOLVER_THREAD_STACK_BYTES: usize = 8 * 1024 * 1024;
