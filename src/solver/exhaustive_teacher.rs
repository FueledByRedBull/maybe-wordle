use std::time::{Duration, Instant};

use anyhow::Result;

#[derive(Debug)]
pub(super) struct BudgetExceeded(pub(super) &'static str);

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for BudgetExceeded {}

pub(super) struct WorkBudget {
    started: Instant,
    prior_elapsed: Duration,
    maximum_time: Duration,
    maximum_memory_bytes: Option<u64>,
    last_memory_check: Option<Instant>,
    peak_memory_bytes: Option<u64>,
}

impl WorkBudget {
    pub(super) fn new(
        started: Instant,
        prior_elapsed_ms: u64,
        maximum_time: Duration,
        maximum_memory_bytes: Option<u64>,
    ) -> Self {
        Self {
            started,
            prior_elapsed: Duration::from_millis(prior_elapsed_ms),
            maximum_time,
            maximum_memory_bytes,
            last_memory_check: None,
            peak_memory_bytes: None,
        }
    }

    pub(super) fn peak_memory_bytes(&self) -> Option<u64> {
        self.peak_memory_bytes
    }

    pub(super) fn check_now(&mut self) -> Result<()> {
        self.last_memory_check = None;
        self.check()
    }

    pub(super) fn check(&mut self) -> Result<()> {
        if self.prior_elapsed.saturating_add(self.started.elapsed()) >= self.maximum_time {
            return Err(BudgetExceeded(
                "offline evaluation exceeded its cumulative wall-clock budget",
            )
            .into());
        }
        // This is a cooperative process-level guard, not an allocator limit. Keep the
        // clock check on every unit of work, but throttle process measurements.
        if self
            .last_memory_check
            .is_none_or(|last| last.elapsed() >= Duration::from_millis(100))
        {
            if let Some(snapshot) = crate::process_memory::process_memory_snapshot() {
                let peak = self
                    .peak_memory_bytes
                    .unwrap_or(0)
                    .max(snapshot.peak_working_set_bytes);
                self.peak_memory_bytes = Some(peak);
                if let Some(maximum) = self.maximum_memory_bytes
                    && peak > maximum
                {
                    return Err(BudgetExceeded(
                        "offline evaluation process peak working set exceeded its memory budget",
                    )
                    .into());
                }
            } else if self.maximum_memory_bytes.is_some() {
                return Err(BudgetExceeded(
                    "offline evaluation cannot measure its process memory budget",
                )
                .into());
            }
            self.last_memory_check = Some(Instant::now());
        }
        Ok(())
    }
}
