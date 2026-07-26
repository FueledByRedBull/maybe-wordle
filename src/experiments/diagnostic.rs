use std::collections::HashSet;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub const DIAGNOSTIC_SUITE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreeGuessDiagnosticSpec {
    pub profile: String,
    pub root_candidate_limit: usize,
    pub reply_candidate_limit: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardCaseDiagnosticSpec {
    pub target_count: usize,
    pub top_posterior_scan: usize,
    pub minimum_trap_neighbors: usize,
    pub low_prior_splitter_scan: usize,
    pub maximum_cluster_hamming_distance: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatencyDiagnosticSpec {
    pub evidence_runs: usize,
    pub evaluation_runs: usize,
    pub study_runs: usize,
    pub top_suggestions: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BookDiagnosticSpec {
    pub forced_suggestion_top: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticExperimentSuite {
    pub format_version: u32,
    pub name: String,
    pub three_guess_rescue: ThreeGuessDiagnosticSpec,
    pub default_four_guess_openers: Vec<String>,
    pub hard_cases: HardCaseDiagnosticSpec,
    pub latency: LatencyDiagnosticSpec,
    pub book_build: BookDiagnosticSpec,
}

impl DiagnosticExperimentSuite {
    pub fn parse_json(source: &str) -> Result<Self> {
        let suite: Self = serde_json::from_str(source)?;
        suite.validate()?;
        Ok(suite)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format_version != DIAGNOSTIC_SUITE_FORMAT_VERSION {
            bail!(
                "unsupported diagnostic suite format {}; expected {}",
                self.format_version,
                DIAGNOSTIC_SUITE_FORMAT_VERSION
            );
        }
        if self.name.trim().is_empty() || self.three_guess_rescue.profile.trim().is_empty() {
            bail!("diagnostic suite and profile names must not be empty");
        }
        for (name, value) in [
            (
                "three_guess_rescue.root_candidate_limit",
                self.three_guess_rescue.root_candidate_limit,
            ),
            (
                "three_guess_rescue.reply_candidate_limit",
                self.three_guess_rescue.reply_candidate_limit,
            ),
            ("hard_cases.target_count", self.hard_cases.target_count),
            (
                "hard_cases.top_posterior_scan",
                self.hard_cases.top_posterior_scan,
            ),
            (
                "hard_cases.minimum_trap_neighbors",
                self.hard_cases.minimum_trap_neighbors,
            ),
            (
                "hard_cases.low_prior_splitter_scan",
                self.hard_cases.low_prior_splitter_scan,
            ),
            (
                "hard_cases.maximum_cluster_hamming_distance",
                self.hard_cases.maximum_cluster_hamming_distance,
            ),
            ("latency.evidence_runs", self.latency.evidence_runs),
            ("latency.evaluation_runs", self.latency.evaluation_runs),
            ("latency.study_runs", self.latency.study_runs),
            ("latency.top_suggestions", self.latency.top_suggestions),
            (
                "book_build.forced_suggestion_top",
                self.book_build.forced_suggestion_top,
            ),
        ] {
            if value == 0 {
                bail!("{name} must be greater than zero");
            }
        }
        if self.default_four_guess_openers.is_empty() {
            bail!("default four-guess opener list must not be empty");
        }
        let mut seen = HashSet::new();
        for opener in &self.default_four_guess_openers {
            if opener.len() != 5
                || !opener.bytes().all(|byte| byte.is_ascii_lowercase())
                || !seen.insert(opener)
            {
                bail!("default four-guess openers must be unique lowercase five-letter words");
            }
        }
        if self.hard_cases.target_count > 5 {
            bail!("hard_cases.target_count cannot exceed the five declared case categories");
        }
        if self.hard_cases.maximum_cluster_hamming_distance > 5 {
            bail!("hard-case Hamming distance cannot exceed the word length");
        }
        if self.hard_cases.minimum_trap_neighbors >= self.hard_cases.top_posterior_scan {
            bail!("hard-case trap-neighbor minimum must be smaller than its scan");
        }
        Ok(())
    }
}

pub fn default_diagnostic_suite() -> Result<DiagnosticExperimentSuite> {
    DiagnosticExperimentSuite::parse_json(include_str!(
        "../../config/experiments/diagnostic-suite.json"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_diagnostic_suite_is_valid_and_explicit() {
        let suite = default_diagnostic_suite().expect("suite");
        assert_eq!(suite.format_version, DIAGNOSTIC_SUITE_FORMAT_VERSION);
        assert_eq!(suite.three_guess_rescue.profile, "aggressive-three-guess");
        assert_eq!(suite.default_four_guess_openers.len(), 8);
        assert!(suite.latency.evidence_runs >= suite.latency.evaluation_runs);
    }

    #[test]
    fn diagnostic_suite_rejects_duplicate_openers_and_zero_budgets() {
        let mut suite = default_diagnostic_suite().expect("suite");
        suite.default_four_guess_openers[1] = suite.default_four_guess_openers[0].clone();
        assert!(suite.validate().is_err());

        let mut suite = default_diagnostic_suite().expect("suite");
        suite.latency.study_runs = 0;
        assert!(suite.validate().is_err());
    }
}
