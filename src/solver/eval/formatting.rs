use anyhow::{Result, anyhow, bail};

use crate::solver::{PredictiveEvidenceArtifact, RollingComparisonArtifact, Solver};

use super::validate_rolling_comparison_artifact;

impl Solver {
    pub fn render_development_evidence_markdown(
        artifact: &PredictiveEvidenceArtifact,
    ) -> Result<String> {
        artifact.validate_identity()?;
        let mut output = String::new();
        output.push_str("<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->\n");
        output.push_str("## Predictive solver evidence\n\n");
        let selected_ranges = artifact
            .selected_ranges
            .iter()
            .map(|range| format!("{}..{}", range.start, range.end))
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&format!(
            "Development-only diagnostic for `{}` through `{}` using selection `{}` ({}) and history through `{}`. The sealed test was **not** evaluated.\n\n",
            artifact.evaluation_from,
            artifact.evaluation_to,
            artifact.evaluation_selection,
            selected_ranges,
            artifact.history_snapshot_end
        ));
        if let Some(peak_bytes) = artifact.resources.peak_working_set_bytes {
            output.push_str(&format!(
                "Measured generation compute time: {:.2} s; process peak working set: {:.1} MiB; enforced budget: {} s / {} MiB.\n\n",
                artifact.resources.generation_compute_ms as f64 / 1_000.0,
                peak_bytes as f64 / (1024.0 * 1024.0),
                artifact.resource_budget.maximum_seconds,
                artifact.resource_budget.maximum_memory_mb
            ));
        }
        output.push_str("| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs reference | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for baseline in &artifact.baselines {
            let metrics = &baseline.result.backtest.canonical;
            let paired = baseline
                .paired_vs_selected_default
                .expect("generated evidence always has paired comparisons");
            output.push_str(&format!(
                "| `{}` | {:.1}% ({}/{}) | {:.1}% ({}/{}) | {:.4} [{:.4}, {:.4}] | {} | {:.1}% | {:.1}% | {:+.4} [{:+.4}, {:+.4}] | {}/{}/{} | {} | {} | {:.2} ms | {}/{} |\n",
                baseline.id,
                metrics.coverage_rate * 100.0,
                metrics.modeled_games,
                metrics.scheduled_games,
                metrics.solve_rate * 100.0,
                metrics.solved_games,
                metrics.scheduled_games,
                metrics.all_game_penalized_mean_guesses,
                metrics.all_game_penalized_mean_guesses_ci95.lower,
                metrics.all_game_penalized_mean_guesses_ci95.upper,
                metrics.conditional_mean_summary(),
                metrics.solved_in_guess_counts[..3].iter().sum::<usize>() as f64
                    / metrics.scheduled_games.max(1) as f64
                    * 100.0,
                metrics.solved_in_guess_counts[..4].iter().sum::<usize>() as f64
                    / metrics.scheduled_games.max(1) as f64
                    * 100.0,
                paired.candidate_minus_baseline,
                paired.ci95.lower,
                paired.ci95.upper,
                paired.candidate_wins,
                paired.ties,
                paired.baseline_wins,
                baseline.result.average_log_loss.map_or_else(
                    || format!("unavailable (measured_prior_games=0/{})", metrics.scheduled_games),
                    |value| format!("{value:.4}"),
                ),
                baseline.result.average_brier.map_or_else(
                    || format!("unavailable (measured_prior_games=0/{})", metrics.scheduled_games),
                    |value| format!("{value:.4}"),
                ),
                baseline.result.latency_p95_ms,
                baseline.result.session_fallback_cold_ms.map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
                baseline.result.session_fallback_warm_ms.map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
            ));
        }
        output.push_str("\nSession-fallback timings are milliseconds; n/a means live session books are not used by that profile and were not benchmarked.\n");
        if !artifact.resources.artifact_sizes.is_empty() {
            output.push_str("\nMeasured artifact sizes: ");
            output.push_str(
                &artifact
                    .resources
                    .artifact_sizes
                    .iter()
                    .map(|artifact| format!("`{}` = {} bytes", artifact.name, artifact.bytes))
                    .collect::<Vec<_>>()
                    .join("; "),
            );
            output.push_str(".\n");
        }
        output.push_str("\n| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X/F/T | Recovery/fallback steps | Artifact/session hits |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for baseline in &artifact.baselines {
            let prior = baseline.result.prior_evidence.as_ref();
            let telemetry = &baseline.result.execution;
            let recall = |value: Option<f64>| {
                value.map_or_else(
                    || "n/a".to_string(),
                    |value| format!("{:.1}%", value * 100.0),
                )
            };
            let ece = prior.map_or_else(
                || "n/a".to_string(),
                |metrics| {
                    format!(
                        "{:.4} [{:.4}, {:.4}]",
                        metrics.expected_calibration_error,
                        metrics.expected_calibration_error_ci95.lower,
                        metrics.expected_calibration_error_ci95.upper
                    )
                },
            );
            output.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {}/{}/{}/{}/{}/{} | {}/{} | {}/{} |\n",
                baseline.id,
                recall(prior.map(|metrics| metrics.top_1_recall)),
                recall(prior.map(|metrics| metrics.top_3_recall)),
                recall(prior.map(|metrics| metrics.top_5_recall)),
                ece,
                telemetry.proxy_steps,
                telemetry.lookahead_steps,
                telemetry.escalated_exact_steps,
                telemetry.exact_steps,
                telemetry.finite_steps,
                telemetry.terminal_steps,
                telemetry.strict_recovery_steps
                    + telemetry.uniform_recovery_steps
                    + telemetry.epsilon_repair_steps,
                telemetry.dormant_fallback_steps,
                telemetry.exact_date_opener_artifact_hits
                    + telemetry.recent_opener_artifact_hits
                    + telemetry.reply_book_hits,
                telemetry.session_fallback_hits,
            ));
        }
        if artifact
            .baselines
            .iter()
            .any(|baseline| !baseline.result.posterior_calibration.is_empty())
        {
            output.push_str(
                "\nPost-feedback posterior proper scores (means are conditional on scored states; scored/total keeps unscored gaps visible):\n\n",
            );
            output.push_str(
                "| Baseline | Stratum | Turn | Scored/total states | Target probability | Log loss | Brier |\n",
            );
            output.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
            for baseline in &artifact.baselines {
                for summary in &baseline.result.posterior_calibration {
                    if summary.total_states == 0 {
                        continue;
                    }
                    let (target_probability, log_loss, brier) = summary.mean_score.map_or(
                        ("n/a".to_string(), "n/a".to_string(), "n/a".to_string()),
                        |score| {
                            (
                                format!("{:.4}", score.target_probability),
                                format!("{:.4}", score.log_loss),
                                format!("{:.4}", score.brier),
                            )
                        },
                    );
                    output.push_str(&format!(
                        "| `{}` | {} | {} | {}/{} | {} | {} | {} |\n",
                        baseline.id,
                        summary.stratum,
                        summary.turn,
                        summary.scored_states,
                        summary.total_states,
                        target_probability,
                        log_loss,
                        brier,
                    ));
                }
            }
        }
        if let Some(reference) = artifact
            .baselines
            .iter()
            .find(|baseline| baseline.id == artifact.reference_profile_id)
        {
            output.push_str(&format!(
                "\nReference `{}` all-game mean sensitivity: ",
                artifact.reference_profile_id
            ));
            for (index, metric) in reference
                .result
                .failure_penalty_sensitivity
                .iter()
                .enumerate()
            {
                if index > 0 {
                    output.push_str("; ");
                }
                output.push_str(&format!(
                    "penalty {:.0} = {:.4} [{:.4}, {:.4}]",
                    metric.penalty_guesses,
                    metric.all_game_mean_guesses,
                    metric.ci95.lower,
                    metric.ci95.upper
                ));
            }
            output.push_str(".\n");
        }
        output.push_str("\nThe old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.\n\n");
        output.push_str("The source JSON artifact records the `release_command`, full provenance, per-game paths, effective profile configs, paired comparisons, and limitations. Regenerate documentation with `benchmark-evidence-docs --evidence <source-json> --markdown-output <fragment> --readme <readme> --update`.\n");
        output.push_str("<!-- END GENERATED PREDICTIVE EVIDENCE -->\n");
        Ok(output)
    }

    pub fn render_rolling_comparison_markdown(
        comparisons: &[RollingComparisonArtifact],
    ) -> Result<String> {
        let first = comparisons
            .first()
            .ok_or_else(|| anyhow!("at least one rolling comparison is required"))?;
        for comparison in comparisons {
            validate_rolling_comparison_artifact(comparison)?;
        }
        if comparisons.iter().any(|comparison| {
            comparison.evaluation_plan != first.evaluation_plan
                || comparison.baseline.label != first.baseline.label
                || comparison.baseline.config_toml != first.baseline.config_toml
                || comparison.baseline.aggregate != first.baseline.aggregate
        }) {
            bail!("rolling comparisons must share one development plan and baseline");
        }
        let baseline = &first.baseline;
        let mut output = String::new();
        output.push_str("<!-- BEGIN GENERATED ROLLING EVIDENCE -->\n");
        output.push_str("### Rolling-origin promotion guard\n\n");
        output.push_str(&format!(
            "Across {} non-overlapping development folds ({} scheduled games), the sealed test was **not** evaluated. Coverage gaps and six-guess failures are hard constraints before mean score.\n\n",
            first.evaluation_plan.folds.len(),
            baseline.aggregate.scheduled_games
        ));
        output.push_str("| Configuration | Solved | All-game mean | Delta vs baseline | W/T/L | Latency p95 | Guard decision |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n");
        output.push_str(&format!(
            "| `{}` | {}/{} | {:.4} [{:.4}, {:.4}] | reference | -- | {:.2} ms | retained |\n",
            baseline.label,
            baseline.aggregate.solved_games,
            baseline.aggregate.scheduled_games,
            baseline.aggregate.all_game_penalized_mean_guesses,
            baseline
                .aggregate
                .all_game_penalized_mean_guesses_ci95
                .lower,
            baseline
                .aggregate
                .all_game_penalized_mean_guesses_ci95
                .upper,
            baseline.latency_p95_ms,
        ));
        for comparison in comparisons {
            let candidate = &comparison.candidate;
            let paired = comparison.candidate_minus_baseline;
            let baseline_failures =
                baseline.aggregate.unsolved_games + baseline.aggregate.coverage_gaps;
            let candidate_failures =
                candidate.aggregate.unsolved_games + candidate.aggregate.coverage_gaps;
            let decision = if candidate_failures > baseline_failures {
                "rejected: added failures"
            } else if paired.ci95.upper < 0.0 {
                "eligible on solve quality"
            } else if paired.candidate_minus_baseline < 0.0 {
                "not promoted: improvement uncertain"
            } else {
                "rejected: no solve-quality gain"
            };
            output.push_str(&format!(
                "| `{}` | {}/{} | {:.4} [{:.4}, {:.4}] | {:+.4} [{:+.4}, {:+.4}] | {}/{}/{} | {:.2} ms | {} |\n",
                candidate.label,
                candidate.aggregate.solved_games,
                candidate.aggregate.scheduled_games,
                candidate.aggregate.all_game_penalized_mean_guesses,
                candidate.aggregate.all_game_penalized_mean_guesses_ci95.lower,
                candidate.aggregate.all_game_penalized_mean_guesses_ci95.upper,
                paired.candidate_minus_baseline,
                paired.ci95.lower,
                paired.ci95.upper,
                paired.candidate_wins,
                paired.ties,
                paired.baseline_wins,
                candidate.latency_p95_ms,
                decision,
            ));
        }
        output.push_str("\n| Configuration | Prior top-1/3/5 | Confidence ECE | Search steps P/L/XE/X/F/T | Recovery/fallback steps |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: |\n");
        for evidence in std::iter::once(baseline)
            .chain(comparisons.iter().map(|comparison| &comparison.candidate))
        {
            let prior = evidence.prior_evidence.as_ref();
            let telemetry = &evidence.execution;
            output.push_str(&format!(
                "| `{}` | {} | {} | {}/{}/{}/{}/{}/{} | {}/{} |\n",
                evidence.label,
                prior.map_or_else(
                    || "n/a".to_string(),
                    |metrics| format!(
                        "{:.1}%/{:.1}%/{:.1}%",
                        metrics.top_1_recall * 100.0,
                        metrics.top_3_recall * 100.0,
                        metrics.top_5_recall * 100.0
                    )
                ),
                prior.map_or_else(
                    || "n/a".to_string(),
                    |metrics| format!(
                        "{:.4} [{:.4}, {:.4}]",
                        metrics.expected_calibration_error,
                        metrics.expected_calibration_error_ci95.lower,
                        metrics.expected_calibration_error_ci95.upper
                    )
                ),
                telemetry.proxy_steps,
                telemetry.lookahead_steps,
                telemetry.escalated_exact_steps,
                telemetry.exact_steps,
                telemetry.finite_steps,
                telemetry.terminal_steps,
                telemetry.strict_recovery_steps
                    + telemetry.uniform_recovery_steps
                    + telemetry.epsilon_repair_steps,
                telemetry.dormant_fallback_steps,
            ));
        }
        output.push_str("\nDevelopment decisions:\n\n");
        for comparison in comparisons {
            let candidate = &comparison.candidate;
            let paired = comparison.candidate_minus_baseline;
            let baseline_failures =
                baseline.aggregate.unsolved_games + baseline.aggregate.coverage_gaps;
            let candidate_failures =
                candidate.aggregate.unsolved_games + candidate.aggregate.coverage_gaps;
            let explanation = if candidate_failures > baseline_failures {
                format!(
                    "rejected because it added {} failure(s)",
                    candidate_failures - baseline_failures
                )
            } else if paired.ci95.upper < 0.0 {
                "eligible on solve quality because the paired interval is entirely below zero"
                    .to_string()
            } else if paired.candidate_minus_baseline < 0.0 {
                "retained as a development finalist, not promoted, because the observed improvement's paired interval includes zero"
                    .to_string()
            } else {
                "rejected because it did not improve solve quality".to_string()
            };
            output.push_str(&format!("- `{}` is {}.\n", candidate.label, explanation));
        }
        output.push_str(
            "\nThis development comparison did not access the sealed window and does not establish prospective performance. Any later sealed evaluation requires separate evidence.\n",
        );
        output.push_str("<!-- END GENERATED ROLLING EVIDENCE -->\n");
        Ok(output)
    }
}
