use std::{
    env,
    io::{self, Write},
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use maybe_wordle::{
    config::PriorConfig,
    data::{ProjectPaths, sync_nyt_history},
    formal::{
        DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime, FormalVerificationMode, build_optimal_policy,
        parse_observations as parse_formal_observations, verify_optimal_policy_with_mode,
    },
    gui::run_gui,
    model::build_model_artifacts,
    model::{ModelVariant, WeightMode},
    seed::{MergeStrategy, add_manual_addition, merge_seed_lists, reconcile_seed_lists},
    solver::{AbsurdleSuggestion, Solver},
};

#[derive(Parser, Debug)]
#[command(name = "maybe-wordle")]
#[command(about = "Weighted Wordle solver for current NYT behavior")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    SyncData,
    BuildModel,
    BuildOptimalPolicy {
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID)]
        model: String,
    },
    VerifyOptimalPolicy {
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID)]
        model: String,
        #[arg(long, default_value_t = false)]
        oracle: bool,
    },
    Gui,
    AddManual {
        word: String,
    },
    ReconcileSeeds,
    MergeSeeds {
        #[arg(long, default_value = "union")]
        strategy: String,
        #[arg(long, default_value_t = false)]
        apply: bool,
    },
    Suggest {
        #[arg(long = "guess")]
        guess: Vec<String>,
        #[arg(long = "feedback")]
        feedback: Vec<String>,
        #[arg(long, default_value_t = 10)]
        top: usize,
        #[arg(long)]
        date: Option<String>,
        #[arg(long, default_value = "predictive")]
        mode: String,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID)]
        model: String,
    },
    SolveInteractive {
        #[arg(long, default_value_t = 10)]
        top: usize,
        #[arg(long)]
        date: Option<String>,
        #[arg(long, default_value = "predictive")]
        mode: String,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID)]
        model: String,
    },
    ExplainState {
        #[arg(long = "guess")]
        guess: Vec<String>,
        #[arg(long = "feedback")]
        feedback: Vec<String>,
        #[arg(long, default_value_t = 5)]
        top: usize,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID)]
        model: String,
    },
    Backtest {
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        to: Option<String>,
        #[arg(long, default_value_t = 5)]
        top: usize,
        #[arg(long, default_value_t = false)]
        detailed: bool,
        #[arg(long, default_value_t = false)]
        failures_only: bool,
    },
    PredictiveAblations {
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        to: Option<String>,
        #[arg(long, default_value_t = 5)]
        top: usize,
    },
    BuildPredictiveOpener {
        #[arg(long)]
        date: Option<String>,
        #[arg(long, default_value = "weighted")]
        weight_mode: String,
        #[arg(long, default_value = "seed_plus_history")]
        variant: String,
    },
    BuildPredictiveReplies {
        #[arg(long)]
        date: Option<String>,
        #[arg(long, default_value = "weighted")]
        weight_mode: String,
        #[arg(long, default_value = "seed_plus_history")]
        variant: String,
    },
    Experiments {
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        to: Option<String>,
        #[arg(long, default_value_t = 5)]
        top: usize,
    },
    TunePrior,
    FitProxyWeights {
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        to: Option<String>,
    },
    Benchmark {
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[arg(long, default_value = "predictive")]
        mode: String,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID)]
        model: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SolverMode {
    Predictive,
    Absurdle,
    FormalOptimal,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let root = env::current_dir().context("failed to resolve current directory")?;
    let paths = ProjectPaths::new(root);
    paths.ensure_layout()?;
    let config = PriorConfig::load_or_create(&paths.config_prior)?;

    match cli.command {
        Command::SyncData => {
            let summary = sync_nyt_history(&paths, &config, Solver::today())?;
            println!(
                "synced {} entries from {} to {} (fetched {}, reverified {}, changed {})",
                summary.total,
                summary.first_date,
                summary.last_date,
                summary.fetched,
                summary.reverified,
                summary.changed
            );
            if !summary.changed_dates.is_empty() {
                println!(
                    "changed_dates={}",
                    summary
                        .changed_dates
                        .iter()
                        .map(|date| date.format("%Y-%m-%d").to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        Command::BuildModel => {
            let summary = build_model_artifacts(&paths, &config, Solver::today())?;
            println!(
                "built model with {} guesses, {} modeled answers, {} historical answers across {} daily rows",
                summary.guess_count,
                summary.answer_count,
                summary.historical_answers,
                summary.history_rows
            );
        }
        Command::BuildOptimalPolicy { model } => {
            let summary = build_optimal_policy(&paths, &model)?;
            println!(
                "model={} manifest={} states={} deduped_signatures={} bound_hits={} build_ms={} best_guess={} worst_case_depth={} expected_guesses={:.6}",
                summary.model_id,
                summary.manifest_hash,
                summary.solved_states,
                summary.deduped_signatures,
                summary.bound_hits,
                summary.build_millis,
                summary.root_best_guess,
                summary.root_objective.worst_case_depth,
                summary.root_objective.expected_guesses
            );
        }
        Command::VerifyOptimalPolicy { model, oracle } => {
            let summary = verify_optimal_policy_with_mode(
                &paths,
                &model,
                if oracle {
                    FormalVerificationMode::Oracle
                } else {
                    FormalVerificationMode::Certificate
                },
            )?;
            println!(
                "model={} manifest={} mode={} verified_cached_states={} verified_small_states={} verified_medium_states={}",
                summary.model_id,
                summary.manifest_hash,
                if summary.mode == FormalVerificationMode::Oracle {
                    "oracle"
                } else {
                    "certificate"
                },
                summary.verified_cached_states,
                summary.verified_small_states,
                summary.verified_medium_states
            );
        }
        Command::Gui => {
            run_gui(paths.root.clone())?;
        }
        Command::AddManual { word } => {
            add_manual_addition(&paths, &word)?;
            println!("added manual answer {}", word.to_ascii_lowercase());
        }
        Command::ReconcileSeeds => {
            let summary = reconcile_seed_lists(&paths)?;
            println!(
                "primary={} reference={} shared={} primary_only={} reference_only={}",
                summary.primary_count,
                summary.reference_count,
                summary.shared_count,
                summary.primary_only_count,
                summary.reference_only_count
            );
        }
        Command::MergeSeeds { strategy, apply } => {
            let strategy = parse_merge_strategy(&strategy)?;
            let summary = merge_seed_lists(&paths, strategy, apply)?;
            println!(
                "strategy={} primary={} reference={} merged={} output={} applied={}",
                summary.strategy.label(),
                summary.primary_count,
                summary.reference_count,
                summary.merged_count,
                summary.output_path,
                summary.applied_to_primary
            );
        }
        Command::Suggest {
            guess,
            feedback,
            top,
            date,
            mode,
            model,
        } => match parse_solver_mode(&mode)? {
            SolverMode::Predictive => {
                let observations = Solver::parse_observations(&guess, &feedback)?;
                let as_of = parse_or_today(date.as_deref())?;
                let solver = Solver::from_paths(&paths, &config)?;
                let state = solver.apply_history(as_of, &observations)?;
                println!(
                    "mode=predictive date={} surviving={} total_weight={:.4}",
                    as_of,
                    state.surviving.len(),
                    state.total_weight
                );
                for suggestion in solver.suggestions_for_history(as_of, &observations, top)? {
                    println!("{}", format_predictive_suggestion(&suggestion));
                }
            }
            SolverMode::Absurdle => {
                let observations = Solver::parse_observations(&guess, &feedback)?;
                let solver = Solver::from_paths(&paths, &config)?;
                let state = solver.absurdle_apply_history(&observations)?;
                println!("mode=absurdle surviving={}", state.surviving.len());
                for suggestion in solver.absurdle_suggestions(&observations, top)? {
                    println!("{}", format_absurdle_suggestion(&suggestion));
                }
            }
            SolverMode::FormalOptimal => {
                let observations = parse_formal_observations(&guess, &feedback)?;
                let runtime = FormalPolicyRuntime::load(&paths, &model)?;
                let state = runtime.apply_history(&observations)?;
                println!(
                    "mode=formal-optimal model={} manifest={} surviving={}",
                    runtime.manifest().model_id,
                    runtime.manifest().manifest_hash,
                    state.count()
                );
                for suggestion in runtime.suggest(&state, top)? {
                    println!(
                        "{} worst_case_depth={} expected_guesses={:.6} bucket_sizes={}",
                        suggestion.word,
                        suggestion.objective.worst_case_depth,
                        suggestion.objective.expected_guesses,
                        suggestion
                            .bucket_sizes
                            .iter()
                            .map(|size| size.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
            }
        },
        Command::SolveInteractive {
            top,
            date,
            mode,
            model,
        } => match parse_solver_mode(&mode)? {
            SolverMode::Predictive => {
                let as_of = parse_or_today(date.as_deref())?;
                let solver = Solver::from_paths(&paths, &config)?;
                let mut observations = Vec::new();

                loop {
                    let state = solver.apply_history(as_of, &observations)?;
                    println!(
                        "mode=predictive surviving={} total_weight={:.4}",
                        state.surviving.len(),
                        state.total_weight
                    );
                    for suggestion in solver.suggestions_for_history(as_of, &observations, top)? {
                        println!("{}", format_predictive_suggestion(&suggestion));
                    }

                    print!("guess (blank to stop): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let guess = read_line()?;
                    if guess.trim().is_empty() {
                        break;
                    }
                    let guess = match normalize_interactive_guess(&guess, |candidate| {
                        solver.has_guess(candidate)
                    }) {
                        Ok(guess) => guess,
                        Err(error) => {
                            println!("error: {error}");
                            continue;
                        }
                    };

                    print!("feedback (01020 or bgybb): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let feedback = read_line()?;
                    match try_append_observation(&observations, &guess, &feedback, |next| {
                        solver.apply_history(as_of, next).map(|_| ())
                    }) {
                        Ok(next) => observations = next,
                        Err(error) => println!("error: {error}"),
                    }
                }
            }
            SolverMode::Absurdle => {
                let solver = Solver::from_paths(&paths, &config)?;
                let mut observations = Vec::new();

                loop {
                    let state = solver.absurdle_apply_history(&observations)?;
                    println!("mode=absurdle surviving={}", state.surviving.len());
                    for suggestion in solver.absurdle_suggestions(&observations, top)? {
                        println!("{}", format_absurdle_suggestion(&suggestion));
                    }

                    print!("guess (blank to stop): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let guess = read_line()?;
                    if guess.trim().is_empty() {
                        break;
                    }
                    let guess = match normalize_interactive_guess(&guess, |candidate| {
                        solver.has_guess(candidate)
                    }) {
                        Ok(guess) => guess,
                        Err(error) => {
                            println!("error: {error}");
                            continue;
                        }
                    };

                    print!("feedback (01020 or bgybb): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let feedback = read_line()?;
                    match try_append_observation(&observations, &guess, &feedback, |next| {
                        solver.absurdle_apply_history(next).map(|_| ())
                    }) {
                        Ok(next) => observations = next,
                        Err(error) => println!("error: {error}"),
                    }
                }
            }
            SolverMode::FormalOptimal => {
                let runtime = FormalPolicyRuntime::load(&paths, &model)?;
                let mut observations = Vec::new();

                loop {
                    let state = runtime.apply_history(&observations)?;
                    println!(
                        "mode=formal-optimal model={} manifest={} surviving={}",
                        runtime.manifest().model_id,
                        runtime.manifest().manifest_hash,
                        state.count()
                    );
                    for suggestion in runtime.suggest(&state, top)? {
                        println!(
                            "{} worst_case_depth={} expected_guesses={:.6} bucket_sizes={}",
                            suggestion.word,
                            suggestion.objective.worst_case_depth,
                            suggestion.objective.expected_guesses,
                            suggestion
                                .bucket_sizes
                                .iter()
                                .map(|size| size.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                    }

                    print!("guess (blank to stop): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let guess = read_line()?;
                    if guess.trim().is_empty() {
                        break;
                    }
                    let guess = match normalize_interactive_guess(&guess, |candidate| {
                        runtime.has_guess(candidate)
                    }) {
                        Ok(guess) => guess,
                        Err(error) => {
                            println!("error: {error}");
                            continue;
                        }
                    };

                    print!("feedback (01020 or bgybb): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let feedback = read_line()?;
                    match try_append_observation(&observations, &guess, &feedback, |next| {
                        runtime.apply_history(next).map(|_| ())
                    }) {
                        Ok(next) => observations = next,
                        Err(error) => println!("error: {error}"),
                    }
                }
            }
        },
        Command::ExplainState {
            guess,
            feedback,
            top,
            model,
        } => {
            let observations = parse_formal_observations(&guess, &feedback)?;
            let runtime = FormalPolicyRuntime::load(&paths, &model)?;
            let state = runtime.apply_history(&observations)?;
            let explanation = runtime.explain_state(&state, top)?;
            println!(
                "model={} manifest={} surviving={} best_guess={} worst_case_depth={} expected_guesses={:.6} bucket_sizes={}",
                explanation.model_id,
                explanation.manifest_hash,
                explanation.surviving_answers,
                explanation.best_guess,
                explanation.objective.worst_case_depth,
                explanation.objective.expected_guesses,
                explanation
                    .bucket_sizes
                    .iter()
                    .map(|size| size.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            for tied in explanation.tied_moves {
                println!(
                    "candidate={} worst_case_depth={} expected_guesses={:.6} bucket_sizes={}",
                    tied.word,
                    tied.objective.worst_case_depth,
                    tied.objective.expected_guesses,
                    tied.bucket_sizes
                        .iter()
                        .map(|size| size.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        Command::Backtest {
            from,
            to,
            top,
            detailed,
            failures_only,
        } => {
            let solver = Solver::from_paths(&paths, &config)?;
            let (default_from, default_to) = Solver::latest_history_range(&paths)?
                .ok_or_else(|| anyhow!("run sync-data before backtesting"))?;
            let from = parse_date(from.as_deref())?.unwrap_or(default_from);
            let to = parse_date(to.as_deref())?.unwrap_or(default_to);
            if from > to {
                bail!("--from cannot be after --to");
            }
            let report = solver.backtest_detailed(from, to, top)?;
            let stats = &report.summary;
            println!(
                "games={} average_guesses={:.4} p95={} max={} failures={} coverage_gaps={}",
                stats.games,
                stats.average_guesses,
                stats.p95_guesses,
                stats.max_guesses,
                stats.failures,
                stats.coverage_gaps
            );
            if detailed {
                for run in report.runs.iter().filter(|run| {
                    if failures_only {
                        !run.solved
                    } else {
                        !run.solved || run.steps.len() >= 5
                    }
                }) {
                    println!(
                        "target={} date={} solved={} guesses={}",
                        run.target,
                        run.date,
                        run.solved,
                        run.steps.len()
                    );
                    for (index, step) in run.steps.iter().enumerate() {
                        println!(
                            "step={} guess={} feedback={} survivors={}=>{} regime={} danger_score={:.3} danger_escalated={} chosen_force_in_two={} alternative_force_in_two={}",
                            index + 1,
                            step.guess,
                            maybe_wordle::scoring::format_feedback_letters(step.feedback),
                            step.surviving_before,
                            step.surviving_after,
                            step.regime_used.label(),
                            step.danger_score,
                            step.danger_escalated,
                            step.chosen_force_in_two,
                            step.alternative_force_in_two
                        );
                        for suggestion in &step.top_suggestions {
                            println!(
                                "  top={} force_in_two={} worst_non_green_bucket_size={} largest_non_green_bucket_mass={:.5}{}{}",
                                suggestion.word,
                                suggestion.force_in_two,
                                suggestion.worst_non_green_bucket_size,
                                suggestion.largest_non_green_bucket_mass,
                                suggestion
                                    .proxy_cost
                                    .map(|value| format!(" proxy_cost={:.5}", value))
                                    .unwrap_or_default(),
                                suggestion
                                    .exact_cost
                                    .map(|value| format!(" exact_cost={:.5}", value))
                                    .or_else(|| suggestion
                                        .lookahead_cost
                                        .map(|value| format!(" lookahead_cost={:.5}", value)))
                                    .unwrap_or_default()
                            );
                        }
                    }
                }
            }
        }
        Command::PredictiveAblations { from, to, top } => {
            let (default_from, default_to) = Solver::latest_history_range(&paths)?
                .ok_or_else(|| anyhow!("run sync-data before predictive-ablations"))?;
            let from = parse_date(from.as_deref())?.unwrap_or(default_from);
            let to = parse_date(to.as_deref())?.unwrap_or(default_to);
            if from > to {
                bail!("--from cannot be after --to");
            }
            for row in Solver::predictive_ablation_report(&paths, &config, from, to, top)? {
                println!(
                    "label={} config={} mode={} variant={} games={} avg_guesses={:.4} p95={} max={} failures={} avg_target_prob={:.6} avg_target_rank={:.2} latency_p95_ms={:.3} session_cold_ms={:.3} session_warm_ms={:.3} lookahead_pool_ratio={:.3} exact_pool_ratio={:.3}",
                    row.label,
                    row.result.config_id,
                    row.result.mode.label(),
                    row.result.variant.label(),
                    row.result.backtest.games,
                    row.result.backtest.average_guesses,
                    row.result.backtest.p95_guesses,
                    row.result.backtest.max_guesses,
                    row.result.backtest.failures,
                    row.result.average_target_probability,
                    row.result.average_target_rank,
                    row.result.latency_p95_ms,
                    row.result.session_fallback_cold_ms,
                    row.result.session_fallback_warm_ms,
                    row.result.average_lookahead_pool_ratio,
                    row.result.average_exact_pool_ratio,
                );
            }
        }
        Command::BuildPredictiveOpener {
            date,
            weight_mode,
            variant,
        } => {
            let as_of = parse_or_today(date.as_deref())?;
            let solver = Solver::from_paths_with_settings(
                &paths,
                &config,
                parse_weight_mode(&weight_mode)?,
                parse_model_variant(&variant)?,
            )?;
            let summary = solver.build_predictive_opener_cache(as_of)?;
            println!(
                "mode={} variant={} as_of={} opener={} games={} avg_guesses={:.4} failures={} fingerprint={} path={}",
                solver.mode.label(),
                solver.variant.label(),
                summary.as_of,
                summary.opener,
                summary.games,
                summary.average_guesses,
                summary.failures,
                summary.config_fingerprint,
                summary.path.display()
            );
        }
        Command::BuildPredictiveReplies {
            date,
            weight_mode,
            variant,
        } => {
            let as_of = parse_or_today(date.as_deref())?;
            let solver = Solver::from_paths_with_settings(
                &paths,
                &config,
                parse_weight_mode(&weight_mode)?,
                parse_model_variant(&variant)?,
            )?;
            let summary = solver.build_predictive_reply_book(as_of)?;
            println!(
                "mode={} variant={} as_of={} opener={} replies={} fingerprint={} path={}",
                solver.mode.label(),
                solver.variant.label(),
                summary.as_of,
                summary.opener,
                summary.reply_count,
                summary.config_fingerprint,
                summary.path.display()
            );
        }
        Command::Experiments { from, to, top } => {
            let (default_from, default_to) = Solver::latest_history_range(&paths)?
                .ok_or_else(|| anyhow!("run sync-data before experiments"))?;
            let from = parse_date(from.as_deref())?.unwrap_or(default_from);
            let to = parse_date(to.as_deref())?.unwrap_or(default_to);
            if from > to {
                bail!("--from cannot be after --to");
            }
            for mode in [
                WeightMode::Uniform,
                WeightMode::CooldownOnly,
                WeightMode::Weighted,
            ] {
                for variant in [ModelVariant::SeedOnly, ModelVariant::SeedPlusHistory] {
                    let solver = Solver::from_paths_with_settings(&paths, &config, mode, variant)?;
                    let result = solver.experiment_report(from, to, top)?;
                    println!(
                        "config={} mode={} variant={} games={} avg_guesses={:.4} p95={} max={} failures={} avg_log_loss={:.6} avg_brier={:.6} avg_target_prob={:.6} avg_target_rank={:.2} latency_p95_ms={:.3} session_cold_ms={:.3} session_warm_ms={:.3} lookahead_pool_ratio={:.3} exact_pool_ratio={:.3}",
                        result.config_id,
                        result.mode.label(),
                        result.variant.label(),
                        result.backtest.games,
                        result.backtest.average_guesses,
                        result.backtest.p95_guesses,
                        result.backtest.max_guesses,
                        result.backtest.failures,
                        result.average_log_loss,
                        result.average_brier,
                        result.average_target_probability,
                        result.average_target_rank,
                        result.latency_p95_ms,
                        result.session_fallback_cold_ms,
                        result.session_fallback_warm_ms,
                        result.average_lookahead_pool_ratio,
                        result.average_exact_pool_ratio
                    );
                }
            }
        }
        Command::TunePrior => {
            let summary = Solver::tune_prior(&paths, &config)?;
            println!(
                "search_window={}..{} validation_window={}..{} current_avg_guesses={:.4} current_failures={} current_coverage_gaps={} current_log_loss={:.6} current_target_rank={:.2} current_latency_p95_ms={:.3} current_hard_case_avg_guesses={:.4} current_hard_case_failures={} current_regime_mix=proxy:{:.1}%/lookahead:{:.1}%/escalated_exact:{:.1}%/exact:{:.1}%",
                summary.search_window_start,
                summary.search_window_end,
                summary.validation_window_start,
                summary.validation_window_end,
                summary.current.average_guesses,
                summary.current.failures,
                summary.current.coverage_gaps,
                summary.current.average_log_loss,
                summary.current.average_target_rank,
                summary.current.latency_p95_ms,
                summary.current.hard_case_average_guesses,
                summary.current.hard_case_failures,
                summary.current.proxy_step_pct * 100.0,
                summary.current.lookahead_step_pct * 100.0,
                summary.current.escalated_exact_step_pct * 100.0,
                summary.current.exact_step_pct * 100.0
            );
            println!(
                "best_avg_guesses={:.4} best_failures={} best_coverage_gaps={} best_log_loss={:.6} best_target_rank={:.2} best_latency_p95_ms={:.3} best_hard_case_avg_guesses={:.4} best_hard_case_failures={} best_regime_mix=proxy:{:.1}%/lookahead:{:.1}%/escalated_exact:{:.1}%/exact:{:.1}%",
                summary.best.average_guesses,
                summary.best.failures,
                summary.best.coverage_gaps,
                summary.best.average_log_loss,
                summary.best.average_target_rank,
                summary.best.latency_p95_ms,
                summary.best.hard_case_average_guesses,
                summary.best.hard_case_failures,
                summary.best.proxy_step_pct * 100.0,
                summary.best.lookahead_step_pct * 100.0,
                summary.best.escalated_exact_step_pct * 100.0,
                summary.best.exact_step_pct * 100.0
            );
            println!("{}", summary.replacement_toml.trim_end());
        }
        Command::FitProxyWeights { from, to } => {
            let (history_start, history_end) = Solver::latest_history_range(&paths)?
                .ok_or_else(|| anyhow!("run sync-data before fitting proxy weights"))?;
            let default_from = history_end
                .checked_sub_days(chrono::Days::new(364))
                .map_or(history_start, |date| date.max(history_start));
            let from = parse_date(from.as_deref())?.unwrap_or(default_from);
            let to = parse_date(to.as_deref())?.unwrap_or(history_end);
            if from > to {
                bail!("--from cannot be after --to");
            }
            let solver = Solver::from_paths(&paths, &config)?;
            let summary = solver.fit_proxy_weights(from, to)?;
            println!(
                "rows={} states={} train_avg={:.4} validation_avg={:.4}",
                summary.row_count,
                summary.state_count,
                summary.training_average_guesses,
                summary.validation_average_guesses
            );
            println!("{}", summary.replacement_toml.trim_end());
        }
        Command::Benchmark { runs, mode, model } => {
            if runs == 0 {
                bail!("runs must be greater than 0");
            }
            match parse_solver_mode(&mode)? {
                SolverMode::Predictive => {
                    let solver = Solver::from_paths(&paths, &config)?;
                    let state = solver.initial_state(Solver::today());
                    let mut elapsed = std::time::Duration::ZERO;
                    for _ in 0..runs {
                        let started = std::time::Instant::now();
                        let _ = solver.suggestions(&state, 10)?;
                        elapsed += started.elapsed();
                    }
                    let average_ms = elapsed.as_secs_f64() * 1000.0 / runs as f64;
                    println!(
                        "mode=predictive runs={} surviving={} pattern_table_bytes={} average_ms={:.3}",
                        runs,
                        state.surviving.len(),
                        solver.pattern_table_bytes(),
                        average_ms
                    );
                }
                SolverMode::Absurdle => {
                    let solver = Solver::from_paths(&paths, &config)?;
                    let state = solver.absurdle_initial_state();
                    let mut elapsed = std::time::Duration::ZERO;
                    for _ in 0..runs {
                        let started = std::time::Instant::now();
                        let _ = solver.absurdle_suggestions_for_state(&state, 10)?;
                        elapsed += started.elapsed();
                    }
                    let average_ms = elapsed.as_secs_f64() * 1000.0 / runs as f64;
                    println!(
                        "mode=absurdle runs={} surviving={} pattern_table_bytes={} average_ms={:.3}",
                        runs,
                        state.surviving.len(),
                        solver.pattern_table_bytes(),
                        average_ms
                    );
                }
                SolverMode::FormalOptimal => {
                    let runtime = FormalPolicyRuntime::load(&paths, &model)?;
                    let state = runtime.initial_state();
                    let mut elapsed = std::time::Duration::ZERO;
                    for _ in 0..runs {
                        let started = std::time::Instant::now();
                        let _ = runtime.suggest(&state, 10)?;
                        elapsed += started.elapsed();
                    }
                    let average_ms = elapsed.as_secs_f64() * 1000.0 / runs as f64;
                    println!(
                        "mode=formal-optimal runs={} surviving={} states={} average_ms={:.3}",
                        runs,
                        state.count(),
                        runtime.metadata().solved_states,
                        average_ms
                    );
                }
            }
        }
    }

    Ok(())
}

fn parse_or_today(raw: Option<&str>) -> Result<NaiveDate> {
    Ok(parse_date(raw)?.unwrap_or_else(Solver::today))
}

fn parse_date(raw: Option<&str>) -> Result<Option<NaiveDate>> {
    raw.map(|value| {
        NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .with_context(|| format!("invalid date: {value}"))
    })
    .transpose()
}

fn read_line() -> Result<String> {
    let mut buffer = String::new();
    io::stdin()
        .read_line(&mut buffer)
        .context("failed to read stdin")?;
    Ok(buffer)
}

fn normalize_interactive_guess<F>(guess: &str, has_guess: F) -> std::result::Result<String, String>
where
    F: FnOnce(&str) -> bool,
{
    let normalized = guess.trim().to_ascii_lowercase();
    if !has_guess(&normalized) {
        return Err(format!("unknown guess: {}", normalized));
    }
    Ok(normalized)
}

fn try_append_observation<F>(
    observations: &[(String, u8)],
    guess: &str,
    feedback: &str,
    validate: F,
) -> std::result::Result<Vec<(String, u8)>, String>
where
    F: FnOnce(&[(String, u8)]) -> Result<()>,
{
    let pattern =
        maybe_wordle::scoring::parse_feedback(feedback).map_err(|error| error.to_string())?;
    let mut next = observations.to_vec();
    next.push((guess.to_ascii_lowercase(), pattern));
    validate(&next).map_err(|error| error.to_string())?;
    Ok(next)
}

fn format_predictive_suggestion(suggestion: &maybe_wordle::solver::Suggestion) -> String {
    let mut line = format!(
        "{} entropy={:.5} solve_prob={:.5} expected_remaining={:.3}",
        suggestion.word,
        suggestion.entropy,
        suggestion.solve_probability,
        suggestion.expected_remaining
    );
    if suggestion.force_in_two {
        line.push_str(" force_in_two=true");
    }
    if let Some(exact_cost) = suggestion.exact_cost {
        line.push_str(&format!(" exact_cost={:.5}", exact_cost));
    }
    line
}

fn format_absurdle_suggestion(suggestion: &AbsurdleSuggestion) -> String {
    format!(
        "{} worst_bucket={} second_worst_bucket={} multi_answer_buckets={} entropy={:.5}",
        suggestion.word,
        suggestion.largest_bucket_size,
        suggestion.second_largest_bucket_size,
        suggestion.multi_answer_bucket_count,
        suggestion.entropy
    )
}

fn parse_merge_strategy(raw: &str) -> Result<MergeStrategy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "union" => Ok(MergeStrategy::Union),
        "keep_primary" => Ok(MergeStrategy::KeepPrimary),
        _ => bail!("merge strategy must be one of: union, keep_primary"),
    }
}

fn parse_solver_mode(raw: &str) -> Result<SolverMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "predictive" => Ok(SolverMode::Predictive),
        "absurdle" => Ok(SolverMode::Absurdle),
        "formal-optimal" | "formal_optimal" | "formal" | "optimal" => Ok(SolverMode::FormalOptimal),
        _ => bail!("mode must be one of: predictive, absurdle, formal-optimal"),
    }
}

fn parse_weight_mode(raw: &str) -> Result<WeightMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "weighted" => Ok(WeightMode::Weighted),
        "uniform" => Ok(WeightMode::Uniform),
        "cooldown_only" | "cooldown-only" => Ok(WeightMode::CooldownOnly),
        _ => bail!("weight mode must be one of: weighted, uniform, cooldown_only"),
    }
}

fn parse_model_variant(raw: &str) -> Result<ModelVariant> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "seed_only" | "seed-only" => Ok(ModelVariant::SeedOnly),
        "seed_plus_history" | "seed-plus-history" | "seed" | "default" => {
            Ok(ModelVariant::SeedPlusHistory)
        }
        _ => bail!("variant must be one of: seed_only, seed_plus_history"),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use maybe_wordle::solver::AbsurdleSuggestion;

    use super::{
        format_absurdle_suggestion, format_predictive_suggestion, normalize_interactive_guess,
        parse_solver_mode, try_append_observation,
    };

    #[test]
    fn try_append_observation_rejects_invalid_feedback_without_mutation() {
        let observations = vec![("crane".to_string(), 0)];
        let result = try_append_observation(&observations, "slate", "oops", |_| Ok(()));
        assert!(result.is_err());
        assert_eq!(observations.len(), 1);
    }

    #[test]
    fn try_append_observation_rejects_contradictions_without_mutation() {
        let observations = vec![("crane".to_string(), 0)];
        let result = try_append_observation(&observations, "slate", "00000", |_| {
            Err(anyhow!("no answers remain"))
        });
        assert!(result.is_err());
        assert_eq!(observations.len(), 1);
    }

    #[test]
    fn try_append_observation_commits_valid_observation() {
        let observations = vec![("crane".to_string(), 0)];
        let result = try_append_observation(&observations, "slate", "00000", |_| Ok(()))
            .expect("valid observation");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], observations[0]);
        assert_eq!(result[1].0, "slate");
    }

    #[test]
    fn normalize_interactive_guess_rejects_unknown_guess() {
        let result = normalize_interactive_guess("slate", |guess| guess == "crane");
        assert_eq!(result.expect_err("must fail"), "unknown guess: slate");
    }

    #[test]
    fn predictive_suggestion_format_includes_force_in_two_marker() {
        let formatted = format_predictive_suggestion(&maybe_wordle::solver::Suggestion {
            word: "crane".into(),
            entropy: 4.0,
            solve_probability: 0.2,
            expected_remaining: 3.0,
            force_in_two: true,
            known_absent_letter_hits: 0,
            worst_non_green_bucket_size: 1,
            largest_non_green_bucket_mass: 0.05,
            large_non_green_bucket_count: 0,
            dangerous_mass_bucket_count: 0,
            non_green_mass_in_large_buckets: 0.0,
            proxy_cost: Some(2.0),
            large_state_score: Some(1.0),
            posterior_answer_probability: 0.1,
            lookahead_cost: None,
            exact_cost: Some(2.5),
        });
        assert!(formatted.contains("force_in_two=true"));
        assert!(formatted.contains("exact_cost=2.50000"));
    }

    #[test]
    fn absurdle_suggestion_format_includes_worst_bucket_metrics() {
        let formatted = format_absurdle_suggestion(&AbsurdleSuggestion {
            word: "crane".into(),
            entropy: 3.5,
            largest_bucket_size: 8,
            second_largest_bucket_size: 3,
            multi_answer_bucket_count: 2,
        });
        assert!(formatted.contains("worst_bucket=8"));
        assert!(formatted.contains("second_worst_bucket=3"));
    }

    #[test]
    fn parse_solver_mode_accepts_absurdle() {
        assert!(matches!(
            parse_solver_mode("absurdle").expect("mode"),
            super::SolverMode::Absurdle
        ));
    }
}
