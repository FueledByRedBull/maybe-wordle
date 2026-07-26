use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use maybe_wordle::{
    SOLVER_THREAD_STACK_BYTES,
    atomic_file::atomic_write,
    config::PriorConfig,
    data::{ProjectPaths, SyncSummary, sync_nyt_history},
    experiments::{
        DateRange, RollingOriginConfig, StudyFoldSelection, StudySearchStrategy, StudySpec,
        StudyStage, build_rolling_origin_plan, predictive_parameter_registry,
    },
    formal::{
        DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime, FormalScaleRequest, FormalVerificationMode,
        benchmark_formal_scale, build_optimal_policy,
        parse_observations as parse_formal_observations, verify_optimal_policy_with_mode,
    },
    gui::run_gui,
    model::build_model_artifacts,
    model::{ModelVariant, WeightMode},
    predictive::{PredictiveSuggestRequest, PredictiveSuggestResponse, PredictiveSuggestionMode},
    seed::{MergeStrategy, add_manual_addition, merge_seed_lists, reconcile_seed_lists},
    solver::{AbsurdleSuggestion, EvidenceResourceBudget, SearchRegretRequest, Solver},
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
    #[command(about = "Fetch NYT daily answer JSON into data/raw and reverify recent synced dates")]
    SyncData {
        #[arg(
            long,
            default_value_t = false,
            help = "Fail if any date could not be synced after retries"
        )]
        strict: bool,
    },
    #[command(about = "Build modeled answer CSVs under data/derived")]
    BuildModel,
    #[command(about = "Build formal optimal-policy artifacts into data/formal")]
    BuildOptimalPolicy {
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID, help = "Formal model id to build")]
        model: String,
    },
    #[command(about = "Verify a formal optimal-policy artifact by certificate or oracle checks")]
    VerifyOptimalPolicy {
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID, help = "Formal model id to verify")]
        model: String,
        #[arg(
            long,
            default_value_t = false,
            help = "Use the slower oracle verifier instead of certificate mode"
        )]
        oracle: bool,
    },
    #[command(
        about = "Run or resume a resource-bounded formal scale projection on pinned prefixes"
    )]
    FormalScale {
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "3,4,5,6,8,10,12",
            help = "Strictly increasing pinned answer counts, comma separated; capped at 16"
        )]
        answer_counts: Vec<usize>,
        #[arg(
            long,
            default_value_t = 0,
            help = "Pinned guess-prefix size, including every selected answer; 0 uses the complete pinned guess list"
        )]
        guess_limit: usize,
        #[arg(long, default_value_t = 1800)]
        maximum_seconds: u64,
        #[arg(long, default_value_t = 4096)]
        maximum_memory_mb: u64,
        #[arg(long, default_value_t = 4096)]
        maximum_disk_mb: u64,
        #[arg(
            long,
            default_value = "benchmarks/formal/scale-v2.json",
            help = "Atomic resumable machine-readable scale report"
        )]
        output: PathBuf,
    },
    #[command(about = "Open the desktop GUI")]
    Gui,
    #[command(about = "Append a manually curated answer candidate to the seed list")]
    AddManual { word: String },
    #[command(about = "Compare the primary and reference seed answer lists")]
    ReconcileSeeds,
    #[command(about = "Merge the primary and reference seed answer lists")]
    MergeSeeds {
        #[arg(
            long,
            default_value = "union",
            help = "Seed merge strategy: union or keep_primary"
        )]
        strategy: String,
        #[arg(
            long,
            default_value_t = false,
            help = "Write the merged list back to the primary seed file"
        )]
        apply: bool,
    },
    #[command(
        about = "Suggest the next move for predictive Wordle, Absurdle, or formal-optimal mode"
    )]
    Suggest {
        #[arg(
            long = "guess",
            help = "Applied guesses in order; repeat once per committed row"
        )]
        guess: Vec<String>,
        #[arg(
            long = "feedback",
            help = "Feedback per guess in 01020 or bgybb form; repeat to match --guess"
        )]
        feedback: Vec<String>,
        #[arg(
            long,
            default_value_t = 10,
            help = "Maximum number of suggestions to print"
        )]
        top: usize,
        #[arg(long, help = "Predictive as-of date in YYYY-MM-DD; defaults to today")]
        date: Option<String>,
        #[arg(
            long,
            default_value = "predictive",
            help = "Solver mode: predictive, absurdle, or formal-optimal"
        )]
        mode: String,
        #[arg(
            long,
            default_value_t = false,
            help = "Require predictive guesses to satisfy hard mode constraints"
        )]
        hard: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Allow slower predictive live-session promotion when disk artifacts are missing"
        )]
        live_fallback: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Return the fast proxy preview without lookahead or exact refinement"
        )]
        proxy_preview: bool,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID, help = "Formal model id when --mode formal-optimal is used")]
        model: String,
    },
    #[command(
        about = "Run an interactive suggestion loop in predictive, Absurdle, or formal-optimal mode"
    )]
    SolveInteractive {
        #[arg(
            long,
            default_value_t = 10,
            help = "Maximum number of suggestions to print each turn"
        )]
        top: usize,
        #[arg(long, help = "Predictive as-of date in YYYY-MM-DD; defaults to today")]
        date: Option<String>,
        #[arg(
            long,
            default_value = "predictive",
            help = "Solver mode: predictive, absurdle, or formal-optimal"
        )]
        mode: String,
        #[arg(
            long,
            default_value_t = false,
            help = "Require predictive guesses to satisfy hard mode constraints"
        )]
        hard: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Allow slower predictive live-session promotion when disk artifacts are missing"
        )]
        live_fallback: bool,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID, help = "Formal model id when --mode formal-optimal is used")]
        model: String,
    },
    #[command(about = "Explain a formal-optimal state after a sequence of guesses and feedback")]
    ExplainState {
        #[arg(
            long = "guess",
            help = "Applied guesses in order; repeat once per committed row"
        )]
        guess: Vec<String>,
        #[arg(
            long = "feedback",
            help = "Feedback per guess in 01020 or bgybb form; repeat to match --guess"
        )]
        feedback: Vec<String>,
        #[arg(
            long,
            default_value_t = 5,
            help = "Maximum number of tied candidates to print"
        )]
        top: usize,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID, help = "Formal model id to explain")]
        model: String,
    },
    #[command(about = "Backtest the predictive solver across a synced NYT date range")]
    Backtest {
        #[arg(long, help = "Optional alternate prior TOML config")]
        config: Option<PathBuf>,
        #[arg(
            long,
            help = "Backtest start date in YYYY-MM-DD; defaults to earliest synced date"
        )]
        from: Option<String>,
        #[arg(
            long,
            help = "Backtest end date in YYYY-MM-DD; defaults to latest synced date"
        )]
        to: Option<String>,
        #[arg(
            long,
            default_value_t = 5,
            help = "Number of suggestions tracked per step in detailed output"
        )]
        top: usize,
        #[arg(
            long,
            default_value_t = false,
            help = "Print per-game and per-step detail"
        )]
        detailed: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "With --detailed, print only failed runs"
        )]
        failures_only: bool,
    },
    #[command(about = "Compare predictive ablation configurations over a synced date range")]
    PredictiveAblations {
        #[arg(
            long,
            help = "Evaluation start date in YYYY-MM-DD; defaults to earliest synced date"
        )]
        from: Option<String>,
        #[arg(
            long,
            help = "Evaluation end date in YYYY-MM-DD; defaults to latest synced date"
        )]
        to: Option<String>,
        #[arg(
            long,
            default_value_t = 5,
            help = "Top suggestion count used during evaluation"
        )]
        top: usize,
    },
    #[command(about = "Evaluate an alternate prior config file against a fixed date window")]
    EvaluateLiveConfig {
        #[arg(long, help = "Path to the candidate prior TOML file to evaluate")]
        config: String,
        #[arg(long, help = "Evaluation start date in YYYY-MM-DD")]
        from: String,
        #[arg(long, help = "Evaluation end date in YYYY-MM-DD")]
        to: String,
        #[arg(
            long,
            default_value_t = 5,
            help = "Top suggestion count used during evaluation"
        )]
        top: usize,
        #[arg(
            long,
            default_value_t = false,
            help = "Emit the evaluation as JSON instead of a text summary"
        )]
        json: bool,
    },
    #[command(about = "Report states where aggressive three-guess play closes a gap")]
    ThreeGuessGap {
        #[arg(long, help = "Evaluation start date in YYYY-MM-DD")]
        from: String,
        #[arg(long, help = "Evaluation end date in YYYY-MM-DD")]
        to: String,
        #[arg(
            long,
            default_value_t = 5,
            help = "Top suggestion count used during evaluation"
        )]
        top: usize,
    },
    #[command(about = "Compare specified opener words against four-guess targets")]
    FourGuessOpeners {
        #[arg(long, help = "Evaluation start date in YYYY-MM-DD")]
        from: String,
        #[arg(long, help = "Evaluation end date in YYYY-MM-DD")]
        to: String,
        #[arg(
            long,
            default_value_t = 5,
            help = "Top suggestion count used during evaluation"
        )]
        top: usize,
        #[arg(
            long = "opener",
            help = "Candidate opener word; repeat to compare multiple openers"
        )]
        opener: Vec<String>,
    },
    #[command(about = "Build a predictive opener artifact for one date and one model variant")]
    BuildPredictiveOpener {
        #[arg(long, help = "Artifact date in YYYY-MM-DD; defaults to today")]
        date: Option<String>,
        #[arg(
            long,
            default_value = "weighted",
            help = "Answer-weight model: weighted, uniform, or cooldown_only"
        )]
        weight_mode: String,
        #[arg(
            long,
            default_value = "seed_plus_history",
            help = "Model variant: seed_only or seed_plus_history"
        )]
        variant: String,
    },
    #[command(about = "Build a predictive reply-book artifact for one date and one model variant")]
    BuildPredictiveReplies {
        #[arg(long, help = "Artifact date in YYYY-MM-DD; defaults to today")]
        date: Option<String>,
        #[arg(
            long,
            default_value = "weighted",
            help = "Answer-weight model: weighted, uniform, or cooldown_only"
        )]
        weight_mode: String,
        #[arg(
            long,
            default_value = "seed_plus_history",
            help = "Model variant: seed_only or seed_plus_history"
        )]
        variant: String,
    },
    #[command(about = "Run the predictive experiment matrix over a synced date range")]
    Experiments {
        #[arg(
            long,
            help = "Evaluation start date in YYYY-MM-DD; defaults to earliest synced date"
        )]
        from: Option<String>,
        #[arg(
            long,
            help = "Evaluation end date in YYYY-MM-DD; defaults to latest synced date"
        )]
        to: Option<String>,
        #[arg(
            long,
            default_value_t = 5,
            help = "Top suggestion count used during evaluation"
        )]
        top: usize,
    },
    #[command(about = "Print the canonical rolling-origin and sealed-test evaluation plan as JSON")]
    EvaluationPlan {
        #[arg(long, default_value_t = 365)]
        minimum_training_days: u64,
        #[arg(long, default_value_t = 30)]
        validation_days: u64,
        #[arg(long, default_value_t = 30)]
        step_days: u64,
        #[arg(long, default_value_t = 30)]
        sealed_test_days: u64,
        #[arg(long, default_value_t = 12)]
        maximum_folds: usize,
    },
    #[command(about = "Print the complete predictive parameter registry as JSON")]
    ParameterRegistry,
    #[command(about = "Run or resume a deterministic rolling-origin predictive study")]
    StudyRun {
        #[arg(long, help = "Stable study name recorded in trial identities")]
        name: String,
        #[arg(
            long,
            help = "Optional TOML base config; defaults to config/prior.toml"
        )]
        base_config: Option<PathBuf>,
        #[arg(
            long,
            default_value = "calibration",
            help = "Study stage or typed cohort; use proxy-ranker/solve-policy for aggregate compatibility or proxy-core, proxy-risk, proxy-small-state, search-routing, search-exact, search-coverage, search-lookahead, search-pool, search-danger, and search-penalty for coherent studies"
        )]
        stage: String,
        #[arg(
            long,
            default_value_t = 16,
            help = "Total deterministic candidates including the baseline"
        )]
        trials: usize,
        #[arg(
            long,
            default_value_t = 1,
            help = "Maximum concurrent candidate evaluations"
        )]
        jobs: usize,
        #[arg(long, default_value_t = 20260315, help = "Deterministic study seed")]
        seed: u64,
        #[arg(
            long,
            default_value = "low-discrepancy",
            help = "Candidate strategy: grid, low-discrepancy, random, local-refinement, or model-based"
        )]
        strategy: String,
        #[arg(
            long,
            default_value_t = 12,
            help = "Maximum chronological validation folds evaluated per candidate"
        )]
        maximum_validation_folds: usize,
        #[arg(
            long,
            default_value_t = 3,
            help = "Validation folds in the first successive-halving rung"
        )]
        initial_validation_folds: usize,
        #[arg(
            long,
            default_value_t = 3,
            help = "Successive-halving reduction factor between fidelity rungs"
        )]
        reduction_factor: usize,
        #[arg(
            long,
            default_value_t = 7200,
            help = "Candidate wall-clock budget in seconds, checked between games/folds"
        )]
        maximum_trial_seconds: u64,
        #[arg(
            long,
            default_value_t = 4096,
            help = "Hard process peak-working-set budget in MiB"
        )]
        maximum_memory_mb: u64,
        #[arg(long, help = "Resumable JSON study-state path")]
        state: PathBuf,
        #[arg(
            long,
            help = "Pause cooperatively between games/folds while this file exists"
        )]
        cancel_file: Option<PathBuf>,
        #[arg(long, help = "Optional TOML path for the current best config")]
        output_config: Option<PathBuf>,
        #[arg(
            long,
            default_value_t = 5,
            help = "Top suggestions retained during solve evaluation"
        )]
        top: usize,
    },
    #[command(about = "Search for a better predictive prior policy and print a replacement TOML")]
    TunePrior,
    #[command(
        about = "Tune registered proxy-ranker weights through the common rolling study runner"
    )]
    FitProxyWeights,
    #[command(
        about = "Measure production, proxy, and bounded-lookahead regret against exhaustive search"
    )]
    SearchRegret {
        #[arg(long, help = "Optional alternate prior TOML config")]
        config: Option<PathBuf>,
        #[arg(long, help = "Historical audit start date in YYYY-MM-DD")]
        from: String,
        #[arg(long, help = "Historical audit end date in YYYY-MM-DD")]
        to: String,
        #[arg(long, default_value_t = 3)]
        minimum_survivors: usize,
        #[arg(long, default_value_t = 6)]
        maximum_survivors: usize,
        #[arg(long, default_value_t = 16)]
        maximum_states: usize,
        #[arg(
            long,
            default_value_t = 1800,
            help = "Hard wall-clock budget in seconds, checked between games and states"
        )]
        maximum_seconds: u64,
        #[arg(long, help = "Versioned JSON report output path")]
        output: PathBuf,
    },
    #[command(about = "Benchmark predictive, Absurdle, or formal-optimal suggestion latency")]
    Benchmark {
        #[arg(
            long,
            default_value_t = 3,
            help = "Number of repeated suggestion runs to average"
        )]
        runs: usize,
        #[arg(
            long,
            default_value = "predictive",
            help = "Solver mode: predictive, absurdle, or formal-optimal"
        )]
        mode: String,
        #[arg(long, default_value = DEFAULT_FORMAL_MODEL_ID, help = "Formal model id when --mode formal-optimal is used")]
        model: String,
    },
    #[command(about = "Generate versioned predictive development evidence as JSON and Markdown")]
    BenchmarkEvidence {
        #[arg(long, help = "Development evaluation start date in YYYY-MM-DD")]
        from: String,
        #[arg(long, help = "Development evaluation end date in YYYY-MM-DD")]
        to: String,
        #[arg(long, default_value_t = 5)]
        top: usize,
        #[arg(
            long,
            default_value_t = 3600,
            help = "Hard evidence-generation wall-clock budget in seconds"
        )]
        maximum_seconds: u64,
        #[arg(
            long,
            default_value_t = 4096,
            help = "Hard process peak-working-set budget in MiB"
        )]
        maximum_memory_mb: u64,
        #[arg(long, help = "Versioned JSON evidence output path")]
        output: PathBuf,
        #[arg(long, help = "Generated README fragment output path")]
        markdown_output: PathBuf,
    },
    #[command(about = "Update or verify README evidence from a versioned benchmark artifact")]
    BenchmarkEvidenceDocs {
        #[arg(long, help = "Versioned predictive evidence JSON path")]
        evidence: PathBuf,
        #[arg(long, default_value = "docs/generated/predictive-evidence.md")]
        markdown_output: PathBuf,
        #[arg(long, default_value = "README.md")]
        readme: PathBuf,
        #[arg(
            long,
            default_value_t = false,
            help = "Atomically update both documentation files instead of checking them"
        )]
        update: bool,
    },
    #[command(
        about = "Compare a candidate config with the default over every rolling development fold"
    )]
    RollingCompare {
        #[arg(
            long,
            help = "Optional baseline TOML config; defaults to config/prior.toml"
        )]
        baseline_config: Option<PathBuf>,
        #[arg(long, default_value = "current_default")]
        baseline_label: String,
        #[arg(long, help = "Candidate TOML config path")]
        candidate_config: PathBuf,
        #[arg(long, default_value = "candidate")]
        candidate_label: String,
        #[arg(long, default_value_t = 5)]
        top: usize,
        #[arg(long, help = "Versioned rolling comparison JSON output")]
        output: PathBuf,
        #[arg(
            long,
            help = "Optional prior rolling-comparison JSON whose matching default baseline can be reused"
        )]
        baseline_artifact: Option<PathBuf>,
    },
    #[command(
        about = "Freeze an eligible 12-fold development winner without opening the sealed test"
    )]
    FreezeCandidate {
        #[arg(long, help = "Complete candidate TOML config to freeze")]
        config: PathBuf,
        #[arg(
            long,
            help = "Current rolling-comparison JSON proving eligibility against its parent"
        )]
        comparison: PathBuf,
        #[arg(long, default_value = "benchmarks/predictive/frozen-candidate-v1.json")]
        output: PathBuf,
    },
    #[command(about = "Evaluate one frozen candidate on the sealed test exactly once")]
    EvaluateSealed {
        #[arg(long, default_value = "benchmarks/predictive/frozen-candidate-v1.json")]
        frozen: PathBuf,
        #[arg(long, default_value = "benchmarks/predictive/sealed-test-v1.json")]
        output: PathBuf,
    },
    #[command(about = "Update or verify rolling-comparison README evidence")]
    RollingEvidenceDocs {
        #[arg(
            long,
            required = true,
            help = "Rolling comparison JSON; repeat for each candidate"
        )]
        comparison: Vec<PathBuf>,
        #[arg(long, default_value = "docs/generated/rolling-evidence.md")]
        markdown_output: PathBuf,
        #[arg(long, default_value = "README.md")]
        readme: PathBuf,
        #[arg(long, default_value_t = false)]
        update: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SolverMode {
    Predictive,
    Absurdle,
    FormalOptimal,
}

fn main() {
    let result = configure_global_solver_pool().and_then(|()| {
        if env::args_os().count() == 1 {
            resolve_project_root().and_then(run_gui)
        } else {
            run_cli_on_sized_stack()
        }
    });
    if let Err(error) = result {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn configure_global_solver_pool() -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .stack_size(SOLVER_THREAD_STACK_BYTES)
        .build_global()
        .context("failed to configure the global solver worker pool")
}

fn run_cli_on_sized_stack() -> Result<()> {
    thread::Builder::new()
        .name("maybe-wordle-cli".to_string())
        .stack_size(SOLVER_THREAD_STACK_BYTES)
        .spawn(run)
        .context("failed to start CLI worker")?
        .join()
        .map_err(|_| anyhow!("CLI worker panicked"))?
}

fn run() -> Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    let root = resolve_project_root()?;
    let cli = Cli::parse_from(args);
    let paths = ProjectPaths::new(root);
    paths.ensure_layout()?;
    let config = PriorConfig::load_or_create(&paths.config_prior)?;

    match cli.command {
        Command::SyncData { strict } => {
            let summary = sync_nyt_history(&paths, &config, Solver::today())?;
            println!("{}", format_sync_summary(&summary));
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
            if summary.partial_sync {
                println!(
                    "failed_dates={}",
                    summary
                        .failed_dates
                        .iter()
                        .map(|date| date.format("%Y-%m-%d").to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
                if summary.total > summary.fetched {
                    println!(
                        "preserved_existing_history=true last_successful_date={}",
                        summary
                            .last_successful_date
                            .map(|date| date.format("%Y-%m-%d").to_string())
                            .unwrap_or_else(|| "none".to_string())
                    );
                }
            }
            enforce_sync_policy(strict, &summary)?;
        }
        Command::BuildModel => {
            let summary = build_model_artifacts(&paths, &config, Solver::today())?;
            println!(
                "built model with {} guesses, {} primary answers, {} dormant fallback answers, {} historical answers across {} daily rows",
                summary.guess_count,
                summary.answer_count,
                summary.fallback_answer_count,
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
                "model={} manifest={} mode={} status={} certificate_format={} verified_cached_states={} verified_small_states={} verified_medium_states={}",
                summary.model_id,
                summary.manifest_hash,
                if summary.mode == FormalVerificationMode::Oracle {
                    "oracle"
                } else {
                    "certificate"
                },
                if summary.mode == FormalVerificationMode::Certificate {
                    "proved_for_exact_manifest"
                } else {
                    "oracle_cross_check_only"
                },
                summary.certificate_format_version,
                summary.verified_cached_states,
                summary.verified_small_states,
                summary.verified_medium_states
            );
        }
        Command::FormalScale {
            answer_counts,
            guess_limit,
            maximum_seconds,
            maximum_memory_mb,
            maximum_disk_mb,
            output,
        } => {
            let report = benchmark_formal_scale(
                &paths,
                &FormalScaleRequest {
                    answer_counts,
                    guess_limit,
                    maximum_seconds,
                    maximum_memory_mb,
                    maximum_disk_mb,
                    output: output.clone(),
                },
            )?;
            println!(
                "formal_scale_points={} completed={} stopped_reason={} projected_full_log10_seconds={} projected_full_log10_certificate_bytes={} output={}",
                report.points.len(),
                report.completed,
                report.stopped_reason.as_deref().unwrap_or("none"),
                report
                    .full_model_projection
                    .projected_log10_seconds
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "unavailable".to_string()),
                report
                    .full_model_projection
                    .projected_log10_certificate_bytes
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "unavailable".to_string()),
                output.display()
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
            hard,
            live_fallback,
            proxy_preview,
            model,
        } => match parse_solver_mode(&mode)? {
            SolverMode::Predictive => {
                let observations = Solver::parse_observations(&guess, &feedback)?;
                let as_of = parse_or_today(date.as_deref())?;
                warn_predictive_history_range(&paths, as_of)?;
                let solver = Solver::from_paths(&paths, &config)?;
                let predictive_mode = predictive_cli_mode(live_fallback);
                let request = PredictiveSuggestRequest {
                    as_of,
                    observations: &observations,
                    top,
                    hard_mode: hard,
                    force_in_two_only: false,
                    mode: predictive_mode,
                };
                let response = if proxy_preview {
                    solver.suggest_predictive_proxy_preview(request)?
                } else {
                    solver.suggest_predictive(request)?
                };
                for warning in
                    predictive_warning_lines(as_of, &observations, predictive_mode, &response)
                {
                    eprintln!("warning: {warning}");
                }
                println!(
                    "mode=predictive model={} manifest={} history_snapshot={} history_hash={} artifact_status={} promoted_from_cache={} date={} surviving={} total_weight={:.4}",
                    response.model_version,
                    response.model_manifest_hash,
                    response
                        .history_snapshot_date
                        .map(|date| date.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    response.history_snapshot_hash,
                    response.artifact_state.banner_text(),
                    response.promotion_source.is_some(),
                    as_of,
                    response.state.surviving,
                    response.state.effective_total_weight
                );
                if let Some(mode) = response.state.recovery_mode_used {
                    println!("recovery_mode={}", mode.label());
                }
                for suggestion in response.suggestions {
                    println!("{}", format_predictive_suggestion(&suggestion));
                }
            }
            SolverMode::Absurdle => {
                if proxy_preview {
                    bail!("--proxy-preview is only supported in predictive mode");
                }
                reject_hard_mode_for_non_predictive(hard, "absurdle")?;
                reject_live_fallback_for_non_predictive(live_fallback, "absurdle")?;
                let observations = Solver::parse_observations(&guess, &feedback)?;
                let solver = Solver::from_paths(&paths, &config)?;
                let state = solver.absurdle_apply_history(&observations)?;
                println!("mode=absurdle surviving={}", state.surviving.len());
                for suggestion in solver.absurdle_suggestions(&observations, top)? {
                    println!("{}", format_absurdle_suggestion(&suggestion));
                }
            }
            SolverMode::FormalOptimal => {
                if proxy_preview {
                    bail!("--proxy-preview is only supported in predictive mode");
                }
                reject_hard_mode_for_non_predictive(hard, "formal-optimal")?;
                reject_live_fallback_for_non_predictive(live_fallback, "formal-optimal")?;
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
            hard,
            live_fallback,
            model,
        } => match parse_solver_mode(&mode)? {
            SolverMode::Predictive => {
                let as_of = parse_or_today(date.as_deref())?;
                warn_predictive_history_range(&paths, as_of)?;
                let solver = Solver::from_paths(&paths, &config)?;
                let mut observations = Vec::new();
                let predictive_mode = predictive_cli_mode(live_fallback);

                loop {
                    let response = solver.suggest_predictive(PredictiveSuggestRequest {
                        as_of,
                        observations: &observations,
                        top,
                        hard_mode: hard,
                        force_in_two_only: false,
                        mode: predictive_mode,
                    })?;
                    for warning in
                        predictive_warning_lines(as_of, &observations, predictive_mode, &response)
                    {
                        eprintln!("warning: {warning}");
                    }
                    println!(
                        "mode=predictive surviving={} total_weight={:.4}",
                        response.state.surviving, response.state.effective_total_weight
                    );
                    if let Some(mode) = response.state.recovery_mode_used {
                        println!("recovery_mode={}", mode.label());
                    }
                    for suggestion in response.suggestions {
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
                    if hard && let Some(error) = solver.hard_mode_violation(&observations, &guess) {
                        println!("error: {error}");
                        continue;
                    }

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
                reject_hard_mode_for_non_predictive(hard, "absurdle")?;
                reject_live_fallback_for_non_predictive(live_fallback, "absurdle")?;
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
                reject_hard_mode_for_non_predictive(hard, "formal-optimal")?;
                reject_live_fallback_for_non_predictive(live_fallback, "formal-optimal")?;
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
            config: backtest_config,
            from,
            to,
            top,
            detailed,
            failures_only,
        } => {
            let backtest_config = backtest_config
                .as_deref()
                .map(PriorConfig::load)
                .transpose()?
                .unwrap_or_else(|| config.clone());
            let solver = Solver::from_paths(&paths, &backtest_config)?;
            let (default_from, default_to) = Solver::latest_history_range(&paths)?
                .ok_or_else(|| anyhow!("run sync-data before backtesting"))?;
            let from = parse_date(from.as_deref())?.unwrap_or(default_from);
            let to = parse_date(to.as_deref())?.unwrap_or(default_to);
            if from > to {
                bail!("--from cannot be after --to");
            }
            let report = solver.backtest_detailed(from, to, top)?;
            let stats = &report.summary;
            let canonical = &stats.canonical;
            println!(
                "scheduled_games={} modeled_games={} solved_games={} unsolved_games={} coverage_gaps={} coverage_rate={:.6} solve_rate={:.6} conditional_mean_guesses={:.4} conditional_mean_guesses_ci95={:.4}..{:.4} all_game_penalized_mean_guesses={:.4} all_game_penalized_mean_guesses_ci95={:.4}..{:.4} failure_penalty_guesses={:.1} median={:.2} p90={} p95={} max={} solved_distribution={}",
                canonical.scheduled_games,
                canonical.modeled_games,
                canonical.solved_games,
                canonical.unsolved_games,
                canonical.coverage_gaps,
                canonical.coverage_rate,
                canonical.solve_rate,
                canonical.conditional_mean_guesses,
                canonical.conditional_mean_guesses_ci95.lower,
                canonical.conditional_mean_guesses_ci95.upper,
                canonical.all_game_penalized_mean_guesses,
                canonical.all_game_penalized_mean_guesses_ci95.lower,
                canonical.all_game_penalized_mean_guesses_ci95.upper,
                canonical.failure_penalty_guesses,
                canonical.median_guesses,
                canonical.p90_guesses,
                canonical.p95_guesses,
                canonical.max_guesses,
                canonical
                    .solved_in_guess_counts
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
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
                                    .map(|value| format!(" candidate_pool_exact_cost={:.5}", value))
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
        Command::EvaluateLiveConfig {
            config: config_path,
            from,
            to,
            top,
            json,
        } => {
            let evaluation_config = PriorConfig::load(std::path::Path::new(&config_path))?;
            let from = NaiveDate::parse_from_str(&from, "%Y-%m-%d")
                .with_context(|| format!("invalid date '{}'", from))?;
            let to = NaiveDate::parse_from_str(&to, "%Y-%m-%d")
                .with_context(|| format!("invalid date '{}'", to))?;
            if from > to {
                bail!("--from cannot be after --to");
            }
            let evaluation =
                Solver::evaluate_live_config(&paths, &evaluation_config, from, to, top)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&evaluation)
                        .context("failed to serialize live config evaluation")?
                );
            } else {
                println!(
                    "avg_guesses={:.4} failures={} coverage_gaps={} latency_p95_ms={:.3} hard_case_avg_guesses={:.4} hard_case_failures={}",
                    evaluation.average_guesses,
                    evaluation.failures,
                    evaluation.coverage_gaps,
                    evaluation.latency_p95_ms,
                    evaluation.hard_case_average_guesses,
                    evaluation.hard_case_failures
                );
            }
        }
        Command::ThreeGuessGap { from, to, top } => {
            let from = NaiveDate::parse_from_str(&from, "%Y-%m-%d")
                .with_context(|| format!("invalid date '{}'", from))?;
            let to = NaiveDate::parse_from_str(&to, "%Y-%m-%d")
                .with_context(|| format!("invalid date '{}'", to))?;
            if from > to {
                bail!("--from cannot be after --to");
            }
            let report = Solver::three_guess_gap_report(&paths, &config, from, to, top)?;
            println!(
                "games={} base_avg_guesses={:.4} aggressive_case_avg_guesses={:.4} base_four_guess_cases={} aggressive_four_guess_cases={} converted_by_aggressive={} converted_by_targeted_search={}",
                report.games,
                report.base_average_guesses,
                report.aggressive_case_average_guesses,
                report.base_four_guess_cases,
                report.aggressive_four_guess_cases,
                report.converted_by_aggressive,
                report.converted_by_targeted_search
            );
            for case in report.cases {
                println!(
                    "target={} date={} base_guesses={} aggressive_guesses={} best_forced_guesses={} converted_by_aggressive={} converted_by_targeted_search={} base_path={} aggressive_path={} best_forced_path={}",
                    case.target,
                    case.date,
                    case.base_guesses,
                    case.aggressive_guesses,
                    case.best_forced_guesses,
                    case.converted_by_aggressive,
                    case.converted_by_targeted_search,
                    case.base_path.join("/"),
                    case.aggressive_path.join("/"),
                    case.best_forced_path.join("/")
                );
            }
        }
        Command::FourGuessOpeners {
            from,
            to,
            top,
            opener,
        } => {
            let from = NaiveDate::parse_from_str(&from, "%Y-%m-%d")
                .with_context(|| format!("invalid date '{}'", from))?;
            let to = NaiveDate::parse_from_str(&to, "%Y-%m-%d")
                .with_context(|| format!("invalid date '{}'", to))?;
            if from > to {
                bail!("--from cannot be after --to");
            }
            let report = Solver::four_guess_opener_report(&paths, &config, from, to, top, &opener)?;
            println!("games={}", report.games);
            for target in report.targets {
                println!(
                    "target={} date={} base_path={}",
                    target.target,
                    target.date,
                    target.base_path.join("/")
                );
            }
            for evaluation in report.evaluations {
                println!(
                    "opener={} avg_guesses={:.4} three_guess_solves={} failures={} p95={} max={}",
                    evaluation.opener,
                    evaluation.average_guesses,
                    evaluation.three_guess_solves,
                    evaluation.failures,
                    evaluation.p95_guesses,
                    evaluation.max_guesses
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
                "mode={} variant={} as_of={} opener={} games={} four_guess_games={} avg_guesses={:.4} failures={} holdout_games={} holdout_four_guess_games={} holdout_avg_guesses={:.4} holdout_failures={} fingerprint={} path={}",
                solver.mode.label(),
                solver.variant.label(),
                summary.as_of,
                summary.opener,
                summary.games,
                summary.four_guess_games,
                summary.average_guesses,
                summary.failures,
                summary.holdout_games,
                summary.holdout_four_guess_games,
                summary.holdout_average_guesses,
                summary.holdout_failures,
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
                "mode={} variant={} as_of={} opener={} replies={} third_replies={} fingerprint={} path={}",
                solver.mode.label(),
                solver.variant.label(),
                summary.as_of,
                summary.opener,
                summary.reply_count,
                summary.third_reply_count,
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
        Command::EvaluationPlan {
            minimum_training_days,
            validation_days,
            step_days,
            sealed_test_days,
            maximum_folds,
        } => {
            let (history_start, history_end) = Solver::latest_history_range(&paths)?
                .ok_or_else(|| anyhow!("run sync-data before evaluation-plan"))?;
            let plan = build_rolling_origin_plan(
                DateRange::new(history_start, history_end)?,
                RollingOriginConfig {
                    minimum_training_days,
                    validation_days,
                    step_days,
                    sealed_test_days,
                    maximum_folds,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        Command::ParameterRegistry => {
            let registry = predictive_parameter_registry(&config);
            registry.validate()?;
            println!("{}", serde_json::to_string_pretty(&registry)?);
        }
        Command::StudyRun {
            name,
            base_config,
            stage,
            trials,
            jobs,
            seed,
            strategy,
            maximum_validation_folds,
            initial_validation_folds,
            reduction_factor,
            maximum_trial_seconds,
            maximum_memory_mb,
            state,
            cancel_file,
            output_config,
            top,
        } => {
            let study_base_config = base_config
                .as_deref()
                .map(PriorConfig::load)
                .transpose()?
                .unwrap_or_else(|| config.clone());
            let summary = Solver::run_predictive_study(
                &paths,
                &study_base_config,
                StudySpec {
                    name,
                    stage: parse_study_stage(&stage)?,
                    seed,
                    trial_count: trials,
                    parallelism: jobs,
                    strategy: parse_study_strategy(&strategy)?,
                    maximum_validation_folds,
                    initial_validation_folds,
                    reduction_factor,
                    fold_selection: StudyFoldSelection::NestedTimeSpread,
                    maximum_trial_seconds,
                    maximum_memory_mb,
                },
                &state,
                top,
                cancel_file.as_deref(),
            )?;
            if let (Some(path), Some(best_config)) =
                (output_config.as_deref(), summary.best_config.as_ref())
            {
                best_config.save(path)?;
            }
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::TunePrior => {
            let summary = Solver::tune_prior(&paths, &config)?;
            println!(
                "rolling_folds={} train_span={}..{} validation_span={}..{} sealed_test_window={}..{} sealed_test_evaluated=false current_conditional_mean_guesses={:.4} current_all_game_penalized_mean_guesses={:.4} current_failures={} current_coverage_gaps={} current_log_loss={:.6} current_target_rank={:.2} current_latency_p95_ms={:.3} current_hard_case_avg_guesses={:.4} current_hard_case_failures={} current_regime_mix=proxy:{:.1}%/lookahead:{:.1}%/escalated_exact:{:.1}%/exact:{:.1}%",
                summary.evaluation_plan.folds.len(),
                summary.search_window_start,
                summary.search_window_end,
                summary.validation_window_start,
                summary.validation_window_end,
                summary.test_window_start,
                summary.test_window_end,
                summary.current.average_guesses,
                summary.current.all_game_penalized_mean_guesses,
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
                "best_conditional_mean_guesses={:.4} best_all_game_penalized_mean_guesses={:.4} best_failures={} best_coverage_gaps={} best_log_loss={:.6} best_target_rank={:.2} best_latency_p95_ms={:.3} best_hard_case_avg_guesses={:.4} best_hard_case_failures={} best_regime_mix=proxy:{:.1}%/lookahead:{:.1}%/escalated_exact:{:.1}%/exact:{:.1}%",
                summary.best.average_guesses,
                summary.best.all_game_penalized_mean_guesses,
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
        Command::FitProxyWeights => {
            let maximum_validation_folds = Solver::latest_history_range(&paths)?
                .map(|(start, end)| DateRange::new(start, end))
                .transpose()?
                .map_or(1, |history| {
                    build_rolling_origin_plan(history, RollingOriginConfig::default())
                        .map_or(1, |plan| plan.folds.len())
                });
            let summary = Solver::run_predictive_study(
                &paths,
                &config,
                StudySpec {
                    name: "fit-proxy-weights".to_string(),
                    stage: StudyStage::ProxyRanker,
                    seed: 20_260_315,
                    trial_count: 24,
                    parallelism: std::thread::available_parallelism()
                        .map_or(1, usize::from)
                        .min(4),
                    strategy: StudySearchStrategy::LowDiscrepancy,
                    maximum_validation_folds,
                    initial_validation_folds: maximum_validation_folds.min(3),
                    reduction_factor: 3,
                    fold_selection: StudyFoldSelection::NestedTimeSpread,
                    maximum_trial_seconds: 7_200,
                    maximum_memory_mb: 4_096,
                },
                &paths.root.join("target/studies/fit-proxy-weights-v16.json"),
                5,
                None,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::SearchRegret {
            config: audit_config,
            from,
            to,
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
            output,
        } => {
            let audit_config = audit_config
                .as_deref()
                .map(PriorConfig::load)
                .transpose()?
                .unwrap_or_else(|| config.clone());
            let from = parse_date(Some(&from))?
                .ok_or_else(|| anyhow!("--from is required for search-regret"))?;
            let to = parse_date(Some(&to))?
                .ok_or_else(|| anyhow!("--to is required for search-regret"))?;
            let solver = Solver::from_paths(&paths, &audit_config)?;
            let report = solver.search_regret_report(
                &paths,
                SearchRegretRequest {
                    from,
                    to,
                    minimum_survivors,
                    maximum_survivors,
                    maximum_states,
                    maximum_seconds,
                },
            )?;
            let encoded = serde_json::to_vec_pretty(&report)
                .context("failed to encode search-regret JSON")?;
            atomic_write(&output, &encoded)?;
            println!(
                "search_regret={} states={} available={} production_mean={:.6} proxy_mean={:.6} lookahead_mean={:.6}",
                output.display(),
                report.sampled_states,
                report.available_states,
                report.production.mean_regret,
                report.proxy.mean_regret,
                report.lookahead.mean_regret,
            );
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
        Command::BenchmarkEvidence {
            from,
            to,
            top,
            maximum_seconds,
            maximum_memory_mb,
            output,
            markdown_output,
        } => {
            let from = NaiveDate::parse_from_str(&from, "%Y-%m-%d")
                .with_context(|| format!("invalid --from date: {from}"))?;
            let to = NaiveDate::parse_from_str(&to, "%Y-%m-%d")
                .with_context(|| format!("invalid --to date: {to}"))?;
            let artifact = Solver::build_development_evidence_with_budget(
                &paths,
                &config,
                from,
                to,
                top,
                EvidenceResourceBudget {
                    maximum_seconds,
                    maximum_memory_mb,
                },
            )?;
            atomic_write(
                &output,
                &serde_json::to_vec_pretty(&artifact)
                    .context("failed to serialize predictive evidence")?,
            )?;
            let markdown = Solver::render_development_evidence_markdown(&artifact)?;
            atomic_write(&markdown_output, markdown.as_bytes())?;
            println!(
                "evidence_json={} evidence_markdown={} baselines={} sealed_test_evaluated=false",
                output.display(),
                markdown_output.display(),
                artifact.baselines.len()
            );
        }
        Command::BenchmarkEvidenceDocs {
            evidence,
            markdown_output,
            readme,
            update,
        } => {
            let bytes = std::fs::read(&evidence)
                .with_context(|| format!("failed to read {}", evidence.display()))?;
            let artifact: maybe_wordle::solver::PredictiveEvidenceArtifact =
                serde_json::from_slice(&bytes)
                    .with_context(|| format!("failed to parse {}", evidence.display()))?;
            let generated = Solver::render_development_evidence_markdown(&artifact)?;
            let readme_text = std::fs::read_to_string(&readme)
                .with_context(|| format!("failed to read {}", readme.display()))?;
            let updated_readme = replace_generated_evidence(&readme_text, &generated)?;
            if update {
                atomic_write(&markdown_output, generated.as_bytes())?;
                atomic_write(&readme, updated_readme.as_bytes())?;
                println!(
                    "updated_markdown={} updated_readme={}",
                    markdown_output.display(),
                    readme.display()
                );
            } else {
                let existing_markdown = std::fs::read_to_string(&markdown_output)
                    .with_context(|| format!("failed to read {}", markdown_output.display()))?;
                if existing_markdown != generated {
                    bail!(
                        "generated evidence fragment is stale: run benchmark-evidence-docs --evidence {} --update",
                        evidence.display()
                    );
                }
                if updated_readme != readme_text {
                    bail!(
                        "README evidence fragment is stale: run benchmark-evidence-docs --evidence {} --update",
                        evidence.display()
                    );
                }
                println!("predictive evidence documentation is current");
            }
        }
        Command::RollingCompare {
            baseline_config,
            baseline_label,
            candidate_config,
            candidate_label,
            top,
            output,
            baseline_artifact,
        } => {
            let rolling_baseline = baseline_config
                .as_deref()
                .map(PriorConfig::load)
                .transpose()?
                .unwrap_or_else(|| config.clone());
            let candidate = PriorConfig::load(&candidate_config)?;
            let reusable = baseline_artifact
                .as_deref()
                .map(|path| {
                    let bytes = std::fs::read(path)
                        .with_context(|| format!("failed to read {}", path.display()))?;
                    serde_json::from_slice::<maybe_wordle::solver::RollingComparisonArtifact>(
                        &bytes,
                    )
                    .with_context(|| format!("failed to parse {}", path.display()))
                })
                .transpose()?;
            let artifact = Solver::build_rolling_config_comparison(
                &paths,
                &rolling_baseline,
                &baseline_label,
                &candidate,
                &candidate_label,
                top,
                reusable.as_ref(),
            )?;
            atomic_write(
                &output,
                &serde_json::to_vec_pretty(&artifact)
                    .context("failed to serialize rolling comparison")?,
            )?;
            println!(
                "rolling_folds={} baseline_all_game_mean={:.4} candidate_all_game_mean={:.4} delta={:+.4} ci95={:+.4}..{:+.4} wins={} ties={} losses={} sealed_test_evaluated=false output={}",
                artifact.evaluation_plan.folds.len(),
                artifact.baseline.aggregate.all_game_penalized_mean_guesses,
                artifact.candidate.aggregate.all_game_penalized_mean_guesses,
                artifact.candidate_minus_baseline.candidate_minus_baseline,
                artifact.candidate_minus_baseline.ci95.lower,
                artifact.candidate_minus_baseline.ci95.upper,
                artifact.candidate_minus_baseline.candidate_wins,
                artifact.candidate_minus_baseline.ties,
                artifact.candidate_minus_baseline.baseline_wins,
                output.display()
            );
        }
        Command::FreezeCandidate {
            config,
            comparison,
            output,
        } => {
            let frozen = Solver::freeze_predictive_candidate(&paths, &config, &comparison)?;
            let output = if output.is_absolute() {
                output
            } else {
                paths.root.join(output)
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            atomic_write(
                &output,
                &serde_json::to_vec_pretty(&frozen)
                    .context("failed to serialize frozen candidate")?,
            )?;
            println!(
                "candidate={} freeze={} development_all_game_mean={:.4} development_failures={} sealed_test_evaluated=false output={}",
                frozen.candidate_label,
                frozen.freeze_fingerprint,
                frozen.development_metrics.all_game_penalized_mean_guesses,
                frozen.development_metrics.unsolved_games
                    + frozen.development_metrics.coverage_gaps,
                output.display()
            );
        }
        Command::EvaluateSealed { frozen, output } => {
            let bytes = std::fs::read(&frozen)
                .with_context(|| format!("failed to read {}", frozen.display()))?;
            let frozen: maybe_wordle::solver::FrozenPredictiveCandidate =
                serde_json::from_slice(&bytes)
                    .with_context(|| format!("failed to parse {}", frozen.display()))?;
            let report =
                Solver::evaluate_frozen_candidate_on_sealed_test(&paths, &frozen, &output)?;
            println!(
                "sealed_games={} solved={} failures={} coverage_gaps={} all_game_mean={:.4} ci95={:.4}..{:.4} latency_p95_ms={:.3} evaluated_once=true output={}",
                report.metrics.scheduled_games,
                report.metrics.solved_games,
                report.metrics.unsolved_games,
                report.metrics.coverage_gaps,
                report.metrics.all_game_penalized_mean_guesses,
                report.metrics.all_game_penalized_mean_guesses_ci95.lower,
                report.metrics.all_game_penalized_mean_guesses_ci95.upper,
                report.latency_p95_ms,
                output.display()
            );
        }
        Command::RollingEvidenceDocs {
            comparison,
            markdown_output,
            readme,
            update,
        } => {
            let comparisons = comparison
                .iter()
                .map(|path| {
                    let bytes = std::fs::read(path)
                        .with_context(|| format!("failed to read {}", path.display()))?;
                    serde_json::from_slice::<maybe_wordle::solver::RollingComparisonArtifact>(
                        &bytes,
                    )
                    .with_context(|| format!("failed to parse {}", path.display()))
                })
                .collect::<Result<Vec<_>>>()?;
            let generated = Solver::render_rolling_comparison_markdown(&comparisons)?;
            let readme_text = std::fs::read_to_string(&readme)
                .with_context(|| format!("failed to read {}", readme.display()))?;
            let updated_readme = replace_generated_rolling_evidence(&readme_text, &generated)?;
            if update {
                atomic_write(&markdown_output, generated.as_bytes())?;
                atomic_write(&readme, updated_readme.as_bytes())?;
                println!(
                    "updated_markdown={} updated_readme={}",
                    markdown_output.display(),
                    readme.display()
                );
            } else {
                let existing = std::fs::read_to_string(&markdown_output)
                    .with_context(|| format!("failed to read {}", markdown_output.display()))?;
                if canonical_newlines(&existing) != canonical_newlines(&generated)
                    || canonical_newlines(&updated_readme) != canonical_newlines(&readme_text)
                {
                    bail!("rolling evidence documentation is stale; rerun with --update");
                }
                println!("rolling evidence documentation is current");
            }
        }
    }

    Ok(())
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

fn replace_generated_evidence(readme: &str, generated: &str) -> Result<String> {
    const START: &str = "<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->";
    const END: &str = "<!-- END GENERATED PREDICTIVE EVIDENCE -->";
    let start = readme
        .find(START)
        .ok_or_else(|| anyhow!("README is missing the generated evidence start marker"))?;
    let end_start = readme[start..]
        .find(END)
        .map(|offset| start + offset)
        .ok_or_else(|| anyhow!("README is missing the generated evidence end marker"))?;
    let end = end_start + END.len();
    let mut updated = String::with_capacity(readme.len() + generated.len());
    updated.push_str(&readme[..start]);
    updated.push_str(generated.trim_end());
    updated.push_str(&readme[end..]);
    Ok(updated)
}

fn replace_generated_rolling_evidence(readme: &str, generated: &str) -> Result<String> {
    const START: &str = "<!-- BEGIN GENERATED ROLLING EVIDENCE -->";
    const END: &str = "<!-- END GENERATED ROLLING EVIDENCE -->";
    let start = readme
        .find(START)
        .ok_or_else(|| anyhow!("README is missing the generated rolling evidence start marker"))?;
    let end_start = readme[start..]
        .find(END)
        .map(|offset| start + offset)
        .ok_or_else(|| anyhow!("README is missing the generated rolling evidence end marker"))?;
    let end = end_start + END.len();
    let mut updated = String::with_capacity(readme.len() + generated.len());
    updated.push_str(&readme[..start]);
    updated.push_str(generated.trim_end());
    updated.push_str(&readme[end..]);
    Ok(updated)
}

fn canonical_newlines(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn warn_predictive_history_range(paths: &ProjectPaths, as_of: NaiveDate) -> Result<()> {
    let Some((first_synced, last_synced)) = Solver::latest_history_range(paths)? else {
        eprintln!(
            "warning: no synced NYT history found; run cargo run -- sync-data before relying on predictive history or artifacts"
        );
        return Ok(());
    };
    if as_of < first_synced {
        eprintln!(
            "warning: requested date {} is before the earliest synced NYT date {}; predictive history and artifacts may be incomplete",
            as_of, first_synced
        );
    }
    if as_of > last_synced {
        eprintln!(
            "warning: requested date {} is after the latest synced NYT date {}; predictive history and artifacts may be stale",
            as_of, last_synced
        );
    }
    Ok(())
}

fn format_sync_summary(summary: &SyncSummary) -> String {
    let status = if summary.partial_sync {
        "partial"
    } else {
        "complete"
    };
    format!(
        "sync_status={} entries={} range={}..{} fetched={} reverified={} changed={}",
        status,
        summary.total,
        summary.first_date,
        summary.last_date,
        summary.fetched,
        summary.reverified,
        summary.changed
    )
}

fn enforce_sync_policy(strict: bool, summary: &SyncSummary) -> Result<()> {
    if strict && summary.partial_sync {
        bail!(
            "partial sync encountered failed dates: {}",
            summary
                .failed_dates
                .iter()
                .map(|date| date.format("%Y-%m-%d").to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    Ok(())
}

fn predictive_warning_lines(
    as_of: NaiveDate,
    observations: &[(String, u8)],
    mode: PredictiveSuggestionMode,
    response: &PredictiveSuggestResponse,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let artifact_warning = match response.artifact_state {
        maybe_wordle::predictive::PredictiveArtifactState::ExactDateArtifact => {
            if observations.is_empty() {
                Some(format!(
                    "exact-date predictive opener artifact is available for {}",
                    as_of
                ))
            } else {
                Some(format!(
                    "exact-date predictive reply-book artifact is available for {}",
                    as_of
                ))
            }
        }
        maybe_wordle::predictive::PredictiveArtifactState::RecentOpenerArtifact => Some(format!(
            "no exact-date opener artifact for {}; reusing a recent opener artifact",
            as_of
        )),
        maybe_wordle::predictive::PredictiveArtifactState::LiveSessionFallback => {
            Some("predictive artifact unavailable for this state; using live session fallback".to_string())
        }
        maybe_wordle::predictive::PredictiveArtifactState::NoPredictiveArtifactAvailable => {
            Some(match mode {
                PredictiveSuggestionMode::FastDiskOnly => {
                    "predictive artifact unavailable for this state; disk-only mode will use live ranking without promotion".to_string()
                }
                PredictiveSuggestionMode::Full => {
                    "predictive artifact unavailable for this state; using live ranking without artifact promotion".to_string()
                }
                PredictiveSuggestionMode::LiveOnly => {
                    "predictive artifact lookup disabled; using live ranking only".to_string()
                }
            })
        }
    };
    if let Some(warning) = artifact_warning {
        warnings.push(warning);
    }
    if matches!(observations.len(), 1 | 2)
        && response.artifact_state
            != maybe_wordle::predictive::PredictiveArtifactState::ExactDateArtifact
    {
        warnings.push(
            "reply-book artifact is missing for this date or branch; branch suggestions are coming from live evaluation".to_string(),
        );
    }
    warnings
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
        line.push_str(&format!(" candidate_pool_exact_cost={:.5}", exact_cost));
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

fn parse_study_stage(raw: &str) -> Result<StudyStage> {
    match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "calibration" | "prior" => Ok(StudyStage::Calibration),
        "coverage-recovery" | "recovery" => Ok(StudyStage::CoverageRecovery),
        "proxy-core" => Ok(StudyStage::ProxyCore),
        "proxy-risk" => Ok(StudyStage::ProxyRisk),
        "proxy-small-state" => Ok(StudyStage::ProxySmallState),
        "proxy-ranker" | "proxy" => Ok(StudyStage::ProxyRanker),
        "search-routing" => Ok(StudyStage::SearchRouting),
        "search-exact" => Ok(StudyStage::SearchExact),
        "search-coverage" => Ok(StudyStage::SearchCoverage),
        "search-lookahead" => Ok(StudyStage::SearchLookahead),
        "search-pool" => Ok(StudyStage::SearchPool),
        "search-danger" => Ok(StudyStage::SearchDanger),
        "search-penalty" => Ok(StudyStage::SearchPenalty),
        "solve-policy" | "solve" => Ok(StudyStage::SolvePolicy),
        "book-policy" | "book" => Ok(StudyStage::BookPolicy),
        "joint" | "all" => Ok(StudyStage::Joint),
        _ => bail!(
            "study stage must be one of: calibration, coverage-recovery, proxy-core, proxy-risk, proxy-small-state, proxy-ranker, search-routing, search-exact, search-coverage, search-lookahead, search-pool, search-danger, search-penalty, solve-policy, book-policy, joint"
        ),
    }
}

fn parse_study_strategy(raw: &str) -> Result<StudySearchStrategy> {
    match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "grid" => Ok(StudySearchStrategy::Grid),
        "low-discrepancy" | "quasi-random" => Ok(StudySearchStrategy::LowDiscrepancy),
        "random" => Ok(StudySearchStrategy::Random),
        "local-refinement" | "local" => Ok(StudySearchStrategy::LocalRefinement),
        "model-based" | "model_based" | "tpe" => Ok(StudySearchStrategy::ModelBased),
        _ => {
            bail!(
                "study strategy must be one of: grid, low-discrepancy, random, local-refinement, model-based"
            )
        }
    }
}

fn reject_hard_mode_for_non_predictive(hard: bool, mode: &str) -> Result<()> {
    if hard {
        bail!("--hard is only supported in predictive Wordle mode, not {mode}");
    }
    Ok(())
}

fn reject_live_fallback_for_non_predictive(live_fallback: bool, mode: &str) -> Result<()> {
    if live_fallback {
        bail!("--live-fallback is only supported in predictive Wordle mode, not {mode}");
    }
    Ok(())
}

fn predictive_cli_mode(live_fallback: bool) -> PredictiveSuggestionMode {
    if live_fallback {
        PredictiveSuggestionMode::Full
    } else {
        PredictiveSuggestionMode::FastDiskOnly
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
    use std::fs;

    use anyhow::anyhow;
    use chrono::NaiveDate;
    use clap::CommandFactory;
    use maybe_wordle::data::SyncSummary;
    use maybe_wordle::experiments::StudySearchStrategy;
    use maybe_wordle::predictive::{PredictiveArtifactState, PredictiveSuggestResponse};
    use maybe_wordle::solver::AbsurdleSuggestion;

    use super::{
        Cli, canonical_newlines, enforce_sync_policy, find_project_root,
        format_absurdle_suggestion, format_predictive_suggestion, format_sync_summary,
        normalize_interactive_guess, parse_solver_mode, parse_study_stage, parse_study_strategy,
        predictive_cli_mode, predictive_warning_lines, reject_hard_mode_for_non_predictive,
        reject_live_fallback_for_non_predictive, try_append_observation,
    };

    #[test]
    fn documentation_verification_ignores_platform_line_endings() {
        assert_eq!(
            canonical_newlines("alpha\r\nbeta\r\n"),
            canonical_newlines("alpha\nbeta\n")
        );
        assert_ne!(
            canonical_newlines("alpha\r\nbeta\r\n"),
            canonical_newlines("alpha\ngamma\n")
        );
    }

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
        assert!(formatted.contains("candidate_pool_exact_cost=2.50000"));
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

    #[test]
    fn parse_study_strategy_accepts_documented_aliases_and_rejects_unknown_values() {
        assert_eq!(
            parse_study_strategy("low_discrepancy").expect("low discrepancy"),
            StudySearchStrategy::LowDiscrepancy
        );
        assert_eq!(
            parse_study_strategy("quasi-random").expect("quasi-random"),
            StudySearchStrategy::LowDiscrepancy
        );
        assert_eq!(
            parse_study_strategy("local").expect("local"),
            StudySearchStrategy::LocalRefinement
        );
        assert_eq!(
            parse_study_strategy("tpe").expect("tpe"),
            StudySearchStrategy::ModelBased
        );
    }

    #[test]
    fn parse_study_stage_accepts_proxy_ranker_aliases() {
        assert_eq!(
            parse_study_stage("proxy-ranker").expect("proxy ranker"),
            maybe_wordle::experiments::StudyStage::ProxyRanker
        );
        assert_eq!(
            parse_study_stage("proxy").expect("proxy alias"),
            maybe_wordle::experiments::StudyStage::ProxyRanker
        );
    }

    #[test]
    fn parse_study_stage_accepts_typed_cohort_aliases() {
        assert_eq!(
            parse_study_stage("proxy_small_state").expect("proxy small state"),
            maybe_wordle::experiments::StudyStage::ProxySmallState
        );
        assert_eq!(
            parse_study_stage("search-exact").expect("search exact"),
            maybe_wordle::experiments::StudyStage::SearchExact
        );
        assert_eq!(
            parse_study_stage("search_penalty").expect("search penalty"),
            maybe_wordle::experiments::StudyStage::SearchPenalty
        );
    }

    #[test]
    fn reject_hard_mode_for_non_predictive_modes() {
        assert!(reject_hard_mode_for_non_predictive(false, "absurdle").is_ok());
        assert!(reject_hard_mode_for_non_predictive(true, "absurdle").is_err());
    }

    #[test]
    fn reject_live_fallback_for_non_predictive_modes() {
        assert!(reject_live_fallback_for_non_predictive(false, "absurdle").is_ok());
        assert!(reject_live_fallback_for_non_predictive(true, "absurdle").is_err());
    }

    #[test]
    fn predictive_cli_mode_requires_explicit_live_fallback() {
        assert_eq!(
            predictive_cli_mode(false),
            maybe_wordle::predictive::PredictiveSuggestionMode::FastDiskOnly
        );
        assert_eq!(
            predictive_cli_mode(true),
            maybe_wordle::predictive::PredictiveSuggestionMode::Full
        );
    }

    #[test]
    fn find_project_root_walks_up_from_nested_binary_path() {
        let temp_root =
            std::env::temp_dir().join(format!("maybe-wordle-root-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_root);
        fs::create_dir_all(temp_root.join("config")).expect("config dir");
        fs::create_dir_all(temp_root.join("data/seed")).expect("seed dir");
        fs::create_dir_all(temp_root.join("target/release")).expect("release dir");
        fs::write(temp_root.join("config/prior.toml"), "").expect("prior");
        fs::write(temp_root.join("data/seed/valid_guesses.txt"), "crane\n").expect("guesses");
        fs::write(temp_root.join("data/seed/candidate_answers.txt"), "crane\n").expect("answers");

        let nested_exe = temp_root.join("target/release/maybe-wordle.exe");
        fs::write(&nested_exe, "").expect("exe");

        assert_eq!(
            find_project_root(&nested_exe).expect("project root"),
            temp_root
        );

        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn predictive_warning_lines_report_live_branch_fallback() {
        let warnings = predictive_warning_lines(
            NaiveDate::from_ymd_opt(2026, 3, 26).expect("date"),
            &[("crane".to_string(), 17)],
            maybe_wordle::predictive::PredictiveSuggestionMode::Full,
            &PredictiveSuggestResponse {
                state: maybe_wordle::predictive::PredictiveStateSummary {
                    surviving: 3,
                    modeled_total_weight: 1.0,
                    effective_total_weight: 1.0,
                    recovery_mode_used: None,
                },
                suggestions: Vec::new(),
                candidates: Vec::new(),
                promoted_word: None,
                promotion_source: None,
                artifact_state: PredictiveArtifactState::LiveSessionFallback,
                model_version: "test".to_string(),
                model_manifest_hash: "test".to_string(),
                history_snapshot_date: None,
                history_snapshot_hash: "test".to_string(),
            },
        );
        assert!(
            warnings
                .iter()
                .any(|line| line.contains("live session fallback"))
        );
        assert!(
            warnings
                .iter()
                .any(|line| line.contains("reply-book artifact is missing"))
        );
    }

    #[test]
    fn format_sync_summary_marks_partial_sync() {
        let summary = SyncSummary {
            fetched: 2,
            reverified: 1,
            changed: 0,
            total: 5,
            first_date: NaiveDate::from_ymd_opt(2021, 6, 19).expect("first"),
            last_date: NaiveDate::from_ymd_opt(2021, 6, 23).expect("last"),
            changed_dates: Vec::new(),
            partial_sync: true,
            failed_dates: vec![NaiveDate::from_ymd_opt(2021, 6, 24).expect("failed")],
            last_successful_date: Some(NaiveDate::from_ymd_opt(2021, 6, 23).expect("success")),
        };
        assert!(format_sync_summary(&summary).contains("sync_status=partial"));
    }

    #[test]
    fn strict_sync_policy_rejects_partial_sync() {
        let summary = SyncSummary {
            fetched: 2,
            reverified: 1,
            changed: 0,
            total: 5,
            first_date: NaiveDate::from_ymd_opt(2021, 6, 19).expect("first"),
            last_date: NaiveDate::from_ymd_opt(2021, 6, 23).expect("last"),
            changed_dates: Vec::new(),
            partial_sync: true,
            failed_dates: vec![NaiveDate::from_ymd_opt(2021, 6, 24).expect("failed")],
            last_successful_date: Some(NaiveDate::from_ymd_opt(2021, 6, 23).expect("success")),
        };
        let error = enforce_sync_policy(true, &summary).expect_err("strict should fail");
        assert!(format!("{error:#}").contains("2021-06-24"));
    }

    #[test]
    fn non_strict_sync_policy_allows_partial_sync() {
        let summary = SyncSummary {
            fetched: 2,
            reverified: 1,
            changed: 0,
            total: 5,
            first_date: NaiveDate::from_ymd_opt(2021, 6, 19).expect("first"),
            last_date: NaiveDate::from_ymd_opt(2021, 6, 23).expect("last"),
            changed_dates: Vec::new(),
            partial_sync: true,
            failed_dates: vec![NaiveDate::from_ymd_opt(2021, 6, 24).expect("failed")],
            last_successful_date: Some(NaiveDate::from_ymd_opt(2021, 6, 23).expect("success")),
        };
        enforce_sync_policy(false, &summary).expect("non-strict should pass");
    }

    #[test]
    fn help_text_mentions_predictive_mode_and_weight_mode() {
        let mut suggest = Cli::command()
            .find_subcommand_mut("suggest")
            .expect("suggest help")
            .clone();
        let mut suggest_help = Vec::new();
        suggest.write_long_help(&mut suggest_help).expect("help");
        let suggest_rendered = String::from_utf8(suggest_help).expect("utf8");
        assert!(suggest_rendered.contains("Solver mode: predictive, absurdle, or formal-optimal"));
        assert!(suggest_rendered.contains("Allow slower predictive live-session promotion"));

        let mut opener = Cli::command()
            .find_subcommand_mut("build-predictive-opener")
            .expect("opener help")
            .clone();
        let mut opener_help = Vec::new();
        opener.write_long_help(&mut opener_help).expect("help");
        let opener_rendered = String::from_utf8(opener_help).expect("utf8");
        assert!(
            opener_rendered.contains("Answer-weight model: weighted, uniform, or cooldown_only")
        );

        let mut study = Cli::command()
            .find_subcommand_mut("study-run")
            .expect("study help")
            .clone();
        let mut study_help = Vec::new();
        study.write_long_help(&mut study_help).expect("help");
        let study_rendered = String::from_utf8(study_help).expect("utf8");
        assert!(
            study_rendered
                .contains("grid, low-discrepancy, random, local-refinement, or model-based")
        );
        assert!(study_rendered.contains("first successive-halving rung"));
        assert!(study_rendered.contains("Pause cooperatively"));
        assert!(study_rendered.contains("peak-working-set budget"));
        assert!(study_rendered.contains("Optional TOML base config"));
    }
}
