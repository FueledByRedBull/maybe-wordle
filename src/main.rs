use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{NaiveDate, Utc};
use clap::{Parser, Subcommand};
use maybe_wordle::predictive::types::{SearchActionScope, SuggestionValueKind};
use maybe_wordle::{
    SOLVER_THREAD_STACK_BYTES,
    atomic_file::atomic_write,
    config::PriorConfig,
    data::{ProjectPaths, SyncSummary, sync_nyt_history},
    experiments::{
        DateRange, EvaluationPolicy, RollingOriginConfig, StudyFoldSelection, StudySearchStrategy,
        StudySpec, StudyStage, build_declared_rolling_origin_plan, predictive_parameter_registry,
    },
    formal::{
        DEFAULT_FORMAL_MODEL_ID, FormalAlternativesStatus, FormalPolicyRuntime, FormalScaleRequest,
        FormalVerificationMode, benchmark_formal_scale, build_optimal_policy,
        parse_observations as parse_formal_observations, verify_optimal_policy_with_mode,
    },
    game::{GameRules, GameStatus, status as game_status, try_append_observation},
    gui::run_gui,
    model::build_model_artifacts,
    model::{ModelVariant, WeightMode},
    predictive::{
        PredictiveSuggestRequest, PredictiveSuggestResponse, PredictiveSuggestionMode,
        history_cutoff,
    },
    research::{fit_learned_proxy_experiment, run_survival_experiment},
    seed::{MergeStrategy, add_manual_addition, merge_seed_lists, reconcile_seed_lists},
    solver::{
        AbsurdleSuggestion, EvidenceDateSelection, EvidenceResourceBudget,
        FiniteSearchRegretRequest, LearnedProxyDatasetRequest, SearchRegretRequest, Solver,
        StagedZeroFailureCertificateRequest,
    },
};

#[path = "cli/evidence.rs"]
mod evidence;

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
        #[arg(long, help = "Last NYT print date to fetch; defaults to today")]
        through: Option<String>,
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
        #[arg(long, value_parser = ["fast", "strong"], conflicts_with = "live_fallback",
            help = "Use artifact-free finite-horizon search with a fast or strong budget")]
        search_budget: Option<String>,
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
            help = "Backtest start date in YYYY-MM-DD (required; declared development only)"
        )]
        from: String,
        #[arg(
            long,
            help = "Backtest end date in YYYY-MM-DD (required; declared development only)"
        )]
        to: String,
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
        #[arg(long, help = "Run only the named matrix profile")]
        profile: Option<String>,
    },
    #[command(about = "Screen prior families on the declared rolling development folds")]
    PriorAblations {
        #[arg(long = "profile", help = "Matrix profile to include; repeat as needed")]
        profiles: Vec<String>,
        #[arg(long, default_value = "benchmarks/predictive/prior-ablation-v1.json")]
        output: PathBuf,
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
            help = "Answer-weight model: weighted, uniform, cooldown_only, used_unused, recency_buckets, empirical_frequency, or regularized_frequency"
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
            help = "Answer-weight model: weighted, uniform, cooldown_only, used_unused, recency_buckets, empirical_frequency, or regularized_frequency"
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
            help = "Evaluation start date in YYYY-MM-DD (required; declared development only)"
        )]
        from: String,
        #[arg(
            long,
            help = "Evaluation end date in YYYY-MM-DD (required; declared development only)"
        )]
        to: String,
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
            help = "Maximum concurrent candidate evaluations; finite policies require 1 to preserve wall-clock search budgets"
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
            help = "Cumulative candidate wall-clock budget in seconds across resumes; checked cooperatively, not a hard deadline"
        )]
        maximum_trial_seconds: u64,
        #[arg(
            long,
            default_value_t = 4096,
            help = "Shared process peak-working-set budget in MiB; sampled cooperatively, not an allocation cap"
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
    #[command(about = "Build development-only exhaustive costs and fit a residual ridge proxy")]
    LearnedProxyExperiment {
        #[arg(long, default_value_t = 3)]
        minimum_survivors: usize,
        #[arg(long, default_value_t = 12)]
        maximum_survivors: usize,
        #[arg(long, default_value_t = 8)]
        maximum_states_per_split: usize,
        #[arg(long, default_value_t = 18)]
        guesses_per_state: usize,
        #[arg(long, default_value_t = 1200)]
        maximum_seconds: u64,
        #[arg(long, default_value_t = 4096)]
        maximum_memory_mb: u64,
        #[arg(
            long,
            default_value = "target/studies/learned-proxy-checkpoint-v1.json",
            help = "Atomic resumable checkpoint written after each exact state"
        )]
        checkpoint: PathBuf,
        #[arg(
            long,
            default_value = "benchmarks/predictive/learned-proxy-dataset-v1.json"
        )]
        dataset_output: PathBuf,
        #[arg(
            long,
            default_value = "benchmarks/predictive/learned-proxy-experiment-v1.json"
        )]
        output: PathBuf,
    },
    #[command(about = "Evaluate a fold-local policy-era survival prior against logistic recency")]
    SurvivalExperiment {
        #[arg(
            long,
            default_value = "benchmarks/predictive/survival-experiment-v1.json"
        )]
        output: PathBuf,
    },
    #[command(
        about = "Audit legacy or finite-horizon search choices against tractable exact references"
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
        #[arg(
            long,
            help = "Maximum reachable states to audit (finite defaults to 6; legacy defaults to 16)"
        )]
        maximum_states: Option<usize>,
        #[arg(
            long,
            help = "Cooperative cumulative audit budget in seconds (finite defaults to 60; legacy defaults to 1800); searches poll internally and completed rows are retained on timeout"
        )]
        maximum_seconds: Option<u64>,
        #[arg(
            long,
            help = "Run the artifact-free finite-horizon audit; requires a finite_fast or finite_strong config"
        )]
        finite: bool,
        #[arg(
            long,
            requires = "finite",
            help = "Apply hard-mode legality to reachable paths and references"
        )]
        hard_mode: bool,
        #[arg(long, help = "Versioned JSON report output path")]
        output: PathBuf,
    },
    #[command(
        about = "Compare staged and finite_fast_dynamic choices against an exhaustive same-state dynamic reference"
    )]
    SameStateDynamicRegret {
        #[arg(long, help = "Selected staged prior TOML config")]
        config: Option<PathBuf>,
        #[arg(long, help = "Development-only NYT print date in YYYY-MM-DD")]
        date: String,
        #[arg(
            long,
            default_value_t = 1,
            help = "Turn on the selected staged path (1-6)"
        )]
        turn: usize,
        #[arg(
            long,
            default_value_t = 30,
            help = "Shared path and exact-reference budget in seconds"
        )]
        maximum_seconds: u64,
        #[arg(long, help = "Summary-only JSON report output path")]
        output: PathBuf,
    },
    #[command(
        about = "Count selected artifact-free staged roots with a conservative modeled zero-failure witness"
    )]
    StagedZeroFailureCertificate {
        #[arg(long, help = "Selected staged prior TOML config")]
        config: Option<PathBuf>,
        #[arg(long, help = "Development-only range start date in YYYY-MM-DD")]
        from: String,
        #[arg(long, help = "Development-only range end date in YYYY-MM-DD")]
        to: String,
        #[arg(long, help = "Maximum selected staged decision roots")]
        maximum_states: usize,
        #[arg(long, help = "Shared diagnostic budget in seconds")]
        maximum_seconds: u64,
        #[arg(long, help = "Require full hard-mode legality in witness children")]
        hard_mode: bool,
        #[arg(long, help = "Summary/provenance JSON report output path")]
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
        #[arg(
            long,
            required_unless_present = "rolling_folds",
            conflicts_with = "rolling_folds",
            help = "Development evaluation start date in YYYY-MM-DD"
        )]
        from: Option<String>,
        #[arg(
            long,
            required_unless_present = "rolling_folds",
            conflicts_with = "rolling_folds",
            help = "Development evaluation end date in YYYY-MM-DD"
        )]
        to: Option<String>,
        #[arg(
            long,
            conflicts_with_all = ["from", "to"],
            help = "Evaluate the exact canonical rolling development validation folds"
        )]
        rolling_folds: bool,
        #[arg(
            long,
            help = "JSON experiment matrix path; defaults to the shipped development matrix"
        )]
        matrix: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        top: usize,
        #[arg(
            long,
            default_value_t = 3600,
            help = "Cumulative evidence wall-clock budget in seconds across resumes; checked cooperatively, not a hard deadline"
        )]
        maximum_seconds: u64,
        #[arg(
            long,
            default_value_t = 4096,
            help = "Shared process peak-working-set budget in MiB; sampled cooperatively, not an allocation cap"
        )]
        maximum_memory_mb: u64,
        #[arg(long, help = "Versioned JSON evidence output path")]
        output: PathBuf,
        #[arg(long, help = "Generated README fragment output path")]
        markdown_output: PathBuf,
        #[arg(
            long,
            default_value = "target/evidence-checkpoints/development-v1.json",
            help = "Atomic per-profile resume checkpoint"
        )]
        checkpoint: PathBuf,
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
    #[command(
        about = "Freeze an eligible development winner for the next UTC 30-day prospective window"
    )]
    FreezeProspective {
        #[arg(long, help = "Complete candidate TOML config to freeze")]
        config: PathBuf,
        #[arg(
            long,
            help = "Current rolling-comparison JSON proving eligibility against its parent"
        )]
        comparison: PathBuf,
        #[arg(
            long,
            default_value = "benchmarks/predictive/prospective-frozen-v1.json"
        )]
        output: PathBuf,
    },
    #[command(
        about = "Evaluate one prospective candidate on the reserved 30-day window exactly once"
    )]
    EvaluateProspective {
        #[arg(
            long,
            default_value = "benchmarks/predictive/prospective-frozen-v1.json"
        )]
        frozen: PathBuf,
        #[arg(long, default_value = "target/diagnostics/prospective-window-v1.json")]
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
        Command::SyncData { strict, through } => {
            let through = parse_or_today(through.as_deref())?;
            let summary = sync_nyt_history(&paths, &config, through)?;
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
                for failure in &summary.failures {
                    eprintln!("sync failure on {}: {}", failure.date, failure.message);
                }
                if !summary.missing_dates.is_empty() {
                    eprintln!(
                        "persisted history is missing {} requested dates",
                        summary.missing_dates.len()
                    );
                }
                if summary.retained_existing_archive {
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
                "built model with {} guesses, {} primary answers, {} dormant fallback answers, {} historical answers across {} effective daily rows ({} raw source rows)",
                summary.guess_count,
                summary.answer_count,
                summary.fallback_answer_count,
                summary.historical_answers,
                summary.history_rows,
                summary.source_history_rows
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
            search_budget,
            model,
        } => match parse_solver_mode(&mode)? {
            SolverMode::Predictive => {
                let observations = Solver::parse_observations(&guess, &feedback)?;
                let as_of = parse_or_today(date.as_deref())?;
                warn_predictive_history_range(&paths, as_of)?;
                let solver = Solver::from_paths(&paths, &config)?;
                let predictive_mode = predictive_cli_mode(live_fallback);
                let request = PredictiveSuggestRequest {
                    puzzle_date: as_of,
                    observations: &observations,
                    top,
                    hard_mode: hard,
                    force_in_two_only: false,
                    mode: predictive_mode,
                };
                let response = if let Some(profile) = search_budget.as_deref() {
                    let mut options = if profile == "strong" {
                        maybe_wordle::solver::FiniteSearchOptions::strong()
                    } else {
                        maybe_wordle::solver::FiniteSearchOptions::fast()
                    };
                    if proxy_preview {
                        options.budget = std::time::Duration::from_millis(30);
                    }
                    solver.suggest_predictive_controlled(request, options, &|| false)?
                } else if proxy_preview {
                    solver.suggest_predictive_proxy_preview(request)?
                } else {
                    solver.suggest_predictive(request)?
                };
                for warning in
                    predictive_warning_lines(as_of, &observations, predictive_mode, hard, &response)
                {
                    eprintln!("warning: {warning}");
                }
                println!(
                    "mode=predictive model={} manifest={} history_snapshot={} history_hash={} artifact_status={} promoted_from_cache={} promoted_artifact_date={} date={} surviving={} total_weight={:.4}",
                    response.model_version,
                    response.model_manifest_hash,
                    response
                        .history_snapshot_date
                        .map(|date| date.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    response.history_snapshot_hash,
                    response.artifact_state.banner_text(),
                    response.promotion_source.is_some(),
                    response
                        .promoted_artifact_date
                        .map_or_else(|| "none".to_string(), |date| date.to_string()),
                    as_of,
                    response.state.surviving,
                    response.state.effective_total_weight
                );
                if let Some(mode) = response.state.recovery_mode_used {
                    println!("recovery_mode={}", mode.label());
                }
                println!(
                    "{}",
                    format_predictive_context(response.puzzle_date, response.history_cutoff)
                );
                println!("{}", response.execution.summary());
                if let Some(search) = &response.finite_search {
                    println!(
                        "search_status={:?} nodes={} work_units={} cache_hits={} proposal_sampled={} artifact_free=true",
                        search.reason,
                        search.nodes_visited,
                        search.work_units,
                        search.cache_hits,
                        search.proposal_sampled
                    );
                }
                for suggestion in response.suggestions {
                    println!("{}", format_predictive_suggestion(&suggestion));
                }
            }
            SolverMode::Absurdle => {
                if search_budget.is_some() {
                    bail!("--search-budget is only supported in predictive mode");
                }
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
                if search_budget.is_some() {
                    bail!("--search-budget is only supported in predictive mode");
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
                let explanation = runtime.explain_state(&state, top)?;
                println!(
                    "alternatives={}",
                    formal_alternatives_label(explanation.alternatives_status)
                );
                for suggestion in explanation.tied_moves {
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
                    let terminal = game_status(&observations, GameRules::Wordle)?;
                    if terminal != GameStatus::Active {
                        println!("game={}", terminal.label());
                        break;
                    }
                    let response = solver.suggest_predictive(PredictiveSuggestRequest {
                        puzzle_date: as_of,
                        observations: &observations,
                        top,
                        hard_mode: hard,
                        force_in_two_only: false,
                        mode: predictive_mode,
                    })?;
                    for warning in predictive_warning_lines(
                        as_of,
                        &observations,
                        predictive_mode,
                        hard,
                        &response,
                    ) {
                        eprintln!("warning: {warning}");
                    }
                    println!(
                        "mode=predictive surviving={} total_weight={:.4}",
                        response.state.surviving, response.state.effective_total_weight
                    );
                    println!(
                        "{}",
                        format_predictive_context(response.puzzle_date, response.history_cutoff)
                    );
                    println!("{}", response.execution.summary());
                    if let Some(mode) = response.state.recovery_mode_used {
                        println!("recovery_mode={}", mode.label());
                    }
                    for suggestion in response.suggestions {
                        println!("{}", format_predictive_suggestion(&suggestion));
                    }

                    print!("guess (blank to stop): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let Some(guess) = read_line(&mut io::stdin().lock())? else {
                        break;
                    };
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
                    let Some(feedback) = read_line(&mut io::stdin().lock())? else {
                        break;
                    };
                    match try_append_observation(
                        &observations,
                        &guess,
                        &feedback,
                        GameRules::Wordle,
                        |next| solver.validate_game_history(as_of, next, hard).map(|_| ()),
                    ) {
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
                    let terminal = game_status(&observations, GameRules::Absurdle)?;
                    if terminal != GameStatus::Active {
                        println!("game={}", terminal.label());
                        break;
                    }
                    let state = solver.absurdle_apply_history(&observations)?;
                    println!("mode=absurdle surviving={}", state.surviving.len());
                    for suggestion in solver.absurdle_suggestions(&observations, top)? {
                        println!("{}", format_absurdle_suggestion(&suggestion));
                    }

                    print!("guess (blank to stop): ");
                    io::stdout().flush().context("failed to flush stdout")?;
                    let Some(guess) = read_line(&mut io::stdin().lock())? else {
                        break;
                    };
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
                    let Some(feedback) = read_line(&mut io::stdin().lock())? else {
                        break;
                    };
                    match try_append_observation(
                        &observations,
                        &guess,
                        &feedback,
                        GameRules::Absurdle,
                        |next| solver.absurdle_apply_history(next).map(|_| ()),
                    ) {
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
                    let terminal = game_status(&observations, GameRules::Wordle)?;
                    if terminal != GameStatus::Active {
                        println!("game={}", terminal.label());
                        break;
                    }
                    let state = runtime.apply_history(&observations)?;
                    println!(
                        "mode=formal-optimal model={} manifest={} surviving={}",
                        runtime.manifest().model_id,
                        runtime.manifest().manifest_hash,
                        state.count()
                    );
                    let explanation = runtime.explain_state(&state, top)?;
                    println!(
                        "alternatives={}",
                        formal_alternatives_label(explanation.alternatives_status)
                    );
                    for suggestion in explanation.tied_moves {
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
                    let Some(guess) = read_line(&mut io::stdin().lock())? else {
                        break;
                    };
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
                    let Some(feedback) = read_line(&mut io::stdin().lock())? else {
                        break;
                    };
                    match try_append_observation(
                        &observations,
                        &guess,
                        &feedback,
                        GameRules::Wordle,
                        |next| runtime.apply_history(next).map(|_| ()),
                    ) {
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
                "alternatives={}",
                formal_alternatives_label(explanation.alternatives_status)
            );
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
            let from = parse_date(Some(&from))?.expect("--from is required by clap");
            let to = parse_date(Some(&to))?.expect("--to is required by clap");
            let requested = validate_development_target_range(&paths, from, to, "backtest")?;
            let backtest_config = backtest_config
                .as_deref()
                .map(PriorConfig::load)
                .transpose()?
                .unwrap_or_else(|| config.clone());
            let solver = Solver::from_paths(&paths, &backtest_config)?;
            let report = solver.backtest_detailed(requested.start, requested.end, top)?;
            let stats = &report.summary;
            let canonical = &stats.canonical;
            println!(
                "scheduled_games={} modeled_games={} solved_games={} unsolved_games={} coverage_gaps={} coverage_rate={:.6} solve_rate={:.6} conditional_mean_guesses={} all_game_penalized_mean_guesses={:.4} all_game_penalized_mean_guesses_ci95={:.4}..{:.4} failure_penalty_guesses={:.1} median={} p90={} p95={} max={} solved_distribution={}",
                canonical.scheduled_games,
                canonical.modeled_games,
                canonical.solved_games,
                canonical.unsolved_games,
                canonical.coverage_gaps,
                canonical.coverage_rate,
                canonical.solve_rate,
                canonical.conditional_mean_summary(),
                canonical.all_game_penalized_mean_guesses,
                canonical.all_game_penalized_mean_guesses_ci95.lower,
                canonical.all_game_penalized_mean_guesses_ci95.upper,
                canonical.failure_penalty_guesses,
                format_optional_metric(canonical.median_guesses, 2),
                format_optional_metric(canonical.p90_guesses, 0),
                format_optional_metric(canonical.p95_guesses, 0),
                format_optional_metric(canonical.max_guesses, 0),
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
                                    .map(|value| format!(" continuation_cost={:.5}", value))
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
        Command::PredictiveAblations {
            from,
            to,
            top,
            profile,
        } => {
            let (default_from, default_to) = Solver::latest_history_range(&paths)?
                .ok_or_else(|| anyhow!("run sync-data before predictive-ablations"))?;
            let from = parse_date(from.as_deref())?.unwrap_or(default_from);
            let to = parse_date(to.as_deref())?.unwrap_or(default_to);
            if from > to {
                bail!("--from cannot be after --to");
            }
            for row in Solver::predictive_ablation_report_filtered(
                &paths,
                &config,
                from,
                to,
                top,
                profile.as_deref(),
            )? {
                println!(
                    "label={} config={} mode={} variant={} games={} prior_measured_games={} avg_guesses={} p95={} max={} failures={} avg_target_prob={} avg_target_rank={} latency_p95_ms={:.3} session_cold_ms={} session_warm_ms={} lookahead_pool_ratio={:.3} exact_pool_ratio={:.3}",
                    row.label,
                    row.result.config_id,
                    row.result.mode.label(),
                    row.result.variant.label(),
                    row.result.backtest.games,
                    row.result
                        .prior_evidence
                        .as_ref()
                        .map_or(0, |prior| prior.measured_games),
                    format_optional_metric(row.result.backtest.average_guesses, 4),
                    format_optional_metric(row.result.backtest.p95_guesses, 0),
                    format_optional_metric(row.result.backtest.max_guesses, 0),
                    row.result.backtest.failures,
                    format_optional_prior_metric(row.result.average_target_probability, 6),
                    format_optional_prior_metric(row.result.average_target_rank, 2),
                    row.result.latency_p95_ms,
                    row.result
                        .session_fallback_cold_ms
                        .map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
                    row.result
                        .session_fallback_warm_ms
                        .map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
                    row.result.average_lookahead_pool_ratio,
                    row.result.average_exact_pool_ratio,
                );
            }
        }
        Command::PriorAblations { profiles, output } => {
            let report = Solver::predictive_prior_ablation_report(&paths, &config, &profiles)?;
            atomic_write(&output, &serde_json::to_vec_pretty(&report)?)?;
            println!(
                "prior_ablations={} profiles={} folds={} elapsed_ms={}",
                output.display(),
                report.profiles.len(),
                report.evaluation_plan.folds.len(),
                report.elapsed_ms
            );
            for profile in report.profiles {
                println!(
                    "label={} coverage_gaps={} measured_games={}/{} log_loss={} brier={} promotable={}",
                    profile.label,
                    profile.coverage_gaps,
                    profile.measured_games,
                    profile.scheduled_games,
                    format_optional_prior_metric(profile.average_log_loss, 6),
                    format_optional_prior_metric(profile.average_brier, 6),
                    profile.promotable
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
                    "avg_guesses={} failures={} coverage_gaps={} latency_p95_ms={:.3} hard_case_avg_guesses={:.4} hard_case_failures={}",
                    format_optional_metric(evaluation.average_guesses, 4),
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
                "games={} base_avg_guesses={} aggressive_case_avg_guesses={:.4} base_four_guess_cases={} aggressive_four_guess_cases={} converted_by_aggressive={} converted_by_targeted_search={}",
                report.games,
                format_optional_metric(report.base_average_guesses, 4),
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
            let from = parse_date(Some(&from))?.expect("--from is required by clap");
            let to = parse_date(Some(&to))?.expect("--to is required by clap");
            let requested = validate_development_target_range(&paths, from, to, "experiments")?;
            for mode in [
                WeightMode::Uniform,
                WeightMode::CooldownOnly,
                WeightMode::Weighted,
            ] {
                for variant in [ModelVariant::SeedOnly, ModelVariant::SeedPlusHistory] {
                    let solver = Solver::from_paths_with_settings(&paths, &config, mode, variant)?;
                    let result = solver.experiment_report(requested.start, requested.end, top)?;
                    println!(
                        "config={} mode={} variant={} games={} prior_measured_games={} avg_guesses={} p95={} max={} failures={} avg_log_loss={} avg_brier={} avg_target_prob={} avg_target_rank={} latency_p95_ms={:.3} session_cold_ms={} session_warm_ms={} lookahead_pool_ratio={:.3} exact_pool_ratio={:.3}",
                        result.config_id,
                        result.mode.label(),
                        result.variant.label(),
                        result.backtest.games,
                        result
                            .prior_evidence
                            .as_ref()
                            .map_or(0, |prior| prior.measured_games),
                        format_optional_metric(result.backtest.average_guesses, 4),
                        format_optional_metric(result.backtest.p95_guesses, 0),
                        format_optional_metric(result.backtest.max_guesses, 0),
                        result.backtest.failures,
                        format_optional_prior_metric(result.average_log_loss, 6),
                        format_optional_prior_metric(result.average_brier, 6),
                        format_optional_prior_metric(result.average_target_probability, 6),
                        format_optional_prior_metric(result.average_target_rank, 2),
                        result.latency_p95_ms,
                        result
                            .session_fallback_cold_ms
                            .map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
                        result
                            .session_fallback_warm_ms
                            .map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.3}")),
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
            let policy = EvaluationPolicy::load(&paths.root.join("config/evaluation.toml"))?;
            let plan = build_declared_rolling_origin_plan(
                DateRange::new(history_start, history_end)?,
                RollingOriginConfig {
                    minimum_training_days,
                    validation_days,
                    step_days,
                    sealed_test_days,
                    maximum_folds,
                },
                &policy,
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
                "rolling_folds={} train_span={}..{} validation_span={}..{} sealed_test_window={}..{} sealed_test_evaluated=false current_conditional_mean_guesses={} current_all_game_penalized_mean_guesses={:.4} current_failures={} current_coverage_gaps={} current_prior_measured_games={}/{} current_log_loss={} current_target_rank={} current_latency_p95_ms={:.3} current_hard_case_avg_guesses={:.4} current_hard_case_failures={} current_regime_mix=proxy:{:.1}%/lookahead:{:.1}%/escalated_exact:{:.1}%/exact:{:.1}%",
                summary.evaluation_plan.folds.len(),
                summary.search_window_start,
                summary.search_window_end,
                summary.validation_window_start,
                summary.validation_window_end,
                summary.test_window_start,
                summary.test_window_end,
                format_optional_metric(summary.current.average_guesses, 4),
                summary.current.all_game_penalized_mean_guesses,
                summary.current.failures,
                summary.current.coverage_gaps,
                summary.current.measured_prior_games,
                summary.current.scheduled_games,
                format_optional_prior_metric(summary.current.average_log_loss, 6),
                format_optional_prior_metric(summary.current.average_target_rank, 2),
                summary.current.latency_p95_ms,
                summary.current.hard_case_average_guesses,
                summary.current.hard_case_failures,
                summary.current.proxy_step_pct * 100.0,
                summary.current.lookahead_step_pct * 100.0,
                summary.current.escalated_exact_step_pct * 100.0,
                summary.current.exact_step_pct * 100.0
            );
            println!(
                "current_finite_step_pct={:.1}% current_terminal_step_pct={:.1}%",
                summary.current.finite_step_pct * 100.0,
                summary.current.terminal_step_pct * 100.0
            );
            println!(
                "best_conditional_mean_guesses={} best_all_game_penalized_mean_guesses={:.4} best_failures={} best_coverage_gaps={} best_prior_measured_games={}/{} best_log_loss={} best_target_rank={} best_latency_p95_ms={:.3} best_hard_case_avg_guesses={:.4} best_hard_case_failures={} best_regime_mix=proxy:{:.1}%/lookahead:{:.1}%/escalated_exact:{:.1}%/exact:{:.1}%",
                format_optional_metric(summary.best.average_guesses, 4),
                summary.best.all_game_penalized_mean_guesses,
                summary.best.failures,
                summary.best.coverage_gaps,
                summary.best.measured_prior_games,
                summary.best.scheduled_games,
                format_optional_prior_metric(summary.best.average_log_loss, 6),
                format_optional_prior_metric(summary.best.average_target_rank, 2),
                summary.best.latency_p95_ms,
                summary.best.hard_case_average_guesses,
                summary.best.hard_case_failures,
                summary.best.proxy_step_pct * 100.0,
                summary.best.lookahead_step_pct * 100.0,
                summary.best.escalated_exact_step_pct * 100.0,
                summary.best.exact_step_pct * 100.0
            );
            println!(
                "best_finite_step_pct={:.1}% best_terminal_step_pct={:.1}%",
                summary.best.finite_step_pct * 100.0,
                summary.best.terminal_step_pct * 100.0
            );
            println!("{}", summary.replacement_toml.trim_end());
        }
        Command::FitProxyWeights => {
            let maximum_validation_folds = Solver::development_evaluation_plan(&paths)?.folds.len();
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
                &paths.root.join("target/studies/fit-proxy-weights-v18.json"),
                5,
                None,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::LearnedProxyExperiment {
            minimum_survivors,
            maximum_survivors,
            maximum_states_per_split,
            guesses_per_state,
            maximum_seconds,
            maximum_memory_mb,
            checkpoint,
            dataset_output,
            output,
        } => {
            let solver = Solver::from_paths(&paths, &config)?;
            let dataset = solver.learned_proxy_dataset(
                &paths,
                LearnedProxyDatasetRequest {
                    minimum_survivors,
                    maximum_survivors,
                    maximum_states_per_split,
                    guesses_per_state,
                    maximum_seconds,
                    maximum_memory_mb,
                    checkpoint_path: Some(checkpoint),
                },
            )?;
            atomic_write(&dataset_output, dataset.to_json()?.as_bytes())?;
            let mut report = fit_learned_proxy_experiment(&dataset)?;
            let test_window = dataset
                .split
                .chronological
                .as_ref()
                .ok_or_else(|| anyhow!("learned-proxy dataset has no chronological split"))?;
            report.reference_search_regret = Some(solver.search_regret_report(
                &paths,
                SearchRegretRequest {
                    from: test_window.test_start,
                    to: test_window.test_end,
                    minimum_survivors,
                    maximum_survivors: maximum_survivors.min(6),
                    maximum_states: maximum_states_per_split,
                    maximum_seconds: maximum_seconds.clamp(60, 600),
                },
            )?);
            atomic_write(&output, &serde_json::to_vec_pretty(&report)?)?;
            println!(
                "learned_proxy={} dataset={} rows={} lambda={} validation_regret={:.6} test_regret={:.6} promotable={}",
                output.display(),
                dataset_output.display(),
                dataset.rows.len(),
                report.selected_lambda,
                report.learned_validation.mean_regret,
                report.learned_test.mean_regret,
                report.promotable
            );
        }
        Command::SurvivalExperiment { output } => {
            let solver = Solver::from_paths(&paths, &config)?;
            let report = run_survival_experiment(&paths, &config, &solver)?;
            atomic_write(&output, &serde_json::to_vec_pretty(&report)?)?;
            println!(
                "survival={} folds={} reuse_events={} logistic_logloss={} survival_logloss={} promotable={}",
                output.display(),
                report.folds.len(),
                report.total_reuse_events,
                report.logistic.mean_log_loss.map_or_else(
                    || format!(
                        "unavailable (covered_games={})",
                        report.logistic.covered_games
                    ),
                    |value| format!("{value:.6}"),
                ),
                report.survival.mean_log_loss.map_or_else(
                    || format!(
                        "unavailable (covered_games={})",
                        report.survival.covered_games
                    ),
                    |value| format!("{value:.6}"),
                ),
                report.promotable
            );
        }
        Command::SearchRegret {
            config: audit_config,
            from,
            to,
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
            finite,
            hard_mode,
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
            let maximum_states = maximum_states.unwrap_or(if finite { 6 } else { 16 });
            let maximum_seconds = maximum_seconds.unwrap_or(if finite { 60 } else { 1800 });
            let solver = Solver::from_paths(&paths, &audit_config)?;
            if finite {
                let report = solver.finite_search_regret_report(
                    &paths,
                    FiniteSearchRegretRequest {
                        from,
                        to,
                        minimum_survivors,
                        maximum_survivors,
                        maximum_states,
                        maximum_seconds,
                        hard_mode,
                    },
                )?;
                let encoded = serde_json::to_vec_pretty(&report)
                    .context("failed to encode finite search-regret JSON")?;
                atomic_write(&output, &encoded)?;
                println!(
                    "finite_search_regret={} states={} available={} resolved={} failure_regret_states={}",
                    output.display(),
                    report.sampled_states,
                    report.available_states,
                    report.summary.resolved_states,
                    report.summary.failure_regret_states,
                );
            } else {
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
                    "search_regret={} status={} states={}/{} available={} stop_reason={} production_mean={} proxy_mean={} lookahead_mean={}",
                    output.display(),
                    if report.complete {
                        "complete"
                    } else {
                        "incomplete"
                    },
                    report.sampled_states,
                    report.planned_states,
                    report.available_states,
                    report.stop_reason.as_deref().unwrap_or("none"),
                    format_optional_metric_for_population(
                        report.production.mean_regret,
                        6,
                        "sampled_states"
                    ),
                    format_optional_metric_for_population(
                        report.proxy.mean_regret,
                        6,
                        "sampled_states"
                    ),
                    format_optional_metric_for_population(
                        report.lookahead.mean_regret,
                        6,
                        "sampled_states"
                    ),
                );
            }
        }
        Command::SameStateDynamicRegret {
            config: audit_config,
            date,
            turn,
            maximum_seconds,
            output,
        } => {
            let audit_config = audit_config
                .as_deref()
                .map(PriorConfig::load)
                .transpose()?
                .unwrap_or_else(|| config.clone());
            let date = parse_date(Some(&date))?
                .ok_or_else(|| anyhow!("--date is required for same-state-dynamic-regret"))?;
            let solver = Solver::from_paths(&paths, &audit_config)?;
            let report =
                solver.same_state_dynamic_regret_report(&paths, date, turn, maximum_seconds)?;
            let encoded = serde_json::to_vec_pretty(&report)
                .context("failed to encode same-state dynamic-regret JSON")?;
            atomic_write(&output, &encoded)?;
            println!(
                "same_state_dynamic_regret={} date={} turn={} reference={} staged={} finite_fast_dynamic={}",
                output.display(),
                report.date,
                report.turn,
                report.reference_status,
                report.staged.reference_status,
                report.finite_fast_dynamic.reference_status,
            );
        }
        Command::StagedZeroFailureCertificate {
            config: audit_config,
            from,
            to,
            maximum_states,
            maximum_seconds,
            hard_mode,
            output,
        } => {
            if maximum_states == 0 {
                bail!("--maximum-states must be greater than zero");
            }
            if maximum_seconds == 0 {
                bail!("--maximum-seconds must be greater than zero");
            }
            let from = parse_date(Some(&from))?.expect("--from is required by clap");
            let to = parse_date(Some(&to))?.expect("--to is required by clap");
            let requested = validate_development_target_range(
                &paths,
                from,
                to,
                "staged zero-failure certificate",
            )?;
            let audit_config = audit_config
                .as_deref()
                .map(PriorConfig::load)
                .transpose()?
                .unwrap_or_else(|| config.clone());
            let solver = Solver::from_paths(&paths, &audit_config)?;
            let report = solver.staged_zero_failure_certificate_report(
                &paths,
                StagedZeroFailureCertificateRequest {
                    from: requested.start,
                    to: requested.end,
                    maximum_states,
                    maximum_seconds,
                    hard_mode,
                },
            )?;
            let encoded = serde_json::to_vec_pretty(&report)
                .context("failed to encode staged zero-failure certificate JSON")?;
            atomic_write(&output, &encoded)?;
            println!(
                "staged_zero_failure_certificate={} complete={} selected_roots={} evaluated_roots={} certified_roots={} coverage_gaps={} duplicate_history_dates={} unsupported_target_games={} replayed_games={} path_replay_failures={} state_cap_reached={} deadline_reached={}",
                output.display(),
                report.complete,
                report.selected_roots,
                report.evaluated_roots,
                report.certified_roots,
                report.coverage_gaps,
                report.duplicate_history_dates,
                report.unsupported_target_games,
                report.replayed_games,
                report.path_replay_failures,
                report.state_cap_reached,
                report.deadline_reached,
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
            rolling_folds,
            matrix,
            top,
            maximum_seconds,
            maximum_memory_mb,
            output,
            markdown_output,
            checkpoint,
        } => {
            let selection = if rolling_folds {
                EvidenceDateSelection::RollingFolds
            } else {
                let from = from
                    .as_deref()
                    .ok_or_else(|| anyhow!("--from is required unless --rolling-folds is used"))?;
                let to = to
                    .as_deref()
                    .ok_or_else(|| anyhow!("--to is required unless --rolling-folds is used"))?;
                let from = NaiveDate::parse_from_str(from, "%Y-%m-%d")
                    .with_context(|| format!("invalid --from date: {from}"))?;
                let to = NaiveDate::parse_from_str(to, "%Y-%m-%d")
                    .with_context(|| format!("invalid --to date: {to}"))?;
                if from > to {
                    bail!("--from cannot be after --to");
                }
                EvidenceDateSelection::Range(DateRange::new(from, to)?)
            };
            let artifact = Solver::build_development_evidence_with_selection(
                &paths,
                &config,
                selection,
                top,
                EvidenceResourceBudget {
                    maximum_seconds,
                    maximum_memory_mb,
                },
                matrix.as_deref(),
                Some(&checkpoint),
            )?;
            atomic_write(
                &output,
                &serde_json::to_vec(&artifact)
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
        } => evidence::benchmark_docs(&evidence, &markdown_output, &readme, update)?,
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
        Command::FreezeProspective {
            config,
            comparison,
            output,
        } => {
            let frozen =
                Solver::freeze_prospective_candidate(&paths, &config, &comparison, Utc::now())?;
            let output = if output.is_absolute() {
                output
            } else {
                paths.root.join(output)
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            Solver::write_prospective_frozen_candidate(&output, &frozen)?;
            println!(
                "candidate={} freeze={} frozen_at_utc={} window={}..{} sealed_test_evaluated=false output={}",
                frozen.frozen.candidate_label,
                frozen.window_fingerprint,
                frozen.frozen_at_utc,
                frozen.window.start,
                frozen.window.end,
                output.display()
            );
        }
        Command::EvaluateProspective { frozen, output } => {
            let bytes = std::fs::read(&frozen)
                .with_context(|| format!("failed to read {}", frozen.display()))?;
            let frozen: maybe_wordle::solver::ProspectiveFrozenCandidate =
                serde_json::from_slice(&bytes)
                    .with_context(|| format!("failed to parse {}", frozen.display()))?;
            let report = Solver::evaluate_prospective_candidate(&paths, &frozen, &output)?;
            println!(
                "prospective_games={} solved={} failures={} coverage_gaps={} all_game_mean={:.4} ci95={:.4}..{:.4} latency_p95_ms={:.3} evaluated_once=true window={}..{} output={}",
                report.metrics.scheduled_games,
                report.metrics.solved_games,
                report.metrics.unsolved_games,
                report.metrics.coverage_gaps,
                report.metrics.all_game_penalized_mean_guesses,
                report.metrics.all_game_penalized_mean_guesses_ci95.lower,
                report.metrics.all_game_penalized_mean_guesses_ci95.upper,
                report.latency_p95_ms,
                report.window.start,
                report.window.end,
                output.display()
            );
        }
        Command::RollingEvidenceDocs {
            comparison,
            markdown_output,
            readme,
            update,
        } => evidence::rolling_docs(&comparison, &markdown_output, &readme, update)?,
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

fn validate_development_target_range(
    paths: &ProjectPaths,
    from: NaiveDate,
    to: NaiveDate,
    operation: &str,
) -> Result<DateRange> {
    if from > to {
        bail!("--from cannot be after --to");
    }
    let requested = DateRange::new(from, to)?;
    let policy = EvaluationPolicy::load(&paths.root.join("config/evaluation.toml"))?;
    policy
        .validate_development_target_range(requested)
        .with_context(|| format!("cannot validate {operation} target range"))?;
    Ok(requested)
}

fn warn_predictive_history_range(paths: &ProjectPaths, puzzle_date: NaiveDate) -> Result<()> {
    let Some((first_synced, last_synced)) = Solver::latest_history_range(paths)? else {
        eprintln!(
            "warning: no synced NYT history found; run cargo run -- sync-data before relying on predictive history or artifacts"
        );
        return Ok(());
    };
    let history_cutoff = history_cutoff(puzzle_date)?;
    for warning in
        predictive_history_range_warnings(puzzle_date, history_cutoff, first_synced, last_synced)
    {
        eprintln!("warning: {warning}");
    }
    Ok(())
}

fn predictive_history_range_warnings(
    puzzle_date: NaiveDate,
    history_cutoff: NaiveDate,
    first_synced: NaiveDate,
    last_synced: NaiveDate,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if history_cutoff < first_synced {
        warnings.push(format!(
            "requested puzzle date {} has usable history cutoff {} before the earliest synced NYT date {}; predictive history and artifacts may be incomplete",
            puzzle_date, history_cutoff, first_synced
        ));
    }
    if history_cutoff > last_synced {
        warnings.push(format!(
            "usable history cutoff {} for requested puzzle date {} is after the latest synced NYT date {}; predictive history and artifacts may be stale",
            history_cutoff, puzzle_date, last_synced
        ));
    }
    warnings
}

fn formal_alternatives_label(status: FormalAlternativesStatus) -> &'static str {
    match status {
        FormalAlternativesStatus::NotRequested => "not_requested",
        FormalAlternativesStatus::Complete => "complete",
        FormalAlternativesStatus::WorkLimitReached => "work_limit_reached_exact_primary_retained",
    }
}

fn format_predictive_context(puzzle_date: NaiveDate, history_cutoff: NaiveDate) -> String {
    format!("puzzle_date={puzzle_date} history_cutoff={history_cutoff}")
}

fn format_sync_summary(summary: &SyncSummary) -> String {
    let status = if summary.partial_sync {
        "partial"
    } else {
        "complete"
    };
    format!(
        "sync_status={} entries={} range={}..{} requested={}..{} attempted={} fetched={} reverified={} applied={} retained={} changed={} coverage_complete={} cancelled={}",
        status,
        summary.total,
        summary.first_date,
        summary.last_date,
        summary.requested_first_date,
        summary.requested_last_date,
        summary.attempted,
        summary.fetched,
        summary.reverified,
        summary.applied,
        summary.retained,
        summary.changed,
        summary.coverage_complete,
        summary.cancelled
    )
}

fn enforce_sync_policy(strict: bool, summary: &SyncSummary) -> Result<()> {
    if strict && summary.partial_sync {
        bail!(
            "partial sync encountered failed dates: {}; missing requested dates: {}; cancelled={}",
            summary
                .failed_dates
                .iter()
                .map(|date| date.format("%Y-%m-%d").to_string())
                .collect::<Vec<_>>()
                .join(","),
            summary
                .missing_dates
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            summary.cancelled
        );
    }
    Ok(())
}

fn predictive_warning_lines(
    as_of: NaiveDate,
    observations: &[(String, u8)],
    mode: PredictiveSuggestionMode,
    _hard_mode: bool,
    response: &PredictiveSuggestResponse,
) -> Vec<String> {
    if response.execution.route == maybe_wordle::predictive::PredictiveRegime::Finite {
        return Vec::new();
    }
    let mut warnings = Vec::new();
    if response.execution.action_scope == SearchActionScope::HardRootNormalContinuation {
        warnings.push("legacy continuation costs assume normal-mode replies; finite profiles enforce hard-mode legality recursively".to_string());
    }
    let artifact_date = response
        .promoted_artifact_date
        .map_or_else(|| "unreported".to_string(), |date| date.to_string());
    let artifact_warning = match response.artifact_state {
        maybe_wordle::predictive::PredictiveArtifactState::ExactDateArtifact => {
            if observations.is_empty() {
                Some(format!(
                    "exact-date predictive opener artifact dated {artifact_date} is available for puzzle {as_of}"
                ))
            } else {
                Some(format!(
                    "exact-date predictive reply-book artifact dated {artifact_date} is available for puzzle {as_of}"
                ))
            }
        }
        maybe_wordle::predictive::PredictiveArtifactState::RecentOpenerArtifact => Some(format!(
            "no exact-date opener artifact for {as_of}; reusing a recent opener artifact dated {artifact_date}"
        )),
        maybe_wordle::predictive::PredictiveArtifactState::RecentReplyArtifact => Some(format!(
            "no exact-date reply artifact for {as_of}; reusing a recent reply-book artifact dated {artifact_date}"
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
        && !matches!(
            response.artifact_state,
            maybe_wordle::predictive::PredictiveArtifactState::ExactDateArtifact
                | maybe_wordle::predictive::PredictiveArtifactState::RecentReplyArtifact
        )
    {
        warnings.push(
            "reply-book artifact is missing for this date or branch; branch suggestions are coming from live evaluation".to_string(),
        );
    }
    warnings
}

fn read_line(reader: &mut impl io::BufRead) -> Result<Option<String>> {
    let mut buffer = String::new();
    let bytes = reader
        .read_line(&mut buffer)
        .context("failed to read stdin")?;
    Ok((bytes != 0).then_some(buffer))
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

fn format_predictive_suggestion(suggestion: &maybe_wordle::solver::Suggestion) -> String {
    let mut line = format!(
        "{} value_kind=\"{}\" entropy={:.5} solve_prob={:.5} expected_remaining={:.3}",
        suggestion.word,
        suggestion.value_kind.label(),
        suggestion.entropy,
        suggestion.solve_probability,
        suggestion.expected_remaining
    );
    if suggestion.force_in_two {
        line.push_str(" force_in_two=true");
    }
    if let Some(value) = suggestion.finite_value {
        if value.quality == maybe_wordle::solver::FiniteSearchQuality::Heuristic {
            line.push_str(" finite_value=unevaluated");
        } else {
            line.push_str(&format!(
                " modeled_failure_prob={:.6} expected_attempts_remaining={:.5} value_quality={:?}",
                value.failure_probability, value.expected_attempts, value.quality
            ));
        }
    }
    if let Some(exact_cost) = suggestion.exact_cost {
        let label = if suggestion.value_kind == SuggestionValueKind::ExactAction {
            "model_exact_action_cost"
        } else {
            "continuation_estimate"
        };
        line.push_str(&format!(" {label}={exact_cost:.5}"));
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
        "used_unused" | "used-unused" => Ok(WeightMode::UsedUnused),
        "recency_buckets" | "recency-buckets" => Ok(WeightMode::RecencyBuckets),
        "empirical_frequency" | "empirical-frequency" => Ok(WeightMode::EmpiricalFrequency),
        "regularized_frequency" | "regularized-frequency" => Ok(WeightMode::RegularizedFrequency),
        _ => bail!(
            "weight mode must be one of: weighted, uniform, cooldown_only, used_unused, recency_buckets, empirical_frequency, regularized_frequency"
        ),
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

fn format_optional_prior_metric(value: Option<f64>, precision: usize) -> String {
    format_optional_metric_for_population(value, precision, "measured_prior_games")
}

fn format_optional_metric<T: std::fmt::Display>(value: Option<T>, precision: usize) -> String {
    format_optional_metric_for_population(value, precision, "modeled_games")
}

fn format_optional_metric_for_population<T: std::fmt::Display>(
    value: Option<T>,
    precision: usize,
    population: &str,
) -> String {
    value.map_or_else(
        || format!("unavailable ({population}=0)"),
        |value| format!("{value:.precision$}"),
    )
}

#[cfg(test)]
#[path = "test_support.rs"]
mod test_support;

#[cfg(test)]
mod tests {
    #[test]
    fn formal_alternative_labels_distinguish_bounded_primary_from_complete_ranking() {
        use super::{FormalAlternativesStatus, formal_alternatives_label};
        assert_eq!(
            formal_alternatives_label(FormalAlternativesStatus::NotRequested),
            "not_requested"
        );
        assert_eq!(
            formal_alternatives_label(FormalAlternativesStatus::Complete),
            "complete"
        );
        assert_eq!(
            formal_alternatives_label(FormalAlternativesStatus::WorkLimitReached),
            "work_limit_reached_exact_primary_retained"
        );
    }

    #[test]
    fn optional_metric_output_is_explicit_about_an_empty_population() {
        assert_eq!(
            super::format_optional_metric(None::<f64>, 4),
            "unavailable (modeled_games=0)"
        );
        assert_eq!(super::format_optional_metric(Some(3.25), 4), "3.2500");
        assert_eq!(super::format_optional_metric(Some(6usize), 0), "6");
    }

    #[test]
    fn regret_output_distinguishes_measured_zero_from_no_completed_states() {
        assert_eq!(
            super::format_optional_metric_for_population(None::<f64>, 6, "sampled_states"),
            "unavailable (sampled_states=0)"
        );
        assert_eq!(
            super::format_optional_metric_for_population(Some(0.0), 6, "sampled_states"),
            "0.000000"
        );
    }

    use std::{fs, path::PathBuf};

    use anyhow::anyhow;
    use chrono::NaiveDate;
    use clap::{CommandFactory, Parser};
    use maybe_wordle::data::SyncSummary;
    use maybe_wordle::experiments::StudySearchStrategy;
    use maybe_wordle::predictive::{PredictiveArtifactState, PredictiveSuggestResponse};
    use maybe_wordle::solver::AbsurdleSuggestion;

    use super::{
        Cli, Command, enforce_sync_policy, find_project_root, format_absurdle_suggestion,
        format_predictive_context, format_predictive_suggestion, format_sync_summary,
        normalize_interactive_guess, parse_solver_mode, parse_study_stage, parse_study_strategy,
        predictive_cli_mode, predictive_history_range_warnings, predictive_warning_lines,
        reject_hard_mode_for_non_predictive, reject_live_fallback_for_non_predictive,
        try_append_observation,
    };

    #[test]
    fn sync_data_accepts_an_explicit_safe_cutoff() {
        let cli = Cli::try_parse_from([
            "maybe-wordle",
            "sync-data",
            "--strict",
            "--through",
            "2026-08-26",
        ])
        .expect("CLI");
        match cli.command {
            Command::SyncData { strict, through } => {
                assert!(strict);
                assert_eq!(through.as_deref(), Some("2026-08-26"));
            }
            _ => panic!("expected sync-data"),
        }
    }

    #[test]
    fn predictive_ablations_accepts_a_profile_filter() {
        let cli = Cli::try_parse_from([
            "maybe-wordle",
            "predictive-ablations",
            "--profile",
            "regularized_frequency_baseline",
        ])
        .expect("CLI");
        match cli.command {
            Command::PredictiveAblations { profile, .. } => {
                assert_eq!(profile.as_deref(), Some("regularized_frequency_baseline"));
            }
            _ => panic!("expected predictive-ablations"),
        }
    }

    #[test]
    fn backtest_and_experiments_require_both_evaluation_range_flags() {
        assert!(Cli::try_parse_from(["maybe-wordle", "backtest"]).is_err());
        assert!(
            Cli::try_parse_from(["maybe-wordle", "backtest", "--from", "2026-08-01",]).is_err()
        );
        assert!(Cli::try_parse_from(["maybe-wordle", "experiments"]).is_err());
        assert!(
            Cli::try_parse_from(["maybe-wordle", "experiments", "--to", "2026-08-01",]).is_err()
        );
    }

    #[test]
    fn evaluate_prospective_defaults_to_private_diagnostic_output() {
        let cli = Cli::try_parse_from(["maybe-wordle", "evaluate-prospective"])
            .expect("evaluate-prospective CLI");
        match cli.command {
            Command::EvaluateProspective { frozen, output } => {
                assert_eq!(
                    frozen,
                    PathBuf::from("benchmarks/predictive/prospective-frozen-v1.json")
                );
                assert_eq!(
                    output,
                    PathBuf::from("target/diagnostics/prospective-window-v1.json")
                );
                assert!(!output.to_string_lossy().contains("benchmarks/predictive"));
            }
            _ => panic!("expected evaluate-prospective"),
        }
    }

    #[test]
    fn interactive_wordle_transition_rejects_terminal_and_short_rows() {
        for (history, guess) in [
            (vec![("cigar".to_string(), 242)], "rebut"),
            (vec![("cigar".to_string(), 0); 6], "rebut"),
            (Vec::new(), "four"),
        ] {
            assert!(
                try_append_observation(&history, guess, "00000", super::GameRules::Wordle, |_| Ok(
                    ()
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn try_append_observation_rejects_invalid_feedback_without_mutation() {
        let observations = vec![("crane".to_string(), 0)];
        let result = try_append_observation(
            &observations,
            "slate",
            "oops",
            super::GameRules::Wordle,
            |_| Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(observations.len(), 1);
    }

    #[test]
    fn try_append_observation_rejects_contradictions_without_mutation() {
        let observations = vec![("crane".to_string(), 0)];
        let result = try_append_observation(
            &observations,
            "slate",
            "00000",
            super::GameRules::Wordle,
            |_| Err(anyhow!("no answers remain")),
        );
        assert!(result.is_err());
        assert_eq!(observations.len(), 1);
    }

    #[test]
    fn try_append_observation_commits_valid_observation() {
        let observations = vec![("crane".to_string(), 0)];
        let result = try_append_observation(
            &observations,
            "slate",
            "00000",
            super::GameRules::Wordle,
            |_| Ok(()),
        )
        .expect("valid observation");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], observations[0]);
        assert_eq!(result[1].0, "slate");
    }

    #[test]
    fn interactive_eof_is_distinct_from_a_blank_or_final_line() {
        let mut input = std::io::Cursor::new(b"\nrebut");
        assert_eq!(super::read_line(&mut input).unwrap(), Some("\n".into()));
        assert_eq!(super::read_line(&mut input).unwrap(), Some("rebut".into()));
        assert_eq!(super::read_line(&mut input).unwrap(), None);
        assert_eq!(super::read_line(&mut input).unwrap(), None);
    }

    #[test]
    fn prior_metric_format_distinguishes_zero_loss_from_unmeasured_prior() {
        assert_eq!(super::format_optional_prior_metric(Some(0.0), 4), "0.0000");
        assert_eq!(
            super::format_optional_prior_metric(None, 4),
            "unavailable (measured_prior_games=0)"
        );
    }

    #[test]
    fn normalize_interactive_guess_rejects_unknown_guess() {
        let result = normalize_interactive_guess("slate", |guess| guess == "crane");
        assert_eq!(result.expect_err("must fail"), "unknown guess: slate");
    }

    #[test]
    fn predictive_suggestion_format_includes_force_in_two_marker() {
        let mut suggestion = maybe_wordle::solver::Suggestion {
            value_kind: maybe_wordle::predictive::types::SuggestionValueKind::Proxy,
            finite_value: None,
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
        };
        let formatted = format_predictive_suggestion(&suggestion);
        assert!(formatted.contains("force_in_two=true"));
        assert!(formatted.contains("value_kind=\"heuristic proxy\""));
        assert!(formatted.contains("continuation_estimate=2.50000"));
        assert!(!formatted.contains("exact_cost"));
        suggestion.value_kind = super::SuggestionValueKind::ExactAction;
        assert!(
            format_predictive_suggestion(&suggestion).contains("model_exact_action_cost=2.50000")
        );
        suggestion.value_kind = super::SuggestionValueKind::Finite(
            maybe_wordle::solver::FiniteSearchQuality::UpperBound,
        );
        suggestion.exact_cost = None;
        suggestion.finite_value = Some(maybe_wordle::solver::FiniteSearchCandidate {
            guess_index: 0,
            failure_probability: 0.1,
            expected_attempts: 2.4,
            quality: maybe_wordle::solver::FiniteSearchQuality::UpperBound,
        });
        let formatted = format_predictive_suggestion(&suggestion);
        assert!(formatted.contains("modeled_failure_prob=0.100000"));
        assert!(formatted.contains("expected_attempts_remaining=2.40000"));
        suggestion.finite_value.as_mut().unwrap().quality =
            maybe_wordle::solver::FiniteSearchQuality::Heuristic;
        suggestion.value_kind = super::SuggestionValueKind::Finite(
            maybe_wordle::solver::FiniteSearchQuality::Heuristic,
        );
        let formatted = format_predictive_suggestion(&suggestion);
        assert!(formatted.contains("finite_value=unevaluated"));
        assert!(!formatted.contains("modeled_failure_prob"));
    }

    #[test]
    fn suggest_accepts_only_declared_finite_budget_profiles() {
        let cli = Cli::try_parse_from(["maybe-wordle", "suggest", "--search-budget", "strong"])
            .expect("finite profile");
        assert!(
            matches!(cli.command, Command::Suggest { search_budget: Some(profile), .. } if profile == "strong")
        );
        assert!(
            Cli::try_parse_from(["maybe-wordle", "suggest", "--search-budget", "infinite"])
                .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "maybe-wordle",
                "suggest",
                "--search-budget",
                "fast",
                "--live-fallback"
            ])
            .is_err()
        );
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
    fn predictive_history_range_warning_uses_the_preceding_day_cutoff() {
        let first_synced = NaiveDate::from_ymd_opt(2024, 1, 1).expect("date");
        let last_synced = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");
        let next_puzzle = NaiveDate::from_ymd_opt(2024, 1, 5).expect("date");
        assert!(
            predictive_history_range_warnings(
                next_puzzle,
                NaiveDate::from_ymd_opt(2024, 1, 4).expect("cutoff"),
                first_synced,
                last_synced,
            )
            .is_empty()
        );

        let later_puzzle = NaiveDate::from_ymd_opt(2024, 1, 6).expect("date");
        let warnings = predictive_history_range_warnings(
            later_puzzle,
            NaiveDate::from_ymd_opt(2024, 1, 5).expect("cutoff"),
            first_synced,
            last_synced,
        );
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("history cutoff 2024-01-05"));
    }

    #[test]
    fn predictive_context_format_names_both_public_dates() {
        assert_eq!(
            format_predictive_context(
                NaiveDate::from_ymd_opt(2024, 1, 5).expect("puzzle date"),
                NaiveDate::from_ymd_opt(2024, 1, 4).expect("history cutoff"),
            ),
            "puzzle_date=2024-01-05 history_cutoff=2024-01-04"
        );
    }

    #[test]
    fn find_project_root_walks_up_from_nested_binary_path() {
        let fixture = super::test_support::TestDirectory::new("cli-root");
        let temp_root = fixture.path().to_path_buf();
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
        let mut response = PredictiveSuggestResponse {
            execution: maybe_wordle::predictive::types::SearchExecution {
                route: maybe_wordle::predictive::PredictiveRegime::Proxy,
                objective: maybe_wordle::predictive::types::SearchObjective::ProxyRanking,
                action_scope: super::SearchActionScope::HardRootNormalContinuation,
                candidate_scope: maybe_wordle::predictive::types::SearchCandidateScope::AllActions,
                roots_considered: 0,
                roots_evaluated: 0,
                selected_value_kind: None,
                root_selection_optimal: false,
                stop_reason: None,
            },
            finite_search: None,
            puzzle_date: NaiveDate::from_ymd_opt(2026, 3, 26).expect("date"),
            history_cutoff: NaiveDate::from_ymd_opt(2026, 3, 25).expect("date"),
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
            promoted_artifact_date: None,
            artifact_state: PredictiveArtifactState::LiveSessionFallback,
            model_version: "test".to_string(),
            model_manifest_hash: "test".to_string(),
            history_snapshot_date: None,
            history_snapshot_hash: "test".to_string(),
        };
        let warnings = predictive_warning_lines(
            response.puzzle_date,
            &[("crane".to_string(), 17)],
            maybe_wordle::predictive::PredictiveSuggestionMode::Full,
            true,
            &response,
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
        assert!(
            warnings
                .iter()
                .any(|line| line.contains("normal-mode replies"))
        );
        response.artifact_state = PredictiveArtifactState::RecentReplyArtifact;
        response.execution.action_scope = super::SearchActionScope::Normal;
        response.promotion_source =
            Some(maybe_wordle::predictive::PredictivePromotionSource::RecentReplyBook);
        response.promoted_artifact_date = Some(NaiveDate::from_ymd_opt(2026, 3, 22).unwrap());
        let warnings = predictive_warning_lines(
            response.puzzle_date,
            &[("crane".to_string(), 17)],
            maybe_wordle::predictive::PredictiveSuggestionMode::Full,
            false,
            &response,
        );
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("recent reply-book artifact dated 2026-03-22"));
        assert!(!warnings[0].contains("opener"));
        assert!(!warnings[0].contains("live evaluation"));
    }

    #[test]
    fn offline_budget_help_does_not_claim_hard_preemption() {
        for command in ["study-run", "benchmark-evidence"] {
            let mut cli = Cli::command();
            let help = cli
                .find_subcommand_mut(command)
                .unwrap()
                .render_long_help()
                .to_string();
            assert!(help.contains("across resumes"));
            assert!(help.contains("not a hard deadline"));
            assert!(help.contains("not an allocation cap"));
        }
    }

    fn partial_sync_summary() -> SyncSummary {
        let first = NaiveDate::from_ymd_opt(2021, 6, 19).expect("first");
        let last = NaiveDate::from_ymd_opt(2021, 6, 23).expect("last");
        let missing = NaiveDate::from_ymd_opt(2021, 6, 24).expect("missing");
        SyncSummary {
            attempted: 3,
            fetched: 2,
            reverified: 1,
            applied: 0,
            retained: 5,
            changed: 0,
            total: 5,
            first_date: first,
            last_date: last,
            changed_dates: Vec::new(),
            partial_sync: true,
            failed_dates: vec![missing],
            last_successful_date: Some(last),
            retained_existing_archive: true,
            coverage_complete: false,
            requested_first_date: first,
            requested_last_date: missing,
            missing_dates: vec![missing],
            failures: vec![maybe_wordle::data::SyncFailure {
                date: missing,
                message: "HTTP 500".into(),
            }],
            cancelled: false,
        }
    }

    #[test]
    fn format_sync_summary_marks_partial_sync() {
        let rendered = format_sync_summary(&partial_sync_summary());
        for field in [
            "sync_status=partial",
            "attempted=3",
            "applied=0",
            "retained=5",
            "coverage_complete=false",
        ] {
            assert!(rendered.contains(field), "{rendered}");
        }
    }

    #[test]
    fn strict_sync_policy_rejects_partial_sync() {
        let summary = partial_sync_summary();
        let error = enforce_sync_policy(true, &summary).expect_err("strict should fail");
        assert!(format!("{error:#}").contains("2021-06-24"));
    }

    #[test]
    fn non_strict_sync_policy_allows_partial_sync() {
        let summary = partial_sync_summary();
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
        assert!(opener_rendered.contains("Answer-weight model:"));
        assert!(opener_rendered.contains("used_unused"));

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

    #[test]
    fn benchmark_evidence_accepts_range_and_rolling_selection() {
        let range = Cli::try_parse_from([
            "maybe-wordle",
            "benchmark-evidence",
            "--from",
            "2026-01-01",
            "--to",
            "2026-01-31",
            "--matrix",
            "matrix.json",
            "--output",
            "evidence.json",
            "--markdown-output",
            "evidence.md",
        ])
        .expect("range selection should parse");
        match range.command {
            Command::BenchmarkEvidence {
                from,
                to,
                rolling_folds,
                matrix,
                ..
            } => {
                assert_eq!(from.as_deref(), Some("2026-01-01"));
                assert_eq!(to.as_deref(), Some("2026-01-31"));
                assert!(!rolling_folds);
                assert_eq!(matrix, Some(PathBuf::from("matrix.json")));
            }
            _ => panic!("expected benchmark-evidence"),
        }

        let rolling = Cli::try_parse_from([
            "maybe-wordle",
            "benchmark-evidence",
            "--rolling-folds",
            "--output",
            "evidence.json",
            "--markdown-output",
            "evidence.md",
        ])
        .expect("rolling selection should parse");
        match rolling.command {
            Command::BenchmarkEvidence {
                from,
                to,
                rolling_folds,
                ..
            } => {
                assert!(from.is_none());
                assert!(to.is_none());
                assert!(rolling_folds);
            }
            _ => panic!("expected benchmark-evidence"),
        }
    }

    #[test]
    fn search_regret_hard_mode_requires_finite_and_preserves_budget_overrides() {
        let mut arguments = vec![
            "maybe-wordle",
            "search-regret",
            "--from",
            "2026-08-01",
            "--to",
            "2026-08-02",
            "--output",
            "regret.json",
            "--hard-mode",
        ];
        assert!(Cli::try_parse_from(&arguments).is_err());
        arguments.extend([
            "--finite",
            "--maximum-states",
            "3",
            "--maximum-seconds",
            "20",
        ]);
        let parsed = Cli::try_parse_from(arguments).expect("finite hard-mode audit");
        match parsed.command {
            Command::SearchRegret {
                finite,
                hard_mode,
                maximum_states,
                maximum_seconds,
                ..
            } => {
                assert!(finite && hard_mode);
                assert_eq!(maximum_states, Some(3));
                assert_eq!(maximum_seconds, Some(20));
            }
            _ => panic!("expected search-regret"),
        }
    }

    #[test]
    fn same_state_dynamic_regret_is_an_explicit_single_path_command() {
        let parsed = Cli::try_parse_from([
            "maybe-wordle",
            "same-state-dynamic-regret",
            "--date",
            "2026-08-01",
            "--turn",
            "3",
            "--maximum-seconds",
            "20",
            "--output",
            "same-state.json",
        ])
        .expect("explicit same-state diagnostic");
        match parsed.command {
            Command::SameStateDynamicRegret {
                date,
                turn,
                maximum_seconds,
                output,
                ..
            } => {
                assert_eq!(date, "2026-08-01");
                assert_eq!(turn, 3);
                assert_eq!(maximum_seconds, 20);
                assert_eq!(output, PathBuf::from("same-state.json"));
            }
            _ => panic!("expected same-state-dynamic-regret"),
        }
    }

    #[test]
    fn staged_zero_failure_certificate_requires_explicit_range_and_bounds() {
        assert!(
            Cli::try_parse_from([
                "maybe-wordle",
                "staged-zero-failure-certificate",
                "--from",
                "2026-08-01",
                "--to",
                "2026-08-02",
                "--maximum-seconds",
                "1",
                "--output",
                "report.json",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "maybe-wordle",
                "staged-zero-failure-certificate",
                "--from",
                "2026-08-01",
                "--to",
                "2026-08-02",
                "--maximum-states",
                "1",
                "--output",
                "report.json",
            ])
            .is_err()
        );
        let parsed = Cli::try_parse_from([
            "maybe-wordle",
            "staged-zero-failure-certificate",
            "--from",
            "2026-08-01",
            "--to",
            "2026-08-02",
            "--maximum-states",
            "1",
            "--maximum-seconds",
            "1",
            "--hard-mode",
            "--output",
            "report.json",
        ])
        .expect("explicit certificate bounds");
        match parsed.command {
            Command::StagedZeroFailureCertificate {
                from,
                to,
                maximum_states,
                maximum_seconds,
                hard_mode,
                output,
                ..
            } => {
                assert_eq!(from, "2026-08-01");
                assert_eq!(to, "2026-08-02");
                assert_eq!(maximum_states, 1);
                assert_eq!(maximum_seconds, 1);
                assert!(hard_mode);
                assert_eq!(output, PathBuf::from("report.json"));
            }
            _ => panic!("expected staged-zero-failure-certificate"),
        }
    }

    #[test]
    fn benchmark_evidence_rejects_incomplete_or_mixed_date_selection() {
        let missing_to = Cli::try_parse_from([
            "maybe-wordle",
            "benchmark-evidence",
            "--from",
            "2026-01-01",
            "--output",
            "evidence.json",
            "--markdown-output",
            "evidence.md",
        ])
        .expect_err("range selection must require --to");
        assert!(missing_to.to_string().contains("--to"));

        let mixed = Cli::try_parse_from([
            "maybe-wordle",
            "benchmark-evidence",
            "--rolling-folds",
            "--from",
            "2026-01-01",
            "--to",
            "2026-01-31",
            "--output",
            "evidence.json",
            "--markdown-output",
            "evidence.md",
        ])
        .expect_err("rolling selection must not mix explicit dates");
        assert!(mixed.to_string().contains("cannot be used with"));
    }
}
