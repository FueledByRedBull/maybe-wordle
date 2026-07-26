use std::path::Path;

use chrono::NaiveDate;
use maybe_wordle::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, write_history_jsonl},
    experiments::{
        StudyFoldSelection, StudySearchStrategy, StudySpec, StudyStage, StudyState, TrialStatus,
    },
    formal::{
        DEFAULT_EXPECTED_ONLY_MODEL_ID, DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime,
        FormalVerificationMode, build_optimal_policy, verify_optimal_policy_with_mode,
    },
    model::{ModelVariant, WeightMode, build_model_artifacts},
    scoring::{format_feedback_letters, parse_feedback, score_guess},
    solver::Solver,
};

#[test]
fn parses_human_feedback() {
    let pattern = parse_feedback("bgybb").expect("valid feedback");
    assert_eq!(format_feedback_letters(pattern), "bgybb");
}

#[test]
fn repeated_letter_fixture_is_stable() {
    assert_eq!(
        format_feedback_letters(score_guess("lilly", "alley")),
        "ybgbg"
    );
}

fn write_fixture(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    std::fs::write(path, contents).expect("fixture");
}

fn write_predictive_fixture(paths: &ProjectPaths) {
    write_fixture(
        &paths.seed_guesses,
        "cigar\nrebut\nsissy\nhumph\nawake\nblush\nfocal\nevade\nnaval\nserve\nheath\ndwarf\nmodel\nkarma\nstink\ngrade\n",
    );
    write_fixture(
        &paths.seed_answers,
        "cigar\nrebut\nsissy\nhumph\nawake\nblush\nfocal\nevade\nnaval\nserve\nheath\ndwarf\n",
    );
    write_fixture(&paths.seed_reference_answers, "");
    write_fixture(&paths.seed_sources, "");
    write_fixture(&paths.manual_additions, "");
    write_history_jsonl(
        &paths.raw_history,
        &[
            NytDailyEntry {
                id: Some(1),
                solution: "cigar".into(),
                print_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(2),
                solution: "rebut".into(),
                print_date: NaiveDate::from_ymd_opt(2024, 1, 2).expect("date"),
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(3),
                solution: "sissy".into(),
                print_date: NaiveDate::from_ymd_opt(2024, 1, 3).expect("date"),
                days_since_launch: None,
                editor: None,
            },
            NytDailyEntry {
                id: Some(4),
                solution: "humph".into(),
                print_date: NaiveDate::from_ymd_opt(2024, 1, 4).expect("date"),
                days_since_launch: None,
                editor: None,
            },
        ],
    )
    .expect("history");
}

#[test]
fn formal_policy_builds_and_verifies_certificate() {
    let root = std::env::temp_dir().join("maybe-wordle-integration-formal");
    let _ = std::fs::remove_dir_all(&root);
    let paths = ProjectPaths::new(&root);
    paths.ensure_layout().expect("layout");
    let formal_dir = root.join(format!("data/formal/{DEFAULT_FORMAL_MODEL_ID}"));
    std::fs::create_dir_all(&formal_dir).expect("formal dir");
    write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
    write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
    write_fixture(&formal_dir.join("prior.toml"), "kind = \"uniform\"\n");

    let summary = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("policy");
    assert!(summary.solved_states > 0);
    let verify = verify_optimal_policy_with_mode(
        &paths,
        DEFAULT_FORMAL_MODEL_ID,
        FormalVerificationMode::Certificate,
    )
    .expect("verify");
    assert_eq!(verify.verified_cached_states, summary.solved_states);
    let oracle = verify_optimal_policy_with_mode(
        &paths,
        DEFAULT_FORMAL_MODEL_ID,
        FormalVerificationMode::Oracle,
    )
    .expect("oracle verify");
    assert!(oracle.verified_small_states > 0 || oracle.verified_medium_states > 0);
    let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("load");
    assert!(runtime.initial_state().count() > 0);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn expected_only_model_builds_separately() {
    let root = std::env::temp_dir().join("maybe-wordle-integration-expected");
    let _ = std::fs::remove_dir_all(&root);
    let paths = ProjectPaths::new(&root);
    paths.ensure_layout().expect("layout");
    let formal_dir = root.join(format!("data/formal/{DEFAULT_EXPECTED_ONLY_MODEL_ID}"));
    std::fs::create_dir_all(&formal_dir).expect("formal dir");
    write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
    write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
    write_fixture(&formal_dir.join("prior.toml"), "kind = \"uniform\"\n");

    let summary = build_optimal_policy(&paths, DEFAULT_EXPECTED_ONLY_MODEL_ID).expect("policy");
    assert!(summary.solved_states > 0);
    let verify = verify_optimal_policy_with_mode(
        &paths,
        DEFAULT_EXPECTED_ONLY_MODEL_ID,
        FormalVerificationMode::Certificate,
    )
    .expect("verify");
    assert_eq!(verify.verified_cached_states, summary.solved_states);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn predictive_experiments_and_tuning_work_on_toy_fixture() {
    let root = std::env::temp_dir().join("maybe-wordle-integration-predictive");
    let _ = std::fs::remove_dir_all(&root);
    let paths = ProjectPaths::new(&root);
    paths.ensure_layout().expect("layout");
    write_predictive_fixture(&paths);

    let config = PriorConfig::default();
    write_fixture(
        &paths.config_prior,
        &toml::to_string_pretty(&config).expect("config"),
    );
    build_model_artifacts(
        &paths,
        &config,
        NaiveDate::from_ymd_opt(2024, 1, 5).expect("date"),
    )
    .expect("model");

    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let state = solver
        .apply_history(NaiveDate::from_ymd_opt(2024, 1, 5).expect("date"), &[])
        .expect("state");
    let suggestions = solver.suggestions(&state, 5).expect("suggestions");
    assert!(!suggestions.is_empty());

    for mode in [
        WeightMode::Uniform,
        WeightMode::CooldownOnly,
        WeightMode::Weighted,
    ] {
        for variant in [ModelVariant::SeedOnly, ModelVariant::SeedPlusHistory] {
            let solver =
                Solver::from_paths_with_settings(&paths, &config, mode, variant).expect("solver");
            let report = solver
                .experiment_report(
                    NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
                    NaiveDate::from_ymd_opt(2024, 1, 4).expect("date"),
                    5,
                )
                .expect("report");
            assert!(report.latency_p95_ms >= 0.0);
            assert_eq!(report.backtest.canonical.scheduled_games, 4);
            assert_eq!(
                report.backtest.canonical.solved_games
                    + report.backtest.canonical.unsolved_games
                    + report.backtest.canonical.coverage_gaps,
                report.backtest.canonical.scheduled_games
            );
            assert!(
                report.backtest.canonical.all_game_penalized_mean_guesses
                    >= report.backtest.canonical.conditional_mean_guesses
            );
        }
    }

    let summary = Solver::tune_prior(&paths, &config).expect("tune");
    assert_eq!(summary.evaluation_plan.folds.len(), 1);
    assert_eq!(
        summary
            .evaluation_plan
            .folds
            .last()
            .expect("fold")
            .validation
            .end,
        summary.evaluation_plan.development.end
    );
    assert!(summary.evaluation_plan.development.end < summary.evaluation_plan.sealed_test.start);
    let sealed_live_config_error = Solver::evaluate_live_config(
        &paths,
        &config,
        NaiveDate::from_ymd_opt(2024, 1, 4).expect("date"),
        NaiveDate::from_ymd_opt(2024, 1, 4).expect("date"),
        5,
    )
    .expect_err("live-config evaluation must not reach the sealed test");
    assert!(format!("{sealed_live_config_error:#}").contains("reaches the sealed test"));
    assert!(
        summary
            .replacement_toml
            .contains("exact_exhaustive_threshold")
    );
    let tuned: PriorConfig = toml::from_str(&summary.replacement_toml).expect("parse tuned config");
    assert!(tuned.lookahead_threshold >= tuned.exact_threshold);
    let hard_cases = solver.hard_case_report(5).expect("hard cases");
    assert!(!hard_cases.cases.is_empty());
    assert!(summary.current.hard_case_failures <= hard_cases.cases.len());

    let study_path = root.join("study-state.json");
    let cancellation_path = root.join("pause-study");
    write_fixture(&cancellation_path, "pause\n");
    let study_spec = StudySpec {
        name: "toy-calibration".to_string(),
        stage: StudyStage::Calibration,
        seed: 17,
        trial_count: 7,
        parallelism: 2,
        strategy: StudySearchStrategy::Random,
        maximum_validation_folds: 1,
        initial_validation_folds: 1,
        reduction_factor: 2,
        fold_selection: StudyFoldSelection::NestedTimeSpread,
        maximum_trial_seconds: 60,
        maximum_memory_mb: 4_096,
    };
    let paused = Solver::run_predictive_study(
        &paths,
        &config,
        study_spec.clone(),
        &study_path,
        5,
        Some(&cancellation_path),
    )
    .expect("paused study");
    assert_eq!(paused.completed_trials, 0);
    let paused_state = StudyState::load(&study_path).expect("paused state");
    assert!(
        paused_state
            .trials
            .iter()
            .all(|trial| trial.status == TrialStatus::Pending)
    );
    std::fs::remove_file(&cancellation_path).expect("remove cancellation file");
    let resumed = Solver::run_predictive_study(
        &paths,
        &config,
        study_spec,
        &study_path,
        5,
        Some(&cancellation_path),
    )
    .expect("resumed study");
    assert_eq!(resumed.completed_trials, 7);
    let resumed_state = StudyState::load(&study_path).expect("resumed state");
    assert!(resumed_state.trials.iter().all(|trial| {
        trial.status == TrialStatus::Complete
            && trial
                .measurement
                .as_ref()
                .is_some_and(|measurement| measurement.validation_fold_indices == vec![0])
            && trial.pareto_rank.is_some()
    }));

    let proxy_study_path = root.join("proxy-study-state.json");
    let proxy_summary = Solver::run_predictive_study(
        &paths,
        &config,
        StudySpec {
            name: "toy-proxy-ranker".to_string(),
            stage: StudyStage::ProxyRanker,
            seed: 23,
            trial_count: 2,
            parallelism: 2,
            strategy: StudySearchStrategy::LowDiscrepancy,
            maximum_validation_folds: 1,
            initial_validation_folds: 1,
            reduction_factor: 2,
            fold_selection: StudyFoldSelection::NestedTimeSpread,
            maximum_trial_seconds: 60,
            maximum_memory_mb: 4_096,
        },
        &proxy_study_path,
        5,
        None,
    )
    .expect("proxy-ranker study");
    assert_eq!(proxy_summary.completed_trials, 2);
    assert!(
        proxy_summary
            .best_measurement
            .as_ref()
            .is_some_and(|measurement| measurement.solve_metrics_recorded)
    );
    let proxy_state = StudyState::load(&proxy_study_path).expect("proxy state");
    assert!(proxy_state.trials.iter().all(|trial| {
        trial.candidate.parameters.keys().all(|name| {
            name.starts_with("proxy_weights.")
                || name == "proxy_small_state_lower_bound_threshold"
                || name == "ambiguous_mass_threshold"
        })
    }));
    let book_study = Solver::run_predictive_study(
        &paths,
        &config,
        StudySpec {
            name: "toy-book-policy".to_string(),
            stage: StudyStage::BookPolicy,
            seed: 29,
            trial_count: 6,
            parallelism: 1,
            strategy: StudySearchStrategy::Grid,
            maximum_validation_folds: 1,
            initial_validation_folds: 1,
            reduction_factor: 2,
            fold_selection: StudyFoldSelection::NestedTimeSpread,
            maximum_trial_seconds: 60,
            maximum_memory_mb: 4_096,
        },
        &root.join("book-study-state.json"),
        5,
        None,
    )
    .expect("cutoff-safe book study");
    assert_eq!(book_study.completed_trials, 6);
    assert_eq!(book_study.failed_trials, 0);
    let oversized_fold_study = Solver::run_predictive_study(
        &paths,
        &config,
        StudySpec {
            name: "oversized-fold-budget".to_string(),
            stage: StudyStage::Calibration,
            seed: 30,
            trial_count: 2,
            parallelism: 1,
            strategy: StudySearchStrategy::Grid,
            maximum_validation_folds: 2,
            initial_validation_folds: 1,
            reduction_factor: 2,
            fold_selection: StudyFoldSelection::NestedTimeSpread,
            maximum_trial_seconds: 60,
            maximum_memory_mb: 4_096,
        },
        &root.join("oversized-study-state.json"),
        5,
        None,
    )
    .expect_err("fold budget must fit the plan");
    assert!(format!("{oversized_fold_study:#}").contains("plan contains only 1"));
    let memory_limited_path = root.join("memory-limited-study-state.json");
    let memory_limited = Solver::run_predictive_study(
        &paths,
        &config,
        StudySpec {
            name: "memory-limited".to_string(),
            stage: StudyStage::Calibration,
            seed: 30,
            trial_count: 7,
            parallelism: 1,
            strategy: StudySearchStrategy::Grid,
            maximum_validation_folds: 1,
            initial_validation_folds: 1,
            reduction_factor: 2,
            fold_selection: StudyFoldSelection::NestedTimeSpread,
            maximum_trial_seconds: 60,
            maximum_memory_mb: 1,
        },
        &memory_limited_path,
        5,
        None,
    )
    .expect("memory-limited study records a failed trial");
    assert_eq!(memory_limited.failed_trials, 7);
    assert!(
        StudyState::load(&memory_limited_path)
            .expect("memory-limited state")
            .trials
            .iter()
            .any(|trial| trial
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("peak working set")))
    );

    let model_based_path = root.join("model-based-study-state.json");
    let model_based_spec = StudySpec {
        name: "toy-model-based".to_string(),
        stage: StudyStage::Calibration,
        seed: 31,
        trial_count: 7,
        parallelism: 4,
        strategy: StudySearchStrategy::ModelBased,
        maximum_validation_folds: 1,
        initial_validation_folds: 1,
        reduction_factor: 2,
        fold_selection: StudyFoldSelection::NestedTimeSpread,
        maximum_trial_seconds: 60,
        maximum_memory_mb: 4_096,
    };
    let model_based = Solver::run_predictive_study(
        &paths,
        &config,
        model_based_spec.clone(),
        &model_based_path,
        5,
        None,
    )
    .expect("model-based study");
    assert_eq!(model_based.requested_parallelism, 4);
    assert_eq!(model_based.effective_parallelism, 1);
    assert_eq!(model_based.completed_trials, 7);
    let first_model_state = StudyState::load(&model_based_path).expect("model state");
    let resumed_model_based = Solver::run_predictive_study(
        &paths,
        &config,
        model_based_spec,
        &model_based_path,
        5,
        None,
    )
    .expect("resume model-based study");
    assert_eq!(resumed_model_based.completed_trials, 7);
    assert_eq!(
        StudyState::load(&model_based_path)
            .expect("resumed model state")
            .trials,
        first_model_state.trials
    );

    let ablations = Solver::predictive_ablation_report(
        &paths,
        &config,
        NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
        NaiveDate::from_ymd_opt(2024, 1, 4).expect("date"),
        5,
    )
    .expect("ablations");
    assert!(ablations.len() >= 6);
    assert!(ablations.iter().any(|row| row.label == "weighted_baseline"));
    let three_guess_gap = Solver::three_guess_gap_report(
        &paths,
        &config,
        NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
        NaiveDate::from_ymd_opt(2024, 1, 4).expect("date"),
        5,
    )
    .expect("three guess gap");
    assert_eq!(three_guess_gap.games, 4);
    assert!(three_guess_gap.base_four_guess_cases >= three_guess_gap.converted_by_aggressive);
    assert!(three_guess_gap.base_four_guess_cases >= three_guess_gap.converted_by_targeted_search);
    assert_eq!(
        three_guess_gap.base_four_guess_cases,
        three_guess_gap.cases.len()
    );

    let as_of = NaiveDate::from_ymd_opt(2024, 1, 5).expect("date");
    let opener = solver
        .build_predictive_opener_cache(as_of)
        .expect("build opener");
    let root_suggestions = solver
        .suggestions_for_history(as_of, &[], 1)
        .expect("root suggestions");
    assert_eq!(root_suggestions[0].word, opener.opener);

    let replies = solver
        .build_predictive_reply_book(as_of)
        .expect("build replies");
    let target = solver.answers[0].word.clone();
    let first_feedback = score_guess(&opener.opener, &target);
    if replies.reply_count > 0 && first_feedback != parse_feedback("22222").expect("green") {
        let second_move = solver
            .suggestions_for_history(as_of, &[(opener.opener.clone(), first_feedback)], 1)
            .expect("second move");
        assert!(!second_move.is_empty());
    }
    let _ = std::fs::remove_dir_all(&root);
}
