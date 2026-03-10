use std::path::Path;

use chrono::NaiveDate;
use maybe_wordle::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, write_history_jsonl},
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
        }
    }

    let summary = Solver::tune_prior(&paths, &config).expect("tune");
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
