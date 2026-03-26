use std::path::Path;

use chrono::NaiveDate;
use maybe_wordle::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, write_history_jsonl},
    predictive::{
        PredictivePromotionSource, PredictiveSuggestRequest, PredictiveSuggestionMode, RecoveryMode,
    },
    solver::Solver,
};

fn write_fixture(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    std::fs::write(path, contents).expect("fixture");
}

fn write_standard_fixture(paths: &ProjectPaths) {
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

fn write_zero_mass_fixture(paths: &ProjectPaths) {
    write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
    write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\nhumph\n");
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

fn fixture_paths(label: &str) -> ProjectPaths {
    let root = std::env::temp_dir().join(format!(
        "maybe-wordle-predictive-{label}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let paths = ProjectPaths::new(&root);
    paths.ensure_layout().expect("layout");
    paths
}

#[test]
fn predictive_api_uses_exact_date_opener_artifact_in_fast_mode() {
    let paths = fixture_paths("exact-artifact");
    write_standard_fixture(&paths);
    let mut config = PriorConfig::default();
    config.session_window_days = 1;
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let as_of = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");

    let opener = solver
        .build_predictive_opener_cache(as_of)
        .expect("build opener");
    let response = solver
        .suggest_predictive(PredictiveSuggestRequest {
            as_of,
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::FastDiskOnly,
        })
        .expect("suggest");

    assert_eq!(
        response.promotion_source,
        Some(PredictivePromotionSource::ExactDateOpenerArtifact)
    );
    assert_eq!(response.promoted_word, Some(opener.opener));
    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn predictive_api_uses_recent_opener_artifact_when_exact_date_is_missing() {
    let paths = fixture_paths("recent-artifact");
    write_standard_fixture(&paths);
    let mut config = PriorConfig::default();
    config.session_window_days = 1;
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let artifact_date = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");
    let request_date = NaiveDate::from_ymd_opt(2024, 1, 5).expect("date");

    let opener = solver
        .build_predictive_opener_cache(artifact_date)
        .expect("build opener");
    let response = solver
        .suggest_predictive(PredictiveSuggestRequest {
            as_of: request_date,
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::FastDiskOnly,
        })
        .expect("suggest");

    assert_eq!(
        response.promotion_source,
        Some(PredictivePromotionSource::RecentOpenerArtifact)
    );
    assert_eq!(response.promoted_word, Some(opener.opener));
    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn predictive_api_distinguishes_full_and_disk_only_session_fallbacks() {
    let paths = fixture_paths("session-fallback");
    write_standard_fixture(&paths);
    let mut config = PriorConfig::default();
    config.session_window_days = 1;
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let as_of = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");

    let fast = solver
        .suggest_predictive(PredictiveSuggestRequest {
            as_of,
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::FastDiskOnly,
        })
        .expect("fast");
    let full = solver
        .suggest_predictive(PredictiveSuggestRequest {
            as_of,
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::Full,
        })
        .expect("full");

    assert_eq!(fast.promotion_source, None);
    assert_eq!(
        full.promotion_source,
        Some(PredictivePromotionSource::SessionRootFallback)
    );
    assert!(full.promoted_word.is_some());
    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn recovery_modes_are_explicit_in_predictive_api() {
    let paths = fixture_paths("recovery");
    write_zero_mass_fixture(&paths);
    let as_of = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");

    let mut epsilon_config = PriorConfig::default();
    epsilon_config.session_window_days = 1;
    epsilon_config.recovery.mode = RecoveryMode::EpsilonRepair;
    let epsilon_solver = Solver::from_paths(&paths, &epsilon_config).expect("solver");
    let epsilon = epsilon_solver
        .suggest_predictive(PredictiveSuggestRequest {
            as_of,
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        })
        .expect("epsilon");
    assert_eq!(
        epsilon.state.recovery_mode_used,
        Some(RecoveryMode::EpsilonRepair)
    );

    let mut strict_config = PriorConfig::default();
    strict_config.session_window_days = 1;
    strict_config.recovery.mode = RecoveryMode::Strict;
    let strict_solver = Solver::from_paths(&paths, &strict_config).expect("solver");
    let error = strict_solver
        .suggest_predictive(PredictiveSuggestRequest {
            as_of,
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        })
        .expect_err("strict should fail");
    assert!(
        error
            .to_string()
            .contains("no positive answer mass remains"),
        "unexpected error: {error}"
    );
    let _ = std::fs::remove_dir_all(paths.root);
}
