use std::path::Path;

use chrono::NaiveDate;
use maybe_wordle::{
    config::PriorConfig,
    data::{NytDailyEntry, ProjectPaths, read_history_jsonl, write_history_jsonl},
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

fn write_baseline_fixture(paths: &ProjectPaths) {
    write_fixture(
        &paths.seed_guesses,
        "aaaaa\nbbbbb\nccccc\nddddd\neeeee\nfffff\nggggg\n",
    );
    write_fixture(&paths.seed_answers, "aaaaa\nbbbbb\nccccc\n");
    write_fixture(&paths.seed_reference_answers, "");
    write_fixture(&paths.seed_sources, "");
    write_fixture(&paths.manual_additions, "");
    write_history_jsonl(&paths.raw_history, &[]).expect("history");
}

fn write_dynamic_public_fixture(paths: &ProjectPaths) {
    write_fixture(
        &paths.seed_guesses,
        "abaaa\nacaaa\nadaaa\naeaaa\nafaaa\nagaaa\nazaaa\nzzzzz\n",
    );
    write_fixture(&paths.seed_answers, "abaaa\nacaaa\nadaaa\naeaaa\nafaaa\n");
    write_fixture(&paths.seed_reference_answers, "");
    write_fixture(&paths.seed_sources, "");
    write_fixture(&paths.manual_additions, "");
    write_history_jsonl(
        &paths.raw_history,
        &[NytDailyEntry {
            id: Some(1),
            solution: "abaaa".into(),
            print_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("date"),
            days_since_launch: None,
            editor: None,
        }],
    )
    .expect("history");
}

fn recursively_replay_finite_baseline(
    solver: &Solver,
    puzzle_date: NaiveDate,
    observations: &[(String, u8)],
    hard_mode: bool,
) -> (f64, f64) {
    let response = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date,
            observations,
            top: 1,
            hard_mode,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        })
        .expect("baseline replay response");
    assert_eq!(
        response
            .finite_search
            .as_ref()
            .expect("baseline search metadata")
            .reason,
        maybe_wordle::solver::FiniteSearchReason::Complete
    );
    let suggestion = response.suggestions.first().expect("baseline action");
    let value = suggestion.finite_value.expect("baseline value");
    if observations.len() >= 5 {
        return (value.failure_probability, value.expected_attempts);
    }

    let guess = suggestion.word.clone();
    let mut failure_probability = 0.0;
    let mut expected_attempts = 1.0;
    for candidate in &response.candidates {
        let pattern = maybe_wordle::scoring::score_guess(&guess, &candidate.word);
        if pattern == maybe_wordle::scoring::ALL_GREEN_PATTERN {
            continue;
        }
        let probability = candidate.probability;
        let mut child_observations = observations.to_vec();
        child_observations.push((guess.clone(), pattern));
        let (child_failure, child_attempts) =
            recursively_replay_finite_baseline(solver, puzzle_date, &child_observations, hard_mode);
        failure_probability += probability * child_failure;
        expected_attempts += probability * child_attempts;
    }
    (failure_probability, expected_attempts)
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

fn assert_same_predictive_response(
    before: &maybe_wordle::predictive::PredictiveSuggestResponse,
    after: &maybe_wordle::predictive::PredictiveSuggestResponse,
) {
    assert_eq!(before.puzzle_date, after.puzzle_date);
    assert_eq!(before.history_cutoff, after.history_cutoff);
    assert_eq!(before.history_snapshot_date, after.history_snapshot_date);
    assert_eq!(before.history_snapshot_hash, after.history_snapshot_hash);
    assert_eq!(before.model_manifest_hash, after.model_manifest_hash);
    assert_eq!(before.promotion_source, after.promotion_source);
    assert_eq!(before.promoted_word, after.promoted_word);
    assert_eq!(before.artifact_state, after.artifact_state);
    assert_eq!(before.candidates.len(), after.candidates.len());
    for (left, right) in before.candidates.iter().zip(&after.candidates) {
        assert_eq!(left.word, right.word);
        assert!((left.probability - right.probability).abs() < 1e-12);
    }
    assert_eq!(
        before
            .suggestions
            .iter()
            .map(|suggestion| &suggestion.word)
            .collect::<Vec<_>>(),
        after
            .suggestions
            .iter()
            .map(|suggestion| &suggestion.word)
            .collect::<Vec<_>>()
    );
}

fn disk_only_request<'a>(
    puzzle_date: NaiveDate,
    observations: &'a [(String, u8)],
) -> PredictiveSuggestRequest<'a> {
    PredictiveSuggestRequest {
        puzzle_date,
        observations,
        top: 1,
        hard_mode: false,
        force_in_two_only: false,
        mode: PredictiveSuggestionMode::FastDiskOnly,
    }
}

#[test]
fn predictive_public_books_ignore_same_day_future_history_and_storage_order() {
    let paths = fixture_paths("public-book-date-boundary");
    write_standard_fixture(&paths);
    let config = PriorConfig {
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::ProxyOnly,
        ..PriorConfig::default()
    };
    let as_of = NaiveDate::from_ymd_opt(2024, 1, 4).expect("history cutoff");
    let puzzle_date = as_of.succ_opt().expect("puzzle date");
    let baseline = Solver::from_paths(&paths, &config).expect("baseline solver");

    let baseline_opener = baseline
        .build_predictive_opener_cache(as_of)
        .expect("baseline opener");
    let baseline_reply = baseline
        .build_predictive_reply_book(as_of)
        .expect("baseline reply book");
    let baseline_root = baseline
        .suggest_predictive(disk_only_request(puzzle_date, &[]))
        .expect("baseline root book lookup");
    assert_eq!(
        baseline_root.promotion_source,
        Some(PredictivePromotionSource::ExactDateOpenerArtifact)
    );
    assert_eq!(baseline_root.history_cutoff, as_of);

    assert!(
        baseline_reply.reply_count > 0,
        "baseline reply book is empty"
    );
    let baseline_reply_json = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(&baseline_reply.path).expect("baseline reply artifact"),
    )
    .expect("baseline reply artifact JSON");
    let forced_pattern = baseline_reply_json
        .get("replies")
        .and_then(serde_json::Value::as_array)
        .and_then(|replies| replies.first())
        .and_then(|reply| reply.get("feedback_pattern"))
        .and_then(serde_json::Value::as_u64)
        .expect("reply feedback pattern") as u8;
    let forced_observations = vec![(baseline_opener.opener.clone(), forced_pattern)];
    let baseline_forced = baseline
        .suggest_predictive(disk_only_request(puzzle_date, &forced_observations))
        .expect("baseline forced-prefix book lookup");
    assert_eq!(
        baseline_forced.promotion_source,
        Some(PredictivePromotionSource::ReplyBook)
    );
    let baseline_root_history = baseline
        .suggestions_for_history_disk_books_only(as_of, &[], 1)
        .expect("baseline history root book lookup");
    let baseline_forced_history = baseline
        .suggestions_for_history_disk_books_only_with_filters(
            as_of,
            &forced_observations,
            1,
            false,
            false,
        )
        .expect("baseline history forced-prefix book lookup");

    let mut history = read_history_jsonl(&paths.raw_history).expect("history");
    history.extend([
        NytDailyEntry {
            id: Some(5),
            solution: "model".into(),
            print_date: puzzle_date,
            days_since_launch: None,
            editor: None,
        },
        NytDailyEntry {
            id: Some(6),
            solution: "grade".into(),
            print_date: puzzle_date.succ_opt().expect("tomorrow"),
            days_since_launch: None,
            editor: None,
        },
        NytDailyEntry {
            id: Some(7),
            solution: "zzzzz".into(),
            print_date: puzzle_date
                .succ_opt()
                .and_then(|date| date.succ_opt())
                .expect("day after tomorrow"),
            days_since_launch: None,
            editor: None,
        },
    ]);
    let reversed_storage = history
        .iter()
        .rev()
        .map(|entry| serde_json::to_string(entry).expect("history JSON"))
        .collect::<Vec<_>>()
        .join("\n");
    write_fixture(&paths.raw_history, &(reversed_storage + "\n"));
    let updated = Solver::from_paths(&paths, &config).expect("updated solver");

    let updated_opener = updated
        .build_predictive_opener_cache(as_of)
        .expect("updated opener");
    let updated_reply = updated
        .build_predictive_reply_book(as_of)
        .expect("updated reply book");
    assert_eq!(baseline_opener.path, updated_opener.path);
    assert_eq!(
        baseline_opener.config_fingerprint,
        updated_opener.config_fingerprint
    );
    assert_eq!(baseline_opener.opener, updated_opener.opener);
    assert_eq!(baseline_reply.path, updated_reply.path);
    assert_eq!(
        baseline_reply.config_fingerprint,
        updated_reply.config_fingerprint
    );
    assert_eq!(baseline_reply.opener, updated_reply.opener);
    assert_eq!(baseline_reply.reply_count, updated_reply.reply_count);
    assert_eq!(
        baseline_reply.third_reply_count,
        updated_reply.third_reply_count
    );

    let updated_root = updated
        .suggest_predictive(disk_only_request(puzzle_date, &[]))
        .expect("updated root book lookup");
    assert_same_predictive_response(&baseline_root, &updated_root);
    let updated_forced = updated
        .suggest_predictive(disk_only_request(puzzle_date, &forced_observations))
        .expect("updated forced-prefix book lookup");
    assert_same_predictive_response(&baseline_forced, &updated_forced);
    assert_eq!(
        baseline_root_history
            .iter()
            .map(|suggestion| &suggestion.word)
            .collect::<Vec<_>>(),
        updated
            .suggestions_for_history_disk_books_only(as_of, &[], 1)
            .expect("updated history root book lookup")
            .iter()
            .map(|suggestion| &suggestion.word)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        baseline_forced_history
            .iter()
            .map(|suggestion| &suggestion.word)
            .collect::<Vec<_>>(),
        updated
            .suggestions_for_history_disk_books_only_with_filters(
                as_of,
                &forced_observations,
                1,
                false,
                false,
            )
            .expect("updated history forced-prefix book lookup")
            .iter()
            .map(|suggestion| &suggestion.word)
            .collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn predictive_api_uses_exact_date_opener_artifact_in_fast_mode() {
    let paths = fixture_paths("exact-artifact");
    write_standard_fixture(&paths);
    let config = PriorConfig {
        session_window_days: 1,
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::ProxyOnly,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let as_of = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");

    let opener = solver
        .build_predictive_opener_cache(as_of)
        .expect("build opener");
    let response = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date: as_of.succ_opt().expect("puzzle date"),
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
fn puzzle_replay_ignores_same_day_and_future_history_including_new_primary_words() {
    let paths = fixture_paths("puzzle-date-boundary");
    write_standard_fixture(&paths);
    let config = PriorConfig::default();
    let puzzle_date = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");
    let baseline = Solver::from_paths(&paths, &config).expect("baseline");
    let mut history = read_history_jsonl(&paths.raw_history).expect("history");
    history[3].solution = "model".to_string();
    history.push(NytDailyEntry {
        id: Some(5),
        solution: "grade".to_string(),
        print_date: puzzle_date.succ_opt().expect("tomorrow"),
        days_since_launch: None,
        editor: None,
    });
    history.push(NytDailyEntry {
        id: Some(6),
        solution: "zzzzz".to_string(),
        print_date: puzzle_date
            .succ_opt()
            .and_then(|date| date.succ_opt())
            .expect("day after tomorrow"),
        days_since_launch: None,
        editor: None,
    });
    write_history_jsonl(&paths.raw_history, &history).expect("future history");
    let updated = Solver::from_paths(&paths, &config).expect("updated");
    let cutoff = puzzle_date.pred_opt().expect("yesterday");
    let before_fixed = baseline
        .fixed_posterior_state(cutoff)
        .expect("fixed baseline");
    let after_fixed = updated
        .fixed_posterior_state(cutoff)
        .expect("fixed updated");
    assert_eq!(before_fixed.surviving.len(), after_fixed.surviving.len());
    let before_tail = baseline.initial_state(cutoff);
    let after_tail = updated.initial_state(cutoff);
    assert_eq!(
        before_tail.fallback_surviving.len(),
        after_tail.fallback_surviving.len()
    );
    let replay = |solver: &Solver, observations: &[(String, u8)]| {
        solver
            .suggest_predictive(PredictiveSuggestRequest {
                puzzle_date,
                observations,
                top: 5,
                hard_mode: false,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::LiveOnly,
            })
            .expect("replay")
    };
    let feedback = maybe_wordle::scoring::score_guess("cigar", "humph");
    for observations in [Vec::new(), vec![("cigar".to_string(), feedback)]] {
        let before = replay(&baseline, &observations);
        let after = replay(&updated, &observations);
        assert_eq!(before.puzzle_date, puzzle_date);
        assert_eq!(
            before.history_cutoff,
            puzzle_date.pred_opt().expect("yesterday")
        );
        assert_eq!(before.history_snapshot_hash, after.history_snapshot_hash);
        assert_eq!(before.model_manifest_hash, after.model_manifest_hash);
        assert_eq!(before.state.surviving, after.state.surviving);
        assert_eq!(before.candidates.len(), after.candidates.len());
        for (left, right) in before.candidates.iter().zip(&after.candidates) {
            assert_eq!(left.word, right.word);
            assert!((left.probability - right.probability).abs() < 1e-12);
        }
        assert_eq!(
            before
                .suggestions
                .iter()
                .map(|row| &row.word)
                .collect::<Vec<_>>(),
            after
                .suggestions
                .iter()
                .map(|row| &row.word)
                .collect::<Vec<_>>()
        );
        let finite_replay = |solver: &Solver| {
            let mut options = maybe_wordle::solver::FiniteSearchOptions::fast();
            // A deterministic work budget makes this an information-boundary
            // test rather than a comparison of scheduler-dependent deadlines.
            options.node_limit = Some(10_000);
            options.budget = std::time::Duration::from_secs(10);
            solver
                .suggest_predictive_controlled(
                    PredictiveSuggestRequest {
                        puzzle_date,
                        observations: &observations,
                        top: 5,
                        hard_mode: false,
                        force_in_two_only: false,
                        mode: PredictiveSuggestionMode::LiveOnly,
                    },
                    options,
                    &|| false,
                )
                .expect("finite replay")
        };
        let before = finite_replay(&baseline);
        let after = finite_replay(&updated);
        assert_eq!(before.model_manifest_hash, after.model_manifest_hash);
        assert_eq!(
            before.finite_search.as_ref().unwrap().reason,
            after.finite_search.as_ref().unwrap().reason
        );
        assert_eq!(before.suggestions.len(), after.suggestions.len());
        for (left, right) in before.suggestions.iter().zip(&after.suggestions) {
            assert_eq!(left.word, right.word);
            let left = left.finite_value.as_ref().unwrap();
            let right = right.finite_value.as_ref().unwrap();
            assert_eq!(left.quality, right.quality);
            assert_eq!(left.failure_probability, right.failure_probability);
            assert_eq!(left.expected_attempts, right.expected_attempts);
        }
    }
    let live = replay(&baseline, &[]);
    let full = baseline
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date,
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::Full,
        })
        .expect("full policy");
    let evaluated = baseline
        .solve_target_detailed("humph", puzzle_date, 5)
        .expect("evaluate");
    assert_eq!(full.suggestions[0].word, evaluated.steps[0].guess);
    history[2].solution = "awake".to_string();
    write_history_jsonl(&paths.raw_history, &history).expect("changed yesterday");
    let changed = replay(&Solver::from_paths(&paths, &config).expect("changed"), &[]);
    assert_ne!(live.history_snapshot_hash, changed.history_snapshot_hash);
    assert_ne!(live.model_manifest_hash, changed.model_manifest_hash);
    assert_ne!(
        live.candidates
            .iter()
            .find(|row| row.word == "awake")
            .unwrap()
            .probability,
        changed
            .candidates
            .iter()
            .find(|row| row.word == "awake")
            .unwrap()
            .probability
    );
    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn finite_live_and_evaluation_share_policy_and_stop_after_six_turns() {
    let paths = fixture_paths("finite-live-evaluation");
    write_standard_fixture(&paths);
    let config = PriorConfig {
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::FiniteFast,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let puzzle_date = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");
    let request = PredictiveSuggestRequest {
        puzzle_date,
        observations: &[],
        top: 5,
        hard_mode: false,
        force_in_two_only: false,
        mode: PredictiveSuggestionMode::LiveOnly,
    };
    let response = solver.suggest_predictive(request).expect("finite live");
    assert!(response.finite_search.is_some());
    assert!(response.promotion_source.is_none());
    assert!(response.suggestions[0].finite_value.is_some());
    assert_eq!(response.candidates.len(), 16);
    let evaluated = solver
        .solve_target_detailed("humph", puzzle_date, 5)
        .expect("finite evaluation");
    assert_eq!(response.suggestions[0].word, evaluated.steps[0].guess);
    assert!(
        evaluated
            .steps
            .iter()
            .all(|step| step.regime_used == maybe_wordle::predictive::PredictiveRegime::Finite)
    );
    assert!(
        evaluated
            .steps
            .iter()
            .all(|step| step.lookahead_pool_base == 0 && step.exact_pool_base == 0)
    );
    let feedback = maybe_wordle::scoring::score_guess("cigar", "humph");
    let observations = vec![("cigar".to_string(), feedback); 6];
    let finished = solver
        .suggest_predictive(PredictiveSuggestRequest {
            observations: &observations,
            ..request
        })
        .expect("completed six turns");
    assert!(finished.suggestions.is_empty());
    let mut too_many = observations;
    too_many.push(("cigar".to_string(), feedback));
    assert!(
        solver
            .suggest_predictive(PredictiveSuggestRequest {
                observations: &too_many,
                ..request
            })
            .is_err()
    );
    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn finite_fast_dynamic_public_history_preserves_hard_mode_fallback_and_six_turn_boundary() {
    let paths = fixture_paths("finite-fast-dynamic-public");
    write_dynamic_public_fixture(&paths);
    let config = PriorConfig {
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::FiniteFastDynamic,
        ..PriorConfig::default()
    };
    let as_of = NaiveDate::from_ymd_opt(2024, 1, 1).expect("history cutoff");
    let puzzle_date = as_of.succ_opt().expect("puzzle date");
    let solver =
        Solver::from_paths_with_mode(&paths, &config, maybe_wordle::model::WeightMode::Uniform)
            .expect("dynamic solver");
    let answer_words = |indices: &[usize]| {
        indices
            .iter()
            .map(|index| solver.answers[*index].word.as_str())
            .collect::<Vec<_>>()
    };

    let initial = solver.initial_state(as_of);
    assert!(!initial.condition_only);
    assert!(!initial.fallback_active);
    assert_eq!(
        answer_words(&initial.surviving),
        vec!["abaaa", "acaaa", "adaaa", "aeaaa", "afaaa"]
    );
    assert_eq!(
        answer_words(&initial.fallback_surviving),
        vec!["agaaa", "azaaa", "zzzzz"]
    );

    let shared_hint = 236u8;
    let dormant_history = vec![("azaaa".to_string(), shared_hint)];
    let dormant = solver
        .apply_history(as_of, &dormant_history)
        .expect("dormant fallback state");
    assert_eq!(
        answer_words(&dormant.surviving),
        vec!["abaaa", "acaaa", "adaaa", "aeaaa", "afaaa"]
    );
    assert_eq!(answer_words(&dormant.fallback_surviving), vec!["agaaa"]);
    assert!(!dormant.fallback_active);

    let activated_history = vec![
        ("azaaa".to_string(), shared_hint),
        ("abaaa".to_string(), shared_hint),
    ];
    let activated = solver
        .apply_history(as_of, &activated_history)
        .expect("activated fallback state");
    assert_eq!(
        answer_words(&activated.surviving),
        vec!["acaaa", "adaaa", "aeaaa", "afaaa", "agaaa"]
    );
    assert!(activated.fallback_active);
    assert!(activated.fallback_surviving.is_empty());

    let response = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date,
            observations: &dormant_history,
            top: usize::MAX,
            hard_mode: true,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        })
        .expect("dynamic finite response");
    assert_eq!(response.model_version, "predictive-finite-dynamic-v1");
    assert_eq!(response.state.surviving, 5);
    let finite = response.finite_search.as_ref().expect("finite metadata");
    assert_eq!(
        finite.reason,
        maybe_wordle::solver::FiniteSearchReason::Complete
    );
    assert_eq!(finite.candidates.len(), 7);
    assert!(finite.candidates.iter().all(|candidate| {
        let word = solver.guesses[candidate.guess_index].as_bytes();
        word[0] == b'a' && word[2..].iter().all(|letter| *letter == b'a')
    }));
    let a_candidate = finite
        .candidates
        .iter()
        .find(|candidate| solver.guesses[candidate.guess_index] == "abaaa")
        .expect("abaaa finite candidate");
    assert!(
        a_candidate.failure_probability > 0.0,
        "fallback activation must contribute failure mass after the abaaa non-green branch"
    );

    let six_turn_history = vec![
        ("azaaa".to_string(), shared_hint),
        ("abaaa".to_string(), shared_hint),
        ("acaaa".to_string(), shared_hint),
        ("adaaa".to_string(), shared_hint),
        ("aeaaa".to_string(), shared_hint),
        ("agaaa".to_string(), shared_hint),
    ];
    let finished = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date,
            observations: &six_turn_history,
            top: 1,
            hard_mode: true,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        })
        .expect("completed six turns");
    assert_eq!(finished.state.surviving, 1);
    assert_eq!(finished.candidates[0].word, "afaaa");
    assert!(finished.suggestions.is_empty());
    assert_eq!(
        finished
            .finite_search
            .as_ref()
            .expect("six-turn finite metadata")
            .reason,
        maybe_wordle::solver::FiniteSearchReason::Complete
    );
    assert!(
        finished
            .finite_search
            .as_ref()
            .expect("six-turn finite metadata")
            .candidates
            .is_empty()
    );

    let mut too_many = six_turn_history;
    too_many.push(("afaaa".to_string(), shared_hint));
    let error = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date,
            observations: &too_many,
            top: 1,
            hard_mode: true,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        })
        .expect_err("seven turns must be rejected");
    assert!(error.to_string().contains("at most six turns"));

    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn staged_fixed_belief_matches_finite_posterior_without_changing_search() {
    use maybe_wordle::config::SearchPolicyMode;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("test-staged-fixed-belief-{}", std::process::id()));
    let paths = ProjectPaths::new(&root);
    paths.ensure_layout().expect("layout");
    write_standard_fixture(&paths);
    let base = PriorConfig::default();
    let staged = Solver::from_paths(
        &paths,
        &PriorConfig {
            search_policy_mode: SearchPolicyMode::StagedFixedBelief,
            ..base.clone()
        },
    )
    .expect("staged fixed-belief solver");
    let finite = Solver::from_paths(
        &paths,
        &PriorConfig {
            search_policy_mode: SearchPolicyMode::FiniteFastFixedWork,
            ..base.clone()
        },
    )
    .expect("finite solver");
    let ordinary = Solver::from_paths(&paths, &base).expect("ordinary staged solver");
    let cutoff = NaiveDate::from_ymd_opt(2024, 1, 3).expect("cutoff");
    let mut staged_state = staged.initial_state(cutoff);
    let mut finite_state = finite.initial_state(cutoff);
    assert!(staged_state.condition_only && finite_state.condition_only);
    assert!(!ordinary.initial_state(cutoff).condition_only);
    assert_eq!(staged_state.surviving, finite_state.surviving);
    assert_eq!(staged_state.weights, finite_state.weights);
    assert_eq!(staged_state.total_weight, finite_state.total_weight);
    assert_eq!(
        staged_state.fallback_surviving,
        finite_state.fallback_surviving
    );

    let feedback = maybe_wordle::scoring::score_guess("cigar", "humph");
    staged
        .apply_feedback(&mut staged_state, "cigar", feedback)
        .expect("staged feedback");
    finite
        .apply_feedback(&mut finite_state, "cigar", feedback)
        .expect("finite feedback");
    assert_eq!(staged_state.surviving, finite_state.surviving);
    assert_eq!(staged_state.weights, finite_state.weights);
    assert_eq!(staged_state.total_weight, finite_state.total_weight);
    assert_eq!(staged_state.fallback_active, finite_state.fallback_active);

    let request = PredictiveSuggestRequest {
        puzzle_date: NaiveDate::from_ymd_opt(2024, 1, 4).expect("puzzle date"),
        observations: &[],
        top: 5,
        hard_mode: false,
        force_in_two_only: false,
        mode: PredictiveSuggestionMode::LiveOnly,
    };
    let staged_response = staged
        .suggest_predictive(request)
        .expect("staged suggestions");
    let finite_response = finite
        .suggest_predictive(request)
        .expect("finite suggestions");
    assert!(staged_response.finite_search.is_none());
    assert!(finite_response.finite_search.is_some());
    assert_eq!(
        staged_response.candidates.len(),
        finite_response.candidates.len()
    );
    for (left, right) in staged_response
        .candidates
        .iter()
        .zip(&finite_response.candidates)
    {
        assert_eq!(left.word, right.word);
        assert!((left.probability - right.probability).abs() < 1e-12);
    }
    std::fs::remove_dir_all(root).expect("fixture cleanup");
}

#[test]
fn selected_staged_policy_keeps_dynamic_finite_search_experimental() {
    use maybe_wordle::config::SearchPolicyMode;

    let paths = fixture_paths("staged-six-turn-dispatch");
    write_standard_fixture(&paths);
    let config = PriorConfig {
        search_policy_mode: SearchPolicyMode::Staged,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("staged solver");
    let date = NaiveDate::from_ymd_opt(2024, 1, 4).expect("puzzle date");
    assert!(
        !solver
            .initial_state(date.pred_opt().unwrap())
            .condition_only
    );
    let guesses = ["rebut", "sissy", "humph", "awake", "blush"];
    let observations = guesses
        .iter()
        .map(|guess| {
            (
                (*guess).to_string(),
                maybe_wordle::scoring::score_guess(guess, "cigar"),
            )
        })
        .collect::<Vec<_>>();
    for turn in [0, 3] {
        let response = solver
            .suggest_predictive(PredictiveSuggestRequest {
                puzzle_date: date,
                observations: &observations[..turn],
                top: 3,
                hard_mode: false,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::LiveOnly,
            })
            .expect("early staged suggestion");
        assert!(response.finite_search.is_none(), "turn {turn}");
        assert!(
            response.suggestions[0].finite_value.is_none(),
            "turn {turn}"
        );
    }
    for turn in [4, 5] {
        let response = solver
            .suggest_predictive(PredictiveSuggestRequest {
                puzzle_date: date,
                observations: &observations[..turn],
                top: 3,
                hard_mode: false,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::LiveOnly,
            })
            .expect("late staged suggestion");
        assert!(response.finite_search.is_none(), "turn {turn}");
    }
    let opening = PredictiveSuggestRequest {
        puzzle_date: date,
        observations: &[],
        top: 3,
        hard_mode: false,
        force_in_two_only: false,
        mode: PredictiveSuggestionMode::LiveOnly,
    };
    let configured = solver
        .suggest_predictive(opening)
        .expect("configured search");
    let profiled = solver
        .suggest_predictive_controlled(
            opening,
            maybe_wordle::solver::FiniteSearchOptions::fast(),
            &|| false,
        )
        .expect("profiled search");
    assert_eq!(profiled.model_version, "predictive-finite-dynamic-v1");
    assert_eq!(configured.model_version, "predictive-v1");
    assert_ne!(configured.model_manifest_hash, profiled.model_manifest_hash);
    assert_eq!(configured.candidates.len(), profiled.candidates.len());
    for (left, right) in configured.candidates.iter().zip(&profiled.candidates) {
        assert_eq!(left.word, right.word);
        assert_eq!(left.probability, right.probability);
    }
    std::fs::remove_dir_all(paths.root).expect("fixture cleanup");
}

#[test]
fn finite_post_search_cancellation_keeps_one_action_and_stops_formatting() {
    use maybe_wordle::solver::{FiniteSearchOptions, FiniteSearchReason};
    use std::cell::Cell;

    let paths = fixture_paths("finite-post-search-cancellation");
    write_standard_fixture(&paths);
    let solver = Solver::from_paths(&paths, &PriorConfig::default()).expect("solver");
    let observations = vec![("cigar".to_string(), 0); 5];
    let request = PredictiveSuggestRequest {
        puzzle_date: NaiveDate::from_ymd_opt(2024, 1, 4).expect("date"),
        observations: &observations,
        top: usize::MAX,
        hard_mode: false,
        force_in_two_only: false,
        mode: PredictiveSuggestionMode::LiveOnly,
    };
    let mut options = FiniteSearchOptions::fast();
    options.budget = std::time::Duration::from_secs(5);
    let polls = Cell::new(0usize);
    let complete = solver
        .suggest_predictive_controlled(request, options, &|| {
            polls.set(polls.get() + 1);
            false
        })
        .expect("complete final-turn ranking");
    assert_eq!(
        complete.finite_search.as_ref().unwrap().reason,
        FiniteSearchReason::Complete
    );
    assert!(complete.suggestions.len() > 1);
    // Formatting polls for every row after the first, plus its final status check.
    let search_polls = polls.get() - complete.suggestions.len();
    polls.set(0);
    let interrupted = solver
        .suggest_predictive_controlled(request, options, &|| {
            polls.set(polls.get() + 1);
            polls.get() > search_polls
        })
        .expect("cancelled formatting");
    assert_eq!(
        interrupted.finite_search.as_ref().unwrap().reason,
        FiniteSearchReason::Cancelled
    );
    assert_eq!(interrupted.suggestions.len(), 1);
    assert_eq!(
        interrupted.suggestions[0].word,
        complete.suggestions[0].word
    );
    std::fs::remove_dir_all(paths.root).expect("remove toy fixture");
}

#[test]
fn finite_force_in_two_filter_precedes_requested_top_limit() {
    let paths = fixture_paths("finite-force-in-two-limit");
    write_standard_fixture(&paths);
    write_fixture(&paths.seed_guesses, "cigar\ncigam\ncigap\nrampy\nzzzzz\n");
    write_fixture(&paths.seed_answers, "cigar\ncigam\ncigap\n");
    let history = read_history_jsonl(&paths.raw_history).expect("history");
    write_history_jsonl(&paths.raw_history, &history[..1]).expect("single past answer");
    let solver = Solver::from_paths(&paths, &PriorConfig::default()).expect("solver");
    let observations = vec![("zzzzz".to_string(), 0); 5];
    let mut request = PredictiveSuggestRequest {
        puzzle_date: NaiveDate::from_ymd_opt(2024, 1, 2).expect("date"),
        observations: &observations,
        top: 1,
        hard_mode: false,
        force_in_two_only: false,
        mode: PredictiveSuggestionMode::LiveOnly,
    };
    let options = maybe_wordle::solver::FiniteSearchOptions::strong();
    let unfiltered = solver
        .suggest_predictive_controlled(request, options, &|| false)
        .expect("unfiltered finite ranking");
    assert_eq!(unfiltered.suggestions.len(), 1);
    assert!(!unfiltered.suggestions[0].force_in_two);
    request.force_in_two_only = true;
    let filtered = solver
        .suggest_predictive_controlled(request, options, &|| false)
        .expect("filtered finite ranking");
    assert_eq!(filtered.suggestions.len(), 1);
    assert_eq!(filtered.suggestions[0].word, "rampy");
    assert!(filtered.suggestions[0].force_in_two);
    std::fs::remove_dir_all(paths.root).expect("remove toy fixture");
}

#[test]
fn finite_baseline_replay_matches_grouped_normal_and_hard_policy_value() {
    let paths = fixture_paths("finite-baseline-replay");
    write_baseline_fixture(&paths);
    let config = PriorConfig {
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::FiniteBaseline,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("baseline solver");
    let puzzle_date = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");
    let history = [
        ("ddddd".to_string(), 0),
        ("eeeee".to_string(), 0),
        ("fffff".to_string(), 0),
        ("ggggg".to_string(), 0),
    ];

    for observations in [&history[..3], &history[..]] {
        for hard_mode in [false, true] {
            let response = solver
                .suggest_predictive(PredictiveSuggestRequest {
                    puzzle_date,
                    observations,
                    top: 1,
                    hard_mode,
                    force_in_two_only: false,
                    mode: PredictiveSuggestionMode::LiveOnly,
                })
                .expect("baseline response");
            let search = response.finite_search.as_ref().expect("baseline metadata");
            assert_eq!(
                search.reason,
                maybe_wordle::solver::FiniteSearchReason::Complete
            );
            assert_eq!(response.suggestions.len(), 1);
            assert_eq!(
                response.suggestions[0]
                    .finite_value
                    .expect("baseline value")
                    .quality,
                maybe_wordle::solver::FiniteSearchQuality::UpperBound
            );

            let (expected_failure, expected_attempts) =
                recursively_replay_finite_baseline(&solver, puzzle_date, observations, hard_mode);
            let value = response.suggestions[0]
                .finite_value
                .expect("root baseline value");
            assert!((value.failure_probability - expected_failure).abs() < 1e-12);
            assert!((value.expected_attempts - expected_attempts).abs() < 1e-12);
        }
    }
    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn predictive_boundaries_reject_invalid_hard_history_and_post_solve_turns() {
    let paths = fixture_paths("invalid-hard-history");
    write_standard_fixture(&paths);
    let solver = Solver::from_paths(&paths, &PriorConfig::default()).unwrap();
    let observations = vec![
        (
            "cigar".to_string(),
            maybe_wordle::scoring::score_guess("cigar", "grade"),
        ),
        (
            "humph".to_string(),
            maybe_wordle::scoring::score_guess("humph", "grade"),
        ),
    ];
    let solved = vec![("cigar".to_string(), 242), ("cigar".to_string(), 242)];
    for (history, message) in [
        (&observations, "invalid hard-mode turn 2"),
        (&solved, "solved Wordle game"),
    ] {
        let request = PredictiveSuggestRequest {
            puzzle_date: NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
            observations: history,
            top: 5,
            hard_mode: true,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        };
        assert!(
            solver
                .suggest_predictive(request)
                .unwrap_err()
                .to_string()
                .contains(message)
        );
        assert!(
            solver
                .suggest_predictive_controlled(
                    request,
                    maybe_wordle::solver::FiniteSearchOptions::fast(),
                    &|| false
                )
                .unwrap_err()
                .to_string()
                .contains(message)
        );
    }
    for history in [
        vec![("x".to_string(), 0)],
        vec![("UPPER".to_string(), 0)],
        vec![("cigar".to_string(), 243)],
    ] {
        assert!(solver.hard_mode_violation(&history, "cigar").is_some());
    }
    assert!(solver.hard_mode_violation(&[], "x").is_some());
    let _ = std::fs::remove_dir_all(paths.root);
}

#[test]
fn predictive_api_uses_recent_opener_artifact_when_exact_date_is_missing() {
    let paths = fixture_paths("recent-artifact");
    write_standard_fixture(&paths);
    let config = PriorConfig {
        session_window_days: 1,
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::ProxyOnly,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let artifact_date = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");
    let request_date = NaiveDate::from_ymd_opt(2024, 1, 5).expect("date");

    let opener = solver
        .build_predictive_opener_cache(artifact_date)
        .expect("build opener");
    let response = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date: request_date.succ_opt().expect("puzzle date"),
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
    let config = PriorConfig {
        session_window_days: 1,
        search_policy_mode: maybe_wordle::config::SearchPolicyMode::ProxyOnly,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let as_of = NaiveDate::from_ymd_opt(2024, 1, 4).expect("date");

    let fast = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date: as_of.succ_opt().expect("puzzle date"),
            observations: &[],
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::FastDiskOnly,
        })
        .expect("fast");
    let full = solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date: as_of.succ_opt().expect("puzzle date"),
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

    let mut epsilon_config = PriorConfig {
        session_window_days: 1,
        cooldown_floor: 0.0,
        ..PriorConfig::default()
    };
    epsilon_config.recovery.mode = RecoveryMode::EpsilonRepair;
    let epsilon_solver = Solver::from_paths(&paths, &epsilon_config).expect("solver");
    let epsilon = epsilon_solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date: as_of.succ_opt().expect("puzzle date"),
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

    let mut strict_config = PriorConfig {
        session_window_days: 1,
        cooldown_floor: 0.0,
        ..PriorConfig::default()
    };
    strict_config.recovery.mode = RecoveryMode::Strict;
    let strict_solver = Solver::from_paths(&paths, &strict_config).expect("solver");
    let error = strict_solver
        .suggest_predictive(PredictiveSuggestRequest {
            puzzle_date: as_of.succ_opt().expect("puzzle date"),
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
