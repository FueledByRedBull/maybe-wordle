use std::cell::Cell;

use super::*;
use crate::test_support::TestDirectory;

thread_local! {
    pub(super) static SOLVE_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static PARTITION_CALLS: Cell<usize> = const { Cell::new(0) };
}

fn fixture() -> (TestDirectory, FormalPolicyRuntime) {
    let root = TestDirectory::new("formal-live");
    let paths = ProjectPaths::new(root.path());
    paths.ensure_layout().unwrap();
    let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
    fs::create_dir_all(&artifacts.model_dir).unwrap();
    fs::write(&paths.seed_guesses, "abzyx\nabxyz\naaaaa\nbbbbb\nccccc\n").unwrap();
    fs::write(&paths.seed_answers, "aaaaa\nbbbbb\nccccc\n").unwrap();
    fs::write(
        &artifacts.prior_spec,
        "objective = \"lexicographic\"\nkind = \"uniform\"\n",
    )
    .unwrap();
    let built = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
    assert_eq!(built.root_best_guess, "abxyz");
    verify_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
    let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
    (root, runtime)
}

#[test]
fn stored_primary_requires_no_recursive_search() {
    let (_root, runtime) = fixture();
    SOLVE_CALLS.set(0);
    PARTITION_CALLS.set(0);
    let suggestions = runtime.suggest(&runtime.initial_state(), 1).unwrap();
    assert_eq!(SOLVE_CALLS.get(), 0);
    assert_eq!(PARTITION_CALLS.get(), 1);
    assert_eq!(suggestions[0].word, "abxyz");
}

#[test]
fn equivalent_probe_dedup_keeps_lexical_representative() {
    let (_root, runtime) = fixture();
    let state = runtime.initial_state();
    let ranked = runtime.evaluate_state_ranked(&state).unwrap();
    assert_eq!(runtime.model.guesses[ranked[0].guess_index], "abxyz");
    let started = Instant::now();
    let mut builder = FormalPolicyBuilder {
        model: runtime.model.clone(),
        memo: HashMap::new(),
        hot_tt: HotTranspositionTable::new(4096),
        deduped_signatures: 0,
        bound_hits: 0,
        root_refinement_pruned: 0,
        local_refinement_pruned: 0,
        partition_calls: 0,
        quick_plan_calls: 0,
        started,
        last_progress: started,
        cancelled: None,
    };
    let plans = builder.collect_quick_plans_for_state(&state).unwrap();
    assert!(
        plans
            .iter()
            .any(|plan| builder.model.guesses[plan.guess_index] == "abxyz")
    );
    assert!(
        !plans
            .iter()
            .any(|plan| builder.model.guesses[plan.guess_index] == "abzyx")
    );
}

#[test]
fn bounded_alternatives_keep_only_the_canonical_primary_when_incomplete() {
    let (_root, runtime) = fixture();
    for limit in [0, 1] {
        SOLVE_CALLS.set(0);
        PARTITION_CALLS.set(0);
        let explanation = runtime
            .explain_state_cancellable(&runtime.initial_state(), 5, limit, &|| false)
            .unwrap();
        assert_eq!(
            explanation.alternatives_status,
            FormalAlternativesStatus::WorkLimitReached
        );
        assert_eq!(explanation.best_guess, "abxyz");
        assert_eq!(explanation.tied_moves.len(), 1);
        assert_eq!(explanation.tied_moves[0].word, explanation.best_guess);
        assert!(PARTITION_CALLS.get() <= limit + 1);
    }
}

#[test]
fn public_suggest_keeps_primary_when_alternatives_hit_default_limit() {
    let (root, mut runtime) = fixture();
    let mut guesses = runtime.model.guesses.clone();
    guesses.extend(
        (0..=DEFAULT_FORMAL_ALTERNATIVE_PARTITIONS).map(|mut value| {
            let mut suffix = [b'a'; 3];
            for slot in (0..3).rev() {
                suffix[slot] = b'a' + (value % 26) as u8;
                value /= 26;
            }
            format!(
                "zz{}{}{}",
                suffix[0] as char, suffix[1] as char, suffix[2] as char
            )
        }),
    );
    let answers = runtime
        .model
        .answers
        .iter()
        .map(|word| crate::model::AnswerRecord {
            word: word.clone(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: Vec::new(),
        })
        .collect::<Vec<_>>();
    let pattern_path = root.path().join("expanded-pattern-table.bin");
    runtime.model.pattern_table =
        PatternTable::load_or_build_at(&pattern_path, &guesses, &answers).unwrap();
    runtime.model.guesses = guesses;
    runtime.model.guess_index = runtime
        .model
        .guesses
        .iter()
        .enumerate()
        .map(|(index, guess)| (guess.clone(), index))
        .collect();

    let explanation = runtime.explain_state(&runtime.initial_state(), 2).unwrap();
    assert_eq!(
        explanation.alternatives_status,
        FormalAlternativesStatus::WorkLimitReached
    );
    let suggestions = runtime.suggest(&runtime.initial_state(), 2).unwrap();

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].word, "abxyz");
}

#[test]
fn cancellation_interrupts_uncached_recursive_work() {
    let (_root, runtime) = fixture();
    let error = runtime
        .explain_state_cancellable(&runtime.initial_state(), 1, 0, &|| true)
        .expect_err("stored primary still respects cancellation");
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::Cancelled)
    ));
    let off_policy = runtime.initial_state().with_horizon(Some(3));
    assert!(!runtime.policy.contains_key(&off_policy));
    SOLVE_CALLS.set(0);
    PARTITION_CALLS.set(0);
    let error = runtime
        .explain_state_cancellable(&off_policy, 5, 1000, &|| SOLVE_CALLS.get() >= 2)
        .expect_err("cancel recursive search");
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::Cancelled)
    ));
    assert_eq!(SOLVE_CALLS.get(), 2);
    assert_eq!(PARTITION_CALLS.get(), 1);
    let error = runtime
        .explain_state_cancellable(&off_policy, 1, 0, &|| false)
        .expect_err("off-policy primary must respect budget");
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::WorkLimit)
    ));
    let stopped = Cell::new(false);
    let mut solver = IndependentExactSolver::new(&runtime.model);
    let cancelled = || stopped.get();
    solver.cancelled = Some(&cancelled);
    solver.solve(&off_policy).unwrap();
    stopped.set(true);
    let error = solver
        .solve(&off_policy)
        .expect_err("memo hit respects cancellation");
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::Cancelled)
    ));
}

#[test]
fn combined_explanation_reuses_one_ranking_and_candidate_memo() {
    let (_root, runtime) = fixture();
    let state = runtime.initial_state().with_horizon(Some(3));
    SOLVE_CALLS.set(0);
    PARTITION_CALLS.set(0);
    let explanation = runtime
        .explain_state_cancellable(&state, 5, 1000, &|| false)
        .unwrap();
    assert_eq!(
        explanation.alternatives_status,
        FormalAlternativesStatus::Complete
    );
    // Five root guesses and three two-answer continuations cost at most 20
    // partitions; the combined response needs only five more for presentation.
    assert!(
        PARTITION_CALLS.get() <= 25,
        "combined response repeated exact search"
    );
    assert_eq!(explanation.best_guess, explanation.tied_moves[0].word);
    let mut shared = IndependentExactSolver::new(&runtime.model);
    let ranked = runtime
        .evaluate_state_ranked_with_solver(&state, &mut shared, None)
        .unwrap();
    assert_eq!(ranked.len(), explanation.tied_moves.len());
    let memo_states = shared.local_memo.len();
    assert!(memo_states > 0);
    PARTITION_CALLS.set(0);
    let repeated = runtime
        .evaluate_state_ranked_with_solver(&state, &mut shared, None)
        .unwrap();
    assert_eq!(shared.local_memo.len(), memo_states);
    assert_eq!(PARTITION_CALLS.get(), runtime.model.guesses.len());
    assert_eq!(repeated[0].guess_index, ranked[0].guess_index);
}
