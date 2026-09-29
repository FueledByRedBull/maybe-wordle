use super::*;

fn fixture(name: &str, model_id: &str, prior: &str) -> ProjectPaths {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/audit-work/formal-tests")
        .join(format!("{name}-{}", std::process::id()));
    let paths = ProjectPaths::new(&root);
    paths.ensure_layout().expect("fixture layout");
    let artifacts = PolicyArtifactSet::for_model(&paths, model_id);
    fs::create_dir_all(&artifacts.model_dir).expect("fixture model directory");
    let answers = "tower\npower\nbower\nrower\nsware\ncrare\nbeare\nurare\nblare\n";
    fs::write(&paths.seed_answers, answers).expect("fixture answers");
    fs::write(&paths.seed_guesses, format!("{answers}mesne\n")).expect("fixture guesses");
    fs::write(&artifacts.prior_spec, prior).expect("fixture prior");
    paths
}

const WEIGHTED_PRIOR: &str = "kind = \"explicit\"\n[weights]\ntower = 1\npower = 10\nbower = 10\nrower = 2\nsware = 50\ncrare = 50\nbeare = 5\nurare = 2\nblare = 2\n";

#[test]
fn weighted_nine_answer_policy_uses_child_depth_slack() {
    let paths = fixture("weighted-slack", DEFAULT_FORMAL_MODEL_ID, WEIGHTED_PRIOR);
    let built = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build");
    assert_eq!(built.root_objective.worst_case_depth, 4);
    assert!(
        (built.root_objective.expected_guesses - 235.0 / 132.0).abs() < 1e-12,
        "expected 235/132, got {} with {}",
        built.root_objective.expected_guesses,
        built.root_best_guess
    );
    assert_eq!(built.root_best_guess, "crare");
    let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("reload");
    let root = runtime.initial_state();
    assert_eq!(
        runtime.suggest(&root, 1).expect("root suggestion")[0].word,
        "crare"
    );
    let feedback = crate::scoring::score_guess("crare", "sware");
    let child = runtime
        .apply_feedback(&root, "crare", feedback)
        .expect("child");
    let suggestion = runtime.suggest(&child, 1).expect("child suggestion");
    assert_eq!(suggestion[0].word, "sware");
    assert_eq!(suggestion[0].objective.worst_case_depth, 3);
    assert!((suggestion[0].objective.expected_guesses - 66.0 / 57.0).abs() < 1e-12);
    verify_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("independent certificate");
}

#[test]
fn explicit_objective_is_independent_of_model_name() {
    for model_id in ["unexpected-case", "renamed-case"] {
        for (kind, id) in [
            ("lexicographic", "worst_case_depth_then_expected_guesses"),
            ("expected_only", "expected_guesses_only"),
        ] {
            let paths = fixture(
                &format!("{model_id}-{kind}"),
                model_id,
                &format!("objective = \"{kind}\"\nkind = \"uniform\"\n"),
            );
            let model = FormalModel::load(&paths, model_id).expect("explicit objective");
            assert_eq!(model.manifest.objective_id, id);
        }
    }
}

#[test]
fn objective_configuration_rejects_unknown_and_ambiguous_values() {
    for (name, prior) in [
        ("missing-objective", "kind = \"uniform\"\n"),
        (
            "unknown-objective",
            "objective = \"unexpected\"\nkind = \"uniform\"\n",
        ),
    ] {
        let paths = fixture(name, "unexpected-custom", prior);
        assert!(FormalModel::load(&paths, "unexpected-custom").is_err());
    }
    for (id, objective) in [
        (DEFAULT_FORMAL_MODEL_ID, FormalObjectiveKind::Lexicographic),
        (
            DEFAULT_EXPECTED_ONLY_MODEL_ID,
            FormalObjectiveKind::ExpectedOnly,
        ),
    ] {
        let paths = fixture(
            &format!("builtin-migration-{id}"),
            id,
            "kind = \"uniform\"\n",
        );
        assert_eq!(
            FormalModel::load(&paths, id)
                .expect("built-in migration")
                .manifest
                .objective_kind,
            objective
        );
    }
}

#[test]
fn reloading_rejects_changed_objective_and_old_binary_versions() {
    let id = "explicit-artifacts";
    let paths = fixture(
        "artifact-identity",
        id,
        "objective = \"lexicographic\"\nkind = \"uniform\"\n",
    );
    build_optimal_policy(&paths, id).expect("build");
    let artifacts = PolicyArtifactSet::published(&paths, id).expect("generation");
    let original_manifest = fs::read(&artifacts.manifest).unwrap();
    let mut manifest: FormalManifest = serde_json::from_slice(&original_manifest).unwrap();
    manifest.objective_kind = FormalObjectiveKind::ExpectedOnly;
    fs::write(&artifacts.manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(FormalPolicyRuntime::load(&paths, id).is_err());
    fs::write(&artifacts.manifest, original_manifest).unwrap();
    let source_prior = PolicyArtifactSet::for_model(&paths, id).prior_spec;
    fs::write(
        &source_prior,
        "objective = \"expected_only\"\nkind = \"uniform\"\n",
    )
    .unwrap();
    assert!(FormalPolicyRuntime::load(&paths, id).is_err());
    fs::write(
        &source_prior,
        "objective = \"lexicographic\"\nkind = \"uniform\"\n",
    )
    .unwrap();
    FormalPolicyRuntime::load(&paths, id).expect("restored current prior matches snapshot");
    let model = FormalModel::load(&paths, id).unwrap();
    let mut values = fs::read(&artifacts.values).unwrap();
    values[..8].copy_from_slice(b"MWORDVV2");
    fs::write(&artifacts.values, values).unwrap();
    assert!(read_values(&artifacts.values, &model).is_err());
    let mut policies = fs::read(&artifacts.policy).unwrap();
    policies[..8].copy_from_slice(b"MWORDPV2");
    fs::write(&artifacts.policy, policies).unwrap();
    assert!(read_policy(&artifacts.policy, &model).is_err());
}

#[test]
fn certificates_reject_invented_infeasibility_and_changed_horizons() {
    let paths = fixture(
        "certificate-horizons",
        DEFAULT_FORMAL_MODEL_ID,
        WEIGHTED_PRIOR,
    );
    build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build");
    let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("reload");
    let certificate = read_proof_certificate(&paths, DEFAULT_FORMAL_MODEL_ID).expect("certificate");
    let mut changed = certificate.clone();
    changed.states[changed.root_state_id as usize].horizon = Some(5);
    assert!(verify_certificate(&runtime, &changed).is_err());
    let mut changed = certificate.clone();
    let candidate = changed
        .states
        .iter_mut()
        .flat_map(|state| &mut state.candidates)
        .find(|candidate| matches!(candidate.witness, PersistedCandidateWitness::Exact { .. }))
        .unwrap();
    candidate.witness = PersistedCandidateWitness::Infeasible;
    assert!(verify_certificate(&runtime, &changed).is_err());
    let mut changed = certificate.clone();
    changed.states[0].horizon = changed.states[0].horizon.map(|horizon| horizon + 1);
    assert!(verify_certificate(&runtime, &changed).is_err());
    let mut changed = certificate;
    changed.certificate_format_version -= 1;
    assert!(verify_certificate(&runtime, &changed).is_err());
}

#[test]
fn state_identity_and_binary_round_trip_include_horizon() {
    let tokens = build_zobrist_tokens(4);
    let state = StateKey::full(4, &tokens);
    let states = [
        state.clone(),
        state.with_horizon(Some(2)),
        state.with_horizon(Some(3)),
    ];
    let map = states
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect::<HashMap<_, _>>();
    assert_eq!(map.len(), 3);
    for state in states {
        let mut bytes = Vec::new();
        state.write_tagged(&mut bytes, 4).unwrap();
        let restored = StateKey::read_tagged(&mut std::io::Cursor::new(bytes), 4, &tokens).unwrap();
        assert_eq!(state, restored);
    }
}

#[test]
fn formal_model_rejects_empty_or_unguessable_answer_universes() {
    for (name, guesses, answers) in [
        ("empty-guesses", "", "cigar\n"),
        ("empty-answers", "cigar\n", ""),
        ("unguessable-answer", "cigar\n", "cigar\nrebut\n"),
    ] {
        let paths = fixture(name, DEFAULT_FORMAL_MODEL_ID, "kind = \"uniform\"\n");
        fs::write(&paths.seed_guesses, guesses).unwrap();
        fs::write(&paths.seed_answers, answers).unwrap();
        let error =
            FormalModel::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect_err("invalid universe");
        assert!(
            error.to_string().contains("formal answer universe"),
            "{error:#}"
        );
    }
}

#[test]
fn metadata_cannot_change_the_persisted_root_horizon() {
    let paths = fixture("metadata-root", DEFAULT_FORMAL_MODEL_ID, WEIGHTED_PRIOR);
    build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build");
    let artifacts =
        PolicyArtifactSet::published(&paths, DEFAULT_FORMAL_MODEL_ID).expect("generation");
    let original_bytes = fs::read(&artifacts.metadata).unwrap();
    let original: ProofMetadata = serde_json::from_slice(&original_bytes).unwrap();
    for depth in [0, 5] {
        let mut changed = original.clone();
        changed.root_objective.worst_case_depth = depth;
        fs::write(&artifacts.metadata, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(
            FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).is_err(),
            "metadata depth {depth} must not override the persisted depth 4"
        );
        assert!(
            FormalPolicyRuntime::load_generation(
                &paths,
                DEFAULT_FORMAL_MODEL_ID,
                artifacts.clone()
            )
            .is_err()
        );
    }
    for mutation in 0..4 {
        let mut changed = original.clone();
        match mutation {
            0 => changed.root_objective.expected_guesses += 0.25,
            1 => changed.model_id = "different-model".to_string(),
            2 => changed.manifest_hash = "different-manifest".to_string(),
            3 => changed.solved_states += 1,
            _ => unreachable!(),
        }
        fs::write(&artifacts.metadata, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).is_err());
        assert!(
            FormalPolicyRuntime::load_generation(
                &paths,
                DEFAULT_FORMAL_MODEL_ID,
                artifacts.clone()
            )
            .is_err()
        );
    }
    fs::write(&artifacts.metadata, original_bytes).unwrap();
    let mut runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
    let certificate = read_proof_certificate(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
    runtime.metadata.root_objective.worst_case_depth = 5;
    assert_eq!(runtime.initial_state().horizon, Some(4));
    assert!(verify_certificate(&runtime, &certificate).is_err());
}

// Enumerate exact-depth policy frontiers bottom-up with integer weighted path
// lengths. This shares neither the recursive horizon search nor its partitions,
// floating-point costs, memo keys, or pruning with the production solvers.
fn policy_frontier(model: &FormalModel, weights: &[u64]) -> Vec<Option<(u64, usize)>> {
    let size = 1usize << model.answers.len();
    let mut frontiers: Vec<Vec<Option<(u64, usize)>>> = vec![Vec::new(); size];
    for mask in 1usize..size {
        let count = mask.count_ones() as usize;
        let mass: u64 = weights
            .iter()
            .enumerate()
            .filter(|(index, _)| mask & (1 << index) != 0)
            .map(|(_, weight)| *weight)
            .sum();
        let mut frontier: Vec<Option<(u64, usize)>> = vec![None; count + 1];
        for (guess_index, guess) in model.guesses.iter().enumerate() {
            let mut branches = [0usize; PATTERN_SPACE];
            for (answer_index, answer) in model.answers.iter().enumerate() {
                if mask & (1 << answer_index) != 0 {
                    let pattern = crate::scoring::score_guess(guess, answer) as usize;
                    if pattern != ALL_GREEN_PATTERN as usize {
                        branches[pattern] |= 1 << answer_index;
                    }
                }
            }
            if branches.contains(&mask) {
                continue;
            }
            let mut combinations = vec![None; count];
            combinations[0] = Some(0u64);
            for child in branches.into_iter().filter(|child| *child != 0) {
                let mut next = vec![None; count];
                for (depth, cost) in combinations.iter().enumerate() {
                    let Some(cost) = cost else {
                        continue;
                    };
                    for (child_depth, child_value) in frontiers[child].iter().enumerate() {
                        let Some((child_cost, _)) = child_value else {
                            continue;
                        };
                        let slot = &mut next[depth.max(child_depth)];
                        let candidate = cost + child_cost;
                        if slot.is_none_or(|current| candidate < current) {
                            *slot = Some(candidate);
                        }
                    }
                }
                combinations = next;
            }
            for (depth, cost) in combinations.into_iter().enumerate() {
                let Some(cost) = cost else {
                    continue;
                };
                let candidate = (mass + cost, guess_index);
                let slot = &mut frontier[depth + 1];
                if slot.is_none_or(|current| {
                    candidate.0 < current.0
                        || (candidate.0 == current.0 && guess < &model.guesses[current.1])
                }) {
                    *slot = Some(candidate);
                }
            }
        }
        frontiers[mask] = frontier;
    }
    frontiers.pop().expect("nonempty universe")
}

fn builder(model: FormalModel) -> FormalPolicyBuilder<'static> {
    let started = Instant::now();
    FormalPolicyBuilder {
        model,
        memo: HashMap::new(),
        hot_tt: HotTranspositionTable::new(1024 * 1024),
        deduped_signatures: 0,
        bound_hits: 0,
        root_refinement_pruned: 0,
        local_refinement_pruned: 0,
        partition_calls: 0,
        quick_plan_calls: 0,
        started,
        last_progress: started,
        cancelled: None,
    }
}

#[test]
fn weighted_and_uniform_horizons_match_integer_policy_frontier() {
    for (name, prior) in [
        ("frontier-weighted", WEIGHTED_PRIOR),
        ("frontier-uniform", "kind = \"uniform\"\n"),
    ] {
        let paths = fixture(name, DEFAULT_FORMAL_MODEL_ID, prior);
        let mut model = FormalModel::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("model");
        let weights = model
            .answers
            .iter()
            .map(|answer| {
                if prior == WEIGHTED_PRIOR {
                    match answer.as_str() {
                        "tower" => 1,
                        "power" | "bower" => 10,
                        "rower" | "urare" | "blare" => 2,
                        "sware" | "crare" => 50,
                        "beare" => 5,
                        _ => unreachable!(),
                    }
                } else {
                    1
                }
            })
            .collect::<Vec<_>>();
        let frontier = policy_frontier(&model, &weights);
        let mass = weights.iter().sum::<u64>() as f64;
        let state = StateKey::full(model.answers.len(), &model.zobrist);
        // Exercise the bounded optimizer as well as the small-state oracle.
        model.small_state_table.max_size = 0;
        let mut optimized = builder(model);
        for horizon in 1..=5 {
            let expected = frontier
                .iter()
                .enumerate()
                .take(horizon + 1)
                .filter_map(|(depth, entry)| entry.map(|(cost, guess)| (cost, depth, guess)))
                .min_by(|left, right| {
                    left.0
                        .cmp(&right.0)
                        .then_with(|| left.1.cmp(&right.1))
                        .then_with(|| {
                            optimized.model.guesses[left.2].cmp(&optimized.model.guesses[right.2])
                        })
                });
            let state = state.with_horizon(Some(horizon as u8));
            let actual = optimized.solve_conditioned(&state).expect("optimizer");
            let independent = IndependentExactSolver::new(&optimized.model)
                .solve_conditioned(&state)
                .expect("oracle");
            assert_eq!(actual.is_some(), expected.is_some());
            assert_eq!(independent.is_some(), expected.is_some());
            if let Some((cost, depth, guess)) = expected {
                for actual in [actual.unwrap(), independent.unwrap()] {
                    assert!((actual.objective.expected_guesses - cost as f64 / mass).abs() < 1e-12);
                    assert_eq!(actual.objective.worst_case_depth as usize, depth);
                    assert_eq!(actual.best_guess, guess);
                }
            }
        }
    }
}

#[test]
fn expected_only_above_small_state_limit_matches_unconstrained_frontier() {
    let paths = fixture(
        "expected-large",
        DEFAULT_EXPECTED_ONLY_MODEL_ID,
        WEIGHTED_PRIOR,
    );
    let mut model = FormalModel::load(&paths, DEFAULT_EXPECTED_ONLY_MODEL_ID).expect("model");
    // The eight -ower words need two probes when tower is guessed first,
    // while clomp/stubs can classify them in two turns before the answer.
    model.answers = "tower power bower rower sower lower cower mower cigar humph vivid junky beare"
        .split_whitespace()
        .map(str::to_string)
        .collect();
    model.guesses = model.answers.clone();
    model
        .guesses
        .extend(["clomp".to_string(), "stubs".to_string()]);
    model.guess_index = model
        .guesses
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i))
        .collect();
    model.zobrist = build_zobrist_tokens(model.answers.len());
    let weights = model
        .answers
        .iter()
        .map(|answer| match answer.as_str() {
            "tower" => 10000,
            "power" => 1000,
            "bower" => 100,
            "rower" => 10,
            _ => 1,
        })
        .collect::<Vec<u64>>();
    let mass = weights.iter().sum::<u64>() as f64;
    model.prior = weights.iter().map(|weight| *weight as f64 / mass).collect();
    let answers = model
        .answers
        .iter()
        .map(|word| AnswerRecord {
            word: word.clone(),
            in_seed: true,
            manual_entry: false,
            manual_weight: 1.0,
            history_dates: Vec::new(),
        })
        .collect::<Vec<_>>();
    model.pattern_table = PatternTable::load_or_build_at(
        &paths.root.join("expanded-patterns.bin"),
        &model.guesses,
        &answers,
    )
    .expect("expanded patterns");
    let frontier = policy_frontier(&model, &weights);
    let expected = frontier
        .iter()
        .enumerate()
        .filter_map(|(depth, entry)| entry.map(|(cost, guess)| (cost, depth, guess)))
        .min_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| model.guesses[left.2].cmp(&model.guesses[right.2]))
        })
        .expect("frontier");
    let minimum_depth = frontier
        .iter()
        .position(Option::is_some)
        .expect("minimum depth");
    assert!(
        expected.1 > minimum_depth,
        "fixture must distinguish objectives: {frontier:?}"
    );
    let state = StateKey::full(model.answers.len(), &model.zobrist);
    assert!(state.count() > model.small_state_table.max_size);
    let mut optimized = builder(model);
    let actual = optimized
        .solve_state(&state)
        .expect("expected-only optimizer");
    assert!(optimized.quick_plan_calls > 0);
    assert_eq!(actual.objective.worst_case_depth as usize, expected.1);
    assert_eq!(actual.best_guess, expected.2);
    assert!((actual.objective.expected_guesses - expected.0 as f64 / mass).abs() < 1e-12);
}
