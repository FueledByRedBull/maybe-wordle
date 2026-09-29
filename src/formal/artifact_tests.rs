use super::*;
use std::io::Cursor;

fn inline_state(horizon: u16, indices: &[u16]) -> Vec<u8> {
    let mut bytes = horizon.to_le_bytes().to_vec();
    bytes.push(STATE_TAG_INLINE);
    bytes.extend_from_slice(&(indices.len() as u16).to_le_bytes());
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    bytes
}

#[test]
fn malformed_state_encodings_are_rejected_without_panicking() {
    let tokens = build_zobrist_tokens(3);
    let mut invalid_bitset = 2u16.to_le_bytes().to_vec();
    invalid_bitset.push(STATE_TAG_BITSET);
    invalid_bitset.extend_from_slice(&1u16.to_le_bytes());
    invalid_bitset.extend_from_slice(&(1u64 << 63).to_le_bytes());
    for bytes in [
        inline_state(2, &[]),
        inline_state(2, &[3]),
        inline_state(2, &[0, 0]),
        inline_state(2, &[1, 0]),
        inline_state(0, &[0]),
        invalid_bitset,
    ] {
        let parsed =
            std::panic::catch_unwind(|| StateKey::read_tagged(&mut Cursor::new(bytes), 3, &tokens));
        assert!(parsed.is_ok(), "malformed state panicked");
        assert!(parsed.unwrap().is_err(), "malformed state was accepted");
    }
}

fn fixture() -> (
    crate::test_support::TestDirectory,
    ProjectPaths,
    FormalPolicyRuntime,
) {
    let root = crate::test_support::TestDirectory::new("formal-artifacts");
    let paths = ProjectPaths::new(root.path());
    paths.ensure_layout().unwrap();
    let inputs = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
    fs::create_dir_all(&inputs.model_dir).unwrap();
    fs::write(&paths.seed_guesses, "aaaaa\nbbbbb\nccccc\nabcde\n").unwrap();
    fs::write(&paths.seed_answers, "aaaaa\nbbbbb\nccccc\n").unwrap();
    fs::write(
        &inputs.prior_spec,
        "objective = \"lexicographic\"\nkind = \"uniform\"\n",
    )
    .unwrap();
    let summary = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
    assert!(
        summary.build_millis
            >= summary.solve_millis + summary.certificate_millis + summary.persistence_millis
    );
    let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
    (root, paths, runtime)
}

#[test]
fn binary_counts_duplicates_nonfinite_and_trailing_data_are_rejected() {
    let (root, _paths, runtime) = fixture();
    let original = fs::read(&runtime.artifacts.values).unwrap();
    let scratch = root.path().join("mutated-values.bin");
    for change in 0..4 {
        let mut bytes = original.clone();
        match change {
            0 => bytes[8 + TAGGED_DIGEST_LENGTH..8 + TAGGED_DIGEST_LENGTH + 8]
                .copy_from_slice(&u64::MAX.to_le_bytes()),
            1 => bytes.push(0),
            2 => {
                let end = bytes.len();
                bytes[end - 8..].copy_from_slice(&f64::NAN.to_le_bytes());
            }
            _ => {
                let entry = &original[BINARY_HEADER_BYTES as usize..];
                bytes.extend_from_slice(entry);
                let count = u64::from_le_bytes(original[82..90].try_into().unwrap());
                bytes[82..90].copy_from_slice(&(count * 2).to_le_bytes());
            }
        }
        fs::write(&scratch, bytes).unwrap();
        assert!(read_values(&scratch, &runtime.model).is_err());
    }
    for position in 0..original.len() {
        let mut bytes = original.clone();
        bytes[position] ^= 0xff;
        fs::write(&scratch, bytes).unwrap();
        assert!(std::panic::catch_unwind(|| read_values(&scratch, &runtime.model)).is_ok());
    }
    for length in 0..original.len() {
        fs::write(&scratch, &original[..length]).unwrap();
        assert!(
            std::panic::catch_unwind(|| read_values(&scratch, &runtime.model))
                .unwrap()
                .is_err()
        );
    }
}

#[test]
fn json_vectors_and_file_sizes_are_bounded_before_decoding() {
    #[derive(Deserialize)]
    struct Tiny {
        #[serde(deserialize_with = "parsing::bounded_vec::<_, _, 2>")]
        entries: Vec<u8>,
    }
    assert_eq!(
        serde_json::from_str::<Tiny>("{\"entries\":[1,2]}")
            .unwrap()
            .entries
            .len(),
        2
    );
    assert!(serde_json::from_str::<Tiny>("{\"entries\":[1,2,3]}").is_err());
    let root = crate::test_support::TestDirectory::new("formal-size-limit");
    let path = root.path().join("oversized.json");
    File::create(&path)
        .unwrap()
        .set_len(parsing::MAX_JSON_BYTES + 1)
        .unwrap();
    assert!(parsing::read_json::<ProofMetadata>(&path, parsing::MAX_JSON_BYTES).is_err());
}

#[test]
fn certificate_mutations_never_panic_or_escape_structural_validation() {
    let (_root, _paths, runtime) = fixture();
    let certificate: ProofCertificate = parsing::read_json(
        &runtime.artifacts.certificate,
        parsing::MAX_CERTIFICATE_BYTES,
    )
    .unwrap();
    for mutation in 0..8 {
        let mut changed = certificate.clone();
        match mutation {
            0 => changed.root_state_id = u32::MAX,
            1 => changed.state_count = usize::MAX,
            2 => changed.states[0].answer_indices[0] = u16::MAX,
            3 => changed.states[0].horizon = Some(0),
            4 => changed.states[0].best_objective.worst_case_depth = u8::MAX,
            5 => changed.states[0].best_objective.expected_guesses = f64::INFINITY,
            6 => changed.states[0].candidates[0].guess_index = usize::MAX,
            _ => {
                changed.states[0].candidates[0].witness =
                    PersistedCandidateWitness::NonProgress { pattern: u8::MAX }
            }
        }
        assert!(
            std::panic::catch_unwind(|| verify_certificate(&runtime, &changed))
                .unwrap()
                .is_err()
        );
    }
}

#[test]
fn interrupted_generation_publication_preserves_old_policy_and_prior() {
    let (_root, paths, runtime) = fixture();
    let input = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
    let pointer = input.model_dir.join("current.json");
    let old_pointer = fs::read(&pointer).unwrap();
    let prior = fs::read(&input.prior_spec).unwrap();
    let certificate: ProofCertificate = parsing::read_json(
        &runtime.artifacts.certificate,
        parsing::MAX_CERTIFICATE_BYTES,
    )
    .unwrap();
    for stage in 1..=9 {
        generation::FAIL_AFTER_STAGE.set(Some(stage));
        let result = generation::publish(
            &runtime.model,
            &runtime.policy,
            runtime.metadata.clone(),
            &certificate,
            &paths,
            Instant::now(),
            &|| false,
        );
        generation::FAIL_AFTER_STAGE.set(None);
        assert!(result.is_err(), "stage {stage}");
        assert_eq!(fs::read(&pointer).unwrap(), old_pointer);
        assert_eq!(fs::read(&input.prior_spec).unwrap(), prior);
        let reloaded = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
        assert_eq!(reloaded.artifacts.model_dir, runtime.artifacts.model_dir);
    }
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..12 {
                FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
            }
        });
        generation::publish(
            &runtime.model,
            &runtime.policy,
            runtime.metadata.clone(),
            &certificate,
            &paths,
            Instant::now(),
            &|| false,
        )
        .unwrap();
    });
    assert_ne!(fs::read(&pointer).unwrap(), old_pointer);
    assert_eq!(fs::read(&input.prior_spec).unwrap(), prior);
    assert!(
        runtime.artifacts.certificate.exists(),
        "old generation must remain intact"
    );
    verify_certificate(&runtime, &certificate).unwrap();
}

#[test]
fn generation_pointer_and_member_tampering_fail_without_repair() {
    let (_root, paths, runtime) = fixture();
    let input = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
    let pointer = input.model_dir.join("current.json");
    let original = fs::read(&pointer).unwrap();
    for mutation in 0..4 {
        let mut changed: serde_json::Value = serde_json::from_slice(&original).unwrap();
        match mutation {
            0 => changed["generation"] = "../outside".into(),
            1 => changed["members"][0]["name"] = "../prior.toml".into(),
            2 => changed["members"][0]["bytes"] = u64::MAX.into(),
            _ => changed["manifest_hash"] = "unbound".into(),
        }
        fs::write(&pointer, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).is_err());
    }
    fs::write(&pointer, original).unwrap();
    let pattern = fs::read(&runtime.artifacts.pattern_table).unwrap();
    fs::write(&runtime.artifacts.pattern_table, b"broken").unwrap();
    assert!(FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).is_err());
    assert_eq!(
        fs::read(&runtime.artifacts.pattern_table).unwrap(),
        b"broken"
    );
    fs::write(&runtime.artifacts.pattern_table, pattern).unwrap();
    FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap();
}

#[test]
fn bounded_formal_reader_rejects_non_regular_paths() {
    let root = crate::test_support::TestDirectory::new("formal-non-regular");
    let directory = root.path().join("artifact");
    fs::create_dir(&directory).unwrap();

    let error = parsing::read_bounded(&directory, 1024).unwrap_err();

    assert!(
        error.to_string().contains("regular file"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn generation_rejects_directory_members_before_digesting() {
    let (_root, paths, runtime) = fixture();
    let member = runtime.artifacts.small_state_table.clone();
    fs::remove_file(&member).unwrap();
    fs::create_dir(&member).unwrap();

    let error = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap_err();

    assert!(
        error.to_string().contains("regular, local file"),
        "unexpected error: {error:#}"
    );
}

#[cfg(unix)]
#[test]
fn generation_rejects_fifo_members_without_opening() {
    let (_root, paths, runtime) = fixture();
    let member = runtime.artifacts.small_state_table.clone();
    fs::remove_file(&member).unwrap();
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&member)
            .status()
            .expect("POSIX mkfifo utility")
            .success()
    );

    let error = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).unwrap_err();

    assert!(
        error.to_string().contains("regular, local file"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn controlled_build_certificate_verify_and_publication_stop_without_replacing_policy() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (_root, paths, runtime) = fixture();
    let pointer = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID)
        .model_dir
        .join("current.json");
    let original = fs::read(&pointer).unwrap();
    let calls = AtomicUsize::new(0);
    let error = build_optimal_policy_controlled(&paths, DEFAULT_FORMAL_MODEL_ID, &|| {
        calls.fetch_add(1, Ordering::Relaxed) >= 7
    })
    .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::Cancelled)
    ));
    assert_eq!(fs::read(&pointer).unwrap(), original);
    let calls = AtomicUsize::new(0);
    let error = build_exhaustive_proof_certificate(&runtime.model, &runtime.policy, &|| {
        calls.fetch_add(1, Ordering::Relaxed) >= 5
    })
    .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::Cancelled)
    ));
    let certificate: ProofCertificate = parsing::read_json(
        &runtime.artifacts.certificate,
        parsing::MAX_CERTIFICATE_BYTES,
    )
    .unwrap();
    let calls = AtomicUsize::new(0);
    let error = verifier::verify_certificate_witnesses_controlled(&runtime, &certificate, &|| {
        calls.fetch_add(1, Ordering::Relaxed) >= 3
    })
    .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::Cancelled)
    ));
    let calls = AtomicUsize::new(0);
    let error = generation::publish(
        &runtime.model,
        &runtime.policy,
        runtime.metadata.clone(),
        &certificate,
        &paths,
        Instant::now(),
        &|| calls.fetch_add(1, Ordering::Relaxed) >= 3,
    )
    .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<FormalSearchStop>(),
        Some(FormalSearchStop::Cancelled)
    ));
    assert_eq!(fs::read(&pointer).unwrap(), original);
}

#[test]
fn artifact_presence_does_not_hide_corruption_or_flat_artifacts() {
    let root = crate::test_support::TestDirectory::new("formal-presence");
    let paths = ProjectPaths::new(root.path());
    let artifacts = PolicyArtifactSet::for_model(&paths, DEFAULT_FORMAL_MODEL_ID);
    fs::create_dir_all(&artifacts.model_dir).unwrap();
    fs::write(&artifacts.prior_spec, "kind = \"uniform\"\n").unwrap();
    assert!(!artifacts_exist(&paths, DEFAULT_FORMAL_MODEL_ID));
    fs::write(&artifacts.certificate, "invalid flat artifact").unwrap();
    assert!(artifacts_exist(&paths, DEFAULT_FORMAL_MODEL_ID));
    assert!(FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).is_err());
}
