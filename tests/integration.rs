use std::path::Path;

use maybe_wordle::{
    data::ProjectPaths,
    formal::{
        DEFAULT_EXPECTED_ONLY_MODEL_ID, DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime,
        FormalVerificationMode, build_optimal_policy, verify_optimal_policy_with_mode,
    },
    scoring::{format_feedback_letters, parse_feedback, score_guess},
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
