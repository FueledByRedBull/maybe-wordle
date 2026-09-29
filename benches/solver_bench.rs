use std::{
    hint::black_box,
    path::{Path, PathBuf},
    time::Duration,
};

#[path = "../src/test_support.rs"]
mod test_support;
use test_support::TestDirectory;

use chrono::NaiveDate;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use maybe_wordle::{
    config::{PriorConfig, SearchPolicyMode},
    data::ProjectPaths,
    formal::{
        DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime, FormalVerificationMode, build_optimal_policy,
        verify_optimal_policy_with_mode,
    },
    model::build_model_artifacts,
    predictive::{PredictiveSuggestRequest, PredictiveSuggestionMode, history_cutoff},
    scoring::parse_feedback,
    solver::Solver,
};

fn bench_predictive_recursive_exact(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let config = PriorConfig {
        exact_threshold: 16,
        exact_exhaustive_threshold: 8,
        exact_candidate_pool: 12,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let state = solver.initial_state(bench_date());

    c.bench_function("predictive_recursive_exact_suggestions", |bench| {
        bench.iter(|| solver.suggestions(&state, 5).expect("suggestions"));
    });
}

fn bench_predictive_proxy_only(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let config = PriorConfig {
        search_policy_mode: SearchPolicyMode::ProxyOnly,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let state = solver.initial_state(bench_date());

    c.bench_function("predictive_proxy_only_suggestions", |bench| {
        bench.iter(|| solver.suggestions(&state, 5).expect("suggestions"));
    });
}

fn bench_predictive_lookahead(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let config = PriorConfig {
        exact_threshold: 8,
        exact_exhaustive_threshold: 6,
        lookahead_threshold: 16,
        medium_state_lookahead_threshold: 12,
        lookahead_candidate_pool: 8,
        lookahead_reply_pool: 4,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let state = solver.initial_state(bench_date());

    c.bench_function("predictive_lookahead_suggestions", |bench| {
        bench.iter(|| solver.suggestions(&state, 5).expect("suggestions"));
    });
}

fn bench_predictive_danger_escalated_exact(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let config = PriorConfig {
        exact_threshold: 4,
        exact_exhaustive_threshold: 2,
        lookahead_threshold: 8,
        medium_state_lookahead_threshold: 6,
        danger_lookahead_threshold: 0.0,
        danger_exact_threshold: 0.0,
        danger_exact_root_pool: 10,
        danger_exact_survivor_cap: 16,
        ..PriorConfig::default()
    };
    let solver = Solver::from_paths(&paths, &config).expect("solver");
    let state = solver.initial_state(bench_date());

    c.bench_function("predictive_danger_escalated_exact_suggestions", |bench| {
        bench.iter(|| solver.suggestions(&state, 5).expect("suggestions"));
    });
}

fn bench_predictive_hard_cases(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let solver = Solver::from_paths(&paths, &PriorConfig::default()).expect("solver");

    c.bench_function("predictive_hard_case_report", |bench| {
        bench.iter(|| solver.hard_case_report(5).expect("hard cases"));
    });
}

fn bench_predictive_session_fallback_warm(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let solver = Solver::from_paths(&paths, &PriorConfig::default()).expect("solver");
    let as_of = bench_date();
    let _ = solver
        .suggestions_for_history(as_of, &[], 1)
        .expect("prime session fallback");

    c.bench_function("predictive_session_fallback_root_warm", |bench| {
        bench.iter(|| {
            solver
                .suggestions_for_history(as_of, &[], 1)
                .expect("session fallback suggestions")
        });
    });
}

fn bench_staged_gameplay(c: &mut Criterion) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let paths = ProjectPaths::new(&root);
    let config = PriorConfig::load(&paths.config_prior).expect("selected config");
    assert_eq!(config.search_policy_mode, SearchPolicyMode::Staged);
    let solver = Solver::from_paths(&paths, &config).expect("selected solver");
    let puzzle_date = NaiveDate::from_ymd_opt(2026, 8, 26).expect("development date");
    let as_of = history_cutoff(puzzle_date).expect("history cutoff");

    for (label, clues) in [
        ("first_feedback", &[("olate", "10001")][..]),
        (
            "four_feedback",
            &[
                ("olate", "10100"),
                ("vroom", "00020"),
                ("axion", "10022"),
                ("bagsy", "02000"),
            ][..],
        ),
    ] {
        let observations = clues
            .iter()
            .map(|(guess, code)| {
                (
                    (*guess).to_string(),
                    parse_feedback(code).expect("feedback"),
                )
            })
            .collect::<Vec<_>>();
        assert!(
            !solver
                .apply_history(as_of, &observations)
                .expect("reachable state")
                .surviving
                .is_empty()
        );
        let request = || PredictiveSuggestRequest {
            puzzle_date,
            observations: &observations,
            top: 5,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::LiveOnly,
        };
        let mut group = c.benchmark_group(format!("staged_gameplay_{label}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(2));
        group.measurement_time(Duration::from_secs(5));
        group.bench_function("history", |bench| {
            bench.iter(|| black_box(solver.apply_history(as_of, &observations).expect("history")))
        });
        group.bench_function("preview", |bench| {
            bench.iter(|| {
                black_box(
                    solver
                        .suggest_predictive_proxy_preview(request())
                        .expect("preview"),
                )
            })
        });
        // The first-feedback pooled exact path needs bounded single-call timing.
        if label == "four_feedback" {
            group.bench_function("full", |bench| {
                bench.iter(|| black_box(solver.suggest_predictive(request()).expect("full")))
            });
        }
        group.finish();
    }
}

// Set MAYBE_WORDLE_FORMAL_PROGRESS=0 before launching to silence formal progress.
fn bench_formal_build(c: &mut Criterion) {
    let fixture = formal_fixture();
    let paths = ProjectPaths::new(fixture.path());

    c.bench_function("formal_build_toy_policy", |bench| {
        bench.iter_batched(
            || (),
            |_| build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build"),
            BatchSize::SmallInput,
        );
    });
}

fn bench_formal_suggest(c: &mut Criterion) {
    let fixture = formal_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let _ = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build");
    let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("runtime");
    let state = runtime.initial_state();

    c.bench_function("formal_suggest_toy_root", |bench| {
        bench.iter(|| runtime.suggest(&state, 3).expect("suggest"));
    });
}

fn bench_formal_verify_certificate(c: &mut Criterion) {
    let fixture = formal_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let _ = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build");
    c.bench_function("formal_verify_toy_certificate", |bench| {
        bench.iter(|| {
            verify_optimal_policy_with_mode(
                &paths,
                DEFAULT_FORMAL_MODEL_ID,
                FormalVerificationMode::Certificate,
            )
            .expect("verify")
        });
    });
}

fn bench_formal_verify_oracle(c: &mut Criterion) {
    let fixture = formal_fixture();
    let paths = ProjectPaths::new(fixture.path());
    let _ = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build");
    c.bench_function("formal_verify_toy_oracle", |bench| {
        bench.iter(|| {
            verify_optimal_policy_with_mode(
                &paths,
                DEFAULT_FORMAL_MODEL_ID,
                FormalVerificationMode::Oracle,
            )
            .expect("verify")
        });
    });
}

fn predictive_fixture() -> TestDirectory {
    let fixture = TestDirectory::new("bench-predictive");
    let paths = ProjectPaths::new(fixture.path());
    paths.ensure_layout().expect("layout");

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
    write_fixture(&paths.raw_history, "");

    write_fixture(
        &paths.config_prior,
        &toml::to_string_pretty(&PriorConfig::default()).expect("config toml"),
    );
    build_model_artifacts(&paths, &PriorConfig::default(), bench_date()).expect("model");

    fixture
}

fn formal_fixture() -> TestDirectory {
    let fixture = TestDirectory::new("bench-formal");
    let paths = ProjectPaths::new(fixture.path());
    paths.ensure_layout().expect("layout");
    let formal_dir = fixture
        .path()
        .join(format!("data/formal/{DEFAULT_FORMAL_MODEL_ID}"));
    std::fs::create_dir_all(&formal_dir).expect("formal dir");

    write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
    write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
    write_fixture(&paths.seed_reference_answers, "");
    write_fixture(&paths.seed_sources, "");
    write_fixture(&paths.manual_additions, "");
    write_fixture(&paths.raw_history, "");
    write_fixture(&formal_dir.join("prior.toml"), "kind = \"uniform\"\n");

    fixture
}

fn write_fixture(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    std::fs::write(path, contents).expect("write fixture");
}

fn bench_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 3, 9).expect("valid date")
}

criterion_group!(
    benches,
    bench_predictive_recursive_exact,
    bench_predictive_proxy_only,
    bench_predictive_lookahead,
    bench_predictive_danger_escalated_exact,
    bench_predictive_hard_cases,
    bench_predictive_session_fallback_warm,
    bench_staged_gameplay,
    bench_formal_build,
    bench_formal_suggest,
    bench_formal_verify_certificate,
    bench_formal_verify_oracle
);
criterion_main!(benches);
