use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use chrono::NaiveDate;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use maybe_wordle::{
    config::PriorConfig,
    data::ProjectPaths,
    formal::{
        DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime, FormalVerificationMode, build_optimal_policy,
        verify_optimal_policy_with_mode,
    },
    model::build_model_artifacts,
    small_state::SmallStateTable,
    solver::Solver,
};

struct PredictiveBenchFixture {
    root: PathBuf,
}

struct FormalBenchFixture {
    root: PathBuf,
}

fn bench_predictive_recursive_exact(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(&fixture.root);
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
    let paths = ProjectPaths::new(&fixture.root);
    let config = PriorConfig {
        exact_threshold: 0,
        lookahead_threshold: 0,
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
    let paths = ProjectPaths::new(&fixture.root);
    let config = PriorConfig {
        exact_threshold: 8,
        exact_exhaustive_threshold: 6,
        lookahead_threshold: 16,
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
    let paths = ProjectPaths::new(&fixture.root);
    let config = PriorConfig {
        exact_threshold: 4,
        exact_exhaustive_threshold: 2,
        lookahead_threshold: 4,
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
    let paths = ProjectPaths::new(&fixture.root);
    let solver = Solver::from_paths(&paths, &PriorConfig::default()).expect("solver");

    c.bench_function("predictive_hard_case_report", |bench| {
        bench.iter(|| solver.hard_case_report(5).expect("hard cases"));
    });
}

fn bench_predictive_session_fallback_warm(c: &mut Criterion) {
    let fixture = predictive_fixture();
    let paths = ProjectPaths::new(&fixture.root);
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

fn bench_formal_build(c: &mut Criterion) {
    let fixture = formal_fixture();
    let paths = ProjectPaths::new(&fixture.root);
    unsafe {
        std::env::set_var("MAYBE_WORDLE_FORMAL_PROGRESS", "0");
    }

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
    let paths = ProjectPaths::new(&fixture.root);
    unsafe {
        std::env::set_var("MAYBE_WORDLE_FORMAL_PROGRESS", "0");
    }
    let _ = build_optimal_policy(&paths, DEFAULT_FORMAL_MODEL_ID).expect("build");
    let runtime = FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).expect("runtime");
    let state = runtime.initial_state();

    c.bench_function("formal_suggest_toy_root", |bench| {
        bench.iter(|| runtime.suggest(&state, 3).expect("suggest"));
    });
}

fn bench_formal_verify_certificate(c: &mut Criterion) {
    let fixture = formal_fixture();
    let paths = ProjectPaths::new(&fixture.root);
    unsafe {
        std::env::set_var("MAYBE_WORDLE_FORMAL_PROGRESS", "0");
    }
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
    let paths = ProjectPaths::new(&fixture.root);
    unsafe {
        std::env::set_var("MAYBE_WORDLE_FORMAL_PROGRESS", "0");
    }
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

fn bench_small_state_table(c: &mut Criterion) {
    c.bench_function("small_state_table_build_12", |bench| {
        bench.iter(|| SmallStateTable::build(12));
    });
}

fn predictive_fixture() -> &'static PredictiveBenchFixture {
    static FIXTURE: OnceLock<PredictiveBenchFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let root = unique_bench_root("predictive");
        let paths = ProjectPaths::new(&root);
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

        PredictiveBenchFixture { root }
    })
}

fn formal_fixture() -> &'static FormalBenchFixture {
    static FIXTURE: OnceLock<FormalBenchFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let root = unique_bench_root("formal");
        let paths = ProjectPaths::new(&root);
        paths.ensure_layout().expect("layout");
        let formal_dir = root.join(format!("data/formal/{DEFAULT_FORMAL_MODEL_ID}"));
        std::fs::create_dir_all(&formal_dir).expect("formal dir");

        write_fixture(&paths.seed_guesses, "cigar\nrebut\nsissy\nhumph\n");
        write_fixture(&paths.seed_answers, "cigar\nrebut\nsissy\n");
        write_fixture(&paths.seed_reference_answers, "");
        write_fixture(&paths.seed_sources, "");
        write_fixture(&paths.manual_additions, "");
        write_fixture(&paths.raw_history, "");
        write_fixture(&formal_dir.join("prior.toml"), "kind = \"uniform\"\n");

        FormalBenchFixture { root }
    })
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

fn unique_bench_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("maybe-wordle-bench-{label}-{nanos}"))
}

criterion_group!(
    benches,
    bench_predictive_recursive_exact,
    bench_predictive_proxy_only,
    bench_predictive_lookahead,
    bench_predictive_danger_escalated_exact,
    bench_predictive_hard_cases,
    bench_predictive_session_fallback_warm,
    bench_formal_build,
    bench_formal_suggest,
    bench_formal_verify_certificate,
    bench_formal_verify_oracle,
    bench_small_state_table
);
criterion_main!(benches);
