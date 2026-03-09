use criterion::{Criterion, criterion_group, criterion_main};
use maybe_wordle::scoring::score_guess;

fn bench_score_guess(c: &mut Criterion) {
    c.bench_function("score_guess_lilly_alley", |bench| {
        bench.iter(|| score_guess("lilly", "alley"));
    });
}

criterion_group!(benches, bench_score_guess);
criterion_main!(benches);
