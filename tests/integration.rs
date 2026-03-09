use maybe_wordle::scoring::{format_feedback_letters, parse_feedback, score_guess};

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
