//! Shared history rules for interactive and replay consumers.

use anyhow::{Result, bail};

use crate::scoring::{ALL_GREEN_PATTERN, PATTERN_SPACE, parse_feedback};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameRules {
    Wordle,
    Absurdle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameStatus {
    Active,
    Solved,
    Exhausted,
}

impl GameStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Solved => "solved",
            Self::Exhausted => "exhausted",
        }
    }
}

/// Validates structural rules; vocabulary, clues and legality belong to the solver.
pub fn status(observations: &[(String, u8)], rules: GameRules) -> Result<GameStatus> {
    if rules == GameRules::Wordle && observations.len() > 6 {
        bail!("a Wordle game has at most six turns");
    }
    for (index, (guess, pattern)) in observations.iter().enumerate() {
        if guess.len() != 5 || !guess.bytes().all(|byte| byte.is_ascii_lowercase()) {
            bail!("history guess must be exactly 5 lowercase letters");
        }
        if *pattern as usize >= PATTERN_SPACE {
            bail!("history contains an invalid feedback pattern");
        }
        if index > 0 && observations[index - 1].1 == ALL_GREEN_PATTERN {
            bail!("a solved game cannot contain further turns");
        }
    }
    if observations
        .last()
        .is_some_and(|(_, pattern)| *pattern == ALL_GREEN_PATTERN)
    {
        Ok(GameStatus::Solved)
    } else if rules == GameRules::Wordle && observations.len() == 6 {
        Ok(GameStatus::Exhausted)
    } else {
        Ok(GameStatus::Active)
    }
}

/// Returns a replacement only after the entire proposed history passes validation.
/// The caller retains both committed history and its draft on any error.
pub fn try_append_observation(
    observations: &[(String, u8)],
    guess: &str,
    feedback: &str,
    rules: GameRules,
    validate: impl FnOnce(&[(String, u8)]) -> Result<()>,
) -> Result<Vec<(String, u8)>> {
    match status(observations, rules)? {
        GameStatus::Solved => {
            bail!("the game is already solved; undo or reset before adding a row")
        }
        GameStatus::Exhausted => {
            bail!("all six Wordle turns are used; undo or reset before adding a row")
        }
        GameStatus::Active => {}
    }
    let pattern = parse_feedback(feedback)?;
    let mut next = observations.to_vec();
    next.push((guess.trim().to_ascii_lowercase(), pattern));
    status(&next, rules)?;
    validate(&next)?;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordle_terminal_boundaries_and_undo_are_explicit() {
        let misses = vec![("cigar".into(), 0); 6];
        assert_eq!(
            status(&misses, GameRules::Wordle).unwrap(),
            GameStatus::Exhausted
        );
        assert_eq!(
            status(&misses[..5], GameRules::Wordle).unwrap(),
            GameStatus::Active
        );
        let mut solved = misses[..5].to_vec();
        solved.push(("rebut".into(), ALL_GREEN_PATTERN));
        assert_eq!(
            status(&solved, GameRules::Wordle).unwrap(),
            GameStatus::Solved
        );
        assert!(
            try_append_observation(&solved, "cigar", "00000", GameRules::Wordle, |_| Ok(()))
                .is_err()
        );
        assert_eq!(
            status(&solved[..5], GameRules::Wordle).unwrap(),
            GameStatus::Active
        );
    }

    #[test]
    fn absurdle_is_not_limited_to_six_turns_but_stops_after_green() {
        let misses = vec![("cigar".into(), 0); 6];
        let next =
            try_append_observation(&misses, "REBUT", "ggggg", GameRules::Absurdle, |_| Ok(()))
                .unwrap();
        assert_eq!(next.len(), 7);
        assert_eq!(next[6].0, "rebut");
        assert_eq!(
            status(&next, GameRules::Absurdle).unwrap(),
            GameStatus::Solved
        );
        assert!(
            try_append_observation(&next, "cigar", "00000", GameRules::Absurdle, |_| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn invalid_draft_or_failed_solver_validation_never_mutates_history() {
        let history = vec![("cigar".into(), 0)];
        for (guess, feedback) in [("four", "00000"), ("rebut", "00"), ("rebut", "00?00")] {
            assert!(
                try_append_observation(&history, guess, feedback, GameRules::Wordle, |_| panic!(
                    "invalid draft reached solver"
                ))
                .is_err()
            );
        }
        let error = try_append_observation(&history, "rebut", "00000", GameRules::Wordle, |_| {
            bail!("contradictory clues")
        })
        .unwrap_err();
        assert!(error.to_string().contains("contradictory clues"));
        assert_eq!(history, [("cigar".into(), 0)]);
    }

    #[test]
    fn malformed_replayed_history_is_rejected() {
        for history in [
            vec![("CIGAR".into(), 0)],
            vec![("cigar".into(), 243)],
            vec![("cigar".into(), 242), ("rebut".into(), 0)],
        ] {
            assert!(status(&history, GameRules::Absurdle).is_err());
            assert!(status(&history, GameRules::Wordle).is_err());
        }
    }
}
