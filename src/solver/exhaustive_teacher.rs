use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow, ensure};

use crate::scoring::{ALL_GREEN_PATTERN, PATTERN_SPACE};

pub(super) const CONTRACT: &str = "independent-fixed-normal-expected-guesses-v1";

#[cfg(test)]
thread_local! {
    // Deterministic cumulative-clock fault for checkpoint integration tests.
    pub(super) static EXHAUST_AFTER_STATES: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[derive(Debug)]
pub(super) struct BudgetExceeded(pub(super) &'static str);

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for BudgetExceeded {}

pub(super) struct WorkBudget {
    started: Instant,
    prior_elapsed: Duration,
    maximum_time: Duration,
    maximum_memory_bytes: Option<u64>,
    last_memory_check: Option<Instant>,
    peak_memory_bytes: Option<u64>,
}

impl WorkBudget {
    pub(super) fn new(
        started: Instant,
        prior_elapsed_ms: u64,
        maximum_time: Duration,
        maximum_memory_bytes: Option<u64>,
    ) -> Self {
        Self {
            started,
            prior_elapsed: Duration::from_millis(prior_elapsed_ms),
            maximum_time,
            maximum_memory_bytes,
            last_memory_check: None,
            peak_memory_bytes: None,
        }
    }

    pub(super) fn remaining_run_time(&self) -> Duration {
        self.maximum_time.saturating_sub(self.prior_elapsed)
    }

    pub(super) fn elapsed_ms(&self) -> u64 {
        self.prior_elapsed
            .saturating_add(self.started.elapsed())
            .as_millis()
            .min(u64::MAX as u128) as u64
    }

    pub(super) fn peak_memory_bytes(&self) -> Option<u64> {
        self.peak_memory_bytes
    }

    pub(super) fn check_now(&mut self) -> Result<()> {
        self.last_memory_check = None;
        self.check()
    }

    pub(super) fn check(&mut self) -> Result<()> {
        if self.prior_elapsed.saturating_add(self.started.elapsed()) >= self.maximum_time {
            return Err(BudgetExceeded(
                "offline evaluation exceeded its cumulative wall-clock budget",
            )
            .into());
        }
        // This is a cooperative process-level guard, not an allocator limit. Keep the
        // clock check on every unit of work, but throttle process measurements.
        if self
            .last_memory_check
            .is_none_or(|last| last.elapsed() >= Duration::from_millis(100))
        {
            if let Some(snapshot) = crate::process_memory::process_memory_snapshot() {
                let peak = self
                    .peak_memory_bytes
                    .unwrap_or(0)
                    .max(snapshot.peak_working_set_bytes);
                self.peak_memory_bytes = Some(peak);
                if let Some(maximum) = self.maximum_memory_bytes
                    && peak > maximum
                {
                    return Err(BudgetExceeded(
                        "offline evaluation process peak working set exceeded its memory budget",
                    )
                    .into());
                }
            } else if self.maximum_memory_bytes.is_some() {
                return Err(BudgetExceeded(
                    "offline evaluation cannot measure its process memory budget",
                )
                .into());
            }
            self.last_memory_check = Some(Instant::now());
        }
        Ok(())
    }
}

pub(super) fn label_root_actions(
    survivors: &[usize],
    weights: &[f64],
    guess_count: usize,
    requested_guesses: &[usize],
    maximum_labels: usize,
    feedback: impl Fn(usize, usize) -> u8,
    budget: &mut WorkBudget,
) -> Result<Vec<(usize, f64)>> {
    #[cfg(test)]
    EXHAUST_AFTER_STATES.with(|remaining| {
        if let Some(count) = remaining.get() {
            if count == 0 {
                remaining.set(None);
                budget.prior_elapsed = budget.maximum_time;
            } else {
                remaining.set(Some(count - 1));
            }
        }
    });
    if requested_guesses.len() > maximum_labels {
        return Err(BudgetExceeded(
            "exhaustive teacher root labels exceed the remaining row budget",
        )
        .into());
    }
    budget.check_now()?;
    ensure!(
        !requested_guesses.is_empty()
            && requested_guesses.iter().all(|guess| *guess < guess_count)
            && requested_guesses
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                == requested_guesses.len(),
        "exhaustive teacher requested actions must be nonempty, unique, and legal"
    );
    ensure!(
        !survivors.is_empty()
            && survivors.windows(2).all(|pair| pair[0] < pair[1])
            && survivors.iter().all(|&answer| answer < weights.len()
                && weights[answer].is_finite()
                && weights[answer] >= 0.0),
        "exhaustive teacher survivors or weights are invalid"
    );
    let mut memo = HashMap::new();
    let mut labels = Vec::with_capacity(requested_guesses.len());
    for &guess in requested_guesses {
        budget.check()?;
        let action = partition(survivors, guess, weights, &feedback, budget)?.ok_or_else(|| {
            anyhow!("exhaustive teacher requested action {guess} makes no progress")
        })?;
        let mut cost = 1.0;
        for (child, probability) in action.children {
            budget.check()?;
            cost +=
                probability * best_cost(child, weights, guess_count, &feedback, &mut memo, budget)?;
        }
        ensure!(
            cost.is_finite() && cost >= 1.0,
            "exhaustive teacher produced an invalid cost"
        );
        labels.push((guess, cost));
    }
    // A final slow feedback call or allocation must not escape the deadline/peak check.
    budget.check_now()?;
    Ok(labels)
}

struct Action {
    children: Vec<(Vec<usize>, f64)>,
    next_child: usize,
    cost: f64,
}

fn partition(
    survivors: &[usize],
    guess: usize,
    weights: &[f64],
    feedback: &impl Fn(usize, usize) -> u8,
    budget: &mut WorkBudget,
) -> Result<Option<Action>> {
    let mut branches = BTreeMap::<u8, Vec<usize>>::new();
    let mut total = 0.0;
    for &answer in survivors {
        budget.check()?;
        let pattern = feedback(guess, answer);
        budget.check()?;
        ensure!(
            usize::from(pattern) < PATTERN_SPACE,
            "exhaustive teacher feedback is invalid"
        );
        total += weights[answer];
        if pattern != ALL_GREEN_PATTERN && weights[answer] > 0.0 {
            branches.entry(pattern).or_default().push(answer);
        }
    }
    ensure!(
        total.is_finite() && total > 0.0,
        "exhaustive teacher state mass is invalid"
    );
    let mut children = Vec::with_capacity(branches.len());
    for child in branches.into_values() {
        budget.check()?;
        let mass: f64 = child.iter().map(|&answer| weights[answer]).sum();
        // An unchanged non-solving subset can only add cost. Strict subsets
        // still progress even when the discarded answers have zero weight.
        if child.len() == survivors.len() {
            return Ok(None);
        }
        children.push((child, mass / total));
    }
    Ok(Some(Action {
        children,
        next_child: 0,
        cost: 1.0,
    }))
}

struct StateFrame {
    survivors: Vec<usize>,
    next_guess: usize,
    best: f64,
    action: Option<Action>,
}

impl StateFrame {
    fn new(survivors: Vec<usize>) -> Self {
        Self {
            survivors,
            next_guess: 0,
            best: f64::INFINITY,
            action: None,
        }
    }
}

fn best_cost(
    survivors: Vec<usize>,
    weights: &[f64],
    guess_count: usize,
    feedback: &impl Fn(usize, usize) -> u8,
    memo: &mut HashMap<Vec<usize>, f64>,
    budget: &mut WorkBudget,
) -> Result<f64> {
    if let Some(&cost) = memo.get(&survivors) {
        return Ok(cost);
    }
    // Heap frames avoid a call-stack limit. Only reached subsets are memoized;
    // no whole graph, production shortlist, or production exact-search code is used.
    let mut stack = vec![StateFrame::new(survivors)];
    loop {
        budget.check()?;
        let frame = stack.last_mut().expect("active teacher frame");
        if let Some(action) = &mut frame.action {
            if let Some((child, probability)) = action.children.get(action.next_child) {
                if let Some(&cost) = memo.get(child) {
                    action.cost += probability * cost;
                    action.next_child += 1;
                } else {
                    let child = child.clone();
                    stack.push(StateFrame::new(child));
                }
            } else {
                frame.best = frame.best.min(action.cost);
                frame.action = None;
                frame.next_guess += 1;
            }
        } else if frame.next_guess == guess_count {
            ensure!(
                frame.best.is_finite() && frame.best >= 1.0,
                "exhaustive teacher state has no finite solving action"
            );
            budget.check()?;
            let finished = stack.pop().expect("finished teacher frame");
            let cost = finished.best;
            memo.insert(finished.survivors, cost);
            if stack.is_empty() {
                return Ok(cost);
            }
        } else {
            frame.action = partition(
                &frame.survivors,
                frame.next_guess,
                weights,
                feedback,
                budget,
            )?;
            if frame.action.is_none() {
                frame.next_guess += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn budget() -> WorkBudget {
        WorkBudget::new(Instant::now(), 0, Duration::from_secs(5), None)
    }

    // Separate duplicate-letter scorer, using consumed answer positions rather than counts.
    fn feedback(guess: &str, answer: &str) -> u8 {
        let guess = guess.as_bytes();
        let answer = answer.as_bytes();
        let mut consumed = [false; 5];
        let mut marks = [0_u8; 5];
        for position in 0..5 {
            if guess[position] == answer[position] {
                consumed[position] = true;
                marks[position] = 2;
            }
        }
        for position in 0..5 {
            if marks[position] != 0 {
                continue;
            }
            if let Some(found) =
                (0..5).find(|&other| !consumed[other] && guess[position] == answer[other])
            {
                consumed[found] = true;
                marks[position] = 1;
            }
        }
        marks.iter().rev().fold(0, |value, mark| value * 3 + mark)
    }

    fn reference_action(
        mask: usize,
        guess: usize,
        words: &[&str],
        answers: &[&str],
        weights: &[f64],
        memo: &mut [Option<f64>],
    ) -> f64 {
        let mut branches = [0_usize; 243];
        let mut total = 0.0;
        for (answer, word) in answers.iter().enumerate() {
            if mask & (1 << answer) != 0 {
                branches[feedback(words[guess], word) as usize] |= 1 << answer;
                total += weights[answer];
            }
        }
        let mut result = 1.0;
        for child in branches[..242].iter().copied().filter(|child| *child != 0) {
            if child == mask {
                return f64::INFINITY;
            }
            let mass: f64 = weights
                .iter()
                .enumerate()
                .filter(|(index, _)| child & (1 << index) != 0)
                .map(|(_, weight)| weight)
                .sum();
            let cost = if let Some(cost) = memo[child] {
                cost
            } else {
                let cost = (0..words.len())
                    .map(|guess| reference_action(child, guess, words, answers, weights, memo))
                    .fold(f64::INFINITY, f64::min);
                memo[child] = Some(cost);
                cost
            };
            result += mass / total * cost;
        }
        result
    }

    #[test]
    fn labels_match_independent_weighted_subset_oracle() {
        let answers = [
            "tower", "power", "bower", "rower", "sware", "crare", "beare", "urare", "blare",
        ];
        let mut guesses = answers.to_vec();
        guesses.push("mesne");
        let weights = [43.0, 43.0, 19.0, 4.0, 4.0, 5.0, 6.0, 6.0, 2.0];
        let survivors = (0..answers.len()).collect::<Vec<_>>();
        let requested = (0..guesses.len()).collect::<Vec<_>>();
        let labels = label_root_actions(
            &survivors,
            &weights,
            guesses.len(),
            &requested,
            requested.len(),
            |guess, answer| feedback(guesses[guess], answers[answer]),
            &mut budget(),
        )
        .unwrap();
        let mut memo = vec![None; 1 << answers.len()];
        for (guess, cost) in labels {
            let expected = reference_action(
                (1 << answers.len()) - 1,
                guess,
                &guesses,
                &answers,
                &weights,
                &mut memo,
            );
            assert!(
                (cost - expected).abs() < 1e-12,
                "{}: {cost} != {expected}",
                guesses[guess]
            );
        }
    }

    #[test]
    fn labels_preserve_repeated_letter_feedback_and_zero_mass_branches() {
        let words = ["allee", "llama", "apple", "cigar"];
        for guess in words {
            for answer in words {
                assert_eq!(
                    feedback(guess, answer),
                    crate::scoring::score_guess(guess, answer)
                );
            }
        }
        let labels = label_root_actions(
            &[0, 1],
            &[1.0, 0.0],
            2,
            &[0, 1],
            2,
            |guess, answer| feedback(words[guess], words[answer]),
            &mut budget(),
        )
        .unwrap();
        assert_eq!(labels, vec![(0, 1.0), (1, 2.0)]);
    }

    #[test]
    fn row_limit_rejects_before_any_teacher_feedback() {
        let calls = Cell::new(0);
        let error = label_root_actions(
            &[0, 1],
            &[1.0, 1.0],
            2,
            &[0, 1],
            1,
            |_, _| {
                calls.set(calls.get() + 1);
                242
            },
            &mut budget(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("row budget"));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn cumulative_deadline_rejects_before_work_and_after_slow_feedback() {
        let calls = Cell::new(0);
        let mut expired = WorkBudget::new(Instant::now(), 10, Duration::from_millis(1), None);
        let error = label_root_actions(
            &[0],
            &[1.0],
            1,
            &[0],
            1,
            |_, _| {
                calls.set(calls.get() + 1);
                242
            },
            &mut expired,
        )
        .unwrap_err();
        assert!(error.to_string().contains("wall-clock budget"));
        assert_eq!(calls.get(), 0);
        let mut slow = WorkBudget::new(Instant::now(), 0, Duration::from_millis(1), None);
        let error = label_root_actions(
            &[0],
            &[1.0],
            1,
            &[0],
            1,
            |_, _| {
                std::thread::sleep(Duration::from_millis(5));
                242
            },
            &mut slow,
        )
        .unwrap_err();
        assert!(error.to_string().contains("wall-clock budget"));
    }

    #[test]
    fn deadline_interrupts_a_single_descendant_before_more_actions_or_labels() {
        let calls = Cell::new(0);
        let mut limited = WorkBudget::new(Instant::now(), 0, Duration::from_millis(100), None);
        let error = label_root_actions(
            &[0, 1],
            &[1.0, 1.0],
            2,
            &[0],
            1,
            |guess, answer| {
                let count = calls.get() + 1;
                calls.set(count);
                if count == 3 {
                    std::thread::sleep(Duration::from_millis(150));
                }
                if guess == answer { 242 } else { 0 }
            },
            &mut limited,
        )
        .unwrap_err();
        assert!(error.to_string().contains("wall-clock budget"));
        assert_eq!(
            calls.get(),
            3,
            "must stop inside the first descendant action"
        );
    }

    #[test]
    fn impossible_memory_limit_rejects_before_work() {
        let mut limited = WorkBudget::new(Instant::now(), 0, Duration::from_secs(5), Some(1));
        let error = label_root_actions(
            &[0],
            &[1.0],
            1,
            &[0],
            1,
            |_, _| panic!("memory guard must precede feedback"),
            &mut limited,
        )
        .unwrap_err();
        assert!(error.to_string().contains("memory"));
    }

    #[test]
    fn nonprogress_or_invalid_requested_actions_are_explicit_errors() {
        for requested in [vec![0], vec![0, 0], vec![2], vec![]] {
            assert!(
                label_root_actions(
                    &[0, 1],
                    &[1.0, 1.0],
                    2,
                    &requested,
                    10,
                    |_, _| 0,
                    &mut budget()
                )
                .is_err()
            );
        }
        assert!(
            label_root_actions(
                &[0, 1],
                &[1.0, f64::NAN],
                2,
                &[0],
                1,
                |_, _| 242,
                &mut budget()
            )
            .is_err()
        );
    }
}
