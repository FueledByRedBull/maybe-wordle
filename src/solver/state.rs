use crate::predictive::PredictiveArtifactState;

use super::*;

impl Solver {
    pub fn today() -> NaiveDate {
        Utc::now().date_naive()
    }

    pub fn initial_state(&self, as_of: NaiveDate) -> SolveState {
        let recovery_policy = &self.config.recovery;
        let mut modeled_weights = vec![0.0; self.answers.len()];
        let mut recovery_weights = vec![0.0; self.answers.len()];
        let mut weights = vec![0.0; self.answers.len()];
        let mut positive_survivors = Vec::new();
        let mut recovery_survivors = Vec::new();
        let mut modeled_total_weight = 0.0;
        let mut total_weight = 0.0;

        for (index, answer) in self.answers.iter().enumerate() {
            let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
            let modeled_weight = snapshot.final_weight.max(0.0);
            if snapshot.base_weight > 0.0 {
                recovery_weights[index] = snapshot.base_weight * snapshot.manual_weight;
                modeled_weights[index] = modeled_weight;
                if modeled_weight > 0.0 {
                    modeled_total_weight += modeled_weight;
                    total_weight += modeled_weight;
                    weights[index] = modeled_weight;
                    positive_survivors.push(index);
                } else {
                    recovery_survivors.push(index);
                }
            }
        }

        let (surviving, recovery_mode_used) = if modeled_total_weight > 0.0 {
            (positive_survivors, None)
        } else {
            let support_count = recovery_survivors.len();
            match self.config.recovery.mode {
                RecoveryMode::Strict => {
                    for index in &recovery_survivors {
                        weights[*index] = 0.0;
                    }
                    total_weight = 0.0;
                    (recovery_survivors, None)
                }
                mode => {
                    for index in &recovery_survivors {
                        weights[*index] =
                            recovery_policy.repair_weight(recovery_weights[*index], support_count);
                    }
                    total_weight = recovery_survivors
                        .iter()
                        .map(|index| weights[*index])
                        .sum::<f64>();
                    (recovery_survivors, Some(mode))
                }
            }
        };

        SolveState {
            surviving,
            modeled_weights,
            recovery_weights,
            weights,
            modeled_total_weight,
            total_weight,
            recovery_mode_used,
        }
    }

    pub fn apply_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
    ) -> Result<SolveState> {
        let mut state = self.initial_state(as_of);
        for (guess, pattern) in observations {
            self.apply_feedback(&mut state, guess, *pattern)?;
        }
        Ok(state)
    }

    pub fn absurdle_initial_state(&self) -> SolveState {
        SolveState {
            surviving: (0..self.answers.len()).collect(),
            modeled_weights: vec![1.0; self.answers.len()],
            recovery_weights: vec![1.0; self.answers.len()],
            weights: vec![1.0; self.answers.len()],
            modeled_total_weight: self.answers.len() as f64,
            total_weight: self.answers.len() as f64,
            recovery_mode_used: None,
        }
    }

    pub fn absurdle_apply_history(&self, observations: &[(String, u8)]) -> Result<SolveState> {
        let mut state = self.absurdle_initial_state();
        for (guess, pattern) in observations {
            self.apply_feedback(&mut state, guess, *pattern)?;
        }
        Ok(state)
    }

    pub fn apply_feedback(&self, state: &mut SolveState, guess: &str, pattern: u8) -> Result<()> {
        let guess_index = self
            .guess_index
            .get(&guess.to_ascii_lowercase())
            .copied()
            .ok_or_else(|| anyhow!("unknown guess: {}", guess))?;
        state
            .surviving
            .retain(|answer_index| self.pattern_table.get(guess_index, *answer_index) == pattern);
        state.modeled_total_weight = state
            .surviving
            .iter()
            .map(|index| state.modeled_weights[*index])
            .sum::<f64>();
        if state.modeled_total_weight > 0.0 && state.recovery_mode_used.is_none() {
            for index in &state.surviving {
                state.weights[*index] = state.modeled_weights[*index];
            }
            state.recovery_mode_used = None;
            state.total_weight = state.modeled_total_weight;
        } else {
            state.recovery_mode_used = match self.config.recovery.mode {
                RecoveryMode::Strict => {
                    for index in &state.surviving {
                        state.weights[*index] = 0.0;
                    }
                    None
                }
                mode => {
                    for index in &state.surviving {
                        state.weights[*index] = self
                            .config
                            .recovery
                            .repair_weight(state.recovery_weights[*index], state.surviving.len());
                    }
                    Some(mode)
                }
            };
            state.total_weight = state
                .surviving
                .iter()
                .map(|index| state.weights[*index])
                .sum::<f64>();
        }

        if state.surviving.is_empty() {
            bail!(
                "no answers remain after applying {} {}",
                guess,
                format_feedback_letters(pattern)
            );
        }
        if state.total_weight <= 0.0 && matches!(self.config.recovery.mode, RecoveryMode::Strict) {
            bail!(
                "no positive answer mass remains after applying {} {}",
                guess,
                format_feedback_letters(pattern)
            );
        }
        Ok(())
    }

    pub fn suggestions(&self, state: &SolveState, top: usize) -> Result<Vec<Suggestion>> {
        Ok(self
            .suggestion_batch_internal(state, top, None, PredictiveBookUsage::None)?
            .suggestions)
    }

    pub fn suggest_predictive(
        &self,
        request: PredictiveSuggestRequest<'_>,
    ) -> Result<PredictiveSuggestResponse> {
        let state = self.apply_history(request.as_of, request.observations)?;
        let suggestions = if request.hard_mode || request.force_in_two_only {
            self.filtered_suggestion_batch_for_history(
                request.as_of,
                request.observations,
                request.top,
                request.mode,
                request.hard_mode,
                request.force_in_two_only,
            )?
        } else {
            self.suggestion_batch_internal(
                &state,
                request.top,
                Some(PredictiveContext {
                    as_of: request.as_of,
                    observations: request.observations,
                }),
                book_usage_for_mode(request.mode),
            )?
        };

        Ok(PredictiveSuggestResponse {
            state: PredictiveStateSummary {
                surviving: state.surviving.len(),
                modeled_total_weight: state.modeled_total_weight,
                effective_total_weight: state.total_weight,
                recovery_mode_used: state.recovery_mode_used,
            },
            suggestions: suggestions.suggestions,
            promoted_word: suggestions.promoted_word,
            promotion_source: suggestions.promotion_source,
            artifact_state: PredictiveArtifactState::from_promotion_source(
                suggestions.promotion_source,
            ),
        })
    }

    pub fn suggestions_for_history(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        Ok(self
            .suggest_predictive(PredictiveSuggestRequest {
                as_of,
                observations,
                top,
                hard_mode: false,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::Full,
            })?
            .suggestions)
    }

    pub fn suggestions_for_history_hard_mode(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        Ok(self
            .suggest_predictive(PredictiveSuggestRequest {
                as_of,
                observations,
                top,
                hard_mode: true,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::Full,
            })?
            .suggestions)
    }

    pub fn suggestions_for_history_disk_books_only(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        Ok(self
            .suggest_predictive(PredictiveSuggestRequest {
                as_of,
                observations,
                top,
                hard_mode: false,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::FastDiskOnly,
            })?
            .suggestions)
    }

    pub fn suggestions_for_history_disk_books_only_with_filters(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
        hard_mode: bool,
        force_in_two_only: bool,
    ) -> Result<Vec<Suggestion>> {
        Ok(self
            .suggest_predictive(PredictiveSuggestRequest {
                as_of,
                observations,
                top,
                hard_mode,
                force_in_two_only,
                mode: PredictiveSuggestionMode::FastDiskOnly,
            })?
            .suggestions)
    }

    pub fn force_in_two_suggestions_for_history_disk_books_only(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<Suggestion>> {
        self.suggestions_for_history_disk_books_only_with_filters(
            as_of,
            observations,
            top,
            false,
            true,
        )
    }

    pub fn absurdle_suggestions(
        &self,
        observations: &[(String, u8)],
        top: usize,
    ) -> Result<Vec<AbsurdleSuggestion>> {
        let state = self.absurdle_apply_history(observations)?;
        self.absurdle_suggestions_for_state(&state, top)
    }

    pub fn absurdle_suggestions_for_state(
        &self,
        state: &SolveState,
        top: usize,
    ) -> Result<Vec<AbsurdleSuggestion>> {
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        let total = state.surviving.len() as f64;
        let mut suggestions = (0..self.guesses.len())
            .into_par_iter()
            .map(|guess_index| self.absurdle_score_guess(guess_index, &state.surviving, total))
            .collect::<Vec<_>>();
        suggestions.sort_by(compare_absurdle_suggestions);
        suggestions.truncate(top.min(suggestions.len()));
        Ok(suggestions)
    }

    pub fn hard_mode_violation(
        &self,
        observations: &[(String, u8)],
        guess: &str,
    ) -> Option<String> {
        hard_mode_violation_message(observations, guess)
    }
}

pub(super) fn hard_mode_violation_message(
    observations: &[(String, u8)],
    guess: &str,
) -> Option<String> {
    if observations.is_empty() {
        return None;
    }
    if guess.len() != HARD_MODE_WORD_LENGTH || !guess.bytes().all(|byte| byte.is_ascii_lowercase())
    {
        return Some("hard mode guess must be exactly 5 lowercase letters".to_string());
    }

    let constraints = build_hard_mode_constraints(observations);
    let guess_bytes = guess.as_bytes();
    let mut guess_counts = [0u8; 26];
    for (index, &byte) in guess_bytes.iter().enumerate() {
        let letter_index = (byte - b'a') as usize;
        guess_counts[letter_index] += 1;

        if let Some(expected) = constraints.greens[index]
            && byte != expected
        {
            return Some(format!(
                "hard mode requires {} in position {}",
                char::from(expected).to_ascii_uppercase(),
                index + 1
            ));
        }

        if (constraints.yellow_forbidden[index] & (1u32 << letter_index)) != 0 {
            return Some(format!(
                "hard mode forbids {} in position {}",
                char::from(byte).to_ascii_uppercase(),
                index + 1
            ));
        }
    }

    for (letter_index, &required) in constraints.required_counts.iter().enumerate() {
        if required > 0 && guess_counts[letter_index] < required {
            let letter = char::from(b'a' + letter_index as u8).to_ascii_uppercase();
            return Some(format!(
                "hard mode requires {} occurrence{} of {}",
                required,
                if required == 1 { "" } else { "s" },
                letter
            ));
        }
    }

    None
}

struct HardModeConstraints {
    greens: [Option<u8>; HARD_MODE_WORD_LENGTH],
    yellow_forbidden: [u32; HARD_MODE_WORD_LENGTH],
    required_counts: [u8; 26],
}

fn build_hard_mode_constraints(observations: &[(String, u8)]) -> HardModeConstraints {
    let mut constraints = HardModeConstraints {
        greens: [None; HARD_MODE_WORD_LENGTH],
        yellow_forbidden: [0; HARD_MODE_WORD_LENGTH],
        required_counts: [0; 26],
    };

    for (guess, pattern) in observations {
        let feedback = decode_feedback(*pattern);
        let guess_bytes = guess.as_bytes();
        let mut positive_counts = [0u8; 26];

        for index in 0..HARD_MODE_WORD_LENGTH {
            let byte = guess_bytes[index];
            let letter_index = (byte - b'a') as usize;
            match feedback[index] {
                2 => {
                    constraints.greens[index] = Some(byte);
                    positive_counts[letter_index] += 1;
                }
                1 => {
                    constraints.yellow_forbidden[index] |= 1u32 << letter_index;
                    positive_counts[letter_index] += 1;
                }
                _ => {}
            }
        }

        for (letter_index, &count) in positive_counts.iter().enumerate() {
            constraints.required_counts[letter_index] =
                constraints.required_counts[letter_index].max(count);
        }
    }

    constraints
}
