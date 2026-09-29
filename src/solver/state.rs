use crate::predictive::PredictiveArtifactState;

use super::search::check_predictive_search_cancelled;
use super::*;

impl Solver {
    pub fn today() -> NaiveDate {
        chrono::Local::now().date_naive()
    }

    pub fn initial_state(&self, as_of: NaiveDate) -> SolveState {
        if self.config.search_policy_mode.uses_fixed_belief() {
            return self
                .fixed_posterior_state(as_of)
                .expect("validated finite posterior must contain positive support");
        }
        self.initial_state_with_modeled_weights(as_of, None)
            .expect("default weight snapshots must construct a valid initial state")
    }

    /// Freeze one core/tail distribution; feedback only conditions these weights.
    pub fn fixed_posterior_state(&self, as_of: NaiveDate) -> Result<SolveState> {
        self.freeze_core_tail(self.initial_state_with_modeled_weights(as_of, None)?)
    }

    fn freeze_core_tail(&self, mut state: SolveState) -> Result<SolveState> {
        if state.condition_only {
            return Ok(state);
        }
        let eligible = state
            .surviving
            .iter()
            .chain(&state.fallback_surviving)
            .copied()
            .collect::<Vec<_>>();
        let core_total = eligible
            .iter()
            .map(|index| state.modeled_weights[*index])
            .sum::<f64>();
        let tail_count = eligible
            .iter()
            .filter(|index| state.modeled_weights[**index] <= 0.0)
            .count();
        let tail_mass = if core_total <= 0.0 {
            1.0
        } else if tail_count == 0 {
            0.0
        } else {
            self.config.fallback_prior_mass
        };
        if !tail_mass.is_finite() || !(0.0..=1.0).contains(&tail_mass) {
            bail!("fallback prior mass must be a finite probability");
        }
        for index in &eligible {
            state.weights[*index] = if state.modeled_weights[*index] > 0.0 {
                (1.0 - tail_mass) * state.modeled_weights[*index] / core_total
            } else if tail_count > 0 {
                tail_mass / tail_count as f64
            } else {
                0.0
            };
        }
        state.surviving = eligible
            .into_iter()
            .filter(|index| state.weights[*index] > 0.0)
            .collect();
        state.surviving.sort_unstable();
        state.fallback_surviving.clear();
        state.fallback_active = tail_count > 0 && tail_mass > 0.0;
        state.recovery_mode_used = None;
        state.condition_only = true;
        state.total_weight = state
            .surviving
            .iter()
            .map(|index| state.weights[*index])
            .sum();
        if state.total_weight <= 0.0 {
            bail!("fixed posterior has no positive answer mass");
        }
        Ok(state)
    }

    pub(super) fn initial_state_with_modeled_weights(
        &self,
        as_of: NaiveDate,
        modeled_weight_override: Option<&[f64]>,
    ) -> Result<SolveState> {
        if let Some(overrides) = modeled_weight_override {
            if overrides.len() != self.primary_answer_count {
                bail!(
                    "modeled-weight override has {} values; expected {}",
                    overrides.len(),
                    self.primary_answer_count
                );
            }
            if overrides
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            {
                bail!("modeled-weight override contains a non-finite or negative value");
            }
        }
        let recovery_policy = &self.config.recovery;
        let mut modeled_weights = vec![0.0; self.answers.len()];
        let mut recovery_weights = vec![0.0; self.answers.len()];
        let mut weights = vec![0.0; self.answers.len()];
        let mut supported_survivors = Vec::new();
        let mut fallback_surviving =
            (self.primary_answer_count..self.answers.len()).collect::<Vec<_>>();
        let mut modeled_total_weight = 0.0;
        let mut total_weight = 0.0;

        for (index, answer) in self
            .answers
            .iter()
            .take(self.primary_answer_count)
            .enumerate()
        {
            let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
            let modeled_weight = modeled_weight_override
                .map_or(snapshot.final_weight, |weights| weights[index])
                .max(0.0);
            if snapshot.base_weight > 0.0 || answer.in_seed || answer.manual_entry {
                supported_survivors.push(index);
                recovery_weights[index] = snapshot.base_weight * snapshot.manual_weight;
                modeled_weights[index] = modeled_weight;
                if modeled_weight > 0.0 {
                    modeled_total_weight += modeled_weight;
                    total_weight += modeled_weight;
                    weights[index] = modeled_weight;
                }
            } else if self.guess_index.contains_key(&answer.word) {
                fallback_surviving.push(index);
                recovery_weights[index] = 1.0;
            }
        }
        fallback_surviving.sort_unstable();
        let fallback_weight = if fallback_surviving.is_empty() {
            0.0
        } else {
            let prior_mass = self.config.fallback_prior_mass.clamp(0.0, 0.999_999);
            modeled_total_weight * (prior_mass / (1.0 - prior_mass))
                / fallback_surviving.len() as f64
        };
        for index in &fallback_surviving {
            recovery_weights[*index] = fallback_weight;
        }

        let (surviving, recovery_mode_used) = if modeled_total_weight > 0.0 {
            (supported_survivors, None)
        } else {
            let support_count = supported_survivors.len();
            match self.config.recovery.mode {
                RecoveryMode::Strict => {
                    for index in &supported_survivors {
                        weights[*index] = 0.0;
                    }
                    total_weight = 0.0;
                    (supported_survivors, None)
                }
                mode => {
                    for index in &supported_survivors {
                        weights[*index] =
                            recovery_policy.repair_weight(recovery_weights[*index], support_count);
                    }
                    total_weight = supported_survivors
                        .iter()
                        .map(|index| weights[*index])
                        .sum::<f64>();
                    (supported_survivors, Some(mode))
                }
            }
        };

        let state = SolveState {
            condition_only: false,
            surviving,
            fallback_surviving,
            fallback_active: false,
            modeled_weights,
            recovery_weights,
            weights,
            modeled_total_weight,
            total_weight,
            recovery_mode_used,
        };
        if self.config.search_policy_mode.uses_fixed_belief() {
            self.freeze_core_tail(state)
        } else {
            Ok(state)
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

    /// Validates a proposed live history without ranking guesses or mutating callers.
    pub fn validate_game_history(
        &self,
        puzzle_date: NaiveDate,
        observations: &[(String, u8)],
        hard_mode: bool,
    ) -> Result<SolveState> {
        validate_predictive_history(observations, hard_mode)?;
        self.apply_history(
            crate::predictive::history_cutoff(puzzle_date)?,
            observations,
        )
    }

    pub fn absurdle_initial_state(&self) -> SolveState {
        SolveState {
            condition_only: false,
            surviving: (0..self.primary_answer_count).collect(),
            fallback_surviving: Vec::new(),
            fallback_active: false,
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
        if pattern as usize >= PATTERN_SPACE {
            bail!("feedback pattern must be in 0..243");
        }
        let guess_index = self
            .guess_index
            .get(&guess.to_ascii_lowercase())
            .copied()
            .ok_or_else(|| anyhow!("unknown guess: {}", guess))?;
        state
            .surviving
            .retain(|answer_index| self.answer_pattern(guess_index, *answer_index) == pattern);
        state
            .fallback_surviving
            .retain(|answer_index| self.answer_pattern(guess_index, *answer_index) == pattern);
        if state.condition_only {
            state.modeled_total_weight = state
                .surviving
                .iter()
                .map(|index| state.modeled_weights[*index])
                .sum();
            state.total_weight = state
                .surviving
                .iter()
                .map(|index| state.weights[*index])
                .sum();
            if state.total_weight <= 0.0 {
                bail!(
                    "no positive answer mass remains after applying {} {}",
                    guess,
                    format_feedback_letters(pattern)
                );
            }
            return Ok(());
        }
        if state.surviving.is_empty() && !state.fallback_surviving.is_empty() {
            state.surviving = std::mem::take(&mut state.fallback_surviving);
            state.fallback_active = true;
        } else if self.config.fallback_activation_threshold > 0
            && state.surviving.len() <= self.config.fallback_activation_threshold
            && !state.fallback_surviving.is_empty()
        {
            state
                .surviving
                .extend(std::mem::take(&mut state.fallback_surviving));
            state.surviving.sort_unstable();
            state.fallback_active = true;
        }
        state.modeled_total_weight = state
            .surviving
            .iter()
            .map(|index| state.modeled_weights[*index])
            .sum::<f64>();
        if state.modeled_total_weight > 0.0 {
            for index in &state.surviving {
                state.weights[*index] = if state.modeled_weights[*index] > 0.0 {
                    state.modeled_weights[*index]
                } else if state.fallback_active {
                    state.recovery_weights[*index]
                } else {
                    0.0
                };
            }
            state.recovery_mode_used = None;
            state.total_weight = state
                .surviving
                .iter()
                .map(|index| state.weights[*index])
                .sum::<f64>();
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
        self.suggest_predictive_with_search_mode(request, None)
    }

    pub fn suggest_predictive_proxy_preview(
        &self,
        request: PredictiveSuggestRequest<'_>,
    ) -> Result<PredictiveSuggestResponse> {
        self.suggest_predictive_with_search_mode(request, Some(PredictiveSearchMode::ProxyOnly))
    }

    /// Run the configured predictive policy with cooperative cancellation.
    /// Cancellation is observed at search boundaries, not a hard wall-clock deadline.
    /// Full session-book construction is excluded; use LiveOnly or FastDiskOnly.
    pub fn suggest_predictive_cancellable(
        &self,
        request: PredictiveSuggestRequest<'_>,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<PredictiveSuggestResponse> {
        self.suggest_predictive_with_search_mode_controlled(request, None, cancelled, false)
    }

    fn suggest_predictive_with_search_mode(
        &self,
        request: PredictiveSuggestRequest<'_>,
        forced_search_mode: Option<PredictiveSearchMode>,
    ) -> Result<PredictiveSuggestResponse> {
        self.suggest_predictive_with_search_mode_controlled(
            request,
            forced_search_mode,
            &|| false,
            true,
        )
    }

    fn suggest_predictive_with_search_mode_controlled(
        &self,
        request: PredictiveSuggestRequest<'_>,
        forced_search_mode: Option<PredictiveSearchMode>,
        cancelled: &(dyn Fn() -> bool + Sync),
        allow_full_book: bool,
    ) -> Result<PredictiveSuggestResponse> {
        check_predictive_search_cancelled(cancelled)?;
        if !allow_full_book && request.mode == PredictiveSuggestionMode::Full {
            bail!(
                "controlled predictive search does not allow the full session book; use FastDiskOnly"
            );
        }
        if self.config.search_policy_mode.is_finite() {
            let mut options = self.finite_search_options();
            if forced_search_mode == Some(PredictiveSearchMode::ProxyOnly) {
                options.budget = std::time::Duration::from_millis(30);
            }
            return self.suggest_predictive_controlled(request, options, cancelled);
        }
        validate_predictive_history(request.observations, request.hard_mode)?;
        let as_of = crate::predictive::history_cutoff(request.puzzle_date)?;
        check_predictive_search_cancelled(cancelled)?;
        let state = self.apply_history(as_of, request.observations)?;
        check_predictive_search_cancelled(cancelled)?;
        let suggestions = if request.hard_mode || request.force_in_two_only {
            let filters = PredictiveSuggestionFilters {
                mode: request.mode,
                hard_mode: request.hard_mode,
                force_in_two_only: request.force_in_two_only,
                forced_search_mode,
            };
            self.filtered_suggestion_batch_for_history_with_search_mode_controlled(
                as_of,
                request.observations,
                request.top,
                filters,
                cancelled,
            )?
        } else {
            let context = Some(PredictiveContext {
                hard_mode: request.hard_mode,
                as_of,
                observations: request.observations,
            });
            self.suggestion_batch_internal_with_search_mode_controlled(
                &state,
                request.top,
                context,
                book_usage_for_mode(request.mode),
                forced_search_mode,
                cancelled,
            )?
        };

        self.predictive_response(request, state, suggestions)
    }

    pub fn suggest_predictive_controlled(
        &self,
        request: PredictiveSuggestRequest<'_>,
        options: FiniteSearchOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<PredictiveSuggestResponse> {
        validate_predictive_history(request.observations, request.hard_mode)?;
        let as_of = crate::predictive::history_cutoff(request.puzzle_date)?;
        let mut state = if !self.config.search_policy_mode.uses_fixed_belief() {
            self.initial_state_with_modeled_weights(as_of, None)?
        } else {
            self.fixed_posterior_state(as_of)?
        };
        for (guess, pattern) in request.observations {
            self.apply_feedback(&mut state, guess, *pattern)?;
        }
        if state.total_weight <= 0.0 {
            bail!("cannot score guesses when no positive answer mass remains");
        }
        let limit = if request.force_in_two_only {
            self.guesses.len()
        } else {
            request.top
        };
        let mut batch = self.finite_suggestion_batch(
            &state,
            limit,
            Some(PredictiveContext {
                as_of,
                observations: request.observations,
                hard_mode: request.hard_mode,
            }),
            options,
            cancelled,
        )?;
        if request.force_in_two_only {
            batch
                .suggestions
                .retain(|suggestion| suggestion.force_in_two);
            batch.execution.root_selection_optimal = false;
        }
        batch
            .suggestions
            .truncate(request.top.min(batch.suggestions.len()));
        batch.execution.selected_value_kind = batch.suggestions.first().map(|row| row.value_kind);
        let dynamic_belief = !state.condition_only;
        let mut response = self.predictive_response(request, state, batch)?;
        let mut identity = crate::identity::CanonicalSha256::new("maybe-wordle-finite-policy-v4");
        identity
            .field(response.model_manifest_hash.as_bytes())
            .field(&options.root_shortlist.to_le_bytes())
            .field(&options.reply_shortlist.to_le_bytes())
            .field(&options.exact_state_threshold.to_le_bytes())
            .field(&options.budget.as_nanos().to_le_bytes())
            .field(&[
                options.node_limit.is_some() as u8,
                request.hard_mode as u8,
                options.baseline_only as u8,
                dynamic_belief as u8,
            ])
            .field(&options.node_limit.unwrap_or_default().to_le_bytes());
        response.model_version = if dynamic_belief {
            "predictive-finite-dynamic-v1"
        } else {
            "predictive-finite-v1"
        }
        .to_string();
        response.model_manifest_hash = identity.finish_hex();
        Ok(response)
    }

    fn predictive_response(
        &self,
        request: PredictiveSuggestRequest<'_>,
        state: SolveState,
        suggestions: SuggestionBatch,
    ) -> Result<PredictiveSuggestResponse> {
        let as_of = crate::predictive::history_cutoff(request.puzzle_date)?;
        let identity = self.predictive_book_identity(as_of);
        let (history_snapshot_date, history_snapshot_hash) =
            self.predictive_history_snapshot(as_of);
        let mut candidates = state
            .surviving
            .iter()
            .map(|index| PredictiveCandidateSummary {
                word: self.answers[*index].word.clone(),
                probability: if state.total_weight > 0.0 {
                    state.weights[*index] / state.total_weight
                } else {
                    0.0
                },
                modeled_weight: state.modeled_weights[*index],
                fallback_support: state.modeled_weights[*index] <= 0.0,
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .probability
                .total_cmp(&left.probability)
                .then_with(|| left.word.cmp(&right.word))
        });
        Ok(PredictiveSuggestResponse {
            execution: suggestions.execution,
            finite_search: suggestions.finite_search,
            puzzle_date: request.puzzle_date,
            history_cutoff: as_of,
            state: PredictiveStateSummary {
                surviving: state.surviving.len(),
                modeled_total_weight: state.modeled_total_weight,
                effective_total_weight: state.total_weight,
                recovery_mode_used: state.recovery_mode_used,
            },
            suggestions: suggestions.suggestions,
            candidates,
            promoted_word: suggestions.promoted_word,
            promotion_source: suggestions.promotion_source,
            promoted_artifact_date: suggestions.promoted_artifact_date,
            artifact_state: PredictiveArtifactState::from_promotion_source(
                suggestions.promotion_source,
            ),
            model_version: identity.policy_id,
            model_manifest_hash: identity.model_manifest_hash,
            history_snapshot_date,
            history_snapshot_hash,
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
                puzzle_date: as_of
                    .succ_opt()
                    .ok_or_else(|| anyhow!("history cutoff has no following puzzle date"))?,
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
                puzzle_date: as_of
                    .succ_opt()
                    .ok_or_else(|| anyhow!("history cutoff has no following puzzle date"))?,
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
                puzzle_date: as_of
                    .succ_opt()
                    .ok_or_else(|| anyhow!("history cutoff has no following puzzle date"))?,
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
                puzzle_date: as_of
                    .succ_opt()
                    .ok_or_else(|| anyhow!("history cutoff has no following puzzle date"))?,
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
        self.absurdle_suggestions_cancellable(observations, top, &|| false)
    }

    pub fn absurdle_suggestions_cancellable(
        &self,
        observations: &[(String, u8)],
        top: usize,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<AbsurdleSuggestion>> {
        check_predictive_search_cancelled(cancelled)?;
        let terminal = crate::game::status(observations, crate::game::GameRules::Absurdle)?;
        let mut state = self.absurdle_initial_state();
        for (guess, pattern) in observations {
            check_predictive_search_cancelled(cancelled)?;
            self.apply_feedback(&mut state, guess, *pattern)?;
        }
        if terminal == crate::game::GameStatus::Solved {
            return Ok(Vec::new());
        }
        self.absurdle_suggestions_for_state_controlled(&state, top, cancelled)
    }

    pub fn absurdle_suggestions_for_state(
        &self,
        state: &SolveState,
        top: usize,
    ) -> Result<Vec<AbsurdleSuggestion>> {
        self.absurdle_suggestions_for_state_controlled(state, top, &|| false)
    }

    fn absurdle_suggestions_for_state_controlled(
        &self,
        state: &SolveState,
        top: usize,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<AbsurdleSuggestion>> {
        check_predictive_search_cancelled(cancelled)?;
        if state.surviving.is_empty() {
            bail!("cannot score guesses with an empty state");
        }
        let total = state.surviving.len() as f64;
        let mut suggestions = (0..self.guesses.len())
            .into_par_iter()
            .map(|guess_index| {
                self.absurdle_score_guess(guess_index, &state.surviving, total, cancelled)
            })
            .collect::<Result<Vec<_>>>()?;
        check_predictive_search_cancelled(cancelled)?;
        suggestions.sort_by(compare_absurdle_suggestions);
        suggestions.truncate(top.min(suggestions.len()));
        check_predictive_search_cancelled(cancelled)?;
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

fn validate_predictive_history(observations: &[(String, u8)], hard_mode: bool) -> Result<()> {
    crate::game::status(observations, crate::game::GameRules::Wordle)?;
    for (index, (guess, _)) in observations.iter().enumerate() {
        if hard_mode && let Some(error) = hard_mode_violation_message(&observations[..index], guess)
        {
            bail!("invalid hard-mode turn {}: {}", index + 1, error);
        }
    }
    Ok(())
}

pub(super) fn hard_mode_violation_message(
    observations: &[(String, u8)],
    guess: &str,
) -> Option<String> {
    if guess.len() != HARD_MODE_WORD_LENGTH || !guess.bytes().all(|byte| byte.is_ascii_lowercase())
    {
        return Some("hard mode guess must be exactly 5 lowercase letters".to_string());
    }
    if observations.iter().any(|(word, pattern)| {
        word.len() != HARD_MODE_WORD_LENGTH
            || !word.bytes().all(|byte| byte.is_ascii_lowercase())
            || *pattern as usize >= PATTERN_SPACE
    }) {
        return Some("hard mode history contains an invalid guess or feedback pattern".to_string());
    }

    let constraints = build_hard_mode_constraints(observations);
    constraints
        .violation(guess)
        .map(|violation| match violation {
            HardModeViolation::Green(expected, index) => format!(
                "hard mode requires {} in position {}",
                char::from(expected).to_ascii_uppercase(),
                index + 1
            ),
            HardModeViolation::Yellow(byte, index) => format!(
                "hard mode forbids {} in position {}",
                char::from(byte).to_ascii_uppercase(),
                index + 1
            ),
            HardModeViolation::Count(letter, required) => format!(
                "hard mode requires {} occurrence{} of {}",
                required,
                if required == 1 { "" } else { "s" },
                char::from(letter).to_ascii_uppercase()
            ),
        })
}

enum HardModeViolation {
    Green(u8, usize),
    Yellow(u8, usize),
    Count(u8, u8),
}

impl HardModeConstraints {
    pub(super) fn allows(&self, guess: &str) -> bool {
        self.violation(guess).is_none()
    }

    fn violation(&self, guess: &str) -> Option<HardModeViolation> {
        debug_assert_eq!(guess.len(), HARD_MODE_WORD_LENGTH);
        debug_assert!(guess.bytes().all(|byte| byte.is_ascii_lowercase()));
        let guess_bytes = guess.as_bytes();
        let mut guess_counts = [0u8; 26];
        for (index, &byte) in guess_bytes.iter().enumerate() {
            let letter_index = (byte - b'a') as usize;
            guess_counts[letter_index] += 1;

            if let Some(expected) = self.greens[index]
                && byte != expected
            {
                return Some(HardModeViolation::Green(expected, index));
            }

            if (self.yellow_forbidden[index] & (1u32 << letter_index)) != 0 {
                return Some(HardModeViolation::Yellow(byte, index));
            }
        }

        for (letter_index, &required) in self.required_counts.iter().enumerate() {
            if required > 0 && guess_counts[letter_index] < required {
                return Some(HardModeViolation::Count(
                    b'a' + letter_index as u8,
                    required,
                ));
            }
        }

        None
    }
}

pub(super) struct HardModeConstraints {
    greens: [Option<u8>; HARD_MODE_WORD_LENGTH],
    yellow_forbidden: [u32; HARD_MODE_WORD_LENGTH],
    required_counts: [u8; 26],
}

pub(super) fn build_hard_mode_constraints(observations: &[(String, u8)]) -> HardModeConstraints {
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
