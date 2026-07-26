use super::*;

struct StudyEvaluationRequest<'a> {
    paths: &'a ProjectPaths,
    config: &'a PriorConfig,
    stage: StudyStage,
    artifact_namespace: &'a str,
    evaluation_plan: &'a EvaluationPlan,
    top: usize,
    target_validation_folds: usize,
    validation_fold_indices: &'a [usize],
    maximum_trial_seconds: u64,
    maximum_memory_mb: u64,
    measure_latency: bool,
    measurement: StudyMeasurement,
    prior_elapsed_ms: u64,
    cancellation_path: Option<&'a Path>,
}

fn needs_serial_study_latency(
    status: TrialStatus,
    completed_folds: usize,
    maximum_folds: usize,
    has_latency: bool,
) -> bool {
    status == TrialStatus::Complete && completed_folds >= maximum_folds && !has_latency
}

#[derive(Clone, Debug)]
struct SearchRegretCandidateState {
    date: NaiveDate,
    target: String,
    turn: usize,
    observations: Vec<(String, u8)>,
    surviving_answers: usize,
}

fn evenly_spaced_indices(total: usize, maximum: usize) -> Vec<usize> {
    let take = total.min(maximum);
    match take {
        0 => Vec::new(),
        1 => vec![total / 2],
        _ => (0..take)
            .map(|index| index * (total - 1) / (take - 1))
            .collect(),
    }
}

fn summarize_search_regret(
    states: &[SearchRegretState],
    choice: impl Fn(&SearchRegretState) -> &SearchRegretChoice,
) -> SearchRegretSummary {
    let regrets = states
        .iter()
        .map(|state| choice(state).regret)
        .collect::<Vec<_>>();
    SearchRegretSummary {
        states: states.len(),
        exact_matches: states
            .iter()
            .filter(|state| choice(state).matches_optimum)
            .count(),
        positive_regret_states: regrets.iter().filter(|regret| **regret > 1e-9).count(),
        mean_regret: if regrets.is_empty() {
            0.0
        } else {
            regrets.iter().sum::<f64>() / regrets.len() as f64
        },
        maximum_regret: regrets.into_iter().fold(0.0, f64::max),
    }
}

impl Solver {
    pub fn solve_target(&self, target: &str, date: NaiveDate, top: usize) -> Result<SolveRun> {
        Ok(self.solve_target_detailed(target, date, top)?.into())
    }

    pub fn solve_target_detailed(
        &self,
        target: &str,
        date: NaiveDate,
        top: usize,
    ) -> Result<DetailedSolveRun> {
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        self.solve_target_from_state_detailed(target, as_of, date, top, PredictiveBookUsage::Full)
    }

    pub(super) fn solve_target_from_state_detailed(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedSolveRun> {
        let target = target.to_ascii_lowercase();
        let mut state = self.initial_state(as_of);
        let mut observations = Vec::new();

        if !state
            .surviving
            .iter()
            .chain(state.fallback_surviving.iter())
            .any(|index| self.answers[*index].word == target)
        {
            return Ok(DetailedSolveRun {
                target,
                date,
                steps: Vec::new(),
                solved: false,
            });
        }

        let mut steps = Vec::new();
        while steps.len() < 6 {
            let surviving_before = state.surviving.len();
            let batch = self.suggestion_batch_for_history(
                as_of,
                &observations,
                &state,
                top.max(1),
                book_usage,
            )?;
            let chosen = batch
                .suggestions
                .first()
                .ok_or_else(|| anyhow!("solver returned no suggestions"))?
                .clone();
            let feedback = score_guess(&chosen.word, &target);
            let surviving_after = if feedback == ALL_GREEN_PATTERN {
                1
            } else {
                let mut next_state = state.clone();
                self.apply_feedback(&mut next_state, &chosen.word, feedback)?;
                next_state.surviving.len()
            };
            steps.push(DetailedSolveStep {
                guess: chosen.word.clone(),
                feedback,
                surviving_before,
                surviving_after,
                chosen_force_in_two: chosen.force_in_two,
                alternative_force_in_two: batch
                    .suggestions
                    .iter()
                    .skip(1)
                    .any(|suggestion| suggestion.force_in_two),
                danger_score: batch.danger_score,
                danger_escalated: batch.danger_escalated,
                regime_used: batch.regime_used,
                promotion_source: batch.promotion_source,
                recovery_mode_used: state.recovery_mode_used,
                fallback_active: state.fallback_active,
                lookahead_pool_base: batch.lookahead_pool_base,
                lookahead_pool_size: batch.lookahead_pool_size,
                exact_pool_base: batch.exact_pool_base,
                exact_pool_size: batch.exact_pool_size,
                root_candidate_count: batch.root_candidate_count,
                top_suggestions: batch
                    .suggestions
                    .iter()
                    .take(top.max(1))
                    .map(Self::snapshot_suggestion)
                    .collect(),
            });
            if feedback == ALL_GREEN_PATTERN {
                return Ok(DetailedSolveRun {
                    target,
                    date,
                    steps,
                    solved: true,
                });
            }
            observations.push((chosen.word.clone(), feedback));
            self.apply_feedback(&mut state, &chosen.word, feedback)?;
        }

        Ok(DetailedSolveRun {
            target,
            date,
            steps,
            solved: false,
        })
    }

    pub fn backtest(&self, from: NaiveDate, to: NaiveDate, top: usize) -> Result<BacktestStats> {
        Ok(self.backtest_detailed(from, to, top)?.summary)
    }

    pub fn backtest_detailed(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<DetailedBacktestReport> {
        self.backtest_detailed_with_book_usage(from, to, top, PredictiveBookUsage::DiskOnly)
    }

    pub fn search_regret_report(
        &self,
        paths: &ProjectPaths,
        request: SearchRegretRequest,
    ) -> Result<SearchRegretReport> {
        let SearchRegretRequest {
            from,
            to,
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
        } = request;
        if from > to {
            bail!("search-regret start date cannot be after end date");
        }
        if minimum_survivors < 2 {
            bail!("search-regret minimum survivors must be at least 2");
        }
        if minimum_survivors > maximum_survivors {
            bail!("search-regret minimum survivors cannot exceed maximum survivors");
        }
        if maximum_states == 0 {
            bail!("search-regret maximum states must be greater than zero");
        }
        if maximum_seconds == 0 {
            bail!("search-regret maximum seconds must be greater than zero");
        }

        let started = Instant::now();
        let input_fingerprint = rolling_source_identity(paths)?;
        let budget = std::time::Duration::from_secs(maximum_seconds);
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .cloned()
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no games found in the requested search-regret range");
        }

        let historical_games = games.len();
        let game_scan_limit = historical_games.min(maximum_states.saturating_mul(2).max(4));
        let game_indices = evenly_spaced_indices(historical_games, game_scan_limit);
        let games = game_indices
            .into_iter()
            .map(|index| games[index].clone())
            .collect::<Vec<_>>();
        let total_games = games.len();
        let mut candidates = Vec::new();
        for (game_index, entry) in games.into_iter().enumerate() {
            if started.elapsed() > budget {
                bail!(
                    "search-regret exceeded its {} second budget while collecting reachable states",
                    maximum_seconds
                );
            }
            let as_of = entry
                .print_date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
            let target = entry.solution.to_ascii_lowercase();
            let mut state = self.initial_state(as_of);
            let mut observations = Vec::new();
            if state
                .surviving
                .iter()
                .chain(state.fallback_surviving.iter())
                .any(|index| self.answers[*index].word == target)
            {
                for step_index in 0..5 {
                    if (minimum_survivors..=maximum_survivors).contains(&state.surviving.len()) {
                        candidates.push(SearchRegretCandidateState {
                            date: entry.print_date,
                            target: target.clone(),
                            turn: step_index + 1,
                            observations: observations.clone(),
                            surviving_answers: state.surviving.len(),
                        });
                        break;
                    }
                    if state.surviving.len() < minimum_survivors {
                        break;
                    }
                    let chosen = self
                        .suggestion_batch_internal_with_search_mode(
                            &state,
                            1,
                            Some(PredictiveContext {
                                as_of,
                                observations: &observations,
                            }),
                            PredictiveBookUsage::None,
                            Some(PredictiveSearchMode::ProxyOnly),
                        )?
                        .suggestions
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow!("solver returned no suggestion during audit"))?;
                    let feedback = score_guess(&chosen.word, &target);
                    if feedback == ALL_GREEN_PATTERN {
                        break;
                    }
                    observations.push((chosen.word.clone(), feedback));
                    self.apply_feedback(&mut state, &chosen.word, feedback)?;
                }
            }
            if game_index < 2 || (game_index + 1) % 5 == 0 || game_index + 1 == total_games {
                eprintln!(
                    "search-regret phase=collect games={}/{} eligible_states={} elapsed_s={:.1}",
                    game_index + 1,
                    total_games,
                    candidates.len(),
                    started.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
            }
        }
        if candidates.is_empty() {
            bail!(
                "no reachable states had between {} and {} survivors",
                minimum_survivors,
                maximum_survivors
            );
        }

        let available_states = candidates.len();
        let selected_indices = evenly_spaced_indices(available_states, maximum_states);
        let mut exhaustive_solver = self.clone();
        exhaustive_solver.config.exact_threshold = maximum_survivors;
        exhaustive_solver.config.exact_exhaustive_threshold = maximum_survivors;
        let mut states = Vec::with_capacity(selected_indices.len());
        let sampled_states = selected_indices.len();
        for (sample_index, candidate_index) in selected_indices.into_iter().enumerate() {
            if started.elapsed() > budget {
                bail!(
                    "search-regret exceeded its {} second budget before completing all sampled states",
                    maximum_seconds
                );
            }
            let candidate = &candidates[candidate_index];
            let as_of = candidate
                .date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot audit a game before launch date"))?;
            let state = self.apply_history(as_of, &candidate.observations)?;
            if state.surviving.len() != candidate.surviving_answers {
                bail!(
                    "search-regret state reconstruction mismatch for {} turn {}: expected {} survivors, reconstructed {}",
                    candidate.date,
                    candidate.turn,
                    candidate.surviving_answers,
                    state.surviving.len()
                );
            }

            let context = Some(PredictiveContext {
                as_of,
                observations: &candidate.observations,
            });
            let proxy_guess = self
                .suggestion_batch_internal_with_search_mode(
                    &state,
                    1,
                    context,
                    PredictiveBookUsage::None,
                    Some(PredictiveSearchMode::ProxyOnly),
                )?
                .suggestions
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("proxy audit returned no suggestion"))?
                .word;
            let lookahead_guess = self
                .suggestion_batch_internal_with_search_mode(
                    &state,
                    1,
                    context,
                    PredictiveBookUsage::None,
                    Some(PredictiveSearchMode::Lookahead),
                )?
                .suggestions
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("lookahead audit returned no suggestion"))?
                .word;
            let production_is_exhaustive = self.config.search_policy_mode
                != crate::config::SearchPolicyMode::ProxyOnly
                && matches!(
                    exact_suggestion_mode(&self.config, state.surviving.len()),
                    Some(ExactSuggestionMode::Exhaustive)
                );
            let (production_guess, production_regime) = if production_is_exhaustive {
                (None, PredictiveRegime::Exact)
            } else {
                let production = self.suggestion_batch_for_history(
                    as_of,
                    &candidate.observations,
                    &state,
                    1,
                    PredictiveBookUsage::None,
                )?;
                (
                    Some(
                        production
                            .suggestions
                            .into_iter()
                            .next()
                            .ok_or_else(|| anyhow!("production audit returned no suggestion"))?
                            .word,
                    ),
                    production.regime_used,
                )
            };
            states.push(exhaustive_solver.audit_search_regret_state(
                candidate,
                &state,
                production_guess.as_deref(),
                production_regime,
                &proxy_guess,
                &lookahead_guess,
            )?);
            eprintln!(
                "search-regret phase=exhaustive states={}/{} survivors={} elapsed_s={:.1}",
                sample_index + 1,
                sampled_states,
                candidate.surviving_answers,
                started.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }

        let config_toml =
            toml::to_string_pretty(&self.config).context("failed to serialize audit config")?;
        ensure_rolling_source_identity(paths, &input_fingerprint)?;
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        Ok(SearchRegretReport {
            schema_version: 1,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            config_fingerprint: crate::identity::digest_bytes_tagged(
                "maybe-wordle-search-regret-config-v1",
                config_toml.as_bytes(),
            ),
            code_revision,
            code_dirty,
            evaluation_from: from,
            evaluation_to: to,
            state_path_policy: "forced_proxy_without_artifacts".to_string(),
            minimum_survivors,
            maximum_survivors,
            maximum_states,
            maximum_seconds,
            historical_games,
            scanned_games: total_games,
            available_states,
            sampled_states: states.len(),
            generation_elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            production: summarize_search_regret(&states, |state| &state.production),
            proxy: summarize_search_regret(&states, |state| &state.proxy),
            lookahead: summarize_search_regret(&states, |state| &state.lookahead),
            states,
        })
    }

    fn audit_search_regret_state(
        &self,
        candidate: &SearchRegretCandidateState,
        state: &SolveState,
        production_guess: Option<&str>,
        production_regime: PredictiveRegime,
        proxy_guess: &str,
        lookahead_guess: &str,
    ) -> Result<SearchRegretState> {
        let production_index = production_guess
            .map(|guess| {
                self.guess_index
                    .get(guess)
                    .copied()
                    .with_context(|| format!("unknown production guess {guess}"))
            })
            .transpose()?;
        let proxy_index = self
            .guess_index
            .get(proxy_guess)
            .copied()
            .with_context(|| format!("unknown proxy guess {proxy_guess}"))?;
        let lookahead_index = self
            .guess_index
            .get(lookahead_guess)
            .copied()
            .with_context(|| format!("unknown lookahead guess {lookahead_guess}"))?;

        let selected = production_index
            .into_iter()
            .chain([proxy_index, lookahead_index])
            .collect::<Vec<_>>();
        let mut memo = PredictiveMemoMap::default();
        let mut scratch = ExactSearchScratch::new();
        let lower_bound = weighted_exact_lower_bound(&state.surviving, &state.weights)?;
        let mut optimal_index = proxy_index;
        let mut optimal_cost = f64::INFINITY;
        for guess_index in 0..self.guesses.len() {
            let cost = self.exact_cost_for_guess(
                guess_index,
                ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    small_state_table: &self.exact_small_state_table,
                    memo: &mut memo,
                    best_bound: optimal_cost,
                    scratch: &mut scratch,
                    depth: 0,
                },
            )?;
            if cost.total_cmp(&optimal_cost).is_lt() {
                optimal_index = guess_index;
                optimal_cost = cost;
                if optimal_cost <= lower_bound + 1e-12 {
                    break;
                }
            }
        }
        if !optimal_cost.is_finite() {
            bail!(
                "search-regret found no exhaustive optimum for {} turn {}",
                candidate.date,
                candidate.turn
            );
        }

        let mut selected_costs = HashMap::new();
        for guess_index in selected.iter().copied() {
            if selected_costs.contains_key(&guess_index) {
                continue;
            }
            let cost = self.exact_cost_for_guess(
                guess_index,
                ExactCostContext {
                    subset: &state.surviving,
                    weights: &state.weights,
                    small_state_table: &self.exact_small_state_table,
                    memo: &mut memo,
                    best_bound: f64::INFINITY,
                    scratch: &mut scratch,
                    depth: 0,
                },
            )?;
            if !cost.is_finite() {
                bail!(
                    "search-regret selected non-progressing guess {} for {} turn {}",
                    self.guesses[guess_index],
                    candidate.date,
                    candidate.turn
                );
            }
            selected_costs.insert(guess_index, cost);
        }

        let choice = |guess_index: usize| {
            let exact_cost = selected_costs[&guess_index];
            let regret = (exact_cost - optimal_cost).max(0.0);
            SearchRegretChoice {
                word: self.guesses[guess_index].clone(),
                exact_cost,
                regret,
                matches_optimum: regret <= 1e-9,
            }
        };
        Ok(SearchRegretState {
            date: candidate.date,
            target: candidate.target.clone(),
            turn: candidate.turn,
            surviving_answers: candidate.surviving_answers,
            observations: candidate
                .observations
                .iter()
                .map(|(guess, feedback)| SearchRegretObservation {
                    guess: guess.clone(),
                    feedback: format_feedback_letters(*feedback),
                })
                .collect(),
            production_regime: production_regime.label().to_string(),
            optimal_word: self.guesses[optimal_index].clone(),
            optimal_exact_cost: optimal_cost,
            production: production_index.map_or_else(
                || SearchRegretChoice {
                    word: self.guesses[optimal_index].clone(),
                    exact_cost: optimal_cost,
                    regret: 0.0,
                    matches_optimum: true,
                },
                choice,
            ),
            proxy: choice(proxy_index),
            lookahead: choice(lookahead_index),
        })
    }

    pub(super) fn backtest_detailed_with_book_usage(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedBacktestReport> {
        self.backtest_detailed_with_book_usage_and_progress(from, to, top, book_usage, None)
    }

    fn backtest_detailed_with_book_usage_and_progress(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<DetailedBacktestReport> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested backtest range");
        }

        self.backtest_selected_games_with_progress(&games, top, book_usage, progress)
    }

    pub(super) fn recovery_backtest_detailed_with_book_usage(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedBacktestReport> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .filter(|entry| {
                let Some(as_of) = entry.print_date.checked_sub_days(Days::new(1)) else {
                    return false;
                };
                let target = entry.solution.to_ascii_lowercase();
                let state = self.initial_state(as_of);
                !state
                    .surviving
                    .iter()
                    .any(|index| self.answers[*index].word == target)
                    && state
                        .fallback_surviving
                        .iter()
                        .any(|index| self.answers[*index].word == target)
            })
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no out-of-primary recovery games found in the requested range");
        }
        self.backtest_selected_games(&games, top, book_usage)
    }

    pub(super) fn backtest_selected_games(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<DetailedBacktestReport> {
        self.backtest_selected_games_with_progress(games, top, book_usage, None)
    }

    pub(super) fn backtest_selected_games_with_progress(
        &self,
        games: &[&NytDailyEntry],
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<DetailedBacktestReport> {
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let total = games.len();
        let evaluated = games
            .par_iter()
            .map(|entry| {
                let result = self.solve_backtest_entry(entry, top, book_usage);
                let current = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if let Some(progress) = progress {
                    progress(current, total);
                }
                result
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        // `games.par_iter()` is indexed, so Rayon preserves the canonical
        // chronological input order even though independent games run in parallel.
        let (outcomes, runs) = evaluated.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

        let canonical = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let failure_rate_ci95 = (
            1.0 - canonical.solve_rate_ci95.upper,
            1.0 - canonical.solve_rate_ci95.lower,
        );

        Ok(DetailedBacktestReport {
            summary: BacktestStats {
                games: canonical.scheduled_games,
                average_guesses: canonical.conditional_mean_guesses,
                p95_guesses: canonical.p95_guesses,
                max_guesses: canonical.max_guesses,
                failures: canonical.unsolved_games + canonical.coverage_gaps,
                coverage_gaps: canonical.coverage_gaps,
                average_guesses_ci95: (
                    canonical.conditional_mean_guesses_ci95.lower,
                    canonical.conditional_mean_guesses_ci95.upper,
                ),
                failure_rate_ci95,
                canonical,
            },
            runs,
        })
    }

    pub(super) fn solve_backtest_entry(
        &self,
        entry: &NytDailyEntry,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<(GameOutcome, DetailedSolveRun)> {
        let as_of = entry
            .print_date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let run = self.solve_target_from_state_detailed(
            &entry.solution,
            as_of,
            entry.print_date,
            top,
            book_usage,
        )?;
        let outcome = if run.steps.is_empty() {
            GameOutcome::coverage_gap(entry.print_date)
        } else if run.solved {
            GameOutcome::solved(entry.print_date, run.steps.len())
        } else {
            GameOutcome::unsolved(entry.print_date, run.steps.len())
        };
        Ok((outcome, run))
    }

    pub fn hard_case_report(&self, top: usize) -> Result<HardCaseReport> {
        self.hard_case_report_with_book_usage(top, PredictiveBookUsage::DiskOnly)
    }

    pub(super) fn hard_case_report_with_book_usage(
        &self,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<HardCaseReport> {
        let as_of = Self::today();
        let hard_case_spec = default_diagnostic_suite()?.hard_cases;
        let cases = self.select_hard_case_targets(as_of, top, &hard_case_spec)?;
        let mut results = Vec::new();
        let mut failures = 0usize;
        let mut guess_total = 0usize;

        for (label, target) in cases {
            let run =
                self.solve_target_from_state_detailed(&target, as_of, as_of, top, book_usage)?;
            if !run.solved {
                failures += 1;
            }
            guess_total += run.steps.len();
            results.push(HardCaseResult { label, run });
        }

        let average_guesses = if results.is_empty() {
            0.0
        } else {
            guess_total as f64 / results.len() as f64
        };
        Ok(HardCaseReport {
            average_guesses,
            failures,
            cases: results,
        })
    }

    pub fn experiment_report(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<ExperimentResult> {
        self.experiment_report_with_book_usage(from, to, top, PredictiveBookUsage::DiskOnly)
    }

    fn experiment_report_with_book_usage(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
    ) -> Result<ExperimentResult> {
        self.experiment_report_with_book_usage_and_progress(from, to, top, book_usage, None)
    }

    fn experiment_report_with_book_usage_and_progress(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        book_usage: PredictiveBookUsage,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<ExperimentResult> {
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let detailed = self
            .backtest_detailed_with_book_usage_and_progress(from, to, top, book_usage, progress)?;
        let backtest = detailed.summary.clone();
        let (proxy_step_pct, lookahead_step_pct, escalated_exact_step_pct, exact_step_pct) =
            Self::regime_mix(&detailed.runs);
        let mut lookahead_pool_ratio_sum = 0.0;
        let mut lookahead_pool_ratio_count = 0usize;
        let mut exact_pool_ratio_sum = 0.0;
        let mut exact_pool_ratio_count = 0usize;
        for run in &detailed.runs {
            for step in &run.steps {
                if step.lookahead_pool_base > 0 && step.lookahead_pool_size > 0 {
                    lookahead_pool_ratio_sum +=
                        step.lookahead_pool_size as f64 / step.lookahead_pool_base as f64;
                    lookahead_pool_ratio_count += 1;
                }
                if step.exact_pool_base > 0 && step.exact_pool_size > 0 {
                    exact_pool_ratio_sum +=
                        step.exact_pool_size as f64 / step.exact_pool_base as f64;
                    exact_pool_ratio_count += 1;
                }
            }
        }
        let mut total_log_loss = 0.0;
        let mut total_brier = 0.0;
        let mut total_target_probability = 0.0;
        let mut total_rank = 0.0;
        let mut measured = 0usize;
        let mut prior_observations = Vec::new();

        for entry in games {
            if let Some(metrics) = self.initial_prior_metrics(&entry.solution, entry.print_date) {
                total_log_loss += metrics.log_loss;
                total_brier += metrics.brier;
                total_target_probability += metrics.target_probability;
                total_rank += metrics.target_rank as f64;
                measured += 1;
                prior_observations.push(RankedProbabilityObservation {
                    target_rank: metrics.target_rank,
                    top_probability: metrics.top_probability,
                    top_prediction_correct: metrics.top_prediction_correct,
                });
            }
        }

        let divisor = measured.max(1) as f64;
        let outcomes = detailed
            .runs
            .iter()
            .map(|run| {
                if run.steps.is_empty() {
                    GameOutcome::coverage_gap(run.date)
                } else if run.solved {
                    GameOutcome::solved(run.date, run.steps.len())
                } else {
                    GameOutcome::unsolved(run.date, run.steps.len())
                }
            })
            .collect::<Vec<_>>();
        let failure_penalty_sensitivity = [6.0, 7.0, 8.0]
            .into_iter()
            .map(|penalty_guesses| {
                let metrics = summarize_predictive_outcomes(
                    &outcomes,
                    penalty_guesses,
                    BootstrapConfig::default(),
                )?;
                Ok(FailurePenaltyEvidence {
                    penalty_guesses,
                    all_game_mean_guesses: metrics.all_game_penalized_mean_guesses,
                    ci95: metrics.all_game_penalized_mean_guesses_ci95,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let fallback_as_of = to
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("session-fallback benchmark cutoff underflowed"))?;
        let (session_fallback_cold_ms, session_fallback_warm_ms) =
            self.benchmark_session_fallback_latency(fallback_as_of)?;
        Ok(ExperimentResult {
            config_id: format!(
                "{}-et{}-ee{}-cp{}-lt{}-lc{}-lr{}-ls{}",
                self.config.search_policy_mode.label(),
                self.config.exact_threshold,
                self.config.exact_exhaustive_threshold,
                self.config.exact_candidate_pool,
                self.config.lookahead_threshold,
                self.config.lookahead_candidate_pool,
                self.config.lookahead_reply_pool,
                self.config.large_state_split_threshold,
            ),
            mode: self.mode,
            variant: self.variant,
            backtest,
            average_log_loss: total_log_loss / divisor,
            average_brier: total_brier / divisor,
            average_target_probability: total_target_probability / divisor,
            average_target_rank: total_rank / divisor,
            prior_evidence: (!prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            execution: Self::execution_telemetry(&detailed.runs),
            failure_penalty_sensitivity,
            latency_p95_ms: self
                .benchmark_predictive_latency(default_diagnostic_suite()?.latency.evidence_runs)?,
            session_fallback_cold_ms,
            session_fallback_warm_ms,
            proxy_step_pct,
            lookahead_step_pct,
            escalated_exact_step_pct,
            exact_step_pct,
            average_lookahead_pool_ratio: if lookahead_pool_ratio_count == 0 {
                0.0
            } else {
                lookahead_pool_ratio_sum / lookahead_pool_ratio_count as f64
            },
            average_exact_pool_ratio: if exact_pool_ratio_count == 0 {
                0.0
            } else {
                exact_pool_ratio_sum / exact_pool_ratio_count as f64
            },
            games: detailed
                .runs
                .iter()
                .map(|run| ExperimentGameResult {
                    target: run.target.clone(),
                    outcome: if run.steps.is_empty() {
                        GameOutcome::coverage_gap(run.date)
                    } else if run.solved {
                        GameOutcome::solved(run.date, run.steps.len())
                    } else {
                        GameOutcome::unsolved(run.date, run.steps.len())
                    },
                    path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                })
                .collect(),
        })
    }

    pub fn build_development_evidence(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<PredictiveEvidenceArtifact> {
        Self::build_development_evidence_with_budget(
            paths,
            config,
            from,
            to,
            top,
            EvidenceResourceBudget::default(),
        )
    }

    pub fn build_development_evidence_with_budget(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        resource_budget: EvidenceResourceBudget,
    ) -> Result<PredictiveEvidenceArtifact> {
        if resource_budget.maximum_seconds == 0 || resource_budget.maximum_memory_mb == 0 {
            bail!("evidence time and memory budgets must be positive");
        }
        let generation_started = Instant::now();
        let input_fingerprint = rolling_source_identity(paths)?;
        enforce_evidence_resource_budget(generation_started, resource_budget)?;
        if from > to {
            bail!("evidence start date cannot be after end date");
        }
        let plan = canonical_development_evaluation_plan(paths, "generating evidence")?;
        if from > plan.development.end || to > plan.development.end {
            bail!(
                "evidence range {}..{} reaches the sealed test; development evidence must end on or before {}",
                from,
                to,
                plan.development.end
            );
        }

        let matrix = PredictiveExperimentMatrix::parse_json(include_str!(
            "../../config/experiments/development-evidence.json"
        ))?;
        let total_profiles = matrix.profiles.len();
        let completed_games = std::sync::atomic::AtomicUsize::new(0);
        let mut baselines = Vec::with_capacity(total_profiles);
        eprintln!(
            "benchmark-evidence phase=start profiles={} rayon_threads={} from={} to={} elapsed_s=0.0",
            total_profiles,
            rayon::current_num_threads(),
            from,
            to,
        );
        let _ = std::io::stderr().flush();
        for (profile_index, profile) in matrix.profiles.into_iter().enumerate() {
            let profile_id = profile.id.clone();
            eprintln!(
                "benchmark-evidence phase=profile-start profile={}/{} id={} elapsed_s={:.1}",
                profile_index + 1,
                total_profiles,
                profile_id,
                generation_started.elapsed().as_secs_f64(),
            );
            let _ = std::io::stderr().flush();
            let profile_base = profile.load_base_config(&paths.root, config)?;
            let profile_config =
                profile.apply(&predictive_parameter_registry(&profile_base), &profile_base)?;
            let book_usage = match profile.artifact_mode {
                ExperimentArtifactMode::Disabled => PredictiveBookUsage::None,
                ExperimentArtifactMode::DiskOnly => PredictiveBookUsage::DiskOnly,
            };
            let solver = Self::from_paths_with_settings(
                paths,
                &profile_config,
                profile.weight_mode,
                profile.model_variant,
            )?;
            let effective_config_toml = toml::to_string_pretty(&profile_config)
                .context("failed to serialize evidence profile config")?;
            let progress = |profile_completed: usize, profile_total: usize| {
                let total_games = profile_total.saturating_mul(total_profiles);
                let global_completed =
                    completed_games.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let elapsed = generation_started.elapsed().as_secs_f64();
                let eta = if global_completed == 0 {
                    0.0
                } else {
                    elapsed / global_completed as f64
                        * total_games.saturating_sub(global_completed) as f64
                };
                eprintln!(
                    "benchmark-evidence phase=games id={} profile_games={}/{} total_games={}/{} elapsed_s={:.1} eta_s={:.1}",
                    profile_id,
                    profile_completed,
                    profile_total,
                    global_completed,
                    total_games,
                    elapsed,
                    eta,
                );
                let _ = std::io::stderr().flush();
            };
            let result = solver.experiment_report_with_book_usage_and_progress(
                from,
                to,
                top,
                book_usage,
                Some(&progress),
            )?;
            eprintln!(
                "benchmark-evidence phase=profile-complete profile={}/{} id={} solved={}/{} failures={} mean_guesses={:.4} elapsed_s={:.1}",
                profile_index + 1,
                total_profiles,
                profile_id,
                result.backtest.canonical.solved_games,
                result.backtest.canonical.scheduled_games,
                result.backtest.failures,
                result.backtest.canonical.all_game_penalized_mean_guesses,
                generation_started.elapsed().as_secs_f64(),
            );
            let _ = std::io::stderr().flush();
            baselines.push(EvidenceBaseline {
                id: profile.id,
                description: profile.description,
                artifacts: match book_usage {
                    PredictiveBookUsage::None => "disabled",
                    PredictiveBookUsage::DiskOnly => "valid_disk_only",
                    PredictiveBookUsage::Full => "disk_then_live",
                }
                .to_string(),
                config_fingerprint: crate::identity::digest_bytes_tagged(
                    "maybe-wordle-benchmark-config-v1",
                    effective_config_toml.as_bytes(),
                ),
                effective_config_toml,
                paired_vs_selected_default: None,
                result,
            });
            enforce_evidence_resource_budget(generation_started, resource_budget)?;
        }
        let selected_outcomes = baselines
            .iter()
            .find(|baseline| baseline.id == "selected_default_disk_artifacts")
            .map(|baseline| {
                baseline
                    .result
                    .games
                    .iter()
                    .map(|game| game.outcome)
                    .collect::<Vec<_>>()
            })
            .ok_or_else(|| anyhow!("selected-default evidence baseline is missing"))?;
        for baseline in &mut baselines {
            let candidate = baseline
                .result
                .games
                .iter()
                .map(|game| game.outcome)
                .collect::<Vec<_>>();
            baseline.paired_vs_selected_default = Some(PairedDifference::all_game_penalized(
                &selected_outcomes,
                &candidate,
                7.0,
                BootstrapConfig::default(),
            )?);
        }

        let (code_revision, code_dirty) = git_provenance(&paths.root);
        let generation_compute_ms = generation_started
            .elapsed()
            .as_millis()
            .min(u64::MAX as u128) as u64;
        let memory = enforce_evidence_resource_budget(generation_started, resource_budget)?;
        let config_toml =
            toml::to_string_pretty(config).context("failed to serialize evidence config")?;
        let config_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-benchmark-root-config-v1",
            config_toml.as_bytes(),
        );
        ensure_rolling_source_identity(paths, &input_fingerprint)?;
        Ok(PredictiveEvidenceArtifact {
            schema_version: 4,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            config_fingerprint,
            scope: "rolling-development-diagnostic; not sealed-test evidence".to_string(),
            sealed_test_evaluated: false,
            evaluation_from: from,
            evaluation_to: to,
            history_snapshot_start: plan.history.start,
            history_snapshot_end: plan.history.end,
            code_revision,
            code_dirty,
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            cpu: std::env::var("PROCESSOR_IDENTIFIER").ok(),
            release_command: format!(
                "cargo run --release -- benchmark-evidence --from {from} --to {to} --maximum-seconds {} --maximum-memory-mb {} --output <json> --markdown-output <md>",
                resource_budget.maximum_seconds,
                resource_budget.maximum_memory_mb
            ),
            config_toml,
            resource_budget,
            resources: EvidenceResourceTelemetry {
                generation_compute_ms,
                current_working_set_bytes: Some(memory.current_working_set_bytes),
                peak_working_set_bytes: Some(memory.peak_working_set_bytes),
                artifact_sizes: evidence_artifact_sizes(paths)?,
            },
            historical_diagnostic: HistoricalDiagnosticBaseline {
                date_range: "historical 30-game diagnostic before dormant-support repair"
                    .to_string(),
                scheduled_games: 30,
                modeled_games: 27,
                coverage_gaps: 3,
                conditional_mean_guesses: 3.2222,
                average_log_loss: 7.327027,
                average_brier_score: 0.999241,
                interpretation: "Attribution baseline only: its guess mean excluded three coverage gaps and is not comparable to an all-game score".to_string(),
            },
            baselines,
            limitations: vec![
                "This artifact evaluates development dates only; the sealed final window remains unopened.".to_string(),
                "Prior probabilities remain heuristic until calibration improves on rolling-origin validation.".to_string(),
                "Identity fields remain non-cryptographic until the planned SHA-256 artifact format is approved and implemented.".to_string(),
            ],
        })
    }

    pub fn render_development_evidence_markdown(
        artifact: &PredictiveEvidenceArtifact,
    ) -> Result<String> {
        artifact.validate_identity()?;
        let mut output = String::new();
        output.push_str("<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->\n");
        output.push_str("## Predictive solver evidence\n\n");
        output.push_str(&format!(
            "Development-only diagnostic for `{}` through `{}` using history through `{}`. The sealed test was **not** evaluated.\n\n",
            artifact.evaluation_from, artifact.evaluation_to, artifact.history_snapshot_end
        ));
        if let Some(peak_bytes) = artifact.resources.peak_working_set_bytes {
            output.push_str(&format!(
                "Measured generation compute time: {:.2} s; process peak working set: {:.1} MiB; enforced budget: {} s / {} MiB.\n\n",
                artifact.resources.generation_compute_ms as f64 / 1_000.0,
                peak_bytes as f64 / (1024.0 * 1024.0),
                artifact.resource_budget.maximum_seconds,
                artifact.resource_budget.maximum_memory_mb
            ));
        }
        output.push_str("| Baseline | Coverage | Solved | All-game mean (7-guess penalty) | Conditional mean | 3 guesses | 4 guesses | Paired delta vs default | W/T/L | Log loss | Brier | Latency p95 | Session fallback cold/warm |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for baseline in &artifact.baselines {
            let metrics = &baseline.result.backtest.canonical;
            let paired = baseline
                .paired_vs_selected_default
                .expect("generated evidence always has paired comparisons");
            output.push_str(&format!(
                "| `{}` | {:.1}% ({}/{}) | {:.1}% ({}/{}) | {:.4} [{:.4}, {:.4}] | {:.4} [{:.4}, {:.4}] | {:.1}% | {:.1}% | {:+.4} [{:+.4}, {:+.4}] | {}/{}/{} | {:.4} | {:.4} | {:.2} ms | {:.2}/{:.3} ms |\n",
                baseline.id,
                metrics.coverage_rate * 100.0,
                metrics.modeled_games,
                metrics.scheduled_games,
                metrics.solve_rate * 100.0,
                metrics.solved_games,
                metrics.scheduled_games,
                metrics.all_game_penalized_mean_guesses,
                metrics.all_game_penalized_mean_guesses_ci95.lower,
                metrics.all_game_penalized_mean_guesses_ci95.upper,
                metrics.conditional_mean_guesses,
                metrics.conditional_mean_guesses_ci95.lower,
                metrics.conditional_mean_guesses_ci95.upper,
                metrics.solved_in_guess_counts[..3].iter().sum::<usize>() as f64
                    / metrics.scheduled_games.max(1) as f64
                    * 100.0,
                metrics.solved_in_guess_counts[..4].iter().sum::<usize>() as f64
                    / metrics.scheduled_games.max(1) as f64
                    * 100.0,
                paired.candidate_minus_baseline,
                paired.ci95.lower,
                paired.ci95.upper,
                paired.candidate_wins,
                paired.ties,
                paired.baseline_wins,
                baseline.result.average_log_loss,
                baseline.result.average_brier,
                baseline.result.latency_p95_ms,
                baseline.result.session_fallback_cold_ms,
                baseline.result.session_fallback_warm_ms,
            ));
        }
        if !artifact.resources.artifact_sizes.is_empty() {
            output.push_str("\nMeasured artifact sizes: ");
            output.push_str(
                &artifact
                    .resources
                    .artifact_sizes
                    .iter()
                    .map(|artifact| format!("`{}` = {} bytes", artifact.name, artifact.bytes))
                    .collect::<Vec<_>>()
                    .join("; "),
            );
            output.push_str(".\n");
        }
        output.push_str("\n| Baseline | Prior top-1 | Prior top-3 | Prior top-5 | Confidence ECE | Search steps P/L/XE/X | Recovery/fallback steps | Artifact/session hits |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for baseline in &artifact.baselines {
            let prior = baseline.result.prior_evidence.as_ref();
            let telemetry = &baseline.result.execution;
            let recall = |value: Option<f64>| {
                value.map_or_else(
                    || "n/a".to_string(),
                    |value| format!("{:.1}%", value * 100.0),
                )
            };
            let ece = prior.map_or_else(
                || "n/a".to_string(),
                |metrics| {
                    format!(
                        "{:.4} [{:.4}, {:.4}]",
                        metrics.expected_calibration_error,
                        metrics.expected_calibration_error_ci95.lower,
                        metrics.expected_calibration_error_ci95.upper
                    )
                },
            );
            output.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {}/{}/{}/{} | {}/{} | {}/{} |\n",
                baseline.id,
                recall(prior.map(|metrics| metrics.top_1_recall)),
                recall(prior.map(|metrics| metrics.top_3_recall)),
                recall(prior.map(|metrics| metrics.top_5_recall)),
                ece,
                telemetry.proxy_steps,
                telemetry.lookahead_steps,
                telemetry.escalated_exact_steps,
                telemetry.exact_steps,
                telemetry.strict_recovery_steps
                    + telemetry.uniform_recovery_steps
                    + telemetry.epsilon_repair_steps,
                telemetry.dormant_fallback_steps,
                telemetry.exact_date_opener_artifact_hits
                    + telemetry.recent_opener_artifact_hits
                    + telemetry.reply_book_hits,
                telemetry.session_fallback_hits,
            ));
        }
        if let Some(selected) = artifact
            .baselines
            .iter()
            .find(|baseline| baseline.id == "selected_default_disk_artifacts")
        {
            output.push_str("\nSelected-default all-game mean sensitivity: ");
            for (index, metric) in selected
                .result
                .failure_penalty_sensitivity
                .iter()
                .enumerate()
            {
                if index > 0 {
                    output.push_str("; ");
                }
                output.push_str(&format!(
                    "penalty {:.0} = {:.4} [{:.4}, {:.4}]",
                    metric.penalty_guesses,
                    metric.all_game_mean_guesses,
                    metric.ci95.lower,
                    metric.ci95.upper
                ));
            }
            output.push_str(".\n");
        }
        output.push_str("\nThe old `3.2222` figure was conditional on 27 modeled games and omitted three coverage gaps. It is retained only as an attribution baseline, not as current performance. A flat three guesses is an aspiration; it is not supported unless the failure-penalized all-game sealed-test result reaches it after configuration freeze.\n\n");
        output.push_str("Regenerate both files with the `release_command` recorded in [`benchmarks/predictive/development-2026-06-17.json`](./benchmarks/predictive/development-2026-06-17.json). Full provenance, per-game paths, effective profile configs, paired comparisons, and limitations live in that artifact. The generated source fragment is [`docs/generated/predictive-evidence.md`](./docs/generated/predictive-evidence.md).\n");
        output.push_str("<!-- END GENERATED PREDICTIVE EVIDENCE -->\n");
        Ok(output)
    }

    pub fn build_rolling_config_comparison(
        paths: &ProjectPaths,
        baseline_config: &PriorConfig,
        baseline_label: &str,
        candidate_config: &PriorConfig,
        candidate_label: &str,
        top: usize,
        reusable_baseline: Option<&RollingComparisonArtifact>,
    ) -> Result<RollingComparisonArtifact> {
        let input_fingerprint = rolling_source_identity(paths)?;
        let evaluation_plan = canonical_development_evaluation_plan(paths, "rolling comparison")?;
        let baseline_toml = toml::to_string_pretty(baseline_config)
            .context("failed to serialize baseline config")?;
        let baseline = if let Some(reusable) = reusable_baseline {
            reusable.validate_identity()?;
            if reusable.input_fingerprint != input_fingerprint {
                bail!("reusable baseline input fingerprint is stale; regenerate it");
            }
            if reusable.sealed_test_evaluated {
                bail!("cannot reuse a baseline artifact that evaluated the sealed test");
            }
            if reusable.evaluation_plan != evaluation_plan {
                bail!("reusable baseline uses a different rolling evaluation plan");
            }
            if reusable.baseline.config_toml != baseline_toml {
                bail!("reusable baseline uses a different default config");
            }
            reusable.baseline.clone()
        } else {
            Self::evaluate_config_on_rolling_folds(
                paths,
                baseline_config,
                baseline_label,
                &evaluation_plan,
                top,
            )?
        };
        let candidate = Self::evaluate_config_on_rolling_folds(
            paths,
            candidate_config,
            candidate_label,
            &evaluation_plan,
            top,
        )?;
        let baseline_outcomes = baseline
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        let candidate_outcomes = candidate
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        let comparison = PairedDifference::all_game_penalized(
            &baseline_outcomes,
            &candidate_outcomes,
            7.0,
            BootstrapConfig::default(),
        )?;
        ensure_rolling_source_identity(paths, &input_fingerprint)?;
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        Ok(RollingComparisonArtifact {
            schema_version: 3,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            evaluation_plan,
            sealed_test_evaluated: false,
            code_revision,
            code_dirty,
            baseline,
            candidate,
            candidate_minus_baseline: comparison,
        })
    }

    pub fn freeze_predictive_candidate(
        paths: &ProjectPaths,
        config_path: &Path,
        comparison_path: &Path,
    ) -> Result<FrozenPredictiveCandidate> {
        let config = PriorConfig::load(config_path)?;
        let config_toml =
            toml::to_string_pretty(&config).context("serialize frozen candidate config")?;
        let comparison_bytes = fs::read(comparison_path)
            .with_context(|| format!("failed to read {}", comparison_path.display()))?;
        let comparison: RollingComparisonArtifact = serde_json::from_slice(&comparison_bytes)
            .with_context(|| format!("failed to parse {}", comparison_path.display()))?;
        comparison.validate_identity()?;
        let input_fingerprint = rolling_source_identity(paths)?;
        let evaluation_plan =
            canonical_development_evaluation_plan(paths, "freezing predictive candidate")?;
        if comparison.input_fingerprint != input_fingerprint
            || comparison.evaluation_plan != evaluation_plan
            || comparison.sealed_test_evaluated
        {
            bail!("development comparison is stale, uses another plan, or touched the sealed test");
        }
        if comparison.candidate.config_toml != config_toml {
            bail!("frozen config does not match the comparison candidate");
        }
        let candidate_failures = comparison.candidate.aggregate.unsolved_games
            + comparison.candidate.aggregate.coverage_gaps;
        if comparison.candidate.aggregate.scheduled_games == 0
            || comparison.candidate.aggregate.modeled_games
                != comparison.candidate.aggregate.scheduled_games
            || candidate_failures != 0
        {
            bail!("candidate cannot be frozen without full coverage and zero failures");
        }
        if comparison.candidate_minus_baseline.candidate_minus_baseline >= 0.0
            || comparison.candidate_minus_baseline.ci95.upper >= 0.0
        {
            bail!(
                "candidate cannot be frozen unless its paired development interval is entirely below zero"
            );
        }
        let development_comparison_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-development-comparison-v1",
            &comparison_bytes,
        );
        let config_fingerprint = comparison.candidate.config_fingerprint.clone();
        let evaluation_artifact_policy = "artifact_free".to_string();
        let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-frozen-candidate-v1");
        hash.field(input_fingerprint.as_bytes())
            .field(config_fingerprint.as_bytes())
            .field(development_comparison_fingerprint.as_bytes())
            .field(comparison.candidate.label.as_bytes())
            .field(evaluation_artifact_policy.as_bytes())
            .field(
                &serde_json::to_vec(&evaluation_plan)
                    .context("serialize frozen evaluation plan identity")?,
            )
            .field(
                &serde_json::to_vec(&comparison.candidate.aggregate)
                    .context("serialize frozen development metrics identity")?,
            )
            .field(
                &serde_json::to_vec(&comparison.candidate_minus_baseline)
                    .context("serialize frozen paired difference identity")?,
            );
        let frozen = FrozenPredictiveCandidate {
            schema_version: 1,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            input_fingerprint,
            freeze_fingerprint: hash.finish_tagged(),
            evaluation_plan,
            config_toml,
            config_fingerprint,
            candidate_label: comparison.candidate.label,
            development_comparison_fingerprint,
            development_metrics: comparison.candidate.aggregate,
            development_paired_difference: comparison.candidate_minus_baseline,
            evaluation_artifact_policy,
            sealed_test_evaluated: false,
        };
        frozen.validate_identity()?;
        Ok(frozen)
    }

    pub fn evaluate_frozen_candidate_on_sealed_test(
        paths: &ProjectPaths,
        frozen: &FrozenPredictiveCandidate,
        output_path: &Path,
    ) -> Result<SealedTestReport> {
        frozen.validate_identity()?;
        if rolling_source_identity(paths)? != frozen.input_fingerprint {
            bail!(
                "source, executable, or data changed after candidate freeze; the sealed test remains closed"
            );
        }
        let plan = canonical_development_evaluation_plan(paths, "opening the sealed final test")?;
        if plan != frozen.evaluation_plan {
            bail!("evaluation plan changed after candidate freeze");
        }
        let output_path = if output_path.is_absolute() {
            output_path.to_path_buf()
        } else {
            paths.root.join(output_path)
        };
        let marker_path = paths
            .root
            .join("benchmarks/predictive/sealed-test-once.json");
        if output_path.exists() || marker_path.exists() {
            bail!(
                "sealed test has already been started or completed; marker={} output={}",
                marker_path.display(),
                output_path.display()
            );
        }
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        if let Some(parent) = marker_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let relative_output = output_path
            .strip_prefix(&paths.root)
            .unwrap_or(&output_path)
            .to_string_lossy()
            .replace('\\', "/");
        let mut marker = SealedTestMarker {
            schema_version: 1,
            freeze_fingerprint: frozen.freeze_fingerprint.clone(),
            output_path: relative_output,
            status: "started_irreversible".to_string(),
        };
        crate::atomic_file::atomic_write(
            &marker_path,
            &serde_json::to_vec_pretty(&marker).context("serialize sealed-test marker")?,
        )?;

        let config: PriorConfig =
            toml::from_str(&frozen.config_toml).context("parse frozen candidate config")?;
        let solver = Self::from_paths_with_settings(
            paths,
            &config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let report = solver.backtest_detailed_with_book_usage(
            plan.sealed_test.start,
            plan.sealed_test.end,
            5,
            PredictiveBookUsage::None,
        )?;
        let games = report
            .runs
            .iter()
            .map(|run| ExperimentGameResult {
                target: run.target.clone(),
                outcome: if run.steps.is_empty() {
                    GameOutcome::coverage_gap(run.date)
                } else if run.solved {
                    GameOutcome::solved(run.date, run.steps.len())
                } else {
                    GameOutcome::unsolved(run.date, run.steps.len())
                },
                path: run.steps.iter().map(|step| step.guess.clone()).collect(),
            })
            .collect::<Vec<_>>();
        let outcomes = games.iter().map(|game| game.outcome).collect::<Vec<_>>();
        let metrics = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let prior_observations = solver
            .history_dates
            .iter()
            .filter(|entry| {
                entry.print_date >= plan.sealed_test.start
                    && entry.print_date <= plan.sealed_test.end
            })
            .filter_map(|entry| {
                solver
                    .initial_prior_metrics(&entry.solution, entry.print_date)
                    .map(|metrics| RankedProbabilityObservation {
                        target_rank: metrics.target_rank,
                        top_probability: metrics.top_probability,
                        top_prediction_correct: metrics.top_prediction_correct,
                    })
            })
            .collect::<Vec<_>>();
        let sealed = SealedTestReport {
            schema_version: 1,
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            freeze_fingerprint: frozen.freeze_fingerprint.clone(),
            input_fingerprint: frozen.input_fingerprint.clone(),
            config_fingerprint: frozen.config_fingerprint.clone(),
            evaluation_plan: plan,
            evaluation_artifact_policy: frozen.evaluation_artifact_policy.clone(),
            sealed_test_evaluated: true,
            evaluated_once: true,
            metrics,
            prior_evidence: (!prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            execution: Self::execution_telemetry(&report.runs),
            games,
            latency_p95_ms: solver
                .benchmark_predictive_latency(default_diagnostic_suite()?.latency.evidence_runs)?,
        };
        crate::atomic_file::atomic_write(
            &output_path,
            &serde_json::to_vec_pretty(&sealed).context("serialize sealed-test report")?,
        )?;
        marker.status = "completed".to_string();
        crate::atomic_file::atomic_write(
            &marker_path,
            &serde_json::to_vec_pretty(&marker).context("serialize sealed-test marker")?,
        )?;
        Ok(sealed)
    }

    pub fn render_rolling_comparison_markdown(
        comparisons: &[RollingComparisonArtifact],
    ) -> Result<String> {
        let first = comparisons
            .first()
            .ok_or_else(|| anyhow!("at least one rolling comparison is required"))?;
        for comparison in comparisons {
            comparison.validate_identity()?;
        }
        if comparisons.iter().any(|comparison| {
            comparison.sealed_test_evaluated
                || comparison.evaluation_plan != first.evaluation_plan
                || comparison.baseline.config_toml != first.baseline.config_toml
                || comparison.baseline.aggregate != first.baseline.aggregate
        }) {
            bail!("rolling comparisons must share one development plan and default baseline");
        }
        let baseline = &first.baseline;
        let mut output = String::new();
        output.push_str("<!-- BEGIN GENERATED ROLLING EVIDENCE -->\n");
        output.push_str("### Rolling-origin promotion guard\n\n");
        output.push_str(&format!(
            "Across {} non-overlapping development folds ({} scheduled games), the sealed test was **not** evaluated. Coverage gaps and six-guess failures are hard constraints before mean score.\n\n",
            first.evaluation_plan.folds.len(),
            baseline.aggregate.scheduled_games
        ));
        output.push_str("| Configuration | Solved | All-game mean | Delta vs default | W/T/L | Latency p95 | Guard decision |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n");
        output.push_str(&format!(
            "| `current_default` | {}/{} | {:.4} [{:.4}, {:.4}] | reference | -- | {:.2} ms | retained |\n",
            baseline.aggregate.solved_games,
            baseline.aggregate.scheduled_games,
            baseline.aggregate.all_game_penalized_mean_guesses,
            baseline.aggregate.all_game_penalized_mean_guesses_ci95.lower,
            baseline.aggregate.all_game_penalized_mean_guesses_ci95.upper,
            baseline.latency_p95_ms,
        ));
        for comparison in comparisons {
            let candidate = &comparison.candidate;
            let paired = comparison.candidate_minus_baseline;
            let baseline_failures =
                baseline.aggregate.unsolved_games + baseline.aggregate.coverage_gaps;
            let candidate_failures =
                candidate.aggregate.unsolved_games + candidate.aggregate.coverage_gaps;
            let decision = if candidate_failures > baseline_failures {
                "rejected: added failures"
            } else if paired.ci95.upper < 0.0 {
                "eligible on solve quality"
            } else if paired.candidate_minus_baseline < 0.0 {
                "not promoted: improvement uncertain"
            } else {
                "rejected: no solve-quality gain"
            };
            output.push_str(&format!(
                "| `{}` | {}/{} | {:.4} [{:.4}, {:.4}] | {:+.4} [{:+.4}, {:+.4}] | {}/{}/{} | {:.2} ms | {} |\n",
                candidate.label,
                candidate.aggregate.solved_games,
                candidate.aggregate.scheduled_games,
                candidate.aggregate.all_game_penalized_mean_guesses,
                candidate.aggregate.all_game_penalized_mean_guesses_ci95.lower,
                candidate.aggregate.all_game_penalized_mean_guesses_ci95.upper,
                paired.candidate_minus_baseline,
                paired.ci95.lower,
                paired.ci95.upper,
                paired.candidate_wins,
                paired.ties,
                paired.baseline_wins,
                candidate.latency_p95_ms,
                decision,
            ));
        }
        output.push_str("\n| Configuration | Prior top-1/3/5 | Confidence ECE | Search steps P/L/XE/X | Recovery/fallback steps |\n");
        output.push_str("| --- | ---: | ---: | ---: | ---: |\n");
        for evidence in std::iter::once(baseline)
            .chain(comparisons.iter().map(|comparison| &comparison.candidate))
        {
            let prior = evidence.prior_evidence.as_ref();
            let telemetry = &evidence.execution;
            output.push_str(&format!(
                "| `{}` | {} | {} | {}/{}/{}/{} | {}/{} |\n",
                evidence.label,
                prior.map_or_else(
                    || "n/a".to_string(),
                    |metrics| format!(
                        "{:.1}%/{:.1}%/{:.1}%",
                        metrics.top_1_recall * 100.0,
                        metrics.top_3_recall * 100.0,
                        metrics.top_5_recall * 100.0
                    )
                ),
                prior.map_or_else(
                    || "n/a".to_string(),
                    |metrics| format!(
                        "{:.4} [{:.4}, {:.4}]",
                        metrics.expected_calibration_error,
                        metrics.expected_calibration_error_ci95.lower,
                        metrics.expected_calibration_error_ci95.upper
                    )
                ),
                telemetry.proxy_steps,
                telemetry.lookahead_steps,
                telemetry.escalated_exact_steps,
                telemetry.exact_steps,
                telemetry.strict_recovery_steps
                    + telemetry.uniform_recovery_steps
                    + telemetry.epsilon_repair_steps,
                telemetry.dormant_fallback_steps,
            ));
        }
        output.push_str("\nDevelopment decisions:\n\n");
        for comparison in comparisons {
            let candidate = &comparison.candidate;
            let paired = comparison.candidate_minus_baseline;
            let baseline_failures =
                baseline.aggregate.unsolved_games + baseline.aggregate.coverage_gaps;
            let candidate_failures =
                candidate.aggregate.unsolved_games + candidate.aggregate.coverage_gaps;
            let explanation = if candidate_failures > baseline_failures {
                format!(
                    "rejected because it added {} failure(s)",
                    candidate_failures - baseline_failures
                )
            } else if paired.ci95.upper < 0.0 {
                "eligible on solve quality because the paired interval is entirely below zero"
                    .to_string()
            } else if paired.candidate_minus_baseline < 0.0 {
                "retained as a development finalist, not promoted, because the observed improvement's paired interval includes zero"
                    .to_string()
            } else {
                "rejected because it did not improve solve quality".to_string()
            };
            output.push_str(&format!("- `{}` is {}.\n", candidate.label, explanation));
        }
        output.push_str(
            "\nThis comparison did not access the sealed window; the release summary records its subsequent once-only evaluation.\n",
        );
        output.push_str("<!-- END GENERATED ROLLING EVIDENCE -->\n");
        Ok(output)
    }

    fn evaluate_config_on_rolling_folds(
        paths: &ProjectPaths,
        config: &PriorConfig,
        label: &str,
        evaluation_plan: &EvaluationPlan,
        top: usize,
    ) -> Result<RollingConfigEvidence> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let config_toml = toml::to_string_pretty(config)
            .context("failed to serialize rolling comparison config")?;
        let source_identity = rolling_source_identity(paths)?;
        let checkpoint_path = rolling_checkpoint_path(paths, label, &config_toml, &source_identity);
        let mut checkpoint = if checkpoint_path.exists() {
            let raw = fs::read(&checkpoint_path)
                .with_context(|| format!("failed to read {}", checkpoint_path.display()))?;
            let checkpoint: RollingEvaluationCheckpoint = serde_json::from_slice(&raw)
                .with_context(|| format!("failed to parse {}", checkpoint_path.display()))?;
            if checkpoint.schema_version != 1
                || checkpoint.source_identity != source_identity
                || checkpoint.evaluation_plan != *evaluation_plan
                || checkpoint.label != label
                || checkpoint.config_toml != config_toml
            {
                bail!(
                    "rolling checkpoint {} does not match the current source/config/plan; remove the rebuildable checkpoint and retry",
                    checkpoint_path.display()
                );
            }
            checkpoint
        } else {
            RollingEvaluationCheckpoint {
                schema_version: 1,
                source_identity,
                evaluation_plan: evaluation_plan.clone(),
                label: label.to_string(),
                config_toml: config_toml.clone(),
                folds: Vec::with_capacity(evaluation_plan.folds.len()),
                games: Vec::new(),
                prior_observations: Vec::new(),
                execution: ExecutionTelemetry::default(),
            }
        };
        let completed_folds = checkpoint
            .folds
            .iter()
            .map(|fold| fold.fold_index)
            .collect::<HashSet<_>>();
        for fold in &evaluation_plan.folds {
            if completed_folds.contains(&fold.index) {
                eprintln!(
                    "rolling-compare label={} fold={}/{} resumed_from_checkpoint=true",
                    label,
                    fold.index + 1,
                    evaluation_plan.folds.len()
                );
                continue;
            }
            let started = Instant::now();
            let report = solver.backtest_detailed_with_book_usage(
                fold.validation.start,
                fold.validation.end,
                top,
                PredictiveBookUsage::None,
            )?;
            checkpoint.folds.push(RollingFoldEvidence {
                fold_index: fold.index,
                validation: fold.validation,
                metrics: report.summary.canonical.clone(),
            });
            checkpoint
                .games
                .extend(report.runs.iter().map(|run| ExperimentGameResult {
                    target: run.target.clone(),
                    outcome: if run.steps.is_empty() {
                        GameOutcome::coverage_gap(run.date)
                    } else if run.solved {
                        GameOutcome::solved(run.date, run.steps.len())
                    } else {
                        GameOutcome::unsolved(run.date, run.steps.len())
                    },
                    path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                }));
            merge_execution_telemetry(
                &mut checkpoint.execution,
                &Self::execution_telemetry(&report.runs),
            );
            for entry in solver.history_dates.iter().filter(|entry| {
                entry.print_date >= fold.validation.start && entry.print_date <= fold.validation.end
            }) {
                if let Some(metrics) =
                    solver.initial_prior_metrics(&entry.solution, entry.print_date)
                {
                    checkpoint
                        .prior_observations
                        .push(RankedProbabilityObservation {
                            target_rank: metrics.target_rank,
                            top_probability: metrics.top_probability,
                            top_prediction_correct: metrics.top_prediction_correct,
                        });
                }
            }
            checkpoint.folds.sort_by_key(|fold| fold.fold_index);
            checkpoint.games.sort_by_key(|game| game.outcome.date);
            crate::atomic_file::atomic_write(
                &checkpoint_path,
                &serde_json::to_vec_pretty(&checkpoint)
                    .context("failed to serialize rolling checkpoint")?,
            )?;
            eprintln!(
                "rolling-compare label={} fold={}/{} validation={}..{} all_game_mean={:.4} failures={} elapsed_ms={}",
                label,
                fold.index + 1,
                evaluation_plan.folds.len(),
                fold.validation.start,
                fold.validation.end,
                report.summary.canonical.all_game_penalized_mean_guesses,
                report.summary.canonical.unsolved_games + report.summary.canonical.coverage_gaps,
                started.elapsed().as_millis()
            );
        }
        if checkpoint.folds.len() != evaluation_plan.folds.len() {
            bail!("rolling evaluation ended before every planned fold completed");
        }
        checkpoint.games.sort_by_key(|game| game.outcome.date);
        let outcomes = checkpoint
            .games
            .iter()
            .map(|game| game.outcome)
            .collect::<Vec<_>>();
        let aggregate = summarize_predictive_outcomes(&outcomes, 7.0, BootstrapConfig::default())?;
        let failure_penalty_sensitivity = [6.0, 7.0, 8.0]
            .into_iter()
            .map(|penalty_guesses| {
                let metrics = summarize_predictive_outcomes(
                    &outcomes,
                    penalty_guesses,
                    BootstrapConfig::default(),
                )?;
                Ok(FailurePenaltyEvidence {
                    penalty_guesses,
                    all_game_mean_guesses: metrics.all_game_penalized_mean_guesses,
                    ci95: metrics.all_game_penalized_mean_guesses_ci95,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let config_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-rolling-config-v1",
            config_toml.as_bytes(),
        );
        Ok(RollingConfigEvidence {
            label: label.to_string(),
            config_toml,
            config_fingerprint,
            folds: checkpoint.folds,
            aggregate,
            prior_evidence: (!checkpoint.prior_observations.is_empty())
                .then(|| {
                    summarize_ranked_probability_observations(
                        &checkpoint.prior_observations,
                        10,
                        BootstrapConfig::default(),
                    )
                })
                .transpose()?,
            execution: checkpoint.execution,
            failure_penalty_sensitivity,
            games: checkpoint.games,
            latency_p95_ms: solver
                .benchmark_predictive_latency(default_diagnostic_suite()?.latency.evidence_runs)?,
        })
    }

    pub fn parse_observations(
        guesses: &[String],
        feedbacks: &[String],
    ) -> Result<Vec<(String, u8)>> {
        if guesses.len() != feedbacks.len() {
            bail!("--guess and --feedback must appear the same number of times");
        }

        guesses
            .iter()
            .zip(feedbacks)
            .map(|(guess, feedback)| {
                Ok((guess.trim().to_ascii_lowercase(), parse_feedback(feedback)?))
            })
            .collect()
    }

    pub fn latest_history_range(paths: &ProjectPaths) -> Result<Option<(NaiveDate, NaiveDate)>> {
        let history = read_history_jsonl(&paths.raw_history)?;
        Ok(history
            .first()
            .zip(history.last())
            .map(|(first, last)| (first.print_date, last.print_date)))
    }

    pub fn pattern_table_bytes(&self) -> usize {
        self.pattern_table.bytes_len()
    }

    pub fn has_guess(&self, guess: &str) -> bool {
        self.guess_index.contains_key(&guess.to_ascii_lowercase())
    }

    pub fn build_predictive_opener_cache(
        &self,
        as_of: NaiveDate,
    ) -> Result<PredictiveOpenerBuildSummary> {
        let offline = self.offline_book_solver()?;
        let (window_start, window_end, targets) =
            offline.recent_history_targets_for_books(as_of)?;
        let holdout = offline.previous_history_targets_for_books(window_start)?;
        let state = offline.initial_state(as_of);
        let candidates = offline
            .suggestion_batch_internal(
                &state,
                offline.config.session_opener_pool.max(1),
                Some(PredictiveContext {
                    as_of,
                    observations: &[],
                }),
                PredictiveBookUsage::None,
            )?
            .suggestions;
        let selected = offline
            .select_validated_opener(
                as_of,
                &candidates,
                &targets,
                holdout.as_ref().map(|(_, _, entries)| entries.as_slice()),
                default_diagnostic_suite()?.book_build.forced_suggestion_top,
            )?
            .ok_or_else(|| anyhow!("missing predictive opener candidate"))?;
        let opener = selected.word.clone();
        let artifact = PredictiveOpenerArtifact {
            identity: self.predictive_book_identity(as_of),
            opener: opener.clone(),
            search_window_start: window_start,
            search_window_end: window_end,
            games: selected.primary.games,
            four_guess_games: selected.primary.four_guess_games,
            average_guesses: selected.primary.average_guesses,
            failures: selected.primary.failures,
            holdout_window_start: holdout.as_ref().map(|(start, _, _)| *start),
            holdout_window_end: holdout.as_ref().map(|(_, end, _)| *end),
            holdout_games: selected.holdout.as_ref().map_or(0, |eval| eval.games),
            holdout_four_guess_games: selected
                .holdout
                .as_ref()
                .map_or(0, |eval| eval.four_guess_games),
            holdout_average_guesses: selected
                .holdout
                .as_ref()
                .map_or(0.0, |eval| eval.average_guesses),
            holdout_failures: selected.holdout.as_ref().map_or(0, |eval| eval.failures),
            proxy_cost: None,
            lookahead_cost: None,
            exact_cost: None,
        };
        let path = self.opener_artifact_path(as_of);
        write_predictive_artifact(&path, &artifact)?;
        Ok(PredictiveOpenerBuildSummary {
            path,
            opener: artifact.opener,
            as_of,
            config_fingerprint: artifact.identity.config_fingerprint,
            games: artifact.games,
            four_guess_games: artifact.four_guess_games,
            average_guesses: artifact.average_guesses,
            failures: artifact.failures,
            holdout_games: artifact.holdout_games,
            holdout_four_guess_games: artifact.holdout_four_guess_games,
            holdout_average_guesses: artifact.holdout_average_guesses,
            holdout_failures: artifact.holdout_failures,
        })
    }

    pub fn build_predictive_reply_book(
        &self,
        as_of: NaiveDate,
    ) -> Result<PredictiveReplyBuildSummary> {
        let opener_artifact = self
            .load_predictive_opener_artifact(as_of)?
            .ok_or_else(|| anyhow!("build the predictive opener cache first"))?;
        let opener_index = self
            .guess_index
            .get(&opener_artifact.opener)
            .copied()
            .ok_or_else(|| anyhow!("cached opener is not in the current guess list"))?;
        let offline = self.offline_book_solver()?;
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let root = offline.initial_state(as_of);
        let mut seen_patterns = HashSet::new();
        let mut replies = Vec::new();
        let reply_candidate_limit = offline.config.session_reply_pool;
        let forced_suggestion_top = default_diagnostic_suite()?.book_build.forced_suggestion_top;

        for answer_index in &root.surviving {
            let pattern = offline.answer_pattern(opener_index, *answer_index);
            if pattern == ALL_GREEN_PATTERN || !seen_patterns.insert(pattern) {
                continue;
            }
            let mut child = root.clone();
            offline.apply_feedback(&mut child, &opener_artifact.opener, pattern)?;
            if child.surviving.len() <= 1 {
                continue;
            }
            let scoped_targets = targets
                .iter()
                .filter(|(_, target)| score_guess(&opener_artifact.opener, target) == pattern)
                .cloned()
                .collect::<Vec<_>>();
            if scoped_targets.is_empty() {
                continue;
            }
            let observation = [(opener_artifact.opener.clone(), pattern)];
            let batch = offline.suggestion_batch_internal(
                &child,
                reply_candidate_limit,
                Some(PredictiveContext {
                    as_of,
                    observations: &observation,
                }),
                PredictiveBookUsage::None,
            )?;
            let mut best_reply: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
            for suggestion in batch.suggestions.into_iter().take(reply_candidate_limit) {
                let guess_index = offline
                    .guess_index
                    .get(&suggestion.word)
                    .copied()
                    .ok_or_else(|| anyhow!("missing reply guess {}", suggestion.word))?;
                let evaluation = offline.evaluate_forced_reply(
                    &opener_artifact.opener,
                    pattern,
                    &scoped_targets,
                    guess_index,
                    forced_suggestion_top,
                )?;
                if best_reply.as_ref().is_none_or(|(_, current)| {
                    compare_forced_openers(&evaluation, current, &offline.guesses)
                        == std::cmp::Ordering::Less
                }) {
                    best_reply = Some((suggestion, evaluation));
                }
            }
            if let Some((reply, _)) = best_reply {
                let reply_word = reply.word.clone();
                let reply_index = offline
                    .guess_index
                    .get(&reply_word)
                    .copied()
                    .ok_or_else(|| anyhow!("missing reply guess {}", reply_word))?;
                let mut seen_second_patterns = HashSet::new();
                let mut grandchild = child.clone();
                let mut third_replies = Vec::new();
                for target_index in &child.surviving {
                    let second_feedback = offline.answer_pattern(reply_index, *target_index);
                    if second_feedback == ALL_GREEN_PATTERN
                        || !seen_second_patterns.insert(second_feedback)
                    {
                        continue;
                    }
                    grandchild.clone_from(&child);
                    offline.apply_feedback(&mut grandchild, &reply_word, second_feedback)?;
                    if grandchild.surviving.len() <= 1 {
                        continue;
                    }
                    let grand_targets = scoped_targets
                        .iter()
                        .filter(|(_, target)| score_guess(&reply_word, target) == second_feedback)
                        .cloned()
                        .collect::<Vec<_>>();
                    if grand_targets.is_empty() {
                        continue;
                    }
                    let grand_observations = [
                        (opener_artifact.opener.clone(), pattern),
                        (reply_word.clone(), second_feedback),
                    ];
                    let grand_batch = offline.suggestion_batch_internal(
                        &grandchild,
                        reply_candidate_limit,
                        Some(PredictiveContext {
                            as_of,
                            observations: &grand_observations,
                        }),
                        PredictiveBookUsage::None,
                    )?;
                    let mut best_third: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
                    for suggestion in grand_batch
                        .suggestions
                        .into_iter()
                        .take(reply_candidate_limit)
                    {
                        let guess_index = offline
                            .guess_index
                            .get(&suggestion.word)
                            .copied()
                            .ok_or_else(|| anyhow!("missing third guess {}", suggestion.word))?;
                        let evaluation = offline.evaluate_forced_continuation(
                            &[opener_artifact.opener.clone(), reply_word.clone()],
                            &grand_targets,
                            guess_index,
                            5,
                        )?;
                        if best_third.as_ref().is_none_or(|(_, current)| {
                            compare_forced_openers(&evaluation, current, &offline.guesses)
                                == std::cmp::Ordering::Less
                        }) {
                            best_third = Some((suggestion, evaluation));
                        }
                    }
                    if let Some((third, _)) = best_third {
                        third_replies.push(PredictiveThirdReplyEntry {
                            second_feedback_pattern: second_feedback,
                            reply: third.word,
                            surviving_answers: grandchild.surviving.len(),
                            proxy_cost: third.proxy_cost,
                            lookahead_cost: third.lookahead_cost,
                            exact_cost: third.exact_cost,
                        });
                    }
                }
                third_replies.sort_by(|left, right| {
                    left.second_feedback_pattern
                        .cmp(&right.second_feedback_pattern)
                });
                replies.push(PredictiveReplyEntry {
                    feedback_pattern: pattern,
                    reply: reply_word,
                    surviving_answers: child.surviving.len(),
                    proxy_cost: reply.proxy_cost,
                    lookahead_cost: reply.lookahead_cost,
                    exact_cost: reply.exact_cost,
                    third_replies,
                });
            }
        }
        replies.sort_by(|left, right| left.feedback_pattern.cmp(&right.feedback_pattern));
        let artifact = PredictiveReplyBookArtifact {
            identity: self.predictive_book_identity(as_of),
            opener: opener_artifact.opener.clone(),
            replies,
        };
        let path = self.reply_book_artifact_path(as_of);
        write_predictive_artifact(&path, &artifact)?;
        let third_reply_count = artifact
            .replies
            .iter()
            .map(|entry| entry.third_replies.len())
            .sum();
        Ok(PredictiveReplyBuildSummary {
            path,
            opener: artifact.opener,
            reply_count: artifact.replies.len(),
            third_reply_count,
            as_of,
            config_fingerprint: artifact.identity.config_fingerprint,
        })
    }

    pub fn predictive_ablation_report(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<Vec<PredictiveAblationResult>> {
        let registry = predictive_parameter_registry(config);
        let matrix = PredictiveExperimentMatrix::parse_json(include_str!(
            "../../config/experiments/predictive-ablations.json"
        ))?;
        let mut rows = Vec::with_capacity(matrix.profiles.len());
        for profile in matrix.profiles {
            let candidate = profile.apply(&registry, config)?;
            let book_usage = match profile.artifact_mode {
                ExperimentArtifactMode::Disabled => PredictiveBookUsage::None,
                ExperimentArtifactMode::DiskOnly => PredictiveBookUsage::DiskOnly,
            };
            let solver = Self::from_paths_with_settings(
                paths,
                &candidate,
                profile.weight_mode,
                profile.model_variant,
            )?;
            rows.push(PredictiveAblationResult {
                label: profile.id,
                result: solver.experiment_report_with_book_usage(from, to, top, book_usage)?,
            });
        }
        Ok(rows)
    }

    pub fn build_proxy_calibration_set(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<Vec<ProxyCalibrationRow>> {
        let started = Instant::now();
        let emit_progress = |message: String| {
            eprintln!("{message}");
            let _ = std::io::stderr().flush();
        };
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no games found in the requested calibration range");
        }

        let mut rows = Vec::new();
        let total_games = games.len();
        emit_progress(format!(
            "fit-proxy-weights phase=calibration-start games={} from={} to={} elapsed_s=0.0",
            total_games, from, to
        ));
        for (game_index, entry) in games.into_iter().enumerate() {
            let date = entry.print_date;
            let as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot calibrate before launch date"))?;
            let target = entry.solution.to_ascii_lowercase();
            let mut state = self.initial_state(as_of);
            if !state
                .surviving
                .iter()
                .any(|index| self.answers[*index].word == target)
            {
                continue;
            }

            let mut observations = Vec::new();
            let mut step_index = 0usize;
            let game_started = Instant::now();
            while step_index < PROXY_CALIBRATION_MAX_STEPS
                && state.surviving.len() > self.config.large_state_split_threshold
            {
                if game_started.elapsed().as_secs_f64() > PROXY_CALIBRATION_MAX_GAME_SECONDS {
                    emit_progress(format!(
                        "fit-proxy-weights phase=calibration-skip game={}/{} date={} reason=budget rows={} elapsed_s={:.1}",
                        game_index + 1,
                        total_games,
                        date,
                        rows.len(),
                        started.elapsed().as_secs_f64(),
                    ));
                    break;
                }
                let mut metrics = self.score_guess_metrics_for_subset(
                    &state.surviving,
                    &state.weights,
                    &self.exact_small_state_table,
                );
                let known_absent_mask = known_absent_letter_mask(&observations);
                for metric in &mut metrics {
                    metric.known_absent_letter_hits =
                        count_masked_letters(&self.guesses[metric.guess_index], known_absent_mask);
                    metric.large_state_score = proxy_row_score_from_weights(
                        &self.config.proxy_weights,
                        ProxyRowStats::from_metric(metric),
                    );
                }
                metrics.sort_by(|left, right| {
                    compare_guess_metrics_for_state(left, right, &self.guesses, true)
                });

                let state_id = format!("{date}:{step_index}");
                let candidate_limit =
                    if state.surviving.len() <= PROXY_CALIBRATION_MAX_SURVIVORS_FOR_FORCED_ROWS {
                        PROXY_CALIBRATION_MAX_CANDIDATES_PER_STATE.min(metrics.len())
                    } else {
                        0
                    };
                if candidate_limit == 0 {
                    emit_progress(format!(
                        "fit-proxy-weights phase=calibration-step game={}/{} date={} step={} survivors={} candidates=0 reason=survivor-cap elapsed_s={:.1}",
                        game_index + 1,
                        total_games,
                        date,
                        step_index,
                        state.surviving.len(),
                        started.elapsed().as_secs_f64(),
                    ));
                } else {
                    emit_progress(format!(
                        "fit-proxy-weights phase=calibration-step game={}/{} date={} step={} survivors={} candidates={} elapsed_s={:.1}",
                        game_index + 1,
                        total_games,
                        date,
                        step_index,
                        state.surviving.len(),
                        candidate_limit,
                        started.elapsed().as_secs_f64(),
                    ));
                }
                for metric in metrics.iter().take(candidate_limit) {
                    if game_started.elapsed().as_secs_f64() > PROXY_CALIBRATION_MAX_GAME_SECONDS {
                        emit_progress(format!(
                            "fit-proxy-weights phase=calibration-skip game={}/{} date={} reason=budget rows={} elapsed_s={:.1}",
                            game_index + 1,
                            total_games,
                            date,
                            rows.len(),
                            started.elapsed().as_secs_f64(),
                        ));
                        break;
                    }
                    let guess = self.guesses[metric.guess_index].clone();
                    let mut forced = observations.clone();
                    forced.push((guess.clone(), 0));
                    let run =
                        self.solve_target_with_forced_prefix(&target, as_of, date, &forced, 3)?;
                    let realized_cost = if run.solved {
                        run.steps.len().saturating_sub(observations.len()) as f64
                    } else {
                        7.0
                    };
                    rows.push(ProxyCalibrationRow {
                        state_id: state_id.clone(),
                        date,
                        step_index,
                        surviving_answers: state.surviving.len(),
                        guess,
                        entropy: metric.entropy,
                        largest_non_green_bucket_mass: metric.largest_non_green_bucket_mass,
                        worst_non_green_bucket_size: metric.worst_non_green_bucket_size,
                        high_mass_ambiguous_bucket_count: metric.high_mass_ambiguous_bucket_count,
                        proxy_cost: metric.proxy_cost,
                        solve_probability: metric.solve_probability,
                        posterior_answer_probability: metric.posterior_answer_probability,
                        smoothness_penalty: metric.smoothness_penalty,
                        known_absent_letter_hits: metric.known_absent_letter_hits,
                        large_non_green_bucket_count: metric.large_non_green_bucket_count,
                        dangerous_mass_bucket_count: metric.dangerous_mass_bucket_count,
                        non_green_mass_in_large_buckets: metric.non_green_mass_in_large_buckets,
                        realized_cost,
                    });
                }

                let chosen = metrics
                    .first()
                    .ok_or_else(|| anyhow!("missing top calibration guess"))?;
                let guess = self.guesses[chosen.guess_index].clone();
                let feedback = score_guess(&guess, &target);
                observations.push((guess.clone(), feedback));
                if feedback == ALL_GREEN_PATTERN {
                    break;
                }
                self.apply_feedback(&mut state, &guess, feedback)?;
                step_index += 1;
            }

            if game_index < 3 || (game_index + 1) % 10 == 0 || game_index + 1 == total_games {
                emit_progress(format!(
                    "fit-proxy-weights phase=calibration games={}/{} rows={} elapsed_s={:.1}",
                    game_index + 1,
                    total_games,
                    rows.len(),
                    started.elapsed().as_secs_f64(),
                ));
            }
        }
        Ok(rows)
    }

    pub(super) fn snapshot_suggestion(suggestion: &Suggestion) -> SuggestionSnapshot {
        SuggestionSnapshot {
            word: suggestion.word.clone(),
            force_in_two: suggestion.force_in_two,
            worst_non_green_bucket_size: suggestion.worst_non_green_bucket_size,
            largest_non_green_bucket_mass: suggestion.largest_non_green_bucket_mass,
            large_non_green_bucket_count: suggestion.large_non_green_bucket_count,
            dangerous_mass_bucket_count: suggestion.dangerous_mass_bucket_count,
            non_green_mass_in_large_buckets: suggestion.non_green_mass_in_large_buckets,
            proxy_cost: suggestion.proxy_cost,
            lookahead_cost: suggestion.lookahead_cost,
            exact_cost: suggestion.exact_cost,
        }
    }

    pub(super) fn assess_state_danger(
        &self,
        state: &SolveState,
        metrics: &[GuessMetrics],
    ) -> StateDangerAssessment {
        self.assess_subset_danger(
            &state.surviving,
            &state.weights,
            state.total_weight,
            metrics,
        )
    }

    pub(super) fn assess_subset_danger(
        &self,
        subset: &[usize],
        weights: &[f64],
        total_weight: f64,
        metrics: &[GuessMetrics],
    ) -> StateDangerAssessment {
        if metrics.is_empty() || total_weight <= 0.0 {
            return StateDangerAssessment {
                danger_score: 0.0,
                dangerous_lookahead: false,
                dangerous_exact: false,
            };
        }

        let mut posterior = subset
            .iter()
            .map(|index| weights[*index] / total_weight)
            .collect::<Vec<_>>();
        posterior.sort_by(|left, right| right.total_cmp(left));
        let top_concentration = posterior
            .iter()
            .take(self.config.danger_posterior_window)
            .sum::<f64>();
        let best = metrics[0];
        let top_window = metrics
            .iter()
            .take(self.config.danger_candidate_window)
            .copied()
            .collect::<Vec<_>>();
        let disagreement = top_window.iter().skip(1).any(|metric| {
            metric.force_in_two != best.force_in_two
                || (metric.largest_non_green_bucket_mass - best.largest_non_green_bucket_mass).abs()
                    >= self.config.danger_mass_disagreement_threshold
                || metric
                    .worst_non_green_bucket_size
                    .abs_diff(best.worst_non_green_bucket_size)
                    >= self.config.danger_size_disagreement_threshold
        });
        let worst_bucket_ratio =
            best.worst_non_green_bucket_size as f64 / subset.len().max(1) as f64;
        let ambiguous_bucket_pressure = (best.high_mass_ambiguous_bucket_count as f64
            / self.config.danger_ambiguity_saturation_count as f64)
            .min(1.0);
        let weight_sum = self.config.danger_top_concentration_w
            + self.config.danger_bucket_mass_w
            + self.config.danger_bucket_ratio_w
            + self.config.danger_ambiguous_w
            + self.config.danger_disagreement_w;
        let danger_score = ((self.config.danger_top_concentration_w * top_concentration)
            + (self.config.danger_bucket_mass_w * best.largest_non_green_bucket_mass)
            + (self.config.danger_bucket_ratio_w * worst_bucket_ratio)
            + (self.config.danger_ambiguous_w * ambiguous_bucket_pressure)
            + if disagreement {
                self.config.danger_disagreement_w
            } else {
                0.0
            })
            / weight_sum;
        StateDangerAssessment {
            danger_score,
            dangerous_lookahead: danger_score >= self.config.danger_lookahead_threshold,
            dangerous_exact: danger_score >= self.config.danger_exact_threshold,
        }
    }

    pub(super) fn regime_mix(runs: &[DetailedSolveRun]) -> (f64, f64, f64, f64) {
        let mut proxy_steps = 0usize;
        let mut lookahead_steps = 0usize;
        let mut escalated_exact_steps = 0usize;
        let mut exact_steps = 0usize;
        let mut total_steps = 0usize;

        for run in runs {
            for step in &run.steps {
                total_steps += 1;
                match step.regime_used {
                    PredictiveRegime::Proxy => proxy_steps += 1,
                    PredictiveRegime::Lookahead => lookahead_steps += 1,
                    PredictiveRegime::EscalatedExact => escalated_exact_steps += 1,
                    PredictiveRegime::Exact => exact_steps += 1,
                }
            }
        }

        if total_steps == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let divisor = total_steps as f64;
        (
            proxy_steps as f64 / divisor,
            lookahead_steps as f64 / divisor,
            escalated_exact_steps as f64 / divisor,
            exact_steps as f64 / divisor,
        )
    }

    pub(super) fn execution_telemetry(runs: &[DetailedSolveRun]) -> ExecutionTelemetry {
        let mut telemetry = ExecutionTelemetry::default();
        for step in runs.iter().flat_map(|run| &run.steps) {
            telemetry.total_steps += 1;
            match step.regime_used {
                PredictiveRegime::Proxy => telemetry.proxy_steps += 1,
                PredictiveRegime::Lookahead => telemetry.lookahead_steps += 1,
                PredictiveRegime::EscalatedExact => telemetry.escalated_exact_steps += 1,
                PredictiveRegime::Exact => telemetry.exact_steps += 1,
            }
            telemetry.danger_escalated_steps += usize::from(step.danger_escalated);
            telemetry.dormant_fallback_steps += usize::from(step.fallback_active);
            match step.recovery_mode_used {
                Some(RecoveryMode::Strict) => telemetry.strict_recovery_steps += 1,
                Some(RecoveryMode::UniformOverSupport) => telemetry.uniform_recovery_steps += 1,
                Some(RecoveryMode::EpsilonRepair) => telemetry.epsilon_repair_steps += 1,
                None => {}
            }
            match step.promotion_source {
                Some(PredictivePromotionSource::ExactDateOpenerArtifact) => {
                    telemetry.exact_date_opener_artifact_hits += 1;
                }
                Some(PredictivePromotionSource::RecentOpenerArtifact) => {
                    telemetry.recent_opener_artifact_hits += 1;
                }
                Some(PredictivePromotionSource::ReplyBook) => telemetry.reply_book_hits += 1,
                Some(
                    PredictivePromotionSource::SessionRootFallback
                    | PredictivePromotionSource::SessionReplyFallback
                    | PredictivePromotionSource::SessionThirdFallback,
                ) => telemetry.session_fallback_hits += 1,
                None => {}
            }
        }
        telemetry
    }

    pub(super) fn select_hard_case_targets(
        &self,
        as_of: NaiveDate,
        top: usize,
        spec: &crate::experiments::HardCaseDiagnosticSpec,
    ) -> Result<Vec<(String, String)>> {
        let state = self.initial_state(as_of);
        let weighted_answers = state
            .surviving
            .iter()
            .map(|answer_index| {
                (
                    *answer_index,
                    self.answers[*answer_index].word.clone(),
                    state.weights[*answer_index],
                )
            })
            .collect::<Vec<_>>();
        let repeated_letters = weighted_answers
            .iter()
            .find(|(_, word, _)| has_repeated_letters(word))
            .map(|(_, word, _)| word.clone());
        let dense_cluster = weighted_answers
            .iter()
            .max_by_key(|(answer_index, _, _)| {
                weighted_answers
                    .iter()
                    .filter(|(other_index, _, _)| {
                        *answer_index != *other_index
                            && hamming_distance(
                                &self.answers[*answer_index].word,
                                &self.answers[*other_index].word,
                            ) <= spec.maximum_cluster_hamming_distance
                    })
                    .count()
            })
            .map(|(_, word, _)| word.clone());
        let low_prior_outlier = weighted_answers
            .iter()
            .filter(|(_, _, weight)| *weight > 0.0)
            .min_by(|left, right| left.2.total_cmp(&right.2))
            .map(|(_, word, _)| word.clone());
        let high_posterior_trap = {
            let mut ranked = weighted_answers.clone();
            ranked.sort_by(|left, right| right.2.total_cmp(&left.2));
            ranked
                .iter()
                .take(spec.top_posterior_scan)
                .filter_map(|(answer_index, word, weight)| {
                    let cluster_mass = ranked
                        .iter()
                        .take(spec.top_posterior_scan)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= spec.maximum_cluster_hamming_distance
                        })
                        .map(|(_, _, other_weight)| *other_weight)
                        .sum::<f64>();
                    let neighbors = ranked
                        .iter()
                        .take(spec.top_posterior_scan)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= spec.maximum_cluster_hamming_distance
                        })
                        .count();
                    (neighbors >= spec.minimum_trap_neighbors)
                        .then_some((cluster_mass + *weight, word.clone()))
                })
                .max_by(|left, right| left.0.total_cmp(&right.0))
                .map(|(_, word)| word)
        };

        let opener = self
            .suggestions(&state, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("missing predictive opener"))?;
        let mut non_answer_splitter_needed = None;
        let mut candidate_answers = weighted_answers;
        candidate_answers.sort_by(|left, right| left.2.total_cmp(&right.2));
        for (_, target, _) in candidate_answers
            .into_iter()
            .take(spec.low_prior_splitter_scan)
        {
            let feedback = score_guess(&opener.word, &target);
            if feedback == ALL_GREEN_PATTERN {
                continue;
            }
            let mut child_state = state.clone();
            self.apply_feedback(&mut child_state, &opener.word, feedback)?;
            if child_state.surviving.len() <= 1 {
                continue;
            }
            let reply = self
                .suggestions(&child_state, top.max(1))?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("missing predictive reply"))?;
            let surviving_words = child_state
                .surviving
                .iter()
                .map(|index| self.answers[*index].word.as_str())
                .collect::<HashSet<_>>();
            if !surviving_words.contains(reply.word.as_str()) {
                non_answer_splitter_needed = Some(target);
                break;
            }
        }

        let mut selected = Vec::new();
        for (label, target) in [
            ("repeated_letters", repeated_letters),
            ("dense_cluster", dense_cluster),
            ("low_prior_outlier", low_prior_outlier),
            ("non_answer_splitter_needed", non_answer_splitter_needed),
            ("high_posterior_trap", high_posterior_trap),
        ] {
            if let Some(target) = target
                && (label == "high_posterior_trap"
                    || selected
                        .iter()
                        .all(|(_, existing): &(String, String)| existing != &target))
            {
                selected.push((label.to_string(), target));
            }
        }
        if selected.is_empty() {
            bail!("unable to construct hard-case suite from current model");
        }
        selected.truncate(spec.target_count);
        Ok(selected)
    }

    pub fn run_predictive_study(
        paths: &ProjectPaths,
        base_config: &PriorConfig,
        spec: StudySpec,
        state_path: &Path,
        top: usize,
        cancellation_path: Option<&Path>,
    ) -> Result<StudyRunSummary> {
        spec.validate()?;
        if top == 0 {
            bail!("study top-suggestion count must be positive");
        }
        let evaluation_plan = canonical_development_evaluation_plan(paths, "running a study")?;
        if spec.maximum_validation_folds > evaluation_plan.folds.len() {
            bail!(
                "study requests {} validation folds but the development plan contains only {}",
                spec.maximum_validation_folds,
                evaluation_plan.folds.len()
            );
        }
        let registry = predictive_parameter_registry(base_config);
        let base_config_toml =
            toml::to_string_pretty(base_config).context("failed to serialize study base config")?;
        let registry_json = serde_json::to_vec(&registry)
            .context("failed to serialize study parameter registry")?;
        let registry_fingerprint = crate::identity::digest_bytes_tagged(
            "maybe-wordle-parameter-registry-v6",
            &registry_json,
        );
        let (code_revision, code_dirty) = git_provenance(&paths.root);
        let compute_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let provenance = crate::experiments::StudyProvenance {
            identity_format: crate::identity::IDENTITY_FORMAT.to_string(),
            base_config_toml,
            registry_format_version: registry.format_version,
            registry_fingerprint,
            input_fingerprint: rolling_source_identity(paths)?,
            operating_system: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            compute_threads,
            code_revision,
            code_dirty,
            history_snapshot_start: evaluation_plan.history.start,
            history_snapshot_end: evaluation_plan.history.end,
            development_cutoff: evaluation_plan.development.end,
            top_suggestions: top,
        };
        if spec.strategy == StudySearchStrategy::ModelBased {
            return Self::run_model_based_study(
                paths,
                base_config,
                &registry,
                spec,
                evaluation_plan,
                provenance,
                state_path,
                top,
                cancellation_path,
            );
        }
        let candidates = generate_candidates(&registry, base_config, &spec)?;
        let mut state = if state_path.exists() {
            let state = StudyState::load(state_path)?;
            if state.spec != spec {
                bail!("existing study state does not match the requested study specification");
            }
            if state.evaluation_plan != evaluation_plan {
                bail!("existing study state uses a different evaluation plan");
            }
            if state.provenance != provenance {
                bail!(
                    "existing study state provenance differs from the current base config, registry, source/data snapshot, cutoff, or evaluation settings"
                );
            }
            state
        } else {
            StudyState::new(spec.clone(), evaluation_plan.clone(), provenance.clone())?
        };
        let mut normalized_checkpoint = false;
        for trial in &mut state.trials {
            let Some((generated, _)) = candidates.get(trial.candidate.number) else {
                bail!(
                    "checkpoint contains out-of-range trial {}",
                    trial.candidate.number
                );
            };
            if !trial.candidate.equivalent_to(generated) {
                bail!(
                    "checkpoint trial {} differs from deterministic generation",
                    trial.candidate.number
                );
            }
            let canonical_identity = generated.identity(&spec, &provenance)?;
            if trial.candidate != *generated || trial.identity != canonical_identity {
                trial.candidate = generated.clone();
                trial.identity = canonical_identity;
                normalized_checkpoint = true;
            }
        }
        if normalized_checkpoint {
            state.save(state_path)?;
        }

        let existing = state
            .trials
            .iter()
            .map(|trial| trial.identity.clone())
            .collect::<HashSet<_>>();
        for (candidate, _) in &candidates {
            let identity = candidate.identity(&spec, &provenance)?;
            if !existing.contains(identity.as_str()) {
                state.trials.push(StudyTrial {
                    candidate: candidate.clone(),
                    identity,
                    status: TrialStatus::Pending,
                    measurement: None,
                    reason: None,
                    elapsed_ms: Some(0),
                    pareto_rank: None,
                    hard_constraint_violations: Vec::new(),
                });
            }
        }
        state.trials.sort_by_key(|trial| trial.candidate.number);
        state.save(state_path)?;

        let effective_parallelism = spec.parallelism.min(candidates.len()).max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(compute_threads)
            .stack_size(crate::SOLVER_THREAD_STACK_BYTES)
            .build()
            .context("failed to create the study worker pool")?;
        let shared_state = Arc::new(Mutex::new(state));
        'fidelity: for target_folds in spec.fidelity_schedule() {
            ensure_rolling_source_identity(paths, &provenance.input_fingerprint)?;
            if cancellation_path.is_some_and(Path::exists) {
                break;
            }
            let validation_fold_indices =
                spec.fidelity_fold_indices(evaluation_plan.folds.len(), target_folds)?;
            let runnable = {
                let state = shared_state
                    .lock()
                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                candidates
                    .iter()
                    .filter_map(|(candidate, config)| {
                        let trial = state
                            .trials
                            .iter()
                            .find(|trial| trial.candidate.number == candidate.number)?;
                        let completed_folds = trial
                            .measurement
                            .as_ref()
                            .map_or(0, |measurement| measurement.validation_fold_indices.len());
                        (!matches!(
                            trial.status,
                            TrialStatus::Complete
                                | TrialStatus::Failed
                                | TrialStatus::Rejected
                                | TrialStatus::Pruned
                        ) && completed_folds < target_folds)
                            .then(|| (candidate.clone(), config.clone(), trial.identity.clone()))
                    })
                    .collect::<Vec<_>>()
            };
            for batch in runnable.chunks(effective_parallelism) {
                if cancellation_path.is_some_and(Path::exists) {
                    break 'fidelity;
                }
                let batch_results = pool.install(|| {
                    batch
                        .par_iter()
                        .map(|(_candidate, config, identity)| -> Result<()> {
                        let (initial_measurement, prior_elapsed_ms) = {
                            let mut state = shared_state
                                .lock()
                                .map_err(|_| anyhow!("study state lock poisoned"))?;
                            let trial = state
                                .trials
                                .iter_mut()
                                .find(|trial| trial.identity == *identity)
                                .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                            trial.status = TrialStatus::Running;
                            trial.reason = None;
                            let measurement = trial.measurement.clone().unwrap_or_default();
                            let elapsed_ms = trial.elapsed_ms.unwrap_or_default();
                            state.save(state_path)?;
                            (measurement, elapsed_ms)
                        };
                        let checkpoint_state = Arc::clone(&shared_state);
                        let result = Self::evaluate_study_candidate(
                            StudyEvaluationRequest {
                                paths,
                                config,
                                stage: spec.stage,
                                artifact_namespace: identity,
                                evaluation_plan: &evaluation_plan,
                                top,
                                target_validation_folds: target_folds,
                                validation_fold_indices: &validation_fold_indices,
                                maximum_trial_seconds: spec.maximum_trial_seconds,
                                maximum_memory_mb: spec.maximum_memory_mb,
                                measure_latency: false,
                                measurement: initial_measurement,
                                prior_elapsed_ms,
                                cancellation_path,
                            },
                            |measurement, elapsed_ms| {
                                let mut state = checkpoint_state
                                    .lock()
                                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                                let trial = state
                                    .trials
                                    .iter_mut()
                                    .find(|trial| trial.identity == *identity)
                                    .ok_or_else(|| {
                                        anyhow!("study trial disappeared during evaluation")
                                    })?;
                                trial.status = TrialStatus::Running;
                                trial.measurement = Some(measurement.clone());
                                trial.elapsed_ms = Some(elapsed_ms);
                                state.save(state_path)
                            },
                        );
                        let mut state = shared_state
                            .lock()
                            .map_err(|_| anyhow!("study state lock poisoned"))?;
                        let trial = state
                            .trials
                            .iter_mut()
                            .find(|trial| trial.identity == *identity)
                            .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                        match result {
                            Ok(Some(measurement)) => {
                                trial.status = if target_folds == spec.maximum_validation_folds {
                                    TrialStatus::Complete
                                } else {
                                    TrialStatus::Running
                                };
                                trial.measurement = Some(measurement);
                                trial.reason = (target_folds < spec.maximum_validation_folds)
                                    .then(|| {
                                        format!(
                                            "completed fidelity rung {target_folds}; awaiting promotion"
                                        )
                                    });
                            }
                            Ok(None) => {
                                trial.status = TrialStatus::Running;
                                trial.reason = Some(
                                    "paused by cooperative cancellation file; safe to resume"
                                        .to_string(),
                                );
                            }
                            Err(error) => {
                                trial.status = TrialStatus::Failed;
                                trial.reason = Some(format!("{error:#}"));
                            }
                        }
                        state.save(state_path)
                    })
                        .collect::<Vec<_>>()
                });
                for result in batch_results {
                    result?;
                }
                if cancellation_path.is_some_and(Path::exists) {
                    break 'fidelity;
                }
            }

            if target_folds == spec.maximum_validation_folds
                && !spec.stage.evaluates_prior_only()
                && !cancellation_path.is_some_and(Path::exists)
            {
                // Every parallel fold worker has joined above. Time finalists one at a time so
                // latency is comparable rather than a measurement of study-worker contention.
                for (candidate, config) in &candidates {
                    if cancellation_path.is_some_and(Path::exists) {
                        break;
                    }
                    let identity = candidate.identity(&spec, &provenance)?;
                    let latency_request = {
                        let mut state = shared_state
                            .lock()
                            .map_err(|_| anyhow!("study state lock poisoned"))?;
                        let trial = state
                            .trials
                            .iter_mut()
                            .find(|trial| trial.identity == identity)
                            .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                        let Some(measurement) = trial.measurement.clone() else {
                            continue;
                        };
                        if !needs_serial_study_latency(
                            trial.status,
                            measurement.validation_fold_indices.len(),
                            spec.maximum_validation_folds,
                            measurement.latency_p95_ms.is_some(),
                        ) {
                            continue;
                        }
                        trial.status = TrialStatus::Running;
                        trial.reason =
                            Some("awaiting serialized contention-free latency measurement".into());
                        let prior_elapsed_ms = trial.elapsed_ms.unwrap_or_default();
                        state.save(state_path)?;
                        Some((measurement, prior_elapsed_ms))
                    };
                    let Some((measurement, prior_elapsed_ms)) = latency_request else {
                        continue;
                    };
                    ensure_rolling_source_identity(paths, &provenance.input_fingerprint)?;
                    let result = Self::measure_study_candidate_latency(
                        paths,
                        config,
                        measurement,
                        prior_elapsed_ms,
                        spec.maximum_trial_seconds,
                        spec.maximum_memory_mb,
                    );
                    let mut state = shared_state
                        .lock()
                        .map_err(|_| anyhow!("study state lock poisoned"))?;
                    let trial = state
                        .trials
                        .iter_mut()
                        .find(|trial| trial.identity == identity)
                        .ok_or_else(|| anyhow!("generated study trial is missing"))?;
                    match result {
                        Ok((measurement, elapsed_ms)) => {
                            trial.status = TrialStatus::Complete;
                            trial.measurement = Some(measurement);
                            trial.elapsed_ms = Some(elapsed_ms);
                            trial.reason = None;
                        }
                        Err(error) => {
                            trial.status = TrialStatus::Failed;
                            trial.reason = Some(format!("{error:#}"));
                        }
                    }
                    state.save(state_path)?;
                }
            }

            {
                let mut state = shared_state
                    .lock()
                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                crate::experiments::annotate_trial_outcomes(&mut state.trials, target_folds);
                state.save(state_path)?;
            }

            if target_folds < spec.maximum_validation_folds {
                let mut state = shared_state
                    .lock()
                    .map_err(|_| anyhow!("study state lock poisoned"))?;
                let survivors = crate::experiments::successive_halving_survivors(
                    &state.trials,
                    target_folds,
                    spec.reduction_factor,
                );
                for trial in &mut state.trials {
                    let completed_folds = trial
                        .measurement
                        .as_ref()
                        .map_or(0, |measurement| measurement.validation_fold_indices.len());
                    if completed_folds >= target_folds
                        && !matches!(
                            trial.status,
                            TrialStatus::Failed | TrialStatus::Rejected | TrialStatus::Pruned
                        )
                    {
                        if survivors.contains(&trial.candidate.number) {
                            trial.status = TrialStatus::Running;
                            trial.reason = Some(format!(
                                "promoted after {target_folds} folds by reduction factor {}",
                                spec.reduction_factor
                            ));
                        } else {
                            trial.status = TrialStatus::Pruned;
                            trial.reason = Some(format!(
                                "pruned after {target_folds} folds by reduction factor {}",
                                spec.reduction_factor
                            ));
                        }
                    }
                }
                state.save(state_path)?;
            }
        }

        let state = Arc::try_unwrap(shared_state)
            .map_err(|_| anyhow!("study state still has active worker references"))?
            .into_inner()
            .map_err(|_| anyhow!("study state lock poisoned"))?;
        ensure_rolling_source_identity(paths, &provenance.input_fingerprint)?;

        Self::summarize_study_run(
            state,
            &registry,
            base_config,
            state_path,
            spec.parallelism,
            effective_parallelism,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_model_based_study(
        paths: &ProjectPaths,
        base_config: &PriorConfig,
        registry: &ParameterRegistry,
        spec: StudySpec,
        evaluation_plan: EvaluationPlan,
        provenance: StudyProvenance,
        state_path: &Path,
        top: usize,
        cancellation_path: Option<&Path>,
    ) -> Result<StudyRunSummary> {
        let mut state = if state_path.exists() {
            let state = StudyState::load(state_path)?;
            if state.spec != spec {
                bail!("existing study state does not match the requested study specification");
            }
            if state.evaluation_plan != evaluation_plan {
                bail!("existing study state uses a different evaluation plan");
            }
            if state.provenance != provenance {
                bail!(
                    "existing study state provenance differs from the current base config, registry, source/data snapshot, cutoff, or evaluation settings"
                );
            }
            state
        } else {
            StudyState::new(spec.clone(), evaluation_plan.clone(), provenance.clone())?
        };
        let mut numbers = state
            .trials
            .iter()
            .map(|trial| trial.candidate.number)
            .collect::<Vec<_>>();
        numbers.sort_unstable();
        if numbers.iter().copied().ne(0..numbers.len()) || state.trials.len() > spec.trial_count {
            bail!("model-based checkpoint candidates are not a valid contiguous prefix");
        }
        for trial in &state.trials {
            registry.apply_tunable_values(base_config, &trial.candidate.parameters)?;
        }
        state.save(state_path)?;

        loop {
            ensure_rolling_source_identity(paths, &provenance.input_fingerprint)?;
            if cancellation_path.is_some_and(Path::exists) {
                break;
            }
            let active_number = state
                .trials
                .iter()
                .find(|trial| {
                    !matches!(
                        trial.status,
                        TrialStatus::Complete
                            | TrialStatus::Failed
                            | TrialStatus::Rejected
                            | TrialStatus::Pruned
                    )
                })
                .map(|trial| trial.candidate.number);
            let candidate_number = if let Some(number) = active_number {
                number
            } else if state.trials.len() < spec.trial_count {
                let (candidate, _) = crate::experiments::generate_model_based_candidate(
                    registry,
                    base_config,
                    &spec,
                    &state.trials,
                )?;
                let identity = candidate.identity(&spec, &provenance)?;
                let number = candidate.number;
                state.trials.push(StudyTrial {
                    candidate,
                    identity,
                    status: TrialStatus::Pending,
                    measurement: None,
                    reason: Some(
                        "checkpointed deterministic observation-driven suggestion".to_string(),
                    ),
                    elapsed_ms: Some(0),
                    pareto_rank: None,
                    hard_constraint_violations: Vec::new(),
                });
                state.save(state_path)?;
                number
            } else {
                break;
            };

            let trial = state
                .trials
                .iter()
                .find(|trial| trial.candidate.number == candidate_number)
                .ok_or_else(|| anyhow!("model-based trial disappeared"))?;
            let config = registry
                .apply_tunable_values(base_config, &trial.candidate.parameters)
                .context("failed to apply model-based candidate")?;
            let identity = trial.identity.clone();
            let measurement = trial.measurement.clone().unwrap_or_default();
            let prior_elapsed_ms = trial.elapsed_ms.unwrap_or_default();
            {
                let trial = state
                    .trials
                    .iter_mut()
                    .find(|trial| trial.identity == identity)
                    .ok_or_else(|| anyhow!("model-based trial disappeared"))?;
                trial.status = TrialStatus::Running;
                trial.reason = None;
            }
            state.save(state_path)?;
            let result = Self::evaluate_study_candidate(
                StudyEvaluationRequest {
                    paths,
                    config: &config,
                    stage: spec.stage,
                    artifact_namespace: &identity,
                    evaluation_plan: &evaluation_plan,
                    top,
                    target_validation_folds: spec.maximum_validation_folds,
                    validation_fold_indices: &spec.fidelity_fold_indices(
                        evaluation_plan.folds.len(),
                        spec.maximum_validation_folds,
                    )?,
                    maximum_trial_seconds: spec.maximum_trial_seconds,
                    maximum_memory_mb: spec.maximum_memory_mb,
                    measure_latency: true,
                    measurement,
                    prior_elapsed_ms,
                    cancellation_path,
                },
                |measurement, elapsed_ms| {
                    let trial = state
                        .trials
                        .iter_mut()
                        .find(|trial| trial.identity == identity)
                        .ok_or_else(|| {
                            anyhow!("model-based trial disappeared during checkpoint")
                        })?;
                    trial.status = TrialStatus::Running;
                    trial.measurement = Some(measurement.clone());
                    trial.elapsed_ms = Some(elapsed_ms);
                    state.save(state_path)
                },
            );
            let trial = state
                .trials
                .iter_mut()
                .find(|trial| trial.identity == identity)
                .ok_or_else(|| anyhow!("model-based trial disappeared"))?;
            match result {
                Ok(Some(measurement)) => {
                    trial.status = TrialStatus::Complete;
                    trial.measurement = Some(measurement);
                    trial.reason = Some("completed observation-driven evaluation".to_string());
                }
                Ok(None) => {
                    trial.status = TrialStatus::Running;
                    trial.reason =
                        Some("paused by cooperative cancellation file; safe to resume".to_string());
                }
                Err(error) => {
                    trial.status = TrialStatus::Failed;
                    trial.reason = Some(format!("{error:#}"));
                }
            }
            crate::experiments::annotate_trial_outcomes(
                &mut state.trials,
                spec.maximum_validation_folds,
            );
            state.save(state_path)?;
            if cancellation_path.is_some_and(Path::exists) {
                break;
            }
        }
        ensure_rolling_source_identity(paths, &provenance.input_fingerprint)?;

        Self::summarize_study_run(
            state,
            registry,
            base_config,
            state_path,
            spec.parallelism,
            1,
        )
    }

    fn summarize_study_run(
        state: StudyState,
        registry: &ParameterRegistry,
        base_config: &PriorConfig,
        state_path: &Path,
        requested_parallelism: usize,
        effective_parallelism: usize,
    ) -> Result<StudyRunSummary> {
        let best = state.best_completed();
        let best_trial_number = best.map(|trial| trial.candidate.number);
        let best_measurement = best.and_then(|trial| trial.measurement.clone());
        let best_config = best
            .map(|trial| registry.apply_tunable_values(base_config, &trial.candidate.parameters))
            .transpose()?;
        Ok(StudyRunSummary {
            state_path: state_path.to_path_buf(),
            requested_parallelism,
            effective_parallelism,
            compute_threads: state.provenance.compute_threads,
            completed_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Complete)
                .count(),
            pending_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Pending)
                .count(),
            running_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Running)
                .count(),
            pruned_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Pruned)
                .count(),
            rejected_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Rejected)
                .count(),
            failed_trials: state
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Failed)
                .count(),
            best_trial_number,
            best_measurement,
            best_config,
            sealed_test_evaluated: false,
        })
    }

    fn evaluate_study_candidate<F>(
        request: StudyEvaluationRequest<'_>,
        mut checkpoint: F,
    ) -> Result<Option<StudyMeasurement>>
    where
        F: FnMut(&StudyMeasurement, u64) -> Result<()>,
    {
        let StudyEvaluationRequest {
            paths,
            config,
            stage,
            artifact_namespace,
            evaluation_plan,
            top,
            target_validation_folds,
            validation_fold_indices,
            maximum_trial_seconds,
            maximum_memory_mb,
            measure_latency,
            mut measurement,
            prior_elapsed_ms,
            cancellation_path,
        } = request;
        let started = Instant::now();
        let time_budget_ms = maximum_trial_seconds.saturating_mul(1_000);
        let memory_budget_bytes = maximum_memory_mb.saturating_mul(1024 * 1024);
        let observe_memory = |measurement: &mut StudyMeasurement| -> Result<()> {
            let snapshot = crate::process_memory::process_memory_snapshot().ok_or_else(|| {
                anyhow!(
                    "hard memory budgets are unsupported on this platform; supported platforms are Windows, Linux, and macOS"
                )
            })?;
            measurement.peak_memory_bytes = Some(
                measurement
                    .peak_memory_bytes
                    .unwrap_or_default()
                    .max(snapshot.peak_working_set_bytes),
            );
            if snapshot.peak_working_set_bytes > memory_budget_bytes {
                bail!(
                    "study process peak working set {} MiB exceeded the {} MiB hard budget",
                    snapshot.peak_working_set_bytes.div_ceil(1024 * 1024),
                    maximum_memory_mb
                );
            }
            Ok(())
        };
        if validation_fold_indices.len() != target_validation_folds {
            bail!(
                "study fidelity selected {} folds but target is {target_validation_folds}",
                validation_fold_indices.len()
            );
        }
        let selected_folds = validation_fold_indices
            .iter()
            .map(|index| {
                evaluation_plan
                    .folds
                    .iter()
                    .find(|fold| fold.index == *index)
                    .ok_or_else(|| anyhow!("study selected unknown validation fold {index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        observe_memory(&mut measurement)?;
        for fold in selected_folds {
            if measurement.validation_fold_indices.contains(&fold.index) {
                continue;
            }
            if cancellation_path.is_some_and(Path::exists) {
                return Ok(None);
            }
            observe_memory(&mut measurement)?;
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            if elapsed_ms > time_budget_ms {
                bail!(
                    "study candidate exceeded {} second wall-clock budget before fold {}",
                    maximum_trial_seconds,
                    fold.index
                );
            }

            let mut fold_measurement = StudyMeasurement {
                validation_fold_indices: vec![fold.index],
                ..StudyMeasurement::default()
            };
            if stage.evaluates_prior_only() {
                for entry in solver
                    .history_dates
                    .iter()
                    .filter(|entry| fold.validation.contains(entry.print_date))
                {
                    observe_memory(&mut fold_measurement)?;
                    if cancellation_path.is_some_and(Path::exists) {
                        return Ok(None);
                    }
                    let elapsed_ms = prior_elapsed_ms
                        .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
                    if elapsed_ms > time_budget_ms {
                        bail!(
                            "study candidate exceeded {} second wall-clock budget",
                            maximum_trial_seconds
                        );
                    }
                    fold_measurement.scheduled_games += 1;
                    if let Some(metrics) =
                        solver.initial_prior_metrics(&entry.solution, entry.print_date)
                    {
                        fold_measurement.measured_prior_games += 1;
                        fold_measurement.log_loss_sum += metrics.log_loss;
                        fold_measurement.brier_score_sum += metrics.brier;
                    }
                }
                fold_measurement.coverage_gaps = fold_measurement
                    .scheduled_games
                    .saturating_sub(fold_measurement.measured_prior_games);
            } else {
                let book_usage = if stage.uses_predictive_books() {
                    PredictiveBookUsage::DiskOnly
                } else {
                    PredictiveBookUsage::None
                };
                let fold_solver = if stage.uses_predictive_books() {
                    let mut fold_solver = solver.clone();
                    fold_solver.artifact_dir =
                        study_book_artifact_dir(paths, artifact_namespace, fold.index);
                    fs::create_dir_all(&fold_solver.artifact_dir).with_context(|| {
                        format!(
                            "failed to create study book directory {}",
                            fold_solver.artifact_dir.display()
                        )
                    })?;
                    let last_artifact_cutoff =
                        fold.validation
                            .end
                            .checked_sub_days(Days::new(1))
                            .ok_or_else(|| anyhow!("book-study cutoff underflowed"))?;
                    let mut artifact_cutoff = fold.training.end;
                    loop {
                        fold_solver.build_predictive_opener_cache(artifact_cutoff)?;
                        fold_solver.build_predictive_reply_book(artifact_cutoff)?;
                        observe_memory(&mut fold_measurement)?;
                        if cancellation_path.is_some_and(Path::exists) {
                            return Ok(None);
                        }
                        let elapsed_ms = prior_elapsed_ms.saturating_add(
                            started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                        );
                        if elapsed_ms > time_budget_ms {
                            bail!(
                                "study candidate exceeded {} second wall-clock budget while building fold {} books",
                                maximum_trial_seconds,
                                fold.index
                            );
                        }
                        let Some(next_cutoff) = artifact_cutoff.checked_add_days(Days::new(
                            fold_solver.config.session_artifact_freshness_days as u64,
                        )) else {
                            break;
                        };
                        if next_cutoff > last_artifact_cutoff {
                            break;
                        }
                        artifact_cutoff = next_cutoff;
                    }
                    Some(fold_solver)
                } else {
                    None
                };
                let active_solver = fold_solver.as_ref().unwrap_or(&solver);
                let report = if stage.evaluates_recovery_only() {
                    match active_solver.recovery_backtest_detailed_with_book_usage(
                        fold.validation.start,
                        fold.validation.end,
                        top,
                        book_usage,
                    ) {
                        Ok(report) => Some(report),
                        Err(error)
                            if error
                                .to_string()
                                .contains("no out-of-primary recovery games") =>
                        {
                            None
                        }
                        Err(error) => return Err(error),
                    }
                } else {
                    Some(active_solver.backtest_detailed_with_book_usage(
                        fold.validation.start,
                        fold.validation.end,
                        top,
                        book_usage,
                    )?)
                };
                if let Some(report) = report {
                    let metrics = &report.summary.canonical;
                    fold_measurement.solve_metrics_recorded = true;
                    fold_measurement.scheduled_games = metrics.scheduled_games;
                    fold_measurement.solved_games = metrics.solved_games;
                    fold_measurement.failures = metrics.unsolved_games;
                    fold_measurement.coverage_gaps = metrics.coverage_gaps;
                    fold_measurement.penalized_guess_sum =
                        metrics.all_game_penalized_mean_guesses * metrics.scheduled_games as f64;
                    fold_measurement.solved_guess_sum =
                        metrics.conditional_mean_guesses * metrics.solved_games as f64;
                    for entry in active_solver
                        .history_dates
                        .iter()
                        .filter(|entry| fold.validation.contains(entry.print_date))
                    {
                        if let Some(prior) =
                            active_solver.initial_prior_metrics(&entry.solution, entry.print_date)
                        {
                            fold_measurement.measured_prior_games += 1;
                            fold_measurement.log_loss_sum += prior.log_loss;
                            fold_measurement.brier_score_sum += prior.brier;
                        }
                    }
                }
                observe_memory(&mut fold_measurement)?;
            }
            fold_measurement.refresh_derived();
            measurement.merge_fold(&fold_measurement)?;
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            checkpoint(&measurement, elapsed_ms)?;
        }
        if measurement.scheduled_games == 0 && !stage.evaluates_recovery_only() {
            bail!("no games were evaluated by the requested study stage");
        }
        if !stage.evaluates_prior_only() && measure_latency {
            measurement.latency_p95_ms = Some(
                solver
                    .benchmark_predictive_latency(default_diagnostic_suite()?.latency.study_runs)?,
            );
            observe_memory(&mut measurement)?;
            let elapsed_ms = prior_elapsed_ms
                .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            if elapsed_ms > time_budget_ms {
                bail!(
                    "study candidate exceeded {} second wall-clock budget during latency measurement",
                    maximum_trial_seconds
                );
            }
            checkpoint(&measurement, elapsed_ms)?;
        }
        measurement.refresh_derived();
        let elapsed_ms = prior_elapsed_ms
            .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        checkpoint(&measurement, elapsed_ms)?;
        Ok(Some(measurement))
    }

    fn measure_study_candidate_latency(
        paths: &ProjectPaths,
        config: &PriorConfig,
        mut measurement: StudyMeasurement,
        prior_elapsed_ms: u64,
        maximum_trial_seconds: u64,
        maximum_memory_mb: u64,
    ) -> Result<(StudyMeasurement, u64)> {
        let started = Instant::now();
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        measurement.latency_p95_ms = Some(
            solver.benchmark_predictive_latency(default_diagnostic_suite()?.latency.study_runs)?,
        );
        let snapshot = crate::process_memory::process_memory_snapshot().ok_or_else(|| {
            anyhow!(
                "hard memory budgets are unsupported on this platform; supported platforms are Windows, Linux, and macOS"
            )
        })?;
        measurement.peak_memory_bytes = Some(
            measurement
                .peak_memory_bytes
                .unwrap_or_default()
                .max(snapshot.peak_working_set_bytes),
        );
        if snapshot.peak_working_set_bytes > maximum_memory_mb.saturating_mul(1024 * 1024) {
            bail!(
                "study process peak working set {} MiB exceeded the {} MiB hard budget during latency measurement",
                snapshot.peak_working_set_bytes.div_ceil(1024 * 1024),
                maximum_memory_mb
            );
        }
        let elapsed_ms = prior_elapsed_ms
            .saturating_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        if elapsed_ms > maximum_trial_seconds.saturating_mul(1_000) {
            bail!(
                "study candidate exceeded {} second wall-clock budget during latency measurement",
                maximum_trial_seconds
            );
        }
        measurement.refresh_derived();
        Ok((measurement, elapsed_ms))
    }

    pub fn tune_prior(paths: &ProjectPaths, config: &PriorConfig) -> Result<TunePriorSummary> {
        let evaluation_plan = canonical_development_evaluation_plan(paths, "tune-prior")?;
        let first_fold = evaluation_plan
            .folds
            .first()
            .ok_or_else(|| anyhow!("rolling-origin plan contains no folds"))?;
        let last_fold = evaluation_plan
            .folds
            .last()
            .ok_or_else(|| anyhow!("rolling-origin plan contains no folds"))?;
        let window_start = first_fold.training.start;
        let window_end = last_fold.training.end;
        let validation_start = first_fold.validation.start;
        let validation_end = last_fold.validation.end;
        let test_start = evaluation_plan.sealed_test.start;
        let test_end = evaluation_plan.sealed_test.end;
        let study_state_path = paths.root.join("target/studies/tune-prior-v16.json");
        let study_summary = Self::run_predictive_study(
            paths,
            config,
            StudySpec {
                name: "tune-prior".to_string(),
                stage: StudyStage::Calibration,
                seed: 20_260_315,
                trial_count: 24,
                parallelism: std::thread::available_parallelism()
                    .map_or(1, |count| count.get().min(4)),
                strategy: crate::experiments::StudySearchStrategy::LowDiscrepancy,
                maximum_validation_folds: evaluation_plan.folds.len(),
                initial_validation_folds: evaluation_plan.folds.len().min(3),
                reduction_factor: 3,
                fold_selection: crate::experiments::StudyFoldSelection::NestedTimeSpread,
                maximum_trial_seconds: 7_200,
                maximum_memory_mb: 4_096,
            },
            &study_state_path,
            5,
            None,
        )?;
        let best_prior_config = study_summary.best_config.unwrap_or_else(|| config.clone());

        let validation_current =
            Self::evaluate_tuning_candidate(paths, config, validation_start, validation_end)?;
        let candidate = Self::evaluate_tuning_candidate(
            paths,
            &best_prior_config,
            validation_start,
            validation_end,
        )?;
        let selected_config = if candidate.all_game_penalized_mean_guesses
            < validation_current.all_game_penalized_mean_guesses
            && candidate.failures <= validation_current.failures
            && candidate.coverage_gaps <= validation_current.coverage_gaps
            && candidate.latency_p95_ms
                <= (validation_current.latency_p95_ms * 3.0).max(validation_current.latency_p95_ms)
        {
            candidate.config
        } else {
            config.clone()
        };
        // The sealed final period is intentionally not evaluated by tuning.
        let current = validation_current;
        let best = Self::evaluate_tuning_candidate(
            paths,
            &selected_config,
            validation_start,
            validation_end,
        )?;
        let replacement_toml = toml::to_string_pretty(&best.config)
            .context("failed to serialize selected prior-study config")?;

        Ok(TunePriorSummary {
            evaluation_plan,
            search_window_start: window_start,
            search_window_end: window_end,
            validation_window_start: validation_start,
            validation_window_end: validation_end,
            test_window_start: test_start,
            test_window_end: test_end,
            current,
            best,
            replacement_toml,
        })
    }

    pub fn evaluate_live_config(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<LiveConfigEvaluation> {
        if from > to {
            bail!("live-config evaluation start date cannot be after end date");
        }
        let evaluation_plan =
            canonical_development_evaluation_plan(paths, "evaluating a live config")?;
        if to > evaluation_plan.development.end {
            bail!(
                "live-config range {}..{} reaches the sealed test; development evaluation must end on or before {}",
                from,
                to,
                evaluation_plan.development.end
            );
        }
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let backtest =
            solver.backtest_detailed_with_book_usage(from, to, top, PredictiveBookUsage::None)?;
        let hard_cases = solver.hard_case_report_with_book_usage(top, PredictiveBookUsage::None)?;
        let latency_p95_ms = solver
            .benchmark_predictive_latency(default_diagnostic_suite()?.latency.evaluation_runs)?;
        Ok(LiveConfigEvaluation {
            config: config.clone(),
            predictive_metrics: backtest.summary.canonical.clone(),
            average_guesses: backtest.summary.average_guesses,
            all_game_penalized_mean_guesses: backtest
                .summary
                .canonical
                .all_game_penalized_mean_guesses,
            failures: backtest.summary.failures,
            coverage_gaps: backtest.summary.coverage_gaps,
            latency_p95_ms,
            hard_case_average_guesses: hard_cases.average_guesses,
            hard_case_failures: hard_cases.failures,
        })
    }

    pub fn three_guess_gap_report(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<ThreeGuessGapReport> {
        let base_solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let aggressive_solver =
            base_solver.clone_with_config(aggressive_early_exact_config(config)?);
        let diagnostic = default_diagnostic_suite()?.three_guess_rescue;
        if diagnostic.profile != "aggressive-three-guess" {
            bail!(
                "unsupported three-guess diagnostic profile: {}",
                diagnostic.profile
            );
        }
        let (base_backtest, four_guess_runs) = base_solver.four_guess_runs(from, to, top)?;
        let mut cases = four_guess_runs
            .par_iter()
            .map(|run| {
                let as_of = run
                    .date
                    .checked_sub_days(Days::new(1))
                    .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
                let solver = aggressive_solver.clone();
                let aggressive_run = solver.solve_target_from_state_detailed(
                    &run.target,
                    as_of,
                    run.date,
                    top,
                    PredictiveBookUsage::None,
                )?;
                let best_forced = solver.best_three_guess_attempt_for_target(
                    &run.target,
                    run.date,
                    top,
                    diagnostic.root_candidate_limit,
                    diagnostic.reply_candidate_limit,
                )?;
                let converted_aggressive = aggressive_run.solved && aggressive_run.steps.len() <= 3;
                let converted_targeted = best_forced.solved && best_forced.steps.len() <= 3;
                Ok(ThreeGuessGapCase {
                    target: run.target.clone(),
                    date: run.date,
                    base_guesses: run.steps.len(),
                    aggressive_guesses: aggressive_run.steps.len(),
                    best_forced_guesses: best_forced.steps.len(),
                    converted_by_aggressive: converted_aggressive,
                    converted_by_targeted_search: converted_targeted,
                    base_path: run.steps.iter().map(|step| step.guess.clone()).collect(),
                    aggressive_path: aggressive_run
                        .steps
                        .iter()
                        .map(|step| step.guess.clone())
                        .collect(),
                    best_forced_path: best_forced
                        .steps
                        .iter()
                        .map(|step| step.guess.clone())
                        .collect(),
                })
            })
            .collect::<Vec<Result<ThreeGuessGapCase>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        cases.sort_by(|left, right| {
            left.date
                .cmp(&right.date)
                .then_with(|| left.target.cmp(&right.target))
        });
        let converted_by_aggressive = cases
            .iter()
            .filter(|case| case.converted_by_aggressive)
            .count();
        let converted_by_targeted_search = cases
            .iter()
            .filter(|case| case.converted_by_targeted_search)
            .count();
        let aggressive_four_guess_cases = cases
            .iter()
            .filter(|case| case.aggressive_guesses == 4)
            .count();
        let aggressive_guess_total = cases
            .iter()
            .map(|case| case.aggressive_guesses)
            .sum::<usize>();

        Ok(ThreeGuessGapReport {
            games: base_backtest.summary.games,
            base_average_guesses: base_backtest.summary.average_guesses,
            aggressive_case_average_guesses: if cases.is_empty() {
                0.0
            } else {
                aggressive_guess_total as f64 / cases.len() as f64
            },
            base_four_guess_cases: base_backtest
                .runs
                .iter()
                .filter(|run| run.solved && run.steps.len() == 4)
                .count(),
            aggressive_four_guess_cases,
            converted_by_aggressive,
            converted_by_targeted_search,
            cases,
        })
    }

    pub fn four_guess_opener_report(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
        openers: &[String],
    ) -> Result<FourGuessOpenerReport> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let (_, four_guess_runs) = solver.four_guess_runs(from, to, top)?;
        let targets = four_guess_runs
            .iter()
            .map(|run| (run.date, run.target.clone()))
            .collect::<Vec<_>>();
        let opener_list = if openers.is_empty() {
            default_diagnostic_suite()?
                .default_four_guess_openers
                .into_iter()
                .filter(|opener| solver.has_guess(opener))
                .collect::<Vec<_>>()
        } else {
            openers
                .iter()
                .map(|opener| opener.trim().to_ascii_lowercase())
                .collect::<Vec<_>>()
        };
        for opener in &opener_list {
            if !solver.has_guess(opener) {
                bail!("unknown opener: {}", opener);
            }
        }
        let evaluations = opener_list
            .into_par_iter()
            .map(|opener| solver.evaluate_named_opener_on_targets(&targets, &opener, top))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let mut evaluations = evaluations;
        evaluations.sort_by(|left, right| {
            left.average_guesses
                .total_cmp(&right.average_guesses)
                .then_with(|| right.three_guess_solves.cmp(&left.three_guess_solves))
                .then_with(|| left.failures.cmp(&right.failures))
                .then_with(|| left.opener.cmp(&right.opener))
        });
        Ok(FourGuessOpenerReport {
            games: targets.len(),
            targets: four_guess_runs
                .into_iter()
                .map(|run| FourGuessTarget {
                    target: run.target,
                    date: run.date,
                    base_path: run.steps.into_iter().map(|step| step.guess).collect(),
                })
                .collect(),
            evaluations,
        })
    }

    pub(super) fn initial_prior_metrics(
        &self,
        target: &str,
        date: NaiveDate,
    ) -> Option<PriorMetrics> {
        let as_of = date.checked_sub_days(Days::new(1))?;
        let state = self.initial_state(as_of);
        let target = target.to_ascii_lowercase();
        let target_index = state
            .surviving
            .iter()
            .find(|index| self.answers[**index].word == target)
            .copied()?;

        let target_probability = state.weights[target_index] / state.total_weight;
        let mut ordered = state
            .surviving
            .iter()
            .map(|index| (*index, state.weights[*index] / state.total_weight))
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| right.1.total_cmp(&left.1));
        let target_rank = ordered
            .iter()
            .position(|(index, _)| *index == target_index)
            .map(|rank| rank + 1)?;
        let target_position = ordered
            .iter()
            .position(|(index, _)| *index == target_index)?;
        let probability_score = score_multiclass_probabilities(
            &ordered
                .iter()
                .map(|(_, probability)| *probability)
                .collect::<Vec<_>>(),
            target_position,
        )
        .ok()?;

        Some(PriorMetrics {
            target_probability,
            target_rank,
            log_loss: probability_score.log_loss,
            brier: probability_score.brier,
            top_probability: ordered.first()?.1,
            top_prediction_correct: target_position == 0,
        })
    }

    pub(super) fn benchmark_predictive_latency(&self, runs: usize) -> Result<f64> {
        let run_count = runs.max(1);
        let top = default_diagnostic_suite()?.latency.top_suggestions;
        let state = self.initial_state(Self::today());
        let mut samples = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            let start = Instant::now();
            let _ = self.suggestions(&state, top)?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|left, right| left.total_cmp(right));
        let p95_index = ((samples.len() as f64) * 0.95).ceil() as usize;
        Ok(samples[p95_index.saturating_sub(1)].max(0.0))
    }

    pub(super) fn benchmark_session_fallback_latency(
        &self,
        as_of: NaiveDate,
    ) -> Result<(f64, f64)> {
        let mut benchmark = self.clone();
        benchmark.session_opener_cache = Arc::new(Mutex::new(HashMap::new()));
        benchmark.session_reply_cache = Arc::new(Mutex::new(HashMap::new()));
        benchmark.session_third_cache = Arc::new(Mutex::new(HashMap::new()));

        let cold_started = Instant::now();
        let _ = benchmark.session_root_guess(as_of)?;
        let cold_ms = cold_started.elapsed().as_secs_f64() * 1_000.0;
        let warm_started = Instant::now();
        let _ = benchmark.session_root_guess(as_of)?;
        let warm_ms = warm_started.elapsed().as_secs_f64() * 1_000.0;
        Ok((cold_ms.max(0.0), warm_ms.max(0.0)))
    }

    pub(super) fn four_guess_runs(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        top: usize,
    ) -> Result<(DetailedBacktestReport, Vec<DetailedSolveRun>)> {
        let backtest =
            self.backtest_detailed_with_book_usage(from, to, top, PredictiveBookUsage::None)?;
        let runs = backtest
            .runs
            .iter()
            .filter(|run| run.solved && run.steps.len() == 4)
            .cloned()
            .collect::<Vec<_>>();
        Ok((backtest, runs))
    }

    pub(super) fn best_three_guess_attempt_for_target(
        &self,
        target: &str,
        date: NaiveDate,
        top: usize,
        root_candidate_limit: usize,
        reply_candidate_limit: usize,
    ) -> Result<DetailedSolveRun> {
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let root = self.initial_state(as_of);
        let root_batch = self.suggestion_batch_internal(
            &root,
            root_candidate_limit.max(top),
            Some(PredictiveContext {
                as_of,
                observations: &[],
            }),
            PredictiveBookUsage::None,
        )?;
        let mut best = self.solve_target_from_state_detailed(
            target,
            as_of,
            date,
            top,
            PredictiveBookUsage::None,
        )?;

        for opener in root_batch
            .suggestions
            .iter()
            .take(root_candidate_limit.max(top))
            .map(|suggestion| suggestion.word.clone())
        {
            let opener_run =
                self.solve_target_with_forced_opening(target, as_of, date, &opener, top)?;
            if better_targeted_run(&opener_run, &best) {
                best = opener_run.clone();
            }
            if opener_run.solved && opener_run.steps.len() <= 3 {
                return Ok(opener_run);
            }

            let opener_feedback = score_guess(&opener, target);
            if opener_feedback == ALL_GREEN_PATTERN {
                continue;
            }
            let mut child = root.clone();
            self.apply_feedback(&mut child, &opener, opener_feedback)?;
            let observations = [(opener.clone(), opener_feedback)];
            let reply_batch = self.suggestion_batch_internal(
                &child,
                reply_candidate_limit.max(top),
                Some(PredictiveContext {
                    as_of,
                    observations: &observations,
                }),
                PredictiveBookUsage::None,
            )?;
            for reply in reply_batch
                .suggestions
                .iter()
                .take(reply_candidate_limit.max(top))
                .map(|suggestion| suggestion.word.clone())
            {
                let forced = [(opener.clone(), opener_feedback), (reply, 0)];
                let run =
                    self.solve_target_with_forced_prefix(target, as_of, date, &forced, top)?;
                if better_targeted_run(&run, &best) {
                    best = run.clone();
                }
                if run.solved && run.steps.len() <= 3 {
                    return Ok(run);
                }
            }
        }

        Ok(best)
    }

    pub(super) fn medium_second_guess_coverage(
        &self,
        subset: &[usize],
        weights: &[f64],
        metrics: &[GuessMetrics],
    ) -> Result<FxHashMap<usize, ThreeSolveCoverage>> {
        let limit = metrics.len().min(self.config.second_guess_coverage_pool);
        let total_weight = subset.iter().map(|index| weights[*index]).sum::<f64>();
        let mut memo = FxHashMap::default();
        let mut coverage = FxHashMap::default();
        for metric in metrics.iter().take(limit) {
            coverage.insert(
                metric.guess_index,
                self.three_solve_coverage_for_guess(
                    metric.guess_index,
                    subset,
                    weights,
                    total_weight,
                    &mut memo,
                )?,
            );
        }
        Ok(coverage)
    }

    pub(super) fn three_solve_coverage_for_guess(
        &self,
        guess_index: usize,
        subset: &[usize],
        weights: &[f64],
        total_weight: f64,
        memo: &mut FxHashMap<ExactSubsetKey, bool>,
    ) -> Result<ThreeSolveCoverage> {
        let mut masses = [0.0_f64; PATTERN_SPACE];
        let mut touched = Vec::with_capacity(PATTERN_SPACE);
        let mut buckets = array::from_fn::<_, PATTERN_SPACE, _>(|_| Vec::new());
        for answer_index in subset {
            let pattern = self.answer_pattern(guess_index, *answer_index) as usize;
            if buckets[pattern].is_empty() {
                touched.push(pattern as u8);
            }
            masses[pattern] += weights[*answer_index];
            buckets[pattern].push(*answer_index);
        }
        let mut result = ThreeSolveCoverage::default();
        for pattern in touched {
            if pattern == ALL_GREEN_PATTERN {
                continue;
            }
            let child = &buckets[pattern as usize];
            let covered =
                child.len() <= 1 || self.child_subset_has_force_in_two(child, weights, memo)?;
            if covered {
                if total_weight > 0.0 {
                    result.mass += masses[pattern as usize] / total_weight;
                }
            } else {
                result.uncovered_buckets += 1;
                result.uncovered_answers += child.len();
            }
        }
        Ok(result)
    }

    pub(super) fn child_subset_has_force_in_two(
        &self,
        subset: &[usize],
        weights: &[f64],
        memo: &mut FxHashMap<ExactSubsetKey, bool>,
    ) -> Result<bool> {
        if subset.len() <= 1 {
            return Ok(true);
        }
        if subset.len() > self.config.second_guess_coverage_child_cap {
            return Ok(false);
        }
        let key = ExactSubsetKey::from_sorted_subset(subset);
        if let Some(cached) = memo.get(&key) {
            return Ok(*cached);
        }
        let metrics =
            self.score_guess_metrics_for_subset(subset, weights, &self.exact_small_state_table);
        let result = metrics.iter().any(|metric| metric.force_in_two);
        memo.insert(key, result);
        Ok(result)
    }

    pub(super) fn evaluate_named_opener_on_targets(
        &self,
        targets: &[(NaiveDate, String)],
        opener: &str,
        top: usize,
    ) -> Result<FourGuessOpenerEvaluation> {
        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut failures = 0usize;
        let mut three_guess_solves = 0usize;
        for (date, target) in targets {
            let as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate opener before launch date"))?;
            let run = self.solve_target_with_forced_opening(target, as_of, *date, opener, top)?;
            guess_counts.push(run.steps.len());
            failures += usize::from(!run.solved);
            three_guess_solves += usize::from(run.solved && run.steps.len() <= 3);
        }
        guess_counts.sort_unstable();
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(FourGuessOpenerEvaluation {
            opener: opener.to_string(),
            average_guesses,
            three_guess_solves,
            failures,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
        })
    }

    pub(super) fn evaluate_tuning_candidate(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<TuningEvaluation> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let report = solver.experiment_report(from, to, 5)?;
        let hard_cases = solver.hard_case_report(5)?;
        Ok(TuningEvaluation {
            config: config.clone(),
            average_guesses: report.backtest.average_guesses,
            all_game_penalized_mean_guesses: report
                .backtest
                .canonical
                .all_game_penalized_mean_guesses,
            failures: report.backtest.failures,
            coverage_gaps: report.backtest.coverage_gaps,
            average_log_loss: report.average_log_loss,
            average_target_rank: report.average_target_rank,
            latency_p95_ms: report.latency_p95_ms,
            hard_case_average_guesses: hard_cases.average_guesses,
            hard_case_failures: hard_cases.failures,
            proxy_step_pct: report.proxy_step_pct,
            lookahead_step_pct: report.lookahead_step_pct,
            escalated_exact_step_pct: report.escalated_exact_step_pct,
            exact_step_pct: report.exact_step_pct,
        })
    }

    pub(super) fn offline_book_solver(&self) -> Result<Self> {
        let config = apply_embedded_profile(
            &self.config,
            include_str!("../../config/profiles/offline-book.json"),
        )?;
        Ok(self.clone_with_config(config))
    }

    pub(super) fn clone_with_config(&self, config: PriorConfig) -> Self {
        let mut cloned = self.clone();
        cloned.config = config.clone();
        cloned.exact_small_state_table = SmallStateTable::build(
            config
                .exact_exhaustive_threshold
                .max(config.proxy_small_state_lower_bound_threshold)
                .max(2),
        );
        cloned
    }

    pub(super) fn is_medium_state_lookahead(&self, surviving_answers: usize) -> bool {
        surviving_answers > self.config.exact_threshold
            && surviving_answers <= self.config.medium_state_lookahead_threshold
    }

    pub(super) fn lookahead_candidate_pool_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_lookahead_candidate_pool
        } else {
            self.config.lookahead_candidate_pool
        }
    }

    pub(super) fn lookahead_reply_pool_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_lookahead_reply_pool
        } else {
            self.config.lookahead_reply_pool
        }
    }

    pub(super) fn force_in_two_scan_for_state(&self, surviving_answers: usize) -> usize {
        if self.is_medium_state_lookahead(surviving_answers) {
            self.config.medium_state_force_in_two_scan
        } else {
            self.config.lookahead_root_force_in_two_scan
        }
    }

    pub(super) fn recent_history_targets_for_books(
        &self,
        as_of: NaiveDate,
    ) -> Result<BookTargetWindow> {
        let mut entries = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date <= as_of)
            .collect::<Vec<_>>();
        if entries.is_empty() {
            bail!("run sync-data before building predictive books");
        }
        entries.sort_by_key(|entry| entry.print_date);
        let window_end = entries
            .last()
            .map(|entry| entry.print_date)
            .ok_or_else(|| anyhow!("missing recent history"))?;
        let window_days = self.config.session_window_days.saturating_sub(1) as u64;
        let window_start = window_end
            .checked_sub_days(Days::new(window_days))
            .map_or(entries[0].print_date, |date| {
                date.max(entries[0].print_date)
            });
        let targets = entries
            .into_iter()
            .filter(|entry| entry.print_date >= window_start)
            .map(|entry| (entry.print_date, entry.solution.clone()))
            .collect::<Vec<_>>();
        Ok((window_start, window_end, targets))
    }

    pub(super) fn previous_history_targets_for_books(
        &self,
        current_window_start: NaiveDate,
    ) -> Result<Option<BookTargetWindow>> {
        let mut entries = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date < current_window_start)
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Ok(None);
        }
        entries.sort_by_key(|entry| entry.print_date);
        let holdout_end = entries
            .last()
            .map(|entry| entry.print_date)
            .ok_or_else(|| anyhow!("missing holdout history"))?;
        let window_days = self.config.session_window_days.saturating_sub(1) as u64;
        let holdout_start = holdout_end
            .checked_sub_days(Days::new(window_days))
            .map_or(entries[0].print_date, |date| {
                date.max(entries[0].print_date)
            });
        let targets = entries
            .into_iter()
            .filter(|entry| entry.print_date >= holdout_start)
            .map(|entry| (entry.print_date, entry.solution.clone()))
            .collect::<Vec<_>>();
        Ok(Some((holdout_start, holdout_end, targets)))
    }

    pub(super) fn evaluate_forced_opener(
        &self,
        _as_of: NaiveDate,
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        _top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        let opener = self.guesses[guess_index].clone();
        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut four_guess_games = 0usize;
        let mut failures = 0usize;
        for (date, target) in targets {
            let target_as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate opener before launch date"))?;
            let score =
                self.score_target_with_forced_opening(target, target_as_of, *date, &opener)?;
            if score.guesses >= 4 {
                four_guess_games += 1;
            }
            guess_counts.push(score.guesses);
            if !score.solved {
                failures += 1;
            }
        }
        guess_counts.sort_unstable();
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(ForcedOpenerEvaluation {
            guess_index,
            games: guess_counts.len(),
            four_guess_games,
            average_guesses,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
            failures,
        })
    }

    pub(super) fn evaluate_forced_reply(
        &self,
        opener: &str,
        _opener_feedback: u8,
        targets: &[(NaiveDate, String)],
        reply_guess_index: usize,
        top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        self.evaluate_forced_continuation(&[opener.to_string()], targets, reply_guess_index, top)
    }

    pub(super) fn evaluate_forced_continuation(
        &self,
        forced_prefix: &[String],
        targets: &[(NaiveDate, String)],
        guess_index: usize,
        _top: usize,
    ) -> Result<ForcedOpenerEvaluation> {
        let guess = self.guesses[guess_index].clone();
        let forced_prefix = forced_prefix
            .iter()
            .cloned()
            .map(|word| (word, 0))
            .collect::<Vec<_>>();
        let mut guess_counts = Vec::with_capacity(targets.len());
        let mut failures = 0usize;
        for (date, target) in targets {
            let target_as_of = date
                .checked_sub_days(Days::new(1))
                .ok_or_else(|| anyhow!("cannot evaluate reply before launch date"))?;
            let mut forced = forced_prefix.clone();
            forced.push((guess.clone(), 0));
            let score =
                self.score_target_with_forced_prefix(target, target_as_of, *date, &forced)?;
            guess_counts.push(score.guesses);
            if !score.solved {
                failures += 1;
            }
        }
        guess_counts.sort_unstable();
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        Ok(ForcedOpenerEvaluation {
            guess_index,
            games: guess_counts.len(),
            four_guess_games: guess_counts.iter().filter(|count| **count >= 4).count(),
            average_guesses,
            p95_guesses: guess_counts
                .get(p95_index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
            max_guesses: guess_counts.last().copied().unwrap_or_default(),
            failures,
        })
    }

    pub(super) fn select_validated_opener(
        &self,
        as_of: NaiveDate,
        candidates: &[Suggestion],
        primary_targets: &[(NaiveDate, String)],
        holdout_targets: Option<&[(NaiveDate, String)]>,
        top: usize,
    ) -> Result<Option<ValidatedOpenerEvaluation>> {
        let mut evaluations = candidates
            .par_iter()
            .filter_map(|suggestion| {
                let guess_index = self.guess_index.get(&suggestion.word).copied()?;
                let primary = self
                    .evaluate_forced_opener(as_of, primary_targets, guess_index, top)
                    .ok()?;
                Some(ValidatedOpenerEvaluation {
                    word: suggestion.word.clone(),
                    primary,
                    holdout: None,
                })
            })
            .collect::<Vec<_>>();
        evaluations.sort_by(|left, right| {
            compare_forced_openers(&left.primary, &right.primary, &self.guesses)
        });
        let shortlist_len = if holdout_targets.is_some() {
            self.config
                .session_opener_holdout_shortlist
                .min(evaluations.len())
        } else {
            evaluations.len()
        };
        let mut best: Option<ValidatedOpenerEvaluation> = None;
        for mut evaluation in evaluations.into_iter().take(shortlist_len) {
            if let Some(targets) = holdout_targets {
                evaluation.holdout = self
                    .evaluate_forced_opener(as_of, targets, evaluation.primary.guess_index, top)
                    .ok();
            }
            if best.as_ref().is_none_or(|current| {
                should_replace_forced_opener(
                    &evaluation.primary,
                    evaluation.holdout.as_ref(),
                    &current.primary,
                    current.holdout.as_ref(),
                    &self.guesses,
                )
            }) {
                best = Some(evaluation);
            }
        }
        Ok(best)
    }

    pub(super) fn solve_target_with_forced_opening(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        opener: &str,
        top: usize,
    ) -> Result<DetailedSolveRun> {
        let forced = [(opener.to_string(), 0)];
        self.solve_target_with_forced_prefix(target, as_of, date, &forced, top)
    }

    pub(super) fn score_target_with_forced_opening(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        opener: &str,
    ) -> Result<ForcedSolveScore> {
        let forced = [(opener.to_string(), 0)];
        self.score_target_with_forced_prefix(target, as_of, date, &forced)
    }

    pub(super) fn solve_target_with_forced_prefix(
        &self,
        target: &str,
        as_of: NaiveDate,
        date: NaiveDate,
        forced: &[(String, u8)],
        top: usize,
    ) -> Result<DetailedSolveRun> {
        let target = target.to_ascii_lowercase();
        let mut state = self.initial_state(as_of);
        if !state
            .surviving
            .iter()
            .any(|index| self.answers[*index].word == target)
        {
            return Ok(DetailedSolveRun {
                target,
                date,
                steps: Vec::new(),
                solved: false,
            });
        }

        let mut steps = Vec::new();
        let mut observations = Vec::new();
        for (position, (guess, expected_feedback)) in forced.iter().enumerate() {
            let feedback = score_guess(guess, &target);
            if position == 0 && *expected_feedback != 0 && *expected_feedback != feedback {
                bail!(
                    "forced opener feedback mismatch for {}: expected {}, got {}",
                    guess,
                    format_feedback_letters(*expected_feedback),
                    format_feedback_letters(feedback)
                );
            }
            let surviving_before = state.surviving.len();
            let surviving_after = if feedback == ALL_GREEN_PATTERN {
                1
            } else {
                let mut next_state = state.clone();
                self.apply_feedback(&mut next_state, guess, feedback)?;
                next_state.surviving.len()
            };
            steps.push(DetailedSolveStep {
                guess: guess.clone(),
                feedback,
                surviving_before,
                surviving_after,
                chosen_force_in_two: false,
                alternative_force_in_two: false,
                danger_score: 0.0,
                danger_escalated: false,
                regime_used: PredictiveRegime::Proxy,
                promotion_source: None,
                recovery_mode_used: state.recovery_mode_used,
                fallback_active: state.fallback_active,
                lookahead_pool_base: 0,
                lookahead_pool_size: 0,
                exact_pool_base: 0,
                exact_pool_size: 0,
                root_candidate_count: 0,
                top_suggestions: Vec::new(),
            });
            if feedback == ALL_GREEN_PATTERN {
                return Ok(DetailedSolveRun {
                    target,
                    date,
                    steps,
                    solved: true,
                });
            }
            observations.push((guess.clone(), feedback));
            self.apply_feedback(&mut state, guess, feedback)?;
        }

        while steps.len() < 6 {
            let surviving_before = state.surviving.len();
            let batch = self.suggestion_batch_internal(
                &state,
                top.max(1),
                Some(PredictiveContext {
                    as_of,
                    observations: &observations,
                }),
                PredictiveBookUsage::None,
            )?;
            let chosen = batch
                .suggestions
                .first()
                .ok_or_else(|| anyhow!("solver returned no suggestions"))?
                .clone();
            let feedback = score_guess(&chosen.word, &target);
            let surviving_after = if feedback == ALL_GREEN_PATTERN {
                1
            } else {
                let mut next_state = state.clone();
                self.apply_feedback(&mut next_state, &chosen.word, feedback)?;
                next_state.surviving.len()
            };
            steps.push(DetailedSolveStep {
                guess: chosen.word.clone(),
                feedback,
                surviving_before,
                surviving_after,
                chosen_force_in_two: chosen.force_in_two,
                alternative_force_in_two: batch
                    .suggestions
                    .iter()
                    .skip(1)
                    .any(|suggestion| suggestion.force_in_two),
                danger_score: batch.danger_score,
                danger_escalated: batch.danger_escalated,
                regime_used: batch.regime_used,
                promotion_source: batch.promotion_source,
                recovery_mode_used: state.recovery_mode_used,
                fallback_active: state.fallback_active,
                lookahead_pool_base: batch.lookahead_pool_base,
                lookahead_pool_size: batch.lookahead_pool_size,
                exact_pool_base: batch.exact_pool_base,
                exact_pool_size: batch.exact_pool_size,
                root_candidate_count: batch.root_candidate_count,
                top_suggestions: batch
                    .suggestions
                    .iter()
                    .take(top.max(1))
                    .map(Self::snapshot_suggestion)
                    .collect(),
            });
            if feedback == ALL_GREEN_PATTERN {
                return Ok(DetailedSolveRun {
                    target,
                    date,
                    steps,
                    solved: true,
                });
            }
            observations.push((chosen.word.clone(), feedback));
            self.apply_feedback(&mut state, &chosen.word, feedback)?;
        }

        Ok(DetailedSolveRun {
            target,
            date,
            steps,
            solved: false,
        })
    }

    pub(super) fn score_target_with_forced_prefix(
        &self,
        target: &str,
        as_of: NaiveDate,
        _date: NaiveDate,
        forced: &[(String, u8)],
    ) -> Result<ForcedSolveScore> {
        let target = target.to_ascii_lowercase();
        let mut state = self.initial_state(as_of);
        if !state
            .surviving
            .iter()
            .any(|index| self.answers[*index].word == target)
        {
            return Ok(ForcedSolveScore {
                guesses: 0,
                solved: false,
            });
        }

        let mut guess_count = 0usize;
        let mut observations = Vec::new();
        for (position, (guess, expected_feedback)) in forced.iter().enumerate() {
            let feedback = score_guess(guess, &target);
            if position == 0 && *expected_feedback != 0 && *expected_feedback != feedback {
                bail!(
                    "forced opener feedback mismatch for {}: expected {}, got {}",
                    guess,
                    format_feedback_letters(*expected_feedback),
                    format_feedback_letters(feedback)
                );
            }
            guess_count += 1;
            if feedback == ALL_GREEN_PATTERN {
                return Ok(ForcedSolveScore {
                    guesses: guess_count,
                    solved: true,
                });
            }
            observations.push((guess.clone(), feedback));
            self.apply_feedback(&mut state, guess, feedback)?;
        }

        while guess_count < 6 {
            let batch = self.suggestion_batch_internal(
                &state,
                1,
                Some(PredictiveContext {
                    as_of,
                    observations: &observations,
                }),
                PredictiveBookUsage::None,
            )?;
            let chosen = batch
                .suggestions
                .first()
                .ok_or_else(|| anyhow!("solver returned no suggestions"))?;
            let feedback = score_guess(&chosen.word, &target);
            guess_count += 1;
            if feedback == ALL_GREEN_PATTERN {
                return Ok(ForcedSolveScore {
                    guesses: guess_count,
                    solved: true,
                });
            }
            observations.push((chosen.word.clone(), feedback));
            self.apply_feedback(&mut state, &chosen.word, feedback)?;
        }

        Ok(ForcedSolveScore {
            guesses: guess_count,
            solved: false,
        })
    }
}

fn canonical_development_evaluation_plan(
    paths: &ProjectPaths,
    operation: &str,
) -> Result<EvaluationPlan> {
    let (history_start, history_end) = Solver::latest_history_range(paths)?
        .ok_or_else(|| anyhow!("run sync-data before {operation}"))?;
    let history = DateRange::new(history_start, history_end)?;
    build_rolling_origin_plan(history, rolling_origin_config_for_history(history)?)
}

fn rolling_origin_config_for_history(history: DateRange) -> Result<RollingOriginConfig> {
    if history.days() >= 425 {
        return Ok(RollingOriginConfig::default());
    }
    if history.days() < 3 {
        bail!(
            "history has {} days but development evaluation requires at least 3",
            history.days()
        );
    }
    Ok(RollingOriginConfig {
        minimum_training_days: history.days() - 2,
        validation_days: 1,
        step_days: 1,
        sealed_test_days: 1,
        maximum_folds: 1,
    })
}

fn rolling_checkpoint_path(
    paths: &ProjectPaths,
    label: &str,
    config_toml: &str,
    source_identity: &str,
) -> PathBuf {
    let safe_label = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let fingerprint = rolling_checkpoint_fingerprint(config_toml, source_identity);
    paths.root.join(format!(
        "target/rolling-checkpoints/{safe_label}-{fingerprint}.json"
    ))
}

pub(super) fn rolling_checkpoint_fingerprint(config_toml: &str, source_identity: &str) -> String {
    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-rolling-checkpoint-v2");
    hash.field(config_toml.as_bytes())
        .field(source_identity.as_bytes());
    hash.finish_hex()
}

fn study_book_artifact_dir(
    paths: &ProjectPaths,
    artifact_namespace: &str,
    fold_index: usize,
) -> PathBuf {
    let namespace_hash = crate::identity::digest_bytes_hex(
        "maybe-wordle-study-book-namespace-v2",
        artifact_namespace.as_bytes(),
    );
    paths
        .root
        .join("target/studies/predictive-books")
        .join(format!("trial-{namespace_hash}"))
        .join(format!("fold-{fold_index:02}"))
}

fn evidence_artifact_sizes(paths: &ProjectPaths) -> Result<Vec<EvidenceArtifactSize>> {
    let declared = [
        ("pattern_table", paths.pattern_table.as_path()),
        ("answer_history", paths.derived_answer_history.as_path()),
        ("modeled_answers", paths.derived_modeled_answers.as_path()),
        ("predictive_books", paths.derived_predictive.as_path()),
    ];
    declared
        .into_iter()
        .filter(|(_, path)| path.exists())
        .map(|(name, path)| {
            Ok(EvidenceArtifactSize {
                name: name.to_string(),
                path: path
                    .strip_prefix(&paths.root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .replace('\\', "/"),
                bytes: filesystem_tree_bytes(path)?,
            })
        })
        .collect()
}

fn enforce_evidence_resource_budget(
    started: Instant,
    budget: EvidenceResourceBudget,
) -> Result<crate::process_memory::ProcessMemorySnapshot> {
    let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
    if elapsed_ms > budget.maximum_seconds.saturating_mul(1_000) {
        bail!(
            "evidence generation took {:.3} seconds and exceeded the {} second budget",
            elapsed_ms as f64 / 1_000.0,
            budget.maximum_seconds
        );
    }
    let memory = crate::process_memory::process_memory_snapshot().ok_or_else(|| {
        anyhow!(
            "hard evidence memory budgets are unsupported on this platform; supported platforms are Windows, Linux, and macOS"
        )
    })?;
    let memory_budget_bytes = budget.maximum_memory_mb.saturating_mul(1024 * 1024);
    if memory.peak_working_set_bytes > memory_budget_bytes {
        bail!(
            "evidence peak working set {} MiB exceeded the {} MiB budget",
            memory.peak_working_set_bytes.div_ceil(1024 * 1024),
            budget.maximum_memory_mb
        );
    }
    Ok(memory)
}

fn filesystem_tree_bytes(path: &Path) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect artifact size for {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Ok(0);
    }
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }
    let mut bytes = 0u64;
    for entry in fs::read_dir(path)
        .with_context(|| format!("failed to enumerate artifact directory {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", path.display()))?;
        bytes = bytes.saturating_add(filesystem_tree_bytes(&entry.path())?);
    }
    Ok(bytes)
}

fn rolling_source_identity(paths: &ProjectPaths) -> Result<String> {
    let mut files = Vec::new();
    collect_regular_files(&paths.root.join("src"), &mut files)?;
    collect_regular_files(&paths.root.join("tests"), &mut files)?;
    files.extend([
        paths.root.join("Cargo.toml"),
        paths.root.join("Cargo.lock"),
        paths.raw_history.clone(),
        paths.seed_guesses.clone(),
        paths.seed_answers.clone(),
        paths.seed_reference_answers.clone(),
        paths.manual_additions.clone(),
    ]);
    files.sort();
    files.dedup();

    let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-rolling-inputs-v3");
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    hash.field(b"current_executable");
    hash_identity_file(&mut hash, &executable)?;
    for path in files {
        let relative = path.strip_prefix(&paths.root).unwrap_or(&path);
        let relative = relative.to_string_lossy().replace('\\', "/");
        hash.field(relative.as_bytes());
        if path.is_file() {
            hash.field(&[1]);
            hash_identity_file(&mut hash, &path)?;
        } else {
            hash.field(&[0]);
        }
    }
    Ok(hash.finish_tagged())
}

fn hash_identity_file(hash: &mut crate::identity::CanonicalSha256, path: &Path) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to inspect {}", path.display()))?;
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    hash.field_reader(&mut file, metadata.len())
        .with_context(|| format!("failed to fingerprint {}", path.display()))?;
    Ok(())
}

fn ensure_rolling_source_identity(paths: &ProjectPaths, expected: &str) -> Result<()> {
    if rolling_source_identity(paths)? != expected {
        bail!(
            "source, executable, or data inputs changed during evaluation; discard this run and restart from a consistent snapshot"
        );
    }
    Ok(())
}

fn collect_regular_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to enumerate {}", directory.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", directory.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_regular_files(&path, output)?;
        } else if path.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

fn merge_execution_telemetry(target: &mut ExecutionTelemetry, addition: &ExecutionTelemetry) {
    target.total_steps += addition.total_steps;
    target.proxy_steps += addition.proxy_steps;
    target.lookahead_steps += addition.lookahead_steps;
    target.escalated_exact_steps += addition.escalated_exact_steps;
    target.exact_steps += addition.exact_steps;
    target.danger_escalated_steps += addition.danger_escalated_steps;
    target.strict_recovery_steps += addition.strict_recovery_steps;
    target.uniform_recovery_steps += addition.uniform_recovery_steps;
    target.epsilon_repair_steps += addition.epsilon_repair_steps;
    target.dormant_fallback_steps += addition.dormant_fallback_steps;
    target.exact_date_opener_artifact_hits += addition.exact_date_opener_artifact_hits;
    target.recent_opener_artifact_hits += addition.recent_opener_artifact_hits;
    target.reply_book_hits += addition.reply_book_hits;
    target.session_fallback_hits += addition.session_fallback_hits;
}

fn git_provenance(root: &Path) -> (Option<String>, Option<bool>) {
    let revision = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let dirty = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty());
    (revision, dirty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialized_latency_only_selects_complete_unmeasured_finalists() {
        assert!(needs_serial_study_latency(
            TrialStatus::Complete,
            12,
            12,
            false
        ));
        assert!(!needs_serial_study_latency(
            TrialStatus::Running,
            12,
            12,
            false
        ));
        assert!(!needs_serial_study_latency(
            TrialStatus::Complete,
            11,
            12,
            false
        ));
        assert!(!needs_serial_study_latency(
            TrialStatus::Complete,
            12,
            12,
            true
        ));
    }
}
