use super::*;

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

    pub(super) fn backtest_detailed_with_book_usage(
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
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested backtest range");
        }

        let mut guess_counts = Vec::new();
        let mut failures = 0usize;
        let mut coverage_gaps = 0usize;
        let mut runs = Vec::new();

        for entry in games {
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
            if run.steps.is_empty() {
                coverage_gaps += 1;
                failures += 1;
                runs.push(run);
                continue;
            }
            if !run.solved {
                failures += 1;
            }
            guess_counts.push(run.steps.len());
            runs.push(run);
        }

        guess_counts.sort_unstable();
        let games_played = guess_counts.len() + coverage_gaps;
        let average_guesses = if guess_counts.is_empty() {
            0.0
        } else {
            guess_counts.iter().sum::<usize>() as f64 / guess_counts.len() as f64
        };
        let p95_index = ((guess_counts.len() as f64) * 0.95).ceil() as usize;
        let p95_guesses = guess_counts
            .get(p95_index.saturating_sub(1))
            .copied()
            .unwrap_or_default();
        let max_guesses = guess_counts.last().copied().unwrap_or_default();

        Ok(DetailedBacktestReport {
            summary: BacktestStats {
                games: games_played,
                average_guesses,
                p95_guesses,
                max_guesses,
                failures,
                coverage_gaps,
            },
            runs,
        })
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
        let cases = self.select_hard_case_targets(as_of, top)?;
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
        let games = self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();

        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let detailed = self.backtest_detailed(from, to, top)?;
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

        for entry in games {
            if let Some(metrics) = self.initial_prior_metrics(&entry.solution, entry.print_date) {
                total_log_loss += metrics.log_loss;
                total_brier += metrics.brier;
                total_target_probability += metrics.target_probability;
                total_rank += metrics.target_rank as f64;
                measured += 1;
            }
        }

        let divisor = measured.max(1) as f64;
        Ok(ExperimentResult {
            config_id: format!(
                "et{}-ee{}-cp{}-lt{}-lc{}-lr{}-ls{}",
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
            latency_p95_ms: self.benchmark_predictive_latency(5)?,
            session_fallback_cold_ms: 0.0,
            session_fallback_warm_ms: 0.0,
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
        let offline = self.offline_book_solver();
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
                5,
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
        let offline = self.offline_book_solver();
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let root = offline.initial_state(as_of);
        let mut seen_patterns = HashSet::new();
        let mut replies = Vec::new();

        for answer_index in &root.surviving {
            let pattern = offline.pattern_table.get(opener_index, *answer_index);
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
                REPLY_BOOK_CANDIDATE_LIMIT,
                Some(PredictiveContext {
                    as_of,
                    observations: &observation,
                }),
                PredictiveBookUsage::None,
            )?;
            let mut best_reply: Option<(Suggestion, ForcedOpenerEvaluation)> = None;
            for suggestion in batch
                .suggestions
                .into_iter()
                .take(REPLY_BOOK_CANDIDATE_LIMIT)
            {
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
                    5,
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
                    let second_feedback = offline.pattern_table.get(reply_index, *target_index);
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
                        REPLY_BOOK_CANDIDATE_LIMIT,
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
                        .take(REPLY_BOOK_CANDIDATE_LIMIT)
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
        let mut rows = Vec::new();
        let expanded = |mut candidate: PriorConfig| {
            candidate.lookahead_candidate_pool = candidate.lookahead_candidate_pool.max(150);
            candidate.lookahead_reply_pool = candidate.lookahead_reply_pool.max(50);
            candidate.exact_candidate_pool = candidate.exact_candidate_pool.max(160);
            candidate
        };
        let flattened_50 = flatten_weighted_config(config, 0.50);
        let flattened_25 = flatten_weighted_config(config, 0.25);
        for (label, mode, candidate) in [
            ("uniform_baseline", WeightMode::Uniform, config.clone()),
            (
                "uniform_wide_pools",
                WeightMode::Uniform,
                expanded(config.clone()),
            ),
            (
                "cooldown_baseline",
                WeightMode::CooldownOnly,
                config.clone(),
            ),
            (
                "cooldown_wide_pools",
                WeightMode::CooldownOnly,
                expanded(config.clone()),
            ),
            ("weighted_baseline", WeightMode::Weighted, config.clone()),
            (
                "weighted_wide_pools",
                WeightMode::Weighted,
                expanded(config.clone()),
            ),
            (
                "weighted_flat_50",
                WeightMode::Weighted,
                flattened_50.clone(),
            ),
            (
                "weighted_flat_50_wide_pools",
                WeightMode::Weighted,
                expanded(flattened_50),
            ),
            (
                "weighted_flat_25",
                WeightMode::Weighted,
                flattened_25.clone(),
            ),
            (
                "weighted_flat_25_wide_pools",
                WeightMode::Weighted,
                expanded(flattened_25),
            ),
        ] {
            let solver = Self::from_paths_with_settings(
                paths,
                &candidate,
                mode,
                ModelVariant::SeedPlusHistory,
            )?;
            rows.push(PredictiveAblationResult {
                label: label.to_string(),
                result: solver.experiment_report(from, to, top)?,
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

    pub fn fit_proxy_weights(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<ProxyWeightFitSummary> {
        let started = Instant::now();
        let emit_progress = |message: String| {
            eprintln!("{message}");
            let _ = std::io::stderr().flush();
        };
        emit_progress(format!(
            "fit-proxy-weights phase=start from={} to={} elapsed_s=0.0",
            from, to
        ));
        let rows = self.build_proxy_calibration_set(from, to)?;
        if rows.is_empty() {
            bail!("no large-state calibration rows found in the requested range");
        }
        emit_progress(format!(
            "fit-proxy-weights phase=grouping rows={} elapsed_s={:.1}",
            rows.len(),
            started.elapsed().as_secs_f64(),
        ));

        let mut grouped = std::collections::BTreeMap::<String, Vec<&ProxyCalibrationRow>>::new();
        for row in &rows {
            grouped.entry(row.state_id.clone()).or_default().push(row);
        }
        let groups = grouped.into_iter().collect::<Vec<_>>();
        let split = ((groups.len() as f64) * 0.8).floor() as usize;
        let split = split.clamp(1, groups.len());
        let (train_groups, validation_groups) = groups.split_at(split);

        let evaluate = |weights: &crate::config::ProxyWeights,
                        sample: &[(String, Vec<&ProxyCalibrationRow>)]|
         -> f64 {
            if sample.is_empty() {
                return 0.0;
            }
            sample
                .iter()
                .map(|(_, rows)| {
                    rows.iter()
                        .max_by(|left, right| {
                            let left_score = proxy_row_score_from_weights(
                                weights,
                                ProxyRowStats::from_calibration_row(left),
                            );
                            let right_score = proxy_row_score_from_weights(
                                weights,
                                ProxyRowStats::from_calibration_row(right),
                            );
                            right_score.total_cmp(&left_score)
                        })
                        .map(|row| row.realized_cost)
                        .unwrap_or(7.0)
                })
                .sum::<f64>()
                / sample.len() as f64
        };

        emit_progress(format!(
            "fit-proxy-weights phase=search train_states={} validation_states={} baseline_train_avg={:.4} elapsed_s={:.1}",
            train_groups.len(),
            validation_groups.len(),
            evaluate(&self.config.proxy_weights, train_groups),
            started.elapsed().as_secs_f64(),
        ));

        let mut best = self.config.proxy_weights.clone();
        let mut best_train = evaluate(&best, train_groups);
        macro_rules! search_weight {
            ($field:ident) => {{
                emit_progress(format!(
                    "fit-proxy-weights phase=field name={} current={:.4} train_avg={:.4} elapsed_s={:.1}",
                    stringify!($field),
                    best.$field,
                    best_train,
                    started.elapsed().as_secs_f64(),
                ));
                for _ in 0..2 {
                    let mut improved = false;
                    let base = best.$field;
                    for scale in [0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00] {
                        let candidate_value = (base * scale).max(0.001);
                        if (candidate_value - base).abs() <= 1e-9 {
                            continue;
                        }
                        let mut candidate = best.clone();
                        candidate.$field = candidate_value;
                        let train_score = evaluate(&candidate, train_groups);
                        if train_score + 1e-9 < best_train {
                            best = candidate;
                            best_train = train_score;
                            improved = true;
                            emit_progress(format!(
                                "fit-proxy-weights phase=improved name={} value={:.4} train_avg={:.4} elapsed_s={:.1}",
                                stringify!($field),
                                best.$field,
                                best_train,
                                started.elapsed().as_secs_f64(),
                            ));
                        }
                    }
                    if !improved {
                        break;
                    }
                }
            }};
        }

        search_weight!(entropy_w);
        search_weight!(bucket_mass_w);
        search_weight!(bucket_size_w);
        search_weight!(ambiguous_w);
        search_weight!(proxy_w);
        search_weight!(solve_prob_w);
        search_weight!(posterior_w);
        search_weight!(smoothness_w);
        search_weight!(gray_reuse_w);
        search_weight!(large_bucket_count_w);
        search_weight!(dangerous_mass_count_w);
        search_weight!(large_bucket_mass_w);

        let validation_average = evaluate(&best, validation_groups);
        emit_progress(format!(
            "fit-proxy-weights phase=done train_avg={:.4} validation_avg={:.4} elapsed_s={:.1}",
            best_train,
            validation_average,
            started.elapsed().as_secs_f64(),
        ));
        let replacement_toml = format!(
            concat!(
                "[proxy_weights]\n",
                "entropy_w = {:.4}\n",
                "bucket_mass_w = {:.4}\n",
                "bucket_size_w = {:.4}\n",
                "ambiguous_w = {:.4}\n",
                "proxy_w = {:.4}\n",
                "solve_prob_w = {:.4}\n",
                "posterior_w = {:.4}\n",
                "smoothness_w = {:.4}\n",
                "gray_reuse_w = {:.4}\n",
                "large_bucket_count_w = {:.4}\n",
                "dangerous_mass_count_w = {:.4}\n",
                "large_bucket_mass_w = {:.4}\n",
            ),
            best.entropy_w,
            best.bucket_mass_w,
            best.bucket_size_w,
            best.ambiguous_w,
            best.proxy_w,
            best.solve_prob_w,
            best.posterior_w,
            best.smoothness_w,
            best.gray_reuse_w,
            best.large_bucket_count_w,
            best.dangerous_mass_count_w,
            best.large_bucket_mass_w,
        );

        Ok(ProxyWeightFitSummary {
            row_count: rows.len(),
            state_count: train_groups.len() + validation_groups.len(),
            training_average_guesses: best_train,
            validation_average_guesses: validation_average,
            replacement_toml,
        })
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
        let top_concentration = posterior.iter().take(3).sum::<f64>();
        let best = metrics[0];
        let top_window = metrics.iter().take(3).copied().collect::<Vec<_>>();
        let disagreement = top_window.iter().skip(1).any(|metric| {
            metric.force_in_two != best.force_in_two
                || (metric.largest_non_green_bucket_mass - best.largest_non_green_bucket_mass).abs()
                    >= 0.10
                || metric
                    .worst_non_green_bucket_size
                    .abs_diff(best.worst_non_green_bucket_size)
                    >= 2
        });
        let worst_bucket_ratio =
            best.worst_non_green_bucket_size as f64 / subset.len().max(1) as f64;
        let ambiguous_bucket_pressure =
            (best.high_mass_ambiguous_bucket_count as f64 / 4.0).min(1.0);
        let danger_score = (0.30 * top_concentration)
            + (0.25 * best.largest_non_green_bucket_mass)
            + (0.20 * worst_bucket_ratio)
            + (0.15 * ambiguous_bucket_pressure)
            + if disagreement { 0.10 } else { 0.0 };
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

    pub(super) fn select_hard_case_targets(
        &self,
        as_of: NaiveDate,
        top: usize,
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
                            ) <= 1
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
                .take(96)
                .filter_map(|(answer_index, word, weight)| {
                    let cluster_mass = ranked
                        .iter()
                        .take(96)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= 1
                        })
                        .map(|(_, _, other_weight)| *other_weight)
                        .sum::<f64>();
                    let neighbors = ranked
                        .iter()
                        .take(96)
                        .filter(|(other_index, _, _)| {
                            *other_index != *answer_index
                                && hamming_distance(
                                    &self.answers[*answer_index].word,
                                    &self.answers[*other_index].word,
                                ) <= 1
                        })
                        .count();
                    (neighbors >= 2).then_some((cluster_mass + *weight, word.clone()))
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
        for (_, target, _) in candidate_answers.into_iter().take(128) {
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
        Ok(selected)
    }

    pub fn tune_prior(paths: &ProjectPaths, config: &PriorConfig) -> Result<TunePriorSummary> {
        use std::io::Write;

        let (history_start, history_end) = Self::latest_history_range(paths)?
            .ok_or_else(|| anyhow!("run sync-data before tune-prior"))?;
        let window_end = history_end;
        let window_start = history_end
            .checked_sub_days(Days::new(364))
            .map_or(history_start, |date| date.max(history_start));
        let started = Instant::now();
        eprintln!(
            "tune-prior phase=start search_window={}..{} elapsed_s=0.0",
            window_start, window_end
        );
        let _ = std::io::stderr().flush();
        let mut best_prior_config = config.clone();
        let mut best_prior = Self::evaluate_prior_search_candidate(
            paths,
            &best_prior_config,
            window_start,
            window_end,
        )?;

        macro_rules! search_dimension {
            ($field:ident, $label:literal, $values:expr) => {{
                eprintln!(
                    "tune-prior phase=field name={} current={} elapsed_s={:.1}",
                    $label,
                    best_prior_config.$field,
                    started.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
                loop {
                    let mut improved = false;
                    for value in $values {
                        if best_prior_config.$field == value {
                            continue;
                        }
                        let mut candidate_config = best_prior_config.clone();
                        candidate_config.$field = value;
                        let candidate = Self::evaluate_prior_search_candidate(
                            paths,
                            &candidate_config,
                            window_start,
                            window_end,
                        )?;
                        if Self::better_prior_search_evaluation(&candidate, &best_prior) {
                            best_prior_config = candidate_config;
                            best_prior = candidate;
                            eprintln!(
                                "tune-prior phase=improved name={} value={} log_loss={:.6} target_rank={:.2} target_probability={:.6} elapsed_s={:.1}",
                                $label,
                                best_prior_config.$field,
                                best_prior.average_log_loss,
                                best_prior.average_target_rank,
                                best_prior.average_target_probability,
                                started.elapsed().as_secs_f64()
                            );
                            let _ = std::io::stderr().flush();
                            improved = true;
                        }
                    }
                    if !improved {
                        break;
                    }
                }
            }};
        }

        search_dimension!(base_seed_weight, "base_seed_weight", [0.75, 1.0, 1.25]);
        search_dimension!(
            base_history_only_weight,
            "base_history_only_weight",
            [0.10, 0.20, 0.25, 0.33, 0.50]
        );
        search_dimension!(cooldown_days, "cooldown_days", [90_i64, 120, 180, 240, 365]);
        search_dimension!(cooldown_floor, "cooldown_floor", [0.0, 0.01, 0.02, 0.05]);
        search_dimension!(
            midpoint_days,
            "midpoint_days",
            [365.0, 540.0, 720.0, 900.0, 1080.0]
        );
        search_dimension!(logistic_k, "logistic_k", [0.005, 0.01, 0.015, 0.02]);
        search_dimension!(
            pool_tight_gap_threshold,
            "pool_tight_gap_threshold",
            [0.03, 0.05, 0.07]
        );
        search_dimension!(
            pool_medium_gap_threshold,
            "pool_medium_gap_threshold",
            [0.10, 0.15, 0.20]
        );
        search_dimension!(
            danger_lookahead_threshold,
            "danger_lookahead_threshold",
            [0.52, 0.58, 0.64]
        );
        search_dimension!(
            danger_exact_threshold,
            "danger_exact_threshold",
            [0.68, 0.72, 0.76]
        );
        search_dimension!(
            lookahead_trap_penalty,
            "lookahead_trap_penalty",
            [0.25, 0.35, 0.45]
        );
        search_dimension!(
            lookahead_large_bucket_penalty,
            "lookahead_large_bucket_penalty",
            [0.08, 0.12, 0.16]
        );
        search_dimension!(
            lookahead_dangerous_mass_penalty,
            "lookahead_dangerous_mass_penalty",
            [0.05, 0.08, 0.12]
        );
        search_dimension!(
            lookahead_large_bucket_mass_penalty,
            "lookahead_large_bucket_mass_penalty",
            [0.06, 0.10, 0.14]
        );
        search_dimension!(
            medium_state_lookahead_candidate_pool,
            "medium_state_lookahead_candidate_pool",
            [32, 48, 64]
        );
        search_dimension!(
            medium_state_lookahead_reply_pool,
            "medium_state_lookahead_reply_pool",
            [16, 20, 28]
        );
        search_dimension!(
            medium_state_force_in_two_scan,
            "medium_state_force_in_two_scan",
            [96, 160, 224]
        );

        let validation_start = window_end
            .checked_sub_days(Days::new(6))
            .map_or(window_start, |date| date.max(window_start));
        let current = Self::evaluate_tuning_candidate(paths, config, validation_start, window_end)?;
        let candidate = Self::evaluate_tuning_candidate(
            paths,
            &best_prior_config,
            validation_start,
            window_end,
        )?;
        let best = if candidate.average_guesses < current.average_guesses
            && candidate.failures <= current.failures
            && candidate.hard_case_failures <= current.hard_case_failures
            && candidate.latency_p95_ms
                <= (current.latency_p95_ms * 3.0).max(current.latency_p95_ms)
        {
            candidate
        } else {
            current.clone()
        };
        eprintln!(
            "tune-prior phase=done current_avg_guesses={:.4} best_avg_guesses={:.4} current_latency_p95_ms={:.3} best_latency_p95_ms={:.3} elapsed_s={:.1}",
            current.average_guesses,
            best.average_guesses,
            current.latency_p95_ms,
            best.latency_p95_ms,
            started.elapsed().as_secs_f64()
        );
        let _ = std::io::stderr().flush();

        let replacement_toml = format!(
            concat!(
                "base_seed_weight = {:.2}\n",
                "base_history_only_weight = {:.2}\n",
                "cooldown_days = {}\n",
                "cooldown_floor = {:.2}\n",
                "midpoint_days = {:.1}\n",
                "logistic_k = {:.3}\n",
                "exact_threshold = {}\n",
                "exact_exhaustive_threshold = {}\n",
                "exact_candidate_pool = {}\n",
                "lookahead_threshold = {}\n",
                "medium_state_lookahead_threshold = {}\n",
                "lookahead_candidate_pool = {}\n",
                "medium_state_lookahead_candidate_pool = {}\n",
                "lookahead_reply_pool = {}\n",
                "medium_state_lookahead_reply_pool = {}\n",
                "lookahead_root_force_in_two_scan = {}\n",
                "medium_state_force_in_two_scan = {}\n",
                "large_state_split_threshold = {}\n",
                "pool_tight_gap_threshold = {:.2}\n",
                "pool_medium_gap_threshold = {:.2}\n",
                "danger_lookahead_threshold = {:.2}\n",
                "danger_exact_threshold = {:.2}\n",
                "danger_reply_pool_bonus = {}\n",
                "danger_exact_root_pool = {}\n",
                "danger_exact_survivor_cap = {}\n",
                "lookahead_trap_penalty = {:.2}\n",
                "lookahead_large_bucket_penalty = {:.2}\n",
                "lookahead_dangerous_mass_penalty = {:.2}\n",
                "lookahead_large_bucket_mass_penalty = {:.2}\n",
            ),
            best.config.base_seed_weight,
            best.config.base_history_only_weight,
            best.config.cooldown_days,
            best.config.cooldown_floor,
            best.config.midpoint_days,
            best.config.logistic_k,
            best.config.exact_threshold,
            best.config.exact_exhaustive_threshold,
            best.config.exact_candidate_pool,
            best.config.lookahead_threshold,
            best.config.medium_state_lookahead_threshold,
            best.config.lookahead_candidate_pool,
            best.config.medium_state_lookahead_candidate_pool,
            best.config.lookahead_reply_pool,
            best.config.medium_state_lookahead_reply_pool,
            best.config.lookahead_root_force_in_two_scan,
            best.config.medium_state_force_in_two_scan,
            best.config.large_state_split_threshold,
            best.config.pool_tight_gap_threshold,
            best.config.pool_medium_gap_threshold,
            best.config.danger_lookahead_threshold,
            best.config.danger_exact_threshold,
            best.config.danger_reply_pool_bonus,
            best.config.danger_exact_root_pool,
            best.config.danger_exact_survivor_cap,
            best.config.lookahead_trap_penalty,
            best.config.lookahead_large_bucket_penalty,
            best.config.lookahead_dangerous_mass_penalty,
            best.config.lookahead_large_bucket_mass_penalty,
        );

        Ok(TunePriorSummary {
            search_window_start: window_start,
            search_window_end: window_end,
            validation_window_start: validation_start,
            validation_window_end: window_end,
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
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let backtest =
            solver.backtest_detailed_with_book_usage(from, to, top, PredictiveBookUsage::None)?;
        let hard_cases = solver.hard_case_report_with_book_usage(top, PredictiveBookUsage::None)?;
        let latency_p95_ms = solver.benchmark_predictive_latency(3)?;
        Ok(LiveConfigEvaluation {
            config: config.clone(),
            average_guesses: backtest.summary.average_guesses,
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
            base_solver.clone_with_config(aggressive_early_exact_config(config));
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
                let best_forced =
                    solver.best_three_guess_attempt_for_target(&run.target, run.date, top)?;
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
            [
                "olate", "aiery", "reais", "crane", "slate", "trace", "audio", "tarse",
            ]
            .into_iter()
            .map(str::to_string)
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
        let probability_square_sum = ordered
            .iter()
            .map(|(_, probability)| probability * probability)
            .sum::<f64>();

        Some(PriorMetrics {
            target_probability,
            target_rank,
            log_loss: -(target_probability.max(1e-12)).ln(),
            brier: probability_square_sum - (2.0 * target_probability) + 1.0,
        })
    }

    pub(super) fn benchmark_predictive_latency(&self, runs: usize) -> Result<f64> {
        let run_count = runs.max(1);
        let state = self.initial_state(Self::today());
        let mut samples = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            let start = Instant::now();
            let _ = self.suggestions(&state, 10)?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|left, right| left.total_cmp(right));
        let p95_index = ((samples.len() as f64) * 0.95).ceil() as usize;
        Ok(samples[p95_index.saturating_sub(1)].max(0.0))
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
    ) -> Result<DetailedSolveRun> {
        let as_of = date
            .checked_sub_days(Days::new(1))
            .ok_or_else(|| anyhow!("cannot solve before launch date"))?;
        let root = self.initial_state(as_of);
        let root_batch = self.suggestion_batch_internal(
            &root,
            THREE_GUESS_ROOT_CANDIDATE_LIMIT.max(top),
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
            .take(THREE_GUESS_ROOT_CANDIDATE_LIMIT.max(top))
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
                THREE_GUESS_REPLY_CANDIDATE_LIMIT.max(top),
                Some(PredictiveContext {
                    as_of,
                    observations: &observations,
                }),
                PredictiveBookUsage::None,
            )?;
            for reply in reply_batch
                .suggestions
                .iter()
                .take(THREE_GUESS_REPLY_CANDIDATE_LIMIT.max(top))
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
        let limit = metrics.len().min(
            self.lookahead_candidate_pool_for_state(subset.len())
                .max(MEDIUM_SECOND_GUESS_COVERAGE_POOL),
        );
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
            let pattern = self.pattern_table.get(guess_index, *answer_index) as usize;
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
        if subset.len() > THREE_SOLVE_CHILD_CAP {
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

    pub(super) fn evaluate_prior_search_candidate(
        paths: &ProjectPaths,
        config: &PriorConfig,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<PriorSearchEvaluation> {
        let solver = Self::from_paths_with_settings(
            paths,
            config,
            WeightMode::Weighted,
            ModelVariant::SeedPlusHistory,
        )?;
        let games = solver
            .history_dates
            .iter()
            .filter(|entry| entry.print_date >= from && entry.print_date <= to)
            .collect::<Vec<_>>();
        if games.is_empty() {
            bail!("no games found in the requested experiment range");
        }

        let mut total_log_loss = 0.0;
        let mut total_target_rank = 0.0;
        let mut total_target_probability = 0.0;
        let mut measured = 0usize;
        for entry in games {
            if let Some(metrics) = solver.initial_prior_metrics(&entry.solution, entry.print_date) {
                total_log_loss += metrics.log_loss;
                total_target_rank += metrics.target_rank as f64;
                total_target_probability += metrics.target_probability;
                measured += 1;
            }
        }

        let divisor = measured.max(1) as f64;
        Ok(PriorSearchEvaluation {
            average_log_loss: total_log_loss / divisor,
            average_target_rank: total_target_rank / divisor,
            average_target_probability: total_target_probability / divisor,
        })
    }

    pub(super) fn better_prior_search_evaluation(
        candidate: &PriorSearchEvaluation,
        incumbent: &PriorSearchEvaluation,
    ) -> bool {
        candidate
            .average_log_loss
            .total_cmp(&incumbent.average_log_loss)
            .then_with(|| {
                candidate
                    .average_target_rank
                    .total_cmp(&incumbent.average_target_rank)
            })
            .then_with(|| {
                incumbent
                    .average_target_probability
                    .total_cmp(&candidate.average_target_probability)
            })
            == std::cmp::Ordering::Less
    }

    pub(super) fn offline_book_solver(&self) -> Self {
        let mut config = self.config.clone();
        config.medium_state_lookahead_threshold = config.medium_state_lookahead_threshold.max(96);
        config.lookahead_candidate_pool = config.lookahead_candidate_pool.max(150);
        config.medium_state_lookahead_candidate_pool =
            config.medium_state_lookahead_candidate_pool.max(96);
        config.lookahead_reply_pool = config.lookahead_reply_pool.max(50);
        config.medium_state_lookahead_reply_pool = config.medium_state_lookahead_reply_pool.max(32);
        config.medium_state_force_in_two_scan = config.medium_state_force_in_two_scan.max(192);
        config.exact_candidate_pool = config.exact_candidate_pool.max(160);
        config.lookahead_threshold = config.lookahead_threshold.max(224);
        config.danger_exact_root_pool = config.danger_exact_root_pool.max(32);
        config.large_state_split_threshold = config.large_state_split_threshold.min(50);
        self.clone_with_config(config)
    }

    pub(super) fn clone_with_config(&self, config: PriorConfig) -> Self {
        let mut cloned = self.clone();
        cloned.config = config.clone();
        cloned.exact_small_state_table =
            SmallStateTable::build(config.exact_exhaustive_threshold.max(2));
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
            OPENER_HOLDOUT_SHORTLIST.min(evaluations.len())
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
