use super::*;

enum ArtifactLookup<T> {
    Missing,
    Invalid(anyhow::Error),
    Valid(T),
}

impl<T> ArtifactLookup<T> {
    fn into_result(self) -> Result<Option<T>> {
        match self {
            Self::Missing => Ok(None),
            Self::Invalid(error) => Err(error),
            Self::Valid(artifact) => Ok(Some(artifact)),
        }
    }
}

fn checked_predictive_artifact<T: for<'de> Deserialize<'de>>(
    path: &Path,
    valid: impl FnOnce(&T) -> bool,
) -> ArtifactLookup<T> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return ArtifactLookup::Missing;
        }
        Err(error) => {
            return ArtifactLookup::Invalid(
                anyhow!(error).context(format!("read predictive artifact {}", path.display())),
            );
        }
    };
    match serde_json::from_slice::<T>(&bytes) {
        Ok(artifact) if valid(&artifact) => ArtifactLookup::Valid(artifact),
        Ok(_) => ArtifactLookup::Invalid(anyhow!(
            "incompatible or invalid predictive artifact {}",
            path.display()
        )),
        Err(error) => ArtifactLookup::Invalid(
            anyhow!(error).context(format!("decode predictive artifact {}", path.display())),
        ),
    }
}

impl Solver {
    pub(super) fn load_predictive_opener_artifact(
        &self,
        as_of: NaiveDate,
    ) -> Result<Option<PredictiveOpenerArtifact>> {
        let path = self.opener_artifact_path(as_of);
        checked_predictive_artifact(&path, |artifact: &PredictiveOpenerArtifact| {
            artifact.identity == self.predictive_book_identity(as_of)
                && self.guess_index.contains_key(&artifact.opener)
                && artifact.games > 0
                && artifact.average_guesses.is_finite()
                && artifact.average_guesses > 0.0
                && artifact.failures < artifact.games
        })
        .into_result()
    }

    pub(super) fn load_predictive_reply_book(
        &self,
        as_of: NaiveDate,
    ) -> Result<Option<PredictiveReplyBookArtifact>> {
        let path = self.reply_book_artifact_path(as_of);
        checked_predictive_artifact(&path, |artifact: &PredictiveReplyBookArtifact| {
            let mut patterns = HashSet::new();
            artifact.identity == self.predictive_book_identity(as_of)
                && self.guess_index.contains_key(&artifact.opener)
                && artifact.replies.iter().all(|entry| {
                    let mut second_patterns = HashSet::new();
                    entry.feedback_pattern < ALL_GREEN_PATTERN
                        && patterns.insert(entry.feedback_pattern)
                        && self.guess_index.contains_key(&entry.reply)
                        && entry.third_replies.iter().all(|third| {
                            third.second_feedback_pattern < ALL_GREEN_PATTERN
                                && second_patterns.insert(third.second_feedback_pattern)
                                && self.guess_index.contains_key(&third.reply)
                        })
                })
        })
        .into_result()
    }

    pub(super) fn session_root_guess(
        &self,
        as_of: NaiveDate,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<String>> {
        check_predictive_search_cancelled(cancelled)?;
        let identity = self.predictive_book_identity(as_of);
        if let Some(cached) = self
            .session_opener_cache
            .lock()
            .expect("session opener cache")
            .get(&identity)
            .cloned()
        {
            return Ok(cached);
        }

        let computed = self.evaluate_session_opener(as_of, cancelled)?;
        self.session_opener_cache
            .lock()
            .expect("session opener cache")
            .insert(identity, computed.clone());
        Ok(computed)
    }

    pub(super) fn evaluate_session_opener(
        &self,
        as_of: NaiveDate,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<String>> {
        check_predictive_search_cancelled(cancelled)?;
        let offline = self.offline_book_solver()?;
        let (window_start, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let holdout = offline.previous_history_targets_for_books(window_start)?;
        if targets.is_empty() {
            return Ok(None);
        }
        let state = offline.initial_state(as_of);
        let candidates = offline
            .suggestion_batch_internal_with_search_mode_controlled(
                &state,
                offline.config.session_opener_pool.max(1),
                Some(PredictiveContext {
                    hard_mode: false,
                    as_of,
                    observations: &[],
                }),
                PredictiveBookUsage::None,
                None,
                cancelled,
            )?
            .suggestions;
        let best = offline.select_validated_opener(
            as_of,
            &candidates,
            &targets,
            holdout.as_ref().map(|(_, _, entries)| entries.as_slice()),
            cancelled,
        )?;
        Ok(best.map(|evaluation| evaluation.word))
    }

    pub(super) fn session_reply_guess(
        &self,
        as_of: NaiveDate,
        opener: &str,
        pattern: u8,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<String>> {
        check_predictive_search_cancelled(cancelled)?;
        let identity = self.predictive_book_identity(as_of);
        let key = (identity, opener.to_string(), pattern);
        if let Some(cached) = self
            .session_reply_cache
            .lock()
            .expect("session reply cache")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let computed = self.evaluate_session_reply(as_of, opener, pattern, cancelled)?;
        self.session_reply_cache
            .lock()
            .expect("session reply cache")
            .insert(key, computed.clone());
        Ok(computed)
    }

    pub(super) fn session_third_guess(
        &self,
        as_of: NaiveDate,
        opener: &str,
        opener_pattern: u8,
        reply: &str,
        reply_pattern: u8,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<String>> {
        check_predictive_search_cancelled(cancelled)?;
        let identity = self.predictive_book_identity(as_of);
        let key = (
            identity,
            opener.to_string(),
            opener_pattern,
            reply.to_string(),
            reply_pattern,
        );
        if let Some(cached) = self
            .session_third_cache
            .lock()
            .expect("session third cache")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let computed = self.evaluate_session_third(
            as_of,
            opener,
            opener_pattern,
            reply,
            reply_pattern,
            cancelled,
        )?;
        self.session_third_cache
            .lock()
            .expect("session third cache")
            .insert(key, computed.clone());
        Ok(computed)
    }

    pub(super) fn evaluate_session_reply(
        &self,
        as_of: NaiveDate,
        opener: &str,
        pattern: u8,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<String>> {
        check_predictive_search_cancelled(cancelled)?;
        let offline = self.offline_book_solver()?;
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let scoped_targets = targets
            .into_iter()
            .filter(|(_, target)| score_guess(opener, target) == pattern)
            .collect::<Vec<_>>();
        if scoped_targets.is_empty() {
            return Ok(None);
        }
        let root = offline.initial_state(as_of);
        let mut child = root.clone();
        offline.apply_feedback(&mut child, opener, pattern)?;
        if child.surviving.len() <= 1 {
            return Ok(None);
        }
        let observations = vec![(opener.to_string(), pattern)];
        offline.evaluate_session_branch_guess(
            as_of,
            &child,
            &observations,
            &scoped_targets,
            cancelled,
        )
    }

    pub(super) fn evaluate_session_third(
        &self,
        as_of: NaiveDate,
        opener: &str,
        opener_pattern: u8,
        reply: &str,
        reply_pattern: u8,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<String>> {
        check_predictive_search_cancelled(cancelled)?;
        let offline = self.offline_book_solver()?;
        let (_, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let scoped_targets = targets
            .into_iter()
            .filter(|(_, target)| {
                score_guess(opener, target) == opener_pattern
                    && score_guess(reply, target) == reply_pattern
            })
            .collect::<Vec<_>>();
        if scoped_targets.is_empty() {
            return Ok(None);
        }
        let mut state = offline.initial_state(as_of);
        offline.apply_feedback(&mut state, opener, opener_pattern)?;
        if state.surviving.len() <= 1 {
            return Ok(None);
        }
        offline.apply_feedback(&mut state, reply, reply_pattern)?;
        if state.surviving.len() <= 1 {
            return Ok(None);
        }
        let observations = vec![
            (opener.to_string(), opener_pattern),
            (reply.to_string(), reply_pattern),
        ];
        offline.evaluate_session_branch_guess(
            as_of,
            &state,
            &observations,
            &scoped_targets,
            cancelled,
        )
    }

    pub(super) fn evaluate_session_branch_guess(
        &self,
        as_of: NaiveDate,
        state: &SolveState,
        observations: &[(String, u8)],
        scoped_targets: &[(NaiveDate, String)],
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<String>> {
        check_predictive_search_cancelled(cancelled)?;
        let batch = self.suggestion_batch_internal_with_search_mode_controlled(
            state,
            self.config.session_reply_pool.max(1),
            Some(PredictiveContext {
                hard_mode: false,
                as_of,
                observations,
            }),
            PredictiveBookUsage::None,
            None,
            cancelled,
        )?;
        let split_first = state.surviving.len() > self.config.large_state_split_threshold;
        let mut metrics = self.score_guess_metrics_for_subset_controlled(
            &state.surviving,
            &state.weights,
            cancelled,
        )?;
        metrics.sort_by(|left, right| {
            compare_guess_metrics_for_state(left, right, &self.guesses, split_first)
        });
        let total_weight = state
            .surviving
            .iter()
            .map(|index| state.weights[*index])
            .sum::<f64>();
        let assessment =
            self.assess_subset_danger(&state.surviving, &state.weights, total_weight, &metrics);
        let lookahead_pool = self.expanded_pool_size(
            &batch.suggestions,
            self.config.session_reply_pool.max(1),
            split_first,
            state.surviving.len() > self.config.large_state_split_threshold,
            assessment,
        );
        let mut candidate_indexes = self.collect_lookahead_candidates(
            &batch.suggestions,
            state.surviving.len(),
            assessment.dangerous_lookahead,
            lookahead_pool,
        )?;
        if assessment.dangerous_exact
            && state.surviving.len() <= self.config.danger_exact_survivor_cap
        {
            let exact_pool = self.expanded_pool_size(
                &batch.suggestions,
                self.config.session_reply_pool.max(1),
                split_first,
                state.surviving.len() > self.config.exact_threshold,
                assessment,
            );
            let mut seen = candidate_indexes.iter().copied().collect::<HashSet<_>>();
            for guess_index in
                self.collect_exact_candidates(state, &batch.suggestions, exact_pool)?
            {
                if seen.insert(guess_index) {
                    candidate_indexes.push(guess_index);
                }
            }
        }
        let forced_prefix = observations
            .iter()
            .map(|(guess, _)| guess.clone())
            .collect::<Vec<_>>();
        let best = candidate_indexes
            .par_iter()
            .map(|guess_index| {
                let evaluation = self
                    .evaluate_forced_continuation(
                        &forced_prefix,
                        scoped_targets,
                        *guess_index,
                        cancelled,
                    )
                    .with_context(|| {
                        format!(
                            "evaluate book branch candidate {} at {as_of}",
                            self.guesses[*guess_index]
                        )
                    })?;
                Ok((self.guesses[*guess_index].clone(), evaluation))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min_by(|left, right| compare_forced_openers(&left.1, &right.1, &self.guesses));
        Ok(best.map(|(word, _)| word))
    }

    pub(super) fn cached_predictive_choice(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        allow_session_fallback: bool,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<PromotedPredictiveChoice>> {
        check_predictive_search_cancelled(cancelled)?;
        if observations.len() > 2 {
            return Ok(None);
        }
        for age in 0..=self.config.session_artifact_freshness_days as u64 {
            check_predictive_search_cancelled(cancelled)?;
            let Some(date) = as_of.checked_sub_days(Days::new(age)) else {
                break;
            };
            let choice = if observations.is_empty() {
                self.load_predictive_opener_artifact(date)?.map(|artifact| {
                    PromotedPredictiveChoice {
                        word: artifact.opener,
                        source: if age == 0 {
                            PredictivePromotionSource::ExactDateOpenerArtifact
                        } else {
                            PredictivePromotionSource::RecentOpenerArtifact
                        },
                        artifact_date: Some(date),
                    }
                })
            } else {
                self.load_predictive_reply_book(date)?.and_then(|artifact| {
                    if artifact.opener != observations[0].0 {
                        return None;
                    }
                    let entry = artifact
                        .replies
                        .into_iter()
                        .find(|entry| entry.feedback_pattern == observations[0].1)?;
                    let word = if observations.len() == 1 {
                        entry.reply
                    } else {
                        if entry.reply != observations[1].0 {
                            return None;
                        }
                        entry
                            .third_replies
                            .into_iter()
                            .find(|third| third.second_feedback_pattern == observations[1].1)?
                            .reply
                    };
                    Some(PromotedPredictiveChoice {
                        word,
                        source: if age == 0 {
                            PredictivePromotionSource::ReplyBook
                        } else {
                            PredictivePromotionSource::RecentReplyBook
                        },
                        artifact_date: Some(date),
                    })
                })
            };
            if choice.is_some() {
                return Ok(choice);
            }
        }
        // Missing training history makes a session book unavailable, not invalid.
        // Explicit book builds still require history; candidate/artifact errors
        // above and below this availability check remain fatal.
        if !allow_session_fallback
            || !self
                .history_dates
                .iter()
                .any(|entry| entry.print_date <= as_of)
        {
            return Ok(None);
        }
        let (word, source) = match observations {
            [] => (
                self.session_root_guess(as_of, cancelled)?,
                PredictivePromotionSource::SessionRootFallback,
            ),
            [(guess, pattern)] => (
                self.session_reply_guess(as_of, guess, *pattern, cancelled)?,
                PredictivePromotionSource::SessionReplyFallback,
            ),
            [(opener, first), (reply, second)] => (
                self.session_third_guess(as_of, opener, *first, reply, *second, cancelled)?,
                PredictivePromotionSource::SessionThirdFallback,
            ),
            _ => unreachable!("history length checked"),
        };
        Ok(word.map(|word| PromotedPredictiveChoice {
            word,
            source,
            artifact_date: None,
        }))
    }
}

pub(super) fn write_predictive_artifact<T: Serialize>(
    path: &std::path::Path,
    value: &T,
) -> Result<()> {
    let bytes =
        serde_json::to_vec_pretty(value).context("failed to serialize predictive artifact")?;
    crate::atomic_file::atomic_write(path, &bytes)
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    #[test]
    fn missing_or_future_only_training_history_does_not_block_seed_only_play() {
        let mut solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        for future_only in [false, true] {
            if future_only {
                solver.data_mut().history_dates.push(NytDailyEntry {
                    id: None,
                    solution: "cigar".into(),
                    print_date: date.succ_opt().unwrap(),
                    days_since_launch: None,
                    editor: None,
                });
            }
            for observations in [
                vec![],
                vec![("cigar".into(), 0)],
                vec![("cigar".into(), 0), ("rebut".into(), 0)],
            ] {
                assert!(
                    solver
                        .cached_predictive_choice(date, &observations, true, &|| false)
                        .expect("optional session book has no eligible training history")
                        .is_none()
                );
            }
            assert!(solver.build_predictive_opener_cache(date).is_err());
        }
    }

    fn reply_book(solver: &Solver, date: NaiveDate, opener: &str) -> PredictiveReplyBookArtifact {
        PredictiveReplyBookArtifact {
            identity: solver.predictive_book_identity(date),
            opener: opener.to_string(),
            replies: vec![PredictiveReplyEntry {
                feedback_pattern: 0,
                reply: "rebut".to_string(),
                surviving_answers: 2,
                proxy_cost: None,
                lookahead_cost: None,
                exact_cost: None,
                third_replies: Vec::new(),
            }],
        }
    }

    #[test]
    fn mismatched_book_identity_is_an_error_not_absence() {
        let solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        let mut artifact = reply_book(&solver, date, "cigar");
        artifact.identity.config_fingerprint = "wrong".to_string();
        write_predictive_artifact(&solver.reply_book_artifact_path(date), &artifact).unwrap();
        assert!(solver.load_predictive_reply_book(date).is_err());
    }

    #[test]
    fn matching_recent_branch_is_considered_after_ineligible_exact_book() {
        let solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        write_predictive_artifact(
            &solver.reply_book_artifact_path(date),
            &reply_book(&solver, date, "sissy"),
        )
        .unwrap();
        let prior = date.pred_opt().unwrap();
        write_predictive_artifact(
            &solver.reply_book_artifact_path(prior),
            &reply_book(&solver, prior, "cigar"),
        )
        .unwrap();
        let choice = solver
            .cached_predictive_choice(date, &[("cigar".to_string(), 0)], false, &|| false)
            .unwrap();
        assert_eq!(choice.as_ref().unwrap().artifact_date, Some(prior));
        assert_eq!(
            choice.as_ref().unwrap().source,
            PredictivePromotionSource::RecentReplyBook
        );
        assert_eq!(choice.map(|choice| choice.word), Some("rebut".to_string()));
    }

    #[test]
    fn corrupt_exact_artifact_does_not_fall_back_to_a_recent_book() {
        let solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        let prior = date.pred_opt().unwrap();
        write_predictive_artifact(
            &solver.reply_book_artifact_path(prior),
            &reply_book(&solver, prior, "cigar"),
        )
        .unwrap();
        crate::atomic_file::atomic_write(&solver.reply_book_artifact_path(date), b"{broken")
            .unwrap();
        let error = solver
            .cached_predictive_choice(date, &[("cigar".to_string(), 0)], false, &|| false)
            .unwrap_err();
        assert!(format!("{error:#}").contains("decode predictive artifact"));
    }

    #[test]
    fn cached_promotion_requires_a_present_ranked_word() {
        let solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        let mut suggestions = solver
            .suggestion_batch_internal(
                &solver.initial_state(date),
                3,
                None,
                PredictiveBookUsage::None,
            )
            .unwrap()
            .suggestions;
        let before = suggestions
            .iter()
            .map(|s| s.word.clone())
            .collect::<Vec<_>>();
        assert!(!promote_cached_suggestion(&mut suggestions, "zzzzz"));
        assert_eq!(
            before,
            suggestions
                .iter()
                .map(|s| s.word.clone())
                .collect::<Vec<_>>()
        );
        let word = suggestions.last().unwrap().word.clone();
        assert!(promote_cached_suggestion(&mut suggestions, &word));
        assert_eq!(suggestions[0].word, word);
    }

    #[test]
    fn cancellation_reaches_forced_book_evaluation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy", "humph"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        let targets = [(date, "humph".to_string())];
        let calls = AtomicUsize::new(0);
        let cancelled = || calls.fetch_add(1, Ordering::Relaxed) >= 3;
        let error = solver
            .evaluate_forced_continuation(&["cigar".to_string()], &targets, 1, &cancelled)
            .unwrap_err();
        assert!(format!("{error:#}").contains("cancelled"));
        assert!(calls.load(Ordering::Relaxed) >= 4);
    }

    #[test]
    fn required_primary_and_holdout_errors_are_never_silently_validated() {
        let solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        let candidates = solver
            .suggestion_batch_internal(
                &solver.initial_state(date),
                3,
                None,
                PredictiveBookUsage::None,
            )
            .unwrap()
            .suggestions;
        let good = [(date, "cigar".to_string())];
        let primary_error = solver
            .select_validated_opener(date, &candidates, &[], None, &|| false)
            .unwrap_err();
        assert!(format!("{primary_error:#}").contains("primary opener"));
        let holdout_error = solver
            .select_validated_opener(date, &candidates, &good, Some(&[]), &|| false)
            .unwrap_err();
        assert!(format!("{holdout_error:#}").contains("required holdout"));
        assert!(
            solver
                .select_validated_opener(date, &candidates, &good, None, &|| false)
                .unwrap()
                .is_some()
        );
    }
}
