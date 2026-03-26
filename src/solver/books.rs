use super::*;

impl Solver {
    pub(super) fn predictive_book_identity(&self, as_of: NaiveDate) -> PredictiveBookIdentity {
        let policy = self.config.predictive_policy();
        let config_toml =
            toml::to_string(&self.config).expect("predictive config serialization must succeed");
        let payload = format!(
            "policy={};mode={};variant={};as_of={};guesses={};answers={};config={}",
            policy.policy_id,
            self.mode.label(),
            self.variant.label(),
            as_of,
            self.guesses.len(),
            self.answers.len(),
            config_toml
        );
        PredictiveBookIdentity {
            policy_id: policy.policy_id,
            mode: self.mode.label().to_string(),
            variant: self.variant.label().to_string(),
            config_fingerprint: stable_fingerprint(&payload),
            as_of,
        }
    }

    pub(super) fn opener_artifact_path(&self, as_of: NaiveDate) -> PathBuf {
        let identity = self.predictive_book_identity(as_of);
        self.artifact_dir.join(format!(
            "opener-{}-{}-{}-{}-{}.json",
            identity.policy_id,
            identity.mode,
            identity.variant,
            identity.config_fingerprint,
            identity.as_of
        ))
    }

    pub(super) fn reply_book_artifact_path(&self, as_of: NaiveDate) -> PathBuf {
        let identity = self.predictive_book_identity(as_of);
        self.artifact_dir.join(format!(
            "reply-book-{}-{}-{}-{}-{}.json",
            identity.policy_id,
            identity.mode,
            identity.variant,
            identity.config_fingerprint,
            identity.as_of
        ))
    }

    pub(super) fn load_predictive_opener_artifact(
        &self,
        as_of: NaiveDate,
    ) -> Result<Option<PredictiveOpenerArtifact>> {
        let path = self.opener_artifact_path(as_of);
        if !path.exists() {
            return Ok(None);
        }
        let artifact: PredictiveOpenerArtifact = read_predictive_artifact(&path)?;
        let valid = artifact.identity == self.predictive_book_identity(as_of)
            && artifact.games > 0
            && artifact.average_guesses.is_finite()
            && artifact.average_guesses > 0.0
            && artifact.failures < artifact.games;
        Ok(valid.then_some(artifact))
    }

    pub(super) fn load_recent_predictive_opener_artifact(
        &self,
        as_of: NaiveDate,
        max_age_days: u64,
    ) -> Result<Option<PredictiveOpenerArtifact>> {
        for age_days in 1..=max_age_days {
            let Some(candidate_date) = as_of.checked_sub_days(Days::new(age_days)) else {
                break;
            };
            if let Some(artifact) = self.load_predictive_opener_artifact(candidate_date)? {
                return Ok(Some(artifact));
            }
        }
        Ok(None)
    }

    pub(super) fn load_predictive_reply_book(
        &self,
        as_of: NaiveDate,
    ) -> Result<Option<PredictiveReplyBookArtifact>> {
        let path = self.reply_book_artifact_path(as_of);
        if !path.exists() {
            return Ok(None);
        }
        let artifact: PredictiveReplyBookArtifact = read_predictive_artifact(&path)?;
        Ok((artifact.identity == self.predictive_book_identity(as_of)).then_some(artifact))
    }

    pub(super) fn session_root_guess(&self, as_of: NaiveDate) -> Result<Option<String>> {
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

        let computed = self.evaluate_session_opener(as_of)?;
        self.session_opener_cache
            .lock()
            .expect("session opener cache")
            .insert(identity, computed.clone());
        Ok(computed)
    }

    pub(super) fn evaluate_session_opener(&self, as_of: NaiveDate) -> Result<Option<String>> {
        let offline = self.offline_book_solver();
        let (window_start, _, targets) = offline.recent_history_targets_for_books(as_of)?;
        let holdout = offline.previous_history_targets_for_books(window_start)?;
        if targets.is_empty() {
            return Ok(None);
        }
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
        let best = offline.select_validated_opener(
            as_of,
            &candidates,
            &targets,
            holdout.as_ref().map(|(_, _, entries)| entries.as_slice()),
            3,
        )?;
        Ok(best.map(|evaluation| evaluation.word))
    }

    pub(super) fn session_reply_guess(
        &self,
        as_of: NaiveDate,
        opener: &str,
        pattern: u8,
    ) -> Result<Option<String>> {
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

        let computed = self.evaluate_session_reply(as_of, opener, pattern)?;
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
    ) -> Result<Option<String>> {
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

        let computed =
            self.evaluate_session_third(as_of, opener, opener_pattern, reply, reply_pattern)?;
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
    ) -> Result<Option<String>> {
        let offline = self.offline_book_solver();
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
        offline.evaluate_session_branch_guess(as_of, &child, &observations, &scoped_targets)
    }

    pub(super) fn evaluate_session_third(
        &self,
        as_of: NaiveDate,
        opener: &str,
        opener_pattern: u8,
        reply: &str,
        reply_pattern: u8,
    ) -> Result<Option<String>> {
        let offline = self.offline_book_solver();
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
        offline.evaluate_session_branch_guess(as_of, &state, &observations, &scoped_targets)
    }

    pub(super) fn evaluate_session_branch_guess(
        &self,
        as_of: NaiveDate,
        state: &SolveState,
        observations: &[(String, u8)],
        scoped_targets: &[(NaiveDate, String)],
    ) -> Result<Option<String>> {
        let batch = self.suggestion_batch_internal(
            state,
            self.config.session_reply_pool.max(1),
            Some(PredictiveContext {
                as_of,
                observations,
            }),
            PredictiveBookUsage::None,
        )?;
        let split_first = state.surviving.len() > self.config.large_state_split_threshold;
        let mut metrics = self.score_guess_metrics_for_subset(
            &state.surviving,
            &state.weights,
            &self.exact_small_state_table,
        );
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
            .filter_map(|guess_index| {
                let evaluation = self
                    .evaluate_forced_continuation(&forced_prefix, scoped_targets, *guess_index, 3)
                    .ok()?;
                Some((self.guesses[*guess_index].clone(), evaluation))
            })
            .min_by(|left, right| compare_forced_openers(&left.1, &right.1, &self.guesses));
        Ok(best.map(|(word, _)| word))
    }

    pub(super) fn cached_predictive_choice(
        &self,
        as_of: NaiveDate,
        observations: &[(String, u8)],
        allow_session_fallback: bool,
    ) -> Option<PromotedPredictiveChoice> {
        match observations {
            [] => self
                .load_predictive_opener_artifact(as_of)
                .ok()
                .flatten()
                .map(|artifact| PromotedPredictiveChoice {
                    word: artifact.opener,
                    source: PredictivePromotionSource::ExactDateOpenerArtifact,
                })
                .or_else(|| {
                    self.load_recent_predictive_opener_artifact(
                        as_of,
                        OPENER_ARTIFACT_FRESHNESS_DAYS,
                    )
                    .ok()
                    .flatten()
                    .map(|artifact| PromotedPredictiveChoice {
                        word: artifact.opener,
                        source: PredictivePromotionSource::RecentOpenerArtifact,
                    })
                })
                .or_else(|| {
                    allow_session_fallback
                        .then(|| self.session_root_guess(as_of).ok().flatten())
                        .flatten()
                        .map(|word| PromotedPredictiveChoice {
                            word,
                            source: PredictivePromotionSource::SessionRootFallback,
                        })
                }),
            [(guess, pattern)] => self
                .load_predictive_reply_book(as_of)
                .ok()
                .flatten()
                .filter(|artifact| artifact.opener == guess.as_str())
                .and_then(|artifact| {
                    artifact
                        .replies
                        .into_iter()
                        .find(|entry| entry.feedback_pattern == *pattern)
                        .map(|entry| PromotedPredictiveChoice {
                            word: entry.reply,
                            source: PredictivePromotionSource::ReplyBook,
                        })
                })
                .or_else(|| {
                    allow_session_fallback
                        .then(|| {
                            self.session_reply_guess(as_of, guess, *pattern)
                                .ok()
                                .flatten()
                        })
                        .flatten()
                        .map(|word| PromotedPredictiveChoice {
                            word,
                            source: PredictivePromotionSource::SessionReplyFallback,
                        })
                }),
            [(opener, opener_pattern), (reply, reply_pattern)] => self
                .load_predictive_reply_book(as_of)
                .ok()
                .flatten()
                .filter(|artifact| artifact.opener == opener.as_str())
                .and_then(|artifact| {
                    artifact
                        .replies
                        .into_iter()
                        .find(|entry| {
                            entry.feedback_pattern == *opener_pattern
                                && entry.reply == reply.as_str()
                        })
                        .and_then(|entry| {
                            entry
                                .third_replies
                                .into_iter()
                                .find(|entry| entry.second_feedback_pattern == *reply_pattern)
                                .map(|entry| PromotedPredictiveChoice {
                                    word: entry.reply,
                                    source: PredictivePromotionSource::ReplyBook,
                                })
                        })
                })
                .or_else(|| {
                    allow_session_fallback
                        .then(|| {
                            self.session_third_guess(
                                as_of,
                                opener,
                                *opener_pattern,
                                reply,
                                *reply_pattern,
                            )
                            .ok()
                            .flatten()
                        })
                        .flatten()
                        .map(|word| PromotedPredictiveChoice {
                            word,
                            source: PredictivePromotionSource::SessionThirdFallback,
                        })
                }),
            _ => None,
        }
    }
}

pub(super) fn stable_fingerprint(input: &str) -> String {
    let digest = crate::pattern_table::hash_bytes(1469598103934665603u64, input.as_bytes());
    format!("{digest:016x}")
}

pub(super) fn write_predictive_artifact<T: Serialize>(
    path: &std::path::Path,
    value: &T,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let bytes =
        serde_json::to_vec_pretty(value).context("failed to serialize predictive artifact")?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub(super) fn read_predictive_artifact<T: for<'de> Deserialize<'de>>(
    path: &std::path::Path,
) -> Result<T> {
    let raw = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&raw).with_context(|| format!("failed to parse {}", path.display()))
}
