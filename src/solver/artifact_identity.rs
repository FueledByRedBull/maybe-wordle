use super::*;

type IdentityKey = (NaiveDate, String, &'static str, &'static str);

#[derive(Clone, Debug)]
struct IdentitySnapshot {
    identity: PredictiveBookIdentity,
    history: (Option<NaiveDate>, String),
}

#[derive(Debug, Default)]
pub(super) struct PredictiveIdentityCache {
    entries: std::collections::VecDeque<(IdentityKey, IdentitySnapshot)>,
    #[cfg(test)]
    builds: usize,
}

impl Solver {
    pub(super) fn predictive_book_identity(&self, as_of: NaiveDate) -> PredictiveBookIdentity {
        self.predictive_identity_snapshot(as_of).identity
    }

    fn predictive_identity_snapshot(&self, as_of: NaiveDate) -> IdentitySnapshot {
        let config_toml =
            toml::to_string(&self.config).expect("predictive config serialization must succeed");
        let key = (as_of, config_toml, self.mode.label(), self.variant.label());
        if let Ok(cache) = self.identity_cache.lock()
            && let Some((_, snapshot)) = cache.entries.iter().find(|(cached, _)| cached == &key)
        {
            return snapshot.clone();
        }
        let snapshot = IdentitySnapshot {
            identity: self.compute_predictive_book_identity(as_of, &key.1),
            history: self.compute_predictive_history_snapshot(as_of),
        };
        // A poisoned optimization cache is never authoritative; recompute from immutable inputs.
        if let Ok(mut cache) = self.identity_cache.lock() {
            if cache.entries.len() == 32 {
                cache.entries.pop_front();
            }
            cache.entries.push_back((key, snapshot.clone()));
            #[cfg(test)]
            {
                cache.builds += 1;
            }
        }
        snapshot
    }

    fn compute_predictive_book_identity(
        &self,
        as_of: NaiveDate,
        config_toml: &str,
    ) -> PredictiveBookIdentity {
        let policy = self.config.predictive_policy();
        let model_manifest_hash = self.predictive_model_manifest_hash(as_of, config_toml);
        let mut fingerprint =
            crate::identity::CanonicalSha256::new("maybe-wordle-predictive-book-config-v4");
        fingerprint
            .field(model_manifest_hash.as_bytes())
            .field(policy.policy_id.as_bytes())
            .field(self.mode.label().as_bytes())
            .field(self.variant.label().as_bytes())
            .field(as_of.to_string().as_bytes())
            .field(config_toml.as_bytes());
        PredictiveBookIdentity {
            manifest_version: 3,
            model_manifest_hash,
            policy_id: policy.policy_id,
            mode: self.mode.label().to_string(),
            variant: self.variant.label().to_string(),
            config_fingerprint: fingerprint.finish_hex(),
            as_of,
        }
    }

    fn predictive_model_manifest_hash(&self, as_of: NaiveDate, config_toml: &str) -> String {
        let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-predictive-model-v3");
        hash.field(self.mode.label().as_bytes())
            .field(self.variant.label().as_bytes())
            .field(as_of.to_string().as_bytes())
            .field(config_toml.as_bytes());
        for guess in &self.guesses {
            hash.field(guess.as_bytes());
        }
        // Future history can move a word from the tail into the stored primary list.
        // Hash date-eligible content in word order, not storage/index order.
        let mut answers = self.answers.iter().collect::<Vec<_>>();
        answers.sort_unstable_by(|left, right| left.word.cmp(&right.word));
        for answer in answers {
            let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
            if snapshot.base_weight == 0.0 && !self.guess_index.contains_key(&answer.word) {
                continue;
            }
            hash.field(answer.word.as_bytes())
                .field(&[answer.in_seed as u8, answer.manual_entry as u8])
                .field(&answer.manual_weight.to_bits().to_le_bytes());
            for date in answer.history_dates.iter().filter(|date| **date <= as_of) {
                hash.field(date.to_string().as_bytes());
            }
            hash.field(&snapshot.base_weight.to_bits().to_le_bytes())
                .field(&snapshot.recency_weight.to_bits().to_le_bytes())
                .field(&snapshot.final_weight.to_bits().to_le_bytes());
        }
        for entry in self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date <= as_of)
        {
            hash.field(entry.print_date.to_string().as_bytes())
                .field(entry.solution.as_bytes())
                .field(&entry.id.unwrap_or_default().to_le_bytes())
                .field(&entry.days_since_launch.unwrap_or_default().to_le_bytes())
                .field(entry.editor.as_deref().unwrap_or("").as_bytes());
        }
        hash.finish_hex()
    }

    pub(super) fn predictive_history_snapshot(
        &self,
        as_of: NaiveDate,
    ) -> (Option<NaiveDate>, String) {
        self.predictive_identity_snapshot(as_of).history
    }

    fn compute_predictive_history_snapshot(&self, as_of: NaiveDate) -> (Option<NaiveDate>, String) {
        let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-history-v2");
        let mut snapshot_date = None;
        for entry in self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date <= as_of)
        {
            snapshot_date = Some(snapshot_date.map_or(entry.print_date, |date: NaiveDate| {
                date.max(entry.print_date)
            }));
            hash.field(entry.print_date.to_string().as_bytes())
                .field(entry.solution.as_bytes())
                .field(&entry.id.unwrap_or_default().to_le_bytes());
        }
        (snapshot_date, hash.finish_tagged())
    }

    pub(super) fn opener_artifact_path(&self, as_of: NaiveDate) -> PathBuf {
        let identity = self.predictive_book_identity(as_of);
        self.artifact_dir.join(format!(
            "opener-v{}-{}-{}-{}-{}-{}-{}.json",
            identity.manifest_version,
            identity.policy_id,
            identity.mode,
            identity.variant,
            identity.model_manifest_hash,
            identity.config_fingerprint,
            identity.as_of
        ))
    }

    pub(super) fn reply_book_artifact_path(&self, as_of: NaiveDate) -> PathBuf {
        let identity = self.predictive_book_identity(as_of);
        self.artifact_dir.join(format!(
            "reply-book-v{}-{}-{}-{}-{}-{}-{}.json",
            identity.manifest_version,
            identity.policy_id,
            identity.mode,
            identity.variant,
            identity.model_manifest_hash,
            identity.config_fingerprint,
            identity.as_of
        ))
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn repeated_identity_requests_reuse_work_and_mutations_invalidate_it() {
        let mut solver = super::super::tests::test_solver(&["cigar", "rebut", "sissy"]);
        let date = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();
        let first = solver.predictive_book_identity(date);
        assert_eq!(first.manifest_version, 3);
        assert!(
            solver
                .opener_artifact_path(date)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("opener-v3-")
        );
        let history = solver.predictive_history_snapshot(date);
        solver.opener_artifact_path(date);
        solver.reply_book_artifact_path(date);
        assert_eq!(solver.identity_cache.lock().unwrap().builds, 1);
        assert_eq!(first, solver.predictive_book_identity(date));
        assert_eq!(history, solver.predictive_history_snapshot(date));
        solver.config.cooldown_floor *= 0.5;
        assert_ne!(first, solver.predictive_book_identity(date));
        assert_eq!(solver.identity_cache.lock().unwrap().builds, 2);
        for offset in 1..40 {
            solver.predictive_book_identity(date.checked_add_days(Days::new(offset)).unwrap());
        }
        assert!(solver.identity_cache.lock().unwrap().entries.len() <= 32);
        let mut clone = solver.clone();
        assert!(clone.identity_cache.lock().unwrap().entries.is_empty());
        let prior = clone.predictive_book_identity(date);
        clone.data_mut().answers[0].manual_weight = 4.0;
        assert_ne!(prior, clone.predictive_book_identity(date));
    }
}
