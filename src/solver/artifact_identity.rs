use super::*;

impl Solver {
    pub(super) fn predictive_book_identity(&self, as_of: NaiveDate) -> PredictiveBookIdentity {
        let policy = self.config.predictive_policy();
        let config_toml =
            toml::to_string(&self.config).expect("predictive config serialization must succeed");
        let model_manifest_hash = self.predictive_model_manifest_hash(as_of, &config_toml);
        let mut fingerprint =
            crate::identity::CanonicalSha256::new("maybe-wordle-predictive-book-config-v2");
        fingerprint
            .field(model_manifest_hash.as_bytes())
            .field(policy.policy_id.as_bytes())
            .field(self.mode.label().as_bytes())
            .field(self.variant.label().as_bytes())
            .field(as_of.to_string().as_bytes())
            .field(config_toml.as_bytes());
        PredictiveBookIdentity {
            manifest_version: 2,
            model_manifest_hash,
            policy_id: policy.policy_id,
            mode: self.mode.label().to_string(),
            variant: self.variant.label().to_string(),
            config_fingerprint: fingerprint.finish_hex(),
            as_of,
        }
    }

    fn predictive_model_manifest_hash(&self, as_of: NaiveDate, config_toml: &str) -> String {
        let mut hash = crate::identity::CanonicalSha256::new("maybe-wordle-predictive-model-v2");
        hash.field(self.mode.label().as_bytes())
            .field(self.variant.label().as_bytes())
            .field(as_of.to_string().as_bytes())
            .field(config_toml.as_bytes());
        for guess in &self.guesses {
            hash.field(guess.as_bytes());
        }
        for answer in &self.answers {
            hash.field(answer.word.as_bytes())
                .field(&[answer.in_seed as u8, answer.manual_entry as u8])
                .field(&answer.manual_weight.to_bits().to_le_bytes());
            for date in &answer.history_dates {
                hash.field(date.to_string().as_bytes());
            }
            let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
            hash.field(&snapshot.base_weight.to_bits().to_le_bytes())
                .field(&snapshot.recency_weight.to_bits().to_le_bytes())
                .field(&snapshot.final_weight.to_bits().to_le_bytes());
        }
        for entry in &self.history_dates {
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
