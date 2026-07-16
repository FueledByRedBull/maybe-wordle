use super::*;

impl Solver {
    pub(super) fn predictive_book_identity(&self, as_of: NaiveDate) -> PredictiveBookIdentity {
        let policy = self.config.predictive_policy();
        let config_toml =
            toml::to_string(&self.config).expect("predictive config serialization must succeed");
        let model_manifest_hash = self.predictive_model_manifest_hash(as_of, &config_toml);
        let payload = format!(
            "manifest_version=1;model_manifest_hash={};policy={};mode={};variant={};as_of={};config={}",
            model_manifest_hash,
            policy.policy_id,
            self.mode.label(),
            self.variant.label(),
            as_of,
            config_toml
        );
        PredictiveBookIdentity {
            manifest_version: 1,
            model_manifest_hash,
            policy_id: policy.policy_id,
            mode: self.mode.label().to_string(),
            variant: self.variant.label().to_string(),
            config_fingerprint: stable_fingerprint(&payload),
            as_of,
        }
    }

    fn predictive_model_manifest_hash(&self, as_of: NaiveDate, config_toml: &str) -> String {
        let mut hash = 1469598103934665603u64;
        hash = hash_manifest_field(hash, b"maybe-wordle-predictive-model-v1");
        hash = hash_manifest_field(hash, self.mode.label().as_bytes());
        hash = hash_manifest_field(hash, self.variant.label().as_bytes());
        hash = hash_manifest_field(hash, as_of.to_string().as_bytes());
        hash = hash_manifest_field(hash, config_toml.as_bytes());
        for guess in &self.guesses {
            hash = hash_manifest_field(hash, guess.as_bytes());
        }
        for answer in &self.answers {
            hash = hash_manifest_field(hash, answer.word.as_bytes());
            hash = hash_manifest_field(hash, &[answer.in_seed as u8, answer.manual_entry as u8]);
            hash = hash_manifest_field(hash, &answer.manual_weight.to_bits().to_le_bytes());
            for date in &answer.history_dates {
                hash = hash_manifest_field(hash, date.to_string().as_bytes());
            }
            let snapshot = weight_snapshot_for_mode(answer, &self.config, as_of, self.mode);
            hash = hash_manifest_field(hash, &snapshot.base_weight.to_bits().to_le_bytes());
            hash = hash_manifest_field(hash, &snapshot.recency_weight.to_bits().to_le_bytes());
            hash = hash_manifest_field(hash, &snapshot.final_weight.to_bits().to_le_bytes());
        }
        for entry in &self.history_dates {
            hash = hash_manifest_field(hash, entry.print_date.to_string().as_bytes());
            hash = hash_manifest_field(hash, entry.solution.as_bytes());
            hash = hash_manifest_field(hash, &entry.id.unwrap_or_default().to_le_bytes());
            hash = hash_manifest_field(
                hash,
                &entry.days_since_launch.unwrap_or_default().to_le_bytes(),
            );
            hash = hash_manifest_field(hash, entry.editor.as_deref().unwrap_or("").as_bytes());
        }
        format!("{hash:016x}")
    }

    pub(super) fn predictive_history_snapshot(
        &self,
        as_of: NaiveDate,
    ) -> (Option<NaiveDate>, String) {
        let mut hash = 1469598103934665603u64;
        hash = hash_manifest_field(hash, b"maybe-wordle-history-v1");
        let mut snapshot_date = None;
        for entry in self
            .history_dates
            .iter()
            .filter(|entry| entry.print_date <= as_of)
        {
            snapshot_date = Some(snapshot_date.map_or(entry.print_date, |date: NaiveDate| {
                date.max(entry.print_date)
            }));
            hash = hash_manifest_field(hash, entry.print_date.to_string().as_bytes());
            hash = hash_manifest_field(hash, entry.solution.as_bytes());
            hash = hash_manifest_field(hash, &entry.id.unwrap_or_default().to_le_bytes());
        }
        (snapshot_date, format!("{hash:016x}"))
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

fn hash_manifest_field(mut hash: u64, bytes: &[u8]) -> u64 {
    hash = crate::pattern_table::hash_bytes(hash, &(bytes.len() as u64).to_le_bytes());
    crate::pattern_table::hash_bytes(hash, bytes)
}

fn stable_fingerprint(input: &str) -> String {
    let digest = crate::pattern_table::hash_bytes(1469598103934665603u64, input.as_bytes());
    format!("{digest:016x}")
}
