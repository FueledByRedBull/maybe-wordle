use super::*;

impl WordleGuiApp {
    pub(super) fn show_policy_panel(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.label(RichText::new("SEARCH ALLOCATION").monospace().strong());
                ui.add_space(8.0);
                policy_row(ui, "Mode", self.config.search_policy_mode.label());
                policy_row(
                    ui,
                    "Exact",
                    &format!(
                        "≤ {} survivors; exhaustive ≤ {}",
                        self.config.exact_threshold, self.config.exact_exhaustive_threshold
                    ),
                );
                policy_row(
                    ui,
                    "Lookahead",
                    &format!(
                        "≤ {} survivors; medium profile ≤ {}",
                        self.config.lookahead_threshold,
                        self.config.medium_state_lookahead_threshold
                    ),
                );
                policy_row(
                    ui,
                    "Candidate pools",
                    &format!(
                        "exact {} · lookahead {}/{}",
                        self.config.exact_candidate_pool,
                        self.config.lookahead_candidate_pool,
                        self.config.medium_state_lookahead_candidate_pool
                    ),
                );
                policy_row(
                    ui,
                    "Danger escalation",
                    &format!(
                        "lookahead {:.2} · exact {:.2}",
                        self.config.danger_lookahead_threshold, self.config.danger_exact_threshold
                    ),
                );
            });
            columns[1].group(|ui| {
                ui.label(RichText::new("PREDICTIVE CONTRACT").monospace().strong());
                ui.add_space(8.0);
                policy_row(
                    ui,
                    "Puzzle date",
                    if self.date_text.is_empty() {
                        "not selected"
                    } else {
                        &self.date_text
                    },
                );
                policy_row(ui, "Profile", self.search_profile.label());
                policy_row(
                    ui,
                    "Artifact path",
                    if self.predictive_execution.is_none() {
                        "Awaiting current search result"
                    } else if self.predictive_execution.as_ref().is_some_and(|execution| {
                        execution.route == crate::predictive::PredictiveRegime::Finite
                    }) {
                        predictive_finite_banner_text(self.search_profile)
                    } else {
                        predictive_banner_text(self.predictive_artifact_state)
                    },
                );
                if let Some(search) = &self.predictive_finite_search {
                    policy_row(ui, "Finite search", &format_finite_search_summary(search));
                }
                policy_row(
                    ui,
                    "Recovery",
                    self.predictive_recovery_mode
                        .map_or("not active", RecoveryMode::label),
                );
                policy_row(
                    ui,
                    "Manual overrides",
                    &format!(
                        "{} auditable word weights",
                        self.config.manual_weights.len()
                    ),
                );
                policy_row(ui, "Registry", &predictive_registry_summary(&self.config));
            });
        });
        ui.add_space(16.0);
        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(255, 252, 247))
            .inner_margin(16.0)
            .show(ui, |ui| {
                ui.label(RichText::new("IDENTITY & STALENESS").monospace().strong());
                if self.predictive_model_metadata.is_empty() {
                    ui.label("Model identity will appear after the next predictive suggestion.");
                } else {
                    ui.label(&self.predictive_model_metadata);
                }
                if self
                    .search_profile
                    .finite_options(&self.predictive_solver)
                    .is_none()
                    && let Some(message) = predictive_reply_book_text(
                        self.observations.len(),
                        self.predictive_artifact_state,
                    )
                {
                    ui.label(message);
                }
            });
    }
}
