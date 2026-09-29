use super::*;

impl WordleGuiApp {
    pub(super) fn show_diagnostics_panel(&mut self, ui: &mut egui::Ui) {
        let surface = gui_surface_state(
            false,
            self.computing,
            &self.status,
            self.predictive_recovery_mode,
            self.predictive_suggestions.len(),
        );
        ui.horizontal_wrapped(|ui| {
            diagnostic_badge(ui, "SURFACE", surface.label());
            diagnostic_badge(
                ui,
                "WORKERS",
                if self.computing {
                    "latest request running"
                } else {
                    "idle"
                },
            );
            diagnostic_badge(
                ui,
                "ARTIFACT",
                if self
                    .search_profile
                    .finite_options(&self.predictive_solver)
                    .is_some()
                {
                    predictive_finite_banner_text(self.search_profile)
                } else {
                    predictive_banner_text(self.predictive_artifact_state)
                },
            );
            diagnostic_badge(ui, "SURVIVORS", &self.surviving_count.to_string());
        });
        if let Some(search) = &self.predictive_finite_search {
            ui.label(
                RichText::new(format_finite_search_summary(search))
                    .small()
                    .color(Color32::from_rgb(92, 72, 54)),
            );
        }
        ui.add_space(16.0);
        if is_compact_layout(ui.available_width()) {
            ui.group(|ui| self.show_live_trace(ui));
            ui.add_space(10.0);
            ui.group(|ui| self.show_recovery_provenance(ui));
        } else {
            ui.columns(2, |columns| {
                columns[0].group(|ui| self.show_live_trace(ui));
                columns[1].group(|ui| self.show_recovery_provenance(ui));
            });
        }
        ui.add_space(16.0);
        ui.label(
            RichText::new(
                "The cancellable solver worker shares one replaceable pending slot: obsolete queued work is discarded, and bounded finite searches exit when a newer generation arrives.",
            )
            .small()
            .color(Color32::from_rgb(92, 72, 54)),
        );
    }

    pub(super) fn show_live_trace(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("LIVE TRACE").monospace().strong());
        policy_row(ui, "Generation", &self.latest_generation.to_string());
        policy_row(
            ui,
            "Suggestions",
            &self.predictive_suggestions.len().to_string(),
        );
        policy_row(
            ui,
            "Candidates",
            &self.predictive_candidates.len().to_string(),
        );
        policy_row(ui, "Posterior mass", &format!("{:.6}", self.total_weight));
        if let Some(execution) = &self.predictive_execution {
            policy_row(ui, "Actual route", execution.route_label());
            policy_row(ui, "Objective", execution.objective.label());
            policy_row(ui, "Legal actions", execution.action_scope.label());
            policy_row(ui, "Root coverage", execution.candidate_scope.label());
            ui.label(execution.summary());
        } else {
            policy_row(ui, "Actual route", "No current execution");
        }
    }

    pub(super) fn show_recovery_provenance(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("RECOVERY & PROVENANCE").monospace().strong());
        ui.label(if self.status.message().is_empty() {
            "No runtime error."
        } else {
            self.status.message()
        });
        ui.add_space(8.0);
        ui.label(
            RichText::new(if self.predictive_model_metadata.is_empty() {
                "No model response has been received yet."
            } else {
                &self.predictive_model_metadata
            })
            .monospace()
            .small(),
        );
    }
}
