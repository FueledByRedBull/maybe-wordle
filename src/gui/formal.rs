use super::*;

impl WordleGuiApp {
    pub(super) fn show_formal_panel(&mut self, ui: &mut egui::Ui) {
        if self.formal_solver.is_none() {
            ui.group(|ui| {
                ui.label(RichText::new(match self.formal_availability {
                    FormalAvailability::Loading => "FORMAL ARTIFACTS LOADING",
                    FormalAvailability::Unavailable(_) => "FORMAL ARTIFACTS UNAVAILABLE",
                    _ => "FORMAL ARTIFACTS MISSING",
                }).monospace().strong());
                ui.label(formal_unavailable_text());
                match &self.formal_availability {
                    FormalAvailability::Loading => { ui.spinner(); ui.label("Validating optional formal artifacts…"); }
                    FormalAvailability::Unavailable(error) => { ui.colored_label(Color32::from_rgb(150, 45, 45), error); }
                    _ => {}
                }
                ui.label(
                    "Formal mode is deliberately secondary. Predictive play remains available without these artifacts.",
                );
            });
            return;
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Recompute from current history").clicked() {
                self.mode = GuiSolverMode::FormalOptimal;
                self.schedule_recompute();
            }
            ui.label(format!("{} observations applied", self.observations.len()));
        });
        ui.add_space(12.0);
        if let Some(explanation) = &self.formal_explanation {
            ui.group(|ui| {
                ui.label(RichText::new("VERIFIED POLICY STATE").monospace().strong());
                policy_row(ui, "Model", &explanation.model_id);
                policy_row(ui, "Manifest", &explanation.manifest_hash);
                policy_row(
                    ui,
                    "Objective",
                    &format!(
                        "worst {} · expected {:.6}",
                        explanation.objective.worst_case_depth,
                        explanation.objective.expected_guesses
                    ),
                );
                policy_row(
                    ui,
                    "Surviving answers",
                    &explanation.surviving_answers.to_string(),
                );
            });
        }
        ui.add_space(12.0);
        for suggestion in &self.formal_suggestions {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new(suggestion.word.to_ascii_uppercase())
                        .monospace()
                        .strong(),
                );
                ui.label(format!(
                    "worst {} · expected {:.6}",
                    suggestion.objective.worst_case_depth, suggestion.objective.expected_guesses
                ));
            });
            ui.separator();
        }
        if self.computing {
            ui.spinner();
        } else if !self.status.message().is_empty() {
            ui.colored_label(self.status.color(), self.status.message());
        }
    }
}
