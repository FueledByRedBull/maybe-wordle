use super::*;

impl WordleGuiApp {
    pub(super) fn show_narrow_play(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label(RichText::new("BOARD / HISTORY").monospace().strong());
            show_history_timeline(ui, &self.observations);
            ui.add_space(8.0);
            show_game_board(
                ui,
                &self.observations,
                &self.current_guess,
                self.current_feedback,
            );
        });
        ui.add_space(12.0);
        ui.group(|ui| {
            ui.label(RichText::new("NEXT ACTION").monospace().strong());
            match self.mode {
                GuiSolverMode::Predictive => {
                    let sorted = sorted_predictive_indices(
                        &self.predictive_suggestions,
                        self.suggestion_sort,
                        self.suggestion_sort_descending,
                    );
                    for index in sorted.into_iter().take(8) {
                        let suggestion = &self.predictive_suggestions[index];
                        let label = if let Some(value) = &suggestion.finite_value {
                            format!(
                                "{}  {}  failure risk {}  expected remaining attempts {}",
                                suggestion.word.to_ascii_uppercase(),
                                finite_quality_label(value.quality),
                                format_finite_failure_risk(value),
                                format_finite_expected_attempts(value),
                            )
                        } else {
                            format!(
                                "{}  {}  solve {:.3}  H {:.3}  cost {}",
                                suggestion.word.to_ascii_uppercase(),
                                suggestion_method(suggestion),
                                suggestion.solve_probability,
                                suggestion.entropy,
                                format_suggestion_cost(suggestion)
                            )
                        };
                        if ui
                            .selectable_label(
                                self.selected_suggestion.as_deref()
                                    == Some(suggestion.word.as_str()),
                                label,
                            )
                            .clicked()
                        {
                            self.selected_suggestion = Some(suggestion.word.clone());
                        }
                    }
                    ui.separator();
                    show_suggestion_inspector(
                        ui,
                        self.selected_suggestion.as_deref().and_then(|word| {
                            self.predictive_suggestions
                                .iter()
                                .find(|suggestion| suggestion.word == word)
                        }),
                        self.predictive_artifact_state,
                        self.predictive_recovery_mode,
                        self.predictive_execution.as_ref(),
                    );
                }
                GuiSolverMode::Absurdle => {
                    for suggestion in &self.absurdle_suggestions {
                        ui.label(format!(
                            "{} · worst {} · entropy {:.3}",
                            suggestion.word.to_ascii_uppercase(),
                            suggestion.largest_bucket_size,
                            suggestion.entropy
                        ));
                    }
                }
                GuiSolverMode::FormalOptimal => {
                    for suggestion in &self.formal_suggestions {
                        ui.label(format!(
                            "{} · worst {} · expected {:.6}",
                            suggestion.word.to_ascii_uppercase(),
                            suggestion.objective.worst_case_depth,
                            suggestion.objective.expected_guesses
                        ));
                    }
                }
            }
        });
        if self.mode == GuiSolverMode::Predictive {
            ui.add_space(12.0);
            egui::CollapsingHeader::new("Candidate browser")
                .default_open(false)
                .show(ui, |ui| {
                    show_candidate_browser(
                        ui,
                        &mut self.candidate_filter,
                        &self.predictive_candidates,
                    );
                });
        }
    }

    pub(super) fn show_play(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal_wrapped(|ui| {
            let formal_available = self.formal_solver.is_some();
            let previous_mode = self.mode;
            ui.label("Tool");
            egui::ComboBox::from_id_salt("solver-mode")
                .selected_text(self.mode.label(formal_available))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.mode,
                        GuiSolverMode::Predictive,
                        GuiSolverMode::Predictive.label(formal_available),
                    );
                    ui.selectable_value(
                        &mut self.mode,
                        GuiSolverMode::Absurdle,
                        GuiSolverMode::Absurdle.label(formal_available),
                    );
                    if formal_available {
                        ui.selectable_value(
                            &mut self.mode,
                            GuiSolverMode::FormalOptimal,
                            GuiSolverMode::FormalOptimal.label(formal_available),
                        );
                    }
                });
            if !formal_available {
                ui.label(RichText::new(formal_unavailable_text()).small().weak());
            }
            if self.mode != previous_mode {
                self.schedule_recompute();
            }
            if self.mode == GuiSolverMode::Predictive {
                ui.separator();
                let previous_profile = self.search_profile;
                ui.label("Profile");
                egui::ComboBox::from_id_salt("predictive-search-profile")
                    .selected_text(self.search_profile.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.search_profile,
                            GuiSearchProfile::Configured,
                            GuiSearchProfile::Configured.label(),
                        );
                        ui.selectable_value(
                            &mut self.search_profile,
                            GuiSearchProfile::FiniteFast,
                            GuiSearchProfile::FiniteFast.label(),
                        );
                        ui.selectable_value(
                            &mut self.search_profile,
                            GuiSearchProfile::FiniteStrong,
                            GuiSearchProfile::FiniteStrong.label(),
                        );
                    });
                if self.search_profile != previous_profile {
                    self.schedule_recompute();
                }
            }
        });

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Date");
            let mut date_changed = false;
            ui.add_enabled_ui(self.mode == GuiSolverMode::Predictive, |ui| {
                date_changed = ui
                    .add_sized(
                        [120.0, 24.0],
                        egui::TextEdit::singleline(&mut self.date_text),
                    )
                    .changed();
            });
            if self.mode == GuiSolverMode::FormalOptimal {
                if let Some(explanation) = &self.formal_explanation {
                    ui.label(format!(
                        "model {} / manifest {}",
                        explanation.model_id, explanation.manifest_hash
                    ));
                } else {
                    ui.label(format!("model {}", DEFAULT_FORMAL_MODEL_ID));
                }
            }
            ui.label("Top");
            let top_changed = ui.add(egui::Slider::new(&mut self.top, 3..=20)).changed();
            if ui
                .add_enabled(self.can_apply_row(), egui::Button::new("Apply row"))
                .clicked()
            {
                self.commit_current_row();
            }
            if ui.button("Undo").clicked() {
                self.observations.pop();
                self.schedule_recompute();
            }
            if ui.button("Reset").clicked() {
                self.observations.clear();
                self.apply_board_action(BoardAction::Reset);
                self.selected_suggestion = None;
                self.schedule_recompute();
            }
            if self.mode == GuiSolverMode::Predictive {
                let hard_changed = ui.checkbox(&mut self.hard_mode, "Hard Mode").changed();
                if hard_changed {
                    self.schedule_recompute();
                }
                let force_changed = ui
                    .checkbox(&mut self.force_in_two_only, "Force In 2 Only")
                    .changed();
                if force_changed {
                    self.schedule_recompute();
                }
            }
            if date_changed || top_changed {
                self.schedule_recompute();
            }
        });

        ui.add_space(16.0);
        let mut keyboard_apply = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Guess");
            let response = ui.add_sized(
                [120.0, 30.0],
                egui::TextEdit::singleline(&mut self.current_guess).hint_text("e.g. crane"),
            );
            if response.changed() {
                self.apply_board_action(BoardAction::ReplaceGuess(self.current_guess.clone()));
            }
            keyboard_apply |=
                response.lost_focus() && ctx.input(|input| input.key_pressed(egui::Key::Enter));

            ui.label("Feedback");
            let mut edited_feedback = self.feedback_code.clone();
            let feedback_response = ui.add_sized(
                [88.0, 30.0],
                egui::TextEdit::singleline(&mut edited_feedback)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("01210"),
            );
            if feedback_response.changed() {
                self.update_feedback_code(edited_feedback);
            }
            keyboard_apply |= feedback_response.lost_focus()
                && ctx.input(|input| input.key_pressed(egui::Key::Enter));

            let mut clicked_tile = None;
            for (index, value) in self.current_feedback.iter_mut().enumerate() {
                let (_, color, foreground) = tile_label_and_color(*value);
                let letter = self
                    .current_guess
                    .chars()
                    .nth(index)
                    .map(|character| character.to_ascii_uppercase().to_string())
                    .unwrap_or_else(|| " ".to_string());
                let label = format!("{letter}\n{}", feedback_accessible_label(*value));
                if ui
                    .add_sized(
                        [58.0, 58.0],
                        egui::Button::new(
                            RichText::new(label).size(15.0).strong().color(foreground),
                        )
                        .fill(color),
                    )
                    .clicked()
                {
                    clicked_tile = Some(index);
                }
            }
            if let Some(index) = clicked_tile {
                self.apply_board_action(BoardAction::CycleTile(index));
            }
        });
        if keyboard_apply && self.can_apply_row() {
            self.commit_current_row();
        }
        if let Err(error) = parse_feedback(&self.feedback_code) {
            ui.colored_label(
                Color32::from_rgb(150, 45, 45),
                format!("Feedback draft: {error}. Apply is disabled."),
            );
        }
        ui.label(
            RichText::new(
                "Keyboard: type five feedback symbols (0/1/2 or b/y/g) and press Enter. Tile labels supplement color; repeated letters must match Wordle exactly.",
            )
            .small()
            .color(Color32::from_rgb(92, 72, 54)),
        );

        ui.add_space(12.0);
        match self.mode {
            GuiSolverMode::Predictive => {
                let mut summary = format!(
                    "Remaining candidates: {}   total weight: {:.4}",
                    self.surviving_count, self.total_weight
                );
                if let Some(mode) = self.predictive_recovery_mode {
                    summary.push_str(&format!("   recovery: {}", mode.label()));
                }
                ui.label(
                    RichText::new(summary)
                        .strong()
                        .color(Color32::from_rgb(67, 53, 39)),
                );
                let banner = if self.predictive_execution.is_none() {
                    "Awaiting current search result"
                } else if self.predictive_execution.as_ref().is_some_and(|execution| {
                    execution.route == crate::predictive::PredictiveRegime::Finite
                }) {
                    predictive_finite_banner_text(self.search_profile)
                } else {
                    predictive_banner_text(self.predictive_artifact_state)
                };
                ui.label(RichText::new(banner).color(Color32::from_rgb(92, 72, 54)));
                if let Some(execution) = &self.predictive_execution {
                    ui.label(RichText::new(execution.summary()).small());
                }
                if let Some(search) = &self.predictive_finite_search {
                    ui.label(
                        RichText::new(format_finite_search_summary(search))
                            .small()
                            .color(Color32::from_rgb(92, 72, 54)),
                    );
                }
                if !self.predictive_model_metadata.is_empty() {
                    ui.label(
                        RichText::new(&self.predictive_model_metadata)
                            .small()
                            .color(Color32::from_rgb(92, 72, 54)),
                    );
                }
                if self.predictive_execution.as_ref().is_some_and(|execution| {
                    execution.route != crate::predictive::PredictiveRegime::Finite
                }) && let Some(message) = predictive_reply_book_text(
                    self.observations.len(),
                    self.predictive_artifact_state,
                ) {
                    ui.label(RichText::new(message).color(Color32::from_rgb(92, 72, 54)));
                }
            }
            GuiSolverMode::Absurdle => {
                ui.label(
                    RichText::new(format!("Remaining candidates: {}", self.surviving_count))
                        .strong()
                        .color(Color32::from_rgb(67, 53, 39)),
                );
            }
            GuiSolverMode::FormalOptimal => {
                let summary = if let Some(explanation) = &self.formal_explanation {
                    format!(
                        "Remaining candidates: {}   worst-case depth: {}   expected guesses: {:.6}",
                        explanation.surviving_answers,
                        explanation.objective.worst_case_depth,
                        explanation.objective.expected_guesses
                    )
                } else {
                    format!("Remaining candidates: {}", self.surviving_count)
                };
                ui.label(
                    RichText::new(summary)
                        .strong()
                        .color(Color32::from_rgb(67, 53, 39)),
                );
            }
        }

        if !self.status.message().is_empty() {
            ui.add_space(8.0);
            let color = self.status.color();
            ui.colored_label(color, self.status.message());
        }
        if self.request_sender.is_none() && ui.button("Retry worker").clicked() {
            self.retry_worker();
        }

        ui.add_space(16.0);
        if is_compact_layout(ui.available_width()) {
            self.show_narrow_play(ui);
            return;
        }
        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Game Board");
                ui.label(
                    RichText::new(if self.mode == GuiSolverMode::Absurdle {
                        format!(
                            "{} rows applied (Absurdle has no six-turn limit)",
                            self.observations.len()
                        )
                    } else {
                        format!("{} of 6 rows applied", self.observations.len())
                    })
                    .color(Color32::from_rgb(92, 72, 54)),
                );
                ui.add_space(8.0);
                show_history_timeline(ui, &self.observations);
                ui.add_space(8.0);
                show_game_board(
                    ui,
                    &self.observations,
                    &self.current_guess,
                    self.current_feedback,
                );
                ui.separator();
                show_candidate_browser(ui, &mut self.candidate_filter, &self.predictive_candidates);
            });

            columns[1].group(|ui| {
                let (heading, summary) = match self.mode {
                    GuiSolverMode::Predictive => (
                        "Wordle Suggestions",
                        if self
                            .search_profile
                            .finite_options(&self.predictive_solver)
                            .is_some()
                        {
                            "Ranks by modeled six-turn failure risk, then expected attempts."
                        } else {
                            "Ranks guesses by predictive expected progress."
                        },
                    ),
                    GuiSolverMode::Absurdle => (
                        "Absurdle Suggestions",
                        "Ranks guesses by minimizing the largest surviving bucket.",
                    ),
                    GuiSolverMode::FormalOptimal => (
                        "Formal Suggestions",
                        "Ranks guesses by the formal optimal-policy objective.",
                    ),
                };
                ui.heading(heading);
                ui.label(RichText::new(summary).color(Color32::from_rgb(92, 72, 54)));
                ui.add_space(8.0);
                match self.mode {
                    GuiSolverMode::Predictive => {
                        if self.predictive_suggestions.is_empty() {
                            ui.label(
                                RichText::new("No suggestions match the current filters.")
                                    .color(Color32::from_rgb(92, 72, 54)),
                            );
                        } else {
                            let finite_table = self
                                .search_profile
                                .finite_options(&self.predictive_solver)
                                .is_some();
                            let sorted_indices = sorted_predictive_indices(
                                &self.predictive_suggestions,
                                self.suggestion_sort,
                                self.suggestion_sort_descending,
                            );
                            egui::ScrollArea::vertical()
                                .id_salt("predictive-suggestion-scroll")
                                .max_height(330.0)
                                .show(ui, |ui| {
                                    egui::Grid::new("predictive-suggestion-table")
                                        .striped(true)
                                        .min_col_width(54.0)
                                        .show(ui, |ui| {
                                            if ui
                                                .button(
                                                    self.sort_label("Rank", SuggestionSort::Rank),
                                                )
                                                .clicked()
                                            {
                                                self.toggle_suggestion_sort(SuggestionSort::Rank);
                                            }
                                            ui.label("Word");
                                            if finite_table {
                                                ui.label("Quality");
                                                ui.label("Failure risk");
                                                ui.label("Expected attempts");
                                            } else {
                                                ui.label("Method");
                                                if ui
                                                    .button(self.sort_label(
                                                        "Solve",
                                                        SuggestionSort::SolveProbability,
                                                    ))
                                                    .clicked()
                                                {
                                                    self.toggle_suggestion_sort(
                                                        SuggestionSort::SolveProbability,
                                                    );
                                                }
                                                if ui
                                                    .button(self.sort_label(
                                                        "Entropy",
                                                        SuggestionSort::Entropy,
                                                    ))
                                                    .clicked()
                                                {
                                                    self.toggle_suggestion_sort(
                                                        SuggestionSort::Entropy,
                                                    );
                                                }
                                                if ui
                                                    .button(self.sort_label(
                                                        "Remain",
                                                        SuggestionSort::ExpectedRemaining,
                                                    ))
                                                    .clicked()
                                                {
                                                    self.toggle_suggestion_sort(
                                                        SuggestionSort::ExpectedRemaining,
                                                    );
                                                }
                                            }
                                            if ui
                                                .button(self.sort_label(
                                                    "Worst",
                                                    SuggestionSort::WorstBucket,
                                                ))
                                                .clicked()
                                            {
                                                self.toggle_suggestion_sort(
                                                    SuggestionSort::WorstBucket,
                                                );
                                            }
                                            if !finite_table {
                                                ui.label("Cost");
                                            }
                                            ui.end_row();

                                            for index in sorted_indices {
                                                let suggestion =
                                                    &self.predictive_suggestions[index];
                                                ui.label((index + 1).to_string());
                                                if ui
                                                    .selectable_label(
                                                        self.selected_suggestion.as_deref()
                                                            == Some(suggestion.word.as_str()),
                                                        RichText::new(
                                                            suggestion.word.to_ascii_uppercase(),
                                                        )
                                                        .strong()
                                                        .color(Color32::from_rgb(42, 49, 43)),
                                                    )
                                                    .clicked()
                                                {
                                                    self.selected_suggestion =
                                                        Some(suggestion.word.clone());
                                                }
                                                if finite_table {
                                                    if let Some(value) = &suggestion.finite_value {
                                                        ui.label(finite_quality_label(
                                                            value.quality,
                                                        ));
                                                        ui.label(format_finite_failure_risk(value));
                                                        ui.label(format_finite_expected_attempts(
                                                            value,
                                                        ));
                                                    } else {
                                                        ui.label("pending");
                                                        ui.label("pending");
                                                        ui.label("unevaluated");
                                                    }
                                                } else {
                                                    ui.label(suggestion_method(suggestion));
                                                    ui.label(format!(
                                                        "{:.3}",
                                                        suggestion.solve_probability
                                                    ));
                                                    ui.label(format!("{:.3}", suggestion.entropy));
                                                    ui.label(format!(
                                                        "{:.2}",
                                                        suggestion.expected_remaining
                                                    ));
                                                }
                                                ui.label(
                                                    suggestion
                                                        .worst_non_green_bucket_size
                                                        .to_string(),
                                                );
                                                if !finite_table {
                                                    ui.label(format_suggestion_cost(suggestion));
                                                }
                                                ui.end_row();
                                            }
                                        });
                                });
                            ui.separator();
                            show_suggestion_inspector(
                                ui,
                                self.selected_suggestion.as_deref().and_then(|word| {
                                    self.predictive_suggestions
                                        .iter()
                                        .find(|suggestion| suggestion.word == word)
                                }),
                                self.predictive_artifact_state,
                                self.predictive_recovery_mode,
                                self.predictive_execution.as_ref(),
                            );
                        }
                    }
                    GuiSolverMode::Absurdle => {
                        for suggestion in &self.absurdle_suggestions {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new(suggestion.word.to_ascii_uppercase())
                                        .size(18.0)
                                        .strong()
                                        .color(Color32::from_rgb(58, 44, 32)),
                                );
                                ui.label(format!("worst {}", suggestion.largest_bucket_size));
                                ui.label(format!(
                                    "second {}",
                                    suggestion.second_largest_bucket_size
                                ));
                                ui.label(format!("multi {}", suggestion.multi_answer_bucket_count));
                                ui.label(format!("entropy {:.4}", suggestion.entropy));
                            });
                            ui.separator();
                        }
                    }
                    GuiSolverMode::FormalOptimal => {
                        for suggestion in &self.formal_suggestions {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new(suggestion.word.to_ascii_uppercase())
                                        .size(18.0)
                                        .strong()
                                        .color(Color32::from_rgb(58, 44, 32)),
                                );
                                ui.label(format!(
                                    "worst {}",
                                    suggestion.objective.worst_case_depth
                                ));
                                ui.label(format!(
                                    "expected {:.6}",
                                    suggestion.objective.expected_guesses
                                ));
                                ui.label(format!(
                                    "buckets {}",
                                    suggestion
                                        .bucket_sizes
                                        .iter()
                                        .map(|size| size.to_string())
                                        .collect::<Vec<_>>()
                                        .join(",")
                                ));
                            });
                            ui.separator();
                        }
                    }
                }
            });
        });
    }
}

pub(super) fn show_suggestion_inspector(
    ui: &mut egui::Ui,
    suggestion: Option<&Suggestion>,
    artifact_state: PredictiveArtifactState,
    recovery_mode: Option<RecoveryMode>,
    execution: Option<&SearchExecution>,
) {
    ui.heading("Suggestion Inspector");
    let Some(suggestion) = suggestion else {
        ui.label("Select a suggestion row to inspect its evidence.");
        return;
    };
    ui.label(
        RichText::new(suggestion.word.to_ascii_uppercase())
            .size(22.0)
            .strong(),
    );
    if let Some(execution) = execution {
        ui.label(execution.summary());
    }
    if let Some(value) = &suggestion.finite_value {
        ui.label(format!(
            "{} · failure risk {} · expected remaining attempts {}",
            finite_quality_label(value.quality),
            format_finite_failure_risk(value),
            format_finite_expected_attempts(value),
        ));
        if value.quality == FiniteSearchQuality::UpperBound {
            ui.label(
                "Completed rollout: the combined lexicographic objective is bounded, while these two displayed scalars are rollout values rather than individual bounds.",
            );
        }
    } else {
        ui.label(format!(
            "{} ranking · solve probability {:.4} · entropy {:.4} bits · expected remaining {:.2}",
            suggestion_method(suggestion),
            suggestion.solve_probability,
            suggestion.entropy,
            suggestion.expected_remaining
        ));
    }
    ui.label(format!(
        "Worst non-green bucket {} answers ({:.1}% posterior mass); {} large buckets; {} dangerous-mass buckets.",
        suggestion.worst_non_green_bucket_size,
        suggestion.largest_non_green_bucket_mass * 100.0,
        suggestion.large_non_green_bucket_count,
        suggestion.dangerous_mass_bucket_count
    ));
    if suggestion.force_in_two {
        ui.colored_label(
            Color32::from_rgb(37, 99, 72),
            "Force-in-two: every modeled non-green reply has a finishing guess.",
        );
    }
    ui.label(format!(
        "Artifact source: {}. Recovery: {}.",
        if matches!(suggestion.value_kind, SuggestionValueKind::Finite(_)) {
            "bounded finite search (predictive books not used)"
        } else {
            predictive_banner_text(artifact_state)
        },
        recovery_mode.map_or("not active", RecoveryMode::label)
    ));
}

pub(super) fn matching_candidates<'a>(
    candidates: &'a [PredictiveCandidateSummary],
    filter: &str,
) -> Vec<&'a PredictiveCandidateSummary> {
    let normalized = filter.trim().to_ascii_lowercase();
    candidates
        .iter()
        .filter(|candidate| candidate.word.contains(&normalized))
        .collect()
}

pub(super) fn candidate_csv(candidates: &[&PredictiveCandidateSummary]) -> String {
    let mut csv = String::from("word,probability,modeled_weight,fallback_support\n");
    for candidate in candidates {
        csv.push_str(&format!(
            "{},{:.12},{:.12},{}\n",
            candidate.word,
            candidate.probability,
            candidate.modeled_weight,
            candidate.fallback_support
        ));
    }
    csv
}

pub(super) fn show_candidate_browser(
    ui: &mut egui::Ui,
    filter: &mut String,
    candidates: &[PredictiveCandidateSummary],
) -> egui::scroll_area::ScrollAreaOutput<()> {
    ui.heading("Candidate Browser");
    let mut copy = false;
    let mut filter_changed = false;
    ui.horizontal(|ui| {
        filter_changed = ui
            .add_sized(
                [180.0, 24.0],
                egui::TextEdit::singleline(filter).hint_text("Filter answers"),
            )
            .changed();
        copy = ui.button("Copy CSV").clicked();
    });
    let matching = matching_candidates(candidates, filter);
    if copy {
        ui.ctx().copy_text(candidate_csv(&matching));
    }
    ui.label(
        RichText::new(format!(
            "{} matching / {} live candidates",
            matching.len(),
            candidates.len()
        ))
        .small()
        .color(Color32::from_rgb(92, 72, 54)),
    );
    let mut scroll = egui::ScrollArea::vertical()
        .id_salt("predictive-candidate-scroll")
        .max_height(150.0);
    if filter_changed {
        scroll = scroll.vertical_scroll_offset(0.0);
    }
    let row_height = ui
        .text_style_height(&egui::TextStyle::Body)
        .max(ui.spacing().interact_size.y);
    scroll.show_rows(ui, row_height, matching.len(), |ui, rows| {
        for index in rows {
            let candidate = matching[index];
            ui.horizontal(|ui| {
                ui.add_sized(
                    [90.0, row_height],
                    egui::Label::new(RichText::new(candidate.word.to_ascii_uppercase()).strong()),
                );
                ui.add_sized(
                    [90.0, row_height],
                    egui::Label::new(format!("{:.5}%", candidate.probability * 100.0)),
                );
                ui.label(if candidate.fallback_support {
                    "fallback"
                } else {
                    "modeled"
                });
            });
        }
    })
}

pub(super) fn show_game_board(
    ui: &mut egui::Ui,
    observations: &[(String, u8)],
    current_guess: &str,
    current_feedback: [u8; 5],
) {
    ui.label(
        RichText::new("Tile markers: A absent / P present / C correct")
            .small()
            .color(Color32::from_rgb(92, 72, 54)),
    );
    ui.add_space(4.0);
    for row in 0..6.max(observations.len()) {
        ui.horizontal(|ui| {
            let (letters, feedback, applied) = if let Some((guess, pattern)) = observations.get(row)
            {
                (
                    guess.chars().collect::<Vec<_>>(),
                    decode_feedback(*pattern),
                    true,
                )
            } else if row == observations.len() {
                (
                    current_guess.chars().collect::<Vec<_>>(),
                    current_feedback,
                    false,
                )
            } else {
                (Vec::new(), [0; 5], false)
            };
            for (column, value) in feedback.iter().copied().enumerate() {
                let letter = letters
                    .get(column)
                    .copied()
                    .map(|character| character.to_ascii_uppercase().to_string())
                    .unwrap_or_else(|| " ".to_string());
                let color = if applied {
                    tile_label_and_color(value).1
                } else {
                    Color32::from_rgb(225, 218, 208)
                };
                let marker = if applied { feedback_marker(value) } else { " " };
                egui::Frame::default()
                    .fill(color)
                    .corner_radius(3.0)
                    .show(ui, |ui| {
                        ui.set_width(48.0);
                        ui.set_min_height(48.0);
                        if applied || letter != " " {
                            ui.centered_and_justified(|ui| {
                                ui.label(
                                    RichText::new(format!("{letter}\n{marker}"))
                                        .size(15.0)
                                        .strong()
                                        .color(if applied {
                                            tile_label_and_color(value).2
                                        } else {
                                            Color32::from_rgb(42, 49, 43)
                                        }),
                                );
                            });
                        }
                    });
            }
        });
        ui.add_space(4.0);
    }
}

pub(super) fn show_history_timeline(ui: &mut egui::Ui, observations: &[(String, u8)]) {
    if observations.is_empty() {
        ui.label(
            RichText::new("No observations yet")
                .monospace()
                .small()
                .color(Color32::from_rgb(92, 72, 54)),
        );
        return;
    }
    ui.horizontal_wrapped(|ui| {
        for (index, (guess, pattern)) in observations.iter().enumerate() {
            let code = feedback_code(decode_feedback(*pattern));
            egui::Frame::default()
                .fill(Color32::from_rgb(239, 228, 211))
                .corner_radius(5.0)
                .inner_margin(egui::Margin::symmetric(7, 4))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} {} {}",
                            index + 1,
                            guess.to_ascii_uppercase(),
                            code
                        ))
                        .monospace()
                        .small(),
                    );
                });
        }
    });
}
