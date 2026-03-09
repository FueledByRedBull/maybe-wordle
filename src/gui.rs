use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::NaiveDate;
use eframe::egui::{self, Color32, RichText};

use crate::{
    config::PriorConfig,
    data::ProjectPaths,
    formal::{
        DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime, FormalStateExplanation, FormalSuggestion,
        artifacts_exist,
    },
    scoring::parse_feedback,
    solver::{SolveState, Solver, Suggestion},
};

pub fn run_gui(root: PathBuf) -> Result<()> {
    let paths = ProjectPaths::new(root);
    let config = PriorConfig::load_or_create(&paths.config_prior)?;
    let predictive_solver = Solver::from_paths(&paths, &config)?;
    let formal_solver = if artifacts_exist(&paths, DEFAULT_FORMAL_MODEL_ID) {
        Some(FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID)?)
    } else {
        None
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 780.0])
            .with_min_inner_size([760.0, 640.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Maybe Wordle",
        native_options,
        Box::new(move |_cc| {
            Ok(Box::new(WordleGuiApp::new(
                predictive_solver,
                formal_solver,
            )))
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuiSolverMode {
    FormalOptimal,
    Predictive,
}

struct WordleGuiApp {
    predictive_solver: Solver,
    formal_solver: Option<FormalPolicyRuntime>,
    mode: GuiSolverMode,
    date_text: String,
    current_guess: String,
    current_feedback: [u8; 5],
    observations: Vec<(String, u8)>,
    predictive_suggestions: Vec<Suggestion>,
    formal_suggestions: Vec<FormalSuggestion>,
    surviving_count: usize,
    total_weight: f64,
    top: usize,
    status: String,
    formal_explanation: Option<FormalStateExplanation>,
}

impl WordleGuiApp {
    fn new(predictive_solver: Solver, formal_solver: Option<FormalPolicyRuntime>) -> Self {
        let date_text = Solver::today().format("%Y-%m-%d").to_string();
        let mode = if formal_solver.is_some() {
            GuiSolverMode::FormalOptimal
        } else {
            GuiSolverMode::Predictive
        };
        let mut app = Self {
            predictive_solver,
            formal_solver,
            mode,
            date_text,
            current_guess: String::new(),
            current_feedback: [0; 5],
            observations: Vec::new(),
            predictive_suggestions: Vec::new(),
            formal_suggestions: Vec::new(),
            surviving_count: 0,
            total_weight: 0.0,
            top: 10,
            status: String::new(),
            formal_explanation: None,
        };
        app.recompute();
        app
    }

    fn recompute(&mut self) {
        match self.mode {
            GuiSolverMode::Predictive => match self.try_recompute_predictive() {
                Ok((state, suggestions)) => {
                    self.surviving_count = state.surviving.len();
                    self.total_weight = state.total_weight;
                    self.predictive_suggestions = suggestions;
                    self.formal_suggestions.clear();
                    self.formal_explanation = None;
                    self.status.clear();
                }
                Err(error) => self.set_error(error),
            },
            GuiSolverMode::FormalOptimal => match self.try_recompute_formal() {
                Ok((explanation, suggestions)) => {
                    self.surviving_count = explanation.surviving_answers;
                    self.total_weight = 0.0;
                    self.formal_explanation = Some(explanation);
                    self.formal_suggestions = suggestions;
                    self.predictive_suggestions.clear();
                    self.status.clear();
                }
                Err(error) => self.set_error(error),
            },
        }
    }

    fn try_recompute_predictive(&self) -> Result<(SolveState, Vec<Suggestion>)> {
        let date = NaiveDate::parse_from_str(&self.date_text, "%Y-%m-%d")
            .with_context(|| format!("invalid date: {}", self.date_text))?;
        let state = self
            .predictive_solver
            .apply_history(date, &self.observations)?;
        let suggestions = self.predictive_solver.suggestions(&state, self.top)?;
        Ok((state, suggestions))
    }

    fn try_recompute_formal(&self) -> Result<(FormalStateExplanation, Vec<FormalSuggestion>)> {
        let runtime = self
            .formal_solver
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("formal-optimal artifacts are not available"))?;
        let state = runtime.apply_history(&self.observations)?;
        let explanation = runtime.explain_state(&state, self.top)?;
        let suggestions = runtime.suggest(&state, self.top)?;
        Ok((explanation, suggestions))
    }

    fn set_error(&mut self, error: anyhow::Error) {
        self.status = error.to_string();
        self.predictive_suggestions.clear();
        self.formal_suggestions.clear();
        self.formal_explanation = None;
        self.surviving_count = 0;
        self.total_weight = 0.0;
    }

    fn commit_current_row(&mut self) {
        let guess = self.current_guess.trim().to_ascii_lowercase();
        if guess.is_empty() {
            self.recompute();
            return;
        }
        match self.row_pattern() {
            Ok(pattern) => {
                self.observations.push((guess, pattern));
                self.current_guess.clear();
                self.current_feedback = [0; 5];
                self.recompute();
            }
            Err(error) => self.status = error.to_string(),
        }
    }

    fn row_pattern(&self) -> Result<u8> {
        if self.current_guess.trim().len() != 5 {
            anyhow::bail!("current guess must be exactly 5 letters");
        }
        let feedback = self
            .current_feedback
            .iter()
            .map(|value| char::from(b'0' + *value))
            .collect::<String>();
        parse_feedback(&feedback)
    }
}

impl eframe::App for WordleGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(Color32::from_rgb(246, 240, 232))
                    .inner_margin(24.0),
            )
            .show(ctx, |ui| {
                ui.visuals_mut().widgets.inactive.bg_fill = Color32::from_rgb(239, 228, 211);
                ui.visuals_mut().widgets.hovered.bg_fill = Color32::from_rgb(226, 214, 195);
                ui.visuals_mut().widgets.active.bg_fill = Color32::from_rgb(212, 198, 177);

                ui.heading(
                    RichText::new("Maybe Wordle")
                        .size(30.0)
                        .color(Color32::from_rgb(58, 44, 32)),
                );
                ui.label(
                    RichText::new(
                        "Click each tile to cycle gray, yellow, green, then hit Suggest.",
                    )
                    .color(Color32::from_rgb(92, 72, 54)),
                );
                ui.add_space(12.0);

                ui.horizontal(|ui| {
                    let formal_available = self.formal_solver.is_some();
                    ui.label("Mode");
                    ui.selectable_value(
                        &mut self.mode,
                        if formal_available {
                            GuiSolverMode::FormalOptimal
                        } else {
                            GuiSolverMode::Predictive
                        },
                        if formal_available {
                            "Formal Optimal"
                        } else {
                            "Formal Unavailable"
                        },
                    );
                    ui.selectable_value(&mut self.mode, GuiSolverMode::Predictive, "Predictive");
                    if !formal_available {
                        ui.colored_label(Color32::from_rgb(150, 45, 45), "build-optimal-policy first");
                    }
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Date");
                    ui.add_enabled_ui(self.mode == GuiSolverMode::Predictive, |ui| {
                        ui.add_sized([120.0, 24.0], egui::TextEdit::singleline(&mut self.date_text));
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
                    ui.add(egui::Slider::new(&mut self.top, 3..=20));
                    if ui.button("Suggest").clicked() {
                        self.commit_current_row();
                    }
                    if ui.button("Undo").clicked() {
                        self.observations.pop();
                        self.recompute();
                    }
                    if ui.button("Reset").clicked() {
                        self.observations.clear();
                        self.current_guess.clear();
                        self.current_feedback = [0; 5];
                        self.recompute();
                    }
                });

                ui.add_space(16.0);
                ui.horizontal(|ui| {
                    ui.label("Guess");
                    let response = ui.add_sized(
                        [120.0, 30.0],
                        egui::TextEdit::singleline(&mut self.current_guess).hint_text("crane"),
                    );
                    if response.changed() {
                        self.current_guess = self
                            .current_guess
                            .chars()
                            .filter(|character| character.is_ascii_alphabetic())
                            .take(5)
                            .collect::<String>()
                            .to_ascii_lowercase();
                    }

                    for value in &mut self.current_feedback {
                        let (label, color) = tile_label_and_color(*value);
                        if ui
                            .add_sized(
                                [58.0, 58.0],
                                egui::Button::new(RichText::new(label).size(18.0).strong())
                                    .fill(color),
                            )
                            .clicked()
                        {
                            *value = (*value + 1) % 3;
                        }
                    }
                });

                ui.add_space(12.0);
                match self.mode {
                    GuiSolverMode::Predictive => {
                        ui.label(
                            RichText::new(format!(
                                "Remaining candidates: {}   total weight: {:.4}",
                                self.surviving_count, self.total_weight
                            ))
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

                if !self.status.is_empty() {
                    ui.add_space(8.0);
                    ui.colored_label(Color32::from_rgb(150, 45, 45), &self.status);
                }

                ui.add_space(16.0);
                ui.columns(2, |columns| {
                    columns[0].group(|ui| {
                        ui.heading("Applied Rows");
                        ui.add_space(8.0);
                        if self.observations.is_empty() {
                            ui.label("No guesses committed yet.");
                        } else {
                            for (guess, pattern) in &self.observations {
                                ui.horizontal(|ui| {
                                    for (character, value) in
                                        guess.chars().zip(decode_pattern(pattern))
                                    {
                                        let (_, color) = tile_label_and_color(value);
                                        ui.add_sized(
                                            [42.0, 42.0],
                                            egui::Button::new(
                                                RichText::new(
                                                    character.to_ascii_uppercase().to_string(),
                                                )
                                                .strong(),
                                            )
                                            .fill(color),
                                        );
                                    }
                                });
                                ui.add_space(6.0);
                            }
                        }
                    });

                    columns[1].group(|ui| {
                        ui.heading("Suggestions");
                        ui.add_space(8.0);
                        match self.mode {
                            GuiSolverMode::Predictive => {
                                for suggestion in &self.predictive_suggestions {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            RichText::new(suggestion.word.to_ascii_uppercase())
                                                .size(18.0)
                                                .strong()
                                                .color(Color32::from_rgb(58, 44, 32)),
                                        );
                                        ui.label(format!("entropy {:.4}", suggestion.entropy));
                                        ui.label(format!("solve {:.4}", suggestion.solve_probability));
                                        ui.label(format!("remain {:.2}", suggestion.expected_remaining));
                                        if let Some(exact_cost) = suggestion.exact_cost {
                                            ui.label(format!("exact {:.4}", exact_cost));
                                        }
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
            });
    }
}

fn tile_label_and_color(value: u8) -> (&'static str, Color32) {
    match value {
        0 => ("Gray", Color32::from_rgb(124, 126, 130)),
        1 => ("Yellow", Color32::from_rgb(201, 180, 88)),
        _ => ("Green", Color32::from_rgb(106, 170, 100)),
    }
}

fn decode_pattern(pattern: &u8) -> [u8; 5] {
    let mut value = *pattern;
    let mut decoded = [0u8; 5];
    for slot in &mut decoded {
        *slot = value % 3;
        value /= 3;
    }
    decoded
}
