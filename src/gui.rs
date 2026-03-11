use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::Duration,
};

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
    solver::{AbsurdleSuggestion, SolveState, Solver, Suggestion},
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
    Absurdle,
}

impl GuiSolverMode {
    fn label(self, formal_available: bool) -> &'static str {
        match self {
            Self::FormalOptimal if formal_available => "Formal Optimal",
            Self::FormalOptimal => "Formal Unavailable",
            Self::Predictive => "Wordle",
            Self::Absurdle => "Absurdle",
        }
    }
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
    absurdle_suggestions: Vec<AbsurdleSuggestion>,
    formal_suggestions: Vec<FormalSuggestion>,
    surviving_count: usize,
    total_weight: f64,
    top: usize,
    force_in_two_only: bool,
    hard_mode: bool,
    status: String,
    formal_explanation: Option<FormalStateExplanation>,
    request_sender: Sender<WorkerRequest>,
    response_receiver: Receiver<WorkerResponse>,
    latest_generation: u64,
    computing: bool,
}

#[derive(Clone, Debug)]
struct WorkerRequest {
    generation: u64,
    mode: GuiSolverMode,
    date_text: String,
    observations: Vec<(String, u8)>,
    top: usize,
    force_in_two_only: bool,
    hard_mode: bool,
}

#[derive(Clone, Debug)]
enum WorkerPayload {
    Predictive {
        state: SolveState,
        suggestions: Vec<Suggestion>,
    },
    Absurdle {
        state: SolveState,
        suggestions: Vec<AbsurdleSuggestion>,
    },
    Formal {
        explanation: FormalStateExplanation,
        suggestions: Vec<FormalSuggestion>,
    },
}

#[derive(Clone, Debug)]
struct WorkerResponse {
    generation: u64,
    payload: std::result::Result<WorkerPayload, String>,
}

impl WordleGuiApp {
    fn new(predictive_solver: Solver, formal_solver: Option<FormalPolicyRuntime>) -> Self {
        let date_text = Solver::today().format("%Y-%m-%d").to_string();
        let mode = if formal_solver.is_some() {
            GuiSolverMode::FormalOptimal
        } else {
            GuiSolverMode::Predictive
        };
        let (request_sender, response_receiver) =
            spawn_worker(predictive_solver.clone(), formal_solver.clone());
        let mut app = Self {
            predictive_solver,
            formal_solver,
            mode,
            date_text,
            current_guess: String::new(),
            current_feedback: [0; 5],
            observations: Vec::new(),
            predictive_suggestions: Vec::new(),
            absurdle_suggestions: Vec::new(),
            formal_suggestions: Vec::new(),
            surviving_count: 0,
            total_weight: 0.0,
            top: 10,
            force_in_two_only: false,
            hard_mode: false,
            status: String::new(),
            formal_explanation: None,
            request_sender,
            response_receiver,
            latest_generation: 0,
            computing: false,
        };
        app.schedule_recompute();
        app
    }

    fn schedule_recompute(&mut self) {
        self.latest_generation = self.latest_generation.wrapping_add(1);
        let request = WorkerRequest {
            generation: self.latest_generation,
            mode: self.mode,
            date_text: self.date_text.clone(),
            observations: self.observations.clone(),
            top: self.top,
            force_in_two_only: self.force_in_two_only,
            hard_mode: self.hard_mode,
        };
        self.computing = true;
        self.status = "Computing...".to_string();
        if let Err(error) = self.request_sender.send(request) {
            self.set_error(anyhow::anyhow!(error.to_string()));
        }
    }

    fn drain_worker_responses(&mut self) {
        while let Ok(response) = self.response_receiver.try_recv() {
            if !worker_response_is_current(self.latest_generation, response.generation) {
                continue;
            }
            self.computing = false;
            match response.payload {
                Ok(WorkerPayload::Predictive { state, suggestions }) => {
                    self.surviving_count = state.surviving.len();
                    self.total_weight = state.total_weight;
                    self.predictive_suggestions = suggestions;
                    self.absurdle_suggestions.clear();
                    self.formal_suggestions.clear();
                    self.formal_explanation = None;
                    self.status.clear();
                }
                Ok(WorkerPayload::Absurdle { state, suggestions }) => {
                    self.surviving_count = state.surviving.len();
                    self.total_weight = 0.0;
                    self.absurdle_suggestions = suggestions;
                    self.predictive_suggestions.clear();
                    self.formal_suggestions.clear();
                    self.formal_explanation = None;
                    self.status.clear();
                }
                Ok(WorkerPayload::Formal {
                    explanation,
                    suggestions,
                }) => {
                    self.surviving_count = explanation.surviving_answers;
                    self.total_weight = 0.0;
                    self.formal_explanation = Some(explanation);
                    self.formal_suggestions = suggestions;
                    self.predictive_suggestions.clear();
                    self.absurdle_suggestions.clear();
                    self.status.clear();
                }
                Err(error) => self.set_error(anyhow::anyhow!(error)),
            }
        }
    }

    fn set_error(&mut self, error: anyhow::Error) {
        self.status = error.to_string();
        self.predictive_suggestions.clear();
        self.absurdle_suggestions.clear();
        self.formal_suggestions.clear();
        self.formal_explanation = None;
        self.surviving_count = 0;
        self.total_weight = 0.0;
        self.computing = false;
    }

    fn commit_current_row(&mut self) {
        let guess = self.current_guess.trim().to_ascii_lowercase();
        if guess.is_empty() {
            self.schedule_recompute();
            return;
        }
        if self.mode == GuiSolverMode::Predictive {
            if let Some(error) = self
                .predictive_solver
                .hard_mode_violation(&self.observations, &guess)
                .filter(|_| self.hard_mode)
            {
                self.status = error;
                return;
            }
        }
        match self.row_pattern() {
            Ok(pattern) => {
                self.observations.push((guess, pattern));
                self.current_guess.clear();
                self.current_feedback = [0; 5];
                self.schedule_recompute();
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
        self.drain_worker_responses();
        if self.computing {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
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
                    let previous_mode = self.mode;
                    ui.label("Mode");
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
                        ui.colored_label(Color32::from_rgb(150, 45, 45), "build-optimal-policy first");
                    }
                    if self.mode != previous_mode {
                        self.schedule_recompute();
                    }
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Date");
                    let mut date_changed = false;
                    ui.add_enabled_ui(self.mode == GuiSolverMode::Predictive, |ui| {
                        date_changed = ui
                            .add_sized([120.0, 24.0], egui::TextEdit::singleline(&mut self.date_text))
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
                    if ui.button("Suggest").clicked() {
                        self.commit_current_row();
                    }
                    if ui.button("Undo").clicked() {
                        self.observations.pop();
                        self.schedule_recompute();
                    }
                    if ui.button("Reset").clicked() {
                        self.observations.clear();
                        self.current_guess.clear();
                        self.current_feedback = [0; 5];
                        self.schedule_recompute();
                    }
                    if self.mode == GuiSolverMode::Predictive {
                        let hard_changed =
                            ui.checkbox(&mut self.hard_mode, "Hard Mode").changed();
                        if hard_changed {
                            self.schedule_recompute();
                        }
                        let force_changed =
                            ui.checkbox(&mut self.force_in_two_only, "Force In 2 Only").changed();
                        if force_changed {
                            self.schedule_recompute();
                        }
                    }
                    if date_changed || top_changed {
                        self.schedule_recompute();
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

                if !self.status.is_empty() {
                    ui.add_space(8.0);
                    let color = if self.computing {
                        Color32::from_rgb(92, 72, 54)
                    } else {
                        Color32::from_rgb(150, 45, 45)
                    };
                    ui.colored_label(color, &self.status);
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
                        let (heading, summary) = match self.mode {
                            GuiSolverMode::Predictive => (
                                "Wordle Suggestions",
                                "Ranks guesses by predictive expected progress.",
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
                        ui.label(
                            RichText::new(summary).color(Color32::from_rgb(92, 72, 54)),
                        );
                        ui.add_space(8.0);
                        match self.mode {
                            GuiSolverMode::Predictive => {
                                let mut shown_any = false;
                                for suggestion in &self.predictive_suggestions {
                                    shown_any = true;
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
                                        if suggestion.force_in_two {
                                            ui.label("force_in_two");
                                        }
                                        if let Some(exact_cost) = suggestion.exact_cost {
                                            ui.label(format!("exact {:.4}", exact_cost));
                                        } else if let Some(lookahead_cost) = suggestion.lookahead_cost {
                                            ui.label(format!("lookahead {:.4}", lookahead_cost));
                                        }
                                    });
                                    ui.separator();
                                }
                                if !shown_any {
                                    ui.label(
                                        RichText::new(
                                            "No force_in_two suggestions found for this state.",
                                        )
                                        .color(Color32::from_rgb(92, 72, 54)),
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
                                        ui.label(format!(
                                            "multi {}",
                                            suggestion.multi_answer_bucket_count
                                        ));
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
            });
    }
}

fn spawn_worker(
    predictive_solver: Solver,
    formal_solver: Option<FormalPolicyRuntime>,
) -> (Sender<WorkerRequest>, Receiver<WorkerResponse>) {
    let (request_sender, request_receiver) = mpsc::channel::<WorkerRequest>();
    let (response_sender, response_receiver) = mpsc::channel::<WorkerResponse>();
    thread::spawn(move || {
        while let Ok(mut request) = request_receiver.recv() {
            while let Ok(newer_request) = request_receiver.try_recv() {
                request = newer_request;
            }
            let payload = match request.mode {
                GuiSolverMode::Predictive => {
                    let result = (|| -> Result<WorkerPayload> {
                        let date = NaiveDate::parse_from_str(&request.date_text, "%Y-%m-%d")
                            .with_context(|| format!("invalid date: {}", request.date_text))?;
                        let state = predictive_solver.apply_history(date, &request.observations)?;
                        let suggestions = if request.force_in_two_only {
                            predictive_solver.suggestions_for_history_disk_books_only_with_filters(
                                date,
                                &request.observations,
                                request.top,
                                request.hard_mode,
                                true,
                            )?
                        } else {
                            predictive_solver.suggestions_for_history_disk_books_only_with_filters(
                                date,
                                &request.observations,
                                request.top,
                                request.hard_mode,
                                false,
                            )?
                        };
                        Ok(WorkerPayload::Predictive { state, suggestions })
                    })();
                    result.map_err(|error| error.to_string())
                }
                GuiSolverMode::Absurdle => {
                    let result = (|| -> Result<WorkerPayload> {
                        let state =
                            predictive_solver.absurdle_apply_history(&request.observations)?;
                        let suggestions = predictive_solver
                            .absurdle_suggestions(&request.observations, request.top)?;
                        Ok(WorkerPayload::Absurdle { state, suggestions })
                    })();
                    result.map_err(|error| error.to_string())
                }
                GuiSolverMode::FormalOptimal => {
                    let result = (|| -> Result<WorkerPayload> {
                        let runtime = formal_solver.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("formal-optimal artifacts are not available")
                        })?;
                        let state = runtime.apply_history(&request.observations)?;
                        let explanation = runtime.explain_state(&state, request.top)?;
                        let suggestions = runtime.suggest(&state, request.top)?;
                        Ok(WorkerPayload::Formal {
                            explanation,
                            suggestions,
                        })
                    })();
                    result.map_err(|error| error.to_string())
                }
            };
            if response_sender
                .send(WorkerResponse {
                    generation: request.generation,
                    payload,
                })
                .is_err()
            {
                break;
            }
        }
    });
    (request_sender, response_receiver)
}

fn worker_response_is_current(latest_generation: u64, response_generation: u64) -> bool {
    latest_generation == response_generation
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

#[cfg(test)]
mod tests {
    use super::worker_response_is_current;

    #[test]
    fn worker_response_discards_stale_generations() {
        assert!(worker_response_is_current(7, 7));
        assert!(!worker_response_is_current(7, 6));
        assert!(!worker_response_is_current(7, 8));
    }
}
