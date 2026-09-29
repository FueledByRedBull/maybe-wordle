mod diagnostics;
mod formal;
mod play;
mod policy;
#[cfg(test)]
use play::{candidate_csv, matching_candidates, show_candidate_browser, show_game_board};

mod runtime;
use crate::predictive::types::{SearchActionScope, SearchExecution, SuggestionValueKind};
use runtime::*;

use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use chrono::NaiveDate;
use eframe::egui::{self, Color32, RichText};

use crate::{
    SOLVER_THREAD_STACK_BYTES,
    config::PriorConfig,
    data::{ProjectPaths, sync_nyt_history_cancellable},
    experiments::predictive_parameter_registry,
    formal::{
        DEFAULT_FORMAL_ALTERNATIVE_PARTITIONS, DEFAULT_FORMAL_MODEL_ID, FormalAlternativesStatus,
        FormalPolicyRuntime, FormalStateExplanation, FormalSuggestion, artifacts_exist,
    },
    game::{self, GameRules, GameStatus},
    model::build_model_artifacts,
    predictive::{
        PredictiveArtifactState, PredictiveCandidateSummary, PredictiveStateSummary,
        PredictiveSuggestRequest, PredictiveSuggestionMode, RecoveryMode,
    },
    scoring::{decode_feedback, parse_feedback},
    solver::{
        AbsurdleSuggestion, FiniteSearchCandidate, FiniteSearchOptions, FiniteSearchQuality,
        FiniteSearchReason, FiniteSearchResult, SolveState, Solver, Suggestion,
    },
};

pub fn run_gui(root: PathBuf) -> Result<()> {
    let paths = ProjectPaths::new(root);

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 820.0])
            .with_min_inner_size([560.0, 620.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Maybe Wordle",
        native_options,
        Box::new(move |_cc| Ok(Box::new(MaybeWordleApp::load_or_setup(paths)))),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuiSolverMode {
    FormalOptimal,
    Predictive,
    Absurdle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum GuiStatus {
    Loading(String),
    Ready,
    Notice(String),
    InputError(String),
    WorkerError(String),
}

impl GuiStatus {
    fn color(&self) -> Color32 {
        if self.is_error() {
            Color32::from_rgb(150, 45, 45)
        } else {
            Color32::from_rgb(92, 72, 54)
        }
    }
    fn message(&self) -> &str {
        match self {
            Self::Ready => "",
            Self::Loading(message)
            | Self::Notice(message)
            | Self::InputError(message)
            | Self::WorkerError(message) => message,
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, Self::InputError(_) | Self::WorkerError(_))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum WorkspaceView {
    #[default]
    Play,
    Policy,
    Diagnostics,
    Formal,
}

impl WorkspaceView {
    fn label(self) -> &'static str {
        match self {
            Self::Play => "Play",
            Self::Policy => "Policy",
            Self::Diagnostics => "Diagnostics",
            Self::Formal => "Formal lab",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SuggestionSort {
    Rank,
    SolveProbability,
    Entropy,
    ExpectedRemaining,
    WorstBucket,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum GuiSearchProfile {
    #[default]
    Configured,
    FiniteFast,
    FiniteStrong,
}

impl GuiSearchProfile {
    fn label(self) -> &'static str {
        match self {
            Self::Configured => "Configured",
            Self::FiniteFast => "Finite fast (experimental)",
            Self::FiniteStrong => "Finite strong (experimental)",
        }
    }

    fn finite_options(self, solver: &Solver) -> Option<FiniteSearchOptions> {
        match self {
            Self::Configured if solver.config.search_policy_mode.is_finite() => {
                Some(solver.finite_search_options())
            }
            Self::Configured => None,
            Self::FiniteFast => Some(FiniteSearchOptions::fast()),
            Self::FiniteStrong => Some(FiniteSearchOptions::strong()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BoardDraft {
    guess: String,
    feedback: [u8; 5],
}

enum BoardAction {
    ReplaceGuess(String),
    CycleTile(usize),
    Reset,
}

fn reduce_board_draft(current: &BoardDraft, action: BoardAction) -> BoardDraft {
    let mut next = current.clone();
    match action {
        BoardAction::ReplaceGuess(guess) => {
            next.guess = guess
                .chars()
                .filter(|character| character.is_ascii_alphabetic())
                .take(5)
                .collect::<String>()
                .to_ascii_lowercase();
        }
        BoardAction::CycleTile(index) if index < next.feedback.len() => {
            next.feedback[index] = (next.feedback[index] + 1) % 3;
        }
        BoardAction::CycleTile(_) => {}
        BoardAction::Reset => {
            next.guess.clear();
            next.feedback = [0; 5];
        }
    }
    next
}

fn feedback_code(feedback: [u8; 5]) -> String {
    feedback
        .into_iter()
        .map(|value| char::from(b'0' + value))
        .collect()
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
    config: PriorConfig,
    predictive_solver: Solver,
    formal_solver: Option<FormalPolicyRuntime>,
    formal_availability: FormalAvailability,
    formal_receiver: Option<Receiver<std::result::Result<Option<FormalPolicyRuntime>, String>>>,
    formal_cancel: Arc<AtomicBool>,
    workspace_view: WorkspaceView,
    mode: GuiSolverMode,
    search_profile: GuiSearchProfile,
    text_scale: f32,
    date_text: String,
    current_guess: String,
    feedback_code: String,
    current_feedback: [u8; 5],
    observations: Vec<(String, u8)>,
    predictive_suggestions: Vec<Suggestion>,
    predictive_candidates: Vec<PredictiveCandidateSummary>,
    candidate_filter: String,
    absurdle_suggestions: Vec<AbsurdleSuggestion>,
    formal_suggestions: Vec<FormalSuggestion>,
    surviving_count: usize,
    total_weight: f64,
    predictive_recovery_mode: Option<RecoveryMode>,
    predictive_artifact_state: PredictiveArtifactState,
    predictive_model_metadata: String,
    predictive_finite_search: Option<FiniteSearchResult>,
    predictive_execution: Option<SearchExecution>,
    history_coverage: HistoryCoverage,
    top: usize,
    force_in_two_only: bool,
    hard_mode: bool,
    status: GuiStatus,
    formal_explanation: Option<FormalStateExplanation>,
    suggestion_sort: SuggestionSort,
    suggestion_sort_descending: bool,
    selected_suggestion: Option<String>,
    request_sender: Option<LatestWorkerDispatcher>,
    response_receiver: Receiver<WorkerResponse>,
    latest_generation: u64,
    current_request: Option<WorkerRequest>,
    computing: bool,
}

impl WordleGuiApp {
    fn board_draft(&self) -> BoardDraft {
        BoardDraft {
            guess: self.current_guess.clone(),
            feedback: self.current_feedback,
        }
    }

    fn apply_board_action(&mut self, action: BoardAction) {
        let changes_feedback = !matches!(action, BoardAction::ReplaceGuess(_));
        let next = reduce_board_draft(&self.board_draft(), action);
        self.current_guess = next.guess;
        self.current_feedback = next.feedback;
        if changes_feedback {
            self.feedback_code = feedback_code(self.current_feedback);
        }
    }

    fn update_feedback_code(&mut self, code: String) {
        self.feedback_code = code;
        if let Ok(pattern) = parse_feedback(&self.feedback_code) {
            self.current_feedback = decode_feedback(pattern);
        }
    }

    fn new(workspace: LoadedWorkspace) -> Self {
        let LoadedWorkspace {
            config,
            predictive_solver,
            formal_solver,
        } = workspace;
        let date_text = Solver::today().format("%Y-%m-%d").to_string();
        let mode = GuiSolverMode::Predictive;
        let worker = spawn_worker(predictive_solver.clone(), formal_solver.clone());
        let mut app = Self {
            config,
            predictive_solver,
            formal_availability: if formal_solver.is_some() {
                FormalAvailability::Ready
            } else {
                FormalAvailability::Absent
            },
            formal_solver,
            formal_receiver: None,
            formal_cancel: Arc::new(AtomicBool::new(false)),
            workspace_view: WorkspaceView::Play,
            mode,
            search_profile: GuiSearchProfile::Configured,
            text_scale: 1.0,
            date_text,
            current_guess: String::new(),
            feedback_code: "00000".to_string(),
            current_feedback: [0; 5],
            observations: Vec::new(),
            predictive_suggestions: Vec::new(),
            predictive_candidates: Vec::new(),
            candidate_filter: String::new(),
            absurdle_suggestions: Vec::new(),
            formal_suggestions: Vec::new(),
            surviving_count: 0,
            total_weight: 0.0,
            predictive_recovery_mode: None,
            predictive_artifact_state: PredictiveArtifactState::NoPredictiveArtifactAvailable,
            predictive_model_metadata: String::new(),
            predictive_finite_search: None,
            predictive_execution: None,
            history_coverage: HistoryCoverage::NotRequested,
            top: 10,
            force_in_two_only: false,
            hard_mode: false,
            status: GuiStatus::Ready,
            formal_explanation: None,
            suggestion_sort: SuggestionSort::Rank,
            suggestion_sort_descending: false,
            selected_suggestion: None,
            request_sender: None,
            response_receiver: mpsc::channel().1,
            latest_generation: 0,
            current_request: None,
            computing: false,
        };
        app.install_worker(worker);
        app
    }

    fn clear_result(&mut self) {
        self.predictive_suggestions.clear();
        self.predictive_candidates.clear();
        self.absurdle_suggestions.clear();
        self.formal_suggestions.clear();
        self.formal_explanation = None;
        self.selected_suggestion = None;
        self.surviving_count = 0;
        self.total_weight = 0.0;
        self.predictive_recovery_mode = None;
        self.predictive_artifact_state = PredictiveArtifactState::NoPredictiveArtifactAvailable;
        self.predictive_finite_search = None;
        self.predictive_execution = None;
        self.predictive_model_metadata.clear();
    }

    fn commit_current_row(&mut self) {
        let proposed = game::try_append_observation(
            &self.observations,
            &self.current_guess,
            &self.feedback_code,
            self.rules(),
            |history| match self.mode {
                GuiSolverMode::Predictive => {
                    let date = NaiveDate::parse_from_str(&self.date_text, "%Y-%m-%d")
                        .context("invalid puzzle date")?;
                    self.predictive_solver
                        .validate_game_history(date, history, self.hard_mode)?;
                    Ok(())
                }
                GuiSolverMode::Absurdle => {
                    self.predictive_solver.absurdle_apply_history(history)?;
                    Ok(())
                }
                GuiSolverMode::FormalOptimal => {
                    self.formal_solver
                        .as_ref()
                        .context("Formal model is unavailable")?
                        .apply_history(history)?;
                    Ok(())
                }
            },
        );
        match proposed {
            Ok(history) => {
                self.observations = history;
                self.apply_board_action(BoardAction::Reset);
                self.schedule_recompute();
            }
            Err(error) => self.status = GuiStatus::InputError(error.to_string()),
        }
    }

    fn rules(&self) -> GameRules {
        if self.mode == GuiSolverMode::Absurdle {
            GameRules::Absurdle
        } else {
            GameRules::Wordle
        }
    }

    fn can_apply_row(&self) -> bool {
        matches!(
            game::status(&self.observations, self.rules()),
            Ok(GameStatus::Active)
        ) && self.row_pattern().is_ok()
    }

    fn toggle_suggestion_sort(&mut self, sort: SuggestionSort) {
        if self.suggestion_sort == sort {
            self.suggestion_sort_descending = !self.suggestion_sort_descending;
        } else {
            self.suggestion_sort = sort;
            self.suggestion_sort_descending = matches!(
                sort,
                SuggestionSort::SolveProbability | SuggestionSort::Entropy
            );
        }
    }

    fn sort_label(&self, label: &str, sort: SuggestionSort) -> String {
        if self.suggestion_sort == sort {
            format!(
                "{label} {}",
                if self.suggestion_sort_descending {
                    "↓"
                } else {
                    "↑"
                }
            )
        } else {
            label.to_string()
        }
    }

    fn row_pattern(&self) -> Result<u8> {
        if self.current_guess.trim().len() != 5 {
            anyhow::bail!("current guess must be exactly 5 letters");
        }
        parse_feedback(&self.feedback_code)
    }
}

impl eframe::App for WordleGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_zoom_factor(self.text_scale);
        ctx.set_visuals(workspace_visuals());
        self.drain_worker_responses();
        self.drain_formal_load();
        if !ctx.wants_keyboard_input()
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::CTRL, egui::Key::Z))
        {
            self.observations.pop();
            self.schedule_recompute();
        }
        if !ctx.wants_keyboard_input()
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            self.apply_board_action(BoardAction::Reset);
        }
        if self.computing || self.formal_receiver.is_some() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(Color32::from_rgb(246, 240, 232))
                    .inner_margin(24.0),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("workspace-scroll")
                    .show(ui, |ui| {
                ui.visuals_mut().widgets.inactive.bg_fill = Color32::from_rgb(239, 228, 211);
                ui.visuals_mut().widgets.hovered.bg_fill = Color32::from_rgb(226, 214, 195);
                ui.visuals_mut().widgets.active.bg_fill = Color32::from_rgb(212, 198, 177);

                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new("MAYBE / WORDLE")
                            .monospace()
                            .strong()
                            .color(Color32::from_rgb(171, 73, 43)),
                    );
                    ui.separator();
                    let previous_view = self.workspace_view;
                    for view in [
                        WorkspaceView::Play,
                        WorkspaceView::Policy,
                        WorkspaceView::Diagnostics,
                        WorkspaceView::Formal,
                    ] {
                        ui.selectable_value(&mut self.workspace_view, view, view.label());
                    }
                    if self.workspace_view != previous_view {
                        let next_mode = if self.workspace_view == WorkspaceView::Formal
                            && self.formal_solver.is_some()
                        {
                            GuiSolverMode::FormalOptimal
                        } else if previous_view == WorkspaceView::Formal {
                            GuiSolverMode::Predictive
                        } else {
                            self.mode
                        };
                        if next_mode != self.mode {
                            self.mode = next_mode;
                            self.schedule_recompute();
                        }
                    }
                    ui.separator();
                    ui.label(RichText::new("Text").small());
                    ui.add(
                        egui::Slider::new(&mut self.text_scale, 0.85..=1.35)
                            .show_value(false)
                            .custom_formatter(|value, _| format!("{:.0}%", value * 100.0)),
                    );
                });
                ui.add_space(12.0);

                ui.heading(
                    RichText::new(match self.workspace_view {
                        WorkspaceView::Play => "Predictive play desk",
                        WorkspaceView::Policy => "Policy ledger",
                        WorkspaceView::Diagnostics => "Diagnostics bench",
                        WorkspaceView::Formal => "Formal proof lab",
                    })
                        .size(30.0)
                        .color(Color32::from_rgb(58, 44, 32)),
                );
                ui.label(
                    RichText::new(match self.workspace_view {
                        WorkspaceView::Play => {
                            "Enter a guess, encode its feedback, then inspect the predictive alternatives."
                        }
                        WorkspaceView::Policy => {
                            "The active search allocation and predictive artifact contract, separated from play."
                        }
                        WorkspaceView::Diagnostics => {
                            "Runtime state, provenance, recovery, and candidate-pool diagnostics."
                        }
                        WorkspaceView::Formal => {
                            "Secondary exact-policy tooling; predictive play remains the primary product."
                        }
                    })
                    .color(Color32::from_rgb(92, 72, 54)),
                );
                ui.add_space(12.0);

                if self.mode == GuiSolverMode::Predictive
                    && let Some(notice) = self.history_coverage.notice()
                {
                    ui.label(RichText::new(notice).color(Color32::from_rgb(92, 72, 54)));
                    ui.add_space(8.0);
                }

                match self.workspace_view {
                    WorkspaceView::Policy => {
                        self.show_policy_panel(ui);
                        return;
                    }
                    WorkspaceView::Diagnostics => {
                        self.show_diagnostics_panel(ui);
                        return;
                    }
                    WorkspaceView::Formal => {
                        self.show_formal_panel(ui);
                        return;
                    }
                    WorkspaceView::Play => {}
                }

                self.show_play(ui, ctx);
                    });
            });
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuiSurfaceState {
    MissingData,
    Loading,
    Empty,
    Normal,
    Recovery,
    Error,
}

impl GuiSurfaceState {
    fn label(self) -> &'static str {
        match self {
            Self::MissingData => "missing data",
            Self::Loading => "loading",
            Self::Empty => "empty",
            Self::Normal => "normal",
            Self::Recovery => "recovery",
            Self::Error => "error",
        }
    }
}

fn gui_surface_state(
    missing_data: bool,
    computing: bool,
    status: &GuiStatus,
    recovery: Option<RecoveryMode>,
    suggestion_count: usize,
) -> GuiSurfaceState {
    if missing_data {
        GuiSurfaceState::MissingData
    } else if computing {
        GuiSurfaceState::Loading
    } else if status.is_error() {
        GuiSurfaceState::Error
    } else if recovery.is_some() {
        GuiSurfaceState::Recovery
    } else if suggestion_count == 0 {
        GuiSurfaceState::Empty
    } else {
        GuiSurfaceState::Normal
    }
}

fn is_compact_layout(available_width: f32) -> bool {
    available_width < 860.0
}

fn policy_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.label(
            RichText::new(format!("{label}:"))
                .monospace()
                .color(Color32::from_rgb(92, 72, 54)),
        );
        ui.label(value);
    });
}

fn predictive_registry_summary(config: &PriorConfig) -> String {
    let registry = predictive_parameter_registry(config);
    let tunable = registry
        .parameters
        .iter()
        .filter(|parameter| parameter.tunable())
        .count();
    format!(
        "v{} · {} leaves · {} tunable",
        registry.format_version,
        registry.parameters.len(),
        tunable
    )
}

fn diagnostic_badge(ui: &mut egui::Ui, label: &str, value: &str) {
    egui::Frame::default()
        .fill(Color32::from_rgb(239, 228, 211))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.label(
                RichText::new(label)
                    .monospace()
                    .small()
                    .color(Color32::from_rgb(171, 73, 43)),
            );
            ui.label(RichText::new(value).strong());
        });
}

fn workspace_visuals() -> egui::Visuals {
    let ink = Color32::from_rgb(42, 49, 43);
    let paper = Color32::from_rgb(246, 240, 232);
    let mut visuals = egui::Visuals::light();
    visuals.panel_fill = paper;
    visuals.window_fill = paper;
    visuals.faint_bg_color = Color32::from_rgb(238, 231, 221);
    visuals.extreme_bg_color = Color32::from_rgb(255, 252, 247);
    visuals.selection.bg_fill = Color32::from_rgb(46, 112, 82);
    visuals.selection.stroke = egui::Stroke::new(1.0_f32, Color32::WHITE);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0_f32, ink);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0_f32, ink);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5_f32, ink);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.5_f32, ink);
    visuals
}

fn sorted_predictive_indices(
    suggestions: &[Suggestion],
    sort: SuggestionSort,
    descending: bool,
) -> Vec<usize> {
    let mut indices = (0..suggestions.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let left_suggestion = &suggestions[*left];
        let right_suggestion = &suggestions[*right];
        let ordering = match sort {
            SuggestionSort::Rank => left.cmp(right),
            SuggestionSort::SolveProbability => left_suggestion
                .solve_probability
                .total_cmp(&right_suggestion.solve_probability),
            SuggestionSort::Entropy => left_suggestion.entropy.total_cmp(&right_suggestion.entropy),
            SuggestionSort::ExpectedRemaining => left_suggestion
                .expected_remaining
                .total_cmp(&right_suggestion.expected_remaining),
            SuggestionSort::WorstBucket => left_suggestion
                .worst_non_green_bucket_size
                .cmp(&right_suggestion.worst_non_green_bucket_size),
        }
        .then_with(|| left.cmp(right));
        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });
    indices
}

fn suggestion_method(suggestion: &Suggestion) -> &'static str {
    suggestion.value_kind.label()
}

fn finite_quality_label(quality: FiniteSearchQuality) -> &'static str {
    match quality {
        FiniteSearchQuality::Heuristic => "pending heuristic",
        FiniteSearchQuality::UpperBound => "completed rollout",
        FiniteSearchQuality::Exact => "model-exact action value",
    }
}

fn format_finite_failure_risk(candidate: &FiniteSearchCandidate) -> String {
    if candidate.quality == FiniteSearchQuality::Heuristic {
        "pending".to_string()
    } else {
        format!("{:.3}%", candidate.failure_probability * 100.0)
    }
}

fn format_finite_expected_attempts(candidate: &FiniteSearchCandidate) -> String {
    if candidate.quality == FiniteSearchQuality::Heuristic {
        "unevaluated".to_string()
    } else {
        format!("{:.3}", candidate.expected_attempts)
    }
}

fn predictive_finite_banner_text(profile: GuiSearchProfile) -> &'static str {
    match profile {
        GuiSearchProfile::Configured => "Using configured bounded finite search",
        GuiSearchProfile::FiniteFast => "Using finite fast search (experimental)",
        GuiSearchProfile::FiniteStrong => "Using finite strong search (experimental)",
    }
}

fn finite_reason_label(reason: FiniteSearchReason) -> &'static str {
    match reason {
        FiniteSearchReason::Complete => "complete",
        FiniteSearchReason::Deadline => "deadline",
        FiniteSearchReason::Cancelled => "cancelled",
        FiniteSearchReason::NodeBudget => "node budget",
    }
}

fn format_finite_search_summary(search: &FiniteSearchResult) -> String {
    format!(
        "Finite search: {} · {} candidates · {} nodes · {} work units · {} cache hits{}",
        finite_reason_label(search.reason),
        search.candidates.len(),
        search.nodes_visited,
        search.work_units,
        search.cache_hits,
        if search.proposal_sampled {
            " · sampled proposals"
        } else {
            ""
        },
    )
}

fn predictive_finite_compute_status(
    profile: GuiSearchProfile,
    options: FiniteSearchOptions,
) -> String {
    format!(
        "Computing... {} ({} ms budget)",
        predictive_finite_banner_text(profile),
        options.budget.as_millis(),
    )
}

fn predictive_response_status(finite_search: Option<&FiniteSearchResult>) -> GuiStatus {
    match finite_search.map(|search| search.reason) {
        None | Some(FiniteSearchReason::Complete) => GuiStatus::Ready,
        Some(reason) => GuiStatus::Notice(format!(
            "Bounded search stopped: {}. See evaluated quality per action.",
            finite_reason_label(reason)
        )),
    }
}

fn format_suggestion_cost(suggestion: &Suggestion) -> String {
    suggestion
        .exact_cost
        .or(suggestion.lookahead_cost)
        .or(suggestion.proxy_cost)
        .map(|cost| format!("{cost:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

fn feedback_accessible_label(value: u8) -> &'static str {
    match value {
        0 => "Absent",
        1 => "Present",
        _ => "Correct",
    }
}

fn feedback_marker(value: u8) -> &'static str {
    match value {
        0 => "A",
        1 => "P",
        _ => "C",
    }
}

fn tile_label_and_color(value: u8) -> (&'static str, Color32, Color32) {
    match value {
        0 => ("Gray", Color32::from_rgb(124, 126, 130), Color32::BLACK),
        1 => ("Yellow", Color32::from_rgb(201, 180, 88), Color32::BLACK),
        _ => ("Green", Color32::from_rgb(106, 170, 100), Color32::BLACK),
    }
}

fn predictive_banner_text(state: PredictiveArtifactState) -> &'static str {
    state.banner_text()
}

fn predictive_compute_status(state: PredictiveArtifactState) -> String {
    format!("Computing... {}", state.compute_text())
}

fn predictive_reply_book_text(
    observation_count: usize,
    state: PredictiveArtifactState,
) -> Option<&'static str> {
    match observation_count {
        1 | 2 if state == PredictiveArtifactState::ExactDateArtifact => {
            Some("Reply-book artifact is available for this branch.")
        }
        1 | 2 => Some(
            "Reply-book artifact is missing for this date or branch; showing live ranking only.",
        ),
        _ => None,
    }
}

fn formal_unavailable_text() -> &'static str {
    "Formal artifacts missing; run build-optimal-policy first."
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
        mpsc,
    };

    use chrono::NaiveDate;

    use crate::predictive::{
        PredictiveArtifactState, PredictiveSuggestRequest, PredictiveSuggestionMode,
    };

    use super::{
        BoardAction, BoardDraft, FiniteSearchCandidate, FiniteSearchQuality, FiniteSearchReason,
        FiniteSearchResult, GuiSearchProfile, GuiSolverMode, GuiStatus, GuiSurfaceState,
        LatestWorkerDispatcher, LatestWorkerQueue, Solver, Suggestion, WorkerPayload,
        WorkerRequest, feedback_accessible_label, feedback_marker, finite_quality_label,
        formal_unavailable_text, format_finite_expected_attempts, format_finite_failure_risk,
        gui_surface_state, is_compact_layout, predictive_banner_text, predictive_compute_status,
        predictive_finite_banner_text, predictive_registry_summary, predictive_reply_book_text,
        predictive_response_status, reduce_board_draft, spawn_worker, worker_request_cancelled,
    };

    fn worker_fixture_solver() -> (Solver, crate::test_support::TestDirectory) {
        let root = crate::test_support::TestDirectory::new("gui-worker");
        let paths = crate::data::ProjectPaths::new(root.path());
        paths.ensure_layout().expect("fixture layout");
        let words = [
            "cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade", "naval",
            "serve", "heath", "dwarf", "model", "karma", "stink", "grade", "quiet", "bench",
            "abate", "feign", "major", "death", "fresh", "crust", "stool", "colon", "abase",
            "marry", "react", "batty", "pride", "floss",
        ];
        let word_list = format!("{}\n", words.join("\n"));
        std::fs::write(&paths.seed_guesses, &word_list).expect("fixture guesses");
        std::fs::write(&paths.seed_answers, &word_list).expect("fixture answers");
        std::fs::write(&paths.manual_additions, "").expect("fixture manual additions");
        let solver = Solver::from_paths(&paths, &crate::config::PriorConfig::default())
            .expect("fixture solver");
        (solver, root)
    }

    fn predictive_worker_request(generation: u64, profile: GuiSearchProfile) -> WorkerRequest {
        WorkerRequest {
            generation,
            mode: GuiSolverMode::Predictive,
            date_text: "2026-07-26".to_string(),
            observations: vec![("cigar".to_string(), 0)],
            top: 3,
            force_in_two_only: false,
            hard_mode: false,
            search_profile: profile,
        }
    }

    #[test]
    fn suggestion_method_does_not_infer_exactness_from_a_numeric_cost() {
        let (mut solver, _root) = worker_fixture_solver();
        solver.config.search_policy_mode = crate::config::SearchPolicyMode::ProxyOnly;
        let state = solver.initial_state(NaiveDate::from_ymd_opt(2026, 7, 26).unwrap());
        let mut row = solver.suggestions(&state, 1).unwrap().remove(0);
        row.exact_cost = Some(2.0);
        assert!(super::suggestion_method(&row).contains("proxy"));
        for kind in [
            super::SuggestionValueKind::Lookahead,
            super::SuggestionValueKind::ContinuationEstimate,
            super::SuggestionValueKind::ExactAction,
            super::SuggestionValueKind::Finite(FiniteSearchQuality::Heuristic),
            super::SuggestionValueKind::Finite(FiniteSearchQuality::UpperBound),
            super::SuggestionValueKind::Finite(FiniteSearchQuality::Exact),
            super::SuggestionValueKind::Terminal,
        ] {
            row.value_kind = kind;
            assert_eq!(super::suggestion_method(&row), kind.label());
        }
    }

    #[test]
    fn every_feedback_pattern_uses_the_canonical_board_encoding() {
        for pattern in 0..crate::scoring::PATTERN_SPACE as u8 {
            let decoded = crate::scoring::decode_feedback(pattern);
            let expected = [1u8, 3, 9, 27, 81].map(|power| (pattern / power) % 3);
            assert_eq!(decoded, expected);
            let code = super::feedback_code(decoded);
            assert_eq!(crate::scoring::parse_feedback(&code).unwrap(), pattern);
            assert_eq!(code, crate::scoring::format_feedback_trits(pattern));
        }
    }

    #[test]
    fn configured_staged_gui_keeps_experimental_finite_search_opt_in() {
        let (solver, root) = worker_fixture_solver();
        assert!(
            GuiSearchProfile::Configured
                .finite_options(&solver)
                .is_none()
        );
        assert!(
            GuiSearchProfile::FiniteFast
                .finite_options(&solver)
                .is_some()
        );
        drop(root);
    }

    #[test]
    fn board_reducer_normalizes_input_cycles_feedback_and_resets() {
        let initial = BoardDraft {
            guess: String::new(),
            feedback: [0; 5],
        };
        let typed = reduce_board_draft(&initial, BoardAction::ReplaceGuess("Cr4aNe!".to_string()));
        assert_eq!(typed.guess, "crane");
        let present = reduce_board_draft(&typed, BoardAction::CycleTile(1));
        assert_eq!(present.feedback, [0, 1, 0, 0, 0]);
        let correct = reduce_board_draft(&present, BoardAction::CycleTile(1));
        assert_eq!(correct.feedback, [0, 2, 0, 0, 0]);
        let coded = BoardDraft {
            feedback: crate::scoring::decode_feedback(
                crate::scoring::parse_feedback("byg20").expect("mixed code"),
            ),
            ..correct
        };
        assert_eq!(coded.feedback, [0, 1, 2, 2, 0]);
        assert_eq!(feedback_accessible_label(2), "Correct");
        assert_eq!(
            [feedback_marker(0), feedback_marker(1), feedback_marker(2)],
            ["A", "P", "C"]
        );
        assert_eq!(reduce_board_draft(&coded, BoardAction::Reset), initial);
    }

    #[test]
    fn feedback_text_is_authoritative_and_invalid_edits_preserve_the_draft() {
        let (solver, root) = worker_fixture_solver();
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        app.current_guess = "cigar".to_string();
        for invalid in ["0", "", "byg", "2222!2"] {
            app.update_feedback_code("22222".to_string());
            app.update_feedback_code(invalid.to_string());
            assert!(
                app.row_pattern().is_err(),
                "invalid visible draft {invalid:?}"
            );
            app.commit_current_row();
            assert!(app.observations.is_empty());
            assert_eq!(app.current_guess, "cigar");
            assert_eq!(app.feedback_code, invalid);
        }
        drop(app);
        drop(root);
    }

    #[test]
    fn history_submission_is_transactional_and_terminal_states_block_entry() {
        let (solver, root) = worker_fixture_solver();
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        app.date_text = "2026-07-26".to_string();
        for guess in ["zzzzz", "cig"] {
            app.current_guess = guess.to_string();
            app.update_feedback_code("00000".into());
            app.commit_current_row();
            assert!(app.observations.is_empty());
            assert_eq!(app.current_guess, guess);
            assert!(matches!(app.status, GuiStatus::InputError(_)));
        }
        app.current_guess = "cigar".into();
        app.update_feedback_code("22222".into());
        app.commit_current_row();
        assert_eq!(app.observations.len(), 1);
        assert!(!app.computing);
        assert!(!app.can_apply_row());
        app.current_guess = "rebut".into();
        app.commit_current_row();
        assert_eq!(app.observations.len(), 1);
        assert_eq!(app.current_guess, "rebut");
        app.observations.pop();
        app.schedule_recompute();
        assert!(app.can_apply_row());
        drop(app);
        drop(root);
    }

    #[test]
    fn dropped_response_sender_becomes_recoverable_worker_error() {
        let (solver, root) = worker_fixture_solver();
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        let (sender, receiver) = mpsc::channel();
        app.response_receiver = receiver;
        drop(sender);
        app.drain_worker_responses();
        assert!(!app.computing);
        assert!(matches!(app.status, GuiStatus::WorkerError(_)));
        assert!(app.request_sender.is_none());
        app.retry_worker();
        assert!(app.request_sender.is_some());
        assert!(app.computing);
        drop(app);
        drop(root);
    }

    #[test]
    fn slow_loading_first_frame_and_close_do_not_wait_for_a_worker_result() {
        let root = crate::test_support::TestDirectory::new("gui-loading");
        let mut setup =
            super::SetupApp::new(crate::data::ProjectPaths::new(root.path()), String::new());
        let (sender, receiver) = mpsc::channel();
        setup.running = true;
        setup.receiver = Some(receiver);
        let cancellation = Arc::clone(&setup.cancel_requested);
        let mut app = super::MaybeWordleApp::Loading(Box::new(setup));
        let context = eframe::egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let output = context.run(Default::default(), |ctx| {
            eframe::App::update(&mut app, ctx, &mut frame)
        });
        assert!(matches!(app, super::MaybeWordleApp::Loading(_)));
        assert!(!output.shapes.is_empty());
        drop(app);
        assert!(cancellation.load(AtomicOrdering::Acquire));
        drop(sender);
    }

    #[test]
    fn injected_spawn_failure_and_poisoned_queue_are_recoverable() {
        let (solver, root) = worker_fixture_solver();
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        app.install_worker(Err(anyhow::anyhow!("injected thread spawn failure")));
        assert!(app.request_sender.is_none());
        assert!(!app.computing);
        assert!(matches!(app.status, GuiStatus::WorkerError(_)));
        app.retry_worker();
        let shared = Arc::clone(&app.request_sender.as_ref().unwrap().shared);
        let poisoned = std::thread::spawn(move || {
            let _guard = shared.0.lock().unwrap();
            panic!("injected queue poison");
        });
        assert!(poisoned.join().is_err());
        app.schedule_recompute();
        assert!(app.request_sender.is_none());
        assert!(!app.computing);
        assert!(app.status.is_error());
        drop(app);
        drop(root);
    }

    #[test]
    fn virtual_candidate_browser_can_render_its_last_row() {
        use eframe::egui;
        for count in [1, 100, 101, 2315] {
            let candidates = (0..count)
                .map(|index| crate::predictive::PredictiveCandidateSummary {
                    word: format!("w{index:04}"),
                    probability: 0.01,
                    modeled_weight: 1.0,
                    fallback_support: false,
                })
                .collect::<Vec<_>>();
            let context = egui::Context::default();
            let mut filter = String::new();
            let mut output = None;
            let mut declared_content_height = 0.0;
            let mut row_spacing = 0.0;
            for step in 0..3 {
                output = Some(context.run(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(560.0, 620.0),
                        )),
                        ..Default::default()
                    },
                    |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            if step == 1 {
                                let id = ui.make_persistent_id(egui::Id::new(
                                    "predictive-candidate-scroll",
                                ));
                                let mut state = egui::scroll_area::State::load(ctx, id)
                                    .expect("candidate scroll state");
                                // Let the widget clamp to its actual bottom; the test
                                // must not duplicate the production row-height formula.
                                state.offset.y = 1_000_000.0;
                                state.store(ctx, id);
                            }
                            row_spacing = ui.spacing().item_spacing.y;
                            declared_content_height =
                                super::show_candidate_browser(ui, &mut filter, &candidates)
                                    .content_size
                                    .y;
                        });
                    },
                ));
            }
            let last = candidates.last().unwrap().word.to_ascii_uppercase();
            if count > 1 {
                let positions = output
                    .as_ref()
                    .unwrap()
                    .shapes
                    .iter()
                    .filter_map(|shape| match &shape.shape {
                        egui::Shape::Text(text) if text.galley.text().starts_with('W') => {
                            Some(text.pos.y)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let actual_row_pitch = positions[1] - positions[0];
                assert!(
                    (declared_content_height - (actual_row_pitch * count as f32 - row_spacing))
                        .abs()
                        < 0.1,
                    "virtual row geometry disagrees with rendered rows: content={declared_content_height}, pitch={actual_row_pitch}, count={count}"
                );
            }
            assert!(output.unwrap().shapes.iter().any(|shape| matches!(&shape.shape, egui::Shape::Text(text) if text.galley.text() == last)), "last candidate not rendered for {count}");
        }
    }

    #[test]
    fn candidate_browser_filter_and_csv_keep_every_row_in_order() {
        for count in [0, 1, 100, 101, 2315] {
            let candidates = (0..count)
                .map(|index| crate::predictive::PredictiveCandidateSummary {
                    word: format!("w{index:04}"),
                    probability: 0.01,
                    modeled_weight: 1.0,
                    fallback_support: false,
                })
                .collect::<Vec<_>>();
            let matching = super::matching_candidates(&candidates, "");
            assert_eq!(matching.len(), count);
            let csv = super::candidate_csv(&matching);
            let exported = csv
                .lines()
                .skip(1)
                .map(|row| row.split(',').next().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                exported,
                candidates
                    .iter()
                    .map(|candidate| candidate.word.as_str())
                    .collect::<Vec<_>>()
            );
            if let Some(last) = candidates.last() {
                let filtered =
                    super::matching_candidates(&candidates, &last.word.to_ascii_uppercase());
                assert_eq!(filtered.len(), 1);
                assert_eq!(filtered[0].word, last.word);
            }
        }
    }

    #[test]
    fn typed_status_severity_does_not_depend_on_wording_or_budget_reason() {
        for status in [
            GuiStatus::Ready,
            GuiStatus::Notice("error is only a word".into()),
            GuiStatus::Loading("loading".into()),
        ] {
            assert!(!status.is_error());
        }
        for status in [
            GuiStatus::InputError("".into()),
            GuiStatus::WorkerError("".into()),
        ] {
            assert!(status.is_error());
        }
    }

    #[test]
    fn corrupt_optional_formal_artifacts_do_not_block_predictive_workspace() {
        let (solver, root) = worker_fixture_solver();
        let paths = crate::data::ProjectPaths::new(root.path());
        let artifacts = crate::formal::PolicyArtifactSet::for_model(
            &paths,
            crate::formal::DEFAULT_FORMAL_MODEL_ID,
        );
        std::fs::create_dir_all(&artifacts.model_dir).expect("formal fixture");
        for path in [
            &artifacts.prior_spec,
            &artifacts.manifest,
            &artifacts.values,
            &artifacts.policy,
            &artifacts.metadata,
            &artifacts.certificate,
            &artifacts.small_state_table,
            &artifacts.pattern_table,
        ] {
            std::fs::write(path, "corrupt").expect("corrupt fixture artifact");
        }
        let workspace = super::load_workspace(&paths).expect("predictive remains independent");
        let mut app = super::WordleGuiApp::new(workspace);
        app.start_formal_load(paths);
        assert_eq!(app.formal_availability, super::FormalAvailability::Loading);
        let outcome = app
            .formal_receiver
            .as_ref()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("formal completion");
        assert!(outcome.is_err());
        let (sender, receiver) = mpsc::channel();
        sender.send(outcome).expect("inject completed outcome");
        app.formal_receiver = Some(receiver);
        app.drain_formal_load();
        assert!(matches!(
            app.formal_availability,
            super::FormalAvailability::Unavailable(_)
        ));
        assert!(app.request_sender.is_some());
        assert_eq!(app.mode, GuiSolverMode::Predictive);
        drop(app);
        drop(solver);
        drop(root);
    }

    #[test]
    fn successful_seed_only_predictions_keep_a_history_limitation_notice() {
        let (mut solver, root) = worker_fixture_solver();
        solver.config.search_policy_mode = crate::config::SearchPolicyMode::ProxyOnly;
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        let request = app.current_request.clone().expect("initial request");
        let payload =
            super::compute_worker_payload(&app.predictive_solver, None, &request, &|| false)
                .expect("seed-only predictive play remains usable");
        let (sender, receiver) = mpsc::channel();
        app.response_receiver = receiver;
        sender
            .send(super::WorkerResponse {
                request,
                payload: Ok(payload),
            })
            .unwrap();
        app.drain_worker_responses();
        assert!(
            matches!(app.status, GuiStatus::Notice(_)),
            "missing history must not report clean Ready: {:?}",
            app.status
        );
        assert!(app.status.message().contains("history"));
        assert!(!app.status.is_error());
        assert!(!app.predictive_suggestions.is_empty());
        let initial_notice = app.history_coverage.notice().expect("persistent notice");
        app.schedule_recompute();
        assert_eq!(
            app.history_coverage.notice().as_deref(),
            Some(initial_notice.as_str())
        );
        assert!(matches!(app.status, GuiStatus::Loading(_)));
        app.set_error(anyhow::anyhow!("synthetic worker failure"));
        assert_eq!(
            app.history_coverage.notice().as_deref(),
            Some(initial_notice.as_str())
        );
        app.date_text = "2021-06-20".into();
        app.schedule_recompute();
        assert_eq!(
            app.history_coverage,
            super::HistoryCoverage::Limited {
                cutoff: NaiveDate::from_ymd_opt(2021, 6, 19).unwrap(),
                expected_days: 1,
                covered_days: 0,
            }
        );
        app.mode = GuiSolverMode::Absurdle;
        app.schedule_recompute();
        assert_eq!(app.history_coverage, super::HistoryCoverage::NotRequested);
        drop(app);
        drop(root);
    }

    #[test]
    fn history_coverage_counts_unique_dates_only_through_the_requested_cutoff() {
        let cutoff = NaiveDate::from_ymd_opt(2021, 6, 21).unwrap();
        let entry = |day| crate::data::NytDailyEntry {
            id: None,
            print_date: NaiveDate::from_ymd_opt(2021, 6, day).unwrap(),
            solution: "cigar".into(),
            days_since_launch: None,
            editor: None,
        };
        for (days, covered_days) in [
            (vec![], 0),
            (vec![20, 21], 2),
            (vec![19, 21], 2),
            (vec![19, 20], 2),
            (vec![19, 19, 21, 22, 23], 2),
        ] {
            let history = days.into_iter().map(entry).collect::<Vec<_>>();
            assert_eq!(
                super::HistoryCoverage::through_cutoff(&history, cutoff),
                super::HistoryCoverage::Limited {
                    cutoff,
                    expected_days: 3,
                    covered_days,
                }
            );
        }
        assert_eq!(
            super::HistoryCoverage::through_cutoff(
                &[entry(21), entry(20), entry(19), entry(22)],
                cutoff
            ),
            super::HistoryCoverage::Complete
        );
        assert_eq!(
            super::HistoryCoverage::through_cutoff(
                &[],
                NaiveDate::from_ymd_opt(2021, 6, 18).unwrap()
            ),
            super::HistoryCoverage::Complete
        );
    }

    #[test]
    fn worker_response_identity_includes_controls_not_only_generation() {
        let (solver, root) = worker_fixture_solver();
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        let mut stale = app.current_request.clone().unwrap();
        stale.date_text = "2000-01-01".into();
        let (sender, receiver) = mpsc::channel();
        app.response_receiver = receiver;
        sender
            .send(super::WorkerResponse {
                request: stale,
                payload: Err("stale failure".into()),
            })
            .unwrap();
        app.drain_worker_responses();
        assert!(app.computing);
        assert!(!app.status.is_error());
        drop(app);
        drop(root);
    }

    #[test]
    fn feedback_palette_meets_normal_text_contrast() {
        let luminance = |color: eframe::egui::Color32| {
            let channels = [color.r(), color.g(), color.b()].map(|value| {
                let channel = f64::from(value) / 255.0;
                if channel <= 0.04045 {
                    channel / 12.92
                } else {
                    ((channel + 0.055) / 1.055).powf(2.4)
                }
            });
            0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
        };
        for value in 0..3 {
            let background = luminance(super::tile_label_and_color(value).1);
            let foreground = luminance(super::tile_label_and_color(value).2);
            let contrast =
                (background.max(foreground) + 0.05) / (background.min(foreground) + 0.05);
            assert!(contrast >= 4.5, "state {value} contrast {contrast}");
        }
    }

    #[test]
    fn setup_disconnect_is_not_an_infinite_loading_state() {
        let root = crate::test_support::TestDirectory::new("gui-setup-disconnect");
        let mut app =
            super::SetupApp::new(crate::data::ProjectPaths::new(root.path()), String::new());
        let (sender, receiver) = mpsc::channel();
        app.receiver = Some(receiver);
        app.running = true;
        drop(sender);
        let context = eframe::egui::Context::default();
        let _ = context.run(Default::default(), |ctx| {
            assert!(app.update(ctx).is_none());
        });
        assert!(!app.running);
        assert!(app.error.contains("disconnected"));
    }

    #[test]
    fn material_request_changes_invalidate_old_result_details() {
        let (solver, root) = worker_fixture_solver();
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        app.surviving_count = 32;
        app.total_weight = 1.0;
        app.selected_suggestion = Some("cigar".to_string());
        app.predictive_model_metadata = "old request".to_string();
        app.predictive_execution = Some(
            app.predictive_solver
                .suggest_predictive(PredictiveSuggestRequest {
                    puzzle_date: NaiveDate::from_ymd_opt(2026, 7, 26).unwrap(),
                    observations: &[],
                    top: 1,
                    hard_mode: false,
                    force_in_two_only: false,
                    mode: PredictiveSuggestionMode::LiveOnly,
                })
                .unwrap()
                .execution,
        );
        app.date_text = "2026-07-27".to_string();
        app.schedule_recompute();
        assert_eq!(app.surviving_count, 0);
        assert_eq!(app.total_weight, 0.0);
        assert!(app.selected_suggestion.is_none());
        assert!(app.predictive_model_metadata.is_empty());
        assert!(app.predictive_execution.is_none());
        assert!(app.computing);
        drop(app);
        drop(root);
    }

    #[test]
    fn live_trace_renders_actual_execution_not_configured_thresholds() {
        let (mut solver, root) = worker_fixture_solver();
        solver.config.search_policy_mode = crate::config::SearchPolicyMode::ProxyOnly;
        solver.config.exact_exhaustive_threshold = solver.guesses.len();
        solver.config.exact_threshold = solver.guesses.len();
        let response = solver
            .suggest_predictive(PredictiveSuggestRequest {
                puzzle_date: NaiveDate::from_ymd_opt(2026, 7, 26).unwrap(),
                observations: &[],
                top: 1,
                hard_mode: false,
                force_in_two_only: false,
                mode: PredictiveSuggestionMode::LiveOnly,
            })
            .unwrap();
        assert!(response.state.surviving <= solver.config.exact_exhaustive_threshold);
        let mut app = super::WordleGuiApp::new(super::LoadedWorkspace {
            config: solver.config.clone(),
            predictive_solver: solver,
            formal_solver: None,
        });
        app.predictive_execution = Some(response.execution);
        app.surviving_count = response.state.surviving;
        let context = eframe::egui::Context::default();
        let output = context.run(Default::default(), |ctx| {
            eframe::egui::CentralPanel::default().show(ctx, |ui| app.show_live_trace(ui));
        });
        let text = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                eframe::egui::Shape::Text(text) => Some(text.galley.text()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("route=proxy"), "{text}");
        assert!(text.contains("heuristic proxy"), "{text}");
        assert!(!text.contains("exhaustive exact"));
        drop(app);
        drop(root);
    }

    #[test]
    fn latest_worker_queue_replaces_obsolete_pending_request() {
        let shared = Arc::new((Mutex::new(LatestWorkerQueue::default()), Condvar::new()));
        let dispatcher = LatestWorkerDispatcher {
            shared: Arc::clone(&shared),
        };
        let request = |generation| WorkerRequest {
            generation,
            mode: GuiSolverMode::Predictive,
            date_text: "2026-07-26".to_string(),
            observations: Vec::new(),
            top: 10,
            force_in_two_only: false,
            hard_mode: false,
            search_profile: GuiSearchProfile::Configured,
        };
        dispatcher.send(request(4)).expect("first request");
        dispatcher.send(request(5)).expect("replacement request");
        let queue = shared.0.lock().expect("queue");
        assert_eq!(
            queue.pending.as_ref().map(|request| request.generation),
            Some(5)
        );
        assert_eq!(queue.latest_generation, 5);
    }

    #[test]
    fn spawned_worker_hands_off_to_superseding_predictive_request() {
        let (solver, root) = worker_fixture_solver();
        let (dispatcher, receiver) = spawn_worker(solver, None).expect("worker spawn");
        dispatcher
            .send(predictive_worker_request(1, GuiSearchProfile::FiniteStrong))
            .expect("first predictive request");
        dispatcher
            .send(predictive_worker_request(2, GuiSearchProfile::FiniteFast))
            .expect("superseding predictive request");

        let current = loop {
            let response = receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("superseding predictive request did not complete");
            if response.request.generation == 2 {
                break response;
            }
        };
        assert_eq!(current.request.generation, 2);
        match current.payload.expect("current predictive payload") {
            WorkerPayload::Predictive { finite_search, .. } => {
                assert!(
                    finite_search.is_some(),
                    "finite request lost its bounded result"
                );
            }
            payload => panic!("unexpected payload after predictive handoff: {payload:?}"),
        }

        drop(dispatcher);
        drop(root);
    }

    #[test]
    fn spawned_worker_cancels_in_flight_obsolete_request_before_next_request() {
        let (solver, root) = worker_fixture_solver();
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Arc::new(Mutex::new(release_receiver));
        let first_poll = Arc::new(AtomicBool::new(false));
        let hook_poll = Arc::clone(&first_poll);
        let hook_release = Arc::clone(&release_receiver);
        let hook = Arc::new(move |generation| {
            if generation == 1 && !hook_poll.swap(true, AtomicOrdering::SeqCst) {
                started_sender
                    .send(generation)
                    .expect("worker start receiver");
                hook_release
                    .lock()
                    .expect("release receiver")
                    .recv()
                    .expect("release first request");
            }
        });
        let (dispatcher, receiver) = spawn_worker(solver, None).expect("worker spawn");
        {
            let (lock, _) = &*dispatcher.shared;
            let mut queue = lock.lock().expect("queue");
            queue.test_cancel_hook = Some(hook);
        }

        let mut first_request = predictive_worker_request(1, GuiSearchProfile::FiniteStrong);
        first_request.observations.clear();
        dispatcher
            .send(first_request)
            .expect("first predictive request");
        assert_eq!(
            started_receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("first request did not start before supersession"),
            1
        );

        dispatcher
            .send(predictive_worker_request(2, GuiSearchProfile::FiniteFast))
            .expect("superseding predictive request");
        release_sender.send(()).expect("release first request");

        let mut cancelled_obsolete_response = false;
        let current = loop {
            let response = receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("superseding predictive request did not complete");
            if response.request.generation == 1 {
                let WorkerPayload::Predictive { finite_search, .. } =
                    response.payload.expect("obsolete predictive payload")
                else {
                    panic!("unexpected obsolete worker payload");
                };
                assert_eq!(
                    finite_search.expect("obsolete finite metadata").reason,
                    FiniteSearchReason::Cancelled
                );
                cancelled_obsolete_response = true;
            } else if response.request.generation == 2 {
                break response;
            }
        };
        assert!(cancelled_obsolete_response);
        assert_eq!(current.request.generation, 2);
        assert!(current.payload.is_ok());

        drop(dispatcher);
        drop(root);
    }

    #[test]
    fn cancellable_configured_search_stops_after_ranking_begins() {
        let (mut solver, root) = worker_fixture_solver();
        solver.config.search_policy_mode = crate::config::SearchPolicyMode::Staged;
        let polls = Arc::new(AtomicUsize::new(0));
        let cancellation_polls = Arc::clone(&polls);
        let cancelled = move || {
            let poll = cancellation_polls.fetch_add(1, AtomicOrdering::SeqCst);
            // The first four polls are the API/state/search-entry checks. One
            // ranking worker is allowed to start before cancellation becomes
            // true, so this exercises cancellation from the ranking stage.
            poll >= 5
        };
        let observations = [("cigar".to_string(), 0)];
        let request = PredictiveSuggestRequest {
            puzzle_date: NaiveDate::from_ymd_opt(2026, 7, 26).expect("fixture date"),
            observations: &observations,
            top: 3,
            hard_mode: false,
            force_in_two_only: false,
            mode: PredictiveSuggestionMode::FastDiskOnly,
        };

        let error = solver
            .suggest_predictive_cancellable(request, &cancelled)
            .expect_err("configured legacy search should observe cancellation");
        assert!(error.to_string().contains("predictive search cancelled"));
        assert!(
            polls.load(AtomicOrdering::SeqCst) >= 6,
            "cancellation must be observed by the ranking stage"
        );

        drop(root);
    }

    #[test]
    fn rendered_surface_states_cover_setup_loading_normal_recovery_and_error() {
        assert_eq!(
            gui_surface_state(true, false, &GuiStatus::Ready, None, 0),
            GuiSurfaceState::MissingData
        );
        assert_eq!(
            gui_surface_state(false, true, &GuiStatus::Ready, None, 0),
            GuiSurfaceState::Loading
        );
        assert_eq!(
            gui_surface_state(false, false, &GuiStatus::Ready, None, 0),
            GuiSurfaceState::Empty
        );
        assert_eq!(
            gui_surface_state(false, false, &GuiStatus::Ready, None, 3),
            GuiSurfaceState::Normal
        );
        assert_eq!(
            gui_surface_state(
                false,
                false,
                &GuiStatus::Ready,
                Some(crate::predictive::RecoveryMode::EpsilonRepair),
                3,
            ),
            GuiSurfaceState::Recovery
        );
        assert_eq!(
            gui_surface_state(
                false,
                false,
                &GuiStatus::WorkerError("bad model".into()),
                None,
                0
            ),
            GuiSurfaceState::Error
        );
    }

    #[test]
    fn compact_layout_switches_before_the_two_column_workspace_becomes_cramped() {
        assert!(is_compact_layout(559.0));
        assert!(is_compact_layout(859.0));
        assert!(!is_compact_layout(860.0));
        assert!(!is_compact_layout(1180.0));
    }

    #[test]
    fn worker_cancellation_tracks_newer_generation_and_shutdown() {
        let shared = Arc::new((Mutex::new(LatestWorkerQueue::default()), Condvar::new()));
        shared.0.lock().expect("queue").latest_generation = 4;
        assert!(!worker_request_cancelled(&shared, 4));
        {
            let mut queue = shared.0.lock().expect("queue");
            queue.latest_generation = 5;
        }
        assert!(worker_request_cancelled(&shared, 4));
        assert!(!worker_request_cancelled(&shared, 5));
        {
            let mut queue = shared.0.lock().expect("queue");
            queue.shutdown = true;
        }
        assert!(worker_request_cancelled(&shared, 5));
    }

    #[test]
    fn predictive_banner_text_matches_artifact_state() {
        assert_eq!(
            predictive_banner_text(PredictiveArtifactState::ExactDateArtifact),
            "Using exact-date predictive artifact"
        );
        assert_eq!(
            predictive_banner_text(PredictiveArtifactState::RecentOpenerArtifact),
            "Using recent opener artifact"
        );
        assert_eq!(
            predictive_banner_text(PredictiveArtifactState::LiveSessionFallback),
            "Using live session fallback"
        );
        assert_eq!(
            predictive_banner_text(PredictiveArtifactState::NoPredictiveArtifactAvailable),
            "No predictive artifact available"
        );
    }

    #[test]
    fn predictive_compute_status_includes_path_hint() {
        assert_eq!(
            predictive_compute_status(PredictiveArtifactState::ExactDateArtifact),
            "Computing... disk-backed"
        );
        assert_eq!(
            predictive_compute_status(PredictiveArtifactState::LiveSessionFallback),
            "Computing... live session fallback"
        );
    }

    #[test]
    fn predictive_reply_book_text_matches_branch_state() {
        assert_eq!(
            predictive_reply_book_text(1, PredictiveArtifactState::ExactDateArtifact),
            Some("Reply-book artifact is available for this branch.")
        );
        assert_eq!(
            predictive_reply_book_text(2, PredictiveArtifactState::LiveSessionFallback),
            Some(
                "Reply-book artifact is missing for this date or branch; showing live ranking only."
            )
        );
        assert_eq!(
            predictive_reply_book_text(0, PredictiveArtifactState::NoPredictiveArtifactAvailable),
            None
        );
    }

    #[test]
    fn finite_display_keeps_unevaluated_fallbacks_truthful() {
        let heuristic = FiniteSearchCandidate {
            guess_index: 0,
            failure_probability: 0.25,
            expected_attempts: 2.5,
            quality: FiniteSearchQuality::Heuristic,
        };
        assert_eq!(finite_quality_label(heuristic.quality), "pending heuristic");
        assert_eq!(format_finite_failure_risk(&heuristic), "pending");
        assert_eq!(format_finite_expected_attempts(&heuristic), "unevaluated");

        let exact = FiniteSearchCandidate {
            quality: FiniteSearchQuality::Exact,
            ..heuristic
        };
        assert_eq!(
            finite_quality_label(exact.quality),
            "model-exact action value"
        );
        assert_eq!(format_finite_failure_risk(&exact), "25.000%");
        assert_eq!(format_finite_expected_attempts(&exact), "2.500");
    }

    #[test]
    fn finite_profile_and_completion_status_expose_boundaries() {
        assert_eq!(predictive_response_status(None), GuiStatus::Ready);
        assert!(!predictive_response_status(None).is_error());
        assert_eq!(
            GuiSearchProfile::FiniteFast.label(),
            "Finite fast (experimental)"
        );
        assert_eq!(
            GuiSearchProfile::FiniteStrong.label(),
            "Finite strong (experimental)"
        );
        assert_eq!(
            predictive_finite_banner_text(GuiSearchProfile::FiniteFast),
            "Using finite fast search (experimental)"
        );
        let search = FiniteSearchResult {
            root_candidates_considered: 0,
            all_legal_roots_evaluated: false,
            candidates: Vec::new(),
            reason: FiniteSearchReason::Deadline,
            nodes_visited: 3,
            work_units: 4,
            cache_hits: 1,
            proposal_sampled: false,
        };
        assert!(matches!(
            predictive_response_status(Some(&search)),
            GuiStatus::Notice(_)
        ));
    }

    #[test]
    fn predictive_registry_summary_uses_the_current_registry() {
        assert_eq!(
            predictive_registry_summary(&crate::config::PriorConfig::default()),
            "v7 · 84 leaves · 78 tunable"
        );
    }

    #[test]
    fn legacy_hard_mode_payload_labels_relaxed_continuations() {
        let (solver, root) = worker_fixture_solver();
        let mut request = predictive_worker_request(1, GuiSearchProfile::FiniteFast);
        request.hard_mode = true;
        let bounded = super::compute_worker_payload(&solver, None, &request, &|| false)
            .expect("experimental bounded hard-mode preview");
        let super::WorkerPayload::Predictive {
            model_metadata,
            finite_search,
            ..
        } = bounded
        else {
            panic!("expected predictive payload");
        };
        assert!(finite_search.is_some());
        assert!(!model_metadata.contains("normal-mode future replies"));

        request.search_profile = GuiSearchProfile::Configured;
        let payload = super::compute_worker_payload(&solver, None, &request, &|| false)
            .expect("configured hard-mode result");
        let super::WorkerPayload::Predictive { model_metadata, .. } = payload else {
            panic!("expected predictive payload");
        };
        assert!(model_metadata.contains("normal-mode future replies"));
        drop(solver);
        drop(root);
    }

    #[test]
    fn formal_unavailable_text_is_actionable() {
        assert_eq!(
            formal_unavailable_text(),
            "Formal artifacts missing; run build-optimal-policy first."
        );
    }

    #[test]
    fn decorative_game_board_tiles_are_not_accessibility_buttons() {
        let context = eframe::egui::Context::default();
        context.enable_accesskit();
        let output = context.run(eframe::egui::RawInput::default(), |context| {
            eframe::egui::CentralPanel::default().show(context, |ui| {
                super::show_game_board(ui, &[("olate".to_string(), 0)], "crane", [0; 5]);
            });
        });
        let update = output
            .platform_output
            .accesskit_update
            .expect("accessibility tree");
        assert!(
            update
                .nodes
                .iter()
                .all(|(_, node)| node.role() != eframe::egui::accesskit::Role::Button),
            "display-only board tiles must not appear as actionable buttons"
        );
        assert!(
            update
                .nodes
                .iter()
                .filter(|(_, node)| node.role() == eframe::egui::accesskit::Role::Label)
                .count()
                >= 10,
            "filled and draft board tiles should remain readable labels"
        );
    }

    #[test]
    fn first_guess_board_tiles_stay_inside_their_column() {
        let context = eframe::egui::Context::default();
        for width in [861.0, 1180.0] {
            for applied in [false, true] {
                let mut suggestions_left = None;
                let output = context.run(
                    eframe::egui::RawInput {
                        screen_rect: Some(eframe::egui::Rect::from_min_size(
                            eframe::egui::Pos2::ZERO,
                            eframe::egui::vec2(width, 820.0),
                        )),
                        ..Default::default()
                    },
                    |context| {
                        eframe::egui::CentralPanel::default().show(context, |ui| {
                            ui.columns(2, |columns| {
                                columns[0].group(|ui| {
                                    let observations = if applied {
                                        vec![("olate".to_string(), 0)]
                                    } else {
                                        Vec::new()
                                    };
                                    let draft = if applied { "" } else { "olate" };
                                    super::show_game_board(ui, &observations, draft, [0; 5]);
                                });
                                let response = columns[1].group(|ui| {
                                    ui.heading("Wordle Suggestions");
                                });
                                suggestions_left = Some(response.response.rect.left());
                            });
                        });
                    },
                );
                let tile_color = if applied {
                    super::tile_label_and_color(0).1
                } else {
                    eframe::egui::Color32::from_rgb(225, 218, 208)
                };
                let tile_rects = output
                    .shapes
                    .iter()
                    .filter_map(|shape| match &shape.shape {
                        eframe::egui::Shape::Rect(rect) if rect.fill == tile_color => {
                            Some(rect.rect)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let suggestions_left = suggestions_left.expect("suggestions column should render");
                assert!(
                    tile_rects.len() >= 5,
                    "first-guess tiles should be painted: applied={applied}"
                );
                assert!(
                    tile_rects.iter().all(|rect| rect.width() <= 60.0),
                    "first-guess tiles must stay compact at width {width}, applied={applied}: {tile_rects:?}"
                );
                assert!(
                    tile_rects
                        .iter()
                        .all(|rect| rect.right() <= suggestions_left),
                    "first-guess tiles must end before suggestions at width {width}, applied={applied}: {tile_rects:?}; suggestions start at {suggestions_left}"
                );
            }
        }
    }

    #[test]
    fn production_play_layout_bounds_first_guess_and_recommendations_near_compact_threshold() {
        for (width, applied) in [
            (1180.0, false),
            (1210.0, false),
            (1240.0, true),
            (1260.0, true),
        ] {
            let (solver, root) = worker_fixture_solver();
            let (request_sender, response_receiver) =
                spawn_worker(solver.clone(), None).expect("worker spawn");
            let mut app = super::WordleGuiApp {
                config: crate::config::PriorConfig::default(),
                predictive_solver: solver,
                formal_solver: None,
                formal_availability: super::FormalAvailability::Absent,
                formal_receiver: None,
                formal_cancel: Arc::new(AtomicBool::new(false)),
                workspace_view: super::WorkspaceView::Play,
                mode: GuiSolverMode::Predictive,
                search_profile: GuiSearchProfile::Configured,
                text_scale: 1.35,
                date_text: "2026-07-26".to_string(),
                current_guess: "crane".to_string(),
                feedback_code: "01210".to_string(),
                current_feedback: [0, 1, 2, 1, 0],
                observations: if applied {
                    vec![("wawww".to_string(), 48)]
                } else {
                    Vec::new()
                },
                predictive_suggestions: vec![Suggestion {
                    value_kind: crate::predictive::types::SuggestionValueKind::Proxy,
                    finite_value: None,
                    word: "humph".to_string(),
                    entropy: 1.0,
                    solve_probability: 0.2,
                    expected_remaining: 2.0,
                    force_in_two: false,
                    known_absent_letter_hits: 0,
                    worst_non_green_bucket_size: 4,
                    largest_non_green_bucket_mass: 0.4,
                    large_non_green_bucket_count: 1,
                    dangerous_mass_bucket_count: 1,
                    non_green_mass_in_large_buckets: 0.4,
                    proxy_cost: Some(1.0),
                    large_state_score: Some(0.5),
                    posterior_answer_probability: 0.2,
                    lookahead_cost: Some(1.2),
                    exact_cost: Some(1.3),
                }],
                predictive_candidates: Vec::new(),
                candidate_filter: String::new(),
                absurdle_suggestions: Vec::new(),
                formal_suggestions: Vec::new(),
                surviving_count: 1,
                total_weight: 1.0,
                predictive_recovery_mode: None,
                predictive_artifact_state: PredictiveArtifactState::NoPredictiveArtifactAvailable,
                predictive_model_metadata: String::new(),
                predictive_finite_search: None,
                predictive_execution: None,
                history_coverage: super::HistoryCoverage::NotRequested,
                top: 10,
                force_in_two_only: false,
                hard_mode: false,
                status: GuiStatus::Ready,
                formal_explanation: None,
                suggestion_sort: super::SuggestionSort::Rank,
                suggestion_sort_descending: false,
                selected_suggestion: None,
                request_sender: Some(request_sender),
                response_receiver,
                latest_generation: 0,
                current_request: None,
                computing: false,
            };
            let screen_rect = eframe::egui::Rect::from_min_size(
                eframe::egui::Pos2::ZERO,
                eframe::egui::vec2(width, 1400.0),
            );
            let context = eframe::egui::Context::default();
            let mut frame = eframe::Frame::_new_kittest();
            let output = context.run(
                eframe::egui::RawInput {
                    screen_rect: Some(screen_rect),
                    ..Default::default()
                },
                |context| eframe::App::update(&mut app, context, &mut frame),
            );

            let text_rect = |needle: &str| {
                output.shapes.iter().find_map(|shape| match &shape.shape {
                    eframe::egui::Shape::Text(text)
                        if text.galley.text().replace('\n', " ").contains(needle) =>
                    {
                        Some(text.visual_bounding_rect())
                    }
                    _ => None,
                })
            };
            let board_heading = text_rect("Game Board")
                .or_else(|| text_rect("BOARD / HISTORY"))
                .expect("production board heading should render");
            let recommendations_heading = text_rect("Wordle Suggestions")
                .or_else(|| text_rect("NEXT ACTION"))
                .expect("production recommendations heading should render");
            let recommendations_group = output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    eframe::egui::Shape::Rect(rect)
                        if rect.fill == eframe::egui::Color32::TRANSPARENT
                            && rect.stroke.width > 0.0
                            && rect.rect.contains(recommendations_heading.center()) =>
                    {
                        Some(rect.rect)
                    }
                    _ => None,
                })
                .expect("production recommendations group should render");

            let board_fills = [
                eframe::egui::Color32::from_rgb(225, 218, 208),
                super::tile_label_and_color(0).1,
                super::tile_label_and_color(1).1,
                super::tile_label_and_color(2).1,
            ];
            let board_tiles = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    eframe::egui::Shape::Rect(rect)
                        if board_fills.contains(&rect.fill)
                            && rect.rect.top() >= board_heading.bottom() =>
                    {
                        Some(rect.rect)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(
                board_tiles.len() >= 30,
                "the six-row board should render at width {width}, applied={applied}: {board_tiles:?}"
            );
            if text_rect("Wordle Suggestions").is_some() {
                assert!(
                    board_tiles
                        .iter()
                        .all(|rect| rect.right() <= recommendations_heading.left()),
                    "board tiles must stay left of recommendations at width {width}, applied={applied}: {board_tiles:?}; recommendations start at {}",
                    recommendations_heading.left()
                );
            } else {
                assert!(
                    board_tiles
                        .iter()
                        .all(|rect| rect.bottom() <= recommendations_heading.top()),
                    "stacked board must end before recommendations at width {width}, applied={applied}: {board_tiles:?}; recommendations start at {}",
                    recommendations_heading.top()
                );
            }

            let recommendation_text = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    eframe::egui::Shape::Text(text) => Some(text.visual_bounding_rect()),
                    _ => None,
                })
                .filter(|rect| {
                    rect.min.x.is_finite()
                        && rect.min.y.is_finite()
                        && rect.max.x.is_finite()
                        && rect.max.y.is_finite()
                        && rect.top() >= recommendations_heading.top()
                        && rect.top() <= recommendations_group.bottom()
                        && rect.left() >= recommendations_heading.left() - 2.0
                })
                .collect::<Vec<_>>();
            assert!(
                !recommendation_text.is_empty(),
                "recommendation content should render at width {width}"
            );
            assert!(
                text_rect("HUMPH").is_some(),
                "the seeded recommendation should render at width {width}"
            );
            assert!(
                recommendation_text
                    .iter()
                    .all(|rect| recommendations_group.contains_rect(*rect)),
                "recommendation content must stay within its group at width {width}: group {recommendations_group:?}, text {recommendation_text:?}"
            );

            drop(app);
            drop(root);
        }
    }

    #[test]
    fn guess_feedback_controls_stay_inside_minimum_window() {
        let (solver, root) = worker_fixture_solver();
        let (request_sender, response_receiver) =
            spawn_worker(solver.clone(), None).expect("worker spawn");
        let mut app = super::WordleGuiApp {
            config: crate::config::PriorConfig::default(),
            predictive_solver: solver,
            formal_solver: None,
            formal_availability: super::FormalAvailability::Absent,
            formal_receiver: None,
            formal_cancel: Arc::new(AtomicBool::new(false)),
            workspace_view: super::WorkspaceView::Play,
            mode: GuiSolverMode::Predictive,
            search_profile: GuiSearchProfile::Configured,
            text_scale: 1.0,
            date_text: "2026-07-26".to_string(),
            current_guess: "olate".to_string(),
            feedback_code: "00000".to_string(),
            current_feedback: [0; 5],
            observations: Vec::new(),
            predictive_suggestions: Vec::new(),
            predictive_candidates: Vec::new(),
            candidate_filter: String::new(),
            absurdle_suggestions: Vec::new(),
            formal_suggestions: Vec::new(),
            surviving_count: 0,
            total_weight: 0.0,
            predictive_recovery_mode: None,
            predictive_artifact_state: PredictiveArtifactState::NoPredictiveArtifactAvailable,
            predictive_model_metadata: String::new(),
            predictive_finite_search: None,
            predictive_execution: None,
            history_coverage: super::HistoryCoverage::NotRequested,
            top: 10,
            force_in_two_only: false,
            hard_mode: false,
            status: GuiStatus::Ready,
            formal_explanation: None,
            suggestion_sort: super::SuggestionSort::Rank,
            suggestion_sort_descending: false,
            selected_suggestion: None,
            request_sender: Some(request_sender),
            response_receiver,
            latest_generation: 0,
            current_request: None,
            computing: false,
        };
        let screen_rect = eframe::egui::Rect::from_min_size(
            eframe::egui::Pos2::ZERO,
            eframe::egui::vec2(560.0, 620.0),
        );
        let context = eframe::egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let output = context.run(
            eframe::egui::RawInput {
                screen_rect: Some(screen_rect),
                ..Default::default()
            },
            |context| eframe::App::update(&mut app, context, &mut frame),
        );
        let feedback_tiles = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                eframe::egui::Shape::Rect(rect)
                    if rect.fill == super::tile_label_and_color(0).1 =>
                {
                    Some(rect.rect)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        drop(app);
        drop(root);

        assert_eq!(
            feedback_tiles.len(),
            5,
            "all feedback buttons should render"
        );
        assert!(
            feedback_tiles
                .iter()
                .all(|rect| screen_rect.contains_rect(*rect)),
            "feedback buttons should not extend beyond the minimum window: {feedback_tiles:?}"
        );
    }
}
