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
    data::{ProjectPaths, sync_nyt_history},
    formal::{
        DEFAULT_FORMAL_MODEL_ID, FormalPolicyRuntime, FormalStateExplanation, FormalSuggestion,
        artifacts_exist,
    },
    model::build_model_artifacts,
    predictive::{
        PredictiveArtifactState, PredictiveCandidateSummary, PredictiveStateSummary,
        PredictiveSuggestRequest, PredictiveSuggestionMode, RecoveryMode,
    },
    scoring::parse_feedback,
    solver::{AbsurdleSuggestion, SolveState, Solver, Suggestion},
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

struct LoadedWorkspace {
    config: PriorConfig,
    predictive_solver: Solver,
    formal_solver: Option<FormalPolicyRuntime>,
}

fn load_workspace(paths: &ProjectPaths) -> Result<LoadedWorkspace> {
    paths.ensure_layout()?;
    let config = PriorConfig::load_or_create(&paths.config_prior)?;
    let predictive_solver = Solver::from_paths(paths, &config)?;
    let formal_solver = if artifacts_exist(paths, DEFAULT_FORMAL_MODEL_ID) {
        Some(FormalPolicyRuntime::load(paths, DEFAULT_FORMAL_MODEL_ID)?)
    } else {
        None
    };
    Ok(LoadedWorkspace {
        config,
        predictive_solver,
        formal_solver,
    })
}

enum MaybeWordleApp {
    Ready(Box<WordleGuiApp>),
    Setup(Box<SetupApp>),
}

impl MaybeWordleApp {
    fn load_or_setup(paths: ProjectPaths) -> Self {
        match load_workspace(&paths) {
            Ok(workspace) => Self::Ready(Box::new(WordleGuiApp::new(workspace))),
            Err(error) => Self::Setup(Box::new(SetupApp::new(paths, error.to_string()))),
        }
    }
}

impl eframe::App for MaybeWordleApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        match self {
            Self::Ready(app) => app.update(ctx, frame),
            Self::Setup(app) => {
                if let Some(workspace) = app.update(ctx) {
                    *self = Self::Ready(Box::new(WordleGuiApp::new(workspace)));
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SetupAction {
    SyncAndBuild,
    BuildLocal,
    RetryLoad,
}

impl SetupAction {
    fn label(self) -> &'static str {
        match self {
            Self::SyncAndBuild => "Sync history & build",
            Self::BuildLocal => "Build from local data",
            Self::RetryLoad => "Retry existing model",
        }
    }
}

enum SetupEvent {
    Progress(String),
    Complete(Box<std::result::Result<LoadedWorkspace, String>>),
}

struct SetupApp {
    paths: ProjectPaths,
    error: String,
    progress: String,
    running: bool,
    cancel_requested: Arc<AtomicBool>,
    receiver: Option<Receiver<SetupEvent>>,
}

impl SetupApp {
    fn new(paths: ProjectPaths, error: String) -> Self {
        Self {
            paths,
            error,
            progress: "Model data is not ready yet.".to_string(),
            running: false,
            cancel_requested: Arc::new(AtomicBool::new(false)),
            receiver: None,
        }
    }

    fn start(&mut self, action: SetupAction) {
        if self.running {
            return;
        }
        self.cancel_requested = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::clone(&self.cancel_requested);
        let paths = self.paths.clone();
        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);
        self.running = true;
        self.error.clear();
        self.progress = format!("{}…", action.label());
        let spawn_result = thread::Builder::new()
            .name("maybe-wordle-setup".to_string())
            .stack_size(SOLVER_THREAD_STACK_BYTES)
            .spawn(move || {
                let result = (|| -> Result<LoadedWorkspace> {
                    paths.ensure_layout()?;
                    let config = PriorConfig::load_or_create(&paths.config_prior)?;
                    match action {
                        SetupAction::SyncAndBuild => {
                            let _ = sender.send(SetupEvent::Progress(
                                "Syncing NYT history. You can keep using other apps.".to_string(),
                            ));
                            let summary = sync_nyt_history(&paths, &config, Solver::today())?;
                            if cancelled.load(Ordering::Acquire) {
                                anyhow::bail!("setup cancelled after history sync");
                            }
                            let _ = sender.send(SetupEvent::Progress(format!(
                                "Synced {} of {} dates; building local model…",
                                summary.fetched, summary.total
                            )));
                            build_model_artifacts(&paths, &config, Solver::today())?;
                        }
                        SetupAction::BuildLocal => {
                            let _ = sender.send(SetupEvent::Progress(
                                "Building model from the local seed and history files…".to_string(),
                            ));
                            build_model_artifacts(&paths, &config, Solver::today())?;
                        }
                        SetupAction::RetryLoad => {}
                    }
                    if cancelled.load(Ordering::Acquire) {
                        anyhow::bail!("setup cancelled");
                    }
                    load_workspace(&paths)
                })()
                .map_err(|error| format!("{error:#}"));
                let _ = sender.send(SetupEvent::Complete(Box::new(result)));
            });
        if let Err(error) = spawn_result {
            self.receiver = None;
            self.running = false;
            self.progress = "Setup worker could not start.".to_string();
            self.error = format!("failed to start setup worker: {error}");
        }
    }

    fn update(&mut self, ctx: &egui::Context) -> Option<LoadedWorkspace> {
        ctx.set_visuals(workspace_visuals());
        let mut completed = None;
        if let Some(receiver) = &self.receiver {
            while let Ok(event) = receiver.try_recv() {
                match event {
                    SetupEvent::Progress(progress) => self.progress = progress,
                    SetupEvent::Complete(result) => {
                        self.running = false;
                        match *result {
                            Ok(workspace) => {
                                self.progress =
                                    "Model ready; opening the predictive desk…".to_string();
                                completed = Some(workspace);
                            }
                            Err(error) => {
                                self.progress =
                                    "Setup did not complete. Correct the issue or choose an action to retry."
                                        .to_string();
                                self.error = error;
                            }
                        }
                    }
                }
            }
        }
        if self.running {
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(Color32::from_rgb(246, 240, 232))
                    .inner_margin(32.0),
            )
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(28.0);
                    ui.label(
                        RichText::new("MAYBE / WORDLE")
                            .monospace()
                            .size(13.0)
                            .color(Color32::from_rgb(171, 73, 43)),
                    );
                    ui.heading(
                        RichText::new("Prepare the predictive desk")
                            .size(34.0)
                            .color(Color32::from_rgb(42, 49, 43)),
                    );
                    ui.label(
                        RichText::new(
                            "The app opens even when derived data is missing. Sync public history or rebuild from files already on this machine.",
                        )
                        .color(Color32::from_rgb(92, 72, 54)),
                    );
                    ui.add_space(24.0);
                });
                egui::Frame::group(ui.style())
                    .fill(Color32::from_rgb(255, 252, 247))
                    .inner_margin(24.0)
                    .show(ui, |ui| {
                        ui.label(RichText::new("SETUP STATUS").monospace().strong());
                        ui.label(&self.progress);
                        if !self.error.is_empty() {
                            ui.add_space(8.0);
                            ui.colored_label(Color32::from_rgb(150, 45, 45), &self.error);
                        }
                        ui.add_space(16.0);
                        ui.horizontal_wrapped(|ui| {
                            for action in [
                                SetupAction::SyncAndBuild,
                                SetupAction::BuildLocal,
                                SetupAction::RetryLoad,
                            ] {
                                if ui
                                    .add_enabled(!self.running, egui::Button::new(action.label()))
                                    .clicked()
                                {
                                    self.start(action);
                                }
                            }
                            if ui
                                .add_enabled(self.running, egui::Button::new("Cancel"))
                                .clicked()
                            {
                                self.cancel_requested.store(true, Ordering::Release);
                                self.progress =
                                    "Cancellation requested; finishing the current file or request…"
                                        .to_string();
                            }
                        });
                    });
            });
        completed
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuiSolverMode {
    FormalOptimal,
    Predictive,
    Absurdle,
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct BoardDraft {
    guess: String,
    feedback: [u8; 5],
}

enum BoardAction {
    ReplaceGuess(String),
    ReplaceFeedbackCode(String),
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
        BoardAction::ReplaceFeedbackCode(code) => {
            for (index, value) in code
                .chars()
                .filter_map(feedback_shortcut_value)
                .take(5)
                .enumerate()
            {
                next.feedback[index] = value;
            }
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

fn feedback_shortcut_value(character: char) -> Option<u8> {
    match character.to_ascii_lowercase() {
        '0' | 'b' | 'x' => Some(0),
        '1' | 'y' => Some(1),
        '2' | 'g' => Some(2),
        _ => None,
    }
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
    workspace_view: WorkspaceView,
    mode: GuiSolverMode,
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
    top: usize,
    force_in_two_only: bool,
    hard_mode: bool,
    status: String,
    formal_explanation: Option<FormalStateExplanation>,
    suggestion_sort: SuggestionSort,
    suggestion_sort_descending: bool,
    selected_suggestion: Option<usize>,
    request_sender: LatestWorkerDispatcher,
    response_receiver: Receiver<WorkerResponse>,
    latest_generation: u64,
    computing: bool,
}

#[derive(Default)]
struct LatestWorkerQueue {
    pending: Option<WorkerRequest>,
    shutdown: bool,
}

struct LatestWorkerDispatcher {
    shared: Arc<(Mutex<LatestWorkerQueue>, Condvar)>,
}

impl LatestWorkerDispatcher {
    fn send(&self, request: WorkerRequest) -> Result<()> {
        let (lock, ready) = &*self.shared;
        let mut queue = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("suggestion worker queue is poisoned"))?;
        if queue.shutdown {
            anyhow::bail!("suggestion workers have stopped");
        }
        queue.pending = Some(request);
        ready.notify_one();
        Ok(())
    }
}

impl Drop for LatestWorkerDispatcher {
    fn drop(&mut self) {
        let (lock, ready) = &*self.shared;
        if let Ok(mut queue) = lock.lock() {
            queue.shutdown = true;
            ready.notify_all();
        }
    }
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
        state: PredictiveStateSummary,
        suggestions: Vec<Suggestion>,
        candidates: Vec<PredictiveCandidateSummary>,
        artifact_state: PredictiveArtifactState,
        model_metadata: String,
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
    fn board_draft(&self) -> BoardDraft {
        BoardDraft {
            guess: self.current_guess.clone(),
            feedback: self.current_feedback,
        }
    }

    fn apply_board_action(&mut self, action: BoardAction) {
        let next = reduce_board_draft(&self.board_draft(), action);
        self.current_guess = next.guess;
        self.current_feedback = next.feedback;
        self.feedback_code = feedback_code(self.current_feedback);
    }

    fn update_feedback_code(&mut self, code: String) {
        let sanitized = code
            .chars()
            .filter(|character| feedback_shortcut_value(*character).is_some())
            .take(5)
            .collect::<String>();
        self.feedback_code = sanitized.clone();
        if sanitized.chars().count() == 5 {
            let next = reduce_board_draft(
                &self.board_draft(),
                BoardAction::ReplaceFeedbackCode(sanitized),
            );
            self.current_feedback = next.feedback;
            self.feedback_code = feedback_code(self.current_feedback);
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
        let (request_sender, response_receiver) =
            spawn_worker(predictive_solver.clone(), formal_solver.clone());
        let mut app = Self {
            config,
            predictive_solver,
            formal_solver,
            workspace_view: WorkspaceView::Play,
            mode,
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
            top: 10,
            force_in_two_only: false,
            hard_mode: false,
            status: String::new(),
            formal_explanation: None,
            suggestion_sort: SuggestionSort::Rank,
            suggestion_sort_descending: false,
            selected_suggestion: None,
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
        self.status = if self.mode == GuiSolverMode::Predictive {
            predictive_compute_status(self.predictive_artifact_state)
        } else {
            "Computing...".to_string()
        };
        if let Err(error) = self.request_sender.send(request) {
            self.set_error(error);
        }
    }

    fn drain_worker_responses(&mut self) {
        while let Ok(response) = self.response_receiver.try_recv() {
            if !worker_response_is_current(self.latest_generation, response.generation) {
                continue;
            }
            self.computing = false;
            match response.payload {
                Ok(WorkerPayload::Predictive {
                    state,
                    suggestions,
                    candidates,
                    artifact_state,
                    model_metadata,
                }) => {
                    self.surviving_count = state.surviving;
                    self.total_weight = state.effective_total_weight;
                    self.predictive_recovery_mode = state.recovery_mode_used;
                    self.predictive_artifact_state = artifact_state;
                    self.predictive_model_metadata = model_metadata;
                    self.predictive_suggestions = suggestions;
                    self.predictive_candidates = candidates;
                    if self
                        .selected_suggestion
                        .is_some_and(|index| index >= self.predictive_suggestions.len())
                    {
                        self.selected_suggestion = None;
                    }
                    self.absurdle_suggestions.clear();
                    self.formal_suggestions.clear();
                    self.formal_explanation = None;
                    self.status.clear();
                }
                Ok(WorkerPayload::Absurdle { state, suggestions }) => {
                    self.surviving_count = state.surviving.len();
                    self.total_weight = 0.0;
                    self.predictive_recovery_mode = None;
                    self.predictive_artifact_state =
                        PredictiveArtifactState::NoPredictiveArtifactAvailable;
                    self.absurdle_suggestions = suggestions;
                    self.predictive_suggestions.clear();
                    self.predictive_candidates.clear();
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
                    self.predictive_recovery_mode = None;
                    self.predictive_artifact_state =
                        PredictiveArtifactState::NoPredictiveArtifactAvailable;
                    self.formal_explanation = Some(explanation);
                    self.formal_suggestions = suggestions;
                    self.predictive_suggestions.clear();
                    self.predictive_candidates.clear();
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
        self.predictive_candidates.clear();
        self.absurdle_suggestions.clear();
        self.formal_suggestions.clear();
        self.formal_explanation = None;
        self.selected_suggestion = None;
        self.surviving_count = 0;
        self.total_weight = 0.0;
        self.predictive_recovery_mode = None;
        self.predictive_artifact_state = PredictiveArtifactState::NoPredictiveArtifactAvailable;
        self.computing = false;
    }

    fn commit_current_row(&mut self) {
        let guess = self.current_guess.trim().to_ascii_lowercase();
        if guess.is_empty() {
            self.status = "Enter a five-letter guess before applying the row.".to_string();
            return;
        }
        if self.mode == GuiSolverMode::Predictive
            && let Some(error) = self
                .predictive_solver
                .hard_mode_violation(&self.observations, &guess)
                .filter(|_| self.hard_mode)
        {
            self.status = error;
            return;
        }
        match self.row_pattern() {
            Ok(pattern) => {
                self.observations.push((guess, pattern));
                self.apply_board_action(BoardAction::Reset);
                self.schedule_recompute();
            }
            Err(error) => self.status = error.to_string(),
        }
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

    fn show_narrow_play(&mut self, ui: &mut egui::Ui) {
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
                        if ui
                            .selectable_label(
                                self.selected_suggestion == Some(index),
                                format!(
                                    "{}  {}  solve {:.3}  H {:.3}  cost {}",
                                    suggestion.word.to_ascii_uppercase(),
                                    suggestion_method(suggestion),
                                    suggestion.solve_probability,
                                    suggestion.entropy,
                                    format_suggestion_cost(suggestion)
                                ),
                            )
                            .clicked()
                        {
                            self.selected_suggestion = Some(index);
                        }
                    }
                    ui.separator();
                    show_suggestion_inspector(
                        ui,
                        self.selected_suggestion
                            .and_then(|index| self.predictive_suggestions.get(index)),
                        self.predictive_artifact_state,
                        self.predictive_recovery_mode,
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

    fn show_policy_panel(&mut self, ui: &mut egui::Ui) {
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
                    "As-of cutoff",
                    if self.date_text.is_empty() {
                        "not selected"
                    } else {
                        &self.date_text
                    },
                );
                policy_row(
                    ui,
                    "Artifact path",
                    predictive_banner_text(self.predictive_artifact_state),
                );
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
                policy_row(ui, "Registry", "v6 · 85 leaves · 79 tunable");
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
                if let Some(message) = predictive_reply_book_text(
                    self.observations.len(),
                    self.predictive_artifact_state,
                ) {
                    ui.label(message);
                }
            });
    }

    fn show_diagnostics_panel(&mut self, ui: &mut egui::Ui) {
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
                predictive_banner_text(self.predictive_artifact_state),
            );
            diagnostic_badge(ui, "SURVIVORS", &self.surviving_count.to_string());
        });
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
                "The two isolated workers share one replaceable pending slot: obsolete queued work is discarded, while a newer generation can run even if one older request is still expensive.",
            )
            .small()
            .color(Color32::from_rgb(92, 72, 54)),
        );
    }

    fn show_live_trace(&self, ui: &mut egui::Ui) {
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
        policy_row(
            ui,
            "Exact label",
            if self.surviving_count <= self.config.exact_exhaustive_threshold {
                "exhaustive exact"
            } else if self.surviving_count <= self.config.exact_threshold {
                "candidate-pool exact"
            } else {
                "proxy / lookahead allocation"
            },
        );
    }

    fn show_recovery_provenance(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("RECOVERY & PROVENANCE").monospace().strong());
        ui.label(if self.status.is_empty() {
            "No runtime error."
        } else {
            &self.status
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

    fn show_formal_panel(&mut self, ui: &mut egui::Ui) {
        if self.formal_solver.is_none() {
            ui.group(|ui| {
                ui.label(RichText::new("FORMAL ARTIFACTS MISSING").monospace().strong());
                ui.label(formal_unavailable_text());
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
        } else if !self.status.is_empty() {
            ui.colored_label(Color32::from_rgb(150, 45, 45), &self.status);
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
        ctx.set_zoom_factor(self.text_scale);
        ctx.set_visuals(workspace_visuals());
        self.drain_worker_responses();
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
                });

                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
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
                    if ui.button("Apply row").clicked() {
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
                let mut keyboard_apply = false;
                ui.horizontal(|ui| {
                    ui.label("Guess");
                    let response = ui.add_sized(
                        [120.0, 30.0],
                        egui::TextEdit::singleline(&mut self.current_guess).hint_text("e.g. crane"),
                    );
                    if response.changed() {
                        self.apply_board_action(BoardAction::ReplaceGuess(
                            self.current_guess.clone(),
                        ));
                    }
                    keyboard_apply |= response.lost_focus()
                        && ctx.input(|input| input.key_pressed(egui::Key::Enter));

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
                        let (_, color) = tile_label_and_color(*value);
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
                                    RichText::new(label)
                                        .size(15.0)
                                        .strong()
                                        .color(Color32::WHITE),
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
                if keyboard_apply {
                    self.commit_current_row();
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
                        ui.label(
                            RichText::new(predictive_banner_text(self.predictive_artifact_state))
                                .color(Color32::from_rgb(92, 72, 54)),
                        );
                        if !self.predictive_model_metadata.is_empty() {
                            ui.label(
                                RichText::new(&self.predictive_model_metadata)
                                    .small()
                                    .color(Color32::from_rgb(92, 72, 54)),
                            );
                        }
                        if let Some(message) = predictive_reply_book_text(
                            self.observations.len(),
                            self.predictive_artifact_state,
                        ) {
                            ui.label(
                                RichText::new(message).color(Color32::from_rgb(92, 72, 54)),
                            );
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
                if is_compact_layout(ui.available_width()) {
                    self.show_narrow_play(ui);
                    return;
                }
                ui.columns(2, |columns| {
                    columns[0].group(|ui| {
                        ui.heading("Game Board");
                        ui.label(
                            RichText::new(format!(
                                "{} of 6 rows applied",
                                self.observations.len()
                            ))
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
                        show_candidate_browser(
                            ui,
                            &mut self.candidate_filter,
                            &self.predictive_candidates,
                        );
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
                                if self.predictive_suggestions.is_empty() {
                                    ui.label(
                                        RichText::new(
                                            "No suggestions match the current filters.",
                                        )
                                        .color(Color32::from_rgb(92, 72, 54)),
                                    );
                                } else {
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
                                                    if ui.button("Rank").clicked() {
                                                        self.toggle_suggestion_sort(
                                                            SuggestionSort::Rank,
                                                        );
                                                    }
                                                    ui.label("Word");
                                                    ui.label("Method");
                                                    if ui.button("Solve").clicked() {
                                                        self.toggle_suggestion_sort(
                                                            SuggestionSort::SolveProbability,
                                                        );
                                                    }
                                                    if ui.button("Entropy").clicked() {
                                                        self.toggle_suggestion_sort(
                                                            SuggestionSort::Entropy,
                                                        );
                                                    }
                                                    if ui.button("Remain").clicked() {
                                                        self.toggle_suggestion_sort(
                                                            SuggestionSort::ExpectedRemaining,
                                                        );
                                                    }
                                                    if ui.button("Worst").clicked() {
                                                        self.toggle_suggestion_sort(
                                                            SuggestionSort::WorstBucket,
                                                        );
                                                    }
                                                    ui.label("Cost");
                                                    ui.end_row();

                                                    for (rank, index) in
                                                        sorted_indices.into_iter().enumerate()
                                                    {
                                                        let suggestion =
                                                            &self.predictive_suggestions[index];
                                                        ui.label((rank + 1).to_string());
                                                        if ui
                                                            .selectable_label(
                                                                self.selected_suggestion
                                                                    == Some(index),
                                                                RichText::new(
                                                                    suggestion
                                                                        .word
                                                                        .to_ascii_uppercase(),
                                                                )
                                                                .strong()
                                                                .color(Color32::from_rgb(
                                                                    42, 49, 43,
                                                                )),
                                                            )
                                                            .clicked()
                                                        {
                                                            self.selected_suggestion = Some(index);
                                                        }
                                                        ui.label(suggestion_method(suggestion));
                                                        ui.label(format!(
                                                            "{:.3}",
                                                            suggestion.solve_probability
                                                        ));
                                                        ui.label(format!(
                                                            "{:.3}",
                                                            suggestion.entropy
                                                        ));
                                                        ui.label(format!(
                                                            "{:.2}",
                                                            suggestion.expected_remaining
                                                        ));
                                                        ui.label(
                                                            suggestion
                                                                .worst_non_green_bucket_size
                                                                .to_string(),
                                                        );
                                                        ui.label(format_suggestion_cost(
                                                            suggestion,
                                                        ));
                                                        ui.end_row();
                                                    }
                                                });
                                        });
                                    ui.separator();
                                    show_suggestion_inspector(
                                        ui,
                                        self.selected_suggestion.and_then(|index| {
                                            self.predictive_suggestions.get(index)
                                        }),
                                        self.predictive_artifact_state,
                                        self.predictive_recovery_mode,
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
            });
    }
}

fn spawn_worker(
    predictive_solver: Solver,
    formal_solver: Option<FormalPolicyRuntime>,
) -> (LatestWorkerDispatcher, Receiver<WorkerResponse>) {
    const WORKER_COUNT: usize = 2;
    let shared = Arc::new((Mutex::new(LatestWorkerQueue::default()), Condvar::new()));
    let (response_sender, response_receiver) = mpsc::channel::<WorkerResponse>();
    for worker_index in 0..WORKER_COUNT {
        let shared = Arc::clone(&shared);
        let response_sender = response_sender.clone();
        let predictive_solver = predictive_solver.clone();
        let formal_solver = formal_solver.clone();
        thread::Builder::new()
            .name(format!("maybe-wordle-solver-{worker_index}"))
            .stack_size(SOLVER_THREAD_STACK_BYTES)
            .spawn(move || {
                loop {
                    let request = {
                        let (lock, ready) = &*shared;
                        let mut queue = match lock.lock() {
                            Ok(queue) => queue,
                            Err(_) => return,
                        };
                        while queue.pending.is_none() && !queue.shutdown {
                            queue = match ready.wait(queue) {
                                Ok(queue) => queue,
                                Err(_) => return,
                            };
                        }
                        if queue.shutdown {
                            return;
                        }
                        queue.pending.take().expect("pending request")
                    };
                    let payload = compute_worker_payload(
                        &predictive_solver,
                        formal_solver.as_ref(),
                        &request,
                    )
                    .map_err(|error| error.to_string());
                    if response_sender
                        .send(WorkerResponse {
                            generation: request.generation,
                            payload,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            })
            .expect("failed to start GUI solver worker");
    }
    drop(response_sender);
    (LatestWorkerDispatcher { shared }, response_receiver)
}

fn compute_worker_payload(
    predictive_solver: &Solver,
    formal_solver: Option<&FormalPolicyRuntime>,
    request: &WorkerRequest,
) -> Result<WorkerPayload> {
    match request.mode {
        GuiSolverMode::Predictive => {
            let date = NaiveDate::parse_from_str(&request.date_text, "%Y-%m-%d")
                .with_context(|| format!("invalid date: {}", request.date_text))?;
            let response = predictive_solver.suggest_predictive(PredictiveSuggestRequest {
                as_of: date,
                observations: &request.observations,
                top: request.top,
                hard_mode: request.hard_mode,
                force_in_two_only: request.force_in_two_only,
                mode: PredictiveSuggestionMode::FastDiskOnly,
            })?;
            let artifact_state = response.artifact_state;
            let model_metadata = format!(
                "Predictive model: {}\nConfig identity: {}\nHistory snapshot: {} ({})\nCached promotion: {}",
                response.model_version,
                response.model_manifest_hash,
                response
                    .history_snapshot_date
                    .map(|date| date.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                response.history_snapshot_hash,
                response.promotion_source.is_some()
            );
            Ok(WorkerPayload::Predictive {
                state: response.state,
                suggestions: response.suggestions,
                candidates: response.candidates,
                artifact_state,
                model_metadata,
            })
        }
        GuiSolverMode::Absurdle => {
            let state = predictive_solver.absurdle_apply_history(&request.observations)?;
            let suggestions =
                predictive_solver.absurdle_suggestions(&request.observations, request.top)?;
            Ok(WorkerPayload::Absurdle { state, suggestions })
        }
        GuiSolverMode::FormalOptimal => {
            let runtime = formal_solver
                .ok_or_else(|| anyhow::anyhow!("formal-optimal artifacts are not available"))?;
            let state = runtime.apply_history(&request.observations)?;
            let explanation = runtime.explain_state(&state, request.top)?;
            let suggestions = runtime.suggest(&state, request.top)?;
            Ok(WorkerPayload::Formal {
                explanation,
                suggestions,
            })
        }
    }
}

fn worker_response_is_current(latest_generation: u64, response_generation: u64) -> bool {
    latest_generation == response_generation
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
    status: &str,
    recovery: Option<RecoveryMode>,
    suggestion_count: usize,
) -> GuiSurfaceState {
    if missing_data {
        GuiSurfaceState::MissingData
    } else if computing {
        GuiSurfaceState::Loading
    } else if !status.is_empty() {
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
    visuals.selection.stroke = egui::Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, ink);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, ink);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, ink);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.5, ink);
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
    if suggestion.exact_cost.is_some() {
        "exact"
    } else if suggestion.lookahead_cost.is_some() {
        "lookahead"
    } else {
        "proxy"
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

fn show_suggestion_inspector(
    ui: &mut egui::Ui,
    suggestion: Option<&Suggestion>,
    artifact_state: PredictiveArtifactState,
    recovery_mode: Option<RecoveryMode>,
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
    ui.label(format!(
        "{} ranking · solve probability {:.4} · entropy {:.4} bits · expected remaining {:.2}",
        suggestion_method(suggestion),
        suggestion.solve_probability,
        suggestion.entropy,
        suggestion.expected_remaining
    ));
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
        predictive_banner_text(artifact_state),
        recovery_mode.map_or("not active", RecoveryMode::label)
    ));
}

fn show_candidate_browser(
    ui: &mut egui::Ui,
    filter: &mut String,
    candidates: &[PredictiveCandidateSummary],
) {
    ui.heading("Candidate Browser");
    ui.horizontal(|ui| {
        ui.add_sized(
            [180.0, 24.0],
            egui::TextEdit::singleline(filter).hint_text("Filter answers"),
        );
        if ui.button("Copy CSV").clicked() {
            let normalized = filter.trim().to_ascii_lowercase();
            let mut csv = String::from("word,probability,modeled_weight,fallback_support\n");
            for candidate in candidates
                .iter()
                .filter(|candidate| normalized.is_empty() || candidate.word.contains(&normalized))
            {
                csv.push_str(&format!(
                    "{},{:.12},{:.12},{}\n",
                    candidate.word,
                    candidate.probability,
                    candidate.modeled_weight,
                    candidate.fallback_support
                ));
            }
            ui.ctx().copy_text(csv);
        }
    });
    let normalized = filter.trim().to_ascii_lowercase();
    let matching = candidates
        .iter()
        .filter(|candidate| normalized.is_empty() || candidate.word.contains(&normalized))
        .collect::<Vec<_>>();
    ui.label(
        RichText::new(format!(
            "{} matching / {} live candidates",
            matching.len(),
            candidates.len()
        ))
        .small()
        .color(Color32::from_rgb(92, 72, 54)),
    );
    egui::ScrollArea::vertical()
        .id_salt("predictive-candidate-scroll")
        .max_height(150.0)
        .show(ui, |ui| {
            egui::Grid::new("candidate-browser-grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Word");
                    ui.label("Probability");
                    ui.label("Support");
                    ui.end_row();
                    for candidate in matching.into_iter().take(100) {
                        ui.label(RichText::new(candidate.word.to_ascii_uppercase()).strong());
                        ui.label(format!("{:.5}%", candidate.probability * 100.0));
                        ui.label(if candidate.fallback_support {
                            "fallback"
                        } else {
                            "modeled"
                        });
                        ui.end_row();
                    }
                });
        });
}

fn show_game_board(
    ui: &mut egui::Ui,
    observations: &[(String, u8)],
    current_guess: &str,
    current_feedback: [u8; 5],
) {
    for row in 0..6 {
        ui.horizontal(|ui| {
            let (letters, feedback, applied) = if let Some((guess, pattern)) = observations.get(row)
            {
                (
                    guess.chars().collect::<Vec<_>>(),
                    decode_pattern(pattern),
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
                ui.add_sized(
                    [48.0, 48.0],
                    egui::Button::new(
                        RichText::new(format!("{letter}\n{marker}"))
                            .size(15.0)
                            .strong()
                            .color(if applied {
                                Color32::WHITE
                            } else {
                                Color32::from_rgb(42, 49, 43)
                            }),
                    )
                    .fill(color),
                );
            }
        });
        ui.add_space(4.0);
    }
}

fn show_history_timeline(ui: &mut egui::Ui, observations: &[(String, u8)]) {
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
            let code = feedback_code(decode_pattern(pattern));
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

fn feedback_accessible_label(value: u8) -> &'static str {
    match value {
        0 => "Absent",
        1 => "Present",
        _ => "Correct",
    }
}

fn feedback_marker(value: u8) -> &'static str {
    match value {
        0 => "×",
        1 => "●",
        _ => "✓",
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
    use std::sync::{Arc, Condvar, Mutex};

    use crate::predictive::PredictiveArtifactState;

    use super::{
        BoardAction, BoardDraft, GuiSolverMode, GuiSurfaceState, LatestWorkerDispatcher,
        LatestWorkerQueue, WorkerRequest, feedback_accessible_label, formal_unavailable_text,
        gui_surface_state, is_compact_layout, predictive_banner_text, predictive_compute_status,
        predictive_reply_book_text, reduce_board_draft, worker_response_is_current,
    };

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
        let coded = reduce_board_draft(
            &correct,
            BoardAction::ReplaceFeedbackCode("byg20".to_string()),
        );
        assert_eq!(coded.feedback, [0, 1, 2, 2, 0]);
        assert_eq!(feedback_accessible_label(2), "Correct");
        assert_eq!(reduce_board_draft(&coded, BoardAction::Reset), initial);
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
        };
        dispatcher.send(request(4)).expect("first request");
        dispatcher.send(request(5)).expect("replacement request");
        let queue = shared.0.lock().expect("queue");
        assert_eq!(
            queue.pending.as_ref().map(|request| request.generation),
            Some(5)
        );
    }

    #[test]
    fn rendered_surface_states_cover_setup_loading_normal_recovery_and_error() {
        assert_eq!(
            gui_surface_state(true, false, "", None, 0),
            GuiSurfaceState::MissingData
        );
        assert_eq!(
            gui_surface_state(false, true, "", None, 0),
            GuiSurfaceState::Loading
        );
        assert_eq!(
            gui_surface_state(false, false, "", None, 0),
            GuiSurfaceState::Empty
        );
        assert_eq!(
            gui_surface_state(false, false, "", None, 3),
            GuiSurfaceState::Normal
        );
        assert_eq!(
            gui_surface_state(
                false,
                false,
                "",
                Some(crate::predictive::RecoveryMode::EpsilonRepair),
                3,
            ),
            GuiSurfaceState::Recovery
        );
        assert_eq!(
            gui_surface_state(false, false, "bad model", None, 0),
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
    fn worker_response_discards_stale_generations() {
        assert!(worker_response_is_current(7, 7));
        assert!(!worker_response_is_current(7, 6));
        assert!(!worker_response_is_current(7, 8));
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
    fn formal_unavailable_text_is_actionable() {
        assert_eq!(
            formal_unavailable_text(),
            "Formal artifacts missing; run build-optimal-policy first."
        );
    }
}
