use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum HistoryCoverage {
    NotRequested,
    Complete,
    Limited {
        cutoff: NaiveDate,
        expected_days: usize,
        covered_days: usize,
    },
}

impl HistoryCoverage {
    pub(super) fn through_cutoff(
        history: &[crate::data::NytDailyEntry],
        cutoff: NaiveDate,
    ) -> Self {
        let launch = NaiveDate::parse_from_str(crate::data::WORDLE_LAUNCH_DATE, "%Y-%m-%d")
            .expect("valid Wordle launch date constant");
        let expected_days = ((cutoff - launch).num_days() + 1).max(0) as usize;
        let covered_days = history
            .iter()
            .map(|entry| entry.print_date)
            .filter(|date| *date >= launch && *date <= cutoff)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        if covered_days == expected_days {
            Self::Complete
        } else {
            Self::Limited {
                cutoff,
                expected_days,
                covered_days,
            }
        }
    }

    pub(super) fn notice(&self) -> Option<String> {
        match self {
            Self::Limited {
                cutoff,
                expected_days,
                covered_days,
            } => Some(format!(
                "Predictive history is incomplete through {cutoff}: {covered_days}/{expected_days} days available. Seed-supported play remains available; run sync-data to update local history."
            )),
            Self::NotRequested | Self::Complete => None,
        }
    }

    fn result_status(&self, status: GuiStatus) -> GuiStatus {
        match (self.notice(), status) {
            (Some(notice), GuiStatus::Ready) => GuiStatus::Notice(notice),
            (Some(notice), GuiStatus::Notice(message)) => {
                GuiStatus::Notice(format!("{message} {notice}"))
            }
            (_, status) => status,
        }
    }
}

pub(super) struct LoadedWorkspace {
    pub(super) config: PriorConfig,
    pub(super) predictive_solver: Solver,
    pub(super) formal_solver: Option<FormalPolicyRuntime>,
}

pub(super) fn load_workspace(paths: &ProjectPaths) -> Result<LoadedWorkspace> {
    paths.ensure_layout()?;
    let config = PriorConfig::load_or_create(&paths.config_prior)?;
    let predictive_solver = Solver::from_paths(paths, &config)?;
    Ok(LoadedWorkspace {
        config,
        predictive_solver,
        formal_solver: None,
    })
}

pub(super) enum MaybeWordleApp {
    Ready(Box<WordleGuiApp>),
    Loading(Box<SetupApp>),
}

impl MaybeWordleApp {
    pub(super) fn load_or_setup(paths: ProjectPaths) -> Self {
        let mut app = SetupApp::new(paths, String::new());
        app.start(SetupAction::RetryLoad);
        Self::Loading(Box::new(app))
    }
}

impl eframe::App for MaybeWordleApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        match self {
            Self::Ready(app) => app.update(ctx, frame),
            Self::Loading(app) => {
                if let Some(workspace) = app.update(ctx) {
                    let mut ready = WordleGuiApp::new(workspace);
                    ready.start_formal_load(app.paths.clone());
                    *self = Self::Ready(Box::new(ready));
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SetupAction {
    SyncAndBuild,
    BuildLocal,
    RetryLoad,
}

impl SetupAction {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::SyncAndBuild => "Sync history & build",
            Self::BuildLocal => "Build from local data",
            Self::RetryLoad => "Retry existing model",
        }
    }
}

pub(super) enum SetupEvent {
    Progress(String),
    Complete(Box<std::result::Result<LoadedWorkspace, String>>),
}

pub(super) struct SetupApp {
    pub(super) paths: ProjectPaths,
    pub(super) error: String,
    pub(super) progress: String,
    pub(super) running: bool,
    pub(super) cancel_requested: Arc<AtomicBool>,
    pub(super) receiver: Option<Receiver<SetupEvent>>,
}

impl SetupApp {
    pub(super) fn new(paths: ProjectPaths, error: String) -> Self {
        Self {
            paths,
            error,
            progress: "Model data is not ready yet.".to_string(),
            running: false,
            cancel_requested: Arc::new(AtomicBool::new(false)),
            receiver: None,
        }
    }

    pub(super) fn start(&mut self, action: SetupAction) {
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
                            let summary = sync_nyt_history_cancellable(&paths, &config, Solver::today(), &|| cancelled.load(Ordering::Acquire))?;
                            if summary.cancelled || cancelled.load(Ordering::Acquire) {
                                anyhow::bail!("setup cancelled after history sync");
                            }
                            let _ = sender.send(SetupEvent::Progress(format!(
                                "Sync: {} attempted, {} fetched, {} applied, {} retained; {} failed and {} missing dates. Coverage {}. Building local model…",
                                summary.attempted, summary.fetched, summary.applied, summary.retained,
                                summary.failures.len(), summary.missing_dates.len(),
                                if summary.coverage_complete { "complete" } else { "incomplete" },
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
                    let workspace = load_workspace(&paths)?;
                    if cancelled.load(Ordering::Acquire) { anyhow::bail!("setup cancelled"); }
                    Ok(workspace)
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

    pub(super) fn update(&mut self, ctx: &egui::Context) -> Option<LoadedWorkspace> {
        ctx.set_visuals(workspace_visuals());
        let mut completed = None;
        if let Some(receiver) = &self.receiver {
            loop {
                let event = match receiver.try_recv() {
                    Ok(event) => event,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if self.running {
                            self.running = false;
                            self.error =
                                "Setup worker disconnected. Choose an action to retry.".to_string();
                        }
                        break;
                    }
                };
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

impl Drop for SetupApp {
    fn drop(&mut self) {
        self.cancel_requested.store(true, Ordering::Release);
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(super) enum FormalAvailability {
    Absent,
    Loading,
    Ready,
    Unavailable(String),
}

#[derive(Default)]
pub(super) struct LatestWorkerQueue {
    pub(super) pending: Option<WorkerRequest>,
    pub(super) latest_generation: u64,
    pub(super) shutdown: bool,
    #[cfg(test)]
    pub(super) test_cancel_hook: Option<Arc<dyn Fn(u64) + Send + Sync>>,
}

pub(super) struct LatestWorkerDispatcher {
    pub(super) shared: Arc<(Mutex<LatestWorkerQueue>, Condvar)>,
}

impl LatestWorkerDispatcher {
    pub(super) fn cancel(&self, generation: u64) {
        let (lock, ready) = &*self.shared;
        if let Ok(mut queue) = lock.lock() {
            queue.latest_generation = generation;
            queue.pending = None;
            ready.notify_all();
        }
    }

    pub(super) fn send(&self, request: WorkerRequest) -> Result<()> {
        let (lock, ready) = &*self.shared;
        let mut queue = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("suggestion worker queue is poisoned"))?;
        if queue.shutdown {
            anyhow::bail!("suggestion workers have stopped");
        }
        queue.latest_generation = request.generation;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WorkerRequest {
    pub(super) generation: u64,
    pub(super) mode: GuiSolverMode,
    pub(super) date_text: String,
    pub(super) observations: Vec<(String, u8)>,
    pub(super) top: usize,
    pub(super) force_in_two_only: bool,
    pub(super) hard_mode: bool,
    pub(super) search_profile: GuiSearchProfile,
}

#[derive(Clone, Debug)]
pub(super) enum WorkerPayload {
    Predictive {
        state: PredictiveStateSummary,
        suggestions: Vec<Suggestion>,
        candidates: Vec<PredictiveCandidateSummary>,
        artifact_state: PredictiveArtifactState,
        model_metadata: String,
        finite_search: Option<FiniteSearchResult>,
        execution: SearchExecution,
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
pub(super) struct WorkerResponse {
    pub(super) request: WorkerRequest,
    pub(super) payload: std::result::Result<WorkerPayload, String>,
}

impl WordleGuiApp {
    pub(super) fn schedule_recompute(&mut self) {
        self.latest_generation = self.latest_generation.wrapping_add(1);
        self.clear_result();
        self.history_coverage = if self.mode == GuiSolverMode::Predictive {
            match NaiveDate::parse_from_str(&self.date_text, "%Y-%m-%d")
                .map_err(anyhow::Error::from)
                .and_then(crate::predictive::history_cutoff)
            {
                Ok(cutoff) => {
                    HistoryCoverage::through_cutoff(&self.predictive_solver.history_dates, cutoff)
                }
                // The request parser owns the visible invalid-date error.
                Err(_) => HistoryCoverage::NotRequested,
            }
        } else {
            HistoryCoverage::NotRequested
        };
        if let Ok(status) = game::status(&self.observations, self.rules())
            && status != GameStatus::Active
        {
            self.current_request = None;
            self.computing = false;
            self.status = GuiStatus::Notice(status.label().to_string());
            if let Some(sender) = &self.request_sender {
                sender.cancel(self.latest_generation);
            }
            return;
        }
        let request = WorkerRequest {
            generation: self.latest_generation,
            mode: self.mode,
            date_text: self.date_text.clone(),
            observations: self.observations.clone(),
            top: self.top,
            force_in_two_only: self.force_in_two_only,
            hard_mode: self.hard_mode,
            search_profile: self.search_profile,
        };
        self.computing = true;
        self.status = GuiStatus::Loading(if self.mode == GuiSolverMode::Predictive {
            if let Some(options) = self.search_profile.finite_options(&self.predictive_solver) {
                predictive_finite_compute_status(self.search_profile, options)
            } else {
                predictive_compute_status(self.predictive_artifact_state)
            }
        } else {
            "Computing...".to_string()
        });
        self.current_request = Some(request.clone());
        if let Err(error) = self
            .request_sender
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Suggestion worker is unavailable. Retry the worker."))
            .and_then(|sender| sender.send(request))
        {
            self.request_sender = None;
            self.set_error(error);
        }
    }

    pub(super) fn drain_worker_responses(&mut self) {
        loop {
            let response = match self.response_receiver.try_recv() {
                Ok(response) => response,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.request_sender.take().is_some() {
                        self.set_error(anyhow::anyhow!(
                            "Suggestion worker disconnected. Retry the worker."
                        ));
                    }
                    break;
                }
            };
            if self.current_request.as_ref() != Some(&response.request) {
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
                    finite_search,
                    execution,
                }) => {
                    self.surviving_count = state.surviving;
                    self.total_weight = state.effective_total_weight;
                    self.predictive_recovery_mode = state.recovery_mode_used;
                    self.predictive_artifact_state = artifact_state;
                    self.predictive_model_metadata = model_metadata;
                    self.predictive_finite_search = finite_search;
                    self.predictive_execution = Some(execution);
                    self.predictive_suggestions = suggestions;
                    self.predictive_candidates = candidates;
                    self.selected_suggestion = None;
                    self.absurdle_suggestions.clear();
                    self.formal_suggestions.clear();
                    self.formal_explanation = None;
                    self.status = self
                        .history_coverage
                        .result_status(predictive_response_status(
                            self.predictive_finite_search.as_ref(),
                        ));
                }
                Ok(WorkerPayload::Absurdle { state, suggestions }) => {
                    self.surviving_count = state.surviving.len();
                    self.total_weight = 0.0;
                    self.predictive_recovery_mode = None;
                    self.predictive_artifact_state =
                        PredictiveArtifactState::NoPredictiveArtifactAvailable;
                    self.predictive_finite_search = None;
                    self.predictive_execution = None;
                    self.absurdle_suggestions = suggestions;
                    self.predictive_suggestions.clear();
                    self.predictive_candidates.clear();
                    self.formal_suggestions.clear();
                    self.formal_explanation = None;
                    self.status = GuiStatus::Ready;
                }
                Ok(WorkerPayload::Formal {
                    explanation,
                    suggestions,
                }) => {
                    let alternatives_limited = explanation.alternatives_status
                        == FormalAlternativesStatus::WorkLimitReached;
                    self.surviving_count = explanation.surviving_answers;
                    self.total_weight = 0.0;
                    self.predictive_recovery_mode = None;
                    self.predictive_artifact_state =
                        PredictiveArtifactState::NoPredictiveArtifactAvailable;
                    self.predictive_finite_search = None;
                    self.predictive_execution = None;
                    self.formal_explanation = Some(explanation);
                    self.formal_suggestions = suggestions;
                    self.predictive_suggestions.clear();
                    self.predictive_candidates.clear();
                    self.absurdle_suggestions.clear();
                    self.status = if alternatives_limited {
                        GuiStatus::Notice("Showing stored primary policy only; alternative ranking reached its work limit.".into())
                    } else {
                        GuiStatus::Ready
                    };
                }
                Err(error) => self.set_error(anyhow::anyhow!(error)),
            }
        }
    }

    pub(super) fn set_error(&mut self, error: anyhow::Error) {
        self.status = GuiStatus::WorkerError(format!("{error:#}"));
        self.clear_result();
        self.computing = false;
    }

    pub(super) fn retry_worker(&mut self) {
        self.request_sender = None;
        let worker = spawn_worker(self.predictive_solver.clone(), self.formal_solver.clone());
        self.install_worker(worker);
    }

    pub(super) fn install_worker(
        &mut self,
        worker: Result<(LatestWorkerDispatcher, Receiver<WorkerResponse>)>,
    ) {
        self.request_sender = None;
        match worker {
            Ok((sender, receiver)) => {
                self.request_sender = Some(sender);
                self.response_receiver = receiver;
                self.schedule_recompute();
            }
            Err(error) => self.set_error(error),
        }
    }

    pub(super) fn start_formal_load(&mut self, paths: ProjectPaths) {
        self.formal_availability = FormalAvailability::Loading;
        let (sender, receiver) = mpsc::channel();
        self.formal_receiver = Some(receiver);
        let cancelled = Arc::clone(&self.formal_cancel);
        if let Err(error) = thread::Builder::new()
            .name("maybe-wordle-formal-load".into())
            .stack_size(SOLVER_THREAD_STACK_BYTES)
            .spawn(move || {
                if cancelled.load(Ordering::Acquire) {
                    return;
                }
                let result = if artifacts_exist(&paths, DEFAULT_FORMAL_MODEL_ID) {
                    FormalPolicyRuntime::load(&paths, DEFAULT_FORMAL_MODEL_ID).map(Some)
                } else {
                    Ok(None)
                }
                .map_err(|error| format!("{error:#}"));
                if !cancelled.load(Ordering::Acquire) {
                    let _ = sender.send(result);
                }
            })
        {
            self.formal_receiver = None;
            self.formal_availability =
                FormalAvailability::Unavailable(format!("Formal worker could not start: {error}"));
        }
    }

    pub(super) fn drain_formal_load(&mut self) {
        let Some(receiver) = &self.formal_receiver else {
            return;
        };
        let result = match receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Formal loading worker disconnected".into())
            }
        };
        self.formal_receiver = None;
        match result {
            Ok(Some(runtime)) => {
                self.formal_solver = Some(runtime);
                self.formal_availability = FormalAvailability::Ready;
                self.retry_worker();
            }
            Ok(None) => self.formal_availability = FormalAvailability::Absent,
            Err(error) => self.formal_availability = FormalAvailability::Unavailable(error),
        }
    }
}

impl Drop for WordleGuiApp {
    fn drop(&mut self) {
        self.formal_cancel.store(true, Ordering::Release);
    }
}

pub(super) fn spawn_worker(
    predictive_solver: Solver,
    formal_solver: Option<FormalPolicyRuntime>,
) -> Result<(LatestWorkerDispatcher, Receiver<WorkerResponse>)> {
    let shared = Arc::new((Mutex::new(LatestWorkerQueue::default()), Condvar::new()));
    let worker_shared = Arc::clone(&shared);
    let (response_sender, response_receiver) = mpsc::channel::<WorkerResponse>();
    thread::Builder::new()
        .name("maybe-wordle-solver".to_string())
        .stack_size(SOLVER_THREAD_STACK_BYTES)
        .spawn(move || {
            loop {
                let request = {
                    let (lock, ready) = &*worker_shared;
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
                let cancelled = || worker_request_cancelled(&worker_shared, request.generation);
                let payload = compute_worker_payload(
                    &predictive_solver,
                    formal_solver.as_ref(),
                    &request,
                    &cancelled,
                )
                .map_err(|error| format!("{error:#}"));
                if response_sender
                    .send(WorkerResponse { request, payload })
                    .is_err()
                {
                    return;
                }
            }
        })
        .context("failed to start GUI solver worker")?;
    Ok((LatestWorkerDispatcher { shared }, response_receiver))
}

pub(super) fn compute_worker_payload(
    predictive_solver: &Solver,
    formal_solver: Option<&FormalPolicyRuntime>,
    request: &WorkerRequest,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<WorkerPayload> {
    match request.mode {
        GuiSolverMode::Predictive => {
            let date = NaiveDate::parse_from_str(&request.date_text, "%Y-%m-%d")
                .with_context(|| format!("invalid date: {}", request.date_text))?;
            let search_profile = request.search_profile;
            let predictive_request = PredictiveSuggestRequest {
                puzzle_date: date,
                observations: &request.observations,
                top: request.top,
                hard_mode: request.hard_mode,
                force_in_two_only: request.force_in_two_only,
                mode: PredictiveSuggestionMode::FastDiskOnly,
            };
            let response = if let Some(options) = search_profile.finite_options(predictive_solver) {
                predictive_solver.suggest_predictive_controlled(
                    predictive_request,
                    options,
                    cancelled,
                )?
            } else {
                predictive_solver.suggest_predictive_cancellable(predictive_request, cancelled)?
            };
            let artifact_state = response.artifact_state;
            let mut model_metadata = format!(
                "Puzzle date: {} (history through {})\nPredictive model: {}\nConfig identity: {}\nHistory snapshot: {} ({})\nCached promotion: {}",
                response.puzzle_date,
                response.history_cutoff,
                response.model_version,
                response.model_manifest_hash,
                response
                    .history_snapshot_date
                    .map(|date| date.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                response.history_snapshot_hash,
                response.promotion_source.is_some()
            );
            if let Some(date) = response.promoted_artifact_date {
                model_metadata.push_str(&format!("\nPromoted artifact snapshot: {date}"));
            }
            if response.execution.action_scope == SearchActionScope::HardRootNormalContinuation {
                model_metadata.push_str("\nLegacy costs assume normal-mode future replies; finite profiles enforce hard-mode legality recursively.");
            }
            Ok(WorkerPayload::Predictive {
                state: response.state,
                suggestions: response.suggestions,
                candidates: response.candidates,
                artifact_state,
                model_metadata,
                finite_search: response.finite_search,
                execution: response.execution,
            })
        }
        GuiSolverMode::Absurdle => {
            let state = predictive_solver.absurdle_apply_history(&request.observations)?;
            let suggestions = predictive_solver.absurdle_suggestions_cancellable(
                &request.observations,
                request.top,
                cancelled,
            )?;
            Ok(WorkerPayload::Absurdle { state, suggestions })
        }
        GuiSolverMode::FormalOptimal => {
            let runtime = formal_solver
                .ok_or_else(|| anyhow::anyhow!("formal-optimal artifacts are not available"))?;
            let state = runtime.apply_history(&request.observations)?;
            let explanation = runtime.explain_state_cancellable(
                &state,
                request.top,
                DEFAULT_FORMAL_ALTERNATIVE_PARTITIONS,
                cancelled,
            )?;
            let suggestions = explanation.tied_moves.clone();
            Ok(WorkerPayload::Formal {
                explanation,
                suggestions,
            })
        }
    }
}

pub(super) fn worker_request_cancelled(
    shared: &Arc<(Mutex<LatestWorkerQueue>, Condvar)>,
    generation: u64,
) -> bool {
    let (lock, _) = &**shared;
    #[cfg(test)]
    let test_cancel_hook = match lock.lock() {
        Ok(queue) => queue.test_cancel_hook.clone(),
        Err(_) => return true,
    };
    #[cfg(test)]
    if let Some(hook) = test_cancel_hook {
        hook(generation);
    }
    match lock.lock() {
        Ok(queue) => queue.shutdown || queue.latest_generation != generation,
        Err(_) => true,
    }
}
