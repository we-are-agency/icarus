use nannou::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, OnceLock};

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
const MENU_BAR_H: f32 = 56.0;
const PANE_MARGIN: f32 = 28.0;
const PANE_GUTTER: f32 = 20.0;
const PANE_TEXT_INSET: f32 = 14.0;
const PANE_LINE_H: f32 = 12.0;
const SPEC_RATIO: f32 = 0.38;
const GRID_STEP: f32 = 36.0;
const RENDER_VIEW_Y_OFFSET: f32 = 10.0;
const STARTUP_MODEL: &str = "gpt-5.4";
const MAX_TRACE_CYCLE_ENTITIES: usize = 256;
const MAX_LIVE_TRACE_POINTS: usize = 4096;
const MAX_LLM_HISTORY_MESSAGES: usize = 150;
const GRAVITY_ACCEL_Y: f32 = 0.18;
const RELAXATION_ITERATIONS: usize = 96;
const RELAXATION_TOLERANCE: f32 = 0.001;
const RELAXATION_DAMPING: f32 = 0.98;
const RELAXATION_DIVERGENCE_LIMIT: f32 = 10_000.0;
const SCREEN_LOG_PATH: &str = "linkage.log";
const POI_TRACE_DEBUG_PATH: &str = "poi.json";
const POI_TRACE_DEBUG_SAMPLES: usize = 20;
const MAX_TOOL_CORRECTION_ATTEMPTS: usize = 40;

static CONFIG: OnceLock<Config> = OnceLock::new();
static SCREEN_LOG_LAST_SNAPSHOT: OnceLock<Mutex<String>> = OnceLock::new();
static POI_TRACE_LAST_SNAPSHOT: OnceLock<Mutex<String>> = OnceLock::new();
static NEXT_ARTIFACT_TURN: AtomicU32 = AtomicU32::new(1);
#[derive(Clone, Debug)]
struct Config {
    headless: bool,
    capture_path: Option<PathBuf>,
    fixture_index: usize,
}

struct Model {
    headless: bool,
    capture_path: Option<PathBuf>,
    fixtures: Vec<FixturePresentation>,
    selected_fixture: usize,
    playback_progress: f32,
    playback_paused: bool,
    live_trace_cycle: u32,
    live_trace_u: f32,
    active_artifact_turn: Option<u32>,
    live_simulation: Option<LiveSimulationState>,
    headless_capture_state: Cell<HeadlessCaptureState>,
    headless_capture_result: Arc<AtomicU8>,
    headless_proxy: Option<nannou::app::Proxy>,
    spec_scroll_px: f32,
    llm_turn_rx: Option<Receiver<LlmTurnResult>>,
    chat_entries: Vec<ChatEntry>,
    llm_history: Vec<LlmHistoryMessage>,
    chat_input: String,
    chat_input_focused: bool,
    latest_full_cycle_traces: BTreeMap<String, Vec<TraceSample>>,
    trace_cycle_entities: Vec<TraceCycleEntity>,
    emitted_trace_cycles: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum HeadlessCaptureState {
    Pending,
    Queued,
    Flushed,
}

struct FixturePresentation {
    label: String,
    status: FixtureStatus,
    assembly: Option<AssemblySpec>,
}

enum FixtureStatus {
    Solved(Arc<SweepArtifact>),
    ValidationError(String),
    RelaxationError(String),
    GenerationError(String),
}

struct LlmTurnResult {
    fixture: Option<FixturePresentation>,
    entries: Vec<ChatEntry>,
    history_messages: Vec<LlmHistoryMessage>,
    replace_chat: bool,
}

#[derive(Clone)]
struct ChatEntry {
    role: ChatRole,
    text: String,
}

#[derive(Clone, Copy)]
enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Error,
}

#[derive(Clone)]
struct LlmHistoryMessage {
    role: LlmHistoryRole,
    text: String,
}

#[derive(Clone, Copy)]
enum LlmHistoryRole {
    User,
    Tool,
    Assistant,
}

struct TraceCycleEntity {
    completion_progress: f32,
    color_index: usize,
    points: Vec<Point2>,
}

struct LiveSimulationState {
    particles: BTreeMap<String, ParticleState>,
    frame: SolvedFrame,
    point_trails: BTreeMap<String, Vec<Point2>>,
    cycle_samples: BTreeMap<String, Vec<TraceSample>>,
    max_constraint_error: f32,
    settled: bool,
}

#[derive(Clone, Copy)]
struct TraceSample {
    point: Point2,
    phase: f32,
}

struct ToolCall {
    call_id: String,
    arguments: String,
}

struct ToolExchange {
    arguments: String,
    response: String,
}

struct LlmTurnError {
    message: String,
    exchanges: Vec<ToolExchange>,
}

struct SweepArtifact {
    assembly: Arc<AssemblySpec>,
    frames: Vec<SolvedFrame>,
    telemetry: SweepTelemetry,
    turn: u32,
    reset_phase: bool,
    cycle_seconds: f32,
    playback: PlaybackTraversal,
}

struct SweepTelemetry {
    #[cfg_attr(not(test), allow(dead_code))]
    point_paths: BTreeMap<String, Vec<Point2>>,
    unsettled_samples: u32,
    #[cfg_attr(not(test), allow(dead_code))]
    peak_constraint_error: f32,
    notes: Vec<String>,
}

struct SolvedFrame {
    #[cfg_attr(not(test), allow(dead_code))]
    u: f32,
    joint_positions: BTreeMap<String, Point2>,
    drive_values: BTreeMap<String, f32>,
    poi_positions: BTreeMap<String, Point2>,
}

#[derive(Clone, Copy)]
enum PlaybackTraversal {
    Forward,
    PingPong,
}

#[derive(Clone, Copy)]
enum DriveParameter {
    Angle,
    SliderPosition,
}

struct DrivePlan {
    drive_id: String,
    #[cfg_attr(not(test), allow(dead_code))]
    parameter: DriveParameter,
    start_value: f32,
    end_value: f32,
    cycle_seconds: f32,
    playback: PlaybackTraversal,
}

#[derive(Clone)]
struct ParticleState {
    pos: Point2,
    prev_pos: Point2,
    fixed: bool,
    mass: f32,
}

struct LinkConstraint {
    a: String,
    b: String,
    length: f32,
}

#[derive(Clone)]
struct SliderConstraint {
    joint_id: String,
    axis_origin: Point2,
    axis_dir: Vec2,
    range: (f32, f32),
}

enum DriveConstraint {
    Angular {
        pivot_joint_id: String,
        tip_joint_id: String,
        angle: f32,
        length: f32,
    },
    Linear {
        joint_id: String,
        axis_origin: Point2,
        axis_dir: Vec2,
        value: f32,
    },
}

struct RelaxationResult {
    particles: BTreeMap<String, ParticleState>,
    max_constraint_error: f32,
    settled: bool,
}

struct SolvedAssembly {
    joints: BTreeMap<String, SolvedJoint>,
    links: Vec<SolvedLink>,
    sliders: Vec<SolvedSlider>,
    pois: Vec<SolvedPoi>,
}

struct SolvedJoint {
    position: Point2,
    fixed: bool,
}

struct SolvedLink {
    a: String,
    b: String,
}

struct SolvedSlider {
    start: Point2,
    end: Point2,
    joint: String,
}

struct SolvedPoi {
    position: Point2,
}

#[derive(Clone, Deserialize, Serialize)]
struct AssemblySpec {
    joints: BTreeMap<String, JointSpec>,
    parts: BTreeMap<String, PartSpec>,
    drives: BTreeMap<String, DriveSpec>,
    points_of_interest: Vec<PointOfInterestSpec>,
    visualization: Option<VisualizationSpec>,
    meta: AssemblyMeta,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
enum JointSpec {
    Fixed { position: [f32; 2] },
    Free,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
enum PartSpec {
    Link {
        a: String,
        b: String,
        length: f32,
    },
    Slider {
        joint: String,
        axis_origin: [f32; 2],
        axis_dir: [f32; 2],
        range: [f32; 2],
    },
}

#[derive(Clone, Deserialize, Serialize)]
struct DriveSpec {
    kind: DriveKindSpec,
    sweep: SweepSpec,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
enum DriveKindSpec {
    Angular {
        pivot_joint: String,
        tip_joint: String,
        link: String,
        range: Option<[f32; 2]>,
    },
    #[allow(dead_code)]
    Linear {
        slider: String,
        range: Option<[f32; 2]>,
    },
}

#[derive(Clone, Deserialize, Serialize)]
struct SweepSpec {
    samples: u32,
    direction: SweepDirectionSpec,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
enum SweepDirectionSpec {
    Forward,
    Reverse,
    PingPong,
    #[serde(alias = "CW")]
    Clockwise,
    #[serde(alias = "CCW")]
    CounterClockwise,
}

#[derive(Clone, Deserialize, Serialize)]
struct PointOfInterestSpec {
    id: String,
    host: String,
    t: f32,
    perp: f32,
}

#[derive(Clone, Deserialize, Serialize)]
struct VisualizationSpec {
    trace_model: Option<TraceModelSpec>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
enum TraceModelSpec {
    RollingPaper {
        direction: PaperDirectionSpec,
        advance_per_cycle: f32,
    },
}

#[derive(Clone, Deserialize, Serialize)]
enum PaperDirectionSpec {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Deserialize, Serialize)]
struct AssemblyMeta {
    name: String,
    iteration: u32,
    notes: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
struct SetAssemblyArgs {
    reasoning: String,
    assembly: AssemblySpec,
}

fn main() {
    dotenvy::dotenv().ok();
    let config = Config::from_args();
    CONFIG
        .set(config)
        .expect("linkage config should only be initialized once");

    nannou::app(model).update(update).run();
}

impl Config {
    fn from_args() -> Self {
        let mut headless = false;
        let mut capture_path = None;
        let mut fixture_index = 0;
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--headless" => headless = true,
                "--capture" => {
                    let Some(path) = args.next() else {
                        panic!("--capture requires a path argument");
                    };
                    capture_path = Some(PathBuf::from(path));
                }
                "--fixture" => {
                    let Some(value) = args.next() else {
                        panic!("--fixture requires an index");
                    };
                    let parsed = value
                        .parse::<usize>()
                        .expect("--fixture must be a one-based integer");
                    fixture_index = parsed.saturating_sub(1);
                }
                other => panic!("unknown argument: {other}"),
            }
        }

        if headless && capture_path.is_none() {
            capture_path = Some(PathBuf::from("target/linkage-p1-headless.png"));
        }

        Self {
            headless,
            capture_path,
            fixture_index,
        }
    }
}

fn model(app: &App) -> Model {
    let config = CONFIG
        .get()
        .expect("linkage config should be available before model()")
        .clone();
    let (fixtures, llm_turn_rx) =
        build_fixture_presentations().expect("failed to build fixture bank");
    let chat_entries = initial_chat_entries(&fixtures);
    let llm_history = initial_llm_history();

    let mut window = app
        .new_window()
        .title("Linkage")
        .size(WINDOW_W, WINDOW_H)
        .view(view)
        .key_pressed(key_pressed)
        .received_character(received_character)
        .mouse_pressed(mouse_pressed)
        .mouse_wheel(mouse_wheel);
    if config.headless {
        app.set_loop_mode(LoopMode::rate_fps(60.0));
        window = window.visible(false);
    }
    window.build().unwrap();

    Model {
        headless: config.headless,
        capture_path: config.capture_path,
        selected_fixture: config.fixture_index.min(fixtures.len().saturating_sub(1)),
        fixtures,
        playback_progress: 0.0,
        playback_paused: false,
        live_trace_cycle: 0,
        live_trace_u: 0.0,
        active_artifact_turn: None,
        live_simulation: None,
        headless_capture_state: Cell::new(HeadlessCaptureState::Pending),
        headless_capture_result: Arc::new(AtomicU8::new(0)),
        headless_proxy: if config.headless {
            Some(app.create_proxy())
        } else {
            None
        },
        spec_scroll_px: 0.0,
        llm_turn_rx,
        chat_entries,
        llm_history,
        chat_input: String::new(),
        chat_input_focused: !config.headless,
        latest_full_cycle_traces: BTreeMap::new(),
        trace_cycle_entities: Vec::new(),
        emitted_trace_cycles: 0,
    }
}

fn update(app: &App, model: &mut Model, update: Update) {
    if let Some(rx) = &model.llm_turn_rx {
        if let Ok(result) = rx.try_recv() {
            if result.replace_chat {
                model.chat_entries.clear();
                model.llm_history.clear();
            }
            model.chat_entries.extend(result.entries);
            model.llm_history.extend(result.history_messages);
            trim_llm_history(&mut model.llm_history);
            if let Some(fixture) = result.fixture {
                if !model.fixtures.is_empty() {
                    model.fixtures[0] = fixture;
                }
            }
            model.llm_turn_rx = None;
            model.spec_scroll_px = max_chat_scroll_px(model, app.window_rect());
        }
    }

    if let Some(artifact) = current_artifact(model).cloned() {
        if model.active_artifact_turn != Some(artifact.turn) {
            if artifact.reset_phase {
                model.playback_progress = 0.0;
            }
            model.active_artifact_turn = Some(artifact.turn);
            model.latest_full_cycle_traces.clear();
            model.trace_cycle_entities.clear();
            model.emitted_trace_cycles = model.playback_progress.floor().max(0.0) as u32;
            model.live_trace_cycle = model.emitted_trace_cycles;
            model.live_trace_u = playback_sample_u(model.playback_progress, artifact.playback);
            match seed_live_simulation_state(&artifact, model.playback_progress) {
                Ok(state) => model.live_simulation = Some(state),
                Err(err) => {
                    model.live_simulation = None;
                    current_fixture_mut(model).status = FixtureStatus::RelaxationError(err);
                    model.active_artifact_turn = None;
                    return;
                }
            }
        }

        if !model.playback_paused {
            let delta = update.since_last.as_secs_f32() / artifact.cycle_seconds.max(0.001);
            model.playback_progress += delta;
        }
        model.live_trace_u = playback_sample_u(model.playback_progress, artifact.playback);

        if let Some(state) = model.live_simulation.as_mut() {
            let completed_cycles = model.playback_progress.floor().max(0.0) as u32;
            if rolling_paper_config(&artifact).is_some() {
                while model.emitted_trace_cycles < completed_cycles {
                    model.latest_full_cycle_traces = state.cycle_samples.clone();
                    spawn_trace_cycle_entities_from_samples(
                        &mut model.trace_cycle_entities,
                        &state.cycle_samples,
                        &artifact,
                        model.emitted_trace_cycles,
                    );
                    model.emitted_trace_cycles += 1;
                }
            } else {
                model.emitted_trace_cycles = completed_cycles;
            }
            if completed_cycles != model.live_trace_cycle {
                model.live_trace_cycle = completed_cycles;
                reset_cycle_samples(state);
            }

            if let Err(err) = step_live_simulation_state(state, &artifact, model.playback_progress)
            {
                model.live_simulation = None;
                current_fixture_mut(model).status = FixtureStatus::RelaxationError(err);
                model.active_artifact_turn = None;
                return;
            }

            if rolling_paper_config(&artifact).is_some() {
                let (_, render_rect) = content_rects(app.window_rect());
                let local_rect = Rect::from_w_h(render_rect.w() - 28.0, render_rect.h() - 72.0);
                let map = make_world_to_local(local_rect, artifact_bounds(&artifact));
                let current_progress = model.playback_progress;
                let render_offset = vec2(render_rect.x(), render_rect.y() - RENDER_VIEW_Y_OFFSET);
                let window_rect = app.window_rect();
                model.trace_cycle_entities.retain(|entity| {
                    trace_cycle_entity_visible(
                        entity,
                        current_progress,
                        &artifact,
                        &map,
                        render_offset,
                        window_rect,
                    )
                });
            }
        }
    } else {
        model.active_artifact_turn = None;
        model.playback_progress = 0.0;
        model.live_trace_cycle = 0;
        model.live_trace_u = 0.0;
        model.live_simulation = None;
        model.latest_full_cycle_traces.clear();
        model.trace_cycle_entities.clear();
        model.emitted_trace_cycles = 0;
    }

    if model.headless && model.headless_capture_state.get() == HeadlessCaptureState::Queued {
        match model.headless_capture_result.load(Ordering::SeqCst) {
            1 => {
                model
                    .headless_capture_state
                    .set(HeadlessCaptureState::Flushed);
                app.quit();
            }
            2 => panic!("headless capture timed out before the PNG was fully written"),
            _ => {}
        }
    }
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    if model.fixtures.is_empty() {
        return;
    }
    if model.chat_input_focused {
        match key {
            Key::Return => {
                submit_chat_input(model);
            }
            Key::Back => {
                model.chat_input.pop();
            }
            Key::Escape => {
                model.chat_input_focused = false;
            }
            _ => {}
        }
        return;
    }
    if let Some(index) = fixture_key_index(key) {
        select_fixture(model, index);
        return;
    }

    match key {
        Key::S => {
            let path = model
                .capture_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("target/linkage-windowed-capture.png"));
            app.main_window().capture_frame(path);
        }
        Key::R => trigger_startup_generation(model),
        Key::Space => model.playback_paused = !model.playback_paused,
        Key::Tab => {
            if !model.fixtures.is_empty() {
                select_fixture(model, (model.selected_fixture + 1) % model.fixtures.len());
            }
        }
        Key::Up => scroll_spec(model, app.window_rect(), -PANE_LINE_H * 3.0),
        Key::Down => scroll_spec(model, app.window_rect(), PANE_LINE_H * 3.0),
        Key::PageUp => scroll_spec(
            model,
            app.window_rect(),
            -visible_chat_body_height(app.window_rect()) * 0.9,
        ),
        Key::PageDown => scroll_spec(
            model,
            app.window_rect(),
            visible_chat_body_height(app.window_rect()) * 0.9,
        ),
        Key::Home => model.spec_scroll_px = 0.0,
        Key::End => model.spec_scroll_px = max_chat_scroll_px(model, app.window_rect()),
        _ => {}
    }
}

fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    if button != MouseButton::Left {
        return;
    }
    if startup_regenerate_rect(app.window_rect()).contains(app.mouse.position()) {
        trigger_startup_generation(model);
        return;
    }
    let win = app.window_rect();
    if chat_send_rect(win).contains(app.mouse.position()) {
        submit_chat_input(model);
        return;
    }
    model.chat_input_focused = chat_input_rect(win).contains(app.mouse.position());
}

fn received_character(_app: &App, model: &mut Model, ch: char) {
    if !model.chat_input_focused || llm_turn_in_progress(model) {
        return;
    }
    if ch.is_control() {
        return;
    }
    model.chat_input.push(ch);
}

fn fixture_key_index(key: Key) -> Option<usize> {
    match key {
        Key::Key1 => Some(0),
        Key::Key2 => Some(1),
        Key::Key3 => Some(2),
        Key::Key4 => Some(3),
        Key::Key5 => Some(4),
        Key::Key6 => Some(5),
        Key::Key7 => Some(6),
        Key::Key8 => Some(7),
        Key::Key9 => Some(8),
        _ => None,
    }
}

fn llm_turn_in_progress(model: &Model) -> bool {
    model.llm_turn_rx.is_some()
}

fn trigger_startup_generation(model: &mut Model) {
    if llm_turn_in_progress(model) {
        return;
    }
    let prompt = if !model.chat_input.trim().is_empty() {
        model.chat_input.trim().to_string()
    } else if let Some(last_prompt) = model
        .llm_history
        .iter()
        .rev()
        .find(|message| matches!(message.role, LlmHistoryRole::User))
        .map(|message| message.text.clone())
    {
        last_prompt
    } else {
        model.chat_entries.push(ChatEntry {
            role: ChatRole::System,
            text: "Enter a prompt first. Regenerate reuses your latest prompt from a blank startup assembly.".to_string(),
        });
        model.spec_scroll_px = f32::INFINITY;
        model.chat_input_focused = true;
        return;
    };
    model.chat_input.clear();
    let mut history_messages = vec![LlmHistoryMessage {
        role: LlmHistoryRole::User,
        text: prompt,
    }];
    trim_llm_history(&mut history_messages);
    let (rx, entries, history_messages) =
        spawn_llm_turn(empty_startup_assembly(), history_messages, true);
    model.chat_entries = entries;
    model.llm_history = history_messages;
    model.llm_turn_rx = Some(rx);
    select_fixture(model, 0);
}

fn submit_chat_input(model: &mut Model) {
    if llm_turn_in_progress(model) {
        return;
    }
    let prompt = model.chat_input.trim();
    if prompt.is_empty() {
        return;
    }
    let prompt = prompt.to_string();
    let base_assembly = startup_fixture_assembly(model).unwrap_or_else(empty_startup_assembly);
    let poi_trace_context_json = poi_trace_context_json(model)
        .unwrap_or_else(|_| "{\"source\":\"unavailable\"}".to_string());
    let mut history = model.llm_history.clone();
    history.push(LlmHistoryMessage {
        role: LlmHistoryRole::User,
        text: prompt.clone(),
    });
    trim_llm_history(&mut history);
    model.chat_entries.push(ChatEntry {
        role: ChatRole::User,
        text: prompt.clone(),
    });
    model.chat_entries.push(ChatEntry {
        role: ChatRole::System,
        text: format!("calling {}…", startup_model_name()),
    });
    model.spec_scroll_px = f32::INFINITY;
    model.chat_input.clear();
    model.llm_history = history.clone();
    model.llm_turn_rx = Some(spawn_chat_turn(
        base_assembly,
        history,
        poi_trace_context_json,
    ));
    select_fixture(model, 0);
}

fn mouse_wheel(app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    let win = app.window_rect();
    let (spec_rect, _) = content_rects(win);
    let mouse = app.mouse.position();
    if !spec_rect.contains(mouse) {
        return;
    }

    let scroll_delta = match delta {
        MouseScrollDelta::LineDelta(_, y) => -y * PANE_LINE_H * 3.0,
        MouseScrollDelta::PixelDelta(pos) => -(pos.y as f32),
    };
    scroll_spec(model, win, scroll_delta);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let win = app.window_rect();

    draw.background().color(BLACK);
    draw_scene(&draw, win, model);
    write_screen_log_snapshot(model, win);
    write_poi_trace_debug_json(model);

    if model.headless
        && model.headless_capture_state.get() == HeadlessCaptureState::Pending
        && headless_capture_ready(model)
    {
        if let Some(path) = &model.capture_path {
            app.main_window().capture_frame(path);
            let capture_path = path.clone();
            let capture_result = Arc::clone(&model.headless_capture_result);
            let proxy = model
                .headless_proxy
                .as_ref()
                .expect("headless mode should have an app proxy")
                .clone();
            std::thread::spawn(move || {
                let status = if wait_for_capture_flush(&capture_path) {
                    1
                } else {
                    2
                };
                capture_result.store(status, Ordering::SeqCst);
                let _ = proxy.wakeup();
            });
        }
        model
            .headless_capture_state
            .set(HeadlessCaptureState::Queued);
    }

    draw.to_frame(app, &frame).unwrap();
}

fn select_fixture(model: &mut Model, index: usize) {
    if !model.fixtures.is_empty() && index < model.fixtures.len() {
        model.selected_fixture = index;
        model.spec_scroll_px = 0.0;
    }
}

fn current_fixture(model: &Model) -> &FixturePresentation {
    model
        .fixtures
        .get(model.selected_fixture)
        .unwrap_or_else(|| {
            model
                .fixtures
                .first()
                .expect("fixture bank should not be empty")
        })
}

fn current_fixture_mut(model: &mut Model) -> &mut FixturePresentation {
    if model.selected_fixture >= model.fixtures.len() {
        model.selected_fixture = 0;
    }
    model
        .fixtures
        .get_mut(model.selected_fixture)
        .expect("fixture bank should not be empty")
}

fn current_artifact(model: &Model) -> Option<&Arc<SweepArtifact>> {
    match &current_fixture(model).status {
        FixtureStatus::GenerationError(_)
        | FixtureStatus::ValidationError(_)
        | FixtureStatus::RelaxationError(_) => None,
        FixtureStatus::Solved(artifact) => Some(artifact),
    }
}

fn headless_capture_ready(model: &Model) -> bool {
    match &current_fixture(model).status {
        FixtureStatus::Solved(_) => {
            model.live_simulation.is_some() && model.playback_progress >= 1.0
        }
        FixtureStatus::ValidationError(_)
        | FixtureStatus::RelaxationError(_)
        | FixtureStatus::GenerationError(_) => true,
    }
}

fn seed_live_simulation_state(
    artifact: &SweepArtifact,
    progress: f32,
) -> Result<LiveSimulationState, String> {
    let (drive_id, drive) = artifact
        .assembly
        .drives
        .iter()
        .next()
        .ok_or_else(|| "fixture missing drive".to_string())?;
    let plan = drive_plan(&artifact.assembly, drive_id, drive)?;
    let links = collect_link_constraints(&artifact.assembly)?;
    let sliders = collect_slider_constraints(&artifact.assembly)?;
    let drive_value = sample_drive_value(&plan, playback_sample_u(progress, plan.playback));
    let drive_constraint = build_drive_constraint(&artifact.assembly, &plan, drive_value)?;
    let seeded = seed_particles(&artifact.assembly, &links, &sliders, &drive_constraint)?;
    let relaxed = relax_particles(
        &artifact.assembly,
        &seeded,
        &links,
        &sliders,
        &drive_constraint,
    )?;
    let frame = snapshot_live_frame(
        &artifact.assembly,
        &plan.drive_id,
        drive_value,
        &relaxed.particles,
    )?;
    let mut state = LiveSimulationState {
        particles: relaxed.particles,
        frame,
        point_trails: BTreeMap::new(),
        cycle_samples: BTreeMap::new(),
        max_constraint_error: relaxed.max_constraint_error,
        settled: relaxed.settled,
    };
    append_live_trace_samples(&mut state, progress);
    Ok(state)
}

fn step_live_simulation_state(
    state: &mut LiveSimulationState,
    artifact: &SweepArtifact,
    progress: f32,
) -> Result<(), String> {
    let (drive_id, drive) = artifact
        .assembly
        .drives
        .iter()
        .next()
        .ok_or_else(|| "fixture missing drive".to_string())?;
    let plan = drive_plan(&artifact.assembly, drive_id, drive)?;
    let links = collect_link_constraints(&artifact.assembly)?;
    let sliders = collect_slider_constraints(&artifact.assembly)?;
    let drive_value = sample_drive_value(&plan, playback_sample_u(progress, plan.playback));
    let drive_constraint = build_drive_constraint(&artifact.assembly, &plan, drive_value)?;
    let relaxed = relax_particles_live(
        &artifact.assembly,
        &state.particles,
        &links,
        &sliders,
        &drive_constraint,
    )?;
    state.frame = snapshot_live_frame(
        &artifact.assembly,
        &plan.drive_id,
        drive_value,
        &relaxed.particles,
    )?;
    state.particles = relaxed.particles;
    state.max_constraint_error = relaxed.max_constraint_error;
    state.settled = relaxed.settled;
    append_live_trace_samples(state, progress);
    Ok(())
}

fn snapshot_live_frame(
    assembly: &AssemblySpec,
    drive_id: &str,
    drive_value: f32,
    particles: &BTreeMap<String, ParticleState>,
) -> Result<SolvedFrame, String> {
    let joint_positions: BTreeMap<String, Point2> = particles
        .iter()
        .map(|(joint_id, particle)| (joint_id.clone(), particle.pos))
        .collect();
    let poi_positions = solve_poi_positions(assembly, &joint_positions)?;
    Ok(SolvedFrame {
        u: 0.0,
        joint_positions,
        drive_values: BTreeMap::from([(drive_id.to_string(), drive_value)]),
        poi_positions,
    })
}

fn append_live_trace_samples(state: &mut LiveSimulationState, progress: f32) {
    let phase = progress.rem_euclid(1.0);
    for (poi_id, position) in &state.frame.poi_positions {
        let trail = state.point_trails.entry(poi_id.clone()).or_default();
        if trail.last().copied() != Some(*position) {
            trail.push(*position);
            if trail.len() > MAX_LIVE_TRACE_POINTS {
                let excess = trail.len() - MAX_LIVE_TRACE_POINTS;
                trail.drain(..excess);
            }
        }

        let cycle_samples = state.cycle_samples.entry(poi_id.clone()).or_default();
        if cycle_samples
            .last()
            .map(|sample| sample.point == *position && sample.phase.to_bits() == phase.to_bits())
            != Some(true)
        {
            cycle_samples.push(TraceSample {
                point: *position,
                phase,
            });
        }
    }
}

fn reset_cycle_samples(state: &mut LiveSimulationState) {
    state.cycle_samples.clear();
}

fn build_fixture_presentations()
-> Result<(Vec<FixturePresentation>, Option<Receiver<LlmTurnResult>>), String> {
    let (startup_fixture, llm_turn_rx) = startup_fixture_slot();
    let fixtures = vec![
        startup_fixture,
        build_fixture_presentation("2 SAMPLE", slider_crank_fixture())?,
        build_fixture_presentation("3 BAD REF", invalid_reference_fixture())?,
        build_fixture_presentation("4 CHAINY", expressive_chain_fixture())?,
        build_fixture_presentation("5 BRANCHY", expressive_branchy_fixture())?,
        build_fixture_presentation("6 BIRD", bird_flapper_fixture())?,
    ];

    Ok((fixtures, llm_turn_rx))
}

fn build_fixture_presentation(
    label: &str,
    assembly: AssemblySpec,
) -> Result<FixturePresentation, String> {
    let turn = next_artifact_turn();
    let status = match validate_fixture(&assembly) {
        Ok(()) => match build_sweep_artifact(assembly.clone(), turn) {
            Ok(artifact) => FixtureStatus::Solved(Arc::new(artifact)),
            Err(err) => FixtureStatus::RelaxationError(err),
        },
        Err(err) => FixtureStatus::ValidationError(err),
    };
    Ok(FixturePresentation {
        label: label.to_string(),
        status,
        assembly: Some(assembly),
    })
}

fn startup_fixture_slot() -> (FixturePresentation, Option<Receiver<LlmTurnResult>>) {
    let startup_fixture = build_fixture_presentation("1 STARTUP", startup_prompt_sample_fixture())
        .unwrap_or_else(|err| generation_error_fixture("1 STARTUP", err));
    (startup_fixture, None)
}

fn generation_error_fixture(label: &str, error: String) -> FixturePresentation {
    FixturePresentation {
        label: label.to_string(),
        status: FixtureStatus::GenerationError(error),
        assembly: None,
    }
}

fn error_chat_entry(text: impl Into<String>) -> ChatEntry {
    let text = text.into();
    eprintln!("linkage chat error: {text}");
    ChatEntry {
        role: ChatRole::Error,
        text,
    }
}

fn next_artifact_turn() -> u32 {
    NEXT_ARTIFACT_TURN.fetch_add(1, Ordering::Relaxed)
}

fn initial_chat_entries(fixtures: &[FixturePresentation]) -> Vec<ChatEntry> {
    let mut entries = vec![ChatEntry {
        role: ChatRole::System,
        text: "Type a prompt and press Enter. The model responds by calling mutation tools against the current startup assembly.".to_string(),
    }];
    entries.push(ChatEntry {
        role: ChatRole::System,
        text: "The startup assembly begins as the local sample linkage. Send a prompt to mutate it, or click Regenerate to rerun your last prompt from a blank startup assembly.".to_string(),
    });
    if std::env::var("OPENAI_API_KEY").is_err()
        || matches!(
            fixtures.first().map(|fixture| &fixture.status),
            Some(FixtureStatus::GenerationError(_))
        )
    {
        entries.push(error_chat_entry(
            "OPENAI_API_KEY is missing; local fixtures remain available.",
        ));
    }
    entries
}

fn initial_llm_history() -> Vec<LlmHistoryMessage> {
    let mut history = Vec::new();
    trim_llm_history(&mut history);
    history
}

fn trim_llm_history(history: &mut Vec<LlmHistoryMessage>) {
    if history.len() <= MAX_LLM_HISTORY_MESSAGES {
        return;
    }

    let excess = history.len() - MAX_LLM_HISTORY_MESSAGES;
    history.drain(0..excess);
    while history.len() > 1
        && !matches!(
            history.first().map(|message| message.role),
            Some(LlmHistoryRole::User)
        )
    {
        history.remove(0);
    }
}

fn startup_fixture_assembly(model: &Model) -> Option<AssemblySpec> {
    model
        .fixtures
        .first()
        .and_then(|fixture| fixture.assembly.clone())
}

fn spawn_llm_turn(
    base_assembly: AssemblySpec,
    history: Vec<LlmHistoryMessage>,
    replace_chat: bool,
) -> (
    Receiver<LlmTurnResult>,
    Vec<ChatEntry>,
    Vec<LlmHistoryMessage>,
) {
    let (tx, rx) = mpsc::channel();
    let pending_entries = vec![
        ChatEntry {
            role: ChatRole::User,
            text: history
                .last()
                .map(|message| message.text.clone())
                .unwrap_or_else(|| "missing prompt".to_string()),
        },
        ChatEntry {
            role: ChatRole::System,
            text: format!("calling {}…", startup_model_name()),
        },
    ];
    let history_for_thread = history.clone();
    std::thread::spawn(move || {
        let result = run_llm_turn(base_assembly, history_for_thread, replace_chat, None);
        let _ = tx.send(result);
    });
    (rx, pending_entries, history)
}

fn spawn_chat_turn(
    base_assembly: AssemblySpec,
    history: Vec<LlmHistoryMessage>,
    poi_trace_context_json: String,
) -> Receiver<LlmTurnResult> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = run_llm_turn(base_assembly, history, false, Some(poi_trace_context_json));
        let _ = tx.send(result);
    });
    rx
}

fn run_llm_turn(
    base_assembly: AssemblySpec,
    history: Vec<LlmHistoryMessage>,
    replace_chat: bool,
    poi_trace_context_json: Option<String>,
) -> LlmTurnResult {
    let Some(latest_prompt) = history.last().map(|message| message.text.clone()) else {
        return LlmTurnResult {
            fixture: None,
            entries: vec![error_chat_entry("missing user prompt")],
            history_messages: Vec::new(),
            replace_chat,
        };
    };
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            return LlmTurnResult {
                fixture: None,
                entries: vec![error_chat_entry("OPENAI_API_KEY is missing.")],
                history_messages: Vec::new(),
                replace_chat,
            };
        }
    };

    match generate_llm_turn(
        &api_key,
        &base_assembly,
        &history,
        poi_trace_context_json.as_deref(),
    ) {
        Ok((assembly, args, exchanges)) => {
            let fixture = match build_fixture_presentation("1 STARTUP", assembly) {
                Ok(fixture) => Some(fixture),
                Err(err) => {
                    return LlmTurnResult {
                        fixture: None,
                        entries: vec![error_chat_entry(err)],
                        history_messages: Vec::new(),
                        replace_chat,
                    };
                }
            };
            let mut entries = Vec::new();
            let mut history_messages = exchanges
                .iter()
                .map(tool_exchange_history_message)
                .collect::<Vec<_>>();
            for exchange in &exchanges {
                entries.push(ChatEntry {
                    role: ChatRole::Tool,
                    text: format!("set_assembly arguments\n{}", exchange.arguments),
                });
                entries.push(ChatEntry {
                    role: ChatRole::Tool,
                    text: format!("set_assembly response\n{}", exchange.response),
                });
            }
            entries.push(ChatEntry {
                role: ChatRole::Assistant,
                text: if args.reasoning.trim().is_empty() {
                    "Applied a new assembly.".to_string()
                } else {
                    args.reasoning.trim().to_string()
                },
            });
            if !args.reasoning.trim().is_empty() {
                history_messages.push(LlmHistoryMessage {
                    role: LlmHistoryRole::Assistant,
                    text: args.reasoning.trim().to_string(),
                });
            } else {
                history_messages.push(LlmHistoryMessage {
                    role: LlmHistoryRole::Assistant,
                    text: format!("Updated the mechanism in response to: {}", latest_prompt),
                });
            }
            LlmTurnResult {
                fixture,
                entries,
                history_messages,
                replace_chat,
            }
        }
        Err(err) => {
            let mut entries = Vec::new();
            let mut history_messages = err
                .exchanges
                .iter()
                .map(tool_exchange_history_message)
                .collect::<Vec<_>>();
            for exchange in &err.exchanges {
                entries.push(ChatEntry {
                    role: ChatRole::Tool,
                    text: format!("set_assembly arguments\n{}", exchange.arguments),
                });
                entries.push(ChatEntry {
                    role: ChatRole::Tool,
                    text: format!("set_assembly response\n{}", exchange.response),
                });
            }
            entries.push(error_chat_entry(err.message.clone()));
            history_messages.push(LlmHistoryMessage {
                role: LlmHistoryRole::Assistant,
                text: err.message,
            });
            LlmTurnResult {
                fixture: None,
                entries,
                history_messages,
                replace_chat,
            }
        }
    }
}

fn tool_exchange_history_message(exchange: &ToolExchange) -> LlmHistoryMessage {
    LlmHistoryMessage {
        role: LlmHistoryRole::Tool,
        text: format!(
            "set_assembly arguments\n{}\n\nset_assembly response\n{}",
            exchange.arguments, exchange.response
        ),
    }
}

fn slider_crank_fixture() -> AssemblySpec {
    let mut joints = BTreeMap::new();
    joints.insert(
        "j_pivot".to_string(),
        JointSpec::Fixed {
            position: [-80.0, 60.0],
        },
    );
    joints.insert("j_tip".to_string(), JointSpec::Free);
    joints.insert("j_slide".to_string(), JointSpec::Free);

    let mut parts = BTreeMap::new();
    parts.insert(
        "l_crank".to_string(),
        PartSpec::Link {
            a: "j_pivot".to_string(),
            b: "j_tip".to_string(),
            length: 60.0,
        },
    );
    parts.insert(
        "l_coupler".to_string(),
        PartSpec::Link {
            a: "j_tip".to_string(),
            b: "j_slide".to_string(),
            length: 160.0,
        },
    );
    parts.insert(
        "s_track".to_string(),
        PartSpec::Slider {
            joint: "j_slide".to_string(),
            axis_origin: [0.0, 0.0],
            axis_dir: [1.0, 0.0],
            range: [-20.0, 180.0],
        },
    );

    let mut drives = BTreeMap::new();
    drives.insert(
        "d_crank".to_string(),
        DriveSpec {
            kind: DriveKindSpec::Angular {
                pivot_joint: "j_pivot".to_string(),
                tip_joint: "j_tip".to_string(),
                link: "l_crank".to_string(),
                range: None,
            },
            sweep: SweepSpec {
                samples: 180,
                direction: SweepDirectionSpec::Clockwise,
            },
        },
    );

    AssemblySpec {
        joints,
        parts,
        drives,
        points_of_interest: vec![PointOfInterestSpec {
            id: "poi_coupler_mid".to_string(),
            host: "l_coupler".to_string(),
            t: 0.5,
            perp: 0.0,
        }],
        visualization: Some(VisualizationSpec {
            trace_model: Some(TraceModelSpec::RollingPaper {
                direction: PaperDirectionSpec::Up,
                advance_per_cycle: 180.0,
            }),
        }),
        meta: AssemblyMeta {
            name: "p1-slider-crank".to_string(),
            iteration: 1,
            notes: vec!["P1 deterministic relaxation fixture".to_string()],
        },
    }
}

fn short_arm_fixture() -> AssemblySpec {
    let mut joints = BTreeMap::new();
    joints.insert(
        "j_pivot".to_string(),
        JointSpec::Fixed {
            position: [-80.0, 60.0],
        },
    );
    joints.insert("j_tip".to_string(), JointSpec::Free);

    let mut parts = BTreeMap::new();
    parts.insert(
        "l_arm".to_string(),
        PartSpec::Link {
            a: "j_pivot".to_string(),
            b: "j_tip".to_string(),
            length: 36.0,
        },
    );

    let mut drives = BTreeMap::new();
    drives.insert(
        "d_arm".to_string(),
        DriveSpec {
            kind: DriveKindSpec::Angular {
                pivot_joint: "j_pivot".to_string(),
                tip_joint: "j_tip".to_string(),
                link: "l_arm".to_string(),
                range: None,
            },
            sweep: SweepSpec {
                samples: 180,
                direction: SweepDirectionSpec::Clockwise,
            },
        },
    );

    AssemblySpec {
        joints,
        parts,
        drives,
        points_of_interest: Vec::new(),
        visualization: None,
        meta: AssemblyMeta {
            name: "startup-short-arm".to_string(),
            iteration: 1,
            notes: vec!["Minimal startup arm fixture".to_string()],
        },
    }
}

fn invalid_reference_fixture() -> AssemblySpec {
    let mut assembly = slider_crank_fixture();
    if let Some(PartSpec::Link { b, .. }) = assembly.parts.get_mut("l_coupler") {
        *b = "j_missing".to_string();
    }
    assembly.meta.name = "p1-invalid-reference".to_string();
    assembly.meta.notes = vec!["Intentional validator failure".to_string()];
    assembly
}

fn expressive_chain_fixture() -> AssemblySpec {
    let mut assembly = slider_crank_fixture();
    assembly
        .joints
        .insert("j_mid_a".to_string(), JointSpec::Free);
    assembly
        .joints
        .insert("j_mid_b".to_string(), JointSpec::Free);
    assembly.parts.remove("l_coupler");
    assembly.parts.insert(
        "l_chain_a".to_string(),
        PartSpec::Link {
            a: "j_tip".to_string(),
            b: "j_mid_a".to_string(),
            length: 78.0,
        },
    );
    assembly.parts.insert(
        "l_chain_b".to_string(),
        PartSpec::Link {
            a: "j_mid_a".to_string(),
            b: "j_mid_b".to_string(),
            length: 82.0,
        },
    );
    assembly.parts.insert(
        "l_chain_c".to_string(),
        PartSpec::Link {
            a: "j_mid_b".to_string(),
            b: "j_slide".to_string(),
            length: 86.0,
        },
    );
    assembly.points_of_interest = vec![
        PointOfInterestSpec {
            id: "poi_chain_a".to_string(),
            host: "l_chain_a".to_string(),
            t: 0.55,
            perp: 0.0,
        },
        PointOfInterestSpec {
            id: "poi_chain_b".to_string(),
            host: "l_chain_b".to_string(),
            t: 0.45,
            perp: 0.0,
        },
        PointOfInterestSpec {
            id: "poi_chain_c".to_string(),
            host: "l_chain_c".to_string(),
            t: 0.60,
            perp: 0.0,
        },
    ];
    assembly.meta.name = "p3-expressive-chain".to_string();
    assembly.meta.notes = vec![
        "Three-link expressive chain between the crank tip and the slider".to_string(),
        "Ambiguous folds are allowed".to_string(),
    ];
    assembly
}

fn expressive_branchy_fixture() -> AssemblySpec {
    let mut assembly = expressive_chain_fixture();
    assembly.parts.insert(
        "l_brace".to_string(),
        PartSpec::Link {
            a: "j_tip".to_string(),
            b: "j_mid_b".to_string(),
            length: 200.0,
        },
    );
    assembly.meta.name = "p3-branchy-impossible-brace".to_string();
    assembly.meta.notes = vec![
        "Impossible brace forces branchy, under-settled motion".to_string(),
        "The renderer keeps the drawable artifact instead of failing hard".to_string(),
    ];
    assembly
}

fn bird_flapper_fixture() -> AssemblySpec {
    let mut joints = BTreeMap::new();
    joints.insert(
        "j_pivot".to_string(),
        JointSpec::Fixed {
            position: [-80.0, 60.0],
        },
    );
    joints.insert(
        "j_left_shoulder".to_string(),
        JointSpec::Fixed {
            position: [-122.0, 94.0],
        },
    );
    joints.insert(
        "j_right_shoulder".to_string(),
        JointSpec::Fixed {
            position: [-122.0, 26.0],
        },
    );
    joints.insert("j_tip".to_string(), JointSpec::Free);
    joints.insert("j_left_wing".to_string(), JointSpec::Free);
    joints.insert("j_right_wing".to_string(), JointSpec::Free);

    let mut parts = BTreeMap::new();
    parts.insert(
        "l_body".to_string(),
        PartSpec::Link {
            a: "j_pivot".to_string(),
            b: "j_tip".to_string(),
            length: 36.0,
        },
    );
    parts.insert(
        "l_left_root".to_string(),
        PartSpec::Link {
            a: "j_left_shoulder".to_string(),
            b: "j_left_wing".to_string(),
            length: 68.0,
        },
    );
    parts.insert(
        "l_left_span".to_string(),
        PartSpec::Link {
            a: "j_tip".to_string(),
            b: "j_left_wing".to_string(),
            length: 58.0,
        },
    );
    parts.insert(
        "l_right_root".to_string(),
        PartSpec::Link {
            a: "j_right_shoulder".to_string(),
            b: "j_right_wing".to_string(),
            length: 68.0,
        },
    );
    parts.insert(
        "l_right_span".to_string(),
        PartSpec::Link {
            a: "j_tip".to_string(),
            b: "j_right_wing".to_string(),
            length: 58.0,
        },
    );

    let mut drives = BTreeMap::new();
    drives.insert(
        "d_body".to_string(),
        DriveSpec {
            kind: DriveKindSpec::Angular {
                pivot_joint: "j_pivot".to_string(),
                tip_joint: "j_tip".to_string(),
                link: "l_body".to_string(),
                range: Some([-0.55, 0.55]),
            },
            sweep: SweepSpec {
                samples: 180,
                direction: SweepDirectionSpec::PingPong,
            },
        },
    );

    AssemblySpec {
        joints,
        parts,
        drives,
        points_of_interest: vec![
            PointOfInterestSpec {
                id: "poi_left_wing_tip".to_string(),
                host: "l_left_span".to_string(),
                t: 1.0,
                perp: 0.0,
            },
            PointOfInterestSpec {
                id: "poi_right_wing_tip".to_string(),
                host: "l_right_span".to_string(),
                t: 1.0,
                perp: 0.0,
            },
        ],
        visualization: None,
        meta: AssemblyMeta {
            name: "bird-flapper".to_string(),
            iteration: 1,
            notes: vec![
                "Symmetrical long-arm bird flapper inspired by a simple classroom wing mechanism."
                    .to_string(),
                "Left and right wingtip angles change relative to the body through the cycle."
                    .to_string(),
            ],
        },
    }
}

fn validate_fixture(assembly: &AssemblySpec) -> Result<(), String> {
    if assembly.drives.len() != 1 {
        return Err("fixture must contain exactly one drive for the current renderer".to_string());
    }

    validate_unique_entity_ids(assembly)?;

    for (joint_id, joint) in &assembly.joints {
        if let JointSpec::Fixed { position } = joint {
            if !position[0].is_finite() || !position[1].is_finite() {
                return Err(format!("joint {joint_id}: fixed position must be finite"));
            }
        }
    }

    for (part_id, part) in &assembly.parts {
        match part {
            PartSpec::Link { a, b, length } => {
                if !assembly.joints.contains_key(a) {
                    return Err(format!("part {part_id}: missing joint {a}"));
                }
                if !assembly.joints.contains_key(b) {
                    return Err(format!("part {part_id}: missing joint {b}"));
                }
                if !length.is_finite() || *length <= 0.0 {
                    return Err(format!("part {part_id}: non-positive link length"));
                }
            }
            PartSpec::Slider {
                joint,
                axis_origin,
                axis_dir,
                range,
                ..
            } => {
                if !assembly.joints.contains_key(joint) {
                    return Err(format!("part {part_id}: missing slider joint {joint}"));
                }
                if !axis_origin[0].is_finite() || !axis_origin[1].is_finite() {
                    return Err(format!("part {part_id}: slider axis_origin must be finite"));
                }
                if !axis_dir[0].is_finite() || !axis_dir[1].is_finite() {
                    return Err(format!("part {part_id}: slider axis_dir must be finite"));
                }
                if (axis_dir[0] * axis_dir[0] + axis_dir[1] * axis_dir[1]) < 0.99 {
                    return Err(format!("part {part_id}: slider axis is not normalized"));
                }
                if !range[0].is_finite() || !range[1].is_finite() {
                    return Err(format!("part {part_id}: slider range must be finite"));
                }
                if range[0] >= range[1] {
                    return Err(format!("part {part_id}: invalid slider range"));
                }
            }
        }
    }

    for (drive_id, drive) in &assembly.drives {
        match &drive.kind {
            DriveKindSpec::Angular {
                pivot_joint,
                tip_joint,
                link,
                range,
                ..
            } => {
                if !assembly.joints.contains_key(pivot_joint) {
                    return Err(format!(
                        "drive {drive_id}: missing pivot joint {pivot_joint}"
                    ));
                }
                if !assembly.joints.contains_key(tip_joint) {
                    return Err(format!("drive {drive_id}: missing tip joint {tip_joint}"));
                }
                if !assembly.parts.contains_key(link) {
                    return Err(format!("drive {drive_id}: missing link {link}"));
                }
                if let Some([start, end]) = range {
                    if !start.is_finite() || !end.is_finite() || (*start - *end).abs() < 0.000_1 {
                        return Err(format!("drive {drive_id}: invalid angular range"));
                    }
                }
                if range.is_none() && matches!(drive.sweep.direction, SweepDirectionSpec::PingPong)
                {
                    return Err(format!(
                        "drive {drive_id}: full-rotation angular drives require Clockwise, CounterClockwise, Forward, or Reverse"
                    ));
                }
            }
            DriveKindSpec::Linear { slider, range } => {
                let Some(PartSpec::Slider {
                    range: slider_range,
                    ..
                }) = assembly.parts.get(slider)
                else {
                    return Err(format!("drive {drive_id}: missing slider part {slider}"));
                };
                if let Some([start, end]) = range {
                    if !start.is_finite() || !end.is_finite() || (*start - *end).abs() < 0.000_1 {
                        return Err(format!("drive {drive_id}: invalid linear range"));
                    }
                    if *start < slider_range[0]
                        || *start > slider_range[1]
                        || *end < slider_range[0]
                        || *end > slider_range[1]
                    {
                        return Err(format!(
                            "drive {drive_id}: linear range lies outside slider track"
                        ));
                    }
                }
            }
        }
    }

    for poi in &assembly.points_of_interest {
        let Some(host_part) = assembly.parts.get(&poi.host) else {
            return Err(format!("poi {}: missing host part {}", poi.id, poi.host));
        };
        if !matches!(host_part, PartSpec::Link { .. }) {
            return Err(format!("poi {}: host {} is not a link", poi.id, poi.host));
        }
        if !poi.t.is_finite() || !poi.perp.is_finite() {
            return Err(format!("poi {}: t and perp must be finite", poi.id));
        }
    }

    if let Some(VisualizationSpec {
        trace_model:
            Some(TraceModelSpec::RollingPaper {
                advance_per_cycle, ..
            }),
    }) = &assembly.visualization
    {
        if !advance_per_cycle.is_finite() || *advance_per_cycle <= 0.0 {
            return Err(
                "visualization rolling paper advance_per_cycle must be positive".to_string(),
            );
        }
    }

    Ok(())
}

fn validate_unique_entity_ids(assembly: &AssemblySpec) -> Result<(), String> {
    let mut seen = BTreeMap::<String, &'static str>::new();
    for joint_id in assembly.joints.keys() {
        validate_unique_entity_id(&mut seen, joint_id, "joint")?;
    }
    for part_id in assembly.parts.keys() {
        validate_unique_entity_id(&mut seen, part_id, "part")?;
    }
    for drive_id in assembly.drives.keys() {
        validate_unique_entity_id(&mut seen, drive_id, "drive")?;
    }
    for poi in &assembly.points_of_interest {
        validate_unique_entity_id(&mut seen, &poi.id, "poi")?;
    }
    Ok(())
}

fn validate_unique_entity_id(
    seen: &mut BTreeMap<String, &'static str>,
    id: &str,
    kind: &'static str,
) -> Result<(), String> {
    if let Some(existing_kind) = seen.insert(id.to_string(), kind) {
        Err(format!(
            "id collision: {id} is already used by {existing_kind}"
        ))
    } else {
        Ok(())
    }
}

fn llm_system_prompt() -> Result<String, String> {
    let sample_fixture = serde_json::to_string_pretty(&llm_prompt_reference_fixture())
        .map_err(|err| format!("failed to serialize sample fixture: {err}"))?;
    Ok(format!(
        concat!(
            "You are designing and editing a mechanism assembly.\n",
            "Respond only by calling set_assembly. Return the next complete assembly. The current assembly is shown below; copy it, modify what you need, return the full result.\n",
            "Returning the assembly unchanged is valid when the user asks a question or wants a summary. Use `reasoning` for turn-local explanation. For durable annotations that should persist across turns, append to `meta.notes` — that is how the old `note` op is now expressed.\n",
            "The schema requires explicit `null` for optional fields (`visualization`, `visualization.trace_model`, angular drive `range`, linear drive `range`). The current assembly below is rendered with those nulls — copy the shape exactly.\n",
            "TOPOLOGY IS THE FIRST TOOL. When the user asks for more joints, more linkage complexity, a different path shape, or says the trace is 'just a circle', you MUST introduce at least one new free joint in assembly.joints and connect it via new links in assembly.parts. Stacking extra parallel links between the same two existing joints is NOT a substitute for new joints and will keep producing circular traces.\n",
            "A POI traces a circle whenever every part it depends on is rigidly pinned to a single rotating crank. To get a non-circular path, the POI must ride a link whose endpoints depend on at least one joint that is not directly driven by the crank. That almost always means adding a new Free joint plus at least two links that form a closed loop through it (four-bar, crank-slider, or similar).\n",
            "Prefer small, coherent changes unless the user clearly asks for a reset. Adding 1-3 new joints to introduce a real coupler is still a 'small coherent change' when the user is asking for more structure — do not pretend it is risky.\n",
            "Ids are globally unique across joints, parts, drives, and POIs. A joint and a link cannot share the same id. Ids referenced by `parts[*].a`, `parts[*].b`, slider `joint`, drive `pivot_joint`/`tip_joint`/`link`/`slider`, and POI `host` must all exist in the corresponding section of the submitted assembly.\n",
            "Validation rules:\n",
            "- Produce exactly one drive.\n",
            "- Every fixed joint position must be finite.\n",
            "- Every link must reference existing joints and have length > 0.\n",
            "- Every slider must reference an existing joint, have finite axis_origin and axis_dir values, have a normalized axis_dir, and have finite range values with range[0] < range[1].\n",
            "- Angular drives must reference existing pivot/tip joints and an existing link.\n",
            "- If an angular drive includes a range (non-null), it must be finite and non-degenerate.\n",
            "- Full-rotation angular drives without an explicit range (range: null) must not use PingPong; use Clockwise, CounterClockwise, Forward, or Reverse.\n",
            "- Linear drives must reference an existing slider part.\n",
            "- If a linear drive includes a range (non-null), it must be finite, non-degenerate, and lie within that slider track's range.\n",
            "- Every POI must reference an existing host part, that host must be a Link, and t/perp must be finite.\n",
            "- If you use rolling-paper visualization, advance_per_cycle must be finite and > 0.\n",
            "- The final assembly must satisfy all validator rules above.\n",
            "- The rendered sweep must keep every sample finite; diverging geometry is invalid.\n",
            "- You cannot clear to an empty assembly because the renderer requires exactly one valid drive in every accepted assembly.\n",
            "- If the user asks to clear or reset, replace the mechanism with a new valid assembly that still ends with exactly one drive.\n",
            "- Each set_assembly call is evaluated independently against the current assembly shown in the prompt.\n",
            "- If a tool call fails, nothing from that failed assembly is applied.\n",
            "- After a failed tool call, send a complete corrected assembly against the unchanged current assembly rather than assuming partial progress.\n",
            "Trace context:\n",
            "- Each user message may include a JSON block labeled `Latest full cycle POI traces JSON`.\n",
            "- `source = latest_completed_live_cycle` means the trace came from the most recently completed live playback cycle.\n",
            "- `source = artifact_sweep_fallback` means no live cycle has completed yet, so the trace is the current artifact sweep for one full cycle.\n",
            "- `poi_traces` maps each POI id to ordered samples through one cycle.\n",
            "- Each sample has world-space `x` and `y` coordinates in mechanism space plus normalized `phase` in [0, 1].\n",
            "- The purpose of POIs is to inform you about the paths of specific points in the assembly. They are invisible to the user.\n",
            "- Use this trace JSON as motion context when the user asks for behavioral or shape changes.\n",
            "- POIs are diagnostics only. They do not create visible geometry and do not satisfy requests for wings, legs, arms, bodies, spiders, birds, or other visible structure.\n",
            "- For morphology requests, make the visible structure with joints and parts. Do not answer with only POIs, notes, or drive-only tweaks.\n",
            "- The current assembly size is not a cap. You may add new joints and parts when needed.\n",
            "- If an ambitious topology diverges, retry with a simpler visible linkage, not a POI-only proxy.\n",
            "Here is a valid sample assembly for reference. Do not copy its exact dimensions:\n",
            "{}"
        ),
        sample_fixture
    ))
}

fn generate_llm_turn(
    api_key: &str,
    base_assembly: &AssemblySpec,
    history: &[LlmHistoryMessage],
    poi_trace_context_json: Option<&str>,
) -> Result<(AssemblySpec, SetAssemblyArgs, Vec<ToolExchange>), LlmTurnError> {
    let system_prompt = llm_system_prompt().map_err(|message| LlmTurnError {
        message,
        exchanges: Vec::new(),
    })?;
    let current_assembly =
        serde_json::to_string_pretty(base_assembly).map_err(|err| LlmTurnError {
            message: format!("failed to serialize current assembly: {err}"),
            exchanges: Vec::new(),
        })?;
    let current_assembly_summary = assembly_prompt_summary(base_assembly);
    let mut input = vec![serde_json::json!({
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": system_prompt
            }
        ]
    })];

    for message in history.iter().take(history.len().saturating_sub(1)) {
        let (role, content_type) = match message.role {
            LlmHistoryRole::User => ("user", "input_text"),
            LlmHistoryRole::Tool => ("user", "input_text"),
            LlmHistoryRole::Assistant => ("assistant", "output_text"),
        };
        input.push(serde_json::json!({
            "role": role,
            "content": [
                {
                    "type": content_type,
                    "text": message.text
                }
            ]
        }));
    }
    let latest_prompt = history
        .last()
        .map(|message| message.text.as_str())
        .ok_or_else(|| LlmTurnError {
            message: "missing user prompt".to_string(),
            exchanges: Vec::new(),
        })?;
    input.push(serde_json::json!({
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": if let Some(poi_trace_context_json) = poi_trace_context_json {
                    format!(
                        "Current assembly summary:\n{}\n\nCurrent assembly:\n{}\n\nLatest full cycle POI traces JSON:\n{}\n\nUser request:\n{}\n\nCall set_assembly now.",
                        current_assembly_summary,
                        current_assembly,
                        poi_trace_context_json,
                        latest_prompt
                    )
                } else {
                    format!(
                        "Current assembly summary:\n{}\n\nCurrent assembly:\n{}\n\nUser request:\n{}\n\nCall set_assembly now.",
                        current_assembly_summary,
                        current_assembly,
                        latest_prompt
                    )
                }
            }
        ]
    }));
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(45))
        .build()
        .map_err(|err| LlmTurnError {
            message: format!("failed to build OpenAI client: {err}"),
            exchanges: Vec::new(),
        })?;
    let mut tool_input = input;
    let mut previous_response_id = None;
    let mut exchanges = Vec::new();

    for _ in 0..MAX_TOOL_CORRECTION_ATTEMPTS {
        let body = send_llm_request(
            &client,
            api_key,
            &tool_input,
            previous_response_id.as_deref(),
        )
        .map_err(|message| LlmTurnError {
            message,
            exchanges: std::mem::take(&mut exchanges),
        })?;
        previous_response_id = extract_response_id(&body);
        let tool_call = extract_function_call(&body, "set_assembly").ok_or_else(|| {
            let fallback = extract_response_text(&body).unwrap_or_else(|| body.to_string());
            LlmTurnError {
                message: format!(
                    "OpenAI did not return a set_assembly tool call.\n{}",
                    trim_for_error(&fallback, 480)
                ),
                exchanges: std::mem::take(&mut exchanges),
            }
        })?;
        let pretty_arguments = pretty_json_string(&tool_call.arguments);
        log_tool_console("arguments", &pretty_arguments);
        let tool_args: SetAssemblyArgs = match serde_json::from_str(&tool_call.arguments) {
            Ok(tool_args) => tool_args,
            Err(err) => {
                let message = format!(
                    "failed to parse set_assembly arguments: {err}\n{}",
                    trim_for_error(&tool_call.arguments, 480)
                );
                let tool_output =
                    tool_call_error_output(&tool_call.call_id, &message, base_assembly);
                let pretty_response = pretty_json_string(
                    &tool_output
                        .get("output")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("{}")
                        .to_string(),
                );
                log_tool_console("response", &pretty_response);
                exchanges.push(ToolExchange {
                    arguments: pretty_arguments,
                    response: pretty_response,
                });
                tool_input = vec![tool_output];
                continue;
            }
        };
        let candidate = tool_args.assembly.clone();
        match validate_fixture(&candidate)
            .map_err(|err| format!("validation failed: {err}"))
            .and_then(|_| {
                build_sweep_artifact(candidate.clone(), 1)
                    .map_err(|err| format!("relaxation failed: {err}"))
                    .map(|_| candidate)
            }) {
            Ok(assembly) => {
                let pretty_response = pretty_json_string(
                    &serde_json::json!({
                        "ok": true,
                        "message": "assembly applied"
                    })
                    .to_string(),
                );
                log_tool_console("response", &pretty_response);
                exchanges.push(ToolExchange {
                    arguments: pretty_arguments,
                    response: pretty_response,
                });
                return Ok((assembly, tool_args, exchanges));
            }
            Err(err) => {
                let tool_output = tool_call_error_output(&tool_call.call_id, &err, base_assembly);
                let pretty_response = pretty_json_string(
                    &tool_output
                        .get("output")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("{}")
                        .to_string(),
                );
                log_tool_console("response", &pretty_response);
                exchanges.push(ToolExchange {
                    arguments: pretty_arguments,
                    response: pretty_response,
                });
                tool_input = vec![tool_output];
            }
        }
    }

    Err(LlmTurnError {
        message: format!(
            "model did not produce a valid assembly after {} tool correction attempts",
            MAX_TOOL_CORRECTION_ATTEMPTS
        ),
        exchanges,
    })
}

fn send_llm_request(
    client: &reqwest::blocking::Client,
    api_key: &str,
    input: &[serde_json::Value],
    previous_response_id: Option<&str>,
) -> Result<serde_json::Value, String> {
    let mut request_body = serde_json::json!({
        "model": startup_model_name(),
        "input": input,
        "tools": [
            set_assembly_tool_schema()
        ],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {
                    "type": "function",
                    "name": "set_assembly"
                }
            ]
        }
    });
    if let Some(previous_response_id) = previous_response_id {
        request_body["previous_response_id"] =
            serde_json::Value::String(previous_response_id.to_string());
    }
    let response = client
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(api_key)
        .json(&request_body)
        .send()
        .map_err(|err| format!("startup request failed: {err}"))?;
    let status = response.status();
    let body: serde_json::Value = response
        .json()
        .map_err(|err| format!("failed to decode OpenAI response: {err}"))?;
    if !status.is_success() {
        let message = body
            .get("error")
            .and_then(|value| value.get("message"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown OpenAI error");
        return Err(format!("OpenAI returned {}: {}", status.as_u16(), message));
    }
    Ok(body)
}

fn extract_response_id(body: &serde_json::Value) -> Option<String> {
    body.get("id")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn tool_call_error_output(
    call_id: &str,
    error: &str,
    current_assembly: &AssemblySpec,
) -> serde_json::Value {
    let current_assembly_json = serde_json::to_string_pretty(current_assembly)
        .unwrap_or_else(|_| "{\"error\":\"failed to serialize current assembly\"}".to_string());
    let current_assembly_summary = assembly_prompt_summary(current_assembly);
    let mut output = serde_json::json!({
        "ok": false,
        "code": "VALIDATION_FAILED",
        "message": error,
        "assembly_unchanged": true,
        "guidance": tool_error_guidance(error),
        "current_assembly_summary": current_assembly_summary,
        "current_assembly": current_assembly_json
    });
    if let Some(details) = tool_error_details(error, current_assembly) {
        output["error_details"] = details;
    }
    serde_json::json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output.to_string()
    })
}

fn tool_error_guidance(error: &str) -> String {
    let mut guidance =
        "The submitted assembly was rejected. Resubmit set_assembly with the corrected assembly. The current assembly (unchanged) is shown below — modify it and return the full result.".to_string();
    if error.starts_with("id collision: ") {
        guidance.push_str(
            " Every entity id must be globally unique across joints, parts, drives, and POIs. Rename the conflicting id so each entity has its own.",
        );
    }
    if error.contains("missing joint") {
        guidance.push_str(
            " A link or slider references a joint that does not exist in assembly.joints. Add the joint to assembly.joints or change the reference.",
        );
    }
    if error.contains("relaxation failed") {
        guidance.push_str(
            " Simplify the visible linkage geometry and lengths until it settles; do not replace visible structure with POIs.",
        );
    }
    guidance
}

fn tool_error_details(error: &str, current_assembly: &AssemblySpec) -> Option<serde_json::Value> {
    if let Some(rest) = error.strip_prefix("id collision: ") {
        if let Some((id, existing_kind)) = rest.split_once(" is already used by ") {
            return Some(serde_json::json!({
                "category": "id_collision",
                "id": id,
                "existing_kind": existing_kind,
                "rule": "Ids are globally unique across joints, parts, drives, and POIs.",
                "suggested_fix": format!("Rename the conflicting id {id} so each entity in assembly.joints, assembly.parts, assembly.drives, and assembly.points_of_interest has its own.")
            }));
        }
    }

    if let Some((owner_kind, owner_id, missing_kind, missing_id)) = missing_reference_details(error)
    {
        let suggested_fix = if missing_kind == "joint" {
            format!(
                "Add \"{missing_id}\" to assembly.joints (e.g. {{\"type\":\"Free\"}} or {{\"type\":\"Fixed\",\"position\":[x,y]}}), or change the reference in {owner_kind} {owner_id} to an existing joint."
            )
        } else {
            format!(
                "Add \"{missing_id}\" to assembly.{missing_kind}s, or change the reference in {owner_kind} {owner_id} to an existing {missing_kind}."
            )
        };
        return Some(serde_json::json!({
            "category": "missing_reference",
            "owner_kind": owner_kind,
            "owner_id": owner_id,
            "missing_kind": missing_kind,
            "missing_id": missing_id,
            "suggested_fix": suggested_fix
        }));
    }

    if error.contains("fixture must contain exactly one drive for the current renderer") {
        return Some(serde_json::json!({
            "category": "drive_count",
            "required_drives": 1,
            "current_drives": current_assembly.drives.len(),
            "suggested_fix": "assembly.drives must contain exactly one drive in the next set_assembly call."
        }));
    }

    None
}

fn missing_reference_details(error: &str) -> Option<(&str, &str, &str, &str)> {
    for prefix in ["validation failed: ", "mutation application failed: "] {
        if let Some(rest) = error.strip_prefix(prefix) {
            if let Some(rest) = rest.strip_prefix("part ") {
                if let Some((owner_id, missing_id)) = rest.split_once(": missing joint ") {
                    return Some(("part", owner_id, "joint", missing_id));
                }
                if let Some((owner_id, missing_id)) = rest.split_once(": missing slider joint ") {
                    return Some(("part", owner_id, "joint", missing_id));
                }
            }
            if let Some(rest) = rest.strip_prefix("drive ") {
                if let Some((owner_id, missing_id)) = rest.split_once(": missing pivot joint ") {
                    return Some(("drive", owner_id, "joint", missing_id));
                }
                if let Some((owner_id, missing_id)) = rest.split_once(": missing tip joint ") {
                    return Some(("drive", owner_id, "joint", missing_id));
                }
                if let Some((owner_id, missing_id)) = rest.split_once(": missing link ") {
                    return Some(("drive", owner_id, "part", missing_id));
                }
                if let Some((owner_id, missing_id)) = rest.split_once(": missing slider part ") {
                    return Some(("drive", owner_id, "part", missing_id));
                }
            }
            if let Some(rest) = rest.strip_prefix("poi ") {
                if let Some((owner_id, missing_id)) = rest.split_once(": missing host part ") {
                    return Some(("poi", owner_id, "part", missing_id));
                }
            }
        }
    }
    None
}

fn assembly_prompt_summary(assembly: &AssemblySpec) -> String {
    let mut lines = vec![
        format!(
            "Counts: {} joints, {} parts, {} drives, {} POIs.",
            assembly.joints.len(),
            assembly.parts.len(),
            assembly.drives.len(),
            assembly.points_of_interest.len()
        ),
        "Current counts are not a limit. You may add new joints and parts in the next batch."
            .to_string(),
    ];

    if assembly.joints.is_empty() {
        lines.push("Joints: none.".to_string());
    } else {
        let joints = assembly
            .joints
            .iter()
            .map(|(id, joint)| match joint {
                JointSpec::Fixed { position } => {
                    format!("{id}=Fixed({:.2}, {:.2})", position[0], position[1])
                }
                JointSpec::Free => format!("{id}=Free"),
            })
            .collect::<Vec<_>>()
            .join("; ");
        lines.push(format!("Joints: {joints}."));
    }

    if assembly.parts.is_empty() {
        lines.push("Parts: none.".to_string());
    } else {
        let parts = assembly
            .parts
            .iter()
            .map(|(id, part)| match part {
                PartSpec::Link { a, b, length } => {
                    format!("{id}=Link({a}->{b}, len={length:.2})")
                }
                PartSpec::Slider {
                    joint,
                    axis_origin,
                    axis_dir,
                    range,
                } => format!(
                    "{id}=Slider(joint={joint}, origin=({:.2}, {:.2}), dir=({:.2}, {:.2}), range=[{:.2}, {:.2}])",
                    axis_origin[0], axis_origin[1], axis_dir[0], axis_dir[1], range[0], range[1]
                ),
            })
            .collect::<Vec<_>>()
            .join("; ");
        lines.push(format!("Parts: {parts}."));
    }

    if assembly.drives.is_empty() {
        lines.push("Drives: none.".to_string());
    } else {
        let drives = assembly
            .drives
            .iter()
            .map(|(id, drive)| match &drive.kind {
                DriveKindSpec::Angular {
                    pivot_joint,
                    tip_joint,
                    link,
                    range,
                } => format!(
                    "{id}=Angular(link={link}, pivot={pivot_joint}, tip={tip_joint}, range={})",
                    format_optional_range(*range)
                ),
                DriveKindSpec::Linear { slider, range } => format!(
                    "{id}=Linear(slider={slider}, range={})",
                    format_optional_range(*range)
                ),
            })
            .collect::<Vec<_>>()
            .join("; ");
        lines.push(format!("Drives: {drives}."));
    }

    if assembly.points_of_interest.is_empty() {
        lines.push("POIs: none.".to_string());
    } else {
        let pois = assembly
            .points_of_interest
            .iter()
            .map(|poi| {
                format!(
                    "{}@{}(t={:.2}, perp={:.2})",
                    poi.id, poi.host, poi.t, poi.perp
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        lines.push(format!("POIs: {pois}."));
    }

    lines.join("\n")
}

fn format_optional_range(range: Option<[f32; 2]>) -> String {
    match range {
        Some([start, end]) => format!("[{start:.2}, {end:.2}]"),
        None => "none".to_string(),
    }
}

fn pretty_json_string(text: &str) -> String {
    serde_json::from_str::<serde_json::Value>(text)
        .and_then(|value| serde_json::to_string_pretty(&value))
        .unwrap_or_else(|_| text.to_string())
}

fn log_tool_console(label: &str, text: &str) {
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "TOOL: set_assembly {label} {text}");
    let _ = stderr.flush();
}

fn startup_model_name() -> String {
    std::env::var("OPENAI_MODEL").unwrap_or_else(|_| STARTUP_MODEL.to_string())
}

fn startup_prompt_sample_fixture() -> AssemblySpec {
    short_arm_fixture()
}

fn llm_prompt_reference_fixture() -> AssemblySpec {
    slider_crank_fixture()
}

fn set_assembly_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "name": "set_assembly",
        "description": "Submit the complete next assembly. Copy the current assembly shown in the prompt, modify only what the user asks for, and return the full result. Use reasoning for turn-local explanation; put durable notes in assembly.meta.notes.",
        "strict": true,
        "parameters": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "reasoning": { "type": "string" },
                "assembly": assembly_spec_schema(),
            },
            "required": ["reasoning", "assembly"]
        }
    })
}

fn assembly_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "joints": {
                "type": "object",
                "additionalProperties": joint_spec_schema()
            },
            "parts": {
                "type": "object",
                "additionalProperties": part_spec_schema()
            },
            "drives": {
                "type": "object",
                "additionalProperties": drive_spec_schema()
            },
            "points_of_interest": {
                "type": "array",
                "items": poi_spec_schema()
            },
            "visualization": nullable_visualization_spec_schema(),
            "meta": assembly_meta_schema()
        },
        "required": ["joints", "parts", "drives", "points_of_interest", "visualization", "meta"]
    })
}

fn assembly_meta_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "name": { "type": "string" },
            "iteration": { "type": "integer", "minimum": 0 },
            "notes": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["name", "iteration", "notes"]
    })
}

fn nullable_visualization_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "anyOf": [
            visualization_spec_schema(),
            { "type": "null" }
        ]
    })
}

fn visualization_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "trace_model": {
                "anyOf": [
                    trace_model_spec_schema(),
                    { "type": "null" }
                ]
            }
        },
        "required": ["trace_model"]
    })
}

fn trace_model_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "type": { "type": "string", "enum": ["RollingPaper"] },
            "direction": {
                "type": "string",
                "enum": ["Up", "Down", "Left", "Right"]
            },
            "advance_per_cycle": { "type": "number" }
        },
        "required": ["type", "direction", "advance_per_cycle"]
    })
}

fn joint_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "type": { "type": "string", "enum": ["Fixed"] },
                    "position": point2_schema(),
                },
                "required": ["type", "position"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "type": { "type": "string", "enum": ["Free"] }
                },
                "required": ["type"]
            }
        ]
    })
}

fn part_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "type": { "type": "string", "enum": ["Link"] },
                    "a": { "type": "string" },
                    "b": { "type": "string" },
                    "length": { "type": "number" }
                },
                "required": ["type", "a", "b", "length"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "type": { "type": "string", "enum": ["Slider"] },
                    "joint": { "type": "string" },
                    "axis_origin": point2_schema(),
                    "axis_dir": point2_schema(),
                    "range": point2_schema()
                },
                "required": ["type", "joint", "axis_origin", "axis_dir", "range"]
            }
        ]
    })
}

fn drive_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "kind": drive_kind_schema(),
            "sweep": sweep_schema()
        },
        "required": ["kind", "sweep"]
    })
}

fn drive_kind_schema() -> serde_json::Value {
    serde_json::json!({
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "type": { "type": "string", "enum": ["Angular"] },
                    "pivot_joint": { "type": "string" },
                    "tip_joint": { "type": "string" },
                    "link": { "type": "string" },
                    "range": {
                        "type": ["array", "null"],
                        "items": { "type": "number" },
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["type", "pivot_joint", "tip_joint", "link", "range"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "type": { "type": "string", "enum": ["Linear"] },
                    "slider": { "type": "string" },
                    "range": {
                        "type": ["array", "null"],
                        "items": { "type": "number" },
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["type", "slider", "range"]
            }
        ]
    })
}

fn sweep_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "samples": { "type": "integer", "minimum": 2 },
            "direction": {
                "type": "string",
                "enum": [
                    "Forward",
                    "Reverse",
                    "PingPong",
                    "Clockwise",
                    "CounterClockwise",
                    "CW",
                    "CCW"
                ]
            }
        },
        "required": ["samples", "direction"]
    })
}

fn poi_spec_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "id": { "type": "string" },
            "host": { "type": "string" },
            "t": { "type": "number" },
            "perp": { "type": "number" }
        },
        "required": ["id", "host", "t", "perp"]
    })
}

fn point2_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 2,
        "maxItems": 2
    })
}

fn extract_response_text(body: &serde_json::Value) -> Option<String> {
    if let Some(text) = body.get("output_text").and_then(serde_json::Value::as_str) {
        return Some(text.to_string());
    }

    body.get("output")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|item| {
            item.get("content")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
        })
        .find_map(|content| {
            content
                .get("text")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
        })
}

fn extract_function_call(body: &serde_json::Value, tool_name: &str) -> Option<ToolCall> {
    body.get("output")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                let item_type = item.get("type").and_then(serde_json::Value::as_str);
                let item_name = item.get("name").and_then(serde_json::Value::as_str);
                if item_type == Some("function_call") && item_name == Some(tool_name) {
                    Some(ToolCall {
                        call_id: item
                            .get("call_id")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        arguments: item
                            .get("arguments")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    })
                } else {
                    None
                }
            })
        })
}

fn empty_startup_assembly() -> AssemblySpec {
    AssemblySpec {
        joints: BTreeMap::new(),
        parts: BTreeMap::new(),
        drives: BTreeMap::new(),
        points_of_interest: Vec::new(),
        visualization: None,
        meta: AssemblyMeta {
            name: "startup-generated".to_string(),
            iteration: 1,
            notes: Vec::new(),
        },
    }
}

fn trim_for_error(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        text.to_string()
    } else {
        format!("{}...", text.chars().take(max_len).collect::<String>())
    }
}

fn collect_link_constraints(assembly: &AssemblySpec) -> Result<Vec<LinkConstraint>, String> {
    let mut links = Vec::new();
    for (part_id, part) in &assembly.parts {
        if let PartSpec::Link { a, b, length } = part {
            if !assembly.joints.contains_key(a) || !assembly.joints.contains_key(b) {
                return Err(format!("link {part_id} references missing joints"));
            }
            links.push(LinkConstraint {
                a: a.clone(),
                b: b.clone(),
                length: *length,
            });
        }
    }
    Ok(links)
}

fn collect_slider_constraints(assembly: &AssemblySpec) -> Result<Vec<SliderConstraint>, String> {
    let mut sliders = Vec::new();
    for (part_id, part) in &assembly.parts {
        if let PartSpec::Slider {
            joint,
            axis_origin,
            axis_dir,
            range,
        } = part
        {
            if !assembly.joints.contains_key(joint) {
                return Err(format!("slider {part_id} references missing joint {joint}"));
            }
            let dir = vec2(axis_dir[0], axis_dir[1]);
            if dir.length_squared() < 0.99 {
                return Err(format!("slider {part_id} has a non-normalized axis"));
            }
            sliders.push(SliderConstraint {
                joint_id: joint.clone(),
                axis_origin: pt2(axis_origin[0], axis_origin[1]),
                axis_dir: dir.normalize(),
                range: (range[0], range[1]),
            });
        }
    }
    Ok(sliders)
}

fn build_drive_constraint(
    assembly: &AssemblySpec,
    plan: &DrivePlan,
    drive_value: f32,
) -> Result<DriveConstraint, String> {
    let drive = assembly
        .drives
        .get(&plan.drive_id)
        .ok_or_else(|| format!("drive {} is missing", plan.drive_id))?;
    match &drive.kind {
        DriveKindSpec::Angular {
            pivot_joint,
            tip_joint,
            link,
            ..
        } => {
            let length = match assembly.parts.get(link) {
                Some(PartSpec::Link { length, .. }) => *length,
                _ => {
                    return Err(format!(
                        "drive {} references missing link {link}",
                        plan.drive_id
                    ));
                }
            };
            Ok(DriveConstraint::Angular {
                pivot_joint_id: pivot_joint.clone(),
                tip_joint_id: tip_joint.clone(),
                angle: drive_value,
                length,
            })
        }
        DriveKindSpec::Linear { slider, .. } => {
            let slider = assembly
                .parts
                .get(slider)
                .ok_or_else(|| format!("drive {} references missing slider", plan.drive_id))?;
            let PartSpec::Slider {
                joint,
                axis_origin,
                axis_dir,
                ..
            } = slider
            else {
                return Err(format!(
                    "drive {} references a non-slider part",
                    plan.drive_id
                ));
            };
            Ok(DriveConstraint::Linear {
                joint_id: joint.clone(),
                axis_origin: pt2(axis_origin[0], axis_origin[1]),
                axis_dir: vec2(axis_dir[0], axis_dir[1]).normalize(),
                value: drive_value,
            })
        }
    }
}

fn seed_particles(
    assembly: &AssemblySpec,
    links: &[LinkConstraint],
    sliders: &[SliderConstraint],
    drive_constraint: &DriveConstraint,
) -> Result<BTreeMap<String, ParticleState>, String> {
    let mut positions = BTreeMap::<String, Point2>::new();
    let joint_masses = joint_mass_map(assembly, links);

    for (joint_id, joint) in &assembly.joints {
        if let JointSpec::Fixed { position } = joint {
            positions.insert(joint_id.clone(), pt2(position[0], position[1]));
        }
    }

    for slider in sliders {
        let midpoint = (slider.range.0 + slider.range.1) * 0.5;
        let slider_pos = slider.axis_origin + slider.axis_dir * midpoint;
        positions
            .entry(slider.joint_id.clone())
            .or_insert(slider_pos);
    }

    match drive_constraint {
        DriveConstraint::Angular {
            pivot_joint_id,
            tip_joint_id,
            angle,
            length,
        } => {
            let drive_vec = vec2(angle.cos(), angle.sin()) * *length;
            match (
                positions.get(pivot_joint_id).copied(),
                positions.get(tip_joint_id).copied(),
            ) {
                (Some(pivot), Some(_tip)) => {
                    positions.insert(tip_joint_id.clone(), pivot + drive_vec);
                }
                (Some(pivot), None) => {
                    positions.insert(tip_joint_id.clone(), pivot + drive_vec);
                }
                (None, Some(tip)) => {
                    positions.insert(pivot_joint_id.clone(), tip - drive_vec);
                }
                (None, None) => {
                    let pivot = deterministic_seed_point(pivot_joint_id, tip_joint_id);
                    positions.insert(pivot_joint_id.clone(), pivot);
                    positions.insert(tip_joint_id.clone(), pivot + drive_vec);
                }
            }
        }
        DriveConstraint::Linear {
            joint_id,
            axis_origin,
            axis_dir,
            value,
        } => {
            positions.insert(joint_id.clone(), *axis_origin + *axis_dir * *value);
        }
    }

    for _ in 0..(links.len() + assembly.joints.len()).max(1) {
        let mut changed = false;
        for link in links {
            let a = positions.get(&link.a).copied();
            let b = positions.get(&link.b).copied();
            match (a, b) {
                (Some(a_pos), None) => {
                    positions.insert(
                        link.b.clone(),
                        a_pos + deterministic_seed_direction(&link.a, &link.b) * link.length,
                    );
                    changed = true;
                }
                (None, Some(b_pos)) => {
                    positions.insert(
                        link.a.clone(),
                        b_pos - deterministic_seed_direction(&link.a, &link.b) * link.length,
                    );
                    changed = true;
                }
                _ => {}
            }
        }
        if !changed {
            break;
        }
    }

    for (index, joint_id) in assembly.joints.keys().enumerate() {
        positions.entry(joint_id.clone()).or_insert_with(|| {
            let dx = 24.0 + index as f32 * 18.0;
            let dy = if index % 2 == 0 { 28.0 } else { -28.0 };
            pt2(dx, dy)
        });
    }

    let mut particles = BTreeMap::new();
    for (joint_id, joint) in &assembly.joints {
        let pos = *positions
            .get(joint_id)
            .ok_or_else(|| format!("joint {joint_id} could not be seeded"))?;
        particles.insert(
            joint_id.clone(),
            ParticleState {
                pos,
                prev_pos: pos,
                fixed: matches!(joint, JointSpec::Fixed { .. }),
                mass: *joint_masses.get(joint_id).unwrap_or(&1.0),
            },
        );
    }
    Ok(particles)
}

fn joint_mass_map(assembly: &AssemblySpec, links: &[LinkConstraint]) -> BTreeMap<String, f32> {
    let mut masses: BTreeMap<String, f32> = assembly
        .joints
        .keys()
        .map(|joint_id| (joint_id.clone(), 1.0))
        .collect();
    for link in links {
        let share = (link.length * 0.5).max(1.0);
        *masses.entry(link.a.clone()).or_insert(1.0) += share;
        *masses.entry(link.b.clone()).or_insert(1.0) += share;
    }
    masses
}

fn deterministic_seed_direction(a: &str, b: &str) -> Vec2 {
    let mut hash = 2166136261u32;
    for byte in a.bytes().chain([b':']).chain(b.bytes()) {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    let angle = (hash % 8192) as f32 / 8192.0 * TAU;
    vec2(angle.cos(), angle.sin()).normalize()
}

fn deterministic_seed_point(a: &str, b: &str) -> Point2 {
    let dir = deterministic_seed_direction(a, b);
    let radius = 36.0 + ((a.len() + b.len()) % 7) as f32 * 12.0;
    pt2(dir.x * radius, dir.y * radius)
}

fn verlet_step(particles: &mut BTreeMap<String, ParticleState>) {
    for particle in particles.values_mut() {
        if particle.fixed {
            continue;
        }
        let current = particle.pos;
        let velocity = (particle.pos - particle.prev_pos) * RELAXATION_DAMPING;
        particle.pos += velocity + vec2(0.0, -GRAVITY_ACCEL_Y);
        particle.prev_pos = current;
    }
}

fn relax_particles(
    assembly: &AssemblySpec,
    initial_state: &BTreeMap<String, ParticleState>,
    links: &[LinkConstraint],
    sliders: &[SliderConstraint],
    drive_constraint: &DriveConstraint,
) -> Result<RelaxationResult, String> {
    relax_particles_internal(
        assembly,
        initial_state,
        links,
        sliders,
        drive_constraint,
        true,
    )
}

fn relax_particles_live(
    assembly: &AssemblySpec,
    initial_state: &BTreeMap<String, ParticleState>,
    links: &[LinkConstraint],
    sliders: &[SliderConstraint],
    drive_constraint: &DriveConstraint,
) -> Result<RelaxationResult, String> {
    relax_particles_internal(
        assembly,
        initial_state,
        links,
        sliders,
        drive_constraint,
        false,
    )
}

fn relax_particles_internal(
    assembly: &AssemblySpec,
    initial_state: &BTreeMap<String, ParticleState>,
    links: &[LinkConstraint],
    sliders: &[SliderConstraint],
    drive_constraint: &DriveConstraint,
    clear_velocity: bool,
) -> Result<RelaxationResult, String> {
    let mut particles = initial_state.clone();
    verlet_step(&mut particles);

    let mut max_constraint_error = f32::INFINITY;
    for _ in 0..RELAXATION_ITERATIONS {
        let mut max_correction = 0.0f32;
        max_correction = max_correction.max(project_fixed_constraints(assembly, &mut particles)?);
        max_correction =
            max_correction.max(project_drive_constraint(drive_constraint, &mut particles)?);
        max_correction = max_correction.max(project_slider_constraints(sliders, &mut particles)?);
        max_correction = max_correction.max(project_link_constraints(links, &mut particles)?);
        max_correction = max_correction.max(project_slider_constraints(sliders, &mut particles)?);
        max_correction = max_correction.max(project_fixed_constraints(assembly, &mut particles)?);
        max_correction =
            max_correction.max(project_drive_constraint(drive_constraint, &mut particles)?);

        if particles.values().any(|particle| {
            !particle.pos.x.is_finite()
                || !particle.pos.y.is_finite()
                || particle.pos.x.abs() > RELAXATION_DIVERGENCE_LIMIT
                || particle.pos.y.abs() > RELAXATION_DIVERGENCE_LIMIT
        }) {
            return Err("relaxation diverged".to_string());
        }

        max_constraint_error =
            compute_max_constraint_error(assembly, &particles, links, sliders, drive_constraint)?;
        if max_constraint_error <= RELAXATION_TOLERANCE && max_correction <= RELAXATION_TOLERANCE {
            if clear_velocity {
                for particle in particles.values_mut() {
                    particle.prev_pos = particle.pos;
                }
            }
            return Ok(RelaxationResult {
                particles,
                max_constraint_error,
                settled: true,
            });
        }
    }

    if !max_constraint_error.is_finite() {
        return Err("relaxation diverged".to_string());
    }

    if clear_velocity {
        for particle in particles.values_mut() {
            particle.prev_pos = particle.pos;
        }
    }

    Ok(RelaxationResult {
        particles,
        max_constraint_error,
        settled: false,
    })
}

fn project_fixed_constraints(
    assembly: &AssemblySpec,
    particles: &mut BTreeMap<String, ParticleState>,
) -> Result<f32, String> {
    let mut max_correction = 0.0;
    for (joint_id, joint) in &assembly.joints {
        if let JointSpec::Fixed { position } = joint {
            let anchor = pt2(position[0], position[1]);
            let particle = particles
                .get_mut(joint_id)
                .ok_or_else(|| format!("fixed joint {joint_id} is missing from particles"))?;
            max_correction = max_correction.max((anchor - particle.pos).length());
            particle.pos = anchor;
            particle.prev_pos = anchor;
        }
    }
    Ok(max_correction)
}

fn project_drive_constraint(
    drive_constraint: &DriveConstraint,
    particles: &mut BTreeMap<String, ParticleState>,
) -> Result<f32, String> {
    match drive_constraint {
        DriveConstraint::Angular {
            pivot_joint_id,
            tip_joint_id,
            angle,
            length,
        } => {
            let pivot = particles
                .get(pivot_joint_id)
                .map(|particle| particle.pos)
                .ok_or_else(|| format!("drive pivot {pivot_joint_id} is missing"))?;
            let target = pivot + vec2(angle.cos(), angle.sin()) * *length;
            let tip = particles
                .get_mut(tip_joint_id)
                .ok_or_else(|| format!("drive tip {tip_joint_id} is missing"))?;
            let correction = (target - tip.pos).length();
            tip.pos = target;
            tip.prev_pos = target;
            Ok(correction)
        }
        DriveConstraint::Linear {
            joint_id,
            axis_origin,
            axis_dir,
            value,
        } => {
            let target = *axis_origin + *axis_dir * *value;
            let joint = particles
                .get_mut(joint_id)
                .ok_or_else(|| format!("driven slider joint {joint_id} is missing"))?;
            let correction = (target - joint.pos).length();
            joint.pos = target;
            joint.prev_pos = target;
            Ok(correction)
        }
    }
}

fn project_slider_constraints(
    sliders: &[SliderConstraint],
    particles: &mut BTreeMap<String, ParticleState>,
) -> Result<f32, String> {
    let mut max_correction = 0.0;
    for slider in sliders {
        let particle = particles
            .get_mut(&slider.joint_id)
            .ok_or_else(|| format!("slider joint {} is missing", slider.joint_id))?;
        debug_assert!(
            !particle.fixed,
            "slider projection should not target a fixed joint"
        );
        let rel = particle.pos - slider.axis_origin;
        let scalar = rel.dot(slider.axis_dir);
        let clamped = scalar.clamp(slider.range.0, slider.range.1);
        let target = slider.axis_origin + slider.axis_dir * clamped;
        max_correction = max_correction.max((target - particle.pos).length());
        particle.pos = target;
    }
    Ok(max_correction)
}

fn project_link_constraints(
    links: &[LinkConstraint],
    particles: &mut BTreeMap<String, ParticleState>,
) -> Result<f32, String> {
    let mut max_correction = 0.0;
    for link in links {
        let (a_pos, a_fixed, a_mass) = particles
            .get(&link.a)
            .map(|particle| (particle.pos, particle.fixed, particle.mass))
            .ok_or_else(|| format!("link endpoint {} is missing", link.a))?;
        let (b_pos, b_fixed, b_mass) = particles
            .get(&link.b)
            .map(|particle| (particle.pos, particle.fixed, particle.mass))
            .ok_or_else(|| format!("link endpoint {} is missing", link.b))?;
        let delta = b_pos - a_pos;
        let distance = delta.length();
        let dir = if distance > 0.000_1 {
            delta / distance
        } else {
            deterministic_seed_direction(&link.a, &link.b)
        };
        let error = distance - link.length;
        let correction = dir * error;

        match (a_fixed, b_fixed) {
            (false, false) => {
                let a_inv_mass = 1.0 / a_mass.max(0.000_1);
                let b_inv_mass = 1.0 / b_mass.max(0.000_1);
                let inv_mass_sum = (a_inv_mass + b_inv_mass).max(0.000_1);
                let a_share = a_inv_mass / inv_mass_sum;
                let b_share = b_inv_mass / inv_mass_sum;
                if let Some(a) = particles.get_mut(&link.a) {
                    a.pos += correction * a_share;
                }
                if let Some(b) = particles.get_mut(&link.b) {
                    b.pos -= correction * b_share;
                }
                max_correction = max_correction.max(correction.length() * a_share.max(b_share));
            }
            (true, false) => {
                if let Some(b) = particles.get_mut(&link.b) {
                    b.pos -= correction;
                }
                max_correction = max_correction.max(correction.length());
            }
            (false, true) => {
                if let Some(a) = particles.get_mut(&link.a) {
                    a.pos += correction;
                }
                max_correction = max_correction.max(correction.length());
            }
            (true, true) => {}
        }
    }
    Ok(max_correction)
}

fn compute_max_constraint_error(
    assembly: &AssemblySpec,
    particles: &BTreeMap<String, ParticleState>,
    links: &[LinkConstraint],
    sliders: &[SliderConstraint],
    drive_constraint: &DriveConstraint,
) -> Result<f32, String> {
    let mut max_error = 0.0f32;

    for (joint_id, joint) in &assembly.joints {
        if let JointSpec::Fixed { position } = joint {
            let anchor = pt2(position[0], position[1]);
            let particle = particles
                .get(joint_id)
                .ok_or_else(|| format!("fixed joint {joint_id} is missing from particles"))?;
            max_error = max_error.max((particle.pos - anchor).length());
        }
    }

    for link in links {
        let a = particles
            .get(&link.a)
            .ok_or_else(|| format!("link endpoint {} is missing", link.a))?
            .pos;
        let b = particles
            .get(&link.b)
            .ok_or_else(|| format!("link endpoint {} is missing", link.b))?
            .pos;
        max_error = max_error.max(((b - a).length() - link.length).abs());
    }

    for slider in sliders {
        let particle = particles
            .get(&slider.joint_id)
            .ok_or_else(|| format!("slider joint {} is missing", slider.joint_id))?;
        let rel = particle.pos - slider.axis_origin;
        let scalar = rel.dot(slider.axis_dir);
        let clamped = scalar.clamp(slider.range.0, slider.range.1);
        let target = slider.axis_origin + slider.axis_dir * clamped;
        max_error = max_error.max((particle.pos - target).length());
    }

    max_error = max_error.max(match drive_constraint {
        DriveConstraint::Angular {
            pivot_joint_id,
            tip_joint_id,
            angle,
            length,
        } => {
            let pivot = particles
                .get(pivot_joint_id)
                .ok_or_else(|| format!("drive pivot {pivot_joint_id} is missing"))?
                .pos;
            let tip = particles
                .get(tip_joint_id)
                .ok_or_else(|| format!("drive tip {tip_joint_id} is missing"))?
                .pos;
            let target = pivot + vec2(angle.cos(), angle.sin()) * *length;
            (tip - target).length()
        }
        DriveConstraint::Linear {
            joint_id,
            axis_origin,
            axis_dir,
            value,
        } => {
            let joint = particles
                .get(joint_id)
                .ok_or_else(|| format!("driven slider joint {joint_id} is missing"))?
                .pos;
            let target = *axis_origin + *axis_dir * *value;
            (joint - target).length()
        }
    });

    Ok(max_error)
}

fn build_sweep_artifact(assembly: AssemblySpec, turn: u32) -> Result<SweepArtifact, String> {
    let Some((drive_id, drive)) = assembly.drives.iter().next() else {
        return Err("fixture missing drive".to_string());
    };
    let plan = drive_plan(&assembly, drive_id, drive)?;
    let links = collect_link_constraints(&assembly)?;
    let sliders = collect_slider_constraints(&assembly)?;
    let samples = drive.sweep.samples.max(2) as usize;

    let first_drive_value = sample_drive_value(&plan, 0.0);
    let first_drive_constraint = build_drive_constraint(&assembly, &plan, first_drive_value)?;
    let mut particle_state = seed_particles(&assembly, &links, &sliders, &first_drive_constraint)?;

    let mut frames = Vec::with_capacity(samples);
    let mut point_paths: BTreeMap<String, Vec<Point2>> = BTreeMap::new();
    let mut unsettled_samples = 0u32;
    let mut peak_constraint_error = 0.0f32;

    for sample_idx in 0..samples {
        let denom = (samples - 1).max(1) as f32;
        let base_u = sample_idx as f32 / denom;
        let drive_value = sample_drive_value(&plan, base_u);
        let drive_constraint = build_drive_constraint(&assembly, &plan, drive_value)?;
        let relaxed = relax_particles(
            &assembly,
            &particle_state,
            &links,
            &sliders,
            &drive_constraint,
        )?;
        peak_constraint_error = peak_constraint_error.max(relaxed.max_constraint_error);
        if !relaxed.settled {
            unsettled_samples += 1;
        }
        particle_state = relaxed.particles;

        let joint_positions: BTreeMap<String, Point2> = particle_state
            .iter()
            .map(|(joint_id, particle)| (joint_id.clone(), particle.pos))
            .collect();
        let poi_positions = solve_poi_positions(&assembly, &joint_positions)?;
        for (poi_id, position) in &poi_positions {
            point_paths
                .entry(poi_id.clone())
                .or_default()
                .push(*position);
        }

        frames.push(SolvedFrame {
            u: base_u,
            joint_positions,
            drive_values: BTreeMap::from([(plan.drive_id.clone(), drive_value)]),
            poi_positions,
        });

        if !relaxed.max_constraint_error.is_finite() {
            return Err("relaxation diverged".to_string());
        }
    }

    let mut notes = Vec::new();
    if unsettled_samples > 0 {
        notes.push(format!(
            "{} / {} samples stayed loose enough to read as expressive motion",
            unsettled_samples, samples
        ));
        if peak_constraint_error > 2.0 {
            notes.push("Branch switching, snaps, or wobble are likely.".to_string());
        }
    }

    Ok(SweepArtifact {
        assembly: Arc::new(assembly),
        frames,
        telemetry: SweepTelemetry {
            point_paths,
            unsettled_samples,
            peak_constraint_error,
            notes,
        },
        turn,
        reset_phase: false,
        cycle_seconds: plan.cycle_seconds,
        playback: plan.playback,
    })
}

fn solve_poi_positions(
    assembly: &AssemblySpec,
    joint_positions: &BTreeMap<String, Point2>,
) -> Result<BTreeMap<String, Point2>, String> {
    let mut positions = BTreeMap::new();
    for poi in &assembly.points_of_interest {
        let Some(host_part) = assembly.parts.get(&poi.host) else {
            return Err(format!("poi {}: missing host part {}", poi.id, poi.host));
        };
        let PartSpec::Link { a, b, .. } = host_part else {
            return Err(format!("poi {}: host {} is not a link", poi.id, poi.host));
        };
        let Some(a_pos) = joint_positions.get(a) else {
            return Err(format!("poi {}: missing host joint {}", poi.id, a));
        };
        let Some(b_pos) = joint_positions.get(b) else {
            return Err(format!("poi {}: missing host joint {}", poi.id, b));
        };
        let along = *b_pos - *a_pos;
        let normal = if along.length_squared() > 0.0 {
            vec2(-along.y, along.x).normalize()
        } else {
            vec2(0.0, 0.0)
        };
        let pos = *a_pos + along * poi.t + normal * poi.perp;
        positions.insert(poi.id.clone(), pos);
    }
    Ok(positions)
}

fn drive_plan(
    assembly: &AssemblySpec,
    drive_id: &str,
    drive: &DriveSpec,
) -> Result<DrivePlan, String> {
    match &drive.kind {
        DriveKindSpec::Angular { range, .. } => {
            let (start_value, end_value, playback) = match range {
                Some([start, end]) => match drive.sweep.direction {
                    SweepDirectionSpec::Forward => (*start, *end, PlaybackTraversal::Forward),
                    SweepDirectionSpec::Reverse => (*end, *start, PlaybackTraversal::Forward),
                    SweepDirectionSpec::PingPong => (*start, *end, PlaybackTraversal::PingPong),
                    SweepDirectionSpec::Clockwise => (*end, *start, PlaybackTraversal::PingPong),
                    SweepDirectionSpec::CounterClockwise => {
                        (*start, *end, PlaybackTraversal::PingPong)
                    }
                },
                None => match drive.sweep.direction {
                    SweepDirectionSpec::Clockwise | SweepDirectionSpec::Forward => {
                        (TAU, 0.0, PlaybackTraversal::Forward)
                    }
                    SweepDirectionSpec::CounterClockwise | SweepDirectionSpec::Reverse => {
                        (0.0, TAU, PlaybackTraversal::Forward)
                    }
                    SweepDirectionSpec::PingPong => {
                        return Err(format!(
                            "drive {drive_id}: full-rotation angular drives do not support PingPong without an explicit range"
                        ));
                    }
                },
            };
            Ok(DrivePlan {
                drive_id: drive_id.to_string(),
                parameter: DriveParameter::Angle,
                start_value,
                end_value,
                cycle_seconds: 2.0,
                playback,
            })
        }
        DriveKindSpec::Linear { slider, range } => {
            let Some(PartSpec::Slider {
                range: slider_range,
                ..
            }) = assembly.parts.get(slider)
            else {
                return Err(format!("drive {drive_id}: missing slider part {slider}"));
            };
            let [range_start, range_end] = range.unwrap_or(*slider_range);
            let (start_value, end_value, playback) = match drive.sweep.direction {
                SweepDirectionSpec::Forward | SweepDirectionSpec::CounterClockwise => {
                    (range_start, range_end, PlaybackTraversal::Forward)
                }
                SweepDirectionSpec::Reverse | SweepDirectionSpec::Clockwise => {
                    (range_end, range_start, PlaybackTraversal::Forward)
                }
                SweepDirectionSpec::PingPong => {
                    (range_start, range_end, PlaybackTraversal::PingPong)
                }
            };
            Ok(DrivePlan {
                drive_id: drive_id.to_string(),
                parameter: DriveParameter::SliderPosition,
                start_value,
                end_value,
                cycle_seconds: 1.5,
                playback,
            })
        }
    }
}

fn sample_drive_value(plan: &DrivePlan, u: f32) -> f32 {
    plan.start_value + (plan.end_value - plan.start_value) * u
}

fn draw_scene(draw: &Draw, win: Rect, model: &Model) {
    let fixture = current_fixture(model);
    draw_menu_layer(
        draw,
        win,
        model.headless,
        llm_turn_in_progress(model),
        &model.fixtures,
        model.selected_fixture,
    );
    let (spec_rect, render_rect) = content_rects(win);
    draw_chat_pane(draw, spec_rect, model, model.spec_scroll_px);
    draw_render_pane(
        draw,
        render_rect,
        fixture,
        model.playback_progress,
        model.live_trace_u,
        model.playback_paused,
        model.live_simulation.as_ref(),
        &model.trace_cycle_entities,
    );
}

fn write_screen_log_snapshot(model: &Model, win: Rect) {
    let snapshot = screen_text_snapshot(model, win);
    let mutex = SCREEN_LOG_LAST_SNAPSHOT.get_or_init(|| Mutex::new(String::new()));
    let mut last_snapshot = match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if *last_snapshot == snapshot {
        return;
    }
    if let Err(err) = fs::write(SCREEN_LOG_PATH, &snapshot) {
        eprintln!("linkage screen log error: failed to write {SCREEN_LOG_PATH}: {err}");
        return;
    }
    *last_snapshot = snapshot;
}

fn write_poi_trace_debug_json(model: &Model) {
    let snapshot = match poi_trace_context_json(model) {
        Ok(snapshot) => snapshot,
        Err(err) => {
            eprintln!("linkage poi trace log error: {err}");
            return;
        }
    };
    let mutex = POI_TRACE_LAST_SNAPSHOT.get_or_init(|| Mutex::new(String::new()));
    let mut last_snapshot = match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if *last_snapshot == snapshot {
        return;
    }
    if let Err(err) = fs::write(POI_TRACE_DEBUG_PATH, &snapshot) {
        eprintln!("linkage poi trace log error: failed to write {POI_TRACE_DEBUG_PATH}: {err}");
        return;
    }
    *last_snapshot = snapshot;
}

fn poi_trace_context_json(model: &Model) -> Result<String, String> {
    serde_json::to_string_pretty(&poi_trace_context_value(model))
        .map_err(|err| format!("failed to serialize poi trace context: {err}"))
}

fn poi_trace_context_value(model: &Model) -> serde_json::Value {
    let fixture_label = current_fixture(model).label.clone();
    let artifact = current_artifact(model);
    let source = if !model.latest_full_cycle_traces.is_empty() {
        "latest_completed_live_cycle"
    } else if artifact.is_some() {
        "artifact_sweep_fallback"
    } else {
        "none"
    };
    let poi_traces = if !model.latest_full_cycle_traces.is_empty() {
        live_cycle_trace_json(&model.latest_full_cycle_traces)
    } else if let Some(artifact) = artifact {
        artifact_point_paths_json(&artifact.telemetry.point_paths)
    } else {
        serde_json::json!({})
    };
    serde_json::json!({
        "fixture_label": fixture_label,
        "source": source,
        "playback_progress": round2(model.playback_progress),
        "live_trace_u": round2(model.live_trace_u),
        "poi_traces": poi_traces,
    })
}

fn live_cycle_trace_json(traces: &BTreeMap<String, Vec<TraceSample>>) -> serde_json::Value {
    let mut serialized = serde_json::Map::new();
    for (poi_id, samples) in traces {
        let values: Vec<serde_json::Value> =
            resample_trace_samples(samples, POI_TRACE_DEBUG_SAMPLES)
                .iter()
                .map(|sample| {
                    serde_json::json!({
                        "x": round2(sample.point.x),
                        "y": round2(sample.point.y),
                        "phase": round2(sample.phase),
                    })
                })
                .collect();
        serialized.insert(poi_id.clone(), serde_json::Value::Array(values));
    }
    serde_json::Value::Object(serialized)
}

fn artifact_point_paths_json(traces: &BTreeMap<String, Vec<Point2>>) -> serde_json::Value {
    let mut serialized = serde_json::Map::new();
    for (poi_id, points) in traces {
        let values: Vec<serde_json::Value> = resample_point_path(points, POI_TRACE_DEBUG_SAMPLES)
            .iter()
            .map(|sample| {
                serde_json::json!({
                    "x": round2(sample.point.x),
                    "y": round2(sample.point.y),
                    "phase": round2(sample.phase),
                })
            })
            .collect();
        serialized.insert(poi_id.clone(), serde_json::Value::Array(values));
    }
    serde_json::Value::Object(serialized)
}

fn resample_trace_samples(samples: &[TraceSample], count: usize) -> Vec<TraceSample> {
    if samples.is_empty() || count == 0 {
        return Vec::new();
    }
    if samples.len() == 1 {
        return (0..count)
            .map(|index| {
                let phase = if count == 1 {
                    samples[0].phase
                } else {
                    index as f32 / (count - 1) as f32
                };
                TraceSample {
                    point: samples[0].point,
                    phase,
                }
            })
            .collect();
    }

    (0..count)
        .map(|index| {
            let target_phase = if count == 1 {
                0.0
            } else {
                index as f32 / (count - 1) as f32
            };
            interpolate_trace_sample(samples, target_phase)
        })
        .collect()
}

fn resample_point_path(points: &[Point2], count: usize) -> Vec<TraceSample> {
    if points.is_empty() || count == 0 {
        return Vec::new();
    }
    let denom = points.len().saturating_sub(1).max(1) as f32;
    let samples: Vec<TraceSample> = points
        .iter()
        .enumerate()
        .map(|(index, point)| TraceSample {
            point: *point,
            phase: index as f32 / denom,
        })
        .collect();
    resample_trace_samples(&samples, count)
}

fn interpolate_trace_sample(samples: &[TraceSample], target_phase: f32) -> TraceSample {
    let first = samples[0];
    if target_phase <= first.phase {
        return TraceSample {
            point: first.point,
            phase: target_phase,
        };
    }

    for window in samples.windows(2) {
        let a = window[0];
        let b = window[1];
        if target_phase <= b.phase {
            let span = (b.phase - a.phase).max(0.000_1);
            let t = ((target_phase - a.phase) / span).clamp(0.0, 1.0);
            return TraceSample {
                point: pt2(
                    a.point.x + (b.point.x - a.point.x) * t,
                    a.point.y + (b.point.y - a.point.y) * t,
                ),
                phase: target_phase,
            };
        }
    }

    let last = *samples.last().expect("non-empty trace samples");
    TraceSample {
        point: last.point,
        phase: target_phase,
    }
}

fn round2(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}

fn screen_text_snapshot(model: &Model, win: Rect) -> String {
    collect_screen_text_lines(model, win).join("\n")
}

fn collect_screen_text_lines(model: &Model, win: Rect) -> Vec<String> {
    let mut lines = Vec::new();
    let fixture = current_fixture(model);
    let (spec_rect, _) = content_rects(win);

    lines.push("LINKAGE".to_string());
    lines.push(if model.headless {
        "HEADLESS P3".to_string()
    } else {
        "LIVE P3".to_string()
    });
    lines.push(if llm_turn_in_progress(model) {
        "GENERATING".to_string()
    } else {
        "REGENERATE".to_string()
    });
    for fixture in &model.fixtures {
        lines.push(fixture.label.clone());
    }

    lines.push("LLM CHAT".to_string());
    lines.extend(
        chat_display_lines(model, spec_rect)
            .into_iter()
            .map(|(line, _)| line)
            .filter(|line| !line.is_empty()),
    );
    lines.push(if model.chat_input.is_empty() {
        "Type a prompt and press Enter".to_string()
    } else {
        model.chat_input.clone()
    });
    lines.push(if llm_turn_in_progress(model) {
        "WAIT".to_string()
    } else {
        "SEND".to_string()
    });

    lines.push(match fixture.status {
        FixtureStatus::Solved(_) => "LIVE INTEGRATOR".to_string(),
        FixtureStatus::ValidationError(_) => "VALIDATION FAILURE".to_string(),
        FixtureStatus::RelaxationError(_) => "RELAXATION FAILURE".to_string(),
        FixtureStatus::GenerationError(_) => "GENERATION FAILURE".to_string(),
    });
    lines.push(render_subtitle_text(fixture, model.playback_paused).to_string());

    match &fixture.status {
        FixtureStatus::Solved(artifact) => {
            if let Some(live) = model.live_simulation.as_ref() {
                let drive_status = live
                    .frame
                    .drive_values
                    .iter()
                    .next()
                    .map(|(_, value)| {
                        let label = match artifact
                            .assembly
                            .drives
                            .values()
                            .next()
                            .map(|drive| &drive.kind)
                        {
                            Some(DriveKindSpec::Angular { .. }) => "angle",
                            Some(DriveKindSpec::Linear { .. }) => "slider",
                            None => "drive",
                        };
                        format!("{label} {:.2}", value)
                    })
                    .unwrap_or_else(|| "drive n/a".to_string());
                lines.push(format!(
                    "LIVE  |  t {:.2}  |  u {:.2}  |  {drive_status}  |  err {:.2}{}",
                    model.playback_progress,
                    model.live_trace_u,
                    live.max_constraint_error,
                    if live.settled { "" } else { "  |  loose" }
                ));
                if let Some(note) = artifact.telemetry.notes.first() {
                    lines.push(note.clone());
                }
            } else {
                lines.push("LIVE".to_string());
                lines.push("runtime state has not been seeded yet".to_string());
            }
        }
        FixtureStatus::ValidationError(message) => {
            lines.push("VALIDATOR".to_string());
            lines.push(message.clone());
        }
        FixtureStatus::RelaxationError(message) => {
            lines.push("RELAX".to_string());
            lines.push(message.clone());
        }
        FixtureStatus::GenerationError(message) => {
            lines.push("OPENAI".to_string());
            lines.push(message.clone());
        }
    }

    lines.into_iter().filter(|line| !line.is_empty()).collect()
}

fn render_subtitle_text(fixture: &FixturePresentation, playback_paused: bool) -> &'static str {
    match fixture.status {
        FixtureStatus::Solved(_) => {
            let artifact = match &fixture.status {
                FixtureStatus::Solved(artifact) => artifact,
                _ => unreachable!(),
            };
            if playback_paused {
                "space resumes drive motion"
            } else if artifact.telemetry.unsettled_samples > 0 {
                "drive runs live through the relaxed particle system"
            } else {
                "space pauses drive motion"
            }
        }
        FixtureStatus::ValidationError(_) => "validator rejected the selected fixture",
        FixtureStatus::RelaxationError(_) => "constraint relaxation did not settle",
        FixtureStatus::GenerationError(_) => {
            "startup request failed; local fixtures remain available"
        }
    }
}

fn draw_menu_layer(
    draw: &Draw,
    win: Rect,
    headless: bool,
    startup_generating: bool,
    fixtures: &[FixturePresentation],
    selected_fixture: usize,
) {
    let bar_y = win.top() - MENU_BAR_H * 0.5;
    draw.rect()
        .x_y(0.0, bar_y)
        .w_h(win.w(), MENU_BAR_H)
        .color(rgba(0.02, 0.025, 0.035, 0.96));

    draw.text("LINKAGE")
        .x_y(win.left() + 78.0, win.top() - 24.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82));

    let mode_label = if headless { "HEADLESS P3" } else { "LIVE P3" };
    draw.text(mode_label)
        .x_y(win.right() - 78.0, win.top() - 24.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.68));

    let regenerate_rect = startup_regenerate_rect(win);
    draw.rect()
        .x_y(regenerate_rect.x(), regenerate_rect.y())
        .w_h(regenerate_rect.w(), regenerate_rect.h())
        .color(if startup_generating {
            rgba(1.0, 1.0, 1.0, 0.08)
        } else {
            rgba(1.0, 1.0, 1.0, 0.14)
        });
    draw.text(if startup_generating {
        "GENERATING"
    } else {
        "REGENERATE"
    })
    .x_y(regenerate_rect.x(), regenerate_rect.y() + 1.0)
    .font_size(8)
    .color(if startup_generating {
        rgba(1.0, 1.0, 1.0, 0.38)
    } else {
        rgba(1.0, 1.0, 1.0, 0.78)
    });

    let start_x = win.left() + 220.0;
    let step = 94.0;
    let indicator_y = win.top() - 36.0;
    for (index, fixture) in fixtures.iter().enumerate() {
        let x = start_x + index as f32 * step;
        draw.text(&fixture.label)
            .x_y(x, win.top() - 24.0)
            .font_size(8)
            .color(if index == selected_fixture {
                rgba(1.0, 1.0, 1.0, 0.82)
            } else {
                rgba(1.0, 1.0, 1.0, 0.42)
            });

        if index == selected_fixture {
            draw.line()
                .start(pt2(x - 24.0, indicator_y))
                .end(pt2(x + 24.0, indicator_y))
                .color(rgba(1.0, 1.0, 1.0, 0.92))
                .weight(2.0);
        }
    }
}

fn startup_regenerate_rect(win: Rect) -> Rect {
    Rect::from_xy_wh(pt2(win.right() - 208.0, win.top() - 24.0), vec2(92.0, 22.0))
}

fn draw_chat_pane(draw: &Draw, rect: Rect, model: &Model, scroll_px: f32) {
    let text_w = rect.w() - PANE_TEXT_INSET * 2.0;
    let text_x = rect.left() + PANE_TEXT_INSET + text_w * 0.5;
    let body_top = rect.top() - 42.0;
    let input_rect = chat_input_rect_from_spec(rect);
    let send_rect = chat_send_rect_from_spec(rect);
    let body_bottom = input_rect.top() + 16.0;

    draw.text("LLM CHAT")
        .x_y(text_x, rect.top() - 18.0)
        .w_h(text_w, PANE_LINE_H)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82))
        .left_justify();

    let lines = chat_display_lines(model, rect);
    let start_line = (scroll_px / PANE_LINE_H).floor() as usize;
    let intra_line_offset = scroll_px % PANE_LINE_H;
    let mut y = body_top + intra_line_offset;

    for (line, color) in lines.iter().skip(start_line) {
        if y < body_bottom {
            break;
        }
        if y > body_top {
            y -= PANE_LINE_H;
            continue;
        }
        draw.text(line)
            .x_y(text_x, y)
            .w_h(text_w, PANE_LINE_H)
            .font_size(8)
            .color(*color)
            .left_justify();
        y -= PANE_LINE_H;
    }

    draw.rect()
        .x_y(input_rect.x(), input_rect.y())
        .w_h(input_rect.w(), input_rect.h())
        .color(if model.chat_input_focused {
            rgba(1.0, 1.0, 1.0, 0.10)
        } else {
            rgba(1.0, 1.0, 1.0, 0.05)
        });
    let input_text = if model.chat_input.is_empty() {
        "Type a prompt and press Enter"
    } else {
        &model.chat_input
    };
    draw.text(input_text)
        .x_y(
            input_rect.left() + 12.0 + (input_rect.w() - 24.0) * 0.5,
            input_rect.y(),
        )
        .w_h(input_rect.w() - 24.0, input_rect.h() - 10.0)
        .font_size(8)
        .color(if model.chat_input.is_empty() {
            rgba(1.0, 1.0, 1.0, 0.32)
        } else {
            rgba(1.0, 1.0, 1.0, 0.78)
        })
        .left_justify();

    draw.rect()
        .x_y(send_rect.x(), send_rect.y())
        .w_h(send_rect.w(), send_rect.h())
        .color(if llm_turn_in_progress(model) {
            rgba(1.0, 1.0, 1.0, 0.08)
        } else {
            rgba(1.0, 1.0, 1.0, 0.14)
        });
    draw.text(if llm_turn_in_progress(model) {
        "WAIT"
    } else {
        "SEND"
    })
    .x_y(send_rect.x(), send_rect.y() + 1.0)
    .font_size(8)
    .color(if llm_turn_in_progress(model) {
        rgba(1.0, 1.0, 1.0, 0.38)
    } else {
        rgba(1.0, 1.0, 1.0, 0.78)
    });

    let max_scroll = max_chat_scroll_px_for_rect(model, rect);
    if max_scroll > 0.0 {
        let track_h = (body_top - body_bottom).max(1.0);
        let thumb_h = (track_h * (track_h / (track_h + max_scroll))).clamp(28.0, track_h);
        let thumb_travel = (track_h - thumb_h).max(0.0);
        let thumb_t = if max_scroll > 0.0 {
            scroll_px / max_scroll
        } else {
            0.0
        };
        let thumb_center_y = body_top - thumb_h * 0.5 - thumb_travel * thumb_t;
        let track_x = rect.right() - 14.0;

        draw.rect()
            .x_y(track_x, (body_top + body_bottom) * 0.5)
            .w_h(2.0, track_h)
            .color(rgba(1.0, 1.0, 1.0, 0.10));
        draw.rect()
            .x_y(track_x, thumb_center_y)
            .w_h(3.0, thumb_h)
            .color(rgba(1.0, 1.0, 1.0, 0.36));
    }
}

fn draw_render_pane(
    draw: &Draw,
    rect: Rect,
    fixture: &FixturePresentation,
    playback_progress: f32,
    live_trace_u: f32,
    playback_paused: bool,
    live_simulation: Option<&LiveSimulationState>,
    trace_cycle_entities: &[TraceCycleEntity],
) {
    let text_w = rect.w() - PANE_TEXT_INSET * 2.0;
    let text_x = rect.left() + PANE_TEXT_INSET + text_w * 0.5;

    let title = match fixture.status {
        FixtureStatus::Solved(_) => "LIVE INTEGRATOR",
        FixtureStatus::ValidationError(_) => "VALIDATION FAILURE",
        FixtureStatus::RelaxationError(_) => "RELAXATION FAILURE",
        FixtureStatus::GenerationError(_) => "GENERATION FAILURE",
    };
    draw.text(title)
        .x_y(text_x, rect.top() - 18.0)
        .w_h(text_w, PANE_LINE_H)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82))
        .left_justify();

    let subtitle = match fixture.status {
        FixtureStatus::Solved(_) => {
            let artifact = match &fixture.status {
                FixtureStatus::Solved(artifact) => artifact,
                _ => unreachable!(),
            };
            if playback_paused {
                "space resumes drive motion"
            } else if artifact.telemetry.unsettled_samples > 0 {
                "drive runs live through the relaxed particle system"
            } else {
                "space pauses drive motion"
            }
        }
        FixtureStatus::ValidationError(_) => "validator rejected the selected fixture",
        FixtureStatus::RelaxationError(_) => "constraint relaxation did not settle",
        FixtureStatus::GenerationError(_) => {
            "startup request failed; local fixtures remain available"
        }
    };
    draw.text(subtitle)
        .x_y(text_x, rect.top() - 34.0)
        .w_h(text_w, PANE_LINE_H)
        .font_size(8)
        .color(rgba(1.0, 1.0, 1.0, 0.40))
        .left_justify();

    let local_draw = draw.x_y(rect.x(), rect.y() - RENDER_VIEW_Y_OFFSET);
    let local_rect = Rect::from_w_h(rect.w() - 28.0, rect.h() - 72.0);
    match &fixture.status {
        FixtureStatus::Solved(artifact) => {
            draw_grid_local(&local_draw, local_rect);
            let Some(live) = live_simulation else {
                draw_error_state(
                    &local_draw,
                    local_rect,
                    "LIVE",
                    "runtime state has not been seeded yet",
                );
                return;
            };
            let solved = solved_assembly_for_frame(&artifact.assembly, &live.frame);
            let bounds = artifact_bounds(artifact);
            let map = make_world_to_local(local_rect, bounds);
            draw_trace_cycle_entities_local(
                &local_draw,
                trace_cycle_entities,
                artifact,
                playback_progress,
                &map,
            );
            draw_poi_traces_local(&local_draw, artifact, live, playback_progress, &map);
            draw_solution_local(&local_draw, local_rect, &solved, &map);
            let drive_status = live
                .frame
                .drive_values
                .iter()
                .next()
                .map(|(_, value)| {
                    let label = match artifact
                        .assembly
                        .drives
                        .values()
                        .next()
                        .map(|drive| &drive.kind)
                    {
                        Some(DriveKindSpec::Angular { .. }) => "angle",
                        Some(DriveKindSpec::Linear { .. }) => "slider",
                        None => "drive",
                    };
                    format!("{label} {:.2}", value)
                })
                .unwrap_or_else(|| "drive n/a".to_string());
            let status = format!(
                "LIVE  |  t {:.2}  |  u {:.2}  |  {drive_status}  |  err {:.2}{}",
                playback_progress,
                live_trace_u,
                live.max_constraint_error,
                if live.settled { "" } else { "  |  loose" }
            );
            draw.text(&status)
                .x_y(text_x, rect.top() - 50.0)
                .w_h(text_w, PANE_LINE_H)
                .font_size(8)
                .color(rgba(1.0, 1.0, 1.0, 0.34))
                .left_justify();
            if let Some(note) = artifact.telemetry.notes.first() {
                draw.text(note)
                    .x_y(text_x, rect.top() - 64.0)
                    .w_h(text_w, PANE_LINE_H * 2.0)
                    .font_size(8)
                    .color(rgba(1.0, 1.0, 1.0, 0.46))
                    .left_justify();
            }
        }
        FixtureStatus::ValidationError(message) => {
            draw_error_state(&local_draw, local_rect, "VALIDATOR", message);
        }
        FixtureStatus::RelaxationError(message) => {
            draw_error_state(&local_draw, local_rect, "RELAX", message);
        }
        FixtureStatus::GenerationError(message) => {
            draw_error_state(&local_draw, local_rect, "OPENAI", message);
        }
    }
}

fn draw_grid_local(draw: &Draw, rect: Rect) {
    let cols = (rect.w() / GRID_STEP).ceil() as i32;
    for col in 0..=cols {
        let x = rect.left() + col as f32 * GRID_STEP;
        let is_major = col % 4 == 0;
        let alpha = if is_major { 0.20 } else { 0.06 };
        let weight = if is_major { 1.25 } else { 1.0 };
        draw.line()
            .start(pt2(x, rect.bottom()))
            .end(pt2(x, rect.top()))
            .color(rgba(1.0, 1.0, 1.0, alpha))
            .weight(weight);
    }

    let rows = (rect.h() / GRID_STEP).ceil() as i32;
    for row in 0..=rows {
        let y = rect.bottom() + row as f32 * GRID_STEP;
        let is_major = row % 4 == 0;
        let alpha = if is_major { 0.20 } else { 0.06 };
        let weight = if is_major { 1.25 } else { 1.0 };
        draw.line()
            .start(pt2(rect.left(), y))
            .end(pt2(rect.right(), y))
            .color(rgba(1.0, 1.0, 1.0, alpha))
            .weight(weight);
    }
}

fn draw_solution_local<F>(draw: &Draw, _rect: Rect, solved: &SolvedAssembly, map: &F)
where
    F: Fn(Point2) -> Point2,
{
    draw_sliders_local(draw, solved, map);

    for link in &solved.links {
        let a = map(solved.joints[&link.a].position);
        let b = map(solved.joints[&link.b].position);
        draw.line()
            .start(a)
            .end(b)
            .weight(1.5)
            .color(rgba(1.0, 1.0, 1.0, 0.65));
    }

    for (joint_id, joint) in &solved.joints {
        let p = map(joint.position);
        if joint.fixed {
            draw.rect()
                .x_y(p.x, p.y)
                .w_h(12.0, 12.0)
                .color(rgba(1.0, 1.0, 1.0, 0.80));
        } else {
            let _ = joint_id;
            draw.ellipse()
                .x_y(p.x, p.y)
                .w_h(12.0, 12.0)
                .color(rgba(1.0, 1.0, 1.0, 0.80));
        }
    }

    for (index, poi) in solved.pois.iter().enumerate() {
        let p = map(poi.position);
        draw.ellipse()
            .x_y(p.x, p.y)
            .w_h(8.0, 8.0)
            .color(poi_color(index, 0.92));
    }
}

fn draw_error_state(draw: &Draw, rect: Rect, source: &str, message: &str) {
    draw_grid_local(draw, rect);
    let text_w = rect.w() - 96.0;
    let text_x = rect.left() + 28.0 + text_w * 0.5;
    let body_h = 108.0;

    draw.text(source)
        .x_y(text_x, rect.top() - 36.0)
        .w_h(text_w, 16.0)
        .font_size(9)
        .color(rgba(1.0, 0.48, 0.48, 0.88))
        .left_justify();

    draw.text(message)
        .x_y(text_x, rect.top() - 86.0)
        .w_h(text_w, body_h)
        .font_size(8)
        .color(rgba(1.0, 1.0, 1.0, 0.70))
        .left_justify();
}

fn draw_sliders_local<F>(draw: &Draw, solved: &SolvedAssembly, map: &F)
where
    F: Fn(Point2) -> Point2,
{
    let outer_h = 22.0;
    let inner_h = 12.0;
    for slider in &solved.sliders {
        let start = map(slider.start);
        let end = map(slider.end);
        draw.line()
            .start(start)
            .end(end)
            .weight(outer_h)
            .color(rgba(1.0, 1.0, 1.0, 0.22));
        draw.line()
            .start(start)
            .end(end)
            .weight(inner_h)
            .color(rgba(0.02, 0.025, 0.035, 1.0));

        if let Some(joint) = solved.joints.get(&slider.joint) {
            let slider_joint = map(joint.position);
            draw.ellipse()
                .x_y(slider_joint.x, slider_joint.y)
                .w_h(12.0, 12.0)
                .color(rgba(1.0, 1.0, 1.0, 0.92));
        }
    }
}

fn playback_sample_u(progress: f32, traversal: PlaybackTraversal) -> f32 {
    let phase = progress.rem_euclid(1.0);
    match traversal {
        PlaybackTraversal::Forward => phase,
        // User-facing ping-pong is eased at the turnarounds on purpose.
        PlaybackTraversal::PingPong => 0.5 - 0.5 * (phase * TAU).cos(),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn sampled_frame(artifact: &SweepArtifact, sample_u: f32) -> &SolvedFrame {
    let len = artifact.frames.len().max(1);
    let idx = ((sample_u.clamp(0.0, 0.999_999)) * len as f32).floor() as usize;
    &artifact.frames[idx.min(len - 1)]
}

fn solved_assembly_for_frame(assembly: &AssemblySpec, frame: &SolvedFrame) -> SolvedAssembly {
    let joints = assembly
        .joints
        .iter()
        .filter_map(|(joint_id, joint_spec)| {
            frame.joint_positions.get(joint_id).map(|position| {
                (
                    joint_id.clone(),
                    SolvedJoint {
                        position: *position,
                        fixed: matches!(joint_spec, JointSpec::Fixed { .. }),
                    },
                )
            })
        })
        .collect();

    let links = assembly
        .parts
        .iter()
        .filter_map(|(_, part)| match part {
            PartSpec::Link { a, b, .. } => Some(SolvedLink {
                a: a.clone(),
                b: b.clone(),
            }),
            PartSpec::Slider { .. } => None,
        })
        .collect();

    let sliders = assembly
        .parts
        .values()
        .filter_map(|part| match part {
            PartSpec::Slider {
                joint,
                axis_origin,
                axis_dir,
                range,
            } => Some(SolvedSlider {
                start: pt2(
                    axis_origin[0] + axis_dir[0] * range[0],
                    axis_origin[1] + axis_dir[1] * range[0],
                ),
                end: pt2(
                    axis_origin[0] + axis_dir[0] * range[1],
                    axis_origin[1] + axis_dir[1] * range[1],
                ),
                joint: joint.clone(),
            }),
            PartSpec::Link { .. } => None,
        })
        .collect();

    let pois = frame
        .poi_positions
        .iter()
        .map(|(_, position)| SolvedPoi {
            position: *position,
        })
        .collect();

    SolvedAssembly {
        joints,
        links,
        sliders,
        pois,
    }
}

fn artifact_bounds(artifact: &SweepArtifact) -> Rect {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for frame in &artifact.frames {
        for position in frame.joint_positions.values() {
            min_x = min_x.min(position.x);
            max_x = max_x.max(position.x);
            min_y = min_y.min(position.y);
            max_y = max_y.max(position.y);
        }
        for position in frame.poi_positions.values() {
            min_x = min_x.min(position.x);
            max_x = max_x.max(position.x);
            min_y = min_y.min(position.y);
            max_y = max_y.max(position.y);

            if let Some((direction, advance_per_cycle)) = rolling_paper_config(artifact) {
                // Reserve one cycle of paper travel so enabling rolling paper does not
                // immediately squeeze the mechanism framing. Additional completed cycles
                // are expected to scroll out of view rather than expanding the camera.
                let offset = direction * advance_per_cycle;
                let rolled = *position + offset;
                min_x = min_x.min(rolled.x);
                max_x = max_x.max(rolled.x);
                min_y = min_y.min(rolled.y);
                max_y = max_y.max(rolled.y);
            }
        }
    }

    for part in artifact.assembly.parts.values() {
        if let PartSpec::Slider {
            axis_origin,
            axis_dir,
            range,
            ..
        } = part
        {
            let start = pt2(
                axis_origin[0] + axis_dir[0] * range[0],
                axis_origin[1] + axis_dir[1] * range[0],
            );
            let end = pt2(
                axis_origin[0] + axis_dir[0] * range[1],
                axis_origin[1] + axis_dir[1] * range[1],
            );
            min_x = min_x.min(start.x).min(end.x);
            max_x = max_x.max(start.x).max(end.x);
            min_y = min_y.min(start.y).min(end.y);
            max_y = max_y.max(start.y).max(end.y);
        }
    }

    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Rect::from_corners(pt2(-80.0, -80.0), pt2(80.0, 80.0));
    }

    Rect::from_corners(
        pt2(min_x - 20.0, min_y - 20.0),
        pt2(max_x + 20.0, max_y + 20.0),
    )
}

fn make_world_to_local(rect: Rect, bounds: Rect) -> impl Fn(Point2) -> Point2 {
    let pane_w = rect.w() - 120.0;
    let pane_h = rect.h() - 120.0;
    let world_w = (bounds.right() - bounds.left()).max(1.0);
    let world_h = (bounds.top() - bounds.bottom()).max(1.0);
    let scale = (pane_w / world_w).min(pane_h / world_h);
    let world_center = pt2(
        (bounds.left() + bounds.right()) * 0.5,
        (bounds.bottom() + bounds.top()) * 0.5,
    );
    move |point: Point2| {
        pt2(
            (point.x - world_center.x) * scale,
            (point.y - world_center.y) * scale,
        )
    }
}

fn draw_poi_traces_local<F>(
    draw: &Draw,
    artifact: &SweepArtifact,
    live: &LiveSimulationState,
    current_progress: f32,
    map: &F,
) where
    F: Fn(Point2) -> Point2,
{
    if let Some((direction, advance_per_cycle)) = rolling_paper_config(artifact) {
        let current_phase = current_progress.rem_euclid(1.0);
        for (index, (_poi_id, path)) in live.cycle_samples.iter().enumerate() {
            for (point_index, window) in path.windows(2).enumerate() {
                if point_index % 2 == 1 {
                    continue;
                }
                let a = map(rolling_paper_phase_position(
                    window[0].point,
                    direction,
                    advance_per_cycle,
                    current_phase,
                    window[0].phase,
                ));
                let b = map(rolling_paper_phase_position(
                    window[1].point,
                    direction,
                    advance_per_cycle,
                    current_phase,
                    window[1].phase,
                ));
                draw.line()
                    .start(a)
                    .end(b)
                    .weight(1.0)
                    .color(poi_color(index, 0.72));
            }
        }
        return;
    }

    for (index, (_poi_id, path)) in live.point_trails.iter().enumerate() {
        for (point_index, window) in path.windows(2).enumerate() {
            if point_index % 2 == 1 {
                continue;
            }
            let a = map(window[0]);
            let b = map(window[1]);
            draw.line()
                .start(a)
                .end(b)
                .weight(1.0)
                .color(poi_color(index, 0.72));
        }
    }
}

fn draw_trace_cycle_entities_local<F>(
    draw: &Draw,
    entities: &[TraceCycleEntity],
    artifact: &SweepArtifact,
    current_progress: f32,
    map: &F,
) where
    F: Fn(Point2) -> Point2,
{
    let Some((direction, advance_per_cycle)) = rolling_paper_config(artifact) else {
        return;
    };

    for entity in entities {
        let translation = direction
            * (advance_per_cycle * (current_progress - entity.completion_progress).max(0.0));
        for (point_index, window) in entity.points.windows(2).enumerate() {
            if point_index % 2 == 1 {
                continue;
            }
            let a = map(window[0] + translation);
            let b = map(window[1] + translation);
            draw.line()
                .start(a)
                .end(b)
                .weight(1.0)
                .color(poi_color(entity.color_index, 0.72));
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn spawn_trace_cycle_entities(
    entities: &mut Vec<TraceCycleEntity>,
    artifact: &SweepArtifact,
    cycle_index: u32,
) {
    let Some((direction, advance_per_cycle)) = rolling_paper_config(artifact) else {
        return;
    };

    for (color_index, path) in artifact.telemetry.point_paths.values().enumerate() {
        if path.len() < 2 {
            continue;
        }
        let denom = (path.len() - 1) as f32;
        let points = path
            .iter()
            .enumerate()
            .map(|(point_index, point)| {
                let u = point_index as f32 / denom;
                rolling_paper_position(*point, direction, advance_per_cycle, 1.0, u)
            })
            .collect();
        entities.push(TraceCycleEntity {
            completion_progress: cycle_index as f32 + 1.0,
            color_index,
            points,
        });
    }

    if entities.len() > MAX_TRACE_CYCLE_ENTITIES {
        let excess = entities.len() - MAX_TRACE_CYCLE_ENTITIES;
        entities.drain(..excess);
    }
}

fn spawn_trace_cycle_entities_from_samples(
    entities: &mut Vec<TraceCycleEntity>,
    cycle_samples: &BTreeMap<String, Vec<TraceSample>>,
    artifact: &SweepArtifact,
    cycle_index: u32,
) {
    let Some((direction, advance_per_cycle)) = rolling_paper_config(artifact) else {
        return;
    };

    for (color_index, path) in cycle_samples.values().enumerate() {
        if path.len() < 2 {
            continue;
        }
        let points = path
            .iter()
            .map(|sample| {
                rolling_paper_phase_position(
                    sample.point,
                    direction,
                    advance_per_cycle,
                    1.0,
                    sample.phase,
                )
            })
            .collect();
        entities.push(TraceCycleEntity {
            completion_progress: cycle_index as f32 + 1.0,
            color_index,
            points,
        });
    }

    if entities.len() > MAX_TRACE_CYCLE_ENTITIES {
        let excess = entities.len() - MAX_TRACE_CYCLE_ENTITIES;
        entities.drain(..excess);
    }
}

fn trace_cycle_entity_visible<F>(
    entity: &TraceCycleEntity,
    current_progress: f32,
    artifact: &SweepArtifact,
    map: &F,
    render_offset: Vec2,
    window_rect: Rect,
) -> bool
where
    F: Fn(Point2) -> Point2,
{
    let Some((direction, advance_per_cycle)) = rolling_paper_config(artifact) else {
        return false;
    };
    let translation =
        direction * (advance_per_cycle * (current_progress - entity.completion_progress).max(0.0));
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for point in &entity.points {
        let mapped = map(*point + translation) + render_offset;
        min_x = min_x.min(mapped.x);
        max_x = max_x.max(mapped.x);
        min_y = min_y.min(mapped.y);
        max_y = max_y.max(mapped.y);
    }

    !(max_x < window_rect.left()
        || min_x > window_rect.right()
        || max_y < window_rect.bottom()
        || min_y > window_rect.top())
}

fn rolling_paper_config(artifact: &SweepArtifact) -> Option<(Vec2, f32)> {
    match artifact
        .assembly
        .visualization
        .as_ref()
        .and_then(|visualization| visualization.trace_model.as_ref())
    {
        Some(TraceModelSpec::RollingPaper {
            direction,
            advance_per_cycle,
        }) => Some((rolling_paper_direction(direction), *advance_per_cycle)),
        _ => None,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn rolling_paper_position(
    point: Point2,
    direction: Vec2,
    advance_per_cycle: f32,
    current_u: f32,
    mark_u: f32,
) -> Point2 {
    point + direction * (advance_per_cycle * (current_u - mark_u).max(0.0))
}

fn rolling_paper_phase_position(
    point: Point2,
    direction: Vec2,
    advance_per_cycle: f32,
    current_phase: f32,
    mark_phase: f32,
) -> Point2 {
    point + direction * (advance_per_cycle * (current_phase - mark_phase).max(0.0))
}

fn rolling_paper_direction(direction: &PaperDirectionSpec) -> Vec2 {
    match direction {
        PaperDirectionSpec::Up => vec2(0.0, 1.0),
        PaperDirectionSpec::Down => vec2(0.0, -1.0),
        PaperDirectionSpec::Left => vec2(-1.0, 0.0),
        PaperDirectionSpec::Right => vec2(1.0, 0.0),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn update_live_trace_state(
    playback_progress: f32,
    previous_cycle: u32,
    previous_max_u: f32,
    traversal: PlaybackTraversal,
) -> (u32, f32) {
    let cycle = playback_progress.floor().max(0.0) as u32;
    let phase = playback_progress.rem_euclid(1.0);
    let current_u = playback_sample_u(phase, traversal);
    let max_u = if cycle == previous_cycle {
        previous_max_u.max(current_u)
    } else {
        current_u
    };
    (cycle, max_u)
}

fn poi_color(index: usize, alpha: f32) -> LinSrgba {
    const PALETTE: [(f32, f32, f32); 4] = [
        (0.34, 0.78, 0.98),
        (0.98, 0.68, 0.28),
        (0.70, 0.88, 0.36),
        (0.96, 0.46, 0.70),
    ];
    let (r, g, b) = PALETTE[index % PALETTE.len()];
    rgba(r, g, b, alpha).into_linear()
}

fn wait_for_capture_flush(path: &PathBuf) -> bool {
    let mut previous_len = 0;
    let mut stable_reads = 0;

    for _ in 0..2400 {
        if let Ok(metadata) = std::fs::metadata(path) {
            let len = metadata.len();
            if len > 8 {
                if let Ok(bytes) = std::fs::read(path) {
                    let is_png = bytes.starts_with(b"\x89PNG\r\n\x1a\n");
                    if is_png && len == previous_len {
                        stable_reads += 1;
                        if stable_reads >= 3 {
                            return true;
                        }
                    } else {
                        stable_reads = 0;
                    }
                }
            }
            previous_len = len;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }

    false
}

fn scroll_spec(model: &mut Model, win: Rect, delta_px: f32) {
    let next = model.spec_scroll_px + delta_px;
    model.spec_scroll_px = next.clamp(0.0, max_chat_scroll_px(model, win));
}

fn visible_chat_body_height(win: Rect) -> f32 {
    let (spec_rect, _) = content_rects(win);
    visible_chat_body_height_for_rect(spec_rect)
}

fn max_chat_scroll_px(model: &Model, win: Rect) -> f32 {
    let (spec_rect, _) = content_rects(win);
    max_chat_scroll_px_for_rect(model, spec_rect)
}

fn visible_chat_body_height_for_rect(spec_rect: Rect) -> f32 {
    let input_rect = chat_input_rect_from_spec(spec_rect);
    (spec_rect.top() - 42.0) - (input_rect.top() + 16.0)
}

fn max_chat_scroll_px_for_rect(model: &Model, spec_rect: Rect) -> f32 {
    let available = visible_chat_body_height_for_rect(spec_rect).max(0.0);
    let content = chat_display_lines(model, spec_rect).len() as f32 * PANE_LINE_H;
    (content - available).max(0.0)
}

fn chat_input_rect(win: Rect) -> Rect {
    let (spec_rect, _) = content_rects(win);
    chat_input_rect_from_spec(spec_rect)
}

fn chat_input_rect_from_spec(spec_rect: Rect) -> Rect {
    Rect::from_xy_wh(
        pt2(
            spec_rect.left() + (spec_rect.w() - 146.0) * 0.5 + 4.0,
            spec_rect.bottom() + 26.0,
        ),
        vec2(spec_rect.w() - 126.0, 34.0),
    )
}

fn chat_send_rect(win: Rect) -> Rect {
    let (spec_rect, _) = content_rects(win);
    chat_send_rect_from_spec(spec_rect)
}

fn chat_send_rect_from_spec(spec_rect: Rect) -> Rect {
    Rect::from_xy_wh(
        pt2(spec_rect.right() - 44.0, spec_rect.bottom() + 26.0),
        vec2(74.0, 34.0),
    )
}

fn chat_display_lines(model: &Model, rect: Rect) -> Vec<(String, LinSrgba)> {
    let wrap_chars = ((rect.w() - 40.0) / 6.2).floor().max(24.0) as usize;
    let mut lines = Vec::new();
    for entry in &model.chat_entries {
        let label = match entry.role {
            ChatRole::System => "SYSTEM",
            ChatRole::User => "YOU",
            ChatRole::Assistant => "ASSISTANT",
            ChatRole::Tool => "TOOL",
            ChatRole::Error => "ERROR",
        };
        let color = match entry.role {
            ChatRole::System => rgba(1.0, 1.0, 1.0, 0.38).into_linear(),
            ChatRole::User => rgba(0.72, 0.88, 1.0, 0.88).into_linear(),
            ChatRole::Assistant => rgba(1.0, 1.0, 1.0, 0.74).into_linear(),
            ChatRole::Tool => rgba(0.98, 0.76, 0.40, 0.82).into_linear(),
            ChatRole::Error => rgba(1.0, 0.48, 0.48, 0.88).into_linear(),
        };
        let wrapped = wrap_text(&format!("{label}: {}", entry.text), wrap_chars);
        for line in wrapped {
            lines.push((line, color));
        }
        lines.push(("".to_string(), rgba(1.0, 1.0, 1.0, 0.0).into_linear()));
    }
    lines
}

fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let next_len = if current.is_empty() {
            word.len()
        } else {
            current.len() + 1 + word.len()
        };
        if next_len > max_chars && !current.is_empty() {
            lines.push(current);
            current = word.to_string();
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        vec![text.to_string()]
    } else {
        lines
    }
}

fn content_rects(win: Rect) -> (Rect, Rect) {
    let content_top = win.top() - MENU_BAR_H - PANE_MARGIN;
    let content_bottom = win.bottom() + PANE_MARGIN;
    let content_h = content_top - content_bottom;
    let content_w = win.w() - PANE_MARGIN * 2.0;
    let spec_w = content_w * SPEC_RATIO;
    let render_w = content_w - spec_w - PANE_GUTTER;
    let spec_x = win.left() + PANE_MARGIN + spec_w * 0.5;
    let render_x = spec_x + spec_w * 0.5 + PANE_GUTTER + render_w * 0.5;
    let content_y = (content_top + content_bottom) * 0.5;
    let spec_rect = Rect::from_xy_wh(pt2(spec_x, content_y), vec2(spec_w, content_h));
    let render_rect = Rect::from_xy_wh(pt2(render_x, content_y), vec2(render_w, content_h));
    (spec_rect, render_rect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slider_crank_sweep_frames_are_deterministic() {
        let first = build_sweep_artifact(slider_crank_fixture(), 1).expect("first sweep");
        let second = build_sweep_artifact(slider_crank_fixture(), 1).expect("second sweep");
        assert_eq!(
            frame_signature(&first.frames),
            frame_signature(&second.frames)
        );
    }

    #[test]
    fn full_clockwise_angular_drive_spans_full_rotation() {
        let assembly = slider_crank_fixture();
        let (drive_id, drive) = assembly.drives.iter().next().expect("drive");
        let plan = drive_plan(&assembly, drive_id, drive).expect("drive plan");
        assert!(matches!(plan.parameter, DriveParameter::Angle));
        assert!(matches!(plan.playback, PlaybackTraversal::Forward));
        assert_eq!(plan.start_value.to_bits(), TAU.to_bits());
        assert_eq!(plan.end_value.to_bits(), 0.0f32.to_bits());
        assert_eq!(plan.cycle_seconds.to_bits(), 2.0f32.to_bits());
    }

    #[test]
    fn interval_drive_ping_pong_plays_one_eased_out_and_back_per_cycle() {
        let mut assembly = slider_crank_fixture();
        if let Some(DriveSpec {
            kind: DriveKindSpec::Angular { range, .. },
            ..
        }) = assembly.drives.get_mut("d_crank")
        {
            *range = Some([0.25, 1.25]);
        } else {
            panic!("fixture missing angular drive");
        }

        let (drive_id, drive) = assembly.drives.iter().next().expect("drive");
        let plan = drive_plan(&assembly, drive_id, drive).expect("drive plan");
        assert!(matches!(plan.playback, PlaybackTraversal::PingPong));
        assert_eq!(sample_drive_value(&plan, 0.0).to_bits(), 1.25f32.to_bits());
        assert!((sample_drive_value(&plan, 0.5) - 0.75).abs() < 0.000_1);
        assert!((sample_drive_value(&plan, 1.0) - 0.25).abs() < 0.000_1);

        let artifact = build_sweep_artifact(assembly, 1).expect("artifact");
        let q1 = sampled_frame(&artifact, playback_sample_u(0.25, artifact.playback));
        let q2 = sampled_frame(&artifact, playback_sample_u(0.50, artifact.playback));
        let q3 = sampled_frame(&artifact, playback_sample_u(0.75, artifact.playback));
        let q4 = sampled_frame(&artifact, playback_sample_u(1.00, artifact.playback));
        let drive_value = |frame: &SolvedFrame| *frame.drive_values.values().next().expect("drive");

        assert!((drive_value(q1) - 0.75).abs() < 0.02);
        assert!((drive_value(q2) - 0.25).abs() < 0.02);
        assert!((drive_value(q3) - 0.75).abs() < 0.02);
        assert!((drive_value(q4) - 1.25).abs() < 0.02);

        let early = playback_sample_u(0.05, artifact.playback);
        let middle =
            playback_sample_u(0.30, artifact.playback) - playback_sample_u(0.25, artifact.playback);
        assert!(early < middle);
    }

    #[test]
    fn linear_drive_defaults_to_slider_range_and_ping_pongs() {
        let mut assembly = slider_crank_fixture();
        assembly.drives.insert(
            "d_slide".to_string(),
            DriveSpec {
                kind: DriveKindSpec::Linear {
                    slider: "s_track".to_string(),
                    range: None,
                },
                sweep: SweepSpec {
                    samples: 48,
                    direction: SweepDirectionSpec::PingPong,
                },
            },
        );
        assembly.drives.remove("d_crank");

        let (drive_id, drive) = assembly.drives.iter().next().expect("drive");
        let plan = drive_plan(&assembly, drive_id, drive).expect("drive plan");
        assert!(matches!(plan.parameter, DriveParameter::SliderPosition));
        assert!(matches!(plan.playback, PlaybackTraversal::PingPong));
        assert_eq!(plan.start_value.to_bits(), (-20.0f32).to_bits());
        assert_eq!(plan.end_value.to_bits(), 180.0f32.to_bits());
        assert_eq!(plan.cycle_seconds.to_bits(), 1.5f32.to_bits());
    }

    #[test]
    fn tilted_slider_axis_validates() {
        let mut assembly = slider_crank_fixture();
        let PartSpec::Slider { axis_dir, .. } = assembly
            .parts
            .get_mut("s_track")
            .expect("slider part should exist")
        else {
            panic!("fixture slider should be present");
        };
        *axis_dir = [0.70710677, 0.70710677];
        validate_fixture(&assembly).expect("tilted slider should validate");
    }

    #[test]
    fn no_slider_assembly_builds_and_shapes_for_render() {
        let mut joints = BTreeMap::new();
        joints.insert(
            "j_pivot".to_string(),
            JointSpec::Fixed {
                position: [0.0, 0.0],
            },
        );
        joints.insert("j_tip".to_string(), JointSpec::Free);
        joints.insert("j_loose".to_string(), JointSpec::Free);

        let mut parts = BTreeMap::new();
        parts.insert(
            "l_crank".to_string(),
            PartSpec::Link {
                a: "j_pivot".to_string(),
                b: "j_tip".to_string(),
                length: 60.0,
            },
        );
        parts.insert(
            "l_tail".to_string(),
            PartSpec::Link {
                a: "j_tip".to_string(),
                b: "j_loose".to_string(),
                length: 90.0,
            },
        );

        let mut drives = BTreeMap::new();
        drives.insert(
            "d_crank".to_string(),
            DriveSpec {
                kind: DriveKindSpec::Angular {
                    pivot_joint: "j_pivot".to_string(),
                    tip_joint: "j_tip".to_string(),
                    link: "l_crank".to_string(),
                    range: None,
                },
                sweep: SweepSpec {
                    samples: 24,
                    direction: SweepDirectionSpec::Clockwise,
                },
            },
        );

        let assembly = AssemblySpec {
            joints,
            parts,
            drives,
            points_of_interest: vec![PointOfInterestSpec {
                id: "poi_tail".to_string(),
                host: "l_tail".to_string(),
                t: 0.5,
                perp: 0.0,
            }],
            visualization: None,
            meta: AssemblyMeta {
                name: "no-slider".to_string(),
                iteration: 1,
                notes: Vec::new(),
            },
        };

        validate_fixture(&assembly).expect("no-slider assembly should validate");
        let artifact = build_sweep_artifact(assembly.clone(), 1).expect("artifact");
        let solved = solved_assembly_for_frame(&assembly, &artifact.frames[0]);
        assert!(solved.sliders.is_empty());
        assert_eq!(solved.links.len(), 2);
    }

    #[test]
    fn floating_angular_drive_does_not_require_a_preseeded_pivot() {
        let mut joints = BTreeMap::new();
        joints.insert("j_ground".to_string(), JointSpec::Free);
        joints.insert("j_tip".to_string(), JointSpec::Free);

        let mut parts = BTreeMap::new();
        parts.insert(
            "l_arm".to_string(),
            PartSpec::Link {
                a: "j_ground".to_string(),
                b: "j_tip".to_string(),
                length: 64.0,
            },
        );

        let mut drives = BTreeMap::new();
        drives.insert(
            "d_arm".to_string(),
            DriveSpec {
                kind: DriveKindSpec::Angular {
                    pivot_joint: "j_ground".to_string(),
                    tip_joint: "j_tip".to_string(),
                    link: "l_arm".to_string(),
                    range: None,
                },
                sweep: SweepSpec {
                    samples: 24,
                    direction: SweepDirectionSpec::Clockwise,
                },
            },
        );

        let assembly = AssemblySpec {
            joints,
            parts,
            drives,
            points_of_interest: vec![PointOfInterestSpec {
                id: "poi_arm".to_string(),
                host: "l_arm".to_string(),
                t: 0.5,
                perp: 0.0,
            }],
            visualization: None,
            meta: AssemblyMeta {
                name: "floating-arm".to_string(),
                iteration: 1,
                notes: Vec::new(),
            },
        };

        validate_fixture(&assembly).expect("floating drive assembly should validate");
        build_sweep_artifact(assembly, 1).expect("floating angular drive should build");
    }

    #[test]
    fn branchy_fixture_commits_drawable_sweep() {
        let artifact = build_sweep_artifact(expressive_branchy_fixture(), 1)
            .expect("branchy fixture should keep a drawable artifact");
        assert!(!artifact.frames.is_empty());
        assert!(artifact.telemetry.unsettled_samples > 0);
        assert!(artifact.telemetry.peak_constraint_error > RELAXATION_TOLERANCE);
        assert!(!artifact.telemetry.notes.is_empty());
    }

    #[test]
    fn expressive_three_link_chain_builds_multiple_poi_paths() {
        let artifact =
            build_sweep_artifact(expressive_chain_fixture(), 1).expect("expressive chain fixture");
        assert_eq!(artifact.telemetry.point_paths.len(), 3);
        assert_eq!(artifact.telemetry.unsettled_samples, 0);
    }

    #[test]
    fn bird_flapper_fixture_builds_and_changes_wing_angle() {
        let assembly = bird_flapper_fixture();
        validate_fixture(&assembly).expect("bird fixture should validate");
        let artifact = build_sweep_artifact(assembly, 1).expect("bird fixture should build");
        assert!(!artifact.frames.is_empty());
        assert_eq!(artifact.telemetry.point_paths.len(), 2);

        let mut min_relative = f32::INFINITY;
        let mut max_relative = f32::NEG_INFINITY;
        for frame in &artifact.frames {
            let pivot = frame
                .joint_positions
                .get("j_pivot")
                .copied()
                .expect("pivot");
            let tip = frame.joint_positions.get("j_tip").copied().expect("tip");
            let left = frame
                .joint_positions
                .get("j_left_wing")
                .copied()
                .expect("left wing");
            let body = tip - pivot;
            let wing = left - tip;
            let relative = normalize_angle(wing.angle() - body.angle());
            min_relative = min_relative.min(relative);
            max_relative = max_relative.max(relative);
        }

        assert!(
            max_relative - min_relative > 0.2,
            "expected visible wing-angle change, got range {}",
            max_relative - min_relative
        );
    }

    #[test]
    fn relaxation_warm_start_clears_residual_velocity() {
        let assembly = slider_crank_fixture();
        let (drive_id, drive) = assembly.drives.iter().next().expect("drive");
        let plan = drive_plan(&assembly, drive_id, drive).expect("drive plan");
        let links = collect_link_constraints(&assembly).expect("links");
        let sliders = collect_slider_constraints(&assembly).expect("sliders");
        let drive_constraint =
            build_drive_constraint(&assembly, &plan, sample_drive_value(&plan, 0.33))
                .expect("constraint");
        let seeded = seed_particles(&assembly, &links, &sliders, &drive_constraint).expect("seed");
        let relaxed = relax_particles(&assembly, &seeded, &links, &sliders, &drive_constraint)
            .expect("relaxed");

        for particle in relaxed.particles.values() {
            assert_eq!(particle.pos.x.to_bits(), particle.prev_pos.x.to_bits());
            assert_eq!(particle.pos.y.to_bits(), particle.prev_pos.y.to_bits());
        }
    }

    #[test]
    fn rolling_paper_trace_advances_upward_over_the_cycle() {
        let artifact = build_sweep_artifact(slider_crank_fixture(), 1).expect("artifact");
        let path = artifact
            .telemetry
            .point_paths
            .get("poi_coupler_mid")
            .expect("poi path");
        let (direction, advance_per_cycle) =
            rolling_paper_config(&artifact).expect("rolling paper");
        let start = rolling_paper_position(path[0], direction, advance_per_cycle, 0.0, 0.0);
        let end =
            rolling_paper_position(path[path.len() - 1], direction, advance_per_cycle, 1.0, 0.0);
        assert!(end.y > start.y + 100.0);
    }

    #[test]
    fn rolling_paper_direction_maps_all_variants() {
        assert_eq!(
            rolling_paper_direction(&PaperDirectionSpec::Up),
            vec2(0.0, 1.0)
        );
        assert_eq!(
            rolling_paper_direction(&PaperDirectionSpec::Down),
            vec2(0.0, -1.0)
        );
        assert_eq!(
            rolling_paper_direction(&PaperDirectionSpec::Left),
            vec2(-1.0, 0.0)
        );
        assert_eq!(
            rolling_paper_direction(&PaperDirectionSpec::Right),
            vec2(1.0, 0.0)
        );
    }

    #[test]
    fn longer_links_make_their_joints_heavier() {
        let assembly = slider_crank_fixture();
        let links = collect_link_constraints(&assembly).expect("links");
        let masses = joint_mass_map(&assembly, &links);
        assert!(masses["j_tip"] > masses["j_slide"]);
        assert!(masses["j_slide"] > masses["j_pivot"]);
    }

    #[test]
    fn verlet_step_applies_gravity_to_free_particles() {
        let mut particles = BTreeMap::from([(
            "j".to_string(),
            ParticleState {
                pos: pt2(0.0, 0.0),
                prev_pos: pt2(0.0, 0.0),
                fixed: false,
                mass: 1.0,
            },
        )]);
        verlet_step(&mut particles);
        let particle = particles.get("j").expect("particle");
        assert_eq!(particle.pos.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(particle.pos.y.to_bits(), (-GRAVITY_ACCEL_Y).to_bits());
        assert_eq!(particle.prev_pos.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(particle.prev_pos.y.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn rolling_paper_ping_pong_trace_head_does_not_shrink_within_a_cycle() {
        let (cycle, head_u) = update_live_trace_state(0.25, 0, 0.0, PlaybackTraversal::PingPong);
        assert_eq!(cycle, 0);
        assert!(head_u > 0.49 && head_u < 0.51);

        let (_, peak_u) = update_live_trace_state(0.50, cycle, head_u, PlaybackTraversal::PingPong);
        assert!(peak_u > 0.99);

        let (_, return_u) =
            update_live_trace_state(0.75, cycle, peak_u, PlaybackTraversal::PingPong);
        assert_eq!(return_u.to_bits(), peak_u.to_bits());

        let (next_cycle, reset_u) =
            update_live_trace_state(1.10, cycle, return_u, PlaybackTraversal::PingPong);
        assert_eq!(next_cycle, 1);
        assert!(reset_u < 0.10);
    }

    #[test]
    fn trace_cycle_entities_are_capped() {
        let artifact = build_sweep_artifact(slider_crank_fixture(), 1).expect("artifact");
        let mut entities = Vec::new();
        for cycle_index in 0..(MAX_TRACE_CYCLE_ENTITIES as u32 + 32) {
            spawn_trace_cycle_entities(&mut entities, &artifact, cycle_index);
        }
        assert_eq!(entities.len(), MAX_TRACE_CYCLE_ENTITIES);
        assert_eq!(entities[0].completion_progress, 33.0);
        assert_eq!(
            entities[entities.len() - 1].completion_progress,
            (MAX_TRACE_CYCLE_ENTITIES as u32 + 32) as f32
        );
    }

    #[test]
    fn trim_llm_history_keeps_recent_turns_and_prefers_user_lead() {
        let mut history = Vec::new();
        for idx in 0..40 {
            history.push(LlmHistoryMessage {
                role: if idx % 2 == 0 {
                    LlmHistoryRole::User
                } else {
                    LlmHistoryRole::Assistant
                },
                text: format!("m{idx}"),
            });
        }

        trim_llm_history(&mut history);

        assert!(history.len() <= MAX_LLM_HISTORY_MESSAGES);
        assert!(matches!(
            history.first().map(|message| message.role),
            Some(LlmHistoryRole::User)
        ));
        assert_eq!(
            history.last().map(|message| message.text.as_str()),
            Some("m39")
        );
    }

    #[test]
    fn trim_llm_history_drops_non_user_prefix_after_trimming() {
        let mut history = Vec::new();
        history.push(LlmHistoryMessage {
            role: LlmHistoryRole::Tool,
            text: "tool".to_string(),
        });
        for idx in 0..170 {
            history.push(LlmHistoryMessage {
                role: if idx % 3 == 0 {
                    LlmHistoryRole::User
                } else if idx % 3 == 1 {
                    LlmHistoryRole::Tool
                } else {
                    LlmHistoryRole::Assistant
                },
                text: format!("m{idx}"),
            });
        }

        trim_llm_history(&mut history);

        assert!(history.len() <= MAX_LLM_HISTORY_MESSAGES);
        assert!(matches!(
            history.first().map(|message| message.role),
            Some(LlmHistoryRole::User)
        ));
    }

    #[test]
    fn startup_prompt_fixture_is_a_short_driven_arm() {
        let assembly = startup_prompt_sample_fixture();
        assert_eq!(assembly.joints.len(), 2);
        assert_eq!(assembly.parts.len(), 1);
        assert_eq!(assembly.drives.len(), 1);
        assert!(assembly.points_of_interest.is_empty());
        match assembly.parts.get("l_arm") {
            Some(PartSpec::Link { a, b, length }) => {
                assert_eq!(a, "j_pivot");
                assert_eq!(b, "j_tip");
                assert_eq!(length.to_bits(), 36.0f32.to_bits());
            }
            _ => panic!("startup fixture should contain l_arm"),
        }
        match assembly.drives.get("d_arm") {
            Some(DriveSpec {
                kind:
                    DriveKindSpec::Angular {
                        pivot_joint,
                        tip_joint,
                        link,
                        range,
                    },
                ..
            }) => {
                assert_eq!(pivot_joint, "j_pivot");
                assert_eq!(tip_joint, "j_tip");
                assert_eq!(link, "l_arm");
                assert!(range.is_none());
            }
            _ => panic!("startup fixture should contain d_arm"),
        }
    }

    #[test]
    #[ignore = "live OpenAI call; run with --ignored"]
    fn set_assembly_schema_is_accepted_by_openai() {
        dotenvy::dotenv().ok();
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for schema test");
        let assembly_json = serde_json::to_string_pretty(&startup_prompt_sample_fixture())
            .expect("sample fixture should serialize");
        let input = vec![
            serde_json::json!({
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are designing and editing a mechanism assembly. Respond only by calling set_assembly."
                    }
                ]
            }),
            serde_json::json!({
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": format!(
                            "Current assembly:\n{}\n\nUser request:\nReturn the assembly unchanged and append \"schema acceptance probe\" to meta.notes.\n\nCall set_assembly now.",
                            assembly_json
                        )
                    }
                ]
            }),
        ];
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(45))
            .build()
            .expect("OpenAI client");
        let body = send_llm_request(&client, &api_key, &input, None)
            .expect("OpenAI should accept set_assembly schema");
        let tool_call = extract_function_call(&body, "set_assembly").unwrap_or_else(|| {
            panic!("expected set_assembly tool call in OpenAI response: {body}")
        });
        let args: SetAssemblyArgs =
            serde_json::from_str(&tool_call.arguments).unwrap_or_else(|err| {
                panic!(
                    "set_assembly arguments should deserialize as SetAssemblyArgs: {err}\n{}",
                    tool_call.arguments
                )
            });
        assert!(
            !args.assembly.joints.is_empty(),
            "returned assembly should carry at least one joint, got 0"
        );
        assert_eq!(
            args.assembly.drives.len(),
            1,
            "returned assembly must contain exactly one drive, got {}",
            args.assembly.drives.len()
        );
    }

    #[test]
    #[ignore = "live OpenAI call; run with --ignored"]
    fn set_assembly_tool_history_is_accepted_by_openai() {
        dotenvy::dotenv().ok();
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for schema test");
        let history_args = SetAssemblyArgs {
            reasoning: "Describe only — return the current linkage unchanged.".to_string(),
            assembly: startup_prompt_sample_fixture(),
        };
        let history_args_json =
            serde_json::to_string_pretty(&history_args).expect("history args should serialize");
        let history = vec![
            LlmHistoryMessage {
                role: LlmHistoryRole::User,
                text: "Describe the current linkage.".to_string(),
            },
            LlmHistoryMessage {
                role: LlmHistoryRole::Tool,
                text: format!(
                    "set_assembly arguments\n{}\n\nset_assembly response\n{{\n  \"ok\": true,\n  \"message\": \"assembly applied\"\n}}",
                    history_args_json
                ),
            },
            LlmHistoryMessage {
                role: LlmHistoryRole::Assistant,
                text: "Current linkage is a slider-crank.".to_string(),
            },
            LlmHistoryMessage {
                role: LlmHistoryRole::User,
                text: "Shorten the crank by 10%.".to_string(),
            },
        ];

        let (assembly, _args, _exchanges) =
            match generate_llm_turn(&api_key, &startup_prompt_sample_fixture(), &history, None) {
                Ok(result) => result,
                Err(err) => panic!(
                    "OpenAI should accept tool history in prompt context: {}",
                    err.message
                ),
            };
        assert_eq!(assembly.drives.len(), 1);
    }

    #[test]
    fn fixture_validation_rejects_global_id_collisions() {
        let mut assembly = slider_crank_fixture();
        let crank = assembly.parts.remove("l_crank").expect("crank link");
        assembly.parts.insert("j_pivot".to_string(), crank);

        let err = validate_fixture(&assembly).expect_err("id collision should fail");
        assert!(
            err.contains("id collision: j_pivot is already used by joint"),
            "{err}"
        );
    }

    #[test]
    fn llm_prompt_explains_poi_limits_and_global_ids() {
        let prompt = llm_system_prompt().expect("prompt");
        assert!(prompt.contains("set_assembly"));
        assert!(prompt.contains("Return the next complete assembly"));
        assert!(prompt.contains("copy it, modify what you need, return the full result"));
        assert!(prompt.contains("meta.notes"));
        assert!(prompt.contains("Ids are globally unique across joints, parts, drives, and POIs"));
        assert!(prompt.contains("POIs are diagnostics only"));
        assert!(prompt.contains("The current assembly size is not a cap"));
        assert!(prompt.contains("explicit `null` for optional fields"));
        assert!(
            !prompt.contains("Available mutation ops"),
            "prompt should no longer advertise atomic mutation ops"
        );
        assert!(
            !prompt.contains("New joints are created ONLY by add_joint"),
            "prompt should no longer reference add_joint"
        );
    }

    #[test]
    fn tool_error_guidance_adds_missing_joint_hint() {
        let guidance = tool_error_guidance("validation failed: part l_leg: missing joint j_knee");
        assert!(
            guidance.contains("Add the joint to assembly.joints"),
            "{guidance}"
        );
    }

    #[test]
    fn tool_error_guidance_adds_id_collision_hint() {
        let guidance = tool_error_guidance("id collision: j_end is already used by joint");
        assert!(guidance.contains("globally unique"), "{guidance}");
    }

    #[test]
    fn tool_error_guidance_adds_relaxation_hint() {
        let guidance = tool_error_guidance(
            "relaxation failed: relaxation failed to settle: max constraint error 32.570",
        );
        assert!(
            guidance.contains("do not replace visible structure with POIs"),
            "{guidance}"
        );
    }

    #[test]
    fn tool_error_details_reports_missing_reference() {
        let details = tool_error_details(
            "validation failed: part l_leg: missing joint j_knee",
            &slider_crank_fixture(),
        )
        .expect("details");
        assert_eq!(details["category"], "missing_reference");
        assert_eq!(details["owner_kind"], "part");
        assert_eq!(details["missing_kind"], "joint");
        let suggested_fix = details["suggested_fix"].as_str().expect("suggested_fix");
        assert!(suggested_fix.contains("assembly.joints"), "{suggested_fix}");
        assert!(
            !suggested_fix.contains("add_joint"),
            "whole-assembly guidance should not reference the old add_joint op: {suggested_fix}"
        );
    }

    #[test]
    fn set_assembly_roundtrips_through_validator() {
        let args = SetAssemblyArgs {
            reasoning: "round-trip fixture through SetAssemblyArgs serde path".to_string(),
            assembly: slider_crank_fixture(),
        };
        let serialized =
            serde_json::to_string_pretty(&args).expect("SetAssemblyArgs should serialize");
        let decoded: SetAssemblyArgs =
            serde_json::from_str(&serialized).expect("SetAssemblyArgs should deserialize");
        assert_eq!(decoded.reasoning, args.reasoning);
        validate_fixture(&decoded.assembly).expect("round-tripped assembly should still validate");
        assert_eq!(decoded.assembly.drives.len(), 1);
        assert_eq!(decoded.assembly.joints.len(), 3);
        assert_eq!(decoded.assembly.parts.len(), 3);
    }

    #[test]
    fn set_assembly_schema_matches_current_assembly_serialization() {
        // Guards Step 1: nullable fields on AssemblySpec / VisualizationSpec /
        // DriveKindSpec::{Angular,Linear}.range must serialize as explicit null
        // rather than being omitted, so the model can copy the current-assembly
        // JSON and return it through the strict set_assembly schema without
        // having to synthesize fields that were absent in its context.
        let mut assembly = slider_crank_fixture();
        assembly.visualization = None;
        let value = serde_json::to_value(&assembly).expect("assembly should serialize");
        let obj = value.as_object().expect("assembly json is an object");
        assert!(
            obj.contains_key("visualization"),
            "visualization key must be present (as null) when None"
        );
        assert_eq!(
            obj.get("visualization"),
            Some(&serde_json::Value::Null),
            "visualization should serialize as explicit null"
        );
        let drives = obj
            .get("drives")
            .and_then(|v| v.as_object())
            .expect("drives is an object");
        let crank = drives
            .get("d_crank")
            .and_then(|v| v.as_object())
            .expect("d_crank drive present");
        let kind = crank
            .get("kind")
            .and_then(|v| v.as_object())
            .expect("drive kind is an object");
        assert!(
            kind.contains_key("range"),
            "angular drive range must be present (as null) when None"
        );
        assert_eq!(
            kind.get("range"),
            Some(&serde_json::Value::Null),
            "angular drive range should serialize as explicit null"
        );
    }

    fn frame_signature(frames: &[SolvedFrame]) -> Vec<u32> {
        let mut signature = Vec::new();
        for frame in frames {
            signature.push(frame.u.to_bits());
            for position in frame.joint_positions.values() {
                signature.push(position.x.to_bits());
                signature.push(position.y.to_bits());
            }
            for value in frame.drive_values.values() {
                signature.push(value.to_bits());
            }
            for position in frame.poi_positions.values() {
                signature.push(position.x.to_bits());
                signature.push(position.y.to_bits());
            }
        }
        signature
    }

    fn normalize_angle(angle: f32) -> f32 {
        let mut wrapped = angle;
        while wrapped <= -PI {
            wrapped += TAU;
        }
        while wrapped > PI {
            wrapped -= TAU;
        }
        wrapped
    }
}
