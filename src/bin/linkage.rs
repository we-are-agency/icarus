use nannou::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, OnceLock};

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
const STARTUP_MODEL: &str = "gpt-5.3-codex";
const MAX_TRACE_CYCLE_ENTITIES: usize = 256;
const RELAXATION_ITERATIONS: usize = 96;
const RELAXATION_TOLERANCE: f32 = 0.001;
const RELAXATION_DAMPING: f32 = 0.98;
const RELAXATION_DIVERGENCE_LIMIT: f32 = 10_000.0;

static CONFIG: OnceLock<Config> = OnceLock::new();
static STARTUP_VARIATION_COUNTER: AtomicU64 = AtomicU64::new(1);

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
    headless_capture_state: Cell<HeadlessCaptureState>,
    headless_capture_result: Arc<AtomicU8>,
    headless_proxy: Option<nannou::app::Proxy>,
    spec_scroll_px: f32,
    startup_generation_rx: Option<Receiver<StartupGenerationResult>>,
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
    json_lines: Vec<String>,
}

enum FixtureStatus {
    Loading(String),
    Solved(Arc<SweepArtifact>),
    ValidationError(String),
    RelaxationError(String),
    GenerationError(String),
}

struct StartupGenerationResult {
    fixture: FixturePresentation,
}

struct TraceCycleEntity {
    completion_progress: f32,
    color_index: usize,
    points: Vec<Point2>,
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
    point_paths: BTreeMap<String, Vec<Point2>>,
    unsettled_samples: u32,
    peak_constraint_error: f32,
    notes: Vec<String>,
}

struct SolvedFrame {
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
    slider: SolvedSlider,
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
        #[serde(skip_serializing_if = "Option::is_none")]
        range: Option<[f32; 2]>,
    },
    #[allow(dead_code)]
    Linear {
        slider: String,
        #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(
        default = "default_simulation_mode",
        skip_serializing_if = "simulation_mode_is_default"
    )]
    simulation_mode: SimulationModeSpec,
}

#[derive(Clone, Copy, Deserialize, Serialize, Eq, PartialEq)]
enum SimulationModeSpec {
    Strict,
    Expressive,
}

fn default_simulation_mode() -> SimulationModeSpec {
    SimulationModeSpec::Expressive
}

fn simulation_mode_is_default(mode: &SimulationModeSpec) -> bool {
    *mode == default_simulation_mode()
}

#[derive(Deserialize)]
struct ProposeMutationsArgs {
    reasoning: String,
    mutations: Vec<StartupMutation>,
}

#[derive(Deserialize)]
#[serde(tag = "op")]
enum StartupMutation {
    #[serde(rename = "add_joint")]
    AddJoint { id: String, joint: JointSpec },
    #[serde(rename = "add_part")]
    AddPart { id: String, part: PartSpec },
    #[serde(rename = "add_drive")]
    AddDrive { id: String, drive: DriveSpec },
    #[serde(rename = "add_poi")]
    AddPoi { poi: PointOfInterestSpec },
    #[serde(rename = "note")]
    Note { text: String },
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
    let (fixtures, startup_generation_rx) =
        build_fixture_presentations().expect("failed to build fixture bank");

    let mut window = app
        .new_window()
        .title("Linkage")
        .size(WINDOW_W, WINDOW_H)
        .view(view)
        .key_pressed(key_pressed)
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
        headless_capture_state: Cell::new(HeadlessCaptureState::Pending),
        headless_capture_result: Arc::new(AtomicU8::new(0)),
        headless_proxy: if config.headless {
            Some(app.create_proxy())
        } else {
            None
        },
        spec_scroll_px: 0.0,
        startup_generation_rx,
        trace_cycle_entities: Vec::new(),
        emitted_trace_cycles: 0,
    }
}

fn update(app: &App, model: &mut Model, update: Update) {
    if let Some(rx) = &model.startup_generation_rx {
        if let Ok(result) = rx.try_recv() {
            if !model.fixtures.is_empty() {
                model.fixtures[0] = result.fixture;
            }
            model.startup_generation_rx = None;
        }
    }

    if let Some((turn, reset_phase, cycle_seconds, playback)) =
        current_artifact(model).map(|artifact| {
            (
                artifact.turn,
                artifact.reset_phase,
                artifact.cycle_seconds,
                artifact.playback,
            )
        })
    {
        if model.active_artifact_turn != Some(turn) {
            if reset_phase {
                model.playback_progress = 0.0;
            }
            let completed_cycles = model.playback_progress.floor().max(0.0) as u32;
            model.live_trace_cycle = completed_cycles;
            model.live_trace_u = playback_sample_u(model.playback_progress, playback);
            model.active_artifact_turn = Some(turn);
            model.trace_cycle_entities.clear();
            model.emitted_trace_cycles = completed_cycles;
        }

        if !model.playback_paused {
            let delta = update.since_last.as_secs_f32() / cycle_seconds.max(0.001);
            model.playback_progress += delta;
        }

        (model.live_trace_cycle, model.live_trace_u) = update_live_trace_state(
            model.playback_progress,
            model.live_trace_cycle,
            model.live_trace_u,
            playback,
        );

        if let Some(artifact) = current_artifact(model).cloned() {
            if rolling_paper_config(&artifact).is_some() {
                let completed_cycles = model.playback_progress.floor().max(0.0) as u32;
                while model.emitted_trace_cycles < completed_cycles {
                    spawn_trace_cycle_entities(
                        &mut model.trace_cycle_entities,
                        &artifact,
                        model.emitted_trace_cycles,
                    );
                    model.emitted_trace_cycles += 1;
                }

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
        Key::Tab => select_fixture(model, (model.selected_fixture + 1) % model.fixtures.len()),
        Key::Up => scroll_spec(model, app.window_rect(), -PANE_LINE_H * 3.0),
        Key::Down => scroll_spec(model, app.window_rect(), PANE_LINE_H * 3.0),
        Key::PageUp => scroll_spec(
            model,
            app.window_rect(),
            -visible_spec_body_height(app.window_rect()) * 0.9,
        ),
        Key::PageDown => scroll_spec(
            model,
            app.window_rect(),
            visible_spec_body_height(app.window_rect()) * 0.9,
        ),
        Key::Home => model.spec_scroll_px = 0.0,
        Key::End => {
            model.spec_scroll_px = max_spec_scroll_px(current_fixture(model), app.window_rect())
        }
        _ => {}
    }
}

fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    if button != MouseButton::Left {
        return;
    }
    if startup_generation_in_progress(model) {
        return;
    }
    if startup_regenerate_rect(app.window_rect()).contains(app.mouse.position()) {
        trigger_startup_generation(model);
    }
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

fn startup_generation_in_progress(model: &Model) -> bool {
    model.startup_generation_rx.is_some()
        || matches!(
            model.fixtures.first().map(|fixture| &fixture.status),
            Some(FixtureStatus::Loading(_))
        )
}

fn trigger_startup_generation(model: &mut Model) {
    if startup_generation_in_progress(model) {
        return;
    }
    let previous_fixture_json = model
        .fixtures
        .first()
        .map(|fixture| fixture.json_lines.join("\n"));
    let (fixture, rx) = startup_fixture_slot(previous_fixture_json);
    if !model.fixtures.is_empty() {
        model.fixtures[0] = fixture;
        model.startup_generation_rx = rx;
        select_fixture(model, 0);
    }
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

    if model.headless
        && model.headless_capture_state.get() == HeadlessCaptureState::Pending
        && !matches!(current_fixture(model).status, FixtureStatus::Loading(_))
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
    if index < model.fixtures.len() {
        model.selected_fixture = index;
        model.spec_scroll_px = 0.0;
    }
}

fn current_fixture(model: &Model) -> &FixturePresentation {
    &model.fixtures[model.selected_fixture]
}

fn current_artifact(model: &Model) -> Option<&Arc<SweepArtifact>> {
    match &current_fixture(model).status {
        FixtureStatus::Loading(_)
        | FixtureStatus::GenerationError(_)
        | FixtureStatus::ValidationError(_)
        | FixtureStatus::RelaxationError(_) => None,
        FixtureStatus::Solved(artifact) => Some(artifact),
    }
}

fn build_fixture_presentations() -> Result<
    (
        Vec<FixturePresentation>,
        Option<Receiver<StartupGenerationResult>>,
    ),
    String,
> {
    let (startup_fixture, startup_generation_rx) = startup_fixture_slot(None);
    let fixtures = vec![
        startup_fixture,
        build_fixture_presentation("2 STRICT", slider_crank_fixture())?,
        build_fixture_presentation("3 BAD REF", invalid_reference_fixture())?,
        build_fixture_presentation("4 UNSOLVED", unsolved_slider_crank_fixture())?,
        build_fixture_presentation("5 CHAINY", expressive_chain_fixture())?,
        build_fixture_presentation("6 BRANCHY", expressive_branchy_fixture())?,
    ];

    Ok((fixtures, startup_generation_rx))
}

fn build_fixture_presentation(
    label: &str,
    assembly: AssemblySpec,
) -> Result<FixturePresentation, String> {
    let json = serde_json::to_string_pretty(&assembly)
        .map_err(|err| format!("failed to serialize fixture JSON: {err}"))?;
    let status = match validate_fixture(&assembly) {
        Ok(()) => match build_sweep_artifact(assembly.clone(), 1) {
            Ok(artifact) => FixtureStatus::Solved(Arc::new(artifact)),
            Err(err) => FixtureStatus::RelaxationError(err),
        },
        Err(err) => FixtureStatus::ValidationError(err),
    };
    let json_lines = json.lines().map(|line| line.to_string()).collect();
    Ok(FixturePresentation {
        label: label.to_string(),
        status,
        json_lines,
    })
}

fn startup_fixture_slot(
    previous_fixture_json: Option<String>,
) -> (
    FixturePresentation,
    Option<Receiver<StartupGenerationResult>>,
) {
    let model_name = startup_model_name();
    let request_id = STARTUP_VARIATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let (variant_label, variant_brief) = startup_prompt_variant(request_id);
    let placeholder_json = vec![
        "{".to_string(),
        format!("  \"startup_generation\": \"pending\","),
        format!("  \"model\": \"{}\",", model_name),
        format!("  \"variant\": \"{}\",", variant_label),
        format!("  \"request_id\": {},", request_id),
        "  \"source\": \"startup prompt with embedded sample fixture and novelty steering\""
            .to_string(),
        "}".to_string(),
    ];

    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        return (
            FixturePresentation {
                label: "1 STARTUP".to_string(),
                status: FixtureStatus::GenerationError(
                    "OPENAI_API_KEY is missing; using local fixtures only".to_string(),
                ),
                json_lines: vec![
                    "{".to_string(),
                    "  \"startup_generation\": \"disabled\",".to_string(),
                    format!("  \"model\": \"{}\",", model_name),
                    "  \"reason\": \"OPENAI_API_KEY missing\"".to_string(),
                    "}".to_string(),
                ],
            },
            None,
        );
    };

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let fixture = match generate_startup_fixture(
            &api_key,
            request_id,
            variant_label,
            variant_brief,
            previous_fixture_json.as_deref(),
        ) {
            Ok(assembly) => match build_fixture_presentation("1 STARTUP", assembly) {
                Ok(presentation) => presentation,
                Err(err) => generation_error_fixture("1 STARTUP", err),
            },
            Err(err) => generation_error_fixture("1 STARTUP", err),
        };
        let _ = tx.send(StartupGenerationResult { fixture });
    });

    (
        FixturePresentation {
            label: "1 STARTUP".to_string(),
            status: FixtureStatus::Loading(format!(
                "waiting for {} to generate a startup fixture",
                model_name
            )),
            json_lines: placeholder_json,
        },
        Some(rx),
    )
}

fn generation_error_fixture(label: &str, error: String) -> FixturePresentation {
    let json_lines = vec![
        "{".to_string(),
        "  \"startup_generation\": \"failed\",".to_string(),
        format!("  \"model\": \"{}\",", startup_model_name()),
        format!("  \"error\": \"{}\"", error.replace('"', "\\\"")),
        "}".to_string(),
    ];
    FixturePresentation {
        label: label.to_string(),
        status: FixtureStatus::GenerationError(error),
        json_lines,
    }
}

fn startup_prompt_variant(request_id: u64) -> (&'static str, &'static str) {
    match request_id % 4 {
        0 => (
            "branchy",
            "Favor a branchy or ambiguous assembly with a surprising fold, brace, or side path.",
        ),
        1 => (
            "chainy",
            "Favor a longer internal chain with extra articulated joints or an indirect transmission path.",
        ),
        2 => (
            "rolling-paper",
            "Favor multiple POIs and a composition that leaves especially rich rolling-paper traces.",
        ),
        _ => (
            "near-lock",
            "Favor a pose family that comes close to locking, stalling, or snapping while staying drawable.",
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
            simulation_mode: SimulationModeSpec::Strict,
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

fn unsolved_slider_crank_fixture() -> AssemblySpec {
    let mut assembly = slider_crank_fixture();
    if let Some(PartSpec::Slider { range, .. }) = assembly.parts.get_mut("s_track") {
        *range = [500.0, 600.0];
    }
    assembly.meta.name = "p1-unsolved-slider-crank".to_string();
    assembly.meta.notes = vec!["Intentional relaxation range failure".to_string()];
    assembly.meta.simulation_mode = SimulationModeSpec::Strict;
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
    assembly.meta.simulation_mode = SimulationModeSpec::Expressive;
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
        "Expressive mode keeps the drawable artifact instead of failing hard".to_string(),
    ];
    assembly.meta.simulation_mode = SimulationModeSpec::Expressive;
    assembly
}

fn validate_fixture(assembly: &AssemblySpec) -> Result<(), String> {
    if assembly.drives.len() != 1 {
        return Err("fixture must contain exactly one drive for the current renderer".to_string());
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
                if *length <= 0.0 {
                    return Err(format!("part {part_id}: non-positive link length"));
                }
            }
            PartSpec::Slider {
                joint,
                axis_dir,
                range,
                ..
            } => {
                if !assembly.joints.contains_key(joint) {
                    return Err(format!("part {part_id}: missing slider joint {joint}"));
                }
                if (axis_dir[0] * axis_dir[0] + axis_dir[1] * axis_dir[1]) < 0.99 {
                    return Err(format!("part {part_id}: slider axis is not normalized"));
                }
                if axis_dir[0].abs() < 0.999 || axis_dir[1].abs() > 0.001 {
                    return Err(format!(
                        "part {part_id}: current relaxation engine requires a horizontal slider axis"
                    ));
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

fn generate_startup_fixture(
    api_key: &str,
    request_id: u64,
    variant_label: &str,
    variant_brief: &str,
    previous_fixture_json: Option<&str>,
) -> Result<AssemblySpec, String> {
    let sample_fixture = serde_json::to_string_pretty(&startup_prompt_sample_fixture())
        .map_err(|err| format!("failed to serialize sample fixture: {err}"))?;
    let novelty_clause = previous_fixture_json
        .map(|previous| {
            format!(
                "Previous startup fixture to differ from materially:\n{}\nProduce a startup fixture that is clearly different in topology, proportions, POI layout, or overall motion character.\nDo not make a near-copy with only tiny dimension edits.\n",
                previous
            )
        })
        .unwrap_or_else(|| {
            "There is no previous startup fixture yet, so establish a strong visual identity.\n"
                .to_string()
        });
    let system_prompt = format!(
        concat!(
            "You generate one visually interesting linkage fixture for a startup visualization.\n",
            "Current assembly is empty.\n",
            "Call the propose_mutations tool exactly once. Do not emit JSON in prose.\n",
            "Use add_* mutations to construct the full startup fixture in one batch.\n",
            "For this cold start, 6-12 ops is acceptable.\n",
            "Variation token: request-{} / mode-{}.\n",
            "Creative bias for this request: {}\n",
            "Guidance:\n",
            "- Include one slider track; the current renderer expects a slider-based assembly.\n",
            "- Keep the slider axis horizontal and its range increasing.\n",
            "- Use exactly one drive. Angular or Linear are both acceptable; angular usually reads more clearly.\n",
            "- If an angular drive omits range, prefer assemblies that stay drawable through a full rotation.\n",
            "- Include 1 to 4 points_of_interest on links so traces are interesting.\n",
            "- Keep points_of_interest on their host links with perp = 0.0 by default.\n",
            "- Use perp != 0.0 only when the offset is materially important to the intended visual effect.\n",
            "- Branchy, ambiguous, or slightly unstable motion is acceptable if the result stays drawable and visually interesting.\n",
            "- Keep values finite and visually legible for a small mechanism.\n",
            "- Prefer an off-axis pivot so the crank is visibly above or below the slider track.\n",
            "- Prefer small, coherent dimensions over extreme ranges.\n",
            "{}",
            "Here is a valid sample fixture for reference. Do not copy its exact dimensions:\n",
            "{}"
        ),
        request_id, variant_label, variant_brief, novelty_clause, sample_fixture
    );

    let request_body = serde_json::json!({
        "model": startup_model_name(),
        "store": false,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Generate one startup linkage fixture now by calling propose_mutations."
                    }
                ]
            }
        ],
        "tools": [
            propose_mutations_tool_schema()
        ],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {
                    "type": "function",
                    "name": "propose_mutations"
                }
            ]
        }
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(45))
        .build()
        .map_err(|err| format!("failed to build OpenAI client: {err}"))?;
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

    let arguments =
        extract_function_call_arguments(&body, "propose_mutations").ok_or_else(|| {
            let fallback = extract_response_text(&body).unwrap_or_else(|| body.to_string());
            format!(
                "OpenAI did not return a propose_mutations tool call.\n{}",
                trim_for_error(&fallback, 480)
            )
        })?;
    let tool_args: ProposeMutationsArgs = serde_json::from_str(&arguments).map_err(|err| {
        format!(
            "failed to parse propose_mutations arguments: {err}\n{}",
            trim_for_error(&arguments, 480)
        )
    })?;
    build_startup_fixture_from_mutations(&tool_args)
}

fn startup_model_name() -> String {
    std::env::var("OPENAI_MODEL").unwrap_or_else(|_| STARTUP_MODEL.to_string())
}

fn startup_prompt_sample_fixture() -> AssemblySpec {
    let mut sample = slider_crank_fixture();
    sample.visualization = None;
    sample
}

fn propose_mutations_tool_schema() -> serde_json::Value {
    serde_json::from_str(
        r#"
        {
          "type": "function",
          "name": "propose_mutations",
          "description": "Propose one complete startup linkage as atomic mutations from an empty assembly. Use add_* operations to create the initial mechanism.",
          "strict": true,
          "parameters": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "reasoning": { "type": "string" },
              "mutations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                    "op": {
                      "type": "string",
                      "enum": ["add_joint", "add_part", "add_drive", "add_poi", "note"]
                    },
                    "id": {
                      "type": ["string", "null"]
                    },
                    "joint": {
                      "type": ["object", "null"],
                      "additionalProperties": false,
                      "properties": {
                        "type": {
                          "type": "string",
                          "enum": ["Fixed", "Free"]
                        },
                        "position": {
                          "type": ["array", "null"],
                          "items": { "type": "number" },
                          "minItems": 2,
                          "maxItems": 2
                        }
                      },
                      "required": ["type", "position"]
                    },
                    "part": {
                      "type": ["object", "null"],
                      "additionalProperties": false,
                      "properties": {
                        "type": {
                          "type": "string",
                          "enum": ["Link", "Slider"]
                        },
                        "a": { "type": ["string", "null"] },
                        "b": { "type": ["string", "null"] },
                        "length": { "type": ["number", "null"] },
                        "joint": { "type": ["string", "null"] },
                        "axis_origin": {
                          "type": ["array", "null"],
                          "items": { "type": "number" },
                          "minItems": 2,
                          "maxItems": 2
                        },
                        "axis_dir": {
                          "type": ["array", "null"],
                          "items": { "type": "number" },
                          "minItems": 2,
                          "maxItems": 2
                        },
                        "range": {
                          "type": ["array", "null"],
                          "items": { "type": "number" },
                          "minItems": 2,
                          "maxItems": 2
                        }
                      },
                      "required": ["type", "a", "b", "length", "joint", "axis_origin", "axis_dir", "range"]
                    },
                    "drive": {
                      "type": ["object", "null"],
                      "additionalProperties": false,
                      "properties": {
                        "kind": {
                          "type": "object",
                          "additionalProperties": false,
                          "properties": {
                            "type": {
                              "type": "string",
                              "enum": ["Angular"]
                            },
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
                        "sweep": {
                          "type": "object",
                          "additionalProperties": false,
                          "properties": {
                            "samples": { "type": "integer", "minimum": 2 },
                            "direction": {
                              "type": "string",
                              "enum": ["Forward", "Reverse", "PingPong", "Clockwise", "CounterClockwise", "CW", "CCW"]
                            }
                          },
                          "required": ["samples", "direction"]
                        }
                      },
                      "required": ["kind", "sweep"]
                    },
                    "poi": {
                      "type": ["object", "null"],
                      "additionalProperties": false,
                      "properties": {
                        "id": { "type": "string" },
                        "host": { "type": "string" },
                        "t": { "type": "number" },
                        "perp": { "type": "number" }
                      },
                      "required": ["id", "host", "t", "perp"]
                    },
                    "text": {
                      "type": ["string", "null"]
                    }
                  },
                  "required": ["op", "id", "joint", "part", "drive", "poi", "text"]
                }
              }
            },
            "required": ["reasoning", "mutations"]
          }
        }
        "#,
    )
    .expect("propose_mutations tool schema should be valid JSON")
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

fn extract_function_call_arguments(body: &serde_json::Value, tool_name: &str) -> Option<String> {
    body.get("output")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                let item_type = item.get("type").and_then(serde_json::Value::as_str);
                let item_name = item.get("name").and_then(serde_json::Value::as_str);
                if item_type == Some("function_call") && item_name == Some(tool_name) {
                    item.get("arguments")
                        .and_then(serde_json::Value::as_str)
                        .map(ToString::to_string)
                } else {
                    None
                }
            })
        })
}

fn build_startup_fixture_from_mutations(
    args: &ProposeMutationsArgs,
) -> Result<AssemblySpec, String> {
    let assembly = apply_startup_mutation_set(&args.reasoning, &args.mutations)
        .map_err(|err| format!("mutation application failed: {err}"))?;
    if let Err(err) = validate_fixture(&assembly) {
        return Err(format!("validation failed: {err}"));
    }
    if let Err(err) = build_sweep_artifact(assembly.clone(), 1) {
        return Err(format!("relaxation failed: {err}"));
    }
    Ok(assembly)
}

fn apply_startup_mutation_set(
    reasoning: &str,
    mutations: &[StartupMutation],
) -> Result<AssemblySpec, String> {
    let mut assembly = empty_startup_assembly();
    if !reasoning.trim().is_empty() {
        assembly
            .meta
            .notes
            .push(format!("startup reasoning: {}", reasoning.trim()));
    }

    for mutation in mutations {
        match mutation {
            StartupMutation::AddJoint { id, joint } => {
                if assembly.joints.contains_key(id) {
                    return Err(format!("duplicate joint id {id}"));
                }
                assembly.joints.insert(id.clone(), joint.clone());
            }
            StartupMutation::AddPart { id, part } => {
                if assembly.parts.contains_key(id) {
                    return Err(format!("duplicate part id {id}"));
                }
                assembly.parts.insert(id.clone(), part.clone());
            }
            StartupMutation::AddDrive { id, drive } => {
                if assembly.drives.contains_key(id) {
                    return Err(format!("duplicate drive id {id}"));
                }
                assembly.drives.insert(id.clone(), drive.clone());
            }
            StartupMutation::AddPoi { poi } => {
                if assembly
                    .points_of_interest
                    .iter()
                    .any(|existing| existing.id == poi.id)
                {
                    return Err(format!("duplicate poi id {}", poi.id));
                }
                assembly.points_of_interest.push(poi.clone());
            }
            StartupMutation::Note { text } => {
                if !text.trim().is_empty() {
                    assembly.meta.notes.push(text.trim().to_string());
                }
            }
        }
    }

    Ok(assembly)
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
            simulation_mode: default_simulation_mode(),
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
            let pivot = *positions
                .get(pivot_joint_id)
                .ok_or_else(|| format!("drive pivot {pivot_joint_id} is not seeded"))?;
            positions.insert(
                tip_joint_id.clone(),
                pivot + vec2(angle.cos(), angle.sin()) * *length,
            );
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

    Ok(assembly
        .joints
        .iter()
        .map(|(joint_id, joint)| {
            let pos = *positions
                .get(joint_id)
                .expect("every joint should have a seeded position");
            (
                joint_id.clone(),
                ParticleState {
                    pos,
                    prev_pos: pos,
                    fixed: matches!(joint, JointSpec::Fixed { .. }),
                },
            )
        })
        .collect())
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

fn verlet_step(particles: &mut BTreeMap<String, ParticleState>) {
    for particle in particles.values_mut() {
        if particle.fixed {
            continue;
        }
        let current = particle.pos;
        let velocity = (particle.pos - particle.prev_pos) * RELAXATION_DAMPING;
        particle.pos += velocity;
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
    let mut particles = initial_state.clone();
    verlet_step(&mut particles);

    let mut max_constraint_error = f32::INFINITY;
    for _ in 0..RELAXATION_ITERATIONS {
        let mut max_correction = 0.0f32;
        max_correction = max_correction.max(project_fixed_constraints(assembly, &mut particles));
        max_correction =
            max_correction.max(project_drive_constraint(drive_constraint, &mut particles));
        max_correction = max_correction.max(project_slider_constraints(sliders, &mut particles));
        max_correction = max_correction.max(project_link_constraints(links, &mut particles));
        max_correction = max_correction.max(project_slider_constraints(sliders, &mut particles));
        max_correction = max_correction.max(project_fixed_constraints(assembly, &mut particles));
        max_correction =
            max_correction.max(project_drive_constraint(drive_constraint, &mut particles));

        if particles.values().any(|particle| {
            !particle.pos.x.is_finite()
                || !particle.pos.y.is_finite()
                || particle.pos.x.abs() > RELAXATION_DIVERGENCE_LIMIT
                || particle.pos.y.abs() > RELAXATION_DIVERGENCE_LIMIT
        }) {
            return Err("relaxation diverged".to_string());
        }

        max_constraint_error =
            compute_max_constraint_error(assembly, &particles, links, sliders, drive_constraint);
        if max_constraint_error <= RELAXATION_TOLERANCE && max_correction <= RELAXATION_TOLERANCE {
            for particle in particles.values_mut() {
                particle.prev_pos = particle.pos;
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

    for particle in particles.values_mut() {
        particle.prev_pos = particle.pos;
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
) -> f32 {
    let mut max_correction = 0.0;
    for (joint_id, joint) in &assembly.joints {
        if let JointSpec::Fixed { position } = joint {
            let anchor = pt2(position[0], position[1]);
            let particle = particles
                .get_mut(joint_id)
                .expect("fixed joint should have a particle");
            max_correction = max_correction.max((anchor - particle.pos).length());
            particle.pos = anchor;
            particle.prev_pos = anchor;
        }
    }
    max_correction
}

fn project_drive_constraint(
    drive_constraint: &DriveConstraint,
    particles: &mut BTreeMap<String, ParticleState>,
) -> f32 {
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
                .expect("drive pivot should exist");
            let target = pivot + vec2(angle.cos(), angle.sin()) * *length;
            let tip = particles
                .get_mut(tip_joint_id)
                .expect("drive tip should exist");
            let correction = (target - tip.pos).length();
            tip.pos = target;
            tip.prev_pos = target;
            correction
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
                .expect("driven slider joint should exist");
            let correction = (target - joint.pos).length();
            joint.pos = target;
            joint.prev_pos = target;
            correction
        }
    }
}

fn project_slider_constraints(
    sliders: &[SliderConstraint],
    particles: &mut BTreeMap<String, ParticleState>,
) -> f32 {
    let mut max_correction = 0.0;
    for slider in sliders {
        let particle = particles
            .get_mut(&slider.joint_id)
            .expect("slider joint should exist");
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
    max_correction
}

fn project_link_constraints(
    links: &[LinkConstraint],
    particles: &mut BTreeMap<String, ParticleState>,
) -> f32 {
    let mut max_correction = 0.0;
    for link in links {
        let (a_pos, a_fixed) = particles
            .get(&link.a)
            .map(|particle| (particle.pos, particle.fixed))
            .expect("link endpoint should exist");
        let (b_pos, b_fixed) = particles
            .get(&link.b)
            .map(|particle| (particle.pos, particle.fixed))
            .expect("link endpoint should exist");
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
                if let Some(a) = particles.get_mut(&link.a) {
                    a.pos += correction * 0.5;
                }
                if let Some(b) = particles.get_mut(&link.b) {
                    b.pos -= correction * 0.5;
                }
                max_correction = max_correction.max(correction.length() * 0.5);
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
    max_correction
}

fn compute_max_constraint_error(
    assembly: &AssemblySpec,
    particles: &BTreeMap<String, ParticleState>,
    links: &[LinkConstraint],
    sliders: &[SliderConstraint],
    drive_constraint: &DriveConstraint,
) -> f32 {
    let mut max_error = 0.0f32;

    for (joint_id, joint) in &assembly.joints {
        if let JointSpec::Fixed { position } = joint {
            let anchor = pt2(position[0], position[1]);
            let particle = particles
                .get(joint_id)
                .expect("fixed joint should have a particle");
            max_error = max_error.max((particle.pos - anchor).length());
        }
    }

    for link in links {
        let a = particles
            .get(&link.a)
            .expect("link endpoint should exist")
            .pos;
        let b = particles
            .get(&link.b)
            .expect("link endpoint should exist")
            .pos;
        max_error = max_error.max(((b - a).length() - link.length).abs());
    }

    for slider in sliders {
        let particle = particles
            .get(&slider.joint_id)
            .expect("slider joint should exist");
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
                .expect("drive pivot should exist")
                .pos;
            let tip = particles
                .get(tip_joint_id)
                .expect("drive tip should exist")
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
                .expect("driven slider joint should exist")
                .pos;
            let target = *axis_origin + *axis_dir * *value;
            (joint - target).length()
        }
    });

    max_error
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
    let mode = assembly.meta.simulation_mode;

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
            match mode {
                SimulationModeSpec::Strict => {
                    return Err(format!(
                        "relaxation failed to settle: max constraint error {:.3}",
                        relaxed.max_constraint_error
                    ));
                }
                SimulationModeSpec::Expressive => {
                    unsettled_samples += 1;
                }
            }
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

fn simulation_mode_label(mode: SimulationModeSpec) -> &'static str {
    match mode {
        SimulationModeSpec::Strict => "STRICT",
        SimulationModeSpec::Expressive => "EXPRESSIVE",
    }
}

fn draw_scene(draw: &Draw, win: Rect, model: &Model) {
    let fixture = current_fixture(model);
    let render_progress = if model.headless {
        1.0
    } else {
        model.playback_progress
    };
    let live_trace_u = if model.headless {
        1.0
    } else {
        model.live_trace_u
    };
    draw_menu_layer(
        draw,
        win,
        model.headless,
        startup_generation_in_progress(model),
        &model.fixtures,
        model.selected_fixture,
    );
    let (spec_rect, render_rect) = content_rects(win);
    draw_spec_pane(draw, spec_rect, fixture, model.spec_scroll_px);
    draw_render_pane(
        draw,
        render_rect,
        fixture,
        render_progress,
        live_trace_u,
        model.playback_paused,
        &model.trace_cycle_entities,
    );
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

    let mode_label = if headless {
        "HEADLESS P3"
    } else {
        "PLAYBACK P3"
    };
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

fn draw_spec_pane(draw: &Draw, rect: Rect, fixture: &FixturePresentation, scroll_px: f32) {
    let text_w = rect.w() - PANE_TEXT_INSET * 2.0;
    let text_x = rect.left() + PANE_TEXT_INSET + text_w * 0.5;
    let body_top = rect.top() - 42.0;
    let body_bottom = rect.bottom() + 14.0;

    draw.text("ASSEMBLY JSON")
        .x_y(text_x, rect.top() - 18.0)
        .w_h(text_w, PANE_LINE_H)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82))
        .left_justify();

    let start_line = (scroll_px / PANE_LINE_H).floor() as usize;
    let intra_line_offset = scroll_px % PANE_LINE_H;
    let mut y = body_top + intra_line_offset;

    for line in fixture.json_lines.iter().skip(start_line) {
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
            .color(rgba(1.0, 1.0, 1.0, 0.60))
            .left_justify();
        y -= PANE_LINE_H;
    }

    let max_scroll = max_spec_scroll_px_for_rect(fixture, rect);
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
    trace_cycle_entities: &[TraceCycleEntity],
) {
    let text_w = rect.w() - PANE_TEXT_INSET * 2.0;
    let text_x = rect.left() + PANE_TEXT_INSET + text_w * 0.5;

    let title = match fixture.status {
        FixtureStatus::Loading(_) => "STARTUP GENERATION",
        FixtureStatus::Solved(_) => "SWEEP PLAYBACK",
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
        FixtureStatus::Loading(_) => "render loop stays live while the request runs",
        FixtureStatus::Solved(_) => {
            let artifact = match &fixture.status {
                FixtureStatus::Solved(artifact) => artifact,
                _ => unreachable!(),
            };
            if artifact.telemetry.unsettled_samples > 0 {
                "expressive mode keeps drawable under-settled motion live"
            } else if playback_paused {
                "space resumes playback"
            } else {
                "space pauses playback"
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
        FixtureStatus::Loading(message) => {
            draw_error_state(&local_draw, local_rect, "OPENAI", message);
        }
        FixtureStatus::Solved(artifact) => {
            draw_grid_local(&local_draw, local_rect);
            let sampled_u = playback_sample_u(playback_progress, artifact.playback);
            let frame = sampled_frame(artifact, sampled_u);
            let solved = solved_assembly_for_frame(&artifact.assembly, frame);
            let bounds = artifact_bounds(artifact);
            let map = make_world_to_local(local_rect, bounds);
            draw_trace_cycle_entities_local(
                &local_draw,
                trace_cycle_entities,
                artifact,
                playback_progress,
                &map,
            );
            draw_poi_traces_local(&local_draw, artifact, live_trace_u, &map);
            draw_solution_local(&local_draw, local_rect, &solved, &map);
            let drive_status = frame
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
                "{}  |  t {:.2}  |  u {:.2}  |  {drive_status}  |  err {:.2}",
                simulation_mode_label(artifact.assembly.meta.simulation_mode),
                playback_progress,
                frame.u,
                artifact.telemetry.peak_constraint_error
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
    draw_slider_local(draw, solved, map);

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
            let color = if joint_id == "j_tip" {
                rgba(0.34, 0.78, 0.98, 0.92)
            } else {
                rgba(1.0, 1.0, 1.0, 0.80)
            };
            draw.ellipse().x_y(p.x, p.y).w_h(12.0, 12.0).color(color);
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

fn draw_slider_local<F>(draw: &Draw, solved: &SolvedAssembly, map: &F)
where
    F: Fn(Point2) -> Point2,
{
    let start = map(solved.slider.start);
    let end = map(solved.slider.end);
    let center = pt2((start.x + end.x) * 0.5, (start.y + end.y) * 0.5);
    let track_w = (end.x - start.x).abs();
    let outer_h = 22.0;
    let inner_h = 12.0;
    let outer_w = track_w.max(outer_h);
    let inner_w = track_w.max(inner_h);

    draw.rect()
        .x_y(center.x, center.y)
        .w_h(outer_w, outer_h)
        .color(rgba(1.0, 1.0, 1.0, 0.22));
    draw.ellipse()
        .x_y(start.x, start.y)
        .w_h(outer_h, outer_h)
        .color(rgba(1.0, 1.0, 1.0, 0.22));
    draw.ellipse()
        .x_y(end.x, end.y)
        .w_h(outer_h, outer_h)
        .color(rgba(1.0, 1.0, 1.0, 0.22));

    draw.rect()
        .x_y(center.x, center.y)
        .w_h(inner_w, inner_h)
        .color(rgba(0.02, 0.025, 0.035, 1.0));
    draw.ellipse()
        .x_y(start.x, start.y)
        .w_h(inner_h, inner_h)
        .color(rgba(0.02, 0.025, 0.035, 1.0));
    draw.ellipse()
        .x_y(end.x, end.y)
        .w_h(inner_h, inner_h)
        .color(rgba(0.02, 0.025, 0.035, 1.0));

    let slider_joint = map(solved.joints[&solved.slider.joint].position);
    draw.ellipse()
        .x_y(slider_joint.x, slider_joint.y)
        .w_h(12.0, 12.0)
        .color(rgba(1.0, 1.0, 1.0, 0.92));
}

fn playback_sample_u(progress: f32, traversal: PlaybackTraversal) -> f32 {
    let phase = progress.rem_euclid(1.0);
    match traversal {
        PlaybackTraversal::Forward => phase,
        // User-facing ping-pong is eased at the turnarounds on purpose.
        PlaybackTraversal::PingPong => 0.5 - 0.5 * (phase * TAU).cos(),
    }
}

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

    let slider = assembly
        .parts
        .values()
        .find_map(|part| match part {
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
        .expect("solved frame requires one slider track");

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
        slider,
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

fn draw_poi_traces_local<F>(draw: &Draw, artifact: &SweepArtifact, current_u: f32, map: &F)
where
    F: Fn(Point2) -> Point2,
{
    // `mark_u` is the static sample position recorded during the sweep. `current_u`
    // is the live trace head within the current playback cycle after traversal mapping.
    // Rolling paper offsets live samples by the distance between those two values.
    let rolling = rolling_paper_config(artifact);
    for (index, (_poi_id, path)) in artifact.telemetry.point_paths.iter().enumerate() {
        for (point_index, window) in path.windows(2).enumerate() {
            if point_index % 2 == 1 {
                continue;
            }
            let len = path.len().max(2);
            let denom = (len - 1) as f32;
            let u_a = point_index as f32 / denom;
            let u_b = (point_index + 1) as f32 / denom;

            let (a_world, b_world) = if let Some((direction, advance_per_cycle)) = rolling {
                if current_u <= u_a {
                    continue;
                }

                let a =
                    rolling_paper_position(window[0], direction, advance_per_cycle, current_u, u_a);
                if current_u >= u_b {
                    let b = rolling_paper_position(
                        window[1],
                        direction,
                        advance_per_cycle,
                        current_u,
                        u_b,
                    );
                    (a, b)
                } else {
                    let t = ((current_u - u_a) / (u_b - u_a).max(0.000_1)).clamp(0.0, 1.0);
                    let point_now = window[0].lerp(window[1], t);
                    (a, point_now)
                }
            } else {
                (window[0], window[1])
            };

            let a = map(a_world);
            let b = map(b_world);
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

fn rolling_paper_position(
    point: Point2,
    direction: Vec2,
    advance_per_cycle: f32,
    current_u: f32,
    mark_u: f32,
) -> Point2 {
    point + direction * (advance_per_cycle * (current_u - mark_u).max(0.0))
}

fn rolling_paper_direction(direction: &PaperDirectionSpec) -> Vec2 {
    match direction {
        PaperDirectionSpec::Up => vec2(0.0, 1.0),
        PaperDirectionSpec::Down => vec2(0.0, -1.0),
        PaperDirectionSpec::Left => vec2(-1.0, 0.0),
        PaperDirectionSpec::Right => vec2(1.0, 0.0),
    }
}

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
    model.spec_scroll_px = next.clamp(0.0, max_spec_scroll_px(current_fixture(model), win));
}

fn visible_spec_body_height(win: Rect) -> f32 {
    let (spec_rect, _) = content_rects(win);
    visible_spec_body_height_for_rect(spec_rect)
}

fn max_spec_scroll_px(fixture: &FixturePresentation, win: Rect) -> f32 {
    let (spec_rect, _) = content_rects(win);
    max_spec_scroll_px_for_rect(fixture, spec_rect)
}

fn visible_spec_body_height_for_rect(spec_rect: Rect) -> f32 {
    spec_rect.h() - 56.0
}

fn max_spec_scroll_px_for_rect(fixture: &FixturePresentation, spec_rect: Rect) -> f32 {
    let available = visible_spec_body_height_for_rect(spec_rect).max(0.0);
    let content = fixture.json_lines.len() as f32 * PANE_LINE_H;
    (content - available).max(0.0)
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
    fn expressive_mode_commits_branchy_drawable_sweep() {
        let mut strict = expressive_branchy_fixture();
        strict.meta.simulation_mode = SimulationModeSpec::Strict;
        match build_sweep_artifact(strict, 1) {
            Ok(_) => panic!("strict mode should reject the impossible brace fixture"),
            Err(err) => assert!(err.contains("failed to settle") || err.contains("diverged")),
        }

        let artifact = build_sweep_artifact(expressive_branchy_fixture(), 1)
            .expect("expressive mode should keep a drawable artifact");
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
    fn unsolved_fixture_fails_relaxation() {
        match build_sweep_artifact(unsolved_slider_crank_fixture(), 1) {
            Ok(_) => panic!("fixture should fail"),
            Err(err) => assert!(err.contains("relaxation")),
        }
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
}
