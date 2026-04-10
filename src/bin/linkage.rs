use nannou::prelude::*;
use serde::Serialize;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
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

static CONFIG: OnceLock<Config> = OnceLock::new();

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
    playback_phase: f32,
    playback_paused: bool,
    active_artifact_turn: Option<u32>,
    headless_capture_state: Cell<HeadlessCaptureState>,
    headless_capture_result: Arc<AtomicU8>,
    headless_proxy: Option<nannou::app::Proxy>,
    spec_scroll_px: f32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum HeadlessCaptureState {
    Pending,
    Queued,
    Flushed,
}

struct FixturePresentation {
    label: &'static str,
    status: FixtureStatus,
    json_lines: Vec<String>,
}

enum FixtureStatus {
    Solved(Arc<SweepArtifact>),
    ValidationError(String),
    SolverError(String),
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
    parameter: DriveParameter,
    start_value: f32,
    end_value: f32,
    cycle_seconds: f32,
    playback: PlaybackTraversal,
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

#[derive(Clone, Serialize)]
struct AssemblySpec {
    joints: BTreeMap<String, JointSpec>,
    parts: BTreeMap<String, PartSpec>,
    drives: BTreeMap<String, DriveSpec>,
    points_of_interest: Vec<PointOfInterestSpec>,
    meta: AssemblyMeta,
}

#[derive(Clone, Serialize)]
#[serde(tag = "type")]
enum JointSpec {
    Fixed { position: [f32; 2] },
    Free,
}

#[derive(Clone, Serialize)]
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

#[derive(Clone, Serialize)]
struct DriveSpec {
    kind: DriveKindSpec,
    sweep: SweepSpec,
}

#[derive(Clone, Serialize)]
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

#[derive(Clone, Serialize)]
struct SweepSpec {
    samples: u32,
    direction: &'static str,
}

#[derive(Clone, Serialize)]
struct PointOfInterestSpec {
    id: String,
    host: String,
    t: f32,
    perp: f32,
}

#[derive(Clone, Serialize)]
struct AssemblyMeta {
    name: String,
    iteration: u32,
    notes: Vec<String>,
}

fn main() {
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
    let fixtures = build_fixture_presentations().expect("failed to build fixture bank");

    let mut window = app
        .new_window()
        .title("Linkage")
        .size(WINDOW_W, WINDOW_H)
        .view(view)
        .key_pressed(key_pressed)
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
        playback_phase: 0.0,
        playback_paused: false,
        active_artifact_turn: None,
        headless_capture_state: Cell::new(HeadlessCaptureState::Pending),
        headless_capture_result: Arc::new(AtomicU8::new(0)),
        headless_proxy: if config.headless {
            Some(app.create_proxy())
        } else {
            None
        },
        spec_scroll_px: 0.0,
    }
}

fn update(app: &App, model: &mut Model, update: Update) {
    if let Some((turn, reset_phase, cycle_seconds)) = current_artifact(model)
        .map(|artifact| (artifact.turn, artifact.reset_phase, artifact.cycle_seconds))
    {
        if model.active_artifact_turn != Some(turn) {
            if reset_phase {
                model.playback_phase = 0.0;
            }
            model.active_artifact_turn = Some(turn);
        }

        if !model.playback_paused {
            let delta = update.since_last.as_secs_f32() / cycle_seconds.max(0.001);
            model.playback_phase = (model.playback_phase + delta).rem_euclid(1.0);
        }
    } else {
        model.active_artifact_turn = None;
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
    match key {
        Key::S => {
            let path = model
                .capture_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("target/linkage-windowed-capture.png"));
            app.main_window().capture_frame(path);
        }
        Key::Space => model.playback_paused = !model.playback_paused,
        Key::Key1 => select_fixture(model, 0),
        Key::Key2 => select_fixture(model, 1),
        Key::Key3 => select_fixture(model, 2),
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

    if model.headless && model.headless_capture_state.get() == HeadlessCaptureState::Pending {
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
        FixtureStatus::Solved(artifact) => Some(artifact),
        FixtureStatus::ValidationError(_) | FixtureStatus::SolverError(_) => None,
    }
}

fn build_fixture_presentations() -> Result<Vec<FixturePresentation>, String> {
    let fixtures = vec![
        ("1 VALID", slider_crank_fixture()),
        ("2 BAD REF", invalid_reference_fixture()),
        ("3 UNSOLVED", unsolved_slider_crank_fixture()),
    ];

    fixtures
        .into_iter()
        .map(|(label, assembly)| build_fixture_presentation(label, assembly))
        .collect()
}

fn build_fixture_presentation(
    label: &'static str,
    assembly: AssemblySpec,
) -> Result<FixturePresentation, String> {
    let json = serde_json::to_string_pretty(&assembly)
        .map_err(|err| format!("failed to serialize fixture JSON: {err}"))?;
    let status = match validate_fixture(&assembly) {
        Ok(()) => match build_sweep_artifact(assembly.clone(), 1) {
            Ok(artifact) => FixtureStatus::Solved(Arc::new(artifact)),
            Err(err) => FixtureStatus::SolverError(err),
        },
        Err(err) => FixtureStatus::ValidationError(err),
    };
    let json_lines = json.lines().map(|line| line.to_string()).collect();
    Ok(FixturePresentation {
        label,
        status,
        json_lines,
    })
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
                direction: "Clockwise",
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
        meta: AssemblyMeta {
            name: "p1-slider-crank".to_string(),
            iteration: 1,
            notes: vec!["P1 closed-form hard-coded fixture".to_string()],
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
    assembly.meta.notes = vec!["Intentional solver range failure".to_string()];
    assembly
}

fn validate_fixture(assembly: &AssemblySpec) -> Result<(), String> {
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
                        "part {part_id}: P1 fixture renderer requires a horizontal slider axis"
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
                if !matches!(
                    drive.sweep.direction,
                    "Clockwise" | "CW" | "Forward" | "CounterClockwise" | "CCW" | "Reverse"
                ) {
                    return Err(format!(
                        "drive {drive_id}: unsupported angular direction {}",
                        drive.sweep.direction
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

    Ok(())
}

fn solve_slider_crank_pose(assembly: &AssemblySpec, theta: f32) -> Result<SolvedAssembly, String> {
    let pivot = match assembly.joints.get("j_pivot") {
        Some(JointSpec::Fixed { position }) => pt2(position[0], position[1]),
        _ => return Err("fixture missing fixed pivot".to_string()),
    };

    let crank_length = match assembly.parts.get("l_crank") {
        Some(PartSpec::Link { length, .. }) => *length,
        _ => return Err("fixture missing crank link".to_string()),
    };
    let coupler_length = match assembly.parts.get("l_coupler") {
        Some(PartSpec::Link { length, .. }) => *length,
        _ => return Err("fixture missing coupler link".to_string()),
    };
    let (axis_origin, slider_range) = match assembly.parts.get("s_track") {
        Some(PartSpec::Slider {
            axis_origin,
            axis_dir,
            range,
            ..
        }) => {
            if axis_dir[0].abs() < 0.999 || axis_dir[1].abs() > 0.001 {
                return Err("P1 fixture solver assumes a horizontal slider axis".to_string());
            }
            (pt2(axis_origin[0], axis_origin[1]), (range[0], range[1]))
        }
        _ => return Err("fixture missing slider track".to_string()),
    };

    let tip = pt2(
        pivot.x + crank_length * theta.cos(),
        pivot.y + crank_length * theta.sin(),
    );
    let dy = tip.y - axis_origin.y;
    if dy.abs() > coupler_length {
        return Err("coupler cannot reach slider axis at origin pose".to_string());
    }

    let slider_offset = (coupler_length * coupler_length - dy * dy).sqrt();
    let slider_x = tip.x + slider_offset;
    if slider_x < slider_range.0 || slider_x > slider_range.1 {
        return Err("origin pose places slider outside its allowed range".to_string());
    }
    let slide = pt2(slider_x, axis_origin.y);

    let mut joints = BTreeMap::new();
    joints.insert(
        "j_pivot".to_string(),
        SolvedJoint {
            position: pivot,
            fixed: true,
        },
    );
    joints.insert(
        "j_tip".to_string(),
        SolvedJoint {
            position: tip,
            fixed: false,
        },
    );
    joints.insert(
        "j_slide".to_string(),
        SolvedJoint {
            position: slide,
            fixed: false,
        },
    );
    let poi_positions = solve_poi_positions(assembly, &joints)?;

    Ok(SolvedAssembly {
        joints,
        links: vec![
            SolvedLink {
                a: "j_pivot".to_string(),
                b: "j_tip".to_string(),
            },
            SolvedLink {
                a: "j_tip".to_string(),
                b: "j_slide".to_string(),
            },
        ],
        slider: SolvedSlider {
            start: pt2(slider_range.0, axis_origin.y),
            end: pt2(slider_range.1, axis_origin.y),
            joint: "j_slide".to_string(),
        },
        pois: poi_positions
            .into_iter()
            .map(|(_, position)| SolvedPoi { position })
            .collect(),
    })
}

fn solve_slider_crank_from_slider(
    assembly: &AssemblySpec,
    slider_value: f32,
) -> Result<SolvedAssembly, String> {
    let pivot = match assembly.joints.get("j_pivot") {
        Some(JointSpec::Fixed { position }) => pt2(position[0], position[1]),
        _ => return Err("fixture missing fixed pivot".to_string()),
    };
    let crank_length = match assembly.parts.get("l_crank") {
        Some(PartSpec::Link { length, .. }) => *length,
        _ => return Err("fixture missing crank link".to_string()),
    };
    let coupler_length = match assembly.parts.get("l_coupler") {
        Some(PartSpec::Link { length, .. }) => *length,
        _ => return Err("fixture missing coupler link".to_string()),
    };
    let (axis_origin, axis_dir) = match assembly.parts.get("s_track") {
        Some(PartSpec::Slider {
            axis_origin,
            axis_dir,
            ..
        }) => {
            if axis_dir[0].abs() < 0.999 || axis_dir[1].abs() > 0.001 {
                return Err("P2 slider solver assumes a horizontal slider axis".to_string());
            }
            (
                pt2(axis_origin[0], axis_origin[1]),
                vec2(axis_dir[0], axis_dir[1]),
            )
        }
        _ => return Err("fixture missing slider track".to_string()),
    };

    let slide = pt2(
        axis_origin.x + axis_dir.x * slider_value,
        axis_origin.y + axis_dir.y * slider_value,
    );
    let delta = slide - pivot;
    let distance = delta.length();
    if distance > crank_length + coupler_length
        || distance < (crank_length - coupler_length).abs()
        || distance <= 0.000_1
    {
        return Err("slider position cannot be reached by the current link lengths".to_string());
    }

    let a = (crank_length * crank_length - coupler_length * coupler_length + distance * distance)
        / (2.0 * distance);
    let h_sq = crank_length * crank_length - a * a;
    if h_sq < 0.0 {
        return Err("slider position leads to an imaginary crank intersection".to_string());
    }
    let mid = pivot + delta * (a / distance);
    let perp = vec2(-delta.y / distance, delta.x / distance);
    let h = h_sq.sqrt();
    let candidate_a = mid + perp * h;
    let candidate_b = mid - perp * h;
    let tip = if candidate_a.y >= candidate_b.y {
        candidate_a
    } else {
        candidate_b
    };
    let theta = (tip.y - pivot.y).atan2(tip.x - pivot.x);
    solve_slider_crank_pose(assembly, theta)
}

fn build_sweep_artifact(assembly: AssemblySpec, turn: u32) -> Result<SweepArtifact, String> {
    let Some((drive_id, drive)) = assembly.drives.iter().next() else {
        return Err("fixture missing drive".to_string());
    };
    let plan = drive_plan(&assembly, drive_id, drive)?;
    let samples = drive.sweep.samples.max(2) as usize;

    let mut frames = Vec::with_capacity(samples);
    let mut point_paths: BTreeMap<String, Vec<Point2>> = BTreeMap::new();

    for sample_idx in 0..samples {
        let denom = (samples - 1).max(1) as f32;
        let base_u = sample_idx as f32 / denom;
        let drive_value = sample_drive_value(&plan, base_u);
        let solved = match plan.parameter {
            DriveParameter::Angle => solve_slider_crank_pose(&assembly, drive_value)?,
            DriveParameter::SliderPosition => {
                solve_slider_crank_from_slider(&assembly, drive_value)?
            }
        };
        let poi_positions = solve_poi_positions(&assembly, &solved.joints)?;
        for (poi_id, position) in &poi_positions {
            point_paths
                .entry(poi_id.clone())
                .or_default()
                .push(*position);
        }

        frames.push(SolvedFrame {
            u: base_u,
            joint_positions: solved
                .joints
                .iter()
                .map(|(joint_id, joint)| (joint_id.clone(), joint.position))
                .collect(),
            drive_values: BTreeMap::from([(plan.drive_id.clone(), drive_value)]),
            poi_positions,
        });
    }

    Ok(SweepArtifact {
        assembly: Arc::new(assembly),
        frames,
        telemetry: SweepTelemetry { point_paths },
        turn,
        reset_phase: false,
        cycle_seconds: plan.cycle_seconds,
        playback: plan.playback,
    })
}

fn solve_poi_positions(
    assembly: &AssemblySpec,
    joints: &BTreeMap<String, SolvedJoint>,
) -> Result<BTreeMap<String, Point2>, String> {
    let mut positions = BTreeMap::new();
    for poi in &assembly.points_of_interest {
        let Some(host_part) = assembly.parts.get(&poi.host) else {
            return Err(format!("poi {}: missing host part {}", poi.id, poi.host));
        };
        let PartSpec::Link { a, b, .. } = host_part else {
            return Err(format!("poi {}: host {} is not a link", poi.id, poi.host));
        };
        let Some(a_joint) = joints.get(a) else {
            return Err(format!("poi {}: missing host joint {}", poi.id, a));
        };
        let Some(b_joint) = joints.get(b) else {
            return Err(format!("poi {}: missing host joint {}", poi.id, b));
        };
        let a_pos = a_joint.position;
        let b_pos = b_joint.position;
        let along = b_pos - a_pos;
        let normal = if along.length_squared() > 0.0 {
            vec2(-along.y, along.x).normalize()
        } else {
            vec2(0.0, 0.0)
        };
        let pos = a_pos + along * poi.t + normal * poi.perp;
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
            let direction = drive.sweep.direction;
            let (start_value, end_value, playback) = match range {
                Some([start, end]) => {
                    if matches!(direction, "CounterClockwise" | "CCW" | "Reverse") {
                        (*end, *start, PlaybackTraversal::PingPong)
                    } else {
                        (*start, *end, PlaybackTraversal::PingPong)
                    }
                }
                None => match direction {
                    "CounterClockwise" | "CCW" => (0.0, TAU, PlaybackTraversal::Forward),
                    "Clockwise" | "CW" | "Forward" => (TAU, 0.0, PlaybackTraversal::Forward),
                    other => {
                        return Err(format!(
                            "drive {drive_id}: unsupported angular direction {other}"
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
            let [start_value, end_value] = range.unwrap_or(*slider_range);
            Ok(DrivePlan {
                drive_id: drive_id.to_string(),
                parameter: DriveParameter::SliderPosition,
                start_value,
                end_value,
                cycle_seconds: 1.5,
                playback: PlaybackTraversal::PingPong,
            })
        }
    }
}

fn sample_drive_value(plan: &DrivePlan, u: f32) -> f32 {
    let t = playback_sample_u(u, plan.playback);
    plan.start_value + (plan.end_value - plan.start_value) * t
}

fn draw_scene(draw: &Draw, win: Rect, model: &Model) {
    let fixture = current_fixture(model);
    draw_menu_layer(
        draw,
        win,
        model.headless,
        &model.fixtures,
        model.selected_fixture,
    );
    let (spec_rect, render_rect) = content_rects(win);
    draw_spec_pane(draw, spec_rect, fixture, model.spec_scroll_px);
    draw_render_pane(
        draw,
        render_rect,
        fixture,
        model.playback_phase,
        model.playback_paused,
    );
}

fn draw_menu_layer(
    draw: &Draw,
    win: Rect,
    headless: bool,
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
        "HEADLESS P2"
    } else {
        "PLAYBACK P2"
    };
    draw.text(mode_label)
        .x_y(win.right() - 78.0, win.top() - 24.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.68));

    let start_x = win.left() + 230.0;
    let step = 86.0;
    let indicator_y = win.top() - 36.0;
    for (index, fixture) in fixtures.iter().enumerate() {
        let x = start_x + index as f32 * step;
        draw.text(fixture.label)
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
    playback_phase: f32,
    playback_paused: bool,
) {
    let text_w = rect.w() - PANE_TEXT_INSET * 2.0;
    let text_x = rect.left() + PANE_TEXT_INSET + text_w * 0.5;

    let title = match fixture.status {
        FixtureStatus::Solved(_) => "SWEEP PLAYBACK",
        FixtureStatus::ValidationError(_) => "VALIDATION FAILURE",
        FixtureStatus::SolverError(_) => "SOLVER FAILURE",
    };
    draw.text(title)
        .x_y(text_x, rect.top() - 18.0)
        .w_h(text_w, PANE_LINE_H)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82))
        .left_justify();

    let subtitle = match fixture.status {
        FixtureStatus::Solved(_) => {
            if playback_paused {
                "space resumes playback"
            } else {
                "space pauses playback"
            }
        }
        FixtureStatus::ValidationError(_) => "validator rejected the selected fixture",
        FixtureStatus::SolverError(_) => "solver could not produce a static origin pose",
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
            let sampled_u = playback_sample_u(playback_phase, artifact.playback);
            let frame = sampled_frame(artifact, sampled_u);
            let solved = solved_assembly_for_frame(&artifact.assembly, frame);
            let bounds = artifact_bounds(artifact);
            let map = make_world_to_local(local_rect, bounds);
            draw_poi_traces_local(&local_draw, artifact, &map);
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
            let status = format!("u {:.2}  |  {drive_status}", frame.u);
            draw.text(&status)
                .x_y(text_x, rect.top() - 50.0)
                .w_h(text_w, PANE_LINE_H)
                .font_size(8)
                .color(rgba(1.0, 1.0, 1.0, 0.34))
                .left_justify();
        }
        FixtureStatus::ValidationError(message) => {
            draw_error_state(&local_draw, local_rect, "VALIDATOR", message);
        }
        FixtureStatus::SolverError(message) => {
            draw_error_state(&local_draw, local_rect, "SOLVER", message);
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

fn playback_sample_u(phase: f32, traversal: PlaybackTraversal) -> f32 {
    match traversal {
        PlaybackTraversal::Forward => phase,
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

fn draw_poi_traces_local<F>(draw: &Draw, artifact: &SweepArtifact, map: &F)
where
    F: Fn(Point2) -> Point2,
{
    for (index, path) in artifact.telemetry.point_paths.values().enumerate() {
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
    fn interval_drive_ping_pong_eases_at_the_turnarounds() {
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
        assert_eq!(sample_drive_value(&plan, 0.0).to_bits(), 0.25f32.to_bits());
        assert!((sample_drive_value(&plan, 0.5) - 1.25).abs() < 0.000_1);
        assert!((sample_drive_value(&plan, 1.0) - 0.25).abs() < 0.000_1);
        let early = sample_drive_value(&plan, 0.05) - 0.25;
        let middle = sample_drive_value(&plan, 0.30) - sample_drive_value(&plan, 0.25);
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
                    direction: "Forward",
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
