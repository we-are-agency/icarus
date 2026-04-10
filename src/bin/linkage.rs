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
const P1_START_ANGLE: f32 = 0.78;

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
    Solved(SolvedAssembly),
    ValidationError(String),
    SolverError(String),
}

struct SolvedAssembly {
    joints: BTreeMap<String, SolvedJoint>,
    links: Vec<SolvedLink>,
    slider: SolvedSlider,
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

#[derive(Serialize)]
struct AssemblySpec {
    joints: BTreeMap<String, JointSpec>,
    parts: BTreeMap<String, PartSpec>,
    drives: BTreeMap<String, DriveSpec>,
    points_of_interest: Vec<PointOfInterestSpec>,
    meta: AssemblyMeta,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum JointSpec {
    Fixed { position: [f32; 2] },
    Free,
}

#[derive(Serialize)]
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

#[derive(Serialize)]
struct DriveSpec {
    kind: DriveKindSpec,
    sweep: SweepSpec,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum DriveKindSpec {
    Angular {
        pivot_joint: String,
        tip_joint: String,
        link: String,
        range: [f32; 2],
    },
}

#[derive(Serialize)]
struct SweepSpec {
    samples: u32,
    direction: &'static str,
}

#[derive(Serialize)]
struct PointOfInterestSpec {
    id: String,
    host: String,
    t: f32,
    perp: f32,
}

#[derive(Serialize)]
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
                    fixture_index = value
                        .parse::<usize>()
                        .expect("--fixture must be a zero-based integer");
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

fn update(app: &App, model: &mut Model, _update: Update) {
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
        Ok(()) => match solve_slider_crank_origin(&assembly) {
            Ok(solved) => FixtureStatus::Solved(solved),
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
            position: [0.0, 0.0],
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
            range: [-80.0, 260.0],
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
                range: [P1_START_ANGLE, TAU],
            },
            sweep: SweepSpec {
                samples: 180,
                direction: "Forward",
            },
        },
    );

    AssemblySpec {
        joints,
        parts,
        drives,
        points_of_interest: Vec::new(),
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
    if let Some(PartSpec::Link { length, .. }) = assembly.parts.get_mut("l_coupler") {
        *length = 30.0;
    }
    assembly.meta.name = "p1-unsolved-slider-crank".to_string();
    assembly.meta.notes = vec!["Intentional solver failure".to_string()];
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
            }
        }
    }

    Ok(())
}

fn solve_slider_crank_origin(assembly: &AssemblySpec) -> Result<SolvedAssembly, String> {
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
    let theta = match assembly.drives.get("d_crank") {
        Some(DriveSpec {
            kind: DriveKindSpec::Angular { range, .. },
            ..
        }) => range[0],
        _ => return Err("fixture missing crank drive".to_string()),
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
    })
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
    draw_render_pane(draw, render_rect, fixture);
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

    let mode_label = if headless { "HEADLESS P1" } else { "STATIC P1" };
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

fn draw_render_pane(draw: &Draw, rect: Rect, fixture: &FixturePresentation) {
    let text_w = rect.w() - PANE_TEXT_INSET * 2.0;
    let text_x = rect.left() + PANE_TEXT_INSET + text_w * 0.5;

    let title = match fixture.status {
        FixtureStatus::Solved(_) => "STATIC SOLVED ORIGIN",
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
        FixtureStatus::Solved(_) => "closed-form slider-crank fixture",
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
        FixtureStatus::Solved(solved) => {
            draw_grid_local(&local_draw, local_rect);
            draw_solution_local(&local_draw, local_rect, solved);
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

fn draw_solution_local(draw: &Draw, rect: Rect, solved: &SolvedAssembly) {
    let bounds = solved_bounds(solved);
    let pane_w = rect.w() - 120.0;
    let pane_h = rect.h() - 120.0;
    let world_w = (bounds.right() - bounds.left()).max(1.0);
    let world_h = (bounds.top() - bounds.bottom()).max(1.0);
    let scale = (pane_w / world_w).min(pane_h / world_h);
    let world_center = pt2(
        (bounds.left() + bounds.right()) * 0.5,
        (bounds.bottom() + bounds.top()) * 0.5,
    );

    let map = |point: Point2| {
        pt2(
            (point.x - world_center.x) * scale,
            (point.y - world_center.y) * scale,
        )
    };

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
}

fn draw_error_state(draw: &Draw, rect: Rect, source: &str, message: &str) {
    draw_grid_local(draw, rect);
    let text_w = rect.w() - 120.0;
    let text_x = rect.left() + 28.0 + text_w * 0.5;

    draw.text(source)
        .x_y(text_x, rect.top() - 36.0)
        .w_h(text_w, 16.0)
        .font_size(9)
        .color(rgba(1.0, 0.48, 0.48, 0.88))
        .left_justify();

    draw.text(message)
        .x_y(text_x, rect.top() - 62.0)
        .w_h(text_w, 64.0)
        .font_size(8)
        .color(rgba(1.0, 1.0, 1.0, 0.70))
        .left_justify();
}

fn draw_slider_local<F>(draw: &Draw, solved: &SolvedAssembly, map: F)
where
    F: Fn(Point2) -> Point2,
{
    let start = map(solved.slider.start);
    let end = map(solved.slider.end);
    let dash_count = 18;
    for dash in 0..dash_count {
        if dash % 2 == 1 {
            continue;
        }
        let t0 = dash as f32 / dash_count as f32;
        let t1 = (dash + 1) as f32 / dash_count as f32;
        let p0 = start.lerp(end, t0);
        let p1 = start.lerp(end, t1);
        draw.line()
            .start(p0)
            .end(p1)
            .weight(1.0)
            .color(rgba(1.0, 1.0, 1.0, 0.32));
    }

    let cap = 8.0;
    draw.line()
        .start(pt2(start.x, start.y - cap))
        .end(pt2(start.x, start.y + cap))
        .weight(1.0)
        .color(rgba(1.0, 1.0, 1.0, 0.40));
    draw.line()
        .start(pt2(end.x, end.y - cap))
        .end(pt2(end.x, end.y + cap))
        .weight(1.0)
        .color(rgba(1.0, 1.0, 1.0, 0.40));

    let slider_joint = map(solved.joints[&solved.slider.joint].position);
    draw.rect()
        .x_y(slider_joint.x, slider_joint.y)
        .w_h(16.0, 10.0)
        .color(rgba(0.34, 0.78, 0.98, 0.22));
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

fn solved_bounds(solved: &SolvedAssembly) -> Rect {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for joint in solved.joints.values() {
        min_x = min_x.min(joint.position.x);
        max_x = max_x.max(joint.position.x);
        min_y = min_y.min(joint.position.y);
        max_y = max_y.max(joint.position.y);
    }
    min_x = min_x.min(solved.slider.start.x);
    max_x = max_x.max(solved.slider.end.x);
    min_y = min_y.min(solved.slider.start.y);
    max_y = max_y.max(solved.slider.end.y);

    Rect::from_corners(
        pt2(min_x - 20.0, min_y - 20.0),
        pt2(max_x + 20.0, max_y + 20.0),
    )
}
