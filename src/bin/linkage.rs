use nannou::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
const MENU_BAR_H: f32 = 56.0;
const PANE_MARGIN: f32 = 28.0;
const PANE_GUTTER: f32 = 20.0;
const SPEC_RATIO: f32 = 0.38;
const GRID_STEP: f32 = 36.0;

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Clone, Debug)]
struct Config {
    headless: bool,
    capture_path: Option<PathBuf>,
}

struct Model {
    headless: bool,
    capture_path: Option<PathBuf>,
    fixture: FixturePresentation,
    headless_capture_queued: AtomicBool,
}

struct FixturePresentation {
    json_lines: Vec<String>,
    solved: SolvedAssembly,
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
                other => panic!("unknown argument: {other}"),
            }
        }

        if headless && capture_path.is_none() {
            capture_path = Some(PathBuf::from("target/linkage-p1-headless.png"));
        }

        Self {
            headless,
            capture_path,
        }
    }
}

fn model(app: &App) -> Model {
    let config = CONFIG
        .get()
        .expect("linkage config should be available before model()")
        .clone();
    let fixture = build_fixture_presentation().expect("failed to build P1 fixture");

    let mut window = app
        .new_window()
        .title("Linkage")
        .size(WINDOW_W, WINDOW_H)
        .view(view)
        .key_pressed(key_pressed);
    if config.headless {
        window = window.visible(false);
    }
    window.build().unwrap();

    Model {
        headless: config.headless,
        capture_path: config.capture_path,
        fixture,
        headless_capture_queued: AtomicBool::new(false),
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    if key == Key::S {
        let path = model
            .capture_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("target/linkage-windowed-capture.png"));
        app.main_window().capture_frame(path);
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let win = app.window_rect();

    draw.background().color(BLACK);
    draw_scene(&draw, win, &model.fixture, model.headless);

    if model.headless && !model.headless_capture_queued.swap(true, Ordering::SeqCst) {
        if let Some(path) = &model.capture_path {
            app.main_window().capture_frame(path);
            let capture_path = path.clone();
            std::thread::spawn(move || {
                for _ in 0..2400 {
                    if let Ok(bytes) = std::fs::read(&capture_path) {
                        if bytes.len() > 8 && bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
                            std::thread::sleep(Duration::from_millis(100));
                            std::process::exit(0);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                eprintln!(
                    "linkage headless capture timed out waiting for {}",
                    capture_path.display()
                );
                std::process::exit(1);
            });
        }
    }

    draw.to_frame(app, &frame).unwrap();
}

fn build_fixture_presentation() -> Result<FixturePresentation, String> {
    let assembly = slider_crank_fixture();
    validate_fixture(&assembly)?;
    let solved = solve_static_origin(&assembly)?;
    let json = serde_json::to_string_pretty(&assembly)
        .map_err(|err| format!("failed to serialize fixture JSON: {err}"))?;
    let json_lines = json.lines().map(|line| line.to_string()).collect();
    Ok(FixturePresentation { json_lines, solved })
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
                range: [0.78, 6.283185],
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

fn solve_static_origin(assembly: &AssemblySpec) -> Result<SolvedAssembly, String> {
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

fn draw_scene(draw: &Draw, win: Rect, fixture: &FixturePresentation, headless: bool) {
    draw_menu_layer(draw, win, headless);

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

    draw_spec_pane(draw, spec_rect, fixture);
    draw_render_pane(draw, render_rect, &fixture.solved);
}

fn draw_menu_layer(draw: &Draw, win: Rect, headless: bool) {
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
}

fn draw_spec_pane(draw: &Draw, rect: Rect, fixture: &FixturePresentation) {
    draw.rect()
        .xy(rect.xy())
        .wh(rect.wh())
        .color(rgba(0.035, 0.04, 0.055, 1.0));

    draw.text("ASSEMBLY JSON")
        .x_y(rect.left() + 70.0, rect.top() - 18.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82))
        .left_justify();

    draw.line()
        .start(pt2(rect.right(), rect.bottom()))
        .end(pt2(rect.right(), rect.top()))
        .color(rgba(1.0, 1.0, 1.0, 0.18))
        .weight(1.0);

    let mut y = rect.top() - 42.0;
    for line in &fixture.json_lines {
        if y < rect.bottom() + 14.0 {
            break;
        }
        draw.text(line)
            .x_y(rect.left() + 12.0, y)
            .font_size(8)
            .color(rgba(1.0, 1.0, 1.0, 0.46))
            .left_justify();
        y -= 12.0;
    }
}

fn draw_render_pane(draw: &Draw, rect: Rect, solved: &SolvedAssembly) {
    draw.rect()
        .xy(rect.xy())
        .wh(rect.wh())
        .color(rgba(0.018, 0.02, 0.03, 1.0));

    draw.text("STATIC SOLVED ORIGIN")
        .x_y(rect.left() + 88.0, rect.top() - 18.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82))
        .left_justify();

    draw.text("closed-form slider-crank fixture")
        .x_y(rect.left() + 102.0, rect.top() - 34.0)
        .font_size(8)
        .color(rgba(1.0, 1.0, 1.0, 0.40))
        .left_justify();

    let local_draw = draw.x_y(rect.x(), rect.y() - 10.0);
    let local_rect = Rect::from_w_h(rect.w() - 28.0, rect.h() - 72.0);
    draw_grid_local(&local_draw, local_rect);
    draw_solution_local(&local_draw, local_rect, solved);
}

fn draw_grid_local(draw: &Draw, rect: Rect) {
    let mut x = (rect.left() / GRID_STEP).floor() * GRID_STEP;
    while x <= rect.right() {
        let alpha = if x.abs() < 0.5 { 0.20 } else { 0.06 };
        let weight = if x.abs() < 0.5 { 1.25 } else { 1.0 };
        draw.line()
            .start(pt2(x, rect.bottom()))
            .end(pt2(x, rect.top()))
            .color(rgba(1.0, 1.0, 1.0, alpha))
            .weight(weight);
        x += GRID_STEP;
    }

    let mut y = (rect.bottom() / GRID_STEP).floor() * GRID_STEP;
    while y <= rect.top() {
        let alpha = if y.abs() < 0.5 { 0.20 } else { 0.06 };
        let weight = if y.abs() < 0.5 { 1.25 } else { 1.0 };
        draw.line()
            .start(pt2(rect.left(), y))
            .end(pt2(rect.right(), y))
            .color(rgba(1.0, 1.0, 1.0, alpha))
            .weight(weight);
        y += GRID_STEP;
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
