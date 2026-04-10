use nannou::image::{ImageBuffer, ImageError, Rgba};
use nannou::prelude::*;
use std::path::PathBuf;

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
const MENU_BAR_H: f32 = 56.0;
const GRID_STEP: f32 = 36.0;

const BG: [u8; 4] = [0, 0, 0, 255];
const MENU_BG: [u8; 4] = [5, 6, 9, 245];
const GRID_MINOR: [u8; 4] = [34, 38, 44, 255];
const GRID_MAJOR: [u8; 4] = [58, 64, 72, 255];
const LINK: [u8; 4] = [255, 255, 255, 166];
const JOINT: [u8; 4] = [255, 255, 255, 204];
const ACCENT: [u8; 4] = [88, 210, 255, 235];

#[derive(Clone, Debug)]
struct Config {
    headless: bool,
    capture_path: Option<PathBuf>,
}

struct Model {
    capture_path: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct Segment {
    start: Point2,
    end: Point2,
    weight: f32,
    color: LinSrgba,
}

#[derive(Clone, Copy)]
struct Joint {
    center: Point2,
    radius: f32,
    color: LinSrgba,
}

struct SceneGeometry {
    segments: Vec<Segment>,
    joints: Vec<Joint>,
}

fn main() {
    let config = Config::from_args();
    if config.headless {
        render_headless(&config).expect("failed to render headless P0 frame");
        return;
    }

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
            capture_path = Some(PathBuf::from("target/linkage-p0-headless.png"));
        }

        Self {
            headless,
            capture_path,
        }
    }
}

fn model(app: &App) -> Model {
    let config = Config::from_args();
    app.new_window()
        .title("Linkage")
        .size(WINDOW_W, WINDOW_H)
        .view(view)
        .key_pressed(key_pressed)
        .build()
        .unwrap();

    Model {
        capture_path: config.capture_path,
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

fn view(app: &App, _model: &Model, frame: Frame) {
    let draw = app.draw();
    let win = app.window_rect();

    draw.background().color(BLACK);
    draw_scene(&draw, win, app.elapsed_frames());
    draw.to_frame(app, &frame).unwrap();
}

fn draw_scene(draw: &Draw, win: Rect, elapsed_frames: u64) {
    draw_menu_layer(draw, win);
    draw_grid_layer(draw, win);
    draw_geometry(draw, &scene_geometry(elapsed_frames));
    draw_footer(draw, win);
}

fn draw_menu_layer(draw: &Draw, win: Rect) {
    let bar_y = win.top() - MENU_BAR_H * 0.5;
    draw.rect()
        .x_y(0.0, bar_y)
        .w_h(win.w(), MENU_BAR_H)
        .color(rgba(0.02, 0.025, 0.035, 0.96));

    draw.text("LINKAGE")
        .x_y(win.left() + 78.0, win.top() - 24.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82));

    draw.text("WINDOWED P0")
        .x_y(win.right() - 86.0, win.top() - 24.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.68));
}

fn draw_grid_layer(draw: &Draw, win: Rect) {
    let mut x = (win.left() / GRID_STEP).floor() * GRID_STEP;
    while x <= win.right() {
        let alpha = if x.abs() < 0.5 { 0.20 } else { 0.06 };
        let weight = if x.abs() < 0.5 { 1.25 } else { 1.0 };
        draw.line()
            .start(pt2(x, win.bottom()))
            .end(pt2(x, win.top()))
            .color(rgba(0.14, 0.16, 0.18, alpha))
            .weight(weight);
        x += GRID_STEP;
    }

    let mut y = (win.bottom() / GRID_STEP).floor() * GRID_STEP;
    while y <= win.top() {
        let alpha = if y.abs() < 0.5 { 0.20 } else { 0.06 };
        let weight = if y.abs() < 0.5 { 1.25 } else { 1.0 };
        draw.line()
            .start(pt2(win.left(), y))
            .end(pt2(win.right(), y))
            .color(rgba(0.14, 0.16, 0.18, alpha))
            .weight(weight);
        y += GRID_STEP;
    }
}

fn scene_geometry(elapsed_frames: u64) -> SceneGeometry {
    let pulse = ((elapsed_frames as f32) * 0.025).sin() * 0.5 + 0.5;
    let accent_mix = 0.55 + pulse * 0.25;
    let accent = rgba(
        0.18 + accent_mix * 0.20,
        0.62 + accent_mix * 0.20,
        0.95,
        0.92,
    );

    SceneGeometry {
        segments: vec![
            Segment {
                start: pt2(-120.0, -48.0),
                end: pt2(0.0, 72.0),
                weight: 1.5,
                color: rgba(1.0, 1.0, 1.0, 0.65).into_linear(),
            },
            Segment {
                start: pt2(0.0, 72.0),
                end: pt2(144.0, -12.0),
                weight: 1.5,
                color: rgba(1.0, 1.0, 1.0, 0.65).into_linear(),
            },
        ],
        joints: vec![
            Joint {
                center: pt2(-120.0, -48.0),
                radius: 6.0,
                color: rgba(1.0, 1.0, 1.0, 0.80).into_linear(),
            },
            Joint {
                center: pt2(0.0, 72.0),
                radius: 7.0,
                color: accent.into_linear(),
            },
            Joint {
                center: pt2(144.0, -12.0),
                radius: 6.0,
                color: rgba(1.0, 1.0, 1.0, 0.80).into_linear(),
            },
        ],
    }
}

fn draw_geometry(draw: &Draw, scene: &SceneGeometry) {
    for segment in &scene.segments {
        draw.line()
            .start(segment.start)
            .end(segment.end)
            .weight(segment.weight)
            .color(segment.color);
    }

    for joint in &scene.joints {
        draw.ellipse()
            .x_y(joint.center.x, joint.center.y)
            .w_h(joint.radius * 2.0, joint.radius * 2.0)
            .color(joint.color);
    }
}

fn draw_footer(draw: &Draw, win: Rect) {
    draw.text("P0")
        .x_y(0.0, win.bottom() + 110.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.70));

    draw.text("render loop alive")
        .x_y(0.0, win.bottom() + 92.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.45));

    draw.text("grid + title + minimal linkage scaffold")
        .x_y(0.0, win.bottom() + 28.0)
        .font_size(8)
        .color(rgba(1.0, 1.0, 1.0, 0.35));
}

fn render_headless(config: &Config) -> Result<(), ImageError> {
    let path = config
        .capture_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/linkage-p0-headless.png"));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create headless capture directory");
    }

    let mut img = ImageBuffer::from_pixel(WINDOW_W, WINDOW_H, Rgba(BG));
    fill_rect(&mut img, 0, 0, WINDOW_W, MENU_BAR_H as u32, MENU_BG);

    let half_w = WINDOW_W as f32 * 0.5;
    let half_h = WINDOW_H as f32 * 0.5;
    let mut x = (-half_w / GRID_STEP).floor() * GRID_STEP;
    while x <= half_w {
        let color = if x.abs() < 0.5 {
            GRID_MAJOR
        } else {
            GRID_MINOR
        };
        draw_line_pixels(
            &mut img,
            world_to_pixel(pt2(x, -half_h)),
            world_to_pixel(pt2(x, half_h)),
            color,
        );
        x += GRID_STEP;
    }

    let mut y = (-half_h / GRID_STEP).floor() * GRID_STEP;
    while y <= half_h {
        let color = if y.abs() < 0.5 {
            GRID_MAJOR
        } else {
            GRID_MINOR
        };
        draw_line_pixels(
            &mut img,
            world_to_pixel(pt2(-half_w, y)),
            world_to_pixel(pt2(half_w, y)),
            color,
        );
        y += GRID_STEP;
    }

    let scene = scene_geometry(0);
    for segment in &scene.segments {
        draw_line_pixels(
            &mut img,
            world_to_pixel(segment.start),
            world_to_pixel(segment.end),
            LINK,
        );
    }

    for (index, joint) in scene.joints.iter().enumerate() {
        let color = if index == 1 { ACCENT } else { JOINT };
        fill_circle(
            &mut img,
            world_to_pixel(joint.center),
            joint.radius as i32,
            color,
        );
    }

    img.save(&path)?;
    println!("wrote headless frame to {}", path.display());
    Ok(())
}

fn fill_rect(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    color: [u8; 4],
) {
    let x1 = (x0 + w).min(img.width());
    let y1 = (y0 + h).min(img.height());
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, Rgba(color));
        }
    }
}

fn draw_line_pixels(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    start: (i32, i32),
    end: (i32, i32),
    color: [u8; 4],
) {
    let (mut x0, mut y0) = start;
    let (x1, y1) = end;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && y0 >= 0 && x0 < img.width() as i32 && y0 < img.height() as i32 {
            img.put_pixel(x0 as u32, y0 as u32, Rgba(color));
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = err * 2;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn fill_circle(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    center: (i32, i32),
    radius: i32,
    color: [u8; 4],
) {
    let (cx, cy) = center;
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y > radius * radius {
                continue;
            }
            let px = cx + x;
            let py = cy + y;
            if px >= 0 && py >= 0 && px < img.width() as i32 && py < img.height() as i32 {
                img.put_pixel(px as u32, py as u32, Rgba(color));
            }
        }
    }
}

fn world_to_pixel(point: Point2) -> (i32, i32) {
    let x = (point.x + WINDOW_W as f32 * 0.5).round() as i32;
    let y = (WINDOW_H as f32 * 0.5 - point.y).round() as i32;
    (x, y)
}
