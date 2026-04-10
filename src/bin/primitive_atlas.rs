use nannou::image::{DynamicImage, ImageBuffer, Rgba};
use nannou::prelude::*;

const WINDOW_W: u32 = 1660;
const WINDOW_H: u32 = 1080;
const HEADER_H: f32 = 118.0;
const OUTER_MARGIN: f32 = 28.0;
const GUTTER: f32 = 20.0;
const GRID_BACKDROP: [f32; 4] = [0.055, 0.075, 0.105, 1.0];
const CARD_FILL: [f32; 4] = [0.09, 0.11, 0.15, 0.94];
const CARD_STROKE: [f32; 4] = [0.56, 0.68, 0.84, 0.18];
const LABEL: [f32; 4] = [0.96, 0.98, 1.0, 0.98];
const MUTED: [f32; 4] = [0.73, 0.79, 0.88, 0.72];
const CODE: [f32; 4] = [0.86, 0.92, 1.0, 0.64];
const PANEL_FILL: [f32; 4] = [0.085, 0.105, 0.145, 0.96];
const PANEL_SAMPLE_FILL: [f32; 4] = [0.12, 0.14, 0.19, 0.92];
const PANEL_SAMPLE_STROKE: [f32; 4] = [0.85, 0.93, 1.0, 0.06];

const SCREENS: [ScreenKind; 23] = [
    ScreenKind::Primitives,
    ScreenKind::Transforms,
    ScreenKind::HelloShader,
    ScreenKind::ChladniShader,
    ScreenKind::Chladni,
    ScreenKind::NeuronsShader,
    ScreenKind::Neurons,
    ScreenKind::FluidShader,
    ScreenKind::Fluid,
    ScreenKind::TreeShader,
    ScreenKind::Tree,
    ScreenKind::SinhShader,
    ScreenKind::Sinh,
    ScreenKind::DarkCloudsShader,
    ScreenKind::DarkClouds,
    ScreenKind::MandelbulbShader,
    ScreenKind::Mandelbulb,
    ScreenKind::SmoothVoroShader,
    ScreenKind::SmoothVoro,
    ScreenKind::WinterflakeShader,
    ScreenKind::Winterflake,
    ScreenKind::GPUAttractorShader,
    ScreenKind::GPUAttractor,
];

const PRIMITIVES: [PrimitiveKind; 13] = [
    PrimitiveKind::Background,
    PrimitiveKind::Rect,
    PrimitiveKind::Ellipse,
    PrimitiveKind::Line,
    PrimitiveKind::Arrow,
    PrimitiveKind::Tri,
    PrimitiveKind::Quad,
    PrimitiveKind::Polygon,
    PrimitiveKind::Polyline,
    PrimitiveKind::Path,
    PrimitiveKind::Mesh,
    PrimitiveKind::Text,
    PrimitiveKind::Texture,
];

const SHADER_VERTICES: [ShaderVertex; 6] = [
    ShaderVertex {
        position: [-1.0, -1.0],
    },
    ShaderVertex {
        position: [1.0, -1.0],
    },
    ShaderVertex {
        position: [-1.0, 1.0],
    },
    ShaderVertex {
        position: [1.0, -1.0],
    },
    ShaderVertex {
        position: [1.0, 1.0],
    },
    ShaderVertex {
        position: [-1.0, 1.0],
    },
];

struct Model {
    swatch_texture: wgpu::Texture,
    hello_shader: ShaderPass,
    chladni_shader: ShaderPass,
    neurons_shader: ShaderPass,
    darkclouds_shader: ShaderPass,
    gpuattractor_shader: ShaderPass,
    tree_shader: ShaderPass,
    sinh_shader: ShaderPass,
    mandelbulb_shader: ShaderPass,
    winterflake_shader: ShaderPass,
    fluid_shader: TexturedShaderPass,
    smoothvoro_shader: TexturedShaderPass,
    screen_index: usize,
}

struct ShaderPass {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
}

struct TexturedShaderPass {
    texture_bind_group: wgpu::BindGroup,
    uniform_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct HelloShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct ChladniShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct NeuronsShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct DarkCloudsShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct GPUAttractorShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct TreeShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct SinhShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    params2: [f32; 4],
    params3: [f32; 4],
    params4: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct MandelbulbShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    params2: [f32; 4],
    params3: [f32; 4],
    params4: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct WinterflakeShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    params2: [f32; 4],
    params3: [f32; 4],
    params4: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct FluidShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct SmoothVoroShaderUniforms {
    stage_rect: [f32; 4],
    time_data: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

#[derive(Clone, Copy)]
struct TileLayout {
    frame: Rect,
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderVertex {
    position: [f32; 2],
}

#[derive(Clone, Copy)]
struct PixelRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Clone, Copy)]
enum PrimitiveKind {
    Background,
    Rect,
    Ellipse,
    Line,
    Arrow,
    Tri,
    Quad,
    Polygon,
    Polyline,
    Path,
    Mesh,
    Text,
    Texture,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScreenKind {
    Primitives,
    Transforms,
    HelloShader,
    ChladniShader,
    Chladni,
    NeuronsShader,
    Neurons,
    FluidShader,
    Fluid,
    TreeShader,
    Tree,
    SinhShader,
    Sinh,
    DarkCloudsShader,
    DarkClouds,
    MandelbulbShader,
    Mandelbulb,
    SmoothVoroShader,
    SmoothVoro,
    WinterflakeShader,
    Winterflake,
    GPUAttractorShader,
    GPUAttractor,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    app.new_window()
        .title("ICARUS Nannou Atlas")
        .size(WINDOW_W, WINDOW_H)
        .resizable(true)
        .key_pressed(key_pressed)
        .view(view)
        .build()
        .unwrap();

    let window = app.main_window();
    let swatch_texture = build_reference_texture(app);

    Model {
        hello_shader: build_shader_pass(
            &window,
            "primitive_atlas_hello",
            wgpu::include_wgsl!("shaders/atlas_hello.wgsl"),
            &HelloShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
            },
        ),
        chladni_shader: build_shader_pass(
            &window,
            "primitive_atlas_chladni",
            wgpu::include_wgsl!("shaders/atlas_chladni.wgsl"),
            &ChladniShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [4.0, 2.2, 0.6, 1.0],
            },
        ),
        neurons_shader: build_shader_pass(
            &window,
            "primitive_atlas_neurons",
            wgpu::include_wgsl!("shaders/atlas_neurons.wgsl"),
            &NeuronsShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [4.0, 1.78, 34.1, 2.1],
                params1: [1.0, 0.5, 0.0, 0.0],
            },
        ),
        darkclouds_shader: build_shader_pass(
            &window,
            "primitive_atlas_darkclouds",
            wgpu::include_wgsl!("shaders/atlas_darkclouds.wgsl"),
            &DarkCloudsShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [1.78, 2.0, 1.0, 1.0],
                params1: [0.0, 0.0, 0.0, 0.0],
            },
        ),
        gpuattractor_shader: build_shader_pass(
            &window,
            "primitive_atlas_gpuattractor",
            wgpu::include_wgsl!("shaders/atlas_gpu_attractor.wgsl"),
            &GPUAttractorShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [-1.8, -2.0, -0.5, -0.9],
                params1: [3.0, 2.0, 0.0, 0.0],
            },
        ),
        tree_shader: build_shader_pass(
            &window,
            "primitive_atlas_tree",
            wgpu::include_wgsl!("shaders/atlas_tree.wgsl"),
            &TreeShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [1.0, 0.6, 0.8, 1.0],
                params1: [2.0, 2.08, 0.0, 0.0],
            },
        ),
        sinh_shader: build_shader_pass(
            &window,
            "primitive_atlas_sinh",
            wgpu::include_wgsl!("shaders/atlas_sinh.wgsl"),
            &SinhShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [4.0, 0.00004, 0.02040101, 2.0],
                params1: [0.5, 0.1, 1.0, 66.0],
                params2: [67.0, 1.0, 0.0, 0.0],
                params3: [0.5, 0.25, 0.05, 0.0],
                params4: [3.0, 0.0, 0.5, 1.0],
            },
        ),
        mandelbulb_shader: build_shader_pass(
            &window,
            "primitive_atlas_mandelbulb",
            wgpu::include_wgsl!("shaders/atlas_mandelbulb.wgsl"),
            &MandelbulbShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [0.82, 0.41, 0.12, 1.0],
                params1: [0.65, 0.25, 0.95, 0.4],
                params2: [0.55, 0.7, 0.3, 0.5],
                params3: [0.7, 0.5, 8.0, 0.235],
                params4: [0.1, 4.0, 0.0, 0.0],
            },
        ),
        winterflake_shader: build_shader_pass(
            &window,
            "primitive_atlas_winterflake",
            wgpu::include_wgsl!("shaders/atlas_winterflake.wgsl"),
            &WinterflakeShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [180.0, 0.5, 0.7, 1.0],
                params1: [360.0, 0.04, 0.5, 8.0],
                params2: [0.235, 0.1, 64.0, 0.001],
                params3: [0.1, 3.0, 0.65, 0.25],
                params4: [0.95, 0.0, 0.0, 0.0],
            },
        ),
        fluid_shader: build_textured_shader_pass(
            &window,
            "primitive_atlas_fluid",
            wgpu::include_wgsl!("shaders/atlas_fluid.wgsl"),
            &FluidShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [0.3, 0.1, 0.01, 0.0],
                params1: [0.5, 1.0, 0.0, 0.0],
            },
            &swatch_texture,
            wgpu::AddressMode::Repeat,
        ),
        smoothvoro_shader: build_textured_shader_pass(
            &window,
            "primitive_atlas_smoothvoro",
            wgpu::include_wgsl!("shaders/atlas_smoothvoro.wgsl"),
            &SmoothVoroShaderUniforms {
                stage_rect: [0.0, 0.0, 1.0, 1.0],
                time_data: [0.0; 4],
                params0: [50.0, 0.5, 1.0, 0.1],
                params1: [2.0, 0.05, 0.0, 0.0],
            },
            &swatch_texture,
            wgpu::AddressMode::ClampToEdge,
        ),
        swatch_texture,
        screen_index: 0,
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn key_pressed(_app: &App, model: &mut Model, key: Key) {
    match key {
        Key::Left => {
            model.screen_index = (model.screen_index + SCREENS.len() - 1) % SCREENS.len();
        }
        Key::Right => {
            model.screen_index = (model.screen_index + 1) % SCREENS.len();
        }
        _ => {}
    }
}

fn ui_color([r, g, b, a]: [f32; 4]) -> LinSrgba {
    srgba(r, g, b, a).into_linear()
}

fn view(app: &App, model: &Model, frame: Frame) {
    let win = app.window_rect();
    let screen = SCREENS[model.screen_index];

    if matches!(
        screen,
        ScreenKind::HelloShader
            | ScreenKind::ChladniShader
            | ScreenKind::NeuronsShader
            | ScreenKind::FluidShader
            | ScreenKind::DarkCloudsShader
            | ScreenKind::GPUAttractorShader
            | ScreenKind::TreeShader
            | ScreenKind::SinhShader
            | ScreenKind::MandelbulbShader
            | ScreenKind::SmoothVoroShader
            | ScreenKind::WinterflakeShader
    ) {
        view_shader_screen(app, model, &frame, win, screen);
        return;
    }

    let draw = app.draw();
    draw.background().color(ui_color(GRID_BACKDROP));
    draw_backdrop(&draw, win);
    draw_header(&draw, win, screen);

    match screen {
        ScreenKind::Primitives => draw_primitives_screen(&draw, win, model),
        ScreenKind::Transforms => draw_transforms_screen(&draw, win, model, app.time as f32),
        ScreenKind::HelloShader
        | ScreenKind::ChladniShader
        | ScreenKind::NeuronsShader
        | ScreenKind::FluidShader
        | ScreenKind::DarkCloudsShader
        | ScreenKind::GPUAttractorShader
        | ScreenKind::TreeShader
        | ScreenKind::SinhShader
        | ScreenKind::MandelbulbShader
        | ScreenKind::SmoothVoroShader
        | ScreenKind::WinterflakeShader => {}
        ScreenKind::Chladni
        | ScreenKind::Neurons
        | ScreenKind::Fluid
        | ScreenKind::Tree
        | ScreenKind::Sinh
        | ScreenKind::DarkClouds
        | ScreenKind::Mandelbulb
        | ScreenKind::SmoothVoro
        | ScreenKind::Winterflake
        | ScreenKind::GPUAttractor => draw_visual_screen(&draw, win, screen, app.time as f32),
    }

    draw.to_frame(app, &frame).unwrap();
}

fn build_shader_pass<T: Copy>(
    window: &Window,
    label: &'static str,
    fragment_shader: wgpu::ShaderModuleDescriptor<'static>,
    initial_uniform: &T,
) -> ShaderPass {
    let device = window.device();
    let format = Frame::TEXTURE_FORMAT;
    let sample_count = window.msaa_samples();
    let vertex_module = device.create_shader_module(wgpu::include_wgsl!("shaders/atlas_quad.wgsl"));
    let fragment_module = device.create_shader_module(fragment_shader);
    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<T>() as _),
                },
                count: None,
            }],
        });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&uniform_bind_group_layout],
        push_constant_ranges: &[],
    });
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: unsafe { wgpu::bytes::from(initial_uniform) },
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: unsafe { wgpu::bytes::from_slice(&SHADER_VERTICES) },
        usage: wgpu::BufferUsages::VERTEX,
    });
    let pipeline = wgpu::RenderPipelineBuilder::from_layout(&pipeline_layout, &vertex_module)
        .fragment_shader(&fragment_module)
        .color_format(format)
        .add_vertex_buffer::<ShaderVertex>(&wgpu::vertex_attr_array![0 => Float32x2])
        .sample_count(sample_count)
        .build(device);

    ShaderPass {
        bind_group,
        pipeline,
        uniform_buffer,
        vertex_buffer,
    }
}

fn build_textured_shader_pass<T: Copy>(
    window: &Window,
    label: &'static str,
    fragment_shader: wgpu::ShaderModuleDescriptor<'static>,
    initial_uniform: &T,
    texture: &wgpu::Texture,
    address_mode: wgpu::AddressMode,
) -> TexturedShaderPass {
    let device = window.device();
    let format = Frame::TEXTURE_FORMAT;
    let sample_count = window.msaa_samples();
    let vertex_module = device.create_shader_module(wgpu::include_wgsl!("shaders/atlas_quad.wgsl"));
    let fragment_module = device.create_shader_module(fragment_shader);
    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<T>() as _),
                },
                count: None,
            }],
        });
    let texture_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .texture_from(wgpu::ShaderStages::FRAGMENT, texture)
        .sampler(wgpu::ShaderStages::FRAGMENT, true)
        .build(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
        push_constant_ranges: &[],
    });
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: unsafe { wgpu::bytes::from(initial_uniform) },
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        address_mode_w: address_mode,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let texture_view = texture.view().build();
    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: unsafe { wgpu::bytes::from_slice(&SHADER_VERTICES) },
        usage: wgpu::BufferUsages::VERTEX,
    });
    let pipeline = wgpu::RenderPipelineBuilder::from_layout(&pipeline_layout, &vertex_module)
        .fragment_shader(&fragment_module)
        .color_format(format)
        .add_vertex_buffer::<ShaderVertex>(&wgpu::vertex_attr_array![0 => Float32x2])
        .sample_count(sample_count)
        .build(device);

    TexturedShaderPass {
        texture_bind_group,
        uniform_bind_group,
        pipeline,
        uniform_buffer,
        vertex_buffer,
    }
}

fn view_shader_screen(app: &App, model: &Model, frame: &Frame, win: Rect, screen: ScreenKind) {
    let background = app.draw();
    background.background().color(ui_color(GRID_BACKDROP));
    draw_backdrop(&background, win);
    background.to_frame(app, frame).unwrap();

    let stage = visual_stage_rect(win);
    match screen {
        ScreenKind::HelloShader => render_hello_shader(app, model, frame, stage),
        ScreenKind::ChladniShader => render_chladni_shader(app, model, frame, stage),
        ScreenKind::NeuronsShader => render_neurons_shader(app, model, frame, stage),
        ScreenKind::FluidShader => render_fluid_shader(app, model, frame, stage),
        ScreenKind::DarkCloudsShader => render_darkclouds_shader(app, model, frame, stage),
        ScreenKind::GPUAttractorShader => render_gpuattractor_shader(app, model, frame, stage),
        ScreenKind::TreeShader => render_tree_shader(app, model, frame, stage),
        ScreenKind::SinhShader => render_sinh_shader(app, model, frame, stage),
        ScreenKind::MandelbulbShader => render_mandelbulb_shader(app, model, frame, stage),
        ScreenKind::SmoothVoroShader => render_smoothvoro_shader(app, model, frame, stage),
        ScreenKind::WinterflakeShader => render_winterflake_shader(app, model, frame, stage),
        _ => {}
    }

    let overlay = app.draw();
    draw_header(&overlay, win, screen);
    draw_shader_overlay(&overlay, stage, screen);
    overlay.to_frame(app, frame).unwrap();
}

fn render_hello_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = HelloShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
    };
    render_shader_pass(app, frame, stage, &model.hello_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_chladni_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = ChladniShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [4.0, 2.2, 0.6, 1.0],
    };
    render_shader_pass(app, frame, stage, &model.chladni_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_neurons_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = NeuronsShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [4.0, 1.78, 34.1, 2.1],
        params1: [1.0, 0.5, 0.0, 0.0],
    };
    render_shader_pass(app, frame, stage, &model.neurons_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_darkclouds_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = DarkCloudsShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [1.78, 2.0, 1.0, 1.0],
        params1: [0.0, 0.0, 0.0, 0.0],
    };
    render_shader_pass(app, frame, stage, &model.darkclouds_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_gpuattractor_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = GPUAttractorShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [-1.8, -2.0, -0.5, -0.9],
        params1: [3.0, 2.0, 0.0, 0.0],
    };
    render_shader_pass(app, frame, stage, &model.gpuattractor_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_tree_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = TreeShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [1.0, 0.6, 0.8, 1.0],
        params1: [2.0, 2.08, 0.0, 0.0],
    };
    render_shader_pass(app, frame, stage, &model.tree_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_sinh_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = SinhShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [4.0, 0.00004, 0.02040101, 2.0],
        params1: [0.5, 0.1, 1.0, 66.0],
        params2: [67.0, 1.0, 0.0, 0.0],
        params3: [0.5, 0.25, 0.05, 0.0],
        params4: [3.0, 0.0, 0.5, 1.0],
    };
    render_shader_pass(app, frame, stage, &model.sinh_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_mandelbulb_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = MandelbulbShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [0.82, 0.41, 0.12, 1.0],
        params1: [0.65, 0.25, 0.95, 0.4],
        params2: [0.55, 0.7, 0.3, 0.5],
        params3: [0.7, 0.5, 8.0, 0.235],
        params4: [0.1, 4.0, 0.0, 0.0],
    };
    render_shader_pass(app, frame, stage, &model.mandelbulb_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_winterflake_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = WinterflakeShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [180.0, 0.5, 0.7, 1.0],
        params1: [360.0, 0.04, 0.5, 8.0],
        params2: [0.235, 0.1, 64.0, 0.001],
        params3: [0.1, 3.0, 0.65, 0.25],
        params4: [0.95, 0.0, 0.0, 0.0],
    };
    render_shader_pass(app, frame, stage, &model.winterflake_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_fluid_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = FluidShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [0.3, 0.1, 0.01, 0.0],
        params1: [0.5, 1.0, 0.0, 0.0],
    };
    render_textured_shader_pass(app, frame, stage, &model.fluid_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_smoothvoro_shader(app: &App, model: &Model, frame: &Frame, stage: Rect) {
    let window = app.main_window();
    let pixel_stage = rect_to_pixel_rect(window.rect(), stage, window.inner_size_pixels());
    let uniforms = SmoothVoroShaderUniforms {
        stage_rect: [pixel_stage.x, pixel_stage.y, pixel_stage.w, pixel_stage.h],
        time_data: [app.time as f32, 0.0, 0.0, 0.0],
        params0: [50.0, 0.5, 1.0, 0.1],
        params1: [2.0, 0.05, 0.0, 0.0],
    };
    render_textured_shader_pass(app, frame, stage, &model.smoothvoro_shader, unsafe {
        wgpu::bytes::from(&uniforms)
    });
}

fn render_shader_pass(
    app: &App,
    frame: &Frame,
    stage: Rect,
    shader_pass: &ShaderPass,
    uniform_bytes: &[u8],
) {
    let window = app.main_window();
    let (sx, sy, sw, sh) = rect_to_scissor(window.rect(), stage, window.inner_size_pixels());
    frame
        .device_queue_pair()
        .queue()
        .write_buffer(&shader_pass.uniform_buffer, 0, uniform_bytes);

    let mut encoder = frame.command_encoder();
    let mut render_pass = wgpu::RenderPassBuilder::new()
        .color_attachment(frame.texture_view(), |color| {
            color.load_op(wgpu::LoadOp::Load)
        })
        .begin(&mut encoder);
    render_pass.set_scissor_rect(sx, sy, sw, sh);
    render_pass.set_pipeline(&shader_pass.pipeline);
    render_pass.set_bind_group(0, &shader_pass.bind_group, &[]);
    render_pass.set_vertex_buffer(0, shader_pass.vertex_buffer.slice(..));
    render_pass.draw(0..SHADER_VERTICES.len() as u32, 0..1);
}

fn render_textured_shader_pass(
    app: &App,
    frame: &Frame,
    stage: Rect,
    shader_pass: &TexturedShaderPass,
    uniform_bytes: &[u8],
) {
    let window = app.main_window();
    let (sx, sy, sw, sh) = rect_to_scissor(window.rect(), stage, window.inner_size_pixels());
    frame
        .device_queue_pair()
        .queue()
        .write_buffer(&shader_pass.uniform_buffer, 0, uniform_bytes);

    let mut encoder = frame.command_encoder();
    let mut render_pass = wgpu::RenderPassBuilder::new()
        .color_attachment(frame.texture_view(), |color| {
            color.load_op(wgpu::LoadOp::Load)
        })
        .begin(&mut encoder);
    render_pass.set_scissor_rect(sx, sy, sw, sh);
    render_pass.set_pipeline(&shader_pass.pipeline);
    render_pass.set_bind_group(0, &shader_pass.uniform_bind_group, &[]);
    render_pass.set_bind_group(1, &shader_pass.texture_bind_group, &[]);
    render_pass.set_vertex_buffer(0, shader_pass.vertex_buffer.slice(..));
    render_pass.draw(0..SHADER_VERTICES.len() as u32, 0..1);
}

fn rect_to_pixel_rect(win: Rect, rect: Rect, inner_size_pixels: (u32, u32)) -> PixelRect {
    let scale_x = inner_size_pixels.0 as f32 / win.w().max(1.0);
    let scale_y = inner_size_pixels.1 as f32 / win.h().max(1.0);
    PixelRect {
        x: (rect.left() - win.left()) * scale_x,
        y: (win.top() - rect.top()) * scale_y,
        w: rect.w() * scale_x,
        h: rect.h() * scale_y,
    }
}

fn rect_to_scissor(win: Rect, rect: Rect, inner_size_pixels: (u32, u32)) -> (u32, u32, u32, u32) {
    let pixel = rect_to_pixel_rect(win, rect, inner_size_pixels);
    let max_w = inner_size_pixels.0 as f32;
    let max_h = inner_size_pixels.1 as f32;
    let x = pixel.x.floor().clamp(0.0, max_w) as u32;
    let y = pixel.y.floor().clamp(0.0, max_h) as u32;
    let w = pixel.w.ceil().clamp(1.0, max_w - x as f32) as u32;
    let h = pixel.h.ceil().clamp(1.0, max_h - y as f32) as u32;
    (x, y, w, h)
}

fn draw_backdrop(draw: &Draw, win: Rect) {
    draw.rect()
        .w_h(win.w() * 0.92, win.h() * 0.84)
        .rotate(0.12)
        .color(srgba(0.19, 0.23, 0.35, 0.08));

    draw.ellipse()
        .x_y(win.left() + 180.0, win.top() - 120.0)
        .w_h(320.0, 220.0)
        .color(hsla(0.58, 0.42, 0.55, 0.09));

    draw.ellipse()
        .x_y(win.right() - 180.0, win.bottom() + 140.0)
        .w_h(360.0, 260.0)
        .color(hsla(0.08, 0.48, 0.56, 0.08));

    let lines = 16;
    let step = win.w() / lines as f32;
    for i in 0..=lines {
        let x = win.left() + step * i as f32;
        draw.line()
            .points(pt2(x, win.bottom()), pt2(x, win.top()))
            .weight(1.0)
            .color(srgba(1.0, 1.0, 1.0, 0.025));
    }
}

fn draw_header(draw: &Draw, win: Rect, screen: ScreenKind) {
    let left = win.left() + OUTER_MARGIN;
    let top = win.top() - 24.0;

    draw.text(screen.title())
        .x_y(left + 280.0, top - 8.0)
        .w_h(560.0, 48.0)
        .left_justify()
        .align_text_top()
        .font_size(28)
        .color(ui_color(LABEL));

    draw.text(screen.subtitle())
        .x_y(left + 400.0, top - 44.0)
        .w_h(800.0, 32.0)
        .left_justify()
        .align_text_top()
        .font_size(14)
        .color(ui_color(MUTED));

    draw.text(screen.caption())
        .x_y(left + 405.0, top - 74.0)
        .w_h(810.0, 28.0)
        .left_justify()
        .align_text_top()
        .font_size(13)
        .color(ui_color(CODE));

    draw.text("Use Left/Right Arrow Keys")
        .x_y(win.right() - 188.0, top - 18.0)
        .w_h(330.0, 30.0)
        .right_justify()
        .align_text_top()
        .font_size(14)
        .color(ui_color(LABEL));

    draw.text("Run: cargo run --bin primitive_atlas")
        .x_y(win.right() - 188.0, top - 44.0)
        .w_h(330.0, 24.0)
        .right_justify()
        .align_text_top()
        .font_size(12)
        .color(ui_color(MUTED));

    let page_idx = SCREENS
        .iter()
        .position(|candidate| *candidate == screen)
        .unwrap_or(0);
    let current_x = win.right() - OUTER_MARGIN - 124.0;
    let current_y = top - 80.0;

    draw.rect()
        .x_y(current_x, current_y)
        .w_h(248.0, 28.0)
        .color(srgba(0.18, 0.23, 0.31, 0.78))
        .stroke(srgba(0.92, 0.96, 1.0, 0.18))
        .stroke_weight(1.0);

    draw.text(screen.tab_label())
        .x_y(current_x, current_y - 1.0)
        .w_h(180.0, 26.0)
        .center_justify()
        .align_text_middle_y()
        .font_size(12)
        .color(ui_color(LABEL));

    draw.text(&format!("{}/{}", page_idx + 1, SCREENS.len()))
        .x_y(current_x + 87.0, current_y - 1.0)
        .w_h(64.0, 24.0)
        .right_justify()
        .align_text_middle_y()
        .font_size(11)
        .color(ui_color(MUTED));

    let prev = SCREENS[(page_idx + SCREENS.len() - 1) % SCREENS.len()];
    let next = SCREENS[(page_idx + 1) % SCREENS.len()];
    draw.text(&format!("Prev: {}", prev.tab_label()))
        .x_y(current_x - 206.0, current_y - 1.0)
        .w_h(180.0, 24.0)
        .right_justify()
        .align_text_middle_y()
        .font_size(11)
        .color(ui_color(MUTED));
    draw.text(&format!("Next: {}", next.tab_label()))
        .x_y(win.right() - OUTER_MARGIN - 72.0, current_y - 1.0)
        .w_h(180.0, 24.0)
        .left_justify()
        .align_text_middle_y()
        .font_size(11)
        .color(ui_color(MUTED));
}

impl ScreenKind {
    fn title(self) -> &'static str {
        match self {
            Self::Primitives => "ICARUS Primitive Atlas",
            Self::Transforms => "ICARUS Transform Atlas",
            Self::HelloShader => "ICARUS Shader Hello",
            Self::ChladniShader => "ICARUS Shader Atlas",
            Self::NeuronsShader => "ICARUS Shader Atlas",
            Self::FluidShader => "ICARUS Shader Atlas",
            Self::DarkCloudsShader => "ICARUS Shader Atlas",
            Self::GPUAttractorShader => "ICARUS Shader Atlas",
            Self::TreeShader => "ICARUS Shader Atlas",
            Self::SinhShader => "ICARUS Shader Atlas",
            Self::MandelbulbShader => "ICARUS Shader Atlas",
            Self::SmoothVoroShader => "ICARUS Shader Atlas",
            Self::WinterflakeShader => "ICARUS Shader Atlas",
            Self::Chladni => "ICARUS Visual Atlas",
            Self::Neurons => "ICARUS Visual Atlas",
            Self::Fluid => "ICARUS Visual Atlas",
            Self::Tree => "ICARUS Visual Atlas",
            Self::Sinh => "ICARUS Visual Atlas",
            Self::DarkClouds => "ICARUS Visual Atlas",
            Self::Mandelbulb => "ICARUS Visual Atlas",
            Self::SmoothVoro => "ICARUS Visual Atlas",
            Self::Winterflake => "ICARUS Visual Atlas",
            Self::GPUAttractor => "ICARUS Visual Atlas",
        }
    }

    fn subtitle(self) -> &'static str {
        match self {
            Self::Primitives => {
                "A small Nannou draw reference: one tile per primitive, short builder cues, no audio state."
            }
            Self::Transforms => {
                "Animated 2D and 3D transform lessons: translation, local rotation, axis scaling, and depth."
            }
            Self::HelloShader => {
                "A minimal WGSL proof screen: one fullscreen quad, one fragment shader, clipped into the atlas stage."
            }
            Self::ChladniShader => {
                "The first actual shader port: a WGSL Chladni plate adapted into the atlas stage while keeping the CPU interpretation too."
            }
            Self::NeuronsShader => {
                "A shader-native neuron field companion: the original WGSL-style flow and pulse field rendered directly in the atlas stage."
            }
            Self::FluidShader => {
                "A shader-native fluid companion: the original texture-warp flow is rendered directly in WGSL inside the atlas stage."
            }
            Self::DarkCloudsShader => {
                "A shader-native cloud field companion: layered smoky motion rendered with the original fragment-style approach."
            }
            Self::GPUAttractorShader => {
                "A shader-native attractor companion: the orbit glow and point field are rendered directly in WGSL."
            }
            Self::TreeShader => {
                "A shader-native branching companion: the recursive tree field is rendered with the original fragment-style iteration."
            }
            Self::SinhShader => {
                "A shader-native hyperbolic field companion: the original iterative color surface is rendered directly in WGSL."
            }
            Self::MandelbulbShader => {
                "A shader-native raymarched fractal companion: the bulb form is rendered directly in WGSL inside the atlas stage."
            }
            Self::SmoothVoroShader => {
                "A shader-native smooth Voronoi companion: the moving cells and texture lookup are rendered directly in WGSL."
            }
            Self::WinterflakeShader => {
                "A shader-native snow crystal companion: the radial SDF flake is rendered directly in WGSL."
            }
            Self::Chladni => {
                "Chladni-like standing-wave nodes inspired by `chladniwgpu.rs` from altunenes/rusty_art."
            }
            Self::Neurons => {
                "A neuron-network field inspired by `neurons.rs`, translated into moving nodes and pulsing links."
            }
            Self::Fluid => {
                "A fluid-flow interpretation inspired by `fluid.rs`, rendered as animated streamlines through a procedural vector field."
            }
            Self::Tree => {
                "A recursive branch study inspired by `tree.rs`, with wind-driven angle and length modulation."
            }
            Self::Sinh => {
                "A hyperbolic-wave mesh inspired by `sinh.rs`, using sinh-shaped contours and oscillating interference bands."
            }
            Self::DarkClouds => {
                "A smoky density field inspired by `darkclouds.rs`, built from layered procedural cloud bands."
            }
            Self::Mandelbulb => {
                "A rotating pseudo-3D fractal bloom inspired by `mandelbulb.rs`, simplified into an orthographic point cloud."
            }
            Self::SmoothVoro => {
                "A smooth Voronoi field inspired by `smoothvoro.rs`, rebuilt here as an animated CPU cell-distance texture."
            }
            Self::Winterflake => {
                "A six-fold snow crystal inspired by `winterflake.rs`, mixing recursive arms with drifting particles."
            }
            Self::GPUAttractor => {
                "A strange-attractor study inspired by `GPUattractor.rs`, using a De Jong style orbit cloud."
            }
        }
    }

    fn caption(self) -> &'static str {
        match self {
            Self::Primitives => {
                "Shared modifiers: xy/x_y, w_h, color/rgb/rgba/hsl/hsla, stroke vs fill, rotate, scale"
            }
            Self::Transforms => {
                "Important Nannou 0.19 note: Draw uses an orthographic projection, so z changes placement/orientation rather than perspective size."
            }
            Self::HelloShader => {
                "This is the small first step before the larger shader ports: GPU color is drawn directly with WGSL, then the normal atlas chrome is composited over it."
            }
            Self::ChladniShader => {
                "Reference: altunenes/rusty_art/src/chladniwgpu.rs. This one is a real fragment shader running on a fullscreen quad clipped to the panel."
            }
            Self::NeuronsShader => {
                "Reference: altunenes/rusty_art/src/neurons.rs. This is the shader-based companion to the procedural neuron scene."
            }
            Self::FluidShader => {
                "Reference: altunenes/rusty_art/src/fluid.rs. This is the shader-based companion to the procedural fluid scene."
            }
            Self::DarkCloudsShader => {
                "Reference: altunenes/rusty_art/src/darkclouds.rs. This is the shader-based companion to the procedural cloud scene."
            }
            Self::GPUAttractorShader => {
                "Reference: altunenes/rusty_art/src/GPUattractor.rs. This is the shader-based companion to the procedural attractor scene."
            }
            Self::TreeShader => {
                "Reference: altunenes/rusty_art/src/tree.rs. This is the shader-based companion to the procedural tree scene."
            }
            Self::SinhShader => {
                "Reference: altunenes/rusty_art/src/sinh.rs. This is the shader-based companion to the procedural sinh scene."
            }
            Self::MandelbulbShader => {
                "Reference: altunenes/rusty_art/src/mandelbulb.rs. This is the shader-based companion to the procedural mandelbulb scene."
            }
            Self::SmoothVoroShader => {
                "Reference: altunenes/rusty_art/src/smoothvoro.rs. This is the shader-based companion to the procedural smooth Voronoi scene."
            }
            Self::WinterflakeShader => {
                "Reference: altunenes/rusty_art/src/winterflake.rs. This is the shader-based companion to the procedural winterflake scene."
            }
            Self::Chladni => {
                "Reference: altunenes/rusty_art/src/chladniwgpu.rs. This screen uses a point-field interpretation instead of the original WGSL full-screen shader."
            }
            Self::Neurons => {
                "Reference: altunenes/rusty_art/src/neurons.rs. This screen emphasizes graph structure, thresholds, and pulse flow rather than shader glow."
            }
            Self::Fluid => {
                "Reference: altunenes/rusty_art/src/fluid.rs. This version shows the underlying flow vectors as advected lines to make the motion readable."
            }
            Self::Tree => {
                "Reference: altunenes/rusty_art/src/tree.rs. Recursive geometry is exposed directly so the branching logic stays legible."
            }
            Self::Sinh => {
                "Reference: altunenes/rusty_art/src/sinh.rs. Hyperbolic curves and line density replace the original shader surface."
            }
            Self::DarkClouds => {
                "Reference: altunenes/rusty_art/src/darkclouds.rs. Layered bands and coarse volumetric blocks keep the cloud motion inspectable."
            }
            Self::Mandelbulb => {
                "Reference: altunenes/rusty_art/src/mandelbulb.rs. Orthographic Draw cannot mimic the original ray-marched look exactly, so this uses a rotating bulb-like point shell."
            }
            Self::SmoothVoro => {
                "Reference: altunenes/rusty_art/src/smoothvoro.rs. The field is sampled on a coarse CPU grid so cell blending and boundaries remain visible."
            }
            Self::Winterflake => {
                "Reference: altunenes/rusty_art/src/winterflake.rs. Radial symmetry and recursive branching carry the snowflake idea without the original fragment shader."
            }
            Self::GPUAttractor => {
                "Reference: altunenes/rusty_art/src/GPUattractor.rs. This version exposes the orbit itself as points rather than a post-processed shader pass."
            }
        }
    }

    fn tab_label(self) -> &'static str {
        match self {
            Self::Primitives => "Primitives",
            Self::Transforms => "Transforms",
            Self::HelloShader => "Hello Shader",
            Self::ChladniShader => "Chladni Shader",
            Self::NeuronsShader => "Neurons Shader",
            Self::FluidShader => "Fluid Shader",
            Self::DarkCloudsShader => "Dark Clouds Shader",
            Self::GPUAttractorShader => "GPU Attractor Shader",
            Self::TreeShader => "Tree Shader",
            Self::SinhShader => "Sinh Shader",
            Self::MandelbulbShader => "Mandelbulb Shader",
            Self::SmoothVoroShader => "Smooth Voro Shader",
            Self::WinterflakeShader => "Winterflake Shader",
            Self::Chladni => "Chladni",
            Self::Neurons => "Neurons",
            Self::Fluid => "Fluid",
            Self::Tree => "Tree",
            Self::Sinh => "Sinh",
            Self::DarkClouds => "Dark Clouds",
            Self::Mandelbulb => "Mandelbulb",
            Self::SmoothVoro => "Smooth Voro",
            Self::Winterflake => "Winterflake",
            Self::GPUAttractor => "GPU Attractor",
        }
    }

    fn source_file(self) -> Option<&'static str> {
        match self {
            Self::Primitives | Self::Transforms | Self::HelloShader => None,
            Self::ChladniShader => Some("chladniwgpu.rs"),
            Self::Chladni => Some("chladniwgpu.rs"),
            Self::NeuronsShader => Some("neurons.rs"),
            Self::Neurons => Some("neurons.rs"),
            Self::FluidShader => Some("fluid.rs"),
            Self::Fluid => Some("fluid.rs"),
            Self::Tree => Some("tree.rs"),
            Self::Sinh => Some("sinh.rs"),
            Self::DarkCloudsShader => Some("darkclouds.rs"),
            Self::DarkClouds => Some("darkclouds.rs"),
            Self::MandelbulbShader => Some("mandelbulb.rs"),
            Self::Mandelbulb => Some("mandelbulb.rs"),
            Self::SmoothVoroShader => Some("smoothvoro.rs"),
            Self::SmoothVoro => Some("smoothvoro.rs"),
            Self::Winterflake => Some("winterflake.rs"),
            Self::GPUAttractorShader => Some("GPUattractor.rs"),
            Self::GPUAttractor => Some("GPUattractor.rs"),
            Self::TreeShader => Some("tree.rs"),
            Self::SinhShader => Some("sinh.rs"),
            Self::WinterflakeShader => Some("winterflake.rs"),
        }
    }
}

fn screen_content_rect(win: Rect) -> Rect {
    let left = win.left() + OUTER_MARGIN;
    let right = win.right() - OUTER_MARGIN;
    let top = win.top() - HEADER_H;
    let bottom = win.bottom() + OUTER_MARGIN;
    Rect::from_corners(pt2(left, bottom), pt2(right, top))
}

fn visual_stage_rect(win: Rect) -> Rect {
    let frame = screen_content_rect(win);
    Rect::from_corners(
        pt2(frame.left() + 8.0, frame.bottom() + 8.0),
        pt2(frame.right() - 8.0, frame.top() - 8.0),
    )
}

fn draw_primitives_screen(draw: &Draw, win: Rect, model: &Model) {
    for (layout, primitive) in atlas_layout(win).into_iter().zip(PRIMITIVES) {
        draw_tile(draw, layout, primitive, model);
    }
}

fn draw_transforms_screen(draw: &Draw, win: Rect, model: &Model, time: f32) {
    let content = screen_content_rect(win);
    let top_h = content.h() * 0.56;
    let bottom_h = content.h() - top_h - GUTTER;
    let left_w = content.w() * 0.56;
    let right_w = content.w() - left_w - GUTTER;

    let two_d_panel = Rect::from_xy_wh(
        pt2(content.left() + left_w * 0.5, content.top() - top_h * 0.5),
        vec2(left_w, top_h),
    );
    let three_d_panel = Rect::from_xy_wh(
        pt2(content.right() - right_w * 0.5, content.top() - top_h * 0.5),
        vec2(right_w, top_h),
    );
    let depth_panel = Rect::from_xy_wh(
        pt2(content.x(), content.bottom() + bottom_h * 0.5),
        vec2(content.w(), bottom_h),
    );

    let two_d_sample = draw_lesson_panel(
        draw,
        two_d_panel,
        "2D Transform Stack",
        "Code order reads as: draw.x_y(tx, ty).rotate(theta).scale_axes(vec3(sx, sy, 1.0)).rect().\nThe shape is scaled in its local frame, then that local frame rotates, then the result is translated in parent space.",
    );
    draw_2d_transform_demo(draw, two_d_sample, time);

    let three_d_sample = draw_lesson_panel(
        draw,
        three_d_panel,
        "3D Orientation",
        "Use x_y_z(..) to place the model in 3D draw space. pitch/yaw/roll rotate the local axes.\nscale_axes(vec3(..)) stretches each axis independently. Red = x, green = y, blue = z.",
    );
    draw_3d_transform_demo(draw, three_d_sample, time, model);

    let depth_sample = draw_lesson_panel(
        draw,
        depth_panel,
        "Depth In Orthographic Draw",
        "Nannou Draw defaults to an orthographic projection. These cubes share the same mesh size.\nChanging z moves them through the depth range, but it does not make farther cubes appear smaller the way a perspective camera would.",
    );
    draw_orthographic_depth_demo(draw, depth_sample, time);
}

fn draw_lesson_panel(draw: &Draw, frame: Rect, title: &str, detail: &str) -> Rect {
    draw.rect()
        .x_y(frame.x(), frame.y())
        .w_h(frame.w(), frame.h())
        .color(ui_color(PANEL_FILL))
        .stroke(ui_color(CARD_STROKE))
        .stroke_weight(1.5);

    draw.line()
        .points(
            pt2(frame.left() + 20.0, frame.top() - 42.0),
            pt2(frame.right() - 20.0, frame.top() - 42.0),
        )
        .weight(1.0)
        .color(srgba(1.0, 1.0, 1.0, 0.08));

    draw.text(title)
        .x_y(frame.left() + 200.0, frame.top() - 24.0)
        .w_h(frame.w() - 40.0, 28.0)
        .left_justify()
        .align_text_top()
        .font_size(18)
        .color(ui_color(LABEL));

    draw.text(detail)
        .x_y(frame.left() + frame.w() * 0.5, frame.bottom() + 34.0)
        .w_h(frame.w() - 42.0, 54.0)
        .left_justify()
        .align_text_top()
        .line_spacing(4.0)
        .font_size(12)
        .color(ui_color(CODE));

    let sample = Rect::from_corners(
        pt2(frame.left() + 24.0, frame.bottom() + 84.0),
        pt2(frame.right() - 24.0, frame.top() - 56.0),
    );

    draw.rect()
        .x_y(sample.x(), sample.y())
        .w_h(sample.w(), sample.h())
        .color(ui_color(PANEL_SAMPLE_FILL))
        .stroke(ui_color(PANEL_SAMPLE_STROKE))
        .stroke_weight(1.0);

    sample
}

fn draw_visual_screen(draw: &Draw, win: Rect, screen: ScreenKind, time: f32) {
    let stage = visual_stage_rect(win);
    let stage_draw = draw.x_y(stage.x(), stage.y());
    let local = Rect::from_w_h(stage.w(), stage.h());

    draw.rect()
        .x_y(stage.x(), stage.y())
        .w_h(stage.w(), stage.h())
        .color(srgba(0.07, 0.085, 0.12, 0.96))
        .stroke(ui_color(CARD_STROKE))
        .stroke_weight(1.4);

    match screen {
        ScreenKind::Chladni => draw_chladni_visual(&stage_draw, local, time),
        ScreenKind::Neurons => draw_neurons_visual(&stage_draw, local, time),
        ScreenKind::Fluid => draw_fluid_visual(&stage_draw, local, time),
        ScreenKind::Tree => draw_tree_visual(&stage_draw, local, time),
        ScreenKind::Sinh => draw_sinh_visual(&stage_draw, local, time),
        ScreenKind::DarkClouds => draw_darkclouds_visual(&stage_draw, local, time),
        ScreenKind::Mandelbulb => draw_mandelbulb_visual(&stage_draw, local, time),
        ScreenKind::SmoothVoro => draw_smoothvoro_visual(&stage_draw, local, time),
        ScreenKind::Winterflake => draw_winterflake_visual(&stage_draw, local, time),
        ScreenKind::GPUAttractor => draw_gpuattractor_visual(&stage_draw, local, time),
        ScreenKind::Primitives
        | ScreenKind::Transforms
        | ScreenKind::HelloShader
        | ScreenKind::ChladniShader
        | ScreenKind::NeuronsShader
        | ScreenKind::FluidShader
        | ScreenKind::DarkCloudsShader
        | ScreenKind::GPUAttractorShader
        | ScreenKind::TreeShader
        | ScreenKind::SinhShader
        | ScreenKind::MandelbulbShader
        | ScreenKind::SmoothVoroShader
        | ScreenKind::WinterflakeShader => {}
    }

    draw.rect()
        .x_y(stage.left() + 212.0, stage.bottom() + 34.0)
        .w_h(408.0, 48.0)
        .color(srgba(0.05, 0.07, 0.10, 0.72))
        .stroke(srgba(1.0, 1.0, 1.0, 0.08))
        .stroke_weight(1.0);

    draw.text(screen.tab_label())
        .x_y(stage.left() + 108.0, stage.bottom() + 39.0)
        .w_h(180.0, 18.0)
        .left_justify()
        .align_text_top()
        .font_size(16)
        .color(ui_color(LABEL));

    if let Some(source) = screen.source_file() {
        draw.text(&format!("inspired by altunenes/rusty_art/src/{source}"))
            .x_y(stage.left() + 202.0, stage.bottom() + 20.0)
            .w_h(360.0, 16.0)
            .left_justify()
            .align_text_top()
            .font_size(11)
            .color(ui_color(MUTED));
    }
}

fn draw_shader_overlay(draw: &Draw, stage: Rect, screen: ScreenKind) {
    draw.rect()
        .x_y(stage.x(), stage.y())
        .w_h(stage.w(), stage.h())
        .color(srgba(0.04, 0.05, 0.08, 0.04))
        .stroke(ui_color(CARD_STROKE))
        .stroke_weight(1.4);

    draw.rect()
        .x_y(stage.left() + 214.0, stage.bottom() + 34.0)
        .w_h(412.0, 48.0)
        .color(srgba(0.05, 0.07, 0.10, 0.74))
        .stroke(srgba(1.0, 1.0, 1.0, 0.08))
        .stroke_weight(1.0);

    draw.text(screen.tab_label())
        .x_y(stage.left() + 110.0, stage.bottom() + 39.0)
        .w_h(190.0, 18.0)
        .left_justify()
        .align_text_top()
        .font_size(16)
        .color(ui_color(LABEL));

    let detail = match screen {
        ScreenKind::HelloShader => {
            "fullscreen quad + WGSL fragment pass clipped to the atlas stage"
        }
        ScreenKind::ChladniShader => {
            "real WGSL Chladni pass adapted from altunenes/rusty_art/src/chladniwgpu.rs"
        }
        ScreenKind::NeuronsShader => {
            "real WGSL neuron field adapted from altunenes/rusty_art/src/neurons.rs"
        }
        ScreenKind::FluidShader => {
            "real WGSL fluid field adapted from altunenes/rusty_art/src/fluid.rs"
        }
        ScreenKind::DarkCloudsShader => {
            "real WGSL cloud field adapted from altunenes/rusty_art/src/darkclouds.rs"
        }
        ScreenKind::GPUAttractorShader => {
            "real WGSL attractor field adapted from altunenes/rusty_art/src/GPUattractor.rs"
        }
        ScreenKind::TreeShader => {
            "real WGSL tree field adapted from altunenes/rusty_art/src/tree.rs"
        }
        ScreenKind::SinhShader => {
            "real WGSL sinh field adapted from altunenes/rusty_art/src/sinh.rs"
        }
        ScreenKind::MandelbulbShader => {
            "real WGSL mandelbulb field adapted from altunenes/rusty_art/src/mandelbulb.rs"
        }
        ScreenKind::SmoothVoroShader => {
            "real WGSL smooth Voronoi field adapted from altunenes/rusty_art/src/smoothvoro.rs"
        }
        ScreenKind::WinterflakeShader => {
            "real WGSL winterflake field adapted from altunenes/rusty_art/src/winterflake.rs"
        }
        _ => "",
    };

    if !detail.is_empty() {
        draw.text(detail)
            .x_y(stage.left() + 205.0, stage.bottom() + 20.0)
            .w_h(376.0, 16.0)
            .left_justify()
            .align_text_top()
            .font_size(11)
            .color(ui_color(MUTED));
    }
}

fn draw_chladni_visual(draw: &Draw, rect: Rect, time: f32) {
    draw_reference_grid(draw, rect, 10, 0.018);
    let a = 2.0 + 1.3 * (time * 0.18).sin();
    let b = 3.0 + 1.7 * (time * 0.22 + 1.2).sin();
    let c = 4.0 + 1.2 * (time * 0.16 + 2.4).cos();
    let d = 5.0 + 1.4 * (time * 0.20 + 0.5).sin();
    let threshold = 0.08;

    for gx in 0..120 {
        for gy in 0..72 {
            let px = map_range(gx, 0, 119, rect.left(), rect.right());
            let py = map_range(gy, 0, 71, rect.bottom(), rect.top());
            let x = map_range(px, rect.left(), rect.right(), -PI, PI);
            let y = map_range(py, rect.bottom(), rect.top(), -PI, PI);
            let field =
                ((a * x).sin() * (b * y).sin() - (c * x + time * 0.3).cos() * (d * y).sin()).abs();
            if field < threshold {
                let alpha = map_range(field, 0.0, threshold, 0.98, 0.08);
                draw.ellipse().x_y(px, py).w_h(3.0, 3.0).color(hsla(
                    0.15 + 0.45 * (x * y).sin().abs(),
                    0.72,
                    0.68,
                    alpha,
                ));
            }
        }
    }
}

fn draw_neurons_visual(draw: &Draw, rect: Rect, time: f32) {
    let mut nodes = Vec::with_capacity(28);
    for i in 0..28 {
        let fi = i as f32;
        let angle = fi * 0.63 + time * 0.12;
        let radial = 0.24 + 0.56 * ((fi * 1.37 + time * 0.18).sin() * 0.5 + 0.5);
        let x = angle.cos() * rect.w() * radial * 0.46 + (fi * 0.91 + time * 0.38).sin() * 48.0;
        let y = angle.sin() * rect.h() * radial * 0.34 + (fi * 1.27 + time * 0.26).cos() * 34.0;
        nodes.push(pt2(x, y));
    }

    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let a = nodes[i];
            let b = nodes[j];
            let dist = a.distance(b);
            if dist < 180.0 {
                let link = 1.0 - dist / 180.0;
                draw.line()
                    .points(a, b)
                    .weight(1.0 + link * 2.2)
                    .color(hsla(
                        0.56 + 0.08 * (i as f32).sin(),
                        0.70,
                        0.62,
                        0.08 + 0.22 * link,
                    ));

                let pulse = ((time * 0.7 + i as f32 * 0.17 + j as f32 * 0.11).sin() * 0.5 + 0.5)
                    .clamp(0.0, 1.0);
                let pos = a.lerp(b, pulse);
                draw.ellipse()
                    .x_y(pos.x, pos.y)
                    .w_h(4.0 + link * 4.0, 4.0 + link * 4.0)
                    .color(hsla(0.12, 0.82, 0.68, 0.16 + 0.52 * link));
            }
        }
    }

    for (idx, node) in nodes.into_iter().enumerate() {
        let glow = 8.0 + 4.0 * (time * 1.6 + idx as f32).sin().abs();
        draw.ellipse()
            .x_y(node.x, node.y)
            .w_h(glow * 2.0, glow * 2.0)
            .color(hsla(0.58, 0.56, 0.54, 0.08));
        draw.ellipse().x_y(node.x, node.y).w_h(8.0, 8.0).color(hsla(
            0.14 + 0.42 * hash01(idx as f32),
            0.82,
            0.72,
            0.94,
        ));
    }
}

fn draw_fluid_visual(draw: &Draw, rect: Rect, time: f32) {
    draw_reference_grid(draw, rect, 12, 0.012);
    for gx in 0..22 {
        for gy in 0..10 {
            let seed_x = map_range(gx, 0, 21, rect.left() + 28.0, rect.right() - 28.0);
            let seed_y = map_range(gy, 0, 9, rect.bottom() + 20.0, rect.top() - 20.0);
            let mut p = pt2(seed_x, seed_y);
            for step in 0..24 {
                let nx = p.x / rect.w();
                let ny = p.y / rect.h();
                let angle = (nx * 9.0 + time * 0.4).sin() * 1.4
                    + (ny * 11.0 - time * 0.3).cos() * 1.2
                    + layered_wave(nx * 1.8, ny * 1.6, time * 0.18) * 2.2;
                let velocity = vec2(angle.cos(), angle.sin()) * 8.0;
                let next = p + velocity;
                let hue = 0.52 + 0.10 * (step as f32 / 24.0) + 0.05 * (gx as f32 * 0.2).sin();
                draw.line()
                    .points(p, next)
                    .weight(1.0 + step as f32 * 0.06)
                    .color(hsla(hue, 0.72, 0.58, 0.08 + step as f32 * 0.012));
                p = next;
            }
        }
    }
}

fn draw_tree_visual(draw: &Draw, rect: Rect, time: f32) {
    draw.rect()
        .x_y(0.0, rect.bottom() * 0.65)
        .w_h(rect.w(), rect.h() * 0.36)
        .color(hsla(0.34, 0.18, 0.14, 0.30));

    let root = pt2(0.0, rect.bottom() + 36.0);
    let trunk_len = rect.h() * 0.24;
    let tip = root + vec2(0.0, trunk_len);
    draw_branch(draw, root, tip, 9, time, 0.74);
}

fn draw_sinh_visual(draw: &Draw, rect: Rect, time: f32) {
    draw_reference_grid(draw, rect, 8, 0.012);

    for band in 0..22 {
        let offset = map_range(band, 0, 21, rect.bottom() * 0.64, rect.top() * 0.64);
        let mut prev = None;
        for step in 0..180 {
            let x = map_range(step, 0, 179, -2.2, 2.2);
            let y = ((x * (1.2 + band as f32 * 0.03) + time * 0.16).sinh().sin())
                / (1.0 + x.abs() * 1.8);
            let px = map_range(x, -2.2, 2.2, rect.left(), rect.right());
            let py = offset + y * rect.h() * 0.18;
            let point = pt2(px, py);
            if let Some(last) = prev {
                draw.line().points(last, point).weight(1.2).color(hsla(
                    0.58 + 0.12 * (band as f32 * 0.17).sin(),
                    0.62,
                    0.58,
                    0.12,
                ));
            }
            prev = Some(point);
        }
    }

    for band in 0..12 {
        let xoff = map_range(band, 0, 11, rect.left() * 0.76, rect.right() * 0.76);
        let mut prev = None;
        for step in 0..120 {
            let y = map_range(step, 0, 119, -1.8, 1.8);
            let x = ((y * (1.5 + band as f32 * 0.06) - time * 0.22).sinh().cos())
                / (1.0 + y.abs() * 2.3);
            let px = xoff + x * rect.w() * 0.14;
            let py = map_range(y, -1.8, 1.8, rect.bottom(), rect.top());
            let point = pt2(px, py);
            if let Some(last) = prev {
                draw.line()
                    .points(last, point)
                    .weight(1.0)
                    .color(hsla(0.12, 0.70, 0.62, 0.08));
            }
            prev = Some(point);
        }
    }
}

fn draw_darkclouds_visual(draw: &Draw, rect: Rect, time: f32) {
    let cols = 72;
    let rows = 40;
    let cell_w = rect.w() / cols as f32;
    let cell_h = rect.h() / rows as f32;

    for gx in 0..cols {
        for gy in 0..rows {
            let px = map_range(gx, 0, cols - 1, rect.left(), rect.right());
            let py = map_range(gy, 0, rows - 1, rect.bottom(), rect.top());
            let nx = px / rect.w();
            let ny = py / rect.h();
            let cloud = layered_fbm(nx * 1.4 + 0.15, ny * 1.3 - 0.2, time * 0.10);
            let swirl = layered_wave(nx * 2.7, ny * 2.0, time * 0.16);
            let density = (cloud * 0.78 + swirl * 0.22).clamp(0.0, 1.0);
            let vignette = 1.0 - ((nx * 1.5).powi(2) + (ny * 1.8).powi(2)).clamp(0.0, 1.0);
            let lit = 0.06 + density * vignette * 0.18;
            let alpha = 0.22 + density * 0.42;
            draw.rect()
                .x_y(px, py)
                .w_h(cell_w + 1.0, cell_h + 1.0)
                .color(hsla(
                    0.60 - density * 0.08,
                    0.26 + density * 0.18,
                    lit,
                    alpha,
                ));
        }
    }
}

fn draw_mandelbulb_visual(draw: &Draw, rect: Rect, time: f32) {
    let scene = draw
        .pitch(0.66 + 0.14 * (time * 0.21).sin())
        .yaw(time * 0.28)
        .roll(0.18);

    for i in 0..42 {
        for j in 0..22 {
            let theta = map_range(i, 0, 41, 0.0, TAU);
            let phi = map_range(j, 0, 21, -PI * 0.5, PI * 0.5);
            let wave = ((theta * 3.0 + time * 0.2).sin() * (phi * 4.0).cos()).abs();
            let radius = rect.h().min(rect.w()) * (0.10 + 0.18 * wave.powf(1.8));
            let x = radius * phi.cos() * theta.cos();
            let y = radius * phi.sin();
            let z = radius * phi.cos() * theta.sin();
            let size = 1.8 + 3.0 * wave;
            scene.ellipse().x_y_z(x, y, z).w_h(size, size).color(hsla(
                0.08 + wave * 0.18,
                0.82,
                0.62,
                0.10 + wave * 0.38,
            ));
        }
    }
}

fn draw_smoothvoro_visual(draw: &Draw, rect: Rect, time: f32) {
    let mut seeds = Vec::with_capacity(9);
    for idx in 0..9 {
        let fi = idx as f32;
        let x = (fi * 0.73 + time * 0.11).sin() * rect.w() * 0.34
            + (fi * 1.91 + time * 0.07).cos() * rect.w() * 0.12;
        let y = (fi * 1.17 - time * 0.13).cos() * rect.h() * 0.28
            + (fi * 0.61 + time * 0.09).sin() * rect.h() * 0.14;
        seeds.push(pt2(x, y));
    }

    let cols = 64;
    let rows = 38;
    let cell_w = rect.w() / cols as f32;
    let cell_h = rect.h() / rows as f32;

    for gx in 0..cols {
        for gy in 0..rows {
            let px = map_range(gx, 0, cols - 1, rect.left(), rect.right());
            let py = map_range(gy, 0, rows - 1, rect.bottom(), rect.top());
            let p = pt2(px, py);

            let mut nearest = (f32::MAX, 0usize);
            let mut second = f32::MAX;
            for (idx, seed) in seeds.iter().enumerate() {
                let d = p.distance(*seed);
                if d < nearest.0 {
                    second = nearest.0;
                    nearest = (d, idx);
                } else if d < second {
                    second = d;
                }
            }

            let edge = ((second - nearest.0) / 38.0).clamp(0.0, 1.0);
            let hue = 0.08 + 0.76 * hash01(nearest.1 as f32 * 3.17);
            let light = 0.26 + edge * 0.36;
            draw.rect()
                .x_y(px, py)
                .w_h(cell_w + 1.0, cell_h + 1.0)
                .color(hsla(hue, 0.62, light, 0.84));
        }
    }

    for seed in seeds {
        draw.ellipse()
            .x_y(seed.x, seed.y)
            .w_h(8.0, 8.0)
            .color(srgba(1.0, 1.0, 1.0, 0.82));
    }
}

fn draw_winterflake_visual(draw: &Draw, rect: Rect, time: f32) {
    let radius = rect.h().min(rect.w()) * 0.18;
    let center = pt2(0.0, 10.0);
    for arm in 0..6 {
        let angle = arm as f32 * TAU / 6.0 + time * 0.03;
        let tip = center + vec2(angle.cos(), angle.sin()) * radius;
        draw_flake_branch(draw, center, tip, 4, time, angle);
    }

    for i in 0..180 {
        let fi = i as f32;
        let x = map_range(hash01(fi * 2.17 + 0.3), 0.0, 1.0, rect.left(), rect.right());
        let fall = (time * (8.0 + hash01(fi * 1.13) * 20.0) + fi * 1.7).rem_euclid(rect.h());
        let y = rect.top() - fall;
        let size = 1.0 + hash01(fi * 0.27) * 2.4;
        draw.ellipse().x_y(x, y).w_h(size, size).color(srgba(
            0.94,
            0.97,
            1.0,
            0.18 + hash01(fi * 0.81) * 0.36,
        ));
    }
}

fn draw_gpuattractor_visual(draw: &Draw, rect: Rect, time: f32) {
    let a = 1.6 + 0.2 * (time * 0.13).sin();
    let b = -2.2 + 0.3 * (time * 0.17 + 0.4).cos();
    let c = 1.9 + 0.3 * (time * 0.11 + 1.1).sin();
    let d = -0.9 + 0.2 * (time * 0.19 + 2.2).cos();

    let mut x = 0.1;
    let mut y = 0.0;
    for i in 0..8000 {
        let nx = (a * y).sin() - (b * x).cos();
        let ny = (c * x).sin() - (d * y).cos();
        x = nx;
        y = ny;

        if i < 50 {
            continue;
        }

        let px = x * rect.w() * 0.16;
        let py = y * rect.h() * 0.16;
        let alpha = 0.04 + 0.18 * ((i as f32 / 8000.0).powf(0.35));
        draw.ellipse().x_y(px, py).w_h(1.6, 1.6).color(hsla(
            0.56 + 0.18 * y.sin().abs(),
            0.82,
            0.66,
            alpha,
        ));
    }
}

fn layered_wave(x: f32, y: f32, t: f32) -> f32 {
    ((x * 3.1 + t).sin() + (y * 4.3 - t * 1.4).cos() + ((x + y) * 5.2 + t * 0.6).sin()) / 3.0
}

fn layered_fbm(x: f32, y: f32, t: f32) -> f32 {
    let mut value = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..4 {
        value += amp * (0.5 + 0.5 * layered_wave(x * freq, y * freq, t * freq * 0.85));
        amp *= 0.5;
        freq *= 1.9;
    }
    value.clamp(0.0, 1.0)
}

fn hash01(v: f32) -> f32 {
    let fract = (v.sin() * 43_758.547).fract();
    if fract < 0.0 { fract + 1.0 } else { fract }
}

fn draw_branch(draw: &Draw, start: Point2, end: Point2, depth: u32, time: f32, width: f32) {
    let t = depth as f32 / 9.0;
    draw.line()
        .points(start, end)
        .weight((width + depth as f32 * 0.25).max(1.0))
        .color(hsla(0.10 + t * 0.20, 0.54, 0.28 + t * 0.18, 0.95));

    if depth == 0 {
        draw.ellipse().x_y(end.x, end.y).w_h(6.0, 6.0).color(hsla(
            0.28 + 0.12 * (time * 0.4).sin().abs(),
            0.62,
            0.62,
            0.82,
        ));
        return;
    }

    let dir = end - start;
    let length = dir.length() * (0.68 + 0.03 * (depth as f32).sin());
    let base = dir.angle();
    let sway = 0.12 * (time * 0.7 + depth as f32 * 0.4).sin();
    let spread = 0.32 + 0.05 * (time * 0.2 + depth as f32).cos();
    let left = end + vec2((base + spread + sway).cos(), (base + spread + sway).sin()) * length;
    let right = end + vec2((base - spread + sway).cos(), (base - spread + sway).sin()) * length;
    let mid = end + vec2((base + sway * 0.4).cos(), (base + sway * 0.4).sin()) * length * 0.72;

    draw_branch(draw, end, left, depth - 1, time, width * 0.88);
    draw_branch(draw, end, right, depth - 1, time, width * 0.86);
    if depth % 2 == 0 {
        draw_branch(draw, end, mid, depth - 1, time, width * 0.70);
    }
}

fn draw_flake_branch(draw: &Draw, start: Point2, end: Point2, depth: u32, time: f32, angle: f32) {
    draw.line()
        .points(start, end)
        .weight(1.0 + depth as f32 * 0.36)
        .color(hsla(0.58, 0.30, 0.84, 0.92));

    if depth == 0 {
        return;
    }

    let dir = end - start;
    let len = dir.length();
    let branch_len = len * (0.30 + 0.02 * depth as f32);
    let split = 0.72 + 0.06 * (time * 0.4).sin();
    let anchor = start + dir * split;
    let spread = 0.74 + 0.10 * (time * 0.7 + depth as f32).cos();
    let left = anchor + vec2((angle + spread).cos(), (angle + spread).sin()) * branch_len;
    let right = anchor + vec2((angle - spread).cos(), (angle - spread).sin()) * branch_len;
    let stem = anchor + vec2(angle.cos(), angle.sin()) * branch_len * 0.76;

    draw_flake_branch(draw, anchor, left, depth - 1, time, angle + spread);
    draw_flake_branch(draw, anchor, right, depth - 1, time, angle - spread);
    draw_flake_branch(draw, anchor, stem, depth - 1, time, angle);
}

fn atlas_layout(win: Rect) -> Vec<TileLayout> {
    let count = PRIMITIVES.len();
    let available_w = win.w() - OUTER_MARGIN * 2.0;
    let columns = if available_w >= 1400.0 {
        4
    } else if available_w >= 960.0 {
        3
    } else {
        2
    };
    let rows = count.div_ceil(columns);
    let content_h = win.h() - HEADER_H - OUTER_MARGIN * 2.0;
    let tile_w = (available_w - GUTTER * (columns as f32 - 1.0)) / columns as f32;
    let tile_h = (content_h - GUTTER * (rows as f32 - 1.0)) / rows as f32;
    let top = win.top() - HEADER_H - tile_h * 0.5;

    let mut layouts = Vec::with_capacity(count);
    for row in 0..rows {
        let row_start = row * columns;
        let row_len = (count - row_start).min(columns);
        let row_width = tile_w * row_len as f32 + GUTTER * (row_len.saturating_sub(1) as f32);
        let start_x = -row_width * 0.5 + tile_w * 0.5;
        for col in 0..row_len {
            let x = start_x + col as f32 * (tile_w + GUTTER);
            let y = top - row as f32 * (tile_h + GUTTER);
            layouts.push(TileLayout {
                frame: Rect::from_xy_wh(pt2(x, y), vec2(tile_w, tile_h)),
            });
        }
    }

    layouts
}

fn draw_tile(draw: &Draw, layout: TileLayout, primitive: PrimitiveKind, model: &Model) {
    let tile_draw = draw.x_y(layout.frame.x(), layout.frame.y());
    let local = Rect::from_w_h(layout.frame.w(), layout.frame.h());
    let sample = Rect::from_xy_wh(
        pt2(0.0, 8.0),
        vec2(local.w() - 28.0, (local.h() - 106.0).max(90.0)),
    );
    let title = Rect::from_xy_wh(pt2(0.0, local.top() - 22.0), vec2(local.w() - 26.0, 28.0));
    let detail = Rect::from_xy_wh(
        pt2(0.0, local.bottom() + 34.0),
        vec2(local.w() - 28.0, 50.0),
    );

    tile_draw
        .rect()
        .w_h(local.w(), local.h())
        .color(ui_color(CARD_FILL))
        .stroke(ui_color(CARD_STROKE))
        .stroke_weight(1.5);

    tile_draw
        .rect()
        .x_y(sample.x(), sample.y())
        .w_h(sample.w(), sample.h())
        .color(srgba(0.12, 0.14, 0.19, 0.92))
        .stroke(srgba(0.85, 0.93, 1.0, 0.06))
        .stroke_weight(1.0);

    tile_draw
        .text(primitive.title())
        .x_y(title.x(), title.y())
        .w_h(title.w(), title.h())
        .left_justify()
        .align_text_top()
        .font_size(18)
        .color(ui_color(LABEL));

    tile_draw
        .text(primitive.details())
        .x_y(detail.x(), detail.y())
        .w_h(detail.w(), detail.h())
        .left_justify()
        .align_text_top()
        .line_spacing(4.0)
        .font_size(12)
        .color(ui_color(CODE));

    primitive.draw_sample(&tile_draw, sample, model);
}

impl PrimitiveKind {
    fn title(self) -> &'static str {
        match self {
            Self::Background => "background",
            Self::Rect => "rect",
            Self::Ellipse => "ellipse",
            Self::Line => "line",
            Self::Arrow => "arrow",
            Self::Tri => "tri",
            Self::Quad => "quad",
            Self::Polygon => "polygon",
            Self::Polyline => "polyline",
            Self::Path => "path",
            Self::Mesh => "mesh",
            Self::Text => "text",
            Self::Texture => "texture",
        }
    }

    fn details(self) -> &'static str {
        match self {
            Self::Background => "background().color(..)\nclears the frame before every tile",
            Self::Rect => "x_y(..) · w_h(..)\nrgba(..) · stroke(..) · rotate(..)",
            Self::Ellipse => "xy(..) · radius(..)\nhsla(..) · stroke(..)",
            Self::Line => "points(..) · weight(..)\ncolor(..) · caps_round(..)",
            Self::Arrow => "points(..) · weight(..)\nhead_length(..) · head_width(..)",
            Self::Tri => "points(..) · color(..)\nstroke(..) · rotate(..)",
            Self::Quad => "points(..) · rgb(..)\nstroke(..) · rotate(..)",
            Self::Polygon => "points_colored(..)\nstroke(..) · stroke_weight(..)",
            Self::Polyline => "points(..) · weight(..)\ncaps_round(..) · color(..)",
            Self::Path => "fill().events(..)\nstroke().events(..)",
            Self::Mesh => "mesh().tris_colored(..)\nrotate(..) · scale(..)",
            Self::Text => "text(\"...\") · font_size(..)\ncolor(..) · rotate(..) · scale(..)",
            Self::Texture => "texture(&tex) · w_h(..)\narea(..) · rotate(..) · scale(..)",
        }
    }

    fn draw_sample(self, draw: &Draw, sample: Rect, model: &Model) {
        let sample_draw = draw.x_y(sample.x(), sample.y());
        let w = sample.w();
        let h = sample.h();

        match self {
            Self::Background => {
                sample_draw
                    .rect()
                    .w_h(w, h)
                    .color(srgba(0.08, 0.12, 0.18, 1.0));

                sample_draw
                    .ellipse()
                    .x_y(-w * 0.22, h * 0.1)
                    .w_h(w * 0.42, h * 0.54)
                    .color(hsla(0.58, 0.52, 0.53, 0.34));

                sample_draw
                    .rect()
                    .x_y(w * 0.17, -h * 0.08)
                    .w_h(w * 0.28, h * 0.52)
                    .rotate(0.26)
                    .color(hsla(0.08, 0.65, 0.56, 0.22));

                sample_draw
                    .text("Whole-frame tone")
                    .font_size(16)
                    .color(ui_color(LABEL))
                    .x_y(0.0, -h * 0.32);
            }
            Self::Rect => {
                sample_draw
                    .rect()
                    .x_y(-12.0, 0.0)
                    .w_h(w * 0.42, h * 0.54)
                    .rotate(0.24)
                    .color(rgba(0.20, 0.71, 0.88, 0.72))
                    .stroke(srgba(1.0, 1.0, 1.0, 0.9))
                    .stroke_weight(3.5);
            }
            Self::Ellipse => {
                sample_draw
                    .ellipse()
                    .xy(pt2(10.0, 0.0))
                    .radius(h.min(w) * 0.2)
                    .w_h(w * 0.48, h * 0.52)
                    .color(hsla(0.12, 0.78, 0.61, 0.78))
                    .stroke(srgba(0.95, 0.97, 1.0, 0.95))
                    .stroke_weight(3.0);
            }
            Self::Line => {
                sample_draw
                    .line()
                    .points(
                        pt2(sample.left() * 0.58, sample.bottom() * 0.44),
                        pt2(sample.right() * 0.54, sample.top() * 0.38),
                    )
                    .weight(6.0)
                    .caps_round()
                    .color(hsla(0.57, 0.72, 0.64, 0.95));
            }
            Self::Arrow => {
                sample_draw
                    .arrow()
                    .points(
                        pt2(sample.left() * 0.54, sample.bottom() * 0.36),
                        pt2(sample.right() * 0.48, sample.top() * 0.28),
                    )
                    .weight(5.0)
                    .head_length(24.0)
                    .head_width(18.0)
                    .color(hsla(0.02, 0.76, 0.67, 0.96));
            }
            Self::Tri => {
                sample_draw
                    .tri()
                    .points(pt2(-64.0, -46.0), pt2(66.0, -18.0), pt2(-20.0, 56.0))
                    .rotate(0.32)
                    .color(hsla(0.30, 0.68, 0.54, 0.82))
                    .stroke(srgba(1.0, 1.0, 1.0, 0.92))
                    .stroke_weight(3.0);
            }
            Self::Quad => {
                sample_draw
                    .quad()
                    .points(
                        pt2(-74.0, -44.0),
                        pt2(-42.0, 52.0),
                        pt2(54.0, 44.0),
                        pt2(76.0, -36.0),
                    )
                    .rotate(-0.18)
                    .rgb(0.93, 0.54, 0.25)
                    .stroke_weight(3.0);
            }
            Self::Polygon => {
                let points = [
                    (pt2(-70.0, -8.0), hsla(0.01, 0.78, 0.63, 0.88)),
                    (pt2(-44.0, 52.0), hsla(0.10, 0.84, 0.61, 0.88)),
                    (pt2(14.0, 64.0), hsla(0.18, 0.80, 0.60, 0.88)),
                    (pt2(70.0, 14.0), hsla(0.54, 0.75, 0.61, 0.88)),
                    (pt2(42.0, -56.0), hsla(0.63, 0.60, 0.55, 0.88)),
                    (pt2(-28.0, -62.0), hsla(0.76, 0.48, 0.58, 0.88)),
                ];

                sample_draw
                    .polygon()
                    .stroke(srgba(1.0, 1.0, 1.0, 0.78))
                    .stroke_weight(2.0)
                    .points_colored(points);
            }
            Self::Polyline => {
                let wave = [
                    pt2(-82.0, -16.0),
                    pt2(-42.0, 44.0),
                    pt2(0.0, -6.0),
                    pt2(38.0, 32.0),
                    pt2(84.0, -30.0),
                ];

                sample_draw
                    .polyline()
                    .weight(6.0)
                    .caps_round()
                    .color(hsla(0.56, 0.66, 0.62, 0.96))
                    .points(wave);
            }
            Self::Path => {
                let filled = nannou::geom::path::path()
                    .begin(pt2(-72.0, -6.0))
                    .quadratic_bezier_to(pt2(-28.0, 58.0), pt2(18.0, 8.0))
                    .cubic_bezier_to(pt2(54.0, -34.0), pt2(18.0, -66.0), pt2(-44.0, -54.0))
                    .close()
                    .build();

                let mut open = nannou::geom::path::path()
                    .begin(pt2(-78.0, -18.0))
                    .cubic_bezier_to(pt2(-32.0, 56.0), pt2(26.0, -56.0), pt2(80.0, 18.0));
                open.inner_mut().end(false);
                let open = open.build();

                sample_draw
                    .path()
                    .fill()
                    .color(hsla(0.86, 0.66, 0.61, 0.72))
                    .events(filled.iter());

                sample_draw
                    .path()
                    .stroke()
                    .weight(4.0)
                    .caps_round()
                    .color(srgba(0.97, 0.98, 1.0, 0.94))
                    .events(open.iter());
            }
            Self::Mesh => {
                let mesh_draw = sample_draw.rotate(0.15).scale(0.92);
                let tris = [
                    geom::Tri([
                        (pt3(-70.0, -44.0, 0.0), rgba(0.96, 0.35, 0.30, 0.92)),
                        (pt3(-18.0, 58.0, 0.0), rgba(0.98, 0.88, 0.34, 0.92)),
                        (pt3(26.0, -10.0, 0.0), rgba(0.27, 0.75, 0.92, 0.92)),
                    ]),
                    geom::Tri([
                        (pt3(26.0, -10.0, 0.0), rgba(0.27, 0.75, 0.92, 0.92)),
                        (pt3(-18.0, 58.0, 0.0), rgba(0.98, 0.88, 0.34, 0.92)),
                        (pt3(76.0, 36.0, 0.0), rgba(0.38, 0.92, 0.62, 0.92)),
                    ]),
                ];

                mesh_draw.mesh().tris_colored(tris);
            }
            Self::Text => {
                sample_draw
                    .text("ATLAS")
                    .font_size(34)
                    .color(hsla(0.15, 0.90, 0.66, 0.94))
                    .x_y(-8.0, 18.0);

                sample_draw
                    .scale(0.78)
                    .text("rotate + scale")
                    .font_size(22)
                    .rotate(-0.18)
                    .color(hsla(0.57, 0.74, 0.66, 0.94))
                    .x_y(26.0, -34.0);
            }
            Self::Texture => {
                sample_draw
                    .texture(&model.swatch_texture)
                    .w_h(w * 0.46, h * 0.54)
                    .rotate(0.16);

                sample_draw
                    .scale(0.74)
                    .texture(&model.swatch_texture)
                    .area(Rect::from_xy_wh(pt2(0.52, 0.54), vec2(0.58, 0.58)))
                    .w_h(w * 0.42, h * 0.48)
                    .rotate(-0.2)
                    .x_y(34.0, -10.0);
            }
        }
    }
}

fn draw_2d_transform_demo(draw: &Draw, sample: Rect, time: f32) {
    let panel = draw.x_y(sample.x(), sample.y());
    let local = Rect::from_w_h(sample.w(), sample.h());
    let tx = map_range(
        (time * 0.55).sin(),
        -1.0,
        1.0,
        -local.w() * 0.2,
        local.w() * 0.2,
    );
    let ty = map_range(
        (time * 0.35 + 0.8).sin(),
        -1.0,
        1.0,
        -local.h() * 0.14,
        local.h() * 0.14,
    );
    let theta = time * 0.9;
    let sx = map_range((time * 1.2).sin(), -1.0, 1.0, 0.72, 1.32);
    let sy = map_range((time * 0.95 + 0.7).sin(), -1.0, 1.0, 1.28, 0.76);

    draw_reference_grid(&panel, local, 8, 0.045);
    draw_local_axes(&panel, 76.0, 0.28);

    panel
        .rect()
        .w_h(108.0, 64.0)
        .color(srgba(1.0, 1.0, 1.0, 0.02))
        .stroke(srgba(1.0, 1.0, 1.0, 0.16))
        .stroke_weight(1.0);

    panel
        .arrow()
        .points(pt2(0.0, 0.0), pt2(tx, ty))
        .weight(2.0)
        .head_length(10.0)
        .head_width(8.0)
        .color(hsla(0.56, 0.75, 0.68, 0.92));

    let translated = panel.x_y(tx, ty);
    translated
        .rect()
        .w_h(108.0, 64.0)
        .color(srgba(0.38, 0.78, 1.0, 0.05))
        .stroke(srgba(0.38, 0.78, 1.0, 0.42))
        .stroke_weight(1.5);
    draw_local_axes(&translated, 62.0, 0.72);

    let rotated = translated.rotate(theta);
    draw_local_axes(&rotated, 56.0, 0.94);

    let scaled = rotated.scale_axes(vec3(sx, sy, 1.0));
    scaled
        .rect()
        .w_h(108.0, 64.0)
        .color(hsla(0.14, 0.85, 0.63, 0.84))
        .stroke(srgba(1.0, 1.0, 1.0, 0.92))
        .stroke_weight(2.4);
    scaled
        .ellipse()
        .x_y(22.0, 0.0)
        .w_h(16.0, 16.0)
        .color(srgba(1.0, 1.0, 1.0, 0.94));

    panel
        .text("world")
        .x_y(-54.0, -52.0)
        .font_size(11)
        .color(ui_color(MUTED));
    panel
        .text("translated origin")
        .x_y(tx + 34.0, ty - 46.0)
        .font_size(11)
        .color(hsla(0.56, 0.75, 0.72, 0.92));
    panel
        .text(&format!(
            "tx {tx:>5.0}   ty {ty:>5.0}\ntheta {theta:>4.2} rad\nsx {sx:>4.2}   sy {sy:>4.2}",
            theta = theta.rem_euclid(std::f32::consts::TAU)
        ))
        .x_y(local.left() + 92.0, local.top() - 28.0)
        .w_h(180.0, 60.0)
        .left_justify()
        .align_text_top()
        .font_size(12)
        .line_spacing(4.0)
        .color(ui_color(LABEL));
}

fn draw_3d_transform_demo(draw: &Draw, sample: Rect, time: f32, _model: &Model) {
    let panel = draw.x_y(sample.x(), sample.y());
    let local = Rect::from_w_h(sample.w(), sample.h());
    let pitch = 0.5 + 0.2 * (time * 0.65).sin();
    let yaw = time * 0.6;
    let roll = 0.18 + 0.12 * (time * 0.9).sin();
    let sy = map_range((time * 0.9).sin(), -1.0, 1.0, 0.72, 1.14);
    let sz = map_range((time * 0.6 + 1.1).sin(), -1.0, 1.0, 0.82, 1.28);
    let depth = 120.0 * (time * 0.3).sin();

    draw_reference_grid(&panel, local, 7, 0.032);

    let scene = panel
        .x_y_z(0.0, -8.0, depth)
        .pitch(pitch)
        .yaw(yaw)
        .roll(roll)
        .scale_axes(vec3(1.0, sy, sz));

    draw_plane_grid(&scene, 4, 22.0, 0.18);
    draw_local_axes(&scene, 74.0, 0.9);

    let z_axis = scene.yaw(-std::f32::consts::FRAC_PI_2);
    z_axis
        .arrow()
        .points(pt2(0.0, 0.0), pt2(74.0, 0.0))
        .weight(2.4)
        .head_length(10.0)
        .head_width(8.0)
        .color(hsla(0.60, 0.72, 0.66, 0.94));

    draw_cube(&scene, 54.0, 0.84);

    panel
        .text(&format!(
            "pitch {pitch:>4.2}\nyaw   {yaw:>4.2}\nroll  {roll:>4.2}\nz     {depth:>5.0}\nscale_axes(1.00, {sy:>4.2}, {sz:>4.2})"
        ))
        .x_y(local.left() + 104.0, local.top() - 40.0)
        .w_h(210.0, 96.0)
        .left_justify()
        .align_text_top()
        .font_size(12)
        .line_spacing(4.0)
        .color(ui_color(LABEL));

    draw_axis_legend(&panel, local.left() + 56.0, local.bottom() + 22.0);
}

fn draw_orthographic_depth_demo(draw: &Draw, sample: Rect, time: f32) {
    let panel = draw.x_y(sample.x(), sample.y());
    let local = Rect::from_w_h(sample.w(), sample.h());
    let slots = [
        (-local.w() * 0.3, -220.0_f32),
        (0.0_f32, 0.0_f32),
        (local.w() * 0.3, 220.0_f32),
    ];

    draw_reference_grid(&panel, local, 10, 0.022);

    panel
        .text("Same cube mesh, same box size, different z positions")
        .x_y(0.0, local.top() - 18.0)
        .font_size(13)
        .color(ui_color(MUTED));

    for (idx, (x, z)) in slots.into_iter().enumerate() {
        panel
            .rect()
            .x_y(x, 8.0)
            .w_h(160.0, local.h() * 0.56)
            .color(srgba(1.0, 1.0, 1.0, 0.018))
            .stroke(srgba(1.0, 1.0, 1.0, 0.12))
            .stroke_weight(1.0);

        let cube = panel
            .x_y_z(x, 22.0, z)
            .pitch(0.54)
            .yaw(time * 0.55 + idx as f32 * 0.45)
            .roll(0.16);
        draw_cube(&cube, 40.0, 0.86);

        panel
            .text(&format!("z = {z:+.0}"))
            .x_y(x, local.bottom() + 28.0)
            .font_size(13)
            .color(ui_color(LABEL));

        panel
            .text("same projected box")
            .x_y(x, local.bottom() + 10.0)
            .font_size(11)
            .color(ui_color(MUTED));
    }
}

fn draw_reference_grid(draw: &Draw, rect: Rect, divisions: usize, alpha: f32) {
    for i in 0..=divisions {
        let x = map_range(i, 0, divisions, rect.left(), rect.right());
        draw.line()
            .points(pt2(x, rect.bottom()), pt2(x, rect.top()))
            .weight(1.0)
            .color(srgba(1.0, 1.0, 1.0, alpha));
    }

    for i in 0..=divisions {
        let y = map_range(i, 0, divisions, rect.bottom(), rect.top());
        draw.line()
            .points(pt2(rect.left(), y), pt2(rect.right(), y))
            .weight(1.0)
            .color(srgba(1.0, 1.0, 1.0, alpha));
    }

    draw.line()
        .points(pt2(rect.left(), 0.0), pt2(rect.right(), 0.0))
        .weight(1.2)
        .color(srgba(1.0, 1.0, 1.0, alpha * 2.0));
    draw.line()
        .points(pt2(0.0, rect.bottom()), pt2(0.0, rect.top()))
        .weight(1.2)
        .color(srgba(1.0, 1.0, 1.0, alpha * 2.0));
}

fn draw_plane_grid(draw: &Draw, half_steps: i32, step: f32, alpha: f32) {
    let extent = half_steps as f32 * step;
    for i in -half_steps..=half_steps {
        let pos = i as f32 * step;
        draw.line()
            .points(pt2(-extent, pos), pt2(extent, pos))
            .weight(1.0)
            .color(srgba(1.0, 1.0, 1.0, alpha));
        draw.line()
            .points(pt2(pos, -extent), pt2(pos, extent))
            .weight(1.0)
            .color(srgba(1.0, 1.0, 1.0, alpha));
    }
}

fn draw_local_axes(draw: &Draw, size: f32, alpha: f32) {
    draw.arrow()
        .points(pt2(0.0, 0.0), pt2(size, 0.0))
        .weight(2.0)
        .head_length(9.0)
        .head_width(7.0)
        .color(hsla(0.01, 0.82, 0.66, alpha));

    draw.arrow()
        .points(pt2(0.0, 0.0), pt2(0.0, size))
        .weight(2.0)
        .head_length(9.0)
        .head_width(7.0)
        .color(hsla(0.32, 0.76, 0.62, alpha));

    draw.ellipse()
        .w_h(8.0, 8.0)
        .color(srgba(1.0, 1.0, 1.0, alpha * 0.92));
}

fn draw_axis_legend(draw: &Draw, left: f32, bottom: f32) {
    let items = [
        ("x axis", hsla(0.01, 0.82, 0.66, 0.94)),
        ("y axis", hsla(0.32, 0.76, 0.62, 0.94)),
        ("z axis", hsla(0.60, 0.72, 0.66, 0.94)),
    ];

    for (idx, (label, color)) in items.into_iter().enumerate() {
        let y = bottom + idx as f32 * 18.0;
        draw.rect().x_y(left, y).w_h(10.0, 10.0).color(color);
        draw.text(label)
            .x_y(left + 54.0, y - 1.0)
            .w_h(86.0, 14.0)
            .left_justify()
            .align_text_middle_y()
            .font_size(11)
            .color(ui_color(MUTED));
    }
}

fn draw_cube(draw: &Draw, size: f32, alpha: f32) {
    let hs = size * 0.5;
    let red = hsla(0.01, 0.82, 0.60, alpha);
    let red_dark = hsla(0.01, 0.62, 0.38, alpha * 0.95);
    let green = hsla(0.32, 0.76, 0.58, alpha);
    let green_dark = hsla(0.32, 0.52, 0.34, alpha * 0.95);
    let blue = hsla(0.60, 0.72, 0.60, alpha);
    let blue_dark = hsla(0.60, 0.48, 0.36, alpha * 0.95);

    let tris = vec![
        geom::Tri([
            (pt3(hs, -hs, -hs), red),
            (pt3(hs, hs, -hs), red),
            (pt3(hs, hs, hs), red),
        ]),
        geom::Tri([
            (pt3(hs, -hs, -hs), red),
            (pt3(hs, hs, hs), red),
            (pt3(hs, -hs, hs), red),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, -hs), red_dark),
            (pt3(-hs, hs, hs), red_dark),
            (pt3(-hs, hs, -hs), red_dark),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, -hs), red_dark),
            (pt3(-hs, -hs, hs), red_dark),
            (pt3(-hs, hs, hs), red_dark),
        ]),
        geom::Tri([
            (pt3(-hs, hs, -hs), green),
            (pt3(-hs, hs, hs), green),
            (pt3(hs, hs, hs), green),
        ]),
        geom::Tri([
            (pt3(-hs, hs, -hs), green),
            (pt3(hs, hs, hs), green),
            (pt3(hs, hs, -hs), green),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, -hs), green_dark),
            (pt3(hs, -hs, hs), green_dark),
            (pt3(-hs, -hs, hs), green_dark),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, -hs), green_dark),
            (pt3(hs, -hs, -hs), green_dark),
            (pt3(hs, -hs, hs), green_dark),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, hs), blue),
            (pt3(hs, hs, hs), blue),
            (pt3(-hs, hs, hs), blue),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, hs), blue),
            (pt3(hs, -hs, hs), blue),
            (pt3(hs, hs, hs), blue),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, -hs), blue_dark),
            (pt3(-hs, hs, -hs), blue_dark),
            (pt3(hs, hs, -hs), blue_dark),
        ]),
        geom::Tri([
            (pt3(-hs, -hs, -hs), blue_dark),
            (pt3(hs, hs, -hs), blue_dark),
            (pt3(hs, -hs, -hs), blue_dark),
        ]),
    ];

    draw.mesh().tris_colored(tris);
}

fn build_reference_texture(app: &App) -> wgpu::Texture {
    let image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_fn(160, 160, |x, y| {
        let u = x as f32 / 159.0;
        let v = y as f32 / 159.0;
        let checker = ((x / 16) + (y / 16)) % 2;
        let grid = if checker == 0 { 0.12 } else { 0.26 };
        let r = ((0.24 + 0.72 * u) * 255.0) as u8;
        let g = ((0.18 + 0.68 * v) * 255.0) as u8;
        let b = ((0.26 + grid + 0.34 * (1.0 - u)) * 255.0).min(255.0) as u8;
        Rgba([r, g, b, 255])
    });

    let dynamic = DynamicImage::ImageRgba8(image);
    wgpu::Texture::from_image(app, &dynamic)
}
