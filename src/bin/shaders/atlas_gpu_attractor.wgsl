struct GPUAttractorUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: GPUAttractorUniforms;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.2831855;

fn panel_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    return (frag_coord - uni.stage_rect.xy) / uni.stage_rect.zw;
}

fn centered_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    let uv = panel_uv(frag_coord);
    let aspect = uni.stage_rect.z / max(uni.stage_rect.w, 1.0);
    return vec2<f32>((uv.x - 0.5) * aspect, 0.5 - uv.y);
}

fn lambda() -> f32 {
    return uni.params0.x;
}

fn theta() -> f32 {
    return uni.params0.y;
}

fn alpha() -> f32 {
    return uni.params0.z;
}

fn sigma() -> f32 {
    return uni.params0.w;
}

fn gamma() -> f32 {
    return uni.params1.x;
}

fn blue() -> f32 {
    return uni.params1.y;
}

fn oscillate(min_value: f32, max_value: f32, interval: f32, current_time: f32) -> f32 {
    return min_value + (max_value - min_value) * 0.5 * (sin(2.0 * PI * current_time / interval) + 1.0);
}

fn clifford_attractor(p: vec2<f32>, a: f32, b: f32, c: f32, d: f32) -> vec2<f32> {
    var x = sin(a * p.y) + c * cos(a * p.x);
    var y = sin(b * p.x) + d * cos(b * p.y);
    x = sin(a * y) + c * cos(a * x);
    y = sin(b * x) + d * cos(b * y);
    x = sin(a * y) + c * cos(a * x);
    y = sin(b * x) + d * cos(b * y);
    return vec2<f32>(x, y);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = centered_uv(frag_coord.xy) * 8.0;
    let num_circles = 6;
    let num_points = 15;
    let circle_radius = oscillate(3.5, 4.5, 5.0, uni.time_data.x);
    let intensity = oscillate(0.02, 0.01, 12.0, uni.time_data.x);
    let scale = gamma();
    let time_offset = uni.time_data.x * 0.1;
    var color = vec3<f32>(0.0);

    for (var i = 0; i < num_circles; i = i + 1) {
        let angle_step = TAU / f32(num_points);
        let base_radius = f32(i + 1) * circle_radius * 0.1;
        let point_color = 0.5 + 0.5 * sin(vec3<f32>(0.1, TAU / 3.0, TAU * 2.0 / 3.0) + f32(i) * 0.87);
        for (var j = 0; j < num_points; j = j + 1) {
            let t = f32(j) * angle_step + time_offset;
            let initial_point = vec2<f32>(cos(t), sin(t)) * base_radius;
            let attractor_point = clifford_attractor(initial_point, lambda(), theta(), alpha(), sigma()) * scale;
            let dist = length(uv + attractor_point);
            color += point_color * intensity / max(dist, 0.04);
        }
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    color = sqrt(color) * blue();
    return vec4<f32>(color, 1.0);
}
