struct DarkCloudsUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: DarkCloudsUniforms;

const PI: f32 = 3.141592653589793;

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

fn sigma() -> f32 {
    return uni.params0.z;
}

fn gamma() -> f32 {
    return uni.params0.w;
}

fn blue() -> f32 {
    return uni.params1.x;
}

fn random2(st: vec2<f32>) -> vec2<f32> {
    let transformed = vec2<f32>(
        dot(st, vec2<f32>(127.1, 311.7)),
        dot(st, vec2<f32>(269.5, 183.3)),
    );
    return -1.0 + 2.0 * fract(sin(transformed) * 43758.5453123);
}

fn oscillate(min_value: f32, max_value: f32, interval: f32, current_time: f32) -> f32 {
    return min_value + (max_value - min_value) * 0.5 * (sin(2.0 * PI * current_time / interval) + 1.0);
}

fn noise(st: vec2<f32>) -> f32 {
    let i = floor(st);
    let f = fract(st);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(
            dot(random2(i + vec2<f32>(0.0, 0.0)), f - vec2<f32>(0.0, 0.0)),
            dot(random2(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0)),
            u.x,
        ),
        mix(
            dot(random2(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0)),
            dot(random2(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0)),
            u.x,
        ),
        u.y,
    );
}

fn rotate2d(r: f32) -> mat2x2<f32> {
    return mat2x2<f32>(cos(r), -sin(r), sin(r), cos(r));
}

fn hyper_sinh(x: f32) -> f32 {
    return (exp(x) - exp(-x)) / 2.0;
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = centered_uv(frag_coord.xy) * 3.0;
    let t = uni.time_data.x;
    let oscillation_factors = vec3<f32>(
        oscillate(0.5, 0.51, 15.0, t),
        oscillate(2.0, 2.51, 15.0, t),
        oscillate(0.5, 0.51, 15.0, t),
    );

    var accum = vec2<f32>(0.0, 0.0);
    var p = uv + t / 20.0;
    var scale = oscillation_factors.y;
    let rot = rotate2d(oscillation_factors.x);
    let branch_factor = lambda();
    var n = vec2<f32>(0.0, 0.0);

    for (var j = 0; j < 45; j = j + 1) {
        p = rot * p;
        n = rot * n;
        let q = p * scale * f32(j) + n + vec2<f32>(t, t);
        n += branch_factor * cos(q);
        accum += branch_factor * cos(q) / max(scale, 0.001) * oscillation_factors.z;
        scale *= 1.4 * hyper_sinh(0.9);
    }

    let color_offset = vec3<f32>(
        0.1 * smoothstep(0.4, 1.0, sin(accum.x)),
        0.5 * smoothstep(1.0, 1.0, sin(accum.x)),
        0.1 * smoothstep(0.5, 1.0, cos(accum.x)),
    );
    let flow_change = vec3<f32>(
        1.5 * cos(t + accum.x),
        0.5 * sin(t + accum.y),
        1.5 * cos(t + accum.y),
    );
    let flow_intensity = vec3<f32>(
        0.1 / max(length(1.03 * accum), 0.01),
        smoothstep(sigma(), gamma(), accum.x),
        smoothstep(blue(), 1.0, accum.y),
    );

    let color = (vec3<f32>(0.5, 0.0, 2.1) * color_offset + flow_change + theta() * flow_intensity)
        * ((0.5 * accum.x * 0.5 * accum.y) + 0.0015 / max(length(accum), 0.01));
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.4)), 1.0);
}
