struct NeuronsUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: NeuronsUniforms;

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

fn random2(st: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(
        dot(st, vec2<f32>(127.1, 311.7)),
        dot(st, vec2<f32>(269.5, 183.3)),
    );
    return -1.0 + 2.0 * fract(sin(q) * 43758.5453123);
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

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pp = p;
    for (var i = 0; i < 5; i = i + 1) {
        value += amplitude * noise(pp);
        pp *= 2.35;
        amplitude *= 0.5;
    }
    return value;
}

fn rotate2d(r: f32) -> mat2x2<f32> {
    return mat2x2<f32>(cos(r), -sin(r), sin(r), cos(r));
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    var uv = centered_uv(frag_coord.xy) * 1.5;
    let t = uni.time_data.x / 4.0;
    let x_val4 = oscillate(0.5, 0.51, 15.0, t);
    let x_val6 = oscillate(1.5, 1.51, 45.0, t);
    let x_val7 = oscillate(0.5, 2.51, 15.0, t);

    var n = vec2<f32>(0.0, 0.0);
    var accum = vec2<f32>(0.0, 0.0);
    var p = uv + t;
    var scale = x_val6;
    let rot = rotate2d(x_val4);
    var branch = 1.78;

    for (var j = 0; j < 40; j = j + 1) {
        p = rot * p;
        n = rot * n;
        let q = rot * p * scale * f32(j) + n + vec2<f32>(t, t);
        n += branch * sin(q);
        accum += branch * cos(q) / max(scale, 0.001) * x_val7;
        branch *= 1.18;
        scale *= 1.28;
    }

    let pulse = sin(4.0 * t + length(p) + fbm(vec2<f32>(t * 8.1, length(p) * 2.1))) * 0.1 + 0.5;
    let color_offset = vec3<f32>(
        lambda() * smoothstep(0.2, 0.8, sin(n.x)),
        alpha() * smoothstep(0.2, 0.9, sin(n.y)),
        sigma() * smoothstep(0.1, 0.8, cos(n.x)),
    );
    let flow_change = vec3<f32>(
        blue() * cos(3.0 * t + accum.x),
        gamma() * sin(3.0 * t + accum.y),
        theta() * cos(3.0 * t + accum.y),
    );
    let flow_intensity = vec3<f32>(
        0.0021 / max(length(0.03 * accum), 0.01),
        smoothstep(1.5, -0.2, accum.x),
        smoothstep(-0.8, 1.2, accum.y),
    );
    let dark = oscillate(3.0, 10.51, 15.0, t);
    var color = (vec3<f32>(lambda() * pulse, alpha() * pulse, 3.1 * pulse) * color_offset
        + flow_change
        + flow_intensity)
        * ((0.02 * accum.x * accum.y) + 0.24 / max(length(dark * accum), 0.02));
    color += 0.08 * vec3<f32>(0.15, 0.28, 0.5) * fbm(uv * 2.0 + vec2<f32>(0.0, uni.time_data.x * 0.06));
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.5));
    return vec4<f32>(sqrt(color), 1.0);
}
