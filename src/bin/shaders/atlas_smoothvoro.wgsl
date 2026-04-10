struct SmoothVoroUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: SmoothVoroUniforms;

@group(1) @binding(0)
var tex: texture_2d<f32>;

@group(1) @binding(1)
var tex_sampler: sampler;

const PI: f32 = 3.14159265358979323846;

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

fn gamma_param() -> f32 {
    return uni.params1.x;
}

fn blue() -> f32 {
    return uni.params1.y;
}

fn panel_coord(frag_coord: vec2<f32>) -> vec2<f32> {
    return frag_coord - uni.stage_rect.xy;
}

fn hash11(n: f32) -> f32 {
    return fract(sin(n) * 12.5453123);
}

fn osc2(min_value: f32, max_value: f32, interval: f32, current_time: f32) -> f32 {
    return min_value + (max_value - min_value) * 0.5 * (sin(2.0 * PI * current_time / interval) + 1.0);
}

fn static_point(cell: vec2<f32>) -> vec2<f32> {
    let phase = hash11(dot(cell, vec2<f32>(63.7264, 10.873)));
    let amp = 0.5 + 0.4 * hash11(dot(cell, vec2<f32>(305.21, 532.83)));
    return vec2<f32>(cos(phase * 6.2831), sin(phase * 6.2831)) * amp;
}

fn moving_point(cell: vec2<f32>, time: f32) -> vec2<f32> {
    let freq = hash11(dot(cell, vec2<f32>(12.9898, 4.1414))) * gamma_param() + gamma_param();
    let phase = hash11(dot(cell, vec2<f32>(63.7264, 10.873)));
    let amp = 0.5 + 0.4 * hash11(dot(cell, vec2<f32>(305.21, 532.83)));
    let t = time * freq + phase * 6.2831;
    return vec2<f32>(cos(t), sin(t)) * amp;
}

fn borders(min_dist: f32, smd: f32) -> f32 {
    let edge_blend = smoothstep(min_dist, min_dist + blue(), smd);
    return 1.0 - edge_blend;
}

fn smooth_voronoi(
    uv: vec2<f32>,
    time: f32,
    border: ptr<function, f32>,
    smd: ptr<function, f32>,
    ctp: ptr<function, vec2<f32>>,
    closepoint: ptr<function, vec2<f32>>,
) -> f32 {
    let g = floor(uv);
    let f = fract(uv);
    let power = osc2(alpha(), alpha(), 11.0, time);

    var min_dist = power;
    *smd = 1.0;
    *ctp = vec2<f32>(0.0);
    *closepoint = vec2<f32>(0.0);

    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let lattice = vec2<f32>(f32(x), f32(y));
            let offset = moving_point(g + lattice, time);
            let static_offset = static_point(g + lattice);
            let point = lattice + offset - f;
            let still_point = lattice + static_offset - f;
            let dist = dot(point, point);
            if (dist < min_dist) {
                *smd = min_dist;
                min_dist = dist;
                *ctp = g + lattice + offset;
                *closepoint = g + lattice + static_offset;
            } else if (dist < *smd) {
                *smd = dist;
            }
            _ = still_point;
        }
    }

    *border = borders(min_dist, *smd);
    return sqrt(min_dist);
}

fn calc_normal(uv: vec2<f32>, time: f32) -> vec3<f32> {
    let eps = 0.001;
    var border = 0.0;
    var smd = 0.0;
    var ctp = vec2<f32>(0.0);
    var closepoint = vec2<f32>(0.0);
    let dist = smooth_voronoi(uv, time, &border, &smd, &ctp, &closepoint);
    let dx = smooth_voronoi(uv + vec2<f32>(eps, 0.0), time, &border, &smd, &ctp, &closepoint) - dist;
    let dy = smooth_voronoi(uv + vec2<f32>(0.0, eps), time, &border, &smd, &ctp, &closepoint) - dist;
    return normalize(vec3<f32>(dx, dy, eps));
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let local = panel_coord(frag_coord.xy);
    let uv = lambda() * (local / max(uni.stage_rect.zw, vec2<f32>(1.0, 1.0)));
    let time = uni.time_data.x;

    var border = 0.0;
    var smd = 0.0;
    var ctp = vec2<f32>(0.0);
    var closepoint = vec2<f32>(0.0);
    let dist = smooth_voronoi(uv, time, &border, &smd, &ctp, &closepoint);

    let sample_uv = clamp(closepoint / max(lambda(), 0.001), vec2<f32>(0.0), vec2<f32>(1.0));
    let texture_color = textureSample(tex, tex_sampler, sample_uv);
    let normal = calc_normal(uv, time);
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let light_intensity = max(dot(normal, light_dir), theta());
    let lighting = mix(vec3<f32>(0.1), vec3<f32>(1.0), light_intensity);

    var color = texture_color.rgb * lighting;
    color = mix(color, vec3<f32>(1.0, 1.0, 1.0), border * sigma());
    color += smoothstep(0.16, 0.0, abs(sin(18.0 * dist - time * 0.8))) * 0.08 * vec3<f32>(0.10, 0.18, 0.24);

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
