struct HelloUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: HelloUniforms;

fn panel_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    return (frag_coord - uni.stage_rect.xy) / uni.stage_rect.zw;
}

fn centered_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    let uv = panel_uv(frag_coord);
    let aspect = uni.stage_rect.z / max(uni.stage_rect.w, 1.0);
    return vec2<f32>((uv.x - 0.5) * aspect, 0.5 - uv.y);
}

fn ring(p: vec2<f32>, radius: f32, thickness: f32) -> f32 {
    let d = abs(length(p) - radius);
    return smoothstep(thickness, 0.0, d);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = centered_uv(frag_coord.xy);
    let time = uni.time_data.x;
    let wave = 0.5 + 0.5 * sin(uv.x * 7.0 - uv.y * 5.0 + time * 1.8);
    let orbit = ring(
        uv - vec2<f32>(0.24 * cos(time * 0.8), 0.18 * sin(time * 1.1)),
        0.22,
        0.018,
    );
    let pulse = ring(uv, 0.46 + 0.05 * sin(time * 1.4), 0.03);
    let beam = smoothstep(0.14, 0.0, abs(uv.y + 0.08 * sin(time + uv.x * 5.0)));
    let core = exp(-5.5 * length(uv));
    let grid = 0.5 + 0.5 * cos(vec3<f32>(0.0, 2.0, 4.0) + time + uv.xyx * 8.0);

    var color = mix(vec3<f32>(0.03, 0.06, 0.10), vec3<f32>(0.14, 0.44, 0.78), wave);
    color += 0.28 * beam * vec3<f32>(0.95, 0.62, 0.24);
    color += orbit * vec3<f32>(0.98, 0.92, 0.62);
    color += pulse * vec3<f32>(0.18, 0.80, 1.00);
    color += core * grid * 0.65;
    color *= 0.92 + 0.08 * smoothstep(1.5, 0.0, length(uv));

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
