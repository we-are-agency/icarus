struct ChladniUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: ChladniUniforms;

const PI: f32 = 3.14159265;
const L: f32 = 0.7;

fn panel_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    return (frag_coord - uni.stage_rect.xy) / uni.stage_rect.zw;
}

fn centered_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    let uv = panel_uv(frag_coord);
    let aspect = uni.stage_rect.z / max(uni.stage_rect.w, 1.0);
    return vec2<f32>((uv.x - 0.5) * aspect, 0.5 - uv.y);
}

fn lambda() -> f32 {
    return max(uni.params0.x, 0.001);
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

fn implicit(x: f32, y: f32) -> f32 {
    let t = uni.time_data.x / lambda();
    let n1 = 6.0 + 3.0 * sin(t);
    let m1 = 4.0 + 3.0 * cos(t);
    let n2 = 5.0 + 2.5 * cos(2.0 * t);
    let m2 = 3.0 + 2.5 * sin(2.0 * t);
    let val1 = cos(n1 * PI * x / L) * cos(m1 * PI * y / L)
        - cos(m1 * PI * x / L) * cos(n1 * PI * y / L);
    let val2 = cos(n2 * PI * x / L) * cos(m2 * PI * y / L)
        - cos(m2 * PI * x / L) * cos(n2 * PI * y / L);
    return val1 + val2;
}

fn gradient(p: vec2<f32>) -> vec2<f32> {
    let d = 0.001;
    let dx = implicit(p.x + d, p.y) - implicit(p.x - d, p.y);
    let dy = implicit(p.x, p.y + d) - implicit(p.x, p.y - d);
    return vec2<f32>(dx, dy) / (2.0 * d);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = centered_uv(frag_coord.xy);
    let grad = gradient(uv);
    let field = implicit(uv.x, uv.y);
    let unit = 24.0 / max(min(uni.stage_rect.z, uni.stage_rect.w), 1.0);
    let sharp = smoothstep(-unit, unit, abs(field) / max(length(grad), 0.0001));
    let color = 0.5 + 0.5 * cos(
        uni.time_data.x + vec3<f32>(sigma(), gamma(), 1.18) + theta() * PI * vec3<f32>(sharp),
    );
    return vec4<f32>(color, 1.0);
}
