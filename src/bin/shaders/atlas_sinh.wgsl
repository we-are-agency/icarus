struct SinhUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
    params4: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: SinhUniforms;

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
    return max(uni.params0.x, 0.001);
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

fn aa() -> i32 {
    return i32(uni.params1.z);
}

fn iter_count() -> i32 {
    return i32(uni.params1.w);
}

fn bound() -> f32 {
    return uni.params2.x;
}

fn time_divisor() -> f32 {
    return max(uni.params2.y, 0.001);
}

fn grad0() -> vec3<f32> {
    return uni.params3.xyz;
}

fn grad_shift() -> vec3<f32> {
    return uni.params4.xyz;
}

fn apply_gamma(color: vec3<f32>, gamma_value: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / max(gamma_value, 0.001)));
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn c_sinh(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sinh(z.x) * cos(z.y), cosh(z.x) * sin(z.y));
}

fn c_abs(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(abs(sin(z.x)), abs(sin(z.y)));
}

fn c_sinh_pow4(z: vec2<f32>) -> vec2<f32> {
    let sinh_z = c_sinh(z);
    return c_mul(c_mul(sinh_z, sinh_z), c_mul(sinh_z, sinh_z));
}

fn escape(z: vec2<f32>, c: vec2<f32>, time: f32) -> vec2<f32> {
    var local = z;
    for (var i = 0; i < iter_count(); i = i + 1) {
        local = c_abs(c_sinh_pow4(local)) + c;
        local = local + 0.03 * vec2<f32>(cos(1.05 * time / lambda()), cos(1.05 * time / lambda()));
        if (dot(local, local) > bound() * bound()) {
            return vec2<f32>(f32(i), dot(local, local));
        }
    }
    return vec2<f32>(f32(iter_count()), dot(local, local));
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    var col = vec3<f32>(0.0);
    let uv = centered_uv(frag_coord.xy);
    let time = uni.time_data.x / time_divisor();

    for (var m = 0; m < aa(); m = m + 1) {
        for (var n = 0; n < aa(); n = n + 1) {
            let c_value = mix(2.197, 2.99225, 0.01 + 0.01 * sin(0.1 * time / lambda()));
            let oscillation = theta() + alpha() * (sin(0.1 * time / lambda()) + blue());
            let c = vec2<f32>(oscillation, c_value);
            let result = escape(uv, c, time);
            let iter_ratio = result.x / max(f32(iter_count()), 1.0);
            let len_sq = result.y;
            let warm = vec3<f32>(0.4, 0.2, 0.0);
            let ember = vec3<f32>(0.7, 0.35, 0.15);
            let palette = mix(grad0(), warm, smoothstep(0.0, 0.5, iter_ratio));
            let palette2 = mix(warm, ember, smoothstep(0.35, 1.0, iter_ratio));
            let blend = 0.5 + 0.5 * sin(iter_ratio * 12.0 + len_sq * 0.8);
            let band = mix(palette, palette2, blend);
            let base = 0.5 + 0.5 * cos(grad_shift() + time + PI * vec3<f32>(sigma() * len_sq));
            col += sqrt(base * band);
        }
    }

    col = sqrt(col / max(f32(aa() * aa()), 1.0));
    col = apply_gamma(col, gamma());
    return vec4<f32>(col, 1.0);
}
