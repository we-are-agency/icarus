struct TreeUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: TreeUniforms;

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

fn apply_gamma(color: vec3<f32>, gamma_value: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / max(gamma_value, 0.001)));
}

fn complex_power(z: vec2<f32>, exponent: f32) -> vec2<f32> {
    let r = length(z);
    let angle = atan2(z.y, z.x);
    let r_pow = pow(max(r, 0.0001), exponent);
    let exponent_angle = exponent * angle;
    return vec2<f32>(r_pow * cos(exponent_angle), r_pow * sin(exponent_angle));
}

fn iterate_field(z_input: vec2<f32>) -> vec2<f32> {
    var z = z_input;
    var dz = vec2<f32>(0.1, 0.0);
    let bailout = 77.0;
    for (var i = 0; i < 123; i = i + 1) {
        dz = 1.5 * pow(max(length(z), 0.001), 0.5) * dz;
        z = complex_power(z, 1.5) - vec2<f32>(0.2, 0.0);
        if (dot(z, z) > bailout) {
            return vec2<f32>(f32(i), dot(z, z) / max(dot(dz, dz), 0.0001));
        }
    }
    return vec2<f32>(123.0, dot(z, z));
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = centered_uv(frag_coord.xy);
    let pan = vec2<f32>(lambda(), blue());
    let zoom = 0.24;
    let sample = (uv + pan) * zoom;
    let result = iterate_field(sample);
    let iter_ratio = result.x / 123.0;
    let sharpness = result.y;
    let col1 = 0.5 + 0.5 * cos(1.0 + uni.time_data.x + vec3<f32>(theta(), alpha(), sigma()) + PI * vec3<f32>(gamma() * sharpness));
    let col2 = 0.5 + 0.5 * cos(4.1 + uni.time_data.x + PI * vec3<f32>(sharpness));
    let color = apply_gamma(sqrt(mix(col1, col2, iter_ratio)), 0.5);
    return vec4<f32>(color, 1.0);
}
