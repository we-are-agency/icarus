struct FluidUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: FluidUniforms;

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

fn panel_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    return (frag_coord - uni.stage_rect.xy) / max(uni.stage_rect.zw, vec2<f32>(1.0, 1.0));
}

fn scene_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    let local = frag_coord - uni.stage_rect.xy;
    let scale = max(min(uni.stage_rect.z, uni.stage_rect.w), 1.0);
    var uv = (local * 3.0 - uni.stage_rect.zw) / scale;
    uv.x -= 3.0;
    return uv;
}

fn apply_gamma(color: vec3<f32>, gamma_value: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / max(gamma_value, 0.001)));
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    var uv = scene_uv(frag_coord.xy);
    let stage_uv = panel_uv(frag_coord.xy);
    let time = uni.time_data.x;
    let frequency = lambda();
    var color = vec3<f32>(0.0);

    for (var j = 0; j < 5; j = j + 1) {
        let fj = f32(j);
        for (var i = 1; i < 5; i = i + 1) {
            let fi = f32(i);
            let divisor = fi + fj + 1.0;
            uv.x += (0.2 / divisor) * sin(fi * atan(time) * 2.0 * uv.y + time * theta() + fi * fj);
            uv.y += (1.0 / divisor) * cos(fi * 0.6 * uv.x + time * theta() + fi * fj);
            let angle = time * alpha();
            let rotation = mat2x2<f32>(cos(angle), -sin(angle), sin(angle), cos(angle));
            uv = rotation * uv;
        }

        var tex_color = textureSample(tex, tex_sampler, uv).xyz;
        tex_color += textureSample(tex, tex_sampler, uv + vec2<f32>(-0.01, 0.01)).xyz;
        tex_color += textureSample(tex, tex_sampler, uv + vec2<f32>(0.01, 0.01)).xyz;
        tex_color += textureSample(tex, tex_sampler, uv + vec2<f32>(-0.01, -0.01)).xyz;
        tex_color += textureSample(tex, tex_sampler, uv + vec2<f32>(0.01, -0.01)).xyz;
        tex_color /= 5.0;

        let angle_field = atan(uv.y);
        let col1 = 0.1
            + 0.5
                * cos(
                    frequency * (1.0 + time)
                        + vec3<f32>(sigma(), gamma_param(), blue())
                        + PI * vec3<f32>(5.0 * angle_field),
                );
        let col2 = 0.2 + 0.5 * cos(frequency * (1.1 + time) + PI * vec3<f32>(angle_field));
        let col3 = 0.2
            + 0.4
                * cos(
                    frequency * (1.0 + time)
                        + vec3<f32>(blue(), gamma_param(), sigma())
                        + PI * vec3<f32>(2.0 * sin(angle_field)),
                );

        color += tex_color + col1 + col2 + col3 + col3;
    }

    color /= 9.0;
    let background = vec3<f32>(1.0, 1.0, 1.0);
    color = mix(
        color,
        background,
        1.0 - smoothstep(0.0, abs(sin(time * 0.05) * 3.0), length(uv) - 0.1),
    );

    let vignette = 0.92 + 0.08 * smoothstep(0.95, 0.2, distance(stage_uv, vec2<f32>(0.5, 0.5)));
    let final_color = apply_gamma(clamp(color * vignette, vec3<f32>(0.0), vec3<f32>(1.5)), 0.45);
    return vec4<f32>(final_color, 1.0);
}
