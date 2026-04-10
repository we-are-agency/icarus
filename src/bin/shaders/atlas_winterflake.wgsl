struct WinterflakeUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
    params4: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: WinterflakeUniforms;

const PI: f32 = 3.14159265358979323846;

fn lambda() -> f32 { return uni.params0.x; }
fn theta() -> f32 { return uni.params0.y; }
fn alpha() -> f32 { return uni.params0.z; }
fn sigma() -> f32 { return uni.params0.w; }
fn gamma_param() -> f32 { return uni.params1.x; }
fn blue() -> f32 { return uni.params1.y; }
fn aa() -> f32 { return uni.params1.z; }
fn iter_count() -> f32 { return uni.params1.w; }
fn bound() -> f32 { return uni.params2.x; }
fn tt() -> f32 { return uni.params2.y; }
fn steps() -> f32 { return uni.params2.z; }
fn normal_min() -> f32 { return uni.params2.w; }
fn normal_max() -> f32 { return uni.params3.x; }
fn oscillation() -> f32 { return uni.params3.y; }
fn e() -> f32 { return uni.params3.z; }
fn f() -> f32 { return uni.params3.w; }
fn g() -> f32 { return uni.params4.x; }

fn panel_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    return (frag_coord - uni.stage_rect.xy) / uni.stage_rect.zw;
}

fn centered_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    let uv = panel_uv(frag_coord);
    let aspect = uni.stage_rect.w / max(uni.stage_rect.z, 1.0);
    var p = -1.8 + 3.5 * uv;
    p.y *= aspect;
    return p;
}

fn apply_gamma(color: vec3<f32>, gamma_value: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / max(gamma_value, 0.001)));
}

fn fmod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn sd_line(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

fn op_mod_polar_mirrored(p: vec2<f32>, theta_value: f32, offset: f32) -> vec2<f32> {
    var a = atan2(p.y, p.x) - offset;
    a = abs(fmod(a + 0.5 * theta_value, theta_value) - 0.5 * theta_value);
    return length(p) * vec2<f32>(cos(a), sin(a));
}

fn sd_snowflake(p: vec2<f32>) -> f32 {
    let mod_p = op_mod_polar_mirrored(p, radians(gamma_param()) / 6.0, radians(90.0));
    var d = sd_line(mod_p, vec2<f32>(0.0, 0.0), vec2<f32>(0.75, 0.0));
    d = min(d, sd_line(mod_p, vec2<f32>(0.5, 0.0), vec2<f32>(0.6, 0.1)));
    d = min(d, sd_line(mod_p, vec2<f32>(0.25, 0.0), vec2<f32>(0.4, 0.15)));
    return d - blue();
}

fn osc_with_pause(min_value: f32, max_value: f32, interval: f32, pause_duration: f32, current_time: f32) -> f32 {
    let cycle_time = interval * 2.0 + pause_duration;
    let phase = current_time - floor(current_time / cycle_time) * cycle_time;
    if (phase < interval) {
        return mix(max_value, min_value, phase / interval);
    } else if (phase < interval + pause_duration) {
        return min_value;
    }
    return mix(min_value, max_value, (phase - interval - pause_duration) / interval);
}

fn osc(min_value: f32, max_value: f32, interval: f32, current_time: f32) -> f32 {
    return min_value + (max_value - min_value) * 0.5 * (sin(2.0 * PI * current_time / interval) + 1.0);
}

fn get_scene_dist(p: vec2<f32>) -> f32 {
    let angle = uni.time_data.x * 0.5;
    let rot = mat2x2<f32>(cos(angle), sin(angle), -sin(angle), cos(angle));
    let rot_p = rot * p;
    return sd_snowflake(rot_p);
}

fn get_norm(p: vec2<f32>) -> vec2<f32> {
    let normal1 = osc_with_pause(normal_min(), normal_max(), oscillation(), 0.0, uni.time_data.x);
    let eps = vec2<f32>(normal1, 0.0);
    return normalize(vec2<f32>(
        get_scene_dist(p + eps.xy) - get_scene_dist(p - eps.xy),
        get_scene_dist(p + eps.yx) - get_scene_dist(p - eps.yx),
    ));
}

fn calc_lighting(ro: vec2<f32>, rd: vec2<f32>, t: f32) -> vec3<f32> {
    let p = ro + rd * t;
    let normal = get_norm(p);
    let c = osc(0.3, 0.7, 5.0, uni.time_data.x);
    let angle = atan2(rd.y, rd.x);
    let base_color = vec3<f32>(0.7, c, 1.0) + 0.2 * cos(angle + vec3<f32>(2.0, 2.0, 1.0));
    let fresnel = pow(1.0 - abs(dot(normal, -rd)), 5.0);
    let b = osc(0.1, 0.5, 5.0, uni.time_data.x);
    let ao = smoothstep(0.0, 0.1, get_scene_dist(p + normal * b));
    return base_color * (0.5 + 0.2 * fresnel) * (0.5 + 0.5 * ao);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = centered_uv(frag_coord.xy);
    var total = vec3<f32>(0.0);
    let samples = i32(lambda());

    for (var i = 0; i < samples; i = i + 1) {
        let fi = f32(i);
        let angle = 2.0 * PI * (fi / max(f32(samples), 1.0));
        let rd = vec2<f32>(cos(angle), sin(angle));
        var t = 0.0;
        let max_steps = i32(steps());
        var hit = false;

        for (var step = 0; step < max_steps; step = step + 1) {
            let d = get_scene_dist(uv + rd * t);
            if (d < 0.001) {
                hit = true;
                break;
            }
            t += max(d * 0.5, 0.01);
            if (t > 2.0) {
                break;
            }
        }

        if (hit) {
            total += calc_lighting(uv, rd, t);
        }
    }

    total /= max(f32(samples), 1.0);
    let d = get_scene_dist(uv);
    let zoom_level = osc_with_pause(-6.0, -33.0, 5.0, 33.0, uni.time_data.x);
    let glow = vec3<f32>(theta(), alpha(), sigma()) * exp(zoom_level * abs(d));
    total += glow * 0.5;

    let exposure = 1.5;
    let color = pow(total * exposure, vec3<f32>(1.0 / 2.2));
    let bg_color = vec3<f32>(0.1, 0.15, 0.2) + 0.05 * cos(length(uv) - uni.time_data.x * 0.3);
    var final_color = mix(bg_color, color, smoothstep(0.0, 0.01, total.r));
    final_color = apply_gamma(final_color, aa());

    return vec4<f32>(final_color, 1.0);
}
