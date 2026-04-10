struct MandelbulbUniforms {
    stage_rect: vec4<f32>,
    time_data: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
    params4: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: MandelbulbUniforms;

fn panel_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    return (frag_coord - uni.stage_rect.xy) / uni.stage_rect.zw;
}

fn centered_uv(frag_coord: vec2<f32>) -> vec2<f32> {
    let uv = panel_uv(frag_coord);
    let aspect = uni.stage_rect.z / max(uni.stage_rect.w, 1.0);
    return vec2<f32>((uv.x - 0.5) * aspect, 0.5 - uv.y);
}

fn core_color() -> vec3<f32> {
    return uni.params0.xyz;
}

fn warm_color() -> vec3<f32> {
    return vec3<f32>(uni.params0.w, uni.params1.x, uni.params1.y);
}

fn cool_color() -> vec3<f32> {
    return vec3<f32>(uni.params1.z, uni.params1.w, uni.params2.x);
}

fn glow_color() -> vec3<f32> {
    return vec3<f32>(uni.params2.y, uni.params2.z, uni.params2.w);
}

fn gamma_value() -> f32 {
    return uni.params3.x;
}

fn iterations() -> i32 {
    return i32(uni.params3.y);
}

fn power_value() -> f32 {
    return 4.0;
}

struct RayHit {
    distance: f32,
    trap: f32,
    hit: f32,
}

fn mandelbulb_de(pos: vec3<f32>) -> vec2<f32> {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;
    var trap = 10.0;
    let power = power_value();
    for (var i = 0; i < 14; i = i + 1) {
        if (i >= iterations()) {
            break;
        }
        r = max(length(z), 0.0001);
        trap = min(trap, r);
        if (r > 2.2) {
            break;
        }
        let theta = acos(clamp(z.z / r, -1.0, 1.0));
        let phi = atan2(z.y, z.x);
        let power_minus_one = max(power - 1.0, 1.0);
        dr = pow(r, power_minus_one) * power * dr + 1.0;
        let zr = pow(r, power);
        let next_theta = theta * power;
        let next_phi = phi * power + uni.time_data.x * 0.18;
        z = zr * vec3<f32>(
            sin(next_theta) * cos(next_phi),
            sin(next_theta) * sin(next_phi),
            cos(next_theta),
        ) + pos;
    }
    let distance = 0.5 * log(max(r, 1.01)) * r / max(dr, 0.001);
    return vec2<f32>(distance, trap);
}

fn estimate_normal(pos: vec3<f32>) -> vec3<f32> {
    let e = 0.0015;
    let dx = mandelbulb_de(pos + vec3<f32>(e, 0.0, 0.0)).x - mandelbulb_de(pos - vec3<f32>(e, 0.0, 0.0)).x;
    let dy = mandelbulb_de(pos + vec3<f32>(0.0, e, 0.0)).x - mandelbulb_de(pos - vec3<f32>(0.0, e, 0.0)).x;
    let dz = mandelbulb_de(pos + vec3<f32>(0.0, 0.0, e)).x - mandelbulb_de(pos - vec3<f32>(0.0, 0.0, e)).x;
    return normalize(vec3<f32>(dx, dy, dz));
}

fn march(ro: vec3<f32>, rd: vec3<f32>) -> RayHit {
    var t = 0.0;
    var trap = 0.0;
    var hit = 0.0;
    for (var i = 0; i < 96; i = i + 1) {
        let pos = ro + rd * t;
        let de = mandelbulb_de(pos);
        trap = de.y;
        if (de.x < 0.0015) {
            hit = 1.0;
            break;
        }
        t += de.x;
        if (t > 12.0) {
            break;
        }
    }
    return RayHit(t, trap, hit);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = centered_uv(frag_coord.xy);
    let angle = uni.time_data.x * 0.28;
    let camera = vec3<f32>(sin(angle) * 4.8, 1.6 + 0.45 * sin(angle * 0.6), cos(angle) * 4.8);
    let focus = vec3<f32>(0.0, 0.2, 0.0);
    let forward = normalize(focus - camera);
    let right = normalize(cross(forward, vec3<f32>(0.0, 1.0, 0.0)));
    let up = normalize(cross(right, forward));
    let rd = normalize(forward + uv.x * right + uv.y * up);
    let hit = march(camera, rd);

    var color = mix(core_color() * 0.16, glow_color() * 0.9, 0.5 + 0.5 * rd.y);
    if (hit.hit > 0.5) {
        let pos = camera + rd * hit.distance;
        let normal = estimate_normal(pos);
        let light_a = normalize(vec3<f32>(1.2, 1.0, -0.5));
        let light_b = normalize(vec3<f32>(-0.8, 0.6, 1.0));
        let diff_a = max(dot(normal, light_a), 0.0);
        let diff_b = max(dot(normal, light_b), 0.0);
        let fresnel = pow(1.0 - max(dot(normal, -rd), 0.0), 4.0);
        let trap = smoothstep(0.0, 1.6, hit.trap);
        let body = mix(core_color(), warm_color(), trap);
        let rim = mix(warm_color(), cool_color(), 0.5 + 0.5 * normal.y);
        color = body * (0.18 + 0.95 * diff_a) + rim * (0.12 + 0.55 * diff_b);
        color += glow_color() * fresnel * 1.2;
        color += cool_color() * pow(max(1.0 - hit.distance / 12.0, 0.0), 2.2) * 0.25;
    }

    color = pow(clamp(color, vec3<f32>(0.0), vec3<f32>(2.5)), vec3<f32>(1.0 / max(gamma_value(), 0.001)));
    return vec4<f32>(color, 1.0);
}
