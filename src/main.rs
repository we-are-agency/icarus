use nannou::image::{DynamicImage, ImageBuffer, Rgba};
use nannou::prelude::*;
use std::collections::VecDeque;

mod analysis;
mod audio;
mod midi_layer;
mod objects;
use analysis::{Analyser, CQT_BINS, CQT_HISTORY, FFT_DISPLAY_BINS, FFT_HISTORY};
use audio::{AudioCapture, FFT_SIZE};
use midi_layer::MidiLayer;
use objects::SoundObjectDetector;

struct Model {
    audio: AudioCapture,
    analyser: Analyser,
    window: VecDeque<f32>,
    /// Raw RGBA pixel buffer updated every frame (CQT_HISTORY × CQT_BINS × 4)
    spectrogram_pixels: Vec<u8>,
    /// Persistent GPU texture — updated via write_texture each frame
    spectrogram_texture: wgpu::Texture,
    /// Raw RGBA pixel buffer for STFT spectrogram (FFT_HISTORY × FFT_DISPLAY_BINS × 4)
    fft_spectrogram_pixels: Vec<u8>,
    /// Persistent GPU texture for STFT spectrogram
    fft_spectrogram_texture: wgpu::Texture,
    /// Per-beat flash decay state (set to 1.0 on hit, decays each frame)
    kick_flash: f32,
    snare_flash: f32,
    hihat_flash: f32,
    objects: SoundObjectDetector,
    midi: MidiLayer,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    app.new_window()
        .title("Icarus")
        .size(1280, 720)
        .view(view)
        .build()
        .unwrap();

    let audio = AudioCapture::new().expect("failed to start audio capture");
    let analyser = Analyser::new(audio.sample_rate);
    let pixels = vec![0u8; CQT_HISTORY * CQT_BINS * 4];
    let blank = DynamicImage::ImageRgba8(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
            CQT_HISTORY as u32, CQT_BINS as u32, pixels.clone(),
        ).unwrap(),
    );
    let spectrogram_texture = wgpu::Texture::from_image(app, &blank);

    let fft_pixels = vec![0u8; FFT_HISTORY * FFT_DISPLAY_BINS * 4];
    let fft_blank = DynamicImage::ImageRgba8(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
            FFT_HISTORY as u32, FFT_DISPLAY_BINS as u32, fft_pixels.clone(),
        ).unwrap(),
    );
    let fft_spectrogram_texture = wgpu::Texture::from_image(app, &fft_blank);

    Model {
        audio,
        analyser,
        window: VecDeque::from(vec![0.0f32; FFT_SIZE]),
        spectrogram_pixels: pixels,
        spectrogram_texture,
        fft_spectrogram_pixels: fft_pixels,
        fft_spectrogram_texture,
        kick_flash: 0.0,
        snare_flash: 0.0,
        hihat_flash: 0.0,
        objects: SoundObjectDetector::new(),
        midi: MidiLayer::new(),
    }
}

fn update(_app: &App, model: &mut Model, update: Update) {
    while let Some(s) = model.audio.consumer.pop() {
        model.window.pop_front();
        model.window.push_back(s);
    }
    model.analyser.process(&model.window);
    let dt = update.since_last.as_secs_f32();
    model.objects.process(&model.analyser.features, dt);
    if model.objects.just_cleared {
        model.midi.clear_all();
    }
    model.midi.update(&model.objects.live_objects, dt);

    // Beat flash decay
    model.kick_flash  *= 0.82;
    model.snare_flash *= 0.75;
    model.hihat_flash *= 0.70;
    let f = &model.analyser.features;
    if f.kick  { model.kick_flash  = (1.0f32).max(model.kick_flash); }
    if f.snare { model.snare_flash = (1.0f32).max(model.snare_flash); }
    if f.hihat { model.hihat_flash = (1.0f32).max(model.hihat_flash); }

    // Rebuild CQT spectrogram pixel buffer
    let pixels = &mut model.spectrogram_pixels;
    for (t, frame) in model.analyser.cqt_history.iter().enumerate() {
        for (b, &mag) in frame.iter().enumerate() {
            // Flip Y: bin 0 = low freq = bottom of image = high pixel row
            let y = CQT_BINS - 1 - b;
            let idx = (y * CQT_HISTORY + t) * 4;
            let [r, g, bl, a] = spectrogram_rgba(mag);
            pixels[idx]     = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = bl;
            pixels[idx + 3] = a;
        }
    }

    // Rebuild STFT spectrogram pixel buffer
    let fft_pixels = &mut model.fft_spectrogram_pixels;
    for (t, frame) in model.analyser.fft_history.iter().enumerate() {
        for (b, &mag) in frame.iter().enumerate() {
            // Flip Y: bin 0 = low freq = bottom = high pixel row
            let y = FFT_DISPLAY_BINS - 1 - b;
            let idx = (y * FFT_HISTORY + t) * 4;
            let [r, g, bl, a] = spectrogram_rgba(mag);
            fft_pixels[idx]     = r;
            fft_pixels[idx + 1] = g;
            fft_pixels[idx + 2] = bl;
            fft_pixels[idx + 3] = a;
        }
    }
}

/// Inferno-ish colormap: black → dark blue → purple → orange → white
fn spectrogram_rgba(mag: f32) -> [u8; 4] {
    if mag < 0.015 {
        return [0, 0, 0, 255];
    }
    let hue = 0.75 - mag * 0.75;
    let lit = (mag * 0.6 + 0.05).clamp(0.0, 0.65);
    let [r, g, b] = hsl_to_rgb(hue, 1.0, lit);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let win = app.window_rect();
    let f = &model.analyser.features;

    let spec_h = win.h() * 0.78;
    let spec_bottom = win.bottom() + win.h() * 0.18;

    // ── CQT spectrogram texture (hidden, kept for reference) ─────────────
    frame.device_queue_pair().queue().write_texture(
        wgpu::ImageCopyTexture {
            texture: model.spectrogram_texture.inner(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &model.spectrogram_pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(CQT_HISTORY as u32 * 4),
            rows_per_image: Some(CQT_BINS as u32),
        },
        wgpu::Extent3d {
            width: CQT_HISTORY as u32,
            height: CQT_BINS as u32,
            depth_or_array_layers: 1,
        },
    );
    // draw.texture(&model.spectrogram_texture) — hidden

    // ── STFT spectrogram texture (deepest layer) ──────────────────────────
    frame.device_queue_pair().queue().write_texture(
        wgpu::ImageCopyTexture {
            texture: model.fft_spectrogram_texture.inner(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &model.fft_spectrogram_pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(FFT_HISTORY as u32 * 4),
            rows_per_image: Some(FFT_DISPLAY_BINS as u32),
        },
        wgpu::Extent3d {
            width: FFT_HISTORY as u32,
            height: FFT_DISPLAY_BINS as u32,
            depth_or_array_layers: 1,
        },
    );

    draw.texture(&model.fft_spectrogram_texture)
        .x_y(0.0, spec_bottom + spec_h * 0.5)
        .w_h(win.w(), spec_h);


    // ── MIDI track layer ─────────────────────────────────────────────────
    {
        let midi = &model.midi;
        let n_tracks = midi.num_tracks().max(1);
        let history = midi.history_secs();
        let now = midi.elapsed_secs;
        let t_offset = now - history;

        let panel_w = win.w();
        let panel_x0 = win.left();

        let label_w = 40.0_f32;
        let note_x0 = panel_x0 + label_w;
        let note_w  = panel_w - label_w;

        let track_h = spec_h / n_tracks as f32;

        // Thin horizontal separator between every lane
        for ti in 1..n_tracks {
            let ly = spec_bottom + (n_tracks - ti) as f32 * track_h;
            draw.line()
                .start(pt2(panel_x0, ly))
                .end(pt2(panel_x0 + panel_w, ly))
                .color(rgba(1.0_f32, 1.0, 1.0, 0.06))
                .weight(1.0);
        }

        // Group separator: thicker line + label between kind groups
        let mut prev_group: Option<u8> = None;
        const KIND_LABELS: [&str; 3] = ["PERC", "PH", "HARM"];
        for (ti, track) in midi.tracks.iter().enumerate() {
            let g = match track.kind {
                objects::SoundKind::Percussive         => 0u8,
                objects::SoundKind::PercussiveHarmonic => 1,
                objects::SoundKind::Harmonic           => 2,
            };
            if Some(g) != prev_group {
                let ly = spec_bottom + (n_tracks - ti) as f32 * track_h;
                draw.line()
                    .start(pt2(panel_x0, ly))
                    .end(pt2(panel_x0 + panel_w, ly))
                    .color(rgba(1.0_f32, 1.0, 1.0, 0.28))
                    .weight(1.5);
                draw.text(KIND_LABELS[g as usize])
                    .x_y(panel_x0 + label_w + 6.0, ly + 5.0)
                    .font_size(7)
                    .color(rgba(1.0_f32, 1.0, 1.0, 0.35));
                prev_group = Some(g);
            }
        }

        // Notes — all rendered as rectangles proportional to actual duration
        for dn in midi.visible_notes() {
            let ti = dn.track_idx;
            if ti >= n_tracks { continue; }
            let lane_cy = spec_bottom + (n_tracks - 1 - ti) as f32 * track_h + track_h * 0.5;
            let note_bar_h = (track_h * 0.65).max(3.0);
            let x_start = (note_x0 + (dn.start_secs - t_offset) / history * note_w).clamp(note_x0, note_x0 + note_w);
            let x_end   = (note_x0 + (dn.end_secs   - t_offset) / history * note_w).clamp(note_x0, note_x0 + note_w);
            let block_w = (x_end - x_start).max(1.0);
            let lit   = if dn.kind == objects::SoundKind::Percussive { 0.72 } else { 0.58 };
            let alpha = 0.88_f32;
            draw.rect()
                .x_y(x_start + block_w * 0.5, lane_cy)
                .w_h(block_w, note_bar_h)
                .color(hsla(dn.hue, 1.0, lit, alpha));
            if dn.alive {
                draw.rect()
                    .x_y(x_end - 1.0, lane_cy)
                    .w_h(2.0, note_bar_h)
                    .color(rgba(1.0_f32, 1.0, 1.0, 0.75));
            }
        }

        // Track labels: color dot + cluster index
        for (ti, track) in midi.tracks.iter().enumerate() {
            let lane_cy = spec_bottom + (n_tracks - 1 - ti) as f32 * track_h + track_h * 0.5;
            draw.ellipse()
                .x_y(panel_x0 + 8.0, lane_cy)
                .w_h(7.0, 7.0)
                .color(hsl(track.hue, 1.0, 0.60));
            let id_str = format!("{}", track.cluster_id);
            draw.text(&id_str)
                .x_y(panel_x0 + 26.0, lane_cy)
                .font_size(8)
                .color(rgba(1.0_f32, 1.0, 1.0, 0.60));
        }
    }

    // ── 6 band energy bars ────────────────────────────────────────────────
    // Hues from warm (sub-bass) to cool (highs)
    const BAND_HUES: [f32; 6] = [0.95, 0.07, 0.14, 0.35, 0.52, 0.68];
    const BAND_NAMES: [&str; 6] = ["SUB", "BASS", "LO-M", "MID", "HI-M", "HIGH"];

    let total_bar_w = win.w() * 0.72;
    let bar_slot = total_bar_w / 6.0;
    let bar_w = bar_slot * 0.72;

    for i in 0..6 {
        let energy = f.band_energy[i]; // already [0, 1] via per-band auto-norm
        let h_norm = (energy * 9.0 + 1.0).log10(); // log lift, 0..1
        let bar_h = h_norm * spec_h;

        let cx = -total_bar_w * 0.5 + bar_slot * (i as f32 + 0.5);
        let cy = spec_bottom + bar_h * 0.5;

        let hue = BAND_HUES[i];
        let beat = f.band_beat[i];
        let lit = if beat { 0.75 } else { 0.40 };
        let alpha = if beat { 1.0 } else { 0.80 };

        // Bar fill
        draw.rect()
            .x_y(cx, cy)
            .w_h(bar_w, bar_h.max(2.0))
            .color(hsla(hue, 1.0, lit, alpha));

        // Beat strength label (only when hitting)
        let strength = f.band_strength[i];
        if strength > 0.05 {
            draw.text(&format!("{:.1}x", strength + 1.0))
                .x_y(cx, spec_bottom + bar_h + 12.0)
                .font_size(9)
                .color(rgba(1.0_f32, 1.0, 1.0, 0.7));
        }

        // Band name below x-axis
        draw.text(BAND_NAMES[i])
            .x_y(cx, spec_bottom - 10.0)
            .font_size(9)
            .color(rgba(1.0_f32, 1.0, 1.0, 0.4));
    }

    // ── Kick / Snare / Hi-hat composite indicators (top right) ───────────
    let ind_labels: [(&str, bool, f32, f32); 3] = [
        ("KICK",  f.kick,  f.kick_strength,  0.07),
        ("SNARE", f.snare, f.snare_strength, 0.0),
        ("HIHAT", f.hihat, f.hihat_strength, 0.60),
    ];
    let ind_x0 = win.right() - 90.0;
    let ind_y0 = win.top()   - 28.0;
    for (j, &(label, beat, strength, hue)) in ind_labels.iter().enumerate() {
        let iy = ind_y0 - j as f32 * 26.0;
        let lit = if beat { 0.75 } else { 0.20 };
        draw.ellipse()
            .x_y(ind_x0, iy)
            .w_h(14.0, 14.0)
            .color(hsl(hue, 1.0, lit));
        draw.text(label)
            .x_y(ind_x0 + 32.0, iy)
            .font_size(10)
            .color(rgba(1.0_f32, 1.0, 1.0, if beat { 1.0 } else { 0.4 }));
        if strength > 0.0 {
            draw.text(&format!("{:.2}", strength))
                .x_y(ind_x0 + 70.0, iy)
                .font_size(9)
                .color(rgba(1.0_f32, 1.0, 1.0, 0.6));
        }
    }

    // ── Feature label strip (bottom) ─────────────────────────────────────
    let strip_h = win.h() * 0.14;
    let strip_y = win.bottom() + strip_h * 0.5;

    let labels: &[(&str, f32, f32, &str, &str)] = &[
        ("BRIGHT",  f.brightness,                      0.15, "dark",       "bright"),
        ("ROUGH",   f.roughness,                        0.05, "smooth",     "rough"),
        ("SUSTAIN", f.sustain,                          0.55, "percussive", "sustained"),
        ("WIDTH",   f.width,                            0.60, "narrow",     "wide"),
        ("H/P",     f.harmonic_ratio,                   0.35, "percussive", "harmonic"),
        ("ONSET",   f.onset_strength.clamp(0.0, 1.0),  0.0,  "",           ""),
        ("RMS",     f.rms,                              0.45, "",           ""),
    ];

    let col_w = win.w() / labels.len() as f32;
    let bar_y = strip_y - strip_h * 0.10;
    let name_y = strip_y + strip_h * 0.28;
    let lo_hi_y = strip_y - strip_h * 0.38;

    for (i, &(name, val, hue, lo, hi)) in labels.iter().enumerate() {
        let cx = win.left() + col_w * (i as f32 + 0.5);
        let bar_max_w = col_w * 0.82;

        let fill_w = bar_max_w * val;
        draw.rect()
            .x_y(cx - bar_max_w * 0.5 + fill_w * 0.5, bar_y)
            .w_h(fill_w, 10.0)
            .color(hsl(hue, 0.9, 0.55));

        draw.line()
            .start(pt2(cx - bar_max_w * 0.5, bar_y - 5.0))
            .end(pt2(cx + bar_max_w * 0.5, bar_y - 5.0))
            .color(rgba(1.0_f32, 1.0, 1.0, 0.25))
            .weight(1.0);

        draw.text(name)
            .x_y(cx, name_y)
            .font_size(10)
            .color(rgba(1.0_f32, 1.0, 1.0, 0.7));

        if !lo.is_empty() {
            draw.text(lo)
                .x_y(cx - bar_max_w * 0.5 + 10.0, lo_hi_y)
                .font_size(8)
                .color(rgba(1.0_f32, 1.0, 1.0, 0.35));
        }
        if !hi.is_empty() {
            draw.text(hi)
                .x_y(cx + bar_max_w * 0.5 - 10.0, lo_hi_y)
                .font_size(8)
                .color(rgba(1.0_f32, 1.0, 1.0, 0.35));
        }
    }

    draw.to_frame(app, &frame).unwrap();
}
