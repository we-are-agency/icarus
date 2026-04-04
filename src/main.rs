use nannou::image::{DynamicImage, ImageBuffer, Rgba};
use nannou::prelude::*;
use icarus::analysis::{CQT_BINS, CQT_HISTORY, FFT_DISPLAY_BINS, FFT_HISTORY};
use icarus::audio::{AudioCapture, FFT_SIZE};
use icarus::midi_layer::MidiLayer;
use icarus::transcription::{InstrumentSelection, StreamingTranscriber, TRANSCRIPTION_HOP_SIZE};
use std::collections::VecDeque;

const MENU_BAR_H: f32 = 56.0;
const MENU_ITEM_GAP: f32 = 28.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisualLayer {
    Notes,
}

struct LayerMenuItem {
    instrument_selection: InstrumentSelection,
    layers: Vec<VisualLayer>,
    label: &'static str,
    active: bool,
    rms: f32,
}

struct Model {
    audio: AudioCapture,
    transcriber: StreamingTranscriber,
    pending_audio: VecDeque<f32>,
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
    midi: MidiLayer,
    layer_menu: Vec<LayerMenuItem>,
    selected_group: usize,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    app.new_window()
        .title("Icarus")
        .size(1280, 720)
        .key_pressed(key_pressed)
        .view(view)
        .build()
        .unwrap();

    let audio = AudioCapture::new().expect("failed to start audio capture");
    let sample_rate = audio.sample_rate;
    let pixels = vec![0u8; CQT_HISTORY * CQT_BINS * 4];
    let blank = DynamicImage::ImageRgba8(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
            CQT_HISTORY as u32,
            CQT_BINS as u32,
            pixels.clone(),
        )
        .unwrap(),
    );
    let spectrogram_texture = wgpu::Texture::from_image(app, &blank);

    let fft_pixels = vec![0u8; FFT_HISTORY * FFT_DISPLAY_BINS * 4];
    let fft_blank = DynamicImage::ImageRgba8(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
            FFT_HISTORY as u32,
            FFT_DISPLAY_BINS as u32,
            fft_pixels.clone(),
        )
        .unwrap(),
    );
    let fft_spectrogram_texture = wgpu::Texture::from_image(app, &fft_blank);

    Model {
        transcriber: StreamingTranscriber::new(sample_rate, 8),
        audio,
        pending_audio: VecDeque::with_capacity(FFT_SIZE * 4),
        spectrogram_pixels: pixels,
        spectrogram_texture,
        fft_spectrogram_pixels: fft_pixels,
        fft_spectrogram_texture,
        kick_flash: 0.0,
        snare_flash: 0.0,
        hihat_flash: 0.0,
        midi: MidiLayer::new(sample_rate),
        layer_menu: vec![
            LayerMenuItem {
                instrument_selection: InstrumentSelection::Percussive,
                layers: vec![VisualLayer::Notes],
                label: "Percussive",
                active: true,
                rms: 0.0,
            },
            LayerMenuItem {
                instrument_selection: InstrumentSelection::Harmonic,
                layers: vec![VisualLayer::Notes],
                label: "Harmonic",
                active: true,
                rms: 0.0,
            },
            LayerMenuItem {
                instrument_selection: InstrumentSelection::PercussiveHarmonic,
                layers: vec![VisualLayer::Notes],
                label: "PercussiveHarmonic",
                active: true,
                rms: 0.0,
            },
        ],
        selected_group: 0,
    }
}

fn key_pressed(_app: &App, model: &mut Model, key: Key) {
    if model.layer_menu.is_empty() {
        return;
    }

    match key {
        Key::Left => {
            model.selected_group =
                (model.selected_group + model.layer_menu.len() - 1) % model.layer_menu.len();
        }
        Key::Right => {
            model.selected_group = (model.selected_group + 1) % model.layer_menu.len();
        }
        Key::Space => {
            if let Some(item) = model.layer_menu.get_mut(model.selected_group) {
                item.active = !item.active;
            }
        }
        _ => {}
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    while let Some(s) = model.audio.consumer.pop() {
        model.pending_audio.push_back(s);
    }

    while model.pending_audio.len() >= TRANSCRIPTION_HOP_SIZE {
        let block: Vec<f32> = model
            .pending_audio
            .drain(..TRANSCRIPTION_HOP_SIZE)
            .collect();
        model.transcriber.process_block(&block);
    }

    let finished_notes = model.transcriber.drain_finished_notes();
    let active_notes = model.transcriber.active_notes();
    model.midi.update(
        &finished_notes,
        &active_notes,
        model.transcriber.elapsed_secs(),
    );
    for group in &mut model.layer_menu {
        group.rms = model.midi.group_rms(group.instrument_selection);
    }

    // Beat flash decay
    model.kick_flash *= 0.82;
    model.snare_flash *= 0.75;
    model.hihat_flash *= 0.70;
    let f = model.transcriber.features();
    if f.kick {
        model.kick_flash = (1.0f32).max(model.kick_flash);
    }
    if f.snare {
        model.snare_flash = (1.0f32).max(model.snare_flash);
    }
    if f.hihat {
        model.hihat_flash = (1.0f32).max(model.hihat_flash);
    }

    // Rebuild CQT spectrogram pixel buffer
    let pixels = &mut model.spectrogram_pixels;
    pixels.fill(0);
    let cqt_history = model.transcriber.cqt_history();
    let cqt_x_offset = CQT_HISTORY.saturating_sub(cqt_history.len());
    for (t, frame) in cqt_history.iter().enumerate() {
        let x = cqt_x_offset + t;
        for (b, &mag) in frame.iter().enumerate() {
            // Flip Y: bin 0 = low freq = bottom of image = high pixel row
            let y = CQT_BINS - 1 - b;
            let idx = (y * CQT_HISTORY + x) * 4;
            let [r, g, bl, a] = spectrogram_rgba(mag);
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = bl;
            pixels[idx + 3] = a;
        }
    }

    // Rebuild STFT spectrogram pixel buffer
    let fft_pixels = &mut model.fft_spectrogram_pixels;
    fft_pixels.fill(0);
    let fft_history = model.transcriber.fft_history();
    let fft_x_offset = FFT_HISTORY.saturating_sub(fft_history.len());
    for (t, frame) in fft_history.iter().enumerate() {
        let x = fft_x_offset + t;
        for (b, &mag) in frame.iter().enumerate() {
            // Flip Y: bin 0 = low freq = bottom = high pixel row
            let y = FFT_DISPLAY_BINS - 1 - b;
            let idx = (y * FFT_HISTORY + x) * 4;
            let [r, g, bl, a] = spectrogram_rgba(mag);
            fft_pixels[idx] = r;
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
    // Log scale: boost low magnitudes for visibility
    let mag = (mag * 9.0 + 1.0).log10(); // [0,1] → [0,1] with log curve
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
    draw_menu_layer(&draw, win, model);
    draw_analysis_scaffold(&draw, model, &frame, win);

    for group in &model.layer_menu {
        if !group.active {
            continue;
        }

        for layer in &group.layers {
            match layer {
                VisualLayer::Notes => {
                    draw_instrument_selection_notes(&draw, model, win, group.instrument_selection)
                }
            }
        }
    }

    draw.to_frame(app, &frame).unwrap();
}

fn draw_menu_layer(draw: &Draw, win: Rect, model: &Model) {
    let bar_y = win.top() - MENU_BAR_H * 0.5;
    draw.rect()
        .x_y(0.0, bar_y)
        .w_h(win.w(), MENU_BAR_H)
        .color(rgba(0.02, 0.025, 0.035, 0.96));

    let brand_x = win.left() + 74.0;
    draw.text("ICARUS")
        .x_y(brand_x, win.top() - 24.0)
        .font_size(9)
        .color(rgba(1.0, 1.0, 1.0, 0.82));

    let mut cursor_x = brand_x + 92.0;
    for (idx, item) in model.layer_menu.iter().enumerate() {
        let estimated_w = item.label.len() as f32 * 7.2 + 18.0;
        let item_x = cursor_x + estimated_w * 0.5;
        let is_selected = idx == model.selected_group;
        let text_alpha = if is_selected { 0.98 } else { 0.68 };
        let indicator_y = win.top() - MENU_BAR_H + 10.0;
        let indicator_h = 8.0;
        let charcoal = 0.18;
        let brightness = if item.active {
            charcoal + (1.0 - charcoal) * item.rms.clamp(0.0, 1.0)
        } else {
            charcoal
        };
        let indicator_alpha = if item.active { 0.95 } else { 0.65 };

        draw.text(item.label)
            .x_y(item_x, win.top() - 24.0)
            .font_size(9)
            .color(rgba(1.0, 1.0, 1.0, text_alpha));

        draw.rect()
            .x_y(item_x, indicator_y)
            .w_h(estimated_w, indicator_h)
            .color(rgba(brightness, brightness, brightness, indicator_alpha));

        if item.active && is_selected {
            draw.line()
                .start(pt2(item_x - estimated_w * 0.5, indicator_y - indicator_h * 0.5))
                .end(pt2(item_x + estimated_w * 0.5, indicator_y - indicator_h * 0.5))
                .color(rgba(1.0, 1.0, 1.0, 0.95))
                .weight(2.0);
        }

        cursor_x += estimated_w + MENU_ITEM_GAP;
    }
}

fn draw_analysis_scaffold(draw: &Draw, model: &Model, frame: &Frame, win: Rect) {
    let f = model.transcriber.features();
    let content_top = win.top() - MENU_BAR_H - 10.0;
    let spec_bottom = win.bottom() + win.h() * 0.18;
    let spec_h = (content_top - spec_bottom).max(80.0);

    // ── CQT spectrogram texture (pitch-aligned with the piano roll) ──────
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
    draw.texture(&model.spectrogram_texture)
        .x_y(0.0, spec_bottom + spec_h * 0.5)
        .w_h(win.w(), spec_h);

    // ── STFT spectrogram texture (maintained for optional alternate views) ──
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

    // ── Realtime transcription piano roll ────────────────────────────────
    {
        let midi = &model.midi;
        let history = midi.history_secs().max(1e-3);
        let now = midi.elapsed_secs();
        let t_offset = now - history;

        let panel_w = win.w();
        let panel_x0 = win.left();
        let panel_y0 = spec_bottom;
        let panel_y1 = spec_bottom + spec_h;
        let label_w = 48.0_f32;
        let note_x0 = panel_x0 + label_w;
        let note_w = panel_w - label_w;
        let note_min = midi.note_min();
        let note_max = midi.note_max();
        let note_span = (note_max - note_min + 1) as f32;
        let lane_h = (spec_h / note_span).max(2.0);

        for midi_note in note_min..=note_max {
            let y = midi_pitch_y(midi_note, note_min, note_max, panel_y0, panel_y1);
            let pitch_class = midi_note % 12;
            let is_black_key = matches!(pitch_class, 1 | 3 | 6 | 8 | 10);
            if is_black_key {
                draw.rect()
                    .x_y(note_x0 + note_w * 0.5, y)
                    .w_h(note_w, lane_h)
                    .color(rgba(0.02, 0.03, 0.04, 0.22));
            }

            let line_alpha = if pitch_class == 0 { 0.20 } else { 0.06 };
            draw.line()
                .start(pt2(note_x0, y))
                .end(pt2(note_x0 + note_w, y))
                .color(rgba(1.0, 1.0, 1.0, line_alpha))
                .weight(if pitch_class == 0 { 1.25 } else { 1.0 });

            if pitch_class == 0 {
                draw.text(&midi_pitch_label(midi_note))
                    .x_y(panel_x0 + label_w * 0.45, y + 4.0)
                    .font_size(9)
                    .color(rgba(1.0, 1.0, 1.0, 0.45));
            }
        }

        let mut grid_secs = (history / 6.0).max(0.25);
        grid_secs = (grid_secs * 4.0).round() / 4.0;
        if grid_secs <= 0.0 {
            grid_secs = 0.25;
        }
        let first_grid = (t_offset / grid_secs).floor() * grid_secs;
        let mut grid_t = first_grid;
        while grid_t <= now {
            let x = note_x0 + (grid_t - t_offset) / history * note_w;
            draw.line()
                .start(pt2(x, panel_y0))
                .end(pt2(x, panel_y1))
                .color(rgba(1.0, 1.0, 1.0, 0.06))
                .weight(1.0);
            grid_t += grid_secs;
        }

        draw.line()
            .start(pt2(note_x0 + note_w, panel_y0))
            .end(pt2(note_x0 + note_w, panel_y1))
            .color(rgba(1.0, 1.0, 1.0, 0.35))
            .weight(1.5);
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
    let ind_labels: [(&str, bool, f32, f32, f32); 3] = [
        ("KICK", f.kick, f.kick_strength, 0.07, model.kick_flash),
        ("SNARE", f.snare, f.snare_strength, 0.0, model.snare_flash),
        ("HIHAT", f.hihat, f.hihat_strength, 0.60, model.hihat_flash),
    ];
    let ind_x0 = win.right() - 90.0;
    for (j, &(label, beat, strength, hue, flash)) in ind_labels.iter().enumerate() {
        let iy = (content_top - 18.0) - j as f32 * 26.0;
        let lit = 0.18 + 0.57 * flash.max(if beat { 1.0 } else { 0.0 });
        draw.ellipse()
            .x_y(ind_x0, iy)
            .w_h(14.0, 14.0)
            .color(hsl(hue, 1.0, lit));
        draw.text(label)
            .x_y(ind_x0 + 32.0, iy)
            .font_size(9)
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
        ("BRIGHT", f.brightness, 0.15, "dark", "bright"),
        ("ROUGH", f.roughness, 0.05, "smooth", "rough"),
        ("SUSTAIN", f.sustain, 0.55, "percussive", "sustained"),
        ("WIDTH", f.width, 0.60, "narrow", "wide"),
        ("H/P", f.harmonic_ratio, 0.35, "percussive", "harmonic"),
        ("ONSET", f.onset_strength.clamp(0.0, 1.0), 0.0, "", ""),
        ("RMS", f.rms, 0.45, "", ""),
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
            .font_size(9)
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

}

fn draw_instrument_selection_notes(
    draw: &Draw,
    model: &Model,
    win: Rect,
    instrument_selection: InstrumentSelection,
) {
    let midi = &model.midi;
    let history = midi.history_secs().max(1e-3);
    let now = midi.elapsed_secs();
    let t_offset = now - history;
    let content_top = win.top() - MENU_BAR_H - 10.0;
    let spec_bottom = win.bottom() + win.h() * 0.18;
    let spec_h = (content_top - spec_bottom).max(80.0);
    let panel_x0 = win.left();
    let panel_y0 = spec_bottom;
    let panel_y1 = spec_bottom + spec_h;
    let label_w = 48.0_f32;
    let note_x0 = panel_x0 + label_w;
    let note_w = win.w() - label_w;
    let note_min = midi.note_min();
    let note_max = midi.note_max();
    let note_span = (note_max - note_min + 1) as f32;
    let lane_h = (spec_h / note_span).max(2.0);

    for note in midi.visible_notes_for_selection(Some(instrument_selection)) {
        let y = midi_pitch_y(note.midi_note, note_min, note_max, panel_y0, panel_y1);
        let x_start = (note_x0 + (note.start_secs - t_offset) / history * note_w)
            .clamp(note_x0, note_x0 + note_w);
        let x_end = (note_x0 + (note.end_secs - t_offset) / history * note_w)
            .clamp(note_x0, note_x0 + note_w);
        let block_w = (x_end - x_start).max(2.0);
        let alpha = 0.28 + 0.60 * note.confidence;
        let lit = if note.alive { 0.64 } else { 0.54 };

        draw.rect()
            .x_y(x_start + block_w * 0.5, y)
            .w_h(block_w, lane_h * 0.82)
            .color(hsla(note.hue, 0.88, lit, alpha));

        if note.alive {
            draw.rect()
                .x_y(x_end - 1.0, y)
                .w_h(2.0, lane_h.max(4.0))
                .color(rgba(1.0, 1.0, 1.0, 0.78));
        }
    }
}

fn midi_pitch_y(midi_note: u8, note_min: u8, note_max: u8, panel_y0: f32, panel_y1: f32) -> f32 {
    let note_span = (note_max - note_min + 1) as f32;
    let normalized = (midi_note.saturating_sub(note_min) as f32 + 0.5) / note_span.max(1.0);
    panel_y0 + normalized * (panel_y1 - panel_y0)
}

fn midi_pitch_label(midi_note: u8) -> String {
    let octave = midi_note as i32 / 12 - 1;
    format!("C{octave}")
}
