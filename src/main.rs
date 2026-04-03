use nannou::prelude::*;
use realfft::RealFftPlanner;
use std::collections::VecDeque;

mod audio;
use audio::{AudioCapture, FFT_SIZE};

struct Model {
    audio: AudioCapture,
    fft_planner: RealFftPlanner<f32>,
    /// Rolling window of the most recent FFT_SIZE samples
    window: VecDeque<f32>,
    /// Magnitude spectrum in dBFS, FFT_SIZE/2+1 bins
    spectrum: Vec<f32>,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    app.new_window()
        .title("Icarus")
        .view(view)
        .build()
        .unwrap();

    let audio = AudioCapture::new().expect("failed to start audio capture");

    Model {
        audio,
        fft_planner: RealFftPlanner::new(),
        window: VecDeque::from(vec![0.0f32; FFT_SIZE]),
        spectrum: vec![0.0f32; FFT_SIZE / 2 + 1],
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    // drain ring buffer into the rolling window
    while let Some(s) = model.audio.consumer.pop() {
        model.window.pop_front();
        model.window.push_back(s);
    }

    // apply Hann window and copy into a contiguous Vec for FFT
    let r2c = model.fft_planner.plan_fft_forward(FFT_SIZE);
    let mut buf = r2c.make_input_vec();
    for (i, (dst, &src)) in buf.iter_mut().zip(model.window.iter()).enumerate() {
        let hann = 0.5 * (1.0 - (TAU * i as f32 / (FFT_SIZE - 1) as f32).cos());
        *dst = src * hann;
    }

    let mut complex_out = r2c.make_output_vec();
    r2c.process(&mut buf, &mut complex_out).unwrap();

    // convert to dBFS magnitude
    for (bin, c) in model.spectrum.iter_mut().zip(complex_out.iter()) {
        let mag = (c.re * c.re + c.im * c.im).sqrt() / FFT_SIZE as f32;
        *bin = 20.0 * mag.max(1e-10).log10();
    }

}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let win = app.window_rect();

    // Only show the lower half of bins — upper half is ultrasonic mirror noise
    let bins = &model.spectrum[..model.spectrum.len() / 2];
    let n = bins.len() as f32;
    let bar_w = win.w() / n;

    for (i, &db) in bins.iter().enumerate() {
        // map [-80, 0] dBFS → bar height; minimum 2px so bars are always visible
        let h = ((db + 80.0) / 80.0).clamp(0.0, 1.0) * win.h() * 0.9 + 2.0;
        let x = win.left() + i as f32 * bar_w;
        let hue = i as f32 / n;

        draw.rect()
            .x_y(x + bar_w * 0.5, win.bottom() + h * 0.5)
            .w_h(bar_w.max(1.0), h)
            .color(hsl(hue, 1.0, 0.5));
    }

    draw.to_frame(app, &frame).unwrap();
}
