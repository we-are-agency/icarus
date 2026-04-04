use crate::audio::FFT_SIZE;
use realfft::RealFftPlanner;
use std::collections::VecDeque;
use std::f32::consts::TAU;

const NUM_BINS: usize = FFT_SIZE / 2 + 1;
const HP_HISTORY: usize = 16;
const ENERGY_HISTORY: usize = 30;
const SMOOTH: f32 = 0.15;
const ONSET_HISTORY: usize = 12;
const HARMONIC_HISTORY: usize = 12;
const CHROMA_FLUX_HISTORY: usize = 12;
const DETECTOR_WARMUP_SECS: f32 = 0.18;
const BAND_MIN_COOLDOWN_SECS: f32 = 0.12;
const CHROMA_UNIFORM_L2: f32 = 0.288_675_13;
pub const ONSET_MULTIPLIER: f32 = 2.5;

/// CQT: 84 bins = 7 octaves × 12 semitones, A0 (27.5 Hz) → G#7 (~3322 Hz)
pub const CQT_BINS: usize = 84;
pub const CQT_BASE_MIDI_NOTE: u8 = 21;
const CQT_F_MIN: f32 = 27.5;
const CQT_BINS_PER_OCT: f32 = 12.0;
pub const CQT_HISTORY: usize = 256; // power-of-2, row stride = 256×4 = 1024 B (wgpu alignment ✓)

pub fn cqt_bin_center_hz(bin: usize) -> Option<f32> {
    (bin < CQT_BINS).then(|| CQT_F_MIN * 2.0f32.powf(bin as f32 / CQT_BINS_PER_OCT))
}

pub fn cqt_bin_to_midi_note(bin: usize) -> Option<u8> {
    (bin < CQT_BINS).then(|| CQT_BASE_MIDI_NOTE + bin as u8)
}

pub fn midi_note_to_cqt_bin(midi_note: u8) -> Option<usize> {
    let bin = midi_note.checked_sub(CQT_BASE_MIDI_NOTE)? as usize;
    (bin < CQT_BINS).then_some(bin)
}

#[allow(dead_code)]
pub fn midi_note_to_hz(midi_note: u8) -> f32 {
    440.0 * 2.0f32.powf((midi_note as f32 - 69.0) / 12.0)
}

/// Linear (STFT) spectrogram: show lower 512 bins (up to Nyquist/2), 256 frames wide
pub const FFT_DISPLAY_BINS: usize = 512; // row stride = 256×4 = 1024 B (wgpu alignment ✓)
pub const FFT_HISTORY: usize = 256;

/// Beat detection: 6 frequency bands (Hz_lo, Hz_hi, sensitivity)
/// High bands use peak-bin energy (not average) to avoid dilution across many bins
const BAND_FREQS: [(f32, f32, f32); 6] = [
    (20.0, 60.0, 1.3),      // sub-bass  (kick body)     — average energy
    (60.0, 250.0, 1.4),     // bass      (kick punch)    — average energy
    (250.0, 500.0, 1.4),    // low-mid   (snare body)    — average energy
    (500.0, 2000.0, 1.4),   // mid       (snare snap)    — average energy
    (2000.0, 6000.0, 1.2),  // high-mid  (hi-hat)        — peak energy
    (6000.0, 20000.0, 1.2), // high      (transients)    — peak energy
];
/// Bands at this index and above use peak-bin energy instead of band average
const PEAK_BAND_START: usize = 4;
const BAND_HISTORY: usize = 43; // ~1 s of history at ~43 fps
const BAND_DECAY: f32 = 0.93; // smoothed energy release rate

/// All extracted audio properties for one frame.
pub struct AudioFeatures {
    /// dBFS magnitude spectrum, NUM_BINS elements (linear FFT)
    pub spectrum_db: Vec<f32>,
    /// CQT magnitudes [0, 1], CQT_BINS elements (log-frequency)
    pub cqt: Vec<f32>,

    /// Overall RMS level [0, 1]
    pub rms: f32,

    // ── Onset ────────────────────────────────────────────────────────────
    /// Half-wave rectified spectral flux, normalised by adaptive threshold [0, 1+]
    pub onset_strength: f32,
    /// Short-window confidence that this frame contains a meaningful onset [0, 1]
    pub onset_confidence: f32,
    pub is_onset: bool,
    /// True once warmup and short feature histories are ready for object detection
    pub analysis_ready: bool,

    // ── Harmonic / percussive ─────────────────────────────────────────────
    /// 1 = fully harmonic, 0 = fully percussive
    pub harmonic_ratio: f32,
    /// Normalised harmonic energy [0, 1]
    pub harmonic_energy: f32,
    /// Normalised percussive energy [0, 1]
    pub percussive_energy: f32,
    /// Confidence that the current frame belongs to stable pitched content [0, 1]
    pub harmonic_confidence: f32,
    /// Confidence that the current frame belongs to a percussive event [0, 1]
    pub percussive_confidence: f32,

    // ── Texture labels (all [0, 1]) ───────────────────────────────────────
    /// Spectral centroid: 0 = dark (low-freq), 1 = bright (high-freq)
    pub brightness: f32,
    /// Spectral irregularity: 0 = smooth, 1 = rough
    pub roughness: f32,
    /// Energy stability: 0 = percussive/transient, 1 = sustained
    pub sustain: f32,
    /// Spectral bandwidth: 0 = narrow, 1 = wide
    pub width: f32,
    /// Pitch concentration and temporal consistency of the current chroma [0, 1]
    pub pitched_stability: f32,

    // ── Beat detection ────────────────────────────────────────────────────
    /// Smoothed per-band energy for visualization (sub-bass → high)
    pub band_energy: [f32; 6],
    /// Per-band beat trigger this frame
    pub band_beat: [bool; 6],
    /// Normalised beat strength per band (0 = no beat, >0 = beat strength)
    pub band_strength: [f32; 6],
    /// Convenience: kick (bands 0+1), snare (2+3), hi-hat (4+5)
    pub kick: bool,
    pub snare: bool,
    pub hihat: bool,
    pub kick_strength: f32,
    pub snare_strength: f32,
    pub hihat_strength: f32,

    // ── Sound object detection ────────────────────────────────────────────
    /// Chromagram: 12 pitch classes, max over 7 octaves of CQT, L1-normalized
    pub chroma: [f32; 12],
    /// Per-pitch-class energy increase since last frame (half-wave rectified)
    pub chroma_flux: [f32; 12],
    /// Per-band spectral flux at onset moment, L1-normalized (0 when no onset activity)
    pub band_flux_delta: [f32; 6],
    /// How concentrated the onset flux is into one or a few spectral bands [0, 1]
    pub band_flux_peakiness: f32,
}

impl Default for AudioFeatures {
    fn default() -> Self {
        Self {
            spectrum_db: vec![-80.0; NUM_BINS],
            cqt: vec![0.0; CQT_BINS],
            rms: 0.0,
            onset_strength: 0.0,
            onset_confidence: 0.0,
            is_onset: false,
            analysis_ready: false,
            harmonic_ratio: 0.5,
            harmonic_energy: 0.0,
            percussive_energy: 0.0,
            harmonic_confidence: 0.0,
            percussive_confidence: 0.0,
            brightness: 0.5,
            roughness: 0.0,
            sustain: 1.0,
            width: 0.5,
            pitched_stability: 0.0,
            band_energy: [0.0; 6],
            band_beat: [false; 6],
            band_strength: [0.0; 6],
            kick: false,
            snare: false,
            hihat: false,
            kick_strength: 0.0,
            snare_strength: 0.0,
            hihat_strength: 0.0,
            chroma: [0.0; 12],
            chroma_flux: [0.0; 12],
            band_flux_delta: [0.0; 6],
            band_flux_peakiness: 0.0,
        }
    }
}

pub struct Analyser {
    fft_planner: RealFftPlanner<f32>,
    sample_rate: u32,
    prev_magnitudes: Vec<f32>,
    /// Short history of linear magnitude spectra for H/P variance
    mag_history: VecDeque<Vec<f32>>,
    /// EMA of spectral flux — adaptive onset baseline
    flux_ema: f32,
    onset_history: VecDeque<f32>,
    harmonic_ratio_history: VecDeque<f32>,
    chroma_flux_history: VecDeque<f32>,
    /// Short history of frame energy for sustain computation
    energy_history: VecDeque<f32>,
    /// Scrolling CQT history for spectrogram rendering
    pub cqt_history: VecDeque<Vec<f32>>,
    /// Scrolling linear FFT history for STFT spectrogram rendering
    pub fft_history: VecDeque<Vec<f32>>,
    /// Per-band energy history for adaptive beat threshold
    band_energy_history: Vec<VecDeque<f32>>,
    /// Per-band cooldown in seconds before the next trigger can fire
    band_cooldown_secs: [f32; 6],
    /// Per-band smoothed energy (instant attack, exponential release)
    band_smoothed: [f32; 6],
    /// Per-band slow-decaying peak — used to auto-normalise display to [0, 1]
    band_display_max: [f32; 6],
    /// Previous frame's chromagram — for computing chroma_flux
    prev_chroma: [f32; 12],
    elapsed_secs: f32,
    pub features: AudioFeatures,
}

impl Analyser {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            fft_planner: RealFftPlanner::new(),
            sample_rate,
            prev_magnitudes: vec![0.0; NUM_BINS],
            mag_history: VecDeque::with_capacity(HP_HISTORY + 1),
            flux_ema: 0.0,
            onset_history: VecDeque::with_capacity(ONSET_HISTORY + 1),
            harmonic_ratio_history: VecDeque::with_capacity(HARMONIC_HISTORY + 1),
            chroma_flux_history: VecDeque::with_capacity(CHROMA_FLUX_HISTORY + 1),
            energy_history: VecDeque::with_capacity(ENERGY_HISTORY + 1),
            cqt_history: VecDeque::with_capacity(CQT_HISTORY + 1),
            fft_history: VecDeque::with_capacity(FFT_HISTORY + 1),
            band_energy_history: (0..6)
                .map(|_| VecDeque::with_capacity(BAND_HISTORY + 1))
                .collect(),
            band_cooldown_secs: [0.0; 6],
            band_smoothed: [0.0; 6],
            band_display_max: [1e-10; 6],
            prev_chroma: [0.0; 12],
            elapsed_secs: 0.0,
            features: AudioFeatures::default(),
        }
    }

    /// Process one frame. Call once per nannou `update()` tick after draining the ring buffer.
    pub fn process(&mut self, window: &VecDeque<f32>, dt: f32) {
        self.elapsed_secs += dt.max(0.0);
        // ── FFT ───────────────────────────────────────────────────────────
        let r2c = self.fft_planner.plan_fft_forward(FFT_SIZE);
        let mut buf = r2c.make_input_vec();
        for (i, (dst, &src)) in buf.iter_mut().zip(window.iter()).enumerate() {
            let hann = 0.5 * (1.0 - (TAU * i as f32 / (FFT_SIZE - 1) as f32).cos());
            *dst = src * hann;
        }
        let mut cx = r2c.make_output_vec();
        r2c.process(&mut buf, &mut cx).unwrap();

        // Linear magnitudes normalised so a full-scale sine ≈ 1.0
        let scale = 2.0 / FFT_SIZE as f32;
        let mags: Vec<f32> = cx
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt() * scale)
            .collect();

        // ── dBFS spectrum ─────────────────────────────────────────────────
        for (db, &m) in self.features.spectrum_db.iter_mut().zip(mags.iter()) {
            *db = 20.0 * m.max(1e-10).log10();
        }

        // ── RMS ───────────────────────────────────────────────────────────
        let rms = (window.iter().map(|&s| s * s).sum::<f32>() / FFT_SIZE as f32).sqrt();
        self.features.rms = rms.clamp(0.0, 1.0);

        // ── Onset detection (half-wave rectified spectral flux) ───────────
        let flux: f32 = mags
            .iter()
            .zip(self.prev_magnitudes.iter())
            .map(|(&m, &p)| (m - p).max(0.0))
            .sum::<f32>()
            / NUM_BINS as f32;

        self.flux_ema = SMOOTH * flux + (1.0 - SMOOTH) * self.flux_ema;
        let threshold = self.flux_ema * ONSET_MULTIPLIER;
        self.features.onset_strength = (flux / (threshold + 1e-6)).clamp(0.0, 2.0);
        self.onset_history.push_back(self.features.onset_strength);
        trim_history(&mut self.onset_history, ONSET_HISTORY);
        let onset_baseline = median(&self.onset_history).max(1.0);
        let raw_onset_conf = ((self.features.onset_strength - 1.0) / 0.8).clamp(0.0, 1.0);
        let relative_onset_conf =
            ((self.features.onset_strength - onset_baseline) / onset_baseline).clamp(0.0, 1.0);
        self.features.onset_confidence =
            (0.65 * raw_onset_conf + 0.35 * relative_onset_conf).clamp(0.0, 1.0);
        self.features.is_onset = flux > threshold && flux > 1e-4;

        // ── Harmonic / percussive split (temporal variance per bin) ───────
        self.mag_history.push_back(mags.clone());
        if self.mag_history.len() > HP_HISTORY {
            self.mag_history.pop_front();
        }

        if self.mag_history.len() >= 4 {
            let n = self.mag_history.len() as f32;
            let mut h_e = 0.0f32;
            let mut p_e = 0.0f32;

            for k in 0..NUM_BINS {
                let mean = self.mag_history.iter().map(|f| f[k]).sum::<f32>() / n;
                let var = self
                    .mag_history
                    .iter()
                    .map(|f| {
                        let d = f[k] - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / n;

                // High temporal variance → percussive; low → harmonic
                let p_w = (var * 200.0).clamp(0.0, 1.0);
                let cur = mags[k] * mags[k];
                h_e += cur * (1.0 - p_w);
                p_e += cur * p_w;
            }

            let total = h_e + p_e + 1e-10;
            let hr = h_e / total;
            self.features.harmonic_ratio = ema(hr, self.features.harmonic_ratio);
            self.features.harmonic_energy = (h_e / NUM_BINS as f32).clamp(0.0, 1.0);
            self.features.percussive_energy = (p_e / NUM_BINS as f32).clamp(0.0, 1.0);
        }
        self.harmonic_ratio_history
            .push_back(self.features.harmonic_ratio);
        trim_history(&mut self.harmonic_ratio_history, HARMONIC_HISTORY);

        let total_mag: f32 = mags.iter().sum();

        // ── Brightness (spectral centroid) ────────────────────────────────
        let centroid = if total_mag > 1e-6 {
            mags.iter()
                .enumerate()
                .map(|(k, &m)| k as f32 * m)
                .sum::<f32>()
                / total_mag
        } else {
            NUM_BINS as f32 / 2.0
        };
        self.features.brightness = ema(
            (centroid / NUM_BINS as f32).clamp(0.0, 1.0),
            self.features.brightness,
        );

        // ── Roughness (spectral irregularity) ────────────────────────────
        let irreg = mags.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (total_mag + 1e-10);
        self.features.roughness = ema((irreg * 0.5).clamp(0.0, 1.0), self.features.roughness);

        // ── Sustain (energy stability over time) ─────────────────────────
        let energy = total_mag * total_mag / NUM_BINS as f32;
        self.energy_history.push_back(energy);
        if self.energy_history.len() > ENERGY_HISTORY {
            self.energy_history.pop_front();
        }
        if self.energy_history.len() >= 4 {
            let mean_e = self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32;
            let var_e = self
                .energy_history
                .iter()
                .map(|&e| {
                    let d = e - mean_e;
                    d * d
                })
                .sum::<f32>()
                / self.energy_history.len() as f32;
            let cv = (var_e.sqrt() / (mean_e + 1e-6)).clamp(0.0, 1.0);
            self.features.sustain = ema(1.0 - cv, self.features.sustain);
        }

        // ── Width (spectral bandwidth) ────────────────────────────────────
        let bw = if total_mag > 1e-6 {
            mags.iter()
                .enumerate()
                .map(|(k, &m)| {
                    let d = k as f32 - centroid;
                    d * d * m
                })
                .sum::<f32>()
                / total_mag
        } else {
            (NUM_BINS as f32 / 4.0).powi(2)
        };
        self.features.width = ema(
            (bw.sqrt() / NUM_BINS as f32).clamp(0.0, 1.0),
            self.features.width,
        );

        // ── CQT (log-frequency mapping of FFT magnitudes) ─────────────────
        let bin_hz = self.sample_rate as f32 / FFT_SIZE as f32;
        for b in 0..CQT_BINS {
            let f_center = cqt_bin_center_hz(b).expect("CQT bin should have a center frequency");
            let f_lo = f_center * 2.0f32.powf(-0.5 / CQT_BINS_PER_OCT);
            let f_hi = f_center * 2.0f32.powf(0.5 / CQT_BINS_PER_OCT);
            let k_center = f_center / bin_hz;
            let k_lo_f = f_lo / bin_hz;
            let k_hi_f = f_hi / bin_hz;
            let k_lo = (k_lo_f as usize).clamp(0, NUM_BINS - 1);
            let k_hi = (k_hi_f as usize).clamp(k_lo, NUM_BINS - 1);

            let val = if k_lo == k_hi {
                // CQT bin narrower than one FFT bin — linearly interpolate between neighbors
                let k1 = (k_lo + 1).min(NUM_BINS - 1);
                let frac = k_center - k_lo as f32;
                mags[k_lo] * (1.0 - frac) + mags[k1] * frac
            } else {
                // CQT bin spans multiple FFT bins — triangular weighted average
                let half_bw = (k_hi_f - k_lo_f) * 0.5;
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                for k in k_lo..=k_hi {
                    let w = (1.0 - ((k as f32 - k_center) / half_bw).abs()).max(0.0);
                    sum += mags[k] * w;
                    weight_sum += w;
                }
                if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    0.0
                }
            };

            // smooth & store normalised [0,1]
            self.features.cqt[b] = ema((val * 8.0).clamp(0.0, 1.0), self.features.cqt[b]);
        }

        // ── Spectrogram history ───────────────────────────────────────────
        self.cqt_history.push_back(self.features.cqt.clone());
        if self.cqt_history.len() > CQT_HISTORY {
            self.cqt_history.pop_front();
        }

        // ── Chromagram: fold 84 CQT bins into 12 pitch classes ───────────
        // Uses max over 7 octaves so a loud low C doesn't drown a quiet high C.
        // L1-normalized so only pitch proportions matter, not loudness.
        {
            let mut chroma = [0.0f32; 12];
            for octave in 0..7usize {
                for c in 0..12usize {
                    let bin = octave * 12 + c;
                    if bin < CQT_BINS {
                        chroma[c] = chroma[c].max(self.features.cqt[bin]);
                    }
                }
            }
            let chroma_sum: f32 = chroma.iter().sum();
            if chroma_sum > 1e-8 {
                for v in chroma.iter_mut() {
                    *v /= chroma_sum;
                }
            }

            // Chroma flux: half-wave-rectified increase per pitch class.
            // Near-zero for sustaining sounds; positive for newly entering pitch classes.
            for c in 0..12 {
                self.features.chroma_flux[c] = (chroma[c] - self.prev_chroma[c]).max(0.0);
            }
            let chroma_flux_sum: f32 = self.features.chroma_flux.iter().sum();
            self.chroma_flux_history.push_back(chroma_flux_sum);
            trim_history(&mut self.chroma_flux_history, CHROMA_FLUX_HISTORY);

            let chroma_l2 = chroma.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let pitch_concentration =
                ((chroma_l2 - CHROMA_UNIFORM_L2) / (1.0 - CHROMA_UNIFORM_L2)).clamp(0.0, 1.0);
            let chroma_similarity = cosine_sim_12(&chroma, &self.prev_chroma);
            let stable_pitch = pitch_concentration
                * (0.35 + 0.65 * chroma_similarity)
                * (0.25 + 0.75 * self.features.sustain);
            self.features.pitched_stability = ema(
                stable_pitch.clamp(0.0, 1.0),
                self.features.pitched_stability,
            );

            self.prev_chroma = chroma;
            self.features.chroma = chroma;
        }

        // ── Band flux delta: per-band spectral flux at this frame ─────────
        // Computed against prev_magnitudes BEFORE it is updated at end of process().
        // L1-normalized to represent timbral shape of the attack, not loudness.
        {
            let mut band_flux = [0.0f32; 6];
            for (i, &(f_lo, f_hi, _)) in BAND_FREQS.iter().enumerate() {
                let k_lo = ((f_lo / bin_hz) as usize).clamp(0, NUM_BINS - 1);
                let k_hi = ((f_hi / bin_hz) as usize).clamp(k_lo, NUM_BINS - 1);
                let n = (k_hi - k_lo + 1) as f32;
                band_flux[i] = mags[k_lo..=k_hi]
                    .iter()
                    .zip(self.prev_magnitudes[k_lo..=k_hi].iter())
                    .map(|(&m, &p)| (m - p).max(0.0))
                    .sum::<f32>()
                    / n;
            }
            let flux_sum: f32 = band_flux.iter().sum();
            if flux_sum > 1e-8 {
                for v in band_flux.iter_mut() {
                    *v /= flux_sum;
                }
            }
            self.features.band_flux_delta = band_flux;
            let peak = self
                .features
                .band_flux_delta
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            self.features.band_flux_peakiness =
                ((peak - (1.0 / 6.0)) / (1.0 - (1.0 / 6.0))).clamp(0.0, 1.0);
        }

        let harmonic_ratio_avg = mean(self.harmonic_ratio_history.iter().copied());
        self.features.harmonic_confidence =
            (0.6 * harmonic_ratio_avg + 0.4 * self.features.pitched_stability).clamp(0.0, 1.0);
        self.features.percussive_confidence = (0.55 * (1.0 - harmonic_ratio_avg)
            + 0.25 * self.features.onset_confidence
            + 0.20 * self.features.band_flux_peakiness)
            .clamp(0.0, 1.0);
        self.features.analysis_ready = self.elapsed_secs >= DETECTOR_WARMUP_SECS
            && self.onset_history.len() >= 4
            && self.harmonic_ratio_history.len() >= 4
            && self.chroma_flux_history.len() >= 4;

        // ── Beat detection (multi-band adaptive threshold) ────────────────
        {
            let bin_hz = self.sample_rate as f32 / FFT_SIZE as f32;
            for (i, &(f_lo, f_hi, sensitivity)) in BAND_FREQS.iter().enumerate() {
                let k_lo = ((f_lo / bin_hz) as usize).clamp(0, NUM_BINS - 1);
                let k_hi = ((f_hi / bin_hz) as usize).clamp(k_lo, NUM_BINS - 1);
                let energy = if i >= PEAK_BAND_START {
                    // High bands: use peak bin to avoid diluting transients over many bins
                    let peak = mags[k_lo..=k_hi].iter().cloned().fold(0.0f32, f32::max);
                    peak * peak
                } else {
                    let n_bins = (k_hi - k_lo + 1) as f32;
                    mags[k_lo..=k_hi].iter().map(|m| m * m).sum::<f32>() / n_bins
                };

                // Adaptive threshold from history
                let history = &mut self.band_energy_history[i];
                history.push_back(energy);
                if history.len() > BAND_HISTORY {
                    history.pop_front();
                }

                let avg = history.iter().sum::<f32>() / history.len() as f32;
                let var =
                    history.iter().map(|e| (e - avg).powi(2)).sum::<f32>() / history.len() as f32;
                let threshold = avg * sensitivity + var.sqrt() * 0.5;

                let strength = if threshold > 1e-10 {
                    ((energy - threshold) / threshold).max(0.0).min(3.0)
                } else {
                    0.0
                };

                self.band_cooldown_secs[i] = (self.band_cooldown_secs[i] - dt).max(0.0);
                let beat = self.features.analysis_ready
                    && strength > 0.0
                    && self.band_cooldown_secs[i] <= 0.0
                    && self.features.onset_confidence > 0.15;
                if beat {
                    self.band_cooldown_secs[i] = BAND_MIN_COOLDOWN_SECS;
                }

                // Smoothed energy: instant attack, exponential release
                self.band_smoothed[i] = if energy > self.band_smoothed[i] {
                    energy
                } else {
                    self.band_smoothed[i] * BAND_DECAY
                };

                // Auto-normalise: track slow-decaying per-band peak so every band
                // fills [0, 1] relative to its own typical energy scale
                self.band_display_max[i] = (self.band_display_max[i] * 0.9995)
                    .max(self.band_smoothed[i])
                    .max(1e-10);

                self.features.band_energy[i] = self.band_smoothed[i] / self.band_display_max[i];
                self.features.band_beat[i] = beat;
                self.features.band_strength[i] = strength;
            }

            self.features.kick = self.features.band_beat[0] || self.features.band_beat[1];
            self.features.kick_strength =
                self.features.band_strength[0].max(self.features.band_strength[1]);
            self.features.snare = self.features.band_beat[2] || self.features.band_beat[3];
            self.features.snare_strength =
                self.features.band_strength[2].max(self.features.band_strength[3]);
            self.features.hihat = self.features.band_beat[4] || self.features.band_beat[5];
            self.features.hihat_strength =
                self.features.band_strength[4].max(self.features.band_strength[5]);
        }

        if !self.features.analysis_ready {
            self.features.onset_confidence = 0.0;
            self.features.harmonic_confidence = 0.0;
            self.features.percussive_confidence = 0.0;
            self.features.band_flux_peakiness = 0.0;
            self.features.pitched_stability = 0.0;
            self.features.is_onset = false;
            self.features.band_beat = [false; 6];
            self.features.band_strength = [0.0; 6];
            self.features.kick = false;
            self.features.snare = false;
            self.features.hihat = false;
            self.features.kick_strength = 0.0;
            self.features.snare_strength = 0.0;
            self.features.hihat_strength = 0.0;
        } else {
            self.features.is_onset =
                self.features.is_onset && self.features.onset_confidence > 0.25;
        }

        // ── STFT spectrogram history (lower FFT_DISPLAY_BINS bins, normalised dBFS) ──
        let fft_frame: Vec<f32> = self.features.spectrum_db[..FFT_DISPLAY_BINS]
            .iter()
            .map(|&db| {
                let t = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
                (t * 9.0 + 1.0).log10() // log10(1..10) → 0..1, lifts low values
            })
            .collect();
        self.fft_history.push_back(fft_frame);
        if self.fft_history.len() > FFT_HISTORY {
            self.fft_history.pop_front();
        }

        // ── Advance state ─────────────────────────────────────────────────
        self.prev_magnitudes.copy_from_slice(&mags);
    }
}

#[inline]
fn ema(new: f32, old: f32) -> f32 {
    SMOOTH * new + (1.0 - SMOOTH) * old
}

#[inline]
fn trim_history<T>(history: &mut VecDeque<T>, max_len: usize) {
    while history.len() > max_len {
        history.pop_front();
    }
}

fn mean<I>(values: I) -> f32
where
    I: IntoIterator<Item = f32>,
{
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn median(values: &VecDeque<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn cosine_sim_12(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SAMPLE_RATE: u32 = 44_100;

    fn steady_sine_window(midi_note: u8, sample_rate: u32) -> VecDeque<f32> {
        let freq = midi_note_to_hz(midi_note);
        (0..FFT_SIZE)
            .map(|idx| {
                let phase = TAU * freq * idx as f32 / sample_rate as f32;
                0.8 * phase.sin()
            })
            .collect()
    }

    fn analyse_steady_tone(midi_note: u8) -> Analyser {
        let mut analyser = Analyser::new(TEST_SAMPLE_RATE);
        let window = steady_sine_window(midi_note, TEST_SAMPLE_RATE);
        let dt = FFT_SIZE as f32 / TEST_SAMPLE_RATE as f32;
        for _ in 0..8 {
            analyser.process(&window, dt);
        }
        analyser
    }

    fn dominant_cqt_bin(cqt: &[f32]) -> usize {
        cqt.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .expect("CQT should not be empty")
    }

    #[test]
    fn cqt_midi_anchor_formula_is_correct() {
        assert_eq!(cqt_bin_to_midi_note(0), Some(21));
        assert_eq!(midi_note_to_cqt_bin(21), Some(0));
        assert_eq!(cqt_bin_to_midi_note(48), Some(69));
        assert_eq!(midi_note_to_cqt_bin(69), Some(48));
        assert_eq!(cqt_bin_to_midi_note(83), Some(104));
        assert_eq!(midi_note_to_cqt_bin(104), Some(83));

        let a0_hz = cqt_bin_center_hz(0).expect("bin 0 should exist");
        let a4_hz = cqt_bin_center_hz(48).expect("bin 48 should exist");
        let top_hz = cqt_bin_center_hz(83).expect("bin 83 should exist");
        assert!((a0_hz - 27.5).abs() < 1e-4);
        assert!((a4_hz - 440.0).abs() < 1e-3);
        assert!((top_hz - midi_note_to_hz(104)).abs() < 1e-2);
    }

    #[test]
    fn cqt_midi_round_trip_covers_supported_range() {
        for bin in 0..CQT_BINS {
            let midi_note = cqt_bin_to_midi_note(bin).expect("every CQT bin should map to MIDI");
            assert_eq!(midi_note_to_cqt_bin(midi_note), Some(bin));
        }

        assert_eq!(midi_note_to_cqt_bin(20), None);
        assert_eq!(midi_note_to_cqt_bin(105), None);
        assert_eq!(cqt_bin_to_midi_note(CQT_BINS), None);
    }

    #[test]
    fn midband_steady_sine_peaks_stay_close_to_expected_cqt_bins() {
        // The current CQT is an FFT-derived approximation, so verify peak placement in the
        // midband where the analyser has enough frequency resolution to discriminate semitones.
        // Formula-level tests above keep the exact anchor and range checks honest.
        for midi_note in [57u8, 69, 81, 93] {
            let analyser = analyse_steady_tone(midi_note);
            let peak_bin = dominant_cqt_bin(&analyser.features.cqt);
            let expected_bin =
                midi_note_to_cqt_bin(midi_note).expect("test note should fit in CQT range");
            let bin_error = peak_bin as isize - expected_bin as isize;

            assert!(
                bin_error.abs() <= 2,
                "steady sine for MIDI note {midi_note} peaked at bin {peak_bin}, expected {expected_bin}"
            );
            let mapped_note = cqt_bin_to_midi_note(peak_bin).expect("peak bin should map to MIDI");
            assert!(
                (mapped_note as i16 - midi_note as i16).abs() <= 2,
                "steady sine for MIDI note {midi_note} mapped to MIDI note {mapped_note}"
            );
        }
    }
}
