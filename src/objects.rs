use crate::analysis::AudioFeatures;

// ── Constants ────────────────────────────────────────────────────────────────

const MAX_PROTOTYPES: usize = 16;

/// Prototype merge thresholds — cosine distance below this = same sound, update centroid.
/// Generous values prevent over-clustering when the same instrument varies slightly hit-to-hit.
const PERC_THRESHOLD: f32 = 0.25;
const HARM_THRESHOLD: f32 = 0.35;
const PH_THRESHOLD:   f32 = 0.28;

/// Fast-path: match known sounds at frame 0 using flux fingerprint only
const FAST_FLUX_THRESHOLD: f32 = 0.20;

/// Minimum chroma flux sum to count toward harmonic birth (flux-based path)
const HARM_FLUX_THRESHOLD: f32 = 0.03;
/// Consecutive frames with chroma flux above threshold before harmonic object is born
const HARM_BIRTH_FRAMES: u8 = 3;
/// L2 norm of L1-normalised chroma above which sustained pitch content is inferred.
/// Uniform spectrum (noise) ≈ 0.29; strong single pitch ≈ 1.0.
const HARM_ENERGY_THRESHOLD: f32 = 0.50;

/// Minimum onset_strength to spawn a new percussive object
const MIN_ONSET_STRENGTH: f32 = 0.30;
/// Minimum per-band beat strength to trigger from kick/snare/hihat
const MIN_BEAT_STRENGTH: f32 = 0.40;

/// RMS below which the silence timer accumulates
const SILENCE_THRESHOLD_RMS: f32 = 0.003;
/// Seconds of silence before all state is cleared
const SILENCE_CLEAR_SECS: f32 = 10.0;

/// Percussive energy decay time constant (seconds)
const PERC_DECAY_TAU: f32 = 0.5;

/// Fraction of birth chroma energy a pitch class must retain to keep harmonic alive
const HARM_SUSTAIN_RATIO: f32 = 0.25;

/// Grace period after harmonic energy drops, before killing the object
const HARM_GRACE_SECS: f32 = 0.25;

/// PH distance blend weights
const PH_FLUX_WEIGHT: f32 = 0.4;
const PH_HARM_WEIGHT: f32 = 0.6;

/// harmonic_ratio threshold at frame+2 to upgrade percussive → PH
const PH_HARMONIC_RATIO_THRESHOLD: f32 = 0.60;

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SoundKind {
    Percussive,
    Harmonic,
    PercussiveHarmonic,
}

#[derive(Debug, Clone)]
pub enum Fingerprint {
    Perc([f32; 6]),
    Harm([f32; 15]),
    Ph { flux: [f32; 6], harm: [f32; 15] },
}

#[derive(Debug, Clone)]
pub struct Prototype {
    pub fp: Fingerprint,
    pub cluster_id: usize,
    pub last_seen_secs: f32,
    pub count: usize,
    /// CQT pitch axis Y [0,1] for harmonic/PH; None for percussive
    pub pitch_com: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SoundObject {
    pub kind: SoundKind,
    pub cluster_id: usize,
    /// Unique identifier for this object instance — stable across Vec::retain
    pub spawn_id: usize,
    pub age_secs: f32,
    pub energy: f32,
    /// Acoustic X [0,1]: spectral brightness at birth
    pub acoustic_x: f32,
    /// Acoustic Y [0,1]: pitch or band-flux center-of-mass at birth
    pub acoustic_y: f32,
    pub visual_hue: f32,
    pub visual_size: f32,
    /// 0=circle, 1=ring, 2=spike, 3=square
    pub visual_shape: u8,
    /// Chromagram at birth — for harmonic lifecycle tracking
    pub anchor_chroma: [f32; 12],
    /// Grace period countdown before harmonic/PH object dies
    pub release_timer: f32,
}

/// Onset event captured at frame 0, pending reclassification at frame+2
struct PendingReclassify {
    /// Stable identity of the spawned object — index is unstable after Vec::retain
    spawn_id: usize,
    /// Original band_flux_delta from onset frame — stored for PH prototype building
    onset_flux: [f32; 6],
    age_frames: u8,
}

pub struct SoundObjectDetector {
    perc_prototypes: Vec<Prototype>,
    harm_prototypes: Vec<Prototype>,
    ph_prototypes: Vec<Prototype>,
    pub live_objects: Vec<SoundObject>,
    elapsed_secs: f32,
    next_cluster_id: usize,
    next_spawn_id: usize,
    /// Queue of novel onsets pending PH reclassification at frame+2.
    /// Using Vec so dense passages (multiple onsets in 2 frames) don't lose events.
    pending_reclassify: Vec<PendingReclassify>,
    /// Consecutive frames with significant chroma flux for harmonic birth
    harm_flux_streak: u8,
    /// Cooldown in seconds after a harmonic birth (suppresses re-birth of same event)
    harm_birth_cooldown: f32,
    /// Accumulated silence duration — clears all state at SILENCE_CLEAR_SECS
    silence_timer: f32,
    /// True for exactly one frame after a silence-triggered clear
    pub just_cleared: bool,
}

// ── Visual params ─────────────────────────────────────────────────────────────

fn visual_params(id: usize) -> (f32, f32, u8) {
    let h0 = id.wrapping_mul(2654435761).wrapping_add(0x9e3779b9);
    let h1 = h0.wrapping_mul(2246822519).wrapping_add(0x6c62272e);
    let h2 = h1.wrapping_mul(3266489917).wrapping_add(0xb5ad4ec3);
    let hue = (h0 & 0xFFFF) as f32 / 65535.0;
    let size = 0.3 + (h1 & 0xFF) as f32 / 255.0 * 0.7;
    let shape = (h2 & 3) as u8;
    (hue, size, shape)
}

// ── Distance metrics ─────────────────────────────────────────────────────────

fn cosine_dist_6(a: &[f32; 6], b: &[f32; 6]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { return 1.0; }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn cosine_dist_15(a: &[f32; 15], b: &[f32; 15]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { return 1.0; }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn ph_dist(flux: &[f32; 6], harm: &[f32; 15], proto: &Prototype) -> f32 {
    match &proto.fp {
        Fingerprint::Ph { flux: pf, harm: ph } => {
            PH_FLUX_WEIGHT * cosine_dist_6(flux, pf) + PH_HARM_WEIGHT * cosine_dist_15(harm, ph)
        }
        _ => 1.0,
    }
}

// ── Feature helpers ───────────────────────────────────────────────────────────

/// 15D fingerprint: 12D chroma + brightness + roughness + width
fn harm_fingerprint(features: &AudioFeatures) -> [f32; 15] {
    let mut fp = [0.0f32; 15];
    fp[..12].copy_from_slice(&features.chroma);
    fp[12] = features.brightness;
    fp[13] = features.roughness;
    fp[14] = features.width;
    fp
}

/// Acoustic Y [0,1] from chroma flux center-of-mass.
/// Points to the pitch class of what *changed* this frame, not the dominant sustain.
fn chroma_flux_y(features: &AudioFeatures) -> f32 {
    let total: f32 = features.chroma_flux.iter().sum();
    if total < 1e-8 { return 0.5; }
    let com: f32 = features.chroma_flux.iter().enumerate()
        .map(|(c, &v)| c as f32 * v)
        .sum::<f32>() / total;
    (com / 12.0).clamp(0.0, 1.0)
}

/// Acoustic Y [0,1] from band flux center-of-mass (for percussive sounds).
fn band_flux_y(features: &AudioFeatures) -> f32 {
    let total: f32 = features.band_flux_delta.iter().sum();
    if total < 1e-8 { return 0.3; }
    let com: f32 = features.band_flux_delta.iter().enumerate()
        .map(|(i, &v)| i as f32 * v)
        .sum::<f32>() / total;
    (com / 5.0).clamp(0.0, 1.0)
}

// ── Prototype pool helpers ────────────────────────────────────────────────────

fn nearest_perc<'a>(pool: &'a [Prototype], flux: &[f32; 6]) -> Option<(usize, f32)> {
    pool.iter().enumerate()
        .filter_map(|(i, p)| if let Fingerprint::Perc(f) = &p.fp {
            Some((i, cosine_dist_6(flux, f)))
        } else { None })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn nearest_harm(pool: &[Prototype], harm: &[f32; 15]) -> Option<(usize, f32)> {
    pool.iter().enumerate()
        .filter_map(|(i, p)| if let Fingerprint::Harm(h) = &p.fp {
            Some((i, cosine_dist_15(harm, h)))
        } else { None })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn nearest_ph_by_flux(pool: &[Prototype], flux: &[f32; 6]) -> Option<(usize, f32)> {
    pool.iter().enumerate()
        .filter_map(|(i, p)| if let Fingerprint::Ph { flux: pf, .. } = &p.fp {
            Some((i, cosine_dist_6(flux, pf)))
        } else { None })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn nearest_ph(pool: &[Prototype], flux: &[f32; 6], harm: &[f32; 15]) -> Option<(usize, f32)> {
    pool.iter().enumerate()
        .map(|(i, p)| (i, ph_dist(flux, harm, p)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

/// Evict least-recently-seen prototype (pure LRU)
fn evict(pool: &mut Vec<Prototype>) {
    if pool.is_empty() { return; }
    let idx = pool.iter().enumerate()
        .min_by(|a, b| a.1.last_seen_secs.partial_cmp(&b.1.last_seen_secs).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    pool.remove(idx);
}

fn ema_update_6(target: &mut [f32; 6], new: &[f32; 6]) {
    for (t, &n) in target.iter_mut().zip(new.iter()) { *t = 0.9 * *t + 0.1 * n; }
}

fn ema_update_15(target: &mut [f32; 15], new: &[f32; 15]) {
    for (t, &n) in target.iter_mut().zip(new.iter()) { *t = 0.9 * *t + 0.1 * n; }
}

// ── Detector ──────────────────────────────────────────────────────────────────

impl SoundObjectDetector {
    pub fn new() -> Self {
        Self {
            perc_prototypes: Vec::with_capacity(MAX_PROTOTYPES),
            harm_prototypes: Vec::with_capacity(MAX_PROTOTYPES),
            ph_prototypes: Vec::with_capacity(MAX_PROTOTYPES),
            live_objects: Vec::new(),
            elapsed_secs: 0.0,
            next_cluster_id: 0,
            next_spawn_id: 0,
            pending_reclassify: Vec::new(),
            harm_flux_streak: 0,
            harm_birth_cooldown: 0.0,
            silence_timer: 0.0,
            just_cleared: false,
        }
    }

    /// Call once per nannou update() tick.
    /// `dt` = seconds since last frame (update.since_last.secs_f32()).
    pub fn process(&mut self, features: &AudioFeatures, dt: f32) {
        self.just_cleared = false;
        self.elapsed_secs += dt;

        // ── Silence detection — clear all state after prolonged quiet ─────
        if features.rms < SILENCE_THRESHOLD_RMS {
            self.silence_timer += dt;
            if self.silence_timer >= SILENCE_CLEAR_SECS {
                self.live_objects.clear();
                self.pending_reclassify.clear();
                self.harm_flux_streak = 0;
                self.harm_birth_cooldown = 0.0;
                self.silence_timer = 0.0;
                self.just_cleared = true;
                return;
            }
        } else {
            self.silence_timer = 0.0;
        }

        // ── Decay / age all objects ───────────────────────────────────────
        for obj in &mut self.live_objects {
            obj.age_secs += dt;
            if obj.kind == SoundKind::Percussive {
                // Exponential decay: e^(-dt/tau)
                obj.energy *= (-dt / PERC_DECAY_TAU).exp();
            }
            // PH spike → ring transition after attack phase (~50ms)
            if obj.kind == SoundKind::PercussiveHarmonic && obj.visual_shape == 2 && obj.age_secs > 0.05 {
                obj.visual_shape = 1; // ring
            }
        }

        // ── Harmonic lifecycle (Harmonic and PH sustain) ─────────────────
        for obj in &mut self.live_objects {
            if obj.kind != SoundKind::Harmonic && obj.kind != SoundKind::PercussiveHarmonic {
                continue;
            }
            let any_active = obj.anchor_chroma.iter().enumerate().any(|(c, &birth_e)| {
                birth_e > 1e-4 && features.chroma[c] >= birth_e * HARM_SUSTAIN_RATIO
            });
            if any_active {
                obj.energy = 1.0;
                obj.release_timer = 0.0;
            } else {
                obj.release_timer += dt;
            }
        }

        // ── Reclassify pending onsets at frame+2 ─────────────────────────
        for pr in &mut self.pending_reclassify { pr.age_frames += 1; }
        // Partition into ready and still-pending — stable, no nightly features
        let prev = std::mem::take(&mut self.pending_reclassify);
        let (ready, still_pending): (Vec<_>, Vec<_>) = prev.into_iter()
            .partition(|pr| pr.age_frames >= 2);
        self.pending_reclassify = still_pending;
        for pr in ready { self.reclassify_one(pr, features); }

        // ── Remove dead objects ───────────────────────────────────────────
        self.live_objects.retain(|obj| match obj.kind {
            SoundKind::Percussive => obj.energy >= 0.01,
            SoundKind::Harmonic | SoundKind::PercussiveHarmonic => obj.release_timer < HARM_GRACE_SECS,
        });

        // Prune pending entries whose target object was already removed
        let live = &self.live_objects;
        self.pending_reclassify.retain(|pr| live.iter().any(|obj| obj.spawn_id == pr.spawn_id));

        // ── Harmonic birth: sustained chroma flux ────────────────────────
        // No longer gated by !is_onset — pads and strings need to be detected
        // even when percussion is playing simultaneously.
        self.harm_birth_cooldown = (self.harm_birth_cooldown - dt).max(0.0);
        let flux_sum: f32 = features.chroma_flux.iter().sum();
        if flux_sum > HARM_FLUX_THRESHOLD {
            self.harm_flux_streak = self.harm_flux_streak.saturating_add(1);
        } else {
            self.harm_flux_streak = 0;
        }
        if self.harm_flux_streak >= HARM_BIRTH_FRAMES && self.harm_birth_cooldown == 0.0 {
            self.harm_flux_streak = 0;
            self.harm_birth_cooldown = 1.5;
            self.spawn_harmonic(features);
        }

        // ── Chroma-energy fallback: catch already-playing pitched content ─
        // Only fires when nothing harmonic is alive and signal is clearly present.
        let no_harm_alive = !self.live_objects.iter()
            .any(|o| o.kind == SoundKind::Harmonic || o.kind == SoundKind::PercussiveHarmonic);
        if no_harm_alive && self.harm_birth_cooldown == 0.0 && features.rms > 0.03 {
            let chroma_l2: f32 = features.chroma.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if chroma_l2 > HARM_ENERGY_THRESHOLD {
                self.harm_birth_cooldown = 1.5;
                self.spawn_harmonic(features);
            }
        }

        // ── Onset handling: gated on minimum strength to prevent soft noise ─
        let strong_onset = features.is_onset && features.onset_strength > MIN_ONSET_STRENGTH;
        let strong_beat = (features.kick  && features.kick_strength  > MIN_BEAT_STRENGTH)
                       || (features.snare && features.snare_strength > MIN_BEAT_STRENGTH)
                       || (features.hihat && features.hihat_strength > MIN_BEAT_STRENGTH);
        if strong_onset || strong_beat {
            self.handle_onset(features);
        }
    }

    fn handle_onset(&mut self, features: &AudioFeatures) {
        let flux = features.band_flux_delta;
        let ax = features.brightness;
        let ay_perc = band_flux_y(features);
        let energy = features.onset_strength.clamp(0.5, 2.0);

        // Fast path 1: known PH prototype matched by flux alone
        if let Some((idx, dist)) = nearest_ph_by_flux(&self.ph_prototypes, &flux) {
            if dist < FAST_FLUX_THRESHOLD {
                let cluster_id = self.ph_prototypes[idx].cluster_id;
                let ay_ph = self.ph_prototypes[idx].pitch_com.unwrap_or(ay_perc);
                self.ph_prototypes[idx].count += 1;
                self.ph_prototypes[idx].last_seen_secs = self.elapsed_secs;
                self.spawn_object(SoundKind::PercussiveHarmonic, cluster_id, ax, ay_ph,
                                  energy, &features.chroma);
                return;
            }
        }

        // Fast path 2: known percussive prototype
        if let Some((idx, dist)) = nearest_perc(&self.perc_prototypes, &flux) {
            if dist < PERC_THRESHOLD {
                let cluster_id = self.perc_prototypes[idx].cluster_id;
                self.perc_prototypes[idx].count += 1;
                self.perc_prototypes[idx].last_seen_secs = self.elapsed_secs;
                if let Fingerprint::Perc(ref mut f) = self.perc_prototypes[idx].fp {
                    ema_update_6(f, &flux);
                }
                self.spawn_object(SoundKind::Percussive, cluster_id, ax, ay_perc,
                                  energy, &features.chroma);
                return;
            }
        }

        // Unknown sound: spawn as Percussive, build provisional prototype, queue reclassify
        let cluster_id = self.alloc_id();
        if self.perc_prototypes.len() >= MAX_PROTOTYPES { evict(&mut self.perc_prototypes); }
        self.perc_prototypes.push(Prototype {
            fp: Fingerprint::Perc(flux),
            cluster_id,
            last_seen_secs: self.elapsed_secs,
            count: 1,
            pitch_com: None,
        });

        let pending_spawn_id = self.next_spawn_id;
        self.spawn_object(SoundKind::Percussive, cluster_id, ax, ay_perc,
                          energy, &features.chroma);

        self.pending_reclassify.push(PendingReclassify {
            spawn_id: pending_spawn_id,
            onset_flux: flux,
            age_frames: 0,
        });
    }

    fn reclassify_one(&mut self, pr: PendingReclassify, features: &AudioFeatures) {

        // Use per-frame H/P ratio (not EMA-smoothed harmonic_ratio) for onset-local classification.
        let instant_ratio = features.harmonic_energy
            / (features.harmonic_energy + features.percussive_energy + 1e-10);
        if instant_ratio < PH_HARMONIC_RATIO_THRESHOLD {
            return;
        }

        let obj_idx = match self.live_objects.iter().position(|o| o.spawn_id == pr.spawn_id) {
            Some(i) => i,
            None => return,
        };

        let onset_flux = pr.onset_flux;
        let harm = harm_fingerprint(features);
        let ay_ph = chroma_flux_y(features);

        // Remove the provisional perc prototype we just created for this event
        let old_cluster_id = self.live_objects[obj_idx].cluster_id;
        self.perc_prototypes.retain(|p| p.cluster_id != old_cluster_id || p.count > 1);

        // Match or create PH prototype
        let cluster_id = match nearest_ph(&self.ph_prototypes, &onset_flux, &harm) {
            Some((idx, dist)) if dist < PH_THRESHOLD => {
                let id = self.ph_prototypes[idx].cluster_id;
                self.ph_prototypes[idx].count += 1;
                self.ph_prototypes[idx].last_seen_secs = self.elapsed_secs;
                if let Fingerprint::Ph { flux: ref mut pf, harm: ref mut ph } = self.ph_prototypes[idx].fp {
                    ema_update_6(pf, &onset_flux);
                    ema_update_15(ph, &harm);
                }
                self.ph_prototypes[idx].pitch_com = Some(ay_ph);
                id
            }
            _ => {
                let id = self.alloc_id();
                if self.ph_prototypes.len() >= MAX_PROTOTYPES { evict(&mut self.ph_prototypes); }
                self.ph_prototypes.push(Prototype {
                    fp: Fingerprint::Ph { flux: onset_flux, harm },
                    cluster_id: id,
                    last_seen_secs: self.elapsed_secs,
                    count: 1,
                    pitch_com: Some(ay_ph),
                });
                id
            }
        };

        // Upgrade the live object
        let (hue, size, _shape) = visual_params(cluster_id);
        let obj = &mut self.live_objects[obj_idx];
        obj.kind = SoundKind::PercussiveHarmonic;
        obj.cluster_id = cluster_id;
        obj.acoustic_y = ay_ph;
        obj.visual_hue = hue;
        obj.visual_size = size;
        obj.visual_shape = 2; // spike — transitions to ring at age > 50ms in process()
        // Refresh anchor_chroma to post-onset harmonic content (frame+2), not attack frame
        obj.anchor_chroma = features.chroma;
    }

    fn spawn_harmonic(&mut self, features: &AudioFeatures) {
        let harm = harm_fingerprint(features);
        let ax = features.brightness;
        let ay = chroma_flux_y(features);

        let cluster_id = match nearest_harm(&self.harm_prototypes, &harm) {
            Some((idx, dist)) if dist < HARM_THRESHOLD => {
                let id = self.harm_prototypes[idx].cluster_id;
                self.harm_prototypes[idx].count += 1;
                self.harm_prototypes[idx].last_seen_secs = self.elapsed_secs;
                if let Fingerprint::Harm(ref mut h) = self.harm_prototypes[idx].fp {
                    ema_update_15(h, &harm);
                }
                id
            }
            _ => {
                let id = self.alloc_id();
                if self.harm_prototypes.len() >= MAX_PROTOTYPES { evict(&mut self.harm_prototypes); }
                self.harm_prototypes.push(Prototype {
                    fp: Fingerprint::Harm(harm),
                    cluster_id: id,
                    last_seen_secs: self.elapsed_secs,
                    count: 1,
                    pitch_com: Some(ay),
                });
                id
            }
        };

        self.spawn_object(SoundKind::Harmonic, cluster_id, ax, ay, 1.0, &features.chroma);
    }

    fn spawn_object(
        &mut self,
        kind: SoundKind,
        cluster_id: usize,
        acoustic_x: f32,
        acoustic_y: f32,
        energy: f32,
        chroma: &[f32; 12],
    ) {
        let (hue, size, shape) = visual_params(cluster_id);
        // PH always starts as spike (attack transient), transitions to ring in process().
        // Percussive and Harmonic use the cluster-derived hashed shape for stable identity.
        let visual_shape = if kind == SoundKind::PercussiveHarmonic { 2 } else { shape };
        let spawn_id = self.next_spawn_id;
        self.next_spawn_id += 1;
        self.live_objects.push(SoundObject {
            kind,
            cluster_id,
            spawn_id,
            age_secs: 0.0,
            energy,
            acoustic_x: acoustic_x.clamp(0.0, 1.0),
            acoustic_y: acoustic_y.clamp(0.0, 1.0),
            visual_hue: hue,
            visual_size: size,
            visual_shape,
            anchor_chroma: *chroma,
            release_timer: 0.0,
        });
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_cluster_id;
        self.next_cluster_id += 1;
        id
    }
}
