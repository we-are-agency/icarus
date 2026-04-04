use crate::analysis::AudioFeatures;
use std::collections::VecDeque;

const DETECTOR_DEBUG: bool = false;
const MAX_PROTOTYPES: usize = 16;

const TRUSTED_PROTOTYPE_COUNT: usize = 3;
const PERC_TRUSTED_THRESHOLD: f32 = 0.20;
const PERC_WEAK_THRESHOLD: f32 = 0.11;
const HARM_TRUSTED_THRESHOLD: f32 = 0.28;
const HARM_WEAK_THRESHOLD: f32 = 0.18;
const PH_TRUSTED_THRESHOLD: f32 = 0.22;
const PH_WEAK_THRESHOLD: f32 = 0.14;
const FAST_FLUX_THRESHOLD: f32 = 0.11;

const PENDING_LOOKAHEAD_SECS: f32 = 0.05;
const PENDING_MAX_AGE_SECS: f32 = 0.15;
const PENDING_DEBOUNCE_SECS: f32 = 0.04;
const PENDING_DEBOUNCE_DIST: f32 = 0.08;

const MIN_ONSET_CONFIDENCE: f32 = 0.32;
const MIN_PERCUSSIVE_CONFIDENCE: f32 = 0.42;
const MIN_PH_HARMONIC_CONFIDENCE: f32 = 0.58;
const MIN_BEAT_STRENGTH: f32 = 0.40;

const SILENCE_THRESHOLD_RMS: f32 = 0.003;
const SILENCE_CLEAR_SECS: f32 = 10.0;

const PERC_DECAY_TAU: f32 = 0.5;
const PERC_DUPLICATE_SECS: f32 = 0.10;
const PERC_DUPLICATE_DIST: f32 = 0.14;

const HARM_ENTER_THRESHOLD: f32 = 0.58;
const HARM_EXIT_THRESHOLD: f32 = 0.40;
const HARM_ENTER_HOLD_SECS: f32 = 0.10;
const HARM_BIRTH_COOLDOWN_SECS: f32 = 0.35;
const HARM_GRACE_SECS: f32 = 0.30;
const HARM_TEMPLATE_MATCH_THRESHOLD: f32 = 0.78;
const LIVE_HARMONIC_MATCH_THRESHOLD: f32 = 0.84;
const HARM_DUPLICATE_SECS: f32 = 0.25;
const PRESENCE_MIN_RMS: f32 = 0.012;
const PRESENCE_STRONG_RMS: f32 = 0.035;
const PRESENCE_ENTER_SECS: f32 = 0.06;
const PRESENCE_TEMPLATE_MATCH_THRESHOLD: f32 = 0.52;
const PRESENCE_HARMONIC_FLOOR: f32 = 0.18;
const PRESENCE_PITCHED_FLOOR: f32 = 0.16;

const PH_FLUX_WEIGHT: f32 = 0.4;
const PH_HARM_WEIGHT: f32 = 0.6;

const RECENT_BIRTH_KEEP_SECS: f32 = 0.8;

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
    /// Birth chroma snapshot
    pub anchor_chroma: [f32; 12],
    /// Slowly-updated sustain template used for harmonic lifecycle tracking
    pub template_chroma: [f32; 12],
    /// Grace period countdown before harmonic/PH object dies
    pub release_timer: f32,
}

struct PendingCandidate {
    started_secs: f32,
    onset_flux: [f32; 6],
    brightness: f32,
    percussive_y: f32,
    onset_confidence: f32,
    peak_percussive_confidence: f32,
    peak_harmonic_confidence: f32,
    latest_harm: [f32; 15],
    latest_chroma: [f32; 12],
    latest_pitch_y: f32,
}

#[derive(Clone, Copy)]
struct RecentBirth {
    kind: SoundKind,
    cluster_id: usize,
    at_secs: f32,
    flux: [f32; 6],
}

#[derive(Clone, Copy)]
enum PrototypeDecision {
    Reuse(usize),
    CreateNew,
}

pub struct SoundObjectDetector {
    perc_prototypes: Vec<Prototype>,
    harm_prototypes: Vec<Prototype>,
    ph_prototypes: Vec<Prototype>,
    pub live_objects: Vec<SoundObject>,
    elapsed_secs: f32,
    next_cluster_id: usize,
    next_spawn_id: usize,
    pending_candidates: Vec<PendingCandidate>,
    harmonic_enter_timer: f32,
    presence_enter_timer: f32,
    harmonic_birth_cooldown: f32,
    recent_births: VecDeque<RecentBirth>,
    silence_timer: f32,
    pub just_cleared: bool,
}

fn visual_params(id: usize) -> (f32, f32, u8) {
    let h0 = id.wrapping_mul(2654435761).wrapping_add(0x9e3779b9);
    let h1 = h0.wrapping_mul(2246822519).wrapping_add(0x6c62272e);
    let h2 = h1.wrapping_mul(3266489917).wrapping_add(0xb5ad4ec3);
    let hue = (h0 & 0xFFFF) as f32 / 65535.0;
    let size = 0.3 + (h1 & 0xFF) as f32 / 255.0 * 0.7;
    let shape = (h2 & 3) as u8;
    (hue, size, shape)
}

fn cosine_dist_6(a: &[f32; 6], b: &[f32; 6]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 1.0;
    }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn cosine_dist_15(a: &[f32; 15], b: &[f32; 15]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 1.0;
    }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
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

fn ph_dist(flux: &[f32; 6], harm: &[f32; 15], proto: &Prototype) -> f32 {
    match &proto.fp {
        Fingerprint::Ph { flux: pf, harm: ph } => {
            PH_FLUX_WEIGHT * cosine_dist_6(flux, pf) + PH_HARM_WEIGHT * cosine_dist_15(harm, ph)
        }
        _ => 1.0,
    }
}

fn harm_fingerprint(features: &AudioFeatures) -> [f32; 15] {
    let mut fp = [0.0f32; 15];
    fp[..12].copy_from_slice(&features.chroma);
    fp[12] = features.brightness;
    fp[13] = features.roughness;
    fp[14] = features.width;
    fp
}

fn chroma_flux_y(features: &AudioFeatures) -> f32 {
    let total: f32 = features.chroma_flux.iter().sum();
    if total < 1e-8 {
        return chroma_y(features);
    }
    let com: f32 = features
        .chroma_flux
        .iter()
        .enumerate()
        .map(|(c, &v)| c as f32 * v)
        .sum::<f32>()
        / total;
    (com / 12.0).clamp(0.0, 1.0)
}

fn chroma_y(features: &AudioFeatures) -> f32 {
    let total: f32 = features.chroma.iter().sum();
    if total < 1e-8 {
        return 0.5;
    }
    let com: f32 = features
        .chroma
        .iter()
        .enumerate()
        .map(|(c, &v)| c as f32 * v)
        .sum::<f32>()
        / total;
    (com / 12.0).clamp(0.0, 1.0)
}

fn band_flux_y(features: &AudioFeatures) -> f32 {
    let total: f32 = features.band_flux_delta.iter().sum();
    if total < 1e-8 {
        return 0.3;
    }
    let com: f32 = features
        .band_flux_delta
        .iter()
        .enumerate()
        .map(|(i, &v)| i as f32 * v)
        .sum::<f32>()
        / total;
    (com / 5.0).clamp(0.0, 1.0)
}

fn harmonic_presence_score(features: &AudioFeatures) -> f32 {
    (0.6 * features.harmonic_confidence + 0.4 * features.pitched_stability).clamp(0.0, 1.0)
}

fn nearest_perc(pool: &[Prototype], flux: &[f32; 6]) -> Option<(usize, f32)> {
    pool.iter()
        .enumerate()
        .filter_map(|(i, p)| match &p.fp {
            Fingerprint::Perc(f) => Some((i, cosine_dist_6(flux, f))),
            _ => None,
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn nearest_harm(pool: &[Prototype], harm: &[f32; 15]) -> Option<(usize, f32)> {
    pool.iter()
        .enumerate()
        .filter_map(|(i, p)| match &p.fp {
            Fingerprint::Harm(h) => Some((i, cosine_dist_15(harm, h))),
            _ => None,
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn nearest_ph_by_flux(pool: &[Prototype], flux: &[f32; 6]) -> Option<(usize, f32)> {
    pool.iter()
        .enumerate()
        .filter_map(|(i, p)| match &p.fp {
            Fingerprint::Ph { flux: pf, .. } => Some((i, cosine_dist_6(flux, pf))),
            _ => None,
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn nearest_ph(pool: &[Prototype], flux: &[f32; 6], harm: &[f32; 15]) -> Option<(usize, f32)> {
    pool.iter()
        .enumerate()
        .map(|(i, p)| (i, ph_dist(flux, harm, p)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn evict(pool: &mut Vec<Prototype>, now_secs: f32) {
    if pool.is_empty() {
        return;
    }
    let idx = pool
        .iter()
        .enumerate()
        .max_by(|a, b| {
            eviction_score(a.1, now_secs)
                .partial_cmp(&eviction_score(b.1, now_secs))
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();
    pool.remove(idx);
}

fn eviction_score(proto: &Prototype, now_secs: f32) -> f32 {
    let age = (now_secs - proto.last_seen_secs).max(0.0);
    let count_weight = proto.count.min(4) as f32;
    age / count_weight
        + if proto.count < TRUSTED_PROTOTYPE_COUNT {
            2.0
        } else {
            0.0
        }
}

fn prototype_matches(
    proto: &Prototype,
    dist: f32,
    trusted_threshold: f32,
    weak_threshold: f32,
) -> bool {
    if proto.count >= TRUSTED_PROTOTYPE_COUNT {
        dist < trusted_threshold
    } else {
        dist < weak_threshold
    }
}

fn ema_update_6(target: &mut [f32; 6], new: &[f32; 6]) {
    for (t, &n) in target.iter_mut().zip(new.iter()) {
        *t = 0.9 * *t + 0.1 * n;
    }
}

fn ema_update_12(target: &mut [f32; 12], new: &[f32; 12]) {
    for (t, &n) in target.iter_mut().zip(new.iter()) {
        *t = 0.95 * *t + 0.05 * n;
    }
}

fn ema_update_15(target: &mut [f32; 15], new: &[f32; 15]) {
    for (t, &n) in target.iter_mut().zip(new.iter()) {
        *t = 0.9 * *t + 0.1 * n;
    }
}

fn debug_log(message: &str) {
    if DETECTOR_DEBUG {
        eprintln!("detector: {message}");
    }
}

impl PendingCandidate {
    fn new(features: &AudioFeatures, now_secs: f32) -> Self {
        Self {
            started_secs: now_secs,
            onset_flux: features.band_flux_delta,
            brightness: features.brightness,
            percussive_y: band_flux_y(features),
            onset_confidence: features.onset_confidence,
            peak_percussive_confidence: features.percussive_confidence,
            peak_harmonic_confidence: features.harmonic_confidence,
            latest_harm: harm_fingerprint(features),
            latest_chroma: features.chroma,
            latest_pitch_y: chroma_flux_y(features),
        }
    }

    fn update(&mut self, features: &AudioFeatures) {
        self.onset_confidence = self.onset_confidence.max(features.onset_confidence);
        self.peak_percussive_confidence = self
            .peak_percussive_confidence
            .max(features.percussive_confidence);
        if features.harmonic_confidence >= self.peak_harmonic_confidence {
            self.peak_harmonic_confidence = features.harmonic_confidence;
            self.latest_harm = harm_fingerprint(features);
            self.latest_chroma = features.chroma;
            self.latest_pitch_y = chroma_flux_y(features);
        }
    }
}

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
            pending_candidates: Vec::new(),
            harmonic_enter_timer: 0.0,
            presence_enter_timer: 0.0,
            harmonic_birth_cooldown: 0.0,
            recent_births: VecDeque::new(),
            silence_timer: 0.0,
            just_cleared: false,
        }
    }

    pub fn process(&mut self, features: &AudioFeatures, dt: f32) {
        self.just_cleared = false;
        self.elapsed_secs += dt.max(0.0);
        self.prune_recent_births();

        if features.rms < SILENCE_THRESHOLD_RMS {
            self.silence_timer += dt;
            if self.silence_timer >= SILENCE_CLEAR_SECS {
                self.clear_all_state();
                self.just_cleared = true;
                return;
            }
        } else {
            self.silence_timer = 0.0;
        }

        self.age_and_update_live_objects(features, dt);

        for candidate in &mut self.pending_candidates {
            candidate.update(features);
        }
        self.commit_ready_candidates();
        self.pending_candidates
            .retain(|candidate| self.elapsed_secs - candidate.started_secs <= PENDING_MAX_AGE_SECS);

        self.live_objects.retain(|obj| match obj.kind {
            SoundKind::Percussive => obj.energy >= 0.01,
            SoundKind::Harmonic | SoundKind::PercussiveHarmonic => {
                obj.release_timer < HARM_GRACE_SECS
            }
        });

        self.harmonic_birth_cooldown = (self.harmonic_birth_cooldown - dt).max(0.0);
        self.handle_harmonic_presence(features, dt);
        self.ensure_sound_presence(features, dt);

        if self.should_queue_onset(features) {
            self.queue_onset_candidate(features);
        }
    }

    fn clear_all_state(&mut self) {
        self.live_objects.clear();
        self.pending_candidates.clear();
        self.perc_prototypes.clear();
        self.harm_prototypes.clear();
        self.ph_prototypes.clear();
        self.recent_births.clear();
        self.harmonic_enter_timer = 0.0;
        self.presence_enter_timer = 0.0;
        self.harmonic_birth_cooldown = 0.0;
        self.silence_timer = 0.0;
    }

    fn age_and_update_live_objects(&mut self, features: &AudioFeatures, dt: f32) {
        for obj in &mut self.live_objects {
            obj.age_secs += dt;
            if obj.kind == SoundKind::Percussive {
                obj.energy *= (-dt / PERC_DECAY_TAU).exp();
            }
            if obj.kind == SoundKind::PercussiveHarmonic
                && obj.visual_shape == 2
                && obj.age_secs > 0.05
            {
                obj.visual_shape = 1;
            }
            if obj.kind != SoundKind::Harmonic && obj.kind != SoundKind::PercussiveHarmonic {
                continue;
            }
            if !features.analysis_ready {
                continue;
            }

            let similarity = cosine_sim_12(&obj.template_chroma, &features.chroma);
            let harmonic_active = features.harmonic_confidence >= HARM_EXIT_THRESHOLD
                && features.pitched_stability >= 0.30
                && similarity >= HARM_TEMPLATE_MATCH_THRESHOLD;
            let presence_active = features.rms >= PRESENCE_MIN_RMS
                && similarity >= PRESENCE_TEMPLATE_MATCH_THRESHOLD
                && (features.harmonic_confidence >= PRESENCE_HARMONIC_FLOOR
                    || features.pitched_stability >= PRESENCE_PITCHED_FLOOR
                    || features.rms >= PRESENCE_STRONG_RMS);
            if harmonic_active || presence_active {
                obj.energy = 1.0;
                obj.release_timer = 0.0;
                obj.anchor_chroma = features.chroma;
                ema_update_12(&mut obj.template_chroma, &features.chroma);
            } else {
                obj.release_timer += dt;
            }
        }
    }

    fn commit_ready_candidates(&mut self) {
        let ready_after = self.elapsed_secs - PENDING_LOOKAHEAD_SECS;
        let pending = std::mem::take(&mut self.pending_candidates);
        let (ready, still_pending): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|candidate| candidate.started_secs <= ready_after);
        self.pending_candidates = still_pending;

        for candidate in ready {
            self.commit_candidate(candidate);
        }
    }

    fn handle_harmonic_presence(&mut self, features: &AudioFeatures, dt: f32) {
        if !features.analysis_ready {
            self.harmonic_enter_timer = 0.0;
            return;
        }

        let presence = harmonic_presence_score(features);
        if presence >= HARM_ENTER_THRESHOLD {
            self.harmonic_enter_timer += dt;
        } else {
            self.harmonic_enter_timer = 0.0;
        }

        if self.harmonic_enter_timer < HARM_ENTER_HOLD_SECS || self.harmonic_birth_cooldown > 0.0 {
            return;
        }
        self.harmonic_enter_timer = 0.0;

        if self.refresh_matching_live_harmonic(features) {
            self.harmonic_birth_cooldown = HARM_BIRTH_COOLDOWN_SECS;
            return;
        }

        self.spawn_harmonic(features);
        self.harmonic_birth_cooldown = HARM_BIRTH_COOLDOWN_SECS;
    }

    fn ensure_sound_presence(&mut self, features: &AudioFeatures, dt: f32) {
        if !features.analysis_ready
            || !self.live_objects.is_empty()
            || !self.pending_candidates.is_empty()
            || !self.sound_is_present(features)
        {
            self.presence_enter_timer = 0.0;
            return;
        }

        self.presence_enter_timer += dt;
        if self.presence_enter_timer < PRESENCE_ENTER_SECS {
            return;
        }

        self.presence_enter_timer = 0.0;
        self.spawn_presence_object(features);
    }

    fn should_queue_onset(&self, features: &AudioFeatures) -> bool {
        if !features.analysis_ready {
            return false;
        }
        let strong_onset = features.is_onset
            && features.onset_confidence >= MIN_ONSET_CONFIDENCE
            && features.percussive_confidence >= MIN_PERCUSSIVE_CONFIDENCE * 0.75;
        let strong_beat = ((features.kick && features.kick_strength > MIN_BEAT_STRENGTH)
            || (features.snare && features.snare_strength > MIN_BEAT_STRENGTH)
            || (features.hihat && features.hihat_strength > MIN_BEAT_STRENGTH))
            && features.onset_confidence >= 0.20;
        strong_onset || strong_beat
    }

    fn queue_onset_candidate(&mut self, features: &AudioFeatures) {
        let flux = features.band_flux_delta;
        if self.pending_candidates.iter().any(|candidate| {
            self.elapsed_secs - candidate.started_secs < PENDING_DEBOUNCE_SECS
                && cosine_dist_6(&candidate.onset_flux, &flux) < PENDING_DEBOUNCE_DIST
        }) {
            return;
        }
        self.pending_candidates
            .push(PendingCandidate::new(features, self.elapsed_secs));
    }

    fn sound_is_present(&self, features: &AudioFeatures) -> bool {
        features.rms >= PRESENCE_MIN_RMS
            && (features.harmonic_confidence >= PRESENCE_HARMONIC_FLOOR
                || features.percussive_confidence >= MIN_PERCUSSIVE_CONFIDENCE * 0.55
                || features.pitched_stability >= PRESENCE_PITCHED_FLOOR
                || features.rms >= PRESENCE_STRONG_RMS)
    }

    fn commit_candidate(&mut self, candidate: PendingCandidate) {
        if candidate.onset_confidence < MIN_ONSET_CONFIDENCE {
            debug_log("dropping onset candidate: low onset confidence");
            return;
        }

        if candidate.peak_harmonic_confidence >= MIN_PH_HARMONIC_CONFIDENCE
            && candidate.peak_percussive_confidence >= MIN_PERCUSSIVE_CONFIDENCE * 0.80
        {
            self.commit_ph_candidate(candidate);
        } else if candidate.peak_percussive_confidence >= MIN_PERCUSSIVE_CONFIDENCE {
            self.commit_percussive_candidate(candidate);
        } else {
            debug_log("dropping onset candidate: insufficient classification confidence");
        }
    }

    fn commit_percussive_candidate(&mut self, candidate: PendingCandidate) {
        let decision = self.select_perc_prototype(&candidate.onset_flux);
        if let Some(cluster_id) = self.preview_cluster_id(&self.perc_prototypes, decision) {
            if self.recent_duplicate_percussive(cluster_id, &candidate.onset_flux) {
                debug_log("suppressing duplicate percussive birth");
                return;
            }
        }

        let cluster_id = self.commit_perc_prototype(decision, &candidate.onset_flux);
        let energy = (0.45 + candidate.onset_confidence * 1.25).clamp(0.45, 1.6);
        self.spawn_object(
            SoundKind::Percussive,
            cluster_id,
            candidate.brightness,
            candidate.percussive_y,
            energy,
            &candidate.latest_chroma,
        );
        self.record_birth(SoundKind::Percussive, cluster_id, candidate.onset_flux);
    }

    fn commit_ph_candidate(&mut self, candidate: PendingCandidate) {
        if let Some((idx, dist)) = nearest_ph_by_flux(&self.ph_prototypes, &candidate.onset_flux) {
            let proto = &self.ph_prototypes[idx];
            if proto.count >= TRUSTED_PROTOTYPE_COUNT && dist < FAST_FLUX_THRESHOLD {
                let cluster_id = proto.cluster_id;
                self.update_existing_ph_proto(
                    idx,
                    &candidate.onset_flux,
                    &candidate.latest_harm,
                    candidate.latest_pitch_y,
                );
                if self.refresh_live_cluster(
                    cluster_id,
                    SoundKind::PercussiveHarmonic,
                    candidate.latest_pitch_y,
                    &candidate.latest_chroma,
                ) {
                    return;
                }
                if self.recent_duplicate_kind(
                    SoundKind::PercussiveHarmonic,
                    cluster_id,
                    HARM_DUPLICATE_SECS,
                ) {
                    debug_log("suppressing duplicate PH fast-path birth");
                    return;
                }
                let energy = (0.55 + candidate.onset_confidence).clamp(0.55, 1.8);
                self.spawn_object(
                    SoundKind::PercussiveHarmonic,
                    cluster_id,
                    candidate.brightness,
                    candidate.latest_pitch_y,
                    energy,
                    &candidate.latest_chroma,
                );
                self.record_birth(
                    SoundKind::PercussiveHarmonic,
                    cluster_id,
                    candidate.onset_flux,
                );
                return;
            }
        }

        let decision = self.select_ph_prototype(&candidate.onset_flux, &candidate.latest_harm);
        let cluster_id = self.commit_ph_prototype(
            decision,
            &candidate.onset_flux,
            &candidate.latest_harm,
            candidate.latest_pitch_y,
        );
        if self.refresh_live_cluster(
            cluster_id,
            SoundKind::PercussiveHarmonic,
            candidate.latest_pitch_y,
            &candidate.latest_chroma,
        ) {
            return;
        }
        if self.recent_duplicate_kind(
            SoundKind::PercussiveHarmonic,
            cluster_id,
            HARM_DUPLICATE_SECS,
        ) {
            debug_log("suppressing duplicate PH birth");
            return;
        }
        let energy = (0.55 + candidate.onset_confidence).clamp(0.55, 1.8);
        self.spawn_object(
            SoundKind::PercussiveHarmonic,
            cluster_id,
            candidate.brightness,
            candidate.latest_pitch_y,
            energy,
            &candidate.latest_chroma,
        );
        self.record_birth(
            SoundKind::PercussiveHarmonic,
            cluster_id,
            candidate.onset_flux,
        );
    }

    fn spawn_harmonic(&mut self, features: &AudioFeatures) {
        let harm = harm_fingerprint(features);
        let ax = features.brightness;
        let ay = chroma_y(features);
        let decision = self.select_harm_prototype(&harm);
        let cluster_id = self.commit_harm_prototype(decision, &harm, ay);
        if self.refresh_live_cluster(cluster_id, SoundKind::Harmonic, ay, &features.chroma) {
            return;
        }
        if self.recent_duplicate_kind(SoundKind::Harmonic, cluster_id, HARM_DUPLICATE_SECS) {
            debug_log("suppressing duplicate harmonic birth");
            return;
        }
        self.spawn_object(
            SoundKind::Harmonic,
            cluster_id,
            ax,
            ay,
            1.0,
            &features.chroma,
        );
        self.record_birth(SoundKind::Harmonic, cluster_id, [0.0; 6]);
    }

    fn spawn_presence_object(&mut self, features: &AudioFeatures) {
        if self.refresh_matching_live_harmonic(features) {
            return;
        }
        self.spawn_harmonic(features);
    }

    fn refresh_matching_live_harmonic(&mut self, features: &AudioFeatures) -> bool {
        let ay = chroma_y(features);
        let best_match = self
            .live_objects
            .iter()
            .enumerate()
            .filter(|(_, obj)| {
                obj.kind == SoundKind::Harmonic || obj.kind == SoundKind::PercussiveHarmonic
            })
            .map(|(idx, obj)| (idx, cosine_sim_12(&obj.template_chroma, &features.chroma)))
            .filter(|(_, sim)| *sim >= LIVE_HARMONIC_MATCH_THRESHOLD)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(idx, _)| idx);

        if let Some(idx) = best_match {
            let obj = &mut self.live_objects[idx];
            obj.energy = 1.0;
            obj.release_timer = 0.0;
            obj.acoustic_y = ay;
            obj.anchor_chroma = features.chroma;
            ema_update_12(&mut obj.template_chroma, &features.chroma);
            return true;
        }
        false
    }

    fn refresh_live_cluster(
        &mut self,
        cluster_id: usize,
        desired_kind: SoundKind,
        acoustic_y: f32,
        chroma: &[f32; 12],
    ) -> bool {
        let (hue, size, _) = visual_params(cluster_id);
        if let Some(obj) = self.live_objects.iter_mut().find(|obj| {
            obj.cluster_id == cluster_id
                && (obj.kind == SoundKind::Harmonic || obj.kind == SoundKind::PercussiveHarmonic)
        }) {
            obj.kind = desired_kind;
            obj.energy = 1.0;
            obj.release_timer = 0.0;
            obj.acoustic_y = acoustic_y;
            obj.visual_hue = hue;
            obj.visual_size = size;
            if desired_kind == SoundKind::PercussiveHarmonic {
                obj.visual_shape = 2;
                obj.age_secs = 0.0;
            }
            obj.anchor_chroma = *chroma;
            ema_update_12(&mut obj.template_chroma, chroma);
            return true;
        }
        false
    }

    fn select_perc_prototype(&self, flux: &[f32; 6]) -> PrototypeDecision {
        match nearest_perc(&self.perc_prototypes, flux) {
            Some((idx, dist))
                if prototype_matches(
                    &self.perc_prototypes[idx],
                    dist,
                    PERC_TRUSTED_THRESHOLD,
                    PERC_WEAK_THRESHOLD,
                ) =>
            {
                PrototypeDecision::Reuse(idx)
            }
            _ => PrototypeDecision::CreateNew,
        }
    }

    fn select_harm_prototype(&self, harm: &[f32; 15]) -> PrototypeDecision {
        match nearest_harm(&self.harm_prototypes, harm) {
            Some((idx, dist))
                if prototype_matches(
                    &self.harm_prototypes[idx],
                    dist,
                    HARM_TRUSTED_THRESHOLD,
                    HARM_WEAK_THRESHOLD,
                ) =>
            {
                PrototypeDecision::Reuse(idx)
            }
            _ => PrototypeDecision::CreateNew,
        }
    }

    fn select_ph_prototype(&self, flux: &[f32; 6], harm: &[f32; 15]) -> PrototypeDecision {
        match nearest_ph(&self.ph_prototypes, flux, harm) {
            Some((idx, dist))
                if prototype_matches(
                    &self.ph_prototypes[idx],
                    dist,
                    PH_TRUSTED_THRESHOLD,
                    PH_WEAK_THRESHOLD,
                ) =>
            {
                PrototypeDecision::Reuse(idx)
            }
            _ => PrototypeDecision::CreateNew,
        }
    }

    fn preview_cluster_id(&self, pool: &[Prototype], decision: PrototypeDecision) -> Option<usize> {
        match decision {
            PrototypeDecision::Reuse(idx) => Some(pool[idx].cluster_id),
            PrototypeDecision::CreateNew => None,
        }
    }

    fn commit_perc_prototype(&mut self, decision: PrototypeDecision, flux: &[f32; 6]) -> usize {
        match decision {
            PrototypeDecision::Reuse(idx) => {
                let proto = &mut self.perc_prototypes[idx];
                proto.count += 1;
                proto.last_seen_secs = self.elapsed_secs;
                if let Fingerprint::Perc(ref mut stored) = proto.fp {
                    ema_update_6(stored, flux);
                }
                proto.cluster_id
            }
            PrototypeDecision::CreateNew => {
                let cluster_id = self.alloc_id();
                if self.perc_prototypes.len() >= MAX_PROTOTYPES {
                    evict(&mut self.perc_prototypes, self.elapsed_secs);
                }
                self.perc_prototypes.push(Prototype {
                    fp: Fingerprint::Perc(*flux),
                    cluster_id,
                    last_seen_secs: self.elapsed_secs,
                    count: 1,
                    pitch_com: None,
                });
                cluster_id
            }
        }
    }

    fn commit_harm_prototype(
        &mut self,
        decision: PrototypeDecision,
        harm: &[f32; 15],
        ay: f32,
    ) -> usize {
        match decision {
            PrototypeDecision::Reuse(idx) => {
                let proto = &mut self.harm_prototypes[idx];
                proto.count += 1;
                proto.last_seen_secs = self.elapsed_secs;
                proto.pitch_com = Some(ay);
                if let Fingerprint::Harm(ref mut stored) = proto.fp {
                    ema_update_15(stored, harm);
                }
                proto.cluster_id
            }
            PrototypeDecision::CreateNew => {
                let cluster_id = self.alloc_id();
                if self.harm_prototypes.len() >= MAX_PROTOTYPES {
                    evict(&mut self.harm_prototypes, self.elapsed_secs);
                }
                self.harm_prototypes.push(Prototype {
                    fp: Fingerprint::Harm(*harm),
                    cluster_id,
                    last_seen_secs: self.elapsed_secs,
                    count: 1,
                    pitch_com: Some(ay),
                });
                cluster_id
            }
        }
    }

    fn commit_ph_prototype(
        &mut self,
        decision: PrototypeDecision,
        flux: &[f32; 6],
        harm: &[f32; 15],
        ay: f32,
    ) -> usize {
        match decision {
            PrototypeDecision::Reuse(idx) => {
                self.update_existing_ph_proto(idx, flux, harm, ay);
                self.ph_prototypes[idx].cluster_id
            }
            PrototypeDecision::CreateNew => {
                let cluster_id = self.alloc_id();
                if self.ph_prototypes.len() >= MAX_PROTOTYPES {
                    evict(&mut self.ph_prototypes, self.elapsed_secs);
                }
                self.ph_prototypes.push(Prototype {
                    fp: Fingerprint::Ph {
                        flux: *flux,
                        harm: *harm,
                    },
                    cluster_id,
                    last_seen_secs: self.elapsed_secs,
                    count: 1,
                    pitch_com: Some(ay),
                });
                cluster_id
            }
        }
    }

    fn update_existing_ph_proto(&mut self, idx: usize, flux: &[f32; 6], harm: &[f32; 15], ay: f32) {
        let proto = &mut self.ph_prototypes[idx];
        proto.count += 1;
        proto.last_seen_secs = self.elapsed_secs;
        proto.pitch_com = Some(ay);
        if let Fingerprint::Ph {
            flux: ref mut stored_flux,
            harm: ref mut stored_harm,
        } = proto.fp
        {
            ema_update_6(stored_flux, flux);
            ema_update_15(stored_harm, harm);
        }
    }

    fn recent_duplicate_percussive(&self, cluster_id: usize, flux: &[f32; 6]) -> bool {
        if self.live_objects.iter().any(|obj| {
            obj.kind == SoundKind::Percussive
                && obj.cluster_id == cluster_id
                && obj.age_secs < PERC_DUPLICATE_SECS
        }) {
            return true;
        }

        self.recent_births.iter().rev().any(|birth| {
            birth.kind == SoundKind::Percussive
                && birth.cluster_id == cluster_id
                && self.elapsed_secs - birth.at_secs < PERC_DUPLICATE_SECS
                && cosine_dist_6(&birth.flux, flux) < PERC_DUPLICATE_DIST
        })
    }

    fn recent_duplicate_kind(&self, kind: SoundKind, cluster_id: usize, within_secs: f32) -> bool {
        self.recent_births.iter().rev().any(|birth| {
            birth.kind == kind
                && birth.cluster_id == cluster_id
                && self.elapsed_secs - birth.at_secs < within_secs
        })
    }

    fn record_birth(&mut self, kind: SoundKind, cluster_id: usize, flux: [f32; 6]) {
        self.recent_births.push_back(RecentBirth {
            kind,
            cluster_id,
            at_secs: self.elapsed_secs,
            flux,
        });
        self.prune_recent_births();
    }

    fn prune_recent_births(&mut self) {
        while self
            .recent_births
            .front()
            .is_some_and(|birth| self.elapsed_secs - birth.at_secs > RECENT_BIRTH_KEEP_SECS)
        {
            self.recent_births.pop_front();
        }
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
        let visual_shape = if kind == SoundKind::PercussiveHarmonic {
            2
        } else {
            shape
        };
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
            template_chroma: *chroma,
            release_timer: 0.0,
        });
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_cluster_id;
        self.next_cluster_id += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_features() -> AudioFeatures {
        let mut features = AudioFeatures::default();
        features.analysis_ready = true;
        features.rms = 0.10;
        features.brightness = 0.45;
        features.width = 0.35;
        features.roughness = 0.20;
        features.sustain = 0.55;
        features
    }

    fn perc_features(flux: [f32; 6]) -> AudioFeatures {
        let mut features = base_features();
        features.onset_strength = 1.6;
        features.onset_confidence = 0.82;
        features.is_onset = true;
        features.percussive_confidence = 0.78;
        features.harmonic_confidence = 0.18;
        features.band_flux_delta = flux;
        features.band_flux_peakiness = 0.75;
        features.chroma = [
            0.08, 0.06, 0.05, 0.06, 0.07, 0.10, 0.09, 0.11, 0.10, 0.10, 0.09, 0.09,
        ];
        features
    }

    fn ph_features(flux: [f32; 6]) -> AudioFeatures {
        let mut features = perc_features(flux);
        features.harmonic_confidence = 0.74;
        features.pitched_stability = 0.72;
        features.chroma = [
            0.28, 0.04, 0.03, 0.03, 0.05, 0.18, 0.14, 0.08, 0.05, 0.05, 0.04, 0.03,
        ];
        features.chroma_flux = [
            0.20, 0.0, 0.0, 0.0, 0.0, 0.10, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        features
    }

    fn harmonic_features() -> AudioFeatures {
        let mut features = base_features();
        features.onset_strength = 0.2;
        features.onset_confidence = 0.0;
        features.is_onset = false;
        features.harmonic_confidence = 0.84;
        features.percussive_confidence = 0.10;
        features.pitched_stability = 0.86;
        features.sustain = 0.88;
        features.chroma = [
            0.32, 0.04, 0.03, 0.03, 0.04, 0.24, 0.18, 0.05, 0.03, 0.02, 0.01, 0.01,
        ];
        features.chroma_flux = [
            0.08, 0.0, 0.0, 0.0, 0.0, 0.06, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        features
    }

    fn ambient_features() -> AudioFeatures {
        let mut features = base_features();
        features.rms = 0.05;
        features.onset_strength = 0.12;
        features.onset_confidence = 0.0;
        features.is_onset = false;
        features.harmonic_confidence = 0.14;
        features.percussive_confidence = 0.24;
        features.pitched_stability = 0.17;
        features.sustain = 0.82;
        features.chroma = [
            0.10, 0.08, 0.09, 0.08, 0.07, 0.10, 0.09, 0.08, 0.08, 0.08, 0.07, 0.08,
        ];
        features.chroma_flux = [0.01; 12];
        features
    }

    #[test]
    fn stable_percussive_onset_commits_once() {
        let mut detector = SoundObjectDetector::new();
        let flux = [0.72, 0.18, 0.05, 0.03, 0.01, 0.01];

        detector.process(&perc_features(flux), 0.03);
        assert!(detector.live_objects.is_empty());

        let mut release = perc_features(flux);
        release.is_onset = false;
        release.onset_confidence = 0.1;
        release.percussive_confidence = 0.55;
        detector.process(&release, 0.03);
        detector.process(&release, 0.03);

        assert_eq!(detector.live_objects.len(), 1);
        assert_eq!(detector.live_objects[0].kind, SoundKind::Percussive);
    }

    #[test]
    fn ph_candidate_commits_without_provisional_perc() {
        let mut detector = SoundObjectDetector::new();
        let flux = [0.55, 0.14, 0.06, 0.05, 0.10, 0.10];

        detector.process(&ph_features(flux), 0.03);
        assert!(detector.live_objects.is_empty());
        assert_eq!(detector.perc_prototypes.len(), 0);

        let mut settle = ph_features(flux);
        settle.is_onset = false;
        settle.onset_confidence = 0.15;
        detector.process(&settle, 0.03);
        detector.process(&settle, 0.03);

        assert_eq!(detector.live_objects.len(), 1);
        assert_eq!(detector.live_objects[0].kind, SoundKind::PercussiveHarmonic);
        assert_eq!(detector.perc_prototypes.len(), 0);
        assert_eq!(detector.ph_prototypes.len(), 1);
    }

    #[test]
    fn harmonic_object_persists_and_does_not_duplicate_spawn() {
        let mut detector = SoundObjectDetector::new();
        let base = harmonic_features();

        for _ in 0..6 {
            detector.process(&base, 0.02);
        }

        assert_eq!(detector.live_objects.len(), 1);
        assert_eq!(detector.live_objects[0].kind, SoundKind::Harmonic);

        let mut shifted = harmonic_features();
        shifted.chroma = [
            0.29, 0.05, 0.03, 0.03, 0.04, 0.22, 0.18, 0.08, 0.04, 0.02, 0.01, 0.01,
        ];
        shifted.chroma_flux = [
            0.04, 0.0, 0.0, 0.0, 0.0, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        for _ in 0..20 {
            detector.process(&shifted, 0.02);
        }

        assert_eq!(detector.live_objects.len(), 1);
        assert_eq!(detector.live_objects[0].kind, SoundKind::Harmonic);
        assert!(detector.live_objects[0].release_timer < 0.05);
    }

    #[test]
    fn duplicate_percussive_birth_is_suppressed_inside_short_window() {
        let mut detector = SoundObjectDetector::new();
        let flux = [0.75, 0.16, 0.04, 0.03, 0.01, 0.01];

        detector.process(&perc_features(flux), 0.03);
        let mut settle = perc_features(flux);
        settle.is_onset = false;
        settle.onset_confidence = 0.05;
        detector.process(&settle, 0.03);
        detector.process(&settle, 0.03);
        assert_eq!(detector.live_objects.len(), 1);

        detector.process(&perc_features(flux), 0.03);
        detector.process(&settle, 0.03);
        detector.process(&settle, 0.03);

        assert_eq!(detector.live_objects.len(), 1);
    }

    #[test]
    fn continuous_sound_gets_presence_object() {
        let mut detector = SoundObjectDetector::new();
        let ambient = ambient_features();

        for _ in 0..4 {
            detector.process(&ambient, 0.02);
        }

        assert_eq!(detector.live_objects.len(), 1);
        assert_eq!(detector.live_objects[0].kind, SoundKind::Harmonic);

        for _ in 0..20 {
            detector.process(&ambient, 0.02);
        }

        assert_eq!(detector.live_objects.len(), 1);
    }
}
