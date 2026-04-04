#![allow(dead_code)]

use crate::analysis::{
    AudioFeatures, Analyser, CQT_BINS, cqt_bin_to_midi_note, midi_note_to_cqt_bin,
};
use crate::audio::FFT_SIZE;
use crate::objects::{SoundKind, SoundObject, SoundObjectDetector};
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const DATASET_DIR: &str = "midi";
pub const TRANSCRIPTION_HOP_SIZE: usize = FFT_SIZE / 4;
const MAX_PITCH_CANDIDATES: usize = 6;
const MIN_CQT_PEAK: f32 = 0.08;
const MIN_RELATIVE_CQT_PEAK: f32 = 0.30;
const MIN_NOTE_CONFIDENCE: f32 = 0.14;
const NOTE_RELOCK_DISTANCE: i16 = 2;
const NOTE_SPLIT_DISTANCE: i16 = 3;
const NOTE_RETRIGGER_ONSET_CONFIDENCE: f32 = 0.20;
const NOTE_RETRIGGER_MIN_SECS: f32 = 0.18;
const NOTE_RELEASE_HOLD_SECS: f32 = 0.12;
const START_TOLERANCE_SECS: f32 = 0.18;
const END_TOLERANCE_SECS: f32 = 0.30;
const START_HALF_LIFE_SECS: f32 = 0.050;
const END_HALF_LIFE_MIN_SECS: f32 = 0.060;
const END_HALF_LIFE_MAX_SECS: f32 = 0.180;
const END_HALF_LIFE_DURATION_FACTOR: f32 = 0.35;
const PITCH_HALF_LIFE_SEMITONES: f32 = 1.0;
const STREAM_NOTE_DISTANCE_NORM: f32 = 24.0;
const MIX_HEADROOM: f32 = 0.85;
const DEFAULT_TEMPO_US_PER_QUARTER: u32 = 500_000;
const MAX_SIMPLE_POLYPHONY: usize = 2;
const MIN_SIMPLE_NOTES: usize = 1;
const MAX_SIMPLE_NOTES: usize = 64;
const PERCUSSION_KEYWORDS: &[&str] = &["hihat", "snare", "perc", "stomp", "roll", "kick", "drum"];
const EXCLUDED_FIXTURE_KEYWORDS: &[&str] = &["choir", "vocals", "pad", "chord", "saxophone"];
const SYNTH_SPAWN_BASE: usize = 1_000_000;
const SYNTH_CLUSTER_BASE: usize = 2_000_000;
const STANDARD_MIX_COUNT: usize = 10;
const STANDARD_MIX_SIZES: &[usize] = &[2, 3];
pub const TRANSCRIPTION_MIDI_LOW: u8 = 21;
pub const TRANSCRIPTION_MIDI_HIGH: u8 = 104;
const MAX_POLYPHONIC_CANDIDATES: usize = 6;
const HARMONIC_START_THRESHOLD: f32 = 0.10;
const HARMONIC_STOP_THRESHOLD: f32 = 0.04;
const HARMONIC_MIN_NOTE_SECS: f32 = 0.05;
const HARMONIC_RETRIGGER_GAP_SECS: f32 = 0.10;
const DRUM_NOTE_LENGTH_SECS: f32 = 0.09;
const DRUM_COOLDOWN_SECS: f32 = 0.05;
const DRUM_KICK_NOTE: u8 = 36;
const DRUM_SNARE_NOTE: u8 = 38;
const DRUM_HIHAT_NOTE: u8 = 42;
const NOTE_TEMPLATE_SIGMA_BINS: f32 = 0.65;
const NOTE_TEMPLATE_HARMONICS: &[(f32, f32)] = &[
    (1.0, 1.0),
    (2.0, 0.65),
    (3.0, 0.45),
    (4.0, 0.30),
    (5.0, 0.20),
];
const FIXTURE_PREFERENCE_ORDER: &[&str] = &[
    "16_Scary Nights_Dm_140bpm_Bells",
    "09_Rocket Talk_F#m_142bpm_Synth Keys",
    "RS_VAH1_Kit3_drop_loop_lead_dry_123bpm_Bm",
    "SS_BV2_Bellakiar_104_G#min_Piano",
    "RS_VAH1_Kit2_drop_loop_piano2_dry_122bpm_Em",
    "Future Samples_Jazz Cafe_Loop 40_85 BPM_F",
    "CLIK_05_142BPM_Dm_Guitar",
    "Kalimbas",
];

#[derive(Debug, Clone)]
pub struct DatasetPair {
    pub stem: String,
    pub wav_path: PathBuf,
    pub midi_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct WavClip {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct GroundTruthNote {
    pub source_id: usize,
    pub midi_note: u8,
    pub start_secs: f32,
    pub end_secs: f32,
}

#[derive(Debug, Clone)]
pub struct LoadedPair {
    pub pair: DatasetPair,
    pub clip: WavClip,
    pub notes: Vec<GroundTruthNote>,
    pub max_polyphony: usize,
}

#[derive(Debug, Clone)]
pub struct TranscribedNote {
    pub stream_id: usize,
    pub midi_note: u8,
    pub start_secs: f32,
    pub end_secs: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct CompletedTranscribedNote {
    pub id: usize,
    pub note: TranscribedNote,
}

#[derive(Debug, Clone)]
pub struct ActiveTranscribedNote {
    pub id: usize,
    pub stream_id: usize,
    pub midi_note: u8,
    pub start_secs: f32,
    pub end_secs: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Default)]
pub struct NoteMetrics {
    pub predicted_notes: usize,
    pub reference_notes: usize,
    pub matched_onsets: usize,
    pub matched_notes: usize,
    pub matched_pairs: usize,
    pub matched_similarity: f32,
    pub pitch_similarity: f32,
    pub start_similarity: f32,
    pub end_similarity: f32,
    pub total_pitch_error_semitones: f32,
    pub total_start_error_secs: f32,
    pub total_end_error_secs: f32,
}

impl NoteMetrics {
    pub fn onset_recall(&self) -> f32 {
        ratio(self.matched_onsets, self.reference_notes)
    }

    pub fn onset_precision(&self) -> f32 {
        ratio(self.matched_onsets, self.predicted_notes)
    }

    pub fn full_note_recall(&self) -> f32 {
        ratio(self.matched_notes, self.reference_notes)
    }

    pub fn fuzzy_precision(&self) -> f32 {
        normalized_similarity(self.matched_similarity, self.predicted_notes)
    }

    pub fn fuzzy_recall(&self) -> f32 {
        normalized_similarity(self.matched_similarity, self.reference_notes)
    }

    pub fn fuzzy_score_percent(&self) -> f32 {
        100.0 * harmonic_mean(self.fuzzy_precision(), self.fuzzy_recall())
    }

    pub fn mean_pitch_similarity(&self) -> f32 {
        normalized_similarity(self.pitch_similarity, self.matched_pairs)
    }

    pub fn mean_start_similarity(&self) -> f32 {
        normalized_similarity(self.start_similarity, self.matched_pairs)
    }

    pub fn mean_end_similarity(&self) -> f32 {
        normalized_similarity(self.end_similarity, self.matched_pairs)
    }

    pub fn mean_pitch_error_semitones(&self) -> f32 {
        mean_error(self.total_pitch_error_semitones, self.matched_pairs)
    }

    pub fn mean_start_error_ms(&self) -> f32 {
        1000.0 * mean_error(self.total_start_error_secs, self.matched_pairs)
    }

    pub fn mean_end_error_ms(&self) -> f32 {
        1000.0 * mean_error(self.total_end_error_secs, self.matched_pairs)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SourceEval {
    pub source_id: usize,
    pub stream_id: Option<usize>,
    pub metrics: NoteMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct EvalSummary {
    pub per_source: Vec<SourceEval>,
    pub total: NoteMetrics,
}

#[derive(Debug, Clone, Copy)]
struct PitchCandidate {
    midi_note: u8,
    confidence: f32,
    bin: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct MatchAssessment {
    similarity: f32,
    pitch_similarity: f32,
    start_similarity: f32,
    end_similarity: f32,
    pitch_error_semitones: f32,
    start_error_secs: f32,
    end_error_secs: f32,
    onset_match: bool,
    full_note_match: bool,
}

#[derive(Clone, Copy)]
enum AlignmentStep {
    Match,
    SkipPredicted,
    SkipReference,
}

#[derive(Debug, Clone)]
struct ActiveNote {
    spawn_id: usize,
    cluster_id: usize,
    stream_id: usize,
    midi_note: u8,
    start_secs: f32,
    last_seen_secs: f32,
    confidence: f32,
    brightness: f32,
}

#[derive(Debug, Clone)]
struct StreamState {
    id: usize,
    note_center: f32,
    brightness_center: f32,
    last_event_secs: f32,
}

#[derive(Debug, Clone, Copy)]
struct NoteActivation {
    midi_note: u8,
    salience: f32,
    brightness: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveKind {
    Harmonic,
    Percussive,
}

#[derive(Debug, Clone)]
struct RealtimeActiveNote {
    id: usize,
    stream_id: usize,
    midi_note: u8,
    start_secs: f32,
    last_seen_secs: f32,
    confidence: f32,
    brightness: f32,
    kind: ActiveKind,
}

struct RealtimeTranscriber {
    analyser: Analyser,
    sample_rate: u32,
    window: VecDeque<f32>,
    elapsed_secs: f32,
    note_energy_ema: [f32; 128],
    note_cooldown_secs: [f32; 128],
    active_notes: Vec<RealtimeActiveNote>,
    finished_notes: Vec<CompletedTranscribedNote>,
    streams: HashMap<usize, StreamState>,
    next_stream_id: usize,
    next_note_id: usize,
    max_streams: usize,
    drum_cooldowns: [f32; 3],
}

pub struct StreamingTranscriber {
    inner: RealtimeTranscriber,
}

pub struct OnlineNoteTranscriber {
    analyser: Analyser,
    detector: SoundObjectDetector,
    sample_rate: u32,
    window: VecDeque<f32>,
    elapsed_secs: f32,
    active_notes: Vec<ActiveNote>,
    finished_notes: Vec<TranscribedNote>,
    cluster_to_stream: HashMap<usize, usize>,
    streams: HashMap<usize, StreamState>,
    next_stream_id: usize,
    max_streams: usize,
}

impl OnlineNoteTranscriber {
    pub fn new(sample_rate: u32, max_streams: usize) -> Self {
        Self {
            analyser: Analyser::new(sample_rate),
            detector: SoundObjectDetector::new(),
            sample_rate,
            window: VecDeque::from(vec![0.0; FFT_SIZE]),
            elapsed_secs: 0.0,
            active_notes: Vec::new(),
            finished_notes: Vec::new(),
            cluster_to_stream: HashMap::new(),
            streams: HashMap::new(),
            next_stream_id: 0,
            max_streams: max_streams.max(1),
        }
    }

    pub fn process_block(&mut self, block: &[f32]) {
        if block.is_empty() {
            return;
        }

        for &sample in block {
            self.window.pop_front();
            self.window.push_back(sample);
        }

        let dt = block.len() as f32 / self.sample_rate as f32;
        self.elapsed_secs += dt;
        self.analyser.process(&self.window, dt);
        self.detector.process(&self.analyser.features, dt);
        self.update_notes(dt);
    }

    pub fn finish(mut self) -> Vec<TranscribedNote> {
        let end_secs = self.elapsed_secs;
        for active in self.active_notes.drain(..) {
            self.finished_notes.push(TranscribedNote {
                stream_id: active.stream_id,
                midi_note: active.midi_note,
                start_secs: active.start_secs,
                end_secs,
                confidence: active.confidence,
            });
        }
        self.finished_notes.sort_by(|a, b| {
            a.start_secs
                .partial_cmp(&b.start_secs)
                .unwrap_or(Ordering::Equal)
        });
        self.finished_notes
    }

    fn update_notes(&mut self, dt: f32) {
        let frame_end = self.elapsed_secs;
        let frame_start = (frame_end - dt).max(0.0);
        let features = &self.analyser.features;
        let pitch_candidates = extract_pitch_candidates(&features.cqt);

        let mut live_objects: Vec<SoundObject> = self
            .detector
            .live_objects
            .iter()
            .filter(|obj| {
                matches!(
                    obj.kind,
                    SoundKind::Harmonic | SoundKind::PercussiveHarmonic
                )
            })
            .cloned()
            .collect();
        live_objects.extend(self.synthetic_pitch_objects(&pitch_candidates, live_objects.len()));
        live_objects.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(Ordering::Equal));

        let live_ids: HashSet<usize> = live_objects.iter().map(|obj| obj.spawn_id).collect();
        let mut used_bins = HashSet::new();
        let mut restarts = Vec::new();

        for object in &live_objects {
            if let Some(active_idx) = self
                .active_notes
                .iter()
                .position(|note| note.spawn_id == object.spawn_id)
            {
                let split = self.update_existing_note(
                    active_idx,
                    object,
                    &pitch_candidates,
                    &mut used_bins,
                    frame_start,
                    frame_end,
                );
                if let Some(candidate) = split {
                    restarts.push((object.clone(), candidate));
                }
                continue;
            }

            if let Some(candidate) = select_candidate_for_new(object, &pitch_candidates, &used_bins)
            {
                used_bins.insert(candidate.bin);
                self.start_note_for_object(object, candidate, frame_start);
            }
        }

        for active_idx in (0..self.active_notes.len()).rev() {
            let active = &self.active_notes[active_idx];
            let missing = !live_ids.contains(&active.spawn_id);
            let stale = frame_end - active.last_seen_secs >= NOTE_RELEASE_HOLD_SECS;
            if missing && stale {
                self.finish_active_note(active_idx, frame_end);
            }
        }

        for (object, candidate) in restarts {
            if self
                .active_notes
                .iter()
                .any(|note| note.spawn_id == object.spawn_id)
            {
                continue;
            }
            if used_bins.insert(candidate.bin) {
                self.start_note_for_object(&object, candidate, frame_start);
            }
        }
    }

    fn synthetic_pitch_objects(
        &self,
        pitch_candidates: &[PitchCandidate],
        existing_live_count: usize,
    ) -> Vec<SoundObject> {
        let features = &self.analyser.features;
        if !features.analysis_ready
            || features.rms < 0.006
            || pitch_candidates.is_empty()
            || (features.harmonic_confidence < 0.10 && features.pitched_stability < 0.12)
        {
            return Vec::new();
        }

        let target_count = if existing_live_count == 0 {
            self.max_streams.max(1)
        } else {
            self.max_streams.saturating_sub(existing_live_count)
        };
        if target_count == 0 {
            return Vec::new();
        }

        pitch_candidates
            .iter()
            .copied()
            .filter(|candidate| candidate.confidence >= MIN_NOTE_CONFIDENCE * 0.75)
            .take(target_count)
            .map(|candidate| self.build_synthetic_object(candidate))
            .collect()
    }

    fn build_synthetic_object(&self, candidate: PitchCandidate) -> SoundObject {
        let chroma = self.analyser.features.chroma;
        let brightness = local_pitch_brightness(&self.analyser.features.cqt, candidate.bin);
        let midi_norm = if CQT_BINS <= 1 {
            0.5
        } else {
            candidate.bin as f32 / (CQT_BINS - 1) as f32
        };

        SoundObject {
            kind: SoundKind::Harmonic,
            cluster_id: SYNTH_CLUSTER_BASE + candidate.bin,
            spawn_id: SYNTH_SPAWN_BASE + candidate.bin,
            age_secs: 0.0,
            energy: candidate
                .confidence
                .max(self.analyser.features.harmonic_confidence),
            acoustic_x: brightness,
            acoustic_y: ((candidate.midi_note % 12) as f32 / 12.0).clamp(0.0, 1.0),
            visual_hue: midi_norm,
            visual_size: 0.35 + 0.45 * candidate.confidence,
            visual_shape: 1,
            anchor_chroma: chroma,
            template_chroma: chroma,
            release_timer: 0.0,
        }
    }

    fn update_existing_note(
        &mut self,
        active_idx: usize,
        object: &SoundObject,
        pitch_candidates: &[PitchCandidate],
        used_bins: &mut HashSet<usize>,
        frame_start: f32,
        frame_end: f32,
    ) -> Option<PitchCandidate> {
        let onset_conf = self.analyser.features.onset_confidence;
        let candidate = {
            let active = &self.active_notes[active_idx];
            select_candidate_for_existing(active, object, pitch_candidates, used_bins)
        };

        if let Some(candidate) = candidate {
            let (pitch_delta, active_start_secs, active_confidence) = {
                let active = &self.active_notes[active_idx];
                (
                    semitone_distance(active.midi_note, candidate.midi_note),
                    active.start_secs,
                    active.confidence,
                )
            };
            let should_split = pitch_delta >= NOTE_SPLIT_DISTANCE && onset_conf >= 0.18;
            let should_retrigger = pitch_delta <= NOTE_RELOCK_DISTANCE
                && onset_conf >= NOTE_RETRIGGER_ONSET_CONFIDENCE
                && frame_start - active_start_secs >= NOTE_RETRIGGER_MIN_SECS
                && candidate.confidence >= active_confidence * 0.70;

            if should_split || should_retrigger {
                self.finish_active_note(active_idx, frame_start);
                return Some(candidate);
            }

            let active = &mut self.active_notes[active_idx];
            used_bins.insert(candidate.bin);
            active.last_seen_secs = frame_end;

            if pitch_delta <= NOTE_RELOCK_DISTANCE
                && candidate.confidence >= active.confidence * 0.85
            {
                active.midi_note = candidate.midi_note;
            }
            active.confidence = 0.75 * active.confidence + 0.25 * candidate.confidence;
            active.brightness = 0.85 * active.brightness + 0.15 * object.acoustic_x;
        }

        None
    }

    fn start_note_for_object(
        &mut self,
        object: &SoundObject,
        candidate: PitchCandidate,
        start_secs: f32,
    ) {
        if candidate.confidence < MIN_NOTE_CONFIDENCE {
            return;
        }

        let stream_id =
            self.assign_stream(object.cluster_id, candidate.midi_note, object.acoustic_x);
        self.active_notes.push(ActiveNote {
            spawn_id: object.spawn_id,
            cluster_id: object.cluster_id,
            stream_id,
            midi_note: candidate.midi_note,
            start_secs,
            last_seen_secs: start_secs,
            confidence: candidate.confidence,
            brightness: object.acoustic_x,
        });
    }

    fn finish_active_note(&mut self, active_idx: usize, end_secs: f32) {
        let active = self.active_notes.remove(active_idx);
        let end_secs = end_secs.max(active.start_secs + 1e-3);
        self.finished_notes.push(TranscribedNote {
            stream_id: active.stream_id,
            midi_note: active.midi_note,
            start_secs: active.start_secs,
            end_secs,
            confidence: active.confidence,
        });
    }

    fn assign_stream(&mut self, cluster_id: usize, midi_note: u8, brightness: f32) -> usize {
        if let Some(&stream_id) = self.cluster_to_stream.get(&cluster_id) {
            self.update_stream(stream_id, midi_note, brightness);
            return stream_id;
        }

        let stream_id = if self.streams.len() < self.max_streams {
            let id = self.next_stream_id;
            self.next_stream_id += 1;
            self.streams.insert(
                id,
                StreamState {
                    id,
                    note_center: midi_note as f32,
                    brightness_center: brightness,
                    last_event_secs: self.elapsed_secs,
                },
            );
            id
        } else {
            self.streams
                .iter()
                .min_by(|a, b| {
                    stream_distance(a.1, midi_note, brightness)
                        .partial_cmp(&stream_distance(b.1, midi_note, brightness))
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(&id, _)| id)
                .unwrap()
        };

        self.cluster_to_stream.insert(cluster_id, stream_id);
        self.update_stream(stream_id, midi_note, brightness);
        stream_id
    }

    fn update_stream(&mut self, stream_id: usize, midi_note: u8, brightness: f32) {
        if let Some(stream) = self.streams.get_mut(&stream_id) {
            stream.note_center = 0.8 * stream.note_center + 0.2 * midi_note as f32;
            stream.brightness_center = 0.85 * stream.brightness_center + 0.15 * brightness;
            stream.last_event_secs = self.elapsed_secs;
        }
    }
}

impl RealtimeTranscriber {
    fn new(sample_rate: u32, max_streams: usize) -> Self {
        Self {
            analyser: Analyser::new(sample_rate),
            sample_rate,
            window: VecDeque::from(vec![0.0; FFT_SIZE]),
            elapsed_secs: 0.0,
            note_energy_ema: [0.0; 128],
            note_cooldown_secs: [0.0; 128],
            active_notes: Vec::new(),
            finished_notes: Vec::new(),
            streams: HashMap::new(),
            next_stream_id: 0,
            next_note_id: 0,
            max_streams: max_streams.max(1),
            drum_cooldowns: [0.0; 3],
        }
    }

    fn process_block(&mut self, block: &[f32]) {
        if block.is_empty() {
            return;
        }

        for &sample in block {
            self.window.pop_front();
            self.window.push_back(sample);
        }

        let dt = block.len() as f32 / self.sample_rate as f32;
        self.elapsed_secs += dt;
        self.analyser.process(&self.window, dt);
        self.update_note_transcription(dt);
    }

    fn finish(mut self) -> Vec<TranscribedNote> {
        let end_secs = self.elapsed_secs;
        let remaining_notes: Vec<_> = self.active_notes.drain(..).collect();
        for note in remaining_notes {
            self.push_finished_note(
                note.id,
                note.stream_id,
                note.midi_note,
                note.start_secs,
                end_secs.max(note.start_secs + 1e-3),
                note.confidence,
            );
        }
        self.finished_notes
            .sort_by(|a, b| note_order(&a.note, &b.note));
        self.finished_notes.into_iter().map(|completed| completed.note).collect()
    }

    fn push_finished_note(
        &mut self,
        id: usize,
        stream_id: usize,
        midi_note: u8,
        start_secs: f32,
        end_secs: f32,
        confidence: f32,
    ) {
        self.finished_notes.push(CompletedTranscribedNote {
            id,
            note: TranscribedNote {
                stream_id,
                midi_note,
                start_secs,
                end_secs,
                confidence,
            },
        });
    }

    fn update_note_transcription(&mut self, dt: f32) {
        let frame_end = self.elapsed_secs;
        let frame_start = (frame_end - dt).max(0.0);
        let harmonic_confidence = self.analyser.features.harmonic_confidence;
        let pitched_stability = self.analyser.features.pitched_stability;

        for cooldown in &mut self.note_cooldown_secs {
            *cooldown = (*cooldown - dt).max(0.0);
        }
        for cooldown in &mut self.drum_cooldowns {
            *cooldown = (*cooldown - dt).max(0.0);
        }

        let salience = note_salience(&self.analyser.features.cqt);
        self.update_note_energy_ema(&salience);
        let candidates = select_note_activations(
            &salience,
            &self.note_energy_ema,
            harmonic_confidence,
            pitched_stability,
            self.max_streams,
        );
        self.update_harmonic_notes(&candidates, frame_start, frame_end);
        self.emit_percussive_notes(frame_start);
        self.release_stale_notes(frame_end);
    }

    fn update_note_energy_ema(&mut self, salience: &[f32; 128]) {
        for midi_note in TRANSCRIPTION_MIDI_LOW..=TRANSCRIPTION_MIDI_HIGH {
            let idx = midi_note as usize;
            let target = salience[idx];
            let current = self.note_energy_ema[idx];
            let alpha = if target >= current { 0.35 } else { 0.12 };
            self.note_energy_ema[idx] = current + alpha * (target - current);
        }
    }

    fn update_harmonic_notes(
        &mut self,
        candidates: &[NoteActivation],
        frame_start: f32,
        frame_end: f32,
    ) {
        let onset_confidence = self.analyser.features.onset_confidence;
        let mut matched_ids = HashSet::new();
        let mut restarts = Vec::new();

        for candidate in candidates {
            if let Some(active_idx) = self.find_best_harmonic_match(candidate, &matched_ids) {
                let active = &self.active_notes[active_idx];
                let should_retrigger = onset_confidence >= NOTE_RETRIGGER_ONSET_CONFIDENCE
                    && frame_start - active.start_secs >= HARMONIC_RETRIGGER_GAP_SECS
                    && semitone_distance(active.midi_note, candidate.midi_note)
                        <= NOTE_RELOCK_DISTANCE
                    && candidate.salience >= active.confidence * 0.92;
                if should_retrigger {
                    let stream_id = active.stream_id;
                    self.finish_realtime_note(active_idx, frame_start);
                    restarts.push((*candidate, stream_id));
                    continue;
                }

                let active = &mut self.active_notes[active_idx];
                active.last_seen_secs = frame_end;
                active.confidence = 0.7 * active.confidence + 0.3 * candidate.salience;
                active.brightness = 0.8 * active.brightness + 0.2 * candidate.brightness;
                if semitone_distance(active.midi_note, candidate.midi_note) <= NOTE_RELOCK_DISTANCE
                    && candidate.salience >= active.confidence * 0.75
                {
                    active.midi_note = candidate.midi_note;
                }
                matched_ids.insert(active.id);
                continue;
            }

            if self.should_start_harmonic_note(candidate) {
                self.start_harmonic_note(*candidate, frame_start, None);
            }
        }

        for (candidate, stream_id) in restarts {
            self.start_harmonic_note(candidate, frame_start, Some(stream_id));
        }
    }

    fn should_start_harmonic_note(&self, candidate: &NoteActivation) -> bool {
        let idx = candidate.midi_note as usize;
        let features = &self.analyser.features;
        let already_active = self
            .active_notes
            .iter()
            .any(|note| note.kind == ActiveKind::Harmonic && note.midi_note == candidate.midi_note);
        if already_active || self.note_cooldown_secs[idx] > 0.0 {
            return false;
        }

        let activation = self.note_energy_ema[idx];
        let onset_ok = features.onset_confidence >= 0.10;
        let sustained_ok =
            features.harmonic_confidence >= 0.20 || features.pitched_stability >= 0.22;
        candidate.salience >= HARMONIC_START_THRESHOLD.max(activation * 0.95)
            && (onset_ok || sustained_ok)
    }

    fn start_harmonic_note(
        &mut self,
        candidate: NoteActivation,
        start_secs: f32,
        preferred_stream: Option<usize>,
    ) {
        let stream_id = preferred_stream
            .unwrap_or_else(|| self.assign_stream(candidate.midi_note, candidate.brightness));
        self.active_notes.push(RealtimeActiveNote {
            id: self.next_note_id,
            stream_id,
            midi_note: candidate.midi_note,
            start_secs,
            last_seen_secs: start_secs,
            confidence: candidate.salience,
            brightness: candidate.brightness,
            kind: ActiveKind::Harmonic,
        });
        self.next_note_id += 1;
        self.note_cooldown_secs[candidate.midi_note as usize] = HARMONIC_MIN_NOTE_SECS;
        self.update_stream(stream_id, candidate.midi_note, candidate.brightness);
    }

    fn emit_percussive_notes(&mut self, start_secs: f32) {
        let kick = self.analyser.features.kick;
        let snare = self.analyser.features.snare;
        let hihat = self.analyser.features.hihat;
        let kick_strength = self.analyser.features.kick_strength;
        let snare_strength = self.analyser.features.snare_strength;
        let hihat_strength = self.analyser.features.hihat_strength;
        let brightness = self.analyser.features.brightness;
        let onset_confidence = self.analyser.features.onset_confidence;
        let is_onset = self.analyser.features.is_onset;
        let percussive_confidence = self.analyser.features.percussive_confidence;

        let onset_like = is_onset || kick || snare || hihat || onset_confidence >= 0.24;
        if !onset_like || percussive_confidence < 0.18 {
            return;
        }

        let mut emitted = false;
        if (kick || kick_strength >= 0.10) && self.drum_cooldowns[0] <= 0.0 {
            self.push_drum_note(DRUM_KICK_NOTE, 0.18, start_secs, 0);
            emitted = true;
        }
        if (snare || snare_strength >= 0.10) && self.drum_cooldowns[1] <= 0.0 {
            self.push_drum_note(DRUM_SNARE_NOTE, 0.45, start_secs, 1);
            emitted = true;
        }
        if (hihat || hihat_strength >= 0.08 || brightness >= 0.62) && self.drum_cooldowns[2] <= 0.0
        {
            self.push_drum_note(DRUM_HIHAT_NOTE, 0.82, start_secs, 2);
            emitted = true;
        }

        if !emitted {
            let (note, brightness, class_idx) = if brightness < 0.30 {
                (DRUM_KICK_NOTE, 0.18, 0)
            } else if brightness < 0.62 {
                (DRUM_SNARE_NOTE, 0.45, 1)
            } else {
                (DRUM_HIHAT_NOTE, 0.82, 2)
            };
            if self.drum_cooldowns[class_idx] <= 0.0 {
                self.push_drum_note(note, brightness, start_secs, class_idx);
            }
        }
    }

    fn push_drum_note(
        &mut self,
        midi_note: u8,
        brightness: f32,
        start_secs: f32,
        class_idx: usize,
    ) {
        let stream_id = self.assign_stream(midi_note, brightness);
        let id = self.next_note_id;
        self.next_note_id += 1;
        self.push_finished_note(
            id,
            stream_id,
            midi_note,
            start_secs,
            start_secs + DRUM_NOTE_LENGTH_SECS,
            self.analyser.features.percussive_confidence.max(0.3),
        );
        self.update_stream(stream_id, midi_note, brightness);
        self.drum_cooldowns[class_idx] = DRUM_COOLDOWN_SECS;
    }

    fn release_stale_notes(&mut self, frame_end: f32) {
        for idx in (0..self.active_notes.len()).rev() {
            let note = &self.active_notes[idx];
            if note.kind != ActiveKind::Harmonic {
                continue;
            }
            let activation = self.note_energy_ema[note.midi_note as usize];
            let stale = frame_end - note.last_seen_secs >= NOTE_RELEASE_HOLD_SECS;
            if stale || activation < HARMONIC_STOP_THRESHOLD {
                self.finish_realtime_note(idx, frame_end);
            }
        }
    }

    fn finish_realtime_note(&mut self, note_idx: usize, end_secs: f32) {
        let note = self.active_notes.remove(note_idx);
        self.push_finished_note(
            note.id,
            note.stream_id,
            note.midi_note,
            note.start_secs,
            end_secs.max(note.start_secs + HARMONIC_MIN_NOTE_SECS),
            note.confidence,
        );
    }

    fn find_best_harmonic_match(
        &self,
        candidate: &NoteActivation,
        matched_ids: &HashSet<usize>,
    ) -> Option<usize> {
        self.active_notes
            .iter()
            .enumerate()
            .filter(|(_, note)| {
                note.kind == ActiveKind::Harmonic && !matched_ids.contains(&note.id)
            })
            .filter(|(_, note)| {
                semitone_distance(note.midi_note, candidate.midi_note) <= NOTE_SPLIT_DISTANCE
            })
            .min_by(|a, b| {
                harmonic_note_distance(a.1, candidate)
                    .partial_cmp(&harmonic_note_distance(b.1, candidate))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    fn assign_stream(&mut self, midi_note: u8, brightness: f32) -> usize {
        if let Some((&stream_id, _)) = self.streams.iter().min_by(|a, b| {
            let da = stream_distance(a.1, midi_note, brightness);
            let db = stream_distance(b.1, midi_note, brightness);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        }) {
            if self.streams.len() >= self.max_streams
                || stream_distance(self.streams.get(&stream_id).unwrap(), midi_note, brightness)
                    < 0.55
            {
                return stream_id;
            }
        }

        if self.streams.len() < self.max_streams {
            let id = self.next_stream_id;
            self.next_stream_id += 1;
            self.streams.insert(
                id,
                StreamState {
                    id,
                    note_center: midi_note as f32,
                    brightness_center: brightness,
                    last_event_secs: self.elapsed_secs,
                },
            );
            id
        } else {
            self.streams
                .iter()
                .min_by(|a, b| {
                    stream_distance(a.1, midi_note, brightness)
                        .partial_cmp(&stream_distance(b.1, midi_note, brightness))
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(&id, _)| id)
                .unwrap_or(0)
        }
    }

    fn update_stream(&mut self, stream_id: usize, midi_note: u8, brightness: f32) {
        if let Some(stream) = self.streams.get_mut(&stream_id) {
            stream.note_center = 0.75 * stream.note_center + 0.25 * midi_note as f32;
            stream.brightness_center = 0.8 * stream.brightness_center + 0.2 * brightness;
            stream.last_event_secs = self.elapsed_secs;
        }
    }
}

impl StreamingTranscriber {
    pub fn new(sample_rate: u32, max_streams: usize) -> Self {
        Self {
            inner: RealtimeTranscriber::new(sample_rate, max_streams),
        }
    }

    pub fn process_block(&mut self, block: &[f32]) {
        self.inner.process_block(block);
    }

    pub fn drain_finished_notes(&mut self) -> Vec<CompletedTranscribedNote> {
        self.inner.finished_notes.drain(..).collect()
    }

    pub fn active_notes(&self) -> Vec<ActiveTranscribedNote> {
        self.inner
            .active_notes
            .iter()
            .map(|note| ActiveTranscribedNote {
                id: note.id,
                stream_id: note.stream_id,
                midi_note: note.midi_note,
                start_secs: note.start_secs,
                end_secs: self.inner.elapsed_secs.max(note.start_secs + 1e-3),
                confidence: note.confidence,
            })
            .collect()
    }

    pub fn elapsed_secs(&self) -> f32 {
        self.inner.elapsed_secs
    }

    pub fn features(&self) -> &AudioFeatures {
        &self.inner.analyser.features
    }

    pub fn cqt_history(&self) -> &VecDeque<Vec<f32>> {
        &self.inner.analyser.cqt_history
    }

    pub fn fft_history(&self) -> &VecDeque<Vec<f32>> {
        &self.inner.analyser.fft_history
    }
}

pub fn discover_dataset_pairs(dir: &Path) -> Result<Vec<DatasetPair>, Box<dyn Error>> {
    let mut stems: HashMap<String, (Option<PathBuf>, Option<PathBuf>)> = HashMap::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase());
        let stem = match path.file_stem().and_then(|stem| stem.to_str()) {
            Some(stem) => stem.to_owned(),
            None => continue,
        };

        match ext.as_deref() {
            Some("wav") => stems.entry(stem).or_default().0 = Some(path),
            Some("mid") => stems.entry(stem).or_default().1 = Some(path),
            _ => {}
        }
    }

    let mut pairs: Vec<_> = stems
        .into_iter()
        .filter_map(|(stem, (wav_path, midi_path))| {
            Some(DatasetPair {
                stem,
                wav_path: wav_path?,
                midi_path: midi_path?,
            })
        })
        .collect();
    pairs.sort_by(|a, b| a.stem.cmp(&b.stem));
    Ok(pairs)
}

pub fn read_wav_mono(path: &Path) -> Result<WavClip, Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let mut mono = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut frame = Vec::with_capacity(channels);
            for sample in reader.samples::<f32>() {
                frame.push(sample?);
                if frame.len() == channels {
                    mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
                    frame.clear();
                }
            }
        }
        hound::SampleFormat::Int => {
            let scale = ((1i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            let mut frame = Vec::with_capacity(channels);
            for sample in reader.samples::<i32>() {
                frame.push(sample? as f32 / scale);
                if frame.len() == channels {
                    mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
                    frame.clear();
                }
            }
        }
    }

    Ok(WavClip {
        sample_rate: spec.sample_rate,
        samples: mono,
    })
}

pub fn read_midi_notes(
    path: &Path,
    source_id: usize,
) -> Result<Vec<GroundTruthNote>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let smf = Smf::parse(&bytes)?;
    let ticks_per_quarter = match smf.header.timing {
        Timing::Metrical(tpq) => tpq.as_int() as u64,
        Timing::Timecode(_, _) => {
            return Err(format!("unsupported SMPTE MIDI timing in {}", path.display()).into());
        }
    };

    let tempo_us = find_global_tempo(&smf).unwrap_or(DEFAULT_TEMPO_US_PER_QUARTER);
    let secs_per_tick = tempo_us as f32 / 1_000_000.0 / ticks_per_quarter as f32;
    let mut notes = Vec::new();
    let mut active: HashMap<(u8, u8), f32> = HashMap::new();
    let mut track_end_secs = 0.0f32;

    for track in &smf.tracks {
        let mut abs_ticks = 0u64;
        for event in track {
            abs_ticks += event.delta.as_int() as u64;
            let now_secs = abs_ticks as f32 * secs_per_tick;
            track_end_secs = track_end_secs.max(now_secs);
            match event.kind {
                TrackEventKind::Midi { channel, message } => match message {
                    MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                        active.insert((channel.as_int(), key.as_int()), now_secs);
                    }
                    MidiMessage::NoteOff { key, .. } | MidiMessage::NoteOn { key, vel: _ }
                        if matches!(message, MidiMessage::NoteOn { vel, .. } if vel.as_int() == 0)
                            || matches!(message, MidiMessage::NoteOff { .. }) =>
                    {
                        if let Some(start_secs) = active.remove(&(channel.as_int(), key.as_int())) {
                            notes.push(GroundTruthNote {
                                source_id,
                                midi_note: key.as_int(),
                                start_secs,
                                end_secs: now_secs.max(start_secs + 1e-3),
                            });
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    for ((_, midi_note), start_secs) in active {
        notes.push(GroundTruthNote {
            source_id,
            midi_note,
            start_secs,
            end_secs: track_end_secs.max(start_secs + 1e-3),
        });
    }

    notes.sort_by(|a, b| {
        a.start_secs
            .partial_cmp(&b.start_secs)
            .unwrap_or(Ordering::Equal)
    });
    Ok(notes)
}

pub fn transcribe_clip(clip: &WavClip, max_streams: usize) -> Vec<TranscribedNote> {
    let mut transcriber = RealtimeTranscriber::new(clip.sample_rate, max_streams);
    for chunk in clip.samples.chunks(TRANSCRIPTION_HOP_SIZE) {
        transcriber.process_block(chunk);
    }
    transcriber.finish()
}

pub fn mix_clips(clips: &[WavClip]) -> Result<WavClip, Box<dyn Error>> {
    if clips.is_empty() {
        return Err("cannot mix zero clips".into());
    }
    let sample_rate = clips[0].sample_rate;
    if clips.iter().any(|clip| clip.sample_rate != sample_rate) {
        return Err("all clips in a mix must share the same sample rate".into());
    }

    let max_len = clips
        .iter()
        .map(|clip| clip.samples.len())
        .max()
        .unwrap_or(0);
    let mut mix = vec![0.0f32; max_len];
    for clip in clips {
        for (dst, src) in mix.iter_mut().zip(clip.samples.iter().copied()) {
            *dst += src / clips.len() as f32;
        }
    }

    let peak = mix.iter().map(|sample| sample.abs()).fold(0.0f32, f32::max);
    if peak > MIX_HEADROOM {
        let scale = MIX_HEADROOM / peak;
        for sample in &mut mix {
            *sample *= scale;
        }
    }

    Ok(WavClip {
        sample_rate,
        samples: mix,
    })
}

pub fn evaluate_notes(predicted: &[TranscribedNote], reference: &[GroundTruthNote]) -> NoteMetrics {
    let mut predicted_sorted = predicted.to_vec();
    predicted_sorted.sort_by(note_order);
    let mut reference_sorted = reference.to_vec();
    reference_sorted.sort_by(reference_note_order);

    let pred_len = predicted_sorted.len();
    let ref_len = reference_sorted.len();
    let mut dp = vec![vec![0.0f32; ref_len + 1]; pred_len + 1];
    let mut steps = vec![vec![AlignmentStep::SkipPredicted; ref_len + 1]; pred_len + 1];

    for i in (0..pred_len).rev() {
        for j in (0..ref_len).rev() {
            let skip_predicted = dp[i + 1][j];
            let skip_reference = dp[i][j + 1];
            let assessment = assess_match(&predicted_sorted[i], &reference_sorted[j]);
            let matched = assessment.similarity + dp[i + 1][j + 1];

            let (best_score, best_step) = if matched >= skip_predicted && matched >= skip_reference
            {
                (matched, AlignmentStep::Match)
            } else if skip_predicted >= skip_reference {
                (skip_predicted, AlignmentStep::SkipPredicted)
            } else {
                (skip_reference, AlignmentStep::SkipReference)
            };

            dp[i][j] = best_score;
            steps[i][j] = best_step;
        }
    }

    let mut metrics = NoteMetrics {
        predicted_notes: pred_len,
        reference_notes: ref_len,
        ..NoteMetrics::default()
    };

    let mut i = 0usize;
    let mut j = 0usize;
    while i < pred_len && j < ref_len {
        match steps[i][j] {
            AlignmentStep::Match => {
                let assessment = assess_match(&predicted_sorted[i], &reference_sorted[j]);
                if assessment.similarity > 0.0 {
                    metrics.matched_pairs += 1;
                    metrics.matched_similarity += assessment.similarity;
                    metrics.pitch_similarity += assessment.pitch_similarity;
                    metrics.start_similarity += assessment.start_similarity;
                    metrics.end_similarity += assessment.end_similarity;
                    metrics.total_pitch_error_semitones += assessment.pitch_error_semitones;
                    metrics.total_start_error_secs += assessment.start_error_secs;
                    metrics.total_end_error_secs += assessment.end_error_secs;
                    if assessment.onset_match {
                        metrics.matched_onsets += 1;
                    }
                    if assessment.full_note_match {
                        metrics.matched_notes += 1;
                    }
                }
                i += 1;
                j += 1;
            }
            AlignmentStep::SkipPredicted => i += 1,
            AlignmentStep::SkipReference => j += 1,
        }
    }

    metrics
}

pub fn evaluate_source_separated(
    predicted: &[TranscribedNote],
    references: &[Vec<GroundTruthNote>],
) -> EvalSummary {
    let predicted_by_stream = group_predicted_by_stream(predicted);
    let stream_ids: Vec<usize> = predicted_by_stream.keys().copied().collect();

    let mut candidates = Vec::new();
    for (source_id, reference_notes) in references.iter().enumerate() {
        for &stream_id in &stream_ids {
            let metrics = evaluate_notes(&predicted_by_stream[&stream_id], reference_notes);
            let score = metrics.fuzzy_score_percent();
            candidates.push((score, source_id, stream_id, metrics));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut used_sources = HashSet::new();
    let mut used_streams = HashSet::new();
    let mut per_source = vec![SourceEval::default(); references.len()];

    for (_, source_id, stream_id, metrics) in candidates {
        if used_sources.contains(&source_id) || used_streams.contains(&stream_id) {
            continue;
        }
        if metrics.matched_similarity <= 0.0 {
            continue;
        }
        used_sources.insert(source_id);
        used_streams.insert(stream_id);
        per_source[source_id] = SourceEval {
            source_id,
            stream_id: Some(stream_id),
            metrics,
        };
    }

    let mut total = NoteMetrics::default();
    for eval in &per_source {
        accumulate_note_metrics(&mut total, &eval.metrics);
    }
    total.predicted_notes = predicted.len();
    total.reference_notes = references.iter().map(|notes| notes.len()).sum();

    EvalSummary { per_source, total }
}

fn group_predicted_by_stream(
    predicted: &[TranscribedNote],
) -> HashMap<usize, Vec<TranscribedNote>> {
    let mut grouped: HashMap<usize, Vec<TranscribedNote>> = HashMap::new();
    for note in predicted {
        grouped
            .entry(note.stream_id)
            .or_default()
            .push(note.clone());
    }
    grouped
}

fn note_salience(cqt: &[f32]) -> [f32; 128] {
    let bank = note_template_bank();
    let frame: Vec<f32> = cqt.iter().map(|&value| value.max(0.0).sqrt()).collect();
    let frame_norm = frame.iter().map(|value| value * value).sum::<f32>().sqrt();
    let mut salience = [0.0f32; 128];
    if frame_norm <= 1e-6 {
        return salience;
    }

    for midi_note in TRANSCRIPTION_MIDI_LOW..=TRANSCRIPTION_MIDI_HIGH {
        let Some(bin) = midi_note_to_cqt_bin(midi_note) else {
            continue;
        };
        let template = &bank[bin];
        let fundamental = interp_cqt(cqt, bin as f32).clamp(0.0, 1.0);
        let sub_octave = interp_cqt(cqt, bin as f32 - 12.0).clamp(0.0, 1.0);
        let dot = frame
            .iter()
            .zip(template.iter())
            .map(|(sample, weight)| sample * weight)
            .sum::<f32>();
        let similarity = (dot / frame_norm).clamp(0.0, 1.0);
        salience[midi_note as usize] =
            (0.55 * similarity + 0.45 * fundamental - 0.10 * sub_octave).clamp(0.0, 1.0);
    }
    salience
}

fn note_template_bank() -> &'static [[f32; CQT_BINS]; CQT_BINS] {
    static BANK: OnceLock<[[f32; CQT_BINS]; CQT_BINS]> = OnceLock::new();
    BANK.get_or_init(|| {
        let mut templates = [[0.0f32; CQT_BINS]; CQT_BINS];
        for bin in 0..CQT_BINS {
            templates[bin] = build_note_template(bin);
        }
        templates
    })
}

fn build_note_template(base_bin: usize) -> [f32; CQT_BINS] {
    let mut template = [0.0f32; CQT_BINS];
    for &(harmonic, weight) in NOTE_TEMPLATE_HARMONICS {
        let center = base_bin as f32 + 12.0 * harmonic.log2();
        for (bin, value) in template.iter_mut().enumerate() {
            let distance = bin as f32 - center;
            *value += weight * (-0.5 * (distance / NOTE_TEMPLATE_SIGMA_BINS).powi(2)).exp();
        }
    }

    let norm = template
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 1e-6 {
        for value in &mut template {
            *value /= norm;
        }
    }
    template
}

fn interp_cqt(cqt: &[f32], bin: f32) -> f32 {
    if cqt.is_empty() || bin.is_nan() {
        return 0.0;
    }
    if bin <= 0.0 {
        return cqt[0];
    }
    let last = cqt.len() - 1;
    if bin >= last as f32 {
        return cqt[last];
    }

    let lo = bin.floor() as usize;
    let hi = (lo + 1).min(last);
    let frac = bin - lo as f32;
    cqt[lo] * (1.0 - frac) + cqt[hi] * frac
}

fn select_note_activations(
    salience: &[f32; 128],
    note_energy_ema: &[f32; 128],
    harmonic_confidence: f32,
    pitched_stability: f32,
    max_streams: usize,
) -> Vec<NoteActivation> {
    let global_max = salience
        [TRANSCRIPTION_MIDI_LOW as usize..=TRANSCRIPTION_MIDI_HIGH as usize]
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let mut peaks = Vec::new();
    for midi_note in
        TRANSCRIPTION_MIDI_LOW.max(22)..=TRANSCRIPTION_MIDI_HIGH.saturating_sub(1)
    {
        let idx = midi_note as usize;
        let center = salience[idx];
        if center < salience[idx - 1] || center < salience[idx + 1] {
            continue;
        }
        let dynamic_floor = (global_max * 0.32).max(note_energy_ema[idx] * 0.92).max(
            if harmonic_confidence >= 0.2 || pitched_stability >= 0.2 {
                0.05
            } else {
                0.07
            },
        );
        if center < dynamic_floor {
            continue;
        }

        peaks.push(NoteActivation {
            midi_note,
            salience: center,
            brightness: idx as f32 / TRANSCRIPTION_MIDI_HIGH as f32,
        });
    }

    peaks.sort_by(|a, b| {
        b.salience
            .partial_cmp(&a.salience)
            .unwrap_or(Ordering::Equal)
    });
    peaks.truncate(MAX_POLYPHONIC_CANDIDATES.max(max_streams * 2));
    peaks
}

fn harmonic_note_distance(note: &RealtimeActiveNote, candidate: &NoteActivation) -> f32 {
    let pitch_distance = semitone_distance(note.midi_note, candidate.midi_note) as f32;
    let brightness_distance = (note.brightness - candidate.brightness).abs();
    pitch_distance / 8.0 + brightness_distance
}

fn assess_match(predicted: &TranscribedNote, reference: &GroundTruthNote) -> MatchAssessment {
    let pitch_error_semitones = semitone_distance(predicted.midi_note, reference.midi_note) as f32;
    let start_error_secs = (predicted.start_secs - reference.start_secs).abs();
    let end_error_secs = (predicted.end_secs - reference.end_secs).abs();
    let reference_duration = (reference.end_secs - reference.start_secs).max(1e-3);
    let end_half_life_secs = (reference_duration * END_HALF_LIFE_DURATION_FACTOR)
        .clamp(END_HALF_LIFE_MIN_SECS, END_HALF_LIFE_MAX_SECS);

    let pitch_similarity =
        fuzzy_half_life_similarity(pitch_error_semitones, PITCH_HALF_LIFE_SEMITONES);
    let start_similarity = fuzzy_half_life_similarity(start_error_secs, START_HALF_LIFE_SECS);
    let end_similarity = fuzzy_half_life_similarity(end_error_secs, end_half_life_secs);
    let similarity = weighted_geometric_mean(&[pitch_similarity, start_similarity, end_similarity]);

    MatchAssessment {
        similarity,
        pitch_similarity,
        start_similarity,
        end_similarity,
        pitch_error_semitones,
        start_error_secs,
        end_error_secs,
        onset_match: reference.midi_note == predicted.midi_note
            && start_error_secs <= START_TOLERANCE_SECS,
        full_note_match: reference.midi_note == predicted.midi_note
            && start_error_secs <= START_TOLERANCE_SECS
            && end_error_secs <= END_TOLERANCE_SECS,
    }
}

fn fuzzy_half_life_similarity(error: f32, half_life: f32) -> f32 {
    if half_life <= 0.0 {
        return if error <= 0.0 { 1.0 } else { 0.0 };
    }
    2.0f32.powf(-error.max(0.0) / half_life)
}

fn weighted_geometric_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let log_sum: f32 = values.iter().map(|value| value.clamp(1e-6, 1.0).ln()).sum();
    (log_sum / values.len() as f32).exp().clamp(0.0, 1.0)
}

fn note_order(a: &TranscribedNote, b: &TranscribedNote) -> Ordering {
    a.start_secs
        .partial_cmp(&b.start_secs)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.midi_note.cmp(&b.midi_note))
        .then_with(|| {
            a.end_secs
                .partial_cmp(&b.end_secs)
                .unwrap_or(Ordering::Equal)
        })
}

fn reference_note_order(a: &GroundTruthNote, b: &GroundTruthNote) -> Ordering {
    a.start_secs
        .partial_cmp(&b.start_secs)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.midi_note.cmp(&b.midi_note))
        .then_with(|| {
            a.end_secs
                .partial_cmp(&b.end_secs)
                .unwrap_or(Ordering::Equal)
        })
}

fn extract_pitch_candidates(cqt: &[f32]) -> Vec<PitchCandidate> {
    let max_mag = cqt.iter().copied().fold(0.0f32, f32::max);
    if max_mag < MIN_CQT_PEAK {
        return Vec::new();
    }

    let mut peaks = Vec::new();
    for bin in 1..CQT_BINS.saturating_sub(1) {
        let center = cqt[bin];
        if center < MIN_CQT_PEAK || center < max_mag * MIN_RELATIVE_CQT_PEAK {
            continue;
        }
        if center < cqt[bin - 1] || center < cqt[bin + 1] {
            continue;
        }

        let local_contrast = (center - 0.5 * (cqt[bin - 1] + cqt[bin + 1])).max(0.0);
        let confidence =
            (0.7 * (center / max_mag) + 0.3 * (local_contrast / (center + 1e-6))).clamp(0.0, 1.0);
        peaks.push(PitchCandidate {
            midi_note: cqt_bin_to_midi_note(bin).expect("CQT bin should map into MIDI range"),
            confidence,
            bin,
        });
    }

    if peaks.is_empty() {
        if let Some((bin, &mag)) = cqt
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        {
            if mag >= MIN_CQT_PEAK {
                peaks.push(PitchCandidate {
                    midi_note: cqt_bin_to_midi_note(bin)
                        .expect("CQT bin should map into MIDI range"),
                    confidence: 1.0,
                    bin,
                });
            }
        }
    }

    peaks.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });
    peaks.truncate(MAX_PITCH_CANDIDATES);
    peaks
}

fn select_candidate_for_new(
    object: &SoundObject,
    candidates: &[PitchCandidate],
    used_bins: &HashSet<usize>,
) -> Option<PitchCandidate> {
    let target_pitch_class = (((object.acoustic_y * 12.0).round() as i32) % 12 + 12) % 12;
    candidates
        .iter()
        .copied()
        .filter(|candidate| !used_bins.contains(&candidate.bin))
        .max_by(|a, b| {
            new_candidate_score(*a, target_pitch_class)
                .partial_cmp(&new_candidate_score(*b, target_pitch_class))
                .unwrap_or(Ordering::Equal)
        })
        .filter(|candidate| candidate.confidence >= MIN_NOTE_CONFIDENCE)
}

fn select_candidate_for_existing(
    active: &ActiveNote,
    object: &SoundObject,
    candidates: &[PitchCandidate],
    used_bins: &HashSet<usize>,
) -> Option<PitchCandidate> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| !used_bins.contains(&candidate.bin))
        .max_by(|a, b| {
            existing_candidate_score(*a, active, object)
                .partial_cmp(&existing_candidate_score(*b, active, object))
                .unwrap_or(Ordering::Equal)
        })
        .filter(|candidate| candidate.confidence >= MIN_NOTE_CONFIDENCE)
}

fn new_candidate_score(candidate: PitchCandidate, target_pitch_class: i32) -> f32 {
    let pitch_class = (candidate.midi_note as i32).rem_euclid(12);
    let pc_distance = circular_pitch_class_distance(pitch_class, target_pitch_class) as f32;
    candidate.confidence - 0.08 * pc_distance
}

fn existing_candidate_score(
    candidate: PitchCandidate,
    active: &ActiveNote,
    object: &SoundObject,
) -> f32 {
    let note_distance = semitone_distance(active.midi_note, candidate.midi_note) as f32;
    let brightness_distance = (active.brightness - object.acoustic_x).abs();
    candidate.confidence - note_distance / STREAM_NOTE_DISTANCE_NORM - 0.20 * brightness_distance
}

fn stream_distance(stream: &StreamState, midi_note: u8, brightness: f32) -> f32 {
    ((stream.note_center - midi_note as f32).abs() / STREAM_NOTE_DISTANCE_NORM)
        + (stream.brightness_center - brightness).abs()
}

fn semitone_distance(a: u8, b: u8) -> i16 {
    (a as i16 - b as i16).abs()
}

fn circular_pitch_class_distance(a: i32, b: i32) -> i32 {
    let d = (a - b).abs();
    d.min(12 - d)
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn normalized_similarity(total: f32, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        (total / count as f32).clamp(0.0, 1.0)
    }
}

fn mean_error(total: f32, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn harmonic_mean(a: f32, b: f32) -> f32 {
    if a <= 0.0 || b <= 0.0 {
        0.0
    } else {
        2.0 * a * b / (a + b)
    }
}

fn accumulate_note_metrics(total: &mut NoteMetrics, metrics: &NoteMetrics) {
    total.predicted_notes += metrics.predicted_notes;
    total.reference_notes += metrics.reference_notes;
    total.matched_onsets += metrics.matched_onsets;
    total.matched_notes += metrics.matched_notes;
    total.matched_pairs += metrics.matched_pairs;
    total.matched_similarity += metrics.matched_similarity;
    total.pitch_similarity += metrics.pitch_similarity;
    total.start_similarity += metrics.start_similarity;
    total.end_similarity += metrics.end_similarity;
    total.total_pitch_error_semitones += metrics.total_pitch_error_semitones;
    total.total_start_error_secs += metrics.total_start_error_secs;
    total.total_end_error_secs += metrics.total_end_error_secs;
}

fn find_global_tempo(smf: &Smf<'_>) -> Option<u32> {
    for track in &smf.tracks {
        for event in track {
            if let TrackEventKind::Meta(MetaMessage::Tempo(tempo)) = event.kind {
                return Some(tempo.as_int());
            }
        }
    }
    None
}

fn local_pitch_brightness(cqt: &[f32], center_bin: usize) -> f32 {
    if cqt.is_empty() {
        return 0.5;
    }

    let start = center_bin.saturating_sub(4);
    let end = (center_bin + 4).min(cqt.len() - 1);
    let mut weighted_sum = 0.0;
    let mut total = 0.0;
    for (idx, &mag) in cqt.iter().enumerate().take(end + 1).skip(start) {
        weighted_sum += mag * idx as f32;
        total += mag;
    }

    if total <= 1e-6 || cqt.len() == 1 {
        center_bin as f32 / (cqt.len().saturating_sub(1).max(1)) as f32
    } else {
        (weighted_sum / total / (cqt.len() - 1) as f32).clamp(0.0, 1.0)
    }
}

fn max_polyphony(notes: &[GroundTruthNote]) -> usize {
    let mut events = Vec::with_capacity(notes.len() * 2);
    for note in notes {
        events.push((note.start_secs, 1i32));
        events.push((note.end_secs, -1i32));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut current = 0i32;
    let mut max_seen = 0i32;
    for (_, delta) in events {
        current += delta;
        max_seen = max_seen.max(current);
    }
    max_seen.max(0) as usize
}

fn is_simple_harmonic_pair(pair: &LoadedPair) -> bool {
    let stem = pair.pair.stem.to_ascii_lowercase();
    !PERCUSSION_KEYWORDS
        .iter()
        .any(|keyword| stem.contains(keyword))
        && !EXCLUDED_FIXTURE_KEYWORDS
            .iter()
            .any(|keyword| stem.contains(keyword))
        && pair.notes.len() >= MIN_SIMPLE_NOTES
        && pair.notes.len() <= MAX_SIMPLE_NOTES
        && pair.max_polyphony <= MAX_SIMPLE_POLYPHONY
}

fn fixture_rank(pair: &LoadedPair) -> usize {
    FIXTURE_PREFERENCE_ORDER
        .iter()
        .position(|preferred| pair.pair.stem == *preferred)
        .unwrap_or(FIXTURE_PREFERENCE_ORDER.len())
}

fn load_pairs(dir: &Path) -> Result<Vec<LoadedPair>, Box<dyn Error>> {
    let pairs = discover_dataset_pairs(dir)?;
    let mut loaded = Vec::new();
    for (source_id, pair) in pairs.into_iter().enumerate() {
        let clip = read_wav_mono(&pair.wav_path)?;
        let notes = read_midi_notes(&pair.midi_path, source_id)?;
        let polyphony = max_polyphony(&notes);
        loaded.push(LoadedPair {
            pair,
            clip,
            notes,
            max_polyphony: polyphony,
        });
    }
    Ok(loaded)
}

fn standardized_test_pairs(dir: &Path) -> Result<Vec<LoadedPair>, Box<dyn Error>> {
    let mut loaded = load_pairs(dir)?;
    loaded.sort_by(|a, b| a.pair.stem.cmp(&b.pair.stem));
    if loaded.is_empty() {
        return Err("no MIDI/WAV pairs found for standardized transcription tests".into());
    }
    Ok(loaded)
}

fn standardized_mix_index_sets(
    pairs: &[LoadedPair],
    mix_count: usize,
) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    if !STANDARD_MIX_SIZES.contains(&mix_count) {
        return Err(format!("unsupported standardized mix size: {mix_count}").into());
    }

    let mut indices_by_rate: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (idx, pair) in pairs.iter().enumerate() {
        indices_by_rate
            .entry(pair.clip.sample_rate)
            .or_default()
            .push(idx);
    }

    let Some((sample_rate, mut pool)) = indices_by_rate
        .into_iter()
        .max_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| b.0.cmp(&a.0)))
    else {
        return Err("no standardized fixtures available for mix generation".into());
    };

    pool.sort_by(|&a, &b| pairs[a].pair.stem.cmp(&pairs[b].pair.stem));
    if pool.len() < mix_count {
        return Err(format!(
            "need at least {mix_count} same-rate fixtures for standardized mixes at {sample_rate} Hz, found {}",
            pool.len()
        )
        .into());
    }

    let all_combinations = combination_indices(&pool, mix_count);
    if all_combinations.len() < STANDARD_MIX_COUNT {
        return Err(format!(
            "need at least {} distinct {}-source mixes at {sample_rate} Hz, found {}",
            STANDARD_MIX_COUNT,
            mix_count,
            all_combinations.len()
        )
        .into());
    }

    Ok(select_evenly_spaced_combinations(
        &all_combinations,
        STANDARD_MIX_COUNT,
    ))
}

fn combination_indices(pool: &[usize], choose: usize) -> Vec<Vec<usize>> {
    fn recurse(
        pool: &[usize],
        choose: usize,
        start: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == choose {
            out.push(current.clone());
            return;
        }

        let remaining = choose - current.len();
        let max_start = pool.len().saturating_sub(remaining);
        for idx in start..=max_start {
            current.push(pool[idx]);
            recurse(pool, choose, idx + 1, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    recurse(pool, choose, 0, &mut Vec::with_capacity(choose), &mut out);
    out
}

fn select_evenly_spaced_combinations(
    combinations: &[Vec<usize>],
    target_count: usize,
) -> Vec<Vec<usize>> {
    if combinations.len() <= target_count {
        return combinations.to_vec();
    }

    let mut selected = Vec::with_capacity(target_count);
    let span = combinations.len() - 1;
    let steps = target_count - 1;
    for step in 0..target_count {
        let idx = if steps == 0 { 0 } else { step * span / steps };
        selected.push(combinations[idx].clone());
    }
    selected
}

fn mix_label(pairs: &[LoadedPair], mix_indices: &[usize]) -> String {
    mix_indices
        .iter()
        .map(|&idx| pairs[idx].pair.stem.as_str())
        .collect::<Vec<_>>()
        .join(" + ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_and_midi_readers_parse_standardized_pairs() {
        let pairs =
            standardized_test_pairs(Path::new(DATASET_DIR)).expect("failed to load dataset pairs");
        assert!(
            !pairs.is_empty(),
            "expected at least one .wav/.mid pair in midi/"
        );
        for pair in &pairs {
            assert!(pair.clip.sample_rate > 0);
            assert!(
                !pair.clip.samples.is_empty(),
                "{} has no audio samples",
                pair.pair.stem
            );
            assert!(
                !pair.notes.is_empty(),
                "{} has no MIDI notes",
                pair.pair.stem
            );
        }
        let two_source_mixes =
            standardized_mix_index_sets(&pairs, 2).expect("failed to build 2-source mix set");
        let three_source_mixes =
            standardized_mix_index_sets(&pairs, 3).expect("failed to build 3-source mix set");
        assert_eq!(two_source_mixes.len(), STANDARD_MIX_COUNT);
        assert_eq!(three_source_mixes.len(), STANDARD_MIX_COUNT);
    }

    #[test]
    fn fuzzy_metric_scores_exact_match_at_100_percent() {
        let predicted = vec![TranscribedNote {
            stream_id: 0,
            midi_note: 60,
            start_secs: 1.0,
            end_secs: 1.5,
            confidence: 1.0,
        }];
        let reference = vec![GroundTruthNote {
            source_id: 0,
            midi_note: 60,
            start_secs: 1.0,
            end_secs: 1.5,
        }];

        let metrics = evaluate_notes(&predicted, &reference);
        assert!((metrics.fuzzy_score_percent() - 100.0).abs() < 1e-4);
        assert_eq!(metrics.matched_onsets, 1);
        assert_eq!(metrics.matched_notes, 1);
    }

    #[test]
    fn fuzzy_metric_gives_partial_credit_for_near_miss() {
        let predicted = vec![TranscribedNote {
            stream_id: 0,
            midi_note: 61,
            start_secs: 1.04,
            end_secs: 1.58,
            confidence: 1.0,
        }];
        let reference = vec![GroundTruthNote {
            source_id: 0,
            midi_note: 60,
            start_secs: 1.0,
            end_secs: 1.5,
        }];

        let metrics = evaluate_notes(&predicted, &reference);
        assert!(metrics.fuzzy_score_percent() > 0.0);
        assert!(metrics.fuzzy_score_percent() < 100.0);
        assert_eq!(metrics.matched_onsets, 0);
        assert_eq!(metrics.matched_notes, 0);
    }

    #[test]
    #[ignore]
    fn report_current_dataset_scores() {
        let pairs =
            standardized_test_pairs(Path::new(DATASET_DIR)).expect("failed to load score fixtures");
        let mut single_total = NoteMetrics::default();
        let mut two_mix_total = NoteMetrics::default();
        let mut three_mix_total = NoteMetrics::default();

        eprintln!("fixtures:");
        for pair in &pairs {
            eprintln!("  {}", pair.pair.stem);
        }

        eprintln!("single-source:");
        for pair in &pairs {
            let predicted = transcribe_clip(&pair.clip, 1);
            let metrics = evaluate_notes(&predicted, &pair.notes);
            eprintln!(
                "  {:<45} score {:>6.2}%  matched_pairs {:>3}  pitch {:>5.1}c  start {:>6.1}ms  end {:>6.1}ms",
                pair.pair.stem,
                metrics.fuzzy_score_percent(),
                metrics.matched_pairs,
                metrics.mean_pitch_error_semitones() * 100.0,
                metrics.mean_start_error_ms(),
                metrics.mean_end_error_ms(),
            );
            accumulate_note_metrics(&mut single_total, &metrics);
        }

        eprintln!("mixed-source:");
        for &mix_count in STANDARD_MIX_SIZES {
            let mix_index_sets = standardized_mix_index_sets(&pairs, mix_count)
                .expect("failed to build standardized mix fixture set");
            for (mix_idx, mix_indices) in mix_index_sets.iter().enumerate() {
                let clips: Vec<_> = mix_indices
                    .iter()
                    .map(|&pair_idx| pairs[pair_idx].clip.clone())
                    .collect();
                let mixed = mix_clips(&clips).expect("failed to build synthetic mix");
                let predicted = transcribe_clip(&mixed, mix_count);
                let references: Vec<_> = mix_indices
                    .iter()
                    .enumerate()
                    .map(|(source_id, &pair_idx)| {
                        pairs[pair_idx]
                            .notes
                            .iter()
                            .cloned()
                            .map(|mut note| {
                                note.source_id = source_id;
                                note
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let summary = evaluate_source_separated(&predicted, &references);
                eprintln!(
                    "  {:>1}-source mix {:>2}: {:<90} score {:>6.2}%  matched_pairs {:>3}  pitch {:>5.1}c  start {:>6.1}ms  end {:>6.1}ms",
                    mix_count,
                    mix_idx + 1,
                    mix_label(&pairs, mix_indices),
                    summary.total.fuzzy_score_percent(),
                    summary.total.matched_pairs,
                    summary.total.mean_pitch_error_semitones() * 100.0,
                    summary.total.mean_start_error_ms(),
                    summary.total.mean_end_error_ms(),
                );
                if mix_count == 2 {
                    accumulate_note_metrics(&mut two_mix_total, &summary.total);
                } else {
                    accumulate_note_metrics(&mut three_mix_total, &summary.total);
                }
            }
        }

        let mut overall = NoteMetrics::default();
        accumulate_note_metrics(&mut overall, &single_total);
        accumulate_note_metrics(&mut overall, &two_mix_total);
        accumulate_note_metrics(&mut overall, &three_mix_total);

        eprintln!(
            "aggregate single-source score: {:.2}%",
            single_total.fuzzy_score_percent()
        );
        eprintln!(
            "aggregate 2-source mix score: {:.2}%",
            two_mix_total.fuzzy_score_percent()
        );
        eprintln!(
            "aggregate 3-source mix score: {:.2}%",
            three_mix_total.fuzzy_score_percent()
        );
        eprintln!(
            "aggregate overall score:       {:.2}%",
            overall.fuzzy_score_percent()
        );
    }

    #[test]
    fn standardized_suite_evaluates_all_sources() {
        let pairs = standardized_test_pairs(Path::new(DATASET_DIR))
            .expect("failed to load standardized single-source fixtures");
        assert!(
            !pairs.is_empty(),
            "need at least one standardized single-source fixture"
        );

        let mut total = NoteMetrics::default();
        for pair in &pairs {
            let predicted = transcribe_clip(&pair.clip, 1);
            let metrics = evaluate_notes(&predicted, &pair.notes);

            assert!(
                metrics.fuzzy_score_percent().is_finite(),
                "transcription score is not finite for {}",
                pair.pair.stem
            );
            assert!(
                metrics.predicted_notes == predicted.len(),
                "predicted note count mismatch for {}",
                pair.pair.stem
            );
            assert_eq!(
                metrics.reference_notes,
                pair.notes.len(),
                "reference note count mismatch for {}",
                pair.pair.stem
            );
            accumulate_note_metrics(&mut total, &metrics);
        }

        assert!(
            total.reference_notes > 0,
            "standardized suite did not include any reference notes"
        );
        assert!(
            total.predicted_notes > 0,
            "transcriber produced no notes across the standardized single-source suite"
        );
    }

    #[test]
    fn online_transcriber_handles_standardized_two_and_three_source_mixes() {
        let pairs = standardized_test_pairs(Path::new(DATASET_DIR))
            .expect("failed to load standardized mixed-source transcription fixtures");
        assert!(
            pairs.len() >= 3,
            "need at least three standardized fixtures for 2-source and 3-source mix tests"
        );

        for &mix_count in STANDARD_MIX_SIZES {
            let mix_index_sets = standardized_mix_index_sets(&pairs, mix_count)
                .expect("failed to load standardized mix fixtures");
            let unique_mix_count: HashSet<_> = mix_index_sets.iter().cloned().collect();
            assert_eq!(
                mix_index_sets.len(),
                STANDARD_MIX_COUNT,
                "expected {} standardized {}-source mixes",
                STANDARD_MIX_COUNT,
                mix_count
            );
            assert_eq!(
                unique_mix_count.len(),
                STANDARD_MIX_COUNT,
                "expected {} distinct standardized {}-source mixes",
                STANDARD_MIX_COUNT,
                mix_count
            );

            let mut total = NoteMetrics::default();
            for mix_indices in &mix_index_sets {
                let clips: Vec<_> = mix_indices
                    .iter()
                    .map(|&pair_idx| pairs[pair_idx].clip.clone())
                    .collect();
                let mixed = mix_clips(&clips).expect("failed to build synthetic mix");
                let predicted = transcribe_clip(&mixed, mix_count);
                let unique_streams: HashSet<_> =
                    predicted.iter().map(|note| note.stream_id).collect();
                let references: Vec<_> = mix_indices
                    .iter()
                    .enumerate()
                    .map(|(source_id, &pair_idx)| {
                        pairs[pair_idx]
                            .notes
                            .iter()
                            .cloned()
                            .map(|mut note| {
                                note.source_id = source_id;
                                note
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let summary = evaluate_source_separated(&predicted, &references);

                assert!(
                    summary.total.fuzzy_score_percent().is_finite(),
                    "mix transcription score is not finite for {}",
                    mix_label(&pairs, mix_indices)
                );
                assert!(
                    summary.total.predicted_notes == predicted.len(),
                    "predicted note count mismatch for {}",
                    mix_label(&pairs, mix_indices)
                );
                assert!(
                    unique_streams.len() <= predicted.len(),
                    "unique stream count exceeds note count for {}",
                    mix_label(&pairs, mix_indices)
                );
                accumulate_note_metrics(&mut total, &summary.total);
            }

            assert!(
                total.reference_notes > 0,
                "no reference notes were evaluated for {}-source standardized mixes",
                mix_count
            );
            assert!(
                total.predicted_notes > 0,
                "transcriber produced no notes across {}-source standardized mixes",
                mix_count
            );
        }
    }
}
