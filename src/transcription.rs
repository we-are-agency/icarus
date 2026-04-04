#![allow(dead_code)]

use crate::analysis::{
    AudioFeatures, Analyser, CQT_BINS, midi_note_to_cqt_bin,
};
use crate::audio::FFT_SIZE;
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const DATASET_DIR: &str = "midi";
pub const TRANSCRIPTION_HOP_SIZE: usize = FFT_SIZE / 4;
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
const STANDARD_MIX_COUNT: usize = 10;
const STANDARD_MIX_SIZES: &[usize] = &[2, 3];
pub const TRANSCRIPTION_MIDI_LOW: u8 = 21;
pub const TRANSCRIPTION_MIDI_HIGH: u8 = 104;
const MAX_POLYPHONIC_CANDIDATES: usize = 6;
const HARMONIC_START_THRESHOLD: f32 = 0.10;
const HARMONIC_STOP_THRESHOLD: f32 = 0.04;
const HARMONIC_MIN_NOTE_SECS: f32 = 0.05;
const HARMONIC_RETRIGGER_GAP_SECS: f32 = 0.10;
const HARMONIC_PITCH_LOCK_SECS: f32 = 0.036;
const HARMONIC_LOCKED_MATCH_DISTANCE: i16 = 0;
const HARMONIC_PENDING_CONFIRM_HITS: u8 = 2;
const HARMONIC_PENDING_MAX_AGE_SECS: f32 = 0.060;
const HARMONIC_PENDING_MAX_GAP_SECS: f32 = 0.045;
const HARMONIC_IMMEDIATE_ONSET_CONFIDENCE: f32 = 0.12;
const HARMONIC_STRONG_SUSTAIN_CONFIDENCE: f32 = 0.28;
const HARMONIC_LOCAL_CONTRAST_MIN: f32 = 0.010;
const HARMONIC_IMMEDIATE_CONTRAST_MIN: f32 = 0.018;
const HARMONIC_REFERENCE_DECAY: f32 = 0.95;
const HARMONIC_TOTAL_REFERENCE_DECAY: f32 = 0.96;
const NORMALIZATION_COLLAPSE_PEAK_RATIO: f32 = 0.58;
const NORMALIZATION_COLLAPSE_TOTAL_RATIO: f32 = 0.52;
const NORMALIZATION_COLLAPSE_GUARD_DECAY: f32 = 0.68;
const NORMALIZATION_COLLAPSE_FANOUT_START: f32 = 0.34;
const NORMALIZATION_COLLAPSE_STRONG_ONSET: f32 = 0.28;
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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum InstrumentSelection {
    Percussive,
    Harmonic,
    PercussiveHarmonic,
}

impl InstrumentSelection {
    pub fn label(self) -> &'static str {
        match self {
            Self::Percussive => "Percussive",
            Self::Harmonic => "Harmonic",
            Self::PercussiveHarmonic => "PercussiveHarmonic",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscribedNote {
    pub stream_id: usize,
    pub midi_note: u8,
    pub start_secs: f32,
    pub end_secs: f32,
    pub confidence: f32,
    pub instrument_selection: InstrumentSelection,
    pub audio_energy_hop_secs: f32,
    pub audio_energy_envelope: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletedTranscribedNote {
    pub id: usize,
    pub note: TranscribedNote,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveTranscribedNote {
    pub id: usize,
    pub stream_id: usize,
    pub midi_note: u8,
    pub start_secs: f32,
    pub end_secs: f32,
    pub confidence: f32,
    pub instrument_selection: InstrumentSelection,
    pub audio_energy: f32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FailureBucket {
    Correct,
    MissedNote,
    ExtraNote,
    MergeLikely,
    SplitLikely,
    OctaveError,
    PitchError,
    EarlyOnset,
    LateOnset,
    EarlyRelease,
    LateRelease,
    TimingDrift,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticEvent {
    pub bucket: FailureBucket,
    pub source_id: Option<usize>,
    pub stream_id: Option<usize>,
    pub matched: bool,
    pub similarity: f32,
    pub pitch_error_semitones: Option<f32>,
    pub start_error_ms: Option<f32>,
    pub end_error_ms: Option<f32>,
    pub predicted: Option<TranscribedNote>,
    pub reference: Option<GroundTruthNote>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BucketCount {
    pub bucket: FailureBucket,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ErrorStats {
    pub count: usize,
    pub mean: f32,
    pub p50: f32,
    pub p90: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ErrorSummary {
    pub pitch_error_semitones: ErrorStats,
    pub start_error_ms: ErrorStats,
    pub end_error_ms: ErrorStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct NoteDiagnostics {
    pub metrics: NoteMetrics,
    pub bucket_counts: Vec<BucketCount>,
    pub error_summary: ErrorSummary,
    pub events: Vec<DiagnosticEvent>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationCaseKind {
    SingleSource,
    Mix2Source,
    Mix3Source,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceDiagnostic {
    pub source_id: usize,
    pub label: String,
    pub stream_id: Option<usize>,
    pub diagnostics: NoteDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamDiagnostic {
    pub stream_id: usize,
    pub diagnostics: NoteDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationCaseReport {
    pub label: String,
    pub kind: EvaluationCaseKind,
    pub source_count: usize,
    pub sample_rate: u32,
    pub tags: Vec<String>,
    pub diagnostics: NoteDiagnostics,
    pub per_source: Vec<SourceDiagnostic>,
    pub unassigned_streams: Vec<StreamDiagnostic>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceSeparatedDiagnostics {
    pub diagnostics: NoteDiagnostics,
    pub per_source: Vec<SourceDiagnostic>,
    pub unassigned_streams: Vec<StreamDiagnostic>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationSliceReport {
    pub label: String,
    pub case_count: usize,
    pub diagnostics: NoteDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationSuiteReport {
    pub dataset_dir: String,
    pub overall: EvaluationSliceReport,
    pub slices: Vec<EvaluationSliceReport>,
    pub cases: Vec<EvaluationCaseReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NoteMetricSummary {
    pub score_percent: f32,
    pub predicted_notes: usize,
    pub reference_notes: usize,
    pub matched_pairs: usize,
    pub matched_onsets: usize,
    pub matched_notes: usize,
    pub mean_pitch_error_cents: f32,
    pub mean_start_error_ms: f32,
    pub mean_end_error_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticSummary {
    pub metrics: NoteMetricSummary,
    pub bucket_counts: Vec<BucketCount>,
    pub error_summary: ErrorSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationCaseSummary {
    pub label: String,
    pub kind: EvaluationCaseKind,
    pub source_count: usize,
    pub diagnostics: DiagnosticSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationSliceSummary {
    pub label: String,
    pub case_count: usize,
    pub diagnostics: DiagnosticSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationSuiteSummary {
    pub dataset_dir: String,
    pub overall: EvaluationSliceSummary,
    pub slices: Vec<EvaluationSliceSummary>,
    pub cases: Vec<EvaluationCaseSummary>,
}

#[derive(Debug, Clone, Default, Serialize)]
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

#[derive(Debug, Clone, Default, Serialize)]
pub struct SourceEval {
    pub source_id: usize,
    pub stream_id: Option<usize>,
    pub metrics: NoteMetrics,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct EvalSummary {
    pub per_source: Vec<SourceEval>,
    pub total: NoteMetrics,
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
    absolute_strength: f32,
    brightness: f32,
    local_contrast: f32,
    register: usize,
}

#[derive(Debug, Clone)]
struct NoteEvidenceFrame {
    salience: [f32; 128],
    absolute_strength: [f32; 128],
}

#[derive(Debug, Clone)]
struct PendingHarmonicCandidate {
    midi_note: u8,
    first_seen_secs: f32,
    last_seen_secs: f32,
    best_salience: f32,
    best_absolute_strength: f32,
    brightness: f32,
    local_contrast: f32,
    consecutive_hits: u8,
    register: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct SalienceFrameStats {
    global_peak: f32,
    total_energy: f32,
    register_peaks: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BirthDecision {
    Reject,
    Pending,
    Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveKind {
    Harmonic,
    Percussive,
}

fn transcription_hop_secs(sample_rate: u32) -> f32 {
    TRANSCRIPTION_HOP_SIZE as f32 / sample_rate.max(1) as f32
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
    instrument_selection: InstrumentSelection,
    audio_energy: f32,
    audio_energy_envelope: Vec<f32>,
}

struct RealtimeTranscriber {
    analyser: Analyser,
    sample_rate: u32,
    window: VecDeque<f32>,
    elapsed_secs: f32,
    note_energy_ema: [f32; 128],
    note_cooldown_secs: [f32; 128],
    peak_salience_ref: f32,
    total_salience_ref: f32,
    register_peak_refs: [f32; 3],
    collapse_guard: f32,
    active_notes: Vec<RealtimeActiveNote>,
    pending_candidates: Vec<PendingHarmonicCandidate>,
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

impl RealtimeTranscriber {
    fn new(sample_rate: u32, max_streams: usize) -> Self {
        Self {
            analyser: Analyser::new(sample_rate),
            sample_rate,
            window: VecDeque::from(vec![0.0; FFT_SIZE]),
            elapsed_secs: 0.0,
            note_energy_ema: [0.0; 128],
            note_cooldown_secs: [0.0; 128],
            peak_salience_ref: 0.0,
            total_salience_ref: 0.0,
            register_peak_refs: [0.0; 3],
            collapse_guard: 0.0,
            active_notes: Vec::new(),
            pending_candidates: Vec::new(),
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
                note.instrument_selection,
                note.audio_energy_envelope,
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
        instrument_selection: InstrumentSelection,
        audio_energy_envelope: Vec<f32>,
    ) {
        self.finished_notes.push(CompletedTranscribedNote {
            id,
            note: TranscribedNote {
                stream_id,
                midi_note,
                start_secs,
                end_secs,
                confidence,
                instrument_selection,
                audio_energy_hop_secs: transcription_hop_secs(self.sample_rate),
                audio_energy_envelope,
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

        let note_evidence = note_evidence(&self.analyser.features.cqt);
        let frame_stats = evidence_frame_stats(&note_evidence.absolute_strength);
        self.collapse_guard = self.compute_collapse_guard(frame_stats);
        self.update_note_energy_ema(&note_evidence.absolute_strength);
        let candidates = select_note_activations(
            &note_evidence.salience,
            &note_evidence.absolute_strength,
            &self.note_energy_ema,
            self.peak_salience_ref,
            &self.register_peak_refs,
            self.collapse_guard,
            harmonic_confidence,
            pitched_stability,
            self.max_streams,
        );
        self.update_harmonic_notes(&candidates, frame_start, frame_end);
        self.sample_active_note_audio_energy(&note_evidence.absolute_strength);
        self.emit_percussive_notes(frame_start);
        self.release_stale_notes(frame_end);
        self.update_salience_references(frame_stats);
    }

    fn sample_active_note_audio_energy(&mut self, absolute_strength: &[f32; 128]) {
        for note in &mut self.active_notes {
            if note.kind != ActiveKind::Harmonic {
                continue;
            }
            let energy = absolute_strength[note.midi_note as usize].clamp(0.0, 1.0);
            note.audio_energy = energy;
            note.audio_energy_envelope.push(energy);
        }
    }

    fn current_harmonic_instrument_selection(&self) -> InstrumentSelection {
        let features = &self.analyser.features;
        if features.harmonic_confidence >= 0.20
            && features.percussive_confidence >= 0.20
            && features.onset_confidence >= 0.10
        {
            InstrumentSelection::PercussiveHarmonic
        } else {
            InstrumentSelection::Harmonic
        }
    }

    fn update_note_energy_ema(&mut self, note_strength: &[f32; 128]) {
        for midi_note in TRANSCRIPTION_MIDI_LOW..=TRANSCRIPTION_MIDI_HIGH {
            let idx = midi_note as usize;
            let target = note_strength[idx];
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
        let mut pending_candidates = Vec::new();
        let (immediate_budget, pending_budget) = self.collapse_birth_budgets();
        let mut immediate_births = 0usize;
        let mut deferred_births = 0usize;

        for candidate in candidates {
            if let Some(active_idx) =
                self.find_best_harmonic_match(candidate, &matched_ids, frame_start)
            {
                let active = &self.active_notes[active_idx];
                let active_age = frame_start - active.start_secs;
                let pitch_locked = active_age >= HARMONIC_PITCH_LOCK_SECS;
                let should_retrigger = onset_confidence >= NOTE_RETRIGGER_ONSET_CONFIDENCE + 0.08
                    && frame_start - active.start_secs >= HARMONIC_RETRIGGER_GAP_SECS + 0.04
                    && semitone_distance(active.midi_note, candidate.midi_note)
                        <= NOTE_RELOCK_DISTANCE
                    && candidate.salience >= active.confidence * 1.08
                    && candidate.local_contrast >= HARMONIC_IMMEDIATE_CONTRAST_MIN;
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
                if !pitch_locked
                    && semitone_distance(active.midi_note, candidate.midi_note)
                        <= NOTE_RELOCK_DISTANCE
                    && candidate.salience >= active.confidence * 0.82
                {
                    active.midi_note = candidate.midi_note;
                }
                matched_ids.insert(active.id);
                continue;
            }

            match self.limit_birth_fanout(
                candidate,
                self.classify_harmonic_birth(candidate),
                immediate_births,
                deferred_births,
                immediate_budget,
                pending_budget,
            ) {
                BirthDecision::Immediate => {
                    self.start_harmonic_note(*candidate, frame_start, None);
                    immediate_births += 1;
                }
                BirthDecision::Pending => {
                    pending_candidates.push(*candidate);
                    deferred_births += 1;
                }
                BirthDecision::Reject => {}
            }
        }

        for (candidate, stream_id) in restarts {
            self.start_harmonic_note(candidate, frame_start, Some(stream_id));
        }

        self.update_pending_harmonic_candidates(&pending_candidates, frame_start, frame_end);
    }

    fn classify_harmonic_birth(&self, candidate: &NoteActivation) -> BirthDecision {
        let idx = candidate.midi_note as usize;
        let features = &self.analyser.features;
        let already_active = self
            .active_notes
            .iter()
            .any(|note| note.kind == ActiveKind::Harmonic && note.midi_note == candidate.midi_note);
        if already_active || self.note_cooldown_secs[idx] > 0.0 {
            return BirthDecision::Reject;
        }

        let floor = self.harmonic_birth_floor(candidate);
        let collapse_penalty = self.collapse_birth_penalty(candidate);
        let immediate_floor = floor * (0.92 + 0.08 * collapse_penalty);
        let pending_floor = floor * (0.66 + 0.06 * collapse_penalty);
        let onset_ok = features.onset_confidence >= HARMONIC_IMMEDIATE_ONSET_CONFIDENCE;
        let sustained_ok =
            features.harmonic_confidence >= HARMONIC_STRONG_SUSTAIN_CONFIDENCE
                || features.pitched_stability >= HARMONIC_STRONG_SUSTAIN_CONFIDENCE;
        let moderate_support =
            features.onset_confidence >= 0.06 || features.harmonic_confidence >= 0.20
                || features.pitched_stability >= 0.20;

        if candidate.absolute_strength >= immediate_floor
            && candidate.local_contrast >= HARMONIC_IMMEDIATE_CONTRAST_MIN
            && (onset_ok || sustained_ok)
        {
            return BirthDecision::Immediate;
        }

        if candidate.absolute_strength >= pending_floor
            && candidate.local_contrast >= HARMONIC_LOCAL_CONTRAST_MIN
            && moderate_support
        {
            return BirthDecision::Pending;
        }

        BirthDecision::Reject
    }

    fn harmonic_birth_floor(&self, candidate: &NoteActivation) -> f32 {
        let idx = candidate.midi_note as usize;
        let global_ref = self.peak_salience_ref * 0.12;
        let register_ref = self.register_peak_refs[candidate.register] * 0.24;
        let note_ref = self.note_energy_ema[idx] * 0.98;
        let total_ref = (self.total_salience_ref / 12.0) * 0.03;
        HARMONIC_START_THRESHOLD
            .max(global_ref)
            .max(register_ref)
            .max(note_ref)
            .max(total_ref)
    }

    fn collapse_birth_penalty(&self, candidate: &NoteActivation) -> f32 {
        let register_weight = match candidate.register {
            0 => 0.15,
            1 => 0.45,
            _ => 1.0,
        };
        self.collapse_guard * register_weight
    }

    fn collapse_birth_budgets(&self) -> (usize, usize) {
        if self.collapse_guard < NORMALIZATION_COLLAPSE_FANOUT_START {
            return (usize::MAX, usize::MAX);
        }

        let onset_confidence = self.analyser.features.onset_confidence;
        if onset_confidence >= NORMALIZATION_COLLAPSE_STRONG_ONSET {
            (3, 3)
        } else if onset_confidence >= NOTE_RETRIGGER_ONSET_CONFIDENCE {
            (2, 2)
        } else {
            (1, 1)
        }
    }

    fn collapse_birth_override_strength(&self, candidate: &NoteActivation) -> bool {
        if self.collapse_guard < NORMALIZATION_COLLAPSE_FANOUT_START {
            return true;
        }

        let floor = self.harmonic_birth_floor(candidate);
        let collapse_penalty = self.collapse_birth_penalty(candidate);
        let onset_confidence = self.analyser.features.onset_confidence;
        candidate.absolute_strength >= floor * (1.45 + 0.20 * collapse_penalty)
            && candidate.local_contrast
                >= HARMONIC_IMMEDIATE_CONTRAST_MIN * (1.6 + 0.25 * collapse_penalty)
            && onset_confidence >= HARMONIC_IMMEDIATE_ONSET_CONFIDENCE
    }

    fn limit_birth_fanout(
        &self,
        candidate: &NoteActivation,
        decision: BirthDecision,
        immediate_births: usize,
        deferred_births: usize,
        immediate_budget: usize,
        pending_budget: usize,
    ) -> BirthDecision {
        if self.collapse_guard < NORMALIZATION_COLLAPSE_FANOUT_START
            || self.collapse_birth_override_strength(candidate)
        {
            return decision;
        }

        match decision {
            BirthDecision::Immediate => {
                if immediate_births < immediate_budget {
                    BirthDecision::Immediate
                } else if deferred_births < pending_budget {
                    BirthDecision::Pending
                } else {
                    BirthDecision::Reject
                }
            }
            BirthDecision::Pending => {
                if deferred_births < pending_budget {
                    BirthDecision::Pending
                } else {
                    BirthDecision::Reject
                }
            }
            BirthDecision::Reject => BirthDecision::Reject,
        }
    }

    fn update_pending_harmonic_candidates(
        &mut self,
        candidates: &[NoteActivation],
        frame_start: f32,
        frame_end: f32,
    ) {
        let mut pending_candidates = std::mem::take(&mut self.pending_candidates);

        for candidate in candidates {
            let Some((_, pending)) = pending_candidates
                .iter_mut()
                .enumerate()
                .find(|(_, pending)| {
                    semitone_distance(pending.midi_note, candidate.midi_note)
                        <= NOTE_RELOCK_DISTANCE
                })
            else {
                pending_candidates.push(PendingHarmonicCandidate {
                    midi_note: candidate.midi_note,
                    first_seen_secs: frame_start,
                    last_seen_secs: frame_end,
                    best_salience: candidate.salience,
                    best_absolute_strength: candidate.absolute_strength,
                    brightness: candidate.brightness,
                    local_contrast: candidate.local_contrast,
                    consecutive_hits: 1,
                    register: candidate.register,
                });
                continue;
            };

            pending.midi_note = candidate.midi_note;
            pending.last_seen_secs = frame_end;
            pending.best_salience = pending.best_salience.max(candidate.salience);
            pending.best_absolute_strength = pending
                .best_absolute_strength
                .max(candidate.absolute_strength);
            pending.brightness = 0.75 * pending.brightness + 0.25 * candidate.brightness;
            pending.local_contrast = pending.local_contrast.max(candidate.local_contrast);
            pending.consecutive_hits = pending.consecutive_hits.saturating_add(1);
        }

        let mut surviving_pending = Vec::with_capacity(pending_candidates.len());
        for pending in pending_candidates {
            let age_secs = frame_end - pending.first_seen_secs;
            let gap_secs = frame_end - pending.last_seen_secs;
            let candidate = NoteActivation {
                midi_note: pending.midi_note,
                salience: pending.best_salience,
                absolute_strength: pending.best_absolute_strength,
                brightness: pending.brightness,
                local_contrast: pending.local_contrast,
                register: pending.register,
            };
            let commit_floor = self.harmonic_birth_floor(&candidate)
                * (0.72 + 0.05 * self.collapse_birth_penalty(&candidate));
            let should_commit = pending.consecutive_hits >= HARMONIC_PENDING_CONFIRM_HITS
                && pending.best_absolute_strength >= commit_floor
                && pending.local_contrast >= HARMONIC_LOCAL_CONTRAST_MIN;
            let should_drop = gap_secs > HARMONIC_PENDING_MAX_GAP_SECS
                || age_secs > HARMONIC_PENDING_MAX_AGE_SECS;

            if should_commit {
                let committed = NoteActivation {
                    midi_note: pending.midi_note,
                    salience: pending.best_salience,
                    absolute_strength: pending.best_absolute_strength,
                    brightness: pending.brightness,
                    local_contrast: pending.local_contrast,
                    register: pending.register,
                };
                if !self.active_notes.iter().any(|note| {
                    note.kind == ActiveKind::Harmonic
                        && semitone_distance(note.midi_note, committed.midi_note)
                            <= NOTE_RELOCK_DISTANCE
                }) {
                    self.start_harmonic_note(committed, pending.first_seen_secs, None);
                }
                continue;
            }

            if !should_drop {
                surviving_pending.push(pending);
            }
        }

        self.pending_candidates.extend(surviving_pending);
    }

    fn compute_collapse_guard(&self, stats: SalienceFrameStats) -> f32 {
        let peak_ratio = if self.peak_salience_ref > 1e-4 {
            stats.global_peak / self.peak_salience_ref
        } else {
            1.0
        };
        let total_ratio = if self.total_salience_ref > 1e-3 {
            stats.total_energy / self.total_salience_ref
        } else {
            1.0
        };
        let peak_collapse =
            ((NORMALIZATION_COLLAPSE_PEAK_RATIO - peak_ratio) / NORMALIZATION_COLLAPSE_PEAK_RATIO)
                .max(0.0);
        let total_collapse = ((NORMALIZATION_COLLAPSE_TOTAL_RATIO - total_ratio)
            / NORMALIZATION_COLLAPSE_TOTAL_RATIO)
            .max(0.0);
        let combined_collapse = (0.6 * peak_collapse + 0.4 * total_collapse).clamp(0.0, 1.0);
        (self.collapse_guard * NORMALIZATION_COLLAPSE_GUARD_DECAY)
            .max(combined_collapse)
            .clamp(0.0, 1.0)
    }

    fn update_salience_references(&mut self, stats: SalienceFrameStats) {
        self.peak_salience_ref =
            update_reference_envelope(self.peak_salience_ref, stats.global_peak, HARMONIC_REFERENCE_DECAY);
        self.total_salience_ref = update_reference_envelope(
            self.total_salience_ref,
            stats.total_energy,
            HARMONIC_TOTAL_REFERENCE_DECAY,
        );
        for register in 0..self.register_peak_refs.len() {
            self.register_peak_refs[register] = update_reference_envelope(
                self.register_peak_refs[register],
                stats.register_peaks[register],
                HARMONIC_REFERENCE_DECAY,
            );
        }
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
            instrument_selection: self.current_harmonic_instrument_selection(),
            audio_energy: 0.0,
            audio_energy_envelope: Vec::new(),
        });
        self.pending_candidates.retain(|pending| {
            semitone_distance(pending.midi_note, candidate.midi_note) > NOTE_RELOCK_DISTANCE
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

        let onset_like = is_onset || kick || snare || hihat || onset_confidence >= 0.28;
        if !onset_like || percussive_confidence < 0.22 {
            return;
        }
        if self.collapse_guard >= 0.45 && onset_confidence < 0.45 && percussive_confidence < 0.55 {
            return;
        }

        if (kick || kick_strength >= 0.12)
            && percussive_confidence >= 0.24
            && self.drum_cooldowns[0] <= 0.0
        {
            self.push_drum_note(DRUM_KICK_NOTE, 0.18, start_secs, 0);
        }
        if (snare || snare_strength >= 0.12)
            && percussive_confidence >= 0.24
            && self.drum_cooldowns[1] <= 0.0
        {
            self.push_drum_note(DRUM_SNARE_NOTE, 0.45, start_secs, 1);
        }
        if (hihat || hihat_strength >= 0.10)
            && brightness >= 0.45
            && percussive_confidence >= 0.24
            && self.drum_cooldowns[2] <= 0.0
        {
            self.push_drum_note(DRUM_HIHAT_NOTE, 0.82, start_secs, 2);
        }
        if !kick
            && !snare
            && !hihat
            && onset_confidence >= 0.52
            && percussive_confidence >= 0.55
        {
            let (note, brightness, class_idx) = if kick_strength >= snare_strength
                && kick_strength >= hihat_strength
            {
                (DRUM_KICK_NOTE, 0.18, 0)
            } else if snare_strength >= hihat_strength {
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
            InstrumentSelection::Percussive,
            vec![self.analyser.features.percussive_confidence.max(0.3)],
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
            let weak_for_long = activation < HARMONIC_STOP_THRESHOLD
                && frame_end - note.last_seen_secs >= NOTE_RELEASE_HOLD_SECS * 0.5;
            if stale || weak_for_long {
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
            note.instrument_selection,
            note.audio_energy_envelope,
        );
    }

    fn find_best_harmonic_match(
        &self,
        candidate: &NoteActivation,
        matched_ids: &HashSet<usize>,
        frame_start: f32,
    ) -> Option<usize> {
        self.active_notes
            .iter()
            .enumerate()
            .filter(|(_, note)| {
                note.kind == ActiveKind::Harmonic && !matched_ids.contains(&note.id)
            })
            .filter(|(_, note)| {
                let max_distance = if frame_start - note.start_secs >= HARMONIC_PITCH_LOCK_SECS {
                    HARMONIC_LOCKED_MATCH_DISTANCE
                } else {
                    NOTE_SPLIT_DISTANCE
                };
                semitone_distance(note.midi_note, candidate.midi_note) <= max_distance
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
                instrument_selection: note.instrument_selection,
                audio_energy: note.audio_energy,
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
    evaluate_notes_detailed(predicted, reference).metrics
}

pub fn evaluate_notes_detailed(
    predicted: &[TranscribedNote],
    reference: &[GroundTruthNote],
) -> NoteDiagnostics {
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
    let mut events = Vec::new();

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
                    events.push(build_matched_event(
                        &predicted_sorted[i],
                        &reference_sorted[j],
                        assessment,
                    ));
                } else {
                    events.push(build_unmatched_predicted_event(
                        &predicted_sorted[i],
                        &reference_sorted,
                    ));
                    events.push(build_unmatched_reference_event(
                        &reference_sorted[j],
                        &predicted_sorted,
                    ));
                }
                i += 1;
                j += 1;
            }
            AlignmentStep::SkipPredicted => {
                events.push(build_unmatched_predicted_event(
                    &predicted_sorted[i],
                    &reference_sorted,
                ));
                i += 1;
            }
            AlignmentStep::SkipReference => {
                events.push(build_unmatched_reference_event(
                    &reference_sorted[j],
                    &predicted_sorted,
                ));
                j += 1;
            }
        }
    }

    while i < pred_len {
        events.push(build_unmatched_predicted_event(
            &predicted_sorted[i],
            &reference_sorted,
        ));
        i += 1;
    }
    while j < ref_len {
        events.push(build_unmatched_reference_event(
            &reference_sorted[j],
            &predicted_sorted,
        ));
        j += 1;
    }

    NoteDiagnostics {
        bucket_counts: bucket_counts_from_events(&events),
        error_summary: error_summary_from_events(&events),
        metrics,
        events,
    }
}

pub fn evaluate_source_separated(
    predicted: &[TranscribedNote],
    references: &[Vec<GroundTruthNote>],
) -> EvalSummary {
    let detailed = evaluate_source_separated_detailed(predicted, references);
    EvalSummary {
        per_source: detailed
            .per_source
            .iter()
            .map(|source| SourceEval {
                source_id: source.source_id,
                stream_id: source.stream_id,
                metrics: source.diagnostics.metrics.clone(),
            })
            .collect(),
        total: detailed.diagnostics.metrics,
    }
}

pub fn evaluate_source_separated_detailed(
    predicted: &[TranscribedNote],
    references: &[Vec<GroundTruthNote>],
) -> SourceSeparatedDiagnostics {
    let source_labels: Vec<_> = references
        .iter()
        .enumerate()
        .map(|(source_id, _)| format!("source_{source_id}"))
        .collect();
    evaluate_source_separated_detailed_with_labels(predicted, references, &source_labels)
}

fn evaluate_source_separated_detailed_with_labels(
    predicted: &[TranscribedNote],
    references: &[Vec<GroundTruthNote>],
    source_labels: &[String],
) -> SourceSeparatedDiagnostics {
    let predicted_by_stream = group_predicted_by_stream(predicted);
    let mut stream_ids: Vec<usize> = predicted_by_stream.keys().copied().collect();
    stream_ids.sort_unstable();

    let mut candidates = Vec::new();
    for (source_id, reference_notes) in references.iter().enumerate() {
        for &stream_id in &stream_ids {
            let diagnostics =
                evaluate_notes_detailed(&predicted_by_stream[&stream_id], reference_notes);
            let score = diagnostics.metrics.fuzzy_score_percent();
            candidates.push((score, source_id, stream_id, diagnostics));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut used_sources = HashSet::new();
    let mut used_streams = HashSet::new();
    let mut per_source: Vec<Option<SourceDiagnostic>> = vec![None; references.len()];

    for (_, source_id, stream_id, diagnostics) in candidates {
        if used_sources.contains(&source_id) || used_streams.contains(&stream_id) {
            continue;
        }
        if diagnostics.metrics.matched_similarity <= 0.0 {
            continue;
        }
        used_sources.insert(source_id);
        used_streams.insert(stream_id);
        per_source[source_id] = Some(SourceDiagnostic {
            source_id,
            label: source_labels
                .get(source_id)
                .cloned()
                .unwrap_or_else(|| format!("source_{source_id}")),
            stream_id: Some(stream_id),
            diagnostics,
        });
    }

    let mut resolved_sources = Vec::with_capacity(references.len());
    for (source_id, reference_notes) in references.iter().enumerate() {
        let source = per_source[source_id].clone().unwrap_or_else(|| SourceDiagnostic {
            source_id,
            label: source_labels
                .get(source_id)
                .cloned()
                .unwrap_or_else(|| format!("source_{source_id}")),
            stream_id: None,
            diagnostics: evaluate_notes_detailed(&[], reference_notes),
        });
        resolved_sources.push(source);
    }

    let mut unassigned_streams = Vec::new();
    for stream_id in stream_ids {
        if used_streams.contains(&stream_id) {
            continue;
        }
        let diagnostics = evaluate_notes_detailed(&predicted_by_stream[&stream_id], &[]);
        unassigned_streams.push(StreamDiagnostic {
            stream_id,
            diagnostics,
        });
    }

    let diagnostics = merge_note_diagnostics(
        resolved_sources
            .iter()
            .map(|source| &source.diagnostics)
            .chain(unassigned_streams.iter().map(|stream| &stream.diagnostics)),
    );

    SourceSeparatedDiagnostics {
        diagnostics,
        per_source: resolved_sources,
        unassigned_streams,
    }
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

fn build_matched_event(
    predicted: &TranscribedNote,
    reference: &GroundTruthNote,
    assessment: MatchAssessment,
) -> DiagnosticEvent {
    DiagnosticEvent {
        bucket: classify_matched_bucket(predicted, reference, assessment),
        source_id: Some(reference.source_id),
        stream_id: Some(predicted.stream_id),
        matched: true,
        similarity: assessment.similarity,
        pitch_error_semitones: Some(assessment.pitch_error_semitones),
        start_error_ms: Some(assessment.start_error_secs * 1000.0),
        end_error_ms: Some(assessment.end_error_secs * 1000.0),
        predicted: Some(predicted.clone()),
        reference: Some(reference.clone()),
    }
}

fn build_unmatched_predicted_event(
    predicted: &TranscribedNote,
    reference_notes: &[GroundTruthNote],
) -> DiagnosticEvent {
    DiagnosticEvent {
        bucket: classify_unmatched_predicted_bucket(predicted, reference_notes),
        source_id: None,
        stream_id: Some(predicted.stream_id),
        matched: false,
        similarity: 0.0,
        pitch_error_semitones: None,
        start_error_ms: None,
        end_error_ms: None,
        predicted: Some(predicted.clone()),
        reference: None,
    }
}

fn build_unmatched_reference_event(
    reference: &GroundTruthNote,
    predicted_notes: &[TranscribedNote],
) -> DiagnosticEvent {
    DiagnosticEvent {
        bucket: classify_unmatched_reference_bucket(reference, predicted_notes),
        source_id: Some(reference.source_id),
        stream_id: None,
        matched: false,
        similarity: 0.0,
        pitch_error_semitones: None,
        start_error_ms: None,
        end_error_ms: None,
        predicted: None,
        reference: Some(reference.clone()),
    }
}

fn classify_matched_bucket(
    predicted: &TranscribedNote,
    reference: &GroundTruthNote,
    assessment: MatchAssessment,
) -> FailureBucket {
    if assessment.full_note_match {
        return FailureBucket::Correct;
    }

    if assessment.pitch_error_semitones >= 11.5
        && (assessment.pitch_error_semitones % 12.0) <= 0.5
    {
        return FailureBucket::OctaveError;
    }
    if assessment.pitch_error_semitones >= 0.5 {
        return FailureBucket::PitchError;
    }

    let start_delta = predicted.start_secs - reference.start_secs;
    let end_delta = predicted.end_secs - reference.end_secs;
    let start_bad = start_delta.abs() > START_TOLERANCE_SECS;
    let end_bad = end_delta.abs() > END_TOLERANCE_SECS;

    match (start_bad, end_bad) {
        (true, true) => FailureBucket::TimingDrift,
        (true, false) if start_delta < 0.0 => FailureBucket::EarlyOnset,
        (true, false) => FailureBucket::LateOnset,
        (false, true) if end_delta < 0.0 => FailureBucket::EarlyRelease,
        (false, true) => FailureBucket::LateRelease,
        _ => FailureBucket::TimingDrift,
    }
}

fn classify_unmatched_predicted_bucket(
    predicted: &TranscribedNote,
    reference_notes: &[GroundTruthNote],
) -> FailureBucket {
    let overlaps_same_pitch = reference_notes
        .iter()
        .any(|reference| reference.midi_note == predicted.midi_note && notes_overlap(predicted, reference));
    if overlaps_same_pitch {
        FailureBucket::SplitLikely
    } else {
        FailureBucket::ExtraNote
    }
}

fn classify_unmatched_reference_bucket(
    reference: &GroundTruthNote,
    predicted_notes: &[TranscribedNote],
) -> FailureBucket {
    let overlaps_same_pitch = predicted_notes
        .iter()
        .any(|predicted| predicted.midi_note == reference.midi_note && notes_overlap(predicted, reference));
    if overlaps_same_pitch {
        FailureBucket::MergeLikely
    } else {
        FailureBucket::MissedNote
    }
}

fn notes_overlap(predicted: &TranscribedNote, reference: &GroundTruthNote) -> bool {
    let overlap_start = predicted.start_secs.max(reference.start_secs);
    let overlap_end = predicted.end_secs.min(reference.end_secs);
    let overlap = (overlap_end - overlap_start).max(0.0);
    let min_duration = (predicted.end_secs - predicted.start_secs)
        .min(reference.end_secs - reference.start_secs)
        .max(1e-3);
    overlap / min_duration >= 0.35
}

fn bucket_counts_from_events(events: &[DiagnosticEvent]) -> Vec<BucketCount> {
    let mut counts: BTreeMap<FailureBucket, usize> = BTreeMap::new();
    for event in events {
        *counts.entry(event.bucket).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(bucket, count)| BucketCount { bucket, count })
        .collect()
}

fn error_summary_from_events(events: &[DiagnosticEvent]) -> ErrorSummary {
    let pitch_values: Vec<f32> = events
        .iter()
        .filter_map(|event| event.pitch_error_semitones)
        .collect();
    let start_values: Vec<f32> = events
        .iter()
        .filter_map(|event| event.start_error_ms)
        .collect();
    let end_values: Vec<f32> = events
        .iter()
        .filter_map(|event| event.end_error_ms)
        .collect();

    ErrorSummary {
        pitch_error_semitones: summarize_error_values(&pitch_values),
        start_error_ms: summarize_error_values(&start_values),
        end_error_ms: summarize_error_values(&end_values),
    }
}

fn summarize_error_values(values: &[f32]) -> ErrorStats {
    if values.is_empty() {
        return ErrorStats::default();
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let count = sorted.len();
    let mean = sorted.iter().sum::<f32>() / count as f32;

    ErrorStats {
        count,
        mean,
        p50: percentile(&sorted, 0.50),
        p90: percentile(&sorted, 0.90),
        max: sorted.last().copied().unwrap_or(0.0),
    }
}

fn percentile(sorted: &[f32], quantile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn merge_note_diagnostics<'a>(
    diagnostics: impl IntoIterator<Item = &'a NoteDiagnostics>,
) -> NoteDiagnostics {
    let mut metrics = NoteMetrics::default();
    let mut events = Vec::new();
    for diagnostic in diagnostics {
        accumulate_note_metrics(&mut metrics, &diagnostic.metrics);
        events.extend(diagnostic.events.iter().cloned());
    }
    NoteDiagnostics {
        bucket_counts: bucket_counts_from_events(&events),
        error_summary: error_summary_from_events(&events),
        metrics,
        events,
    }
}

fn note_evidence(cqt: &[f32]) -> NoteEvidenceFrame {
    let bank = note_template_bank();
    let frame: Vec<f32> = cqt.iter().map(|&value| value.max(0.0).sqrt()).collect();
    let frame_norm = frame.iter().map(|value| value * value).sum::<f32>().sqrt();
    let mut salience = [0.0f32; 128];
    let mut absolute_strength = [0.0f32; 128];
    if frame_norm <= 1e-6 {
        return NoteEvidenceFrame {
            salience,
            absolute_strength,
        };
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
        let projection = dot.clamp(0.0, 1.0);
        salience[midi_note as usize] =
            (0.55 * similarity + 0.45 * fundamental - 0.10 * sub_octave).clamp(0.0, 1.0);
        absolute_strength[midi_note as usize] =
            (0.65 * projection + 0.35 * fundamental - 0.08 * sub_octave).clamp(0.0, 1.0);
    }
    NoteEvidenceFrame {
        salience,
        absolute_strength,
    }
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
    absolute_strength: &[f32; 128],
    note_energy_ema: &[f32; 128],
    peak_salience_ref: f32,
    register_peak_refs: &[f32; 3],
    collapse_guard: f32,
    harmonic_confidence: f32,
    pitched_stability: f32,
    max_streams: usize,
) -> Vec<NoteActivation> {
    let global_max = absolute_strength
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
        let absolute_center = absolute_strength[idx];
        let register = pitch_register(midi_note);
        if center < salience[idx - 1] || center < salience[idx + 1] {
            continue;
        }
        let local_contrast = (center - 0.5 * (salience[idx - 1] + salience[idx + 1])).max(0.0);
        if local_contrast < HARMONIC_LOCAL_CONTRAST_MIN {
            continue;
        }
        let dynamic_floor = (global_max * 0.24)
            .max(note_energy_ema[idx] * 0.84)
            .max(peak_salience_ref * (0.06 + 0.08 * collapse_guard))
            .max(register_peak_refs[register] * (0.14 + 0.10 * collapse_guard))
            .max(if harmonic_confidence >= 0.2 || pitched_stability >= 0.2 {
                0.04
            } else {
                0.055
            });
        if absolute_center < dynamic_floor {
            continue;
        }
        if midi_note <= 40 && salience[(idx + 12).min(127)] > center * 1.12 {
            continue;
        }

        peaks.push(NoteActivation {
            midi_note,
            salience: center,
            absolute_strength: absolute_center,
            brightness: idx as f32 / TRANSCRIPTION_MIDI_HIGH as f32,
            local_contrast,
            register,
        });
    }

    peaks.sort_by(|a, b| {
        (0.55 * b.salience + 0.45 * b.absolute_strength)
            .partial_cmp(&(0.55 * a.salience + 0.45 * a.absolute_strength))
            .unwrap_or(Ordering::Equal)
    });
    let mut filtered = Vec::with_capacity(peaks.len());
    for candidate in peaks {
        if filtered.iter().any(|selected: &NoteActivation| {
            semitone_distance(selected.midi_note, candidate.midi_note) <= 1
                && selected.salience >= candidate.salience * 0.92
        }) {
            continue;
        }
        filtered.push(candidate);
        if filtered.len() >= MAX_POLYPHONIC_CANDIDATES.max(max_streams * 2) {
            break;
        }
    }
    filtered
}

fn evidence_frame_stats(evidence: &[f32; 128]) -> SalienceFrameStats {
    let mut stats = SalienceFrameStats::default();
    for midi_note in TRANSCRIPTION_MIDI_LOW..=TRANSCRIPTION_MIDI_HIGH {
        let value = evidence[midi_note as usize];
        stats.global_peak = stats.global_peak.max(value);
        stats.total_energy += value;
        let register = pitch_register(midi_note);
        stats.register_peaks[register] = stats.register_peaks[register].max(value);
    }
    stats
}

fn pitch_register(midi_note: u8) -> usize {
    match midi_note {
        ..=45 => 0,
        46..=69 => 1,
        _ => 2,
    }
}

fn update_reference_envelope(previous: f32, current: f32, decay: f32) -> f32 {
    if current >= previous {
        previous * 0.60 + current * 0.40
    } else {
        (previous * decay).max(current)
    }
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

fn stream_distance(stream: &StreamState, midi_note: u8, brightness: f32) -> f32 {
    ((stream.note_center - midi_note as f32).abs() / STREAM_NOTE_DISTANCE_NORM)
        + (stream.brightness_center - brightness).abs()
}

fn semitone_distance(a: u8, b: u8) -> i16 {
    (a as i16 - b as i16).abs()
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

pub fn build_standardized_evaluation_report(
    dir: &Path,
) -> Result<EvaluationSuiteReport, Box<dyn Error>> {
    let pairs = standardized_test_pairs(dir)?;
    let mut cases = Vec::new();

    for pair in &pairs {
        let predicted = transcribe_clip(&pair.clip, 1);
        let diagnostics = evaluate_notes_detailed(&predicted, &pair.notes);
        let tags = single_source_tags(pair);
        cases.push(EvaluationCaseReport {
            label: pair.pair.stem.clone(),
            kind: EvaluationCaseKind::SingleSource,
            source_count: 1,
            sample_rate: pair.clip.sample_rate,
            tags: tags.clone(),
            per_source: vec![SourceDiagnostic {
                source_id: 0,
                label: pair.pair.stem.clone(),
                stream_id: None,
                diagnostics: diagnostics.clone(),
            }],
            unassigned_streams: Vec::new(),
            diagnostics,
        });
    }

    for &mix_count in STANDARD_MIX_SIZES {
        let mix_index_sets = standardized_mix_index_sets(&pairs, mix_count)?;
        for mix_indices in &mix_index_sets {
            let clips: Vec<_> = mix_indices
                .iter()
                .map(|&pair_idx| pairs[pair_idx].clip.clone())
                .collect();
            let mixed = mix_clips(&clips)?;
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
            let source_labels: Vec<_> = mix_indices
                .iter()
                .map(|&pair_idx| pairs[pair_idx].pair.stem.clone())
                .collect();
            let detailed =
                evaluate_source_separated_detailed_with_labels(&predicted, &references, &source_labels);
            cases.push(EvaluationCaseReport {
                label: mix_label(&pairs, mix_indices),
                kind: match mix_count {
                    2 => EvaluationCaseKind::Mix2Source,
                    3 => EvaluationCaseKind::Mix3Source,
                    _ => unreachable!("standardized mix sizes are validated"),
                },
                source_count: mix_count,
                sample_rate: mixed.sample_rate,
                tags: mix_tags(mix_count),
                diagnostics: detailed.diagnostics,
                per_source: detailed.per_source,
                unassigned_streams: detailed.unassigned_streams,
            });
        }
    }

    let overall = build_slice_report("overall", &cases);
    let mut slices = Vec::new();
    for (label, tag) in [
        ("single_source", "single_source"),
        ("mix_2_source", "mix_2_source"),
        ("mix_3_source", "mix_3_source"),
        ("single_monophonic", "single_monophonic"),
        ("single_polyphonic", "single_polyphonic"),
        ("single_percussive", "single_percussive"),
        ("single_pitched", "single_pitched"),
    ] {
        if let Some(slice) = build_tagged_slice_report(label, &cases, tag) {
            slices.push(slice);
        }
    }

    Ok(EvaluationSuiteReport {
        dataset_dir: dir.display().to_string(),
        overall,
        slices,
        cases,
    })
}

pub fn format_evaluation_report(report: &EvaluationSuiteReport) -> String {
    let mut out = String::new();
    let overall = &report.overall.diagnostics.metrics;
    out.push_str(&format!(
        "Dataset: {}\nOverall: {:>6.2}%  predicted {:>4}  reference {:>4}  matched_pairs {:>4}  start {:>6.1}ms  end {:>6.1}ms  pitch {:>5.1}c\n",
        report.dataset_dir,
        overall.fuzzy_score_percent(),
        overall.predicted_notes,
        overall.reference_notes,
        overall.matched_pairs,
        overall.mean_start_error_ms(),
        overall.mean_end_error_ms(),
        overall.mean_pitch_error_semitones() * 100.0,
    ));
    out.push_str(&format!(
        "Top failure buckets: {}\n",
        format_bucket_counts(&report.overall.diagnostics.bucket_counts, 6)
    ));

    out.push_str("\nSlices:\n");
    for slice in &report.slices {
        let metrics = &slice.diagnostics.metrics;
        out.push_str(&format!(
            "  {:<20} {:>6.2}%  cases {:>2}  matched_pairs {:>4}  buckets {}\n",
            slice.label,
            metrics.fuzzy_score_percent(),
            slice.case_count,
            metrics.matched_pairs,
            format_bucket_counts(&slice.diagnostics.bucket_counts, 4),
        ));
    }

    let mut worst_cases = report.cases.iter().collect::<Vec<_>>();
    worst_cases.sort_by(|a, b| {
        a.diagnostics
            .metrics
            .fuzzy_score_percent()
            .partial_cmp(&b.diagnostics.metrics.fuzzy_score_percent())
            .unwrap_or(Ordering::Equal)
    });
    out.push_str("\nWorst cases:\n");
    for case in worst_cases.into_iter().take(8) {
        let metrics = &case.diagnostics.metrics;
        out.push_str(&format!(
            "  {:<100} {:>6.2}%  pitch {:>5.1}c  start {:>6.1}ms  end {:>6.1}ms  {}\n",
            case.label,
            metrics.fuzzy_score_percent(),
            metrics.mean_pitch_error_semitones() * 100.0,
            metrics.mean_start_error_ms(),
            metrics.mean_end_error_ms(),
            format_bucket_counts(&case.diagnostics.bucket_counts, 3),
        ));
    }

    out
}

pub fn summarize_evaluation_report(report: &EvaluationSuiteReport) -> EvaluationSuiteSummary {
    EvaluationSuiteSummary {
        dataset_dir: report.dataset_dir.clone(),
        overall: EvaluationSliceSummary {
            label: report.overall.label.clone(),
            case_count: report.overall.case_count,
            diagnostics: summarize_note_diagnostics(&report.overall.diagnostics, 8),
        },
        slices: report
            .slices
            .iter()
            .map(|slice| EvaluationSliceSummary {
                label: slice.label.clone(),
                case_count: slice.case_count,
                diagnostics: summarize_note_diagnostics(&slice.diagnostics, 6),
            })
            .collect(),
        cases: report
            .cases
            .iter()
            .map(|case| EvaluationCaseSummary {
                label: case.label.clone(),
                kind: case.kind,
                source_count: case.source_count,
                diagnostics: summarize_note_diagnostics(&case.diagnostics, 4),
            })
            .collect(),
    }
}

fn summarize_note_diagnostics(
    diagnostics: &NoteDiagnostics,
    bucket_limit: usize,
) -> DiagnosticSummary {
    DiagnosticSummary {
        metrics: summarize_note_metrics(&diagnostics.metrics),
        bucket_counts: trim_bucket_counts(&diagnostics.bucket_counts, bucket_limit),
        error_summary: summarize_error_summary(&diagnostics.error_summary),
    }
}

fn summarize_note_metrics(metrics: &NoteMetrics) -> NoteMetricSummary {
    NoteMetricSummary {
        score_percent: round2(metrics.fuzzy_score_percent()),
        predicted_notes: metrics.predicted_notes,
        reference_notes: metrics.reference_notes,
        matched_pairs: metrics.matched_pairs,
        matched_onsets: metrics.matched_onsets,
        matched_notes: metrics.matched_notes,
        mean_pitch_error_cents: round1(metrics.mean_pitch_error_semitones() * 100.0),
        mean_start_error_ms: round1(metrics.mean_start_error_ms()),
        mean_end_error_ms: round1(metrics.mean_end_error_ms()),
    }
}

fn summarize_error_summary(summary: &ErrorSummary) -> ErrorSummary {
    ErrorSummary {
        pitch_error_semitones: summarize_error_stats(&summary.pitch_error_semitones),
        start_error_ms: summarize_error_stats(&summary.start_error_ms),
        end_error_ms: summarize_error_stats(&summary.end_error_ms),
    }
}

fn summarize_error_stats(stats: &ErrorStats) -> ErrorStats {
    ErrorStats {
        count: stats.count,
        mean: round2(stats.mean),
        p50: round2(stats.p50),
        p90: round2(stats.p90),
        max: round2(stats.max),
    }
}

fn trim_bucket_counts(bucket_counts: &[BucketCount], limit: usize) -> Vec<BucketCount> {
    let mut trimmed = bucket_counts.to_vec();
    trimmed.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.bucket.cmp(&b.bucket)));
    if limit > 0 && trimmed.len() > limit {
        trimmed.truncate(limit);
    }
    trimmed
}

fn round1(value: f32) -> f32 {
    (value * 10.0).round() / 10.0
}

fn round2(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}

fn build_tagged_slice_report(
    label: &str,
    cases: &[EvaluationCaseReport],
    tag: &str,
) -> Option<EvaluationSliceReport> {
    let tagged: Vec<_> = cases
        .iter()
        .filter(|case| case.tags.iter().any(|case_tag| case_tag == tag))
        .collect();
    if tagged.is_empty() {
        None
    } else {
        Some(build_slice_report_from_refs(label, &tagged))
    }
}

fn build_slice_report(label: &str, cases: &[EvaluationCaseReport]) -> EvaluationSliceReport {
    let refs = cases.iter().collect::<Vec<_>>();
    build_slice_report_from_refs(label, &refs)
}

fn build_slice_report_from_refs(
    label: &str,
    cases: &[&EvaluationCaseReport],
) -> EvaluationSliceReport {
    EvaluationSliceReport {
        label: label.to_owned(),
        case_count: cases.len(),
        diagnostics: merge_note_diagnostics(cases.iter().map(|case| &case.diagnostics)),
    }
}

fn single_source_tags(pair: &LoadedPair) -> Vec<String> {
    let mut tags = vec!["single_source".to_owned()];
    if pair.max_polyphony <= 1 {
        tags.push("single_monophonic".to_owned());
    } else {
        tags.push("single_polyphonic".to_owned());
    }
    let stem = pair.pair.stem.to_ascii_lowercase();
    if PERCUSSION_KEYWORDS.iter().any(|keyword| stem.contains(keyword)) {
        tags.push("single_percussive".to_owned());
    } else {
        tags.push("single_pitched".to_owned());
    }
    tags
}

fn mix_tags(mix_count: usize) -> Vec<String> {
    vec![format!("mix_{mix_count}_source")]
}

fn format_bucket_counts(bucket_counts: &[BucketCount], limit: usize) -> String {
    let mut buckets = bucket_counts.to_vec();
    buckets.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.bucket.cmp(&b.bucket)));
    if buckets.is_empty() {
        return "none".to_owned();
    }
    buckets
        .into_iter()
        .take(limit)
        .map(|bucket| format!("{:?}={}", bucket.bucket, bucket.count))
        .collect::<Vec<_>>()
        .join(", ")
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
            instrument_selection: InstrumentSelection::Harmonic,
            audio_energy_hop_secs: 0.0,
            audio_energy_envelope: Vec::new(),
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
            instrument_selection: InstrumentSelection::Harmonic,
            audio_energy_hop_secs: 0.0,
            audio_energy_envelope: Vec::new(),
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
    fn detailed_evaluation_buckets_common_failures() {
        let reference = vec![
            GroundTruthNote {
                source_id: 0,
                midi_note: 60,
                start_secs: 0.0,
                end_secs: 0.5,
            },
            GroundTruthNote {
                source_id: 0,
                midi_note: 64,
                start_secs: 1.0,
                end_secs: 1.4,
            },
        ];
        let predicted = vec![
            TranscribedNote {
                stream_id: 0,
                midi_note: 72,
                start_secs: 0.0,
                end_secs: 0.5,
                confidence: 1.0,
                instrument_selection: InstrumentSelection::Harmonic,
                audio_energy_hop_secs: 0.0,
                audio_energy_envelope: Vec::new(),
            },
            TranscribedNote {
                stream_id: 0,
                midi_note: 67,
                start_secs: 1.02,
                end_secs: 1.10,
                confidence: 0.8,
                instrument_selection: InstrumentSelection::Harmonic,
                audio_energy_hop_secs: 0.0,
                audio_energy_envelope: Vec::new(),
            },
            TranscribedNote {
                stream_id: 0,
                midi_note: 69,
                start_secs: 2.0,
                end_secs: 2.2,
                confidence: 0.6,
                instrument_selection: InstrumentSelection::Harmonic,
                audio_energy_hop_secs: 0.0,
                audio_energy_envelope: Vec::new(),
            },
        ];

        let diagnostics = evaluate_notes_detailed(&predicted, &reference);
        let buckets: HashSet<_> = diagnostics.events.iter().map(|event| event.bucket).collect();
        assert!(buckets.contains(&FailureBucket::OctaveError));
        assert!(buckets.contains(&FailureBucket::PitchError));
        assert!(buckets.contains(&FailureBucket::ExtraNote));
    }

    #[test]
    fn formatted_report_contains_overall_and_slices() {
        let diagnostics = evaluate_notes_detailed(&[], &[]);
        let report = EvaluationSuiteReport {
            dataset_dir: "midi".to_owned(),
            overall: EvaluationSliceReport {
                label: "overall".to_owned(),
                case_count: 1,
                diagnostics: diagnostics.clone(),
            },
            slices: vec![EvaluationSliceReport {
                label: "single_source".to_owned(),
                case_count: 1,
                diagnostics: diagnostics.clone(),
            }],
            cases: vec![EvaluationCaseReport {
                label: "fixture".to_owned(),
                kind: EvaluationCaseKind::SingleSource,
                source_count: 1,
                sample_rate: 44_100,
                tags: vec!["single_source".to_owned()],
                diagnostics,
                per_source: Vec::new(),
                unassigned_streams: Vec::new(),
            }],
        };

        let formatted = format_evaluation_report(&report);
        assert!(formatted.contains("Overall"));
        assert!(formatted.contains("Slices"));
        assert!(formatted.contains("Worst cases"));
    }

    #[test]
    fn strong_birth_stays_immediate_through_collapse_guard() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.peak_salience_ref = 0.82;
        transcriber.total_salience_ref = 2.8;
        transcriber.register_peak_refs = [0.78, 0.46, 0.32];
        transcriber.collapse_guard = 0.75;
        transcriber.analyser.features.onset_confidence = 0.36;

        let candidate = NoteActivation {
            midi_note: 86,
            salience: 0.44,
            absolute_strength: 0.44,
            brightness: 0.82,
            local_contrast: 0.09,
            register: pitch_register(86),
        };

        assert_eq!(
            transcriber.classify_harmonic_birth(&candidate),
            BirthDecision::Immediate
        );
    }

    #[test]
    fn weak_residual_birth_does_not_commit_after_normalization_collapse() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.peak_salience_ref = 0.88;
        transcriber.total_salience_ref = 3.2;
        transcriber.register_peak_refs = [0.82, 0.40, 0.22];
        transcriber.collapse_guard = 0.80;
        transcriber.analyser.features.harmonic_confidence = 0.24;
        transcriber.analyser.features.pitched_stability = 0.24;

        let candidate = NoteActivation {
            midi_note: 92,
            salience: 0.10,
            absolute_strength: 0.10,
            brightness: 0.88,
            local_contrast: 0.016,
            register: pitch_register(92),
        };

        assert_ne!(
            transcriber.classify_harmonic_birth(&candidate),
            BirthDecision::Immediate
        );
        transcriber.update_pending_harmonic_candidates(&[candidate], 0.00, 0.012);
        assert!(transcriber.active_notes.is_empty());
        transcriber.update_pending_harmonic_candidates(&[], 0.012, 0.080);
        assert!(transcriber.active_notes.is_empty());
    }

    #[test]
    fn ambiguous_pending_birth_commits_after_two_hits() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.analyser.features.onset_confidence = 0.09;
        transcriber.analyser.features.harmonic_confidence = 0.26;
        transcriber.analyser.features.pitched_stability = 0.27;

        let candidate = NoteActivation {
            midi_note: 64,
            salience: 0.12,
            absolute_strength: 0.12,
            brightness: 0.52,
            local_contrast: 0.020,
            register: pitch_register(64),
        };

        assert_eq!(
            transcriber.classify_harmonic_birth(&candidate),
            BirthDecision::Pending
        );

        transcriber.update_pending_harmonic_candidates(&[candidate], 0.00, 0.012);
        assert!(transcriber.active_notes.is_empty());
        assert_eq!(transcriber.pending_candidates.len(), 1);

        transcriber.update_pending_harmonic_candidates(&[candidate], 0.012, 0.024);
        assert_eq!(transcriber.active_notes.len(), 1);
        assert!(transcriber.pending_candidates.is_empty());
        assert!(transcriber.active_notes[0].start_secs <= 0.001);
    }

    #[test]
    fn collapse_guard_limits_weak_birth_fanout() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.peak_salience_ref = 0.92;
        transcriber.total_salience_ref = 3.5;
        transcriber.register_peak_refs = [0.84, 0.42, 0.24];
        transcriber.collapse_guard = 0.82;
        transcriber.analyser.features.harmonic_confidence = 0.26;
        transcriber.analyser.features.pitched_stability = 0.25;
        transcriber.analyser.features.onset_confidence = 0.05;

        let candidates = vec![
            NoteActivation {
                midi_note: 89,
                salience: 0.125,
                absolute_strength: 0.125,
                brightness: 0.86,
                local_contrast: 0.021,
                register: pitch_register(89),
            },
            NoteActivation {
                midi_note: 92,
                salience: 0.118,
                absolute_strength: 0.118,
                brightness: 0.89,
                local_contrast: 0.019,
                register: pitch_register(92),
            },
            NoteActivation {
                midi_note: 96,
                salience: 0.111,
                absolute_strength: 0.111,
                brightness: 0.92,
                local_contrast: 0.018,
                register: pitch_register(96),
            },
        ];

        transcriber.update_harmonic_notes(&candidates, 0.0, 0.012);
        assert!(transcriber.active_notes.is_empty());
        assert_eq!(transcriber.pending_candidates.len(), 1);
        assert_eq!(transcriber.pending_candidates[0].midi_note, 89);
    }

    #[test]
    fn mature_note_keeps_pitch_authority_and_new_pitch_births_separately() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.analyser.features.onset_confidence = 0.34;
        transcriber.analyser.features.harmonic_confidence = 0.34;
        transcriber.analyser.features.pitched_stability = 0.34;
        transcriber.peak_salience_ref = 0.30;
        transcriber.total_salience_ref = 1.8;
        transcriber.register_peak_refs = [0.18, 0.28, 0.30];
        transcriber.active_notes.push(RealtimeActiveNote {
            id: 0,
            stream_id: 0,
            midi_note: 60,
            start_secs: 0.0,
            last_seen_secs: HARMONIC_PITCH_LOCK_SECS,
            confidence: 0.22,
            brightness: 0.56,
            kind: ActiveKind::Harmonic,
            instrument_selection: InstrumentSelection::Harmonic,
            audio_energy: 0.22,
            audio_energy_envelope: vec![0.22],
        });
        transcriber.next_note_id = 1;

        let candidate = NoteActivation {
            midi_note: 62,
            salience: 0.30,
            absolute_strength: 0.30,
            brightness: 0.58,
            local_contrast: 0.05,
            register: pitch_register(62),
        };

        transcriber.update_harmonic_notes(
            &[candidate],
            HARMONIC_PITCH_LOCK_SECS + 0.012,
            HARMONIC_PITCH_LOCK_SECS + 0.024,
        );

        assert_eq!(transcriber.active_notes.len(), 2);
        assert!(transcriber.active_notes.iter().any(|note| note.midi_note == 60));
        assert!(transcriber.active_notes.iter().any(|note| note.midi_note == 62));
    }

    #[test]
    fn fresh_note_can_still_relock_during_short_settle_window() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.active_notes.push(RealtimeActiveNote {
            id: 0,
            stream_id: 0,
            midi_note: 60,
            start_secs: 0.0,
            last_seen_secs: 0.012,
            confidence: 0.20,
            brightness: 0.54,
            kind: ActiveKind::Harmonic,
            instrument_selection: InstrumentSelection::Harmonic,
            audio_energy: 0.20,
            audio_energy_envelope: vec![0.20],
        });

        let candidate = NoteActivation {
            midi_note: 62,
            salience: 0.28,
            absolute_strength: 0.28,
            brightness: 0.56,
            local_contrast: 0.04,
            register: pitch_register(62),
        };

        transcriber.update_harmonic_notes(&[candidate], 0.020, 0.032);

        assert_eq!(transcriber.active_notes.len(), 1);
        assert_eq!(transcriber.active_notes[0].midi_note, 62);
    }

    #[test]
    fn committing_pending_note_can_prune_neighbors_without_panicking() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.analyser.features.onset_confidence = 0.16;
        transcriber.analyser.features.harmonic_confidence = 0.24;
        transcriber.analyser.features.pitched_stability = 0.24;
        transcriber.pending_candidates = vec![
            PendingHarmonicCandidate {
                midi_note: 60,
                first_seen_secs: 0.0,
                last_seen_secs: 0.024,
                best_salience: 0.22,
                best_absolute_strength: 0.22,
                brightness: 0.52,
                local_contrast: 0.022,
                consecutive_hits: HARMONIC_PENDING_CONFIRM_HITS,
                register: pitch_register(60),
            },
            PendingHarmonicCandidate {
                midi_note: 61,
                first_seen_secs: 0.0,
                last_seen_secs: 0.024,
                best_salience: 0.23,
                best_absolute_strength: 0.23,
                brightness: 0.54,
                local_contrast: 0.023,
                consecutive_hits: HARMONIC_PENDING_CONFIRM_HITS,
                register: pitch_register(61),
            },
        ];

        transcriber.update_pending_harmonic_candidates(&[], 0.024, 0.036);

        assert_eq!(transcriber.active_notes.len(), 1);
        assert!(transcriber.pending_candidates.is_empty());
        assert!(matches!(
            transcriber.active_notes[0].midi_note,
            60 | 61
        ));
    }

    #[test]
    fn strong_onset_can_still_start_multiple_notes_during_collapse() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.peak_salience_ref = 0.88;
        transcriber.total_salience_ref = 3.0;
        transcriber.register_peak_refs = [0.82, 0.44, 0.28];
        transcriber.collapse_guard = 0.78;
        transcriber.analyser.features.onset_confidence = 0.38;
        transcriber.analyser.features.harmonic_confidence = 0.34;
        transcriber.analyser.features.pitched_stability = 0.32;

        let candidates = vec![
            NoteActivation {
                midi_note: 69,
                salience: 0.46,
                absolute_strength: 0.46,
                brightness: 0.66,
                local_contrast: 0.11,
                register: pitch_register(69),
            },
            NoteActivation {
                midi_note: 76,
                salience: 0.41,
                absolute_strength: 0.41,
                brightness: 0.73,
                local_contrast: 0.09,
                register: pitch_register(76),
            },
        ];

        transcriber.update_harmonic_notes(&candidates, 0.0, 0.012);
        assert_eq!(transcriber.active_notes.len(), 2);
    }

    #[test]
    fn bright_residual_does_not_emit_fallback_percussion() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.analyser.features.is_onset = true;
        transcriber.analyser.features.onset_confidence = 0.32;
        transcriber.analyser.features.percussive_confidence = 0.34;
        transcriber.analyser.features.brightness = 0.82;
        transcriber.emit_percussive_notes(0.0);
        assert!(transcriber.finished_notes.is_empty());
    }

    #[test]
    fn strong_hihat_still_emits_immediately() {
        let mut transcriber = RealtimeTranscriber::new(44_100, 4);
        transcriber.analyser.features.hihat = true;
        transcriber.analyser.features.hihat_strength = 0.18;
        transcriber.analyser.features.onset_confidence = 0.36;
        transcriber.analyser.features.percussive_confidence = 0.42;
        transcriber.analyser.features.brightness = 0.83;
        transcriber.emit_percussive_notes(0.0);
        assert_eq!(transcriber.finished_notes.len(), 1);
        assert_eq!(transcriber.finished_notes[0].note.midi_note, DRUM_HIHAT_NOTE);
    }

    #[test]
    #[ignore]
    fn report_current_dataset_scores() {
        let report = build_standardized_evaluation_report(Path::new(DATASET_DIR))
            .expect("failed to build standardized transcription report");
        eprintln!("{}", format_evaluation_report(&report));
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
