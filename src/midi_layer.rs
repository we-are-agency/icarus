use crate::analysis::FFT_HISTORY;
use crate::objects::{SoundKind, SoundObject};
use std::collections::{HashMap, HashSet, VecDeque};

const MAX_TRACKS: usize = 16;
/// Harmonic/PH notes extend while energy stays above this (sustain tracking).
const HARM_NOTE_THRESHOLD: f32 = 0.10;
/// Percussive notes extend only while energy is high — makes tick width proportional
/// to hit strength. Minimum onset_strength is 0.5, so quiet ticks → near-zero width.
const PERC_NOTE_THRESHOLD: f32 = 0.55;

struct NoteEvent {
    cluster_id: usize,
    spawn_id: usize,
    kind: SoundKind,
    hue: f32,
    start_secs: f32,
    end_secs: f32,
    alive: bool,
}

/// One lane in the MIDI panel.
pub struct TrackInfo {
    pub cluster_id: usize,
    pub kind: SoundKind,
    pub hue: f32,
}

/// Note ready for rendering with its lane resolved.
pub struct DrawNote {
    pub track_idx: usize,
    pub hue: f32,
    pub kind: SoundKind,
    pub start_secs: f32,
    pub end_secs: f32,
    pub alive: bool,
}

fn kind_group(k: SoundKind) -> u8 {
    match k {
        SoundKind::Percussive         => 0,
        SoundKind::PercussiveHarmonic => 1,
        SoundKind::Harmonic           => 2,
    }
}

pub struct MidiLayer {
    notes: VecDeque<NoteEvent>,
    live_spawn_ids: HashSet<usize>,
    /// Track lanes sorted by kind group (Perc → PH → Harm), then by first appearance.
    pub tracks: Vec<TrackInfo>,
    pub elapsed_secs: f32,
    /// Exponential moving average of frame dt — used to match history window to spectrogram speed.
    smooth_dt: f32,
}

impl MidiLayer {
    pub fn new() -> Self {
        Self {
            notes: VecDeque::new(),
            live_spawn_ids: HashSet::new(),
            tracks: Vec::new(),
            elapsed_secs: 0.0,
            smooth_dt: 1.0 / 43.0,  // assume ~43fps initially
        }
    }

    pub fn clear_all(&mut self) {
        self.notes.clear();
        self.live_spawn_ids.clear();
        self.tracks.clear();
    }

    /// History window in seconds — matches the spectrogram's time span (FFT_HISTORY frames).
    pub fn history_secs(&self) -> f32 {
        FFT_HISTORY as f32 * self.smooth_dt
    }

    pub fn update(&mut self, objects: &[SoundObject], dt: f32) {
        self.elapsed_secs += dt;
        self.smooth_dt = self.smooth_dt * 0.97 + dt * 0.03;
        let now = self.elapsed_secs;
        let history = self.history_secs();

        let current_ids: HashSet<usize> = objects.iter().map(|o| o.spawn_id).collect();
        let energies: HashMap<usize, f32> = objects.iter().map(|o| (o.spawn_id, o.energy)).collect();

        // Births
        for obj in objects {
            if self.live_spawn_ids.contains(&obj.spawn_id) {
                // Update hue / kind on existing track (e.g. Perc → PH upgrade)
                if let Some(t) = self.tracks.iter_mut().find(|t| t.cluster_id == obj.cluster_id) {
                    t.hue = obj.visual_hue;
                    if t.kind != obj.kind {
                        t.kind = obj.kind;
                        self.tracks.sort_by_key(|t| kind_group(t.kind));
                    }
                }
                continue;
            }

            // New spawn_id — register track if cluster not yet known
            if !self.tracks.iter().any(|t| t.cluster_id == obj.cluster_id) {
                if self.tracks.len() >= MAX_TRACKS {
                    self.tracks.remove(0);
                }
                self.tracks.push(TrackInfo {
                    cluster_id: obj.cluster_id,
                    kind: obj.kind,
                    hue: obj.visual_hue,
                });
                self.tracks.sort_by_key(|t| kind_group(t.kind));
            }

            // Mark any previous alive note for this cluster as finished.
            // Do NOT update end_secs — it already stopped at the energy-threshold cutoff.
            for prev in &mut self.notes {
                if prev.alive && prev.cluster_id == obj.cluster_id {
                    prev.alive = false;
                }
            }

            self.notes.push_back(NoteEvent {
                cluster_id: obj.cluster_id,
                spawn_id: obj.spawn_id,
                kind: obj.kind,
                hue: obj.visual_hue,
                start_secs: now,
                end_secs: now,
                alive: true,
            });
        }

        // Deaths
        for note in &mut self.notes {
            if note.alive && !current_ids.contains(&note.spawn_id) {
                note.alive = false;
            }
        }

        // Extend alive notes only while energy is above the kind-specific threshold.
        // Percussive threshold is high so tick width is proportional to hit strength.
        // Harmonic threshold is low so sustained notes extend for their full duration.
        for note in &mut self.notes {
            if note.alive {
                let e = energies.get(&note.spawn_id).copied().unwrap_or(0.0);
                let threshold = if note.kind == SoundKind::Percussive {
                    PERC_NOTE_THRESHOLD
                } else {
                    HARM_NOTE_THRESHOLD
                };
                if e > threshold {
                    note.end_secs = now;
                }
            }
        }

        // Prune notes that have scrolled fully off the left edge
        let cutoff = now - history;
        while self.notes.front().map_or(false, |n| n.end_secs < cutoff) {
            self.notes.pop_front();
        }

        self.live_spawn_ids = current_ids;
    }

    pub fn visible_notes(&self) -> impl Iterator<Item = DrawNote> + '_ {
        let cutoff = self.elapsed_secs - self.history_secs();
        self.notes.iter().filter_map(move |n| {
            if n.end_secs < cutoff { return None; }
            let track_idx = self.tracks.iter().position(|t| t.cluster_id == n.cluster_id)?;
            Some(DrawNote {
                track_idx,
                hue: n.hue,
                kind: n.kind,
                start_secs: n.start_secs,
                end_secs: n.end_secs,
                alive: n.alive,
            })
        })
    }

    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }
}
