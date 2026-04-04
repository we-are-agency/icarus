use crate::analysis::CQT_HISTORY;
use crate::transcription::{
    ActiveTranscribedNote, CompletedTranscribedNote, InstrumentSelection, TRANSCRIPTION_HOP_SIZE,
    TRANSCRIPTION_MIDI_HIGH, TRANSCRIPTION_MIDI_LOW,
};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
struct NoteEvent {
    id: usize,
    stream_id: usize,
    midi_note: u8,
    start_secs: f32,
    end_secs: f32,
    confidence: f32,
    alive: bool,
    instrument_selection: InstrumentSelection,
}

pub struct DrawNote {
    pub id: usize,
    pub midi_note: u8,
    pub hue: f32,
    pub start_secs: f32,
    pub end_secs: f32,
    pub confidence: f32,
    pub alive: bool,
    pub instrument_selection: InstrumentSelection,
}

pub struct MidiLayer {
    finished_notes: VecDeque<NoteEvent>,
    active_notes: HashMap<usize, NoteEvent>,
    elapsed_secs: f32,
    history_secs: f32,
}

impl MidiLayer {
    pub fn new(sample_rate: u32) -> Self {
        let hop_secs = TRANSCRIPTION_HOP_SIZE as f32 / sample_rate.max(1) as f32;
        Self {
            finished_notes: VecDeque::new(),
            active_notes: HashMap::new(),
            elapsed_secs: 0.0,
            history_secs: CQT_HISTORY as f32 * hop_secs,
        }
    }

    pub fn history_secs(&self) -> f32 {
        self.history_secs
    }

    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed_secs
    }

    pub fn note_min(&self) -> u8 {
        TRANSCRIPTION_MIDI_LOW
    }

    pub fn note_max(&self) -> u8 {
        TRANSCRIPTION_MIDI_HIGH
    }

    pub fn update(
        &mut self,
        finished: &[CompletedTranscribedNote],
        active: &[ActiveTranscribedNote],
        elapsed_secs: f32,
    ) {
        self.elapsed_secs = elapsed_secs;

        for completed in finished {
            self.active_notes.remove(&completed.id);
            self.finished_notes
                .push_back(NoteEvent::from_completed(completed));
        }

        let live_ids: HashSet<usize> = active.iter().map(|note| note.id).collect();
        self.active_notes.retain(|id, _| live_ids.contains(id));

        for active_note in active {
            self.active_notes
                .insert(active_note.id, NoteEvent::from_active(active_note));
        }

        self.prune_history();
    }

    pub fn visible_notes(&self) -> Vec<DrawNote> {
        self.visible_notes_for_selection(None)
    }

    pub fn visible_notes_for_selection(
        &self,
        selection: Option<InstrumentSelection>,
    ) -> Vec<DrawNote> {
        let cutoff = self.elapsed_secs - self.history_secs;
        let mut notes =
            Vec::with_capacity(self.finished_notes.len().saturating_add(self.active_notes.len()));

        for note in &self.finished_notes {
            if note.end_secs < cutoff
                || selection.is_some_and(|instrument| note.instrument_selection != instrument)
            {
                continue;
            }
            notes.push(note.as_draw_note());
        }

        for note in self.active_notes.values() {
            if note.end_secs < cutoff
                || selection.is_some_and(|instrument| note.instrument_selection != instrument)
            {
                continue;
            }
            notes.push(note.as_draw_note());
        }

        notes.sort_by(|a, b| {
            a.start_secs
                .partial_cmp(&b.start_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        notes
    }

    pub fn group_rms(&self, selection: InstrumentSelection) -> f32 {
        let mut sum_squares = 0.0f32;
        let mut count = 0usize;

        for note in &self.finished_notes {
            if note.instrument_selection == selection
                && note.start_secs <= self.elapsed_secs
                && note.end_secs >= self.elapsed_secs
            {
                sum_squares += note.confidence * note.confidence;
                count += 1;
            }
        }

        for note in self.active_notes.values() {
            if note.instrument_selection == selection
                && note.start_secs <= self.elapsed_secs
                && note.end_secs >= self.elapsed_secs
            {
                sum_squares += note.confidence * note.confidence;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            (sum_squares / count as f32).sqrt().clamp(0.0, 1.0)
        }
    }

    fn prune_history(&mut self) {
        let cutoff = self.elapsed_secs - self.history_secs;
        while self
            .finished_notes
            .front()
            .is_some_and(|note| note.end_secs < cutoff)
        {
            self.finished_notes.pop_front();
        }
        self.active_notes
            .retain(|_, note| note.end_secs >= cutoff || note.alive);
    }
}

impl NoteEvent {
    fn from_completed(note: &CompletedTranscribedNote) -> Self {
        Self {
            id: note.id,
            stream_id: note.note.stream_id,
            midi_note: note.note.midi_note,
            start_secs: note.note.start_secs,
            end_secs: note.note.end_secs,
            confidence: note.note.confidence,
            alive: false,
            instrument_selection: note.note.instrument_selection,
        }
    }

    fn from_active(note: &ActiveTranscribedNote) -> Self {
        Self {
            id: note.id,
            stream_id: note.stream_id,
            midi_note: note.midi_note,
            start_secs: note.start_secs,
            end_secs: note.end_secs,
            confidence: note.confidence,
            alive: true,
            instrument_selection: note.instrument_selection,
        }
    }

    fn as_draw_note(&self) -> DrawNote {
        DrawNote {
            id: self.id,
            midi_note: self.midi_note,
            hue: stream_hue(self.stream_id),
            start_secs: self.start_secs,
            end_secs: self.end_secs,
            confidence: self.confidence.clamp(0.0, 1.0),
            alive: self.alive,
            instrument_selection: self.instrument_selection,
        }
    }
}

fn stream_hue(stream_id: usize) -> f32 {
    ((stream_id as f32 * 0.173_205_08) + 0.08).fract()
}
