# ICARUS

ICARUS is a realtime audio-analysis and visualization app. It captures system audio, analyzes it online, transcribes it into note-like events, and renders those events over spectral views in a Nannou window.

The repo also contains an evaluation harness that compares the online transcription output against paired `.wav` / `.mid` reference material in [`midi`](./midi).

## What The App Is Doing

At a high level, the runtime pipeline is:

1. Capture system audio.
2. Compute causal spectral and feature analysis.
3. Turn that analysis into higher-level musical events.
4. Buffer those events into visualization-friendly state.
5. Render spectral history, note overlays, indicators, and menu-driven layer groups.

The important point is that this is an online system. The transcriber and visualizer are built around incremental, hop-by-hop updates rather than offline whole-file processing.

## Core Concepts

### Analysis

[`src/analysis.rs`](./src/analysis.rs) is the feature front end.

It is responsible for deriving the low-level evidence the rest of the app uses, such as:

- spectral history
- CQT-like pitch-space energy
- onset and percussive indicators
- harmonic / percussive confidence signals
- support signals used by note tracking

Think of this layer as "what does the audio look like right now?"

### Sound Objects

[`src/objects.rs`](./src/objects.rs) contains the sound-object detector.

This is the older or parallel abstraction for turning analysis into object-like events such as percussive, harmonic, or hybrid sound entities. It is still an important part of the repo and still has tests, but the main visualization path is now driven by the transcription system rather than by rendering sound objects directly.

Think of this layer as "what kinds of sounding things seem to exist right now?"

### Transcriber

[`src/transcription.rs`](./src/transcription.rs) is the current note-oriented interpretation layer.

Its job is to take the live analysis stream and emit note-like events with approximate:

- pitch
- start
- end
- confidence

It also tracks note support over time. For active notes, the transcriber exposes a current note-support energy value. For completed notes, it can retain a short hop-rate energy envelope over the note's lifetime. This is not separated audio stem RMS; it is the note-support signal derived from the same analysis evidence that visually sits under the note in the spectral display.

Think of this layer as "what notes or drum-like hits do we believe are happening?"

### Notes

In this repo, a "note" is a transcribed event, not a MIDI file primitive and not raw audio.

A note is conceptually:

- a pitch or drum hit identity
- a time span
- a confidence
- an instrument selection label used by the UI
- optional note-support energy over time

There are two useful note states:

- active notes: currently sounding according to the transcriber
- completed notes: notes that have ended but remain in history for rendering, evaluation, and diagnostics

### Instrument Selection

Instrument selection is a UI-facing grouping label attached to transcribed notes.

Current selections are:

1. `Percussive`
2. `Harmonic`
3. `PercussiveHarmonic`

Right now these labels are close to the existing `SoundKind` categories, but that is a current implementation choice, not a permanent rule. The name "instrument selection" is meant to stay broader than the underlying detection taxonomy.

Think of this as "which family of notes should this event belong to for display and control purposes?"

### Layer Groups

The visualizer menu is built around layer groups rather than individual draw calls.

A layer group:

- has a label
- has an active / inactive state
- owns one or more visual layers
- is drawn in menu order
- can be toggled as a unit

Today, each instrument selection currently maps to a simple group that mainly controls note overlays, but the structure is intentionally dynamic so that a group can later contain more than one visual layer.

Think of this as "the user-visible display groups that can be shown, hidden, and later rearranged."

### MidiLayer

[`src/midi_layer.rs`](./src/midi_layer.rs) is not a MIDI parser or MIDI output layer in the conventional sense.

It is a visualization/state bridge that:

- accepts active and completed transcribed notes
- keeps a rolling history
- filters notes by instrument selection
- computes group-level activity such as RMS-like brightness drivers for the UI
- exposes draw-ready note data to the Nannou app

Think of this as "presentation state for note history."

### Visualizer

[`src/main.rs`](./src/main.rs) is the live app.

It:

- captures audio
- feeds the streaming transcriber
- updates note history
- draws the spectral background and note overlays
- manages the top menu and layer-group toggles

The menu currently acts as an instrument-selection switchboard. Groups are active by default, can be toggled from the keyboard, and their indicators are driven by group-level note-support energy.

## Evaluation

The evaluation harness lives in [`src/transcription.rs`](./src/transcription.rs) and is exposed through [`src/bin/eval_transcription.rs`](./src/bin/eval_transcription.rs).

It exists to answer two questions:

1. How close is the online transcriber to a reference MIDI?
2. In what ways is it failing?

The suite operates on paired files in [`midi`](./midi), including standardized mixed-audio tests built from those pairs. It uses fuzzy matching rather than strict exact equality, so near-miss timing and pitch errors get partial credit instead of only pass/fail treatment.

Useful commands:

```bash
cargo run
cargo run --bin eval_transcription --
cargo run --bin eval_transcription -- --json out/report.json
cargo test
```

## Repo Map

Top-level pieces you will usually care about:

- [`src/analysis.rs`](./src/analysis.rs): spectral / feature analysis
- [`src/audio.rs`](./src/audio.rs): realtime audio capture
- [`src/objects.rs`](./src/objects.rs): sound-object detector
- [`src/transcription.rs`](./src/transcription.rs): online transcription and evaluation
- [`src/midi_layer.rs`](./src/midi_layer.rs): note-history and visualization bridge
- [`src/main.rs`](./src/main.rs): Nannou visualizer app
- [`src/bin/eval_transcription.rs`](./src/bin/eval_transcription.rs): evaluation CLI
- [`midi`](./midi): paired `.wav` / `.mid` fixtures used by the evaluator
- [`PLANNING.md`](./PLANNING.md): working notes / plans

## What Else To Expect Here

This repo is not a neatly isolated library with one perfectly finished pipeline. Expect a few shenanigans:

- overlapping abstractions: sound objects and note transcription coexist
- realtime and evaluation concerns living close together in the same modules
- visualizer-specific state that exists mainly to make live rendering easier
- experimental heuristics around onset handling, pitch tracking, note birth, and note continuation
- a dataset-backed evaluation workflow that is useful for diagnosis but should not drive benchmark-specific hacks

In other words, this is an actively-shaped realtime music-analysis project. The stable concepts are the pipeline stages and the roles of the modules above; the exact heuristics and thresholds inside them are expected to evolve.
