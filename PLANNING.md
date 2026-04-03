# Icarus — Laser + Projector Projection Software

## Goal
Real-time audio-reactive visuals rendered to both a Nannou window (projector) and a laser DAC (Ether Dream via `nannou_laser`).

## Current State
Audio capture pipeline is wired up but **Background Music virtual device returns silence** (all zeros). Need to validate the implementation and identify the bug.

## Stack
- `nannou 0.19` — windowed rendering
- `nannou_laser 0.19` — Ether Dream DAC (not yet wired)
- `cpal 0.17` — audio capture
- `realfft 3.5` — FFT (Hann window → magnitude → dBFS)
- `ringbuf 0.2` — lock-free SPSC ring buffer between cpal callback thread and nannou render thread
- `dasp 0.11`, `fundsp 0.19`, `spectrum-analyzer 1` — future use

## Key Files
- `/Users/Testsson/Projects/weareagency-icarus/src/main.rs` — Nannou app, FFT update loop, spectrum visualiser view
- `/Users/Testsson/Projects/weareagency-icarus/src/audio.rs` — cpal device selection, stream setup, ring buffer

## Known Issues
1. `Background Music` virtual device streams F32/48kHz but returns all-zero samples even when audio is playing
2. No tests yet
3. Debug prints in production code (need removal once working)

## Architecture
```
cpal input stream (audio thread)
  └─► ringbuf Producer<f32>
        └─► nannou update() (main thread)
              ├─► VecDeque<f32> rolling window (FFT_SIZE=2048)
              ├─► Hann window → realfft → dBFS spectrum
              └─► view() draws spectrum bars
```

## Next Steps
1. Fix audio capture (BlackHole or mic fallback)
2. Wire nannou_laser frame stream
3. Build Scene abstraction (shared geometry for projector + laser)
4. Audio-reactive parameter mapping
