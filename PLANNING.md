# Icarus Plan

## Goal
Build a small Nannou "primitive atlas" inside this repo so it becomes easy to learn, inspect, and compare the core drawing primitives and their builder-style APIs.

## Why This Exists
The current thread has accumulated a lot of implementation history. This file is the reset point for the next pass.

The immediate learning task is not audio/transcription quality. It is:

1. expose Nannou drawing primitives clearly
2. make them easy to run and inspect locally
3. keep the result simple enough to extend later

## Scope
The new work should focus on a dedicated visual reference app or mode that demonstrates the main `Draw` primitives in one place.

Initial primitives to cover:

- background
- rect
- ellipse
- line
- arrow
- tri
- quad
- polygon
- polyline
- path
- mesh
- text
- texture

The atlas should also demonstrate the shared modifiers that matter most in practice:

- `x_y` / `xy`
- `w_h` / size controls
- `color` / `rgb` / `rgba` / `hsl` / `hsla`
- stroke vs fill
- rotation
- scale

## Out Of Scope
Not part of this task unless explicitly added later:

- improving the transcriber
- evaluation-score work
- laser output work
- redesigning the main ICARUS visualization
- exhaustive documentation of every Nannou type

## Current Repo State
Current main concepts in the repo:

- realtime audio capture
- analysis and transcription
- note-history visualization
- evaluation harness

Current worktree is not clean. Before deeper work, account for these in-flight changes:

- `README.md` modified
- `src/lib.rs` modified
- `src/main.rs` modified
- `src/transcription.rs` modified
- `src/objects.rs` deleted

So the next implementation pass should either:

1. finish and commit the current simplification/fix work first, or
2. carefully isolate the primitive-atlas changes from those unrelated edits

## Deliverable
Produce a runnable primitive-atlas view that makes it easy to answer:

- what primitive to use
- what it looks like
- what builder calls shape it
- how it differs from nearby primitives

## Acceptance Criteria
The task is done when all of the following are true:

1. there is a clearly named entry point for the atlas
2. the atlas renders the core primitives listed above
3. each primitive is labeled in the UI
4. the layout is easy to scan visually
5. `cargo check` passes
6. the code is simple enough to use as a reference later

## Recommended Approach
### Phase 1
Decide the surface:

- separate binary in `src/bin/`
- or alternate mode in the existing app

Prefer a separate binary unless there is a strong reason not to.

### Phase 2
Build a grid of primitive demos:

- one tile per primitive
- short label
- minimal, intentional styling
- no audio dependency

### Phase 3
Add a few comparison cues:

- fill vs stroke
- closed vs open paths
- transformed vs untransformed example where helpful

### Phase 4
Document how to run it in `README.md` if the result is worth keeping long-term.

## Constraints
- Keep it local and practical.
- Prefer direct code examples over abstract explanation.
- Do not bury the primitives inside app-specific audio state.
- Avoid turning the atlas into a framework or editor.

## Next Step
From fresh context, implement the primitive atlas as a dedicated runnable target and keep the first version small, visual, and reference-oriented.
