# Linkage Implementation Plan

This document is the initial implementation sequence for the system described in [LINKAGE.md](./LINKAGE.md). The goal is to reach a working vertical slice quickly, prove the main technical risks in order, and avoid introducing LLM/concurrency complexity before the rendering and solver path is stable.

---

## Principles

- Start with the render loop, then add mechanism data, then playback, then LLM generation.
- Keep the left side of the UI read-only at first. It is an inspection panel, not an editor.
- Render the exact `serde_json::to_string_pretty(&assembly)` output on the left so the UI shows the same assembly shape the LLM and validator operate on later.
- Prefer vertical slices over broad scaffolding.
- Keep the right side artifact-oriented as early as possible so later `Committed Sweep` integration does not require a rewrite.
- Defer the async Design Loop, staged ghosts, and live telemetry-driven iteration until one-shot generation is stable.

---

## Design Guide

This section captures the visual language already present in the `icarus` binary so the linkage UI grows in the same direction instead of introducing a second, unrelated aesthetic.

### Overall Tone

- Default to a black or near-black field.
- Use low-chroma structural surfaces and reserve saturated color for live signal, motion, or emphasis.
- Avoid decorative borders, shadows, gradients, or card-heavy framing.
- Favor overlay and scaffold aesthetics over app-chrome aesthetics.

### Typography

- No custom font is currently loaded in the binary; text uses the default Nannou text path.
- Primary UI text is small and restrained:
  - most labels use `font_size(9)`
  - secondary low/high helper labels use `font_size(8)`
- Text is almost always white with alpha rather than fully opaque color:
  - strong label: about `rgba(1, 1, 1, 0.82)` to `0.98`
  - normal label: about `0.68` to `0.70`
  - secondary label: about `0.40` to `0.45`
  - tertiary/helper label: about `0.35`
- Use concise uppercase labels for structural UI where possible (`ICARUS`, `KICK`, `SNARE`, band labels).

### Color

- Base background is true black.
- Structural dark fills in the current UI cluster around:
  - top bar: `rgba(0.02, 0.025, 0.035, 0.96)`
  - muted dark key lanes: `rgba(0.02, 0.03, 0.04, 0.22)`
- White is used as the neutral ink color, mostly with alpha.
- Accent colors are vivid HSL/HSLA fills used for live analytical data, not static chrome:
  - hue-driven bars
  - note blocks
  - beat indicators
- For linkage rendering, this suggests:
  - grid / guides / secondary scaffold in translucent white
  - committed mechanism in restrained neutral or one accent family
  - POI traces and active drive indicators in the stronger accent colors
  - staged ghosts in low-alpha neutral or desaturated accent, never louder than committed geometry

### Lines, Borders, and Geometry

- The binary currently uses thin line weights:
  - regular grid/separator lines: `1.0`
  - emphasized horizontal pitch markers: `1.25`
  - major separators: `1.5`
  - selected-state underline: `2.0`
- Borders are generally implied by lines or contrast, not boxed outlines.
- Rectangles and ellipses are simple fills; there is no ornamental stroke treatment.
- Keep linkage drawing crisp and geometric:
  - grid lines around `1.0`
  - standard linkage strokes around `1.0` to `1.5`
  - selected or active emphasis around `2.0`

### Layout Habits

- The existing app uses full-window composition with layered overlays rather than nested panels.
- The top menu bar is a shallow strip (`56px` high) rather than a heavy header.
- Information density is high, but hierarchy is controlled mostly by alpha and position, not size jumps.
- For the linkage split view:
  - keep the left spec pane visually quiet
  - keep the right render area dominant
  - prefer one or two stable dividers over boxed sub-panels everywhere

### Recommended Mapping For The Linkage UI

- Background: black / near-black.
- Spec pane text: white at `0.40` to `0.70` alpha, small size, monospaced only if readability truly benefits.
- Grid: white at about `0.06` to `0.20` alpha.
- Links and joints: neutral light tone first, with stronger color reserved for selection and motion.
- POI traces: dotted, brighter than the grid, color-coded per point.
- HUD labels: `font_size` around `8` to `9`, mostly white alpha text.
- Error state: brighter accent, but still integrated into the dark palette rather than a warning-card treatment.

### What To Avoid

- Large type jumps
- thick borders around every region
- bright permanent accent colors on static scaffolding
- rounded-card dashboard styling
- gradients or soft shadow treatment that compete with the plotted mechanism

Reference cues in the current binary come primarily from [main.rs](/Users/Testsson/Projects/weareagency-icarus/src/main.rs#L275), [main.rs](/Users/Testsson/Projects/weareagency-icarus/src/main.rs#L396), [main.rs](/Users/Testsson/Projects/weareagency-icarus/src/main.rs#L468), [main.rs](/Users/Testsson/Projects/weareagency-icarus/src/main.rs#L500), and [main.rs](/Users/Testsson/Projects/weareagency-icarus/src/main.rs#L535).

---

## P0 — App Boot + Hello World Render

### Goal

Prove that the binary target starts and the renderer can run in both windowed and headless form.

### Scope

- Choose the initial binary location in this repo: start with `src/bin/linkage.rs`.
- Binary builds and launches.
- Windowed mode opens.
- Headless mode can render a frame without window/input wiring.
- Basic background / text / grid renders.
- Frame loop ticks continuously.

### Exit Criterion

Running the app shows a stable "hello world" style render in windowed mode, and the same renderer can execute a headless frame path from the same binary target.

---

## P1 — Hard-Coded Assembly + Static Solved View

### Goal

Prove the first full data-to-render path using a hard-coded fixture assembly.

### UI

- Left half: hard-coded linkage assembly displayed as exact `serde_json::to_string_pretty(&assembly)` output.
- Right half: linkage drawn in a static solved origin-state.

### Scope

- Define the first assembly fixture in code or serde-backed data.
- Parse/load the fixture.
- Validate it.
- Use a closed-form first fixture topology (slider-crank or pure four-bar) so the first solved pose does not depend on the full general NR path.
- Solve a single static pose.
- Draw joints, links, and sliders.
- Keep the right side static; no playback yet.

### Exit Criterion

The app renders a valid hard-coded linkage from structured input, with the serialized spec visible alongside it.

---

## P1.5 — Validation Failures + Multiple Fixtures

### Goal

Make failure modes visible before LLM-generated input enters the system.

### Scope

- Add at least two or three fixture assemblies.
- Add at least one intentionally invalid validator fixture (bad IDs, invalid patch target, malformed references, or equivalent).
- Add at least one intentionally unsolved / diverging solver fixture.
- Show validator failure clearly in the UI instead of crashing.
- Show solver failure clearly in the UI instead of crashing.
- Keep rendering responsive even when a fixture cannot solve.

### Exit Criterion

The app can switch between valid fixtures, validator-invalid fixtures, and solver-invalid fixtures, rendering the valid ones and surfacing distinct errors for the invalid ones.

---

## P2 — Playback From a Single Drive

### Goal

Prove the basic sweep/playback model with one hard-coded driven assembly.

### UI

- Left half: same read-only assembly JSON.
- Right half: linkage animated with a ping-pong drive.

### Scope

- Add one active drive to a fixture assembly.
- Introduce `SweepArtifact` + `SolvedFrame` in the implementation shape described by `LINKAGE.md` §8.4.
- Produce solved frames for one sweep interval and store them in an `Arc<SweepArtifact>`.
- Have the renderer read from `Arc<SweepArtifact>` even though publication is still synchronous and in-process at this milestone.
- Implement ping-pong traversal.
- Add POIs and dotted POI trace rendering now that sweep playback exists.
- Add a determinism check that running the same fixture sweep twice produces byte-identical frame data.

### Exit Criterion

The right side animates a driven linkage smoothly from `Arc<SweepArtifact>`-backed solved frames, dotted POI traces render correctly, and running the same fixture sweep twice produces byte-identical frame output.

---

## P3a — One-Shot LLM Tool-Use On Startup

### Goal

Introduce the first LLM path without yet introducing the full async iterative design loop or rejection-feedback retry behavior.

### UI

- Left half: assembly JSON derived from the validated current assembly.
- Right half: rendered and animated as in P2 when validation and solving succeed.

### Scope

- On startup, send one fixed prompt to the LLM.
- Use the actual `propose_mutations` tool-use contract from `LINKAGE.md` §6.1 from day one; do not introduce an ad hoc JSON-only temporary shape.
- Verify the provider tool-use API shape against current docs before wiring it up.
- Receive one `propose_mutations` result.
- Validate it.
- If valid, solve and render it.
- If invalid, display the failure clearly and keep the app alive.
- Do not yet add staged streaming, ghost previews, or iterative feedback turns.

### Exit Criterion

A one-shot startup prompt can produce a `propose_mutations` tool response that validates, solves, and renders without manual intervention.

---

## P3b — Rejection Feedback Retries

### Goal

Exercise the `rejected` feedback channel before concurrency lands.

### Scope

- When validation rejects a mutation batch, feed the structured rejection payload back into the next LLM turn using the `LINKAGE.md` §6.3 shape.
- Retry within the same startup session up to a small fixed attempt count.
- Keep the renderer alive and the latest valid artifact visible throughout retry attempts.

### Exit Criterion

The app can recover from at least one initial rejected mutation batch and reach a valid rendered assembly via the structured rejection-feedback loop. This is the earliest milestone that satisfies the spec's current Phase 1 rejection-correction requirement.

---

## P4a — Async Design Loop + Committed Sweep Only

### Goal

Move from one-shot generation to the actual two-loop architecture, but only for `committed_sweep`.

### Scope

- Introduce `SharedState` with `committed_sweep`.
- Publish committed sweep artifacts from a background Design Loop task.
- Render from `committed_sweep` on the Render Loop via the same `Arc<SweepArtifact>` read path already proven in P2.
- Confirm that artifact swaps do not stall render.

### Exit Criterion

The app continuously renders while the Design Loop generates and commits new sweep artifacts in the background, using the same artifact read path already established earlier.

---

## P4b — Staging, HUD Activity, and Swap Semantics

### Goal

Add the auxiliary shared-state channels and the playback behavior around swaps.

### Scope

- Add `staged_assembly`.
- Add `activity_state`.
- Add staged previews / ghosts.
- Add HUD activity state.
- Preserve playback phase across artifact swaps.
- Implement `reset_phase` handling.
- Defer undo/history until after this step.

### Exit Criterion

Committed playback, staged previews, HUD activity, and phase-preserving artifact swaps all work together without blocking the Render Loop.

---

## P5 — Undo / History

### Goal

Add the first local history mechanism after the cross-thread rendering path is stable.

### Scope

- Introduce snapshot history for validated assemblies.
- Re-sweep and republish when undoing.
- Keep the current playback artifact visible until the prior state is re-committed.

### Exit Criterion

Undo returns to a prior assembly state and republishes a valid sweep artifact without freezing the renderer.

---

## Out of Scope For This Initial Plan

- Editable left-side spec panel
- Full telemetry-driven iterative refinement in the first milestone set
- Variant management
- Parametric exploration
- Quantitative scoring / tolerance evaluation (the telemetry loop remains qualitative via `DerivedObservations` in `LINKAGE.md` §8.3 and the future-work note in `LINKAGE.md` §15)
- Export pipeline
- Collaboration / persistence

---

## Critical Path Notes

- P0 fixes the binary location and renderer I/O boundary, including headless mode.
- P1 keeps the first solver path cheap by using a closed-form fixture.
- P2 is the key architectural seam: the renderer must already consume `Arc<SweepArtifact>` here so concurrency later is additive.
- P3a proves provider integration and tool-use shape.
- P3b proves the rejection-feedback contract from the spec.
- P4a proves cross-thread committed artifact publication on its own.
- P4b adds the auxiliary polish layers after the committed-sweep path is stable.
