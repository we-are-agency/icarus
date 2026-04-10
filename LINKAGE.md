# Linkage: LLM-Driven Linkage Simulation Playground & Visualizer

## Product Requirements Document

> File is `LINKAGE.md` because the MVP is all about planar linkages — joints, links, sliders. Within this repo, the project name is simply "Linkage".

---

## 1. Vision

Linkage is an explorative simulation tool in which an LLM acts as co-designer. The user states intent in natural language ("a straight-line linkage", "a compact crank-slider", "something strange that jitters and blooms into spirals"), and the system iteratively constructs, simulates, sweeps, and refines planar assemblies rendered in real time via Nannou. The LLM does not generate once; it enters a **critique → mutate → sweep → observe → refine** loop, consuming telemetry and proposing incremental mutations.

The aim is not to certify mechanisms. The aim is to produce compelling motion, interesting traces, and responsive visual experimentation. A branch switch, near-lockup, wobble, or snap can be a valid aesthetic result rather than an automatic failure.

---

## 2. What Linkage Is *Not*

Scoping up front so neither the user, the LLM, nor future-me wander:

- **Not a CAD tool.** No dimensioning, drafting, tolerancing, STEP export.
- **Not an engineering-grade physics simulator.** MVP uses a Verlet-based constraint integrator to produce compelling constrained motion and traces, but it is not aiming at physically accurate forces, impacts, or material models yet.
- **Not a manufacturing tool.** No DFM, BOMs, fabrication output.
- **Not a proof-oriented kinematics package.** The engine may settle into one branch of an ambiguous assembly, hop branches, or fail to settle cleanly, and those behaviors can still be useful visual outcomes.
- **Not a general rigid-body solver.** The engine is a planar Verlet + constraint-projection system tuned for expressive 2D assemblies first, while leaving room for ropes, textiles, and soft bodies later.
- **No gears, cams, springs, belts in MVP.** See §15 Future Development.

---

## 3. Core Architecture

Linkage is built as **two concurrent loops** that share one primary lock-free handoff buffer plus two auxiliary lock-free slots. The Render Loop is always drawing — the screen never freezes while the LLM thinks, while mutations validate, or while a sweep is being relaxed. The Design Loop, running asynchronously, publishes new swept artifacts when ready; the Render Loop picks them up on the next frame.

```
                      User (NL intent)
                             │
                             ▼
    ┌────────────────────────┬────────────────────────┐
    │  DESIGN LOOP  —  tokio async task, one turn     │
    │                                                 │
    │  Orchestrator → LLM Client → Validator → Sweep  │
    │                                                 │
    │  on success: publish new SweepArtifact          │
    │  on error:   rollback + emit <rejected>         │
    └────────────────────────┬────────────────────────┘
                             │  store
                             ▼
                ┌────────────┬───────────┐
                │    COMMITTED SWEEP     │
                │ ArcSwap<SweepArtifact> │
                │  frames + telemetry    │
                └────────────┬───────────┘
                             │  load (lock-free, per frame)
                             ▼
    ┌────────────────────────┬────────────────────────┐
    │  RENDER LOOP  —  nannou, 60 Hz, own thread      │
    │                                                 │
    │  • Plays back the latest Committed Sweep        │
    │  • Phase u cycles and is preserved across swaps │
    │  • Ghost overlays for staged mutations          │
    │  • Interaction always active (pan/zoom/POI)     │
    │  • HUD reflects Design Loop ActivityState       │
    └─────────────────────────────────────────────────┘
```

Key properties:

- **The Render Loop never awaits the Design Loop.** Even if the LLM takes 30 s to answer, the screen keeps animating the last committed sweep.
- **The shared state has one primary handoff plus two auxiliary slots.** `committed_sweep: ArcSwap<SweepArtifact>` is the main render handoff; `staged_assembly` and `activity_state` are auxiliary lock-free preview/HUD slots.
- **Telemetry is the LLM's sensor.** Each committed artifact carries the telemetry that feeds the next prompt turn.
- **The renderer can run interactive or headless.** The same Committed Sweep playback path is usable for on-screen inspection, CI snapshot tests, offline frame export, and other non-interactive tooling.
- **The simulation core is Verlet-based.** Every committed frame comes from the same particle-and-constraint path that can later be extended to ropes, textiles, and soft bodies.
- **Repeatability matters more than uniqueness.** The runtime should replay the same input the same way on the same platform, but it does not attempt to prove a single globally-correct pose for ambiguous assemblies.

---

## 4. Data Model: Assembly Graph

All mechanical state lives in one structure, round-trippable to JSON.

### 4.1 Identifiers

```rust
type JointId = String;   // stable, LLM-authored, e.g. "j_crank_pivot"
type PartId  = String;   // stable, LLM-authored, e.g. "l_coupler"
type DriveId = String;
```

Stable string IDs (not UUIDs) so the LLM can reference prior parts across turns without the orchestrator rewriting its context.

All spec-visible maps below use ordered maps (`BTreeMap`) rather than `HashMap`, so prompt injection, serialization, telemetry, snapshot tests, and published frame data all have stable ordering.

### 4.2 Joints

A joint is a 2D point. Every joint is either **Fixed** (world-space coordinate provided by the user/LLM) or **Free** (position is produced by the Verlet simulation pass).

```rust
struct Joint {
    id: JointId,
    kind: JointKind,
}

enum JointKind {
    Fixed { position: Vec2 },  // grounded anchor, held constant by projection
    Free,                      // particle integrated then projected by constraints
}
```

A link between two joints implies a distance constraint. A slider implies a line constraint on one of its joints. At runtime, every `Free` joint is represented as a particle with current and previous positions; the simulation step advances particles with Verlet integration and then projects constraints until the assembly becomes drawable enough for the current mode or the iteration budget is exhausted.

### 4.3 Parts

```rust
enum Part {
    Link {
        id: PartId,
        a: JointId,
        b: JointId,
        length: f32,          // canonical; joint positions must match this after relaxation
    },
    Slider {
        id: PartId,
        joint: JointId,       // the joint constrained to lie on this track
        axis_origin: Vec2,    // world-space track origin
        axis_dir: Vec2,       // unit direction; normalized on insertion
        range: (f32, f32),    // allowed travel along axis, in world units
    },
}
```

- `Link.length` is the rest length the simulation tries to honor. If the two joints are already placed, the orchestrator may warn or reject depending on validation mode, but exact agreement is not the primary product goal.
- `Slider` is a constraint and a part at once: it restricts one joint to a track and is visualized as a track.
- There is no separate `Frame` type. Ground is represented by one or more `Fixed` joints.

### 4.4 Drives

Drives are the system's *inputs*. Each drive has a **range**; the simulator **sweeps** or plays through that range to produce telemetry. Drives do not carry real time directly — they carry a target parameter `u ∈ [range.0, range.1]` that the simulation engine uses to steer motion across a sampled interval.

```rust
struct Drive {
    id: DriveId,
    kind: DriveKind,
    sweep: SweepSpec,
}

enum DriveKind {
    /// Rotates a link around one of its joints. The angle is measured between
    /// (pivot → tip) and the +x axis. `tip_joint` must be Free; `pivot_joint`
    /// is typically Fixed. The drive contributes an angular target constraint;
    /// the relaxation pass projects the crank toward that angle, then settles
    /// the remaining constraints.
    Angular {
        pivot_joint: JointId,
        tip_joint:   JointId,
        link:        PartId,        // the link whose angle is being driven
        range:       (Angle, Angle),// radians, inclusive
    },

    /// Translates a joint along a slider's axis. `joint` must be the joint
    /// referenced by `slider`. The drive contributes a target position along
    /// the track; the relaxation pass settles the rest of the assembly around it.
    Linear {
        slider: PartId,
        joint:  JointId,
        range:  (f32, f32),         // must lie within slider.range
    },
}

struct SweepSpec {
    samples:   u32,       // default 180
    direction: SweepDir,  // Forward | Reverse | PingPong playback traversal over one normalized interval
}
```

**Why both drive types:** angular drives correspond to cranks and rocker inputs; linear drives correspond to directly-pushed sliders. Every planar one-DoF linkage can be driven by one of these. The schema keeps `drives` as a map for forward compatibility, but MVP sweep/commit supports exactly **one active drive**. Assemblies may be edited with zero drives attached, but validation rejects any sweep/commit attempt that has zero or multiple drives via `DRIVE_COUNT_INVALID`. True multi-drive / multi-DoF committed playback is deferred to Future Development.

### 4.5 Assembly

```rust
struct Assembly {
    joints: BTreeMap<JointId, Joint>,
    parts:  BTreeMap<PartId, Part>,
    drives: BTreeMap<DriveId, Drive>,
    points_of_interest: Vec<PointOfInterest>,  // points whose path the LLM should trace
    visualization: Option<VisualizationSpec>,  // optional trace-space transforms / rendering intent
    meta:   AssemblyMeta,
}

struct PointOfInterest {
    id:    String,
    host:  PartId,   // must reference a Link; POIs are not attachable to Sliders in MVP
    t:     f32,      // 0.0 at joint a, 1.0 at joint b; can be outside [0,1] for extensions
    perp:  f32,      // perpendicular offset from the link, 0.0 for on-axis
}

struct VisualizationSpec {
    trace_model: Option<TraceModel>,
}

enum TraceModel {
    /// Render selected POI paths as if the linkage is drawing on paper that
    /// continuously rolls in one direction while the sweep advances.
    RollingPaper {
        direction: PaperDirection,   // Up | Down | Left | Right in screen/world trace space
        advance_per_cycle: f32,      // paper travel over one committed playback cycle
    },
}

enum PaperDirection { Up, Down, Left, Right }

struct AssemblyMeta {
    name:      String,
    iteration: u32,
    notes:     Vec<String>,
}
```

`PointOfInterest` is crucial: most interesting goals ("trace a straight line", "describe a figure-8", "leave a trembling ribbon on rolling paper") are properties of a specific point on a coupler, not a joint. POIs are defined only on `Link` parts in MVP; validation rejects any other host with `POI_HOST_INVALID`. The LLM names the point and the sweep records its trajectory from the simulated particle positions at each sample.

`VisualizationSpec` is intentionally separate from the mechanism itself. It does not alter the constraint set or the relaxation pass; it alters how traced paths are interpreted and drawn. The first non-mechanical trace model is `RollingPaper`, which treats all POI paths as if the mechanism is drawing onto paper that translates steadily during the sweep. This is useful for spirograph / plotter / rolling-paper compositions without inventing extra linkage parts just to represent the paper feed.

---

## 5. Serialization Contract

One consistent discriminator style across the schema: domain variants (`JointKind`, `Part`, `DriveKind`, `TraceModel`) are **internally tagged** via a `"type"` field, while mutation variants are tagged by `"op"`. This keeps both the persisted assembly and the tool-call payloads hand-authorable and LLM-friendly.

```json
{
  "joints": {
    "j_pivot":  { "type": "Fixed", "position": [0.0, 0.0] },
    "j_tip":    { "type": "Free" },
    "j_slide":  { "type": "Free" }
  },
  "parts": {
    "l_crank":   { "type": "Link",   "a": "j_pivot", "b": "j_tip",   "length": 40.0 },
    "l_coupler": { "type": "Link",   "a": "j_tip",   "b": "j_slide", "length": 120.0 },
    "s_track":   { "type": "Slider", "joint": "j_slide",
                   "axis_origin": [0.0, 0.0], "axis_dir": [1.0, 0.0],
                   "range": [-200.0, 200.0] }
  },
  "drives": {
    "d_crank": {
      "kind": { "type": "Angular", "pivot_joint": "j_pivot", "tip_joint": "j_tip",
                "link": "l_crank", "range": [0.0, 6.283185] },
      "sweep": { "samples": 360, "direction": "Forward" }
    }
  },
  "points_of_interest": [
    { "id": "p_slider_out", "host": "l_coupler", "t": 1.0, "perp": 0.0 }
  ],
  "visualization": {
    "trace_model": {
      "type": "RollingPaper",
      "direction": "Up",
      "advance_per_cycle": 240.0
    }
  },
  "meta": { "name": "slider-crank", "iteration": 3, "notes": [] }
}
```

Mutations are a list of atomic ops, internally tagged the same way:

```json
{
  "mutations": [
    { "op": "add_joint",      "id": "j_tip",    "joint":  { "type": "Free" } },
    { "op": "modify_joint",   "id": "j_pivot",  "patch":  { "position": [10.0, 0.0] } },
    { "op": "remove_joint",   "id": "j_orphan" },

    { "op": "add_part",       "id": "l_crank",  "part":   { "type": "Link", "a": "j_pivot", "b": "j_tip", "length": 40.0 } },
    { "op": "modify_part",    "id": "l_crank",  "patch":  { "length": 42.0 } },
    { "op": "remove_part",    "id": "l_crank" },

    { "op": "add_drive",      "id": "d_crank",  "drive":  { "kind": { "type": "Angular", "..." : "..." }, "sweep": { "samples": 360, "direction": "Forward" } } },
    { "op": "remove_drive",   "id": "d_crank" },

    { "op": "add_poi",        "poi":    { "id": "p_out", "host": "l_coupler", "t": 0.5, "perp": 0.0 } },
    { "op": "remove_poi",     "id": "p_out" },

    { "op": "note", "text": "Increased crank length to broaden the slider motion." }
  ]
}
```

`modify_*` ops use shallow `patch` objects — only the listed fields change. Patches are type-checked against the target part's schema; unknown keys are rejected (see §6.3).

---

## 6. LLM Integration

### 6.1 Transport: tool-use, not XML scraping

Linkage talks to the LLM via the provider's **structured tool-use API** (Anthropic Messages `tools`, OpenAI `tools` / structured outputs). The LLM never has to emit raw JSON inside prose; it calls `propose_mutations`, and the orchestrator receives already-validated arguments. This eliminates an entire class of parse failures.

```rust
/// Provider-agnostic abstraction so we can swap Anthropic / OpenAI / local.
trait LlmClient: Send + Sync {
    async fn converse(
        &self,
        system: &str,
        history: &[Message],
        tools: &[ToolSpec],
    ) -> Result<LlmResponse, LlmError>;
}
```

One concrete tool:

```jsonc
{
  "name": "propose_mutations",
  "description": "Propose 1-5 atomic mutations to the current assembly. Prefer small incremental changes.",
  "input_schema": {
    "type": "object",
    "properties": {
      "reasoning":  { "type": "string", "description": "Mechanical thinking behind this change." },
      "mutations":  { "type": "array", "items": { "$ref": "#/defs/Mutation" }, "minItems": 1, "maxItems": 5 },
      "variants":   { "type": "array", "description": "Optional alternative mutation sets.",
                      "items": { "type": "array", "items": { "$ref": "#/defs/Mutation" } } }
    },
    "required": ["reasoning", "mutations"]
  }
}
```

Streaming tool-use deltas are applied to a *staging* assembly, so the renderer can show ghosts as the LLM writes.

> **Implementation note:** before wiring this up, verify the exact Anthropic Messages tool-use shape against current docs via context7 — the field names and streaming event formats have shifted across versions.

### 6.2 Context management

`<current_assembly>` is re-injected every turn, but it is not unbounded:

- Assemblies above ~50 parts are summarized: the orchestrator emits full JSON for recently-touched parts and a structural summary ("15 links forming a pantograph, 4 fixed joints, …") for the rest.
- Prior turns' telemetry is dropped from context after 2 turns unless explicitly referenced. Only the latest sweep telemetry is passed verbatim.
- Iteration history is summarized in a rolling `notes` list (one line per prior turn: "iter 7: shortened crank, slider motion became smoother").

### 6.3 Error feedback protocol

When the orchestrator rejects a mutation batch it returns to the LLM, on the next turn, a structured error:

```json
{
  "rejected": [
    { "op_index": 2,
      "code": "JOINT_NOT_FOUND",
      "message": "modify_part l_crank: joint 'j_missing' does not exist",
      "hint":    "Did you mean 'j_pivot'?" },
    { "op_index": 4,
      "code": "LENGTH_MISMATCH",
      "message": "Link l_coupler length 120.0 does not match |a−b|=98.4 within tolerance 0.1",
      "hint":    "Either modify the joint positions or adjust length." }
  ]
}
```

Error codes are a closed enum: `JOINT_NOT_FOUND`, `PART_NOT_FOUND`, `DRIVE_NOT_FOUND`, `DUPLICATE_ID`, `LENGTH_MISMATCH`, `UNKNOWN_PATCH_KEY`, `OVER_CONSTRAINED`, `UNDER_CONSTRAINED`, `DRIVE_TARGET_INVALID`, `DRIVE_COUNT_INVALID`, `SLIDER_RANGE_INVALID`, `POI_HOST_INVALID`, `RELAXATION_FAILED`. The LLM is prompted to reference codes in its retry reasoning.

Malformed references and impossible drive declarations are hard errors. By contrast, visually interesting but unstable behavior is allowed unless the current assembly or turn explicitly requests strict validation; a batch that produces branch switching, jitter, or incomplete settling may still be committed if it yields a drawable artifact and useful telemetry.

A mutation batch is atomic: *all* ops apply, or *none* do. On any rejection the assembly is rolled back and the batch is returned as errors.

### 6.4 System prompt

```
You are Linkage, a planar simulation design assistant.

You work with a 2D Verlet-based constraint simulator. You can create and modify
assemblies of joints (Fixed or Free), links, sliders, and drives (Angular or
Linear). Every drive has a range; the simulator sweeps that range and returns
telemetry from the simulated poses.

RULES:
- Call the `propose_mutations` tool. Do not emit JSON in prose.
- Explain your mechanical thinking in the `reasoning` field.
- Keep mutations small and incremental: 1–5 ops per turn.
- Ground your assembly with at least one Fixed joint.
- In MVP, produce at most one drive for any assembly you expect to sweep/commit.
- Every Free joint should usually be reachable via a chain of links/sliders to
  a Fixed joint, or the system will drift.
- Points of interest may host only on `Link` parts, never on `Slider` parts.
- When you receive <telemetry>, compare against the user's goal and propose
  corrective changes.
- When you receive <rejected>, read the error codes and fix the specific ops
  that failed before proposing new ones.
- Prefer motion that is visually legible, surprising, or expressive over motion
  that is merely conventional.
- Over- or under-constrained assemblies are not automatically bad if they still
  produce drawable and interesting motion; only fix them first when they make
  the result unusable or when strict validation is requested.

VOCABULARY:
- Joints:     Fixed (grounded), Free (relaxation-placed)
- Parts:      Link, Slider
- Drives:     Angular (rotates a link about a pivot), Linear (slides a joint on its track)
- Telemetry:  traces, range_of_motion, degrees_of_freedom, constraint_error,
              relaxation_status, point_paths

CURRENT ASSEMBLY:
<current_assembly>{{ASSEMBLY_JSON}}</current_assembly>

LAST SWEEP TELEMETRY:
<telemetry>{{TELEMETRY_JSON}}</telemetry>

LAST REJECTIONS (may be empty):
<rejected>{{REJECTIONS_JSON}}</rejected>
```

---

## 7. The Iterative Loop

### 7.1 Concurrency model

Linkage runs **two loops concurrently**, sharing a small lock-free `SharedState`. The primary handoff is the Committed Sweep buffer; auxiliary slots carry staged preview state and HUD activity state.

- **Design Loop** — a tokio async task that drives one LLM turn at a time: prompt → LLM tool-call → validate → relax/sweep → publish. When a sweep completes, its artifact is `ArcSwap::store`d into the Committed Sweep slot.
- **Render Loop** — the nannou event loop on its own thread, running at a steady 60 Hz. Every frame it `ArcSwap::load`s the Committed Sweep, advances a local phase `u ∈ [0, 1)`, samples the matching relaxed frame, and draws it.

The Render Loop **never awaits** anything from the Design Loop. It always has something to draw — on cold start, an empty placeholder sweep (just the grid + any seed assembly). The instant the first real sweep lands, playback begins automatically. If the LLM takes 30 seconds, the screen keeps animating the last committed sweep that whole time, and the HUD reflects what the Design Loop is doing.

```
┌──────────────────────────────────────────────────────────────┐
│ DESIGN LOOP (one turn)              runs whenever triggered  │
│                                                              │
│  [1. PROMPT]                                                 │
│    system + <current_assembly> + <telemetry> + <rejected>    │
│        │                                                     │
│        ▼                                                     │
│  [2. LLM tool-call: propose_mutations]                       │
│    Streaming deltas → StagedAssembly (renderer shows ghosts) │
│        │                                                     │
│        ▼                                                     │
│  [3. VALIDATE & APPLY]                                       │
│    Atomic; on error → rollback, publish <rejected>, GOTO 1   │
│        │                                                     │
│        ▼                                                     │
│  [4. RELAX & SWEEP]                                          │
│    For the active drive, step its range across `samples`     │
│    frames; Verlet-integrate and project constraints per      │
│    sample. Build SweepArtifact.                              │
│        │                                                     │
│        ▼                                                     │
│  [5. PUBLISH]                                                │
│    ArcSwap::store(new_artifact) → Committed Sweep            │
│    StagedAssembly cleared.                                   │
│        │                                                     │
│        ▼                                                     │
│  [6. FEEDBACK]                                               │
│    Telemetry summary stored for the next prompt turn.        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ RENDER LOOP (60 Hz)                runs unconditionally      │
│                                                              │
│  every frame:                                                │
│    artifact = committed_sweep.load()    // lock-free         │
│    u = (u + dt / cycle_seconds) mod 1                        │
│    frame = artifact.frames[floor(u * artifact.len())]        │
│    draw(frame)                                               │
│    if let Some(staged) = staged_assembly.load() {            │
│        draw_ghost(staged)                                    │
│    }                                                         │
│    draw_hud(activity_state.load())                           │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Phase preservation across swaps

When a new artifact is published mid-playback, the Render Loop does **not** reset to sample 0 — that would make every LLM turn feel like a stutter. Instead it keeps its current normalized `u` and maps it into the new artifact by ratio:

```rust
// on next frame after a swap:
let new_frame_idx = (u * new_artifact.frames.len() as f32) as usize;
```

If the assembly topology changed so drastically that phase alignment is meaningless (e.g., a Linear drive replacing an Angular one), the Design Loop flags the artifact with `reset_phase: true` and the Render Loop zeroes `u` at the next swap.

### 7.3 Activity state (HUD)

The Design Loop publishes a lightweight state enum (via a second `ArcSwap`) so the HUD can reflect what is happening *without* the Render Loop ever blocking:

```rust
enum ActivityState {
    Idle,
    Thinking   { turn: u32, elapsed_ms: u32 },          // LLM request in flight
    Staging    { ops_received: usize },                 // streaming tool-use deltas
    Validating { ops: usize },                          // applying a batch
    Relaxing   { sample: u32, total: u32 },             // sweep in progress
    Committed  { turn: u32, shown_until_ms: u64 },      // brief "new!" flash
    Error      { code: String, message: String },      // from rejection, auto-clears
}
```

The HUD shows this as a small status line in a corner — a spinner while Thinking/Relaxing, a counter of streamed ops while Staging, a brief highlight when a new sweep is Committed. The main animation continues underneath regardless.

### 7.4 Cancellation

If a new user intent arrives while the Design Loop is mid-turn (LLM still streaming, validator running, or sweep solving), the in-flight task receives a cooperative `CancellationToken`. On cancel:

- The LLM request is aborted (the provider client drops the stream).
- Any in-flight sweep samples are discarded; no partial artifact is published.
- The Committed Sweep is left untouched — the Render Loop keeps looping the last good artifact with no visible interruption.
- The Design Loop immediately begins a fresh turn with the new intent.

### 7.5 Ghost staging

While the LLM is streaming tool-use deltas for a new batch, the Orchestrator materializes a *staged* assembly in a separate slot: `ArcSwap<Option<StagedAssembly>>`. `StagedAssembly` is a render-only best-effort preview, not a validation target: while tool deltas are still arriving it may be incomplete or temporarily inconsistent. The renderer draws only resolvable joints, links, sliders, drives, and POIs, and skips broken references without failing the frame. The staged assembly is **not** swept — ghosts are drawn at the active drive's range midpoint for a stable preview. When the batch commits (or is rejected), the staging slot is cleared.

### 7.6 Auto-refine mode

When enabled, the Design Loop runs without user input. The system prompt gains a short goal clause restating the user's desired motion or behavior. Each turn publishes a new Committed Sweep, so the user sees the mechanism visibly refine itself on screen. **Termination conditions** (any trigger stops):

- `max_iterations` reached (default 15).
- `token_budget` exceeded (default 100k total across the run).
- `wall_clock_budget` exceeded (default 5 min).
- **Oscillation:** same assembly hash visited twice in the last 5 iterations.
- **Relaxation failure:** two consecutive sweeps return `RELAXATION_FAILED`.

All thresholds are configurable; defaults are calibrated for a single-user interactive session and should be re-tuned before any long-running unattended use. Quantitative convergence scoring and tolerance-based stopping are future work.

### 7.7 Undo / history

Each applied mutation batch is a snapshot in an `im::Vector<Assembly>` history stack. Undo is O(1): pop the stack, re-run the sweep on the prior assembly, publish the artifact. Storage is cheap thanks to structural sharing. Undo does *not* interrupt the Render Loop — the screen continues playing the current sweep until the re-swept prior assembly commits, at which point the animation transitions with phase preserved.

---

## 8. Simulation Engine

Planar constraint simulation built on Verlet integration plus iterative constraint projection. The default MVP workflow is still sampled playback from the user's point of view: the drive parameter `u` sweeps its range, each sample is warm-started from the previous one, and the engine projects toward a drawable constrained pose before recording telemetry. This is a simulation-first system, not a proof-oriented solver, so branch switches, near-lockups, and partial settling can be meaningful outcomes rather than automatic invalid states.

### 8.1 Integrator and constraints

- Every `Free` joint is a particle with `pos` and `prev_pos`. A sample begins from the previous sample's simulated state, or from a deterministic seeded layout on cold start.
- A small Verlet step advances particles before projection:

```rust
next_pos = pos + (pos - prev_pos) * damping + accel * dt * dt;
```

- MVP assemblies use zero external acceleration by default; the important motion comes from constraint projection, not gravity or impulses.
- Constraints are projected iteratively in Gauss-Seidel style:
  - `FixedJoint` keeps an anchor at an exact world position.
  - `LinkDistance` enforces `|a - b| = length`.
  - `SliderTrack` keeps a joint on a line and inside its travel range.
  - `AngularDriveTarget` projects a driven crank toward a target angle.
  - `LinearDriveTarget` projects a slider joint toward a target position on its track.
- A sample is considered converged enough for strict mode when max correction and max constraint error both fall below tolerance before the iteration budget is exhausted.
- In normal expressive mode, the engine may still publish a sample sequence that does not fully settle, as long as the artifact is drawable and the telemetry remains meaningful.
- The first sample of a sweep is warm-started from a deterministic seed assembly pose; subsequent samples warm-start from the previous sample so sweeps remain coherent and cheap.
- This choice is intentional: the same particle-and-constraint machinery can later grow from rigid linkages into ropes, textiles, and soft bodies without replacing the simulation core.

### 8.2 Determinism

The simulator should be bit-reproducible across runs on the same platform. This is a runtime repeatability requirement, not a claim that every assembly has one unique physically-correct solution. Ambiguous assemblies may have multiple plausible branches; the current state and projection order simply choose one repeatably.

- Ordered maps throughout persisted and published state (`BTreeMap`, not `HashMap`) so constraint assembly, telemetry, prompts, and snapshots all traverse in a stable order.
- No wall-clock or thread RNG.
- Fixed sample count per sweep; no early exit.
- Fixed iteration counts, projection ordering, and damping constants per assembly version.
- f32 throughout; no mixed-precision fallbacks.

### 8.3 Telemetry

```rust
struct Telemetry {
    relaxation_status: RelaxationStatus,   // Converged | MaxIterations | Diverged
    degrees_of_freedom: i32,
    max_constraint_error: f32,
    sample_count: u32,

    /// Trajectory of each point of interest across the sweep.
    point_paths: BTreeMap<String, Vec<Vec2>>,

    /// Trajectory of each Free joint across the sweep.
    joint_paths: BTreeMap<JointId, Vec<Vec2>>,

    /// Angular range realized at each Free joint (min, max) across the sweep.
    angle_extents: BTreeMap<JointId, (Angle, Angle)>,

    /// Bounding box of each point-of-interest path.
    path_bounds: BTreeMap<String, AABB>,

    /// Derived observations, LLM-friendly summaries.
    derived: DerivedObservations,
}

struct DerivedObservations {
    /// Short qualitative notes distilled from the sweep for prompt reuse.
    notes: Vec<String>,
    /// Self-intersection flag per link over the sweep.
    self_intersection: bool,
}
```

`DerivedObservations` is what the LLM actually reads most often — raw point paths are attached but summarized unless explicitly referenced by name. Quantitative goal scoring, tolerance bands, and metric-specific pass/fail evaluation are deferred to Future Development. Over time this section should grow more vocabulary for expressive events such as snaps, stalls, wobble, branch flips, and dense trace regions.

### 8.4 Sweep artifact and publication

A completed sweep produces a `SweepArtifact` — the unit of handoff between the Design Loop and the Render Loop.

```rust
struct SweepArtifact {
    assembly:    Arc<Assembly>,        // snapshot at the time of the sweep
    frames:      Vec<SolvedFrame>,     // one per sweep sample, deterministic order
    telemetry:   Telemetry,
    produced_at: Instant,              // for "Committed" HUD flash
    turn:        u32,                  // design-loop turn id
    reset_phase: bool,                 // true when topology changed drastically
}

struct SolvedFrame {
    u:               f32,                           // normalized sweep phase, 0..1
    joint_positions: BTreeMap<JointId, Vec2>,       // every joint, relaxed
    drive_values:    BTreeMap<DriveId, f32>,        // angle or position per drive
    poi_positions:   BTreeMap<String,  Vec2>,       // points of interest, relaxed
}
```

Publication is one atomic operation:

```rust
struct StagedAssembly {
    joints:             BTreeMap<JointId, Joint>,
    parts:              BTreeMap<PartId, Part>,
    drives:             BTreeMap<DriveId, Drive>,
    points_of_interest: Vec<PointOfInterest>,
}

struct SharedState {
    committed_sweep:  arc_swap::ArcSwap<SweepArtifact>,       // always present after boot
    staged_assembly:  arc_swap::ArcSwap<Option<StagedAssembly>>, // Some while LLM is streaming
    activity_state:   arc_swap::ArcSwap<ActivityState>,
}

// Design Loop, on successful sweep:
shared.committed_sweep.store(Arc::new(new_artifact));
shared.staged_assembly.store(Arc::new(None));
```

On boot, `committed_sweep` is seeded with a placeholder artifact containing one empty frame — the Render Loop can start drawing immediately without ever checking for `None`.

Artifacts are small (tens of KB for a typical 360-sample slider-crank), and readers hold an `Arc<SweepArtifact>` only for the duration of a single render frame. The previous artifact drops automatically when the last reader releases it.

---

## 9. Nannou Renderer

The renderer supports both the normal interactive windowed mode and a headless mode for testing, offline frame capture, and non-interactive inspection. Headless mode consumes the same `SharedState` and playback logic, but skips user input and window presentation.

### 9.1 Layers (bottom → top)

1. **Grid** — faint coordinate grid, snappable.
2. **Traces** — trajectories of POIs and selected Free joints drawn from the latest Committed Sweep. POI traces render as dotted lines; selected Free-joint traces render as fading solid lines. All traces are color-coded per point.
3. **Slider tracks** — rendered as dashed lines with range caps.
4. **Links** — rounded rectangles or lines between joints, labeled with length.
5. **Joints** — filled circles. Fixed = square outline; Free = round.
6. **Drive indicators** — arc arrows on Angular drives; straight double-ended arrows on Linear drives, showing range.
7. **Ghost overlay** — semi-transparent preview from `staged_assembly`, shown before acceptance.
8. **HUD** — iteration, relaxation status, DoF, token budget, current objective, LLM activity spinner.

### 9.2 Interaction

- Click joint/part → select, show properties panel.
- Drag Free joint → hint position for the next relaxation seed (warm-start only; does not pin).
- Drag Fixed joint → move the anchor.
- `Space` → play/pause sweep animation.
- `R` → trigger one LLM refine iteration.
- `A` → toggle auto-refine.
- `V` → cycle LLM-proposed variants.
- `U` / `Ctrl+Z` → undo last mutation batch.
- `Ctrl+Shift+Z` / `Ctrl+Y` → redo.
- `Ctrl+S` → export assembly JSON.
- `P` → pin a new POI under the cursor (snaps to nearest link).

### 9.3 Continuous playback

At 60 Hz, the renderer calls `shared.committed_sweep.load()` — lock-free, no `await` — and holds the returned `Arc<SweepArtifact>` for the duration of the frame.

It advances a local phase counter `u += dt / cycle_seconds`, wraps `u` into `[0, 1)`, and reads `frames[floor(u * frames.len())]`.

On buffer swap, `u` is preserved rather than reset to `0`. If the incoming artifact has `reset_phase: true`, then `u = 0`. When frame counts differ, phase is mapped by normalized ratio.

Playback time is defined over the normalized committed sweep interval, not the physical angle or distance covered by the drive. For `Forward` and `Reverse`, one traversal of the published interval takes `cycle_seconds`, regardless of drive range. For `PingPong`, the renderer traverses the same interval forward and then backward, so a full out-and-back loop takes `2 * cycle_seconds`.

`cycle_seconds` defaults to `2.0` s for Angular sweeps and `1.5` s for Linear sweeps. The HUD exposes a playback-speed slider.

`Space` pauses phase advancement, but interaction remains live: pan, zoom, selection, and POI pinning all continue to work while paused.

If `shared.staged_assembly.load()` is `Some`, the renderer draws it as translucent ghosts on top of the current playback frame. Ghosts render at the active drive's range midpoint; staging is not swept.

The HUD reads `shared.activity_state.load()` and shows the current state — `Thinking` / `Staging` / `Validating` / `Relaxing` / `Committed` / `Error` — without ever stalling the render.

---

## 10. Exploration Modes

Selected via a system-prompt modifier; all modes talk to the same `propose_mutations` tool.

| Mode          | Prompt modifier                                              | Behavior                             |
|---------------|--------------------------------------------------------------|--------------------------------------|
| Directed      | "Achieve this specific motion: …"                            | Iterates toward a user-described motion |
| Explorative   | "Show me 3 surprising linkages using only links and sliders" | Divergent; fills `variants`          |
| Historical    | "Recreate Watt's parallel motion linkage"                    | References known mechanisms          |
| Adversarial   | "Find the simplest mechanism that locks up"                  | Edge-case probing                    |

Batch parametric exploration and tolerance-aware ranking are future work.

---

## 11. Technical Stack

Aligned with the existing `Cargo.toml` of this repo (`nannou 0.19`, `serde 1`, edition 2024).

| Layer             | Choice                               | Rationale                                     |
|-------------------|--------------------------------------|-----------------------------------------------|
| Language          | Rust (edition 2024)                  | Matches existing repo; performance; safety    |
| Rendering         | `nannou = "0.19"`                    | Already in repo; creative-coding 2D           |
| Serialization     | `serde`, `serde_json`                | Assembly ↔ JSON ↔ LLM                         |
| LLM client        | Trait + `reqwest = "0.12"` (streaming) | Provider-agnostic; swap Anthropic / OpenAI / local |
| Async runtime     | `tokio = "1"`                        | Required by reqwest                           |
| Shared state      | `arc-swap = "1"`                     | Lock-free publish of Committed Sweep + staged assembly + activity state |
| Simulation math   | Hand-rolled Verlet + constraint projection | Matches MVP needs; extends to ropes / cloth / soft bodies |
| Persistent state  | `im = "15"`                          | Cheap snapshot history                        |
| Errors            | `thiserror = "2"` + `anyhow = "1"`   | Error taxonomy for §6.3 rejections            |
| Observability     | `tracing = "0.1"` + `tracing-subscriber` | Loop visibility, token/latency spans       |
| IDs               | Stable strings; no UUID crate        | LLM-authored IDs are stable across turns      |

> **Before pinning exact minor versions:** verify via context7 (or `cargo search`) that nothing has shifted — these are the versions current as of this writing, but crate ecosystems move.

### 11.1 LLM client trait

```rust
#[derive(Debug)]
pub enum LlmProvider {
    Anthropic { model: String, api_key: String },
    OpenAI    { model: String, api_key: String },
    Local     { endpoint: String, model: String },
}

#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn converse(
        &self,
        system: &str,
        history: &[Message],
        tools: &[ToolSpec],
    ) -> Result<LlmResponse, LlmError>;
}
```

### 11.2 Thread model

The tokio multi-thread runtime hosts the Design Loop async task. Nannou owns its own thread for the Render Loop event loop. The only cross-thread state is the three `ArcSwap` slots in `SharedState`: `committed_sweep`, `staged_assembly`, and `activity_state`. There are no mutexes on the hot path.

---

## 12. Testing Strategy

- **Unit:** mutation apply/reject, ID resolution, patch validation, constraint projection. Use `insta` snapshots for telemetry of canonical assemblies.
- **Property tests (`proptest`):**
  - Idempotence: apply → serialize → deserialize → apply-again → same state.
  - Direction semantics: `Forward`, `Reverse`, and `PingPong` traverse the published interval as specified.
  - Repeatability: same assembly + same constants + same platform returns the same sampled artifact.
- **Golden assemblies:** a directory of hand-authored assemblies (four-bar, slider-crank, pantograph, Watt I, Roberts, plus a few intentionally ambiguous toys) with recorded telemetry. Changing the simulation engine or projection order without changing any golden is a CI failure.
- **Repeatable sweep tests:** run the same assembly 100× and assert byte-identical telemetry on the same platform.
- **Headless render tests:** run the renderer without a window against golden sweep artifacts and assert stable trace output, including dotted POI path rendering.
- **LLM integration tests:** mocked `LlmClient` that replays recorded tool-call responses; asserts the loop correctly handles validation rejections, iterative feedback, and termination.

---

## 13. MVP Scope

### Phase 1 — Static Linkages + Single-shot LLM

- [ ] Data model: joints (Fixed/Free), links, sliders, angular + linear drives, POIs
- [ ] JSON (de)serialization with internally-tagged schema
- [ ] LLM client trait + Anthropic implementation using tool-use
- [ ] Orchestrator: validate + apply mutation batches atomically
- [ ] Error feedback protocol (§6.3) end-to-end
- [ ] Nannou renderer: grid, joints, links, sliders, HUD (no traces yet)
- [ ] Single-shot: user prompt → LLM → validated mutations → relax and render (no sweep yet)
- **Exit criterion:** user can type "make a four-bar linkage grounded at (-50,0) and (50,0)" and see it rendered, including `modify_part` corrections after an initial rejection.

### Phase 2 — Sweep + Feedback Loop

- [ ] Verlet integrator + constraint projection (distance + point-on-line + drive target) with repeatable sweeping
- [ ] Telemetry collection (§8.3)
- [ ] Point-of-interest system + path rendering
- [ ] Ghost overlay for staged mutations
- [ ] Full loop: prompt → mutate → sweep → render → feedback
- [ ] Undo/redo via `im` history
- **Exit criterion:** user can describe a desired motion or vibe, watch the assembly update over several telemetry-guided iterations, and see the latest committed sweep animate without blocking.

### Phase 3 — Exploration

- [ ] Auto-refine with termination conditions (§7.6)
- [ ] Variant generation and cycling
- [ ] Historical / explorative / adversarial prompt modifiers
- [ ] Assembly export (JSON, SVG)
- **Exit criterion:** auto-refine produces visibly distinct, telemetry-informed variants within the token budget and lets the user inspect them interactively.

---

## 14. Example Session

A slider-crank — the simplest mechanism that exercises links, a slider, a Free joint on a track, an angular drive, and a point of interest.

```
User: "Create a compact slider-crank with a clear horizontal slider motion."

LLM (iter 1), tool call propose_mutations:
  reasoning: "A slider-crank converts crank rotation to slider translation.
    Starting with a simple grounded crank, a longer coupler, and a horizontal
    slider track should produce readable motion and leave room to refine."
  mutations: [
    { "op": "add_joint", "id": "j_pivot", "joint": { "type": "Fixed", "position": [0,0] } },
    { "op": "add_joint", "id": "j_tip",   "joint": { "type": "Free" } },
    { "op": "add_joint", "id": "j_slide", "joint": { "type": "Free" } },
    { "op": "add_part",  "id": "l_crank",
      "part": { "type": "Link", "a": "j_pivot", "b": "j_tip", "length": 50 } },
    { "op": "add_part",  "id": "l_coupler",
      "part": { "type": "Link", "a": "j_tip", "b": "j_slide", "length": 150 } },
    { "op": "add_part",  "id": "s_track",
      "part": { "type": "Slider", "joint": "j_slide",
                "axis_origin": [0,0], "axis_dir": [1,0], "range": [-250,250] } },
    { "op": "add_drive", "id": "d_crank",
      "drive": { "kind": { "type": "Angular", "pivot_joint": "j_pivot",
                           "tip_joint": "j_tip", "link": "l_crank",
                           "range": [0, 6.283185] },
                 "sweep": { "samples": 360, "direction": "Forward" } } },
    { "op": "add_poi", "poi": { "id": "p_out", "host": "l_coupler", "t": 1.0, "perp": 0 } }
  ]

[Orchestrator applies atomically. Sweep runs.]
[Telemetry: the slider motion is readable and horizontal, the coupler path is clean, no self-intersection.]

LLM (iter 2), reading telemetry, refines for clarity:
  reasoning: "The motion works, but the coupler can be cleaned up and the
    slider travel can be made more visually legible."
  mutations: [
    { "op": "modify_part", "id": "l_coupler", "patch": { "length": 170 } },
    { "op": "note", "text": "Lengthened the coupler to smooth the visible motion." }
  ]

User: "Make the slider travel feel broader."

LLM (iter 3):
  reasoning: "Broadening the visible sweep is easiest by increasing the crank
    radius while keeping the rest of the layout stable."
  mutations: [
    { "op": "modify_part", "id": "l_crank", "patch": { "length": 60 } }
  ]

[Sweep: the slider path spans a wider horizontal range and remains stable.]
```

---

## 15. Future Development

Out of MVP scope; the data model, simulator, and prompt vocabulary should not preclude any of these.

- **Gears, gear trains, belts, pulleys.** Meshing as a constraint coupling angular drives. Rendering as involute tooth silhouettes. Will require extending `Part`, `Constraint`, and the projection set. Deferred because gear meshing introduces tangency constraints and ratio coupling that complicate both the simulator and the LLM prompt vocabulary; not worth the cost until the linkage loop is solid.
- **Cams and cam-followers.** Profile-lookup constraints.
- **Ropes, textiles, and soft bodies.** This is the main reason the core is Verlet-based. These would arrive as additional particles plus distance / bending / area / attachment constraints rather than a second simulation architecture.
- **Springs.** Natural to add once particles and compliance are first-class, but still deferred until the expressive rigid-linkage loop is stable.
- **3D linkages.** The data model keeps Vec2 explicit today; migrating to Vec3 is non-trivial but the topological graph carries over. `wgpu` via Nannou would handle the rendering.
- **CAD export.** SVG is feasible in Phase 3. STEP / IGES requires either an external tool or a heavy Rust crate.
- **Local LLM.** The tool-use schema is simple enough for a fine-tuned 7B model. Worth trying once the protocol stabilizes.
- **Multi-user / collaborative editing.** JSON assembly is naturally shareable; live sync is a separate problem.
- **Collision detection.** Link self-intersection is flagged today as a boolean; proper swept-volume collision checking is Future.
- **Quantitative goal scoring / tolerance-aware evaluation.** Metric-target-tolerance objective functions, parametric sweeps, structured goal schemas, and pass/fail scoring against numeric tolerances are explicitly out of MVP scope. The telemetry loop should preserve enough raw paths and summaries to support this later without redesigning the architecture.

---

## 16. Open Questions

- **Constraint compliance model.** MVP can start with rigid projection and global damping. The open question is when to introduce per-constraint compliance / softness for future cloth and soft-body work.
- **Iteration budget vs. quality.** How many projection iterations per sample are enough for compelling telemetry without making sweeps too expensive?
- **POI discovery.** Should the LLM be able to ask "add a POI wherever the path looks most interesting", or does it always specify (`host`, `t`, `perp`)? Lean toward explicit for now; add a `find_interesting_poi` helper tool later if needed.
- **Session persistence.** Out of MVP, but the JSON format is the natural save file; Phase 3 could add a simple `./sessions/` directory or similar repo-local save path.
