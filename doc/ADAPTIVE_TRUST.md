# Adaptive trust-weights — design

> **Status: design only.** Nothing in the runtime adaptation described here is
> implemented yet. The three *groundwork* pieces it builds on are landed and
> off-by-default (see "Groundwork" below). This doc is the design companion to
> `doc/VISUAL_TERM.md` (the residual-term design), `doc/TUNNEL_FINDINGS.md` (the
> empirical journey), and `doc/TUNNEL_SWEEP_RUNBOOK.md` (the bench procedure).

## Why

dliio fuses several residual terms — geometric GICP, a photometric
(intensity/reflectivity) term, two camera terms (frame-to-frame, frame-to-map),
and a COIN-LIO LiDAR-intensity-image term. Each carries a hand-tuned scalar
`weight`. The original motivating question: *can the estimator switch/blend
between intensity, reflectivity, VO and LiDAR GICP through a runtime trust model,
so we stop hand-tuning per environment?*

> **Motivation update (2026-06-17).** This doc was first written when the tunnel
> looked like a hard, hand-tuning-intensive degeneracy problem. Two setup bugs
> then surfaced (`doc/FINDINGS_2026-06-17.md`): a 180° IMU-yaw extrinsic error
> and a `VoxelGrid`-dead photometric channel. With the extrinsic fixed,
> **geometry-only tracks ~200 m** on that bag — so "hand-tuning the aux-term
> weights is painful" is a **weaker** motivation than it appeared; the apparent
> pain was largely a config bug. The case for adaptive trust now rests on
> **robustness to sensor degradation**, not on tunnel unobservability: the live
> failure is a specular floor-water pool (`FINDINGS_2026-06-17.md` §3) where the
> geometric floor constraint *drops out* and the photometric term gets
> viewpoint-dependent garbage — exactly a "down-weight a term when its
> observability/quality collapses" problem. Read the policy below through that
> lens (graceful degradation under dropout/specular/blinding), not "tune the
> tunnel."

The honest version is **adaptive, observability-aware information weighting**,
not "use whichever term fits best." This doc specifies that policy and the order
to build it.

## The fusion today

All terms accumulate into one shared 6×6 information-form system in
`NanoGICP::computeTransformation` / `linearize` (`src/nano_gicp/nano_gicp.cc`):
`H += weight · JᵀJ`, `b += weight · Jᵀr`, solved per LM iteration. So an adaptive
"trust weight" is literally: make each term's `weight` (and the gate's behaviour)
a function of per-scan runtime signals instead of a YAML constant. The plumbing is
already there:

- Per-term weight setters: `setPhotometricWeight`, `setVisualWeight`,
  `setVisualMapWeight`, `setLidarMapWeight` — and they are **already live-callable**
  each scan (the `kLive` / `applyLiveParams` path in `odom.cc`).
- The degeneracy gate (Zhang/Kaess/Singh solution remapping) holds the IMU prior
  on unobservable eigen-directions; the **soft gate** (`degeneracySoftness`) makes
  that hold continuous; the **IMU-consistency clamp** (`maxCorrTrans/Rot`) bounds
  the per-scan correction magnitude.
- Per-residual IRLS Huber down-weighting already exists on the photometric and
  visual terms (a micro-version of trust, at the residual level).

In other words there are already **three hand-tuned trust heuristics that don't
know about each other** (Huber per-residual, gate per-direction, clamp per-scan).
The goal is to unify and soften them into one coherent, signal-driven policy.

## The central principle: trust ≠ fit

The intuitive scheme — "trust the term with the lowest residual" — **actively
reinforces the #1 tunnel failure**. Map-lock is a *low-residual* state: the scan
snaps onto a self-similar slice of the corridor and fits beautifully while being
wrong along the axis. So trust must key on:

1. **Observability** — the geometric conditioning per direction (the gate already
   computes this; now surfaced as `Geo Rot/Trans Trust Margin`).
2. **Prior-consistency** — disagreement with the high-confidence IMU prior over a
   short scan (the clamp already enforces a hard version).

…and only *then* on fit quality (RMS, count). The right target is **per-axis**,
not per-term: don't "switch to reflectivity," route whichever healthy term
carries information *on the axis the geometry is blind to* into that axis.

## Groundwork (landed, all default-off / bit-identical)

| commit | piece | what it gives the policy |
|---|---|---|
| `7fde911` | **soft degeneracy gate** (`degeneracySoftness`) | the geometric term's trust made continuous (no scan-to-scan toggling) |
| `817e0ed` | **term mass-normalization** (`*RefCount`, `refCountScale`) | per-term weights made **commensurable & density-stable** — required before any cross-term comparison |
| `43b8bbc` | **per-term trust telemetry** | the signals the policy keys on, now observable in `/diagnostics` |

## The telemetry the policy keys on

Already published per scan on `/diagnostics` (read-only; see
`TUNNEL_SWEEP_RUNBOOK.md §3`):

| term | observability / routing signal | quality signals |
|---|---|---|
| geometric | `Geo Rot Trust Margin`, `Geo Trans Trust Margin` (weakest-axis eig / gate thresh; >1 trusted, <1 held, ~1 marginal) | `Degenerate Directions (current)` |
| photometric | (does it stiffen the weak axis — gate Rayleigh test) | `Photometric Points`, `Photometric Residual RMS` |
| camera f2f | `Visual Rescued Axes` | `Visual Points`, `Visual Residual RMS` |
| camera f2m | — | `Visual Map Points`, `Visual Map RMS` |
| LiDAR-image | — | `Lidar Map Points`, `Lidar Map RMS` |

Every term now exposes a **count + quality** pair, and the geometric term exposes
its **observability margin** — the complete signal set for the policy below.

## Proposed policy

A master enable flag (default **off** ⇒ bit-identical to today). When on, two
layers, build in order:

### Layer 1 — per-term availability/quality trust (scalar "soft switching")

Each auxiliary term (NOT geometric — it is the trusted baseline) gets a trust
multiplier `μ ∈ [floor, 1]` recomputed each scan, and its effective weight is
`base_weight · μ` (applied through the existing setters):

- `μ` rises with **count above a floor** (enough gradient-bearing / feature
  points) and **falls when RMS exceeds a ceiling** (fit breaking down).
- **Smoothed**: `μ` is an EMA over scans (time constant ~0.5–1 s) with a rate
  limit, so a term *fades* in/out rather than toggling — this is the "seamless
  switching" the question asked for, and it is what handles sensor dropout (VO in
  the dark, reflectivity on a non-Ouster topic, the camera f2m's measurement
  starvation in Finding 4) without a hard if/else.
- **Floored**: `μ ≥ floor > 0` while the term has *any* signal, so a term that is
  briefly noisy is down-weighted, not killed (avoids lock-in; see Risks).

This alone removes most per-environment weight tuning: you set a *base* weight and
a sensitivity, and the term self-attenuates when it is not trustworthy.

### Layer 2 — per-axis routing (the real prize; deferred)

Use `Geo Trans/Rot Trust Margin < 1` to identify the blind axis, and up-weight
whichever Layer-1-healthy term **stiffens that specific eigen-direction** (the
gate's existing rescue Rayleigh test, `v·Hv > thresh`, already measures this).
This is not "switching"; it is per-axis information routing — the principled
answer to the original question. It needs the term's contribution projected onto
the degenerate eigenvector, so it is more invasive than Layer 1 and should follow
it. Related: FAST-LIVO2 / COIN-LIO complementary patch/feature selection
(`REVIEW` item #8).

## Safety & failure modes

Adaptive weighting introduces a new instability class that fixed weights do not:

- **Oscillation** — `μ` ringing scan-to-scan. Mitigation: EMA + rate limit +
  hysteresis band.
- **Lock-in / death-spiral** — a down-weighted term stops being corrected →
  drifts → looks *even less* trustworthy → stuck at floor forever. Mitigation:
  non-zero floor; periodic "probe" of a floored term; never adapt on a single rep.
- **Confident-but-wrong** — over-trusting a low-RMS map-locked term. Mitigation:
  the central principle — gate trust on observability/prior-consistency, never on
  RMS alone.
- **State / reproducibility** — adaptation adds estimator state, complicating the
  "bit-identical when disabled" guarantee. Mitigation: master flag defaults off;
  the geometric path stays untouched; keep an explicit off-switch tested.

## Validation plan

The tunnel is a chaotic basin — **n ≥ 5, headless on free cores, sim-time +
`--clock`** (per `TUNNEL_SWEEP_RUNBOOK.md §0`). Compare adaptive vs the fixed-weight
baselines **B0** (`ouster_tunnel.yaml`) and **B1** (+LiDAR-image). Success = same
~100 m tracking with **fewer divergences and lower twist**, and graceful behaviour
under a deliberately dropped/blinded sensor. Watch the new telemetry for the
failure signatures above (μ ringing; a term pinned at floor; RMS low while the
axis is wrong).

## What to observe first (before implementing)

Because the telemetry is already live and read-only, **record `/diagnostics` on
the 06042026 bag and plot the signals against the known map-lock / twist events
before writing any adaptation.** Specifically: does `Geo Trans Trust Margin`
actually cross ~1 at the onset of a divergence? Does `Photometric Residual RMS`
rise *before* a map-lock or only after? This tells us which signal→weight mapping
is worth building and with what time constant, and avoids designing a policy
against a signal that turns out to be a lagging indicator.

## Prior art

> A full survey of the relevant papers and OSS packages — with a dliio-specific
> "what's portable and how invasive" shortlist — is in
> `doc/ROBUSTNESS_RESEARCH.md`. The most aligned external method is Super
> Odometry / SuperLoc's per-direction confidence (γ_trans/γ_rot → prior
> covariance), which is essentially this doc's Layer-2 routing computed pre-solve.

- Robust M-estimators / IRLS — already in dliio (Huber on the photometric/visual
  residuals).
- Switchable constraints (Sünderhauf & Protzel) and dynamic covariance scaling
  (Agarwal et al.) — back-end analogues of soft, data-driven constraint weighting.
- FAST-LIVO2 — degeneracy-aware LiDAR-visual front-end fusion (the per-axis
  routing analogue).
- Zhang/Kaess/Singh observability-aware degeneracy detection — already the basis
  of dliio's gate.

## Status / not-yet-done

Groundwork done (gate softened, weights commensurable, signals visible). **The
runtime adaptation itself — Layers 1 and 2 — is not implemented.** It is the first
step that changes the solve's *decisions* rather than instrumenting them, so it is
gated on (a) the "observe first" pass above and (b) an explicit go-ahead, and it
will land behind a default-off master flag with the safety mechanisms above.

> **Bench reality check (2026-06-18 → 2026-06-22).** The n=5 sweep found that a
> scalar weight does not reliably move the divergence basin; the **2026-06-22
> mechanism-level A/Bs** (`doc/FINDINGS_2026-06-22.md`, summarized in
> `doc/ROBUSTNESS_RESEARCH.md` "Validation status") then closed the question: every
> available auxiliary term was instrumented and **none re-constrains the degenerate
> (yaw-rotation) axis on this rig.** Crucially for this doc, **Layer 2 was even
> built and A/B'd** — `conditionScaleTerm` *is* per-direction routing for the
> LiDAR-image term, and it works mechanically (rescues the axis) but **tripled
> divergence**, because the reflectivity anchor it routes is along-axis *aliased*:
> per-direction routing is only as good as the observation it routes, and on this
> rig there is no clean along-axis observation to route (the camera term works but
> is FOV-limited; reflectivity/near-IR are aliased/isotropic). So Layer 2 is not
> blocked on *policy* — it is blocked on a **sensor/geometry change** that supplies
> a non-aliased along-axis observation. The adaptive-trust machinery remains valid
> for a rig where that observation exists.
