# Runtime guards — anti-takeoff / anti-corkscrew defensive stack (2026-07-09)

Response to "do anything you can to prevent corkscrewing, slews, takeoffs".
Motivating data: the 07052026_4_an smoke tests
(doc/FINDINGS_2026-07-09_07052026_rviz.md) produced a 3.3e7 m takeoff (leg 1),
a crash (leg 2), and SLOSH_HI with 58 reversals (leg 3) — three failure modes,
none caught by the existing levers. The existing stack *prevents* (X-ICP
ternary, keyframe degenGate, partial budgets) or *bounds per held axis*
(governor); nothing acted on the failure modes when they were already in
progress, and nothing acted at all when the gate flagged the wrong axis (or
none). These three guards close that: they act on the **output** and the
**velocity state**, unconditionally on gate correctness.

All three are default-off (bit-identical) and independently A/B-able.
Overlay: `cfg/examples/ouster_tunnel_guards.yaml`.

## 1. Physics fuse (anti-takeoff) — `odom/fuse/maxStep{Trans,Rot}`

`dlio::fusePose` (include/dlio/physics_fuse.h, pure, unit-tested in
test/test_physics_fuse.cpp) clamps the TOTAL per-scan output step vs the
previous output pose — any direction — to a physical bound, after the
governor. With `maxStepTrans: 1.0` at 10 Hz the published pose cannot move
faster than 10 m/s: **a takeoff is impossible at the output level by
construction**, including the cases the governor structurally cannot catch:

- the gate missed the scan entirely (the ~20-29% miss rate that seeds
  map-lock jumps),
- the runaway rides an axis the gate never flagged,
- the IMU prior itself diverges (bad deskew, bias runaway — leg 1's
  `IMU data does not cover the scan period` storm; `maxCorr*` passes a
  diverging prior untouched because it bounds only the correction).

A tripped scan is physically implausible, so it is flagged loudly
(`Physics Fuse Trips` in /diagnostics, throttled ERROR log) and **vetoes
keyframing** — a fused pose must never be planted in the map. NaN steps hold
the previous position (composes with the NaN-guard family from the crash
hunts).

Sizing: `maxStepTrans ≈ platform max speed / scan rate × margin`. The crawler
does ≤ 2 m/s; 1.0 m/scan at 10 Hz is a 5× margin — trips indicate estimator
failure, not fast driving. Distinct from `odom/degenGov/maxStep*` (held axes
only) and `odom/gicp/maxCorr*` (correction vs prior, prior excluded).

**Known constraint before you tighten these** (2026-07-26 audit): the caps are
per *scan*, not per second, but the real inter-scan period varies — a dropped
scan under CPU starvation (this repo's documented #1 cause of tunnel-run
failures) means genuinely more motion between outputs, which the fuse reads as
implausible. The 5× margin above absorbs roughly a 5× scan gap at full speed,
but it is doing double duty for speed *and* period, so tightening the caps eats
the dropped-scan headroom silently. If you tighten them — or run a faster
platform — convert to a velocity bound (`cap = max_speed × actual dt`), which is
exact and needs no assumed nominal rate; `getNextPose` has the true period
available (`scan_stamp - prev_scan_stamp` is not yet advanced there). Watch
`Physics Fuse Trips` against `Scans Dropped est` in Round 0 to tell a genuine
trip from a dropped-scan artefact.

## 2. Slosh guard (anti-corkscrew) — `odom/slosh/*`

`dlio::SloshGuard` (include/dlio/slosh_guard.h, pure, unit-tested in
test/test_slosh_guard.cpp) is the offline harness's reversal verdict
(`analyze_traj.py` `revs` → SLOSH_HI) moved into the estimator, so the node
can respond *while the corkscrew is happening* instead of after the bag ends.

Detector: per scan, project the output step onto the tracked weak axis (the
gate's weakest held translation direction — the gate records in ascending
eigenvalue order, so `front()` is the weakest, not an arbitrary member —
sign-aligned scan-to-scan; it **persists across scans the gate misses**,
because gate chatter is itself part of the oscillation loop —
doc/FUSION_ARCHITECTURE.md Loop B). Steps below `deadband` are ignored. Over
the last `window` active steps, the sign-flip fraction is the oscillation
score: a straight traverse scores ~0, a corkscrew ~1. Hysteresis
(`engageFrac`/`disengageFrac`) prevents response chatter.

**Axis lifetime** (`axisHoldScans`, 2026-07-26 correctness fix). Persistence is
bounded in two ways, because an axis that outlives its conditions is a source of
FALSE engagement in healthy operation:
- *Ageing*: after `axisHoldScans` consecutive scans with no weak axis from the
  gate, the tracked axis goes invalid and the evidence window drains one sample
  per scan. (Before this fix the validity flag latched true on the first
  degenerate scan and never cleared, which made the window-decay path
  unreachable in production and let a stale axis score flips indefinitely.)
- *Identity change*: the weak-axis set can change membership, so if the new
  direction differs from the tracked one by more than 30° it is a different
  physical DOF — the accumulated sign history would read as noise projected on
  it, so the evidence is reset outright rather than mixed.

Response while engaged:
- **velocity damping along the tracked axis** (`velDamp`) — the oscillation's
  flywheel is the observer velocity state; draining it removes the energy
  store that carries the overshoot through each reversal,
- **keyframe veto** — a mid-oscillation pose planting duplicated wall
  structure is the map self-contamination loop (Loop A) the degenGate
  targets, caught here by the output-domain detector even when the gate
  misses,
- diagnostics: `Slosh Guard Engaged` / `Slosh Reversal Fraction` /
  `Slosh Guard Activations`.

## 3. Degeneracy velocity damping (flywheel kill) — `odom/geo/degenVelDamp`

Along a held axis the registration supplies no correction, so
`updateState()`'s velocity integrates IMU error unopposed — this is why the
held prior dead-reckons into the km-scale runaway, and why dliio's twist is
deliberately not fused downstream (doc/FUSION_ARCHITECTURE.md). `degenVelDamp`
decays the along-axis velocity component by `(1 - damp)` per held scan,
cross-axis untouched (reuses `attenuateAlongHeldAxes`, the LODESTAR-slice
machinery — but where `degenObsGain` scales the *correction*, this decays the
*state*). At 0.3/scan, 10 Hz: ~97% decay per second — a bounded coast to stop
instead of constant-velocity dead-reckoning. The honest cost: genuine constant
velocity along the weak axis (driving down the tunnel!) is also damped toward
zero — along that axis the estimate leans on whatever else constrains it
(image terms, rig odometry via r_l). This is the standing conservative choice;
prefer it ON for the crawler (slow, stop-and-go) and reconsider for
constant-speed platforms.

## Interaction map

| Failure stage | Lever |
|---|---|
| prevent gate chatter | X-ICP ternary (+ partial budget) |
| bound held-axis step | governor `maxStep*` |
| bound ANY output step | **physics fuse** |
| drain the flywheel while held | **degenVelDamp** |
| detect + respond to oscillation in progress | **slosh guard** |
| keep bad poses out of the map | keyframe degenGate + **fuse/slosh vetoes** |
| tell downstream the axis is untrusted | cov inflation, publishTf off + r_l |

Ordering per scan: align → gate → governor → **fuse** → **slosh detector** →
keyframe vetoes → observer (`degenObsGain` → **degenVelDamp** → **slosh
damping**).

## Validation plan

1. **No-op check** (06042026 clean runs, n≥12): `Physics Fuse Trips` must be
   0 and `Slosh Guard Activations` must be 0 on runs the baseline handles —
   the guards must not fire on healthy operation.
2. **07052026_4_an A/B** (the three-failure-mode bag): guards overlay vs
   without, on top of tunnel+xicp. Watch max|x| (fuse should cap leg-1-style
   takeoffs at the trip ceiling), `revs`/straightness (slosh guard vs leg 3's
   58 reversals), and trips/activations against the failure timeline.
3. **Per-mechanism isolation**: each guard zeroed in turn (the overlay header
   documents the knobs) to attribute any improvement.
