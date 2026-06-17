# dliio tunnel findings — 06042026 Ouster bag

Consolidated empirical findings from the `resple_test_ws` benchmark/replay harness
on the **06042026** dataset (Ouster OS-64 LIO bag: `/ouster/points` + `/ouster/imu`
+ `/lucid_camera_1` + `/tf_static`, ~201 s). Companion to `doc/VISUAL_TERM.md`
(which is the *design* of the photometric terms); this file is the *journey and
results*. Raw artifacts live in the (gitignored) `resple_test_ws/results/` dirs
named below — the tables here are the durable record.

## The problem

The bag traverses a long, smooth, straight rectangular **tunnel** — a
geometrically self-similar environment. Scan-to-map GICP is **unobservable along
the tunnel axis** (no feature pins forward translation), which produces two
coupled failure modes:

- **Map-lock / drag-back:** the current scan snaps onto an earlier identical-looking
  slice of the corridor, so the estimate is dragged back toward the entrance
  (under-travels: ~17–23 m for a corridor that is ~100 m).
- **"Twist":** with the axis dead-reckoning on the IMU prior, accumulated yaw
  wanders — ~150–190° of cumulative yaw travel over a corridor with ~2–3° net
  heading change, escalating to spins/divergence on bad runs.

No ground truth exists for this bag; true length is **~100 m** (inferred from the
best-tracking runs / site geometry — unconfirmed, an open item).

**Cross-cutting caveat — chaotic basins:** the tunnel's evolution is run-to-run
unstable; a single rep is never trustworthy. **n ≥ 3 is required** for any claim.

**Cross-cutting caveat — co-scheduling hazard:** dliio starves under CPU
contention ("IMU data does not cover the scan period" → scan skips → divergence).
It must have a free core or two; **never co-schedule** with RViz, dense-cloud bag
recording, or other heavy containers. Headless on an idle machine is clean (0
warnings, full ~21k poses); the same config under load diverges. This is the #1
reason live demos failed and is NOT an algorithm fault.

(RESPLE, the other estimator in this workspace, was investigated in parallel on
the same bag — degeneracy-gate sweep + `/odom` covariance inflation; see
`results/GATE_SWEEP_NOTES_2026-06-12.md` / `ODOM_COV_VERIFY_2026-06-12.md`. Same
problem, different codebase.)

## Finding 1 — the degeneracy gate alone does NOT stabilize the tunnel

`odom/gicp/degeneracyThreshRatio` sweep (reflectivity + plane reg, pw0.1, n=3),
classified by max\|pos\| excursion. Source: `results/dliio_tunnel_sweep/NOTES.md`.

| gate | r1 | r2 | r3 | verdict |
|---|---|---|---|---|
| 0.0 (off) | 22.9 OK | 16.6 OK | 1098 DIV | bounded 2/3 but **under-travels** (map-lock) |
| 0.005 | 11909 DIV | 9061 DIV | 116 OK | unstable 1/3 |
| 0.01 | 70.7 OK | 313 DIV | 2034 DIV | unstable 1/3 |
| 0.02 | 33174 DIV | 5e5 DIV | 85788 DIV | always diverges |

Non-monotonic and unstable. Higher threshold → worse (suppresses the axis, pose
dead-reckons → diverges). The gate correctly flags **1** degenerate direction (the
axis) on ~71–80% of scans; compute is cheap (~8–10 ms). IMU init is clean (ruled
out as the cause). The gate identifies the problem but cannot fix it alone.

## Finding 2 — reflectivity photometric weight is the "least-bad" LiDAR-only mitigation

`odom/gicp/photometricWeight` sweep (gate 0.005, reflectivity, plane):

| pw | result |
|---|---|
| 0.1 | DIV DIV OK |
| **0.3** | **OK OK OK, ~100 m** (103/101/95 m) — best |
| 0.5 | OK OK OK but less repeatable (one ~568 m) |
| 1.0 | OK OK DIV (over-trusts intensity) |

**pw0.3 + gate0.005** is the recommended LiDAR-only tunnel config
(`cfg/examples/ouster_tunnel.yaml`). BUT expanded sampling showed it is
**least-bad, not reliable**: ~60% of runs track ~100 m, ~40% map-lock to ~17 m,
and it diverges single-threaded or under load. A real fix needs more than tuning.

## Finding 3 — direct visual (camera) frame-to-frame term

Added a direct camera photometric residual (frame-to-frame: current vs previous
image, LiDAR depth), accumulated into the GICP Hessian before the degeneracy gate
so a camera-constrained axis re-opens the gate. Two design iterations on the gate
safety floor. Source: `results/dliio_visual_sweep/NOTES.md`.

- **Per-iteration gate floor:** visual OK rate 9/12 (75%) across weights vs 1/3
  LiDAR-only; w0.5 went 3/3 bounded **but over-travelled** (867/539 m vs ~100 m) —
  the per-iteration clamp let vision inflate the path over the run.
- **Per-scan cumulative budget (the fix):** bounds vision's *total* per-scan
  deviation from the IMU prior on a rescued axis. Produced the **best individual
  tracking seen** (w0.2 run at **99 m**, twist cut to ~160°) and tamed over-travel,
  but did not make any weight reliable (w0.2: 99/477 OK, 992 DIV).

Mechanism verified at runtime: healthy runs `deg=1, rescued=1, ~450 visual pts`;
failures are full collapses `deg=6, vpts→0` resembling the LiDAR-only baseline
instability — **not** visual runaway. Net: improves stability, trades divergence
(low weight) for over-travel (high weight); not a reliable fix.

## Finding 4 — frame-to-MAP camera term is measurement-starved on this rig

Upgraded to an *absolute* anchor: register the current image against per-keyframe
wall-texture references (graffiti). Source: `results/dliio_f2m_ab/NOTES.md`.

The term engages correctly (plumbing verified) **but sees only 2–64 map points
per scan with high residual RMS (0.4–0.6)** — too few to overcome the geometric
drag-back. f2f-only 2/3 OK vs f2f+f2m 1/3 OK (within chaotic-basin noise). Root
cause is **geometry, not math**: the narrow **sideways** camera + forward motion
means each keyframe sees a small slice of the LiDAR FOV, each wall patch is briefly
in view with a fast-changing viewing angle, so few landmarks persist and brightness
constancy breaks down. → motivated the LiDAR-image path (Finding 5).

## Finding 5 — COIN-LIO LiDAR intensity image solves the starvation (tuning open)

Same frame-to-map idea on the LiDAR **reflectivity** image (360° FOV, continuous
walls, range-normalized). Source: `results/dliio_lidarimg_ab/NOTES.md`.

- **Engagement solved:** **7,000–75,000 landmarks per scan** (vs the camera's
  2–64). Self-calibrated spherical model correct (el ±21° = OS-64 FOV, az 360°).
- **But raw weight is untunable** — the Hessian mass scales with the huge,
  variable point count: w0.3 over-travels/diverges, even w0.005 is 1/3 clean.
- **Fixes applied (committed):** count-normalization (`kLidarRefCount=1000`) so
  `weight` is count-independent + comparable to the other terms, plus an iteration
  stride (~4000 pts) to keep the node real-time. 21/21 unit tests
  (incl. spherical-projection numeric Jacobian).
- **Open:** re-tune the *normalized* weight (first cut ~0.05, untested clean) and
  re-run the A/B **headless on free cores** (every real-time attempt so far was
  starved by co-scheduling, not the algorithm).

## Finding 6 — per-scan IMU-consistency clamp (newest lever, efficacy open)

`odom/gicp/maxCorrTrans` / `maxCorrRot` (2026-06-16) bound the TOTAL per-scan GICP
correction (final pose vs the IMU-prior guess), orthogonal to the gate: the gate
holds the prior only on the eigen-directions it flags on the ~71-80% of scans it
fires, so on the scans it misses a map-lock jump runs away unbounded and seeds the
deg=6 collapse. Over a ~0.1 s scan the prior is high-confidence, so the *correction*
(not the motion — the prior contains the motion) is physically tiny; clamping it
caps the runaway without touching healthy motion. Mechanism is unit-tested
(MaxCorrectionClampBounds*, DisabledIsBitIdentical) and 0 = off by default.

**Efficacy not yet benched.** This is the most direct bound on both failure modes
(rot clamp -> twist; trans clamp -> drag-back), so it is now a PRIMARY sweep lever
— see `doc/TUNNEL_SWEEP_RUNBOOK.md §2`. The `ouster_tunnel.yaml` values (0.30 m /
0.05 rad) are loose placeholders; the rot cap especially wants tightening
(milliradian-scale per-scan yaw on a straight corridor).

## Finding 7 — soft degeneracy gate (smooths the gate boundary, efficacy open)

`odom/gicp/degeneracySoftness` (2026-06-17) ramps the gate's keep-fraction with a
smoothstep across a band `[thresh/(1+s), thresh*(1+s)]` instead of toggling each
eigen-direction between full-trust and full prior-hold at the threshold. It is a
third, orthogonal lever on the same failure: the clamp bounds correction
*magnitude* (Finding 6), the gate bounds correction *direction*, and this bounds
the *transition* — the scan-to-scan chatter when a marginally-observable axis
flickers across the threshold and toggles the prior-hold, a documented
divergence seed. It does not loosen a strongly-degenerate axis (eigenvalue
~0 << thresh still keeps ~0; unit-tested `SoftGateStillHoldsStronglyDegenerateAxis`),
and `s = 0` (default) is bit-identical to the original binary gate
(`SoftGateDisabledIsBitIdentical`). The keep-fraction is a pure free function
with its own tests (`SoftGate.*`).

**Efficacy not yet benched.** It pays off only when `Degenerate Directions` is
itself oscillating run-to-run (not when it sits stably at 1); sweep
**0.25 / 0.5 / 1.0** — see `doc/TUNNEL_SWEEP_RUNBOOK.md §2`. This is the first
piece of the larger "adaptive trust-weight" direction (replace hand-tuned
per-term weights with runtime-conditioned ones): the gate is the geometric
term's trust model, and softening it from binary to continuous is the
lowest-risk step toward that.

## Finding 8 — term mass-normalization (commensurable weights, the other step-1 half)

The photometric and camera terms accumulated into the shared Hessian **raw**, so
each term's mass scaled with however many points were valid that scan: the tuned
`weight` drifted with scan density and was not comparable across terms (the same
problem the LiDAR-image term already fixed internally with `kLidarRefCount`).
2026-06-17 generalizes that: `odom/gicp/{photometricRefCount, visualRefCount,
visualMapRefCount}` scale each term's total mass to a nominal reference count via
the shared `refCountScale(refcount, count)` helper (now also the LiDAR-image
term's path, bit-identical). All default **0 = OFF (raw, bit-identical** —
`PhotometricNormalizationDisabledIsBitIdentical`), and the count-independence
property is unit-tested directly (`RefCountScale.NormalizedMassIsCountIndependent`).
A new `Photometric Points` diagnostic exposes the per-scan photometric residual
count (previously untracked).

**Caveat:** enabling a RefCount **re-scales that term's effective weight**
(`weight·ref/count`), so a tuned weight — notably the tunnel config's
`photometricWeight: 0.3` — must be re-derived once on the bench (n≥3, chaotic
basin) before the normalized config is trusted. Left off by default so the
validated `ouster_tunnel.yaml` is unchanged. This is the **other half of step 1**
(with the soft gate): commensurable, density-stable weights are the precondition
for any cross-term adaptive weighting; this change makes the weights comparable
but does not yet adapt them at runtime.

## Finding 9 — per-term trust telemetry (read-only; what an adaptive policy keys on)

2026-06-17 surfaces a per-term count+quality pair in `/diagnostics` so the signals
an adaptive weighting policy would consume can be observed against a real bag
*before* anything auto-adjusts. Every term now reports both: geometric gets a
**trust margin** (`Geo Rot/Trans Trust Margin` = the block's weakest-axis
eigenvalue / the gate threshold; >1 trusted, <1 held, ~1 marginal — the chatter
zone, -1 when the gate is off); photometric gets `Photometric Points` +
`Photometric Residual RMS` (the previously-missing pair); the camera and
LiDAR-image terms already reported count + RMS. All read-only — the margins are
captured inside the gate's existing eigendecomposition and the RMS from an
unweighted residual tally, neither touching H/b (the solve is bit-identical,
covered by the existing alignment + bit-identical tests). This closes the
observability half of the adaptive-trust groundwork: gate softened (step 2),
weights commensurable (step 1), and now the trust signals visible. The actual
runtime adaptation (mapping these signals to per-term/per-axis weights) is the
next, genuinely-new-behavior step and is not yet done.

## Sensor-robustness note — intensity<->reflectivity fallback

The tunnel config uses `photometricChannel: reflectivity` (calibrated, range-
normalized — the right LiDAR-only signal here). As of 2026-06 the node falls back
to range-corrected `intensity` automatically if a cloud lacks the `reflectivity`
field (and the COIN-LIO LiDAR-image term builds from `intensity` instead of an
all-zero image), so this config no longer silently dies on a non-Ouster sensor or
an Ouster topic recorded without reflectivity. The fallback is logged once.

## Recommended configs

| goal | config |
|---|---|
| LiDAR-only tunnel (deploy today, expect ~60% track) | `cfg/examples/ouster_tunnel.yaml` (reflectivity + plane + gate0.005 + pw0.3) |
| + camera anchor (built, weak on this rig) | + `cfg/examples/ouster_tunnel_visual.yaml` |
| + LiDAR intensity-image anchor (best engagement; weight WIP) | + `cfg/examples/ouster_tunnel_lidarimg.yaml` |

All photometric terms are **off by default**; the LiDAR-inertial path is
bit-identical when disabled.

## Open items / next steps

> The concrete, reproducible procedure for the next series (branch, headless/free-core, sim-time+clock, the weight sweep, occlusion-tol sanity check, n>=5, metric, and exact /diagnostics keys to read) is in **`doc/TUNNEL_SWEEP_RUNBOOK.md`**.


1. **Re-tune the normalized LiDAR-image weight** (~0.05) and validate n≥5 headless
   on free cores — the most promising remaining lever.
2. **Confirm the camera↔LiDAR extrinsic** (`extrinsics/cam2lidar`) and **which
   camera serial is the recorded `/lucid_camera_1`** (two serials exist:
   242600094 / 243300379; the bag's embedded intrinsics match neither — see the
   workspace memory). Gates absolute-accuracy interpretation of the camera term.
   Note: the provided `cam2lidar` R had a typo `0.30`→`0.030` (corrected to be
   orthonormal).
3. **Confirm the true tunnel length** (~100 m assumed; no GT).
4. **Early-startup collapse basin** — most divergences seed early; investigate
   protecting the startup window.
5. **Illumination/brightness-constancy** compensation (affine a·I+b) for the
   photometric terms over viewpoint/range change.
6. **Reproduce headless on a quiet machine** — the chaotic + load-sensitive
   behavior means results must be gathered with free cores and n≥3.
