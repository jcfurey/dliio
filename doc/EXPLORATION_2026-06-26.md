# Exploration 2026-06-26 — LODESTAR/ESKF observer slice + novel degeneracy directions

Two things: (1) make a *tractable* start on the LODESTAR/ESKF direction that
`doc/IMPLEMENTATION_NOTES.md` §4 deferred as an architectural misfit, and
(2) explore novel solutions to the 06042026 tunnel degeneracy beyond the
auxiliary-observation levers that all came back negative.

**Where we are.** The degenerate axis is a weak yaw rotation (and the along-tunnel
translation). Negative levers: COIN-LIO intensity, condScale, near-IR, dense
visual, GenZ blend — all *dilute off* the unobservable axis. Positive: the X-ICP
ternary gate (validated, *prevents* the deg=6 collapse) and the governor (bounds
the held-prior runaway after the fact). The recurring failure mode across the
negatives is **dilution**: a term's mass lands on the dominant wall geometry, not
the weak axis.

---

## Part 1 — LODESTAR / ESKF

### 1.1 The architectural reality (unchanged)

LODESTAR (Lee, Marsim, Myung, RA-L 2025) gives degenerate "fixed" states **zero
Kalman gain** and anchors the active states through the **cross-covariance** of a
Joseph-form update — it needs a covariance-propagating filter. DLIO's back-end is
a **contracting hierarchical observer** (Lopez 2023): `updateState()` applies
*scalar* gains (`Kp, Kv, Kq, Kab, Kgb`) to the LiDAR pose error `err = pin − p̂`.
There is no state covariance `P` to host a Schmidt-Kalman update.

### 1.2 The tractable slice — implemented here (default-off)

LODESTAR's *intent* transfers without a filter rewrite. The observer already has a
gain on the correction; the degeneracy gate already exposes the held world-frame
eigen-directions (`lastDegen{Trans,Rot}Dirs()`). So: **on the held axes, attenuate
the observer's correction** so the state rides the IMU-propagated prior there
instead of absorbing registration noise — the contracting-observer analogue of
LODESTAR's reduced-gain fixed state.

- `include/dlio/degeneracy_observer.h` — pure `attenuateAlongHeldAxes(e, dirs,
  gain)`: scales the component of `e` along each held axis to `gain · component`.
  `gain == 1` → unchanged (OFF/bit-identical); `gain == 0` → axis fully frozen to
  the prior; `0 < gain < 1` → partial trust. Unit-tested
  (`test/test_degeneracy_observer.cpp`, 7 cases incl. oblique + multi-axis).
- `updateState()` applies it to `err` (drives **position, velocity, and accel
  bias** via `err_body`) along held translation dirs, and to `qcorr.vec()` (drives
  **orientation**) along held rotation dirs. Guarded by `geo_degen_obs_gain_ < 1`.
- Control: `odom/geo/degenObsGain` (default `1.0` → bit-identical). Live-tunable
  family alongside the other `geo/*` gains.

**Slice vs. governor (complementary, not redundant).** The governor clamps the
*output pose* (`this->T`) to a physical per-scan cap *after* the solve and relies
on the observer to pull velocity back. This slice acts *inside* the observer,
*earlier*: it stops the unreliable correction from entering the velocity/bias
states in the first place, so the runaway is *attenuated at the source* rather
than *bounded after the fact*. Pairs with the governor's covariance inflation
(same held axes marked untrusted downstream).

**What the slice deliberately does NOT do** (and why it's a *slice*, not the full
thing): no gyro-bias attenuation yet (frame-tangled: `qe` is body-frame, held dirs
world — a documented refinement); no covariance propagation, so no LODESTAR
*cross-covariance* anchoring and no automatic **snap-back** when the axis becomes
observable again (the slice simply trusts the correction again once the gate
stops flagging the axis).

### 1.3 What a full ESKF would add — and the cost

A faithful port replaces the observer with an ESKF/UKF carrying `P`, then applies
LODESTAR's fixed/active partition (zero gain + cross-covariance anchoring). Gains:
principled uncertainty, cross-axis coupling, and graceful snap-back. Cost: a
**back-end rewrite** of `propagate/updateState`, full re-tuning, and re-validation
of the *entire* filter (not a default-off feature) — high risk on a working
estimator, unvalidatable without the bag. **Recommendation:** validate the cheap
gain-scheduling slice first; only commit to the ESKF if per-axis gain scheduling
proves insufficient (e.g. if snap-back or cross-axis coupling turns out to matter
on the bag). ALIVE-LIO (arXiv:2604.02706) is the reference if/when DLIO moves to
an ESKF.

### 1.4 How to validate the slice

Bag A/B on the 06042026 tunnel, n ≥ 24: sweep `degenObsGain ∈ {1.0, 0.5, 0.0}`,
each with the governor and X-ICP gate on/off (2×2×3). Hypothesis: the slice
reduces the velocity/bias coasting that the governor only bounds *after* it has
grown, so X-ICP (prevent) + slice (don't ingest noise) + governor (bound) stack.
Watch for over-freezing on legitimately-marginal motion (the gate's held set is
conservative, so `gain = 0` should be safe, but `0.5` is the cautious first try).

> **AMENDED by the 2026-06-26 A/B (`doc/FINDINGS_2026-06-26.md`).** The "`gain = 0`
> should be safe" hypothesis was **WRONG**: `degenObsGain = 0.0` is *catastrophic*
> on the bag (5/5 DIV, monotonic) — a **full freeze removes the observer's only
> correction on the held axis, leaving the IMU-propagated prior to dead-reckon
> unbounded** (precisely the runaway the governor exists to clamp). The slice's
> value is a **partial gain**: `0.5` is the sweet spot (crash-free, DIV 4/24 vs the
> 8/24 baseline — halves it, but insufficient *alone*, p=0.16). So the slice is a
> crash-free *complement* to a hard bounding lever, not a standalone fix, and
> `degenObsGain` must stay strictly between 0 and 1 (never 0). Open follow-ups:
> a finer gain sweep (can a value reach the governor's 0/24 without the governor?)
> and the X-ICP + governor + slice stack. The governor + X-ICP combo is the
> decisive divergence-preventer (0/24 genuine DIV); its only blocker is the
> photometric-loop `std::out_of_range` crash, root-caused and fixed separately.

---

## Part 2 — Novel solutions (ranked)

Each: mechanism · why it fits *this* degeneracy · feasibility · status. The lens
is **dilution-resistance** — the property every negative lever lacked.

### #1 — Saliency-weighted geometric point selection (anti-dilution) ★ top pick
**Mechanism.** Don't add a term — **reweight the geometric GICP itself**.
Up-weight the rare *salient / non-planar* points (tunnel ribs, bolts, junction
boxes, pipe flanges, conduit hangers) that actually constrain the along-axis DOF;
down-weight the dominant, redundant wall points that swamp it. Saliency from the
per-point covariance eigen-structure already computed in `calculate_covariances`
(low planarity / high curvature → salient), or a smoothness score.
**Fit.** Attacks the *root* failure mode (dilution) directly, at the term that
actually has mass on the weak axis when those features exist. No new modality.
**Feasibility.** Medium — saliency is computable from existing covariances; apply
as a per-correspondence weight on `M` (same hook as the GenZ blend), default-off.
**Grounded in.** DAMM-LOAM (normal-map point classification + observability-aware
weighting from translational-Hessian eigenvalues), "Small but Mighty" feature
enhancement. **Status.** Strong candidate; composes with X-ICP. Prototype next.

### #2 — Range/intensity-image direction-separated flow ★ second pick
**Mechanism.** Register consecutive **cylindrical range (or intensity) images**
frame-to-frame; along-tunnel motion appears as image *flow* even when the 3D
geometry is rank-deficient. Fuse the flow estimate **only into the degenerate
direction** (direction-separated fusion), so it can't corrupt the well-observed
axes.
**Fit.** Directly observes the unobservable DOF from a different signal channel;
the "fuse only into the degenerate direction" framing is exactly the anti-dilution
guard the scalar intensity term lacked.
**Feasibility.** Medium-high — the `lidar_image` projection infra already exists;
add a frame-to-frame photometric/flow residual gated to `lastDegenDirs`.
**Grounded in.** LOFF (LiDAR + optical-flow fusion into the degenerate direction),
COIN-LIO (intensity image), ECTLO (range-image odometry).
**Status.** Strong; reuses existing infra. Prototype after #1.

### #3 — LODESTAR observer slice (implemented in Part 1)
Estimator-level complement to the governor; cheap, default-off, validate on the
bag. See Part 1.

### #4 — Observability-gated IMU preintegration + bias freeze
**Mechanism.** Extend the slice: when an axis is held, **freeze its bias state**
(the slice freezes accel bias via `err`; add gyro bias) and lean on a tight
preintegrated IMU prior; detect ZUPT/ZARU (zero-velocity / zero-angular-rate)
epochs to recalibrate bias so it can't drift into the runaway.
**Fit.** Targets the *mechanism* the governor named — accel/gyro bias
double-integration. **Feasibility.** High (extends Part 1). **Grounded in.** the
governor rationale; ALIVE-LIO. **Status.** Natural follow-on to the slice.

### #5 — Periodic-structure ruler
**Mechanism.** Tunnels have regularly-spaced ribs/segments/lights; autocorrelate
the range/intensity profile *along the axis* to recover the spatial period and use
peak-to-peak as a metric **ruler** bounding along-axis displacement per scan.
**Fit.** Turns the very self-similarity that breaks place-recognition into a
constraint. **Feasibility.** Medium; brittle to irregular spacing / junctions.
**Status.** Research-y; revisit if #1/#2 stall.

### #6 — FMCW / Doppler velocity (sensor-gated)
**Mechanism.** An FMCW LiDAR measures per-point radial velocity → directly
observes along-axis motion. **Fit.** Decisive, but the 06042026 rig is a ToF
Ouster. **Status.** Hardware roadmap note, not actionable now.

### Recommendation
Prototype **#1 (saliency reweighting)** and **#2 (direction-separated range-image
flow)** next — both are *dilution-resistant* and target the actual unobservable
axis, which is exactly what every negative auxiliary-term lever was not. Validate
the **#3 observer slice** (this PR) on the bag in parallel since it's already
default-off and cheap. #4 is a low-cost extension of #3.

---

## Sources
- LODESTAR — Lee, Marsim, Myung, RA-L 2025 — [arXiv:2511.09142](https://arxiv.org/abs/2511.09142)
- Lopez — A Contracting Hierarchical Observer for Pose-Inertial Fusion, 2023 — [arXiv:2303.02777](https://arxiv.org/abs/2303.02777)
- DAMM-LOAM — Degeneracy-Aware Multi-Metric LiDAR Odometry & Mapping, 2025 — [arXiv:2510.13287](https://arxiv.org/html/2510.13287v1)
- D²-LIO — Directional-Degeneracy-Aware LiDAR-IMU Odometry, 2025 — [arXiv:2508.14355](https://arxiv.org/pdf/2508.14355)
- LOFF — LiDAR and Optical Flow Fusion Odometry, Drones 2024 — [MDPI 2504-446X/8/8/411](https://www.mdpi.com/2504-446X/8/8/411)
- ECTLO — Effective Continuous-Time Odometry Using Range Image, 2022 — [arXiv:2206.08517](https://arxiv.org/pdf/2206.08517)
- COIN-LIO — Pfreundschuh et al., ICRA 2024 — [arXiv:2310.01235](https://arxiv.org/abs/2310.01235)
- ALIVE-LIO — Degeneracy-Aware Learned Inertial Velocity for ESKF LIO — [arXiv:2604.02706](https://arxiv.org/abs/2604.02706)
- "Small but Mighty" — Lightweight Feature Enhancement for LiDAR Odometry, Remote Sensing 2025 — [MDPI 2072-4292/17/15/2656](https://www.mdpi.com/2072-4292/17/15/2656)
