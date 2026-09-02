# Direct visual (camera) photometric term

Optional LiDAR-visual-inertial extension to dliio that constrains the
along-axis translation a geometrically self-similar environment (a smooth
tunnel / conduit) leaves **LiDAR-degenerate**. Off by default; the LiDAR-
inertial path is bit-identical when disabled.

## Why

In a smooth tunnel, scan-to-map GICP is unobservable along the tunnel axis:
there is no geometric feature to lock forward translation onto. dliio's
degeneracy gate detects this and holds the IMU prior on that axis, so the pose
dead-reckons and slowly drifts — on the 06042026 Ouster bag this manifests as a
"twist" (yaw wander of ~150–190° over a corridor with ~2–3° net heading change)
and, on bad runs, divergence. The reflectivity photometric term mitigates this
but is "least-bad", not reliable (see `results/dliio_tunnel_sweep` in the test
harness). **Camera texture is the missing constraint**: it directly observes the
along-axis motion the LiDAR cannot.

dliio already had a photometric GICP residual (on LiDAR reflectivity) and the
degeneracy gate; this term reuses both rather than bolting on a separate VIO.

## Design

**Frame-to-frame direct image alignment using LiDAR depth** (FAST-LIVO-style
"direct", no feature extraction):

- Each current-scan world point `p_w` (dliio's source cloud is already in the
  world frame) has a brightness in the **current** camera image — a fixed,
  pose-independent reference `I_ref = I_cur(π(T_cw_cur · p_w))` (the point's
  position relative to the current camera is fixed by the rigid extrinsic).
- Its pose-corrected estimate `x' = trans · p_w` is projected into the
  **previous** camera image: `I_mov = I_prev(π(T_cw_prev · x'))`. This depends
  on the pose correction `trans` — the source of observability.
- Residual `r = I_mov − I_ref` (normalized brightness, /255).

Reference = current frame, warp target = previous frame; no persistent visual
map and no per-point storage (the previous image + pose are carried scan to
scan). This is the minimal form that supplies the per-scan along-axis
constraint the gate needs.

### Jacobian (left-perturbation, matches the geometric term)

With `T_cw_prev = [R_cw | t_cw]`, `P_c = R_cw·x' + t_cw`, pinhole projection
`π`, and image gradient `∇I` (1×2):

```
G    = ∇I · dπ/dP_c · R_cw                       (1×3)
J    = [ −G·skew(x') | G ]                        (1×6, [rot|trans])
H   += w·JᵀJ ,   b += w·Jᵀr ,   cost += w·r²      (Huber-weighted)
```

Note the translation block is `+G`, **opposite** the LiDAR photometric term's
`−gᵀ` (its gradient is a 3-D spatial gradient of a fixed target field; here the
moving quantity is `I_prev` sampled at the projected, pose-dependent pixel). The
finite-difference and descent unit tests guard this sign — it is the single most
error-prone part.

### Where it hooks in

`NanoGICP::accumulateVisualResidual` is called inside `computeTransformation`
**between `linearize()` and the degeneracy gate**, accumulating into the *same*
6×6 Hessian as the geometric/reflectivity terms. It iterates the source cloud
(world-frame points with LiDAR depth), bilinearly samples `I_prev` and its
central-difference gradient, culls behind-camera / out-of-FOV / low-gradient
points, and applies a Huber weight. OpenMP thread-private double accumulation,
matching `linearize()`.

## Degeneracy-gate safety floor

A camera term that quiets the gate also *removes its protection*: if the visual
constraint is wrong (extrinsic error, illumination), it can then drive the pose
along the now-"observable" axis. So the gate was hardened:

1. **Judge from geometry alone.** The gate eigen-analyzes the **geometric**
   Hessian `H_geo` (snapshot before the visual term), so visual mass can never
   mask LiDAR degeneracy. A direction is "weak" iff its LiDAR eigenvalue is
   below `degeneracyThreshRatio · λ_max`.
2. **Rescue criterion.** On a weak axis, the visual term is allowed to drive
   motion only if it actually *stiffened* that axis (combined Rayleigh quotient
   `vᵀ H v > thresh`). Axes vision does not rescue stay **held to the IMU
   prior** — the original LiDAR-only behavior.
3. **Per-scan budget (cumulative cap).** Even on a rescued axis, vision's total
   deviation from the IMU prior is bounded to `gateMaxStepTrans` /
   `gateMaxStepRot` **per scan** (summed across LM iterations), so a wrong or
   scale-biased constraint can neither diverge nor inflate the path. (An earlier
   per-iteration clamp bounded single-step blow-ups but allowed slow over-travel
   over a long run.)

When the visual term is disabled, the gate is bit-identical to the original
(rescue guarded by `visual_enabled_`, `H_geo == H`).

## Parameters

Algorithm (`cfg/params.yaml`, all default off):

| key | default | meaning |
|---|---|---|
| `odom/visual/enabled` | `false` | enable the term + camera subscription |
| `odom/visual/weight` | `0.0` | weight relative to the geometric GICP term |
| `odom/visual/huberDelta` | `0.05` | Huber on the normalized residual (≤0 off) |
| `odom/visual/maxTimeDiff` | `0.05` | max \|image−scan\| stamp gap to pair [s] |
| `odom/visual/gateMaxStepTrans` | `0.3` | per-scan translation budget on a rescued axis [m] |
| `odom/visual/gateMaxStepRot` | `0.05` | per-scan rotation budget on a rescued axis [rad] |

Per-rig (`cfg/dlio.yaml`):

| key | meaning |
|---|---|
| `extrinsics/cam2lidar/{t,R}` | camera→lidar transform (same storage convention as `baselink2lidar`: R,t maps a camera point into the lidar frame) |
| `camera/intrinsics` | `[fx, fy, cx, cy]` |
| `camera/distortion` | plumb_bob `[k1,k2,p1,p2,k3]` |

The image topic remaps to `camera` (launch arg `camera_topic`). See
`cfg/examples/ouster_tunnel_visual.yaml` for a full tunnel overlay (stack it
over `ouster_tunnel.yaml`).

## Frame-to-MAP camera term (absolute anchor)

The frame-to-frame term above constrains *relative* motion/yaw but provides no
absolute anchor — so in a self-similar tunnel the geometric GICP still drags the
pose back toward the entrance. The frame-to-map term anchors absolute position
to map landmarks (wall texture/graffiti):

- **Reference data** (`nano_gicp::VisualRef`): each map point carries a reference
  brightness sampled from the keyframe that first observed it, plus its position
  in that keyframe's camera frame (`p_kf_cam`, a fixed viewing ray for viewpoint
  gating). Sampled at keyframe creation (`sampleKeyframeVisualRefs`), threaded
  keyframe→submap→target exactly like `keyframe_normals` (and copied in
  `shareTargetDataFrom`). `p_kf_cam` is camera-frame, so it survives the world
  re-transform in `buildKeyframesAndSubmap` untouched.
- **Residual:** the FIXED map point is projected into the CURRENT (trans-dependent)
  camera, `P_c = (T_prior·baselink2lidar·cam2lidar)⁻¹·trans⁻¹·p_w`,
  `r = I_cur(π(P_c)) − I_ref`. Because `trans` appears INVERTED, the Jacobian is
  **`J = [ +G·skew(p_w) | −G ]`** — *opposite* the frame-to-frame term's
  `[−G·skew(x)|+G]` (guarded by `test_visual_residual`).
- **Gate budget:** the absolute anchor gets a separate, larger per-scan rescue
  budget (`odom/visual/map/gateMaxStep*`, default 1.0 m) — it can undo a real
  drag-back, but is still bounded. Gate still judges degeneracy from `H_geo`.
- **Robustness:** viewpoint-ray gating (`viewAngleMax`) + Huber.

Keys (algorithm): `odom/visual/map/{enabled,weight,gateMaxStepTrans,gateMaxStepRot,viewAngleMax}`
(all default off). Diagnostics: `Visual Map Active/Points/RMS`.

## COIN-LIO LiDAR intensity-image term (frame-to-MAP)

The camera anchor was measurement-starved on this rig (narrow sideways FOV). The
LiDAR sees the walls 360° with consistent, range-normalized reflectivity, so it
has far more persistent landmarks. This term is the same frame-to-map idea on the
LiDAR reflectivity image:

- **Image:** the organized 64×1024 reflectivity image is snapshotted in
  `getScanFromROS` BEFORE NaN removal (the bag is organized; do **not** organize
  the registration cloud — GICP needs the NaN-removed/voxelized one). A spherical
  model `col=(atan2(Y,X)−az_b)/az_a`, `row=(elev−el_b)/el_a` is **self-calibrated**
  from the scan (least-squares az-per-col and el-per-row), so no external beam LUT.
- **Reference:** each map point's own `reflectivity` field — no per-point ref
  threading needed (unlike the camera, which needed `VisualRef`).
- **Residual/Jacobian:** `accumulateLidarMapResidual` projects the FIXED map point
  into the current reflectivity image via `π_L`; same trans-inverse Jacobian shape
  `[+G·skew(p_w)|−G]` with the spherical `dπ_L/dP_l` instead of the pinhole.
- **Count-normalization (critical):** the submap is tens of thousands of points,
  so a raw weight scales the Hessian mass with the (huge, variable) point count
  and is untunable (w=0.005 already over-travels). The term's mass is normalized
  to a nominal reference count (`kLidarRefCount=1000`) so `weight` is comparable
  to the camera/geometric terms and stable run-to-run. The iteration is also
  strided (cap ~4000 points) to bound per-scan cost.
- Reuses the gate + absolute-anchor budget; can rescue the axis with no camera.

Keys: `odom/lidar_image/{enabled,weight}`; diagnostics `Lidar Map Active/Points/RMS`.
Overlay: `cfg/examples/ouster_tunnel_lidarimg.yaml`.

## Tests

`test/test_visual_residual.cpp` (runs in CI, no hardware/extrinsic needed):
residual→0 at truth; analytic-vs-finite-difference Jacobian; Gauss-Newton
descent (sign guard); Hessian stiffness on a targeted axis; disabled = no-op —
for BOTH the frame-to-frame and frame-to-map terms (the f2m tests guard the
trans-inverse sign flip), and the COIN-LIO LiDAR term (spherical-projection
numeric Jacobian). Full suite 21/21.

## Status / results (06042026 tunnel, n=3, see test-harness notes)

> **2026-06-22 update:** per-gate instrumentation confirmed the frame-to-frame
> term engages and produces ~160 in-bounds visual points/scan when the pose is
> healthy (pairing, intrinsics, extrinsic, world-frame source all verified). The
> `Visual Points = 0` reported on 2026-06-18 was a measurement artifact, not a
> bug: the points collapse late-run at the *reference* projection as the
> voxelised source cloud shrinks and the narrow side-camera FOV empties
> (`behind_mov = oob_mov = grad = 0` all run). Highest-leverage de-starve =
> project the dense deskewed cloud instead of the voxelised `input_`. See
> `FINDINGS_2026-06-22.md`.

The term is correctly engineered and the mechanism is verified at runtime
(healthy runs: `deg=1, rescued=1, ~450 visual points`; failures are full
collapses `deg=6, vpts→0` resembling the LiDAR-only baseline instability — not
visual runaway). It produced the **best tunnel tracking observed** — individual
runs at **~80–100 m** (the true tunnel length) with the twist reduced to ~160°,
and improved the bounded-run rate over LiDAR-only.

However it does **not** make this tunnel reliable: run-to-run variance is severe
and divergence persists across weights. Two residual problems dominate and are
beyond gate tuning: (a) **over-travel / scale** at higher weight, pointing at
**extrinsic accuracy** (and an unresolved camera-serial↔topic question); and
(b) an **early-startup collapse basin** that seeds most divergences before
vision can hold the axis. Recommended next steps: verify/refine `cam2lidar`,
protect the startup window, illumination compensation, n≥5 confirmation.

Treat this as a working, well-guarded mitigation and research scaffold — not a
turnkey tunnel fix.

### Frame-to-map camera A/B (2026-06-12, n=3)

The frame-to-map term engages correctly (refs thread through; diverged runs show
the whole submap flooding the frame) but is **measurement-starved on this rig**:
in normal operation it sees only **2–64 map points** with **high residual RMS
(0.4–0.6)**, so it cannot out-vote the geometric drag-back and does not make the
tunnel reliable (f2f-only 2/3 OK vs f2f+f2m 1/3 OK — within chaotic-basin noise).
Root cause is geometry, not the math: a **narrow, sideways camera + forward
motion** means each keyframe sees only a small slice of the LiDAR points and each
wall patch (graffiti) is briefly in view with a fast-changing viewing angle, so
few landmarks persist and brightness constancy breaks down.

**Implication:** the LiDAR intensity image (COIN-LIO path) is the better-suited
absolute anchor for this rig — 360° FOV, continuous wall coverage, consistent
range-normalized viewpoint — and reuses this same frame-to-map backbone (swap the
camera image for the organized 64×1024 reflectivity image). The camera frame-to-map
work built and validated the machinery and identified the LiDAR as the stronger anchor.

### COIN-LIO LiDAR A/B (2026-06-12)

The LiDAR term solved the starvation — it engages with **7k–75k landmarks per
scan** (vs the camera's 2–64), and the self-calibrated spherical model is correct
(±21° elevation, 360° azimuth). But a **raw** weight scales with that huge count
and is untunable: w0.3 over-travels/diverges, and even w0.005 is 1/3 clean. The
**count-normalization + iteration stride** (added after) make `weight` stable and
the term real-time; re-tuning the normalized weight is the open Phase-2 step
(first cut ~0.05). NOTE: real-time replay on a *loaded/co-scheduled* machine
starves the node (IMU-drop scan-skips) — the documented dliio hazard; headless on
free cores is clean (0 warnings, full 21k poses). See
`results/dliio_lidarimg_ab/NOTES.md`.

## Literature cross-reference (the three 2026-06-13 fixes)

The review fixes on branch `claude/visual-term-fixes` were each checked against
the primary sources; all three align with established practice.

- **Degeneracy-gate damping leak (rescue judged on the un-damped information
  matrix).** Zhang, Kaess & Singh, *On Degeneracy of Optimization-based State
  Estimation* (ICRA 2016, [paper](https://frc.ri.cmu.edu/~zhangji/publications/ICRA_2016.pdf))
  remap the solution using the **information matrix's** conditioning. Mixing the
  LM damping term `λI` into the block the rescue test reads contaminates exactly
  that conditioning judgement; snapshotting the combined geometric+photometric
  Hessian *before* `+= λ` keeps the test faithful to the paper.

- **Occlusion / depth-consistency culling of projected map points.** This is the
  one the literature is most explicit about. FAST-LIVO
  ([arXiv:2203.00893](https://arxiv.org/abs/2203.00893)) and FAST-LIVO2
  ([arXiv:2408.14035](https://arxiv.org/abs/2408.14035)) add "a novel outlier
  rejection method … to reject unstable map points that lie on edges or are
  **occluded in the image view** … identifies occluded and **depth-discontinuous**
  visual map points." Our per-scan range image + `|range_p − range_pixel| > tol`
  reject is the direct LiDAR-image analogue. COIN-LIO
  ([arXiv:2310.01235](https://arxiv.org/abs/2310.01235), [code](https://github.com/ethz-asl/COIN-LIO))
  likewise operates on a structured intensity image with brightness/range
  filtering rather than blindly projecting a whole world map. Projecting the
  entire corridor submap with **no** visibility test (the original code) is the
  one thing all of these explicitly avoid — consistent with the high frame-to-map
  RMS in `TUNNEL_FINDINGS.md`.

- **Per-beam (non-uniform) elevation instead of a linear `el(row)`.** Ouster OS
  sensors ship Uniform, **Gradient**, and **Below-Horizon** beam configurations
  with per-beam `beam_altitude_angles` in the metadata
  ([Ouster sensor docs](https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html)).
  COIN-LIO requires Ouster specifically *"as we use the calibration in the
  metadata file for the image projection model"* and a per-sensor column-shift
  calibration. A single linear elevation slope is therefore wrong for two of the
  three modes; our self-calibrated per-row elevation LUT (local-slope Jacobian)
  empirically recovers what COIN-LIO reads from the metadata.

**Where this fork still differs from the references (open, not bugs):** COIN-LIO
and FAST-LIVO attach patches to *sparse, selected* map points and pick patches
**complementary to the degenerate directions**, whereas this term accumulates an
isotropic residual over a strided subset and relies on the gate for
complementarity (REVIEW.md item #8, condition-scaled weighting). None of the
terms here do affine brightness-constancy compensation, which FAST-LIVO/COIN-LIO
do (TUNNEL_FINDINGS open item 5). The occlusion + per-beam fixes close the two
gaps that were outright deviations; patch selection and photometric calibration
remain genuine, literature-backed next steps.
