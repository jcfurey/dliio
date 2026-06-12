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

## Tests

`test/test_visual_residual.cpp` (runs in CI, no hardware/extrinsic needed):
residual→0 at truth; analytic-vs-finite-difference Jacobian; Gauss-Newton
descent (sign guard); Hessian stiffness on a targeted axis; disabled = no-op.
Full suite 15/15.

## Status / results (06042026 tunnel, n=3, see test-harness notes)

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
