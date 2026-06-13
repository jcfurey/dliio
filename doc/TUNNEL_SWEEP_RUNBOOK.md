# Tunnel sweep runbook — next playback series

Exact, reproducible procedure for the next 06042026 (and similar) bag series,
incorporating the 2026-06-13 fixes (gate-damping snapshot, frame-to-map
occlusion cull, OS non-uniform elevation LUT). Companion to
`doc/TUNNEL_FINDINGS.md` (the journey) and `doc/VISUAL_TERM.md` (the design).

## 0. Preconditions (a run is uninterpretable without these)

- [ ] **Code under test = the fixes branch** (`claude/visual-term-fixes` /
      whatever it merged to). The pre-fix build floods the frame-to-map term
      with occluded points; results won't reflect the cull. `colcon build` +
      `colcon test` (23/23) before recording anything.
- [ ] **Free cores.** dliio starves under contention ("IMU data does not cover
      the scan period" -> scan skip -> divergence). Run **headless** (no RViz,
      no dense-cloud recording on the same cores). Pin if needed:
      `taskset -c 0-5 ros2 launch ...` and keep the bag player off those cores.
- [ ] **Sim time + clock.** Default `use_sim_time:=false`; for playback launch
      with `use_sim_time:=true` and play with `--clock`:
      `ros2 bag play <bag> --clock`.
- [ ] **n >= 5 per config.** The tunnel is a chaotic basin; a single rep is
      never trustworthy. Use the SAME seed/order each rep (fresh node per rep).
- [ ] **Success metric fixed up front** (no GT exists). Record per rep:
      bounded vs diverged; final `Distance Traveled (m)` (true length ~100 m,
      unconfirmed); cumulative yaw / "twist" magnitude; max |pos| excursion.

## 0b. Visualization without starving the estimator

Do the scored runs HEADLESS and record only the light outputs, then visualize
later (the dense deskewed cloud and the full map are the heavy topics that
starve the node):

```
ros2 launch direct_lidar_inertial_odometry record_outputs.launch.py out:=run1_outputs
# ... later, offline:
ros2 bag play run1_outputs --clock   # + RViz / Foxglove
```

Live viz is now safe-by-default too: the deskewed-cloud, keyframe-cloud, and map
publishers skip their work when nobody is subscribed, and the map republishes on
a low-rate latched timer (`map/publishRate`, default 1 Hz) rather than per
keyframe. Prefer Foxglove via `foxglove_bridge` with the browser on another
machine; its Diagnostics panel reads `/diagnostics` directly. If a run looks bad,
check `CPU Starved` / `Compute Overruns` FIRST -- a starved run is not an
algorithm failure.

## 1. What to run

LiDAR-image series (no camera; not blocked by the unresolved cam2lidar/serial):

```
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  pointcloud_topic:=/ouster/points imu_topic:=/ouster/imu use_sim_time:=true \
  --ros-args \
    --params-file <pkg>/cfg/examples/ouster_tunnel.yaml \
    --params-file <pkg>/cfg/examples/ouster_tunnel_lidarimg.yaml
```

Baselines to keep in the matrix (so the fixes are attributable):
- **B0** LiDAR-only: `ouster_tunnel.yaml` (reflectivity + plane + gate0.005 + pw0.3).
- **B1** + LiDAR-image anchor at the swept weight (this overlay).

## 2. The sweep (primary lever)

`odom/lidar_image/weight` over **0.02 / 0.05 / 0.1** (count-normalized, so these
are meaningful absolute numbers now). Everything else from the overlay.

Secondary, only if needed:
- Occlusion tolerance `rangeAbsTol / rangeRelTol` (0.5 / 0.1 default). Sanity
  pass: if `Lidar Map Points` collapses to hundreds it's over-culling -> loosen;
  if `Lidar Map RMS` stays ~0.4-0.6 it's not culling enough -> tighten.
- `degeneracyThreshRatio` (0.005) and `photometricWeight` (0.3) are the B0
  tunnel config; leave fixed unless B1 regresses B0.

## 3. What to log and read (exact /diagnostics keys)

`ros2 topic echo /diagnostics` (or record it). Per scan, watch:

| key | healthy | failure signature |
|---|---|---|
| `Degenerate Directions (current)` | 1 (the axis) | 6 = full collapse |
| `Lidar Map Points` | thousands | -> 0 (term starved / all occluded out) |
| `Lidar Map RMS` | lower than pre-fix 0.4-0.6 | high + rising |
| `Visual Rescued Axes` | 1 when deg=1 | 0 while deg>=1 (gate not rescuing) |
| `Distance Traveled (m)` | climbs to ~100 | stalls ~17-23 (map-lock) or runs away |
| `Max Computation Time (ms)` | < scan period | spikes = starvation (see preconditions) |
| `Realtime Factor` | < 1.0 | > 1.0 = can't keep real time |
| `CPU Starved` | 0 | 1 = this scan overran the period |
| `Compute Overruns (cumulative)` | 0 / flat | climbing = CPU-bound run, NOT divergence |
| `Scans Dropped est (cumulative)` | 0 | > 0 = transport dropped scans (contention) |

The fix-specific things to confirm engaged:
- Startup log `LiDAR intensity image NxM; spherical model el=...` -> projection
  self-cal ran. If the LUT was accepted it supersedes the linear el internally
  (no separate log; verify indirectly via lower RMS than the linear build).
- `Lidar Map Points` should be **lower than the pre-fix build at the same scan**
  (occlusion cull removing far-side points) while `Lidar Map RMS` is **lower**
  (the kept points are real correspondences).

## 4. Decision tree after the series

- **B1 reliably beats B0 (bounded rate up, twist down, ~100 m):** lock the
  weight, then attack the **early-startup collapse basin** (findings item 4):
  most divergences seed early -> consider holding the prior harder / tighter
  rescue budget for the first N scans until initialized.
- **B1 ~ B0 (within chaotic noise) but RMS dropped:** the cull worked but the
  signal is still too weak -> next levers are **brightness-constancy** (affine
  a*I+b) and **patch selection complementary to the degenerate axis**
  (FAST-LIVO/COIN-LIO; REVIEW item #8) -- real work, do only if warranted.
- **B1 worse / `Lidar Map Points`->0:** occlusion tol too tight or the pose is
  too wrong for the map to project consistently -> loosen tol; if RMS still high,
  the projection model (LUT/column-shift) needs the Ouster metadata calibration
  (COIN-LIO uses it directly) rather than the self-cal.
- **Camera terms:** blocked until `extrinsics/cam2lidar` and the `/lucid_camera_1`
  serial/intrinsics are confirmed (findings item 2). Don't interpret camera
  absolute accuracy until then.
