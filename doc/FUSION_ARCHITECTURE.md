# Fusion architecture — dliio + robot_localization EKF (2026-07-09)

Response to the operational tunnel-slosh problem (the corkscrew/figure-eight
oscillation along the tunnel axis) and the 2026-07-08 X-ICP runaway. Based on
the survey of degradation-robust systems: the SubT-winning approach is
**redundant odometry sources fused with health gating**, not a perfect single
estimator (CompSLAM, arXiv:2505.06483; SuperLoc, arXiv:2412.02901). The slosh
mode itself is named in the literature — *degradation*: "small, back-and-forth
oscillations of the odometry," as distinct from unidirectional drift — and
mainstream LIO (FAST-LIO2) diverges outright in the same self-similar sections,
so swapping packages is not the fix.

## The three-layer design

**Layer 1 — keep the estimator from eating its own map** (this repo):
- `odom/keyframe/degenGate`: veto keyframe creation while the gate holds
  degenerate axes. A sloshing pose that keyframes plants duplicated wall
  structure in the submap ("map-lock" aliasing), which feeds the oscillation
  back — the self-contamination loop. Field-converged practice (DLIOM's
  accurate-state keyframe selection, arXiv:2305.01843; DAMM-LOAM; DALI-SLAM).
- `odom/xicp/partialBudget{Trans,Rot}` (`xicpBudgetedAdmit`): the ternary
  gate's partial band previously admitted `keep·comp` along marginal axes with
  **no cap** — fail-open when the marginal-band signal is dishonest (the
  2026-07-08 runaway: the aliased photometric drive both lifts the weak axis
  into the band and supplies the pull). Budgeting the per-scan cumulative
  admission (mirroring the visual-rescue budget) makes X-ICP fail-BOUNDED,
  matching Tuna et al.'s controlled partial update.

**Layer 2 — honest per-axis output** (already existed, now load-bearing):
- The governor's rank-1 covariance inflation (`covPosVar`/`covRotVar`) marks
  exactly the held axes untrusted in the published `/odom` covariance — this is
  the signal a downstream fusion layer needs to de-weight the sloshing axis.
- `odom/publishTf: false`: the EKF owns `odom -> base_link`; dliio contributes
  odometry *messages* only.

**Layer 3 — the fusion EKF** (`cfg/robot_localization_ekf.yaml`,
`launch/dlio_rl.launch.py`; requires `ros-<distro>-robot-localization`):
- dliio pose fused **differentially** (deltas, covariance-weighted): slosh
  increments along the inflated axis are largely ignored; the five observable
  axes pass at full weight. A Mahalanobis gate rejects residual map-lock jumps.
- **Rig kinematic odometry** (`rig_odom_topic`) supplies forward velocity — the
  along-tunnel information LiDAR cannot observe. The 07052026 rig's
  `base_link_correction` TF chain indicates a nav stack already runs on the
  platform; its odometry is the cheapest second source. (Extreme-tunnel systems
  end up here: CM-LIUW adds UWB+wheel in coal mines.)
- IMU angular velocity only (dliio already integrates the IMU tightly; r_l has
  no IMU bias states, so re-fusing orientation/accel adds drift).
- dliio **twist is deliberately not fused** — during slosh the velocity state
  is the oscillation's flywheel.

## Honest limits (read before relying on it)

1. **GIGO / downstream-only**: the EKF smooths the *output*; it has no feedback
   into the scan matcher. Layer 1 is what keeps dliio internally sane —
   without it the submap still corrupts and dliio can still diverge internally
   even while `/odometry/filtered` looks smooth.
2. **No second source, less value**: with `rig_odom_topic` absent, the tunnel
   axis rides the EKF motion model between dliio updates — smoother than raw
   slosh, but it will drift. The architecture's value scales with the quality
   of the kinematic source.
3. **SuperLoc's critique applies**: injecting the prior *into* the alignment
   before failure is stronger than blending after it. The merged
   `directionSeparateTerm` machinery is the hook for that upgrade — feeding the
   rig odometry prior along held axes inside the solve is the follow-on if the
   EKF layer proves insufficient (see doc/BAKEOFF_SUPERLOC.md).

## Validation plan

On the 06042026 substrate (n ≥ 24): combo + Layer-1 fixes vs combo alone —
watch DIV rate, crash rate, and **slosh amplitude/period** (the new operational
metric; log along-axis position + yaw against gate state). On the 07052026 bag:
the launch's isolation matrix, with `photometricWeight: 0` per the runaway
analysis. The keyframe gate's freeze-test prediction: slosh collapses toward
slow drift when the gate is on.
