# Bake-off protocol — dliio(combo+fixes) vs SuperLoc (2026-07-09)

SuperLoc (CMU, RA-L/ICRA 2025, arXiv:2412.02901, code + datasets:
superodometry.com/superloc) is the published system closest to this repo's
problem: it *predicts* per-DOF localizability from raw correspondences BEFORE
optimization (γ_trans/γ_rot ∈ [0,1]; < 0.2 ⇒ degradation likely) and, when
degeneracy is predicted, injects pose priors from other odometry sources as
soft factor-graph constraints weighted `Cov_prior = I − Σ_cov` — filling
exactly the degenerate directions. Its stated critique of post-hoc mitigation
(this repo's gate family): "often too late." A bake-off answers whether the
campaign should continue on dliio or pivot.

## Protocol

- **Substrate**: the 06042026 tunnel bag (n ≥ 24, crash-separated, fresh-build
  reproduction caveat applies: FINDINGS_2026-06-25). Secondary: 07052026 with
  `photometricWeight: 0` (see the runaway analysis).
- **Arms**:
  1. dliio `combo` (X-ICP fullRatio 0.05 + governor 0.15) + Layer-1 fixes
     (`partialBudget 0.15/0.05`, `keyframe/degenGate: true`) — the
     `ouster_tunnel_xicp.yaml` overlay as committed.
  2. SuperLoc, default config, same bag, same extrinsics; IMU as its prior
     source (no rig odometry in the bag ⇒ both arms LiDAR+IMU only — fair).
  3. (optional) dliio + `dlio_rl.launch.py` if the bag carries a rig odometry
     topic.
- **Metrics** (per rep):
  - genuine DIV rate + worst max|x| (existing harness verdicts),
  - crash rate,
  - **slosh amplitude and period**: dominant-frequency amplitude of along-axis
    position after detrending, over the in-tunnel segment — the operational
    corkscrew metric this campaign now targets,
  - CPU/scan latency (SuperLoc's factor graph vs dliio's GICP).
- **Decision rule**:
  - SuperLoc ≻ dliio on DIV *and* slosh by a wide margin → pivot: adopt (or
    port its predictive-localizability + prior-injection design into the
    `directionSeparateTerm` hook).
  - Comparable → continue dliio (the campaign's machinery is validated; port
    SuperLoc's *prediction-before-optimization* idea as an upgrade).
  - dliio ≻ SuperLoc → record and continue.

## Integration notes for the SuperLoc arm

- Inputs: /points + /imu; remap to the bag's topics; verify its expected IMU
  convention (it targets legged/handheld/ground rigs).
- TF: run it in its own frame tree (no `odom -> base_link` conflict with the
  replay's static TF; same two-parent caveat as the 07052026 rig).
- The harness lives in the test-ws repo; add a `superloc_` runner mirroring
  `dliio_*_hunt.sh`, reusing the same verdict classifier so DIV/SLOSH labels
  are comparable across arms.
- License/citation: cite Zhao et al., "SuperLoc: The Key to Robust LiDAR-
  Inertial Localization Lies in Predicting Alignment Risks" (arXiv:2412.02901)
  in any result writeup.
