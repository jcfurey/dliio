# Robustness research — papers & packages, lensed to dliio

> Reference survey (2026-06-17/18) of techniques from the literature and
> open-source packages that would make dliio more robust, with a dliio-specific
> "what's portable and how invasive" shortlist. Companion to
> `doc/ADAPTIVE_TRUST.md` (the adaptive-weighting design) and
> `doc/FINDINGS_2026-06-17.md` (the live specular water-pool failure that
> motivates much of this). Produced from a five-angle deep-research pass;
> confidence is calibrated below and single-source/recent-preprint claims are
> held out of the recommendations.

## Framing

The most useful result up front: **almost every technique worth adopting maps
onto a dliio component that already exists** — the block-split Zhang degeneracy
gate, the soft gate (`degeneracySoftness`), the IMU-consistency clamp
(`maxCorrTrans/Rot`), the photometric + COIN-LIO LiDAR-image terms, the per-term
trust telemetry (`Geo Rot/Trans Trust Margin`, `Photometric Points/RMS`), and the
`intensityIncidence`/`intensityCosMin` infra. This is a set of principled
upgrades to existing parts, not a greenfield program.

The **live failure** (`FINDINGS_2026-06-17.md` §3) is a specular floor-water pool:
(a) grazing-incidence **dropout** loses the floor → the vertical/pitch constraint
disappears, and (b) **mirror returns** plant phantom points *below* the surface
that drag the pose down. The shortlist prioritizes that.

## Shortlist — what to do, ranked by leverage ÷ invasiveness

| # | Change | Targets | dliio component it upgrades | Invasiveness |
|---|---|---|---|---|
| 1 | Ground-plane prior + sub-ground point reject — **IMPLEMENTED (off by default)** as `odom/preprocessing/subFloorReject/*`; per-cell gravity-aligned dense-bin floor, not benched on the pool yet | water-pool mode (b) ghosts | new preprocessing stage (uses the now-trusted IMU gravity) | Low |
| 2 | Trigger an IMU-prior lean when the floor constraint collapses — **IMPLEMENTED (off by default)** as `odom/gicp/adaptiveClamp/*`; the clamp caps tighten as the `Geo Trust Margin` degrades | water-pool mode (a) dropout | the `Geo Trust Margin` telemetry → the IMU-consistency clamp | Low–med |
| 3 | Probabilistic soft-attenuation in the gate (per-direction SNR) | degeneracy brittleness | replaces the `degeneracySoftness` smoothstep | Low–med |
| 4 | Per-direction trust from correspondence Jacobians (SuperLoc/X-ICP) | adaptive trust Layer 2 | the per-axis routing in `ADAPTIVE_TRUST.md` | Medium |
| 5 | Chebrolu adaptive-α robust kernel | outlier-tuning burden | replaces the fixed Huber in the IRLS path | Low |
| 6 | Dual-return / sub-ground reject *(if Ouster 2nd returns are populated)* | glass/mirror ghosts | preprocessing reject | Low (conditional) |
| 7 | *(optional, heavier)* GenZ-ICP adaptive plane↔point blend | dropout ill-conditioning | the GICP residual/weighting | Medium |

**Headline:** #1 and #2 together address the live water pool, and #2 closes a loop
with the telemetry already added — the `Geo Trans/Rot Trust Margin` collapsing *is*
the floor-dropout detector, and can trigger leaning on the (post-extrinsic-fix,
now-trustworthy) IMU on exactly the pitch/Z axes that lost their constraint.

---

## Part 1 — Specular / reflective surfaces (the live failure)

**Mode (b) — sub-floor ghost returns.** Lowest-risk, highest-value: a
**ground-plane-prior reject** — maintain a ground plane (RANSAC seeded by the IMU
gravity vector) and drop returns more than a margin below it before correspondence
search. This is literally the water-puddle signature (specular bounce to elevated
objects returns *below* the road plane) and the mechanism behind **Patchwork++'s
Reflected-Noise-Removal** (intensity + below-sensor geometry; IROS 2022). If the
sensor's **second returns** are populated, add dual-return rejection (glass/mirror
points appear in strongest-but-not-last returns) — **STAR-Center
`Reflection_detection`** (Sensors 2024) and **GRASS** (2026, mirror-symmetry
virtual-point removal) are the references but are heavier (plane-fitting, map-side)
and degrade in corridors.

**Mode (a) — grazing dropout / lost pitch constraint.** Not a filtering problem, a
*lost-constraint* problem → fixes live in the solver and in protecting the prior:
- **GenZ-ICP** (RA-L 2024): adaptively blends point-to-plane (planar) with
  point-to-point (non-planar), weighted by the planar fraction, keeping the
  Hessian conditioned when only the floor remains.
- **AdaLIO** (IROS 2023): tightens voxel/search/plane-residual params under
  confined-scene degeneracy (heuristic detector).
- **Protect grazing ground points** from over-aggressive scan cleaning, and **lean
  on the IMU** (Part 2). dliio's `intensityIncidence`/`intensityCosMin` is the hook
  for incidence handling.

**Caveat (consensus across sources):** intensity-based specular rejection is
**brittle alone** — LiDAR intensity is uncalibrated and range/angle-dependent, and
a *coherent* wet-floor or mirror plane is not a sparse "outlier", so
SOR/robust-kernel/intensity filters miss it. The **geometric sub-ground reject** is
far more robust than intensity thresholds for the water case.

---

## Part 2 — Degeneracy & graceful degradation

dliio's gate is the **Zhang–Kaess–Singh** baseline (ICRA 2016); because it already
**splits rotation/translation blocks**, it is past the original's single-global-
threshold weakness. Upgrade path (mostly ETH/Hutter + NTNU lines):

- **Probabilistic degeneracy detection** (Hatleskog & Alexis, RA-L 2024) — the
  principled successor to `degeneracySoftness`: propagate sensor noise into the
  Hessian, compute per-eigen-direction probability *p* that signal exceeds noise,
  and scale the update's pseudo-inverse by *p*. Reuses the existing block
  eigendecomposition; replaces a hand-shaped keep-fraction with a noise-derived
  one. **The single most natural next step for the soft gate.**
- **X-ICP / LP-ICP** (Tuna/Nubert/Hutter, T-RO 2024 / arXiv 2025) — per-direction
  localizability from the *correspondence* Jacobians projected onto the
  rotation/translation eigenvectors dliio already computes (`(Jᵢ·vⱼ)²`
  accumulation), with a ternary full/partial/none action.
- **Field result that should steer the gate's evolution** ("Informed, Constrained,
  Aligned," ETH, arXiv 2024, code `leggedrobotics/perfectlyconstrained`): across
  real degenerate data, **in-solver equality/Tikhonov constraints beat post-hoc
  solution remapping**, and remapping-alone is insufficient for severe/long
  degeneracy. Argues for applying the constraint *inside* the solve.
- **Leaning on the IMU as constraints drop out:** **LODESTAR** (RA-L 2025)
  condition-number-gates states active/fixed and uses a Schmidt-Kalman partial
  update (fixed states anchor but still inform via covariance) — the principled
  generalization of the IMU-consistency clamp. **SuperLoc** (2024) does it
  *pre-solve* from a correspondence-derived risk score.

---

## Part 3 — Adaptive multi-modal trust (the `ADAPTIVE_TRUST.md` direction)

Best fits for the `H += w·JᵀJ` information form, closest-match first:

- **Super Odometry / SuperLoc** (IROS 2021 / 2024) — **closest match to the
  design.** Counts per point-plane correspondence which DOF is constrained,
  normalizes to per-direction confidences γ_trans, γ_rot ∈ [0,1], and sets
  `Cov_prior = I − Σ_cov` so weak directions draw on the IMU prior. Almost exactly
  the Layer-2 per-axis routing, computed pre-solve from Jacobians dliio already
  assembles. *(Original IROS prior-covariance formula UNCERTAIN; mechanism
  confirmed via the SuperLoc supplementary.)*
- **Dynamic Covariance Scaling** (Agarwal et al., ICRA 2013) — closed-form
  residual-driven weight `s = min(1, 2Φ/(Φ+χ²))`, **zero extra state**,
  per-iteration. Cleanest drop-in for a per-term *magnitude* weight; pair with the
  gate (gate = direction, DCS = magnitude).
- **Switchable Constraints** (Sünderhauf & Protzel, IROS 2012) — a per-term latent
  `w∈[0,1]` solved jointly with a prior pulling `w→1`; `H += w·JᵀJ` already
  supports it.
- **Learned per-frame 6×6 covariance** → feed `Σ⁻¹` as the per-term weight (native
  per-direction). Promising but flagged: overconfidence off-distribution, single
  recent source.
- **LVI-SAM**'s binary smallest-eigenvalue switch is the *hard special case the
  soft gate generalizes* — and its known **chatter near threshold** is exactly what
  `degeneracySoftness` exists to prevent (external validation of that choice).

Granularity/cadence: per-direction methods (SuperLoc, learned covariance, the
eigenvalue primitive) are rarer and most aligned with the target; most robust
kernels are per-term scalar only; DCS/GNC/M-estimators are per-iteration,
LVI-SAM/learned-covariance per-scan.

---

## Part 4 — OSS portability (dliio's DLIO ancestry)

- **DLO** (ancestor): robustness via **concave/convex-hull keyframe submap
  selection** — the local map deliberately spans diverse directions, mitigating
  degeneracy through *map coverage* rather than solver math. Most directly
  reusable.
- **FAST-LIO2**: `s = 1 − 0.9·|d|/√‖p‖ > 0.9` normalized point-to-plane inlier gate
  + "≥N neighbors within distance" pre-filter — trivially portable correspondence
  validity test. (Does *no* echo filtering and has *no* explicit degeneracy gate —
  leans entirely on the iEKF/IMU.)
- **KISS-ICP**: `κ = σ/3` Geman-McClure scale and `τ = 3σ` correspondence threshold
  from predicted motion — derive `σ` from dliio's **IMU-preintegrated motion
  prior** for a parameter-free adaptive correspondence radius.
- **COIN-LIO**: **complementary-direction patch selection** — detect the
  geometrically-weak direction and pick intensity-image patches whose gradient
  constrains *that axis*. dliio has the LiDAR-image term but not the *targeting*;
  using the gate to select patches on the degenerate axis is Layer-2 routing with a
  concrete reference implementation.
- **GLIM**: soft prior-factor + LM-damping regularization instead of hard
  remapping — consistent with the "in-solver constraint > post-hoc remap" result.
- **Point-LIO**: per-point update + IMU-as-output (graceful under saturation);
  conflicts with batch GICP, so only the IMU-saturation idea is portable.

---

## Part 5 — Robust kernels

- **Chebrolu et al., "Adaptive Robust Kernels"** (RA-L 2021, on **Barron** CVPR
  2019): per-iteration auto-tuning of the loss shape `α` via a 1-D NLL search — a
  **no-tuning replacement for the fixed Huber** in the IRLS path. Highest-value
  kernel upgrade.
- **GNC** (Yang et al., RA-L 2020): anneal convex→true robust cost; robust to
  ~70–80% outliers (empirical, no global guarantee); wraps the IRLS loop. Only if
  high-outlier regimes appear.
- **LeGO-LOAM** two-step ground-decoupled solve (ground features →
  z/roll/pitch, then edges → x/y/yaw): a structural way to keep the vertical/
  attitude DOFs well-constrained — relevant when specular dropout leaves only the
  floor.

---

## Confidence & caveats

- **High confidence (≥2 sources / canonical primary):** Zhang gate;
  X-ICP/probabilistic degeneracy; the "in-solver > post-hoc" field result;
  SuperLoc per-direction confidence; DCS; switchable constraints; Chebrolu/GNC;
  KISS-ICP; DLO/DLIO/FAST-LIO2/COIN-LIO/GLIM mechanisms; Patchwork++ RNR; the
  sub-ground water signature.
- **Single-source / held out of decisions:** FAST-LIVO2's "degeneration-aware"
  being a true per-direction LiDAR weighter (abstract overstates the confirmed body
  mechanism); the 2025 learned-covariance results; the recent specular preprints
  (IDSOR, Ghost-FWL, MirrorDrift) — useful as mechanism corroboration, not as
  methods to adopt; the LP-ICP numerical thresholds; the exact Zhang remapping
  projection equation (PDF extraction failed; described from secondary sources).
- **Domain gap:** most specular-specific papers are **indoor glass/mirror**; the
  wet-floor-for-a-ground-robot case is closest to Patchwork++ RNR + the geometric
  sub-ground signature, which is why the geometric ground-reject outranks the
  intensity/echo methods for the water pool.

---

## Sources

**Degeneracy & localizability**
- Zhang, Kaess, Singh, "On Degeneracy of Optimization-based State Estimation," ICRA 2016 — https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.pdf
- X-ICP (Tuna et al.), T-RO 2024 — https://arxiv.org/abs/2211.16335
- LP-ICP (Yue et al.), arXiv 2025 — https://arxiv.org/abs/2501.02580
- Probabilistic degeneracy detection (Hatleskog & Alexis), RA-L 2024 — https://arxiv.org/abs/2410.10784
- "Informed, Constrained, Aligned" field analysis (ETH), arXiv 2024 — https://arxiv.org/abs/2408.11809 · code https://github.com/leggedrobotics/perfectlyconstrained
- LODESTAR, RA-L 2025 — https://arxiv.org/abs/2511.09142
- SuperLoc, 2024 — https://arxiv.org/abs/2412.02901
- Learning-based localizability (Nubert et al.), 2022 — https://arxiv.org/abs/2203.05698

**Specular / reflective / ghost-point**
- STAR-Center Reflection_detection (Sensors 2024) — https://arxiv.org/abs/2406.10494 · https://github.com/STAR-Center/Reflection_detection
- Mapping with Reflection (SSRR 2020) — https://arxiv.org/abs/1909.12483
- 3DRef benchmark, 2024 — https://arxiv.org/abs/2403.06538
- GenZ-ICP, RA-L 2024 — https://arxiv.org/abs/2411.06766
- AdaLIO, IROS 2023 — https://arxiv.org/abs/2304.12577
- Patchwork++ (RNR), IROS 2022 — https://arxiv.org/abs/2207.11919 · https://github.com/url-kaist/patchwork-plusplus
- GRASS (glass virtual-point removal), 2026 — https://www.mdpi.com/2072-4292/18/2/332 · https://github.com/wpshao/GRASS
- Removert, IROS 2020 — https://gisbi-kim.github.io/publications/gkim-2020-iros.pdf · ERASOR, RA-L 2021 — https://arxiv.org/abs/2103.04316
- WGICP — https://arxiv.org/pdf/2209.09777 · InTEn-LOAM — https://arxiv.org/pdf/2209.05708
- *(UNCERTAIN/single-source: IDSOR, MirrorDrift, Ghost-FWL, Waymo water-detection patents)*

**Adaptive fusion & robust estimation**
- FAST-LIVO2, T-RO 2024 — https://arxiv.org/abs/2408.14035 · https://github.com/hku-mars/FAST-LIVO2
- R³LIVE (ICRA 2022) — https://arxiv.org/abs/2109.07982 · R²LIVE (RA-L 2021) — https://arxiv.org/abs/2102.12400
- LVI-SAM, ICRA 2021 — https://arxiv.org/abs/2104.10831
- Super Odometry, IROS 2021 — https://arxiv.org/abs/2104.14938
- Switchable Constraints (Sünderhauf & Protzel), IROS 2012 — https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf
- Dynamic Covariance Scaling (Agarwal et al.), ICRA 2013 — http://www2.informatik.uni-freiburg.de/~agarwal/docs/agarwal13_icra_ws.pdf
- Barron, "A General and Adaptive Robust Loss," CVPR 2019 — https://arxiv.org/abs/1701.03077
- Chebrolu et al., "Adaptive Robust Kernels," RA-L 2021 — https://arxiv.org/abs/2004.14938
- GNC (Yang et al.), RA-L 2020 — https://arxiv.org/abs/1909.08605
- Learned per-frame LiDAR covariance, 2025 — https://arxiv.org/abs/2509.18954

**OSS packages**
- KISS-ICP — https://arxiv.org/abs/2209.15397 · https://github.com/PRBonn/kiss-icp
- DLO — https://github.com/vectr-ucla/direct_lidar_odometry · DLIO — https://arxiv.org/abs/2203.03749 · https://github.com/vectr-ucla/direct_lidar_inertial_odometry
- FAST-LIO2 — https://github.com/hku-mars/FAST_LIO · Point-LIO — https://github.com/hku-mars/Point-LIO
- GLIM — https://arxiv.org/abs/2407.10344 · https://github.com/koide3/glim
- LIO-SAM — https://github.com/TixiaoShan/LIO-SAM
- COIN-LIO — https://arxiv.org/abs/2310.01235 · https://github.com/ethz-asl/COIN-LIO
- SuMa / SuMa++ — http://jbehley.github.io/projects/surfel_mapping/ · https://arxiv.org/abs/2105.11320
- LeGO-LOAM — https://github.com/RobustFieldAutonomyLab/LeGO-LOAM
