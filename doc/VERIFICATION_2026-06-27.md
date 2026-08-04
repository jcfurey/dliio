# Verification 2026-06-27 — bug hunt & logic audit of the 2026-06-26/27 cycle

Source-level verification of this cycle's levers and of the `be78056` crash fix,
cross-referenced against the vendored libraries, upstream implementations, and
the literature. Three code corrections landed with this doc (observer rotation
frame, saliency formula, corruption telemetry) plus two new executable
verification harnesses (`test_kdtree.cpp`, spline-order cases in
`test_imu_integration.cpp`).

---

## 1. The crash mechanism in `be78056` is DISPROVEN; the guards stand on
##    defence-in-depth grounds

`be78056` claimed: *a NaN/Inf query makes nanoflann's `nearestKSearch` return a
garbage index that clears the distance gate.* Source inspection of the vendored
kd-tree shows this is **wrong** — the pipeline fails **safe** on non-finite
queries, by three independent facts:

1. `KNNResultSet::init` writes `dists[capacity-1] = FLT_MAX` on **every call**
   (`nanoflann.h`), so with k=1 there is no stale-distance path even through the
   OMP `firstprivate` vector reuse in `update_correspondences`.
2. The leaf loop admits a point only via `if (dist < worst_dist)` — false for a
   NaN or Inf distance (IEEE-754), so `addPoint` is never called; `found == 0`
   and the correspondence gate `k_sq_dists[0] < max_dist_sq` rejects. A
   huge-finite query (diverged pose, ~1e18 coords → dist ~1e36 < FLT_MAX)
   returns a *real* neighbour whose distance the gate rejects.
3. When `addPoint` fires, the index is `vAcc[i]` — a build-time point index,
   in-range by construction; the metric is a plain L2 forward
   (`SO3_Adaptor` → `L2_Simple_Adaptor`).

All three properties are now **pinned as executable tests** in
`test/test_kdtree.cpp` (NaN/Inf/huge fail-safe, 20k-query in-range fuzz,
firstprivate-reuse reset contract, brute-force NN cross-check). Upstream
context: [koide3/fast_gicp] and [vectr-ucla/direct_lidar_inertial_odometry] use
the identical unguarded `k_sq_dists[0] < thresh ? k_indices[0] : -1` pattern —
safe for the same reasons; [jlblancoc/nanoflann] documents that inputs are not
sanitized.

**Also exonerated: the submap concurrency path.** Adoption is gated on
`submap_future.wait_for(0) == ready` (`odom.cc:2469`), a happens-before edge
with the async build's completion ([futures.async], cppreference);
`registerInputTarget` builds **fresh** kd-tree/covariance objects per build so
the registration instance's shared objects are never mutated in place; keyframe
vectors are read under `keyframes_mutex`; `submap_hasChanged` is atomic.

**What the evidence actually indicates: memory corruption by an out-of-bounds
writer.** Two clues:
- Both observed garbage indices are bit-patterns of tiny floats:
  `74285787 = 0x046D8DBB` ≈ 2.8e-36f and (2026-06-25) `327563643 = 0x13866C3B`
  ≈ 3.4e-27f — the signature of an `int` read from memory scribbled by float
  data or a reused freed allocation, not of a computed index.
- In-tree corroboration: `estimate_spatial_intensity_gradient` has carried an
  upper-bound guard on `target_index` since before this cycle
  (`nano_gicp.cc:~842`) — an impossible index was seen in that path historically.

**Consequences (landed here):**
- The `be78056` guards are *kept* — they convert the crash into a skip
  regardless of the writer — but their comments now state the accurate
  rationale.
- The `linearize` guard now **counts** trips (`lastOobCorrespondences()`), and
  the node logs a throttled `MEMORY-CORRUPTION SENTINEL` warning — the crash
  becomes telemetry.
- **The decisive next experiment is one ASan rep of the combo config on the
  06042026 bag** (unit-test ASan runs are clean; the writer only manifests
  under the deg=6 contended state). ASan will name the OOB write at the first
  occurrence.

## 2. Observer slice: rotation attenuation mixed frames — FIXED

`updateState()` attenuated `qcorr.vec()` along **world**-frame held directions,
but `qcorr = q̂ ⊗ qe_corr` and the vector part of a quaternion **product** is not
the rotated error vector (`vec(q̂⊗p) ≠ R(q̂)·vec(p)`; Solà, *Quaternion
kinematics for the error-state Kalman filter*, arXiv:1711.02508 — local vs
global error composition). The attenuation therefore leaked across axes as a
function of attitude. **Fix:** rotate the held world dirs into the body frame
(`d_b = R(q̂)ᵀd`) and attenuate `qe`'s vector part *before* composing `qcorr` —
exact for the small-angle error the observer integrates. Bit-identical when
`degenObsGain = 1`. The translation path (`err`, world-frame) was verified
exact and is unchanged. Relevant to the planned finer `degenObsGain` sweep: the
leakage may have blunted `slice05`'s 4/24.

## 3. Saliency boosted noise, not just edges — FIXED

`pointSaliency = 1 − planarity` gave **both** edges and isotropic-scatter
(noise) neighborhoods saliency ≈ 1 — in the standard eigen-feature taxonomy
(Demantké et al. 2011; Weinmann et al., ISPRS 2015) it conflated *linearity*
with *sphericity*. LOAM (Zhang & Singh, RSS 2014) selects edges, never scatter.
Up-weighting noise at full boost is the likely mechanism of the `sal8` tracking
stall in `FINDINGS_2026-06-26`. **Fix:** `saliency = linearity = (λ2−λ1)/λ2` —
edge/rib → ~1, plane → ~0, **scatter → ~0**. Tests updated
(`ScatterIsNotSalient`). Re-run the boost sweep with this form before judging
the lever.

## 4. GenZ blend violates the gate's observability doctrine — documented,
##    not fixed (default-off, A/B-negative)

The GenZ point-to-point mass `(1−α)·w·I` is blended inside `linearize`, so the
`H_geo` snapshot the degeneracy gate judges **includes manufactured isotropic
mass** on the weak axis — precisely the masking the codebase forbids for the
visual term ("the gate judges observability from [H_geo], NOT the
visual-augmented H") and the failure premise of Zhang/Kaess/Singh (ICRA 2016).
Mechanistically consistent with `pw=100` catastrophic (4/5 DIV): the gate
under-fires and releases the prior hold. Saliency, by contrast, reweights
*measured data* and legitimately belongs in `H_geo`. If GenZ is ever revisited,
the blend must accumulate separately, after the `H_geo` snapshot. Left as-is
(default-off, empirically negative, documented).

## 5. Verified correct (no change)

- **Flow-term Jacobian**: re-derived; `J = [−G·skew(x) | G]`,
  `G = gI·dπ·R_lw_prev` — matches the in-tree visual f2f convention and the
  spherical `dπ` of the map term (direct-alignment form of DSO / COIN-LIO).
- **Flow stash lifecycle**: `buildLidarIntensityImage` constructs a fresh local
  `cv::Mat` every scan and rebinds the member — the stashed shallow ref holds an
  immutable old buffer (the `cv::Mat::create()` reuse hazard is absent), and the
  gicp clones anyway.
- **Governor partial-band recording**: no double-record; orthonormal projection
  argument holds.
- **X-ICP ternary kernel**: matches Tuna et al. (T-RO 2024) ternary + partial
  admit; equal thresholds reduce to the binary gate (tested).

## 6. Scope note — "spline" and "ikd-tree"

This repo's continuous-time trajectory is DLIO's **analytic constant-jerk
piecewise-cubic** (Chen, Nemiroff & Lopez, ICRA 2023), not a B-spline; the new
spline-order tests in `test_imu_integration.cpp` pin the cubic term, knot
continuity, and agreement with an independent numerical integrator. Spline
*estimation* (as in RESPLE / CT-ICP) lives in the bench harness repo, out of
this repo's scope. The kd-tree here is the **static** vendored nanoflann tree,
rebuilt per submap by the background thread — not an incremental ikd-Tree
(Cai, Xu & Zhang, arXiv:2102.10808; hku-mars/ikd-Tree). An ikd-Tree swap would
amortize the per-submap rebuild but is an optimization, not a correctness
issue; the current rebuild-and-swap is what makes the fresh-object concurrency
argument in §1 hold.

## References

- [jlblancoc/nanoflann](https://github.com/jlblancoc/nanoflann) — vendored kd-tree
- [koide3/fast_gicp](https://github.com/koide3/fast_gicp) — upstream correspondence pattern
- [vectr-ucla/direct_lidar_inertial_odometry](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) — upstream DLIO
- Solà — *Quaternion kinematics for the ESKF*, [arXiv:1711.02508](https://arxiv.org/abs/1711.02508)
- Weinmann et al. — *Semantic point cloud interpretation based on optimal neighborhoods...*, ISPRS 2015; Demantké et al. — *Dimensionality based scale selection in 3D lidar point clouds*, 2011 — eigen-features (linearity/planarity/sphericity)
- Zhang & Singh — *LOAM*, RSS 2014 — edge/planar feature selection
- Zhang, Kaess & Singh — ICRA 2016 — degeneracy gate doctrine
- Chen, Nemiroff & Lopez — *DLIO*, ICRA 2023 — constant-jerk analytic deskew
- Cai, Xu & Zhang — *ikd-Tree*, [arXiv:2102.10808](https://arxiv.org/abs/2102.10808)
