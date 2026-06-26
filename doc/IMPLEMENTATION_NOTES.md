# Implementation Notes — acting on the degeneracy literature

Companion to `doc/RELATED_WORK.md` (the survey + lineage). That doc mapped the
governor onto Zhang / X-ICP / LODESTAR / GenZ-ICP; this one records what was
actually implemented from those findings, what was already present, and what does
not fit the architecture — so a reviewer knows the status of each.

> **Build/validation caveat.** These changes follow the branch's invariant
> (default-off, unit-tested, **bit-identical when disabled**). They were authored
> against the source but **compiled and tested in a full ROS 2 / PCL / Eigen
> environment is required** — the authoring environment had no `ament`/`gtest`.
> Empirical tunnel A/B validation stays blocked on the missing bag for all of
> them, exactly as the governor's n=24 validation was deferred.

## Status of each finding

| Finding | Source | Status here |
|---|---|---|
| Governor test gaps (oblique + multi-held-dir) | self (review) | **Landed** — 2 new gtests in `test_degeneracy_governor.cpp` |
| GenZ-ICP adaptive point-to-plane / point-to-point blend | Lee et al., RA-L 2025 | **Landed, wired, default-off** — `genz_weight.h` + gate wiring + params + gtests |
| X-ICP ternary localizability | Tuna et al., T-RO 2024 | **Landed, wired, default-off** — `xicp_localizability.h` + gate wiring + params + kernel & integration gtests |
| LODESTAR Schmidt-Kalman coupling | Lee/Marsim/Myung, RA-L 2025 | **Tractable slice prototyped** — observability-scheduled observer gain (`degeneracy_observer.h`, default-off). Full ESKF still deferred. See §4 + `doc/EXPLORATION_2026-06-26.md` |

---

## 1. Governor test gaps — landed

`test/test_degeneracy_governor.cpp` gains the two cases the original 8 missed,
both of which exercise the projection math the axis-aligned tests masked:

- **`ClampsAlongObliqueHeldAxis`** — held direction `(1,1,0)/√2`; asserts the
  along-axis component clamps to the cap while the orthogonal component is
  preserved exactly. This is the case the real gate produces (skew eigenvectors).
- **`ClampsMultipleHeldAxesIndependently`** — two held axes (x and y) clamped
  independently with the unheld z preserved, pinning the order-independence of
  the sequential per-axis clamp (valid only because held dirs are orthonormal).

No production code changed; pure test hardening of the shipped `governPose`.

## 2. GenZ-ICP adaptive blend — landed, wired, default-off

`include/nano_gicp/genz_weight.h` adds the pure, unit-tested kernel
`genzPlaneWeight(cond_ratio, floor, knee)` → the point-to-plane weight `alpha`.
Wiring in `nano_gicp.cc`:

- **Per-iteration α (lagged, like the adaptive kernel):** after the geometric
  Hessian `H_geo` is snapshotted, its translation block's conditioning
  `λ_min/λ_max` drives `current_genz_alpha_` via `genzPlaneWeight`. A
  well-conditioned scan → `alpha = 1`; an ill-conditioned (tunnel) scan → `alpha`
  ramps down toward `floor`.
- **Blend in `linearize`:** the per-correspondence metric becomes
  `M = alpha·M_plane + (1−alpha)·pointWeight·I`, reusing the *same* residual and
  Jacobian — point-to-point is just the identity-metric limit. `alpha == 1`
  (default / healthy / feature-off) → pure point-to-plane, **bit-identical**.
- **Controls:** `setGenZWeighting(enabled, floor, knee, pointWeight)`; params
  `odom/genz/{enabled,floor,knee,pointWeight}`; documented in `cfg/params.yaml`.
  `floor = 1` (default) ⇒ blend never engages.

**Honest scope:** GenZ-ICP derives `alpha` from *per-point planarity*; this drives
it from the *scan-level* Hessian conditioning the gate already computes, to avoid
a second per-point eigenanalysis in the hot loop. It is therefore *adapted from*,
not a verbatim port of, GenZ-ICP — stated as such in the header. `pointWeight`
[1/m²] must be tuned to the plane metric's scale before an A/B (the default 1.0 is
a placeholder); the right value is roughly the plane metric's typical eigenvalue.

**Tests:** `test/test_genz_weight.cpp` (off-contract, healthy=pure-plane, linear
ramp, collapsed-axis floor, NaN/degenerate-input safety).

## 3. X-ICP ternary localizability — landed, wired, default-off

The binary half of X-ICP **already existed**: the gate at `nano_gicp.cc:~989` is
Zhang solution remapping (hard prior-hold on non-localizable directions), and
`softGateKeepFraction` already does a smooth partial hold near the threshold.
X-ICP's distinct contribution is the **explicit ternary judgment** with a *second*
threshold and a controlled partial-admit band — now implemented.

`include/nano_gicp/xicp_localizability.h` adds the pure, tested kernels:
`xicpCategory(eigval, κ_partial, κ_full)` → {NonLocalizable, Partial, Localizable},
and `xicpPartialScale(...)` → the controlled-update fraction (0 below the partial
bar, linear across the band, 1 above the full bar). Equal thresholds reduce to the
current binary gate; mis-ordered thresholds are sanitized.

**Wiring (applied):** in the gate's per-eigen-direction loop (both the rotation
and translation blocks), when the ternary gate is enabled the binary/soft/prob
`keep` is replaced by `keep = xicpPartialScale(lam, κ_partial, κ_full)` with
`κ_partial = degeneracyThreshRatio·λ_max` (the existing degeneracy bar) and
`κ_full = fullRatio·λ_max` (a new, looser localizable bar). Held-direction
recording, degenerate counting, and the visual-rescue path are unchanged (a
hard-degenerate direction `lam ≤ κ_partial` still records and still gets
`keep = 0`). Controls: `setXicpTernary(enabled, full_ratio)`; params
`odom/xicp/{ternaryEnabled,fullRatio}`; documented in `cfg/params.yaml`.
`ternaryEnabled = false` (default) → the existing gate, **bit-identical**.

**Tests:** `test/test_xicp_localizability.cpp` (kernel: ternary split, linear
partial scale, binary reduction, mis-ordered-threshold safety) plus two gate-level
integration tests in `test/test_nano_gicp.cpp` (`XicpTernaryDisabledIsBitIdentical`,
`XicpTernaryStillHoldsStronglyDegenerateAxis`).

## 4. LODESTAR Schmidt-Kalman — architectural misfit (not implemented)

LODESTAR's DA-ASKF gives degenerate ("fixed") states **zero Kalman gain** and lets
them anchor the active states through the **cross-covariance** blocks `P_uf/P_fu`
of a Joseph-form filter update. That presupposes a covariance-propagating
estimator with a state covariance `P`.

**DLIO has no such `P`.** Its back-end is a *contracting hierarchical observer*
(Lopez 2023, `propagateState`/`updateState`) — deterministic gains, no propagated
covariance matrix. There is nothing for a Schmidt-Kalman cross-covariance update
to attach to. This is *exactly why* the governor's covariance inflation is
**informational-only** (it populates the published `/odom` covariance for
downstream consumers; it does not feed back into our own observer).

A faithful LODESTAR port would mean **replacing the observer with a covariance-
propagating filter (ESKF/UKF)** — a back-end rearchitecture, not a default-off
feature. **Update (2026-06-26):** rather than the full rewrite, a *tractable slice*
is now prototyped — `include/dlio/degeneracy_observer.h`'s
`attenuateAlongHeldAxes`, applied in `updateState()` behind `odom/geo/degenObsGain`
(default 1.0 = off / bit-identical). It is the contracting-observer analogue of
LODESTAR's reduced-gain "fixed" state: on the gate's held axes the observer's
LiDAR correction is scaled toward the IMU prior, so the runaway is attenuated *at
the source* (inside the estimator) rather than only bounded after the fact by the
governor. It does **not** add covariance cross-coupling or snap-back — that still
needs the ESKF. Full design, trade-offs, and the recommended novel directions
(saliency reweighting, direction-separated range-image flow) are in
`doc/EXPLORATION_2026-06-26.md`. If the slice's gain-scheduling proves
insufficient on the bag, adopt the ESKF + LODESTAR fixed/active partition then
(cf. ALIVE-LIO, arXiv:2604.02706).

---

## Files touched

- `test/test_degeneracy_governor.cpp` — +2 gtests (§1)
- `include/nano_gicp/genz_weight.h` *(new)*, `test/test_genz_weight.cpp` *(new)* — §2
- `include/nano_gicp/xicp_localizability.h` *(new)*, `test/test_xicp_localizability.cpp` *(new)* — §3
- `test/test_nano_gicp.cpp` — +2 X-ICP gate integration gtests (§3)
- `include/nano_gicp/nano_gicp.h`, `src/nano_gicp/nano_gicp.cc` — GenZ + X-ICP setters/members/wiring (§2, §3)
- `include/dlio/odom.h`, `src/dlio/odom.cc`, `cfg/params.yaml` — GenZ + X-ICP params (§2, §3)
- `CMakeLists.txt` — register the two new gtests
- `doc/IMPLEMENTATION_NOTES.md` *(this file)*

All new flags default off; with them off the registration path is bit-identical
to pre-change. Build + `colcon test` in a ROS 2 env is the remaining gate before
any tunnel A/B.
