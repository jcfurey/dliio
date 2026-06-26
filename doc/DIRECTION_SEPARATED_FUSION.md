# Direction-separated fusion (anti-dilution, the safe half of EXPLORATION #2)

Implements the **mechanism** of `doc/EXPLORATION_2026-06-26.md` #2 (range-image
direction-separated flow): the LOFF-style guard that lets an auxiliary term act
**only** on the degenerate axis and never perturb the well-observed ones. This is
the reusable, testable core; the frame-to-frame range-*flow signal* (the other
half of #2) is the larger node-level follow-on sketched below.

## Idea

Every negative auxiliary lever (intensity, condScale, near-IR, dense visual, GenZ)
failed by **dilution** — its mass landed on the strong axes. `conditionScaleTerm`
already *boosts* a term toward the weak axis; `directionSeparateTerm` is the
opposite knob: **project the term onto the geometrically-weak subspace**, so its
contribution to the strong axes is removed entirely. A term that is only allowed
to move the degenerate axis cannot corrupt the well-observed ones — and on a
non-degenerate scan it is suppressed to zero (can't hurt).

## How

- **`directionSeparateTerm(H_geo, ratio, H_term, b_term)`** (`nano_gicp.cc`, next
  to `conditionScaleTerm`, pure + unit-tested): for each 3×3 block of the
  *geometric* Hessian, build the orthogonal projector `P = Σ vᵢvᵢᵀ` over the
  eigen-directions with `λᵢ ≤ ratio·λmax` (the weak subspace), then
  `H_term ← P·H_term·P`, `b_term ← P·b_term`. Symmetric PSD preserved. A
  fully-observed block → `P = 0` (term suppressed); an empty block (`λmax ≤ 0`) →
  identity (untouched); `ratio ≤ 0` → off.
- Wired on the LiDAR-image term (`setLidarDirSeparate`, params
  `odom/lidar_image/dirSeparated/{enabled, ratio}`). Composes with `condScale`
  (boost toward the weak axis, then strip the strong axes). `enabled = false`
  (default) → the term is added in full, **bit-identical**.

## Status & honest scope

Default-off; kernel unit-tested (`DirSep.*` in `test_nano_gicp.cpp`, 5 cases:
off-identity, keep-weak/zero-strong, fully-observed-suppressed, empty-untouched,
symmetry/PSD). Build + `colcon test` green.

**This delivers the direction-separation guard, not a new signal.** Applied to the
existing reflectivity frame-to-map term — which the findings showed is along-axis
*aliased* — it mostly removes that term from the strong axes; its value is
unlocked by a term that actually carries weak-axis information. That term is:

### Sketch — frame-to-frame range-flow term (the next, larger piece)
Store the previous scan's **range** image + lidar pose (`lidar_prev_range_`,
`T_lw_prev_`), project the current scan's points into it via the existing
spherical model (reuse `accumulateLidarMapResidual`'s projection + Jacobian, but
frame-to-frame instead of frame-to-map), form a range/photometric flow residual,
and run it through `directionSeparateTerm` so it only constrains the along-axis
DOF. The range image (unlike reflectivity) carries geometric structure — ribs,
junctions — that varies *along* the tunnel, so it may observe the motion
reflectivity can't. The new risk is the **previous-image lifecycle** in the node's
scan loop (the same threading-hazard class as the photometric-loop crash), which
is why it warrants its own focused, separately-reviewed PR rather than riding in
with this pure-kernel change. Validate with a `ratio` × flow-weight A/B on the bag.
