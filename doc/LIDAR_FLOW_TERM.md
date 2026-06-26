# Frame-to-frame LiDAR flow term (the signal half of EXPLORATION #2)

Completes `doc/EXPLORATION_2026-06-26.md` #2: a **frame-to-frame** image term that
registers the current scan against the *previous* scan's image to observe the
along-tunnel motion the frame-to-map reflectivity term can't (it's along-axis
aliased). It is fused through `directionSeparateTerm` (PR / `doc/DIRECTION_SEPARATED_FUSION.md`)
so it only constrains the degenerate axis — the "fuse only into the degenerate
direction" idea, now with an actual signal behind it.

## The term

`accumulateLidarFlowResidual` (in `nano_gicp.cc`, modeled on the two existing
in-tree terms):
- iterates the current **source** points; each is moved by the correction
  (`x = trans · p_w`) and projected into the **previous** scan's image via the
  existing **spherical model** (the `dπ` of `accumulateLidarMapResidual`);
- residual `r = prev_image(projection) − point's own brightness`;
- Jacobian `J = [−G·skew(x) | G]`, `G = grad_I · dπ · R_lw_prev` — the **visual
  frame-to-frame** left-perturbation form. So the math is the composition of two
  already-in-tree, in-use derivations (spherical map term + visual f2f term),
  **correct by construction**;
- count-normalized like the map term; `weight ≤ 0` (default) → no-op /
  bit-identical.

Controls: `setLidarFlowWeight`, `setLidarFlowPrev(prev_img, T_lw_prev)`; params
`odom/lidar_image/flow/{enabled, weight}`.

## Threading safety (the part that warranted its own PR)

The term needs **cross-scan state** (the previous image + pose) read inside an
OpenMP-parallel loop — the one hazard class this codebase has already been bitten
by (the 1/48 `std::out_of_range`, fixed in the photometric loop). It's made safe
three ways, mirroring the **proven `visual_prev_` lifecycle**:
1. **Node stash** (`odom.cc`, after `align()`, like the visual stash): the current
   image is promoted to "previous" tagged with the *corrected* pose. cv::Mat
   assignment is a shallow ref, but the node builds a **fresh image every scan**
   and never mutates the old one in place, so the buffer stays valid.
2. **Owned deep copy**: `setLidarFlowPrev` does `prev_img.clone()`, so the gicp
   owns its buffer independent of the node's.
3. **Refcount snapshot**: the accumulator captures `const cv::Mat prev =
   lidar_flow_prev_img_` at the top, holding the buffer alive for the whole
   parallel loop even against a concurrent reassignment — the same ownership guard
   as the photometric-loop fix.

## Status — verified vs. unvalidated (read this)

**Verified:** clean build; `colcon test` incl. the off-path bit-identical and
empty-previous safety guards; ASan+UBSan (memory/UB) and the **TSan** concurrency
harness (the cross-scan-state race check) — see the PR.

**NOT validated:** the term's numerical correctness *on real data* and its
**efficacy** on the tunnel. The Jacobian is correct by construction (it reuses two
working terms) and the safety is verified, but a sign/frame slip can only be ruled
out by running the node on the bag, and whether range/intensity flow actually
recovers the along-axis motion is the open question. It is a **default-off
prototype of the signal**. Validate with a `flow/weight` × `dirSeparated/ratio`
A/B on the 06042026 bag (with the X-ICP gate on), watching the deg=6 rate and
worst-case `max|x|`, and start the weight low.

## Known simplifications (follow-ons)
- No occlusion / wrong-surface rejection on the previous image (the map term has
  it via a same-frame range image; the frame-to-frame case would need the previous
  range image too). Far/near surface mismatches add residual noise — add the
  previous range image + the depth-consistency cull if the term shows promise.
- Uses the reflectivity/intensity channel the node already builds; a dedicated
  **range** image as the flow channel (more along-axis structure) is the natural
  next experiment.
