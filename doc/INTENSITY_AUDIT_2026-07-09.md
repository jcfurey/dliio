# Intensity-pipeline audit — where the graffiti signal dies (2026-07-09)

Trigger: "my biggest worry ... is that the intensity or reflectivity lidar
fields aren't actually being used in a meaningful way", plus the field
observation that the dated bags carry strong wall texture ("much information on
the concrete walls such as graffiti to key on, COIN-LIO or similar nature").
This audit traced the reflectivity channel end-to-end and found that the
signal is real and present at full resolution — and destroyed before any
consumer reads it.

## The finding: every consumer reads post-voxel data

The organized reflectivity image is built PRE-voxel from the raw organized
scan (`buildLidarIntensityImage`, full sensor resolution — the graffiti is
intact there). But every residual that *references* brightness read it from
point fields that had already been through `pcl::VoxelGrid` at a 0.25 m leaf,
which AVERAGES all fields over the voxel:

| Consumer | Reference it read | Damage |
|---|---|---|
| 3D photometric term (`odom/gicp/photometricWeight`) | voxel-averaged `.reflectivity` of scan + submap points; kNN gradient support ~0.5–1 m | 5–30 cm graffiti strokes averaged away twice: once by the voxel, again by the gradient support |
| LiDAR frame-to-map term (`odom/lidar_image/weight`) | moving side = full-res image (good); reference = target point's voxel-averaged `.reflectivity` | residual compares full-res texture against blurred reference — the texture reads as noise |
| LiDAR frame-to-frame flow term (`flow/weight`) | moving side = previous full-res image (good); reference = source point's voxel-averaged `.reflectivity` | same asymmetry |

A 5–30 cm stroke on concrete at 0.25 m voxel pitch is at/below Nyquist in the
point domain; in the IMAGE domain (Ouster 64×1024/2048, ~0.18–0.35° azimuth
step) the same stroke spans several pixels at tunnel wall ranges. The channel
was never information-poor — the pipeline was reading it after the blur.

Corollary for the slosh/runaway campaign: what survives the voxel average is
low-frequency shading, which is self-similar along the tunnel — an ALIASED
along-axis drive. The 2026-07-08 X-ICP runaway analysis found exactly this
signature: the photometric drive both lifted the weak axis into the partial
band and supplied the pull (doc/FUSION_ARCHITECTURE.md, Layer 1).

## The five fixes (this change set)

1. **Flow term: image-to-image reference** (`odom/lidar_image/flow/imageRef`,
   default true). The reference brightness is sampled from the CURRENT scan's
   full-res image at the point's prior-pose projection (pose-independent),
   instead of the point's voxel-averaged field. Both sides of the residual now
   carry pre-voxel texture — the standard frame-to-frame photometric form
   (COIN-LIO, arXiv:2310.01235). `false` reproduces the original term
   bit-identically for A/B.
2. **Flow term: patch residuals** (`odom/lidar_image/flow/patch`, default 0,
   max 3). Each point compares a zero-mean (2P+1)² window instead of one
   pixel: uses the strokes' spatial STRUCTURE and cancels per-scan
   gain/offset (mini-DSO / COIN-LIO patches). Per-point weight is divided by
   the pixel count, so total influence is patch-size invariant.
3. **Map term: keyframe-image references** (`odom/lidar_image/imageRefs`,
   default false; on in `ouster_tunnel_lidarimg.yaml`). At keyframe creation
   the node samples per-point brightness from that keyframe's FULL-RES image
   (`sampleKeyframeLidarRefs`, mirroring the camera `VisualRef` machinery:
   index-aligned per-keyframe lists, concatenated in `buildSubmap` under
   `keyframes_mutex`, handed to the gicp via `setTargetLidarRefs`,
   propagated through `shareTargetDataFrom`). The map term then uses those
   instead of the voxel-averaged field; < 0 marks a point that didn't project
   into its keyframe's image (skipped). Size-mismatch → field fallback
   (bit-identical), so stale refs after a submap swap are inert.
4. **Tunnel overlay: 3D photometric term off** (`ouster_tunnel.yaml`
   `photometricWeight: 0.3 → 0.0`). The 0.3 was an n=3 verdict tuned before
   the 2026-06-17 extrinsic + dead-channel fixes (confounded), and the term's
   post-voxel reference makes it an aliased along-axis drive (the runaway
   mechanism above). Historical sweep numbers kept in the overlay header for
   provenance; re-derive at n≥24 before resurrecting a nonzero weight.
5. **Instrumentation** (`/diagnostics`): `Lidar Flow Active/Points/RMS`
   (was previously invisible — "term off" vs "term on but starved" was
   indistinguishable) and `Lidar Image Az-Grad Energy` — mean |dI/dcol| over
   valid pixel pairs of the full-res image, i.e. how much along-tunnel texture
   the channel actually carries this scan (~0 = image terms cannot help;
   graffiti walls should read O(0.01–0.1) in /scale units).

Also fixed in passing: a flow-only config (`flow/enabled` without
`lidar_image/enabled`) previously built no image and silently no-oped — the
image build and the reflectivity channel copy now trigger for either.

## What this does NOT claim

- No bag validation yet: correctness of the new reference paths on real data
  (sign, occlusion behavior at patch edges, LUT rows) is a harness question —
  smoke-test flow imageRef+patch at low weight first, as with the original
  flow term (doc/LIDAR_FLOW_TERM.md).
- The keyframe refs inherit the keyframe image's viewpoint: a map point seen
  from far away stores a coarser sample than the current scan sees up close.
  View-dependence of reflectivity (incidence angle) is unmodelled, same as
  the field reference it replaces.
- The 3D photometric term is OFF in the tunnel overlay, not removed: it may
  still be right for feature-rich, non-degenerate environments where the
  voxel average is representative.

## Validation plan

On the 06042026 substrate (n ≥ 24, same harness verdicts):
- Arm A: combo (xicp overlay) as merged — image terms off.
- Arm B: A + lidarimg overlay (map term + imageRefs on).
- Arm C: B + flow overlay (imageRef true, patch 2, weight 0.02).
Watch DIV rate, slosh amplitude/period, `Lidar Flow Points` engagement, and
`Lidar Image Az-Grad Energy` (confirms the substrate actually carries texture
where the tunnel is; if it reads ~0 on the walls, the graffiti hypothesis
fails and the image terms are dead weight). A/B the reference modes
(`flow/imageRef false`, `imageRefs false`) to isolate the pre-voxel effect.
