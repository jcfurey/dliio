# Dense mapping and overlap fusion validation — 2026-09-05

This checkpoint preserves full-resolution deskewed mapping observations and
unvoxelized source submaps. A shared 2 cm world grid combines overlapping
samples. The initial dense-keyframe and grid-only implementation still showed
visible density/color patches in RViz; those patches are not declared solved.
The final change adds independently selected mapping observations and an
incremental, bounded fusion cache.
See [MAPPING_NODE.md](MAPPING_NODE.md) for the controls and limits.

## Regression and installation checks

Before the final observation-stream/cache change, Humble, Jazzy, Kilted,
Lyrical, and the Lyrical host passed **364 test cases**
(262 C++ and 102 Python). Colcon reports 385 when its 21 suite wrappers are also
counted. Each isolated distro also passed installed-node, Livox adapter, and
mapping save/load/export checks. `DLIIO_ENABLE_LIVOX=ON` throughout.

New coverage includes frozen dense keyframe poses/timestamps/corrections,
retaining sparse registration history, initial keyframe publication, optional
sub-floor rejection, coincident returns with distinct scalar values, raw and
voxelized archive invariants, global fusion across submap boundaries, weighted
per-field means, missing channels, and cleanup after failed fused exports.

## Packet replays on all four distros

Each run used the real 07052026_4_an Ouster packet bag at 1× with the packaged
0705 profile. The verification window starts after the first registered scan
and lasts 75 seconds. A five-keyframe/four-submap setting exercises the same
storage path as the full replay. Containers used separate ROS domains and
four assigned logical CPUs each, with OMP=4 and OPENBLAS=1.

| ROS | Sampled scan pairs | Archived returns | Fused export points | Max mapper job (ms) | Peak mapper RSS (MiB) | Frontend rolling average (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Humble | 750 | 225,930 | 158,723 | 274.6 | 188.9 | 14.54 |
| Jazzy | 750 | 220,612 | 157,947 | 318.8 | 206.1 | 14.38 |
| Kilted | 751 | 222,817 | 158,516 | 243.9 | 189.1 | 13.97 |
| Lyrical | 751 | 242,295 | 173,436 | 310.4 | 213.1 | 13.34 |

Every sampled scan pose had its cloud. All accepted keyframes retained every
received point and bit-identical scalar channels in their local archives.
All mapper drop/error counters and frontend cumulative compute-overrun and
estimated-scan-drop counters were zero. Service calls during a partial replay
can include additional frames after the sampled boundary; counts are reported
separately rather than treated as loss.

The initial dense runs exposed 1–3 missed every-scan preview messages on
Humble/Jazzy with the original depth-one publisher. Mapping keyframes were
complete. Increasing that scan publisher's bounded history to ten eliminated
those misses in the final four-distro replay checks. The probe also freezes
its sampled boundary before waiting for separately delivered clouds.

## Density and overlap measurements

A complete initial dense replay with RViz retained **730,666 individual
returns in 31 keyframes**, and exported/reloaded all of them without fusion.
Paired keyframes had a median **19.31×** as many points as their registration
clouds. This is a same-stamp point-count comparison, not a trajectory comparison.

Offline fusion of that archive produced 507,238 points at 2 cm (30.6% fewer
samples), versus 688,680 at 1 cm and 728,765 at 5 mm. All 730,666 original
samples remain in the archive. The shared grid combines nearby observations
across submaps; it does not identify semantic landmarks or optimize poses.

The complete grid-only replay retained 728,605 returns in 31 keyframes and
exported 505,869 fused points. All 5,491 registered scan poses had their cloud;
all mapper drop/error counters were zero. Peak mapper RSS was 339.3 MiB and
maximum per-frame worker time was 536.2 ms. The frontend reported one cumulative
compute overrun and two estimated scan drops; delivery-pair completeness is
not evidence that every sensor scan was processed. The four shorter distro
runs above had neither of those frontend counters increment.

## Final observation-stream checkpoint

The grid-only archive had approximately 1.8 m between ordinary consecutive
odometry keyframes, with long periods receiving no new keyframes on revisits.
The dedicated observation stream instead selects from its last accepted view
at 0.20 m translation or 5 degrees rotation, subject to a 0.20-second minimum
interval and the existing keyframe trust vetoes. The clouds retain dense
resolution; 0.20 m is viewpoint spacing, not point filtering.

Fusion now adds new observations and subtracts evicted submaps incrementally,
reuses freed accumulator slots, and prepares snapshots at the publication rate.
Tests cover revisit/rotation selection, rate limiting, veto behavior, frozen
mapping output, unchanged registration history, channel means, eviction,
capacity reuse, leaf-size changes and agreement with independent full rebuilds.

The Lyrical host passed **372 test cases** (264 C++, 108 Python; colcon reports
393 including its 21 suite wrappers), with no failures, errors or skips.
Livox remained enabled. The initial final-stream replay exposed a probe issue:
live full-map export blocked the worker and dropped five queued observations.
The probe now pauses playback after the 75-second full-speed sample and before
save/export. This is also the documented procedure for costly live exports.
The failed run is retained under `observations-replay/`, rather than hidden.

The corrected Lyrical 75-second replay at 1× passed with **751 paired scan
outputs and 80 mapping observations**, compared with 10 odometry keyframes in
the same run. All 1,905,667 received mapping returns and their scalar channels
were retained; global export contained 600,954 fused points. Every mapping
cloud and pose paired, all mapper error/drop counters were zero, and the
frontend reported zero compute overruns and zero estimated scan drops.

Maximum ingestion time was 174.4 ms and maximum snapshot preparation time was
129.6 ms. Mapper peak RSS, including the export checks, was **835.1 MiB**; this
is a material cost of the denser archive and fusion cache. This short test used
two resident submaps, not a saturated eight-submap window. The installed
synthetic save/reload/export check also passed, including metadata retention
and failed-load preservation. Final evidence is in `observations-host/` and
`observations-replay-paused-export/`.

The continuation below repeats the final stream/cache revision's matrix.
The complete comparison below establishes improved density and reduced narrow
striping, while broader brightness patches remain. Loop-closure work remains
deferred until map quality is accepted.

## Continued four-distro validation

The final observation-stream/cache revision passed **372 actual cases per
distribution** on Humble, Jazzy, Kilted, and Lyrical, with zero errors, failures,
or skips. Colcon reports 393 including suite wrappers. Every installed Livox
adapter and mapping save/load/export check passed, with Livox ON throughout.

The 75-second 1× packet replays now use the default 100-observation/eight-submap
configuration and pause after sampling, before export:

| ROS | Scan pairs | Mapping observations | Archived returns | Fused points | Max ingestion (ms) | Max snapshot (ms) | Peak mapper RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Humble | 751 | 80 | 1,892,664 | 598,942 | 286.6 | 182.6 | 820.7 |
| Jazzy | 751 | 78 | 1,865,792 | 588,141 | 306.0 | 180.1 | 831.1 |
| Kilted | 751 | 79 | 1,886,486 | 592,001 | 322.4 | 190.9 | 794.2 |
| Lyrical | 751 | 79 | 1,889,082 | 589,380 | 250.3 | 187.1 | 788.3 |

All mapping pairs were retained with bit-identical scalar fields. Mapper
drop/error counters, frontend compute overruns, and estimated scan drops were
zero. These short runs reached only two submaps. Builds and installed checks
are in `<distro>/observations/`; packet evidence is in
`<distro>/observations-replay/`. CPU sets and ROS domains were separate, while
host memory bandwidth and other machine resources were shared. Complete host
comparison runs also ran during the short replay matrix; these timings are
measurements of that workload, not isolated worst-case latency guarantees.

## Complete observation/keyframe comparison at default limits

Both complete 552.607-second packet-bag replays passed at 1×, using the same
100-frame/eight-resident-submap limits and 2 cm output grid. The Lyrical host
used separate CPU sets (0–3 for observations, 24–27 for keyframes) and ROS
domains. RViz displayed only `/dlio/map_node/map`, with matched cameras,
one-pixel points, and fixed 0–100 reflectivity bounds. The observation RViz
subscriber ran during most of playback; the keyframe comparison viewer joined
near the end. This is not an isolated A/B performance benchmark.

| Measurement | Observation input | Keyframe input |
|---|---:|---:|
| Registered scan/cloud pairs | 5,493 | 5,493 |
| Archived mapping frames | 574 | 32 |
| Archived individual returns | 14,731,141 | 763,946 |
| Total / final resident submaps | 15 / 8 | 1 / 1 |
| Globally fused export points | 3,067,288 | 521,681 |
| Final live-window fused points | 1,877,080 | 521,681 |
| Peak resident source points | 7,900,724 | 763,946 |
| Peak mapper RSS, including export | 1,918.5 MiB | 665.6 MiB |
| Maximum ingestion job | 1,357.6 ms | 194.0 ms |
| Maximum snapshot preparation | 680.6 ms | 86.5 ms |
| Worker queue peak | 3 of 4 | 1 of 4 |

All received mapping pairs were archived with bit-identical scalar fields;
raw exports retained every accepted sample. Full fused exports matched the
independently reloaded archive exports byte for byte. Mapper drop/error counters,
frontend cumulative compute overruns, and estimated scan drops were all zero.
The observation run exercised eviction through 15 submaps while never exceeding
eight resident submaps. Its peak source-array allocation was 221,221,296 bytes;
fusion arrays added 570,425,344 bytes, and the largest serialized map added
60,158,868 bytes. RSS reached 1.87 GiB despite bounded source storage. Eviction
jobs exceeded one second; zero loss on this bag is not a worst-case throughput
guarantee. A 75-second run would not have exposed that cost.

Matched RViz screenshots show much denser coverage and less narrow scan/keyframe
striping. Broad brightness differences and overlapping geometry remain; visual
patching is **not solved**. The final resident observation map and full archive
cover different windows, so the supplementary projection uses both whole-map
exports. In the explicit lower-tunnel slab `z ∈ [-1.0, -0.35)` m, on one common
27 m × 3 m projection at 2 cm resolution, occupied cells increased from 39,273
to 139,944 (3.56×). This depends on the chosen slab, sampling, and poses and is
not a measure of surface completeness or trajectory accuracy.

The workspace helper `scripts/dliio_map_view_variation.py` compares views at
least 1 m apart. It estimates normals from 12 neighboring returns, accepts
nearest matches within 2 cm with a 30 cm neighborhood radius and a local
planarity check, and samples at most 2,000 points per selected view. A synthetic
planar signal with known cosine dependence recovered its expected correlation.
On real observations, 41,309 accepted planar matches had median absolute
reflectivity difference 4 and 90th percentile 15. Signed reflectivity change
correlated with estimated incidence-cosine change at **0.566** (keyframes:
12,708 matches, correlation 0.572). Material boundaries, nonidentical sampled
locations, pose error, range, and differing selected views are confounders.
This supports investigating view-dependent response, not applying an unvalidated
cosine correction or claiming that radiometry is the sole cause of patching.

Evidence: `full-observations-8/`, `full-keyframes-8/`,
`full-comparison-summary.json`, `quality-comparison/`, and
`view-variation-synthetic/`. The workspace helper
`scripts/dliio_compare_dense_maps.py` reproduces the fixed-slab and resource
plots. Runtime/default fusion behavior and the archived scalar contract were
not changed during this continuation.

## Intensity and reflectivity reviewed together

Following the user's concern about relying on reflectivity, the same complete
archives were analyzed in raw intensity, reflectivity, `intensity_corrected`,
and the denoised `lidar_intensity` image channel. The geometry and 41,309 planar
correspondences are identical across channels; each channel uses its own units.
Raw intensity already supplies inter-frame flow in the 0705 profile, while
reflectivity supplies spatial photometric gradients. Their numerical weights
have different scaling/terms and cannot be compared directly as confidence.

| Observation-input measurement | Raw intensity | Reflectivity | Denoised intensity (`lidar_intensity`) |
|---|---:|---:|---:|
| Median absolute matched difference | 482.0 | 4.0 | 461.3 |
| Median symmetric relative difference | 24.9% | 25.5% | 23.1% |
| Median absolute difference / pooled IQR | 0.447 | 0.364 | 0.440 |
| Signed change / incidence-cosine correlation | -0.238 | 0.566 | -0.273 |
| Signed change / log-range-ratio correlation | 0.159 | -0.432 | 0.185 |

The symmetric relative difference is `2*abs(a-b)/(abs(a)+abs(b))`, excluding
zero denominators. The IQR uses both matched-value populations for that
channel. These descriptors reveal different view responses without comparing
482 signal units directly with 4 reflectivity units or assuming either channel
is universally more reliable. Observations and keyframes select different
viewpoint intervals; this is not a controlled test of how sampling changes a
fixed physical pair.

All 14,731,141 archived `intensity_corrected` values equal raw intensity
exactly under this profile. The correction branch is inactive because spatial
photometric registration selects reflectivity. `lidar_intensity` differs from
raw intensity because the organized image path applies 3×3 denoising. Neither
is another independent measurement. Whole-export intensity/reflectivity
correlation is 0.836, consistent with substantial shared content.

The fixed-slab intensity and denoised-intensity plots show the same density
improvement and broad brightness patterns as the reflectivity view. Changing
the display channel alone does not resolve patching. Continue evaluating both
channels and geometry, with matched-view ablations before retuning their
registration/fusion contributions. No channel-weight change or radiometric
correction was inferred from these correlations.

RViz now exposes persistent-map raw intensity in the packet-replay profile.
The archive viewer names it `Map intensity`, uses fixed 0–4096 display bounds,
and provides camera/selection tools. Reflectivity remains separately available
at fixed 0–100. The updated YAMLs parsed and installed correctly, and an actual
intensity RViz subscription/render was inspected. These display-only edits
follow the completed runtime matrix; estimator and mapping code did not change.
The raw-intensity screenshot was taken after interactive camera/point-size
changes, so controlled comparisons use the fixed-coordinate plots.

Evidence: `quality-comparison/*-view-variation-all-channels.json`,
`channel-identity.json`, `fused-channel-summary.json`,
`lower-tunnel-intensity-comparison.png`, and
`lower-tunnel-lidar_intensity-comparison.png`, plus
`full-observations-8/rviz-intensity-final.png`. The comparison helper accepts
`--channel intensity`, `--channel lidar_intensity`, and an explicit
`--color-max`; both runs share that chosen channel scale.

## Water-reflection feasibility limit

The user then asked about water reflections and questioned feasibility. An
offline probe linked to the tested C++ library applied the current, disabled
`subFloorKeepMask` to all 574 archived observation clouds, centered at their
registered poses. This differs from applying the frontend filter to its
pre-registration prior cloud; it is a feasibility proxy, not an estimator
ablation or a water-detection benchmark.

At the current defaults, it rejected 16,521 of 14,731,141 returns (0.112%) in
432 frames. The largest per-frame removal was 232 points (0.61%). A sparse
synthetic layer below a supported floor was removed, but a dense 12-point
lower layer survived because it became the lowest supported height bin.
The real-data projection includes rejected candidates along apparent walls
and edges as well as below the floor. Those are potential false positives;
there are no water/ground labels to score precision or recall. Both intensity
and reflectivity overlap between retained and rejected populations.

Do not interpret this as successful water filtering. The existing heuristic
does not establish ground orientation/continuity or distinguish real negative
terrain from reflections. It remains disabled. Original maps and scalars
were preserved, with per-frame keep masks saved separately. A stronger method
needs supported surface geometry and evidence across viewpoints; missing floor
returns are a loss of geometric support, not something a rejection filter can
recover. The recorded single-return profile cannot supply dual-return cues.

The workspace helpers `scripts/dliio_subfloor_probe.cpp` and
`scripts/dliio_subfloor_feasibility.py` reproduce this check. Evidence is in
`water-feasibility/`, including the figure, masks, counts, synthetic controls,
and successful build log. The broader research context and links are recorded
in the workspace continuation review. This investigation did not alter
registration, observation selection, or fusion arithmetic.

## Evidence and limits

Workspace artifacts are under `results/dliio_dense_mapping_2026-09-05/`:
`<distro>/fusion/` contains build/test/installed checks,
`<distro>/fusion-replay/` contains grid-only packet replay reports and archives,
`full-0705/` is the initial unmerged dense run, and `full-0705-fused/` is the
complete grid-only fused replay. `overlap-analysis.json` records the offline grid
occupancy measurements. Raw and fused PCDs are separate outputs.

These checks establish output pairing, data retention, fusion arithmetic,
full-speed operation on this bag, and archive reproducibility. They do not
establish trajectory accuracy without ground truth. Registration, odometry keyframe selection, and loop-closure behavior were not
retuned. Mapping observation selection is separate. The map window remains
bounded by stored submap limits; NumPy fusion temporaries, DDS, and RViz add
memory beyond the stored-point-array count.
