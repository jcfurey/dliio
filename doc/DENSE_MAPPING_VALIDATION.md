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

The final stream/cache change has not yet repeated the four-distro matrix or
a complete 0705 playback. Visual confirmation remains open. Next session should
compare `mapping_input:=observations` with `mapping_input:=keyframes`, using
only the persistent map display, before starting loop-closure work.

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
