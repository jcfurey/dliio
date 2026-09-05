# Mapping foundation validation — 2026-09-05

The persistent mapper was added on `cam-dev`, starting from handoff commit
`412148b`. Odometry math and the 0705 registration profile were unchanged.
Tests used x86-64 Ubuntu/ROS installations with Cyclone DDS, Release builds,
four OpenMP threads, one OpenBLAS thread, and `DLIIO_ENABLE_LIVOX=ON`.

## Compatibility and logic

Humble, Jazzy, Kilted, and Lyrical each passed **322 actual test cases**:
259 C++ cases and 63 Python cases. Colcon reports 343 when its 21 CTest wrappers
are counted too. New coverage checks local/world transforms, per-channel voxel
means, exact pairing, endian/padded ROS cloud layouts, memory eviction, archive
round trips, corrupt records, no-overwrite output, and transaction rollback.

Every distribution also passed installed ROS service checks: 12 synthetic
keyframes formed six submaps, with only two resident. Save/reload preserved the
window exactly; full export contained all 24 points. Rejected loads preserved
the current map, late subscribers received the latched map, live configuration
changes were rejected, and input-gap/duplicate/wrong-frame diagnostics worked.
The final service scheduling uses one archive callback at a time, leaving
executor threads available for input and publication; all four service checks
were repeated after that change.

The installed legacy component/Livox smoke also passed on all four versions,
including actual CustomMsg adapter conversion and malformed point-count
rejection against the repository's interface-only Livox fixture. No Livox node
or hardware was required. The new mapper uses the Humble-compatible parameter
prefix API and avoids reserved rclpy Node attributes.

## Packet replay at 1×

The installed packet launch, Ouster driver, mapper, and archive services were
exercised on every distribution. The short checks sample at least 75 seconds
after the first registered scan and wait for sampled keyframes to commit before
saving. Their final counts include the short service/drain interval.

| ROS | Paired registered scans | Archived keyframes | Frontend reported avg/max ms | Mapper observed max worker ms |
|---|---:|---:|---:|---:|
| Humble | 752 | 10 | 14.36 / 44.97 | 21.20 |
| Jazzy | 753 | 9 | 14.41 / 45.14 | 25.02 |
| Kilted | 752 | 9 | 13.19 / 45.19 | 23.21 |
| Lyrical | 753 | 9 | 11.65 / 42.87 | 25.51 |

All short runs reported zero estimated frontend scan drops/compute overruns and
zero mapper missing pairs, queue drops, or processing errors. A keyframe arriving
after the sampling boundary can appear in the final observer count without
belonging to that saved prefix. These are runtime checks, not controlled timing
comparisons among distributions; other replay work ran on the same host.

## Complete 0705 run with RViz

The complete 552.61-second bag played at **rate 1.0** on the Lyrical host:

- 5,493 registered scan clouds paired exactly with 5,493 registered poses.
  The launch performed its configured three-second IMU startup calibration;
  these are published ROS outputs, not the 5,511-scan offline cache count.
- All **32 published keyframes** and their exact poses were archived, matching
  the frontend's final keyframe count. No mapper input gaps or queue drops were
  reported, and its job queue peak was one.
- Seven submaps were archived while only four remained resident. The live
  window ended at 13,525 points and peaked at 16,511 points during the run.
  Full export contained **23,077 points**, including evicted submaps.
- Mapper peak RSS was **89.3 MiB**. Its maximum recorded keyframe worker time
  was **121.55 ms**, including voxel processing, storage, and snapshot creation.
  This worker handles keyframes separately from the 10 Hz odometry callback.
- The frontend reported **one cumulative compute overrun** and **zero estimated
  dropped scans**. Its final bounded-history average/maximum callback times
  were 10.32/59.34 ms; those final values are not all-run timing extrema.
- Saving the archive, reloading it, and exporting again produced byte-identical
  full binary PCDs. SHA-256:
  `08df247fa3e1a9e2263dc0879570992fa7a523c661b5f726389f8d78c853470e`.
- RViz displayed the reflectivity map with Global Status OK. A separate
  installed viewer launch also loaded an actual Ouster snapshot and published
  its map and identity `map -> odom` transform with no odometry or driver node.

The archive retained 38,339 processed local keyframe points, rigid poses,
timestamps, configured extrinsics/frame/IMU parameters, and separate signal,
reflectivity, corrected-intensity, and image-channel values. Estimated runtime
IMU calibration results and raw per-return Ouster metadata are outside the
current keyframe interface.

## Scope

This validates the dedicated mapper's storage, delivery, bounds, and ROS
compatibility. Its memory budget is separate from odometry's existing internal
keyframe-map budget. The archive grows on disk. Loop closure, optimized pose
revisions, global overlap fusion, relocalization, and acknowledged frontend
keyframe IDs remain future work. This pass does not establish metric accuracy
or prove that tunnel-axis drift is eliminated; that requires separate repeated
accuracy evaluation and suitable ground truth.

Local evidence lives in the parent workspace's
`results/dliio_mapping_2026-09-05/`: per-distribution build/test/service logs,
`*/replay3/replay.json`, `full-0705-v2/replay.json`, saved archives and PCDs,
`full-0705-v2/rviz-live.png`, and `viewer-launch/probe.json`.
