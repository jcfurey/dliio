# Persistent mapping

`dlio_mapping_node.py` is a dedicated mapper in this package. It pairs each
registered keyframe cloud with its exact registered pose, builds bounded voxel
submaps, and saves local keyframes and poses to a versioned archive. Geometry
and disk work run in a worker in a separate process from odometry. NumPy and
SQLite are the only additional implementation libraries; there is no external
mapping, transform, database, or fusion node.

This is the mapping foundation. It does not yet detect loop closures, optimize
poses, relocalize, or correct odometry drift. `map -> odom` is an identity static
transform. Do not run another publisher for that transform with this mapper.

## Run

Build using [HANDOFF.md](HANDOFF.md), then source the install. All examples work
with Humble, Jazzy, Kilted, and Lyrical. Livox remains enabled by default.

```bash
ros2 launch direct_lidar_inertial_odometry dlio_mapping.launch.py \
  mode:=packets profile:=0705 bag:=/absolute/path/07052026_4_an \
  rate:=1.0 rviz:=true run_dir:=/absolute/path/dliio_run
```

For a live Ouster driver, use `mode:=points robot_config:=/path/my_robot.yaml`.
The arguments are the same as `dlio_ouster.launch.py`. That original launch
retains its preview default; `mapper:=persistent` selects this mapper directly.
`map:=false` omits either mapper. The legacy C++ preview and persistent mapper
must not publish to the same map topic concurrently.

`mapping_config:=/path/mapping.yaml` supplies mapping settings. By default the
archive directory is `run_dir/mapping`; override with `archive_directory:=...`.
Each recording creates a new `session-<uuid>.dliomap` working database and stores
its own session ID. Restart odometry and mapping when replaying or seeking
backward. A loaded archive opens as a read-only viewer, never as an append target
for a new, unaligned odometry session.

## Save, export, and reload

Use absolute filenames in existing directories. Outputs are installed atomically
and never overwrite existing files. For a consistent end-of-run comparison,
wait for playback to finish and the mapper's `queue_depth` and `pending_pairs`
diagnostics to reach zero. Save and export execute on the storage worker; long
operations during live input can fill its bounded queue. Pause playback or run
them at the end when retaining every delivered keyframe matters.

```bash
ros2 service call /dlio/mapping/save_map \
  direct_lidar_inertial_odometry/srv/MapArchive \
  "{path: '/absolute/path/dliio_run/tunnel.dliomap'}"

ros2 service call /dlio/mapping/export_pcd \
  direct_lidar_inertial_odometry/srv/MapArchive \
  "{path: '/absolute/path/dliio_run/tunnel.pcd'}"
```

Both operations include **all archived submaps**, including those evicted from
RAM. The archive additionally retains every accepted local keyframe and pose.
PCD export streams submaps and does not allocate the entire map. Voxel means
combine all samples within a submap; overlapping voxels from different submaps
remain separate PCD samples. There is no global overlap correction yet.

After stopping the recording launch, open the saved archive without odometry,
Ouster, or a bag player:

```bash
ros2 launch direct_lidar_inertial_odometry dlio_map_viewer.launch.py \
  load_map:=/absolute/path/dliio_run/tunnel.dliomap rviz:=true
```

The viewer publishes the configured recent submap window. Increase
`mapping/resident_submaps` in a copied mapping YAML to view more submaps, within
the supported cap of 64; the full PCD remains available regardless of that
window. Custom frames require matching `map_frame` and `odom_frame` viewer
arguments and RViz's Fixed Frame. The viewer's `/dlio/mapping/load_map` service
can switch to another sealed archive with those same frames. Validation finishes
before replacing the current map; a rejected load leaves the current map intact.

Offline inspection, crash recovery, and export need Python and NumPy but no ROS
graph. These commands also work directly from `scripts/mapping_archive.py`:

```bash
ros2 run direct_lidar_inertial_odometry mapping_archive.py inspect /path/tunnel.dliomap
ros2 run direct_lidar_inertial_odometry mapping_archive.py snapshot \
  /path/session-working.dliomap /absolute/path/recovered.dliomap
ros2 run direct_lidar_inertial_odometry mapping_archive.py export \
  /path/recovered.dliomap /absolute/path/recovered.pcd
```

The working archive uses SQLite WAL transactions. Keep its `.dliomap`, `-wal`,
and `-shm` files together while it is open; copying only the main file can omit
recent committed data. `save_map` and the offline `snapshot` command use the
[SQLite backup API](https://www.sqlite.org/backup.html) to produce a sealed,
self-contained file. A crash can lose a transaction that has not committed;
committed keyframes, their submap updates, and metadata form one transaction.
The viewer rejects unsealed working files to avoid reading a changing archive.

## Coordinates, fields, and archive schema

The input contract is [MAPPING_INTERFACE.md](MAPPING_INTERFACE.md). Input XYZ is
already in `odom`; the mapper applies the inverse **paired keyframe pose** once
to obtain base-frame keyframe points. Each submap is anchored at its first
keyframe pose. It averages points in that anchor frame and transforms its voxel
centers to `map` when publishing or exporting.

Every stored point has seven little-endian float32 values: `x`, `y`, `z`,
`intensity`, `reflectivity`, `intensity_corrected`, and `lidar_intensity`.
Missing scalar channels remain NaN and are excluded from that channel's voxel
mean; missing reflectivity is never invented from intensity. Invalid XYZ is
filtered before ingestion. These are processed keyframes, not lossless raw
Ouster returns: ring, ambient/near-IR, range, and acquisition offsets are not
present on the frontend's processed keyframe topic.

Version 1 uses three SQLite tables:

| Table | Retained data |
|---|---|
| `metadata` | Format/version, session UUID, configured `extrinsics/*`, `imu/*`, and `frames/*` parameters, fields, counts, voxel size, identity map-to-odom transform, pose revision 0 |
| `keyframes` | Contiguous stable ID, exact integer nanosecond stamp, submap ID, registered base pose in odom, local cloud, point count and CRC32 |
| `submaps` | Stable ID, anchor pose, first/last keyframe IDs, latest measurement stamp, voxel centers, point count and CRC32 |

Poses are float64 rigid 4x4 matrices. Reload checks the SQLite format and
integrity, sizes before fetching point blobs, CRCs, proper rotations, finite
XYZ, timestamp ordering, contiguous IDs, submap membership and anchor poses,
and metadata consistency. It streams validation instead of loading the entire
keyframe catalog into RAM. Archives whose per-frame/submap sizes exceed the
configured input/voxel caps are rejected; the offline CLI exposes those same
read caps as `--max-input-points` and `--max-voxels`.

The frontend's runtime IMU bias and attitude calibration results are not
published by this input interface and are not part of that parameter snapshot.
The archive retains the resulting registered poses and deskewed local clouds.

## Bounds and diagnostics

Defaults in `cfg/mapping.yaml`:

| Setting | Default | Effect |
|---|---:|---|
| `mapping/voxel_size` | 0.2 m | Resolution within each submap |
| `mapping/submap_keyframes` | 20 | Maximum keyframes per submap |
| `mapping/resident_submaps` | 8 | Recent submaps retained for live display |
| `mapping/max_voxels` | 100,000 | Maximum voxels per submap |
| `mapping/max_input_points` | 200,000 | Maximum points per input keyframe |
| `mapping/max_cloud_bytes` | 16 MiB | Maximum serialized input cloud |
| `mapping/pending_pairs` | 8 | Incomplete timestamp pairs retained |
| `mapping/pair_timeout` | 2 s | Wall-clock wait for the matching message |
| `mapping/worker_queue` | 4 | Maximum queued geometry/service jobs |
| `mapping/publish_rate` | 1 Hz | Maximum changed-map publication rate |

A submap closes early when adding a keyframe would exceed its voxel cap. A
single frame that exceeds the cap is rejected. Eviction removes voxel arrays
from RAM while preserving all records on disk. The active accumulator, recent
submaps, input queues, and serialized map are bounded independently of total
run length. NumPy temporaries, ROS/DDS buffers, interpreter memory, and an 8 MiB
SQLite cache add overhead: the resident-array diagnostic is **not total RSS**.
Disk use grows with accepted data; archive retention is an operator decision.

These bounds apply to the dedicated mapper. Odometry's internal registration
keyframe map has a separate `odom/keyframe/maxKeyframes` setting, whose existing
default is 0 (unlimited). This change does not retune that estimator budget or
claim a bound on the combined memory of every process in the launch.

`/dlio/map_node/map` is a reliable, transient-local PointCloud2 in `map`, stamped
at the last accepted keyframe measurement. It retains one latest message for
late subscribers and sends a changed snapshot at the configured rate. Publication
and pair expiry use wall time, so a paused bag still permits save/reload/display.

`/dlio/mapping/diagnostics` reports accepted keyframes/submaps, resident points and
array bytes, cached message bytes, queue/pair occupancy, processing time, peak
process RSS in KiB, and explicit rejected/missing/duplicate input counters.
Storage I/O errors stop further recording and set ERROR. Configuration and
calibration changes require a node restart so recorded metadata stays accurate.

Cloud and pose topics are separately delivered. Subscribe before replay. A
known incomplete pair blocks newer pending pairs until it completes or expires;
expiry, overflow, duplicates, and backward stamps are reported. Losing **both**
messages cannot be detected from this interface alone, because the frontend
does not publish an acknowledged keyframe sequence ID. Frontend registration
quality and calibrated pose uncertainty also need a richer future input message.

## Verification

```bash
# In an unused ROS domain; no hardware required:
ROS_DOMAIN_ID=157 ros2 run direct_lidar_inertial_odometry verify_mapping.py
# Real packet input, forced submap eviction, and complete export/reload comparison:
ROS_DOMAIN_ID=157 ros2 run direct_lidar_inertial_odometry verify_mapping_replay.py \
  /absolute/path/07052026_4_an --full --rviz \
  --output /absolute/path/new-run/mapping-replay.json
```

The second check uses a five-keyframe/four-submap window to exercise eviction.
Without `--full` it takes a 75-second sample. `--keep-running` leaves a successful
launch open and records its PID beside the report. Tests verify mapping logic
and runtime delivery; they do not establish trajectory accuracy without ground
truth. Loop-closure work can build on local keyframes and stable archive IDs,
with conservative geometric/texture verification and explicit pose revisions.
Recorded results are in [MAPPING_VALIDATION.md](MAPPING_VALIDATION.md).
