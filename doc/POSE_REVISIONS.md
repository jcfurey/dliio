# Mapping observations, covariance, and pose revisions

This implements the first backend milestone: an atomic observation interface,
explicit covariance semantics, and reconstruction from corrected poses. An
external caller can supply corrections; the [pose-graph backend](POSE_GRAPH.md)
can now validate supplied loop measurements, optimize, and commit corrections
with its factor history. Automatic place recognition/registration remain
separate work; applying an external pose revision does not certify a loop.

## Observation delivery and uncertainty

The persistent Ouster launch now selects `mapping/transport: observation` and
subscribes to `/dlio/odom_node/mapping_observation`. Its
[`MappingObservation`](../msg/MappingObservation.msg) contains:

- The registered scan pose and full-resolution cloud in `odom`, with identical
  measurement timestamps and frame headers, captured before the publish worker.
- A frontend session UUID and monotonic observation sequence. The archive
  allocates its own contiguous IDs after successful ingestion and retains the
  frontend identity/sequence with each observation. A source restart is
  rejected until a new archive session is started. Sequence gaps are reported.
- Registration convergence and the counts of held translation/rotation modes.
- Covariance availability and a model/provenance label, with a full 6x6 matrix
  when available. The frontend currently reports **UNKNOWN**, because its
  observer state and covariance do not describe the registered scan pose.

The matrix convention is a **right/local SE(3) perturbation**:
`T_true = T_est * Exp(delta)`, ordered `[tx, ty, tz, rx, ry, rz]` in metres and
radians. This is explicitly different from the fixed/world-axis convention
of ROS `PoseWithCovariance`. The all-zero UNKNOWN wire payload is stored as
an absent covariance. It must never be used as a zero-noise graph factor.

[`uncertainty.py`](../scripts/dliio_mapping/uncertainty.py) validates full
covariances, transports inverse-pose uncertainty, and propagates relative-pose
uncertainty with translation/rotation coupling. Relative propagation requires
cross-covariance or an explicit assertion of independence. GTSAM conversion
reorders an already compatible local tangent; it does not silently convert
world-frame covariance. These utilities do not estimate or calibrate a new
registration noise model.

`mapping/transport: paired` retains the exact-stamp cloud/PoseStamped interface
for legacy bags and `mapping_input:=keyframes`. Its uncertainty is recorded as
UNKNOWN and its source sequence is unavailable. Select one transport per
recording; the mapper never double-ingests both representations.

## Corrected poses and map ownership

Version 2 introduced immutable original registered poses and base-local clouds;
version 3 additionally retains graph state and loop/request history.
Current optimized poses and correction history are stored separately.
`submaps` contains derived geometry and corrected anchor poses.

A revision supplies a complete ordered prefix of **archive** observation IDs
`0..N` and their `T_map_base` poses. It also identifies the archive session,
expected current revision, unique request ID, and correction provenance.
Unknown IDs, gaps, duplicates, malformed transforms, wrong frames/sessions,
stale revisions, and coverage older than the current optimized prefix are
rejected before publication.

If observations newer than N have arrived, their corrected poses use the
odometric tail:

```text
C = optimized_pose[N] * inverse(original_registered_pose[N])
corrected_pose[j] = C * original_registered_pose[j]   for j > N
```

New observations subsequently use the same correction until another revision
is committed. The original observer/odometry state is not changed.

For now each revision reconstructs **all chronological submaps** from original
local observations. This includes evicted submaps, individual corrections
inside a shared submap, pure rotations, and scalar-channel counts. It provides
a bounded-memory reference for later selective rebuilding. Per-submap capacity
limits remain enforced; a correction that exceeds them is rejected rather than
partially applied. Local samples are transformed and voxelized afresh; cached
centroids are not repeatedly corrected.

The pose tables, rebuilt geometry, and revision record commit in one SQLite
transaction. Resident/active accumulators and fusion caches change after that
commit. The ROS worker then prepares and swaps the map cloud and correction
together. The service acknowledges persistence; `cached_pose_revision` in
diagnostics identifies the revision prepared for map/TF publication. During
reconstruction the previous complete map and correction remain available.
Live input has a bounded queue and reports drops; this first implementation
does not promise lossless intake during a long full-map rebuild.

The mapper now publishes **dynamic `map -> odom`**, initially identity, at
20 Hz using the node's clock. A loaded viewer uses its archived correction.
`mapping/publish_tf: false` disables this authority when an external backend
owns that transform. Exactly one publisher should own `map -> odom`; dliio
continues to own local `odom -> base_link`. A global TF correction alone is
insufficient to reconstruct historical geometry.

## ROS services

The recording mapper exposes:

| Service | Purpose |
|---|---|
| `/dlio/mapping/apply_pose_revision` ([type](../srv/ApplyPoseRevision.srv)) | Commit an externally computed optimized pose prefix and rebuild maps |
| `/dlio/mapping/restore_pose_revision` ([type](../srv/RestorePoseRevision.srv)) | Restore an earlier trajectory as a new revision, retaining history |

Read the archive `session_id`, `pose_revision`, and `keyframes` from
`/dlio/mapping/diagnostics`. The session is the archive session, not the UUID
on incoming frontend observations. Pass `frame_id: map` (or the configured
map frame) and matching arrays of archive IDs and `geometry_msgs/Pose` values.
Requests are bounded to 100,000 poses.

Retries with the same request ID and identical payload return the existing
revision while it is current. A changed payload or superseded request ID is
rejected. If a service times out after execution started, it reports that the
operation may still finish; retry the same request to determine its outcome.
Rollback is a new revision and never erases previous records. Revision 0 means
the original odometric trajectory. Read-only viewers reject correction writes.

## Offline correction and reconstruction

The CLI works on a **new copy** and atomically publishes a sealed result. It
never overwrites its source or an existing destination. It can also snapshot
a working WAL archive and upgrade a version 1/2 archive in the temporary copy.

```bash
ros2 run direct_lidar_inertial_odometry mapping_archive.py inspect /data/original.dliomap
ros2 run direct_lidar_inertial_odometry mapping_archive.py revise \
  /data/original.dliomap /data/corrected.dliomap --revision /data/revision.json
ros2 run direct_lidar_inertial_odometry mapping_archive.py export \
  /data/corrected.dliomap /data/corrected.pcd --fusion-size 0.02
```

Example request format (replace the session and poses with the actual result):

```json
{
  "session_id": "00000000-0000-0000-0000-000000000000",
  "expected_revision": 0,
  "request_id": "solver-result-001",
  "reason": "verified external graph solution",
  "frame_id": "map",
  "poses": [
    {"id": 0, "pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
  ]
}
```

For rollback, use `restore` with a JSON request containing `session_id`,
`expected_revision`, `request_id`, and `target_revision`, and supply a new
destination archive. The CLI limits request JSON to 64 MiB.

## Verification

Verified on ROS 2 Lyrical on 2026-09-07:

- The package builds with Livox support and its existing C++ test targets.
- **190 test cases pass across eight suites:** mapping core/input/revisions/
  uncertainty, launch configuration, frontend output contracts, concurrency,
  and the synthetic frontend pipeline.
- Both `verify_mapping.py` (legacy transport and archive/viewer compatibility)
  and `verify_pose_revisions.py` pass against running ROS nodes on an isolated
  localhost DDS domain.
- The revision test includes forced process exit during an uncommitted rebuild;
  recovery restores the previous committed pose revision and map.

After sourcing ROS and the rebuilt workspace:

```bash
ROS_DOMAIN_ID=176 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  ros2 run direct_lidar_inertial_odometry verify_pose_revisions.py \
  --output /tmp/dliio-pose-revisions.json
```

The deterministic ROS check verifies atomic observation ingestion, unknown
covariance preservation, nonuniform corrections, evicted-submap export,
save/reload, dynamic TF, absence of a competing static TF, idempotent retries,
stale-request rejection, rollback, and equivalent offline-copy reconstruction.
It uses known synthetic transforms, not tunnel-accuracy claims.

Unit coverage in `test_mapping_revisions.py`, `test_mapping_uncertainty.py`,
and `test_mapping_input.py` includes transactional failure, abrupt-worker-exit
recovery from an interrupted rebuild, original-source
preservation, odometric tails, new observations after correction, weighted
active accumulators, capacity limits, legacy migration, source restarts,
covariance correlations, lever arms, and rotations at pi. C++ output-contract
tests verify the real frontend publisher's frozen snapshot, quality fields,
session/sequence, and UNKNOWN covariance representation.

The [graph milestone](POSE_GRAPH.md) now consumes these contracts, validates
externally registered loop candidates, optimizes, and inserts/removes factors
atomically with the map rebuild. Covariance calibration, automatic retrieval
and registration, and real tunnel replay remain required before claiming a
complete loop-closure backend.
