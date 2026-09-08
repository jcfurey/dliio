# Automatic geometric loop closure

The optional loop worker proposes revisits from committed mapping observations,
validates them and submits transactional graph updates to the persistent mapper.
The mapper remains the sole database writer and map-to-odom authority. Local
odometry is continuous; historical map geometry and its published trajectory
follow the optimized observation poses.

Enable this explicitly with an assumed-noise configuration. The example values
are engineering assumptions, **not calibrated pose covariance or a guarantee
of correct place recognition**. An ideal corridor cannot constrain all six
pose axes; such candidates are rejected. Do not flatten a tunnel or insert an
identity endpoint constraint to make a trajectory look closed.

## Live integration

Use the sensor-independent launch with your robot calibration and input topics:

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  robot_config:=/absolute/robot.yaml \
  pointcloud_topic:=/sensors/points imu_topic:=/sensors/imu \
  mapper:=persistent archive_directory:=/absolute/new-run \
  loop_closure_config:=/absolute/reviewed-loop-closure.json rviz:=true
```

Start with [`cfg/loop_closure.json`](../cfg/loop_closure.json) as a documented
example to review for the platform. Empty `loop_closure_config` disables the
worker. The launch requires persistent mapping and increases its bounded
backlog to 64 observations while graph/map rebuilds run. This is a backlog
bound, not a lossless transport claim: inspect `queue_dropped` and
`source_sequence_gaps` in mapper diagnostics.

The worker's parameters are `loop_closure/configuration_file` and
`loop_closure/output_directory`. Topics/services are relative to its ROS
namespace, matching the generic mapper. Diagnostic status `DLIO Loop Closure`
reports work, errors, checked observation count/revision and `loop_idle`.
Diagnostics and work continue after simulation `/clock` stops. Registration
can lag playback; EOF alone does not mean graph work has finished.

Registration now defaults to the native C++ `PreparedLoopCloud` engine in
`_dliio_pose_graph`. It owns immutable coordinates, trees and normals at two
voxel resolutions and reuses them across initial guesses and the reverse
check. It releases the GIL so the bounded candidate workers can run in parallel.
It uses existing Eigen/nanoflann dependencies and requires no CUDA device.
`retrieval.registration_backend: "python"` selects the retained reference;
`"native"` selects the default explicitly. Reports include the selected backend,
preparation seconds and fitting seconds. Registration remains a proposal: it
does not estimate calibrated noise or skip any admission checks below.

Each pass checks the newest two unchecked, mature motion anchors, then works
back through older anchors. Anchor selection uses original motion and quality,
so graph corrections cannot reshuffle the sampling history. The worker reports
`loop_pending_anchors`; an idle result covers the current observation count and
revision, or explicitly reports the configured loop capacity. This prioritizes
recent revisits without claiming that registration always keeps pace with input.

After stopping sensor input, an automated runner can wait for completion:

```bash
ros2 run direct_lidar_inertial_odometry wait_for_mapping.py \
  --loops --timeout 1800 --report /absolute/new-run/drain.json
```

It requires an idle loop worker covering the mapper's current observation
count and pose revision, an empty mapping queue and a refreshed map cache.
Timeout/error returns nonzero. Use matching ROS namespace remapping if needed.

## Saved archives

Source a built workspace and use a sealed `save_map` snapshot. Inputs remain
unchanged; destination/report paths must be new:

```bash
ros2 run direct_lidar_inertial_odometry mapping_loop_closure.py \
  /absolute/original.dliomap /absolute/corrected.dliomap \
  --configuration /absolute/reviewed-loop-closure.json \
  --report /absolute/loop-report.json
```

`--proposals-only` writes a proposal report and trial trajectory without
creating a corrected archive. Reports contain source/configuration/code hashes,
candidate rejection metrics, support observation IDs, transforms and explicit
assumed noise. An empty accepted list means no correction. Retain the report
with the archive; a successful process exit alone does not establish closure.

## Admission and uncertainty

1. Time, path distance and spatial proximity propose bounded candidate sets.
   Proximity is not proof of a revisit. Candidates use immutable original
   base-local observations; corrected fused maps cannot validate themselves.
   The example gathers up to 12 temporally separated places, ranks them by
   geometrically consistent FPFH matches, and fits three candidates, including
   the nearest spatial candidate. With no feature evidence it falls back to
   spatial ordering. `candidate_pool` is bounded between `candidates` and 32;
   feature ranking selects work and never admits a factor by itself.
2. Each visit has distinct fitting and held observation IDs. The windows use
   original relative poses. Failed registrations cannot supply loop anchors or
   support. Adjacent held scans share sensor/systematic errors, so “held” does
   not mean independent ground truth.
3. PCL FPFH correspondences generate bounded deterministic RANSAC initial
   guesses. Centroid/trajectory guesses also test competing basins. A symmetric
   robust point-to-plane fit weights both viewing directions equally and
   differentiates the moving source normal in the reverse terms.
4. The existing graph backend reconstructs both support sets and checks
   bidirectional overlap, absolute point residuals, distinct normal support,
   six-axis observability and discrepancy from the current graph. Query
   sampling is bounded separately from reference surfaces, so different query
   sampling phases do not manufacture missing geometry. Held refitting and
   competing-place checks further reject unstable or ambiguous solutions.
5. Every accepted candidate is trial-solved with all existing factors. The
   native solver's convergence, unrobustified residual, absolute loop residual,
   motion and total-displacement gates must all pass. A batch is committed in
   one revision or rejected entirely. There is no accepted-prefix mutation.
   A failed graph trial does not prevent trying another unambiguous,
   geometrically validated candidate for that observation. Rejected hypotheses
   and their fitted transforms/support remain in the proposal report.

Single-observation validation retains its default 3 m discrepancy gate. The
example explicitly gives multi-observation retrieval a wider 18 m search and
discrepancy envelope, with at least 60% bidirectional overlap and a 0.001 observability-ratio floor.
The example opts into `surface_validation`: point-distance p95 is bounded by
0.35 m, then both viewing directions must provide compatible planar normals
(dot magnitude at least 0.95), at least 15% surface support and 100 distinct
reference matches. Each direction must have surface p95 at most 0.10 m, RMS
at most 0.05 m and full six-axis rank. A linearized refit must move less than
0.10 m / 0.02 rad. These checks apply separately to fitting and held windows.
They distinguish tangential sampling distance from displacement through a
wall, while rejecting low-residual poses supported by biased corner normals.
Legacy validation keeps 65% overlap and the 0.20 m point gate; it does not
enable this mode.
Review the explicit assumptions for other scenes.
The geometric spectrum checks rank; it is not inverted into pose covariance.

The default graph still rejects unconverged odometry. The optional
`bounded_motion_prior` failure policy retains the original failure flags and
adds an explicitly assumed covariance floor to edges touching those failures.
It bounds failed runs and original/optimized translation and angular rates.
This permits graph continuity through bounded bad registrations; it does not
promote those registrations into measured six-axis constraints. Unknown
registered-pose covariance stays unknown. Loop/odometry weighting and their
independence assumptions are recorded with the graph. The example uses a
0.0005 rad rotational standard-deviation floor, 0.002 rad/√s and
0.0005 rad/√m relative-increment terms. This stronger rotational weighting
avoids a large sideways bow observed with the initial loose-noise example;
it remains empirical regularization. There is no absolute graph gravity prior.
Evaluate whole trajectories as well as endpoint and factor residuals.

## Revisions, display and recovery

`dlio/mapping/path` is a latched optimized historical trajectory.
`dlio/mapping/snapshot` carries one committed map cloud, session/revision, and
matching original/optimized pose paths. Historical consumers must use these
image/observation-time corrections; the latest map-to-odom TF alone cannot
describe a nonuniform historical deformation. A revision can change while the
last observation timestamp stays constant at EOF.

The worker copies bounded graph metadata in a short SQLite transaction, then
releases the disk read lock before registration. Original cloud blobs are read
on demand in autocommit mode and must match the frozen count/checksum. Mutable
submaps and optimized poses are never read lazily. The recorder can checkpoint
its WAL while loop fitting runs. It saves exact service request IDs/bodies and retries
timeouts with the same payload. Existing/removed loop IDs remain reserved.
Session changes, detached graphs and configuration mismatches require attention
instead of silently attaching a different noise model. Retained batches remain
reversible through the factor-removal interface in [POSE_GRAPH.md](POSE_GRAPH.md).

Pose revisions rebuild only storage groups whose optimized pose bytes change.
An odometry-only initialization preserves the exact original pose bytes after
checking the native solution, avoiding a dense-map rewrite for numerical
roundoff. Actual loop deformations still rebuild the affected original geometry
transactionally, with the same revision history and rollback semantics.

Bounds include 5,000 graph observations, at most 64 proposed loops per batch,
bounded per-visit support and reference/query point counts. This is a batch
reference implementation, not a guaranteed real-time large-map backend.

PCL documents the [FPFH descriptor](https://pointclouds.org/documentation/tutorials/fpfh_estimation.html)
and [feature-based alignment with geometric rejection](https://pointclouds.org/documentation/tutorials/alignment_prerejective.html).
This implementation uses PCL descriptors with its own bounded correspondence
sampling, prior-orientation gate and subsequent archive validation.

Registration retains complete bounded reference surfaces and limits each fit
iteration to 6,000 queries per direction. `registration_workers` controls 1–4
concurrent numeric fits (example: two); SQLite access stays on its owner thread
and results are consumed in deterministic candidate order. The example checks
anchors at least five seconds and two metres apart. That is a retrieval
sampling policy, not a guarantee that every possible revisit is examined.
