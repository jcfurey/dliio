# Pose graph and validated loop injection

The mapping worker can now build a GTSAM Pose3 graph, validate externally
registered loop candidates, optimize, and rebuild its map from corrected
individual observation poses. Loop insertion/removal, solution poses, audit
records, map caches, and the map-to-odom correction commit in one SQLite
transaction. The frontend's continuous local odometry is independent.

This is the odometry-graph/loop-injection milestone. Automatic place retrieval
and coarse/fine registration are not connected. A caller supplies `Z_ij` and
its full covariance plus external verification provenance. The extra geometry
and consistency gates below do not establish place identity in repetitive
scenes; a repeated tunnel bay can remain geometrically convincing. Real
positive/negative revisit replay and covariance calibration remain necessary.

## Pose and noise contracts

Graph node IDs are **committed archive observation IDs**, contiguous from zero.
They are not frontend sequence numbers or odometry keyframe IDs. Every archived
observation participates; there is no lossy node-to-dense-cloud interpolation.
The initial numerical gauge is the first original registered pose. No gauge
prior or graph marginal is published as a calibrated world-pose covariance.

An edge from `i` to `j` measures `Z_ij = inverse(T_i) * T_j`: it transforms points
from observation **j's local base frame into i's local base frame**. Public
covariance ordering is `[tx, ty, tz, rx, ry, rz]`, metres/radians, for the right
perturbation `Z_true = Z_est * Exp(delta)`. All cross terms are retained. The
native boundary reorders the full matrix for GTSAM. Fixed/world-axis ROS pose
covariance requires a frame/Jacobian conversion before this boundary.
If reversing a supplied measurement to order its IDs, invert the transform
and propagate its covariance with `uncertainty.inverse_covariance`; swapping
the IDs and reusing the matrix is incorrect when the adjoint has a lever arm.

The residual is explicitly `Log(inverse(Z_ij) * inverse(T_i) * T_j)`, including
the Logmap derivative. This avoids dependence on GTSAM's optional Pose3 chart
or fast BetweenFactor Jacobian approximation. See the upstream
[Pose3 API](https://gtsam.org/doxygen/a03727.html) and
[factor-graph tutorial](https://gtsam.org/tutorials/intro.html).

Registered-pose covariance is currently UNKNOWN. The graph never substitutes
its zero wire payload, observer-state covariance, or a sum of correlated
pose marginals. Initialization requires an explicitly **assumed relative
odometry process-noise model**, in the same local tangent:

```text
Sigma_ij = covariance_floor
         + delta_seconds * covariance_per_second
         + norm(translation(Z_ij)) * covariance_per_metre
```

The floor must be positive definite; the two rates must be positive
semidefinite. Off-diagonal terms are supported. A configured time-gap ceiling
rejects unsupported missing-data intervals. This model assumes independent
relative increments for graph weighting; it does not make actual LIO errors
independent or calibrated, and does not infer degenerate directions from the
frontend's mode counts. Explicitly unconverged observations reject a solve;
unknown quality and degenerate-observation counts are reported. Loop covariance must be positive definite and carry
`assumed`, `conditional`, or `calibrated` plus a model description. UNKNOWN,
zero, asymmetric, indefinite, and nonfinite matrices are rejected.

## Request interface

The live worker exposes `/dlio/mapping/update_pose_graph` using
[`UpdatePoseGraph`](../srv/UpdatePoseGraph.srv). `request_json` is a JSON object
bounded to 1 MiB. All actions require the archive session UUID, current pose
revision, a unique request ID, and an action. Diagnostics expose `session_id`,
`pose_revision`, `graph_initialized`, `graph_attached`,
`graph_covered_observations`, and `active_loops`.

| Action | Additional fields | Result |
|---|---|---|
| `initialize` | `configuration` | Establish the odometry chain at pose revision zero; apply its first solution |
| `add_loop` | `loop` | Validate a new pair, trial optimize all active factors, commit on success |
| `remove_loop` | `loop_id`, `reason` | Retain removal history, solve remaining factors, rebuild map |
| `optimize` | none | Include newly ingested observations or explicitly reattach after an external pose revision |

Initialize once before applying external pose revisions. An external
`apply_pose_revision` or `restore_pose_revision` detaches the graph solution;
it retains factor history. `add_loop`/`remove_loop` then fail until an explicit
`optimize` reapplies the retained active factors. To undo a bad loop and keep
the remaining graph, use `remove_loop` directly. Removing the final loop
recovers original odometry and its reconstructed map.

An identical request ID and payload is idempotent while its resulting pose
revision remains current, including rejected trials. Reusing an ID with a
different payload, retrying a superseded request, using the wrong session,
or supplying a stale expected revision fails. Reverse/adjacent IDs and
duplicate active pairs are rejected. Removed IDs remain reserved for history;
a newly verified replacement needs a new ID.

Responses contain `success`, `message`, `pose_revision`, and `report_json`.
A valid rejected trial has `success=false` and a durable report with a reason
and available metrics; map and active factors remain unchanged. Malformed or
stale requests fail without appending an event. Storage/rebuild failure rolls
back the entire proposed change and can be retried. A 60-second service timeout
may leave an already running job in progress; retry the identical request.
Publication follows the worker's atomic map/TF cache refresh, observable as
`cached_pose_revision` in diagnostics.

The offline command always writes a **new sealed archive**, preserving its
source. A rejected trial produces a copy with the rejection record and the
previous geometry. The command's JSON output distinguishes acceptance from
rejection; successful command execution alone does not mean loop acceptance.

```bash
ros2 run direct_lidar_inertial_odometry mapping_archive.py graph \
  /absolute/original.dliomap /absolute/initialized.dliomap \
  --request /absolute/initialize.json
```

Generate an illustrative initialization request as follows. These numbers are
an explicit example assumption, not a tuned or calibrated tunnel model. Keep
the validation defaults until representative data justifies changes.

```python
import json
import uuid
import numpy as np

request = {
    "session_id": "ARCHIVE-SESSION-UUID",
    "expected_revision": 0,
    "request_id": str(uuid.uuid4()),
    "action": "initialize",
    "configuration": {
        "odometry_noise": {
            "kind": "assumed",
            "model": "example independent relative increment process model",
            "covariance_floor": np.diag([0.02**2]*3 + [0.005**2]*3).tolist(),
            "covariance_per_second": np.diag([0.1**2]*3 + [0.02**2]*3).tolist(),
            "covariance_per_metre": np.diag([0.03**2]*3 + [0.005**2]*3).tolist(),
            "max_gap_seconds": 2.0,
        },
        "validation": {},
    },
}
with open("initialize.json", "w") as stream:
    json.dump(request, stream, indent=2, allow_nan=False)
```

For insertion, replace `action` with `add_loop`, remove `configuration`, update
the expected revision/request ID, and supply this `loop` object using actual
verified IDs, transform, and covariance:

```python
request["loop"] = {
    "id": "verified-revisit-001",
    "from_id": i,
    "to_id": j,                      # j > i + 1
    "transform": Z_ij.tolist(),      # 4x4, source j into target i
    "covariance": Sigma_ij.tolist(), # 6x6, RIGHT/local translation first
    "covariance_kind": "conditional",
    "covariance_model": "describe estimation, conditioning, and scale here",
    "provenance": "describe independent verification and reference its evidence",
}
```

## Validation and trial solve

Before the trial solve, a candidate must pass temporal/index exclusion,
an absolute discrepancy ceiling against the current trajectory, and geometry
checks against the two original individual local clouds. No corrected fused
submap containing the source observation is used as its target. Geometry
processing excludes near-body returns, downsamples deterministically, and
caps sample count. It checks overlap in both directions, inlier distance
percentiles, distinct matched target support, and a point-to-plane
observability spectrum. Rotation columns are scaled by RMS source range
before comparing eigenvalues. This spectrum is a shape/rank check, not a
measurement covariance. A featureless plane or ideal extruded corridor fails.

The GTSAM batch Levenberg-Marquardt solve starts from original odometry,
anchors the gauge numerically, and uses a Huber kernel for loops. Acceptance
also requires convergence, bounded **unrobustified** Mahalanobis residuals
for every odometry and active loop factor, absolute post-fit loop residuals,
and an absolute trajectory displacement ceiling. The robust kernel or an
inflated covariance cannot bypass these absolute gates. Settings and their
units are defined in [`Validation`](../scripts/dliio_mapping/loop_validation.py)
and persisted with the graph. Defaults are conservative engineering gates,
not measured false-positive probabilities.

The validator has no held-out-data guarantee: an external registration may
already have used all supplied points. It does not calibrate covariance or
discover geometric aliases within the consistency envelope. External
verification provenance is required but is not machine proof of place identity.
Automatic retrieval should remain disconnected until labeled true revisits
and repeated-scene negatives have been assessed independently.

## Storage, resource bounds, and validation

Archive version 3 adds `graph_state`, `graph_loops`, and `graph_requests` to the
version 2 tables. Original clouds, poses, and observation uncertainty remain
immutable. The stored noise model and original trajectory reconstruct the
odometry factors; loop measurements retain their full covariance/provenance
and added/removed revisions. Request records retain complete inputs, payload
digests, rejection reasons, and solve metrics. Version 1/2 archives remain
readable; editable copies upgrade transactionally. Older dliio readers reject
version 3 rather than silently discarding graph state.

This reference batch path is capped at 5,000 observations, 15,000 retained
loop records, 100,000 requests, and at most 20,000 sampled points per cloud
(6,000 by default). It reconstructs every chronological submap after each
accepted solution, including evicted groups. Optimization and reconstruction
run on the storage worker, so its bounded input queue may drop observations
during a long update. This is not yet an incremental backend with a measured
live replay throughput budget. The frontend remains on its own pipeline.

Build dependencies now include GTSAM >= 4.2, pybind11 headers, Python
development headers, and SciPy at runtime. The native module installs into
the ROS package's Python path and does not require GTSAM's Python bindings.
The current host validation uses GTSAM 4.3 and ROS 2 Lyrical.

```bash
colcon build --packages-select direct_lidar_inertial_odometry --symlink-install
source install/setup.bash
ctest --test-dir build/direct_lidar_inertial_odometry --output-on-failure \
  -R '^(test_mapping_.*|test_loop_validation)$'
ROS_DOMAIN_ID=178 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  ros2 run direct_lidar_inertial_odometry verify_pose_graph.py \
  --output /tmp/dliio-pose-graph.json
```

The native solve is compared with an independent finite-difference optimizer,
including translation/rotation covariance cross terms and lever arms. Tests
cover reverse traversals, planar/corridor degeneracy, low overlap, body returns,
unknown covariance, false transforms, duplicate edges, rejected trials,
factor removal, external rollback, schema migration, transaction failure, and
abrupt process death after factor insertion. The ROS check exercises the
installed service, real optimizer, persistent map, dynamic TF, idempotence,
and factor removal using a known synthetic room revisit.

Validation on 2026-09-07: the Lyrical Release/symlink build passes. All **241
cases** in the seven mapping/launch Python suites and three frontend C++
suites pass, including 51 new graph/geometry cases. The installed ROS graph
check passes: a known 0.8 m synthetic endpoint drift closes to under 2 cm,
the map and dynamic correction agree, an inflated-covariance false candidate
is rejected, and removing the factor restores the original map. These are
deterministic implementation checks; no real tunnel replay result is claimed.
The existing ROS pose-revision check also passes against schema 3, including
external nonuniform corrections, rollback, TF, and the offline-copy workflow.
