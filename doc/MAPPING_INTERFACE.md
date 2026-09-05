# Mapping input contract

This contract describes the output after the 2026-09-05 correctness pass on
`cam-dev`. The [persistent mapper](MAPPING_NODE.md) now consumes this interface;
`MapNode` remains available as the legacy accumulated-cloud preview.

## Frames and measurement times

The standard launch files remap the following topics under `/dlio/odom_node`.
Frame names remain configurable; the table uses `odom` and `base_link`.

| Topic suffix | Contents | Measurement reference |
|---|---|---|
| `odom` | Observer pose in `odom`; linear/angular twist in `base_link` | Latest propagated IMU state |
| `pose` | Same observer pose as `odom` | Same IMU timestamp |
| `scan_pose` | Registered base pose, `T_odom_base` | Scan reference time |
| `pointcloud/deskewed` | Registered XYZ already in `odom` | Exact same header as its `scan_pose` |
| `keyframe_pose` | Registered base pose for one keyframe | Exact same header as its keyframe cloud |
| `pointcloud/keyframe` | Registered keyframe XYZ already in `odom` | Keyframe's scan reference time |
| `path` | Recent registered scan poses | Individual scan reference times |
| `keyframes` | Recent keyframe poses as a legacy `PoseArray` | Header describes only the newest entry |

With per-return deskewing, the reference is the median point time used by the
registration. Without usable per-return timing or with deskew disabled, it is
the input cloud header. The first target produces a keyframe; subsequent
registered scans produce `scan_pose`. Cloud publication still respects subscriber
presence and, for the deskewed topic, `waitUntilMove`.

The scan publisher captures its cloud, registration correction, reference pose,
timestamp, and movement gate before starting the background worker. Keyframe
selection and storage both use this registered pose. A newer IMU update cannot
change the pose or timestamp attached to an older cloud.

`odom` and `pose` serve the high-rate observer. The optional `odom -> base_link`
TF uses that same observer snapshot and timestamp. Its pose may differ from the
registered scan pose because it includes subsequent IMU propagation and observer
gain filtering. To recover a local keyframe, use the **paired `keyframe_pose`**:

```
p_base = inverse(T_odom_base_keyframe) * p_odom
```

Do not apply `T_odom_base` again to cloud XYZ that is already in `odom`, or use
an interpolated observer pose to undo the registered scan transform. Apply the
configured static base/lidar extrinsic additionally if storing lidar-frame XYZ.

The body-frame twist follows the ROS
[Odometry definition](https://github.com/ros2/common_interfaces/blob/rolling/nav_msgs/msg/Odometry.msg).
The cloud's frame and measurement reference follow the
[PointCloud2 definition](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/PointCloud2.msg).

## Fields and delivery

Registered clouds, keyframes, the preview map, and exported PCDs preserve these
float fields: `intensity` (raw return signal), `reflectivity`,
`intensity_corrected`, and `lidar_intensity` (the selected image channel).
Voxelized values are averages, rather than individual raw returns. The raw
signal and reflectivity remain distinct measurements; they are correlated cues.

Processed outputs no longer advertise `t`, `time`, or `timestamp`. Internally
these names overlap in a point-type union, and their original acquisition
offsets are not valid relative to the deskewed reference header or after voxel
averaging. The input Ouster topic retains acquisition time, ring, range, and
ambient/near-IR. Consumers that need all raw return metadata must retain that
input separately; the processed output is not a lossless raw-scan archive.

Pair cloud and pose by exact integer header timestamp within a running session.
The new pose publishers use reliable, volatile QoS with depth 10. Keyframe
clouds use reliable, volatile QoS with depth 1; consumers should subscribe before
replay and bound their synchronization queues. The topics are separate ROS
messages, so receiving one does not guarantee receiving the other. A mapper
must drop/report unmatched inputs rather than substitute a latest pose.

Input scan and IMU streams each require strictly increasing timestamps.
Duplicates and backward timestamps are discarded before modifying their
integration history. The first pre-calibrated IMU sample establishes the time
origin instead of integrating from epoch zero. Restart the odometry and map
nodes when seeking backward or starting a new bag session.

## Preview map behavior

`MapNode` accepts only keyframes in its configured odometry frame, with valid
PointCloud2 layout and XYZ fields. It decodes byte order and padded rows, drops
nonfinite XYZ, and rejects duplicate/out-of-order keyframes. It does not infer
missing transforms or relabel input from another frame.

The node voxelizes each incoming keyframe, then appends it. Overlap between
different keyframes still creates repeated samples. Its map stamp is the latest
accepted keyframe's measurement time. A serialized snapshot is cached until the
map changes and republished at `map/publishRate`; the unchanged-map preparation
cost no longer scales with the whole map. DDS transmission and subscriber
copies still scale with message size. Caching retains one additional serialized
map in memory. PCD export performs the existing global voxel averaging and
requires a finite positive leaf size.

## Persistent backend and next interface

The persistent mapper archives a session ID, stable keyframe ID, scan reference
stamp, registered pose, and local deskewed cloud. Registration quality and
observability still need an extended frontend interface. A single message or acknowledged archive
would make pose/cloud delivery and replay explicit. Reported registration
Hessians and default odometry covariances are not calibrated independent sensor
uncertainties.

Keep these responsibilities outside the odometry callback:

1. Bounded spatial storage, voxel/surfel statistics, and overlap weighting.
2. Retention of local keyframes and pose revisions, so loop closure can rebuild
   affected map regions instead of permanently baking in odometry drift.
3. Session reset, input-gap diagnostics, backpressure, and reproducible export.
4. A global `map -> odom` correction while odometry owns local motion, following
   the frame separation in [REP-105](https://github.com/openrobotics/reps/blob/main/_posts/rep-0105.md).

The preview node currently has no loop closure, pose revision, map-frame
correction, or bounded global storage. Smoother tunnel motion helps map quality,
but does not establish metric accuracy without ground truth or surveyed marks.
