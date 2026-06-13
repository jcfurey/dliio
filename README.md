# Modifications

Lidar intensity aided DLIO. The extra "i" in the name is for intensity.

## New parameters

Added `photometricWeight` (default value: 0.1) and `gradientKNeighbors` (default value: 10) to cfg/params.yaml.

These new user-configurable parameters allow you to control the influence of the intensity information. In featureless environments a higher `photometricWeight` is preferred.

Added `intensityAlpha` (default value: 2.0) and `intensityRRef` (default value: 1.0) to cfg/params.yaml.

These new user-configurable parameters allow you to control the influence of the range correction. Specifically, `intensityAlpha` is the falloff exponent and `intensityRRef` is the reference range in metres.

Added `photometricChannel` (default value: `intensity`) to cfg/params.yaml.

Selects which point field feeds the photometric GICP term: `intensity` (raw return strength, range-dependent, range correction applied) or `reflectivity` (calibrated, range-normalized — e.g. Ouster; range correction is skipped). The input cloud must carry the chosen field.

Added `degeneracyThreshRatio` (default value: 0.005) to cfg/params.yaml.

In geometrically self-similar environments (a featureless conduit/tunnel is the canonical case) scan-to-map registration is unobservable along one or more directions — for a smooth tunnel, translation along its axis. The rotation and translation blocks of the registration Hessian are eigen-analyzed separately every iteration and the update is projected off directions whose eigenvalue falls below `degeneracyThreshRatio * block_lambda_max` (solution remapping, Zhang/Kaess/Singh ICRA 2016), so the IMU prior is held there instead of being overwritten by noise. If the photometric term finds usable intensity texture (seams, joints, stains), it re-constrains those directions automatically. A throttled warning is logged while degeneracy is active. Pair with `regularizationMethod: plane` for the sharpest degeneracy discrimination. See `doc/REVIEW.md` §II.4 and `doc/REFERENCES.md`.

Added `regularizationMethod` (default value: `min_eig`) and `photometricScale` (default value: 255.0) to cfg/params.yaml.

`regularizationMethod` selects the GICP covariance regularization: `min_eig` preserves this fork's historical behavior (clamped singular values, previously mislabeled "plane" internally); `plane` is true GICP plane-to-plane (fixed scale-free discs, recommended for degenerate environments). `photometricScale` is the full-scale value of the photometric channel — the channel is normalized by it so `photometricWeight` is dimensionless and transfers across sensors (255 covers 8-bit intensity and Ouster calibrated reflectivity; 65535 for raw 16-bit channels).

Added `photometricHuberDelta` (default value: 0.05), `maxKeyframes` (default value: 0), and `odom/debug/dashboard` (default value: true) to cfg/params.yaml.

`photometricHuberDelta` Huber-robustifies the photometric residual so specular/wet-surface outliers are downweighted instead of shoving the pose at full weight. `maxKeyframes` optionally bounds the keyframe map by pruning the most spatially redundant keyframe when exceeded (0 = unlimited, the historical behavior). `odom/debug/dashboard` toggles the ANSI terminal dashboard — disable it under multiplexed logging; the composed launch file disables it automatically.

## Configuration & wiring

**Config files** (all parameters are commented in the files themselves):

| File | Contents |
|---|---|
| `cfg/dlio.yaml` | Per-robot: sensor extrinsics, IMU intrinsics, preprocessing switches. Copy per robot and pass via `robot_config:=`. |
| `cfg/params.yaml` | Algorithm tuning: registration, keyframing, observer gains, photometric term, degeneracy gate, published covariance. Override via `params_file:=`. |
| `cfg/examples/ouster_reflectivity.yaml` | Overlay: Ouster calibrated-reflectivity photometric channel + plane regularization for tunnel-like environments. |
| `cfg/examples/ouster_tunnel.yaml` | Overlay: full tunnel mode — reflectivity + plane + degeneracy gate **with photometricWeight 0.3** (strong enough to re-constrain the tunnel axis; the gate alone diverges). Supersedes `ouster_reflectivity.yaml` for tunnels. |
| `cfg/examples/simulation.yaml` | Overlay: sim time, no IMU calibration wait, ideal extrinsics. |

**Launch arguments** (`dlio.launch.py`): `pointcloud_topic`, `imu_topic`, `rviz`, `use_sim_time` (default **false**; set true under Gazebo or `ros2 bag play --clock`), `robot_config`, `params_file`. Overlays can be appended at run time with `--ros-args --params-file <overlay.yaml>` (later files win).

**Composed launch** (`dlio_composed.launch.py`): same arguments (minus `rviz`); runs both nodes as components in one multithreaded container with intra-process communication, so keyframe clouds pass between the odometry and map nodes without serialization. The terminal dashboard is disabled automatically in this mode. Both nodes are also loadable into your own container (`dlio::OdomNode`, `dlio::MapNode`).

**Outputs / downstream wiring:**

| Interface | Name | Notes |
|---|---|---|
| Odometry | `dlio/odom_node/odom` (`nav_msgs/Odometry`) | `odom` → `base_link`, stamped with IMU time at ~IMU rate; constant diagonal covariance from `odom/covariance/*` (tune for your EKF) |
| Pose | `dlio/odom_node/pose` (`PoseStamped`) | same state, no twist |
| TF | `odom` → `base_link` dynamic; `base_link` → `lidar`/`imu` latched on `/tf_static` | follows REP-105; no `map` frame is published — DLIO is odometry, not SLAM with loop closure |
| Deskewed scan | `dlio/odom_node/pointcloud/deskewed` | in `odom` frame; intensity is range-corrected when the intensity channel is active |
| Keyframes / map | `dlio/odom_node/keyframes`, `dlio/map_node/map` | map is keyframe accumulation (unbounded; for visualization/export) |
| Save map | `/save_pcd` service | absolute existing directory required |

Wiring into a fusion/navigation stack: feed `dlio/odom_node/odom` to `robot_localization` (or use it directly as the `odom`→`base_link` source for Nav2, in which case let DLIO own that TF and do **not** also fuse a second publisher of the same transform). The LiDAR subscription uses best-effort `SensorDataQoS`; drivers publishing reliable still match.

## Notes & findings

- `doc/VISUAL_TERM.md` — design of the optional direct-photometric anchors (camera frame-to-frame / frame-to-map, and the COIN-LIO LiDAR intensity-image term), the degeneracy-gate safety floor, parameters, and tests.
- `doc/TUNNEL_FINDINGS.md` — consolidated empirical findings from the `resple_test_ws` harness on the 06042026 tunnel: the degeneracy-gate and reflectivity-weight sweeps, the visual-term A/Bs, the COIN-LIO results, recommended configs, the chaotic-basin / co-scheduling caveats, and open next steps.

---

# Original ReadMe

## Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction

#### [[ IEEE ICRA ](https://ieeexplore.ieee.org/document/10160508)] [[ arXiv ](https://arxiv.org/abs/2203.03749)] [[ Video ](https://www.youtube.com/watch?v=4-oXjG8ow10)] [[ Presentation ](https://www.youtube.com/watch?v=Hmiw66KZ1tU)]

DLIO is a new lightweight LiDAR-inertial odometry algorithm with a novel coarse-to-fine approach in constructing continuous-time trajectories for precise motion correction. It features several algorithmic improvements over its predecessor, [DLO](https://github.com/vectr-ucla/direct_lidar_odometry), and was presented at the IEEE International Conference on Robotics and Automation (ICRA) in London, UK in 2023.

<br>
<p align='center'>
    <img src="./doc/img/dlio.png" alt="drawing" width="720"/>
</p>

## Instructions

### Sensor Setup
DLIO has been extensively tested using a variety of sensor configurations and currently supports Ouster, Velodyne, Hesai, and Livox LiDARs. The point cloud should be of input type `sensor_msgs::msg::PointCloud2` and the 6-axis IMU input type of `sensor_msgs::msg::Imu`.

For best performance, extrinsic calibration between the LiDAR/IMU sensors and the robot's center-of-gravity should be inputted into `cfg/dlio.yaml`. If the exact values of these are unavailable, a rough LiDAR-to-IMU extrinsics can also be used (note however that performance will be degraded).

IMU intrinsics are also necessary for best performance, and there are several open-source calibration tools to get these values. These values should also go into `cfg/dlio.yaml`. In practice however, if you are just testing this work, using the default ideal values and performing the initial calibration procedure should be fine.

Also note that the LiDAR and IMU sensors _need_ to be properly time-synchronized, otherwise DLIO will not work. We recommend using a LiDAR with an integrated IMU (such as an Ouster) for simplicity of extrinsics and synchronization.

### Dependencies
The following has been verified to be compatible, although other configurations may work too:

- Ubuntu 24.04
- ROS 2 Jazzy or newer (`rclcpp`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `pcl_ros`, `pcl_conversions`, `tf2_ros`)
- C++ 17
- CMake >= `3.12.4`
- OpenMP >= `4.5`
- Point Cloud Library >= `1.10.0`
- Eigen >= `3.3.7`

```sh
sudo apt install libomp-dev libpcl-dev libeigen3-dev
```

DLIO currently supports `ROS 1` and `ROS 2`!

### Compiling
Compile with `colcon`:

```sh
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
```
```sh
git clone https://github.com/jcfurey/dliio direct_lidar_inertial_odometry
```
```sh
cd ~/ros2_ws
```
```sh
colcon build --symlink-install --packages-select direct_lidar_inertial_odometry
```

### Execution

<details>
<summary> After compiling, don't forget to source before ROS commands.</summary>

``` bash
source ~/ros2_ws/install/setup.bash
```
</details>

Execute via:

```sh
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  rviz:={true, false} \
  pointcloud_topic:=/robot/lidar \
  imu_topic:=/robot/imu
```

<details>
<summary> Example command: </summary>

``` bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py rviz:=true pointcloud_topic:=/lexus3/os_center/points imu_topic:=/lexus3/os_center/imu
```
</details>

Be sure to change the topic names to your corresponding topics. Alternatively, edit the launch file directly if desired. If successful, you should see the following output in your terminal:
<br>
<p align='center'>
    <img src="./doc/img/terminal.png" alt="drawing" width="480"/>
</p>

### Services
To save DLIO's generated map into `.pcd` format, call the following service:

```sh
ros2 service call /save_pcd direct_lidar_inertial_odometry/srv/SavePCD "{'leaf_size': 0.2, 'save_path': '/home/user/map'}"
```

Note: `save_path` must be an absolute path to an existing directory (`~` is not expanded).

### Test Data
For your convenience, we provide test data [here](https://drive.google.com/file/d/1Sp_Mph4rekXKY2euxYxv6SD6WIzB-wVU/view?usp=sharing) (1.2GB, 1m 13s, Ouster OS1-32) of an aggressive motion to test our motion correction scheme, and [here](https://drive.google.com/file/d/1HbmF5gTHxCAMqBkEd5PTxDNQvcI8tKXn/view?usp=sharing) (16.5GB, 4m 21s, Ouster OSDome) of a longer trajectory outside with lots of trees. Try these two datasets with both deskewing on and off!

<br>
<p align='center'>
    <img src="./doc/gif/aggressive.gif" alt="drawing" width="720"/>
</p>

## Citation
If you found this work useful, please cite our manuscript:

```bibtex
@article{chen2022dlio,
  title={Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction},
  author={Chen, Kenny and Nemiroff, Ryan and Lopez, Brett T},
  journal={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={3983-3989},
  doi={10.1109/ICRA48891.2023.10160508}
}
```

## Acknowledgements

We thank the authors of the [FastGICP](https://github.com/SMRT-AIST/fast_gicp) and [NanoFLANN](https://github.com/jlblancoc/nanoflann) open-source packages:

- Kenji Koide, Masashi Yokozuka, Shuji Oishi, and Atsuhiko Banno, “Voxelized GICP for Fast and Accurate 3D Point Cloud Registration,” in _IEEE International Conference on Robotics and Automation (ICRA)_, IEEE, 2021, pp. 11 054–11 059.
- Jose Luis Blanco and Pranjal Kumar Rai, “NanoFLANN: a C++ Header-Only Fork of FLANN, A Library for Nearest Neighbor (NN) with KD-Trees,” https://github.com/jlblancoc/nanoflann, 2014.

We would also like to thank Helene Levy and David Thorne for their help with data collection.

## License
This work is licensed under the terms of the MIT license.

<br>
<p align='center'>
    <img src="./doc/img/ucla.png" alt="drawing" width="720"/>
</p>
<p align='center'>
    <img src="./doc/img/trees.png" alt="drawing" width="720"/>
</p>
