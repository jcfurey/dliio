# Reference Card — Research Literature & Framework Documentation

Quick-reference for the papers, algorithms, and framework docs relevant to this repository. Companion to `doc/REVIEW.md` (which applies these to the code).

## 1. Core algorithms implemented in this repo

| Reference | Venue | Link | What it grounds here |
|---|---|---|---|
| Chen, Nemiroff, Lopez — *Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction* | ICRA 2023 | [arXiv:2203.03749](https://arxiv.org/abs/2203.03749) · [IEEE](https://ieeexplore.ieee.org/document/10160508) · [upstream repo](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) | The whole architecture: analytic constant-jerk deskew (`integrateImuInternal`), scan-to-map direct registration, keyframe submapping |
| Chen, Lopez, Agha-mohammadi, Mehta — *Direct LiDAR Odometry: Fast Localization with Dense Point Clouds* | RA-L 2022 | [arXiv:2110.00605](https://arxiv.org/abs/2110.00605) | DLIO's predecessor: kNN + convex/concave-hull keyframe submap (`buildSubmap`), adaptive spaciousness keyframing |
| Lopez — *A Contracting Hierarchical Observer for Pose-Inertial Fusion* | 2023 | [arXiv:2303.02777](https://arxiv.org/abs/2303.02777) | `propagateState()` / `updateState()`: cascaded quaternion+gyro-bias and position/velocity/accel-bias observers, gain clamps |
| Segal, Haehnel, Thrun — *Generalized-ICP* | RSS 2009 | [paper](https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf) | Plane-to-plane Mahalanobis registration that `NanoGICP` implements (see REVIEW §II.1 for where this fork diverges) |
| Koide, Yokozuka, Oishi, Banno — *Voxelized GICP for Fast and Accurate 3D Point Cloud Registration* | ICRA 2021 | [paper](https://staff.aist.go.jp/shuji.oishi/assets/papers/preprint/VoxelGICP_ICRA2021.pdf) · [fast_gicp](https://github.com/koide3/fast_gicp) | The multithreaded GICP implementation DLIO's NanoGICP was forked from; reference for the `(1,1,1e-3)` PLANE regularization |
| Zhang, Kaess, Singh — *On Degeneracy of Optimization-based State Estimation Problems* | ICRA 2016 | [paper (CMU)](https://frc.ri.cmu.edu/~zhangji/publications/ICRA_2016.pdf) · [IEEE](https://ieeexplore.ieee.org/document/7487211/) | The solution-remapping degeneracy gate in `computeTransformation()` (`degeneracyThreshRatio`) for featureless tunnels |
| Blanco, Rai — *nanoflann* | 2014– | [github.com/jlblancoc/nanoflann](https://github.com/jlblancoc/nanoflann) | Vendored kd-tree (`include/nano_gicp/nanoflann.h`, adaptor in `nanoflann_adaptor.h`) |

## 2. Intensity / reflectivity-aided LIO (the competition & idea pool)

| Reference | Venue | Link | Key idea relevant here |
|---|---|---|---|
| Zheng, Zhu et al. — *FAST-LIVO / FAST-LIVO2: Fast Direct LiDAR-Inertial-Visual Odometry* | IROS 2022 / T-RO 2024 | [arXiv:2203.00893](https://arxiv.org/abs/2203.00893) · [arXiv:2408.14035](https://arxiv.org/abs/2408.14035) | Sparse-direct photometric map-point alignment; **occlusion + depth-discontinuity outlier rejection** — the reference for the frame-to-map range/occlusion cull |
| Pfreundschuh et al. — *COIN-LIO: Complementary Intensity-Augmented LiDAR Inertial Odometry* | ICRA 2024 | [arXiv:2310.01235](https://arxiv.org/abs/2310.01235) · [code](https://github.com/ethz-asl/COIN-LIO) | Intensity-image patches selected to be *complementary to degenerate geometric directions*; brightness/range filtering; IEKF fusion. The benchmark for the tunnel scenario |
| Zhang et al. — *RI-LIO: Reflectivity Image Assisted Tightly-Coupled LiDAR-Inertial Odometry* | RA-L 2023 | [IEEE](https://ieeexplore.ieee.org/document/10041769) | Photometric residuals from calibrated reflectivity images blended with geometric residuals — closest published analogue to this repo's photometric term |
| *PG-LIO: Photometric-Geometric Fusion for Robust LiDAR-Inertial Odometry* | 2025 | [arXiv:2506.18583](https://arxiv.org/abs/2506.18583) | Recent photometric-geometric fusion formulation |
| Wang, Wang, Xie — *Intensity-SLAM* | RA-L 2021 | [arXiv:2102.03798](https://arxiv.org/abs/2102.03798) | Intensity-map-based weighting in scan matching |
| Kashani, Olsen, Parrish, Wilson — *A Review of LIDAR Radiometric Processing: From Ad Hoc Intensity Correction to Rigorous Radiometric Calibration* | Sensors 2015 | [PMC4701271](https://pmc.ncbi.nlm.nih.gov/articles/PMC4701271/) | The physics behind `intensityAlpha`/`intensityRRef`: `I·R²/(R_ref²·cos α)`, incidence-angle term, near-range non-ideality |
| Ouster — sensor data docs (Calibrated Reflectivity) | — | [Ouster docs](https://static.ouster.dev/sensor-docs/) | Why the `reflectivity` channel is range-normalized uint16 and intensity is not |

## 3. ROS 2 documentation relevant to this package (Jazzy)

| Topic | Doc | Why it matters here |
|---|---|---|
| QoS compatibility | [About QoS settings](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html) | Reliable subscriber × best-effort publisher = no data. The point-cloud sub must use `SensorDataQoS` (REVIEW §III.22) |
| Composition / components | [Composing nodes](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Composition.html) · [rclcpp_components](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Composition.html) | Intra-process zero-copy between odom and map node (REVIEW §III.23) |
| Intra-process comms | [Intra-process communication](https://docs.ros.org/en/jazzy/Tutorials/Demos/Intra-Process-Communication.html) | The keyframe-cloud hop currently serializes through the RMW |
| Parameters | [Parameter concepts](https://docs.ros.org/en/jazzy/Concepts/Basic/About-Parameters.html) · [rclcpp params tutorial](https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-CPP.html) | Descriptors, ranges, read-only flags, `on_set` callbacks for live tuning (REVIEW §III.26) |
| tf2 static broadcaster | [Writing a static broadcaster](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Static-Broadcaster-Cpp.html) | Extrinsics should be latched once, not re-sent per scan (REVIEW §III.24) |
| Executors & callback groups | [Executors](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Executors.html) | The MultiThreadedExecutor + MutuallyExclusive-group concurrency model both nodes rely on |
| Logging | [Logging](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Logging.html) | Replacement for the printf/ANSI dashboard; `RCLCPP_*_THROTTLE` |
| Diagnostics | [diagnostic_updater](https://docs.ros.org/en/jazzy/p/diagnostic_updater/) | The right home for the rates/CPU/RAM/degeneracy dashboard |
| ament_cmake / ament_cmake_auto | [ament_cmake user docs](https://docs.ros.org/en/jazzy/How-To-Guides/Ament-CMake-Documentation.html) | Build-system conventions this package half-follows (REVIEW §III.30) |
| Linters & tests | [ament_lint_auto](https://github.com/ament/ament_lint/blob/jazzy/ament_lint_auto/doc/index.rst) · [Testing](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Testing/Testing-Main.html) | Currently zero of either (REVIEW §III.28) |
| rosidl interfaces | [About interfaces](https://docs.ros.org/en/jazzy/Concepts/Basic/About-Interfaces.html) | `srv/SavePCD.srv` generation (`rosidl_generate_interfaces`) |
| REP 105 (frames) | [REP 105](https://www.ros.org/reps/rep-0105.html) | `odom` → `base_link` conventions the TF tree follows |
| REP 145 (IMU) | [REP 145](https://www.ros.org/reps/rep-0145.html) | IMU frame/orientation conventions `transformImu` should be checked against |
| rosbag2 | [Recording and playback](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html) | Regression testing against the DLIO test datasets |

## 4. Libraries

| Library | Docs | Used for |
|---|---|---|
| PCL ≥ 1.10 | [pointclouds.org](https://pointclouds.org/documentation/) | Cloud containers, filters (VoxelGrid, CropBox), hulls, `pcl::Registration` base |
| Eigen ≥ 3.3.7 | [eigen.tuxfamily.org](https://eigen.tuxfamily.org/dox/) | All linear algebra; `SelfAdjointEigenSolver` for the degeneracy gate; aligned allocators |
| Boost (circular_buffer, range adaptors) | [boost.org](https://www.boost.org/doc/libs/) | IMU ring buffer; timestamp-unique filtering in deskew |
| OpenMP ≥ 4.5 | [openmp.org](https://www.openmp.org/specifications/) | Parallel covariance/gradient/correspondence loops |

## 5. Test data

| Dataset | Link | Notes |
|---|---|---|
| DLIO "aggressive motion" bag (Ouster OS1-32, 1.2 GB) | [Google Drive](https://drive.google.com/file/d/1Sp_Mph4rekXKY2euxYxv6SD6WIzB-wVU/view?usp=sharing) | Deskew stress test (from upstream README) |
| DLIO outdoor trajectory (Ouster OSDome, 16.5 GB) | [Google Drive](https://drive.google.com/file/d/1HbmF5gTHxCAMqBkEd5PTxDNQvcI8tKXn/view?usp=sharing) | Long-run / memory-growth testing |
| *(missing)* featureless-tunnel bag | — | Needed to validate the degeneracy gate; simulated conduit with an intensity-capable Ouster model is the cheapest substitute |
