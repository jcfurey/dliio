# Reference Card — Research Literature & Framework Documentation

Quick-reference for the papers, algorithms, and framework docs relevant to this repository. Companion to [REVIEW.md](REVIEW.md), which applies these to the code.

The **2026-09-05 curved-tunnel research** is in
[TUNNEL_RESEARCH_2026-09-05.md](TUNNEL_RESEARCH_2026-09-05.md).
Its selected sources have stable citation keys in
[tunnel_references.bib](tunnel_references.bib); the reading inventory below
records what was inspected and where conclusions remain provisional.
The September Exyn VLP-16 recording and the earlier Ouster tunnel experiments
are different sensor setups and evidence sets.

## 1. Core algorithms implemented in this repo

| Reference | Venue | Link | What it grounds here |
|---|---|---|---|
| Chen, Nemiroff, Lopez — *Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction* | ICRA 2023 | [arXiv:2203.03749](https://arxiv.org/abs/2203.03749) · [IEEE](https://ieeexplore.ieee.org/document/10160508) · [upstream repo](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) | The whole architecture: analytic constant-jerk deskew (`integrateImuInternal`), scan-to-map direct registration, keyframe submapping |
| Chen, Lopez, Agha-mohammadi, Mehta — *Direct LiDAR Odometry: Fast Localization with Dense Point Clouds* | RA-L 2022 | [arXiv:2110.00605](https://arxiv.org/abs/2110.00605) | DLIO's predecessor: kNN + convex/concave-hull keyframe submap (`buildSubmap`), adaptive spaciousness keyframing |
| Lopez — *A Contracting Hierarchical Observer for Pose-Inertial Fusion* | 2023 | [arXiv:2303.02777](https://arxiv.org/abs/2303.02777) | `propagateState()` / `updateState()`: cascaded quaternion+gyro-bias and position/velocity/accel-bias observers, gain clamps |
| Segal, Haehnel, Thrun — *Generalized-ICP* | RSS 2009 | [paper](https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf) | Plane-to-plane Mahalanobis registration that `NanoGICP` implements (see REVIEW §II.1 for where this fork diverges) |
| Koide, Yokozuka, Oishi, Banno — *Voxelized GICP for Fast and Accurate 3D Point Cloud Registration* | ICRA 2021 | [paper](https://staff.aist.go.jp/shuji.oishi/assets/papers/preprint/VoxelGICP_ICRA2021.pdf) · [fast_gicp](https://github.com/koide3/fast_gicp) | The multithreaded GICP implementation DLIO's NanoGICP was forked from; reference for the `(1,1,1e-3)` PLANE regularization. **BSD-3-Clause** (SMRT-AIST) — license reproduced in `THIRD_PARTY_LICENSES.md` |
| Zhang, Kaess, Singh — *On Degeneracy of Optimization-based State Estimation Problems* | ICRA 2016 | [paper (CMU)](https://frc.ri.cmu.edu/~zhangji/publications/ICRA_2016.pdf) · [IEEE](https://ieeexplore.ieee.org/document/7487211/) | The solution-remapping degeneracy gate in `computeTransformation()` (`degeneracyThreshRatio`) for featureless tunnels |
| Blanco, Rai — *nanoflann* | 2014– | [github.com/jlblancoc/nanoflann](https://github.com/jlblancoc/nanoflann) | Vendored kd-tree (`include/nano_gicp/nanoflann.h`, adaptor in `nanoflann_adaptor.h`) |

## 2. Intensity / reflectivity-aided LIO (the competition & idea pool)

| Reference | Venue | Link | Key idea relevant here |
|---|---|---|---|
| Zheng, Zhu et al. — *FAST-LIVO / FAST-LIVO2: Fast Direct LiDAR-Inertial-Visual Odometry* | IROS 2022 / T-RO 2025 | [arXiv:2203.00893](https://arxiv.org/abs/2203.00893) · [arXiv:2408.14035](https://arxiv.org/abs/2408.14035) | Direct visual alignment and geometric visibility checks; FAST-LIVO2 appeared online in 2024 and in volume 41 (2025) |
| Pfreundschuh et al. — *COIN-LIO: Complementary Intensity-Augmented LiDAR Inertial Odometry* | ICRA 2024 | [arXiv:2310.01235](https://arxiv.org/abs/2310.01235) · [code](https://github.com/ethz-asl/COIN-LIO) | Intensity-image patches selected to be *complementary to degenerate geometric directions*; brightness/range filtering; IEKF fusion. The benchmark for the tunnel scenario |
| Zhang et al. — *RI-LIO: Reflectivity Image Assisted Tightly-Coupled LiDAR-Inertial Odometry* | RA-L 2023 | [IEEE](https://ieeexplore.ieee.org/document/10041769) | Reflectivity-image LIO reference; publication metadata checked, full methods not re-reviewed in the September pass |
| Engel, Koltun, Cremers — *Direct Sparse Odometry (DSO)* | T-PAMI 2018 | [arXiv:1607.02565](https://arxiv.org/abs/1607.02565) | The direct (photometric) image-alignment foundation behind the optional direct-camera frame-to-frame / frame-to-map residuals (`setupVisualForScan`, `accumulateVisualResidual`): minimize a Huber-robust brightness error rather than reprojecting features |
| Khedekar, Alexis — *PG-LIO: Photometric-Geometric Fusion for Robust LiDAR-Inertial Odometry* | Preprint, 2025 | [arXiv:2506.18583](https://arxiv.org/abs/2506.18583) · [author code: MIMOSA](https://github.com/ntnu-arl/mimosa) | Normalized photometric factors in a sliding window; the released photometric path targets Ouster |
| Wang, Wang, Xie — *Intensity-SLAM* | RA-L 2021 | [arXiv:2102.03798](https://arxiv.org/abs/2102.03798) | Intensity-map-based weighting in scan matching |
| Kashani, Olsen, Parrish, Wilson — *A Review of LIDAR Radiometric Processing: From Ad Hoc Intensity Correction to Rigorous Radiometric Calibration* | Sensors 2015 | [publisher](https://www.mdpi.com/1424-8220/15/11/28099) | Range, incidence, and sensor processing affect radiometry; establish calibration provenance before applying a correction |
| Velodyne — *VLP-16 User Manual*, Rev. F, §6.1 | Updated 2022; marked draft | [manufacturer PDF](https://data.ouster.io/downloads/velodyne/user-manual/vlp-16-user-manual-revf.pdf) | Documents the single return-strength byte as calibrated reflectivity; does not establish the Exyn processing chain |
| Ouster — sensor data docs | Living documentation | [sensor data](https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html) | Reflectivity and signal photons are separate fields. Reflectivity is an 8-bit packet quantity; PointCloud storage types need not match packet widths |

## 3. ROS 2 documentation relevant to this package (Jazzy)

These framework links remain pinned to Jazzy. The September workspace replay
used Lyrical; this section is not a record of that build environment.

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
| Linters & tests | [ament_lint_auto](https://github.com/ament/ament_lint/blob/jazzy/ament_lint_auto/doc/index.rst) · [Testing](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Testing/Testing-Main.html) | Test integration; dated workspace reports record the suites used for each build |
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
| September Exyn moving-head VLP-16 | [workspace investigation](../../../docs/dliio-tunnel-investigation.md) · [corrected trials](../../../docs/dliio-surface-texture.md) | Private capture; repeatable replays and regression evidence, without independent trajectory ground truth |
| ENWIDE | [official dataset](https://projects.asl.ethz.ch/datasets/enwide/) | Ouster; total-station prism **position** reference, including tunnel sequences. Apply prism extrinsics; do not treat it as orientation ground truth |
| GEODE | [author dataset instructions](https://github.com/PengYu-Team/GEODE_dataset) · [project](https://thisparticle.github.io/geode/) | Heterogeneous degenerate scenes; Alpha uses VLP-16, Beta Ouster, Gamma Livox. Check each platform's reference frame and calibration |

## 6. Curved-tunnel reading inventory — checked 2026-09-05

This is a problem-focused selection, not an exhaustive systematic review.
Primary manuscripts, author code, manufacturer manuals, and publisher/DOI
metadata support the entries. “Methods” means the relevant methods and
limitations were inspected, not that results were independently reproduced.
An abstract or metadata-only entry cannot support a detailed implementation
claim. BibTeX includes the sources used for this investigation, not every
framework link or historical paper above.

### Estimation, observability, and geometry

| BibTeX key | Primary source | Reading scope and use |
|---|---|---|
| `chen2023dlio` | [DLIO, ICRA 2023](https://arxiv.org/abs/2203.03749) | Architecture reference and local implementation comparison |
| `lopez2023observer` | [Contracting hierarchical observer, 2023 preprint](https://arxiv.org/abs/2303.02777) | Observer reference; changing fusion architecture needs a separate evaluation |
| `koideFastGicp` | [FastGICP LM](https://github.com/koide3/fast_gicp/blob/master/include/fast_gicp/gicp/impl/lsq_registration_impl.hpp) · [error evaluation](https://github.com/koide3/fast_gicp/blob/master/include/fast_gicp/gicp/impl/fast_gicp_impl.hpp) | Read `step_lm`, `linearize`, and `compute_error`; frozen correspondences during trial evaluation |
| `zhang2016degeneracy` | [Zhang, Kaess, Singh, ICRA 2016](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.pdf) | Methods; solution remapping in weak directions |
| `tuna2024xicp` | [X-ICP, T-RO 2024](https://arxiv.org/html/2211.16335v4) | Methods, especially §V-A; scan centering and correspondence contributions. Also splits rotation and translation |
| `hatleskog2024probabilistic` | [Hatleskog & Alexis, RA-L 2024](https://arxiv.org/html/2410.10784v2) | Methods; noise-aware information and directional confidence. Local `probGate` is a simplified adaptation |
| `tuna2025informed` | [Informed, Constrained, Aligned, T-FR 2025](https://arxiv.org/html/2408.11809v3) | Methods, field results, §VI limitations; initialization and method-dependent tuning matter |
| `bonnabel2016covariance` | [Bonnabel, Barczyk, Goulette, ACC 2016](https://arxiv.org/abs/1410.7632) | Initially abstract and publication record; extended to v3 methods on September 7 (see §8) |
| `mcdermott2025scanmatching` | [McDermott, Tufts dissertation, 2025](https://dl.tufts.edu/downloads/w9505f71j) | §2.8, printed pp. 46–47; coupled curved-tunnel and repeated-structure ambiguities |
| `lee2025genz` | [GenZ-ICP, RA-L 2025](https://arxiv.org/html/2411.06766v1) | Methods, §III-E; published blend uses planar-correspondence fraction, unlike the local Hessian-based blend |
| `pfreundschuh2026bievr` | [BIEVR-LIO, RSS 2026](https://www.roboticsproceedings.org/rss22/p049.html) · [manuscript](https://arxiv.org/html/2604.14421v2) | Methods and limitations; finer geometry is a useful controlled experiment, not proof of Exyn observability |

### Appearance and channel semantics

| BibTeX key | Primary source | Reading scope and use |
|---|---|---|
| `pfreundschuh2024coin` | [COIN-LIO, ICRA 2024](https://arxiv.org/html/2310.01235v4) · [code](https://github.com/ethz-asl/COIN-LIO) | Methods and sensor assumptions; complementary patch selection and intensity-image processing |
| `khedekar2025pglio` | [PG-LIO, 2025 preprint](https://arxiv.org/html/2506.18583v1) · [MIMOSA](https://github.com/ntnu-arl/mimosa) | Methods and experiment interpretation; normalized patches and persistent local references |
| `woodford2022ncc` | [Least Squares Normalized Cross Correlation, v3 (2022)](https://arxiv.org/abs/1810.04320v3) | Abstract and revision history; NCC optimization reference. First posted 2018; not a TPAMI publication |
| `zhang2023rilio` | [RI-LIO, RA-L 2023](https://doi.org/10.1109/LRA.2023.3243528) | Publication metadata checked; full-method review remains a follow-up |
| `chen2024igelio` | [IGE-LIO, TIM 2024](https://doi.org/10.1109/TIM.2024.3427795) | Metadata-only lead; no implementation recommendation based on this entry |
| `kashani2015radiometric` | [Radiometric review, Sensors 2015](https://www.mdpi.com/1424-8220/15/11/28099) | Publisher-indexed radiometric discussion; general physics, not calibration of this scanner |
| `velodyne2022vlp16` | [VLP-16 manual](https://data.ouster.io/downloads/velodyne/user-manual/vlp-16-user-manual-revf.pdf) | Read §6.1, printed p. 34; Rev. F updated 2022-03-07, document marked DRAFT |
| `ousterSensorData` | [Ouster sensor data](https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html) | Manufacturer field definitions; useful comparison, different sensor |

### Fusion alternatives and very recent work

| BibTeX key | Primary source | Reading scope and use |
|---|---|---|
| `zheng2025fastlivo2` | [FAST-LIVO2, T-RO 2025](https://arxiv.org/abs/2408.14035) · [code](https://github.com/hku-mars/FAST-LIVO2) | Abstract, publication metadata, author README; vision requires an adequate image stream |
| `lee2026lodestar` | [LODESTAR, RA-L 2026](https://arxiv.org/abs/2511.09142) · [KAIST publication record](https://pure.kaist.ac.kr/en/publications/lodestar-degeneracy-aware-lidar-inertial-odometry-with-adaptive-s/) | Abstract and issue metadata; covariance-aware historical-state fusion lead |
| `nissov2024radar` | [Degradation Resilient LiDAR-Radar-Inertial Odometry, ICRA 2024](https://arxiv.org/abs/2403.05332) | Abstract and author code; independent velocity-measurement option for future recordings |
| `yao2026giflio` | [GIF-LIO, TIM 2026](https://doi.org/10.1109/TIM.2026.3671940) | Publisher-indexed abstract and DOI metadata only; full-method review pending |
| `ding2026adaptiveintensity` | [Adaptive photometric weighting, Electronics 2026](https://www.mdpi.com/2079-9292/15/17/3970) | Publisher-indexed overview and metadata; published September 3, two days before this survey. Not independently reproduced here |

### Evaluation

| BibTeX key | Primary source | Reading scope and use |
|---|---|---|
| `zhang2018evaluation` | [Trajectory evaluation tutorial, IROS 2018](https://www.ifi.uzh.ch/dam/jcr:89d3db14-37b1-431d-94c3-8be9f37466d3/IROS18_Zhang.pdf) | Alignment and error definitions; metric estimation must not receive a fitted scale correction |
| `enwideDataset` | [ENWIDE](https://projects.asl.ethz.ch/datasets/enwide/) | Dataset reference, timestamps, prism extrinsics, and position-reference limitations |
| `chen2024geode` | [GEODE preprint, 2024](https://arxiv.org/abs/2409.04961) · [dataset instructions](https://github.com/PengYu-Team/GEODE_dataset) | Preprint metadata and platform/reference-frame instructions; BibTeX intentionally cites this inspected version |

Publication years in the bibliography follow the journal issue: X-ICP is 2024,
GenZ-ICP and FAST-LIVO2 are 2025, and LODESTAR is 2026, despite earlier years
in their DOIs or preprints. PG-LIO and the cited NCC revision are recorded as
preprints. The new September memo distinguishes published methods, local
adaptations, observed replay results, and proposed experiments.

## 7. Live failure and secondary IMU fusion — checked 2026-09-06

The [live-run handoff](../../../docs/dliio-tunnel-live-run-2026-09-06.md)
separates measured attitude departure from the proposed observer feedback
mechanism. Re-reading `lopez2023observer` (model and observer equations) and
`chen2023dlio` (observer, deskew, and registration initialization), alongside
local `propagateState()` and `updateState()`, supports the coupling mechanism.
These references do not identify the initiating failure in this recording.

The installed `robot_localization` package is **3.10.0 on ROS 2 Lyrical**.
The following primary documentation was read for the BNO80 feasibility work;
these are documentation citations, not evidence of a completed EKF experiment.
The linked `rolling-devel` pages can change after the access date.

| BibTeX key | Primary source | Reading scope and use |
|---|---|---|
| `robotLocalizationConfiguration` | [Configuring robot_localization](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/configuring_robot_localization.rst) | Sensor selection, correlated inputs, absolute/differential orientation, and covariance consistency; avoid treating DLIO pose, twist, and its primary IMU as independent measurements |
| `robotLocalizationSensorData` | [Preparing Your Data](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/preparing_sensor_data.rst) | Sensor/body/world frames, ENU convention, IMU mounting transforms, and covariance handling; the empirical BNO80 axis fit is not reviewed hardware calibration |
| `robotLocalizationStateEstimation` | [State Estimation Nodes](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/state_estimation_nodes.rst) | Filter configuration, rejection thresholds, diagnostics, and `publish_tf`; evaluate a separate output before any integration into the mapper |

The BNO80 attitude payload contains no covariance despite its topic name.
Its raw gyro and onboard attitude estimate also share sensor information.
An EKF needs explicit frame and uncertainty treatment; adding a second
physical IMU does not create an independent position reference or repair
DLIO's internal registration and deskew feedback automatically.

The [subsequent covariance experiments](../../../docs/dliio-dual-imu-covariance.md)
also checked the installed EKF core against the official
[EKF implementation](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/src/ekf.cpp)
and [filter base](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/src/filter_base.cpp):
measurement selection, angle wrapping, Mahalanobis rejection, prediction,
initialization, and covariance update. The standalone probe uses the installed
library, with declared experimental noise; it does not reproduce the ROS
frontend or establish statistical consistency. At this stage,
`bonnabel2016covariance` had been read only through its abstract/publication
record; the later methods review is recorded below.

## 8. Measurement-time observer — checked 2026-09-07 UTC

The [implementation note](TIMED_OBSERVER.md) distinguishes the local observer
covariance derivation and synthetic checks from the cited theory. Re-reading
`lopez2023observer` §§II–III establishes the mean observer coupling and its
assumed upstream pose input; it does not establish uncertainty in the reused
GICP map. The later full-PDF review of `bonnabel2016covariance` does not change
the current observer's declared measurement-noise model.

| BibTeX key | Primary source | Reading scope and use |
|---|---|---|
| `sola2017quaternion` | [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/html/1711.02508v1) | §§4.4 and 5: local/right versus global/left perturbations, IMU error coordinates, and covariance transport. Our fixed-gain observer is not the paper's ESKF |
| `rosRep103` | [REP-103 source](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst) | SI units, body axes, fixed-axis rotation covariance, and row-major ROS covariance ordering. Official rendered site denied automated access; official source was read |
| `bonnabel2016covariance` | [Inspected v3 manuscript](https://arxiv.org/pdf/1410.7632v3) | §§II–IV: full noise covariance in Eq. (8), point-to-plane rematching theorem, and its surface/curvature assumptions; no reused-map covariance derivation |

The new 800-particle consistency check validates the code only under its
declared independent pose/IMU noise model. It does not measure the real BNO
attitude covariance, remove registration-to-map correlations, or calibrate
global tunnel uncertainty. No new citation is evidence that gravity alone
resolves translation along the tunnel's primary curve.

### Map correlation and consistency follow-up

| BibTeX key | Primary source | Reading scope and use |
|---|---|---|
| `geneva2019schmidt` | [SEVIS, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Geneva_An_Efficient_Schmidt-EKF_for_3D_Visual-Inertial_SLAM_CVPR_2019_paper.html) · [manuscript](https://arxiv.org/html/1903.08636v1) | §§3.1, 4.3.2, 4.4: active/map cross-covariance, zero map-state gain, and bounded per-update work. This is visual feature SLAM, not an implemented GICP uncertainty model |
| `barrau2016consistent` | [An EKF-SLAM algorithm with consistency properties, inspected v3](https://arxiv.org/html/1510.06263v3) | §§3–5: false observability, alternative invariant error, and preservation of global-frame unobservable directions. The presented point-landmark SLAM model does not establish tunnel correspondence robustness |
| `huang2010observability` | [Observability-based rules, IJRR 2010](https://journals.sagepub.com/doi/10.1177/0278364909353640) | Publisher abstract and issue metadata: OC-EKF/FEJ and spurious covariance reduction. Full paper not read through the publisher's restricted-access page |
| `shan2020liosam` | [LIO-SAM v3](https://arxiv.org/pdf/2007.00258v3) · [author implementation](https://github.com/TixiaoShan/LIO-SAM) | §§III-A/B and III-E: graph states, IMU factors, and loop constraints. Author README distinguishes rebuilt global maps from accumulated clouds whose old poses remain unchanged. No loop backend is implemented here |

The [workspace follow-up](../../../docs/dliio-timed-observer-experiments.md)
translates these leads into proposed tests and a possible bounded keyframe
state model. None is silently credited to the current conditional observer.
