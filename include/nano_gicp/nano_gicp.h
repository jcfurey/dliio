/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

/*
 * NanoGICP: a multi-threaded GICP scan-to-map matcher, FORKED FROM fast_gicp and
 * extended here. Attribution (full citations in doc/REFERENCES.md §1):
 *
 *   - fast_gicp — Kenji Koide (koide3), BSD-3-Clause License:
 *       https://github.com/koide3/fast_gicp
 *     The GICP covariance/Mahalanobis core and the PCL Registration interface
 *     descend from this (VGICP: Koide, Yokozuka, Oishi, Banno, ICRA 2021).
 *     RETAIN fast_gicp's BSD-3-Clause LICENSE when redistributing this file.
 *   - Generalized-ICP — Segal, Haehnel, Thrun, RSS 2009 (the plane-to-plane
 *     covariance formulation).
 *
 * Extensions added in this fork:
 *   - block-wise degeneracy gate / solution remapping — Zhang, Kaess, Singh,
 *     ICRA 2016;
 *   - COIN-LIO-style LiDAR intensity-image frame-to-map term — Pfreundschuh
 *     et al., ICRA 2024 — with FAST-LIVO-style occlusion/range rejection —
 *     Zheng et al., IROS 2022;
 *   - intensity/reflectivity and direct-camera photometric residuals.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include "nano_gicp/nanoflann_adaptor.h"

namespace nano_gicp {

using Covariance = Eigen::Matrix4f;
using CovarianceList = std::vector<Covariance, Eigen::aligned_allocator<Covariance>>;
using Mahalanobis = Eigen::Matrix4f;
using MahalanobisList = std::vector<Mahalanobis, Eigen::aligned_allocator<Mahalanobis>>;
using GradientList = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

// Per-map-point photometric reference for the FRAME-TO-MAP term: the brightness
// (camera or LiDAR-intensity) of this map point as seen by the keyframe that
// first observed it, plus that point's position in the observing keyframe's
// camera frame (a fixed viewing ray, for viewpoint/occlusion gating). `valid`
// marks points that were in-FOV / in-front / gradient-bearing at sampling time.
struct VisualRef {
  Eigen::Vector3f p_kf_cam{0.f, 0.f, 0.f};
  float ref{0.f};
  uint8_t valid{0};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
using VisualRefList = std::vector<VisualRef, Eigen::aligned_allocator<VisualRef>>;

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

// Soft degeneracy gate keep-fraction: maps a (geometric) Hessian eigenvalue to
// the fraction of the GICP update that is KEPT along that eigen-direction, in
// [0,1] (1 = trust the data fully, 0 = hold the IMU prior fully). The original
// gate is the binary limit: keep 1 above the threshold, 0 at/below it. With
// softness > 0 the keep-fraction ramps smoothly (smoothstep) across a symmetric
// band of half-width ln(1+softness) in log-eigenvalue space centred on thresh,
// so a direction hovering near the threshold no longer flips between full-trust
// and full-hold scan-to-scan (the chatter that seeds divergence in the chaotic
// tunnel basin). softness <= 0 reproduces the binary gate bit-for-bit.
// Free function (type-independent, pure) so it is cheap to unit-test.
double softGateKeepFraction(double eigval, double thresh, double softness);

template<typename PointSource, typename PointTarget>
class NanoGICP : public pcl::Registration<PointSource, PointTarget> {

public:
  using Ptr = std::shared_ptr<NanoGICP<PointSource, PointTarget>>;
  using ConstPtr = std::shared_ptr<const NanoGICP<PointSource, PointTarget>>;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

public:
  NanoGICP();
  virtual ~NanoGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setRegularizationMethod(RegularizationMethod method);
  void setMaxCorrespondenceDistance(float corr);
  void setTransformationEpsilon(float eps);
  void setRotationEpsilon(float eps);
  void setInitialLambdaFactor(float lambda);

  const CovarianceList& getSourceCovariances() const;

  void setPhotometricWeight(float weight);
  void setGradientKNeighbors(int k);
  // Selects which point field feeds the photometric term:
  // false = intensity (default), true = reflectivity.
  void setPhotometricChannel(bool use_reflectivity);
  // Full-scale value of the photometric channel (default 255). The channel is
  // divided by this before gradient estimation and residuals, so
  // photometricWeight is in normalized units and transfers across sensors.
  void setPhotometricScale(float scale);
  // Huber threshold on the (normalized) photometric residual; residuals
  // beyond it are IRLS-downweighted. <= 0 disables robustification.
  void setPhotometricHuberDelta(float delta);

  // --- Direct visual (camera) photometric term (off by default) ---
  // Frame-to-frame direct image alignment using LiDAR depth. Each current-scan
  // world point is projected into the PREVIOUS camera image at the current pose
  // estimate (pose-dependent) and compared to its brightness in the CURRENT
  // image (a fixed, pose-independent reference). The residual is accumulated
  // into the SAME 6x6 Hessian as the geometric/LiDAR-photometric terms, BEFORE
  // the degeneracy gate, so a camera-constrained tunnel axis lifts that block's
  // eigenvalue above the gate threshold and the gate stops holding the prior
  // there. Images are single-channel CV_32F, normalized to the same units as
  // the residual (i.e. divide 8-bit by 255). No-op unless enabled and both
  // frames are set.
  void setVisualEnabled(bool on);
  void setVisualWeight(float weight);
  // Huber threshold on the (normalized) visual residual; <= 0 disables.
  void setVisualHuberDelta(float delta);
  void setVisualIntrinsics(float fx, float fy, float cx, float cy);
  // Current frame: reference brightness image + world->camera transform (built
  // from the prior pose). Raw (uncorrected) world points project here.
  void setVisualCurrentFrame(const cv::Mat& image_norm, const Eigen::Isometry3f& T_cam_world);
  // Previous frame: warp-target image + world->camera transform (previous
  // optimized pose). Pose-corrected world points project here.
  void setVisualPreviousFrame(const cv::Mat& image_norm, const Eigen::Isometry3f& T_cam_world);
  // Safety floor for the degeneracy gate when the visual term is on. The gate
  // judges observability from LiDAR geometry alone; on a geometrically
  // degenerate axis the visual term is allowed to drive motion ONLY if it
  // actually stiffened that axis, and even then its total deviation from the
  // IMU prior along that axis is bounded to these PER-SCAN budgets (summed
  // across LM iterations) so a wrong/biased visual constraint cannot inflate
  // the path or run away. Axes the visual term does not rescue stay held to
  // the prior (LiDAR-only behavior). max_trans [m/scan], max_rot [rad/scan].
  void setVisualGateMaxStep(float max_trans, float max_rot);

  // --- Frame-to-MAP camera term (absolute anchor; off until refs are set) ---
  // Unlike the frame-to-frame term above, this projects FIXED map points (with
  // a stored per-point reference brightness, see setTargetVisualRefs) into the
  // CURRENT camera and compares to the reference, anchoring absolute position to
  // map landmarks (e.g. tunnel-wall graffiti). Uses the same current image /
  // intrinsics set above; reuses T_cw_cur_ as the world->current-camera built
  // from the prior pose. Weight and per-scan gate budget are separate from the
  // frame-to-frame term's.
  void setVisualMapWeight(float weight);
  void setVisualMapGateMaxStep(float max_trans, float max_rot);
  void setVisualMapViewAngleMax(float radians);   // reject refs whose viewing ray moved more than this
  void setTargetVisualRefs(const std::shared_ptr<const VisualRefList>& refs);

  // --- COIN-LIO-style LiDAR intensity-image term (frame-to-MAP) ---
  // Anchors absolute position to wall texture seen in the LiDAR reflectivity
  // image. Like the camera frame-to-map term but the reference brightness is the
  // map point's own (range-normalized) reflectivity field, and the projection is
  // the sensor's spherical model (azimuth->col, elevation->row), self-calibrated
  // from the organized scan. The current scan's reflectivity image + projection
  // + world->lidar transform are set per scan.
  void setLidarMapWeight(float weight);
  void setLidarImage(const cv::Mat& refl_norm);  // CV_32FC1, reflectivity/scale
  // Linear spherical model: col = (atan2(Y,X) - az_b)/az_a, row = (elev - el_b)/el_a.
  void setLidarProjection(float az_a, float az_b, float el_a, float el_b);
  void setLidarFrame(const Eigen::Isometry3f& T_lidar_world);  // world->lidar from the prior pose
  // Optional per-row elevation LUT (size = image rows). OS-series beam
  // elevations are NON-uniform, so the linear el(row) above leaves a
  // row-dependent projection bias; when this LUT is set it supersedes el_a/el_b
  // (el->row by monotonic interpolation, Jacobian uses the local slope). Empty
  // = use the linear model.
  void setLidarElevationLut(const std::vector<float>& el_per_row);
  // Optional range image (CV_32FC1, same dims as the reflectivity image, range
  // in metres, <=0 = no return) + consistency tolerance. When set, a map point
  // whose range to the current sensor disagrees with the pixel's range by more
  // than tol = max(abs_tol, rel_tol*range) is rejected as occluded / wrong-
  // surface, removing the dominant source of spurious frame-to-map residuals.
  // Empty image = no occlusion check (original behavior).
  void setLidarRangeImage(const cv::Mat& range_img);
  void setLidarRangeConsistency(float abs_tol, float rel_tol);
  float lastLidarMapRms() const;
  int lastLidarMapCount() const;

  // RMS of the (normalized) visual residual and number of points used in the
  // last align() (for diagnostics).
  float lastVisualRms() const;
  int lastVisualCount() const;
  float lastVisualMapRms() const;
  int lastVisualMapCount() const;
  // Geometrically-degenerate axes the visual term rescued (allowed bounded
  // motion on) during the last align(); for diagnostics.
  int lastVisualRescuedDirections() const;

  // Degeneracy gating (solution remapping): the rotation and translation
  // Hessian blocks are eigen-analyzed separately; the update is projected off
  // directions with eigenvalue < ratio * block_lambda_max, so the initial
  // guess (IMU prior) is held along unobservable directions, e.g. the axis of
  // a featureless tunnel. 0 disables the gate. Most discriminative with the
  // PLANE regularization method.
  void setDegeneracyThreshRatio(float ratio);
  // Soft-gate band half-width (multiplicative, dimensionless). 0 (default) keeps
  // the original binary gate (full trust above the threshold, full prior-hold
  // at/below it); >0 ramps the keep-fraction smoothly across eigenvalues in
  // [thresh/(1+softness), thresh*(1+softness)] so near-threshold directions are
  // not toggled abruptly between trust and hold. See softGateKeepFraction().
  void setDegeneracySoftness(float softness);
  // Number of degenerate directions detected during the last align() (0-6).
  int lastDegenerateDirections() const;
  // Per-scan IMU-consistency clamp: bound the TOTAL correction (final pose vs
  // the initial guess / IMU prior) to this translation [m] and rotation [rad]
  // envelope. Over a ~0.1s scan the IMU prior is high-confidence, so the
  // correction (not the motion -- the prior already contains the motion) is
  // tiny; this caps runaway along the intermittently un-gated degenerate axis.
  // 0 (default) disables each cap independently (no behavior change).
  void setMaxCorrection(float max_trans, float max_rot);

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

  // Background-preparation API: registerInputTarget() sets the target cloud and
  // builds the kd-tree + photometric gradients but does NOT compute covariances
  // (supply precomputed ones with setTargetCovariances). shareTargetDataFrom()
  // adopts a target prepared on another instance without recomputing anything,
  // so the expensive work can run off the registration hot path.
  void registerInputTarget(const PointCloudTargetConstPtr& cloud);
  void setTargetCovariances(const std::shared_ptr<const CovarianceList>& covs);
  void shareTargetDataFrom(const NanoGICP& other);

  float source_density_ = 0.0f;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Eigen::Matrix4f& guess) override;

  void linearize(const Eigen::Isometry3f& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b, double* cost = nullptr);
  void update_correspondences(const Eigen::Isometry3f& trans);

  template<typename PointT>
  void calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                             const nanoflann::KdTreeFLANN<PointT>& kdtree,
                             CovarianceList& covariances,
                             float* density = nullptr);

  bool estimate_spatial_intensity_gradient(int target_index, Eigen::Vector3f& gradient) const;
  void calculate_target_intensity_gradients();

  // Adds the direct visual photometric contribution to (H, b) for the given
  // pose `trans` (the same source->target correction the LM loop optimizes).
  // Accumulates onto whatever is already in H/b (call after linearize()).
  // Iterates the source cloud (input_); requires only the visual_* state.
  void accumulateVisualResidual(const Eigen::Isometry3f& trans,
                                Eigen::Matrix<double, 6, 6>* H,
                                Eigen::Matrix<double, 6, 1>* b,
                                double* cost = nullptr);

  // Frame-to-MAP camera contribution: iterates target (map) points that carry a
  // valid VisualRef, projects each FIXED map point into the current camera under
  // `trans`, and accumulates onto (H, b). Jacobian uses the trans-INVERSE
  // perturbation: J = [ +G*skew(p_w) | -G ] (opposite the frame-to-frame term).
  void accumulateVisualMapResidual(const Eigen::Isometry3f& trans,
                                   Eigen::Matrix<double, 6, 6>* H,
                                   Eigen::Matrix<double, 6, 1>* b,
                                   double* cost = nullptr);

  // COIN-LIO LiDAR-intensity frame-to-map contribution: projects each map point
  // into the current reflectivity image via the spherical model and compares to
  // the point's own reflectivity. Same trans-inverse Jacobian shape as the
  // camera map term, with the spherical dpi_L instead of the pinhole.
  void accumulateLidarMapResidual(const Eigen::Isometry3f& trans,
                                  Eigen::Matrix<double, 6, 6>* H,
                                  Eigen::Matrix<double, 6, 1>* b,
                                  double* cost = nullptr);

protected:
  using pcl::Registration<PointSource, PointTarget>::reg_name_;
  using pcl::Registration<PointSource, PointTarget>::input_;
  using pcl::Registration<PointSource, PointTarget>::target_;
  using pcl::Registration<PointSource, PointTarget>::corr_dist_threshold_;
  using pcl::Registration<PointSource, PointTarget>::converged_;
  using pcl::Registration<PointSource, PointTarget>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget>::transformation_epsilon_;

  int num_threads_;
  int k_correspondences_;
  RegularizationMethod regularization_method_;
  
  float rotation_epsilon_;
  float lambda_factor_;

  // Photometric gradient validity bounds (normalized channel units).
  // Variance floor rejects intensity-uniform neighborhoods (channel^2);
  // magnitude bounds reject noise-fit and non-physical gradients (channel/m).
  static constexpr float kGradientVarianceFloor = 1e-6f;
  static constexpr float kGradientMagMin = 1e-6f;
  static constexpr float kGradientMagMax = 100.0f;

  std::shared_ptr<nanoflann::KdTreeFLANN<PointSource>> input_kdtree_;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointTarget>> target_kdtree_;

  CovarianceList source_covs_;
  std::shared_ptr<const CovarianceList> target_covs_;

  std::vector<int> correspondences_;
  std::vector<float> sq_distances_;
  MahalanobisList mahalanobis_;

  float photometric_weight_;
  int gradient_k_neighbors_;
  bool photometric_use_reflectivity_;  // false = intensity, true = reflectivity
  float photometric_scale_;            // channel full-scale; channel is divided by this
  float photometric_huber_delta_;      // Huber threshold (normalized units); <=0 disables
  float degeneracy_thresh_ratio_;
  float degeneracy_softness_;          // soft-gate band half-width; 0 = binary gate
  int last_degenerate_directions_;
  float max_corr_trans_;               // [m]   per-scan total-correction translation cap; 0 = off
  float max_corr_rot_;                 // [rad] per-scan total-correction rotation cap;    0 = off

  std::shared_ptr<const GradientList> target_intensity_gradients_;
  std::shared_ptr<const std::vector<bool>> gradient_valid_;

  // --- Direct visual (camera) photometric term state ---
  bool visual_enabled_;
  float visual_weight_;
  float visual_huber_delta_;
  float visual_fx_, visual_fy_, visual_cx_, visual_cy_;
  cv::Mat visual_cur_;          // current image (fixed reference brightness), CV_32F 1ch
  cv::Mat visual_prev_;         // previous image (pose-dependent warp target), CV_32F 1ch
  Eigen::Isometry3f T_cw_cur_;  // world -> current camera (from prior pose)
  Eigen::Isometry3f T_cw_prev_; // world -> previous camera (previous optimized pose)
  float visual_gate_max_trans_; // gate rescue BUDGET: max total translation deviation per scan [m]
  float visual_gate_max_rot_;   // gate rescue BUDGET: max total rotation deviation per scan [rad]
  float last_visual_rms_;
  int last_visual_count_;
  int last_visual_rescued_;

  // --- Frame-to-MAP camera term state ---
  float visual_map_weight_;
  float visual_map_gate_max_trans_;
  float visual_map_gate_max_rot_;
  float visual_map_view_angle_max_;   // [rad] max viewing-ray deviation before a ref is rejected
  std::shared_ptr<const VisualRefList> target_visual_refs_;
  float last_visual_map_rms_;
  int last_visual_map_count_;

  // --- COIN-LIO LiDAR intensity-image term state ---
  float lidar_map_weight_;
  cv::Mat lidar_image_;               // current reflectivity image, CV_32FC1, /scale
  float lidar_az_a_, lidar_az_b_;     // col = (atan2(Y,X) - az_b) / az_a
  float lidar_el_a_, lidar_el_b_;     // row = (elev - el_b) / el_a (fallback)
  std::vector<float> lidar_el_lut_;   // per-row elevation [rad]; supersedes el_a/el_b when non-empty
  Eigen::Isometry3f T_lw_cur_;        // world -> lidar from the prior pose
  cv::Mat lidar_range_img_;           // current range image [m], CV_32FC1, <=0 invalid (optional)
  float lidar_range_abs_tol_;         // occlusion tolerance: absolute [m]
  float lidar_range_rel_tol_;         // occlusion tolerance: relative (fraction of range)
  float last_lidar_map_rms_;
  int last_lidar_map_count_;
};

} // namespace nano_gicp
