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

// Mass-normalization scale for a residual term whose Hessian contribution scales
// with the (variable) number of valid residuals that scan. Multiplying the
// term's accumulated H/b/cost by this makes the tuned `weight` count-independent
// and comparable across terms. refcount <= 0 -> 1.0 (normalization OFF: raw,
// bit-identical); refcount > 0 with count <= 0 -> 0.0 (no points, no
// contribution); otherwise refcount/count. Pure free function (unit-tested).
double refCountScale(double refcount, long count);

// Condition-scaled (directional) reweighting of a map term's Hessian/gradient
// contribution, judged from the GEOMETRIC Hessian's eigenbasis (so "weak" is a
// geometry property, never the term's own). For each 3x3 block of `H_geo` (rot =
// 0..2, trans = 3..5): eigendecompose, form S_block = V·diag(alpha_k)·Vᵀ with
// alpha_k = clamp((lambda_max/lambda_k)^power, 1, cap), then with
// S = blockdiag(S_rr, S_tt) apply *H_term = S·(*H_term)·S, *b_term = S·(*b_term)
// (equivalent to J -> J·S: raises the term's stiffness AND drive along weak
// directions by alpha²/alpha; strong directions alpha≈1 stay ~unchanged). Keeps
// H_term symmetric PSD. A block with lambda_max <= 0 is left untouched (identity).
// Lets a map term clear the degeneracy-gate rescue bar on a geometrically-weak
// axis without inflating the strong axes (doc/REVIEW.md #8). Pure free function
// (unit-tested); defined in nano_gicp.cc next to refCountScale.
void conditionScaleTerm(const Eigen::Matrix<double, 6, 6>& H_geo,
                        double power, double cap,
                        Eigen::Matrix<double, 6, 6>* H_term,
                        Eigen::Matrix<double, 6, 1>* b_term);

// Direction-SEPARATED fusion of a map term (LOFF-style; doc/EXPLORATION_2026-06-26
// #2). Restrict an auxiliary term to the GEOMETRICALLY-WEAK (degenerate) subspace
// so it can ONLY move the unobservable axis and cannot perturb the well-observed
// axes -- the opposite knob to conditionScaleTerm (which boosts the weak axis but
// keeps the strong ones). For each 3x3 block of `H_geo`, P_block = sum over the
// eigen-directions with eigenvalue <= ratio*lambda_max of v·vᵀ (orthogonal
// projector onto the weak subspace); with P = blockdiag(P_rr, P_tt) apply
// *H_term = P·(*H_term)·P, *b_term = P·(*b_term). Keeps H_term symmetric PSD. A
// well-observed block (no weak direction) -> P_block = 0 -> the term is suppressed
// there (so a non-degenerate scan can't be perturbed by it). A block with
// lambda_max <= 0 (no observability info) is left untouched (identity), and
// ratio <= 0 leaves the whole term untouched (feature OFF). Pure free function
// (unit-tested), defined next to conditionScaleTerm.
void directionSeparateTerm(const Eigen::Matrix<double, 6, 6>& H_geo, double ratio,
                           Eigen::Matrix<double, 6, 6>* H_term,
                           Eigen::Matrix<double, 6, 1>* b_term);

// Margin-adaptive clamp scale: shrinks the IMU-consistency clamp cap as the
// geometric trust margin degrades, so the correction is bounded harder toward the
// IMU prior exactly where observability collapses (e.g. a specular floor dropout
// taking the vertical/pitch constraint). Returns a multiplier in [clamp_floor, 1]:
// 1 when margin >= margin_hi (healthy, full base cap), clamp_floor when
// margin <= margin_lo (degenerate, tightest), linear between. margin < 0 (gate
// off / not computed) returns 1 (no tightening when observability is unknown).
double clampScaleFromMargin(double margin, double margin_lo, double margin_hi,
                            double clamp_floor);

// Probabilistic degeneracy keep-fraction -- a tractable adaptation of Hatleskog &
// Alexis, "Probabilistic Degeneracy Detection for Point-to-Plane Error
// Minimization," IEEE RA-L 2024 (arXiv:2410.10784). Returns the per-direction
// confidence that the geometric signal exceeds the noise floor by a factor
// `confidence_s`, used to softly attenuate the update along that eigen-direction
// (their pseudo-inverse scaling lambda+ = p / lambda). p = Phi( ln(eigval /
// (s*noise_floor)) / spread ), Phi = standard-normal CDF: p=0.5 at eigval =
// s*noise_floor, ->1 well above, ->0 well below. Unlike a ratio*lambda_max
// threshold, `noise_floor` is an ABSOLUTE (sensor-noise-derived) quantity, so the
// observability test transfers across scenes/sensors. NOTE: the paper forward-
// propagates sensor noise to get a PER-DIRECTION noise variance; here noise_floor
// is a configured per-block scalar (the deferred faithful extension). spread<=0
// gives a hard step at s*noise_floor; noise_floor<=0 returns 1 (no model).
double probGateKeepFraction(double eigval, double noise_floor, double confidence_s,
                            double spread);

// --- Barron adaptive robust kernel (J. T. Barron, "A General and Adaptive Robust
// Loss Function," CVPR 2019, arXiv:1701.03077) with per-iteration shape fitting
// (Chebrolu, Laebe, Vysotska, Behley, Stachniss, "Adaptive Robust Kernels for
// Non-Linear Least Squares Problems," IEEE RA-L 2021, arXiv:2004.14938). Restricted
// to alpha in (0,2] (L2 at 2 -> pseudo-Huber at 1 -> Cauchy as alpha->0); the
// alpha<0 truncated-loss extension is deferred. c is the kernel scale. ---
// Loss of residual r (Barron's f); used in the negative-log-likelihood alpha fit.
double barronRho(double r, double alpha, double c);
// IRLS weight MULTIPLIER in (0,1]: 1 at r=0, redescending for |r|>0 when alpha<2.
// Drop-in replacement for the Huber multiplier on a robustified residual.
double barronRelWeight(double r, double alpha, double c);
// log of the partition Z(alpha) = integral exp(-rho(x,alpha,1)) dx (the NLL's
// normalizer; precomputed/cached). Prevents the fit collapsing to "all outliers".
double barronLogPartition(double alpha);
// Fit alpha to a residual set by minimizing the NLL sum_i rho(r_i,alpha,c) +
// N*logZ(alpha) over a grid in [alpha_lo, alpha_hi] (clamped to (0,2]).
double fitBarronAlpha(const std::vector<float>& residuals, double c,
                      double alpha_lo, double alpha_hi);
// Robust scale estimate (1.4826 * median|r|, the MAD sigma) for the Barron c, so
// the kernel's scale tracks the residual noise instead of a hand-set constant.
double barronScaleMad(const std::vector<float>& residuals);

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
  // Mass-normalize the photometric term to this nominal residual count so
  // photometricWeight is independent of how many valid-gradient matches a scan
  // has (and comparable to the other terms). 0 (default) = OFF (raw mass,
  // bit-identical). Enabling RE-SCALES the effective weight (weight*ref/count),
  // so the tuned photometricWeight must be re-derived once. See refCountScale().
  void setPhotometricRefCount(float ref_count);
  // Number of photometric residuals accumulated during the last align() (the
  // valid-gradient correspondences). Tracked regardless of normalization.
  int lastPhotometricCount() const;
  // Unweighted photometric residual RMS from the last align() (brightness-
  // constancy fit quality; lower is better). 0 if the term did not engage.
  float lastPhotometricRms() const;
  // Geometric observability telemetry from the last align(): the rotation /
  // translation Hessian block's weakest-axis eigenvalue divided by the gate
  // threshold. >1 = above the gate (trusted), <1 = held as degenerate, ~1 =
  // marginal (chatter zone). -1 if the gate was disabled. Read-only; the solve
  // is unaffected.
  float lastGeoRotMargin() const;
  float lastGeoTransMargin() const;

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
  // Mass-normalize the frame-to-frame visual term to this nominal residual
  // count (see setPhotometricRefCount). 0 (default) = OFF (raw, bit-identical).
  void setVisualRefCount(float ref_count);
  // Huber threshold on the (normalized) visual residual; <= 0 disables.
  void setVisualHuberDelta(float delta);
  void setVisualIntrinsics(float fx, float fy, float cx, float cy);
  // Current frame: reference brightness image + world->camera transform (built
  // from the prior pose). Raw (uncorrected) world points project here.
  void setVisualCurrentFrame(const cv::Mat& image_norm, const Eigen::Isometry3f& T_cam_world);
  // Previous frame: warp-target image + world->camera transform (previous
  // optimized pose). Pose-corrected world points project here.
  void setVisualPreviousFrame(const cv::Mat& image_norm, const Eigen::Isometry3f& T_cam_world);
  // Optional DENSE source cloud for the frame-to-frame term: the f2f residual
  // only needs world-frame points to project for brightness, not the voxelised
  // registration cloud (input_). input_ is too sparse to populate the narrow
  // camera FOV (the in-FOV count collapses to ~0 as it shrinks); the full
  // deskewed scan is the same deskewed world frame but ~30x denser. When set
  // (non-null, non-empty), accumulateVisualResidual iterates this cloud strided
  // to <= max_points instead of input_. Null cloud (default) -> iterate input_
  // at stride 1, bit-identical. See doc/FINDINGS_2026-06-22.md Part 1/4.
  void setVisualSource(const PointCloudSourceConstPtr& cloud, int max_points);
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
  // Mass-normalize the frame-to-map visual term to this nominal residual count
  // (see setPhotometricRefCount). 0 (default) = OFF (raw, bit-identical).
  void setVisualMapRefCount(float ref_count);
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
  // Frame-to-FRAME LiDAR flow term (doc/EXPLORATION_2026-06-26.md #2): register the
  // current scan against the PREVIOUS scan's image to observe the along-tunnel
  // motion the frame-to-map reflectivity term can't. weight <= 0 (default) -> off,
  // bit-identical. setLidarFlowPrev supplies the previous image (deep-copied here
  // -- owns the buffer) and the previous scan's world->lidar pose; the caller
  // sets it before align() and direction-separates the term so it only constrains
  // the degenerate axis. Threading: the accumulator snapshots the image (refcount
  // hold) for the loop's lifetime; the member is never mutated in place.
  void setLidarFlowWeight(float weight);
  void setLidarFlowPrev(const cv::Mat& prev_img, const Eigen::Isometry3f& T_lw_prev);
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
  // Full-scale normalization for the COIN-LIO image term (default 255). The image
  // pixel values and each map point's reference brightness are divided by this so
  // the residual is dimensionless; set per CHANNEL (reflectivity ~255, near-IR/
  // ambient ~thousands). Kept separate from the 3D photometric scale.
  void setLidarImageScale(float scale);
  // Condition-scaled directional weighting of the LiDAR-map term (see
  // conditionScaleTerm): boost the term along geometrically-weak axes so it can
  // clear the degeneracy-gate rescue bar without inflating strong axes. power =
  // exponent on (lambda_max/lambda_k), cap = max per-direction boost. enabled =
  // false (default) -> the term accumulates directly, bit-identical.
  void setLidarCondScale(bool enabled, float power, float cap);
  // Direction-separated LiDAR-image fusion (directionSeparateTerm, LOFF-style):
  // restrict the term to the geometrically-weak (degenerate) subspace so it can
  // only move the unobservable axis and cannot perturb the well-observed ones.
  // ratio = weak-subspace bar as a fraction of lambda_max (per block). enabled =
  // false (default) -> the term is added in full, bit-identical. Composes with
  // setLidarCondScale (boost then separate). See doc/EXPLORATION_2026-06-26.md #2.
  void setLidarDirSeparate(bool enabled, float ratio);
  // GenZ-ICP adaptive point-to-plane / point-to-point blend (genz_weight.h, Lee
  // et al. RA-L 2025): on an ill-conditioned scan -- geometric translation block
  // lambda_min/lambda_max below `knee` -- convex-mix an isotropic point-to-point
  // metric into the GICP cost (weight 1-alpha) to regularize the unconstrained
  // axis; a healthy scan stays pure point-to-plane (alpha = 1). `floor` in [0,1]
  // is the smallest plane weight (1 = OFF / bit-identical; ~0.5 a typical enable);
  // `point_weight` [1/m^2] sets the point-to-point metric scale (~ the plane
  // metric's typical eigenvalue). enabled = false (default) -> bit-identical.
  void setGenZWeighting(bool enabled, float floor, float knee, float point_weight);
  // X-ICP ternary localizability gate (xicp_localizability.h, Tuna et al. T-RO
  // 2024). Generalizes the binary degeneracy gate with a SECOND, looser bar: a
  // direction whose block eigenvalue >= full_ratio*lambda_max is fully trusted,
  // <= degeneracyThreshRatio*lambda_max holds the prior (unchanged), and in
  // between gets a controlled PARTIAL admit (linear across the band). full_ratio
  // should exceed degeneracyThreshRatio. When enabled this REPLACES the soft/prob
  // keep-fraction; enabled = false (default) -> the existing gate, bit-identical.
  void setXicpTernary(bool enabled, float full_ratio);
  // Saliency-weighted point selection (saliency_weight.h, anti-dilution). Up-weight
  // rare salient source points (edges/ribs/corners, from the local covariance
  // shape) by up to `boost` so the abundant planar walls do not swamp the weak
  // along-axis DOF; planar points stay at weight 1. enabled = false / boost <= 1
  // (default) -> unit weights, bit-identical, and the per-point saliency
  // eigendecomposition is skipped entirely. See doc/EXPLORATION_2026-06-26.md #1.
  void setSaliencyWeighting(bool enabled, float boost);
  float lastLidarMapRms() const;
  int lastLidarMapCount() const;

  // RMS of the (normalized) visual residual and number of points used in the
  // last align() (for diagnostics).
  float lastVisualRms() const;
  int lastVisualCount() const;
  // Frame-to-frame visual reject breakdown from the last align() (diagnostics):
  // how many source points were dropped behind the camera / out of image bounds
  // / for ~zero gradient. Pinpoints why "Visual Points" is 0.
  int lastVisualRejBehind() const;
  int lastVisualRejOob() const;
  int lastVisualRejGrad() const;
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
  // Probabilistic degeneracy gate (see probGateKeepFraction): when enabled, the
  // per-direction hold strength is a confidence probability that the eigenvalue
  // exceeds an absolute (sensor-noise) floor, instead of the ratio/smoothstep
  // gate. noise_floor_* are per-block (rot rad^2 / trans m^2); 0 falls back to
  // the ratio threshold. Disabled (default) -> bit-identical to the soft gate.
  void setProbabilisticGate(bool enabled, float noise_floor_rot, float noise_floor_trans,
                            float confidence_s, float spread);
  // Barron adaptive robust kernel on the photometric residual (see barron* above):
  // replaces the fixed photometric Huber with Barron's loss whose shape alpha is
  // re-fit to the residual distribution each iteration (no hand-tuned kernel).
  // scale = Barron c (0 -> reuse the photometric Huber delta). Disabled (default)
  // -> the fixed Huber, bit-identical.
  void setAdaptiveKernel(bool enabled, float alpha_lo, float alpha_hi, float scale);
  // Alpha fitted in the last align() (2 = L2/clean, lower = more outliers).
  float lastKernelAlpha() const;
  // Barron scale c in effect in the last align() (data-driven MAD when scale<=0).
  float lastKernelScale() const;
  // Number of degenerate directions detected during the last align() (0-6).
  int lastDegenerateDirections() const;
  // Corruption telemetry (VERIFICATION_2026-06-27): correspondences with an
  // out-of-range (>= target size) index skipped by linearize's guard in the last
  // align(). The kd-tree provably cannot store one, so any nonzero count means
  // the correspondences_ buffer was scribbled by an out-of-bounds writer
  // elsewhere -- surface it (the node logs a throttled warning) instead of
  // letting the old std::out_of_range kill the node. 0 on healthy scans.
  long lastOobCorrespondences() const;
  // World-frame eigen-directions the gate flagged degenerate AND HELD (not
  // visually rescued) in the last align(). The correction was erased along
  // these, so the pose dead-reckons the IMU prior there -- the downstream
  // "degeneracy governor" clamps per-scan motion along them, and covariance
  // inflation marks them untrusted. The vectors live in the same frame as the
  // optimization step dx (left/world perturbation of T_corr), i.e. world.
  // Empty when the gate did not fire (or is disabled).
  const std::vector<Eigen::Vector3d>& lastDegenRotDirs() const;
  const std::vector<Eigen::Vector3d>& lastDegenTransDirs() const;
  // Per-scan IMU-consistency clamp: bound the TOTAL correction (final pose vs
  // the initial guess / IMU prior) to this translation [m] and rotation [rad]
  // envelope. Over a ~0.1s scan the IMU prior is high-confidence, so the
  // correction (not the motion -- the prior already contains the motion) is
  // tiny; this caps runaway along the intermittently un-gated degenerate axis.
  // 0 (default) disables each cap independently (no behavior change).
  void setMaxCorrection(float max_trans, float max_rot);
  // Margin-adaptive clamp: when enabled, the maxCorrection caps are scaled down
  // as the per-axis geometric trust margin degrades (rot margin -> rot cap,
  // trans margin -> trans cap), leaning on the IMU prior under degeneracy (e.g.
  // specular floor dropout). Requires a base cap (>0) and the degeneracy gate on
  // (for the margins). Disabled (default) -> the base caps are used unchanged
  // (bit-identical). See clampScaleFromMargin().
  void setAdaptiveClamp(bool enabled, float margin_lo, float margin_hi, float clamp_floor);

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
                             float* density = nullptr,
                             std::vector<float>* saliency = nullptr);

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

  // Frame-to-FRAME LiDAR flow contribution: project the CORRECTED source points
  // into the PREVIOUS scan's image (left-perturbation Jacobian + spherical dpi).
  // No-op unless setLidarFlowWeight(>0) and a previous image is set.
  void accumulateLidarFlowResidual(const Eigen::Isometry3f& trans,
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
  float lidar_image_scale_;            // COIN-LIO image full-scale (per-channel); separate from photometric_scale_
  float photometric_huber_delta_;      // Huber threshold (normalized units); <=0 disables
  float photometric_ref_count_;        // mass-normalization nominal count; 0 = off (raw)
  int last_photometric_count_;         // valid-gradient residuals in the last align()
  float last_photometric_rms_;         // telemetry: unweighted photometric residual RMS
  float degeneracy_thresh_ratio_;
  float degeneracy_softness_;          // soft-gate band half-width; 0 = binary gate
  // Probabilistic gate (Hatleskog & Alexis RA-L 2024 adaptation); off -> the
  // ratio/smoothstep gate above is used (bit-identical). Noise floors are the
  // absolute per-block (rad^2 / m^2) eigenvalue noise; 0 falls back to the
  // ratio threshold as the floor (probit shape over the existing gate).
  bool  prob_gate_enabled_;
  float prob_noise_floor_rot_;
  float prob_noise_floor_trans_;
  float prob_confidence_s_;            // signal must exceed s * noise floor
  float prob_spread_;                  // probit transition width (log-eigenvalue)
  // Barron adaptive robust kernel on the photometric term (Chebrolu et al.); off
  // -> the fixed Huber, bit-identical. current_alpha_ is re-fit each linearize().
  bool  adaptive_kernel_enabled_;
  float kernel_alpha_lo_;
  float kernel_alpha_hi_;
  float kernel_scale_;                 // Barron scale c; 0 -> reuse photometric_huber_delta_
  float current_alpha_;                // alpha in effect this iteration (lagged fit)
  float last_fit_alpha_;               // diagnostic: alpha fitted in the last linearize()
  float current_kernel_c_;             // Barron scale in effect (data-driven MAD when scale<=0)
  int last_degenerate_directions_;
  long last_oob_corr_count_;           // out-of-range correspondences skipped last align (corruption telemetry)
  float last_geo_rot_margin_;          // telemetry: rot block weakest-axis eig / gate thresh; -1 = n/a
  float last_geo_trans_margin_;        // telemetry: trans block weakest-axis eig / gate thresh; -1 = n/a
  // World-frame eigen-directions held degenerate (not rescued) in the last
  // align(); consumed by the downstream degeneracy governor + cov inflation.
  std::vector<Eigen::Vector3d> last_degen_rot_dirs_;
  std::vector<Eigen::Vector3d> last_degen_trans_dirs_;
  float max_corr_trans_;               // [m]   per-scan total-correction translation cap; 0 = off
  float max_corr_rot_;                 // [rad] per-scan total-correction rotation cap;    0 = off
  bool  adaptive_clamp_enabled_;       // scale the caps by the trust margin; off = base caps
  float clamp_margin_lo_;              // margin at/below which the cap is fully tightened
  float clamp_margin_hi_;              // margin at/above which the base cap is used (no tightening)
  float clamp_floor_;                  // tightest cap fraction (multiplier in (0,1]) at full degeneracy

  std::shared_ptr<const GradientList> target_intensity_gradients_;
  std::shared_ptr<const std::vector<bool>> gradient_valid_;

  // --- Direct visual (camera) photometric term state ---
  bool visual_enabled_;
  float visual_weight_;
  float visual_ref_count_;      // mass-normalization nominal count; 0 = off (raw)
  float visual_huber_delta_;
  float visual_fx_, visual_fy_, visual_cx_, visual_cy_;
  cv::Mat visual_cur_;          // current image (fixed reference brightness), CV_32F 1ch
  cv::Mat visual_prev_;         // previous image (pose-dependent warp target), CV_32F 1ch
  PointCloudSourceConstPtr visual_src_;  // optional dense f2f source (null -> use input_)
  int visual_src_max_;          // stride the dense source down to <= this many points
  Eigen::Isometry3f T_cw_cur_;  // world -> current camera (from prior pose)
  Eigen::Isometry3f T_cw_prev_; // world -> previous camera (previous optimized pose)
  float visual_gate_max_trans_; // gate rescue BUDGET: max total translation deviation per scan [m]
  float visual_gate_max_rot_;   // gate rescue BUDGET: max total rotation deviation per scan [rad]
  float last_visual_rms_;
  int last_visual_count_;
  int last_visual_rej_behind_;   // diagnostics: f2f points rejected behind the camera
  int last_visual_rej_oob_;      // ... rejected outside the image bounds
  int last_visual_rej_grad_;     // ... rejected for ~zero image gradient
  int last_visual_rescued_;

  // --- Frame-to-MAP camera term state ---
  float visual_map_weight_;
  float visual_map_ref_count_;  // mass-normalization nominal count; 0 = off (raw)
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
  float lidar_flow_weight_;            // frame-to-frame flow term weight; 0 = off (bit-identical)
  cv::Mat lidar_flow_prev_img_;       // PREVIOUS scan's image (owned deep copy), CV_32FC1, /scale
  Eigen::Isometry3f T_lw_prev_flow_;  // world -> previous lidar (previous scan's corrected pose)
  int last_lidar_flow_count_;         // diagnostics: flow residuals last scan
  float last_lidar_flow_rms_;
  cv::Mat lidar_range_img_;           // current range image [m], CV_32FC1, <=0 invalid (optional)
  float lidar_range_abs_tol_;         // occlusion tolerance: absolute [m]
  float lidar_range_rel_tol_;         // occlusion tolerance: relative (fraction of range)
  bool  lidar_dir_separated_enabled_; // restrict the lidar term to the weak subspace (LOFF); off = bit-identical
  float lidar_ds_ratio_;              // weak-subspace bar as a fraction of lambda_max
  bool  lidar_cond_scale_enabled_;    // direction-scale the term along weak geom axes; off = bit-identical
  float lidar_cs_power_;              // exponent on (lambda_max/lambda_k) per direction
  float lidar_cs_cap_;               // max per-direction boost (alpha)
  bool  genz_enabled_;                // GenZ point-to-plane/point-to-point blend; off = bit-identical
  float genz_floor_;                  // min point-to-plane weight alpha (1 = off)
  float genz_knee_;                   // trans-block lambda_min/lambda_max at which blending starts
  float genz_point_weight_;           // isotropic point-to-point metric weight [1/m^2]
  float current_genz_alpha_;          // alpha in effect this iteration (lagged); 1 = pure point-to-plane
  bool  xicp_ternary_enabled_;        // X-ICP ternary localizability gate; off = existing binary/soft gate
  float xicp_full_ratio_;             // upper (localizable) bar as a fraction of lambda_max (partial bar = degeneracy_thresh_ratio_)
  bool  saliency_enabled_;            // anti-dilution saliency weighting of the geometric term; off = unit weights
  float saliency_boost_;              // weight of a maximally-salient source point (1 = off)
  std::vector<float> source_saliency_; // per-source-point saliency in [0,1] (filled only when saliency_enabled_)
  float last_lidar_map_rms_;
  int last_lidar_map_count_;
};

} // namespace nano_gicp
