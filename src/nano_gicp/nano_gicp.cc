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
 * NanoGICP — forked from fast_gicp (Kenji Koide, BSD-3-Clause,
 * https://github.com/koide3/fast_gicp) and extended with a degeneracy gate and
 * intensity/visual photometric terms. See nano_gicp.h and doc/REFERENCES.md §1
 * for the full attribution; retain fast_gicp's BSD-3-Clause LICENSE when
 * redistributing.
 */

#include "nano_gicp/nano_gicp.h"
#include "nano_gicp/genz_weight.h"
#include "nano_gicp/xicp_localizability.h"
#include "nano_gicp/saliency_weight.h"
#include "dlio/dlio.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include "nano_gicp/elevation_lut.h"
#include "nano_gicp/lidar_projection.h"

namespace {
template<typename PointT>
float photometricValue(const PointT& point, bool use_reflectivity) {
  if (use_reflectivity) { return point.reflectivity; }
  return std::isnan(point.intensity_corrected) ? point.intensity : point.intensity_corrected;
}

// Bilinear sample of a single-channel CV_32F image. Caller guarantees the 2x2
// neighborhood (floor(u),floor(v))..(+1,+1) is in bounds.
inline float bilinearSample(const cv::Mat& img, float u, float v) {
  const int x0 = static_cast<int>(std::floor(u));
  const int y0 = static_cast<int>(std::floor(v));
  const float ax = u - static_cast<float>(x0);
  const float ay = v - static_cast<float>(y0);
  const float* r0 = img.ptr<float>(y0);
  const float* r1 = img.ptr<float>(y0 + 1);
  const float top = r0[x0] * (1.f - ax) + r0[x0 + 1] * ax;
  const float bot = r1[x0] * (1.f - ax) + r1[x0 + 1] * ax;
  return top * (1.f - ay) + bot * ay;
}

// Require real returns for every pixel touched by interpolation and gradients.
// A missing return is not a black paint mark. Bounds are checked by the caller.
inline bool validRangeSupport(const cv::Mat& range, float u, float v, int patch = 0) {
  const int x = static_cast<int>(std::floor(u));
  const int y = static_cast<int>(std::floor(v));
  for (int row = y - patch - 1; row <= y + patch + 2; ++row) {
    for (int col = x - patch - 1; col <= x + patch + 2; ++col) {
      const float r = range.ptr<float>(row)[col];
      if (!(r > 0.f) || !std::isfinite(r)) { return false; }
    }
  }
  return true;
}

inline double huberCost(float residual, float delta) {
  const double a = std::abs(static_cast<double>(residual));
  return delta > 0.f && a > delta ? delta * (2.0 * a - delta) : a * a;
}

}  // namespace

template class nano_gicp::NanoGICP<dlio::Point, dlio::Point>;

namespace nano_gicp {

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> skew(const Eigen::MatrixBase<Derived>& v) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> m;
    m.setZero();
    m(0, 1) = -v(2);
    m(0, 2) = v(1);
    m(1, 0) = v(2);
    m(1, 2) = -v(0);
    m(2, 0) = -v(1);
    m(2, 1) = v(0);
    return m;
}

// Soft degeneracy-gate keep-fraction (see nano_gicp.h). Pure and
// type-independent so it can be unit-tested without a NanoGICP instance.
double softGateKeepFraction(double eigval, double thresh, double softness) {
    if (softness <= 0.0) {
        // Binary gate: trust strictly above the threshold, hold at/below it.
        // Matches the original `eigval <= thresh => hold the prior` test exactly,
        // so the default path is bit-identical.
        return (eigval > thresh) ? 1.0 : 0.0;
    }
    if (!(thresh > 0.0) || !(eigval > 0.0)) {
        // Non-positive threshold or (near-)singular direction: hold the prior.
        return 0.0;
    }
    // Symmetric band in log-eigenvalue space (scale-free): half-width W, centred
    // on the threshold. eigval = thresh/(1+softness) -> 0, thresh -> 0.5,
    // thresh*(1+softness) -> 1.
    const double W = std::log1p(softness);        // > 0
    const double r = std::log(eigval / thresh);   // 0 at the threshold
    const double u = (r + W) / (2.0 * W);         // 0 at lower edge, 1 at upper
    if (u <= 0.0) { return 0.0; }
    if (u >= 1.0) { return 1.0; }
    return u * u * (3.0 - 2.0 * u);               // smoothstep (C1, monotone)
}

// Term mass-normalization scale (see nano_gicp.h). Pure and type-independent so
// it is unit-tested directly. refcount=1000 reproduces the LiDAR-image term's
// historical kLidarRefCount factor exactly.
double refCountScale(double refcount, long count) {
    if (refcount <= 0.0) { return 1.0; }            // normalization off: raw mass
    if (count <= 0)      { return 0.0; }            // on, but no residuals
    return refcount / static_cast<double>(count);
}

// Condition-scaled directional reweighting of a map term (see nano_gicp.h).
void conditionScaleTerm(const Eigen::Matrix<double, 6, 6>& H_geo,
                        double power, double cap,
                        Eigen::Matrix<double, 6, 6>* H_term,
                        Eigen::Matrix<double, 6, 1>* b_term) {
    if (cap < 1.0) { cap = 1.0; }
    Eigen::Matrix<double, 6, 6> S = Eigen::Matrix<double, 6, 6>::Identity();
    for (int blk = 0; blk < 2; ++blk) {
        const int o = 3 * blk;                      // 0 = rotation, 3 = translation
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H_geo.block<3, 3>(o, o));
        const double lmax = es.eigenvalues()(2);    // ascending order: (2) is largest
        if (!(lmax > 0.0)) { continue; }            // empty/degenerate block: identity
        Eigen::Vector3d alpha;
        for (int k = 0; k < 3; ++k) {
            const double lam = es.eigenvalues()(k);
            double a = (lam > 0.0) ? std::pow(lmax / lam, power) : cap;
            a = std::min(std::max(a, 1.0), cap);    // boost weak (>=1), cap, never shrink
            alpha(k) = a;
        }
        S.block<3, 3>(o, o) =
            es.eigenvectors() * alpha.asDiagonal() * es.eigenvectors().transpose();
    }
    *H_term = S * (*H_term) * S;                     // J -> J·S : stiffness x alpha²
    *b_term = S * (*b_term);                         //            drive     x alpha
}

// Direction-separated fusion of a map term (see nano_gicp.h). Pure, unit-tested.
void directionSeparateTerm(const Eigen::Matrix<double, 6, 6>& H_geo, double ratio,
                           Eigen::Matrix<double, 6, 6>* H_term,
                           Eigen::Matrix<double, 6, 1>* b_term) {
    if (!(ratio > 0.0)) { return; }                  // off -> term untouched
    Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Identity();
    for (int blk = 0; blk < 2; ++blk) {
        const int o = 3 * blk;                       // 0 = rotation, 3 = translation
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H_geo.block<3, 3>(o, o));
        const double lmax = es.eigenvalues()(2);     // ascending: (2) is largest
        if (!(lmax > 0.0)) { continue; }             // no observability info: keep identity
        const double thresh = ratio * lmax;
        Eigen::Matrix3d Pb = Eigen::Matrix3d::Zero();
        for (int k = 0; k < 3; ++k) {                // project onto the WEAK directions
            if (es.eigenvalues()(k) <= thresh) {
                const Eigen::Vector3d v = es.eigenvectors().col(k);
                Pb += v * v.transpose();
            }
        }
        P.block<3, 3>(o, o) = Pb;                     // 0 if the block is fully observed
    }
    *H_term = P * (*H_term) * P;                      // J -> J·P : keep only the weak-subspace action
    *b_term = P * (*b_term);
}

// Margin-adaptive clamp scale (see nano_gicp.h). Pure and unit-tested.
double clampScaleFromMargin(double margin, double margin_lo, double margin_hi,
                            double clamp_floor) {
    if (margin < 0.0) { return 1.0; }               // unknown observability: don't tighten
    const double floor = (clamp_floor < 0.0) ? 0.0 : (clamp_floor > 1.0 ? 1.0 : clamp_floor);
    if (!(margin_hi > margin_lo)) {                 // degenerate band: step at margin_lo
        return (margin >= margin_hi) ? 1.0 : floor;
    }
    double s = (margin - margin_lo) / (margin_hi - margin_lo);   // 0 at lo, 1 at hi
    if (s <= 0.0) { return floor; }
    if (s >= 1.0) { return 1.0; }
    return floor + (1.0 - floor) * s;               // linear blend
}

// Probabilistic degeneracy keep-fraction (see nano_gicp.h; Hatleskog & Alexis,
// RA-L 2024 adaptation). Pure and unit-tested.
double probGateKeepFraction(double eigval, double noise_floor, double confidence_s,
                            double spread) {
    if (noise_floor <= 0.0) { return 1.0; }         // no noise model -> no attenuation
    if (eigval <= 0.0) { return 0.0; }              // singular direction -> fully held
    const double s = (confidence_s > 0.0) ? confidence_s : 1.0;
    const double center = s * noise_floor;          // eigenvalue at which p = 0.5
    if (spread <= 0.0) { return (eigval > center) ? 1.0 : 0.0; }   // hard step
    const double z = std::log(eigval / center) / spread;
    return 0.5 * std::erfc(-z / 1.4142135623730951);   // Phi(z) = 0.5*erfc(-z/sqrt(2))
}

// --- Barron adaptive robust kernel (CVPR 2019) + Chebrolu et al. alpha fit
// (RA-L 2021). See nano_gicp.h. Restricted to alpha in (0,2]; pure + unit-tested.
namespace {
constexpr double kTwoPi = 6.283185307179586;
}

double barronRho(double r, double alpha, double c) {
    if (c <= 0.0) { return 0.0; }
    const double x2 = (r / c) * (r / c);
    if (alpha >= 2.0) { return 0.5 * x2; }                  // L2
    if (alpha <= 0.0) { return std::log(0.5 * x2 + 1.0); }  // Cauchy/Lorentzian (alpha->0)
    const double a2 = std::abs(alpha - 2.0);
    return (a2 / alpha) * (std::pow(x2 / a2 + 1.0, 0.5 * alpha) - 1.0);
}

double barronRelWeight(double r, double alpha, double c) {
    if (c <= 0.0) { return 1.0; }
    if (alpha >= 2.0) { return 1.0; }                       // L2: no down-weighting
    const double x2 = (r / c) * (r / c);
    const double a2 = std::abs(alpha - 2.0);
    return std::pow(x2 / a2 + 1.0, 0.5 * alpha - 1.0);      // (0,1], redescending
}

double barronLogPartition(double alpha) {
    if (alpha >= 2.0) { return 0.5 * std::log(kTwoPi); }    // Z = sqrt(2 pi)
    // Cache logZ on a fixed alpha grid (0.05 spacing), numerically integrated
    // once (thread-safe function-local static init). Z(alpha) = int exp(-rho).
    static const std::vector<double> table = [] {
        std::vector<double> t(41);
        const double L = 50.0; const int N = 5000; const double h = 2.0 * L / N;
        for (int k = 0; k <= 40; ++k) {
            const double a = 0.05 * k;
            if (a <= 0.0) { t[k] = std::log(std::sqrt(2.0) * 3.14159265358979324); continue; } // Z(0)=pi*sqrt2
            if (a >= 2.0) { t[k] = 0.5 * std::log(kTwoPi); continue; }
            double sum = 0.0;
            for (int i = 0; i <= N; ++i) {
                const double x = -L + i * h;
                const double w = (i == 0 || i == N) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
                sum += w * std::exp(-barronRho(x, a, 1.0));
            }
            t[k] = std::log(sum * h / 3.0);                 // Simpson
        }
        return t;
    }();
    const double a = (alpha < 0.0) ? 0.0 : alpha;
    const double fk = a / 0.05;
    const int k = std::min(39, static_cast<int>(std::floor(fk)));
    const double frac = fk - k;
    return table[k] + frac * (table[k + 1] - table[k]);     // linear interp
}

double fitBarronAlpha(const std::vector<float>& residuals, double c,
                      double alpha_lo, double alpha_hi) {
    if (residuals.empty() || c <= 0.0) { return alpha_hi; }
    const double lo = std::max(0.05, std::min(static_cast<double>(alpha_lo), 2.0));
    const double hi = std::max(lo, std::min(static_cast<double>(alpha_hi), 2.0));
    const int G = 16;
    const double N = static_cast<double>(residuals.size());
    double best_alpha = hi, best_nll = std::numeric_limits<double>::max();
    for (int g = 0; g <= G; ++g) {
        const double alpha = lo + (hi - lo) * (static_cast<double>(g) / G);
        double s = 0.0;
        for (const float r : residuals) { s += barronRho(r, alpha, c); }
        const double nll = s + N * barronLogPartition(alpha);
        if (nll < best_nll) { best_nll = nll; best_alpha = alpha; }
    }
    return best_alpha;
}

double barronScaleMad(const std::vector<float>& residuals) {
    if (residuals.empty()) { return 0.05; }   // fallback ~ the default photometric Huber delta
    std::vector<float> a(residuals.size());
    for (size_t i = 0; i < residuals.size(); ++i) { a[i] = std::abs(residuals[i]); }
    std::nth_element(a.begin(), a.begin() + a.size() / 2, a.end());
    const double med = static_cast<double>(a[a.size() / 2]);  // median|r| ~= MAD (residuals ~0-mean)
    return std::max(1e-4, 1.4826 * med);                      // robust sigma, floored away from 0
}

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::NanoGICP() {
  reg_name_ = "NanoGICP";
  this->num_threads_ = omp_get_max_threads();
  this->k_correspondences_ = 20;
  this->corr_dist_threshold_ = std::numeric_limits<float>::max();
  // MIN_EIG preserves this fork's historical (mislabeled-as-PLANE) behavior;
  // see calculate_covariances for the honest PLANE option.
  this->regularization_method_ = RegularizationMethod::MIN_EIG;
  this->max_iterations_ = 64;
  this->transformation_epsilon_ = 1e-4;
  this->rotation_epsilon_ = 1e-4;
  this->lambda_factor_ = 1e-9;
  this->photometric_weight_ = 0.0f;
  this->gradient_k_neighbors_ = 10;
  this->photometric_use_reflectivity_ = false;
  this->photometric_scale_ = 255.0f;
  this->lidar_image_scale_ = 255.0f;   // COIN-LIO image normalization (per-channel)
  this->photometric_huber_delta_ = 0.05f;
  this->photometric_ref_count_ = 0.0f;   // mass-normalization OFF by default (raw)
  this->last_photometric_count_ = 0;
  this->last_photometric_rms_ = 0.0f;
  this->degeneracy_thresh_ratio_ = 0.0f;  // opt in after environment-specific validation
  this->degeneracy_softness_ = 0.0f;   // binary gate by default (bit-identical)
  this->prob_gate_enabled_ = false;    // probabilistic gate OFF by default
  this->prob_noise_floor_rot_ = 0.0f;
  this->prob_noise_floor_trans_ = 0.0f;
  this->prob_confidence_s_ = 1.0f;
  this->prob_spread_ = 1.0f;
  this->adaptive_kernel_enabled_ = false;   // fixed Huber by default (bit-identical)
  this->kernel_alpha_lo_ = 0.5f;
  this->kernel_alpha_hi_ = 2.0f;
  this->kernel_scale_ = 0.0f;               // 0 -> reuse photometric_huber_delta_
  this->current_alpha_ = 2.0f;
  this->last_fit_alpha_ = 2.0f;
  this->current_kernel_c_ = 0.05f;
  this->last_degenerate_directions_ = 0;
  this->last_oob_corr_count_ = 0;        // corruption telemetry (VERIFICATION_2026-06-27)
  this->last_geo_rot_margin_ = -1.0f;    // telemetry; -1 = gate disabled / not computed
  this->last_geo_trans_margin_ = -1.0f;
  this->max_corr_trans_ = 0.0f;   // per-scan correction clamp OFF by default
  this->max_corr_rot_ = 0.0f;
  this->adaptive_clamp_enabled_ = false;   // margin-adaptive clamp OFF -> base caps
  this->clamp_margin_lo_ = 1.0f;
  this->clamp_margin_hi_ = 3.0f;
  this->clamp_floor_ = 0.25f;
  this->visual_enabled_ = false;
  this->visual_weight_ = 0.0f;
  this->visual_ref_count_ = 0.0f;        // mass-normalization OFF by default (raw)
  this->visual_huber_delta_ = 0.05f;
  this->visual_src_ = nullptr;           // dense f2f source OFF -> iterate input_
  this->visual_src_max_ = 0;
  this->visual_fx_ = this->visual_fy_ = this->visual_cx_ = this->visual_cy_ = 0.0f;
  this->T_cw_cur_ = Eigen::Isometry3f::Identity();
  this->T_cw_prev_ = Eigen::Isometry3f::Identity();
  this->visual_gate_max_trans_ = 0.5f;
  this->visual_gate_max_rot_ = 0.1f;
  this->last_visual_rms_ = 0.0f;
  this->last_visual_count_ = 0;
  this->last_visual_rej_behind_ = 0;
  this->last_visual_rej_oob_ = 0;
  this->last_visual_rej_grad_ = 0;
  this->last_visual_rescued_ = 0;
  this->visual_map_weight_ = 0.0f;
  this->visual_map_ref_count_ = 0.0f;    // mass-normalization OFF by default (raw)
  this->visual_map_gate_max_trans_ = 1.0f;
  this->visual_map_gate_max_rot_ = 0.1f;
  this->visual_map_view_angle_max_ = 0.6f;  // ~34 deg
  this->last_visual_map_rms_ = 0.0f;
  this->last_visual_map_count_ = 0;
  this->lidar_map_weight_ = 0.0f;
  this->lidar_az_a_ = 1.0f; this->lidar_az_b_ = 0.0f;
  this->lidar_el_a_ = 1.0f; this->lidar_el_b_ = 0.0f;
  this->T_lw_cur_ = Eigen::Isometry3f::Identity();
  this->lidar_flow_weight_ = 0.0f;       // frame-to-frame flow term OFF -> bit-identical
  this->lidar_flow_image_ref_ = true;    // image-to-image reference (full-res); false = point field (voxel-averaged)
  this->lidar_flow_patch_ = 0;           // patch half-width; 0 = single pixel
  this->T_lw_prev_flow_ = Eigen::Isometry3f::Identity();
  this->last_lidar_flow_count_ = 0;
  this->last_lidar_flow_rms_ = 0.0f;
  this->lidar_range_abs_tol_ = 0.5f;
  this->lidar_range_rel_tol_ = 0.1f;
  this->lidar_cond_scale_enabled_ = false;  // direction-scaling OFF -> bit-identical
  this->lidar_cs_power_ = 1.0f;
  this->lidar_cs_cap_ = 50.0f;
  this->lidar_dir_separated_enabled_ = false;  // direction-separation OFF -> bit-identical
  this->lidar_ds_ratio_ = 0.05f;               // weak-subspace bar (fraction of lambda_max)
  this->genz_enabled_ = false;       // GenZ point/plane blend OFF -> pure point-to-plane (bit-identical)
  this->genz_floor_ = 1.0f;          // alpha floor 1 = no blend
  this->genz_knee_ = 0.1f;           // start blending below lambda_min/lambda_max = 0.1
  this->genz_point_weight_ = 1.0f;   // isotropic point-to-point metric scale [1/m^2]
  this->current_genz_alpha_ = 1.0f;  // pure point-to-plane until a scan sets it
  this->xicp_ternary_enabled_ = false;  // X-ICP ternary gate OFF -> existing gate (bit-identical)
  this->xicp_full_ratio_ = 0.05f;        // localizable bar; > degeneracy_thresh_ratio_ when enabled
  this->xicp_partial_budget_trans_ = 0.f;  // per-scan partial-band admission cap [m]; 0 = unbudgeted
  this->xicp_partial_budget_rot_ = 0.f;    // per-scan partial-band admission cap [rad]; 0 = unbudgeted
  this->saliency_enabled_ = false;       // anti-dilution saliency weighting OFF -> unit weights (bit-identical)
  this->saliency_boost_ = 1.0f;          // weight of a maximally-salient point (1 = off)
  this->last_lidar_map_rms_ = 0.0f;
  this->last_lidar_map_count_ = 0;
}

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::~NanoGICP() {}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setNumThreads(int n) {
  this->num_threads_ = n;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  this->k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setMaxCorrespondenceDistance(float corr) {
  this->corr_dist_threshold_ = corr;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setTransformationEpsilon(float eps) {
    this->transformation_epsilon_ = eps;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setRotationEpsilon(float eps) {
    this->rotation_epsilon_ = eps;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInitialLambdaFactor(float lambda) {
    this->lambda_factor_ = lambda;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricWeight(float weight) {
    if (weight > 1e-8f && photometric_weight_ <= 1e-8f) { target_gradients_dirty_ = true; }
    this->photometric_weight_ = weight;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setGradientKNeighbors(int k) {
    if (k != gradient_k_neighbors_) { target_gradients_dirty_ = true; }
    this->gradient_k_neighbors_ = k;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricChannel(bool use_reflectivity) {
    if (use_reflectivity != photometric_use_reflectivity_) { target_gradients_dirty_ = true; }
    this->photometric_use_reflectivity_ = use_reflectivity;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricScale(float scale) {
    const float valid_scale = (std::isfinite(scale) && scale > 0.f) ? scale : 1.0f;
    if (valid_scale != photometric_scale_) { target_gradients_dirty_ = true; }
    this->photometric_scale_ = valid_scale;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarImageUseDedicatedChannel(bool enabled) {
    this->lidar_image_use_dedicated_channel_ = enabled;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarImageScale(float scale) {
    this->lidar_image_scale_ = (scale > 0.f) ? scale : 1.0f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricHuberDelta(float delta) {
    this->photometric_huber_delta_ = delta;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricRefCount(float ref_count) {
    this->photometric_ref_count_ = (ref_count > 0.f) ? ref_count : 0.f;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastPhotometricCount() const {
    return this->last_photometric_count_;
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastPhotometricRms() const {
    return this->last_photometric_rms_;
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastGeoRotMargin() const {
    return this->last_geo_rot_margin_;
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastGeoTransMargin() const {
    return this->last_geo_trans_margin_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualEnabled(bool on) {
    this->visual_enabled_ = on;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualWeight(float weight) {
    this->visual_weight_ = weight;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualRefCount(float ref_count) {
    this->visual_ref_count_ = (ref_count > 0.f) ? ref_count : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualHuberDelta(float delta) {
    this->visual_huber_delta_ = delta;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualIntrinsics(float fx, float fy, float cx, float cy) {
    this->visual_fx_ = fx;
    this->visual_fy_ = fy;
    this->visual_cx_ = cx;
    this->visual_cy_ = cy;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualCurrentFrame(
    const cv::Mat& image_norm, const Eigen::Isometry3f& T_cam_world) {
    this->visual_cur_ = image_norm;
    this->T_cw_cur_ = T_cam_world;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualPreviousFrame(
    const cv::Mat& image_norm, const Eigen::Isometry3f& T_cam_world) {
    this->visual_prev_ = image_norm;
    this->T_cw_prev_ = T_cam_world;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualSource(
    const PointCloudSourceConstPtr& cloud, int max_points) {
    this->visual_src_ = cloud;
    this->visual_src_max_ = (max_points > 0) ? max_points : 0;
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastVisualRms() const {
    return this->last_visual_rms_;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastVisualCount() const {
    return this->last_visual_count_;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastVisualRejBehind() const { return this->last_visual_rej_behind_; }

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastVisualRejOob() const { return this->last_visual_rej_oob_; }

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastVisualRejGrad() const { return this->last_visual_rej_grad_; }

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualGateMaxStep(float max_trans, float max_rot) {
    this->visual_gate_max_trans_ = (max_trans > 0.f) ? max_trans : 0.f;
    this->visual_gate_max_rot_ = (max_rot > 0.f) ? max_rot : 0.f;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastVisualRescuedDirections() const {
    return this->last_visual_rescued_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualMapWeight(float weight) {
    this->visual_map_weight_ = weight;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualMapRefCount(float ref_count) {
    this->visual_map_ref_count_ = (ref_count > 0.f) ? ref_count : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualMapGateMaxStep(float max_trans, float max_rot) {
    this->visual_map_gate_max_trans_ = (max_trans > 0.f) ? max_trans : 0.f;
    this->visual_map_gate_max_rot_ = (max_rot > 0.f) ? max_rot : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setVisualMapViewAngleMax(float radians) {
    this->visual_map_view_angle_max_ = radians;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setTargetVisualRefs(const std::shared_ptr<const VisualRefList>& refs) {
    this->target_visual_refs_ = refs;
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastVisualMapRms() const {
    return this->last_visual_map_rms_;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastVisualMapCount() const {
    return this->last_visual_map_count_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarMapWeight(float weight) {
    this->lidar_map_weight_ = weight;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setTargetLidarRefs(
    const std::shared_ptr<const std::vector<float>>& refs) {
    this->target_lidar_refs_ = refs;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarFlowWeight(float weight) {
    this->lidar_flow_weight_ = (weight > 0.f) ? weight : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarFlowMode(bool image_ref, int patch) {
    this->lidar_flow_image_ref_ = image_ref;
    this->lidar_flow_patch_ = std::max(0, std::min(3, patch));   // stack buffers sized for P <= 3
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarFlowPrev(const cv::Mat& prev_img,
    const Eigen::Isometry3f& T_lw_prev, const cv::Mat& prev_range) {
    // Deep copy: own the buffer for the whole next scan, so a concurrent reassign
    // of the node's image can't free it under the parallel flow loop.
    this->lidar_flow_prev_img_ = prev_img.empty() ? cv::Mat() : prev_img.clone();
    this->T_lw_prev_flow_ = T_lw_prev;
    this->lidar_flow_prev_range_ = prev_range.clone();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarImage(const cv::Mat& refl_norm) {
    this->lidar_image_ = refl_norm;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarProjection(float az_a, float az_b, float el_a, float el_b) {
    this->lidar_az_a_ = az_a; this->lidar_az_b_ = az_b;
    this->lidar_el_a_ = el_a; this->lidar_el_b_ = el_b;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarFrame(const Eigen::Isometry3f& T_lidar_world) {
    this->T_lw_cur_ = T_lidar_world;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarElevationLut(const std::vector<float>& el_per_row) {
    this->lidar_el_lut_ = el_per_row;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarRangeImage(const cv::Mat& range_img) {
    this->lidar_range_img_ = range_img;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarRangeConsistency(float abs_tol, float rel_tol) {
    this->lidar_range_abs_tol_ = (abs_tol > 0.f) ? abs_tol : 0.f;
    this->lidar_range_rel_tol_ = (rel_tol > 0.f) ? rel_tol : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarCondScale(bool enabled, float power, float cap) {
    this->lidar_cond_scale_enabled_ = enabled;
    this->lidar_cs_power_ = (power > 0.f) ? power : 0.f;
    this->lidar_cs_cap_   = (cap   > 1.f) ? cap   : 1.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setLidarDirSeparate(bool enabled, float ratio) {
    this->lidar_dir_separated_enabled_ = enabled;
    this->lidar_ds_ratio_ = (ratio > 0.f) ? ratio : 0.f;   // <=0 -> off (term untouched)
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setGenZWeighting(bool enabled, float floor, float knee, float point_weight) {
    this->genz_enabled_ = enabled;
    // floor in [0,1]; >= 1 means no blend (feature effectively off even if enabled).
    this->genz_floor_ = (floor < 1.f) ? ((floor > 0.f) ? floor : 0.f) : 1.f;
    this->genz_knee_ = (knee > 0.f) ? knee : 0.f;
    this->genz_point_weight_ = (point_weight > 0.f) ? point_weight : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setXicpTernary(bool enabled, float full_ratio) {
    this->xicp_ternary_enabled_ = enabled;
    this->xicp_full_ratio_ = (full_ratio > 0.f) ? full_ratio : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setXicpPartialBudget(float max_trans, float max_rot) {
    this->xicp_partial_budget_trans_ = (max_trans > 0.f) ? max_trans : 0.f;
    this->xicp_partial_budget_rot_ = (max_rot > 0.f) ? max_rot : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setSaliencyWeighting(bool enabled, float boost) {
    this->saliency_enabled_ = enabled;
    this->saliency_boost_ = (boost > 1.f) ? boost : 1.f;   // boost <= 1 -> off (unit weights)
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastLidarMapRms() const {
    return this->last_lidar_map_rms_;
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastLidarFlowRms() const {
    return this->last_lidar_flow_rms_;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastLidarFlowCount() const {
    return this->last_lidar_flow_count_;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastLidarMapCount() const {
    return this->last_lidar_map_count_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setDegeneracyThreshRatio(float ratio) {
    this->degeneracy_thresh_ratio_ = ratio;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setDegeneracySoftness(float softness) {
    this->degeneracy_softness_ = (softness > 0.f) ? softness : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setProbabilisticGate(bool enabled, float noise_floor_rot,
                                                             float noise_floor_trans,
                                                             float confidence_s, float spread) {
    this->prob_gate_enabled_ = enabled;
    this->prob_noise_floor_rot_   = (noise_floor_rot   > 0.f) ? noise_floor_rot   : 0.f;
    this->prob_noise_floor_trans_ = (noise_floor_trans > 0.f) ? noise_floor_trans : 0.f;
    this->prob_confidence_s_ = (confidence_s > 0.f) ? confidence_s : 1.f;
    this->prob_spread_ = spread;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setAdaptiveKernel(bool enabled, float alpha_lo,
                                                          float alpha_hi, float scale) {
    this->adaptive_kernel_enabled_ = enabled;
    this->kernel_alpha_lo_ = alpha_lo;
    this->kernel_alpha_hi_ = alpha_hi;
    this->kernel_scale_ = (scale > 0.f) ? scale : 0.f;
    this->current_alpha_ = alpha_hi;        // start at L2 / least-robust, then adapt
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastKernelAlpha() const {
    return this->last_fit_alpha_;
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::lastKernelScale() const {
    return this->current_kernel_c_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setMaxCorrection(float max_trans, float max_rot) {
    this->max_corr_trans_ = (max_trans > 0.f) ? max_trans : 0.f;
    this->max_corr_rot_   = (max_rot   > 0.f) ? max_rot   : 0.f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setAdaptiveClamp(bool enabled, float margin_lo,
                                                         float margin_hi, float clamp_floor) {
    this->adaptive_clamp_enabled_ = enabled;
    this->clamp_margin_lo_ = margin_lo;
    this->clamp_margin_hi_ = margin_hi;
    this->clamp_floor_ = (clamp_floor < 0.f) ? 0.f : (clamp_floor > 1.f ? 1.f : clamp_floor);
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastDegenerateDirections() const {
    return this->last_degenerate_directions_;
}

template <typename PointSource, typename PointTarget>
long NanoGICP<PointSource, PointTarget>::lastOobCorrespondences() const {
    return this->last_oob_corr_count_;
}

template <typename PointSource, typename PointTarget>
const std::vector<Eigen::Vector3d>& NanoGICP<PointSource, PointTarget>::lastDegenRotDirs() const {
    return this->last_degen_rot_dirs_;
}

template <typename PointSource, typename PointTarget>
const std::vector<Eigen::Vector3d>& NanoGICP<PointSource, PointTarget>::lastDegenTransDirs() const {
    return this->last_degen_trans_dirs_;
}

template <typename PointSource, typename PointTarget>
const CovarianceList& NanoGICP<PointSource, PointTarget>::getSourceCovariances() const {
    return source_covs_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  this->regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (!cloud || cloud->empty()) {
    PCL_ERROR("[pcl::NanoGICP::setInputSource] Invalid or empty point cloud dataset given!\n");
    return;
  }

  pcl::Registration<PointSource, PointTarget>::setInputSource(cloud);
  
  input_kdtree_.reset(new nanoflann::KdTreeFLANN<PointSource>(false));
  input_kdtree_->setInputCloud(cloud);

  // Compute per-source-point saliency only when the anti-dilution weighting is
  // on (else the extra per-point eigendecomposition is skipped; source_saliency_
  // stays empty -> linearize falls back to unit weights, bit-identical).
  calculate_covariances(cloud, *input_kdtree_, source_covs_, &source_density_,
                        this->saliency_enabled_ ? &this->source_saliency_ : nullptr);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (!cloud || cloud->empty()) {
    PCL_ERROR("[pcl::NanoGICP::setInputTarget] Invalid or empty point cloud dataset given!\n");
    return;
  }

  registerInputTarget(cloud);

  auto covs = std::make_shared<CovarianceList>();
  calculate_covariances(cloud, *target_kdtree_, *covs);
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (!cloud || cloud->empty()) {
    PCL_ERROR("[pcl::NanoGICP::registerInputTarget] Invalid or empty point cloud dataset given!\n");
    return;
  }

  pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);

  auto kdtree = std::make_shared<nanoflann::KdTreeFLANN<PointTarget>>(false);
  kdtree->setInputCloud(cloud);
  target_kdtree_ = kdtree;

  // covariances are NOT computed here; supply them via setTargetCovariances
  // (computeTransformation falls back to computing them if they are missing)
  target_covs_.reset();

  // Only calculate intensity gradients if photometric weight is enabled
  if (photometric_weight_ > 1e-8) {
      calculate_target_intensity_gradients();
  } else {
      target_intensity_gradients_.reset();
      gradient_valid_.reset();
  }
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setTargetCovariances(const std::shared_ptr<const CovarianceList>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::shareTargetDataFrom(const NanoGICP& other) {
  pcl::Registration<PointSource, PointTarget>::setInputTarget(other.target_);
  target_kdtree_ = other.target_kdtree_;
  target_covs_ = other.target_covs_;
  target_intensity_gradients_ = other.target_intensity_gradients_;
  gradient_valid_ = other.gradient_valid_;
  // A background target may have been built before a live parameter change,
  // or with photometry disabled. Rebuild locally when its gradient units or
  // channel do not match this registration instance.
  target_gradients_dirty_ = other.target_gradients_dirty_ || !gradient_valid_ ||
      photometric_scale_ != other.photometric_scale_ ||
      photometric_use_reflectivity_ != other.photometric_use_reflectivity_ ||
      gradient_k_neighbors_ != other.gradient_k_neighbors_;
  target_visual_refs_ = other.target_visual_refs_;
  target_lidar_refs_ = other.target_lidar_refs_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::calculate_target_intensity_gradients() {
    if (!target_) {
        target_intensity_gradients_.reset();
        gradient_valid_.reset();
        return;
    }

    auto gradients = std::make_shared<GradientList>(target_->size(), Eigen::Vector3f::Zero());
    // vector<bool> packs adjacent flags into one word; concurrent OpenMP
    // writes to different points can otherwise lose each other's bits.
    auto valid = std::make_shared<std::vector<uint8_t>>(target_->size(), 0);

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for (int i = 0; i < target_->size(); ++i) {
        Eigen::Vector3f gradient;
        if(estimate_spatial_intensity_gradient(i, gradient)) {
            (*gradients)[i] = gradient;
            (*valid)[i] = true;
        }
    }

    target_intensity_gradients_ = gradients;
    gradient_valid_ = valid;
    target_gradients_dirty_ = false;
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::estimate_spatial_intensity_gradient(
  int target_index, Eigen::Vector3f& gradient) const {
    
  if (target_index < 0 || !this->target_kdtree_ || target_index >= this->target_->size()) {
      return false;
  }
  
  const int k_neighbors = this->gradient_k_neighbors_;
  std::vector<int> nn_indices(k_neighbors);
  std::vector<float> nn_dists(k_neighbors);
  
  int found_neighbors = this->target_kdtree_->nearestKSearch(
      this->target_->at(target_index), k_neighbors, nn_indices, nn_dists);
  
  if (found_neighbors < 4) {
    return false;
  }
  
  // Read whichever channel (intensity or reflectivity) is selected, normalized
  // by photometric_scale_ so gradients/residuals are dimensionless and
  // photometricWeight transfers across sensors (0-255 intensity, uint16
  // reflectivity, float channels...).
  const bool use_refl = this->photometric_use_reflectivity_;
  const float inv_scale = 1.0f / this->photometric_scale_;
  auto chan = [use_refl, inv_scale](const PointTarget& p) {
    return photometricValue(p, use_refl) * inv_scale;
  };

  Eigen::MatrixXf A(found_neighbors, 4);
  Eigen::VectorXf i(found_neighbors);

  // Solve in coordinates centered on the query point: the gradient is
  // translation-invariant, but with absolute world coordinates cond(AtA)
  // grows ~ ||x||^4, so the condition-number rejection below would fire
  // based on distance from the map origin rather than surface geometry
  // (the photometric term would silently fade out as the map grows).
  const auto& query_pt = this->target_->at(target_index);

  float mean_intensity = 0.0f;
  for (int j = 0; j < found_neighbors; ++j) {
    const auto& pt = this->target_->at(nn_indices[j]);
    A(j, 0) = pt.x - query_pt.x;
    A(j, 1) = pt.y - query_pt.y;
    A(j, 2) = pt.z - query_pt.z;
    A(j, 3) = 1.0f;
    i(j) = chan(pt);
    if (!std::isfinite(i(j))) { return false; }
    mean_intensity += chan(pt);
  }
  mean_intensity /= found_neighbors;

  // Calculate channel variance for validation
  float intensity_variance = 0.0f;
  for (int j = 0; j < found_neighbors; ++j) {
      float diff = chan(this->target_->at(nn_indices[j])) - mean_intensity;
      intensity_variance += diff * diff;
  }
  intensity_variance /= found_neighbors;

  // Reject if the channel is too uniform
  if (intensity_variance < kGradientVarianceFloor) {
      return false;
  }
  
  // Paint on a planar wall has a well-defined tangential gradient, while the
  // normal derivative is unobservable. Requiring a full-rank 3D affine fit
  // discarded precisely these useful wall patches. Fit a minimum-norm
  // gradient in the supported spatial subspace; reject line-like support.
  Eigen::MatrixXf centered = A.leftCols<3>();
  const Eigen::RowVector3f mean_position = centered.colwise().mean();
  centered.rowwise() -= mean_position;
  const Eigen::VectorXf centered_i = i.array() - mean_intensity;
  const Eigen::Matrix3f spread = centered.transpose() * centered;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(spread);
  if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite()) { return false; }
  const float floor = std::max(1e-9f, 1e-4f * eig.eigenvalues()(2));
  if (eig.eigenvalues()(1) <= floor) { return false; }
  Eigen::Vector3f inv;
  for (int k = 0; k < 3; ++k) {
    inv(k) = eig.eigenvalues()(k) > floor ? 1.f / eig.eigenvalues()(k) : 0.f;
  }
  gradient = eig.eigenvectors() * inv.asDiagonal() * eig.eigenvectors().transpose()
             * centered.transpose() * centered_i;
  if (!gradient.allFinite()) { return false; }
  
  // Reject unreasonably large or small gradients
  float gradient_mag = gradient.norm();
  if (gradient_mag > kGradientMagMax || gradient_mag < kGradientMagMin) {
      return false;
  }
  
  return true;
}

template<typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::computeTransformation(
    PointCloudSource& output, const Eigen::Matrix4f& guess) {

    Eigen::Isometry3f trans = Eigen::Isometry3f::Identity();
    trans.matrix() = guess;
    // IMU-prior initial guess, snapshotted for the per-scan correction clamp below.
    const Eigen::Isometry3f trans_init = trans;
    last_surface_texture_ = {};
    last_surface_texture_ratio_ = -1.f;
    SurfaceTextureConstraint surface_constraint;

    // Fallback: if the target was registered without precomputed covariances
    // (or with a mismatched set), compute them here.
    if (!target_covs_ || target_covs_->size() != target_->size()) {
        auto covs = std::make_shared<CovarianceList>();
        calculate_covariances(target_, *target_kdtree_, *covs);
        target_covs_ = covs;
    }

    if (photometric_weight_ > 1e-8f && (target_gradients_dirty_ || !gradient_valid_)) {
        calculate_target_intensity_gradients();
    }

    this->converged_ = false;
    this->last_degenerate_directions_ = 0;
    this->last_oob_corr_count_ = 0;
    this->last_visual_count_ = 0;
    this->last_visual_rms_ = 0.0f;
    this->last_visual_rescued_ = 0;
    this->last_visual_map_count_ = 0;
    this->last_visual_map_rms_ = 0.0f;
    this->last_lidar_map_count_ = 0;
    this->last_lidar_map_rms_ = 0.0f;
    // Read-only observability telemetry: the geometric blocks' trust margin
    // (weakest-axis eigenvalue / gate threshold). >1 = above the gate (trusted),
    // <1 = below (held as degenerate), ~1 = marginal (the chatter zone the soft
    // gate addresses). -1 = not computed (gate disabled). Does NOT affect the solve.
    this->last_geo_rot_margin_ = -1.0f;
    this->last_geo_trans_margin_ = -1.0f;
    // Held-degenerate eigen-directions for the downstream governor + cov
    // inflation; refilled inside the gate, cleared here so a gate-off (or
    // non-firing) scan reports none.
    this->last_degen_rot_dirs_.clear();
    this->last_degen_trans_dirs_.clear();
    // Adaptive kernel: warm-start each scan at the least-robust shape and the
    // default scale, then adapt both (alpha + data-driven c) over the LM loop.
    if (this->adaptive_kernel_enabled_) {
        this->current_alpha_ = this->kernel_alpha_hi_;
        this->current_kernel_c_ = (this->kernel_scale_ > 0.f) ? this->kernel_scale_
                                                              : this->photometric_huber_delta_;
    }
    // GenZ-ICP blend: warm-start each scan at pure point-to-plane (alpha = 1);
    // the per-iteration conditioning estimate below lowers it if degenerate.
    this->current_genz_alpha_ = 1.0f;

    // Step acceptance (retrospective LM-style damping): the plain GN loop
    // took every step unconditionally -- on an ill-conditioned submap a bad
    // step just got worse until NaN. Track the cost; if the last step
    // increased it, revert the pose and retry with stronger damping.
    const double base_lambda = (lambda_factor_ > 0) ? lambda_factor_ : 1e-6;
    constexpr double kLambdaScale = 10.0;
    constexpr double kLambdaMax = 1e6;
    double lambda = base_lambda;
    double prev_cost = std::numeric_limits<double>::max();
    Eigen::Isometry3f prev_trans = trans;

    // Per-SCAN cumulative budget for visual rescue on degenerate axes: the
    // visual term may deviate from the IMU prior along a LiDAR-degenerate axis
    // by at most visual_gate_max_* TOTAL across all LM iterations this scan.
    // This bounds the rescued motion per scan (so it can't inflate the path
    // over a long run -- the over-travel a per-iteration clamp allowed), while
    // still letting vision refine the prior.
    double rescued_t_used = 0.0;  // [m]   translation budget consumed this scan
    double rescued_r_used = 0.0;  // [rad] rotation budget consumed this scan
    // X-ICP partial-band admission budgets (FINDINGS_2026-07-08 runaway fix):
    // the partial band ADMITS keep*comp along a marginal axis with, previously,
    // no cap -- fail-open when the marginal-band signal is dishonest (aliased
    // photometric drive / contaminated map locks). Budgeting the cumulative
    // admitted motion per scan makes X-ICP fail-BOUNDED, matching the rescue
    // path's budget and Tuna et al.'s controlled partial update. 0 = unbudgeted
    // (bit-identical to the pre-budget behavior).
    double xicp_partial_t_used = 0.0;  // [m]
    double xicp_partial_r_used = 0.0;  // [rad]
    double prev_rescued_t = 0.0, prev_rescued_r = 0.0;
    double prev_partial_t = 0.0, prev_partial_r = 0.0;
    bool pending_convergence = false;
    bool proposal_pending = false;
    bool texture_pending = true;
    float next_alpha = current_alpha_, next_kernel_c = current_kernel_c_;
    float next_genz_alpha = current_genz_alpha_;
    update_correspondences(trans);
    const auto has_geometry = [&]() {
        return std::any_of(correspondences_.begin(), correspondences_.end(),
                           [](int index) { return index >= 0; });
    };
    bool base_has_geometry = has_geometry();

    // The extra evaluation validates the final proposed step as well. Never
    // return an untested last step merely because it is small or the iteration
    // budget ran out. Rejected steps also restore their rescue budgets.
    for (int i = 0; i <= this->max_iterations_; ++i) {
        // A trial must retain its base correspondences and covariance metrics.
        // Re-associating first can erase difficult residuals (even ALL of them)
        // and falsely accept a jump into empty space as a zero-cost fit.

        // Accumulated, solved, and gated in double; see linearize() for why.
        Eigen::Matrix<double, 6, 6> H;
        Eigen::Matrix<double, 6, 1> b;
        double cost = 0.0;

        // Snapshot the GEOMETRIC (LiDAR-only) Hessian before adding the visual
        // term. The degeneracy gate judges observability from this, NOT from
        // the visual-augmented H: otherwise a strong-but-wrong visual term
        // masks the degeneracy, the gate releases its prior-hold, and the pose
        // diverges along the (still physically unobservable) axis.
        Eigen::Matrix<double, 6, 6> H_geo;
        const auto evaluate = [&]() {
            const float alpha = current_alpha_, kernel_c = current_kernel_c_;
            const float genz_alpha = current_genz_alpha_;
            cost = 0.0;
            linearize(trans, &H, &b, &cost, &H_geo);

            if (texture_pending && surface_texture_weight_ > 0.f) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(H_geo.template block<3, 3>(3, 3));
                if (eig.info() == Eigen::Success && eig.eigenvalues().allFinite() &&
                    eig.eigenvalues()(2) > 0.0) {
                    last_surface_texture_ratio_ = eig.eigenvalues()(0) / eig.eigenvalues()(2);
                    // A tunnel supplies one weak translation axis. If two axes are
                    // weak, a 1D search cannot separate the unknown motions.
                    if (last_surface_texture_ratio_ < surface_texture_weak_ratio_ &&
                        eig.eigenvalues()(1) > surface_texture_weak_ratio_ * eig.eigenvalues()(2)) {
                        const Eigen::Vector3f axis = eig.eigenvectors().col(0).cast<float>();
                        last_surface_texture_ = matchSurfaceTexture(surface_texture_source_,
                            surface_texture_previous_, trans, axis, surface_texture_config_);
                        if (last_surface_texture_.valid) {
                            surface_constraint.axis = axis;
                            surface_constraint.center = surface_texture_center_;
                            surface_constraint.position = axis.cast<double>().dot(
                                (trans * surface_texture_center_).cast<double>()) + last_surface_texture_.shift;
                            surface_constraint.information = surface_texture_weight_ /
                                (last_surface_texture_.sigma * last_surface_texture_.sigma);
                        }
                    }
                }
            }
            texture_pending = false;
            // Freeze the measured position for ALL LM evaluations, including
            // rejected/final steps. Never re-match it to the proposed pose.
            surface_constraint.accumulate(trans, &H, &b, &cost);

            // GenZ-ICP adaptive blend weight (lagged one iteration, like the kernel):
            // from the geometric translation block's conditioning (lambda_min/lambda_max
            // of H_geo[3:6,3:6]), set how much point-to-plane vs point-to-point the NEXT
            // linearize() mixes. Off (genz_floor_ >= 1) -> alpha stays 1 -> pure
            // point-to-plane (bit-identical). Adapted from GenZ-ICP (Lee et al., RA-L
            // 2025): degenerate scan -> blend in an isotropic point-to-point metric to
            // regularize the unconstrained axis. See nano_gicp/genz_weight.h.
            if (this->genz_enabled_) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_g(H_geo.template block<3, 3>(3, 3));
                const double lmax = eig_g.eigenvalues()(2);
                const double ratio = (lmax > 0.0) ? (eig_g.eigenvalues()(0) / lmax) : 0.0;
                this->current_genz_alpha_ = static_cast<float>(
                    genzPlaneWeight(ratio, static_cast<double>(this->genz_floor_),
                                    static_cast<double>(this->genz_knee_)));
            }

            // Direct visual (camera) photometric term: accumulate into the SAME
            // H/b that linearize() built, BEFORE damping and the degeneracy gate,
            // so a camera-constrained axis can be detected (and rescued) by the
            // gate below. Include the same image residuals in step acceptance: a
            // useful texture-driven step can increase the weak geometric cost.
            // No-op unless setVisualEnabled(true) and both frames are set.
            if (visual_enabled_) {
                accumulateVisualResidual(trans, &H, &b, &cost);
            }
            // Frame-to-MAP camera term (absolute anchor): also into H/b before the
            // gate, so it can rescue the degenerate axis with map landmarks.
            if (visual_map_weight_ > 0.f) {
                accumulateVisualMapResidual(trans, &H, &b, &cost);
            }
            // COIN-LIO LiDAR intensity-image term (absolute anchor via reflectivity).
            if (lidar_map_weight_ > 0.f) {
                if (lidar_cond_scale_enabled_ || lidar_dir_separated_enabled_) {
                    // Reweight the term before adding, judged from the GEOMETRIC
                    // eigenbasis (H_geo, before any term): conditionScale BOOSTS it
                    // toward the weak axes so it can clear the gate's rescue bar there;
                    // directionSeparate RESTRICTS it to the weak subspace (LOFF-style,
                    // so it can only move the degenerate axis, never perturb the strong
                    // ones). Either or both; the per-scan rescue budget still bounds the
                    // resulting motion. See REVIEW.md #8, EXPLORATION_2026-06-26 #2.
                    Eigen::Matrix<double, 6, 6> H_lid = Eigen::Matrix<double, 6, 6>::Zero();
                    Eigen::Matrix<double, 6, 1> b_lid = Eigen::Matrix<double, 6, 1>::Zero();
                    accumulateLidarMapResidual(trans, &H_lid, &b_lid, &cost);
                    if (lidar_cond_scale_enabled_) {
                        conditionScaleTerm(H_geo, static_cast<double>(lidar_cs_power_),
                                           static_cast<double>(lidar_cs_cap_), &H_lid, &b_lid);
                    }
                    if (lidar_dir_separated_enabled_) {
                        directionSeparateTerm(H_geo, static_cast<double>(lidar_ds_ratio_),
                                              &H_lid, &b_lid);
                    }
                    H += H_lid;
                    b += b_lid;
                } else {
                    accumulateLidarMapResidual(trans, &H, &b, &cost);
                }
            }

            // Frame-to-FRAME LiDAR flow term (EXPLORATION #2): accumulate into its own
            // H_flow/b_flow, then direction-separate (when enabled) so the flow only
            // constrains the geometrically-weak axis -- it observes the along-tunnel
            // motion, fused only where it's needed. Weight 0 (default) -> no-op /
            // bit-identical. Judged from H_geo (before any term), like the lidar term.
            if (lidar_flow_weight_ > 0.f) {
                Eigen::Matrix<double, 6, 6> H_flow = Eigen::Matrix<double, 6, 6>::Zero();
                Eigen::Matrix<double, 6, 1> b_flow = Eigen::Matrix<double, 6, 1>::Zero();
                accumulateLidarFlowResidual(trans, &H_flow, &b_flow, &cost);
                if (lidar_dir_separated_enabled_) {
                    directionSeparateTerm(H_geo, static_cast<double>(lidar_ds_ratio_),
                                          &H_flow, &b_flow);
                }
                H += H_flow;
                b += b_flow;
            }
            // Adaptive fits belong to the next accepted model. Applying them
            // between base and trial evaluations would change the objective
            // even though the geometric correspondence set remained fixed.
            next_alpha = current_alpha_; next_kernel_c = current_kernel_c_;
            next_genz_alpha = current_genz_alpha_;
            current_alpha_ = alpha; current_kernel_c_ = kernel_c;
            current_genz_alpha_ = genz_alpha;
        };
        evaluate();

        if (!std::isfinite(cost) || cost > prev_cost + 1e-10 * std::max(1.0, prev_cost)) {
            trans = prev_trans;
            rescued_t_used = prev_rescued_t;
            rescued_r_used = prev_rescued_r;
            xicp_partial_t_used = prev_partial_t;
            xicp_partial_r_used = prev_partial_r;
            pending_convergence = false;
            proposal_pending = false;
            lambda *= kLambdaScale;
            if (lambda > kLambdaMax) { evaluate(); break; }
            continue;
        }
        // A zero-information system is not a converged registration.
        if (H.isZero(0.0)) { trans = prev_trans; break; }
        const bool improved = cost < prev_cost;
        if (proposal_pending) {
            update_correspondences(trans);
            if (base_has_geometry && !has_geometry()) {
                trans = prev_trans;
                update_correspondences(trans);
                rescued_t_used = prev_rescued_t;
                rescued_r_used = prev_rescued_r;
                xicp_partial_t_used = prev_partial_t;
                xicp_partial_r_used = prev_partial_r;
                pending_convergence = false;
                proposal_pending = false;
                lambda *= kLambdaScale;
                if (lambda > kLambdaMax) { evaluate(); break; }
                continue;
            }
            // The accepted pose defines a NEW local ICP model. Its objective
            // becomes the baseline for the next trial; costs from different
            // correspondence sets must not decide acceptance against each other.
            current_alpha_ = next_alpha; current_kernel_c_ = next_kernel_c;
            current_genz_alpha_ = next_genz_alpha;
            evaluate();
            base_has_geometry = has_geometry();
            proposal_pending = false;
        }
        if (!std::isfinite(cost) || !H.allFinite() || !b.allFinite() || H.isZero(0.0)) {
            trans = prev_trans;
            break;
        }
        if (pending_convergence) {
            this->converged_ = true;
            break;
        }
        if (i == this->max_iterations_) { break; }
        // Do not undo increased damping when re-evaluating a reverted pose.
        if (improved) { lambda = std::max(lambda / kLambdaScale, base_lambda); }
        prev_cost = cost;
        prev_trans = trans;
        prev_rescued_t = rescued_t_used;
        prev_rescued_r = rescued_r_used;
        prev_partial_t = xicp_partial_t_used;
        prev_partial_r = xicp_partial_r_used;

        // Snapshot the combined (geometric + photometric) Hessian BEFORE
        // damping: the visual-rescue Rayleigh test below must measure the
        // photometric stiffening of a weak axis, not the LM damping term. If it
        // read H after the diagonal += lambda, an escalated lambda (after a
        // rejected step) could by itself exceed the rescue threshold and
        // falsely "rescue" an axis no photometric term constrained.
        const Eigen::Matrix<double, 6, 6> H_combined = H;

        // Add regularization / damping
        H.diagonal().array() += lambda;

        Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(-b);

        if(dx.hasNaN() || !dx.allFinite()) {
            break;
        }

        // Degeneracy gating (solution remapping, Zhang/Kaess/Singh ICRA'16):
        // in a featureless tunnel the problem is unobservable along the tunnel
        // axis and the solve above pours correspondence-snapping noise into
        // exactly that direction, overwriting the IMU prior. Analyze the
        // GEOMETRIC rotation (H_geo[0:3,0:3], rad^2) and translation
        // (H_geo[3:6,3:6], m^2) blocks SEPARATELY -- a single threshold across
        // the full 6x6 is meaningless because the two blocks have different
        // units and the rotation block additionally scales with the lever arm
        // (range^2).
        //
        // For each geometrically-weak direction v:
        //   - SAFETY FLOOR (visual on): if the visual term actually stiffened v
        //     (combined Rayleigh quotient above threshold), allow motion there
        //     but draw from a per-scan rescue BUDGET (the frame-to-frame cap, or
        //     the larger frame-to-map cap when a map term is active) so a wrong
        //     visual constraint cannot run away; otherwise hold the prior.
        //   - visual off: hold the prior (project the step off v) -- original
        //     gate behavior, bit-identical.
        // NOTE: discrimination is strongest with regularizationMethod 'plane'
        // (scale-free covariance discs); 'min_eig' covariances add artificial
        // in-plane stiffness that partially masks the degeneracy.
        int degenerate = 0;
        int rescued = 0;
        if (degeneracy_thresh_ratio_ > 0.f) {
            // Per-scan rescue budget: the frame-to-MAP term is an ABSOLUTE
            // anchor, so when it is contributing this scan it gets the larger
            // map budget (it can undo a real drag-back); otherwise the tighter
            // frame-to-frame budget applies.
            const bool map_active = (last_visual_map_count_ > 0) || (last_lidar_map_count_ > 0);
            const double cap_t = map_active
                ? static_cast<double>(visual_map_gate_max_trans_) : static_cast<double>(visual_gate_max_trans_);
            const double cap_r = map_active
                ? static_cast<double>(visual_map_gate_max_rot_) : static_cast<double>(visual_gate_max_rot_);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_rr(H_geo.template block<3, 3>(0, 0));
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_tt(H_geo.template block<3, 3>(3, 3));
            // Combined (geometric + visual) blocks, to test visual rescue.
            // Pre-damping (H_combined) so the Rayleigh quotient reflects
            // photometric stiffening, not the LM lambda on the diagonal.
            const Eigen::Matrix3d Hrr = H_combined.template block<3, 3>(0, 0);
            const Eigen::Matrix3d Htt = H_combined.template block<3, 3>(3, 3);

            // For each eigen-direction: a hard-degenerate direction (eigenvalue
            // <= thresh) is counted and, if a visual/map term stiffened it, gets
            // a bounded visual rescue; otherwise the IMU prior is held. The hold
            // strength is a SOFT keep-fraction (softGateKeepFraction): with the
            // default softness 0 it is the binary gate (full hold for degenerate,
            // no-op for observable), bit-identical; with softness > 0 directions
            // near the threshold are partially held instead of toggled, killing
            // the scan-to-scan chatter that seeds divergence.
            const double rr_thresh = degeneracy_thresh_ratio_ * eig_rr.eigenvalues()(2);
            // X-ICP upper (localizable) bar: directions above it are fully
            // trusted; between it and rr_thresh get a partial admit. Only used
            // when the ternary gate is enabled.
            const double rr_full = this->xicp_full_ratio_ * eig_rr.eigenvalues()(2);
            // Telemetry only: weakest-axis margin vs the gate threshold.
            this->last_geo_rot_margin_ = (rr_thresh > 0.0)
                ? static_cast<float>(eig_rr.eigenvalues()(0) / rr_thresh) : -1.0f;
            // Refill the held-direction snapshot each gate pass so it reflects
            // the converged iteration (not the union over LM steps).
            this->last_degen_rot_dirs_.clear();
            this->last_degen_trans_dirs_.clear();
            for (int k = 0; k < 3; ++k) {
                const double lam = eig_rr.eigenvalues()(k);
                const Eigen::Vector3d v = eig_rr.eigenvectors().col(k);
                const double comp = v.dot(dx.head<3>());
                if (lam <= rr_thresh) {
                    ++degenerate;
                    if (v.dot(Hrr * v) > rr_thresh) {
                        // bounded visual-driven motion, drawing from the per-scan budget
                        const double cap = std::max(0.0, cap_r - rescued_r_used);
                        const double cl = std::max(-cap, std::min(cap, comp));
                        dx.head<3>() += v * (cl - comp);
                        rescued_r_used += std::abs(cl);
                        ++rescued;
                        continue;  // rescued: the soft prior-hold below does not apply
                    }
                    // held degenerate (not rescued): record for the governor + cov
                    this->last_degen_rot_dirs_.push_back(v);
                }
                // Probabilistic gate (Hatleskog & Alexis): keep-fraction is the
                // confidence the eigenvalue clears an absolute noise floor; 0
                // floor -> fall back to the ratio threshold (probit shape). Off
                // -> the ratio/smoothstep soft gate (bit-identical).
                // X-ICP ternary gate (Tuna et al., T-RO 2024): a controlled
                // partial admit across [rr_thresh, rr_full], replacing the
                // soft/prob keep-fraction. Off -> the existing gate, bit-identical.
                const double keep = this->xicp_ternary_enabled_
                    ? xicpPartialScale(lam, rr_thresh, rr_full)
                    : (this->prob_gate_enabled_
                        ? probGateKeepFraction(lam,
                            (this->prob_noise_floor_rot_ > 0.f) ? this->prob_noise_floor_rot_ : rr_thresh,
                            this->prob_confidence_s_, this->prob_spread_)
                        : softGateKeepFraction(lam, rr_thresh, this->degeneracy_softness_));
                double admit = comp * keep;
                if (this->xicp_ternary_enabled_ && keep > 0.0 && keep < 1.0) {
                    // budget the partial-band admission (0 cap = unbudgeted)
                    admit = xicpBudgetedAdmit(comp, keep,
                        static_cast<double>(this->xicp_partial_budget_rot_), &xicp_partial_r_used);
                }
                dx.head<3>() -= v * (comp - admit);   // hold everything except the (budgeted) admission
                // Belt-and-suspenders for the X-ICP ternary gate: a PARTIAL-admit
                // band axis (rr_thresh < lam < rr_full, 0 < keep < 1) takes a
                // partial update but is not hard-degenerate, so the block above did
                // not record it for the governor. Record it too, so the degeneracy
                // governor's per-scan cap also bounds a runaway proceeding along a
                // partially-admitted axis (the combo-mode escape in
                // FINDINGS_2026-06-25 PM). Ternary-on only -> the binary/soft/prob
                // gate's recorded set is unchanged (bit-identical).
                if (this->xicp_ternary_enabled_ && lam > rr_thresh && keep < 1.0) {
                    this->last_degen_rot_dirs_.push_back(v);
                }
            }
            const double tt_thresh = degeneracy_thresh_ratio_ * eig_tt.eigenvalues()(2);
            const double tt_full = this->xicp_full_ratio_ * eig_tt.eigenvalues()(2);  // X-ICP localizable bar
            // Telemetry only: weakest-axis margin vs the gate threshold.
            this->last_geo_trans_margin_ = (tt_thresh > 0.0)
                ? static_cast<float>(eig_tt.eigenvalues()(0) / tt_thresh) : -1.0f;
            for (int k = 0; k < 3; ++k) {
                const double lam = eig_tt.eigenvalues()(k);
                const Eigen::Vector3d v = eig_tt.eigenvectors().col(k);
                const double comp = v.dot(dx.tail<3>());
                if (lam <= tt_thresh) {
                    ++degenerate;
                    if (v.dot(Htt * v) > tt_thresh) {
                        // bounded visual-driven motion, drawing from the per-scan budget
                        const double cap = std::max(0.0, cap_t - rescued_t_used);
                        const double cl = std::max(-cap, std::min(cap, comp));
                        dx.tail<3>() += v * (cl - comp);
                        rescued_t_used += std::abs(cl);
                        ++rescued;
                        continue;  // rescued: the soft prior-hold below does not apply
                    }
                    // held degenerate (not rescued): record for the governor + cov
                    this->last_degen_trans_dirs_.push_back(v);
                }
                // X-ICP ternary gate: partial admit across [tt_thresh, tt_full];
                // off -> the existing soft/prob gate (bit-identical).
                const double keep = this->xicp_ternary_enabled_
                    ? xicpPartialScale(lam, tt_thresh, tt_full)
                    : (this->prob_gate_enabled_
                        ? probGateKeepFraction(lam,
                            (this->prob_noise_floor_trans_ > 0.f) ? this->prob_noise_floor_trans_ : tt_thresh,
                            this->prob_confidence_s_, this->prob_spread_)
                        : softGateKeepFraction(lam, tt_thresh, this->degeneracy_softness_));
                double admit = comp * keep;
                if (this->xicp_ternary_enabled_ && keep > 0.0 && keep < 1.0) {
                    admit = xicpBudgetedAdmit(comp, keep,
                        static_cast<double>(this->xicp_partial_budget_trans_), &xicp_partial_t_used);
                }
                dx.tail<3>() -= v * (comp - admit);   // hold everything except the (budgeted) admission
                // Belt-and-suspenders (see the rotation block): record an X-ICP
                // PARTIAL-admit-band translation axis (tt_thresh < lam < tt_full)
                // for the governor too, so its per-scan cap also bounds a runaway
                // along a partially-admitted axis. Ternary-on only -> bit-identical.
                if (this->xicp_ternary_enabled_ && lam > tt_thresh && keep < 1.0) {
                    this->last_degen_trans_dirs_.push_back(v);
                }
            }
        }
        this->last_degenerate_directions_ = std::max(this->last_degenerate_directions_, degenerate);
        this->last_visual_rescued_ = std::max(this->last_visual_rescued_, rescued);

        // Apply transformation update
        // Apply the rotation step as a proper SO(3) exponential (left
        // perturbation) rather than composing per-axis Euler increments,
        // which is only a small-angle approximation and degrades on the
        // large first steps of aggressive motion.
        const Eigen::Vector3f rot_step = dx.head<3>().cast<float>();
        const float angle = rot_step.norm();
        if (angle > 1e-12f) {
            trans.prerotate(Eigen::AngleAxisf(angle, rot_step / angle));
        }
        trans.pretranslate(dx.tail<3>().cast<float>());
        proposal_pending = true;

        // Check convergence: rotation step (dx.head) vs rotation_epsilon_,
        // translation step (dx.tail) vs transformation_epsilon_.
        pending_convergence = dx.head<3>().norm() < rotation_epsilon_ &&
                              dx.tail<3>().norm() < transformation_epsilon_;
    }

    // Per-scan IMU-consistency clamp: bound the TOTAL correction (final GICP
    // pose vs the IMU-prior initial guess) to a configurable envelope. The
    // degeneracy gate above holds the prior only on the eigen-directions it
    // flags, on the ~71-80% of scans it fires; on the scans it misses, a
    // map-lock "jump" otherwise goes through unbounded and seeds a full
    // (deg=6) collapse. Over a ~0.1s scan the IMU prior is high-confidence, so
    // the correction -- NOT the motion; the prior already contains the motion --
    // is physically tiny, and this caps the runaway. Orthogonal to the gate
    // (gate bounds per-iteration step direction; this bounds final magnitude).
    // 0 = off (each cap independent), so default behavior is bit-identical.
    if (this->max_corr_trans_ > 0.f || this->max_corr_rot_ > 0.f) {
        // Margin-adaptive: tighten each cap as that block's trust margin degrades
        // (floor dropout collapses the trans/pitch margin -> tighter cap -> lean on
        // the IMU prior on exactly the axes that lost observability). Disabled ->
        // scale 1 -> base caps unchanged (bit-identical).
        float eff_trans = this->max_corr_trans_;
        float eff_rot   = this->max_corr_rot_;
        if (this->adaptive_clamp_enabled_) {
            eff_trans *= static_cast<float>(clampScaleFromMargin(this->last_geo_trans_margin_,
                this->clamp_margin_lo_, this->clamp_margin_hi_, this->clamp_floor_));
            eff_rot   *= static_cast<float>(clampScaleFromMargin(this->last_geo_rot_margin_,
                this->clamp_margin_lo_, this->clamp_margin_hi_, this->clamp_floor_));
        }
        // World/left-frame correction: trans = corr * trans_init.
        Eigen::Isometry3f corr = trans * trans_init.inverse();
        if (eff_trans > 0.f) {
            const Eigen::Vector3f t = corr.translation();
            const float n = t.norm();
            if (n > eff_trans) {
                corr.translation() = t * (eff_trans / n);
            }
        }
        if (eff_rot > 0.f) {
            Eigen::AngleAxisf aa(corr.rotation());  // .rotation() is orthonormalized
            if (aa.angle() > eff_rot) {
                aa.angle() = eff_rot;
                corr.linear() = aa.toRotationMatrix();
            }
        }
        trans = corr * trans_init;  // recompose
    }

    this->final_transformation_ = trans.matrix();
    pcl::transformPointCloud(*this->input_, output, this->final_transformation_);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3f& trans) {
    correspondences_.assign(input_->size(), -1);
    sq_distances_.assign(input_->size(), std::numeric_limits<float>::max());
    mahalanobis_.assign(input_->size(), Eigen::Matrix4f::Identity());

    std::vector<int> k_indices(1);
    std::vector<float> k_sq_dists(1);
    const int n_target = static_cast<int>(target_->size());

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) \
        firstprivate(k_indices, k_sq_dists)
    for (size_t i = 0; i < input_->size(); ++i) {
        PointTarget transformed_pt;
        transformed_pt.getVector4fMap() = trans * input_->at(i).getVector4fMap();

        // Skip a non-finite transformed query (a diverged deg=6 pose maps source
        // points to NaN/Inf). NOTE (VERIFICATION_2026-06-27): source inspection of
        // the vendored nanoflann shows a non-finite query actually fails SAFE --
        // the leaf loop only admits `dist < worst_dist` (false for NaN/Inf) and
        // KNNResultSet::init resets dists[capacity-1] to FLT_MAX every call, so the
        // distance gate below already rejected these (pinned by test_kdtree.cpp).
        // This guard (and the store conditions below) are defence-in-depth for the
        // still-unattributed out-of-range index of FINDINGS_2026-06-26 (evidence
        // points at memory corruption by an OOB writer elsewhere: the observed
        // garbage indices are bit-patterns of tiny floats), not a demonstrated
        // kd-tree failure path.
        if (!transformed_pt.getVector4fMap().allFinite()) { continue; }

        const int found = target_kdtree_->nearestKSearch(transformed_pt, 1, k_indices, k_sq_dists);

        const float max_dist_sq = this->corr_dist_threshold_ * this->corr_dist_threshold_;
        // Trust k_indices[0] only with a found neighbour, an IN-RANGE index, and a
        // finite in-threshold distance -- defence against a garbage search result
        // (a healthy scan satisfies all of these, so this is bit-identical there).
        if (found > 0 && k_indices[0] >= 0 && k_indices[0] < n_target &&
            std::isfinite(k_sq_dists[0]) && k_sq_dists[0] < max_dist_sq) {
            const Eigen::Matrix4f& source_cov = source_covs_[i];
            const Eigen::Matrix4f& target_cov = (*target_covs_)[k_indices[0]];
            Eigen::Matrix4f RCR = (source_cov + target_cov);
            RCR(3, 3) = 1.0;

            Eigen::Matrix4f mahalanobis = RCR.inverse();
            if (!mahalanobis.allFinite()) { continue; }
            mahalanobis(3, 3) = 0.0;

            correspondences_[i] = k_indices[0];
            sq_distances_[i] = k_sq_dists[0];
            mahalanobis_[i] = mahalanobis;
        }
    }
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::linearize(
    const Eigen::Isometry3f& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* cost,
    Eigen::Matrix<double, 6, 6>* H_geo) {
    
    H->setZero();
    b->setZero();

    // Per-point Jacobians are computed in float (the data is float), but the
    // accumulation across 10k+ points is done in double: summing that many
    // float products loses several significant digits, which matters for the
    // near-singular Hessians the degeneracy gate has to discriminate.
    std::vector<Eigen::Matrix<double, 6, 6>> H_private(num_threads_, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_private(num_threads_, Eigen::Matrix<double, 6, 1>::Zero());
    double cost_sum = 0.0;
    double photo_cost_sum = 0.0;
    double photo_sq_sum = 0.0;   // telemetry only: sum of UNweighted photometric residual^2

    bool use_photometric = (photometric_weight_ > 1e-8)
        && target_intensity_gradients_ && !target_intensity_gradients_->empty();

    // Opt-in photometric mass-normalization: when on, accumulate the photometric
    // contribution into a SEPARATE accumulator so its total Hessian mass can be
    // scaled to a nominal reference count (making photometricWeight independent
    // of the valid-gradient point count, and comparable to the other terms).
    // A separate accumulator also preserves a geometry-only Hessian for the
    // observability gate. The residual count is tallied in either mode.
    const bool normalize_photo = use_photometric && (this->photometric_ref_count_ > 0.f);
    // Keep appearance information out of the geometric observability test.
    const bool separate_photo = use_photometric && (normalize_photo || H_geo != nullptr);
    std::vector<Eigen::Matrix<double, 6, 6>> Hp_private(
        separate_photo ? num_threads_ : 0, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> bp_private(
        separate_photo ? num_threads_ : 0, Eigen::Matrix<double, 6, 1>::Zero());
    std::vector<long> photo_count_private(num_threads_, 0);
    // Adaptive-kernel residual collection (Chebrolu et al.): fit alpha post-loop.
    const bool adaptive_kernel = use_photometric && this->adaptive_kernel_enabled_;
    // Scale c in effect this iteration: explicit kernel_scale_, else the lagged
    // data-driven MAD estimate (current_kernel_c_), updated post-loop.
    const float kernel_c = (this->kernel_scale_ > 0.f) ? this->kernel_scale_ : this->current_kernel_c_;
    std::vector<std::vector<float>> photo_resid_private(adaptive_kernel ? num_threads_ : 0);

    // Telemetry for the FINDINGS_2026-06-26 out-of-range correspondence: the
    // kd-tree provably cannot store one (VERIFICATION_2026-06-27), so any trip of
    // the guard below means the correspondences_ buffer was corrupted by an OOB
    // writer elsewhere. Counting (instead of silently skipping) turns a survivable
    // symptom into a signal the node can surface on the bag.
    long oob_corr = 0;

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:cost_sum,photo_cost_sum,photo_sq_sum,oob_corr)
    for(int i = 0; i < input_->size(); ++i) {
        int thread_num = omp_get_thread_num();
        int target_index = correspondences_[i];

        // Defence in depth: update_correspondences only stores in-range indices,
        // but guard the upper bound too so a stray value can never reach the
        // target_->at()/target_covs_[]/gradient_valid_[] accesses below (the
        // FINDINGS_2026-06-26 crash; attributed to external memory corruption,
        // see VERIFICATION_2026-06-27). Count trips as telemetry.
        if(target_index < 0 || target_index >= static_cast<int>(target_->size())) {
            if (target_index >= 0) { oob_corr += 1; }   // -1 is normal; >= size is corruption
            continue;
        }

        const auto& source_pt = input_->at(i);
        Eigen::Vector4f source_homogeneous = trans * source_pt.getVector4fMap();
        Eigen::Vector3f transformed_source = source_homogeneous.head<3>();
        
        const auto& target_pt = target_->at(target_index);
        Eigen::Vector3f target_pos = target_pt.getVector3fMap();
        
        // Geometric term
        Eigen::Vector3f residual = transformed_source - target_pos;
        Eigen::Matrix<float, 3, 6> J_geometric;
        J_geometric.block<3, 3>(0, 0) = -skew(transformed_source);
        J_geometric.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();
        
        Eigen::Matrix3f M = mahalanobis_[i].block<3, 3>(0, 0);
        // GenZ-ICP blend (Lee et al., RA-L 2025): on an ill-conditioned scan
        // (current_genz_alpha_ < 1) convex-mix an isotropic point-to-point metric
        // into the plane metric to regularize the unconstrained axis, reusing the
        // same residual + Jacobian. alpha == 1 (default / healthy) -> pure
        // point-to-plane, bit-identical. genz_point_weight_ [1/m^2] sets the
        // point-to-point scale (~ the plane metric's typical eigenvalue).
        if (this->genz_enabled_ && this->current_genz_alpha_ < 1.0f) {
            const float a = this->current_genz_alpha_;
            M = a * M + (1.0f - a) * this->genz_point_weight_ * Eigen::Matrix3f::Identity();
        }
        // Saliency weighting (anti-dilution): up-weight rare salient source points
        // (edges/ribs/corners that constrain the along-axis DOF) so the abundant
        // planar walls don't swamp the weak axis. Off / boost<=1 / missing saliency
        // -> unit weight, bit-identical. Scales this point's whole contribution
        // (H, b, cost). See nano_gicp/saliency_weight.h, EXPLORATION_2026-06-26 #1.
        if (this->saliency_enabled_ && i < static_cast<int>(this->source_saliency_.size())) {
            M *= saliencyMultiplier(this->source_saliency_[i], this->saliency_boost_);
        }

        H_private[thread_num] += (J_geometric.transpose() * M * J_geometric).cast<double>();
        b_private[thread_num] += (J_geometric.transpose() * M * residual).cast<double>();
        cost_sum += residual.transpose() * M * residual;
        
        // Photometric term (channel normalized by photometric_scale_ to match
        // the units the target gradients were estimated in)
        if (use_photometric && (*gradient_valid_)[target_index]) {
            float src_val = photometricValue(source_pt, photometric_use_reflectivity_);
            float tgt_val = photometricValue(target_pt, photometric_use_reflectivity_);
            float intensity_diff = (src_val - tgt_val) / photometric_scale_;
            if (!std::isfinite(intensity_diff)) { continue; }
            Eigen::Vector3f gradient = (*target_intensity_gradients_)[target_index];
            
            if (gradient.norm() > kGradientMagMin && gradient.norm() < kGradientMagMax) {
                // Evaluate the same local target intensity model differentiated
                // below. The nearest point's value alone is piecewise constant
                // in pose, so it cannot validate a gradient-driven sub-voxel step.
                intensity_diff -= gradient.dot(residual);
                // Residual is (I_src - I_tgt); its Jacobian w.r.t. the (left-perturbation)
                // pose is d/dθ (I_src - I_tgt(x)) = -gᵀ·dx/dθ = [ gᵀ·skew(x) | -gᵀ ].
                // (Must match the geometric term's convention or the photometric step
                //  ascends intensity error instead of descending it.)
                Eigen::Matrix<float, 1, 6> J_photometric;
                J_photometric.block<1, 3>(0, 0) = gradient.transpose() * skew(transformed_source);
                J_photometric.block<1, 3>(0, 3) = -gradient.transpose();

                // Robustification of the photometric outliers (specular returns,
                // wet patches, exposure-like artifacts). Default: IRLS Huber
                // down-weighting beyond photometric_huber_delta_. Adaptive kernel
                // (Barron/Chebrolu): redescending Barron weight with the shape
                // alpha fitted to the residual distribution (current_alpha_,
                // lagged one iteration); residuals are collected for the fit.
                float weight = photometric_weight_;
                // Huber multiplier: the default robustifier, and the FLOOR for the
                // adaptive kernel (so enabling the kernel can only ADD robustness).
                const float abs_r = std::abs(intensity_diff);
                float huber_w = 1.f;
                if (photometric_huber_delta_ > 0.f && abs_r > photometric_huber_delta_) {
                    huber_w = photometric_huber_delta_ / abs_r;
                }
                if (adaptive_kernel) {
                    // Barron weight floored by the Huber: at alpha=2 (Barron=1.0)
                    // this falls back to exactly the Huber, never strips it. Fixes
                    // the 2026-06-18 finding where the kernel replaced the Huber and
                    // at alpha=2 removed all outlier protection (-> divergence).
                    const float barron_w = static_cast<float>(
                        barronRelWeight(intensity_diff, this->current_alpha_, kernel_c));
                    weight *= std::min(barron_w, huber_w);
                    photo_resid_private[thread_num].push_back(intensity_diff);
                } else {
                    weight *= huber_w;
                }
                ++photo_count_private[thread_num];
                photo_sq_sum += static_cast<double>(intensity_diff) * intensity_diff;  // telemetry only
                if (separate_photo) {
                    // Separate accumulator: scaled to the reference count post-loop.
                    Hp_private[thread_num] += (weight * J_photometric.transpose() * J_photometric).cast<double>();
                    bp_private[thread_num] += (weight * J_photometric.transpose() * intensity_diff).cast<double>();
                    photo_cost_sum += adaptive_kernel ? weight * intensity_diff * intensity_diff
                        : photometric_weight_ * huberCost(intensity_diff, photometric_huber_delta_);
                } else {
                    // Shared accumulation when neither normalization nor a
                    // separate geometric Hessian is requested.
                    H_private[thread_num] += (weight * J_photometric.transpose() * J_photometric).cast<double>();
                    b_private[thread_num] += (weight * J_photometric.transpose() * intensity_diff).cast<double>();
                    cost_sum += adaptive_kernel ? weight * intensity_diff * intensity_diff
                        : photometric_weight_ * huberCost(intensity_diff, photometric_huber_delta_);
                }
            }
        }
    }
    
    for(int i = 0; i < num_threads_; ++i) {
        (*H) += H_private[i];
        (*b) += b_private[i];
    }
    if (cost != nullptr) { *cost = cost_sum; }
    if (H_geo != nullptr) { *H_geo = *H; }

    // Photometric residual count + RMS (telemetry, tallied regardless of
    // normalization; RMS is the UNweighted brightness-constancy fit quality).
    long photo_count = 0;
    for (int i = 0; i < num_threads_; ++i) { photo_count += photo_count_private[i]; }
    this->last_photometric_count_ = static_cast<int>(photo_count);
    this->last_photometric_rms_ = (photo_count > 0)
        ? std::sqrt(static_cast<float>(photo_sq_sum / photo_count)) : 0.0f;

    // Adaptive kernel: from this iteration's photometric residuals, update the
    // scale c (data-driven MAD when kernel_scale_<=0, else the fixed scale) and
    // fit Barron's alpha (NLL minimization); both are used next iteration (lagged)
    // and converge over the LM loop. The MAD scale fixes the 2026-06-18 finding
    // that a fixed c larger than the residual bulk pinned alpha at L2 (no effect).
    if (adaptive_kernel) {
        std::vector<float> resid;
        for (auto& v : photo_resid_private) { resid.insert(resid.end(), v.begin(), v.end()); }
        const float fit_c = (this->kernel_scale_ > 0.f) ? this->kernel_scale_
                                                        : static_cast<float>(barronScaleMad(resid));
        this->current_kernel_c_ = fit_c;
        this->last_fit_alpha_ = static_cast<float>(fitBarronAlpha(
            resid, fit_c, this->kernel_alpha_lo_, this->kernel_alpha_hi_));
        this->current_alpha_ = this->last_fit_alpha_;
    }

    // Merge appearance after the geometric snapshot, optionally scaled to the
    // nominal reference count.
    if (separate_photo) {
        Eigen::Matrix<double, 6, 6> Hp_sum = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> bp_sum = Eigen::Matrix<double, 6, 1>::Zero();
        for (int i = 0; i < num_threads_; ++i) { Hp_sum += Hp_private[i]; bp_sum += bp_private[i]; }
        const double norm = normalize_photo
            ? refCountScale(static_cast<double>(this->photometric_ref_count_), photo_count) : 1.0;
        (*H) += Hp_sum * norm;
        (*b) += bp_sum * norm;
        if (cost != nullptr) { *cost += photo_cost_sum * norm; }
    }

    // Accumulate the corruption telemetry across LM iterations; reset per align()
    // in computeTransformation, surfaced via lastOobCorrespondences().
    this->last_oob_corr_count_ += oob_corr;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::accumulateVisualResidual(
    const Eigen::Isometry3f& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* cost) {

    this->last_visual_count_ = 0;
    this->last_visual_rms_ = 0.0f;
    this->last_visual_rej_behind_ = 0;   // diagnostics: per-gate reject reasons
    this->last_visual_rej_oob_ = 0;
    this->last_visual_rej_grad_ = 0;

    if (!visual_enabled_ || visual_weight_ <= 0.f) { return; }
    if (visual_cur_.empty() || visual_prev_.empty() || !input_) { return; }
    // Single-channel float images are required (normalized brightness).
    if (visual_cur_.type() != CV_32FC1 || visual_prev_.type() != CV_32FC1) { return; }

    const float fx = visual_fx_, fy = visual_fy_, cx = visual_cx_, cy = visual_cy_;
    if (fx <= 0.f || fy <= 0.f) { return; }

    // world -> camera for the previous (warp-target) and current (reference) frames
    const Eigen::Matrix3f R_prev = T_cw_prev_.linear();
    const Eigen::Vector3f t_prev = T_cw_prev_.translation();
    const Eigen::Matrix3f R_cur  = T_cw_cur_.linear();
    const Eigen::Vector3f t_cur  = T_cw_cur_.translation();

    const float zmin = 1e-3f;
    const float bw = 2.f;  // border (central-difference needs a 1px ring inside bilinear's)
    const float prev_umax = static_cast<float>(visual_prev_.cols) - 1.f - bw;
    const float prev_vmax = static_cast<float>(visual_prev_.rows) - 1.f - bw;
    const float cur_umax  = static_cast<float>(visual_cur_.cols)  - 1.f - bw;
    const float cur_vmax  = static_cast<float>(visual_cur_.rows)  - 1.f - bw;

    std::vector<Eigen::Matrix<double, 6, 6>> H_private(num_threads_, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_private(num_threads_, Eigen::Matrix<double, 6, 1>::Zero());
    double cost_sum = 0.0;
    double sq_sum = 0.0;
    long count = 0;
    long rej_behind = 0, rej_oob = 0, rej_grad = 0;   // diagnostics only

    // Source cloud: the dense deskewed scan if provided (strided to bound cost),
    // else the voxelised registration cloud (input_) at stride 1 (bit-identical).
    // Both are the same deskewed WORLD frame, so the projection math is identical.
    const bool use_dense = (this->visual_src_ && !this->visual_src_->empty());
    // Hold a shared_ptr to the source cloud for the WHOLE parallel loop, not just
    // a reference: binding `const auto& src = *input_` keeps no ownership, so if
    // another thread drops the last external ref to input_/visual_src_ mid-loop
    // (next-scan setInputSource / setVisualSource), the cloud is freed and `src`
    // dangles -> garbage size() -> the load-triggered std::out_of_range at the
    // src.at(i) below (FINDINGS_2026-06-25, 1/48 reps under contention). Owning a
    // local ref keeps it alive until this function returns.
    const PointCloudSourceConstPtr src_ptr = use_dense ? this->visual_src_ : input_;
    if (!src_ptr) { return; }
    const auto& src = *src_ptr;
    const int n_src = static_cast<int>(src.size());
    const int stride = (use_dense && this->visual_src_max_ > 0 && n_src > this->visual_src_max_)
                           ? (n_src / this->visual_src_max_) : 1;

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) \
        reduction(+:cost_sum,sq_sum,count,rej_behind,rej_oob,rej_grad)
    for (int i = 0; i < n_src; i += stride) {
        const auto& sp = src.at(i);
        const Eigen::Vector3f p_w(sp.x, sp.y, sp.z);     // raw world point (prior pose)
        const Eigen::Vector3f x = trans * p_w;           // pose-corrected world point

        // Reference brightness: project the RAW world point into the CURRENT
        // image. This is pose-independent (the point's position relative to the
        // current camera is fixed by the rigid extrinsic), hence a fixed target.
        const Eigen::Vector3f Pc_ref = R_cur * p_w + t_cur;
        if (Pc_ref.z() <= zmin) { ++rej_behind; continue; }
        const float u_ref = fx * Pc_ref.x() / Pc_ref.z() + cx;
        const float v_ref = fy * Pc_ref.y() / Pc_ref.z() + cy;
        // Positive-form bounds test so a non-finite projection is REJECTED: NaN
        // fails every comparison, so the old `u<bw || u>umax` let NaN through to
        // bilinearSample's unchecked pointer read. Identical to the old test for
        // finite (u, v). (FINDINGS_2026-06-25 hardening.)
        if (!(u_ref >= bw && u_ref <= cur_umax && v_ref >= bw && v_ref <= cur_vmax)) { ++rej_oob; continue; }
        const float I_ref = bilinearSample(visual_cur_, u_ref, v_ref);

        // Moving brightness: project the CORRECTED world point into the PREVIOUS
        // image. This depends on `trans` -- the source of pose observability.
        const Eigen::Vector3f Pc = R_prev * x + t_prev;
        if (Pc.z() <= zmin) { ++rej_behind; continue; }
        const float invz = 1.f / Pc.z();
        const float u = fx * Pc.x() * invz + cx;
        const float v = fy * Pc.y() * invz + cy;
        if (!(u >= bw && u <= prev_umax && v >= bw && v <= prev_vmax)) { ++rej_oob; continue; }  // NaN-safe (see above)
        const float I_mov = bilinearSample(visual_prev_, u, v);

        // Image gradient (central difference) on the previous image, normalized.
        const float gu = 0.5f * (bilinearSample(visual_prev_, u + 1.f, v)
                               - bilinearSample(visual_prev_, u - 1.f, v));
        const float gv = 0.5f * (bilinearSample(visual_prev_, u, v + 1.f)
                               - bilinearSample(visual_prev_, u, v - 1.f));
        if (std::abs(gu) < 1e-6f && std::abs(gv) < 1e-6f) { ++rej_grad; continue; }

        const float r = I_mov - I_ref;

        // dπ/dPc (2x3)
        Eigen::Matrix<float, 2, 3> dpi;
        dpi << fx * invz, 0.f,      -fx * Pc.x() * invz * invz,
               0.f,       fy * invz, -fy * Pc.y() * invz * invz;
        Eigen::Matrix<float, 1, 2> gI;
        gI << gu, gv;
        // G = grad_I · dπ/dPc · R_prev (1x3). Residual r = I_mov(x) - I_ref;
        // since x = trans·p_w perturbs on the LEFT (dx' = [-skew(x)|I]·ξ), the
        // Jacobian is [ -G·skew(x) | G ]. NOTE the translation block is +G,
        // OPPOSITE the LiDAR photometric term's -gᵀ -- see test_visual_residual.
        const Eigen::Matrix<float, 1, 3> G = gI * dpi * R_prev;

        Eigen::Matrix<float, 1, 6> J;
        J.block<1, 3>(0, 0) = -G * skew(x);
        J.block<1, 3>(0, 3) = G;

        float weight = visual_weight_;
        const float abs_r = std::abs(r);
        if (visual_huber_delta_ > 0.f && abs_r > visual_huber_delta_) {
            weight *= visual_huber_delta_ / abs_r;
        }

        const int tn = omp_get_thread_num();
        H_private[tn] += (weight * J.transpose() * J).cast<double>();
        b_private[tn] += (weight * J.transpose() * r).cast<double>();
        cost_sum += visual_weight_ * huberCost(r, visual_huber_delta_);
        sq_sum += static_cast<double>(r) * r;
        count += 1;
    }

    // Opt-in mass-normalization (see refCountScale): off (default) keeps the
    // exact per-thread reduction -> bit-identical; on scales the term's total
    // mass to visual_ref_count_ so visual_weight_ is count-independent.
    if (this->visual_ref_count_ > 0.f) {
        Eigen::Matrix<double, 6, 6> H_sum = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b_sum = Eigen::Matrix<double, 6, 1>::Zero();
        for (int t = 0; t < num_threads_; ++t) { H_sum += H_private[t]; b_sum += b_private[t]; }
        const double norm = refCountScale(static_cast<double>(this->visual_ref_count_), count);
        (*H) += H_sum * norm;
        (*b) += b_sum * norm;
        if (cost != nullptr) { *cost += cost_sum * norm; }
    } else {
        for (int t = 0; t < num_threads_; ++t) {
            (*H) += H_private[t];
            (*b) += b_private[t];
        }
        if (cost != nullptr) { *cost += cost_sum; }
    }
    this->last_visual_count_ = static_cast<int>(count);
    this->last_visual_rms_ = (count > 0) ? std::sqrt(static_cast<float>(sq_sum / count)) : 0.0f;
    this->last_visual_rej_behind_ = static_cast<int>(rej_behind);
    this->last_visual_rej_oob_ = static_cast<int>(rej_oob);
    this->last_visual_rej_grad_ = static_cast<int>(rej_grad);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::accumulateVisualMapResidual(
    const Eigen::Isometry3f& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* cost) {

    this->last_visual_map_count_ = 0;
    this->last_visual_map_rms_ = 0.0f;

    if (visual_map_weight_ <= 0.f) { return; }
    if (visual_cur_.empty() || visual_cur_.type() != CV_32FC1) { return; }
    if (!target_ || !target_visual_refs_ || target_visual_refs_->size() != target_->size()) { return; }

    const float fx = visual_fx_, fy = visual_fy_, cx = visual_cx_, cy = visual_cy_;
    if (fx <= 0.f || fy <= 0.f) { return; }

    // world -> current camera INCLUDING the correction: T_cw_cur_ = (T_prior*T_bc)^-1
    // (set from the prior pose by the caller), composed with trans^-1 so the
    // FIXED map point sees the corrected camera. This is the source of pose
    // observability (the f2f term moved the point instead).
    const Eigen::Isometry3f T_cw = T_cw_cur_ * trans.inverse();
    const Eigen::Matrix3f R_cw = T_cw.linear();

    const float zmin = 1e-3f;
    const float bw = 2.f;
    const float umax = static_cast<float>(visual_cur_.cols) - 1.f - bw;
    const float vmax = static_cast<float>(visual_cur_.rows) - 1.f - bw;
    const float cos_view_max = std::cos(visual_map_view_angle_max_);

    std::vector<Eigen::Matrix<double, 6, 6>> H_private(num_threads_, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_private(num_threads_, Eigen::Matrix<double, 6, 1>::Zero());
    double cost_sum = 0.0, sq_sum = 0.0;
    long count = 0;

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:cost_sum,sq_sum,count)
    for (int j = 0; j < target_->size(); ++j) {
        const VisualRef& vr = (*target_visual_refs_)[j];
        if (!vr.valid) { continue; }

        const auto& tp = target_->at(j);
        const Eigen::Vector3f p_w(tp.x, tp.y, tp.z);   // FIXED map point (world)
        const Eigen::Vector3f Pc = R_cw * p_w + T_cw.translation();
        if (Pc.z() <= zmin) { continue; }

        // Viewpoint gating: brightness constancy breaks down when the current
        // viewing ray differs much from the keyframe ray the reference was
        // sampled along.
        const float vn = vr.p_kf_cam.norm();
        if (vn > 1e-6f) {
            const float cosang = (Pc.normalized()).dot(vr.p_kf_cam / vn);
            if (cosang < cos_view_max) { continue; }
        }

        const float invz = 1.f / Pc.z();
        const float u = fx * Pc.x() * invz + cx;
        const float v = fy * Pc.y() * invz + cy;
        if (!(u >= bw && u <= umax && v >= bw && v <= vmax)) { continue; }  // NaN-safe bounds

        const float I_mov = bilinearSample(visual_cur_, u, v);
        const float gu = 0.5f * (bilinearSample(visual_cur_, u + 1.f, v) - bilinearSample(visual_cur_, u - 1.f, v));
        const float gv = 0.5f * (bilinearSample(visual_cur_, u, v + 1.f) - bilinearSample(visual_cur_, u, v - 1.f));
        if (std::abs(gu) < 1e-6f && std::abs(gv) < 1e-6f) { continue; }

        const float r = I_mov - vr.ref;

        Eigen::Matrix<float, 2, 3> dpi;
        dpi << fx * invz, 0.f,      -fx * Pc.x() * invz * invz,
               0.f,       fy * invz, -fy * Pc.y() * invz * invz;
        Eigen::Matrix<float, 1, 2> gI;
        gI << gu, gv;
        // FIXED map point under the trans-INVERSE perturbation:
        // d(trans^-1 p_w)/dxi = [ skew(p_w) | -I ], so J = G*[skew(p_w)|-I]
        // = [ +G*skew(p_w) | -G ]. NOTE: OPPOSITE the frame-to-frame term's
        // [ -G*skew(x) | +G ] -- guarded by test_visual_residual.
        const Eigen::Matrix<float, 1, 3> G = gI * dpi * R_cw;
        Eigen::Matrix<float, 1, 6> J;
        J.block<1, 3>(0, 0) = G * skew(p_w);
        J.block<1, 3>(0, 3) = -G;

        float weight = visual_map_weight_;
        const float abs_r = std::abs(r);
        if (visual_huber_delta_ > 0.f && abs_r > visual_huber_delta_) {
            weight *= visual_huber_delta_ / abs_r;
        }

        const int tn = omp_get_thread_num();
        H_private[tn] += (weight * J.transpose() * J).cast<double>();
        b_private[tn] += (weight * J.transpose() * r).cast<double>();
        cost_sum += visual_map_weight_ * huberCost(r, visual_huber_delta_);
        sq_sum += static_cast<double>(r) * r;
        count += 1;
    }

    // Opt-in mass-normalization (see refCountScale): off (default) keeps the
    // exact per-thread reduction -> bit-identical; on scales the term's total
    // mass to visual_map_ref_count_ so visual_map_weight_ is count-independent.
    if (this->visual_map_ref_count_ > 0.f) {
        Eigen::Matrix<double, 6, 6> H_sum = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b_sum = Eigen::Matrix<double, 6, 1>::Zero();
        for (int t = 0; t < num_threads_; ++t) { H_sum += H_private[t]; b_sum += b_private[t]; }
        const double norm = refCountScale(static_cast<double>(this->visual_map_ref_count_), count);
        (*H) += H_sum * norm;
        (*b) += b_sum * norm;
        if (cost != nullptr) { *cost += cost_sum * norm; }
    } else {
        for (int t = 0; t < num_threads_; ++t) {
            (*H) += H_private[t];
            (*b) += b_private[t];
        }
        if (cost != nullptr) { *cost += cost_sum; }
    }
    this->last_visual_map_count_ = static_cast<int>(count);
    this->last_visual_map_rms_ = (count > 0) ? std::sqrt(static_cast<float>(sq_sum / count)) : 0.0f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::accumulateLidarMapResidual(
    const Eigen::Isometry3f& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* cost) {

    this->last_lidar_map_count_ = 0;
    this->last_lidar_map_rms_ = 0.0f;

    if (lidar_map_weight_ <= 0.f) { return; }
    if (lidar_image_.empty() || lidar_image_.type() != CV_32FC1 || !target_) { return; }
    if (std::abs(lidar_az_a_) < 1e-12f || std::abs(lidar_el_a_) < 1e-12f) { return; }

    const float inv_scale = 1.f / lidar_image_scale_;  // COIN-LIO image scale (per-channel)
    // Keyframe-image references (INTENSITY_AUDIT_2026-07-09): when the caller
    // supplied per-target-point brightness sampled from each keyframe's FULL-RES
    // image (index-aligned, already /scale), prefer it over the voxel-AVERAGED
    // .reflectivity field, which blurs away the graffiti-scale texture the
    // residual keys on. Snapshot the shared_ptr so a concurrent target swap
    // can't free the list under the loop.
    const std::shared_ptr<const std::vector<float>> kf_refs = this->target_lidar_refs_;
    const bool use_kf_refs =
        (kf_refs != nullptr && kf_refs->size() == static_cast<size_t>(target_->size()));
    // world -> current lidar INCLUDING the correction (same trans-inverse form as
    // the camera map term, with the lidar frame instead of the camera).
    const Eigen::Isometry3f T_lw = T_lw_cur_ * trans.inverse();
    const Eigen::Matrix3f R_lw = T_lw.linear();

    const float bw = 2.f;
    const float umax = static_cast<float>(lidar_image_.cols) - 1.f - bw;
    const float vmax = static_cast<float>(lidar_image_.rows) - 1.f - bw;
    const float inv_az_a = 1.f / lidar_az_a_;
    const float inv_el_a = 1.f / lidar_el_a_;
    const bool use_el_lut = (static_cast<int>(lidar_el_lut_.size()) == lidar_image_.rows);
    // Occlusion check active only when a same-size range image was supplied.
    const bool use_range = (!lidar_range_img_.empty()
                            && lidar_range_img_.type() == CV_32FC1
                            && lidar_range_img_.rows == lidar_image_.rows
                            && lidar_range_img_.cols == lidar_image_.cols);

    std::vector<Eigen::Matrix<double, 6, 6>> H_private(num_threads_, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_private(num_threads_, Eigen::Matrix<double, 6, 1>::Zero());
    double cost_sum = 0.0, sq_sum = 0.0;
    long count = 0;

    // Cap the iterated map points: the submap can be tens of thousands of points
    // and this runs every LM iteration; striding bounds the per-scan cost so the
    // node keeps real time (the term is count-normalized, so a subset is fine).
    constexpr int kLidarMaxPoints = 4000;
    const int n_target = static_cast<int>(target_->size());
    const int stride = std::max(1, (n_target + kLidarMaxPoints - 1) / kLidarMaxPoints);

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:cost_sum,sq_sum,count)
    for (int j = 0; j < n_target; j += stride) {
        const auto& tp = target_->at(j);
        const Eigen::Vector3f p_w(tp.x, tp.y, tp.z);            // FIXED map point (world)
        // Reference brightness: keyframe-image sample when available (< 0 =
        // invalid, e.g. outside that keyframe's image), else the point's own
        // voxel-averaged field.
        float ref;
        if (use_kf_refs) {
            ref = (*kf_refs)[j];
            if (!(ref >= 0.f)) { continue; }   // NaN-safe invalid check
        } else {
            ref = (lidar_image_use_dedicated_channel_ ? tp.lidar_intensity : tp.reflectivity) * inv_scale;
        }
        if (!std::isfinite(ref)) { continue; }
        const Eigen::Vector3f Pl = R_lw * p_w + T_lw.translation();

        const float X = Pl.x(), Y = Pl.y(), Z = Pl.z();
        const float rxy2 = X * X + Y * Y;
        if (rxy2 < 1e-6f) { continue; }
        const float rxy = std::sqrt(rxy2);
        const float rr2 = rxy2 + Z * Z;

        const float az = std::atan2(Y, X);
        const float el = std::atan2(Z, rxy);
        const float u = columnFromAzimuth(az, lidar_az_a_, lidar_az_b_, lidar_image_.cols);
        // Row from the per-row elevation LUT (non-uniform beams) or the linear
        // fallback; inv_el_eff is the local rows-per-radian used by the Jacobian.
        float v, inv_el_eff;
        if (use_el_lut) {
            float slope;
            if (!rowFromElevationLut(lidar_el_lut_, el, v, slope)) { continue; }
            inv_el_eff = 1.f / slope;
        } else {
            v = (el - lidar_el_b_) * inv_el_a;
            inv_el_eff = inv_el_a;
        }
        if (!(u >= bw && u <= umax && v >= bw && v <= vmax)) { continue; }  // NaN-safe; drops the azimuth seam strip

        // Occlusion / wrong-surface rejection (FAST-LIVO-style depth-consistency
        // cull, Zheng et al. IROS 2022): the FIXED map point must be the surface
        // actually visible at this pixel. Without this, the whole-corridor
        // submap projects far-side / occluded points onto near walls and floods
        // the residual with mismatches (the dominant cause of high frame-to-map RMS).
        if (use_range) {
            if (!validRangeSupport(lidar_range_img_, u, v)) { continue; }
            const int ui = static_cast<int>(std::lround(u));
            const int vi = static_cast<int>(std::lround(v));
            const float ri = lidar_range_img_.ptr<float>(vi)[ui];
            if (ri <= 0.f) { continue; }                       // no return: can't verify
            const float rng_p = std::sqrt(rr2);                // map point range from current sensor
            const float tol = std::max(lidar_range_abs_tol_, lidar_range_rel_tol_ * ri);
            if (std::abs(rng_p - ri) > tol) { continue; }      // occluded / different surface
        }

        const float I_mov = bilinearSample(lidar_image_, u, v);
        const float gu = 0.5f * (bilinearSample(lidar_image_, u + 1.f, v) - bilinearSample(lidar_image_, u - 1.f, v));
        const float gv = 0.5f * (bilinearSample(lidar_image_, u, v + 1.f) - bilinearSample(lidar_image_, u, v - 1.f));
        if (std::abs(gu) < 1e-6f && std::abs(gv) < 1e-6f) { continue; }

        const float r = I_mov - ref;

        // Spherical projection Jacobian dpi_L/dP_l (2x3):
        //   col: d/dP (az)/az_a,  az=atan2(Y,X) -> daz/dP = (-Y/rxy2, X/rxy2, 0)
        //   row: d/dP (el)*drow/del, el=atan2(Z,rxy) -> del/dP = (-ZX/(rxy*rr2), -ZY/(rxy*rr2), rxy/rr2)
        //   (drow/del = inv_el_eff: local LUT slope, or the linear 1/el_a)
        Eigen::Matrix<float, 2, 3> dpi;
        dpi(0, 0) = inv_az_a * (-Y / rxy2);
        dpi(0, 1) = inv_az_a * ( X / rxy2);
        dpi(0, 2) = 0.f;
        dpi(1, 0) = inv_el_eff * (-Z * X / (rxy * rr2));
        dpi(1, 1) = inv_el_eff * (-Z * Y / (rxy * rr2));
        dpi(1, 2) = inv_el_eff * ( rxy / rr2);
        Eigen::Matrix<float, 1, 2> gI;
        gI << gu, gv;
        // Same trans-inverse perturbation as the camera map term:
        // J = G_L * [skew(p_w) | -I] = [ +G_L*skew(p_w) | -G_L ].
        const Eigen::Matrix<float, 1, 3> G = gI * dpi * R_lw;
        Eigen::Matrix<float, 1, 6> J;
        J.block<1, 3>(0, 0) = G * skew(p_w);
        J.block<1, 3>(0, 3) = -G;

        float weight = lidar_map_weight_;
        const float abs_r = std::abs(r);
        if (photometric_huber_delta_ > 0.f && abs_r > photometric_huber_delta_) {
            weight *= photometric_huber_delta_ / abs_r;
        }

        const int tn = omp_get_thread_num();
        H_private[tn] += (weight * J.transpose() * J).cast<double>();
        b_private[tn] += (weight * J.transpose() * r).cast<double>();
        cost_sum += lidar_map_weight_ * huberCost(r, photometric_huber_delta_);
        sq_sum += static_cast<double>(r) * r;
        count += 1;
    }

    // COUNT-NORMALIZE: the LiDAR image contributes 5k-75k points per scan, so a
    // raw weight scales the Hessian mass with the (huge, variable) point count
    // and is impossible to tune (w=0.005 already over-travels). Normalize the
    // term's total mass to a nominal reference count so `weight` is comparable
    // to the camera/geometric terms and stable run-to-run regardless of how many
    // points happen to be visible.
    Eigen::Matrix<double, 6, 6> H_sum = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b_sum = Eigen::Matrix<double, 6, 1>::Zero();
    for (int t = 0; t < num_threads_; ++t) {
        H_sum += H_private[t];
        b_sum += b_private[t];
    }
    constexpr double kLidarRefCount = 1000.0;  // always-on for this term (validated)
    const double norm = refCountScale(kLidarRefCount, count);
    (*H) += H_sum * norm;
    (*b) += b_sum * norm;
    if (cost != nullptr) { *cost += cost_sum * norm; }
    this->last_lidar_map_count_ = static_cast<int>(count);
    this->last_lidar_map_rms_ = (count > 0) ? std::sqrt(static_cast<float>(sq_sum / count)) : 0.0f;
}

// Frame-to-FRAME LiDAR range/intensity flow term (doc/EXPLORATION_2026-06-26.md
// #2): register the current scan against the PREVIOUS scan's image to observe the
// along-tunnel motion the frame-to-map reflectivity term can't (aliased). Iterates
// the current SOURCE points, moves each by the correction (x = trans * p_w), and
// projects it into the previous lidar image via the spherical model; residual =
// prev_image(projection) - the point's own brightness. Jacobian is the visual
// frame-to-frame LEFT-perturbation form [-G·skew(x) | G] (G = grad_I·dpi·R_lw_prev)
// with the spherical dpi of the map term -- correct by construction from those two
// in-tree terms. The caller direction-separates the result so it only constrains
// the degenerate axis. weight <= 0 (default) -> no-op / bit-identical.
template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::accumulateLidarFlowResidual(
    const Eigen::Isometry3f& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* cost) {

    this->last_lidar_flow_count_ = 0;
    this->last_lidar_flow_rms_ = 0.0f;

    if (lidar_flow_weight_ <= 0.f) { return; }
    // Snapshot the previous-frame image (cv::Mat refcount hold) so the parallel
    // loop owns a stable buffer for its whole lifetime even if another thread
    // reassigns the member mid-loop -- the same ownership guard as the
    // photometric-loop fix. The member is set via clone(), never mutated in place.
    const cv::Mat prev = this->lidar_flow_prev_img_;
    if (prev.empty() || prev.type() != CV_32FC1 || !input_) { return; }
    if (std::abs(lidar_az_a_) < 1e-12f || std::abs(lidar_el_a_) < 1e-12f) { return; }
    // Image-to-image mode (INTENSITY_AUDIT_2026-07-09): the reference brightness
    // is sampled from the CURRENT full-res image at the RAW point's projection
    // (pose-independent, mirroring the visual f2f term) instead of the point's
    // voxel-averaged .reflectivity field -- the voxel blur destroyed the
    // graffiti-scale texture this term needs. Snapshot the current image with
    // the same refcount-hold guard as `prev`.
    const bool image_ref = this->lidar_flow_image_ref_;
    const cv::Mat cur = this->lidar_image_;
    if (image_ref && (cur.empty() || cur.type() != CV_32FC1)) { return; }
    const cv::Mat prev_range = this->lidar_flow_prev_range_;
    const cv::Mat cur_range = this->lidar_range_img_;
    const bool use_prev_range = !prev_range.empty();
    const bool use_cur_range = image_ref && !cur_range.empty();
    if (use_prev_range && (prev_range.type() != CV_32FC1 || prev_range.size() != prev.size())) { return; }
    if (use_cur_range && (cur_range.type() != CV_32FC1 || cur_range.size() != cur.size())) { return; }
    const Eigen::Matrix3f R_lw_cur = this->T_lw_cur_.linear();
    const Eigen::Vector3f t_lw_cur = this->T_lw_cur_.translation();

    const float inv_scale = 1.f / lidar_image_scale_;
    // world -> PREVIOUS lidar (the previous scan's corrected pose, fixed here).
    const Eigen::Matrix3f R_lw_prev = this->T_lw_prev_flow_.linear();
    const Eigen::Vector3f t_lw_prev = this->T_lw_prev_flow_.translation();

    // Zero-mean patch residuals (half-width P; 0 = single pixel): use the
    // graffiti's spatial STRUCTURE and cancel brightness offsets (mini-DSO).
    const int P = this->lidar_flow_patch_;
    const int n_pix = (2 * P + 1) * (2 * P + 1);
    const float bw = 2.f + static_cast<float>(P);
    const float umax = static_cast<float>(prev.cols) - 1.f - bw;
    const float vmax = static_cast<float>(prev.rows) - 1.f - bw;
    const float cur_umax = image_ref ? static_cast<float>(cur.cols) - 1.f - bw : 0.f;
    const float cur_vmax = image_ref ? static_cast<float>(cur.rows) - 1.f - bw : 0.f;
    const float inv_az_a = 1.f / lidar_az_a_;
    const float inv_el_a = 1.f / lidar_el_a_;
    const bool use_el_lut = (static_cast<int>(lidar_el_lut_.size()) == prev.rows);

    std::vector<Eigen::Matrix<double, 6, 6>> H_private(num_threads_, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_private(num_threads_, Eigen::Matrix<double, 6, 1>::Zero());
    double cost_sum = 0.0, sq_sum = 0.0;
    long count = 0;

    constexpr int kFlowMaxPoints = 4000;                 // stride the source to bound per-iter cost
    const int n_src = static_cast<int>(input_->size());
    const int stride = std::max(1, (n_src + kFlowMaxPoints - 1) / kFlowMaxPoints);

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:cost_sum,sq_sum,count)
    for (int i = 0; i < n_src; i += stride) {
        const auto& sp = input_->at(i);
        const Eigen::Vector3f p_w(sp.x, sp.y, sp.z);         // current source point (world)
        const Eigen::Vector3f x = trans * p_w;               // pose-corrected world point
        const Eigen::Vector3f Pl = R_lw_prev * x + t_lw_prev;  // into the PREVIOUS lidar frame

        const float X = Pl.x(), Y = Pl.y(), Z = Pl.z();
        const float rxy2 = X * X + Y * Y;
        if (rxy2 < 1e-6f) { continue; }
        const float rxy = std::sqrt(rxy2);
        const float rr2 = rxy2 + Z * Z;

        const float az = std::atan2(Y, X);
        const float el = std::atan2(Z, rxy);
        const float u = columnFromAzimuth(az, lidar_az_a_, lidar_az_b_, prev.cols);
        float v, inv_el_eff;
        if (use_el_lut) {
            float slope;
            if (!rowFromElevationLut(lidar_el_lut_, el, v, slope)) { continue; }
            inv_el_eff = 1.f / slope;
        } else {
            v = (el - lidar_el_b_) * inv_el_a;
            inv_el_eff = inv_el_a;
        }
        if (!(u >= bw && u <= umax && v >= bw && v <= vmax)) { continue; }   // NaN-safe bounds
        if (use_prev_range) {
            if (!validRangeSupport(prev_range, u, v, P)) { continue; }
            const float depth = bilinearSample(prev_range, u, v);
            const float tol = std::max(lidar_range_abs_tol_, lidar_range_rel_tol_ * depth);
            if (std::abs(std::sqrt(rr2) - depth) > tol) { continue; }
        }

        // Reference: sample the CURRENT full-res image at the RAW point's
        // projection (pose-independent -- the point's position relative to the
        // current sensor is fixed by the prior), or fall back to the point's
        // (voxel-averaged) field when image_ref is off.
        float u_ref = 0.f, v_ref = 0.f;
        if (image_ref) {
            const Eigen::Vector3f Pc = R_lw_cur * p_w + t_lw_cur;
            const float cX = Pc.x(), cY = Pc.y(), cZ = Pc.z();
            const float crxy2 = cX * cX + cY * cY;
            if (crxy2 < 1e-6f) { continue; }
            const float caz = std::atan2(cY, cX);
            const float cel = std::atan2(cZ, std::sqrt(crxy2));
            u_ref = columnFromAzimuth(caz, lidar_az_a_, lidar_az_b_, cur.cols);
            if (use_el_lut) {
                float slope;
                if (!rowFromElevationLut(lidar_el_lut_, cel, v_ref, slope)) { continue; }
            } else {
                v_ref = (cel - lidar_el_b_) * inv_el_a;
            }
            if (!(u_ref >= bw && u_ref <= cur_umax && v_ref >= bw && v_ref <= cur_vmax)) { continue; }
            if (use_cur_range) {
                if (!validRangeSupport(cur_range, u_ref, v_ref, P)) { continue; }
                const float depth = bilinearSample(cur_range, u_ref, v_ref);
                const float tol = std::max(lidar_range_abs_tol_, lidar_range_rel_tol_ * depth);
                if (std::abs(Pc.norm() - depth) > tol) { continue; }
            }
        }

        const float gu = 0.5f * (bilinearSample(prev, u + 1.f, v) - bilinearSample(prev, u - 1.f, v));
        const float gv = 0.5f * (bilinearSample(prev, u, v + 1.f) - bilinearSample(prev, u, v - 1.f));
        if (std::abs(gu) < 1e-6f && std::abs(gv) < 1e-6f) { continue; }

        Eigen::Matrix<float, 2, 3> dpi;                      // spherical dpi/dP_l (as the map term)
        dpi(0, 0) = inv_az_a * (-Y / rxy2);
        dpi(0, 1) = inv_az_a * ( X / rxy2);
        dpi(0, 2) = 0.f;
        dpi(1, 0) = inv_el_eff * (-Z * X / (rxy * rr2));
        dpi(1, 1) = inv_el_eff * (-Z * Y / (rxy * rr2));
        dpi(1, 2) = inv_el_eff * ( rxy / rr2);
        // Patch residuals: r_k over the (2P+1)^2 window, zero-mean on both sides
        // for P > 0 (brightness-offset invariant); P = 0 reduces to the single
        // plain difference. Per-pixel Jacobian uses the pixel's own prev-image
        // gradient; contributions are divided by n_pix so a patch point carries
        // the same total mass as a single-pixel point (count-normalization).
        float mean_mov = 0.f, mean_ref = 0.f, mean_gu = 0.f, mean_gv = 0.f;
        float mov_px[49], ref_px[49], gu_px[49], gv_px[49];   // P <= 3 (enforced by setter)
        int k = 0;
        for (int dv = -P; dv <= P; ++dv) {
            for (int du = -P; du <= P; ++du, ++k) {
                const float uu = u + static_cast<float>(du), vv = v + static_cast<float>(dv);
                mov_px[k] = bilinearSample(prev, uu, vv);
                gu_px[k] = 0.5f * (bilinearSample(prev, uu + 1.f, vv) - bilinearSample(prev, uu - 1.f, vv));
                gv_px[k] = 0.5f * (bilinearSample(prev, uu, vv + 1.f) - bilinearSample(prev, uu, vv - 1.f));
                ref_px[k] = image_ref
                    ? bilinearSample(cur, u_ref + static_cast<float>(du), v_ref + static_cast<float>(dv))
                    : (lidar_image_use_dedicated_channel_ ? sp.lidar_intensity : sp.reflectivity) * inv_scale;
                mean_mov += mov_px[k];
                mean_ref += ref_px[k];
                mean_gu += gu_px[k];
                mean_gv += gv_px[k];
            }
        }
        mean_mov /= static_cast<float>(n_pix);
        mean_ref /= static_cast<float>(n_pix);
        mean_gu /= static_cast<float>(n_pix);
        mean_gv /= static_cast<float>(n_pix);

        const int tn = omp_get_thread_num();
        double pt_sq = 0.0;
        for (k = 0; k < n_pix; ++k) {
            // Zero-mean only for a real patch; a single pixel keeps the raw diff.
            const float r = (P > 0) ? ((mov_px[k] - mean_mov) - (ref_px[k] - mean_ref))
                                    : (mov_px[k] - ref_px[k]);
            Eigen::Matrix<float, 1, 2> gI;
            // The moving patch mean depends on pose too. Without its
            // derivative a uniform ramp falsely reports motion information
            // even though subtracting the patch mean removes that information.
            gI << gu_px[k] - (P > 0 ? mean_gu : 0.f),
                  gv_px[k] - (P > 0 ? mean_gv : 0.f);
            // x = trans·p_w perturbs on the LEFT -> J = [-G·skew(x) | G], same as
            // the visual frame-to-frame term; G = grad_I · dpi · R_lw_prev.
            const Eigen::Matrix<float, 1, 3> G = gI * dpi * R_lw_prev;
            Eigen::Matrix<float, 1, 6> J;
            J.block<1, 3>(0, 0) = -G * skew(x);
            J.block<1, 3>(0, 3) = G;

            float weight = lidar_flow_weight_ / static_cast<float>(n_pix);
            const float abs_r = std::abs(r);
            if (photometric_huber_delta_ > 0.f && abs_r > photometric_huber_delta_) {
                weight *= photometric_huber_delta_ / abs_r;
            }
            H_private[tn] += (weight * J.transpose() * J).cast<double>();
            b_private[tn] += (weight * J.transpose() * r).cast<double>();
            cost_sum += (lidar_flow_weight_ / n_pix) * huberCost(r, photometric_huber_delta_);
            pt_sq += static_cast<double>(r) * r;
        }
        sq_sum += pt_sq / static_cast<double>(n_pix);
        count += 1;
    }

    // Count-normalize like the frame-to-map term so `weight` is count-independent.
    Eigen::Matrix<double, 6, 6> H_sum = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b_sum = Eigen::Matrix<double, 6, 1>::Zero();
    for (int t = 0; t < num_threads_; ++t) { H_sum += H_private[t]; b_sum += b_private[t]; }
    constexpr double kFlowRefCount = 1000.0;
    const double norm = refCountScale(kFlowRefCount, count);
    (*H) += H_sum * norm;
    (*b) += b_sum * norm;
    if (cost != nullptr) { *cost += cost_sum * norm; }
    this->last_lidar_flow_count_ = static_cast<int>(count);
    this->last_lidar_flow_rms_ = (count > 0) ? std::sqrt(static_cast<float>(sq_sum / count)) : 0.0f;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
void NanoGICP<PointSource, PointTarget>::calculate_covariances(
    const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    const nanoflann::KdTreeFLANN<PointT>& kdtree,
    CovarianceList& covs,
    float* density,
    std::vector<float>* saliency) {

    covs.resize(cloud->size());
    // Per-point saliency (anti-dilution GICP weighting): only filled when
    // requested (source cloud, feature on) so a disabled run pays no extra cost.
    if (saliency != nullptr) { saliency->assign(cloud->size(), 0.f); }
    float sum_k_sq_distances = 0.0f;

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) \
        reduction(+:sum_k_sq_distances)
    for(int i = 0; i < cloud->size(); ++i) {
        std::vector<int> k_indices(k_correspondences_);
        std::vector<float> k_sq_distances(k_correspondences_);

        // nearestKSearch resizes the outputs to k but only fills the first
        // `found` entries -- the rest are garbage, so never read past `found`.
        int found = kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

        if (found < 4) {
            // not enough neighbors for a meaningful covariance; use a small
            // isotropic placeholder instead of reading uninitialized indices
            Eigen::Matrix4f cov = Eigen::Matrix4f::Zero();
            cov.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * 0.001f;
            cov(3, 3) = 1.0;
            covs[i] = cov;
            continue;
        }

        // accumulate normalized neighborhood spread for the density metric
        // (skip k_sq_distances[0], the query point itself)
        const int normalization = ((found - 1) * (2 + found)) / 2;
        sum_k_sq_distances +=
            std::accumulate(k_sq_distances.begin() + 1, k_sq_distances.begin() + found, 0.0f) / normalization;

        Eigen::Matrix<float, 4, -1> neighbors(4, found);
        for(int j = 0; j < found; ++j) {
            neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap();
        }

        neighbors.row(3).array() = 1.0f;
        Eigen::Vector4f mean = neighbors.rowwise().mean();
        Eigen::Matrix<float, 4, -1> centered = neighbors.colwise() - mean;
        centered.row(3).array() = 0.0f;

        Eigen::Matrix4f cov = (centered * centered.transpose()) / static_cast<float>(found);
        cov(3, 3) = 1.0;

        // Saliency from the RAW neighborhood covariance shape (before the
        // regularization below flattens it): planar wall -> ~0, edge/rib/corner
        // -> ~1. Only when requested (source cloud, feature on).
        if (saliency != nullptr) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(
                cov.block<3, 3>(0, 0), Eigen::EigenvaluesOnly);
            (*saliency)[i] = pointSaliency(es.eigenvalues());
        }

        // NOTE: this rewrite historically labeled its clamped-singular-value
        // regularization "PLANE"; true GICP plane-to-plane (fast_gicp) replaces
        // the singular values with the fixed, scale-free (1, 1, 1e-3). Both are
        // available; MIN_EIG preserves the behavior this fork was tuned on.
        switch (regularization_method_) {
          case RegularizationMethod::PLANE: {
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU);
            Eigen::Vector3f values(1.0f, 1.0f, 1e-3f);
            cov.block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixU().transpose();
            break;
          }
          case RegularizationMethod::MIN_EIG: {
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU);
            Eigen::Vector3f values = svd.singularValues();
            values = values.array().max(0.001f);
            cov.block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixU().transpose();
            break;
          }
          case RegularizationMethod::NORMALIZED_MIN_EIG: {
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU);
            Eigen::Vector3f values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3f);
            cov.block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixU().transpose();
            break;
          }
          case RegularizationMethod::FROBENIUS: {
            cov.block<3, 3>(0, 0) += Eigen::Matrix3f::Identity() * 0.001f;
            break;
          }
          case RegularizationMethod::NONE:
          default:
            break;
        }

        covs[i] = cov;
    }

    if (density != nullptr && !cloud->empty()) {
        *density = sum_k_sq_distances / cloud->size();
    }
}

} // namespace nano_gicp
