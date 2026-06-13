#include "nano_gicp/nano_gicp.h"
#include "dlio/dlio.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <Eigen/Dense>
#include <pcl/common/transforms.h>

namespace {
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
  this->photometric_huber_delta_ = 0.05f;
  this->degeneracy_thresh_ratio_ = 0.005f;
  this->last_degenerate_directions_ = 0;
  this->visual_enabled_ = false;
  this->visual_weight_ = 0.0f;
  this->visual_huber_delta_ = 0.05f;
  this->visual_fx_ = this->visual_fy_ = this->visual_cx_ = this->visual_cy_ = 0.0f;
  this->T_cw_cur_ = Eigen::Isometry3f::Identity();
  this->T_cw_prev_ = Eigen::Isometry3f::Identity();
  this->visual_gate_max_trans_ = 0.5f;
  this->visual_gate_max_rot_ = 0.1f;
  this->last_visual_rms_ = 0.0f;
  this->last_visual_count_ = 0;
  this->last_visual_rescued_ = 0;
  this->visual_map_weight_ = 0.0f;
  this->visual_map_gate_max_trans_ = 1.0f;
  this->visual_map_gate_max_rot_ = 0.1f;
  this->visual_map_view_angle_max_ = 0.6f;  // ~34 deg
  this->last_visual_map_rms_ = 0.0f;
  this->last_visual_map_count_ = 0;
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
    this->photometric_weight_ = weight;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setGradientKNeighbors(int k) {
    this->gradient_k_neighbors_ = k;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricChannel(bool use_reflectivity) {
    this->photometric_use_reflectivity_ = use_reflectivity;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricScale(float scale) {
    this->photometric_scale_ = (scale > 0.f) ? scale : 1.0f;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricHuberDelta(float delta) {
    this->photometric_huber_delta_ = delta;
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
float NanoGICP<PointSource, PointTarget>::lastVisualRms() const {
    return this->last_visual_rms_;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastVisualCount() const {
    return this->last_visual_count_;
}

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
void NanoGICP<PointSource, PointTarget>::setDegeneracyThreshRatio(float ratio) {
    this->degeneracy_thresh_ratio_ = ratio;
}

template <typename PointSource, typename PointTarget>
int NanoGICP<PointSource, PointTarget>::lastDegenerateDirections() const {
    return this->last_degenerate_directions_;
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
  pcl::Registration<PointSource, PointTarget>::setInputSource(cloud);
  
  input_kdtree_.reset(new nanoflann::KdTreeFLANN<PointSource>(false));
  input_kdtree_->setInputCloud(cloud);

  calculate_covariances(cloud, *input_kdtree_, source_covs_, &source_density_);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  registerInputTarget(cloud);

  auto covs = std::make_shared<CovarianceList>();
  calculate_covariances(cloud, *target_kdtree_, *covs);
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputTarget(const PointCloudTargetConstPtr& cloud) {
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
  target_visual_refs_ = other.target_visual_refs_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::calculate_target_intensity_gradients() {
    if (!target_) {
        target_intensity_gradients_.reset();
        gradient_valid_.reset();
        return;
    }

    auto gradients = std::make_shared<GradientList>(target_->size(), Eigen::Vector3f::Zero());
    auto valid = std::make_shared<std::vector<bool>>(target_->size(), false);

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
    return (use_refl ? p.reflectivity : p.intensity) * inv_scale;
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
  
  // Solve least squares
  Eigen::Matrix4f AtA = A.transpose() * A;
  Eigen::Vector4f Ati = A.transpose() * i;
  
  // Check condition number
  Eigen::JacobiSVD<Eigen::Matrix4f> svd(AtA);
  float cond = svd.singularValues()(0) / svd.singularValues()(3);
  if (cond > 1e6 || !std::isfinite(cond)) {
      return false;
  }
  
  Eigen::Vector4f g = AtA.ldlt().solve(Ati);
  
  if (g.hasNaN() || !g.allFinite()) {
      return false;
  }
  
  gradient = g.head<3>();
  
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

    // Fallback: if the target was registered without precomputed covariances
    // (or with a mismatched set), compute them here.
    if (!target_covs_ || target_covs_->size() != target_->size()) {
        auto covs = std::make_shared<CovarianceList>();
        calculate_covariances(target_, *target_kdtree_, *covs);
        target_covs_ = covs;
    }

    this->converged_ = false;
    this->last_degenerate_directions_ = 0;
    this->last_visual_count_ = 0;
    this->last_visual_rms_ = 0.0f;
    this->last_visual_rescued_ = 0;
    this->last_visual_map_count_ = 0;
    this->last_visual_map_rms_ = 0.0f;

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

    for (int i = 0; i < this->max_iterations_; ++i) {
        update_correspondences(trans);

        // Accumulated, solved, and gated in double; see linearize() for why.
        Eigen::Matrix<double, 6, 6> H;
        Eigen::Matrix<double, 6, 1> b;
        double cost = 0.0;

        linearize(trans, &H, &b, &cost);

        if (cost > prev_cost) {
            // last step made things worse: revert it and damp harder
            trans = prev_trans;
            lambda *= kLambdaScale;
            if (lambda > kLambdaMax) { break; }  // no progress possible
            update_correspondences(trans);
            linearize(trans, &H, &b, &cost);
        } else {
            lambda = std::max(lambda / kLambdaScale, base_lambda);
        }
        prev_cost = cost;
        prev_trans = trans;

        // Snapshot the GEOMETRIC (LiDAR-only) Hessian before adding the visual
        // term. The degeneracy gate judges observability from this, NOT from
        // the visual-augmented H: otherwise a strong-but-wrong visual term
        // masks the degeneracy, the gate releases its prior-hold, and the pose
        // diverges along the (still physically unobservable) axis.
        const Eigen::Matrix<double, 6, 6> H_geo = H;

        // Direct visual (camera) photometric term: accumulate into the SAME
        // H/b that linearize() built, BEFORE damping and the degeneracy gate,
        // so a camera-constrained axis can be detected (and rescued) by the
        // gate below. The LM step-acceptance cost above stays geometric-only
        // (visual is a secondary constraint); the visual term still shapes dx.
        // No-op unless setVisualEnabled(true) and both frames are set.
        if (visual_enabled_) {
            accumulateVisualResidual(trans, &H, &b, nullptr);
        }
        // Frame-to-MAP camera term (absolute anchor): also into H/b before the
        // gate, so it can rescue the degenerate axis with map landmarks.
        if (visual_map_weight_ > 0.f) {
            accumulateVisualMapResidual(trans, &H, &b, nullptr);
        }

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
        //     but CLAMP the step to ±visual_gate_max_* so a wrong visual
        //     constraint cannot run away; otherwise hold the prior.
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
            const double cap_t = (last_visual_map_count_ > 0)
                ? static_cast<double>(visual_map_gate_max_trans_) : static_cast<double>(visual_gate_max_trans_);
            const double cap_r = (last_visual_map_count_ > 0)
                ? static_cast<double>(visual_map_gate_max_rot_) : static_cast<double>(visual_gate_max_rot_);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_rr(H_geo.template block<3, 3>(0, 0));
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_tt(H_geo.template block<3, 3>(3, 3));
            // Combined (geometric + visual) blocks, to test visual rescue.
            const Eigen::Matrix3d Hrr = H.template block<3, 3>(0, 0);
            const Eigen::Matrix3d Htt = H.template block<3, 3>(3, 3);

            const double rr_thresh = degeneracy_thresh_ratio_ * eig_rr.eigenvalues()(2);
            for (int k = 0; k < 3; ++k) {
                if (eig_rr.eigenvalues()(k) <= rr_thresh) {
                    ++degenerate;
                    const Eigen::Vector3d v = eig_rr.eigenvectors().col(k);
                    const double comp = v.dot(dx.head<3>());
                    if (visual_enabled_ && v.dot(Hrr * v) > rr_thresh) {
                        // bounded visual-driven motion, drawing from the per-scan budget
                        const double cap = std::max(0.0, cap_r - rescued_r_used);
                        const double cl = std::max(-cap, std::min(cap, comp));
                        dx.head<3>() += v * (cl - comp);
                        rescued_r_used += std::abs(cl);
                        ++rescued;
                    } else {
                        dx.head<3>() -= v * comp;          // hold the prior
                    }
                }
            }
            const double tt_thresh = degeneracy_thresh_ratio_ * eig_tt.eigenvalues()(2);
            for (int k = 0; k < 3; ++k) {
                if (eig_tt.eigenvalues()(k) <= tt_thresh) {
                    ++degenerate;
                    const Eigen::Vector3d v = eig_tt.eigenvectors().col(k);
                    const double comp = v.dot(dx.tail<3>());
                    if (visual_enabled_ && v.dot(Htt * v) > tt_thresh) {
                        // bounded visual-driven motion, drawing from the per-scan budget
                        const double cap = std::max(0.0, cap_t - rescued_t_used);
                        const double cl = std::max(-cap, std::min(cap, comp));
                        dx.tail<3>() += v * (cl - comp);
                        rescued_t_used += std::abs(cl);
                        ++rescued;
                    } else {
                        dx.tail<3>() -= v * comp;          // hold the prior
                    }
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

        // Check convergence: rotation step (dx.head) vs rotation_epsilon_,
        // translation step (dx.tail) vs transformation_epsilon_.
        if (dx.head<3>().norm() < rotation_epsilon_ &&
            dx.tail<3>().norm() < transformation_epsilon_) {
            this->converged_ = true;
            break;
        }
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

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) \
        firstprivate(k_indices, k_sq_dists)
    for (size_t i = 0; i < input_->size(); ++i) {
        PointTarget transformed_pt;
        transformed_pt.getVector4fMap() = trans * input_->at(i).getVector4fMap();

        target_kdtree_->nearestKSearch(transformed_pt, 1, k_indices, k_sq_dists);
        
        float max_dist_sq = this->corr_dist_threshold_ * this->corr_dist_threshold_;
        if (k_sq_dists[0] < max_dist_sq) {
            correspondences_[i] = k_indices[0];
            sq_distances_[i] = k_sq_dists[0];
            
            const Eigen::Matrix4f& source_cov = source_covs_[i];
            const Eigen::Matrix4f& target_cov = (*target_covs_)[k_indices[0]];
            Eigen::Matrix4f RCR = (source_cov + target_cov);
            RCR(3, 3) = 1.0;
            
            mahalanobis_[i] = RCR.inverse();
            mahalanobis_[i](3, 3) = 0.0;
        }
    }
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::linearize(
    const Eigen::Isometry3f& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* cost) {
    
    H->setZero();
    b->setZero();

    // Per-point Jacobians are computed in float (the data is float), but the
    // accumulation across 10k+ points is done in double: summing that many
    // float products loses several significant digits, which matters for the
    // near-singular Hessians the degeneracy gate has to discriminate.
    std::vector<Eigen::Matrix<double, 6, 6>> H_private(num_threads_, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_private(num_threads_, Eigen::Matrix<double, 6, 1>::Zero());
    double cost_sum = 0.0;
    
    bool use_photometric = (photometric_weight_ > 1e-8)
        && target_intensity_gradients_ && !target_intensity_gradients_->empty();
    
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:cost_sum)
    for(int i = 0; i < input_->size(); ++i) {
        int thread_num = omp_get_thread_num();
        int target_index = correspondences_[i];
        
        if(target_index < 0) {
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
        
        H_private[thread_num] += (J_geometric.transpose() * M * J_geometric).cast<double>();
        b_private[thread_num] += (J_geometric.transpose() * M * residual).cast<double>();
        cost_sum += residual.transpose() * M * residual;
        
        // Photometric term (channel normalized by photometric_scale_ to match
        // the units the target gradients were estimated in)
        if (use_photometric && (*gradient_valid_)[target_index]) {
            float src_val = photometric_use_reflectivity_ ? source_pt.reflectivity : source_pt.intensity;
            float tgt_val = photometric_use_reflectivity_ ? target_pt.reflectivity : target_pt.intensity;
            float intensity_diff = (src_val - tgt_val) / photometric_scale_;
            Eigen::Vector3f gradient = (*target_intensity_gradients_)[target_index];
            
            if (gradient.norm() > kGradientMagMin && gradient.norm() < kGradientMagMax) {
                // Residual is (I_src - I_tgt); its Jacobian w.r.t. the (left-perturbation)
                // pose is d/dθ (I_src - I_tgt(x)) = -gᵀ·dx/dθ = [ gᵀ·skew(x) | -gᵀ ].
                // (Must match the geometric term's convention or the photometric step
                //  ascends intensity error instead of descending it.)
                Eigen::Matrix<float, 1, 6> J_photometric;
                J_photometric.block<1, 3>(0, 0) = gradient.transpose() * skew(transformed_source);
                J_photometric.block<1, 3>(0, 3) = -gradient.transpose();

                // Huber robustification: photometric outliers (specular
                // returns, wet patches, exposure-like artifacts) otherwise
                // shove the pose at full weight -- IRLS down-weighting
                // beyond photometric_huber_delta_ (normalized units).
                float weight = photometric_weight_;
                const float abs_r = std::abs(intensity_diff);
                if (photometric_huber_delta_ > 0.f && abs_r > photometric_huber_delta_) {
                    weight *= photometric_huber_delta_ / abs_r;
                }
                H_private[thread_num] += (weight * J_photometric.transpose() * J_photometric).cast<double>();
                b_private[thread_num] += (weight * J_photometric.transpose() * intensity_diff).cast<double>();
                cost_sum += weight * intensity_diff * intensity_diff;
            }
        }
    }
    
    for(int i = 0; i < num_threads_; ++i) {
        (*H) += H_private[i];
        (*b) += b_private[i];
    }
    if (cost != nullptr) { *cost = cost_sum; }
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::accumulateVisualResidual(
    const Eigen::Isometry3f& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* cost) {

    this->last_visual_count_ = 0;
    this->last_visual_rms_ = 0.0f;

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

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:cost_sum,sq_sum,count)
    for (int i = 0; i < input_->size(); ++i) {
        const auto& sp = input_->at(i);
        const Eigen::Vector3f p_w(sp.x, sp.y, sp.z);     // raw world point (prior pose)
        const Eigen::Vector3f x = trans * p_w;           // pose-corrected world point

        // Reference brightness: project the RAW world point into the CURRENT
        // image. This is pose-independent (the point's position relative to the
        // current camera is fixed by the rigid extrinsic), hence a fixed target.
        const Eigen::Vector3f Pc_ref = R_cur * p_w + t_cur;
        if (Pc_ref.z() <= zmin) { continue; }
        const float u_ref = fx * Pc_ref.x() / Pc_ref.z() + cx;
        const float v_ref = fy * Pc_ref.y() / Pc_ref.z() + cy;
        if (u_ref < bw || u_ref > cur_umax || v_ref < bw || v_ref > cur_vmax) { continue; }
        const float I_ref = bilinearSample(visual_cur_, u_ref, v_ref);

        // Moving brightness: project the CORRECTED world point into the PREVIOUS
        // image. This depends on `trans` -- the source of pose observability.
        const Eigen::Vector3f Pc = R_prev * x + t_prev;
        if (Pc.z() <= zmin) { continue; }
        const float invz = 1.f / Pc.z();
        const float u = fx * Pc.x() * invz + cx;
        const float v = fy * Pc.y() * invz + cy;
        if (u < bw || u > prev_umax || v < bw || v > prev_vmax) { continue; }
        const float I_mov = bilinearSample(visual_prev_, u, v);

        // Image gradient (central difference) on the previous image, normalized.
        const float gu = 0.5f * (bilinearSample(visual_prev_, u + 1.f, v)
                               - bilinearSample(visual_prev_, u - 1.f, v));
        const float gv = 0.5f * (bilinearSample(visual_prev_, u, v + 1.f)
                               - bilinearSample(visual_prev_, u, v - 1.f));
        if (std::abs(gu) < 1e-6f && std::abs(gv) < 1e-6f) { continue; }

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
        cost_sum += weight * r * r;
        sq_sum += static_cast<double>(r) * r;
        count += 1;
    }

    for (int t = 0; t < num_threads_; ++t) {
        (*H) += H_private[t];
        (*b) += b_private[t];
    }
    if (cost != nullptr) { *cost += cost_sum; }
    this->last_visual_count_ = static_cast<int>(count);
    this->last_visual_rms_ = (count > 0) ? std::sqrt(static_cast<float>(sq_sum / count)) : 0.0f;
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
        if (u < bw || u > umax || v < bw || v > vmax) { continue; }

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
        cost_sum += weight * r * r;
        sq_sum += static_cast<double>(r) * r;
        count += 1;
    }

    for (int t = 0; t < num_threads_; ++t) {
        (*H) += H_private[t];
        (*b) += b_private[t];
    }
    if (cost != nullptr) { *cost += cost_sum; }
    this->last_visual_map_count_ = static_cast<int>(count);
    this->last_visual_map_rms_ = (count > 0) ? std::sqrt(static_cast<float>(sq_sum / count)) : 0.0f;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
void NanoGICP<PointSource, PointTarget>::calculate_covariances(
    const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    const nanoflann::KdTreeFLANN<PointT>& kdtree,
    CovarianceList& covs,
    float* density) {

    covs.resize(cloud->size());
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
