#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
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

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

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

  // Degeneracy gating (solution remapping): the rotation and translation
  // Hessian blocks are eigen-analyzed separately; the update is projected off
  // directions with eigenvalue < ratio * block_lambda_max, so the initial
  // guess (IMU prior) is held along unobservable directions, e.g. the axis of
  // a featureless tunnel. 0 disables the gate. Most discriminative with the
  // PLANE regularization method.
  void setDegeneracyThreshRatio(float ratio);
  // Number of degenerate directions detected during the last align() (0-6).
  int lastDegenerateDirections() const;

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
  int last_degenerate_directions_;
  
  std::shared_ptr<const GradientList> target_intensity_gradients_;
  std::shared_ptr<const std::vector<bool>> gradient_valid_;
};

} // namespace nano_gicp
