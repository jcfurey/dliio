#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace nano_gicp {

using TextureCloud = pcl::PointCloud<pcl::PointXYZI>;

// Experimental frame-to-frame texture measurement on deskewed 3D surfaces.
// The caller must compensate BOTH moving-head and vehicle motion before use.
// No ring layout, spherical projection, or intensity-as-reflectivity assumption.
struct SurfaceTextureConfig {
  float search_radius = 0.4f;       // prior correction, not platform motion [m]
  float sample_radius = 0.12f;      // maximum interpolation support distance [m]
  float min_std = 0.006f;           // normalized intensity contrast
  int max_patches = 120;
  int min_patches = 8;
};

struct SurfaceTextureMatch {
  bool valid = false;
  int candidates = 0;
  int supported = 0;
  int unique = 0;
  int inliers = 0;
  float shift = 0.f;               // translation to ADD to the source [m]
  float sigma = 0.f;               // robust patch disagreement, floored at 3 cm
  float correlation = 0.f;
};

SurfaceTextureMatch matchSurfaceTexture(
    const TextureCloud::ConstPtr& source, const TextureCloud::ConstPtr& reference,
    const Eigen::Isometry3f& source_to_reference, const Eigen::Vector3f& axis,
    const SurfaceTextureConfig& config = {});

// A fixed measurement of the scan center along a world-frame weak axis.
// Its residual, Jacobian and acceptance cost describe the SAME objective under
// the optimizer's left SE(3) perturbation, including the rotation lever arm.
struct SurfaceTextureConstraint {
  Eigen::Vector3f axis = Eigen::Vector3f::UnitX();
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  double position = 0.0;
  double information = 0.0;

  void accumulate(const Eigen::Isometry3f& transform,
                  Eigen::Matrix<double, 6, 6>* H,
                  Eigen::Matrix<double, 6, 1>* b, double* cost) const;
};

}  // namespace nano_gicp
