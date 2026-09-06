#pragma once

#include <Eigen/Core>

namespace nano_gicp {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// Read-only analysis of one geometry-only linearization at the registration
// prior. Coordinates are [L * rotation_about_scan_center, center_translation],
// in meters and expressed along the target/world axes. These normal-matrix
// quantities are NOT calibrated measurement information or a pose covariance.
struct GeometryDiagnostics {
  bool valid = false;
  double length_scale = 0.0;
  Matrix6d hessian = Matrix6d::Zero();  // symmetric, centered and scaled
  Vector6d eigenvalues = Vector6d::Constant(-1.0);  // ascending
  Vector6d weakest = Vector6d::Zero();             // unit vector; sign canonicalized
  double rotation_ratio = -1.0;
  double translation_ratio = -1.0;
  // Minimum translation eigenvalue after allowing rotation to adjust, divided
  // by the conditional translation maximum / minimum respectively. -1 = n/a.
  double schur_ratio = -1.0;
  double schur_retained = -1.0;
  double half_length_ratio = -1.0;
  double double_length_ratio = -1.0;
  // Final registration correction in the same centered/scaled coordinates.
  // Finite rotation vector + center displacement, not an SE(3) logarithm.
  Vector6d correction = Vector6d::Zero();
  double correction_projection = 0.0;
  // |J_texture * weakest| for a unit scalar center-translation residual.
  // -1 when no texture measurement was accepted.
  double texture_projection = -1.0;
};

GeometryDiagnostics analyzeGeometry(const Matrix6d& world_hessian,
                                    const Eigen::Vector3d& scan_center,
                                    double length_scale);

}  // namespace nano_gicp
