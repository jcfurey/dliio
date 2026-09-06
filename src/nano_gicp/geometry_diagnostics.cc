#include "nano_gicp/geometry_diagnostics.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace nano_gicp {
namespace {
Eigen::Matrix3d crossMatrix(const Eigen::Vector3d& p) {
  Eigen::Matrix3d m;
  m << 0.0, -p.z(), p.y(), p.z(), 0.0, -p.x(), -p.y(), p.x(), 0.0;
  return m;
}

template<int N>
bool spectrum(const Eigen::Matrix<double, N, N>& h,
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>>& eig) {
  if (!h.allFinite()) { return false; }
  eig.compute((h + h.transpose()).eval() * 0.5);
  return eig.info() == Eigen::Success && eig.eigenvalues().allFinite() &&
      eig.eigenvalues()(N - 1) > 0.0 &&
      eig.eigenvalues()(0) >= -1e-9 * eig.eigenvalues()(N - 1);
}

double scaledRatio(const Matrix6d& centered, double length) {
  Matrix6d scale = Matrix6d::Identity();
  scale.topLeftCorner<3, 3>() /= length;
  const Matrix6d h = scale * centered * scale;
  Eigen::SelfAdjointEigenSolver<Matrix6d> eig;
  if (!spectrum<6>(h, eig)) { return -1.0; }
  return std::max(0.0, eig.eigenvalues()(0)) / eig.eigenvalues()(5);
}
}  // namespace

GeometryDiagnostics analyzeGeometry(const Matrix6d& world_hessian,
                                    const Eigen::Vector3d& scan_center,
                                    double length_scale) {
  GeometryDiagnostics out;
  if (!world_hessian.allFinite() || !scan_center.allFinite() ||
      !std::isfinite(length_scale) || length_scale <= 0.0) { return out; }
  out.length_scale = length_scale;

  // World left perturbation [w, v] and centered perturbation [w, dc] obey
  // v = dc + center x w. Thus J_world * A = [-skew(p - center), I].
  // Transform a copy; never change the system that the estimator solves.
  Matrix6d A = Matrix6d::Identity();
  A.bottomLeftCorner<3, 3>() = crossMatrix(scan_center);
  const Matrix6d centered = A.transpose() * world_hessian * A;
  Matrix6d scale = Matrix6d::Identity();
  scale.topLeftCorner<3, 3>() /= length_scale;
  const Matrix6d h = scale * centered * scale;
  Eigen::SelfAdjointEigenSolver<Matrix6d> eig;
  if (!spectrum<6>(h, eig)) { return out; }
  out.valid = true;
  out.hessian = (h + h.transpose()).eval() * 0.5;
  out.eigenvalues = eig.eigenvalues().cwiseMax(0.0);
  out.weakest = eig.eigenvectors().col(0);
  Eigen::Index largest;
  out.weakest.cwiseAbs().maxCoeff(&largest);
  if (out.weakest(largest) < 0.0) { out.weakest = -out.weakest; }
  out.half_length_ratio = scaledRatio(centered, length_scale * 0.5);
  out.double_length_ratio = scaledRatio(centered, length_scale * 2.0);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> rr, tt;
  const Eigen::Matrix3d Hrr = h.topLeftCorner<3, 3>();
  const Eigen::Matrix3d Htt = h.bottomRightCorner<3, 3>();
  const bool rr_valid = spectrum<3>(Hrr, rr);
  const bool tt_valid = spectrum<3>(Htt, tt);
  if (rr_valid) { out.rotation_ratio = std::max(0.0, rr.eigenvalues()(0)) / rr.eigenvalues()(2); }
  if (tt_valid) { out.translation_ratio = std::max(0.0, tt.eigenvalues()(0)) / tt.eigenvalues()(2); }
  if (rr_valid && tt_valid) {
    Eigen::Vector3d inverse = Eigen::Vector3d::Zero();
    const double tolerance = 1e-9 * rr.eigenvalues()(2);
    for (int i = 0; i < 3; ++i) {
      if (rr.eigenvalues()(i) > tolerance) { inverse(i) = 1.0 / rr.eigenvalues()(i); }
    }
    const Eigen::Matrix3d pinv = rr.eigenvectors() * inverse.asDiagonal() * rr.eigenvectors().transpose();
    const Eigen::Matrix3d Hrt = h.topRightCorner<3, 3>();
    const Eigen::Matrix3d schur = Htt - Hrt.transpose() * pinv * Hrt;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> se((schur + schur.transpose()).eval() * 0.5);
    // The Schur matrix may be entirely zero. Compare roundoff with Htt,
    // rather than the vanishing Schur maximum, before accepting that case.
    if (se.info() == Eigen::Success && se.eigenvalues().allFinite() &&
        se.eigenvalues()(0) >= -1e-8 * tt.eigenvalues()(2)) {
      const double minimum = std::max(0.0, se.eigenvalues()(0));
      out.schur_ratio = minimum / tt.eigenvalues()(2);
      if (tt.eigenvalues()(0) > 1e-9 * tt.eigenvalues()(2)) {
        out.schur_retained = minimum / tt.eigenvalues()(0);
      }
    }
  }
  return out;
}

}  // namespace nano_gicp
