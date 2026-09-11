// Bounded batch Pose3 optimization for the archive worker. No ROS callbacks or
// map storage live here. The Python boundary uses translation-first RIGHT/local
// covariance; GTSAM uses rotation-first. Neither is ROS fixed-axis covariance.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Cholesky>
#include <gtsam/config.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
void bindLoopFeatures(py::module_& module);
void bindLoopRegistration(py::module_& module);
void bindMapGeometry(py::module_& module);

namespace {
constexpr std::size_t kMaxNodes = 5000;
constexpr std::size_t kMaxEdges = 20000;

gtsam::Pose3 checkedPose(const Eigen::Matrix4d& value) {
  const Eigen::Matrix3d rotation = value.topLeftCorner<3, 3>();
  if (!value.allFinite() ||
      !value.row(3).isApprox(Eigen::RowVector4d(0., 0., 0., 1.), 1e-9) ||
      !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
      std::abs(rotation.determinant() - 1.) > 1e-6) {
    throw std::invalid_argument("Graph pose must be a finite proper rigid transform");
  }
  return gtsam::Pose3(value);
}

Matrix6 nativeCovariance(const Matrix6& value) {
  if (!value.allFinite() || !value.isApprox(value.transpose(), 1e-8)) {
    throw std::invalid_argument("Graph covariance must be finite and symmetric");
  }
  const Matrix6 symmetric = .5 * (value + value.transpose());
  Eigen::LLT<Matrix6> check(symmetric);
  if (check.info() != Eigen::Success) {
    throw std::invalid_argument("Graph covariance must be positive definite");
  }
  const int order[] = {3, 4, 5, 0, 1, 2};
  Matrix6 result;
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c) result(r, c) = symmetric(order[r], order[c]);
  return result;
}

Matrix6 nativeProjection(const Matrix6& value, bool loop) {
  if (!value.allFinite() || value.cwiseAbs().maxCoeff() > 1e4 || value.norm() < 1e-12 ||
      (!loop && !value.isApprox(Matrix6::Identity(), 1e-12))) {
    throw std::invalid_argument("Invalid graph residual projection; odometry must remain full rank");
  }
  const int order[] = {3, 4, 5, 0, 1, 2};
  Matrix6 result;
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c) result(r, c) = value(order[r], order[c]);
  return result;
}

// Explicit Logmap and its derivative keep residuals independent of GTSAM's
// optional Pose3 chart and its fast BetweenFactor Jacobian approximation.
class LogBetweenFactor final : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  LogBetweenFactor(gtsam::Key from, gtsam::Key to, const gtsam::Pose3& measured,
                   const gtsam::SharedNoiseModel& noise, const Matrix6& projection)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(noise, from, to),
        measured_(measured), projection_(projection) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& from, const gtsam::Pose3& to,
#if GTSAM_VERSION_NUMERIC >= 40300
      gtsam::OptionalMatrixType h1, gtsam::OptionalMatrixType h2) const override {
#else
      boost::optional<gtsam::Matrix&> h1 = boost::none,
      boost::optional<gtsam::Matrix&> h2 = boost::none) const override {
#endif
    Matrix6 between1, between2, logarithm;
    const auto relative = from.between(to, between1, between2);
    const auto error = gtsam::Pose3::Logmap(measured_.inverse() * relative, logarithm);
    if (h1) *h1 = projection_ * logarithm * between1;
    if (h2) *h2 = projection_ * logarithm * between2;
    return projection_ * error;
  }

 private:
  gtsam::Pose3 measured_;
  Matrix6 projection_;
};

Vector6 residual(const Eigen::Matrix4d& measured, const Eigen::Matrix4d& from,
                 const Eigen::Matrix4d& to) {
  const Vector6 native = gtsam::Pose3::Logmap(
      checkedPose(measured).inverse() * checkedPose(from).between(checkedPose(to)));
  Vector6 result;
  result << native.tail<3>(), native.head<3>();
  return result;
}

py::dict optimize(const std::vector<Eigen::Matrix4d>& poses,
                  const std::vector<std::size_t>& from,
                  const std::vector<std::size_t>& to,
                  const std::vector<Eigen::Matrix4d>& measurements,
                  const std::vector<Matrix6>& covariances,
                  const std::vector<bool>& loops, unsigned int maxIterations,
                  const std::vector<Matrix6>& projections) {
  const auto count = from.size();
  if (poses.size() < 2 || poses.size() > kMaxNodes || count > kMaxEdges ||
      count != to.size() || count != measurements.size() || count != covariances.size() ||
      count != loops.size() || (!projections.empty() && count != projections.size()) ||
      maxIterations < 1 || maxIterations > 200) {
    throw std::invalid_argument("Invalid or oversized pose graph");
  }
  std::vector<Eigen::Matrix4d> corrected;
  std::vector<double> squaredErrors;
  double initialError, finalError;
  unsigned int iterations;
  {
    py::gil_scoped_release release;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    for (std::size_t i = 0; i < poses.size(); ++i) initial.insert(i, checkedPose(poses[i]));
    // A numerical gauge anchor, not a measured covariance or world certainty.
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        0, checkedPose(poses[0]), gtsam::noiseModel::Isotropic::Sigma(6, 1e-6));
    std::vector<Matrix6> nativeCovariances;
    std::vector<Matrix6> nativeProjections;
    std::vector<bool> connected(poses.size() - 1, false);
    for (std::size_t i = 0; i < count; ++i) {
      if (from[i] >= poses.size() || to[i] >= poses.size() || from[i] >= to[i])
        throw std::invalid_argument("Graph edges require ordered, distinct, known IDs");
      if (!loops[i]) {
        if (to[i] != from[i] + 1 || connected[from[i]])
          throw std::invalid_argument("Odometry must form a unique consecutive chain");
        connected[from[i]] = true;
      }
      nativeCovariances.push_back(nativeCovariance(covariances[i]));
      nativeProjections.push_back(nativeProjection(
          projections.empty() ? Matrix6::Identity() : projections[i], loops[i]));
      gtsam::SharedNoiseModel noise = gtsam::noiseModel::Gaussian::Covariance(nativeCovariances.back());
      if (loops[i]) noise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(2.5), noise);
      graph.emplace_shared<LogBetweenFactor>(from[i], to[i], checkedPose(measurements[i]), noise,
                                             nativeProjections.back());
    }
    for (bool present : connected)
      if (!present) throw std::invalid_argument("Graph odometry chain is disconnected");
    gtsam::LevenbergMarquardtParams parameters;
    parameters.maxIterations = maxIterations;
    parameters.relativeErrorTol = 1e-8;
    parameters.absoluteErrorTol = 1e-8;
    parameters.errorTol = 1e-12;
    initialError = graph.error(initial);
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, parameters);
    const auto result = optimizer.optimize();
    iterations = optimizer.iterations();
    finalError = graph.error(result);
    // Remove numerical gauge drift without changing any relative constraint.
    const auto gauge = checkedPose(poses[0]) * result.at<gtsam::Pose3>(0).inverse();
    for (std::size_t i = 0; i < poses.size(); ++i)
      corrected.push_back((gauge * result.at<gtsam::Pose3>(i)).matrix());
    for (std::size_t i = 0; i < count; ++i) {
      const Vector6 error = nativeProjections[i] * gtsam::Pose3::Logmap(checkedPose(measurements[i]).inverse() *
          checkedPose(corrected[from[i]]).between(checkedPose(corrected[to[i]])));
      // Unrobustified residual: downweighting must never hide a failed loop.
      squaredErrors.push_back(error.dot(nativeCovariances[i].llt().solve(error)));
    }
  }
  py::dict result;
  result["poses"] = corrected;
  result["squared_errors"] = squaredErrors;
  result["initial_error"] = initialError;
  result["final_error"] = finalError;
  result["iterations"] = iterations;
  result["converged"] = std::isfinite(initialError) && std::isfinite(finalError) &&
      finalError <= initialError + 1e-8 && iterations < maxIterations;
  return result;
}
}  // namespace

PYBIND11_MODULE(_dliio_pose_graph, module) {
  bindLoopFeatures(module);
  bindLoopRegistration(module);
  bindMapGeometry(module);
  module.doc() = "Bounded GTSAM batch optimizer; all inputs use right/local translation-first covariance";
  module.def("optimize", &optimize, py::arg("poses"), py::arg("from_ids"), py::arg("to_ids"),
      py::arg("measurements"), py::arg("covariances"), py::arg("loops"), py::arg("max_iterations") = 100,
      py::arg("projections") = std::vector<Matrix6>{});
  module.def("residual", &residual);
  module.attr("gtsam_version") = GTSAM_VERSION_STRING;
  module.attr("supports_projected_loops") = true;
}
