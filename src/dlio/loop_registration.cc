// Symmetric point-to-plane loop proposals. These are not graph factors or
// covariance estimates: the independent held-observation gates still apply.
// Prepared clouds own immutable coordinates, trees and normals. They can be
// reused across starts and queried concurrently without holding Python's GIL.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <nano_gicp/nanoflann.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
constexpr double pi = 3.14159265358979323846;

void checkPose(const Eigen::Matrix4d& pose) {
  const auto rotation = pose.topLeftCorner<3, 3>();
  if (!pose.allFinite() || (pose.array().abs() > 1.e6).any() ||
      !pose.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1.e-8) ||
      !(rotation.transpose()*rotation).isApprox(Eigen::Matrix3d::Identity(), 1.e-6) ||
      std::abs(rotation.determinant()-1.) > 1.e-6)
    throw std::invalid_argument("Registration needs a finite rigid pose");
}

class PreparedCloud {
  using Metric = nanoflann::L2_Simple_Adaptor<double, PreparedCloud>;
  using Tree = nanoflann::KDTreeSingleIndexAdaptor<Metric, PreparedCloud, 3>;
 public:
  explicit PreparedCloud(const Eigen::MatrixXd& input) {
    if (input.cols() != 3 || input.rows() < 100 || input.rows() > 30000 ||
        !input.allFinite() || (input.array().abs() > 1.e6).any())
      throw std::invalid_argument("Registration needs 100..30000 finite bounded XYZ points");
    points_ = input;  // Own a copy; mutation of the caller's array cannot stale the tree.
    tree_ = std::make_unique<Tree>(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(16));
    normals_.resize(input.rows(), 3);
    planar_.resize(input.rows());
    std::array<uint32_t, 20> ids{};
    std::array<double, 20> distances{};
    for (Eigen::Index i = 0; i < points_.rows(); ++i) {
      tree_->knnSearch(points_.row(i).data(), ids.size(), ids.data(), distances.data());
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (auto id : ids) mean += points_.row(id).transpose();
      mean /= ids.size();
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
      for (auto id : ids) {
        const Eigen::Vector3d delta = points_.row(id).transpose()-mean;
        covariance.noalias() += delta*delta.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solve(covariance);
      if (solve.info() != Eigen::Success) throw std::invalid_argument("Normal decomposition failed");
      const auto values = solve.eigenvalues();
      normals_.row(i) = solve.eigenvectors().col(0).transpose();
      planar_[i] = values[1] > 1.e-6 && values[0]/std::max(values.sum(), 1.e-12) < .04;
    }
  }
  size_t kdtree_get_point_count() const { return points_.rows(); }
  double kdtree_get_pt(size_t index, size_t dimension) const { return points_(index, dimension); }
  template <class Box> bool kdtree_get_bbox(Box&) const { return false; }

  Eigen::Matrix4d fit(const PreparedCloud& target, const Eigen::Matrix4d& initial,
                      int iterations, bool fine, int maxQueries) const {
    checkPose(initial);
    if (iterations < 1 || iterations > 100 || maxQueries < 100 || maxQueries > 6000)
      throw std::invalid_argument("Registration iterations/queries outside bounded limits");
    Eigen::Matrix4d result = initial;
    const auto sourceIds = queryIds(points_.rows(), maxQueries);
    const auto targetIds = queryIds(target.points_.rows(), maxQueries);
    struct Term { Eigen::Vector3d a, b, n; };
    std::vector<Term> terms;
    terms.reserve(sourceIds.size()+targetIds.size());
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const Eigen::Matrix3d rotation = result.topLeftCorner<3, 3>();
      const Eigen::Vector3d translation = result.topRightCorner<3, 1>();
      const double radius = fine ? .4 : iteration < 15 ? 2.5 : iteration < 35 ? 1. : .4;
      const double radiusSquared = radius*radius;
      terms.clear();
      for (auto id : sourceIds) {
        if (!planar_[id]) continue;
        const Eigen::Vector3d moved = rotation*points_.row(id).transpose()+translation;
        uint32_t nearest;
        double distance;
        target.tree_->knnSearch(moved.data(), 1, &nearest, &distance);
        const Eigen::Vector3d normal = target.normals_.row(nearest).transpose();
        if (distance < radiusSquared && target.planar_[nearest] &&
            std::abs(normal.dot(rotation*normals_.row(id).transpose())) > .85)
          terms.push_back({moved, target.points_.row(nearest).transpose(), normal});
      }
      const size_t forwardCount = terms.size();
      for (auto id : targetIds) {
        if (!target.planar_[id]) continue;
        const Eigen::Vector3d fixed = target.points_.row(id).transpose();
        const Eigen::Vector3d query = rotation.transpose()*(fixed-translation);
        uint32_t nearest;
        double distance;
        tree_->knnSearch(query.data(), 1, &nearest, &distance);
        const Eigen::Vector3d normal = rotation*normals_.row(nearest).transpose();
        if (distance < radiusSquared && planar_[nearest] &&
            std::abs(normal.dot(target.normals_.row(id).transpose())) > .85)
          terms.push_back({rotation*points_.row(nearest).transpose()+translation, fixed, normal});
      }
      const size_t reverseCount = terms.size()-forwardCount;
      if (std::min(forwardCount, reverseCount) < 100)
        throw std::invalid_argument("Insufficient symmetric registration correspondences");
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      for (const auto& term : terms) center += term.a+term.b;
      center /= 2.*terms.size();
      Matrix6 information = Matrix6::Zero();
      Vector6 gradient = Vector6::Zero();
      for (size_t i = 0; i < terms.size(); ++i) {
        const auto& term = terms[i];
        const double residual = (term.a-term.b).dot(term.n);
        const double weight = std::min(1., .1/std::max(std::abs(residual), 1.e-12)) /
                              (i < forwardCount ? forwardCount : reverseCount);
        Vector6 jacobian;
        jacobian.head<3>() = term.n;
        // Reverse residual differentiates the moving normal as well as the
        // point. Its rotational lever arm is therefore the fixed target.
        jacobian.tail<3>() = ((i < forwardCount ? term.a : term.b)-center).cross(term.n);
        information.noalias() += weight*jacobian*jacobian.transpose();
        gradient.noalias() -= (weight*residual)*jacobian;
      }
      Eigen::SelfAdjointEigenSolver<Matrix6> solve(information);
      if (solve.info() != Eigen::Success) throw std::invalid_argument("Registration decomposition failed");
      const auto values = solve.eigenvalues();
      // Python reference uses singular-value rcond=1e-5. Eigenvalues of J'WJ
      // are squared singular values, so apply the squared relative cutoff.
      const double cutoff = std::max(0., values[5])*1.e-10;
      Vector6 delta = Vector6::Zero();
      for (int i = 0; i < 6; ++i)
        if (values[i] > cutoff)
          delta.noalias() += solve.eigenvectors().col(i)*
            (solve.eigenvectors().col(i).dot(gradient)/values[i]);
      if (!delta.allFinite()) throw std::invalid_argument("Nonfinite registration step");
      delta *= std::min({1., .5/std::max(delta.head<3>().norm(), 1.e-12),
                        (5.*pi/180.)/std::max(delta.tail<3>().norm(), 1.e-12)});
      const double angle = delta.tail<3>().norm();
      Eigen::Matrix3d step = Eigen::Matrix3d::Identity();
      if (angle > 0.) step = Eigen::AngleAxisd(angle, delta.tail<3>()/angle).toRotationMatrix();
      result.topLeftCorner<3, 3>() = step*rotation;
      result.topRightCorner<3, 1>() = step*(translation-center)+center+delta.head<3>();
      if ((fine || iteration >= 35) && delta.norm() < 1.e-4) break;
    }
    checkPose(result);
    return result;
  }

  double score(const PreparedCloud& target, const Eigen::Matrix4d& pose) const {
    checkPose(pose);
    const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
    const Eigen::Vector3d translation = pose.topRightCorner<3, 1>();
    double forward = 0., reverse = 0.;
    uint32_t nearest;
    double distance;
    for (Eigen::Index i = 0; i < points_.rows(); ++i) {
      const Eigen::Vector3d moved = rotation*points_.row(i).transpose()+translation;
      target.tree_->knnSearch(moved.data(), 1, &nearest, &distance);
      forward += std::min(distance, 1.);
    }
    for (Eigen::Index i = 0; i < target.points_.rows(); ++i) {
      const Eigen::Vector3d query = rotation.transpose()*(target.points_.row(i).transpose()-translation);
      tree_->knnSearch(query.data(), 1, &nearest, &distance);
      reverse += std::min(distance, 1.);
    }
    return std::sqrt((forward/points_.rows()+reverse/target.points_.rows())/2.);
  }

 private:
  static std::vector<size_t> queryIds(size_t count, int maximum) {
    const size_t size = std::min(count, static_cast<size_t>(maximum));
    std::vector<size_t> result(size);
    for (size_t i = 0; i < size; ++i) result[i] = i*(count-1)/(size-1);
    return result;
  }
  Points points_, normals_;
  std::vector<bool> planar_;
  std::unique_ptr<Tree> tree_;
};
}  // namespace

void bindLoopRegistration(py::module_& module) {
  py::class_<PreparedCloud>(module, "PreparedLoopCloud")
    .def(py::init([](const Eigen::MatrixXd& points) {
      py::gil_scoped_release release;
      return std::make_unique<PreparedCloud>(points);
    }), py::arg("points"))
    .def("fit", &PreparedCloud::fit, py::arg("target"), py::arg("initial"),
         py::arg("iterations") = 70, py::arg("fine") = false, py::arg("max_queries") = 6000,
         py::call_guard<py::gil_scoped_release>())
    .def("score", &PreparedCloud::score, py::arg("target"), py::arg("pose"),
         py::call_guard<py::gil_scoped_release>());
}
