// Dense revision geometry only. Immutable observations and SQLite transaction
// ownership remain in the storage worker; this function cannot commit a map.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/Geometry>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
Eigen::Matrix4d checkedPose(py::handle value, bool rigid = true) {
  auto array = py::array::ensure(value);
  if (!array || array.ndim() != 2 || array.shape(0) != 4 || array.shape(1) != 4)
    throw std::invalid_argument("Pose must be a finite 4x4 matrix");
  auto data = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(array);
  if (!data) throw std::invalid_argument("Pose must be a finite 4x4 matrix");
  Eigen::Matrix4d pose = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(data.data());
  if (!pose.allFinite()) throw std::invalid_argument("Pose must be a finite 4x4 matrix");
  // Relative matrices come from the reference's inv(anchor) @ pose. Combining
  // two rotations accepted at the tolerance boundary may exceed that tolerance.
  if (!rigid) return pose;
  const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
  if (!pose.allFinite() ||
      (pose.row(3)-Eigen::RowVector4d(0, 0, 0, 1)).cwiseAbs().maxCoeff() > 1.e-9 ||
      (rotation.transpose()*rotation-Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1.e-6 ||
      std::abs(rotation.determinant()-1.) > 1.e-6)
    throw std::invalid_argument("Pose must contain a proper rigid rotation");
  return pose;
}

py::array_t<float> reconstructDense(const py::list& clouds, const py::list& world_poses,
                                  const py::list& relative_poses, py::ssize_t max_points) {
  const auto groups = clouds.size();
  if (groups < 1 || groups > 1000 || world_poses.size() != groups || relative_poses.size() != groups ||
      max_points < 1 || max_points > 1000000)
    throw std::invalid_argument("Dense reconstruction needs bounded matching observations and poses");
  std::vector<py::array> inputs;
  std::vector<Eigen::Matrix4d> world, relative;
  py::ssize_t total = 0;
  for (py::size_t i = 0; i < groups; ++i) {
    if (!py::isinstance<py::array>(clouds[i]))
      throw std::invalid_argument("Dense observations must be contiguous float32 Nx7 arrays");
    auto points = py::reinterpret_borrow<py::array>(clouds[i]);
    if (!points.dtype().is(py::dtype::of<float>()) || points.ndim() != 2 || points.shape(1) != 7 ||
        points.shape(0) < 1 || !(points.flags() & py::array::c_style))
      throw std::invalid_argument("Dense observations must be contiguous float32 Nx7 arrays");
    if (points.shape(0) > max_points-total)
      throw std::invalid_argument("Corrected submap exceeds point capacity");
    total += points.shape(0);
    inputs.push_back(points);
    world.push_back(checkedPose(world_poses[i]));
    relative.push_back(checkedPose(relative_poses[i], false));
  }
  // Copy all seven fields while holding the GIL. The numeric loop owns its
  // input storage even when callers supplied mutable arrays. memcpy retains
  // scalar NaN payloads, signed zero, point order and coincident samples.
  py::array_t<float> output({total, py::ssize_t(7)});
  float* values = output.mutable_data();
  py::ssize_t offset = 0;
  for (const auto& points : inputs) {
    std::memcpy(values+offset*7, points.data(), points.shape(0)*7*sizeof(float));
    offset += points.shape(0);
  }
  {
    py::gil_scoped_release release;
    offset = 0;
    for (py::size_t i = 0; i < groups; ++i) {
      const auto count = inputs[i].shape(0);
      for (py::ssize_t j = 0; j < count; ++j) {
        float* row = values+(offset+j)*7;
        for (int k = 0; k < 7; ++k)
          if ((k < 3 && !std::isfinite(row[k])) || (k >= 3 && std::isinf(row[k])))
            throw std::invalid_argument("Invalid archived scalar values");
        const Eigen::Vector3d xyz(row[0], row[1], row[2]);
        const Eigen::Vector3f global = (world[i].topLeftCorner<3, 3>()*xyz+
                                      world[i].topRightCorner<3, 1>()).cast<float>();
        if (!global.allFinite())
          throw std::invalid_argument("Transformed XYZ is not representable as float32");
        if (global.cast<double>().cwiseAbs().maxCoeff()/.001 >= std::ldexp(1., 62))
          throw std::invalid_argument("Corrected map exceeds the supported fusion coordinate range");
        // The first observation defines the anchor. Preserve its local XYZ
        // exactly instead of applying a numerically approximate identity.
        if (i != 0) {
          const Eigen::Vector3f moved = (relative[i].topLeftCorner<3, 3>()*xyz+
                                        relative[i].topRightCorner<3, 1>()).cast<float>();
          if (!moved.allFinite())
            throw std::invalid_argument("Transformed XYZ is not representable as float32");
          for (int k = 0; k < 3; ++k) row[k] = moved[k];
        }
      }
      offset += count;
    }
  }
  return output;
}
}  // namespace

void bindMapGeometry(py::module_& module) {
  module.def("reconstruct_dense_submap", &reconstructDense, py::arg("clouds"),
             py::arg("world_poses"), py::arg("relative_poses"), py::arg("max_points"),
             "Rebuild bounded dense Nx7 observations in anchor coordinates; copy before releasing the GIL.");
}
