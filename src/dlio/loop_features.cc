// Bounded PCL feature extraction for loop proposals, never factor admission.
// PCL is already a required DLIO dependency. Keep the native descriptor here
// instead of carrying a second, subtly different Python FPFH implementation.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

Eigen::MatrixXf loopFeatures(const Eigen::MatrixXd& points, double normalRadius, double featureRadius) {
  if (points.cols() != 3 || points.rows() < 100 || points.rows() > 50000 || !points.allFinite() ||
      (points.array().abs() > 1.e6).any() ||
      !std::isfinite(normalRadius) || !std::isfinite(featureRadius) ||
      normalRadius < .1 || featureRadius <= normalRadius || featureRadius > 5.)
    throw std::invalid_argument("FPFH needs 100..50000 finite XYZ points and bounded increasing radii");
  Eigen::MatrixXf result(points.rows(), 33);
  {
    py::gil_scoped_release release;
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->resize(points.rows());
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
      (*cloud)[i].x = points(i, 0);
      (*cloud)[i].y = points(i, 1);
      (*cloud)[i].z = points(i, 2);
    }
    auto normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> estimate;
    estimate.setInputCloud(cloud);
    estimate.setRadiusSearch(normalRadius);
    estimate.setViewPoint(0.f, 0.f, 0.f);
    estimate.compute(*normals);
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> features;
    features.setInputCloud(cloud);
    features.setInputNormals(normals);
    features.setRadiusSearch(featureRadius);
    pcl::PointCloud<pcl::FPFHSignature33> descriptors;
    features.compute(descriptors);
    for (Eigen::Index i = 0; i < points.rows(); ++i)
      for (int j = 0; j < 33; ++j) result(i, j) = descriptors[i].histogram[j];
  }
  return result;
}

void bindLoopFeatures(py::module_& module) {
  module.def("fpfh", &loopFeatures, py::arg("points"), py::arg("normal_radius") = .6,
             py::arg("feature_radius") = 1.2);
}
