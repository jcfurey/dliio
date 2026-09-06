#include "nano_gicp/surface_texture.h"
#include "nano_gicp/nanoflann_adaptor.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace nano_gicp {
namespace {
using Tree = nanoflann::KdTreeFLANN<pcl::PointXYZI>;
constexpr int kSamples = 21;
using Samples = Eigen::Matrix<float, kSamples, 1>;

float median(std::vector<float> values) {
  const auto mid = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), mid, values.end());
  return *mid;
}

bool normalize(Samples& values, float min_std) {
  values.array() -= values.mean();
  const float norm = values.norm();
  if (!std::isfinite(norm) || norm < min_std * std::sqrt(float(kSamples))) { return false; }
  values /= norm;
  return true;
}

// Smooth interpolation with a hard spatial support bound. Missing returns are
// never treated as dark paint. Nearby, separated surfaces cannot supply support
// across a depth gap; the caller additionally requires a planar reference patch.
bool sample(const TextureCloud& cloud, const Tree& tree, const Eigen::Vector3f& x,
            float radius, float& value) {
  pcl::PointXYZI query;
  query.getVector3fMap() = x;
  std::vector<int> ids(4);
  std::vector<float> distances(4);
  const int found = tree.nearestKSearch(query, 4, ids, distances);
  if (found < 3 || distances[2] > radius * radius) { return false; }
  float sum = 0.f, weight_sum = 0.f;
  for (int i = 0; i < found; ++i) {
    if (distances[i] > radius * radius) { continue; }
    const float intensity = cloud[ids[i]].intensity;
    if (!std::isfinite(intensity)) { return false; }
    const float weight = std::exp(-2.f * distances[i] / (radius * radius));
    sum += weight * intensity;
    weight_sum += weight;
  }
  value = sum / weight_sum;
  return std::isfinite(value);
}
}  // namespace

SurfaceTextureMatch matchSurfaceTexture(
    const TextureCloud::ConstPtr& source, const TextureCloud::ConstPtr& reference,
    const Eigen::Isometry3f& source_to_reference, const Eigen::Vector3f& direction,
    const SurfaceTextureConfig& config) {
  SurfaceTextureMatch result;
  if (!source || !reference || source->size() < 32 || reference->size() < 32 ||
      !source_to_reference.matrix().allFinite() || !direction.allFinite() ||
      direction.norm() < 0.5f || !(config.search_radius >= 0.12f) ||
      !(config.sample_radius > 0.f) || !(config.min_std > 0.f) ||
      config.max_patches < 1 || config.min_patches < 3) { return result; }
  // Bounds prevent accidental pathological work from non-ROS API callers too.
  if (config.search_radius > 2.f || config.max_patches > 1000) { return result; }
  const Eigen::Vector3f axis = direction.normalized();
  const Eigen::Isometry3f inverse = source_to_reference.inverse();
  Tree source_tree(false), reference_tree(false);
  source_tree.setInputCloud(source);
  reference_tree.setInputCloud(reference);
  constexpr int half_steps = 10;
  const float step = config.search_radius / half_steps;
  std::vector<Eigen::Vector3f> centers;
  std::vector<float> shifts, correlations;
  const size_t stride = std::max(size_t(1), reference->size() / (config.max_patches * 12));

  for (size_t index = 0; index < reference->size(); index += stride) {
    if (result.candidates >= config.max_patches) { break; }
    const auto& anchor = (*reference)[index];
    const Eigen::Vector3f center = anchor.getVector3fMap();
    if (!center.allFinite()) { continue; }
    bool near = false;
    for (const auto& c : centers) {
      if ((center - c).squaredNorm() < 0.36f) { near = true; break; }
    }
    if (near) { continue; }

    std::vector<int> ids(20);
    std::vector<float> distances(20);
    const int found = reference_tree.nearestKSearch(anchor, 20, ids, distances);
    if (found < 12 || distances.back() > 0.25f) { continue; }
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (int id : ids) { mean += (*reference)[id].getVector3fMap() - center; }
    mean /= found;
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int id : ids) {
      const Eigen::Vector3f d = (*reference)[id].getVector3fMap() - center - mean;
      covariance.noalias() += d * d.transpose() / found;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(covariance);
    if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite() ||
        eig.eigenvalues()(1) < 1e-4f ||
        eig.eigenvalues()(0) > 0.08f * eig.eigenvalues()(1)) { continue; }
    const Eigen::Vector3f normal = eig.eigenvectors().col(0);
    if (std::abs(normal.dot(axis)) > 0.25f) { continue; }
    const Eigen::Vector3f along = (axis - normal * normal.dot(axis)).normalized();
    const Eigen::Vector3f across = normal.cross(along);
    std::array<Eigen::Vector3f, kSamples> positions;
    Samples ref;
    bool valid = true;
    int k = 0;
    for (int row = -1; row <= 1; ++row) {
      for (int col = -3; col <= 3; ++col, ++k) {
        // Stagger rows along the search axis. Collinear sample columns can
        // land on every zero crossing of stripes and correlate interpolation
        // noise instead of the material's brightness pattern.
        positions[k] = center + (0.12f * col + 0.04f * row) * along + 0.18f * row * across;
        valid &= sample(*reference, reference_tree, positions[k], config.sample_radius, ref(k));
      }
    }
    if (!valid || !normalize(ref, config.min_std)) { continue; }
    ++result.candidates;
    centers.push_back(center);

    // Also test the reference's own repetition. A sampled periodic pattern
    // can acquire a spuriously unique cross-frame peak from return spacing
    // alone. Sampling artifacts are not a material landmark.
    bool repeated = false;
    for (int s = -half_steps; s <= half_steps && !repeated; ++s) {
      if (std::abs(s) * step < 0.12f) { continue; }
      Samples shifted;
      bool supported = true;
      for (int j = 0; j < kSamples; ++j) {
        if (!sample(*reference, reference_tree, positions[j] - s * step * along,
                    config.sample_radius, shifted(j))) { supported = false; break; }
      }
      if (supported && normalize(shifted, config.min_std) && ref.dot(shifted) > 0.95f) {
        repeated = true;
      }
    }
    if (repeated) { continue; }

    std::array<float, 2 * half_steps + 1> scores;
    for (int s = -half_steps; s <= half_steps && valid; ++s) {
      Samples cur;
      for (int j = 0; j < kSamples; ++j) {
        // A positive candidate shifts SOURCE points toward +axis, so sample
        // the unshifted source at reference - candidate (not + candidate).
        const Eigen::Vector3f q = inverse * (positions[j] - s * step * axis);
        if (!sample(*source, source_tree, q, config.sample_radius, cur(j))) {
          valid = false; break;
        }
      }
      if (!valid || !normalize(cur, config.min_std)) { valid = false; break; }
      scores[s + half_steps] = ref.dot(cur);
    }
    // Every candidate uses the identical, fully supported patch. Otherwise
    // pose-dependent missing samples can manufacture a correlation peak.
    if (!valid) { continue; }
    ++result.supported;
    const int best = std::max_element(scores.begin(), scores.end()) - scores.begin();
    if (best == 0 || best == 2 * half_steps || scores[best] < 0.7f) { continue; }
    float second = -1.f;
    for (int j = 0; j <= 2 * half_steps; ++j) {
      if (std::abs(j - best) * step >= 0.12f) { second = std::max(second, scores[j]); }
    }
    if (scores[best] - second < 0.04f) { continue; }
    const float curvature = 2.f * scores[best] - scores[best - 1] - scores[best + 1];
    if (!(curvature > 1e-4f)) { continue; }
    const float substep = std::clamp(0.5f * (scores[best + 1] - scores[best - 1]) / curvature, -0.5f, 0.5f);
    shifts.push_back((best - half_steps + substep) * step);
    correlations.push_back(scores[best]);
  }
  result.unique = shifts.size();
  if (result.unique < config.min_patches) { return result; }
  const float location = median(shifts);
  std::vector<float> errors;
  for (float shift : shifts) { errors.push_back(std::abs(shift - location)); }
  const float sigma = 1.4826f * median(errors);
  // Conflicting patch motions are not a confident translation measurement.
  if (sigma > 0.08f) { return result; }
  std::vector<float> inliers;
  float correlation = 0.f;
  for (size_t i = 0; i < shifts.size(); ++i) {
    if (errors[i] <= std::max(0.06f, 2.5f * sigma)) {
      inliers.push_back(shifts[i]); correlation += correlations[i];
    }
  }
  result.inliers = inliers.size();
  if (result.inliers < config.min_patches || result.inliers < 0.6f * result.unique) { return result; }
  result.valid = true;
  result.shift = median(inliers);
  result.sigma = std::max(0.03f, sigma); // correlated samples: no sqrt(N) precision claim
  result.correlation = correlation / result.inliers;
  return result;
}

void SurfaceTextureConstraint::accumulate(
    const Eigen::Isometry3f& transform, Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b, double* cost) const {
  if (!(information > 0.0)) { return; }
  const Eigen::Vector3d point = (transform * center).cast<double>();
  const Eigen::Vector3d direction = axis.cast<double>();
  const double residual = direction.dot(point) - position;
  Eigen::Matrix<double, 1, 6> J;
  J.head<3>() = point.cross(direction).transpose();
  J.tail<3>() = direction.transpose();
  H->noalias() += information * J.transpose() * J;
  b->noalias() += information * J.transpose() * residual;
  if (cost) { *cost += information * residual * residual; }
}
}  // namespace nano_gicp
