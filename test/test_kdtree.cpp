// Verification harness for the vendored nanoflann kd-tree
// (include/nano_gicp/nanoflann.h + nanoflann_adaptor.h) -- the correspondence
// search underneath update_correspondences and calculate_covariances.
//
// Pins as EXECUTABLE TESTS the fail-safe properties the VERIFICATION_2026-06-27
// source analysis of the FINDINGS_2026-06-26 crash relied on:
//   1. A non-finite (NaN/Inf) query returns found == 0 with the distance slot
//      reset to FLT_MAX -- so the caller's `k_sq_dists[0] < thresh` gate rejects
//      (the leaf loop admits only `dist < worst_dist`, false for NaN/Inf, and
//      KNNResultSet::init writes dists[capacity-1] = max on EVERY call).
//   2. Returned indices are always within [0, cloud size) -- the tree cannot
//      emit the out-of-range index observed in the crash (fuzzed).
//   3. Reused result vectors (the OMP `firstprivate` pattern in
//      update_correspondences) cannot leak a stale passing distance into a
//      failed search: init()'s FLT_MAX reset overwrites the k=1 slot each call.
//   4. Nearest-neighbour correctness against brute force (random cloud).
// Upstream context: koide3/fast_gicp and vectr-ucla DLIO use the identical
// unguarded `k_sq_dists[0] < thresh ? k_indices[0] : -1` pattern; jlblancoc/
// nanoflann documents that inputs are not sanitized -- these tests prove the
// combination still fails safe for this vendored version.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <set>

#include "dlio/dlio.h"
#include "nano_gicp/nanoflann_adaptor.h"

namespace {

using Cloud = pcl::PointCloud<dlio::Point>;

dlio::Point makePoint(float x, float y, float z) {
  dlio::Point p;
  p.x = x; p.y = y; p.z = z;
  p.intensity = 0.f;
  p.reflectivity = 0.f;
  return p;
}

// Deterministic random cloud in a [-10,10]^3 box.
Cloud::Ptr makeRandomCloud(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-10.f, 10.f);
  auto cloud = std::make_shared<Cloud>();
  for (int i = 0; i < n; ++i) { cloud->push_back(makePoint(u(rng), u(rng), u(rng))); }
  return cloud;
}

int bruteForceNN(const Cloud& cloud, const dlio::Point& q, float* sq_dist) {
  int best = -1;
  float best_d = std::numeric_limits<float>::max();
  for (size_t i = 0; i < cloud.size(); ++i) {
    const float dx = cloud[i].x - q.x, dy = cloud[i].y - q.y, dz = cloud[i].z - q.z;
    const float d = dx * dx + dy * dy + dz * dz;
    if (d < best_d) { best_d = d; best = static_cast<int>(i); }
  }
  if (sq_dist) { *sq_dist = best_d; }
  return best;
}

}  // namespace

// (1) NaN query: found == 0 and the distance slot is reset to FLT_MAX, so the
// correspondence gate `k_sq_dists[0] < max_dist_sq` rejects. THE load-bearing
// fail-safe of the crash analysis.
TEST(KdTree, NanQueryFailsSafe) {
  auto cloud = makeRandomCloud(500, 42);
  nanoflann::KdTreeFLANN<dlio::Point> tree(false);
  tree.setInputCloud(cloud);

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);
  dlio::Point q = makePoint(std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f);
  const int found = tree.nearestKSearch(q, 1, k_indices, k_sq_dists);
  EXPECT_EQ(found, 0);
  EXPECT_FALSE(k_sq_dists[0] < 1.0f);   // gate (dist < thresh) must reject
}

// (1b) Inf and huge-finite queries also fail the correspondence gate: Inf never
// enters the result set; a huge-finite query returns a genuine neighbour whose
// distance is astronomically above any correspondence threshold.
TEST(KdTree, InfAndHugeQueriesFailGate) {
  auto cloud = makeRandomCloud(500, 43);
  nanoflann::KdTreeFLANN<dlio::Point> tree(false);
  tree.setInputCloud(cloud);

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

  dlio::Point qi = makePoint(std::numeric_limits<float>::infinity(), 0.f, 0.f);
  int found = tree.nearestKSearch(qi, 1, k_indices, k_sq_dists);
  if (found > 0) {   // implementation may or may not admit an Inf-distance point
    EXPECT_FALSE(k_sq_dists[0] < 1.0f);
  } else {
    EXPECT_FALSE(k_sq_dists[0] < 1.0f);
  }

  dlio::Point qh = makePoint(1e18f, 1e18f, 1e18f);   // diverged-pose magnitude
  found = tree.nearestKSearch(qh, 1, k_indices, k_sq_dists);
  ASSERT_GT(found, 0);                                 // finite -> real neighbour
  EXPECT_GE(k_indices[0], 0);
  EXPECT_LT(k_indices[0], static_cast<int>(cloud->size()));
  EXPECT_FALSE(k_sq_dists[0] < 1.0f);                 // but the gate rejects it
}

// (2) Fuzz: across normal, boundary, denormal, and hostile queries, any index the
// tree returns is within [0, size). The tree cannot emit the crash's 74285787.
TEST(KdTree, IndicesAlwaysInRangeFuzz) {
  auto cloud = makeRandomCloud(1000, 44);
  nanoflann::KdTreeFLANN<dlio::Point> tree(false);
  tree.setInputCloud(cloud);

  std::mt19937 rng(45);
  std::uniform_real_distribution<float> u(-50.f, 50.f);
  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);
  const int n = static_cast<int>(cloud->size());

  for (int t = 0; t < 20000; ++t) {
    dlio::Point q = makePoint(u(rng), u(rng), u(rng));
    switch (t % 7) {   // inject hostile coordinates
      case 3: q.x = std::numeric_limits<float>::quiet_NaN(); break;
      case 4: q.y = std::numeric_limits<float>::infinity(); break;
      case 5: q.z = 1e30f; break;
      case 6: q.x = std::numeric_limits<float>::denorm_min(); break;
      default: break;
    }
    const int found = tree.nearestKSearch(q, 1, k_indices, k_sq_dists);
    if (found > 0) {
      ASSERT_GE(k_indices[0], 0) << "iteration " << t;
      ASSERT_LT(k_indices[0], n) << "iteration " << t;
    }
  }
}

// (3) The OMP firstprivate reuse pattern: a passing search followed by a NaN
// search on the SAME vectors must not leak the stale passing distance --
// KNNResultSet::init resets dists[capacity-1] (== [0] for k=1) every call.
TEST(KdTree, ReusedVectorsCannotLeakStaleDistance) {
  auto cloud = makeRandomCloud(500, 46);
  nanoflann::KdTreeFLANN<dlio::Point> tree(false);
  tree.setInputCloud(cloud);

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

  // First: a query ON a cloud point -> tiny distance stored in the reused slot.
  dlio::Point hit = (*cloud)[123];
  int found = tree.nearestKSearch(hit, 1, k_indices, k_sq_dists);
  ASSERT_GT(found, 0);
  ASSERT_LT(k_sq_dists[0], 1e-6f);   // stale value a naive reuse would leak

  // Then: a NaN query with the SAME vectors. The old code's gate consulted
  // k_sq_dists[0] without checking `found` -- this pins that the slot was reset.
  dlio::Point nanq = makePoint(std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f);
  found = tree.nearestKSearch(nanq, 1, k_indices, k_sq_dists);
  EXPECT_EQ(found, 0);
  EXPECT_FALSE(k_sq_dists[0] < 1.0f);   // reset to FLT_MAX, not the stale 1e-6
}

// (4) Correctness: kd-tree NN matches brute force on random queries.
TEST(KdTree, MatchesBruteForceNN) {
  auto cloud = makeRandomCloud(800, 47);
  nanoflann::KdTreeFLANN<dlio::Point> tree(false);
  tree.setInputCloud(cloud);

  std::mt19937 rng(48);
  std::uniform_real_distribution<float> u(-12.f, 12.f);
  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

  for (int t = 0; t < 500; ++t) {
    dlio::Point q = makePoint(u(rng), u(rng), u(rng));
    const int found = tree.nearestKSearch(q, 1, k_indices, k_sq_dists);
    ASSERT_EQ(found, 1);
    float bf_d = 0.f;
    const int bf = bruteForceNN(*cloud, q, &bf_d);
    EXPECT_EQ(k_indices[0], bf) << "query " << t;
    EXPECT_NEAR(k_sq_dists[0], bf_d, 1e-3f * (1.f + bf_d));
  }
}

// (5) k larger than the cloud: count == cloud size, all indices valid and unique.
TEST(KdTree, KLargerThanCloud) {
  auto cloud = makeRandomCloud(5, 49);
  nanoflann::KdTreeFLANN<dlio::Point> tree(false);
  tree.setInputCloud(cloud);

  std::vector<int> k_indices(8);
  std::vector<float> k_sq_dists(8);
  dlio::Point q = makePoint(0.f, 0.f, 0.f);
  const int found = tree.nearestKSearch(q, 8, k_indices, k_sq_dists);
  EXPECT_EQ(found, 5);
  std::set<int> seen;
  for (int i = 0; i < found; ++i) {
    EXPECT_GE(k_indices[i], 0);
    EXPECT_LT(k_indices[i], 5);
    seen.insert(k_indices[i]);
  }
  EXPECT_EQ(static_cast<int>(seen.size()), found);   // unique
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
