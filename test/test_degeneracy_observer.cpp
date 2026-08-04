// Unit tests for the LODESTAR-flavored observer attenuation
// (dlio::attenuateAlongHeldAxes): on the held-degenerate axes, the observer's
// correction is scaled toward the IMU prior; off-axis and disabled cases are
// untouched. Mirrors test_degeneracy_governor (pure geometry, no node state).

#include <gtest/gtest.h>
#include "dlio/degeneracy_observer.h"

#include <vector>
#include <Eigen/Core>

using dlio::attenuateAlongHeldAxes;

namespace {
const std::vector<Eigen::Vector3d> kNoDirs{};
const std::vector<Eigen::Vector3d> kX{Eigen::Vector3d::UnitX()};
}  // namespace

// gain == 1 -> OFF: correction returned unchanged even with held dirs.
TEST(DegeneracyObserver, GainOneIsNoOp) {
  const Eigen::Vector3f e(3.f, -2.f, 1.f);
  EXPECT_TRUE(attenuateAlongHeldAxes(e, kX, 1.0f).isApprox(e));
  EXPECT_TRUE(attenuateAlongHeldAxes(e, kX, 1.5f).isApprox(e));   // >1 clamped to off
}

// No held dirs -> unchanged (a non-degenerate scan is untouched).
TEST(DegeneracyObserver, NoDirsIsNoOp) {
  const Eigen::Vector3f e(3.f, -2.f, 1.f);
  EXPECT_TRUE(attenuateAlongHeldAxes(e, kNoDirs, 0.0f).isApprox(e));
}

// gain == 0 -> the along-axis component is fully removed (axis "fixed"), the
// orthogonal components are preserved exactly.
TEST(DegeneracyObserver, ZeroGainFreezesHeldAxis) {
  const Eigen::Vector3f e(3.f, -2.f, 1.f);
  const Eigen::Vector3f out = attenuateAlongHeldAxes(e, kX, 0.0f);
  EXPECT_NEAR(out.x(), 0.0f, 1e-6f);   // x (held) removed
  EXPECT_NEAR(out.y(), -2.0f, 1e-6f);  // y preserved
  EXPECT_NEAR(out.z(), 1.0f, 1e-6f);   // z preserved
}

// 0 < gain < 1 -> the along-axis component is scaled to gain*comp.
TEST(DegeneracyObserver, PartialGainScalesHeldComponent) {
  const Eigen::Vector3f e(4.f, 5.f, 0.f);
  const Eigen::Vector3f out = attenuateAlongHeldAxes(e, kX, 0.25f);
  EXPECT_NEAR(out.x(), 1.0f, 1e-6f);   // 0.25 * 4
  EXPECT_NEAR(out.y(), 5.0f, 1e-6f);   // untouched
}

// Oblique held axis: only the along-axis component is attenuated; the orthogonal
// part is preserved exactly (the real gate's eigenvectors are skew).
TEST(DegeneracyObserver, AttenuatesAlongObliqueAxis) {
  const Eigen::Vector3d d = Eigen::Vector3d(1.0, 1.0, 0.0).normalized();
  const Eigen::Vector3d ortho = Eigen::Vector3d(1.0, -1.0, 0.0).normalized();
  const Eigen::Vector3f e = (10.0 * d + 3.0 * ortho).cast<float>();
  const Eigen::Vector3f out = attenuateAlongHeldAxes(e, {d}, 0.0f);
  EXPECT_NEAR(d.cast<float>().dot(out), 0.0f, 1e-4f);     // along-axis frozen
  EXPECT_NEAR(ortho.cast<float>().dot(out), 3.0f, 1e-4f); // orthogonal preserved
}

// Two held orthonormal axes are attenuated independently; the unheld axis stays.
TEST(DegeneracyObserver, MultipleHeldAxesIndependent) {
  const std::vector<Eigen::Vector3d> dirs{Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY()};
  const Eigen::Vector3f e(4.f, 6.f, 9.f);
  const Eigen::Vector3f out = attenuateAlongHeldAxes(e, dirs, 0.0f);
  EXPECT_NEAR(out.x(), 0.0f, 1e-6f);
  EXPECT_NEAR(out.y(), 0.0f, 1e-6f);
  EXPECT_NEAR(out.z(), 9.0f, 1e-6f);   // z untouched
}

// Non-unit / degenerate direction entries are renormalized / skipped, not UB.
TEST(DegeneracyObserver, DirNormalizationAndZeroGuard) {
  const std::vector<Eigen::Vector3d> dirs{Eigen::Vector3d(5.0, 0.0, 0.0),  // non-unit
                                          Eigen::Vector3d(0.0, 0.0, 0.0)}; // zero -> skipped
  const Eigen::Vector3f e(2.f, 7.f, 0.f);
  const Eigen::Vector3f out = attenuateAlongHeldAxes(e, dirs, 0.0f);
  EXPECT_NEAR(out.x(), 0.0f, 1e-6f);   // renormalized x axis still freezes x
  EXPECT_NEAR(out.y(), 7.0f, 1e-6f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
