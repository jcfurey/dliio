// Unit tests for NanoGICP: alignment on well-conditioned geometry, the
// block-wise degeneracy gate on a planar (tunnel-like) target, and the
// small-cloud covariance guard.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "dlio/dlio.h"
#include "nano_gicp/nano_gicp.h"

namespace {

using Cloud = pcl::PointCloud<dlio::Point>;

dlio::Point makePoint(float x, float y, float z) {
  dlio::Point p;
  p.x = x; p.y = y; p.z = z;
  p.intensity = 0.f;
  p.reflectivity = 0.f;
  return p;
}

// Grid on the z=0 plane, `half` meters in each direction, `step` spacing.
Cloud::Ptr makePlane(float half, float step) {
  auto cloud = std::make_shared<Cloud>();
  for (float x = -half; x <= half; x += step) {
    for (float y = -half; y <= half; y += step) {
      cloud->push_back(makePoint(x, y, 0.f));
    }
  }
  return cloud;
}

// Three orthogonal planes meeting at the origin (a room corner): fully
// constrains all six degrees of freedom.
Cloud::Ptr makeCorner(float extent, float step) {
  auto cloud = std::make_shared<Cloud>();
  for (float a = step; a <= extent; a += step) {
    for (float b = step; b <= extent; b += step) {
      cloud->push_back(makePoint(a, b, 0.f));  // floor
      cloud->push_back(makePoint(0.f, a, b));  // wall x=0
      cloud->push_back(makePoint(a, 0.f, b));  // wall y=0
    }
  }
  return cloud;
}

// Corner geometry carrying a smooth spatial intensity/reflectivity ramp, so the
// photometric term has non-zero target gradients and actually engages.
Cloud::Ptr makeIntensityCorner(float extent, float step) {
  auto cloud = std::make_shared<Cloud>();
  auto add = [&](float x, float y, float z) {
    dlio::Point p;
    p.x = x; p.y = y; p.z = z;
    p.intensity = 100.f + 40.f * x + 25.f * y + 15.f * z;  // smooth ramp -> gradient
    p.reflectivity = p.intensity;
    cloud->push_back(p);
  };
  for (float a = step; a <= extent; a += step) {
    for (float b = step; b <= extent; b += step) {
      add(a, b, 0.f);  // floor
      add(0.f, a, b);  // wall x=0
      add(a, 0.f, b);  // wall y=0
    }
  }
  return cloud;
}

Cloud::Ptr transformCloud(const Cloud::ConstPtr& in, const Eigen::Matrix4f& T) {
  auto out = std::make_shared<Cloud>();
  pcl::transformPointCloud(*in, *out, T);
  return out;
}

nano_gicp::NanoGICP<dlio::Point, dlio::Point> makeGICP() {
  nano_gicp::NanoGICP<dlio::Point, dlio::Point> gicp;
  gicp.setMaxCorrespondenceDistance(1.0f);
  gicp.setMaximumIterations(32);
  gicp.setCorrespondenceRandomness(16);
  gicp.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
  // Most NanoGICP tests exercise the Lyrical degeneracy implementation. The
  // deployed integration default is deliberately off pending bag validation.
  gicp.setDegeneracyThreshRatio(0.005f);
  return gicp;
}

}  // namespace

TEST(NanoGICP, RecoversSmallTransformOnCornerGeometry) {
  auto target = makeCorner(1.0f, 0.05f);

  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 3>(0, 0) =
      Eigen::AngleAxisf(0.02f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);

  // source = T_true^-1 * target, so the recovered transform should be T_true
  auto source = transformCloud(target, T_true.inverse());

  auto gicp = makeGICP();
  gicp.setInputTarget(target);
  gicp.setInputSource(source);

  Cloud aligned;
  gicp.align(aligned);

  ASSERT_TRUE(gicp.hasConverged());
  EXPECT_EQ(gicp.lastDegenerateDirections(), 0);

  Eigen::Matrix4f T = gicp.getFinalTransformation();
  EXPECT_LT((T.block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm(), 0.02f);
  Eigen::AngleAxisf rot_err(
      Eigen::Matrix3f(T.block<3, 3>(0, 0).transpose() * T_true.block<3, 3>(0, 0)));
  EXPECT_LT(std::abs(rot_err.angle()), 0.02f);
}

TEST(NanoGICP, GatesUnobservableDirectionsOnPlanarGeometry) {
  // A single plane is the canonical degenerate target: in-plane translation
  // (2 dof) and rotation about the normal (1 dof) are unobservable -- the
  // tunnel scenario in miniature.
  auto target = makePlane(2.0f, 0.05f);

  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;  // large in-plane shift the registration cannot observe
  auto source = transformCloud(target, T_shift);

  auto gicp = makeGICP();
  gicp.setInputTarget(target);
  gicp.setInputSource(source);

  Cloud aligned;
  gicp.align(aligned);

  // The gate must flag the unobservable directions...
  EXPECT_GE(gicp.lastDegenerateDirections(), 2);
  // ...and hold the prior (identity guess) there rather than inventing motion.
  Eigen::Matrix4f T = gicp.getFinalTransformation();
  float translation_norm = T.block<3, 1>(0, 3).norm();
  EXPECT_LT(translation_norm, 0.05f);
}

TEST(NanoGICP, DegeneracyGateCanBeDisabled) {
  auto target = makePlane(1.0f, 0.05f);
  auto gicp = makeGICP();
  gicp.setDegeneracyThreshRatio(0.f);
  gicp.setInputTarget(target);
  gicp.setInputSource(target);

  Cloud aligned;
  gicp.align(aligned);
  EXPECT_EQ(gicp.lastDegenerateDirections(), 0);
}

TEST(NanoGICP, DegeneracyGateIsDisabledByIntegrationDefault) {
  auto target = makePlane(1.0f, 0.05f);
  nano_gicp::NanoGICP<dlio::Point, dlio::Point> gicp;
  gicp.setMaxCorrespondenceDistance(1.0f);
  gicp.setMaximumIterations(32);
  gicp.setCorrespondenceRandomness(16);
  gicp.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
  gicp.setInputTarget(target);
  gicp.setInputSource(target);

  Cloud aligned;
  gicp.align(aligned);
  EXPECT_EQ(gicp.lastDegenerateDirections(), 0);
  EXPECT_FLOAT_EQ(gicp.lastGeoRotMargin(), -1.0f);
  EXPECT_FLOAT_EQ(gicp.lastGeoTransMargin(), -1.0f);
}

// The held degenerate eigen-directions are EXPOSED (world-frame unit vectors)
// for the downstream governor + covariance inflation. On a single plane the
// in-plane translation dofs are held and must show up.
TEST(NanoGICP, ExposesHeldDegenerateDirs) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto gicp = makeGICP();
  gicp.setInputTarget(target);
  gicp.setInputSource(source);

  Cloud aligned;
  gicp.align(aligned);

  const auto& tdirs = gicp.lastDegenTransDirs();
  const auto& rdirs = gicp.lastDegenRotDirs();
  EXPECT_GE(tdirs.size(), 2u);                 // >= 2 in-plane translation dofs held
  for (const auto& d : tdirs) { EXPECT_NEAR(d.norm(), 1.0, 1e-6); }
  for (const auto& d : rdirs) { EXPECT_NEAR(d.norm(), 1.0, 1e-6); }
  // No rescue term here, so every held dir is exposed; the converged-iteration
  // count cannot exceed the max-over-iterations reported count.
  EXPECT_LE(static_cast<int>(tdirs.size() + rdirs.size()), gicp.lastDegenerateDirections());
  EXPECT_GE(static_cast<int>(tdirs.size() + rdirs.size()), 2);
}

// Gate off -> no held dirs exposed (mirrors lastDegenerateDirections == 0).
TEST(NanoGICP, DegenDirsEmptyWhenGateDisabled) {
  auto target = makePlane(1.0f, 0.05f);
  auto gicp = makeGICP();
  gicp.setDegeneracyThreshRatio(0.f);
  gicp.setInputTarget(target);
  gicp.setInputSource(target);

  Cloud aligned;
  gicp.align(aligned);
  EXPECT_TRUE(gicp.lastDegenTransDirs().empty());
  EXPECT_TRUE(gicp.lastDegenRotDirs().empty());
}

// Per-scan IMU-consistency clamp: bound the TOTAL correction (final pose vs the
// identity guess) so a large map-lock "jump" cannot run away. Well-conditioned
// corner geometry + gate OFF, so the solve genuinely wants the full transform;
// the clamp must cap it.
TEST(NanoGICP, MaxCorrectionClampBoundsTotalTranslation) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.20f, 0.0f, 0.0f);  // 0.20 m truth
  auto source = transformCloud(target, T_true.inverse());

  // Unclamped: confirm the registration drives a step larger than the cap.
  auto g0 = makeGICP();
  g0.setDegeneracyThreshRatio(0.f);
  g0.setInputTarget(target);
  g0.setInputSource(source);
  Cloud a0; g0.align(a0);
  const float unclamped = g0.getFinalTransformation().block<3, 1>(0, 3).norm();
  ASSERT_GT(unclamped, 0.05f);  // setup sanity: the clamp will actually bite

  // Clamped at 5 cm: identity guess, so the correction == final translation.
  auto g1 = makeGICP();
  g1.setDegeneracyThreshRatio(0.f);
  g1.setMaxCorrection(0.05f, 0.f);
  g1.setInputTarget(target);
  g1.setInputSource(source);
  Cloud a1; g1.align(a1);
  const float clamped = g1.getFinalTransformation().block<3, 1>(0, 3).norm();
  EXPECT_LT(clamped, 0.05f + 1e-4f);
}

TEST(NanoGICP, MaxCorrectionClampBoundsTotalRotation) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 3>(0, 0) =
      Eigen::AngleAxisf(0.30f, Eigen::Vector3f::UnitZ()).toRotationMatrix();  // 0.30 rad
  auto source = transformCloud(target, T_true.inverse());

  auto g0 = makeGICP();
  g0.setDegeneracyThreshRatio(0.f);
  g0.setInputTarget(target);
  g0.setInputSource(source);
  Cloud a0; g0.align(a0);
  Eigen::AngleAxisf aa0(Eigen::Matrix3f(g0.getFinalTransformation().block<3, 3>(0, 0)));
  ASSERT_GT(aa0.angle(), 0.10f);  // setup sanity

  auto g1 = makeGICP();
  g1.setDegeneracyThreshRatio(0.f);
  g1.setMaxCorrection(0.f, 0.10f);  // cap rotation at 0.10 rad
  g1.setInputTarget(target);
  g1.setInputSource(source);
  Cloud a1; g1.align(a1);
  Eigen::AngleAxisf aa1(Eigen::Matrix3f(g1.getFinalTransformation().block<3, 3>(0, 0)));
  EXPECT_LT(aa1.angle(), 0.10f + 1e-4f);
}

// Clamp OFF (the default, 0/0) must be bit-identical: the guard skips the clamp
// branch entirely, so the result equals the unclamped path exactly. A cap larger
// than the actual correction is a no-op too (within float recompose epsilon).
TEST(NanoGICP, MaxCorrectionDisabledIsBitIdentical) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto run = [&](float ct, float cr) {
    auto g = makeGICP();
    g.setMaxCorrection(ct, cr);
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  auto run_default = [&]() {
    auto g = makeGICP();  // never call setMaxCorrection -> default 0/0
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };

  const Eigen::Matrix4f base = run_default();
  EXPECT_TRUE(run(0.f, 0.f).isApprox(base, 0.f));      // explicit off: exact
  EXPECT_TRUE(run(10.f, 3.0f).isApprox(base, 1e-5f));  // cap >> correction: no-op
}

// --- Soft degeneracy gate keep-fraction (pure helper) ---
// The free function maps a Hessian eigenvalue to the fraction of the GICP update
// kept along that direction. softness 0 is the original binary gate; softness>0
// ramps smoothly across a log-eigenvalue band centred on the threshold.

TEST(SoftGate, BinaryWhenSoftnessZero) {
  // softness <= 0 reproduces the original binary gate exactly: trust strictly
  // above the threshold, hold (keep 0) at/below it -- matching `eig <= thresh`.
  EXPECT_EQ(nano_gicp::softGateKeepFraction(2.0, 1.0, 0.0), 1.0);
  EXPECT_EQ(nano_gicp::softGateKeepFraction(1.0, 1.0, 0.0), 0.0);   // == thresh -> hold
  EXPECT_EQ(nano_gicp::softGateKeepFraction(0.5, 1.0, 0.0), 0.0);
  EXPECT_EQ(nano_gicp::softGateKeepFraction(2.0, 1.0, -1.0), 1.0);  // negative == off
}

TEST(SoftGate, SmoothstepEdgesAndCentre) {
  const double thresh = 0.4, s = 0.5;
  // lower band edge thresh/(1+s): fully held; upper edge thresh*(1+s): fully kept
  EXPECT_DOUBLE_EQ(nano_gicp::softGateKeepFraction(thresh / (1.0 + s), thresh, s), 0.0);
  EXPECT_DOUBLE_EQ(nano_gicp::softGateKeepFraction(thresh * (1.0 + s), thresh, s), 1.0);
  // centre (== thresh) is the smoothstep midpoint, 0.5
  EXPECT_NEAR(nano_gicp::softGateKeepFraction(thresh, thresh, s), 0.5, 1e-12);
}

TEST(SoftGate, MonotoneAndClamped) {
  const double thresh = 1.0, s = 1.0;
  double prev = -1.0;
  for (double e = 0.1; e <= 5.0; e += 0.1) {
    const double k = nano_gicp::softGateKeepFraction(e, thresh, s);
    EXPECT_GE(k, 0.0);
    EXPECT_LE(k, 1.0);
    EXPECT_GE(k, prev - 1e-12);  // non-decreasing in eigenvalue
    prev = k;
  }
  EXPECT_DOUBLE_EQ(nano_gicp::softGateKeepFraction(1e-6, thresh, s), 0.0);  // well below band
  EXPECT_DOUBLE_EQ(nano_gicp::softGateKeepFraction(1e6, thresh, s), 1.0);   // well above band
}

TEST(SoftGate, NonPositiveInputsHoldPrior) {
  EXPECT_EQ(nano_gicp::softGateKeepFraction(-1.0, 1.0, 0.5), 0.0);  // eigval <= 0
  EXPECT_EQ(nano_gicp::softGateKeepFraction(0.0, 1.0, 0.5), 0.0);
  EXPECT_EQ(nano_gicp::softGateKeepFraction(1.0, 0.0, 0.5), 0.0);   // thresh <= 0
}

// Soft gate OFF (default 0) must take the exact binary-gate path: result equals
// the never-configured default bit-for-bit.
TEST(NanoGICP, SoftGateDisabledIsBitIdentical) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto run = [&](bool set) {
    auto g = makeGICP();
    if (set) { g.setDegeneracySoftness(0.f); }  // explicit off
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);          // never configured -> default 0
  EXPECT_TRUE(run(true).isApprox(base, 0.f));        // explicit 0: exact
}

// A wide soft band must NOT leak motion onto a *strongly* unobservable axis: the
// in-plane directions of a plane have eigenvalue ~0 << thresh, so keep stays ~0
// and the prior is held -- the soft ramp only affects near-threshold directions.
TEST(NanoGICP, SoftGateStillHoldsStronglyDegenerateAxis) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto g = makeGICP();
  g.setDegeneracySoftness(1.0f);  // wide band
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  const float translation_norm = g.getFinalTransformation().block<3, 1>(0, 3).norm();
  EXPECT_LT(translation_norm, 0.05f);
  EXPECT_GE(g.lastDegenerateDirections(), 2);  // still flags the unobservable dofs
}

// X-ICP ternary gate OFF (default) must be bit-identical to the existing gate:
// explicitly disabling it changes nothing vs. a never-configured instance.
TEST(NanoGICP, XicpTernaryDisabledIsBitIdentical) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto run = [&](bool set) {
    auto g = makeGICP();
    if (set) { g.setXicpTernary(false, 0.05f); }  // explicit off
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);          // never configured -> default off
  EXPECT_TRUE(run(true).isApprox(base, 0.f));        // explicit off: exact
}

// With the X-ICP ternary gate ON, a STRONGLY unobservable axis (in-plane on a
// single plane, eigenvalue ~0 << the partial bar) still gets xicpPartialScale 0,
// so the prior is fully held -- the partial-admit band only affects directions
// between the two bars, never a hard-degenerate one.
TEST(NanoGICP, XicpTernaryStillHoldsStronglyDegenerateAxis) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto g = makeGICP();
  g.setXicpTernary(true, 0.05f);  // ternary on; partial bar = degeneracyThreshRatio
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  const float translation_norm = g.getFinalTransformation().block<3, 1>(0, 3).norm();
  EXPECT_LT(translation_norm, 0.05f);          // in-plane shift still held
  EXPECT_GE(g.lastDegenerateDirections(), 2);  // unobservable dofs still flagged
}

// GenZ-ICP blend OFF (default / floor 1) is bit-identical to the existing solve.
TEST(NanoGICP, GenZDisabledIsBitIdentical) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto run = [&](bool set) {
    auto g = makeGICP();
    if (set) { g.setGenZWeighting(false, 0.5f, 0.1f, 1.0f); }  // explicit off
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);          // never configured -> default off
  EXPECT_TRUE(run(true).isApprox(base, 0.f));        // explicit off: exact
}

// GenZ-ICP blend exercised end-to-end on a degenerate (planar) scan: the
// translation block is ill-conditioned, so alpha drops below 1 and the
// point-to-point metric is mixed into the GICP cost in linearize(). Asserts the
// solve stays finite and bounded -- this is also the ASan/UBSan coverage for the
// linearize() blend edit (the hot loop the off-path tests never enter).
TEST(NanoGICP, GenZBlendRunsOnDegenerateScan) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto g = makeGICP();
  g.setGenZWeighting(true, 0.5f, 0.1f, 1.0f);  // blend engages on the ill-conditioned plane
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  const Eigen::Matrix4f T = g.getFinalTransformation();
  const float tnorm = T.block<3, 1>(0, 3).norm();  // hoisted: comma in <3,1> breaks the macro
  EXPECT_TRUE(T.allFinite());
  EXPECT_LT(tnorm, 1.0f);  // bounded; no divergence from the blend
}

// X-ICP partial-band recording for the governor (FINDINGS_2026-06-25 fix): with
// the ternary gate on, the gate still records the hard-degenerate in-plane axes
// for the governor and the new partial-band recording path runs without
// disturbing that or the solve. (Partial-band efficacy itself is validated by the
// empirical combo A/B; an eigenvalue in (thresh, fullRatio)*lmax is not
// constructible deterministically from a clean plane here.)
TEST(NanoGICP, XicpTernaryStillRecordsGovernorDirs) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);

  auto g = makeGICP();
  g.setXicpTernary(true, 0.05f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  EXPECT_GE(g.lastDegenTransDirs().size(), 2u);  // in-plane axes recorded for the governor
  const float tnorm = g.getFinalTransformation().block<3, 1>(0, 3).norm();
  EXPECT_LT(tnorm, 0.05f);                        // still held / bounded
}

// A fully-observable scan records NOTHING for the governor even with the ternary
// gate on -- the partial-band recording must not fire on localizable axes.
TEST(NanoGICP, XicpTernaryRecordsNothingOnObservableGeometry) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto g = makeGICP();
  g.setXicpTernary(true, 0.05f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  EXPECT_EQ(g.lastDegenerateDirections(), 0);
  EXPECT_TRUE(g.lastDegenTransDirs().empty());   // no false partial-band recording
  EXPECT_TRUE(g.lastDegenRotDirs().empty());
}

// Saliency weighting OFF (default / boost 1) is bit-identical to the existing
// solve, and skips the per-point saliency eigendecomposition entirely.
TEST(NanoGICP, SaliencyDisabledIsBitIdentical) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto run = [&](bool set) {
    auto g = makeGICP();
    if (set) { g.setSaliencyWeighting(false, 4.0f); }  // explicit off
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);
  EXPECT_TRUE(run(true).isApprox(base, 0.f));        // explicit off: exact
}

// With saliency weighting ON, a well-conditioned solve still converges accurately
// (up-weighting edge/corner points must not break correct registration).
TEST(NanoGICP, SaliencyOnStillRecoversTransform) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 3>(0, 0) =
      Eigen::AngleAxisf(0.02f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto g = makeGICP();
  g.setSaliencyWeighting(true, 4.0f);   // up-weight salient (edge/corner) points
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  ASSERT_TRUE(g.hasConverged());
  const Eigen::Matrix4f T = g.getFinalTransformation();
  EXPECT_LT((T.block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm(), 0.03f);
}

// --- Term mass-normalization (refCountScale + the photometric path) ---

TEST(RefCountScale, OffReturnsRawScale) {
  // refcount <= 0 -> 1.0 (normalization off, raw mass, bit-identical path).
  EXPECT_EQ(nano_gicp::refCountScale(0.0, 5000), 1.0);
  EXPECT_EQ(nano_gicp::refCountScale(-3.0, 5000), 1.0);
}

TEST(RefCountScale, NoResidualsReturnsZero) {
  // refcount > 0 but no points this scan -> 0 contribution.
  EXPECT_EQ(nano_gicp::refCountScale(1000.0, 0), 0.0);
  EXPECT_EQ(nano_gicp::refCountScale(1000.0, -1), 0.0);
}

TEST(RefCountScale, RatioOtherwiseAndLidarParity) {
  EXPECT_DOUBLE_EQ(nano_gicp::refCountScale(1000.0, 2000), 0.5);
  EXPECT_DOUBLE_EQ(nano_gicp::refCountScale(500.0, 100), 5.0);
  // refcount == count -> 1.0; ref=1000 reproduces the LiDAR-image kLidarRefCount.
  EXPECT_DOUBLE_EQ(nano_gicp::refCountScale(1000.0, 1000), 1.0);
}

TEST(RefCountScale, NormalizedMassIsCountIndependent) {
  // The whole point: a term whose RAW mass scales with the residual count has a
  // NORMALIZED mass that does not. mass(count) * scale(ref,count) == unit*ref.
  const double ref = 1000.0, unit_mass = 3.5;
  double prev = -1.0;
  for (long count = 10; count <= 100000; count *= 10) {
    const double raw_mass = unit_mass * static_cast<double>(count);  // mass ~ count
    const double normalized = raw_mass * nano_gicp::refCountScale(ref, count);
    EXPECT_NEAR(normalized, unit_mass * ref, 1e-9);   // count-free
    if (prev >= 0.0) { EXPECT_NEAR(normalized, prev, 1e-9); }
    prev = normalized;
  }
}

// Photometric normalization OFF (default 0) must take the exact raw shared-
// accumulator path: result equals the never-configured default bit-for-bit.
TEST(NanoGICP, PhotometricNormalizationDisabledIsBitIdentical) {
  auto target = makeIntensityCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto run = [&](bool set_zero) {
    auto g = makeGICP();
    // This test isolates setter-path identity. Independent multithreaded solves
    // may differ in floating-point reduction order even when they take the same
    // code path, so use a deterministic reduction for the exact comparison.
    g.setNumThreads(1);
    g.setPhotometricWeight(0.5f);                 // engage photometric (gradients on setInputTarget)
    if (set_zero) { g.setPhotometricRefCount(0.f); }
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);        // never configured -> default 0
  EXPECT_TRUE(run(true).isApprox(base, 0.f));      // explicit 0: exact
}

// The new per-scan photometric residual count is tracked (diagnostic + future
// adaptive weighting), and is 0 when the photometric term is off.
TEST(NanoGICP, PhotometricCountIsTracked) {
  auto target = makeIntensityCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.02f, -0.01f, 0.03f);
  auto source = transformCloud(target, T_true.inverse());

  auto g = makeGICP();
  g.setPhotometricWeight(0.5f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  EXPECT_GT(g.lastPhotometricCount(), 0);          // photometric engaged

  auto g0 = makeGICP();                            // no photometric weight
  g0.setInputTarget(target);
  g0.setInputSource(source);
  Cloud a0; g0.align(a0);
  EXPECT_EQ(g0.lastPhotometricCount(), 0);
}

// Normalization ON must keep the solve healthy (the separate-accumulator + scale
// fold-in path runs, no NaN, still converges on well-conditioned geometry).
TEST(NanoGICP, PhotometricNormalizationOnStillConverges) {
  auto target = makeIntensityCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto g = makeGICP();
  g.setPhotometricWeight(0.5f);
  g.setPhotometricRefCount(1000.f);                // normalization ON
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  ASSERT_TRUE(g.hasConverged());
  const Eigen::Matrix4f T = g.getFinalTransformation();
  const float err = (T.block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm();
  EXPECT_LT(err, 0.03f);
}

// Read-only telemetry: the photometric residual RMS (fit quality) is reported
// when the term engages, and 0 when it is off.
TEST(NanoGICP, PhotometricRmsTracked) {
  auto target = makeIntensityCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());
  // Break brightness constancy so the photometric residual stays non-zero at
  // geometric convergence (on perfectly consistent data RMS legitimately -> 0).
  for (auto& p : source->points) { p.intensity += 12.f; p.reflectivity += 12.f; }

  auto g = makeGICP();
  g.setPhotometricWeight(0.5f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  EXPECT_GT(g.lastPhotometricRms(), 0.0f);
  EXPECT_TRUE(std::isfinite(g.lastPhotometricRms()));

  auto g0 = makeGICP();                            // photometric off
  g0.setInputTarget(target);
  g0.setInputSource(source);
  Cloud a0; g0.align(a0);
  EXPECT_EQ(g0.lastPhotometricRms(), 0.0f);
}

// Read-only telemetry: the geometric trust margin (weakest-axis eigenvalue / gate
// threshold) is > 1 on well-conditioned corner geometry and < 1 along the
// degenerate in-plane translation of a single plane.
TEST(NanoGICP, GeoTrustMarginReflectsConditioning) {
  auto corner = makeCorner(1.0f, 0.05f);
  auto gc = makeGICP();
  gc.setInputTarget(corner);
  gc.setInputSource(corner);
  Cloud ac; gc.align(ac);
  EXPECT_GT(gc.lastGeoRotMargin(), 1.0f);          // fully observable -> above the gate
  EXPECT_GT(gc.lastGeoTransMargin(), 1.0f);

  auto plane = makePlane(2.0f, 0.05f);
  auto gp = makeGICP();
  gp.setInputTarget(plane);
  gp.setInputSource(plane);
  Cloud ap; gp.align(ap);
  EXPECT_GE(gp.lastGeoTransMargin(), 0.0f);         // computed (test helper enables the gate)
  EXPECT_LT(gp.lastGeoTransMargin(), 1.0f);         // weakest trans axis is degenerate
  EXPECT_LT(gp.lastGeoTransMargin(), gc.lastGeoTransMargin());
}

// --- Margin-adaptive clamp (clampScaleFromMargin + the clamp path) ---

TEST(ClampScale, HealthyMarginUsesBaseCap) {
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(5.0, 1.0, 3.0, 0.25), 1.0);   // >= hi
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(3.0, 1.0, 3.0, 0.25), 1.0);   // == hi
}

TEST(ClampScale, DegenerateMarginTightensToFloor) {
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(1.0, 1.0, 3.0, 0.25), 0.25);  // == lo
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(0.2, 1.0, 3.0, 0.25), 0.25);  // < lo
}

TEST(ClampScale, BlendsLinearlyBetween) {
  EXPECT_NEAR(nano_gicp::clampScaleFromMargin(2.0, 1.0, 3.0, 0.25), 0.625, 1e-12);  // s=0.5
  // monotone non-decreasing in margin
  double prev = -1.0;
  for (double m = 0.0; m <= 4.0; m += 0.25) {
    const double s = nano_gicp::clampScaleFromMargin(m, 1.0, 3.0, 0.25);
    EXPECT_GE(s, 0.25 - 1e-12);
    EXPECT_LE(s, 1.0 + 1e-12);
    EXPECT_GE(s, prev - 1e-12);
    prev = s;
  }
}

TEST(ClampScale, UnknownMarginDoesNotTighten) {
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(-1.0, 1.0, 3.0, 0.25), 1.0);  // gate off
}

TEST(ClampScale, DegenerateBandAndFloorClamp) {
  // hi <= lo -> step at margin_lo
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(2.0, 3.0, 3.0, 0.25), 0.25);
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(3.0, 3.0, 3.0, 0.25), 1.0);
  // floor clamped into [0,1]
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(1.0, 1.0, 3.0, -5.0), 0.0);
  EXPECT_DOUBLE_EQ(nano_gicp::clampScaleFromMargin(1.0, 1.0, 3.0,  5.0), 1.0);
}

// Adaptive clamp OFF (default) must be bit-identical to the plain base cap.
TEST(NanoGICP, AdaptiveClampDisabledIsBitIdentical) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.20f, 0.0f, 0.0f);
  auto source = transformCloud(target, T_true.inverse());

  auto run = [&](bool set_off) {
    auto g = makeGICP();
    g.setMaxCorrection(0.05f, 0.f);
    if (set_off) { g.setAdaptiveClamp(false, 1.f, 3.f, 0.25f); }
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);          // never configured -> default off
  EXPECT_TRUE(run(true).isApprox(base, 0.f));        // explicit off: exact
}

// Forcing the HEALTHY regime (margin thresholds well below the corner's high
// margin) -> scale 1 -> the base cap is used, identical to base-cap-only.
TEST(NanoGICP, AdaptiveClampHealthyRegimeUsesBaseCap) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.20f, 0.0f, 0.0f);
  auto source = transformCloud(target, T_true.inverse());

  auto run = [&](bool adaptive) {
    auto g = makeGICP();
    g.setMaxCorrection(0.05f, 0.f);
    if (adaptive) { g.setAdaptiveClamp(true, /*lo*/0.1f, /*hi*/0.2f, 0.25f); }  // corner margin >> hi
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation().block<3, 1>(0, 3).norm();
  };
  EXPECT_NEAR(run(true), run(false), 1e-5f);   // healthy -> no tightening
}

// Forcing the DEGENERATE regime (margin thresholds well ABOVE the corner margin)
// -> scale = floor -> the cap tightens to base*floor end-to-end.
TEST(NanoGICP, AdaptiveClampDegenerateRegimeTightensCap) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.20f, 0.0f, 0.0f);
  auto source = transformCloud(target, T_true.inverse());

  auto base_cap = makeGICP();
  base_cap.setMaxCorrection(0.05f, 0.f);
  base_cap.setInputTarget(target);
  base_cap.setInputSource(source);
  Cloud a0; base_cap.align(a0);
  const float base_norm = base_cap.getFinalTransformation().block<3, 1>(0, 3).norm();
  ASSERT_NEAR(base_norm, 0.05f, 5e-3f);   // base cap bites at 0.05

  auto adapt = makeGICP();
  adapt.setMaxCorrection(0.05f, 0.f);
  adapt.setAdaptiveClamp(true, /*lo*/1e6f, /*hi*/2e6f, /*floor*/0.25f);  // corner margin << lo
  adapt.setInputTarget(target);
  adapt.setInputSource(source);
  Cloud a1; adapt.align(a1);
  const float adapt_norm = adapt.getFinalTransformation().block<3, 1>(0, 3).norm();
  EXPECT_LT(adapt_norm, base_norm);                 // tighter than the base cap
  EXPECT_LT(adapt_norm, 0.05f * 0.25f + 1e-3f);     // ~ base * floor = 0.0125
}

// --- Probabilistic degeneracy gate (Hatleskog & Alexis RA-L 2024 adaptation) ---

TEST(ProbGate, NoModelOrSingularEdges) {
  EXPECT_DOUBLE_EQ(nano_gicp::probGateKeepFraction(1.0, 0.0, 1.0, 1.0), 1.0);   // no floor -> no attenuation
  EXPECT_DOUBLE_EQ(nano_gicp::probGateKeepFraction(-1.0, 1.0, 1.0, 1.0), 0.0);  // singular -> held
  EXPECT_DOUBLE_EQ(nano_gicp::probGateKeepFraction(0.0, 1.0, 1.0, 1.0), 0.0);
}

TEST(ProbGate, HalfAtSTimesFloor) {
  EXPECT_NEAR(nano_gicp::probGateKeepFraction(1.0, 1.0, 1.0, 1.0), 0.5, 1e-9);  // eigval = s*floor
  EXPECT_NEAR(nano_gicp::probGateKeepFraction(2.0, 1.0, 2.0, 1.0), 0.5, 1e-9);  // s scales the centre
}

TEST(ProbGate, SaturatesAndMonotone) {
  EXPECT_GT(nano_gicp::probGateKeepFraction(1e6, 1.0, 1.0, 1.0), 0.999);
  EXPECT_LT(nano_gicp::probGateKeepFraction(1e-6, 1.0, 1.0, 1.0), 0.001);
  double prev = -1.0;
  for (double e = 0.1; e <= 10.0; e += 0.1) {
    const double p = nano_gicp::probGateKeepFraction(e, 1.0, 1.0, 1.0);
    EXPECT_GE(p, 0.0); EXPECT_LE(p, 1.0);
    EXPECT_GE(p, prev - 1e-12);
    prev = p;
  }
}

TEST(ProbGate, SpreadZeroIsHardStep) {
  EXPECT_DOUBLE_EQ(nano_gicp::probGateKeepFraction(2.0, 1.0, 1.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(nano_gicp::probGateKeepFraction(0.5, 1.0, 1.0, 0.0), 0.0);
}

// Disabled (default) -> the ratio/smoothstep gate, bit-identical.
TEST(NanoGICP, ProbGateDisabledIsBitIdentical) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);
  auto run = [&](bool set_off) {
    auto g = makeGICP();
    if (set_off) { g.setProbabilisticGate(false, 0.f, 0.f, 1.f, 1.f); }
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);
  EXPECT_TRUE(run(true).isApprox(base, 0.f));
}

// floors=0 -> ratio threshold is the noise floor; the unobservable in-plane
// translation (eigenvalue << thresh) gets p~0 and is held, like the soft gate.
TEST(NanoGICP, ProbGateFallbackHoldsDegenerateAxis) {
  auto target = makePlane(2.0f, 0.05f);
  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;
  auto source = transformCloud(target, T_shift);
  auto g = makeGICP();
  g.setProbabilisticGate(true, 0.f, 0.f, 1.0f, 0.5f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  const float tnorm = g.getFinalTransformation().block<3, 1>(0, 3).norm();
  EXPECT_LT(tnorm, 0.05f);
  EXPECT_GE(g.lastDegenerateDirections(), 2);
}

// A sharp probit keeps observable directions ~fully -> a corner still recovers.
TEST(NanoGICP, ProbGatePreservesWellConditioned) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());
  auto g = makeGICP();
  g.setProbabilisticGate(true, 0.f, 0.f, 1.0f, /*sharp*/0.3f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  ASSERT_TRUE(g.hasConverged());
  const float terr = (g.getFinalTransformation().block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm();
  EXPECT_LT(terr, 0.03f);
}

// --- Barron adaptive robust kernel (CVPR 2019 + Chebrolu et al. RA-L 2021) ---

TEST(BarronKernel, RelWeightEdges) {
  // alpha = 2 -> L2, no down-weighting, for any r.
  EXPECT_DOUBLE_EQ(nano_gicp::barronRelWeight(0.5, 2.0, 0.05), 1.0);
  EXPECT_DOUBLE_EQ(nano_gicp::barronRelWeight(0.0, 2.0, 0.05), 1.0);
  // r = 0 -> weight 1 for any alpha.
  EXPECT_DOUBLE_EQ(nano_gicp::barronRelWeight(0.0, 0.5, 0.05), 1.0);
  // alpha = 0 at r = c: ((1)/2 + 1)^(-1) = 1/1.5.
  EXPECT_NEAR(nano_gicp::barronRelWeight(0.05, 0.0, 0.05), 1.0 / 1.5, 1e-9);
}

TEST(BarronKernel, RelWeightRedescends) {
  // For alpha < 2 the weight strictly decreases with |r| (and stays in (0,1]).
  const double w0 = nano_gicp::barronRelWeight(0.0, 1.0, 0.05);
  const double w1 = nano_gicp::barronRelWeight(0.05, 1.0, 0.05);
  const double w2 = nano_gicp::barronRelWeight(0.20, 1.0, 0.05);
  EXPECT_DOUBLE_EQ(w0, 1.0);
  EXPECT_LT(w1, w0);
  EXPECT_LT(w2, w1);
  EXPECT_GT(w2, 0.0);
}

TEST(BarronKernel, RhoSpecialForms) {
  EXPECT_NEAR(nano_gicp::barronRho(0.05, 2.0, 0.05), 0.5, 1e-9);              // L2: 0.5*(r/c)^2
  EXPECT_NEAR(nano_gicp::barronRho(0.05, 0.0, 0.05), std::log(1.5), 1e-9);   // log(0.5*(r/c)^2+1)
  EXPECT_NEAR(nano_gicp::barronRho(0.0, 1.0, 0.05), 0.0, 1e-12);             // r=0 -> 0
}

TEST(BarronKernel, LogPartitionEndpointsAndMonotone) {
  EXPECT_NEAR(nano_gicp::barronLogPartition(2.0), 0.5 * std::log(6.283185307179586), 1e-9);
  EXPECT_NEAR(nano_gicp::barronLogPartition(0.0), std::log(std::sqrt(2.0) * 3.14159265358979324), 1e-3);
  // Z shrinks as alpha grows -> logZ decreasing in alpha.
  EXPECT_GT(nano_gicp::barronLogPartition(0.5), nano_gicp::barronLogPartition(1.0));
  EXPECT_GT(nano_gicp::barronLogPartition(1.0), nano_gicp::barronLogPartition(2.0));
}

TEST(BarronKernel, FitAlphaCleanVsOutliers) {
  const double c = 0.05;
  std::vector<float> clean(200, 0.002f);                    // tiny residuals -> L2
  std::vector<float> outliers(200, 0.002f);
  for (int i = 0; i < 30; ++i) { outliers.push_back(0.4f); } // heavy outliers -> robust
  const double a_clean = nano_gicp::fitBarronAlpha(clean, c, 0.5, 2.0);
  const double a_out   = nano_gicp::fitBarronAlpha(outliers, c, 0.5, 2.0);
  EXPECT_GT(a_clean, 1.8);          // clean data -> near L2
  EXPECT_LT(a_out, a_clean);        // outliers pull the shape toward robust
  // empty -> alpha_hi
  EXPECT_DOUBLE_EQ(nano_gicp::fitBarronAlpha({}, c, 0.5, 2.0), 2.0);
}

TEST(BarronKernel, ScaleMadIsRobustSigma) {
  // 1.4826 * median|r|; robust to a few large outliers.
  std::vector<float> r(101, 0.02f);          // median|r| = 0.02
  for (int i = 0; i < 10; ++i) { r.push_back(5.0f); }   // outliers don't move the median
  EXPECT_NEAR(nano_gicp::barronScaleMad(r), 1.4826 * 0.02, 1e-6);
  // empty -> fallback; floored away from 0.
  EXPECT_GT(nano_gicp::barronScaleMad({}), 0.0);
  EXPECT_GE(nano_gicp::barronScaleMad(std::vector<float>(5, 0.0f)), 1e-4);
}

// Adaptive kernel OFF (default) -> the fixed Huber path, bit-identical.
TEST(NanoGICP, AdaptiveKernelDisabledIsBitIdentical) {
  auto target = makeIntensityCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto run = [&](bool set_off) {
    auto g = makeGICP();
    // See PhotometricNormalizationDisabledIsBitIdentical: exact equality is a
    // code-path assertion, not a cross-run OpenMP reproducibility assertion.
    g.setNumThreads(1);
    g.setPhotometricWeight(0.5f);
    if (set_off) { g.setAdaptiveKernel(false, 0.5f, 2.0f, 0.f); }
    g.setInputTarget(target);
    g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);
  EXPECT_TRUE(run(true).isApprox(base, 0.f));
}

// Adaptive kernel ON: the solve stays healthy and a sane alpha is reported.
TEST(NanoGICP, AdaptiveKernelConvergesAndReportsAlpha) {
  auto target = makeIntensityCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());

  auto g = makeGICP();
  g.setPhotometricWeight(0.5f);
  g.setAdaptiveKernel(true, 0.5f, 2.0f, 0.f);  // scale 0 -> reuse Huber delta
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  ASSERT_TRUE(g.hasConverged());
  const float terr = (g.getFinalTransformation().block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm();
  EXPECT_LT(terr, 0.03f);
  EXPECT_GT(g.lastKernelAlpha(), 0.0f);
  EXPECT_LE(g.lastKernelAlpha(), 2.0f);
}

TEST(NanoGICP, TinyCloudDoesNotCrashCovarianceEstimation) {
  // Fewer points than kCorrespondences: previously read uninitialized
  // kd-tree result slots (out-of-bounds indices).
  auto tiny = std::make_shared<Cloud>();
  tiny->push_back(makePoint(0.f, 0.f, 0.f));
  tiny->push_back(makePoint(0.1f, 0.f, 0.f));
  tiny->push_back(makePoint(0.f, 0.1f, 0.f));

  auto gicp = makeGICP();
  EXPECT_NO_THROW({
    gicp.setInputTarget(tiny);
    gicp.setInputSource(tiny);
  });
  EXPECT_EQ(gicp.getSourceCovariances().size(), tiny->size());
}

TEST(NanoGICP, NullAndEmptyCloudsDoNotReplaceValidInputs) {
  auto target = makeCorner(1.0f, 0.1f);
  auto source = transformCloud(target, Eigen::Matrix4f::Identity());
  auto gicp = makeGICP();
  gicp.setInputTarget(target);
  gicp.setInputSource(source);
  ASSERT_EQ(gicp.getSourceCovariances().size(), source->size());

  Cloud::ConstPtr null_cloud;
  auto empty_cloud = std::make_shared<Cloud>();
  EXPECT_NO_THROW(gicp.setInputSource(null_cloud));
  EXPECT_NO_THROW(gicp.setInputSource(empty_cloud));
  EXPECT_NO_THROW(gicp.setInputTarget(null_cloud));
  EXPECT_NO_THROW(gicp.setInputTarget(empty_cloud));
  EXPECT_NO_THROW(gicp.registerInputTarget(null_cloud));
  EXPECT_NO_THROW(gicp.registerInputTarget(empty_cloud));

  // Invalid updates leave the last valid source/target and their derived data
  // intact, matching PCL's empty-cloud behavior without dereferencing null.
  EXPECT_EQ(gicp.getSourceCovariances().size(), source->size());
  Cloud aligned;
  ASSERT_NO_THROW(gicp.align(aligned));
  EXPECT_TRUE(gicp.getFinalTransformation().allFinite());
}

TEST(NanoGICP, RejectsNonFiniteMahalanobisInverse) {
  auto target = makeCorner(1.0f, 0.1f);
  auto gicp = makeGICP();
  gicp.setInputTarget(target);
  gicp.setInputSource(target);

  Eigen::Matrix4f invalid_cov = Eigen::Matrix4f::Identity();
  invalid_cov(0, 0) = std::numeric_limits<float>::quiet_NaN();
  auto invalid_covs = std::make_shared<nano_gicp::CovarianceList>(
      target->size(), invalid_cov);
  gicp.setTargetCovariances(invalid_covs);

  Cloud aligned;
  ASSERT_NO_THROW(gicp.align(aligned));
  EXPECT_TRUE(gicp.getFinalTransformation().allFinite());
  EXPECT_TRUE(gicp.getFinalTransformation().isApprox(Eigen::Matrix4f::Identity(), 0.f));
}

// Covariance regularization (synthetic plane): PLANE forces the scale-free
// (1,1,1e-3) disc; MIN_EIG only clamps the smallest eigenvalue, leaving the two
// in-plane eigenvalues at the data variance. Locks the "honest regularization"
// behavior (these were silently identical before the PLANE/MIN_EIG split).
TEST(NanoGICP, PlaneRegularizationProducesUnitDisc) {
  auto cloud = makePlane(2.0f, 0.04f);   // z=0 plane
  nano_gicp::NanoGICP<dlio::Point, dlio::Point> gicp;
  gicp.setCorrespondenceRandomness(16);
  gicp.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
  gicp.setInputSource(cloud);

  const auto& covs = gicp.getSourceCovariances();
  ASSERT_EQ(covs.size(), cloud->size());
  int checked = 0;
  for (size_t i = 0; i < covs.size(); i += 37) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covs[i].block<3,3>(0,0));
    Eigen::Vector3f ev = es.eigenvalues();  // ascending
    EXPECT_NEAR(ev(2), 1.0f, 1e-4) << "i=" << i;   // two unit eigenvalues
    EXPECT_NEAR(ev(1), 1.0f, 1e-4) << "i=" << i;
    EXPECT_NEAR(ev(0), 1e-3f, 1e-4) << "i=" << i;  // flattened normal
    ++checked;
  }
  EXPECT_GT(checked, 5);
}

TEST(NanoGICP, MinEigRegularizationKeepsInPlaneVariance) {
  auto cloud = makePlane(2.0f, 0.04f);
  nano_gicp::NanoGICP<dlio::Point, dlio::Point> gicp;
  gicp.setCorrespondenceRandomness(16);
  gicp.setRegularizationMethod(nano_gicp::RegularizationMethod::MIN_EIG);
  gicp.setInputSource(cloud);

  const auto& covs = gicp.getSourceCovariances();
  ASSERT_EQ(covs.size(), cloud->size());
  bool any_non_unit_inplane = false;
  for (size_t i = 0; i < covs.size(); i += 37) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covs[i].block<3,3>(0,0));
    Eigen::Vector3f ev = es.eigenvalues();
    EXPECT_GE(ev(0), 1e-3f - 1e-5f);              // smallest clamped up to >= 1e-3
    if (std::abs(ev(2) - 1.0f) > 1e-2f) { any_non_unit_inplane = true; }
  }
  // unlike PLANE, the in-plane eigenvalues are the data variance, not 1.0
  EXPECT_TRUE(any_non_unit_inplane);
}

// --- Condition-scaled directional weighting (conditionScaleTerm) ---
//
// Reweights a map term toward geometrically-weak axes (judged from H_geo) so it
// can clear the degeneracy-gate rescue bar there. alpha_k =
// clamp((lambda_max/lambda_k)^power, 1, cap), S = V·diag(alpha)·Vᵀ per 3x3 block,
// *H = S·H·S, *b = S·b. Pure function -> tested directly.

namespace {
Eigen::Matrix<double, 6, 6> diagGeo(double a, double b, double c,
                                    double d, double e, double f) {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  H(0, 0) = a; H(1, 1) = b; H(2, 2) = c;
  H(3, 3) = d; H(4, 4) = e; H(5, 5) = f;
  return H;
}
double alphaOf(double lam, double lmax, double power, double cap) {
  double a = std::pow(lmax / lam, power);
  return std::min(std::max(a, 1.0), cap);
}
}  // namespace

// power = 0 -> every alpha = (lmax/lam)^0 = 1 -> S = I -> exact no-op.
TEST(CondScale, PowerZeroIsIdentity) {
  const auto Hgeo = diagGeo(100, 50, 1, 80, 2, 40);
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity() * 3.0;
  Eigen::Matrix<double, 6, 1> b; b << 1, 2, 3, 4, 5, 6;
  const auto H0 = H; const auto b0 = b;
  nano_gicp::conditionScaleTerm(Hgeo, 0.0, 50.0, &H, &b);
  EXPECT_TRUE(H.isApprox(H0, 0.0));
  EXPECT_TRUE(b.isApprox(b0, 0.0));
}

// cap = 1 -> alpha clamped to [1,1] = 1 -> exact no-op (regardless of power).
TEST(CondScale, CapOneIsIdentity) {
  const auto Hgeo = diagGeo(100, 50, 1, 80, 2, 40);
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity() * 3.0;
  Eigen::Matrix<double, 6, 1> b; b << 1, 2, 3, 4, 5, 6;
  const auto H0 = H; const auto b0 = b;
  nano_gicp::conditionScaleTerm(Hgeo, 2.0, 1.0, &H, &b);
  EXPECT_TRUE(H.isApprox(H0, 0.0));
  EXPECT_TRUE(b.isApprox(b0, 0.0));
}

// A block whose geometric Hessian is empty (lambda_max <= 0) is left untouched.
TEST(CondScale, EmptyGeoBlockIsIdentity) {
  // rotation block all-zero (no geometric rotation info), translation strong.
  const auto Hgeo = diagGeo(0, 0, 0, 100, 100, 100);
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Ones();
  nano_gicp::conditionScaleTerm(Hgeo, 1.0, 50.0, &H, &b);
  // rotation block unchanged (identity), translation strong -> alpha=1 -> unchanged.
  EXPECT_NEAR(H(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(H(1, 1), 1.0, 1e-12);
  EXPECT_NEAR(H(2, 2), 1.0, 1e-12);
  EXPECT_TRUE(b.isApprox(Eigen::Matrix<double, 6, 1>::Ones(), 1e-12));
}

// Diagonal H_geo -> eigenvectors are the axes -> S is diagonal diag(alpha) with
// each axis scaled by alpha of ITS OWN eigenvalue. H term = I -> H' = diag(alpha²),
// b ones -> b' = alpha. Exact, deterministic.
TEST(CondScale, DiagonalExactBoostAndStrongUnchanged) {
  const double p = 0.5, cap = 100.0;
  // rot eigenvalues {100,100,1} (weak axis 2); trans {1,50,50} (weak axis 0).
  const auto Hgeo = diagGeo(100, 100, 1, 1, 50, 50);
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Ones();
  nano_gicp::conditionScaleTerm(Hgeo, p, cap, &H, &b);

  const double a_rot_weak = alphaOf(1, 100, p, cap);   // sqrt(100) = 10
  const double a_tr_weak  = alphaOf(1, 50, p, cap);    // sqrt(50)  ~ 7.071
  EXPECT_NEAR(H(2, 2), a_rot_weak * a_rot_weak, 1e-9);  // weak rot stiffness x alpha²
  EXPECT_NEAR(b(2), a_rot_weak, 1e-9);                  // weak rot drive     x alpha
  EXPECT_NEAR(H(3, 3), a_tr_weak * a_tr_weak, 1e-9);    // weak trans
  EXPECT_NEAR(b(3), a_tr_weak, 1e-9);
  // strong axes (eigenvalue == lambda_max) -> alpha = 1 -> unchanged.
  EXPECT_NEAR(H(0, 0), 1.0, 1e-9);
  EXPECT_NEAR(b(0), 1.0, 1e-9);
  EXPECT_NEAR(H(4, 4), 1.0, 1e-9);
  EXPECT_NEAR(b(4), 1.0, 1e-9);
}

// The mechanism: a tiny isotropic-diluted term (like the measured ~2-16) clears
// the rescue bar on the weak rotation axis only after condition-scaling.
TEST(CondScale, WeakAxisClearsRescueBar) {
  // rot eigenvalues {1e6,1e6,1e3} -> lambda_max=1e6, thresh = 0.005*1e6 = 5000;
  // weak axis 2 has lambda=1e3 < thresh (degenerate). trans all strong.
  const auto Hgeo = diagGeo(1e6, 1e6, 1e3, 1e6, 1e6, 1e6);
  const double thresh = 0.005 * 1e6;
  const Eigen::Vector3d vweak(0, 0, 1);
  const double lam_geo_weak = 1e3;

  Eigen::Matrix<double, 6, 6> Hterm = Eigen::Matrix<double, 6, 6>::Zero();
  Hterm(2, 2) = 5.0;  // term supplies only ~5 on the weak axis (measured 2-16)
  Eigen::Matrix<double, 6, 1> bterm = Eigen::Matrix<double, 6, 1>::Zero();

  // Without cond-scale the gate's Rayleigh (lambda_geo + term) is below the bar.
  EXPECT_LT(lam_geo_weak + 5.0, thresh);

  nano_gicp::conditionScaleTerm(Hgeo, 0.5, 1000.0, &Hterm, &bterm);
  const double comb = lam_geo_weak +
                      vweak.dot(Hterm.block<3, 3>(0, 0) * vweak);
  EXPECT_GT(comb, thresh);  // cond-scale lifts it past the rescue bar
}

// Symmetry + PSD preserved for a coupled PSD term.
TEST(CondScale, PreservesSymmetryAndPSD) {
  const auto Hgeo = diagGeo(100, 10, 1, 80, 5, 1);
  Eigen::Matrix<double, 6, 6> M = Eigen::Matrix<double, 6, 6>::Random();
  Eigen::Matrix<double, 6, 6> H = M * M.transpose();  // symmetric PSD, with coupling
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Random();
  nano_gicp::conditionScaleTerm(Hgeo, 1.0, 50.0, &H, &b);
  EXPECT_LT((H - H.transpose()).norm(), 1e-9);  // symmetric
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(H);
  EXPECT_GT(es.eigenvalues()(0), -1e-9);        // PSD
}

// cap bounds the per-direction boost even for an extreme weakness/power.
TEST(CondScale, CapBoundsBoost) {
  const double cap = 8.0;
  const auto Hgeo = diagGeo(1e8, 1e8, 1.0, 1e8, 1e8, 1e8);  // huge ratio on rot axis 2
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Ones();
  nano_gicp::conditionScaleTerm(Hgeo, 4.0, cap, &H, &b);
  EXPECT_NEAR(H(2, 2), cap * cap, 1e-6);  // alpha clamped to cap -> stiffness cap²
  EXPECT_NEAR(b(2), cap, 1e-9);
}

// --- Direction-separated fusion (directionSeparateTerm) ---

// ratio <= 0 -> OFF: the term is returned untouched (bit-identical).
TEST(DirSep, RatioOffIsIdentity) {
  const auto Hgeo = diagGeo(100, 50, 1, 80, 2, 40);
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity() * 3.0;
  Eigen::Matrix<double, 6, 1> b; b << 1, 2, 3, 4, 5, 6;
  const auto H0 = H; const auto b0 = b;
  nano_gicp::directionSeparateTerm(Hgeo, 0.0, &H, &b);
  EXPECT_TRUE(H.isApprox(H0, 0.0));
  EXPECT_TRUE(b.isApprox(b0, 0.0));
}

// The term is projected onto the WEAK axis of each block; the strong-axis
// components are removed. rot {1,100,100} + trans {1,50,50}, ratio 0.05 -> axis 0
// of each block is weak (1 <= 0.05*lmax), the rest strong.
TEST(DirSep, KeepsWeakZeroesStrong) {
  const auto Hgeo = diagGeo(1, 100, 100, 1, 50, 50);
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Ones();
  nano_gicp::directionSeparateTerm(Hgeo, 0.05, &H, &b);
  EXPECT_NEAR(H(0, 0), 1.0, 1e-12); EXPECT_NEAR(H(1, 1), 0.0, 1e-12); EXPECT_NEAR(H(2, 2), 0.0, 1e-12);
  EXPECT_NEAR(H(3, 3), 1.0, 1e-12); EXPECT_NEAR(H(4, 4), 0.0, 1e-12); EXPECT_NEAR(H(5, 5), 0.0, 1e-12);
  Eigen::Matrix<double, 6, 1> bexp; bexp << 1, 0, 0, 1, 0, 0;
  EXPECT_TRUE(b.isApprox(bexp, 1e-12));
}

// A FULLY-OBSERVED block (no weak direction) is suppressed to zero -> a
// non-degenerate scan cannot be perturbed by the term.
TEST(DirSep, FullyObservedBlockSuppressed) {
  const auto Hgeo = diagGeo(100, 100, 100, 1, 50, 50);  // rot all strong, trans axis 0 weak
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Ones();
  nano_gicp::directionSeparateTerm(Hgeo, 0.05, &H, &b);
  EXPECT_NEAR(H(0, 0), 0.0, 1e-12); EXPECT_NEAR(H(1, 1), 0.0, 1e-12); EXPECT_NEAR(H(2, 2), 0.0, 1e-12);
  EXPECT_NEAR(H(3, 3), 1.0, 1e-12);  // trans weak axis kept
}

// An EMPTY block (no observability info, lambda_max <= 0) is left untouched.
TEST(DirSep, EmptyBlockUntouched) {
  const auto Hgeo = diagGeo(0, 0, 0, 1, 50, 50);  // rot block empty
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Ones();
  nano_gicp::directionSeparateTerm(Hgeo, 0.05, &H, &b);
  EXPECT_NEAR(H(0, 0), 1.0, 1e-12); EXPECT_NEAR(H(1, 1), 1.0, 1e-12); EXPECT_NEAR(H(2, 2), 1.0, 1e-12);
}

// Symmetry + PSD preserved for a coupled PSD term (P H P with P a projector).
TEST(DirSep, PreservesSymmetryAndPSD) {
  const auto Hgeo = diagGeo(100, 10, 1, 80, 5, 1);
  Eigen::Matrix<double, 6, 6> M = Eigen::Matrix<double, 6, 6>::Random();
  Eigen::Matrix<double, 6, 6> H = M * M.transpose();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Random();
  nano_gicp::directionSeparateTerm(Hgeo, 0.05, &H, &b);
  EXPECT_LT((H - H.transpose()).norm(), 1e-9);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(H);
  EXPECT_GT(es.eigenvalues()(0), -1e-9);
}

// --- Frame-to-frame LiDAR flow term (plumbing / safety guards) ---

// Flow term OFF (default weight 0) is bit-identical to the existing solve.
TEST(NanoGICP, LidarFlowDisabledIsBitIdentical) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());
  auto run = [&](bool set) {
    auto g = makeGICP();
    if (set) { g.setLidarFlowWeight(0.f); }   // explicit off
    g.setInputTarget(target); g.setInputSource(source);
    Cloud a; g.align(a);
    return g.getFinalTransformation();
  };
  const Eigen::Matrix4f base = run(false);
  EXPECT_TRUE(run(true).isApprox(base, 0.f));
}

// Flow ENABLED but no previous image set -> the accumulator's empty-image guard
// no-ops, so a well-conditioned solve is unaffected (and nothing crashes).
TEST(NanoGICP, LidarFlowNoPrevIsSafe) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());
  auto g = makeGICP();
  g.setLidarFlowWeight(1.0f);   // enabled, but setLidarFlowPrev never called -> empty prev
  g.setInputTarget(target); g.setInputSource(source);
  Cloud a; g.align(a);
  ASSERT_TRUE(g.hasConverged());
  EXPECT_LT((g.getFinalTransformation().block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm(), 0.03f);
}

// Exercise the flow accumulator's PARALLEL loop end-to-end in the DEFAULT
// image-to-image mode with a patch (prev + current images + spherical model set,
// weight > 0): ASan/UBSan/TSan coverage of the photometric loop, patch-window
// bilinearSample reads (both images), and the stack patch buffers. Assertions
// are loose (finite, bounded) -- correctness on real data is a bag question.
TEST(NanoGICP, LidarFlowAccumulatorRunsUnderSanitizers) {
  auto target = makeCorner(1.0f, 0.02f);
  auto source = makeCorner(1.0f, 0.02f);
  auto g = makeGICP();
  const int rows = 64, cols = 512;
  cv::Mat prev(rows, cols, CV_32FC1), cur(rows, cols, CV_32FC1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      prev.at<float>(r, c) = 0.001f * static_cast<float>((c * 7 + r * 3) % 97);  // textured
      cur.at<float>(r, c)  = 0.001f * static_cast<float>((c * 5 + r * 11) % 89);
    }
  const float az_a = 2.f * static_cast<float>(M_PI) / cols, az_b = -static_cast<float>(M_PI);
  const float el_a = 0.6f / rows, el_b = -0.3f;
  g.setLidarProjection(az_a, az_b, el_a, el_b);
  g.setLidarImage(cur);                              // reference image (image_ref mode)
  g.setLidarFlowPrev(prev, Eigen::Isometry3f::Identity());
  g.setLidarFlowMode(true, 2);                       // image-to-image, 5x5 patch
  g.setLidarFlowWeight(0.01f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  const Eigen::Matrix4f T = g.getFinalTransformation();
  const float tnorm = T.block<3, 1>(0, 3).norm();  // hoisted: comma in <3,1> breaks the macro
  EXPECT_TRUE(T.allFinite());                    // accumulator ran without NaN / crash
  EXPECT_LT(tnorm, 5.0f);                         // bounded
}

// Legacy point-field reference mode (imageRef false, patch 0): the ORIGINAL
// term's path -- reference = the source point's own .reflectivity -- must still
// run (no current image required) and stay finite. Guards the bit-identical
// fallback the imageRef param promises.
TEST(NanoGICP, LidarFlowLegacyFieldRefStillRuns) {
  auto target = makeCorner(1.0f, 0.02f);
  auto source = makeCorner(1.0f, 0.02f);
  auto g = makeGICP();
  const int rows = 64, cols = 512;
  cv::Mat prev(rows, cols, CV_32FC1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      prev.at<float>(r, c) = 0.001f * static_cast<float>((c * 7 + r * 3) % 97);
  const float az_a = 2.f * static_cast<float>(M_PI) / cols, az_b = -static_cast<float>(M_PI);
  const float el_a = 0.6f / rows, el_b = -0.3f;
  g.setLidarProjection(az_a, az_b, el_a, el_b);
  g.setLidarFlowPrev(prev, Eigen::Isometry3f::Identity());
  g.setLidarFlowMode(false, 0);                      // legacy: point-field reference
  g.setLidarFlowWeight(0.01f);
  g.setInputTarget(target);
  g.setInputSource(source);
  Cloud a; g.align(a);
  const Eigen::Matrix4f T = g.getFinalTransformation();
  const float tnorm = T.block<3, 1>(0, 3).norm();
  EXPECT_TRUE(T.allFinite());
  EXPECT_LT(tnorm, 5.0f);
}

// Image-ref mode with NO current image set must no-op (the guard), not crash:
// a well-conditioned solve is unaffected.
TEST(NanoGICP, LidarFlowImageRefWithoutImageIsSafe) {
  auto target = makeCorner(1.0f, 0.05f);
  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);
  auto source = transformCloud(target, T_true.inverse());
  auto g = makeGICP();
  const int rows = 32, cols = 256;
  cv::Mat prev(rows, cols, CV_32FC1, cv::Scalar(0.5f));
  g.setLidarProjection(2.f * static_cast<float>(M_PI) / cols, -static_cast<float>(M_PI),
                       0.6f / rows, -0.3f);
  g.setLidarFlowPrev(prev, Eigen::Isometry3f::Identity());
  g.setLidarFlowMode(true, 1);   // image_ref, but setLidarImage never called
  g.setLidarFlowWeight(1.0f);
  g.setInputTarget(target); g.setInputSource(source);
  Cloud a; g.align(a);
  ASSERT_TRUE(g.hasConverged());
  EXPECT_LT((g.getFinalTransformation().block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm(), 0.03f);
}

// A non-finite pose guess (the diverged deg=6 case that fed NaN queries to the
// kd-tree and produced garbage correspondences -> std::out_of_range,
// FINDINGS_2026-06-26) must not crash: update_correspondences skips non-finite
// queries and only stores in-range indices, so align() returns instead of
// throwing on the out-of-range target access.
TEST(NanoGICP, NonFinitePoseGuessDoesNotThrow) {
  auto target = makeCorner(1.0f, 0.05f);
  auto source = makeCorner(1.0f, 0.05f);
  auto g = makeGICP();
  g.setInputTarget(target);
  g.setInputSource(source);
  Eigen::Matrix4f bad = Eigen::Matrix4f::Identity();
  bad(0, 3) = std::numeric_limits<float>::quiet_NaN();   // NaN translation -> NaN queries
  Cloud aligned;
  EXPECT_NO_THROW(g.align(aligned, bad));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
