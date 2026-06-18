// Unit tests for NanoGICP: alignment on well-conditioned geometry, the
// block-wise degeneracy gate on a planar (tunnel-like) target, and the
// small-cloud covariance guard.

#include <gtest/gtest.h>

#include <cmath>

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
  EXPECT_GE(gp.lastGeoTransMargin(), 0.0f);         // computed (gate on by default)
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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
