// Unit tests for the X-ICP ternary localizability kernel
// (nano_gicp::xicpCategory / xicpPartialScale).
//
// These pin the three-way classification (localizable / partial / non-localizable)
// against the two thresholds and the controlled-update scale across the partial
// band, plus the reductions: equal thresholds collapse to the existing gate's
// binary behavior, and the endpoints match a hard prior-hold / full-trust.

#include <gtest/gtest.h>
#include "nano_gicp/xicp_localizability.h"

using nano_gicp::Localizability;
using nano_gicp::xicpCategory;
using nano_gicp::xicpPartialScale;

// (1) Ternary classification against the two thresholds.
TEST(XIcpLocalizability, ClassifiesThreeWays) {
  const double kp = 1.0, kf = 4.0;   // partial bar, full bar
  EXPECT_EQ(xicpCategory(5.0, kp, kf), Localizability::Localizable);     // >= full
  EXPECT_EQ(xicpCategory(4.0, kp, kf), Localizability::Localizable);     // == full (inclusive)
  EXPECT_EQ(xicpCategory(2.5, kp, kf), Localizability::Partial);         // in band
  EXPECT_EQ(xicpCategory(1.0, kp, kf), Localizability::Partial);         // == partial (inclusive)
  EXPECT_EQ(xicpCategory(0.5, kp, kf), Localizability::NonLocalizable);  // < partial
}

// (2) The controlled-update scale: 1 above full, 0 below partial, linear between.
TEST(XIcpLocalizability, PartialScaleRampsLinearly) {
  const double kp = 1.0, kf = 3.0;   // band width 2
  EXPECT_DOUBLE_EQ(xicpPartialScale(3.0, kp, kf), 1.0);   // at/above full -> full admit
  EXPECT_DOUBLE_EQ(xicpPartialScale(5.0, kp, kf), 1.0);
  EXPECT_DOUBLE_EQ(xicpPartialScale(1.0, kp, kf), 0.0);   // at/below partial -> hold prior
  EXPECT_DOUBLE_EQ(xicpPartialScale(0.2, kp, kf), 0.0);
  EXPECT_NEAR(xicpPartialScale(2.0, kp, kf), 0.5, 1e-12); // midpoint of the band
  EXPECT_NEAR(xicpPartialScale(1.5, kp, kf), 0.25, 1e-12);
}

// (3) Equal thresholds (no partial band) reduce to the binary gate: a hard step
// at kappa_full, matching the existing Zhang solution-remapping behavior.
TEST(XIcpLocalizability, EqualThresholdsAreBinary) {
  const double k = 2.0;
  EXPECT_EQ(xicpCategory(2.0, k, k), Localizability::Localizable);
  EXPECT_EQ(xicpCategory(1.99, k, k), Localizability::NonLocalizable);
  EXPECT_DOUBLE_EQ(xicpPartialScale(2.0, k, k), 1.0);    // full trust at/above
  EXPECT_DOUBLE_EQ(xicpPartialScale(1.99, k, k), 0.0);   // full hold below
}

// (4) Mis-ordered thresholds (partial > full) are sanitized, not UB: the
// classifier clamps the partial bar down to the full bar (-> binary at full).
TEST(XIcpLocalizability, MisorderedThresholdsAreSafe) {
  EXPECT_EQ(xicpCategory(3.0, 5.0, 2.0), Localizability::Localizable);     // >= full(2)
  EXPECT_EQ(xicpCategory(1.0, 5.0, 2.0), Localizability::NonLocalizable);  // < full(2)
  EXPECT_DOUBLE_EQ(xicpPartialScale(3.0, 5.0, 2.0), 1.0);
  EXPECT_DOUBLE_EQ(xicpPartialScale(1.0, 5.0, 2.0), 0.0);
}

// --- Budgeted partial-band admission (xicpBudgetedAdmit, 2026-07-08 fix) ---

// cap <= 0 -> unbudgeted: the raw keep*comp passes through and `used` is
// untouched (bit-identical to the pre-budget behavior).
TEST(XIcpBudget, UnbudgetedIsRawAdmit) {
  double used = 0.0;
  EXPECT_DOUBLE_EQ(nano_gicp::xicpBudgetedAdmit(0.4, 0.5, 0.0, &used), 0.2);
  EXPECT_DOUBLE_EQ(nano_gicp::xicpBudgetedAdmit(-0.4, 0.5, -1.0, &used), -0.2);
  EXPECT_DOUBLE_EQ(used, 0.0);
}

// Admission under the cap passes through and accrues into `used`.
TEST(XIcpBudget, UnderCapPassesAndAccrues) {
  double used = 0.0;
  EXPECT_DOUBLE_EQ(nano_gicp::xicpBudgetedAdmit(0.10, 0.5, 0.15, &used), 0.05);
  EXPECT_DOUBLE_EQ(used, 0.05);
}

// A large marginal pull is clamped to the remaining budget -- the fail-open
// path of the 2026-07-08 runaway (uncapped keep*comp) is closed.
TEST(XIcpBudget, ClampsToRemainingBudget) {
  double used = 0.0;
  // wants 0.5*2.0 = 1.0 m; only 0.15 available.
  EXPECT_DOUBLE_EQ(nano_gicp::xicpBudgetedAdmit(2.0, 0.5, 0.15, &used), 0.15);
  EXPECT_DOUBLE_EQ(used, 0.15);
  // budget exhausted: nothing more admitted this scan.
  EXPECT_DOUBLE_EQ(nano_gicp::xicpBudgetedAdmit(2.0, 0.5, 0.15, &used), 0.0);
  EXPECT_DOUBLE_EQ(used, 0.15);
}

// Accumulation across calls (multiple marginal axes / LM iterations): the cap
// bounds the CUMULATIVE admitted magnitude, sign-independent.
TEST(XIcpBudget, AccumulatesAcrossCallsSignIndependent) {
  double used = 0.0;
  EXPECT_DOUBLE_EQ(nano_gicp::xicpBudgetedAdmit( 0.16, 0.5, 0.15, &used),  0.08);
  EXPECT_DOUBLE_EQ(nano_gicp::xicpBudgetedAdmit(-0.16, 0.5, 0.15, &used), -0.07);  // clamped to remaining
  EXPECT_DOUBLE_EQ(used, 0.15);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
