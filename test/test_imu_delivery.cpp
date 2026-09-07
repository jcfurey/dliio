#include <gtest/gtest.h>
#include "dlio/imu_delivery.h"

TEST(ImuDelivery, ReplayPauseDoesNotBecomeAcquisitionLoss) {
  dlio::ImuDeliveryAudit audit;
  audit.receipt(1000000); audit.accepted(1000000000); audit.buffered(1000000000);
  audit.receipt(5001000000); audit.accepted(1002500000); audit.buffered(1002500000); audit.propagated();
  const auto s = audit.snapshot();
  EXPECT_EQ(s.max_receipt_gap_ns, 5000000000);
  EXPECT_EQ(s.max_acquisition_gap_ns, 2500000);
  EXPECT_EQ(s.received, 2u); EXPECT_EQ(s.buffered, 2u); EXPECT_EQ(s.propagated, 1u);
}

TEST(ImuDelivery, StartupAndRejectedSamplesDoNotMoveAcceptedTime) {
  dlio::ImuDeliveryAudit audit;
  audit.receipt(0); audit.waitingExtrinsics();
  audit.receipt(1); audit.accepted(1000000000);
  audit.receipt(2); audit.rejectedTime();
  audit.receipt(3); audit.accepted(1010000000); audit.buffered(1010000000);
  const auto s = audit.snapshot();
  EXPECT_EQ(s.received, s.waiting_extrinsics+s.rejected_time+s.accepted);
  EXPECT_EQ(s.first_accepted_ns, 1000000000);
  EXPECT_EQ(s.last_accepted_ns, 1010000000);
  EXPECT_EQ(s.max_acquisition_gap_ns, 10000000);
  EXPECT_EQ(s.max_gap_stamp_ns, 1010000000);
}
