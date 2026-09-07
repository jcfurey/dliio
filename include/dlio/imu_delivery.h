#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>

namespace dlio {

// Counts the estimator's own callback stages. Acquisition gaps and steady-clock
// receipt gaps are distinct; a replay pause must not look like missing IMU data.
class ImuDeliveryAudit {
 public:
  struct Snapshot {
    uint64_t received = 0, waiting_extrinsics = 0, rejected_time = 0;
    uint64_t accepted = 0, buffered = 0, propagated = 0;
    int64_t first_accepted_ns = -1, last_accepted_ns = -1, last_buffered_ns = -1;
    int64_t last_receipt_ns = -1, max_receipt_gap_ns = 0;
    int64_t max_acquisition_gap_ns = 0, max_gap_stamp_ns = -1;
  };
  void receipt(int64_t steady_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++data_.received;
    if (data_.last_receipt_ns >= 0)
      data_.max_receipt_gap_ns = std::max(data_.max_receipt_gap_ns, steady_ns-data_.last_receipt_ns);
    data_.last_receipt_ns = steady_ns;
  }
  void waitingExtrinsics() { std::lock_guard<std::mutex> lock(mutex_); ++data_.waiting_extrinsics; }
  void rejectedTime() { std::lock_guard<std::mutex> lock(mutex_); ++data_.rejected_time; }
  void accepted(int64_t stamp_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++data_.accepted;
    if (data_.first_accepted_ns < 0) data_.first_accepted_ns = stamp_ns;
    if (data_.last_accepted_ns >= 0 && stamp_ns-data_.last_accepted_ns > data_.max_acquisition_gap_ns) {
      data_.max_acquisition_gap_ns = stamp_ns-data_.last_accepted_ns;
      data_.max_gap_stamp_ns = stamp_ns;
    }
    data_.last_accepted_ns = stamp_ns;
  }
  void buffered(int64_t stamp_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++data_.buffered;
    data_.last_buffered_ns = stamp_ns;
  }
  void propagated() { std::lock_guard<std::mutex> lock(mutex_); ++data_.propagated; }
  Snapshot snapshot() const { std::lock_guard<std::mutex> lock(mutex_); return data_; }
 private:
  mutable std::mutex mutex_;
  Snapshot data_;
};

}  // namespace dlio
