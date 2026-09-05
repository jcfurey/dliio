// Isolate map snapshot preparation from DDS transmission and keyframe ingestion.
// The legacy branch reproduces the former publishMap copy + toROSMsg hot path.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "dlio/map.h"

namespace dlio {
struct MapNodeTestAccess {
  static void seed(MapNode& node, size_t count) {
    node.dlio_map->resize(count);
    for (size_t i = 0; i < count; ++i) {
      auto& p = (*node.dlio_map)[i];
      p.x = static_cast<float>(i % 1000) * .1f;
      p.y = static_cast<float>((i / 1000) % 1000) * .1f;
      p.z = 1.f;
      p.intensity = 1000.f; p.reflectivity = 42.f;
      p.intensity_corrected = 1500.f; p.lidar_intensity = 900.f; p.timestamp = 0.;
    }
    node.last_keyframe_stamp_ = rclcpp::Time(10, 0, RCL_ROS_TIME);
    node.have_keyframe_stamp_ = true; ++node.map_revision_;
  }
  static sensor_msgs::msg::PointCloud2::ConstSharedPtr cached(MapNode& node) { return node.mapMessage(); }
  static size_t legacy(MapNode& node) {
    auto snapshot = std::make_shared<pcl::PointCloud<PointType>>(*node.dlio_map);
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(*snapshot, message);
    return message.data.size();
  }
};
}

int main(int argc, char** argv) {
  const size_t count = argc > 1 ? std::stoull(argv[1]) : 1000000;
  constexpr int repeats = 30;
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options; options.append_parameter_override("map/publishRate", 0.);
  {
    dlio::MapNode node(options);
    dlio::MapNodeTestAccess::seed(node, count);
    std::vector<double> legacy, cached;
    size_t bytes = 0;
    auto measure = [&](auto fn) {
      const auto start = std::chrono::steady_clock::now();
      bytes += fn();
      return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    };
    const double changed_us = measure([&] { return dlio::MapNodeTestAccess::cached(node)->data.size(); });
    for (int i = 0; i < repeats; ++i) {
      legacy.push_back(measure([&] { return dlio::MapNodeTestAccess::legacy(node); }));
      cached.push_back(measure([&] { return dlio::MapNodeTestAccess::cached(node)->data.size(); }));
    }
    std::sort(legacy.begin(), legacy.end()); std::sort(cached.begin(), cached.end());
    std::cout << "{\"points\":" << count << ",\"repeats\":" << repeats
              << ",\"changed_snapshot_us\":" << changed_us
              << ",\"legacy_p50_us\":" << legacy[repeats / 2]
              << ",\"cached_p50_us\":" << cached[repeats / 2]
              << ",\"legacy_p95_us\":" << legacy[repeats * 95 / 100]
              << ",\"cached_p95_us\":" << cached[repeats * 95 / 100]
              << ",\"bytes_consumed\":" << bytes << "}\n";
  }
  rclcpp::shutdown();
}
