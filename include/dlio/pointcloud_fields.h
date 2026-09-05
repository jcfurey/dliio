#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace dlio {

// A checked scalar view of a PointCloud2 field. Unlike a typed ROS iterator,
// this respects row padding, unaligned offsets, and the message byte order.
// Names are tried in order so the original Ouster names take precedence over
// native-profile aliases (intensity/signal and ambient/near_ir).
class PointCloudScalarField {
public:
  PointCloudScalarField(const sensor_msgs::msg::PointCloud2& cloud,
                       std::initializer_list<const char*> names) : cloud_(cloud) {
    if (!validLayout(cloud)) { return; }
    for (const char* name : names) {
      for (const auto& field : cloud.fields) {
        const size_t bytes = scalarSize(field.datatype);
        if (field.name == name && field.count == 1 && bytes > 0 &&
            static_cast<uint64_t>(field.offset) + bytes <= cloud.point_step) {
          field_ = &field;
          return;
        }
      }
    }
  }

  explicit operator bool() const { return field_ != nullptr; }
  const std::string& name() const { return field_->name; }

  static bool validLayout(const sensor_msgs::msg::PointCloud2& cloud) {
    if (cloud.point_step == 0 ||
        static_cast<uint64_t>(cloud.width) * cloud.point_step > cloud.row_step ||
        static_cast<uint64_t>(cloud.height) * cloud.row_step > cloud.data.size()) {
      return false;
    }
    // PCL also reads the advertised fields. Validate their extents before the
    // caller invokes fromROSMsg, not just before our own scalar conversions.
    for (const auto& field : cloud.fields) {
      const size_t bytes = scalarSize(field.datatype);
      if (bytes == 0 || field.count == 0 || static_cast<uint64_t>(field.offset) +
          static_cast<uint64_t>(field.count) * bytes > cloud.point_step) {
        return false;
      }
    }
    return true;
  }

  float operator[](size_t i) const {
    if (!field_ || i >= static_cast<uint64_t>(cloud_.width) * cloud_.height) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const size_t offset = (i / cloud_.width) * cloud_.row_step +
        (i % cloud_.width) * cloud_.point_step + field_->offset;
    const uint8_t* data = cloud_.data.data() + offset;
    using PF = sensor_msgs::msg::PointField;
    switch (field_->datatype) {
      case PF::INT8: return read<int8_t>(data);
      case PF::UINT8: return read<uint8_t>(data);
      case PF::INT16: return read<int16_t>(data);
      case PF::UINT16: return read<uint16_t>(data);
      case PF::INT32: return read<int32_t>(data);
      case PF::UINT32: return read<uint32_t>(data);
      case PF::FLOAT32: return read<float>(data);
      case PF::FLOAT64: return read<double>(data);
      default: return std::numeric_limits<float>::quiet_NaN();
    }
  }

private:
  static size_t scalarSize(uint8_t datatype) {
    using PF = sensor_msgs::msg::PointField;
    switch (datatype) {
      case PF::INT8: case PF::UINT8: return 1;
      case PF::INT16: case PF::UINT16: return 2;
      case PF::INT32: case PF::UINT32: case PF::FLOAT32: return 4;
      case PF::FLOAT64: return 8;
      default: return 0;
    }
  }

  template<typename T>
  float read(const uint8_t* data) const {
    T value;
    std::memcpy(&value, data, sizeof(T));
    const uint16_t endian_probe = 1;
    const bool host_bigendian = *reinterpret_cast<const uint8_t*>(&endian_probe) == 0;
    if (sizeof(T) > 1 && cloud_.is_bigendian != host_bigendian) {
      auto* bytes = reinterpret_cast<uint8_t*>(&value);
      std::reverse(bytes, bytes + sizeof(T));
    }
    return static_cast<float>(value);
  }

  const sensor_msgs::msg::PointCloud2& cloud_;
  const sensor_msgs::msg::PointField* field_ = nullptr;
};

}  // namespace dlio
