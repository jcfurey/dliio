#pragma once

#include <cmath>

namespace nano_gicp {

// The calibrated column model uses unwrapped azimuth, while atan2 returns
// [-pi, pi]. Choose the equivalent angle nearest the image center. This is
// needed for clockwise Ouster scans spanning approximately [0, -2*pi], and
// preserves the local derivative 1 / azimuth_step away from the image seam.
inline float columnFromAzimuth(float azimuth, float azimuth_step,
                               float azimuth_origin, int width) {
  constexpr float two_pi = 6.2831853071795864769f;
  const float center = azimuth_origin + azimuth_step * (width - 1) * 0.5f;
  const float turns = std::round((center - azimuth) / two_pi);
  return (azimuth + turns * two_pi - azimuth_origin) / azimuth_step;
}

}  // namespace nano_gicp
