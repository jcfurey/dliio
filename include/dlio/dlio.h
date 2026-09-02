#pragma once

/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

// SYSTEM
#include <atomic>

#ifdef HAS_CPUID
#include <cpuid.h>
#endif

#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/times.h>
#include <thread>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// BOOST
#include <boost/format.hpp>

// PCL
// (also passed on the command line by CMake; guard to avoid redefinition warnings)
#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

// DLIO
#include <nano_gicp/nano_gicp.h>

namespace dlio {
  enum class SensorType { OUSTER, VELODYNE, HESAI, LIVOX, UNKNOWN };

  class OdomNode;
  class MapNode;

  struct Point {
    Point(): data{0.f, 0.f, 0.f, 1.f}, intensity(0.f), reflectivity(0.f) {}

    PCL_ADD_POINT4D;
    float intensity;    // return signal strength (range-dependent)
    float reflectivity; // calibrated reflectivity (range-normalized, e.g. Ouster)
    union {
    std::uint32_t t;   // (Ouster) time since beginning of scan in nanoseconds
    float time;        // (Velodyne) time since beginning of scan in seconds
    double timestamp;  // (Hesai)  absolute timestamp in seconds      (< 1e14)
                       // (Livox)  absolute timestamp in nanoseconds  (> 1e14, = seconds * 1e9)
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT(dlio::Point,
                                 (float, x, x)
                                 (float, y, y)
                                 (float, z, z)
                                 (float, intensity, intensity)
                                 (float, reflectivity, reflectivity)
                                 (std::uint32_t, t, t)
                                 (float, time, time)
                                 (double, timestamp, timestamp))

// PCL field registration above relies on this exact layout (overlapping
// union members registered by offset); fail loudly if the struct changes.
static_assert(sizeof(dlio::Point) == 32, "dlio::Point layout changed; update PCL field registration");

typedef dlio::Point PointType;
