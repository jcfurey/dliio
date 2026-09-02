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

// Component registration: lets both nodes run inside a (multithreaded)
// component container with intra-process communication, so the keyframe
// clouds between the odometry and map nodes skip RMW serialization.

#include <rclcpp_components/register_node_macro.hpp>

#include "dlio/odom.h"
#include "dlio/map.h"

RCLCPP_COMPONENTS_REGISTER_NODE(dlio::OdomNode)
RCLCPP_COMPONENTS_REGISTER_NODE(dlio::MapNode)
