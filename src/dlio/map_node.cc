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

#include "dlio/map.h"

#include <cstdlib>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

int main(int argc, char** argv) {

  // Limit glibc to a single malloc arena; with many short-lived threads the
  // per-thread arenas fragment and inflate resident memory over long runs.
#if defined(__GLIBC__)
  mallopt(M_ARENA_MAX, 1);
#endif

  rclcpp::init(argc, argv);
  auto node = std::make_shared<dlio::MapNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();

  return 0;

}
