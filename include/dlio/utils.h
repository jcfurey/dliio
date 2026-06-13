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

#include "rclcpp/rclcpp.hpp"

namespace dlio {

    template <typename T>
    struct identity { typedef T type; };

    template <typename T>
    void declare_param(rclcpp::Node* node, const std::string param_name, T& param, const typename identity<T>::type& default_value) {
        node->declare_parameter(param_name, default_value);
        node->get_parameter(param_name, param);
    }

    // Overload with a descriptor description, surfaced by `ros2 param describe`.
    template <typename T>
    void declare_param(rclcpp::Node* node, const std::string param_name, T& param,
                       const typename identity<T>::type& default_value,
                       const std::string& description) {
        rcl_interfaces::msg::ParameterDescriptor desc;
        desc.description = description;
        node->declare_parameter(param_name, default_value, desc);
        node->get_parameter(param_name, param);
    }

    // Double param with a description AND a FloatingPointRange, so `ros2 param
    // describe` reports bounds, rqt shows a slider, and out-of-range `set`s are
    // rejected by rclcpp before the on-set callback runs.
    inline void declare_param(rclcpp::Node* node, const std::string param_name, double& param,
                              double default_value, const std::string& description,
                              double min_value, double max_value, double step = 0.0) {
        rcl_interfaces::msg::ParameterDescriptor desc;
        desc.description = description;
        rcl_interfaces::msg::FloatingPointRange range;
        range.from_value = min_value;
        range.to_value = max_value;
        range.step = step;
        desc.floating_point_range.push_back(range);
        node->declare_parameter(param_name, default_value, desc);
        node->get_parameter(param_name, param);
    }

}
