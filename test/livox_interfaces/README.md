# Livox adapter test fixture

These wire definitions match the public
[Livox CustomMsg](https://github.com/Livox-SDK/livox_ros_driver2/blob/master/msg/CustomMsg.msg)
and CustomPoint schema. CI builds this interface-only fixture in a separate
test underlay so every ROS distribution compiles and exercises the optional
raw-message adapter. It contains no driver, SDK, or runtime sensor node.

Normal colcon discovery stops at the parent dliio package and does not build
this fixture. Do not install it alongside the real `livox_ros_driver2`; use
the real driver's message package for deployment. This fixture does not test
hardware, the Livox SDK, or driver support for a particular ROS distribution.
