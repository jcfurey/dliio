#!/usr/bin/env python3
"""Wait for retained observations, graph work and the map cache after bag EOF."""
import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray
from dliio_mapping.drain import drained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loops', action='store_true')
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--report', type=Path)
    args, ros_args = parser.parse_known_args()
    if not 1 <= args.timeout <= 7200 or (args.report and args.report.exists()):
        parser.error('Use a timeout in [1, 7200] seconds and a new report path')
    rclpy.init(args=ros_args)
    node = rclpy.create_node('wait_for_mapping')
    state = {}
    def receive(message):
        for status in message.status:
            if status.name in ('DLIO Mapping', 'DLIO Loop Closure'):
                state[status.name] = {item.key: item.value for item in status.values}
    subscriptions = [node.create_subscription(DiagnosticArray, topic, receive, qos_profile_sensor_data)
                     for topic in ('dlio/mapping/diagnostics', 'diagnostics')]
    start, settled, result = time.monotonic(), None, False
    try:
        while rclpy.ok() and time.monotonic()-start < args.timeout:
            rclpy.spin_once(node, timeout_sec=.2)
            mapping = state.get('DLIO Mapping', {})
            loops = state.get('DLIO Loop Closure', {}) if args.loops else None
            if loops and loops.get('loop_error'):
                break
            if drained(mapping, loops):
                settled = time.monotonic() if settled is None else settled
                if time.monotonic()-settled >= 3.:
                    result = True
                    break
            else:
                settled = None
        report = dict(drained=result, elapsed_seconds=time.monotonic()-start, diagnostics=state)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()
    return 0 if result else 1


if __name__ == '__main__':
    raise SystemExit(main())
