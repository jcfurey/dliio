"""Real DDS: automatic proposals commit and publish one consistent map revision."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import numpy as np
import pytest
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from direct_lidar_inertial_odometry.msg import MappingObservation, MappingSnapshot
from rclpy.qos import DurabilityPolicy, QoSProfile

from dliio_mapping.input import pose_message
from dliio_mapping.node import cloud_message
from dliio_mapping.core import Store, Limits, snapshot_database
from graph_fixtures import placed, pose, room
from test_loop_retrieval import config


def test_automatic_loop_commit_and_snapshot_after_sensor_stream_stops(tmp_path, monkeypatch):
    monkeypatch.setenv('ROS_LOG_DIR',str(tmp_path/'ros-log'))
    namespace = '/loop_check_'+uuid.uuid4().hex[:8]
    configuration = tmp_path/'loops.json'
    configuration.write_text(json.dumps(config()))
    environment = dict(os.environ, ROS_LOG_DIR=str(tmp_path/'ros-log'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    processes, logs, state, snapshots = [], [], {}, []
    rclpy.init()
    node = rclpy.create_node('loop_pipeline_check', namespace=namespace)
    def diagnose(message):
        for item in message.status:
            state[item.name] = {v.key: v.value for v in item.values}
    subscriptions = [node.create_subscription(DiagnosticArray, topic, diagnose, 10)
                     for topic in ('dlio/mapping/diagnostics', 'diagnostics')]
    subscriptions.append(node.create_subscription(MappingSnapshot, 'dlio/mapping/snapshot',
        lambda value: snapshots.__setitem__(slice(None), [value]),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)))
    publisher = node.create_publisher(MappingObservation, 'observation', 16)
    def until(condition, timeout=90):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            assert all(process.poll() is None for process in processes), [p.returncode for p in processes]
            if condition(): return
        pytest.fail('Automatic loop pipeline timed out: '+json.dumps(state))
    try:
        for executable, parameters in [
            ('dlio_mapping_node.py', {'mapping/transport':'observation', 'mapping/storage_directory':str(tmp_path/'map'),
                'mapping/fusion_size':'0.0', 'mapping/submap_keyframes':'4', 'mapping/resident_submaps':'1',
                'mapping/publish_rate':'5.0', 'mapping/worker_queue':'128', 'mapping/publish_tf':'false'}),
            ('dlio_loop_closure_node.py', {'use_sim_time':'true',
                'loop_closure/configuration_file':str(configuration), 'loop_closure/output_directory':str(tmp_path/'loop-log')})]:
            command = ['ros2','run','direct_lidar_inertial_odometry',executable,'--ros-args','-r','__ns:='+namespace]
            for key, value in parameters.items(): command += ['-p',key+':='+value]
            log = (tmp_path/(executable+'.log')).open('w')
            logs.append(log)
            processes.append(subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        until(lambda: publisher.get_subscription_count() == 1)
        source = str(uuid.uuid4())
        for i in range(16):
            truth, original = pose(.1*min(i,15-i)), pose(.1*min(i,15-i)+i*.04)
            points = placed(placed(room(),np.linalg.inv(truth)),original)
            cloud = cloud_message(points,(i+1)*1000000000,'odom')
            publisher.publish(MappingObservation(header=cloud.header,cloud=cloud,
                source_session_id=source,observation_id=i,base_frame_id='base_link',
                registered_pose=pose_message(original),covariance_kind=MappingObservation.UNKNOWN,
                covariance_model='Synthetic unavailable registered covariance',registration_converged=True))
            until(lambda: state.get('DLIO Mapping',{}).get('keyframes') == str(i+1))
        # No /clock is ever published. Loop status must remain live at EOF.
        from dliio_mapping.drain import drained
        until(lambda: drained(state.get('DLIO Mapping',{}),state.get('DLIO Loop Closure',{})) and
              int(state.get('DLIO Mapping',{}).get('active_loops',0)) > 0 and snapshots and snapshots[0].pose_revision >= 2)
        latest = snapshots[0]
        assert latest.cloud.header == latest.header == latest.optimized_trajectory.header
        assert len(latest.original_trajectory.poses) == len(latest.optimized_trajectory.poses) == 16
        assert latest.original_trajectory.poses[-1].pose.position.x == pytest.approx(.6)
        assert latest.optimized_trajectory.poses[-1].pose.position.x < .35
        assert state['DLIO Mapping'].get('queue_dropped','0') == '0'
        sealed = tmp_path/'saved.dliomap'
        snapshot_database(state['DLIO Mapping']['archive'],sealed)
        store = Store(sealed,Limits(resident_submaps=1))
        try:
            assert store.graph_stats()['graph_attached']
            assert store.stats()['active_loops'] > 0
            assert all(json.loads(row[0])['covariance_kind'] == 'unknown' for row in store.db.execute('SELECT json FROM observations'))
        finally: store.close()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    process.wait(timeout=10)
        for log in logs: log.close()
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()
